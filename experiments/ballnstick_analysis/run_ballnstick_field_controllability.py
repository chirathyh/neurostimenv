"""Map the realistic-dose open-loop control subspace of BallAndStick.

The discovery stage evaluates a predeclared signed-DC/AC/montage library on
matched A/B circuit seeds. It reports both actual action reachability and the
SVD span of the mean feature responses. Optional disjoint validation seeds run
only the frozen top discovery actions. This is system identification, not RL.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as so
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from env.models.neuron.stimulation import (  # noqa: E402
    apply_raised_cosine_block_envelope,
)
from experiments.ballnstick_analysis.run_ballnstick import (  # noqa: E402
    _extract_eeg_features,
    _preprocess_eeg,
)


def _plain_copy(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _mpi_variables(comm, size: int, rank: int, seed: int) -> dict[str, Any]:
    return {
        "COMM": comm,
        "SIZE": size,
        "RANK": rank,
        "GLOBALSEED": int(seed),
        "SEED": int(seed) * 10_000,
    }


def _action_id(action: dict[str, Any]) -> str:
    return str(action["id"])


def _environment_action(action: dict[str, Any]) -> dict[str, Any]:
    ignored = {"id", "include_in_subspace"}
    return {key: value for key, value in action.items() if key not in ignored}


def _episode_config(
    base_cfg: DictConfig,
    *,
    inhibition_scale: float,
    seed: int,
    output_dir: Path,
) -> DictConfig:
    run_cfg = _plain_copy(base_cfg)
    OmegaConf.set_struct(run_cfg, False)
    burn = int(run_cfg.analysis.timeline.burn_in_steps)
    stimulation = int(run_cfg.analysis.timeline.stimulation_steps)
    if burn < 1 or stimulation < 2:
        raise ValueError(
            "The screen requires at least one burn-in and two stimulation windows."
        )
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    with open_dict(run_cfg):
        run_cfg.experiment.seed = int(seed)
        run_cfg.experiment.dir = str(output_dir)
        run_cfg.env.simulation.duration = (burn + stimulation) * window_ms
        run_cfg.env.network.inhibition_scale = float(inhibition_scale)
        run_cfg.env.ts.apply = True
        run_cfg.env.online.temperature_mode = "configured"
        run_cfg.env.online.stimulation.parameterization = "uniform_field"
    return run_cfg


def _bandpower(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (frequencies_hz >= low_hz) & (frequencies_hz < high_hz)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapz(psd[mask], frequencies_hz[mask]))


def _control_features(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    cfg: DictConfig,
) -> dict[str, float]:
    masked = np.asarray(psd, dtype=np.float64).copy()
    half_width = float(cfg.analysis.stimulus_exclusion_half_width_hz)
    for frequency_hz in cfg.analysis.excluded_frequencies_hz:
        masked[
            np.abs(frequencies_hz - float(frequency_hz)) <= half_width
        ] = 0.0
    total = _bandpower(frequencies_hz, masked, 1.0, 80.000001)
    gamma = _bandpower(frequencies_hz, masked, 30.0, 80.000001)
    eps = np.finfo(np.float64).tiny
    return {
        "log10_total_power_1_80_excluding_actions": float(
            np.log10(max(total, eps))
        ),
        "relative_gamma_power_excluding_actions": (
            gamma / total if total > 0.0 else float("nan")
        ),
    }


def _analyze_eeg_details(raw_eeg: np.ndarray, cfg: DictConfig):
    processed, analysis_fs_hz = _preprocess_eeg(
        raw_eeg,
        fs_hz=1000.0 / float(cfg.env.network.dt),
        target_fs_hz=int(cfg.analysis.target_fs_hz),
        low_hz=float(cfg.analysis.low_hz),
        high_hz=float(cfg.analysis.high_hz),
    )
    features, frequencies_hz, psd = _extract_eeg_features(
        processed, analysis_fs_hz
    )
    features.update(_control_features(frequencies_hz, psd, cfg))
    return features, processed, frequencies_hz, psd


def _analyze_eeg(raw_eeg: np.ndarray, cfg: DictConfig) -> dict[str, float]:
    return _analyze_eeg_details(raw_eeg, cfg)[0]


def _run_episode(
    base_cfg: DictConfig,
    *,
    cohort: str,
    condition: str,
    inhibition_scale: float,
    seed: int,
    action: dict[str, Any],
    output_dir: Path,
    comm,
    size: int,
    rank: int,
) -> dict[str, Any] | None:
    run_cfg = _episode_config(
        base_cfg,
        inhibition_scale=inhibition_scale,
        seed=seed,
        output_dir=output_dir,
    )
    np.random.seed(int(seed) * 10_000 + rank)
    random.seed(int(seed) * 10_000 + rank)
    environment = OnlineNeuronEnv(
        run_cfg,
        _mpi_variables(comm, size, rank, seed),
        ENV_SEED=0,
    )
    burn_steps = int(run_cfg.analysis.timeline.burn_in_steps)
    stimulation_steps = int(run_cfg.analysis.timeline.stimulation_steps)
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    block_start_ms = burn_steps * window_ms
    block_stop_ms = (burn_steps + stimulation_steps) * window_ms
    zero_action = {
        "montage": "axial",
        "dc_offset_v_per_m": 0.0,
        "ac_amplitude_v_per_m": 0.0,
        "frequency_hz": 0.0,
    }
    active_action = _environment_action(action)
    outputs: list[dict[str, Any]] | None = [] if rank == 0 else None
    try:
        for _ in range(burn_steps):
            output = environment.step_online(zero_action)
            if rank == 0:
                outputs.append(output)
        for step in range(stimulation_steps):
            # Phase is an action at stimulation onset. Subsequent windows use
            # the environment's accumulated phase to avoid waveform restarts.
            step_action = dict(active_action)
            if step > 0:
                step_action.pop("phase_rad", None)
            output = environment.step_online(
                step_action,
                phase_continuous=True,
                block_envelope={
                    "start_ms": block_start_ms,
                    "stop_ms": block_stop_ms,
                    "ramp_ms": float(run_cfg.analysis.timeline.block_ramp_ms),
                },
            )
            if rank == 0:
                outputs.append(output)
    finally:
        environment.close()

    if rank != 0:
        return None
    retained = outputs[burn_steps:]
    raw_eeg = np.concatenate(
        [np.asarray(output["eeg_v"]).reshape(-1) for output in retained]
    ).astype(np.float64, copy=False)
    features = _analyze_eeg(raw_eeg, run_cfg)
    duration_s = stimulation_steps * window_ms / 1000.0
    rates: dict[str, float] = {}
    for population_name in ("E", "I"):
        spike_count = sum(
            int(output["spikes"][population_name]["times_ms"].size)
            for output in retained
        )
        population_size = len(
            retained[0]["spikes"][population_name]["per_cell"]
        )
        rates[f"{population_name}_firing_rate_hz"] = (
            spike_count / (population_size * duration_s)
        )
    rates["E_I_firing_rate_ratio"] = (
        rates["E_firing_rate_hz"] / rates["I_firing_rate_hz"]
        if rates["I_firing_rate_hz"] > 0.0
        else float("nan")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(run_cfg.analysis.save_raw_eeg):
        np.save(output_dir / "eeg_raw_v.npy", raw_eeg)
    return {
        "cohort": cohort,
        "condition": condition,
        "seed": int(seed),
        "action_id": _action_id(action),
        "montage": str(action.get("montage", "axial")),
        "dc_offset_v_per_m": float(action.get("dc_offset_v_per_m", 0.0)),
        "ac_amplitude_v_per_m": float(
            action.get("ac_amplitude_v_per_m", 0.0)
        ),
        "frequency_hz": float(action.get("frequency_hz", 0.0)),
        "phase_rad": float(action.get("phase_rad", 0.0)),
        "include_in_subspace": bool(action.get("include_in_subspace", False)),
        "_raw_eeg": raw_eeg,
        **features,
        **rates,
    }


def _make_standardizer(
    a_rows: dict[int, dict[str, Any]],
    b_rows: dict[int, dict[str, Any]],
    seeds: list[int],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack(
        [
            [rows[seed][name] for name in feature_names]
            for rows in (a_rows, b_rows)
            for seed in seeds
        ]
    ).astype(np.float64)
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    positive = scale[scale > np.finfo(float).eps]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return center, np.where(scale > np.finfo(float).eps, scale, fallback)


def _vector(
    row: dict[str, Any],
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray([row[name] for name in feature_names], dtype=np.float64)
        - center
    ) / scale


def _rate_safe(
    candidate: dict[str, Any],
    target: dict[str, Any],
    sham: dict[str, Any],
    cfg: DictConfig,
) -> bool:
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    limits = cfg.analysis.rate_guardrails_hz
    for population in ("E", "I"):
        name = f"{population}_firing_rate_hz"
        value = float(candidate[name])
        reference_low = min(float(target[name]), float(sham[name]))
        reference_high = max(float(target[name]), float(sham[name]))
        if not (
            float(limits[f"{population}_min"])
            <= value
            <= float(limits[f"{population}_max"])
            and max(0.0, reference_low * (1.0 - tolerance))
            <= value
            <= reference_high * (1.0 + tolerance)
        ):
            return False
    return True


def _decorate(
    candidate: dict[str, Any],
    *,
    target: dict[str, Any],
    sham: dict[str, Any],
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    cfg: DictConfig,
) -> dict[str, Any]:
    target_vector = _vector(target, feature_names, center, scale)
    sham_vector = _vector(sham, feature_names, center, scale)
    candidate_vector = _vector(candidate, feature_names, center, scale)
    target_shift = target_vector - sham_vector
    response = candidate_vector - sham_vector
    target_norm = float(np.linalg.norm(target_shift))
    response_norm = float(np.linalg.norm(response))
    sham_distance = target_norm
    candidate_distance = float(np.linalg.norm(candidate_vector - target_vector))
    candidate["sham_distance_to_A"] = sham_distance
    candidate["candidate_distance_to_A"] = candidate_distance
    candidate["fractional_distance_improvement"] = (
        1.0 - candidate_distance / sham_distance if sham_distance > 0 else 0.0
    )
    candidate["target_shift_alignment"] = (
        float(np.dot(target_shift, response) / (target_norm * response_norm))
        if target_norm > 0.0 and response_norm > 0.0
        else 0.0
    )
    candidate["response_to_target_norm_ratio"] = (
        response_norm / target_norm if target_norm > 0.0 else 0.0
    )
    candidate["rate_safe"] = _rate_safe(candidate, target, sham, cfg)
    for population in ("E", "I"):
        name = f"{population}_firing_rate_hz"
        candidate[f"{population}_rate_change_vs_B_hz"] = float(
            candidate[name]
        ) - float(sham[name])
    return candidate


def _matched_synthetic_improvement(
    *,
    candidate: dict[str, Any],
    target: dict[str, Any],
    sham: dict[str, Any],
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    cfg: DictConfig,
) -> float:
    """Return improvement from adding only the candidate AC sine to B EEG.

    The sine amplitude is calibrated to match power in the candidate's driven
    band. DC actions have no sinusoidal observation control and return zero.
    """
    frequency_hz = float(candidate["frequency_hz"])
    ac_amplitude = float(candidate["ac_amplitude_v_per_m"])
    if frequency_hz <= 0.0 or ac_amplitude <= 0.0:
        return 0.0

    b_raw = np.asarray(sham["_raw_eeg"], dtype=np.float64)
    candidate_raw = np.asarray(candidate["_raw_eeg"], dtype=np.float64)
    _, _, candidate_frequencies, candidate_psd = _analyze_eeg_details(
        candidate_raw, cfg
    )
    half_width = float(cfg.analysis.stimulus_exclusion_half_width_hz)
    target_power = _bandpower(
        candidate_frequencies,
        candidate_psd,
        max(0.0, frequency_hz - half_width),
        frequency_hz + half_width + np.finfo(float).eps,
    )
    simulator_fs_hz = 1000.0 / float(cfg.env.network.dt)
    left_time_ms = np.arange(b_raw.size, dtype=np.float64) * (
        1000.0 / simulator_fs_hz
    )
    unit_sine = np.sin(
        2.0
        * np.pi
        * frequency_hz
        * (left_time_ms / 1000.0)
        + float(candidate["phase_rad"])
    )
    duration_ms = b_raw.size * 1000.0 / simulator_fs_hz
    unit_sine = apply_raised_cosine_block_envelope(
        unit_sine,
        time_ms=left_time_ms,
        block_start_ms=0.0,
        block_stop_ms=duration_ms,
        ramp_ms=float(cfg.analysis.timeline.block_ramp_ms),
    )

    def driven_power(amplitude_v: float) -> float:
        _, _, frequencies_hz, psd = _analyze_eeg_details(
            b_raw + float(amplitude_v) * unit_sine, cfg
        )
        return _bandpower(
            frequencies_hz,
            psd,
            max(0.0, frequency_hz - half_width),
            frequency_hz + half_width + np.finfo(float).eps,
        )

    baseline_power = driven_power(0.0)
    if target_power <= baseline_power:
        synthetic_raw = b_raw.copy()
    else:
        high = max(float(np.std(b_raw)) * 0.01, 1e-15)
        for _ in range(50):
            if driven_power(high) >= target_power:
                break
            high *= 2.0
        else:
            raise RuntimeError("Could not bracket matched synthetic EEG amplitude.")
        amplitude = float(
            so.brentq(
                lambda value: driven_power(value) - target_power,
                0.0,
                high,
                xtol=1e-18,
                rtol=1e-12,
            )
        )
        synthetic_raw = b_raw + amplitude * unit_sine

    synthetic = dict(sham)
    synthetic.update(_analyze_eeg(synthetic_raw, cfg))
    target_vector = _vector(target, feature_names, center, scale)
    sham_vector = _vector(sham, feature_names, center, scale)
    synthetic_vector = _vector(synthetic, feature_names, center, scale)
    sham_distance = float(np.linalg.norm(sham_vector - target_vector))
    synthetic_distance = float(np.linalg.norm(synthetic_vector - target_vector))
    return (
        1.0 - synthetic_distance / sham_distance
        if sham_distance > 0.0
        else 0.0
    )


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int):
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize(rows: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for (cohort, action_id), group in rows.groupby(["cohort", "action_id"]):
        improvement = group["fractional_distance_improvement"].to_numpy(float)
        ci_low, ci_high = _bootstrap_ci(
            improvement,
            int(cfg.analysis.n_bootstrap),
            seed=int(cfg.experiment.seed) + len(summaries),
        )
        beyond = group["improvement_beyond_synthetic"].to_numpy(float)
        beyond_ci_low, beyond_ci_high = _bootstrap_ci(
            beyond,
            int(cfg.analysis.n_bootstrap),
            seed=int(cfg.experiment.seed) + 10000 + len(summaries),
        )
        first = group.iloc[0]
        summaries.append(
            {
                "cohort": cohort,
                "action_id": action_id,
                "montage": first["montage"],
                "dc_offset_v_per_m": float(first["dc_offset_v_per_m"]),
                "ac_amplitude_v_per_m": float(first["ac_amplitude_v_per_m"]),
                "frequency_hz": float(first["frequency_hz"]),
                "phase_rad": float(first["phase_rad"]),
                "n_seeds": int(group["seed"].nunique()),
                "mean_fractional_improvement": float(np.mean(improvement)),
                "ci_2.5": ci_low,
                "ci_97.5": ci_high,
                "positive_seed_fraction": float(np.mean(improvement > 0.0)),
                "mean_synthetic_fractional_improvement": float(
                    group["synthetic_fractional_improvement"].mean()
                ),
                "mean_improvement_beyond_synthetic": float(
                    np.mean(beyond)
                ),
                "beyond_synthetic_ci_2.5": beyond_ci_low,
                "beyond_synthetic_ci_97.5": beyond_ci_high,
                "positive_beyond_synthetic_seed_fraction": float(
                    np.mean(group["improvement_beyond_synthetic"] > 0.0)
                ),
                "median_alignment": float(
                    np.median(group["target_shift_alignment"])
                ),
                "mean_response_to_target_norm_ratio": float(
                    group["response_to_target_norm_ratio"].mean()
                ),
                "all_rate_safe": bool(group["rate_safe"].all()),
                "mean_E_rate_change_vs_B_hz": float(
                    group["E_rate_change_vs_B_hz"].mean()
                ),
                "mean_I_rate_change_vs_B_hz": float(
                    group["I_rate_change_vs_B_hz"].mean()
                ),
            }
        )
    return pd.DataFrame(summaries)


def _controllable_subspace(
    rows: list[dict[str, Any]],
    *,
    a_rows: dict[int, dict[str, Any]],
    b_rows: dict[int, dict[str, Any]],
    seeds: list[int],
    actions: list[dict[str, Any]],
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    cfg: DictConfig,
) -> dict[str, Any]:
    action_ids = [
        _action_id(action)
        for action in actions
        if bool(action.get("include_in_subspace", False))
    ]
    target_vectors = np.vstack(
        [
            _vector(a_rows[seed], feature_names, center, scale)
            - _vector(b_rows[seed], feature_names, center, scale)
            for seed in seeds
        ]
    )
    mean_target = np.mean(target_vectors, axis=0)
    response_columns = []
    for action_id in action_ids:
        by_seed = {
            int(row["seed"]): row
            for row in rows
            if row["action_id"] == action_id
        }
        responses = np.vstack(
            [
                _vector(by_seed[seed], feature_names, center, scale)
                - _vector(b_rows[seed], feature_names, center, scale)
                for seed in seeds
            ]
        )
        response_columns.append(np.mean(responses, axis=0))
    response_matrix = np.column_stack(response_columns)
    u, singular_values, _ = np.linalg.svd(response_matrix, full_matrices=False)
    relative_threshold = float(
        cfg.analysis.subspace.relative_singular_value_threshold
    )
    threshold = (
        relative_threshold * float(singular_values[0])
        if singular_values.size and singular_values[0] > 0.0
        else float("inf")
    )
    rank = int(np.count_nonzero(singular_values >= threshold))
    target_norm = float(np.linalg.norm(mean_target))
    if rank > 0 and target_norm > 0.0:
        projection = u[:, :rank] @ (u[:, :rank].T @ mean_target)
        projection_fraction = float(np.linalg.norm(projection) / target_norm)
    else:
        projection_fraction = 0.0
    return {
        "feature_names": feature_names,
        "action_ids": action_ids,
        "response_matrix_columns_are_actions": response_matrix.tolist(),
        "singular_values": singular_values.tolist(),
        "relative_singular_value_threshold": relative_threshold,
        "effective_rank": rank,
        "feature_dimension": len(feature_names),
        "target_vector": mean_target.tolist(),
        "target_projection_fraction": projection_fraction,
        "important_caveat": (
            "Span alignment is necessary but not sufficient: it permits linear "
            "combinations that may not be physically simultaneous actions. "
            "Actual per-action distance improvement remains the reachability gate."
        ),
    }


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5))
    for cohort, marker in (("discovery", "o"), ("validation", "s")):
        subset = summary[summary["cohort"] == cohort]
        if subset.empty:
            continue
        axis.scatter(
            subset["action_id"],
            subset["mean_fractional_improvement"],
            marker=marker,
            label=cohort,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Fractional distance improvement toward A")
    axis.tick_params(axis="x", rotation=75)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "field_action_screen.png", dpi=220)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = (
        Path(to_absolute_path(str(cfg.experiment.dir)))
        / "field_controllability"
    )
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### BallAndStick open-loop field controllability")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    actions = [
        dict(action)
        for action in OmegaConf.to_container(cfg.analysis.actions, resolve=True)
    ]
    action_by_id = {_action_id(action): action for action in actions}
    if "sham" not in action_by_id:
        raise ValueError("The action library must contain action id 'sham'.")
    feature_names = list(cfg.analysis.distance_features)
    base_seed = int(cfg.experiment.seed)
    discovery_seeds = [
        base_seed + int(cfg.analysis.discovery.seed_offset) + index
        for index in range(int(cfg.analysis.discovery.n_seeds))
    ]
    if not discovery_seeds:
        raise ValueError("At least one discovery seed is required.")

    episode_rows: list[dict[str, Any]] = []
    discovery_a: dict[int, dict[str, Any]] = {}
    discovery_b: dict[int, dict[str, Any]] = {}
    for seed in discovery_seeds:
        for condition, scale in (
            ("A", float(cfg.analysis.condition_a_inhibition_scale)),
            ("B", float(cfg.analysis.condition_b_inhibition_scale)),
        ):
            row = _run_episode(
                cfg,
                cohort="discovery",
                condition=condition,
                inhibition_scale=scale,
                seed=seed,
                action=action_by_id["sham"],
                output_dir=root / "discovery" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                (discovery_a if condition == "A" else discovery_b)[seed] = row
                episode_rows.append(row)
        for action in actions:
            if _action_id(action) == "sham":
                continue
            row = _run_episode(
                cfg,
                cohort="discovery",
                condition="B",
                inhibition_scale=float(cfg.analysis.condition_b_inhibition_scale),
                seed=seed,
                action=action,
                output_dir=(
                    root / "discovery" / _action_id(action) / f"seed_{seed}"
                ),
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                episode_rows.append(row)

    if rank == 0:
        center, scale = _make_standardizer(
            discovery_a, discovery_b, discovery_seeds, feature_names
        )
        decorated: list[dict[str, Any]] = []
        for row in episode_rows:
            if row["cohort"] == "discovery" and row["condition"] == "B" and row[
                "action_id"
            ] != "sham":
                decorated.append(
                    _decorate(
                        row,
                        target=discovery_a[int(row["seed"])],
                        sham=discovery_b[int(row["seed"])],
                        feature_names=feature_names,
                        center=center,
                        scale=scale,
                        cfg=cfg,
                    )
                )
                decorated[-1]["synthetic_fractional_improvement"] = (
                    _matched_synthetic_improvement(
                        candidate=decorated[-1],
                        target=discovery_a[int(row["seed"])],
                        sham=discovery_b[int(row["seed"])],
                        feature_names=feature_names,
                        center=center,
                        scale=scale,
                        cfg=cfg,
                    )
                )
                decorated[-1]["improvement_beyond_synthetic"] = (
                    decorated[-1]["fractional_distance_improvement"]
                    - decorated[-1]["synthetic_fractional_improvement"]
                )
        discovery_summary = _summarize(pd.DataFrame(decorated), cfg)
        selectable_ids = {
            _action_id(action)
            for action in actions
            if bool(action.get("include_in_subspace", False))
        }
        selectable = discovery_summary[
            discovery_summary["action_id"].isin(selectable_ids)
        ]
        safe = selectable[selectable["all_rate_safe"]]
        ranked = safe if not safe.empty else selectable
        selected_ids = (
            ranked.sort_values(
                ["mean_improvement_beyond_synthetic", "median_alignment"],
                ascending=[False, False],
            )
            .head(int(cfg.analysis.discovery.top_k))["action_id"]
            .tolist()
        )
        subspace = _controllable_subspace(
            decorated,
            a_rows=discovery_a,
            b_rows=discovery_b,
            seeds=discovery_seeds,
            actions=actions,
            feature_names=feature_names,
            center=center,
            scale=scale,
            cfg=cfg,
        )
    else:
        center = scale = selected_ids = subspace = None
    center = comm.bcast(center, root=0)
    scale = comm.bcast(scale, root=0)
    selected_ids = comm.bcast(selected_ids, root=0)

    validation_rows: list[dict[str, Any]] = []
    validation_seeds = [
        base_seed + int(cfg.analysis.validation.seed_offset) + index
        for index in range(int(cfg.analysis.validation.n_seeds))
    ]
    for seed in validation_seeds:
        references: dict[str, dict[str, Any]] = {}
        for condition, scale_value in (
            ("A", float(cfg.analysis.condition_a_inhibition_scale)),
            ("B", float(cfg.analysis.condition_b_inhibition_scale)),
        ):
            row = _run_episode(
                cfg,
                cohort="validation",
                condition=condition,
                inhibition_scale=scale_value,
                seed=seed,
                action=action_by_id["sham"],
                output_dir=root / "validation" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                references[condition] = row
                episode_rows.append(row)
        for action_id in selected_ids:
            row = _run_episode(
                cfg,
                cohort="validation",
                condition="B",
                inhibition_scale=float(cfg.analysis.condition_b_inhibition_scale),
                seed=seed,
                action=action_by_id[action_id],
                output_dir=root / "validation" / action_id / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                episode_rows.append(row)
                validation_rows.append(
                    _decorate(
                        row,
                        target=references["A"],
                        sham=references["B"],
                        feature_names=feature_names,
                        center=center,
                        scale=scale,
                        cfg=cfg,
                    )
                )
                validation_rows[-1]["synthetic_fractional_improvement"] = (
                    _matched_synthetic_improvement(
                        candidate=validation_rows[-1],
                        target=references["A"],
                        sham=references["B"],
                        feature_names=feature_names,
                        center=center,
                        scale=scale,
                        cfg=cfg,
                    )
                )
                validation_rows[-1]["improvement_beyond_synthetic"] = (
                    validation_rows[-1]["fractional_distance_improvement"]
                    - validation_rows[-1]["synthetic_fractional_improvement"]
                )

    if rank == 0:
        summary_parts = [discovery_summary]
        if validation_rows:
            summary_parts.append(_summarize(pd.DataFrame(validation_rows), cfg))
        summary = pd.concat(summary_parts, ignore_index=True)
        top = discovery_summary[
            discovery_summary["action_id"].isin(
                {
                    _action_id(action)
                    for action in actions
                    if bool(action.get("include_in_subspace", False))
                }
            )
        ].sort_values(
            "mean_improvement_beyond_synthetic", ascending=False
        ).iloc[0]
        gate = cfg.analysis.exploratory_gate
        checks = {
            "minimum_improvement": bool(
                top["mean_fractional_improvement"]
                >= float(gate.minimum_mean_fractional_improvement)
            ),
            "beyond_synthetic_control": bool(
                top["mean_improvement_beyond_synthetic"]
                >= float(gate.minimum_mean_improvement_beyond_synthetic)
            ),
            "seed_consistency": bool(
                top["positive_beyond_synthetic_seed_fraction"]
                >= float(gate.minimum_positive_seed_fraction)
            ),
            "positive_alignment": bool(
                top["median_alignment"] >= float(gate.minimum_alignment)
            ),
            "nontrivial_response_magnitude": bool(
                top["mean_response_to_target_norm_ratio"]
                >= float(gate.minimum_response_to_target_norm_ratio)
            ),
            "rate_safe": bool(top["all_rate_safe"]),
        }
        validation_confirmation = None
        if validation_rows:
            validation_primary = summary[
                (summary["cohort"] == "validation")
                & (summary["action_id"] == selected_ids[0])
            ].iloc[0]
            validation_gate = cfg.analysis.validation_gate
            validation_checks = {
                "improvement_ci_above_zero": bool(
                    validation_primary["ci_2.5"] > 0.0
                ),
                "beyond_synthetic_ci_above_zero": bool(
                    validation_primary["beyond_synthetic_ci_2.5"] > 0.0
                ),
                "seed_consistency": bool(
                    validation_primary["positive_seed_fraction"]
                    >= float(validation_gate.minimum_positive_seed_fraction)
                ),
                "beyond_synthetic_seed_consistency": bool(
                    validation_primary[
                        "positive_beyond_synthetic_seed_fraction"
                    ]
                    >= float(
                        validation_gate.minimum_positive_beyond_synthetic_seed_fraction
                    )
                ),
                "positive_alignment": bool(
                    validation_primary["median_alignment"]
                    > float(validation_gate.minimum_median_alignment)
                ),
                "rate_safe": bool(validation_primary["all_rate_safe"]),
            }
            validation_confirmation = {
                "primary_action_id": selected_ids[0],
                "summary": validation_primary.to_dict(),
                "checks": validation_checks,
                "confirmed": bool(all(validation_checks.values())),
            }
        conclusion = {
            "screen_only": not bool(validation_rows),
            "n_discovery_seeds": len(discovery_seeds),
            "n_validation_seeds": len(validation_seeds),
            "selected_action_ids": selected_ids,
            "top_discovery_action": top.to_dict(),
            "exploratory_checks": checks,
            "realistic_controllability_signal": bool(all(checks.values())),
            "validation_confirmation": validation_confirmation,
            "realistic_controllability_confirmed": (
                None
                if validation_confirmation is None
                else bool(validation_confirmation["confirmed"])
            ),
            "controllable_subspace": subspace,
            "interpretation": (
                "A one/few-seed screen can reject an inert actuator but cannot "
                "confirm reachability. Only disjoint validation with a frozen "
                "protocol supports a positive scientific conclusion."
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        serializable_episode_rows = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in episode_rows
        ]
        serializable_metric_rows = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in decorated + validation_rows
        ]
        pd.DataFrame(serializable_episode_rows).to_csv(
            root / "episode_features.csv", index=False
        )
        pd.DataFrame(serializable_metric_rows).to_csv(
            root / "action_seed_metrics.csv", index=False
        )
        summary.to_csv(root / "action_summary.csv", index=False)
        with (root / "controllable_subspace.json").open("w") as handle:
            json.dump(subspace, handle, indent=2)
        with (root / "screen_conclusion.json").open("w") as handle:
            json.dump(conclusion, handle, indent=2)
        _plot(summary, root)
        print("\n### Exploratory conclusion")
        print(json.dumps(conclusion, indent=2))
        print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
