"""Discover and validate whether stimulation moves Condition B toward A.

Protocol selection and evaluation use disjoint circuit seeds.  Candidate
actions are uniform electric-field amplitude (V/m at tissue) and frequency
(Hz), applied causally one observation window at a time.
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
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from experiments.ballnstick_analysis.run_ballnstick import (  # noqa: E402
    _extract_eeg_features,
    _preprocess_eeg,
)


def _plain_copy(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _episode_config(
    base_cfg: DictConfig,
    *,
    inhibition_scale: float,
    seed: int,
    output_dir: Path,
) -> DictConfig:
    run_cfg = _plain_copy(base_cfg)
    OmegaConf.set_struct(run_cfg, False)
    n_steps = int(run_cfg.analysis.burn_in_steps) + int(
        run_cfg.analysis.stimulation_steps
    )
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    with open_dict(run_cfg):
        run_cfg.experiment.seed = int(seed)
        run_cfg.experiment.dir = str(output_dir)
        run_cfg.env.simulation.duration = n_steps * window_ms
        run_cfg.env.network.inhibition_scale = float(inhibition_scale)
        run_cfg.env.ts.apply = True
        run_cfg.env.online.temperature_mode = "configured"
        run_cfg.env.online.stimulation.parameterization = "uniform_field"
    return run_cfg


def _mpi_variables(comm, size: int, rank: int, seed: int) -> dict[str, Any]:
    return {
        "COMM": comm,
        "SIZE": size,
        "RANK": rank,
        "GLOBALSEED": int(seed),
        "SEED": int(seed) * 10_000,
    }


def _run_episode(
    base_cfg: DictConfig,
    *,
    inhibition_scale: float,
    seed: int,
    action: tuple[float, float],
    output_dir: Path,
    comm,
    size: int,
    rank: int,
) -> dict[str, Any] | None:
    """Run burn-in followed by a fixed candidate action on one fresh circuit."""
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
    burn_in_steps = int(run_cfg.analysis.burn_in_steps)
    stimulation_steps = int(run_cfg.analysis.stimulation_steps)
    actions = [[0.0, 0.0]] * burn_in_steps + [
        [float(action[0]), float(action[1])]
    ] * stimulation_steps

    try:
        outputs = environment.analysis_rollout_online(
            actions,
            phase_continuous=True,
            ramp_ms=float(run_cfg.env.online.ramp_ms),
        )
    finally:
        environment.close()

    if rank != 0:
        return None
    retained = outputs[burn_in_steps:]
    raw_eeg = np.concatenate(
        [np.asarray(output["eeg_v"], dtype=np.float64).reshape(-1) for output in retained]
    )
    simulator_fs_hz = 1000.0 / float(run_cfg.env.network.dt)
    processed_eeg, analysis_fs_hz = _preprocess_eeg(
        raw_eeg,
        fs_hz=simulator_fs_hz,
        target_fs_hz=int(run_cfg.analysis.target_fs_hz),
        low_hz=float(run_cfg.analysis.low_hz),
        high_hz=float(run_cfg.analysis.high_hz),
    )
    feature_values, frequencies_hz, psd = _extract_eeg_features(
        processed_eeg,
        analysis_fs_hz,
    )
    rates = {
        name: float(
            np.mean([output["firing_rates"][name] for output in retained])
        )
        for name in (
            "E_firing_rate_hz",
            "I_firing_rate_hz",
            "E_I_firing_rate_ratio",
        )
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(run_cfg.analysis.save_raw_eeg):
        np.save(output_dir / "eeg_raw.npy", raw_eeg)
        np.save(output_dir / "eeg_preprocessed.npy", processed_eeg)
        np.savez_compressed(
            output_dir / "psd.npz",
            frequencies_hz=frequencies_hz,
            psd_v2_per_hz=psd,
        )
    return {
        "seed": int(seed),
        "amplitude_v_per_m": float(action[0]),
        "frequency_hz": float(action[1]),
        **feature_values,
        **rates,
    }


def _protocol_id(amplitude: float, frequency: float) -> str:
    return f"E{amplitude:g}_F{frequency:g}".replace(".", "p")


def _make_standardizer(
    reference_rows: dict[int, dict[str, Any]],
    sham_rows: dict[int, dict[str, Any]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scale using discovery references only."""
    seed_order = sorted(reference_rows)
    pooled = np.vstack(
        [
            [reference_rows[seed][name] for name in feature_names]
            for seed in seed_order
        ]
        + [
            [sham_rows[seed][name] for name in feature_names]
            for seed in seed_order
        ]
    ).astype(np.float64)
    center = np.mean(pooled, axis=0)
    scale = np.std(pooled, axis=0, ddof=1)
    positive = scale[scale > np.finfo(np.float64).eps]
    fallback = float(np.median(positive)) if positive.size else 1.0
    scale = np.where(scale > np.finfo(np.float64).eps, scale, fallback)
    return center, scale


def _reachability_metrics(
    *,
    target: dict[str, Any],
    sham: dict[str, Any],
    stimulated: dict[str, Any],
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    def vector(row):
        values = np.asarray([row[name] for name in feature_names], dtype=float)
        return (values - center) / scale

    target_vector = vector(target)
    sham_vector = vector(sham)
    stimulated_vector = vector(stimulated)
    target_shift = target_vector - sham_vector
    stimulation_shift = stimulated_vector - sham_vector
    sham_distance = float(np.linalg.norm(sham_vector - target_vector))
    stimulated_distance = float(
        np.linalg.norm(stimulated_vector - target_vector)
    )
    denominator = float(
        np.linalg.norm(target_shift) * np.linalg.norm(stimulation_shift)
    )
    alignment = (
        float(np.dot(target_shift, stimulation_shift) / denominator)
        if denominator > 0
        else 0.0
    )
    improvement = (
        1.0 - stimulated_distance / sham_distance
        if sham_distance > 0
        else 0.0
    )
    return {
        "sham_distance_to_A": sham_distance,
        "stimulated_distance_to_A": stimulated_distance,
        "fractional_distance_improvement": float(improvement),
        "target_shift_alignment": alignment,
    }


def _within_rate_guardrails(row: dict[str, Any], cfg: DictConfig) -> bool:
    limits = cfg.analysis.rate_guardrails_hz
    return bool(
        float(limits.E_min)
        <= float(row["E_firing_rate_hz"])
        <= float(limits.E_max)
        and float(limits.I_min)
        <= float(row["I_firing_rate_hz"])
        <= float(limits.I_max)
    )


def _bootstrap_ci(values: np.ndarray, *, seed: int, n_bootstrap: int):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize_protocols(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            ["protocol_id", "amplitude_v_per_m", "frequency_hz"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_fractional_improvement=(
                "fractional_distance_improvement",
                "mean",
            ),
            median_fractional_improvement=(
                "fractional_distance_improvement",
                "median",
            ),
            mean_alignment=("target_shift_alignment", "mean"),
            all_rate_safe=("rate_safe", "all"),
            mean_E_rate_hz=("E_firing_rate_hz", "mean"),
            mean_I_rate_hz=("I_firing_rate_hz", "mean"),
        )
        .sort_values(
            ["all_rate_safe", "mean_fractional_improvement"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def _plot_discovery_surface(summary: pd.DataFrame, output_dir: Path) -> None:
    pivot = summary.pivot(
        index="amplitude_v_per_m",
        columns="frequency_hz",
        values="mean_fractional_improvement",
    ).sort_index()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    axis.set_xticks(np.arange(pivot.columns.size), pivot.columns)
    axis.set_yticks(np.arange(pivot.index.size), pivot.index)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Field amplitude (V/m)")
    axis.set_title("Discovery: fractional distance improvement toward A")
    figure.colorbar(image, ax=axis, label="1 − d(stim,A) / d(sham,A)")
    figure.tight_layout()
    figure.savefig(output_dir / "discovery_response_surface.png", dpi=250)
    plt.close(figure)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("The stimulation reachability experiment requires online mode.")

    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "stimulation_reachability"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### BallAndStick stimulation reachability")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    base_seed = int(cfg.experiment.seed)
    discovery_seeds = [
        base_seed + int(cfg.analysis.discovery.seed_offset) + index
        for index in range(int(cfg.analysis.discovery.n_seeds))
    ]
    validation_seeds = [
        base_seed + int(cfg.analysis.validation.seed_offset) + index
        for index in range(int(cfg.analysis.validation.n_seeds))
    ]
    if set(discovery_seeds).intersection(validation_seeds):
        raise ValueError("Discovery and validation seed sets must be disjoint.")
    all_seeds = discovery_seeds + validation_seeds
    feature_names = list(cfg.analysis.distance_features)
    reference_a: dict[int, dict[str, Any]] = {}
    sham_b: dict[int, dict[str, Any]] = {}

    # Matched no-stimulation counterfactuals for every circuit.
    for seed in all_seeds:
        if rank == 0:
            print(f"reference seed={seed}: A and B-sham")
        a_row = _run_episode(
            cfg,
            inhibition_scale=float(cfg.analysis.condition_a_inhibition_scale),
            seed=seed,
            action=(0.0, 0.0),
            output_dir=root / "reference_A" / f"seed_{seed}",
            comm=comm,
            size=size,
            rank=rank,
        )
        b_row = _run_episode(
            cfg,
            inhibition_scale=float(cfg.analysis.condition_b_inhibition_scale),
            seed=seed,
            action=(0.0, 0.0),
            output_dir=root / "sham_B" / f"seed_{seed}",
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            reference_a[seed] = a_row
            sham_b[seed] = b_row

    if rank == 0:
        center, scale = _make_standardizer(
            {seed: reference_a[seed] for seed in discovery_seeds},
            {seed: sham_b[seed] for seed in discovery_seeds},
            feature_names,
        )
    else:
        center = scale = None
    center = comm.bcast(center, root=0)
    scale = comm.bcast(scale, root=0)

    candidates = [
        (float(amplitude), float(frequency))
        for amplitude in cfg.analysis.discovery.amplitudes_v_per_m
        for frequency in cfg.analysis.discovery.frequencies_hz
    ]
    discovery_rows: list[dict[str, Any]] = []
    for candidate_index, action in enumerate(candidates, start=1):
        protocol_id = _protocol_id(*action)
        for seed in discovery_seeds:
            if rank == 0:
                print(
                    f"discovery {candidate_index}/{len(candidates)} "
                    f"{protocol_id}, seed={seed}"
                )
            row = _run_episode(
                cfg,
                inhibition_scale=float(
                    cfg.analysis.condition_b_inhibition_scale
                ),
                seed=seed,
                action=action,
                output_dir=root / "discovery" / protocol_id / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                row.update(
                    _reachability_metrics(
                        target=reference_a[seed],
                        sham=sham_b[seed],
                        stimulated=row,
                        feature_names=feature_names,
                        center=center,
                        scale=scale,
                    )
                )
                row["protocol_id"] = protocol_id
                row["rate_safe"] = _within_rate_guardrails(row, cfg)
                discovery_rows.append(row)

    if rank == 0:
        discovery_df = pd.DataFrame(discovery_rows)
        discovery_df.to_csv(root / "discovery_seed_results.csv", index=False)
        discovery_summary = _summarize_protocols(discovery_df)
        discovery_summary.to_csv(root / "discovery_protocol_summary.csv", index=False)
        safe = discovery_summary[discovery_summary["all_rate_safe"]]
        ranked = safe if not safe.empty else discovery_summary
        selected = ranked.head(int(cfg.analysis.discovery.top_k))[
            ["amplitude_v_per_m", "frequency_hz", "protocol_id"]
        ].to_dict("records")
        if bool(cfg.experiment.plot):
            _plot_discovery_surface(discovery_summary, root)
    else:
        selected = None
    selected = comm.bcast(selected, root=0)

    validation_rows: list[dict[str, Any]] = []
    for selection in selected:
        action = (
            float(selection["amplitude_v_per_m"]),
            float(selection["frequency_hz"]),
        )
        protocol_id = str(selection["protocol_id"])
        for seed in validation_seeds:
            if rank == 0:
                print(f"validation {protocol_id}, seed={seed}")
            row = _run_episode(
                cfg,
                inhibition_scale=float(
                    cfg.analysis.condition_b_inhibition_scale
                ),
                seed=seed,
                action=action,
                output_dir=root / "validation" / protocol_id / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                row.update(
                    _reachability_metrics(
                        target=reference_a[seed],
                        sham=sham_b[seed],
                        stimulated=row,
                        feature_names=feature_names,
                        center=center,
                        scale=scale,
                    )
                )
                row["protocol_id"] = protocol_id
                row["rate_safe"] = _within_rate_guardrails(row, cfg)
                validation_rows.append(row)

    if rank == 0:
        validation_df = pd.DataFrame(validation_rows)
        validation_df.to_csv(root / "validation_seed_results.csv", index=False)
        summary_rows = []
        for selection_index, selection in enumerate(selected):
            protocol_id = str(selection["protocol_id"])
            subset = validation_df[validation_df["protocol_id"] == protocol_id]
            improvements = subset[
                "fractional_distance_improvement"
            ].to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_ci(
                improvements,
                seed=base_seed + 80_003 + selection_index,
                n_bootstrap=int(cfg.analysis.validation.n_bootstrap),
            )
            summary_rows.append(
                {
                    **selection,
                    "selection_rank": selection_index + 1,
                    "n_validation_seeds": int(subset["seed"].nunique()),
                    "mean_fractional_improvement": float(
                        np.mean(improvements)
                    ),
                    "ci_2.5": ci_low,
                    "ci_97.5": ci_high,
                    "median_alignment": float(
                        np.median(subset["target_shift_alignment"])
                    ),
                    "all_rate_safe": bool(subset["rate_safe"].all()),
                }
            )
        validation_summary = pd.DataFrame(summary_rows)
        validation_summary.to_csv(root / "validation_protocol_summary.csv", index=False)
        primary = summary_rows[0]
        success = bool(
            primary["ci_2.5"] > 0.0
            and primary["median_alignment"] > 0.0
            and primary["all_rate_safe"]
        )
        conclusion = {
            "primary_protocol": primary,
            "success_criterion": (
                "Discovery-ranked protocol has held-out 95% bootstrap CI "
                "above zero for fractional distance improvement, positive "
                "median target-shift alignment, and satisfies rate guardrails."
            ),
            "evidence_of_reachability": success,
            "distance_features": feature_names,
            "standardizer_center": center.tolist(),
            "standardizer_scale": scale.tolist(),
            "action_units": {
                "amplitude": "V/m at modeled tissue",
                "frequency": "Hz",
            },
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        with (root / "reachability_conclusion.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(conclusion, handle, indent=2)

        print("\n### Held-out validation")
        print(validation_summary.to_string(index=False))
        print(
            "\nReachability criterion: "
            f"{'PASSED' if success else 'NOT PASSED'}"
        )
        print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
