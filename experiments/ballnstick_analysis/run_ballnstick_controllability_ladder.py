"""Validate the EEG target and a selective-I positive-control actuator.

The first ladder stage varies the causal I-to-E conductance scale without
stimulation and tests whether the prespecified EEG distance approaches the
Condition-A reference monotonically. The second stage keeps Condition B fixed
and increases only the excitatory background-synapse weight onto I cells.
Discovery selects a multiplier; disjoint validation seeds test it.

Selective I drive is a mechanistic controllability control. It is not a model
of transcranial stimulation, optogenetic dose, or clinical intervention.
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
from scipy.stats import spearmanr


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from experiments.ballnstick_analysis.run_ballnstick import (  # noqa: E402
    _extract_eeg_features,
    _preprocess_eeg,
)


FAMILY_INTERPOLATION = "inhibition_interpolation"
FAMILY_I_DRIVE = "selective_i_drive"


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


def _level_id(prefix: str, value: float) -> str:
    encoded = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_{encoded}"


def _episode_config(
    base_cfg: DictConfig,
    *,
    inhibition_scale: float,
    i_background_weight_multiplier: float,
    seed: int,
    output_dir: Path,
) -> DictConfig:
    """Make one stationary episode without mutating the Hydra config."""
    if float(inhibition_scale) < 0.0:
        raise ValueError("inhibition_scale must be non-negative.")
    if float(i_background_weight_multiplier) <= 0.0:
        raise ValueError("I-background weight multiplier must be positive.")

    run_cfg = _plain_copy(base_cfg)
    OmegaConf.set_struct(run_cfg, False)
    burn_steps = int(run_cfg.analysis.timeline.burn_in_steps)
    analysis_steps = int(run_cfg.analysis.timeline.analysis_steps)
    if burn_steps < 1 or analysis_steps < 1:
        raise ValueError("Burn-in and analysis must each contain at least one window.")
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    base_i_weight = float(run_cfg.env.network.background.I.weight)

    with open_dict(run_cfg):
        run_cfg.experiment.seed = int(seed)
        run_cfg.experiment.dir = str(output_dir)
        run_cfg.env.simulation.duration = (burn_steps + analysis_steps) * window_ms
        run_cfg.env.network.inhibition_scale = float(inhibition_scale)
        run_cfg.env.network.background.I.weight = (
            base_i_weight * float(i_background_weight_multiplier)
        )
        run_cfg.env.ts.apply = True
        run_cfg.env.online.temperature_mode = "configured"
        run_cfg.env.online.stimulation.parameterization = "uniform_field"
    return run_cfg


def _run_episode(
    base_cfg: DictConfig,
    *,
    cohort: str,
    family: str,
    level_id: str,
    inhibition_scale: float,
    i_background_weight_multiplier: float,
    seed: int,
    output_dir: Path,
    comm,
    size: int,
    rank: int,
) -> dict[str, Any] | None:
    """Run a fresh matched circuit and return its post-burn-in features."""
    run_cfg = _episode_config(
        base_cfg,
        inhibition_scale=inhibition_scale,
        i_background_weight_multiplier=i_background_weight_multiplier,
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
    analysis_steps = int(run_cfg.analysis.timeline.analysis_steps)

    try:
        outputs = environment.analysis_rollout_online(
            [[0.0, 0.0]] * (burn_steps + analysis_steps),
            phase_continuous=True,
            ramp_ms=0.0,
        )
    finally:
        environment.close()

    if rank != 0:
        return None
    retained = outputs[burn_steps:]
    if len(retained) != analysis_steps:
        raise RuntimeError(
            f"Expected {analysis_steps} retained windows, received {len(retained)}."
        )
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
    features, frequencies_hz, psd = _extract_eeg_features(
        processed_eeg,
        analysis_fs_hz,
    )

    duration_s = sum(
        float(output["t_stop_ms"] - output["t_start_ms"]) for output in retained
    ) / 1000.0
    rates: dict[str, float] = {}
    for population_name in ("E", "I"):
        count = sum(
            int(np.asarray(output["spikes"][population_name]["times_ms"]).size)
            for output in retained
        )
        population_size = len(retained[0]["spikes"][population_name]["per_cell"])
        rates[f"{population_name}_spike_count"] = int(count)
        rates[f"{population_name}_firing_rate_hz"] = (
            count / (population_size * duration_s)
        )
    rates["E_I_firing_rate_ratio"] = (
        rates["E_firing_rate_hz"] / rates["I_firing_rate_hz"]
        if rates["I_firing_rate_hz"] > 0.0
        else float("nan")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(run_cfg.analysis.save_raw_eeg):
        np.savez_compressed(
            output_dir / "signals.npz",
            eeg_raw_v=raw_eeg,
            eeg_preprocessed_v=processed_eeg,
        )
        np.savez_compressed(
            output_dir / "psd.npz",
            frequencies_hz=frequencies_hz,
            psd_v2_per_hz=psd,
        )

    return {
        "cohort": cohort,
        "seed": int(seed),
        "family": family,
        "level_id": level_id,
        "inhibition_scale": float(inhibition_scale),
        "i_background_weight_multiplier": float(i_background_weight_multiplier),
        "i_background_weight_uS": float(
            run_cfg.env.network.background.I.weight
        ),
        "analysis_duration_s": duration_s,
        **features,
        **rates,
    }


def _make_standardizer(
    reference_a: dict[int, dict[str, Any]],
    reference_b: dict[int, dict[str, Any]],
    seeds: list[int],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack(
        [
            [reference_a[seed][name] for name in feature_names]
            for seed in seeds
        ]
        + [
            [reference_b[seed][name] for name in feature_names]
            for seed in seeds
        ]
    ).astype(np.float64)
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    positive = scale[scale > np.finfo(np.float64).eps]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return center, np.where(scale > np.finfo(np.float64).eps, scale, fallback)


def _reachability_metrics(
    *,
    target: dict[str, Any],
    sham: dict[str, Any],
    candidate: dict[str, Any],
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    def vector(row: dict[str, Any]) -> np.ndarray:
        values = np.asarray([row[name] for name in feature_names], dtype=float)
        return (values - center) / scale

    target_vector = vector(target)
    sham_vector = vector(sham)
    candidate_vector = vector(candidate)
    target_shift = target_vector - sham_vector
    candidate_shift = candidate_vector - sham_vector
    sham_distance = float(np.linalg.norm(sham_vector - target_vector))
    candidate_distance = float(np.linalg.norm(candidate_vector - target_vector))
    denominator = float(np.linalg.norm(target_shift) * np.linalg.norm(candidate_shift))
    alignment = (
        float(np.dot(target_shift, candidate_shift) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return {
        "sham_distance_to_A": sham_distance,
        "candidate_distance_to_A": candidate_distance,
        "fractional_distance_improvement": (
            1.0 - candidate_distance / sham_distance
            if sham_distance > 0.0
            else 0.0
        ),
        "target_shift_alignment": alignment,
        "max_abs_target_error_z": float(
            np.max(np.abs(candidate_vector - target_vector))
        ),
    }


def _rate_safe(
    candidate: dict[str, Any],
    target: dict[str, Any],
    sham: dict[str, Any],
    cfg: DictConfig,
) -> bool:
    limits = cfg.analysis.rate_guardrails_hz
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    for population_name in ("E", "I"):
        name = f"{population_name}_firing_rate_hz"
        value = float(candidate[name])
        absolute_low = float(limits[f"{population_name}_min"])
        absolute_high = float(limits[f"{population_name}_max"])
        reference_low = min(float(target[name]), float(sham[name]))
        reference_high = max(float(target[name]), float(sham[name]))
        relative_low = max(0.0, reference_low * (1.0 - tolerance))
        relative_high = reference_high * (1.0 + tolerance)
        if not (
            absolute_low <= value <= absolute_high
            and relative_low <= value <= relative_high
        ):
            return False
    return True


def _decorate_row(
    row: dict[str, Any],
    *,
    target: dict[str, Any],
    sham: dict[str, Any],
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    cfg: DictConfig,
) -> dict[str, Any]:
    row.update(
        _reachability_metrics(
            target=target,
            sham=sham,
            candidate=row,
            feature_names=feature_names,
            center=center,
            scale=scale,
        )
    )
    row["rate_safe"] = _rate_safe(row, target, sham, cfg)
    for rate_name in (
        "E_firing_rate_hz",
        "I_firing_rate_hz",
        "E_I_firing_rate_ratio",
    ):
        row[f"{rate_name}_change_vs_B"] = float(row[rate_name]) - float(
            sham[rate_name]
        )
        row[f"{rate_name}_target_A_minus_B"] = float(target[rate_name]) - float(
            sham[rate_name]
        )
    return row


def _bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_p(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(np.mean(values)))
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, values.size))
    null = np.abs(np.mean(signs * values, axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))


def _summarize_levels(
    rows: pd.DataFrame,
    *,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    margin = float(cfg.analysis.criteria.equivalence_z_margin)
    grouping = [
        "cohort",
        "family",
        "level_id",
        "inhibition_scale",
        "i_background_weight_multiplier",
    ]
    for keys, group in rows.groupby(grouping, sort=False, dropna=False):
        improvement = group["fractional_distance_improvement"].to_numpy(float)
        ci_low, ci_high = _bootstrap_ci(
            improvement,
            rng=rng,
            n_bootstrap=int(cfg.analysis.n_bootstrap),
        )
        summaries.append(
            {
                **dict(zip(grouping, keys)),
                "n_seeds": int(group["seed"].nunique()),
                "mean_fractional_improvement": float(np.mean(improvement)),
                "ci_2.5": ci_low,
                "ci_97.5": ci_high,
                "paired_sign_flip_p": _sign_flip_p(
                    improvement,
                    rng=rng,
                    n_permutations=int(cfg.analysis.n_permutations),
                ),
                "positive_seed_fraction": float(np.mean(improvement > 0.0)),
                "median_alignment": float(np.median(group["target_shift_alignment"])),
                "mean_sham_distance": float(group["sham_distance_to_A"].mean()),
                "mean_candidate_distance": float(group["candidate_distance_to_A"].mean()),
                "equivalent_seed_fraction": float(
                    np.mean(group["max_abs_target_error_z"] <= margin)
                ),
                "all_rate_safe": bool(group["rate_safe"].all()),
                "mean_E_rate_hz": float(group["E_firing_rate_hz"].mean()),
                "mean_I_rate_hz": float(group["I_firing_rate_hz"].mean()),
                "mean_E_I_rate_ratio": float(group["E_I_firing_rate_ratio"].mean()),
                "mean_E_rate_change_vs_B_hz": float(
                    group["E_firing_rate_hz_change_vs_B"].mean()
                ),
                "mean_I_rate_change_vs_B_hz": float(
                    group["I_firing_rate_hz_change_vs_B"].mean()
                ),
                "mean_E_I_ratio_change_vs_B": float(
                    group["E_I_firing_rate_ratio_change_vs_B"].mean()
                ),
            }
        )
    return pd.DataFrame(summaries)


def _interpolation_seed_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure rank monotonicity of distance versus causal IE scale."""
    records: list[dict[str, Any]] = []
    subset = rows[rows["family"] == FAMILY_INTERPOLATION]
    for seed, group in subset.groupby("seed"):
        ordered = group.sort_values("inhibition_scale")
        scales = ordered["inhibition_scale"].to_numpy(dtype=float)
        distances = ordered["candidate_distance_to_A"].to_numpy(dtype=float)
        statistic = float(spearmanr(scales, distances).statistic)
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(distances))))
        records.append(
            {
                "cohort": str(ordered["cohort"].iloc[0]),
                "seed": int(seed),
                "scale_distance_spearman": statistic,
                "negative_spearman": bool(statistic < 0.0),
                "strictly_nonincreasing_distance": bool(
                    np.all(np.diff(distances) <= tolerance)
                ),
                "distance_at_B": float(distances[0]),
                "distance_at_A": float(distances[-1]),
            }
        )
    return pd.DataFrame(records)


def _select_i_drive(summary: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    candidates = summary[
        (summary["family"] == FAMILY_I_DRIVE)
        & (summary["i_background_weight_multiplier"] > 1.0)
    ].copy()
    if candidates.empty:
        raise ValueError("At least one selective-I multiplier above 1.0 is required.")
    safe = candidates[candidates["all_rate_safe"]]
    ranked = safe if not safe.empty else candidates
    ranked = ranked.sort_values(
        ["mean_fractional_improvement", "median_alignment"],
        ascending=[False, False],
    )
    return ranked.head(int(top_k))[
        ["level_id", "i_background_weight_multiplier"]
    ].to_dict("records")


def _plot_ladder(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for cohort, style in (("discovery", "--o"), ("validation", "-o")):
        interpolation = summary[
            (summary["cohort"] == cohort)
            & (summary["family"] == FAMILY_INTERPOLATION)
        ].sort_values("inhibition_scale")
        if not interpolation.empty:
            axes[0].plot(
                interpolation["inhibition_scale"],
                interpolation["mean_fractional_improvement"],
                style,
                label=cohort,
            )
        i_drive = summary[
            (summary["cohort"] == cohort)
            & (summary["family"] == FAMILY_I_DRIVE)
        ].sort_values("i_background_weight_multiplier")
        if not i_drive.empty:
            axes[1].plot(
                i_drive["i_background_weight_multiplier"],
                i_drive["mean_fractional_improvement"],
                style,
                label=cohort,
            )
    axes[0].set_xlabel("I-to-E conductance scale")
    axes[0].set_ylabel("Fractional distance improvement toward A")
    axes[0].set_title("Ground-truth causal interpolation")
    axes[1].set_xlabel("I-background weight multiplier")
    axes[1].set_title("Selective inhibitory-population drive")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "controllability_ladder.png", dpi=250)
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
        raise ValueError("The controllability ladder requires the online simulator.")

    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "controllability_ladder"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### BallAndStick causal controllability ladder")
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
    if not discovery_seeds or not validation_seeds:
        raise ValueError("Discovery and validation must each contain at least one seed.")
    if set(discovery_seeds).intersection(validation_seeds):
        raise ValueError("Discovery and validation seed sets must be disjoint.")

    inhibition_scales = sorted(
        {float(value) for value in cfg.analysis.causal_interpolation.inhibition_scales}
    )
    if len(inhibition_scales) < 2:
        raise ValueError("At least two causal interpolation scales are required.")
    condition_a = float(cfg.analysis.condition_a_inhibition_scale)
    condition_b = float(cfg.analysis.condition_b_inhibition_scale)
    if not np.isclose(inhibition_scales[0], condition_b) or not np.isclose(
        inhibition_scales[-1], condition_a
    ):
        raise ValueError("Interpolation must include Condition B first and A last.")
    i_multipliers = sorted(
        {
            float(value)
            for value in cfg.analysis.selective_i_drive.background_weight_multipliers
        }
    )
    if len(i_multipliers) < 2:
        raise ValueError("At least two selective-I multipliers are required.")
    if int(cfg.analysis.selective_i_drive.top_k) < 1:
        raise ValueError("selective_i_drive.top_k must be at least one.")
    if not np.isclose(i_multipliers[0], 1.0):
        raise ValueError("Selective-I grid must include multiplier 1.0 as sham.")
    feature_names = list(cfg.analysis.distance_features)

    references_a: dict[int, dict[str, Any]] = {}
    references_b: dict[int, dict[str, Any]] = {}
    for cohort, seeds in (("discovery", discovery_seeds), ("validation", validation_seeds)):
        for seed in seeds:
            if rank == 0:
                print(f"{cohort} references seed={seed}: A and B")
            a_row = _run_episode(
                cfg,
                cohort=cohort,
                family=FAMILY_INTERPOLATION,
                level_id=_level_id("IE", condition_a),
                inhibition_scale=condition_a,
                i_background_weight_multiplier=1.0,
                seed=seed,
                output_dir=root / cohort / "references" / "A" / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            b_row = _run_episode(
                cfg,
                cohort=cohort,
                family=FAMILY_INTERPOLATION,
                level_id=_level_id("IE", condition_b),
                inhibition_scale=condition_b,
                i_background_weight_multiplier=1.0,
                seed=seed,
                output_dir=root / cohort / "references" / "B" / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                references_a[seed] = a_row
                references_b[seed] = b_row

    if rank == 0:
        center, scale = _make_standardizer(
            references_a,
            references_b,
            discovery_seeds,
            feature_names,
        )
    else:
        center = scale = None
    center = comm.bcast(center, root=0)
    scale = comm.bcast(scale, root=0)

    rows_by_cohort: dict[str, list[dict[str, Any]]] = {
        "discovery": [],
        "validation": [],
    }

    # Both cohorts receive the complete causal interpolation; it is not tuned.
    for cohort, seeds in (("discovery", discovery_seeds), ("validation", validation_seeds)):
        for seed in seeds:
            for inhibition_scale in inhibition_scales:
                if np.isclose(inhibition_scale, condition_a):
                    row = dict(references_a[seed]) if rank == 0 else None
                elif np.isclose(inhibition_scale, condition_b):
                    row = dict(references_b[seed]) if rank == 0 else None
                else:
                    if rank == 0:
                        print(
                            f"{cohort} causal scale={inhibition_scale:g}, seed={seed}"
                        )
                    row = _run_episode(
                        cfg,
                        cohort=cohort,
                        family=FAMILY_INTERPOLATION,
                        level_id=_level_id("IE", inhibition_scale),
                        inhibition_scale=inhibition_scale,
                        i_background_weight_multiplier=1.0,
                        seed=seed,
                        output_dir=(
                            root
                            / cohort
                            / "inhibition_interpolation"
                            / _level_id("IE", inhibition_scale)
                            / f"seed_{seed}"
                        ),
                        comm=comm,
                        size=size,
                        rank=rank,
                    )
                if rank == 0:
                    row["cohort"] = cohort
                    row["family"] = FAMILY_INTERPOLATION
                    row["level_id"] = _level_id("IE", inhibition_scale)
                    row["inhibition_scale"] = inhibition_scale
                    row["i_background_weight_multiplier"] = 1.0
                    rows_by_cohort[cohort].append(
                        _decorate_row(
                            row,
                            target=references_a[seed],
                            sham=references_b[seed],
                            feature_names=feature_names,
                            center=center,
                            scale=scale,
                            cfg=cfg,
                        )
                    )

    # Discovery selective-I grid. Multiplier 1.0 reuses the matched B row.
    for seed in discovery_seeds:
        for multiplier in i_multipliers:
            if np.isclose(multiplier, 1.0):
                row = dict(references_b[seed]) if rank == 0 else None
            else:
                if rank == 0:
                    print(f"discovery selective-I x{multiplier:g}, seed={seed}")
                row = _run_episode(
                    cfg,
                    cohort="discovery",
                    family=FAMILY_I_DRIVE,
                    level_id=_level_id("Ibg", multiplier),
                    inhibition_scale=condition_b,
                    i_background_weight_multiplier=multiplier,
                    seed=seed,
                    output_dir=(
                        root
                        / "discovery"
                        / "selective_i_drive"
                        / _level_id("Ibg", multiplier)
                        / f"seed_{seed}"
                    ),
                    comm=comm,
                    size=size,
                    rank=rank,
                )
            if rank == 0:
                row["cohort"] = "discovery"
                row["family"] = FAMILY_I_DRIVE
                row["level_id"] = _level_id("Ibg", multiplier)
                row["inhibition_scale"] = condition_b
                row["i_background_weight_multiplier"] = multiplier
                rows_by_cohort["discovery"].append(
                    _decorate_row(
                        row,
                        target=references_a[seed],
                        sham=references_b[seed],
                        feature_names=feature_names,
                        center=center,
                        scale=scale,
                        cfg=cfg,
                    )
                )

    rng = np.random.default_rng(base_seed + 910_001) if rank == 0 else None
    if rank == 0:
        discovery_df = pd.DataFrame(rows_by_cohort["discovery"])
        discovery_summary = _summarize_levels(discovery_df, cfg=cfg, rng=rng)
        selected = _select_i_drive(
            discovery_summary,
            top_k=int(cfg.analysis.selective_i_drive.top_k),
        )
    else:
        selected = None
    selected = comm.bcast(selected, root=0)

    # Freeze the discovery ranking and evaluate only those I-drive levels.
    for selection in selected:
        multiplier = float(selection["i_background_weight_multiplier"])
        level_id = str(selection["level_id"])
        for seed in validation_seeds:
            if rank == 0:
                print(f"validation selective-I x{multiplier:g}, seed={seed}")
            row = _run_episode(
                cfg,
                cohort="validation",
                family=FAMILY_I_DRIVE,
                level_id=level_id,
                inhibition_scale=condition_b,
                i_background_weight_multiplier=multiplier,
                seed=seed,
                output_dir=(
                    root
                    / "validation"
                    / "selective_i_drive"
                    / level_id
                    / f"seed_{seed}"
                ),
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                rows_by_cohort["validation"].append(
                    _decorate_row(
                        row,
                        target=references_a[seed],
                        sham=references_b[seed],
                        feature_names=feature_names,
                        center=center,
                        scale=scale,
                        cfg=cfg,
                    )
                )

    if rank == 0:
        validation_df = pd.DataFrame(rows_by_cohort["validation"])
        validation_summary = _summarize_levels(validation_df, cfg=cfg, rng=rng)
        all_rows = pd.concat([discovery_df, validation_df], ignore_index=True)
        all_summary = pd.concat(
            [discovery_summary, validation_summary], ignore_index=True
        )
        discovery_df.to_csv(root / "discovery_episode_features.csv", index=False)
        discovery_summary.to_csv(root / "discovery_level_summary.csv", index=False)
        validation_df.to_csv(root / "validation_episode_features.csv", index=False)
        validation_summary.to_csv(root / "validation_level_summary.csv", index=False)
        all_rows.to_csv(root / "all_episode_features.csv", index=False)
        all_summary.to_csv(root / "all_level_summary.csv", index=False)

        monotonicity = _interpolation_seed_metrics(validation_df)
        monotonicity.to_csv(
            root / "validation_interpolation_monotonicity_by_seed.csv",
            index=False,
        )
        interpolation_validation = validation_summary[
            validation_summary["family"] == FAMILY_INTERPOLATION
        ]
        high_scale = float(cfg.analysis.causal_interpolation.high_scale_for_gate)
        high_row = interpolation_validation[
            np.isclose(interpolation_validation["inhibition_scale"], high_scale)
        ]
        if len(high_row) != 1:
            raise ValueError(
                "high_scale_for_gate must identify exactly one interpolation level."
            )
        high_row = high_row.iloc[0]
        criteria = cfg.analysis.criteria
        interpolation_checks = {
            "median_rank_monotonicity": bool(
                float(monotonicity["scale_distance_spearman"].median())
                <= float(criteria.maximum_median_scale_distance_spearman)
            ),
            "seed_rank_consistency": bool(
                float(monotonicity["negative_spearman"].mean())
                >= float(criteria.minimum_negative_spearman_seed_fraction)
            ),
            "high_scale_closure": bool(
                float(high_row["ci_2.5"])
                > float(criteria.minimum_high_scale_mean_improvement)
            ),
            "rate_safe": bool(interpolation_validation["all_rate_safe"].all()),
        }
        metric_tracks_causal_parameter = bool(all(interpolation_checks.values()))

        primary_selection = selected[0]
        primary = validation_summary[
            (validation_summary["family"] == FAMILY_I_DRIVE)
            & (validation_summary["level_id"] == primary_selection["level_id"])
        ].iloc[0]
        i_drive_checks = {
            "practically_meaningful_improvement": bool(
                float(primary["ci_2.5"])
                > float(criteria.minimum_mean_improvement)
            ),
            "seed_consistency": bool(
                float(primary["positive_seed_fraction"])
                >= float(criteria.minimum_positive_seed_fraction)
            ),
            "positive_alignment": bool(float(primary["median_alignment"]) > 0.0),
            "rate_safe": bool(primary["all_rate_safe"]),
        }
        selective_i_directional_control = bool(all(i_drive_checks.values()))
        selective_i_a_like_reachability = bool(
            metric_tracks_causal_parameter
            and selective_i_directional_control
            and float(primary["equivalent_seed_fraction"])
            >= float(criteria.minimum_equivalent_seed_fraction)
        )

        conclusion = {
            "scientific_scope": (
                "A transparent 40-cell mechanistic control experiment. "
                "Selective I drive is not transcranial stimulation or treatment."
            ),
            "distance_features": feature_names,
            "standardizer": {"center": center.tolist(), "scale": scale.tolist()},
            "discovery_seeds": discovery_seeds,
            "validation_seeds": validation_seeds,
            "selected_i_drive_protocols": selected,
            "causal_interpolation_checks": interpolation_checks,
            "metric_tracks_causal_parameter": metric_tracks_causal_parameter,
            "primary_selective_i_drive": primary.to_dict(),
            "selective_i_drive_checks": i_drive_checks,
            "selective_i_directional_control": selective_i_directional_control,
            "selective_i_a_like_reachability": selective_i_a_like_reachability,
            "next_decision": (
                "Proceed to a uniform-field transfer-function/controllability "
                "screen only if the EEG metric tracks the causal interpolation. "
                "Interpret selective-I success as actuator positive control, "
                "not evidence for TES."
            ),
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        with (root / "controllability_conclusion.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(conclusion, handle, indent=2)
        if bool(cfg.experiment.plot):
            _plot_ladder(all_summary, root)

        print("\n### Held-out causal interpolation")
        print(interpolation_validation.to_string(index=False))
        print("\n### Held-out selective-I controls")
        print(
            validation_summary[
                validation_summary["family"] == FAMILY_I_DRIVE
            ].to_string(index=False)
        )
        print("\nCausal metric gate: " + ("PASSED" if metric_tracks_causal_parameter else "NOT PASSED"))
        print(
            "Selective-I directional control: "
            + ("PASSED" if selective_i_directional_control else "NOT PASSED")
        )
        print(
            "Selective-I A-like reachability: "
            + ("PASSED" if selective_i_a_like_reachability else "NOT PASSED")
        )
        print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
