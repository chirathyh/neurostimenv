"""Compare two ball-and-stick circuit conditions using simulated EEG.

The experiment uses matched circuit seeds: seed k is simulated once under
condition A and once under condition B. The circuit, rather than each EEG
window, is treated as the statistical unit.
"""

from __future__ import annotations

import math
import random
import shutil
import sys
import time
import warnings
import json
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

warnings.simplefilter("ignore", Warning)

from env.models.neuron.env import NeuronEnv  # noqa: E402
from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402


FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 80.0),
}

STATISTICAL_FEATURES = [
    "log10_rms_v",
    "log10_total_power_1_80",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_beta_power",
    "relative_gamma_power",
    "dominant_frequency_hz",
    "spectral_entropy",
    "E_firing_rate_hz",
    "I_firing_rate_hz",
    "E_I_firing_rate_ratio",
]

MECHANISTIC_FEATURES = {
    "E_firing_rate_hz",
    "I_firing_rate_hz",
    "E_I_firing_rate_ratio",
}

PLOT_FEATURES = [
    "log10_total_power_1_80",
    "relative_beta_power",
    "dominant_frequency_hz",
    "spectral_entropy",
]


def _as_plain_config(cfg: DictConfig) -> DictConfig:
    """Return an independent, fully resolved copy of a Hydra configuration."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _prepare_run_config(
    base_cfg: DictConfig,
    condition_cfg: DictConfig,
    circuit_seed: int,
    run_dir: Path,
) -> DictConfig:
    """Apply a condition's arbitrary Hydra overrides to a fresh config copy."""
    cfg_copy = _as_plain_config(base_cfg)
    overrides = condition_cfg.get("overrides", {})
    run_cfg = OmegaConf.merge(cfg_copy, overrides)
    OmegaConf.set_struct(run_cfg, False)

    n_steps = int(run_cfg.analysis.n_steps)
    obs_win_ms = float(run_cfg.env.simulation.obs_win_len)

    with open_dict(run_cfg):
        run_cfg.experiment.seed = int(circuit_seed)
        run_cfg.experiment.dir = str(run_dir)

        # The NetworkCell is created with env.simulation.duration, while
        # analysis_rollout later sets network.tstop to n_steps * obs_win_len.
        # Keeping these consistent avoids a hidden duration mismatch.
        run_cfg.env.simulation.duration = n_steps * obs_win_ms

        # A/B characterisation is performed without stimulation.
        run_cfg.env.ts.apply = False

    return run_cfg


def _fresh_mpi_variables(
    comm: Any,
    size: int,
    rank: int,
    circuit_seed: int,
) -> dict[str, Any]:
    """Create fresh per-run MPI variables.

    NeuronEnv mutates the dictionary passed to it, so the same dictionary
    should not be reused across circuit simulations.
    """
    return {
        "COMM": comm,
        "SIZE": size,
        "RANK": rank,
        "GLOBALSEED": int(circuit_seed),
        "SEED": int(circuit_seed) * 10_000,
    }


def _seed_local_generators(circuit_seed: int, rank: int) -> None:
    """Reset process-local generators before each matched A/B simulation."""
    rank_seed = int(circuit_seed) * 10_000 + int(rank)
    np.random.seed(rank_seed)
    random.seed(rank_seed)


def _preprocess_eeg(
    eeg: np.ndarray,
    fs_hz: float,
    target_fs_hz: int,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, float]:
    """Detrend, band-pass filter, and downsample one EEG trace."""
    x = np.asarray(eeg, dtype=np.float64).reshape(-1)
    if x.size < 16:
        raise ValueError(f"EEG trace is too short: {x.size} samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("EEG trace contains NaN or infinite values")

    x = ss.detrend(x, type="linear")

    nyquist_hz = 0.5 * fs_hz
    effective_high_hz = min(float(high_hz), 0.95 * nyquist_hz)
    if not 0.0 < low_hz < effective_high_hz:
        raise ValueError(
            f"Invalid filter range [{low_hz}, {effective_high_hz}] Hz "
            f"for sampling frequency {fs_hz} Hz"
        )

    sos = ss.butter(
        4,
        [float(low_hz), effective_high_hz],
        btype="bandpass",
        fs=fs_hz,
        output="sos",
    )
    x = ss.sosfiltfilt(sos, x)

    output_fs_hz = float(fs_hz)
    if target_fs_hz > 0 and target_fs_hz < fs_hz:
        rounded_fs = int(round(fs_hz))
        if not math.isclose(fs_hz, rounded_fs, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "This script expects an integer-valued simulator sampling "
                f"frequency for resample_poly, received {fs_hz}."
            )

        divisor = math.gcd(int(target_fs_hz), rounded_fs)
        up = int(target_fs_hz) // divisor
        down = rounded_fs // divisor
        x = ss.resample_poly(x, up=up, down=down)
        output_fs_hz = float(target_fs_hz)

    x = x - np.mean(x)
    return x, output_fs_hz


def _bandpower(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    limits_hz: tuple[float, float],
) -> float:
    mask = (
        (frequencies_hz >= limits_hz[0])
        & (frequencies_hz < limits_hz[1])
    )
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapz(psd[mask], frequencies_hz[mask]))


def _extract_eeg_features(
    eeg: np.ndarray,
    fs_hz: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calculate one feature vector for one complete circuit trajectory."""
    x = np.asarray(eeg, dtype=np.float64).reshape(-1)
    eps = np.finfo(np.float64).tiny

    nperseg = min(x.size, max(256, int(round(2.0 * fs_hz))))
    frequencies_hz, psd = ss.welch(
        x,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    analysis_mask = (
        (frequencies_hz >= 1.0)
        & (frequencies_hz <= 80.0)
    )
    analysis_freqs = frequencies_hz[analysis_mask]
    analysis_psd = psd[analysis_mask]

    if analysis_freqs.size < 2:
        raise ValueError("Insufficient PSD bins between 1 and 80 Hz")

    total_power = float(np.trapz(analysis_psd, analysis_freqs))
    rms_v = float(np.sqrt(np.mean(np.square(x))))

    probability = analysis_psd / max(float(np.sum(analysis_psd)), eps)
    spectral_entropy = float(
        -np.sum(probability * np.log(probability + eps))
        / np.log(probability.size)
    )

    dominant_frequency_hz = float(
        analysis_freqs[int(np.argmax(analysis_psd))]
    )

    features: dict[str, float] = {
        "mean_v": float(np.mean(x)),
        "std_v": float(np.std(x, ddof=1)),
        "rms_v": rms_v,
        "log10_rms_v": float(np.log10(max(rms_v, eps))),
        "peak_to_peak_v": float(np.ptp(x)),
        "total_power_1_80": total_power,
        "log10_total_power_1_80": float(
            np.log10(max(total_power, eps))
        ),
        "dominant_frequency_hz": dominant_frequency_hz,
        "spectral_entropy": spectral_entropy,
    }

    for band_name, limits_hz in FREQUENCY_BANDS.items():
        power = _bandpower(frequencies_hz, psd, limits_hz)
        features[f"{band_name}_power"] = power
        features[f"log10_{band_name}_power"] = float(
            np.log10(max(power, eps))
        )
        features[f"relative_{band_name}_power"] = (
            power / total_power if total_power > 0 else float("nan")
        )

    return features, frequencies_hz, psd


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_sign_flip_test(
    differences: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    """Two-sided paired permutation test using random sign flips."""
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    if np.allclose(differences, 0.0):
        return 1.0

    observed = abs(float(np.mean(differences)))
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(int(n_permutations), differences.size),
    )
    permuted = np.abs(np.mean(signs * differences, axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (n_permutations + 1))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full_like(p_values, np.nan)
    valid = np.isfinite(p_values)
    if not np.any(valid):
        return adjusted

    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    m = ranked.size

    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked
    adjusted[valid] = q_valid
    return adjusted


def _compare_conditions(
    features_df: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    output_dir: Path,
    seed: int,
    n_bootstrap: int,
    n_permutations: int,
) -> pd.DataFrame:
    """Compare A and B using paired circuits with matching seeds."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []

    a_df = features_df[features_df["condition"] == condition_a].set_index(
        "circuit_seed"
    )
    b_df = features_df[features_df["condition"] == condition_b].set_index(
        "circuit_seed"
    )
    paired_seeds = a_df.index.intersection(b_df.index).sort_values()

    if paired_seeds.size < 3:
        raise ValueError(
            "At least three matched circuit seeds are needed for comparison."
        )

    for feature_name in STATISTICAL_FEATURES:
        a_values = a_df.loc[paired_seeds, feature_name].to_numpy(dtype=float)
        b_values = b_df.loc[paired_seeds, feature_name].to_numpy(dtype=float)
        finite = np.isfinite(a_values) & np.isfinite(b_values)
        a_values = a_values[finite]
        b_values = b_values[finite]
        differences = b_values - a_values

        if differences.size < 3:
            continue

        ci_low, ci_high = _bootstrap_mean_ci(
            differences,
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        p_value = _paired_sign_flip_test(
            differences,
            rng=rng,
            n_permutations=n_permutations,
        )

        difference_sd = float(np.std(differences, ddof=1))
        cohen_dz = (
            float(np.mean(differences) / difference_sd)
            if difference_sd > 0
            else float("nan")
        )

        rows.append(
            {
                "feature": feature_name,
                "feature_family": (
                    "mechanistic_spiking"
                    if feature_name in MECHANISTIC_FEATURES
                    else "eeg"
                ),
                "n_pairs": int(differences.size),
                "A_mean": float(np.mean(a_values)),
                "A_sd": float(np.std(a_values, ddof=1)),
                "B_mean": float(np.mean(b_values)),
                "B_sd": float(np.std(b_values, ddof=1)),
                "mean_difference_B_minus_A": float(np.mean(differences)),
                "difference_ci_2.5": ci_low,
                "difference_ci_97.5": ci_high,
                "cohen_dz": cohen_dz,
                "permutation_p": p_value,
            }
        )

    results = pd.DataFrame(rows)
    results["fdr_q"] = _benjamini_hochberg(
        results["permutation_p"].to_numpy(dtype=float)
    )
    results = results.sort_values(
        ["fdr_q", "permutation_p"],
        na_position="last",
    ).reset_index(drop=True)
    results.to_csv(output_dir / "statistical_comparison.csv", index=False)
    return results


def _nearest_centroid_accuracy(
    features_df: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    feature_names: list[str],
) -> float:
    """Leave-one-circuit-pair-out EEG classification accuracy."""
    pivoted = {
        name: features_df[features_df["condition"] == name].set_index(
            "circuit_seed"
        )
        for name in (condition_a, condition_b)
    }
    paired_seeds = (
        pivoted[condition_a]
        .index.intersection(pivoted[condition_b].index)
        .sort_values()
    )
    correct = 0
    total = 0
    eps = np.finfo(np.float64).eps

    for held_seed in paired_seeds:
        train_seeds = paired_seeds[paired_seeds != held_seed]
        if train_seeds.size < 2:
            continue
        train_a = pivoted[condition_a].loc[
            train_seeds, feature_names
        ].to_numpy(dtype=float)
        train_b = pivoted[condition_b].loc[
            train_seeds, feature_names
        ].to_numpy(dtype=float)
        pooled = np.vstack((train_a, train_b))
        mean = np.mean(pooled, axis=0)
        scale = np.std(pooled, axis=0, ddof=1)
        scale = np.where(scale > eps, scale, 1.0)
        centroid_a = np.mean((train_a - mean) / scale, axis=0)
        centroid_b = np.mean((train_b - mean) / scale, axis=0)

        for true_index, condition_name in enumerate((condition_a, condition_b)):
            sample = pivoted[condition_name].loc[
                held_seed, feature_names
            ].to_numpy(dtype=float)
            standardized = (sample - mean) / scale
            distances = [
                float(np.linalg.norm(standardized - centroid_a)),
                float(np.linalg.norm(standardized - centroid_b)),
            ]
            correct += int(int(np.argmin(distances)) == true_index)
            total += 1

    return float(correct / total) if total else float("nan")


def _condition_discriminability(
    features_df: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    feature_names: list[str],
    n_permutations: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Estimate out-of-sample EEG separability without window leakage."""
    required = {"condition", "circuit_seed", *feature_names}
    missing = required.difference(features_df.columns)
    if missing:
        raise ValueError(f"Classification features are missing: {sorted(missing)}")

    working = features_df[
        features_df["condition"].isin([condition_a, condition_b])
    ][list(required)].copy()
    if not np.all(
        np.isfinite(working[feature_names].to_numpy(dtype=float))
    ):
        raise ValueError("Classification features contain non-finite values.")

    observed = _nearest_centroid_accuracy(
        working,
        condition_a=condition_a,
        condition_b=condition_b,
        feature_names=feature_names,
    )
    rng = np.random.default_rng(seed)
    null_accuracies = np.empty(int(n_permutations), dtype=np.float64)
    paired_seeds = np.sort(working["circuit_seed"].unique())

    for permutation_index in range(int(n_permutations)):
        permuted = working.copy()
        # Exchangeability is within each matched circuit: randomly swap A/B.
        swap_seeds = paired_seeds[
            rng.integers(0, 2, size=paired_seeds.size).astype(bool)
        ]
        for circuit_seed in swap_seeds:
            mask = permuted["circuit_seed"] == circuit_seed
            permuted.loc[mask, "condition"] = permuted.loc[
                mask, "condition"
            ].map({condition_a: condition_b, condition_b: condition_a})
        null_accuracies[permutation_index] = _nearest_centroid_accuracy(
            permuted,
            condition_a=condition_a,
            condition_b=condition_b,
            feature_names=feature_names,
        )

    p_value = float(
        (np.count_nonzero(null_accuracies >= observed) + 1)
        / (null_accuracies.size + 1)
    )
    result = {
        "method": "leave-one-circuit-pair-out nearest centroid",
        "features": feature_names,
        "n_circuit_pairs": int(paired_seeds.size),
        "accuracy": observed,
        "chance_accuracy": 0.5,
        "paired_label_swap_p": p_value,
        "n_permutations": int(n_permutations),
    }
    with (output_dir / "condition_discriminability.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    return result


def _mean_and_interval(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and normal-approximation 95% CI across circuits."""
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    if matrix.shape[0] < 2:
        return mean, mean, mean
    sem = np.std(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _plot_time_courses(
    signals: dict[str, list[np.ndarray]],
    fs_hz: float,
    labels: dict[str, str],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(9, 4.5))
    for condition_name, condition_signals in signals.items():
        matrix = np.stack(condition_signals)
        mean, lower, upper = _mean_and_interval(matrix)
        time_s = np.arange(mean.size) / fs_hz
        line = plt.plot(time_s, mean, label=labels[condition_name])[0]
        plt.fill_between(
            time_s,
            lower,
            upper,
            alpha=0.2,
            color=line.get_color(),
        )

    plt.xlabel("Time after burn-in (s)")
    plt.ylabel("EEG potential (V)")
    plt.title("Mean EEG by circuit condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eeg_time_course.png", dpi=250)
    plt.close()


def _plot_psd_curves(
    psd_by_condition: dict[str, list[np.ndarray]],
    frequencies_hz: np.ndarray,
    labels: dict[str, str],
    output_dir: Path,
) -> None:
    eps = np.finfo(np.float64).tiny
    mask = (frequencies_hz >= 1.0) & (frequencies_hz <= 80.0)

    plt.figure(figsize=(8, 4.5))
    for condition_name, condition_psds in psd_by_condition.items():
        db_matrix = 10.0 * np.log10(
            np.maximum(np.stack(condition_psds)[:, mask], eps)
        )
        mean, lower, upper = _mean_and_interval(db_matrix)
        line = plt.plot(
            frequencies_hz[mask],
            mean,
            label=labels[condition_name],
        )[0]
        plt.fill_between(
            frequencies_hz[mask],
            lower,
            upper,
            alpha=0.2,
            color=line.get_color(),
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB re V²/Hz)")
    plt.title("EEG power spectral density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eeg_psd_comparison.png", dpi=250)
    plt.close()


def _plot_paired_feature(
    features_df: pd.DataFrame,
    feature_name: str,
    condition_a: str,
    condition_b: str,
    labels: dict[str, str],
    output_dir: Path,
) -> None:
    pivot = features_df.pivot(
        index="circuit_seed",
        columns="condition",
        values=feature_name,
    ).dropna(subset=[condition_a, condition_b])

    if pivot.empty:
        return

    plt.figure(figsize=(5, 4.5))
    for _, row in pivot.iterrows():
        plt.plot(
            [0, 1],
            [row[condition_a], row[condition_b]],
            marker="o",
            alpha=0.55,
        )

    plt.xticks([0, 1], [labels[condition_a], labels[condition_b]])
    plt.ylabel(feature_name.replace("_", " "))
    plt.title(f"Matched-circuit comparison: {feature_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"paired_{feature_name}.png", dpi=250)
    plt.close()


def _plot_effect_sizes(
    statistics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_df = statistics_df.dropna(subset=["cohen_dz"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("cohen_dz")

    plt.figure(figsize=(8, 5.5))
    y = np.arange(plot_df.shape[0])
    plt.scatter(plot_df["cohen_dz"], y)
    plt.axvline(0.0, linewidth=1)
    plt.yticks(y, plot_df["feature"].str.replace("_", " "))
    plt.xlabel("Paired effect size, Cohen's dz (B − A)")
    plt.title("Standardised condition effects")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_effect_sizes.png", dpi=250)
    plt.close()


def _run_condition(
    base_cfg: DictConfig,
    condition_name: str,
    condition_cfg: DictConfig,
    comm: Any,
    size: int,
    rank: int,
    root_dir: Path,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], np.ndarray | None, float | None]:
    """Run all matched circuits for one condition."""
    n_circuits = int(base_cfg.analysis.n_circuits)
    n_steps = int(base_cfg.analysis.n_steps)
    burn_in_steps = int(base_cfg.analysis.burn_in_steps)
    base_seed = int(base_cfg.experiment.seed)

    if not 0 <= burn_in_steps < n_steps:
        raise ValueError(
            "analysis.burn_in_steps must satisfy "
            "0 <= burn_in_steps < analysis.n_steps"
        )

    condition_dir = root_dir / f"condition_{condition_name}"
    if rank == 0:
        condition_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    feature_rows: list[dict[str, Any]] = []
    processed_signals: list[np.ndarray] = []
    psd_curves: list[np.ndarray] = []
    common_frequencies: np.ndarray | None = None
    processed_fs_hz: float | None = None

    for circuit_index in range(n_circuits):
        circuit_seed = base_seed + circuit_index
        run_dir = condition_dir / f"seed_{circuit_seed:05d}"
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        run_cfg = _prepare_run_config(
            base_cfg=base_cfg,
            condition_cfg=condition_cfg,
            circuit_seed=circuit_seed,
            run_dir=run_dir,
        )
        mpi_variables = _fresh_mpi_variables(
            comm=comm,
            size=size,
            rank=rank,
            circuit_seed=circuit_seed,
        )
        _seed_local_generators(circuit_seed=circuit_seed, rank=rank)

        if rank == 0:
            print(
                f"\n[{condition_name}] circuit "
                f"{circuit_index + 1}/{n_circuits}, seed={circuit_seed}"
            )

        no_stimulation_actions = [[0.0, 1.0] for _ in range(n_steps)]
        simulator = str(run_cfg.analysis.get("simulator", "online"))
        firing_rate_summary = {
            "E_firing_rate_hz": float("nan"),
            "I_firing_rate_hz": float("nan"),
            "E_I_firing_rate_ratio": float("nan"),
        }

        if simulator == "online":
            environment = OnlineNeuronEnv(
                run_cfg,
                mpi_variables,
                ENV_SEED=0,
            )
            try:
                outputs = environment.analysis_rollout_online(
                    policy_seq=no_stimulation_actions,
                )
            finally:
                environment.close()
            if rank == 0:
                eeg_chunks = [output["eeg_v"] for output in outputs]
                retained_outputs = outputs[burn_in_steps:]
                for rate_name in firing_rate_summary:
                    firing_rate_summary[rate_name] = float(
                        np.mean(
                            [
                                output["firing_rates"][rate_name]
                                for output in retained_outputs
                            ]
                        )
                    )
            else:
                eeg_chunks = None
        elif simulator == "legacy":
            environment = NeuronEnv(
                run_cfg,
                mpi_variables,
                ENV_SEED=0,
            )
            try:
                _, eeg_chunks = environment.analysis_rollout(
                    policy_seq=no_stimulation_actions,
                    buffer=None,
                    steps=n_steps,
                )
            finally:
                environment.close()
        else:
            raise ValueError("analysis.simulator must be 'online' or 'legacy'.")

        if rank != 0:
            continue

        if eeg_chunks is None or len(eeg_chunks) != n_steps:
            raise RuntimeError(
                f"Expected {n_steps} EEG chunks, received "
                f"{None if eeg_chunks is None else len(eeg_chunks)}"
            )

        retained_chunks = eeg_chunks[burn_in_steps:]
        raw_eeg = np.concatenate(
            [np.asarray(chunk, dtype=np.float64) for chunk in retained_chunks]
        )

        simulator_fs_hz = (
            1.0 / float(run_cfg.env.network.dt)
        ) * 1000.0
        processed_eeg, current_fs_hz = _preprocess_eeg(
            eeg=raw_eeg,
            fs_hz=simulator_fs_hz,
            target_fs_hz=int(run_cfg.analysis.target_fs_hz),
            low_hz=float(run_cfg.analysis.low_hz),
            high_hz=float(run_cfg.analysis.high_hz),
        )
        features, frequencies_hz, psd = _extract_eeg_features(
            eeg=processed_eeg,
            fs_hz=current_fs_hz,
        )

        np.save(run_dir / "eeg_raw_after_burn_in.npy", raw_eeg)
        np.save(run_dir / "eeg_preprocessed.npy", processed_eeg)
        np.save(run_dir / "psd_frequencies_hz.npy", frequencies_hz)
        np.save(run_dir / "psd_v2_per_hz.npy", psd)

        feature_rows.append(
            {
                "condition": condition_name,
                "condition_label": str(condition_cfg.label),
                "circuit_index": circuit_index,
                "circuit_seed": circuit_seed,
                "simulator_fs_hz": simulator_fs_hz,
                "analysis_fs_hz": current_fs_hz,
                **firing_rate_summary,
                **features,
            }
        )
        processed_signals.append(processed_eeg)
        psd_curves.append(psd)

        if common_frequencies is None:
            common_frequencies = frequencies_hz
            processed_fs_hz = current_fs_hz
        elif not np.array_equal(common_frequencies, frequencies_hz):
            raise RuntimeError("PSD frequency grids differ across circuits")

    return (
        feature_rows,
        processed_signals,
        psd_curves,
        common_frequencies,
        processed_fs_hz,
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("\n### Experiment configuration")
        print(OmegaConf.to_yaml(cfg, resolve=True))

    condition_names = list(cfg.analysis.conditions.keys())
    if len(condition_names) != 2:
        raise ValueError(
            "This analysis expects exactly two conditions under "
            "analysis.conditions."
        )
    condition_a, condition_b = condition_names

    # Keep the Hydra run directory intact and replace only this experiment's
    # dedicated A/B-analysis subdirectory.
    experiment_root = Path(to_absolute_path(str(cfg.experiment.dir)))
    root_dir = experiment_root / "ab_eeg_analysis"
    if rank == 0:
        if root_dir.exists():
            shutil.rmtree(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        (root_dir / "analysis").mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    start_time = time.perf_counter()

    all_feature_rows: list[dict[str, Any]] = []
    signals_by_condition: dict[str, list[np.ndarray]] = {}
    psd_by_condition: dict[str, list[np.ndarray]] = {}
    frequencies_hz: np.ndarray | None = None
    analysis_fs_hz: float | None = None
    labels: dict[str, str] = {}

    for condition_name in condition_names:
        condition_cfg = cfg.analysis.conditions[condition_name]
        labels[condition_name] = str(condition_cfg.label)

        (
            feature_rows,
            condition_signals,
            condition_psds,
            condition_frequencies,
            condition_fs_hz,
        ) = _run_condition(
            base_cfg=cfg,
            condition_name=condition_name,
            condition_cfg=condition_cfg,
            comm=comm,
            size=size,
            rank=rank,
            root_dir=root_dir,
        )

        if rank == 0:
            all_feature_rows.extend(feature_rows)
            signals_by_condition[condition_name] = condition_signals
            psd_by_condition[condition_name] = condition_psds

            if frequencies_hz is None:
                frequencies_hz = condition_frequencies
                analysis_fs_hz = condition_fs_hz
            elif not np.array_equal(frequencies_hz, condition_frequencies):
                raise RuntimeError("PSD grids differ between conditions")

    if rank == 0:
        analysis_dir = root_dir / "analysis"
        features_df = pd.DataFrame(all_feature_rows)
        features_df.to_csv(analysis_dir / "circuit_features.csv", index=False)

        # Save condition-level summaries and the reference distribution for A.
        numeric_feature_columns = [
            column
            for column in features_df.select_dtypes(include=[np.number]).columns
            if column
            not in {
                "circuit_index",
                "circuit_seed",
                "simulator_fs_hz",
                "analysis_fs_hz",
            }
        ]
        condition_summary = (
            features_df.groupby("condition")[numeric_feature_columns]
            .agg(["mean", "std"])
        )
        condition_summary.to_csv(analysis_dir / "condition_feature_summary.csv")

        reference_a = (
            features_df[features_df["condition"] == condition_a][
                numeric_feature_columns
            ]
            .agg(["mean", "std"])
            .transpose()
            .rename(columns={"mean": "reference_mean", "std": "reference_std"})
        )
        reference_a.to_csv(analysis_dir / "condition_A_reference.csv")

        if frequencies_hz is not None:
            for condition_name in condition_names:
                psd_matrix = np.stack(psd_by_condition[condition_name])
                np.savez_compressed(
                    analysis_dir / f"mean_psd_{condition_name}.npz",
                    frequencies_hz=frequencies_hz,
                    mean_psd=np.mean(psd_matrix, axis=0),
                    std_psd=np.std(psd_matrix, axis=0, ddof=1),
                )

        statistics_df = _compare_conditions(
            features_df=features_df,
            condition_a=condition_a,
            condition_b=condition_b,
            output_dir=analysis_dir,
            seed=int(cfg.experiment.seed) + 91_337,
            n_bootstrap=int(cfg.analysis.n_bootstrap),
            n_permutations=int(cfg.analysis.n_permutations),
        )
        discriminability = _condition_discriminability(
            features_df=features_df,
            condition_a=condition_a,
            condition_b=condition_b,
            feature_names=list(cfg.analysis.classification_features),
            n_permutations=int(cfg.analysis.classifier_permutations),
            seed=int(cfg.experiment.seed) + 71_119,
            output_dir=analysis_dir,
        )

        if bool(cfg.experiment.plot):
            if frequencies_hz is None or analysis_fs_hz is None:
                raise RuntimeError("Missing EEG analysis outputs")

            _plot_time_courses(
                signals=signals_by_condition,
                fs_hz=analysis_fs_hz,
                labels=labels,
                output_dir=analysis_dir,
            )
            _plot_psd_curves(
                psd_by_condition=psd_by_condition,
                frequencies_hz=frequencies_hz,
                labels=labels,
                output_dir=analysis_dir,
            )
            for feature_name in PLOT_FEATURES:
                _plot_paired_feature(
                    features_df=features_df,
                    feature_name=feature_name,
                    condition_a=condition_a,
                    condition_b=condition_b,
                    labels=labels,
                    output_dir=analysis_dir,
                )
            _plot_effect_sizes(
                statistics_df=statistics_df,
                output_dir=analysis_dir,
            )

        print("\n### Statistical comparison")
        print(
            statistics_df[
                [
                    "feature",
                    "mean_difference_B_minus_A",
                    "cohen_dz",
                    "permutation_p",
                    "fdr_q",
                ]
            ].to_string(index=False)
        )
        print(
            "\nEEG-only held-out condition accuracy: "
            f"{discriminability['accuracy']:.3f} "
            f"(paired-swap p={discriminability['paired_label_swap_p']:.4g})"
        )
        print(f"\nResults saved to: {analysis_dir}")
        print(
            "Total experiment time: "
            f"{(time.perf_counter() - start_time) / 60.0:.2f} minutes"
        )


if __name__ == "__main__":
    main()
