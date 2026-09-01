"""Identify and validate a compact tACS action using observed EEG only.

The hidden B generator is never exposed to the selector. Unstimulated A/B EEG
first defines a frequency-generic spectral target. Disjoint circuit seeds then
select one of three frequencies, followed by one of four phase quadrants.
Finally, the frozen frequency and phase are evaluated at sham, 0.5, and
0.8 V/m on held-out seeds. Spikes and firing rates are used only for hidden
mechanistic checks and figures; they never enter action selection or reward.

The experiment reports ideal forward-model EEG reachability separately from
concurrent-EEG robustness. The latter requires generalization after removing
the selected-frequency bins and superiority to an observation-only sinusoid
that matches both Fourier quadratures of the primary active arm.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.stimulation import (  # noqa: E402
    apply_raised_cosine_block_envelope,
)
from experiments.ballnstick_analysis.run_ballnstick import (  # noqa: E402
    _extract_eeg_features,
    _preprocess_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_eeg_reachability import (  # noqa: E402
    _run_condition,
)
from experiments.ballnstick_analysis.run_ballnstick_entrainment_state import (  # noqa: E402
    _condition_config,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _bootstrap_ci,
    _relative_rate_safe,
    _sign_flip_p,
)


REFERENCE_CONDITIONS = ("A_async", "B_rhythmic_reference")
PRIMARY_ACTION = "A_tacs_primary"
LOWER_ACTION = "A_tacs_lower"
TRANSVERSE_CONTROL = "A_tacs_transverse"
SYNTHETIC_CONTROL = "A_synthetic_complex_match"
VALIDATION_CONDITIONS = (
    "A_async",
    "B_rhythmic_reference",
    LOWER_ACTION,
    PRIMARY_ACTION,
    TRANSVERSE_CONTROL,
)
EPOCHS = ("baseline", "stimulation", "washout")


def _plain(value: Any) -> Any:
    """Return JSON-safe Python scalars and containers."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _validate_design(cfg: DictConfig) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("Hierarchical tACS identification requires online mode.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("A, B, and all tACS arms require inhibition_scale=1.0.")
    frequencies = [
        float(value)
        for value in cfg.analysis.target_discovery.candidate_frequencies_hz
    ]
    phases = [float(value) for value in cfg.analysis.phase_discovery.phases_rad]
    if len(frequencies) < 2 or len(set(frequencies)) != len(frequencies):
        raise ValueError("Candidate frequencies must contain unique alternatives.")
    if any(value <= 0.0 for value in frequencies):
        raise ValueError("Candidate frequencies must be positive.")
    if len(phases) < 2 or len(set(phases)) != len(phases):
        raise ValueError("Phase candidates must contain unique alternatives.")
    maximum = float(cfg.analysis.maximum_field_v_per_m)
    amplitudes = (
        float(cfg.analysis.frequency_discovery.amplitude_v_per_m),
        float(cfg.analysis.phase_discovery.amplitude_v_per_m),
        float(cfg.analysis.validation.lower_amplitude_v_per_m),
        float(cfg.analysis.validation.primary_amplitude_v_per_m),
    )
    if any(value <= 0.0 or value > maximum for value in amplitudes):
        raise ValueError("Every active field amplitude must be in (0, maximum].")
    if not amplitudes[2] < amplitudes[3]:
        raise ValueError("Validation lower amplitude must be below primary.")
    seed_blocks = []
    base = int(cfg.experiment.seed)
    for stage in (
        cfg.analysis.target_discovery,
        cfg.analysis.frequency_discovery,
        cfg.analysis.phase_discovery,
        cfg.analysis.validation,
    ):
        if int(stage.n_seeds) < 1:
            raise ValueError("Every hierarchy stage requires at least one seed.")
        seed_blocks.append(
            {
                base + int(stage.seed_offset) + index
                for index in range(int(stage.n_seeds))
            }
        )
    for left in range(len(seed_blocks)):
        for right in range(left + 1, len(seed_blocks)):
            if seed_blocks[left].intersection(seed_blocks[right]):
                raise ValueError("All hierarchy stages require disjoint seeds.")


def _seeds(cfg: DictConfig, stage_name: str) -> list[int]:
    stage = cfg.analysis[stage_name]
    base = int(cfg.experiment.seed) + int(stage.seed_offset)
    return [base + index for index in range(int(stage.n_seeds))]


def _action(
    *,
    identifier: str,
    role: str,
    amplitude_v_per_m: float,
    frequency_hz: float,
    phase_rad: float,
    montage: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "role": role,
        "montage": montage,
        "dc_offset_v_per_m": 0.0,
        "ac_amplitude_v_per_m": float(amplitude_v_per_m),
        "frequency_hz": float(frequency_hz),
        "phase_rad": float(phase_rad),
    }


def _sham_action(cfg: DictConfig, *, identifier: str = "A_async") -> dict[str, Any]:
    # This metadata frequency is not an input to target-frequency selection.
    frequency = float(cfg.analysis.target_discovery.candidate_frequencies_hz[0])
    return _action(
        identifier=identifier,
        role="unstimulated_reference",
        amplitude_v_per_m=0.0,
        frequency_hz=frequency,
        phase_rad=0.0,
        montage=str(cfg.analysis.tacs.axial_montage),
    )


def _active_action(
    cfg: DictConfig,
    *,
    identifier: str,
    role: str,
    amplitude_v_per_m: float,
    frequency_hz: float,
    phase_rad: float,
    montage: str | None = None,
) -> dict[str, Any]:
    return _action(
        identifier=identifier,
        role=role,
        amplitude_v_per_m=amplitude_v_per_m,
        frequency_hz=frequency_hz,
        phase_rad=phase_rad,
        montage=(
            str(cfg.analysis.tacs.axial_montage)
            if montage is None
            else str(montage)
        ),
    )


def _epoch_raw(episode: dict[str, Any], epoch: str) -> np.ndarray:
    return np.asarray(episode["raw_by_epoch"][epoch], dtype=np.float64)


def _process_eeg(
    raw_eeg: np.ndarray,
    *,
    simulator_fs_hz: float,
    cfg: DictConfig,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict[str, float]]:
    processed, fs_hz = _preprocess_eeg(
        raw_eeg,
        fs_hz=simulator_fs_hz,
        target_fs_hz=int(cfg.analysis.target_fs_hz),
        low_hz=float(cfg.analysis.low_hz),
        high_hz=float(cfg.analysis.high_hz),
    )
    features, frequencies_hz, psd = _extract_eeg_features(processed, fs_hz)
    return processed, fs_hz, frequencies_hz, psd, features


def _band_power(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    *,
    center_hz: float,
    half_width_hz: float,
) -> float:
    mask = (
        (frequencies_hz >= float(center_hz) - float(half_width_hz))
        & (frequencies_hz <= float(center_hz) + float(half_width_hz))
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Insufficient PSD bins around {center_hz:g} Hz.")
    return float(np.trapz(psd[mask], frequencies_hz[mask]))


def _fourier_coefficients(
    processed_eeg: np.ndarray,
    *,
    fs_hz: float,
    start_ms: float,
    frequency_hz: float,
) -> tuple[float, float]:
    """Return unnormalized cosine and sine coefficients in EEG volts."""
    signal = np.asarray(processed_eeg, dtype=np.float64).reshape(-1)
    time_s = (
        float(start_ms) / 1000.0
        + (np.arange(signal.size, dtype=np.float64) + 1.0) / float(fs_hz)
    )
    angle = 2.0 * np.pi * float(frequency_hz) * time_s
    return (
        float(2.0 * np.mean(signal * np.cos(angle))),
        float(2.0 * np.mean(signal * np.sin(angle))),
    )


def _eeg_feature_row(
    raw_eeg: np.ndarray,
    *,
    simulator_fs_hz: float,
    start_ms: float,
    selected_frequency_hz: float,
    candidate_frequencies_hz: Iterable[float],
    cfg: DictConfig,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    processed, fs_hz, frequencies_hz, psd, features = _process_eeg(
        raw_eeg, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    half_width = float(cfg.analysis.target_discovery.band_half_width_hz)
    eps = np.finfo(np.float64).tiny
    row = dict(features)
    for frequency in candidate_frequencies_hz:
        token = f"{float(frequency):g}".replace(".", "p")
        power = _band_power(
            frequencies_hz,
            psd,
            center_hz=float(frequency),
            half_width_hz=half_width,
        )
        row[f"band_power_{token}_hz"] = power
        row[f"log10_band_power_{token}_hz"] = float(np.log10(max(power, eps)))
    selected_power = _band_power(
        frequencies_hz,
        psd,
        center_hz=selected_frequency_hz,
        half_width_hz=half_width,
    )
    total = max(float(features["total_power_1_80"]), eps)
    cosine, sine = _fourier_coefficients(
        processed,
        fs_hz=fs_hz,
        start_ms=start_ms,
        frequency_hz=selected_frequency_hz,
    )
    row.update(
        {
            "selected_band_power": selected_power,
            "relative_selected_band_power": selected_power / total,
            "selected_eeg_cosine_v": cosine,
            "selected_eeg_sine_v": sine,
            "selected_eeg_resultant_v": float(np.hypot(cosine, sine)),
            "selected_eeg_phase_rad": float(np.arctan2(sine, cosine)),
        }
    )

    keep = np.abs(frequencies_hz - selected_frequency_hz) > float(
        cfg.analysis.stimulus_exclusion_half_width_hz
    )
    masked = np.where(keep, psd, 0.0)
    total_mask = (frequencies_hz >= 1.0) & (frequencies_hz <= 80.0)
    broad_mask = (frequencies_hz >= 30.0) & (frequencies_hz <= 80.0)
    excluded_total = float(
        np.trapz(masked[total_mask], frequencies_hz[total_mask])
    )
    excluded_broad = float(
        np.trapz(masked[broad_mask], frequencies_hz[broad_mask])
    )
    row.update(
        {
            "log10_total_power_excluding_selected": float(
                np.log10(max(excluded_total, eps))
            ),
            "relative_30_80_power_excluding_selected": (
                excluded_broad / excluded_total if excluded_total > 0.0 else 0.0
            ),
        }
    )
    return row, processed, frequencies_hz, psd


def _fit_centroid_model(
    rows: pd.DataFrame,
    *,
    feature_names: list[str],
) -> dict[str, Any]:
    subset = rows[
        rows["condition_id"].isin(REFERENCE_CONDITIONS)
        & rows["epoch"].eq("stimulation")
    ]
    a = subset[subset.condition_id.eq("A_async")][feature_names].to_numpy(float)
    b = subset[subset.condition_id.eq("B_rhythmic_reference")][
        feature_names
    ].to_numpy(float)
    if a.shape != b.shape or a.shape[0] < 2:
        raise ValueError("Matched A/B discovery rows are required.")
    pooled = np.vstack((a, b))
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0, ddof=1)
    positive = scale[scale > np.finfo(float).eps]
    fallback = float(np.median(positive)) if positive.size else 1.0
    scale = np.where(scale > np.finfo(float).eps, scale, fallback)
    a_z = (a - center) / scale
    b_z = (b - center) / scale
    return {
        "feature_names": feature_names,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "A_centroid": a_z.mean(axis=0).tolist(),
        "B_centroid": b_z.mean(axis=0).tolist(),
    }


def _distances(rows: pd.DataFrame, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    names = model["feature_names"]
    values = rows[names].to_numpy(float)
    z = (values - np.asarray(model["center"])) / np.asarray(model["scale"])
    a = np.asarray(model["A_centroid"])
    b = np.asarray(model["B_centroid"])
    return np.linalg.norm(z - a, axis=1), np.linalg.norm(z - b, axis=1)


def _add_distances(
    rows: pd.DataFrame,
    model: dict[str, Any],
    *,
    prefix: str,
) -> pd.DataFrame:
    result = rows.copy()
    distance_a, distance_b = _distances(result, model)
    result[f"{prefix}_distance_to_A"] = distance_a
    result[f"{prefix}_distance_to_B"] = distance_b
    result[f"{prefix}_predicted_condition"] = np.where(
        distance_b < distance_a, "B", "A"
    )
    return result


def _summary(
    values: np.ndarray,
    *,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    low, high = _bootstrap_ci(
        values,
        rng=rng,
        n_bootstrap=int(cfg.analysis.n_bootstrap),
    )
    return {
        "n_seeds": int(values.size),
        "mean": float(values.mean()),
        "ci_2.5": float(low),
        "ci_97.5": float(high),
        "positive_seed_fraction": float(np.mean(values > 0.0)),
        "paired_sign_flip_p": float(
            _sign_flip_p(
                values,
                rng=rng,
                n_permutations=int(cfg.analysis.n_permutations),
            )
        ),
    }


def _simulate(
    *,
    cfg: DictConfig,
    condition_cfg: DictConfig,
    condition_id: str,
    action: dict[str, Any],
    stimulate: bool,
    seed: int,
    action_index: int,
    output_dir: Path,
    comm,
    size: int,
    rank: int,
) -> dict[str, Any] | None:
    return _run_condition(
        condition_id=condition_id,
        condition_cfg=condition_cfg,
        action=action,
        stimulate=stimulate,
        seed=seed,
        action_index=action_index,
        output_dir=output_dir,
        comm=comm,
        size=size,
        rank=rank,
    )


def _epoch_metadata(
    episode: dict[str, Any],
    *,
    condition_id: str,
    epoch: str,
) -> dict[str, Any]:
    source = next(row for row in episode["epoch_rows"] if row["epoch"] == epoch)
    return {
        "seed": int(source["seed"]),
        "condition_id": condition_id,
        "epoch": epoch,
        "action_id": str(source["action_id"]),
        "montage": str(source["montage"]),
        "ac_amplitude_v_per_m": float(source["ac_amplitude_v_per_m"]),
        "frequency_hz": float(source["frequency_hz"]),
        "phase_rad": float(source["phase_rad"]),
        "E_firing_rate_hz": float(source["E_firing_rate_hz"]),
        "I_firing_rate_hz": float(source["I_firing_rate_hz"]),
        "E_ppc": float(source["E_ppc"]),
        "I_ppc": float(source["I_ppc"]),
        "E_mean_phase_rad": float(source["E_mean_phase_rad"]),
        "I_mean_phase_rad": float(source["I_mean_phase_rad"]),
    }


def _episode_epoch_feature_row(
    episode: dict[str, Any],
    *,
    condition_id: str,
    epoch: str,
    selected_frequency_hz: float,
    candidate_frequencies_hz: list[float],
    cfg: DictConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    outputs = episode["simulation"]["outputs_by_epoch"][epoch]
    features, processed, frequencies, psd = _eeg_feature_row(
        _epoch_raw(episode, epoch),
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        start_ms=float(outputs[0]["t_start_ms"]),
        selected_frequency_hz=selected_frequency_hz,
        candidate_frequencies_hz=candidate_frequencies_hz,
        cfg=cfg,
    )
    return (
        {**_epoch_metadata(episode, condition_id=condition_id, epoch=epoch), **features},
        processed,
        frequencies,
        psd,
    )


def _episode_window_feature_rows(
    episode: dict[str, Any],
    *,
    condition_id: str,
    selected_frequency_hz: float,
    candidate_frequencies_hz: list[float],
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    action = episode["simulation"]["action"]
    simulator_fs_hz = float(episode["simulator_fs_hz"])
    for epoch in EPOCHS:
        outputs = episode["simulation"]["outputs_by_epoch"][epoch]
        for window_index, output in enumerate(outputs):
            features, _, _, _ = _eeg_feature_row(
                np.asarray(output["eeg_v"], dtype=np.float64),
                simulator_fs_hz=simulator_fs_hz,
                start_ms=float(output["t_start_ms"]),
                selected_frequency_hz=selected_frequency_hz,
                candidate_frequencies_hz=candidate_frequencies_hz,
                cfg=cfg,
            )
            rows.append(
                {
                    "seed": int(episode["simulation"]["seed"]),
                    "condition_id": condition_id,
                    "epoch": epoch,
                    "epoch_window_index": int(window_index),
                    "t_start_ms": float(output["t_start_ms"]),
                    "t_stop_ms": float(output["t_stop_ms"]),
                    "ac_amplitude_v_per_m": float(
                        action["ac_amplitude_v_per_m"]
                    ),
                    "frequency_hz": float(action["frequency_hz"]),
                    "phase_rad": float(action.get("phase_rad", 0.0)),
                    **features,
                }
            )
    return rows


def _frequency_feature_names(frequencies_hz: Iterable[float]) -> list[str]:
    return [
        f"log10_band_power_{float(value):g}_hz".replace(".", "p")
        for value in frequencies_hz
    ]


def _select_target_frequency(
    target_rows: pd.DataFrame,
    *,
    candidate_frequencies_hz: list[float],
    spectral_model: dict[str, Any],
) -> tuple[float, pd.DataFrame]:
    """Select an EEG frequency without access to the hidden B configuration."""
    stimulation = target_rows[target_rows.epoch.eq("stimulation")]
    a = stimulation[stimulation.condition_id.eq("A_async")].set_index("seed")
    b = stimulation[
        stimulation.condition_id.eq("B_rhythmic_reference")
    ].set_index("seed")
    common = a.index.intersection(b.index)
    scale = np.asarray(spectral_model["scale"], dtype=float)
    rows = []
    for index, (frequency, feature) in enumerate(
        zip(candidate_frequencies_hz, spectral_model["feature_names"])
    ):
        paired = (b.loc[common, feature] - a.loc[common, feature]).to_numpy(float)
        standardized = paired / float(scale[index])
        rows.append(
            {
                "frequency_hz": float(frequency),
                "feature_name": feature,
                "mean_log10_power_shift": float(paired.mean()),
                "mean_standardized_shift": float(standardized.mean()),
                "positive_seed_fraction": float(np.mean(paired > 0.0)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("frequency_hz")
    positive = frame[frame.mean_standardized_shift > 0.0]
    if positive.empty:
        raise RuntimeError("B has no positive spectral target among candidates.")
    selected = float(
        positive.sort_values(
            ["mean_standardized_shift", "frequency_hz"],
            ascending=[False, True],
        ).iloc[0].frequency_hz
    )
    frame["selected_from_eeg"] = np.isclose(frame.frequency_hz, selected)
    return selected, frame


def _state_improvement_rows(
    rows: pd.DataFrame,
    *,
    model: dict[str, Any],
    prefix: str,
    candidate_column: str,
) -> pd.DataFrame:
    scored = _add_distances(rows, model, prefix=prefix)
    sham = scored[scored[candidate_column].eq("A_async")].set_index("seed")
    output: list[dict[str, Any]] = []
    for _, row in scored[~scored[candidate_column].eq("A_async")].iterrows():
        seed = int(row.seed)
        baseline = float(sham.loc[seed, f"{prefix}_distance_to_B"])
        candidate = float(row[f"{prefix}_distance_to_B"])
        output.append(
            {
                **row.to_dict(),
                "A_distance_to_B": baseline,
                "candidate_distance_to_B": candidate,
                "target_distance_improvement": baseline - candidate,
                "fractional_target_distance_improvement": (
                    1.0 - candidate / baseline if baseline > 0.0 else 0.0
                ),
            }
        )
    return pd.DataFrame(output)


def _selection_summary(
    rows: pd.DataFrame,
    *,
    group_columns: list[str],
    cfg: DictConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    summaries = []
    for values, group in rows.groupby(group_columns, sort=True):
        values_tuple = values if isinstance(values, tuple) else (values,)
        summaries.append(
            {
                **dict(zip(group_columns, values_tuple)),
                **_summary(
                    group.target_distance_improvement.to_numpy(float),
                    cfg=cfg,
                    rng=rng,
                ),
                "mean_fractional_improvement": float(
                    group.fractional_target_distance_improvement.mean()
                ),
                "rate_safe_fraction": float(group.rate_safe.mean()),
            }
        )
    return pd.DataFrame(summaries)


def _select_safe_action(
    summary: pd.DataFrame,
    *,
    sort_columns: list[str],
) -> pd.Series:
    safe = summary[summary.rate_safe_fraction >= 1.0]
    if safe.empty:
        raise RuntimeError("No discovery action was rate safe on every seed.")
    return safe.sort_values(
        ["mean", *sort_columns], ascending=[False, *([True] * len(sort_columns))]
    ).iloc[0]


def _match_complex_observation(
    a_episode: dict[str, Any],
    active_episode: dict[str, Any],
    *,
    selected_frequency_hz: float,
    cfg: DictConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Match the active EEG's two Fourier quadratures without changing neurons."""
    epoch = "stimulation"
    a_raw = _epoch_raw(a_episode, epoch)
    active_raw = _epoch_raw(active_episode, epoch)
    simulator_fs_hz = float(a_episode["simulator_fs_hz"])
    outputs = a_episode["simulation"]["outputs_by_epoch"][epoch]
    time_ms = np.concatenate(
        [np.asarray(output["sample_times_ms"], dtype=float) for output in outputs]
    )
    start_ms = float(outputs[0]["t_start_ms"])
    stop_ms = float(outputs[-1]["t_stop_ms"])
    ramp_ms = float(cfg.analysis.timeline.block_ramp_ms)
    angle = 2.0 * np.pi * float(selected_frequency_hz) * time_ms / 1000.0
    basis_cos = apply_raised_cosine_block_envelope(
        np.cos(angle),
        time_ms=time_ms,
        block_start_ms=start_ms,
        block_stop_ms=stop_ms,
        ramp_ms=ramp_ms,
    )
    basis_sin = apply_raised_cosine_block_envelope(
        np.sin(angle),
        time_ms=time_ms,
        block_start_ms=start_ms,
        block_stop_ms=stop_ms,
        ramp_ms=ramp_ms,
    )

    def coefficients(raw: np.ndarray) -> np.ndarray:
        processed, fs_hz, _, _, _ = _process_eeg(
            raw, simulator_fs_hz=simulator_fs_hz, cfg=cfg
        )
        return np.asarray(
            _fourier_coefficients(
                processed,
                fs_hz=fs_hz,
                start_ms=start_ms,
                frequency_hz=selected_frequency_hz,
            )
        )

    target_delta = coefficients(active_raw) - coefficients(a_raw)
    matrix = np.column_stack((coefficients(basis_cos), coefficients(basis_sin)))
    if np.linalg.cond(matrix) > 1e8:
        raise RuntimeError("Complex observation-control basis is ill-conditioned.")
    weights = np.linalg.solve(matrix, target_delta)
    synthetic = a_raw + weights[0] * basis_cos + weights[1] * basis_sin
    residual = coefficients(synthetic) - coefficients(active_raw)
    return synthetic, {
        "cosine_weight_v": float(weights[0]),
        "sine_weight_v": float(weights[1]),
        "peak_added_v": float(np.max(np.abs(synthetic - a_raw))),
        "cosine_residual_v": float(residual[0]),
        "sine_residual_v": float(residual[1]),
    }


def _synthetic_feature_row(
    a_episode: dict[str, Any],
    active_episode: dict[str, Any],
    *,
    selected_frequency_hz: float,
    candidate_frequencies_hz: list[float],
    cfg: DictConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    raw, match = _match_complex_observation(
        a_episode,
        active_episode,
        selected_frequency_hz=selected_frequency_hz,
        cfg=cfg,
    )
    outputs = a_episode["simulation"]["outputs_by_epoch"]["stimulation"]
    features, processed, frequencies, psd = _eeg_feature_row(
        raw,
        simulator_fs_hz=float(a_episode["simulator_fs_hz"]),
        start_ms=float(outputs[0]["t_start_ms"]),
        selected_frequency_hz=selected_frequency_hz,
        candidate_frequencies_hz=candidate_frequencies_hz,
        cfg=cfg,
    )
    metadata = _epoch_metadata(
        a_episode, condition_id=SYNTHETIC_CONTROL, epoch="stimulation"
    )
    metadata.update(
        {
            "action_id": SYNTHETIC_CONTROL,
            "condition_id": SYNTHETIC_CONTROL,
            "frequency_hz": float(selected_frequency_hz),
            **match,
            **features,
        }
    )
    return metadata, processed, frequencies, psd


def _hidden_validation_rows(
    epoch_rows: pd.DataFrame,
    episodes: dict[int, dict[str, dict[str, Any]]],
    *,
    cfg: DictConfig,
) -> pd.DataFrame:
    indexed = {
        condition: epoch_rows[
            epoch_rows.condition_id.eq(condition)
        ].set_index(["seed", "epoch"])
        for condition in VALIDATION_CONDITIONS
    }
    rows = []
    residual_fraction = float(
        cfg.analysis.criteria.maximum_washout_residual_fraction
    )
    for seed, seed_episodes in episodes.items():
        for condition in (LOWER_ACTION, PRIMARY_ACTION, TRANSVERSE_CONTROL):
            a = indexed["A_async"]
            active = indexed[condition]
            ppc_gain = float(
                (active.loc[(seed, "stimulation"), "E_ppc"]
                 - active.loc[(seed, "baseline"), "E_ppc"])
                - (a.loc[(seed, "stimulation"), "E_ppc"]
                   - a.loc[(seed, "baseline"), "E_ppc"])
            )
            washout_gain = float(
                (active.loc[(seed, "washout"), "E_ppc"]
                 - active.loc[(seed, "baseline"), "E_ppc"])
                - (a.loc[(seed, "washout"), "E_ppc"]
                   - a.loc[(seed, "baseline"), "E_ppc"])
            )
            active_stim = active.loc[(seed, "stimulation")]
            a_stim = a.loc[(seed, "stimulation")]
            rows.append(
                {
                    "seed": int(seed),
                    "condition_id": condition,
                    "E_ppc_gain_difference_in_differences": ppc_gain,
                    "E_rate_change_vs_A_hz": float(
                        active_stim.E_firing_rate_hz - a_stim.E_firing_rate_hz
                    ),
                    "I_rate_change_vs_A_hz": float(
                        active_stim.I_firing_rate_hz - a_stim.I_firing_rate_hz
                    ),
                    "rate_safe": bool(
                        _relative_rate_safe(active_stim, a_stim, cfg)
                    ),
                    "washout_recovered": bool(
                        ppc_gain > 0.0
                        and abs(washout_gain)
                        <= residual_fraction
                        * max(abs(ppc_gain), np.finfo(float).eps)
                    ),
                    "baseline_relative_rms_error_vs_A": float(
                        _relative_rms_error(
                            _epoch_raw(seed_episodes["A_async"], "baseline"),
                            _epoch_raw(seed_episodes[condition], "baseline"),
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _validation_reachability(
    scored_rows: pd.DataFrame,
    *,
    prefix: str,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stimulation = scored_rows[scored_rows.epoch.eq("stimulation")]
    a = stimulation[stimulation.condition_id.eq("A_async")].set_index("seed")
    output = []
    for condition in (
        "B_rhythmic_reference",
        LOWER_ACTION,
        PRIMARY_ACTION,
        TRANSVERSE_CONTROL,
        SYNTHETIC_CONTROL,
    ):
        candidate = stimulation[stimulation.condition_id.eq(condition)].set_index(
            "seed"
        )
        for seed in a.index.intersection(candidate.index):
            baseline = float(a.loc[seed, f"{prefix}_distance_to_B"])
            distance = float(candidate.loc[seed, f"{prefix}_distance_to_B"])
            output.append(
                {
                    "seed": int(seed),
                    "condition_id": condition,
                    "feature_space": prefix,
                    "A_distance_to_B": baseline,
                    "candidate_distance_to_B": distance,
                    "target_distance_improvement": baseline - distance,
                    "fractional_target_distance_improvement": (
                        1.0 - distance / baseline if baseline > 0.0 else 0.0
                    ),
                }
            )
    seed_frame = pd.DataFrame(output)
    summaries = []
    for condition, group in seed_frame.groupby("condition_id", sort=False):
        summaries.append(
            {
                "condition_id": condition,
                "feature_space": prefix,
                **_summary(
                    group.target_distance_improvement.to_numpy(float),
                    cfg=cfg,
                    rng=rng,
                ),
                "mean_fractional_improvement": float(
                    group.fractional_target_distance_improvement.mean()
                ),
            }
        )
    return seed_frame, pd.DataFrame(summaries)


def _classification_accuracy(
    rows: pd.DataFrame,
    *,
    prefix: str,
    epoch: str = "stimulation",
) -> float:
    reference = rows[
        rows.epoch.eq(epoch) & rows.condition_id.isin(REFERENCE_CONDITIONS)
    ]
    expected = np.where(
        reference.condition_id.eq("B_rhythmic_reference"), "B", "A"
    )
    return float(np.mean(reference[f"{prefix}_predicted_condition"] == expected))


def _representative_spike_rows(
    episodes: dict[str, dict[str, Any]],
    *,
    selected_frequency_hz: float,
) -> pd.DataFrame:
    rows = []
    for condition in ("A_async", "B_rhythmic_reference", PRIMARY_ACTION):
        outputs = episodes[condition]["simulation"]["outputs_by_epoch"][
            "stimulation"
        ]
        # Use the middle second, away from both block ramps.
        output = outputs[len(outputs) // 2]
        start = float(output["t_start_ms"])
        for gid, times in output["spikes"]["E"]["per_cell"].items():
            for spike_ms in np.asarray(times, dtype=float):
                rows.append(
                    {
                        "condition_id": condition,
                        "gid": int(gid),
                        "relative_time_ms": float(spike_ms - start),
                        "phase_rad": float(
                            np.mod(
                                2.0
                                * np.pi
                                * selected_frequency_hz
                                * spike_ms
                                / 1000.0,
                                2.0 * np.pi,
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    mean = np.mean(array, axis=0)
    sem = (
        np.std(array, axis=0, ddof=1) / np.sqrt(array.shape[0])
        if array.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


def _plot_psd(
    psd_rows: pd.DataFrame,
    *,
    selected_frequency_hz: float,
    root: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {
        "A_async": "#2878B5",
        "B_rhythmic_reference": "#2CA02C",
        PRIMARY_ACTION: "#E67E22",
        SYNTHETIC_CONTROL: "#777777",
    }
    labels = {
        "A_async": "A (sham)",
        "B_rhythmic_reference": "B reference",
        PRIMARY_ACTION: "A + selected tACS",
        SYNTHETIC_CONTROL: "A + observation sine",
    }
    for condition in colors:
        group = psd_rows[psd_rows.condition_id.eq(condition)]
        pivot = group.pivot(index="seed", columns="frequency_hz", values="psd_v2_hz")
        frequencies = pivot.columns.to_numpy(float)
        mean, sem = _mean_sem(10.0 * np.log10(np.maximum(pivot.to_numpy(), 1e-30)))
        for axis, limits in zip(axes, ((1.0, 100.0), (30.0, 90.0))):
            mask = (frequencies >= limits[0]) & (frequencies <= limits[1])
            axis.plot(
                frequencies[mask], mean[mask], color=colors[condition],
                label=labels[condition], linewidth=1.8,
            )
            axis.fill_between(
                frequencies[mask], mean[mask] - sem[mask], mean[mask] + sem[mask],
                color=colors[condition], alpha=0.16,
            )
            axis.axvline(selected_frequency_hz, color="black", linestyle="--", linewidth=0.9)
            axis.set_xlim(*limits)
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("PSD (dB V²/Hz)")
    axes[0].set_title("Validation EEG spectra")
    axes[1].set_title("Driven-band detail")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_validation_psd.png", dpi=250)
    plt.close(figure)


def _plot_selection(
    frequency_rows: pd.DataFrame,
    phase_rows: pd.DataFrame,
    *,
    selected_frequency_hz: float,
    selected_phase_rad: float,
    root: Path,
) -> None:
    figure = plt.figure(figsize=(10.0, 4.3))
    frequency_axis = figure.add_subplot(1, 2, 1)
    for _, group in frequency_rows.groupby("seed"):
        frequency_axis.plot(
            group.frequency_hz,
            group.target_distance_improvement,
            color="0.75",
            marker="o",
            linewidth=1,
        )
    summary = frequency_rows.groupby("frequency_hz").target_distance_improvement.agg(
        ["mean", "sem"]
    )
    frequency_axis.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["sem"].fillna(0.0),
        color="#E67E22",
        marker="o",
        linewidth=2,
        capsize=3,
    )
    frequency_axis.axhline(0.0, color="black", linewidth=0.8)
    frequency_axis.axvline(selected_frequency_hz, color="black", linestyle="--")
    frequency_axis.set_xlabel("tACS frequency (Hz)")
    frequency_axis.set_ylabel("EEG target-distance improvement")
    frequency_axis.set_title("EEG-only frequency screen")

    phase_axis = figure.add_subplot(1, 2, 2)
    for _, group in phase_rows.groupby("seed"):
        group = group.sort_values("phase_rad")
        phase_axis.plot(
            np.degrees(group.phase_rad),
            group.target_distance_improvement,
            color="0.75",
            marker="o",
            linewidth=1,
        )
    phase_summary = phase_rows.groupby("phase_rad").target_distance_improvement.agg(
        ["mean", "sem"]
    ).sort_index()
    phase_degrees = np.degrees(phase_summary.index.to_numpy(float))
    phase_axis.errorbar(
        phase_degrees,
        phase_summary["mean"],
        yerr=phase_summary["sem"].fillna(0.0),
        color="#E67E22",
        marker="o",
        linewidth=2,
        capsize=3,
    )
    phase_axis.plot(
        [np.degrees(selected_phase_rad)],
        [phase_summary.loc[selected_phase_rad, "mean"]],
        "k*",
        markersize=11,
    )
    phase_axis.axhline(0.0, color="black", linewidth=0.8)
    phase_axis.set_xticks([0, 90, 180, 270])
    phase_axis.set_xlabel("tACS phase at block onset (degrees)")
    phase_axis.set_ylabel("EEG target-distance improvement")
    phase_axis.set_title("EEG-only phase screen")
    figure.tight_layout()
    figure.savefig(root / "figure_02_hierarchical_selection.png", dpi=250)
    plt.close(figure)


def _plot_validation_effects(
    reachability: pd.DataFrame,
    hidden: pd.DataFrame,
    epoch_rows: pd.DataFrame,
    *,
    root: Path,
) -> None:
    condition_order = ["A_async", LOWER_ACTION, PRIMARY_ACTION]
    stimulation = epoch_rows[epoch_rows.epoch.eq("stimulation")]
    amplitudes = [
        float(
            stimulation[stimulation.condition_id.eq(condition)]
            .ac_amplitude_v_per_m.iloc[0]
        )
        for condition in condition_order
    ]
    labels = ["Sham", f"{amplitudes[1]:g}", f"{amplitudes[2]:g}"]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.0))

    effects = {"A_async": np.zeros(epoch_rows.seed.nunique())}
    for condition in (LOWER_ACTION, PRIMARY_ACTION):
        effects[condition] = reachability[
            reachability.condition_id.eq(condition)
        ].sort_values("seed").target_distance_improvement.to_numpy(float)
    for seed_index in range(len(effects["A_async"])):
        axes[0].plot(
            amplitudes,
            [effects[name][seed_index] for name in condition_order],
            color="0.75",
            linewidth=0.9,
        )
    means = [np.mean(effects[name]) for name in condition_order]
    sems = [
        np.std(effects[name], ddof=1) / np.sqrt(len(effects[name]))
        if len(effects[name]) > 1
        else 0.0
        for name in condition_order
    ]
    axes[0].errorbar(amplitudes, means, yerr=sems, color="#E67E22", marker="o", capsize=3)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Field amplitude (V/m)")
    axes[0].set_ylabel("EEG target-distance improvement")
    axes[0].set_title("Held-out EEG movement")

    ppc_values = {"A_async": np.zeros(epoch_rows.seed.nunique())}
    for condition in (LOWER_ACTION, PRIMARY_ACTION):
        ppc_values[condition] = hidden[
            hidden.condition_id.eq(condition)
        ].sort_values("seed").E_ppc_gain_difference_in_differences.to_numpy(float)
    for seed_index in range(len(ppc_values["A_async"])):
        axes[1].plot(
            amplitudes,
            [ppc_values[name][seed_index] for name in condition_order],
            color="0.75",
            linewidth=0.9,
        )
    axes[1].plot(
        amplitudes,
        [np.mean(ppc_values[name]) for name in condition_order],
        color="#E67E22",
        marker="o",
        linewidth=2,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Field amplitude (V/m)")
    axes[1].set_ylabel("E-population PPC gain")
    axes[1].set_title("Hidden spike entrainment")

    x = np.arange(3)
    width = 0.34
    for offset, population, color in (
        (-width / 2, "E", "#2878B5"),
        (width / 2, "I", "#C44E52"),
    ):
        values = [
            stimulation[stimulation.condition_id.eq(condition)][
                f"{population}_firing_rate_hz"
            ].to_numpy(float)
            for condition in condition_order
        ]
        axes[2].bar(
            x + offset,
            [np.mean(value) for value in values],
            width,
            yerr=[
                np.std(value, ddof=1) / np.sqrt(len(value)) if len(value) > 1 else 0.0
                for value in values
            ],
            label=population,
            color=color,
            alpha=0.85,
            capsize=3,
        )
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel("Field amplitude (V/m)")
    axes[2].set_ylabel("Firing rate (Hz)")
    axes[2].set_title("Population rate guardrail")
    axes[2].legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(root / "figure_03_validation_dose_response.png", dpi=250)
    plt.close(figure)


def _plot_spike_timing(
    spike_rows: pd.DataFrame,
    epoch_rows: pd.DataFrame,
    *,
    representative_seed: int,
    selected_frequency_hz: float,
    selected_phase_rad: float,
    window_ms: float,
    root: Path,
) -> None:
    conditions = ["A_async", PRIMARY_ACTION]
    colors = {"A_async": "#2878B5", PRIMARY_ACTION: "#E67E22"}
    labels = {"A_async": "A sham", PRIMARY_ACTION: "A + selected tACS"}
    figure = plt.figure(figsize=(10.5, 6.4))
    for column, condition in enumerate(conditions):
        raster = figure.add_subplot(2, 3, 1 + column)
        subset = spike_rows[spike_rows.condition_id.eq(condition)]
        raster.scatter(
            subset.relative_time_ms,
            subset.gid,
            s=5,
            color=colors[condition],
            linewidths=0,
        )
        if condition == PRIMARY_ACTION:
            time_ms = np.linspace(0.0, window_ms, 1000)
            wave = np.sin(
                2.0 * np.pi * selected_frequency_hz * time_ms / 1000.0
                + selected_phase_rad
            )
            upper = subset.gid.max() + 1 if not subset.empty else 32
            raster.plot(time_ms, upper + 1.2 + wave, color="black", linewidth=0.7)
        raster.set_xlim(0.0, window_ms)
        raster.set_xlabel("Time in middle stimulation window (ms)")
        raster.set_ylabel("E-cell GID")
        raster.set_title(labels[condition])

        polar = figure.add_subplot(2, 3, 4 + column, projection="polar")
        phases = subset.phase_rad.to_numpy(float)
        bins = np.linspace(0.0, 2.0 * np.pi, 17)
        counts, edges = np.histogram(phases, bins=bins)
        polar.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=colors[condition],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.4,
        )
        if phases.size:
            vector = np.mean(np.exp(1j * phases))
            polar.annotate(
                "",
                xy=(np.angle(vector), np.max(counts) * np.abs(vector)),
                xytext=(0.0, 0.0),
                arrowprops=dict(color="black", width=1.0, headwidth=5),
            )
        polar.set_title(f"{labels[condition]} spike phase", pad=15)

    rates = figure.add_subplot(1, 3, 3)
    stimulation = epoch_rows[
        epoch_rows.seed.eq(representative_seed)
        & epoch_rows.epoch.eq("stimulation")
        & epoch_rows.condition_id.isin(conditions)
    ].set_index("condition_id")
    x = np.arange(2)
    width = 0.34
    rates.bar(
        x - width / 2,
        [stimulation.loc[name, "E_firing_rate_hz"] for name in conditions],
        width,
        color="#2878B5",
        label="E",
    )
    rates.bar(
        x + width / 2,
        [stimulation.loc[name, "I_firing_rate_hz"] for name in conditions],
        width,
        color="#C44E52",
        label="I",
    )
    rates.set_xticks(x, [labels[name] for name in conditions], rotation=20)
    rates.set_ylabel("Firing rate (Hz)")
    rates.set_title("Representative seed rates")
    rates.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_04_representative_spike_timing.png", dpi=250)
    plt.close(figure)


def _plot_time_course(window_rows: pd.DataFrame, *, root: Path) -> None:
    conditions = ["A_async", "B_rhythmic_reference", PRIMARY_ACTION]
    colors = {
        "A_async": "#2878B5",
        "B_rhythmic_reference": "#2CA02C",
        PRIMARY_ACTION: "#E67E22",
    }
    labels = {
        "A_async": "A sham",
        "B_rhythmic_reference": "B reference",
        PRIMARY_ACTION: "A + selected tACS",
    }
    rows = window_rows[window_rows.condition_id.isin(conditions)].copy()
    n_stimulation_windows = int(
        rows[rows.epoch.eq("stimulation")].epoch_window_index.max()
    ) + 1
    order = {
        "baseline": 0,
        "stimulation": 1,
        "washout": 1 + n_stimulation_windows,
    }
    rows["plot_window"] = [
        order[epoch] + int(index)
        for epoch, index in zip(rows.epoch, rows.epoch_window_index)
    ]
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    for condition in conditions:
        group = rows[rows.condition_id.eq(condition)]
        summary = group.groupby("plot_window").full_distance_to_B.agg(["mean", "sem"])
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["sem"].fillna(0.0),
            marker="o",
            color=colors[condition],
            label=labels[condition],
            capsize=3,
        )
    axis.axvspan(
        0.5,
        n_stimulation_windows + 0.5,
        color="#E67E22",
        alpha=0.08,
        label="tACS block",
    )
    tick_labels = ["Base"] + [
        f"Stim {index + 1}" for index in range(n_stimulation_windows)
    ] + ["Wash"]
    axis.set_xticks(range(len(tick_labels)), tick_labels)
    axis.set_ylabel("EEG distance to frozen B centroid")
    axis.set_title("Online window-by-window state trajectory")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_05_online_state_trajectory.png", dpi=250)
    plt.close(figure)


def _plot_artifact_control(
    full_reach: pd.DataFrame,
    excluded_reach: pd.DataFrame,
    *,
    root: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), sharey=False)
    for axis, frame, title in (
        (axes[0], full_reach, "Full ideal EEG state"),
        (axes[1], excluded_reach, "Selected-frequency bins removed"),
    ):
        real = frame[frame.condition_id.eq(PRIMARY_ACTION)].sort_values("seed")
        synthetic = frame[frame.condition_id.eq(SYNTHETIC_CONTROL)].sort_values("seed")
        x = np.arange(len(real))
        for index in range(len(real)):
            axis.plot(
                [0, 1],
                [
                    synthetic.iloc[index].target_distance_improvement,
                    real.iloc[index].target_distance_improvement,
                ],
                color="0.75",
                linewidth=1,
            )
        axis.scatter(
            np.zeros(len(synthetic)),
            synthetic.target_distance_improvement,
            color="#777777",
            label="Observation sine",
        )
        axis.scatter(
            np.ones(len(real)),
            real.target_distance_improvement,
            color="#E67E22",
            label="Real tACS",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks([0, 1], ["Observation\nsine", "Real\ntACS"])
        axis.set_ylabel("EEG target-distance improvement")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(root / "figure_06_observation_control.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate_design(cfg)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "hierarchical_tacs"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### Hierarchical EEG-only BallAndStick tACS experiment")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    rng = np.random.default_rng(int(cfg.experiment.seed) + 1_400_003)

    candidates = [
        float(value)
        for value in cfg.analysis.target_discovery.candidate_frequencies_hz
    ]
    placeholder_frequency = candidates[0]
    asynchronous_cfg = _condition_config(cfg, modulation_depth=0.0)
    reference_cfg = _condition_config(
        cfg, modulation_depth=float(cfg.analysis.reference.modulation_depth)
    )
    sham = _sham_action(cfg)

    # Stage 1: only unstimulated A and B are visible. Fit a generic spectral
    # target, then estimate its leading frequency from EEG without consulting
    # the hidden stochastic-generator configuration.
    target_episodes: dict[int, dict[str, dict[str, Any]]] = {}
    target_initial_rows: list[dict[str, Any]] = []
    for seed in _seeds(cfg, "target_discovery"):
        if rank == 0:
            target_episodes[seed] = {}
        for action_index, condition in enumerate(REFERENCE_CONDITIONS):
            if rank == 0:
                print(f"target discovery seed={seed}, condition={condition}")
            condition_action = dict(sham)
            condition_action["id"] = condition
            episode = _simulate(
                cfg=cfg,
                condition_cfg=(
                    asynchronous_cfg if condition == "A_async" else reference_cfg
                ),
                condition_id=condition,
                action=condition_action,
                stimulate=False,
                seed=seed,
                action_index=action_index,
                output_dir=root / "target_discovery" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                target_episodes[seed][condition] = episode
                row, _, _, _ = _episode_epoch_feature_row(
                    episode,
                    condition_id=condition,
                    epoch="stimulation",
                    selected_frequency_hz=placeholder_frequency,
                    candidate_frequencies_hz=candidates,
                    cfg=cfg,
                )
                target_initial_rows.append(row)

    if rank == 0:
        target_initial = pd.DataFrame(target_initial_rows)
        spectral_model = _fit_centroid_model(
            target_initial,
            feature_names=_frequency_feature_names(candidates),
        )
        target_peak_hz, target_frequency_table = _select_target_frequency(
            target_initial,
            candidate_frequencies_hz=candidates,
            spectral_model=spectral_model,
        )
        target_frequency_table.to_csv(
            root / "target_frequency_from_unstimulated_eeg.csv", index=False
        )
    else:
        spectral_model = None
        target_peak_hz = None
    spectral_model = comm.bcast(spectral_model, root=0)
    target_peak_hz = float(comm.bcast(target_peak_hz, root=0))

    # Stage 2: screen only three frequencies with phase-invariant spectral
    # distance. Spikes/rates are retained solely as safety/mechanistic output.
    frequency_stage_rows: list[dict[str, Any]] = []
    frequency_amplitude = float(
        cfg.analysis.frequency_discovery.amplitude_v_per_m
    )
    frequency_phase = float(cfg.analysis.frequency_discovery.phase_rad)
    for seed in _seeds(cfg, "frequency_discovery"):
        seed_rows: list[dict[str, Any]] = []
        if rank == 0:
            print(f"frequency discovery seed={seed}, condition=A_async")
        a_episode = _simulate(
            cfg=cfg,
            condition_cfg=asynchronous_cfg,
            condition_id="A_async",
            action=sham,
            stimulate=False,
            seed=seed,
            action_index=0,
            output_dir=root / "frequency_discovery" / "A_async" / f"seed_{seed}",
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            a_row, _, _, _ = _episode_epoch_feature_row(
                a_episode,
                condition_id="A_async",
                epoch="stimulation",
                selected_frequency_hz=target_peak_hz,
                candidate_frequencies_hz=candidates,
                cfg=cfg,
            )
            a_row["rate_safe"] = True
            seed_rows.append(a_row)
        for action_index, frequency in enumerate(candidates, start=1):
            condition = f"tacs_f{frequency:g}".replace(".", "p")
            action = _active_action(
                cfg,
                identifier=condition,
                role="frequency_discovery",
                amplitude_v_per_m=frequency_amplitude,
                frequency_hz=frequency,
                phase_rad=frequency_phase,
            )
            if rank == 0:
                print(f"frequency discovery seed={seed}, action={condition}")
            episode = _simulate(
                cfg=cfg,
                condition_cfg=asynchronous_cfg,
                condition_id=condition,
                action=action,
                stimulate=True,
                seed=seed,
                action_index=action_index,
                output_dir=root / "frequency_discovery" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                row, _, _, _ = _episode_epoch_feature_row(
                    episode,
                    condition_id=condition,
                    epoch="stimulation",
                    selected_frequency_hz=target_peak_hz,
                    candidate_frequencies_hz=candidates,
                    cfg=cfg,
                )
                row["rate_safe"] = bool(
                    _relative_rate_safe(pd.Series(row), pd.Series(a_row), cfg)
                )
                seed_rows.append(row)
        if rank == 0:
            frequency_stage_rows.extend(seed_rows)

    if rank == 0:
        frequency_stage = pd.DataFrame(frequency_stage_rows)
        frequency_effects = _state_improvement_rows(
            frequency_stage,
            model=spectral_model,
            prefix="spectral",
            candidate_column="condition_id",
        )
        frequency_summary = _selection_summary(
            frequency_effects,
            group_columns=["frequency_hz"],
            cfg=cfg,
            rng=rng,
        )
        selected_frequency_row = _select_safe_action(
            frequency_summary, sort_columns=["frequency_hz"]
        )
        selected_frequency_hz = float(selected_frequency_row.frequency_hz)
        frequency_effects.to_csv(root / "frequency_screen_seed_metrics.csv", index=False)
        frequency_summary.to_csv(root / "frequency_screen_summary.csv", index=False)
    else:
        frequency_effects = frequency_summary = None
        selected_frequency_hz = None
    selected_frequency_hz = float(comm.bcast(selected_frequency_hz, root=0))

    # Recompute target features at the EEG-selected action frequency, then
    # freeze both full and fundamental-excluded A/B centroid models.
    if rank == 0:
        target_rows: list[dict[str, Any]] = []
        for seed, seed_episodes in target_episodes.items():
            for condition, episode in seed_episodes.items():
                row, _, _, _ = _episode_epoch_feature_row(
                    episode,
                    condition_id=condition,
                    epoch="stimulation",
                    selected_frequency_hz=selected_frequency_hz,
                    candidate_frequencies_hz=candidates,
                    cfg=cfg,
                )
                target_rows.append(row)
        target_frame = pd.DataFrame(target_rows)
        full_feature_names = [
            "log10_total_power_1_80",
            "relative_selected_band_power",
            "selected_eeg_cosine_v",
            "selected_eeg_sine_v",
        ]
        excluded_feature_names = [
            "log10_total_power_excluding_selected",
            "relative_30_80_power_excluding_selected",
        ]
        full_model = _fit_centroid_model(
            target_frame, feature_names=full_feature_names
        )
        excluded_model = _fit_centroid_model(
            target_frame, feature_names=excluded_feature_names
        )
        target_frame = _add_distances(target_frame, full_model, prefix="full")
        target_frame = _add_distances(
            target_frame, excluded_model, prefix="excluded"
        )
        target_frame.to_csv(root / "target_discovery_eeg_metrics.csv", index=False)
    else:
        full_model = excluded_model = None
    full_model = comm.bcast(full_model, root=0)
    excluded_model = comm.bcast(excluded_model, root=0)

    # Stage 3: screen four phase quadrants at only the selected frequency.
    phase_stage_rows: list[dict[str, Any]] = []
    phase_amplitude = float(cfg.analysis.phase_discovery.amplitude_v_per_m)
    phases = [float(value) for value in cfg.analysis.phase_discovery.phases_rad]
    for seed in _seeds(cfg, "phase_discovery"):
        if rank == 0:
            print(f"phase discovery seed={seed}, condition=A_async")
        a_episode = _simulate(
            cfg=cfg,
            condition_cfg=asynchronous_cfg,
            condition_id="A_async",
            action=sham,
            stimulate=False,
            seed=seed,
            action_index=0,
            output_dir=root / "phase_discovery" / "A_async" / f"seed_{seed}",
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            a_row, _, _, _ = _episode_epoch_feature_row(
                a_episode,
                condition_id="A_async",
                epoch="stimulation",
                selected_frequency_hz=selected_frequency_hz,
                candidate_frequencies_hz=candidates,
                cfg=cfg,
            )
            a_row["rate_safe"] = True
            phase_stage_rows.append(a_row)
        for action_index, phase in enumerate(phases, start=1):
            condition = f"tacs_phase_{action_index - 1}"
            action = _active_action(
                cfg,
                identifier=condition,
                role="phase_discovery",
                amplitude_v_per_m=phase_amplitude,
                frequency_hz=selected_frequency_hz,
                phase_rad=phase,
            )
            if rank == 0:
                print(f"phase discovery seed={seed}, phase={phase:g} rad")
            episode = _simulate(
                cfg=cfg,
                condition_cfg=asynchronous_cfg,
                condition_id=condition,
                action=action,
                stimulate=True,
                seed=seed,
                action_index=action_index,
                output_dir=root / "phase_discovery" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                row, _, _, _ = _episode_epoch_feature_row(
                    episode,
                    condition_id=condition,
                    epoch="stimulation",
                    selected_frequency_hz=selected_frequency_hz,
                    candidate_frequencies_hz=candidates,
                    cfg=cfg,
                )
                row["rate_safe"] = bool(
                    _relative_rate_safe(pd.Series(row), pd.Series(a_row), cfg)
                )
                phase_stage_rows.append(row)

    if rank == 0:
        phase_stage = pd.DataFrame(phase_stage_rows)
        phase_effects = _state_improvement_rows(
            phase_stage,
            model=full_model,
            prefix="full",
            candidate_column="condition_id",
        )
        phase_summary = _selection_summary(
            phase_effects,
            group_columns=["phase_rad"],
            cfg=cfg,
            rng=rng,
        )
        selected_phase_row = _select_safe_action(
            phase_summary, sort_columns=["phase_rad"]
        )
        selected_phase_rad = float(selected_phase_row.phase_rad)
        phase_effects.to_csv(root / "phase_screen_seed_metrics.csv", index=False)
        phase_summary.to_csv(root / "phase_screen_summary.csv", index=False)
    else:
        phase_effects = phase_summary = None
        selected_phase_rad = None
    selected_phase_rad = float(comm.bcast(selected_phase_rad, root=0))

    frozen_protocol = {
        "frequency_hz": selected_frequency_hz,
        "phase_rad": selected_phase_rad,
        "phase_degrees": float(np.degrees(selected_phase_rad)),
        "policy_amplitudes_v_per_m": [
            0.0,
            float(cfg.analysis.validation.lower_amplitude_v_per_m),
            float(cfg.analysis.validation.primary_amplitude_v_per_m),
        ],
        "montage": str(cfg.analysis.tacs.axial_montage),
        "selection_inputs": "EEG only",
        "target_frequency_estimated_from_eeg_hz": target_peak_hz,
    }
    if rank == 0:
        with (root / "frozen_hierarchical_protocol.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    **frozen_protocol,
                    "spectral_target_model": spectral_model,
                    "full_target_model": full_model,
                    "excluded_target_model": excluded_model,
                },
                handle,
                indent=2,
            )

    # Stage 4: held-out validation. No selection parameter changes below here.
    validation_actions = {
        "A_async": _action(
            identifier="A_async",
            role="sham_policy_action",
            amplitude_v_per_m=0.0,
            frequency_hz=selected_frequency_hz,
            phase_rad=selected_phase_rad,
            montage=str(cfg.analysis.tacs.axial_montage),
        ),
        "B_rhythmic_reference": _action(
            identifier="B_rhythmic_reference",
            role="unstimulated_target",
            amplitude_v_per_m=0.0,
            frequency_hz=selected_frequency_hz,
            phase_rad=selected_phase_rad,
            montage=str(cfg.analysis.tacs.axial_montage),
        ),
        LOWER_ACTION: _active_action(
            cfg,
            identifier=LOWER_ACTION,
            role="lower_policy_action",
            amplitude_v_per_m=float(
                cfg.analysis.validation.lower_amplitude_v_per_m
            ),
            frequency_hz=selected_frequency_hz,
            phase_rad=selected_phase_rad,
        ),
        PRIMARY_ACTION: _active_action(
            cfg,
            identifier=PRIMARY_ACTION,
            role="primary_policy_action",
            amplitude_v_per_m=float(
                cfg.analysis.validation.primary_amplitude_v_per_m
            ),
            frequency_hz=selected_frequency_hz,
            phase_rad=selected_phase_rad,
        ),
        TRANSVERSE_CONTROL: _active_action(
            cfg,
            identifier=TRANSVERSE_CONTROL,
            role="orientation_control",
            amplitude_v_per_m=float(
                cfg.analysis.validation.primary_amplitude_v_per_m
            ),
            frequency_hz=selected_frequency_hz,
            phase_rad=selected_phase_rad,
            montage=str(cfg.analysis.tacs.transverse_montage),
        ),
    }

    validation_epoch_rows: list[dict[str, Any]] = []
    validation_window_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    validation_episodes: dict[int, dict[str, dict[str, Any]]] = {}
    representative_spikes = None
    validation_seeds = _seeds(cfg, "validation")
    for seed in validation_seeds:
        seed_episodes: dict[str, dict[str, Any]] = {}
        for action_index, condition in enumerate(VALIDATION_CONDITIONS):
            if rank == 0:
                print(f"validation seed={seed}, condition={condition}")
            episode = _simulate(
                cfg=cfg,
                condition_cfg=(
                    reference_cfg
                    if condition == "B_rhythmic_reference"
                    else asynchronous_cfg
                ),
                condition_id=condition,
                action=validation_actions[condition],
                stimulate=condition not in REFERENCE_CONDITIONS,
                seed=seed,
                action_index=action_index,
                output_dir=root / "validation" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                seed_episodes[condition] = episode
                for epoch in EPOCHS:
                    row, _, frequencies_hz, psd = _episode_epoch_feature_row(
                        episode,
                        condition_id=condition,
                        epoch=epoch,
                        selected_frequency_hz=selected_frequency_hz,
                        candidate_frequencies_hz=candidates,
                        cfg=cfg,
                    )
                    validation_epoch_rows.append(row)
                    if epoch == "stimulation" and condition in (
                        "A_async", "B_rhythmic_reference", PRIMARY_ACTION
                    ):
                        psd_rows.extend(
                            {
                                "seed": seed,
                                "condition_id": condition,
                                "frequency_hz": float(frequency),
                                "psd_v2_hz": float(value),
                            }
                            for frequency, value in zip(frequencies_hz, psd)
                            if 1.0 <= frequency <= 100.0
                        )
                validation_window_rows.extend(
                    _episode_window_feature_rows(
                        episode,
                        condition_id=condition,
                        selected_frequency_hz=selected_frequency_hz,
                        candidate_frequencies_hz=candidates,
                        cfg=cfg,
                    )
                )
        if rank == 0:
            synthetic, _, frequencies_hz, psd = _synthetic_feature_row(
                seed_episodes["A_async"],
                seed_episodes[PRIMARY_ACTION],
                selected_frequency_hz=selected_frequency_hz,
                candidate_frequencies_hz=candidates,
                cfg=cfg,
            )
            validation_epoch_rows.append(synthetic)
            psd_rows.extend(
                {
                    "seed": seed,
                    "condition_id": SYNTHETIC_CONTROL,
                    "frequency_hz": float(frequency),
                    "psd_v2_hz": float(value),
                }
                for frequency, value in zip(frequencies_hz, psd)
                if 1.0 <= frequency <= 100.0
            )
            validation_episodes[seed] = seed_episodes
            if representative_spikes is None:
                representative_spikes = _representative_spike_rows(
                    seed_episodes,
                    selected_frequency_hz=selected_frequency_hz,
                )

    if rank != 0:
        return

    epoch_frame = pd.DataFrame(validation_epoch_rows)
    window_frame = pd.DataFrame(validation_window_rows)
    epoch_frame = _add_distances(epoch_frame, full_model, prefix="full")
    epoch_frame = _add_distances(epoch_frame, excluded_model, prefix="excluded")
    window_frame = _add_distances(window_frame, full_model, prefix="full")
    window_frame = _add_distances(window_frame, excluded_model, prefix="excluded")
    epoch_frame.to_csv(root / "validation_epoch_eeg_metrics.csv", index=False)
    window_frame.to_csv(root / "validation_window_eeg_metrics.csv", index=False)
    pd.DataFrame(psd_rows).to_csv(root / "validation_psd_long.csv", index=False)
    representative_spikes.to_csv(root / "representative_E_spikes.csv", index=False)

    full_seed, full_summary = _validation_reachability(
        epoch_frame, prefix="full", cfg=cfg, rng=rng
    )
    excluded_seed, excluded_summary = _validation_reachability(
        epoch_frame, prefix="excluded", cfg=cfg, rng=rng
    )
    seed_reachability = pd.concat((full_seed, excluded_seed), ignore_index=True)
    summary_reachability = pd.concat(
        (full_summary, excluded_summary), ignore_index=True
    )
    seed_reachability.to_csv(root / "validation_reachability.csv", index=False)
    summary_reachability.to_csv(
        root / "validation_reachability_summary.csv", index=False
    )

    hidden = _hidden_validation_rows(
        epoch_frame[epoch_frame.condition_id.isin(VALIDATION_CONDITIONS)],
        validation_episodes,
        cfg=cfg,
    )
    hidden.to_csv(root / "validation_hidden_mechanism.csv", index=False)

    def reach_rows(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
        return frame[frame.condition_id.eq(condition)].sort_values("seed")

    def summary_row(frame: pd.DataFrame, condition: str) -> dict[str, Any]:
        return frame[frame.condition_id.eq(condition)].iloc[0].to_dict()

    primary_full = reach_rows(full_seed, PRIMARY_ACTION)
    lower_full = reach_rows(full_seed, LOWER_ACTION)
    transverse_full = reach_rows(full_seed, TRANSVERSE_CONTROL)
    synthetic_full = reach_rows(full_seed, SYNTHETIC_CONTROL)
    reference_full = reach_rows(full_seed, "B_rhythmic_reference")
    primary_excluded = reach_rows(excluded_seed, PRIMARY_ACTION)
    synthetic_excluded = reach_rows(excluded_seed, SYNTHETIC_CONTROL)
    reference_excluded = reach_rows(excluded_seed, "B_rhythmic_reference")

    orientation_advantage = _summary(
        primary_full.target_distance_improvement.to_numpy(float)
        - transverse_full.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    beyond_synthetic = _summary(
        primary_full.target_distance_improvement.to_numpy(float)
        - synthetic_full.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    reference_excluded_summary = _summary(
        reference_excluded.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    primary_excluded_summary = _summary(
        primary_excluded.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    beyond_synthetic_excluded = _summary(
        primary_excluded.target_distance_improvement.to_numpy(float)
        - synthetic_excluded.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    primary_hidden = hidden[hidden.condition_id.eq(PRIMARY_ACTION)]
    ppc_summary = _summary(
        primary_hidden.E_ppc_gain_difference_in_differences.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    primary_summary = summary_row(full_summary, PRIMARY_ACTION)
    lower_summary = summary_row(full_summary, LOWER_ACTION)
    reference_summary = summary_row(full_summary, "B_rhythmic_reference")

    criteria = cfg.analysis.criteria
    minimum_positive = float(criteria.minimum_positive_seed_fraction)
    selected_target_frequency_row = target_frequency_table[
        target_frequency_table.selected_from_eeg
    ].iloc[0]
    seed_classification = _classification_accuracy(epoch_frame, prefix="full")
    window_classification = _classification_accuracy(window_frame, prefix="full")
    excluded_seed_classification = _classification_accuracy(
        epoch_frame, prefix="excluded"
    )
    ideal_checks = {
        "minimum_validation_seeds": len(validation_seeds)
        >= int(criteria.minimum_validation_seeds),
        "target_frequency_visible_in_unstimulated_eeg": (
            float(selected_target_frequency_row.mean_standardized_shift) > 0.0
            and float(selected_target_frequency_row.positive_seed_fraction)
            >= minimum_positive
        ),
        "frequency_discovery_positive": (
            float(selected_frequency_row["mean"]) > 0.0
            and float(selected_frequency_row["positive_seed_fraction"])
            >= minimum_positive
        ),
        "phase_discovery_positive": (
            float(selected_phase_row["mean"]) > 0.0
            and float(selected_phase_row["positive_seed_fraction"])
            >= minimum_positive
        ),
        "heldout_reference_eeg_distinct": (
            float(reference_summary["ci_2.5"]) > 0.0
            and float(reference_summary["positive_seed_fraction"])
            >= minimum_positive
        ),
        "heldout_reference_classification": seed_classification
        >= float(criteria.minimum_reference_classification_accuracy),
        "one_second_eeg_observable": window_classification
        >= float(criteria.minimum_window_classification_accuracy),
        "selected_tacs_moves_eeg_toward_B": (
            float(primary_summary["ci_2.5"]) > 0.0
            and float(primary_summary["positive_seed_fraction"])
            >= minimum_positive
        ),
        "orientation_specific": orientation_advantage["ci_2.5"] > 0.0,
        "hidden_spike_timing_modulated": (
            ppc_summary["ci_2.5"] > 0.0
            and ppc_summary["positive_seed_fraction"] >= minimum_positive
        ),
        "rate_safe": float(primary_hidden.rate_safe.mean())
        >= float(criteria.minimum_rate_safe_seed_fraction),
        "washout_reversible": float(primary_hidden.washout_recovered.mean())
        >= float(criteria.minimum_washout_recovery_seed_fraction),
        "baseline_causality": float(
            primary_hidden.baseline_relative_rms_error_vs_A.max()
        )
        <= float(criteria.maximum_baseline_relative_rms_error),
    }
    robustness_checks = {
        "excluded_space_reference_observable": (
            reference_excluded_summary["ci_2.5"] > 0.0
            and excluded_seed_classification
            >= float(criteria.minimum_reference_classification_accuracy)
        ),
        "selected_tacs_moves_in_excluded_space": (
            primary_excluded_summary["ci_2.5"] > 0.0
            and primary_excluded_summary["positive_seed_fraction"]
            >= minimum_positive
        ),
        "beyond_complex_matched_observation": beyond_synthetic["ci_2.5"] > 0.0,
        "beyond_complex_match_after_exclusion": (
            beyond_synthetic_excluded["ci_2.5"] > 0.0
        ),
    }

    # Export a compact dataset for a later policy comparison, but do not train
    # a bandit until context-by-action interaction is confirmed on more seeds.
    baseline = epoch_frame[
        epoch_frame.condition_id.eq("A_async") & epoch_frame.epoch.eq("baseline")
    ].set_index("seed")
    transition_rows = []
    for seed in validation_seeds:
        for condition, amplitude in (
            ("A_async", 0.0),
            (LOWER_ACTION, float(cfg.analysis.validation.lower_amplitude_v_per_m)),
            (PRIMARY_ACTION, float(cfg.analysis.validation.primary_amplitude_v_per_m)),
        ):
            if condition == "A_async":
                improvement = 0.0
                outcome_distance = float(
                    epoch_frame[
                        epoch_frame.seed.eq(seed)
                        & epoch_frame.condition_id.eq("A_async")
                        & epoch_frame.epoch.eq("stimulation")
                    ].iloc[0].full_distance_to_B
                )
            else:
                row = full_seed[
                    full_seed.seed.eq(seed) & full_seed.condition_id.eq(condition)
                ].iloc[0]
                improvement = float(row.target_distance_improvement)
                outcome_distance = float(row.candidate_distance_to_B)
            transition_rows.append(
                {
                    "seed": seed,
                    "action_id": condition,
                    "amplitude_v_per_m": amplitude,
                    "frequency_hz": selected_frequency_hz,
                    "phase_rad": selected_phase_rad,
                    "context_distance_to_B": float(
                        baseline.loc[seed, "full_distance_to_B"]
                    ),
                    "outcome_distance_to_B": outcome_distance,
                    "target_distance_improvement": improvement,
                    "reward": -outcome_distance,
                }
            )
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(root / "future_policy_transition_table.csv", index=False)
    action_rewards = transitions.groupby("action_id").reward.mean()
    oracle = transitions.loc[transitions.groupby("seed").reward.idxmax()]
    policy_diagnostic = {
        "best_fixed_action": str(action_rewards.idxmax()),
        "mean_reward_by_action": _plain(action_rewards.to_dict()),
        "oracle_best_action_counts": _plain(oracle.action_id.value_counts().to_dict()),
        "oracle_advantage_over_best_fixed": float(
            oracle.reward.mean() - action_rewards.max()
        ),
        "interpretation": (
            "Descriptive only. More seeds and grouped cross-validation are "
            "required before fitting a contextual policy."
        ),
    }

    psd_frame = pd.DataFrame(psd_rows)
    if bool(cfg.experiment.plot):
        _plot_psd(
            psd_frame,
            selected_frequency_hz=selected_frequency_hz,
            root=root,
        )
        _plot_selection(
            frequency_effects,
            phase_effects,
            selected_frequency_hz=selected_frequency_hz,
            selected_phase_rad=selected_phase_rad,
            root=root,
        )
        _plot_validation_effects(
            full_seed,
            hidden,
            epoch_frame,
            root=root,
        )
        _plot_spike_timing(
            representative_spikes,
            epoch_frame,
            representative_seed=validation_seeds[0],
            selected_frequency_hz=selected_frequency_hz,
            selected_phase_rad=selected_phase_rad,
            window_ms=float(cfg.env.simulation.obs_win_len),
            root=root,
        )
        _plot_time_course(window_frame, root=root)
        _plot_artifact_control(full_seed, excluded_seed, root=root)

    conclusion = {
        "scope": (
            "Hierarchical EEG-only action identification in an ideal 40-cell "
            "toy circuit; not clinical simultaneous tACS-EEG validation."
        ),
        "hidden_reference_generator": {
            "frequency_hz": float(cfg.analysis.reference.frequency_hz),
            "modulation_depth": float(cfg.analysis.reference.modulation_depth),
            "used_by_selector": False,
        },
        "target_frequency_estimated_from_unstimulated_eeg_hz": target_peak_hz,
        "hidden_frequency_recovered_posthoc": bool(
            np.isclose(target_peak_hz, float(cfg.analysis.reference.frequency_hz))
        ),
        "frozen_protocol": frozen_protocol,
        "target_frequency_table": _plain(target_frequency_table.to_dict("records")),
        "frequency_screen_summary": _plain(frequency_summary.to_dict("records")),
        "phase_screen_summary": _plain(phase_summary.to_dict("records")),
        "reference_full": reference_summary,
        "lower_dose_full": lower_summary,
        "primary_full": primary_summary,
        "orientation_advantage": orientation_advantage,
        "hidden_primary_E_ppc_gain": ppc_summary,
        "reference_excluded": reference_excluded_summary,
        "primary_excluded": primary_excluded_summary,
        "advantage_beyond_complex_observation": beyond_synthetic,
        "advantage_beyond_complex_observation_excluded": (
            beyond_synthetic_excluded
        ),
        "classification": {
            "full_seed_accuracy": seed_classification,
            "full_one_second_window_accuracy": window_classification,
            "excluded_seed_accuracy": excluded_seed_classification,
        },
        "ideal_eeg_checks": ideal_checks,
        "concurrent_eeg_robustness_checks": robustness_checks,
        "ideal_eeg_reachability_passed": bool(all(ideal_checks.values())),
        "artifact_robust_reachability_passed": bool(
            all(ideal_checks.values()) and all(robustness_checks.values())
        ),
        "future_policy_diagnostic": policy_diagnostic,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    with (root / "experiment_conclusion.json").open("w", encoding="utf-8") as handle:
        json.dump(_plain(conclusion), handle, indent=2)

    print("\n### EEG-selected protocol")
    print(json.dumps(frozen_protocol, indent=2))
    print("\n### Ideal EEG reachability checks")
    for name, passed in ideal_checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Concurrent-EEG robustness checks")
    for name, passed in robustness_checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nIdeal hierarchical EEG reachability: "
        f"{'PASSED' if conclusion['ideal_eeg_reachability_passed'] else 'NOT PASSED'}"
    )
    print(
        "Artifact-robust hierarchical reachability: "
        f"{'PASSED' if conclusion['artifact_robust_reachability_passed'] else 'NOT PASSED'}"
    )
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
