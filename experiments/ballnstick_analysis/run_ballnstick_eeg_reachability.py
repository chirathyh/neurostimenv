"""Evaluate A -> B-like tACS reachability using observed EEG only.

The conditions are frozen from the preceding mechanistic pilot:

* A: homogeneous independent Poisson afferent event times;
* B: identical circuit and mean afferent rate, with a 0.04-depth 60-Hz
  sinusoidal modulation of independent afferent event probabilities;
* A+tACS: A's afferent statistics plus a uniform extracellular field.

Discovery circuits see A and B only and define a transparent standardized
centroid axis in a predeclared EEG feature space.  This EEG mapping is frozen
before disjoint validation circuits are stimulated.  The primary endpoint is
seed-level improvement toward the frozen B EEG centroid.  Spike locking and
rates are hidden mechanistic checks, never inputs to the EEG state score.

The primary EEG is the ideal artifact-free forward-model signal produced by
this simulator.  A second score removes the stimulation-frequency bins and an
observation-only matched sinusoid audits how much of the primary result could
be mimicked without changing the circuit.  These controls are reported as a
separate artifact-robust conclusion rather than silently redefining the
primary endpoint after results are observed.
"""

from __future__ import annotations

import json
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
from omegaconf import DictConfig, OmegaConf


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.stimulation import (  # noqa: E402
    apply_raised_cosine_block_envelope,
)
from experiments.ballnstick_analysis.run_ballnstick_entrainment_state import (  # noqa: E402
    _condition_config,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _analyze_eeg,
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _analyze_episode,
    _bootstrap_ci,
    _phase_locking_metrics,
    _phase_rng,
    _relative_rate_safe,
    _sign_flip_p,
    _simulate_episode,
)


ANALYSIS_EPOCHS = ("baseline", "stimulation", "washout")
PRIMARY_ACTION = "A_tacs_0p8"
LOWER_ACTION = "A_tacs_0p5"
TRANSVERSE_CONTROL = "A_tacs_transverse"
SYNTHETIC_CONTROL = "A_synthetic_observation"
REFERENCE_CONDITIONS = ("A_async", "B_rhythmic_reference")
VALIDATION_CONDITIONS = (
    "A_async",
    "B_rhythmic_reference",
    PRIMARY_ACTION,
    LOWER_ACTION,
    TRANSVERSE_CONTROL,
)


def _validate_design(cfg: DictConfig) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("EEG reachability requires the online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("A, B, and every tACS arm require inhibition_scale=1.0.")
    if not np.isclose(
        float(cfg.analysis.reference.frequency_hz),
        float(cfg.analysis.tacs.frequency_hz),
    ):
        raise ValueError("The fixed B rhythm and tACS frequency must match.")
    depth = float(cfg.analysis.reference.modulation_depth)
    envelope = float(
        cfg.analysis.reference.thinning_envelope_modulation_depth
    )
    if depth <= 0.0 or depth > 1.0 or envelope < depth or envelope > 1.0:
        raise ValueError("Invalid fixed B modulation/envelope depth.")
    low = float(cfg.analysis.tacs.lower_amplitude_v_per_m)
    high = float(cfg.analysis.tacs.primary_amplitude_v_per_m)
    maximum = float(cfg.analysis.maximum_field_v_per_m)
    if not 0.0 < low < high <= maximum:
        raise ValueError("Require 0 < lower dose < primary dose <= maximum field.")
    if int(cfg.analysis.discovery.n_seeds) < 2:
        raise ValueError("At least two A/B discovery circuit seeds are required.")
    if int(cfg.analysis.validation.n_seeds) < 1:
        raise ValueError("At least one validation circuit seed is required.")
    for feature_set_name in (
        "primary_eeg_features",
        "fundamental_excluded_eeg_features",
    ):
        names = [str(value) for value in cfg.analysis[feature_set_name]]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{feature_set_name} must be nonempty and unique.")


def _action(
    cfg: DictConfig,
    *,
    identifier: str,
    role: str,
    amplitude_v_per_m: float,
    montage: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "role": role,
        "montage": montage,
        "dc_offset_v_per_m": 0.0,
        "ac_amplitude_v_per_m": float(amplitude_v_per_m),
        "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
        "phase_rad": float(cfg.analysis.tacs.phase_rad),
    }


def _actions(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    axial = str(cfg.analysis.tacs.axial_montage)
    transverse = str(cfg.analysis.tacs.transverse_montage)
    return {
        "A_async": _action(
            cfg,
            identifier="A_async",
            role="asynchronous_state",
            amplitude_v_per_m=0.0,
            montage=axial,
        ),
        "B_rhythmic_reference": _action(
            cfg,
            identifier="B_rhythmic_reference",
            role="synaptic_reference_state",
            amplitude_v_per_m=0.0,
            montage=axial,
        ),
        PRIMARY_ACTION: _action(
            cfg,
            identifier=PRIMARY_ACTION,
            role="primary_tacs_action",
            amplitude_v_per_m=float(
                cfg.analysis.tacs.primary_amplitude_v_per_m
            ),
            montage=axial,
        ),
        LOWER_ACTION: _action(
            cfg,
            identifier=LOWER_ACTION,
            role="lower_dose_action",
            amplitude_v_per_m=float(
                cfg.analysis.tacs.lower_amplitude_v_per_m
            ),
            montage=axial,
        ),
        TRANSVERSE_CONTROL: _action(
            cfg,
            identifier=TRANSVERSE_CONTROL,
            role="orientation_control",
            amplitude_v_per_m=float(
                cfg.analysis.tacs.primary_amplitude_v_per_m
            ),
            montage=transverse,
        ),
    }


def _eeg_target_quadratures(
    processed_eeg: np.ndarray,
    *,
    fs_hz: float,
    start_ms: float,
    frequency_hz: float,
) -> dict[str, float]:
    """Return normalized sine/cosine EEG coefficients at one frequency."""
    signal = np.asarray(processed_eeg, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(signal))))
    if signal.size == 0 or rms <= np.finfo(float).tiny:
        return {
            "eeg_target_cosine": 0.0,
            "eeg_target_sine": 0.0,
            "eeg_target_resultant": 0.0,
            "eeg_target_phase_rad": 0.0,
        }
    # Online observations use (start, stop]. The resampled first sample is
    # therefore assigned one output-sampling interval after the left boundary.
    time_s = (
        float(start_ms) / 1000.0
        + (np.arange(signal.size, dtype=float) + 1.0) / float(fs_hz)
    )
    angle = 2.0 * np.pi * float(frequency_hz) * time_s
    normalization = np.sqrt(2.0) / rms
    cosine = float(np.mean(signal * np.cos(angle)) * normalization)
    sine = float(np.mean(signal * np.sin(angle)) * normalization)
    return {
        "eeg_target_cosine": cosine,
        "eeg_target_sine": sine,
        "eeg_target_resultant": float(np.hypot(cosine, sine)),
        "eeg_target_phase_rad": float(np.arctan2(sine, cosine)),
    }


def _augment_eeg_features(
    features: dict[str, Any],
    processed: np.ndarray,
    *,
    analysis_fs_hz: float,
    start_ms: float,
    frequency_hz: float,
) -> dict[str, Any]:
    result = dict(features)
    eps = np.finfo(np.float64).tiny
    result["log10_stimulus_frequency_power"] = float(
        np.log10(max(float(result["stimulus_frequency_power"]), eps))
    )
    result.update(
        _eeg_target_quadratures(
            processed,
            fs_hz=analysis_fs_hz,
            start_ms=start_ms,
            frequency_hz=frequency_hz,
        )
    )
    return result


def _analyze_windows(
    simulation: dict[str, Any],
    *,
    condition_id: str,
    action: dict[str, Any],
    cfg: DictConfig,
    action_index: int,
) -> list[dict[str, Any]]:
    simulator_fs_hz = 1000.0 / float(cfg.env.network.dt)
    analysis_fs_hz = float(cfg.analysis.target_fs_hz)
    frequency_hz = float(cfg.analysis.reference.frequency_hz)
    rows: list[dict[str, Any]] = []
    for epoch_index, epoch in enumerate(ANALYSIS_EPOCHS):
        outputs = simulation["outputs_by_epoch"][epoch]
        for window_index, output in enumerate(outputs):
            raw = np.asarray(output["eeg_v"], dtype=float).reshape(-1)
            features, processed, _, _ = _analyze_eeg(
                raw,
                simulator_fs_hz=simulator_fs_hz,
                cfg=cfg,
            )
            augmented = _augment_eeg_features(
                features,
                processed,
                analysis_fs_hz=analysis_fs_hz,
                start_ms=float(output["t_start_ms"]),
                frequency_hz=frequency_hz,
            )
            duration_s = float(
                output["t_stop_ms"] - output["t_start_ms"]
            ) / 1000.0
            row: dict[str, Any] = {
                "seed": int(simulation["seed"]),
                "condition_id": condition_id,
                "epoch": epoch,
                "epoch_window_index": int(window_index),
                "t_start_ms": float(output["t_start_ms"]),
                "t_stop_ms": float(output["t_stop_ms"]),
                "ac_amplitude_v_per_m": float(
                    action["ac_amplitude_v_per_m"]
                ),
                "frequency_hz": frequency_hz,
                "phase_rad": float(action.get("phase_rad", 0.0)),
                **augmented,
            }
            for population_index, population_name in enumerate(("E", "I")):
                times = np.asarray(
                    output["spikes"][population_name]["times_ms"],
                    dtype=float,
                )
                population_size = len(
                    output["spikes"][population_name]["per_cell"]
                )
                row[f"{population_name}_firing_rate_hz"] = float(
                    times.size / (population_size * duration_s)
                )
                locking = _phase_locking_metrics(
                    times,
                    frequency_hz=frequency_hz,
                    phase_origin_ms=float(simulation["block_start_ms"]),
                    n_surrogates=0,
                    rng=_phase_rng(
                        int(simulation["seed"]),
                        action_index,
                        epoch_index,
                        population_index,
                        1,
                    ),
                )
                row[f"{population_name}_plv"] = locking["plv"]
                row[f"{population_name}_ppc"] = locking["ppc"]
                row[f"{population_name}_mean_phase_rad"] = locking[
                    "mean_phase_rad"
                ]
            rows.append(row)
    return rows


def _run_condition(
    *,
    condition_id: str,
    condition_cfg: DictConfig,
    action: dict[str, Any],
    stimulate: bool,
    seed: int,
    action_index: int,
    output_dir: Path,
    comm,
    size: int,
    rank: int,
) -> dict[str, Any] | None:
    simulation = _simulate_episode(
        condition_cfg,
        seed=seed,
        action=action,
        stimulate=stimulate,
        output_dir=output_dir,
        comm=comm,
        size=size,
        rank=rank,
    )
    if rank != 0:
        return None
    epoch_rows, raw_by_epoch = _analyze_episode(
        simulation,
        action=action,
        action_index=action_index,
        arm=condition_id,
        cfg=condition_cfg,
        output_dir=output_dir / "analysis",
    )
    simulator_fs_hz = 1000.0 / float(condition_cfg.env.network.dt)
    analysis_fs_hz = float(condition_cfg.analysis.target_fs_hz)
    for row in epoch_rows:
        epoch = str(row["epoch"])
        outputs = simulation["outputs_by_epoch"][epoch]
        raw = raw_by_epoch[epoch]
        features, processed, _, _ = _analyze_eeg(
            raw,
            simulator_fs_hz=simulator_fs_hz,
            cfg=condition_cfg,
        )
        row.update(
            _augment_eeg_features(
                features,
                processed,
                analysis_fs_hz=analysis_fs_hz,
                start_ms=float(outputs[0]["t_start_ms"]),
                frequency_hz=float(condition_cfg.analysis.reference.frequency_hz),
            )
        )
        row["condition_id"] = condition_id
    window_rows = _analyze_windows(
        simulation,
        condition_id=condition_id,
        action=action,
        cfg=condition_cfg,
        action_index=action_index,
    )
    return {
        "simulation": simulation,
        "epoch_rows": epoch_rows,
        "window_rows": window_rows,
        "raw_by_epoch": raw_by_epoch,
        "simulator_fs_hz": simulator_fs_hz,
    }


def _fit_eeg_axis(
    discovery_rows: pd.DataFrame,
    *,
    feature_names: list[str],
) -> dict[str, Any]:
    subset = discovery_rows[
        (discovery_rows["epoch"] == "stimulation")
        & discovery_rows["condition_id"].isin(REFERENCE_CONDITIONS)
    ]
    a = subset[subset["condition_id"] == "A_async"][feature_names].to_numpy(
        dtype=float
    )
    b = subset[
        subset["condition_id"] == "B_rhythmic_reference"
    ][feature_names].to_numpy(dtype=float)
    if a.shape != b.shape or a.shape[0] < 2:
        raise ValueError("Matched A/B discovery rows are required to fit EEG state.")
    pooled = np.vstack((a, b))
    center = np.mean(pooled, axis=0)
    scale = np.std(pooled, axis=0, ddof=1)
    positive = scale[scale > np.finfo(float).eps]
    fallback = float(np.median(positive)) if positive.size else 1.0
    scale = np.where(scale > np.finfo(float).eps, scale, fallback)
    a_centroid = np.mean((a - center) / scale, axis=0)
    b_centroid = np.mean((b - center) / scale, axis=0)
    direction = b_centroid - a_centroid
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(float).eps:
        raise ValueError("Discovery A and B have identical EEG centroids.")
    direction /= norm
    midpoint = 0.5 * (a_centroid + b_centroid)
    a_score = float(np.dot(a_centroid - midpoint, direction))
    b_score = float(np.dot(b_centroid - midpoint, direction))
    return {
        "feature_names": list(feature_names),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "direction": direction.tolist(),
        "midpoint": midpoint.tolist(),
        "A_centroid_score": a_score,
        "B_centroid_score": b_score,
        "classification_threshold": 0.0,
    }


def _score_eeg_rows(rows: pd.DataFrame, axis: dict[str, Any]) -> np.ndarray:
    names = list(axis["feature_names"])
    center = np.asarray(axis["center"], dtype=float)
    scale = np.asarray(axis["scale"], dtype=float)
    direction = np.asarray(axis["direction"], dtype=float)
    midpoint = np.asarray(axis["midpoint"], dtype=float)
    standardized = (rows[names].to_numpy(dtype=float) - center) / scale
    return (standardized - midpoint) @ direction


def _match_target_band_sine(
    a_episode: dict[str, Any],
    active_episode: dict[str, Any],
    *,
    cfg: DictConfig,
) -> tuple[np.ndarray, float, float]:
    """Add an observation-only sine to A to match active 60-Hz power."""
    if not bool(cfg.analysis.synthetic_control.match_target_band_power):
        raise ValueError("The target-band synthetic control must remain enabled.")
    a_raw = np.asarray(a_episode["raw_by_epoch"]["stimulation"], dtype=float)
    active_raw = np.asarray(
        active_episode["raw_by_epoch"]["stimulation"], dtype=float
    )
    simulator_fs_hz = float(a_episode["simulator_fs_hz"])
    a_features, _, _, _ = _analyze_eeg(
        a_raw, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    active_features, _, _, _ = _analyze_eeg(
        active_raw, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    baseline_power = float(a_features["stimulus_frequency_power"])
    target_power = float(active_features["stimulus_frequency_power"])
    outputs = a_episode["simulation"]["outputs_by_epoch"]["stimulation"]
    time_ms = np.concatenate(
        [np.asarray(output["sample_times_ms"], dtype=float) for output in outputs]
    )
    frequency_hz = float(cfg.analysis.tacs.frequency_hz)
    phase_rad = float(cfg.analysis.tacs.phase_rad)
    unit_sine = np.sin(2.0 * np.pi * frequency_hz * time_ms / 1000.0 + phase_rad)
    first = float(outputs[0]["t_start_ms"])
    last = float(outputs[-1]["t_stop_ms"])
    unit_sine = apply_raised_cosine_block_envelope(
        unit_sine,
        time_ms=time_ms,
        block_start_ms=first,
        block_stop_ms=last,
        ramp_ms=float(cfg.analysis.timeline.block_ramp_ms),
    )
    if target_power <= baseline_power:
        return a_raw.copy(), 0.0, baseline_power

    def objective(amplitude_v: float) -> float:
        features, _, _, _ = _analyze_eeg(
            a_raw + float(amplitude_v) * unit_sine,
            simulator_fs_hz=simulator_fs_hz,
            cfg=cfg,
        )
        return float(features["stimulus_frequency_power"] - target_power)

    high = max(float(np.std(a_raw)) * 0.01, 1e-15)
    for _ in range(50):
        if objective(high) >= 0.0:
            break
        high *= 2.0
    else:
        raise RuntimeError("Could not bracket matched-sinusoid amplitude.")
    amplitude = float(so.brentq(objective, 0.0, high, xtol=1e-18, rtol=1e-12))
    synthetic = a_raw + amplitude * unit_sine
    achieved, _, _, _ = _analyze_eeg(
        synthetic, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    return synthetic, amplitude, float(achieved["stimulus_frequency_power"])


def _synthetic_rows(
    a_episode: dict[str, Any],
    active_episode: dict[str, Any],
    *,
    cfg: DictConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    synthetic, amplitude_v, achieved_power = _match_target_band_sine(
        a_episode, active_episode, cfg=cfg
    )
    simulator_fs_hz = float(a_episode["simulator_fs_hz"])
    analysis_fs_hz = float(cfg.analysis.target_fs_hz)
    outputs = a_episode["simulation"]["outputs_by_epoch"]["stimulation"]
    features, processed, _, _ = _analyze_eeg(
        synthetic, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    source = next(
        row
        for row in a_episode["epoch_rows"]
        if row["epoch"] == "stimulation"
    )
    row = dict(source)
    row.update(
        _augment_eeg_features(
            features,
            processed,
            analysis_fs_hz=analysis_fs_hz,
            start_ms=float(outputs[0]["t_start_ms"]),
            frequency_hz=float(cfg.analysis.reference.frequency_hz),
        )
    )
    row["condition_id"] = SYNTHETIC_CONTROL
    row["arm"] = SYNTHETIC_CONTROL
    row["synthetic_added_peak_v"] = amplitude_v
    row["synthetic_achieved_target_band_power"] = achieved_power

    window_rows: list[dict[str, Any]] = []
    offset = 0
    for window_index, output in enumerate(outputs):
        sample_count = int(np.asarray(output["eeg_v"]).size)
        window_raw = synthetic[offset : offset + sample_count]
        offset += sample_count
        window_features, window_processed, _, _ = _analyze_eeg(
            window_raw, simulator_fs_hz=simulator_fs_hz, cfg=cfg
        )
        augmented = _augment_eeg_features(
            window_features,
            window_processed,
            analysis_fs_hz=analysis_fs_hz,
            start_ms=float(output["t_start_ms"]),
            frequency_hz=float(cfg.analysis.reference.frequency_hz),
        )
        window_rows.append(
            {
                "seed": int(a_episode["simulation"]["seed"]),
                "condition_id": SYNTHETIC_CONTROL,
                "epoch": "stimulation",
                "epoch_window_index": int(window_index),
                "t_start_ms": float(output["t_start_ms"]),
                "t_stop_ms": float(output["t_stop_ms"]),
                "ac_amplitude_v_per_m": 0.0,
                "frequency_hz": float(cfg.analysis.reference.frequency_hz),
                "phase_rad": float(cfg.analysis.tacs.phase_rad),
                **augmented,
            }
        )
    return row, window_rows


def _summary(
    values: np.ndarray,
    *,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    ci_low, ci_high = _bootstrap_ci(
        values,
        rng=rng,
        n_bootstrap=int(cfg.analysis.n_bootstrap),
    )
    return {
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "ci_2.5": ci_low,
        "ci_97.5": ci_high,
        "positive_seed_fraction": float(np.mean(values > 0.0)),
        "paired_sign_flip_p": _sign_flip_p(
            values,
            rng=rng,
            n_permutations=int(cfg.analysis.n_permutations),
        ),
    }


def _condition_epoch(rows: pd.DataFrame, condition: str, epoch: str) -> pd.DataFrame:
    return rows[
        (rows["condition_id"] == condition) & (rows["epoch"] == epoch)
    ].set_index("seed").sort_index()


def _state_metrics(
    rows: pd.DataFrame,
    *,
    score_name: str,
    target_score: float,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = _condition_epoch(rows, "A_async", "stimulation")
    b = _condition_epoch(rows, "B_rhythmic_reference", "stimulation")
    seed_rows: list[dict[str, Any]] = []
    for condition in (
        "A_async",
        "B_rhythmic_reference",
        PRIMARY_ACTION,
        LOWER_ACTION,
        TRANSVERSE_CONTROL,
        SYNTHETIC_CONTROL,
    ):
        candidate = _condition_epoch(rows, condition, "stimulation")
        common = a.index.intersection(b.index).intersection(candidate.index)
        for seed in common:
            a_score = float(a.loc[seed, score_name])
            b_score = float(b.loc[seed, score_name])
            candidate_score = float(candidate.loc[seed, score_name])
            baseline_distance = abs(float(target_score) - a_score)
            candidate_distance = abs(float(target_score) - candidate_score)
            seed_rows.append(
                {
                    "seed": int(seed),
                    "condition_id": condition,
                    "feature_set": score_name,
                    "A_score": a_score,
                    "validation_B_score": b_score,
                    "candidate_score": candidate_score,
                    "frozen_B_target_score": float(target_score),
                    "A_distance_to_target": baseline_distance,
                    "candidate_distance_to_target": candidate_distance,
                    "target_distance_improvement": (
                        baseline_distance - candidate_distance
                    ),
                    "fractional_target_distance_improvement": (
                        1.0 - candidate_distance / baseline_distance
                        if baseline_distance > 0.0
                        else 0.0
                    ),
                    "target_direction_aligned": bool(
                        (float(target_score) - a_score)
                        * (candidate_score - a_score)
                        > 0.0
                    ),
                }
            )
    seed_frame = pd.DataFrame(seed_rows)
    summaries = []
    for condition, group in seed_frame.groupby("condition_id", sort=False):
        effect = _summary(
            group["target_distance_improvement"].to_numpy(dtype=float),
            cfg=cfg,
            rng=rng,
        )
        summaries.append(
            {
                "condition_id": condition,
                "feature_set": score_name,
                **effect,
                "mean_fractional_improvement": float(
                    group["fractional_target_distance_improvement"].mean()
                ),
                "direction_aligned_seed_fraction": float(
                    group["target_direction_aligned"].mean()
                ),
            }
        )
    return seed_frame, pd.DataFrame(summaries)


def _classification_metrics(
    epoch_rows: pd.DataFrame,
    window_rows: pd.DataFrame,
    *,
    score_name: str,
) -> dict[str, Any]:
    reference = epoch_rows[
        (epoch_rows["epoch"] == "stimulation")
        & epoch_rows["condition_id"].isin(REFERENCE_CONDITIONS)
    ].copy()
    expected = reference["condition_id"].eq("B_rhythmic_reference")
    predicted = reference[score_name] > 0.0
    epoch_accuracy = float(np.mean(expected.to_numpy() == predicted.to_numpy()))

    windows = window_rows[
        (window_rows["epoch"] == "stimulation")
        & window_rows["condition_id"].isin(REFERENCE_CONDITIONS)
    ].copy()
    windows["correct"] = (
        (windows[score_name] > 0.0)
        == windows["condition_id"].eq("B_rhythmic_reference")
    )
    per_seed = windows.groupby("seed")["correct"].mean()
    return {
        "seed_level_accuracy": epoch_accuracy,
        "one_second_window_accuracy": float(windows["correct"].mean()),
        "mean_within_seed_window_accuracy": float(per_seed.mean()),
        "per_seed_window_accuracy": {
            str(int(seed)): float(value) for seed, value in per_seed.items()
        },
    }


def _hidden_mechanistic_rows(
    epoch_rows: pd.DataFrame,
    episodes: dict[int, dict[str, dict[str, Any]]],
    *,
    cfg: DictConfig,
) -> pd.DataFrame:
    a_stim = _condition_epoch(epoch_rows, "A_async", "stimulation")
    a_base = _condition_epoch(epoch_rows, "A_async", "baseline")
    a_wash = _condition_epoch(epoch_rows, "A_async", "washout")
    b_stim = _condition_epoch(epoch_rows, "B_rhythmic_reference", "stimulation")
    rows = []
    residual_fraction = float(
        cfg.analysis.criteria.maximum_washout_residual_fraction
    )
    for condition in (PRIMARY_ACTION, LOWER_ACTION, TRANSVERSE_CONTROL):
        stim = _condition_epoch(epoch_rows, condition, "stimulation")
        base = _condition_epoch(epoch_rows, condition, "baseline")
        wash = _condition_epoch(epoch_rows, condition, "washout")
        for seed in stim.index:
            ppc_gain = (
                float(stim.loc[seed, "E_ppc"] - base.loc[seed, "E_ppc"])
                - float(a_stim.loc[seed, "E_ppc"] - a_base.loc[seed, "E_ppc"])
            )
            washout_gain = (
                float(wash.loc[seed, "E_ppc"] - base.loc[seed, "E_ppc"])
                - float(a_wash.loc[seed, "E_ppc"] - a_base.loc[seed, "E_ppc"])
            )
            phase_difference = float(
                np.angle(
                    np.exp(
                        1j
                        * (
                            float(stim.loc[seed, "E_mean_phase_rad"])
                            - float(b_stim.loc[seed, "E_mean_phase_rad"])
                        )
                    )
                )
            )
            active_raw = episodes[int(seed)][condition]["raw_by_epoch"]["baseline"]
            a_raw = episodes[int(seed)]["A_async"]["raw_by_epoch"]["baseline"]
            rows.append(
                {
                    "seed": int(seed),
                    "condition_id": condition,
                    "E_ppc_gain_difference_in_differences": ppc_gain,
                    "E_phase_difference_to_B_rad": phase_difference,
                    "E_phase_difference_to_B_degrees": float(
                        np.degrees(phase_difference)
                    ),
                    "E_rate_change_vs_A_hz": float(
                        stim.loc[seed, "E_firing_rate_hz"]
                        - a_stim.loc[seed, "E_firing_rate_hz"]
                    ),
                    "I_rate_change_vs_A_hz": float(
                        stim.loc[seed, "I_firing_rate_hz"]
                        - a_stim.loc[seed, "I_firing_rate_hz"]
                    ),
                    "rate_safe": _relative_rate_safe(
                        stim.loc[seed], a_stim.loc[seed], cfg
                    ),
                    "washout_recovered": bool(
                        ppc_gain > 0.0
                        and abs(washout_gain)
                        <= residual_fraction
                        * max(abs(ppc_gain), np.finfo(float).eps)
                    ),
                    "baseline_relative_rms_error_vs_A": _relative_rms_error(
                        a_raw, active_raw
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_scores(rows: pd.DataFrame, root: Path) -> None:
    stimulation = rows[rows["epoch"] == "stimulation"]
    order = [
        "A_async",
        "B_rhythmic_reference",
        LOWER_ACTION,
        PRIMARY_ACTION,
        TRANSVERSE_CONTROL,
        SYNTHETIC_CONTROL,
    ]
    values = [
        stimulation[stimulation["condition_id"] == condition][
            "primary_eeg_score"
        ].to_numpy(dtype=float)
        for condition in order
    ]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.boxplot(values, labels=order, showmeans=True)
    axis.axhline(0.0, color="0.5", linewidth=1)
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylabel("Frozen A-to-B EEG score")
    axis.set_title("Held-out EEG-state responses")
    figure.tight_layout()
    figure.savefig(root / "validation_eeg_state_scores.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate_design(cfg)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "eeg_reachability"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### BallAndStick EEG-primary entrainment-state reachability")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    actions = _actions(cfg)
    asynchronous_cfg = _condition_config(cfg, modulation_depth=0.0)
    reference_cfg = _condition_config(
        cfg, modulation_depth=float(cfg.analysis.reference.modulation_depth)
    )
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
        raise ValueError("Discovery and validation seeds must be disjoint.")

    discovery_epoch_rows: list[dict[str, Any]] = []
    discovery_window_rows: list[dict[str, Any]] = []
    for seed in discovery_seeds:
        for action_index, condition in enumerate(REFERENCE_CONDITIONS):
            if rank == 0:
                print(f"discovery seed={seed}, condition={condition}")
            episode = _run_condition(
                condition_id=condition,
                condition_cfg=(
                    asynchronous_cfg if condition == "A_async" else reference_cfg
                ),
                action=actions[condition],
                stimulate=False,
                seed=seed,
                action_index=action_index,
                output_dir=root / "discovery" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                discovery_epoch_rows.extend(episode["epoch_rows"])
                discovery_window_rows.extend(episode["window_rows"])

    if rank == 0:
        discovery_epochs = pd.DataFrame(discovery_epoch_rows)
        discovery_windows = pd.DataFrame(discovery_window_rows)
        primary_axis = _fit_eeg_axis(
            discovery_epochs,
            feature_names=[
                str(value) for value in cfg.analysis.primary_eeg_features
            ],
        )
        excluded_axis = _fit_eeg_axis(
            discovery_epochs,
            feature_names=[
                str(value)
                for value in cfg.analysis.fundamental_excluded_eeg_features
            ],
        )
        discovery_epochs["primary_eeg_score"] = _score_eeg_rows(
            discovery_epochs, primary_axis
        )
        discovery_epochs["fundamental_excluded_eeg_score"] = _score_eeg_rows(
            discovery_epochs, excluded_axis
        )
        discovery_windows["primary_eeg_score"] = _score_eeg_rows(
            discovery_windows, primary_axis
        )
        discovery_windows["fundamental_excluded_eeg_score"] = _score_eeg_rows(
            discovery_windows, excluded_axis
        )
        discovery_epochs.to_csv(root / "discovery_epoch_metrics.csv", index=False)
        discovery_windows.to_csv(root / "discovery_window_metrics.csv", index=False)
        axes_payload = {
            "primary_ideal_eeg": primary_axis,
            "fundamental_excluded": excluded_axis,
            "interpretation": (
                "Axes are fitted from unstimulated A/B discovery circuits only "
                "and frozen before all validation stimulation responses."
            ),
        }
        with (root / "frozen_eeg_state_mapping.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(axes_payload, handle, indent=2)
    else:
        primary_axis = excluded_axis = None
    primary_axis = comm.bcast(primary_axis, root=0)
    excluded_axis = comm.bcast(excluded_axis, root=0)

    validation_epoch_rows: list[dict[str, Any]] = []
    validation_window_rows: list[dict[str, Any]] = []
    episodes: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in validation_seeds:
        seed_episodes: dict[str, dict[str, Any]] = {}
        for action_index, condition in enumerate(VALIDATION_CONDITIONS):
            if rank == 0:
                print(f"validation seed={seed}, condition={condition}")
            is_reference = condition == "B_rhythmic_reference"
            is_stimulated = condition not in REFERENCE_CONDITIONS
            episode = _run_condition(
                condition_id=condition,
                condition_cfg=reference_cfg if is_reference else asynchronous_cfg,
                action=actions[condition],
                stimulate=is_stimulated,
                seed=seed,
                action_index=action_index,
                output_dir=root / "validation" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                seed_episodes[condition] = episode
                validation_epoch_rows.extend(episode["epoch_rows"])
                validation_window_rows.extend(episode["window_rows"])
        if rank == 0:
            synthetic_row, synthetic_windows = _synthetic_rows(
                seed_episodes["A_async"],
                seed_episodes[PRIMARY_ACTION],
                cfg=cfg,
            )
            validation_epoch_rows.append(synthetic_row)
            validation_window_rows.extend(synthetic_windows)
            episodes[int(seed)] = seed_episodes

    if rank != 0:
        return

    epoch_frame = pd.DataFrame(validation_epoch_rows)
    window_frame = pd.DataFrame(validation_window_rows)
    for score_name, axis in (
        ("primary_eeg_score", primary_axis),
        ("fundamental_excluded_eeg_score", excluded_axis),
    ):
        epoch_frame[score_name] = _score_eeg_rows(epoch_frame, axis)
        window_frame[score_name] = _score_eeg_rows(window_frame, axis)
    epoch_frame.to_csv(root / "validation_epoch_metrics.csv", index=False)
    window_frame.to_csv(root / "validation_window_metrics.csv", index=False)

    rng = np.random.default_rng(base_seed + 1_300_001)
    primary_seed, primary_summary = _state_metrics(
        epoch_frame,
        score_name="primary_eeg_score",
        target_score=float(primary_axis["B_centroid_score"]),
        cfg=cfg,
        rng=rng,
    )
    excluded_seed, excluded_summary = _state_metrics(
        epoch_frame,
        score_name="fundamental_excluded_eeg_score",
        target_score=float(excluded_axis["B_centroid_score"]),
        cfg=cfg,
        rng=rng,
    )
    state_seed_frame = pd.concat((primary_seed, excluded_seed), ignore_index=True)
    state_summary_frame = pd.concat(
        (primary_summary, excluded_summary), ignore_index=True
    )
    state_seed_frame.to_csv(root / "validation_eeg_reachability.csv", index=False)
    state_summary_frame.to_csv(
        root / "validation_eeg_reachability_summary.csv", index=False
    )

    classification = _classification_metrics(
        epoch_frame, window_frame, score_name="primary_eeg_score"
    )
    a = _condition_epoch(epoch_frame, "A_async", "stimulation")
    b = _condition_epoch(epoch_frame, "B_rhythmic_reference", "stimulation")
    reference_shift = _summary(
        (b["primary_eeg_score"] - a["primary_eeg_score"]).to_numpy(dtype=float),
        cfg=cfg,
        rng=rng,
    )

    hidden_frame = _hidden_mechanistic_rows(
        epoch_frame, episodes, cfg=cfg
    )
    hidden_frame.to_csv(root / "validation_hidden_mechanism.csv", index=False)
    hidden_primary = hidden_frame[hidden_frame["condition_id"] == PRIMARY_ACTION]
    hidden_ppc = _summary(
        hidden_primary["E_ppc_gain_difference_in_differences"].to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    phase_complex = np.mean(
        np.exp(
            1j
            * hidden_primary["E_phase_difference_to_B_rad"].to_numpy(float)
        )
    )

    def summary_row(frame: pd.DataFrame, condition: str) -> pd.Series:
        return frame[frame["condition_id"] == condition].iloc[0]

    primary = summary_row(primary_summary, PRIMARY_ACTION)
    lower = summary_row(primary_summary, LOWER_ACTION)
    transverse = summary_row(primary_summary, TRANSVERSE_CONTROL)
    synthetic = summary_row(primary_summary, SYNTHETIC_CONTROL)
    excluded_primary = summary_row(excluded_summary, PRIMARY_ACTION)

    primary_seed_rows = primary_seed.set_index(["seed", "condition_id"])
    orientation_values = []
    synthetic_values = []
    for seed in validation_seeds:
        high_value = float(
            primary_seed_rows.loc[
                (seed, PRIMARY_ACTION), "target_distance_improvement"
            ]
        )
        orientation_values.append(
            high_value
            - float(
                primary_seed_rows.loc[
                    (seed, TRANSVERSE_CONTROL),
                    "target_distance_improvement",
                ]
            )
        )
        synthetic_values.append(
            high_value
            - float(
                primary_seed_rows.loc[
                    (seed, SYNTHETIC_CONTROL),
                    "target_distance_improvement",
                ]
            )
        )
    orientation_advantage = _summary(
        np.asarray(orientation_values), cfg=cfg, rng=rng
    )
    beyond_synthetic = _summary(
        np.asarray(synthetic_values), cfg=cfg, rng=rng
    )

    criteria = cfg.analysis.criteria
    minimum_positive = float(criteria.minimum_positive_seed_fraction)
    ideal_checks = {
        "minimum_validation_seeds": len(validation_seeds)
        >= int(criteria.minimum_validation_seeds),
        "heldout_reference_eeg_distinct": (
            reference_shift["ci_2.5"] > 0.0
            and reference_shift["positive_seed_fraction"] >= minimum_positive
        ),
        "heldout_reference_classification": float(
            classification["seed_level_accuracy"]
        )
        >= float(criteria.minimum_reference_classification_accuracy),
        "one_second_eeg_observable": float(
            classification["one_second_window_accuracy"]
        )
        >= float(criteria.minimum_window_classification_accuracy),
        "primary_tacs_moves_eeg_toward_B": (
            float(primary["ci_2.5"]) > 0.0
            and float(primary["positive_seed_fraction"]) >= minimum_positive
            and float(primary["direction_aligned_seed_fraction"])
            >= minimum_positive
        ),
        "orientation_specific_eeg_movement": (
            orientation_advantage["ci_2.5"] > 0.0
        ),
        "hidden_spike_timing_modulated": (
            hidden_ppc["ci_2.5"] > 0.0
            and hidden_ppc["positive_seed_fraction"] >= minimum_positive
        ),
        "rate_safe": float(hidden_primary["rate_safe"].mean())
        >= float(criteria.minimum_rate_safe_seed_fraction),
        "washout_reversible": float(
            hidden_primary["washout_recovered"].mean()
        )
        >= float(criteria.minimum_washout_recovery_seed_fraction),
        "baseline_causality": float(
            hidden_primary["baseline_relative_rms_error_vs_A"].max()
        )
        <= float(criteria.maximum_baseline_relative_rms_error),
    }
    robustness_checks = {
        "fundamental_excluded_movement": (
            float(excluded_primary["ci_2.5"]) > 0.0
            and float(excluded_primary["positive_seed_fraction"])
            >= minimum_positive
        ),
        "beyond_matched_sinusoid": (
            beyond_synthetic["ci_2.5"] > 0.0
        ),
    }

    # Export one transition per validation circuit and candidate policy action.
    baseline_context = _condition_epoch(epoch_frame, "A_async", "baseline")
    transition_rows = []
    for seed in validation_seeds:
        for condition, amplitude in (
            ("A_async", 0.0),
            (LOWER_ACTION, float(cfg.analysis.tacs.lower_amplitude_v_per_m)),
            (PRIMARY_ACTION, float(cfg.analysis.tacs.primary_amplitude_v_per_m)),
        ):
            reach = primary_seed_rows.loc[(seed, condition)]
            context = baseline_context.loc[seed]
            transition_rows.append(
                {
                    "seed": int(seed),
                    "action_id": condition,
                    "amplitude_v_per_m": amplitude,
                    "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
                    "phase_rad": float(cfg.analysis.tacs.phase_rad),
                    "context_primary_eeg_score": float(
                        context["primary_eeg_score"]
                    ),
                    **{
                        f"context_{feature}": float(context[feature])
                        for feature in primary_axis["feature_names"]
                    },
                    "outcome_primary_eeg_score": float(
                        reach["candidate_score"]
                    ),
                    "target_distance": float(
                        reach["candidate_distance_to_target"]
                    ),
                    "unpenalized_reward": -float(
                        reach["candidate_distance_to_target"]
                    ),
                }
            )
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(root / "contextual_bandit_transition_table.csv", index=False)
    action_means = transitions.groupby("action_id")["unpenalized_reward"].mean()
    best_fixed = str(action_means.idxmax())
    oracle = transitions.loc[
        transitions.groupby("seed")["unpenalized_reward"].idxmax()
    ]
    action_diagnostics = {
        "best_fixed_action": best_fixed,
        "mean_reward_by_action": {
            str(key): float(value) for key, value in action_means.items()
        },
        "oracle_best_action_counts": {
            str(key): int(value)
            for key, value in oracle["action_id"].value_counts().items()
        },
        "oracle_mean_reward": float(oracle["unpenalized_reward"].mean()),
        "best_fixed_mean_reward": float(action_means.max()),
        "oracle_advantage_over_best_fixed": float(
            oracle["unpenalized_reward"].mean() - action_means.max()
        ),
        "interpretation": (
            "Action heterogeneity is necessary but not sufficient for a "
            "contextual bandit; held-out EEG must predict it before RL."
        ),
    }
    with (root / "action_space_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(action_diagnostics, handle, indent=2)

    conclusion = {
        "scientific_scope": (
            "Observed-EEG reachability in an ideal artifact-free 40-cell toy "
            "circuit; not clinical simultaneous tACS-EEG validation."
        ),
        "papers_informing_design": [
            {
                "title": (
                    "Transcranial alternating current stimulation entrains "
                    "single-neuron activity in the primate brain"
                ),
                "design_constraint": (
                    "Treat timing modulation as mechanistic and firing rate "
                    "as a guardrail; expect heterogeneous weak-field effects."
                ),
            },
            {
                "title": (
                    "Entrainment of Brain Oscillations by Transcranial "
                    "Alternating Current Stimulation"
                ),
                "design_constraint": (
                    "Audit the stimulation-frequency EEG component and phase; "
                    "do not ignore the concurrent-EEG artifact problem."
                ),
            },
        ],
        "fixed_conditions": {
            "A": "homogeneous independent Poisson afferent timing",
            "B": {
                "modulation_depth": float(
                    cfg.analysis.reference.modulation_depth
                ),
                "frequency_hz": float(cfg.analysis.reference.frequency_hz),
                "mean_afferent_rate_changed": False,
            },
        },
        "fixed_actions": actions,
        "discovery_seeds": discovery_seeds,
        "validation_seeds": validation_seeds,
        "primary_endpoint": (
            "Seed-level reduction in distance from A to the frozen discovery-B "
            "centroid along an EEG-only standardized A-to-B axis."
        ),
        "reference_eeg_shift": reference_shift,
        "classification": classification,
        "primary_action_eeg_reachability": primary.to_dict(),
        "lower_dose_eeg_reachability": lower.to_dict(),
        "transverse_eeg_reachability": transverse.to_dict(),
        "synthetic_eeg_reachability": synthetic.to_dict(),
        "orientation_advantage": orientation_advantage,
        "fundamental_excluded_primary_action": excluded_primary.to_dict(),
        "advantage_beyond_matched_sinusoid": beyond_synthetic,
        "hidden_primary_E_ppc_gain": hidden_ppc,
        "hidden_primary_E_phase_difference_to_B": {
            "circular_mean_rad": float(np.angle(phase_complex)),
            "circular_mean_degrees": float(np.degrees(np.angle(phase_complex))),
            "resultant_length": float(np.abs(phase_complex)),
        },
        "ideal_eeg_checks": ideal_checks,
        "artifact_robustness_checks": robustness_checks,
        "ideal_eeg_reachability_passed": bool(all(ideal_checks.values())),
        "artifact_robust_eeg_reachability_passed": bool(
            all(ideal_checks.values()) and all(robustness_checks.values())
        ),
        "action_space_diagnostics": action_diagnostics,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    with (root / "experiment_conclusion.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(conclusion, handle, indent=2)
    if bool(cfg.experiment.plot):
        _plot_scores(epoch_frame, root)

    print("\n### EEG-primary reachability checks")
    for name, passed in ideal_checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Concurrent-EEG robustness checks")
    for name, passed in robustness_checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nIdeal simulated-EEG A -> B-like reachability: "
        f"{'PASSED' if conclusion['ideal_eeg_reachability_passed'] else 'NOT PASSED'}"
    )
    print(
        "Artifact-robust A -> B-like reachability: "
        f"{'PASSED' if conclusion['artifact_robust_eeg_reachability_passed'] else 'NOT PASSED'}"
    )
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
