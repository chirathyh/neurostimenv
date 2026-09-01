"""EEG-guided tACS suppression of an elevated-alpha BallAndStick toy state.

Condition A and condition B use identical cells, recurrence, mean afferent
rate, synaptic weights, and inhibition. A alone has weak 10-Hz modulation of
the independent Poisson afferent rates; B has homogeneous Poisson afferents.
The labels are operational toy states and are not depression/healthy labels.

The experiment first calibrates the smallest EEG-visible A generator without
stimulation. Disjoint seeds then discover one of four field phases relative to
the phase causally estimated from the preceding EEG. The frozen suppressive
offset is tested on held-out seeds against sham, the opposite phase, and a
transverse field. The primary endpoint is multi-second log alpha-band power;
spike PPC and rates are hidden mechanism/safety checks.

This establishes, at most, ideal neural-only EEG controllability. A complex
observation-only sinusoid audit is retained because the forward model does not
include the much larger artifact present in concurrent human tACS-EEG.
"""

from __future__ import annotations

import json
import random
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
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _band_power,
    _fourier_coefficients,
    _match_complex_observation,
    _process_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_entrainment_state import (  # noqa: E402
    _condition_config,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _analyze_episode,
    _bootstrap_ci,
    _environment_action,
    _episode_config,
    _mpi_variables,
    _relative_rate_safe,
    _sign_flip_p,
    _simulate_episode,
    _timeline,
    _validate_online_outputs,
    _zero_action,
)


B_LOW = "B_low_alpha_reference"
A_HIGH = "A_elevated_alpha"
SELECTED = "A_tacs_suppressive"
OPPOSITE = "A_tacs_opposite_phase"
TRANSVERSE = "A_tacs_transverse"
SYNTHETIC = "A_observation_only_complex_match"
EPOCHS = ("baseline", "stimulation", "washout")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _seeds(cfg: DictConfig, stage: str) -> list[int]:
    block = cfg.analysis[stage]
    first = int(cfg.experiment.seed) + int(block.seed_offset)
    return [first + index for index in range(int(block.n_seeds))]


def _wrap_phase(value: float) -> float:
    return float(np.mod(float(value), 2.0 * np.pi))


def _validate_design(cfg: DictConfig) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("Alpha suppression requires the online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("A, B, and every tACS arm require inhibition_scale=1.")
    reference_hz = float(cfg.analysis.reference.frequency_hz)
    action_hz = float(cfg.analysis.tacs.frequency_hz)
    if reference_hz <= 0.0 or not np.isclose(reference_hz, action_hz):
        raise ValueError("The toy alpha generator and tACS must both be 10 Hz.")
    if not 8.0 <= reference_hz <= 12.0:
        raise ValueError("The target frequency must lie in the alpha band.")

    depths = [float(x) for x in cfg.analysis.calibration.modulation_depths]
    envelope = float(cfg.analysis.reference.thinning_envelope_modulation_depth)
    if not depths or len(depths) != len(set(depths)):
        raise ValueError("Calibration depths must be nonempty and unique.")
    if any(x <= 0.0 or x > envelope for x in depths) or envelope > 1.0:
        raise ValueError("Every modulation depth must be in (0, envelope].")
    offsets = [
        _wrap_phase(x)
        for x in cfg.analysis.phase_discovery.relative_phase_offsets_rad
    ]
    if len(offsets) < 4 or len({round(x, 12) for x in offsets}) != len(offsets):
        raise ValueError("Use four or more unique EEG-relative phase offsets.")
    amplitude = float(cfg.analysis.tacs.amplitude_v_per_m)
    if not 0.0 < amplitude <= float(cfg.analysis.maximum_field_v_per_m):
        raise ValueError("tACS amplitude must be in (0, maximum field].")

    timeline = cfg.analysis.timeline
    if min(count for _, count in _timeline(cfg)) < 1:
        raise ValueError("Every online epoch requires at least one window.")
    window_ms = float(cfg.env.simulation.obs_win_len)
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    stimulation_ms = int(timeline.stimulation_steps) * window_ms
    if trim_ms < float(timeline.block_ramp_ms) or 2.0 * trim_ms >= stimulation_ms:
        raise ValueError("Analysis trim must contain both ramps and leave data.")

    blocks = [set(_seeds(cfg, name)) for name in (
        "calibration", "phase_discovery", "validation"
    )]
    if any(not block for block in blocks):
        raise ValueError("Every experiment stage requires at least one seed.")
    if any(blocks[i].intersection(blocks[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Calibration, discovery, and validation seeds must be disjoint.")
    if max(set.union(*blocks)) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Circuit seeds are too large for seed * 10,000 mapping.")


def _reference_phase(seed: int) -> float:
    """Randomize the hidden afferent phase across circuit seeds."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1_048_583]))
    return float(rng.uniform(0.0, 2.0 * np.pi))


def _condition_for_seed(
    cfg: DictConfig, *, seed: int, modulation_depth: float
) -> DictConfig:
    result = _condition_config(cfg, modulation_depth=float(modulation_depth))
    phase = _reference_phase(seed)
    with open_dict(result):
        for population in ("E", "I"):
            result.env.network.background[population].rhythm.phase_rad = phase
    return result


def _action(
    cfg: DictConfig,
    *,
    identifier: str,
    role: str,
    amplitude: float,
    montage: str,
    relative_offset: float | None = None,
) -> dict[str, Any]:
    result = {
        "id": identifier,
        "role": role,
        "montage": montage,
        "dc_offset_v_per_m": 0.0,
        "ac_amplitude_v_per_m": float(amplitude),
        "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
        "phase_rad": 0.0,
    }
    if relative_offset is not None:
        result["eeg_relative_phase_offset_rad"] = _wrap_phase(relative_offset)
    return result


def _sham(cfg: DictConfig, identifier: str) -> dict[str, Any]:
    return _action(
        cfg,
        identifier=identifier,
        role="unstimulated_state",
        amplitude=0.0,
        montage=str(cfg.analysis.tacs.axial_montage),
    )


def _field_phase_from_eeg_coefficients(
    cosine: float,
    sine: float,
    *,
    block_start_ms: float,
    frequency_hz: float,
    relative_offset_rad: float,
) -> tuple[float, float]:
    """Map baseline EEG phase to field phase at the intervention boundary.

    The EEG is represented as ``C cos(wt) + S sin(wt) = R cos(wt-phi)``.
    Offset zero makes the field sine have the same waveform phase as this EEG
    component at block onset. The screened offset then captures the unknown
    membrane/network transfer lag; anti-phase is not assumed a priori.
    """
    if np.hypot(cosine, sine) <= np.finfo(float).tiny:
        raise RuntimeError("The baseline 10-Hz EEG phase is undefined.")
    coefficient_phase = float(np.arctan2(sine, cosine))
    eeg_phase_at_block = _wrap_phase(
        2.0 * np.pi * float(frequency_hz) * float(block_start_ms) / 1000.0
        - coefficient_phase
    )
    field_phase = _wrap_phase(
        eeg_phase_at_block + np.pi / 2.0 + float(relative_offset_rad)
    )
    return field_phase, eeg_phase_at_block


def _estimate_relative_field_phase(
    outputs: list[dict[str, Any]],
    *,
    simulator_fs_hz: float,
    block_start_ms: float,
    relative_offset_rad: float,
    cfg: DictConfig,
) -> dict[str, float]:
    raw = np.concatenate([
        np.asarray(output["eeg_v"], dtype=float).reshape(-1) for output in outputs
    ])
    processed, fs_hz, _, _, features = _process_eeg(
        raw, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    start_ms = float(outputs[0]["t_start_ms"])
    cosine, sine = _fourier_coefficients(
        processed,
        fs_hz=fs_hz,
        start_ms=start_ms,
        frequency_hz=float(cfg.analysis.tacs.frequency_hz),
    )
    field_phase, eeg_phase = _field_phase_from_eeg_coefficients(
        cosine,
        sine,
        block_start_ms=block_start_ms,
        frequency_hz=float(cfg.analysis.tacs.frequency_hz),
        relative_offset_rad=relative_offset_rad,
    )
    return {
        "phase_rad": field_phase,
        "baseline_eeg_phase_at_block_rad": eeg_phase,
        "baseline_eeg_10hz_resultant_v": float(np.hypot(cosine, sine)),
        "baseline_eeg_rms_v": float(features["rms_v"]),
    }


def _phase_estimation_outputs(
    outputs: list[dict[str, Any]], cfg: DictConfig
) -> list[dict[str, Any]]:
    """Select the causal baseline tail used to initialize field phase.

    Historical experiments estimate phase from the complete baseline because
    their afferent carrier is stationary.  D1 studies a phase-diffusing
    carrier, for which an old phase estimate is not relevant at the action
    boundary.  The optional ``analysis.tacs.phase_estimation_steps`` setting
    therefore selects only the most recent complete online windows.  Omitting
    the setting preserves the historical behavior exactly.
    """
    requested = OmegaConf.select(
        cfg, "analysis.tacs.phase_estimation_steps", default=None
    )
    if requested is None:
        return outputs
    count = int(requested)
    if count < 1:
        raise ValueError("phase_estimation_steps must be a positive integer.")
    if count > len(outputs):
        raise ValueError(
            "phase_estimation_steps exceeds the available baseline windows."
        )
    return outputs[-count:]


def _simulate_relative_phase_episode(
    base_cfg: DictConfig,
    *,
    seed: int,
    action: dict[str, Any],
    output_dir: Path,
    comm,
    size: int,
    rank: int,
    structure_seed: int | None = None,
    drive_seed: int | None = None,
    future_drive_seed: int | None = None,
) -> dict[str, Any] | None:
    """Run one persistent episode and choose phase after observing baseline."""
    run_cfg = _episode_config(base_cfg, seed=seed, output_dir=output_dir)
    structure = int(seed if structure_seed is None else structure_seed)
    drive = int(seed if drive_seed is None else drive_seed)
    schedule = _timeline(run_cfg)
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    pre_steps = int(run_cfg.analysis.timeline.burn_in_steps) + int(
        run_cfg.analysis.timeline.baseline_steps
    )
    block_start_ms = pre_steps * window_ms
    block_stop_ms = block_start_ms + int(
        run_cfg.analysis.timeline.stimulation_steps
    ) * window_ms
    envelope = {
        "start_ms": block_start_ms,
        "stop_ms": block_stop_ms,
        "ramp_ms": float(run_cfg.analysis.timeline.block_ramp_ms),
    }
    np.random.seed(structure * 10_000 + rank)
    random.seed(structure * 10_000 + rank)
    environment = OnlineNeuronEnv(
        run_cfg,
        _mpi_variables(
            comm,
            size,
            rank,
            seed,
            structure_seed=structure,
            drive_seed=drive,
            future_drive_seed=future_drive_seed,
            future_start_ms=(
                block_start_ms if future_drive_seed is not None else None
            ),
        ),
        ENV_SEED=0,
    )
    outputs = {name: [] for name, _ in schedule} if rank == 0 else None
    zero = _zero_action(run_cfg)
    diagnostics: dict[str, float] | None = None
    realized = dict(action)
    final_residual_mV = float("nan")
    try:
        for epoch, count in schedule:
            if epoch == "stimulation":
                if rank == 0:
                    phase_outputs = _phase_estimation_outputs(
                        outputs["baseline"], run_cfg
                    )
                    diagnostics = _estimate_relative_field_phase(
                        phase_outputs,
                        simulator_fs_hz=1000.0 / float(run_cfg.env.network.dt),
                        block_start_ms=block_start_ms,
                        relative_offset_rad=float(
                            action["eeg_relative_phase_offset_rad"]
                        ),
                        cfg=run_cfg,
                    )
                    diagnostics["phase_estimation_steps_used"] = int(
                        len(phase_outputs)
                    )
                    diagnostics["phase_estimation_start_ms"] = float(
                        phase_outputs[0]["t_start_ms"]
                    )
                    diagnostics["phase_estimation_stop_ms"] = float(
                        phase_outputs[-1]["t_stop_ms"]
                    )
                diagnostics = comm.bcast(diagnostics, root=0)
                realized["phase_rad"] = float(diagnostics["phase_rad"])
                environment_action = _environment_action({
                    key: value
                    for key, value in realized.items()
                    if key != "eeg_relative_phase_offset_rad"
                })
            for epoch_step in range(count):
                is_active = epoch == "stimulation"
                step_action = dict(environment_action if is_active else zero)
                if is_active and epoch_step > 0:
                    step_action.pop("phase_rad", None)
                output = environment.step_online(
                    step_action,
                    duration_ms=window_ms,
                    phase_continuous=True,
                    ramp_ms=0.0,
                    block_envelope=envelope if is_active else None,
                )
                if rank == 0:
                    outputs[epoch].append(output)
        if rank == 0:
            final_residual_mV = environment.stimulation_controller.max_abs_extracellular(
                environment.network
            )
    finally:
        environment.close()

    if rank != 0:
        return None
    _validate_online_outputs(outputs)
    if final_residual_mV != 0.0:
        raise RuntimeError("Washout left residual extracellular voltage.")
    return {
        "seed": int(seed),
        "structure_seed": structure,
        "drive_seed": drive,
        "future_drive_seed": (
            drive if future_drive_seed is None else int(future_drive_seed)
        ),
        "future_start_ms": (
            None if future_drive_seed is None else float(block_start_ms)
        ),
        "action": realized,
        "stimulate": True,
        "block_start_ms": float(block_start_ms),
        "outputs_by_epoch": outputs,
        "final_residual_mV": float(final_residual_mV),
        **diagnostics,
    }


def _epoch_raw(episode: dict[str, Any], epoch: str) -> np.ndarray:
    return np.asarray(episode["raw_by_epoch"][epoch], dtype=np.float64)


def _feature_from_raw(
    raw: np.ndarray,
    *,
    simulator_fs_hz: float,
    start_ms: float,
    cfg: DictConfig,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    processed, fs_hz, frequencies, psd, features = _process_eeg(
        raw, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    low = float(cfg.analysis.alpha_low_hz)
    high = float(cfg.analysis.alpha_high_hz)
    alpha_power = _band_power(
        frequencies, psd, center_hz=(low + high) / 2.0,
        half_width_hz=(high - low) / 2.0,
    )
    cosine, sine = _fourier_coefficients(
        processed,
        fs_hz=fs_hz,
        start_ms=start_ms,
        frequency_hz=float(cfg.analysis.tacs.frequency_hz),
    )
    eps = np.finfo(float).tiny
    keep = ((frequencies >= 1.0) & (frequencies <= 80.0)
            & ((frequencies < low) | (frequencies > high)))
    excluded_power = float(np.trapz(psd[keep], frequencies[keep]))
    target_index = int(np.argmin(np.abs(frequencies - float(cfg.analysis.tacs.frequency_hz))))
    flank = (((frequencies >= 6.0) & (frequencies <= 7.5))
             | ((frequencies >= 12.5) & (frequencies <= 14.0)))
    flank_level = float(np.median(psd[flank])) if np.any(flank) else eps
    nperseg = min(processed.size, max(256, int(round(2.0 * fs_hz))))
    step = max(1, nperseg // 2)
    n_segments = 1 + max(0, (processed.size - nperseg) // step)
    result = {
        **features,
        "alpha_power_8_12_hz": alpha_power,
        "log10_alpha_power_8_12_hz": float(np.log10(max(alpha_power, eps))),
        "eeg_10hz_cosine_v": cosine,
        "eeg_10hz_sine_v": sine,
        "eeg_10hz_resultant_v": float(np.hypot(cosine, sine)),
        "eeg_10hz_phase_rad": float(np.arctan2(sine, cosine)),
        "alpha_peak_prominence_db": float(
            10.0 * np.log10(max(float(psd[target_index]), eps) / max(flank_level, eps))
        ),
        "log10_power_excluding_alpha": float(np.log10(max(excluded_power, eps))),
        "analysis_duration_s": float(processed.size / fs_hz),
        "welch_frequency_resolution_hz": float(fs_hz / nperseg),
        "welch_segment_count": int(n_segments),
    }
    return result, processed, frequencies, psd


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
    structure_seed: int | None = None,
    drive_seed: int | None = None,
    future_drive_seed: int | None = None,
    phase_seed: int | None = None,
) -> dict[str, Any] | None:
    if stimulate:
        simulation = _simulate_relative_phase_episode(
            condition_cfg,
            seed=seed,
            action=action,
            output_dir=output_dir,
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=structure_seed,
            drive_seed=drive_seed,
            future_drive_seed=future_drive_seed,
        )
    else:
        simulation = _simulate_episode(
            condition_cfg,
            seed=seed,
            action=action,
            stimulate=False,
            output_dir=output_dir,
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=structure_seed,
            drive_seed=drive_seed,
        )
    if rank != 0:
        return None
    realized_action = simulation["action"]
    epoch_rows, raw_by_epoch = _analyze_episode(
        simulation,
        action=realized_action,
        action_index=action_index,
        arm=condition_id,
        cfg=condition_cfg,
        output_dir=output_dir / "analysis",
    )
    episode = {
        "simulation": simulation,
        "epoch_rows": epoch_rows,
        "raw_by_epoch": raw_by_epoch,
        "simulator_fs_hz": 1000.0 / float(condition_cfg.env.network.dt),
    }
    trim_ms = float(condition_cfg.analysis.timeline.stimulation_analysis_trim_ms)
    dt = float(condition_cfg.env.network.dt)
    for row in epoch_rows:
        epoch = str(row["epoch"])
        raw = _epoch_raw(episode, epoch)
        outputs = simulation["outputs_by_epoch"][epoch]
        start_ms = float(outputs[0]["t_start_ms"])
        if epoch == "stimulation" and trim_ms > 0.0:
            trim_samples = int(round(trim_ms / dt))
            raw = raw[trim_samples:-trim_samples]
            start_ms += trim_samples * dt
        feature, _, _, _ = _feature_from_raw(
            raw,
            simulator_fs_hz=float(episode["simulator_fs_hz"]),
            start_ms=start_ms,
            cfg=condition_cfg,
        )
        row.update(feature)
        row["condition_id"] = condition_id
        row["input_phase_rad"] = _reference_phase(
            seed if phase_seed is None else phase_seed
        )
        row["structure_seed"] = int(
            seed if structure_seed is None else structure_seed
        )
        row["drive_seed"] = int(seed if drive_seed is None else drive_seed)
        row["future_drive_seed"] = int(
            (seed if drive_seed is None else drive_seed)
            if future_drive_seed is None else future_drive_seed
        )
        row["phase_seed"] = int(seed if phase_seed is None else phase_seed)
        row["eeg_relative_phase_offset_rad"] = float(
            realized_action.get("eeg_relative_phase_offset_rad", np.nan)
        )
        row["baseline_eeg_phase_at_block_rad"] = float(
            simulation.get("baseline_eeg_phase_at_block_rad", np.nan)
        )
    return episode


def _epoch_row(episode: dict[str, Any], epoch: str = "stimulation") -> pd.Series:
    return pd.Series(next(row for row in episode["epoch_rows"] if row["epoch"] == epoch))


def _summary(values: np.ndarray, cfg: DictConfig, rng: np.random.Generator) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    low, high = _bootstrap_ci(values, rng=rng, n_bootstrap=int(cfg.analysis.n_bootstrap))
    return {
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "ci_2.5": float(low),
        "ci_97.5": float(high),
        "positive_seed_count": int(np.count_nonzero(values > 0.0)),
        "positive_seed_fraction": float(np.mean(values > 0.0)),
        "paired_sign_flip_p": float(_sign_flip_p(
            values, rng=rng, n_permutations=int(cfg.analysis.n_permutations)
        )),
    }


def _calibration_metrics(rows: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    b = rows[rows.condition_id.eq(B_LOW)].set_index("seed")
    output = []
    for _, a in rows[rows.condition_id.eq(A_HIGH)].iterrows():
        seed = int(a.seed)
        reference = b.loc[seed]
        output.append({
            "seed": seed,
            "modulation_depth": float(a.modulation_depth),
            "A_minus_B_log10_alpha_power": float(
                a.log10_alpha_power_8_12_hz - reference.log10_alpha_power_8_12_hz
            ),
            "A_minus_B_10hz_resultant_v": float(
                a.eeg_10hz_resultant_v - reference.eeg_10hz_resultant_v
            ),
            "A_minus_B_alpha_prominence_db": float(
                a.alpha_peak_prominence_db - reference.alpha_peak_prominence_db
            ),
            "rate_matched": bool(_relative_rate_safe(a, reference, cfg)),
        })
    return pd.DataFrame(output)


def _calibration_summary(
    metrics: pd.DataFrame, cfg: DictConfig, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for depth, group in metrics.groupby("modulation_depth", sort=True):
        rows.append({
            "modulation_depth": float(depth),
            **_summary(group.A_minus_B_log10_alpha_power.to_numpy(float), cfg, rng),
            "rate_matched_fraction": float(group.rate_matched.mean()),
            "mean_10hz_resultant_shift_v": float(group.A_minus_B_10hz_resultant_v.mean()),
            "mean_alpha_prominence_shift_db": float(group.A_minus_B_alpha_prominence_db.mean()),
        })
    return pd.DataFrame(rows)


def _select_reference_depth(summary: pd.DataFrame, cfg: DictConfig) -> tuple[float, bool]:
    criteria = cfg.analysis.criteria
    qualified = summary[
        (summary["mean"] >= float(criteria.minimum_reference_log10_alpha_shift))
        & (summary.positive_seed_count >= int(criteria.minimum_calibration_positive_seeds))
        & np.isclose(summary.rate_matched_fraction, 1.0)
    ]
    if not qualified.empty:
        return float(qualified.sort_values("modulation_depth").iloc[0].modulation_depth), True
    safe = summary[np.isclose(summary.rate_matched_fraction, 1.0)]
    pool = safe if not safe.empty else summary
    selected = pool.sort_values(["mean", "modulation_depth"], ascending=[False, True]).iloc[0]
    return float(selected.modulation_depth), False


def _alpha_target_model(calibration_rows: pd.DataFrame, selected_depth: float) -> dict[str, float]:
    selected = calibration_rows[
        calibration_rows.condition_id.eq(B_LOW)
        | (calibration_rows.condition_id.eq(A_HIGH)
           & np.isclose(calibration_rows.modulation_depth, selected_depth))
    ]
    a = selected[selected.condition_id.eq(A_HIGH)].log10_alpha_power_8_12_hz.to_numpy(float)
    b = selected[selected.condition_id.eq(B_LOW)].log10_alpha_power_8_12_hz.to_numpy(float)
    pooled = np.concatenate((a, b))
    scale = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 1.0
    if scale <= np.finfo(float).eps:
        scale = max(abs(float(np.mean(a) - np.mean(b))), 1.0e-6)
    threshold = float((np.mean(a) + np.mean(b)) / 2.0)
    return {
        "A_mean_log10_alpha": float(np.mean(a)),
        "B_mean_log10_alpha": float(np.mean(b)),
        "pooled_scale": scale,
        "classification_threshold": threshold,
        "A_is_above_threshold": bool(np.mean(a) > np.mean(b)),
    }


def _target_distance(value: float, model: dict[str, float]) -> float:
    return abs(float(value) - float(model["B_mean_log10_alpha"])) / float(model["pooled_scale"])


def _discovery_metric(
    sham: pd.Series,
    active: pd.Series,
    *,
    offset: float,
    model: dict[str, float],
    cfg: DictConfig,
) -> dict[str, Any]:
    sham_value = float(sham.log10_alpha_power_8_12_hz)
    active_value = float(active.log10_alpha_power_8_12_hz)
    return {
        "seed": int(sham.seed),
        "relative_phase_offset_rad": _wrap_phase(offset),
        "relative_phase_offset_deg": float(np.degrees(_wrap_phase(offset))),
        "alpha_suppression_log10": sham_value - active_value,
        "target_distance_improvement": _target_distance(sham_value, model)
        - _target_distance(active_value, model),
        "coherent_10hz_suppression_v": float(
            sham.eeg_10hz_resultant_v - active.eeg_10hz_resultant_v
        ),
        "E_ppc_reduction": float(sham.E_ppc - active.E_ppc),
        "E_rate_change_hz": float(active.E_firing_rate_hz - sham.E_firing_rate_hz),
        "I_rate_change_hz": float(active.I_firing_rate_hz - sham.I_firing_rate_hz),
        "rate_safe": bool(_relative_rate_safe(active, sham, cfg)),
    }


def _discovery_summary(
    metrics: pd.DataFrame, cfg: DictConfig, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for offset, group in metrics.groupby("relative_phase_offset_rad", sort=True):
        rows.append({
            "relative_phase_offset_rad": float(offset),
            "relative_phase_offset_deg": float(np.degrees(offset)),
            **_summary(group.alpha_suppression_log10.to_numpy(float), cfg, rng),
            "mean_target_distance_improvement": float(group.target_distance_improvement.mean()),
            "mean_coherent_10hz_suppression_v": float(group.coherent_10hz_suppression_v.mean()),
            "mean_E_ppc_reduction": float(group.E_ppc_reduction.mean()),
            "rate_safe_fraction": float(group.rate_safe.mean()),
        })
    return pd.DataFrame(rows)


def _select_suppressive_offset(summary: pd.DataFrame) -> tuple[float, bool]:
    safe = summary[np.isclose(summary.rate_safe_fraction, 1.0)]
    pool = safe if not safe.empty else summary
    selected = pool.sort_values(
        ["mean", "mean_target_distance_improvement", "relative_phase_offset_rad"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(selected.relative_phase_offset_rad), bool(float(selected["mean"]) > 0.0)


def _two_second_rows(
    episode: dict[str, Any], *, condition_id: str, cfg: DictConfig
) -> list[dict[str, Any]]:
    outputs = episode["simulation"]["outputs_by_epoch"]["stimulation"]
    result = []
    for index in range(0, len(outputs) - 1, 2):
        pair = outputs[index:index + 2]
        raw = np.concatenate([np.asarray(x["eeg_v"]).reshape(-1) for x in pair])
        feature, _, _, _ = _feature_from_raw(
            raw,
            simulator_fs_hz=float(episode["simulator_fs_hz"]),
            start_ms=float(pair[0]["t_start_ms"]),
            cfg=cfg,
        )
        result.append({
            "seed": int(episode["simulation"]["seed"]),
            "condition_id": condition_id,
            "bin_index": index // 2,
            "t_start_ms": float(pair[0]["t_start_ms"]),
            "t_stop_ms": float(pair[-1]["t_stop_ms"]),
            **feature,
        })
    return result


def _synthetic_feature(
    sham_episode: dict[str, Any], active_episode: dict[str, Any], cfg: DictConfig
) -> tuple[dict[str, float], dict[str, float]]:
    raw, diagnostics = _match_complex_observation(
        sham_episode,
        active_episode,
        selected_frequency_hz=float(cfg.analysis.tacs.frequency_hz),
        cfg=cfg,
    )
    outputs = sham_episode["simulation"]["outputs_by_epoch"]["stimulation"]
    dt = float(cfg.env.network.dt)
    trim_samples = int(round(float(cfg.analysis.timeline.stimulation_analysis_trim_ms) / dt))
    start_ms = float(outputs[0]["t_start_ms"]) + trim_samples * dt
    trimmed = raw[trim_samples:-trim_samples]
    feature, _, _, _ = _feature_from_raw(
        trimmed,
        simulator_fs_hz=float(sham_episode["simulator_fs_hz"]),
        start_ms=start_ms,
        cfg=cfg,
    )
    return feature, diagnostics


def _validation_seed_metric(
    *,
    seed: int,
    episodes: dict[str, dict[str, Any]],
    model: dict[str, float],
    cfg: DictConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = {name: _epoch_row(episode) for name, episode in episodes.items()}
    a, b = rows[A_HIGH], rows[B_LOW]
    selected, opposite, transverse = rows[SELECTED], rows[OPPOSITE], rows[TRANSVERSE]
    a_value = float(a.log10_alpha_power_8_12_hz)
    b_value = float(b.log10_alpha_power_8_12_hz)
    selected_value = float(selected.log10_alpha_power_8_12_hz)
    opposite_value = float(opposite.log10_alpha_power_8_12_hz)
    transverse_value = float(transverse.log10_alpha_power_8_12_hz)
    synthetic, synthetic_diagnostics = _synthetic_feature(
        episodes[A_HIGH], episodes[SELECTED], cfg
    )
    real_improvement = abs(a_value - b_value) - abs(selected_value - b_value)
    synthetic_improvement = abs(a_value - b_value) - abs(
        float(synthetic["log10_alpha_power_8_12_hz"]) - b_value
    )
    selected_baseline = _epoch_row(episodes[SELECTED], "baseline")
    selected_washout = _epoch_row(episodes[SELECTED], "washout")
    a_baseline = _epoch_row(episodes[A_HIGH], "baseline")
    a_washout = _epoch_row(episodes[A_HIGH], "washout")
    stimulation_effect = a_value - selected_value
    washout_effect = float(
        (a_washout.log10_alpha_power_8_12_hz - a_baseline.log10_alpha_power_8_12_hz)
        - (selected_washout.log10_alpha_power_8_12_hz
           - selected_baseline.log10_alpha_power_8_12_hz)
    )
    residual_limit = float(cfg.analysis.criteria.maximum_washout_residual_fraction)
    baseline_errors = {
        name: _relative_rms_error(
            _epoch_raw(episodes[A_HIGH], "baseline"),
            _epoch_raw(episodes[name], "baseline"),
        )
        for name in (SELECTED, OPPOSITE, TRANSVERSE)
    }
    metric = {
        "seed": int(seed),
        "A_minus_B_log10_alpha_power": a_value - b_value,
        "selected_alpha_suppression_log10": stimulation_effect,
        "selected_target_distance_improvement_log10": real_improvement,
        "selected_fractional_target_distance_improvement": (
            real_improvement / abs(a_value - b_value)
            if not np.isclose(a_value, b_value) else 0.0
        ),
        "selected_vs_opposite_phase_advantage_log10": opposite_value - selected_value,
        "selected_vs_transverse_advantage_log10": transverse_value - selected_value,
        "selected_10hz_resultant_suppression_v": float(
            a.eeg_10hz_resultant_v - selected.eeg_10hz_resultant_v
        ),
        "selected_alpha_prominence_reduction_db": float(
            a.alpha_peak_prominence_db - selected.alpha_peak_prominence_db
        ),
        "selected_E_ppc_reduction": float(a.E_ppc - selected.E_ppc),
        "selected_I_ppc_reduction": float(a.I_ppc - selected.I_ppc),
        "selected_E_rate_change_hz": float(selected.E_firing_rate_hz - a.E_firing_rate_hz),
        "selected_I_rate_change_hz": float(selected.I_firing_rate_hz - a.I_firing_rate_hz),
        "reference_rate_matched": bool(_relative_rate_safe(a, b, cfg)),
        "selected_rate_safe": bool(_relative_rate_safe(selected, a, cfg)),
        "opposite_rate_safe": bool(_relative_rate_safe(opposite, a, cfg)),
        "transverse_rate_safe": bool(_relative_rate_safe(transverse, a, cfg)),
        "washout_effect_log10": washout_effect,
        "washout_recovered": bool(
            stimulation_effect > 0.0
            and abs(washout_effect) <= residual_limit * max(abs(stimulation_effect), np.finfo(float).eps)
        ),
        "maximum_baseline_relative_rms_error": float(max(baseline_errors.values())),
        "A_minus_B_log10_power_excluding_alpha": float(
            a.log10_power_excluding_alpha - b.log10_power_excluding_alpha
        ),
        "selected_excluded_power_movement": float(
            abs(a.log10_power_excluding_alpha - b.log10_power_excluding_alpha)
            - abs(selected.log10_power_excluding_alpha - b.log10_power_excluding_alpha)
        ),
        "synthetic_target_distance_improvement_log10": synthetic_improvement,
        "real_beyond_synthetic_improvement_log10": real_improvement - synthetic_improvement,
        "synthetic_peak_added_v": float(synthetic_diagnostics["peak_added_v"]),
    }
    return metric, {"seed": int(seed), **synthetic_diagnostics, **synthetic}


def _metric_summaries(
    metrics: pd.DataFrame, cfg: DictConfig, rng: np.random.Generator
) -> pd.DataFrame:
    names = [
        "A_minus_B_log10_alpha_power",
        "selected_alpha_suppression_log10",
        "selected_target_distance_improvement_log10",
        "selected_vs_opposite_phase_advantage_log10",
        "selected_vs_transverse_advantage_log10",
        "selected_10hz_resultant_suppression_v",
        "selected_alpha_prominence_reduction_db",
        "selected_E_ppc_reduction",
        "A_minus_B_log10_power_excluding_alpha",
        "selected_excluded_power_movement",
        "real_beyond_synthetic_improvement_log10",
    ]
    return pd.DataFrame([
        {"metric": name, **_summary(metrics[name].to_numpy(float), cfg, rng)}
        for name in names
    ])


def _classify(value: float, model: dict[str, float]) -> str:
    above = float(value) > float(model["classification_threshold"])
    is_a = above == bool(model["A_is_above_threshold"])
    return "A" if is_a else "B"


def _check_table(
    *,
    cfg: DictConfig,
    calibration_passed: bool,
    discovery_summary: pd.DataFrame,
    selected_offset: float,
    validation: pd.DataFrame,
    summaries: pd.DataFrame,
    epoch_rows: pd.DataFrame,
    two_second: pd.DataFrame,
    model: dict[str, float],
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    summary = summaries.set_index("metric")
    selected_discovery = discovery_summary[
        np.isclose(discovery_summary.relative_phase_offset_rad, selected_offset)
    ].iloc[0]
    required = int(criteria.minimum_validation_positive_seeds)
    validation_reference = epoch_rows[
        epoch_rows.condition_id.isin((A_HIGH, B_LOW))
        & epoch_rows.epoch.eq("stimulation")
    ]
    expected = np.where(validation_reference.condition_id.eq(A_HIGH), "A", "B")
    predictions = [
        _classify(x, model)
        for x in validation_reference.log10_alpha_power_8_12_hz.to_numpy(float)
    ]
    classification_accuracy = float(np.mean(np.asarray(predictions) == expected))
    reference_bins = two_second[two_second.condition_id.isin((A_HIGH, B_LOW))]
    bin_expected = np.where(reference_bins.condition_id.eq(A_HIGH), "A", "B")
    bin_predictions = [
        _classify(x, model)
        for x in reference_bins.log10_alpha_power_8_12_hz.to_numpy(float)
    ]
    bin_accuracy = float(np.mean(np.asarray(bin_predictions) == bin_expected))

    positive = lambda name: int(summary.loc[name, "positive_seed_count"]) >= required
    checks = {
        "minimum_calibration_seeds": len(_seeds(cfg, "calibration")) >= int(criteria.minimum_calibration_seeds),
        "minimum_discovery_seeds": len(_seeds(cfg, "phase_discovery")) >= int(criteria.minimum_discovery_seeds),
        "minimum_validation_seeds": len(validation) >= int(criteria.minimum_validation_seeds),
        "calibrated_elevated_alpha_state": bool(calibration_passed),
        "heldout_elevated_alpha_state": bool(
            float(summary.loc["A_minus_B_log10_alpha_power", "mean"]) > 0.0
            and positive("A_minus_B_log10_alpha_power")
        ),
        "heldout_reference_classification": classification_accuracy >= 0.75,
        "two_second_eeg_observable": bin_accuracy >= 0.65,
        "phase_discovery_positive": bool(
            float(selected_discovery["mean"]) > 0.0
            and int(selected_discovery.positive_seed_count)
            >= int(criteria.minimum_discovery_positive_seeds)
        ),
        "frozen_phase_reduces_alpha": bool(
            float(summary.loc["selected_alpha_suppression_log10", "mean"]) > 0.0
            and positive("selected_alpha_suppression_log10")
        ),
        "frozen_phase_moves_eeg_toward_B": bool(
            float(summary.loc["selected_target_distance_improvement_log10", "mean"]) > 0.0
            and positive("selected_target_distance_improvement_log10")
        ),
        "phase_specific": bool(
            float(summary.loc["selected_vs_opposite_phase_advantage_log10", "mean"]) > 0.0
            and positive("selected_vs_opposite_phase_advantage_log10")
        ),
        "orientation_specific": bool(
            float(summary.loc["selected_vs_transverse_advantage_log10", "mean"]) > 0.0
            and positive("selected_vs_transverse_advantage_log10")
        ),
        "coherent_10hz_component_reduced": bool(
            float(summary.loc["selected_10hz_resultant_suppression_v", "mean"]) > 0.0
            and positive("selected_10hz_resultant_suppression_v")
        ),
        "alpha_peak_prominence_reduced": bool(
            float(summary.loc["selected_alpha_prominence_reduction_db", "mean"]) > 0.0
            and positive("selected_alpha_prominence_reduction_db")
        ),
        "hidden_spike_synchrony_reduced": bool(
            float(summary.loc["selected_E_ppc_reduction", "mean"]) > 0.0
            and positive("selected_E_ppc_reduction")
        ),
        "reference_rate_matched": bool(validation.reference_rate_matched.all()),
        "all_tacs_arms_rate_safe": bool(
            validation[["selected_rate_safe", "opposite_rate_safe", "transverse_rate_safe"]].all().all()
        ),
        "washout_reversible": int(validation.washout_recovered.sum()) >= required,
        "baseline_causality": bool(
            validation.maximum_baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "beyond_complex_observation": bool(
            float(summary.loc["real_beyond_synthetic_improvement_log10", "mean"]) > 0.0
            and positive("real_beyond_synthetic_improvement_log10")
        ),
        "reference_observable_outside_alpha": bool(
            abs(float(summary.loc["A_minus_B_log10_power_excluding_alpha", "mean"])) > 0.02
            and int(summary.loc["A_minus_B_log10_power_excluding_alpha", "positive_seed_count"])
            in (0, len(validation))
        ),
    }
    core = [
        "calibrated_elevated_alpha_state", "heldout_elevated_alpha_state",
        "heldout_reference_classification", "phase_discovery_positive",
        "frozen_phase_reduces_alpha", "frozen_phase_moves_eeg_toward_B",
        "phase_specific", "orientation_specific",
        "coherent_10hz_component_reduced", "alpha_peak_prominence_reduced",
        "hidden_spike_synchrony_reduced",
        "reference_rate_matched", "all_tacs_arms_rate_safe",
        "washout_reversible", "baseline_causality",
    ]
    sample_checks = [
        "minimum_calibration_seeds", "minimum_discovery_seeds", "minimum_validation_seeds"
    ]
    ideal_poc = all(checks[name] for name in core + sample_checks)
    confirmatory = bool(
        ideal_poc and len(validation) >= 8
        and float(summary.loc["selected_alpha_suppression_log10", "ci_2.5"]) > 0.0
        and float(summary.loc["selected_target_distance_improvement_log10", "ci_2.5"]) > 0.0
    )
    conclusions = {
        "ideal_neural_eeg_directional_proof_of_concept": ideal_poc,
        "confirmatory_heldout_evidence": confirmatory,
        "artifact_robust_concurrent_eeg_claim": bool(
            ideal_poc and checks["beyond_complex_observation"]
            and checks["reference_observable_outside_alpha"]
        ),
        "heldout_reference_classification_accuracy": classification_accuracy,
        "two_second_reference_classification_accuracy": bin_accuracy,
    }
    return checks, conclusions


def _plot_results(
    *,
    root: Path,
    calibration: pd.DataFrame,
    discovery: pd.DataFrame,
    selected_offset: float,
    validation_rows: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    episodes: dict[int, dict[str, dict[str, Any]]],
    cfg: DictConfig,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for seed, group in calibration.groupby("seed"):
        axis.plot(group.modulation_depth, group.A_minus_B_log10_alpha_power,
                  marker="o", alpha=0.55, label=f"seed {seed}")
    mean = calibration.groupby("modulation_depth").A_minus_B_log10_alpha_power.mean()
    axis.plot(mean.index, mean.values, color="black", marker="o", linewidth=2.4, label="mean")
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.set(xlabel="Afferent modulation depth", ylabel="A - B log10 alpha power",
             title="Unstimulated elevated-alpha calibration")
    axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(root / "figure_01_state_calibration.png", dpi=250); plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for seed, group in discovery.groupby("seed"):
        group = group.sort_values("relative_phase_offset_deg")
        axis.plot(group.relative_phase_offset_deg, group.alpha_suppression_log10,
                  marker="o", alpha=0.55)
    mean = discovery.groupby("relative_phase_offset_deg").alpha_suppression_log10.mean()
    axis.plot(mean.index, mean.values, color="black", marker="o", linewidth=2.4, label="mean")
    axis.axvline(np.degrees(selected_offset), color="#D62728", linestyle="--", label="frozen")
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.set(xlabel="Field phase offset from ongoing EEG (degrees)",
             ylabel="Alpha suppression (log10 sham - tACS)", title="EEG-relative phase discovery")
    axis.legend(); figure.tight_layout(); figure.savefig(root / "figure_02_phase_discovery.png", dpi=250); plt.close(figure)

    conditions = [B_LOW, A_HIGH, SELECTED, OPPOSITE, TRANSVERSE]
    labels = ["B low-alpha", "A elevated-alpha", "A + selected", "A + opposite", "A + transverse"]
    colors = ["#2CA02C", "#9467BD", "#E67E22", "#1F77B4", "#777777"]
    psds: dict[str, list[np.ndarray]] = {name: [] for name in conditions}
    frequencies = None
    trim_samples = int(round(float(cfg.analysis.timeline.stimulation_analysis_trim_ms) / float(cfg.env.network.dt)))
    for seed_episodes in episodes.values():
        for condition in conditions:
            episode = seed_episodes[condition]
            raw = _epoch_raw(episode, "stimulation")[trim_samples:-trim_samples]
            start = float(episode["simulation"]["outputs_by_epoch"]["stimulation"][0]["t_start_ms"])
            start += float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
            _, _, frequencies, psd = _feature_from_raw(
                raw, simulator_fs_hz=float(episode["simulator_fs_hz"]), start_ms=start, cfg=cfg
            )
            psds[condition].append(psd)
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for condition, label, color in zip(conditions, labels, colors):
        values = np.asarray(psds[condition])
        axis.plot(frequencies, 10.0 * np.log10(np.maximum(values.mean(axis=0), np.finfo(float).tiny)),
                  color=color, label=label)
    axis.axvspan(8.0, 12.0, color="gold", alpha=0.15)
    axis.set_xlim(2.0, 25.0); axis.set(xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)",
        title="Held-out ideal EEG spectra during intervention")
    axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(root / "figure_03_validation_psd.png", dpi=250); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    wide = validation_rows[validation_rows.epoch.eq("stimulation")].pivot(
        index="seed", columns="condition_id", values="log10_alpha_power_8_12_hz"
    )
    for _, row in wide.iterrows():
        axes[0].plot(range(len(conditions)), [row[x] for x in conditions], color="0.75", linewidth=1)
    axes[0].plot(range(len(conditions)), [wide[x].mean() for x in conditions], color="black", marker="o", linewidth=2)
    axes[0].set_xticks(range(len(conditions)), ["B", "A", "selected", "opposite", "transverse"], rotation=25)
    axes[0].set_ylabel("log10 alpha power"); axes[0].set_title("Paired held-out EEG endpoint")
    axes[1].scatter(validation_metrics.selected_E_ppc_reduction,
                    validation_metrics.selected_alpha_suppression_log10, color="#E67E22")
    axes[1].axhline(0, color="0.5", linewidth=0.8); axes[1].axvline(0, color="0.5", linewidth=0.8)
    axes[1].set(xlabel="Hidden E-PPC reduction", ylabel="EEG alpha suppression",
                title="Neural mechanism audit")
    figure.tight_layout(); figure.savefig(root / "figure_04_validation_effects.png", dpi=250); plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate_design(cfg)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "alpha_suppression"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### EEG-guided alpha-suppression proof of concept")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    rng = np.random.default_rng(int(cfg.experiment.seed) + 1_700_003)

    # Stage 1: calibrate A without exposing any stimulation response.
    calibration_rows: list[dict[str, Any]] = []
    depths = [float(x) for x in cfg.analysis.calibration.modulation_depths]
    for seed in _seeds(cfg, "calibration"):
        b_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=0.0)
        episode = _run_condition(
            condition_id=B_LOW, condition_cfg=b_cfg, action=_sham(cfg, B_LOW),
            stimulate=False, seed=seed, action_index=0,
            output_dir=root / "calibration" / B_LOW / f"seed_{seed}",
            comm=comm, size=size, rank=rank,
        )
        if rank == 0:
            row = _epoch_row(episode).to_dict(); row["modulation_depth"] = 0.0
            calibration_rows.append(row)
        for index, depth in enumerate(depths, start=1):
            if rank == 0: print(f"calibration seed={seed}, modulation_depth={depth:g}")
            a_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=depth)
            episode = _run_condition(
                condition_id=A_HIGH, condition_cfg=a_cfg, action=_sham(cfg, A_HIGH),
                stimulate=False, seed=seed, action_index=index,
                output_dir=root / "calibration" / f"depth_{depth:g}" / f"seed_{seed}",
                comm=comm, size=size, rank=rank,
            )
            if rank == 0:
                row = _epoch_row(episode).to_dict(); row["modulation_depth"] = depth
                calibration_rows.append(row)
    if rank == 0:
        calibration_frame = pd.DataFrame(calibration_rows)
        calibration_metrics = _calibration_metrics(calibration_frame, cfg)
        calibration_summary = _calibration_summary(calibration_metrics, cfg, rng)
        selected_depth, calibration_passed = _select_reference_depth(calibration_summary, cfg)
        target_model = _alpha_target_model(calibration_frame, selected_depth)
        calibration_frame.to_csv(root / "calibration_epoch_eeg.csv", index=False)
        calibration_metrics.to_csv(root / "calibration_seed_metrics.csv", index=False)
        calibration_summary.to_csv(root / "calibration_summary.csv", index=False)
        (root / "frozen_alpha_target.json").write_text(json.dumps(_plain({
            "selected_modulation_depth": selected_depth,
            "calibration_passed": calibration_passed,
            "target_model": target_model,
        }), indent=2))
    else:
        selected_depth = calibration_passed = target_model = None
    selected_depth = float(comm.bcast(selected_depth, root=0))
    calibration_passed = bool(comm.bcast(calibration_passed, root=0))
    target_model = comm.bcast(target_model, root=0)

    # Stage 2: phase screen on disjoint seeds using EEG suppression only.
    discovery_metrics: list[dict[str, Any]] = []
    offsets = [float(x) for x in cfg.analysis.phase_discovery.relative_phase_offsets_rad]
    for seed in _seeds(cfg, "phase_discovery"):
        a_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=selected_depth)
        sham_episode = _run_condition(
            condition_id=A_HIGH, condition_cfg=a_cfg, action=_sham(cfg, A_HIGH),
            stimulate=False, seed=seed, action_index=0,
            output_dir=root / "phase_discovery" / A_HIGH / f"seed_{seed}",
            comm=comm, size=size, rank=rank,
        )
        sham_row = _epoch_row(sham_episode) if rank == 0 else None
        for index, offset in enumerate(offsets, start=1):
            action = _action(
                cfg, identifier=f"phase_{int(round(np.degrees(_wrap_phase(offset)))):03d}",
                role="phase_discovery", amplitude=float(cfg.analysis.tacs.amplitude_v_per_m),
                montage=str(cfg.analysis.tacs.axial_montage), relative_offset=offset,
            )
            if rank == 0: print(f"phase discovery seed={seed}, relative_offset={np.degrees(_wrap_phase(offset)):.0f} deg")
            active_episode = _run_condition(
                condition_id=str(action["id"]), condition_cfg=a_cfg, action=action,
                stimulate=True, seed=seed, action_index=index,
                output_dir=root / "phase_discovery" / str(action["id"]) / f"seed_{seed}",
                comm=comm, size=size, rank=rank,
            )
            if rank == 0:
                discovery_metrics.append(_discovery_metric(
                    sham_row, _epoch_row(active_episode), offset=offset,
                    model=target_model, cfg=cfg,
                ))
    if rank == 0:
        discovery_frame = pd.DataFrame(discovery_metrics)
        discovery_summary = _discovery_summary(discovery_frame, cfg, rng)
        selected_offset, discovery_positive = _select_suppressive_offset(discovery_summary)
        discovery_frame.to_csv(root / "phase_discovery_seed_metrics.csv", index=False)
        discovery_summary.to_csv(root / "phase_discovery_summary.csv", index=False)
    else:
        selected_offset = discovery_positive = None
    selected_offset = float(comm.bcast(selected_offset, root=0))
    discovery_positive = bool(comm.bcast(discovery_positive, root=0))

    frozen_protocol = {
        "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
        "amplitude_v_per_m": float(cfg.analysis.tacs.amplitude_v_per_m),
        "montage": str(cfg.analysis.tacs.axial_montage),
        "selected_eeg_relative_phase_offset_rad": selected_offset,
        "selected_eeg_relative_phase_offset_deg": float(np.degrees(selected_offset)),
        "opposite_offset_rad": _wrap_phase(selected_offset + np.pi),
        "phase_is_estimated_from_preceding_eeg": True,
        "selected_modulation_depth": selected_depth,
    }
    if rank == 0:
        (root / "frozen_tacs_protocol.json").write_text(json.dumps(_plain(frozen_protocol), indent=2))

    # Stage 3: frozen held-out evaluation.
    validation_episodes: dict[int, dict[str, dict[str, Any]]] = {}
    validation_epoch_rows: list[dict[str, Any]] = []
    validation_metrics: list[dict[str, Any]] = []
    synthetic_rows: list[dict[str, Any]] = []
    two_second_rows: list[dict[str, Any]] = []
    action_specs = [
        (SELECTED, selected_offset, str(cfg.analysis.tacs.axial_montage)),
        (OPPOSITE, _wrap_phase(selected_offset + np.pi), str(cfg.analysis.tacs.axial_montage)),
        (TRANSVERSE, selected_offset, str(cfg.analysis.tacs.transverse_montage)),
    ]
    for seed in _seeds(cfg, "validation"):
        if rank == 0: validation_episodes[seed] = {}
        configs = {
            B_LOW: _condition_for_seed(cfg, seed=seed, modulation_depth=0.0),
            A_HIGH: _condition_for_seed(cfg, seed=seed, modulation_depth=selected_depth),
        }
        for index, condition in enumerate((B_LOW, A_HIGH)):
            episode = _run_condition(
                condition_id=condition, condition_cfg=configs[condition], action=_sham(cfg, condition),
                stimulate=False, seed=seed, action_index=index,
                output_dir=root / "validation" / condition / f"seed_{seed}",
                comm=comm, size=size, rank=rank,
            )
            if rank == 0:
                validation_episodes[seed][condition] = episode
                validation_epoch_rows.extend(episode["epoch_rows"])
                two_second_rows.extend(_two_second_rows(episode, condition_id=condition, cfg=cfg))
        for index, (condition, offset, montage) in enumerate(action_specs, start=2):
            action = _action(
                cfg, identifier=condition, role="heldout_control" if condition != SELECTED else "frozen_primary",
                amplitude=float(cfg.analysis.tacs.amplitude_v_per_m), montage=montage,
                relative_offset=offset,
            )
            if rank == 0: print(f"validation seed={seed}, condition={condition}")
            episode = _run_condition(
                condition_id=condition, condition_cfg=configs[A_HIGH], action=action,
                stimulate=True, seed=seed, action_index=index,
                output_dir=root / "validation" / condition / f"seed_{seed}",
                comm=comm, size=size, rank=rank,
            )
            if rank == 0:
                validation_episodes[seed][condition] = episode
                validation_epoch_rows.extend(episode["epoch_rows"])
                two_second_rows.extend(_two_second_rows(episode, condition_id=condition, cfg=cfg))
        if rank == 0:
            metric, synthetic = _validation_seed_metric(
                seed=seed, episodes=validation_episodes[seed], model=target_model, cfg=cfg
            )
            validation_metrics.append(metric); synthetic_rows.append(synthetic)

    if rank == 0:
        epoch_frame = pd.DataFrame(validation_epoch_rows)
        metric_frame = pd.DataFrame(validation_metrics)
        synthetic_frame = pd.DataFrame(synthetic_rows)
        two_second_frame = pd.DataFrame(two_second_rows)
        summary_frame = _metric_summaries(metric_frame, cfg, rng)
        checks, conclusions = _check_table(
            cfg=cfg, calibration_passed=calibration_passed,
            discovery_summary=discovery_summary, selected_offset=selected_offset,
            validation=metric_frame, summaries=summary_frame,
            epoch_rows=epoch_frame, two_second=two_second_frame, model=target_model,
        )
        epoch_frame.to_csv(root / "validation_epoch_eeg_and_hidden_metrics.csv", index=False)
        metric_frame.to_csv(root / "validation_seed_metrics.csv", index=False)
        summary_frame.to_csv(root / "validation_summary.csv", index=False)
        synthetic_frame.to_csv(root / "observation_only_complex_match_audit.csv", index=False)
        two_second_frame.to_csv(root / "two_second_eeg_bins.csv", index=False)
        payload = {
            "state_definition": {
                "A": "elevated-alpha toy state from mean-rate-matched 10-Hz afferent modulation",
                "B": "low-alpha toy reference from homogeneous Poisson afferents",
                "not_a_claim_about": ["depression", "health", "treatment", "human EEG"],
            },
            "frozen_protocol": frozen_protocol,
            "checks": checks,
            "conclusions": conclusions,
            "interpretation_boundary": (
                "The ideal EEG contains neural forward-model signal only. Concurrent tACS-EEG "
                "artifact robustness requires a separate measurement model and is not inferred "
                "from a positive neural-only result."
            ),
            "runtime_seconds": float(time.perf_counter() - started),
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(_plain(payload), indent=2))
        if bool(cfg.experiment.plot):
            _plot_results(
                root=root, calibration=calibration_metrics, discovery=discovery_frame,
                selected_offset=selected_offset, validation_rows=epoch_frame,
                validation_metrics=metric_frame, episodes=validation_episodes, cfg=cfg,
            )
        print("\n### Frozen EEG-relative tACS protocol")
        print(json.dumps(_plain(frozen_protocol), indent=2))
        print("\n### Alpha-suppression checks")
        for name, passed in checks.items(): print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
        print("\nIdeal neural-EEG directional proof of concept:", "PASSED" if conclusions["ideal_neural_eeg_directional_proof_of_concept"] else "NOT PASSED")
        print("Confirmatory held-out evidence:", "PASSED" if conclusions["confirmatory_heldout_evidence"] else "NOT PASSED")
        print("Artifact-robust concurrent-EEG claim:", "PASSED" if conclusions["artifact_robust_concurrent_eeg_claim"] else "NOT PASSED")
        print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
