"""Confirm the 0.5-V/m, 10-Hz lead and audit its mechanism.

The protocol is fixed before confirmation.  Calibration A/B-sham seeds define
feature scaling; disjoint confirmation seeds compare parallel stimulation,
perpendicular-field stimulation, and an observation-only synthetic sinusoid.
The primary EEG distance removes bins around the stimulation fundamental so a
controller cannot pass merely by painting the rewarded spectrum.
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
    FREQUENCY_BANDS,
    _extract_eeg_features,
    _preprocess_eeg,
)


ACTUAL_ARMS = ("A_sham", "B_sham", "B_parallel", "B_perpendicular")
CONTROL_ARMS = ("B_parallel", "B_perpendicular", "B_synthetic")
ANALYSIS_EPOCHS = ("baseline", "stimulation", "post")


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


def _timeline(cfg: DictConfig) -> list[tuple[str, int]]:
    timeline = cfg.analysis.timeline
    values = [
        ("burn_in", int(timeline.burn_in_steps)),
        ("baseline", int(timeline.baseline_steps)),
        ("stimulation", int(timeline.stimulation_steps)),
        ("post", int(timeline.post_steps)),
    ]
    if any(count <= 0 for _, count in values):
        raise ValueError("Every mechanism-experiment epoch must contain a window.")
    return values


def _episode_config(
    base_cfg: DictConfig,
    *,
    inhibition_scale: float,
    seed: int,
    field_direction: list[float],
    output_dir: Path,
) -> DictConfig:
    run_cfg = _plain_copy(base_cfg)
    OmegaConf.set_struct(run_cfg, False)
    n_steps = sum(count for _, count in _timeline(run_cfg))
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    with open_dict(run_cfg):
        run_cfg.experiment.seed = int(seed)
        run_cfg.experiment.dir = str(output_dir)
        run_cfg.env.simulation.duration = n_steps * window_ms
        run_cfg.env.network.inhibition_scale = float(inhibition_scale)
        run_cfg.env.ts.apply = True
        run_cfg.env.online.temperature_mode = "configured"
        run_cfg.env.online.stimulation.parameterization = "uniform_field"
        run_cfg.env.online.stimulation.field_direction = list(field_direction)
    return run_cfg


def _bandpower_from_psd(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (frequencies_hz >= low_hz) & (frequencies_hz < high_hz)
    if np.count_nonzero(mask) < 2:
        return 0.0
    selected_freqs = frequencies_hz[mask]
    selected_psd = psd[mask]
    return float(np.trapz(selected_psd, selected_freqs))


def _stimulus_excluded_features(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    *,
    stimulus_frequency_hz: float,
    half_width_hz: float,
) -> dict[str, float]:
    """Calculate the primary features after zeroing the fundamental bins."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    keep = np.ones_like(psd, dtype=bool)
    keep &= np.abs(frequencies_hz - float(stimulus_frequency_hz)) > float(
        half_width_hz
    )
    masked_psd = np.where(keep, psd, 0.0)
    total = _bandpower_from_psd(frequencies_hz, masked_psd, 1.0, 80.000001)
    gamma = _bandpower_from_psd(frequencies_hz, masked_psd, 30.0, 80.000001)
    stimulus_power = _bandpower_from_psd(
        frequencies_hz,
        psd,
        max(0.0, stimulus_frequency_hz - half_width_hz),
        stimulus_frequency_hz + half_width_hz + np.finfo(float).eps,
    )
    eps = np.finfo(np.float64).tiny
    return {
        "total_power_1_80_excluding_stimulus": total,
        "log10_total_power_1_80_excluding_stimulus": float(
            np.log10(max(total, eps))
        ),
        "gamma_power_excluding_stimulus": gamma,
        "relative_gamma_power_excluding_stimulus": (
            gamma / total if total > 0 else float("nan")
        ),
        "stimulus_frequency_power": stimulus_power,
    }


def _analyze_eeg(
    raw_eeg: np.ndarray,
    *,
    simulator_fs_hz: float,
    cfg: DictConfig,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    processed, analysis_fs_hz = _preprocess_eeg(
        raw_eeg,
        fs_hz=simulator_fs_hz,
        target_fs_hz=int(cfg.analysis.target_fs_hz),
        low_hz=float(cfg.analysis.low_hz),
        high_hz=float(cfg.analysis.high_hz),
    )
    features, frequencies_hz, psd = _extract_eeg_features(
        processed,
        analysis_fs_hz,
    )
    features.update(
        _stimulus_excluded_features(
            frequencies_hz,
            psd,
            stimulus_frequency_hz=float(cfg.analysis.protocol.frequency_hz),
            half_width_hz=float(
                cfg.analysis.stimulus_exclusion_half_width_hz
            ),
        )
    )
    return features, processed, frequencies_hz, psd


def _phase_metrics(
    spike_times_ms: np.ndarray,
    *,
    frequency_hz: float,
    phase_origin_ms: float,
) -> dict[str, float]:
    spike_times_ms = np.asarray(spike_times_ms, dtype=np.float64)
    if spike_times_ms.size == 0:
        return {
            "spike_count": 0.0,
            "plv": float("nan"),
            "mean_phase_rad": float("nan"),
            "rayleigh_z": float("nan"),
        }
    phases = (
        2.0
        * np.pi
        * float(frequency_hz)
        * ((spike_times_ms - float(phase_origin_ms)) / 1000.0)
    )
    resultant = np.mean(np.exp(1j * phases))
    plv = float(np.abs(resultant))
    return {
        "spike_count": float(spike_times_ms.size),
        "plv": plv,
        "mean_phase_rad": float(np.angle(resultant)),
        "rayleigh_z": float(spike_times_ms.size * plv * plv),
    }


def _collect_epoch_spikes(outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    collected: dict[str, dict[str, Any]] = {}
    for population_name in ("E", "I"):
        times = np.concatenate(
            [
                np.asarray(output["spikes"][population_name]["times_ms"])
                for output in outputs
            ]
        )
        gids = np.concatenate(
            [
                np.asarray(output["spikes"][population_name]["gids"])
                for output in outputs
            ]
        )
        population_size = len(outputs[0]["spikes"][population_name]["per_cell"])
        collected[population_name] = {
            "times_ms": times.astype(np.float64, copy=False),
            "gids": gids.astype(np.int64, copy=False),
            "population_size": int(population_size),
        }
    return collected


def _epoch_row(
    *,
    seed: int,
    arm: str,
    epoch: str,
    features: dict[str, float],
    outputs: list[dict[str, Any]],
    block_start_ms: float,
    cfg: DictConfig,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    spikes = _collect_epoch_spikes(outputs)
    duration_s = sum(
        float(output["t_stop_ms"] - output["t_start_ms"]) for output in outputs
    ) / 1000.0
    row: dict[str, Any] = {"seed": int(seed), "arm": arm, "epoch": epoch}
    row.update(features)
    for population_name in ("E", "I"):
        population_spikes = spikes[population_name]
        count = int(population_spikes["times_ms"].size)
        size = int(population_spikes["population_size"])
        row[f"{population_name}_spike_count"] = count
        row[f"{population_name}_firing_rate_hz"] = (
            count / (size * duration_s) if size > 0 else float("nan")
        )
        phase = _phase_metrics(
            population_spikes["times_ms"],
            frequency_hz=float(cfg.analysis.protocol.frequency_hz),
            phase_origin_ms=block_start_ms,
        )
        row[f"{population_name}_plv"] = phase["plv"]
        row[f"{population_name}_mean_phase_rad"] = phase["mean_phase_rad"]
        row[f"{population_name}_rayleigh_z"] = phase["rayleigh_z"]
    i_rate = float(row["I_firing_rate_hz"])
    row["E_I_firing_rate_ratio"] = (
        float(row["E_firing_rate_hz"]) / i_rate if i_rate > 0 else float("nan")
    )
    return row, spikes


def _run_episode(
    base_cfg: DictConfig,
    *,
    seed: int,
    arm: str,
    inhibition_scale: float,
    field_direction: list[float],
    stimulate: bool,
    output_dir: Path,
    comm,
    size: int,
    rank: int,
) -> dict[str, Any] | None:
    run_cfg = _episode_config(
        base_cfg,
        inhibition_scale=inhibition_scale,
        seed=seed,
        field_direction=field_direction,
        output_dir=output_dir,
    )
    np.random.seed(int(seed) * 10_000 + rank)
    random.seed(int(seed) * 10_000 + rank)
    environment = OnlineNeuronEnv(
        run_cfg,
        _mpi_variables(comm, size, rank, seed),
        ENV_SEED=0,
    )

    window_ms = float(run_cfg.env.simulation.obs_win_len)
    schedule = _timeline(run_cfg)
    pre_stimulation_steps = int(run_cfg.analysis.timeline.burn_in_steps) + int(
        run_cfg.analysis.timeline.baseline_steps
    )
    block_start_ms = pre_stimulation_steps * window_ms
    block_stop_ms = block_start_ms + int(
        run_cfg.analysis.timeline.stimulation_steps
    ) * window_ms
    block_envelope = {
        "start_ms": block_start_ms,
        "stop_ms": block_stop_ms,
        "ramp_ms": float(run_cfg.analysis.protocol.block_ramp_ms),
    }
    amplitude = float(run_cfg.analysis.protocol.amplitude_v_per_m)
    frequency = float(run_cfg.analysis.protocol.frequency_hz)

    outputs_by_epoch: dict[str, list[dict[str, Any]]] = {
        epoch: [] for epoch, _ in schedule
    }
    try:
        for epoch, count in schedule:
            for _ in range(count):
                active = bool(stimulate and epoch == "stimulation")
                output = environment.step_online(
                    [amplitude, frequency] if active else [0.0, 0.0],
                    phase_continuous=True,
                    ramp_ms=0.0,
                    block_envelope=block_envelope if active else None,
                )
                if rank == 0:
                    outputs_by_epoch[epoch].append(output)
    finally:
        environment.close()

    if rank != 0:
        return None

    simulator_fs_hz = 1000.0 / float(run_cfg.env.network.dt)
    epoch_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    raw_by_epoch: dict[str, np.ndarray] = {}
    processed_by_epoch: dict[str, np.ndarray] = {}
    psd_by_epoch: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    spikes_by_epoch: dict[str, dict[str, dict[str, Any]]] = {}

    for epoch in ANALYSIS_EPOCHS:
        epoch_outputs = outputs_by_epoch[epoch]
        raw = np.concatenate(
            [np.asarray(output["eeg_v"]).reshape(-1) for output in epoch_outputs]
        ).astype(np.float64, copy=False)
        features, processed, frequencies_hz, psd = _analyze_eeg(
            raw,
            simulator_fs_hz=simulator_fs_hz,
            cfg=run_cfg,
        )
        row, spikes = _epoch_row(
            seed=seed,
            arm=arm,
            epoch=epoch,
            features=features,
            outputs=epoch_outputs,
            block_start_ms=block_start_ms,
            cfg=run_cfg,
        )
        epoch_rows.append(row)
        raw_by_epoch[epoch] = raw
        processed_by_epoch[epoch] = processed
        psd_by_epoch[epoch] = (frequencies_hz, psd)
        spikes_by_epoch[epoch] = spikes

        for window_index, output in enumerate(epoch_outputs):
            window_features, _, _, _ = _analyze_eeg(
                np.asarray(output["eeg_v"]).reshape(-1),
                simulator_fs_hz=simulator_fs_hz,
                cfg=run_cfg,
            )
            window_row = {
                "seed": int(seed),
                "arm": arm,
                "epoch": epoch,
                "epoch_window_index": window_index,
                "t_start_ms": float(output["t_start_ms"]),
                "t_stop_ms": float(output["t_stop_ms"]),
                "amplitude_v_per_m": float(output["action"][0]),
                "frequency_hz": float(output["action"][1]),
                **window_features,
                **output["firing_rates"],
            }
            for population_name in ("E", "I"):
                phase = _phase_metrics(
                    output["spikes"][population_name]["times_ms"],
                    frequency_hz=frequency,
                    phase_origin_ms=block_start_ms,
                )
                window_row[f"{population_name}_plv"] = phase["plv"]
            window_rows.append(window_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(run_cfg.analysis.save_raw_eeg):
        signal_payload: dict[str, np.ndarray] = {}
        psd_payload: dict[str, np.ndarray] = {}
        for epoch in ANALYSIS_EPOCHS:
            signal_payload[f"{epoch}_eeg_raw_v"] = raw_by_epoch[epoch]
            signal_payload[f"{epoch}_eeg_preprocessed_v"] = processed_by_epoch[
                epoch
            ]
            psd_payload[f"{epoch}_frequencies_hz"] = psd_by_epoch[epoch][0]
            psd_payload[f"{epoch}_psd_v2_per_hz"] = psd_by_epoch[epoch][1]
        np.savez_compressed(output_dir / "signals.npz", **signal_payload)
        np.savez_compressed(output_dir / "psd.npz", **psd_payload)
    if bool(run_cfg.analysis.save_spikes):
        spike_payload: dict[str, np.ndarray] = {}
        for epoch in ANALYSIS_EPOCHS:
            for population_name in ("E", "I"):
                spike_payload[f"{epoch}_{population_name}_times_ms"] = (
                    spikes_by_epoch[epoch][population_name]["times_ms"]
                )
                spike_payload[f"{epoch}_{population_name}_gids"] = (
                    spikes_by_epoch[epoch][population_name]["gids"]
                )
        np.savez_compressed(output_dir / "spikes.npz", **spike_payload)

    return {
        "seed": int(seed),
        "arm": arm,
        "epoch_rows": epoch_rows,
        "window_rows": window_rows,
        "raw_by_epoch": raw_by_epoch,
        "processed_by_epoch": processed_by_epoch,
        "psd_by_epoch": psd_by_epoch,
        "simulator_fs_hz": simulator_fs_hz,
    }


def _driven_band_name(frequency_hz: float) -> str:
    for band_name, (low_hz, high_hz) in FREQUENCY_BANDS.items():
        if low_hz <= frequency_hz < high_hz:
            return band_name
    raise ValueError(f"No configured EEG band contains {frequency_hz} Hz.")


def _make_synthetic_epoch(
    b_raw: np.ndarray,
    active_raw: np.ndarray,
    *,
    simulator_fs_hz: float,
    cfg: DictConfig,
) -> tuple[np.ndarray, float, float]:
    """Add a pure sine to B until driven-band power matches active B."""
    frequency_hz = float(cfg.analysis.protocol.frequency_hz)
    phase_rad = float(cfg.analysis.synthetic_control.phase_rad)
    if not bool(cfg.analysis.synthetic_control.match_driven_band_power):
        raise ValueError(
            "This confirmatory runner requires match_driven_band_power=true."
        )
    band_name = _driven_band_name(frequency_hz)
    active_features, _, _, _ = _analyze_eeg(
        active_raw,
        simulator_fs_hz=simulator_fs_hz,
        cfg=cfg,
    )
    b_features, _, _, _ = _analyze_eeg(
        b_raw,
        simulator_fs_hz=simulator_fs_hz,
        cfg=cfg,
    )
    target_power = float(active_features[f"{band_name}_power"])
    baseline_power = float(b_features[f"{band_name}_power"])
    if target_power <= baseline_power:
        return np.asarray(b_raw, dtype=np.float64).copy(), 0.0, baseline_power

    left_boundary_ms = (
        np.arange(b_raw.size, dtype=np.float64) * 1000.0 / simulator_fs_hz
    )
    duration_ms = float(b_raw.size) * 1000.0 / simulator_fs_hz
    sample_time_s = (left_boundary_ms + 1000.0 / simulator_fs_hz) / 1000.0
    unit_sine = np.sin(
        2.0 * np.pi * frequency_hz * sample_time_s + phase_rad
    )
    unit_sine = apply_raised_cosine_block_envelope(
        unit_sine,
        time_ms=left_boundary_ms,
        block_start_ms=0.0,
        block_stop_ms=duration_ms,
        ramp_ms=float(cfg.analysis.protocol.block_ramp_ms),
    )

    def objective(amplitude_v: float) -> float:
        features, _, _, _ = _analyze_eeg(
            b_raw + float(amplitude_v) * unit_sine,
            simulator_fs_hz=simulator_fs_hz,
            cfg=cfg,
        )
        return float(features[f"{band_name}_power"] - target_power)

    high = max(float(np.std(b_raw)) * 0.01, 1e-15)
    for _ in range(40):
        if objective(high) >= 0.0:
            break
        high *= 2.0
    else:
        raise RuntimeError("Could not bracket the synthetic sinusoid amplitude.")
    amplitude = float(so.brentq(objective, 0.0, high, xtol=1e-18, rtol=1e-12))
    synthetic = np.asarray(b_raw, dtype=np.float64) + amplitude * unit_sine
    achieved_features, _, _, _ = _analyze_eeg(
        synthetic,
        simulator_fs_hz=simulator_fs_hz,
        cfg=cfg,
    )
    achieved = float(achieved_features[f"{band_name}_power"])
    return synthetic, amplitude, achieved


def _make_synthetic_control(
    b_episode: dict[str, Any],
    active_episode: dict[str, Any],
    *,
    output_dir: Path,
    cfg: DictConfig,
) -> dict[str, Any]:
    raw_by_epoch = {
        epoch: np.asarray(b_episode["raw_by_epoch"][epoch]).copy()
        for epoch in ANALYSIS_EPOCHS
    }
    synthetic, amplitude_v, achieved_power = _make_synthetic_epoch(
        raw_by_epoch["stimulation"],
        active_episode["raw_by_epoch"]["stimulation"],
        simulator_fs_hz=float(b_episode["simulator_fs_hz"]),
        cfg=cfg,
    )
    raw_by_epoch["stimulation"] = synthetic

    b_rows = {row["epoch"]: row for row in b_episode["epoch_rows"]}
    epoch_rows = []
    processed_payload: dict[str, np.ndarray] = {}
    psd_payload: dict[str, np.ndarray] = {}
    for epoch in ANALYSIS_EPOCHS:
        features, processed, frequencies_hz, psd = _analyze_eeg(
            raw_by_epoch[epoch],
            simulator_fs_hz=float(b_episode["simulator_fs_hz"]),
            cfg=cfg,
        )
        source = b_rows[epoch]
        row = {
            "seed": int(b_episode["seed"]),
            "arm": "B_synthetic",
            "epoch": epoch,
            **features,
        }
        for name in (
            "E_spike_count",
            "I_spike_count",
            "E_firing_rate_hz",
            "I_firing_rate_hz",
            "E_I_firing_rate_ratio",
            "E_plv",
            "I_plv",
            "E_mean_phase_rad",
            "I_mean_phase_rad",
            "E_rayleigh_z",
            "I_rayleigh_z",
        ):
            row[name] = source[name]
        row["synthetic_added_peak_v"] = amplitude_v if epoch == "stimulation" else 0.0
        row["synthetic_achieved_driven_band_power"] = achieved_power
        epoch_rows.append(row)
        processed_payload[f"{epoch}_eeg_preprocessed_v"] = processed
        psd_payload[f"{epoch}_frequencies_hz"] = frequencies_hz
        psd_payload[f"{epoch}_psd_v2_per_hz"] = psd

    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.analysis.save_raw_eeg):
        np.savez_compressed(
            output_dir / "signals.npz",
            **{
                f"{epoch}_eeg_raw_v": raw_by_epoch[epoch]
                for epoch in ANALYSIS_EPOCHS
            },
            **processed_payload,
        )
        np.savez_compressed(output_dir / "psd.npz", **psd_payload)
    return {"epoch_rows": epoch_rows, "raw_by_epoch": raw_by_epoch}


def _make_standardizer(
    rows: pd.DataFrame,
    *,
    epoch: str,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    subset = rows[
        (rows["epoch"] == epoch) & rows["arm"].isin(["A_sham", "B_sham"])
    ]
    values = subset[feature_names].to_numpy(dtype=np.float64)
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    positive = scale[scale > np.finfo(np.float64).eps]
    fallback = float(np.median(positive)) if positive.size else 1.0
    scale = np.where(scale > np.finfo(np.float64).eps, scale, fallback)
    return center, scale


def _reachability_metrics(
    target: pd.Series,
    sham: pd.Series,
    stimulated: pd.Series,
    *,
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    def vector(row: pd.Series) -> np.ndarray:
        return (row[feature_names].to_numpy(dtype=float) - center) / scale

    target_vector = vector(target)
    sham_vector = vector(sham)
    stimulated_vector = vector(stimulated)
    target_shift = target_vector - sham_vector
    stimulation_shift = stimulated_vector - sham_vector
    sham_distance = float(np.linalg.norm(sham_vector - target_vector))
    stimulated_distance = float(np.linalg.norm(stimulated_vector - target_vector))
    denominator = float(
        np.linalg.norm(target_shift) * np.linalg.norm(stimulation_shift)
    )
    return {
        "sham_distance_to_A": sham_distance,
        "stimulated_distance_to_A": stimulated_distance,
        "fractional_distance_improvement": (
            1.0 - stimulated_distance / sham_distance
            if sham_distance > 0
            else 0.0
        ),
        "target_shift_alignment": (
            float(np.dot(target_shift, stimulation_shift) / denominator)
            if denominator > 0
            else 0.0
        ),
        "max_abs_target_error_z": float(
            np.max(np.abs(stimulated_vector - target_vector))
        ),
    }


def _rate_safe(
    stimulated: pd.Series,
    target: pd.Series,
    sham: pd.Series,
    cfg: DictConfig,
) -> bool:
    limits = cfg.analysis.rate_guardrails_hz
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    for population_name in ("E", "I"):
        name = f"{population_name}_firing_rate_hz"
        value = float(stimulated[name])
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


def _paired_contrast(
    rows: pd.DataFrame,
    *,
    arm_a: str,
    arm_b: str,
    epoch: str,
    feature_set: str,
    value_name: str,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    subset = rows[
        (rows["epoch"] == epoch) & (rows["feature_set"] == feature_set)
    ]
    pivot = subset.pivot(index="seed", columns="arm", values=value_name)
    differences = (pivot[arm_a] - pivot[arm_b]).to_numpy(dtype=float)
    ci_low, ci_high = _bootstrap_ci(
        differences,
        rng=rng,
        n_bootstrap=int(cfg.analysis.n_bootstrap),
    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "epoch": epoch,
        "feature_set": feature_set,
        "value": value_name,
        "n_seeds": int(differences.size),
        "mean_difference": float(np.mean(differences)),
        "ci_2.5": ci_low,
        "ci_97.5": ci_high,
        "paired_sign_flip_p": _sign_flip_p(
            differences,
            rng=rng,
            n_permutations=int(cfg.analysis.n_permutations),
        ),
    }


def _summarize_reachability(
    rows: pd.DataFrame,
    *,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    summaries = []
    equivalence_margin = float(cfg.analysis.criteria.equivalence_z_margin)
    for keys, group in rows.groupby(["arm", "epoch", "feature_set"], sort=False):
        values = group["fractional_distance_improvement"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(
            values,
            rng=rng,
            n_bootstrap=int(cfg.analysis.n_bootstrap),
        )
        summaries.append(
            {
                "arm": keys[0],
                "epoch": keys[1],
                "feature_set": keys[2],
                "n_seeds": int(group["seed"].nunique()),
                "mean_fractional_improvement": float(np.mean(values)),
                "ci_2.5": ci_low,
                "ci_97.5": ci_high,
                "positive_seed_fraction": float(np.mean(values > 0.0)),
                "median_alignment": float(
                    np.median(group["target_shift_alignment"])
                ),
                "mean_sham_distance": float(group["sham_distance_to_A"].mean()),
                "mean_stimulated_distance": float(
                    group["stimulated_distance_to_A"].mean()
                ),
                "equivalent_seed_fraction": float(
                    np.mean(group["max_abs_target_error_z"] <= equivalence_margin)
                ),
                "all_rate_safe": bool(group["rate_safe"].all()),
            }
        )
    return pd.DataFrame(summaries)


def _relative_rms_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    denominator = float(np.sqrt(np.mean(np.square(reference))))
    numerator = float(np.sqrt(np.mean(np.square(candidate - reference))))
    return numerator / denominator if denominator > 0 else numerator


def _plot_results(
    reachability_rows: pd.DataFrame,
    episodes: dict[int, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    stimulation = reachability_rows[reachability_rows["epoch"] == "stimulation"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, feature_set in zip(axes, ("raw", "fundamental_excluded")):
        subset = stimulation[stimulation["feature_set"] == feature_set]
        arms = list(CONTROL_ARMS)
        values = [
            subset[subset["arm"] == arm]["fractional_distance_improvement"]
            for arm in arms
        ]
        axis.boxplot(values, labels=arms, showmeans=True)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(feature_set.replace("_", " "))
        axis.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Fractional distance improvement toward A")
    figure.tight_layout()
    figure.savefig(output_dir / "confirmation_improvement.png", dpi=250)
    plt.close(figure)

    psd_by_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in ACTUAL_ARMS}
    frequencies_hz = None
    for seed_episodes in episodes.values():
        for arm in ACTUAL_ARMS:
            frequencies_hz, psd = seed_episodes[arm]["psd_by_epoch"]["stimulation"]
            psd_by_arm[arm].append(psd)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for arm, values in psd_by_arm.items():
        axis.semilogy(
            frequencies_hz,
            np.mean(np.vstack(values), axis=0),
            label=arm,
        )
    axis.set_xlim(1.0, 80.0)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (V²/Hz)")
    axis.set_title("Confirmation stimulation-epoch spectra")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "stimulation_psd_comparison.png", dpi=250)
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
        raise ValueError("The mechanism experiment requires online mode.")

    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "stimulation_mechanism"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### BallAndStick stimulation mechanism confirmation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    base_seed = int(cfg.experiment.seed)
    calibration_seeds = [
        base_seed + int(cfg.analysis.calibration.seed_offset) + index
        for index in range(int(cfg.analysis.calibration.n_seeds))
    ]
    confirmation_seeds = [
        base_seed + int(cfg.analysis.confirmation.seed_offset) + index
        for index in range(int(cfg.analysis.confirmation.n_seeds))
    ]
    if set(calibration_seeds).intersection(confirmation_seeds):
        raise ValueError("Calibration and confirmation seeds must be disjoint.")

    parallel = list(cfg.analysis.protocol.parallel_direction)
    perpendicular = list(cfg.analysis.protocol.perpendicular_direction)
    condition_a = float(cfg.analysis.condition_a_inhibition_scale)
    condition_b = float(cfg.analysis.condition_b_inhibition_scale)
    calibration_epoch_rows: list[dict[str, Any]] = []

    for seed in calibration_seeds:
        for arm, inhibition in (("A_sham", condition_a), ("B_sham", condition_b)):
            if rank == 0:
                print(f"calibration seed={seed}, arm={arm}")
            episode = _run_episode(
                cfg,
                seed=seed,
                arm=arm,
                inhibition_scale=inhibition,
                field_direction=parallel,
                stimulate=False,
                output_dir=root / "calibration" / arm / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                calibration_epoch_rows.extend(episode["epoch_rows"])

    if rank == 0:
        calibration_df = pd.DataFrame(calibration_epoch_rows)
        calibration_df.to_csv(root / "calibration_epoch_features.csv", index=False)
        standardizers: dict[str, dict[str, list[float]]] = {}
        feature_sets = {
            "raw": list(cfg.analysis.raw_distance_features),
            "fundamental_excluded": list(
                cfg.analysis.primary_distance_features
            ),
        }
        for epoch in ("stimulation", "post"):
            for feature_set, names in feature_sets.items():
                center, scale = _make_standardizer(
                    calibration_df,
                    epoch=epoch,
                    feature_names=names,
                )
                standardizers[f"{epoch}:{feature_set}"] = {
                    "center": center.tolist(),
                    "scale": scale.tolist(),
                }
    else:
        feature_sets = standardizers = None
    feature_sets = comm.bcast(feature_sets, root=0)
    standardizers = comm.bcast(standardizers, root=0)

    confirmation_epoch_rows: list[dict[str, Any]] = []
    confirmation_window_rows: list[dict[str, Any]] = []
    episodes: dict[int, dict[str, dict[str, Any]]] = {}
    baseline_checks: list[dict[str, Any]] = []

    arm_specs = (
        ("A_sham", condition_a, parallel, False),
        ("B_sham", condition_b, parallel, False),
        ("B_parallel", condition_b, parallel, True),
        ("B_perpendicular", condition_b, perpendicular, True),
    )
    for seed in confirmation_seeds:
        seed_episodes: dict[str, dict[str, Any]] = {}
        for arm, inhibition, direction, stimulate in arm_specs:
            if rank == 0:
                print(f"confirmation seed={seed}, arm={arm}")
            episode = _run_episode(
                cfg,
                seed=seed,
                arm=arm,
                inhibition_scale=inhibition,
                field_direction=direction,
                stimulate=stimulate,
                output_dir=root / "confirmation" / arm / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                seed_episodes[arm] = episode
                confirmation_epoch_rows.extend(episode["epoch_rows"])
                confirmation_window_rows.extend(episode["window_rows"])

        if rank == 0:
            synthetic = _make_synthetic_control(
                seed_episodes["B_sham"],
                seed_episodes["B_parallel"],
                output_dir=(
                    root / "confirmation" / "B_synthetic" / f"seed_{seed}"
                ),
                cfg=cfg,
            )
            confirmation_epoch_rows.extend(synthetic["epoch_rows"])
            for arm in ("B_parallel", "B_perpendicular"):
                baseline_checks.append(
                    {
                        "seed": int(seed),
                        "arm": arm,
                        "baseline_relative_rms_error_vs_B_sham": (
                            _relative_rms_error(
                                seed_episodes["B_sham"]["raw_by_epoch"]["baseline"],
                                seed_episodes[arm]["raw_by_epoch"]["baseline"],
                            )
                        ),
                    }
                )
            episodes[int(seed)] = {
                arm: {"psd_by_epoch": seed_episodes[arm]["psd_by_epoch"]}
                for arm in ACTUAL_ARMS
            }

    if rank != 0:
        return

    epoch_df = pd.DataFrame(confirmation_epoch_rows)
    window_df = pd.DataFrame(confirmation_window_rows)
    baseline_df = pd.DataFrame(baseline_checks)
    epoch_df.to_csv(root / "confirmation_epoch_features.csv", index=False)
    window_df.to_csv(root / "confirmation_window_features.csv", index=False)
    baseline_df.to_csv(root / "baseline_causality_checks.csv", index=False)

    reachability_rows: list[dict[str, Any]] = []
    for seed in confirmation_seeds:
        for epoch in ("stimulation", "post"):
            seed_epoch = epoch_df[
                (epoch_df["seed"] == seed) & (epoch_df["epoch"] == epoch)
            ].set_index("arm")
            target = seed_epoch.loc["A_sham"]
            sham = seed_epoch.loc["B_sham"]
            for arm in CONTROL_ARMS:
                stimulated = seed_epoch.loc[arm]
                for feature_set, names in feature_sets.items():
                    standardizer = standardizers[f"{epoch}:{feature_set}"]
                    metrics = _reachability_metrics(
                        target,
                        sham,
                        stimulated,
                        feature_names=names,
                        center=np.asarray(standardizer["center"]),
                        scale=np.asarray(standardizer["scale"]),
                    )
                    reachability_rows.append(
                        {
                            "seed": int(seed),
                            "arm": arm,
                            "epoch": epoch,
                            "feature_set": feature_set,
                            **metrics,
                            "rate_safe": _rate_safe(
                                stimulated,
                                target,
                                sham,
                                cfg,
                            ),
                        }
                    )
    reachability_df = pd.DataFrame(reachability_rows)
    reachability_df.to_csv(root / "confirmation_reachability_by_seed.csv", index=False)

    rng = np.random.default_rng(base_seed + 90_001)
    summary_df = _summarize_reachability(
        reachability_df,
        cfg=cfg,
        rng=rng,
    )
    summary_df.to_csv(root / "confirmation_reachability_summary.csv", index=False)

    contrasts = []
    for feature_set in ("raw", "fundamental_excluded"):
        for control_arm in ("B_synthetic", "B_perpendicular"):
            contrasts.append(
                _paired_contrast(
                    reachability_df,
                    arm_a="B_parallel",
                    arm_b=control_arm,
                    epoch="stimulation",
                    feature_set=feature_set,
                    value_name="fractional_distance_improvement",
                    cfg=cfg,
                    rng=rng,
                )
            )
    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(root / "confirmation_paired_contrasts.csv", index=False)

    plv_contrasts: dict[str, dict[str, Any]] = {}
    stimulation_epoch = epoch_df[epoch_df["epoch"] == "stimulation"]
    for population_name in ("E", "I"):
        pivot = stimulation_epoch.pivot(
            index="seed",
            columns="arm",
            values=f"{population_name}_plv",
        )
        differences = (
            pivot["B_parallel"] - pivot["B_sham"]
        ).to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(
            differences,
            rng=rng,
            n_bootstrap=int(cfg.analysis.n_bootstrap),
        )
        plv_contrasts[population_name] = {
            "mean_difference": float(np.mean(differences)),
            "ci_2.5": ci_low,
            "ci_97.5": ci_high,
            "paired_sign_flip_p": _sign_flip_p(
                differences,
                rng=rng,
                n_permutations=int(cfg.analysis.n_permutations),
            ),
        }

    primary = summary_df[
        (summary_df["arm"] == "B_parallel")
        & (summary_df["epoch"] == "stimulation")
        & (summary_df["feature_set"] == "fundamental_excluded")
    ].iloc[0]
    criteria = cfg.analysis.criteria

    def contrast_low(control_arm: str) -> float:
        row = contrast_df[
            (contrast_df["feature_set"] == "fundamental_excluded")
            & (contrast_df["arm_b"] == control_arm)
        ].iloc[0]
        return float(row["ci_2.5"])

    checks = {
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
        "beyond_synthetic_control": bool(contrast_low("B_synthetic") > 0.0),
        "orientation_specific": bool(contrast_low("B_perpendicular") > 0.0),
        "population_plv_increase": bool(
            any(value["ci_2.5"] > 0.0 for value in plv_contrasts.values())
        ),
        "baseline_causality": bool(
            float(baseline_df["baseline_relative_rms_error_vs_B_sham"].max())
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
    }
    if not bool(criteria.require_beyond_synthetic_control):
        checks["beyond_synthetic_control"] = True
    if not bool(criteria.require_orientation_specificity):
        checks["orientation_specific"] = True
    if not bool(criteria.require_population_plv_increase):
        checks["population_plv_increase"] = True

    mechanistic_modulation = bool(all(checks.values()))
    a_like_reachability = bool(
        mechanistic_modulation
        and float(primary["equivalent_seed_fraction"])
        >= float(criteria.minimum_equivalent_seed_fraction)
    )
    post_primary = summary_df[
        (summary_df["arm"] == "B_parallel")
        & (summary_df["epoch"] == "post")
        & (summary_df["feature_set"] == "fundamental_excluded")
    ].iloc[0]
    conclusion = {
        "fixed_protocol": OmegaConf.to_container(
            cfg.analysis.protocol,
            resolve=True,
        ),
        "primary_endpoint": primary.to_dict(),
        "post_stimulation_endpoint": post_primary.to_dict(),
        "mechanistic_checks": checks,
        "plv_contrasts_parallel_minus_sham": plv_contrasts,
        "evidence_of_mechanistic_directional_modulation": mechanistic_modulation,
        "evidence_of_A_like_reachability": a_like_reachability,
        "standardizers": standardizers,
        "calibration_seeds": calibration_seeds,
        "confirmation_seeds": confirmation_seeds,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        "interpretation_boundary": (
            "A positive result demonstrates acute output-state compensation in "
            "this static 40-cell model, not restoration of I-to-E weights, "
            "plasticity, depression treatment, or artifact-free human EEG."
        ),
    }
    with (root / "experiment_conclusion.json").open("w", encoding="utf-8") as handle:
        json.dump(conclusion, handle, indent=2)

    if bool(cfg.experiment.plot):
        _plot_results(reachability_df, episodes, root)

    print("\n### Primary confirmation")
    print(primary.to_string())
    print("\n### Mechanistic checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nMechanistic directional modulation: "
        f"{'PASSED' if mechanistic_modulation else 'NOT PASSED'}"
    )
    print(
        "A-like reachability: "
        f"{'PASSED' if a_like_reachability else 'NOT PASSED'}"
    )
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
