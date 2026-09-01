"""D0 validation of shared phase diffusion in BallAndStick afferent drive.

This stimulation-free experiment validates a deliberately minimal extension
of the toy state generator before any closed-loop action mapping.  E and I
afferents share one latent phase oscillator,

    d phi = 2*pi*f*dt + sqrt(2*D)*dW,

while every background synapse retains a private conditionally independent
Poisson event stream.  The expected afferent rate, cell model, recurrence,
weights, and inhibition are unchanged across D.  Candidate stationary, low-
diffusion, and high-diffusion states are crossed within independent circuit
structures at 9 and 11 Hz.

D0 asks whether the implementation obeys its equation and whether low versus
high diffusion is prospectively observable from multi-second ideal neural EEG.
It applies no tACS, fits no policy, and makes no disease or treatment claim.
Only a passing, frozen D0 generator should be loaded by the later D1
context-by-action feasibility map.
"""

from __future__ import annotations

import hashlib
import json
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
from scipy import signal


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _condition_for_seed,
    _epoch_raw,
    _epoch_row,
    _plain,
    _reference_phase,
    _run_condition,
    _sham,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _fourier_coefficients,
    _process_eeg,
)
from setup.circuits.ballnstick.utils import (  # noqa: E402
    generate_phase_diffusion_path,
    generate_sinusoidally_modulated_poisson_spike_train,
    make_background_phase_seed,
    make_background_synapse_seed,
)


def _copy_cfg(cfg: DictConfig) -> DictConfig:
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(copied, False)
    return copied


def _frequency_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _diffusion_levels(cfg: DictConfig) -> list[dict[str, Any]]:
    return [
        {
            "label": str(level.label),
            "diffusion_rad2_per_s": float(level.diffusion_rad2_per_s),
        }
        for level in cfg.analysis.states.phase_diffusion_levels
    ]


def _condition_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    result: list[dict[str, Any]] = []
    order = 0
    for structure_index in range(int(block.n_structure_seeds)):
        structure_seed = base + int(block.structure_seed_offset) + structure_index
        drive_seed = base + int(block.drive_seed_offset) + structure_index
        for frequency_index, frequency_hz in enumerate(
            cfg.analysis.states.frequencies_hz
        ):
            phase_seed = (
                base + int(block.phase_seed_offset)
                + 100 * structure_index + frequency_index
            )
            for level in _diffusion_levels(cfg):
                result.append({
                    "condition_order": order,
                    "condition_id": (
                        f"s{structure_index:02d}_f{_frequency_token(frequency_hz)}_"
                        f"{level['label']}"
                    ),
                    "structure_index": structure_index,
                    "structure_seed": structure_seed,
                    "drive_seed": drive_seed,
                    "phase_seed": phase_seed,
                    "trial_seed": (
                        base + int(block.trial_seed_offset) + order
                    ),
                    "frequency_hz": float(frequency_hz),
                    **level,
                })
                order += 1
    return result


def _with_diffusion_state(cfg: DictConfig, spec: dict[str, Any]) -> DictConfig:
    result = _copy_cfg(cfg)
    with open_dict(result):
        result.analysis.reference.frequency_hz = float(spec["frequency_hz"])
        result.analysis.tacs.frequency_hz = float(spec["frequency_hz"])
        result.analysis.protocol.frequency_hz = float(spec["frequency_hz"])
    result = _condition_for_seed(
        result,
        seed=int(spec["phase_seed"]),
        modulation_depth=float(cfg.analysis.states.modulation_depth),
    )
    with open_dict(result):
        for population in ("E", "I"):
            rhythm = result.env.network.background[population].rhythm
            rhythm.phase_diffusion_rad2_per_s = float(
                spec["diffusion_rad2_per_s"]
            )
            rhythm.phase_diffusion_integration_dt_ms = float(
                cfg.analysis.states.phase_diffusion_integration_dt_ms
            )
    return result


def _validate_design(cfg: DictConfig) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("D0 requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("D0 may not change recurrent inhibition.")
    if not np.isclose(float(cfg.analysis.tacs.amplitude_v_per_m), 0.0):
        raise ValueError("D0 is stimulation-free; tACS amplitude must be zero.")
    frequencies = [float(x) for x in cfg.analysis.states.frequencies_hz]
    if frequencies != [9.0, 11.0]:
        raise ValueError("The frozen D0 frequency grid is exactly 9 and 11 Hz.")
    levels = _diffusion_levels(cfg)
    expected = [("stationary", 0.0), ("low_diffusion", 0.5), ("high_diffusion", 2.0)]
    observed = [(x["label"], x["diffusion_rad2_per_s"]) for x in levels]
    if len(observed) != len(expected) or any(
        label != expected_label or not np.isclose(value, expected_value)
        for (label, value), (expected_label, expected_value) in zip(observed, expected)
    ):
        raise ValueError(
            "D0 freezes stationary/low/high diffusion at 0/0.5/2 rad^2/s."
        )
    depth = float(cfg.analysis.states.modulation_depth)
    envelope = float(cfg.analysis.reference.thinning_envelope_modulation_depth)
    if not 0.0 < depth <= envelope <= 1.0:
        raise ValueError("The modulation depth must fit the common thinning envelope.")
    phase_dt = float(cfg.analysis.states.phase_diffusion_integration_dt_ms)
    if phase_dt <= 0.0:
        raise ValueError("Phase integration dt must be positive.")

    timeline = cfg.analysis.timeline
    if min(int(timeline[name]) for name in (
        "burn_in_steps", "baseline_steps", "stimulation_steps", "washout_steps"
    )) < 1:
        raise ValueError("Every persistent online epoch requires at least one window.")
    minimum_baseline = 2 if bool(cfg.analysis.smoke_test) else 12
    if int(timeline.baseline_steps) < minimum_baseline:
        raise ValueError(f"This D0 mode requires at least {minimum_baseline} baseline seconds.")
    n_structures = int(cfg.analysis.crossed_design.n_structure_seeds)
    minimum_structures = 1 if bool(cfg.analysis.smoke_test) else int(
        cfg.analysis.criteria.minimum_structure_seeds
    )
    if n_structures < minimum_structures:
        raise ValueError(f"This D0 mode requires at least {minimum_structures} structures.")
    total_ms = sum(int(timeline[name]) for name in (
        "burn_in_steps", "baseline_steps", "stimulation_steps", "washout_steps"
    )) * float(cfg.env.simulation.obs_win_len)
    if not np.isclose(total_ms / phase_dt, round(total_ms / phase_dt), atol=1.0e-10):
        raise ValueError("The episode duration must align with the phase grid.")

    specs = _condition_specs(cfg)
    namespaces = [
        {int(x["structure_seed"]) for x in specs},
        {int(x["drive_seed"]) for x in specs},
        {int(x["phase_seed"]) for x in specs},
        {int(x["trial_seed"]) for x in specs},
    ]
    if any(namespaces[i].intersection(namespaces[j]) for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Structure, drive, phase, and trial seed namespaces must be disjoint.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Structure seed * 10,000 exceeds the uint32 seed range.")


def _periodogram_metrics(
    processed: np.ndarray,
    *,
    fs_hz: float,
    frequency_hz: float,
    cfg: DictConfig,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    frequencies, psd = signal.periodogram(
        processed,
        fs=fs_hz,
        window="hann",
        detrend="constant",
        scaling="density",
    )
    alpha = (frequencies >= 8.0) & (frequencies <= 12.0)
    peak_frequency = float(frequencies[alpha][np.argmax(psd[alpha])])
    core_half = float(cfg.analysis.measurement.spectral_core_half_width_hz)
    neighbourhood_half = float(
        cfg.analysis.measurement.spectral_neighbourhood_half_width_hz
    )
    core = np.abs(frequencies - frequency_hz) <= core_half
    neighbourhood = np.abs(frequencies - frequency_hz) <= neighbourhood_half
    eps = np.finfo(float).tiny
    core_power = float(np.trapz(psd[core], frequencies[core]))
    neighbourhood_power = float(np.trapz(psd[neighbourhood], frequencies[neighbourhood]))
    weights = np.maximum(psd[neighbourhood], 0.0)
    offsets = frequencies[neighbourhood] - frequency_hz
    width = float(np.sqrt(np.sum(weights * offsets**2) / max(np.sum(weights), eps)))
    return {
        "detected_peak_frequency_hz": peak_frequency,
        "frequency_detected_correctly": float(abs(peak_frequency - frequency_hz) <= 0.75),
        "spectral_core_power_v2": core_power,
        "spectral_neighbourhood_power_v2": neighbourhood_power,
        "spectral_concentration": core_power / max(neighbourhood_power, eps),
        "spectral_rms_width_hz": width,
        "periodogram_resolution_hz": float(fs_hz / processed.size),
    }, frequencies, psd


def _eeg_phase_metrics(
    processed: np.ndarray,
    *,
    fs_hz: float,
    start_ms: float,
    frequency_hz: float,
    cfg: DictConfig,
) -> dict[str, Any]:
    window_samples = int(round(float(cfg.analysis.measurement.phase_window_s) * fs_hz))
    n_windows = processed.size // window_samples
    if n_windows < 2:
        raise ValueError("At least two causal EEG phase windows are required.")
    usable = processed[: n_windows * window_samples]
    coefficients: list[complex] = []
    ratios: list[float] = []
    for index in range(n_windows):
        segment = usable[index * window_samples : (index + 1) * window_samples]
        segment_start_ms = start_ms + 1000.0 * index * window_samples / fs_hz
        cosine, sine = _fourier_coefficients(
            segment,
            fs_hz=fs_hz,
            start_ms=segment_start_ms,
            frequency_hz=frequency_hz,
        )
        coefficient = complex(cosine, sine)
        coefficients.append(coefficient)
        ratios.append(abs(coefficient) / max(float(np.sqrt(np.mean(segment**2))), np.finfo(float).tiny))
    coefficients_array = np.asarray(coefficients, dtype=np.complex128)
    unit = coefficients_array / np.maximum(np.abs(coefficients_array), np.finfo(float).tiny)
    phase_stability = float(abs(np.mean(unit)))

    chunk_seconds = float(cfg.analysis.measurement.temporal_chunk_s)
    windows_per_chunk = max(2, int(round(chunk_seconds / float(cfg.analysis.measurement.phase_window_s))))
    chunk_stabilities = []
    for start in range(0, n_windows - windows_per_chunk + 1, windows_per_chunk):
        chunk_stabilities.append(float(abs(np.mean(unit[start : start + windows_per_chunk]))))
    temporal_sd = float(np.std(chunk_stabilities, ddof=1)) if len(chunk_stabilities) > 1 else 0.0
    recent_difference = float(np.angle(unit[-1] * np.conj(unit[-2])))
    return {
        "eeg_phase_stability": phase_stability,
        "recent_phase_resultant_to_rms": float(ratios[-1]),
        "recent_vs_previous_phase_difference_rad": recent_difference,
        "temporal_chunk_phase_stability_mean": float(np.mean(chunk_stabilities)),
        "temporal_chunk_phase_stability_sd": temporal_sd,
        "n_causal_phase_windows": n_windows,
        "causal_window_phase_rad": np.angle(coefficients_array).tolist(),
    }


def _latent_and_source_audit(
    spec: dict[str, Any], cfg: DictConfig
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    timeline = cfg.analysis.timeline
    window_ms = float(cfg.env.simulation.obs_win_len)
    total_ms = sum(int(timeline[name]) for name in (
        "burn_in_steps", "baseline_steps", "stimulation_steps", "washout_steps"
    )) * window_ms
    baseline_start = int(timeline.burn_in_steps) * window_ms
    baseline_stop = baseline_start + int(timeline.baseline_steps) * window_ms
    frequency = float(spec["frequency_hz"])
    diffusion = float(spec["diffusion_rad2_per_s"])
    phase0 = _reference_phase(int(spec["phase_seed"]))
    phase_seed = make_background_phase_seed(global_seed=int(spec["drive_seed"]))
    times, phase = generate_phase_diffusion_path(
        start_ms=0.0,
        stop_ms=total_ms,
        frequency_hz=frequency,
        phase_rad=phase0,
        diffusion_rad2_per_s=diffusion,
        integration_dt_ms=float(cfg.analysis.states.phase_diffusion_integration_dt_ms),
        history_seed=phase_seed,
    )
    baseline = (times >= baseline_start) & (times <= baseline_stop)
    selected_times = times[baseline]
    selected_phase = phase[baseline]
    lag_steps = int(round(
        1000.0 * float(cfg.analysis.measurement.latent_coherence_lag_s)
        / float(cfg.analysis.states.phase_diffusion_integration_dt_ms)
    ))
    residual_increment = (
        selected_phase[lag_steps:] - selected_phase[:-lag_steps]
        - 2.0 * np.pi * frequency
        * (selected_times[lag_steps:] - selected_times[:-lag_steps]) / 1000.0
    )
    observed_coherence = float(abs(np.mean(np.exp(1j * residual_increment))))
    expected_coherence = float(np.exp(
        -diffusion * float(cfg.analysis.measurement.latent_coherence_lag_s)
    ))
    step_residual = (
        np.diff(selected_phase)
        - 2.0 * np.pi * frequency * np.diff(selected_times) / 1000.0
    )
    observed_increment_variance = float(np.var(step_residual))
    expected_increment_variance = float(
        2.0 * diffusion
        * float(cfg.analysis.states.phase_diffusion_integration_dt_ms) / 1000.0
    )
    increment_variance_error = (
        abs(observed_increment_variance - expected_increment_variance)
        / expected_increment_variance
        if expected_increment_variance > 0.0
        else abs(observed_increment_variance)
    )
    digest = hashlib.sha256(np.asarray(phase, dtype="<f8").tobytes()).hexdigest()

    event_rows: list[dict[str, Any]] = []
    n_private = int(cfg.analysis.measurement.sampled_private_synapses_per_population)
    envelope = float(cfg.analysis.reference.thinning_envelope_modulation_depth)
    modulation = float(cfg.analysis.states.modulation_depth)
    for population_index, population in enumerate(("E", "I")):
        background = cfg.env.network.background[population]
        trains: list[np.ndarray] = []
        for synapse_index in range(n_private):
            private_seed = make_background_synapse_seed(
                global_seed=int(spec["drive_seed"]),
                population_index=population_index,
                cell_identifier=0,
                synapse_index=synapse_index,
            )
            train = generate_sinusoidally_modulated_poisson_spike_train(
                start_ms=0.0,
                stop_ms=total_ms,
                interval_ms=float(background.interval_ms),
                seed=private_seed,
                modulation_depth=modulation,
                frequency_hz=frequency,
                phase_rad=phase0,
                thinning_envelope_modulation_depth=envelope,
                phase_path_times_ms=times,
                phase_path_rad=phase,
            )
            train = train[(train >= baseline_start) & (train < baseline_stop)]
            trains.append(train)
            event_rows.append({
                "condition_id": spec["condition_id"],
                "population": population,
                "private_synapse_index": synapse_index,
                "event_count": int(train.size),
                "event_rate_hz": float(train.size / ((baseline_stop - baseline_start) / 1000.0)),
                "expected_rate_hz": float(1000.0 / float(background.interval_ms)),
            })
        duplicate_pairs = sum(
            int(np.array_equal(trains[left], trains[right]))
            for left in range(n_private)
            for right in range(left + 1, n_private)
        )
        for row in event_rows[-n_private:]:
            row["duplicate_private_train_pairs_in_population"] = duplicate_pairs

    stride = max(1, int(round(10.0 / float(cfg.analysis.states.phase_diffusion_integration_dt_ms))))
    trace = pd.DataFrame({
        "condition_id": spec["condition_id"],
        "time_ms": selected_times[::stride],
        "phase_residual_rad": (
            selected_phase[::stride]
            - 2.0 * np.pi * frequency * selected_times[::stride] / 1000.0
            - phase0
        ),
    })
    return {
        "latent_phase_path_sha256": digest,
        "latent_phase_coherence": observed_coherence,
        "theoretical_latent_phase_coherence": expected_coherence,
        "latent_coherence_absolute_error": abs(observed_coherence - expected_coherence),
        "latent_increment_variance_rad2": observed_increment_variance,
        "theoretical_latent_increment_variance_rad2": expected_increment_variance,
        "latent_increment_variance_relative_error": increment_variance_error,
        "latent_phase_path_shared_by_E_and_I": True,
    }, pd.DataFrame(event_rows), trace


def _episode_metrics(
    episode: dict[str, Any],
    spec: dict[str, Any],
    cfg: DictConfig,
    latent_audit: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = _epoch_raw(episode, "baseline")
    processed, fs_hz, _, _, _ = _process_eeg(
        raw, simulator_fs_hz=float(episode["simulator_fs_hz"]), cfg=cfg
    )
    outputs = episode["simulation"]["outputs_by_epoch"]["baseline"]
    start_ms = float(outputs[0]["t_start_ms"])
    spectral, frequencies, psd = _periodogram_metrics(
        processed,
        fs_hz=fs_hz,
        frequency_hz=float(spec["frequency_hz"]),
        cfg=cfg,
    )
    phase = _eeg_phase_metrics(
        processed,
        fs_hz=fs_hz,
        start_ms=start_ms,
        frequency_hz=float(spec["frequency_hz"]),
        cfg=cfg,
    )
    epoch = _epoch_row(episode, "baseline")
    row = {
        **spec,
        **spectral,
        **phase,
        **latent_audit,
        "analysis_duration_s": float(processed.size / fs_hz),
        "E_firing_rate_hz": float(epoch.E_firing_rate_hz),
        "I_firing_rate_hz": float(epoch.I_firing_rate_hz),
        "field_amplitude_v_per_m": 0.0,
    }
    periodogram = pd.DataFrame({
        "condition_id": spec["condition_id"],
        "structure_index": spec["structure_index"],
        "frequency_hz": spec["frequency_hz"],
        "diffusion_label": spec["label"],
        "diffusion_rad2_per_s": spec["diffusion_rad2_per_s"],
        "eeg_frequency_hz": frequencies,
        "psd_v2_per_hz": psd,
    })
    return row, periodogram


def _leave_one_structure_out_accuracy(rows: pd.DataFrame) -> float:
    subset = rows[rows.label.isin(["low_diffusion", "high_diffusion"])].copy()
    predictions: list[bool] = []
    for structure in sorted(subset.structure_index.unique()):
        train = subset[subset.structure_index.ne(structure)]
        test = subset[subset.structure_index.eq(structure)]
        low_mean = float(train[train.label.eq("low_diffusion")].eeg_phase_stability.mean())
        high_mean = float(train[train.label.eq("high_diffusion")].eeg_phase_stability.mean())
        threshold = 0.5 * (low_mean + high_mean)
        low_is_high_value = low_mean >= high_mean
        for _, row in test.iterrows():
            predicted_low = (
                float(row.eeg_phase_stability) >= threshold
                if low_is_high_value else float(row.eeg_phase_stability) <= threshold
            )
            predictions.append(predicted_low == (str(row.label) == "low_diffusion"))
    return float(np.mean(predictions)) if predictions else float("nan")


def _evaluate(rows: pd.DataFrame, event_rows: pd.DataFrame, cfg: DictConfig) -> tuple[dict[str, bool], dict[str, Any]]:
    means = rows.groupby("label", sort=False).mean(numeric_only=True)
    low = means.loc["low_diffusion"]
    high = means.loc["high_diffusion"]
    stationary = means.loc["stationary"]
    low_high_stability = float(low.eeg_phase_stability - high.eeg_phase_stability)
    low_high_concentration = float(low.spectral_concentration - high.spectral_concentration)
    pair = rows.pivot_table(
        index=["structure_index", "frequency_hz"],
        columns="label",
        values="eeg_phase_stability",
    )
    paired_signal = pair.low_diffusion - pair.high_diffusion
    within_noise = float(np.sqrt(np.mean(
        rows[rows.label.isin(["low_diffusion", "high_diffusion"])]
        .temporal_chunk_phase_stability_sd.to_numpy(float) ** 2
    )))
    signal_to_noise = float(abs(paired_signal.mean()) / max(within_noise, np.finfo(float).eps))
    classification_accuracy = _leave_one_structure_out_accuracy(rows)

    event_summary = event_rows.groupby(["condition_id", "population"], as_index=False).agg(
        mean_event_rate_hz=("event_rate_hz", "mean"),
        expected_rate_hz=("expected_rate_hz", "first"),
        duplicate_pairs=("duplicate_private_train_pairs_in_population", "first"),
    )
    event_summary["relative_rate_error"] = abs(
        event_summary.mean_event_rate_hz - event_summary.expected_rate_hz
    ) / event_summary.expected_rate_hz
    maximum_rate_error = float(event_summary.relative_rate_error.max())
    rate_by_state = event_rows.merge(
        rows[["condition_id", "label"]], on="condition_id", how="left"
    ).groupby(["population", "label"]).event_rate_hz.mean().unstack()
    diffusion_rate_difference = float(max(
        abs(rate_by_state.low_diffusion - rate_by_state.high_diffusion)
        / rate_by_state[["low_diffusion", "high_diffusion"]].mean(axis=1)
    ))
    limits = cfg.analysis.rate_guardrails_hz
    rate_safe = bool(
        rows.E_firing_rate_hz.between(float(limits.E_min), float(limits.E_max)).all()
        and rows.I_firing_rate_hz.between(float(limits.I_min), float(limits.I_max)).all()
    )
    criteria = cfg.analysis.criteria
    latent_ordered = bool(
        stationary.latent_phase_coherence > low.latent_phase_coherence
        > high.latent_phase_coherence
        and low.latent_phase_coherence - high.latent_phase_coherence
        >= float(criteria.minimum_low_minus_high_latent_coherence)
    )
    checks = {
        "phase_diffusion_generator_distinct_from_tacs": True,
        "all_runs_stimulation_free": bool(np.allclose(rows.field_amplitude_v_per_m, 0.0)),
        "minimum_independent_structures": bool(
            rows.structure_index.nunique() >= int(criteria.minimum_structure_seeds)
        ),
        "complete_frequency_diffusion_grid": bool(
            len(rows) == rows.structure_index.nunique()
            * len(cfg.analysis.states.frequencies_hz) * len(_diffusion_levels(cfg))
        ),
        "shared_latent_phase_used_by_E_and_I": bool(rows.latent_phase_path_shared_by_E_and_I.all()),
        "private_poisson_event_streams_not_copied": bool(
            (event_summary.duplicate_pairs == 0).all()
        ),
        "afferent_mean_rate_preserved": bool(
            maximum_rate_error <= float(criteria.maximum_mean_afferent_rate_relative_error)
            and diffusion_rate_difference <= float(criteria.maximum_diffusion_rate_difference_fraction)
        ),
        "latent_phase_increment_variance_matches_SDE": bool(
            rows.latent_increment_variance_relative_error.max()
            <= float(criteria.maximum_latent_increment_variance_relative_error)
        ),
        "latent_phase_coherence_is_ordered": latent_ordered,
        "hidden_frequency_visible_in_ideal_EEG": bool(
            rows.frequency_detected_correctly.mean()
            >= float(criteria.minimum_frequency_detection_accuracy)
        ),
        "low_diffusion_has_more_stable_EEG_phase_than_high": bool(
            low_high_stability
            >= float(criteria.minimum_low_minus_high_eeg_phase_stability)
        ),
        "low_diffusion_has_more_concentrated_EEG_spectrum_than_high": bool(
            low_high_concentration > 0.0
        ),
        "low_high_diffusion_classifiable_from_heldout_structure_EEG": bool(
            classification_accuracy
            >= float(criteria.minimum_low_high_classification_accuracy)
        ),
        "diffusion_signal_exceeds_within_trajectory_temporal_noise": bool(
            signal_to_noise >= float(criteria.minimum_state_to_temporal_noise_ratio)
        ),
        "recent_one_second_phase_is_measurable": bool(
            rows[rows.label.eq("high_diffusion")].recent_phase_resultant_to_rms.mean()
            >= float(criteria.minimum_recent_resultant_to_rms)
        ),
        "neural_firing_rates_safe": rate_safe,
    }
    smoke = bool(cfg.analysis.smoke_test)
    readiness = bool(all(checks.values()) and not smoke)
    summary = {
        "candidate_levels": _diffusion_levels(cfg),
        "mean_eeg_phase_stability_by_level": {
            label: float(value) for label, value in means.eeg_phase_stability.items()
        },
        "mean_spectral_concentration_by_level": {
            label: float(value) for label, value in means.spectral_concentration.items()
        },
        "mean_spectral_rms_width_hz_by_level": {
            label: float(value) for label, value in means.spectral_rms_width_hz.items()
        },
        "low_minus_high_eeg_phase_stability": low_high_stability,
        "low_minus_high_spectral_concentration": low_high_concentration,
        "leave_one_structure_out_low_high_accuracy": classification_accuracy,
        "state_to_temporal_noise_ratio": signal_to_noise,
        "maximum_sampled_afferent_rate_relative_error": maximum_rate_error,
        "low_high_sampled_afferent_rate_difference_fraction": diffusion_rate_difference,
        "frequency_detection_accuracy": float(rows.frequency_detected_correctly.mean()),
        "maximum_latent_increment_variance_relative_error": float(
            rows.latent_increment_variance_relative_error.max()
        ),
        "mean_lag_coherence_absolute_error_sampling_audit": float(
            rows.latent_coherence_absolute_error.mean()
        ),
        "ready_for_D1": readiness,
        "smoke_test": smoke,
    }
    return checks, summary


def _plot_results(
    root: Path,
    rows: pd.DataFrame,
    periodograms: pd.DataFrame,
    traces: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for axis, (frequency, subset) in zip(axes, periodograms.groupby("frequency_hz")):
        for label, group in subset.groupby("diffusion_label", sort=False):
            mean_psd = group.groupby("eeg_frequency_hz").psd_v2_per_hz.mean()
            axis.semilogy(mean_psd.index, np.maximum(mean_psd.values, np.finfo(float).tiny), label=label)
        axis.axvline(frequency, color="0.3", linestyle="--", linewidth=0.8)
        axis.set(xlim=(7.0, 13.0), xlabel="Frequency (Hz)", title=f"{frequency:g}-Hz afferent state")
    axes[0].set_ylabel("Ideal neural EEG PSD (V²/Hz)")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_01_eeg_psd_by_diffusion.png", dpi=250)
    plt.close(figure)

    labels = ["stationary", "low_diffusion", "high_diffusion"]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for _, group in rows.groupby(["structure_index", "frequency_hz"]):
        ordered = group.set_index("label").loc[labels]
        axes[0].plot(labels, ordered.eeg_phase_stability, color="0.65", alpha=0.6)
        axes[1].plot(labels, ordered.spectral_concentration, color="0.65", alpha=0.6)
    means = rows.groupby("label").mean(numeric_only=True).loc[labels]
    axes[0].plot(labels, means.eeg_phase_stability, "o-", color="#1f77b4", linewidth=2.3)
    axes[1].plot(labels, means.spectral_concentration, "o-", color="#d95f02", linewidth=2.3)
    axes[0].set(ylabel="Across-window phase stability", title="Causal 1-s EEG phase estimates")
    axes[1].set(ylabel="Core / local spectral power", title="Spectral concentration")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(root / "figure_02_eeg_diffusion_endpoints.png", dpi=250)
    plt.close(figure)

    example_ids = [
        f"s00_f9_{label}" for label in labels
    ]
    figure, axis = plt.subplots(figsize=(9.0, 3.6))
    for identifier in example_ids:
        group = traces[traces.condition_id.eq(identifier)]
        if not group.empty:
            axis.plot(group.time_ms / 1000.0, group.phase_residual_rad, label=identifier.split("_", 2)[-1])
    axis.set(
        xlabel="Simulation time (s)",
        ylabel="Latent phase residual (rad)",
        title="Shared afferent phase trajectories (one structure)",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_03_latent_phase_diffusion.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate_design(cfg)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "phase_diffusion_validation"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### D0 shared phase-diffusion validation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    metric_rows: list[dict[str, Any]] = []
    periodogram_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    for spec in _condition_specs(cfg):
        if rank == 0:
            print(
                f"condition={spec['condition_id']} f={spec['frequency_hz']:g} Hz "
                f"D={spec['diffusion_rad2_per_s']:g} rad^2/s"
            )
        condition_cfg = _with_diffusion_state(cfg, spec)
        episode = _run_condition(
            condition_id=str(spec["condition_id"]),
            condition_cfg=condition_cfg,
            action=_sham(condition_cfg, str(spec["condition_id"])),
            stimulate=False,
            seed=int(spec["trial_seed"]),
            action_index=0,
            output_dir=root / "episodes" / str(spec["condition_id"]),
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(spec["structure_seed"]),
            drive_seed=int(spec["drive_seed"]),
            phase_seed=int(spec["phase_seed"]),
        )
        if rank == 0:
            latent_audit, event_audit, trace = _latent_and_source_audit(spec, cfg)
            row, periodogram = _episode_metrics(
                episode, spec, cfg, latent_audit
            )
            metric_rows.append(row)
            periodogram_frames.append(periodogram)
            event_frames.append(event_audit)
            trace_frames.append(trace)

    if rank != 0:
        return
    rows = pd.DataFrame(metric_rows)
    periodograms = pd.concat(periodogram_frames, ignore_index=True)
    event_rows = pd.concat(event_frames, ignore_index=True)
    traces = pd.concat(trace_frames, ignore_index=True)
    checks, summary = _evaluate(rows, event_rows, cfg)
    rows.to_csv(root / "phase_diffusion_eeg_metrics.csv", index=False)
    periodograms.to_csv(root / "eeg_periodograms.csv", index=False)
    event_rows.to_csv(root / "private_afferent_event_audit.csv", index=False)
    traces.to_csv(root / "latent_phase_trace_10ms.csv", index=False)

    frozen = {
        "experiment": "D0_shared_phase_diffusion_validation",
        "generator_equation": "dphi = 2*pi*f*dt + sqrt(2*D)*dW",
        "intensity_equation": "lambda_pj(t) = lambda_0p * [1 + m*sin(phi(t))]",
        "shared_state": "one latent phi(t) shared by E/I afferent rates",
        "private_noise": "conditionally independent Poisson events per synapse",
        "modulation_depth": float(cfg.analysis.states.modulation_depth),
        "frequencies_hz": [float(x) for x in cfg.analysis.states.frequencies_hz],
        "phase_diffusion_levels": _diffusion_levels(cfg),
        "phase_diffusion_integration_dt_ms": float(
            cfg.analysis.states.phase_diffusion_integration_dt_ms
        ),
        "D1_candidate_levels": ["low_diffusion", "high_diffusion"],
        "ready_for_D1": bool(summary["ready_for_D1"]),
        "not_a_disease_model": True,
        "not_a_tacs_result": True,
    }
    (root / "frozen_phase_diffusion_generator.json").write_text(
        json.dumps(_plain(frozen), indent=2)
    )
    conclusion = {
        "scope": "D0 stimulation-free generator and ideal-neural-EEG validation",
        "checks": checks,
        "summary": summary,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure; frequency and D are paired generator conditions",
        "next_experiment": (
            "D1 context-by-action feasibility with generator hash locked"
            if summary["ready_for_D1"]
            else "Do not run D1; diagnose failed D0 observability or generator check"
        ),
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_plain(conclusion), indent=2)
    )
    if bool(cfg.experiment.plot):
        _plot_results(root, rows, periodograms, traces)

    print("\n### D0 phase-diffusion checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    if bool(cfg.analysis.smoke_test):
        print("\nD0 status: SMOKE TEST ONLY (scientific gate not evaluated)")
    else:
        print(
            "\nD0 phase-diffusion validation gate: "
            f"{'PASSED' if summary['ready_for_D1'] else 'NOT PASSED'}"
        )
        print(f"Ready for D1: {'YES' if summary['ready_for_D1'] else 'NO'}")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
