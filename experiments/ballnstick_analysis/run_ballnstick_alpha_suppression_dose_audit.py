"""Explore whether a lower frozen-phase tACS dose suppresses toy alpha EEG.

This is an explicitly exploratory mechanism and dose-response experiment.  It
loads the previously frozen A/B state generator and the 10-Hz, EEG-relative
180-degree phase convention, but it does not reuse the unsuccessful 0.8-V/m
confirmation as positive evidence.  Matched circuit seeds receive sham and a
small preregistered axial-field dose grid.

The primary directional endpoint is the paired reduction in ideal neural-only
EEG 8--12-Hz power.  Exact 10-Hz Fourier coefficients are additionally
decomposed into their cross/interference and induced-component terms.  The
online simulator's three-component total current-dipole trace is retained as a
neural-source diagnostic.  Spike PPC and firing rates remain hidden mechanism
and safety variables; they never select the action from an EEG observation.

Any dose selected here is only an exploratory candidate and requires a new,
disjoint confirmation experiment before use as an RL action.
"""

from __future__ import annotations

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
from omegaconf import DictConfig, OmegaConf


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    A_HIGH,
    B_LOW,
    _action,
    _condition_for_seed,
    _epoch_raw,
    _epoch_row,
    _feature_from_raw,
    _fourier_coefficients,
    _plain,
    _run_condition,
    _sham,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_confirmation import (  # noqa: E402
    _load_frozen_source,
    _phase_quality_row,
    _summary,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _relative_rate_safe,
)


PRIMARY_METRIC = "alpha_suppression_log10"
MECHANISM_METRICS = (
    PRIMARY_METRIC,
    "target_distance_improvement_log10",
    "coherent_10hz_suppression_v",
    "alpha_peak_prominence_reduction_db",
    "E_ppc_reduction",
    "I_ppc_reduction",
    "coherent_interference_cross_term_fraction",
    "coherent_induced_component_fraction",
    "coherent_net_change_fraction",
    "induced_dipole_10hz_vector_nA_um",
)


def _seeds(cfg: DictConfig) -> list[int]:
    first = int(cfg.experiment.seed) + int(cfg.analysis.exploration.seed_offset)
    return [first + index for index in range(int(cfg.analysis.exploration.n_seeds))]


def _dose_id(amplitude_v_per_m: float) -> str:
    token = f"{float(amplitude_v_per_m):.6g}".replace("-", "m").replace(".", "p")
    return f"A_tacs_180deg_{token}_vpm"


def _dose_values(cfg: DictConfig) -> list[float]:
    return [float(value) for value in cfg.analysis.doses_v_per_m]


def _validate_design(cfg: DictConfig, frozen: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("The alpha dose audit requires the online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("Every dose-audit arm requires inhibition_scale=1.")
    if int(cfg.analysis.exploration.n_seeds) < 1:
        raise ValueError("At least one exploratory circuit seed is required.")

    doses = _dose_values(cfg)
    maximum = float(cfg.analysis.maximum_field_v_per_m)
    if not doses or len(doses) != len(set(doses)):
        raise ValueError("doses_v_per_m must be nonempty and unique.")
    if doses != sorted(doses):
        raise ValueError("doses_v_per_m must be ordered from low to high.")
    if any(not 0.0 < dose <= maximum for dose in doses):
        raise ValueError("Every active dose must lie in (0, maximum_field_v_per_m].")
    frozen_amplitude = float(frozen["protocol"]["amplitude_v_per_m"])
    if bool(cfg.analysis.require_frozen_amplitude_in_grid) and not any(
        np.isclose(dose, frozen_amplitude) for dose in doses
    ):
        raise ValueError("The audit grid must retain the frozen 0.8-V/m dose.")
    if not np.isclose(float(frozen["protocol"]["frequency_hz"]), 10.0):
        raise ValueError("The frozen actuator frequency must remain 10 Hz.")
    if not np.isclose(
        float(frozen["protocol"]["selected_eeg_relative_phase_offset_rad"]),
        np.pi,
    ):
        raise ValueError("The audit must retain the frozen 180-degree offset.")

    previous_seeds = set(range(160001, 162005)) | set(range(170001, 170009))
    seeds = set(_seeds(cfg))
    if previous_seeds.intersection(seeds):
        raise ValueError("Dose-audit seeds overlap pilot or confirmation seeds.")
    if max(seeds) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Circuit seeds are too large for seed * 10,000 mapping.")

    timeline = cfg.analysis.timeline
    if int(timeline.baseline_steps) < 4:
        raise ValueError("Phase-quality auditing requires at least four baseline windows.")
    if int(timeline.stimulation_steps) < 4:
        raise ValueError("Dose-response EEG estimation requires at least four windows.")
    stimulation_ms = (
        int(timeline.stimulation_steps) * float(cfg.env.simulation.obs_win_len)
    )
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    if trim_ms < float(timeline.block_ramp_ms) or 2.0 * trim_ms >= stimulation_ms:
        raise ValueError("Analysis trimming must remove both ramps and leave data.")


def _dipole_fourier(
    episode: dict[str, Any], *, epoch: str, cfg: DictConfig
) -> dict[str, Any]:
    """Return exact 10-Hz coefficients of the total current dipole."""
    outputs = episode["simulation"]["outputs_by_epoch"][epoch]
    chunks = []
    for output in outputs:
        probe_data = output.get("probe_data")
        if probe_data is None or len(probe_data) < 2:
            raise RuntimeError("Online output does not contain the current-dipole probe.")
        chunk = np.asarray(probe_data[1], dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] != 3:
            raise RuntimeError(
                f"Expected current dipole shape (3, samples), received {chunk.shape}."
            )
        chunks.append(chunk)
    raw = np.concatenate(chunks, axis=1)
    start_ms = float(outputs[0]["t_start_ms"])
    if epoch == "stimulation":
        trim_ms = float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
        trim_samples = int(round(trim_ms / float(cfg.env.network.dt)))
        raw = raw[:, trim_samples:-trim_samples]
        start_ms += trim_ms
    fs_hz = 1000.0 / float(cfg.env.network.dt)
    cosine = np.empty(3, dtype=float)
    sine = np.empty(3, dtype=float)
    for axis in range(3):
        cosine[axis], sine[axis] = _fourier_coefficients(
            raw[axis],
            fs_hz=fs_hz,
            start_ms=start_ms,
            frequency_hz=float(cfg.analysis.tacs.frequency_hz),
        )
    resultant = np.hypot(cosine, sine)
    return {
        "cosine_nA_um": cosine,
        "sine_nA_um": sine,
        "axis_resultant_nA_um": resultant,
        "vector_resultant_nA_um": float(
            np.sqrt(np.sum(cosine**2 + sine**2))
        ),
    }


def _complex_response_decomposition(
    *,
    sham_cosine: float,
    sham_sine: float,
    active_cosine: float,
    active_sine: float,
) -> dict[str, float]:
    """Decompose active-minus-sham coherent energy at the target frequency.

    If ``a`` is the sham Fourier vector and ``d`` the active-minus-sham
    response, then ``|a+d|^2-|a|^2 = 2 a.d + |d|^2``.  A negative cross term
    represents destructive interference, while the induced-component term is
    non-negative.  These are neural EEG coefficients, not electrode artifact.
    """
    sham = np.asarray([sham_cosine, sham_sine], dtype=float)
    active = np.asarray([active_cosine, active_sine], dtype=float)
    delta = active - sham
    baseline_energy = float(np.dot(sham, sham))
    cross = float(2.0 * np.dot(sham, delta))
    induced = float(np.dot(delta, delta))
    net = float(np.dot(active, active) - baseline_energy)
    scale = max(baseline_energy, np.finfo(float).tiny)
    return {
        "delta_eeg_10hz_cosine_v": float(delta[0]),
        "delta_eeg_10hz_sine_v": float(delta[1]),
        "induced_eeg_10hz_resultant_v": float(np.linalg.norm(delta)),
        "coherent_interference_cross_term_v2": cross,
        "coherent_induced_component_v2": induced,
        "coherent_net_change_v2": net,
        "coherent_interference_cross_term_fraction": cross / scale,
        "coherent_induced_component_fraction": induced / scale,
        "coherent_net_change_fraction": net / scale,
        "coherent_decomposition_error_v2": net - (cross + induced),
    }


def _field_removal_status(
    *, effect_log10: float, residual_log10: float, cfg: DictConfig
) -> tuple[bool, float]:
    """Audit field removal independently of whether the acute effect helped."""
    tolerance = max(
        float(cfg.analysis.criteria.maximum_washout_absolute_log10),
        float(cfg.analysis.criteria.maximum_washout_residual_fraction)
        * abs(float(effect_log10)),
    )
    return abs(float(residual_log10)) <= tolerance, tolerance


def _seed_dose_metrics(
    *,
    seed: int,
    episodes: dict[str, dict[str, Any]],
    phase_quality: dict[str, Any],
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    a = _epoch_row(episodes[A_HIGH])
    b = _epoch_row(episodes[B_LOW])
    a_baseline = _epoch_row(episodes[A_HIGH], "baseline")
    a_washout = _epoch_row(episodes[A_HIGH], "washout")
    a_power = float(a.log10_alpha_power_8_12_hz)
    b_power = float(b.log10_alpha_power_8_12_hz)
    initial_distance = abs(a_power - b_power)
    a_dipole = _dipole_fourier(episodes[A_HIGH], epoch="stimulation", cfg=cfg)
    result = []
    for amplitude in _dose_values(cfg):
        condition = _dose_id(amplitude)
        active_episode = episodes[condition]
        active = _epoch_row(active_episode)
        active_baseline = _epoch_row(active_episode, "baseline")
        active_washout = _epoch_row(active_episode, "washout")
        active_power = float(active.log10_alpha_power_8_12_hz)
        suppression = a_power - active_power
        distance_improvement = initial_distance - abs(active_power - b_power)
        washout_residual = float(
            (a_washout.log10_alpha_power_8_12_hz
             - a_baseline.log10_alpha_power_8_12_hz)
            - (active_washout.log10_alpha_power_8_12_hz
               - active_baseline.log10_alpha_power_8_12_hz)
        )
        recovered, recovery_tolerance = _field_removal_status(
            effect_log10=suppression,
            residual_log10=washout_residual,
            cfg=cfg,
        )
        decomposition = _complex_response_decomposition(
            sham_cosine=float(a.eeg_10hz_cosine_v),
            sham_sine=float(a.eeg_10hz_sine_v),
            active_cosine=float(active.eeg_10hz_cosine_v),
            active_sine=float(active.eeg_10hz_sine_v),
        )
        active_dipole = _dipole_fourier(
            active_episode, epoch="stimulation", cfg=cfg
        )
        delta_dipole_cosine = (
            active_dipole["cosine_nA_um"] - a_dipole["cosine_nA_um"]
        )
        delta_dipole_sine = (
            active_dipole["sine_nA_um"] - a_dipole["sine_nA_um"]
        )
        result.append({
            "seed": int(seed),
            "dose_v_per_m": float(amplitude),
            "condition_id": condition,
            "A_minus_B_log10_alpha_power": a_power - b_power,
            PRIMARY_METRIC: suppression,
            "target_distance_improvement_log10": distance_improvement,
            "fractional_target_distance_improvement": (
                distance_improvement / initial_distance
                if initial_distance > np.finfo(float).eps else 0.0
            ),
            "coherent_10hz_suppression_v": float(
                a.eeg_10hz_resultant_v - active.eeg_10hz_resultant_v
            ),
            "alpha_peak_prominence_reduction_db": float(
                a.alpha_peak_prominence_db - active.alpha_peak_prominence_db
            ),
            "E_ppc_reduction": float(a.E_ppc - active.E_ppc),
            "I_ppc_reduction": float(a.I_ppc - active.I_ppc),
            "E_rate_change_hz": float(active.E_firing_rate_hz - a.E_firing_rate_hz),
            "I_rate_change_hz": float(active.I_firing_rate_hz - a.I_firing_rate_hz),
            "reference_rate_matched": bool(_relative_rate_safe(a, b, cfg)),
            "rate_safe": bool(_relative_rate_safe(active, a, cfg)),
            "field_removal_residual_log10": washout_residual,
            "field_removal_tolerance_log10": recovery_tolerance,
            "field_removal_recovered": recovered,
            "desired_effect_and_reversible": bool(suppression > 0.0 and recovered),
            "baseline_relative_rms_error": float(_relative_rms_error(
                _epoch_raw(episodes[A_HIGH], "baseline"),
                _epoch_raw(active_episode, "baseline"),
            )),
            "sham_dipole_10hz_vector_nA_um": float(
                a_dipole["vector_resultant_nA_um"]
            ),
            "active_dipole_10hz_vector_nA_um": float(
                active_dipole["vector_resultant_nA_um"]
            ),
            "active_dipole_z_10hz_resultant_nA_um": float(
                active_dipole["axis_resultant_nA_um"][2]
            ),
            "dipole_z_10hz_change_nA_um": float(
                active_dipole["axis_resultant_nA_um"][2]
                - a_dipole["axis_resultant_nA_um"][2]
            ),
            "induced_dipole_10hz_vector_nA_um": float(np.sqrt(np.sum(
                delta_dipole_cosine**2 + delta_dipole_sine**2
            ))),
            **decomposition,
            **{key: value for key, value in phase_quality.items() if key != "seed"},
        })
    return result


def _metric_summaries(
    metrics: pd.DataFrame, *, cfg: DictConfig, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for dose, group in metrics.groupby("dose_v_per_m", sort=True):
        for metric in MECHANISM_METRICS:
            rows.append({
                "dose_v_per_m": float(dose),
                "metric": metric,
                **_summary(group[metric].to_numpy(float), cfg=cfg, rng=rng),
            })
    return pd.DataFrame(rows)


def _guardrail_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dose, group in metrics.groupby("dose_v_per_m", sort=True):
        rows.append({
            "dose_v_per_m": float(dose),
            "n_seeds": int(len(group)),
            "rate_safe_fraction": float(group.rate_safe.mean()),
            "field_removal_recovery_fraction": float(
                group.field_removal_recovered.mean()
            ),
            "phase_quality_fraction": float(group.phase_quality_pass.mean()),
            "maximum_baseline_relative_rms_error": float(
                group.baseline_relative_rms_error.max()
            ),
            "maximum_coherent_decomposition_error_v2": float(
                group.coherent_decomposition_error_v2.abs().max()
            ),
        })
    return pd.DataFrame(rows)


def _fit_zero_intercept_quadratic(
    doses: np.ndarray, effects: np.ndarray
) -> dict[str, float | None]:
    """Fit y=b1*dose+b2*dose^2 to paired mean effects, including sham at 0."""
    doses = np.asarray(doses, dtype=float)
    effects = np.asarray(effects, dtype=float)
    if doses.ndim != 1 or effects.shape != doses.shape or doses.size < 2:
        raise ValueError("At least two aligned dose/effect values are required.")
    design = np.column_stack((doses, doses**2))
    coefficients, _, _, _ = np.linalg.lstsq(design, effects, rcond=None)
    fitted = design @ coefficients
    residual = effects - fitted
    total = float(np.sum((effects - np.mean(effects)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 1.0
    linear, quadratic = map(float, coefficients)
    turning = None
    if abs(quadratic) > np.finfo(float).eps:
        candidate = -linear / (2.0 * quadratic)
        if 0.0 <= candidate <= float(np.max(doses)):
            turning = float(candidate)
    return {
        "linear_coefficient": linear,
        "quadratic_coefficient": quadratic,
        "r_squared": float(r_squared),
        "turning_dose_v_per_m": turning,
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def _response_models(metrics: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for metric in (
        PRIMARY_METRIC,
        "coherent_10hz_suppression_v",
        "E_ppc_reduction",
        "coherent_induced_component_fraction",
    ):
        means = metrics.groupby("dose_v_per_m", sort=True)[metric].mean()
        doses = np.concatenate(([0.0], means.index.to_numpy(float)))
        effects = np.concatenate(([0.0], means.to_numpy(float)))
        result[metric] = _fit_zero_intercept_quadratic(doses, effects)
    return result


def _candidate_protocol(
    *, summaries: pd.DataFrame, guardrails: pd.DataFrame, cfg: DictConfig
) -> dict[str, Any]:
    indexed = summaries.set_index(["dose_v_per_m", "metric"])
    candidates = []
    for dose in sorted(summaries.dose_v_per_m.unique()):
        alpha = indexed.loc[(dose, PRIMARY_METRIC)]
        movement = indexed.loc[(dose, "target_distance_improvement_log10")]
        coherent = indexed.loc[(dose, "coherent_10hz_suppression_v")]
        row = guardrails[np.isclose(guardrails.dose_v_per_m, dose)].iloc[0]
        directional = bool(
            float(alpha["mean"])
            > float(cfg.analysis.criteria.minimum_candidate_mean_suppression_log10)
            and int(alpha["positive_seed_count"])
            >= int(cfg.analysis.criteria.minimum_candidate_positive_seeds)
            and float(movement["mean"]) > 0.0
            and float(coherent["mean"]) > 0.0
            and float(row.rate_safe_fraction)
            >= float(cfg.analysis.criteria.minimum_rate_safe_fraction)
            and float(row.field_removal_recovery_fraction)
            >= float(cfg.analysis.criteria.minimum_field_removal_fraction)
        )
        candidates.append({
            "dose_v_per_m": float(dose),
            "directional_candidate": directional,
            "mean_alpha_suppression_log10": float(alpha["mean"]),
            "positive_seed_count": int(alpha["positive_seed_count"]),
            "mean_target_distance_improvement_log10": float(movement["mean"]),
            "mean_coherent_10hz_suppression_v": float(coherent["mean"]),
        })
    eligible = [row for row in candidates if row["directional_candidate"]]
    pool = eligible if eligible else candidates
    selected = sorted(
        pool,
        key=lambda row: (
            -row["mean_alpha_suppression_log10"], row["dose_v_per_m"]
        ),
    )[0]
    return {
        "selection_stage": "exploratory_directional_dose_audit",
        "efficacy_ranking_uses_only_ideal_eeg_and_action_metadata": True,
        "hidden_rates_used_only_as_safety_guardrail": True,
        "relative_phase_offset_rad": float(np.pi),
        "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
        "montage": str(cfg.analysis.tacs.axial_montage),
        "selected_dose_v_per_m": float(selected["dose_v_per_m"]),
        "directional_candidate_found": bool(eligible),
        "selected_summary": selected,
        "all_doses": candidates,
        "requires_disjoint_confirmation": True,
        "not_an_rl_action_yet": True,
    }


def _checks(
    *,
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    guardrails: pd.DataFrame,
    candidate: dict[str, Any],
    frozen: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    indexed = summaries.set_index(["dose_v_per_m", "metric"])
    dose = float(candidate["selected_dose_v_per_m"])
    selected = {metric: indexed.loc[(dose, metric)] for metric in MECHANISM_METRICS}
    selected_rows = metrics[np.isclose(metrics.dose_v_per_m, dose)]
    unique_seed = metrics.drop_duplicates("seed")
    criteria = cfg.analysis.criteria
    checks = {
        "minimum_exploration_seeds": int(metrics.seed.nunique())
        >= int(criteria.minimum_exploration_seeds),
        "frozen_state_and_phase_loaded": bool(
            np.isclose(float(frozen["target"]["selected_modulation_depth"]), 0.04)
            and np.isclose(
                float(frozen["protocol"]["selected_eeg_relative_phase_offset_rad"]),
                np.pi,
            )
        ),
        "elevated_alpha_state_replicates_directionally": bool(
            float(unique_seed.A_minus_B_log10_alpha_power.mean()) > 0.0
            and float((unique_seed.A_minus_B_log10_alpha_power > 0.0).mean())
            >= float(criteria.minimum_reference_positive_fraction)
        ),
        "baseline_phase_estimate_stable": float(unique_seed.phase_quality_pass.mean())
        >= float(criteria.minimum_phase_quality_fraction),
        "directional_alpha_suppression_candidate_found": bool(
            candidate["directional_candidate_found"]
        ),
        "candidate_moves_eeg_toward_B_directionally": bool(
            float(selected["target_distance_improvement_log10"]["mean"]) > 0.0
        ),
        "candidate_coherent_10hz_component_reduced": bool(
            float(selected["coherent_10hz_suppression_v"]["mean"]) > 0.0
        ),
        "candidate_alpha_peak_prominence_reduced": bool(
            float(selected["alpha_peak_prominence_reduction_db"]["mean"]) > 0.0
        ),
        "candidate_hidden_E_spike_synchrony_reduced": bool(
            float(selected["E_ppc_reduction"]["mean"]) > 0.0
        ),
        "all_doses_rate_safe": bool(
            guardrails.rate_safe_fraction.min()
            >= float(criteria.minimum_rate_safe_fraction)
        ),
        "field_removal_recovered_independent_of_effect_sign": bool(
            guardrails.field_removal_recovery_fraction.min()
            >= float(criteria.minimum_field_removal_fraction)
        ),
        "baseline_causality": bool(
            guardrails.maximum_baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "coherent_energy_decomposition_exact": bool(
            np.allclose(
                metrics.coherent_net_change_v2,
                metrics.coherent_interference_cross_term_v2
                + metrics.coherent_induced_component_v2,
                rtol=1.0e-10,
                atol=1.0e-30,
            )
        ),
    }
    gate_names = (
        "minimum_exploration_seeds",
        "frozen_state_and_phase_loaded",
        "elevated_alpha_state_replicates_directionally",
        "baseline_phase_estimate_stable",
        "directional_alpha_suppression_candidate_found",
        "candidate_moves_eeg_toward_B_directionally",
        "candidate_coherent_10hz_component_reduced",
        "all_doses_rate_safe",
        "field_removal_recovered_independent_of_effect_sign",
        "baseline_causality",
        "coherent_energy_decomposition_exact",
    )
    conclusions = {
        "exploratory_dose_mechanism_gate_passed": all(
            checks[name] for name in gate_names
        ),
        "lower_than_frozen_dose_candidate_found": bool(
            candidate["directional_candidate_found"]
            and dose < float(frozen["protocol"]["amplitude_v_per_m"])
        ),
        "selected_exploratory_dose_v_per_m": dose,
        "selected_seed_fraction_with_alpha_suppression": float(
            (selected_rows[PRIMARY_METRIC] > 0.0).mean()
        ),
        "selected_mean_alpha_suppression_log10": float(
            selected_rows[PRIMARY_METRIC].mean()
        ),
        "selected_mean_E_ppc_reduction": float(selected_rows.E_ppc_reduction.mean()),
        "confirmation_status": "NOT TESTED; exploratory candidate only",
    }
    return checks, conclusions


def _stimulation_psd(
    episode: dict[str, Any], *, cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray]:
    raw = _epoch_raw(episode, "stimulation")
    trim_ms = float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    trim_samples = int(round(trim_ms / float(cfg.env.network.dt)))
    raw = raw[trim_samples:-trim_samples]
    outputs = episode["simulation"]["outputs_by_epoch"]["stimulation"]
    start_ms = float(outputs[0]["t_start_ms"]) + trim_ms
    _, _, frequencies, psd = _feature_from_raw(
        raw,
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        start_ms=start_ms,
        cfg=cfg,
    )
    return frequencies, psd


def _plot_results(
    *,
    root: Path,
    psds: dict[str, list[np.ndarray]],
    frequencies: np.ndarray,
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    doses = sorted(metrics.dose_v_per_m.unique())
    colors = plt.cm.viridis(np.linspace(0.18, 0.9, len(doses)))
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for condition, label, color in (
        (B_LOW, "B low-alpha", "#2CA02C"),
        (A_HIGH, "A elevated-alpha sham", "#9467BD"),
    ):
        mean_psd = np.mean(np.asarray(psds[condition]), axis=0)
        axis.plot(
            frequencies,
            10.0 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny)),
            label=label,
            color=color,
            linewidth=2.0,
        )
    for dose, color in zip(doses, colors):
        mean_psd = np.mean(np.asarray(psds[_dose_id(dose)]), axis=0)
        axis.plot(
            frequencies,
            10.0 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny)),
            label=f"A + {dose:g} V/m",
            color=color,
        )
    axis.axvspan(8.0, 12.0, color="gold", alpha=0.14)
    axis.set_xlim(2.0, 25.0)
    axis.set(
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB V²/Hz)",
        title="Ideal neural EEG spectra across frozen-phase tACS doses",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_dose_psd.png", dpi=250)
    plt.close(figure)

    indexed = summaries.set_index(["dose_v_per_m", "metric"])
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for seed, group in metrics.groupby("seed"):
        ordered = group.sort_values("dose_v_per_m")
        x = np.concatenate(([0.0], ordered.dose_v_per_m.to_numpy(float)))
        y_alpha = np.concatenate(([0.0], ordered[PRIMARY_METRIC].to_numpy(float)))
        y_coherent = np.concatenate((
            [0.0], ordered.coherent_10hz_suppression_v.to_numpy(float) * 1.0e9
        ))
        axes[0].plot(x, y_alpha, color="0.75", marker="o", linewidth=1.0)
        axes[1].plot(x, y_coherent, color="0.75", marker="o", linewidth=1.0)
    alpha_means = np.asarray([
        indexed.loc[(dose, PRIMARY_METRIC), "mean"] for dose in doses
    ])
    coherent_means = np.asarray([
        indexed.loc[(dose, "coherent_10hz_suppression_v"), "mean"] * 1.0e9
        for dose in doses
    ])
    axes[0].plot([0.0, *doses], [0.0, *alpha_means], color="black", marker="o", linewidth=2.3)
    axes[1].plot([0.0, *doses], [0.0, *coherent_means], color="black", marker="o", linewidth=2.3)
    for axis in axes:
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_xlabel("Axial field amplitude (V/m)")
    axes[0].set_ylabel("A sham − active log10 alpha power")
    axes[0].set_title("Primary paired dose response")
    axes[1].set_ylabel("Exact 10-Hz resultant reduction (nV)")
    axes[1].set_title("Coherent 10-Hz response")
    figure.tight_layout()
    figure.savefig(root / "figure_02_paired_dose_response.png", dpi=250)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    mean = metrics.groupby("dose_v_per_m", sort=True).mean(numeric_only=True)
    axes[0].plot([0.0, *doses], [0.0, *mean.E_ppc_reduction], marker="o", color="#D95F02")
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].set(
        xlabel="Field amplitude (V/m)", ylabel="E-population PPC reduction",
        title="Hidden spike synchrony",
    )
    axes[1].plot(doses, mean.coherent_interference_cross_term_fraction, marker="o", label="cross/interference")
    axes[1].plot(doses, mean.coherent_induced_component_fraction, marker="o", label="induced component")
    axes[1].plot(doses, mean.coherent_net_change_fraction, marker="o", label="net coherent change")
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set(
        xlabel="Field amplitude (V/m)", ylabel="Fraction of sham 10-Hz energy",
        title="Neural EEG vector decomposition",
    )
    axes[1].legend(fontsize=8)
    axes[2].plot(
        [0.0, *doses],
        [0.0, *mean.induced_dipole_10hz_vector_nA_um],
        marker="o",
        color="#1B9E77",
    )
    axes[2].set(
        xlabel="Field amplitude (V/m)", ylabel="Induced dipole resultant (nA·µm)",
        title="Total current-dipole response",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_03_mechanism_decomposition.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen = _load_frozen_source(cfg)
    _validate_design(cfg, frozen)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "alpha_suppression_dose_audit"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### Exploratory alpha-suppression dose/mechanism audit")
        print(json.dumps(_plain(frozen), indent=2))
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()

    started = time.perf_counter()
    rng = np.random.default_rng(int(cfg.experiment.seed) + 2_000_003)
    depth = float(frozen["target"]["selected_modulation_depth"])
    relative_offset = float(
        frozen["protocol"]["selected_eeg_relative_phase_offset_rad"]
    )
    montage = str(frozen["protocol"]["montage"])
    epoch_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    psds: dict[str, list[np.ndarray]] = {
        B_LOW: [], A_HIGH: [], **{_dose_id(dose): [] for dose in _dose_values(cfg)}
    }
    frequencies = None

    for seed in _seeds(cfg):
        if rank == 0:
            print(f"dose-audit seed={seed}")
            episodes: dict[str, dict[str, Any]] = {}
        b_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=0.0)
        a_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=depth)
        specifications = [
            (B_LOW, b_cfg, _sham(cfg, B_LOW), False),
            (A_HIGH, a_cfg, _sham(cfg, A_HIGH), False),
        ]
        specifications.extend([
            (
                _dose_id(dose),
                a_cfg,
                _action(
                    cfg,
                    identifier=_dose_id(dose),
                    role="exploratory_frozen_phase_dose",
                    amplitude=dose,
                    montage=montage,
                    relative_offset=relative_offset,
                ),
                True,
            )
            for dose in _dose_values(cfg)
        ])
        for action_index, (condition, condition_cfg, action, stimulate) in enumerate(
            specifications
        ):
            episode = _run_condition(
                condition_id=condition,
                condition_cfg=condition_cfg,
                action=action,
                stimulate=stimulate,
                seed=seed,
                action_index=action_index,
                output_dir=root / "episodes" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                episodes[condition] = episode
                epoch_rows.extend(episode["epoch_rows"])
                if bool(cfg.experiment.plot):
                    frequencies, psd = _stimulation_psd(episode, cfg=cfg)
                    psds[condition].append(psd)
        if rank == 0:
            representative = _dose_id(max(_dose_values(cfg)))
            phase_quality = _phase_quality_row(
                seed=seed,
                a_episode=episodes[A_HIGH],
                selected_episode=episodes[representative],
                selected_offset=relative_offset,
                cfg=cfg,
            )
            phase_rows.append(phase_quality)
            metric_rows.extend(_seed_dose_metrics(
                seed=seed,
                episodes=episodes,
                phase_quality=phase_quality,
                cfg=cfg,
            ))
            del episodes

    if rank == 0:
        epoch_frame = pd.DataFrame(epoch_rows)
        metrics = pd.DataFrame(metric_rows)
        phase_frame = pd.DataFrame(phase_rows)
        summaries = _metric_summaries(metrics, cfg=cfg, rng=rng)
        guardrails = _guardrail_summary(metrics)
        response_models = _response_models(metrics)
        candidate = _candidate_protocol(
            summaries=summaries, guardrails=guardrails, cfg=cfg
        )
        checks, conclusions = _checks(
            metrics=metrics,
            summaries=summaries,
            guardrails=guardrails,
            candidate=candidate,
            frozen=frozen,
            cfg=cfg,
        )

        epoch_frame.to_csv(root / "dose_epoch_eeg_and_hidden_metrics.csv", index=False)
        metrics.to_csv(root / "dose_seed_metrics.csv", index=False)
        phase_frame.to_csv(root / "baseline_phase_quality.csv", index=False)
        summaries.to_csv(root / "dose_metric_summary.csv", index=False)
        guardrails.to_csv(root / "dose_guardrails.csv", index=False)
        (root / "dose_response_models.json").write_text(
            json.dumps(_plain(response_models), indent=2)
        )
        (root / "exploratory_candidate_protocol.json").write_text(
            json.dumps(_plain(candidate), indent=2)
        )
        provenance = {
            **frozen,
            "exploration_seeds": _seeds(cfg),
            "dose_grid_v_per_m": _dose_values(cfg),
            "phase_or_frequency_selection_performed": False,
            "dose_selection_performed": True,
            "requires_disjoint_confirmation": True,
        }
        (root / "frozen_protocol_provenance.json").write_text(
            json.dumps(_plain(provenance), indent=2)
        )
        result = {
            "scope": "exploratory ideal neural-only simulated EEG",
            "checks": checks,
            "conclusions": conclusions,
            "primary_metric": PRIMARY_METRIC,
            "candidate": candidate,
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": (
                "A passing directional gate identifies only a low-cost dose candidate. "
                "It does not confirm EEG control, validate a depression mechanism, or "
                "authorize an RL action without disjoint held-out confirmation."
            ),
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        if bool(cfg.experiment.plot):
            _plot_results(
                root=root,
                psds=psds,
                frequencies=np.asarray(frequencies),
                metrics=metrics,
                summaries=summaries,
            )
        print("\n### Exploratory dose/mechanism checks")
        for name, passed in checks.items():
            print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
        print("\n### Exploratory candidate")
        print(json.dumps(_plain(candidate), indent=2))
        print(
            "\nDose/mechanism directional gate:",
            "PASSED" if conclusions["exploratory_dose_mechanism_gate_passed"] else "NOT PASSED",
        )
        print("Confirmation status: NOT TESTED")
        print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
