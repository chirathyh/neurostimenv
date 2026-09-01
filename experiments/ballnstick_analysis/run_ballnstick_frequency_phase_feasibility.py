"""F0 feasibility map for EEG-conditioned frequency/phase tACS.

This deliberately small, full-information experiment asks whether a useful
context--action interaction exists before fitting a contextual bandit.  The
toy elevated-alpha state is generated at either 9 or 11 Hz by weak sinusoidal
modulation of otherwise mean-rate-matched Poisson afferents.  The low-alpha
reference has homogeneous Poisson afferents.  Cell, recurrent-network, and
mean-afferent parameters are identical in both states.

Each eligible context is replayed under sham and four constant tACS actions:
9 or 11 Hz crossed with 0 or pi phase relative to the preceding ideal EEG.
Amplitude is fixed at 0.4 V/m.  Screening and the candidate EEG rule inspect
only prestimulation EEG.  Hidden generator frequency and spike PPC are used
only after simulation as system-identification and mechanism audits.

This is exploratory system identification, not a trained bandit, a held-out
confirmation, or a model of depression or treatment.
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
from omegaconf import DictConfig, OmegaConf, open_dict


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _action,
    _condition_for_seed,
    _epoch_raw,
    _epoch_row,
    _estimate_relative_field_phase,
    _plain,
    _run_condition,
    _sham,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (  # noqa: E402
    _field_removal_status,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _band_power,
    _fourier_coefficients,
    _process_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _relative_rate_safe,
)


REFERENCE = "B_low_alpha_reference"
SHAM = "sham"
ACTIVE_PHASE_LABELS = {0.0: "inphase", float(np.pi): "antiphase"}


def _copy_cfg(cfg: DictConfig) -> DictConfig:
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(result, False)
    return result


def _frequency_token(frequency_hz: float) -> str:
    return f"{float(frequency_hz):g}".replace(".", "p")


def _action_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    result = [{
        "id": SHAM,
        "role": "unstimulated_counterfactual",
        "frequency_hz": float(cfg.analysis.tacs.frequencies_hz[0]),
        "relative_phase_offset_rad": 0.0,
        "ac_amplitude_v_per_m": 0.0,
    }]
    amplitude = float(cfg.analysis.tacs.amplitude_v_per_m)
    for frequency_hz in cfg.analysis.tacs.frequencies_hz:
        for raw_offset in cfg.analysis.tacs.relative_phase_offsets_rad:
            offset = _wrap_phase(float(raw_offset))
            label = (
                "inphase" if np.isclose(offset, 0.0)
                else "antiphase" if np.isclose(offset, np.pi)
                else f"phase_{np.degrees(offset):g}deg"
            )
            result.append({
                "id": f"f{_frequency_token(frequency_hz)}_{label}",
                "role": "frequency_phase_action",
                "frequency_hz": float(frequency_hz),
                "relative_phase_offset_rad": offset,
                "ac_amplitude_v_per_m": amplitude,
            })
    return result


def _reference_seeds(cfg: DictConfig) -> list[int]:
    block = cfg.analysis.reference_calibration
    first = int(cfg.experiment.seed) + int(block.seed_offset)
    return [first + index for index in range(int(block.n_seeds))]


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    structures = [
        base + int(block.structure_seed_offset) + index
        for index in range(int(block.n_structure_seeds))
    ]
    histories = [
        base + int(block.history_seed_offset) + index
        for index in range(int(block.n_history_seeds))
    ]
    result = []
    order = 0
    for structure_index, structure_seed in enumerate(structures):
        for frequency_hz in cfg.analysis.states.frequencies_hz:
            for history_index, history_seed in enumerate(histories):
                result.append({
                    "context_order": order,
                    "context_id": (
                        f"s{structure_index:02d}_f{_frequency_token(frequency_hz)}"
                        f"_h{history_index:02d}"
                    ),
                    "structure_index": structure_index,
                    "structure_seed": structure_seed,
                    "history_index": history_index,
                    "drive_seed": history_seed,
                    "hidden_frequency_hz": float(frequency_hz),
                    "phase_seed": base + int(block.phase_seed_offset) + order,
                    "trial_seed": base + int(block.trial_seed_offset) + order,
                })
                order += 1
    return result


def _future_seed(cfg: DictConfig, context_order: int, future_index: int) -> int:
    block = cfg.analysis.crossed_design
    return (
        int(cfg.experiment.seed)
        + int(block.future_seed_offset)
        + 100 * int(context_order)
        + int(future_index)
    )


def _with_hidden_frequency(
    cfg: DictConfig, *, frequency_hz: float, phase_seed: int, modulation_depth: float
) -> DictConfig:
    result = _copy_cfg(cfg)
    with open_dict(result):
        result.analysis.reference.frequency_hz = float(frequency_hz)
        # Keep the sham analysis at the hidden state frequency; active action
        # copies overwrite only the analysis/actuator frequency below.
        result.analysis.tacs.frequency_hz = float(frequency_hz)
        result.analysis.protocol.frequency_hz = float(frequency_hz)
    return _condition_for_seed(
        result, seed=int(phase_seed), modulation_depth=float(modulation_depth)
    )


def _with_action_frequency(cfg: DictConfig, frequency_hz: float) -> DictConfig:
    result = _copy_cfg(cfg)
    with open_dict(result):
        result.analysis.tacs.frequency_hz = float(frequency_hz)
        result.analysis.protocol.frequency_hz = float(frequency_hz)
    return result


def _materialize_action(cfg: DictConfig, spec: dict[str, Any]) -> dict[str, Any]:
    action_cfg = _with_action_frequency(cfg, float(spec["frequency_hz"]))
    if str(spec["id"]) == SHAM:
        # The persistent relative-phase episode estimates phase even for the
        # zero-amplitude counterfactual, so retain an explicit (irrelevant)
        # relative offset in this action representation.
        return _action(
            action_cfg,
            identifier=SHAM,
            role="unstimulated_counterfactual",
            amplitude=0.0,
            montage=str(cfg.analysis.tacs.axial_montage),
            relative_offset=0.0,
        )
    return _action(
        action_cfg,
        identifier=str(spec["id"]),
        role=str(spec["role"]),
        amplitude=float(spec["ac_amplitude_v_per_m"]),
        montage=str(cfg.analysis.tacs.axial_montage),
        relative_offset=float(spec["relative_phase_offset_rad"]),
    )


def _validate_design(cfg: DictConfig) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("F0 frequency/phase mapping requires the online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("A, B, and all actions must keep inhibition_scale=1.")
    state_frequencies = [float(x) for x in cfg.analysis.states.frequencies_hz]
    action_frequencies = [float(x) for x in cfg.analysis.tacs.frequencies_hz]
    if state_frequencies != [9.0, 11.0] or action_frequencies != state_frequencies:
        raise ValueError("This minimal experiment freezes both state/action grids to 9 and 11 Hz.")
    offsets = sorted(_wrap_phase(float(x)) for x in cfg.analysis.tacs.relative_phase_offsets_rad)
    if len(offsets) != 2 or not np.isclose(offsets[0], 0.0) or not np.isclose(offsets[1], np.pi):
        raise ValueError("The phase grid must be exactly {0, pi} EEG-relative radians.")
    amplitude = float(cfg.analysis.tacs.amplitude_v_per_m)
    if not 0.0 < amplitude <= float(cfg.analysis.maximum_field_v_per_m):
        raise ValueError("The fixed tACS amplitude must be positive and within the field limit.")
    if not 0.0 < float(cfg.analysis.states.modulation_depth) <= float(
        cfg.analysis.reference.thinning_envelope_modulation_depth
    ):
        raise ValueError("The hidden afferent modulation must fit its thinning envelope.")

    timeline = cfg.analysis.timeline
    if int(timeline.baseline_steps) < (2 if bool(cfg.analysis.smoke_test) else 6):
        raise ValueError("Full F0 requires six seconds of stimulation-free EEG.")
    stimulation_ms = int(timeline.stimulation_steps) * float(cfg.env.simulation.obs_win_len)
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    if trim_ms < float(timeline.block_ramp_ms) or 2.0 * trim_ms >= stimulation_ms:
        raise ValueError("Stimulation trimming must remove both ramps and leave EEG.")
    pre_ms = (int(timeline.burn_in_steps) + int(timeline.baseline_steps)) * float(
        cfg.env.simulation.obs_win_len
    )
    for frequency_hz in state_frequencies:
        cycles = frequency_hz * pre_ms / 1000.0
        if not np.isclose(cycles, round(cycles), atol=1.0e-10):
            raise ValueError("The intervention boundary must contain integer 9- and 11-Hz cycles.")

    references = set(_reference_seeds(cfg))
    contexts = _context_specs(cfg)
    structures = {int(x["structure_seed"]) for x in contexts}
    histories = {int(x["drive_seed"]) for x in contexts}
    phases = {int(x["phase_seed"]) for x in contexts}
    trials = {int(x["trial_seed"]) for x in contexts}
    futures = {
        _future_seed(cfg, int(x["context_order"]), future_index)
        for x in contexts
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations))
    }
    namespaces = [references, structures, histories, phases, trials, futures]
    if any(not values for values in namespaces):
        raise ValueError("Every independent seed namespace must be nonempty.")
    if any(namespaces[i].intersection(namespaces[j]) for i in range(len(namespaces)) for j in range(i + 1, len(namespaces))):
        raise ValueError("Reference, structure, history, phase, trial, and future seeds must be disjoint.")
    if max(structures) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Structure seed * 10,000 exceeds the uint32 seed range.")
    if not bool(cfg.analysis.smoke_test):
        if int(cfg.analysis.reference_calibration.n_seeds) < 6:
            raise ValueError("Full F0 requires at least six independent B calibration seeds.")
        if int(cfg.analysis.crossed_design.n_structure_seeds) < 3:
            raise ValueError("Full F0 requires at least three independent structures.")
        if int(cfg.analysis.crossed_design.n_future_continuations) < 2:
            raise ValueError("Full F0 requires at least two independent futures per action.")


def _epoch_signal(
    episode: dict[str, Any], epoch: str, cfg: DictConfig
) -> tuple[np.ndarray, float]:
    raw = _epoch_raw(episode, epoch)
    outputs = episode["simulation"]["outputs_by_epoch"][epoch]
    start_ms = float(outputs[0]["t_start_ms"])
    if epoch == "stimulation":
        trim_ms = float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
        trim_samples = int(round(trim_ms / float(cfg.env.network.dt)))
        if trim_samples:
            raw = raw[trim_samples:-trim_samples]
            start_ms += trim_ms
    return raw, start_ms


def _spectral_feature(
    raw: np.ndarray, *, start_ms: float, simulator_fs_hz: float, cfg: DictConfig
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    processed, fs_hz, frequencies, psd, generic = _process_eeg(
        raw, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    eps = np.finfo(float).tiny
    result: dict[str, float] = {
        "rms_v": float(generic["rms_v"]),
        "analysis_duration_s": float(processed.size / fs_hz),
    }
    half_width = float(cfg.analysis.spectral_target.band_half_width_hz)
    for frequency_hz in cfg.analysis.spectral_target.candidate_frequencies_hz:
        token = _frequency_token(float(frequency_hz))
        power = _band_power(
            frequencies,
            psd,
            center_hz=float(frequency_hz),
            half_width_hz=half_width,
        )
        cosine, sine = _fourier_coefficients(
            processed,
            fs_hz=fs_hz,
            start_ms=start_ms,
            frequency_hz=float(frequency_hz),
        )
        result[f"log10_power_{token}hz"] = float(np.log10(max(power, eps)))
        result[f"eeg_{token}hz_cosine_v"] = float(cosine)
        result[f"eeg_{token}hz_sine_v"] = float(sine)
        result[f"eeg_{token}hz_resultant_v"] = float(np.hypot(cosine, sine))
    alpha_power = _band_power(frequencies, psd, center_hz=10.0, half_width_hz=2.0)
    result["log10_alpha_power_8_12_hz"] = float(np.log10(max(alpha_power, eps)))
    return result, frequencies, psd


def _episode_feature(
    episode: dict[str, Any], epoch: str, cfg: DictConfig
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    raw, start_ms = _epoch_signal(episode, epoch, cfg)
    return _spectral_feature(
        raw,
        start_ms=start_ms,
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        cfg=cfg,
    )


def _fit_reference_target(rows: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    features = [f"log10_power_{_frequency_token(x)}hz" for x in cfg.analysis.states.frequencies_hz]
    means = {name: float(rows[name].mean()) for name in features}
    minimum = float(cfg.analysis.spectral_target.minimum_scale_log10)
    scales = {
        name: max(float(rows[name].std(ddof=1)) if len(rows) > 1 else 0.0, minimum)
        for name in features
    }
    distances = np.sqrt(np.mean(np.column_stack([
        ((rows[name].to_numpy(float) - means[name]) / scales[name]) ** 2
        for name in features
    ]), axis=1))
    return {
        "features": features,
        "means": means,
        "scales": scales,
        "reference_distance_quantile": float(cfg.analysis.spectral_target.reference_quantile),
        "reference_distance_threshold": float(np.quantile(
            distances, float(cfg.analysis.spectral_target.reference_quantile)
        )),
        "calibration_distances": distances.tolist(),
    }


def _target_distance(feature: dict[str, float] | pd.Series, target: dict[str, Any]) -> float:
    z = [
        (float(feature[name]) - float(target["means"][name]))
        / float(target["scales"][name])
        for name in target["features"]
    ]
    return float(np.sqrt(np.mean(np.square(z))))


def _circular_difference(a: float, b: float) -> float:
    return float(np.angle(np.exp(1j * (float(a) - float(b)))))


def _phase_quality(
    episode: dict[str, Any], *, detected_frequency_hz: float, cfg: DictConfig
) -> dict[str, Any]:
    phase_cfg = _with_action_frequency(cfg, detected_frequency_hz)
    outputs = episode["simulation"]["outputs_by_epoch"]["baseline"]
    midpoint = len(outputs) // 2
    block_start_ms = float(episode["simulation"]["block_start_ms"])
    fs_hz = float(episode["simulator_fs_hz"])

    def estimate(part: list[dict[str, Any]]) -> dict[str, float]:
        return _estimate_relative_field_phase(
            part,
            simulator_fs_hz=fs_hz,
            block_start_ms=block_start_ms,
            relative_offset_rad=0.0,
            cfg=phase_cfg,
        )

    full, first, second = estimate(outputs), estimate(outputs[:midpoint]), estimate(outputs[midpoint:])
    split = abs(_circular_difference(
        float(first["baseline_eeg_phase_at_block_rad"]),
        float(second["baseline_eeg_phase_at_block_rad"]),
    ))
    ratio = float(full["baseline_eeg_10hz_resultant_v"]) / max(
        float(full["baseline_eeg_rms_v"]), np.finfo(float).tiny
    )
    passed = bool(
        split <= np.deg2rad(float(cfg.analysis.screening.maximum_phase_split_error_deg))
        and ratio >= float(cfg.analysis.screening.minimum_resultant_to_rms)
    )
    return {
        "estimated_eeg_phase_at_block_rad": float(full["baseline_eeg_phase_at_block_rad"]),
        "split_half_phase_error_rad": split,
        "split_half_phase_error_deg": float(np.degrees(split)),
        "dominant_resultant_to_rms": ratio,
        "phase_actionable": passed,
    }


def _screen_context(
    context: dict[str, Any], episode: dict[str, Any], target: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    feature, _, _ = _episode_feature(episode, "baseline", cfg)
    excesses = {
        float(frequency): float(feature[f"log10_power_{_frequency_token(frequency)}hz"])
        - float(target["means"][f"log10_power_{_frequency_token(frequency)}hz"])
        for frequency in cfg.analysis.states.frequencies_hz
    }
    detected = max(excesses, key=excesses.get)
    phase = _phase_quality(episode, detected_frequency_hz=detected, cfg=cfg)
    distance = _target_distance(feature, target)
    row = _epoch_row(episode, "baseline")
    limits = cfg.analysis.rate_guardrails_hz
    rate_safe = bool(
        float(limits.E_min) <= float(row.E_firing_rate_hz) <= float(limits.E_max)
        and float(limits.I_min) <= float(row.I_firing_rate_hz) <= float(limits.I_max)
    )
    phenotype = bool(
        distance > float(target["reference_distance_threshold"])
        and max(excesses.values()) >= float(cfg.analysis.screening.minimum_peak_excess_log10)
    )
    reasons = []
    if not phenotype:
        reasons.append("alpha_phenotype_absent")
    if not bool(phase["phase_actionable"]):
        reasons.append("phase_unstable_or_weak")
    return {
        **context,
        **feature,
        **phase,
        "detected_frequency_hz": float(detected),
        "frequency_detected_correctly": bool(np.isclose(detected, context["hidden_frequency_hz"])),
        "dominant_excess_log10": float(max(excesses.values())),
        "baseline_distance_to_B": distance,
        "alpha_phenotype_present": phenotype,
        "baseline_rates_safe": rate_safe,
        "baseline_E_firing_rate_hz": float(row.E_firing_rate_hz),
        "baseline_I_firing_rate_hz": float(row.I_firing_rate_hz),
        "eligible": bool(not reasons),
        "exclusion_reasons": ";".join(reasons) if reasons else "none",
        "screen_uses_only_predecision_ideal_eeg": True,
        "screen_does_not_use_hidden_frequency_or_action_outcomes": True,
    }


def _metric_row(
    *,
    context: dict[str, Any],
    screening: dict[str, Any],
    future_index: int,
    spec: dict[str, Any],
    sham_episode: dict[str, Any],
    active_episode: dict[str, Any],
    target: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    sham_feature, _, _ = _episode_feature(sham_episode, "stimulation", cfg)
    active_feature, _, _ = _episode_feature(active_episode, "stimulation", cfg)
    sham_row, active_row = _epoch_row(sham_episode), _epoch_row(active_episode)
    sham_distance = _target_distance(sham_feature, target)
    active_distance = _target_distance(active_feature, target)
    action_frequency = float(spec["frequency_hz"])
    matched = bool(np.isclose(action_frequency, context["hidden_frequency_hz"]))
    offset = float(spec["relative_phase_offset_rad"])
    antiphase = bool(np.isclose(_wrap_phase(offset), np.pi))
    active_baseline, _ = _epoch_signal(active_episode, "baseline", cfg)
    sham_baseline, _ = _epoch_signal(sham_episode, "baseline", cfg)
    baseline_error = _relative_rms_error(sham_baseline, active_baseline)
    sham_washout, _, _ = _episode_feature(sham_episode, "washout", cfg)
    active_washout, _, _ = _episode_feature(active_episode, "washout", cfg)
    effect = float(sham_feature["log10_alpha_power_8_12_hz"] - active_feature["log10_alpha_power_8_12_hz"])
    residual = float(
        (float(sham_washout["log10_alpha_power_8_12_hz"]) - float(screening["log10_alpha_power_8_12_hz"]))
        - (float(active_washout["log10_alpha_power_8_12_hz"]) - float(screening["log10_alpha_power_8_12_hz"]))
    )
    recovered, recovery_tolerance = _field_removal_status(
        effect_log10=effect, residual_log10=residual, cfg=cfg
    )
    if str(spec["id"]) == SHAM:
        phase_error = 0.0
    else:
        expected = _wrap_phase(
            float(active_episode["simulation"]["baseline_eeg_phase_at_block_rad"])
            + np.pi / 2.0 + offset
        )
        realized = float(active_episode["simulation"]["action"]["phase_rad"])
        phase_error = abs(_circular_difference(realized, expected))
    hidden_frequency_token = _frequency_token(float(context["hidden_frequency_hz"]))
    # PPC is a hidden, post hoc mechanism measure. For matched actions the
    # row's action frequency equals the hidden state frequency.
    ppc_reduction = float(sham_row.E_ppc - active_row.E_ppc) if matched else float("nan")
    return {
        **context,
        "future_index": int(future_index),
        "future_drive_seed": _future_seed(cfg, int(context["context_order"]), future_index),
        "action_id": str(spec["id"]),
        "action_frequency_hz": action_frequency,
        "relative_phase_offset_rad": offset,
        "relative_phase_offset_deg": float(np.degrees(_wrap_phase(offset))),
        "amplitude_v_per_m": float(spec["ac_amplitude_v_per_m"]),
        "detected_frequency_hz": float(screening["detected_frequency_hz"]),
        "frequency_matched_hidden_state": matched,
        "antiphase_action": antiphase,
        "sham_distance_to_B": sham_distance,
        "active_distance_to_B": active_distance,
        "distance_improvement_vs_sham": sham_distance - active_distance,
        "sham_log10_alpha_power": float(sham_feature["log10_alpha_power_8_12_hz"]),
        "active_log10_alpha_power": float(active_feature["log10_alpha_power_8_12_hz"]),
        "alpha_suppression_log10": effect,
        "sham_hidden_band_log10_power": float(sham_feature[f"log10_power_{hidden_frequency_token}hz"]),
        "active_hidden_band_log10_power": float(active_feature[f"log10_power_{hidden_frequency_token}hz"]),
        "hidden_band_suppression_log10": float(
            sham_feature[f"log10_power_{hidden_frequency_token}hz"]
            - active_feature[f"log10_power_{hidden_frequency_token}hz"]
        ),
        "hidden_E_ppc_reduction": ppc_reduction,
        "E_firing_rate_change_hz": float(active_row.E_firing_rate_hz - sham_row.E_firing_rate_hz),
        "I_firing_rate_change_hz": float(active_row.I_firing_rate_hz - sham_row.I_firing_rate_hz),
        "rate_safe": bool(_relative_rate_safe(active_row, sham_row, cfg)),
        "baseline_relative_rms_error": baseline_error,
        "action_phase_tracking_error_rad": phase_error,
        "washout_residual_log10": residual,
        "washout_tolerance_log10": recovery_tolerance,
        "field_removal_recovered": recovered,
        "policy_observation_uses_hidden_variables": False,
    }


def _expected_action_map(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "context_id", "context_order", "structure_index", "structure_seed",
        "hidden_frequency_hz", "detected_frequency_hz", "action_id",
        "action_frequency_hz", "relative_phase_offset_rad", "relative_phase_offset_deg",
        "amplitude_v_per_m", "frequency_matched_hidden_state", "antiphase_action",
    ]
    return (
        metrics.groupby(keys, as_index=False)
        .agg(
            expected_distance_to_B=("active_distance_to_B", "mean"),
            expected_improvement_vs_sham=("distance_improvement_vs_sham", "mean"),
            expected_alpha_suppression_log10=("alpha_suppression_log10", "mean"),
            expected_hidden_band_suppression_log10=("hidden_band_suppression_log10", "mean"),
            expected_hidden_E_ppc_reduction=("hidden_E_ppc_reduction", "mean"),
            future_distance_sd=("active_distance_to_B", "std"),
            future_count=("future_index", "nunique"),
            all_rates_safe=("rate_safe", "all"),
            all_field_removal_recovered=("field_removal_recovered", "all"),
            maximum_baseline_relative_rms_error=("baseline_relative_rms_error", "max"),
            maximum_action_phase_tracking_error_rad=("action_phase_tracking_error_rad", "max"),
        )
    )


def _policy_comparison(
    expected: pd.DataFrame, screening: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    active = expected[expected.action_id.ne(SHAM)].copy()
    fixed = (
        active.groupby(["structure_seed", "action_id"], as_index=False)
        .expected_distance_to_B.mean()
        .groupby("action_id", as_index=False)
        .expected_distance_to_B.mean()
        .sort_values(["expected_distance_to_B", "action_id"])
    )
    best_fixed = str(fixed.iloc[0].action_id)
    rows = []
    for context_id, group in active.groupby("context_id"):
        screen = screening[screening.context_id.eq(context_id)].iloc[0]
        detected = float(screen.detected_frequency_hz)
        policy = group[
            np.isclose(group.action_frequency_hz, detected)
            & np.isclose(group.relative_phase_offset_rad, np.pi)
        ].iloc[0]
        fixed_row = group[group.action_id.eq(best_fixed)].iloc[0]
        oracle = group.sort_values(["expected_distance_to_B", "action_id"]).iloc[0]
        rows.append({
            "context_id": context_id,
            "context_order": int(screen.context_order),
            "structure_index": int(screen.structure_index),
            "structure_seed": int(screen.structure_seed),
            "hidden_frequency_hz": float(screen.hidden_frequency_hz),
            "detected_frequency_hz": detected,
            "policy_action_id": str(policy.action_id),
            "best_fixed_action_id": best_fixed,
            "oracle_action_id": str(oracle.action_id),
            "policy_distance_to_B": float(policy.expected_distance_to_B),
            "best_fixed_distance_to_B": float(fixed_row.expected_distance_to_B),
            "oracle_distance_to_B": float(oracle.expected_distance_to_B),
            "policy_advantage_over_best_fixed": float(
                fixed_row.expected_distance_to_B - policy.expected_distance_to_B
            ),
            "policy_oracle_regret": float(
                policy.expected_distance_to_B - oracle.expected_distance_to_B
            ),
            "policy_uses_only_detected_frequency_and_frozen_antiphase": True,
        })
    comparison = pd.DataFrame(rows)
    structures = (
        comparison.groupby(["structure_index", "structure_seed"], as_index=False)
        .agg(
            mean_policy_advantage_over_best_fixed=("policy_advantage_over_best_fixed", "mean"),
            mean_policy_oracle_regret=("policy_oracle_regret", "mean"),
            context_count=("context_id", "nunique"),
        )
    )
    return comparison, structures, best_fixed


def _crossover_summary(expected: pd.DataFrame) -> pd.DataFrame:
    active = expected[expected.action_id.ne(SHAM)]
    rows = []
    for context_id, group in active.groupby("context_id"):
        hidden = float(group.hidden_frequency_hz.iloc[0])
        anti = group[np.isclose(group.relative_phase_offset_rad, np.pi)]
        inphase = group[np.isclose(group.relative_phase_offset_rad, 0.0)]
        matched_anti = anti[np.isclose(anti.action_frequency_hz, hidden)].iloc[0]
        mismatched_anti = anti[~np.isclose(anti.action_frequency_hz, hidden)].iloc[0]
        matched_inphase = inphase[np.isclose(inphase.action_frequency_hz, hidden)].iloc[0]
        rows.append({
            "context_id": context_id,
            "structure_index": int(group.structure_index.iloc[0]),
            "structure_seed": int(group.structure_seed.iloc[0]),
            "hidden_frequency_hz": hidden,
            "matched_antiphase_improvement_vs_sham": float(matched_anti.expected_improvement_vs_sham),
            "frequency_crossover_advantage": float(
                mismatched_anti.expected_distance_to_B - matched_anti.expected_distance_to_B
            ),
            "phase_specific_advantage": float(
                matched_inphase.expected_distance_to_B - matched_anti.expected_distance_to_B
            ),
            "matched_antiphase_hidden_E_ppc_reduction": float(
                matched_anti.expected_hidden_E_ppc_reduction
            ),
        })
    return pd.DataFrame(rows)


def _shuffle_null(
    expected: pd.DataFrame,
    comparison: pd.DataFrame,
    screening: pd.DataFrame,
    best_fixed: str,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, float]:
    active = expected[expected.action_id.ne(SHAM)]
    fixed = active[active.action_id.eq(best_fixed)].set_index("context_id")
    contexts = comparison.context_id.tolist()
    detected = screening.set_index("context_id").loc[contexts, "detected_frequency_hz"].to_numpy(float)
    rng = np.random.default_rng(int(cfg.experiment.seed) + 932_519)
    observed = float(comparison.groupby("structure_seed").policy_advantage_over_best_fixed.mean().mean())
    values = []
    for permutation in range(int(cfg.analysis.context_shuffle.n_permutations)):
        shuffled = rng.permutation(detected)
        context_advantages = []
        context_structures = []
        for context_id, selected_frequency in zip(contexts, shuffled):
            group = active[active.context_id.eq(context_id)]
            selected = group[
                np.isclose(group.action_frequency_hz, selected_frequency)
                & np.isclose(group.relative_phase_offset_rad, np.pi)
            ].iloc[0]
            context_advantages.append(
                float(fixed.loc[context_id].expected_distance_to_B)
                - float(selected.expected_distance_to_B)
            )
            context_structures.append(int(selected.structure_seed))
        frame = pd.DataFrame({"structure_seed": context_structures, "advantage": context_advantages})
        values.append(float(frame.groupby("structure_seed").advantage.mean().mean()))
    null = pd.DataFrame({"permutation": np.arange(len(values)), "shuffled_policy_advantage": values})
    p_value = float((1 + np.count_nonzero(np.asarray(values) >= observed)) / (len(values) + 1))
    return null, p_value


def _checks(
    *,
    calibration: pd.DataFrame,
    screening: pd.DataFrame,
    metrics: pd.DataFrame,
    expected: pd.DataFrame,
    crossover: pd.DataFrame,
    comparison: pd.DataFrame,
    structures: pd.DataFrame,
    shuffle_p: float,
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    state_frequency_means = crossover.groupby("hidden_frequency_hz").frequency_crossover_advantage.mean()
    phase_means = crossover.groupby("hidden_frequency_hz").phase_specific_advantage.mean()
    policy_actions = set(comparison.policy_action_id)
    reference_rates = {
        population: float(calibration[f"{population}_firing_rate_hz"].mean())
        for population in ("E", "I")
    }
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    reference_rate_matched = []
    for row in eligible.itertuples():
        reference_rate_matched.append(all(
            abs(float(getattr(row, f"baseline_{population}_firing_rate_hz")) - reference_rates[population])
            <= tolerance * max(reference_rates[population], np.finfo(float).tiny)
            for population in ("E", "I")
        ))
    contexts_per_frequency = eligible.groupby("hidden_frequency_hz").context_id.nunique()
    checks = {
        "reference_target_calibrated_on_disjoint_B_seeds": len(calibration) >= int(criteria.minimum_reference_seeds),
        "complete_crossed_screening_grid": len(screening) == len(_context_specs(cfg)),
        "screening_uses_only_predecision_eeg": bool(screening.screen_uses_only_predecision_ideal_eeg.all()),
        "screening_does_not_use_hidden_state_or_action_outcomes": bool(screening.screen_does_not_use_hidden_frequency_or_action_outcomes.all()),
        "state_generator_is_distinct_from_tacs_action": True,
        "afferent_mean_rate_matched_across_states_by_construction": True,
        "minimum_eligible_contexts": len(eligible) >= int(criteria.minimum_eligible_contexts),
        "minimum_independent_structures": eligible.structure_seed.nunique() >= int(criteria.minimum_structure_seeds),
        "both_hidden_frequencies_enrolled": eligible.hidden_frequency_hz.nunique() == 2,
        "minimum_contexts_per_hidden_frequency": bool(
            len(contexts_per_frequency) == 2
            and (contexts_per_frequency >= int(criteria.minimum_contexts_per_hidden_frequency)).all()
        ),
        "multiple_independent_futures_per_action": int(expected.future_count.min()) >= int(criteria.minimum_future_continuations),
        "frequency_identified_from_predecision_eeg": float(eligible.frequency_detected_correctly.mean()) >= float(criteria.minimum_frequency_detection_accuracy),
        "all_enrolled_phase_actionable": bool(eligible.phase_actionable.all()),
        "identical_predecision_eeg_across_counterfactual_actions": float(metrics.baseline_relative_rms_error.max()) <= float(criteria.maximum_baseline_relative_rms_error),
        "single_constant_action_per_intervention": True,
        "fixed_amplitude_for_all_active_actions": bool(np.allclose(
            metrics.loc[metrics.action_id.ne(SHAM), "amplitude_v_per_m"],
            float(cfg.analysis.tacs.amplitude_v_per_m),
        )),
        "action_phase_tracks_predecision_eeg": float(metrics.action_phase_tracking_error_rad.max()) <= float(criteria.maximum_phase_tracking_error_rad),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "reference_rate_matched": bool(reference_rate_matched and all(reference_rate_matched)),
        "field_removal_recovered": bool(metrics.field_removal_recovered.all()),
        "matched_antiphase_improves_over_sham": float(crossover.matched_antiphase_improvement_vs_sham.mean()) > 0.0,
        "frequency_specific_crossover": bool(
            len(state_frequency_means) == 2
            and (state_frequency_means >= float(criteria.minimum_frequency_crossover_improvement)).all()
        ),
        "phase_specific_crossover": bool(
            len(phase_means) == 2
            and (phase_means >= float(criteria.minimum_phase_specific_improvement)).all()
        ),
        "eeg_rule_uses_both_frequency_actions": len(policy_actions) == 2,
        "eeg_rule_beats_best_fixed_directionally": float(comparison.policy_advantage_over_best_fixed.mean()) >= float(criteria.minimum_policy_advantage_over_best_fixed),
        "policy_advantage_positive_across_structures": float(
            np.mean(structures.mean_policy_advantage_over_best_fixed > 0.0)
        ) >= float(criteria.minimum_positive_structure_fraction),
        "eeg_frequency_context_beats_shuffled_context": shuffle_p <= float(criteria.maximum_context_shuffle_p_value),
        "hidden_spike_synchrony_reduced": float(crossover.matched_antiphase_hidden_E_ppc_reduction.mean()) > float(criteria.minimum_hidden_E_ppc_reduction),
    }
    key_gate = [
        "minimum_eligible_contexts", "minimum_independent_structures",
        "both_hidden_frequencies_enrolled", "minimum_contexts_per_hidden_frequency",
        "frequency_identified_from_predecision_eeg",
        "all_enrolled_phase_actionable", "identical_predecision_eeg_across_counterfactual_actions",
        "all_actions_rate_safe", "reference_rate_matched", "field_removal_recovered",
        "matched_antiphase_improves_over_sham", "frequency_specific_crossover",
        "phase_specific_crossover", "eeg_rule_uses_both_frequency_actions",
        "eeg_rule_beats_best_fixed_directionally", "policy_advantage_positive_across_structures",
        "eeg_frequency_context_beats_shuffled_context", "hidden_spike_synchrony_reduced",
    ]
    conclusions = {
        "frequency_phase_contextual_feasibility_gate_passed": bool(all(checks[name] for name in key_gate)),
        "eligible_context_count": int(len(eligible)),
        "screening_yield": float(len(eligible) / len(screening)),
        "frequency_detection_accuracy": float(eligible.frequency_detected_correctly.mean()),
        "mean_frequency_crossover_advantage": float(crossover.frequency_crossover_advantage.mean()),
        "mean_phase_specific_advantage": float(crossover.phase_specific_advantage.mean()),
        "mean_policy_advantage_over_best_fixed": float(comparison.policy_advantage_over_best_fixed.mean()),
        "positive_structure_fraction": float(np.mean(structures.mean_policy_advantage_over_best_fixed > 0.0)),
        "context_shuffle_p_value": float(shuffle_p),
        "contextual_bandit_status": "NOT TESTED",
        "claim_scope": "exploratory ideal-neural-EEG system identification",
    }
    return checks, conclusions


def _plot_results(
    *,
    root: Path,
    calibration: pd.DataFrame,
    screening: pd.DataFrame,
    expected: pd.DataFrame,
    crossover: pd.DataFrame,
    structures: pd.DataFrame,
) -> None:
    eligible = screening[screening.eligible]
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.scatter(
        calibration.screen_log10_power_9hz,
        calibration.screen_log10_power_11hz,
        label="B: homogeneous afferents",
        s=65,
        marker="s",
    )
    for frequency_hz, group in eligible.groupby("hidden_frequency_hz"):
        axis.scatter(
            group.log10_power_9hz,
            group.log10_power_11hz,
            label=f"A: hidden {frequency_hz:g} Hz",
            s=65,
        )
    axis.set(
        xlabel="Prestimulation log10 9-Hz band power",
        ylabel="Prestimulation log10 11-Hz band power",
        title="EEG observability and prospective phenotype screen",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_predecision_eeg_states.png", dpi=250)
    plt.close(figure)

    active = expected[expected.action_id.ne(SHAM)]
    heat = active.pivot_table(
        index="hidden_frequency_hz", columns="action_id",
        values="expected_improvement_vs_sham", aggfunc="mean"
    )
    figure, axis = plt.subplots(figsize=(7.4, 3.2))
    image = axis.imshow(heat.to_numpy(), cmap="RdBu_r", aspect="auto")
    axis.set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(heat.index)), [f"A {x:g} Hz" for x in heat.index])
    axis.set(title="Context–action map", xlabel="Constant 0.4-V/m action")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Target-distance improvement vs sham")
    figure.tight_layout()
    figure.savefig(root / "figure_02_frequency_phase_action_map.png", dpi=250)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for frequency_hz, group in crossover.groupby("hidden_frequency_hz"):
        axes[0].scatter(
            np.full(len(group), frequency_hz), group.frequency_crossover_advantage,
            label=f"{frequency_hz:g} Hz", s=55,
        )
        axes[1].scatter(
            np.full(len(group), frequency_hz), group.phase_specific_advantage,
            label=f"{frequency_hz:g} Hz", s=55,
        )
    for axis in axes:
        axis.axhline(0.0, color="0.3", linewidth=0.9)
        axis.set_xticks([9.0, 11.0])
        axis.set_xlabel("Hidden afferent frequency (Hz)")
    axes[0].set(ylabel="Matched minus mismatched advantage", title="Frequency specificity")
    axes[1].set(ylabel="Antiphase minus in-phase advantage", title="Phase specificity")
    figure.tight_layout()
    figure.savefig(root / "figure_03_crossover_tests.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.bar(
        structures.structure_index.astype(str),
        structures.mean_policy_advantage_over_best_fixed,
        color=np.where(structures.mean_policy_advantage_over_best_fixed > 0.0, "#3a923a", "#c44e52"),
    )
    axis.axhline(0.0, color="0.2", linewidth=0.9)
    axis.set(
        xlabel="Independent circuit structure",
        ylabel="EEG rule advantage over best fixed action",
        title="Structure-level contextual opportunity",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_04_policy_advantage.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate_design(cfg)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "frequency_phase_feasibility"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### F0 frequency/phase contextual feasibility")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    calibration_rows: list[dict[str, Any]] = []
    for seed in _reference_seeds(cfg):
        if rank == 0:
            print(f"B calibration seed={seed}")
        b_cfg = _with_hidden_frequency(
            cfg, frequency_hz=9.0, phase_seed=seed, modulation_depth=0.0
        )
        episode = _run_condition(
            condition_id=REFERENCE,
            condition_cfg=b_cfg,
            action=_sham(b_cfg, REFERENCE),
            stimulate=False,
            seed=seed,
            action_index=0,
            output_dir=root / "calibration" / str(seed),
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=seed,
            drive_seed=seed,
            phase_seed=seed,
        )
        if rank == 0:
            screen_feature, _, _ = _episode_feature(episode, "baseline", cfg)
            outcome_feature, _, _ = _episode_feature(episode, "stimulation", cfg)
            row = _epoch_row(episode)
            calibration_rows.append({
                "seed": seed,
                **outcome_feature,
                **{f"screen_{name}": value for name, value in screen_feature.items()},
                "E_firing_rate_hz": float(row.E_firing_rate_hz),
                "I_firing_rate_hz": float(row.I_firing_rate_hz),
            })
    if rank == 0:
        calibration = pd.DataFrame(calibration_rows)
        screening_calibration = pd.DataFrame({
            "seed": calibration.seed,
            **{
                f"log10_power_{_frequency_token(frequency)}hz": calibration[
                    f"screen_log10_power_{_frequency_token(frequency)}hz"
                ]
                for frequency in cfg.analysis.states.frequencies_hz
            },
        })
        target = {
            "screening": _fit_reference_target(screening_calibration, cfg),
            "outcome": _fit_reference_target(calibration, cfg),
            "duration_matching": (
                "screening target uses the B baseline duration; outcome target "
                "uses the ramp-trimmed B intervention duration"
            ),
        }
        calibration["screen_distance_to_B_target"] = [
            _target_distance(row, target["screening"])
            for _, row in screening_calibration.iterrows()
        ]
        calibration["outcome_distance_to_B_target"] = [
            _target_distance(row, target["outcome"])
            for _, row in calibration.iterrows()
        ]
        calibration.to_csv(root / "reference_B_calibration.csv", index=False)
        (root / "frozen_B_spectral_target.json").write_text(
            json.dumps(_plain(target), indent=2)
        )
    else:
        calibration, target = None, None
    target = comm.bcast(target, root=0)

    action_specs = _action_specs(cfg)
    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for context in _context_specs(cfg):
        if rank == 0:
            print(
                f"context={context['context_id']} structure={context['structure_seed']} "
                f"hidden_f={context['hidden_frequency_hz']:g} Hz"
            )
        state_cfg = _with_hidden_frequency(
            cfg,
            frequency_hz=float(context["hidden_frequency_hz"]),
            phase_seed=int(context["phase_seed"]),
            modulation_depth=float(cfg.analysis.states.modulation_depth),
        )
        first_future_seed = _future_seed(cfg, int(context["context_order"]), 0)
        sham_episode = _run_condition(
            condition_id=SHAM,
            condition_cfg=state_cfg,
            action=_materialize_action(state_cfg, action_specs[0]),
            stimulate=True,
            seed=int(context["trial_seed"]),
            action_index=0,
            output_dir=root / "episodes" / str(context["context_id"]) / "future_00" / SHAM,
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(context["structure_seed"]),
            drive_seed=int(context["drive_seed"]),
            future_drive_seed=first_future_seed,
            phase_seed=int(context["phase_seed"]),
        )
        if rank == 0:
            screening = _screen_context(
                context, sham_episode, target["screening"], cfg
            )
            screening_rows.append(screening)
            eligible = bool(screening["eligible"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'EXCLUDED'}; "
                f"detected={screening['detected_frequency_hz']:g} Hz; "
                f"reason={screening['exclusion_reasons']}"
            )
        else:
            screening, eligible = None, None
        eligible = bool(comm.bcast(eligible, root=0))
        if not eligible:
            continue

        n_futures = int(cfg.analysis.crossed_design.n_future_continuations)
        for future_index in range(n_futures):
            future_seed = _future_seed(cfg, int(context["context_order"]), future_index)
            if future_index == 0:
                paired_sham = sham_episode
            else:
                paired_sham = _run_condition(
                    condition_id=SHAM,
                    condition_cfg=state_cfg,
                    action=_materialize_action(state_cfg, action_specs[0]),
                    stimulate=True,
                    seed=int(context["trial_seed"]),
                    action_index=0,
                    output_dir=root / "episodes" / str(context["context_id"]) / f"future_{future_index:02d}" / SHAM,
                    comm=comm,
                    size=size,
                    rank=rank,
                    structure_seed=int(context["structure_seed"]),
                    drive_seed=int(context["drive_seed"]),
                    future_drive_seed=future_seed,
                    phase_seed=int(context["phase_seed"]),
                )
            for action_index, spec in enumerate(action_specs):
                if str(spec["id"]) == SHAM:
                    active_episode = paired_sham
                else:
                    action_cfg = _with_action_frequency(state_cfg, float(spec["frequency_hz"]))
                    active_episode = _run_condition(
                        condition_id=str(spec["id"]),
                        condition_cfg=action_cfg,
                        action=_materialize_action(action_cfg, spec),
                        stimulate=True,
                        seed=int(context["trial_seed"]),
                        action_index=action_index,
                        output_dir=root / "episodes" / str(context["context_id"]) / f"future_{future_index:02d}" / str(spec["id"]),
                        comm=comm,
                        size=size,
                        rank=rank,
                        structure_seed=int(context["structure_seed"]),
                        drive_seed=int(context["drive_seed"]),
                        future_drive_seed=future_seed,
                        phase_seed=int(context["phase_seed"]),
                    )
                if rank == 0:
                    metric_rows.append(_metric_row(
                        context=context,
                        screening=screening,
                        future_index=future_index,
                        spec=spec,
                        sham_episode=paired_sham,
                        active_episode=active_episode,
                        target=target["outcome"],
                        cfg=cfg,
                    ))
                if str(spec["id"]) != SHAM:
                    del active_episode
            if future_index > 0:
                del paired_sham
        del sham_episode

    if rank != 0:
        return

    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "prospective_screening.csv", index=False)
    if not metric_rows:
        conclusion = {
            "scope": "F0 exploratory frequency/phase system identification",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "frequency_phase_contextual_feasibility_gate_passed": False,
                "eligible_context_count": 0,
                "contextual_bandit_status": "NOT TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No context passed the prospective EEG phenotype and phase screen.",
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(conclusion, indent=2))
        print("\nNo eligible contexts; F0 feasibility gate: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    expected = _expected_action_map(metrics)
    eligible_screening = screening[screening.eligible].copy()
    comparison, structures, best_fixed = _policy_comparison(expected, eligible_screening)
    crossover = _crossover_summary(expected)
    shuffle_null, shuffle_p = _shuffle_null(
        expected, comparison, eligible_screening, best_fixed, cfg
    )
    checks, conclusions = _checks(
        calibration=calibration,
        screening=screening,
        metrics=metrics,
        expected=expected,
        crossover=crossover,
        comparison=comparison,
        structures=structures,
        shuffle_p=shuffle_p,
        cfg=cfg,
    )

    metrics.to_csv(root / "context_action_future_metrics.csv", index=False)
    expected.to_csv(root / "expected_context_action_map.csv", index=False)
    crossover.to_csv(root / "frequency_phase_crossover_summary.csv", index=False)
    comparison.to_csv(root / "eeg_rule_vs_fixed_comparison.csv", index=False)
    structures.to_csv(root / "structure_level_policy_comparison.csv", index=False)
    shuffle_null.to_csv(root / "frequency_context_shuffle_null.csv", index=False)
    provenance = {
        "state_A": {
            "description": "elevated-alpha toy state from mean-rate-matched sinusoidally modulated Poisson afferents",
            "hidden_frequencies_hz": [float(x) for x in cfg.analysis.states.frequencies_hz],
            "modulation_depth": float(cfg.analysis.states.modulation_depth),
            "continuous_phase_randomized_by_context": True,
        },
        "state_B": {
            "description": "low-alpha reference from homogeneous Poisson afferents",
            "modulation_depth": 0.0,
        },
        "invariants": [
            "identical cells and recurrence", "identical inhibition scale",
            "identical expected afferent rate and weights",
        ],
        "actions": action_specs,
        "candidate_eeg_rule": "detect the larger standardized 9/11-Hz excess; apply that frequency at pi EEG-relative phase",
        "best_fixed_action_id": best_fixed,
        "screening_is_prospective": True,
        "selection_performed": "none; all counterfactual actions mapped",
        "hidden_variables_used_by_policy": False,
        "requires_disjoint_confirmation_before_bandit": True,
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(_plain(provenance), indent=2))
    conclusion = {
        "scope": "F0 exploratory ideal-neural-EEG system identification",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure; histories and futures are repeated measures",
        "inference_boundary": "directional feasibility only; three structures do not support confirmatory efficacy claims",
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(_plain(conclusion), indent=2))

    if bool(cfg.experiment.plot):
        _plot_results(
            root=root,
            calibration=calibration,
            screening=screening,
            expected=expected,
            crossover=crossover,
            structures=structures,
        )

    print("\n### F0 screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### F0 frequency/phase feasibility checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nFrequency/phase contextual feasibility gate: "
        f"{'PASSED' if conclusions['frequency_phase_contextual_feasibility_gate_passed'] else 'NOT PASSED'}"
    )
    print("Contextual bandit status: NOT TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
