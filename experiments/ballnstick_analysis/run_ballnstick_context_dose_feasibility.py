"""Low-cost crossed-seed feasibility gate for EEG-contextual tACS dosing.

This CL0 experiment is deliberately not an RL experiment.  It separates
stable circuit-structure randomness from stochastic afferent-event timing,
screens contexts using only unstimulated ideal EEG, and evaluates sham,
0.2-V/m and 0.4-V/m counterfactual outcomes under matched conditions.  A
leave-one-structure-out EEG-only rule tests whether context contains enough
information to improve on the confirmed fixed 0.4-V/m strategy.

Passing is only a directional gate for a later disjoint contextual-bandit
experiment.  It is not confirmation, a depression model, or human evidence.
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
import scipy.stats as st
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    A_HIGH,
    _action,
    _condition_for_seed,
    _epoch_raw,
    _epoch_row,
    _plain,
    _reference_phase,
    _run_condition,
    _sham,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (  # noqa: E402
    _complex_response_decomposition,
    _dose_id,
    _field_removal_status,
    _stimulation_psd,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_screened_confirmation import (  # noqa: E402
    _load_frozen_candidate,
    _screen_phase_quality,
    _screening_decision,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _relative_rate_safe,
)


SHAM_DOSE = 0.0
PRIMARY_CONTEXT_FEATURES = (
    "context_alpha_excess_log10",
    "context_coherent_alpha_fraction",
)


def _seed_values(cfg: DictConfig, *, kind: str) -> list[int]:
    design = cfg.analysis.crossed_design
    count = int(design[f"n_{kind}_seeds"])
    first = int(cfg.experiment.seed) + int(design[f"{kind}_seed_offset"])
    return [first + index for index in range(count)]


def _phase_seed(cfg: DictConfig) -> int:
    return int(cfg.experiment.seed) + int(
        cfg.analysis.crossed_design.fixed_phase_seed_offset
    )


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    structures = _seed_values(cfg, kind="structure")
    drives = _seed_values(cfg, kind="drive")
    phase_seed = _phase_seed(cfg)
    trial_first = int(cfg.experiment.seed) + int(
        cfg.analysis.crossed_design.trial_seed_offset
    )
    result = []
    for structure_index, structure_seed in enumerate(structures):
        for drive_index, drive_seed in enumerate(drives):
            order = structure_index * len(drives) + drive_index
            result.append({
                "context_id": f"s{structure_index + 1}_d{drive_index + 1}",
                "context_order": order + 1,
                "trial_seed": trial_first + order,
                "structure_seed": structure_seed,
                "drive_seed": drive_seed,
                "phase_seed": phase_seed,
                "structure_index": structure_index + 1,
                "drive_index": drive_index + 1,
            })
    return result


def _active_doses(cfg: DictConfig) -> list[float]:
    return [float(value) for value in cfg.analysis.actions.active_doses_v_per_m]


def _all_doses(cfg: DictConfig) -> list[float]:
    return [SHAM_DOSE, *_active_doses(cfg)]


def _dose_condition(dose: float) -> str:
    return A_HIGH if np.isclose(dose, SHAM_DOSE) else _dose_id(dose)


def _validate_design(cfg: DictConfig, frozen: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("CL0 requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("Every CL0 arm requires inhibition_scale=1.")
    design = cfg.analysis.crossed_design
    if int(design.n_structure_seeds) < 1 or int(design.n_drive_seeds) < 1:
        raise ValueError("The crossed design requires structure and drive seeds.")
    structures = _seed_values(cfg, kind="structure")
    drives = _seed_values(cfg, kind="drive")
    if max(structures) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Structure seeds are too large for seed * 10,000.")
    if len(set(structures)) != len(structures) or len(set(drives)) != len(drives):
        raise ValueError("Structure and drive seeds must be unique within namespace.")

    doses = _active_doses(cfg)
    if len(doses) != 2 or doses != sorted(set(doses)):
        raise ValueError("CL0 requires exactly two ordered unique active doses.")
    if any(dose <= 0.0 for dose in doses):
        raise ValueError("Active doses must be positive; sham is added separately.")
    if max(doses) > float(cfg.analysis.maximum_field_v_per_m):
        raise ValueError("An active dose exceeds maximum_field_v_per_m.")
    low = float(cfg.analysis.actions.low_dose_v_per_m)
    fixed = float(cfg.analysis.actions.fixed_comparator_v_per_m)
    if not any(np.isclose(low, dose) for dose in doses):
        raise ValueError("low_dose_v_per_m must appear in the active action set.")
    if not any(np.isclose(fixed, dose) for dose in doses):
        raise ValueError("The fixed comparator must appear in the active action set.")
    candidate = frozen["candidate"]
    if not np.isclose(float(candidate["selected_dose_v_per_m"]), fixed):
        raise ValueError("The fixed comparator must be the confirmed frozen dose.")
    if not np.isclose(float(candidate["frequency_hz"]), 10.0):
        raise ValueError("CL0 retains the frozen 10-Hz frequency.")
    if not np.isclose(float(candidate["relative_phase_offset_rad"]), np.pi):
        raise ValueError("CL0 retains the frozen EEG-relative 180-degree phase.")
    if str(candidate["montage"]) != str(cfg.analysis.tacs.axial_montage):
        raise ValueError("CL0 retains the frozen axial montage.")
    if not np.isclose(float(frozen["target"]["selected_modulation_depth"]), 0.04):
        raise ValueError("CL0 requires the frozen 0.04 elevated-alpha state.")

    timeline = cfg.analysis.timeline
    if int(timeline.baseline_steps) < 4:
        raise ValueError("CL0 phase estimation requires at least four baseline windows.")
    if int(timeline.stimulation_steps) < 4:
        raise ValueError("CL0 EEG estimation requires at least four stimulation windows.")
    stimulation_ms = (
        int(timeline.stimulation_steps) * float(cfg.env.simulation.obs_win_len)
    )
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    if trim_ms < float(timeline.block_ramp_ms) or 2.0 * trim_ms >= stimulation_ms:
        raise ValueError("Analysis trimming must remove both ramps and leave data.")


def _add_context_metadata(
    episode: dict[str, Any], context: dict[str, Any]
) -> None:
    for row in episode["epoch_rows"]:
        row.update(context)


def _phase_tracking(
    *,
    active_episode: dict[str, Any],
    screen_phase: dict[str, Any],
    relative_offset: float,
    phase_seed: int,
) -> dict[str, Any]:
    expected = _wrap_phase(
        float(screen_phase["screen_phase_at_action_rad"])
        + np.pi / 2.0
        + float(relative_offset)
    )
    realized = float(active_episode["simulation"]["action"]["phase_rad"])
    error = abs(float(np.angle(np.exp(1j * (realized - expected)))))
    return {
        "realized_field_phase_rad": realized,
        "expected_field_phase_from_screen_rad": expected,
        "action_phase_tracking_error_rad": error,
        "hidden_input_phase_rad": _reference_phase(phase_seed),
        "phase_quality_pass": bool(
            screen_phase["screen_phase_quality_pass"] and error <= 1.0e-10
        ),
    }


def _context_features(
    *, sham_episode: dict[str, Any], target_model: dict[str, Any]
) -> dict[str, float]:
    sham = _epoch_row(sham_episode)
    alpha_power = max(
        float(sham.alpha_power_8_12_hz), np.finfo(float).tiny
    )
    resultant = float(sham.eeg_10hz_resultant_v)
    return {
        "context_log10_alpha_power": float(
            sham.log10_alpha_power_8_12_hz
        ),
        "context_alpha_excess_log10": float(
            sham.log10_alpha_power_8_12_hz
            - float(target_model["B_mean_log10_alpha"])
        ),
        "context_10hz_resultant_v": resultant,
        "context_coherent_alpha_fraction": float(
            0.5 * resultant**2 / alpha_power
        ),
        "context_alpha_peak_prominence_db": float(
            sham.alpha_peak_prominence_db
        ),
        "context_E_firing_rate_hz": float(sham.E_firing_rate_hz),
        "context_I_firing_rate_hz": float(sham.I_firing_rate_hz),
    }


def _context_action_rows(
    *,
    context: dict[str, Any],
    episodes: dict[str, dict[str, Any]],
    screening: dict[str, Any],
    screen_phase: dict[str, Any],
    target_model: dict[str, Any],
    relative_offset: float,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    sham_episode = episodes[A_HIGH]
    sham = _epoch_row(sham_episode)
    sham_baseline = _epoch_row(sham_episode, "baseline")
    sham_washout = _epoch_row(sham_episode, "washout")
    target = float(target_model["B_mean_log10_alpha"])
    context_values = _context_features(
        sham_episode=sham_episode, target_model=target_model
    )
    initial_distance = abs(
        float(sham.log10_alpha_power_8_12_hz) - target
    )
    base = {
        **context,
        **context_values,
        "screen_margin_toward_A_log10": float(
            screening["screen_margin_toward_A_log10"]
        ),
        "screen_phase_split_error_deg": float(
            screening["screen_phase_split_error_deg"]
        ),
        "screen_10hz_resultant_to_rms": float(
            screening["screen_10hz_resultant_to_rms"]
        ),
        "frozen_B_mean_log10_alpha": target,
        "pre_action_distance_to_B_log10": initial_distance,
    }
    rows = []
    for dose in _all_doses(cfg):
        condition = _dose_condition(dose)
        episode = episodes[condition]
        outcome = _epoch_row(episode)
        post_power = float(outcome.log10_alpha_power_8_12_hz)
        post_distance = abs(post_power - target)
        suppression = float(
            sham.log10_alpha_power_8_12_hz - post_power
        )
        if np.isclose(dose, SHAM_DOSE):
            recovered, tolerance = True, 0.0
            washout_residual = 0.0
            baseline_error = 0.0
            phase = {
                "realized_field_phase_rad": float("nan"),
                "expected_field_phase_from_screen_rad": float("nan"),
                "action_phase_tracking_error_rad": 0.0,
                "hidden_input_phase_rad": _reference_phase(
                    int(context["phase_seed"])
                ),
                "phase_quality_pass": bool(
                    screen_phase["screen_phase_quality_pass"]
                ),
            }
            decomposition = _complex_response_decomposition(
                sham_cosine=float(sham.eeg_10hz_cosine_v),
                sham_sine=float(sham.eeg_10hz_sine_v),
                active_cosine=float(sham.eeg_10hz_cosine_v),
                active_sine=float(sham.eeg_10hz_sine_v),
            )
        else:
            active_baseline = _epoch_row(episode, "baseline")
            active_washout = _epoch_row(episode, "washout")
            washout_residual = float(
                (sham_washout.log10_alpha_power_8_12_hz
                 - sham_baseline.log10_alpha_power_8_12_hz)
                - (active_washout.log10_alpha_power_8_12_hz
                   - active_baseline.log10_alpha_power_8_12_hz)
            )
            recovered, tolerance = _field_removal_status(
                effect_log10=suppression,
                residual_log10=washout_residual,
                cfg=cfg,
            )
            baseline_error = _relative_rms_error(
                _epoch_raw(sham_episode, "baseline"),
                _epoch_raw(episode, "baseline"),
            )
            phase = _phase_tracking(
                active_episode=episode,
                screen_phase=screen_phase,
                relative_offset=relative_offset,
                phase_seed=int(context["phase_seed"]),
            )
            decomposition = _complex_response_decomposition(
                sham_cosine=float(sham.eeg_10hz_cosine_v),
                sham_sine=float(sham.eeg_10hz_sine_v),
                active_cosine=float(outcome.eeg_10hz_cosine_v),
                active_sine=float(outcome.eeg_10hz_sine_v),
            )
        rows.append({
            **base,
            "dose_v_per_m": float(dose),
            "condition_id": condition,
            "post_log10_alpha_power": post_power,
            "post_distance_to_B_log10": post_distance,
            "reward_negative_distance": -post_distance,
            "target_distance_improvement_log10": initial_distance - post_distance,
            "alpha_suppression_log10": suppression,
            "finishes_below_frozen_B": bool(post_power < target),
            "coherent_10hz_suppression_v": float(
                sham.eeg_10hz_resultant_v - outcome.eeg_10hz_resultant_v
            ),
            "alpha_peak_prominence_reduction_db": float(
                sham.alpha_peak_prominence_db - outcome.alpha_peak_prominence_db
            ),
            "E_ppc_reduction": float(sham.E_ppc - outcome.E_ppc),
            "I_ppc_reduction": float(sham.I_ppc - outcome.I_ppc),
            "E_rate_change_hz": float(
                outcome.E_firing_rate_hz - sham.E_firing_rate_hz
            ),
            "I_rate_change_hz": float(
                outcome.I_firing_rate_hz - sham.I_firing_rate_hz
            ),
            "rate_safe": bool(
                np.isclose(dose, SHAM_DOSE)
                or _relative_rate_safe(outcome, sham, cfg)
            ),
            "field_removal_residual_log10": washout_residual,
            "field_removal_tolerance_log10": tolerance,
            "field_removal_recovered": recovered,
            "baseline_relative_rms_error": float(baseline_error),
            **decomposition,
            **phase,
        })
    return rows


def _row_at_dose(group: pd.DataFrame, dose: float) -> pd.Series:
    rows = group[np.isclose(group.dose_v_per_m, dose)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row at dose {dose:g}, found {len(rows)}.")
    return rows.iloc[0]


def _counterfactual_summary(
    metrics: pd.DataFrame, *, cfg: DictConfig
) -> pd.DataFrame:
    low = float(cfg.analysis.actions.low_dose_v_per_m)
    fixed = float(cfg.analysis.actions.fixed_comparator_v_per_m)
    practical = float(cfg.analysis.criteria.practical_advantage_log10)
    rows = []
    for context_id, group in metrics.groupby("context_id", sort=False):
        sham = _row_at_dose(group, SHAM_DOSE)
        low_row = _row_at_dose(group, low)
        fixed_row = _row_at_dose(group, fixed)
        oracle = group.sort_values(
            ["post_distance_to_B_log10", "dose_v_per_m"],
            ascending=[True, True],
        ).iloc[0]
        advantage = float(
            fixed_row.post_distance_to_B_log10
            - oracle.post_distance_to_B_log10
        )
        rows.append({
            **{
                key: sham[key]
                for key in (
                    "context_id", "context_order", "trial_seed",
                    "structure_seed", "drive_seed", "phase_seed",
                    "structure_index", "drive_index",
                    *PRIMARY_CONTEXT_FEATURES,
                    "context_log10_alpha_power",
                    "context_10hz_resultant_v",
                    "context_alpha_peak_prominence_db",
                    "screen_margin_toward_A_log10",
                    "screen_phase_split_error_deg",
                    "screen_10hz_resultant_to_rms",
                    "pre_action_distance_to_B_log10",
                )
            },
            "sham_distance_to_B_log10": float(
                sham.post_distance_to_B_log10
            ),
            "low_dose_v_per_m": low,
            "low_distance_to_B_log10": float(
                low_row.post_distance_to_B_log10
            ),
            "fixed_dose_v_per_m": fixed,
            "fixed_distance_to_B_log10": float(
                fixed_row.post_distance_to_B_log10
            ),
            "low_minus_fixed_distance_log10": float(
                low_row.post_distance_to_B_log10
                - fixed_row.post_distance_to_B_log10
            ),
            "oracle_dose_v_per_m": float(oracle.dose_v_per_m),
            "oracle_distance_to_B_log10": float(
                oracle.post_distance_to_B_log10
            ),
            "oracle_advantage_over_fixed_log10": advantage,
            "oracle_practically_beats_fixed": bool(
                advantage >= practical
                and not np.isclose(float(oracle.dose_v_per_m), fixed)
            ),
            "fixed_alpha_suppression_log10": float(
                fixed_row.alpha_suppression_log10
            ),
            "fixed_induced_10hz_gain_v_per_vpm": float(
                fixed_row.induced_eeg_10hz_resultant_v / fixed
            ),
            "fixed_E_ppc_reduction": float(fixed_row.E_ppc_reduction),
        })
    return pd.DataFrame(rows)


def _fit_arm_models(
    training: pd.DataFrame,
    *,
    doses: list[float],
    ridge: float,
) -> dict[str, Any]:
    raw = training.loc[:, PRIMARY_CONTEXT_FEATURES].to_numpy(float)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    x = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
    penalty = np.diag([0.0, 1.0, 1.0]) * float(ridge)
    coefficients = {}
    for dose in doses:
        column = f"reward_dose_{dose:g}"
        y = training[column].to_numpy(float)
        coefficients[dose] = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _wide_context_table(
    metrics: pd.DataFrame, summary: pd.DataFrame, *, cfg: DictConfig
) -> pd.DataFrame:
    result = summary.copy()
    for dose in _all_doses(cfg):
        values = metrics[np.isclose(metrics.dose_v_per_m, dose)].set_index(
            "context_id"
        )["reward_negative_distance"]
        result[f"reward_dose_{dose:g}"] = result.context_id.map(values)
    return result


def _cross_validated_context_policy(
    context_table: pd.DataFrame, *, cfg: DictConfig
) -> pd.DataFrame:
    structures = sorted(context_table.structure_seed.unique())
    doses = _all_doses(cfg)
    fixed = float(cfg.analysis.actions.fixed_comparator_v_per_m)
    ridge = float(cfg.analysis.context_model.ridge_penalty)
    if len(structures) < 2:
        return pd.DataFrame(columns=[
            "context_id", "structure_seed", "drive_seed", "cv_available",
            "selected_dose_v_per_m", "selected_distance_to_B_log10",
            "fixed_distance_to_B_log10", "contextual_advantage_over_fixed_log10",
        ])
    rows = []
    for heldout in structures:
        train = context_table[context_table.structure_seed.ne(heldout)]
        test = context_table[context_table.structure_seed.eq(heldout)]
        model = _fit_arm_models(training=train, doses=doses, ridge=ridge)
        raw = test.loc[:, PRIMARY_CONTEXT_FEATURES].to_numpy(float)
        x = np.column_stack((
            np.ones(len(raw)),
            (raw - model["mean"]) / model["scale"],
        ))
        predictions = np.column_stack([
            x @ model["coefficients"][dose] for dose in doses
        ])
        selected_indices = np.argmax(predictions, axis=1)
        for row_index, (_, source) in enumerate(test.iterrows()):
            dose = float(doses[int(selected_indices[row_index])])
            reward = float(source[f"reward_dose_{dose:g}"])
            distance = -reward
            fixed_distance = -float(source[f"reward_dose_{fixed:g}"])
            result = {
                "context_id": source.context_id,
                "structure_seed": int(source.structure_seed),
                "drive_seed": int(source.drive_seed),
                "heldout_structure_seed": int(heldout),
                "cv_available": True,
                "selected_dose_v_per_m": dose,
                "selected_distance_to_B_log10": distance,
                "fixed_distance_to_B_log10": fixed_distance,
                "contextual_advantage_over_fixed_log10": fixed_distance - distance,
            }
            for dose_index, candidate in enumerate(doses):
                result[f"predicted_reward_dose_{candidate:g}"] = float(
                    predictions[row_index, dose_index]
                )
            rows.append(result)
    return pd.DataFrame(rows)


def _full_context_model(
    context_table: pd.DataFrame, *, cfg: DictConfig
) -> dict[str, Any]:
    doses = _all_doses(cfg)
    model = _fit_arm_models(
        training=context_table,
        doses=doses,
        ridge=float(cfg.analysis.context_model.ridge_penalty),
    )
    return {
        "status": "exploratory_full_information_fit_not_frozen_for_RL",
        "features": list(PRIMARY_CONTEXT_FEATURES),
        "feature_mean": model["mean"],
        "feature_scale": model["scale"],
        "arm_coefficients": {
            f"{dose:g}": model["coefficients"][dose] for dose in doses
        },
        "ridge_penalty": float(cfg.analysis.context_model.ridge_penalty),
    }


def _two_way_variance(
    frame: pd.DataFrame, *, value: str
) -> dict[str, Any]:
    table = frame.pivot(
        index="structure_seed", columns="drive_seed", values=value
    )
    if table.isna().any().any():
        return {
            "metric": value,
            "balanced_complete_grid": False,
            "total_sum_squares": float("nan"),
            "structure_fraction": float("nan"),
            "drive_fraction": float("nan"),
            "interaction_fraction": float("nan"),
        }
    values = table.to_numpy(float)
    grand = float(values.mean())
    ss_structure = float(
        values.shape[1] * np.sum((values.mean(axis=1) - grand) ** 2)
    )
    ss_drive = float(
        values.shape[0] * np.sum((values.mean(axis=0) - grand) ** 2)
    )
    ss_total = float(np.sum((values - grand) ** 2))
    ss_interaction = max(0.0, ss_total - ss_structure - ss_drive)
    denominator = max(ss_total, np.finfo(float).tiny)
    return {
        "metric": value,
        "balanced_complete_grid": True,
        "total_sum_squares": ss_total,
        "structure_sum_squares": ss_structure,
        "drive_sum_squares": ss_drive,
        "interaction_sum_squares": ss_interaction,
        "structure_fraction": ss_structure / denominator,
        "drive_fraction": ss_drive / denominator,
        "interaction_fraction": ss_interaction / denominator,
    }


def _variance_audit(screening: pd.DataFrame) -> pd.DataFrame:
    names = (
        "screen_log10_alpha_power",
        "screen_10hz_resultant_v",
        "screen_10hz_resultant_to_rms",
        "screen_phase_split_error_deg",
        "screen_E_firing_rate_hz",
        "screen_I_firing_rate_hz",
    )
    return pd.DataFrame([
        _two_way_variance(screening, value=name) for name in names
    ])


def _mechanistic_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    predictors = (
        "context_alpha_excess_log10",
        "context_coherent_alpha_fraction",
        "context_10hz_resultant_v",
        "context_alpha_peak_prominence_db",
    )
    outcomes = (
        "low_minus_fixed_distance_log10",
        "oracle_dose_v_per_m",
        "fixed_alpha_suppression_log10",
        "fixed_induced_10hz_gain_v_per_vpm",
        "fixed_E_ppc_reduction",
    )
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            x = summary[predictor].to_numpy(float)
            y = summary[outcome].to_numpy(float)
            if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
                rho, p_value = float("nan"), float("nan")
            else:
                rho, p_value = st.spearmanr(x, y)
            rows.append({
                "predictor": predictor,
                "outcome": outcome,
                "n_contexts": int(len(x)),
                "spearman_rho": float(rho),
                "unadjusted_p_value": float(p_value),
                "confirmatory_test": False,
            })
    return pd.DataFrame(rows)


def _experiment_checks(
    *,
    screening: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    cv: pd.DataFrame,
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any], pd.DataFrame]:
    criteria = cfg.analysis.criteria
    fixed = float(cfg.analysis.actions.fixed_comparator_v_per_m)
    active = metrics[metrics.dose_v_per_m.gt(0.0)]
    structure_level = (
        cv.groupby("structure_seed", as_index=False)[
            "contextual_advantage_over_fixed_log10"
        ].mean()
        if not cv.empty else pd.DataFrame(columns=[
            "structure_seed", "contextual_advantage_over_fixed_log10"
        ])
    )
    cv_mean = float(
        structure_level.contextual_advantage_over_fixed_log10.mean()
    ) if not structure_level.empty else float("nan")
    cv_positive = float(
        (structure_level.contextual_advantage_over_fixed_log10 > 0.0).mean()
    ) if not structure_level.empty else 0.0
    selected_count = int(cv.selected_dose_v_per_m.nunique()) if not cv.empty else 0
    practical_count = int(summary.oracle_practically_beats_fixed.sum())
    oracle_mean = float(summary.oracle_advantage_over_fixed_log10.mean())
    configured_separation = bool(
        set(_seed_values(cfg, kind="structure")).isdisjoint(
            _seed_values(cfg, kind="drive")
        )
        and _phase_seed(cfg) not in set(_seed_values(cfg, kind="structure"))
        and _phase_seed(cfg) not in set(_seed_values(cfg, kind="drive"))
    )
    checks = {
        "independent_seed_namespaces_configured": configured_separation,
        "complete_crossed_screening_grid": len(screening)
        == int(cfg.analysis.crossed_design.n_structure_seeds)
        * int(cfg.analysis.crossed_design.n_drive_seeds),
        "screening_precedes_and_excludes_stimulation_outcomes": bool(
            (~screening.screening_uses_stimulation_outcome).all()
        ),
        "screening_does_not_use_seed_specific_B": bool(
            (~screening.screening_uses_seed_specific_B).all()
        ),
        "minimum_eligible_contexts": len(summary)
        >= int(criteria.minimum_eligible_contexts),
        "eligible_structure_coverage": summary.structure_seed.nunique()
        >= int(criteria.minimum_eligible_structure_seeds),
        "eligible_drive_coverage": summary.drive_seed.nunique()
        >= int(criteria.minimum_eligible_drive_seeds),
        "all_enrolled_phase_actionable": bool(active.phase_quality_pass.all()),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "field_removal_recovered": bool(active.field_removal_recovered.all()),
        "baseline_causality": bool(
            active.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "coherent_decomposition_exact": bool(np.allclose(
            active.coherent_net_change_v2,
            active.coherent_interference_cross_term_v2
            + active.coherent_induced_component_v2,
            rtol=1.0e-10,
            atol=1.0e-30,
        )),
        "oracle_has_practical_advantage_over_fixed": bool(
            oracle_mean >= float(criteria.minimum_mean_oracle_advantage_log10)
            and practical_count >= int(criteria.minimum_practical_nonfixed_contexts)
            and summary.oracle_dose_v_per_m.nunique() >= 2
        ),
        "leave_one_structure_out_policy_available": not cv.empty,
        "contextual_rule_uses_multiple_actions": selected_count
        >= int(criteria.minimum_contextual_selected_action_count),
        "crossvalidated_contextual_rule_beats_fixed": bool(
            np.isfinite(cv_mean)
            and cv_mean
            >= float(criteria.minimum_mean_contextual_advantage_log10)
            and cv_positive
            >= float(criteria.minimum_positive_structure_fraction)
        ),
        "fixed_comparator_is_frozen_0p4": np.isclose(fixed, 0.4),
    }
    primary = (
        "independent_seed_namespaces_configured",
        "complete_crossed_screening_grid",
        "screening_precedes_and_excludes_stimulation_outcomes",
        "screening_does_not_use_seed_specific_B",
        "minimum_eligible_contexts",
        "eligible_structure_coverage",
        "eligible_drive_coverage",
        "all_enrolled_phase_actionable",
        "all_actions_rate_safe",
        "field_removal_recovered",
        "baseline_causality",
        "oracle_has_practical_advantage_over_fixed",
        "leave_one_structure_out_policy_available",
        "contextual_rule_uses_multiple_actions",
        "crossvalidated_contextual_rule_beats_fixed",
        "fixed_comparator_is_frozen_0p4",
    )
    conclusions = {
        "contextual_bandit_feasibility_gate_passed": all(
            checks[name] for name in primary
        ),
        "scope": "directional full-information CL0 gate; not held-out confirmation",
        "screened_context_count": int(len(screening)),
        "eligible_context_count": int(len(summary)),
        "screening_yield": float(screening.eligible.mean()),
        "oracle_mean_advantage_over_fixed_log10": oracle_mean,
        "practical_nonfixed_oracle_context_count": practical_count,
        "crossvalidated_mean_advantage_over_fixed_log10": cv_mean,
        "crossvalidated_positive_structure_fraction": cv_positive,
        "crossvalidated_selected_doses_v_per_m": sorted(
            cv.selected_dose_v_per_m.unique().tolist()
        ) if not cv.empty else [],
        "fixed_comparator_v_per_m": fixed,
        "policy_observes_only_ideal_EEG": True,
        "hidden_spikes_and_rates_used_only_for_mechanism_and_safety": True,
        "ready_for_RL": all(checks[name] for name in primary),
    }
    return checks, conclusions, structure_level


def _plot_results(
    *,
    root: Path,
    frequencies: np.ndarray,
    psds: dict[str, list[np.ndarray]],
    summary: pd.DataFrame,
    cv: pd.DataFrame,
    cfg: DictConfig,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    colors = {0.0: "#9467BD", 0.2: "#1F77B4", 0.4: "#D62728"}
    for dose in _all_doses(cfg):
        condition = _dose_condition(dose)
        if not psds[condition]:
            continue
        mean_psd = np.mean(np.asarray(psds[condition]), axis=0)
        label = "A sham" if np.isclose(dose, 0.0) else f"A + {dose:g} V/m"
        axis.plot(
            frequencies,
            10.0 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny)),
            label=label,
            color=colors.get(dose),
            linewidth=2.0,
        )
    axis.axvspan(8.0, 12.0, color="gold", alpha=0.14)
    axis.set_xlim(2.0, 25.0)
    axis.set(
        xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)",
        title="Screen-positive ideal neural EEG across CL0 actions",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_context_action_psd.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    scatter = axis.scatter(
        summary.context_alpha_excess_log10,
        summary.low_minus_fixed_distance_log10,
        c=summary.context_coherent_alpha_fraction,
        cmap="viridis",
        s=65,
        edgecolor="black",
        linewidth=0.4,
    )
    axis.axhline(0.0, color="0.4", linewidth=0.9)
    axis.set(
        xlabel="Baseline alpha excess above frozen B (log10)",
        ylabel="Distance at 0.2 − distance at 0.4 (log10)",
        title="EEG context-by-dose opportunity",
    )
    figure.colorbar(scatter, ax=axis, label="Coherent alpha fraction")
    figure.tight_layout()
    figure.savefig(root / "figure_02_context_dose_interaction.png", dpi=250)
    plt.close(figure)

    if cv.empty:
        return
    merged = summary.merge(
        cv[["context_id", "selected_distance_to_B_log10"]],
        on="context_id",
    ).sort_values("context_order")
    x = np.arange(len(merged))
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.plot(x, merged.fixed_distance_to_B_log10, "o-", label="Fixed 0.4 V/m")
    axis.plot(x, merged.selected_distance_to_B_log10, "o-", label="Cross-fitted context rule")
    axis.plot(x, merged.oracle_distance_to_B_log10, "o--", label="Counterfactual oracle")
    axis.set_xticks(x, merged.context_id, rotation=45, ha="right")
    axis.set(
        ylabel="Absolute distance to frozen B (log10)",
        title="CL0 contextual opportunity versus fixed control",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_03_context_policy_comparison.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen = _load_frozen_candidate(cfg)
    _validate_design(cfg, frozen)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "context_dose_feasibility"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### CL0 crossed-seed EEG-context dose feasibility")
        print(json.dumps(_plain(frozen), indent=2))
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()

    started = time.perf_counter()
    target_model = frozen["target"]["target_model"]
    depth = float(frozen["target"]["selected_modulation_depth"])
    candidate = frozen["candidate"]
    relative_offset = float(candidate["relative_phase_offset_rad"])
    montage = str(candidate["montage"])
    contexts = _context_specs(cfg)
    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    psds = {_dose_condition(dose): [] for dose in _all_doses(cfg)}
    frequencies = None

    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} "
                f"structure={context['structure_seed']} drive={context['drive_seed']}"
            )
            episodes: dict[str, dict[str, Any]] = {}
        a_cfg = _condition_for_seed(
            cfg,
            seed=int(context["phase_seed"]),
            modulation_depth=depth,
        )
        sham_episode = _run_condition(
            condition_id=A_HIGH,
            condition_cfg=a_cfg,
            action=_sham(cfg, A_HIGH),
            stimulate=False,
            seed=int(context["trial_seed"]),
            action_index=0,
            output_dir=root / "episodes" / context["context_id"] / A_HIGH,
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(context["structure_seed"]),
            drive_seed=int(context["drive_seed"]),
            phase_seed=int(context["phase_seed"]),
        )
        if rank == 0:
            _add_context_metadata(sham_episode, context)
            screen_phase = _screen_phase_quality(
                episode=sham_episode,
                relative_offset=relative_offset,
                cfg=cfg,
            )
            screen = _screening_decision(
                seed=int(context["trial_seed"]),
                screening_order=int(context["context_order"]),
                a_episode=sham_episode,
                phase_quality=screen_phase,
                target_model=target_model,
                cfg=cfg,
            )
            screen.update(context)
            screening_rows.append(screen)
            eligible = bool(screen["eligible"])
            print(
                f"screen {context['context_id']}: "
                f"{'ELIGIBLE' if eligible else 'EXCLUDED'} "
                f"({screen['exclusion_reasons']})"
            )
        else:
            eligible = None
        eligible = bool(comm.bcast(eligible, root=0))
        if not eligible:
            continue
        if rank == 0:
            episodes[A_HIGH] = sham_episode
            epoch_rows.extend(sham_episode["epoch_rows"])
            if bool(cfg.experiment.plot):
                frequencies, psd = _stimulation_psd(sham_episode, cfg=cfg)
                psds[A_HIGH].append(psd)

        for action_index, dose in enumerate(_active_doses(cfg), start=1):
            condition = _dose_condition(dose)
            episode = _run_condition(
                condition_id=condition,
                condition_cfg=a_cfg,
                action=_action(
                    cfg,
                    identifier=condition,
                    role="CL0_counterfactual_context_dose",
                    amplitude=dose,
                    montage=montage,
                    relative_offset=relative_offset,
                ),
                stimulate=True,
                seed=int(context["trial_seed"]),
                action_index=action_index,
                output_dir=root / "episodes" / context["context_id"] / condition,
                comm=comm,
                size=size,
                rank=rank,
                structure_seed=int(context["structure_seed"]),
                drive_seed=int(context["drive_seed"]),
                phase_seed=int(context["phase_seed"]),
            )
            if rank == 0:
                _add_context_metadata(episode, context)
                episodes[condition] = episode
                epoch_rows.extend(episode["epoch_rows"])
                if bool(cfg.experiment.plot):
                    frequencies, psd = _stimulation_psd(episode, cfg=cfg)
                    psds[condition].append(psd)
        if rank == 0:
            metric_rows.extend(_context_action_rows(
                context=context,
                episodes=episodes,
                screening=screening_rows[-1],
                screen_phase=screen_phase,
                target_model=target_model,
                relative_offset=relative_offset,
                cfg=cfg,
            ))
            del episodes

    if rank != 0:
        return

    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "screening_audit.csv", index=False)
    if not metric_rows:
        result = {
            "scope": "directional full-information CL0 gate",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "contextual_bandit_feasibility_gate_passed": False,
                "eligible_context_count": 0,
                "ready_for_RL": False,
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No crossed context passed the frozen EEG/phase screen.",
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        print("\nNo eligible CL0 contexts; feasibility gate: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    epochs = pd.DataFrame(epoch_rows)
    summary = _counterfactual_summary(metrics, cfg=cfg)
    context_table = _wide_context_table(metrics, summary, cfg=cfg)
    cv = _cross_validated_context_policy(context_table, cfg=cfg)
    variance = _variance_audit(screening)
    correlations = _mechanistic_correlations(summary)
    model = _full_context_model(context_table, cfg=cfg)
    checks, conclusions, structure_level = _experiment_checks(
        screening=screening,
        metrics=metrics,
        summary=summary,
        cv=cv,
        cfg=cfg,
    )

    epochs.to_csv(root / "context_epoch_eeg_and_hidden_metrics.csv", index=False)
    metrics.to_csv(root / "context_action_metrics.csv", index=False)
    summary.to_csv(root / "context_counterfactual_summary.csv", index=False)
    cv.to_csv(root / "cross_validated_context_policy.csv", index=False)
    structure_level.to_csv(root / "structure_level_policy_comparison.csv", index=False)
    variance.to_csv(root / "seed_variance_decomposition.csv", index=False)
    correlations.to_csv(root / "mechanistic_context_correlations.csv", index=False)
    (root / "exploratory_context_model.json").write_text(
        json.dumps(_plain(model), indent=2)
    )
    provenance = {
        **frozen,
        "crossed_contexts": contexts,
        "active_doses_v_per_m": _active_doses(cfg),
        "fixed_phase_seed": _phase_seed(cfg),
        "absolute_input_phase_held_fixed_for_variance_attribution": True,
        "structure_seed_controls": (
            "LFPy/NumPy cell placement, recurrent topology, weights, delays, "
            "multapses, and synaptic locations"
        ),
        "drive_seed_controls": "per-synapse stochastic Poisson candidate events",
        "selection_performed": "exploratory full-information contextual feasibility only",
        "requires_disjoint_confirmation": True,
    }
    (root / "frozen_protocol_provenance.json").write_text(
        json.dumps(_plain(provenance), indent=2)
    )
    result = {
        "scope": "ideal neural-only EEG, screen-positive crossed toy contexts",
        "checks": checks,
        "conclusions": conclusions,
        "primary_comparator": "frozen EEG-relative 10-Hz axial 0.4-V/m policy",
        "context_features": list(PRIMARY_CONTEXT_FEATURES),
        "reward": "negative absolute log10 alpha-power distance to frozen B mean",
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "A pass supports implementing a disjoint contextual-bandit trial; it "
            "does not itself validate an RL policy. A failure means the tested "
            "baseline EEG context/action set did not beat fixed 0.4 V/m."
        ),
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_plain(result), indent=2)
    )
    if bool(cfg.experiment.plot):
        _plot_results(
            root=root,
            frequencies=np.asarray(frequencies),
            psds=psds,
            summary=summary,
            cv=cv,
            cfg=cfg,
        )

    print("\n### CL0 screening")
    print(f"crossed contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### CL0 feasibility checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nContextual bandit feasibility gate:",
        "PASSED" if conclusions["contextual_bandit_feasibility_gate_passed"]
        else "NOT PASSED",
    )
    print("Confirmation status: NOT TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
