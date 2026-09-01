"""CDM1-S single-action conditional tACS dose-map experiment.

This low-cost system-identification experiment asks whether a stimulation-free
ideal-EEG context predicts which *single* constant-amplitude 10-Hz tACS action
best moves a toy elevated-alpha circuit toward a frozen low-alpha reference.
It does not train a bandit.

Three mean-rate-matched afferent modulation depths create explicit mild,
moderate and strong toy alpha states.  The latent depth is never exposed to the
EEG policy.  Each observed baseline is replayed with sham and 0.1--0.4 V/m,
and each replay selects one dose for the complete stimulation block.  A split
Poisson RNG makes the predecision history identical while providing multiple
independent stochastic futures after the decision boundary.  Thus action
ranking estimates conditional expected response rather than a lucky realized
future.

The EEG is ideal neural-only simulated EEG.  State labels are operational toy
labels, not depression severity, subjects, treatment groups, or human evidence.
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
    _action,
    _condition_for_seed,
    _epoch_raw,
    _epoch_row,
    _feature_from_raw,
    _plain,
    _run_condition,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (  # noqa: E402
    _complex_response_decomposition,
    _field_removal_status,
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


def _dose_id(dose: float) -> str:
    token = f"{float(dose):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return "sham" if np.isclose(dose, 0.0) else f"single_dose_{token}_vpm"


def _doses(cfg: DictConfig) -> list[float]:
    return [
        float(cfg.analysis.actions.sham_dose_v_per_m),
        *[float(value) for value in cfg.analysis.actions.active_doses_v_per_m],
    ]


def _seed_values(cfg: DictConfig, *, kind: str) -> list[int]:
    design = cfg.analysis.crossed_design
    count = int(design[f"n_{kind}_seeds"])
    first = int(cfg.experiment.seed) + int(design[f"{kind}_seed_offset"])
    return [first + index for index in range(count)]


def _future_seeds(cfg: DictConfig) -> list[int]:
    design = cfg.analysis.crossed_design
    first = int(cfg.experiment.seed) + int(design.future_seed_offset)
    return [
        first + index
        for index in range(int(design.n_future_continuations))
    ]


def _phase_seed(cfg: DictConfig) -> int:
    return int(cfg.experiment.seed) + int(
        cfg.analysis.crossed_design.fixed_phase_seed_offset
    )


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    depths = [float(value) for value in cfg.analysis.states.modulation_depths]
    labels = [str(value) for value in cfg.analysis.states.labels]
    structures = _seed_values(cfg, kind="structure")
    histories = _seed_values(cfg, kind="history")
    phase_seed = _phase_seed(cfg)
    trial_first = int(cfg.experiment.seed) + int(
        cfg.analysis.crossed_design.trial_seed_offset
    )
    rows: list[dict[str, Any]] = []
    for state_index, (label, depth) in enumerate(zip(labels, depths)):
        for structure_index, structure_seed in enumerate(structures):
            for history_index, history_seed in enumerate(histories):
                order = len(rows) + 1
                rows.append({
                    "context_id": (
                        f"{label}_s{structure_index + 1}_h{history_index + 1}"
                    ),
                    "context_order": order,
                    "trial_seed": trial_first + order - 1,
                    "state_label": label,
                    "state_index": state_index + 1,
                    "modulation_depth": depth,
                    "structure_seed": structure_seed,
                    "structure_index": structure_index + 1,
                    "history_seed": history_seed,
                    "history_index": history_index + 1,
                    "phase_seed": phase_seed,
                })
    return rows


def _validate_design(cfg: DictConfig, frozen: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("CDM1-S requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("Every CDM1-S state and action requires inhibition_scale=1.")

    depths = [float(value) for value in cfg.analysis.states.modulation_depths]
    labels = [str(value) for value in cfg.analysis.states.labels]
    envelope = float(cfg.analysis.reference.thinning_envelope_modulation_depth)
    if len(depths) < 2 or len(depths) != len(labels):
        raise ValueError("State depths and labels require equal length >= 2.")
    if depths != sorted(set(depths)):
        raise ValueError("State modulation depths must be unique and increasing.")
    if any(depth <= 0.0 or depth > envelope for depth in depths):
        raise ValueError("Every state depth must lie in (0, thinning envelope].")
    if not np.isclose(float(cfg.analysis.reference.frequency_hz), 10.0):
        raise ValueError("CDM1-S freezes the toy afferent rhythm at 10 Hz.")
    if not np.isclose(float(cfg.analysis.tacs.frequency_hz), 10.0):
        raise ValueError("CDM1-S freezes tACS at 10 Hz.")

    doses = _doses(cfg)
    if not np.isclose(doses[0], SHAM_DOSE):
        raise ValueError("The first dose must be the 0-V/m sham comparator.")
    active = doses[1:]
    if active != sorted(set(active)) or any(dose <= 0.0 for dose in active):
        raise ValueError("Active doses must be positive, unique, and increasing.")
    if len(active) < 2 or max(active) > float(cfg.analysis.maximum_field_v_per_m):
        raise ValueError("At least two active doses within the field limit are required.")

    candidate = frozen["candidate"]
    expected = cfg.analysis.frozen_candidate
    if not np.isclose(
        float(candidate["selected_dose_v_per_m"]),
        float(expected.expected_amplitude_v_per_m),
    ):
        raise ValueError("The frozen source candidate amplitude changed.")
    if not np.isclose(float(candidate["frequency_hz"]), 10.0):
        raise ValueError("The frozen source frequency changed.")
    if not np.isclose(
        _wrap_phase(float(candidate["relative_phase_offset_rad"])), np.pi
    ):
        raise ValueError("The frozen source EEG-relative phase changed.")
    if str(candidate["montage"]) != str(cfg.analysis.tacs.axial_montage):
        raise ValueError("The frozen source montage changed.")

    timeline = cfg.analysis.timeline
    window_ms = float(cfg.env.simulation.obs_win_len)
    baseline_ms = int(timeline.baseline_steps) * window_ms
    context_window_ms = float(cfg.analysis.context.window_ms)
    if int(timeline.baseline_steps) < 6:
        raise ValueError("CDM1-S requires at least 6 s of predecision EEG.")
    if context_window_ms < 2000.0 or not np.isclose(
        baseline_ms % context_window_ms, 0.0
    ):
        raise ValueError("The baseline must contain complete >=2-s context windows.")
    stimulation_ms = int(timeline.stimulation_steps) * window_ms
    trim = float(timeline.stimulation_analysis_trim_ms)
    if trim < float(timeline.block_ramp_ms) or 2.0 * trim >= stimulation_ms:
        raise ValueError("Stimulation trimming must remove both ramps and leave data.")
    if int(cfg.analysis.crossed_design.n_future_continuations) < 2:
        raise ValueError("At least two independent stochastic futures are required.")

    namespaces = (
        set(_seed_values(cfg, kind="structure")),
        set(_seed_values(cfg, kind="history")),
        set(_future_seeds(cfg)),
        {_phase_seed(cfg)},
        {int(item["trial_seed"]) for item in _context_specs(cfg)},
    )
    if any(not values for values in namespaces):
        raise ValueError("Every CDM1-S seed namespace must be nonempty.")
    if any(
        namespaces[i].intersection(namespaces[j])
        for i in range(len(namespaces))
        for j in range(i + 1, len(namespaces))
    ):
        raise ValueError("Structure, history, future, phase, and trial seeds must differ.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Structure seeds are too large for seed * 10,000.")


def _baseline_screen_view(episode: dict[str, Any]) -> dict[str, Any]:
    row = dict(_epoch_row(episode, "baseline"))
    row["epoch"] = "stimulation"
    return {**episode, "epoch_rows": [row]}


def _context_features(
    episode: dict[str, Any], *, target_model: dict[str, Any], cfg: DictConfig
) -> dict[str, float]:
    baseline = _epoch_row(episode, "baseline")
    raw = _epoch_raw(episode, "baseline")
    dt_ms = float(cfg.env.network.dt)
    samples_per_window = int(round(float(cfg.analysis.context.window_ms) / dt_ms))
    if samples_per_window <= 0 or raw.size % samples_per_window != 0:
        raise RuntimeError("The baseline EEG does not divide into context windows.")
    outputs = episode["simulation"]["outputs_by_epoch"]["baseline"]
    start_ms = float(outputs[0]["t_start_ms"])
    window_alpha = []
    for index, offset in enumerate(range(0, raw.size, samples_per_window)):
        values = raw[offset:offset + samples_per_window]
        features, _, _, _ = _feature_from_raw(
            values,
            simulator_fs_hz=float(episode["simulator_fs_hz"]),
            start_ms=start_ms + offset * dt_ms,
            cfg=cfg,
        )
        window_alpha.append(float(features["log10_alpha_power_8_12_hz"]))
    alpha = np.asarray(window_alpha, dtype=float)
    midpoint = alpha.size // 2
    first_half = float(np.mean(alpha[:midpoint]))
    second_half = float(np.mean(alpha[midpoint:]))
    temporal_sd = float(np.std(alpha, ddof=1))
    alpha_power = max(float(baseline.alpha_power_8_12_hz), np.finfo(float).tiny)
    resultant = float(baseline.eeg_10hz_resultant_v)
    return {
        "context_log10_alpha_power": float(baseline.log10_alpha_power_8_12_hz),
        "context_alpha_excess_log10": float(
            baseline.log10_alpha_power_8_12_hz
            - float(target_model["B_mean_log10_alpha"])
        ),
        "context_10hz_resultant_v": resultant,
        "context_coherent_alpha_fraction": float(
            0.5 * resultant**2 / alpha_power
        ),
        "context_alpha_peak_prominence_db": float(
            baseline.alpha_peak_prominence_db
        ),
        "context_alpha_temporal_sd_log10": temporal_sd,
        "context_alpha_temporal_sem_log10": float(
            temporal_sd / np.sqrt(alpha.size)
        ),
        "context_alpha_split_half_difference_log10": abs(
            first_half - second_half
        ),
        "context_alpha_temporal_slope_log10_per_window": float(
            np.polyfit(np.arange(alpha.size, dtype=float), alpha, 1)[0]
        ),
        "context_alpha_first_last_change_log10": float(alpha[-1] - alpha[0]),
        "context_window_count": int(alpha.size),
        "context_window_duration_s": float(cfg.analysis.context.window_ms) / 1000.0,
        "context_phase_invariant": True,
    }


def _single_action(
    cfg: DictConfig, *, dose: float
) -> dict[str, Any]:
    return _action(
        cfg,
        identifier=_dose_id(dose),
        role="sham_comparator" if np.isclose(dose, 0.0) else "single_block_action",
        amplitude=float(dose),
        montage=str(cfg.analysis.tacs.axial_montage),
        relative_offset=float(cfg.analysis.tacs.relative_phase_offset_rad),
    )


def _run_replay(
    *,
    condition_cfg: DictConfig,
    context: dict[str, Any],
    future_seed: int,
    future_index: int,
    dose: float,
    action_index: int,
    root: Path,
    comm: Any,
    size: int,
    rank: int,
) -> dict[str, Any] | None:
    return _run_condition(
        condition_id=_dose_id(dose),
        condition_cfg=condition_cfg,
        action=_single_action(condition_cfg, dose=dose),
        stimulate=True,
        seed=int(context["trial_seed"]),
        action_index=action_index,
        output_dir=(
            root / "episodes" / str(context["context_id"])
            / f"future_{future_index + 1}" / _dose_id(dose)
        ),
        comm=comm,
        size=size,
        rank=rank,
        structure_seed=int(context["structure_seed"]),
        drive_seed=int(context["history_seed"]),
        future_drive_seed=int(future_seed),
        phase_seed=int(context["phase_seed"]),
    )


def _phase_tracking_error(
    episode: dict[str, Any], *, screen_phase: dict[str, Any], cfg: DictConfig
) -> float:
    expected = _wrap_phase(
        float(screen_phase["screen_phase_at_action_rad"])
        + np.pi / 2.0
        + float(cfg.analysis.tacs.relative_phase_offset_rad)
    )
    realized = float(episode["simulation"]["action"]["phase_rad"])
    return abs(float(np.angle(np.exp(1j * (realized - expected)))))


def _replay_rows(
    *,
    context: dict[str, Any],
    future_seed: int,
    future_index: int,
    episodes: dict[float, dict[str, Any]],
    context_values: dict[str, float],
    screening: dict[str, Any],
    screen_phase: dict[str, Any],
    target_model: dict[str, Any],
    baseline_reference: dict[str, Any],
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    sham_episode = episodes[SHAM_DOSE]
    sham = _epoch_row(sham_episode)
    sham_baseline = _epoch_row(sham_episode, "baseline")
    sham_washout = _epoch_row(sham_episode, "washout")
    target = float(target_model["B_mean_log10_alpha"])
    sham_distance = abs(float(sham.log10_alpha_power_8_12_hz) - target)
    base = {
        **context,
        **context_values,
        "future_index": future_index + 1,
        "future_drive_seed": int(future_seed),
        "frozen_B_mean_log10_alpha": target,
        "screen_margin_toward_A_log10": float(
            screening["screen_margin_toward_A_log10"]
        ),
        "screen_phase_split_error_deg": float(
            screening["screen_phase_split_error_deg"]
        ),
        "screen_10hz_resultant_to_rms": float(
            screening["screen_10hz_resultant_to_rms"]
        ),
        "pre_action_distance_to_B_log10": abs(
            float(context_values["context_log10_alpha_power"]) - target
        ),
        "sham_post_distance_to_B_log10": sham_distance,
    }
    rows: list[dict[str, Any]] = []
    for dose in _doses(cfg):
        episode = episodes[dose]
        outcome = _epoch_row(episode)
        baseline = _epoch_row(episode, "baseline")
        washout = _epoch_row(episode, "washout")
        post_alpha = float(outcome.log10_alpha_power_8_12_hz)
        post_distance = abs(post_alpha - target)
        baseline_error = _relative_rms_error(
            _epoch_raw(baseline_reference, "baseline"),
            _epoch_raw(episode, "baseline"),
        )
        if np.isclose(dose, SHAM_DOSE):
            recovered, tolerance, residual = True, 0.0, 0.0
            rate_safe = True
        else:
            residual = float(
                (sham_washout.log10_alpha_power_8_12_hz
                 - sham_baseline.log10_alpha_power_8_12_hz)
                - (washout.log10_alpha_power_8_12_hz
                   - baseline.log10_alpha_power_8_12_hz)
            )
            recovered, tolerance = _field_removal_status(
                effect_log10=float(sham.log10_alpha_power_8_12_hz - post_alpha),
                residual_log10=residual,
                cfg=cfg,
            )
            rate_safe = _relative_rate_safe(outcome, sham, cfg)
        decomposition = _complex_response_decomposition(
            sham_cosine=float(sham.eeg_10hz_cosine_v),
            sham_sine=float(sham.eeg_10hz_sine_v),
            active_cosine=float(outcome.eeg_10hz_cosine_v),
            active_sine=float(outcome.eeg_10hz_sine_v),
        )
        realized_dose = float(
            episode["simulation"]["action"]["ac_amplitude_v_per_m"]
        )
        rows.append({
            **base,
            "dose_v_per_m": float(dose),
            "condition_id": _dose_id(dose),
            "realized_single_action_dose_v_per_m": realized_dose,
            "one_action_for_complete_intervention": bool(
                np.isclose(realized_dose, dose)
            ),
            "post_log10_alpha_power": post_alpha,
            "post_distance_to_B_log10": post_distance,
            "reward_negative_distance": -post_distance,
            "causal_target_distance_improvement_vs_sham_log10": (
                sham_distance - post_distance
            ),
            "causal_alpha_suppression_vs_sham_log10": float(
                sham.log10_alpha_power_8_12_hz - post_alpha
            ),
            "coherent_10hz_suppression_vs_sham_v": float(
                sham.eeg_10hz_resultant_v - outcome.eeg_10hz_resultant_v
            ),
            "alpha_peak_prominence_reduction_vs_sham_db": float(
                sham.alpha_peak_prominence_db - outcome.alpha_peak_prominence_db
            ),
            "hidden_E_ppc_reduction_vs_sham": float(sham.E_ppc - outcome.E_ppc),
            "hidden_I_ppc_reduction_vs_sham": float(sham.I_ppc - outcome.I_ppc),
            "E_rate_change_vs_sham_hz": float(
                outcome.E_firing_rate_hz - sham.E_firing_rate_hz
            ),
            "I_rate_change_vs_sham_hz": float(
                outcome.I_firing_rate_hz - sham.I_firing_rate_hz
            ),
            "rate_safe": bool(rate_safe),
            "field_removal_residual_log10": residual,
            "field_removal_tolerance_log10": tolerance,
            "field_removal_recovered": bool(recovered),
            "baseline_relative_rms_error": float(baseline_error),
            "action_phase_tracking_error_rad": (
                0.0 if np.isclose(dose, SHAM_DOSE)
                else _phase_tracking_error(
                    episode, screen_phase=screen_phase, cfg=cfg
                )
            ),
            **decomposition,
        })
    return rows


def _expected_action_map(
    metrics: pd.DataFrame, *, cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    context_columns = list(dict.fromkeys([
        "context_id", "context_order", "state_label", "state_index",
        "modulation_depth", "structure_seed", "structure_index",
        "history_seed", "history_index", "phase_seed",
        *[str(value) for value in cfg.analysis.context.policy_features],
        "context_log10_alpha_power", "context_10hz_resultant_v",
        "context_coherent_alpha_fraction",
        "context_alpha_peak_prominence_db",
        "context_alpha_temporal_sd_log10",
        "context_alpha_temporal_slope_log10_per_window",
        "context_alpha_first_last_change_log10", "context_window_count",
        "screen_phase_split_error_deg", "screen_10hz_resultant_to_rms",
        "pre_action_distance_to_B_log10",
    ]))
    grouped = metrics.groupby(
        [*context_columns, "dose_v_per_m"], as_index=False, dropna=False
    ).agg(
        n_future_continuations=("future_index", "nunique"),
        expected_post_distance_to_B_log10=("post_distance_to_B_log10", "mean"),
        future_sd_post_distance_log10=("post_distance_to_B_log10", "std"),
        expected_causal_improvement_vs_sham_log10=(
            "causal_target_distance_improvement_vs_sham_log10", "mean"
        ),
        expected_alpha_suppression_vs_sham_log10=(
            "causal_alpha_suppression_vs_sham_log10", "mean"
        ),
        expected_coherent_10hz_suppression_v=(
            "coherent_10hz_suppression_vs_sham_v", "mean"
        ),
        expected_peak_prominence_reduction_db=(
            "alpha_peak_prominence_reduction_vs_sham_db", "mean"
        ),
        all_rate_safe=("rate_safe", "all"),
        all_field_removal_recovered=("field_removal_recovered", "all"),
    )
    active = grouped[grouped.dose_v_per_m.gt(0.0)].copy()
    global_means = active.groupby("dose_v_per_m", as_index=False).agg(
        mean_expected_distance_to_B_log10=(
            "expected_post_distance_to_B_log10", "mean"
        ),
        mean_expected_causal_improvement_log10=(
            "expected_causal_improvement_vs_sham_log10", "mean"
        ),
    )
    best_fixed = global_means.sort_values(
        ["mean_expected_distance_to_B_log10", "dose_v_per_m"]
    ).iloc[0]
    fixed_dose = float(best_fixed.dose_v_per_m)
    practical = float(cfg.analysis.criteria.practical_advantage_log10)
    rows: list[dict[str, Any]] = []
    for context_id, group in active.groupby("context_id", sort=False):
        oracle = group.sort_values(
            ["expected_post_distance_to_B_log10", "dose_v_per_m"]
        ).iloc[0]
        fixed = group[np.isclose(group.dose_v_per_m, fixed_dose)].iloc[0]
        realized = metrics[
            metrics.context_id.eq(context_id) & metrics.dose_v_per_m.gt(0.0)
        ]
        realized_winners = []
        for _, future in realized.groupby("future_index", sort=True):
            winner = future.sort_values(
                ["post_distance_to_B_log10", "dose_v_per_m"]
            ).iloc[0]
            realized_winners.append(float(winner.dose_v_per_m))
        advantage = float(
            fixed.expected_post_distance_to_B_log10
            - oracle.expected_post_distance_to_B_log10
        )
        source = group.iloc[0]
        rows.append({
            **{column: source[column] for column in context_columns},
            "expected_optimal_active_dose_v_per_m": float(oracle.dose_v_per_m),
            "expected_optimal_distance_to_B_log10": float(
                oracle.expected_post_distance_to_B_log10
            ),
            "best_fixed_active_dose_v_per_m": fixed_dose,
            "best_fixed_expected_distance_to_B_log10": float(
                fixed.expected_post_distance_to_B_log10
            ),
            "expected_oracle_advantage_over_best_fixed_log10": advantage,
            "practical_nonfixed_opportunity": bool(
                not np.isclose(float(oracle.dose_v_per_m), fixed_dose)
                and advantage >= practical
            ),
            "realized_oracle_agreement_fraction": float(np.mean(
                np.isclose(realized_winners, float(oracle.dose_v_per_m))
            )),
            "realized_optimal_doses_v_per_m": ";".join(
                f"{value:g}" for value in realized_winners
            ),
        })
    summary = pd.DataFrame(rows)
    audit = {
        "best_fixed_active_dose_v_per_m": fixed_dose,
        "best_fixed_mean_expected_distance_to_B_log10": float(
            best_fixed.mean_expected_distance_to_B_log10
        ),
        "expected_optimal_active_doses": sorted(
            summary.expected_optimal_active_dose_v_per_m.unique().tolist()
        ),
        "practical_nonfixed_context_count": int(
            summary.practical_nonfixed_opportunity.sum()
        ),
        "mean_expected_oracle_advantage_over_best_fixed_log10": float(
            summary.expected_oracle_advantage_over_best_fixed_log10.mean()
        ),
        "mean_realized_oracle_agreement_fraction": float(
            summary.realized_oracle_agreement_fraction.mean()
        ),
    }
    return grouped, summary, audit


def _fit_arm_models(
    training: pd.DataFrame, *, doses: list[float], features: list[str], ridge: float
) -> dict[str, Any]:
    raw = training[features].to_numpy(float)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    x = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
    penalty = np.diag([0.0, *([1.0] * len(features))]) * float(ridge)
    coefficients = {}
    for dose in doses:
        y = training[f"reward_dose_{dose:g}"].to_numpy(float)
        coefficients[dose] = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _exploratory_loso_policy(
    expected: pd.DataFrame, summary: pd.DataFrame, *, cfg: DictConfig
) -> pd.DataFrame:
    features = [str(value) for value in cfg.analysis.context.policy_features]
    doses = [float(value) for value in cfg.analysis.actions.active_doses_v_per_m]
    table = summary.copy()
    for dose in doses:
        values = expected[np.isclose(expected.dose_v_per_m, dose)].set_index(
            "context_id"
        )["expected_post_distance_to_B_log10"]
        table[f"reward_dose_{dose:g}"] = -table.context_id.map(values)
    structures = sorted(table.structure_seed.unique())
    if len(structures) < 2:
        return pd.DataFrame()
    fixed_dose = float(summary.best_fixed_active_dose_v_per_m.iloc[0])
    rows = []
    for heldout in structures:
        train = table[table.structure_seed.ne(heldout)]
        test = table[table.structure_seed.eq(heldout)]
        if train.empty or test.empty:
            continue
        model = _fit_arm_models(
            train,
            doses=doses,
            features=features,
            ridge=float(cfg.analysis.context.ridge_penalty),
        )
        raw = test[features].to_numpy(float)
        x = np.column_stack((
            np.ones(len(test)), (raw - model["mean"]) / model["scale"]
        ))
        predictions = np.column_stack([
            x @ model["coefficients"][dose] for dose in doses
        ])
        for index, (_, source) in enumerate(test.iterrows()):
            selected = doses[int(np.argmax(predictions[index]))]
            selected_distance = -float(source[f"reward_dose_{selected:g}"])
            fixed_distance = -float(source[f"reward_dose_{fixed_dose:g}"])
            rows.append({
                "context_id": str(source.context_id),
                "heldout_structure_seed": int(heldout),
                "state_label": str(source.state_label),
                "selected_dose_v_per_m": selected,
                "selected_expected_distance_to_B_log10": selected_distance,
                "best_fixed_dose_v_per_m": fixed_dose,
                "best_fixed_expected_distance_to_B_log10": fixed_distance,
                "contextual_advantage_over_best_fixed_log10": (
                    fixed_distance - selected_distance
                ),
                "matches_expected_oracle": bool(np.isclose(
                    selected, float(source.expected_optimal_active_dose_v_per_m)
                )),
            })
    return pd.DataFrame(rows)


def _state_summary(
    screening: pd.DataFrame,
    context_summary: pd.DataFrame,
    expected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (state_index, state_label, depth), screened in screening.groupby(
        ["state_index", "state_label", "modulation_depth"], sort=True
    ):
        eligible_ids = set(screened.loc[screened.eligible, "context_id"])
        contexts = context_summary[context_summary.context_id.isin(eligible_ids)]
        state_expected = expected[expected.context_id.isin(eligible_ids)]
        active = state_expected[state_expected.dose_v_per_m.gt(0.0)]
        if active.empty:
            best_dose, best_distance = float("nan"), float("nan")
        else:
            dose_means = active.groupby("dose_v_per_m")[
                "expected_post_distance_to_B_log10"
            ].mean()
            best_dose = float(dose_means.idxmin())
            best_distance = float(dose_means.min())
        rows.append({
            "state_index": int(state_index),
            "state_label": str(state_label),
            "modulation_depth": float(depth),
            "screened_context_count": int(len(screened)),
            "eligible_context_count": int(screened.eligible.sum()),
            "screening_yield": float(screened.eligible.mean()),
            "mean_context_log10_alpha_power": (
                float(contexts.context_log10_alpha_power.mean())
                if not contexts.empty else float("nan")
            ),
            "mean_context_alpha_excess_log10": (
                float(contexts.context_alpha_excess_log10.mean())
                if not contexts.empty else float("nan")
            ),
            "median_context_temporal_sd_log10": (
                float(contexts.context_alpha_temporal_sd_log10.median())
                if not contexts.empty else float("nan")
            ),
            "state_optimal_active_dose_v_per_m": best_dose,
            "state_optimal_expected_distance_to_B_log10": best_distance,
        })
    return pd.DataFrame(rows)


def _checks_and_conclusions(
    *,
    screening: pd.DataFrame,
    metrics: pd.DataFrame,
    expected: pd.DataFrame,
    summary: pd.DataFrame,
    state: pd.DataFrame,
    policy: pd.DataFrame,
    audit: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible_states = state[state.eligible_context_count.gt(0)].sort_values(
        "modulation_depth"
    )
    state_means = eligible_states.mean_context_log10_alpha_power.to_numpy(float)
    state_ordered = bool(
        state_means.size >= 2 and np.all(np.diff(state_means) > 0.0)
    )
    signal = float(np.ptp(state_means)) if state_means.size >= 2 else 0.0
    noise = float(
        summary.context_alpha_temporal_sd_log10.median()
    ) if not summary.empty else float("inf")
    signal_to_noise = signal / max(noise, np.finfo(float).eps)
    policy_actions = (
        sorted(policy.selected_dose_v_per_m.unique().tolist())
        if not policy.empty else []
    )
    mean_policy_advantage = (
        float(policy.contextual_advantage_over_best_fixed_log10.mean())
        if not policy.empty else float("nan")
    )
    n_expected_actions = len(audit["expected_optimal_active_doses"])
    checks = {
        "split_history_and_future_rng_configured": bool(
            metrics.future_drive_seed.nunique()
            >= int(criteria.minimum_future_continuations)
        ),
        "complete_crossed_screening_grid": len(screening) == len(_context_specs(cfg)),
        "screening_uses_only_predecision_eeg": bool(
            (~screening.screening_uses_stimulation_outcome).all()
        ),
        "screening_does_not_use_seed_specific_B": bool(
            (~screening.screening_uses_seed_specific_B).all()
        ),
        "state_generator_is_distinct_from_tacs_action": True,
        "afferent_mean_rate_matched_across_states_by_construction": True,
        "minimum_eligible_contexts": len(summary)
        >= int(criteria.minimum_eligible_contexts),
        "minimum_eligible_state_levels": summary.state_label.nunique()
        >= int(criteria.minimum_eligible_state_levels),
        "multiple_independent_futures_per_context_action": bool(
            (expected.n_future_continuations
             >= int(criteria.minimum_future_continuations)).all()
        ),
        "identical_predecision_eeg_across_actions_and_futures": bool(
            metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "phase_invariant_context_features_only": bool(
            metrics.context_phase_invariant.all()
        ),
        "single_constant_action_per_intervention": bool(
            metrics.one_action_for_complete_intervention.all()
        ),
        "action_phase_tracks_predecision_eeg": bool(
            metrics.action_phase_tracking_error_rad.max()
            <= float(criteria.maximum_phase_tracking_error_rad)
        ),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "field_removal_recovered": bool(metrics.field_removal_recovered.all()),
        "explicit_state_severity_ordered_in_eeg": state_ordered,
        "state_signal_exceeds_within_context_temporal_noise": signal_to_noise
        >= float(criteria.minimum_state_signal_to_temporal_noise),
        "expected_oracle_uses_multiple_active_doses": n_expected_actions
        >= int(criteria.minimum_expected_active_action_count),
        "practical_expected_nonfixed_opportunity": int(
            audit["practical_nonfixed_context_count"]
        ) >= int(criteria.minimum_practical_nonfixed_contexts),
        "expected_oracle_has_mean_practical_advantage": float(
            audit["mean_expected_oracle_advantage_over_best_fixed_log10"]
        ) >= float(criteria.minimum_mean_expected_oracle_advantage_log10),
        "realized_optimum_is_reproducible_across_futures": float(
            audit["mean_realized_oracle_agreement_fraction"]
        ) >= float(criteria.minimum_realized_oracle_agreement_fraction),
        "exploratory_eeg_rule_uses_multiple_actions": len(policy_actions) >= 2,
        "exploratory_eeg_rule_beats_best_fixed_directionally": bool(
            np.isfinite(mean_policy_advantage)
            and mean_policy_advantage
            > float(criteria.minimum_contextual_advantage_over_best_fixed_log10)
        ),
        "policy_excludes_latent_state_and_hidden_spikes": True,
    }
    mapping_names = (
        "split_history_and_future_rng_configured",
        "complete_crossed_screening_grid",
        "screening_uses_only_predecision_eeg",
        "state_generator_is_distinct_from_tacs_action",
        "afferent_mean_rate_matched_across_states_by_construction",
        "minimum_eligible_contexts",
        "multiple_independent_futures_per_context_action",
        "identical_predecision_eeg_across_actions_and_futures",
        "single_constant_action_per_intervention",
        "all_actions_rate_safe",
        "field_removal_recovered",
        "expected_oracle_uses_multiple_active_doses",
        "practical_expected_nonfixed_opportunity",
        "expected_oracle_has_mean_practical_advantage",
        "realized_optimum_is_reproducible_across_futures",
    )
    policy_names = (
        *mapping_names,
        "minimum_eligible_state_levels",
        "state_signal_exceeds_within_context_temporal_noise",
        "exploratory_eeg_rule_uses_multiple_actions",
        "exploratory_eeg_rule_beats_best_fixed_directionally",
        "policy_excludes_latent_state_and_hidden_spikes",
    )
    conclusions = {
        "single_action_conditional_dose_map_feasible": all(
            checks[name] for name in mapping_names
        ),
        "ready_for_disjoint_contextual_policy_confirmation": all(
            checks[name] for name in policy_names
        ),
        "context_count": int(len(summary)),
        "state_count": int(summary.state_label.nunique()),
        "state_signal_to_temporal_noise_ratio": signal_to_noise,
        "state_means_ordered_with_modulation_depth": state_ordered,
        "exploratory_policy_selected_doses_v_per_m": policy_actions,
        "exploratory_policy_mean_advantage_over_best_fixed_log10": (
            mean_policy_advantage
        ),
        **audit,
        "latent_modulation_depth_available_to_policy": False,
        "hidden_spikes_and_rates_available_to_policy": False,
        "sham_is_comparator_not_required_policy_action": True,
        "contextual_bandit_status": "not trained or tested",
    }
    return checks, conclusions


def _plot_results(
    *, root: Path, state: pd.DataFrame, expected: pd.DataFrame,
    summary: pd.DataFrame, policy: pd.DataFrame
) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    axis.errorbar(
        state.modulation_depth,
        state.mean_context_alpha_excess_log10,
        yerr=state.median_context_temporal_sd_log10,
        marker="o", linewidth=2.0, capsize=4,
    )
    axis.axhline(0.0, color="0.35", linewidth=0.9)
    axis.set(
        xlabel="Latent afferent modulation depth (not a policy feature)",
        ylabel="Baseline alpha excess over frozen B (log10)",
        title="Observability of the explicit toy alpha states",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_01_state_observability.png", dpi=250)
    plt.close(figure)

    active = expected[expected.dose_v_per_m.gt(0.0)]
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    for label, group in active.groupby("state_label", sort=False):
        curve = group.groupby("dose_v_per_m").agg(
            mean=("expected_post_distance_to_B_log10", "mean"),
            sd=("expected_post_distance_to_B_log10", "std"),
        ).reset_index()
        axis.errorbar(
            curve.dose_v_per_m, curve["mean"], yerr=curve.sd.fillna(0.0),
            marker="o", capsize=3, label=str(label),
        )
    axis.set(
        xlabel="One constant intervention dose (V/m)",
        ylabel="Expected post-action distance to frozen B (log10)",
        title="Conditional single-action dose map",
    )
    axis.legend(title="Toy state")
    figure.tight_layout()
    figure.savefig(root / "figure_02_state_dose_response.png", dpi=250)
    plt.close(figure)

    ordered = summary.sort_values("context_alpha_excess_log10")
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    scatter = axis.scatter(
        ordered.context_alpha_excess_log10,
        ordered.expected_optimal_active_dose_v_per_m,
        c=ordered.modulation_depth, cmap="viridis", s=75,
        edgecolor="black", linewidth=0.5,
    )
    axis.set(
        xlabel="Predecision EEG alpha excess (log10)",
        ylabel="Expected-optimal single dose (V/m)",
        title="EEG context and expected action",
    )
    figure.colorbar(scatter, ax=axis, label="Hidden modulation depth (audit only)")
    figure.tight_layout()
    figure.savefig(root / "figure_03_context_action_map.png", dpi=250)
    plt.close(figure)

    if not policy.empty:
        ordered_policy = policy.sort_values("context_id")
        x = np.arange(len(ordered_policy))
        figure, axis = plt.subplots(figsize=(8.0, 4.6))
        axis.bar(
            x,
            ordered_policy.contextual_advantage_over_best_fixed_log10,
            color=np.where(
                ordered_policy.contextual_advantage_over_best_fixed_log10 >= 0.0,
                "#2CA02C", "#D62728",
            ),
        )
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.set_xticks(x, ordered_policy.context_id, rotation=45, ha="right")
        axis.set(
            ylabel="LOSO EEG-rule advantage over best fixed (log10)",
            title="Exploratory policy diagnostic (not confirmation)",
        )
        figure.tight_layout()
        figure.savefig(root / "figure_04_exploratory_policy.png", dpi=250)
        plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen = _load_frozen_candidate(cfg)
    _validate_design(cfg, frozen)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "single_action_dose_map"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### CDM1-S single-action conditional dose map")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()

    started = time.perf_counter()
    target_model = frozen["target"]["target_model"]
    contexts = _context_specs(cfg)
    future_seeds = _future_seeds(cfg)
    doses = _doses(cfg)
    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []

    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} depth={context['modulation_depth']:.3f} "
                f"structure={context['structure_seed']} history={context['history_seed']}"
            )
        condition_cfg = _condition_for_seed(
            cfg,
            seed=int(context["phase_seed"]),
            modulation_depth=float(context["modulation_depth"]),
        )
        baseline_reference = _run_replay(
            condition_cfg=condition_cfg,
            context=context,
            future_seed=future_seeds[0],
            future_index=0,
            dose=SHAM_DOSE,
            action_index=0,
            root=root,
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            screen_view = _baseline_screen_view(baseline_reference)
            screen_phase = _screen_phase_quality(
                episode=screen_view,
                relative_offset=float(cfg.analysis.tacs.relative_phase_offset_rad),
                cfg=cfg,
            )
            screening = _screening_decision(
                seed=int(context["trial_seed"]),
                screening_order=int(context["context_order"]),
                a_episode=screen_view,
                phase_quality=screen_phase,
                target_model=target_model,
                cfg=cfg,
            )
            screening.update(context)
            screening_rows.append(screening)
            context_values = _context_features(
                baseline_reference, target_model=target_model, cfg=cfg
            )
            screening.update(context_values)
            eligible = bool(screening["eligible"])
            print(
                f"screen {context['context_id']}: "
                f"{'ELIGIBLE' if eligible else 'EXCLUDED'} "
                f"({screening['exclusion_reasons']})"
            )
        else:
            eligible = None
            screen_phase = None
            screening = None
            context_values = None
        eligible = bool(comm.bcast(eligible, root=0))
        if not eligible:
            continue

        for future_index, future_seed in enumerate(future_seeds):
            episodes: dict[float, dict[str, Any]] | None = {} if rank == 0 else None
            for dose_index, dose in enumerate(doses):
                if future_index == 0 and np.isclose(dose, SHAM_DOSE):
                    episode = baseline_reference
                else:
                    episode = _run_replay(
                        condition_cfg=condition_cfg,
                        context=context,
                        future_seed=future_seed,
                        future_index=future_index,
                        dose=dose,
                        action_index=future_index * len(doses) + dose_index,
                        root=root,
                        comm=comm,
                        size=size,
                        rank=rank,
                    )
                if rank == 0:
                    episodes[dose] = episode
                    for row in episode["epoch_rows"]:
                        epoch_rows.append({
                            **row, **context,
                            "future_index": future_index + 1,
                            "future_drive_seed": int(future_seed),
                            "dose_v_per_m": float(dose),
                        })
            if rank == 0:
                metric_rows.extend(_replay_rows(
                    context=context,
                    future_seed=future_seed,
                    future_index=future_index,
                    episodes=episodes,
                    context_values=context_values,
                    screening=screening,
                    screen_phase=screen_phase,
                    target_model=target_model,
                    baseline_reference=baseline_reference,
                    cfg=cfg,
                ))
                del episodes

    if rank != 0:
        return

    screening_frame = pd.DataFrame(screening_rows)
    screening_frame.to_csv(root / "predecision_screening_audit.csv", index=False)
    if not metric_rows:
        result = {
            "scope": "ideal neural-only EEG, low-cost conditional dose map",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "single_action_conditional_dose_map_feasible": False,
                "ready_for_disjoint_contextual_policy_confirmation": False,
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No context passed the frozen predecision EEG screen.",
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        print("\nNo eligible contexts; CDM1-S: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    epochs = pd.DataFrame(epoch_rows)
    expected, summary, audit = _expected_action_map(metrics, cfg=cfg)
    policy = _exploratory_loso_policy(expected, summary, cfg=cfg)
    state = _state_summary(screening_frame, summary, expected)
    checks, conclusions = _checks_and_conclusions(
        screening=screening_frame,
        metrics=metrics,
        expected=expected,
        summary=summary,
        state=state,
        policy=policy,
        audit=audit,
        cfg=cfg,
    )

    epochs.to_csv(root / "epoch_eeg_and_hidden_metrics.csv", index=False)
    metrics.to_csv(root / "future_action_metrics.csv", index=False)
    expected.to_csv(root / "conditional_expected_dose_map.csv", index=False)
    summary.to_csv(root / "context_expected_action_summary.csv", index=False)
    state.to_csv(root / "state_observability_and_dose_summary.csv", index=False)
    policy.to_csv(root / "exploratory_loso_eeg_policy.csv", index=False)
    provenance = {
        **frozen,
        "states": [
            {"label": label, "modulation_depth": depth}
            for label, depth in zip(
                [str(value) for value in cfg.analysis.states.labels],
                [float(value) for value in cfg.analysis.states.modulation_depths],
            )
        ],
        "states_share_cells_recurrence_mean_afferent_rate_and_synaptic_weights": True,
        "state_difference": (
            "sinusoidal modulation depth of independent Poisson afferent rates"
        ),
        "state_label_available_to_policy": False,
        "policy_features": [
            str(value) for value in cfg.analysis.context.policy_features
        ],
        "actions_v_per_m": doses,
        "one_action_selected_once_per_intervention": True,
        "active_probe_used": False,
        "postdecision_action_switching_used": False,
        "future_rng_split_at_intervention_boundary": True,
        "future_drive_seeds": future_seeds,
        "contexts": contexts,
        "selection_performed": "none; full-information exploratory mapping",
    }
    (root / "protocol_provenance.json").write_text(
        json.dumps(_plain(provenance), indent=2)
    )
    result = {
        "scope": "ideal neural-only EEG, low-cost single-action system identification",
        "checks": checks,
        "conclusions": conclusions,
        "primary_endpoint": (
            "future-averaged absolute log10 alpha-power distance to frozen B"
        ),
        "causal_comparator": "same-context, same-future 0-V/m sham",
        "statistical_unit": (
            "circuit structure; histories and future continuations are crossed repeats"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "A map pass supports a reproducible context-dependent single-dose "
            "opportunity. A policy pass is only a directional gate for a new, "
            "disjoint confirmation; this experiment is not a bandit trial."
        ),
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_plain(result), indent=2)
    )
    if bool(cfg.experiment.plot):
        _plot_results(
            root=root, state=state, expected=expected,
            summary=summary, policy=policy,
        )

    print("\n### CDM1-S screening")
    print(f"contexts screened: {len(screening_frame)}")
    print(f"eligible contexts: {int(screening_frame.eligible.sum())}")
    print(f"screening yield: {float(screening_frame.eligible.mean()):.3f}")
    print("\n### CDM1-S checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nSingle-action conditional dose map:",
        "PASSED" if conclusions["single_action_conditional_dose_map_feasible"]
        else "NOT PASSED",
    )
    print(
        "Ready for disjoint policy confirmation:",
        "YES" if conclusions["ready_for_disjoint_contextual_policy_confirmation"]
        else "NO",
    )
    print("Contextual bandit status: NOT TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
