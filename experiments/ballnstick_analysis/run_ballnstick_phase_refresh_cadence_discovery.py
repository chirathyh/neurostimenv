"""H4-BW2 targeted causal phase-refresh cadence discovery.

H4-BW found that the 0.5-s/125-ms controller had the best expected EEG
endpoint but failed future reliability and a window-dependent phase-error
gate.  This disjoint follow-up adds the missing 1-s/125-ms controller.  It
compares that controller with sham, one-time initialization, the frozen
1-s/250-ms controller, and the prior 0.5-s/125-ms exploratory controller.

The intervention is lengthened so multi-step phase drift and feedback can be
observed.  A new homogeneous-B calibration supplies a duration-matched
eight-second outcome target while the prior screening target remains frozen.
Every controller is evaluated by the same one-second phase auditor on common
250-ms boundaries.  Reliability uses the within-context standard deviation of
the paired controller-minus-one-time effect, not marginal outcome-SD ratios.
This is discovery, not H4 confirmation, a bandit, or a clinical model.
"""

from __future__ import annotations

import copy
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
    _epoch_row,
    _run_condition,
    _sham,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _episode_feature,
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (  # noqa: E402
    HIGH,
    LOW,
    _context_features,
    _fit_B_target,
    _future_seed,
    _reference_cfg,
    _reference_seeds,
    _run_context_specs,
    _with_diffusion_state,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (  # noqa: E402
    ONE_TIME,
    SHAM,
    _controller_modes,
    _metric_rows,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_bandwidth_discovery import (  # noqa: E402
    CURRENT,
    FAST,
    _augment_metric_rows,
    _common_initialization,
    _json_ready,
    _load_sources as _load_h4bw_upstream,
    _profile,
    _run_controller,
    _sha256,
)


NEW = "refresh_1000ms_125ms"
EXPECTED_MODES = [SHAM, ONE_TIME, CURRENT, NEW, FAST]


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    """Verify the upstream chain and freeze the failed H4-BW discovery."""
    sources = _load_h4bw_upstream(cfg)
    root = Path(to_absolute_path(str(cfg.analysis.source_h4bw.result_dir)))
    files = {
        "conclusion": root / "experiment_conclusion.json",
        "provenance": root / "protocol_and_provenance.json",
        "screening": root / "prospective_screening.csv",
        "metrics": root / "context_controller_future_metrics.csv",
        "updates": root / "causal_phase_updates.csv",
        "summary": root / "controller_selection_summary.csv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen H4-BW sources: {missing}")
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {
        name: str(cfg.analysis.source_h4bw.expected_sha256[name]) for name in files
    }
    if observed != expected:
        raise RuntimeError(
            f"H4-BW source hash mismatch: expected={expected}, observed={observed}"
        )
    conclusion = json.loads(files["conclusion"].read_text())
    failed = [
        name for name, passed in conclusion["checks"].items() if not bool(passed)
    ]
    expected_failure = "at_least_one_short_history_candidate_passes_frozen_gate"
    if failed != [expected_failure]:
        raise RuntimeError(
            "H4-BW2 requires the frozen H4-BW run to have failed only candidate "
            f"selection; observed failed checks={failed}."
        )
    if conclusion["conclusions"]["selected_controller"] is not None:
        raise RuntimeError("Frozen H4-BW unexpectedly selected a controller.")
    summary = pd.read_csv(files["summary"])
    if bool(summary.passes_bandwidth_gate.astype(bool).any()):
        raise RuntimeError("Frozen H4-BW summary unexpectedly contains a passing arm.")
    for table_path in (files["screening"], files["metrics"]):
        table = pd.read_csv(table_path)
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed",
        ):
            if column in table:
                sources["source_seed_union"].update(
                    table[column].dropna().astype(int).tolist()
                )
    sources["roots"]["h4bw"] = str(root)
    sources["hashes"]["h4bw"] = observed
    sources["H4BW_failed_controller_selection"] = True
    return sources


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H4-BW2 requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H4-BW2 may not change recurrent inhibition.")
    if [float(value) for value in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H4-BW2 freezes the 9/11-Hz generator grid.")
    levels = [
        (str(value.label), float(value.diffusion_rad2_per_s))
        for value in cfg.analysis.states.phase_diffusion_levels
    ]
    if levels != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("H4-BW2 freezes D to 0.5 and 2 rad^2/s.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H4-BW2 freezes modulation depth to 0.04.")
    if _controller_modes(cfg) != EXPECTED_MODES:
        raise ValueError(f"H4-BW2 controller order must be {EXPECTED_MODES}.")
    expected_profiles = {
        SHAM: (False, 1000.0, 250.0),
        ONE_TIME: (False, 1000.0, 250.0),
        CURRENT: (True, 1000.0, 250.0),
        NEW: (True, 1000.0, 125.0),
        FAST: (True, 500.0, 125.0),
    }
    for mode, expected in expected_profiles.items():
        profile = _profile(cfg, mode)
        observed = (
            profile["adaptive"], profile["history_ms"],
            profile["update_interval_ms"],
        )
        if observed != expected:
            raise ValueError(
                f"H4-BW2 profile {mode} changed: expected={expected}, observed={observed}."
            )
    if [str(value) for value in cfg.analysis.selection.candidates] != [NEW, FAST]:
        raise ValueError("H4-BW2 freezes the 125-ms candidate pair.")
    if str(cfg.analysis.selection.benchmark_controller) != CURRENT:
        raise ValueError("H4-BW2 benchmark must remain the 1-s/250-ms controller.")
    amplitude = float(cfg.analysis.actions.amplitude_v_per_m)
    if not np.isclose(amplitude, 0.2):
        raise ValueError("H4-BW2 freezes every active controller to 0.2 V/m.")
    if not np.isclose(
        _wrap_phase(float(cfg.analysis.tacs.relative_phase_offset_rad)), np.pi
    ):
        raise ValueError("H4-BW2 freezes the pi-relative phase target.")
    for name, expected in (
        ("initialization_history_ms", 1000.0),
        ("correction_horizon_ms", 250.0),
        ("common_audit_history_ms", 1000.0),
        ("common_audit_interval_ms", 250.0),
        ("maximum_frequency_correction_hz", 2.0),
    ):
        if not np.isclose(float(cfg.analysis.tacs[name]), expected):
            raise ValueError(f"H4-BW2 freezes {name}={expected:g}.")
    if not np.isclose(float(cfg.env.simulation.obs_win_len), 1000.0):
        raise ValueError("H4-BW2 requires 1000-ms outer online windows.")
    timeline = cfg.analysis.timeline
    minimum_baseline = 4 if bool(cfg.analysis.smoke_test) else 12
    if int(timeline.baseline_steps) < minimum_baseline:
        raise ValueError(f"H4-BW2 requires at least {minimum_baseline} baseline seconds.")
    stimulation_ms = int(timeline.stimulation_steps) * float(
        cfg.env.simulation.obs_win_len
    )
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    endpoint_ms = stimulation_ms - 2.0 * trim_ms
    if trim_ms < float(timeline.block_ramp_ms) or endpoint_ms <= 0:
        raise ValueError("H4-BW2 trim must contain both ramps and leave EEG.")
    if not bool(cfg.analysis.smoke_test) and not np.isclose(endpoint_ms, 8000.0):
        raise ValueError("Full H4-BW2 freezes the eight-second EEG endpoint.")
    if not np.isclose(endpoint_ms / 1000.0, round(endpoint_ms / 1000.0)):
        raise ValueError("H4-BW2 endpoint must split into one-second windows.")
    if not bool(cfg.analysis.smoke_test):
        if int(cfg.analysis.reference_calibration.n_seeds) < 12:
            raise ValueError("Full H4-BW2 requires 12 duration-matched B references.")
        if int(cfg.analysis.crossed_design.n_structure_seeds) < 3:
            raise ValueError("Full H4-BW2 requires three structures.")
        if int(cfg.analysis.crossed_design.n_future_continuations) < 6:
            raise ValueError("Full H4-BW2 requires six futures per controller.")

    contexts = _run_context_specs(cfg)
    references = set(_reference_seeds(cfg))
    namespaces = [
        references,
        {int(value["structure_seed"]) for value in contexts},
        {int(value["history_seed"]) for value in contexts},
        {int(value["phase_seed"]) for value in contexts},
        {int(value["trial_seed"]) for value in contexts},
        {
            _future_seed(cfg, context, future_index)
            for context in contexts
            for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations))
        },
    ]
    if any(not values for values in namespaces):
        raise ValueError("Every H4-BW2 seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H4-BW2 seed namespaces overlap.")
    if set().union(*namespaces).intersection(sources["source_seed_union"]):
        raise ValueError("H4-BW2 seeds overlap frozen upstream experiments.")
    if max(references | namespaces[1]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H4-BW2 structure seed exceeds the uint32 mapping range.")


def _duration_matched_target(
    calibration: pd.DataFrame, sources: dict[str, Any], cfg: DictConfig,
) -> dict[str, Any]:
    """Keep the frozen screen and replace only the duration-matched outcome."""
    fitted = _fit_B_target(calibration, cfg)
    target = copy.deepcopy(sources["target"])
    target["outcome"] = fitted["outcome"]
    target["reference_E_firing_rate_hz"] = fitted["reference_E_firing_rate_hz"]
    target["reference_I_firing_rate_hz"] = fitted["reference_I_firing_rate_hz"]
    target["screening_target_remains_frozen_from_D1R"] = True
    target["outcome_target_calibrated_before_active_outcomes"] = True
    target["outcome_duration_s"] = float(
        int(cfg.analysis.timeline.stimulation_steps)
        * float(cfg.env.simulation.obs_win_len)
        - 2.0 * float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    ) / 1000.0
    return target


def _augment_common_audit(
    rows: list[dict[str, Any]], episodes: dict[str, dict[str, Any]], cfg: DictConfig,
) -> None:
    """Add one common phase metric on common 250-ms boundaries to each row."""
    by_mode = {str(row["controller_mode"]): row for row in rows}
    interval = float(cfg.analysis.tacs.common_audit_interval_ms)
    threshold = float(cfg.analysis.screening.minimum_recent_resultant_to_rms)
    for mode, episode in episodes.items():
        updates = pd.DataFrame(episode["simulation"]["phase_updates"])
        start = float(updates.boundary_ms.min())
        elapsed = updates.boundary_ms.to_numpy(float) - start
        common = updates[
            (elapsed > 1.0e-9)
            & np.isclose(np.mod(elapsed, interval), 0.0, atol=1.0e-8)
        ]
        if common.empty:
            raise RuntimeError(f"No common phase-audit boundaries for {mode}.")
        row = by_mode[mode]
        row.update({
            "common_phase_audit_count": int(len(common)),
            "mean_abs_common_phase_error_rad": float(
                common.common_audit_phase_error_before_correction_rad.abs().mean()
            ),
            "common_phase_estimate_actionable_fraction": float(np.mean(
                common.common_audit_resultant_to_rms.to_numpy(float) >= threshold
            )),
        })


def _expected_map(metrics: pd.DataFrame) -> pd.DataFrame:
    group = [
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "diffusion_rad2_per_s", "context_C1", "controller_mode",
    ]
    return (
        metrics.groupby(group, as_index=False)
        .agg(
            n_futures=("future_index", "nunique"),
            expected_post_distance_to_B_log10=("post_distance_to_B_log10", "mean"),
            future_sd_post_distance_log10=("post_distance_to_B_log10", "std"),
            expected_improvement_vs_sham_log10=(
                "causal_distance_improvement_vs_sham_log10", "mean"
            ),
            all_rate_safe=("rate_safe", "all"),
            all_field_removal_recovered=("field_removal_recovered", "all"),
            mean_abs_controller_innovation_rad=(
                "mean_abs_phase_error_before_correction_rad", "mean"
            ),
            mean_abs_common_phase_error_rad=(
                "mean_abs_common_phase_error_rad", "mean"
            ),
            common_phase_estimate_actionable_fraction=(
                "common_phase_estimate_actionable_fraction", "mean"
            ),
            correction_saturation_fraction=(
                "frequency_correction_saturation_fraction", "mean"
            ),
            maximum_field_boundary_discontinuity_v_per_m=(
                "maximum_field_boundary_discontinuity_v_per_m", "max"
            ),
        )
        .sort_values(group)
        .reset_index(drop=True)
    )


def _comparison_tables(
    expected: pd.DataFrame, metrics: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = [
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "diffusion_rad2_per_s", "context_C1",
    ]
    expected_lookup = {
        (str(row.context_id), str(row.controller_mode)): row
        for row in expected.itertuples()
    }
    realized = metrics[metrics.controller_mode.ne(SHAM)].pivot(
        index=["context_id", "future_index"], columns="controller_mode",
        values="post_distance_to_B_log10",
    ).reset_index()
    comparison_rows: list[dict[str, Any]] = []
    for context_values in expected[keys].drop_duplicates().itertuples(index=False):
        context = dict(zip(keys, context_values))
        context_id = str(context["context_id"])
        future = realized[realized.context_id.eq(context_id)]
        one = expected_lookup[(context_id, ONE_TIME)]
        current = expected_lookup[(context_id, CURRENT)]
        current_effect = future[ONE_TIME].to_numpy(float) - future[CURRENT].to_numpy(float)
        current_effect_sd = float(np.std(current_effect, ddof=1))
        for mode in (CURRENT, NEW, FAST):
            candidate = expected_lookup[(context_id, mode)]
            effect = future[ONE_TIME].to_numpy(float) - future[mode].to_numpy(float)
            comparison_rows.append({
                **context,
                "controller_mode": mode,
                "selection_candidate": mode in (NEW, FAST),
                "advantage_over_one_time_log10": float(
                    one.expected_post_distance_to_B_log10
                    - candidate.expected_post_distance_to_B_log10
                ),
                "advantage_over_current_log10": float(
                    current.expected_post_distance_to_B_log10
                    - candidate.expected_post_distance_to_B_log10
                ),
                "realized_candidate_win_fraction": float(np.mean(effect > 0.0)),
                "paired_effect_sd_log10": float(np.std(effect, ddof=1)),
                "current_paired_effect_sd_log10": current_effect_sd,
                "mean_abs_common_phase_error_rad": float(
                    candidate.mean_abs_common_phase_error_rad
                ),
                "current_mean_abs_common_phase_error_rad": float(
                    current.mean_abs_common_phase_error_rad
                ),
                "common_phase_estimate_actionable_fraction": float(
                    candidate.common_phase_estimate_actionable_fraction
                ),
                "mean_abs_controller_innovation_rad": float(
                    candidate.mean_abs_controller_innovation_rad
                ),
                "correction_saturation_fraction": float(
                    candidate.correction_saturation_fraction
                ),
            })
    comparison = pd.DataFrame(comparison_rows)
    structure = (
        comparison.groupby(["controller_mode", "structure_seed"], as_index=False)
        .agg(
            context_count=("context_id", "nunique"),
            mean_advantage_over_one_time_log10=(
                "advantage_over_one_time_log10", "mean"
            ),
            mean_advantage_over_current_log10=(
                "advantage_over_current_log10", "mean"
            ),
            mean_realized_candidate_win_fraction=(
                "realized_candidate_win_fraction", "mean"
            ),
        )
    )
    criteria = cfg.analysis.criteria
    summaries: list[dict[str, Any]] = []
    for mode in (CURRENT, NEW, FAST):
        context = comparison[comparison.controller_mode.eq(mode)]
        structures = structure[structure.controller_mode.eq(mode)]
        diffusion = context.groupby("label").advantage_over_one_time_log10.mean()
        mean_advantage = float(
            structures.mean_advantage_over_one_time_log10.mean()
        )
        mean_vs_current = float(
            structures.mean_advantage_over_current_log10.mean()
        )
        positive_fraction = float(np.mean(
            structures.mean_advantage_over_one_time_log10 > 0.0
        ))
        candidate_sd = float(context.paired_effect_sd_log10.mean())
        current_sd = float(context.current_paired_effect_sd_log10.mean())
        sd_ratio = candidate_sd / max(current_sd, np.finfo(float).tiny)
        common_error = float(context.mean_abs_common_phase_error_rad.mean())
        current_error = float(
            context.current_mean_abs_common_phase_error_rad.mean()
        )
        actionable = float(
            context.common_phase_estimate_actionable_fraction.mean()
        )
        gates = {
            "gate_practical_advantage_over_one_time": mean_advantage
            >= float(criteria.practical_advantage_over_one_time_log10),
            "gate_positive_across_structures": positive_fraction
            >= float(criteria.minimum_positive_structure_fraction),
            "gate_nonadverse_in_both_diffusion_levels": bool(
                len(diffusion) == 2 and (diffusion >= 0.0).all()
            ),
            "gate_realized_winner_reproducible": float(
                context.realized_candidate_win_fraction.mean()
            ) >= float(criteria.minimum_realized_candidate_win_fraction),
            "gate_paired_effect_variability_not_increased": sd_ratio
            <= float(criteria.maximum_paired_effect_sd_ratio_to_current) + 1.0e-12,
            "gate_noninferior_to_current_endpoint": mean_vs_current
            >= -float(criteria.current_tracker_noninferiority_margin_log10),
            "gate_common_phase_error_lower_than_current": common_error
            < current_error,
            "gate_common_phase_estimate_actionable": actionable
            >= float(criteria.minimum_common_phase_estimate_actionable_fraction),
        }
        summaries.append({
            "controller_mode": mode,
            "selection_candidate": mode in (NEW, FAST),
            "mean_advantage_over_one_time_log10": mean_advantage,
            "mean_advantage_over_current_log10": mean_vs_current,
            "positive_structure_fraction": positive_fraction,
            "minimum_diffusion_advantage_log10": float(diffusion.min()),
            "mean_realized_candidate_win_fraction": float(
                context.realized_candidate_win_fraction.mean()
            ),
            "mean_paired_effect_sd_log10": candidate_sd,
            "paired_effect_sd_ratio_to_current": sd_ratio,
            "mean_abs_common_phase_error_rad": common_error,
            "current_mean_abs_common_phase_error_rad": current_error,
            "mean_abs_controller_innovation_rad": float(
                context.mean_abs_controller_innovation_rad.mean()
            ),
            "mean_common_phase_estimate_actionable_fraction": actionable,
            "mean_correction_saturation_fraction": float(
                context.correction_saturation_fraction.mean()
            ),
            **gates,
            "passes_cadence_gate": bool(
                mode in (NEW, FAST) and all(gates.values())
            ),
        })
    summary = pd.DataFrame(summaries)
    selection = _select_controller(summary, cfg)
    return comparison, structure, summary, selection


def _select_controller(summary: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    passing = summary[
        summary.selection_candidate.astype(bool)
        & summary.passes_cadence_gate.astype(bool)
    ].copy()
    if passing.empty:
        return {
            "selected_controller": None,
            "cadence_candidate_found": False,
            "selection_rule": (
                "pass all frozen gates; maximize structure-mean advantage over "
                "one-time; within 0.01 log10 prefer the one-second estimator"
            ),
        }
    best = float(passing.mean_advantage_over_one_time_log10.max())
    margin = float(cfg.analysis.selection.practical_tie_log10)
    tied = set(passing[
        passing.mean_advantage_over_one_time_log10 >= best - margin
    ].controller_mode.astype(str))
    preference = [str(value) for value in cfg.analysis.selection.tie_preference]
    selected = next(mode for mode in preference if mode in tied)
    row = passing[passing.controller_mode.eq(selected)].iloc[0]
    return {
        "selected_controller": selected,
        "cadence_candidate_found": True,
        "selected_profile": _profile(cfg, selected),
        "selected_mean_advantage_over_one_time_log10": float(
            row.mean_advantage_over_one_time_log10
        ),
        "selected_mean_realized_candidate_win_fraction": float(
            row.mean_realized_candidate_win_fraction
        ),
        "selection_rule": (
            "pass all frozen gates; maximize structure-mean advantage over "
            "one-time; within 0.01 log10 prefer refresh_1000ms_125ms"
        ),
    }


def _temporal_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    expected = trajectories.groupby([
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "controller_mode", "analysis_window_index",
    ], as_index=False).distance_to_B_log10.mean()
    pivot = expected.pivot(
        index=[
            "context_id", "structure_seed", "hidden_frequency_hz", "label",
            "analysis_window_index",
        ], columns="controller_mode", values="distance_to_B_log10",
    ).reset_index()
    rows = []
    for mode in (CURRENT, NEW, FAST):
        value = pivot.copy()
        value["advantage_over_one_time_log10"] = value[ONE_TIME] - value[mode]
        for window, group in value.groupby("analysis_window_index"):
            rows.append({
                "controller_mode": mode,
                "analysis_window_index": int(window),
                "mean_advantage_over_one_time_log10": float(
                    group.advantage_over_one_time_log10.mean()
                ),
                "positive_context_fraction": float(np.mean(
                    group.advantage_over_one_time_log10 > 0.0
                )),
            })
    return pd.DataFrame(rows)


def _checks(
    *, calibration: pd.DataFrame, target: dict[str, Any],
    screening: pd.DataFrame, metrics: pd.DataFrame, expected: pd.DataFrame,
    updates: pd.DataFrame, summary: pd.DataFrame, selection: dict[str, Any],
    sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    active = metrics[metrics.controller_mode.ne(SHAM)]
    refreshed = updates[
        updates.controller_mode.isin([CURRENT, NEW, FAST])
        & updates.phase_refresh_applied.astype(bool)
    ]
    common_interval = float(cfg.analysis.tacs.common_audit_interval_ms)
    first_boundary = updates.groupby([
        "context_id", "future_index", "controller_mode"
    ]).boundary_ms.transform("min")
    common_boundary = np.isclose(
        np.mod(updates.boundary_ms - first_boundary, common_interval),
        0.0, atol=1.0e-8,
    )
    common = updates[common_boundary]
    references = {
        "E": float(target["reference_E_firing_rate_hz"]),
        "I": float(target["reference_I_firing_rate_hz"]),
    }
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    rate_matched = all(
        abs(float(getattr(row, f"baseline_{population}_firing_rate_hz"))
            - references[population])
        <= tolerance * max(references[population], np.finfo(float).tiny)
        for row in eligible.itertuples() for population in ("E", "I")
    )
    checks = {
        "source_H4BW_hash_locked_with_no_selected_controller": bool(
            sources["H4BW_failed_controller_selection"]
        ),
        "H4BW2_seed_namespaces_disjoint_from_all_sources": True,
        "duration_matched_B_target_calibrated_before_active_outcomes": bool(
            len(calibration) >= int(criteria.minimum_reference_seeds)
            or bool(cfg.analysis.smoke_test)
        ) and bool(target["outcome_target_calibrated_before_active_outcomes"]),
        "frozen_screening_target_retained": target["screening"]
        == sources["target"]["screening"],
        "eight_second_primary_EEG_endpoint": bool(
            np.isclose(
                float(cfg.analysis.timeline.stimulation_steps)
                * float(cfg.env.simulation.obs_win_len)
                - 2.0 * float(cfg.analysis.timeline.stimulation_analysis_trim_ms),
                8000.0,
            ) or bool(cfg.analysis.smoke_test)
        ),
        "complete_crossed_screening_grid": len(screening) == len(
            _run_context_specs(cfg)
        ),
        "screening_uses_only_predecision_ideal_EEG": bool(
            len(screening) and screening.screen_uses_only_predecision_ideal_EEG.all()
        ),
        "minimum_eligible_contexts": len(eligible)
        >= int(criteria.minimum_eligible_contexts) or bool(cfg.analysis.smoke_test),
        "minimum_independent_structures": eligible.structure_seed.nunique()
        >= int(criteria.minimum_structure_seeds) or bool(cfg.analysis.smoke_test),
        "both_diffusion_levels_and_frequencies_enrolled": bool(
            eligible.label.nunique() == 2 and eligible.hidden_frequency_hz.nunique() == 2
        ) or bool(cfg.analysis.smoke_test),
        "frequency_identified_from_predecision_EEG": bool(
            len(eligible) and eligible.EEG_frequency_selection_correct.mean()
            >= float(criteria.minimum_frequency_detection_accuracy)
        ),
        "complete_controller_grid_for_enrolled_contexts": bool(
            len(expected)
            and expected.groupby("context_id").controller_mode.nunique().min()
            == len(EXPECTED_MODES)
        ),
        "six_independent_postdecision_futures": bool(
            len(expected)
            and expected.n_futures.min() >= int(criteria.minimum_future_continuations)
        ) or bool(cfg.analysis.smoke_test),
        "identical_predecision_EEG_across_controllers_and_futures": bool(
            len(metrics)
            and metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "all_active_arms_use_identical_0p2_V_per_m": bool(
            len(active) and np.allclose(active.amplitude_v_per_m, 0.2)
        ),
        "all_active_arms_share_one_second_initialization": _common_initialization(
            updates
        ),
        "phase_updates_use_only_preceding_EEG": bool(
            len(updates)
            and updates.estimate_is_strictly_causal.all()
            and (updates.estimate_stop_ms - updates.boundary_ms).max()
            <= float(criteria.maximum_causal_timing_error_ms)
        ),
        "common_phase_auditor_uses_one_second_on_250ms_boundaries": bool(
            len(common)
            and np.allclose(common.common_audit_history_ms, 1000.0)
            and common.common_audit_phase_error_before_correction_rad.notna().all()
        ),
        "correction_horizon_fixed_across_refresh_rates": bool(
            len(refreshed)
            and np.allclose(refreshed.correction_horizon_ms, 250.0)
        ),
        "phase_correction_is_frequency_bounded": bool(
            len(refreshed)
            and refreshed.frequency_correction_hz.abs().max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "field_waveform_continuous_across_update_boundaries": bool(
            len(active)
            and active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
        ),
        "all_actions_rate_safe": bool(len(metrics) and metrics.rate_safe.all()),
        "reference_rate_matched": bool(rate_matched),
        "field_removal_recovered": bool(
            len(metrics) and metrics.field_removal_recovered.all()
            and np.allclose(metrics.final_extracellular_residual_mV, 0.0)
        ),
        "at_least_one_125ms_candidate_passes_frozen_gate": bool(
            selection["cadence_candidate_found"]
        ),
        "selection_uses_no_hidden_state_or_spikes": bool(
            len(metrics) and (~metrics.policy_uses_hidden_state_or_spikes.astype(bool)).all()
        ),
    }
    gate = list(checks)
    ready = bool(all(checks[name] for name in gate) and not bool(cfg.analysis.smoke_test))
    return checks, {
        **selection,
        "ready_for_disjoint_12_structure_H4_confirmation": ready,
        "contextual_bandit_status": "NOT TRAINED OR TESTED",
        "claim_scope": "exploratory ideal-neural-EEG cadence/reliability discovery",
        "candidate_summaries": summary.to_dict("records"),
    }


def _plots(
    *, root: Path, expected: pd.DataFrame, summary: pd.DataFrame,
    structure: pd.DataFrame, updates: pd.DataFrame,
    temporal: pd.DataFrame,
) -> None:
    modes = EXPECTED_MODES
    colors = {
        SHAM: "0.55", ONE_TIME: "tab:orange", CURRENT: "tab:blue",
        NEW: "tab:red", FAST: "tab:purple",
    }
    labels = {
        SHAM: "sham", ONE_TIME: "one-time", CURRENT: "1 s / 250 ms",
        NEW: "1 s / 125 ms", FAST: "0.5 s / 125 ms",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, diffusion in zip(axes, (LOW, HIGH)):
        subset = expected[expected.label.eq(diffusion)]
        means = subset.groupby("controller_mode").expected_post_distance_to_B_log10.mean()
        axis.bar(
            np.arange(len(modes)), [means.get(mode, np.nan) for mode in modes],
            color=[colors[mode] for mode in modes],
        )
        axis.set_xticks(np.arange(len(modes)), [labels[x] for x in modes], rotation=25)
        axis.set_title(diffusion.replace("_", " "))
        axis.set_ylabel("Expected 8-s distance to B (log10)")
    figure.suptitle("H4-BW2 phase-refresh cadence discovery")
    figure.tight_layout()
    figure.savefig(root / "figure_01_controller_outcomes.png", dpi=250)
    plt.close(figure)

    common_interval = 250.0
    first = updates.groupby([
        "context_id", "future_index", "controller_mode"
    ]).boundary_ms.transform("min")
    values = updates.copy()
    values["time_s"] = (values.boundary_ms - first) / 1000.0
    values = values[np.isclose(
        np.mod(values.time_s * 1000.0, common_interval), 0.0, atol=1.0e-8
    )]
    phase = values.groupby([
        "controller_mode", "time_s"
    ]).common_audit_phase_error_before_correction_rad.apply(
        lambda x: float(np.mean(np.abs(x)))
    ).reset_index(name="mean_abs_common_phase_error_rad")
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for mode in modes[1:]:
        subset = phase[phase.controller_mode.eq(mode)]
        axis.plot(
            subset.time_s, subset.mean_abs_common_phase_error_rad,
            label=labels[mode], color=colors[mode], linewidth=1.4,
        )
    axis.set(
        xlabel="Time since tACS onset (s)",
        ylabel="Common-auditor |phase error| (rad)",
        title="Like-for-like causal phase audit",
    )
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(root / "figure_02_common_phase_tracking.png", dpi=250)
    plt.close(figure)

    candidates = summary[summary.selection_candidate]
    positions = np.arange(len(candidates))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    fields = [
        ("mean_advantage_over_one_time_log10", 0.01, "Advantage over one-time"),
        ("mean_realized_candidate_win_fraction", 0.75, "Paired future win fraction"),
        ("paired_effect_sd_ratio_to_current", 1.0, "Paired-effect SD ratio"),
    ]
    for axis, (field, threshold, title) in zip(axes, fields):
        axis.bar(
            positions, candidates[field],
            color=[colors[x] for x in candidates.controller_mode],
        )
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(
            positions, [labels[x] for x in candidates.controller_mode], rotation=20
        )
        axis.set_title(title)
    figure.suptitle("Frozen H4-BW2 candidate gates")
    figure.tight_layout()
    figure.savefig(root / "figure_03_candidate_gate.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for mode in (CURRENT, NEW, FAST):
        subset = temporal[temporal.controller_mode.eq(mode)]
        axis.plot(
            subset.analysis_window_index,
            subset.mean_advantage_over_one_time_log10,
            marker="o", label=labels[mode], color=colors[mode],
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(
        xlabel="One-second endpoint window",
        ylabel="Advantage over one-time (log10)",
        title="Accumulation of adaptive advantage",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_04_temporal_advantage.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    for mode in (CURRENT, NEW, FAST):
        subset = structure[structure.controller_mode.eq(mode)]
        axis.scatter(
            np.full(len(subset), modes.index(mode)),
            subset.mean_advantage_over_one_time_log10,
            color=colors[mode], s=45,
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(
        [modes.index(x) for x in (CURRENT, NEW, FAST)],
        [labels[x] for x in (CURRENT, NEW, FAST)],
    )
    axis.set(
        ylabel="Structure-level advantage over one-time (log10)",
        title="Independent-structure consistency",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_05_structure_advantage.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "phase_refresh_cadence_discovery"
    if rank == 0:
        exists = bool(root.exists() and any(root.iterdir()))
    else:
        exists = None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H4-BW2 phase-refresh cadence discovery")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    calibration_rows: list[dict[str, Any]] = []
    for seed in _reference_seeds(cfg):
        if rank == 0:
            print(f"duration-matched B reference seed={seed}")
        reference_cfg = _reference_cfg(cfg, seed)
        episode = _run_condition(
            condition_id="B_homogeneous_reference_8s",
            condition_cfg=reference_cfg,
            action=_sham(reference_cfg, "B_homogeneous_reference_8s"),
            stimulate=False,
            seed=int(seed),
            action_index=0,
            output_dir=root / "reference_B" / str(seed),
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(seed),
            drive_seed=int(seed),
            phase_seed=int(seed),
        )
        if rank == 0:
            baseline, _, _ = _episode_feature(episode, "baseline", cfg)
            outcome, _, _ = _episode_feature(episode, "stimulation", cfg)
            row = _epoch_row(episode, "stimulation")
            calibration_rows.append({
                "seed": int(seed),
                "baseline_log10_alpha_power": float(
                    baseline["log10_alpha_power_8_12_hz"]
                ),
                "outcome_log10_alpha_power": float(
                    outcome["log10_alpha_power_8_12_hz"]
                ),
                "E_firing_rate_hz": float(row.E_firing_rate_hz),
                "I_firing_rate_hz": float(row.I_firing_rate_hz),
            })
    if rank == 0:
        calibration = pd.DataFrame(calibration_rows)
        target = _duration_matched_target(calibration, sources, cfg)
        calibration.to_csv(root / "reference_B_calibration.csv", index=False)
        (root / "frozen_duration_matched_B_target.json").write_text(json.dumps(
            _json_ready(target), indent=2, allow_nan=False
        ))
    else:
        calibration, target = None, None
    target = comm.bcast(target, root=0)

    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for context in _run_context_specs(cfg):
        if rank == 0:
            print(
                f"context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"D={context['diffusion_rad2_per_s']:g} rad^2/s"
            )
        state_cfg = _with_diffusion_state(cfg, {
            "frequency_hz": float(context["hidden_frequency_hz"]),
            "phase_seed": int(context["phase_seed"]),
            "diffusion_rad2_per_s": float(context["diffusion_rad2_per_s"]),
        })
        first_future = _future_seed(cfg, context, 0)
        baseline_reference = _run_controller(
            condition_cfg=state_cfg,
            context=context,
            future_seed=first_future,
            future_index=0,
            mode=SHAM,
            action_index=0,
            root=root,
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            screening = _context_features(baseline_reference, context, target, cfg)
            screening_rows.append(screening)
            eligible = bool(screening["eligible"])
            selected_frequency = float(screening["EEG_selected_frequency_hz"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'EXCLUDED'}; "
                f"selected={selected_frequency:g} Hz; "
                f"reason={screening['exclusion_reasons']}"
            )
        else:
            screening, eligible, selected_frequency = None, None, None
        eligible = bool(comm.bcast(eligible, root=0))
        selected_frequency = float(comm.bcast(selected_frequency, root=0))
        if not eligible:
            continue
        action_cfg = _with_action_frequency(state_cfg, selected_frequency)
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations)):
            future_seed = _future_seed(cfg, context, future_index)
            episodes: dict[str, dict[str, Any]] | None = {} if rank == 0 else None
            for action_index, mode in enumerate(_controller_modes(cfg)):
                if future_index == 0 and mode == SHAM:
                    episode = baseline_reference
                else:
                    episode = _run_controller(
                        condition_cfg=action_cfg,
                        context=context,
                        future_seed=future_seed,
                        future_index=future_index,
                        mode=mode,
                        action_index=action_index,
                        root=root,
                        comm=comm,
                        size=size,
                        rank=rank,
                    )
                if rank == 0:
                    episodes[mode] = episode
            if rank == 0:
                rows, trajectories, updates = _metric_rows(
                    context=context,
                    screening=screening,
                    future_index=future_index,
                    future_seed=future_seed,
                    episodes=episodes,
                    baseline_reference=baseline_reference,
                    target=target,
                    cfg=cfg,
                )
                _augment_metric_rows(rows, episodes, cfg)
                _augment_common_audit(rows, episodes, cfg)
                metric_rows.extend(rows)
                trajectory_rows.extend(trajectories)
                update_rows.extend(updates)
        del baseline_reference

    if rank != 0:
        return
    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "prospective_screening.csv", index=False)
    if not metric_rows:
        conclusion = {
            "scope": "H4-BW2 exploratory cadence/reliability discovery",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "selected_controller": None,
                "ready_for_disjoint_12_structure_H4_confirmation": False,
            },
            "runtime_seconds": float(time.perf_counter() - started),
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(
            conclusion, indent=2, allow_nan=False
        ))
        print("No eligible contexts; H4-BW2: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    updates = pd.DataFrame(update_rows)
    expected = _expected_map(metrics)
    comparison, structure, summary, selection = _comparison_tables(
        expected, metrics, cfg
    )
    temporal = _temporal_summary(trajectories)
    checks, conclusions = _checks(
        calibration=calibration,
        target=target,
        screening=screening,
        metrics=metrics,
        expected=expected,
        updates=updates,
        summary=summary,
        selection=selection,
        sources=sources,
        cfg=cfg,
    )
    metrics.to_csv(root / "context_controller_future_metrics.csv", index=False)
    expected.to_csv(root / "expected_context_controller_map.csv", index=False)
    comparison.to_csv(root / "controller_comparison_by_context.csv", index=False)
    structure.to_csv(root / "structure_level_controller_advantage.csv", index=False)
    summary.to_csv(root / "controller_selection_summary.csv", index=False)
    trajectories.to_csv(root / "one_second_eeg_trajectories.csv", index=False)
    temporal.to_csv(root / "temporal_advantage_summary.csv", index=False)
    updates.to_csv(root / "causal_phase_updates.csv", index=False)
    frozen = {
        **selection,
        "selected_profile": (
            _profile(cfg, str(selection["selected_controller"]))
            if selection["selected_controller"] is not None else None
        ),
        "initialization_history_ms": 1000.0,
        "common_audit_history_ms": 1000.0,
        "common_audit_interval_ms": 250.0,
        "correction_horizon_ms": 250.0,
        "amplitude_v_per_m": 0.2,
        "relative_phase_offset_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
        "montage": str(cfg.analysis.tacs.axial_montage),
        "requires_disjoint_H4_confirmation": True,
    }
    (root / "frozen_controller_candidate.json").write_text(json.dumps(
        _json_ready(frozen), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H4BW2_phase_refresh_cadence_discovery",
        "frozen_sources": {"roots": sources["roots"], "hashes": sources["hashes"]},
        "controller_profiles": {
            mode: _profile(cfg, mode) for mode in EXPECTED_MODES
        },
        "primary_endpoint": "eight-second ideal-EEG absolute log-alpha distance to B",
        "secondary_temporal_audit": "eight consecutive one-second endpoint windows",
        "common_phase_auditor": "one-second causal estimator on 250-ms boundaries",
        "reliability_metric": "within-context paired-effect standard deviation",
        "statistical_unit": "independent circuit structure",
        "not_a_bandit_or_confirmatory_experiment": True,
        "concurrent_EEG_is_ideal_and_artifact_free": True,
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H4-BW2 exploratory ideal-neural-EEG cadence/reliability discovery",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure; other axes are repeats",
        "inference_boundary": (
            "discovery only; a selected controller must be frozen and tested on "
            "new 12-structure H4 confirmation seeds"
        ),
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot):
        _plots(
            root=root,
            expected=expected,
            summary=summary,
            structure=structure,
            updates=updates,
            temporal=temporal,
        )
    print("\n### H4-BW2 screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### H4-BW2 cadence/reliability checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Candidate summaries")
    print(summary.to_string(index=False))
    print(f"\nCadence candidate selected: {conclusions['selected_controller'] or 'NONE'}")
    print(
        "Ready for disjoint 12-structure H4 confirmation: "
        f"{'YES' if conclusions['ready_for_disjoint_12_structure_H4_confirmation'] else 'NO'}"
    )
    print("Contextual bandit status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
