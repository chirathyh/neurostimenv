"""CL1-C held-out confirmation of an EEG-trajectory tACS dose rule.

CL1-P found a genuine 0.2-versus-0.4-V/m action crossover, but its frozen
absolute-target probe rule collapsed to fixed 0.4 V/m.  A single exploratory
EEG-only observation was then frozen: maintain 0.2 V/m when alpha power falls
from a duration-matched baseline to the common probe, otherwise escalate to
0.4 V/m.  This runner evaluates that rule without refitting on wholly disjoint
crossed structure/drive seeds.

Every eligible active replay has identical history through the 0.2-V/m probe.
Both post-probe actions are simulated so the policy, two fixed-dose controls,
and a counterfactual oracle can be compared with common random numbers.  A
baseline-only severity rule and a same-time sham-trajectory rule are audits;
neither may replace the primary held-out comparison.

This is ideal neural-only EEG in a toy circuit.  It is not RL, a depression
model, a treatment result, or a claim about artifact-contaminated human EEG.
"""

from __future__ import annotations

import hashlib
import itertools
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
    _condition_for_seed,
    _plain,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_screened_confirmation import (  # noqa: E402
    _load_frozen_candidate,
    _screen_phase_quality,
    _screening_decision,
)
from experiments.ballnstick_analysis.run_ballnstick_context_dose_feasibility import (  # noqa: E402
    _add_context_metadata,
    _context_specs,
    _phase_seed,
    _seed_values,
)
from experiments.ballnstick_analysis.run_ballnstick_context_probe_feasibility import (  # noqa: E402
    ACTIVE_ARMS,
    ESCALATE,
    MAINTAIN,
    SHAM,
    _context_action_rows,
    _decision_psd,
    _probe_context,
    _probe_timeline,
    _run_episode,
    _screen_view,
)


POLICY = "frozen_eeg_trajectory_rule"
BASELINE_CONTROL = "frozen_baseline_severity_rule"
SHAM_TRAJECTORY_CONTROL = "paired_sham_trajectory_audit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_rule_arm(
    baseline_minus_probe_log10: float, *, threshold: float = 0.0
) -> str:
    """Frozen CL1-C rule: a decreasing trajectory prevents escalation."""
    return (
        MAINTAIN
        if float(baseline_minus_probe_log10) > float(threshold)
        else ESCALATE
    )


def _baseline_severity_rule_arm(
    screen_log10_alpha: float, *, frozen_a_mean_log10_alpha: float
) -> str:
    """Outcome-independent control using the previously frozen A mean."""
    return (
        MAINTAIN
        if float(screen_log10_alpha) <= float(frozen_a_mean_log10_alpha)
        else ESCALATE
    )


def _source_seed_set(provenance: dict[str, Any]) -> set[int]:
    seeds: set[int] = set()
    for context in provenance.get("crossed_contexts", []):
        for name in ("trial_seed", "structure_seed", "drive_seed", "phase_seed"):
            if name in context:
                seeds.add(int(context[name]))
    for seed in provenance.get("probe_target_calibration_seeds", []):
        seeds.add(int(seed))
    return seeds


def _load_discovery(cfg: DictConfig) -> dict[str, Any]:
    root = Path(to_absolute_path(str(cfg.analysis.discovery.result_dir)))
    required = {
        "conclusion": root / "experiment_conclusion.json",
        "summary": root / "context_counterfactual_summary.csv",
        "probe_target": root / "frozen_probe_target.json",
        "provenance": root / "frozen_protocol_provenance.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CL1-C requires the completed CL1-P discovery outputs: "
            + ", ".join(missing)
        )
    hashes = {name: _sha256(path) for name, path in required.items()}
    expected_hashes = OmegaConf.to_container(
        cfg.analysis.discovery.expected_sha256, resolve=True
    )
    for name, expected in expected_hashes.items():
        if hashes.get(str(name)) != str(expected):
            raise ValueError(
                f"CL1-P source file {name!r} changed after the CL1-C protocol "
                "was frozen; use the recorded discovery result or define a new "
                "confirmation protocol."
            )
    conclusion = json.loads(required["conclusion"].read_text())
    source_summary = pd.read_csv(required["summary"])
    probe_target = json.loads(required["probe_target"].read_text())
    provenance = json.loads(required["provenance"].read_text())
    checks = conclusion.get("checks", {})
    if not bool(checks.get("oracle_has_practical_contextual_opportunity", False)):
        raise ValueError(
            "The source CL1-P experiment did not establish an oracle action "
            "opportunity; the trajectory rule must not be confirmed from it."
        )
    threshold = float(cfg.analysis.trajectory_rule.threshold_log10)
    chosen = np.where(
        source_summary.context_probe_alpha_suppression_log10.to_numpy(float)
        > threshold,
        source_summary.maintain_distance_to_B_log10.to_numpy(float),
        source_summary.escalate_distance_to_B_log10.to_numpy(float),
    )
    chosen_arms = np.where(
        source_summary.context_probe_alpha_suppression_log10.to_numpy(float)
        > threshold,
        MAINTAIN,
        ESCALATE,
    )
    discovery_audit = {
        "selection_stage": "post_CL1-P_exploratory_EEG_trajectory_audit",
        "source_result_dir": str(root),
        "source_context_count": int(len(source_summary)),
        "frozen_feature": "matched_baseline_minus_active_probe_log10_alpha",
        "frozen_threshold_log10": threshold,
        "frozen_rule": (
            "maintain 0.2 V/m when matched baseline minus active-probe "
            "log10 alpha is > 0; otherwise escalate to 0.4 V/m"
        ),
        "source_selected_arms": sorted(set(chosen_arms.tolist())),
        "source_oracle_match_fraction": float(
            np.mean(chosen_arms == source_summary.oracle_arm.to_numpy(str))
        ),
        "source_mean_advantage_over_fixed_0p4_log10": float(
            np.mean(
                source_summary.escalate_distance_to_B_log10.to_numpy(float)
                - chosen
            )
        ),
        "source_outcomes_used_only_to_freeze_rule_before_confirmation": True,
        "confirmation_performs_no_rule_refitting": True,
    }
    if set(chosen_arms.tolist()) != {MAINTAIN, ESCALATE}:
        raise ValueError("The frozen discovery trajectory rule used only one action.")
    if not np.isclose(discovery_audit["source_oracle_match_fraction"], 1.0):
        raise ValueError(
            "The source trajectory rule no longer reproduces its frozen CL1-P "
            "discovery audit."
        )
    return {
        "root": str(root),
        "conclusion": conclusion,
        "probe_target": probe_target,
        "provenance": provenance,
        "discovery_audit": discovery_audit,
        "sha256": hashes,
    }


def _validate_design(
    cfg: DictConfig, frozen: dict[str, Any], discovery: dict[str, Any]
) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("CL1-C requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("Every CL1-C arm requires inhibition_scale=1.")
    _probe_timeline(cfg)
    probe = float(cfg.analysis.actions.probe_dose_v_per_m)
    maintain = float(cfg.analysis.actions.maintain_dose_v_per_m)
    escalate = float(cfg.analysis.actions.escalate_dose_v_per_m)
    maximum = float(cfg.analysis.maximum_field_v_per_m)
    if not (
        np.isclose(probe, 0.2)
        and np.isclose(maintain, 0.2)
        and np.isclose(escalate, 0.4)
        and escalate <= maximum
    ):
        raise ValueError("CL1-C freezes probe/maintain=0.2 and escalate=0.4 V/m.")
    if not np.isclose(float(cfg.analysis.trajectory_rule.threshold_log10), 0.0):
        raise ValueError("CL1-C freezes the EEG-trajectory threshold at zero.")

    candidate = frozen["candidate"]
    if not np.isclose(float(candidate["selected_dose_v_per_m"]), escalate):
        raise ValueError("The 0.4-V/m arm must equal the frozen candidate.")
    if not np.isclose(float(candidate["frequency_hz"]), 10.0):
        raise ValueError("CL1-C retains the frozen 10-Hz frequency.")
    if not np.isclose(float(candidate["relative_phase_offset_rad"]), np.pi):
        raise ValueError("CL1-C retains the frozen EEG-relative 180-degree phase.")
    if str(candidate["montage"]) != str(cfg.analysis.tacs.axial_montage):
        raise ValueError("CL1-C retains the frozen axial montage.")
    if not np.isclose(float(frozen["target"]["selected_modulation_depth"]), 0.04):
        raise ValueError("CL1-C requires the frozen 0.04 elevated-alpha state.")

    timeline = cfg.analysis.timeline
    window_ms = float(cfg.env.simulation.obs_win_len)
    if int(timeline.baseline_steps) < 4 or int(timeline.probe_steps) < 3:
        raise ValueError("CL1-C retains at least 4-s baseline and 3-s probe epochs.")
    if float(timeline.probe_analysis_trim_start_ms) < float(
        timeline.block_ramp_ms
    ):
        raise ValueError("The probe analysis must exclude the onset ramp.")
    if float(timeline.decision_analysis_trim_start_ms) < float(
        timeline.amplitude_transition_ms
    ):
        raise ValueError("The decision analysis must exclude dose transition.")
    probe_ms = int(timeline.probe_steps) * window_ms
    decision_ms = int(timeline.decision_steps) * window_ms
    if float(timeline.probe_analysis_trim_start_ms) >= probe_ms:
        raise ValueError("Probe trimming removed the complete probe epoch.")
    if (
        float(timeline.decision_analysis_trim_start_ms)
        + float(timeline.decision_analysis_trim_end_ms)
        >= decision_ms
    ):
        raise ValueError("Decision trimming removed the complete decision epoch.")

    structures = set(_seed_values(cfg, kind="structure"))
    drives = set(_seed_values(cfg, kind="drive"))
    phase = {_phase_seed(cfg)}
    trials = {int(item["trial_seed"]) for item in _context_specs(cfg)}
    namespaces = (structures, drives, phase, trials)
    if any(not values for values in namespaces):
        raise ValueError("Every CL1-C seed namespace must be nonempty.")
    if any(
        namespaces[i].intersection(namespaces[j])
        for i in range(len(namespaces))
        for j in range(i + 1, len(namespaces))
    ):
        raise ValueError("CL1-C seed namespaces must be internally disjoint.")
    current = set.union(*namespaces)
    if current.intersection(_source_seed_set(discovery["provenance"])):
        raise ValueError("CL1-C confirmation seeds overlap CL1-P discovery seeds.")
    if max(current) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("CL1-C seeds are too large for seed * 10,000.")


def _confirmation_action_rows(
    *,
    context: dict[str, Any],
    episodes: dict[str, dict[str, Any]],
    screening: dict[str, Any],
    screen_phase: dict[str, Any],
    probe_target: dict[str, Any],
    target_model: dict[str, Any],
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    rows = _context_action_rows(
        context=context,
        episodes=episodes,
        screening=screening,
        screen_phase=screen_phase,
        probe_target=probe_target,
        target_model=target_model,
        cfg=cfg,
    )
    sham_context = _probe_context(
        episodes[SHAM], probe_target=probe_target, cfg=cfg
    )
    sham_trajectory = float(
        sham_context["context_probe_alpha_suppression_log10"]
    )
    active_trajectory = float(
        rows[0]["context_probe_alpha_suppression_log10"]
    )
    shared = {
        "screen_log10_alpha_power": float(
            screening["screen_log10_alpha_power"]
        ),
        "screen_alpha_excess_to_B_log10": float(
            screening["screen_log10_alpha_power"]
            - float(target_model["B_mean_log10_alpha"])
        ),
        "context_sham_trajectory_log10": sham_trajectory,
        "context_causal_probe_suppression_log10": float(
            active_trajectory - sham_trajectory
        ),
    }
    return [{**row, **shared} for row in rows]


def _selected_distance(maintain: pd.Series, escalate: pd.Series, arm: str) -> float:
    if arm == MAINTAIN:
        return float(maintain.post_distance_to_B_log10)
    if arm == ESCALATE:
        return float(escalate.post_distance_to_B_log10)
    raise ValueError(f"Unknown selected arm {arm!r}.")


def _confirmation_summary(
    metrics: pd.DataFrame, *, target_model: dict[str, Any], cfg: DictConfig
) -> pd.DataFrame:
    threshold = float(cfg.analysis.trajectory_rule.threshold_log10)
    a_mean = float(target_model["A_mean_log10_alpha"])
    practical = float(cfg.analysis.criteria.practical_advantage_log10)
    rows: list[dict[str, Any]] = []
    shared_names = (
        "context_id", "context_order", "trial_seed", "structure_seed",
        "drive_seed", "phase_seed", "structure_index", "drive_index",
        "screen_log10_alpha_power", "screen_alpha_excess_to_B_log10",
        "context_baseline_matched_log10_alpha", "context_probe_log10_alpha",
        "context_probe_signed_error_to_B_log10",
        "context_probe_alpha_suppression_log10",
        "context_probe_gain_log10_per_vpm",
        "context_sham_trajectory_log10",
        "context_causal_probe_suppression_log10",
        "pre_action_distance_to_B_log10",
        "paired_predecision_relative_rms_error",
    )
    for _, group in metrics.groupby("context_id", sort=False):
        maintain = group[group.arm.eq(MAINTAIN)].iloc[0]
        escalate = group[group.arm.eq(ESCALATE)].iloc[0]
        maintain_distance = float(maintain.post_distance_to_B_log10)
        escalate_distance = float(escalate.post_distance_to_B_log10)
        oracle_arm = MAINTAIN if maintain_distance <= escalate_distance else ESCALATE
        oracle_distance = min(maintain_distance, escalate_distance)
        policy_arm = _trajectory_rule_arm(
            float(maintain.context_probe_alpha_suppression_log10),
            threshold=threshold,
        )
        baseline_arm = _baseline_severity_rule_arm(
            float(maintain.screen_log10_alpha_power),
            frozen_a_mean_log10_alpha=a_mean,
        )
        sham_arm = _trajectory_rule_arm(
            float(maintain.context_sham_trajectory_log10),
            threshold=threshold,
        )
        rows.append({
            **{name: maintain[name] for name in shared_names},
            "maintain_distance_to_B_log10": maintain_distance,
            "escalate_distance_to_B_log10": escalate_distance,
            "maintain_minus_escalate_distance_log10": (
                maintain_distance - escalate_distance
            ),
            "oracle_arm": oracle_arm,
            "oracle_distance_to_B_log10": oracle_distance,
            "oracle_practical_action_difference": bool(
                abs(maintain_distance - escalate_distance) >= practical
            ),
            "trajectory_rule_arm": policy_arm,
            "trajectory_rule_distance_to_B_log10": _selected_distance(
                maintain, escalate, policy_arm
            ),
            "trajectory_rule_matches_oracle": bool(policy_arm == oracle_arm),
            "baseline_rule_arm": baseline_arm,
            "baseline_rule_distance_to_B_log10": _selected_distance(
                maintain, escalate, baseline_arm
            ),
            "sham_trajectory_rule_arm": sham_arm,
            "sham_trajectory_rule_distance_to_B_log10": _selected_distance(
                maintain, escalate, sham_arm
            ),
        })
    summary = pd.DataFrame(rows)
    summary["trajectory_advantage_over_fixed_0p2_log10"] = (
        summary.maintain_distance_to_B_log10
        - summary.trajectory_rule_distance_to_B_log10
    )
    summary["trajectory_advantage_over_fixed_0p4_log10"] = (
        summary.escalate_distance_to_B_log10
        - summary.trajectory_rule_distance_to_B_log10
    )
    summary["trajectory_advantage_over_baseline_rule_log10"] = (
        summary.baseline_rule_distance_to_B_log10
        - summary.trajectory_rule_distance_to_B_log10
    )
    summary["trajectory_advantage_over_sham_trajectory_rule_log10"] = (
        summary.sham_trajectory_rule_distance_to_B_log10
        - summary.trajectory_rule_distance_to_B_log10
    )
    summary["trajectory_oracle_regret_log10"] = (
        summary.trajectory_rule_distance_to_B_log10
        - summary.oracle_distance_to_B_log10
    )
    return summary


def _exact_one_sided_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    observed = float(values.mean())
    exceed = 0
    total = 2 ** int(values.size)
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        if float(np.mean(values * np.asarray(signs))) >= observed - 1.0e-15:
            exceed += 1
    return float(exceed / total)


def _bootstrap_structure_mean_ci(
    values: np.ndarray, *, seed: int, n_bootstrap: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(n_bootstrap), values.size))
    boot = values[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _context_shuffle_null(
    summary: pd.DataFrame, *, best_fixed_arm: str, cfg: DictConfig
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.experiment.seed), 2_000_033])
    )
    context = summary.context_probe_alpha_suppression_log10.to_numpy(float)
    maintain = summary.maintain_distance_to_B_log10.to_numpy(float)
    escalate = summary.escalate_distance_to_B_log10.to_numpy(float)
    fixed = maintain if best_fixed_arm == MAINTAIN else escalate
    observed = float(
        np.mean(fixed - summary.trajectory_rule_distance_to_B_log10.to_numpy(float))
    )
    threshold = float(cfg.analysis.trajectory_rule.threshold_log10)
    rows = []
    for permutation in range(int(cfg.analysis.context_shuffle.n_permutations)):
        shuffled = rng.permutation(context)
        selected = np.where(shuffled > threshold, maintain, escalate)
        rows.append({
            "permutation": permutation + 1,
            "mean_advantage_over_best_fixed_log10": float(
                np.mean(fixed - selected)
            ),
        })
    null = pd.DataFrame(rows)
    values = null.mean_advantage_over_best_fixed_log10.to_numpy(float)
    p_value = float(
        (1 + np.count_nonzero(values >= observed)) / (values.size + 1)
    )
    return null, {
        "observed_mean_advantage_over_best_fixed_log10": observed,
        "shuffled_mean_advantage_log10": float(values.mean()),
        "shuffled_95th_percentile_log10": float(np.quantile(values, 0.95)),
        "context_shuffle_p_value": p_value,
    }


def _evaluate_confirmation(
    *,
    screening: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    discovery: dict[str, Any],
    cfg: DictConfig,
) -> tuple[
    dict[str, bool], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    criteria = cfg.analysis.criteria
    mean_02 = float(summary.maintain_distance_to_B_log10.mean())
    mean_04 = float(summary.escalate_distance_to_B_log10.mean())
    best_fixed_arm = MAINTAIN if mean_02 <= mean_04 else ESCALATE
    best_fixed_name = "fixed_0p2" if best_fixed_arm == MAINTAIN else "fixed_0p4"
    best_fixed_column = (
        "maintain_distance_to_B_log10"
        if best_fixed_arm == MAINTAIN
        else "escalate_distance_to_B_log10"
    )
    summary = summary.copy()
    summary["best_fixed_arm"] = best_fixed_arm
    summary["best_fixed_distance_to_B_log10"] = summary[best_fixed_column]
    summary["trajectory_advantage_over_best_fixed_log10"] = (
        summary.best_fixed_distance_to_B_log10
        - summary.trajectory_rule_distance_to_B_log10
    )
    summary["best_fixed_oracle_regret_log10"] = (
        summary.best_fixed_distance_to_B_log10
        - summary.oracle_distance_to_B_log10
    )

    structure = summary.groupby(
        ["structure_seed", "structure_index"], as_index=False
    ).agg(
        eligible_context_count=("context_id", "size"),
        trajectory_distance_to_B_log10=(
            "trajectory_rule_distance_to_B_log10", "mean"
        ),
        fixed_0p2_distance_to_B_log10=(
            "maintain_distance_to_B_log10", "mean"
        ),
        fixed_0p4_distance_to_B_log10=(
            "escalate_distance_to_B_log10", "mean"
        ),
        best_fixed_distance_to_B_log10=(
            "best_fixed_distance_to_B_log10", "mean"
        ),
        oracle_distance_to_B_log10=("oracle_distance_to_B_log10", "mean"),
        trajectory_advantage_over_best_fixed_log10=(
            "trajectory_advantage_over_best_fixed_log10", "mean"
        ),
        trajectory_advantage_over_fixed_0p2_log10=(
            "trajectory_advantage_over_fixed_0p2_log10", "mean"
        ),
        trajectory_advantage_over_fixed_0p4_log10=(
            "trajectory_advantage_over_fixed_0p4_log10", "mean"
        ),
    )
    structure_values = structure[
        "trajectory_advantage_over_best_fixed_log10"
    ].to_numpy(float)
    sign_flip_p = _exact_one_sided_sign_flip(structure_values)
    ci_low, ci_high = _bootstrap_structure_mean_ci(
        structure_values,
        seed=int(cfg.experiment.seed) + 2_000_039,
        n_bootstrap=int(cfg.analysis.structure_bootstrap.n_resamples),
    )
    shuffle_frame, shuffle = _context_shuffle_null(
        summary, best_fixed_arm=best_fixed_arm, cfg=cfg
    )

    policy_mean = float(summary.trajectory_rule_distance_to_B_log10.mean())
    oracle_mean = float(summary.oracle_distance_to_B_log10.mean())
    best_fixed_mean = min(mean_02, mean_04)
    best_fixed_regret = best_fixed_mean - oracle_mean
    policy_regret = policy_mean - oracle_mean
    regret_reduction = (
        (best_fixed_regret - policy_regret) / best_fixed_regret
        if best_fixed_regret > 0.0 else 0.0
    )
    mean_adv_02 = mean_02 - policy_mean
    mean_adv_04 = mean_04 - policy_mean
    positive_structure_fraction = float(
        (structure_values > 0.0).mean()
    )
    source_seeds = _source_seed_set(discovery["provenance"])
    confirmation_seeds = set(_seed_values(cfg, kind="structure")) | set(
        _seed_values(cfg, kind="drive")
    ) | {_phase_seed(cfg)} | {
        int(item["trial_seed"]) for item in _context_specs(cfg)
    }
    oracle_arms = set(summary.oracle_arm)
    practical_oracle = int(summary.oracle_practical_action_difference.sum())

    checks = {
        "source_discovery_oracle_opportunity_passed": bool(
            discovery["conclusion"]["checks"][
                "oracle_has_practical_contextual_opportunity"
            ]
        ),
        "confirmation_seeds_disjoint_from_discovery": confirmation_seeds.isdisjoint(
            source_seeds
        ),
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
        "paired_history_identical_through_probe": bool(
            metrics.paired_predecision_relative_rms_error.max()
            <= float(criteria.maximum_predecision_relative_rms_error)
        ),
        "all_enrolled_phase_actionable": bool(metrics.phase_quality_pass.all()),
        "action_phase_tracks_screen_estimate": bool(
            metrics.action_phase_tracking_error_rad.max()
            <= float(criteria.maximum_phase_tracking_error_rad)
        ),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "field_removal_recovered": bool(metrics.field_removal_recovered.all()),
        "baseline_causality": bool(
            metrics.baseline_relative_rms_error_vs_sham.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "coherent_decomposition_exact": bool(np.allclose(
            metrics.coherent_net_change_v2,
            metrics.coherent_interference_cross_term_v2
            + metrics.coherent_induced_component_v2,
            rtol=1.0e-10,
            atol=1.0e-30,
        )),
        "oracle_retains_both_action_types": oracle_arms == {MAINTAIN, ESCALATE},
        "minimum_practical_oracle_crossovers": practical_oracle
        >= int(criteria.minimum_practical_oracle_crossovers),
        "trajectory_rule_uses_multiple_actions": summary.trajectory_rule_arm.nunique()
        >= int(criteria.minimum_selected_action_count),
        "trajectory_rule_beats_fixed_0p2": mean_adv_02
        >= float(criteria.minimum_mean_advantage_over_each_fixed_log10),
        "trajectory_rule_beats_fixed_0p4": mean_adv_04
        >= float(criteria.minimum_mean_advantage_over_each_fixed_log10),
        "trajectory_rule_positive_across_structures": positive_structure_fraction
        >= float(criteria.minimum_positive_structure_fraction),
        "structure_level_sign_flip_supports_advantage": sign_flip_p
        <= float(criteria.maximum_structure_sign_flip_p_value),
        "trajectory_context_beats_shuffled_context": bool(
            shuffle["observed_mean_advantage_over_best_fixed_log10"]
            > shuffle["shuffled_mean_advantage_log10"]
            and shuffle["context_shuffle_p_value"]
            <= float(criteria.maximum_context_shuffle_p_value)
        ),
        "trajectory_rule_reduces_oracle_regret": regret_reduction
        >= float(criteria.minimum_oracle_regret_reduction_fraction),
        "trajectory_rule_beats_frozen_baseline_only_control": float(
            summary.trajectory_advantage_over_baseline_rule_log10.mean()
        ) >= float(criteria.minimum_control_advantage_log10),
        "trajectory_rule_beats_sham_trajectory_audit": float(
            summary.trajectory_advantage_over_sham_trajectory_rule_log10.mean()
        ) >= float(criteria.minimum_control_advantage_log10),
        "primary_policy_uses_only_predecision_ideal_EEG": True,
        "hidden_variables_excluded_from_policy": True,
    }
    primary = (
        "source_discovery_oracle_opportunity_passed",
        "confirmation_seeds_disjoint_from_discovery",
        "complete_crossed_screening_grid",
        "screening_precedes_and_excludes_stimulation_outcomes",
        "screening_does_not_use_seed_specific_B",
        "minimum_eligible_contexts",
        "eligible_structure_coverage",
        "eligible_drive_coverage",
        "paired_history_identical_through_probe",
        "all_enrolled_phase_actionable",
        "action_phase_tracks_screen_estimate",
        "all_actions_rate_safe",
        "field_removal_recovered",
        "baseline_causality",
        "coherent_decomposition_exact",
        "oracle_retains_both_action_types",
        "minimum_practical_oracle_crossovers",
        "trajectory_rule_uses_multiple_actions",
        "trajectory_rule_beats_fixed_0p2",
        "trajectory_rule_beats_fixed_0p4",
        "trajectory_rule_positive_across_structures",
        "structure_level_sign_flip_supports_advantage",
        "trajectory_context_beats_shuffled_context",
        "trajectory_rule_reduces_oracle_regret",
        "primary_policy_uses_only_predecision_ideal_EEG",
        "hidden_variables_excluded_from_policy",
    )
    core_passed = all(checks[name] for name in primary)
    conclusions = {
        "closed_loop_trajectory_policy_confirmed": core_passed,
        "ready_for_contextual_bandit_trial": core_passed,
        "probe_interval_adds_over_baseline_only": checks[
            "trajectory_rule_beats_frozen_baseline_only_control"
        ],
        "trajectory_signal_is_probe_specific": checks[
            "trajectory_rule_beats_sham_trajectory_audit"
        ],
        "screened_context_count": int(len(screening)),
        "eligible_context_count": int(len(summary)),
        "screening_yield": float(screening.eligible.mean()),
        "best_fixed_comparator": best_fixed_name,
        "mean_distance_fixed_0p2_log10": mean_02,
        "mean_distance_fixed_0p4_log10": mean_04,
        "mean_distance_trajectory_rule_log10": policy_mean,
        "mean_distance_oracle_log10": oracle_mean,
        "mean_advantage_over_fixed_0p2_log10": mean_adv_02,
        "mean_advantage_over_fixed_0p4_log10": mean_adv_04,
        "positive_structure_fraction": positive_structure_fraction,
        "structure_level_sign_flip_p_value_one_sided": sign_flip_p,
        "structure_bootstrap_ci_2p5_log10": ci_low,
        "structure_bootstrap_ci_97p5_log10": ci_high,
        "oracle_regret_reduction_fraction": float(regret_reduction),
        "trajectory_selected_arms": sorted(summary.trajectory_rule_arm.unique()),
        "trajectory_oracle_match_fraction": float(
            summary.trajectory_rule_matches_oracle.mean()
        ),
        "mean_advantage_over_baseline_rule_log10": float(
            summary.trajectory_advantage_over_baseline_rule_log10.mean()
        ),
        "mean_advantage_over_sham_trajectory_rule_log10": float(
            summary.trajectory_advantage_over_sham_trajectory_rule_log10.mean()
        ),
        **shuffle,
        "policy_observes_only_ideal_EEG": True,
        "hidden_spikes_and_rates_used_only_for_mechanism_and_safety": True,
    }
    return checks, conclusions, structure, shuffle_frame, summary


def _plot_results(
    *,
    root: Path,
    frequencies: np.ndarray,
    psds: dict[str, list[np.ndarray]],
    summary: pd.DataFrame,
    structure: pd.DataFrame,
) -> None:
    colors = {SHAM: "#9467BD", MAINTAIN: "#1F77B4", ESCALATE: "#D62728"}
    labels = {
        SHAM: "A sham",
        MAINTAIN: "probe + maintain 0.2 V/m",
        ESCALATE: "probe + escalate 0.4 V/m",
    }
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for arm in (SHAM, MAINTAIN, ESCALATE):
        mean_psd = np.mean(np.asarray(psds[arm]), axis=0)
        axis.plot(
            frequencies,
            10.0 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny)),
            label=labels[arm], color=colors[arm], linewidth=2.0,
        )
    axis.axvspan(8.0, 12.0, color="gold", alpha=0.14)
    axis.set_xlim(2.0, 25.0)
    axis.set(
        xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)",
        title="CL1-C held-out decision-period ideal neural EEG",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_confirmation_decision_psd.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.scatter(
        summary.context_probe_alpha_suppression_log10,
        summary.maintain_minus_escalate_distance_log10,
        c=summary.structure_index, cmap="viridis", s=65,
        edgecolor="black", linewidth=0.5,
    )
    axis.axvline(0.0, color="0.4", linewidth=0.9)
    axis.axhline(0.0, color="0.4", linewidth=0.9)
    axis.set(
        xlabel="Matched baseline − active-probe alpha power (log10)",
        ylabel="Distance 0.2 − distance 0.4 (log10)",
        title="Frozen EEG-trajectory rule on held-out contexts",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_02_trajectory_action_interaction.png", dpi=250)
    plt.close(figure)

    ordered = summary.sort_values("context_order")
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.plot(x, ordered.maintain_distance_to_B_log10, "o-", label="Fixed 0.2 V/m")
    axis.plot(x, ordered.escalate_distance_to_B_log10, "o-", label="Fixed 0.4 V/m")
    axis.plot(
        x, ordered.trajectory_rule_distance_to_B_log10,
        "o-", linewidth=2.2, label="Frozen trajectory rule",
    )
    axis.plot(
        x, ordered.oracle_distance_to_B_log10,
        "o--", linewidth=1.6, label="Counterfactual oracle",
    )
    axis.set_xticks(x, ordered.context_id, rotation=45, ha="right")
    axis.set(
        ylabel="Absolute distance to frozen B (log10)",
        title="Held-out adaptive policy versus both fixed doses",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_03_confirmation_policy_comparison.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    axis.bar(
        structure.structure_index.astype(str),
        structure.trajectory_advantage_over_best_fixed_log10,
        color=np.where(
            structure.trajectory_advantage_over_best_fixed_log10 >= 0.0,
            "#2CA02C", "#D62728",
        ),
    )
    axis.axhline(0.0, color="0.25", linewidth=0.9)
    axis.set(
        xlabel="Held-out structure group",
        ylabel="Trajectory-rule advantage over best fixed (log10)",
        title="Structure-level confirmation unit",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_04_structure_level_advantage.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen = _load_frozen_candidate(cfg)
    discovery = _load_discovery(cfg)
    _validate_design(cfg, frozen, discovery)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = (
        Path(to_absolute_path(str(cfg.experiment.dir)))
        / "context_trajectory_confirmation"
    )
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### CL1-C frozen EEG-trajectory confirmation")
        print(json.dumps(_plain(discovery["discovery_audit"]), indent=2))
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()

    started = time.perf_counter()
    target_model = frozen["target"]["target_model"]
    depth = float(frozen["target"]["selected_modulation_depth"])
    probe_target = discovery["probe_target"]
    contexts = _context_specs(cfg)
    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    psds = {arm: [] for arm in (SHAM, MAINTAIN, ESCALATE)}
    frequencies = None
    phase_seed = _phase_seed(cfg)
    relative_offset = float(
        cfg.analysis.frozen_candidate.expected_relative_phase_offset_rad
    )

    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} "
                f"structure={context['structure_seed']} drive={context['drive_seed']}"
            )
        a_cfg = _condition_for_seed(cfg, seed=phase_seed, modulation_depth=depth)
        sham = _run_episode(
            condition_cfg=a_cfg,
            arm=SHAM,
            seed=int(context["trial_seed"]),
            action_index=0,
            output_dir=root / "episodes" / context["context_id"] / SHAM,
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(context["structure_seed"]),
            drive_seed=int(context["drive_seed"]),
        )
        if rank == 0:
            _add_context_metadata(sham, context)
            screen_episode = _screen_view(sham)
            screen_phase = _screen_phase_quality(
                episode=screen_episode,
                relative_offset=relative_offset,
                cfg=cfg,
            )
            screening = _screening_decision(
                seed=int(context["trial_seed"]),
                screening_order=int(context["context_order"]),
                a_episode=screen_episode,
                phase_quality=screen_phase,
                target_model=target_model,
                cfg=cfg,
            )
            screening.update(context)
            screening_rows.append(screening)
            epoch_rows.extend(sham["epoch_rows"])
            eligible = bool(screening["eligible"])
            print(
                f"screen {context['context_id']}: "
                f"{'ELIGIBLE' if eligible else 'EXCLUDED'} "
                f"({screening['exclusion_reasons']})"
            )
        else:
            eligible = None
        eligible = bool(comm.bcast(eligible, root=0))
        if not eligible:
            continue

        episodes = {SHAM: sham} if rank == 0 else None
        for action_index, arm in enumerate(ACTIVE_ARMS, start=1):
            episode = _run_episode(
                condition_cfg=a_cfg,
                arm=arm,
                seed=int(context["trial_seed"]),
                action_index=action_index,
                output_dir=root / "episodes" / context["context_id"] / arm,
                comm=comm,
                size=size,
                rank=rank,
                structure_seed=int(context["structure_seed"]),
                drive_seed=int(context["drive_seed"]),
            )
            if rank == 0:
                _add_context_metadata(episode, context)
                episodes[arm] = episode
                epoch_rows.extend(episode["epoch_rows"])

        if rank == 0:
            metric_rows.extend(_confirmation_action_rows(
                context=context,
                episodes=episodes,
                screening=screening_rows[-1],
                screen_phase=screen_phase,
                probe_target=probe_target,
                target_model=target_model,
                cfg=cfg,
            ))
            if bool(cfg.experiment.plot):
                for arm, episode in episodes.items():
                    frequencies, psd = _decision_psd(episode, cfg)
                    psds[arm].append(psd)
            del episodes

    if rank != 0:
        return

    screening_frame = pd.DataFrame(screening_rows)
    screening_frame.to_csv(root / "screening_audit.csv", index=False)
    (root / "discovery_rule_audit.json").write_text(
        json.dumps(_plain(discovery["discovery_audit"]), indent=2)
    )
    if not metric_rows:
        result = {
            "scope": "held-out ideal-EEG CL1-C confirmation",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "closed_loop_trajectory_policy_confirmed": False,
                "ready_for_contextual_bandit_trial": False,
                "eligible_context_count": 0,
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No held-out context passed the frozen screen.",
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        print("\nNo eligible CL1-C contexts; confirmation: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    epochs = pd.DataFrame(epoch_rows)
    summary = _confirmation_summary(metrics, target_model=target_model, cfg=cfg)
    checks, conclusions, structure, shuffle, summary = _evaluate_confirmation(
        screening=screening_frame,
        metrics=metrics,
        summary=summary,
        discovery=discovery,
        cfg=cfg,
    )

    epochs.to_csv(root / "context_epoch_eeg_and_hidden_metrics.csv", index=False)
    metrics.to_csv(root / "context_action_metrics.csv", index=False)
    summary.to_csv(root / "heldout_context_policy_summary.csv", index=False)
    structure.to_csv(root / "structure_level_policy_comparison.csv", index=False)
    shuffle.to_csv(root / "trajectory_context_shuffle_null.csv", index=False)
    provenance = {
        **frozen,
        "source_CL1_P_result_dir": discovery["root"],
        "source_CL1_P_sha256": discovery["sha256"],
        "source_CL1_P_seed_set": sorted(_source_seed_set(discovery["provenance"])),
        "frozen_probe_target": probe_target,
        "frozen_trajectory_rule": discovery["discovery_audit"],
        "frozen_baseline_only_control": (
            "maintain 0.2 V/m when screened log10 alpha is <= the previously "
            "frozen A population mean; otherwise escalate to 0.4 V/m"
        ),
        "confirmation_crossed_contexts": contexts,
        "common_probe_v_per_m": float(cfg.analysis.actions.probe_dose_v_per_m),
        "post_probe_actions_v_per_m": [
            float(cfg.analysis.actions.maintain_dose_v_per_m),
            float(cfg.analysis.actions.escalate_dose_v_per_m),
        ],
        "paired_history_identical_until_decision": True,
        "selection_performed_on_confirmation": "none",
        "policy_uses_hidden_variables": False,
    }
    (root / "frozen_protocol_provenance.json").write_text(
        json.dumps(_plain(provenance), indent=2)
    )
    result = {
        "scope": "ideal neural-only EEG, screen-positive held-out toy contexts",
        "checks": checks,
        "conclusions": conclusions,
        "primary_policy": POLICY,
        "primary_comparators": ["fixed_0p2", "fixed_0p4"],
        "secondary_controls": [BASELINE_CONTROL, SHAM_TRAJECTORY_CONTROL],
        "reward": "negative absolute log10 alpha-power distance to frozen B mean",
        "statistical_unit": "structure seed; drive seeds are crossed sessions",
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "A primary pass confirms that the frozen predecision EEG-trajectory "
            "rule outperforms both fixed doses on disjoint toy circuit structures. "
            "It supports, but does not itself test, a two-action contextual bandit."
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
            structure=structure,
        )

    print("\n### CL1-C screening")
    print(f"crossed contexts screened: {len(screening_frame)}")
    print(f"eligible contexts: {int(screening_frame.eligible.sum())}")
    print(f"screening yield: {float(screening_frame.eligible.mean()):.3f}")
    print("\n### CL1-C confirmation checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nClosed-loop EEG-trajectory policy:",
        "CONFIRMED" if conclusions["closed_loop_trajectory_policy_confirmed"]
        else "NOT CONFIRMED",
    )
    print("Contextual bandit status: NOT TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
