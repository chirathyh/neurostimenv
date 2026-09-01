"""CDM2-D expanded discovery of a monotone EEG-severity dose rule.

Each screen-positive context is observed for twelve stimulation-free seconds,
then replayed under sham, 0.2 V/m, or 0.4 V/m. One amplitude is held for the
complete intervention. Three independent postdecision Poisson continuations
estimate conditional expected outcomes while the structure and predecision
history remain identical.

The only policy variable is predecision ideal-EEG alpha excess over a frozen B
population reference. Discovery fits the transparent monotone rule

    alpha excess < threshold -> 0.2 V/m; otherwise -> 0.4 V/m.

Leave-one-structure-out performance is the predictive gate. A qualified
threshold is only a candidate for a later disjoint confirmation. This runner
does not train or test a contextual bandit and makes no human/depression claim.
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
from experiments.ballnstick_analysis.run_ballnstick_single_action_dose_map import (  # noqa: E402
    SHAM_DOSE,
    _baseline_screen_view,
    _context_features,
    _context_specs,
    _doses,
    _expected_action_map,
    _future_seeds,
    _replay_rows,
    _run_replay,
    _state_summary,
    _validate_design as _validate_cdm1_design,
)


POLICY_FEATURE = "context_alpha_excess_log10"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_cdm1(cfg: DictConfig) -> dict[str, Any]:
    root = Path(to_absolute_path(str(cfg.analysis.source_cdm1.result_dir)))
    paths = {
        "conclusion": root / "experiment_conclusion.json",
        "summary": root / "context_expected_action_summary.csv",
        "provenance": root / "protocol_provenance.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CDM2-D requires the completed CDM1-S result: " + ", ".join(missing)
        )
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = OmegaConf.to_container(
        cfg.analysis.source_cdm1.expected_sha256, resolve=True
    )
    for name, value in expected.items():
        if hashes[str(name)] != str(value):
            raise ValueError(
                f"CDM1-S source {name!r} changed after CDM2-D was specified."
            )
    conclusion = json.loads(paths["conclusion"].read_text())
    source_summary = pd.read_csv(paths["summary"])
    provenance = json.loads(paths["provenance"].read_text())
    if not bool(conclusion["conclusions"]["single_action_conditional_dose_map_feasible"]):
        raise ValueError("The source CDM1-S conditional dose-map gate did not pass.")
    expected_actions = sorted(
        float(value)
        for value in conclusion["conclusions"]["expected_optimal_active_doses"]
    )
    if expected_actions != [0.2, 0.4]:
        raise ValueError("CDM1-S did not justify the frozen {0.2, 0.4}-V/m set.")
    return {
        "root": str(root),
        "sha256": hashes,
        "conclusion": conclusion,
        "summary": source_summary,
        "provenance": provenance,
    }


def _validate_design(
    cfg: DictConfig, frozen: dict[str, Any], source: dict[str, Any]
) -> None:
    _validate_cdm1_design(cfg, frozen)
    timeline = cfg.analysis.timeline
    window_ms = float(cfg.env.simulation.obs_win_len)
    smoke_test = bool(cfg.analysis.smoke_test)
    if not smoke_test and int(timeline.baseline_steps) * window_ms < 12_000.0:
        raise ValueError("CDM2-D requires at least 12 s of stimulation-free EEG.")
    if [float(value) for value in cfg.analysis.actions.active_doses_v_per_m] != [
        0.2, 0.4
    ]:
        raise ValueError("CDM2-D freezes the active action set to {0.2, 0.4} V/m.")
    if [str(value) for value in cfg.analysis.context.policy_features] != [
        POLICY_FEATURE
    ]:
        raise ValueError("CDM2-D threshold selection may use only EEG alpha excess.")
    discovery = cfg.analysis.threshold_discovery
    if not (
        np.isclose(float(discovery.low_dose_v_per_m), 0.2)
        and np.isclose(float(discovery.high_dose_v_per_m), 0.4)
    ):
        raise ValueError("CDM2-D freezes low/high actions to 0.2/0.4 V/m.")
    design = cfg.analysis.crossed_design
    if not smoke_test and (
        int(design.n_structure_seeds) < 3 or int(design.n_history_seeds) < 2
    ):
        raise ValueError("CDM2-D requires >=3 structures crossed with >=2 histories.")
    if not smoke_test and int(design.n_future_continuations) < 3:
        raise ValueError("CDM2-D requires >=3 independent postdecision futures.")
    source_seeds = set()
    for context in source["provenance"].get("contexts", []):
        for name in (
            "trial_seed", "structure_seed", "history_seed", "phase_seed"
        ):
            if name in context:
                source_seeds.add(int(context[name]))
    source_seeds.update(
        int(value) for value in source["provenance"].get("future_drive_seeds", [])
    )
    current_seeds = set(_future_seeds(cfg))
    for context in _context_specs(cfg):
        current_seeds.update(int(context[name]) for name in (
            "trial_seed", "structure_seed", "history_seed", "phase_seed"
        ))
    if current_seeds.intersection(source_seeds):
        raise ValueError("CDM2-D seed namespaces overlap CDM1-S discovery seeds.")


def _threshold_table(
    expected: pd.DataFrame, summary: pd.DataFrame, *, cfg: DictConfig
) -> pd.DataFrame:
    low = float(cfg.analysis.threshold_discovery.low_dose_v_per_m)
    high = float(cfg.analysis.threshold_discovery.high_dose_v_per_m)
    result = summary.copy()
    for dose, name in ((low, "low"), (high, "high")):
        values = expected[np.isclose(expected.dose_v_per_m, dose)].set_index(
            "context_id"
        )["expected_post_distance_to_B_log10"]
        result[f"{name}_dose_v_per_m"] = dose
        result[f"{name}_expected_distance_to_B_log10"] = result.context_id.map(
            values
        )
    result["low_advantage_over_high_log10"] = (
        result.high_expected_distance_to_B_log10
        - result.low_expected_distance_to_B_log10
    )
    result["expected_optimal_binary_dose_v_per_m"] = np.where(
        result.low_expected_distance_to_B_log10
        <= result.high_expected_distance_to_B_log10,
        low,
        high,
    )
    result["expected_low_practically_better"] = (
        result.low_advantage_over_high_log10
        >= float(cfg.analysis.criteria.practical_advantage_log10)
    )
    return result


def _candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(values, dtype=float))
    if unique.size < 2:
        return np.empty(0, dtype=float)
    return (unique[:-1] + unique[1:]) / 2.0


def _fit_monotone_threshold(
    training: pd.DataFrame, *, cfg: DictConfig
) -> dict[str, Any] | None:
    low = float(cfg.analysis.threshold_discovery.low_dose_v_per_m)
    high = float(cfg.analysis.threshold_discovery.high_dose_v_per_m)
    minimum = int(
        cfg.analysis.threshold_discovery.minimum_training_contexts_per_action
    )
    candidates = []
    x = training[POLICY_FEATURE].to_numpy(float)
    for threshold in _candidate_thresholds(x):
        selected_low = x < threshold
        low_count = int(selected_low.sum())
        high_count = int((~selected_low).sum())
        if min(low_count, high_count) < minimum:
            continue
        distance = np.where(
            selected_low,
            training.low_expected_distance_to_B_log10.to_numpy(float),
            training.high_expected_distance_to_B_log10.to_numpy(float),
        )
        candidates.append({
            "threshold_log10_alpha_excess": float(threshold),
            "mean_selected_expected_distance_to_B_log10": float(distance.mean()),
            "low_action_count": low_count,
            "high_action_count": high_count,
            "minimum_arm_support": min(low_count, high_count),
        })
    if not candidates:
        return None
    candidates.sort(key=lambda row: (
        row["mean_selected_expected_distance_to_B_log10"],
        -row["minimum_arm_support"],
        row["threshold_log10_alpha_excess"],
    ))
    return candidates[0]


def _apply_threshold(
    frame: pd.DataFrame, *, threshold: float, cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray]:
    low = float(cfg.analysis.threshold_discovery.low_dose_v_per_m)
    high = float(cfg.analysis.threshold_discovery.high_dose_v_per_m)
    choose_low = frame[POLICY_FEATURE].to_numpy(float) < float(threshold)
    actions = np.where(choose_low, low, high)
    distances = np.where(
        choose_low,
        frame.low_expected_distance_to_B_log10.to_numpy(float),
        frame.high_expected_distance_to_B_log10.to_numpy(float),
    )
    return actions, distances


def _crossvalidated_threshold_policy(
    table: pd.DataFrame, *, cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high = float(cfg.analysis.threshold_discovery.high_dose_v_per_m)
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for heldout in sorted(table.structure_seed.unique()):
        train = table[table.structure_seed.ne(heldout)]
        test = table[table.structure_seed.eq(heldout)]
        model = _fit_monotone_threshold(train, cfg=cfg)
        if model is None:
            fold_rows.append({
                "heldout_structure_seed": int(heldout),
                "training_context_count": int(len(train)),
                "test_context_count": int(len(test)),
                "threshold_available": False,
            })
            continue
        actions, distances = _apply_threshold(
            test,
            threshold=float(model["threshold_log10_alpha_excess"]),
            cfg=cfg,
        )
        fixed = test.high_expected_distance_to_B_log10.to_numpy(float)
        fold_rows.append({
            "heldout_structure_seed": int(heldout),
            "training_context_count": int(len(train)),
            "test_context_count": int(len(test)),
            "threshold_available": True,
            **model,
            "mean_test_advantage_over_fixed_0p4_log10": float(
                np.mean(fixed - distances)
            ),
        })
        for index, (_, source) in enumerate(test.iterrows()):
            rows.append({
                "context_id": str(source.context_id),
                "heldout_structure_seed": int(heldout),
                "state_label": str(source.state_label),
                "history_seed": int(source.history_seed),
                POLICY_FEATURE: float(source[POLICY_FEATURE]),
                "training_threshold_log10_alpha_excess": float(
                    model["threshold_log10_alpha_excess"]
                ),
                "selected_dose_v_per_m": float(actions[index]),
                "selected_expected_distance_to_B_log10": float(distances[index]),
                "fixed_0p4_expected_distance_to_B_log10": float(fixed[index]),
                "advantage_over_fixed_0p4_log10": float(
                    fixed[index] - distances[index]
                ),
                "expected_optimal_binary_dose_v_per_m": float(
                    source.expected_optimal_binary_dose_v_per_m
                ),
                "matches_expected_binary_oracle": bool(np.isclose(
                    actions[index], source.expected_optimal_binary_dose_v_per_m
                )),
            })
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def _shuffle_null(
    table: pd.DataFrame, *, observed: float, cfg: DictConfig
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.experiment.seed), 4_000_037])
    )
    n_permutations = int(cfg.analysis.context_shuffle.n_permutations)
    rows = []
    original = table[POLICY_FEATURE].to_numpy(float)
    for permutation in range(n_permutations):
        shuffled = table.copy()
        shuffled[POLICY_FEATURE] = rng.permutation(original)
        policy, folds = _crossvalidated_threshold_policy(shuffled, cfg=cfg)
        if policy.empty or not bool(folds.threshold_available.all()):
            advantage = 0.0
        else:
            advantage = float(policy.advantage_over_fixed_0p4_log10.mean())
        rows.append({
            "permutation": permutation + 1,
            "mean_advantage_over_fixed_0p4_log10": advantage,
        })
    result = pd.DataFrame(rows)
    values = result.mean_advantage_over_fixed_0p4_log10.to_numpy(float)
    p_value = float(
        (1 + np.count_nonzero(values >= observed)) / (values.size + 1)
    )
    return result, p_value


def _evaluate(
    *,
    screening: pd.DataFrame,
    metrics: pd.DataFrame,
    expected: pd.DataFrame,
    table: pd.DataFrame,
    state: pd.DataFrame,
    policy: pd.DataFrame,
    folds: pd.DataFrame,
    candidate: dict[str, Any] | None,
    shuffle_p: float,
    oracle_audit: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any], pd.DataFrame]:
    criteria = cfg.analysis.criteria
    low = float(cfg.analysis.threshold_discovery.low_dose_v_per_m)
    high = float(cfg.analysis.threshold_discovery.high_dose_v_per_m)
    low_rows = table[np.isclose(table.expected_optimal_binary_dose_v_per_m, low)]
    action_counts = table.expected_optimal_binary_dose_v_per_m.value_counts()
    minimum_action_fraction = float(action_counts.min() / len(table))
    eligible_screen = screening[screening.eligible]
    state_means = state[state.eligible_context_count.gt(0)].sort_values(
        "modulation_depth"
    ).mean_context_log10_alpha_power.to_numpy(float)
    state_signal = float(np.ptp(state_means)) if state_means.size >= 2 else 0.0
    sem_noise = float(eligible_screen.context_alpha_temporal_sem_log10.median())
    split_half_difference = float(
        eligible_screen.context_alpha_split_half_difference_log10.median()
    )
    # Adjacent 2-s windows need not be independent. Avoid relying only on the
    # naive SD/sqrt(n) estimate by also using the empirical disagreement
    # between the two 6-s half-baselines (converted to one-half noise scale).
    mean_noise = max(sem_noise, split_half_difference / np.sqrt(2.0))
    state_snr = state_signal / max(mean_noise, np.finfo(float).eps)
    policy_available = bool(
        not folds.empty and len(folds) == table.structure_seed.nunique()
        and folds.threshold_available.all()
        and len(policy) == len(table)
    )
    cv_advantage = (
        float(policy.advantage_over_fixed_0p4_log10.mean())
        if policy_available else float("nan")
    )
    cv_actions = (
        sorted(policy.selected_dose_v_per_m.unique().tolist())
        if policy_available else []
    )
    structure = (
        policy.groupby("heldout_structure_seed", as_index=False).agg(
            context_count=("context_id", "size"),
            mean_advantage_over_fixed_0p4_log10=(
                "advantage_over_fixed_0p4_log10", "mean"
            ),
            oracle_match_fraction=("matches_expected_binary_oracle", "mean"),
        ) if policy_available else pd.DataFrame()
    )
    positive_structure_fraction = (
        float((structure.mean_advantage_over_fixed_0p4_log10 > 0.0).mean())
        if not structure.empty else 0.0
    )
    checks = {
        "source_CDM1_map_passed_and_frozen": True,
        "complete_crossed_screening_grid": len(screening) == len(_context_specs(cfg)),
        "screening_uses_only_predecision_eeg": bool(
            (~screening.screening_uses_stimulation_outcome).all()
        ),
        "screening_does_not_use_seed_specific_B": bool(
            (~screening.screening_uses_seed_specific_B).all()
        ),
        "twelve_second_stimulation_free_context": bool(
            (screening.context_window_count >= 6).all()
        ),
        "policy_uses_only_predecision_alpha_excess": True,
        "latent_state_and_hidden_spikes_excluded_from_policy": True,
        "state_generator_distinct_from_tacs_action": True,
        "afferent_mean_rate_matched_across_states_by_construction": True,
        "minimum_eligible_contexts": len(table)
        >= int(criteria.minimum_eligible_contexts),
        "minimum_eligible_state_levels": table.state_label.nunique()
        >= int(criteria.minimum_eligible_state_levels),
        "eligible_structure_coverage": table.structure_seed.nunique()
        >= int(criteria.minimum_eligible_structure_seeds),
        "eligible_history_coverage": table.history_seed.nunique()
        >= int(criteria.minimum_eligible_history_seeds),
        "multiple_independent_futures_per_context_action": bool(
            (expected.n_future_continuations
             >= int(criteria.minimum_future_continuations)).all()
        ),
        "identical_predecision_eeg_across_actions_and_futures": bool(
            metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
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
        "state_signal_exceeds_context_mean_noise": state_snr
        >= float(criteria.minimum_state_signal_to_context_mean_noise),
        "expected_oracle_uses_both_actions": set(
            table.expected_optimal_binary_dose_v_per_m.unique()
        ) == {low, high},
        "low_dose_opportunity_in_multiple_contexts": int(
            low_rows.expected_low_practically_better.sum()
        ) >= int(criteria.minimum_expected_low_dose_contexts),
        "low_dose_opportunity_across_structures": low_rows.loc[
            low_rows.expected_low_practically_better, "structure_seed"
        ].nunique() >= int(criteria.minimum_expected_low_dose_structures),
        "low_dose_opportunity_across_histories": low_rows.loc[
            low_rows.expected_low_practically_better, "history_seed"
        ].nunique() >= int(criteria.minimum_expected_low_dose_histories),
        "both_expected_actions_have_practical_support": minimum_action_fraction
        >= float(criteria.minimum_expected_action_fraction),
        "expected_oracle_has_mean_practical_advantage": float(
            oracle_audit["mean_expected_oracle_advantage_over_best_fixed_log10"]
        ) >= float(criteria.minimum_mean_expected_oracle_advantage_log10),
        "realized_optimum_reproducible_across_futures": float(
            oracle_audit["mean_realized_oracle_agreement_fraction"]
        ) >= float(criteria.minimum_realized_oracle_agreement_fraction),
        "full_discovery_threshold_available": candidate is not None,
        "all_leave_one_structure_out_thresholds_available": policy_available,
        "crossvalidated_rule_uses_both_actions": set(cv_actions) == {low, high},
        "crossvalidated_rule_beats_fixed_0p4": bool(
            np.isfinite(cv_advantage)
            and cv_advantage
            >= float(criteria.minimum_crossvalidated_advantage_over_best_fixed_log10)
        ),
        "crossvalidated_advantage_positive_across_structures": (
            positive_structure_fraction
            >= float(criteria.minimum_positive_structure_fraction)
        ),
        "alpha_context_beats_shuffled_context": bool(
            np.isfinite(cv_advantage)
            and shuffle_p <= float(criteria.maximum_context_shuffle_p_value)
        ),
    }
    primary = tuple(checks)
    passed = all(checks[name] for name in primary)
    conclusions = {
        "severity_threshold_discovery_gate_passed": passed,
        "candidate_ready_for_disjoint_confirmation": passed,
        "contextual_bandit_status": "not trained or tested",
        "screened_context_count": int(len(screening)),
        "eligible_context_count": int(len(table)),
        "screening_yield": float(screening.eligible.mean()),
        "state_signal_to_context_mean_noise_ratio": state_snr,
        "effective_context_mean_noise_log10": mean_noise,
        "median_context_alpha_sem_log10": sem_noise,
        "median_context_split_half_difference_log10": split_half_difference,
        "expected_low_dose_context_count": int(len(low_rows)),
        "practical_low_dose_context_count": int(
            low_rows.expected_low_practically_better.sum()
        ),
        "practical_low_dose_structure_count": int(low_rows.loc[
            low_rows.expected_low_practically_better, "structure_seed"
        ].nunique()),
        "practical_low_dose_history_count": int(low_rows.loc[
            low_rows.expected_low_practically_better, "history_seed"
        ].nunique()),
        "minimum_expected_action_fraction": minimum_action_fraction,
        "crossvalidated_selected_doses_v_per_m": cv_actions,
        "crossvalidated_mean_advantage_over_fixed_0p4_log10": cv_advantage,
        "crossvalidated_oracle_match_fraction": (
            float(policy.matches_expected_binary_oracle.mean())
            if policy_available else float("nan")
        ),
        "positive_structure_fraction": positive_structure_fraction,
        "context_shuffle_p_value": shuffle_p,
        **oracle_audit,
    }
    return checks, conclusions, structure


def _plot_results(
    *, root: Path, state: pd.DataFrame, table: pd.DataFrame,
    expected: pd.DataFrame, policy: pd.DataFrame,
    candidate: dict[str, Any] | None
) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    eligible = state[state.eligible_context_count.gt(0)]
    axis.plot(
        eligible.modulation_depth,
        eligible.mean_context_alpha_excess_log10,
        "o-", linewidth=2.0,
    )
    axis.axhline(0.0, color="0.35", linewidth=0.9)
    axis.set(
        xlabel="Latent afferent modulation depth (audit only)",
        ylabel="12-s EEG alpha excess over frozen B (log10)",
        title="CDM2-D predecision state observability",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_01_state_observability.png", dpi=250)
    plt.close(figure)

    active = expected[expected.dose_v_per_m.gt(0.0)]
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    for context_id, group in active.groupby("context_id", sort=False):
        axis.plot(
            group.dose_v_per_m,
            group.expected_post_distance_to_B_log10,
            "o-", alpha=0.45,
        )
    axis.set(
        xlabel="One constant intervention dose (V/m)",
        ylabel="Expected distance to frozen B (log10)",
        title="Future-averaged binary dose responses",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_02_binary_dose_responses.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.scatter(
        table[POLICY_FEATURE], table.low_advantage_over_high_log10,
        c=table.modulation_depth, cmap="viridis", s=70,
        edgecolor="black", linewidth=0.5,
    )
    axis.axhline(0.0, color="0.35", linewidth=0.9)
    if candidate is not None:
        axis.axvline(
            float(candidate["threshold_log10_alpha_excess"]),
            color="#D62728", linestyle="--", linewidth=1.5,
            label="full-discovery threshold",
        )
        axis.legend()
    axis.set(
        xlabel="Predecision EEG alpha excess (log10)",
        ylabel="0.4 distance − 0.2 distance (positive: 0.2 better)",
        title="Monotone EEG severity/action hypothesis",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_03_threshold_discovery.png", dpi=250)
    plt.close(figure)

    if not policy.empty:
        structure = policy.groupby("heldout_structure_seed")[
            "advantage_over_fixed_0p4_log10"
        ].mean()
        figure, axis = plt.subplots(figsize=(6.8, 4.5))
        axis.bar(
            structure.index.astype(str), structure.values,
            color=np.where(structure.values >= 0.0, "#2CA02C", "#D62728"),
        )
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.set(
            xlabel="Held-out structure seed",
            ylabel="Threshold-rule advantage over fixed 0.4 (log10)",
            title="Leave-one-structure-out prediction",
        )
        figure.tight_layout()
        figure.savefig(root / "figure_04_crossvalidated_policy.png", dpi=250)
        plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen = _load_frozen_candidate(cfg)
    source = _load_source_cdm1(cfg)
    _validate_design(cfg, frozen, source)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = (
        Path(to_absolute_path(str(cfg.experiment.dir)))
        / "severity_threshold_discovery"
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
        print("\n### CDM2-D expanded EEG-severity threshold discovery")
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
            context_values = _context_features(
                baseline_reference, target_model=target_model, cfg=cfg
            )
            screening.update(context_values)
            screening_rows.append(screening)
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

    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "predecision_screening_audit.csv", index=False)
    if not metric_rows:
        result = {
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "severity_threshold_discovery_gate_passed": False,
                "candidate_ready_for_disjoint_confirmation": False,
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No context passed the frozen predecision screen.",
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        print("\nNo eligible CDM2-D contexts; discovery: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    epochs = pd.DataFrame(epoch_rows)
    expected, context_summary, oracle_audit = _expected_action_map(metrics, cfg=cfg)
    table = _threshold_table(expected, context_summary, cfg=cfg)
    state = _state_summary(screening, context_summary, expected)
    candidate = _fit_monotone_threshold(table, cfg=cfg)
    policy, folds = _crossvalidated_threshold_policy(table, cfg=cfg)
    observed_advantage = (
        float(policy.advantage_over_fixed_0p4_log10.mean())
        if not policy.empty and bool(folds.threshold_available.all()) else 0.0
    )
    shuffle, shuffle_p = _shuffle_null(
        table, observed=observed_advantage, cfg=cfg
    )
    checks, conclusions, structure = _evaluate(
        screening=screening,
        metrics=metrics,
        expected=expected,
        table=table,
        state=state,
        policy=policy,
        folds=folds,
        candidate=candidate,
        shuffle_p=shuffle_p,
        oracle_audit=oracle_audit,
        cfg=cfg,
    )

    qualified = bool(conclusions["candidate_ready_for_disjoint_confirmation"])
    threshold_protocol = {
        "selection_stage": "CDM2_D_expanded_full_information_discovery",
        "qualified_for_disjoint_confirmation": qualified,
        "requires_disjoint_confirmation": True,
        "not_a_contextual_bandit": True,
        "policy_feature": POLICY_FEATURE,
        "policy_uses_only_predecision_ideal_eeg": True,
        "latent_state_and_hidden_variables_excluded": True,
        "rule": (
            "choose 0.2 V/m when predecision log10 alpha excess is below "
            "the frozen threshold; otherwise choose 0.4 V/m"
        ),
        "low_dose_v_per_m": float(
            cfg.analysis.threshold_discovery.low_dose_v_per_m
        ),
        "high_dose_v_per_m": float(
            cfg.analysis.threshold_discovery.high_dose_v_per_m
        ),
        "threshold": candidate,
        "crossvalidated_mean_advantage_over_fixed_0p4_log10": conclusions[
            "crossvalidated_mean_advantage_over_fixed_0p4_log10"
        ],
        "context_shuffle_p_value": shuffle_p,
    }

    epochs.to_csv(root / "epoch_eeg_and_hidden_metrics.csv", index=False)
    metrics.to_csv(root / "future_action_metrics.csv", index=False)
    expected.to_csv(root / "conditional_expected_binary_dose_map.csv", index=False)
    table.to_csv(root / "context_threshold_discovery_table.csv", index=False)
    state.to_csv(root / "state_observability_summary.csv", index=False)
    policy.to_csv(root / "crossvalidated_threshold_policy.csv", index=False)
    folds.to_csv(root / "crossvalidation_fold_thresholds.csv", index=False)
    structure.to_csv(root / "structure_level_policy_comparison.csv", index=False)
    shuffle.to_csv(root / "alpha_context_shuffle_null.csv", index=False)
    (root / "candidate_threshold_protocol.json").write_text(
        json.dumps(_plain(threshold_protocol), indent=2)
    )
    provenance = {
        **frozen,
        "source_CDM1_result_dir": source["root"],
        "source_CDM1_sha256": source["sha256"],
        "states": [
            {"label": label, "modulation_depth": depth}
            for label, depth in zip(
                [str(value) for value in cfg.analysis.states.labels],
                [float(value) for value in cfg.analysis.states.modulation_depths],
            )
        ],
        "state_labels_available_to_policy": False,
        "actions_v_per_m": _doses(cfg),
        "policy_feature": POLICY_FEATURE,
        "active_probe_used": False,
        "one_action_selected_once_per_intervention": True,
        "postdecision_future_seeds": _future_seeds(cfg),
        "contexts": _context_specs(cfg),
    }
    (root / "protocol_provenance.json").write_text(
        json.dumps(_plain(provenance), indent=2)
    )
    result = {
        "scope": "ideal neural-only EEG, expanded severity-rule discovery",
        "checks": checks,
        "conclusions": conclusions,
        "candidate_threshold_protocol": threshold_protocol,
        "primary_endpoint": (
            "future-averaged absolute log10 alpha-power distance to frozen B"
        ),
        "primary_comparator": "frozen fixed 0.4-V/m single action",
        "statistical_unit": "circuit structure; histories and futures are repeats",
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "A pass qualifies one monotone EEG-alpha severity threshold for a "
            "new disjoint confirmation. It does not establish closed-loop or "
            "contextual-bandit superiority by itself."
        ),
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_plain(result), indent=2)
    )
    if bool(cfg.experiment.plot):
        _plot_results(
            root=root, state=state, table=table,
            expected=expected, policy=policy, candidate=candidate,
        )

    print("\n### CDM2-D screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### CDM2-D discovery checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nEEG-severity threshold discovery:",
        "PASSED" if conclusions["severity_threshold_discovery_gate_passed"]
        else "NOT PASSED",
    )
    print(
        "Candidate ready for disjoint confirmation:",
        "YES" if qualified else "NO",
    )
    print("Contextual bandit status: NOT TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
