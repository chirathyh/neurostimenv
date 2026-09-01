"""CDM2-C disjoint confirmation of the frozen EEG-severity dose rule.

The complete CDM2-D rule is loaded by hash and applied without refitting:

    predecision log10 alpha excess < 0.380131... -> 0.2 V/m
    otherwise                                  -> 0.4 V/m

Circuit structure is the independent statistical unit. Two histories and
three postdecision futures estimate each structure's conditional response but
are never counted as independent samples. The single primary contrast is the
frozen rule versus fixed 0.4 V/m, tested with an exact structure-level sign-
flip randomization test and a prespecified practical-effect threshold.
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
from scipy import stats


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
from experiments.ballnstick_analysis.run_ballnstick_severity_threshold_discovery import (  # noqa: E402
    POLICY_FEATURE,
    _threshold_table,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_discovery(cfg: DictConfig) -> dict[str, Any]:
    root = Path(to_absolute_path(str(cfg.analysis.frozen_discovery.result_dir)))
    paths = {
        "conclusion": root / "experiment_conclusion.json",
        "candidate": root / "candidate_threshold_protocol.json",
        "provenance": root / "protocol_provenance.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CDM2-C requires the completed CDM2-D discovery: " + ", ".join(missing)
        )
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected_hashes = OmegaConf.to_container(
        cfg.analysis.frozen_discovery.expected_sha256, resolve=True
    )
    for name, expected in expected_hashes.items():
        if hashes[str(name)] != str(expected):
            raise ValueError(
                f"Frozen CDM2-D {name!r} changed after confirmation was specified."
            )

    conclusion = json.loads(paths["conclusion"].read_text())
    candidate = json.loads(paths["candidate"].read_text())
    provenance = json.loads(paths["provenance"].read_text())
    if not bool(conclusion["conclusions"]["severity_threshold_discovery_gate_passed"]):
        raise ValueError("The source CDM2-D discovery gate did not pass.")
    if not bool(candidate["qualified_for_disjoint_confirmation"]):
        raise ValueError("The source CDM2-D candidate was not qualified.")

    threshold = float(candidate["threshold"]["threshold_log10_alpha_excess"])
    expected = cfg.analysis.frozen_discovery
    if not np.isclose(threshold, float(expected.expected_threshold_log10_alpha_excess)):
        raise ValueError("The frozen CDM2-D EEG threshold changed.")
    if not np.isclose(
        float(candidate["low_dose_v_per_m"]), float(expected.expected_low_dose_v_per_m)
    ):
        raise ValueError("The frozen CDM2-D low action changed.")
    if not np.isclose(
        float(candidate["high_dose_v_per_m"]),
        float(expected.expected_high_dose_v_per_m),
    ):
        raise ValueError("The frozen CDM2-D high action changed.")
    if str(candidate["policy_feature"]) != POLICY_FEATURE:
        raise ValueError("The frozen CDM2-D policy feature changed.")
    return {
        "root": str(root),
        "sha256": hashes,
        "conclusion": conclusion,
        "candidate": candidate,
        "provenance": provenance,
        "threshold_log10_alpha_excess": threshold,
    }


def _one_sample_t_power(*, n: int, effect_size: float, alpha: float) -> float:
    """One-sided one-sample/paired t-test power under standardized effect d_z."""
    if n < 2 or effect_size <= 0.0 or not 0.0 < alpha < 1.0:
        return 0.0
    critical = stats.t.ppf(1.0 - alpha, df=n - 1)
    return float(stats.nct.sf(critical, df=n - 1, nc=effect_size * np.sqrt(n)))


def _required_n_one_sample_t(
    *, effect_size: float, alpha: float, target_power: float, maximum_n: int = 10000
) -> int:
    for n in range(2, maximum_n + 1):
        if _one_sample_t_power(n=n, effect_size=effect_size, alpha=alpha) >= target_power:
            return n
    raise ValueError("The requested power was not attained within maximum_n.")


def _current_seed_set(cfg: DictConfig) -> set[int]:
    values = set(_future_seeds(cfg))
    for context in _context_specs(cfg):
        for name in ("trial_seed", "structure_seed", "history_seed", "phase_seed"):
            values.add(int(context[name]))
    return values


def _source_seed_set(discovery: dict[str, Any]) -> set[int]:
    values = {
        int(value)
        for value in discovery["provenance"].get("postdecision_future_seeds", [])
    }
    for context in discovery["provenance"].get("contexts", []):
        for name in ("trial_seed", "structure_seed", "history_seed", "phase_seed"):
            if name in context:
                values.add(int(context[name]))
    return values


def _validate_confirmation_design(
    cfg: DictConfig,
    frozen_candidate: dict[str, Any],
    discovery: dict[str, Any],
) -> dict[str, Any]:
    _validate_cdm1_design(cfg, frozen_candidate)
    smoke_test = bool(cfg.analysis.smoke_test)
    window_ms = float(cfg.env.simulation.obs_win_len)
    baseline_ms = int(cfg.analysis.timeline.baseline_steps) * window_ms
    if baseline_ms < 2.0 * float(cfg.analysis.context.window_ms):
        raise ValueError(
            "CDM2-C requires at least two complete context windows, including smoke tests."
        )
    if not smoke_test and baseline_ms < 12_000.0:
        raise ValueError("CDM2-C requires at least 12 s of stimulation-free EEG.")
    if [float(value) for value in cfg.analysis.actions.active_doses_v_per_m] != [
        0.2,
        0.4,
    ]:
        raise ValueError("CDM2-C freezes the active action set to {0.2, 0.4} V/m.")
    if [str(value) for value in cfg.analysis.context.policy_features] != [POLICY_FEATURE]:
        raise ValueError("CDM2-C may use only predecision EEG alpha excess.")
    if not (
        np.isclose(float(cfg.analysis.threshold_discovery.low_dose_v_per_m), 0.2)
        and np.isclose(float(cfg.analysis.threshold_discovery.high_dose_v_per_m), 0.4)
    ):
        raise ValueError("CDM2-C shared table aliases must remain 0.2/0.4 V/m.")

    power = cfg.analysis.power_design
    alpha = float(power.alpha_one_sided)
    target = float(power.target_power)
    effect = float(power.minimum_standardized_effect_dz)
    minimum_mean = float(power.minimum_mean_advantage_log10)
    anticipated_sd = float(power.anticipated_structure_sd_log10)
    if not np.isclose(effect, minimum_mean / anticipated_sd):
        raise ValueError("CDM2-C raw and standardized power assumptions disagree.")
    if not np.isclose(
        minimum_mean,
        float(cfg.analysis.criteria.minimum_practical_mean_advantage_log10),
    ):
        raise ValueError("CDM2-C power effect and practical primary effect disagree.")
    planned = int(power.planned_independent_structures)
    required = _required_n_one_sample_t(
        effect_size=effect, alpha=alpha, target_power=target
    )
    configured = int(cfg.analysis.crossed_design.n_structure_seeds)
    if not smoke_test and configured != planned:
        raise ValueError("Configured structures must equal the frozen power design.")
    if not smoke_test and planned < required:
        raise ValueError(
            f"Power design requires at least {required} structures, not {planned}."
        )
    if not smoke_test and int(cfg.analysis.crossed_design.n_history_seeds) < 2:
        raise ValueError("CDM2-C requires at least two histories per structure.")
    if not smoke_test and int(cfg.analysis.crossed_design.n_future_continuations) < 3:
        raise ValueError("CDM2-C requires at least three postdecision futures.")

    overlap = _current_seed_set(cfg).intersection(_source_seed_set(discovery))
    if overlap:
        raise ValueError(
            "CDM2-C seeds overlap frozen CDM2-D discovery seeds: "
            + ", ".join(str(value) for value in sorted(overlap))
        )
    return {
        "alpha_one_sided": alpha,
        "target_power": target,
        "minimum_mean_advantage_log10": minimum_mean,
        "anticipated_structure_sd_log10": anticipated_sd,
        "minimum_standardized_effect_dz": effect,
        "planned_independent_structures": planned,
        "required_independent_structures": required,
        "a_priori_t_approximation_power": _one_sample_t_power(
            n=planned, effect_size=effect, alpha=alpha
        ),
        "power_unit": "circuit structure",
        "histories_and_futures_count_as_repeats": True,
    }


def _frozen_policy_table(
    table: pd.DataFrame, *, threshold: float, cfg: DictConfig
) -> pd.DataFrame:
    low = float(cfg.analysis.frozen_discovery.expected_low_dose_v_per_m)
    high = float(cfg.analysis.frozen_discovery.expected_high_dose_v_per_m)
    primary = float(cfg.analysis.inference.primary_comparator_dose_v_per_m)
    secondary = float(cfg.analysis.inference.secondary_comparator_dose_v_per_m)
    if not np.isclose(primary, high) or not np.isclose(secondary, low):
        raise ValueError("CDM2-C comparator doses must remain frozen at 0.4 and 0.2 V/m.")
    result = table.copy()
    choose_low = result[POLICY_FEATURE].to_numpy(float) < float(threshold)
    result["frozen_threshold_log10_alpha_excess"] = float(threshold)
    result["selected_dose_v_per_m"] = np.where(choose_low, low, high)
    result["selected_expected_distance_to_B_log10"] = np.where(
        choose_low,
        result.low_expected_distance_to_B_log10.to_numpy(float),
        result.high_expected_distance_to_B_log10.to_numpy(float),
    )
    result["fixed_0p4_expected_distance_to_B_log10"] = (
        result.high_expected_distance_to_B_log10
    )
    result["fixed_0p2_expected_distance_to_B_log10"] = (
        result.low_expected_distance_to_B_log10
    )
    result["primary_advantage_over_fixed_0p4_log10"] = (
        result.fixed_0p4_expected_distance_to_B_log10
        - result.selected_expected_distance_to_B_log10
    )
    result["secondary_advantage_over_fixed_0p2_log10"] = (
        result.fixed_0p2_expected_distance_to_B_log10
        - result.selected_expected_distance_to_B_log10
    )
    result["matches_expected_binary_oracle"] = np.isclose(
        result.selected_dose_v_per_m,
        result.expected_optimal_binary_dose_v_per_m,
    )
    return result


def _structure_summary(policy: pd.DataFrame) -> pd.DataFrame:
    return policy.groupby("structure_seed", as_index=False).agg(
        eligible_context_count=("context_id", "size"),
        mean_policy_distance_to_B_log10=(
            "selected_expected_distance_to_B_log10",
            "mean",
        ),
        mean_fixed_0p4_distance_to_B_log10=(
            "fixed_0p4_expected_distance_to_B_log10",
            "mean",
        ),
        mean_fixed_0p2_distance_to_B_log10=(
            "fixed_0p2_expected_distance_to_B_log10",
            "mean",
        ),
        primary_advantage_over_fixed_0p4_log10=(
            "primary_advantage_over_fixed_0p4_log10",
            "mean",
        ),
        secondary_advantage_over_fixed_0p2_log10=(
            "secondary_advantage_over_fixed_0p2_log10",
            "mean",
        ),
        low_action_fraction=(
            "selected_dose_v_per_m",
            lambda values: float(np.mean(np.isclose(values, 0.2))),
        ),
        oracle_match_fraction=("matches_expected_binary_oracle", "mean"),
    )


def _exact_sign_flip_p_value(
    values: np.ndarray,
    *,
    maximum_exact_n: int,
    monte_carlo_samples: int,
    rng: np.random.Generator,
) -> tuple[float, str, int]:
    """One-sided randomization p-value for a positive mean paired advantage."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        return float("nan"), "unavailable", 0
    observed = float(values.mean())
    n = int(values.size)
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 16.0
    if n <= int(maximum_exact_n):
        indices = np.arange(1 << n, dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(n, dtype=np.uint64)) & np.uint64(1)
        signs = np.where(bits == 0, -1.0, 1.0)
        null = (signs @ values) / n
        p_value = float(np.mean(null >= observed - tolerance))
        return p_value, "exact", int(null.size)
    signs = rng.choice((-1.0, 1.0), size=(int(monte_carlo_samples), n))
    null = (signs @ values) / n
    p_value = float((1 + np.count_nonzero(null >= observed - tolerance)) / (len(null) + 1))
    return p_value, "monte_carlo", int(len(null))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(int(n_resamples), values.size), replace=True)
    means = samples.mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return tuple(float(value) for value in np.quantile(means, [tail, 1.0 - tail]))


def _paired_inference(
    structure: pd.DataFrame, *, power_design: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    values = structure.primary_advantage_over_fixed_0p4_log10.to_numpy(float)
    n = int(values.size)
    mean = float(values.mean()) if n else float("nan")
    sd = float(values.std(ddof=1)) if n >= 2 else float("nan")
    se = sd / np.sqrt(n) if n >= 2 else float("nan")
    dz = mean / sd if n >= 2 and sd > 0.0 else float("nan")
    if n >= 2 and se > 0.0:
        statistic = mean / se
        t_one_sided_p = float(stats.t.sf(statistic, df=n - 1))
        ci95 = (
            mean - float(stats.t.ppf(0.975, n - 1)) * se,
            mean + float(stats.t.ppf(0.975, n - 1)) * se,
        )
        ci90 = (
            mean - float(stats.t.ppf(0.95, n - 1)) * se,
            mean + float(stats.t.ppf(0.95, n - 1)) * se,
        )
    else:
        statistic, t_one_sided_p = float("nan"), float("nan")
        ci95 = ci90 = (float("nan"), float("nan"))

    seed = int(cfg.experiment.seed) + int(cfg.analysis.inference.random_seed_offset)
    rng = np.random.default_rng(seed)
    sign_flip_p, sign_flip_method, sign_flip_samples = _exact_sign_flip_p_value(
        values,
        maximum_exact_n=int(cfg.analysis.inference.exact_sign_flip_max_structures),
        monte_carlo_samples=int(cfg.analysis.inference.monte_carlo_sign_flips),
        rng=rng,
    )
    bootstrap95 = _bootstrap_mean_ci(
        values,
        confidence=0.95,
        n_resamples=int(cfg.analysis.inference.bootstrap_resamples),
        rng=rng,
    )
    try:
        wilcoxon = stats.wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        )
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_statistic = wilcoxon_p = float("nan")

    return {
        "independent_structure_count": n,
        "mean_primary_advantage_log10": mean,
        "sd_primary_advantage_log10": sd,
        "se_primary_advantage_log10": se,
        "paired_standardized_effect_dz": dz,
        "positive_structure_count": int(np.count_nonzero(values > 0.0)),
        "zero_structure_count": int(np.count_nonzero(np.isclose(values, 0.0))),
        "positive_structure_fraction": float(np.mean(values > 0.0)) if n else 0.0,
        "paired_t_statistic": statistic,
        "paired_t_one_sided_p_value": t_one_sided_p,
        "t_interval_90_log10": [float(ci90[0]), float(ci90[1])],
        "t_interval_95_log10": [float(ci95[0]), float(ci95[1])],
        "structure_bootstrap_interval_95_log10": [
            float(bootstrap95[0]),
            float(bootstrap95[1]),
        ],
        "primary_sign_flip_one_sided_p_value": sign_flip_p,
        "primary_sign_flip_method": sign_flip_method,
        "primary_sign_flip_samples": sign_flip_samples,
        "wilcoxon_signed_rank_statistic": wilcoxon_statistic,
        "wilcoxon_one_sided_p_value": wilcoxon_p,
        "a_priori_power_design": power_design,
        "multiplicity": "one prespecified primary contrast; secondary tests are audits",
    }


def _evaluate_confirmation(
    *,
    screening: pd.DataFrame,
    metrics: pd.DataFrame,
    policy: pd.DataFrame,
    structure: pd.DataFrame,
    inference: dict[str, Any],
    power_design: dict[str, Any],
    seeds_disjoint: bool,
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    action_counts = policy.selected_dose_v_per_m.value_counts(normalize=True)
    minimum_action_fraction = float(action_counts.min()) if len(action_counts) else 0.0
    exact_p = float(inference["primary_sign_flip_one_sided_p_value"])
    mean_advantage = float(inference["mean_primary_advantage_log10"])
    positive_fraction = float(inference["positive_structure_fraction"])
    checks = {
        "frozen_discovery_passed_and_hash_locked": True,
        "confirmation_seeds_disjoint_from_discovery": bool(seeds_disjoint),
        "threshold_loaded_without_refitting": True,
        "a_priori_structure_sample_size_powered": bool(
            int(power_design["planned_independent_structures"])
            >= int(power_design["required_independent_structures"])
            and float(power_design["a_priori_t_approximation_power"])
            >= float(power_design["target_power"])
        ),
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
        "minimum_eligible_contexts": len(policy) >= int(criteria.minimum_eligible_contexts),
        "minimum_eligible_state_levels": policy.state_label.nunique()
        >= int(criteria.minimum_eligible_state_levels),
        "minimum_analyzable_structure_seeds": len(structure)
        >= int(criteria.minimum_analyzable_structure_seeds),
        "minimum_eligible_history_seeds": policy.history_seed.nunique()
        >= int(criteria.minimum_eligible_history_seeds),
        "multiple_independent_futures_per_context_action": bool(
            metrics.groupby(["context_id", "dose_v_per_m"]).future_index.nunique().min()
            >= int(criteria.minimum_future_continuations)
        ),
        "identical_predecision_eeg_across_actions_and_futures": bool(
            metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "policy_uses_only_predecision_alpha_excess": True,
        "latent_state_and_hidden_spikes_excluded_from_policy": True,
        "single_constant_action_per_intervention": bool(
            metrics.one_action_for_complete_intervention.all()
        ),
        "action_phase_tracks_predecision_eeg": bool(
            metrics.action_phase_tracking_error_rad.max()
            <= float(criteria.maximum_phase_tracking_error_rad)
        ),
        "frozen_rule_uses_both_actions": set(policy.selected_dose_v_per_m.unique())
        == {0.2, 0.4},
        "both_actions_have_minimum_support": minimum_action_fraction
        >= float(criteria.minimum_selected_action_fraction),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "field_removal_recovered": bool(metrics.field_removal_recovered.all()),
        "primary_mean_advantage_is_practically_meaningful": mean_advantage
        >= float(criteria.minimum_practical_mean_advantage_log10),
        "primary_exact_structure_test_rejects_null": bool(
            np.isfinite(exact_p)
            and exact_p <= float(criteria.maximum_primary_one_sided_p_value)
        ),
        "primary_advantage_consistent_across_structures": positive_fraction
        >= float(criteria.minimum_positive_structure_fraction),
    }
    confirmed = all(checks.values())
    conclusions = {
        "frozen_severity_rule_confirmed": confirmed,
        "candidate_ready_for_contextual_bandit": confirmed,
        "contextual_bandit_status": "not trained or tested",
        "screened_context_count": int(len(screening)),
        "eligible_context_count": int(len(policy)),
        "screening_yield": float(screening.eligible.mean()),
        "selected_action_counts": {
            f"{float(dose):g}_v_per_m": int(count)
            for dose, count in policy.selected_dose_v_per_m.value_counts().sort_index().items()
        },
        "minimum_selected_action_fraction": minimum_action_fraction,
        "mean_policy_distance_to_B_log10": float(
            policy.selected_expected_distance_to_B_log10.mean()
        ),
        "mean_fixed_0p4_distance_to_B_log10": float(
            policy.fixed_0p4_expected_distance_to_B_log10.mean()
        ),
        "mean_fixed_0p2_distance_to_B_log10": float(
            policy.fixed_0p2_expected_distance_to_B_log10.mean()
        ),
        "mean_secondary_advantage_over_fixed_0p2_log10": float(
            structure.secondary_advantage_over_fixed_0p2_log10.mean()
        ),
        "mean_oracle_match_fraction": float(policy.matches_expected_binary_oracle.mean()),
        **inference,
    }
    return checks, conclusions


def _plot_results(
    *,
    root: Path,
    screening: pd.DataFrame,
    policy: pd.DataFrame,
    structure: pd.DataFrame,
    threshold: float,
) -> None:
    eligible = screening[screening.eligible]
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    for label, group in eligible.groupby("state_label", sort=False):
        axis.scatter(
            group.structure_seed,
            group.context_alpha_excess_log10,
            label=str(label),
            alpha=0.75,
        )
    axis.axhline(threshold, color="#D62728", linestyle="--", label="frozen threshold")
    axis.set(
        xlabel="Independent circuit structure seed",
        ylabel="Predecision EEG alpha excess (log10)",
        title="CDM2-C disjoint context observations",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "figure_01_disjoint_contexts.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(len(structure))
    axis.plot(
        x,
        structure.mean_fixed_0p4_distance_to_B_log10,
        "o-",
        label="fixed 0.4 V/m",
    )
    axis.plot(
        x,
        structure.mean_policy_distance_to_B_log10,
        "o-",
        label="frozen EEG threshold",
    )
    axis.set_xticks(x, structure.structure_seed.astype(str), rotation=45)
    axis.set(
        xlabel="Independent circuit structure",
        ylabel="Expected distance to frozen B (log10; lower is better)",
        title="Paired primary comparison",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "figure_02_paired_policy_vs_fixed.png", dpi=250)
    plt.close(figure)

    values = structure.primary_advantage_over_fixed_0p4_log10.to_numpy(float)
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.bar(
        structure.structure_seed.astype(str),
        values,
        color=np.where(values > 0.0, "#2CA02C", "#D62728"),
    )
    axis.axhline(0.0, color="0.25", linewidth=0.9)
    axis.set(
        xlabel="Independent circuit structure",
        ylabel="Frozen-rule advantage over fixed 0.4 (log10)",
        title="Structure-level confirmation effects",
    )
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(root / "figure_03_structure_effects.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    colors = np.where(np.isclose(policy.selected_dose_v_per_m, 0.2), "#1F77B4", "#FF7F0E")
    axis.scatter(
        policy.context_alpha_excess_log10,
        policy.primary_advantage_over_fixed_0p4_log10,
        c=colors,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.3,
    )
    axis.axvline(threshold, color="#D62728", linestyle="--")
    axis.axhline(0.0, color="0.35", linewidth=0.9)
    axis.set(
        xlabel="Predecision EEG alpha excess (log10)",
        ylabel="Frozen-rule advantage over fixed 0.4 (log10)",
        title="Frozen decision boundary and paired outcome",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_04_frozen_rule_outcomes.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen_candidate = _load_frozen_candidate(cfg)
    discovery = _load_frozen_discovery(cfg)
    power_design = _validate_confirmation_design(cfg, frozen_candidate, discovery)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = (
        Path(to_absolute_path(str(cfg.experiment.dir)))
        / "severity_threshold_confirmation"
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
        print("\n### CDM2-C frozen EEG-severity rule confirmation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
        print("\n### A priori power design")
        print(json.dumps(power_design, indent=2))
    comm.Barrier()

    started = time.perf_counter()
    target_model = frozen_candidate["target"]["target_model"]
    threshold = float(discovery["threshold_log10_alpha_excess"])
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
                        epoch_rows.append(
                            {
                                **row,
                                **context,
                                "future_index": future_index + 1,
                                "future_drive_seed": int(future_seed),
                                "dose_v_per_m": float(dose),
                            }
                        )
            if rank == 0:
                metric_rows.extend(
                    _replay_rows(
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
                    )
                )
                del episodes

    if rank != 0:
        return

    screening_frame = pd.DataFrame(screening_rows)
    screening_frame.to_csv(root / "predecision_screening_audit.csv", index=False)
    if not metric_rows:
        result = {
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "frozen_severity_rule_confirmed": False,
                "candidate_ready_for_contextual_bandit": False,
            },
            "a_priori_power_design": power_design,
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No context passed the frozen predecision screen.",
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        print("\nNo eligible CDM2-C contexts; confirmation: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    epochs = pd.DataFrame(epoch_rows)
    expected, context_summary, oracle_audit = _expected_action_map(metrics, cfg=cfg)
    table = _threshold_table(expected, context_summary, cfg=cfg)
    policy = _frozen_policy_table(table, threshold=threshold, cfg=cfg)
    state = _state_summary(screening_frame, context_summary, expected)
    structure = _structure_summary(policy)
    paired = _paired_inference(structure, power_design=power_design, cfg=cfg)
    seeds_disjoint = not bool(_current_seed_set(cfg).intersection(_source_seed_set(discovery)))
    checks, conclusions = _evaluate_confirmation(
        screening=screening_frame,
        metrics=metrics,
        policy=policy,
        structure=structure,
        inference=paired,
        power_design=power_design,
        seeds_disjoint=seeds_disjoint,
        cfg=cfg,
    )

    epochs.to_csv(root / "epoch_eeg_and_hidden_metrics.csv", index=False)
    metrics.to_csv(root / "future_action_metrics.csv", index=False)
    expected.to_csv(root / "conditional_expected_binary_dose_map.csv", index=False)
    table.to_csv(root / "context_confirmation_table.csv", index=False)
    policy.to_csv(root / "frozen_threshold_policy_outcomes.csv", index=False)
    state.to_csv(root / "state_observability_summary.csv", index=False)
    structure.to_csv(root / "structure_level_primary_contrast.csv", index=False)
    (root / "statistical_inference.json").write_text(
        json.dumps(_plain(paired), indent=2)
    )
    frozen_protocol = {
        "selection_stage": "CDM2_D_expanded_full_information_discovery",
        "confirmation_stage": "CDM2_C_seed_disjoint_frozen_confirmation",
        "threshold_refitted": False,
        "policy_feature": POLICY_FEATURE,
        "threshold_log10_alpha_excess": threshold,
        "rule": discovery["candidate"]["rule"],
        "low_dose_v_per_m": 0.2,
        "high_dose_v_per_m": 0.4,
        "one_constant_action_per_intervention": True,
        "primary_endpoint": "future-averaged absolute log10 alpha-power distance to frozen B",
        "primary_comparator": "fixed 0.4 V/m",
        "independent_statistical_unit": "circuit structure",
    }
    (root / "frozen_confirmation_protocol.json").write_text(
        json.dumps(_plain(frozen_protocol), indent=2)
    )
    provenance = {
        **frozen_candidate,
        "source_CDM2_D_result_dir": discovery["root"],
        "source_CDM2_D_sha256": discovery["sha256"],
        "source_CDM2_D_candidate": discovery["candidate"],
        "confirmation_seeds_disjoint": seeds_disjoint,
        "power_design": power_design,
        "states": [
            {"label": label, "modulation_depth": depth}
            for label, depth in zip(
                [str(value) for value in cfg.analysis.states.labels],
                [float(value) for value in cfg.analysis.states.modulation_depths],
            )
        ],
        "state_labels_available_to_policy": False,
        "actions_v_per_m": doses,
        "policy_feature": POLICY_FEATURE,
        "active_probe_used": False,
        "one_action_selected_once_per_intervention": True,
        "postdecision_future_seeds": future_seeds,
        "contexts": contexts,
    }
    (root / "protocol_provenance.json").write_text(
        json.dumps(_plain(provenance), indent=2)
    )
    result = {
        "scope": "ideal neural-only EEG, frozen severity-rule confirmation",
        "checks": checks,
        "conclusions": conclusions,
        "frozen_protocol": frozen_protocol,
        "a_priori_power_design": power_design,
        "oracle_audit_secondary_only": oracle_audit,
        "primary_endpoint": frozen_protocol["primary_endpoint"],
        "primary_comparator": frozen_protocol["primary_comparator"],
        "statistical_unit": "circuit structure; histories and futures are repeats",
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "Confirmation requires a practically meaningful frozen-rule advantage, "
            "one-sided exact structure-level evidence, and cross-structure consistency. "
            "A pass permits a subsequent contextual-bandit experiment; it is not itself "
            "a trained or tested bandit."
        ),
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_plain(result), indent=2)
    )
    if bool(cfg.experiment.plot):
        _plot_results(
            root=root,
            screening=screening_frame,
            policy=policy,
            structure=structure,
            threshold=threshold,
        )

    print("\n### CDM2-C screening")
    print(f"contexts screened: {len(screening_frame)}")
    print(f"eligible contexts: {int(screening_frame.eligible.sum())}")
    print(f"screening yield: {float(screening_frame.eligible.mean()):.3f}")
    print("\n### CDM2-C confirmation checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Primary structure-level inference")
    print(json.dumps(_plain(paired), indent=2))
    print(
        "\nFrozen EEG-severity rule:",
        "CONFIRMED" if conclusions["frozen_severity_rule_confirmed"] else "NOT CONFIRMED",
    )
    print(
        "Candidate ready for contextual bandit:",
        "YES" if conclusions["candidate_ready_for_contextual_bandit"] else "NO",
    )
    print("Contextual bandit status: NOT TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
