"""H5-O1S: B-only rhythm-presence calibration and held-out EEG screening.

The spatial measurement and population-B power target remain frozen. The only
new fitted scalar is an upper null cutoff for the existing multitaper spectral
evidence. All neural trajectories are zero-field; no policy is trained.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import OmegaConf, open_dict
from scipy.stats import beta, binom

sys.path.insert(1, config("MAIN_PATH"))
from experiments.ballnstick_analysis import (
    run_ballnstick_h5_spatial_measurement_audit as spatial,
)


SCORE = "maximum_residual_evidence_db"
INPUTS = (*spatial.SCREEN_INPUT_FIELDS, SCORE)
SOURCE_FILES = {
    "conclusion": "experiment_conclusion.json",
    "completion": "run_complete.json",
    "protocol": "frozen_measurement_protocol.json",
    "target": "frozen_EEG_only_B_targets.json",
    "config": "resolved_config.yaml",
    "trajectories": "trajectory_audit.csv",
    "measurements": "evaluation_measurements.csv",
    "provenance": "protocol_and_provenance.json",
}


def _load_source(cfg):
    root = Path(to_absolute_path(str(cfg.analysis.source_h5o1m.result_dir)))
    files, hashes = spatial._hash_locked_files(
        root, SOURCE_FILES, cfg.analysis.source_h5o1m.expected_sha256
    )
    conclusion = json.loads(files["conclusion"].read_text())
    if conclusion["status"] != "NOT PASSED" or conclusion["failed_checks"] != [
        "screen_specificity"
    ]:
        raise ValueError(
            "H5-O1S requires the exact H5-O1M screening-specificity failure."
        )
    if (
        conclusion["smoke_test"]
        or not json.loads(files["completion"].read_text())["completed"]
    ):
        raise ValueError("The source must be a completed full experiment.")
    seeds = set()
    for name in ("trajectories", "measurements"):
        frame = pd.read_csv(files[name])
        for key in (
            "structure_seed",
            "history_seed",
            "phase_seed",
            "trial_seed",
            "future_drive_seed",
        ):
            seeds.update(frame[key].astype(int))
        if "sensor_noise_seeds" in frame:
            for value in frame.sensor_noise_seeds.unique():
                seeds.update(np.asarray(json.loads(value), int).reshape(-1))
    return {
        "root": str(root),
        "hashes": hashes,
        "seed_union": seeds,
        "config": OmegaConf.load(files["config"]),
        "target": json.loads(files["target"].read_text()),
        "provenance": json.loads(files["provenance"].read_text()),
    }


def _power_design(cfg):
    p = cfg.analysis.rhythm_screen
    n = int(cfg.analysis.measurement_design.evaluation_structures)
    critical = next(
        (
            k
            for k in range(n + 1)
            if binom.sf(k - 1, n, float(p.primary_specificity_null))
            <= float(p.alpha_one_sided)
        ),
        n + 1,
    )
    return {
        "unit": "independent B structure with no phenotype-positive measurement view",
        "planned_structures": n,
        "critical_clean_structure_count": critical,
        "null_clean_structure_probability": float(p.primary_specificity_null),
        "anticipated_clean_structure_probability": float(
            p.anticipated_clean_B_structure_probability
        ),
        "alpha_one_sided": float(p.alpha_one_sided),
        "actual_test_size": float(
            binom.sf(critical - 1, n, float(p.primary_specificity_null))
        ),
        "anticipated_exact_power": float(
            binom.sf(
                critical - 1, n, float(p.anticipated_clean_B_structure_probability)
            )
        ),
        "power_is_conditional_on_assumed_true_specificity_not_guaranteed": True,
    }


def _validate_design(cfg, source):
    # Freeze the measurement pipeline rather than changing it after observing
    # the failed O1M screen. Seed namespaces and calibration sizes alone change.
    for key in (
        "env.network",
        "env.eeg",
        "env.online",
        "analysis.states",
        "analysis.spatial_measurement",
        "analysis.multitaper",
        "analysis.iaf",
        "analysis.target_fs_hz",
        "analysis.low_hz",
        "analysis.high_hz",
        "analysis.eeg_array",
        "analysis.inhibition_scale",
        "analysis.reference",
        "analysis.criteria",
        "analysis.observation_noise",
    ):
        actual, expected = OmegaConf.select(cfg, key), OmegaConf.select(
            source["config"], key
        )

        def plain(value):
            return (
                OmegaConf.to_container(value, resolve=True)
                if OmegaConf.is_config(value)
                else value
            )

        if plain(actual) != plain(expected):
            raise ValueError(f"Frozen source setting changed: {key}")
    if (
        float(cfg.env.simulation.obs_win_len) != 1000
        or str(cfg.analysis.simulator) != "online"
    ):
        raise ValueError("Use the persistent online simulator with 1000-ms windows.")
    p = cfg.analysis.rhythm_screen
    if str(p.score_name) != SCORE or not (
        p.strict_greater_than_cutoff
        and p.retain_frozen_alpha_excess_threshold
        and p.preserve_population_B_target
    ):
        raise ValueError(
            "Use only the fixed spectral-evidence score; preserve the B target and alpha screen."
        )
    if str(p.quantile_rule) != "ceil_n_plus_one_times_one_minus_alpha":
        raise ValueError("The grouped rank-based null calibration is frozen.")
    if [
        float(p.null_cluster_false_positive_target),
        float(p.primary_specificity_null),
        float(p.anticipated_clean_B_structure_probability),
        float(p.alpha_one_sided),
    ] != [0.05, 0.8, 0.95, 0.05]:
        raise ValueError("The null calibration and inference design are frozen.")
    if not cfg.analysis.smoke_test:
        if cfg.analysis.timeline != source["config"].analysis.timeline:
            raise ValueError(
                "Full timing must remain 1/30/9/2 seconds, all zero field."
            )
        if (
            int(cfg.analysis.measurement_design.calibration_structures),
            int(cfg.analysis.measurement_design.evaluation_structures),
            int(cfg.analysis.measurement_design.noise_repeats),
        ) != (19, 30, 3):
            raise ValueError(
                "The full design is 19 calibration and 30 evaluation structures, with three noise views."
            )
        if _power_design(cfg)["anticipated_exact_power"] < float(p.target_power):
            raise ValueError(
                "The prespecified exact-binomial power requirement failed."
            )
    specs = spatial._trajectory_specs(cfg)
    namespaces = [
        {int(row[key]) for row in specs}
        for key in (
            "structure_seed",
            "history_seed",
            "phase_seed",
            "trial_seed",
            "future_drive_seed",
        )
    ]
    namespaces.append(
        {
            seed
            for i in range(len(specs))
            for repeat in range(int(cfg.analysis.measurement_design.noise_repeats))
            for sensor in range(3)
            for seed in spatial._noise_seeds(cfg, i, repeat, sensor)
        }
    )
    if (
        any(a & b for a, b in itertools.combinations(namespaces, 2))
        or set.union(*namespaces) & source["seed_union"]
    ):
        raise ValueError("Seed namespaces overlap each other or H5-O1M.")
    if max(namespaces[0]) * 10000 + 255 >= np.iinfo(np.uint32).max:
        raise ValueError("Structure seed exceeds the simulator's uint32 namespace.")
    return specs


def _add_evidence(table, cfg, root):
    """Recover the frozen evidence from saved prestimulation EEG only.

    Keeping this separate leaves the H5-O1M runner unchanged. Both estimators
    share the same carrier evidence; ideal EEG is an attribution view only.
    """
    records = []
    for (trajectory, angle, repeat, view), _ in table.groupby(
        ["trajectory_id", "true_orientation_deg", "noise_repeat", "signal_view"]
    ):
        path = root / "processed_EEG" / f"{trajectory}_o{angle:02.0f}_n{repeat}.npz"
        with np.load(path) as data:
            signal = data[
                "baseline_observed_v" if view == "observed" else "baseline_neural_v"
            ][0]
            rows, _, _ = spatial._estimate_multitaper_methods(
                signal,
                fs_hz=float(data["fs_hz"]),
                hidden_frequency_hz=float("nan"),
                input_signal=spatial.OBSERVED,
                cfg=cfg,
            )
        result = next(row for row in rows if row["estimator"] == spatial.MT_POOLED)
        records.append(
            {
                "trajectory_id": trajectory,
                "true_orientation_deg": angle,
                "noise_repeat": repeat,
                "signal_view": view,
                SCORE: float(result[SCORE]),
                "evidence_margin_db": float(result["evidence_margin_db"]),
                "soft_support_fraction": float(result["soft_support_fraction"]),
            }
        )
    return table.merge(
        pd.DataFrame(records),
        on=["trajectory_id", "true_orientation_deg", "noise_repeat", "signal_view"],
        validate="many_to_one",
    )


def _calibrate_null(calibration, cfg):
    if (
        not calibration.stage.eq("calibration").all()
        or not calibration.condition.eq("B").all()
    ):
        raise ValueError(
            "Calibration must receive B-only calibration rows, never A or validation observations."
        )
    selected = calibration[
        calibration.signal_view.eq("observed")
        & calibration.estimator.eq(spatial.PRIMARY)
    ]
    if selected.empty or not np.isfinite(selected[SCORE]).all():
        raise ValueError("Missing or nonfinite B null scores.")
    scores = selected.groupby("structure_seed")[SCORE].max().sort_index()
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Missing or nonfinite B null scores.")
    alpha = float(cfg.analysis.rhythm_screen.null_cluster_false_positive_target)
    rank = math.ceil((len(scores) + 1) * (1 - alpha) - 1.0e-12)
    finite = rank <= len(scores)
    # With insufficient calibration data (e.g. smoke), +infinity is the honest
    # conservative cutoff. Persist it as null plus an explicit abstention flag.
    threshold = float(np.sort(scores)[rank - 1]) if finite else None
    return {
        "score_name": SCORE,
        "cutoff_db": threshold,
        "abstain_all_for_insufficient_calibration": not finite,
        "comparison": "strictly greater than cutoff",
        "calibration_structures": len(scores),
        "calibration_structure_seeds": [int(x) for x in scores.index],
        "calibration_structure_maximum_scores_db": {
            str(k): float(v) for k, v in scores.items()
        },
        "order_statistic_rank": rank,
        "target_cluster_false_positive_probability": alpha,
        "marginal_rank_bound": (
            float((len(scores) + 1 - rank) / (len(scores) + 1)) if finite else 0.0
        ),
        "uses_no_A_rows_no_treatment_outcomes_no_ideal_EEG": True,
        "scope": "Exchangeable B structures under the frozen finite orientation/noise grid; marginal over calibration and a new structure, not guaranteed conditional specificity of the realized cutoff.",
    }


def _screen(measurement, target, rule, cfg):
    legacy = spatial._screen_measurement(
        {key: measurement[key] for key in spatial.SCREEN_INPUT_FIELDS}, target, cfg
    )
    score = float(measurement[SCORE])
    rhythm = bool(
        not rule["abstain_all_for_insufficient_calibration"]
        and np.isfinite(score)
        and score > rule["cutoff_db"]
    )
    phenotype = bool(legacy["phenotype_positive"] and rhythm)
    eligible = bool(legacy["treatment_eligible"] and rhythm)
    return {
        "alpha_excess_over_B_log10": legacy["alpha_excess_over_B_log10"],
        "legacy_alpha_positive": legacy["phenotype_positive"],
        "legacy_treatment_eligible": legacy["treatment_eligible"],
        "rhythm_present": rhythm,
        "phenotype_positive": phenotype,
        "treatment_eligible": eligible,
        "fallback_action": "not_applied_measurement_only" if eligible else "sham",
    }


def _apply_screen(table, target, rule, cfg):
    rows = [
        _screen(
            {key: row[key] for key in INPUTS},
            target[f"{row.signal_view}/{row.estimator}"],
            rule,
            cfg,
        )
        for _, row in table.iterrows()
    ]
    return pd.concat([table.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _binomial_inference(clean, cfg):
    clean = np.asarray(clean, bool)
    n = len(clean)
    k = int(clean.sum())
    if n == 0:
        raise ValueError("Specificity requires at least one held-out B structure.")
    alpha = float(cfg.analysis.rhythm_screen.alpha_one_sided)
    lower = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
    interval = [
        0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)),
    ]
    return {
        "independent_structure_count": n,
        "clean_B_structure_count": k,
        "structures_with_any_false_positive": n - k,
        "clean_structure_fraction": k / n,
        "one_sided_exact_binomial_p": float(
            binom.sf(
                k - 1, n, float(cfg.analysis.rhythm_screen.primary_specificity_null)
            )
        ),
        "one_sided_95_lower_bound": lower,
        "two_sided_95_Clopper_Pearson_interval": interval,
        "unit": "One Bernoulli outcome per B structure: all its orientation/noise views phenotype-negative",
        "confidence_rejection_cannot_hide_phenotype_false_positives": True,
    }


def _summarize(evaluation, calibration, trajectories, source, rule, cfg):
    rows = []
    for (view, method, seed), g in evaluation.groupby(
        ["signal_view", "estimator", "structure_seed"]
    ):
        a, b = g[g.condition.eq("A")], g[g.condition.eq("B")]
        accepted = a[a.carrier_identified]
        for mode, column in [
            ("legacy_alpha_only", "legacy_alpha_positive"),
            ("rhythm_and_alpha", "phenotype_positive"),
        ]:
            eligible = (
                "legacy_treatment_eligible"
                if mode == "legacy_alpha_only"
                else "treatment_eligible"
            )
            sensitivity = float(a[column].mean())
            specificity = float(1 - b[column].mean())
            row = {
                "signal_view": view,
                "estimator": method,
                "structure_seed": int(seed),
                "screen": mode,
                "A_sensitivity": sensitivity,
                "B_specificity": specificity,
                "balanced_accuracy": 0.5 * (sensitivity + specificity),
                "A_treatment_coverage": float(a[eligible].mean()),
                "B_treatment_false_positive_fraction": float(b[eligible].mean()),
                "B_structure_clean": bool(not b[column].any()),
                "spatial_balanced_accuracy": spatial._balanced_accuracy(a),
                "mean_angle_error_deg": float(a.absolute_angle_error_deg.mean()),
                "carrier_coverage": float(a.carrier_identified.mean()),
                "accepted_carrier_accuracy": (
                    float(accepted.carrier_correct.mean())
                    if len(accepted)
                    else float("nan")
                ),
                "recent_phase_actionable_fraction": float(
                    a.recent_phase_actionable.mean()
                ),
            }
            for f in (9, 11):
                part = a[np.isclose(a.hidden_frequency_hz, f)]
                row[f"A{f}_sensitivity"] = float(part[column].mean())
                row[f"A{f}_treatment_coverage"] = float(part[eligible].mean())
            rows.append(row)
    structures = pd.DataFrame(rows)
    numeric = [
        k
        for k in structures.select_dtypes(include=["number", "bool"]).columns
        if k != "structure_seed"
    ]
    summary = (
        structures.groupby(["signal_view", "estimator", "screen"])[numeric]
        .agg(lambda x: x.mean(skipna=False))
        .reset_index()
    )
    primary = structures[
        structures.signal_view.eq("observed")
        & structures.estimator.eq(spatial.PRIMARY)
        & structures.screen.eq("rhythm_and_alpha")
    ]
    inference = _binomial_inference(primary.B_structure_clean, cfg)
    inference["a_priori_power_design"] = _power_design(cfg)
    rng = np.random.default_rng(int(cfg.analysis.rhythm_screen.bootstrap_seed))
    uncertainty = {}
    for column in [
        "A_sensitivity",
        "B_specificity",
        "balanced_accuracy",
        "A_treatment_coverage",
        "spatial_balanced_accuracy",
    ]:
        values = primary[column].to_numpy(float)
        draws = rng.choice(
            values,
            (int(cfg.analysis.rhythm_screen.bootstrap_repetitions), len(values)),
            replace=True,
        ).mean(axis=1)
        uncertainty[column] = {
            "mean": float(values.mean()),
            "structure_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    inference["secondary_descriptive_uncertainty"] = uncertainty
    inference["multiplicity"] = (
        "One prespecified primary exact specificity test; other endpoints are mandatory descriptive guardrails, not substitute significant results."
    )
    # Undefined accuracy in a structure with no accepted carrier must fail,
    # not disappear when averaging the easier structures.
    mean = primary.select_dtypes(include=["number", "bool"]).agg(
        lambda values: values.mean(skipna=False)
    )
    criteria = cfg.analysis.screening_criteria
    eligible = evaluation[evaluation.estimator.eq(spatial.PRIMARY)]
    ideal_b = calibration[
        calibration.signal_view.eq("neural_audit")
        & calibration.estimator.eq(spatial.PRIMARY)
    ]
    spread = float(
        ideal_b.groupby("structure_seed")
        .outcome_geometry_normalized_log10_alpha.agg(lambda x: x.max() - x.min())
        .max()
    )
    rates = trajectories[trajectories.stage.eq("evaluation")]
    b_rates = rates[rates.condition.eq("B")].set_index("structure_seed")
    a_rates = rates[rates.condition.eq("A")]
    rate_diff = max(
        float(np.max(np.abs(a_rates[k] - a_rates.structure_seed.map(b_rates[k]))))
        for k in ("E_firing_rate_hz", "I_firing_rate_hz")
    )
    checks = {
        "negative_H5O1M_preserved_by_hash": True,
        "source_spatial_estimator_and_B_target_unchanged": True,
        "calibration_contains_only_new_B_structures": bool(
            calibration.condition.eq("B").all()
        ),
        "calibration_and_evaluation_structures_disjoint": bool(
            set(calibration.structure_seed).isdisjoint(evaluation.structure_seed)
        ),
        "minimum_null_calibration_structures": bool(
            rule["calibration_structures"] >= 19
        ),
        "finite_B_only_cutoff_frozen_before_evaluation": bool(
            not rule["abstain_all_for_insufficient_calibration"]
        ),
        "minimum_heldout_structures": bool(len(primary) >= 30),
        "a_priori_exact_binomial_design_powered": bool(
            _power_design(cfg)["anticipated_exact_power"]
            >= float(cfg.analysis.rhythm_screen.target_power)
        ),
        "complete_matched_A9_A11_B_grid": bool(
            len(evaluation)
            == len(primary)
            * 3
            * 4
            * int(cfg.analysis.measurement_design.noise_repeats)
            * 2
            * 2
        ),
        "all_simulations_stimulation_free": bool(
            trajectories.applied_amplitude_v_per_m.eq(0).all()
        ),
        "screen_inputs_exclude_hidden_labels_and_postdecision_data": True,
        "primary_structure_specificity_test_passes": bool(
            inference["one_sided_exact_binomial_p"]
            <= float(cfg.analysis.rhythm_screen.alpha_one_sided)
        ),
        "B_view_specificity": bool(
            mean.B_specificity >= criteria.minimum_B_view_specificity
        ),
        "A_sensitivity": bool(
            mean.A_sensitivity >= criteria.minimum_A_phenotype_sensitivity
        ),
        "A_sensitivity_in_both_carriers": bool(
            min(mean.A9_sensitivity, mean.A11_sensitivity)
            >= criteria.minimum_A_phenotype_sensitivity
        ),
        "A_treatment_coverage_in_both_carriers": bool(
            min(mean.A9_treatment_coverage, mean.A11_treatment_coverage)
            >= criteria.minimum_A_treatment_coverage
        ),
        "screen_balanced_accuracy": bool(
            mean.balanced_accuracy >= criteria.minimum_screen_balanced_accuracy
        ),
        "spatial_decoding_retained": bool(
            mean.spatial_balanced_accuracy >= criteria.minimum_spatial_balanced_accuracy
        ),
        "spatial_angle_accuracy_retained": bool(
            mean.mean_angle_error_deg <= criteria.maximum_mean_angle_error_deg
        ),
        "carrier_coverage": bool(
            mean.carrier_coverage >= criteria.minimum_carrier_coverage
        ),
        "accepted_carrier_accuracy": bool(
            mean.accepted_carrier_accuracy >= criteria.minimum_accepted_carrier_accuracy
        ),
        "recent_phase_actionable": bool(
            mean.recent_phase_actionable_fraction
            >= criteria.minimum_recent_phase_actionable_fraction
        ),
        "noise_scale_frozen_from_predecision_EEG": bool(
            np.allclose(evaluation.achieved_vertex_noise_RMS_fraction, 0.25, atol=0.005)
        ),
        "paired_B_geometry_invariance": bool(
            spread
            <= float(
                cfg.analysis.measurement_criteria.maximum_normalized_reference_spread_log10
            )
        ),
        "all_measurements_finite": bool(
            np.isfinite(
                eligible[
                    [
                        SCORE,
                        "geometry_normalized_log10_alpha",
                        "outcome_geometry_normalized_log10_alpha",
                    ]
                ]
            )
            .all()
            .all()
        ),
        "rates_within_guardrails": bool(
            trajectories.E_firing_rate_hz.between(
                0.0,
                float(cfg.analysis.measurement_criteria.maximum_rate_E_hz),
                inclusive="right",
            ).all()
            and trajectories.I_firing_rate_hz.between(
                0.0,
                float(cfg.analysis.measurement_criteria.maximum_rate_I_hz),
                inclusive="right",
            ).all()
        ),
        "A_B_rates_matched": bool(
            rate_diff
            <= float(
                cfg.analysis.measurement_criteria.maximum_paired_rate_difference_hz
            )
        ),
        "exact_zero_field_removal": bool(
            trajectories.final_extracellular_residual_mV.eq(0).all()
        ),
    }
    inference.update(
        maximum_paired_rate_difference_hz=rate_diff,
        maximum_B_rotation_spread_log10=spread,
    )
    return structures, summary, inference, checks


def _legacy_plot_metrics(structures):
    """Adapt descriptive columns without running O1M's 2**n sign-flip test."""
    return (
        structures[structures.screen.eq("legacy_alpha_only")]
        .drop(columns="balanced_accuracy")
        .rename(
            columns={
                "spatial_balanced_accuracy": "balanced_accuracy",
                "A_sensitivity": "A_screen_sensitivity",
                "B_specificity": "B_screen_specificity",
            }
        )
    )


def _plots(root, evaluation, calibration, structures, rule, cfg):
    primary = evaluation[
        evaluation.signal_view.eq("observed") & evaluation.estimator.eq(spatial.PRIMARY)
    ]
    # Individual views are displayed, but are not independent replicates.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, condition in zip(axes, ("B", "A")):
        g = primary[primary.condition.eq(condition)]
        for _, part in g.groupby("structure_seed"):
            ax.scatter(part.alpha_excess_over_B_log10, part[SCORE], s=9, alpha=0.45)
        ax.axvline(
            float(cfg.analysis.spatial_measurement.phenotype_minimum_excess_log10),
            color="k",
            ls="--",
        )
        if rule["cutoff_db"] is not None:
            ax.axhline(rule["cutoff_db"], color="k", ls=":")
        ax.set(
            xlabel="Alpha excess over frozen B (log10)",
            ylabel="Rhythmic evidence (dB)",
            title=condition,
        )
    spatial._save_figure(fig, root, "figure_08_rhythm_and_power_screen")
    fig, ax = plt.subplots(figsize=(8, 4))
    g = structures[
        structures.signal_view.eq("observed") & structures.estimator.eq(spatial.PRIMARY)
    ]
    for label, group in g.groupby("screen"):
        group = group.sort_values("structure_seed")
        ax.plot(np.arange(len(group)), group.B_specificity, "o-", label=label)
    ax.axhline(0.8, color="k", ls="--")
    ax.set(
        xlabel="Independent structure index",
        ylabel="B view specificity",
        ylim=(-0.03, 1.03),
    )
    ax.legend()
    spatial._save_figure(fig, root, "figure_09_specificity_by_structure")
    fig, ax = plt.subplots(figsize=(8, 4))
    cal = np.asarray(list(rule["calibration_structure_maximum_scores_db"].values()))
    ax.plot(
        np.sort(cal),
        np.arange(1, len(cal) + 1) / len(cal),
        "o-",
        label="Calibration B structure maxima",
    )
    for condition in ("B", "A"):
        values = (
            primary[primary.condition.eq(condition)]
            .groupby("structure_seed")[SCORE]
            .max()
            .sort_values()
        )
        ax.plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            "o-",
            label=f"Held-out {condition} structure maxima",
        )
    if rule["cutoff_db"] is not None:
        ax.axvline(rule["cutoff_db"], color="k", ls="--")
    ax.set(
        xlabel="Maximum rhythmic evidence across views (dB)",
        ylabel="Empirical cumulative fraction",
    )
    ax.legend(fontsize=8)
    spatial._save_figure(fig, root, "figure_10_null_calibration")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    started = time.perf_counter()
    source = _load_source(cfg)
    specs = _validate_design(cfg, source)
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / str(
        cfg.analysis.output_root_name
    )
    occupied = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if comm.bcast(occupied, root=0):
        raise FileExistsError(f"Refusing to overwrite {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        for name in ("canonical_dipoles", "processed_EEG", "unit_noise"):
            (root / name).mkdir()
        OmegaConf.save(cfg, root / "resolved_config.yaml", resolve=True)
        spatial._write_json(root / "frozen_EEG_only_B_targets.json", source["target"])
        model = spatial._forward_model(cfg)
        spatial._write_json(
            root / "prespecified_protocol.json",
            {
                "source_hashes": source["hashes"],
                "design": specs,
                "null_calibration": OmegaConf.to_container(
                    cfg.analysis.rhythm_screen, resolve=True
                ),
                "power": _power_design(cfg),
                "threshold_is_fitted_only_on_B_calibration": True,
                "all_actions": "sham",
                "one_afferent_history_per_structure": True,
            },
        )
        print(f"H5-O1S: {len(specs)} zero-field episodes; {size} MPI ranks", flush=True)
    else:
        model = None
    tables = []
    spectra = []
    trajectory_rows = []
    rule = None
    rule_hash = None
    for i, context in enumerate(specs):
        if context["stage"] == "evaluation" and rule is None:
            if rank == 0:
                rule = _calibrate_null(pd.concat(tables, ignore_index=True), cfg)
                spatial._write_json(root / "frozen_rhythm_presence_rule.json", rule)
                rule_hash = hashlib.sha256(
                    (root / "frozen_rhythm_presence_rule.json").read_bytes()
                ).hexdigest()
                print(
                    f"Frozen B-only rhythm cutoff: {rule['cutoff_db']} dB", flush=True
                )
            rule = comm.bcast(rule, root=0)
        if rank == 0:
            print(
                f"[{i+1}/{len(specs)}] {context['trajectory_id']} structure={context['structure_seed']}",
                flush=True,
            )
        run_cfg = spatial._with_orientation_state(
            cfg, context, homogeneous_B=context["condition"] == "B"
        )
        with open_dict(run_cfg):
            run_cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg = 0.0
            run_cfg.env.online.record_representative_state = False
        episode = spatial._run_profile(
            condition_cfg=run_cfg,
            context=context,
            future_seed=context["future_drive_seed"],
            future_index=0,
            profile=spatial.SHAM,
            phase_sensor_index=0,
            root=root / "zero_field_episodes",
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            table, psd = spatial._analyze_trajectory(episode, context, cfg, model, root)
            table = _add_evidence(table, cfg, root)
            if rule is not None:
                table = _apply_screen(table, source["target"], rule, cfg)
            tables.append(table)
            spectra.append(psd)
            rate = spatial._epoch_row(episode, "baseline")
            trajectory_rows.append(
                {
                    **context,
                    "actual_modulation_depth": (
                        0.0 if context["condition"] == "B" else 0.04
                    ),
                    "actual_phase_diffusion_rad2_per_s": (
                        0.0 if context["condition"] == "B" else 0.5
                    ),
                    "E_firing_rate_hz": float(rate.E_firing_rate_hz),
                    "I_firing_rate_hz": float(rate.I_firing_rate_hz),
                    "baseline_spike_sha256": spatial._spike_hash(episode, "baseline"),
                    "applied_amplitude_v_per_m": float(
                        episode["simulation"]["action"]["ac_amplitude_v_per_m"]
                    ),
                    "final_extracellular_residual_mV": float(
                        episode["simulation"]["final_residual_mV"]
                    ),
                }
            )
            pd.concat(tables, ignore_index=True).to_csv(
                root / "measurement_checkpoint.csv", index=False
            )
            pd.DataFrame(trajectory_rows).to_csv(
                root / "trajectory_audit.csv", index=False
            )
    comm.Barrier()
    if rank != 0:
        return
    complete = pd.concat(tables, ignore_index=True)
    psd = pd.concat(spectra, ignore_index=True)
    calibration = complete[complete.stage.eq("calibration")].copy()
    evaluation = complete[complete.stage.eq("evaluation")].copy()
    for key in (
        "legacy_alpha_positive",
        "legacy_treatment_eligible",
        "rhythm_present",
        "phenotype_positive",
        "treatment_eligible",
    ):
        evaluation[key] = evaluation[key].astype(bool)
    trajectories = pd.DataFrame(trajectory_rows)
    structures, summary, inference, checks = _summarize(
        evaluation, calibration, trajectories, source, rule, cfg
    )
    checks["cutoff_file_unchanged_since_before_evaluation"] = (
        hashlib.sha256(
            (root / "frozen_rhythm_presence_rule.json").read_bytes()
        ).hexdigest()
        == rule_hash
    )
    checks["population_B_target_identical_to_frozen_source"] = (
        json.loads((root / "frozen_EEG_only_B_targets.json").read_text())
        == source["target"]
    )
    for name, table in [
        ("calibration_measurements", calibration),
        ("evaluation_measurements", evaluation),
        ("structure_metrics", structures),
        ("screening_summary", summary),
        ("baseline_PSD", psd),
    ]:
        table.to_csv(root / f"{name}.csv", index=False)
    spatial._write_json(root / "primary_structure_inference.json", inference)
    if cfg.experiment.plot:
        # Reuse O1M's PSD/geometry diagnostics; new plots explicitly show rhythm
        # and the old versus revised screen. The shared panel cutoff remains
        # the unchanged alpha prerequisite, not the complete new decision.
        legacy_eval = evaluation.copy()
        legacy_eval["phenotype_positive"] = legacy_eval.legacy_alpha_positive
        legacy_eval["treatment_eligible"] = legacy_eval.legacy_treatment_eligible
        old_structures = _legacy_plot_metrics(structures)
        spatial._plots(root, legacy_eval, calibration, psd, old_structures, cfg)
        _plots(root, evaluation, calibration, structures, rule, cfg)
    smoke = bool(cfg.analysis.smoke_test)
    passed = all(checks.values()) and not smoke
    spatial._write_json(
        root / "protocol_and_provenance.json",
        {
            "source": {
                "root": source["root"],
                "hashes": source["hashes"],
                "upstream": source["provenance"],
            },
            "frozen_rule_sha256": rule_hash,
            "mpi_ranks": size,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "trajectory_count": len(specs),
            "smoke_test": smoke,
            "all_stimulation_zero": True,
            "claim_boundary": "Specificity of a toy EEG rhythm/alpha screen, not treatment response, learned control, clinical diagnosis, or H5.",
            "one_afferent_history_per_structure_no_longitudinal_generalization_claim": True,
            "figures_01_through_07_show_inherited_measurement_and_alpha_only_diagnostics": True,
        },
    )
    result = {
        "status": (
            "SMOKE COMPLETED" if smoke else ("PASSED" if passed else "NOT PASSED")
        ),
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "smoke_test": smoke,
        "completed_trajectories": len(specs),
        "runtime_seconds": time.perf_counter() - started,
        "ready_for_small_response_reassessment": passed,
        "ready_for_ML_or_H5_confirmation": False,
        "H5_status": "NOT ESTABLISHED",
    }
    spatial._write_json(root / "experiment_conclusion.json", result)
    spatial._write_json(
        root / "run_complete.json",
        {
            "completed": True,
            "runtime_seconds": time.perf_counter() - started,
            "files_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.iterdir())
                if p.is_file()
            },
        },
    )
    print("\nH5-O1S screening checks", flush=True)
    for k, v in checks.items():
        print(f"{k}: {'PASSED' if v else 'NOT PASSED'}")
    print(summary.to_string(index=False))
    print(json.dumps(inference, indent=2))
    print(
        f"\nRhythm screening validation: {result['status']}\nResults saved to: {root}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if MPI.COMM_WORLD.Get_size() > 1:
            import traceback

            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
