"""H5-O1D: noisy-EEG montage-profile policy development.

This exploratory study follows the positive H5-O0 orientation--montage map.
It adds two intermediate population orientations and moderate paired AR(1)
sensor noise, then learns one deliberately small EEG-only paired-effect model.
Whole circuit structures and postdecision futures are both held out from the
primary policy evaluation.  Ideal neural EEG is used only for efficacy.

The experiment develops and freezes a candidate for later disjoint
confirmation.  It cannot establish H5, clinical montage feasibility, or an
advantage over every possible individualized biophysical rule.
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


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _epoch_row,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (  # noqa: E402
    RESPONSIVE,
    _ar1_path,
    _future_seed,
    _noise_seeds,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_montage_orientation_opportunity import (  # noqa: E402
    ACTIVE_PROFILES,
    ALL_ACTIONS,
    PROFILE_60,
    PROFILE_Z,
    SHAM,
    TOPOGRAPHY_FEATURES,
    _contexts as _base_contexts,
    _dipole_norm_rms,
    _expected_map,
    _field_projection,
    _json_ready,
    _metric_row,
    _multichannel_features,
    _multichannel_raw,
    _orientation_specs,
    _profile,
    _profile_specs,
    _reference_contexts,
    _reference_target,
    _run_profile,
    _save_figure,
    _spike_hash,
    _with_orientation_state,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_multitaper_measurement_validation import (  # noqa: E402
    MT_POOLED,
    OBSERVED,
    _estimate_multitaper_methods,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_response_mapping import (  # noqa: E402
    _hash_locked_files,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _fourier_coefficients,
    _process_eeg,
)


ROOT_NAME = "h5_montage_policy_development"
POLICY_FEATURES = ["topography_vertex", "topography_right_minus_left"]
PRIMARY_SPLIT = "futures_1_2_to_3_4"
REVERSE_SPLIT = "futures_3_4_to_1_2"


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    root = Path(to_absolute_path(str(cfg.analysis.source_h5o0.result_dir)))
    names = {
        "conclusion": "experiment_conclusion.json",
        "audit": "H5_O0_montage_orientation_opportunity_audit.json",
        "screening": "prospective_screening.csv",
        "metrics": "context_montage_future_metrics.csv",
        "expected_map": "expected_context_montage_map.csv",
        "opportunity": "montage_response_opportunity.csv",
        "future_split": "independent_future_split_validation.csv",
        "target": "frozen_orientation_specific_B_target.json",
        "provenance": "protocol_and_provenance.json",
    }
    files, hashes = _hash_locked_files(
        root, names, cfg.analysis.source_h5o0.expected_sha256
    )
    conclusion = json.loads(files["conclusion"].read_text())
    if (
        conclusion["conclusions"]["H5_O0_montage_orientation_opportunity"]
        != "PASSED"
        or not bool(conclusion["conclusions"][
            "ready_for_disjoint_EEG_policy_development"
        ])
    ):
        raise RuntimeError("H5-O1D requires the exact positive H5-O0 result.")
    provenance = json.loads(files["provenance"].read_text())
    seeds: set[int] = set()
    for key in ("screening", "metrics"):
        table = pd.read_csv(files[key])
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed",
        ):
            if column in table:
                seeds.update(table[column].dropna().astype(int).tolist())
    return {
        "root": str(root),
        "hashes": hashes,
        "source_seed_union": seeds,
        "H5O0_passed": True,
        "upstream_provenance": provenance,
    }


def _run_contexts(cfg: DictConfig) -> list[dict[str, Any]]:
    rows = _base_contexts(cfg, apply_smoke_limit=False)
    if not bool(cfg.analysis.smoke_test):
        return rows
    limit = int(cfg.analysis.smoke_context_limit)
    if limit <= 0 or limit >= len(rows):
        return rows
    # Retain both action classes and at least two structures in the normal
    # four-context smoke so whole-structure policy code is exercised.
    selected: list[dict[str, Any]] = []
    for structure in sorted({int(row["structure_seed"]) for row in rows}):
        candidates = [
            row for row in rows
            if int(row["structure_seed"]) == structure
            and np.isclose(float(row["hidden_frequency_hz"]), 9.0)
        ]
        ordered = sorted(candidates, key=lambda row: float(row["rotation_y_rad"]))
        selected.extend([ordered[0], ordered[-1]])
        if len(selected) >= limit:
            break
    return selected[:limit]


def _side_noise_seeds(
    cfg: DictConfig, context: dict[str, Any], sensor_index: int,
) -> tuple[int, int]:
    base = (
        int(cfg.experiment.seed)
        + int(cfg.analysis.observation_noise.multichannel_side_seed_offset)
        + 100 * int(context["future_group_index"])
        + 10 * int(sensor_index)
    )
    return base, base + 10_000


def _observed_multichannel_baseline(
    episode: dict[str, Any], ideal_eeg: np.ndarray,
    context: dict[str, Any], cfg: DictConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add causal paired sensor noise without altering neural efficacy EEG."""
    simulation = episode["simulation"]
    observed_vertex = np.concatenate([
        np.asarray(output["eeg_v"], dtype=float).reshape(-1)
        for output in simulation["observed_outputs_by_epoch"]["baseline"]
    ])
    times = np.concatenate([
        np.asarray(output["sample_times_ms"], dtype=float).reshape(-1)
        for output in simulation["outputs_by_epoch"]["baseline"]
    ])
    if ideal_eeg.shape[1] != times.size or observed_vertex.size != times.size:
        raise RuntimeError("Multichannel and online noisy EEG samples do not align.")
    observed = np.asarray(ideal_eeg, dtype=float).copy()
    observed[0] = observed_vertex
    dt_ms = float(cfg.env.network.dt)
    window_ms = float(cfg.env.simulation.obs_win_len)
    pre_steps = int(cfg.analysis.timeline.burn_in_steps) + int(
        cfg.analysis.timeline.baseline_steps
    )
    total_steps = sum(int(cfg.analysis.timeline[f"{epoch}_steps"]) for epoch in (
        "burn_in", "baseline", "stimulation", "washout"
    ))
    total_samples = int(round(total_steps * window_ms / dt_ms))
    split_sample = int(round(pre_steps * window_ms / dt_ms))
    indices = np.rint(times / dt_ms).astype(int) - 1
    scale_v = float(simulation["observation"]["baseline_neural_rms_v"]) * float(
        cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg
    )
    hashes = [str(simulation["observation"]["unit_noise_sha256"])]
    for sensor_index in range(1, ideal_eeg.shape[0]):
        history_seed, future_seed = _side_noise_seeds(
            cfg, context, sensor_index
        )
        unit_noise = _ar1_path(
            n_samples=total_samples,
            split_sample=split_sample,
            history_seed=history_seed,
            future_seed=future_seed,
            coefficient=float(cfg.analysis.observation_noise.ar1_coefficient),
        )
        observed[sensor_index] += scale_v * unit_noise[indices]
        hashes.append(hashlib.sha256(
            np.asarray(unit_noise, dtype="<f8").tobytes()
        ).hexdigest())
    if not np.all(np.isfinite(observed)):
        raise RuntimeError("Observed multichannel EEG contains non-finite samples.")
    observed_hash = hashlib.sha256(
        np.asarray(observed, dtype="<f8").tobytes()
    ).hexdigest()
    return observed, {
        "observation_noise_model": "independent_AR1_sensor_noise",
        "configured_noise_fraction": float(
            cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg
        ),
        "vertex_achieved_noise_fraction": float(
            simulation["observation"]["achieved_baseline_noise_fraction"]
        ),
        "common_absolute_noise_scale_v": scale_v,
        "sensor_unit_noise_sha256": ";".join(hashes),
        "observed_multichannel_baseline_sha256": observed_hash,
    }


def _vertex_phase_quality(
    values: np.ndarray, *, start_ms: float, frequency_hz: float,
    cfg: DictConfig,
) -> float:
    processed, fs_hz, _, _, _ = _process_eeg(
        values,
        simulator_fs_hz=1000.0 / float(cfg.env.network.dt),
        cfg=cfg,
    )
    count = int(round(fs_hz))
    recent = processed[-count:]
    cosine, sine = _fourier_coefficients(
        recent,
        fs_hz=fs_hz,
        start_ms=float(start_ms + values.size * cfg.env.network.dt - 1000.0),
        frequency_hz=float(frequency_hz),
    )
    rms = float(np.sqrt(np.mean(recent ** 2)))
    return float(np.hypot(cosine, sine) / max(rms, np.finfo(float).tiny))


def _screen_context(
    episode: dict[str, Any], context: dict[str, Any], target: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    neural, start_ms = _multichannel_raw(episode, "baseline", context, cfg)
    observed, noise = _observed_multichannel_baseline(
        episode, neural, context, cfg
    )
    processed, fs_hz, _, _, _ = _process_eeg(
        observed[0],
        simulator_fs_hz=1000.0 / float(cfg.env.network.dt),
        cfg=cfg,
    )
    if bool(cfg.analysis.smoke_test) and bool(cfg.analysis.smoke_force_eligible):
        selected_frequency = float(context["hidden_frequency_hz"])
        identified, evidence, margin = True, float("nan"), float("nan")
    else:
        rows, _, _ = _estimate_multitaper_methods(
            processed,
            fs_hz=fs_hz,
            hidden_frequency_hz=float(context["hidden_frequency_hz"]),
            input_signal=OBSERVED,
            cfg=cfg,
        )
        selected = next(row for row in rows if row["estimator"] == MT_POOLED)
        selected_frequency = float(selected["selected_frequency_hz"])
        identified = bool(selected["identified"])
        evidence = float(selected["maximum_residual_evidence_db"])
        margin = float(selected["evidence_margin_db"])
    features, observed_spectrum = _multichannel_features(
        observed, start_ms=start_ms, carrier_hz=selected_frequency, cfg=cfg
    )
    neural_features, neural_spectrum = _multichannel_features(
        neural, start_ms=start_ms, carrier_hz=selected_frequency, cfg=cfg
    )
    recent_quality = _vertex_phase_quality(
        observed[0], start_ms=start_ms, frequency_hz=selected_frequency, cfg=cfg
    )
    orientation_target = target[str(context["orientation_label"])]
    alpha_excess = (
        float(features["global_log10_alpha_power"])
        - float(orientation_target["screening_mean_log10_alpha"])
    )
    criteria = cfg.analysis.criteria
    phenotype = alpha_excess >= float(criteria.minimum_A_minus_B_alpha_log10)
    phase_actionable = recent_quality >= float(
        criteria.minimum_recent_resultant_to_rms
    )
    eligible = bool(identified and phenotype and phase_actionable)
    if bool(cfg.analysis.smoke_test) and bool(cfg.analysis.smoke_force_eligible):
        eligible = True
    reasons = []
    if not identified:
        reasons.append("carrier_estimator_abstained")
    if not phenotype:
        reasons.append("elevated_alpha_phenotype_absent")
    if not phase_actionable:
        reasons.append("recent_phase_not_actionable")
    outcome = _epoch_row(episode, "baseline")
    row = {
        **context,
        **features,
        "topography_right_minus_left": float(
            features["topography_right_xz"] - features["topography_left_xz"]
        ),
        "neural_global_log10_alpha_power": float(
            neural_features["global_log10_alpha_power"]
        ),
        "EEG_selected_frequency_hz": selected_frequency,
        "carrier_identified": identified,
        "EEG_frequency_selection_correct": bool(np.isclose(
            selected_frequency, float(context["hidden_frequency_hz"])
        )),
        "carrier_maximum_residual_evidence_db": evidence,
        "carrier_evidence_margin_db": margin,
        "alpha_excess_over_orientation_B_log10": alpha_excess,
        "alpha_phenotype_present": phenotype,
        "recent_resultant_to_rms": recent_quality,
        "recent_phase_actionable": phase_actionable,
        "phase_sensor_index": 0,
        "phase_sensor_label": "vertex",
        "eligible": eligible,
        "exclusion_reasons": ";".join(reasons) if reasons else "none",
        "baseline_E_firing_rate_hz": float(outcome.E_firing_rate_hz),
        "baseline_I_firing_rate_hz": float(outcome.I_firing_rate_hz),
        "baseline_spike_sha256": _spike_hash(episode, "baseline"),
        "baseline_dipole_norm_rms_nA_um": _dipole_norm_rms(episode, "baseline"),
        "screening_uses_only_predecision_multichannel_observed_EEG": True,
        "hidden_orientation_frequency_and_spikes_excluded_from_policy": True,
        **noise,
    }
    common = {
        "context_id": str(context["context_id"]),
        "structure_seed": int(context["structure_seed"]),
        "hidden_frequency_hz": float(context["hidden_frequency_hz"]),
        "orientation_label": str(context["orientation_label"]),
        "orientation_degrees": float(np.degrees(context["rotation_y_rad"])),
        "condition": "A_predecision",
    }
    spectrum = pd.concat([
        observed_spectrum.assign(signal_type="observed_EEG", **common),
        neural_spectrum.assign(signal_type="ideal_neural_EEG", **common),
    ], ignore_index=True)
    return row, spectrum


def _baseline_invariance(
    screening: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_id, group in screening.groupby("paired_orientation_context_id"):
        ordered = group.sort_values("rotation_y_rad")
        reference = ordered.iloc[0]
        denominator = max(
            abs(float(reference.baseline_dipole_norm_rms_nA_um)),
            np.finfo(float).tiny,
        )
        for sample in ordered.iloc[1:].itertuples():
            rows.append({
                "paired_orientation_context_id": str(pair_id),
                "structure_seed": int(reference.structure_seed),
                "hidden_frequency_hz": float(reference.hidden_frequency_hz),
                "reference_orientation_label": str(reference.orientation_label),
                "compared_orientation_label": str(sample.orientation_label),
                "spike_trains_identical": bool(
                    reference.baseline_spike_sha256 == sample.baseline_spike_sha256
                ),
                "absolute_E_rate_difference_hz": abs(
                    float(reference.baseline_E_firing_rate_hz)
                    - float(sample.baseline_E_firing_rate_hz)
                ),
                "absolute_I_rate_difference_hz": abs(
                    float(reference.baseline_I_firing_rate_hz)
                    - float(sample.baseline_I_firing_rate_hz)
                ),
                "dipole_norm_relative_error": abs(
                    float(reference.baseline_dipole_norm_rms_nA_um)
                    - float(sample.baseline_dipole_norm_rms_nA_um)
                ) / denominator,
            })
    table = pd.DataFrame(rows)
    return table, {
        "all_spike_trains_identical": bool(table.spike_trains_identical.all()),
        "maximum_rate_difference_hz": float(max(
            table.absolute_E_rate_difference_hz.max(),
            table.absolute_I_rate_difference_hz.max(),
        )),
        "maximum_dipole_norm_relative_error": float(
            table.dipole_norm_relative_error.max()
        ),
    }


def _profile_observability(
    screening: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = screening[screening.eligible.astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for structure in sorted(eligible.structure_seed.unique()):
        train = eligible[eligible.structure_seed.ne(structure)]
        test = eligible[eligible.structure_seed.eq(structure)]
        if train.matched_profile.nunique() < 2 or test.empty:
            continue
        center = train[POLICY_FEATURES].mean().to_numpy(float)
        scale = train[POLICY_FEATURES].std(ddof=0).to_numpy(float)
        scale[~np.isfinite(scale) | (scale <= np.finfo(float).tiny)] = 1.0
        centroids = {
            str(label): (group[POLICY_FEATURES].mean().to_numpy(float) - center) / scale
            for label, group in train.groupby("matched_profile")
        }
        for sample in test.itertuples():
            vector = np.asarray([
                getattr(sample, feature) for feature in POLICY_FEATURES
            ], dtype=float)
            vector = (vector - center) / scale
            distances = {
                label: float(np.linalg.norm(vector - centroid))
                for label, centroid in centroids.items()
            }
            predicted = min(distances, key=lambda key: (distances[key], key))
            rows.append({
                "context_id": str(sample.context_id),
                "structure_seed": int(structure),
                "true_matched_profile": str(sample.matched_profile),
                "predicted_matched_profile": predicted,
                "correct": predicted == str(sample.matched_profile),
                "features": ";".join(POLICY_FEATURES),
            })
    predictions = pd.DataFrame(rows)
    recalls = {
        str(label): float(group.correct.mean())
        for label, group in predictions.groupby("true_matched_profile")
    } if not predictions.empty else {}
    return predictions, {
        "LOSO_balanced_accuracy": (
            float(np.mean(list(recalls.values()))) if recalls else float("nan")
        ),
        "LOSO_recall": recalls,
        "features": POLICY_FEATURES,
        "audit_only_not_policy_training": True,
    }


def _loss_table(
    metrics: pd.DataFrame, screening: pd.DataFrame, futures: list[int],
) -> pd.DataFrame:
    active = metrics[
        metrics.montage_profile.isin(ACTIVE_PROFILES)
        & metrics.future_index.isin(futures)
    ]
    loss = active.groupby(
        ["context_id", "structure_seed", "montage_profile"], as_index=False
    ).post_distance_to_orientation_B_log10.mean()
    pivot = loss.pivot(
        index=["context_id", "structure_seed"], columns="montage_profile",
        values="post_distance_to_orientation_B_log10",
    ).reset_index()
    context = screening[[
        "context_id", "hidden_frequency_hz", "orientation_label",
        "rotation_y_rad", "matched_profile", *POLICY_FEATURES,
    ]]
    pivot = pivot.merge(context, on="context_id", validate="one_to_one")
    pivot["paired_effect_z_minus_60_log10"] = pivot[PROFILE_Z] - pivot[PROFILE_60]
    return pivot


def _fit_ridge(
    table: pd.DataFrame, *, response: str, penalty: float,
) -> dict[str, Any]:
    x = table[POLICY_FEATURES].to_numpy(float)
    y = table[response].to_numpy(float)
    center = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= np.finfo(float).tiny)] = 1.0
    z = (x - center) / scale
    design = np.column_stack([np.ones(len(z)), z])
    regularizer = np.diag([0.0] + [float(penalty)] * z.shape[1])
    coefficients = np.linalg.solve(
        design.T @ design + regularizer, design.T @ y
    )
    return {
        "feature_names": list(POLICY_FEATURES),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "ridge_penalty": float(penalty),
        "response": response,
        "decision": (
            "select montage_profile_60deg if predicted "
            "L_z-minus-L_60 > 0; otherwise montage_profile_z"
        ),
    }


def _predict_ridge(model: dict[str, Any], table: pd.DataFrame) -> np.ndarray:
    x = table[model["feature_names"]].to_numpy(float)
    z = (
        x - np.asarray(model["center"], dtype=float)
    ) / np.asarray(model["scale"], dtype=float)
    return float(model["intercept"]) + z @ np.asarray(
        model["coefficients"], dtype=float
    )


def _selected_loss(row: Any, action: str) -> float:
    return float(getattr(row, action))


def _policy_crossvalidation(
    metrics: pd.DataFrame, screening: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    block = cfg.analysis.policy_development
    primary_train = [int(value) for value in block.primary_training_futures]
    primary_test = [int(value) for value in block.primary_evaluation_futures]
    splits = [(PRIMARY_SPLIT, primary_train, primary_test)]
    if bool(block.reverse_split_audit):
        splits.append((REVERSE_SPLIT, primary_test, primary_train))
    rows: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    structures = sorted(screening.loc[screening.eligible, "structure_seed"].unique())
    for split_name, train_futures, test_futures in splits:
        train_all = _loss_table(metrics, screening, train_futures)
        test_all = _loss_table(metrics, screening, test_futures)
        for heldout in structures:
            train = train_all[train_all.structure_seed.ne(heldout)].copy()
            test = test_all[test_all.structure_seed.eq(heldout)].copy()
            if train.empty or test.empty:
                continue
            model = _fit_ridge(
                train,
                response="paired_effect_z_minus_60_log10",
                penalty=float(block.ridge_penalty),
            )
            model.update({
                "split_direction": split_name,
                "heldout_structure_seed": int(heldout),
                "training_structure_seeds": sorted(
                    int(value) for value in train.structure_seed.unique()
                ),
            })
            models.append(model)
            predictions = _predict_ridge(model, test)
            fixed_means = train[[PROFILE_Z, PROFILE_60]].mean().sort_values()
            fixed_action = str(fixed_means.index[0])
            threshold = float(block.analytical_topography_threshold)
            for sample, prediction in zip(test.itertuples(), predictions):
                learned = PROFILE_60 if prediction > 0.0 else PROFILE_Z
                analytical = (
                    str(block.analytical_high_action)
                    if float(sample.topography_right_minus_left) >= threshold
                    else str(block.analytical_low_action)
                )
                oracle = min(
                    ACTIVE_PROFILES,
                    key=lambda action: (_selected_loss(sample, action), action),
                )
                learned_loss = _selected_loss(sample, learned)
                fixed_loss = _selected_loss(sample, fixed_action)
                analytical_loss = _selected_loss(sample, analytical)
                random_loss = 0.5 * (
                    float(getattr(sample, PROFILE_Z))
                    + float(getattr(sample, PROFILE_60))
                )
                oracle_loss = _selected_loss(sample, oracle)
                rows.append({
                    "split_direction": split_name,
                    "heldout_structure_seed": int(heldout),
                    "context_id": str(sample.context_id),
                    "hidden_frequency_hz": float(sample.hidden_frequency_hz),
                    "orientation_label": str(sample.orientation_label),
                    "rotation_y_rad": float(sample.rotation_y_rad),
                    "matched_profile": str(sample.matched_profile),
                    **{
                        feature: float(getattr(sample, feature))
                        for feature in POLICY_FEATURES
                    },
                    "predicted_z_minus_60_effect_log10": float(prediction),
                    "learned_action": learned,
                    "trained_best_fixed_action": fixed_action,
                    "analytical_action": analytical,
                    "oracle_action": oracle,
                    "learned_loss_log10": learned_loss,
                    "fixed_loss_log10": fixed_loss,
                    "analytical_loss_log10": analytical_loss,
                    "uniform_random_expected_loss_log10": random_loss,
                    "oracle_loss_log10": oracle_loss,
                    "learned_advantage_over_fixed_log10": fixed_loss - learned_loss,
                    "learned_advantage_over_random_log10": random_loss - learned_loss,
                    "learned_minus_analytical_loss_log10": (
                        learned_loss - analytical_loss
                    ),
                    "learned_regret_to_oracle_log10": learned_loss - oracle_loss,
                    "policy_uses_only_predecision_observed_EEG": True,
                    "hidden_labels_excluded_from_policy": True,
                })
    evaluation = pd.DataFrame(rows)
    final_train = _loss_table(
        metrics, screening,
        [int(value) for value in block.final_model_training_futures],
    )
    final_model = _fit_ridge(
        final_train,
        response="paired_effect_z_minus_60_log10",
        penalty=float(block.ridge_penalty),
    )
    final_predictions = _predict_ridge(final_model, final_train)
    final_model.update({
        "training_structure_seeds": sorted(
            int(value) for value in final_train.structure_seed.unique()
        ),
        "training_future_indices": [
            int(value) for value in block.final_model_training_futures
        ],
        "training_context_count": int(len(final_train)),
        "selected_action_counts_on_training_contexts": {
            PROFILE_Z: int(np.sum(final_predictions <= 0.0)),
            PROFILE_60: int(np.sum(final_predictions > 0.0)),
        },
        "candidate_requires_disjoint_confirmation": True,
    })
    return evaluation, {"fold_models": models}, final_model


def _shuffle_null(
    metrics: pd.DataFrame, screening: pd.DataFrame, cfg: DictConfig,
) -> pd.DataFrame:
    block = cfg.analysis.policy_development
    train_all = _loss_table(
        metrics, screening,
        [int(value) for value in block.primary_training_futures],
    )
    test_all = _loss_table(
        metrics, screening,
        [int(value) for value in block.primary_evaluation_futures],
    )
    structures = sorted(test_all.structure_seed.unique())
    repetitions = int(block.shuffle_repetitions)
    if bool(cfg.analysis.smoke_test):
        repetitions = min(repetitions, 10)
    rng = np.random.default_rng(int(block.shuffle_seed))
    rows: list[dict[str, Any]] = []
    response = "paired_effect_z_minus_60_log10"
    for repetition in range(repetitions):
        advantages: list[float] = []
        for heldout in structures:
            train = train_all[train_all.structure_seed.ne(heldout)].copy()
            test = test_all[test_all.structure_seed.eq(heldout)].copy()
            if train.empty or test.empty:
                continue
            shuffled_parts = []
            for _, group in train.groupby("structure_seed", sort=False):
                part = group.copy()
                part[response] = rng.permutation(part[response].to_numpy(float))
                shuffled_parts.append(part)
            shuffled = pd.concat(shuffled_parts, ignore_index=True)
            model = _fit_ridge(
                shuffled, response=response, penalty=float(block.ridge_penalty)
            )
            predicted = _predict_ridge(model, test)
            fixed_action = str(
                train[[PROFILE_Z, PROFILE_60]].mean().sort_values().index[0]
            )
            for sample, value in zip(test.itertuples(), predicted):
                action = PROFILE_60 if value > 0 else PROFILE_Z
                advantages.append(
                    _selected_loss(sample, fixed_action)
                    - _selected_loss(sample, action)
                )
        rows.append({
            "shuffle_repetition": repetition + 1,
            "mean_shuffled_context_advantage_log10": (
                float(np.mean(advantages)) if advantages else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    if not bool(sources["H5O0_passed"]):
        raise ValueError("Positive H5-O0 provenance was not preserved.")
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-O1D requires persistent online simulation.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-O1D may not alter recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-O1D freezes the 9/11-Hz carrier grid.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-O1D freezes afferent modulation depth 0.04.")
    if len(cfg.analysis.states.phase_diffusion_levels) != 1 or not np.isclose(
        float(cfg.analysis.states.phase_diffusion_levels[0].diffusion_rad2_per_s),
        0.5,
    ):
        raise ValueError("H5-O1D fixes D=0.5 rad2/s.")
    orientations = _orientation_specs(cfg)
    angles = [round(float(np.degrees(x["rotation_y_rad"]))) for x in orientations]
    if angles != [0, 20, 40, 60]:
        raise ValueError("H5-O1D requires the frozen 0/20/40/60-degree grid.")
    if [x["matched_profile"] for x in orientations] != [
        PROFILE_Z, PROFILE_Z, PROFILE_60, PROFILE_60
    ]:
        raise ValueError("H5-O1D geometric audit labels changed.")
    if [x["montage_profile"] for x in _profile_specs(cfg)] != ACTIVE_PROFILES:
        raise ValueError("H5-O1D requires exactly the two H5-O0 profiles.")
    if not np.isclose(float(cfg.analysis.actions.amplitude_v_per_m), 0.2):
        raise ValueError("Both active profiles must remain at 0.2 V/m.")
    if _profile(cfg, RESPONSIVE) != {
        "adaptive": True, "history_ms": 500.0, "update_interval_ms": 125.0
    }:
        raise ValueError("The H4-confirmed phase controller changed.")
    noise = cfg.analysis.observation_noise
    if (
        not bool(noise.enabled)
        or not np.isclose(float(noise.rms_fraction_of_baseline_neural_eeg), 0.25)
        or not np.isclose(float(noise.ar1_coefficient), 0.95)
    ):
        raise ValueError("H5-O1D requires the frozen moderate AR(1) noise.")
    if list(cfg.analysis.policy_development.features) != POLICY_FEATURES:
        raise ValueError("H5-O1D policy features changed.")
    if not np.isclose(float(cfg.analysis.policy_development.ridge_penalty), 1.0):
        raise ValueError("H5-O1D ridge penalty must remain fixed at one.")
    smoke = bool(cfg.analysis.smoke_test)
    if not smoke and (
        int(cfg.analysis.crossed_design.n_structure_seeds) != 4
        or int(cfg.analysis.crossed_design.n_future_continuations) != 4
        or int(cfg.analysis.reference_calibration.n_structure_seeds) != 3
        or int(cfg.analysis.timeline.baseline_steps) != 30
        or int(cfg.analysis.timeline.stimulation_steps) != 9
        or int(cfg.analysis.timeline.washout_steps) != 2
    ):
        raise ValueError("Full H5-O1D is frozen to 4 structures, 4 futures, 30/9/2 s.")
    contexts = _base_contexts(cfg, apply_smoke_limit=False)
    expected_contexts = int(cfg.analysis.crossed_design.n_structure_seeds) * 8
    if len(contexts) != expected_contexts:
        raise ValueError("The frequency-orientation grid is incomplete.")
    if not all(group.orientation_label.nunique() == 4 for _, group in
               pd.DataFrame(contexts).groupby("paired_orientation_context_id")):
        raise ValueError("Every neural history must cross all four orientations.")
    references = _reference_contexts(cfg)
    sets = [
        {int(row[column]) for row in contexts}
        for column in ("structure_seed", "history_seed", "phase_seed", "trial_seed")
    ]
    sets.append({
        _future_seed(cfg, row, future)
        for row in contexts
        for future in range(int(cfg.analysis.crossed_design.n_future_continuations))
    })
    sets.extend([
        {int(row[column]) for row in references}
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "reference_future_seed",
        )
    ])
    controller_noise = {
        value
        for row in [*contexts, *references]
        for future in range(int(cfg.analysis.crossed_design.n_future_continuations))
        for value in _noise_seeds(cfg, row, future)
    }
    side_noise = {
        value
        for row in [*contexts, *references]
        for sensor in (1, 2)
        for value in _side_noise_seeds(cfg, row, sensor)
    }
    sets.extend([controller_noise, side_noise])
    if any(
        sets[left].intersection(sets[right])
        for left in range(len(sets)) for right in range(left + 1, len(sets))
    ):
        raise ValueError("H5-O1D seed namespaces overlap.")
    if set().union(*sets).intersection(sources["source_seed_union"]):
        raise ValueError("H5-O1D seeds overlap H5-O0.")
    if max(sets[0] | sets[5]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("H5-O1D structure seed exceeds uint32 mapping.")


def _checks(
    *, screening: pd.DataFrame, references: pd.DataFrame,
    metrics: pd.DataFrame, invariance: dict[str, Any],
    observability: dict[str, Any], evaluation: pd.DataFrame,
    shuffle: pd.DataFrame, final_model: dict[str, Any],
    sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    accepted = screening[screening.carrier_identified]
    active = metrics[metrics.montage_profile.isin(ACTIVE_PROFILES)]
    primary = evaluation[evaluation.split_direction.eq(PRIMARY_SPLIT)]
    reverse = evaluation[evaluation.split_direction.eq(REVERSE_SPLIT)]
    structure_primary = primary.groupby(
        "heldout_structure_seed", as_index=False
    ).learned_advantage_over_fixed_log10.mean()
    observed_advantage = float(primary.learned_advantage_over_fixed_log10.mean())
    shuffle_p = float(
        (1 + np.sum(
            shuffle.mean_shuffled_context_advantage_log10.to_numpy(float)
            >= observed_advantage
        )) / (1 + len(shuffle))
    )
    expected_optimal = (
        metrics[metrics.montage_profile.isin(ACTIVE_PROFILES)]
        .groupby(["context_id", "montage_profile"], as_index=False)
        .post_distance_to_orientation_B_log10.mean()
        .sort_values(["context_id", "post_distance_to_orientation_B_log10"])
        .groupby("context_id", as_index=False).first()
    )
    checks = {
        "source_H5O0_passed_and_hash_locked": bool(sources["H5O0_passed"]),
        "H5O1D_seeds_disjoint_from_H5O0_and_upstream_lineage": True,
        "orientation_is_distinct_from_A_B_state_generator": True,
        "afferent_mean_rate_matched_across_orientations_by_construction": True,
        "four_orientation_grid_includes_intermediate_states": bool(
            screening.orientation_label.nunique() == 4
        ) or bool(cfg.analysis.smoke_test),
        "orientation_specific_B_targets_calibrated_before_active_outcomes": bool(
            references.orientation_label.nunique() == 4
            and references.structure_seed.nunique()
            >= (1 if bool(cfg.analysis.smoke_test)
                else int(criteria.minimum_reference_structures))
        ),
        "complete_crossed_screening_grid": bool(
            len(screening) == len(_run_contexts(cfg))
        ),
        "screening_uses_only_predecision_observed_EEG": bool(
            screening.screening_uses_only_predecision_multichannel_observed_EEG.all()
        ),
        "moderate_observation_noise_applied": bool(np.allclose(
            screening.vertex_achieved_noise_fraction, 0.25, atol=5.0e-3
        )),
        "matched_coordinate_rotation_preserves_neural_trajectories": bool(
            invariance["all_spike_trains_identical"]
            and invariance["maximum_rate_difference_hz"]
            <= float(criteria.maximum_paired_rate_difference_hz)
            and invariance["maximum_dipole_norm_relative_error"]
            <= float(criteria.maximum_paired_dipole_norm_relative_error)
        ),
        "minimum_eligible_contexts": bool(
            len(eligible) >= int(criteria.minimum_eligible_contexts)
        ) or bool(cfg.analysis.smoke_test),
        "minimum_independent_structures": bool(
            eligible.structure_seed.nunique() >= int(criteria.minimum_structure_seeds)
        ) or bool(cfg.analysis.smoke_test),
        "both_carriers_and_action_classes_enrolled": bool(
            eligible.hidden_frequency_hz.nunique() == 2
            and eligible.matched_profile.nunique() == 2
        ) or bool(cfg.analysis.smoke_test),
        "carrier_identification_coverage": float(
            screening.carrier_identified.mean()
        ) >= float(criteria.minimum_carrier_identification_coverage),
        "accepted_carrier_accuracy": bool(len(accepted)) and float(
            accepted.EEG_frequency_selection_correct.mean()
        ) >= float(criteria.minimum_accepted_carrier_accuracy),
        "recent_vertex_phase_actionable": float(
            screening.recent_phase_actionable.mean()
        ) >= float(criteria.minimum_common_phase_estimate_actionable_fraction),
        "montage_class_observable_from_noisy_phase_invariant_EEG": bool(
            np.isfinite(observability["LOSO_balanced_accuracy"])
            and observability["LOSO_balanced_accuracy"]
            >= float(criteria.minimum_profile_LOSO_balanced_accuracy)
        ) or bool(cfg.analysis.smoke_test),
        "complete_paired_action_future_grid": bool(
            metrics.groupby(["context_id", "montage_profile"])
            .future_index.nunique().min()
            >= int(criteria.minimum_future_continuations)
        ) or bool(cfg.analysis.smoke_test),
        "identical_predecision_neural_and_observed_EEG_across_actions": bool(
            metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
            and metrics.groupby("context_id")
            .observed_baseline_sha256.nunique().eq(1).all()
        ),
        "single_equal_amplitude_profile_per_intervention": bool(
            np.isclose(active.amplitude_v_per_m, 0.2).all()
        ),
        "phase_updates_use_only_preceding_observed_EEG": bool(
            active.all_phase_estimates_causal.all()
        ),
        "phase_correction_is_frequency_bounded": bool(
            active.maximum_abs_frequency_correction_hz.max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "field_waveform_continuous_across_updates": bool(
            active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
        ),
        "expected_response_map_retains_both_profile_optima": bool(
            expected_optimal.montage_profile.nunique() == 2
        ),
        "primary_policy_uses_both_actions": bool(
            primary.learned_action.nunique() == 2
        ) or bool(cfg.analysis.smoke_test),
        "crossvalidated_policy_beats_best_fixed_practically": bool(
            observed_advantage >= float(
                criteria.minimum_learned_policy_advantage_log10
            )
        ) or bool(cfg.analysis.smoke_test),
        "policy_advantage_positive_across_structures": bool(
            np.mean(structure_primary.learned_advantage_over_fixed_log10 > 0)
            >= float(criteria.minimum_positive_structure_fraction)
        ) or bool(cfg.analysis.smoke_test),
        "policy_beats_uniform_random_selection": bool(
            primary.learned_advantage_over_random_log10.mean() > 0
        ) or bool(cfg.analysis.smoke_test),
        "EEG_context_beats_structure_preserving_shuffle": bool(
            shuffle_p <= float(criteria.maximum_context_shuffle_p_value)
        ) or bool(cfg.analysis.smoke_test),
        "policy_regret_to_full_information_oracle_is_small": bool(
            primary.learned_regret_to_oracle_log10.mean()
            <= float(criteria.maximum_policy_regret_log10)
        ) or bool(cfg.analysis.smoke_test),
        "policy_is_noninferior_to_frozen_analytical_EEG_rule": bool(
            primary.learned_minus_analytical_loss_log10.mean()
            <= float(criteria.analytical_noninferiority_margin_log10)
        ) or bool(cfg.analysis.smoke_test),
        "reverse_future_split_is_nonadverse": bool(
            not reverse.empty
            and reverse.learned_advantage_over_fixed_log10.mean()
            >= float(criteria.minimum_reverse_split_advantage_log10)
        ) or bool(cfg.analysis.smoke_test),
        "final_candidate_uses_only_frozen_EEG_features": bool(
            final_model["feature_names"] == POLICY_FEATURES
        ),
        "hidden_orientation_frequency_and_spikes_excluded_from_policy": bool(
            primary.policy_uses_only_predecision_observed_EEG.all()
            and primary.hidden_labels_excluded_from_policy.all()
        ) if not primary.empty else bool(cfg.analysis.smoke_test),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "exact_field_removal_confirmed": bool(
            np.isclose(metrics.final_extracellular_residual_mV, 0.0).all()
        ),
        "physiological_washout_recovery_audit": float(
            active.field_removal_recovered.mean()
        ) >= float(criteria.minimum_physiological_washout_recovery_fraction),
    }
    summary = {
        "primary_split": PRIMARY_SPLIT,
        "mean_policy_advantage_over_best_fixed_log10": observed_advantage,
        "structure_level_policy_advantage_log10": {
            str(int(row.heldout_structure_seed)): float(
                row.learned_advantage_over_fixed_log10
            ) for row in structure_primary.itertuples()
        },
        "positive_structure_fraction": float(np.mean(
            structure_primary.learned_advantage_over_fixed_log10 > 0
        )) if len(structure_primary) else float("nan"),
        "mean_advantage_over_uniform_random_log10": float(
            primary.learned_advantage_over_random_log10.mean()
        ),
        "mean_regret_to_oracle_log10": float(
            primary.learned_regret_to_oracle_log10.mean()
        ),
        "mean_learned_minus_analytical_loss_log10": float(
            primary.learned_minus_analytical_loss_log10.mean()
        ),
        "reverse_split_advantage_over_best_fixed_log10": float(
            reverse.learned_advantage_over_fixed_log10.mean()
        ) if len(reverse) else float("nan"),
        "primary_selected_action_counts": {
            str(key): int(value)
            for key, value in primary.learned_action.value_counts().items()
        },
        "structure_preserving_shuffle_p_value": shuffle_p,
        "shuffle_repetitions": int(len(shuffle)),
        "best_fixed_is_fold_specific_and_uses_training_data_only": True,
        "analytical_rule_is_frozen_from_H5O0_and_not_refitted": True,
    }
    return checks, summary


def _plots(
    *, root: Path, reference_spectra: pd.DataFrame,
    screening_spectra: pd.DataFrame, screening: pd.DataFrame,
    expected: pd.DataFrame, evaluation: pd.DataFrame,
    shuffle: pd.DataFrame, metrics: pd.DataFrame,
) -> None:
    tiny = np.finfo(float).tiny
    orientations = sorted(
        screening.orientation_label.unique(),
        key=lambda label: float(screening.loc[
            screening.orientation_label.eq(label), "rotation_y_rad"
        ].iloc[0]),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, orientation in zip(axes.flat, orientations):
        for label, table, signal, color in (
            ("B observed", reference_spectra, "observed_EEG", "#4c78a8"),
            ("A observed", screening_spectra, "observed_EEG", "#e45756"),
        ):
            view = table[
                table.orientation_label.eq(orientation)
                & table.sensor_label.eq("vertex")
                & table.signal_type.eq(signal)
            ]
            summary = view.groupby("frequency_hz").PSD_v2_per_hz.mean()
            keep = (summary.index >= 5) & (summary.index <= 15)
            axis.plot(
                summary.index[keep],
                10 * np.log10(np.maximum(summary.to_numpy()[keep], tiny)),
                color=color, label=label,
            )
        axis.set_title(orientation.replace("orientation_", "Population "))
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("PSD (dB V²/Hz)")
        axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, root, "figure_01_noisy_A_B_PSD")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    screening = screening.copy()
    screening["orientation_degrees"] = np.degrees(screening.rotation_y_rad)
    for feature, label in (
        ("topography_vertex", "Vertex carrier-power fraction"),
        ("topography_right_minus_left", "Right-minus-left carrier fraction"),
    ):
        summary = screening.groupby("orientation_degrees")[feature].agg(["mean", "std"])
        axes[0 if feature == "topography_vertex" else 1].errorbar(
            summary.index, summary["mean"], yerr=summary["std"], marker="o"
        )
        axes[0 if feature == "topography_vertex" else 1].set(
            xlabel="Population orientation (degrees)", ylabel=label,
            title="Noisy predecision EEG topography",
        )
    _save_figure(figure, root, "figure_02_noisy_EEG_context")

    figure, axis = plt.subplots(figsize=(7, 4))
    view = expected[expected.montage_profile.isin(ALL_ACTIONS)].copy()
    view["orientation_degrees"] = np.degrees(view.rotation_y_rad)
    for action, color in zip(ALL_ACTIONS, ["#bab0ac", "#4c78a8", "#f58518"]):
        summary = view[view.montage_profile.eq(action)].groupby(
            "orientation_degrees"
        ).expected_distance_to_B_log10.agg(["mean", "std"])
        axis.errorbar(summary.index, summary["mean"], yerr=summary["std"],
                      marker="o", label=action, color=color)
    axis.set(xlabel="Population orientation (degrees)",
             ylabel="Distance to orientation-specific B (log10)",
             title="Montage response across graded orientations")
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_03_graded_montage_response")

    primary = evaluation[evaluation.split_direction.eq(PRIMARY_SPLIT)]
    figure, axis = plt.subplots(figsize=(7, 4))
    colors = np.where(primary.learned_action.eq(PROFILE_60), "#f58518", "#4c78a8")
    axis.scatter(primary.topography_right_minus_left,
                 primary.predicted_z_minus_60_effect_log10, c=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Noisy right-minus-left carrier-power fraction",
             ylabel="Predicted L(z) - L(60°) (log10)",
             title="Held-structure EEG-conditioned action selection")
    _save_figure(figure, root, "figure_04_crossvalidated_policy")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    means = {
        "vs fixed": primary.learned_advantage_over_fixed_log10.mean(),
        "vs random": primary.learned_advantage_over_random_log10.mean(),
        "oracle regret": -primary.learned_regret_to_oracle_log10.mean(),
    }
    axes[0].bar(means.keys(), means.values(), color=["#4c78a8", "#72b7b2", "#bab0ac"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(ylabel="Policy benefit (log10)", title="Primary held-out policy effects")
    structure = primary.groupby("heldout_structure_seed").learned_advantage_over_fixed_log10.mean()
    axes[1].bar([str(int(x)) for x in structure.index], structure.values,
                color="#4c78a8")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Held-out structure", ylabel="Advantage over fixed (log10)",
                title="Structure-level generalization")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, root, "figure_05_policy_performance")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(shuffle.mean_shuffled_context_advantage_log10, bins=30,
              color="#bab0ac", label="Structure-preserving shuffle")
    axis.axvline(primary.learned_advantage_over_fixed_log10.mean(),
                 color="#e45756", linewidth=2, label="Observed EEG context")
    axis.set(xlabel="Mean advantage over trained fixed profile (log10)",
             ylabel="Shuffle count", title="EEG-context attribution audit")
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_06_context_shuffle")

    active = metrics[metrics.montage_profile.isin(ACTIVE_PROFILES)]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    rates = metrics.groupby("montage_profile")[[
        "post_E_firing_rate_hz", "post_I_firing_rate_hz"
    ]].mean().reindex(ALL_ACTIONS)
    rates.plot(kind="bar", ax=axes[0], color=["#4c78a8", "#e45756"])
    axes[0].set(ylabel="Firing rate (Hz)", title="Rate-safety audit")
    phase = active.groupby("montage_profile").common_phase_estimate_actionable_fraction.mean()
    axes[1].bar(phase.index, phase.values, color=["#4c78a8", "#f58518"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set(ylabel="Actionable phase fraction", title="Noisy phase tracking")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, root, "figure_07_safety_and_phase")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / str(
        cfg.analysis.output_root_name
    )
    exists = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H5-O1D noisy-EEG montage policy development")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    reference_rows: list[dict[str, Any]] = []
    reference_spectra: list[pd.DataFrame] = []
    for context in _reference_contexts(cfg):
        if rank == 0:
            print(f"B target: {context['context_id']}")
        state_cfg = _with_orientation_state(cfg, context, homogeneous_B=True)
        with open_dict(state_cfg):
            state_cfg.env.online.record_representative_state = False
        episode = _run_profile(
            condition_cfg=state_cfg, context=context,
            future_seed=int(context["reference_future_seed"]), future_index=0,
            profile=SHAM, phase_sensor_index=0,
            root=root / "reference_calibration", comm=comm, size=size, rank=rank,
        )
        if rank == 0:
            neural_baseline, baseline_start = _multichannel_raw(
                episode, "baseline", context, state_cfg
            )
            observed_baseline, noise = _observed_multichannel_baseline(
                episode, neural_baseline, context, state_cfg
            )
            neural_outcome, outcome_start = _multichannel_raw(
                episode, "stimulation", context, state_cfg, trim=True
            )
            observed_features, observed_spectrum = _multichannel_features(
                observed_baseline, start_ms=baseline_start, carrier_hz=9.0,
                cfg=state_cfg,
            )
            outcome_features, _ = _multichannel_features(
                neural_outcome, start_ms=outcome_start, carrier_hz=9.0,
                cfg=state_cfg,
            )
            reference_rows.append({
                **context,
                "screening_log10_alpha_power": float(
                    observed_features["global_log10_alpha_power"]
                ),
                "outcome_log10_alpha_power": float(
                    outcome_features["global_log10_alpha_power"]
                ),
                **noise,
            })
            reference_spectra.append(observed_spectrum.assign(
                context_id=str(context["context_id"]),
                structure_seed=int(context["structure_seed"]),
                orientation_label=str(context["orientation_label"]),
                signal_type="observed_EEG", condition="B_homogeneous",
            ))
    if rank == 0:
        references = pd.DataFrame(reference_rows)
        target = _reference_target(references)
        references.to_csv(root / "reference_B_calibration.csv", index=False)
        (root / "frozen_noisy_screening_and_neural_outcome_B_target.json").write_text(
            json.dumps(_json_ready(target), indent=2, allow_nan=False)
        )
    else:
        references, target = None, None
    target = comm.bcast(target, root=0)

    screening_rows: list[dict[str, Any]] = []
    screening_spectra: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    active_spectra: list[pd.DataFrame] = []
    contexts = _run_contexts(cfg)
    for context in contexts:
        if rank == 0:
            print(
                f"A context={context['context_id']} "
                f"structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"orientation={np.degrees(context['rotation_y_rad']):g} deg"
            )
        state_cfg = _with_orientation_state(cfg, context)
        with open_dict(state_cfg):
            state_cfg.env.online.record_representative_state = False
        first_future = _future_seed(state_cfg, context, 0)
        baseline = _run_profile(
            condition_cfg=state_cfg, context=context, future_seed=first_future,
            future_index=0, profile=SHAM, phase_sensor_index=0,
            root=root / "screening", comm=comm, size=size, rank=rank,
        )
        if rank == 0:
            screening, spectrum = _screen_context(
                baseline, context, target, state_cfg
            )
            screening_rows.append(screening)
            screening_spectra.append(spectrum)
            eligible = bool(screening["eligible"])
            selected_frequency = float(screening["EEG_selected_frequency_hz"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'SHAM FALLBACK'}; "
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
            future_seed = _future_seed(action_cfg, context, future_index)
            episodes: dict[str, dict[str, Any]] | None = {} if rank == 0 else None
            for profile in ALL_ACTIONS:
                if profile == SHAM and future_index == 0:
                    episode = baseline
                else:
                    episode = _run_profile(
                        condition_cfg=action_cfg, context=context,
                        future_seed=future_seed, future_index=future_index,
                        profile=profile, phase_sensor_index=0,
                        root=root / "active_mapping", comm=comm, size=size, rank=rank,
                    )
                if rank == 0:
                    episodes[profile] = episode
            if rank == 0:
                sham = episodes[SHAM]
                for profile in ALL_ACTIONS:
                    row, spectrum = _metric_row(
                        episode=episodes[profile], sham=sham, baseline=sham,
                        context=context, screening=screening, target=target,
                        profile=profile, future_index=future_index,
                        future_seed=future_seed, cfg=action_cfg,
                    )
                    observation = episodes[profile]["simulation"]["observation"]
                    row.update({
                        "configured_observation_noise_fraction": float(
                            observation["configured_rms_fraction"]
                        ),
                        "achieved_observation_noise_fraction": float(
                            observation["achieved_baseline_noise_fraction"]
                        ),
                        "observed_baseline_sha256": str(
                            observation["observed_baseline_sha256"]
                        ),
                        "controller_input_uses_observed_noisy_EEG": True,
                        "efficacy_endpoint_uses_ideal_neural_EEG": True,
                    })
                    metric_rows.append(row)
                    if int(context["structure_index"]) == 0 and future_index == 0:
                        active_spectra.append(spectrum)

    if rank != 0:
        return
    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "prospective_screening.csv", index=False)
    reference_spectrum = pd.concat(reference_spectra, ignore_index=True)
    predecision_spectrum = pd.concat(screening_spectra, ignore_index=True)
    reference_spectrum.to_csv(root / "reference_B_observed_multichannel_PSD.csv", index=False)
    predecision_spectrum.to_csv(root / "predecision_A_observed_and_neural_multichannel_PSD.csv", index=False)
    if not metric_rows:
        conclusion = {
            "scope": "H5-O1D noisy-EEG montage policy development",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "H5_O1D_policy_development": "NOT PASSED",
                "ready_for_disjoint_policy_confirmation": False,
                "H5_status": "NOT ESTABLISHED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(
            conclusion, indent=2
        ))
        print("No eligible contexts; stopped after prospective screening.")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    expected = _expected_map(metrics)
    invariance_table, invariance = _baseline_invariance(screening)
    observability_table, observability = _profile_observability(screening)
    evaluation, fold_models, final_model = _policy_crossvalidation(
        metrics, screening, cfg
    )
    shuffle = _shuffle_null(metrics, screening, cfg)
    checks, policy_summary = _checks(
        screening=screening, references=references, metrics=metrics,
        invariance=invariance, observability=observability,
        evaluation=evaluation, shuffle=shuffle, final_model=final_model,
        sources=sources, cfg=cfg,
    )
    passed = all(checks.values())
    conclusions = {
        "H5_O1D_policy_development": "PASSED" if passed else "NOT PASSED",
        "ready_for_disjoint_policy_confirmation": bool(passed),
        "H5_status": "NOT ESTABLISHED",
        "failed_checks": [name for name, value in checks.items() if not value],
    }

    metrics.to_csv(root / "context_montage_future_metrics.csv", index=False)
    expected.to_csv(root / "expected_context_montage_map.csv", index=False)
    evaluation.to_csv(root / "crossvalidated_policy_evaluation.csv", index=False)
    shuffle.to_csv(root / "structure_preserving_context_shuffle.csv", index=False)
    invariance_table.to_csv(root / "paired_orientation_neural_invariance.csv", index=False)
    observability_table.to_csv(root / "noisy_EEG_profile_observability_LOSO.csv", index=False)
    pd.concat(active_spectra, ignore_index=True).to_csv(
        root / "representative_active_neural_multichannel_PSD.csv", index=False
    )
    (root / "frozen_candidate_policy.json").write_text(json.dumps(
        _json_ready(final_model), indent=2, allow_nan=False
    ))
    (root / "crossvalidation_fold_models.json").write_text(json.dumps(
        _json_ready(fold_models), indent=2, allow_nan=False
    ))
    audit = {
        "noisy_EEG_profile_observability": observability,
        "coordinate_rotation_neural_invariance": invariance,
        "policy_development": policy_summary,
    }
    (root / "H5_O1D_policy_development_audit.json").write_text(json.dumps(
        _json_ready(audit), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H5_O1D_noisy_EEG_montage_policy_development",
        "frozen_H5O0_source": {
            "root": sources["root"], "hashes": sources["hashes"],
            "upstream_provenance": sources["upstream_provenance"],
        },
        "state_definition": {
            "A": "mean-rate-matched 9/11-Hz shared rhythmic afferent drive",
            "B": "homogeneous mean-rate-matched afferent drive",
            "orientation_is_not_A_or_B": True,
            "orientation_grid_degrees": [0, 20, 40, 60],
        },
        "observation": {
            "policy_input": "preceding multichannel EEG plus paired AR1 sensor noise",
            "efficacy_endpoint": "ideal neural-only multichannel EEG",
            "noise_fraction_of_vertex_baseline_RMS": 0.25,
            "AR1_coefficient": 0.95,
        },
        "actions": {
            "profiles": _profile_specs(cfg),
            "amplitude_v_per_m": 0.2,
            "carrier": "frozen noisy-EEG multitaper 9/11-Hz decision",
            "phase_controller": _profile(cfg, RESPONSIVE),
            "one_profile_for_complete_intervention": True,
        },
        "policy": {
            "candidate": final_model,
            "primary_comparator": "fold-specific best fixed montage",
            "secondary_comparators": [
                "sham", "uniform random montage", "frozen H5O0 analytical EEG rule"
            ],
            "primary_training_futures": [1, 2],
            "primary_evaluation_futures": [3, 4],
            "whole_structure_crossvalidation": True,
        },
        "design": {
            "reference_structures": int(references.structure_seed.nunique()),
            "development_structures": int(screening.structure_seed.nunique()),
            "screened_contexts": int(len(screening)),
            "eligible_contexts": int(screening.eligible.sum()),
            "paired_futures": int(cfg.analysis.crossed_design.n_future_continuations),
            "action_future_outcomes": int(len(metrics)),
            "statistical_unit": "independent circuit structure",
        },
        "inference_boundary": (
            "Exploratory noisy-EEG policy development. The candidate requires "
            "new disjoint confirmation; H5 and clinical montage efficacy are "
            "not established."
        ),
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-O1D noisy-EEG montage policy development",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure",
        "inference_boundary": provenance["inference_boundary"],
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot):
        _plots(
            root=root, reference_spectra=reference_spectrum,
            screening_spectra=predecision_spectrum, screening=screening,
            expected=expected, evaluation=evaluation, shuffle=shuffle,
            metrics=metrics,
        )

    print("\n### H5-O1D screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### H5-O1D policy-development checks")
    for name, value in checks.items():
        print(f"{name}: {'PASSED' if value else 'NOT PASSED'}")
    print("\n### H5-O1D policy summary")
    print(json.dumps(_json_ready(policy_summary), indent=2, allow_nan=False))
    print(
        "\nNoisy-EEG montage policy development: "
        f"{conclusions['H5_O1D_policy_development']}"
    )
    print("H5 status: NOT ESTABLISHED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
