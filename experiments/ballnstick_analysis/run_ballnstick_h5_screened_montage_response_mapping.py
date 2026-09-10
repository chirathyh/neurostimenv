"""H5-O2: frozen-screen active reassessment, not ML or confirmation.

Two distinct questions: does the EEG-geometry rule control the phenotype, and
is there consequential *residual* response opportunity beyond that rule?
Counterfactual futures are paired; structures, not futures, are independent.
Older experiment runners and simulator/controller code are deliberately reused
without modification. See H5_O2_PROTOCOL.md for claim boundaries and equations.
"""
from __future__ import annotations

import hashlib
import itertools
import json
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
from omegaconf import OmegaConf

sys.path.insert(1, config("MAIN_PATH"))
from experiments.ballnstick_analysis import (  # noqa: E402
    run_ballnstick_h5_controller_profile_feasibility as online,
    run_ballnstick_h5_montage_orientation_opportunity as montage,
    run_ballnstick_h5_rhythm_screening_validation as rhythm,
    run_ballnstick_h5_spatial_measurement_audit as spatial,
)

SHAM = montage.SHAM
ACTIVE = montage.ACTIVE_PROFILES
ACTIONS = montage.ALL_ACTIONS
PRIMARY = spatial.PRIMARY
SOURCE_FILES = {
    "conclusion": "experiment_conclusion.json",
    "completion": "run_complete.json",
    "target": "frozen_EEG_only_B_targets.json",
    "rule": "frozen_rhythm_presence_rule.json",
    "protocol": "prespecified_protocol.json",
    "provenance": "protocol_and_provenance.json",
    "config": "resolved_config.yaml",
    "trajectories": "trajectory_audit.csv",
    "spectra": "baseline_PSD.csv",
}
write_json = spatial._write_json


def _hash_array(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def _seed_values(value, key=""):
    if isinstance(value, dict):
        return set().union(*(_seed_values(v, k) for k, v in value.items()), set())
    if isinstance(value, list):
        return set().union(*(_seed_values(v, key) for v in value), set())
    return {int(value)} if "seed" in key and isinstance(value, int) else set()


def _load_source(cfg):
    root = Path(to_absolute_path(str(cfg.analysis.source_h5o1s.result_dir)))
    files, hashes = montage._hash_locked_files(
        root, SOURCE_FILES, cfg.analysis.source_h5o1s.expected_sha256
    )
    conclusion = json.loads(files["conclusion"].read_text())
    complete = json.loads(files["completion"].read_text())
    if (conclusion["status"] != "PASSED" or conclusion["smoke_test"]
            or not conclusion["ready_for_small_response_reassessment"]
            or not complete["completed"]):
        raise ValueError("Requires the hash-locked, completed positive H5-O1S.")
    source = {k: json.loads(files[k].read_text()) for k in (
        "target", "rule", "protocol", "provenance"
    )}
    source.update(root=str(root), hashes=hashes, config=OmegaConf.load(files["config"]))
    seeds = _seed_values(source)
    table = pd.read_csv(files["trajectories"])
    for col in table:
        if col.endswith("seed") and pd.api.types.is_numeric_dtype(table[col]):
            seeds.update(table[col].dropna().astype(int))
    source["seed_union"] = seeds
    return source


def _contexts(cfg):
    rows = montage._contexts(cfg, apply_smoke_limit=False)
    if cfg.analysis.smoke_test and cfg.analysis.smoke_context_limit:
        # Two endpoints exercise both montage classes in the integration smoke.
        rows = sorted(rows, key=lambda r: (
            r["structure_index"], r["hidden_frequency_hz"],
            min(abs(r["rotation_y_rad"]), abs(r["rotation_y_rad"] - np.pi/3)),
            r["rotation_y_rad"],
        ))[:int(cfg.analysis.smoke_context_limit)]
    return rows


def _noise_seeds(cfg, context, future_index, sensor):
    # The online episode helper sets experiment.seed to context.trial_seed
    # before constructing its noise, not to the top-level experiment seed.
    if sensor == 0:
        base = int(context["trial_seed"]) + int(cfg.analysis.observation_noise.seed_offset)
        group = int(context["future_group_index"])
        return base + 100 * group, base + 10000 + 100 * group + future_index
    history = (int(context["trial_seed"])
               + int(cfg.analysis.observation_noise.multichannel_side_seed_offset)
               + 100 * int(context["future_group_index"]) + 10 * sensor)
    return history, history + 10000 + future_index


def _validate(cfg, source):
    smoke = bool(cfg.analysis.smoke_test)
    frozen = [
        "env.network", "env.eeg", "env.online", "analysis.states",
        "analysis.spatial_measurement", "analysis.multitaper", "analysis.iaf",
        "analysis.target_fs_hz", "analysis.low_hz", "analysis.high_hz",
        "analysis.eeg_array", "analysis.actions", "analysis.tacs",
        "analysis.reference", "analysis.rate_guardrails_hz", "analysis.rate_reference_tolerance_fraction",
        "analysis.controller_profiles",
        "analysis.criteria", "analysis.simulator", "analysis.inhibition_scale",
    ]
    if not smoke:
        frozen += ["analysis.timeline"]
    for path in frozen:
        actual = OmegaConf.select(cfg, path)
        old = OmegaConf.select(source["config"], path)
        if actual != old:
            raise ValueError(f"Frozen O1S setting changed: {path}")
    for key in ("enabled", "rms_fraction_of_baseline_neural_eeg", "ar1_coefficient"):
        if cfg.analysis.observation_noise[key] != source["config"].analysis.observation_noise[key]:
            raise ValueError(f"Frozen observation noise changed: {key}")
    if (not smoke and cfg.analysis.smoke_force_eligible) or int(cfg.env.simulation.obs_win_len) != 1000:
        raise ValueError("Eligibility forcing is smoke-only; online windows must be 1 s.")
    rows = _contexts(cfg)
    if not rows:
        raise ValueError("No contexts configured.")
    n_futures = int(cfg.analysis.crossed_design.n_future_continuations)
    if n_futures < 2 or int(cfg.analysis.timeline.baseline_steps) < 4:
        raise ValueError("Even a smoke needs two futures and a four-second baseline.")
    if not smoke and (len(rows) != 24 or n_futures != 4
                      or int(cfg.analysis.crossed_design.n_history_seeds) != 1):
        raise ValueError("Bounded full design is 3 x 2 x 4 contexts, one history, four futures.")
    spaces = [{int(r[k]) for r in rows} for k in (
        "structure_seed", "history_seed", "phase_seed", "trial_seed"
    )]
    spaces.append({online._future_seed(cfg, r, f) for r in rows for f in range(n_futures)})
    spaces.append({s for r in rows for f in range(n_futures) for sensor in range(3)
                   for s in _noise_seeds(cfg, r, f, sensor)})
    if any(a & b for a, b in itertools.combinations(spaces, 2)):
        raise ValueError("New seed namespaces overlap.")
    if set.union(*spaces) & source["seed_union"]:
        raise ValueError("Seeds overlap frozen source.")
    # Prior structure namespaces are exhausted up to 429430. Avoid uint32
    # overflow in the inherited structure*10000 + rank seed mapping.
    if min(spaces[0]) <= 429430 or max(spaces[0]) * 10000 + 9999 > 2**32 - 1:
        raise ValueError("Use new structures above 429430 within the uint32 mapping.")
    return rows


def _unit_noise(cfg, context, future_index):
    dt = float(cfg.env.network.dt)
    timeline = cfg.analysis.timeline
    total = int(round(sum(int(timeline[f"{x}_steps"]) for x in (
        "burn_in", "baseline", "stimulation", "washout"
    )) * 1000 / dt))
    split = int(round((timeline.burn_in_steps + timeline.baseline_steps) * 1000 / dt))
    seeds = [_noise_seeds(cfg, context, future_index, sensor) for sensor in range(3)]
    paths = np.stack([online._ar1_path(
        n_samples=total, split_sample=split, history_seed=h, future_seed=f,
        coefficient=float(cfg.analysis.observation_noise.ar1_coefficient)
    ) for h, f in seeds])
    return paths, split, np.asarray(seeds, dtype=np.int64)


def _measure(values, fs, model, cfg):
    frequency, csd, covariance = spatial._cross_spectrum(values, fs, cfg)
    estimate = spatial._estimate_spatial(covariance, model, cfg)[PRIMARY]
    if not np.isfinite(estimate["geometry_normalized_log10_alpha"]):
        raise RuntimeError("Nonfinite EEG endpoint.")
    return estimate, frequency, csd


def _epoch_measurement(episode, context, epoch, unit_noise, cfg, model, *, trim=False):
    neural, start = montage._multichannel_raw(episode, epoch, context, cfg, trim=trim)
    index = int(round(start / float(cfg.env.network.dt)))
    noise = unit_noise[:, index:index + neural.shape[1]]
    info = episode["simulation"]["observation"]
    if _hash_array(unit_noise[0]) != info["unit_noise_sha256"]:
        raise RuntimeError(f"Reconstructed unit noise differs from online controller (recorded seeds "
                           f"{info['history_noise_seed']}, {info['future_noise_seed']}).")
    scale = float(info["baseline_neural_rms_v"]) * float(info["configured_rms_fraction"])
    observed = neural + scale * noise
    recorded = np.asarray(episode["observed_raw_by_epoch"][epoch])
    if trim:
        count = int(round(cfg.analysis.timeline.stimulation_analysis_trim_ms / cfg.env.network.dt))
        if count:
            recorded = recorded[count:-count]
    error = montage._relative_rms_error(recorded, observed[0])
    if error > 1e-10:
        raise RuntimeError(f"Reconstructed observed EEG disagrees with online EEG: {error}")
    # Exactly the frozen measurement preprocessing convention: process each
    # epoch separately and exploit linearity. No postdecision samples in baseline.
    processed_neural, fs = spatial._preprocess_channels(neural, cfg)
    processed_noise, _ = spatial._preprocess_channels(noise, cfg)
    processed_observed = processed_neural + scale * processed_noise
    measurements = {}
    spectra = []
    for view, data in (("neural_audit", processed_neural), ("observed", processed_observed)):
        estimate, frequency, csd = _measure(data, fs, model, cfg)
        measurements[view] = estimate
        keep = (frequency >= 1) & (frequency <= 30)
        for sensor in range(3):
            spectra.append(pd.DataFrame({
                "frequency_hz": frequency[keep], "PSD_v2_per_hz": csd[sensor, sensor, keep].real,
                "sensor_index": sensor, "signal_view": view, "epoch": epoch,
            }))
    return {
        "estimates": measurements, "neural": processed_neural,
        "observed": processed_observed, "fs_hz": fs, "start_ms": start,
        "neural_raw_sha256": _hash_array(neural),
        "observed_raw_sha256": _hash_array(observed),
        "online_observed_relative_error": error,
        "spectra": pd.concat(spectra, ignore_index=True),
    }


def _screen(baseline, source, cfg):
    """No context, latent generator, rates, B-of-this-seed, or future inputs."""
    observed = baseline["observed"]
    fs = baseline["fs_hz"]
    values = dict(baseline["estimates"]["observed"])
    values.update(spatial._carrier_and_phase(observed[0], fs, cfg))
    estimates, _, _ = montage._estimate_multitaper_methods(
        observed[0], fs_hz=fs, hidden_frequency_hz=float("nan"),
        input_signal=montage.OBSERVED, cfg=cfg
    )
    selected = next(x for x in estimates if x["estimator"] == montage.MT_POOLED)
    values[rhythm.SCORE] = float(selected[rhythm.SCORE])
    target = source["target"][f"observed/{PRIMARY}"]
    values.update(rhythm._screen(values, target, source["rule"], cfg))
    values["analytical_action"] = values["predicted_profile"] if values["treatment_eligible"] else SHAM
    values["fallback_action"] = SHAM if not values["treatment_eligible"] else "not_needed"
    return values


def _carrier_ppc(episode, population, frequency, cfg):
    """Finite-count corrected fixed-carrier spike locking, central endpoint.

    Recompute also for the reused first sham, which was simulated before the
    EEG carrier decision. This is a hidden mechanism audit, not a policy input
    or a claim about pairwise neuronal desynchronization.
    """
    outputs = episode["simulation"]["outputs_by_epoch"]["stimulation"]
    spikes = np.concatenate([np.asarray(x["spikes"][population]["times_ms"]) for x in outputs])
    trim = float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    start = float(outputs[0]["t_start_ms"]) + trim
    stop = float(outputs[-1]["t_stop_ms"]) - trim
    spikes = spikes[(spikes > start) & (spikes <= stop)]
    n = len(spikes)
    if n < 2:
        return float("nan")
    total = np.exp(2j*np.pi*float(frequency)*spikes/1000).sum()
    return float((abs(total)**2-n)/(n*(n-1)))


def _metric(episode, sham, context, screen, measurement, sham_measurement,
            washout, sham_washout, history, future_index, cfg, source):
    target = float(source["target"][f"neural_audit/{PRIMARY}"]["outcome_mean_log10"])
    y = measurement["estimates"]["neural_audit"]["geometry_normalized_log10_alpha"]
    y_sham = sham_measurement["estimates"]["neural_audit"]["geometry_normalized_log10_alpha"]
    y_noise = measurement["estimates"]["observed"]["geometry_normalized_log10_alpha"]
    wy = washout["estimates"]["neural_audit"]["geometry_normalized_log10_alpha"]
    ws = sham_washout["estimates"]["neural_audit"]["geometry_normalized_log10_alpha"]
    recovered, tolerance = montage._field_removal_status(
        effect_log10=y_sham - y, residual_log10=ws - wy, cfg=cfg
    )
    outcome = montage._epoch_row(episode, "stimulation")
    sham_outcome = montage._epoch_row(sham, "stimulation")
    updates = pd.DataFrame(episode["simulation"]["phase_updates"])
    action = episode["simulation"]["action"]["montage_profile"]
    profile_spec = next(p for p in montage._profile_specs(cfg)
                        if p["montage_profile"] == (ACTIVE[0] if action == SHAM else action))
    projection = 0.0 if action == SHAM else montage._field_projection(context, profile_spec)
    ppc_e = _carrier_ppc(episode, "E", screen["EEG_selected_frequency_hz"], cfg)
    ppc_i = _carrier_ppc(episode, "I", screen["EEG_selected_frequency_hz"], cfg)
    ppc_sham = _carrier_ppc(sham, "E", screen["EEG_selected_frequency_hz"], cfg)
    row = {
        **context, "future_index": future_index + 1, "montage_profile": action,
        "future_drive_seed": int(episode["simulation"]["future_drive_seed"]),
        "analytical_action": screen["predicted_profile"],
        "EEG_selected_frequency_hz": screen["EEG_selected_frequency_hz"],
        "estimated_orientation_deg": screen["estimated_orientation_deg"],
        "configured_amplitude_v_per_m": episode["simulation"]["action"]["ac_amplitude_v_per_m"],
        "applied_carrier_hz": episode["simulation"]["action"]["frequency_hz"],
        "relative_phase_offset_rad": episode["simulation"]["action"]["eeg_relative_phase_offset_rad"],
        "phase_history_ms": episode["simulation"]["action"]["phase_history_ms"],
        "refresh_interval_ms": episode["simulation"]["action"]["update_interval_ms"],
        "post_neural_log10_alpha": y, "post_observed_log10_alpha_audit": y_noise,
        "observed_minus_neural_log10_audit": y_noise - y,
        "frozen_B_neural_target_log10": target,
        "loss_log10": abs(y - target), "sham_loss_log10": abs(y_sham - target),
        "improvement_vs_sham_log10": abs(y_sham - target) - abs(y - target),
        "alpha_suppression_vs_sham_log10": y_sham - y,
        "below_B_target": bool(y < target),
        "effective_axial_amplitude_v_per_m": float(cfg.analysis.actions.amplitude_v_per_m) * projection,
        "post_E_firing_rate_hz": float(outcome.E_firing_rate_hz),
        "post_I_firing_rate_hz": float(outcome.I_firing_rate_hz),
        "post_E_ppc": ppc_e, "post_I_ppc": ppc_i,
        "hidden_E_ppc_reduction_vs_sham": ppc_sham - ppc_e,
        "rate_safe": bool(montage._relative_rate_safe(outcome, sham_outcome, cfg)),
        "washout_residual_log10": ws - wy, "washout_tolerance_log10": tolerance,
        "physiological_washout_recovered": bool(recovered),
        "final_extracellular_residual_mV": float(episode["simulation"]["final_residual_mV"]),
        "baseline_neural_sha256": history["neural_raw_sha256"],
        "baseline_observed_sha256": history["observed_raw_sha256"],
        "baseline_spikes_sha256": montage._spike_hash(episode, "baseline"),
        "baseline_local_dipole_sha256": _hash_array(montage._dipole_by_epoch(episode, "baseline")),
        "original_vertex_unit_noise_sha256": episode["simulation"]["observation"]["unit_noise_sha256"],
        "achieved_noise_RMS_fraction": episode["simulation"]["observation"]["achieved_baseline_noise_fraction"],
        "online_observed_reconstruction_error": measurement["online_observed_relative_error"],
        "phase_update_count": len(updates),
        "all_phase_estimates_causal": bool(updates.estimate_is_strictly_causal.all()),
        "max_frequency_correction_hz": float(updates.frequency_correction_hz.abs().max()),
        "max_field_boundary_discontinuity_v_per_m": float(updates.field_boundary_discontinuity_v_per_m.max()),
        "phase_actionable_fraction": float(np.mean(updates.common_audit_resultant_to_rms >= .03)),
    }
    return row, updates.assign(context_id=context["context_id"], future_index=future_index+1,
                               montage_profile=action, structure_seed=context["structure_seed"])


def _structure_mean(table, column):
    return float(table.groupby("structure_seed")[column].mean().mean())


def _inference(values, cfg):
    """Exploratory intervals; three structures cannot confirm a one-sided 5% test."""
    x = np.asarray(values, float)
    if not len(x):
        return {"n_structures": 0, "mean": None, "exact_one_sided_p": None}
    rng = np.random.default_rng(int(cfg.analysis.mapping_criteria.inference_seed))
    boot = rng.choice(x, size=(int(cfg.analysis.mapping_criteria.bootstrap_repetitions), len(x))).mean(axis=1)
    null = np.asarray([np.mean(x * s) for s in itertools.product([-1, 1], repeat=len(x))])
    return {
        "n_structures": len(x), "mean": float(x.mean()),
        "positive_structure_fraction": float(np.mean(x > 0)),
        "structure_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
        "exact_one_sided_p": float(np.mean(null >= x.mean() - 1e-14)),
        "minimum_attainable_one_sided_p": 2.0**(-len(x)),
        "confirmatory": False,
    }


def _response_analysis(metrics, cfg):
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"available": False}
    keys = ["context_id", "structure_seed", "hidden_frequency_hz", "orientation_label",
            "rotation_y_rad", "analytical_action", "montage_profile"]
    expected = metrics.groupby(keys, as_index=False).loss_log10.mean()
    wide = expected.pivot(index=keys[:-1], columns="montage_profile", values="loss_log10").reset_index()
    wide["empirical_oracle_action"] = wide[ACTIVE].idxmin(axis=1)
    wide["empirical_oracle_loss"] = wide[ACTIVE].min(axis=1)
    wide["analytical_loss"] = [row[row.analytical_action] for _, row in wide.iterrows()]
    wide["analytical_improvement_vs_sham"] = wide[SHAM] - wide.analytical_loss
    wide["uniform_random_active_loss"] = wide[ACTIVE].mean(axis=1)
    wide["analytical_advantage_vs_random"] = wide.uniform_random_active_loss - wide.analytical_loss
    wide["active_margin_log10"] = abs(wide[ACTIVE[0]] - wide[ACTIVE[1]])
    best_fixed = min(ACTIVE, key=lambda a: _structure_mean(wide, a))
    wide["empirical_oracle_advantage_vs_fixed"] = wide[best_fixed] - wide.empirical_oracle_loss
    wide["empirical_oracle_advantage_vs_analytical"] = wide.analytical_loss - wide.empirical_oracle_loss
    wide["analytical_advantage_vs_fixed"] = wide[best_fixed] - wide.analytical_loss
    # Select actions on one pair of futures, evaluate on a disjoint pair.
    # Reverse split is prespecified replication, not an alternative primary test.
    future_ids = sorted(metrics.future_index.unique())
    half = len(future_ids) // 2
    splits = [("primary", future_ids[:half], future_ids[half:]),
              ("reverse", future_ids[half:], future_ids[:half])]
    split_rows = []
    for label, train_ids, test_ids in splits:
        def table(ids):
            return metrics[metrics.future_index.isin(ids)].groupby(
                ["context_id", "structure_seed", "analytical_action", "montage_profile"]
            ).loss_log10.mean().unstack("montage_profile").reset_index()
        train, test = table(train_ids), table(test_ids)
        fixed = min(ACTIVE, key=lambda a: _structure_mean(train, a))
        for _, tr in train.iterrows():
            ev = test[test.context_id.eq(tr.context_id)].iloc[0]
            chosen = min(ACTIVE, key=lambda a: tr[a])
            split_rows.append({
                "split": label, "context_id": tr.context_id, "structure_seed": tr.structure_seed,
                "selection_futures": str(train_ids), "evaluation_futures": str(test_ids),
                "selected_action": chosen, "training_best_fixed": fixed,
                "analytical_action": ev.analytical_action,
                "selected_loss": ev[chosen], "fixed_loss": ev[fixed],
                "analytical_loss": ev[ev.analytical_action],
                "selected_advantage_vs_fixed": ev[fixed] - ev[chosen],
                "selected_advantage_vs_analytical": ev[ev.analytical_action] - ev[chosen],
                "selected_improvement_vs_sham": ev[SHAM] - ev[chosen],
            })
    split_table = pd.DataFrame(split_rows)
    cols = ["selected_advantage_vs_fixed", "selected_advantage_vs_analytical", "selected_improvement_vs_sham"]
    structures = split_table.groupby(["split", "structure_seed"], as_index=False)[cols].mean()
    summary = {
        "available": True, "best_fixed_profile_in_sample": best_fixed,
        "fixed_profile_expected_loss": {a: _structure_mean(wide, a) for a in ACTIVE},
        "empirical_oracle_is_optimistic_not_deployable": True,
        "empirical_oracle_advantage_vs_fixed": _structure_mean(wide, "empirical_oracle_advantage_vs_fixed"),
        "empirical_oracle_advantage_vs_analytical": _structure_mean(wide, "empirical_oracle_advantage_vs_analytical"),
        "analytical_improvement_vs_sham": _inference(wide.groupby("structure_seed").analytical_improvement_vs_sham.mean(), cfg),
        "analytical_advantage_vs_fixed": _inference(wide.groupby("structure_seed").analytical_advantage_vs_fixed.mean(), cfg),
        "analytical_advantage_vs_random": _inference(wide.groupby("structure_seed").analytical_advantage_vs_random.mean(), cfg),
        "future_split_inference": {label: {
            col: _inference(group[col], cfg) for col in cols
        } for label, group in structures.groupby("split")},
        "expected_optimal_profile_counts": wide.empirical_oracle_action.value_counts().to_dict(),
        "analytical_rule_profile_counts": wide.analytical_action.value_counts().to_dict(),
        "one_history_per_structure_frequency": True,
    }
    return wide, split_table, structures, summary


def _checks(screening, metrics, wide, summary, cfg):
    eligible = screening[screening.enrolled]
    criteria = cfg.analysis.mapping_criteria
    active = metrics[metrics.montage_profile.ne(SHAM)] if len(metrics) else metrics
    integrity = {
        "positive_O1S_hash_locked_and_measurement_frozen": True,
        "new_disjoint_seed_namespaces": True,
        "screening_uses_only_predecision_observed_EEG": True,
        "no_policy_or_threshold_fitted": True,
        "ideal_neural_EEG_target_not_noisy_B_target": True,
        "minimum_eligible_contexts": len(eligible) >= int(criteria.minimum_eligible_contexts),
        "minimum_independent_structures": eligible.structure_seed.nunique() >= int(criteria.minimum_structures),
        "all_four_orientations_represented": (eligible.orientation_label.nunique() == 4 and
            eligible.groupby("orientation_label").size().min() >= int(criteria.minimum_contexts_per_orientation)),
        "both_carriers_represented": eligible.hidden_frequency_hz.nunique() == 2,
        "enrolled_carrier_accuracy": bool(len(eligible) and np.mean(np.isclose(
            eligible.EEG_selected_frequency_hz, eligible.hidden_frequency_hz)) >= float(criteria.minimum_carrier_accuracy)),
        "enrolled_geometry_accuracy": bool(len(eligible) and np.mean(
            eligible.predicted_profile == eligible.matched_profile) >= float(criteria.minimum_geometry_accuracy)),
        "all_enrolled_pass_actual_frozen_screen": bool(len(eligible) and eligible.treatment_eligible.all()),
        "complete_paired_action_future_grid": bool(len(metrics) and
            len(metrics) == len(eligible) * 3 * int(cfg.analysis.crossed_design.n_future_continuations)
            and not metrics.duplicated(["context_id", "future_index", "montage_profile"]).any()),
    }
    if len(metrics):
        grouped = metrics.groupby("context_id")
        integrity.update({
            "identical_predecision_neural_EEG": bool(grouped.baseline_neural_sha256.nunique().eq(1).all()),
            "identical_predecision_observed_EEG": bool(grouped.baseline_observed_sha256.nunique().eq(1).all()),
            "identical_predecision_spikes": bool(grouped.baseline_spikes_sha256.nunique().eq(1).all()),
            "orientation_does_not_change_prestimulation_local_dynamics": bool(metrics.groupby(
                "paired_orientation_context_id").baseline_local_dipole_sha256.nunique().eq(1).all()),
            "same_future_sensor_noise_across_actions": bool(metrics.groupby(
                ["context_id", "future_index"]).original_vertex_unit_noise_sha256.nunique().eq(1).all()),
            "future_external_drive_seeds_paired_across_actions": bool(metrics.groupby(
                ["context_id", "future_index"]).future_drive_seed.nunique().eq(1).all()),
            "independent_external_drive_future_seeds": bool(grouped.future_drive_seed.nunique().eq(
                int(cfg.analysis.crossed_design.n_future_continuations)).all()),
            "future_sensor_noise_is_distinct": bool(grouped.original_vertex_unit_noise_sha256.nunique().eq(
                int(cfg.analysis.crossed_design.n_future_continuations)).all()),
            "observed_EEG_matches_online_controller": bool(metrics.online_observed_reconstruction_error.max() <= 1e-10),
            "all_phase_estimates_causal": bool(metrics.all_phase_estimates_causal.all()),
            "field_waveform_continuous": bool(metrics.max_field_boundary_discontinuity_v_per_m.max() <= 1e-10),
            "phase_correction_frequency_bounded": bool(metrics.max_frequency_correction_hz.max() <= 2.000000001),
            "active_controllers_refresh_after_onset": bool(len(active) and active.phase_update_count.min() > 1),
            "active_profiles_have_equal_frozen_dose": bool(len(active) and np.allclose(active.configured_amplitude_v_per_m, .2)),
            "active_carrier_selected_only_from_EEG": bool(len(active) and np.allclose(active.applied_carrier_hz, active.EEG_selected_frequency_hz)),
            "active_profiles_keep_frozen_controller": bool(len(active) and np.allclose(active.phase_history_ms, 500)
                and np.allclose(active.refresh_interval_ms, 125) and np.allclose(active.relative_phase_offset_rad, np.pi)),
            "phase_estimates_actionable": bool(len(active) and active.phase_actionable_fraction.mean() >= .8),
            "all_actions_rate_safe": bool(metrics.rate_safe.all()),
            "exact_field_removal": bool(metrics.final_extracellular_residual_mV.eq(0).all()),
            "physiological_washout_recovered": bool(len(active) and
                active.physiological_washout_recovered.mean() >= float(cfg.analysis.criteria.minimum_physiological_washout_recovery_fraction)),
        })
    else:
        integrity["active_outcomes_available"] = False
    practical = float(criteria.practical_advantage_log10)
    mechanism = {"analytical_rule_uses_both_profiles": bool(len(wide) and wide.analytical_action.nunique() == 2)}
    for contrast in ("analytical_improvement_vs_sham", "analytical_advantage_vs_fixed"):
        infer = summary.get(contrast, {})
        mechanism[contrast + "_practical"] = bool(infer.get("mean", -np.inf) >= practical)
        mechanism[contrast + "_across_structures"] = bool(
            infer.get("positive_structure_fraction", 0) >= float(criteria.minimum_positive_structure_fraction))
    opportunity = {}
    for profile in ACTIVE:
        group = wide[(wide.empirical_oracle_action == profile) & (wide.active_margin_log10 >= practical)] if len(wide) else wide
        opportunity[f"practical_{profile}_support"] = bool(len(group) >= int(criteria.minimum_practical_contexts_per_profile)
            and group.structure_seed.nunique() >= int(criteria.minimum_practical_structures_per_profile))
    for comparator in ("fixed", "analytical"):
        opportunity[f"empirical_oracle_headroom_over_{comparator}"] = bool(
            summary.get(f"empirical_oracle_advantage_vs_{comparator}", -np.inf) >= practical)
        for label in ("primary", "reverse"):
            infer = summary.get("future_split_inference", {}).get(label, {}).get(f"selected_advantage_vs_{comparator}", {})
            opportunity[f"{label}_independent_future_advantage_over_{comparator}"] = bool(
                infer.get("mean", -np.inf) >= practical and
                infer.get("positive_structure_fraction", 0) >= float(criteria.minimum_positive_structure_fraction))
    for label in ("primary", "reverse"):
        infer = summary.get("future_split_inference", {}).get(label, {}).get("selected_improvement_vs_sham", {})
        opportunity[f"{label}_selected_active_improves_over_sham"] = bool(
            infer.get("mean", -np.inf) >= practical and
            infer.get("positive_structure_fraction", 0) >= float(criteria.minimum_positive_structure_fraction))
    return integrity, mechanism, opportunity


def _readiness(integrity, mechanism, opportunity, *, smoke):
    clean = bool(integrity and all(integrity.values()) and not smoke)
    # An independently beneficial alternative can be useful even when actual
    # neural response contradicts the maximum-field-projection heuristic.
    return (clean and bool(mechanism) and all(mechanism.values()),
            clean and bool(opportunity) and all(opportunity.values()))


def _plots(root, screening, metrics, wide, splits, spectra, source, cfg):
    if metrics.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off"); ax.text(.05, .5, "No enrolled contexts: no active efficacy claim.")
        montage._save_figure(fig, root, "figure_01_no_enrollment")
        return
    # Plot actual measurements, not a post-hoc synthetic reference spectrum.
    reference = pd.read_csv(Path(source["root"]) / SOURCE_FILES["spectra"])
    reference = reference[(reference.condition == "B") & (reference.signal_view == "neural_audit")
                          & (reference.sensor_index == 0)]
    frequencies = sorted(screening.hidden_frequency_hz.unique())
    angles = sorted(screening.rotation_y_rad.unique())
    fig, axes = plt.subplots(len(frequencies), len(angles),
                             figsize=(3.75*len(angles), 3.5*len(frequencies)), sharex=True, squeeze=False)
    for i, f in enumerate(frequencies):
        for j, angle in enumerate(angles):
            ax = axes[i, j]
            g = spectra[(spectra.hidden_frequency_hz == f) & np.isclose(spectra.rotation_y_rad, angle)
                        & (spectra.signal_view == "neural_audit") & (spectra.sensor_index == 0)]
            for (epoch, action), sub in g.groupby(["epoch", "montage_profile"]):
                if epoch == "washout" or (epoch == "baseline" and action != SHAM):
                    continue
                # First average repeats, then structures; every structure one vote.
                psd = sub.groupby(["structure_seed", "frequency_hz"]).PSD_v2_per_hz.mean().groupby("frequency_hz").mean()
                ax.semilogy(psd.index, psd, label=f"{epoch}: {action.replace('montage_profile_', '')}")
            ref = reference[np.isclose(reference.true_orientation_deg, np.degrees(angle))]
            psd = ref.groupby(["structure_seed", "frequency_hz"]).PSD_v2_per_hz.mean().groupby("frequency_hz").mean()
            ax.semilogy(psd.index, psd, "k--", label="Frozen B baseline (30 s)")
            ax.set(xlim=(7, 13), title=f"{f:g} Hz / {np.degrees(angle):g}°", xlabel="Frequency (Hz)", ylabel="Vertex PSD (V²/Hz)")
    axes[0, 0].legend(fontsize=6)
    baseline_s = int(cfg.analysis.timeline.baseline_steps)
    outcome_s = float(cfg.analysis.timeline.stimulation_steps) - 2*float(cfg.analysis.timeline.stimulation_analysis_trim_ms)/1000
    prefix = "SMOKE, not scientific evidence. " if cfg.analysis.smoke_test else ""
    fig.suptitle(f"{prefix}Neural-only PSD: baseline {baseline_s} s; central endpoint {outcome_s:g} s\n"
                 f"Welch segments up to {cfg.analysis.spatial_measurement.segment_seconds:g} s")
    fig.tight_layout(); montage._save_figure(fig, root, "figure_01_alpha_PSD")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {SHAM: "0.4", ACTIVE[0]: "tab:blue", ACTIVE[1]: "tab:orange"}
    for action in ACTIONS:
        g = metrics[metrics.montage_profile.eq(action)].groupby(
            ["structure_seed", "rotation_y_rad"]).loss_log10.mean().reset_index()
        for structure, sub in g.groupby("structure_seed"):
            axes[0].plot(np.degrees(sub.rotation_y_rad), sub.loss_log10, "o-", alpha=.65, color=colors[action],
                         label=action.replace("montage_profile_", "") if structure == g.structure_seed.min() else None)
    axes[0].set(xlabel="True orientation (audit, degrees)", ylabel="Absolute neural log-alpha distance to B")
    axes[0].legend(fontsize=8)
    for label, g in splits.groupby("split"):
        effects = g.groupby("structure_seed")[["selected_advantage_vs_fixed", "selected_advantage_vs_analytical"]].mean()
        for i, col in enumerate(effects):
            axes[1].scatter(np.repeat(i + (0 if label == "primary" else .15), len(effects)), effects[col], label=f"{label}: {col}")
    axes[1].axhline(.01, ls="--", color="gray"); axes[1].axhline(0, color="black", lw=.6)
    axes[1].set(xticks=[.075, 1.075], xticklabels=["versus fixed", "versus EEG geometry"], ylabel="Independent-future advantage (log10)")
    axes[1].legend(fontsize=6)
    fig.tight_layout(); montage._save_figure(fig, root, "figure_02_response_and_residual_opportunity")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for enrolled, group in screening.groupby("enrolled"):
        axes[0].scatter(np.degrees(group.rotation_y_rad), group.estimated_orientation_deg,
                        label="enrolled" if enrolled else "abstained", color="tab:blue" if enrolled else "0.6")
    axes[0].plot([0, 60], [0, 60], "k--"); axes[0].axhline(30, ls=":")
    axes[0].set(xlabel="True orientation (audit)", ylabel="Observed-EEG estimate (degrees)")
    axes[0].legend(fontsize=7)
    active = metrics[metrics.montage_profile.ne(SHAM)]
    axes[1].scatter(active.alpha_suppression_vs_sham_log10, active.hidden_E_ppc_reduction_vs_sham, s=12, alpha=.5)
    axes[1].axhline(0, color="gray"); axes[1].axvline(0, color="gray")
    axes[1].set(xlabel="Neural alpha suppression (log10)", ylabel="Hidden E PPC reduction")
    axes[2].scatter(metrics.post_neural_log10_alpha, metrics.post_observed_log10_alpha_audit, s=12, alpha=.5)
    lim = [metrics.post_neural_log10_alpha.min(), metrics.post_neural_log10_alpha.max()]
    axes[2].plot(lim, lim, "k--")
    axes[2].set(xlabel="Ideal neural log-alpha", ylabel="Noisy log-alpha (audit only)")
    fig.tight_layout(); montage._save_figure(fig, root, "figure_03_measurement_and_mechanism")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, action in enumerate(ACTIONS):
        g = metrics[metrics.montage_profile.eq(action)]
        for ax, col in zip(axes, ["post_E_firing_rate_hz", "post_I_firing_rate_hz", "washout_residual_log10"]):
            values = g.groupby("structure_seed")[col].mean()
            ax.scatter(np.repeat(i, len(values)), values)
            ax.set(xticks=range(3), xticklabels=["sham", "z", "60°"], ylabel=col)
    fig.tight_layout(); montage._save_figure(fig, root, "figure_04_safety_and_washout")
    first = str(metrics.context_id.iloc[0])
    fig, axes = plt.subplots(2, 3, figsize=(13, 5), sharex=True)
    for i, action in enumerate(ACTIONS):
        files = sorted((root / "representative_traces").glob(f"{first}_fut*_{action}.npz"))
        if not files:
            continue
        with np.load(files[0]) as data:
            t = (data["time_ms"]-data["time_ms"][0]) / 1000
            thin = max(1, int(round(2.0 / float(data["dt_ms"]))))
            axes[0, i].plot(t[::thin], data["observed_eeg_v"][::thin]*1e9, lw=.4, alpha=.5, label="observed")
            axes[0, i].plot(t[::thin], data["neural_eeg_v"][::thin]*1e9, lw=.5, label="neural")
            axes[1, i].plot(t[::thin], data["field_v_per_m"][::thin], lw=.5)
        axes[0, i].set(title=action.replace("montage_profile_", ""), ylabel="Vertex EEG (nV)")
        axes[1, i].set(xlabel="Time since stimulation onset (s)", ylabel="Field along profile (V/m)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("First enrolled context / first paired future: actual recorded waveforms (illustration, not inference)")
    fig.tight_layout(); montage._save_figure(fig, root, "figure_05_causal_waveform")


def _save_episode(root, episode, context, profile, future_index, measurements, paths, split, seeds, cfg):
    stem = f"{context['context_id']}_fut{future_index+1:02d}"
    if cfg.analysis.mapping_storage.save_original_unit_noise:
        file = root / "unit_noise" / f"{stem}.npz"
        if not file.exists():
            np.savez_compressed(file, original_unit_noise=paths, split_sample=split,
                                seeds=seeds, dt_ms=float(cfg.env.network.dt))
    if cfg.analysis.mapping_storage.save_processed_eeg:
        np.savez_compressed(root / "processed_EEG" / f"{stem}_{profile}.npz",
            **{f"{epoch}_{view}_v": m[view] for epoch, m in measurements.items()
               for view in ("neural", "observed")},
            fs_hz=measurements["baseline"]["fs_hz"],
            epoch_start_ms=np.asarray([m["start_ms"] for m in measurements.values()]),
            epoch_names=np.asarray(list(measurements)),
        )
    if future_index + 1 == int(cfg.analysis.mapping_storage.representative_future_index):
        outputs = episode["simulation"]["outputs_by_epoch"]["stimulation"]
        # Full resolution field, time, and dipole retained for waveform audits.
        payload = {"local_dipole_nA_um": montage._dipole_by_epoch(episode, "stimulation"),
                   "time_ms": np.concatenate([x["sample_times_ms"] for x in outputs]),
                   "neural_eeg_v": np.concatenate([np.asarray(x["eeg_v"]).reshape(-1) for x in outputs]),
                   "observed_eeg_v": np.asarray(episode["observed_raw_by_epoch"]["stimulation"]),
                   # Waveforms include an extra left endpoint; observations use
                   # (start, stop]. Drop each waveform's left endpoint only.
                   "field_v_per_m": np.concatenate([x["stimulation"]["field_v_per_m"][1:] for x in outputs]),
                   "dt_ms": float(cfg.env.network.dt)}
        for population in ("E", "I"):
            payload[f"{population}_spike_times_ms"] = np.concatenate([
                x["spikes"][population]["times_ms"] for x in outputs])
        if payload["time_ms"].shape != payload["field_v_per_m"].shape:
            raise RuntimeError("Field and observation sample axes do not align.")
        np.savez_compressed(root / "representative_traces" / f"{stem}_{profile}.npz", **payload)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    started = time.perf_counter()
    source = _load_source(cfg)
    contexts = _validate(cfg, source)
    if not cfg.analysis.smoke_test and size != int(source["provenance"]["mpi_ranks"]):
        raise ValueError("Full H5-O2 must retain the source MPI layout "
                         f"(-n {source['provenance']['mpi_ranks']}); rank-dependent network "
                         "seeding means changing rank count changes the realization. "
                         "Smokes may use fewer ranks.")
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / str(cfg.analysis.output_root_name)
    occupied = root.exists() and any(root.iterdir()) if rank == 0 else None
    if comm.bcast(occupied, root=0):
        raise FileExistsError(f"Refusing to overwrite {root}; use a new experiment.name.")
    if rank == 0:
        root.mkdir(parents=True)
        for folder in ("processed_EEG", "unit_noise", "representative_traces"):
            (root / folder).mkdir()
        OmegaConf.save(cfg, root / "resolved_config.yaml", resolve=True)
        write_json(root / "frozen_EEG_only_B_targets.json", source["target"])
        write_json(root / "frozen_rhythm_presence_rule.json", source["rule"])
        write_json(root / "prespecified_protocol.json", {
            "contexts": contexts, "source_hashes": source["hashes"],
            "actions": ACTIONS, "maximum_action_future_outcomes": len(contexts)*3*int(cfg.analysis.crossed_design.n_future_continuations),
            "efficacy_target_key": f"neural_audit/{PRIMARY}",
            "screening_target_key": f"observed/{PRIMARY}",
            "analytical_rule": "observed CSD angle <=30 degrees: z, otherwise 60deg; confidence failure: sham",
            "future_splits": [[1, 2], [3, 4]], "no_policy_training": True,
            "screening_uses_only_baseline_despite_full_sham_simulation": True,
        })
        model = spatial._forward_model(cfg)
    else:
        model = None
    screens, rows, spectra, update_tables, episode_audits = [], [], [], [], []
    completed = 0
    n_futures = int(cfg.analysis.crossed_design.n_future_continuations)
    for context_number, context in enumerate(contexts, 1):
        if rank == 0:
            print(f"H5-O2 context {context_number}/{len(contexts)}: {context['context_id']}", flush=True)
        condition_cfg = montage._with_orientation_state(cfg, context)
        kwargs = dict(condition_cfg=condition_cfg, context=context, phase_sensor_index=0,
                      root=root, comm=comm, size=size, rank=rank)
        baseline_episode = montage._run_profile(
            **kwargs, future_seed=online._future_seed(cfg, context, 0), future_index=0, profile=SHAM)
        completed += 1
        if rank == 0:
            paths, split, seeds = _unit_noise(cfg, context, 0)
            baseline = _epoch_measurement(baseline_episode, context, "baseline", paths, cfg, model)
            screening = _screen(baseline, source, cfg)
            enrolled = bool(screening["treatment_eligible"] or (
                cfg.analysis.smoke_test and cfg.analysis.smoke_force_eligible))
            screen_row = {**context, **screening, "enrolled": enrolled,
                          "smoke_forced": bool(enrolled and not screening["treatment_eligible"])}
            screens.append(screen_row)
            pd.DataFrame(screens).to_csv(root / "prospective_screening.csv", index=False)
            # Persist screening before any active outcomes; future labels cannot
            # rescue abstention. Keep baseline EEG even for rejected contexts.
            np.savez_compressed(root / "processed_EEG" / f"{context['context_id']}_screen.npz",
                                neural_v=baseline["neural"], observed_v=baseline["observed"], fs_hz=baseline["fs_hz"])
            if cfg.analysis.mapping_storage.save_original_unit_noise:
                np.savez_compressed(root / "unit_noise" / f"{context['context_id']}_fut01.npz",
                    original_unit_noise=paths, split_sample=split, seeds=seeds, dt_ms=float(cfg.env.network.dt))
        else:
            screening, enrolled = None, None
        screening, enrolled = comm.bcast((screening, enrolled), root=0)
        if not enrolled:
            if rank == 0:
                episode_audits.append({"context_id": context["context_id"], "status": "screen_rejected_sham_only"})
            continue
        condition_cfg = montage._with_action_frequency(condition_cfg, float(screening["EEG_selected_frequency_hz"]))
        kwargs["condition_cfg"] = condition_cfg
        for future in range(n_futures):
            future_seed = online._future_seed(cfg, context, future)
            sham = baseline_episode if future == 0 else montage._run_profile(
                **kwargs, future_seed=future_seed, future_index=future, profile=SHAM)
            if future:
                completed += 1
            if rank == 0:
                paths, split, seeds = _unit_noise(cfg, context, future)
                sham_measurements = {epoch: _epoch_measurement(
                    sham, context, epoch, paths, cfg, model, trim=epoch == "stimulation"
                ) for epoch in ("baseline", "stimulation", "washout")}
            for profile in ACTIONS:
                episode = sham if profile == SHAM else montage._run_profile(
                    **kwargs, future_seed=future_seed, future_index=future, profile=profile)
                if profile != SHAM:
                    completed += 1
                if rank != 0:
                    continue
                measurements = sham_measurements if profile == SHAM else {
                    epoch: _epoch_measurement(episode, context, epoch, paths, cfg, model, trim=epoch == "stimulation")
                    for epoch in ("baseline", "stimulation", "washout")}
                row, updates = _metric(episode, sham, context, screening,
                    measurements["stimulation"], sham_measurements["stimulation"],
                    measurements["washout"], sham_measurements["washout"],
                    measurements["baseline"], future, cfg, source)
                row["full_sensor_unit_noise_sha256"] = _hash_array(paths)
                row["sensor_noise_history_sha256"] = _hash_array(paths[:, :split])
                rows.append(row); update_tables.append(updates)
                for epoch, value in measurements.items():
                    spectra.append(value["spectra"].assign(
                        context_id=context["context_id"], structure_seed=context["structure_seed"],
                        hidden_frequency_hz=context["hidden_frequency_hz"], rotation_y_rad=context["rotation_y_rad"],
                        future_index=future+1, montage_profile=profile))
                _save_episode(root, episode, context, profile, future, measurements, paths, split, seeds, cfg)
                pd.DataFrame(rows).to_csv(root / "context_montage_future_metrics.csv", index=False)
                pd.concat(update_tables, ignore_index=True).to_csv(root / "causal_phase_updates.csv", index=False)
                write_json(root / "progress.json", {"completed_episodes": completed,
                    "completed_action_future_outcomes": len(rows), "runtime_seconds": time.perf_counter()-started})
    # No collectives after this barrier: a rank-zero plotting failure cannot
    # strand workers waiting at finalization. The wrapper aborts on any error.
    comm.Barrier()
    if rank != 0:
        return
    metrics = pd.DataFrame(rows)
    screening = pd.DataFrame(screens)
    wide, splits, structures, summary = _response_analysis(metrics, cfg)
    integrity, mechanism, opportunity = _checks(screening, metrics, wide, summary, cfg)
    if len(metrics):
        integrity["all_sensor_future_noise_paired"] = bool(metrics.groupby(
            ["context_id", "future_index"]).full_sensor_unit_noise_sha256.nunique().eq(1).all())
        integrity["all_sensor_history_noise_identical"] = bool(metrics.groupby(
            "context_id").sensor_noise_history_sha256.nunique().eq(1).all())
    spectra = pd.concat(spectra, ignore_index=True) if spectra else pd.DataFrame()
    for name, table in (("expected_response_map", wide), ("independent_future_split", splits),
                        ("structure_effects", structures), ("spectral_metrics", spectra),
                        ("screen_rejections", pd.DataFrame(episode_audits))):
        table.to_csv(root / f"{name}.csv", index=False)
    write_json(root / "response_opportunity_summary.json", summary)
    smoke = bool(cfg.analysis.smoke_test)
    analytical_ready, residual_ready = _readiness(integrity, mechanism, opportunity, smoke=smoke)
    result = {
        "smoke_test": smoke, "integrity_checks": integrity,
        "analytical_control_checks": mechanism, "residual_H5_opportunity_checks": opportunity,
        "analytical_EEG_control_supported": analytical_ready,
        "residual_opportunity_for_policy_development": residual_ready,
        "H5_status": "NOT ESTABLISHED", "ML_policy": "NOT TRAINED OR TESTED",
        "completed_episodes": completed, "eligible_contexts": int(screening.enrolled.sum()),
        "completed_action_future_outcomes": len(metrics),
        "status": "SMOKE COMPLETED" if smoke else "EXPLORATORY MAP COMPLETED",
    }
    if cfg.experiment.plot:
        _plots(root, screening, metrics, wide, splits, spectra, source, cfg)
    write_json(root / "protocol_and_provenance.json", {
        "source_hashes": source["hashes"], "upstream_provenance": source["provenance"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "hostname": platform.node(), "python": sys.version, "mpi_ranks": size,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed_union": sorted(set.union(*[{int(r[k]) for r in contexts} for k in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed")],
            {online._future_seed(cfg, r, f) for r in contexts for f in range(n_futures)},
            {s for r in contexts for f in range(n_futures) for sensor in range(3) for s in _noise_seeds(cfg, r, f, sensor)})),
        "claim_boundary": "Toy local-field neural-current phenotype; known single-source leadfield and equal independent sensor noise; no scalp-current solution, stimulation artifact, clinical efficacy, or H5 confirmation.",
    })
    result["runtime_seconds"] = time.perf_counter() - started
    write_json(root / "experiment_conclusion.json", result)
    # Hash every nested artifact, not just the summary. Marker is written last.
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root / "run_complete.json", {"completed": True,
        "runtime_seconds": time.perf_counter()-started, "files_sha256": manifest})
    for label, checks in (("Integrity", integrity), ("Analytical control", mechanism), ("Residual H5 opportunity", opportunity)):
        print(f"\nH5-O2 {label}")
        for key, value in checks.items():
            print(f"{key}: {'PASSED' if value else 'NOT PASSED'}")
    print(json.dumps(montage._json_ready(summary), indent=2))
    print(f"\nResults saved to: {root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if MPI.COMM_WORLD.Get_size() > 1:
            import traceback
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
