"""H5-O1M: zero-field spatial EEG measurement and paired-reference audit.

One local neural trajectory is projected to four head orientations offline.
For independent equal-spectrum sensor noise, the real alpha cross-spectral
matrix has expectation S = P g(theta) g(theta).T + N I. Its leading
eigenvector identifies the source pattern; lambda1 - mean(lambda2,lambda3)
estimates neural alpha energy. Both the orientation and noise level are
estimated from observed EEG, never from the simulated orientation/noise path.

This deliberately favorable, known-head/single-source measurement model is
tested without fitting a treatment policy or applying any stimulation.
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
from lfpykit.eegmegcalc import FourSphereVolumeConductor
from mpi4py import MPI
from omegaconf import OmegaConf, open_dict
from scipy import signal

sys.path.insert(1, config("MAIN_PATH"))

from experiments.ballnstick_analysis.run_ballnstick import _preprocess_eeg
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import _epoch_row
from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (
    _ar1_path,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_montage_orientation_opportunity import (
    SHAM,
    _dipole_by_epoch,
    _json_ready,
    _orientation_specs,
    _run_profile,
    _save_figure,
    _spike_hash,
    _with_orientation_state,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_multitaper_measurement_validation import (
    MT_POOLED,
    OBSERVED,
    _estimate_multitaper_methods,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_response_mapping import (
    _hash_locked_files,
)


PRIMARY = "equal_noise_csd_rank1"
AUTO = "auto_power_template"
METHODS = [AUTO, PRIMARY]
SCREEN_INPUT_FIELDS = (
    "geometry_normalized_log10_alpha",
    "spatial_accepted",
    "carrier_identified",
    "recent_phase_actionable",
)
SOURCE_FILES = {
    "conclusion": "experiment_conclusion.json",
    "audit": "H5_O1D_policy_development_audit.json",
    "screening": "prospective_screening.csv",
    "metrics": "context_montage_future_metrics.csv",
    "evaluation": "crossvalidated_policy_evaluation.csv",
    "target": "frozen_noisy_screening_and_neural_outcome_B_target.json",
    "provenance": "protocol_and_provenance.json",
}


def _write_json(path, value):
    path.write_text(json.dumps(_json_ready(value), indent=2, allow_nan=False))


def _load_source(cfg):
    root = Path(to_absolute_path(str(cfg.analysis.source_h5o1d.result_dir)))
    files, hashes = _hash_locked_files(
        root, SOURCE_FILES, cfg.analysis.source_h5o1d.expected_sha256
    )
    conclusion = json.loads(files["conclusion"].read_text())
    if conclusion["conclusions"]["H5_O1D_policy_development"] != "NOT PASSED":
        raise ValueError("This audit preserves the exact failed H5-O1D experiment.")
    seeds = set()
    for name in ("screening", "metrics"):
        frame = pd.read_csv(files[name])
        for column in (
            "structure_seed",
            "history_seed",
            "phase_seed",
            "trial_seed",
            "future_drive_seed",
        ):
            if column in frame:
                seeds.update(frame[column].dropna().astype(int))
    return {
        "root": str(root),
        "hashes": hashes,
        "seed_union": seeds,
        "provenance": json.loads(files["provenance"].read_text()),
    }


def _trajectory_specs(cfg):
    """B once per structure, paired A9/A11 histories; rotate only offline."""
    design = cfg.analysis.measurement_design
    seed = int(cfg.experiment.seed)
    zero_orientation = _orientation_specs(cfg)[0]
    rows = []
    for stage_index, stage in enumerate(("calibration", "evaluation")):
        count = int(design[f"{stage}_structures"])
        if cfg.analysis.smoke_test:
            count = 1
        for index in range(count):
            structure = seed + int(design[f"{stage}_structure_seed_offset"]) + index
            pair = stage_index * 100 + index
            conditions = (
                [("B", 9.0)]
                if stage == "calibration"
                else [("B", 9.0), ("A", 9.0), ("A", 11.0)]
            )
            for condition, frequency in conditions:
                rows.append(
                    {
                        **zero_orientation,
                        "trajectory_id": f"{stage}_s{index:02d}_{condition}_f{int(frequency):02d}",
                        "context_id": f"{stage}_s{index:02d}_{condition}_f{int(frequency):02d}",
                        "stage": stage,
                        "condition": condition,
                        "structure_index": index,
                        "structure_seed": structure,
                        "history_index": 0,
                        "history_seed": seed + int(design.history_seed_offset) + pair,
                        "phase_seed": seed + int(design.phase_seed_offset) + pair,
                        "trial_seed": seed + int(design.trial_seed_offset) + len(rows),
                        "future_drive_seed": seed
                        + int(design.future_seed_offset)
                        + pair,
                        "future_group_index": len(rows),
                        "hidden_frequency_hz": frequency,
                        "label": "low_diffusion",
                        "diffusion_rad2_per_s": 0.5,
                        "shared_drive_label": "full_shared_drive",
                        "shared_modulated_fraction": 1.0,
                    }
                )
    return rows


def _noise_seeds(cfg, trajectory_index, repeat, sensor):
    history = (
        int(cfg.experiment.seed)
        + int(cfg.analysis.measurement_design.sensor_noise_seed_offset)
        + 100 * trajectory_index
        + 10 * repeat
        + sensor
    )
    return history, history + 100_000


def _validate_design(cfg, source):
    block = cfg.analysis.spatial_measurement
    if str(cfg.analysis.simulator) != "online" or not np.isclose(
        cfg.analysis.inhibition_scale, 1.0
    ):
        raise ValueError("Keep the online model and recurrent inhibition unchanged.")
    if list(cfg.analysis.states.frequencies_hz) != [9.0, 11.0]:
        raise ValueError("The 9/11-Hz task is frozen.")
    if not np.isclose(cfg.analysis.states.modulation_depth, 0.04):
        raise ValueError("The A generator modulation depth must remain 0.04.")
    if len(cfg.analysis.states.phase_diffusion_levels) != 1 or not np.isclose(
        cfg.analysis.states.phase_diffusion_levels[0].diffusion_rad2_per_s, 0.5
    ):
        raise ValueError("The diffusion level must remain 0.5 rad2/s.")
    if len(cfg.analysis.states.shared_drive_levels) != 1 or not np.isclose(
        cfg.analysis.states.shared_drive_levels[0].shared_modulated_fraction, 1.0
    ):
        raise ValueError("The full shared rhythmic drive must remain fixed.")
    if not np.isclose(block.noise_rms_fraction, 0.25) or not np.isclose(
        block.ar1_coefficient, 0.95
    ):
        raise ValueError("The inherited moderate observation noise must remain fixed.")
    if str(block.primary_estimator) != PRIMARY or str(block.comparator) != AUTO:
        raise ValueError("There is no estimator selection in H5-O1M.")
    if float(cfg.env.simulation.obs_win_len) != 1000.0:
        raise ValueError("Use env.simulation.obs_win_len=1000.")
    if (
        not np.isclose(cfg.env.network.celsius, 6.3)
        or str(cfg.env.online.temperature_mode) != "configured"
    ):
        raise ValueError("Canonical HH requires the configured 6.3-C model.")
    if [round(np.degrees(x["rotation_y_rad"])) for x in _orientation_specs(cfg)] != [
        0,
        20,
        40,
        60,
    ]:
        raise ValueError("The orientation grid is frozen.")
    if not cfg.analysis.smoke_test:
        if [
            int(cfg.analysis.timeline[f"{x}_steps"])
            for x in ("burn_in", "baseline", "stimulation", "washout")
        ] != [1, 30, 9, 2]:
            raise ValueError("Full timing must remain 1/30/9/2 seconds, all sham.")
        if (
            int(cfg.analysis.measurement_design.calibration_structures),
            int(cfg.analysis.measurement_design.evaluation_structures),
            int(cfg.analysis.measurement_design.noise_repeats),
        ) != (3, 4, 3):
            raise ValueError(
                "Full design is three calibration and four evaluation structures, three noise repeats."
            )
    specs = _trajectory_specs(cfg)
    namespaces = [
        {int(row[col]) for row in specs}
        for col in (
            "structure_seed",
            "history_seed",
            "phase_seed",
            "trial_seed",
            "future_drive_seed",
        )
    ]
    namespaces.append(
        {
            value
            for i in range(len(specs))
            for repeat in range(int(cfg.analysis.measurement_design.noise_repeats))
            for sensor in range(3)
            for value in _noise_seeds(cfg, i, repeat, sensor)
        }
    )
    if any(a & b for a, b in itertools.combinations(namespaces, 2)):
        raise ValueError("Seed namespaces overlap.")
    if set.union(*namespaces) & source["seed_union"]:
        raise ValueError("Measurement seeds overlap H5-O1D.")
    if max(namespaces[0]) * 10_000 + 255 >= np.iinfo(np.uint32).max:
        raise ValueError("Structure seed exceeds simulator uint32 namespace.")
    return specs


def _forward_model(cfg):
    head = FourSphereVolumeConductor(
        np.asarray(cfg.analysis.eeg_array.locations_um, float),
        cfg.env.eeg.foursphereheadmodel.radii,
        cfg.env.eeg.foursphereheadmodel.sigmas,
    )
    lead = (
        np.asarray(
            head.get_dipole_potential(
                np.eye(3), np.asarray(cfg.env.network.position, float)
            )
        )
        * 1e-3
    )
    block = cfg.analysis.spatial_measurement
    angles = np.arange(
        float(block.orientation_grid_min_deg),
        float(block.orientation_grid_max_deg)
        + 0.5 * float(block.orientation_grid_step_deg),
        float(block.orientation_grid_step_deg),
    )
    vectors = np.column_stack(
        [np.sin(np.radians(angles)), np.zeros(len(angles)), np.cos(np.radians(angles))]
    )
    gain = vectors @ lead.T
    energy = np.sum(gain**2, axis=1)
    reference_gain = float(np.sum(lead[:, 2] ** 2))
    return {
        "leadfield_v_per_nA_um": lead,
        "angles": angles,
        "patterns": gain / np.sqrt(energy[:, None]),
        "power_patterns": gain**2 / energy[:, None],
        "log10_gain_relative_to_zero": np.log10(energy / reference_gain),
    }


def _project(local_dipole, angle_deg, model):
    angle = np.radians(angle_deg)
    rotation = np.asarray(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return model["leadfield_v_per_nA_um"] @ rotation @ local_dipole


def _preprocess_channels(eeg, cfg):
    processed = [
        _preprocess_eeg(
            x,
            fs_hz=1000.0 / float(cfg.env.network.dt),
            target_fs_hz=int(cfg.analysis.target_fs_hz),
            low_hz=float(cfg.analysis.low_hz),
            high_hz=float(cfg.analysis.high_hz),
        )
        for x in eeg
    ]
    return np.stack([x[0] for x in processed]), processed[0][1]


def _cross_spectrum(processed, fs, cfg):
    block = cfg.analysis.spatial_measurement
    length = min(processed.shape[1], int(round(float(block.segment_seconds) * fs)))
    if length < 16:
        raise ValueError("Too few samples for the spatial spectrum.")
    spectrum = None
    for i in range(3):
        for j in range(i, 3):
            frequency, value = signal.csd(
                processed[i],
                processed[j],
                fs=fs,
                window="hann",
                nperseg=length,
                noverlap=int(length * float(block.overlap_fraction)),
                detrend="constant",
                scaling="density",
            )
            if spectrum is None:
                spectrum = np.zeros((3, 3, len(frequency)), complex)
            spectrum[i, j] = value
            spectrum[j, i] = np.conjugate(value)
    mask = (frequency >= float(block.alpha_low_hz)) & (
        frequency <= float(block.alpha_high_hz)
    )
    if mask.sum() < 2:
        raise ValueError("Alpha spectrum needs at least two bins.")
    covariance = np.trapz(spectrum[:, :, mask], frequency[mask], axis=-1).real
    return frequency, spectrum, 0.5 * (covariance + covariance.T)


def _estimate_spatial(covariance, model, cfg):
    """Pure measurement function: no true orientation, condition, or noise path."""
    c = np.asarray(covariance, float)
    if c.shape != (3, 3) or not np.isfinite(c).all():
        raise ValueError("Expected a finite three-channel covariance.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (c + c.T))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    noise = float(np.mean(eigenvalues[:2]))
    energy = max(float(eigenvalues[-1]) - noise, 0.0)
    total = max(float(np.trace(c)), np.finfo(float).tiny)
    vector = eigenvectors[:, -1]
    scores = np.abs(model["patterns"] @ vector)
    csd_index = int(np.argmax(scores))
    fraction = energy / max(float(eigenvalues[-1]), np.finfo(float).tiny)
    residual = float(np.sqrt(max(0.0, 1.0 - min(1.0, scores[csd_index]) ** 2)))
    auto_pattern = np.maximum(np.diag(c), 0.0) / total
    auto_index = int(
        np.argmin(np.sum((model["power_patterns"] - auto_pattern) ** 2, axis=1))
    )
    output = {}
    for method, index, alpha in (
        (PRIMARY, csd_index, energy),
        (AUTO, auto_index, total),
    ):
        angle = float(model["angles"][index])
        accepted = (
            bool(
                energy > 0
                and fraction
                >= float(cfg.analysis.spatial_measurement.minimum_signal_fraction)
                and residual
                <= float(cfg.analysis.spatial_measurement.maximum_pattern_residual)
            )
            if method == PRIMARY
            else True
        )
        output[method] = {
            "estimated_orientation_deg": angle,
            "predicted_profile": (
                "montage_profile_z"
                if angle <= float(cfg.analysis.spatial_measurement.action_boundary_deg)
                else "montage_profile_60deg"
            ),
            "spatial_accepted": accepted,
            "global_alpha_estimate_v2": float(alpha),
            "geometry_normalized_log10_alpha": float(
                np.log10(max(alpha, np.finfo(float).tiny))
                - model["log10_gain_relative_to_zero"][index]
            ),
            "estimated_noise_alpha_per_sensor_v2": noise if method == PRIMARY else 0.0,
            "signal_fraction": fraction,
            "pattern_residual": residual,
        }
    return output


def _carrier_and_phase(processed_vertex, fs, cfg):
    # Hidden frequency is a mandatory scoring argument of the inherited
    # estimator. A constant NaN prevents truth from being available here.
    rows, _, _ = _estimate_multitaper_methods(
        processed_vertex,
        fs_hz=fs,
        hidden_frequency_hz=float("nan"),
        input_signal=OBSERVED,
        cfg=cfg,
    )
    estimate = next(x for x in rows if x["estimator"] == MT_POOLED)
    frequency = float(estimate["selected_frequency_hz"])
    tail = processed_vertex[-int(round(fs)) :]
    time_s = np.arange(tail.size) / fs
    quality = (
        2
        * abs(np.mean(tail * np.exp(-2j * np.pi * frequency * time_s)))
        / max(float(np.sqrt(np.mean(tail**2))), np.finfo(float).tiny)
    )
    return {
        "EEG_selected_frequency_hz": frequency,
        "carrier_identified": bool(estimate["identified"]),
        "recent_resultant_to_rms": float(quality),
        "recent_phase_actionable": bool(quality >= 0.03),
    }


def _calibrate_targets(table):
    targets = {}
    for (view, method), group in table.groupby(["signal_view", "estimator"]):
        # Noise views and rotations are repeats; each B structure gets one vote.
        mean_by_structure = group.groupby(
            "structure_seed"
        ).geometry_normalized_log10_alpha.mean()
        outcome_by_structure = group.groupby(
            "structure_seed"
        ).outcome_geometry_normalized_log10_alpha.mean()
        targets[f"{view}/{method}"] = {
            "baseline_mean_log10": float(mean_by_structure.mean()),
            "baseline_structure_sd_log10": float(mean_by_structure.std(ddof=1)),
            "outcome_mean_log10": float(outcome_by_structure.mean()),
            "outcome_structure_sd_log10": float(outcome_by_structure.std(ddof=1)),
            "reference_structures": sorted(int(x) for x in mean_by_structure.index),
            "threshold_uses_no_true_orientation": True,
        }
    return targets


def _screen_measurement(measurement, target, cfg):
    # This deliberately has no context, latent label, condition or anatomy argument.
    excess = float(measurement["geometry_normalized_log10_alpha"]) - float(
        target["baseline_mean_log10"]
    )
    phenotype = excess >= float(
        cfg.analysis.spatial_measurement.phenotype_minimum_excess_log10
    )
    eligible = bool(
        phenotype
        and measurement["spatial_accepted"]
        and measurement["carrier_identified"]
        and measurement["recent_phase_actionable"]
    )
    return {
        "alpha_excess_over_B_log10": excess,
        "phenotype_positive": bool(phenotype),
        "treatment_eligible": eligible,
        "fallback_action": "not_applied_measurement_only" if eligible else "sham",
    }


def _analyze_trajectory(episode, context, cfg, model, root):
    """Project one persistent trajectory, sharing identical noise across angles."""
    baseline = _dipole_by_epoch(episode, "baseline")
    outcome = _dipole_by_epoch(episode, "stimulation")
    trim = int(
        round(
            float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
            / float(cfg.env.network.dt)
        )
    )
    if trim:
        outcome = outcome[:, trim:-trim]
    baseline_start = int(cfg.analysis.timeline.burn_in_steps) * int(
        round(1000 / float(cfg.env.network.dt))
    )
    split = baseline_start + baseline.shape[1]
    total = int(
        round(
            sum(
                int(cfg.analysis.timeline[f"{x}_steps"])
                for x in ("burn_in", "baseline", "stimulation", "washout")
            )
            * 1000
            / float(cfg.env.network.dt)
        )
    )
    if cfg.analysis.spatial_measurement.save_canonical_dipole:
        np.savez_compressed(
            root / "canonical_dipoles" / f"{context['trajectory_id']}.npz",
            baseline_nA_um=baseline,
            outcome_nA_um=outcome,
            dt_ms=float(cfg.env.network.dt),
            baseline_start_ms=float(cfg.analysis.timeline.burn_in_steps) * 1000,
            outcome_start_ms=float(
                cfg.analysis.timeline.burn_in_steps
                + cfg.analysis.timeline.baseline_steps
            )
            * 1000
            + float(cfg.analysis.timeline.stimulation_analysis_trim_ms),
        )
    rows, spectra = [], []
    neural_views = {}
    for orientation in _orientation_specs(cfg):
        angle = float(np.degrees(orientation["rotation_y_rad"]))
        raw = _project(baseline, angle, model)
        future_raw = _project(outcome, angle, model)
        processed, fs = _preprocess_channels(raw, cfg)
        future_processed, _ = _preprocess_channels(future_raw, cfg)
        neural_views[angle] = (raw, processed, future_processed, fs)
    for repeat in range(int(cfg.analysis.measurement_design.noise_repeats)):
        noise_raw, noise_future, hashes, seeds, unit_paths = [], [], [], [], []
        for sensor in range(3):
            history_seed, future_seed = _noise_seeds(
                cfg, int(context["future_group_index"]), repeat, sensor
            )
            path = _ar1_path(
                n_samples=total,
                split_sample=split,
                history_seed=history_seed,
                future_seed=future_seed,
                coefficient=float(cfg.analysis.spatial_measurement.ar1_coefficient),
            )
            hashes.append(hashlib.sha256(path.astype("<f8").tobytes()).hexdigest())
            seeds.append([history_seed, future_seed])
            unit_paths.append(path)
            noise_raw.append(path[baseline_start:split])
            noise_future.append(path[split + trim : split + trim + outcome.shape[1]])
        noise_raw = np.asarray(noise_raw)
        if cfg.analysis.spatial_measurement.save_unit_noise:
            np.savez_compressed(
                root / "unit_noise" / f"{context['trajectory_id']}_n{repeat+1}.npz",
                original_unit_noise=np.asarray(unit_paths),
                seeds=np.asarray(seeds),
                split_sample=split,
                dt_ms=float(cfg.env.network.dt),
                ar1_coefficient=float(cfg.analysis.spatial_measurement.ar1_coefficient),
            )
        # Filtering/resampling is linear. Process the shared unit paths once;
        # baseline and outcome are handled separately to exclude future data.
        noise_processed, _ = _preprocess_channels(noise_raw, cfg)
        future_noise_processed, _ = _preprocess_channels(np.asarray(noise_future), cfg)
        for angle, (raw, neural, future_neural, fs) in neural_views.items():
            scale = float(cfg.analysis.spatial_measurement.noise_rms_fraction) * float(
                np.sqrt(np.mean(raw[0] ** 2))
            )
            observed = neural + scale * noise_processed
            future_observed = future_neural + scale * future_noise_processed
            common = {
                **context,
                "true_orientation_deg": angle,
                "noise_repeat": repeat + 1,
                "sensor_noise_seeds": json.dumps(seeds),
                "unit_noise_sha256": ";".join(hashes),
                "absolute_noise_scale_v": scale,
                "achieved_vertex_noise_RMS_fraction": float(
                    np.sqrt(np.mean((scale * noise_raw[0]) ** 2))
                    / np.sqrt(np.mean(raw[0] ** 2))
                ),
                "true_matched_profile": (
                    "montage_profile_z" if angle <= 30 else "montage_profile_60deg"
                ),
            }
            if cfg.analysis.spatial_measurement.save_processed_views:
                np.savez_compressed(
                    root
                    / "processed_EEG"
                    / f"{context['trajectory_id']}_o{angle:02.0f}_n{repeat+1}.npz",
                    baseline_neural_v=neural,
                    baseline_observed_v=observed,
                    outcome_neural_v=future_neural,
                    outcome_observed_v=future_observed,
                    fs_hz=fs,
                    unit_noise_history_seeds=np.asarray(seeds)[:, 0],
                    unit_noise_future_seeds=np.asarray(seeds)[:, 1],
                )
            for view, values, future_values in (
                ("observed", observed, future_observed),
                ("neural_audit", neural, future_neural),
            ):
                frequency, csd, covariance = _cross_spectrum(values, fs, cfg)
                estimates = _estimate_spatial(covariance, model, cfg)
                _, _, future_covariance = _cross_spectrum(future_values, fs, cfg)
                future_estimates = _estimate_spatial(future_covariance, model, cfg)
                carrier = _carrier_and_phase(values[0], fs, cfg)
                for method, estimate in estimates.items():
                    rows.append(
                        {
                            **common,
                            "signal_view": view,
                            "estimator": method,
                            **estimate,
                            **carrier,
                            "outcome_geometry_normalized_log10_alpha": future_estimates[
                                method
                            ]["geometry_normalized_log10_alpha"],
                            "absolute_angle_error_deg": abs(
                                estimate["estimated_orientation_deg"] - angle
                            ),
                            "profile_correct": estimate["predicted_profile"]
                            == common["true_matched_profile"],
                            "carrier_correct": bool(
                                np.isclose(
                                    carrier["EEG_selected_frequency_hz"],
                                    context["hidden_frequency_hz"],
                                )
                            ),
                            "neural_local_trajectory_shared_across_rotations": True,
                            "all_measurements_use_only_preceding_EEG": True,
                        }
                    )
                keep = (frequency >= 1) & (frequency <= 30)
                if repeat == 0:
                    for sensor in range(3):
                        spectra.append(
                            pd.DataFrame(
                                {
                                    "frequency_hz": frequency[keep],
                                    "PSD_v2_per_hz": csd[sensor, sensor, keep].real,
                                    "trajectory_id": context["trajectory_id"],
                                    "stage": context["stage"],
                                    "structure_seed": context["structure_seed"],
                                    "condition": context["condition"],
                                    "hidden_frequency_hz": context[
                                        "hidden_frequency_hz"
                                    ],
                                    "true_orientation_deg": angle,
                                    "signal_view": view,
                                    "sensor_index": sensor,
                                }
                            )
                        )
    return pd.DataFrame(rows), pd.concat(spectra, ignore_index=True)


def _balanced_accuracy(group):
    if group.true_matched_profile.nunique() != 2:
        return float("nan")
    return float(group.groupby("true_matched_profile").profile_correct.mean().mean())


def _summarize(evaluation, calibration, targets, trajectories, cfg):
    rows = []
    for (view, method, structure), group in evaluation.groupby(
        ["signal_view", "estimator", "structure_seed"]
    ):
        a, b = group[group.condition.eq("A")], group[group.condition.eq("B")]
        accepted = a[a.spatial_accepted]
        # Repeats and carrier/orientation cells stay within their structure.
        agreement = (
            a.groupby(["hidden_frequency_hz", "true_orientation_deg"])
            .predicted_profile.apply(lambda x: x.value_counts().max() / len(x))
            .mean()
        )
        rows.append(
            {
                "signal_view": view,
                "estimator": method,
                "structure_seed": int(structure),
                "balanced_accuracy": _balanced_accuracy(a),
                "accepted_balanced_accuracy": _balanced_accuracy(accepted),
                "spatial_coverage": float(a.spatial_accepted.mean()),
                "mean_absolute_angle_error_deg": float(
                    a.absolute_angle_error_deg.mean()
                ),
                "carrier_coverage": float(a.carrier_identified.mean()),
                "accepted_carrier_accuracy": float(
                    a.loc[a.carrier_identified, "carrier_correct"].mean()
                ),
                "recent_phase_actionable_fraction": float(
                    a.recent_phase_actionable.mean()
                ),
                "A_screen_sensitivity": float(a.phenotype_positive.mean()),
                "B_screen_specificity": float(1 - b.phenotype_positive.mean()),
                "screen_balanced_accuracy": float(
                    0.5
                    * (a.phenotype_positive.mean() + 1 - b.phenotype_positive.mean())
                ),
                "A_treatment_eligible_fraction": float(a.treatment_eligible.mean()),
                "noise_repeat_disagreement_fraction": float(1 - agreement),
            }
        )
    structures = pd.DataFrame(rows)
    columns = [
        x for x in structures.select_dtypes("number").columns if x != "structure_seed"
    ]
    # Do not silently drop a structure whose accepted subset contains only
    # one class. Undefined accepted accuracy must fail that gate, rather than
    # averaging only the easier structures that retained both classes.
    summary = (
        structures.groupby(["signal_view", "estimator"])[columns]
        .agg(lambda values: values.mean(skipna=False))
        .reset_index()
    )
    primary = summary[
        summary.signal_view.eq("observed") & summary.estimator.eq(PRIMARY)
    ].iloc[0]
    auto = summary[
        summary.signal_view.eq("observed") & summary.estimator.eq(AUTO)
    ].iloc[0]
    paired = structures[structures.signal_view.eq("observed")].pivot(
        index="structure_seed", columns="estimator", values="balanced_accuracy"
    )
    differences = (paired[PRIMARY] - paired[AUTO]).to_numpy()
    rng = np.random.default_rng(int(cfg.analysis.spatial_measurement.inference_seed))
    draws = rng.choice(
        differences,
        (int(cfg.analysis.spatial_measurement.bootstrap_repetitions), len(differences)),
        replace=True,
    ).mean(axis=1)
    null = np.asarray(
        [
            np.mean(differences * np.asarray(signs))
            for signs in itertools.product([-1, 1], repeat=len(differences))
        ]
    )
    inference = {
        "unit": "independent circuit structure",
        "n_structures": len(differences),
        "primary_CSD_minus_auto_balanced_accuracy": float(differences.mean()),
        "structure_bootstrap_interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "exact_sign_flip_one_sided_p": float(
            np.mean(null >= differences.mean() - 1e-12)
        ),
        "scope": "Exploratory measurement audit; four structures cannot support a one-sided exact p below 0.0625. No significance gate or policy confirmation.",
        "noise_views_rotations_carriers_are_repeats": True,
    }
    ideal_b = calibration[
        calibration.signal_view.eq("neural_audit") & calibration.estimator.eq(PRIMARY)
    ]
    spread = (
        ideal_b.groupby("structure_seed")
        .outcome_geometry_normalized_log10_alpha.agg(lambda x: x.max() - x.min())
        .max()
    )
    criteria = cfg.analysis.measurement_criteria
    paired_rates = trajectories[trajectories.stage.eq("evaluation")].copy()
    reference_rates = paired_rates[paired_rates.condition.eq("B")].set_index(
        "structure_seed"
    )
    active_rates = paired_rates[paired_rates.condition.eq("A")]
    maximum_rate_difference = max(
        float(
            np.max(
                np.abs(
                    active_rates[column].to_numpy()
                    - active_rates.structure_seed.map(
                        reference_rates[column]
                    ).to_numpy()
                )
            )
        )
        for column in ("E_firing_rate_hz", "I_firing_rate_hz")
    )
    checks = {
        "source_H5O1D_negative_result_hash_locked": True,
        "all_simulations_stimulation_free": bool(
            (trajectories.applied_amplitude_v_per_m == 0).all()
        ),
        "one_neural_trajectory_reused_exactly_across_orientations": bool(
            evaluation.neural_local_trajectory_shared_across_rotations.all()
        ),
        "B_future_shared_across_reference_orientations": bool(
            spread <= float(criteria.maximum_normalized_reference_spread_log10)
        ),
        "calibration_precedes_disjoint_evaluation": bool(
            set(calibration.structure_seed).isdisjoint(evaluation.structure_seed)
        ),
        "screening_and_estimation_do_not_receive_hidden_orientation": True,
        "all_observation_processing_precedes_decision": bool(
            evaluation.all_measurements_use_only_preceding_EEG.all()
        ),
        "noise_scale_frozen_from_predecision_vertex": bool(
            np.allclose(evaluation.achieved_vertex_noise_RMS_fraction, 0.25, atol=0.005)
        ),
        "complete_evaluation_grid": bool(
            len(evaluation)
            == int(trajectories.stage.eq("evaluation").sum())
            * 4
            * int(cfg.analysis.measurement_design.noise_repeats)
            * 2
            * 2
        ),
        "minimum_independent_evaluation_structures": bool(
            evaluation.structure_seed.nunique() >= 4
        ),
        "measurements_finite": bool(
            np.isfinite(
                evaluation[
                    [
                        "estimated_orientation_deg",
                        "geometry_normalized_log10_alpha",
                        "outcome_geometry_normalized_log10_alpha",
                        "recent_resultant_to_rms",
                    ]
                ].to_numpy()
            ).all()
        ),
        "carrier_coverage": bool(
            primary.carrier_coverage >= criteria.minimum_carrier_coverage
        ),
        "accepted_carrier_accuracy": bool(
            primary.accepted_carrier_accuracy
            >= criteria.minimum_accepted_carrier_accuracy
        ),
        "recent_phase_actionable": bool(
            primary.recent_phase_actionable_fraction
            >= criteria.minimum_recent_phase_actionable_fraction
        ),
        "spatial_accuracy": bool(
            primary.balanced_accuracy >= criteria.minimum_spatial_balanced_accuracy
        ),
        "spatial_angle_error": bool(
            primary.mean_absolute_angle_error_deg
            <= criteria.maximum_mean_absolute_angle_error_deg
        ),
        "spatial_coverage": bool(
            primary.spatial_coverage >= criteria.minimum_spatial_accepted_fraction
        ),
        "accepted_spatial_accuracy": bool(
            primary.accepted_balanced_accuracy
            >= criteria.minimum_accepted_spatial_balanced_accuracy
        ),
        "CSD_improves_on_auto_power": bool(
            primary.balanced_accuracy - auto.balanced_accuracy
            > criteria.minimum_accuracy_advantage_over_auto
        ),
        "screen_sensitivity": bool(
            primary.A_screen_sensitivity >= criteria.minimum_A_screen_sensitivity
        ),
        "screen_specificity": bool(
            primary.B_screen_specificity >= criteria.minimum_B_screen_specificity
        ),
        "screen_balanced_accuracy": bool(
            primary.screen_balanced_accuracy
            >= criteria.minimum_screen_balanced_accuracy
        ),
        "noise_repeat_stability": bool(
            primary.noise_repeat_disagreement_fraction
            <= criteria.maximum_noise_repeat_class_disagreement
        ),
        "neural_rates_within_guardrails": bool(
            (trajectories.E_firing_rate_hz > 0).all()
            and (trajectories.E_firing_rate_hz <= criteria.maximum_rate_E_hz).all()
            and (trajectories.I_firing_rate_hz > 0).all()
            and (trajectories.I_firing_rate_hz <= criteria.maximum_rate_I_hz).all()
        ),
        "exact_zero_field_at_episode_end": bool(
            (trajectories.final_extracellular_residual_mV == 0).all()
        ),
        "A_B_firing_rates_matched": bool(
            maximum_rate_difference <= criteria.maximum_paired_rate_difference_hz
        ),
    }
    inference["maximum_geometry_normalized_B_spread_log10"] = float(spread)
    inference["maximum_paired_A_B_rate_difference_hz"] = maximum_rate_difference
    return structures, summary, inference, checks


def _plots(root, evaluation, calibration, spectra, structures, cfg):
    colors = {AUTO: "#e45756", PRIMARY: "#4c78a8"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, view in zip(axes, ("neural_audit", "observed")):
        selected = spectra[
            spectra.stage.eq("evaluation")
            & spectra.signal_view.eq(view)
            & spectra.true_orientation_deg.eq(0)
            & spectra.sensor_index.eq(0)
        ]
        for condition, freq, label in (
            ("B", 9, "B homogeneous"),
            ("A", 9, "A 9 Hz"),
            ("A", 11, "A 11 Hz"),
        ):
            g = (
                selected[
                    selected.condition.eq(condition)
                    & selected.hidden_frequency_hz.eq(freq)
                ]
                .groupby("frequency_hz")
                .PSD_v2_per_hz.mean()
            )
            ax.plot(
                g.index, 10 * np.log10(g.clip(lower=np.finfo(float).tiny)), label=label
            )
        ax.set(
            xlim=(5, 15),
            xlabel="Frequency (Hz)",
            ylabel="PSD (dB V²/Hz)",
            title=view.replace("_", " "),
        )
        ax.legend(fontsize=8)
    _save_figure(fig, root, "figure_01_A_B_PSD")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        g = evaluation[
            evaluation.signal_view.eq("observed")
            & evaluation.estimator.eq(method)
            & evaluation.condition.eq("A")
        ]
        for structure, part in g.groupby("structure_seed"):
            ax.scatter(
                part.true_orientation_deg,
                part.estimated_orientation_deg,
                s=12,
                alpha=0.5,
                label=str(structure),
            )
        ax.plot([0, 60], [0, 60], "k--", lw=1)
        ax.set(
            xlabel="True angle (audit only, deg)",
            ylabel="EEG-estimated angle (deg)",
            title=method,
        )
    _save_figure(fig, root, "figure_02_spatial_estimation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    a = evaluation[evaluation.signal_view.eq("observed") & evaluation.condition.eq("A")]
    for method in METHODS:
        # First average repeats within each independent structure.
        g = (
            a[a.estimator.eq(method)]
            .groupby(["structure_seed", "true_orientation_deg"])
            .absolute_angle_error_deg.mean()
            .groupby("true_orientation_deg")
            .agg(["mean", "std"])
        )
        axes[0].errorbar(
            g.index,
            g["mean"],
            yerr=g["std"].fillna(0),
            marker="o",
            label=method,
            color=colors[method],
        )
        g = structures[
            structures.signal_view.eq("observed") & structures.estimator.eq(method)
        ]
        axes[1].plot(
            np.arange(len(g)),
            g.balanced_accuracy,
            "o-",
            label=method,
            color=colors[method],
        )
    axes[0].set(
        xlabel="Orientation (deg)",
        ylabel="Absolute error (deg)",
        title="Mean ± SD across structures",
    )
    axes[1].axhline(0.8, color="k", ls="--")
    axes[1].set(xlabel="Structure index", ylabel="Balanced accuracy", ylim=(0, 1.05))
    axes[1].legend(fontsize=8)
    _save_figure(fig, root, "figure_03_structure_generalization")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, method in zip(axes, METHODS):
        g = evaluation[
            evaluation.signal_view.eq("observed") & evaluation.estimator.eq(method)
        ]
        for x, condition in enumerate(("B", "A")):
            values = (
                g[g.condition.eq(condition)]
                .groupby("structure_seed")
                .alpha_excess_over_B_log10.mean()
            )
            ax.scatter(np.full(len(values), x), values, label=condition)
        ax.axhline(
            float(cfg.analysis.spatial_measurement.phenotype_minimum_excess_log10),
            color="k",
            ls="--",
        )
        ax.set(
            xticks=[0, 1],
            xticklabels=["B reference state", "A rhythmic state"],
            ylabel="Estimated alpha excess (log10)",
            title=method,
        )
    _save_figure(fig, root, "figure_04_EEG_only_screening")
    fig, ax = plt.subplots(figsize=(7, 4))
    g = calibration[
        calibration.signal_view.eq("neural_audit") & calibration.estimator.eq(PRIMARY)
    ]
    for structure, part in g.groupby("structure_seed"):
        means = part.groupby(
            "true_orientation_deg"
        ).outcome_geometry_normalized_log10_alpha.mean()
        ax.plot(means.index, means, "o-", label=str(structure))
    ax.set(
        xlabel="Orientation (deg)",
        ylabel="Gain-normalized B outcome (log10)",
        title="Identical B futures across coordinate rotations",
    )
    ax.legend(fontsize=8)
    _save_figure(fig, root, "figure_05_paired_B_calibration")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        g = structures[
            structures.signal_view.eq("observed") & structures.estimator.eq(method)
        ]
        ax.scatter(
            g.A_screen_sensitivity,
            g.B_screen_specificity,
            label=method,
            color=colors[method],
        )
    ax.set(
        xlabel="A phenotype sensitivity",
        ylabel="B phenotype specificity",
        xlim=(0, 1.05),
        ylim=(0, 1.05),
        title="Each point is an independent structure",
    )
    ax.legend(fontsize=8)
    _save_figure(fig, root, "figure_06_screening_generalization")
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for column, angle in enumerate((0, 60)):
        for sensor in range(3):
            ax = axes[sensor, column]
            selected = spectra[
                spectra.stage.eq("evaluation")
                & np.isclose(spectra.true_orientation_deg, angle)
                & spectra.sensor_index.eq(sensor)
            ]
            for condition in ("B", "A"):
                for view, style in (("neural_audit", "-"), ("observed", "--")):
                    g = (
                        selected[
                            selected.condition.eq(condition)
                            & selected.hidden_frequency_hz.eq(9)
                            & selected.signal_view.eq(view)
                        ]
                        .groupby("frequency_hz")
                        .PSD_v2_per_hz.mean()
                    )
                    ax.plot(
                        g.index,
                        10 * np.log10(g.clip(lower=np.finfo(float).tiny)),
                        style,
                        color="#4c78a8" if condition == "A" else "#e45756",
                        label=f"{condition} {view}",
                    )
            ax.set(
                xlim=(5, 15),
                ylabel="PSD (dB V²/Hz)",
                title=f"Sensor {sensor}, {angle}°",
            )
            if sensor == 2:
                ax.set_xlabel("Frequency (Hz)")
    axes[0, 0].legend(fontsize=7)
    _save_figure(fig, root, "figure_07_side_sensor_noise_PSD")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    source = _load_source(cfg)
    specs = _validate_design(cfg, source)
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / str(
        cfg.analysis.output_root_name
    )
    exists = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if comm.bcast(exists, root=0):
        raise FileExistsError(f"Refusing to overwrite {root}")
    started = time.perf_counter()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        (root / "canonical_dipoles").mkdir()
        (root / "processed_EEG").mkdir()
        (root / "unit_noise").mkdir()
        OmegaConf.save(cfg, root / "resolved_config.yaml", resolve=True)
        model = _forward_model(cfg)
        _write_json(
            root / "frozen_measurement_protocol.json",
            {
                "source_hashes": source["hashes"],
                "measurement": OmegaConf.to_container(
                    cfg.analysis.spatial_measurement, resolve=True
                ),
                "design": specs,
                "forward_model": {k: v.tolist() for k, v in model.items()},
                "known_priors": "FourSphere conductivities, sensor positions, fixed source location, one oriented population, equal independent sensor-noise spectra; no true orientation supplied",
                "all_actions": "zero field only",
                "no_measurement_or_policy_selection": True,
            },
        )
        print(
            f"H5-O1M: {len(specs)} zero-field trajectories; {size} MPI ranks",
            flush=True,
        )
    else:
        model = None
    tables = []
    spectra = []
    trajectory_rows = []
    targets = None
    for i, context in enumerate(specs):
        if context["stage"] == "evaluation" and targets is None:
            if rank == 0:
                calibration = pd.concat(tables, ignore_index=True)
                targets = _calibrate_targets(calibration)
                _write_json(root / "frozen_EEG_only_B_targets.json", targets)
            targets = comm.bcast(targets, root=0)
        if rank == 0:
            print(
                f"[{i+1}/{len(specs)}] {context['trajectory_id']} structure={context['structure_seed']}",
                flush=True,
            )
        run_cfg = _with_orientation_state(
            cfg, context, homogeneous_B=context["condition"] == "B"
        )
        with open_dict(run_cfg):
            # Noise is applied offline with a common path across rotations.
            run_cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg = 0.0
            run_cfg.env.online.record_representative_state = False
        episode = _run_profile(
            condition_cfg=run_cfg,
            context=context,
            future_seed=context["future_drive_seed"],
            future_index=0,
            profile=SHAM,
            phase_sensor_index=0,
            root=root / "zero_field_episodes",
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            table, psd = _analyze_trajectory(episode, context, cfg, model, root)
            if targets is not None:
                screens = [
                    _screen_measurement(
                        {key: row[key] for key in SCREEN_INPUT_FIELDS},
                        targets[f"{row.signal_view}/{row.estimator}"],
                        cfg,
                    )
                    for _, row in table.iterrows()
                ]
                table = pd.concat(
                    [table, pd.DataFrame(screens, index=table.index)], axis=1
                )
            tables.append(table)
            spectra.append(psd)
            rate = _epoch_row(episode, "baseline")
            trajectory_rows.append(
                {
                    **context,
                    "E_firing_rate_hz": float(rate.E_firing_rate_hz),
                    "I_firing_rate_hz": float(rate.I_firing_rate_hz),
                    "baseline_spike_sha256": _spike_hash(episode, "baseline"),
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
    # Workers finish every NEURON collective before root-only analysis.
    comm.Barrier()
    if rank != 0:
        return
    complete = pd.concat(tables, ignore_index=True)
    spectra = pd.concat(spectra, ignore_index=True)
    calibration = complete[complete.stage.eq("calibration")].copy()
    evaluation = complete[complete.stage.eq("evaluation")].copy()
    for column in ("phenotype_positive", "treatment_eligible"):
        evaluation[column] = evaluation[column].astype(bool)
    trajectories = pd.DataFrame(trajectory_rows)
    structures, summary, inference, checks = _summarize(
        evaluation, calibration, targets, trajectories, cfg
    )
    calibration.to_csv(root / "calibration_measurements.csv", index=False)
    evaluation.to_csv(root / "evaluation_measurements.csv", index=False)
    structures.to_csv(root / "structure_metrics.csv", index=False)
    summary.to_csv(root / "measurement_summary.csv", index=False)
    spectra.to_csv(root / "baseline_PSD.csv", index=False)
    _write_json(root / "exploratory_structure_inference.json", inference)
    if cfg.experiment.plot:
        _plots(root, evaluation, calibration, spectra, structures, cfg)
    smoke = bool(cfg.analysis.smoke_test)
    passed = all(checks.values()) and not smoke
    _write_json(
        root / "protocol_and_provenance.json",
        {
            "source": {
                "root": source["root"],
                "hashes": source["hashes"],
                "upstream": source["provenance"],
            },
            "mpi_ranks": size,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
            ).stdout.strip(),
            "trajectory_count": len(specs),
            "independent_evaluation_structures": int(
                evaluation.structure_seed.nunique()
            ),
            "zero_field_only": True,
            "smoke_test": smoke,
            "noise_repeats_and_rotations_are_not_independent_structures": True,
            "causality": "Baseline filtering, scale estimation, spectra and screening use baseline samples only. Later sham EEG is used solely for outcome-reference calibration audits.",
            "scope": "Known-head, single-source, equal independent sensor-noise measurement audit. No controller, learned policy, action opportunity or clinical efficacy is validated.",
        },
    )
    conclusion = {
        "checks": checks,
        "status": (
            "SMOKE COMPLETED" if smoke else ("PASSED" if passed else "NOT PASSED")
        ),
        "ready_for_small_response_reassessment": passed,
        "ready_for_policy_confirmation": False,
        "H5_status": "NOT ESTABLISHED",
        "failed_checks": [k for k, v in checks.items() if not v],
        "runtime_seconds": time.perf_counter() - started,
        "smoke_test": smoke,
        "completed_trajectories": len(specs),
    }
    _write_json(root / "experiment_conclusion.json", conclusion)
    # Written last, after figures and conclusions, to distinguish a complete
    # run from a partially saved result after an MPI/Python failure.
    _write_json(
        root / "run_complete.json",
        {
            "completed": True,
            "runtime_seconds": time.perf_counter() - started,
            "files_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.iterdir())
                if p.is_file()
            },
        },
    )
    print("\nH5-O1M checks", flush=True)
    for name, value in checks.items():
        print(f"{name}: {'PASSED' if value else 'NOT PASSED'}")
    print(summary.to_string(index=False))
    print(
        f"\nMeasurement audit: {conclusion['status']}\nResults saved to: {root}",
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
