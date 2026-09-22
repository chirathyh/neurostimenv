"""CF0b: bounded phase-pipeline replay, B-only screen, disjoint qualification.

All new neuronal episodes are stimulation-free. A failed replay stops full
execution before new simulations. No historical controller or cell is edited.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import OmegaConf
from scipy import signal

from experiments.ballnstick_analysis.qualification import common, continuous_alpha as ca
from experiments.ballnstick_analysis.qualification import phase_measurement as pm
from experiments.ballnstick_analysis import run_ballnstick_cf0 as cf0
from experiments.ballnstick_analysis import run_ballnstick_tes_entrainment as online


def validate(cfg):
    cf0.validate(cfg)
    a = cfg.analysis
    if a.execution not in ("replay", "full"):
        raise ValueError("execution must be replay or full")
    p = OmegaConf.to_container(a.phase_audit, resolve=True)
    expected = [("cf0_raw", "raw_ols", .1), ("h4_original", "h4_tail", .03)]
    if [(r["name"], r["pipeline"], r["cutoff"]) for r in p["candidates"]] != expected:
        raise ValueError("The two historical pipelines and cutoffs are frozen")
    fixed = dict(history_s=.5, initialization_s=1., update_s=.125,
        reference_history_s=1., prediction_s=.125, minimum_coverage=.8,
        minimum_diffusion_coverage=.7, minimum_structure_coverage=.6,
        maximum_noise_error_deg=20., maximum_reference_error_deg=45.,
        maximum_large_error_fraction=.1, maximum_prediction_error_deg=60.)
    if any(p[k] != v for k, v in fixed.items()) or p["diagnostic_cutoffs"] != [.03, .1]:
        raise ValueError("Do not retune phase qualification criteria in this study")
    if float(a.rhythm_screen.null_alpha) != .05:
        raise ValueError("The null rank-screen level is fixed at 0.05")
    if a.rhythm_screen.minimum_sensitivity != .8 or a.rhythm_screen.minimum_specificity != .8:
        raise ValueError("Keep the declared rhythm qualification criteria")
    if a.inhibition_scale != 1.0:
        raise ValueError("Do not change recurrent inhibition in a measurement study")
    if not a.smoke and (a.design.calibration_structures != 19 or a.design.validation_structures != 6):
        raise ValueError("Full CF0b uses 19 B calibration and 6 validation structures")
    if min(a.design.calibration_structures, a.design.validation_structures) < 1:
        raise ValueError("Need positive stage counts")
    if cfg.analysis.target_fs_hz != 500 or a.statistics.bootstrap_repeats < 1:
        raise ValueError("Preserve 500-Hz spectral analysis and positive bootstrap count")


def load_source(cfg):
    root = Path(to_absolute_path(str(cfg.analysis.source_cf0.result_dir))).resolve()
    required = ["run_complete.json", "experiment_conclusion.json", "frozen_estimator.json",
                "resolved_config.yaml", "context_metrics.csv", "prespecified_contexts.json"]
    if any(not (root/name).is_file() for name in required):
        raise FileNotFoundError(f"Missing CF0 source files under {root}; copy the COMPLETE cf0 folder including traces")
    conclusion = json.loads((root/"experiment_conclusion.json").read_text())
    if common.sha256(root/"experiment_conclusion.json") != cfg.analysis.source_cf0.conclusion_sha256:
        raise ValueError("The source must be the exact completed negative CF0 result")
    if conclusion["smoke"] or conclusion["continuous_carrier_qualification"] != "NOT PASSED":
        raise ValueError("Expected negative full CF0 source")
    manifest = json.loads((root/"run_complete.json").read_text())
    if not manifest["completed"]:
        raise ValueError("CF0 has no completed manifest")
    for name, digest in manifest["files_sha256"].items():
        path = (root/name).resolve()
        if root not in path.parents or not path.is_file() or common.sha256(path) != digest:
            raise ValueError(f"CF0 source artifact missing/changed: {name}")
    if any(name not in manifest["files_sha256"] for name in required if name != "run_complete.json"):
        raise ValueError("Source manifest does not cover all required metadata")
    frozen = json.loads((root/"frozen_estimator.json").read_text())
    if frozen["smoke_only"] or not frozen["selected_before_replication"]:
        raise ValueError("Source estimator was not frozen")
    source_cfg = OmegaConf.load(root/"resolved_config.yaml")
    for section in ("network", "eeg", "online"):
        if OmegaConf.to_container(source_cfg.env[section], resolve=True) != OmegaConf.to_container(cfg.env[section], resolve=True):
            raise ValueError(f"CF0b must retain the exact CF0 {section} configuration")
    metrics = pd.read_csv(root/"context_metrics.csv")
    metrics = metrics[metrics.estimator == frozen["parameters"]["name"]].copy()
    if len(metrics) != 15 or metrics.id.duplicated().any():
        raise ValueError("Expected the complete 15-episode CF0 discovery set")
    return {"root": str(root), "parameters": frozen["parameters"],
        "rows": metrics.to_dict("records"),
        "file_hashes": {**manifest["files_sha256"], "run_complete.json": common.sha256(root/"run_complete.json")},
        "source_mpi_ranks": conclusion["mpi_ranks"],
        "source_contexts": json.loads((root/"prespecified_contexts.json").read_text())}


def specs(cfg, source):
    design = OmegaConf.to_container(cfg.analysis.design, resolve=True)
    # Engineering smoke observations must not expose full qualification
    # structures, even when shorter recordings cannot qualify.
    if cfg.analysis.smoke:
        design["calibration_structure_start"] += 10000
        design["validation_structure_start"] += 10000
    design["structures_per_stage"] = design["validation_structures"]
    validation = ca.contexts(design, "validation")
    # Safe explicit filename identifiers; do NOT change historical CF0 files.
    for row in validation:
        row["id"] = row["id"].replace(".", "p")
    calibration = []
    for index in range(design["calibration_structures"]):
        structure = design["calibration_structure_start"]+index
        calibration.append({"id": f"calibration_s{index:02d}_B", "stage": "calibration",
            "structure_seed": structure, "drive_seed": structure+1100000,
            "noise_seed": structure+2200000, "carrier_hz": 10., "D": 0., "state": "B"})
    old = {r[k] for stage in source["source_contexts"] for r in stage
           for k in ("structure_seed", "drive_seed", "noise_seed")}
    new_sets = [{r[k] for r in calibration+validation} for k in
                ("structure_seed", "drive_seed", "noise_seed")]
    if old & set.union(*new_sets) or any(new_sets[i] & new_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Seed namespaces overlap sources or stochastic roles")
    if {r["structure_seed"] for r in calibration} & {r["structure_seed"] for r in validation}:
        raise ValueError("Calibration and validation structures overlap")
    if max(new_sets[0])*10000+MPI.COMM_WORLD.size >= 2**32:
        raise ValueError("Structure namespace exceeds uint32")
    return calibration, validation


def save_spectrum(root, spec, result, neural_result):
    folder = root/"spectra"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f"{spec['id']}.npz",
        **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
        neural_psd_v2_per_hz=neural_result["psd_v2_per_hz"])


def measure(neural, unit, scale, fs, baseline_count, spec, parameters, cfg, root, phase=True):
    observed = neural+scale*unit
    factor = int(round(fs/cfg.analysis.target_fs_hz))
    if not np.isclose(fs/factor, cfg.analysis.target_fs_hz):
        raise ValueError("Noninteger EEG downsampling factor")
    b_neural = signal.resample_poly(neural[:baseline_count], 1, factor)
    b_observed = signal.resample_poly(observed[:baseline_count], 1, factor)
    result = ca.estimate(b_observed, cfg.analysis.target_fs_hz, parameters)
    neural_result = ca.estimate(b_neural, cfg.analysis.target_fs_hz, parameters)
    save_spectrum(root, spec, result, neural_result)
    row = {**spec, **cf0.compact_estimate(result),
        "absolute_error_hz": abs(result["frequency_hz"]-spec["carrier_hz"]) if spec["state"] == "A" else np.nan,
        "neural_frequency_hz": neural_result["frequency_hz"],
        "neural_error_hz": abs(neural_result["frequency_hz"]-spec["carrier_hz"]) if spec["state"] == "A" else np.nan,
        "noise_scale_v": scale, "noise_unit_sha256": common.array_hash(unit),
        "baseline_neural_sha256": common.array_hash(neural[:baseline_count])}
    phase_rows = pd.DataFrame()
    if phase:
        phase_rows = pm.audit_trajectory(neural, observed, fs, baseline_count,
            result["frequency_hz"], OmegaConf.to_container(cfg.analysis.phase_audit, resolve=True))
        for key in ("structure_seed", "D", "state"):
            phase_rows[key] = spec[key]
        phase_rows["context"] = spec["id"]
        phase_rows["carrier_accepted"] = result["accepted"]
    return row, phase_rows


def replay(source, cfg, root):
    rows, phases = [], []
    for old in source["rows"]:
        # CF0 used with_suffix; resolve that exact legacy path read-only.
        path = (Path(source["root"])/"traces"/old["id"]).with_suffix(".npz")
        with np.load(path, allow_pickle=False) as data:
            neural, unit = data["neural_raw_v"], data["unit_noise"]
            count, fs, scale = int(data["baseline_count"]), float(data["raw_fs_hz"]), float(data["observation_noise_scale_v"])
        if common.array_hash(unit) != old["noise_unit_sha256"] or common.array_hash(neural[:count]) != old["baseline_neural_sha256"]:
            raise ValueError("Source trajectory/noise hashes do not match")
        spec = {k: old[k] for k in ("id", "structure_seed", "drive_seed", "noise_seed", "carrier_hz", "D", "state")}
        spec["id"] = "replay_"+spec["id"].replace(".", "p")
        spec["stage"] = "replay"
        print(f"CF0b replay {spec['id']}", flush=True)
        row, phase_rows = measure(neural, unit, scale, fs, count, spec, source["parameters"], cfg, root)
        if abs(row["frequency_hz"]-old["frequency_hz"]) > 1e-10 or row["accepted"] != old["accepted"]:
            raise RuntimeError("Frozen carrier does not reproduce source decision")
        row.update(rate_safe=old["rate_safe"], field_residual_mV=old["field_residual_mV"], source_trace=str(path))
        rows.append(row)
        phases.append(phase_rows)
    return pd.DataFrame(rows), pd.concat(phases, ignore_index=True)


def collect_new(episode, spec, cfg, root, parameters, phase):
    epochs = episode["outputs_by_epoch"]
    outputs = epochs["baseline"]+epochs["stimulation"]+epochs["washout"]
    neural = np.concatenate([np.asarray(o["eeg_v"]).reshape(-1) for o in outputs])
    count = sum(np.asarray(o["eeg_v"]).size for o in epochs["baseline"])
    fs = 1000/cfg.env.network.dt
    _, unit, scale = ca.baseline_scaled_noise(neural, count,
        cfg.analysis.observation.ar1_rho, cfg.analysis.observation.rms_fraction, spec["noise_seed"])
    folder = root/"traces"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f"{spec['id']}.npz", neural_raw_v=neural,
        unit_noise=unit, observation_noise_scale_v=scale, raw_fs_hz=fs, baseline_count=count)
    row, phases = measure(neural, unit, scale, fs, count, spec, parameters, cfg, root, phase)
    rates = {}
    for pop in ("E", "I"):
        spikes, n_cells = online._collect_epoch_spikes(outputs, pop)
        rates[pop+"_rate_hz"] = len(spikes)/n_cells/(len(neural)/fs)
    safe = all(cfg.analysis.rate_guardrails_hz[p+"_min"] <= rates[p+"_rate_hz"] <=
               cfg.analysis.rate_guardrails_hz[p+"_max"] for p in ("E", "I"))
    row.update(**rates, rate_safe=safe, field_residual_mV=episode["final_residual_mV"])
    return row, phases


def save_phase_analysis(rows, cfg, root, stage, minimum_structures):
    settings = OmegaConf.to_container(cfg.analysis.phase_audit, resolve=True)
    summaries, contexts, structures = pm.phase_summary(rows, settings, minimum_structures)
    rows.to_csv(root/f"{stage}_phase_windows.csv", index=False)
    contexts.to_csv(root/f"{stage}_phase_contexts.csv", index=False)
    structures.to_csv(root/f"{stage}_phase_structures.csv", index=False)
    # Matched-cutoff attribution only. These extra rows cannot be selected.
    crossed = []
    for cutoff in settings["diagnostic_cutoffs"]:
        subset = rows[rows.state == "A"].copy()
        subset["accepted_at_cutoff"] = subset.confidence >= cutoff
        subset["accepted_noise_error_deg"] = subset.noise_error_deg.where(subset.accepted_at_cutoff)
        subset["accepted_common_error_deg"] = subset.reference_error_deg.where(subset.accepted_at_cutoff)
        subset["accepted_large_error_fraction"] = subset.large_error.where(subset.accepted_at_cutoff)
        metrics = ["accepted_at_cutoff", "accepted_noise_error_deg", "accepted_common_error_deg",
                   "accepted_large_error_fraction"]
        context = subset.groupby(["profile", "structure_seed", "context"])[metrics].mean()
        values = context.groupby(["profile", "structure_seed"]).mean().reset_index()
        values["cutoff"] = cutoff
        crossed.append(values)
    pd.concat(crossed).to_csv(root/f"{stage}_matched_cutoff_audit.csv", index=False)
    for profile in summaries:
        group = structures[structures.profile == profile]
        summaries[profile]["exploratory_structure_bootstrap_95"] = {
            key: pm.structure_bootstrap(group[key], cfg.analysis.statistics.bootstrap_repeats,
                                       cfg.analysis.statistics.bootstrap_seed)
            for key in ("coverage", "reference_error_deg", "prediction_error_deg")}
    paired = structures.pivot(index="structure_seed", columns="profile",
                              values=["coverage", "reference_error_deg", "prediction_error_deg"])
    contrasts = {}
    for metric in ("coverage", "reference_error_deg", "prediction_error_deg"):
        differences = paired[metric]["h4_original"]-paired[metric]["cf0_raw"]
        contrasts[metric] = {"contrast": "H4 minus CF0; higher coverage but lower error is preferable",
            "mean": float(differences.mean()), "independent_structure_count": len(differences),
            "paired_structure_bootstrap_95": pm.structure_bootstrap(differences,
                cfg.analysis.statistics.bootstrap_repeats, cfg.analysis.statistics.bootstrap_seed),
            "confirmatory_superiority_test": False}
    common.write_json(root/f"{stage}_paired_pipeline_contrasts.json", contrasts)
    common.write_json(root/f"{stage}_phase_summary.json", summaries)
    return summaries


def validation_summary(frame, phase_rows, selected, screen, cfg, root):
    a = frame[frame.state == "A"].copy()
    b = frame[frame.state == "B"].copy()
    a["correct"] = a.absolute_error_hz <= .25
    a["treatment_eligible"] = a.rhythm_present & a.accepted
    table = a.groupby("structure_seed").agg(frequency_accuracy=("correct", "mean"),
        carrier_coverage=("accepted", "mean"), phenotype_sensitivity=("rhythm_present", "mean"),
        treatment_coverage=("treatment_eligible", "mean"), mae_hz=("absolute_error_hz", "mean"))
    table.to_csv(root/"validation_measurement_structures.csv")
    accepted = a[a.accepted]
    accepted_accuracy = accepted.groupby("structure_seed").correct.mean().mean() if len(accepted) else 0.
    specific = (~b.rhythm_present).astype(int)
    phase = save_phase_analysis(phase_rows, cfg, root, "validation", 6)
    checks = {
        "six_disjoint_structures": len(table) >= 6,
        "finite_B_calibrated_rank_threshold": screen["threshold_db"] is not None,
        "all_context_carrier_accuracy": table.frequency_accuracy.mean() >= .8,
        "accepted_carrier_accuracy": accepted_accuracy >= .9,
        "carrier_coverage": table.carrier_coverage.mean() >= .8,
        "each_diffusion_carrier_accuracy": bool((a.groupby("D").correct.mean() >= .75).all()),
        "each_structure_carrier_accuracy": bool((table.frequency_accuracy >= .5).all()),
        "rhythm_sensitivity": table.phenotype_sensitivity.mean() >= cfg.analysis.rhythm_screen.minimum_sensitivity,
        "rhythm_specificity_point_estimate": specific.mean() >= cfg.analysis.rhythm_screen.minimum_specificity,
        "combined_carrier_rhythm_coverage": table.treatment_coverage.mean() >= .75,
        "frozen_phase_profile_qualifies": phase[selected["name"]]["passes"],
        "all_rates_safe": bool(frame.rate_safe.all()),
        "all_fields_zero": bool((frame.field_residual_mV == 0).all()),
        "phase_inputs_causal": bool((phase_rows.latest_input_s <= phase_rows.boundary_s+1e-12).all()),
    }
    return {"checks": checks, "passes": all(checks.values()),
        "measurement_means": table.mean().to_dict(), "accepted_frequency_accuracy": accepted_accuracy,
        "B_specificity": float(specific.mean()), "B_correct_count": int(specific.sum()),
        "B_structure_count": len(b),
        "B_specificity_exact_95_interval": pm.exact_binomial_interval(int(specific.sum()), len(b)),
        "not_powered_specificity_confirmation": True,
        "phase_summary": phase[selected["name"]]}


def apply_frozen_screen(row, phases, screen, selected):
    """Record sham fallback from observable prerequisites, without treatment."""
    row["rhythm_present"] = pm.rhythm_present(row["evidence_db"], screen)
    row["screening_uses_only_predecision_observed_EEG"] = True
    row["carrier_and_rhythm_eligible"] = bool(row["rhythm_present"] and row["accepted"])
    initial = phases[(phases.profile == selected["name"]) & (phases.elapsed_s == 0)]
    if len(initial) != 1:
        raise RuntimeError("Expected exactly one initialization estimate for frozen profile")
    row["initial_phase_actionable"] = bool(initial.actionable.iloc[0])
    row["initial_measurement_eligible"] = row["carrier_and_rhythm_eligible"] and row["initial_phase_actionable"]
    row["future_treatment_fallback"] = "none_required" if row["initial_measurement_eligible"] else "sham"
    row["stimulation_actually_applied"] = False
    phases["rhythm_present"] = row["rhythm_present"]
    phases["deployable_phase_available"] = phases.actionable & row["carrier_and_rhythm_eligible"]
    return row, phases


def figures(frame, phases, cfg, root, stage, screen=None):
    if not cfg.experiment.plot:
        return
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(root/f"{stage}_{name}.{suffix}", dpi=160)
        plt.close(fig)
    # First complete structure chosen by seed, not by a visually attractive PSD.
    chosen = frame[frame.structure_seed == frame.structure_seed.min()]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 2.25*len(chosen)), squeeze=False)
    for (_, row), axs in zip(chosen.iterrows(), axes):
        with np.load(root/"spectra"/f"{row.id}.npz") as data:
            f = data["f_hz"]; keep = (f >= 6) & (f <= 14)
            axs[0].semilogy(f[keep], data["psd_v2_per_hz"][keep], label="Observed")
            axs[0].semilogy(f[keep], data["neural_psd_v2_per_hz"][keep], label="Neural", alpha=.7)
            axs[0].axvline(row.frequency_hz, color="C3", ls="--", label="Estimated")
            if row.state == "A":
                axs[0].axvline(row.carrier_hz, color="k", ls=":", label="Generator (audit)")
            for line in data["window_evidence_db"]:
                axs[1].plot(data["grid_hz"], line, color=".8", lw=.5)
            axs[1].plot(data["grid_hz"], data["pooled_evidence_db"], color="C1", label="Pooled")
            if screen and screen["threshold_db"] is not None:
                axs[1].axhline(screen["threshold_db"], color="k", ls="--", label="B-only cutoff")
        axs[0].set(title=row.id, xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)")
        axs[1].set(xlabel="Candidate carrier (Hz)", ylabel="Adjusted evidence (dB)")
    for ax in axes[0]:
        ax.legend(fontsize=7)
    save(fig, "PSD_evidence")
    if len(phases):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        ctx = pd.read_csv(root/f"{stage}_phase_contexts.csv")
        for index, (name, group) in enumerate(ctx.groupby("profile")):
            for ax, key in zip(axes, ("coverage", "reference_error_deg", "prediction_error_deg")):
                structure = group.groupby("structure_seed")[key].mean()
                ax.scatter(np.full(len(structure), index), structure, label=name)
                ax.plot([index-.2, index+.2], [structure.mean()]*2, color="k")
        for ax, title in zip(axes, ("Actionable fraction", "Common-reference error (deg)", "125-ms prediction error (deg)")):
            ax.set(xticks=[0, 1], xticklabels=sorted(ctx.profile.unique()), ylabel=title)
        axes[0].axhline(.8, color="k", ls=":")
        axes[1].axhline(45, color="k", ls=":")
        axes[2].axhline(60, color="k", ls=":")
        save(fig, "phase_comparison")
        first = phases[(phases.context == phases[phases.state == "A"].context.iloc[0])]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for name, group in first.groupby("profile"):
            axes[0].plot(group.elapsed_s, group.reference_error_deg, label=name)
            axes[1].plot(group.elapsed_s, group.confidence, label=name)
        axes[0].set(ylabel="Offline reference error (deg)", title="First rhythmic context; benchmark is not phase ground truth")
        axes[1].set(xlabel="Continuation time (s)", ylabel="Pipeline amplitude/RMS")
        axes[0].legend(); axes[1].legend()
        save(fig, "phase_trace")
    if screen:
        calibration = pd.read_csv(root/"calibration_measurements.csv")
        fig, ax = plt.subplots(figsize=(7, 4))
        for index, (label, values) in enumerate((
                ("Calibration B", calibration.evidence_db),
                ("Validation B", frame[frame.state == "B"].evidence_db),
                ("Validation A", frame[frame.state == "A"].evidence_db))):
            ax.scatter(np.full(len(values), index), values, label=label, alpha=.7)
        if screen["threshold_db"] is not None:
            ax.axhline(screen["threshold_db"], color="k", ls="--", label="Frozen rank threshold")
        ax.set(xticks=[0, 1, 2], xticklabels=["Calibration B", "Validation B", "Validation A"],
               ylabel="Maximum pooled evidence (dB)")
        ax.legend(fontsize=8)
        save(fig, "rhythm_screen")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    start = time.perf_counter()
    comm = MPI.COMM_WORLD
    validate(cfg)
    source = load_source(cfg) if comm.rank == 0 else None
    source = comm.bcast(source, root=0)
    if cfg.analysis.execution == "full" and not cfg.analysis.smoke and comm.size != source["source_mpi_ranks"]:
        raise ValueError(f"Full CF0b retains CF0's MPI size: use -n {source['source_mpi_ranks']}")
    calibration, validation = specs(cfg, source)
    root = Path(to_absolute_path(str(cfg.experiment.dir)))/"cf0b"
    selected, replay_summaries, replay_passed, frozen_hash = None, None, False, None
    if comm.rank == 0:
        common.begin(root, cfg, __file__)
        common.write_json(root/"source_lock.json", {k: v for k, v in source.items() if k != "rows"})
        common.write_json(root/"prespecified_contexts.json", {"calibration": calibration, "validation": validation})
        common.write_json(root/"protocol.json", {"phase_audit": OmegaConf.to_container(cfg.analysis.phase_audit),
            "rhythm_screen": OmegaConf.to_container(cfg.analysis.rhythm_screen),
            "stimulation": "none", "controller_edits": "none",
            "reference": "Centered 1-s neural-only OLS at EEG-selected f; offline benchmark, not biological ground truth",
            "prediction": "Extrapolate current phase 125 ms at EEG-selected carrier; compare to same offline benchmark",
            "selection": "Passing profiles only; minimum unconditional common-reference error then noise error",
            "future_samples_used_by_deployable_estimator": False,
            "new_runs_mpi_size": comm.size,
            "smoke_structure_namespace_offset": 10000 if cfg.analysis.smoke else 0,
            "source_helper_sha256": {str(Path(module.__file__).relative_to(common.REPO)): common.sha256(module.__file__)
                                      for module in (cf0, online)},
            "historical_phase_primitive_sha256": {name: common.sha256(common.REPO/name) for name in (
                "experiments/ballnstick_analysis/run_ballnstick.py",
                "experiments/ballnstick_analysis/run_ballnstick_hierarchical_tacs.py")},
            "claims": "Exploratory measurement qualification only. No tACS efficacy, instantaneous-phase ground truth or H5 claim."})
        replay_frame, replay_phases = replay(source, cfg, root)
        replay_frame.to_csv(root/"replay_measurements.csv", index=False)
        replay_summaries = save_phase_analysis(replay_phases, cfg, root, "replay", 3)
        selected = pm.choose_profile(replay_summaries, OmegaConf.to_container(cfg.analysis.phase_audit.candidates))
        replay_passed = selected is not None and bool(replay_frame.rate_safe.all()) and bool((replay_frame.field_residual_mV == 0).all())
        # A smoke can force plumbing after failure, but can NEVER qualify.
        if selected is None and cfg.analysis.smoke:
            selected = OmegaConf.to_container(cfg.analysis.phase_audit.candidates[1])
        common.write_json(root/"frozen_measurement.json", {"carrier": source["parameters"],
            "phase_profile": selected, "phase_criteria": OmegaConf.to_container(cfg.analysis.phase_audit),
            "replay_passed": replay_passed, "smoke_only": bool(cfg.analysis.smoke),
            "selected_before_new_simulations": True})
        frozen_hash = common.sha256(root/"frozen_measurement.json")
        figures(replay_frame, replay_phases, cfg, root, "replay")
    selected, replay_summaries, replay_passed, frozen_hash = comm.bcast(
        (selected, replay_summaries, replay_passed, frozen_hash), root=0)
    new_rows, new_phases, screen, outcome = [], [], None, None
    calibration_passed = None
    run_new = cfg.analysis.execution == "full" and (replay_passed or cfg.analysis.smoke)
    for stage, stage_specs in (("calibration", calibration), ("validation", validation)) if run_new else []:
        stage_rows = []
        for spec in stage_specs:
            if comm.rank == 0:
                if common.sha256(root/"frozen_measurement.json") != frozen_hash:
                    raise RuntimeError("Frozen measurement changed during experiment")
                if stage == "validation" and common.sha256(root/"frozen_rhythm_screen.json") != screen_hash:
                    raise RuntimeError("Frozen B screen changed during validation")
                print(f"CF0b {spec['id']}: zero field", flush=True)
            episode = online._simulate_episode(cf0.condition(cfg, spec), seed=spec["drive_seed"],
                action=online._zero_action(cfg), stimulate=False, output_dir=root/"episodes"/spec["id"],
                comm=comm, size=comm.size, rank=comm.rank,
                structure_seed=spec["structure_seed"], drive_seed=spec["drive_seed"])
            if comm.rank == 0:
                row, phase_rows = collect_new(episode, spec, cfg, root, source["parameters"], stage == "validation")
                if screen:
                    row, phase_rows = apply_frozen_screen(row, phase_rows, screen, selected)
                stage_rows.append(row)
                new_rows.append(row)
                if len(phase_rows):
                    new_phases.append(phase_rows)
                pd.DataFrame(stage_rows).to_csv(root/f"{stage}_measurements.csv", index=False)
            comm.barrier()
        if comm.rank == 0:
            frame = pd.DataFrame(stage_rows)
            if stage == "calibration":
                screen = pm.rank_threshold(frame.evidence_db, cfg.analysis.rhythm_screen.null_alpha)
                common.write_json(root/"frozen_rhythm_screen.json", screen)
                screen_hash = common.sha256(root/"frozen_rhythm_screen.json")
                calibration_passed = bool(screen["threshold_db"] is not None and frame.rate_safe.all()
                    and (frame.field_residual_mV == 0).all())
                common.write_json(root/"calibration_summary.json", {"passes": calibration_passed,
                    "screen": screen, "all_rates_safe": bool(frame.rate_safe.all()),
                    "all_fields_zero": bool((frame.field_residual_mV == 0).all())})
            else:
                if common.sha256(root/"frozen_rhythm_screen.json") != screen_hash:
                    raise RuntimeError("Frozen B screen changed during validation")
                phases = pd.concat(new_phases, ignore_index=True)
                outcome = validation_summary(frame, phases, selected, screen, cfg, root)
                common.write_json(root/"validation_summary.json", outcome)
                figures(frame, phases, cfg, root, "validation", screen)
        screen, calibration_passed = comm.bcast((screen, calibration_passed), root=0)
        if stage == "calibration" and not calibration_passed and not cfg.analysis.smoke:
            break
    if comm.rank == 0:
        # All safety records, including calibration, are mandatory.
        safe = all(r["rate_safe"] and r["field_residual_mV"] == 0 for r in new_rows)
        qualified = bool(not cfg.analysis.smoke and replay_passed and outcome and outcome["passes"] and safe)
        common.finish(root, start, {"experiment": "CF0b", "smoke": bool(cfg.analysis.smoke),
            "execution": cfg.analysis.execution, "replayed_episodes": 15,
            "new_neural_episodes": len(new_rows), "mpi_ranks": comm.size,
            "replay_passed": replay_passed, "selected_phase_profile": selected,
            "replay_summaries": replay_summaries, "validation": outcome,
            "new_simulations_skipped": not run_new,
            "calibration_passed": calibration_passed,
            "all_new_runs_safe_and_zero_field": safe,
            "frozen_measurement_sha256": frozen_hash,
            "frozen_screen_sha256": common.sha256(root/"frozen_rhythm_screen.json") if screen else None,
            "measurement_qualification": "PASSED" if qualified else "NOT PASSED",
            "H5": "NOT TESTED", "phase_reference_is_not_ground_truth": True,
            "qualification_is_exploratory_not_powered_confirmation": True})
    comm.barrier()


if __name__ == "__main__":
    common.guarded_main(main)
