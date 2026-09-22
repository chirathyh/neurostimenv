"""CF0: continuous-alpha measurement development and disjoint qualification.

No stimulation, no policy, no claims of statistically powered confirmation.
The historical binary carrier estimator and all H1--H4 code are unchanged.
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
from scipy import signal, stats

from experiments.ballnstick_analysis.qualification import common, continuous_alpha as ca
from experiments.ballnstick_analysis import run_ballnstick_tes_entrainment as online


def validate(cfg):
    a = cfg.analysis
    if cfg.env.name != "ballnstick":
        raise ValueError("CF0 is BallAndStick only; pass env=ballnstick")
    if dict(cfg.env.network.population.sizes) != {"E": 32, "I": 8} or cfg.env.network.dt != .0625:
        raise ValueError("CF0 retains the 40-cell, 0.0625-ms reference network")
    if cfg.env.network.celsius != 6.3 or cfg.env.simulation.obs_win_len != 1000:
        raise ValueError("CF0 requires canonical HH at 6.3 C and 1000-ms online windows")
    if a.design.structures_per_stage < 1 or a.timeline.baseline_steps < 2:
        raise ValueError("Need structures and at least two baseline seconds")
    if not a.smoke and (a.design.structures_per_stage < 3 or a.timeline.burn_in_steps != 1 or
            a.timeline.baseline_steps != 30 or
            a.timeline.stimulation_steps + a.timeline.washout_steps != 8):
        raise ValueError("Full CF0 requires >=3 structures per stage and 1+30+8 s")
    if list(a.design.diffusion_levels) != [0.5, 2.0] or a.design.modulation_depth != .04:
        raise ValueError("Keep the prespecified D={0.5,2}, m=.04 generator")
    if a.observation.rms_fraction != .25 or a.observation.ar1_rho != .95:
        raise ValueError("Keep the prespecified observation model")
    stages = [ca.contexts(OmegaConf.to_container(a.design), s) for s in ("discovery", "replication")]
    seed_sets = [{r["structure_seed"] for r in stage} for stage in stages]
    if seed_sets[0] & seed_sets[1] or max(seed_sets[0] | seed_sets[1]) * 10000 + MPI.COMM_WORLD.size >= 2**32:
        raise ValueError("Disjoint, uint32-safe structure namespaces required")
    return stages


def condition(cfg, spec):
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for population in ("E", "I"):
        rhythm = out.env.network.background[population].rhythm
        rhythm.enabled = True
        rhythm.modulation_depth = out.analysis.design.modulation_depth if spec["state"] == "A" else 0.
        rhythm.thinning_envelope_modulation_depth = out.analysis.design.modulation_depth
        rhythm.frequency_hz = spec["carrier_hz"]
        rhythm.phase_rad = float(np.random.default_rng(spec["drive_seed"]).uniform(0, 2*np.pi))
        rhythm.phase_diffusion_rad2_per_s = spec["D"]
        rhythm.phase_diffusion_integration_dt_ms = 1.
        rhythm.shared_modulated_fraction = 1.
    return out


def compact_estimate(result):
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def analyze_episode(episode, spec, candidates, cfg, root):
    epochs = episode["outputs_by_epoch"]
    outputs = epochs["baseline"] + epochs["stimulation"] + epochs["washout"]
    raw = np.concatenate([np.asarray(o["eeg_v"]).reshape(-1) for o in outputs])
    baseline_count = sum(np.asarray(o["eeg_v"]).size for o in epochs["baseline"])
    fs = 1000 / cfg.env.network.dt
    observed, unit, scale = ca.baseline_scaled_noise(raw, baseline_count,
        cfg.analysis.observation.ar1_rho, cfg.analysis.observation.rms_fraction, spec["noise_seed"])
    # Each baseline is preprocessed independently of its future continuation.
    factor = int(round(fs / cfg.analysis.target_fs_hz))
    if not np.isclose(fs / factor, cfg.analysis.target_fs_hz):
        raise ValueError("CF0 requires an integer raw-to-analysis sample ratio")
    b_neural = signal.resample_poly(raw[:baseline_count], 1, factor)
    b_observed = signal.resample_poly(observed[:baseline_count], 1, factor)
    path = root / "traces" / spec["id"]
    path.parent.mkdir(exist_ok=True)
    # Save the ORIGINAL noise vector, not a reconstruction from rounded EEG.
    np.savez_compressed(path.with_suffix(".npz"), neural_raw_v=raw, unit_noise=unit,
        observation_noise_scale_v=scale, raw_fs_hz=fs, baseline_count=baseline_count,
        baseline_neural_v=b_neural, baseline_observed_v=b_observed,
        analysis_fs_hz=float(cfg.analysis.target_fs_hz))
    baseline_s = baseline_count / fs
    start_s = float(cfg.analysis.timeline.burn_in_steps)
    rates = {}
    for pop in ("E", "I"):
        spikes, n_cells = online._collect_epoch_spikes(outputs, pop)
        rates[pop + "_rate_hz"] = len(spikes) / n_cells / (len(raw) / fs)
    safe = all(cfg.analysis.rate_guardrails_hz[f"{p}_min"] <= rates[p + "_rate_hz"] <=
               cfg.analysis.rate_guardrails_hz[f"{p}_max"] for p in ("E", "I"))
    rows, phase_rows = [], []
    for parameters in candidates:
        estimate = ca.estimate(b_observed, cfg.analysis.target_fs_hz, parameters)
        neural = ca.estimate(b_neural, cfg.analysis.target_fs_hz, parameters)
        freq = estimate["frequency_hz"]
        np.savez_compressed(path.parent / f"{spec['id']}_{parameters['name']}_spectra.npz",
            **{k: v for k, v in estimate.items() if isinstance(v, np.ndarray)},
            neural_psd_v2_per_hz=neural["psd_v2_per_hz"], neural_evidence_db=neural["pooled_evidence_db"])
        phases = []
        # All estimates use only [boundary-history, boundary]; initialization 1s.
        for elapsed in np.arange(0, len(raw)/fs - baseline_s + 1e-8, cfg.analysis.estimator.phase_update_s):
            stop = int(round((baseline_s + elapsed) * fs))
            history = 1. if elapsed == 0 else float(cfg.analysis.estimator.phase_history_s)
            phase, confidence = ca.phase_at_boundary(observed[:stop], fs, freq,
                start_s + baseline_s + elapsed, history)
            truth, _ = ca.phase_at_boundary(raw[:stop], fs, freq,
                start_s + baseline_s + elapsed, history)
            row = {"context": spec["id"], "estimator": parameters["name"],
                "elapsed_s": elapsed, "history_s": history, "latest_input_s": start_s + stop/fs,
                "boundary_s": start_s + baseline_s + elapsed, "phase_rad": phase,
                "neural_same_frequency_phase_rad": truth,
                "observation_phase_error_rad": abs(np.angle(np.exp(1j * (phase-truth)))),
                "amplitude_to_rms": confidence,
                "actionable": confidence >= cfg.analysis.estimator.phase_minimum_amplitude_to_rms}
            phases.append(row)
        phase_rows.extend(phases)
        rows.append({**spec, "estimator": parameters["name"], **compact_estimate(estimate),
            "absolute_error_hz": abs(freq-spec["carrier_hz"]) if spec["state"] == "A" else np.nan,
            "neural_estimate_hz": neural["frequency_hz"], "neural_accepted": neural["accepted"],
            "neural_error_hz": abs(neural["frequency_hz"]-spec["carrier_hz"]) if spec["state"] == "A" else np.nan,
            "phase_actionable_fraction": np.mean([r["actionable"] for r in phases]),
            "mean_observation_phase_error_rad": np.mean([r["observation_phase_error_rad"] for r in phases]),
            "noise_unit_sha256": common.array_hash(unit), "noise_scale_v": scale,
            "baseline_neural_sha256": common.array_hash(raw[:baseline_count]),
            "field_residual_mV": episode["final_residual_mV"], "rate_safe": safe, **rates})
    return rows, phase_rows


def summarize(frame, cfg):
    a = frame[frame.state == "A"].copy()
    a["correct"] = a.absolute_error_hz <= cfg.analysis.estimator.error_tolerance_hz
    per_structure = a.groupby("structure_seed").agg(coverage=("accepted", "mean"),
        accuracy=("correct", "mean"), mae_hz=("absolute_error_hz", "mean"),
        phase_actionability=("phase_actionable_fraction", "mean"))
    accepted = a[a.accepted]
    groups = a.groupby("D").agg(coverage=("accepted", "mean"), accuracy=("correct", "mean"))
    coverage = float(per_structure.coverage.mean())
    accuracy = float(per_structure.accuracy.mean())
    # Equal structure weight, not treating windows as independent observations.
    accepted_accuracy = float(accepted.groupby("structure_seed").correct.mean().mean()) if len(accepted) else 0.
    checks = {
        "minimum_independent_structures": len(per_structure) >= 3,
        "identification_coverage": coverage >= cfg.analysis.estimator.minimum_coverage,
        "accepted_frequency_accuracy": accepted_accuracy >= cfg.analysis.estimator.minimum_accepted_accuracy,
        "all_context_frequency_accuracy": accuracy >= cfg.analysis.estimator.minimum_all_context_accuracy,
        "both_diffusion_levels_measurable": bool((groups.accuracy >= .75).all() and (groups.coverage >= .75).all()),
        "no_structure_accuracy_below_half": bool((per_structure.accuracy >= .5).all()),
        "recent_phase_actionable": float(per_structure.phase_actionability.mean()) >= cfg.analysis.estimator.minimum_phase_actionable_fraction,
        "rate_safe": bool(frame.rate_safe.all()),
        "exact_zero_field": bool((frame.field_residual_mV == 0).all()),
    }
    means = per_structure.mae_hz.to_numpy()
    ci = stats.t.interval(.95, len(means)-1, loc=np.mean(means), scale=stats.sem(means)) if len(means)>1 and np.std(means)>0 else [np.nan, np.nan]
    summary = {"checks": checks, "passes": all(checks.values()),
        "coverage": coverage, "all_context_accuracy": accuracy,
        "accepted_accuracy": accepted_accuracy, "mean_absolute_error_hz": float(np.mean(means)),
        "exploratory_structure_t_interval_95_mae_hz": ci,
        "B_rhythm_false_acceptance_audit": float(frame[frame.state == "B"].accepted.mean()),
        "B_audit_is_not_a_validated_rhythm_screen": True,
        "neural_only_accuracy_audit": float((a.neural_error_hz <= cfg.analysis.estimator.error_tolerance_hz).mean()),
        "by_diffusion": groups.reset_index().to_dict("records")}
    return summary, per_structure.reset_index()


def plots(frame, selected_name, root):
    chosen = frame[frame.estimator == selected_name]
    nrows = len(chosen)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, max(3, 2.2*nrows)), squeeze=False)
    for (_, row), ax in zip(chosen.iterrows(), axes):
        data = np.load(root / "traces" / f"{row.id}_{selected_name}_spectra.npz")
        f = data["f_hz"]
        mask = (f >= 6) & (f <= 14)
        ax[0].semilogy(f[mask], data["psd_v2_per_hz"][mask], label="Observed")
        ax[0].semilogy(f[mask], data["neural_psd_v2_per_hz"][mask], alpha=.7, label="Neural only")
        if row.state == "A":
            ax[0].axvline(row.carrier_hz, color="k", ls=":", label="Generator (audit)")
        ax[0].axvline(row.frequency_hz, color="C3", ls="--", label="EEG estimate")
        ax[0].set(title=f"{row.id}; accepted={row.accepted}", xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)")
        for e in data["window_evidence_db"]:
            ax[1].plot(data["grid_hz"], e, color=".75", lw=.7)
        ax[1].plot(data["grid_hz"], data["pooled_evidence_db"], color="C1", label="Pooled")
        ax[1].set(xlabel="Candidate frequency (Hz)", ylabel="Adjusted evidence (dB)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"{chosen.iloc[0].stage}_PSD_evidence.{ext}", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    a = chosen[chosen.state == "A"]
    for diffusion, group in a.groupby("D"):
        axes[0].scatter(group.carrier_hz, group.frequency_hz, label=f"D={diffusion:g}")
        axes[1].scatter(group.absolute_error_hz, group.phase_actionable_fraction, label=f"D={diffusion:g}")
    axes[0].plot([8, 12], [8, 12], "k:")
    axes[0].set(xlabel="Hidden carrier (Hz; evaluation only)", ylabel="EEG estimate (Hz)")
    axes[1].set(xlabel="Absolute frequency error (Hz)", ylabel="Causal phase actionability fraction", ylim=(-.05, 1.05))
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"{chosen.iloc[0].stage}_measurement_summary.{ext}", dpi=160)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    started = time.perf_counter()
    comm = MPI.COMM_WORLD
    stages = validate(cfg)
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "cf0"
    if comm.rank == 0:
        common.begin(root, cfg, __file__)
        common.write_json(root / "prespecified_contexts.json", stages)
    comm.barrier()
    candidates = OmegaConf.to_container(cfg.analysis.estimator.candidates, resolve=True)
    all_rows, all_phases, summaries = [], [], {}
    selected, frozen_hash = None, None
    for stage_index, specs in enumerate(stages):
        if stage_index == 1 and not summaries["discovery"]["passes"] and not cfg.analysis.smoke:
            break
        stage_rows = []
        active_candidates = candidates if stage_index == 0 else [selected]
        for spec in specs:
            if comm.rank == 0:
                print(f"CF0 {spec['id']}: {spec['carrier_hz']:.5f} Hz, D={spec['D']}; zero field", flush=True)
            episode = online._simulate_episode(condition(cfg, spec), seed=spec["drive_seed"],
                action=online._zero_action(cfg), stimulate=False, output_dir=root / "episodes" / spec["id"],
                comm=comm, size=comm.size, rank=comm.rank,
                structure_seed=spec["structure_seed"], drive_seed=spec["drive_seed"])
            if comm.rank == 0:
                rows, phases = analyze_episode(episode, spec, active_candidates, cfg, root)
                stage_rows.extend(rows)
                all_rows.extend(rows)
                all_phases.extend(phases)
                pd.DataFrame(all_rows).to_csv(root / "context_metrics.csv", index=False)
                pd.DataFrame(all_phases).to_csv(root / "causal_phase_audit.csv", index=False)
            comm.barrier()
        if comm.rank == 0:
            frame = pd.DataFrame(stage_rows)
            candidate_summaries = {}
            for p in active_candidates:
                summary, structures = summarize(frame[frame.estimator == p["name"]], cfg)
                candidate_summaries[p["name"]] = summary
                structures.to_csv(root / f"{specs[0]['stage']}_{p['name']}_structure_metrics.csv", index=False)
            if stage_index == 0:
                # Declared lexicographic selection: complete gate, accepted accuracy,
                # all-context accuracy, coverage, then smaller unconditional MAE.
                selected = max(candidates, key=lambda p: (
                    candidate_summaries[p["name"]]["passes"],
                    candidate_summaries[p["name"]]["accepted_accuracy"],
                    candidate_summaries[p["name"]]["all_context_accuracy"],
                    candidate_summaries[p["name"]]["coverage"],
                    -candidate_summaries[p["name"]]["mean_absolute_error_hz"]))
                common.write_json(root / "frozen_estimator.json", {"parameters": selected,
                    "scoring_and_abstention": OmegaConf.to_container(cfg.analysis.estimator),
                    "discovery": candidate_summaries, "selected_before_replication": True,
                    "smoke_only": bool(cfg.analysis.smoke)})
                frozen_hash = common.sha256(root / "frozen_estimator.json")
            elif common.sha256(root / "frozen_estimator.json") != frozen_hash:
                raise RuntimeError("Frozen estimator changed during replication")
            summaries[specs[0]["stage"]] = candidate_summaries[selected["name"]]
            common.write_json(root / f"{specs[0]['stage']}_summary.json", candidate_summaries)
            if cfg.experiment.plot:
                plots(frame, selected["name"], root)
        summaries, selected, frozen_hash = comm.bcast((summaries, selected, frozen_hash), root=0)
    if comm.rank == 0:
        qualified = not cfg.analysis.smoke and all(summaries.get(s, {}).get("passes", False) for s in ("discovery", "replication"))
        common.finish(root, started, {"experiment": "CF0", "smoke": bool(cfg.analysis.smoke),
            "neural_episodes": len({r["id"] for r in all_rows}), "mpi_ranks": comm.size,
            "selected_estimator": selected, "frozen_estimator_sha256": frozen_hash,
            "stimulation_applied": False, "summaries": summaries,
            "continuous_carrier_qualification": "PASSED" if qualified else "NOT PASSED",
            "H5": "NOT TESTED", "replication_is_exploratory_not_powered_confirmation": True})
    comm.barrier()


if __name__ == "__main__":
    common.guarded_main(main)
