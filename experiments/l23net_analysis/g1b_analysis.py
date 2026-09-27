"""Frozen G1B analysis and pairing checks; usable without importing NEURON."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import h5py
import numpy as np
from scipy import signal

from experiments.l23net_analysis.replay_validation import canonical_json_sha256, trace_content_summary

REPORT_NAME = "l23net_g1b_run.json"
TRACE_NAME = "l23net_g1b_trace.h5"
PILOT_SEEDS = (7101, 7102)
BANDS = {"theta": (4., 8.), "alpha": (8., 12.), "low_beta": (12., 16.)}
POPULATIONS = {"HL23PYR": 800, "HL23SST": 50, "HL23PV": 70, "HL23VIP": 80}


def spectral_views(eeg, dt_ms):
    """Legacy double 4-s trim and stable SOS sensitivity, both effective 8--28 s.

    The new trace convention is (0, T], so retained sample centers are
    8000+dt,...,28000 ms. The historical CSV included t=0; this one-sample
    convention is explicitly retained rather than fabricating an initial sample.
    """
    eeg = np.asarray(eeg, dtype=float).reshape(-1)
    fs = 1000. / dt_ms
    n4 = int(round(4000. / dt_ms))
    if eeg.size != int(round(28000. / dt_ms)) or not np.all(np.isfinite(eeg)):
        raise ValueError("Scientific G1B analysis requires one finite 28-s EEG channel.")
    b, a = signal.butter(2, [.1, 100.], btype="bandpass", fs=fs, output="ba")
    sos = signal.butter(2, [.1, 100.], btype="bandpass", fs=fs, output="sos")
    traces = {
        "legacy": signal.filtfilt(b, a, eeg[n4:])[n4:],
        "corrected_sos": signal.sosfiltfilt(sos, eeg)[2*n4:],
    }
    rows, spectra = {}, {}
    for view, values in traces.items():
        # Explicit SciPy Welch defaults used by case_study/replication.py.
        freq, psd = signal.welch(values, fs=fs, window="hann", nperseg=int(fs/2),
                                 noverlap=int(fs/4), detrend="constant", scaling="density")
        if not np.all(np.isfinite(psd)):
            raise ValueError(f"{view} produced non-finite PSD.")
        powers = {name: float(np.trapz(psd[(freq >= low) & (freq <= high)], freq[(freq >= low) & (freq <= high)]))
                  for name, (low, high) in BANDS.items()}
        if any(not np.isfinite(p) or p <= 0 for p in powers.values()):
            raise ValueError(f"{view} has non-positive/non-finite band power.")
        rows[view] = {"band_power_v2": powers, "samples": int(values.size),
                      "effective_interval_ms": [8000., 28000.], "welch_resolution_hz": float(freq[1]-freq[0])}
        spectra[view] = {"frequency_hz": freq, "psd_v2_per_hz": psd}
    return rows, spectra


def normalized_pair_contract(report):
    contract = copy.deepcopy(report["replay_contract"])
    contract.pop("condition")
    contract["simulation"].pop("MDD")
    return contract


def pairing_errors(reference, mdd):
    errors = []
    for label, report in (("reference", reference), ("mdd", mdd)):
        if report.get("status") != "passed" or report.get("errors"):
            errors.append(f"{label} trajectory did not pass internal checks.")
        contract = report.get("replay_contract", {})
        if canonical_json_sha256(contract) != report.get("replay_contract_sha256"):
            errors.append(f"{label} replay contract hash is inconsistent.")
        if contract.get("condition") != label or contract.get("simulation", {}).get("MDD") != (label == "mdd"):
            errors.append(f"{label} condition/MDD label is incorrect.")
        if not report.get("build_audit") or report["build_audit"].get("errors"):
            errors.append(f"{label} construction audit is absent or failed.")
    try:
        if normalized_pair_contract(reference) != normalized_pair_contract(mdd):
            errors.append("Scientific contracts differ beyond the intended MDD intervention.")
    except KeyError as exc:
        errors.append(f"Incomplete scientific contract: {exc}")
    for key in ("seed_manifest", "structure"):
        if not reference.get(key) or reference.get(key) != mdd.get(key):
            errors.append(f"Paired {key} differ or are missing.")
    for report in (reference, mdd):
        if not report.get("build_audit", {}).get("invariant_sha256"):
            errors.append("Missing actual topology/OU fingerprint.")
    if reference.get("build_audit", {}).get("invariant_sha256") != mdd.get("build_audit", {}).get("invariant_sha256"):
        errors.append("Actual recurrent targets/kinetics or OU setup differ.")
    # The intended intervention must be present, including in tiny test circuits.
    a = reference.get("build_audit", {}).get("expected_parameters")
    b = mdd.get("build_audit", {}).get("expected_parameters")
    if not a or not b or a == b:
        errors.append("Reference and MDD parameter audits do not identify an intervention.")
    return errors


def resource_and_rate_checks(report, *, debug=False):
    errors = []
    windows = [w for w in report["windows"] if w["start_ms"] >= (0 if debug else 8000)]
    seconds = sum(w["stop_ms"] - w["start_ms"] for w in windows) / 1000.
    if seconds <= 0:
        raise ValueError("No post-recovery windows available.")
    rates = {}
    for pop, n in report["build"]["population_counts"].items():
        rate = sum(w["spikes"]["counts"][pop] for w in windows) / (n * seconds)
        rates[pop] = rate
        # Broad engineering plausibility checks, not clinical safety bounds.
        upper = 50. if pop == "HL23PYR" else 100.
        if not debug and not (.01 <= rate <= upper):
            errors.append(f"{pop} post-recovery mean rate {rate:g} Hz outside [0.01,{upper:g}].")
    snapshots = [r for r in report["memory_snapshots"] if r["simulated_ms"] >= (0 if debug else 8000)]
    times = np.asarray([r["simulated_ms"] / 1000 for r in snapshots])
    rss = np.asarray([r["rss_gib"]["sum"] for r in snapshots])
    if times.size < 2 or not np.all(np.isfinite(rss)):
        raise ValueError("Insufficient finite memory checkpoints.")
    growth = float(np.max(rss) - rss[0])
    limit = max(2., .05 * float(rss[0]))
    if not debug and growth > limit:
        errors.append(f"Post-8-s aggregate RSS increased by {growth:.3f} GiB; frozen diagnostic limit {limit:.3f} GiB.")
    return {"mean_rates_hz": rates, "approximate_rss_peak_gib": float(max(s["rss_gib"]["sum"] for s in report["memory_snapshots"])),
            "post_recovery_rss_growth_gib": growth, "rss_growth_limit_gib": limit,
            "post_recovery_rss_slope_gib_per_sim_s": float(np.polyfit(times, rss, 1)[0]),
            "performance": report["performance"], "errors": errors,
            "memory_interpretation": "Aggregate process RSS can double-count shared pages. Check PBS epilogue for authoritative job-level peak; this is no maximum-duration estimate."}


def analyze_pair(directory, *, debug=False):
    directory = Path(directory)
    reports = {c: json.loads((directory/c/REPORT_NAME).read_text()) for c in ("reference", "mdd")}
    errors = pairing_errors(reports["reference"], reports["mdd"])
    run_rows, psd_arrays = {}, {}
    for condition, report in reports.items():
        cfg = report["configuration"]
        if not debug and (not cfg["analysis"]["require_full_network"] or cfg["experiment"]["debug"]):
            errors.append(f"{condition} is a debug run; cannot qualify G1B.")
        if not debug and (report["build"]["population_counts"] != POPULATIONS or report["mpi"]["size"] != 624
                          or report["seed_manifest"]["experiment_seed"] not in PILOT_SEEDS
                          or report["seed_manifest"]["env_seed"] != 0):
            errors.append(f"{condition} population/rank/seed contract does not match G1B.")
        summary = trace_content_summary(directory/condition/TRACE_NAME)
        if summary != report["artifacts"]["trace_summary"]:
            errors.append(f"{condition} saved trace content differs from the completed report.")
        dt = float(report["replay_contract"]["network"]["dt_ms"])
        duration = float(report["completed_simulated_ms"])
        if not debug and (dt != .025 or duration != 28000. or len(report["windows"]) != 28):
            errors.append(f"{condition} duration, dt or window count differs from G1B.")
        expected = int(round(duration/dt))
        if summary["committed_samples"] != expected or summary["committed_windows"] != len(report["windows"]):
            errors.append(f"{condition} committed sample/window count is incorrect.")
        with h5py.File(directory/condition/TRACE_NAME, "r") as trace:
            times = trace["sample_time_ms"][:]
            if times.size != expected or not np.allclose(times, (np.arange(expected)+1)*dt, rtol=0, atol=1e-7):
                errors.append(f"{condition} saved time grid is incomplete.")
            if np.count_nonzero(trace["field_left_boundary_v_per_m"][:]):
                errors.append(f"{condition} saved field was not zero.")
            eeg = trace["eeg_v"][:]
            if eeg.shape != (1, expected) or not np.all(np.isfinite(eeg)):
                errors.append(f"{condition} invalid EEG shape or values.")
            views, spectra = ({}, {}) if debug else spectral_views(eeg, dt)
        for view, values in spectra.items():
            for name, value in values.items():
                psd_arrays[f"{condition}_{view}_{name}"] = value
        resources = resource_and_rate_checks(report, debug=debug)
        errors.extend(f"{condition}: {e}" for e in resources["errors"])
        run_rows[condition] = {"views": views, "resources": resources,
                               "trace_content_sha256": summary["content_sha256"]}
    effects = {}
    if not debug:
        for view in ("legacy", "corrected_sos"):
            a = run_rows["reference"]["views"][view]["band_power_v2"]
            b = run_rows["mdd"]["views"][view]["band_power_v2"]
            log_ratios = {band: float(np.log10(b[band]/a[band])) for band in BANDS}
            effects[view] = {"log10_power_ratio_mdd_over_reference": log_ratios,
                             "equal_weight_log_composite": float(np.mean(list(log_ratios.values())))}
        np.savez_compressed(directory/"g1b_psd.npz", **psd_arrays)
    positive = bool(effects) and all(row["equal_weight_log_composite"] > 0 for row in effects.values())
    result = {"status": "failed" if errors else ("debug_smoke_passed" if debug else ("passed" if positive else "direction_not_reproduced")),
              "technical_passed": not errors, "pilot_pair_gate_passed": bool(not errors and positive and not debug),
              "errors": errors, "debug": debug,
              "seed": reports["reference"]["seed_manifest"]["experiment_seed"],
              "directory": str(directory.resolve()), "runs": run_rows, "effects": effects,
              "structure_sha256": reports["reference"]["structure"]["global_sha256"],
              "limitations": ["Two circuit seeds permit a directional technical pilot, not inferential/clinical conclusions.",
                              "Pilot composite is the unstandardized equal-weight mean of three paired log10 band-power ratios. R1 standardization requires a separate frozen reference cohort.",
                              "Ideal neural-only EEG; no tACS applied. Historical 4-s internal event retained.",
                              "The explicit scientific temperature is 34 C; phenotype preservation must be demonstrated prospectively."]}
    return result


def analyze_suite(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"submission.json").read_text())
    pairs, errors = [], []
    provenance = []
    for job in manifest["pairs"]:
        path = directory/f"seed_{job['seed']}"/"g1b_pair_summary.json"
        if not path.exists():
            errors.append(f"Missing completed pair summary: {path}")
            continue
        row = json.loads(path.read_text())
        if row.get("seed") != job["seed"] or row.get("debug"):
            errors.append(f"Seed/debug mismatch in {path}")
        if not row.get("technical_passed"):
            errors.extend(row.get("errors") or [f"{path} failed technical checks"])
        job_directory = path.parent
        if (job_directory/"worker_exit_code.txt").exists():
            if (job_directory/"worker_exit_code.txt").read_text().strip() != "0":
                errors.append(f"Seed {job['seed']} worker exited unsuccessfully.")
        else:
            errors.append(f"Seed {job['seed']} worker exit status is missing.")
        files = ("git_commit.txt", "mechanism_sha256.txt", "environment_versions.json")
        if all((job_directory/name).exists() for name in files):
            code = (job_directory/"git_commit.txt").read_text().strip()
            if code != manifest["commit"]:
                errors.append(f"Seed {job['seed']} used a different submitted commit.")
            provenance.append({name: (job_directory/name).read_text() for name in files})
        else:
            errors.append(f"Seed {job['seed']} provenance is incomplete.")
        snapshot = job_directory/"qstat_at_exit.txt"
        row["pbs"] = parse_pbs_snapshot(snapshot.read_text()) if snapshot.exists() else {"note": "PBS snapshot missing; inspect the PBS epilogue."}
        row["project"] = job["project"]
        pairs.append(row)
    if len(pairs) != 2 or sorted(row["seed"] for row in pairs) != list(PILOT_SEEDS):
        errors.append("G1B requires exactly the two frozen paired structures.")
    if len(pairs) == 2 and pairs[0]["structure_sha256"] == pairs[1]["structure_sha256"]:
        errors.append("The two candidate structures have identical fingerprints.")
    if len(provenance) == 2 and provenance[0] != provenance[1]:
        errors.append("Pilot jobs used different code, mechanisms, or environments.")
    passed = not errors and all(row["pilot_pair_gate_passed"] for row in pairs)
    return {"status": "failed" if errors else ("passed" if passed else "direction_not_reproduced"),
            "pilot_gate_passed": passed, "errors": errors, "pairs": pairs,
            "next_step": "Freeze pilot, then implement disjoint R1 replication and independent reference calibration." if passed else "Stop before R1; inspect technical errors or the negative phenotype without changing this pilot's endpoints.",
            "submission": manifest}


def parse_pbs_snapshot(text):
    """Summarize last in-job PBS sample; preserve its limited timing precision."""
    values = dict(re.findall(r"^\s*(resources_used\.\w+)\s*=\s*(\S+)", text, flags=re.M))
    output = {"raw": values, "note": "qstat_at_exit is captured just before process exit and may lag; archive the final PBS epilogue."}
    raw = values.get("resources_used.walltime")
    if raw:
        h, m, s = map(int, raw.split(":"))
        hours = h + m/60 + s/3600
        output.update(wall_hours=hours, allocated_core_hours=624*hours,
                      allocated_node_hours=13*hours, estimated_normal_ksu=624*hours*2/1000)
    raw = values.get("resources_used.mem", "")
    match = re.fullmatch(r"([0-9.]+)(b|kb|mb|gb)", raw, flags=re.I)
    if match:
        output["peak_memory_gib"] = float(match[1]) * {"b": 2**-30, "kb": 2**-20, "mb": 2**-10, "gb": 1}[match[2].lower()]
    return output
