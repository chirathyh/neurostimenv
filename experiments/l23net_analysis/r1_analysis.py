"""R1 raw-data audits and frozen, structure-level statistics (no NEURON import)."""
import copy
import json
from pathlib import Path
import re

import h5py
import numpy as np
from scipy import stats

from experiments.l23net_analysis.g1b_analysis import (
    BANDS, POPULATIONS, pairing_errors, resource_and_rate_checks, spectral_views,
)
from experiments.l23net_analysis.replay_validation import canonical_json_sha256, trace_content_summary
from experiments.l23net_analysis.r1_protocol import COHORTS, PROTOCOL, REPORT_NAME, TRACE_NAME, save_json, sha256


def audit_run(directory, seed, condition, cohort, *, debug=False):
    """Re-read the actual HDF5, not just a cached success flag."""
    directory = Path(directory)
    report = json.loads((directory / REPORT_NAME).read_text())
    contract = report["replay_contract"]
    if report["status"] != "passed" or report["errors"] or report["build_audit"]["errors"]:
        raise ValueError(f"{directory}: trajectory/build did not pass")
    if canonical_json_sha256(contract) != report["replay_contract_sha256"]:
        raise ValueError("Contract hash mismatch")
    if (seed not in COHORTS[cohort] or contract["experiment_seed"] != seed
            or report["configuration"]["analysis"]["cohort"] != cohort
            or contract["condition"] != condition
            or contract["simulation"]["MDD"] != (condition == "mdd")
            or (cohort == "calibration" and condition != "reference")):
        raise ValueError("Seed/cohort/condition mismatch")
    if not debug:
        if (contract["debug"] or contract["mpi_ranks"] != 624 or contract["env_seed"] != 0
                or report["build"]["population_counts"] != POPULATIONS
                or not report["configuration"]["analysis"]["require_full_network"]):
            raise ValueError("Not a full production R1 trajectory")
        network = {"celsius": 34., "dt_ms": .025, "syn_activity": True, "tstart_ms": 0., "v_init_mV": -80.}
        if contract["network"] != network or contract["simulation"] != {
                "DRUG": False, "MDD": condition == "mdd", "duration_ms": 28000., "window_ms": 1000.}:
            raise ValueError("Production parameters differ from frozen R1")
    if contract["stimulation_enabled"] or contract["online"]["temperature_mode"] != "configured":
        raise ValueError("Wrong field or temperature mode")
    duration = float(report["completed_simulated_ms"])
    dt = float(contract["network"]["dt_ms"])
    count = int(round(duration / dt))
    windows = report["windows"]
    if duration != contract["simulation"]["duration_ms"] or not windows:
        raise ValueError("Incomplete trajectory")
    for i, window in enumerate(windows):
        width = contract["simulation"]["window_ms"]
        if (window["errors"] or window["index"] != i or window["start_ms"] != i*width
                or window["stop_ms"] != (i+1)*width or window["sample_count"] != round(width/dt)):
            raise ValueError("Window sequence is incomplete or failed checks")
    if windows[-1]["stop_ms"] != duration:
        raise ValueError("Final window boundary mismatch")
    summary = trace_content_summary(directory / TRACE_NAME)
    if (summary != report["artifacts"]["trace_summary"] or summary["committed_samples"] != count
            or summary["committed_windows"] != len(windows)):
        raise ValueError("Saved trace hash or committed extent mismatch")
    with h5py.File(directory / TRACE_NAME, "r") as trace:
        for name in trace:
            if np.issubdtype(trace[name].dtype, np.number) and not np.isfinite(trace[name][:]).all():
                raise ValueError(f"Nonfinite saved {name}")
        times = trace["sample_time_ms"][:]
        if times.size != count or not np.allclose(times, (np.arange(count)+1)*dt, rtol=0, atol=1e-7):
            raise ValueError("Wrong saved time grid")
        if (np.count_nonzero(trace["field_left_boundary_v_per_m"][:])
                or trace["eeg_v"].shape != (1, count) or trace["dipole_nA_um"].shape != (3, count)):
            raise ValueError("Wrong field or forward-model output dimensions")
        views, spectra = ({}, {}) if debug else spectral_views(trace["eeg_v"][:], dt)
    resources = resource_and_rate_checks(report, debug=debug)
    if resources["errors"]:
        raise ValueError("; ".join(resources["errors"]))
    row = {"seed": seed, "condition": condition, "cohort": cohort,
           "report_sha256": sha256(directory / REPORT_NAME), "trace_content_sha256": summary["content_sha256"],
           "structure_sha256": report["structure"]["global_sha256"], "views": views, "resources": resources}
    return row, report, spectra


def frozen_json(path, value):
    """Idempotent recomputation is allowed; a different frozen result is not."""
    path = Path(path)
    if path.exists():
        if canonical_json_sha256(json.loads(path.read_text())) != canonical_json_sha256(value):
            raise ValueError(f"Refusing to replace a different frozen result: {path}")
    else:
        save_json(path, value)


def reference_target(rows):
    if sorted(r["seed"] for r in rows) != list(COHORTS["calibration"]):
        raise ValueError("All 16 independent reference structures are required")
    if any(r["condition"] != "reference" or r["cohort"] != "calibration" for r in rows):
        raise ValueError("Target calibration may use only the reference cohort")
    target = {"protocol": PROTOCOL, "sources": [{k: r[k] for k in
              ("seed", "report_sha256", "trace_content_sha256")} for r in rows], "views": {}}
    for view in ("legacy", "corrected_sos"):
        x = np.array([[np.log10(r["views"][view]["band_power_v2"][b]) for b in BANDS] for r in rows])
        sd = np.std(x, axis=0, ddof=1)
        floor = max(.01, .1 * float(np.median(sd)))
        target["views"][view] = {"bands": list(BANDS), "mean_log10_power_v2": np.mean(x, axis=0).tolist(),
                                  "sample_sd_log10": sd.tolist(), "scale_floor_log10": floor,
                                  "scale_log10": np.maximum(sd, floor).tolist()}
    return target


def paired_inference(values, *, rng_seed=982017):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite independent paired effects")
    n, mean, sd = x.size, float(x.mean()), float(x.std(ddof=1))
    rng = np.random.default_rng(rng_seed)
    exact = n <= 20
    draws = 2**n if exact else PROTOCOL["monte_carlo_sign_flips"]
    exceed = 0
    for first in range(0, draws, 4096):
        size = min(4096, draws-first)
        if exact:
            signs = 2*((np.arange(first, first+size, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)-1
        else:
            signs = 2*rng.integers(0, 2, size=(size, n))-1
        exceed += int(np.count_nonzero((signs*x).mean(axis=1) >= mean - 1e-14*max(1., abs(mean))))
    p = exceed/draws if exact else (exceed+1)/(draws+1)
    bootstrap = x[rng.integers(0, n, size=(PROTOCOL["bootstrap_replicates"], n))].mean(axis=1)
    half = float(stats.t.ppf(.975, n-1) * sd / np.sqrt(n))
    return {"n_structures": int(n), "mean": mean, "sample_sd": sd,
            "paired_dz": mean/sd if sd > 0 else None, "positive_structures": int(np.sum(x > 0)),
            "p_one_sided": p, "permutation_method": "exact_sign_flip" if exact else "monte_carlo_sign_flip_plus_one",
            "permutation_draws": draws, "mean_t_ci95": [mean-half, mean+half],
            "mean_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "uncertainty_scope": "Structure variability conditional on the frozen reference calibration; no clinical uncertainty."}


def bh_adjust(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order]*p.size/np.arange(1, p.size+1))[::-1])[::-1]
    output = np.empty(p.size)
    output[order] = np.minimum(adjusted, 1.)
    return output.tolist()


def cohort_inference(rows, target, seeds):
    if sorted({r["seed"] for r in rows}) != sorted(seeds) or len(rows) != 2*len(seeds):
        raise ValueError("Paired cohort is incomplete or duplicated")
    by_key = {(r["seed"], r["condition"]): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate trajectory")
    result = {"seeds": list(seeds), "target_sha256": canonical_json_sha256(target), "views": {}}
    for view in ("legacy", "corrected_sos"):
        differences = np.array([[np.log10(by_key[s, "mdd"]["views"][view]["band_power_v2"][b] /
                                          by_key[s, "reference"]["views"][view]["band_power_v2"][b])
                                 for b in BANDS] for s in seeds])
        scale = np.asarray(target["views"][view]["scale_log10"])
        composite = (differences/scale).mean(axis=1)
        bands = {b: paired_inference(differences[:, i]) for i, b in enumerate(BANDS)}
        for b, q in zip(BANDS, bh_adjust([r["p_one_sided"] for r in bands.values()])):
            bands[b]["q_bh_three_bands"] = q
        result["views"][view] = {"composite": paired_inference(composite), "bands": bands,
                                  "paired_log10_ratios": differences.tolist(), "composite_by_structure": composite.tolist()}
        # Secondary observability audit: never split the two conditions of a
        # structure across training and testing. No fitted feature selection.
        mu = np.asarray(target["views"][view]["mean_log10_power_v2"])
        scores = np.array([[(np.array([np.log10(by_key[s, c]["views"][view]["band_power_v2"][b])
                                       for b in BANDS])-mu)/scale for c in ("reference", "mdd")]
                           for s in seeds]).mean(axis=2)
        correct, thresholds = [], []
        for i in range(len(seeds)):
            threshold = float(np.delete(scores, i, axis=0).mean())
            thresholds.append(threshold)
            correct.append((int(scores[i, 0] <= threshold)+int(scores[i, 1] > threshold))/2)
        rng = np.random.default_rng(PROTOCOL["analysis_random_seed"])
        accuracy_boot = np.asarray(correct)[rng.integers(0, len(seeds), size=(20000, len(seeds)))].mean(axis=1)
        result["views"][view]["held_out_discriminability_secondary"] = {
            "method": "Leave-one-structure-out midpoint threshold on frozen equal-weight standardized three-band score; higher is MDD.",
            "balanced_accuracy": float(np.mean(correct)), "accuracy_by_structure": correct,
            "threshold_by_held_out_structure": thresholds,
            "structure_bootstrap_ci95": np.quantile(accuracy_boot, [.025, .975]).tolist(),
            "limitation": "Secondary simulator observability audit; threshold-training uncertainty not included in bootstrap, not clinical diagnosis."}
    primary = result["views"]["legacy"]["composite"]
    result["phenotype_confirmed"] = bool(primary["mean"] > 0 and primary["p_one_sided"] <= .05
                                          and result["views"]["corrected_sos"]["composite"]["mean"] > 0)
    return result


def pbs_epilogue(path):
    """Use final PBS accounting when present; never silently call a snapshot final."""
    text = Path(path).read_text() if Path(path).exists() else ""
    matches = re.findall(r"Service Units:\s*([\d.]+)", text)
    walls = re.findall(r"Walltime Used:\s*(\d+:\d+:\d+)", text)
    memory = re.findall(r"Memory Used:\s*([\d.]+)GB", text)
    exits = re.findall(r"Exit Status:\s*(-?\d+)", text)
    if not (matches and walls and memory and exits):
        return {"final_accounting_available": False, "note": "PBS epilogue not yet copied/written; re-analyze after job exit."}
    h, m, s = map(int, walls[-1].split(":"))
    return {"final_accounting_available": True, "service_units": float(matches[-1]),
            "wall_hours": h+m/60+s/3600, "memory_gb_pbs": float(memory[-1]), "exit_status": int(exits[-1])}


def analyze_suite(directory, *, core_override=None, output_directory=None):
    root = Path(directory)
    output = Path(output_directory) if output_directory is not None else root
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/"submission.json").read_text())
    if canonical_json_sha256(manifest["protocol"]) != canonical_json_sha256(PROTOCOL):
        raise ValueError("Submission protocol differs from this analysis")
    rows, errors, provenance, structures, spectra = [], [], [], {}, {}
    pbs_rows = []
    normalized_contract = None
    expected_cohorts = ("calibration", "primary") if manifest["stage"] == "core" else ("extension",)
    for job in manifest["jobs"]:
        folder = root/job["name"]
        try:
            if (folder/"worker_exit_code.txt").read_text().strip() != "0":
                raise ValueError("Worker exited unsuccessfully")
            if (folder/"git_commit.txt").read_text().strip() != manifest["commit"]:
                raise ValueError("Worker code identity differs from submission")
            provenance.append([(folder/f).read_text() for f in ("git_commit.txt", "mechanism_sha256.txt", "environment_versions.json")])
            reports = {}
            for run in job["runs"]:
                key = f"seed_{run['seed']}/{run['condition']}"
                row, report, psd = audit_run(folder/key, run["seed"], run["condition"], job["cohort"])
                reports[run["condition"]] = report
                c = copy.deepcopy(report["replay_contract"])
                for k in ("experiment_seed", "condition"):
                    c.pop(k)
                c["simulation"].pop("MDD")
                if normalized_contract is None:
                    normalized_contract = c
                if c != normalized_contract:
                    raise ValueError("Contracts differ between structures beyond seed/MDD")
                structures.setdefault(row["seed"], set()).add(row["structure_sha256"])
                rows.append(row)
                for view, arrays in psd.items():
                    for name, values in arrays.items():
                        spectra[f"s{row['seed']}_{row['condition']}_{view}_{name}"] = values
            if job["cohort"] != "calibration":
                pair_errors = pairing_errors(reports["reference"], reports["mdd"])
                if pair_errors:
                    raise ValueError("; ".join(pair_errors))
            accounting = pbs_epilogue(folder/"pbs.out")
            if accounting.get("exit_status", 0) != 0:
                raise ValueError("Nonzero final PBS exit status")
            pbs_rows.append({"job": job["name"], "project": job["project"], **accounting})
        except Exception as exc:
            errors.append(f"{job['name']}: {type(exc).__name__}: {exc}")
    expected = {(s, c, cohort) for cohort in expected_cohorts for s in COHORTS[cohort]
                for c in (("reference",) if cohort == "calibration" else ("reference", "mdd"))}
    actual = {(r["seed"], r["condition"], r["cohort"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        errors.append("Missing, unexpected or duplicate trajectories; no partial-cohort inference permitted")
    if provenance and any(p != provenance[0] for p in provenance):
        errors.append("Workers have different code/mechanisms/environments")
    if any(len(v) != 1 for v in structures.values()) or len({next(iter(v)) for v in structures.values()}) != len(structures):
        errors.append("Pair structures mismatch or independent structures have duplicate fingerprints")
    result = {"status": "failed", "technical_passed": not errors, "errors": errors,
              "stage": manifest["stage"], "rows": rows, "pbs": pbs_rows,
              "interpretation": "Ideal EEG, no stimulation. Structure is the unit; primary 16-pair inference is never replaced by the extension."}
    if errors:
        return result
    if manifest["stage"] == "core":
        target = reference_target([r for r in rows if r["cohort"] == "calibration"])
        frozen_json(output/"reference_target.json", target)  # Freeze before reading candidate effects.
        primary = cohort_inference([r for r in rows if r["cohort"] == "primary"], target, COHORTS["primary"])
        frozen_json(output/"primary_frozen.json", primary)
        frozen_json(output/"core_data_frozen.json", {"rows": rows, "contract": normalized_contract,
                    "provenance": provenance[0], "technical_passed": True, "protocol": PROTOCOL})
        result["primary"] = primary
        result["status"] = "passed" if primary["phenotype_confirmed"] else "phenotype_not_confirmed"
    else:
        core = Path(core_override or manifest["core_directory"])
        for name, digest in manifest["core_hashes"].items():
            if sha256(core/name) != digest:
                raise ValueError(f"Frozen upstream file changed: {name}")
        core_result = json.loads((core/"core_data_frozen.json").read_text())
        if not core_result["technical_passed"]:
            raise ValueError("Core technical gate did not pass")
        if normalized_contract != core_result["contract"] or provenance[0] != core_result["provenance"]:
            raise ValueError("Extension code/environment/scientific contract differs from frozen core")
        target = json.loads((core/"reference_target.json").read_text())
        result["primary_frozen"] = json.loads((core/"primary_frozen.json").read_text())
        result["extension_44"] = cohort_inference(rows, target, COHORTS["extension"])
        original = [r for r in core_result["rows"] if r["cohort"] == "primary"]
        old_structures = {r["structure_sha256"] for r in core_result["rows"]}
        if any(r["structure_sha256"] in old_structures for r in rows):
            raise ValueError("Extension structure duplicates a core structure")
        result["pooled_60_secondary"] = cohort_inference(original+rows, target, COHORTS["primary"]+COHORTS["extension"])
        result["status"] = "completed_extension"
    np.savez_compressed(output/"r1_psd.npz", **spectra)
    result["reference_target_sha256"] = canonical_json_sha256(target)
    return result


def summary_markdown(result):
    lines = ["# L23Net R1", "", f"Status: {result['status']}", "", f"Technical pass: {result['technical_passed']}", ""]
    for error in result["errors"]:
        lines.append(f"- {error}")
    for name in ("primary", "primary_frozen", "extension_44", "pooled_60_secondary"):
        if name not in result:
            continue
        r = result[name]["views"]["legacy"]["composite"]
        lines += [f"## {name}", "", f"n={r['n_structures']}; mean standardized composite={r['mean']:.6g}; "
                  f"one-sided p={r['p_one_sided']:.6g}; paired dz={r['paired_dz']}; positive={r['positive_structures']}.",
                  f"Two-sided 95% t CI: {r['mean_t_ci95']}; structure-bootstrap CI: {r['mean_bootstrap_ci95']}.", ""]
    lines += ["PBS final accounting (missing epilogues are not final zeros):", "",
              "| Job | Project | Final SU | Wall hours | Peak GB |", "|---|---|---:|---:|---:|"]
    for row in result.get("pbs", []):
        lines.append(f"| {row['job']} | {row['project']} | {row.get('service_units', 'pending')} | {row.get('wall_hours', 'pending')} | {row.get('memory_gb_pbs', 'pending')} |")
    lines += ["", result["interpretation"], "", "No clinical, tACS efficacy, or maximum-simulation-duration claim follows from R1.", ""]
    return "\n".join(lines)
