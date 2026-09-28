"""Frozen R1 design; standard-library-only so submission works on Gadi login."""
import hashlib
import json
from pathlib import Path

COHORTS = {
    "calibration": tuple(range(8101, 8117)),
    "primary": tuple(range(8201, 8217)),
    "extension": tuple(range(8301, 8345)),
}
REPORT_NAME = "l23net_r1_run.json"
TRACE_NAME = "l23net_r1_trace.h5"
G1B_SHA256 = "5ceb776cd279b9e7d1743182854d194bd29476e34b2ceb77f395ac2a01c74dc8"
PAIR_RESERVATION_KSU = 3.12
PROTOCOL = {
    "version": 1, "cohorts": COHORTS, "duration_ms": 28000, "dt_ms": .025,
    "window_ms": 1000, "mpi_ranks": 624, "celsius": 34, "env_seed": 0,
    "primary_view": "legacy", "sensitivity_view": "corrected_sos",
    "bands": ["theta", "alpha", "low_beta"], "alpha_one_sided": .05,
    "scale_floor_rule": "max(0.01, 0.1 * median(reference_sample_sd_log10))",
    "bootstrap_replicates": 20000, "monte_carlo_sign_flips": 100000,
    "analysis_random_seed": 982017,
    "extension_rule": "After technically complete core and frozen primary analysis, regardless of primary significance; no replacement of primary inference.",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def jobs_for_stage(stage):
    if stage == "core":
        seeds = COHORTS["calibration"]
        jobs = [{"name": f"calibration_{i//2+1:02d}", "cohort": "calibration",
                 "runs": [{"seed": s, "condition": "reference"} for s in seeds[i:i+2]]}
                for i in range(0, len(seeds), 2)]
        pair_cohort = "primary"
    elif stage == "extension":
        jobs, pair_cohort = [], "extension"
    else:
        raise ValueError("Stage must be core or extension.")
    jobs += [{"name": f"{pair_cohort}_{s}", "cohort": pair_cohort,
              "runs": [{"seed": s, "condition": c} for c in ("reference", "mdd")]}
             for s in COHORTS[pair_cohort]]
    return jobs


def verify_g1b(path):
    if sha256(path) != G1B_SHA256:
        raise ValueError("G1B summary differs from the raw-trace-audited, frozen pilot.")
    report = json.loads(Path(path).read_text())
    if report["status"] != "passed" or report["errors"] or not report["pilot_gate_passed"]:
        raise ValueError("G1B prerequisite failed.")
    return report
