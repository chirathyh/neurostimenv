"""Quota-checked, throttled PBS submission. Dry-run by default; login-safe stdlib."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.l23net_analysis.r1_protocol import (
    PAIR_RESERVATION_KSU, PROTOCOL, jobs_for_stage, save_json, sha256, verify_g1b,
)

PROJECTS = ("sj53", "fa32", "ny83")
CORE_FILES = ("reference_target.json", "primary_frozen.json", "core_data_frozen.json")


def parse_balance(text):
    matches = re.findall(r"\bAvail(?:able)?\s*:\s*([\d,]+(?:\.\d+)?)\s*(KSU|MSU|SU)\b", text, re.I)
    if len(matches) != 1:
        raise ValueError("Cannot uniquely parse nci_account Avail; use --available-ksu with freshly checked balances.")
    amount, unit = matches[0]
    return float(amount.replace(",", "")) * {"ksu": 1., "msu": 1000., "su": .001}[unit.lower()]


def allocate_projects(jobs, balances, reserve_ksu=1.):
    """Reserve maximum walltime, not an optimistic expected runtime."""
    jobs = json.loads(json.dumps(jobs))
    index = 0
    for project in PROJECTS:
        capacity = max(0, math.floor((balances[project]-reserve_ksu+1e-10)/PAIR_RESERVATION_KSU))
        for _ in range(min(capacity, len(jobs)-index)):
            jobs[index]["project"] = project
            index += 1
    if index != len(jobs):
        raise ValueError(f"Insufficient currently available quota for full requested-walltime reservations: {index}/{len(jobs)} jobs fit. No jobs submitted.")
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "extension"), required=True)
    parser.add_argument("--g1b", type=Path, default=ROOT/"results/l23net_g1b_20260928_094057_kCtCZq/g1b_summary.json")
    parser.add_argument("--core", type=Path, help="Completed R1 core suite; required for extension")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--available-ksu", nargs=3, metavar="PROJECT=KSU", help="Manual fallback only; use freshly checked Avail balances, not original allocation")
    parser.add_argument("--submit", action="store_true", help="Actually qsub; without this print the plan only")
    args = parser.parse_args(argv)
    if not 1 <= args.max_concurrent <= 8:
        raise ValueError("Use between one and eight concurrent 624-rank workers")
    verify_g1b(args.g1b)
    core_hashes = {}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.stage == "extension":
        if args.core is None:
            raise ValueError("Extension requires --core pointing to the completed core suite")
        data = json.loads((args.core/"core_data_frozen.json").read_text())
        core_manifest = json.loads((args.core/"submission.json").read_text())
        if not data["technical_passed"] or core_manifest["commit"] != commit:
            raise ValueError("Core must be technically complete and use the same code commit")
        if data["protocol"] != json.loads(json.dumps(PROTOCOL)):
            raise ValueError("Core protocol differs")
        core_hashes = {f: sha256(args.core/f) for f in CORE_FILES}
        # A negative primary result is retained; extension is not p-value-dependent.
    accounts = {}
    if args.available_ksu:
        balances = {k: float(v) for k, v in (item.split("=", 1) for item in args.available_ksu)}
        if set(balances) != set(PROJECTS) or any(not math.isfinite(v) or v < 0 for v in balances.values()):
            raise ValueError("Supply nonnegative balances for exactly sj53, fa32 and ny83")
    else:
        for project in PROJECTS:
            accounts[project] = subprocess.check_output(["nci_account", "-P", project], text=True, stderr=subprocess.STDOUT)
        balances = {p: parse_balance(t) for p, t in accounts.items()}
    jobs = allocate_projects(jobs_for_stage(args.stage), balances)
    plan = {p: {"jobs": sum(j["project"] == p for j in jobs), "available_ksu": balances[p],
                "maximum_worker_reservation_ksu": round(sum(j["project"] == p for j in jobs)*PAIR_RESERVATION_KSU, 3)} for p in PROJECTS}
    print(json.dumps({"stage": args.stage, "workers": len(jobs), "trajectories": 2*len(jobs),
                      "max_concurrent": args.max_concurrent, "projects": plan,
                      "resources_per_worker": "624 CPUs / 256GB / normal / 02:30:00",
                      "note": "1 KSU per project retained, including small summary job. Queue start/deadline not guaranteed."}, indent=2), flush=True)
    if not args.submit:
        print("DRY RUN: no jobs or suite directory created. Add --submit to execute this plan.")
        return 0
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        subprocess.run(command, cwd=ROOT, check=True)
    if not shutil.which("qsub"):
        raise ValueError("qsub unavailable; submit from Gadi login")
    (ROOT/"results").mkdir(exist_ok=True)
    suite = Path(tempfile.mkdtemp(prefix=f"l23net_r1_{args.stage}_{datetime.now():%Y%m%d_%H%M%S}_", dir=ROOT/"results"))
    shutil.copy2(args.g1b, suite/"g1b_prerequisite.json")
    manifest = {"stage": args.stage, "commit": commit, "protocol": PROTOCOL, "jobs": jobs,
                "max_concurrent": args.max_concurrent, "quota_plan": plan, "account_snapshots": accounts,
                "core_directory": str(args.core.resolve()) if args.core else None, "core_hashes": core_hashes,
                "g1b_sha256": sha256(args.g1b), "submission_status": "submitting"}
    path = suite/"submission.json"
    save_json(path, manifest)
    print(f"SUITE_DIRECTORY={suite}", flush=True)
    scripts = ROOT/"experiments/l23net_analysis/nci"
    try:
        for index, job in enumerate(jobs):
            folder = suite/job["name"]
            folder.mkdir()
            environment = f"EXPECTED_COMMIT={commit},SUITE_DIRECTORY={suite},JOB_INDEX={index}"
            command = ["qsub", "-P", job["project"], "-N", f"r1_{args.stage[0]}{index:02d}", "-v", environment,
                       "-o", str(folder/"pbs.out"), "-e", str(folder/"pbs.err")]
            if index >= args.max_concurrent:
                predecessor = jobs[index-args.max_concurrent]["job_id"]
                command += ["-W", f"depend=afterany:{predecessor}"]
            job["submission_attempted"] = True
            save_json(path, manifest)
            command.append(str(scripts/"run_l23net_r1_worker.sh"))
            job["job_id"] = subprocess.check_output(command, cwd=ROOT, text=True).strip()
            if not re.fullmatch(r"\d+(?:\.[\w.-]+)?", job["job_id"]):
                raise ValueError("Unexpected qsub response; inspect before retrying")
            save_json(path, manifest)
            print(f"{job['name']} -> {job['project']}: {job['job_id']}", flush=True)
        dependencies = ":".join(j["job_id"] for j in jobs)
        command = ["qsub", "-P", jobs[0]["project"], "-v", f"EXPECTED_COMMIT={commit},SUITE_DIRECTORY={suite}",
                   "-W", f"depend=afterany:{dependencies}", "-o", str(suite/"summary_pbs.out"),
                   "-e", str(suite/"summary_pbs.err"), str(scripts/"run_l23net_r1_summary.sh")]
        manifest["summary_job_id"] = subprocess.check_output(command, cwd=ROOT, text=True).strip()
        manifest["submission_status"] = "submitted"
    except Exception as exc:
        manifest["submission_status"] = "partial_failure"
        manifest["submission_error"] = repr(exc)
        print(f"Partial submission: inspect {path}; do NOT blindly resubmit the suite.", file=sys.stderr)
        raise
    finally:
        save_json(path, manifest)
    print(f"Summary job: {manifest['summary_job_id']}")
    print(f"Status: python3 {scripts}/collect_l23net_r1_status.py '{suite}'")
    print("Keep this checkout/environment unchanged until all jobs finish; copy the entire suite afterwards.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"R1 submission stopped: {exc}")
