"""Analyze a completed pair or aggregate both G1B pilot pairs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.l23net_analysis.g1b_analysis import analyze_pair, analyze_suite


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pair", type=Path)
    mode.add_argument("--suite", type=Path)
    parser.add_argument("--debug-smoke", action="store_true")
    args = parser.parse_args()
    directory = args.pair or args.suite
    output = directory/("g1b_pair_summary.json" if args.pair else "g1b_summary.json")
    try:
        if args.suite and args.debug_smoke:
            raise ValueError("Debug artifacts cannot qualify a G1B suite.")
        result = analyze_pair(directory, debug=args.debug_smoke) if args.pair else analyze_suite(directory)
    except Exception as exc:
        result = {"status": "failed", "errors": [f"{type(exc).__name__}: {exc}"], "pilot_gate_passed": False}
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(output)
    pairs = [result] if args.pair else result.get("pairs", [])
    lines = ["# G1B pilot", "", f"Status: **{result['status']}**", "",
             "| Seed | Technical | Legacy composite | SOS composite | Pair gate |", "|---|---|---|---|---|"]
    for row in pairs:
        effects = row.get("effects", {})
        lines.append(f"| {row.get('seed')} | {row.get('technical_passed')} | {effects.get('legacy', {}).get('equal_weight_log_composite')} | {effects.get('corrected_sos', {}).get('equal_weight_log_composite')} | {row.get('pilot_pair_gate_passed')} |")
    if args.suite:
        lines += ["", "| Seed | Project | PBS wall hours | PBS peak GiB | Estimated KSU |", "|---|---|---|---|---|"]
        for row in pairs:
            pbs = row.get("pbs", {})
            lines.append(f"| {row.get('seed')} | {row.get('project')} | {pbs.get('wall_hours')} | {pbs.get('peak_memory_gib')} | {pbs.get('estimated_normal_ksu')} |")
    lines += ["", "Two independent structures; no significance/clinical claim. Positive means higher MDD-configured three-band power.", "", *result.get("errors", [])]
    output.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)
    print(f"Report: {output.resolve()}", flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
