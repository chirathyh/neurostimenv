"""Summarize L23Net NCI scaling and duration-profile results.

This utility combines the profiler JSON with ``qstat_at_exit.txt``.  The PBS
job-level memory peak and CPU time are preferred for allocation decisions;
summed process RSS remains useful for within-run drift diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable


REPORT_NAME = "l23net_tacs_full_scale_profile.json"


def parse_pbs_duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    fields = value.strip().split(":")
    if len(fields) != 3:
        raise ValueError(f"Unsupported PBS duration: {value!r}")
    hours, minutes, seconds = (float(field) for field in fields)
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_pbs_memory_gib(value: str | None) -> float | None:
    if not value:
        return None
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?b)\s*",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Unsupported PBS memory value: {value!r}")
    number = float(match.group(1))
    unit = match.group(2).lower()
    scale = {
        "b": 1.0,
        "kb": 1024.0,
        "mb": 1024.0**2,
        "gb": 1024.0**3,
        "tb": 1024.0**4,
    }[unit]
    return number * scale / 1024.0**3


def parse_qstat(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    pattern = re.compile(r"^\s*([^=]+?)\s*=\s*(.*?)\s*$")
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.match(line)
            if match is not None:
                values[match.group(1).strip()] = match.group(2).strip()
    return values


def _result_report_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    supplied = list(inputs)
    if not supplied:
        supplied = ["results/l23net_tacs_scaling_*", "results/l23net_tacs_duration15s_*"]
    for raw in supplied:
        matches = sorted(Path().glob(raw)) if any(c in raw for c in "*?[") else [Path(raw)]
        for candidate in matches:
            report = candidate / REPORT_NAME if candidate.is_dir() else candidate
            if report.name == REPORT_NAME and report.is_file():
                paths.append(report.resolve())
    return sorted(set(paths))


def summarize_report(report_path: Path) -> dict[str, Any]:
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    config = report["configuration"]
    resource = config["analysis"]["resource_request"]
    performance = report.get("performance", {})
    trend = performance.get("aggregate_rss_trend", {})
    qstat = parse_qstat(report_path.parent / "qstat_at_exit.txt")

    pbs_wall_s = parse_pbs_duration_seconds(qstat.get("resources_used.walltime"))
    pbs_cpu_s = parse_pbs_duration_seconds(qstat.get("resources_used.cput"))
    pbs_memory_gib = parse_pbs_memory_gib(qstat.get("resources_used.mem"))
    ncpus = int(resource["ncpus"])
    simulated_s = float(performance.get("simulated_s", 0.0))
    effective_cores = (
        pbs_cpu_s / pbs_wall_s
        if pbs_cpu_s is not None and pbs_wall_s and pbs_wall_s > 0.0
        else None
    )
    allocation_utilization = (
        effective_cores / ncpus
        if effective_cores is not None and ncpus > 0
        else None
    )
    allocation_core_hours = (
        ncpus * pbs_wall_s / 3600.0
        if pbs_wall_s is not None
        else None
    )

    stages = performance.get("stage_performance", {}).get("by_stage", {})
    return {
        "result_directory": str(report_path.parent),
        "status": report.get("status"),
        "mpi_ranks": int(report.get("mpi", {}).get("size", resource["mpi_ranks"])),
        "ncpus": ncpus,
        "requested_memory_gib": float(resource["memory_gb"]),
        "simulated_s": simulated_s,
        "build_wall_s": performance.get("build_wall_s"),
        "integration_wall_s": performance.get("integration_wall_s"),
        "simulated_s_per_wall_hour": performance.get("simulated_s_per_wall_hour"),
        "pbs_wall_s": pbs_wall_s,
        "pbs_cpu_s": pbs_cpu_s,
        "pbs_average_effective_cores": effective_cores,
        "pbs_average_allocated_cpu_utilization_fraction": allocation_utilization,
        "pbs_peak_memory_gib": pbs_memory_gib,
        "allocation_core_hours": allocation_core_hours,
        "allocation_core_hours_per_simulated_s": (
            allocation_core_hours / simulated_s
            if allocation_core_hours is not None and simulated_s > 0.0
            else None
        ),
        "approximate_peak_aggregate_rss_gib": trend.get(
            "all_checkpoint_peak_gib"
        ),
        "post_warmup_rss_slope_gib_per_simulated_s": trend.get(
            "slope_gib_per_simulated_s"
        ),
        "stimulation_to_inactive_wall_ratio": performance.get(
            "stage_performance", {}
        ).get("stimulation_to_inactive_median_window_wall_ratio"),
        "baseline_total_wall_s": stages.get("baseline", {}).get("total_wall_s"),
        "stimulation_total_wall_s": stages.get("stimulation", {}).get(
            "total_wall_s"
        ),
        "washout_total_wall_s": stages.get("washout", {}).get("total_wall_s"),
        "report": str(report_path),
    }


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("ranks", "mpi_ranks"),
        ("CPUs", "ncpus"),
        ("status", "status"),
        ("sim s/h", "simulated_s_per_wall_hour"),
        ("integration s", "integration_wall_s"),
        ("effective cores", "pbs_average_effective_cores"),
        ("CPU util", "pbs_average_allocated_cpu_utilization_fraction"),
        ("PBS peak GiB", "pbs_peak_memory_gib"),
        ("core-h/sim-s", "allocation_core_hours_per_simulated_s"),
        ("RSS slope GiB/sim-s", "post_warmup_rss_slope_gib_per_simulated_s"),
        ("active/inactive", "stimulation_to_inactive_wall_ratio"),
    ]
    print("| " + " | ".join(label for label, _ in columns) + " |")
    print("|" + "|".join("---:" for _ in columns) + "|")
    for row in rows:
        rendered = []
        for _label, key in columns:
            value = row.get(key)
            if key == "pbs_average_allocated_cpu_utilization_fraction" and value is not None:
                rendered.append(f"{100.0 * value:.1f}%")
            else:
                rendered.append(_format(value))
        print("| " + " | ".join(rendered) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        nargs="*",
        help="Result directories, report JSON paths, or glob patterns.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Optional path prefix for .json and .csv summary files.",
    )
    args = parser.parse_args()

    reports = _result_report_paths(args.results)
    if not reports:
        raise SystemExit("No L23Net profile reports were found.")
    rows = sorted(
        [summarize_report(path) for path in reports],
        key=lambda row: (row["mpi_ranks"], row["result_directory"]),
    )
    print_markdown(rows)

    if args.output_prefix is not None:
        prefix = args.output_prefix.resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with prefix.with_suffix(".json").open("w", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2, sort_keys=True)
            stream.write("\n")
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
