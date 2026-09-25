"""Content hashing and comparison for independent L23Net replay runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


REPORT_NAME = "l23net_no_field_replay.json"
TRACE_NAME = "l23net_no_field_trace.h5"
NUMERIC_TRACE_DATASETS = (
    "sample_time_ms",
    "eeg_v",
    "dipole_nA_um",
    "field_left_boundary_time_ms",
    "field_left_boundary_v_per_m",
    "stage_code",
)


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using an unambiguous canonical encoding."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dataset_chunks(dataset: h5py.Dataset) -> Iterable[np.ndarray]:
    if dataset.chunks is None:
        yield np.asarray(dataset[...])
        return
    for selection in dataset.iter_chunks():
        yield np.asarray(dataset[selection])


def dataset_sha256(dataset: h5py.Dataset) -> str:
    """Hash dtype, shape, and values without loading a full trace into RAM."""
    digest = hashlib.sha256()
    digest.update(str(dataset.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(dataset.shape)).encode("ascii"))
    digest.update(b"\0")
    for chunk in _dataset_chunks(dataset):
        digest.update(np.ascontiguousarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def trace_content_summary(path: str | Path) -> dict[str, Any]:
    """Return deterministic hashes and committed sizes for a streamed trace."""
    path = Path(path)
    with h5py.File(path, "r") as trace:
        missing = [name for name in NUMERIC_TRACE_DATASETS if name not in trace]
        if missing:
            raise ValueError(f"Trace {path} is missing datasets: {missing}.")
        stage_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in trace["stage_names"][:]
        ]
        datasets = {
            name: {
                "dtype": str(trace[name].dtype),
                "shape": list(trace[name].shape),
                "sha256": dataset_sha256(trace[name]),
            }
            for name in NUMERIC_TRACE_DATASETS
        }
        summary = {
            "format": str(trace.attrs.get("format", "")),
            "format_version": int(trace.attrs.get("format_version", -1)),
            "committed_samples": int(trace.attrs.get("committed_samples", -1)),
            "committed_windows": int(trace.attrs.get("committed_windows", -1)),
            "stage_names": stage_names,
            "datasets": datasets,
        }
    summary["content_sha256"] = canonical_json_sha256(summary)
    return summary


def _maximum_absolute_difference(
    first: h5py.Dataset,
    second: h5py.Dataset,
) -> float | None:
    if first.shape != second.shape:
        return None
    maximum = 0.0
    if first.chunks is None:
        selections = [tuple(slice(0, size) for size in first.shape)]
    else:
        selections = first.iter_chunks()
    for selection in selections:
        a = np.asarray(first[selection])
        b = np.asarray(second[selection])
        if not np.issubdtype(a.dtype, np.number):
            continue
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if difference.size:
            maximum = max(maximum, float(np.max(difference)))
    return maximum


def compare_replay_artifacts(
    first_directory: str | Path,
    second_directory: str | Path,
) -> dict[str, Any]:
    """Compare independently constructed replay reports and trace contents."""
    directories = [Path(first_directory), Path(second_directory)]
    reports: list[dict[str, Any]] = []
    trace_summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    for label, directory in zip(("A", "B"), directories):
        report_path = directory / REPORT_NAME
        trace_path = directory / TRACE_NAME
        if not report_path.is_file():
            errors.append(f"Replay {label} report is missing: {report_path}.")
            reports.append({})
        else:
            with report_path.open(encoding="utf-8") as stream:
                reports.append(json.load(stream))
        if not trace_path.is_file():
            errors.append(f"Replay {label} trace is missing: {trace_path}.")
            trace_summaries.append({})
        else:
            trace_summaries.append(trace_content_summary(trace_path))

    if all(reports):
        for label, report in zip(("A", "B"), reports):
            if report.get("status") != "passed":
                errors.append(
                    f"Replay {label} did not pass its internal validation: "
                    f"{report.get('status')!r}."
                )
        if reports[0].get("replay_contract_sha256") != reports[1].get(
            "replay_contract_sha256"
        ):
            errors.append("Replay scientific contracts differ.")
        if reports[0].get("seed_manifest") != reports[1].get("seed_manifest"):
            errors.append("Replay seed manifests differ.")
        if reports[0].get("structure") != reports[1].get("structure"):
            errors.append("Replay structure/connectivity fingerprints differ.")
        if [row.get("spikes") for row in reports[0].get("windows", [])] != [
            row.get("spikes") for row in reports[1].get("windows", [])
        ]:
            errors.append("Replay spike-event fingerprints differ.")

    dataset_comparisons: dict[str, Any] = {}
    if all(trace_summaries):
        if trace_summaries[0]["stage_names"] != trace_summaries[1]["stage_names"]:
            errors.append("Trace stage names differ.")
        if trace_summaries[0]["committed_samples"] != trace_summaries[1][
            "committed_samples"
        ]:
            errors.append("Committed trace sample counts differ.")
        if trace_summaries[0]["committed_windows"] != trace_summaries[1][
            "committed_windows"
        ]:
            errors.append("Committed trace window counts differ.")

        with h5py.File(directories[0] / TRACE_NAME, "r") as first_trace, h5py.File(
            directories[1] / TRACE_NAME, "r"
        ) as second_trace:
            for name in NUMERIC_TRACE_DATASETS:
                first_summary = trace_summaries[0]["datasets"][name]
                second_summary = trace_summaries[1]["datasets"][name]
                exact = first_summary == second_summary
                maximum_difference = (
                    0.0
                    if exact
                    else _maximum_absolute_difference(
                        first_trace[name], second_trace[name]
                    )
                )
                dataset_comparisons[name] = {
                    "exact": bool(exact),
                    "maximum_absolute_difference": maximum_difference,
                    "first": first_summary,
                    "second": second_summary,
                }
                if not exact:
                    errors.append(f"Trace dataset {name!r} is not exactly reproducible.")

    return {
        "status": "passed" if not errors else "failed",
        "scope": (
            "Independent same-seed zero-field replay reproducibility; not a "
            "reference-versus-reduced-inhibition or stimulation result."
        ),
        "errors": errors,
        "runs": {
            "first": str(directories[0].resolve()),
            "second": str(directories[1].resolve()),
        },
        "replay_contract_sha256": [
            report.get("replay_contract_sha256") for report in reports
        ],
        "structure_sha256": [
            report.get("structure", {}).get("global_sha256") for report in reports
        ],
        "trace_content_sha256": [
            summary.get("content_sha256") for summary in trace_summaries
        ],
        "dataset_comparisons": dataset_comparisons,
        "trace_summaries": trace_summaries,
        "limitations": [
            "Exact replay at a fixed rank count does not establish reproducibility after changing MPI decomposition.",
            "This gate detects deterministic disagreement but cannot prove biological validity.",
        ],
    }
