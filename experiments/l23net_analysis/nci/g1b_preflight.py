"""Small NCI helpers; progress and prerequisite modes need only standard Python."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys


def prerequisite(path):
    report = json.loads(Path(path).read_text())
    if report.get("status") != "passed" or report.get("errors"):
        raise ValueError("G1A did not pass.")
    hashes = report.get("structure_sha256", [])
    if len(hashes) != 2 or not hashes[0] or hashes[0] != hashes[1]:
        raise ValueError("G1A structure comparison is absent or inconsistent.")
    required = {"eeg_v", "dipole_nA_um", "sample_time_ms", "field_left_boundary_time_ms", "field_left_boundary_v_per_m", "stage_code"}
    comparisons = report.get("dataset_comparisons", {})
    if set(comparisons) != required or not all(row.get("exact") for row in comparisons.values()):
        raise ValueError("G1A trace comparisons are incomplete or not exact.")
    summaries = report.get("trace_summaries", [])
    if len(summaries) != 2 or any(row.get("committed_samples") != 240000 or row.get("committed_windows") != 6 for row in summaries):
        raise ValueError("G1A did not commit both complete six-second traces.")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mode, raw_directory = sys.argv[1:3]
    directory = Path(raw_directory)
    if mode == "prerequisite":
        print(prerequisite(directory))
    elif mode == "progress":
        values = []
        for condition in ("reference", "mdd"):
            path = directory/condition/"l23net_g1b_run.json"
            try:
                values.append(str(json.loads(path.read_text()).get("completed_simulated_ms", 0)))
            except (OSError, ValueError):
                values.append("0")
        print("\t".join(values))
    elif mode == "environment":
        expected = {"LFPy": "2.3", "NEURON": "8.2.3", "numpy": "1.26.3", "scipy": "1.11.4", "mpi4py": "3.1.5"}
        modules = {name: importlib.import_module("neuron" if name == "NEURON" else name) for name in (*expected, "h5py")}
        versions = {name: module.__version__ for name, module in modules.items()}
        errors = [f"{k}: {versions[k]} != {v}" for k, v in expected.items() if versions[k] != v]
        report = {"versions": versions, "expected": expected, "python": sys.version, "errors": errors,
                  "module_paths": {name: module.__file__ for name, module in modules.items()}}
        (directory/"environment_versions.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)
        if errors:
            raise SystemExit("Environment differs from validated G1A environment.")
    elif mode == "mechanisms":
        from mpi4py import MPI
        import neuron
        from neuron import h
        comm = MPI.COMM_WORLD
        if not neuron.load_mechanisms(os.environ["L23NET_MECHANISM_PATH"]):
            comm.Abort(2)
        sec = h.Section(name=f"g1b_preflight_{comm.rank}")
        sec.insert("tonic")
        sec.insert("Ih")
        hosts = comm.allgather(socket.gethostname())
        if len(set(hosts)) != 13 or comm.size != 13:
            comm.Abort(2)
        if comm.rank == 0:
            print("Shared mechanism library verified on all 13 nodes.", flush=True)
        h.delete_section(sec=sec)
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
