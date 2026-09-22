"""Small provenance helpers shared only by the new qualification studies."""
from __future__ import annotations

import hashlib
import json
import platform
import importlib.metadata
import subprocess
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def begin(root, cfg, runner):
    root = Path(root)
    # Never mix new measurements with an earlier (possibly incomplete) run.
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a new experiment.name; result directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, root / "resolved_config.yaml")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode().split("\0")
    sources = {p: sha256(REPO / p) for p in tracked if p and (
        p.startswith(("env/models/neuron/", "setup/circuits/ballnstick/")) or
        p.endswith(("run_ballnstick_stationary_h1_h3_confirmation.py", "run_ballnstick_h4_confirmation.py")))}
    for path in Path(__file__).parent.rglob("*"):
        if path.suffix in (".py", ".mod"):
            sources[str(path.relative_to(REPO))] = sha256(path)
    sources[str(Path(runner).relative_to(REPO))] = sha256(runner)
    write_json(root / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO).decode(),
        "python": platform.python_version(), "source_sha256": sources,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "pandas", "NEURON", "LFPy", "mpi4py", "hydra-core")},
        "started_unix_s": time.time(), "claims": "Qualification only; not H5, disease validation, or clinical safety.",
    })


def finish(root, started, conclusion):
    conclusion["runtime_seconds"] = time.perf_counter() - started
    write_json(root / "experiment_conclusion.json", conclusion)
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.name != "run_complete.json"}
    write_json(root / "run_complete.json", {"completed": True,
        "runtime_seconds": conclusion["runtime_seconds"], "files_sha256": files})
    print(json.dumps(json_ready(conclusion), indent=2), flush=True)
    print(f"Results saved to: {root}", flush=True)


def guarded_main(main):
    from mpi4py import MPI
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise
