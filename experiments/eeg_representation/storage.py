"""Versioned atomic artifacts and strict resume; never uses legacy folder helpers."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
import numpy as np
from . import SCHEMA_VERSION
from .schema import config_hash


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(jsonable(value), f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hash():
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((root / "setup/circuits/ballnstick").glob("*.py"))
    paths += list((root / "setup/circuits/ballnstick").glob("*.hoc"))
    paths += [root / "configs/env/ballnstick.yaml"]
    paths += list((root / "env/models/neuron").glob("*.py"))
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def provenance(config):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    versions = {}
    for name in ["numpy", "scipy", "h5py", "PyYAML", "torch", "scikit-learn", "NEURON", "LFPy", "mpi4py"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(schema_version=SCHEMA_VERSION, config_hash=config_hash(config), commit=git("rev-parse", "HEAD"),
        source_hash=source_hash(), dirty=bool(git("status", "--porcelain")), versions=versions,
        python=platform.python_version(), created_unix=time.time(), units={"eeg": "V", "time": "ms", "dipole": "nA um"})


@contextmanager
def output_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".eegrepr.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another writer holds this output root; use one coordinator") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def initialize_output(root, config, resume=False):
    root = Path(root).resolve()
    marker = root / "provenance.json"
    if marker.exists():
        old = read_json(marker)
        if old.get("schema_version") != SCHEMA_VERSION or old.get("config_hash") != config_hash(config):
            raise ValueError("Incompatible resume: schema/config hash differs; use a fresh output root")
        if old.get("source_hash") != source_hash():
            raise ValueError("Incompatible resume: source hash changed; use a fresh output root")
        if not resume:
            raise FileExistsError("Output exists; pass --resume for verified reuse")
        return old
    if root.exists() and any(p.name != ".eegrepr.lock" for p in root.iterdir()):
        raise FileExistsError("Refusing a nonempty unversioned output directory")
    root.mkdir(parents=True, exist_ok=True)
    p = provenance(config)
    atomic_json(root / "resolved_config.json", config)
    atomic_json(marker, p)
    return p


def require_compatible(root, config):
    p = read_json(Path(root) / "provenance.json")
    if p["config_hash"] != config_hash(config) or p["schema_version"] != SCHEMA_VERSION or p["source_hash"] != source_hash():
        raise ValueError("Config/schema/source mismatch; do not mix datasets or implementations")
    return p


def write_trial(path, trial, metadata):
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temp, "w") as h:
            h.attrs["schema_version"] = SCHEMA_VERSION
            h.attrs["metadata_json"] = json.dumps(jsonable(metadata), sort_keys=True, allow_nan=False)
            for k in ["time_ms", "eeg_v", "dipole_na_um"]:
                a = np.asarray(trial[k], dtype=np.float64)
                if not np.isfinite(a).all():
                    raise ValueError(f"Nonfinite {k}")
                h.create_dataset(k, data=a, compression="gzip", shuffle=True)
            g = h.create_group("privileged")
            for key, value in trial.get("privileged", {}).items():
                a = np.asarray(value)
                if a.dtype.kind in "biufc" and a.size:
                    g.create_dataset(key, data=a, compression="gzip" if a.ndim else None)
                else:
                    g.attrs[key] = json.dumps(jsonable(value))
            h.flush()
        with open(temp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return file_hash(path)


def verify_shard(root, row):
    path = Path(root) / row["path"]
    if not path.exists() or file_hash(path) != row["sha256"]:
        raise ValueError(f"Missing or corrupt shard: {path}")


def rng_seed(namespace, *parts):
    b = json.dumps([namespace, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:4], "little")
