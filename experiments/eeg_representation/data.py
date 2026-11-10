"""Trial-sharded generation, causal observations, structural splits and EEG-only inputs."""
from __future__ import annotations
from pathlib import Path
import json
import multiprocessing as mp
import os
import tempfile
import time
import numpy as np
from . import SCHEMA_VERSION
from .schema import config_hash
from .storage import (atomic_json, read_json, initialize_output, output_lock, require_compatible,
    write_trial, verify_shard, file_hash, rng_seed, jsonable)
from .tasks import TASKS, labels_for, validate_tasks

PROTOCOLS = ("leave-task-label-out", "leave-generator-family-out")


def trial_specs(config):
    d = config["data"]
    result = []
    for split, structures in d["structures"].items():
        for structure in structures:
            for trial in range(d["trials_per_regime"]):
                family = f"s{structure}-p{trial:04d}"
                for regime in d["regimes"]:
                    tid = f"{family}-{regime['name']}"
                    result.append(dict(trial_id=tid, parent_trial_id=tid, paired_family_id=family,
                        structure_id=str(structure), structure_seed=structure, split=split,
                        trial_seed=rng_seed(d["namespace"], "events", structure, trial),
                        phase_seed=rng_seed(d["namespace"], "phase", structure, trial),
                        envelope_seed=rng_seed(d["namespace"], "envelope", structure, trial),
                        noise_seed=rng_seed(d["namespace"], "sensor", structure, trial),
                        history_id=family, future_id=None, regime=regime, family=regime["family"]))
    # Record order is explicitly label-blind and unrelated to split or carrier.
    np.random.default_rng(rng_seed(d["namespace"], "record_order")).shuffle(result)
    return result


def assert_splits(rows):
    for key in ["structure_id", "parent_trial_id", "paired_family_id", "history_id"]:
        seen = {}
        for row in rows:
            if row.get(key) is None:
                continue
            old = seen.setdefault(row[key], row["split"])
            if old != row["split"]:
                raise ValueError(f"Leakage: {key} {row[key]} appears in {old} and {row['split']}")
    if len({r["trial_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trial ID")


def _generate_one(args):
    config, root, spec = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if config["data"]["backend"] == "neuron":
        from .simulator import generate_trial
    else:
        from .analytical import generate_trial
    start = time.perf_counter()
    trial = generate_trial(config, spec)
    t = np.asarray(trial["time_ms"])
    eeg = np.asarray(trial["eeg_v"])
    n = round((config["simulation"]["burn_in_s"] + config["simulation"]["duration_s"]) * 1000 / config["simulation"]["native_dt_ms"])
    if t.shape != (n,) or eeg.shape != (n, 1) or not np.all(np.diff(t) > 0):
        raise ValueError(f"Unexpected native time/EEG shape {t.shape}/{eeg.shape}; expected {n} samples")
    if not np.allclose(t, np.arange(1, n + 1) * config["simulation"]["native_dt_ms"], atol=1e-7, rtol=0):
        raise ValueError("Native times violate (start, stop] convention")
    prov = read_json(Path(root) / "provenance.json")
    row = dict(spec, schema_version=SCHEMA_VERSION, config_hash=config_hash(config),
        commit=prov["commit"], source_hash=prov["source_hash"], software_versions=prov["versions"],
        measurement_variant="clean", units={"eeg": "V", "time": "ms", "dipole": "nA um"},
        metadata=trial.get("metadata", {}), status=config["claim_status"],
        labels=labels_for(spec["regime"], trial.get("privileged", {})))
    row["path"] = f"trials/{spec['trial_id']}.h5"
    row["runtime_s"] = time.perf_counter() - start
    row["sha256"] = write_trial(Path(root) / row["path"], trial, row)
    row["bytes"] = (Path(root) / row["path"]).stat().st_size
    return row


def generate(config, output_root, resume=False, workers=None):
    root = Path(output_root)
    validate_tasks(config)
    if config["claim_status"] == "confirmation" and not config["evaluation"]["frozen"]:
        raise ValueError("Provisional confirmation cannot generate test circuits: freeze the configuration first")
    with output_lock(root):
        initialize_output(root, config, resume)
        specs = trial_specs(config)
        assert_splits(specs)
        atomic_json(root / "splits.json", dict(schema_version=SCHEMA_VERSION,
            structures=config["data"]["structures"], trials=[{k: r[k] for k in
                ["trial_id", "parent_trial_id", "paired_family_id", "history_id", "structure_id", "split"]} for r in specs]))
        atomic_json(root / "task_registry.json", {"schema_version": SCHEMA_VERSION, "tasks": TASKS})
        manifest = root / "manifest.json"
        rows = read_json(manifest)["trials"] if manifest.exists() else []
        for row in rows:
            verify_shard(root, row)
        known = {r["trial_id"] for r in rows}
        pending = [s for s in specs if s["trial_id"] not in known]
        # A crash after shard rename but before manifest commit is recovered by regenerating
        # that deterministic shard; no committed shard is overwritten.
        args = [(config, str(root), s) for s in pending]
        count = workers or config["data"]["workers"]
        if count < 1:
            raise ValueError("workers must be positive")
        if count == 1:
            iterator = map(_generate_one, args)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(count, maxtasksperchild=1)
            iterator = pool.imap_unordered(_generate_one, args)
        try:
            for row in iterator:
                rows.append(row)
                assert_splits(rows)
                atomic_json(manifest, dict(schema_version=SCHEMA_VERSION, config_hash=config_hash(config), trials=rows))
                print(f"generated {len(rows)}/{len(specs)} {row['trial_id']} ({row['runtime_s']:.2f}s)", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        total = dict(trials=len(rows), native_bytes=sum(r["bytes"] for r in rows),
            summed_trial_runtime_s=sum(r["runtime_s"] for r in rows),
            claim_status=config["claim_status"], backend=config["data"]["backend"])
        atomic_json(root / "generation_summary.json", total)
        return total


def _atomic_h5(path, arrays, attrs):
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temp, "w") as h:
            for key, value in arrays.items():
                h.create_dataset(key, data=value, compression="gzip", shuffle=True)
            h.attrs["metadata_json"] = json.dumps(jsonable(attrs), allow_nan=False, sort_keys=True)
            h.flush()
        with open(temp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _native_to_model(root, row, config):
    import h5py
    from .preprocessing import causal_resample
    verify_shard(root, row)
    with h5py.File(root / row["path"], "r") as h:
        x, times, meta = causal_resample(h["eeg_v"][:], h["time_ms"][:], target_fs=config["simulation"]["model_sample_rate_hz"])
    keep = times > config["simulation"]["burn_in_s"] * 1000 + 1e-8
    x, times = x[keep], times[keep]
    expected = round(config["simulation"]["duration_s"] * config["simulation"]["model_sample_rate_hz"])
    if len(x) != expected:
        raise ValueError(f"Derived stream has {len(x)} samples, expected {expected}")
    return x.astype(np.float64), times, meta


def _eligible_pretrain(row, config, protocol):
    return protocol != "leave-generator-family-out" or row["family"] not in config["evaluation"]["heldout_families"]


def derive(config, output_root, resume=False):
    root = Path(output_root)
    with output_lock(root):
        require_compatible(root, config)
        rows = read_json(root / "manifest.json")["trials"]
        if len(rows) != len(trial_specs(config)):
            raise ValueError("Finish generation before fitting observation statistics")
        assert_splits(rows)
        for protocol in PROTOCOLS:
            target = root / "derived" / protocol
            marker = target / "manifest.json"
            if marker.exists() and not resume:
                raise FileExistsError("Derived output exists; use --resume")
            # Streaming moments in float64; no absolute epsilon at scalp-voltage scale.
            count, mean, m2 = 0, None, None
            for row in rows:
                if row["split"] != "train" or not _eligible_pretrain(row, config, protocol):
                    continue
                x, _, _ = _native_to_model(root, row, config)
                n, mu = len(x), x.mean(axis=0)
                varsum = ((x - mu) ** 2).sum(axis=0)
                if mean is None:
                    count, mean, m2 = n, mu, varsum
                else:
                    delta = mu - mean
                    m2 += varsum + delta * delta * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
            if count == 0:
                raise ValueError("No training recordings remain under this transfer protocol")
            scale = np.sqrt(m2 / count)
            if np.any(scale <= 0) or not np.isfinite(scale).all():
                raise ValueError("Genuinely zero-variance or nonfinite training EEG")
            stats = dict(mean_v=mean, scale_v=scale, reference_power_v2=m2 / count, samples=count,
                fit_split="train", protocol=protocol, conversion="none; physical arrays remain V float64",
                noise_covariance="independent white Gaussian channels, diagonal reference_power * 10**(-SNR/10)",
                snr_definition="10 log10(training pooled demeaned channel power / sensor noise variance)")
            atomic_json(target / "normalization.json", stats)
            derived_rows = read_json(marker)["trials"] if marker.exists() else []
            known = {r["trial_id"]: r for r in derived_rows}
            for row in rows:
                if row["trial_id"] in known:
                    verify_shard(root, known[row["trial_id"]])
                    continue
                x, times, filter_meta = _native_to_model(root, row, config)
                arrays = {"time_ms": times.astype(np.float64), "clean": x}
                unit_noise = np.random.default_rng(row["noise_seed"]).normal(size=x.shape)
                for snr in config["observation"]["snr_db"]:
                    arrays[f"snr_{snr:g}"] = x + unit_noise * scale * 10 ** (-snr / 20)
                arrays["unit_noise"] = unit_noise.astype(np.float64)
                path = target / "trials" / f"{row['trial_id']}.h5"
                _atomic_h5(path, arrays, dict(filter=filter_meta, parent=row["trial_id"],
                    normalization=stats, noise_seed=row["noise_seed"], schema_version=SCHEMA_VERSION))
                dr = dict(row, path=str(path.relative_to(root)), sha256=file_hash(path), variants=[k for k in arrays if k not in {"time_ms", "unit_noise"}])
                # Signal diagnostics are observed quantities, kept separate from generator labels.
                from scipy.signal import welch
                fs = config["simulation"]["model_sample_rate_hz"]
                freq, psd = welch(x[:, 0], fs=fs, nperseg=min(len(x), 4 * fs))
                band = (freq >= 4) & (freq <= 40)
                half = len(x) // 2
                dr["observed_diagnostics"] = dict(peak_4_40_hz=float(freq[band][np.argmax(psd[band])]),
                    variance_v2=float(x.var()), first_half_variance_v2=float(x[:half].var()),
                    second_half_variance_v2=float(x[half:].var()),
                    stationarity_status="descriptive only; burn-in adequacy pending pilot")
                derived_rows.append(dr)
                atomic_json(marker, dict(schema_version=SCHEMA_VERSION, protocol=protocol, trials=derived_rows))
        return {"protocols": list(PROTOCOLS), "trials_per_protocol": len(rows), "variants": ["clean"] + [f"snr_{s:g}" for s in config["observation"]["snr_db"]]}


def _budget_rows(rows, budget, seed):
    if budget is None:
        return rows
    selected = set()
    for condition in sorted({r["regime"]["name"] for r in rows}):
        candidates = [r for r in rows if r["regime"]["name"] == condition]
        order = sorted(candidates, key=lambda r: rng_seed("nested-label-subset", seed, r["paired_family_id"]))
        selected.update(r["trial_id"] for r in order[:budget])
    return [r for r in rows if r["trial_id"] in selected]


def load_examples(root, config, split, protocol, stage="pretrain", budget=None, seed=0, variant=None, gain=None):
    """Only EEG tensors enter x; metadata is for grouping/scoring, never model input.

    Windows use (cutoff-history, cutoff]; target begins at cutoff+lead. Causal
    filtering uses full prefix state, never future context. Missing labels stay masked.
    """
    import h5py
    root = Path(root)
    require_compatible(root, config)
    if protocol not in PROTOCOLS or split not in {"train", "development", "test"}:
        raise ValueError("Unknown protocol/split")
    if split == "test" and config["claim_status"] == "confirmation" and not config["evaluation"]["allow_test"]:
        raise ValueError("Final confirmation test access is locked; freeze a separate final configuration before generation")
    path = root / "derived" / protocol
    manifest = read_json(path / "manifest.json")
    assert_splits(manifest["trials"])
    rows = [r for r in manifest["trials"] if r["split"] == split]
    if stage == "pretrain":
        rows = [r for r in rows if _eligible_pretrain(r, config, protocol)]
    rows = _budget_rows(rows, budget, seed)
    norm = read_json(path / "normalization.json")
    mean, scale = np.asarray(norm["mean_v"]), np.asarray(norm["scale_v"])
    fs = config["simulation"]["model_sample_rate_hz"]
    hist = round(config["forecast"]["history_s"] * fs)
    lead = round(config["forecast"]["lead_ms"] * fs / 1000)
    block = round(config["forecast"]["block_ms"] * fs / 1000)
    stride = max(1, round(config["forecast"]["stride_s"] * fs))
    variant = variant or config["observation"]["variant"]
    gain = config["observation"]["gain"] if gain is None else gain
    if stage == "pretrain" and (variant != config["observation"]["variant"] or gain != config["observation"]["gain"]):
        raise ValueError("View overrides are evaluation-only")
    xs, futures, metadata = [], [], []
    labels = {k: [] for k in TASKS}
    for row in rows:
        verify_shard(root, row)
        with h5py.File(root / row["path"], "r") as h:
            if variant not in row["variants"]:
                raise ValueError(f"Observation variant {variant} is not in derived manifest")
            observed = h[variant][:] * gain
            target = h["clean" if config["forecast"]["target"] == "clean" else variant][:]
        for cutoff in range(hist - 1, len(observed) - lead - block + 1, stride):
            x = (observed[cutoff - hist + 1:cutoff + 1] - mean) / scale
            y = (target[cutoff + lead:cutoff + lead + block] - mean) / scale
            xs.append(x.T.astype(np.float32))
            futures.append(y.T.astype(np.float32))
            metadata.append({k: row[k] for k in ["structure_id", "parent_trial_id", "trial_id", "family", "paired_family_id"]} |
                {"cutoff_index": cutoff, "measurement_variant": variant, "gain": gain, "split": split})
            for task, definition in TASKS.items():
                value = row["labels"].get(task)
                if task == "envelope":
                    value = float(np.sqrt(np.mean(y ** 2)))
                elif task == "log_bandpower":
                    from scipy.signal import periodogram
                    ff, pp = periodogram(y, fs=fs, axis=0, window="hann", detrend="constant")
                    band = (ff >= 4) & (ff <= 30)
                    power = float(pp[band].sum() * (ff[1] - ff[0]))
                    value = float(np.log(power)) if power > 0 else float("nan")
                if stage == "pretrain" and task not in config["training"]["source_tasks"]:
                    value = None
                if value is None:
                    value = -1 if definition["kind"] == "classification" else float("nan")
                labels[task].append(value)
    if not xs:
        raise ValueError("No eligible windows; check split, family holdout and time support")
    return dict(x=np.stack(xs), future=np.stack(futures), labels={k: np.asarray(v) for k, v in labels.items()},
        metadata=metadata, normalization=norm, sample_rate_hz=fs,
        allowed_input_columns=["eeg"], input_units="training-normalized V; float32")


def assert_input_columns(columns):
    if set(columns) != {"eeg"}:
        raise ValueError("Model inputs are restricted to EEG; privileged/identifier columns are forbidden")
