"""H1d scale analysis of executed panels; this module defines no target physics.

Input sampling/acquisition precision and output target coarsening are separate
axes. Only equal physical target arrays and matching IDs qualify for a paired
fixed-target comparison. The shared target/acquisition registry owns operators.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import time

import numpy as np

from ..storage import atomic_json, file_hash
from .compression import _finalize, _identity, _npz, _verify_completed, validate_splits
from .scoring import fork_risk_components, circuit_summary

VERSION = "h1-scale-v1"


def full_complex_fft(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("Finite nonempty waveform required")
    return np.fft.fft(values, axis=-1, norm="ortho")


def fft_invariance(futures, predictions=None):
    """Full complex DFT preserves squared Euclidean risk and covariance trace.

    Both risks divide by the ORIGINAL real waveform dimension D. Splitting real
    and imaginary coordinates and dividing by 2D would change the loss by 1/2.
    """
    y = np.asarray(futures, dtype=float)
    if y.ndim != 2 or len(y) < 2 or y.shape[1] < 1 or not np.isfinite(y).all():
        raise ValueError("Need >=2 finite future vectors [M,D]")
    if predictions is None:
        predictions = np.zeros_like(y)
    pred = np.broadcast_to(np.asarray(predictions, dtype=float), y.shape)
    if not np.isfinite(pred).all():
        raise ValueError("Finite predictions required")
    transformed, prediction_fft = full_complex_fft(y), full_complex_fft(pred)
    error = np.mean((y - pred) ** 2, axis=1)
    fft_error = np.sum(np.abs(transformed - prediction_fft) ** 2, axis=1) / y.shape[-1]
    trace = float(np.sum(np.var(y, axis=0, ddof=1)))
    centered = transformed - transformed.mean(axis=0)
    fft_trace = float(np.sum(np.abs(centered) ** 2) / (len(y) - 1))
    if not np.allclose(error, fft_error, rtol=2e-12, atol=1e-14) or not np.isclose(trace, fft_trace, rtol=2e-12, atol=1e-14):
        raise AssertionError("Orthonormal FFT Parseval/covariance-trace invariance failed")
    return dict(dimension=y.shape[-1], futures=len(y), real_mse=float(error.mean()), complex_fft_mse=float(fft_error.mean()),
                covariance_trace=trace, complex_fft_covariance_trace=fft_trace,
                max_per_future_mse_error=float(np.max(np.abs(error - fft_error))),
                per_real_coordinate_floor=trace / y.shape[-1], fft_per_original_coordinate_floor=fft_trace / y.shape[-1],
                normalization="sum squared complex magnitudes divided by original real dimension D, not 2D")


def fft_controls(output=None, seed=8711):
    rng = np.random.default_rng(seed)
    rows = []
    for dimension in (1, 7, 16, 65):
        latent = rng.normal(size=(128, 3))
        transform = rng.normal(size=(3, dimension))
        y = latent @ transform + .2 * rng.normal(size=(128, dimension))
        pred = np.broadcast_to(rng.normal(size=dimension), y.shape)
        for factor in (1., 1e-6, 1e3):
            report = fft_invariance(y * factor, pred * factor)
            report.update(unit_factor=factor)
            rows.append(report)
    phase = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    first = np.cos(phase)
    second = np.cos(phase + np.pi / 2)
    power_difference = np.max(np.abs(np.abs(full_complex_fft(first)) ** 2 - np.abs(full_complex_fft(second)) ** 2))
    result = dict(schema_version=VERSION, seed=seed, invariance_cases=rows,
                  phase_removing_counterexample=dict(waveform_mse=float(np.mean((first - second) ** 2)),
                                                     max_power_spectrum_difference=float(power_difference),
                                                     same_psd_loses_phase=True),
                  loss_convention="unitary FFT keeps unrestricted waveform-MSE floor; PSD/magnitude are different lossy targets",
                  dft_bin_examples=[dict(duration_ms=ms, bin_spacing_hz=1000 / ms) for ms in (32, 128, 512, 2048)],
                  frequency_caveat="DFT bin spacing is not a universal impossibility theorem for sinusoidal parameter estimation")
    if output is not None:
        output = Path(output)
        atomic_json(output / "fft_controls.json", result)
        _npz(output / "phase_control.npz", waveform_a=first, waveform_b=second,
             fft_a=full_complex_fft(first), fft_b=full_complex_fft(second))
    return result


def _physical_fingerprint(manifest):
    value = manifest.get("physical_target_fingerprint")
    if not isinstance(value, str) or not value:
        raise ValueError("Missing declared physical_target_fingerprint: do not infer equality from target names")
    return value


def assert_fixed_physical_target(first, second, split="evaluation"):
    """Reject mismatched target identity, contexts, future IDs/masks or physical Y."""
    if _physical_fingerprint(first["manifest"]) != _physical_fingerprint(second["manifest"]):
        raise ValueError("Different physical target fingerprints")
    a, b = first[split], second[split]
    for name in ("context_id", "structure_id", "future_id", "future_mask"):
        if not np.array_equal(a[name], b[name]):
            raise ValueError(f"Unpaired fixed-target panels: {name}")
    mask = np.asarray(a["future_mask"], bool)
    if np.shape(a["Y_physical"]) != np.shape(b["Y_physical"]) or not np.array_equal(a["Y_physical"][mask], b["Y_physical"][mask]):
        raise ValueError("Physical future targets differ despite declared fingerprint")
    for name in ("reference_future_id", "reference_mask", "reference_Y_physical"):
        if name in a or name in b:
            if name not in a or name not in b or not np.array_equal(a[name], b[name]):
                raise ValueError(f"Unpaired reference pool: {name}")
    return True


def assert_fft_equivalent(first, second, split="evaluation"):
    """Verify the shared registry's invertible FFT packing on the same projection."""
    from .targets import TargetSpec, operator_from_projected
    a, b = first, second
    if a["manifest"]["operator"] == "full_complex_fft":
        a, b = b, a
    if (a["manifest"]["operator"], b["manifest"]["operator"]) != ("bin_average_waveform", "full_complex_fft"):
        raise ValueError("Not a common-projection waveform/full-complex-FFT pair")
    one, two = a["manifest"]["target_spec"], b["manifest"]["target_spec"]
    for key in ("lead_ms", "duration_ms", "output_grid_hz", "native_dt_ms", "source_signal", "units", "physical_support_ms"):
        if one.get(key) != two.get(key):
            raise ValueError(f"FFT control physical support/units mismatch: {key}")
    aa, bb = a[split], b[split]
    for key in ("context_id", "structure_id", "future_id", "future_mask"):
        if not np.array_equal(aa[key], bb[key]):
            raise ValueError(f"Unpaired FFT control: {key}")
    mask = aa["future_mask"]
    spec = TargetSpec.from_dict(two)
    expected = np.stack([operator_from_projected(v, spec) for v in aa["Y_physical"][mask]])
    if not np.array_equal(expected, bb["Y_physical"][mask]):
        raise ValueError("FFT panel does not equal shared registry transform of the same physical futures")
    if not np.isclose(np.mean(aa["Y_physical"][mask] ** 2), np.mean(expected ** 2), rtol=2e-12, atol=0):
        raise AssertionError("Registry FFT loss scaling does not preserve original waveform MSE")
    return True


def _future_training_mean(block):
    y = np.asarray(block["Y"], float)
    mask = np.asarray(block["future_mask"], bool)
    # Equal prefix weights, including when prefix future counts differ.
    return np.mean(np.sum(np.where(mask[..., None], y, 0), axis=1) / mask.sum(1)[:, None], axis=0)


def summarize_panel(panel):
    groups = validate_splits(panel)
    manifest = panel["manifest"]
    evaluation = panel["evaluation"]
    mean = _future_training_mean(panel["train"])
    targets = evaluation["Y"]
    baseline = np.broadcast_to(mean, (len(targets), targets.shape[-1]))
    context = fork_risk_components(targets, baseline, evaluation["future_mask"])
    y_scale = float(manifest["normalization"]["y_scale"])
    for name in ("mean_risk", "state_floor", "corrected_excess", "risk_mc_se", "floor_jackknife_mc_se"):
        context[name + "_physical"] = context[name] * y_scale ** 2
    input_spec = manifest["input"]
    rate = float(input_spec["sample_rate_hz"])
    if rate <= 0:
        raise ValueError("Nonpositive executed input rate")
    times = np.asarray(evaluation["input_time_ms"], float)
    if times.ndim != 1 or len(times) != evaluation["X"].shape[1] or np.any(times > 1e-9):
        raise ValueError("Input availability grid extends past cutoff or shape differs")
    if len(times) > 1 and not np.allclose(np.diff(times), 1000 / rate, rtol=1e-9, atol=1e-9):
        raise ValueError("Executed input rate disagrees with recorded timestamps")
    precision = input_spec.get("acquisition_bits", input_spec.get("bits"))
    if precision in ("float", "floating", "float64", 0):
        precision = None
    if precision is not None and (type(precision) is not int or precision <= 0):
        raise ValueError("Invalid acquisition-bit metadata")
    dimension = targets.shape[-1]
    if "target_spec" in manifest:
        from .targets import TargetSpec
        if TargetSpec.from_dict(manifest["target_spec"]).dimension != dimension:
            raise ValueError("Executed target array dimension disagrees with TargetSpec")
    floor_valid = np.isfinite(context["state_floor"])
    summary = dict(job_id=manifest["job_id"], status="executed_panel_analyzed", target_fingerprint=manifest["target_fingerprint"],
                   physical_target_fingerprint=_physical_fingerprint(manifest), comparison_fingerprint=manifest.get("comparison_fingerprint"),
                   operator=manifest.get("operator", manifest.get("target_spec", {}).get("operator")),
                   input_sample_rate_hz=rate, history_s=float(input_spec["history_s"]),
                   input_samples=int(evaluation["X"].shape[1]), input_acquisition_bits=precision,
                   input_fixed_bits_per_history=int(evaluation["X"].shape[1] * precision) if precision else None,
                   floating_array_storage_bits_per_value=int(evaluation["X"].dtype.itemsize * 8),
                   acquisition_vs_latent_rate="input acquisition bits and latent code bits are distinct; floating storage is not a physical acquisition rate",
                   lead_ms=float(manifest["lead_ms"]), duration_ms=float(manifest["duration_ms"]), output_dimensions=dimension,
                   output_grid_hz=float(manifest["output_grid_hz"]), dft_bin_spacing_hz=1000 / float(manifest["duration_ms"]),
                   input_metadata=input_spec, normalization=manifest["normalization"], split_circuits=groups,
                   contexts={s: len(panel[s]["X"]) for s in ("train", "development", "evaluation")},
                   scoring_futures=int(np.sum(evaluation["future_mask"])), floor_available_contexts=int(floor_valid.sum()),
                   r0_standardized=float(np.mean(context["mean_risk"])), r0_physical=float(np.mean(context["mean_risk_physical"])),
                   state_floor_standardized=float(np.mean(context["state_floor"][floor_valid])) if floor_valid.any() else None,
                   state_floor_physical=float(np.mean(context["state_floor_physical"][floor_valid])) if floor_valid.any() else None,
                   floor_scope="causal-state fork dispersion; unavailable at M=1, not EEG-history Bayes risk",
                   circuit_summary=circuit_summary(context, evaluation["structure_id"], bootstrap=0),
                   computation=manifest.get("compute", manifest.get("timing", {})),
                   scientific_limit="Each target has its own baseline risk. Lower absolute MSE from unit changes/coarsening does not establish improved information retention.")
    if floor_valid.any():
        matched_r0 = context["mean_risk"][floor_valid].mean()
        summary["state_floor_over_matched_r0"] = float(np.mean(context["state_floor"][floor_valid]) / matched_r0) if matched_r0 > 0 else None
    arrays = {k: v for k, v in context.items() if isinstance(v, np.ndarray)}
    arrays.update(context_id=evaluation["context_id"], structure_id=evaluation["structure_id"], future_id=evaluation["future_id"],
                  future_mask=evaluation["future_mask"], target_fingerprint=np.asarray(manifest["target_fingerprint"]), baseline_mean=mean)
    if summary["operator"] in {"native_waveform", "waveform", "native_bin_average", "bin_average_waveform", "legacy_filtered_stream", "native_block_average"}:
        summary["fft_invariance"] = [dict(context_id=str(evaluation["context_id"][i]), **fft_invariance(
            targets[i, evaluation["future_mask"][i]], mean)) for i in np.flatnonzero(floor_valid)]
    return summary, arrays


def _analysis_sources(root):
    paths = [root / "benchmarks" / "manifest.json", root / "compression" / "summary.json"]
    paths += sorted((root / "benchmarks").glob("*/summary.json"))
    paths += sorted((root / "benchmarks").glob("*/scores/*.npz"))
    return {str(p.relative_to(root)): file_hash(p) for p in paths if p.exists()}


def _benchmark_rows(root, panels):
    rows = []
    for job, panel in panels.items():
        directory = root / "benchmarks" / job
        marker = directory / "summary.json"
        if not marker.exists():
            continue
        saved = json.loads(marker.read_text())
        if saved.get("schema_version") != "eegrepr-h1-benchmark-v2" or saved.get("target_fingerprint") != panel["manifest"]["target_fingerprint"]:
            raise ValueError("Stale/incompatible benchmark summary in scale analysis")
        fit = {r["identity"]: r for r in saved.get("fits", [])}
        for report in saved["models"]:
            relative = Path(report["prediction_file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe benchmark score artifact path")
            path = directory / relative
            if file_hash(path) != saved["artifacts"].get(str(relative)):
                raise ValueError("Benchmark raw score checksum mismatch")
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {k: loaded[k] for k in loaded.files}
            data = panel["evaluation"]
            for key in ("context_id", "structure_id", "future_id", "future_mask"):
                if not np.array_equal(arrays[key], data[key]):
                    raise ValueError(f"Benchmark/scale panel pairing differs: {key}")
            if str(arrays["target_fingerprint"]) != panel["manifest"]["target_fingerprint"]:
                raise ValueError("Benchmark numeric target fingerprint differs")
            if not np.array_equal(arrays["truth"][data["future_mask"]], data["Y"][data["future_mask"]]):
                raise ValueError("Benchmark scored different future target arrays")
            raw = np.mean((arrays["truth"] - arrays["mean"][:, None]) ** 2, axis=-1)
            recomputed = np.sum(np.where(data["future_mask"], raw, 0), axis=1) / data["future_mask"].sum(1)
            if not np.allclose(recomputed, arrays["context__mean_risk"], rtol=1e-7, atol=1e-10):
                raise ValueError("Saved benchmark MSE cannot be reproduced from predictions")
            per_circuit = []
            for structure in np.unique(data["structure_id"]):
                ix = data["structure_id"] == structure
                item = {"structure_id": str(structure), "mean_risk": float(recomputed[ix].mean())}
                for metric in ("r0", "state_floor", "corrected_excess", "gaussian_nll", "marginal_crps", "energy", "coverage_95", "predictive_sd"):
                    values = arrays.get("context__" + metric)
                    if values is not None:
                        valid = np.isfinite(values[ix])
                        item[metric] = float(np.mean(values[ix][valid])) if valid.any() else None
                per_circuit.append(item)
            mean_risk = float(np.mean([r["mean_risk"] for r in per_circuit]))
            r0 = float(np.mean([r["r0"] for r in per_circuit]))
            rows.append(dict(job_id=job, model=report["model"], target_fingerprint=panel["manifest"]["target_fingerprint"],
                             physical_target_fingerprint=_physical_fingerprint(panel["manifest"]),
                             mean_circuit_risk=mean_risk, mean_circuit_r0=r0,
                             risk_over_mean_r0=mean_risk / r0 if r0 > 0 else None,
                             skill=1 - mean_risk / r0 if r0 > 0 else None,
                             physical_mse=mean_risk * panel["manifest"]["normalization"]["y_scale"] ** 2,
                             per_circuit=per_circuit, inference_seconds=report.get("inference_seconds"),
                             fit=fit.get(report["model"], {}).get("fit", {}),
                             raw_scores=str(path.relative_to(root)), raw_sha256=file_hash(path),
                             raw_mse_reaggregated=True, weighting="equal contexts within circuit, equal circuits"))
    return rows


def _compression_rows(root, panels):
    marker = root / "compression" / "summary.json"
    if not marker.exists():
        return []
    saved = json.loads(marker.read_text())
    if saved.get("schema_version") != "h1-compression-v1":
        raise ValueError("Stale compression schema in scale analysis")
    rows = []
    keys = ("job_id", "representation", "dimension", "code_bits", "task", "readout", "label_budget", "target_fingerprint", "mean_context_loss", "r0", "nmse", "skill", "task_ce_bits", "task_mi_lower_bound_estimate_bits", "per_circuit", "fit_seconds", "inference_seconds", "readout_parameter_count")
    for row in saved.get("rows", []):
        if row.get("status") != "complete":
            continue
        job = row["job_id"]
        if job not in panels or row["target_fingerprint"] != panels[job]["manifest"]["target_fingerprint"]:
            raise ValueError("Compression target does not match executed panel")
        rows.append({k: row[k] for k in keys if k in row})
    return rows


def run_scale(config, output_root, resume=False):
    from .dataset import list_jobs, load_panel
    root = Path(output_root)
    if (root / "provenance.json").exists():
        from .artifacts import require_compatible
        require_compatible(root, config)
    destination = root / "scale"
    jobs = list_jobs(root)
    panels = {job: load_panel(root, job) for job in jobs}
    sources = _analysis_sources(root)
    identity = _identity(dict(config=config, analysis_sources=sources, version=VERSION), [p["manifest"] for p in panels.values()])
    marker = destination / "summary.json"
    done = _verify_completed(destination, marker, identity, resume)
    if done is not None:
        return done
    guard = destination / "analysis_identity.json"
    if guard.exists() and json.loads(guard.read_text()).get("identity") != identity:
        raise ValueError("Incompatible partial scale output; use a fresh output root")
    if guard.exists() and not resume:
        raise FileExistsError("Partial scale output exists; use --resume")
    atomic_json(guard, dict(identity=identity))
    started = time.monotonic()
    fft_controls(destination / "controls")
    rows = []
    for job, panel in panels.items():
        row, arrays = summarize_panel(panel)
        _npz(destination / job / "context_scores.npz", **arrays)
        atomic_json(destination / job / "scale.json", row)
        rows.append(row)
    comparisons = []
    for (name_a, a), (name_b, b) in itertools.combinations(panels.items(), 2):
        common = _physical_fingerprint(a["manifest"]) == _physical_fingerprint(b["manifest"])
        if common:
            try:
                for split in ("train", "development", "evaluation"):
                    assert_fixed_physical_target(a, b, split)
            except ValueError as exc:
                # Report a failed pairing; never silently claim a fixed-target contrast.
                comparisons.append(dict(jobs=[name_a, name_b], status="unpaired_not_comparable", reason=str(exc)))
                continue
            fields = ("sample_rate_hz", "history_s", "acquisition_bits", "bits", "snr_db")
            altered = {k: [a["manifest"]["input"].get(k), b["manifest"]["input"].get(k)] for k in fields
                       if a["manifest"]["input"].get(k) != b["manifest"]["input"].get(k)}
            comparisons.append(dict(jobs=[name_a, name_b], status="verified_paired_fixed_physical_target",
                                    altered_input_axes=altered, physical_target_fingerprint=_physical_fingerprint(a["manifest"]),
                                    scope="input information/acquisition comparison; target arrays and all context/future IDs identical"))
        elif {a["manifest"].get("operator"), b["manifest"].get("operator")} == {"bin_average_waveform", "full_complex_fft"}:
            try:
                for split in ("train", "development", "evaluation"):
                    assert_fft_equivalent(a, b, split)
            except (ValueError, KeyError) as exc:
                comparisons.append(dict(jobs=[name_a, name_b], status="unpaired_fft_not_comparable", reason=str(exc)))
            else:
                comparisons.append(dict(jobs=[name_a, name_b], status="verified_unitary_equivalent_physical_waveform_loss",
                                        altered_input_axes={k: [a["manifest"]["input"].get(k), b["manifest"]["input"].get(k)] for k in ("sample_rate_hz", "history_s", "acquisition_bits", "snr_db") if a["manifest"]["input"].get(k) != b["manifest"]["input"].get(k)},
                                        scope="same projected waveform futures under invertible orthonormal DFT with registry loss-preserving packing; operator alone cannot reduce the causal-state squared-error floor; any input changes remain separate"))
        else:
            comparisons.append(dict(jobs=[name_a, name_b], status="distinct_target_estimands",
                                    scope="compare own normalized risk/task retention; absolute floor subtraction is invalid"))
    linked = {}
    for relative, digest in sources.items():
        if not relative.endswith(".json"):
            continue
        payload = json.loads((root / relative).read_text())
        linked[relative] = dict(sha256=digest, status=payload.get("status"),
                               interpretation="separate saved model/task scores; no refitting or target redefinition in scale analysis")
    axes = {key: sorted({str(r[key]) for r in rows}) for key in
            ("input_sample_rate_hz", "input_acquisition_bits", "history_s", "lead_ms", "duration_ms", "operator")}
    model_rows = _benchmark_rows(root, panels)
    task_rows = _compression_rows(root, panels)
    result = dict(schema_version=VERSION, status="complete" if jobs else "controls_only_no_panels", executed_jobs=jobs,
                  model_tradeoffs=model_rows, task_retention=task_rows,
                  actual_executed_axes=axes, rows=rows, comparisons=comparisons, linked_score_artifacts=linked,
                  configured_lists_are_not_execution_evidence=True, elapsed_seconds=time.monotonic() - started,
                  latency_note="Use recorded causal filter support/group delay and unchanged availability timestamps; never relabel delayed streams as instantaneous physical leads.",
                  scientific_status="Engineering/measurement summary; no human biological threshold or universal EEG information fraction")
    atomic_json(destination / "executed_scale_table.json", result)
    _plot(rows, destination)
    return _finalize(destination, marker, identity, result)


def _plot(rows, output):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    indexes = np.arange(len(rows))
    axes[0].bar(indexes, [r["r0_standardized"] for r in rows], label="Own training-mean reference")
    axes[0].scatter(indexes, [r["state_floor_standardized"] if r["state_floor_standardized"] is not None else np.nan for r in rows],
                    color="black", label="Own causal-state fork dispersion")
    axes[0].set(ylabel="Standardized target MSE", xticks=indexes, xticklabels=[r["job_id"] for r in rows], title="Distinct targets have distinct references")
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].legend(fontsize=7)
    axes[1].scatter([r["input_samples"] for r in rows], [r["output_dimensions"] for r in rows])
    axes[1].set(xlabel="Actually observed history samples", ylabel="Actual target dimensions", title="Executed size / compute inputs")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"executed_scale_summary.{ext}", dpi=150)
    plt.close(fig)
