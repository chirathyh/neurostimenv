"""CF0b-only causal estimators and explicitly nondeployable scoring references.

Importing this module never inserts mechanisms or changes a historical runner.
The common centered neural reference is an offline comparison convention, NOT
instantaneous biological ground truth. No reference samples enter an estimator.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats

from experiments.ballnstick_analysis.qualification import continuous_alpha as ca


def wrap(x):
    return np.angle(np.exp(1j * np.asarray(x)))


def estimate_phase(raw, fs, frequency, stop_s, history_s, pipeline):
    n = int(round(history_s * fs))
    tail = np.asarray(raw[-n:], dtype=float)
    if len(tail) != n or not np.isfinite(tail).all():
        raise ValueError("Need a finite complete preceding phase tail")
    if pipeline == "raw_ols":
        return ca.phase_at_boundary(tail, fs, frequency, stop_s, history_s)
    if pipeline != "h4_tail":
        raise ValueError(f"Unknown phase pipeline: {pipeline}")
    # Call historical primitives, retaining the H4 sample-time convention.
    # filtfilt is used ONLY inside the already available preceding tail. This
    # is causal as a boundary decision, not a streaming zero-phase filter.
    from experiments.ballnstick_analysis.run_ballnstick import _preprocess_eeg
    from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import _fourier_coefficients
    processed, out_fs = _preprocess_eeg(tail, fs, 500, .5, 100.)
    cosine, sine = _fourier_coefficients(processed, fs_hz=out_fs,
        start_ms=(stop_s-history_s)*1000, frequency_hz=frequency)
    amplitude = np.hypot(cosine, sine)
    phase = float(wrap(2*np.pi*frequency*stop_s + np.arctan2(-sine, cosine)))
    # _extract_eeg_features defines rms_v after mean removal.
    confidence = float(amplitude / max(np.sqrt(np.mean(processed**2)), 1e-300))
    return phase, confidence


def centered_reference(neural, fs, frequency, center_s, history_s):
    """Center times relative to saved trace; may use future NEURAL samples."""
    left, right = [int(round(t*fs)) for t in
                   (center_s-history_s/2, center_s+history_s/2)]
    if left < 0 or right > len(neural):
        return np.nan, np.nan
    phase_at_end, confidence = ca.phase_at_boundary(neural[left:right], fs,
        frequency, right/fs, history_s)
    return float(wrap(phase_at_end-2*np.pi*frequency*(right/fs-center_s))), confidence


def audit_trajectory(neural, observed, fs, baseline_count, frequency, settings):
    """No hidden carrier or generator state supplied; fixed EEG-selected f only."""
    neural, observed = np.asarray(neural), np.asarray(observed)
    if neural.shape != observed.shape or neural.ndim != 1:
        raise ValueError("Expected paired flat neural/observed traces")
    if not np.isfinite(neural).all() or not np.isfinite(observed).all():
        raise ValueError("Nonfinite trace")
    step = int(round(settings["update_s"]*fs))
    candidates = settings["candidates"]
    rows = []
    for stop in range(int(baseline_count), len(neural)+1, step):
        boundary = stop/fs
        history = settings["initialization_s"] if stop == baseline_count else settings["history_s"]
        reference, ref_confidence = centered_reference(neural, fs, frequency,
            boundary, settings["reference_history_s"])
        forward_ref, _ = centered_reference(neural, fs, frequency,
            boundary+settings["prediction_s"], settings["reference_history_s"])
        for candidate in candidates:
            phase, confidence = estimate_phase(observed[:stop], fs, frequency,
                boundary, history, candidate["pipeline"])
            neural_phase, neural_confidence = estimate_phase(neural[:stop], fs,
                frequency, boundary, history, candidate["pipeline"])
            reference_error = abs(float(wrap(phase-reference)))
            prediction = phase+2*np.pi*frequency*settings["prediction_s"]
            rows.append({"profile": candidate["name"], "pipeline": candidate["pipeline"],
                "elapsed_s": boundary-baseline_count/fs, "boundary_s": boundary,
                "latest_input_s": boundary, "history_s": history,
                "phase_rad": phase, "confidence": confidence,
                "actionable": confidence >= candidate["cutoff"],
                "neural_phase_rad": neural_phase, "neural_confidence": neural_confidence,
                "neural_actionable": neural_confidence >= candidate["cutoff"],
                "noise_error_deg": np.degrees(abs(float(wrap(phase-neural_phase)))),
                "common_reference_rad": reference, "reference_confidence": ref_confidence,
                "reference_valid": bool(np.isfinite(reference)),
                "reference_error_deg": np.degrees(reference_error),
                "neural_reference_error_deg": np.degrees(abs(float(wrap(neural_phase-reference)))),
                "large_error": bool(reference_error > np.pi/2) if np.isfinite(reference) else np.nan,
                "prediction_reference_valid": bool(np.isfinite(forward_ref)),
                "prediction_error_deg": np.degrees(abs(float(wrap(prediction-forward_ref)))),
                "reference_latest_input_s": boundary+settings["reference_history_s"]/2,
                "reference_is_offline_neural_benchmark_not_truth": True})
    return pd.DataFrame(rows)


def phase_summary(rows, settings, minimum_structures):
    """Structure -> crossed contexts -> windows; never window-wise inference."""
    records = []
    for (profile, structure, context, diffusion), group in rows[rows.state == "A"].groupby(
            ["profile", "structure_seed", "context", "D"]):
        accepted = group[group.actionable]
        valid = group[group.reference_valid]
        records.append({"profile": profile, "structure_seed": structure, "context": context,
            "D": diffusion, "coverage": group.actionable.mean(),
            "neural_coverage": group.neural_actionable.mean(),
            "noise_error_deg": accepted.noise_error_deg.mean(),
            "reference_error_deg": valid.reference_error_deg.mean(),
            "accepted_reference_error_deg": accepted.reference_error_deg.mean(),
            "large_error_fraction": accepted.large_error.mean(),
            "prediction_error_deg": group.prediction_error_deg.mean(),
            "valid_reference_fraction": group.reference_valid.mean()})
    contexts = pd.DataFrame(records)
    summaries, structures = {}, []
    metrics = ["coverage", "neural_coverage", "noise_error_deg", "reference_error_deg",
               "accepted_reference_error_deg", "large_error_fraction", "prediction_error_deg",
               "valid_reference_fraction"]
    for profile, group in contexts.groupby("profile"):
        table = group.groupby("structure_seed")[metrics].mean()
        table["profile"] = profile
        structures.append(table.reset_index())
        means = table[metrics].mean()
        # Abstain-all contexts must fail; pandas' skip-NaN cannot hide them.
        finite = bool(np.isfinite(group[metrics].to_numpy()).all())
        by_d = group.groupby(["structure_seed", "D"]).coverage.mean().groupby("D").mean()
        checks = {
            "minimum_structures": len(table) >= minimum_structures,
            "finite_context_endpoints": finite,
            "coverage": means.coverage >= settings["minimum_coverage"],
            "both_diffusion_coverages": bool((by_d >= settings["minimum_diffusion_coverage"]).all()),
            "each_structure_coverage": bool((table.coverage >= settings["minimum_structure_coverage"]).all()),
            "accepted_noise_error": means.noise_error_deg <= settings["maximum_noise_error_deg"],
            "unconditional_common_reference_error": means.reference_error_deg <= settings["maximum_reference_error_deg"],
            "accepted_large_error_fraction": means.large_error_fraction <= settings["maximum_large_error_fraction"],
            "unconditional_forward_reference_error": means.prediction_error_deg <= settings["maximum_prediction_error_deg"],
        }
        summaries[profile] = {"checks": checks, "passes": all(checks.values()),
            "structure_count": len(table), "means": means.to_dict(),
            "coverage_by_diffusion": by_d.to_dict()}
    return summaries, contexts, pd.concat(structures, ignore_index=True)


def choose_profile(summaries, candidates):
    eligible = [p for p in candidates if summaries[p["name"]]["passes"]]
    if not eligible:
        return None
    return min(eligible, key=lambda p: (summaries[p["name"]]["means"]["reference_error_deg"],
        summaries[p["name"]]["means"]["noise_error_deg"], p["name"]))


def rank_threshold(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or not np.isfinite(scores).all() or not len(scores):
        raise ValueError("Need finite independent B calibration scores and 0<alpha<1")
    rank = int(math.ceil((len(scores)+1)*(1-alpha)-1e-12))
    return {"threshold_db": float(np.sort(scores)[rank-1]) if rank <= len(scores) else None,
            "rank": rank, "n_calibration_structures": len(scores), "alpha": alpha,
            "strict_exceedance": True, "insufficient_calibration_abstains": rank > len(scores)}


def rhythm_present(evidence_db, screen):
    threshold = screen["threshold_db"]
    return bool(threshold is not None and np.isfinite(evidence_db) and evidence_db > threshold)


def structure_bootstrap(values, n, seed):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return np.quantile(boot, [.025, .975]).tolist()


def exact_binomial_interval(successes, total):
    if not total:
        return [np.nan, np.nan]
    return [0. if successes == 0 else stats.beta.ppf(.025, successes, total-successes+1),
            1. if successes == total else stats.beta.ppf(.975, successes+1, total-successes)]
