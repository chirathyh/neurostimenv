"""H5-I0b multitaper pooled-evidence carrier measurement validation.

H5-I0 correctly stopped before confirmation because its robust peak estimators
did not meet the prespecified identification-coverage gate.  H5-I0b preserves
that negative result by hash and changes only the stimulation-free EEG
measurement pipeline.  The circuit generator, 30-s observation, 9/11-Hz
carrier set, phase diffusion, shared-drive fractions, and AR(1) observation
noise remain frozen.

The selectable estimators use DPSS multitaper spectra and integrate
aperiodic-corrected evidence around the two available carrier actions.  One
uses the whole-record evidence; the other combines it with a robust aggregate
of graded temporal-window evidence.  The H5-I0 Gaussian log-Welch estimator is
a frozen benchmark and cannot be selected.  Selection uses discovery
structures only and is frozen before disjoint confirmation structures are
simulated.  No tACS or machine-learning policy is used in this experiment.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf
from scipy import signal, stats


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (  # noqa: E402
    FULL,
    PARTIAL,
    _with_context_state,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_iaf_measurement_validation import (  # noqa: E402
    GAUSSIAN,
    _aperiodic_residual_db,
    _context_specs as _base_context_specs,
    _estimate_iaf_methods,
    _json_ready,
    _population_rate,
    _run_context_specs as _base_run_context_specs,
    _sha256,
    _unit_ar1_noise,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _process_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (  # noqa: E402
    HIGH,
    LOW,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (  # noqa: E402
    _tail_phase_estimate,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _simulate_episode,
    _validate_online_outputs,
    _zero_action,
)


ROOT_NAME = "h5_multitaper_measurement_validation"
MT_POOLED = "multitaper_pooled_evidence"
MT_TEMPORAL = "multitaper_temporal_pooled_evidence"
EXPECTED_ESTIMATORS = [GAUSSIAN, MT_POOLED, MT_TEMPORAL]
OBSERVED = "observed_EEG"
NEURAL = "neural_only_EEG"


def _load_h5i0_source(cfg: DictConfig) -> dict[str, Any]:
    """Load and hash-lock the completed negative H5-I0 discovery."""
    root = Path(to_absolute_path(str(cfg.analysis.source_h5i0.result_dir)))
    files = {
        "conclusion": root / "experiment_conclusion.json",
        "frozen_estimator": root / "frozen_iaf_estimator.json",
        "context_metrics": root / "discovery_estimator_context_metrics.csv",
        "estimator_summary": root / "discovery_estimator_summary.csv",
        "estimator_selection": root / "discovery_estimator_selection.csv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen H5-I0 source files: {missing}")
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {
        name: str(cfg.analysis.source_h5i0.expected_sha256[name]) for name in files
    }
    if observed != expected:
        raise RuntimeError(
            f"H5-I0 source hash mismatch: expected={expected}, observed={observed}"
        )

    conclusion = json.loads(files["conclusion"].read_text())
    frozen = json.loads(files["frozen_estimator"].read_text())
    selection = pd.read_csv(files["estimator_selection"])
    if conclusion["conclusions"]["H5_I0_robust_IAF_measurement"] != "NOT CONFIRMED":
        raise RuntimeError("H5-I0b requires the frozen negative H5-I0 conclusion.")
    if not bool(conclusion.get("stopped_before_confirmation", False)):
        raise RuntimeError("H5-I0b expects H5-I0 to have stopped before confirmation.")
    if frozen.get("selected_estimator") is not None or bool(
        frozen.get("discovery_gate_passed", True)
    ):
        raise RuntimeError("H5-I0b requires the frozen H5-I0 coverage failure.")
    if bool(selection.passes_discovery_gate.astype(bool).any()):
        raise RuntimeError("H5-I0 source unexpectedly contains a passing estimator.")

    metrics = pd.read_csv(files["context_metrics"])
    source_seeds: set[int] = set()
    for column in (
        "structure_seed", "history_seed", "phase_seed", "trial_seed", "noise_seed"
    ):
        if column in metrics:
            source_seeds.update(metrics[column].dropna().astype(int).tolist())
    return {
        "root": str(root),
        "hashes": observed,
        "source_seeds": source_seeds,
        "negative_coverage_result_preserved": True,
        "legacy_gaussian_estimator": GAUSSIAN,
    }


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    return _base_context_specs(cfg)


def _run_context_specs(cfg: DictConfig, split: str) -> list[dict[str, Any]]:
    return _base_run_context_specs(cfg, split)


def _validate_design(cfg: DictConfig, source: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-I0b requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-I0b may not alter recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-I0b retains the frozen 9/11-Hz generator grid.")
    diffusion = [
        (str(x.label), float(x.diffusion_rad2_per_s))
        for x in cfg.analysis.states.phase_diffusion_levels
    ]
    if diffusion != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("H5-I0b retains D={0.5,2.0} rad^2/s.")
    shared = [
        (str(x.label), float(x.shared_modulated_fraction))
        for x in cfg.analysis.states.shared_drive_levels
    ]
    if shared != [(PARTIAL, 0.5), (FULL, 1.0)]:
        raise ValueError("H5-I0b retains shared-drive fractions {0.5,1.0}.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-I0b retains modulation depth 0.04.")
    if not bool(cfg.analysis.observation_noise.enabled) or not np.isclose(
        float(cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg),
        0.25,
    ) or not np.isclose(float(cfg.analysis.observation_noise.ar1_coefficient), 0.95):
        raise ValueError("H5-I0b must retain the H5-P0/H5-I0 observation model.")
    if float(cfg.env.simulation.obs_win_len) != 1000.0:
        raise ValueError("H5-I0b requires one-second online windows.")
    if int(cfg.analysis.timeline.baseline_steps) != 30 and not bool(
        cfg.analysis.smoke_test
    ):
        raise ValueError("Full H5-I0b requires exactly 30 seconds of observed EEG.")
    if list(cfg.analysis.multitaper.selectable_estimators) != [
        MT_POOLED, MT_TEMPORAL
    ]:
        raise ValueError("H5-I0b selectable estimator set changed.")
    if list(cfg.analysis.multitaper.benchmark_estimators) != [GAUSSIAN]:
        raise ValueError("The frozen Gaussian estimator must remain benchmark-only.")
    if list(cfg.analysis.multitaper.tie_break_priority) != [MT_POOLED, MT_TEMPORAL]:
        raise ValueError("H5-I0b estimator tie-break changed.")
    if [float(x) for x in cfg.analysis.iaf.candidate_action_frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-I0b may select only 9- or 11-Hz carrier actions.")
    if str(cfg.analysis.multitaper.evidence_kernel) != "cosine":
        raise ValueError("H5-I0b requires the frozen cosine evidence kernel.")
    if not 0.0 <= float(cfg.analysis.multitaper.trim_fraction) < 0.5:
        raise ValueError("Invalid temporal evidence trim fraction.")
    if not bool(cfg.analysis.smoke_test):
        if int(cfg.analysis.crossed_design.n_discovery_structure_seeds) != 6:
            raise ValueError("Full H5-I0b requires six discovery structures.")
        if int(cfg.analysis.crossed_design.n_confirmation_structure_seeds) != 12:
            raise ValueError("Full H5-I0b requires twelve confirmation structures.")

    contexts = _context_specs(cfg)
    namespaces = [
        {int(row[column]) for row in contexts}
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed", "noise_seed"
        )
    ]
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H5-I0b seed namespaces overlap.")
    if set().union(*namespaces).intersection(source["source_seeds"]):
        raise ValueError("H5-I0b seeds overlap frozen H5-I0 seeds.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H5-I0b structure seed exceeds the uint32 mapping range.")


def _multitaper_log_psd(
    values: np.ndarray,
    *,
    fs_hz: float,
    time_bandwidth: float,
    number_of_tapers: int,
    zero_padding_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an eigenvalue-weighted DPSS multitaper log10 PSD."""
    x = signal.detrend(np.asarray(values, dtype=float).reshape(-1), type="constant")
    if x.size < 8:
        raise ValueError("A multitaper spectrum requires at least eight samples.")
    nw = float(time_bandwidth)
    kmax = int(number_of_tapers)
    if not nw > 0.5 or not 1 <= kmax <= int(np.floor(2.0 * nw)):
        raise ValueError("Invalid DPSS time-bandwidth/taper combination.")
    tapers, ratios = signal.windows.dpss(
        x.size, NW=nw, Kmax=kmax, sym=False, return_ratios=True
    )
    nfft = max(x.size, int(round(float(zero_padding_s) * float(fs_hz))))
    psds: list[np.ndarray] = []
    frequencies = None
    for taper in np.asarray(tapers):
        frequencies, psd = signal.periodogram(
            x,
            fs=float(fs_hz),
            window=taper,
            detrend=False,
            nfft=nfft,
            scaling="density",
        )
        psds.append(psd)
    matrix = np.asarray(psds, dtype=float)
    weights = np.asarray(ratios, dtype=float)
    averaged = np.average(matrix, axis=0, weights=weights)
    if not np.all(np.isfinite(averaged)) or np.any(averaged < 0.0):
        raise RuntimeError("Multitaper PSD is invalid.")
    return np.asarray(frequencies, dtype=float), np.log10(
        np.maximum(averaged, np.finfo(float).tiny)
    )


def _band_evidence(
    frequencies: np.ndarray,
    residual_db: np.ndarray,
    *,
    center_hz: float,
    half_width_hz: float,
) -> float:
    """Cosine-weighted mean aperiodic-adjusted evidence around one action."""
    distance = np.abs(np.asarray(frequencies, dtype=float) - float(center_hz))
    mask = distance <= float(half_width_hz)
    if np.count_nonzero(mask) < 3:
        raise RuntimeError("Insufficient bins in a carrier-evidence band.")
    weights = 0.5 * (
        1.0 + np.cos(np.pi * distance[mask] / float(half_width_hz))
    )
    return float(np.average(np.asarray(residual_db, dtype=float)[mask], weights=weights))


def _evidence_from_spectrum(
    frequencies: np.ndarray, residual_db: np.ndarray, cfg: DictConfig,
) -> tuple[float, float, float]:
    half_width = float(cfg.analysis.multitaper.evidence_half_width_hz)
    evidence_9 = _band_evidence(
        frequencies, residual_db, center_hz=9.0, half_width_hz=half_width
    )
    evidence_11 = _band_evidence(
        frequencies, residual_db, center_hz=11.0, half_width_hz=half_width
    )
    return evidence_9, evidence_11, evidence_11 - evidence_9


def _temporal_multitaper_evidence(
    values: np.ndarray, *, fs_hz: float, cfg: DictConfig,
) -> pd.DataFrame:
    x = np.asarray(values, dtype=float).reshape(-1)
    requested = int(round(float(cfg.analysis.multitaper.temporal_window_s) * fs_hz))
    nperseg = min(x.size, requested)
    overlap = float(cfg.analysis.multitaper.temporal_overlap_fraction)
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    starts = list(range(0, x.size - nperseg + 1, step)) or [0]
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        stop = start + nperseg
        frequencies, log_psd = _multitaper_log_psd(
            x[start:stop],
            fs_hz=fs_hz,
            time_bandwidth=float(cfg.analysis.multitaper.temporal_time_bandwidth),
            number_of_tapers=int(cfg.analysis.multitaper.temporal_number_of_tapers),
            zero_padding_s=max(
                float(cfg.analysis.multitaper.temporal_window_s),
                float(cfg.analysis.multitaper.zero_padding_s),
            ),
        )
        _, residual = _aperiodic_residual_db(frequencies, log_psd, cfg)
        evidence_9, evidence_11, delta = _evidence_from_spectrum(
            frequencies, residual, cfg
        )
        rows.append({
            "window_index": index,
            "start_s": float(start / fs_hz),
            "stop_s": float(stop / fs_hz),
            "center_s": float((start + stop) / (2.0 * fs_hz)),
            "evidence_9_db": evidence_9,
            "evidence_11_db": evidence_11,
            "evidence_delta_11_minus_9_db": delta,
        })
    return pd.DataFrame(rows)


def _selected_from_score(score: float) -> float:
    return 11.0 if float(score) > 0.0 else 9.0


def _soft_support(deltas: np.ndarray, selected_hz: float) -> float:
    values = np.asarray(deltas, dtype=float)
    desired = 1.0 if np.isclose(float(selected_hz), 11.0) else -1.0
    weights = np.abs(values)
    denominator = float(np.sum(weights))
    if denominator <= np.finfo(float).tiny:
        return 0.5
    return float(np.sum(weights[np.sign(values) == desired]) / denominator)


def _continuous_peak_in_selected_band(
    frequencies: np.ndarray,
    residual_db: np.ndarray,
    *,
    selected_hz: float,
    cfg: DictConfig,
) -> float:
    half_width = float(cfg.analysis.multitaper.evidence_half_width_hz)
    mask = np.abs(np.asarray(frequencies) - float(selected_hz)) <= half_width
    indices = np.flatnonzero(mask)
    return float(frequencies[int(indices[np.argmax(np.asarray(residual_db)[mask])])])


def _candidate_row(
    *,
    estimator: str,
    score: float,
    evidence_9: float,
    evidence_11: float,
    frequencies: np.ndarray,
    residual_db: np.ndarray,
    temporal: pd.DataFrame,
    hidden_frequency_hz: float,
    analysis_duration_s: float,
    cfg: DictConfig,
) -> dict[str, Any]:
    selected = _selected_from_score(score)
    deltas = temporal.evidence_delta_11_minus_9_db.to_numpy(float)
    desired = 1.0 if np.isclose(selected, 11.0) else -1.0
    vote_fraction = float(np.mean(np.sign(deltas) == desired))
    support = _soft_support(deltas, selected)
    maximum_evidence = max(float(evidence_9), float(evidence_11))
    accepted = bool(
        maximum_evidence
        >= float(cfg.analysis.multitaper.minimum_residual_evidence_db)
        and abs(float(score))
        >= float(cfg.analysis.multitaper.minimum_evidence_margin_db)
        and support
        >= float(cfg.analysis.multitaper.minimum_soft_support_fraction)
    )
    peak = _continuous_peak_in_selected_band(
        frequencies, residual_db, selected_hz=selected, cfg=cfg
    )
    return {
        "hidden_frequency_hz": float(hidden_frequency_hz),
        "analysis_duration_s": float(analysis_duration_s),
        "n_spectral_windows": int(len(temporal)),
        "estimator": estimator,
        "peak_frequency_hz": peak,
        "selected_frequency_hz": selected,
        "evidence_9_db": float(evidence_9),
        "evidence_11_db": float(evidence_11),
        "evidence_delta_11_minus_9_db": float(score),
        "maximum_residual_evidence_db": maximum_evidence,
        "evidence_margin_db": abs(float(score)),
        "soft_support_fraction": support,
        "window_vote_fraction": vote_fraction,
        "window_score_sd_db": (
            float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0
        ),
        "identified": accepted,
        "identification_reason": (
            "accepted"
            if accepted
            else "insufficient_evidence_margin_or_soft_temporal_support"
        ),
        "frequency_detected_correctly": bool(
            np.isclose(selected, float(hidden_frequency_hz))
        ),
        "absolute_peak_error_hz": abs(peak - float(hidden_frequency_hz)),
    }


def _estimate_multitaper_methods(
    eeg: np.ndarray,
    *,
    fs_hz: float,
    hidden_frequency_hz: float,
    input_signal: str,
    cfg: DictConfig,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Evaluate the frozen Gaussian benchmark and both new candidates."""
    values = np.asarray(eeg, dtype=float).reshape(-1)
    frequencies, log_psd = _multitaper_log_psd(
        values,
        fs_hz=fs_hz,
        time_bandwidth=float(cfg.analysis.multitaper.full_time_bandwidth),
        number_of_tapers=int(cfg.analysis.multitaper.full_number_of_tapers),
        zero_padding_s=float(cfg.analysis.multitaper.zero_padding_s),
    )
    background, residual = _aperiodic_residual_db(frequencies, log_psd, cfg)
    evidence_9, evidence_11, whole_score = _evidence_from_spectrum(
        frequencies, residual, cfg
    )
    temporal = _temporal_multitaper_evidence(values, fs_hz=fs_hz, cfg=cfg)
    temporal_scores = temporal.evidence_delta_11_minus_9_db.to_numpy(float)
    robust_temporal_score = float(stats.trim_mean(
        temporal_scores, proportiontocut=float(cfg.analysis.multitaper.trim_fraction)
    ))
    combined_score = 0.5 * whole_score + 0.5 * robust_temporal_score

    rows = [
        _candidate_row(
            estimator=MT_POOLED,
            score=whole_score,
            evidence_9=evidence_9,
            evidence_11=evidence_11,
            frequencies=frequencies,
            residual_db=residual,
            temporal=temporal,
            hidden_frequency_hz=hidden_frequency_hz,
            analysis_duration_s=values.size / fs_hz,
            cfg=cfg,
        ),
        _candidate_row(
            estimator=MT_TEMPORAL,
            score=combined_score,
            evidence_9=evidence_9,
            evidence_11=evidence_11,
            frequencies=frequencies,
            residual_db=residual,
            temporal=temporal,
            hidden_frequency_hz=hidden_frequency_hz,
            analysis_duration_s=values.size / fs_hz,
            cfg=cfg,
        ),
    ]

    legacy_rows, legacy_spectrum = _estimate_iaf_methods(
        values,
        fs_hz=fs_hz,
        hidden_frequency_hz=hidden_frequency_hz,
        cfg=cfg,
    )
    gaussian = dict(next(row for row in legacy_rows if row["estimator"] == GAUSSIAN))
    gaussian.update({
        "evidence_9_db": float("nan"),
        "evidence_11_db": float("nan"),
        "evidence_delta_11_minus_9_db": float("nan"),
        "maximum_residual_evidence_db": float("nan"),
        "evidence_margin_db": float("nan"),
        "soft_support_fraction": float(gaussian["window_vote_fraction"]),
        "window_score_sd_db": float("nan"),
    })
    rows.insert(0, gaussian)
    for row in rows:
        row["input_signal"] = input_signal

    spectrum = pd.DataFrame({
        "frequency_hz": frequencies,
        f"{input_signal}_multitaper_log10_psd": log_psd,
        f"{input_signal}_multitaper_aperiodic_log10_psd": background,
        f"{input_signal}_multitaper_residual_db": residual,
        f"{input_signal}_welch_smoothed_residual_db": np.interp(
            frequencies,
            legacy_spectrum.frequency_hz.to_numpy(float),
            legacy_spectrum.smoothed_residual_db.to_numpy(float),
        ),
    })
    temporal = temporal.assign(input_signal=input_signal)
    return rows, spectrum, temporal


def _measurement_context(
    cfg: DictConfig,
    context: dict[str, Any],
    *,
    output_dir: Path,
    result_root: Path,
    comm: Any,
    size: int,
    rank: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame] | None:
    state_cfg = _with_context_state(cfg, context)
    simulation = _simulate_episode(
        state_cfg,
        seed=int(context["trial_seed"]),
        action=_zero_action(state_cfg),
        stimulate=False,
        output_dir=output_dir,
        comm=comm,
        size=size,
        rank=rank,
        structure_seed=int(context["structure_seed"]),
        drive_seed=int(context["history_seed"]),
    )
    if rank != 0:
        return None
    _validate_online_outputs(simulation["outputs_by_epoch"])
    outputs = simulation["outputs_by_epoch"]["baseline"]
    neural_raw = np.concatenate([
        np.asarray(output["eeg_v"], dtype=float).reshape(-1) for output in outputs
    ])
    times = np.concatenate([
        np.asarray(output["sample_times_ms"], dtype=float).reshape(-1)
        for output in outputs
    ])
    if neural_raw.size != times.size or not np.all(np.isfinite(neural_raw)):
        raise RuntimeError("H5-I0b obtained invalid neural EEG.")
    neural_rms = float(np.sqrt(np.mean(neural_raw**2)))
    unit_noise = _unit_ar1_noise(
        neural_raw.size,
        seed=int(context["noise_seed"]),
        coefficient=float(cfg.analysis.observation_noise.ar1_coefficient),
    )
    noise_scale = (
        float(cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg)
        * neural_rms
    )
    observed_raw = neural_raw + noise_scale * unit_noise
    simulator_fs = 1000.0 / float(cfg.env.network.dt)
    neural, neural_fs, _, _, _ = _process_eeg(
        neural_raw, simulator_fs_hz=simulator_fs, cfg=cfg
    )
    observed, observed_fs, _, _, _ = _process_eeg(
        observed_raw, simulator_fs_hz=simulator_fs, cfg=cfg
    )
    if not np.isclose(neural_fs, observed_fs) or neural.size != observed.size:
        raise RuntimeError("Neural and observed EEG preprocessing diverged.")
    if bool(cfg.analysis.save_processed_eeg):
        destination = result_root / "processed_eeg" / str(context["split"])
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / f"{context['context_id']}.npz",
            neural_eeg_v=neural.astype(np.float32),
            observed_eeg_v=observed.astype(np.float32),
            fs_hz=np.asarray([observed_fs], dtype=float),
        )

    observed_rows, observed_spectrum, observed_windows = _estimate_multitaper_methods(
        observed,
        fs_hz=observed_fs,
        hidden_frequency_hz=float(context["hidden_frequency_hz"]),
        input_signal=OBSERVED,
        cfg=cfg,
    )
    neural_rows, neural_spectrum, neural_windows = _estimate_multitaper_methods(
        neural,
        fs_hz=neural_fs,
        hidden_frequency_hz=float(context["hidden_frequency_hz"]),
        input_signal=NEURAL,
        cfg=cfg,
    )
    rows = observed_rows + neural_rows
    e_rate = _population_rate(outputs, "E")
    i_rate = _population_rate(outputs, "I")
    limits = cfg.analysis.rate_guardrails_hz
    rates_safe = bool(
        float(limits.E_min) <= e_rate <= float(limits.E_max)
        and float(limits.I_min) <= i_rate <= float(limits.I_max)
    )
    observed_output = {"sample_times_ms": times, "eeg_v": observed_raw}
    boundary_ms = float(times[-1])
    for row in rows:
        frequency_cfg = _with_action_frequency(
            state_cfg, float(row["selected_frequency_hz"])
        )
        phase = _tail_phase_estimate(
            [observed_output],
            boundary_ms=boundary_ms,
            history_ms=1000.0,
            simulator_fs_hz=simulator_fs,
            relative_offset_rad=0.0,
            cfg=frequency_cfg,
        )
        row.update({
            **context,
            "estimator_uses_predecision_EEG_only": True,
            "deployable_estimator_uses_observed_EEG_only": bool(
                row["input_signal"] == OBSERVED
            ),
            "hidden_frequency_used_only_for_scoring": True,
            "observation_noise_model": "AR1_additive_sensor_noise",
            "observation_noise_rms_fraction": float(
                np.sqrt(np.mean((noise_scale * unit_noise) ** 2))
                / max(neural_rms, np.finfo(float).tiny)
            ),
            "neural_eeg_rms_v": neural_rms,
            "observed_eeg_rms_v": float(np.sqrt(np.mean(observed_raw**2))),
            "recent_phase_resultant_to_rms": float(phase["resultant_to_rms"]),
            "recent_phase_actionable": bool(
                float(phase["resultant_to_rms"])
                >= float(cfg.analysis.criteria.minimum_recent_resultant_to_rms)
            ),
            "baseline_E_firing_rate_hz": e_rate,
            "baseline_I_firing_rate_hz": i_rate,
            "rates_safe": rates_safe,
            "stimulation_applied": False,
            "field_removed": bool(simulation["final_residual_mV"] == 0.0),
        })

    spectrum = observed_spectrum.merge(neural_spectrum, on="frequency_hz")
    spectrum = spectrum.assign(**{
        key: value for key, value in context.items()
        if key in (
            "split", "context_id", "structure_seed", "hidden_frequency_hz",
            "label", "diffusion_rad2_per_s", "shared_drive_label",
            "shared_modulated_fraction",
        )
    })
    windows = pd.concat([observed_windows, neural_windows], ignore_index=True)
    windows = windows.assign(**{
        key: value for key, value in context.items()
        if key in (
            "split", "context_id", "structure_seed", "hidden_frequency_hz",
            "label", "diffusion_rad2_per_s", "shared_drive_label",
            "shared_modulated_fraction",
        )
    })
    return rows, spectrum, windows


def _structure_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in table.groupby(
        ["split", "input_signal", "estimator", "structure_seed"], sort=False
    ):
        accepted = group[group.identified.astype(bool)]
        rows.append({
            "split": keys[0],
            "input_signal": keys[1],
            "estimator": keys[2],
            "structure_seed": int(keys[3]),
            "context_count": int(group.context_id.nunique()),
            "accuracy": float(group.frequency_detected_correctly.mean()),
            "accepted_fraction": float(group.identified.mean()),
            "accepted_accuracy": (
                float(accepted.frequency_detected_correctly.mean())
                if not accepted.empty else float("nan")
            ),
            "wrong_action_rate": float(
                (group.identified.astype(bool) & ~group.frequency_detected_correctly.astype(bool)).mean()
            ),
            "correct_active_rate": float(
                (group.identified.astype(bool) & group.frequency_detected_correctly.astype(bool)).mean()
            ),
            "mean_absolute_peak_error_hz": float(group.absolute_peak_error_hz.mean()),
            "mean_soft_support_fraction": float(group.soft_support_fraction.mean()),
        })
    return pd.DataFrame(rows)


def _estimator_summary(table: pd.DataFrame) -> pd.DataFrame:
    structures = _structure_metrics(table)
    rows: list[dict[str, Any]] = []
    for keys, group in table.groupby(["input_signal", "estimator"], sort=False):
        accepted = group[group.identified.astype(bool)]
        structure = structures[
            structures.input_signal.eq(keys[0]) & structures.estimator.eq(keys[1])
        ]
        rows.append({
            "split": str(group.split.iloc[0]),
            "input_signal": keys[0],
            "estimator": keys[1],
            "context_count": int(group.context_id.nunique()),
            "structure_count": int(group.structure_seed.nunique()),
            "accuracy": float(group.frequency_detected_correctly.mean()),
            "accepted_fraction": float(group.identified.mean()),
            "accepted_accuracy": (
                float(accepted.frequency_detected_correctly.mean())
                if not accepted.empty else float("nan")
            ),
            "wrong_action_rate": float(
                (group.identified.astype(bool) & ~group.frequency_detected_correctly.astype(bool)).mean()
            ),
            "correct_active_rate": float(
                (group.identified.astype(bool) & group.frequency_detected_correctly.astype(bool)).mean()
            ),
            "mean_absolute_peak_error_hz": float(group.absolute_peak_error_hz.mean()),
            "mean_soft_support_fraction": float(group.soft_support_fraction.mean()),
            "minimum_frequency_accuracy": float(
                group.groupby("hidden_frequency_hz").frequency_detected_correctly.mean().min()
            ),
            "minimum_diffusion_accuracy": float(
                group.groupby("label").frequency_detected_correctly.mean().min()
            ),
            "minimum_shared_drive_accuracy": float(
                group.groupby("shared_drive_label").frequency_detected_correctly.mean().min()
            ),
            "minimum_structure_accuracy": float(structure.accuracy.min()),
            "phase_actionable_fraction": float(group.recent_phase_actionable.mean()),
            "rate_safe_fraction": float(group.rates_safe.mean()),
            "all_stimulation_free": bool((~group.stimulation_applied.astype(bool)).all()),
            "all_field_removed": bool(group.field_removed.astype(bool).all()),
            "mean_observation_noise_rms_fraction": float(
                group.observation_noise_rms_fraction.mean()
            ),
        })
    return pd.DataFrame(rows)


def _select_discovery_estimator(
    summary: pd.DataFrame, cfg: DictConfig,
) -> tuple[str | None, pd.DataFrame]:
    criteria = cfg.analysis.selection
    candidates = summary[
        summary.input_signal.eq(OBSERVED)
        & summary.estimator.isin(list(cfg.analysis.multitaper.selectable_estimators))
    ].copy()
    candidates["passes_discovery_gate"] = (
        (candidates.accuracy >= float(criteria.minimum_discovery_accuracy))
        & (candidates.accepted_fraction >= float(criteria.minimum_discovery_accepted_fraction))
        & (candidates.accepted_accuracy >= float(criteria.minimum_discovery_accepted_accuracy))
        & (candidates.minimum_frequency_accuracy >= float(criteria.minimum_discovery_frequency_accuracy))
        & (candidates.minimum_diffusion_accuracy >= float(criteria.minimum_discovery_diffusion_accuracy))
        & (candidates.minimum_shared_drive_accuracy >= float(criteria.minimum_discovery_shared_drive_accuracy))
        & (candidates.minimum_structure_accuracy >= float(criteria.minimum_discovery_structure_accuracy))
    )
    priority = {
        name: index
        for index, name in enumerate(cfg.analysis.multitaper.tie_break_priority)
    }
    candidates["tie_break_priority"] = candidates.estimator.map(priority).astype(int)
    qualified = candidates[candidates.passes_discovery_gate.astype(bool)].copy()
    if qualified.empty and not bool(cfg.analysis.smoke_test):
        return None, candidates
    pool = qualified if not qualified.empty else candidates
    ordered = pool.sort_values(
        [
            "wrong_action_rate", "accuracy", "accepted_fraction",
            "mean_absolute_peak_error_hz", "tie_break_priority",
        ],
        ascending=[True, False, False, True, True],
    )
    return str(ordered.iloc[0].estimator), candidates


def _stratified_metrics(table: pd.DataFrame, estimator: str) -> pd.DataFrame:
    selected = table[
        table.input_signal.eq(OBSERVED) & table.estimator.eq(estimator)
    ]
    rows: list[dict[str, Any]] = []
    for axis, column in (
        ("frequency", "hidden_frequency_hz"),
        ("diffusion", "label"),
        ("shared_drive", "shared_drive_label"),
    ):
        for level, group in selected.groupby(column, sort=False):
            accepted = group[group.identified.astype(bool)]
            rows.append({
                "axis": axis,
                "level": str(level),
                "context_count": int(group.context_id.nunique()),
                "accuracy": float(group.frequency_detected_correctly.mean()),
                "accepted_fraction": float(group.identified.mean()),
                "accepted_accuracy": (
                    float(accepted.frequency_detected_correctly.mean())
                    if not accepted.empty else float("nan")
                ),
                "wrong_action_rate": float(
                    (group.identified.astype(bool) & ~group.frequency_detected_correctly.astype(bool)).mean()
                ),
                "mean_soft_support_fraction": float(group.soft_support_fraction.mean()),
                "phase_actionable_fraction": float(group.recent_phase_actionable.mean()),
            })
    return pd.DataFrame(rows)


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> list[float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(data, size=(20_000, data.size), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def _exact_sign_flip_p(values: np.ndarray) -> tuple[float, int, str]:
    data = np.asarray(values, dtype=float)
    if data.size <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=data.size)))
        null = np.mean(signs * data[None, :], axis=1)
        return float(np.mean(null >= np.mean(data) - 1.0e-15)), int(null.size), "exact"
    rng = np.random.default_rng(510001)
    signs = rng.choice((-1.0, 1.0), size=(100_000, data.size))
    null = np.mean(signs * data[None, :], axis=1)
    return float((1 + np.count_nonzero(null >= np.mean(data))) / (null.size + 1)), int(null.size), "monte_carlo"


def _confirmation_inference(
    structure: pd.DataFrame, selected: str, cfg: DictConfig,
) -> dict[str, Any]:
    view = structure[
        structure.split.eq("confirmation") & structure.input_signal.eq(OBSERVED)
    ]
    candidate = view[view.estimator.eq(selected)].sort_values("structure_seed")
    benchmark = view[view.estimator.eq(GAUSSIAN)].set_index("structure_seed")
    differences = np.asarray([
        float(row.accuracy - benchmark.loc[int(row.structure_seed)].accuracy)
        for row in candidate.itertuples()
    ])
    p_value, samples, method = _exact_sign_flip_p(differences)
    neural = structure[
        structure.split.eq("confirmation")
        & structure.input_signal.eq(NEURAL)
        & structure.estimator.eq(selected)
    ].set_index("structure_seed")
    neural_minus_observed = np.asarray([
        float(neural.loc[int(row.structure_seed)].accuracy - row.accuracy)
        for row in candidate.itertuples()
    ])
    return {
        "primary_measurement_validation_is_defined_by_prespecified_thresholds": True,
        "selected_estimator": selected,
        "independent_structure_count": int(len(candidate)),
        "mean_observed_accuracy": float(candidate.accuracy.mean()),
        "sd_observed_accuracy": float(candidate.accuracy.std(ddof=1)),
        "structure_bootstrap_interval_95_observed_accuracy": _bootstrap_interval(
            candidate.accuracy.to_numpy(float), seed=int(cfg.experiment.seed) + 510002
        ),
        "mean_accuracy_advantage_over_frozen_gaussian": float(np.mean(differences)),
        "positive_structure_advantage_count": int(np.count_nonzero(differences > 0.0)),
        "exact_sign_flip_one_sided_p_value": p_value,
        "exact_sign_flip_samples": samples,
        "exact_sign_flip_method": method,
        "mean_neural_minus_observed_accuracy": float(np.mean(neural_minus_observed)),
        "attribution_audit_only": (
            "Neural-only EEG never enters estimator selection or a deployable input."
        ),
        "multiplicity": (
            "Prespecified performance thresholds define confirmation; the paired "
            "Gaussian comparison and neural-only attribution are secondary audits."
        ),
    }


def _confirmation_checks(
    *,
    discovery_summary: pd.DataFrame,
    confirmation_summary: pd.DataFrame,
    selection_table: pd.DataFrame,
    selected: str,
    structure: pd.DataFrame,
    stratified: pd.DataFrame,
    source: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    row = confirmation_summary[
        confirmation_summary.input_signal.eq(OBSERVED)
        & confirmation_summary.estimator.eq(selected)
    ].iloc[0]
    selected_structure = structure[
        structure.split.eq("confirmation")
        & structure.input_signal.eq(OBSERVED)
        & structure.estimator.eq(selected)
    ]
    selected_discovery = selection_table[selection_table.estimator.eq(selected)].iloc[0]
    frequency = stratified[stratified.axis.eq("frequency")]
    diffusion = stratified[stratified.axis.eq("diffusion")]
    shared = stratified[stratified.axis.eq("shared_drive")]
    discovery_structures = set(
        discovery_summary.attrs.get("structure_seeds", set())
    )
    confirmation_structures = set(
        confirmation_summary.attrs.get("structure_seeds", set())
    )
    checks = {
        "source_H5I0_hash_locked_with_coverage_gate_failed": bool(
            source["negative_coverage_result_preserved"]
        ),
        "H5I0b_seeds_disjoint_from_H5I0": True,
        "complete_discovery_frequency_diffusion_shared_drive_grid": bool(
            int(discovery_summary.context_count.max())
            == len(_run_context_specs(cfg, "discovery"))
        ),
        "minimum_discovery_structures": bool(
            int(discovery_summary.structure_count.max())
            >= (1 if bool(cfg.analysis.smoke_test) else int(criteria.minimum_discovery_structure_seeds))
        ),
        "estimator_selected_only_on_discovery_observed_EEG": bool(
            selected in list(cfg.analysis.multitaper.selectable_estimators)
            and bool(selected_discovery.passes_discovery_gate or cfg.analysis.smoke_test)
        ),
        "frozen_gaussian_excluded_from_selection": bool(
            GAUSSIAN not in list(cfg.analysis.multitaper.selectable_estimators)
        ),
        "confirmation_structures_disjoint_from_discovery": bool(
            discovery_structures.isdisjoint(confirmation_structures)
        ),
        "complete_confirmation_frequency_diffusion_shared_drive_grid": bool(
            int(row.context_count) == len(_run_context_specs(cfg, "confirmation"))
        ),
        "minimum_confirmation_structures": bool(
            int(row.structure_count)
            >= (1 if bool(cfg.analysis.smoke_test) else int(criteria.minimum_confirmation_structure_seeds))
        ),
        "all_runs_stimulation_free_and_field_removed": bool(
            discovery_summary.all_stimulation_free.astype(bool).all()
            and confirmation_summary.all_stimulation_free.astype(bool).all()
            and discovery_summary.all_field_removed.astype(bool).all()
            and confirmation_summary.all_field_removed.astype(bool).all()
        ),
        "deployable_estimator_uses_only_predecision_observed_EEG": True,
        "neural_only_EEG_excluded_from_selection": bool(
            selection_table.input_signal.eq(OBSERVED).all()
        ),
        "hidden_frequency_used_only_for_scoring": True,
        "observation_noise_frozen": bool(
            np.allclose(discovery_summary.mean_observation_noise_rms_fraction, 0.25)
            and np.allclose(confirmation_summary.mean_observation_noise_rms_fraction, 0.25)
        ),
        "confirmation_carrier_accuracy": bool(
            float(row.accuracy) >= float(criteria.minimum_confirmation_accuracy)
        ),
        "confirmation_identifiable_coverage": bool(
            float(row.accepted_fraction)
            >= float(criteria.minimum_confirmation_accepted_fraction)
        ),
        "confirmation_accepted_accuracy": bool(
            float(row.accepted_accuracy)
            >= float(criteria.minimum_confirmation_accepted_accuracy)
        ),
        "confirmation_wrong_action_rate_bounded": bool(
            float(row.wrong_action_rate)
            <= float(criteria.maximum_confirmation_wrong_action_rate)
        ),
        "accuracy_preserved_at_both_frequencies": bool(
            float(frequency.accuracy.min())
            >= float(criteria.minimum_confirmation_frequency_accuracy)
        ),
        "accuracy_preserved_at_both_diffusion_levels": bool(
            float(diffusion.accuracy.min())
            >= float(criteria.minimum_confirmation_diffusion_accuracy)
        ),
        "accuracy_preserved_at_both_shared_drive_levels": bool(
            float(shared.accuracy.min())
            >= float(criteria.minimum_confirmation_shared_drive_accuracy)
        ),
        "accuracy_consistent_across_structures": bool(
            np.mean(
                selected_structure.accuracy
                >= float(criteria.minimum_structure_accuracy)
            ) >= float(criteria.minimum_structures_meeting_accuracy_fraction)
        ),
        "recent_phase_remains_actionable": bool(
            float(row.phase_actionable_fraction)
            >= float(criteria.minimum_phase_actionable_fraction)
        ),
        "neural_firing_rates_safe": bool(
            float(row.rate_safe_fraction)
            >= float(criteria.minimum_rate_safe_fraction)
        ),
    }
    if bool(cfg.analysis.smoke_test):
        conclusions = {
            "H5_I0b_multitaper_carrier_measurement": "SMOKE TEST ONLY",
            "ready_for_H5_P1_response_mapping": False,
            "machine_learning_status": "NOT TRAINED OR TESTED",
        }
    else:
        passed = all(checks.values())
        conclusions = {
            "H5_I0b_multitaper_carrier_measurement": (
                "CONFIRMED" if passed else "NOT CONFIRMED"
            ),
            "ready_for_H5_P1_response_mapping": bool(passed),
            "machine_learning_status": "NOT TRAINED OR TESTED",
        }
    return checks, conclusions


def _save_performance_figure(root: Path, summaries: pd.DataFrame) -> None:
    observed = summaries[summaries.input_signal.eq(OBSERVED)]
    splits = [split for split in ("discovery", "confirmation") if split in set(observed.split)]
    figure, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.3), squeeze=False)
    for axis, split in zip(axes.flat, splits):
        group = observed[observed.split.eq(split)]
        x = np.arange(len(group))
        axis.bar(x - 0.22, group.accuracy, width=0.22, label="all-context accuracy")
        axis.bar(x, group.accepted_fraction, width=0.22, label="coverage")
        axis.bar(x + 0.22, group.accepted_accuracy, width=0.22, label="accepted accuracy")
        axis.axhline(0.8 if split == "confirmation" else 0.75, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(x, [str(v).replace("_", "\n") for v in group.estimator], fontsize=7)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Fraction")
        axis.set_title(split.capitalize())
    axes.flat[-1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("H5-I0b carrier measurement performance")
    figure.tight_layout()
    figure.savefig(root / "figure_01_estimator_performance.png", dpi=300)
    figure.savefig(root / "figure_01_estimator_performance.pdf")
    plt.close(figure)


def _representative_contexts(table: pd.DataFrame, split: str) -> pd.DataFrame:
    view = table[table.split.eq(split)]
    if view.empty:
        return view
    structure_seed = int(view.structure_seed.min())
    return view[view.structure_seed.eq(structure_seed)].copy()


def _save_spectral_figure(
    root: Path, spectra: pd.DataFrame, metrics: pd.DataFrame, split: str,
) -> None:
    examples = _representative_contexts(spectra, split)
    if examples.empty:
        return
    keys = list(examples.groupby("context_id", sort=False))
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.0), sharex=True, sharey=True)
    metric_view = metrics[
        metrics.split.eq(split)
        & metrics.input_signal.eq(OBSERVED)
        & metrics.estimator.eq(MT_POOLED)
        & metrics.structure_seed.eq(int(examples.structure_seed.min()))
    ]
    for axis, (context_id, group) in zip(axes.flat, keys):
        view = group[group.frequency_hz.between(6.0, 14.0)]
        axis.plot(
            view.frequency_hz, view.observed_EEG_multitaper_residual_db,
            color="#2166ac", linewidth=1.8, label="observed EEG",
        )
        axis.plot(
            view.frequency_hz, view.neural_only_EEG_multitaper_residual_db,
            color="#4daf4a", linewidth=1.2, alpha=0.85, label="neural-only audit",
        )
        axis.plot(
            view.frequency_hz, view.observed_EEG_welch_smoothed_residual_db,
            color="0.55", linewidth=1.0, linestyle=":", label="Gaussian/Welch input",
        )
        axis.axvspan(8.25, 9.75, color="#fdae61", alpha=0.14)
        axis.axvspan(10.25, 11.75, color="#8073ac", alpha=0.12)
        hidden = float(group.hidden_frequency_hz.iloc[0])
        axis.axvline(hidden, color="black", linestyle="--", linewidth=1)
        match = metric_view[metric_view.context_id.eq(context_id)]
        decision = ""
        if not match.empty:
            decision = (
                f"; choose {float(match.selected_frequency_hz.iloc[0]):g} Hz"
                f" ({'accept' if bool(match.identified.iloc[0]) else 'abstain'})"
            )
        axis.set_title(
            f"true {hidden:g} Hz; {group.label.iloc[0]}\n"
            f"{group.shared_drive_label.iloc[0]}{decision}", fontsize=8,
        )
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Aperiodic-adjusted power (dB)")
    for axis in axes.flat[len(keys):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=7)
    figure.suptitle(f"Prospective representative {split} structure: alpha spectra")
    figure.tight_layout()
    figure.savefig(root / f"figure_02_{split}_representative_spectra.png", dpi=300)
    figure.savefig(root / f"figure_02_{split}_representative_spectra.pdf")
    plt.close(figure)


def _save_temporal_figure(root: Path, windows: pd.DataFrame, split: str) -> None:
    examples = _representative_contexts(
        windows[windows.input_signal.eq(OBSERVED)], split
    )
    if examples.empty:
        return
    keys = list(examples.groupby("context_id", sort=False))
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 6.7), sharex=True, sharey=True)
    for axis, (_, group) in zip(axes.flat, keys):
        axis.axhline(0.0, color="black", linewidth=1)
        axis.plot(
            group.center_s,
            group.evidence_delta_11_minus_9_db,
            marker="o",
            color="#b2182b",
            linewidth=1.2,
        )
        hidden = float(group.hidden_frequency_hz.iloc[0])
        axis.set_title(
            f"true {hidden:g} Hz; {group.label.iloc[0]}\n{group.shared_drive_label.iloc[0]}",
            fontsize=8,
        )
        axis.set_xlabel("Window centre (s)")
        axis.set_ylabel(r"Evidence $E_{11}-E_9$ (dB)")
    for axis in axes.flat[len(keys):]:
        axis.set_visible(False)
    figure.suptitle(
        f"Prospective representative {split} structure: graded temporal evidence"
    )
    figure.tight_layout()
    figure.savefig(root / f"figure_03_{split}_temporal_evidence.png", dpi=300)
    figure.savefig(root / f"figure_03_{split}_temporal_evidence.pdf")
    plt.close(figure)


def _save_context_heatmap(
    root: Path, metrics: pd.DataFrame, selected: str,
) -> None:
    view = metrics[
        metrics.split.eq("confirmation")
        & metrics.input_signal.eq(OBSERVED)
        & metrics.estimator.eq(selected)
    ].copy()
    if view.empty:
        return
    view["condition"] = view.apply(
        lambda row: (
            f"f{int(row.hidden_frequency_hz)} | D{float(row.diffusion_rad2_per_s):g}"
            f" | q{float(row.shared_modulated_fraction):g}"
        ), axis=1,
    )
    condition_order = list(dict.fromkeys(view.condition.tolist()))
    pivot = view.pivot(index="structure_seed", columns="condition", values="evidence_delta_11_minus_9_db")
    pivot = pivot.reindex(columns=condition_order)
    bound = max(float(np.nanmax(np.abs(pivot.to_numpy(float)))), 1.0e-6)
    figure, axis = plt.subplots(figsize=(12.0, 6.0))
    image = axis.imshow(pivot.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(pivot.index)), [str(x) for x in pivot.index], fontsize=8)
    axis.set_xlabel("Hidden condition (evaluation label only)")
    axis.set_ylabel("Independent structure seed")
    axis.set_title(f"Frozen {selected}: observed-EEG carrier evidence")
    figure.colorbar(image, ax=axis, label=r"$E_{11}-E_9$ (dB)")
    figure.tight_layout()
    figure.savefig(root / "figure_04_confirmation_context_evidence.png", dpi=300)
    figure.savefig(root / "figure_04_confirmation_context_evidence.pdf")
    plt.close(figure)


def _save_structure_figure(
    root: Path, structure: pd.DataFrame, selected: str,
) -> None:
    view = structure[
        structure.split.eq("confirmation") & structure.input_signal.eq(OBSERVED)
    ]
    if view.empty:
        return
    selected_rows = view[view.estimator.eq(selected)].sort_values("structure_seed")
    benchmark = view[view.estimator.eq(GAUSSIAN)].set_index("structure_seed")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for axis, metric, title in (
        (axes[0], "accuracy", "All-context accuracy"),
        (axes[1], "accepted_fraction", "Decision coverage"),
    ):
        for index, row in enumerate(selected_rows.itertuples()):
            other = float(benchmark.loc[int(row.structure_seed), metric])
            value = float(getattr(row, metric))
            axis.plot([0, 1], [other, value], color="0.72", linewidth=1)
            axis.scatter([0, 1], [other, value], color=["#777777", "#1b9e77"], s=28)
        axis.set_xticks([0, 1], ["Frozen Gaussian", "Selected multitaper"])
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Fraction")
        axis.set_title(title)
    figure.suptitle("Paired independent-structure confirmation performance")
    figure.tight_layout()
    figure.savefig(root / "figure_05_structure_level_comparison.png", dpi=300)
    figure.savefig(root / "figure_05_structure_level_comparison.pdf")
    plt.close(figure)


def _save_attribution_figure(
    root: Path, summaries: pd.DataFrame, selected: str,
) -> None:
    view = summaries[
        summaries.split.eq("confirmation") & summaries.estimator.eq(selected)
    ]
    if view.empty:
        return
    order = [OBSERVED, NEURAL]
    view = view.set_index("input_signal").reindex(order)
    x = np.arange(2)
    figure, axis = plt.subplots(figsize=(6.5, 4.3))
    axis.bar(x - 0.18, view.accuracy, width=0.36, label="all-context accuracy")
    axis.bar(x + 0.18, view.accepted_fraction, width=0.36, label="coverage")
    axis.set_xticks(x, ["Noisy observed EEG\n(deployable)", "Neural-only EEG\n(attribution audit)"])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Fraction")
    axis.set_title("Observation-noise attribution")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_06_neural_observed_attribution.png", dpi=300)
    figure.savefig(root / "figure_06_neural_observed_attribution.pdf")
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    source = _load_h5i0_source(cfg)
    _validate_design(cfg, source)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / ROOT_NAME
    exists = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H5-I0b multitaper pooled-evidence measurement validation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    all_metric_rows: list[dict[str, Any]] = []
    spectrum_rows: list[pd.DataFrame] = []
    window_rows: list[pd.DataFrame] = []

    def run_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_rows: list[dict[str, Any]] = []
        contexts = _run_context_specs(cfg, split)
        representative_structure = min(int(row["structure_seed"]) for row in contexts)
        for context in contexts:
            if rank == 0:
                print(
                    f"{split}: structure={context['structure_seed']} "
                    f"f={context['hidden_frequency_hz']:g} Hz "
                    f"D={context['diffusion_rad2_per_s']:g} "
                    f"q={context['shared_modulated_fraction']:g}"
                )
            result = _measurement_context(
                cfg,
                context,
                output_dir=root / "simulations",
                result_root=root,
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                rows, spectrum, windows = result
                split_rows.extend(rows)
                all_metric_rows.extend(rows)
                if int(context["structure_seed"]) == representative_structure:
                    spectrum_rows.append(spectrum)
                    window_rows.append(windows)
        if rank != 0:
            return pd.DataFrame(), pd.DataFrame()
        table = pd.DataFrame(split_rows)
        summary = _estimator_summary(table)
        summary.attrs["structure_seeds"] = set(table.structure_seed.astype(int))
        return table, summary

    discovery_table, discovery_summary = run_split("discovery")
    if rank == 0:
        discovery_summary.attrs["structure_seeds"] = set(
            discovery_table.structure_seed.astype(int)
        )
        selected, selection_table = _select_discovery_estimator(
            discovery_summary, cfg
        )
        discovery_table.to_csv(root / "discovery_estimator_context_metrics.csv", index=False)
        discovery_summary.to_csv(root / "discovery_estimator_summary.csv", index=False)
        selection_table.to_csv(root / "discovery_estimator_selection.csv", index=False)
        spectra = pd.concat(spectrum_rows, ignore_index=True)
        windows = pd.concat(window_rows, ignore_index=True)
        spectra.to_csv(root / "representative_discovery_spectra.csv", index=False)
        windows.to_csv(root / "representative_discovery_temporal_evidence.csv", index=False)
        frozen = {
            "selected_estimator": selected,
            "selection_split": "discovery",
            "confirmation_data_seen_at_selection": False,
            "selectable_estimators": list(cfg.analysis.multitaper.selectable_estimators),
            "frozen_gaussian_is_benchmark_only": True,
            "multitaper_configuration": OmegaConf.to_container(
                cfg.analysis.multitaper, resolve=True
            ),
            "selection_criteria": {
                name: float(cfg.analysis.selection[name])
                for name in (
                    "minimum_discovery_accuracy",
                    "minimum_discovery_accepted_fraction",
                    "minimum_discovery_accepted_accuracy",
                    "minimum_discovery_frequency_accuracy",
                    "minimum_discovery_diffusion_accuracy",
                    "minimum_discovery_shared_drive_accuracy",
                    "minimum_discovery_structure_accuracy",
                )
            },
            "discovery_gate_passed": bool(selected is not None),
        }
        (root / "frozen_multitaper_estimator.json").write_text(json.dumps(
            _json_ready(frozen), indent=2, allow_nan=False
        ))
        if bool(cfg.experiment.plot):
            _save_performance_figure(root, discovery_summary)
            _save_spectral_figure(
                root, spectra, discovery_table, "discovery"
            )
            _save_temporal_figure(root, windows, "discovery")
    else:
        selected = None
    selected = comm.bcast(selected, root=0)
    if selected is None:
        if rank == 0:
            conclusion = {
                "scope": "H5-I0b multitaper measurement discovery with no stimulation",
                "checks": {
                    "source_H5I0_hash_locked_with_coverage_gate_failed": True,
                    "at_least_one_multitaper_estimator_passes_discovery": False,
                },
                "conclusions": {
                    "H5_I0b_multitaper_carrier_measurement": "NOT CONFIRMED",
                    "ready_for_H5_P1_response_mapping": False,
                    "machine_learning_status": "NOT TRAINED OR TESTED",
                },
                "runtime_seconds": float(time.perf_counter() - started),
                "stopped_before_confirmation": True,
            }
            (root / "experiment_conclusion.json").write_text(json.dumps(
                conclusion, indent=2, allow_nan=False
            ))
            print("No multitaper estimator passed discovery; confirmation not run.")
            print(f"Results saved to: {root}")
        return

    confirmation_table, confirmation_summary = run_split("confirmation")
    if rank != 0:
        return
    confirmation_summary.attrs["structure_seeds"] = set(
        confirmation_table.structure_seed.astype(int)
    )
    discovery_summary.attrs["structure_seeds"] = set(
        discovery_table.structure_seed.astype(int)
    )
    all_metrics = pd.DataFrame(all_metric_rows)
    all_summary = pd.concat(
        [discovery_summary, confirmation_summary], ignore_index=True
    )
    structure = _structure_metrics(all_metrics)
    stratified = _stratified_metrics(confirmation_table, selected)
    inference = _confirmation_inference(structure, selected, cfg)
    checks, conclusions = _confirmation_checks(
        discovery_summary=discovery_summary,
        confirmation_summary=confirmation_summary,
        selection_table=selection_table,
        selected=selected,
        structure=structure,
        stratified=stratified,
        source=source,
        cfg=cfg,
    )

    confirmation_table.to_csv(root / "confirmation_estimator_context_metrics.csv", index=False)
    confirmation_summary.to_csv(root / "confirmation_estimator_summary.csv", index=False)
    all_metrics.to_csv(root / "all_estimator_context_metrics.csv", index=False)
    all_summary.to_csv(root / "all_estimator_summary.csv", index=False)
    structure.to_csv(root / "structure_level_estimator_metrics.csv", index=False)
    stratified.to_csv(root / "confirmation_selected_estimator_strata.csv", index=False)
    spectra = pd.concat(spectrum_rows, ignore_index=True)
    windows = pd.concat(window_rows, ignore_index=True)
    spectra.to_csv(root / "representative_all_spectra.csv", index=False)
    windows.to_csv(root / "representative_all_temporal_evidence.csv", index=False)
    (root / "confirmation_inference.json").write_text(json.dumps(
        _json_ready(inference), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H5_I0b_multitaper_pooled_evidence_measurement_validation",
        "source_H5I0": {"root": source["root"], "hashes": source["hashes"]},
        "protocol": {
            "all_runs_stimulation_free": True,
            "burn_in_s": int(cfg.analysis.timeline.burn_in_steps),
            "observed_EEG_s": int(cfg.analysis.timeline.baseline_steps),
            "compatibility_zero_field_s": (
                int(cfg.analysis.timeline.stimulation_steps)
                + int(cfg.analysis.timeline.washout_steps)
            ),
            "discovery_structures": int(cfg.analysis.crossed_design.n_discovery_structure_seeds),
            "confirmation_structures": int(cfg.analysis.crossed_design.n_confirmation_structure_seeds),
            "crossed_repeats": "9/11 Hz x low/high D x q=0.5/1.0",
        },
        "observation_model": {
            "AR1_coefficient": float(cfg.analysis.observation_noise.ar1_coefficient),
            "noise_RMS_fraction_of_neural_EEG": float(
                cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg
            ),
            "stimulation_artifact_modelled": False,
        },
        "selected_estimator": selected,
        "selection_used_discovery_observed_EEG_only": True,
        "neural_only_EEG_is_attribution_audit_only": True,
        "hidden_frequency_used_only_for_scoring": True,
        "statistical_unit": "independent circuit structure",
        "not_a_stimulation_or_machine_learning_experiment": True,
        "not_a_clinical_IAF_validation": True,
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-I0b multitaper measurement discovery and disjoint confirmation",
        "selected_estimator": selected,
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure",
        "inference_boundary": (
            "Measurement validation only. H5-I0b applies no stimulation and "
            "does not train or test a machine-learning policy."
        ),
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot):
        _save_performance_figure(root, all_summary)
        _save_spectral_figure(root, spectra, all_metrics, "confirmation")
        _save_temporal_figure(root, windows, "confirmation")
        _save_context_heatmap(root, all_metrics, selected)
        _save_structure_figure(root, structure, selected)
        _save_attribution_figure(root, all_summary, selected)

    print("\n### H5-I0b multitaper confirmation checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(f"\nFrozen estimator: {selected}")
    print(
        "Multitaper carrier measurement: "
        f"{conclusions['H5_I0b_multitaper_carrier_measurement']}"
    )
    print(
        "Ready for H5-P1 response mapping: "
        f"{'YES' if conclusions['ready_for_H5_P1_response_mapping'] else 'NO'}"
    )
    print("Machine-learning policy status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
