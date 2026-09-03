"""H5-I0 robust individual-alpha-frequency measurement validation.

H5-P0 found that its raw alpha-periodogram selector identified the 9/11-Hz
carrier in only two thirds of eligible noisy-EEG contexts.  H5-I0 repairs that
measurement bottleneck before any new stimulation-response map or learning
policy is attempted.  Every episode is stimulation-free.

Two literature-motivated estimators use averaged single-epoch log spectra,
side-band aperiodic subtraction, and either a smoothed peak or a Gaussian peak
fit.  The legacy raw-periodogram maximum is retained only as a benchmark.  A
robust estimator is selected on discovery structures and frozen before it is
evaluated on disjoint confirmation structures.  Hidden generator frequency is
used only to score estimators, never as an estimator input.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy import optimize, signal, stats


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (  # noqa: E402
    FULL,
    PARTIAL,
    _with_context_state,
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


ROOT_NAME = "h5_iaf_measurement_validation"
LEGACY = "raw_periodogram_12s"
RAW_LONG = "raw_periodogram_30s"
SMOOTHED = "smoothed_log_welch_peak"
GAUSSIAN = "gaussian_log_welch_peak"
EXPECTED_ESTIMATORS = [LEGACY, RAW_LONG, SMOOTHED, GAUSSIAN]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    return value


def _load_h5p0_source(cfg: DictConfig) -> dict[str, Any]:
    root = Path(to_absolute_path(str(cfg.analysis.source_h5p0.result_dir)))
    files = {
        "conclusion": root / "experiment_conclusion.json",
        "audit": root / "H5_P0_feasibility_audit.json",
        "screening": root / "prospective_screening.csv",
        "metrics": root / "context_controller_future_metrics.csv",
        "provenance": root / "protocol_and_provenance.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen H5-P0 source files: {missing}")
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {
        name: str(cfg.analysis.source_h5p0.expected_sha256[name])
        for name in files
    }
    if observed != expected:
        raise RuntimeError(
            f"H5-P0 source hash mismatch: expected={expected}, observed={observed}"
        )
    conclusion = json.loads(files["conclusion"].read_text())
    checks = conclusion["checks"]
    if bool(checks["frequency_identified_from_predecision_EEG"]):
        raise RuntimeError("H5-I0 requires the frozen H5-P0 carrier failure.")
    if conclusion["conclusions"][
        "H5_P0_contextual_controller_profile_opportunity"
    ] != "NOT PASSED":
        raise RuntimeError("H5-I0 expects the frozen negative H5-P0 conclusion.")
    if conclusion["conclusions"]["machine_learning_status"] != "NOT TRAINED OR TESTED":
        raise RuntimeError("H5-I0 must precede machine-learning policy fitting.")

    source_seeds: set[int] = set()
    for name in ("screening", "metrics"):
        table = pd.read_csv(files[name])
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed",
        ):
            if column in table:
                source_seeds.update(table[column].dropna().astype(int).tolist())
    return {
        "root": str(root),
        "hashes": observed,
        "source_seeds": source_seeds,
        "carrier_failure_preserved": True,
    }


def _diffusion_levels(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "label": str(value.label),
        "diffusion_rad2_per_s": float(value.diffusion_rad2_per_s),
    } for value in cfg.analysis.states.phase_diffusion_levels]


def _shared_levels(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "shared_drive_label": str(value.label),
        "shared_modulated_fraction": float(value.shared_modulated_fraction),
    } for value in cfg.analysis.states.shared_drive_levels]


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    """Return a discovery/confirmation crossed grid with nested q pairs."""
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    split_counts = [
        ("discovery", int(block.n_discovery_structure_seeds)),
        ("confirmation", int(block.n_confirmation_structure_seeds)),
    ]
    rows: list[dict[str, Any]] = []
    global_structure_index = 0
    trial_group = 0
    for split, count in split_counts:
        for split_structure_index in range(count):
            structure_index = global_structure_index
            structure_seed = base + int(block.structure_seed_offset) + structure_index
            history_seed = base + int(block.history_seed_offset) + structure_index
            for frequency_index, frequency in enumerate(
                cfg.analysis.states.frequencies_hz
            ):
                phase_seed = (
                    base + int(block.phase_seed_offset)
                    + 2 * structure_index + frequency_index
                )
                for diffusion_index, diffusion in enumerate(_diffusion_levels(cfg)):
                    paired_id = (
                        f"{split}_s{split_structure_index:02d}_"
                        f"f{int(round(float(frequency))):02d}_d{diffusion_index:02d}"
                    )
                    trial_seed = base + int(block.trial_seed_offset) + trial_group
                    noise_seed = (
                        base + int(cfg.analysis.observation_noise.seed_offset)
                        + trial_group
                    )
                    for shared_index, shared in enumerate(_shared_levels(cfg)):
                        rows.append({
                            "context_order": len(rows),
                            "split": split,
                            "split_structure_index": split_structure_index,
                            "structure_index": structure_index,
                            "structure_seed": structure_seed,
                            "history_seed": history_seed,
                            "phase_seed": phase_seed,
                            "trial_seed": trial_seed,
                            "noise_seed": noise_seed,
                            "frequency_index": frequency_index,
                            "diffusion_index": diffusion_index,
                            "shared_index": shared_index,
                            "hidden_frequency_hz": float(frequency),
                            "paired_shared_drive_context_id": paired_id,
                            "context_id": (
                                f"{paired_id}_{diffusion['label']}_"
                                f"{shared['shared_drive_label']}"
                            ),
                            **diffusion,
                            **shared,
                        })
                    trial_group += 1
            global_structure_index += 1
    return rows


def _run_context_specs(cfg: DictConfig, split: str) -> list[dict[str, Any]]:
    rows = [row for row in _context_specs(cfg) if row["split"] == split]
    if not bool(cfg.analysis.smoke_test):
        return rows
    # Four contexts cover both carriers, both diffusion levels, and both q
    # levels without turning a smoke test into a full experiment.
    limit = int(cfg.analysis.smoke_contexts_per_split)
    representative = [0, 3, 4, 7]
    first_structure = min(row["split_structure_index"] for row in rows)
    pool = [row for row in rows if row["split_structure_index"] == first_structure]
    return [pool[index] for index in representative[:limit]]


def _validate_design(cfg: DictConfig, source: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-I0 requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-I0 may not alter recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-I0 retains the H5-P0 9/11-Hz generator grid.")
    diffusion = [
        (x["label"], x["diffusion_rad2_per_s"])
        for x in _diffusion_levels(cfg)
    ]
    if diffusion != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("H5-I0 retains D={0.5,2.0} rad^2/s.")
    shared = [
        (x["shared_drive_label"], x["shared_modulated_fraction"])
        for x in _shared_levels(cfg)
    ]
    if shared != [(PARTIAL, 0.5), (FULL, 1.0)]:
        raise ValueError("H5-I0 retains shared-drive fractions {0.5,1.0}.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-I0 retains modulation depth 0.04.")
    if not bool(cfg.analysis.observation_noise.enabled):
        raise ValueError("H5-I0 must validate the noisy observed-EEG pipeline.")
    if not np.isclose(
        float(cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg),
        0.25,
    ) or not np.isclose(float(cfg.analysis.observation_noise.ar1_coefficient), 0.95):
        raise ValueError("H5-I0 must retain the frozen H5-P0 observation noise.")
    if float(cfg.env.simulation.obs_win_len) != 1000.0:
        raise ValueError("H5-I0 requires one-second online windows.")
    timeline = cfg.analysis.timeline
    if any(int(timeline[f"{name}_steps"]) <= 0 for name in (
        "burn_in", "baseline", "stimulation", "washout"
    )):
        raise ValueError("Every compatibility epoch must remain nonempty.")
    minimum_baseline = 4 if bool(cfg.analysis.smoke_test) else 30
    if int(timeline.baseline_steps) < minimum_baseline:
        raise ValueError(f"H5-I0 requires at least {minimum_baseline} baseline seconds.")
    if list(cfg.analysis.iaf.selectable_estimators) != [SMOOTHED, GAUSSIAN]:
        raise ValueError("H5-I0 robust candidate estimator set changed.")
    if list(cfg.analysis.iaf.benchmark_estimators) != [LEGACY, RAW_LONG]:
        raise ValueError("Both raw-periodogram estimators must remain benchmark-only.")
    if list(cfg.analysis.iaf.tie_break_priority) != [SMOOTHED, GAUSSIAN]:
        raise ValueError("H5-I0 estimator tie-break changed.")
    if [float(x) for x in cfg.analysis.iaf.candidate_action_frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-I0 must select only the frozen 9/11-Hz actions.")
    if not bool(cfg.analysis.smoke_test):
        if int(cfg.analysis.crossed_design.n_discovery_structure_seeds) < 3:
            raise ValueError("Full H5-I0 requires three discovery structures.")
        if int(cfg.analysis.crossed_design.n_confirmation_structure_seeds) < 6:
            raise ValueError("Full H5-I0 requires six confirmation structures.")

    contexts = _context_specs(cfg)
    namespaces = [
        {int(row["structure_seed"]) for row in contexts},
        {int(row["history_seed"]) for row in contexts},
        {int(row["phase_seed"]) for row in contexts},
        {int(row["trial_seed"]) for row in contexts},
        {int(row["noise_seed"]) for row in contexts},
    ]
    if any(not values for values in namespaces):
        raise ValueError("Every H5-I0 seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H5-I0 seed namespaces overlap.")
    if set().union(*namespaces).intersection(source["source_seeds"]):
        raise ValueError("H5-I0 seeds overlap frozen H5-P0 seeds.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H5-I0 structure seed exceeds the uint32 mapping range.")


def _unit_ar1_noise(n_samples: int, *, seed: int, coefficient: float) -> np.ndarray:
    """Generate one deterministic unit-RMS AR(1) observation-noise path."""
    if n_samples <= 0 or not 0.0 <= float(coefficient) < 1.0:
        raise ValueError("Invalid AR(1) noise request.")
    rng = np.random.default_rng(int(seed))
    innovation_scale = np.sqrt(1.0 - float(coefficient) ** 2)
    values = signal.lfilter(
        [innovation_scale], [1.0, -float(coefficient)],
        rng.standard_normal(int(n_samples)),
    )
    rms = float(np.sqrt(np.mean(values**2)))
    if not np.isfinite(rms) or rms <= np.finfo(float).tiny:
        raise RuntimeError("AR(1) observation noise has invalid RMS.")
    return values / rms


def _epoch_log_spectra(
    eeg: np.ndarray, *, fs_hz: float, cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency bins, epoch log-PSDs, and their log-domain mean."""
    values = np.asarray(eeg, dtype=float).reshape(-1)
    nperseg = int(round(float(cfg.analysis.iaf.epoch_length_s) * fs_hz))
    if nperseg > values.size:
        nperseg = values.size
    overlap = float(cfg.analysis.iaf.epoch_overlap_fraction)
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    starts = list(range(0, values.size - nperseg + 1, step))
    if not starts:
        starts = [0]
    nfft = max(nperseg, int(round(float(cfg.analysis.iaf.zero_padding_s) * fs_hz)))
    eps = np.finfo(float).tiny
    epoch_rows = []
    frequencies = None
    for start in starts:
        frequencies, psd = signal.periodogram(
            values[start:start + nperseg],
            fs=fs_hz,
            window="hann",
            detrend="constant",
            nfft=nfft,
            scaling="density",
        )
        epoch_rows.append(np.log10(np.maximum(psd, eps)))
    matrix = np.asarray(epoch_rows, dtype=float)
    return np.asarray(frequencies, dtype=float), matrix, np.mean(matrix, axis=0)


def _aperiodic_residual_db(
    frequencies: np.ndarray, log_psd: np.ndarray, cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit log power on flanking bands and return background and residual dB."""
    sidebands = [tuple(float(x) for x in pair) for pair in cfg.analysis.iaf.aperiodic_sidebands_hz]
    mask = np.zeros(frequencies.size, dtype=bool)
    for low, high in sidebands:
        mask |= (frequencies >= low) & (frequencies <= high)
    mask &= frequencies > 0.0
    if np.count_nonzero(mask) < 4:
        raise RuntimeError("Insufficient side-band bins for aperiodic fitting.")
    coefficients = np.polyfit(
        np.log10(frequencies[mask]), np.asarray(log_psd)[mask], deg=1
    )
    background = np.polyval(coefficients, np.log10(np.maximum(frequencies, 1.0e-9)))
    return background, 10.0 * (np.asarray(log_psd) - background)


def _smooth(values: np.ndarray, frequencies: np.ndarray, cfg: DictConfig) -> np.ndarray:
    spacing = float(np.median(np.diff(frequencies)))
    window = max(3, int(round(float(cfg.analysis.iaf.smoothing_width_hz) / spacing)))
    if window % 2 == 0:
        window += 1
    if window > values.size:
        window = values.size if values.size % 2 == 1 else values.size - 1
    polynomial = min(int(cfg.analysis.iaf.smoothing_polynomial_order), window - 1)
    return signal.savgol_filter(values, window_length=window, polyorder=polynomial)


def _nearest_action_frequency(peak_hz: float, cfg: DictConfig) -> float:
    candidates = np.asarray(cfg.analysis.iaf.candidate_action_frequencies_hz, dtype=float)
    return float(candidates[np.argmin(np.abs(candidates - float(peak_hz)))])


def _smoothed_peak(
    frequencies: np.ndarray, residual_db: np.ndarray, cfg: DictConfig,
) -> dict[str, float]:
    smoothed = _smooth(residual_db, frequencies, cfg)
    alpha = (
        (frequencies >= float(cfg.analysis.iaf.alpha_low_hz))
        & (frequencies <= float(cfg.analysis.iaf.alpha_high_hz))
    )
    indices = np.flatnonzero(alpha)
    local_index = int(np.argmax(smoothed[alpha]))
    index = int(indices[local_index])
    alpha_values = smoothed[alpha]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prominence = float(signal.peak_prominences(alpha_values, [local_index])[0][0])
        widths = signal.peak_widths(alpha_values, [local_index], rel_height=0.5)[0]
    if not np.isfinite(prominence) or prominence <= 0.0:
        prominence = float(alpha_values[local_index] - np.quantile(alpha_values, 0.25))
    spacing = float(np.median(np.diff(frequencies)))
    width_hz = float(widths[0] * spacing) if widths.size else float("nan")
    return {
        "peak_frequency_hz": float(frequencies[index]),
        "peak_prominence_db": prominence,
        "peak_width_hz": width_hz,
        "smoothed_residual_db": smoothed,
    }


def _gaussian_model(
    frequency: np.ndarray, offset: float, amplitude: float, center: float,
    sigma: float,
) -> np.ndarray:
    return offset + amplitude * np.exp(-0.5 * ((frequency - center) / sigma) ** 2)


def _gaussian_peak(
    frequencies: np.ndarray, smoothed_residual_db: np.ndarray, cfg: DictConfig,
) -> dict[str, float]:
    alpha = (
        (frequencies >= float(cfg.analysis.iaf.alpha_low_hz))
        & (frequencies <= float(cfg.analysis.iaf.alpha_high_hz))
    )
    x = frequencies[alpha]
    y = np.asarray(smoothed_residual_db)[alpha]
    initial_center = float(x[np.argmax(y)])
    initial_offset = float(np.quantile(y, 0.20))
    initial_amplitude = max(float(np.max(y) - initial_offset), 1.0e-3)
    fit_succeeded = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
            parameters, _ = optimize.curve_fit(
                _gaussian_model,
                x,
                y,
                p0=[initial_offset, initial_amplitude, initial_center, 0.5],
                bounds=(
                    [-100.0, 0.0, float(cfg.analysis.iaf.alpha_low_hz),
                     float(cfg.analysis.iaf.gaussian_minimum_sigma_hz)],
                    [100.0, 100.0, float(cfg.analysis.iaf.alpha_high_hz),
                     float(cfg.analysis.iaf.gaussian_maximum_sigma_hz)],
                ),
                maxfev=20_000,
            )
    except (RuntimeError, ValueError, FloatingPointError):
        fit_succeeded = False
        parameters = np.asarray([
            initial_offset, initial_amplitude, initial_center, 0.5
        ])
    fitted = _gaussian_model(x, *parameters)
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual_sum / max(total_sum, np.finfo(float).tiny)
    return {
        "peak_frequency_hz": float(parameters[2]),
        "peak_prominence_db": float(parameters[1]),
        "peak_width_hz": float(2.354820045 * parameters[3]),
        "gaussian_r2": float(r2),
        "gaussian_fit_succeeded": bool(fit_succeeded),
    }


def _window_votes(
    frequencies: np.ndarray, epoch_log_psds: np.ndarray, aggregate_selected: float,
    cfg: DictConfig,
) -> tuple[float, float, int]:
    peaks: list[float] = []
    votes: list[float] = []
    for log_psd in epoch_log_psds:
        _, residual = _aperiodic_residual_db(frequencies, log_psd, cfg)
        peak = float(_smoothed_peak(frequencies, residual, cfg)["peak_frequency_hz"])
        peaks.append(peak)
        votes.append(_nearest_action_frequency(peak, cfg))
    return (
        float(np.mean(np.isclose(votes, aggregate_selected))),
        float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0,
        len(peaks),
    )


def _estimate_iaf_methods(
    observed_eeg: np.ndarray, *, fs_hz: float, hidden_frequency_hz: float,
    cfg: DictConfig,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Evaluate all estimators from observed EEG; hidden f scores only."""
    values = np.asarray(observed_eeg, dtype=float).reshape(-1)
    frequencies, epoch_logs, mean_log_psd = _epoch_log_spectra(
        values, fs_hz=fs_hz, cfg=cfg
    )
    background, residual_db = _aperiodic_residual_db(
        frequencies, mean_log_psd, cfg
    )
    smoothed = _smoothed_peak(frequencies, residual_db, cfg)
    smooth_selected = _nearest_action_frequency(smoothed["peak_frequency_hz"], cfg)
    vote_fraction, window_sd, n_windows = _window_votes(
        frequencies, epoch_logs, smooth_selected, cfg
    )

    eps = np.finfo(float).tiny

    def raw_periodogram_row(
        estimator: str, raw_values: np.ndarray,
    ) -> dict[str, Any]:
        periodogram_frequency, periodogram_psd = signal.periodogram(
            raw_values,
            fs=fs_hz,
            window="hann",
            detrend="constant",
            scaling="density",
        )
        alpha = (
            (periodogram_frequency >= float(cfg.analysis.iaf.alpha_low_hz))
            & (periodogram_frequency <= float(cfg.analysis.iaf.alpha_high_hz))
        )
        peak = float(
            periodogram_frequency[alpha][np.argmax(periodogram_psd[alpha])]
        )
        selected = _nearest_action_frequency(peak, cfg)
        vote_frequencies, vote_logs, _ = _epoch_log_spectra(
            raw_values, fs_hz=fs_hz, cfg=cfg
        )
        vote, window_sd_value, raw_window_count = _window_votes(
            vote_frequencies, vote_logs, selected, cfg
        )
        raw_db = 10.0 * np.log10(np.maximum(periodogram_psd[alpha], eps))
        return {
            "hidden_frequency_hz": float(hidden_frequency_hz),
            "n_spectral_windows": int(raw_window_count),
            "analysis_duration_s": float(raw_values.size / fs_hz),
            "estimator": estimator,
            "peak_frequency_hz": peak,
            "selected_frequency_hz": selected,
            "peak_prominence_db": float(np.max(raw_db) - np.quantile(raw_db, 0.25)),
            "peak_width_hz": float("nan"),
            "gaussian_r2": float("nan"),
            "window_vote_fraction": vote,
            "window_peak_sd_hz": window_sd_value,
            "identified": True,
            "identification_reason": "raw_periodogram_has_no_rejection_rule",
        }

    gaussian = _gaussian_peak(
        frequencies, smoothed["smoothed_residual_db"], cfg
    )
    gaussian_selected = _nearest_action_frequency(gaussian["peak_frequency_hz"], cfg)
    gaussian_vote, gaussian_window_sd, _ = _window_votes(
        frequencies, epoch_logs, gaussian_selected, cfg
    )
    minimum_prominence = float(cfg.analysis.iaf.minimum_peak_prominence_db)
    minimum_vote = float(cfg.analysis.iaf.minimum_window_vote_fraction)
    common = {
        "hidden_frequency_hz": float(hidden_frequency_hz),
        "n_spectral_windows": int(n_windows),
        "analysis_duration_s": float(values.size / fs_hz),
    }
    rows = [
        raw_periodogram_row(
            LEGACY,
            values[-min(values.size, int(round(12.0 * fs_hz))):],
        ),
        raw_periodogram_row(RAW_LONG, values),
        {
            **common,
            "estimator": SMOOTHED,
            "peak_frequency_hz": float(smoothed["peak_frequency_hz"]),
            "selected_frequency_hz": smooth_selected,
            "peak_prominence_db": float(smoothed["peak_prominence_db"]),
            "peak_width_hz": float(smoothed["peak_width_hz"]),
            "gaussian_r2": float("nan"),
            "window_vote_fraction": vote_fraction,
            "window_peak_sd_hz": window_sd,
            "identified": bool(
                float(smoothed["peak_prominence_db"]) >= minimum_prominence
                and vote_fraction >= minimum_vote
            ),
            "identification_reason": (
                "accepted" if (
                    float(smoothed["peak_prominence_db"]) >= minimum_prominence
                    and vote_fraction >= minimum_vote
                ) else "insufficient_prominence_or_temporal_stability"
            ),
        },
        {
            **common,
            "estimator": GAUSSIAN,
            "peak_frequency_hz": float(gaussian["peak_frequency_hz"]),
            "selected_frequency_hz": gaussian_selected,
            "peak_prominence_db": float(gaussian["peak_prominence_db"]),
            "peak_width_hz": float(gaussian["peak_width_hz"]),
            "gaussian_r2": float(gaussian["gaussian_r2"]),
            "window_vote_fraction": gaussian_vote,
            "window_peak_sd_hz": gaussian_window_sd,
            "identified": bool(
                gaussian["gaussian_fit_succeeded"]
                and float(gaussian["peak_prominence_db"]) >= minimum_prominence
                and float(gaussian["gaussian_r2"])
                >= float(cfg.analysis.iaf.minimum_gaussian_r2)
                and gaussian_vote >= minimum_vote
            ),
            "identification_reason": (
                "accepted" if (
                    gaussian["gaussian_fit_succeeded"]
                    and float(gaussian["peak_prominence_db"]) >= minimum_prominence
                    and float(gaussian["gaussian_r2"])
                    >= float(cfg.analysis.iaf.minimum_gaussian_r2)
                    and gaussian_vote >= minimum_vote
                ) else "fit_quality_prominence_or_stability_failed"
            ),
        },
    ]
    for row in rows:
        row["frequency_detected_correctly"] = bool(np.isclose(
            float(row["selected_frequency_hz"]), float(hidden_frequency_hz)
        ))
        row["absolute_peak_error_hz"] = abs(
            float(row["peak_frequency_hz"]) - float(hidden_frequency_hz)
        )

    spectrum = pd.DataFrame({
        "frequency_hz": frequencies,
        "mean_log10_psd": mean_log_psd,
        "aperiodic_log10_psd": background,
        "aperiodic_residual_db": residual_db,
        "smoothed_residual_db": smoothed["smoothed_residual_db"],
    })
    return rows, spectrum


def _population_rate(outputs: list[dict[str, Any]], population: str) -> float:
    first = outputs[0]["spikes"][population]
    n_cells = len(first["per_cell"])
    n_spikes = sum(
        len(np.asarray(output["spikes"][population]["times_ms"]))
        for output in outputs
    )
    duration_s = sum(
        float(output["t_stop_ms"] - output["t_start_ms"]) for output in outputs
    ) / 1000.0
    return float(n_spikes / max(n_cells * duration_s, np.finfo(float).tiny))


def _measurement_context(
    cfg: DictConfig, context: dict[str, Any], *, output_dir: Path, comm: Any,
    size: int, rank: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame] | None:
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
    neural = np.concatenate([
        np.asarray(output["eeg_v"], dtype=float).reshape(-1) for output in outputs
    ])
    times = np.concatenate([
        np.asarray(output["sample_times_ms"], dtype=float).reshape(-1)
        for output in outputs
    ])
    if neural.size != times.size or not np.all(np.isfinite(neural)):
        raise RuntimeError("H5-I0 obtained invalid neural EEG.")
    neural_rms = float(np.sqrt(np.mean(neural**2)))
    unit_noise = _unit_ar1_noise(
        neural.size,
        seed=int(context["noise_seed"]),
        coefficient=float(cfg.analysis.observation_noise.ar1_coefficient),
    )
    noise_scale = (
        float(cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg)
        * neural_rms
    )
    observed = neural + noise_scale * unit_noise
    processed, processed_fs, _, _, _ = _process_eeg(
        observed,
        simulator_fs_hz=1000.0 / float(cfg.env.network.dt),
        cfg=cfg,
    )
    estimator_rows, spectrum = _estimate_iaf_methods(
        processed,
        fs_hz=processed_fs,
        hidden_frequency_hz=float(context["hidden_frequency_hz"]),
        cfg=cfg,
    )
    e_rate = _population_rate(outputs, "E")
    i_rate = _population_rate(outputs, "I")
    limits = cfg.analysis.rate_guardrails_hz
    rates_safe = bool(
        float(limits.E_min) <= e_rate <= float(limits.E_max)
        and float(limits.I_min) <= i_rate <= float(limits.I_max)
    )
    observed_output = {"sample_times_ms": times, "eeg_v": observed}
    boundary_ms = float(times[-1])
    for row in estimator_rows:
        frequency_cfg = _with_action_frequency(
            state_cfg, float(row["selected_frequency_hz"])
        )
        phase = _tail_phase_estimate(
            [observed_output],
            boundary_ms=boundary_ms,
            history_ms=1000.0,
            simulator_fs_hz=1000.0 / float(cfg.env.network.dt),
            relative_offset_rad=0.0,
            cfg=frequency_cfg,
        )
        row.update({
            **context,
            "input_is_predecision_observed_eeg_only": True,
            "hidden_frequency_used_only_for_scoring": True,
            "observation_noise_model": "AR1_additive_sensor_noise",
            "observation_noise_rms_fraction": float(
                np.sqrt(np.mean((noise_scale * unit_noise) ** 2))
                / max(neural_rms, np.finfo(float).tiny)
            ),
            "neural_eeg_rms_v": neural_rms,
            "observed_eeg_rms_v": float(np.sqrt(np.mean(observed**2))),
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
    spectrum = spectrum.assign(**{
        key: value for key, value in context.items()
        if key in (
            "split", "context_id", "structure_seed", "hidden_frequency_hz",
            "label", "diffusion_rad2_per_s", "shared_drive_label",
            "shared_modulated_fraction",
        )
    })
    return estimator_rows, spectrum


def _structure_metrics(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(["split", "estimator", "structure_seed"], as_index=False)
        .agg(
            context_count=("context_id", "nunique"),
            accuracy=("frequency_detected_correctly", "mean"),
            accepted_fraction=("identified", "mean"),
            mean_window_vote_fraction=("window_vote_fraction", "mean"),
            mean_absolute_peak_error_hz=("absolute_peak_error_hz", "mean"),
        )
    )


def _estimator_summary(table: pd.DataFrame) -> pd.DataFrame:
    structures = _structure_metrics(table)
    rows: list[dict[str, Any]] = []
    for estimator, group in table.groupby("estimator", sort=False):
        accepted = group[group.identified.astype(bool)]
        by_frequency = group.groupby("hidden_frequency_hz").frequency_detected_correctly.mean()
        by_diffusion = group.groupby("label").frequency_detected_correctly.mean()
        by_shared = group.groupby("shared_drive_label").frequency_detected_correctly.mean()
        structure = structures[structures.estimator.eq(estimator)]
        rows.append({
            "split": str(group.split.iloc[0]),
            "estimator": str(estimator),
            "context_count": int(group.context_id.nunique()),
            "structure_count": int(group.structure_seed.nunique()),
            "accuracy": float(group.frequency_detected_correctly.mean()),
            "accepted_fraction": float(group.identified.mean()),
            "accepted_accuracy": (
                float(accepted.frequency_detected_correctly.mean())
                if not accepted.empty else float("nan")
            ),
            "mean_absolute_peak_error_hz": float(group.absolute_peak_error_hz.mean()),
            "mean_window_vote_fraction": float(group.window_vote_fraction.mean()),
            "minimum_frequency_accuracy": float(by_frequency.min()),
            "minimum_diffusion_accuracy": float(by_diffusion.min()),
            "minimum_shared_drive_accuracy": float(by_shared.min()),
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
    criteria = cfg.analysis.iaf_selection
    candidates = summary[
        summary.estimator.isin(list(cfg.analysis.iaf.selectable_estimators))
    ].copy()
    candidates["passes_discovery_gate"] = (
        (candidates.accuracy >= float(criteria.minimum_discovery_accuracy))
        & (candidates.accepted_fraction >= float(criteria.minimum_discovery_accepted_fraction))
        & (candidates.accepted_accuracy >= float(criteria.minimum_discovery_accepted_accuracy))
        & (candidates.minimum_frequency_accuracy >= float(criteria.minimum_discovery_frequency_accuracy))
        & (candidates.minimum_structure_accuracy >= float(criteria.minimum_discovery_structure_accuracy))
    )
    priority = {
        name: index for index, name in enumerate(cfg.analysis.iaf.tie_break_priority)
    }
    candidates["tie_break_priority"] = candidates.estimator.map(priority).astype(int)
    qualified = candidates[candidates.passes_discovery_gate.astype(bool)].copy()
    if qualified.empty and not bool(cfg.analysis.smoke_test):
        return None, candidates
    pool = qualified if not qualified.empty else candidates
    ordered = pool.sort_values(
        [
            "accuracy", "accepted_accuracy", "minimum_structure_accuracy",
            "mean_absolute_peak_error_hz", "tie_break_priority",
        ],
        ascending=[False, False, False, True, True],
    )
    return str(ordered.iloc[0].estimator), candidates


def _stratified_metrics(table: pd.DataFrame, estimator: str) -> pd.DataFrame:
    selected = table[table.estimator.eq(estimator)]
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
                "mean_window_vote_fraction": float(group.window_vote_fraction.mean()),
                "phase_actionable_fraction": float(group.recent_phase_actionable.mean()),
            })
    return pd.DataFrame(rows)


def _structure_bootstrap_interval(
    values: np.ndarray, *, seed: int, n_resamples: int = 20_000,
) -> list[float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(data, size=(int(n_resamples), data.size), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def _confirmation_inference(
    table: pd.DataFrame, structure: pd.DataFrame, estimator: str, cfg: DictConfig,
) -> dict[str, Any]:
    selected_structure = structure[
        (structure.split.eq("confirmation"))
        & structure.estimator.eq(estimator)
    ].sort_values("structure_seed")
    legacy_structure = structure[
        (structure.split.eq("confirmation"))
        & structure.estimator.eq(LEGACY)
    ].set_index("structure_seed")
    differences = np.asarray([
        float(row.accuracy - legacy_structure.loc[int(row.structure_seed)].accuracy)
        for row in selected_structure.itertuples()
    ])
    nonzero = differences[~np.isclose(differences, 0.0)]
    sign_p = (
        float(stats.binomtest(
            int(np.count_nonzero(nonzero > 0.0)), len(nonzero), 0.5,
            alternative="greater",
        ).pvalue)
        if nonzero.size else 1.0
    )
    accuracies = selected_structure.accuracy.to_numpy(float)
    return {
        "selected_estimator": estimator,
        "independent_structure_count": int(accuracies.size),
        "mean_structure_accuracy": float(np.mean(accuracies)),
        "sd_structure_accuracy": float(np.std(accuracies, ddof=1)),
        "structure_bootstrap_interval_95": _structure_bootstrap_interval(
            accuracies, seed=int(cfg.experiment.seed) + 449000
        ),
        "mean_structure_accuracy_advantage_over_legacy": float(np.mean(differences)),
        "positive_structure_advantage_count": int(np.count_nonzero(differences > 0.0)),
        "nonzero_structure_sign_test_one_sided_p_value": sign_p,
        "multiplicity": (
            "Performance thresholds define validation; the paired legacy "
            "comparison is a secondary benchmark audit."
        ),
    }


def _confirmation_checks(
    discovery: pd.DataFrame, confirmation: pd.DataFrame, selection_table: pd.DataFrame,
    selected_estimator: str, structure: pd.DataFrame, stratified: pd.DataFrame,
    source: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    row = confirmation[confirmation.estimator.eq(selected_estimator)].iloc[0]
    selected_structure = structure[
        (structure.split.eq("confirmation"))
        & structure.estimator.eq(selected_estimator)
    ]
    discovery_selection = selection_table[
        selection_table.estimator.eq(selected_estimator)
    ].iloc[0]
    frequency = stratified[stratified.axis.eq("frequency")]
    diffusion = stratified[stratified.axis.eq("diffusion")]
    shared = stratified[stratified.axis.eq("shared_drive")]
    checks = {
        "source_H5P0_hash_locked_with_frequency_gate_failed": bool(
            source["carrier_failure_preserved"]
        ),
        "H5I0_seeds_disjoint_from_H5P0": True,
        "all_runs_stimulation_free": bool(
            discovery.all_stimulation_free.astype(bool).all()
            and confirmation.all_stimulation_free.astype(bool).all()
            and discovery.all_field_removed.astype(bool).all()
            and confirmation.all_field_removed.astype(bool).all()
        ),
        "complete_discovery_frequency_diffusion_shared_drive_grid": bool(
            discovery.context_count.max()
            == len(_run_context_specs(cfg, "discovery"))
        ),
        "minimum_discovery_structures": bool(
            int(discovery.structure_count.max())
            >= (1 if bool(cfg.analysis.smoke_test) else int(criteria.minimum_discovery_structure_seeds))
        ),
        "robust_estimator_selected_only_on_discovery": bool(
            selected_estimator in list(cfg.analysis.iaf.selectable_estimators)
            and bool(discovery_selection.passes_discovery_gate)
        ),
        "legacy_raw_periodogram_excluded_from_selection": bool(
            LEGACY not in list(cfg.analysis.iaf.selectable_estimators)
        ),
        "confirmation_structures_disjoint_from_discovery": bool(
            set(discovery.attrs.get("structure_seeds", set())).isdisjoint(
                confirmation.attrs.get("structure_seeds", set())
            )
        ),
        "complete_confirmation_frequency_diffusion_shared_drive_grid": bool(
            int(row.context_count)
            == len(_run_context_specs(cfg, "confirmation"))
        ),
        "minimum_confirmation_structures": bool(
            int(row.structure_count)
            >= (1 if bool(cfg.analysis.smoke_test) else int(criteria.minimum_confirmation_structure_seeds))
        ),
        "estimator_uses_only_predecision_observed_EEG": True,
        "hidden_frequency_used_only_for_scoring": True,
        "H5P0_observation_noise_frozen": bool(
            np.isclose(
                float(cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg),
                0.25,
            ) and np.isclose(float(cfg.analysis.observation_noise.ar1_coefficient), 0.95)
            and np.allclose(discovery.mean_observation_noise_rms_fraction, 0.25)
            and np.allclose(confirmation.mean_observation_noise_rms_fraction, 0.25)
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
        "subwindow_frequency_estimates_stable": bool(
            float(row.mean_window_vote_fraction)
            >= float(criteria.minimum_mean_window_vote_fraction)
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
        conclusion = {
            "H5_I0_robust_IAF_measurement": "SMOKE TEST ONLY",
            "ready_for_H5_P1_response_mapping": False,
            "machine_learning_status": "NOT TRAINED OR TESTED",
        }
    else:
        passed = all(checks.values())
        conclusion = {
            "H5_I0_robust_IAF_measurement": "CONFIRMED" if passed else "NOT CONFIRMED",
            "ready_for_H5_P1_response_mapping": bool(passed),
            "machine_learning_status": "NOT TRAINED OR TESTED",
        }
    return checks, conclusion


def _save_figures(
    root: Path, all_metrics: pd.DataFrame, summaries: pd.DataFrame,
    stratified: pd.DataFrame, spectra: pd.DataFrame, selected: str,
) -> None:
    colors = {
        LEGACY: "#777777", RAW_LONG: "#4d4d4d",
        SMOOTHED: "#1b9e77", GAUSSIAN: "#d95f02",
    }

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, split in zip(axes, ("discovery", "confirmation")):
        group = summaries[summaries.split.eq(split)]
        positions = np.arange(len(group))
        axis.bar(
            positions,
            group.accuracy,
            color=[colors[str(name)] for name in group.estimator],
            alpha=0.9,
            label="all contexts",
        )
        axis.scatter(positions, group.accepted_accuracy, color="black", marker="D", zorder=3,
                     label="accepted contexts")
        axis.scatter(positions, group.accepted_fraction, color="white", edgecolor="black",
                     marker="o", zorder=3, label="identified fraction")
        axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(positions, [str(x).replace("_", "\n") for x in group.estimator], fontsize=8)
        axis.set_ylim(0.0, 1.05)
        axis.set_title(split.capitalize())
        axis.set_ylabel("Fraction")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("H5-I0 carrier identification and identifiability")
    figure.tight_layout()
    figure.savefig(root / "figure_01_estimator_performance.png", dpi=300)
    figure.savefig(root / "figure_01_estimator_performance.pdf")
    plt.close(figure)

    confirmation = all_metrics[all_metrics.split.eq("confirmation")]
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True, sharey=True)
    for axis, estimator in zip(axes.flat, EXPECTED_ESTIMATORS):
        group = confirmation[confirmation.estimator.eq(estimator)]
        jitter = np.linspace(-0.06, 0.06, len(group))
        axis.scatter(
            group.hidden_frequency_hz + jitter,
            group.peak_frequency_hz,
            c=np.where(group.identified, "#1b9e77", "#d95f02"),
            alpha=0.75,
            s=24,
        )
        axis.plot([8, 12], [8, 12], color="black", linestyle="--", linewidth=1)
        axis.set_xlim(8.5, 11.5)
        axis.set_ylim(7.8, 12.2)
        axis.set_title(estimator.replace("_", "\n"), fontsize=9)
        axis.set_xlabel("Hidden carrier (Hz)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Estimated continuous alpha peak (Hz)")
    figure.suptitle("Disjoint confirmation peak estimates")
    figure.tight_layout()
    figure.savefig(root / "figure_02_confirmation_peak_estimates.png", dpi=300)
    figure.savefig(root / "figure_02_confirmation_peak_estimates.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    positions = np.arange(len(stratified))
    axis.bar(positions, stratified.accuracy, color="#386cb0")
    axis.scatter(positions, stratified.accepted_fraction, marker="o", color="#f0027f",
                 label="identified fraction", zorder=3)
    axis.axhline(0.8, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(
        positions,
        [f"{axis_name}\n{level}" for axis_name, level in zip(stratified.axis, stratified.level)],
        fontsize=8,
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Fraction")
    axis.set_title(f"Frozen {selected}: confirmation strata")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_03_confirmation_strata.png", dpi=300)
    figure.savefig(root / "figure_03_confirmation_strata.pdf")
    plt.close(figure)

    examples = spectra[
        (spectra.split.eq("confirmation"))
        & np.isclose(spectra.hidden_frequency_hz, 9.0)
    ]
    if not examples.empty:
        first_structure = int(examples.structure_seed.min())
        examples = examples[examples.structure_seed.eq(first_structure)]
        groups = list(examples.groupby(["label", "shared_drive_label"], sort=False))[:4]
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True, sharey=True)
        metric_lookup = all_metrics[
            all_metrics.split.eq("confirmation")
            & all_metrics.structure_seed.eq(first_structure)
        ]
        for axis, ((diffusion, shared), group) in zip(axes.flat, groups):
            keep = group.frequency_hz.between(6.0, 14.0)
            view = group[keep]
            axis.plot(view.frequency_hz, view.aperiodic_residual_db, color="0.75",
                      label="log-PSD residual")
            axis.plot(view.frequency_hz, view.smoothed_residual_db, color="#1b9e77",
                      linewidth=2, label="smoothed residual")
            context_id = str(group.context_id.iloc[0])
            context_metrics = metric_lookup[metric_lookup.context_id.eq(context_id)]
            axis.axvline(9.0, color="black", linestyle="--", label="hidden carrier")
            for estimator, color in ((SMOOTHED, colors[SMOOTHED]), (GAUSSIAN, colors[GAUSSIAN])):
                match = context_metrics[context_metrics.estimator.eq(estimator)]
                if not match.empty:
                    axis.axvline(float(match.peak_frequency_hz.iloc[0]), color=color,
                                 linestyle=":" if estimator == GAUSSIAN else "-.")
            axis.set_title(f"{diffusion}; {shared}", fontsize=9)
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("Aperiodic-adjusted power (dB)")
        axes.flat[0].legend(frameon=False, fontsize=7)
        figure.suptitle("Representative observed-EEG alpha spectra")
        figure.tight_layout()
        figure.savefig(root / "figure_04_representative_spectra.png", dpi=300)
        figure.savefig(root / "figure_04_representative_spectra.pdf")
        plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    source = _load_h5p0_source(cfg)
    _validate_design(cfg, source)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / ROOT_NAME
    exists = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H5-I0 robust IAF measurement validation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    all_metric_rows: list[dict[str, Any]] = []
    spectrum_rows: list[pd.DataFrame] = []

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
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                rows, spectrum = result
                split_rows.extend(rows)
                all_metric_rows.extend(rows)
                if (
                    int(context["structure_seed"]) == representative_structure
                    and np.isclose(float(context["hidden_frequency_hz"]), 9.0)
                ):
                    spectrum_rows.append(spectrum)
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
        frozen = {
            "selected_estimator": selected,
            "selection_split": "discovery",
            "confirmation_data_seen_at_selection": False,
            "selectable_estimators": list(cfg.analysis.iaf.selectable_estimators),
            "legacy_estimator_is_benchmark_only": True,
            "iaf_configuration": OmegaConf.to_container(cfg.analysis.iaf, resolve=True),
            "selection_criteria": OmegaConf.to_container(
                cfg.analysis.iaf_selection, resolve=True
            ),
            "discovery_gate_passed": bool(selected is not None),
        }
        (root / "frozen_iaf_estimator.json").write_text(json.dumps(
            _json_ready(frozen), indent=2, allow_nan=False
        ))
    else:
        selected = None
    selected = comm.bcast(selected, root=0)
    if selected is None:
        if rank == 0:
            conclusion = {
                "scope": "H5-I0 robust IAF discovery with no stimulation",
                "checks": {"at_least_one_robust_estimator_passes_discovery": False},
                "conclusions": {
                    "H5_I0_robust_IAF_measurement": "NOT CONFIRMED",
                    "ready_for_H5_P1_response_mapping": False,
                    "machine_learning_status": "NOT TRAINED OR TESTED",
                },
                "runtime_seconds": float(time.perf_counter() - started),
                "stopped_before_confirmation": True,
            }
            (root / "experiment_conclusion.json").write_text(json.dumps(
                conclusion, indent=2, allow_nan=False
            ))
            print("No robust IAF estimator passed discovery; confirmation not run.")
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
    inference = _confirmation_inference(
        confirmation_table, structure, selected, cfg
    )
    checks, conclusions = _confirmation_checks(
        discovery_summary,
        confirmation_summary,
        selection_table,
        selected,
        structure,
        stratified,
        source,
        cfg,
    )

    confirmation_table.to_csv(root / "confirmation_estimator_context_metrics.csv", index=False)
    confirmation_summary.to_csv(root / "confirmation_estimator_summary.csv", index=False)
    all_metrics.to_csv(root / "all_estimator_context_metrics.csv", index=False)
    all_summary.to_csv(root / "all_estimator_summary.csv", index=False)
    structure.to_csv(root / "structure_level_estimator_metrics.csv", index=False)
    stratified.to_csv(root / "confirmation_selected_estimator_strata.csv", index=False)
    spectra = pd.concat(spectrum_rows, ignore_index=True) if spectrum_rows else pd.DataFrame()
    if not spectra.empty:
        spectra.to_csv(root / "representative_observed_eeg_spectra.csv", index=False)
    (root / "confirmation_inference.json").write_text(json.dumps(
        _json_ready(inference), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H5_I0_robust_IAF_measurement_validation",
        "source_H5P0": {"root": source["root"], "hashes": source["hashes"]},
        "protocol": {
            "all_runs_stimulation_free": True,
            "burn_in_s": int(cfg.analysis.timeline.burn_in_steps),
            "observed_EEG_s": int(cfg.analysis.timeline.baseline_steps),
            "compatibility_zero_field_s": (
                int(cfg.analysis.timeline.stimulation_steps)
                + int(cfg.analysis.timeline.washout_steps)
            ),
            "discovery_structures": int(
                cfg.analysis.crossed_design.n_discovery_structure_seeds
            ),
            "confirmation_structures": int(
                cfg.analysis.crossed_design.n_confirmation_structure_seeds
            ),
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
        "selection_used_discovery_only": True,
        "hidden_frequency_used_only_for_estimator_scoring": True,
        "statistical_unit": "independent circuit structure",
        "not_a_stimulation_or_machine_learning_experiment": True,
        "not_a_clinical_IAF_validation": True,
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-I0 robust IAF measurement discovery and disjoint confirmation",
        "selected_estimator": selected,
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure",
        "inference_boundary": (
            "Measurement validation only. H5-I0 applies no stimulation and "
            "does not train or test a machine-learning policy."
        ),
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot) and not spectra.empty:
        _save_figures(root, all_metrics, all_summary, stratified, spectra, selected)

    print("\n### H5-I0 robust IAF confirmation checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(f"\nFrozen estimator: {selected}")
    print(
        "Robust IAF measurement: "
        f"{conclusions['H5_I0_robust_IAF_measurement']}"
    )
    print(
        "Ready for H5-P1 response mapping: "
        f"{'YES' if conclusions['ready_for_H5_P1_response_mapping'] else 'NO'}"
    )
    print("Machine-learning policy status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
