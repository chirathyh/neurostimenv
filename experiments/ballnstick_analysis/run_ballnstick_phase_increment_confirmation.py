"""D0b disjoint confirmation of an EEG phase-increment diffusion endpoint.

D0 validated the shared phase-diffusion generator but its preregistered global
12-second phase resultant did not classify low versus high diffusion on held-
out structures.  Post-hoc analysis of D0 nominated the mechanistically matched
endpoint

    C1 = mean_k cos(theta[k+1] - theta[k]),

where theta is the phase of a one-second EEG Fourier coefficient demodulated at
the EEG-visible 9- or 11-Hz carrier.  D0b hash-locks the failed D0 result,
freezes this endpoint and its discovery threshold, and evaluates them on new
circuit structures and afferent histories.  No threshold, state, frequency,
or success criterion is fitted here.

Every episode is stimulation-free.  D0b can validate an EEG observation for a
later D1 action map; it is not tACS evidence, a contextual bandit, or a model of
depression or treatment.
"""

from __future__ import annotations

import ast
import hashlib
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
from scipy import stats


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _epoch_raw,
    _epoch_row,
    _plain,
    _run_condition,
    _sham,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _fourier_coefficients,
    _process_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_validation import (  # noqa: E402
    _periodogram_metrics,
    _with_diffusion_state,
)


LOW = "low_diffusion"
HIGH = "high_diffusion"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_increment_from_phases(phases_rad: np.ndarray) -> dict[str, float]:
    """Return coherence of successive demodulated EEG phase increments."""
    phases = np.asarray(phases_rad, dtype=np.float64).reshape(-1)
    if phases.size < 2 or not np.all(np.isfinite(phases)):
        raise ValueError("At least two finite EEG phases are required.")
    increments = np.angle(np.exp(1j * np.diff(phases)))
    phasors = np.exp(1j * increments)
    return {
        "phase_increment_coherence_real": float(np.mean(np.cos(increments))),
        "phase_increment_coherence_abs": float(abs(np.mean(phasors))),
        "phase_increment_coherence_imag": float(np.mean(np.sin(increments))),
        "phase_increment_circular_variance": float(1.0 - abs(np.mean(phasors))),
        "phase_increment_count": int(increments.size),
    }


def _source_phase_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float64)


def _load_and_validate_source(cfg: DictConfig) -> dict[str, Any]:
    block = cfg.analysis.source_d0
    root = Path(to_absolute_path(str(block.result_dir)))
    files = {
        "conclusion": root / "experiment_conclusion.json",
        "generator": root / "frozen_phase_diffusion_generator.json",
        "metrics": root / "phase_diffusion_eeg_metrics.csv",
    }
    if any(not path.is_file() for path in files.values()):
        missing = [str(path) for path in files.values() if not path.is_file()]
        raise FileNotFoundError(f"Missing frozen D0 inputs: {missing}")
    hashes = {name: _sha256(path) for name, path in files.items()}
    expected_hashes = {
        name: str(block.expected_sha256[name]) for name in files
    }
    if hashes != expected_hashes:
        raise RuntimeError(
            f"D0 input hash mismatch. expected={expected_hashes}, observed={hashes}"
        )

    conclusion = json.loads(files["conclusion"].read_text())
    generator = json.loads(files["generator"].read_text())
    source_rows = pd.read_csv(files["metrics"])
    failed = {name for name, passed in conclusion["checks"].items() if not passed}
    expected_failed = {str(name) for name in block.expected_failed_checks}
    if failed != expected_failed:
        raise RuntimeError(
            f"Unexpected D0 failed-check set: expected={expected_failed}, observed={failed}."
        )
    if bool(conclusion["summary"]["ready_for_D1"]):
        raise RuntimeError("D0b expects the original D0 observability gate to have failed.")
    required_generator_checks = {
        "phase_diffusion_generator_distinct_from_tacs",
        "all_runs_stimulation_free",
        "shared_latent_phase_used_by_E_and_I",
        "private_poisson_event_streams_not_copied",
        "afferent_mean_rate_preserved",
        "latent_phase_increment_variance_matches_SDE",
        "latent_phase_coherence_is_ordered",
        "hidden_frequency_visible_in_ideal_EEG",
        "recent_one_second_phase_is_measurable",
        "neural_firing_rates_safe",
    }
    if any(not bool(conclusion["checks"].get(name, False)) for name in required_generator_checks):
        raise RuntimeError("A required D0 generator/safety check did not pass.")

    frozen = cfg.analysis.frozen_endpoint
    source_rows = source_rows[source_rows.label.isin([LOW, HIGH])].copy()
    if not bool(source_rows.frequency_detected_correctly.astype(bool).all()):
        raise RuntimeError(
            "D0 did not identify every discrete carrier from EEG; its frozen "
            "phase endpoint would not be EEG-only reproducible."
        )
    source_rows["C1"] = [
        _phase_increment_from_phases(_source_phase_array(value))[
            "phase_increment_coherence_real"
        ]
        for value in source_rows.causal_window_phase_rad
    ]
    low_mean = float(source_rows[source_rows.label.eq(LOW)].C1.mean())
    high_mean = float(source_rows[source_rows.label.eq(HIGH)].C1.mean())
    source_effect = low_mean - high_mean
    derived_threshold = 0.5 * (low_mean + high_mean)
    tolerance = float(frozen.threshold_tolerance)
    expected = {
        "low_mean": float(frozen.expected_source_low_mean),
        "high_mean": float(frozen.expected_source_high_mean),
        "effect": float(frozen.expected_source_low_minus_high),
        "threshold": float(frozen.classification_threshold),
    }
    observed = {
        "low_mean": low_mean,
        "high_mean": high_mean,
        "effect": source_effect,
        "threshold": derived_threshold,
    }
    if any(not np.isclose(observed[key], expected[key], atol=tolerance, rtol=0.0) for key in expected):
        raise RuntimeError(
            f"Frozen D0 endpoint values do not reproduce: expected={expected}, observed={observed}."
        )

    generator_levels = {
        str(level["label"]): float(level["diffusion_rad2_per_s"])
        for level in generator["phase_diffusion_levels"]
    }
    if (
        not np.isclose(generator_levels.get(LOW, np.nan), 0.5)
        or not np.isclose(generator_levels.get(HIGH, np.nan), 2.0)
        or [float(x) for x in generator["frequencies_hz"]] != [9.0, 11.0]
        or not np.isclose(float(generator["modulation_depth"]), 0.04)
    ):
        raise RuntimeError("Frozen D0 generator does not match the D0b state definition.")

    source_seed_sets = {
        column: set(source_rows[column].astype(int))
        for column in ("structure_seed", "drive_seed", "phase_seed", "trial_seed")
    }
    return {
        "root": str(root),
        "hashes": hashes,
        "failed_source_checks": sorted(failed),
        "source_statistics": observed,
        "source_seed_sets": source_seed_sets,
        "source_gate_passed": False,
        "endpoint_selection_status": "post_hoc_D0_discovery_frozen_before_D0b",
    }


def _levels(cfg: DictConfig) -> list[dict[str, Any]]:
    return [
        {
            "label": str(level.label),
            "diffusion_rad2_per_s": float(level.diffusion_rad2_per_s),
        }
        for level in cfg.analysis.states.phase_diffusion_levels
    ]


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    result: list[dict[str, Any]] = []
    order = 0
    for structure_index in range(int(block.n_structure_seeds)):
        structure_seed = base + int(block.structure_seed_offset) + structure_index
        for history_index in range(int(block.n_history_seeds)):
            drive_seed = (
                base + int(block.drive_seed_offset)
                + 10 * structure_index + history_index
            )
            for frequency_index, frequency_hz in enumerate(
                cfg.analysis.states.frequencies_hz
            ):
                phase_seed = (
                    base + int(block.phase_seed_offset)
                    + 20 * structure_index + 2 * history_index + frequency_index
                )
                context_id = (
                    f"s{structure_index:02d}_h{history_index:02d}_"
                    f"f{float(frequency_hz):g}"
                )
                for level in _levels(cfg):
                    result.append({
                        "condition_order": order,
                        "condition_id": f"{context_id}_{level['label']}",
                        "context_id": context_id,
                        "structure_index": structure_index,
                        "structure_seed": structure_seed,
                        "history_index": history_index,
                        "drive_seed": drive_seed,
                        "phase_seed": phase_seed,
                        "trial_seed": base + int(block.trial_seed_offset) + order,
                        "frequency_hz": float(frequency_hz),
                        **level,
                    })
                    order += 1
    return result


def _power_design(cfg: DictConfig) -> dict[str, Any]:
    block = cfg.analysis.power_design
    alpha = float(block.alpha_one_sided)
    target_power = float(block.target_power)
    minimum_effect = float(block.minimum_mean_low_minus_high)
    anticipated_sd = float(block.anticipated_structure_sd)
    effect_dz = minimum_effect / anticipated_sd

    def power(n: int) -> float:
        critical = float(stats.t.ppf(1.0 - alpha, n - 1))
        return float(stats.nct.sf(critical, n - 1, effect_dz * np.sqrt(n)))

    required = next(n for n in range(2, 1001) if power(n) >= target_power)
    planned = int(block.planned_independent_structures)
    return {
        "alpha_one_sided": alpha,
        "target_power": target_power,
        "minimum_mean_low_minus_high": minimum_effect,
        "anticipated_structure_sd": anticipated_sd,
        "minimum_standardized_effect_dz": effect_dz,
        "planned_independent_structures": planned,
        "required_independent_structures": required,
        "a_priori_t_approximation_power": power(planned),
        "power_unit": "independent circuit structure",
        "histories_and_frequencies_are_repeated_measures": True,
    }


def _validate_design(
    cfg: DictConfig, source: dict[str, Any], power: dict[str, Any]
) -> bool:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("D0b requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("D0b may not change recurrent inhibition.")
    if not np.isclose(float(cfg.analysis.tacs.amplitude_v_per_m), 0.0):
        raise ValueError("D0b is stimulation-free.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("D0b freezes the frequency grid to 9 and 11 Hz.")
    observed_levels = [(x["label"], x["diffusion_rad2_per_s"]) for x in _levels(cfg)]
    expected_levels = [(LOW, 0.5), (HIGH, 2.0)]
    if len(observed_levels) != 2 or any(
        label != expected_label or not np.isclose(value, expected_value)
        for (label, value), (expected_label, expected_value)
        in zip(observed_levels, expected_levels)
    ):
        raise ValueError("D0b freezes low/high diffusion to 0.5/2 rad^2/s.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("D0b freezes afferent modulation depth to 0.04.")
    if not np.isclose(float(cfg.analysis.frozen_endpoint.phase_window_s), 1.0):
        raise ValueError("D0b freezes one-second phase windows.")

    timeline = cfg.analysis.timeline
    minimum_baseline = 4 if bool(cfg.analysis.smoke_test) else 12
    if int(timeline.baseline_steps) < minimum_baseline:
        raise ValueError(f"This D0b mode requires at least {minimum_baseline} EEG seconds.")
    if any(int(timeline[name]) < 1 for name in (
        "burn_in_steps", "baseline_steps", "stimulation_steps", "washout_steps"
    )):
        raise ValueError("Every persistent episode epoch requires at least one window.")
    n_structures = int(cfg.analysis.crossed_design.n_structure_seeds)
    n_histories = int(cfg.analysis.crossed_design.n_history_seeds)
    if bool(cfg.analysis.smoke_test):
        if n_structures < 1 or n_histories < 1:
            raise ValueError("The smoke requires at least one structure and history.")
    else:
        if n_structures != int(cfg.analysis.criteria.minimum_structure_seeds):
            raise ValueError("Full D0b freezes exactly six independent structures.")
        if n_histories != int(cfg.analysis.criteria.minimum_history_seeds):
            raise ValueError("Full D0b freezes exactly two histories per structure.")
        if n_structures != int(power["planned_independent_structures"]):
            raise ValueError("The run does not match its a priori power design.")

    specs = _context_specs(cfg)
    expected_count = n_structures * n_histories * 2 * 2
    if len(specs) != expected_count:
        raise ValueError("The crossed D0b grid is incomplete.")
    confirmation_sets = {
        column: {int(row[column]) for row in specs}
        for column in ("structure_seed", "drive_seed", "phase_seed", "trial_seed")
    }
    if any(
        confirmation_sets[left].intersection(confirmation_sets[right])
        for index, left in enumerate(confirmation_sets)
        for right in list(confirmation_sets)[index + 1 :]
    ):
        raise ValueError("D0b seed namespaces must be mutually disjoint.")
    if max(confirmation_sets["structure_seed"]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Structure seed * 10,000 exceeds the uint32 range.")
    source_union = set().union(*source["source_seed_sets"].values())
    confirmation_union = set().union(*confirmation_sets.values())
    disjoint = not bool(source_union.intersection(confirmation_union))
    if not disjoint:
        raise ValueError("D0b seeds overlap the frozen D0 discovery seeds.")
    return disjoint


def _window_phase_metrics(
    processed: np.ndarray,
    *,
    fs_hz: float,
    start_ms: float,
    frequency_hz: float,
    phase_window_s: float,
    temporal_chunk_s: float,
) -> dict[str, Any]:
    samples_per_window = int(round(float(phase_window_s) * fs_hz))
    n_windows = processed.size // samples_per_window
    if n_windows < 2:
        raise ValueError("At least two phase windows are required.")
    usable = processed[: n_windows * samples_per_window]
    coefficients: list[complex] = []
    ratios: list[float] = []
    for index in range(n_windows):
        segment = usable[index * samples_per_window : (index + 1) * samples_per_window]
        segment_start = start_ms + 1000.0 * index * samples_per_window / fs_hz
        cosine, sine = _fourier_coefficients(
            segment,
            fs_hz=fs_hz,
            start_ms=segment_start,
            frequency_hz=frequency_hz,
        )
        coefficient = complex(cosine, sine)
        coefficients.append(coefficient)
        ratios.append(
            abs(coefficient)
            / max(float(np.sqrt(np.mean(segment**2))), np.finfo(float).tiny)
        )
    coefficient_array = np.asarray(coefficients, dtype=np.complex128)
    phases = np.angle(coefficient_array)
    result = _phase_increment_from_phases(phases)
    windows_per_chunk = max(2, int(round(float(temporal_chunk_s) / phase_window_s)))
    chunk_values: list[float] = []
    for begin in range(0, n_windows - windows_per_chunk + 1, windows_per_chunk):
        chunk = phases[begin : begin + windows_per_chunk]
        chunk_values.append(
            _phase_increment_from_phases(chunk)["phase_increment_coherence_real"]
        )
    result.update({
        "phase_window_s": float(phase_window_s),
        "n_phase_windows": int(n_windows),
        "phase_window_rad": phases.tolist(),
        "recent_resultant_to_rms": float(ratios[-1]),
        "mean_resultant_to_rms": float(np.mean(ratios)),
        "temporal_chunk_C1_mean": float(np.mean(chunk_values)),
        "temporal_chunk_C1_sd": (
            float(np.std(chunk_values, ddof=1)) if len(chunk_values) > 1 else 0.0
        ),
        "temporal_chunk_C1_values": chunk_values,
    })
    return result


def _episode_metrics(
    episode: dict[str, Any], spec: dict[str, Any], cfg: DictConfig
) -> dict[str, Any]:
    raw = _epoch_raw(episode, "baseline")
    processed, fs_hz, _, _, _ = _process_eeg(
        raw,
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        cfg=cfg,
    )
    outputs = episode["simulation"]["outputs_by_epoch"]["baseline"]
    start_ms = float(outputs[0]["t_start_ms"])
    hidden_frequency = float(spec["frequency_hz"])
    peak_audit, _, _ = _periodogram_metrics(
        processed,
        fs_hz=fs_hz,
        frequency_hz=hidden_frequency,
        cfg=cfg,
    )
    candidate_frequencies = np.asarray(
        cfg.analysis.states.frequencies_hz, dtype=np.float64
    )
    # The phase endpoint is deployable from EEG: first map the observed alpha
    # peak to the frozen 9/11-Hz candidate grid, then demodulate at that EEG-
    # selected carrier. The hidden generator frequency is used only to audit
    # whether the selector was correct.
    selected_frequency = float(candidate_frequencies[np.argmin(
        np.abs(candidate_frequencies - float(peak_audit["detected_peak_frequency_hz"]))
    )])
    spectral, _, _ = _periodogram_metrics(
        processed,
        fs_hz=fs_hz,
        frequency_hz=selected_frequency,
        cfg=cfg,
    )
    primary = _window_phase_metrics(
        processed,
        fs_hz=fs_hz,
        start_ms=start_ms,
        frequency_hz=selected_frequency,
        phase_window_s=float(cfg.analysis.frozen_endpoint.phase_window_s),
        temporal_chunk_s=float(cfg.analysis.measurement.temporal_chunk_s),
    )
    row: dict[str, Any] = {
        **spec,
        **spectral,
        **{
            f"primary_{key}": value for key, value in primary.items()
        },
        "analysis_duration_s": float(processed.size / fs_hz),
        "field_amplitude_v_per_m": 0.0,
        "EEG_selected_frequency_hz": selected_frequency,
        "EEG_frequency_selection_correct": float(
            np.isclose(selected_frequency, hidden_frequency)
        ),
        "phase_endpoint_uses_hidden_frequency": False,
    }
    for horizon in cfg.analysis.measurement.phase_horizon_audit_s:
        horizon_value = float(horizon)
        token = f"{horizon_value:g}".replace(".", "p")
        metrics = _window_phase_metrics(
            processed,
            fs_hz=fs_hz,
            start_ms=start_ms,
            frequency_hz=selected_frequency,
            phase_window_s=horizon_value,
            temporal_chunk_s=float(cfg.analysis.measurement.temporal_chunk_s),
        )
        for key in (
            "phase_increment_coherence_real",
            "phase_increment_coherence_abs",
            "phase_increment_coherence_imag",
            "recent_resultant_to_rms",
            "mean_resultant_to_rms",
        ):
            row[f"horizon_{token}s_{key}"] = metrics[key]
    epoch = _epoch_row(episode, "baseline")
    row.update({
        "E_firing_rate_hz": float(epoch.E_firing_rate_hz),
        "I_firing_rate_hz": float(epoch.I_firing_rate_hz),
    })
    return row


def _exact_sign_flip(
    values: np.ndarray, cfg: DictConfig
) -> tuple[float, str, int]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0 or not np.all(np.isfinite(values)):
        return float("nan"), "unavailable", 0
    observed = float(values.mean())
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 16.0
    if n <= int(cfg.analysis.inference.exact_sign_flip_max_structures):
        indices = np.arange(1 << n, dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(n, dtype=np.uint64)) & np.uint64(1)
        signs = np.where(bits == 0, -1.0, 1.0)
        null = (signs @ values) / n
        return float(np.mean(null >= observed - tolerance)), "exact", int(null.size)
    rng = np.random.default_rng(
        int(cfg.experiment.seed) + int(cfg.analysis.inference.random_seed_offset)
    )
    count = int(cfg.analysis.inference.monte_carlo_sign_flips)
    signs = rng.choice((-1.0, 1.0), size=(count, n))
    null = (signs @ values) / n
    return (
        float((1 + np.count_nonzero(null >= observed - tolerance)) / (count + 1)),
        "monte_carlo",
        count,
    )


def _bootstrap_ci(values: np.ndarray, cfg: DictConfig) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(
        int(cfg.experiment.seed) + int(cfg.analysis.inference.random_seed_offset)
    )
    samples = rng.choice(
        values,
        size=(int(cfg.analysis.inference.bootstrap_resamples), values.size),
        replace=True,
    )
    return tuple(float(x) for x in np.quantile(samples.mean(axis=1), [0.025, 0.975]))


def _paired_tables(
    rows: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classification = rows.copy()
    classification["predicted_label"] = np.where(
        classification.primary_phase_increment_coherence_real >= threshold,
        LOW,
        HIGH,
    )
    classification["classification_correct"] = (
        classification.predicted_label == classification.label
    )
    index = [
        "context_id", "structure_index", "structure_seed", "history_index",
        "drive_seed", "phase_seed", "frequency_hz",
    ]
    pivot = rows.pivot(index=index, columns="label")
    paired = pd.DataFrame(index=pivot.index).reset_index()
    for feature in (
        "primary_phase_increment_coherence_real",
        "primary_temporal_chunk_C1_mean",
        "spectral_concentration",
        "spectral_rms_width_hz",
        "E_firing_rate_hz",
        "I_firing_rate_hz",
    ):
        paired[f"low_{feature}"] = pivot[feature][LOW].to_numpy(float)
        paired[f"high_{feature}"] = pivot[feature][HIGH].to_numpy(float)
        paired[f"low_minus_high_{feature}"] = (
            paired[f"low_{feature}"] - paired[f"high_{feature}"]
        )
    structure_classification = classification.groupby(
        ["structure_index", "structure_seed"]
    ).classification_correct.mean()
    structure = paired.groupby(
        ["structure_index", "structure_seed"], as_index=False
    ).agg(
        repeated_context_count=("context_id", "size"),
        mean_low_C1=("low_primary_phase_increment_coherence_real", "mean"),
        mean_high_C1=("high_primary_phase_increment_coherence_real", "mean"),
        primary_low_minus_high_C1=(
            "low_minus_high_primary_phase_increment_coherence_real", "mean"
        ),
        local_stability_low_minus_high=(
            "low_minus_high_primary_temporal_chunk_C1_mean", "mean"
        ),
        spectral_concentration_low_minus_high=(
            "low_minus_high_spectral_concentration", "mean"
        ),
        spectral_width_low_minus_high=(
            "low_minus_high_spectral_rms_width_hz", "mean"
        ),
    )
    structure["classification_accuracy"] = [
        float(structure_classification.loc[(row.structure_index, row.structure_seed)])
        for row in structure.itertuples()
    ]
    return paired, classification, structure


def _inference(structure: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    values = structure.primary_low_minus_high_C1.to_numpy(float)
    n = int(values.size)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    if n > 1 and se > 0.0:
        t_statistic = mean / se
        t_p = float(stats.t.sf(t_statistic, df=n - 1))
        ci95 = [
            mean - float(stats.t.ppf(0.975, n - 1)) * se,
            mean + float(stats.t.ppf(0.975, n - 1)) * se,
        ]
    else:
        t_statistic = t_p = float("nan")
        ci95 = [float("nan"), float("nan")]
    sign_p, sign_method, sign_samples = _exact_sign_flip(values, cfg)
    bootstrap = _bootstrap_ci(values, cfg)
    try:
        wilcoxon = stats.wilcoxon(values, alternative="greater", method="auto")
        wilcoxon_stat, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_stat = wilcoxon_p = float("nan")
    return {
        "independent_structure_count": n,
        "mean_primary_low_minus_high_C1": mean,
        "sd_primary_low_minus_high_C1": sd,
        "se_primary_low_minus_high_C1": se,
        "paired_standardized_effect_dz": mean / sd if sd > 0.0 else float("nan"),
        "positive_structure_count": int(np.count_nonzero(values > 0.0)),
        "positive_structure_fraction": float(np.mean(values > 0.0)),
        "paired_t_statistic": t_statistic,
        "paired_t_one_sided_p_value": t_p,
        "t_interval_95": [float(x) for x in ci95],
        "structure_bootstrap_interval_95": [float(x) for x in bootstrap],
        "exact_sign_flip_one_sided_p_value": sign_p,
        "exact_sign_flip_method": sign_method,
        "exact_sign_flip_samples": sign_samples,
        "wilcoxon_signed_rank_statistic": wilcoxon_stat,
        "wilcoxon_one_sided_p_value": wilcoxon_p,
        "multiplicity": "one prespecified primary contrast; frequency tests are FDR-controlled secondary audits",
    }


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _frequency_audits(paired: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    output = []
    for frequency, group in paired.groupby("frequency_hz"):
        structure_values = group.groupby("structure_seed")[
            "low_minus_high_primary_phase_increment_coherence_real"
        ].mean().to_numpy(float)
        p_value, method, samples = _exact_sign_flip(structure_values, cfg)
        output.append({
            "frequency_hz": float(frequency),
            "independent_structures": int(structure_values.size),
            "mean_low_minus_high_C1": float(structure_values.mean()),
            "positive_structure_fraction": float(np.mean(structure_values > 0.0)),
            "one_sided_sign_flip_p": p_value,
            "sign_flip_method": method,
            "sign_flip_samples": samples,
        })
    frame = pd.DataFrame(output)
    frame["BH_FDR_q"] = _bh_fdr(frame.one_sided_sign_flip_p.to_numpy(float))
    return frame


def _evaluate(
    *,
    rows: pd.DataFrame,
    paired: pd.DataFrame,
    classification: pd.DataFrame,
    structure: pd.DataFrame,
    inference: dict[str, Any],
    source_disjoint: bool,
    power: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    low_correct = float(
        classification[classification.label.eq(LOW)].classification_correct.mean()
    )
    high_correct = float(
        classification[classification.label.eq(HIGH)].classification_correct.mean()
    )
    balanced_accuracy = 0.5 * (low_correct + high_correct)
    above_chance_structure_fraction = float(
        np.mean(structure.classification_accuracy > 0.5)
    )
    temporal_noise = float(np.sqrt(np.mean(
        rows.primary_temporal_chunk_C1_sd.to_numpy(float) ** 2
    )))
    signal = float(
        paired.low_minus_high_primary_phase_increment_coherence_real.mean()
    )
    signal_to_noise = signal / max(temporal_noise, np.finfo(float).eps)
    limits = cfg.analysis.rate_guardrails_hz
    rates_safe = bool(
        rows.E_firing_rate_hz.between(float(limits.E_min), float(limits.E_max)).all()
        and rows.I_firing_rate_hz.between(float(limits.I_min), float(limits.I_max)).all()
    )
    threshold = float(cfg.analysis.frozen_endpoint.classification_threshold)
    mean_effect = float(inference["mean_primary_low_minus_high_C1"])
    sign_p = float(inference["exact_sign_flip_one_sided_p_value"])
    positive_fraction = float(inference["positive_structure_fraction"])
    checks = {
        "source_D0_hash_locked_with_original_gate_failed": True,
        "endpoint_and_threshold_loaded_without_refitting": True,
        "confirmation_seeds_disjoint_from_D0": bool(source_disjoint),
        "a_priori_structure_sample_size_powered": bool(
            int(power["planned_independent_structures"])
            >= int(power["required_independent_structures"])
            and float(power["a_priori_t_approximation_power"])
            >= float(power["target_power"])
        ),
        "all_runs_stimulation_free": bool(np.allclose(rows.field_amplitude_v_per_m, 0.0)),
        "complete_crossed_frequency_diffusion_history_grid": bool(
            len(rows) == len(_context_specs(cfg))
        ),
        "minimum_independent_structures": bool(
            rows.structure_seed.nunique() >= int(criteria.minimum_structure_seeds)
        ),
        "minimum_independent_histories_per_structure": bool(
            rows.history_index.nunique() >= int(criteria.minimum_history_seeds)
        ),
        "paired_low_high_common_random_number_design": bool(
            paired.groupby("context_id").size().eq(1).all()
            and len(paired) * 2 == len(rows)
        ),
        "hidden_frequency_visible_in_ideal_EEG": bool(
            rows.EEG_frequency_selection_correct.mean()
            >= float(criteria.minimum_frequency_detection_accuracy)
        ),
        "recent_one_second_phase_measurable": bool(
            np.mean(
                rows.primary_recent_resultant_to_rms
                >= float(criteria.minimum_recent_resultant_to_rms)
            ) >= float(criteria.minimum_recent_phase_measurable_fraction)
        ),
        "primary_mean_effect_practically_meaningful": bool(
            mean_effect >= float(criteria.minimum_mean_low_minus_high)
        ),
        "primary_exact_structure_test_rejects_null": bool(
            sign_p <= float(criteria.maximum_primary_one_sided_p_value)
        ),
        "primary_effect_consistent_across_structures": bool(
            positive_fraction >= float(criteria.minimum_positive_structure_fraction)
        ),
        "frozen_threshold_balanced_accuracy_confirmed": bool(
            balanced_accuracy
            >= float(criteria.minimum_frozen_threshold_balanced_accuracy)
        ),
        "classification_generalizes_across_structures": bool(
            above_chance_structure_fraction
            >= float(criteria.minimum_structures_above_chance_classification_fraction)
        ),
        "diffusion_signal_exceeds_within_trajectory_temporal_noise": bool(
            signal_to_noise >= float(criteria.minimum_state_to_temporal_noise_ratio)
        ),
        "neural_firing_rates_safe": rates_safe,
    }
    primary_names = [
        "source_D0_hash_locked_with_original_gate_failed",
        "endpoint_and_threshold_loaded_without_refitting",
        "confirmation_seeds_disjoint_from_D0",
        "a_priori_structure_sample_size_powered",
        "all_runs_stimulation_free",
        "complete_crossed_frequency_diffusion_history_grid",
        "minimum_independent_structures",
        "minimum_independent_histories_per_structure",
        "paired_low_high_common_random_number_design",
        "hidden_frequency_visible_in_ideal_EEG",
        "recent_one_second_phase_measurable",
        "primary_mean_effect_practically_meaningful",
        "primary_exact_structure_test_rejects_null",
        "primary_effect_consistent_across_structures",
        "frozen_threshold_balanced_accuracy_confirmed",
        "classification_generalizes_across_structures",
        "diffusion_signal_exceeds_within_trajectory_temporal_noise",
        "neural_firing_rates_safe",
    ]
    confirmed = bool(all(checks[name] for name in primary_names))
    horizons = {}
    for horizon in cfg.analysis.measurement.phase_horizon_audit_s:
        token = f"{float(horizon):g}".replace(".", "p")
        column = f"horizon_{token}s_phase_increment_coherence_real"
        horizons[f"{float(horizon):g}_s"] = {
            label: float(value)
            for label, value in rows.groupby("label")[column].mean().items()
        }
    summary = {
        "frozen_threshold": threshold,
        "low_sensitivity": low_correct,
        "high_specificity": high_correct,
        "balanced_accuracy": balanced_accuracy,
        "structures_above_chance_classification_fraction": above_chance_structure_fraction,
        "state_to_temporal_noise_ratio": signal_to_noise,
        "frequency_detection_accuracy": float(
            rows.EEG_frequency_selection_correct.mean()
        ),
        "recent_phase_measurable_fraction": float(np.mean(
            rows.primary_recent_resultant_to_rms
            >= float(criteria.minimum_recent_resultant_to_rms)
        )),
        "mean_secondary_local_stability_low_minus_high": float(
            paired.low_minus_high_primary_temporal_chunk_C1_mean.mean()
        ),
        "mean_secondary_spectral_concentration_low_minus_high": float(
            paired.low_minus_high_spectral_concentration.mean()
        ),
        "phase_horizon_audit": horizons,
        "D0b_phase_increment_observability_confirmed": confirmed,
        "ready_for_D1_system_identification": bool(
            confirmed and not bool(cfg.analysis.smoke_test)
        ),
        "smoke_test": bool(cfg.analysis.smoke_test),
    }
    return checks, summary


def _plot_results(
    root: Path,
    rows: pd.DataFrame,
    paired: pd.DataFrame,
    classification: pd.DataFrame,
    structure: pd.DataFrame,
    cfg: DictConfig,
) -> None:
    threshold = float(cfg.analysis.frozen_endpoint.classification_threshold)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for _, group in paired.groupby(["structure_seed", "history_index", "frequency_hz"]):
        axis.plot(
            [0, 1],
            [group.low_primary_phase_increment_coherence_real.iloc[0],
             group.high_primary_phase_increment_coherence_real.iloc[0]],
            color="0.75",
            alpha=0.65,
        )
    means = rows.groupby("label").primary_phase_increment_coherence_real.mean()
    axis.plot([0, 1], [means[LOW], means[HIGH]], "o-", linewidth=2.5, color="#1F77B4")
    axis.set_xticks([0, 1], ["low diffusion", "high diffusion"])
    axis.set(
        ylabel="Frozen EEG phase-increment coherence C1",
        title="D0b paired low–high diffusion effects",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_01_paired_phase_increment_effects.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    jitter = np.where(classification.label.eq(LOW), -0.08, 0.08)
    axis.scatter(
        classification.structure_index + jitter,
        classification.primary_phase_increment_coherence_real,
        c=np.where(classification.label.eq(LOW), "#2CA02C", "#D62728"),
        alpha=0.7,
        label=None,
    )
    axis.axhline(threshold, color="black", linestyle="--", label="frozen D0 threshold")
    axis.set(
        xlabel="Independent circuit structure",
        ylabel="C1",
        title="Frozen-threshold classification on disjoint structures",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_02_frozen_threshold_classification.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    effects = structure.primary_low_minus_high_C1.to_numpy(float)
    axis.bar(
        structure.structure_seed.astype(str),
        effects,
        color=np.where(effects > 0.0, "#2CA02C", "#D62728"),
    )
    axis.axhline(0.0, color="0.25", linewidth=0.9)
    axis.axhline(
        float(cfg.analysis.criteria.minimum_mean_low_minus_high),
        color="#1F77B4",
        linestyle="--",
        label="minimum practical mean",
    )
    axis.set(
        xlabel="Independent circuit structure seed",
        ylabel="Mean low-minus-high C1",
        title="Structure-level primary confirmation effects",
    )
    axis.tick_params(axis="x", rotation=30)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_03_structure_level_effects.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    horizons = [float(x) for x in cfg.analysis.measurement.phase_horizon_audit_s]
    for label, color in ((LOW, "#2CA02C"), (HIGH, "#D62728")):
        values = []
        for horizon in horizons:
            token = f"{horizon:g}".replace(".", "p")
            values.append(float(rows[rows.label.eq(label)][
                f"horizon_{token}s_phase_increment_coherence_real"
            ].mean()))
        axis.plot(horizons, values, "o-", label=label, color=color)
    axis.set(
        xlabel="EEG phase-estimation interval (s)",
        ylabel="Phase-increment coherence",
        title="Action-cadence observability audit",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "figure_04_phase_horizon_audit.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    source = _load_and_validate_source(cfg)
    power = _power_design(cfg)
    source_disjoint = _validate_design(cfg, source, power)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "phase_increment_confirmation"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### D0b frozen EEG phase-increment confirmation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
        print("\n### A priori power design")
        print(json.dumps(_plain(power), indent=2))
    comm.Barrier()
    started = time.perf_counter()

    metric_rows: list[dict[str, Any]] = []
    for spec in _context_specs(cfg):
        if rank == 0:
            print(
                f"condition={spec['condition_id']} structure={spec['structure_seed']} "
                f"history={spec['history_index']} f={spec['frequency_hz']:g} Hz "
                f"D={spec['diffusion_rad2_per_s']:g} rad^2/s"
            )
        condition_cfg = _with_diffusion_state(cfg, spec)
        episode = _run_condition(
            condition_id=str(spec["condition_id"]),
            condition_cfg=condition_cfg,
            action=_sham(condition_cfg, str(spec["condition_id"])),
            stimulate=False,
            seed=int(spec["trial_seed"]),
            action_index=0,
            output_dir=root / "episodes" / str(spec["condition_id"]),
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(spec["structure_seed"]),
            drive_seed=int(spec["drive_seed"]),
            phase_seed=int(spec["phase_seed"]),
        )
        if rank == 0:
            metric_rows.append(_episode_metrics(episode, spec, cfg))

    if rank != 0:
        return
    rows = pd.DataFrame(metric_rows)
    threshold = float(cfg.analysis.frozen_endpoint.classification_threshold)
    paired, classification, structure = _paired_tables(rows, threshold)
    inference = _inference(structure, cfg)
    frequency = _frequency_audits(paired, cfg)
    checks, summary = _evaluate(
        rows=rows,
        paired=paired,
        classification=classification,
        structure=structure,
        inference=inference,
        source_disjoint=source_disjoint,
        power=power,
        cfg=cfg,
    )

    rows.to_csv(root / "confirmation_eeg_metrics.csv", index=False)
    paired.to_csv(root / "paired_context_effects.csv", index=False)
    classification.to_csv(root / "frozen_threshold_classification.csv", index=False)
    structure.to_csv(root / "structure_level_primary_effects.csv", index=False)
    frequency.to_csv(root / "frequency_level_FDR_audits.csv", index=False)
    statistical = {"power_design": power, "primary_inference": inference}
    (root / "statistical_inference.json").write_text(
        json.dumps(_plain(statistical), indent=2)
    )
    provenance = {
        "experiment": "D0b_phase_increment_observability_confirmation",
        "source_D0": {
            key: value for key, value in source.items() if key != "source_seed_sets"
        },
        "frozen_endpoint": OmegaConf.to_container(
            cfg.analysis.frozen_endpoint, resolve=True
        ),
        "frozen_generator": {
            "modulation_depth": float(cfg.analysis.states.modulation_depth),
            "frequencies_hz": [float(x) for x in cfg.analysis.states.frequencies_hz],
            "phase_diffusion_levels": _levels(cfg),
        },
        "policy_inputs": "none; no policy or tACS is evaluated",
        "hidden_variables_used_for_classification": False,
        "frequency_selected_from_EEG_candidate_peak": True,
        "classification_uses_only_ideal_preaction_EEG": True,
        "statistical_unit": "independent circuit structure",
        "not_a_disease_or_treatment_model": True,
    }
    (root / "frozen_endpoint_provenance.json").write_text(
        json.dumps(_plain(provenance), indent=2)
    )
    conclusion = {
        "scope": "D0b disjoint stimulation-free ideal-EEG observability confirmation",
        "checks": checks,
        "summary": summary,
        "primary_inference": inference,
        "runtime_seconds": float(time.perf_counter() - started),
        "next_experiment": (
            "D1 full-information context-by-action system identification"
            if summary["ready_for_D1_system_identification"]
            else "Do not run D1; phase-increment observability was not confirmed"
        ),
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_plain(conclusion), indent=2)
    )
    if bool(cfg.experiment.plot):
        _plot_results(root, rows, paired, classification, structure, cfg)

    print("\n### D0b confirmation checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Primary structure-level inference")
    print(json.dumps(_plain(inference), indent=2))
    if bool(cfg.analysis.smoke_test):
        print("\nD0b status: SMOKE TEST ONLY (confirmation gate not evaluated)")
    else:
        print(
            "\nFrozen EEG phase-increment observability: "
            f"{'CONFIRMED' if summary['D0b_phase_increment_observability_confirmed'] else 'NOT CONFIRMED'}"
        )
        print(
            "Ready for D1 system identification: "
            f"{'YES' if summary['ready_for_D1_system_identification'] else 'NO'}"
        )
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
