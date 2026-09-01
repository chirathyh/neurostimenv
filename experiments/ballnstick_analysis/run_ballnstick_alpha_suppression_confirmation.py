"""Confirm a frozen EEG-relative alpha-suppression protocol on new seeds.

This runner performs no calibration and no action selection. It loads the
state generator and tACS protocol produced by the exploratory pilot, verifies
their expected values, and evaluates them on disjoint circuit seeds. The
primary endpoint is the paired reduction in ideal neural-only EEG 8--12-Hz
power. Exact sign-flip tests use the circuit seed as the statistical unit.

The experiment also audits whether the pre-action 10-Hz EEG phase is stable
enough to define the action. Hidden afferent phase, spike PPC, and firing rates
are diagnostics only and never enter phase estimation or protocol selection.
"""

from __future__ import annotations

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


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick import (  # noqa: E402
    _benjamini_hochberg,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    A_HIGH,
    B_LOW,
    OPPOSITE,
    SELECTED,
    TRANSVERSE,
    _action,
    _condition_for_seed,
    _epoch_raw,
    _epoch_row,
    _estimate_relative_field_phase,
    _feature_from_raw,
    _plain,
    _reference_phase,
    _run_condition,
    _sham,
    _two_second_rows,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _bootstrap_ci,
    _relative_rate_safe,
)


CONDITIONS = (B_LOW, A_HIGH, SELECTED, OPPOSITE, TRANSVERSE)
ACTIVE_CONDITIONS = (SELECTED, OPPOSITE, TRANSVERSE)
PRIMARY_METRIC = "selected_alpha_suppression_log10"
CONTROL_FAMILY = (
    "selected_vs_opposite_phase_advantage_log10",
    "selected_vs_transverse_advantage_log10",
)


def _seeds(cfg: DictConfig) -> list[int]:
    first = int(cfg.experiment.seed) + int(cfg.analysis.validation.seed_offset)
    return [first + index for index in range(int(cfg.analysis.validation.n_seeds))]


def _circular_difference(left: float, right: float) -> float:
    return float(np.angle(np.exp(1j * (float(left) - float(right)))))


def _exact_sign_flip_p(values: np.ndarray) -> float:
    """Return an exact two-sided paired sign-flip p-value."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if np.all(values == 0.0):
        return 1.0
    if values.size > 20:
        raise ValueError("Exact sign enumeration is limited to 20 seeds.")
    observed = abs(float(np.mean(values)))
    masks = np.arange(1 << values.size, dtype=np.uint64)[:, None]
    bits = (masks >> np.arange(values.size, dtype=np.uint64)) & 1
    signs = 2.0 * bits.astype(float) - 1.0
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    return float(np.mean(null >= observed - 1.0e-15))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_source(cfg: DictConfig) -> dict[str, Any]:
    root = Path(to_absolute_path(str(cfg.analysis.frozen_source.result_dir)))
    target_path = root / "frozen_alpha_target.json"
    protocol_path = root / "frozen_tacs_protocol.json"
    if not target_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(
            "The confirmation requires frozen_alpha_target.json and "
            f"frozen_tacs_protocol.json under {root}."
        )
    target = json.loads(target_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    expected = cfg.analysis.frozen_source
    comparisons = {
        "modulation depth": (
            float(target["selected_modulation_depth"]),
            float(expected.expected_modulation_depth),
        ),
        "frequency": (
            float(protocol["frequency_hz"]),
            float(expected.expected_frequency_hz),
        ),
        "amplitude": (
            float(protocol["amplitude_v_per_m"]),
            float(expected.expected_amplitude_v_per_m),
        ),
        "relative phase": (
            _wrap_phase(protocol["selected_eeg_relative_phase_offset_rad"]),
            _wrap_phase(expected.expected_relative_phase_offset_rad),
        ),
    }
    failures = [name for name, pair in comparisons.items() if not np.isclose(*pair)]
    if str(protocol["montage"]) != str(expected.expected_montage):
        failures.append("montage")
    if not bool(protocol.get("phase_is_estimated_from_preceding_eeg", False)):
        failures.append("phase policy")
    if not bool(target.get("calibration_passed", False)):
        failures.append("target calibration")
    if failures:
        raise ValueError("Frozen source differs from the preregistered values: " + ", ".join(failures))
    return {
        "source_result_dir": str(root),
        "target": target,
        "protocol": protocol,
        "source_sha256": {
            target_path.name: _sha256(target_path),
            protocol_path.name: _sha256(protocol_path),
        },
    }


def _validate_design(cfg: DictConfig, frozen: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("Alpha confirmation requires the online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("Every confirmation arm requires inhibition_scale=1.")
    if int(cfg.analysis.validation.n_seeds) < 1:
        raise ValueError("At least one confirmation seed is required.")
    if max(_seeds(cfg)) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Circuit seeds are too large for seed * 10,000 mapping.")
    pilot_seed_range = set(range(160001, 162005))
    if pilot_seed_range.intersection(_seeds(cfg)):
        raise ValueError("Confirmation seeds overlap the exploratory pilot.")
    protocol = frozen["protocol"]
    if float(protocol["amplitude_v_per_m"]) > float(cfg.analysis.maximum_field_v_per_m):
        raise ValueError("Frozen field exceeds maximum_field_v_per_m.")
    timeline = cfg.analysis.timeline
    if int(timeline.baseline_steps) < 4:
        raise ValueError("Split-half phase validation requires >=4 baseline windows.")
    if int(timeline.stimulation_steps) < 4:
        raise ValueError("Primary EEG estimation requires >=4 stimulation windows.")
    trim = float(timeline.stimulation_analysis_trim_ms)
    if trim < float(timeline.block_ramp_ms):
        raise ValueError("The primary analysis must exclude the complete field ramps.")


def _summary(
    values: np.ndarray,
    *,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    low, high = _bootstrap_ci(
        values,
        rng=rng,
        n_bootstrap=int(cfg.analysis.n_bootstrap),
    )
    return {
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_2.5": float(low),
        "ci_97.5": float(high),
        "positive_seed_count": int(np.count_nonzero(values > 0.0)),
        "positive_seed_fraction": float(np.mean(values > 0.0)),
        "exact_paired_sign_flip_p": _exact_sign_flip_p(values),
    }


def _phase_quality_row(
    *,
    seed: int,
    a_episode: dict[str, Any],
    selected_episode: dict[str, Any],
    selected_offset: float,
    cfg: DictConfig,
) -> dict[str, Any]:
    outputs = a_episode["simulation"]["outputs_by_epoch"]["baseline"]
    midpoint = len(outputs) // 2
    block_start_ms = float(a_episode["simulation"]["block_start_ms"])
    fs = float(a_episode["simulator_fs_hz"])

    def estimate(part: list[dict[str, Any]]) -> dict[str, float]:
        return _estimate_relative_field_phase(
            part,
            simulator_fs_hz=fs,
            block_start_ms=block_start_ms,
            relative_offset_rad=selected_offset,
            cfg=cfg,
        )

    full = estimate(outputs)
    first = estimate(outputs[:midpoint])
    second = estimate(outputs[midpoint:])
    disagreement = abs(_circular_difference(
        first["baseline_eeg_phase_at_block_rad"],
        second["baseline_eeg_phase_at_block_rad"],
    ))
    coherent_ratio = float(
        full["baseline_eeg_10hz_resultant_v"]
        / max(full["baseline_eeg_rms_v"], np.finfo(float).tiny)
    )
    realized = float(selected_episode["simulation"]["action"]["phase_rad"])
    action_error = abs(_circular_difference(realized, full["phase_rad"]))
    hidden_input_phase = _reference_phase(seed)
    estimated_lag = _circular_difference(
        full["baseline_eeg_phase_at_block_rad"], hidden_input_phase
    )
    quality = bool(
        disagreement
        <= np.deg2rad(float(cfg.analysis.phase_quality.maximum_split_half_error_deg))
        and coherent_ratio
        >= float(cfg.analysis.phase_quality.minimum_10hz_resultant_to_rms)
        and action_error <= 1.0e-10
    )
    return {
        "seed": int(seed),
        "full_eeg_phase_at_action_rad": float(full["baseline_eeg_phase_at_block_rad"]),
        "first_half_eeg_phase_at_action_rad": float(first["baseline_eeg_phase_at_block_rad"]),
        "second_half_eeg_phase_at_action_rad": float(second["baseline_eeg_phase_at_block_rad"]),
        "split_half_phase_error_rad": disagreement,
        "split_half_phase_error_deg": float(np.degrees(disagreement)),
        "baseline_10hz_resultant_v": float(full["baseline_eeg_10hz_resultant_v"]),
        "baseline_eeg_rms_v": float(full["baseline_eeg_rms_v"]),
        "baseline_10hz_resultant_to_rms": coherent_ratio,
        "realized_field_phase_rad": realized,
        "action_phase_tracking_error_rad": action_error,
        "hidden_input_phase_rad": hidden_input_phase,
        "estimated_eeg_minus_hidden_input_phase_rad": estimated_lag,
        "phase_quality_pass": quality,
    }


def _validation_seed_row(
    *,
    seed: int,
    episodes: dict[str, dict[str, Any]],
    phase_quality: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    stimulation = {name: _epoch_row(episode) for name, episode in episodes.items()}
    a, b = stimulation[A_HIGH], stimulation[B_LOW]
    selected = stimulation[SELECTED]
    opposite = stimulation[OPPOSITE]
    transverse = stimulation[TRANSVERSE]
    a_power = float(a.log10_alpha_power_8_12_hz)
    b_power = float(b.log10_alpha_power_8_12_hz)
    selected_power = float(selected.log10_alpha_power_8_12_hz)
    opposite_power = float(opposite.log10_alpha_power_8_12_hz)
    transverse_power = float(transverse.log10_alpha_power_8_12_hz)
    initial_distance = abs(a_power - b_power)
    selected_distance = abs(selected_power - b_power)

    selected_baseline = _epoch_row(episodes[SELECTED], "baseline")
    selected_washout = _epoch_row(episodes[SELECTED], "washout")
    a_baseline = _epoch_row(episodes[A_HIGH], "baseline")
    a_washout = _epoch_row(episodes[A_HIGH], "washout")
    suppression = a_power - selected_power
    washout_effect = float(
        (a_washout.log10_alpha_power_8_12_hz - a_baseline.log10_alpha_power_8_12_hz)
        - (selected_washout.log10_alpha_power_8_12_hz
           - selected_baseline.log10_alpha_power_8_12_hz)
    )
    residual_fraction = float(cfg.analysis.criteria.maximum_washout_residual_fraction)
    baseline_errors = {
        name: _relative_rms_error(
            _epoch_raw(episodes[A_HIGH], "baseline"),
            _epoch_raw(episodes[name], "baseline"),
        )
        for name in ACTIVE_CONDITIONS
    }
    return {
        "seed": int(seed),
        "A_minus_B_log10_alpha_power": a_power - b_power,
        PRIMARY_METRIC: suppression,
        "selected_target_distance_improvement_log10": initial_distance - selected_distance,
        "selected_fractional_target_distance_improvement": (
            (initial_distance - selected_distance) / initial_distance
            if initial_distance > 0.0 else 0.0
        ),
        "selected_distance_to_B_log10": selected_distance,
        "selected_vs_opposite_phase_advantage_log10": opposite_power - selected_power,
        "selected_vs_transverse_advantage_log10": transverse_power - selected_power,
        "selected_10hz_resultant_suppression_v": float(
            a.eeg_10hz_resultant_v - selected.eeg_10hz_resultant_v
        ),
        "selected_alpha_prominence_reduction_db": float(
            a.alpha_peak_prominence_db - selected.alpha_peak_prominence_db
        ),
        "selected_E_ppc_reduction": float(a.E_ppc - selected.E_ppc),
        "selected_I_ppc_reduction": float(a.I_ppc - selected.I_ppc),
        "selected_E_rate_change_hz": float(selected.E_firing_rate_hz - a.E_firing_rate_hz),
        "selected_I_rate_change_hz": float(selected.I_firing_rate_hz - a.I_firing_rate_hz),
        "reference_rate_matched": bool(_relative_rate_safe(a, b, cfg)),
        "selected_rate_safe": bool(_relative_rate_safe(selected, a, cfg)),
        "opposite_rate_safe": bool(_relative_rate_safe(opposite, a, cfg)),
        "transverse_rate_safe": bool(_relative_rate_safe(transverse, a, cfg)),
        "washout_effect_log10": washout_effect,
        "washout_recovered": bool(
            suppression > 0.0
            and abs(washout_effect)
            <= residual_fraction * max(abs(suppression), np.finfo(float).eps)
        ),
        "maximum_baseline_relative_rms_error": float(max(baseline_errors.values())),
        **phase_quality,
    }


def _metric_summaries(
    metrics: pd.DataFrame,
    *,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    names = [
        "A_minus_B_log10_alpha_power",
        PRIMARY_METRIC,
        "selected_target_distance_improvement_log10",
        *CONTROL_FAMILY,
        "selected_10hz_resultant_suppression_v",
        "selected_alpha_prominence_reduction_db",
        "selected_E_ppc_reduction",
        "selected_I_ppc_reduction",
    ]
    result = pd.DataFrame([
        {"metric": name, **_summary(metrics[name].to_numpy(float), cfg=cfg, rng=rng)}
        for name in names
    ])
    result["control_family_fdr_q"] = np.nan
    control_mask = result.metric.isin(CONTROL_FAMILY)
    result.loc[control_mask, "control_family_fdr_q"] = _benjamini_hochberg(
        result.loc[control_mask, "exact_paired_sign_flip_p"].to_numpy(float)
    )
    return result


def _classify(value: float, target_model: dict[str, Any]) -> str:
    above = float(value) > float(target_model["classification_threshold"])
    is_a = above == bool(target_model["A_is_above_threshold"])
    return "A" if is_a else "B"


def _positive_confirmed(
    summary: pd.Series,
    *,
    cfg: DictConfig,
    p_name: str = "exact_paired_sign_flip_p",
) -> bool:
    return bool(
        float(summary["mean"]) > 0.0
        and float(summary["ci_2.5"]) > 0.0
        and int(summary["positive_seed_count"])
        >= int(cfg.analysis.criteria.minimum_positive_seeds)
        and float(summary[p_name])
        <= float(cfg.analysis.criteria.maximum_primary_p_value)
    )


def _conclusions(
    *,
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    epoch_rows: pd.DataFrame,
    two_second: pd.DataFrame,
    target_model: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    indexed = summaries.set_index("metric")
    reference = epoch_rows[
        epoch_rows.epoch.eq("stimulation")
        & epoch_rows.condition_id.isin((A_HIGH, B_LOW))
    ]
    expected = np.where(reference.condition_id.eq(A_HIGH), "A", "B")
    predictions = np.asarray([
        _classify(value, target_model)
        for value in reference.log10_alpha_power_8_12_hz.to_numpy(float)
    ])
    reference_accuracy = float(np.mean(predictions == expected))
    reference_bins = two_second[two_second.condition_id.isin((A_HIGH, B_LOW))]
    bin_expected = np.where(reference_bins.condition_id.eq(A_HIGH), "A", "B")
    bin_predictions = np.asarray([
        _classify(value, target_model)
        for value in reference_bins.log10_alpha_power_8_12_hz.to_numpy(float)
    ])
    bin_accuracy = float(np.mean(bin_predictions == bin_expected))
    phase_quality_fraction = float(metrics.phase_quality_pass.mean())
    rate_safe_fraction = float(metrics[
        ["selected_rate_safe", "opposite_rate_safe", "transverse_rate_safe"]
    ].all(axis=1).mean())
    washout_fraction = float(metrics.washout_recovered.mean())

    phase_summary = indexed.loc[CONTROL_FAMILY[0]].copy()
    phase_summary["fdr_q"] = phase_summary["control_family_fdr_q"]
    orientation_summary = indexed.loc[CONTROL_FAMILY[1]].copy()
    orientation_summary["fdr_q"] = orientation_summary["control_family_fdr_q"]
    checks = {
        "minimum_confirmation_seeds": len(metrics) >= int(cfg.analysis.criteria.minimum_validation_seeds),
        "frozen_protocol_loaded_without_selection": True,
        "heldout_elevated_alpha_state_confirmed": _positive_confirmed(indexed.loc["A_minus_B_log10_alpha_power"], cfg=cfg),
        "frozen_reference_classification": reference_accuracy >= float(cfg.analysis.criteria.minimum_reference_classification_accuracy),
        "two_second_eeg_observable": bin_accuracy >= float(cfg.analysis.criteria.minimum_two_second_classification_accuracy),
        "baseline_phase_estimate_stable": phase_quality_fraction >= float(cfg.analysis.criteria.minimum_phase_quality_fraction),
        "frozen_phase_reduces_alpha_confirmed": _positive_confirmed(indexed.loc[PRIMARY_METRIC], cfg=cfg),
        "frozen_phase_moves_eeg_toward_B_confirmed": _positive_confirmed(indexed.loc["selected_target_distance_improvement_log10"], cfg=cfg),
        "phase_specific_confirmed": _positive_confirmed(phase_summary, cfg=cfg, p_name="fdr_q"),
        "orientation_specific_confirmed": _positive_confirmed(orientation_summary, cfg=cfg, p_name="fdr_q"),
        "coherent_10hz_component_reduced": _positive_confirmed(indexed.loc["selected_10hz_resultant_suppression_v"], cfg=cfg),
        "alpha_peak_prominence_reduced": _positive_confirmed(indexed.loc["selected_alpha_prominence_reduction_db"], cfg=cfg),
        "hidden_E_spike_synchrony_reduced": _positive_confirmed(indexed.loc["selected_E_ppc_reduction"], cfg=cfg),
        "reference_rate_matched": bool(metrics.reference_rate_matched.all()),
        "all_tacs_arms_rate_safe": rate_safe_fraction >= float(cfg.analysis.criteria.minimum_rate_safe_fraction),
        "washout_reversible": washout_fraction >= float(cfg.analysis.criteria.minimum_washout_recovery_fraction),
        "baseline_causality": bool(
            metrics.maximum_baseline_relative_rms_error.max()
            <= float(cfg.analysis.criteria.maximum_baseline_relative_rms_error)
        ),
    }
    primary_names = (
        "minimum_confirmation_seeds",
        "frozen_protocol_loaded_without_selection",
        "heldout_elevated_alpha_state_confirmed",
        "frozen_reference_classification",
        "two_second_eeg_observable",
        "baseline_phase_estimate_stable",
        "frozen_phase_reduces_alpha_confirmed",
        "frozen_phase_moves_eeg_toward_B_confirmed",
        "phase_specific_confirmed",
        "orientation_specific_confirmed",
        "reference_rate_matched",
        "all_tacs_arms_rate_safe",
        "washout_reversible",
        "baseline_causality",
    )
    conclusions = {
        "ideal_neural_eeg_phase_control_confirmed": all(checks[name] for name in primary_names),
        "hidden_spike_mechanism_supported": checks["hidden_E_spike_synchrony_reduced"],
        "heldout_reference_classification_accuracy": reference_accuracy,
        "two_second_reference_classification_accuracy": bin_accuracy,
        "phase_quality_seed_fraction": phase_quality_fraction,
        "all_action_rate_safe_seed_fraction": rate_safe_fraction,
        "washout_recovery_seed_fraction": washout_fraction,
    }
    return checks, conclusions


def _plot_confirmation(
    *,
    root: Path,
    episodes: dict[int, dict[str, dict[str, Any]]],
    epoch_rows: pd.DataFrame,
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    cfg: DictConfig,
) -> None:
    labels = {
        B_LOW: "B low-alpha",
        A_HIGH: "A elevated-alpha",
        SELECTED: "A + frozen 180°",
        OPPOSITE: "A + opposite 0°",
        TRANSVERSE: "A + transverse",
    }
    colors = {
        B_LOW: "#2CA02C", A_HIGH: "#9467BD", SELECTED: "#E67E22",
        OPPOSITE: "#1F77B4", TRANSVERSE: "#777777",
    }
    trim_samples = int(round(
        float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
        / float(cfg.env.network.dt)
    ))
    psds: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    frequencies = None
    for seed_episodes in episodes.values():
        for condition in CONDITIONS:
            episode = seed_episodes[condition]
            raw = _epoch_raw(episode, "stimulation")[trim_samples:-trim_samples]
            outputs = episode["simulation"]["outputs_by_epoch"]["stimulation"]
            start_ms = float(outputs[0]["t_start_ms"]) + float(
                cfg.analysis.timeline.stimulation_analysis_trim_ms
            )
            _, _, frequencies, psd = _feature_from_raw(
                raw,
                simulator_fs_hz=float(episode["simulator_fs_hz"]),
                start_ms=start_ms,
                cfg=cfg,
            )
            psds[condition].append(psd)
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    for condition in CONDITIONS:
        values = np.asarray(psds[condition])
        mean_db = 10.0 * np.log10(np.maximum(values.mean(axis=0), np.finfo(float).tiny))
        axis.plot(frequencies, mean_db, color=colors[condition], label=labels[condition])
    axis.axvspan(8.0, 12.0, color="gold", alpha=0.15)
    axis.set_xlim(2.0, 25.0)
    axis.set(xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)",
             title="New-seed ideal EEG spectra during intervention")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_confirmation_psd.png", dpi=250)
    plt.close(figure)

    stimulation = epoch_rows[epoch_rows.epoch.eq("stimulation")]
    wide = stimulation.pivot(
        index="seed", columns="condition_id", values="log10_alpha_power_8_12_hz"
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    for _, row in wide.iterrows():
        axis.plot(
            range(len(CONDITIONS)), [row[name] for name in CONDITIONS],
            color="0.78", linewidth=1.0,
        )
    axis.plot(
        range(len(CONDITIONS)), [wide[name].mean() for name in CONDITIONS],
        color="black", marker="o", linewidth=2.4, label="mean",
    )
    axis.set_xticks(
        range(len(CONDITIONS)),
        ["B", "A", "frozen 180°", "opposite 0°", "transverse"],
        rotation=20,
    )
    axis.set_ylabel("log10 EEG alpha power")
    axis.set_title("Paired confirmation endpoints")
    figure.tight_layout()
    figure.savefig(root / "figure_02_paired_alpha_power.png", dpi=250)
    plt.close(figure)

    effect_names = [PRIMARY_METRIC, *CONTROL_FAMILY, "selected_E_ppc_reduction"]
    effect_labels = ["alpha suppression", "vs opposite", "vs transverse", "E-PPC reduction"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for position, (name, label) in enumerate(zip(effect_names[:3], effect_labels[:3])):
        values = metrics[name].to_numpy(float)
        axes[0].scatter(np.full(values.size, position), values, alpha=0.75)
        row = summaries[summaries.metric.eq(name)].iloc[0]
        axes[0].errorbar(
            position, row["mean"],
            yerr=[[row["mean"] - row["ci_2.5"]], [row["ci_97.5"] - row["mean"]]],
            color="black", marker="o", capsize=4,
        )
    axes[0].axhline(0.0, color="0.45", linewidth=0.8)
    axes[0].set_xticks(range(3), effect_labels[:3], rotation=20)
    axes[0].set_ylabel("Paired log10 alpha-power effect")
    axes[0].set_title("Frozen action and causal controls")
    axes[1].scatter(
        metrics.split_half_phase_error_deg,
        metrics.baseline_10hz_resultant_to_rms,
        c=np.where(metrics.phase_quality_pass, "#2CA02C", "#D62728"),
    )
    axes[1].axvline(
        float(cfg.analysis.phase_quality.maximum_split_half_error_deg),
        color="0.4", linestyle="--",
    )
    axes[1].axhline(
        float(cfg.analysis.phase_quality.minimum_10hz_resultant_to_rms),
        color="0.4", linestyle="--",
    )
    axes[1].set(
        xlabel="Baseline split-half phase error (degrees)",
        ylabel="10-Hz resultant / EEG RMS",
        title="Pre-action phase quality",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_03_effects_and_phase_quality.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen = _load_frozen_source(cfg)
    _validate_design(cfg, frozen)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "alpha_suppression_confirmation"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### Frozen alpha-suppression confirmation")
        print(json.dumps(_plain(frozen), indent=2))
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    rng = np.random.default_rng(int(cfg.experiment.seed) + 1_900_003)

    protocol = frozen["protocol"]
    target = frozen["target"]
    depth = float(target["selected_modulation_depth"])
    selected_offset = float(protocol["selected_eeg_relative_phase_offset_rad"])
    amplitude = float(protocol["amplitude_v_per_m"])
    axial = str(protocol["montage"])
    transverse_montage = str(cfg.analysis.tacs.transverse_montage)
    specifications = (
        (SELECTED, selected_offset, axial, "frozen_primary"),
        (OPPOSITE, _wrap_phase(selected_offset + np.pi), axial, "opposite_phase_control"),
        (TRANSVERSE, selected_offset, transverse_montage, "orientation_control"),
    )

    all_episodes: dict[int, dict[str, dict[str, Any]]] = {}
    epoch_rows: list[dict[str, Any]] = []
    two_second_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for seed in _seeds(cfg):
        if rank == 0:
            print(f"confirmation seed={seed}")
            all_episodes[seed] = {}
        b_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=0.0)
        a_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=depth)
        for action_index, (condition, condition_cfg) in enumerate(
            ((B_LOW, b_cfg), (A_HIGH, a_cfg))
        ):
            episode = _run_condition(
                condition_id=condition,
                condition_cfg=condition_cfg,
                action=_sham(cfg, condition),
                stimulate=False,
                seed=seed,
                action_index=action_index,
                output_dir=root / "episodes" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                all_episodes[seed][condition] = episode
                epoch_rows.extend(episode["epoch_rows"])
                two_second_rows.extend(_two_second_rows(
                    episode, condition_id=condition, cfg=cfg
                ))
        for action_index, (condition, offset, montage, role) in enumerate(
            specifications, start=2
        ):
            action = _action(
                cfg,
                identifier=condition,
                role=role,
                amplitude=amplitude,
                montage=montage,
                relative_offset=offset,
            )
            episode = _run_condition(
                condition_id=condition,
                condition_cfg=a_cfg,
                action=action,
                stimulate=True,
                seed=seed,
                action_index=action_index,
                output_dir=root / "episodes" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                all_episodes[seed][condition] = episode
                epoch_rows.extend(episode["epoch_rows"])
                two_second_rows.extend(_two_second_rows(
                    episode, condition_id=condition, cfg=cfg
                ))
        if rank == 0:
            quality = _phase_quality_row(
                seed=seed,
                a_episode=all_episodes[seed][A_HIGH],
                selected_episode=all_episodes[seed][SELECTED],
                selected_offset=selected_offset,
                cfg=cfg,
            )
            phase_rows.append(quality)
            seed_rows.append(_validation_seed_row(
                seed=seed,
                episodes=all_episodes[seed],
                phase_quality=quality,
                cfg=cfg,
            ))

    if rank == 0:
        epoch_frame = pd.DataFrame(epoch_rows)
        two_second_frame = pd.DataFrame(two_second_rows)
        metrics = pd.DataFrame(seed_rows)
        phase_frame = pd.DataFrame(phase_rows)
        summaries = _metric_summaries(metrics, cfg=cfg, rng=rng)
        checks, conclusions = _conclusions(
            metrics=metrics,
            summaries=summaries,
            epoch_rows=epoch_frame,
            two_second=two_second_frame,
            target_model=target["target_model"],
            cfg=cfg,
        )
        epoch_frame.to_csv(root / "confirmation_epoch_eeg_and_hidden_metrics.csv", index=False)
        two_second_frame.to_csv(root / "confirmation_two_second_eeg_bins.csv", index=False)
        metrics.to_csv(root / "confirmation_seed_metrics.csv", index=False)
        phase_frame.to_csv(root / "baseline_phase_quality.csv", index=False)
        summaries.to_csv(root / "confirmation_summary.csv", index=False)
        frozen_copy = {
            **frozen,
            "confirmation_seeds": _seeds(cfg),
            "selection_performed_in_confirmation": False,
        }
        (root / "frozen_protocol_provenance.json").write_text(
            json.dumps(_plain(frozen_copy), indent=2)
        )
        result = {
            "scope": "ideal neural-only simulated EEG",
            "checks": checks,
            "conclusions": conclusions,
            "primary_metric": PRIMARY_METRIC,
            "primary_summary": _plain(
                summaries[summaries.metric.eq(PRIMARY_METRIC)].iloc[0].to_dict()
            ),
            "selection_performed_in_confirmation": False,
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": (
                "A positive result confirms acute ideal-EEG phase control in this "
                "toy circuit only. It does not validate depression biology, human "
                "treatment, persistent plasticity, or a contextual RL policy."
            ),
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        if bool(cfg.experiment.plot):
            _plot_confirmation(
                root=root,
                episodes=all_episodes,
                epoch_rows=epoch_frame,
                metrics=metrics,
                summaries=summaries,
                cfg=cfg,
            )
        print("\n### Confirmation checks")
        for name, passed in checks.items():
            print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
        print(
            "\nIdeal neural-EEG phase control:",
            "CONFIRMED" if conclusions["ideal_neural_eeg_phase_control_confirmed"] else "NOT CONFIRMED",
        )
        print(
            "Hidden spike mechanism:",
            "SUPPORTED" if conclusions["hidden_spike_mechanism_supported"] else "NOT SUPPORTED",
        )
        print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
