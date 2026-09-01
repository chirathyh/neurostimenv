"""Confirm frozen 0.4-V/m alpha control in an EEG-screen-positive subgroup.

Candidate circuit seeds first undergo an unstimulated A-state screening
episode. Eligibility uses only the frozen ideal-EEG alpha classifier, baseline
10-Hz phase quality, and baseline firing-rate safety. No stimulation outcome
and no seed-specific B counterfactual enters screening. Rejected seeds are
retained in an audit table and receive no active intervention.

The first eight eligible seeds then receive a frozen 10-Hz, 0.4-V/m,
EEG-relative 180-degree axial action plus opposite-phase and transverse
controls. The primary endpoint is paired ideal neural-only EEG alpha
suppression; movement toward the subsequently simulated B reference is a
separate required endpoint. The estimand is explicitly conditional on the
screen-positive, phase-actionable toy subgroup.
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
    _plain,
    _run_condition,
    _sham,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_confirmation import (  # noqa: E402
    _circular_difference,
    _exact_sign_flip_p,
    _phase_quality_row,
    _positive_confirmed,
    _summary,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (  # noqa: E402
    _complex_response_decomposition,
    _field_removal_status,
    _stimulation_psd,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _relative_rate_safe,
)


PRIMARY_METRIC = "selected_alpha_suppression_log10"
CONTROL_FAMILY = (
    "selected_vs_opposite_phase_advantage_log10",
    "selected_vs_transverse_advantage_log10",
)
CONDITIONS = (B_LOW, A_HIGH, SELECTED, OPPOSITE, TRANSVERSE)
ACTIVE_CONDITIONS = (SELECTED, OPPOSITE, TRANSVERSE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_seeds(cfg: DictConfig) -> list[int]:
    first = int(cfg.experiment.seed) + int(cfg.analysis.screening.seed_offset)
    return [
        first + index
        for index in range(int(cfg.analysis.screening.maximum_candidate_seeds))
    ]


def _screening_target_reached(
    *,
    comm: Any,
    rank: int,
    enrolled_count: int,
    target_count: int,
) -> bool:
    """Return rank 0's enrollment-stop decision on every MPI rank.

    Only rank 0 owns ``enrolled_seeds`` and the associated analysis tables.
    Broadcasting its decision keeps all ranks on the same side of the next
    collective network simulation instead of letting workers continue after
    rank 0 has reached the target.
    """
    reached = enrolled_count >= target_count if rank == 0 else None
    return bool(comm.bcast(reached, root=0))


def _load_frozen_candidate(cfg: DictConfig) -> dict[str, Any]:
    root = Path(to_absolute_path(str(cfg.analysis.frozen_candidate.result_dir)))
    candidate_path = root / "exploratory_candidate_protocol.json"
    provenance_path = root / "frozen_protocol_provenance.json"
    if not candidate_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            "Screened confirmation requires exploratory_candidate_protocol.json "
            f"and frozen_protocol_provenance.json under {root}."
        )
    candidate = json.loads(candidate_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    expected = cfg.analysis.frozen_candidate
    comparisons = {
        "dose": (
            float(candidate["selected_dose_v_per_m"]),
            float(expected.expected_amplitude_v_per_m),
        ),
        "frequency": (
            float(candidate["frequency_hz"]),
            float(expected.expected_frequency_hz),
        ),
        "relative phase": (
            _wrap_phase(candidate["relative_phase_offset_rad"]),
            _wrap_phase(expected.expected_relative_phase_offset_rad),
        ),
        "state modulation depth": (
            float(provenance["target"]["selected_modulation_depth"]),
            float(expected.expected_modulation_depth),
        ),
    }
    failures = [name for name, pair in comparisons.items() if not np.isclose(*pair)]
    if str(candidate["montage"]) != str(expected.expected_montage):
        failures.append("montage")
    if not bool(candidate.get("directional_candidate_found", False)):
        failures.append("directional candidate status")
    if not bool(candidate.get("requires_disjoint_confirmation", False)):
        failures.append("confirmation requirement")
    if failures:
        raise ValueError(
            "Frozen exploratory candidate differs from expected values: "
            + ", ".join(failures)
        )
    return {
        "source_result_dir": str(root),
        "candidate": candidate,
        "target": provenance["target"],
        "pilot_protocol": provenance["protocol"],
        "source_sha256": {
            candidate_path.name: _sha256(candidate_path),
            provenance_path.name: _sha256(provenance_path),
        },
    }


def _validate_design(cfg: DictConfig, frozen: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("Screened alpha confirmation requires the online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("Every screened-confirmation arm requires inhibition_scale=1.")
    target_count = int(cfg.analysis.screening.target_enrolled_seeds)
    maximum_count = int(cfg.analysis.screening.maximum_candidate_seeds)
    if target_count < 1 or maximum_count < target_count:
        raise ValueError("maximum_candidate_seeds must cover target_enrolled_seeds.")
    seeds = set(_candidate_seeds(cfg))
    prior = (
        set(range(160001, 162005))
        | set(range(170001, 170009))
        | set(range(180001, 180005))
    )
    if prior.intersection(seeds):
        raise ValueError("Screened-confirmation candidates overlap prior scientific seeds.")
    if max(seeds) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Candidate seeds are too large for seed * 10,000 mapping.")
    if float(frozen["candidate"]["selected_dose_v_per_m"]) > float(
        cfg.analysis.maximum_field_v_per_m
    ):
        raise ValueError("Frozen candidate exceeds maximum_field_v_per_m.")
    timeline = cfg.analysis.timeline
    if int(timeline.baseline_steps) < 4:
        raise ValueError("Screening phase quality requires at least four baseline windows.")
    if int(timeline.stimulation_steps) < 4:
        raise ValueError("Confirmation EEG estimation requires at least four windows.")
    stimulation_ms = (
        int(timeline.stimulation_steps) * float(cfg.env.simulation.obs_win_len)
    )
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    if trim_ms < float(timeline.block_ramp_ms) or 2.0 * trim_ms >= stimulation_ms:
        raise ValueError("Analysis trimming must remove both ramps and leave data.")


def _classify_alpha(value: float, target_model: dict[str, Any]) -> str:
    above = float(value) > float(target_model["classification_threshold"])
    is_a = above == bool(target_model["A_is_above_threshold"])
    return "A" if is_a else "B"


def _screen_phase_quality(
    *, episode: dict[str, Any], relative_offset: float, cfg: DictConfig
) -> dict[str, Any]:
    """Estimate phase actionability using only the unstimulated A episode."""
    outputs = episode["simulation"]["outputs_by_epoch"]["baseline"]
    midpoint = len(outputs) // 2
    block_start_ms = float(episode["simulation"]["block_start_ms"])
    fs_hz = float(episode["simulator_fs_hz"])

    def estimate(part: list[dict[str, Any]]) -> dict[str, float]:
        return _estimate_relative_field_phase(
            part,
            simulator_fs_hz=fs_hz,
            block_start_ms=block_start_ms,
            relative_offset_rad=relative_offset,
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
    passed = bool(
        disagreement
        <= np.deg2rad(float(cfg.analysis.phase_quality.maximum_split_half_error_deg))
        and coherent_ratio
        >= float(cfg.analysis.phase_quality.minimum_10hz_resultant_to_rms)
    )
    return {
        "screen_phase_at_action_rad": float(
            full["baseline_eeg_phase_at_block_rad"]
        ),
        "screen_phase_split_error_deg": float(np.degrees(disagreement)),
        "screen_10hz_resultant_v": float(full["baseline_eeg_10hz_resultant_v"]),
        "screen_eeg_rms_v": float(full["baseline_eeg_rms_v"]),
        "screen_10hz_resultant_to_rms": coherent_ratio,
        "screen_phase_quality_pass": passed,
    }


def _absolute_rate_safe(row: pd.Series, cfg: DictConfig) -> bool:
    limits = cfg.analysis.rate_guardrails_hz
    return bool(
        float(limits.E_min) <= float(row.E_firing_rate_hz) <= float(limits.E_max)
        and float(limits.I_min) <= float(row.I_firing_rate_hz) <= float(limits.I_max)
    )


def _screening_decision(
    *,
    seed: int,
    screening_order: int,
    a_episode: dict[str, Any],
    phase_quality: dict[str, Any],
    target_model: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    """Return a treatment-blind eligibility decision for one candidate seed."""
    a = _epoch_row(a_episode)
    alpha = float(a.log10_alpha_power_8_12_hz)
    classified = _classify_alpha(alpha, target_model)
    finite = bool(np.isfinite(alpha))
    elevated = bool(finite and classified == "A")
    phase_actionable = bool(phase_quality["screen_phase_quality_pass"])
    rates_safe = _absolute_rate_safe(a, cfg)
    reasons = []
    if not finite:
        reasons.append("nonfinite_eeg")
    if not elevated:
        reasons.append("not_elevated_by_frozen_eeg_threshold")
    if not phase_actionable:
        reasons.append("unstable_or_weak_10hz_phase")
    if not rates_safe:
        reasons.append("baseline_rate_guardrail")
    eligible = bool(finite and elevated and phase_actionable and rates_safe)
    threshold = float(target_model["classification_threshold"])
    direction = 1.0 if bool(target_model["A_is_above_threshold"]) else -1.0
    return {
        "seed": int(seed),
        "screening_order": int(screening_order),
        "eligible": eligible,
        "exclusion_reasons": "eligible" if eligible else ";".join(reasons),
        "screen_log10_alpha_power": alpha,
        "frozen_alpha_classification_threshold": threshold,
        "screen_margin_toward_A_log10": direction * (alpha - threshold),
        "screen_minus_frozen_B_mean_log10": alpha
        - float(target_model["B_mean_log10_alpha"]),
        "screen_classification": classified,
        "screen_elevated_alpha": elevated,
        "screen_phase_actionable": phase_actionable,
        "screen_baseline_rates_safe": rates_safe,
        "screen_E_firing_rate_hz": float(a.E_firing_rate_hz),
        "screen_I_firing_rate_hz": float(a.I_firing_rate_hz),
        "screening_uses_stimulation_outcome": False,
        "screening_uses_seed_specific_B": False,
        **phase_quality,
    }


def _enrolled_seed_metrics(
    *,
    seed: int,
    episodes: dict[str, dict[str, Any]],
    screening: dict[str, Any],
    phase_quality: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    rows = {name: _epoch_row(episode) for name, episode in episodes.items()}
    a, b = rows[A_HIGH], rows[B_LOW]
    selected, opposite, transverse = rows[SELECTED], rows[OPPOSITE], rows[TRANSVERSE]
    a_power = float(a.log10_alpha_power_8_12_hz)
    b_power = float(b.log10_alpha_power_8_12_hz)
    selected_power = float(selected.log10_alpha_power_8_12_hz)
    initial_distance = abs(a_power - b_power)
    selected_distance = abs(selected_power - b_power)
    suppression = a_power - selected_power

    a_baseline = _epoch_row(episodes[A_HIGH], "baseline")
    a_washout = _epoch_row(episodes[A_HIGH], "washout")
    selected_baseline = _epoch_row(episodes[SELECTED], "baseline")
    selected_washout = _epoch_row(episodes[SELECTED], "washout")
    washout_residual = float(
        (a_washout.log10_alpha_power_8_12_hz
         - a_baseline.log10_alpha_power_8_12_hz)
        - (selected_washout.log10_alpha_power_8_12_hz
           - selected_baseline.log10_alpha_power_8_12_hz)
    )
    recovered, recovery_tolerance = _field_removal_status(
        effect_log10=suppression,
        residual_log10=washout_residual,
        cfg=cfg,
    )
    decomposition = _complex_response_decomposition(
        sham_cosine=float(a.eeg_10hz_cosine_v),
        sham_sine=float(a.eeg_10hz_sine_v),
        active_cosine=float(selected.eeg_10hz_cosine_v),
        active_sine=float(selected.eeg_10hz_sine_v),
    )
    baseline_errors = {
        name: _relative_rms_error(
            _epoch_raw(episodes[A_HIGH], "baseline"),
            _epoch_raw(episodes[name], "baseline"),
        )
        for name in ACTIVE_CONDITIONS
    }
    return {
        "seed": int(seed),
        "screening_order": int(screening["screening_order"]),
        "screen_margin_toward_A_log10": float(
            screening["screen_margin_toward_A_log10"]
        ),
        "A_minus_B_log10_alpha_power": a_power - b_power,
        PRIMARY_METRIC: suppression,
        "selected_target_distance_improvement_log10": (
            initial_distance - selected_distance
        ),
        "selected_fractional_target_distance_improvement": (
            (initial_distance - selected_distance) / initial_distance
            if initial_distance > np.finfo(float).eps else 0.0
        ),
        "selected_distance_to_B_log10": selected_distance,
        "selected_finishes_below_B": bool(selected_power < b_power),
        "selected_vs_opposite_phase_advantage_log10": float(
            opposite.log10_alpha_power_8_12_hz - selected_power
        ),
        "selected_vs_transverse_advantage_log10": float(
            transverse.log10_alpha_power_8_12_hz - selected_power
        ),
        "selected_10hz_resultant_suppression_v": float(
            a.eeg_10hz_resultant_v - selected.eeg_10hz_resultant_v
        ),
        "selected_alpha_prominence_reduction_db": float(
            a.alpha_peak_prominence_db - selected.alpha_peak_prominence_db
        ),
        "selected_E_ppc_reduction": float(a.E_ppc - selected.E_ppc),
        "selected_I_ppc_reduction": float(a.I_ppc - selected.I_ppc),
        "selected_E_rate_change_hz": float(
            selected.E_firing_rate_hz - a.E_firing_rate_hz
        ),
        "selected_I_rate_change_hz": float(
            selected.I_firing_rate_hz - a.I_firing_rate_hz
        ),
        "reference_rate_matched": bool(_relative_rate_safe(a, b, cfg)),
        "selected_rate_safe": bool(_relative_rate_safe(selected, a, cfg)),
        "opposite_rate_safe": bool(_relative_rate_safe(opposite, a, cfg)),
        "transverse_rate_safe": bool(_relative_rate_safe(transverse, a, cfg)),
        "field_removal_residual_log10": washout_residual,
        "field_removal_tolerance_log10": recovery_tolerance,
        "field_removal_recovered": recovered,
        "maximum_baseline_relative_rms_error": float(max(baseline_errors.values())),
        **decomposition,
        **{key: value for key, value in phase_quality.items() if key != "seed"},
    }


def _metric_summaries(
    metrics: pd.DataFrame, *, cfg: DictConfig, rng: np.random.Generator
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
        "coherent_interference_cross_term_fraction",
        "coherent_induced_component_fraction",
        "coherent_net_change_fraction",
    ]
    result = pd.DataFrame([
        {"metric": name, **_summary(metrics[name].to_numpy(float), cfg=cfg, rng=rng)}
        for name in names
    ])
    result["control_family_fdr_q"] = np.nan
    control = result.metric.isin(CONTROL_FAMILY)
    result.loc[control, "control_family_fdr_q"] = _benjamini_hochberg(
        result.loc[control, "exact_paired_sign_flip_p"].to_numpy(float)
    )
    return result


def _conclusions(
    *,
    screening: pd.DataFrame,
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    epoch_rows: pd.DataFrame,
    target_model: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    indexed = summaries.set_index("metric")
    criteria = cfg.analysis.criteria
    b_rows = epoch_rows[
        epoch_rows.epoch.eq("stimulation") & epoch_rows.condition_id.eq(B_LOW)
    ]
    b_accuracy = float(np.mean([
        _classify_alpha(value, target_model) == "B"
        for value in b_rows.log10_alpha_power_8_12_hz.to_numpy(float)
    ]))
    phase_summary = indexed.loc[CONTROL_FAMILY[0]].copy()
    phase_summary["fdr_q"] = phase_summary["control_family_fdr_q"]
    orientation_summary = indexed.loc[CONTROL_FAMILY[1]].copy()
    orientation_summary["fdr_q"] = orientation_summary["control_family_fdr_q"]
    rate_safe_fraction = float(metrics[
        ["selected_rate_safe", "opposite_rate_safe", "transverse_rate_safe"]
    ].all(axis=1).mean())
    checks = {
        "minimum_enrolled_seeds": len(metrics)
        >= int(cfg.analysis.screening.target_enrolled_seeds),
        "frozen_candidate_loaded_without_selection": True,
        "screening_precedes_and_excludes_stimulation_outcomes": bool(
            (~screening.screening_uses_stimulation_outcome).all()
        ),
        "screening_does_not_use_seed_specific_B": bool(
            (~screening.screening_uses_seed_specific_B).all()
        ),
        "all_enrolled_pass_frozen_EEG_and_phase_screen": bool(
            screening[screening.eligible].screen_elevated_alpha.all()
            and screening[screening.eligible].screen_phase_actionable.all()
        ),
        "screened_A_minus_B_confirmed": _positive_confirmed(
            indexed.loc["A_minus_B_log10_alpha_power"], cfg=cfg
        ),
        "enrolled_B_matches_frozen_reference": b_accuracy
        >= float(criteria.minimum_B_classification_accuracy),
        "frozen_0p4_reduces_alpha_confirmed": _positive_confirmed(
            indexed.loc[PRIMARY_METRIC], cfg=cfg
        ),
        "frozen_0p4_moves_eeg_toward_B_confirmed": _positive_confirmed(
            indexed.loc["selected_target_distance_improvement_log10"], cfg=cfg
        ),
        "phase_specific_confirmed": _positive_confirmed(
            phase_summary, cfg=cfg, p_name="fdr_q"
        ),
        "orientation_specific_confirmed": _positive_confirmed(
            orientation_summary, cfg=cfg, p_name="fdr_q"
        ),
        "coherent_10hz_component_reduced": _positive_confirmed(
            indexed.loc["selected_10hz_resultant_suppression_v"], cfg=cfg
        ),
        "alpha_peak_prominence_reduced": _positive_confirmed(
            indexed.loc["selected_alpha_prominence_reduction_db"], cfg=cfg
        ),
        "hidden_E_spike_synchrony_reduced": _positive_confirmed(
            indexed.loc["selected_E_ppc_reduction"], cfg=cfg
        ),
        "reference_rate_matched": bool(metrics.reference_rate_matched.all()),
        "all_tacs_arms_rate_safe": rate_safe_fraction
        >= float(criteria.minimum_rate_safe_fraction),
        "field_removal_recovered_independent_of_effect_sign": float(
            metrics.field_removal_recovered.mean()
        ) >= float(criteria.minimum_field_removal_fraction),
        "baseline_causality": bool(
            metrics.maximum_baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "coherent_energy_decomposition_exact": bool(np.allclose(
            metrics.coherent_net_change_v2,
            metrics.coherent_interference_cross_term_v2
            + metrics.coherent_induced_component_v2,
            rtol=1.0e-10,
            atol=1.0e-30,
        )),
    }
    primary_names = (
        "minimum_enrolled_seeds",
        "frozen_candidate_loaded_without_selection",
        "screening_precedes_and_excludes_stimulation_outcomes",
        "screening_does_not_use_seed_specific_B",
        "all_enrolled_pass_frozen_EEG_and_phase_screen",
        "screened_A_minus_B_confirmed",
        "enrolled_B_matches_frozen_reference",
        "frozen_0p4_reduces_alpha_confirmed",
        "frozen_0p4_moves_eeg_toward_B_confirmed",
        "phase_specific_confirmed",
        "orientation_specific_confirmed",
        "reference_rate_matched",
        "all_tacs_arms_rate_safe",
        "field_removal_recovered_independent_of_effect_sign",
        "baseline_causality",
    )
    conclusions = {
        "ideal_neural_eeg_control_in_screen_positive_subgroup_confirmed": all(
            checks[name] for name in primary_names
        ),
        "hidden_spike_mechanism_supported": checks[
            "hidden_E_spike_synchrony_reduced"
        ],
        "candidate_seeds_screened": int(len(screening)),
        "eligible_seeds_enrolled": int(screening.eligible.sum()),
        "screening_yield": float(screening.eligible.mean()),
        "B_reference_classification_accuracy": b_accuracy,
        "all_action_rate_safe_seed_fraction": rate_safe_fraction,
        "field_removal_recovery_seed_fraction": float(
            metrics.field_removal_recovered.mean()
        ),
        "estimand": (
            "acute effect among toy circuits passing the frozen elevated-alpha "
            "and phase-actionability screen"
        ),
    }
    return checks, conclusions


def _plot_results(
    *,
    root: Path,
    screening: pd.DataFrame,
    psds: dict[str, list[np.ndarray]],
    frequencies: np.ndarray,
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    target_model: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.3))
    accepted = screening.eligible.to_numpy(bool)
    axis.scatter(
        screening.screening_order[~accepted],
        screening.screen_log10_alpha_power[~accepted],
        color="#999999", marker="x", s=55, label="excluded",
    )
    axis.scatter(
        screening.screening_order[accepted],
        screening.screen_log10_alpha_power[accepted],
        color="#2CA02C", s=55, label="enrolled",
    )
    axis.axhline(
        float(target_model["classification_threshold"]),
        color="black", linestyle="--", label="frozen A/B threshold",
    )
    axis.set(
        xlabel="Prospective candidate order",
        ylabel="Unstimulated log10 EEG alpha power",
        title="Treatment-blind elevated-alpha screening",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "figure_01_screening_audit.png", dpi=250)
    plt.close(figure)

    labels = {
        B_LOW: "B reference", A_HIGH: "A sham",
        SELECTED: "A + frozen 0.4 V/m", OPPOSITE: "A + opposite phase",
        TRANSVERSE: "A + transverse",
    }
    colors = {
        B_LOW: "#2CA02C", A_HIGH: "#9467BD", SELECTED: "#E67E22",
        OPPOSITE: "#1F77B4", TRANSVERSE: "#777777",
    }
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for condition in CONDITIONS:
        mean_psd = np.mean(np.asarray(psds[condition]), axis=0)
        axis.plot(
            frequencies,
            10.0 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny)),
            label=labels[condition], color=colors[condition],
        )
    axis.axvspan(8.0, 12.0, color="gold", alpha=0.14)
    axis.set_xlim(2.0, 25.0)
    axis.set(
        xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)",
        title="Screen-positive held-out ideal EEG spectra",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_02_confirmation_psd.png", dpi=250)
    plt.close(figure)

    indexed = summaries.set_index("metric")
    names = [PRIMARY_METRIC, "selected_target_distance_improvement_log10", *CONTROL_FAMILY]
    labels = ["alpha suppression", "movement to B", "vs opposite", "vs transverse"]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    for position, (name, label) in enumerate(zip(names, labels)):
        values = metrics[name].to_numpy(float)
        axis.scatter(np.full(values.size, position), values, alpha=0.7)
        row = indexed.loc[name]
        axis.errorbar(
            position, row["mean"],
            yerr=[[row["mean"] - row["ci_2.5"]], [row["ci_97.5"] - row["mean"]]],
            color="black", marker="o", capsize=4,
        )
    axis.axhline(0.0, color="0.45", linewidth=0.8)
    axis.set_xticks(range(len(labels)), labels, rotation=15)
    axis.set_ylabel("Paired log10 alpha-power effect")
    axis.set_title("Frozen protocol and causal controls in screened circuits")
    figure.tight_layout()
    figure.savefig(root / "figure_03_paired_confirmation_effects.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen = _load_frozen_candidate(cfg)
    _validate_design(cfg, frozen)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "alpha_suppression_screened_confirmation"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### Screened frozen-dose alpha confirmation")
        print(json.dumps(_plain(frozen), indent=2))
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()

    started = time.perf_counter()
    rng = np.random.default_rng(int(cfg.experiment.seed) + 2_100_003)
    candidate = frozen["candidate"]
    target_model = frozen["target"]["target_model"]
    depth = float(frozen["target"]["selected_modulation_depth"])
    amplitude = float(candidate["selected_dose_v_per_m"])
    selected_offset = float(candidate["relative_phase_offset_rad"])
    axial = str(candidate["montage"])
    transverse = str(cfg.analysis.tacs.transverse_montage)
    target_count = int(cfg.analysis.screening.target_enrolled_seeds)

    screening_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    psds: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    frequencies = None
    enrolled_seeds: list[int] = []

    for screening_order, seed in enumerate(_candidate_seeds(cfg), start=1):
        if _screening_target_reached(
            comm=comm,
            rank=rank,
            enrolled_count=len(enrolled_seeds),
            target_count=target_count,
        ):
            break
        if rank == 0:
            print(f"screen candidate seed={seed}")
            episodes: dict[str, dict[str, Any]] = {}
        a_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=depth)
        a_episode = _run_condition(
            condition_id=A_HIGH,
            condition_cfg=a_cfg,
            action=_sham(cfg, A_HIGH),
            stimulate=False,
            seed=seed,
            action_index=0,
            output_dir=root / "episodes" / A_HIGH / f"seed_{seed}",
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            screen_phase = _screen_phase_quality(
                episode=a_episode,
                relative_offset=selected_offset,
                cfg=cfg,
            )
            screen = _screening_decision(
                seed=seed,
                screening_order=screening_order,
                a_episode=a_episode,
                phase_quality=screen_phase,
                target_model=target_model,
                cfg=cfg,
            )
            screening_rows.append(screen)
            eligible = bool(screen["eligible"])
            print(
                f"screen seed={seed}: {'ELIGIBLE' if eligible else 'EXCLUDED'} "
                f"({screen['exclusion_reasons']})"
            )
        else:
            eligible = None
        eligible = bool(comm.bcast(eligible, root=0))
        if not eligible:
            continue

        if rank == 0:
            episodes[A_HIGH] = a_episode
            enrolled_seeds.append(seed)
            epoch_rows.extend(a_episode["epoch_rows"])
            if bool(cfg.experiment.plot):
                frequencies, psd = _stimulation_psd(a_episode, cfg=cfg)
                psds[A_HIGH].append(psd)
        b_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=0.0)
        b_episode = _run_condition(
            condition_id=B_LOW,
            condition_cfg=b_cfg,
            action=_sham(cfg, B_LOW),
            stimulate=False,
            seed=seed,
            action_index=1,
            output_dir=root / "episodes" / B_LOW / f"seed_{seed}",
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            episodes[B_LOW] = b_episode
            epoch_rows.extend(b_episode["epoch_rows"])
            if bool(cfg.experiment.plot):
                frequencies, psd = _stimulation_psd(b_episode, cfg=cfg)
                psds[B_LOW].append(psd)

        specifications = (
            (SELECTED, selected_offset, axial, "frozen_primary_0p4_vpm"),
            (
                OPPOSITE,
                _wrap_phase(selected_offset + np.pi),
                axial,
                "opposite_phase_control",
            ),
            (TRANSVERSE, selected_offset, transverse, "orientation_control"),
        )
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
                episodes[condition] = episode
                epoch_rows.extend(episode["epoch_rows"])
                if bool(cfg.experiment.plot):
                    frequencies, psd = _stimulation_psd(episode, cfg=cfg)
                    psds[condition].append(psd)
        if rank == 0:
            quality = _phase_quality_row(
                seed=seed,
                a_episode=episodes[A_HIGH],
                selected_episode=episodes[SELECTED],
                selected_offset=selected_offset,
                cfg=cfg,
            )
            phase_rows.append(quality)
            metric_rows.append(_enrolled_seed_metrics(
                seed=seed,
                episodes=episodes,
                screening=screening_rows[-1],
                phase_quality=quality,
                cfg=cfg,
            ))
            del episodes

    if rank == 0:
        screening_frame = pd.DataFrame(screening_rows)
        screening_frame.to_csv(root / "screening_audit.csv", index=False)
        if not metric_rows:
            raise RuntimeError(
                "No candidate seed passed the frozen EEG/phase screen; see screening_audit.csv."
            )
        epoch_frame = pd.DataFrame(epoch_rows)
        metrics = pd.DataFrame(metric_rows)
        phase_frame = pd.DataFrame(phase_rows)
        summaries = _metric_summaries(metrics, cfg=cfg, rng=rng)
        checks, conclusions = _conclusions(
            screening=screening_frame,
            metrics=metrics,
            summaries=summaries,
            epoch_rows=epoch_frame,
            target_model=target_model,
            cfg=cfg,
        )
        epoch_frame.to_csv(
            root / "screened_confirmation_epoch_eeg_and_hidden_metrics.csv",
            index=False,
        )
        metrics.to_csv(root / "screened_confirmation_seed_metrics.csv", index=False)
        phase_frame.to_csv(root / "enrolled_phase_quality.csv", index=False)
        summaries.to_csv(root / "screened_confirmation_summary.csv", index=False)
        frozen_copy = {
            **frozen,
            "candidate_seed_order": _candidate_seeds(cfg),
            "screened_seeds": screening_frame.seed.to_list(),
            "enrolled_seeds": enrolled_seeds,
            "selection_performed_in_confirmation": False,
            "screening_rule": {
                "alpha_threshold": float(target_model["classification_threshold"]),
                "A_is_above_threshold": bool(target_model["A_is_above_threshold"]),
                "phase_split_error_max_deg": float(
                    cfg.analysis.phase_quality.maximum_split_half_error_deg
                ),
                "phase_resultant_to_rms_min": float(
                    cfg.analysis.phase_quality.minimum_10hz_resultant_to_rms
                ),
                "uses_stimulation_outcome": False,
                "uses_seed_specific_B": False,
            },
        }
        (root / "frozen_candidate_provenance.json").write_text(
            json.dumps(_plain(frozen_copy), indent=2)
        )
        result = {
            "scope": "ideal neural-only simulated EEG in a prospectively screened toy subgroup",
            "checks": checks,
            "conclusions": conclusions,
            "primary_metric": PRIMARY_METRIC,
            "primary_summary": _plain(
                summaries[summaries.metric.eq(PRIMARY_METRIC)].iloc[0].to_dict()
            ),
            "selection_performed_in_confirmation": False,
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": (
                "A positive result applies only to toy circuits that pass the "
                "frozen elevated-alpha and phase-actionability screen. Screening "
                "yield is not human prevalence, and the result does not validate "
                "depression biology, treatment efficacy, or an RL policy."
            ),
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        if bool(cfg.experiment.plot):
            _plot_results(
                root=root,
                screening=screening_frame,
                psds=psds,
                frequencies=np.asarray(frequencies),
                metrics=metrics,
                summaries=summaries,
                target_model=target_model,
            )
        print("\n### Prospective screening")
        print(f"candidate seeds screened: {len(screening_frame)}")
        print(f"eligible seeds enrolled: {int(screening_frame.eligible.sum())}")
        print(f"screening yield: {float(screening_frame.eligible.mean()):.3f}")
        print("\n### Screened confirmation checks")
        for name, passed in checks.items():
            print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
        print(
            "\nIdeal neural-EEG control in screen-positive subgroup:",
            "CONFIRMED"
            if conclusions[
                "ideal_neural_eeg_control_in_screen_positive_subgroup_confirmed"
            ] else "NOT CONFIRMED",
        )
        print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
