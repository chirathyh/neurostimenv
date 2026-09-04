"""H5-P2B active response mapping after the H5-P2A tracker crossover.

H5-P2A established, without stimulation, a frozen measurement-layer
bias--variance crossover between the conservative 1-s/250-ms tracker and the
responsive 0.5-s/125-ms tracker. H5-P2B asks whether that crossover transfers
to a causal tACS endpoint on new circuit structures. It crosses biological
phase diffusion with frozen AR(1) observation-noise severity, applies both
profiles and sham through paired stochastic futures, and evaluates efficacy
using ideal neural-only EEG distance to the frozen population-B target.

This is full-information discovery. Hidden generator labels are evaluation
strata only; the carrier, phase trackers, and candidate deployable features see
preceding noisy EEG. No machine-learning policy is fitted or confirmed here.
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
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy import signal, stats


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (  # noqa: E402
    CONSERVATIVE,
    FULL,
    RESPONSIVE,
    _augment_observation_rows,
    _future_seed,
    _noise_seeds,
    _run_controller,
    _with_context_state,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_phase_tracker_tradeoff import (  # noqa: E402
    _phase_estimate,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_response_mapping import (  # noqa: E402
    P1_CONTEXT_FEATURES,
    _frozen_carrier_screen,
    _hash_locked_files,
    _load_sources as _load_p1_sources,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (  # noqa: E402
    HIGH,
    LOW,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (  # noqa: E402
    SHAM,
    _controller_modes,
    _metric_rows,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_bandwidth_discovery import (  # noqa: E402
    _augment_metric_rows,
    _json_ready,
    _profile,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_cadence_discovery import (  # noqa: E402
    _augment_common_audit,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _process_eeg,
)


ROOT_NAME = "h5_phase_tracker_response_mapping"
LOW_NOISE = "low_observation_noise"
HIGH_NOISE = "high_observation_noise"
EXPECTED_MODES = [SHAM, CONSERVATIVE, RESPONSIVE]
P2B_NEW_FEATURES = [
    "pre_fast_mean_abs_innovation_rad",
    "pre_slow_mean_abs_innovation_rad",
    "pre_fast_minus_slow_innovation_rad",
    "pre_fast_innovation_coherence",
    "pre_slow_innovation_coherence",
    "pre_fast_mean_resultant_to_rms",
    "pre_slow_mean_resultant_to_rms",
    "pre_fast_minus_slow_resultant_to_rms",
    "pre_fast_slow_phase_disagreement_rad",
    "pre_offcarrier_to_alpha_log10_power_ratio",
]
P2B_CONTEXT_FEATURES = list(dict.fromkeys(P1_CONTEXT_FEATURES + P2B_NEW_FEATURES))


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    """Load the frozen target and hash-lock P1 plus positive P2A."""
    sources = _load_p1_sources(cfg)

    p1_root = Path(to_absolute_path(str(cfg.analysis.source_h5p1.result_dir)))
    p1_names = {
        "conclusion": "experiment_conclusion.json",
        "audit": "H5_P1_response_mapping_audit.json",
        "screening": "prospective_screening.csv",
        "metrics": "context_controller_future_metrics.csv",
        "response_map": "controller_profile_response_map.csv",
        "associations": "EEG_feature_response_associations.csv",
        "provenance": "protocol_and_provenance.json",
    }
    p1_files, p1_hashes = _hash_locked_files(
        p1_root, p1_names, cfg.analysis.source_h5p1.expected_sha256
    )
    p1_conclusion = json.loads(p1_files["conclusion"].read_text())
    expected_p1_failures = {
        "carrier_identification_coverage_replicated",
        "field_removal_recovered",
        "expected_oracle_has_practical_advantage_over_best_fixed",
        "expected_oracle_has_practical_advantage_over_H4_profile",
        "realized_optimal_profile_reproducible_across_futures",
        "at_least_one_observed_EEG_feature_maps_relative_response",
    }
    if (
        p1_conclusion["conclusions"]["H5_P1_contextual_response_mapping"]
        != "NOT PASSED"
        or bool(p1_conclusion["conclusions"]["ready_for_H5_policy_development"])
        or {
            name for name, value in p1_conclusion["checks"].items()
            if not bool(value)
        } != expected_p1_failures
    ):
        raise RuntimeError("H5-P2B requires the exact frozen negative H5-P1 result.")

    p2a_root = Path(to_absolute_path(str(cfg.analysis.source_h5p2a.result_dir)))
    p2a_names = {
        "conclusion": "experiment_conclusion.json",
        "frozen_conditions": "frozen_h5_p2_conditions.json",
        "protocol": "protocol_and_provenance.json",
        "candidate_selection": "candidate_noise_pair_selection.csv",
        "carrier": "carrier_measurement_by_noise.csv",
        "tracking": "causal_phase_tracker_boundaries.csv",
        "context_summary": "context_tracker_summary.csv",
        "advantage": "context_tracker_advantage.csv",
        "structure": "structure_directional_tradeoff.csv",
    }
    p2a_files, p2a_hashes = _hash_locked_files(
        p2a_root, p2a_names, cfg.analysis.source_h5p2a.expected_sha256
    )
    p2a_conclusion = json.loads(p2a_files["conclusion"].read_text())
    frozen = json.loads(p2a_files["frozen_conditions"].read_text())
    if (
        not all(bool(value) for value in p2a_conclusion["checks"].values())
        or p2a_conclusion["conclusions"][
            "H5_P2A_phase_tracker_bias_variance_tradeoff"
        ] != "PASSED"
        or not bool(p2a_conclusion["conclusions"][
            "ready_for_active_H5_P2B_response_mapping"
        ])
        or not bool(frozen["gate_passed"])
        or not bool(frozen["ready_for_active_H5_P2B_mapping"])
        or not np.isclose(float(frozen["fixed_low_noise_fraction"]), 0.25)
        or not np.isclose(float(frozen["selected_high_noise_fraction"]), 0.50)
        or not np.isclose(float(frozen["fixed_shared_modulated_fraction"]), 1.0)
    ):
        raise RuntimeError("H5-P2B requires the exact positive H5-P2A result.")
    expected_profiles = {
        CONSERVATIVE: {"history_ms": 1000.0, "update_interval_ms": 250.0},
        RESPONSIVE: {"history_ms": 500.0, "update_interval_ms": 125.0},
    }
    if frozen["tracker_profiles"] != expected_profiles:
        raise RuntimeError("H5-P2A frozen tracker profiles changed.")

    source_seeds = set(sources["source_seed_union"])
    for path in (
        p1_files["screening"], p1_files["metrics"],
        p2a_files["carrier"], p2a_files["tracking"],
    ):
        table = pd.read_csv(path)
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed", "noise_seed", "history_noise_seed",
            "future_noise_seed",
        ):
            if column in table:
                source_seeds.update(table[column].dropna().astype(int).tolist())
    sources.update({
        "roots": {
            **sources["roots"], "h5p1": str(p1_root), "h5p2a": str(p2a_root)
        },
        "hashes": {
            **sources["hashes"], "h5p1": p1_hashes, "h5p2a": p2a_hashes
        },
        "source_seed_union": source_seeds,
        "H5P1_negative_preserved": True,
        "H5P2A_confirmed": True,
        "p2a_frozen": frozen,
    })
    return sources


def _noise_conditions(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "observation_noise_label": str(value.label),
        "observation_noise_fraction": float(
            value.rms_fraction_of_baseline_neural_eeg
        ),
    } for value in cfg.analysis.measurement_conditions]


def _diffusion_conditions(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "label": str(value.label),
        "diffusion_rad2_per_s": float(value.diffusion_rad2_per_s),
    } for value in cfg.analysis.states.phase_diffusion_levels]


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    rows: list[dict[str, Any]] = []
    future_group = 0
    for structure_index in range(int(block.n_structure_seeds)):
        structure_seed = base + int(block.structure_seed_offset) + structure_index
        for history_index in range(int(block.n_history_seeds)):
            history_seed = (
                base + int(block.history_seed_offset)
                + 10 * structure_index + history_index
            )
            for frequency_index, frequency in enumerate(
                cfg.analysis.states.frequencies_hz
            ):
                phase_seed = (
                    base + int(block.phase_seed_offset)
                    + 10 * structure_index + frequency_index
                )
                for diffusion_index, diffusion in enumerate(
                    _diffusion_conditions(cfg)
                ):
                    paired_id = (
                        f"s{structure_index:02d}_h{history_index:02d}_"
                        f"f{int(round(float(frequency))):02d}_d{diffusion_index:02d}"
                    )
                    trial_seed = base + int(block.trial_seed_offset) + future_group
                    for noise_index, noise in enumerate(_noise_conditions(cfg)):
                        rows.append({
                            "context_order": len(rows),
                            "context_id": (
                                f"{paired_id}_n{noise_index:02d}_"
                                f"{diffusion['label']}_{noise['observation_noise_label']}"
                            ),
                            "paired_noise_context_id": paired_id,
                            "paired_shared_drive_context_id": paired_id,
                            "future_group_index": future_group,
                            "structure_index": structure_index,
                            "structure_seed": structure_seed,
                            "history_index": history_index,
                            "history_seed": history_seed,
                            "phase_seed": phase_seed,
                            "trial_seed": trial_seed,
                            "hidden_frequency_hz": float(frequency),
                            **diffusion,
                            "shared_drive_label": FULL,
                            "shared_modulated_fraction": 1.0,
                            **noise,
                        })
                    future_group += 1
    return rows


def _run_context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    rows = _context_specs(cfg)
    if not bool(cfg.analysis.smoke_test):
        return rows
    limit = int(cfg.analysis.smoke_context_limit)
    if limit <= 0:
        return rows
    preferred = []
    for frequency in (9.0, 11.0):
        for diffusion in (HIGH, LOW):
            for noise in (LOW_NOISE, HIGH_NOISE):
                preferred.append(next(
                    row for row in rows
                    if row["hidden_frequency_hz"] == frequency
                    and row["label"] == diffusion
                    and row["observation_noise_label"] == noise
                ))
    return preferred[:limit]


def _with_noise_fraction(cfg: DictConfig, fraction: float) -> DictConfig:
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(result):
        result.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg = (
            float(fraction)
        )
    return result


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    smoke = bool(cfg.analysis.smoke_test)
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-P2B requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-P2B may not alter recurrent inhibition.")
    if [float(value) for value in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-P2B freezes the 9/11-Hz carrier grid.")
    if [(x["label"], x["diffusion_rad2_per_s"])
        for x in _diffusion_conditions(cfg)] != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("H5-P2B freezes D={0.5,2.0} rad^2/s.")
    shared = list(cfg.analysis.states.shared_drive_levels)
    if len(shared) != 1 or str(shared[0].label) != FULL or not np.isclose(
        float(shared[0].shared_modulated_fraction), 1.0
    ):
        raise ValueError("H5-P2B freezes q=1.0 shared afferent drive.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-P2B retains modulation depth 0.04.")
    if _noise_conditions(cfg) != [
        {"observation_noise_label": LOW_NOISE, "observation_noise_fraction": 0.25},
        {"observation_noise_label": HIGH_NOISE, "observation_noise_fraction": 0.50},
    ]:
        raise ValueError("H5-P2B must use the P2A-frozen 0.25/0.50 noise pair.")
    if _controller_modes(cfg) != EXPECTED_MODES:
        raise ValueError(f"H5-P2B controller order must be {EXPECTED_MODES}.")
    for mode, expected in {
        CONSERVATIVE: {"adaptive": True, "history_ms": 1000.0,
                       "update_interval_ms": 250.0},
        RESPONSIVE: {"adaptive": True, "history_ms": 500.0,
                     "update_interval_ms": 125.0},
    }.items():
        if _profile(cfg, mode) != expected:
            raise ValueError(f"H5-P2B controller profile changed: {mode}.")
    if not np.isclose(float(cfg.analysis.actions.amplitude_v_per_m), 0.2):
        raise ValueError("Both active H5-P2B profiles must use 0.2 V/m.")
    if str(cfg.analysis.response_mapping.frozen_estimator) != str(
        sources["p2a_frozen"]["carrier_estimator"]
    ):
        raise ValueError("H5-P2B carrier estimator differs from P2A.")
    if str(cfg.analysis.response_mapping.safe_abstention_action) != SHAM:
        raise ValueError("H5-P2B carrier abstention must map to sham.")
    if not bool(cfg.analysis.observation_noise.enabled) or not np.isclose(
        float(cfg.analysis.observation_noise.ar1_coefficient), 0.95
    ):
        raise ValueError("H5-P2B retains the frozen AR(1) observation model.")
    if float(cfg.env.simulation.obs_win_len) != 1000.0:
        raise ValueError("H5-P2B requires one-second outer online windows.")
    if not smoke and (
        int(cfg.analysis.timeline.baseline_steps) != 30
        or int(cfg.analysis.timeline.stimulation_steps) != 8
        or int(cfg.analysis.timeline.washout_steps) != 1
        or int(cfg.analysis.crossed_design.n_structure_seeds) != 6
        or int(cfg.analysis.crossed_design.n_history_seeds) != 1
        or int(cfg.analysis.crossed_design.n_future_continuations) != 4
    ):
        raise ValueError("Full H5-P2B requires the frozen 30/8/1-s, 6x1x4 design.")
    if not np.isclose(float(sources["target"]["outcome_duration_s"]), 8.0):
        raise ValueError("The frozen population-B outcome duration changed.")

    contexts = _context_specs(cfg)
    expected_count = int(cfg.analysis.crossed_design.n_structure_seeds) * 2 * 2 * 2
    if len(contexts) != expected_count:
        raise ValueError("H5-P2B crossed context grid is incomplete.")
    namespaces = [
        {int(row[column]) for row in contexts}
        for column in ("structure_seed", "history_seed", "phase_seed", "trial_seed")
    ]
    namespaces.append({
        _future_seed(cfg, row, future)
        for row in contexts
        for future in range(int(cfg.analysis.crossed_design.n_future_continuations))
    })
    namespaces.append({
        seed
        for row in contexts
        for future in range(int(cfg.analysis.crossed_design.n_future_continuations))
        for seed in _noise_seeds(cfg, row, future)
    })
    if any(not values for values in namespaces):
        raise ValueError("Every H5-P2B seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H5-P2B seed namespaces overlap.")
    if set().union(*namespaces).intersection(sources["source_seed_union"]):
        raise ValueError("H5-P2B seeds overlap an upstream H5 experiment.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H5-P2B structure seed exceeds the uint32 mapping range.")
    forbidden = ("hidden", "diffusion", "noise_fraction", "spike", "phase_rad")
    if any(any(token in name for token in forbidden) for name in P2B_CONTEXT_FEATURES):
        raise ValueError("A deployable H5-P2B feature exposes a hidden label.")


def _predecision_tracker_features(
    episode: dict[str, Any], selected_frequency_hz: float, cfg: DictConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Extract causal phase-tracker diagnostics from noisy baseline EEG only."""
    outputs = episode["simulation"]["observed_outputs_by_epoch"]["baseline"]
    values = np.concatenate([
        np.asarray(output["eeg_v"], dtype=float).reshape(-1) for output in outputs
    ])
    times = np.concatenate([
        np.asarray(output["sample_times_ms"], dtype=float).reshape(-1)
        for output in outputs
    ])
    frequency_cfg = _with_action_frequency(cfg, float(selected_frequency_hz))
    block = cfg.analysis.response_mapping
    stop_ms = float(outputs[-1]["t_stop_ms"])
    start_ms = stop_ms - 1000.0 * float(block.predecision_tracker_diagnostic_tail_s)
    interval_ms = float(block.predecision_common_interval_ms)
    boundaries = np.arange(start_ms + interval_ms, stop_ms + 1.0e-9, interval_ms)
    profile_phases: dict[str, list[float]] = {name: [] for name in (CONSERVATIVE, RESPONSIVE)}
    innovations: dict[str, list[float]] = {name: [] for name in (CONSERVATIVE, RESPONSIVE)}
    resultants: dict[str, list[float]] = {name: [] for name in (CONSERVATIVE, RESPONSIVE)}
    diagnostics: list[dict[str, Any]] = []
    carrier = float(selected_frequency_hz)
    for mode in (CONSERVATIVE, RESPONSIVE):
        profile = _profile(cfg, mode)
        history_ms = float(profile["history_ms"])
        update_ms = float(profile["update_interval_ms"])
        estimate = _phase_estimate(
            values, times, boundary_ms=start_ms, history_ms=history_ms,
            frequency_cfg=frequency_cfg, cfg=cfg,
        )
        last_update_ms = start_ms
        previous_update_phase = float(estimate["estimated_eeg_phase_at_boundary_rad"])
        for index, boundary in enumerate(boundaries, start=1):
            elapsed = float(boundary - start_ms)
            applied = bool(np.isclose(
                elapsed / update_ms, round(elapsed / update_ms), atol=1.0e-10
            ))
            innovation = float("nan")
            if applied:
                new_estimate = _phase_estimate(
                    values, times, boundary_ms=float(boundary), history_ms=history_ms,
                    frequency_cfg=frequency_cfg, cfg=cfg,
                )
                expected_phase = previous_update_phase + 2.0 * np.pi * carrier * (
                    float(boundary) - last_update_ms
                ) / 1000.0
                innovation = float(_wrap_phase(
                    float(new_estimate["estimated_eeg_phase_at_boundary_rad"])
                    - expected_phase
                ))
                innovations[mode].append(innovation)
                resultants[mode].append(float(new_estimate["resultant_to_rms"]))
                estimate = new_estimate
                previous_update_phase = float(
                    new_estimate["estimated_eeg_phase_at_boundary_rad"]
                )
                last_update_ms = float(boundary)
            propagated = float(_wrap_phase(
                float(estimate["estimated_eeg_phase_at_boundary_rad"])
                + 2.0 * np.pi * carrier * (float(boundary) - last_update_ms) / 1000.0
            ))
            profile_phases[mode].append(propagated)
            diagnostics.append({
                "tracker_profile": mode,
                "boundary_index": int(index),
                "boundary_ms": float(boundary),
                "profile_update_applied": applied,
                "estimated_phase_rad": propagated,
                "phase_innovation_rad": innovation,
                "resultant_to_rms": float(estimate["resultant_to_rms"]),
                "uses_only_predecision_observed_EEG": True,
            })

    fast = np.asarray(profile_phases[RESPONSIVE], dtype=float)
    slow = np.asarray(profile_phases[CONSERVATIVE], dtype=float)
    disagreement = np.abs(_wrap_phase(fast - slow))
    summary: dict[str, float] = {}
    for prefix, mode in (("slow", CONSERVATIVE), ("fast", RESPONSIVE)):
        innovation_values = np.asarray(innovations[mode], dtype=float)
        resultant_values = np.asarray(resultants[mode], dtype=float)
        summary[f"pre_{prefix}_mean_abs_innovation_rad"] = float(
            np.mean(np.abs(innovation_values))
        )
        summary[f"pre_{prefix}_innovation_coherence"] = float(abs(
            np.mean(np.exp(1j * innovation_values))
        ))
        summary[f"pre_{prefix}_mean_resultant_to_rms"] = float(
            np.mean(resultant_values)
        )
    summary["pre_fast_minus_slow_innovation_rad"] = (
        summary["pre_fast_mean_abs_innovation_rad"]
        - summary["pre_slow_mean_abs_innovation_rad"]
    )
    summary["pre_fast_minus_slow_resultant_to_rms"] = (
        summary["pre_fast_mean_resultant_to_rms"]
        - summary["pre_slow_mean_resultant_to_rms"]
    )
    summary["pre_fast_slow_phase_disagreement_rad"] = float(
        np.mean(disagreement)
    )
    processed, fs_hz, _, _, _ = _process_eeg(
        values, simulator_fs_hz=float(episode["simulator_fs_hz"]), cfg=cfg
    )
    frequencies, psd = signal.welch(
        processed, fs=float(fs_hz), window="hann",
        nperseg=min(len(processed), int(round(4.0 * fs_hz))),
        noverlap=min(len(processed) // 2, int(round(2.0 * fs_hz))),
        detrend="constant", scaling="density",
    )
    alpha = psd[(frequencies >= 8.0) & (frequencies <= 12.0)]
    off = psd[(frequencies >= 20.0) & (frequencies <= 80.0)]
    tiny = np.finfo(float).tiny
    summary["pre_offcarrier_to_alpha_log10_power_ratio"] = float(np.log10(
        max(float(np.mean(off)), tiny) / max(float(np.mean(alpha)), tiny)
    ))
    if not all(np.isfinite(value) for value in summary.values()):
        raise RuntimeError("H5-P2B produced a non-finite predecision EEG feature.")
    return summary, pd.DataFrame(diagnostics)


def _screen_context(
    episode: dict[str, Any], context: dict[str, Any], target: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    screening, spectrum, temporal = _frozen_carrier_screen(
        episode, context, target, cfg
    )
    features, diagnostics = _predecision_tracker_features(
        episode, float(screening["EEG_selected_frequency_hz"]), cfg
    )
    neural = np.asarray(episode["raw_by_epoch"]["baseline"], dtype=float)
    observed = np.asarray(episode["observed_raw_by_epoch"]["baseline"], dtype=float)
    scale = float(episode["simulation"]["observation"]["baseline_noise_rms_v"])
    standardized_noise = (observed - neural) / max(scale, np.finfo(float).tiny)
    screening.update({
        **features,
        "observation_noise_label": str(context["observation_noise_label"]),
        "observation_noise_fraction": float(context["observation_noise_fraction"]),
        "paired_noise_context_id": str(context["paired_noise_context_id"]),
        "neural_baseline_sha256": hashlib.sha256(
            np.asarray(neural, dtype="<f8").tobytes()
        ).hexdigest(),
        "standardized_noise_sha256": hashlib.sha256(
            # Round only for the audit hash so floating-point cancellation in
            # (neural + scale*noise) - neural cannot make paired paths appear
            # different at the last machine bit.
            np.asarray(np.round(standardized_noise, 10), dtype="<f8").tobytes()
        ).hexdigest(),
        "history_noise_seed": int(
            episode["simulation"]["observation"]["history_noise_seed"]
        ),
        "predecision_features_use_only_observed_EEG": True,
        "configured_noise_or_hidden_diffusion_excluded_from_features": True,
    })
    spectrum = spectrum.assign(
        observation_noise_label=str(context["observation_noise_label"]),
        observation_noise_fraction=float(context["observation_noise_fraction"]),
        paired_noise_context_id=str(context["paired_noise_context_id"]),
    )
    temporal = temporal.assign(
        observation_noise_label=str(context["observation_noise_label"]),
        observation_noise_fraction=float(context["observation_noise_fraction"]),
        paired_noise_context_id=str(context["paired_noise_context_id"]),
    )
    diagnostics = diagnostics.assign(
        context_id=str(context["context_id"]),
        structure_seed=int(context["structure_seed"]),
        observation_noise_label=str(context["observation_noise_label"]),
        observation_noise_fraction=float(context["observation_noise_fraction"]),
        EEG_selected_frequency_hz=float(screening["EEG_selected_frequency_hz"]),
    )
    return screening, spectrum, temporal, diagnostics


def _run_p2b_controller(**kwargs: Any) -> dict[str, Any] | None:
    episode = _run_controller(**kwargs)
    if episode is not None:
        episode["simulation"]["action"]["role"] = (
            "H5_P2B_active_phase_tracker_response_mapping"
        )
    return episode


def _add_context_fields(
    rows: list[dict[str, Any]], screening: dict[str, Any]
) -> None:
    for row in rows:
        for feature in P2B_CONTEXT_FEATURES:
            row[feature] = float(screening[feature])
        row.update({
            "paired_noise_context_id": str(screening["paired_noise_context_id"]),
            "observation_noise_label": str(screening["observation_noise_label"]),
            "observation_noise_fraction": float(screening["observation_noise_fraction"]),
            "carrier_estimator": str(screening["carrier_estimator"]),
            "carrier_identified": bool(screening["carrier_identified"]),
            "carrier_selection_correct_evaluation_only": bool(
                screening["EEG_frequency_selection_correct"]
            ),
            "action_frequency_hz": float(screening["EEG_selected_frequency_hz"]),
            "action_frequency_uses_frozen_EEG_estimator": True,
            "policy_feature_source": "predecision_observed_EEG_only",
            "hidden_generator_and_noise_labels_excluded_from_policy_features": True,
        })


def _expected_map(metrics: pd.DataFrame) -> pd.DataFrame:
    group = [
        "context_id", "paired_noise_context_id", "structure_seed",
        "hidden_frequency_hz", "label", "diffusion_rad2_per_s",
        "observation_noise_label", "observation_noise_fraction",
        "shared_drive_label", "shared_modulated_fraction",
        "EEG_selected_frequency_hz", *P2B_CONTEXT_FEATURES, "controller_mode",
    ]
    return (
        metrics.groupby(group, as_index=False)
        .agg(
            n_futures=("future_index", "nunique"),
            expected_post_distance_to_B_log10=("post_distance_to_B_log10", "mean"),
            future_sd_post_distance_log10=("post_distance_to_B_log10", "std"),
            expected_improvement_vs_sham_log10=(
                "causal_distance_improvement_vs_sham_log10", "mean"
            ),
            expected_alpha_suppression_vs_sham_log10=(
                "causal_alpha_suppression_vs_sham_log10", "mean"
            ),
            all_rate_safe=("rate_safe", "all"),
            all_field_removal_recovered=("field_removal_recovered", "all"),
            mean_abs_common_phase_error_rad=(
                "mean_abs_common_phase_error_rad", "mean"
            ),
            common_phase_estimate_actionable_fraction=(
                "common_phase_estimate_actionable_fraction", "mean"
            ),
            correction_saturation_fraction=(
                "frequency_correction_saturation_fraction", "mean"
            ),
            maximum_field_boundary_discontinuity_v_per_m=(
                "maximum_field_boundary_discontinuity_v_per_m", "max"
            ),
        )
        .sort_values(group)
        .reset_index(drop=True)
    )


def _response_map(
    expected: pd.DataFrame, metrics: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = [
        "context_id", "paired_noise_context_id", "structure_seed",
        "hidden_frequency_hz", "label", "diffusion_rad2_per_s",
        "observation_noise_label", "observation_noise_fraction",
        "shared_drive_label", "shared_modulated_fraction",
        "EEG_selected_frequency_hz", *P2B_CONTEXT_FEATURES,
    ]
    active = expected[expected.controller_mode.isin([CONSERVATIVE, RESPONSIVE])]
    pivot = active.pivot(
        index=keys, columns="controller_mode",
        values=[
            "expected_post_distance_to_B_log10",
            "future_sd_post_distance_log10",
            "mean_abs_common_phase_error_rad",
        ],
    ).reset_index()
    pivot.columns = [
        "_".join(str(part) for part in value if str(part))
        if isinstance(value, tuple) else str(value)
        for value in pivot.columns
    ]
    pivot = pivot.rename(columns={f"{key}_": key for key in keys})
    slow_distance = f"expected_post_distance_to_B_log10_{CONSERVATIVE}"
    fast_distance = f"expected_post_distance_to_B_log10_{RESPONSIVE}"
    slow_phase = f"mean_abs_common_phase_error_rad_{CONSERVATIVE}"
    fast_phase = f"mean_abs_common_phase_error_rad_{RESPONSIVE}"
    pivot["fast_advantage_over_slow_log10"] = (
        pivot[slow_distance] - pivot[fast_distance]
    )
    pivot["fast_phase_error_advantage_rad"] = pivot[slow_phase] - pivot[fast_phase]
    pivot["expected_optimal_profile"] = np.where(
        pivot.fast_advantage_over_slow_log10 > 0.0, RESPONSIVE, CONSERVATIVE
    )
    margin = float(cfg.analysis.response_mapping.practical_context_margin_log10)
    pivot["expected_winner_margin_log10"] = pivot.fast_advantage_over_slow_log10.abs()
    pivot["practical_optimal_profile"] = np.select(
        [
            pivot.fast_advantage_over_slow_log10 >= margin,
            pivot.fast_advantage_over_slow_log10 <= -margin,
        ],
        [RESPONSIVE, CONSERVATIVE], default="no_practical_difference",
    )
    fixed_mean = {
        mode: float(active[active.controller_mode.eq(mode)]
                    .expected_post_distance_to_B_log10.mean())
        for mode in (CONSERVATIVE, RESPONSIVE)
    }
    best_fixed = min(fixed_mean, key=lambda mode: (fixed_mean[mode], mode))
    fixed_column = f"expected_post_distance_to_B_log10_{best_fixed}"
    pivot["best_fixed_profile"] = best_fixed
    pivot["expected_oracle_distance_to_B_log10"] = np.minimum(
        pivot[slow_distance], pivot[fast_distance]
    )
    pivot["oracle_advantage_over_best_fixed_log10"] = (
        pivot[fixed_column] - pivot.expected_oracle_distance_to_B_log10
    )

    realized = metrics[metrics.controller_mode.isin([CONSERVATIVE, RESPONSIVE])]
    agreements, paired_sd = [], []
    for row in pivot.itertuples():
        group = realized[realized.context_id.eq(str(row.context_id))]
        winners, effects = [], []
        for _, future in group.groupby("future_index"):
            by_mode = future.set_index("controller_mode")
            effects.append(float(
                by_mode.loc[CONSERVATIVE, "post_distance_to_B_log10"]
                - by_mode.loc[RESPONSIVE, "post_distance_to_B_log10"]
            ))
            winners.append(str(future.sort_values([
                "post_distance_to_B_log10", "controller_mode"
            ]).iloc[0].controller_mode))
        agreements.append(float(np.mean(
            np.asarray(winners) == str(row.expected_optimal_profile)
        )))
        paired_sd.append(float(np.std(effects, ddof=1)))
    pivot["realized_optimal_profile_agreement_fraction"] = agreements
    pivot["paired_future_response_sd_log10"] = paired_sd

    structure = pivot.groupby("structure_seed", as_index=False).agg(
        context_count=("context_id", "nunique"),
        mean_oracle_advantage_over_best_fixed_log10=(
            "oracle_advantage_over_best_fixed_log10", "mean"
        ),
        mean_realized_optimal_profile_agreement_fraction=(
            "realized_optimal_profile_agreement_fraction", "mean"
        ),
        mean_paired_future_response_sd_log10=(
            "paired_future_response_sd_log10", "mean"
        ),
    )
    high_d_low_noise = pivot[
        pivot.label.eq(HIGH) & pivot.observation_noise_label.eq(LOW_NOISE)
    ]
    low_d_high_noise = pivot[
        pivot.label.eq(LOW) & pivot.observation_noise_label.eq(HIGH_NOISE)
    ]
    direction_rows = []
    for structure_seed in sorted(pivot.structure_seed.unique()):
        fast = high_d_low_noise[
            high_d_low_noise.structure_seed.eq(structure_seed)
        ].fast_advantage_over_slow_log10
        slow = -low_d_high_noise[
            low_d_high_noise.structure_seed.eq(structure_seed)
        ].fast_advantage_over_slow_log10
        direction_rows.append({
            "structure_seed": int(structure_seed),
            "fast_advantage_high_diffusion_low_noise_log10": (
                float(fast.mean()) if len(fast) else float("nan")
            ),
            "slow_advantage_low_diffusion_high_noise_log10": (
                float(slow.mean()) if len(slow) else float("nan")
            ),
        })
    directions = pd.DataFrame(direction_rows)
    directions["both_target_directions_positive"] = (
        (directions.fast_advantage_high_diffusion_low_noise_log10 > 0.0)
        & (directions.slow_advantage_low_diffusion_high_noise_log10 > 0.0)
    )
    optimal_counts = pivot.expected_optimal_profile.value_counts().to_dict()
    optimal_structures = (
        pivot.groupby("expected_optimal_profile").structure_seed.nunique().to_dict()
    )
    practical = pivot[pivot.practical_optimal_profile.ne("no_practical_difference")]
    practical_counts = practical.practical_optimal_profile.value_counts().to_dict()
    practical_structures = (
        practical.groupby("practical_optimal_profile").structure_seed.nunique().to_dict()
    )
    phase_rho = float(stats.spearmanr(
        pivot.fast_phase_error_advantage_rad,
        pivot.fast_advantage_over_slow_log10,
    ).statistic)
    opportunity = {
        "best_fixed_profile": best_fixed,
        "fixed_profile_expected_distance_log10": fixed_mean,
        "oracle_expected_distance_log10": float(
            pivot.expected_oracle_distance_to_B_log10.mean()
        ),
        "mean_oracle_advantage_over_best_fixed_log10": float(
            structure.mean_oracle_advantage_over_best_fixed_log10.mean()
        ),
        "positive_structure_oracle_fraction": float(np.mean(
            structure.mean_oracle_advantage_over_best_fixed_log10 > 0.0
        )),
        "optimal_profile_context_count": optimal_counts,
        "optimal_profile_structure_count": optimal_structures,
        "practical_optimal_profile_context_count": practical_counts,
        "practical_optimal_profile_structure_count": practical_structures,
        "mean_realized_optimal_profile_agreement_fraction": float(
            pivot.realized_optimal_profile_agreement_fraction.mean()
        ),
        "mean_fast_advantage_high_diffusion_low_noise_log10": float(
            high_d_low_noise.fast_advantage_over_slow_log10.mean()
        ),
        "mean_slow_advantage_low_diffusion_high_noise_log10": float(
            -low_d_high_noise.fast_advantage_over_slow_log10.mean()
        ),
        "mean_response_crossover_log10": float(
            high_d_low_noise.fast_advantage_over_slow_log10.mean()
            - low_d_high_noise.fast_advantage_over_slow_log10.mean()
        ),
        "fast_target_direction_structure_fraction": float(np.mean(
            directions.fast_advantage_high_diffusion_low_noise_log10 > 0.0
        )),
        "slow_target_direction_structure_fraction": float(np.mean(
            directions.slow_advantage_low_diffusion_high_noise_log10 > 0.0
        )),
        "mean_fast_phase_advantage_high_diffusion_low_noise_rad": float(
            high_d_low_noise.fast_phase_error_advantage_rad.mean()
        ),
        "mean_slow_phase_advantage_low_diffusion_high_noise_rad": float(
            -low_d_high_noise.fast_phase_error_advantage_rad.mean()
        ),
        "phase_error_advantage_response_spearman_rho": phase_rho,
        "oracle_is_post_hoc_full_information_and_not_deployable": True,
    }
    return pivot, structure, directions, opportunity


def _feature_response_associations(
    action_map: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = action_map.fast_advantage_over_slow_log10.to_numpy(float)
    groups = [
        np.flatnonzero(action_map.structure_seed.to_numpy() == seed)
        for seed in sorted(action_map.structure_seed.unique())
    ]
    centered_y = y.copy()
    for indices in groups:
        centered_y[indices] -= np.mean(centered_y[indices])
    draws = int(cfg.analysis.response_mapping.association_permutations)
    if bool(cfg.analysis.smoke_test):
        draws = min(draws, 199)
    rng = np.random.default_rng(
        int(cfg.experiment.seed)
        + int(cfg.analysis.response_mapping.association_seed_offset)
    )
    permuted = np.empty((draws, len(action_map)), dtype=float)
    for draw in range(draws):
        permuted[draw] = centered_y
        for indices in groups:
            permuted[draw, indices] = rng.permutation(centered_y[indices])
    rows = []
    for feature in P2B_CONTEXT_FEATURES:
        x = action_map[feature].to_numpy(float)
        centered_x = x.copy()
        for indices in groups:
            centered_x[indices] -= np.mean(centered_x[indices])
        rho = float(stats.spearmanr(centered_x, centered_y).statistic)
        if np.isfinite(rho):
            null = np.asarray([
                stats.spearmanr(centered_x, permuted[draw]).statistic
                for draw in range(draws)
            ], dtype=float)
            null = null[np.isfinite(null)]
            p_value = float((1 + np.sum(np.abs(null) >= abs(rho))) / (1 + len(null)))
        else:
            p_value = 1.0
        signs = []
        for indices in groups:
            local = float(stats.spearmanr(x[indices], y[indices]).statistic)
            if np.isfinite(local) and not np.isclose(local, 0.0):
                signs.append(float(np.sign(local)))
        dominant_fraction = (
            max(signs.count(1.0), signs.count(-1.0)) / len(signs)
            if signs else 0.0
        )
        rows.append({
            "feature": feature,
            "structure_centered_spearman_rho": rho,
            "structure_preserving_permutation_p_value": p_value,
            "dominant_structure_sign_fraction": float(dominant_fraction),
            "permutation_draws": draws,
            "uses_predecision_observed_EEG_only": True,
        })
    table = pd.DataFrame(rows)
    order = np.argsort(table.structure_preserving_permutation_p_value.to_numpy(float))
    ranked = (
        table.structure_preserving_permutation_p_value.to_numpy(float)[order]
        * len(table) / np.arange(1, len(table) + 1)
    )
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty(len(table), dtype=float)
    q_values[order] = np.minimum(ranked, 1.0)
    table["FDR_q_value"] = q_values
    criteria = cfg.analysis.criteria
    table["passes_response_association_gate"] = (
        table.structure_centered_spearman_rho.abs().ge(
            float(criteria.minimum_response_feature_abs_spearman)
        )
        & table.FDR_q_value.le(float(criteria.maximum_response_feature_fdr_q))
        & table.dominant_structure_sign_fraction.ge(
            float(criteria.minimum_response_feature_structure_sign_fraction)
        )
    )
    passing = table[table.passes_response_association_gate]
    selected = None
    if len(passing):
        selected = str(passing.sort_values(
            ["FDR_q_value", "structure_centered_spearman_rho"],
            ascending=[True, False],
        ).iloc[0].feature)
    return table.sort_values("FDR_q_value").reset_index(drop=True), {
        "selected_candidate_response_feature": selected,
        "at_least_one_feature_passes": selected is not None,
        "outcome_is_expected_paired_fast_minus_slow_response": True,
        "features_use_only_predecision_observed_EEG": True,
        "configured_noise_and_hidden_generator_labels_excluded": True,
        "association_is_exploratory_not_a_fitted_policy": True,
    }


def _carrier_by_noise(screening: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in screening.groupby("observation_noise_label", sort=False):
        accepted = group[group.carrier_identified.astype(bool)]
        rows.append({
            "observation_noise_label": label,
            "observation_noise_fraction": float(group.observation_noise_fraction.iloc[0]),
            "context_count": int(len(group)),
            "carrier_coverage": float(group.carrier_identified.mean()),
            "accepted_carrier_accuracy": (
                float(accepted.EEG_frequency_selection_correct.mean())
                if len(accepted) else float("nan")
            ),
            "enrollment_fraction": float(group.eligible.mean()),
        })
    return pd.DataFrame(rows)


def _checks(
    *, screening: pd.DataFrame, metrics: pd.DataFrame, expected: pd.DataFrame,
    updates: pd.DataFrame, action_map: pd.DataFrame, structure: pd.DataFrame,
    directions: pd.DataFrame, opportunity: dict[str, Any],
    associations: pd.DataFrame, association_audit: dict[str, Any],
    sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    smoke = bool(cfg.analysis.smoke_test)
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible.astype(bool)]
    active = metrics[metrics.controller_mode.ne(SHAM)]
    carrier = _carrier_by_noise(screening).set_index("observation_noise_label")
    target_counts = action_map.groupby([
        "label", "observation_noise_label"
    ]).context_id.nunique()
    profile_counts = opportunity["optimal_profile_context_count"]
    profile_structures = opportunity["optimal_profile_structure_count"]
    practical_counts = opportunity["practical_optimal_profile_context_count"]
    practical_structures = opportunity["practical_optimal_profile_structure_count"]
    checks = {
        "source_H5P1_hash_locked_with_response_map_failed": bool(
            sources["H5P1_negative_preserved"]
        ),
        "source_H5P2A_passed_hash_locked_and_noise_pair_frozen": bool(
            sources["H5P2A_confirmed"]
        ),
        "H5P2B_seeds_disjoint_from_all_sources": True,
        "complete_frequency_diffusion_noise_screening_grid": bool(
            len(screening) == len(_run_context_specs(cfg))
        ),
        "screening_uses_only_predecision_observed_EEG": bool(
            screening.screen_uses_only_predecision_observed_EEG.astype(bool).all()
            and screening.predecision_features_use_only_observed_EEG.astype(bool).all()
        ),
        "hidden_generator_and_configured_noise_excluded_from_policy_features": bool(
            screening.configured_noise_or_hidden_diffusion_excluded_from_features
            .astype(bool).all()
        ),
        "full_shared_afferent_fraction_fixed": bool(
            np.allclose(screening.shared_modulated_fraction, 1.0)
        ),
        "paired_neural_baseline_identical_across_noise_levels": bool(
            screening.groupby("paired_noise_context_id")
            .neural_baseline_sha256.nunique().eq(1).all()
        ),
        "paired_standardized_noise_path_identical_across_noise_levels": bool(
            screening.groupby("paired_noise_context_id")
            .standardized_noise_sha256.nunique().eq(1).all()
        ),
        "minimum_screened_and_eligible_contexts": bool(
            smoke or (
                len(screening) >= int(criteria.minimum_screened_contexts)
                and len(eligible) >= int(criteria.minimum_eligible_contexts)
            )
        ),
        "minimum_independent_structures": bool(
            screening.structure_seed.nunique()
            >= (1 if smoke else int(criteria.minimum_structure_seeds))
        ),
        "carrier_coverage_and_accuracy_replicate_at_both_noise_levels": bool(
            smoke or all(
                float(carrier.loc[label, "carrier_coverage"])
                >= float(criteria.minimum_carrier_identification_coverage_each_noise)
                and float(carrier.loc[label, "accepted_carrier_accuracy"])
                >= float(criteria.minimum_accepted_carrier_accuracy_each_noise)
                for label in (LOW_NOISE, HIGH_NOISE)
            )
        ),
        "complete_controller_grid_for_enrolled_contexts": bool(
            expected.groupby("context_id").controller_mode.nunique().eq(3).all()
        ),
        "four_independent_paired_futures": bool(
            metrics.groupby(["context_id", "controller_mode"])
            .future_index.nunique().min()
            >= (1 if smoke else int(criteria.minimum_future_continuations))
        ),
        "identical_predecision_observed_EEG_across_actions_and_futures": bool(
            metrics.groupby("context_id").observed_baseline_sha256.nunique().eq(1).all()
        ),
        "single_frozen_controller_profile_per_intervention": bool(
            active.groupby(["context_id", "future_index", "controller_mode"])
            .size().eq(1).all()
        ),
        "both_active_profiles_use_identical_0p2_V_per_m": bool(
            np.allclose(active.amplitude_v_per_m, 0.2)
        ),
        "action_frequency_uses_frozen_noisy_EEG_estimator": bool(
            active.action_frequency_uses_frozen_EEG_estimator.astype(bool).all()
        ),
        "phase_updates_use_only_preceding_observed_EEG": bool(
            active.all_phase_estimates_causal.astype(bool).all()
        ),
        "field_waveform_continuous_and_frequency_bounded": bool(
            active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
            and active.maximum_abs_frequency_correction_hz.max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "all_actions_rate_safe": bool(metrics.rate_safe.astype(bool).all()),
        "field_removal_recovered": bool(
            metrics.field_removal_recovered.astype(bool).all()
            and np.isclose(metrics.final_extracellular_residual_mV, 0.0).all()
        ),
        "both_target_corners_have_minimum_contexts": bool(
            smoke or (
                int(target_counts.get((HIGH, LOW_NOISE), 0))
                >= int(criteria.minimum_target_corner_contexts)
                and int(target_counts.get((LOW, HIGH_NOISE), 0))
                >= int(criteria.minimum_target_corner_contexts)
            )
        ),
        "expected_oracle_uses_both_profiles": bool(all(
            int(profile_counts.get(mode, 0))
            >= (1 if smoke else int(criteria.minimum_contexts_per_optimal_profile))
            and int(profile_structures.get(mode, 0))
            >= (1 if smoke else int(criteria.minimum_structures_per_optimal_profile))
            for mode in (CONSERVATIVE, RESPONSIVE)
        )),
        "both_profiles_have_practical_contextual_support": bool(all(
            int(practical_counts.get(mode, 0))
            >= (0 if smoke else int(criteria.minimum_practical_contexts_per_profile))
            and int(practical_structures.get(mode, 0))
            >= (0 if smoke else int(criteria.minimum_practical_structures_per_profile))
            for mode in (CONSERVATIVE, RESPONSIVE)
        )),
        "fast_profile_improves_high_diffusion_low_noise": bool(
            opportunity["mean_fast_advantage_high_diffusion_low_noise_log10"]
            >= (0.0 if smoke else float(
                criteria.minimum_fast_advantage_high_diffusion_low_noise_log10
            ))
        ),
        "slow_profile_improves_low_diffusion_high_noise": bool(
            opportunity["mean_slow_advantage_low_diffusion_high_noise_log10"]
            >= (0.0 if smoke else float(
                criteria.minimum_slow_advantage_low_diffusion_high_noise_log10
            ))
        ),
        "response_crossover_is_practically_meaningful": bool(
            opportunity["mean_response_crossover_log10"]
            >= (0.0 if smoke else float(criteria.minimum_response_crossover_log10))
        ),
        "target_directions_positive_across_structures": bool(
            smoke or (
                opportunity["fast_target_direction_structure_fraction"]
                >= float(criteria.minimum_target_direction_structure_fraction)
                and opportunity["slow_target_direction_structure_fraction"]
                >= float(criteria.minimum_target_direction_structure_fraction)
            )
        ),
        "expected_oracle_has_practical_advantage_over_best_fixed": bool(
            opportunity["mean_oracle_advantage_over_best_fixed_log10"]
            >= (0.0 if smoke else float(
                criteria.minimum_oracle_advantage_over_best_fixed_log10
            ))
        ),
        "oracle_opportunity_positive_across_structures": bool(
            smoke or opportunity["positive_structure_oracle_fraction"]
            >= float(criteria.minimum_positive_structure_oracle_fraction)
        ),
        "realized_optimal_profile_reproducible_across_futures": bool(
            opportunity["mean_realized_optimal_profile_agreement_fraction"]
            >= (0.0 if smoke else float(
                criteria.minimum_realized_winner_agreement_fraction
            ))
        ),
        "active_phase_error_crossover_matches_P2A_direction": bool(
            opportunity["mean_fast_phase_advantage_high_diffusion_low_noise_rad"] > 0.0
            and opportunity["mean_slow_phase_advantage_low_diffusion_high_noise_rad"] > 0.0
        ),
        "phase_error_advantage_maps_tacs_response": bool(
            opportunity["phase_error_advantage_response_spearman_rho"]
            >= (0.0 if smoke else float(criteria.minimum_phase_response_spearman))
        ),
        "at_least_one_predecision_EEG_feature_maps_relative_response": bool(
            association_audit["at_least_one_feature_passes"]
        ) or smoke,
        "policy_inputs_and_efficacy_endpoint_are_separated": bool(
            active.context_features_use_observed_EEG.astype(bool).all()
            and active.efficacy_endpoint_uses_neural_only_EEG.astype(bool).all()
        ),
    }
    opportunity_checks = [
        "expected_oracle_uses_both_profiles",
        "both_profiles_have_practical_contextual_support",
        "fast_profile_improves_high_diffusion_low_noise",
        "slow_profile_improves_low_diffusion_high_noise",
        "response_crossover_is_practically_meaningful",
        "target_directions_positive_across_structures",
        "expected_oracle_has_practical_advantage_over_best_fixed",
        "oracle_opportunity_positive_across_structures",
        "realized_optimal_profile_reproducible_across_futures",
        "active_phase_error_crossover_matches_P2A_direction",
        "phase_error_advantage_maps_tacs_response",
        "at_least_one_predecision_EEG_feature_maps_relative_response",
    ]
    passed = bool(all(checks.values()) and not smoke)
    return checks, {
        "H5_P2B_active_response_mapping": (
            "PASSED" if passed else "NOT PASSED"
        ) if not smoke else "SMOKE TEST ONLY",
        "ready_for_H5_P2C_policy_development": passed,
        "machine_learning_policy_status": "NOT TRAINED OR TESTED",
        "failed_opportunity_checks": [
            name for name in opportunity_checks if not checks[name]
        ],
    }


def _save_figure(figure: Any, root: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(root / f"{name}.png", dpi=300)
    figure.savefig(root / f"{name}.pdf")
    plt.close(figure)


def _plots(
    *, root: Path, screening: pd.DataFrame, spectra: pd.DataFrame,
    action_map: pd.DataFrame, directions: pd.DataFrame,
    associations: pd.DataFrame, trajectories: pd.DataFrame,
) -> None:
    carrier = _carrier_by_noise(screening)
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(carrier.observation_noise_fraction, carrier.carrier_coverage,
              "o-", label="coverage")
    axis.plot(carrier.observation_noise_fraction, carrier.accepted_carrier_accuracy,
              "s-", label="accepted accuracy")
    axis.plot(carrier.observation_noise_fraction, carrier.enrollment_fraction,
              "^-", label="enrollment")
    axis.set(xlabel="Observation-noise RMS / neural-EEG RMS", ylabel="Fraction",
             ylim=(0, 1.05), title="Frozen carrier and prospective screening")
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_01_carrier_screening")

    if not spectra.empty:
        example = spectra[
            spectra.paired_noise_context_id.eq(spectra.paired_noise_context_id.iloc[0])
        ]
        figure, axis = plt.subplots(figsize=(7.5, 4.6))
        for label, group in example.groupby("observation_noise_label"):
            view = group[group.frequency_hz.between(6.0, 14.0)]
            axis.plot(view.frequency_hz, view.observed_EEG_multitaper_residual_db,
                      label=label.replace("_", " "))
        axis.axvline(float(example.hidden_frequency_hz.iloc[0]), color="black",
                     linestyle="--", linewidth=1)
        axis.set(xlabel="Frequency (Hz)", ylabel="Aperiodic-adjusted power (dB)",
                 title="Representative predecision carrier evidence")
        axis.legend(frameon=False)
        _save_figure(figure, root, "figure_02_representative_PSD")

    aggregate = action_map.groupby([
        "label", "observation_noise_fraction"
    ], as_index=False).fast_advantage_over_slow_log10.mean()
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for label, color, marker in ((LOW, "#4C78A8", "o"), (HIGH, "#E45756", "^")):
        view = aggregate[aggregate.label.eq(label)]
        axis.plot(view.observation_noise_fraction,
                  view.fast_advantage_over_slow_log10, marker=marker,
                  color=color, label=label.replace("_", " "))
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Observation-noise fraction",
             ylabel="Fast advantage over slow (log10 distance)",
             title="Active tACS controller-response crossover")
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_03_active_response_crossover")

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(directions)); width = 0.36
    axis.bar(x - width / 2,
             directions.fast_advantage_high_diffusion_low_noise_log10,
             width, label="fast: high D / low noise")
    axis.bar(x + width / 2,
             directions.slow_advantage_low_diffusion_high_noise_log10,
             width, label="slow: low D / high noise")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, directions.structure_seed.astype(str), rotation=30)
    axis.set(ylabel="Directional endpoint advantage (log10)",
             title="Structure-level target directions")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, root, "figure_04_structure_directions")

    figure, axis = plt.subplots(figsize=(6.8, 5.0))
    for label, color in ((LOW_NOISE, "#4C78A8"), (HIGH_NOISE, "#E45756")):
        view = action_map[action_map.observation_noise_label.eq(label)]
        axis.scatter(view.fast_phase_error_advantage_rad,
                     view.fast_advantage_over_slow_log10,
                     color=color, alpha=0.8, label=label.replace("_", " "))
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Fast phase-error advantage (rad)",
             ylabel="Fast endpoint advantage (log10)",
             title="Mechanism-to-efficacy transfer")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, root, "figure_05_phase_response_mapping")

    top = associations.head(min(10, len(associations))).sort_values(
        "structure_centered_spearman_rho"
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.6))
    colors = np.where(top.passes_response_association_gate, "#2ca02c", "0.55")
    axis.barh(top.feature, top.structure_centered_spearman_rho, color=colors)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Within-structure Spearman rho",
             title="Predecision EEG feature--response associations")
    _save_figure(figure, root, "figure_06_EEG_response_associations")

    target = trajectories[
        trajectories.label.eq(HIGH)
        & trajectories.observation_noise_label.eq(LOW_NOISE)
    ]
    if not target.empty:
        timecourse = target.groupby([
            "controller_mode", "analysis_window_index"
        ], as_index=False).distance_to_B_log10.mean()
        figure, axis = plt.subplots(figsize=(8.0, 4.6))
        for mode, color in ((SHAM, "0.45"), (CONSERVATIVE, "#4C78A8"),
                            (RESPONSIVE, "#E45756")):
            view = timecourse[timecourse.controller_mode.eq(mode)]
            axis.plot(view.analysis_window_index, view.distance_to_B_log10,
                      marker="o", color=color, label=mode.replace("refresh_", ""))
        axis.set(xlabel="One-second intervention window",
                 ylabel="Distance to frozen B (log10)",
                 title="High-diffusion/low-noise endpoint trajectory")
        axis.legend(frameon=False, fontsize=8)
        _save_figure(figure, root, "figure_07_target_trajectory")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / ROOT_NAME
    exists = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H5-P2B active phase-tracker response mapping")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    target = sources["target"]

    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    spectra_rows: list[pd.DataFrame] = []
    temporal_rows: list[pd.DataFrame] = []
    contexts = _run_context_specs(cfg)
    representative_structure = min(int(row["structure_seed"]) for row in contexts)
    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"D={context['diffusion_rad2_per_s']:g} rad^2/s "
                f"noise={context['observation_noise_fraction']:g}"
            )
        state_cfg = _with_context_state(cfg, context)
        state_cfg = _with_noise_fraction(
            state_cfg, float(context["observation_noise_fraction"])
        )
        first_future = _future_seed(state_cfg, context, 0)
        baseline_reference = _run_p2b_controller(
            condition_cfg=state_cfg, context=context, future_seed=first_future,
            future_index=0, mode=SHAM, action_index=0, root=root,
            comm=comm, size=size, rank=rank,
        )
        if rank == 0:
            screening, spectrum, temporal, diagnostics = _screen_context(
                baseline_reference, context, target, state_cfg
            )
            screening_rows.append(screening)
            diagnostic_rows.extend(diagnostics.to_dict("records"))
            if int(context["structure_seed"]) == representative_structure:
                spectra_rows.append(spectrum)
                temporal_rows.append(temporal)
            eligible = bool(screening["eligible"])
            selected_frequency = float(screening["EEG_selected_frequency_hz"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'SHAM FALLBACK'}; "
                f"selected={selected_frequency:g} Hz; "
                f"carrier={'accepted' if screening['carrier_identified'] else 'abstained'}; "
                f"reason={screening['exclusion_reasons']}"
            )
        else:
            screening, eligible, selected_frequency = None, None, None
        eligible = bool(comm.bcast(eligible, root=0))
        selected_frequency = float(comm.bcast(selected_frequency, root=0))
        if not eligible:
            del baseline_reference
            continue

        action_cfg = _with_action_frequency(state_cfg, selected_frequency)
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations)):
            future_seed = _future_seed(action_cfg, context, future_index)
            episodes: dict[str, dict[str, Any]] | None = {} if rank == 0 else None
            for action_index, mode in enumerate(_controller_modes(action_cfg)):
                if future_index == 0 and mode == SHAM:
                    episode = baseline_reference
                else:
                    episode = _run_p2b_controller(
                        condition_cfg=action_cfg, context=context,
                        future_seed=future_seed, future_index=future_index,
                        mode=mode, action_index=action_index, root=root,
                        comm=comm, size=size, rank=rank,
                    )
                if rank == 0:
                    episodes[mode] = episode
            if rank == 0:
                rows, trajectories, updates = _metric_rows(
                    context=context, screening=screening,
                    future_index=future_index, future_seed=future_seed,
                    episodes=episodes, baseline_reference=baseline_reference,
                    target=target, cfg=action_cfg,
                )
                _augment_metric_rows(rows, episodes, action_cfg)
                _augment_common_audit(rows, episodes, action_cfg)
                _augment_observation_rows(rows, episodes)
                _add_context_fields(rows, screening)
                shared = {
                    "paired_noise_context_id": str(context["paired_noise_context_id"]),
                    "observation_noise_label": str(context["observation_noise_label"]),
                    "observation_noise_fraction": float(
                        context["observation_noise_fraction"]
                    ),
                }
                for row in trajectories:
                    row.update(shared)
                for row in updates:
                    row.update(shared)
                metric_rows.extend(rows)
                trajectory_rows.extend(trajectories)
                update_rows.extend(updates)
        del baseline_reference

    if rank != 0:
        return
    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "prospective_screening.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        root / "predecision_phase_tracker_diagnostics.csv", index=False
    )
    spectra = pd.concat(spectra_rows, ignore_index=True) if spectra_rows else pd.DataFrame()
    temporal = pd.concat(temporal_rows, ignore_index=True) if temporal_rows else pd.DataFrame()
    if bool(cfg.analysis.save_representative_spectra):
        spectra.to_csv(root / "representative_predecision_spectra.csv", index=False)
        temporal.to_csv(root / "representative_temporal_evidence.csv", index=False)
    if not metric_rows:
        conclusion = {
            "scope": "H5-P2B active response mapping",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "H5_P2B_active_response_mapping": "NOT PASSED",
                "ready_for_H5_P2C_policy_development": False,
                "machine_learning_policy_status": "NOT TRAINED OR TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "stopped_after_safe_sham_fallback_screening": True,
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(
            conclusion, indent=2, allow_nan=False
        ))
        print("No eligible contexts; H5-P2B stopped after screening.")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    updates = pd.DataFrame(update_rows)
    expected = _expected_map(metrics)
    action_map, structure, directions, opportunity = _response_map(
        expected, metrics, cfg
    )
    associations, association_audit = _feature_response_associations(
        action_map, cfg
    )
    checks, conclusions = _checks(
        screening=screening, metrics=metrics, expected=expected,
        updates=updates, action_map=action_map, structure=structure,
        directions=directions, opportunity=opportunity,
        associations=associations, association_audit=association_audit,
        sources=sources, cfg=cfg,
    )

    metrics.to_csv(root / "context_controller_future_metrics.csv", index=False)
    trajectories.to_csv(root / "one_second_EEG_trajectories.csv", index=False)
    updates.to_csv(root / "causal_phase_updates.csv", index=False)
    expected.to_csv(root / "expected_context_controller_map.csv", index=False)
    action_map.to_csv(root / "controller_profile_response_map.csv", index=False)
    structure.to_csv(root / "structure_level_oracle_opportunity.csv", index=False)
    directions.to_csv(root / "structure_target_directions.csv", index=False)
    associations.to_csv(root / "EEG_feature_response_associations.csv", index=False)
    _carrier_by_noise(screening).to_csv(root / "carrier_by_noise.csv", index=False)
    audit = {
        "carrier_measurement": _carrier_by_noise(screening).to_dict("records"),
        "controller_profile_opportunity": opportunity,
        "EEG_feature_response_mapping": association_audit,
    }
    (root / "H5_P2B_response_mapping_audit.json").write_text(json.dumps(
        _json_ready(audit), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H5_P2B_active_phase_tracker_response_mapping",
        "frozen_sources": {"roots": sources["roots"], "hashes": sources["hashes"]},
        "frozen_P2A_conditions": sources["p2a_frozen"],
        "frozen_population_B_target": sources["target"],
        "state_generator": {
            "carrier_hz": [9.0, 11.0],
            "phase_diffusion_rad2_per_s": [0.5, 2.0],
            "shared_modulated_afferent_fraction": 1.0,
            "modulation_depth": 0.04,
            "mean_afferent_rate_matched": True,
        },
        "measurement_conditions": {
            "AR1_coefficient": 0.95,
            "noise_RMS_fractions": [0.25, 0.50],
            "noise_path_paired_across_severity": True,
            "configured_noise_level_excluded_from_policy_features": True,
        },
        "causal_protocol": {
            "predecision_observed_EEG_s": 30,
            "intervention_s": 8,
            "washout_s": 1,
            "active_amplitude_v_per_m": 0.2,
            "carrier_estimator": str(cfg.analysis.response_mapping.frozen_estimator),
            "relative_phase_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
            "montage": str(cfg.analysis.tacs.axial_montage),
            "controller_profiles": {
                mode: _profile(cfg, mode) for mode in EXPECTED_MODES
            },
        },
        "design": {
            "independent_structures": int(screening.structure_seed.nunique()),
            "screened_contexts": int(len(screening)),
            "eligible_contexts": int(screening.eligible.sum()),
            "paired_futures": int(cfg.analysis.crossed_design.n_future_continuations),
            "crossed_repeats": "9/11 Hz x low/high D x low/high sensor noise",
            "statistical_unit": "independent circuit structure",
        },
        "inference_boundary": (
            "Full-information active system identification only. The oracle and "
            "feature associations are not a trained or deployable ML policy."
        ),
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-P2B active phase-tracker response mapping",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure",
        "inference_boundary": provenance["inference_boundary"],
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot):
        _plots(
            root=root, screening=screening, spectra=spectra,
            action_map=action_map, directions=directions,
            associations=associations, trajectories=trajectories,
        )

    print("\n### H5-P2B screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### H5-P2B active response-mapping checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### H5-P2B response opportunity")
    print(json.dumps(_json_ready(audit), indent=2, allow_nan=False))
    print(
        "\nH5-P2B active response mapping: "
        f"{conclusions['H5_P2B_active_response_mapping']}"
    )
    print("Machine-learning policy status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
