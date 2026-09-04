"""H5-P1 full-information EEG-context/controller response mapping.

H5-I0b established that a frozen DPSS multitaper estimator can select the
9/11-Hz carrier from 30 seconds of moderately noisy predecision EEG. H5-P1
places that estimator causally in front of the two safe phase-maintenance
profiles mapped in H5-P0. It asks whether their *expected* relative response is
large, reproducible across paired stochastic futures, and associated with
phase-invariant EEG context strongly enough to justify later policy fitting.

This is discovery-stage system identification. The expected-outcome oracle is
post hoc and nondeployable, and the response-feature association is an
exploratory mapping audit. No machine-learning stimulation policy is trained
or tested, and a failed opportunity gate is a valid stopping result.
"""

from __future__ import annotations

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
from scipy import stats


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (  # noqa: E402
    CONSERVATIVE,
    EXPECTED_MODES,
    FULL,
    PARTIAL,
    RESPONSIVE,
    _add_context_features_to_rows as _add_base_context_features,
    _augment_observation_rows,
    _context_specs as _base_context_specs,
    _future_seed,
    _noise_seeds,
    _observed_episode_view,
    _run_controller,
    _shared_drive_loso,
    _with_context_state,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_multitaper_measurement_validation import (  # noqa: E402
    MT_POOLED,
    OBSERVED,
    _estimate_multitaper_methods,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (  # noqa: E402
    HIGH,
    LOW,
    _context_features,
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
    _sha256,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_cadence_discovery import (  # noqa: E402
    _augment_common_audit,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _process_eeg,
)


ROOT_NAME = "h5_response_mapping"
P1_CONTEXT_FEATURES = [
    "context_C1",
    "context_C1_abs",
    "context_C1_temporal_sd",
    "context_spectral_concentration",
    "context_spectral_rms_width_hz",
    "context_alpha_excess_log10",
    "recent_resultant_to_rms",
    "carrier_maximum_residual_evidence_db",
    "carrier_evidence_margin_db",
    "carrier_soft_support_fraction",
    "carrier_window_score_sd_db",
]


def _source_files(root: Path, names: dict[str, str]) -> dict[str, Path]:
    files = {name: root / filename for name, filename in names.items()}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen source files: {missing}")
    return files


def _hash_locked_files(
    root: Path, names: dict[str, str], expected_cfg: DictConfig,
) -> tuple[dict[str, Path], dict[str, str]]:
    files = _source_files(root, names)
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {name: str(expected_cfg[name]) for name in names}
    if observed != expected:
        raise RuntimeError(
            f"Frozen source hash mismatch at {root}: "
            f"expected={expected}, observed={observed}"
        )
    return files, observed


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    """Hash-lock negative H5-P0 and positive H5-I0b without older raw files."""
    p0_root = Path(to_absolute_path(str(cfg.analysis.source_h5p0.result_dir)))
    p0_names = {
        "conclusion": "experiment_conclusion.json",
        "audit": "H5_P0_feasibility_audit.json",
        "screening": "prospective_screening.csv",
        "metrics": "context_controller_future_metrics.csv",
        "provenance": "protocol_and_provenance.json",
    }
    p0_files, p0_hashes = _hash_locked_files(
        p0_root, p0_names, cfg.analysis.source_h5p0.expected_sha256
    )
    p0_conclusion = json.loads(p0_files["conclusion"].read_text())
    p0_status = p0_conclusion["conclusions"]
    expected_failures = {
        "expected_oracle_has_practical_advantage_over_best_fixed",
        "realized_optimal_profile_reproducible_across_futures",
    }
    if (
        p0_status["H5_P0_contextual_controller_profile_opportunity"]
        != "NOT PASSED"
        or bool(p0_status["ready_for_disjoint_H5_policy_development"])
        or set(p0_status.get("failed_feasibility_checks", []))
        != expected_failures
    ):
        raise RuntimeError("H5-P1 requires the exact frozen negative H5-P0 result.")

    i0b_root = Path(to_absolute_path(str(cfg.analysis.source_h5i0b.result_dir)))
    i0b_names = {
        "conclusion": "experiment_conclusion.json",
        "frozen_estimator": "frozen_multitaper_estimator.json",
        "selection": "discovery_estimator_selection.csv",
        "discovery_metrics": "discovery_estimator_context_metrics.csv",
        "discovery_summary": "discovery_estimator_summary.csv",
        "confirmation_metrics": "confirmation_estimator_context_metrics.csv",
        "inference": "confirmation_inference.json",
        "provenance": "protocol_and_provenance.json",
    }
    i0b_files, i0b_hashes = _hash_locked_files(
        i0b_root, i0b_names, cfg.analysis.source_h5i0b.expected_sha256
    )
    i0b_conclusion = json.loads(i0b_files["conclusion"].read_text())
    i0b_frozen = json.loads(i0b_files["frozen_estimator"].read_text())
    if (
        not all(bool(value) for value in i0b_conclusion["checks"].values())
        or i0b_conclusion["conclusions"][
            "H5_I0b_multitaper_carrier_measurement"
        ] != "CONFIRMED"
        or not bool(
            i0b_conclusion["conclusions"]["ready_for_H5_P1_response_mapping"]
        )
        or str(i0b_frozen["selected_estimator"]) != MT_POOLED
        or not bool(i0b_frozen["discovery_gate_passed"])
    ):
        raise RuntimeError("H5-P1 requires the exact positive H5-I0b result.")

    source_seed_union: set[int] = set()
    for path in (
        p0_files["screening"], p0_files["metrics"],
        i0b_files["discovery_metrics"],
        i0b_files["confirmation_metrics"],
    ):
        table = pd.read_csv(path)
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed", "noise_seed",
        ):
            if column in table:
                source_seed_union.update(table[column].dropna().astype(int))

    target = OmegaConf.to_container(cfg.analysis.frozen_target, resolve=True)
    target_source_hash = str(target.pop("source_sha256"))
    if target_source_hash != (
        "67412b211172d894d7eb31673c7e32d32be4c36e3a4f23ad86f4f4f0c2543f41"
    ):
        raise RuntimeError("The embedded H4 population target provenance changed.")
    return {
        "roots": {"h5p0": str(p0_root), "h5i0b": str(i0b_root)},
        "hashes": {"h5p0": p0_hashes, "h5i0b": i0b_hashes},
        "source_seed_union": source_seed_union,
        "H5P0_negative_preserved": True,
        "H5I0b_confirmed": True,
        "frozen_estimator": i0b_frozen,
        "target": target,
        "target_source_hash": target_source_hash,
    }


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    return _base_context_specs(cfg)


def _run_context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    rows = _context_specs(cfg)
    if not bool(cfg.analysis.smoke_test):
        return rows
    limit = int(cfg.analysis.smoke_context_limit)
    if limit <= 0:
        return rows
    first_structure = min(int(row["structure_index"]) for row in rows)
    pool = [row for row in rows if int(row["structure_index"]) == first_structure]
    # In the short smoke, prioritize both carriers before expanding the D/q grid.
    representative = [0, 4, 3, 7]
    return [pool[index] for index in representative[:limit]]


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    smoke = bool(cfg.analysis.smoke_test)
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-P1 requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-P1 may not alter recurrent inhibition.")
    if [float(value) for value in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-P1 retains the frozen 9/11-Hz carrier grid.")
    diffusion = [
        (str(value.label), float(value.diffusion_rad2_per_s))
        for value in cfg.analysis.states.phase_diffusion_levels
    ]
    if diffusion != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("H5-P1 retains D={0.5,2.0} rad^2/s.")
    shared = [
        (str(value.label), float(value.shared_modulated_fraction))
        for value in cfg.analysis.states.shared_drive_levels
    ]
    if shared != [(PARTIAL, 0.5), (FULL, 1.0)]:
        raise ValueError("H5-P1 retains q={0.5,1.0} shared afferent drive.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-P1 retains afferent modulation depth 0.04.")
    if _controller_modes(cfg) != EXPECTED_MODES:
        raise ValueError(f"H5-P1 controller order must be {EXPECTED_MODES}.")
    expected_profiles = {
        CONSERVATIVE: {"adaptive": True, "history_ms": 1000.0,
                       "update_interval_ms": 250.0},
        RESPONSIVE: {"adaptive": True, "history_ms": 500.0,
                     "update_interval_ms": 125.0},
    }
    for mode, expected in expected_profiles.items():
        if _profile(cfg, mode) != expected:
            raise ValueError(f"H5-P1 controller profile changed: {mode}.")
    if str(cfg.analysis.response_mapping.frozen_estimator) != MT_POOLED:
        raise ValueError("H5-P1 must use the H5-I0b frozen estimator.")
    if str(cfg.analysis.response_mapping.safe_abstention_action) != SHAM:
        raise ValueError("H5-P1 abstention must fall back to sham.")
    if str(cfg.analysis.response_mapping.h4_reference_profile) != RESPONSIVE:
        raise ValueError("The H4 reference profile must remain frozen.")
    frozen_multitaper = sources["frozen_estimator"]["multitaper_configuration"]
    configured_multitaper = OmegaConf.to_container(
        cfg.analysis.multitaper, resolve=True
    )
    if configured_multitaper != frozen_multitaper:
        raise ValueError("H5-P1 multitaper configuration differs from H5-I0b.")
    if not np.isclose(float(cfg.analysis.actions.amplitude_v_per_m), 0.2):
        raise ValueError("Both active H5-P1 profiles must use 0.2 V/m.")
    if float(cfg.env.simulation.obs_win_len) != 1000.0:
        raise ValueError("H5-P1 requires one-second outer online windows.")
    if not bool(cfg.analysis.observation_noise.enabled) or not np.isclose(
        float(cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg),
        0.25,
    ) or not np.isclose(float(cfg.analysis.observation_noise.ar1_coefficient), 0.95):
        raise ValueError("H5-P1 retains the frozen AR(1) observation model.")
    if not smoke:
        if int(cfg.analysis.timeline.baseline_steps) != 30:
            raise ValueError("Full H5-P1 requires the confirmed 30-s estimator input.")
        if int(cfg.analysis.timeline.stimulation_steps) != 8:
            raise ValueError("Full H5-P1 requires the H4-matched 8-s endpoint.")
        if int(cfg.analysis.crossed_design.n_structure_seeds) != 6:
            raise ValueError("Full H5-P1 requires six independent structures.")
        if int(cfg.analysis.crossed_design.n_history_seeds) != 1:
            raise ValueError("Full H5-P1 requires one history per structure/grid cell.")
        if int(cfg.analysis.crossed_design.n_future_continuations) != 4:
            raise ValueError("Full H5-P1 requires four paired futures.")
    if not np.isclose(float(sources["target"]["outcome"]["mean_log10_alpha"]),
                      -20.834968281097733):
        raise ValueError("The H4 population-B outcome target changed.")
    if not np.isclose(float(sources["target"]["outcome_duration_s"]), 8.0):
        raise ValueError("The frozen population-B target duration changed.")

    contexts = _context_specs(cfg)
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
        raise ValueError("Every H5-P1 seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H5-P1 seed namespaces overlap.")
    if set().union(*namespaces).intersection(sources["source_seed_union"]):
        raise ValueError("H5-P1 seeds overlap H5-P0 or H5-I0b.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H5-P1 structure seed exceeds the uint32 mapping range.")


def _frozen_carrier_screen(
    episode: dict[str, Any], context: dict[str, Any], target: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Apply the frozen carrier estimator and then compute matched EEG context."""
    observed_raw = np.asarray(
        episode["observed_raw_by_epoch"]["baseline"], dtype=float
    ).reshape(-1)
    processed, fs_hz, _, _, _ = _process_eeg(
        observed_raw,
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        cfg=cfg,
    )
    estimator_rows, spectrum, temporal = _estimate_multitaper_methods(
        processed,
        fs_hz=float(fs_hz),
        hidden_frequency_hz=float(context["hidden_frequency_hz"]),
        input_signal=OBSERVED,
        cfg=cfg,
    )
    selected_name = str(cfg.analysis.response_mapping.frozen_estimator)
    result = next(row for row in estimator_rows if row["estimator"] == selected_name)
    selected_frequency = float(result["selected_frequency_hz"])

    forced_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(forced_cfg):
        forced_cfg.analysis.states.frequencies_hz = [selected_frequency]
    screening = _context_features(
        _observed_episode_view(episode), context, target, forced_cfg
    )
    base_eligible = bool(screening["eligible"])
    carrier_identified = bool(result["identified"])
    reasons = [] if screening["exclusion_reasons"] == "none" else str(
        screening["exclusion_reasons"]
    ).split(";")
    if not carrier_identified:
        reasons.append("frozen_carrier_estimator_abstained")
    screening.update({
        "EEG_selected_frequency_hz": selected_frequency,
        "EEG_frequency_selection_correct": bool(
            result["frequency_detected_correctly"]
        ),
        "carrier_estimator": selected_name,
        "carrier_identified": carrier_identified,
        "carrier_peak_frequency_hz": float(result["peak_frequency_hz"]),
        "carrier_evidence_9_db": float(result["evidence_9_db"]),
        "carrier_evidence_11_db": float(result["evidence_11_db"]),
        "carrier_evidence_delta_11_minus_9_db": float(
            result["evidence_delta_11_minus_9_db"]
        ),
        "carrier_maximum_residual_evidence_db": float(
            result["maximum_residual_evidence_db"]
        ),
        "carrier_evidence_margin_db": float(result["evidence_margin_db"]),
        "carrier_soft_support_fraction": float(result["soft_support_fraction"]),
        "carrier_window_score_sd_db": float(result["window_score_sd_db"]),
        "carrier_decision_uses_hidden_frequency": False,
        "hidden_frequency_used_only_for_scoring": True,
        "base_EEG_and_safety_screen_passed": base_eligible,
        "eligible": bool(base_eligible and carrier_identified),
        "safe_fallback_action": (
            "not_required" if base_eligible and carrier_identified else SHAM
        ),
        "exclusion_reasons": ";".join(reasons) if reasons else "none",
        "screen_uses_only_predecision_ideal_EEG": False,
        "screen_uses_only_predecision_observed_EEG": True,
        "screen_uses_hidden_diffusion_or_frequency": False,
        "screen_uses_action_outcome": False,
        "observation_noise_model": "AR1_additive_sensor_noise",
        "configured_observation_noise_fraction": float(
            cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg
        ),
    })
    spectrum = spectrum.assign(**{
        key: value for key, value in context.items()
        if key in (
            "context_id", "structure_seed", "hidden_frequency_hz", "label",
            "diffusion_rad2_per_s", "shared_drive_label",
            "shared_modulated_fraction",
        )
    })
    temporal = temporal.assign(**{
        key: value for key, value in context.items()
        if key in (
            "context_id", "structure_seed", "hidden_frequency_hz", "label",
            "diffusion_rad2_per_s", "shared_drive_label",
            "shared_modulated_fraction",
        )
    })
    return screening, spectrum, temporal


def _run_p1_controller(**kwargs: Any) -> dict[str, Any] | None:
    episode = _run_controller(**kwargs)
    if episode is not None:
        episode["simulation"]["action"]["role"] = (
            "H5_P1_full_information_controller_profile"
        )
    return episode


def _add_p1_context_features(
    rows: list[dict[str, Any]], screening: dict[str, Any]
) -> None:
    _add_base_context_features(rows, screening)
    for row in rows:
        for feature in P1_CONTEXT_FEATURES:
            row[feature] = float(screening[feature])
        row.update({
            "carrier_estimator": str(screening["carrier_estimator"]),
            "carrier_identified": bool(screening["carrier_identified"]),
            "carrier_selection_correct_evaluation_only": bool(
                screening["EEG_frequency_selection_correct"]
            ),
            "action_frequency_hz": float(screening["EEG_selected_frequency_hz"]),
            "action_frequency_uses_frozen_EEG_estimator": True,
        })


def _expected_map(metrics: pd.DataFrame) -> pd.DataFrame:
    group = [
        "context_id", "paired_shared_drive_context_id", "structure_seed",
        "hidden_frequency_hz", "label", "diffusion_rad2_per_s",
        "shared_drive_label", "shared_modulated_fraction",
        "EEG_selected_frequency_hz", *P1_CONTEXT_FEATURES, "controller_mode",
    ]
    return (
        metrics.groupby(group, as_index=False)
        .agg(
            n_futures=("future_index", "nunique"),
            expected_post_distance_to_B_log10=(
                "post_distance_to_B_log10", "mean"
            ),
            future_sd_post_distance_log10=("post_distance_to_B_log10", "std"),
            expected_improvement_vs_sham_log10=(
                "causal_distance_improvement_vs_sham_log10", "mean"
            ),
            expected_alpha_suppression_vs_sham_log10=(
                "causal_alpha_suppression_vs_sham_log10", "mean"
            ),
            all_rate_safe=("rate_safe", "all"),
            all_field_removal_recovered=("field_removal_recovered", "all"),
            mean_abs_controller_phase_error_rad=(
                "mean_abs_phase_error_before_correction_rad", "mean"
            ),
            mean_abs_common_phase_error_rad=(
                "mean_abs_common_phase_error_rad", "mean"
            ),
            common_phase_estimate_actionable_fraction=(
                "common_phase_estimate_actionable_fraction", "mean"
            ),
            mean_phase_resultant_to_rms=(
                "mean_phase_resultant_to_rms", "mean"
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = [
        "context_id", "paired_shared_drive_context_id", "structure_seed",
        "hidden_frequency_hz", "label", "diffusion_rad2_per_s",
        "shared_drive_label", "shared_modulated_fraction",
        "EEG_selected_frequency_hz", *P1_CONTEXT_FEATURES,
    ]
    active = expected[expected.controller_mode.isin([CONSERVATIVE, RESPONSIVE])]
    pivot = active.pivot(
        index=keys,
        columns="controller_mode",
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
    conservative_distance = f"expected_post_distance_to_B_log10_{CONSERVATIVE}"
    responsive_distance = f"expected_post_distance_to_B_log10_{RESPONSIVE}"
    pivot["responsive_advantage_over_conservative_log10"] = (
        pivot[conservative_distance] - pivot[responsive_distance]
    )
    pivot["expected_optimal_profile"] = np.where(
        pivot.responsive_advantage_over_conservative_log10 > 0.0,
        RESPONSIVE,
        CONSERVATIVE,
    )
    margin = float(cfg.analysis.response_mapping.practical_context_margin_log10)
    pivot["expected_winner_margin_log10"] = (
        pivot.responsive_advantage_over_conservative_log10.abs()
    )
    pivot["practical_optimal_profile"] = np.select(
        [
            pivot.responsive_advantage_over_conservative_log10 >= margin,
            pivot.responsive_advantage_over_conservative_log10 <= -margin,
        ],
        [RESPONSIVE, CONSERVATIVE],
        default="no_practical_difference",
    )

    fixed_mean = {
        mode: float(active[active.controller_mode.eq(mode)]
                    .expected_post_distance_to_B_log10.mean())
        for mode in (CONSERVATIVE, RESPONSIVE)
    }
    best_fixed = min(fixed_mean, key=lambda mode: (fixed_mean[mode], mode))
    best_fixed_column = f"expected_post_distance_to_B_log10_{best_fixed}"
    h4_profile = str(cfg.analysis.response_mapping.h4_reference_profile)
    h4_column = f"expected_post_distance_to_B_log10_{h4_profile}"
    pivot["best_fixed_profile"] = best_fixed
    pivot["h4_reference_profile"] = h4_profile
    pivot["expected_oracle_distance_to_B_log10"] = np.minimum(
        pivot[conservative_distance], pivot[responsive_distance]
    )
    pivot["oracle_advantage_over_best_fixed_log10"] = (
        pivot[best_fixed_column] - pivot.expected_oracle_distance_to_B_log10
    )
    pivot["oracle_advantage_over_h4_profile_log10"] = (
        pivot[h4_column] - pivot.expected_oracle_distance_to_B_log10
    )

    realized = metrics[metrics.controller_mode.isin([CONSERVATIVE, RESPONSIVE])]
    agreement: list[float] = []
    response_sd: list[float] = []
    for row in pivot.itertuples():
        group = realized[realized.context_id.eq(str(row.context_id))]
        realized_winners: list[str] = []
        paired_responses: list[float] = []
        for _, future in group.groupby("future_index"):
            by_mode = future.set_index("controller_mode")
            paired_responses.append(float(
                by_mode.loc[CONSERVATIVE, "post_distance_to_B_log10"]
                - by_mode.loc[RESPONSIVE, "post_distance_to_B_log10"]
            ))
            realized_winners.append(str(future.sort_values([
                "post_distance_to_B_log10", "controller_mode"
            ]).iloc[0].controller_mode))
        agreement.append(float(np.mean(
            np.asarray(realized_winners) == str(row.expected_optimal_profile)
        )))
        response_sd.append(float(np.std(paired_responses, ddof=1)))
    pivot["realized_optimal_profile_agreement_fraction"] = agreement
    pivot["paired_future_response_sd_log10"] = response_sd

    structure = (
        pivot.groupby("structure_seed", as_index=False)
        .agg(
            context_count=("context_id", "nunique"),
            mean_oracle_advantage_over_best_fixed_log10=(
                "oracle_advantage_over_best_fixed_log10", "mean"
            ),
            mean_oracle_advantage_over_h4_profile_log10=(
                "oracle_advantage_over_h4_profile_log10", "mean"
            ),
            mean_responsive_advantage_over_conservative_log10=(
                "responsive_advantage_over_conservative_log10", "mean"
            ),
            mean_realized_optimal_profile_agreement_fraction=(
                "realized_optimal_profile_agreement_fraction", "mean"
            ),
            mean_paired_future_response_sd_log10=(
                "paired_future_response_sd_log10", "mean"
            ),
        )
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
    shared_response = (
        pivot.groupby("shared_drive_label")
        .responsive_advantage_over_conservative_log10.mean().to_dict()
    )
    diffusion_response = (
        pivot.groupby("label")
        .responsive_advantage_over_conservative_log10.mean().to_dict()
    )
    frequency_response = (
        pivot.groupby("hidden_frequency_hz")
        .responsive_advantage_over_conservative_log10.mean().to_dict()
    )
    opportunity = {
        "best_fixed_profile": best_fixed,
        "h4_reference_profile": h4_profile,
        "fixed_profile_expected_distance_log10": fixed_mean,
        "oracle_expected_distance_log10": float(
            pivot.expected_oracle_distance_to_B_log10.mean()
        ),
        "mean_oracle_advantage_over_best_fixed_log10": float(
            structure.mean_oracle_advantage_over_best_fixed_log10.mean()
        ),
        "mean_oracle_advantage_over_h4_profile_log10": float(
            structure.mean_oracle_advantage_over_h4_profile_log10.mean()
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
        "mean_realized_practical_profile_agreement_fraction": (
            float(practical.realized_optimal_profile_agreement_fraction.mean())
            if len(practical) else float("nan")
        ),
        "mean_paired_future_response_sd_log10": float(
            pivot.paired_future_response_sd_log10.mean()
        ),
        "mean_responsive_advantage_by_shared_drive_log10": shared_response,
        "mean_responsive_advantage_by_diffusion_log10": diffusion_response,
        "mean_responsive_advantage_by_frequency_log10": frequency_response,
        "absolute_shared_drive_by_profile_response_interaction_log10": abs(
            float(shared_response.get(FULL, np.nan))
            - float(shared_response.get(PARTIAL, np.nan))
        ),
        "oracle_is_post_hoc_full_information_and_not_deployable": True,
    }
    return pivot, structure, opportunity


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * values.size / np.arange(1, values.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def _feature_response_associations(
    action_map: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Structure-preserving exploratory EEG-feature/response associations."""
    table = action_map.copy()
    y = table.responsive_advantage_over_conservative_log10.to_numpy(float)
    groups = [
        np.flatnonzero(table.structure_seed.to_numpy() == seed)
        for seed in sorted(table.structure_seed.unique())
    ]
    centered_y = y.copy()
    for indices in groups:
        centered_y[indices] -= np.mean(centered_y[indices])
    rng = np.random.default_rng(
        int(cfg.experiment.seed)
        + int(cfg.analysis.response_mapping.association_seed_offset)
    )
    permutations = int(cfg.analysis.response_mapping.association_permutations)
    permuted_y = np.empty((permutations, len(table)), dtype=float)
    for draw in range(permutations):
        permuted_y[draw] = centered_y
        for indices in groups:
            permuted_y[draw, indices] = rng.permutation(centered_y[indices])

    rows: list[dict[str, Any]] = []
    for feature in P1_CONTEXT_FEATURES:
        x = table[feature].to_numpy(float)
        centered_x = x.copy()
        for indices in groups:
            centered_x[indices] -= np.mean(centered_x[indices])
        rho = float(stats.spearmanr(centered_x, centered_y).statistic)
        if np.isfinite(rho):
            null = np.asarray([
                stats.spearmanr(centered_x, permuted_y[draw]).statistic
                for draw in range(permutations)
            ], dtype=float)
            finite_null = null[np.isfinite(null)]
            p_value = float(
                (1 + np.count_nonzero(np.abs(finite_null) >= abs(rho)))
                / (finite_null.size + 1)
            )
        else:
            p_value = 1.0
        structure_rho: list[float] = []
        for indices in groups:
            if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
                continue
            value = float(stats.spearmanr(x[indices], y[indices]).statistic)
            if np.isfinite(value):
                structure_rho.append(value)
        sign = np.sign(rho)
        consistent = float(np.mean(
            np.sign(np.asarray(structure_rho)) == sign
        )) if structure_rho and sign != 0.0 else 0.0
        rows.append({
            "feature": feature,
            "structure_centered_spearman_rho": rho,
            "structure_preserving_permutation_p_value": p_value,
            "permutation_samples": permutations,
            "finite_structure_correlations": len(structure_rho),
            "structure_sign_consistency_fraction": consistent,
            "feature_uses_only_predecision_observed_EEG": True,
        })
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = _bh_fdr(
        result.structure_preserving_permutation_p_value.to_numpy(float)
    )
    criteria = cfg.analysis.criteria
    result["passes_exploratory_response_association_gate"] = (
        result.structure_centered_spearman_rho.abs()
        >= float(criteria.minimum_response_feature_abs_spearman)
    ) & (
        result.fdr_q_value <= float(criteria.maximum_response_feature_fdr_q)
    ) & (
        result.structure_sign_consistency_fraction
        >= float(criteria.minimum_response_feature_structure_sign_fraction)
    )
    passing = result[result.passes_exploratory_response_association_gate].copy()
    if passing.empty:
        selected = None
    else:
        passing["negative_absolute_rho"] = -passing.structure_centered_spearman_rho.abs()
        selected = str(passing.sort_values([
            "fdr_q_value", "negative_absolute_rho", "feature"
        ]).iloc[0].feature)
    audit = {
        "selected_candidate_response_feature": selected,
        "candidate_selected_only_in_exploratory_response_map": True,
        "candidate_requires_policy_development_and_disjoint_confirmation": True,
        "features_are_phase_invariant_predecision_observed_EEG": True,
        "hidden_state_and_spikes_excluded": True,
        "multiplicity_control": "Benjamini-Hochberg FDR over predeclared features",
        "permutation_unit": "response labels permuted within structure",
    }
    return result, audit


def _checks(
    *, screening: pd.DataFrame, metrics: pd.DataFrame, expected: pd.DataFrame,
    updates: pd.DataFrame, action_map: pd.DataFrame, structure: pd.DataFrame,
    observability: dict[str, Any], opportunity: dict[str, Any],
    associations: pd.DataFrame, association_audit: dict[str, Any],
    sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    smoke = bool(cfg.analysis.smoke_test)
    eligible = screening[screening.eligible.astype(bool)]
    accepted = screening[screening.carrier_identified.astype(bool)]
    active = metrics[metrics.controller_mode.ne(SHAM)]
    adaptive_updates = updates[
        updates.controller_mode.isin([CONSERVATIVE, RESPONSIVE])
        & updates.phase_refresh_applied.astype(bool)
    ]
    profile_counts = opportunity["optimal_profile_context_count"]
    profile_structures = opportunity["optimal_profile_structure_count"]
    practical_counts = opportunity["practical_optimal_profile_context_count"]
    practical_structures = opportunity["practical_optimal_profile_structure_count"]
    state_minimum = 1 if smoke else int(criteria.minimum_contexts_per_state_axis_level)
    complete_states = bool(
        eligible.hidden_frequency_hz.value_counts().min() >= state_minimum
        and eligible.label.value_counts().min() >= state_minimum
        and eligible.shared_drive_label.value_counts().min() >= state_minimum
    ) if len(eligible) else False
    checks = {
        "source_H5P0_hash_locked_with_opportunity_gate_failed": bool(
            sources["H5P0_negative_preserved"]
        ),
        "source_H5I0b_hash_locked_and_confirmed": bool(
            sources["H5I0b_confirmed"]
        ),
        "frozen_population_B_target_loaded_without_recalibration": True,
        "H5P1_seed_namespaces_disjoint_from_H5P0_and_H5I0b": True,
        "complete_frequency_diffusion_shared_drive_screening_grid": bool(
            len(screening) == len(_run_context_specs(cfg))
        ),
        "thirty_second_stimulation_free_predecision_EEG": bool(
            smoke or int(cfg.analysis.timeline.baseline_steps) == 30
        ),
        "screening_uses_only_predecision_observed_EEG": bool(
            len(screening)
            and screening.screen_uses_only_predecision_observed_EEG.all()
            and (~screening.screen_uses_hidden_diffusion_or_frequency.astype(bool)).all()
            and (~screening.screen_uses_action_outcome.astype(bool)).all()
            and (~screening.carrier_decision_uses_hidden_frequency.astype(bool)).all()
        ),
        "frozen_carrier_estimator_used_without_refitting": bool(
            len(screening) and screening.carrier_estimator.eq(MT_POOLED).all()
        ),
        "carrier_abstention_maps_to_safe_sham_fallback": bool(
            screening.loc[
                ~screening.eligible.astype(bool), "safe_fallback_action"
            ].eq(SHAM).all()
        ),
        "minimum_screened_contexts": bool(
            smoke or len(screening) >= int(criteria.minimum_screened_contexts)
        ),
        "minimum_eligible_contexts": bool(
            smoke or len(eligible) >= int(criteria.minimum_eligible_contexts)
        ),
        "minimum_independent_structures": bool(
            smoke or eligible.structure_seed.nunique()
            >= int(criteria.minimum_structure_seeds)
        ),
        "eligible_frequency_diffusion_shared_drive_coverage": complete_states,
        "carrier_identification_coverage_replicated": bool(
            screening.carrier_identified.mean()
            >= float(criteria.minimum_carrier_identification_coverage)
        ),
        "accepted_carrier_accuracy_replicated": bool(
            len(accepted) and accepted.EEG_frequency_selection_correct.mean()
            >= float(criteria.minimum_accepted_carrier_accuracy)
        ),
        "complete_controller_grid_for_enrolled_contexts": bool(
            len(expected)
            and expected.groupby("context_id").controller_mode.nunique().min()
            == len(EXPECTED_MODES)
        ),
        "multiple_independent_paired_postdecision_futures": bool(
            smoke or (
                len(expected)
                and expected.n_futures.min()
                >= int(criteria.minimum_future_continuations)
            )
        ),
        "identical_predecision_neural_EEG_across_actions_and_futures": bool(
            len(metrics)
            and metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "identical_predecision_observation_noise_across_actions_and_futures": bool(
            len(metrics)
            and metrics.groupby("context_id").observed_baseline_sha256.nunique().max()
            == 1
        ),
        "all_active_actions_use_frozen_EEG_carrier": bool(
            len(active)
            and active.action_frequency_uses_frozen_EEG_estimator.all()
            and np.allclose(active.action_frequency_hz, active.EEG_selected_frequency_hz)
        ),
        "both_active_profiles_use_identical_0p2_V_per_m": bool(
            len(active) and np.allclose(active.amplitude_v_per_m, 0.2)
        ),
        "one_controller_profile_fixed_for_each_intervention": bool(
            len(active)
            and active.groupby([
                "context_id", "future_index", "controller_mode"
            ]).size().eq(1).all()
        ),
        "phase_updates_use_only_preceding_observed_EEG": bool(
            len(updates)
            and updates.estimate_is_strictly_causal.all()
            and (updates.estimate_stop_ms - updates.boundary_ms).max()
            <= float(criteria.maximum_causal_timing_error_ms)
        ),
        "phase_correction_is_frequency_bounded": bool(
            len(adaptive_updates)
            and adaptive_updates.frequency_correction_hz.abs().max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "field_waveform_continuous_across_updates": bool(
            len(active)
            and active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
        ),
        "common_phase_estimates_actionable": bool(
            len(active)
            and active.common_phase_estimate_actionable_fraction.mean()
            >= float(criteria.minimum_common_phase_estimate_actionable_fraction)
        ),
        "all_actions_rate_safe": bool(len(metrics) and metrics.rate_safe.all()),
        "field_removal_recovered": bool(
            len(metrics) and metrics.field_removal_recovered.all()
            and np.allclose(metrics.final_extracellular_residual_mV, 0.0)
        ),
        "expected_oracle_uses_both_controller_profiles": all(
            int(profile_counts.get(mode, 0))
            >= (1 if smoke else int(criteria.minimum_contexts_per_optimal_profile))
            for mode in (CONSERVATIVE, RESPONSIVE)
        ),
        "both_optimal_profiles_replicate_across_structures": all(
            int(profile_structures.get(mode, 0))
            >= (1 if smoke else int(criteria.minimum_structures_per_optimal_profile))
            for mode in (CONSERVATIVE, RESPONSIVE)
        ),
        "both_profiles_have_practical_contextual_support": all(
            int(practical_counts.get(mode, 0))
            >= (0 if smoke else int(criteria.minimum_practical_contexts_per_profile))
            for mode in (CONSERVATIVE, RESPONSIVE)
        ),
        "practical_crossovers_replicate_across_structures": all(
            int(practical_structures.get(mode, 0))
            >= (0 if smoke else int(criteria.minimum_practical_structures_per_profile))
            for mode in (CONSERVATIVE, RESPONSIVE)
        ),
        "expected_oracle_has_practical_advantage_over_best_fixed": bool(
            opportunity["mean_oracle_advantage_over_best_fixed_log10"]
            >= float(criteria.minimum_oracle_advantage_over_best_fixed_log10)
        ),
        "expected_oracle_has_practical_advantage_over_H4_profile": bool(
            opportunity["mean_oracle_advantage_over_h4_profile_log10"]
            >= float(criteria.minimum_oracle_advantage_over_h4_profile_log10)
        ),
        "oracle_opportunity_positive_across_structures": bool(
            opportunity["positive_structure_oracle_fraction"]
            >= float(criteria.minimum_positive_structure_oracle_fraction)
        ),
        "realized_optimal_profile_reproducible_across_futures": bool(
            opportunity["mean_realized_optimal_profile_agreement_fraction"]
            >= float(criteria.minimum_realized_winner_agreement_fraction)
        ),
        "shared_drive_changes_relative_controller_response": bool(
            opportunity[
                "absolute_shared_drive_by_profile_response_interaction_log10"
            ] >= float(criteria.minimum_shared_drive_response_interaction_log10)
        ),
        "shared_drive_observable_from_predecision_EEG": bool(
            np.isfinite(observability["LOSO_shared_drive_balanced_accuracy"])
            and observability["LOSO_shared_drive_balanced_accuracy"]
            >= float(criteria.minimum_shared_drive_classification_balanced_accuracy)
        ) or smoke,
        "at_least_one_observed_EEG_feature_maps_relative_response": bool(
            associations.passes_exploratory_response_association_gate.any()
        ) or smoke,
        "response_feature_mapping_excludes_hidden_state_and_spikes": bool(
            association_audit["features_are_phase_invariant_predecision_observed_EEG"]
            and association_audit["hidden_state_and_spikes_excluded"]
        ),
        "efficacy_uses_neural_only_EEG_and_controller_uses_observed_EEG": bool(
            len(metrics)
            and metrics.context_features_use_observed_EEG.all()
            and metrics.efficacy_endpoint_uses_neural_only_EEG.all()
            and (~metrics.policy_uses_hidden_state_or_spikes.astype(bool)).all()
        ),
    }
    opportunity_names = [
        "expected_oracle_uses_both_controller_profiles",
        "both_optimal_profiles_replicate_across_structures",
        "both_profiles_have_practical_contextual_support",
        "practical_crossovers_replicate_across_structures",
        "expected_oracle_has_practical_advantage_over_best_fixed",
        "expected_oracle_has_practical_advantage_over_H4_profile",
        "oracle_opportunity_positive_across_structures",
        "realized_optimal_profile_reproducible_across_futures",
        "shared_drive_changes_relative_controller_response",
        "shared_drive_observable_from_predecision_EEG",
        "at_least_one_observed_EEG_feature_maps_relative_response",
    ]
    passed = bool(all(checks.values()) and not smoke)
    conclusions = {
        "H5_P1_contextual_response_mapping": "PASSED" if passed else "NOT PASSED",
        "ready_for_H5_policy_development": passed,
        "failed_opportunity_checks": [
            name for name in opportunity_names if not checks[name]
        ],
        "machine_learning_policy_status": "NOT TRAINED OR TESTED",
    }
    return checks, conclusions


def _save_figure(figure: Any, root: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(root / f"{stem}.png", dpi=300)
    figure.savefig(root / f"{stem}.pdf")
    plt.close(figure)


def _plots(
    *, root: Path, screening: pd.DataFrame, spectra: pd.DataFrame,
    expected: pd.DataFrame, action_map: pd.DataFrame, structure: pd.DataFrame,
    association_audit: dict[str, Any],
    metrics: pd.DataFrame, cfg: DictConfig,
) -> None:
    colors = {PARTIAL: "#4C78A8", FULL: "#E45756"}
    markers = {LOW: "o", HIGH: "^"}

    summary = screening.groupby("shared_drive_label").agg(
        carrier_accuracy=("EEG_frequency_selection_correct", "mean"),
        carrier_coverage=("carrier_identified", "mean"),
        enrollment=("eligible", "mean"),
    ).reindex([PARTIAL, FULL])
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(2)
    width = 0.24
    for offset, column, label in (
        (-width, "carrier_accuracy", "forced carrier accuracy"),
        (0.0, "carrier_coverage", "carrier decision coverage"),
        (width, "enrollment", "final enrollment"),
    ):
        axis.bar(x + offset, summary[column], width=width, label=label)
    axis.set_xticks(x, ["partial shared drive", "full shared drive"])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Fraction of screened contexts")
    axis.set_title("Frozen noisy-EEG carrier screening")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, root, "figure_01_carrier_screening")

    if not spectra.empty:
        keys = list(spectra.groupby("context_id", sort=False))
        figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.0), sharex=True, sharey=True)
        screen_index = screening.set_index("context_id")
        for axis, (context_id, group) in zip(axes.flat, keys):
            view = group[group.frequency_hz.between(6.0, 14.0)]
            axis.plot(
                view.frequency_hz,
                view.observed_EEG_multitaper_residual_db,
                color="#2166ac", linewidth=1.7,
            )
            axis.axvspan(8.25, 9.75, color="#fdae61", alpha=0.14)
            axis.axvspan(10.25, 11.75, color="#8073ac", alpha=0.12)
            row = screen_index.loc[str(context_id)]
            axis.axvline(
                float(row.hidden_frequency_hz), color="black", linestyle="--",
                linewidth=1,
            )
            axis.set_title(
                f"true {float(row.hidden_frequency_hz):g} Hz; {row['label']}\n"
                f"{row.shared_drive_label}; choose "
                f"{float(row.EEG_selected_frequency_hz):g} Hz "
                f"({'accept' if bool(row.carrier_identified) else 'abstain'})",
                fontsize=8,
            )
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("Aperiodic-adjusted power (dB)")
        for axis in axes.flat[len(keys):]:
            axis.set_visible(False)
        figure.suptitle("Representative prospective predecision spectra")
        _save_figure(figure, root, "figure_02_representative_predecision_PSD")

    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    for (shared, diffusion), group in screening.groupby([
        "shared_drive_label", "label"
    ]):
        axis.scatter(
            group.context_C1,
            group.carrier_soft_support_fraction,
            color=colors[str(shared)], marker=markers[str(diffusion)], s=55,
            alpha=0.85,
            label=f"{str(shared).replace('_', ' ')}, {str(diffusion).replace('_', ' ')}",
        )
    axis.set_xlabel("Predecision phase-increment coherence C1")
    axis.set_ylabel("Frozen carrier temporal support")
    axis.set_title("Observed phase-invariant EEG context")
    axis.legend(frameon=False, fontsize=7)
    _save_figure(figure, root, "figure_03_observed_EEG_context")

    active = expected[expected.controller_mode.isin([CONSERVATIVE, RESPONSIVE])]
    structure_response = active.groupby([
        "structure_seed", "shared_drive_label", "controller_mode"
    ], as_index=False).expected_post_distance_to_B_log10.mean()
    response_summary = structure_response.groupby([
        "shared_drive_label", "controller_mode"
    ]).expected_post_distance_to_B_log10.agg(["mean", "sem"]).reset_index()
    figure, axis = plt.subplots(figsize=(7.0, 4.7))
    x = np.arange(2)
    width = 0.34
    for index, mode in enumerate((CONSERVATIVE, RESPONSIVE)):
        group = response_summary[
            response_summary.controller_mode.eq(mode)
        ].set_index("shared_drive_label").reindex([PARTIAL, FULL])
        axis.bar(
            x + (index - 0.5) * width, group["mean"], width,
            yerr=group["sem"], capsize=3,
            label=("conservative 1 s / 250 ms" if mode == CONSERVATIVE
                   else "responsive 0.5 s / 125 ms"),
        )
    axis.set_xticks(x, ["partial shared drive", "full shared drive"])
    axis.set_ylabel("Expected neural-EEG distance to B (log10)")
    axis.set_title("Controller response; structure-level mean ± SEM")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, root, "figure_04_controller_response")

    candidate = association_audit["selected_candidate_response_feature"]
    if candidate is None:
        candidate = "context_C1"
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    for (shared, diffusion), group in action_map.groupby([
        "shared_drive_label", "label"
    ]):
        axis.scatter(
            group[candidate],
            group.responsive_advantage_over_conservative_log10,
            color=colors[str(shared)], marker=markers[str(diffusion)], s=55,
            alpha=0.85,
            label=f"{str(shared).replace('_', ' ')}, {str(diffusion).replace('_', ' ')}",
        )
    margin = float(cfg.analysis.response_mapping.practical_context_margin_log10)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axhline(margin, color="0.45", linestyle="--", linewidth=1)
    axis.axhline(-margin, color="0.45", linestyle="--", linewidth=1)
    axis.set_xlabel(candidate.replace("_", " "))
    axis.set_ylabel("Responsive advantage over conservative (log10)")
    axis.set_title("Full-information EEG context–controller response map")
    axis.legend(frameon=False, fontsize=7)
    _save_figure(figure, root, "figure_05_context_controller_interaction")

    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(structure))
    width = 0.36
    axis.bar(
        x - width / 2,
        structure.mean_oracle_advantage_over_best_fixed_log10,
        width=width, label="oracle vs best fixed",
    )
    axis.bar(
        x + width / 2,
        structure.mean_oracle_advantage_over_h4_profile_log10,
        width=width, label="oracle vs frozen H4 profile",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, structure.structure_seed.astype(str), rotation=45)
    axis.set_xlabel("Independent circuit structure seed")
    axis.set_ylabel("Expected distance advantage (log10)")
    axis.set_title("Full-information opportunity by independent structure")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, root, "figure_06_structure_opportunity")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for shared, group in action_map.groupby("shared_drive_label"):
        axes[0].scatter(
            np.full(len(group), 0 if shared == PARTIAL else 1)
            + np.linspace(-0.06, 0.06, len(group)),
            group.realized_optimal_profile_agreement_fraction,
            color=colors[str(shared)], alpha=0.75, s=35,
        )
    axes[0].set_xticks([0, 1], ["partial", "full"])
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Expected-winner agreement across futures")
    axes[0].set_title("Paired-future reliability")
    phase = metrics[metrics.controller_mode.isin([CONSERVATIVE, RESPONSIVE])]
    phase_structure = phase.groupby([
        "structure_seed", "controller_mode"
    ], as_index=False).mean_abs_common_phase_error_rad.mean()
    for index, mode in enumerate((CONSERVATIVE, RESPONSIVE)):
        values = phase_structure[
            phase_structure.controller_mode.eq(mode)
        ].mean_abs_common_phase_error_rad
        axes[1].scatter(
            np.full(len(values), index), values, s=40,
            color=("#777777" if mode == CONSERVATIVE else "#1b9e77"),
        )
    axes[1].set_xticks([0, 1], ["1 s / 250 ms", "0.5 s / 125 ms"])
    axes[1].set_ylabel("Mean common phase error (rad)")
    axes[1].set_title("Causal controller tracking audit")
    _save_figure(figure, root, "figure_07_reliability_and_phase")


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
        print("\n### H5-P1 full-information response mapping")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    target = sources["target"]

    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    representative_spectra: list[pd.DataFrame] = []
    representative_temporal: list[pd.DataFrame] = []
    contexts = _run_context_specs(cfg)
    representative_structure = min(int(row["structure_seed"]) for row in contexts)
    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"D={context['diffusion_rad2_per_s']:g} rad^2/s "
                f"q={context['shared_modulated_fraction']:g}"
            )
        state_cfg = _with_context_state(cfg, context)
        first_future = _future_seed(cfg, context, 0)
        baseline_reference = _run_p1_controller(
            condition_cfg=state_cfg,
            context=context,
            future_seed=first_future,
            future_index=0,
            mode=SHAM,
            action_index=0,
            root=root,
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            screening, spectrum, temporal = _frozen_carrier_screen(
                baseline_reference, context, target, cfg
            )
            screening_rows.append(screening)
            if int(context["structure_seed"]) == representative_structure:
                representative_spectra.append(spectrum)
                representative_temporal.append(temporal)
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
        futures = int(cfg.analysis.crossed_design.n_future_continuations)
        for future_index in range(futures):
            future_seed = _future_seed(cfg, context, future_index)
            episodes: dict[str, dict[str, Any]] | None = {} if rank == 0 else None
            for action_index, mode in enumerate(_controller_modes(cfg)):
                if future_index == 0 and mode == SHAM:
                    episode = baseline_reference
                else:
                    episode = _run_p1_controller(
                        condition_cfg=action_cfg,
                        context=context,
                        future_seed=future_seed,
                        future_index=future_index,
                        mode=mode,
                        action_index=action_index,
                        root=root,
                        comm=comm,
                        size=size,
                        rank=rank,
                    )
                if rank == 0:
                    episodes[mode] = episode
            if rank == 0:
                rows, trajectories, updates = _metric_rows(
                    context=context,
                    screening=screening,
                    future_index=future_index,
                    future_seed=future_seed,
                    episodes=episodes,
                    baseline_reference=baseline_reference,
                    target=target,
                    cfg=cfg,
                )
                _augment_metric_rows(rows, episodes, cfg)
                _augment_common_audit(rows, episodes, cfg)
                _augment_observation_rows(rows, episodes)
                _add_p1_context_features(rows, screening)
                shared_fields = {
                    "paired_shared_drive_context_id": str(
                        context["paired_shared_drive_context_id"]
                    ),
                    "shared_drive_label": str(context["shared_drive_label"]),
                    "shared_modulated_fraction": float(
                        context["shared_modulated_fraction"]
                    ),
                    "carrier_estimator": str(screening["carrier_estimator"]),
                    "carrier_identified": bool(screening["carrier_identified"]),
                    "EEG_selected_frequency_hz": selected_frequency,
                }
                for row in trajectories:
                    row.update(shared_fields)
                for row in updates:
                    row.update(shared_fields)
                metric_rows.extend(rows)
                trajectory_rows.extend(trajectories)
                update_rows.extend(updates)
        del baseline_reference

    if rank != 0:
        return
    screening_table = pd.DataFrame(screening_rows)
    screening_table.to_csv(root / "prospective_screening.csv", index=False)
    spectra = (
        pd.concat(representative_spectra, ignore_index=True)
        if representative_spectra else pd.DataFrame()
    )
    temporal = (
        pd.concat(representative_temporal, ignore_index=True)
        if representative_temporal else pd.DataFrame()
    )
    if bool(cfg.analysis.save_representative_spectra):
        spectra.to_csv(root / "representative_predecision_spectra.csv", index=False)
        temporal.to_csv(root / "representative_temporal_evidence.csv", index=False)
    if not metric_rows:
        conclusion = {
            "scope": "H5-P1 full-information response mapping",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "H5_P1_contextual_response_mapping": "NOT PASSED",
                "ready_for_H5_policy_development": False,
                "machine_learning_policy_status": "NOT TRAINED OR TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "stopped_after_safe_sham_fallback_screening": True,
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(
            conclusion, indent=2, allow_nan=False
        ))
        print("No eligible contexts; H5-P1 response mapping stopped.")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    updates = pd.DataFrame(update_rows)
    expected = _expected_map(metrics)
    action_map, structure, opportunity = _response_map(expected, metrics, cfg)
    classifier_rows, observability = _shared_drive_loso(screening_table)
    associations, association_audit = _feature_response_associations(
        action_map, cfg
    )
    checks, conclusions = _checks(
        screening=screening_table,
        metrics=metrics,
        expected=expected,
        updates=updates,
        action_map=action_map,
        structure=structure,
        observability=observability,
        opportunity=opportunity,
        associations=associations,
        association_audit=association_audit,
        sources=sources,
        cfg=cfg,
    )

    metrics.to_csv(root / "context_controller_future_metrics.csv", index=False)
    trajectories.to_csv(root / "one_second_eeg_trajectories.csv", index=False)
    updates.to_csv(root / "causal_phase_updates.csv", index=False)
    expected.to_csv(root / "expected_context_controller_map.csv", index=False)
    action_map.to_csv(root / "controller_profile_response_map.csv", index=False)
    structure.to_csv(root / "structure_level_oracle_opportunity.csv", index=False)
    classifier_rows.to_csv(root / "shared_drive_observability_loso.csv", index=False)
    associations.to_csv(root / "EEG_feature_response_associations.csv", index=False)
    audit = {
        "carrier_measurement": {
            "frozen_estimator": MT_POOLED,
            "screened_contexts": int(len(screening_table)),
            "carrier_identification_coverage": float(
                screening_table.carrier_identified.mean()
            ),
            "accepted_carrier_accuracy": float(
                screening_table.loc[
                    screening_table.carrier_identified.astype(bool),
                    "EEG_frequency_selection_correct",
                ].mean()
            ),
            "final_enrollment_fraction": float(screening_table.eligible.mean()),
            "abstention_fallback": SHAM,
        },
        "shared_drive_observability": observability,
        "controller_profile_opportunity": opportunity,
        "EEG_feature_response_mapping": association_audit,
    }
    (root / "H5_P1_response_mapping_audit.json").write_text(json.dumps(
        _json_ready(audit), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H5_P1_full_information_response_mapping",
        "frozen_sources": {
            "roots": sources["roots"], "hashes": sources["hashes"]
        },
        "frozen_population_B_target": target,
        "frozen_population_B_target_source_sha256": sources["target_source_hash"],
        "state_generator": {
            "carrier_hz": [9.0, 11.0],
            "phase_diffusion_rad2_per_s": [0.5, 2.0],
            "shared_modulated_afferent_fraction": [0.5, 1.0],
            "modulation_depth": 0.04,
            "mean_afferent_rate_is_matched": True,
            "private_Poisson_events_are_independent": True,
        },
        "causal_protocol": {
            "predecision_observed_EEG_s": int(cfg.analysis.timeline.baseline_steps),
            "intervention_s": int(cfg.analysis.timeline.stimulation_steps),
            "washout_s": int(cfg.analysis.timeline.washout_steps),
            "carrier_estimator": MT_POOLED,
            "estimator_refitted": False,
            "abstention_fallback": SHAM,
            "active_amplitude_v_per_m": 0.2,
            "relative_phase_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
            "montage": str(cfg.analysis.tacs.axial_montage),
            "controller_profiles": {
                mode: _profile(cfg, mode) for mode in EXPECTED_MODES
            },
        },
        "observation_and_outcome": {
            "controller_and_context_input": "neural EEG plus frozen AR(1) noise",
            "noise_RMS_fraction": 0.25,
            "AR1_coefficient": 0.95,
            "efficacy_endpoint": "ideal neural-only EEG distance to frozen B",
            "stimulation_artifact_modelled": False,
        },
        "design": {
            "independent_structures": int(
                cfg.analysis.crossed_design.n_structure_seeds
            ),
            "paired_futures": int(
                cfg.analysis.crossed_design.n_future_continuations
            ),
            "crossed_repeats": "9/11 Hz x low/high D x q=0.5/1.0",
            "statistical_unit": "independent circuit structure",
        },
        "inference_boundary": (
            "Exploratory system identification. The full-information oracle and "
            "feature-response associations are not deployable policies."
        ),
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-P1 full-information EEG-context/controller response mapping",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure",
        "inference_boundary": (
            "H5-P1 maps opportunity only. It does not train or test a "
            "machine-learning stimulation policy and cannot establish H5."
        ),
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot):
        _plots(
            root=root,
            screening=screening_table,
            spectra=spectra,
            expected=expected,
            action_map=action_map,
            structure=structure,
            association_audit=association_audit,
            metrics=metrics,
            cfg=cfg,
        )

    print("\n### H5-P1 screening")
    print(f"contexts screened: {len(screening_table)}")
    print(f"eligible contexts: {int(screening_table.eligible.sum())}")
    print(f"screening yield: {float(screening_table.eligible.mean()):.3f}")
    print("\n### H5-P1 response-mapping checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### H5-P1 response opportunity")
    print(json.dumps(_json_ready(audit), indent=2, allow_nan=False))
    print(
        "\nContextual response-mapping gate: "
        f"{conclusions['H5_P1_contextual_response_mapping']}"
    )
    print("Machine-learning policy status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
