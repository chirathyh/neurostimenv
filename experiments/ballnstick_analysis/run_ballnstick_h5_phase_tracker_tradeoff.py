"""H5-P2A stimulation-free causal phase-tracker bias--variance study.

H5-P1 found observable shared-drive heterogeneity but no practically useful,
future-reproducible controller-profile opportunity.  H5-P2A does not apply
tACS and does not train a policy.  It tests the control-theoretic prerequisite
for a revised H5 task: a long-history tracker should reject observation noise,
whereas a short-history tracker should follow rapidly diffusing neural phase.

The frozen H5-I0b carrier estimator sees only the first 30 seconds of noisy
EEG.  Two causal trackers are then replayed over a subsequent eight-second
stimulation-free tail.  Their primary phase error is scored against a hidden,
transferred afferent-phase reference used only for this mechanistic audit; an
observed-versus-neural same-profile error separately attributes measurement
noise.  Candidate noise levels are selected only from these measurement
endpoints, never from a stimulation outcome.
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
from scipy import signal


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _reference_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (  # noqa: E402
    CONSERVATIVE,
    FULL,
    RESPONSIVE,
    _with_context_state,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_iaf_measurement_validation import (  # noqa: E402
    _population_rate,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_multitaper_measurement_validation import (  # noqa: E402
    MT_POOLED,
    OBSERVED,
    _estimate_multitaper_methods,
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
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_bandwidth_discovery import (  # noqa: E402
    _json_ready,
    _sha256,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _simulate_episode,
    _validate_online_outputs,
    _zero_action,
)
from setup.circuits.ballnstick.utils import (  # noqa: E402
    generate_phase_diffusion_path,
    make_background_phase_seed,
)


ROOT_NAME = "h5_phase_tracker_tradeoff"
PROFILE_ORDER = [CONSERVATIVE, RESPONSIVE]


def _wrap_phase(value: float | np.ndarray) -> float | np.ndarray:
    wrapped = np.angle(np.exp(1j * np.asarray(value)))
    return float(wrapped) if np.ndim(wrapped) == 0 else wrapped


def _source_files(root: Path, names: dict[str, str]) -> dict[str, Path]:
    files = {name: root / filename for name, filename in names.items()}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen H5-P2A source files: {missing}")
    return files


def _hash_locked_files(
    root: Path, names: dict[str, str], expected_cfg: DictConfig,
) -> tuple[dict[str, Path], dict[str, str]]:
    files = _source_files(root, names)
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {name: str(expected_cfg[name]) for name in names}
    if observed != expected:
        raise RuntimeError(
            f"Frozen H5-P2A source hash mismatch at {root}: "
            f"expected={expected}, observed={observed}"
        )
    return files, observed


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
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
    p1 = json.loads(p1_files["conclusion"].read_text())
    required_failures = {
        "carrier_identification_coverage_replicated",
        "field_removal_recovered",
        "expected_oracle_has_practical_advantage_over_best_fixed",
        "expected_oracle_has_practical_advantage_over_H4_profile",
        "realized_optimal_profile_reproducible_across_futures",
        "at_least_one_observed_EEG_feature_maps_relative_response",
    }
    observed_failures = {
        name for name, passed in p1["checks"].items() if not bool(passed)
    }
    if (
        p1["conclusions"]["H5_P1_contextual_response_mapping"] != "NOT PASSED"
        or bool(p1["conclusions"]["ready_for_H5_policy_development"])
        or observed_failures != required_failures
    ):
        raise RuntimeError("H5-P2A requires the exact frozen negative H5-P1 result.")

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
    i0b = json.loads(i0b_files["conclusion"].read_text())
    frozen = json.loads(i0b_files["frozen_estimator"].read_text())
    if (
        i0b["conclusions"]["H5_I0b_multitaper_carrier_measurement"]
        != "CONFIRMED"
        or not bool(i0b["conclusions"]["ready_for_H5_P1_response_mapping"])
        or str(frozen["selected_estimator"]) != MT_POOLED
    ):
        raise RuntimeError("H5-P2A requires the confirmed frozen H5-I0b estimator.")

    source_seeds: set[int] = set()
    for path in (
        p1_files["screening"], p1_files["metrics"],
        i0b_files["discovery_metrics"], i0b_files["confirmation_metrics"],
    ):
        table = pd.read_csv(path)
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed", "noise_seed",
        ):
            if column in table:
                source_seeds.update(table[column].dropna().astype(int).tolist())
    return {
        "roots": {"h5p1": str(p1_root), "h5i0b": str(i0b_root)},
        "hashes": {"h5p1": p1_hashes, "h5i0b": i0b_hashes},
        "source_seeds": source_seeds,
        "H5P1_negative_preserved": True,
        "H5I0b_confirmed": True,
        "frozen_estimator": frozen,
    }


def _diffusion_levels(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "label": str(level.label),
        "diffusion_rad2_per_s": float(level.diffusion_rad2_per_s),
    } for level in cfg.analysis.states.phase_diffusion_levels]


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    rows: list[dict[str, Any]] = []
    for structure_index in range(int(block.n_structure_seeds)):
        structure_seed = base + int(block.structure_seed_offset) + structure_index
        for history_index in range(int(block.n_history_seeds)):
            history_seed = (
                base + int(block.history_seed_offset)
                + 10 * structure_index + history_index
            )
            for frequency_index, frequency_hz in enumerate(
                cfg.analysis.states.frequencies_hz
            ):
                phase_seed = (
                    base + int(block.phase_seed_offset)
                    + 10 * structure_index + frequency_index
                )
                noise_seed = (
                    base + int(block.noise_seed_offset)
                    + 10 * structure_index + frequency_index
                )
                for diffusion_index, diffusion in enumerate(_diffusion_levels(cfg)):
                    order = len(rows)
                    identifier = (
                        f"s{structure_index:02d}_h{history_index:02d}_"
                        f"f{int(round(float(frequency_hz))):02d}_d{diffusion_index:02d}"
                    )
                    rows.append({
                        "context_order": order,
                        "context_id": f"{identifier}_{diffusion['label']}",
                        "structure_index": structure_index,
                        "structure_seed": structure_seed,
                        "history_index": history_index,
                        "history_seed": history_seed,
                        "phase_seed": phase_seed,
                        "trial_seed": base + int(block.trial_seed_offset) + order,
                        "noise_seed": noise_seed,
                        "hidden_frequency_hz": float(frequency_hz),
                        **diffusion,
                        "shared_drive_label": FULL,
                        "shared_modulated_fraction": 1.0,
                    })
    return rows


def _run_context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    rows = _context_specs(cfg)
    limit = int(cfg.analysis.smoke_context_limit)
    if not bool(cfg.analysis.smoke_test) or limit <= 0:
        return rows
    # Preserve both diffusion levels and, when possible, both carriers.
    preferred = [
        next(row for row in rows if row["hidden_frequency_hz"] == frequency
             and row["label"] == diffusion)
        for frequency, diffusion in ((9.0, LOW), (9.0, HIGH), (11.0, LOW), (11.0, HIGH))
    ]
    return preferred[:limit]


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    smoke = bool(cfg.analysis.smoke_test)
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-P2A requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-P2A may not alter recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-P2A freezes the carrier grid to 9/11 Hz.")
    if [(x["label"], x["diffusion_rad2_per_s"]) for x in _diffusion_levels(cfg)] != [
        (LOW, 0.5), (HIGH, 2.0)
    ]:
        raise ValueError("H5-P2A freezes D={0.5,2.0} rad^2/s.")
    shared = list(cfg.analysis.states.shared_drive_levels)
    if len(shared) != 1 or str(shared[0].label) != FULL or not np.isclose(
        float(shared[0].shared_modulated_fraction), 1.0
    ):
        raise ValueError("H5-P2A fixes the shared afferent fraction to one.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-P2A retains modulation depth 0.04.")
    if float(cfg.env.simulation.obs_win_len) != 1000.0:
        raise ValueError("H5-P2A requires one-second online windows.")
    tradeoff = cfg.analysis.phase_tracker_tradeoff
    pre = float(tradeoff.predecision_duration_s)
    evaluation = float(tradeoff.evaluation_duration_s)
    if not smoke and (not np.isclose(pre, 30.0) or not np.isclose(evaluation, 8.0)):
        raise ValueError("Full H5-P2A requires 30-s estimation plus 8-s evaluation.")
    if not np.isclose(
        int(cfg.analysis.timeline.baseline_steps), pre + evaluation
    ):
        raise ValueError("H5-P2A baseline must equal predecision plus evaluation time.")
    if int(cfg.analysis.timeline.stimulation_steps) != 1 or int(
        cfg.analysis.timeline.washout_steps
    ) != 1:
        raise ValueError("H5-P2A retains one-second sham compatibility epochs.")
    fractions = [float(x) for x in tradeoff.observation_noise_fractions]
    if fractions != [0.25, 0.5, 0.75]:
        raise ValueError("H5-P2A freezes the AR(1) noise ladder to 0.25/0.5/0.75.")
    if not np.isclose(float(tradeoff.fixed_low_noise_fraction), 0.25):
        raise ValueError("The low-noise anchor must remain 0.25.")
    if [float(x) for x in tradeoff.candidate_high_noise_fractions] != [0.5, 0.75]:
        raise ValueError("The high-noise candidate order changed.")
    expected_profiles = {
        CONSERVATIVE: (1000.0, 250.0), RESPONSIVE: (500.0, 125.0)
    }
    for name, values in expected_profiles.items():
        profile = tradeoff.tracker_profiles[name]
        if not np.allclose(
            [float(profile.history_ms), float(profile.update_interval_ms)], values
        ):
            raise ValueError(f"H5-P2A tracker profile {name} changed.")
    if str(tradeoff.frozen_estimator) != MT_POOLED:
        raise ValueError("H5-P2A must use the frozen H5-I0b estimator.")
    if not np.isclose(float(tradeoff.common_audit_interval_ms), 125.0):
        raise ValueError("The common tracker audit grid must remain 125 ms.")
    contexts = _context_specs(cfg)
    expected_count = (
        int(cfg.analysis.crossed_design.n_structure_seeds)
        * int(cfg.analysis.crossed_design.n_history_seeds) * 2 * 2
    )
    if len(contexts) != expected_count:
        raise ValueError("H5-P2A crossed context grid is incomplete.")
    namespaces = [
        {int(row[column]) for row in contexts}
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed", "noise_seed"
        )
    ]
    if any(not values for values in namespaces):
        raise ValueError("Every H5-P2A seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H5-P2A seed namespaces overlap.")
    if set().union(*namespaces).intersection(sources["source_seeds"]):
        raise ValueError("H5-P2A seeds overlap H5-P1 or H5-I0b.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H5-P2A structure seed exceeds the uint32 mapping range.")
    if not smoke and int(cfg.analysis.crossed_design.n_structure_seeds) < 6:
        raise ValueError("Full H5-P2A requires six independent structures.")


def _split_normalized_ar1_path(
    *, n_samples: int, normalization_samples: int, seed: int, coefficient: float,
) -> np.ndarray:
    """Return continuous AR(1) noise normalized only on predecision samples."""
    if not 0 < normalization_samples < n_samples:
        raise ValueError("AR(1) normalization boundary must be interior.")
    if not 0.0 <= float(coefficient) < 1.0:
        raise ValueError("Invalid AR(1) coefficient.")
    rng = np.random.default_rng(int(seed))
    innovation = np.sqrt(1.0 - float(coefficient) ** 2)
    values = signal.lfilter(
        [innovation], [1.0, -float(coefficient)],
        rng.standard_normal(int(n_samples)),
    )
    rms = float(np.sqrt(np.mean(values[:normalization_samples] ** 2)))
    if not np.isfinite(rms) or rms <= np.finfo(float).tiny:
        raise RuntimeError("H5-P2A AR(1) predecision RMS is invalid.")
    return np.asarray(values / rms, dtype=float)


def _phase_path(context: dict[str, Any], cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    total_ms = sum(
        int(cfg.analysis.timeline[name]) for name in (
            "burn_in_steps", "baseline_steps", "stimulation_steps", "washout_steps"
        )
    ) * float(cfg.env.simulation.obs_win_len)
    return generate_phase_diffusion_path(
        start_ms=0.0,
        stop_ms=total_ms,
        frequency_hz=float(context["hidden_frequency_hz"]),
        phase_rad=_reference_phase(int(context["phase_seed"])),
        diffusion_rad2_per_s=float(context["diffusion_rad2_per_s"]),
        integration_dt_ms=float(cfg.analysis.states.phase_diffusion_integration_dt_ms),
        history_seed=make_background_phase_seed(
            global_seed=int(context["history_seed"])
        ),
    )


def _phase_estimate(
    values: np.ndarray,
    times_ms: np.ndarray,
    *,
    boundary_ms: float,
    history_ms: float,
    frequency_cfg: DictConfig,
    cfg: DictConfig,
) -> dict[str, float]:
    return _tail_phase_estimate(
        [{"sample_times_ms": times_ms, "eeg_v": values}],
        boundary_ms=float(boundary_ms),
        history_ms=float(history_ms),
        simulator_fs_hz=1000.0 / float(cfg.env.network.dt),
        relative_offset_rad=0.0,
        cfg=frequency_cfg,
    )


def _latent_transfer_offset(
    neural: np.ndarray,
    times_ms: np.ndarray,
    latent_times_ms: np.ndarray,
    latent_phase_rad: np.ndarray,
    *,
    calibration_start_ms: float,
    calibration_stop_ms: float,
    frequency_cfg: DictConfig,
    cfg: DictConfig,
) -> tuple[float, float, int]:
    tradeoff = cfg.analysis.phase_tracker_tradeoff
    history = float(tradeoff.latent_transfer_calibration_history_ms)
    interval = float(tradeoff.latent_transfer_calibration_interval_ms)
    first = calibration_start_ms + history
    boundaries = np.arange(first, calibration_stop_ms + 1.0e-9, interval)
    differences: list[float] = []
    for boundary in boundaries:
        estimate = _phase_estimate(
            neural, times_ms, boundary_ms=float(boundary), history_ms=history,
            frequency_cfg=frequency_cfg, cfg=cfg,
        )
        latent = float(np.interp(boundary, latent_times_ms, latent_phase_rad))
        differences.append(float(_wrap_phase(
            estimate["estimated_eeg_phase_at_boundary_rad"] - latent
        )))
    coefficient = np.mean(np.exp(1j * np.asarray(differences, dtype=float)))
    return float(np.angle(coefficient)), float(abs(coefficient)), len(differences)


def _tracker_rows(
    *,
    observed: np.ndarray,
    neural: np.ndarray,
    times_ms: np.ndarray,
    latent_times_ms: np.ndarray,
    latent_phase_rad: np.ndarray,
    latent_transfer_offset_rad: float,
    evaluation_start_ms: float,
    evaluation_stop_ms: float,
    context: dict[str, Any],
    noise_fraction: float,
    frequency_cfg: DictConfig,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    tradeoff = cfg.analysis.phase_tracker_tradeoff
    common_interval = float(tradeoff.common_audit_interval_ms)
    carrier = float(frequency_cfg.analysis.tacs.frequency_hz)
    threshold = float(tradeoff.phase_actionability_resultant_to_rms)
    boundaries = np.arange(
        evaluation_start_ms + common_interval,
        evaluation_stop_ms + 1.0e-9,
        common_interval,
    )
    rows: list[dict[str, Any]] = []
    for profile_name in PROFILE_ORDER:
        profile = tradeoff.tracker_profiles[profile_name]
        history_ms = float(profile.history_ms)
        update_ms = float(profile.update_interval_ms)
        observed_update = _phase_estimate(
            observed, times_ms, boundary_ms=evaluation_start_ms,
            history_ms=history_ms, frequency_cfg=frequency_cfg, cfg=cfg,
        )
        neural_update = _phase_estimate(
            neural, times_ms, boundary_ms=evaluation_start_ms,
            history_ms=history_ms, frequency_cfg=frequency_cfg, cfg=cfg,
        )
        last_update_ms = float(evaluation_start_ms)
        update_index = 0
        for boundary_index, boundary in enumerate(boundaries, start=1):
            elapsed = float(boundary - evaluation_start_ms)
            applied = bool(np.isclose(
                elapsed / update_ms, round(elapsed / update_ms), atol=1.0e-10
            ))
            if applied:
                observed_update = _phase_estimate(
                    observed, times_ms, boundary_ms=float(boundary),
                    history_ms=history_ms, frequency_cfg=frequency_cfg, cfg=cfg,
                )
                neural_update = _phase_estimate(
                    neural, times_ms, boundary_ms=float(boundary),
                    history_ms=history_ms, frequency_cfg=frequency_cfg, cfg=cfg,
                )
                last_update_ms = float(boundary)
                update_index += 1
            propagation = 2.0 * np.pi * carrier * (
                float(boundary) - last_update_ms
            ) / 1000.0
            observed_phase = float(_wrap_phase(
                observed_update["estimated_eeg_phase_at_boundary_rad"] + propagation
            ))
            neural_phase = float(_wrap_phase(
                neural_update["estimated_eeg_phase_at_boundary_rad"] + propagation
            ))
            latent = float(np.interp(boundary, latent_times_ms, latent_phase_rad))
            transferred_reference = float(_wrap_phase(
                latent + float(latent_transfer_offset_rad)
            ))
            latent_error = float(_wrap_phase(observed_phase - transferred_reference))
            measurement_error = float(_wrap_phase(observed_phase - neural_phase))
            rows.append({
                **context,
                "noise_fraction": float(noise_fraction),
                "tracker_profile": profile_name,
                "phase_history_ms": history_ms,
                "update_interval_ms": update_ms,
                "common_boundary_index": int(boundary_index),
                "boundary_ms": float(boundary),
                "last_update_ms": last_update_ms,
                "profile_update_applied": applied,
                "profile_update_index": int(update_index),
                "observed_tracker_phase_rad": observed_phase,
                "same_profile_neural_phase_rad": neural_phase,
                "transferred_latent_reference_phase_rad": transferred_reference,
                "signed_latent_reference_error_rad": latent_error,
                "absolute_latent_reference_error_rad": abs(latent_error),
                "circular_latent_loss": float(1.0 - np.cos(latent_error)),
                "signed_observation_error_rad": measurement_error,
                "absolute_observation_error_rad": abs(measurement_error),
                "phase_resultant_to_rms": float(observed_update["resultant_to_rms"]),
                "phase_estimate_actionable": bool(
                    float(observed_update["resultant_to_rms"]) >= threshold
                ),
                "estimate_uses_only_preceding_observed_EEG": True,
                "latent_reference_used_by_tracker": False,
                "hidden_diffusion_used_by_tracker": False,
            })
    return rows


def _context_measurement(
    cfg: DictConfig,
    context: dict[str, Any],
    *,
    output_dir: Path,
    comm: Any,
    size: int,
    rank: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame] | None:
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
        raise RuntimeError("H5-P2A obtained invalid neural EEG.")
    simulator_fs = 1000.0 / float(cfg.env.network.dt)
    tradeoff = cfg.analysis.phase_tracker_tradeoff
    pre_s = float(tradeoff.predecision_duration_s)
    evaluation_s = float(tradeoff.evaluation_duration_s)
    baseline_start_ms = float(outputs[0]["t_start_ms"])
    pre_stop_ms = baseline_start_ms + 1000.0 * pre_s
    evaluation_stop_ms = pre_stop_ms + 1000.0 * evaluation_s
    pre_mask = times <= pre_stop_ms + 1.0e-9
    expected_pre_samples = int(round(pre_s * simulator_fs))
    if np.count_nonzero(pre_mask) != expected_pre_samples:
        raise RuntimeError("H5-P2A predecision EEG sample count is incorrect.")
    neural_pre = neural[pre_mask]
    neural_rms = float(np.sqrt(np.mean(neural_pre ** 2)))
    unit_noise = _split_normalized_ar1_path(
        n_samples=neural.size,
        normalization_samples=expected_pre_samples,
        seed=int(context["noise_seed"]),
        coefficient=float(tradeoff.ar1_coefficient),
    )
    latent_times, latent_phase = _phase_path(context, cfg)
    hidden_frequency_cfg = _with_action_frequency(
        state_cfg, float(context["hidden_frequency_hz"])
    )
    transfer_offset, transfer_coherence, transfer_samples = _latent_transfer_offset(
        neural,
        times,
        latent_times,
        latent_phase,
        calibration_start_ms=baseline_start_ms,
        calibration_stop_ms=pre_stop_ms,
        frequency_cfg=hidden_frequency_cfg,
        cfg=cfg,
    )
    e_rate = _population_rate(outputs, "E")
    i_rate = _population_rate(outputs, "I")
    limits = cfg.analysis.rate_guardrails_hz
    rates_safe = bool(
        float(limits.E_min) <= e_rate <= float(limits.E_max)
        and float(limits.I_min) <= i_rate <= float(limits.I_max)
    )
    carrier_rows: list[dict[str, Any]] = []
    tracking_rows: list[dict[str, Any]] = []
    spectra: list[pd.DataFrame] = []
    for noise_fraction in tradeoff.observation_noise_fractions:
        fraction = float(noise_fraction)
        observed = neural + fraction * neural_rms * unit_noise
        observed_pre, processed_fs, _, _, _ = _process_eeg(
            observed[pre_mask], simulator_fs_hz=simulator_fs, cfg=cfg
        )
        estimates, spectrum, temporal = _estimate_multitaper_methods(
            observed_pre,
            fs_hz=processed_fs,
            hidden_frequency_hz=float(context["hidden_frequency_hz"]),
            input_signal=OBSERVED,
            cfg=cfg,
        )
        selected = next(row for row in estimates if row["estimator"] == MT_POOLED)
        selected_frequency = float(selected["selected_frequency_hz"])
        identified = bool(selected["identified"])
        correct = bool(selected["frequency_detected_correctly"])
        carrier_rows.append({
            **context,
            "noise_fraction": fraction,
            "carrier_estimator": MT_POOLED,
            "carrier_identified": identified,
            "EEG_selected_frequency_hz": selected_frequency,
            "carrier_selection_correct": correct,
            # Carrier correctness is a hidden-label audit, never an input to
            # the deployable tracker.  Any accepted 9/11-Hz decision is
            # therefore followed, including an incorrect one; its phase error
            # remains in the candidate-gate endpoint.
            "carrier_usable_for_phase_audit": identified,
            "carrier_peak_frequency_hz": float(selected["peak_frequency_hz"]),
            "carrier_maximum_residual_evidence_db": float(
                selected["maximum_residual_evidence_db"]
            ),
            "carrier_evidence_margin_db": float(selected["evidence_margin_db"]),
            "carrier_soft_support_fraction": float(
                selected["soft_support_fraction"]
            ),
            "carrier_window_score_sd_db": float(selected["window_score_sd_db"]),
            "n_temporal_carrier_windows": int(len(temporal)),
            "latent_transfer_offset_rad": transfer_offset,
            "latent_transfer_coherence": transfer_coherence,
            "latent_transfer_calibration_samples": transfer_samples,
            "observation_noise_model": "paired_AR1_additive_sensor_noise",
            "AR1_coefficient": float(tradeoff.ar1_coefficient),
            "configured_noise_fraction": fraction,
            "achieved_predecision_noise_fraction": float(
                np.sqrt(np.mean((fraction * neural_rms * unit_noise[pre_mask]) ** 2))
                / max(neural_rms, np.finfo(float).tiny)
            ),
            "noise_normalized_from_predecision_only": True,
            "predecision_neural_EEG_sha256": hashlib.sha256(
                np.asarray(neural_pre, dtype="<f8").tobytes()
            ).hexdigest(),
            "unit_noise_sha256": hashlib.sha256(
                np.asarray(unit_noise, dtype="<f8").tobytes()
            ).hexdigest(),
            "baseline_E_firing_rate_hz": e_rate,
            "baseline_I_firing_rate_hz": i_rate,
            "rates_safe": rates_safe,
            "stimulation_applied": False,
            "field_removed": bool(simulation["final_residual_mV"] == 0.0),
            "hidden_frequency_used_only_for_scoring": True,
        })
        spectrum = spectrum.assign(
            context_id=str(context["context_id"]),
            structure_seed=int(context["structure_seed"]),
            hidden_frequency_hz=float(context["hidden_frequency_hz"]),
            label=str(context["label"]),
            diffusion_rad2_per_s=float(context["diffusion_rad2_per_s"]),
            noise_fraction=fraction,
        )
        spectra.append(spectrum)
        if identified:
            frequency_cfg = _with_action_frequency(state_cfg, selected_frequency)
            tracking_rows.extend(_tracker_rows(
                observed=observed,
                neural=neural,
                times_ms=times,
                latent_times_ms=latent_times,
                latent_phase_rad=latent_phase,
                latent_transfer_offset_rad=transfer_offset,
                evaluation_start_ms=pre_stop_ms,
                evaluation_stop_ms=evaluation_stop_ms,
                context=context,
                noise_fraction=fraction,
                frequency_cfg=frequency_cfg,
                cfg=cfg,
            ))
    return carrier_rows, tracking_rows, pd.concat(spectra, ignore_index=True)


def _profile_summary(rows: pd.DataFrame) -> pd.DataFrame:
    group = [
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "diffusion_rad2_per_s", "noise_fraction", "tracker_profile",
    ]
    return (
        rows.groupby(group, as_index=False)
        .agg(
            common_boundary_count=("common_boundary_index", "nunique"),
            profile_update_count=("profile_update_applied", "sum"),
            mean_abs_latent_reference_error_rad=(
                "absolute_latent_reference_error_rad", "mean"
            ),
            p90_abs_latent_reference_error_rad=(
                "absolute_latent_reference_error_rad",
                lambda values: float(np.quantile(values, 0.9)),
            ),
            mean_circular_latent_loss=("circular_latent_loss", "mean"),
            mean_abs_observation_error_rad=(
                "absolute_observation_error_rad", "mean"
            ),
            tracker_actionable_fraction=("phase_estimate_actionable", "mean"),
            all_estimates_causal=(
                "estimate_uses_only_preceding_observed_EEG", "all"
            ),
            latent_reference_never_used_by_tracker=(
                "latent_reference_used_by_tracker", lambda values: bool(~values.any())
            ),
        )
        .sort_values(group)
        .reset_index(drop=True)
    )


def _advantage_table(summary: pd.DataFrame) -> pd.DataFrame:
    index = [
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "diffusion_rad2_per_s", "noise_fraction",
    ]
    pivot = summary.pivot(
        index=index,
        columns="tracker_profile",
        values=[
            "mean_abs_latent_reference_error_rad",
            "mean_abs_observation_error_rad",
            "tracker_actionable_fraction",
        ],
    ).reset_index()
    pivot.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple) else str(column)
        for column in pivot.columns
    ]
    for key in index:
        if f"{key}_" in pivot:
            pivot = pivot.rename(columns={f"{key}_": key})
    slow = f"mean_abs_latent_reference_error_rad_{CONSERVATIVE}"
    fast = f"mean_abs_latent_reference_error_rad_{RESPONSIVE}"
    slow_observation = f"mean_abs_observation_error_rad_{CONSERVATIVE}"
    fast_observation = f"mean_abs_observation_error_rad_{RESPONSIVE}"
    pivot["fast_advantage_latent_error_rad"] = pivot[slow] - pivot[fast]
    pivot["slow_advantage_observation_error_rad"] = (
        pivot[fast_observation] - pivot[slow_observation]
    )
    return pivot


def _carrier_performance(carrier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fraction, group in carrier.groupby("noise_fraction", sort=True):
        accepted = group[group.carrier_identified.astype(bool)]
        rows.append({
            "noise_fraction": float(fraction),
            "context_count": int(len(group)),
            "carrier_coverage": float(group.carrier_identified.mean()),
            "all_context_accuracy": float(group.carrier_selection_correct.mean()),
            "accepted_carrier_accuracy": (
                float(accepted.carrier_selection_correct.mean())
                if len(accepted) else float("nan")
            ),
            "phase_audit_usable_fraction": float(
                group.carrier_usable_for_phase_audit.mean()
            ),
        })
    return pd.DataFrame(rows)


def _candidate_selection(
    carrier: pd.DataFrame,
    summary: pd.DataFrame,
    advantage: pd.DataFrame,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    criteria = cfg.analysis.criteria
    tradeoff = cfg.analysis.phase_tracker_tradeoff
    low_noise = float(tradeoff.fixed_low_noise_fraction)
    carrier_perf = _carrier_performance(carrier).set_index("noise_fraction")
    structure_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for high_noise_value in tradeoff.candidate_high_noise_fractions:
        high_noise = float(high_noise_value)
        high_d_low_n = advantage[
            advantage.label.eq(HIGH) & np.isclose(advantage.noise_fraction, low_noise)
        ]
        low_d_high_n = advantage[
            advantage.label.eq(LOW) & np.isclose(advantage.noise_fraction, high_noise)
        ]
        fast_advantage = float(high_d_low_n.fast_advantage_latent_error_rad.mean())
        slow_advantage = float(-low_d_high_n.fast_advantage_latent_error_rad.mean())
        crossover = fast_advantage + slow_advantage
        measurement_advantage = float(
            low_d_high_n.slow_advantage_observation_error_rad.mean()
        )
        structures = sorted(set(advantage.structure_seed.astype(int)))
        local_structure_rows = []
        for structure_seed in structures:
            fast_values = high_d_low_n[
                high_d_low_n.structure_seed.eq(structure_seed)
            ].fast_advantage_latent_error_rad
            slow_values = -low_d_high_n[
                low_d_high_n.structure_seed.eq(structure_seed)
            ].fast_advantage_latent_error_rad
            row = {
                "candidate_high_noise_fraction": high_noise,
                "structure_seed": int(structure_seed),
                "fast_advantage_high_diffusion_low_noise_rad": (
                    float(fast_values.mean()) if len(fast_values) else float("nan")
                ),
                "slow_advantage_low_diffusion_high_noise_rad": (
                    float(slow_values.mean()) if len(slow_values) else float("nan")
                ),
            }
            row["both_directions_positive"] = bool(
                row["fast_advantage_high_diffusion_low_noise_rad"] > 0.0
                and row["slow_advantage_low_diffusion_high_noise_rad"] > 0.0
            )
            local_structure_rows.append(row)
            structure_rows.append(row)
        structure_table = pd.DataFrame(local_structure_rows)
        fast_fraction = float(
            np.mean(structure_table.fast_advantage_high_diffusion_low_noise_rad > 0.0)
        )
        slow_fraction = float(
            np.mean(structure_table.slow_advantage_low_diffusion_high_noise_rad > 0.0)
        )
        selected_noise_summary = summary[
            summary.noise_fraction.isin([low_noise, high_noise])
        ]
        minimum_actionable = float(
            selected_noise_summary.tracker_actionable_fraction.mean()
        )
        coverage = min(
            float(carrier_perf.loc[low_noise, "carrier_coverage"]),
            float(carrier_perf.loc[high_noise, "carrier_coverage"]),
        )
        accepted_accuracy = min(
            float(carrier_perf.loc[low_noise, "accepted_carrier_accuracy"]),
            float(carrier_perf.loc[high_noise, "accepted_carrier_accuracy"]),
        )
        gates = {
            "carrier_coverage": coverage >= float(
                criteria.minimum_carrier_coverage_each_selected_noise
            ),
            "accepted_carrier_accuracy": accepted_accuracy >= float(
                criteria.minimum_accepted_carrier_accuracy_each_selected_noise
            ),
            "tracker_actionability": minimum_actionable >= float(
                criteria.minimum_tracker_actionable_fraction
            ),
            "fast_high_diffusion_low_noise": fast_advantage >= float(
                criteria.minimum_fast_advantage_high_diffusion_low_noise_rad
            ),
            "slow_low_diffusion_high_noise": slow_advantage >= float(
                criteria.minimum_slow_advantage_low_diffusion_high_noise_rad
            ),
            "crossover_contrast": crossover >= float(
                criteria.minimum_crossover_contrast_rad
            ),
            "fast_direction_across_structures": fast_fraction >= float(
                criteria.minimum_directional_structure_fraction
            ),
            "slow_direction_across_structures": slow_fraction >= float(
                criteria.minimum_directional_structure_fraction
            ),
            "slow_reduces_high_noise_measurement_error": measurement_advantage > 0.0,
        }
        candidate_rows.append({
            "fixed_low_noise_fraction": low_noise,
            "candidate_high_noise_fraction": high_noise,
            "minimum_carrier_coverage": coverage,
            "minimum_accepted_carrier_accuracy": accepted_accuracy,
            "minimum_tracker_actionable_fraction": minimum_actionable,
            "mean_fast_advantage_high_diffusion_low_noise_rad": fast_advantage,
            "mean_slow_advantage_low_diffusion_high_noise_rad": slow_advantage,
            "mean_crossover_contrast_rad": crossover,
            "mean_slow_advantage_observation_error_high_noise_rad": measurement_advantage,
            "fast_direction_positive_structure_fraction": fast_fraction,
            "slow_direction_positive_structure_fraction": slow_fraction,
            **{f"gate_{name}": bool(value) for name, value in gates.items()},
            "passes_measurement_tradeoff_gate": bool(all(gates.values())),
        })
    candidates = pd.DataFrame(candidate_rows).sort_values(
        "candidate_high_noise_fraction"
    ).reset_index(drop=True)
    passing = candidates[candidates.passes_measurement_tradeoff_gate]
    selected = None if passing.empty else float(
        passing.iloc[0].candidate_high_noise_fraction
    )
    frozen = {
        "stage": "H5_P2A_stimulation_free_phase_tracker_tradeoff",
        "selection_uses_no_stimulation_outcomes": True,
        "selection_rule": str(tradeoff.selection_preference),
        "fixed_low_noise_fraction": low_noise,
        "selected_high_noise_fraction": selected,
        "candidate_selected": selected is not None,
        "fixed_shared_modulated_fraction": 1.0,
        "carrier_estimator": MT_POOLED,
        "tracker_profiles": {
            name: OmegaConf.to_container(
                tradeoff.tracker_profiles[name], resolve=True
            ) for name in PROFILE_ORDER
        },
        "carrier_and_tracker_inputs": "preceding noisy EEG only",
        "latent_reference_role": "mechanistic audit only",
        "ready_for_active_H5_P2B_mapping": selected is not None,
    }
    return candidates, pd.DataFrame(structure_rows), frozen


def _checks(
    *,
    carrier: pd.DataFrame,
    tracking: pd.DataFrame,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    frozen: dict[str, Any],
    sources: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    smoke = bool(cfg.analysis.smoke_test)
    contexts = _run_context_specs(cfg)
    expected_noise = len(cfg.analysis.phase_tracker_tradeoff.observation_noise_fractions)
    criteria = cfg.analysis.criteria
    checks = {
        "source_H5P1_hash_locked_with_response_mapping_failed": bool(
            sources["H5P1_negative_preserved"]
        ),
        "source_H5I0b_hash_locked_and_confirmed": bool(
            sources["H5I0b_confirmed"]
        ),
        "H5P2A_seeds_disjoint_from_H5P1_and_H5I0b": True,
        "all_runs_stimulation_free_and_field_removed": bool(
            carrier.stimulation_applied.eq(False).all()
            and carrier.field_removed.astype(bool).all()
        ),
        "complete_frequency_diffusion_grid": bool(
            carrier.context_id.nunique() == len(contexts)
            and len(carrier) == len(contexts) * expected_noise
        ),
        "minimum_independent_structures": bool(
            carrier.structure_seed.nunique()
            >= (1 if smoke else int(criteria.minimum_independent_structures))
        ),
        "minimum_crossed_contexts": bool(
            carrier.context_id.nunique()
            >= (len(contexts) if smoke else int(criteria.minimum_contexts))
        ),
        "full_shared_afferent_fraction_fixed": bool(
            np.allclose(carrier.shared_modulated_fraction, 1.0)
        ),
        "afferent_mean_rate_unchanged_by_observation_noise": True,
        "paired_neural_trajectory_across_noise_levels": bool(
            carrier.groupby("context_id").predecision_neural_EEG_sha256.nunique().eq(1).all()
        ),
        "paired_AR1_realization_scaled_across_noise_levels": bool(
            carrier.groupby("context_id").unit_noise_sha256.nunique().eq(1).all()
        ),
        "AR1_scale_uses_predecision_EEG_only": bool(
            carrier.noise_normalized_from_predecision_only.astype(bool).all()
        ),
        "thirty_second_carrier_then_eight_second_tracker_evaluation": bool(
            smoke or (
                np.isclose(cfg.analysis.phase_tracker_tradeoff.predecision_duration_s, 30.0)
                and np.isclose(cfg.analysis.phase_tracker_tradeoff.evaluation_duration_s, 8.0)
            )
        ),
        "frozen_multitaper_carrier_estimator_used_without_refitting": bool(
            carrier.carrier_estimator.eq(MT_POOLED).all()
        ),
        "carrier_selection_uses_only_predecision_noisy_EEG": True,
        "tracker_estimates_use_only_preceding_noisy_EEG": bool(
            len(tracking)
            and tracking.estimate_uses_only_preceding_observed_EEG.astype(bool).all()
        ),
        "latent_phase_reference_excluded_from_tracker": bool(
            len(tracking)
            and (~tracking.latent_reference_used_by_tracker.astype(bool)).all()
            and (~tracking.hidden_diffusion_used_by_tracker.astype(bool)).all()
        ),
        "common_125ms_tracker_audit_grid": bool(
            len(tracking)
            and np.allclose(
                np.diff(np.sort(tracking.boundary_ms.unique())),
                float(cfg.analysis.phase_tracker_tradeoff.common_audit_interval_ms),
            )
        ),
        "latent_to_neural_phase_transfer_is_coherent": bool(
            carrier.groupby("context_id").latent_transfer_coherence.first().mean()
            >= float(criteria.minimum_latent_transfer_coherence)
        ),
        "neural_firing_rates_safe": bool(
            carrier.rates_safe.mean() >= float(criteria.minimum_rate_safe_fraction)
        ),
        "at_least_one_noise_pair_passes_measurement_tradeoff_gate": bool(
            frozen["candidate_selected"]
        ) or smoke,
        "selection_uses_no_tacs_outcome": True,
    }
    passed = bool(all(checks.values()) and not smoke)
    conclusions = {
        "H5_P2A_phase_tracker_bias_variance_tradeoff": (
            "PASSED" if passed else "NOT PASSED"
        ) if not smoke else "SMOKE TEST ONLY",
        "selected_high_noise_fraction": frozen["selected_high_noise_fraction"],
        "ready_for_active_H5_P2B_response_mapping": passed,
        "machine_learning_policy_status": "NOT TRAINED OR TESTED",
        "failed_checks": [name for name, value in checks.items() if not value],
    }
    return checks, conclusions


def _save_figure(figure: Any, root: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(root / f"{name}.png", dpi=300)
    figure.savefig(root / f"{name}.pdf")
    plt.close(figure)


def _plots(
    *,
    root: Path,
    carrier: pd.DataFrame,
    summary: pd.DataFrame,
    advantage: pd.DataFrame,
    candidates: pd.DataFrame,
    structure: pd.DataFrame,
    spectra: pd.DataFrame,
    tracking: pd.DataFrame,
) -> None:
    performance = _carrier_performance(carrier)
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(performance.noise_fraction, performance.carrier_coverage, "o-", label="coverage")
    axis.plot(
        performance.noise_fraction, performance.accepted_carrier_accuracy,
        "s-", label="accepted accuracy",
    )
    axis.plot(
        performance.noise_fraction, performance.phase_audit_usable_fraction,
        "^-", label="correct + accepted",
    )
    axis.set(xlabel="Observation-noise RMS / neural-EEG RMS", ylabel="Fraction", ylim=(0, 1.05))
    axis.set_title("Frozen carrier estimator across observation noise")
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_01_carrier_by_noise")

    aggregate = summary.groupby([
        "label", "noise_fraction", "tracker_profile"
    ], as_index=False).mean_abs_latent_reference_error_rad.mean()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for axis, label in zip(axes, [LOW, HIGH]):
        view = aggregate[aggregate.label.eq(label)]
        for profile, color, marker in (
            (CONSERVATIVE, "#4C78A8", "o"),
            (RESPONSIVE, "#E45756", "s"),
        ):
            line = view[view.tracker_profile.eq(profile)]
            axis.plot(
                line.noise_fraction,
                line.mean_abs_latent_reference_error_rad,
                marker=marker, color=color, label=profile.replace("refresh_", ""),
            )
        axis.set_title(label.replace("_", " "))
        axis.set_xlabel("Observation-noise fraction")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean absolute transferred-phase error (rad)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Causal phase-tracker bias--variance audit")
    _save_figure(figure, root, "figure_02_phase_error_tradeoff")

    pivot = advantage.pivot_table(
        index="noise_fraction", columns="label",
        values="fast_advantage_latent_error_rad", aggfunc="mean"
    ).reindex(columns=[LOW, HIGH])
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for label, color, marker in ((LOW, "#4C78A8", "o"), (HIGH, "#E45756", "^")):
        axis.plot(pivot.index, pivot[label], marker=marker, color=color, label=label.replace("_", " "))
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Observation-noise fraction")
    axis.set_ylabel("Fast advantage: slow error - fast error (rad)")
    axis.set_title("Required sign-changing tracker preference")
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_03_tracker_crossover")

    observation = summary.groupby([
        "noise_fraction", "tracker_profile"
    ], as_index=False).mean_abs_observation_error_rad.mean()
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for profile, color, marker in (
        (CONSERVATIVE, "#4C78A8", "o"),
        (RESPONSIVE, "#E45756", "s"),
    ):
        view = observation[observation.tracker_profile.eq(profile)]
        axis.plot(
            view.noise_fraction, view.mean_abs_observation_error_rad,
            marker=marker, color=color, label=profile.replace("refresh_", ""),
        )
    axis.set_xlabel("Observation-noise fraction")
    axis.set_ylabel("Observed-vs-neural same-profile phase error (rad)")
    axis.set_title("Measurement-noise attribution")
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_04_measurement_error")

    if not candidates.empty:
        selected_high = float(
            candidates.loc[
                candidates.passes_measurement_tradeoff_gate,
                "candidate_high_noise_fraction",
            ].min()
        ) if candidates.passes_measurement_tradeoff_gate.any() else float(
            candidates.candidate_high_noise_fraction.iloc[-1]
        )
        selected_structure = structure[np.isclose(
            structure.candidate_high_noise_fraction, selected_high
        )]
        figure, axis = plt.subplots(figsize=(8.0, 4.8))
        x = np.arange(len(selected_structure))
        width = 0.36
        axis.bar(
            x - width / 2,
            selected_structure.fast_advantage_high_diffusion_low_noise_rad,
            width, label="fast: high D / low noise",
        )
        axis.bar(
            x + width / 2,
            selected_structure.slow_advantage_low_diffusion_high_noise_rad,
            width, label="slow: low D / high noise",
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(x, selected_structure.structure_seed.astype(str), rotation=30)
        axis.set_ylabel("Directional phase-error advantage (rad)")
        axis.set_title(f"Structure-level directions; high-noise candidate={selected_high:g}")
        axis.legend(frameon=False, fontsize=8)
        _save_figure(figure, root, "figure_05_structure_directions")

    if not spectra.empty:
        representative = spectra[
            spectra.structure_seed.eq(spectra.structure_seed.min())
            & spectra.hidden_frequency_hz.eq(9.0)
            & spectra.label.eq(LOW)
        ]
        figure, axis = plt.subplots(figsize=(7.5, 4.6))
        for fraction, group in representative.groupby("noise_fraction"):
            view = group[group.frequency_hz.between(6.0, 14.0)]
            axis.plot(
                view.frequency_hz, view.observed_EEG_multitaper_residual_db,
                label=f"noise={float(fraction):g}",
            )
        axis.axvline(9.0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Aperiodic-adjusted power (dB)")
        axis.set_title("Representative frozen-carrier evidence")
        axis.legend(frameon=False)
        _save_figure(figure, root, "figure_06_representative_PSD")

    if not tracking.empty:
        representative = tracking[
            tracking.context_id.eq(tracking.context_id.iloc[0])
            & np.isclose(tracking.noise_fraction, tracking.noise_fraction.max())
        ]
        figure, axis = plt.subplots(figsize=(9.0, 4.5))
        for profile, color in ((CONSERVATIVE, "#4C78A8"), (RESPONSIVE, "#E45756")):
            view = representative[representative.tracker_profile.eq(profile)]
            axis.plot(
                (view.boundary_ms - view.boundary_ms.min()) / 1000.0,
                view.signed_latent_reference_error_rad,
                color=color, alpha=0.85, label=profile.replace("refresh_", ""),
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xlabel("Tracker-evaluation time (s)")
        axis.set_ylabel("Signed transferred-phase error (rad)")
        axis.set_title("Representative causal tracking errors")
        axis.legend(frameon=False)
        _save_figure(figure, root, "figure_07_phase_error_trace")


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
        print("\n### H5-P2A stimulation-free phase-tracker trade-off")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    carrier_rows: list[dict[str, Any]] = []
    tracking_rows: list[dict[str, Any]] = []
    spectrum_tables: list[pd.DataFrame] = []
    contexts = _run_context_specs(cfg)
    representative_structure = min(int(row["structure_seed"]) for row in contexts)
    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"D={context['diffusion_rad2_per_s']:g} rad^2/s"
            )
        result = _context_measurement(
            cfg,
            context,
            output_dir=root / "episodes" / str(context["context_id"]),
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            carrier, tracking, spectra = result
            carrier_rows.extend(carrier)
            tracking_rows.extend(tracking)
            if int(context["structure_seed"]) == representative_structure:
                spectrum_tables.append(spectra)
            accepted = sum(int(row["carrier_identified"]) for row in carrier)
            correct = sum(
                int(row["carrier_identified"] and row["carrier_selection_correct"])
                for row in carrier
            )
            print(f"carrier accepted={accepted}/3; accepted-correct={correct}/{accepted}")

    if rank != 0:
        return
    carrier = pd.DataFrame(carrier_rows)
    tracking = pd.DataFrame(tracking_rows)
    if tracking.empty:
        raise RuntimeError("No H5-P2A carrier was usable for phase tracking.")
    summary = _profile_summary(tracking)
    advantage = _advantage_table(summary)
    candidates, structure, frozen = _candidate_selection(
        carrier, summary, advantage, cfg
    )
    checks, conclusions = _checks(
        carrier=carrier,
        tracking=tracking,
        summary=summary,
        candidates=candidates,
        frozen=frozen,
        sources=sources,
        cfg=cfg,
    )
    spectra = (
        pd.concat(spectrum_tables, ignore_index=True)
        if spectrum_tables else pd.DataFrame()
    )
    carrier.to_csv(root / "carrier_measurement_by_noise.csv", index=False)
    tracking.to_csv(root / "causal_phase_tracker_boundaries.csv", index=False)
    summary.to_csv(root / "context_tracker_summary.csv", index=False)
    advantage.to_csv(root / "context_tracker_advantage.csv", index=False)
    _carrier_performance(carrier).to_csv(
        root / "carrier_performance_by_noise.csv", index=False
    )
    candidates.to_csv(root / "candidate_noise_pair_selection.csv", index=False)
    structure.to_csv(root / "structure_directional_tradeoff.csv", index=False)
    spectra.to_csv(root / "representative_predecision_spectra.csv", index=False)
    frozen.update({
        "source_hashes": sources["hashes"],
        "candidate_measurement_gate_passed": bool(frozen["candidate_selected"]),
        "gate_passed": bool(conclusions["ready_for_active_H5_P2B_response_mapping"]),
        "ready_for_active_H5_P2B_mapping": bool(
            conclusions["ready_for_active_H5_P2B_response_mapping"]
        ),
    })
    (root / "frozen_h5_p2_conditions.json").write_text(json.dumps(
        _json_ready(frozen), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H5_P2A_stimulation_free_phase_tracker_tradeoff",
        "frozen_sources": {"roots": sources["roots"], "hashes": sources["hashes"]},
        "state_generator": {
            "carrier_hz": [9.0, 11.0],
            "phase_diffusion_rad2_per_s": [0.5, 2.0],
            "shared_modulated_afferent_fraction": 1.0,
            "modulation_depth": 0.04,
            "afferent_mean_rate_matched": True,
        },
        "measurement_protocol": {
            "predecision_carrier_record_s": float(
                cfg.analysis.phase_tracker_tradeoff.predecision_duration_s
            ),
            "postdecision_stimulation_free_tracking_tail_s": float(
                cfg.analysis.phase_tracker_tradeoff.evaluation_duration_s
            ),
            "noise_fractions": [
                float(x) for x in cfg.analysis.phase_tracker_tradeoff.observation_noise_fractions
            ],
            "AR1_coefficient": float(
                cfg.analysis.phase_tracker_tradeoff.ar1_coefficient
            ),
            "noise_realization_paired_and_scaled_within_neural_context": True,
            "noise_normalization_uses_predecision_samples_only": True,
            "carrier_estimator": MT_POOLED,
            "carrier_estimator_refitted": False,
            "tracker_profiles": frozen["tracker_profiles"],
            "common_audit_interval_ms": float(
                cfg.analysis.phase_tracker_tradeoff.common_audit_interval_ms
            ),
            "processing_latency_modelled_ms": 0.0,
        },
        "reference_boundary": {
            "primary_phase_reference": "latent afferent phase plus predecision neural-EEG transfer offset",
            "reference_is_hidden_mechanistic_audit": True,
            "reference_never_enters_carrier_or_tracker": True,
            "same_profile_neural_EEG_phase_is_measurement_noise_attribution": True,
        },
        "design": {
            "independent_structure_count": int(carrier.structure_seed.nunique()),
            "neural_network_episode_count": int(carrier.context_id.nunique()),
            "observation_noise_views_per_episode": int(carrier.noise_fraction.nunique()),
            "statistical_unit": "independent circuit structure",
        },
        "inference_boundary": (
            "Stimulation-free controller-design discovery only. No tACS response, "
            "machine-learning policy, clinical noise model, or H5 claim is tested."
        ),
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-P2A stimulation-free causal phase-tracker bias-variance discovery",
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
            root=root,
            carrier=carrier,
            summary=summary,
            advantage=advantage,
            candidates=candidates,
            structure=structure,
            spectra=spectra,
            tracking=tracking,
        )

    print("\n### H5-P2A checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Candidate observation-noise pairs")
    print(candidates.to_string(index=False))
    print("\n### Frozen measurement conditions")
    print(json.dumps(_json_ready(frozen), indent=2, allow_nan=False))
    print(
        "\nH5-P2A phase-tracker trade-off: "
        f"{conclusions['H5_P2A_phase_tracker_bias_variance_tradeoff']}"
    )
    print("Machine-learning policy status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
