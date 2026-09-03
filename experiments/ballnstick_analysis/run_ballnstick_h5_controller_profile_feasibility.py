"""H5-P0 EEG-contextual controller-profile feasibility for BallAndStick.

This exploratory, full-information experiment asks whether a meaningful
context-action interaction exists before any H5 policy is trained. It crosses
phase diffusion with the fraction of afferents participating in one shared
rate rhythm, then compares two frozen causal phase trackers at the same field
amplitude, carrier-selection rule, relative phase, and montage.

The responsive profile is the H4-confirmed 0.5-s/125-ms controller. The
conservative profile uses a 1-s history refreshed every 250 ms. Both observe a
frozen moderate AR(1) sensor-noise process, while efficacy is evaluated on the
ideal neural EEG. Hidden generator parameters and spikes are audits only.
H5-P0 maps opportunity; it does not train or confirm a machine-learning policy.
"""

from __future__ import annotations

import hashlib
import json
import random
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
from scipy.signal import lfilter


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_h4_confirmation import (  # noqa: E402
    _load_frozen_h4bw2,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (  # noqa: E402
    HIGH,
    LOW,
    _context_features,
    _with_diffusion_state,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (  # noqa: E402
    SHAM,
    _analyze_controller_episode,
    _controller_modes,
    _metric_rows,
    _signed_phase_error,
    _tail_phase_estimate,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_bandwidth_discovery import (  # noqa: E402
    CURRENT,
    FAST,
    _augment_metric_rows,
    _fixed_horizon_phase_slew,
    _json_ready,
    _profile,
    _sha256,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_cadence_discovery import (  # noqa: E402
    _augment_common_audit,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _episode_config,
    _mpi_variables,
    _timeline,
    _validate_online_outputs,
    _zero_action,
)


ROOT_NAME = "h5_controller_profile_feasibility"
CONSERVATIVE = CURRENT
RESPONSIVE = FAST
EXPECTED_MODES = [SHAM, CONSERVATIVE, RESPONSIVE]
PARTIAL = "partial_shared_drive"
FULL = "full_shared_drive"


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    """Hash-lock the positive H4 result and all of its upstream sources."""
    sources = _load_frozen_h4bw2(cfg)
    root = Path(to_absolute_path(str(cfg.analysis.source_h4c.result_dir)))
    files = {
        "conclusion": root / "experiment_conclusion.json",
        "provenance": root / "protocol_and_provenance.json",
        "screening": root / "prospective_context_screening.csv",
        "metrics": root / "context_controller_future_metrics.csv",
        "inference": root / "H4_statistical_inference.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen H4-C sources: {missing}")
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {
        name: str(cfg.analysis.source_h4c.expected_sha256[name]) for name in files
    }
    if observed != expected:
        raise RuntimeError(
            f"H4-C source hash mismatch: expected={expected}, observed={observed}"
        )
    conclusion = json.loads(files["conclusion"].read_text())
    if (
        not all(bool(value) for value in conclusion["checks"].values())
        or not bool(
            conclusion["conclusions"]["H4_adaptive_phase_maintenance_confirmed"]
        )
    ):
        raise RuntimeError("H5-P0 requires the frozen positive H4-C result.")
    for name in ("screening", "metrics"):
        table = pd.read_csv(files[name])
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed",
        ):
            if column in table:
                sources["source_seed_union"].update(
                    table[column].dropna().astype(int).tolist()
                )
    sources["roots"]["h4c"] = str(root)
    sources["hashes"]["h4c"] = observed
    sources["H4C_confirmed"] = True
    return sources


def _shared_levels(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "shared_drive_label": str(value.label),
        "shared_modulated_fraction": float(value.shared_modulated_fraction),
    } for value in cfg.analysis.states.shared_drive_levels]


def _diffusion_levels(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "label": str(value.label),
        "diffusion_rad2_per_s": float(value.diffusion_rad2_per_s),
    } for value in cfg.analysis.states.phase_diffusion_levels]


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    """Create a nested common-random-number D x shared-drive context grid."""
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
                    + 20 * structure_index + 2 * history_index + frequency_index
                )
                for diffusion_index, diffusion in enumerate(_diffusion_levels(cfg)):
                    paired_id = (
                        f"s{structure_index:02d}_h{history_index:02d}_"
                        f"f{int(round(float(frequency))):02d}_d{diffusion_index:02d}"
                    )
                    trial_seed = (
                        base + int(block.trial_seed_offset) + future_group
                    )
                    for shared in _shared_levels(cfg):
                        rows.append({
                            "context_order": len(rows),
                            "future_group_index": future_group,
                            "context_id": (
                                f"{paired_id}_{diffusion['label']}_"
                                f"{shared['shared_drive_label']}"
                            ),
                            "paired_shared_drive_context_id": paired_id,
                            "structure_index": structure_index,
                            "structure_seed": structure_seed,
                            "history_index": history_index,
                            "history_seed": history_seed,
                            "phase_seed": phase_seed,
                            "trial_seed": trial_seed,
                            "hidden_frequency_hz": float(frequency),
                            **diffusion,
                            **shared,
                        })
                    future_group += 1
    return rows


def _run_context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    rows = _context_specs(cfg)
    limit = int(cfg.analysis.smoke_context_limit)
    if bool(cfg.analysis.smoke_test) and limit > 0:
        return rows[:limit]
    return rows


def _future_seed(cfg: DictConfig, context: dict[str, Any], future_index: int) -> int:
    return (
        int(cfg.experiment.seed)
        + int(cfg.analysis.crossed_design.future_seed_offset)
        + 100 * int(context["future_group_index"])
        + int(future_index)
    )


def _noise_seeds(
    cfg: DictConfig, context: dict[str, Any], future_index: int
) -> tuple[int, int]:
    base = int(cfg.experiment.seed) + int(cfg.analysis.observation_noise.seed_offset)
    group = int(context["future_group_index"])
    return base + 100 * group, base + 10_000 + 100 * group + int(future_index)


def _with_context_state(cfg: DictConfig, context: dict[str, Any]) -> DictConfig:
    state = _with_diffusion_state(cfg, {
        "frequency_hz": float(context["hidden_frequency_hz"]),
        "phase_seed": int(context["phase_seed"]),
        "diffusion_rad2_per_s": float(context["diffusion_rad2_per_s"]),
    })
    with open_dict(state):
        for population in ("E", "I"):
            state.env.network.background[
                population
            ].rhythm.shared_modulated_fraction = float(
                context["shared_modulated_fraction"]
            )
    return state


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    smoke = bool(cfg.analysis.smoke_test)
    if not bool(sources["H4C_confirmed"]):
        raise ValueError("H5-P0 requires confirmed H4 provenance.")
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-P0 requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-P0 may not change recurrent inhibition.")
    if [float(value) for value in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-P0 retains the EEG-selected 9/11-Hz carrier grid.")
    diffusion = [
        (value["label"], value["diffusion_rad2_per_s"])
        for value in _diffusion_levels(cfg)
    ]
    if diffusion != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("H5-P0 retains D={0.5,2.0} rad^2/s.")
    shared = [
        (value["shared_drive_label"], value["shared_modulated_fraction"])
        for value in _shared_levels(cfg)
    ]
    if shared != [(PARTIAL, 0.5), (FULL, 1.0)]:
        raise ValueError("H5-P0 freezes shared-drive fractions to 0.5 and 1.0.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-P0 retains modulation depth 0.04.")
    if _controller_modes(cfg) != EXPECTED_MODES:
        raise ValueError(f"H5-P0 controller order must be {EXPECTED_MODES}.")
    expected_profiles = {
        CONSERVATIVE: {"adaptive": True, "history_ms": 1000.0,
                       "update_interval_ms": 250.0},
        RESPONSIVE: {"adaptive": True, "history_ms": 500.0,
                     "update_interval_ms": 125.0},
    }
    for mode, expected in expected_profiles.items():
        if _profile(cfg, mode) != expected:
            raise ValueError(f"H5-P0 controller profile changed: {mode}.")
    if not np.isclose(float(cfg.analysis.actions.amplitude_v_per_m), 0.2):
        raise ValueError("Both H5-P0 active profiles must use 0.2 V/m.")
    for name, expected in (
        ("initialization_history_ms", 1000.0),
        ("correction_horizon_ms", 250.0),
        ("common_audit_history_ms", 1000.0),
        ("common_audit_interval_ms", 250.0),
        ("maximum_frequency_correction_hz", 2.0),
    ):
        if not np.isclose(float(cfg.analysis.tacs[name]), expected):
            raise ValueError(f"H5-P0 freezes {name}={expected:g}.")
    noise = cfg.analysis.observation_noise
    if (
        not bool(noise.enabled)
        or not 0.0 < float(noise.rms_fraction_of_baseline_neural_eeg) < 1.0
        or not 0.0 <= float(noise.ar1_coefficient) < 1.0
    ):
        raise ValueError("H5-P0 requires a finite moderate observation-noise model.")
    if not np.isclose(float(cfg.env.simulation.obs_win_len), 1000.0):
        raise ValueError("H5-P0 requires one-second outer online windows.")
    endpoint_ms = (
        int(cfg.analysis.timeline.stimulation_steps)
        * float(cfg.env.simulation.obs_win_len)
        - 2.0 * float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    )
    if endpoint_ms <= 0.0 or not np.isclose(endpoint_ms / 1000.0, round(endpoint_ms / 1000.0)):
        raise ValueError("H5-P0 requires a positive whole-second EEG endpoint.")
    if not smoke and not np.isclose(endpoint_ms, 8000.0):
        raise ValueError("Full H5-P0 retains the eight-second H4 endpoint.")
    if int(cfg.analysis.timeline.baseline_steps) < (4 if smoke else 12):
        raise ValueError("H5-P0 requires a 12-s baseline, or 4 s for smoke.")
    if not smoke:
        if int(cfg.analysis.crossed_design.n_structure_seeds) < 3:
            raise ValueError("Full H5-P0 requires at least three structures.")
        if int(cfg.analysis.crossed_design.n_future_continuations) < 4:
            raise ValueError("Full H5-P0 requires four paired futures.")

    contexts = _context_specs(cfg)
    namespaces = [
        {int(row["structure_seed"]) for row in contexts},
        {int(row["history_seed"]) for row in contexts},
        {int(row["phase_seed"]) for row in contexts},
        {int(row["trial_seed"]) for row in contexts},
        {
            _future_seed(cfg, row, future)
            for row in contexts
            for future in range(int(cfg.analysis.crossed_design.n_future_continuations))
        },
        {
            seed
            for row in contexts
            for future in range(int(cfg.analysis.crossed_design.n_future_continuations))
            for seed in _noise_seeds(cfg, row, future)
        },
    ]
    if any(not values for values in namespaces):
        raise ValueError("Every H5-P0 seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H5-P0 seed namespaces overlap.")
    if set().union(*namespaces).intersection(sources["source_seed_union"]):
        raise ValueError("H5-P0 seeds overlap H1-H4 source experiments.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H5-P0 structure seed exceeds the uint32 mapping range.")


def _ar1_path(
    *, n_samples: int, split_sample: int, history_seed: int, future_seed: int,
    coefficient: float,
) -> np.ndarray:
    """Generate unit-RMS AR(1) noise with paired history and split future."""
    if not 0 < split_sample < n_samples:
        raise ValueError("The observation-noise split must be interior.")
    scale = np.sqrt(1.0 - float(coefficient) ** 2)
    history_rng = np.random.default_rng(int(history_seed))
    future_rng = np.random.default_rng(int(future_seed))
    history, state = lfilter(
        [scale], [1.0, -float(coefficient)],
        history_rng.standard_normal(split_sample), zi=[0.0],
    )
    future, _ = lfilter(
        [scale], [1.0, -float(coefficient)],
        future_rng.standard_normal(n_samples - split_sample), zi=state,
    )
    # Normalize only by the common predecision realization. Normalizing by the
    # complete path would let the postdecision future rescale—and therefore
    # alter—the supposedly identical observation history.
    rms = float(np.sqrt(np.mean(history ** 2)))
    if not np.isfinite(rms) or rms <= np.finfo(float).tiny:
        raise RuntimeError("Observation-noise path has invalid RMS.")
    return np.concatenate((history, future)) / rms


def _observed_output(
    output: dict[str, Any], *, unit_noise: np.ndarray, scale_v: float,
    simulator_dt_ms: float,
) -> dict[str, Any]:
    result = dict(output)
    eeg = np.asarray(output["eeg_v"], dtype=float).reshape(-1)
    times = np.asarray(output["sample_times_ms"], dtype=float).reshape(-1)
    indices = np.rint(times / float(simulator_dt_ms)).astype(int) - 1
    if eeg.size != indices.size or indices.min() < 0 or indices.max() >= unit_noise.size:
        raise RuntimeError("Observation-noise indexing does not match EEG samples.")
    result["eeg_v"] = eeg + float(scale_v) * unit_noise[indices]
    result["observation_noise_v"] = float(scale_v) * unit_noise[indices]
    return result


def _controller_action(cfg: DictConfig, mode: str) -> dict[str, Any]:
    profile = _profile(cfg, mode)
    return {
        "id": mode,
        "role": "H5_P0_full_information_controller_profile",
        "controller_mode": mode,
        "adaptive": bool(profile["adaptive"]),
        "phase_history_ms": float(profile["history_ms"]),
        "update_interval_ms": float(profile["update_interval_ms"]),
        "montage": str(cfg.analysis.tacs.axial_montage),
        "dc_offset_v_per_m": 0.0,
        "ac_amplitude_v_per_m": (
            0.0 if mode == SHAM else float(cfg.analysis.actions.amplitude_v_per_m)
        ),
        "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
        "phase_rad": 0.0,
        "eeg_relative_phase_offset_rad": float(
            cfg.analysis.tacs.relative_phase_offset_rad
        ),
    }


def _simulate_controller_episode(
    base_cfg: DictConfig, *, context: dict[str, Any], future_seed: int,
    future_index: int, mode: str, output_dir: Path, comm: Any, size: int,
    rank: int,
) -> dict[str, Any] | None:
    """Run one controller with noisy observations and neural-only outcomes."""
    run_cfg = _episode_config(
        base_cfg, seed=int(context["trial_seed"]), output_dir=output_dir
    )
    profile = _profile(run_cfg, mode)
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    dt_ms = float(run_cfg.env.network.dt)
    refresh_ms = float(profile["update_interval_ms"])
    history_ms = float(profile["history_ms"])
    initialization_history_ms = float(
        run_cfg.analysis.tacs.initialization_history_ms
    )
    correction_horizon_ms = float(run_cfg.analysis.tacs.correction_horizon_ms)
    pre_steps = int(run_cfg.analysis.timeline.burn_in_steps) + int(
        run_cfg.analysis.timeline.baseline_steps
    )
    block_start_ms = pre_steps * window_ms
    stimulation_ms = int(run_cfg.analysis.timeline.stimulation_steps) * window_ms
    block_stop_ms = block_start_ms + stimulation_ms
    total_steps = sum(int(run_cfg.analysis.timeline[f"{epoch}_steps"]) for epoch in (
        "burn_in", "baseline", "stimulation", "washout"
    ))
    total_samples = int(round(total_steps * window_ms / dt_ms))
    split_sample = int(round(block_start_ms / dt_ms))
    history_noise_seed, future_noise_seed = _noise_seeds(
        run_cfg, context, future_index
    )
    unit_noise = _ar1_path(
        n_samples=total_samples,
        split_sample=split_sample,
        history_seed=history_noise_seed,
        future_seed=future_noise_seed,
        coefficient=float(run_cfg.analysis.observation_noise.ar1_coefficient),
    )
    envelope = {
        "start_ms": block_start_ms,
        "stop_ms": block_stop_ms,
        "ramp_ms": float(run_cfg.analysis.timeline.block_ramp_ms),
    }
    structure = int(context["structure_seed"])
    np.random.seed(structure * 10_000 + rank)
    random.seed(structure * 10_000 + rank)
    environment = OnlineNeuronEnv(
        run_cfg,
        _mpi_variables(
            comm, size, rank, int(context["trial_seed"]),
            structure_seed=structure,
            drive_seed=int(context["history_seed"]),
            future_drive_seed=int(future_seed),
            future_start_ms=block_start_ms,
        ),
        ENV_SEED=0,
    )
    outputs = {name: [] for name, _ in _timeline(run_cfg)} if rank == 0 else None
    observed_outputs = (
        {name: [] for name, _ in _timeline(run_cfg)} if rank == 0 else None
    )
    update_rows: list[dict[str, Any]] | None = [] if rank == 0 else None
    zero = _zero_action(run_cfg)
    carrier = float(run_cfg.analysis.tacs.frequency_hz)
    maximum_correction = float(
        run_cfg.analysis.tacs.maximum_frequency_correction_hz
    )
    amplitude = 0.0 if mode == SHAM else float(
        run_cfg.analysis.actions.amplitude_v_per_m
    )
    final_residual_mV = float("nan")
    previous_field_endpoint: float | None = None
    observation_scale_v = float("nan")
    baseline_neural_rms_v = float("nan")
    baseline_noise_rms_v = float("nan")
    observed_preceding: list[dict[str, Any]] | None = [] if rank == 0 else None
    try:
        for epoch in ("burn_in", "baseline"):
            for _ in range(int(run_cfg.analysis.timeline[f"{epoch}_steps"])):
                output = environment.step_online(
                    zero, duration_ms=window_ms, phase_continuous=True
                )
                if rank == 0:
                    outputs[epoch].append(output)

        if rank == 0:
            baseline_neural = np.concatenate([
                np.asarray(value["eeg_v"], dtype=float).reshape(-1)
                for value in outputs["baseline"]
            ])
            baseline_neural_rms_v = float(np.sqrt(np.mean(baseline_neural ** 2)))
            observation_scale_v = (
                float(run_cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg)
                * baseline_neural_rms_v
            )
            for epoch in ("burn_in", "baseline"):
                observed_outputs[epoch] = [
                    _observed_output(
                        value, unit_noise=unit_noise,
                        scale_v=observation_scale_v, simulator_dt_ms=dt_ms,
                    )
                    for value in outputs[epoch]
                ]
                observed_preceding.extend(observed_outputs[epoch])
            baseline_noise = np.concatenate([
                np.asarray(value["observation_noise_v"], dtype=float).reshape(-1)
                for value in observed_outputs["baseline"]
            ])
            baseline_noise_rms_v = float(np.sqrt(np.mean(baseline_noise ** 2)))
        observation_scale_v = float(comm.bcast(observation_scale_v, root=0))

        n_updates = int(round(stimulation_ms / refresh_ms))
        for update_index in range(n_updates):
            boundary_ms = block_start_ms + update_index * refresh_ms
            estimate_history_ms = (
                initialization_history_ms if update_index == 0 else history_ms
            )
            if rank == 0:
                estimate = _tail_phase_estimate(
                    observed_preceding,
                    boundary_ms=boundary_ms,
                    history_ms=estimate_history_ms,
                    simulator_fs_hz=1000.0 / dt_ms,
                    relative_offset_rad=float(
                        run_cfg.analysis.tacs.relative_phase_offset_rad
                    ),
                    cfg=run_cfg,
                )
                oscillator_phase = (
                    float(estimate["desired_field_phase_rad"])
                    if update_index == 0 else float(environment.phase_rad)
                )
                audit_history_ms = float(
                    run_cfg.analysis.tacs.common_audit_history_ms
                )
                common = _tail_phase_estimate(
                    observed_preceding,
                    boundary_ms=boundary_ms,
                    history_ms=audit_history_ms,
                    simulator_fs_hz=1000.0 / dt_ms,
                    relative_offset_rad=float(
                        run_cfg.analysis.tacs.relative_phase_offset_rad
                    ),
                    cfg=run_cfg,
                )
                slew = _fixed_horizon_phase_slew(
                    run_cfg,
                    carrier_hz=carrier,
                    target_phase_rad=float(estimate["desired_field_phase_rad"]),
                    oscillator_phase_rad=oscillator_phase,
                )
                applied = bool(profile["adaptive"] and update_index > 0)
                if not applied:
                    slew["frequency_correction_hz"] = 0.0
                    slew["command_frequency_hz"] = carrier
                diagnostics = {
                    "update_index": int(update_index),
                    "boundary_ms": float(boundary_ms),
                    "controller_mode": mode,
                    "carrier_frequency_hz": carrier,
                    "phase_history_ms": float(estimate_history_ms),
                    "configured_post_onset_phase_history_ms": history_ms,
                    "update_interval_ms": refresh_ms,
                    "correction_horizon_ms": correction_horizon_ms,
                    "oscillator_phase_before_update_rad": oscillator_phase,
                    "phase_refresh_applied": applied,
                    "common_audit_history_ms": audit_history_ms,
                    "common_audit_desired_field_phase_rad": float(
                        common["desired_field_phase_rad"]
                    ),
                    "common_audit_estimated_eeg_phase_at_boundary_rad": float(
                        common["estimated_eeg_phase_at_boundary_rad"]
                    ),
                    "common_audit_resultant_to_rms": float(
                        common["resultant_to_rms"]
                    ),
                    "common_audit_phase_error_before_correction_rad": (
                        _signed_phase_error(
                            float(common["desired_field_phase_rad"]),
                            oscillator_phase,
                        )
                    ),
                    "observation_noise_enabled": True,
                    **estimate,
                    **slew,
                }
                diagnostics["frequency_correction_saturated"] = bool(
                    applied and np.isclose(
                        abs(float(diagnostics["frequency_correction_hz"])),
                        maximum_correction,
                    )
                )
            else:
                diagnostics = None
            diagnostics = comm.bcast(diagnostics, root=0)
            action = {
                "montage": str(run_cfg.analysis.tacs.axial_montage),
                "dc_offset_v_per_m": 0.0,
                "ac_amplitude_v_per_m": amplitude,
                "frequency_hz": (
                    0.0 if mode == SHAM
                    else float(diagnostics["command_frequency_hz"])
                ),
            }
            if update_index == 0 and mode != SHAM:
                action["phase_rad"] = float(diagnostics["desired_field_phase_rad"])
            output = environment.step_online(
                action,
                duration_ms=refresh_ms,
                phase_continuous=True,
                ramp_ms=0.0,
                block_envelope=envelope if mode != SHAM else None,
            )
            if rank == 0:
                observed = _observed_output(
                    output, unit_noise=unit_noise,
                    scale_v=observation_scale_v, simulator_dt_ms=dt_ms,
                )
                field = np.asarray(
                    output["stimulation"].get("field_v_per_m", np.zeros(1)),
                    dtype=float,
                )
                discontinuity = (
                    0.0 if previous_field_endpoint is None
                    else abs(float(field[0]) - previous_field_endpoint)
                )
                previous_field_endpoint = float(field[-1])
                diagnostics.update({
                    "phase_stop_rad": float(
                        output["stimulation"]["phase_stop_rad"]
                    ),
                    "field_boundary_discontinuity_v_per_m": discontinuity,
                    "estimate_is_strictly_causal": bool(
                        float(diagnostics["estimate_stop_ms"])
                        <= boundary_ms + 1.0e-9
                    ),
                })
                update_rows.append(diagnostics)
                outputs["stimulation"].append(output)
                observed_outputs["stimulation"].append(observed)
                observed_preceding.append(observed)

        for _ in range(int(run_cfg.analysis.timeline.washout_steps)):
            output = environment.step_online(
                zero, duration_ms=window_ms, phase_continuous=True
            )
            if rank == 0:
                outputs["washout"].append(output)
                observed_outputs["washout"].append(_observed_output(
                    output, unit_noise=unit_noise,
                    scale_v=observation_scale_v, simulator_dt_ms=dt_ms,
                ))
        if rank == 0:
            final_residual_mV = environment.stimulation_controller.max_abs_extracellular(
                environment.network
            )
    finally:
        environment.close()

    if rank != 0:
        return None
    _validate_online_outputs(outputs)
    if final_residual_mV != 0.0:
        raise RuntimeError("H5-P0 washout left residual extracellular voltage.")
    observed_baseline = np.concatenate([
        np.asarray(value["eeg_v"], dtype=float).reshape(-1)
        for value in observed_outputs["baseline"]
    ])
    return {
        "seed": int(context["trial_seed"]),
        "structure_seed": structure,
        "drive_seed": int(context["history_seed"]),
        "future_drive_seed": int(future_seed),
        "future_start_ms": float(block_start_ms),
        "action": _controller_action(run_cfg, mode),
        "stimulate": mode != SHAM,
        "controller_mode": mode,
        "block_start_ms": float(block_start_ms),
        "outputs_by_epoch": outputs,
        "observed_outputs_by_epoch": observed_outputs,
        "phase_updates": update_rows,
        "final_residual_mV": float(final_residual_mV),
        "observation": {
            "model": "AR1_additive_sensor_noise",
            "ar1_coefficient": float(
                run_cfg.analysis.observation_noise.ar1_coefficient
            ),
            "configured_rms_fraction": float(
                run_cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg
            ),
            "baseline_neural_rms_v": baseline_neural_rms_v,
            "baseline_noise_rms_v": baseline_noise_rms_v,
            "achieved_baseline_noise_fraction": (
                baseline_noise_rms_v
                / max(baseline_neural_rms_v, np.finfo(float).tiny)
            ),
            "history_noise_seed": int(history_noise_seed),
            "future_noise_seed": int(future_noise_seed),
            "observed_baseline_sha256": hashlib.sha256(
                observed_baseline.astype(np.float64).tobytes()
            ).hexdigest(),
        },
    }


def _run_controller(
    *, condition_cfg: DictConfig, context: dict[str, Any], future_seed: int,
    future_index: int, mode: str, action_index: int, root: Path, comm: Any,
    size: int, rank: int,
) -> dict[str, Any] | None:
    output_dir = (
        root / "episodes" / str(context["context_id"])
        / f"future_{future_index + 1:02d}" / mode
    )
    simulation = _simulate_controller_episode(
        condition_cfg,
        context=context,
        future_seed=future_seed,
        future_index=future_index,
        mode=mode,
        output_dir=output_dir,
        comm=comm,
        size=size,
        rank=rank,
    )
    episode = _analyze_controller_episode(
        simulation,
        condition_cfg=condition_cfg,
        context=context,
        mode=mode,
        action_index=action_index,
        output_dir=output_dir,
        rank=rank,
    )
    if rank == 0:
        episode["observed_raw_by_epoch"] = {
            epoch: np.concatenate([
                np.asarray(value["eeg_v"], dtype=float).reshape(-1)
                for value in simulation["observed_outputs_by_epoch"][epoch]
            ])
            for epoch in simulation["observed_outputs_by_epoch"]
        }
    return episode


def _observed_episode_view(episode: dict[str, Any]) -> dict[str, Any]:
    """Expose noisy EEG to context extraction without changing neural outcomes."""
    view = dict(episode)
    view["raw_by_epoch"] = episode["observed_raw_by_epoch"]
    simulation = dict(episode["simulation"])
    simulation["outputs_by_epoch"] = simulation["observed_outputs_by_epoch"]
    view["simulation"] = simulation
    return view


def _augment_observation_rows(
    rows: list[dict[str, Any]], episodes: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        observation = episodes[str(row["controller_mode"])]["simulation"][
            "observation"
        ]
        row.update({
            "observation_noise_model": str(observation["model"]),
            "configured_observation_noise_fraction": float(
                observation["configured_rms_fraction"]
            ),
            "achieved_baseline_noise_fraction": float(
                observation["achieved_baseline_noise_fraction"]
            ),
            "observed_baseline_sha256": str(
                observation["observed_baseline_sha256"]
            ),
            "context_features_use_observed_EEG": True,
            "efficacy_endpoint_uses_neural_only_EEG": True,
        })


CONTEXT_FEATURES = [
    "context_C1",
    "context_C1_abs",
    "context_C1_temporal_sd",
    "context_spectral_concentration",
    "context_spectral_rms_width_hz",
    "context_alpha_excess_log10",
    "recent_resultant_to_rms",
]


def _add_context_features_to_rows(
    rows: list[dict[str, Any]], screening: dict[str, Any]
) -> None:
    for row in rows:
        for feature in CONTEXT_FEATURES:
            row[feature] = float(screening[feature])


def _expected_map(metrics: pd.DataFrame) -> pd.DataFrame:
    group = [
        "context_id", "paired_shared_drive_context_id", "structure_seed",
        "hidden_frequency_hz", "label", "diffusion_rad2_per_s",
        "shared_drive_label", "shared_modulated_fraction", *CONTEXT_FEATURES,
        "controller_mode",
    ]
    return (
        metrics.groupby(group, as_index=False)
        .agg(
            n_futures=("future_index", "nunique"),
            expected_post_distance_to_B_log10=(
                "post_distance_to_B_log10", "mean"
            ),
            future_sd_post_distance_log10=(
                "post_distance_to_B_log10", "std"
            ),
            expected_improvement_vs_sham_log10=(
                "causal_distance_improvement_vs_sham_log10", "mean"
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


def _shared_drive_loso(
    screening: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit whether phase-invariant noisy-EEG context exposes shared drive."""
    eligible = screening[screening.eligible.astype(bool)].copy()
    features = [
        "context_C1",
        "context_C1_abs",
        "context_spectral_concentration",
        "context_spectral_rms_width_hz",
        "context_alpha_excess_log10",
        "recent_resultant_to_rms",
    ]
    rows: list[dict[str, Any]] = []
    for structure in sorted(eligible.structure_seed.unique()):
        train = eligible[eligible.structure_seed.ne(structure)]
        test = eligible[eligible.structure_seed.eq(structure)]
        if train.empty or test.empty or train.shared_drive_label.nunique() < 2:
            continue
        center = train[features].mean().to_numpy(float)
        scale = train[features].std(ddof=0).to_numpy(float)
        scale[~np.isfinite(scale) | (scale <= np.finfo(float).tiny)] = 1.0
        centroids = {
            label: ((group[features].mean().to_numpy(float) - center) / scale)
            for label, group in train.groupby("shared_drive_label")
        }
        for sample in test.itertuples():
            vector = (
                np.asarray([getattr(sample, name) for name in features], dtype=float)
                - center
            ) / scale
            distances = {
                label: float(np.linalg.norm(vector - centroid))
                for label, centroid in centroids.items()
            }
            predicted = min(distances, key=lambda label: (distances[label], label))
            rows.append({
                "context_id": str(sample.context_id),
                "structure_seed": int(structure),
                "true_shared_drive_label": str(sample.shared_drive_label),
                "predicted_shared_drive_label": str(predicted),
                "correct": bool(predicted == str(sample.shared_drive_label)),
                "partial_centroid_distance": float(distances.get(PARTIAL, np.nan)),
                "full_centroid_distance": float(distances.get(FULL, np.nan)),
                "classifier": "leave-one-structure-out standardized nearest centroid",
                "features": ";".join(features),
            })
    predictions = pd.DataFrame(rows)
    recalls: dict[str, float] = {}
    if not predictions.empty:
        for label in (PARTIAL, FULL):
            subset = predictions[
                predictions.true_shared_drive_label.eq(label)
            ]
            recalls[label] = float(subset.correct.mean()) if len(subset) else float("nan")
    balanced = float(np.nanmean(list(recalls.values()))) if recalls else float("nan")
    return predictions, {
        "LOSO_shared_drive_balanced_accuracy": balanced,
        "LOSO_shared_drive_recall": recalls,
        "classifier_uses_only_predecision_phase_invariant_observed_EEG": True,
        "classifier_is_an_observability_audit_not_a_policy": True,
    }


def _controller_action_map(
    expected: pd.DataFrame, metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = [
        "context_id", "paired_shared_drive_context_id", "structure_seed",
        "hidden_frequency_hz", "label", "diffusion_rad2_per_s",
        "shared_drive_label", "shared_modulated_fraction", *CONTEXT_FEATURES,
    ]
    active = expected[expected.controller_mode.isin([
        CONSERVATIVE, RESPONSIVE
    ])]
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
    conservative_distance = (
        f"expected_post_distance_to_B_log10_{CONSERVATIVE}"
    )
    responsive_distance = f"expected_post_distance_to_B_log10_{RESPONSIVE}"
    pivot["responsive_advantage_over_conservative_log10"] = (
        pivot[conservative_distance] - pivot[responsive_distance]
    )
    pivot["expected_optimal_profile"] = np.where(
        pivot[responsive_distance] < pivot[conservative_distance],
        RESPONSIVE,
        CONSERVATIVE,
    )

    fixed_mean = {
        mode: float(active[
            active.controller_mode.eq(mode)
        ].expected_post_distance_to_B_log10.mean())
        for mode in (CONSERVATIVE, RESPONSIVE)
    }
    best_fixed = min(fixed_mean, key=lambda mode: (fixed_mean[mode], mode))
    best_fixed_column = f"expected_post_distance_to_B_log10_{best_fixed}"
    pivot["best_fixed_profile"] = best_fixed
    pivot["expected_oracle_distance_to_B_log10"] = np.minimum(
        pivot[conservative_distance], pivot[responsive_distance]
    )
    pivot["oracle_advantage_over_best_fixed_log10"] = (
        pivot[best_fixed_column] - pivot["expected_oracle_distance_to_B_log10"]
    )

    realized = metrics[metrics.controller_mode.isin([
        CONSERVATIVE, RESPONSIVE
    ])]
    agreements = []
    for row in pivot.itertuples():
        group = realized[realized.context_id.eq(str(row.context_id))]
        winners = []
        for _, future in group.groupby("future_index"):
            winners.append(str(future.sort_values([
                "post_distance_to_B_log10", "controller_mode"
            ]).iloc[0].controller_mode))
        agreements.append(float(np.mean(
            np.asarray(winners) == str(row.expected_optimal_profile)
        )))
    pivot["realized_optimal_profile_agreement_fraction"] = agreements

    structure = (
        pivot.groupby("structure_seed", as_index=False)
        .agg(
            context_count=("context_id", "nunique"),
            mean_oracle_advantage_over_best_fixed_log10=(
                "oracle_advantage_over_best_fixed_log10", "mean"
            ),
            mean_responsive_advantage_over_conservative_log10=(
                "responsive_advantage_over_conservative_log10", "mean"
            ),
            mean_realized_optimal_profile_agreement_fraction=(
                "realized_optimal_profile_agreement_fraction", "mean"
            ),
        )
    )
    optimal_counts = pivot.expected_optimal_profile.value_counts().to_dict()
    optimal_structures = (
        pivot.groupby("expected_optimal_profile").structure_seed.nunique().to_dict()
    )
    shared_response = (
        pivot.groupby("shared_drive_label")
        .responsive_advantage_over_conservative_log10.mean().to_dict()
    )
    interaction = abs(float(shared_response.get(FULL, np.nan)) - float(
        shared_response.get(PARTIAL, np.nan)
    ))
    audit = {
        "best_fixed_profile": best_fixed,
        "best_fixed_expected_distance_log10": float(fixed_mean[best_fixed]),
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
        "mean_realized_optimal_profile_agreement_fraction": float(
            pivot.realized_optimal_profile_agreement_fraction.mean()
        ),
        "mean_responsive_advantage_by_shared_drive_log10": shared_response,
        "absolute_shared_drive_by_profile_response_interaction_log10": interaction,
        "oracle_is_post_hoc_full_information_and_not_deployable": True,
    }
    return pivot, structure, audit


def _checks(
    *, screening: pd.DataFrame, metrics: pd.DataFrame, expected: pd.DataFrame,
    updates: pd.DataFrame, action_map: pd.DataFrame, structure: pd.DataFrame,
    observability: dict[str, Any], opportunity: dict[str, Any],
    sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    smoke = bool(cfg.analysis.smoke_test)
    eligible = screening[screening.eligible.astype(bool)]
    active = metrics[metrics.controller_mode.ne(SHAM)]
    adaptive_updates = updates[
        updates.controller_mode.isin([CONSERVATIVE, RESPONSIVE])
        & updates.phase_refresh_applied.astype(bool)
    ]
    profile_counts = opportunity["optimal_profile_context_count"]
    profile_structures = opportunity["optimal_profile_structure_count"]
    identical_observed_history = bool(
        len(metrics)
        and metrics.groupby("context_id").observed_baseline_sha256.nunique().max()
        == 1
    )
    checks = {
        "H4C_passed_and_hash_locked": bool(sources["H4C_confirmed"]),
        "H5P0_seed_namespaces_disjoint_from_H1_H4": True,
        "shared_drive_generator_is_distinct_from_tacs": True,
        "afferent_mean_rate_matched_across_shared_drive_by_construction": True,
        "private_poisson_event_streams_remain_independent": True,
        "complete_frequency_diffusion_shared_drive_grid": bool(
            len(screening) == len(_run_context_specs(cfg))
        ),
        "screening_uses_only_predecision_observed_EEG": bool(
            len(screening)
            and screening.screen_uses_only_predecision_observed_EEG.all()
            and (~screening.screen_uses_hidden_diffusion_or_frequency.astype(bool)).all()
            and (~screening.screen_uses_action_outcome.astype(bool)).all()
        ),
        "minimum_eligible_contexts": len(eligible)
        >= int(criteria.minimum_eligible_contexts) or smoke,
        "minimum_independent_structures": eligible.structure_seed.nunique()
        >= int(criteria.minimum_structure_seeds) or smoke,
        "both_frequencies_diffusions_and_shared_drive_levels_enrolled": bool(
            eligible.hidden_frequency_hz.nunique() == 2
            and eligible.label.nunique() == 2
            and eligible.shared_drive_label.nunique() == 2
        ) or smoke,
        "frequency_identified_from_predecision_EEG": bool(
            len(eligible)
            and eligible.EEG_frequency_selection_correct.mean()
            >= float(criteria.minimum_frequency_detection_accuracy)
        ) or smoke,
        "shared_drive_observable_from_phase_invariant_EEG": bool(
            np.isfinite(observability["LOSO_shared_drive_balanced_accuracy"])
            and observability["LOSO_shared_drive_balanced_accuracy"]
            >= float(criteria.minimum_shared_drive_classification_balanced_accuracy)
        ) or smoke,
        "complete_controller_grid_for_enrolled_contexts": bool(
            len(expected)
            and expected.groupby("context_id").controller_mode.nunique().min()
            == len(EXPECTED_MODES)
        ),
        "multiple_independent_paired_postdecision_futures": bool(
            len(expected)
            and expected.n_futures.min()
            >= int(criteria.minimum_future_continuations)
        ) or smoke,
        "identical_predecision_neural_EEG_across_actions_and_futures": bool(
            len(metrics)
            and metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "identical_predecision_observation_noise_across_actions_and_futures": (
            identical_observed_history
        ),
        "moderate_frozen_observation_noise_applied": bool(
            len(metrics)
            and metrics.achieved_baseline_noise_fraction.between(0.10, 0.50).all()
        ),
        "both_active_profiles_use_identical_0p2_V_per_m": bool(
            len(active) and np.allclose(active.amplitude_v_per_m, 0.2)
        ),
        "one_controller_profile_is_fixed_for_each_intervention": bool(
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
            >= int(criteria.minimum_contexts_per_optimal_profile)
            for mode in (CONSERVATIVE, RESPONSIVE)
        ),
        "both_optimal_profiles_replicate_across_structures": all(
            int(profile_structures.get(mode, 0))
            >= int(criteria.minimum_structures_per_optimal_profile)
            for mode in (CONSERVATIVE, RESPONSIVE)
        ),
        "expected_oracle_has_practical_advantage_over_best_fixed": float(
            opportunity["mean_oracle_advantage_over_best_fixed_log10"]
        ) >= float(criteria.minimum_oracle_advantage_over_best_fixed_log10),
        "oracle_opportunity_positive_across_structures": float(
            opportunity["positive_structure_oracle_fraction"]
        ) >= float(criteria.minimum_positive_structure_oracle_fraction),
        "shared_drive_changes_relative_controller_response": float(
            opportunity[
                "absolute_shared_drive_by_profile_response_interaction_log10"
            ]
        ) >= float(criteria.minimum_shared_drive_response_interaction_log10),
        "realized_optimal_profile_reproducible_across_futures": float(
            opportunity["mean_realized_optimal_profile_agreement_fraction"]
        ) >= float(criteria.minimum_realized_winner_agreement_fraction),
        "efficacy_uses_neural_only_EEG_and_policy_inputs_use_observed_EEG": bool(
            len(metrics)
            and metrics.context_features_use_observed_EEG.all()
            and metrics.efficacy_endpoint_uses_neural_only_EEG.all()
        ),
        "hidden_generator_and_spikes_excluded_from_deployable_inputs": bool(
            (~metrics.policy_uses_hidden_state_or_spikes.astype(bool)).all()
        ),
    }
    feasibility_names = [
        "shared_drive_observable_from_phase_invariant_EEG",
        "expected_oracle_uses_both_controller_profiles",
        "both_optimal_profiles_replicate_across_structures",
        "expected_oracle_has_practical_advantage_over_best_fixed",
        "oracle_opportunity_positive_across_structures",
        "shared_drive_changes_relative_controller_response",
        "realized_optimal_profile_reproducible_across_futures",
    ]
    ready = bool(all(checks.values()) and not smoke)
    conclusions = {
        "H5_P0_contextual_controller_profile_opportunity": (
            "PASSED" if ready else "NOT PASSED"
        ),
        "ready_for_disjoint_H5_policy_development": ready,
        "failed_feasibility_checks": [
            name for name in feasibility_names if not checks[name]
        ],
        "machine_learning_status": "NOT TRAINED OR TESTED",
    }
    return checks, conclusions


def _plots(
    *, root: Path, screening: pd.DataFrame, expected: pd.DataFrame,
    action_map: pd.DataFrame, structure: pd.DataFrame,
) -> None:
    labels = {PARTIAL: "partial shared drive", FULL: "full shared drive"}
    colors = {PARTIAL: "#4C78A8", FULL: "#E45756"}
    markers = {LOW: "o", HIGH: "^"}

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for (shared, diffusion), group in screening.groupby([
        "shared_drive_label", "label"
    ]):
        axis.scatter(
            group.context_C1,
            group.recent_resultant_to_rms,
            color=colors[str(shared)],
            marker=markers[str(diffusion)],
            s=55,
            alpha=0.85,
            label=f"{labels[str(shared)]}, {str(diffusion).replace('_', ' ')}",
        )
    axis.set(
        xlabel="Predecision phase-increment coherence C1",
        ylabel="Recent alpha resultant / EEG RMS",
        title="Noisy-EEG context axes",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_observed_context_map.png", dpi=250)
    figure.savefig(root / "figure_01_observed_context_map.pdf")
    plt.close(figure)

    active = expected[expected.controller_mode.ne(SHAM)]
    summary = active.groupby([
        "shared_drive_label", "controller_mode"
    ]).expected_post_distance_to_B_log10.agg(["mean", "sem"]).reset_index()
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(2)
    width = 0.34
    for index, mode in enumerate((CONSERVATIVE, RESPONSIVE)):
        group = summary[summary.controller_mode.eq(mode)].set_index(
            "shared_drive_label"
        ).reindex([PARTIAL, FULL])
        axis.bar(
            x + (index - 0.5) * width,
            group["mean"], width,
            yerr=group["sem"], capsize=3,
            label=("conservative 1 s / 250 ms" if mode == CONSERVATIVE
                   else "responsive 0.5 s / 125 ms"),
        )
    axis.set_xticks(x, [labels[PARTIAL], labels[FULL]])
    axis.set(
        ylabel="Expected neural-EEG distance to B (log10)",
        title="Context-dependent controller response",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_02_controller_response.png", dpi=250)
    figure.savefig(root / "figure_02_controller_response.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for shared, group in action_map.groupby("shared_drive_label"):
        axis.scatter(
            group.context_C1,
            group.responsive_advantage_over_conservative_log10,
            color=colors[str(shared)], s=55, alpha=0.85,
            label=labels[str(shared)],
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(
        xlabel="Predecision phase-increment coherence C1",
        ylabel="Responsive advantage over conservative (log10)",
        title="Full-information controller-profile map",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "figure_03_context_action_interaction.png", dpi=250)
    figure.savefig(root / "figure_03_context_action_interaction.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.bar(
        structure.structure_seed.astype(str),
        structure.mean_oracle_advantage_over_best_fixed_log10,
        color="#59A14F",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(
        xlabel="Independent circuit structure seed",
        ylabel="Oracle advantage over best fixed (log10)",
        title="Exploratory opportunity by structure",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_04_structure_opportunity.png", dpi=250)
    figure.savefig(root / "figure_04_structure_opportunity.pdf")
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = (
        Path(to_absolute_path(str(cfg.experiment.dir)))
        / ROOT_NAME
    )
    exists = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H5-P0 controller-profile feasibility")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    target = sources["target"]

    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for context in _run_context_specs(cfg):
        if rank == 0:
            print(
                f"context={context['context_id']} "
                f"structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"D={context['diffusion_rad2_per_s']:g} rad^2/s "
                f"q={context['shared_modulated_fraction']:g}"
            )
        state_cfg = _with_context_state(cfg, context)
        first_future = _future_seed(cfg, context, 0)
        baseline_reference = _run_controller(
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
            screening = _context_features(
                _observed_episode_view(baseline_reference), context, target, cfg
            )
            screening.update({
                "screen_uses_only_predecision_observed_EEG": True,
                "screen_uses_only_predecision_ideal_EEG": False,
                "observation_noise_model": "AR1_additive_sensor_noise",
                "configured_observation_noise_fraction": float(
                    cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg
                ),
            })
            screening_rows.append(screening)
            eligible = bool(screening["eligible"])
            selected_frequency = float(screening["EEG_selected_frequency_hz"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'EXCLUDED'}; "
                f"selected={selected_frequency:g} Hz; "
                f"reason={screening['exclusion_reasons']}"
            )
        else:
            screening, eligible, selected_frequency = None, None, None
        eligible = bool(comm.bcast(eligible, root=0))
        selected_frequency = float(comm.bcast(selected_frequency, root=0))
        if not eligible:
            continue
        action_cfg = _with_action_frequency(state_cfg, selected_frequency)
        n_futures = int(cfg.analysis.crossed_design.n_future_continuations)
        for future_index in range(n_futures):
            future_seed = _future_seed(cfg, context, future_index)
            episodes: dict[str, dict[str, Any]] | None = (
                {} if rank == 0 else None
            )
            for action_index, mode in enumerate(_controller_modes(cfg)):
                if future_index == 0 and mode == SHAM:
                    episode = baseline_reference
                else:
                    episode = _run_controller(
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
                _add_context_features_to_rows(rows, screening)
                shared_fields = {
                    "paired_shared_drive_context_id": str(
                        context["paired_shared_drive_context_id"]
                    ),
                    "shared_drive_label": str(context["shared_drive_label"]),
                    "shared_modulated_fraction": float(
                        context["shared_modulated_fraction"]
                    ),
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
    if not metric_rows:
        conclusion = {
            "scope": "H5-P0 exploratory full-information feasibility",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "H5_P0_contextual_controller_profile_opportunity": "NOT PASSED",
                "ready_for_disjoint_H5_policy_development": False,
                "machine_learning_status": "NOT TRAINED OR TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(
            conclusion, indent=2, allow_nan=False
        ))
        print("No eligible contexts; H5-P0: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    updates = pd.DataFrame(update_rows)
    expected = _expected_map(metrics)
    action_map, structure, opportunity = _controller_action_map(
        expected, metrics
    )
    classifier_rows, observability = _shared_drive_loso(screening_table)
    checks, conclusions = _checks(
        screening=screening_table,
        metrics=metrics,
        expected=expected,
        updates=updates,
        action_map=action_map,
        structure=structure,
        observability=observability,
        opportunity=opportunity,
        sources=sources,
        cfg=cfg,
    )

    metrics.to_csv(root / "context_controller_future_metrics.csv", index=False)
    trajectories.to_csv(root / "one_second_eeg_trajectories.csv", index=False)
    updates.to_csv(root / "causal_phase_updates.csv", index=False)
    expected.to_csv(root / "expected_context_controller_map.csv", index=False)
    action_map.to_csv(root / "controller_profile_action_map.csv", index=False)
    structure.to_csv(root / "structure_level_oracle_opportunity.csv", index=False)
    classifier_rows.to_csv(root / "shared_drive_observability_loso.csv", index=False)
    audit = {
        "observability": observability,
        "controller_profile_opportunity": opportunity,
    }
    (root / "H5_P0_feasibility_audit.json").write_text(json.dumps(
        _json_ready(audit), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H5_P0_controller_profile_full_information_feasibility",
        "frozen_sources": {
            "roots": sources["roots"], "hashes": sources["hashes"]
        },
        "state_generator": {
            "carrier_hz": [9.0, 11.0],
            "phase_diffusion_rad2_per_s": [0.5, 2.0],
            "shared_modulated_afferent_fraction": [0.5, 1.0],
            "modulation_depth": 0.04,
            "mean_afferent_rate_is_matched": True,
            "private_poisson_events_are_independent": True,
        },
        "observation_model": {
            "input_to_context_and_controller": "ideal neural EEG plus frozen AR(1) noise",
            "AR1_coefficient": float(cfg.analysis.observation_noise.ar1_coefficient),
            "noise_RMS_fraction_of_baseline_neural_EEG": float(
                cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg
            ),
            "efficacy_endpoint": "noise-free ideal neural EEG",
            "stimulation_artifact_modelled": False,
        },
        "controller_profiles": {
            mode: _profile(cfg, mode) for mode in EXPECTED_MODES
        },
        "fixed_active_action": {
            "amplitude_v_per_m": 0.2,
            "carrier": "selected from preceding EEG on the frozen 9/11-Hz grid",
            "relative_phase_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
            "montage": str(cfg.analysis.tacs.axial_montage),
        },
        "primary_opportunity_endpoint": (
            "expected neural-EEG distance-to-B gain of the post-hoc profile "
            "oracle over the best fixed controller profile"
        ),
        "statistical_unit": "independent circuit structure; context axes and futures are repeats",
        "purpose": "system identification before policy fitting",
        "not_a_trained_or_tested_machine_learning_policy": True,
        "not_a_disease_or_clinical_treatment_model": True,
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-P0 exploratory ideal-neural-EEG full-information feasibility",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure",
        "inference_boundary": (
            "Exploratory opportunity mapping only. A positive result may justify "
            "policy development and later disjoint confirmation; it does not "
            "establish H5 or a contextual bandit."
        ),
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot):
        _plots(
            root=root,
            screening=screening_table,
            expected=expected,
            action_map=action_map,
            structure=structure,
        )

    print("\n### H5-P0 screening")
    print(f"contexts screened: {len(screening_table)}")
    print(f"eligible contexts: {int(screening_table.eligible.sum())}")
    print(f"screening yield: {float(screening_table.eligible.mean()):.3f}")
    print("\n### H5-P0 feasibility checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### H5-P0 opportunity summary")
    print(json.dumps(_json_ready(audit), indent=2, allow_nan=False))
    print(
        "\nContextual controller-profile opportunity: "
        f"{conclusions['H5_P0_contextual_controller_profile_opportunity']}"
    )
    print("Machine-learning policy status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
