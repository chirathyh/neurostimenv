"""D1-R causal EEG phase-maintenance audit for the BallAndStick toy circuit.

D1 initialized an EEG-relative antiphase tACS waveform once and then let that
oscillator run open loop.  That is a poor mechanistic match to a carrier whose
phase follows a diffusion process.  D1-R keeps the carrier selector, 0.2-V/m
dose, montage, and relative-phase target fixed, and compares three paired arms:

* sham;
* one-time phase initialization (the D1 controller); and
* causal phase refresh from the preceding ideal-neural-EEG tail.

The refreshed controller never jumps the electric field phase.  At update k it
computes a wrapped phase error and realizes the correction as a bounded
frequency slew over the next update interval.  The block envelope and field
waveform therefore remain continuous.  This is exploratory mechanism and
reliability testing, not a bandit, a clinical model, or confirmatory evidence.
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
from omegaconf import DictConfig, OmegaConf


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from env.models.neuron.stimulation import (  # noqa: E402
    apply_raised_cosine_block_envelope,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _analyze_episode,
    _epoch_raw,
    _epoch_row,
    _feature_from_raw,
    _field_phase_from_eeg_coefficients,
    _plain,
    _run_condition,
    _sham,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (  # noqa: E402
    _field_removal_status,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _episode_feature,
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _fourier_coefficients,
    _process_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (  # noqa: E402
    HIGH,
    LOW,
    _context_features,
    _copy_cfg,
    _fit_B_target,
    _future_seed,
    _load_sources as _load_d0b_f0_sources,
    _reference_cfg,
    _reference_seeds,
    _relative_rate_safe,
    _run_context_specs,
    _with_diffusion_state,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _environment_action,
    _episode_config,
    _mpi_variables,
    _timeline,
    _validate_online_outputs,
    _zero_action,
)


SHAM = "sham"
ONE_TIME = "one_time"
REFRESHED = "phase_refreshed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _signed_phase_error(target_rad: float, observed_rad: float) -> float:
    """Return target minus observed in [-pi, pi]."""
    return float(np.angle(np.exp(1j * (float(target_rad) - float(observed_rad)))))


def _phase_slew_frequency(
    *, carrier_hz: float, target_phase_rad: float, oscillator_phase_rad: float,
    update_interval_ms: float, maximum_correction_hz: float,
) -> dict[str, float]:
    """Compute a phase-continuous, bounded frequency command for one update."""
    interval_s = float(update_interval_ms) / 1000.0
    if interval_s <= 0.0:
        raise ValueError("update_interval_ms must be positive.")
    if maximum_correction_hz < 0.0:
        raise ValueError("maximum_correction_hz must be non-negative.")
    error = _signed_phase_error(target_phase_rad, oscillator_phase_rad)
    unconstrained = error / (2.0 * np.pi * interval_s)
    correction = float(np.clip(
        unconstrained, -float(maximum_correction_hz), float(maximum_correction_hz)
    ))
    return {
        "phase_error_before_correction_rad": error,
        "unconstrained_frequency_correction_hz": float(unconstrained),
        "frequency_correction_hz": correction,
        "command_frequency_hz": float(carrier_hz) + correction,
    }


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    sources = _load_d0b_f0_sources(cfg)
    root = Path(to_absolute_path(str(cfg.analysis.source_d1.result_dir)))
    files = {
        "conclusion": root / "experiment_conclusion.json",
        "provenance": root / "protocol_and_provenance.json",
        "metrics": root / "context_action_future_metrics.csv",
        "screening": root / "prospective_screening.csv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen D1 sources: {missing}")
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {
        name: str(cfg.analysis.source_d1.expected_sha256[name]) for name in files
    }
    if observed != expected:
        raise RuntimeError(
            f"D1 source hash mismatch: expected={expected}, observed={observed}"
        )
    conclusion = json.loads(files["conclusion"].read_text())
    failed = [name for name, passed in conclusion["checks"].items() if not bool(passed)]
    if failed != ["realized_optimum_reproducible_across_futures"]:
        raise RuntimeError(
            "D1-R requires the frozen D1 run to have failed only its future "
            f"reproducibility gate; observed failed checks={failed}."
        )
    if bool(conclusion["conclusions"]["D1_full_information_action_map_feasible"]):
        raise RuntimeError("The frozen D1 source unexpectedly reports a passed map.")
    d1_metrics = pd.read_csv(files["metrics"])
    d1_screening = pd.read_csv(files["screening"])
    for table in (d1_metrics, d1_screening):
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed",
        ):
            if column in table:
                sources["source_seed_union"].update(
                    table[column].dropna().astype(int).tolist()
                )
    sources["roots"]["d1"] = str(root)
    sources["hashes"]["d1"] = observed
    sources["D1_failed_only_future_reproducibility"] = True
    return sources


def _controller_modes(cfg: DictConfig) -> list[str]:
    return [str(value) for value in cfg.analysis.actions.controller_modes]


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("D1-R requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("D1-R may not change recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("D1-R freezes the 9/11-Hz carrier grid.")
    levels = [
        (str(x.label), float(x.diffusion_rad2_per_s))
        for x in cfg.analysis.states.phase_diffusion_levels
    ]
    if levels != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("D1-R freezes D to 0.5 and 2 rad^2/s.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("D1-R freezes afferent modulation depth to 0.04.")
    if _controller_modes(cfg) != [SHAM, ONE_TIME, REFRESHED]:
        raise ValueError("D1-R controller order must be sham/one_time/phase_refreshed.")
    amplitude = float(cfg.analysis.actions.amplitude_v_per_m)
    if not np.isclose(amplitude, 0.2) or amplitude > float(
        cfg.analysis.maximum_field_v_per_m
    ):
        raise ValueError("D1-R freezes both active controllers to 0.2 V/m.")
    if not np.isclose(
        _wrap_phase(float(cfg.analysis.tacs.relative_phase_offset_rad)), np.pi
    ):
        raise ValueError("D1-R freezes the EEG-relative phase target to pi.")
    window_ms = float(cfg.env.simulation.obs_win_len)
    refresh_ms = float(cfg.analysis.tacs.refresh_interval_ms)
    history_ms = float(cfg.analysis.tacs.phase_estimation_history_ms)
    if not np.isclose(window_ms, 1000.0):
        raise ValueError("D1-R requires 1000-ms outer online windows.")
    if refresh_ms <= 0.0 or not np.isclose(window_ms / refresh_ms, round(window_ms / refresh_ms)):
        raise ValueError("refresh_interval_ms must divide the 1000-ms outer window.")
    if history_ms < 500.0 or not np.isclose(history_ms / refresh_ms, round(history_ms / refresh_ms)):
        raise ValueError("The causal phase history must align to refresh windows.")
    if not np.isclose(
        float(cfg.analysis.tacs.maximum_frequency_correction_hz), 2.0
    ):
        raise ValueError("D1-R freezes the phase-slew limit to +/-2 Hz.")
    timeline = cfg.analysis.timeline
    minimum_baseline = 4 if bool(cfg.analysis.smoke_test) else 12
    if int(timeline.baseline_steps) < minimum_baseline:
        raise ValueError(f"D1-R requires at least {minimum_baseline} baseline seconds.")
    stimulation_ms = int(timeline.stimulation_steps) * window_ms
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    if trim_ms < float(timeline.block_ramp_ms) or 2.0 * trim_ms >= stimulation_ms:
        raise ValueError("D1-R trimming must contain both block ramps and leave EEG.")
    if not np.isclose((stimulation_ms - 2.0 * trim_ms) / 1000.0, round(
        (stimulation_ms - 2.0 * trim_ms) / 1000.0
    )):
        raise ValueError("The trimmed endpoint must contain complete one-second windows.")
    if not bool(cfg.analysis.smoke_test):
        if int(cfg.analysis.reference_calibration.n_seeds) < 12:
            raise ValueError("Full D1-R requires at least 12 B references.")
        if int(cfg.analysis.crossed_design.n_structure_seeds) < 3:
            raise ValueError("Full D1-R requires at least three structures.")
        if int(cfg.analysis.crossed_design.n_future_continuations) < 4:
            raise ValueError("Full D1-R requires at least four futures per arm.")

    contexts = _run_context_specs(cfg)
    references = set(_reference_seeds(cfg))
    namespaces = [
        references,
        {int(x["structure_seed"]) for x in contexts},
        {int(x["history_seed"]) for x in contexts},
        {int(x["phase_seed"]) for x in contexts},
        {int(x["trial_seed"]) for x in contexts},
        {
            _future_seed(cfg, context, future_index)
            for context in contexts
            for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations))
        },
    ]
    if any(not values for values in namespaces):
        raise ValueError("Every D1-R seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("D1-R seed namespaces overlap.")
    if set().union(*namespaces).intersection(sources["source_seed_union"]):
        raise ValueError("D1-R seeds overlap D0b, F0, or D1 source seeds.")
    if max(namespaces[1]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("A D1-R structure seed exceeds the uint32 mapping range.")


def _tail_phase_estimate(
    outputs: list[dict[str, Any]], *, boundary_ms: float, history_ms: float,
    simulator_fs_hz: float, relative_offset_rad: float, cfg: DictConfig,
) -> dict[str, float]:
    """Estimate EEG and desired field phase using only samples before boundary."""
    times = np.concatenate([
        np.asarray(output["sample_times_ms"], dtype=float).reshape(-1)
        for output in outputs
    ])
    eeg = np.concatenate([
        np.asarray(output["eeg_v"], dtype=float).reshape(-1) for output in outputs
    ])
    dt_ms = 1000.0 / float(simulator_fs_hz)
    expected = int(round(float(history_ms) / dt_ms))
    # Recorder convention is (t_start, t_stop]; exclude the left endpoint
    # explicitly even when it is present because of floating-point rounding.
    keep = (times > float(boundary_ms) - float(history_ms) + 1.0e-9) & (
        times <= float(boundary_ms) + 1.0e-9
    )
    selected = eeg[keep]
    selected_times = times[keep]
    if selected.size != expected:
        raise RuntimeError(
            f"Causal phase tail has {selected.size} samples; expected {expected}."
        )
    processed, fs_hz, _, _, generic = _process_eeg(
        selected, simulator_fs_hz=simulator_fs_hz, cfg=cfg
    )
    start_ms = float(selected_times[0] - dt_ms)
    cosine, sine = _fourier_coefficients(
        processed,
        fs_hz=fs_hz,
        start_ms=start_ms,
        frequency_hz=float(cfg.analysis.tacs.frequency_hz),
    )
    field_phase, eeg_phase = _field_phase_from_eeg_coefficients(
        cosine,
        sine,
        block_start_ms=float(boundary_ms),
        frequency_hz=float(cfg.analysis.tacs.frequency_hz),
        relative_offset_rad=float(relative_offset_rad),
    )
    return {
        "estimate_start_ms": float(selected_times[0] - dt_ms),
        "estimate_stop_ms": float(selected_times[-1]),
        "desired_field_phase_rad": float(field_phase),
        "estimated_eeg_phase_at_boundary_rad": float(eeg_phase),
        "eeg_resultant_v": float(np.hypot(cosine, sine)),
        "eeg_rms_v": float(generic["rms_v"]),
        "resultant_to_rms": float(
            np.hypot(cosine, sine)
            / max(float(generic["rms_v"]), np.finfo(float).tiny)
        ),
    }


def _controller_action(cfg: DictConfig, mode: str) -> dict[str, Any]:
    amplitude = 0.0 if mode == SHAM else float(cfg.analysis.actions.amplitude_v_per_m)
    return {
        "id": mode,
        "role": "causal_phase_maintenance_audit",
        "controller_mode": mode,
        "montage": str(cfg.analysis.tacs.axial_montage),
        "dc_offset_v_per_m": 0.0,
        "ac_amplitude_v_per_m": amplitude,
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
    """Run one persistent paired episode with causal phase updates."""
    run_cfg = _episode_config(
        base_cfg, seed=int(context["trial_seed"]), output_dir=output_dir
    )
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    refresh_ms = float(run_cfg.analysis.tacs.refresh_interval_ms)
    history_ms = float(run_cfg.analysis.tacs.phase_estimation_history_ms)
    pre_steps = int(run_cfg.analysis.timeline.burn_in_steps) + int(
        run_cfg.analysis.timeline.baseline_steps
    )
    block_start_ms = pre_steps * window_ms
    stimulation_ms = int(run_cfg.analysis.timeline.stimulation_steps) * window_ms
    block_stop_ms = block_start_ms + stimulation_ms
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
            comm,
            size,
            rank,
            int(context["trial_seed"]),
            structure_seed=structure,
            drive_seed=int(context["history_seed"]),
            future_drive_seed=int(future_seed),
            future_start_ms=block_start_ms,
        ),
        ENV_SEED=0,
    )
    outputs = (
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
    all_preceding_outputs: list[dict[str, Any]] | None = [] if rank == 0 else None
    try:
        for epoch in ("burn_in", "baseline"):
            count = int(run_cfg.analysis.timeline[f"{epoch}_steps"])
            for _ in range(count):
                output = environment.step_online(
                    zero, duration_ms=window_ms, phase_continuous=True
                )
                if rank == 0:
                    outputs[epoch].append(output)
                    all_preceding_outputs.append(output)

        n_updates = int(round(stimulation_ms / refresh_ms))
        for update_index in range(n_updates):
            boundary_ms = block_start_ms + update_index * refresh_ms
            if rank == 0:
                estimate = _tail_phase_estimate(
                    all_preceding_outputs,
                    boundary_ms=boundary_ms,
                    history_ms=history_ms,
                    simulator_fs_hz=1000.0 / float(run_cfg.env.network.dt),
                    relative_offset_rad=float(
                        run_cfg.analysis.tacs.relative_phase_offset_rad
                    ),
                    cfg=run_cfg,
                )
                oscillator_phase = (
                    float(estimate["desired_field_phase_rad"])
                    if update_index == 0
                    else float(environment.phase_rad)
                )
                slew = _phase_slew_frequency(
                    carrier_hz=carrier,
                    target_phase_rad=float(estimate["desired_field_phase_rad"]),
                    oscillator_phase_rad=oscillator_phase,
                    update_interval_ms=refresh_ms,
                    maximum_correction_hz=maximum_correction,
                )
                if mode != REFRESHED or update_index == 0:
                    slew["frequency_correction_hz"] = 0.0
                    slew["command_frequency_hz"] = carrier
                diagnostics = {
                    "update_index": update_index,
                    "boundary_ms": boundary_ms,
                    "controller_mode": mode,
                    "carrier_frequency_hz": carrier,
                    "oscillator_phase_before_update_rad": oscillator_phase,
                    "phase_refresh_applied": bool(
                        mode == REFRESHED and update_index > 0
                    ),
                    **estimate,
                    **slew,
                }
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
                action["phase_rad"] = float(
                    diagnostics["desired_field_phase_rad"]
                )
            output = environment.step_online(
                action,
                duration_ms=refresh_ms,
                phase_continuous=True,
                ramp_ms=0.0,
                block_envelope=envelope if mode != SHAM else None,
            )
            if rank == 0:
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
                all_preceding_outputs.append(output)

        for _ in range(int(run_cfg.analysis.timeline.washout_steps)):
            output = environment.step_online(
                zero, duration_ms=window_ms, phase_continuous=True
            )
            if rank == 0:
                outputs["washout"].append(output)
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
        raise RuntimeError("D1-R washout left residual extracellular voltage.")
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
        "phase_updates": update_rows,
        "final_residual_mV": float(final_residual_mV),
    }


def _analyze_controller_episode(
    simulation: dict[str, Any] | None, *, condition_cfg: DictConfig,
    context: dict[str, Any], mode: str, action_index: int, output_dir: Path,
    rank: int,
) -> dict[str, Any] | None:
    if rank != 0:
        return None
    action = simulation["action"]
    epoch_rows, raw_by_epoch = _analyze_episode(
        simulation,
        action=action,
        action_index=action_index,
        arm=mode,
        cfg=condition_cfg,
        output_dir=output_dir / "analysis",
    )
    episode = {
        "simulation": simulation,
        "epoch_rows": epoch_rows,
        "raw_by_epoch": raw_by_epoch,
        "simulator_fs_hz": 1000.0 / float(condition_cfg.env.network.dt),
    }
    trim_ms = float(condition_cfg.analysis.timeline.stimulation_analysis_trim_ms)
    dt_ms = float(condition_cfg.env.network.dt)
    for row in epoch_rows:
        epoch = str(row["epoch"])
        raw = _epoch_raw(episode, epoch)
        start_ms = float(simulation["outputs_by_epoch"][epoch][0]["t_start_ms"])
        if epoch == "stimulation" and trim_ms > 0.0:
            trim_samples = int(round(trim_ms / dt_ms))
            raw = raw[trim_samples:-trim_samples]
            start_ms += trim_ms
        feature, _, _, _ = _feature_from_raw(
            raw,
            simulator_fs_hz=float(episode["simulator_fs_hz"]),
            start_ms=start_ms,
            cfg=condition_cfg,
        )
        row.update(feature)
        row.update({
            "condition_id": mode,
            "controller_mode": mode,
            "structure_seed": int(context["structure_seed"]),
            "drive_seed": int(context["history_seed"]),
            "future_drive_seed": int(simulation["future_drive_seed"]),
            "phase_seed": int(context["phase_seed"]),
        })
    return episode


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
    return _analyze_controller_episode(
        simulation,
        condition_cfg=condition_cfg,
        context=context,
        mode=mode,
        action_index=action_index,
        output_dir=output_dir,
        rank=rank,
    )


def _one_second_rows(
    episode: dict[str, Any], *, context: dict[str, Any], screening: dict[str, Any],
    future_index: int, mode: str, target_alpha: float, cfg: DictConfig,
) -> list[dict[str, Any]]:
    raw = _epoch_raw(episode, "stimulation")
    dt_ms = float(cfg.env.network.dt)
    trim_samples = int(round(
        float(cfg.analysis.timeline.stimulation_analysis_trim_ms) / dt_ms
    ))
    # ``array[0:-0]`` is empty, so preserve the complete epoch when no trim is
    # configured. This trajectory-only bug did not affect whole-epoch primary
    # endpoints, but silently removed all one-second manuscript rows whenever
    # stimulation_analysis_trim_ms was zero.
    central = raw if trim_samples == 0 else raw[trim_samples:-trim_samples]
    samples = int(round(1000.0 / dt_ms))
    if central.size % samples:
        raise RuntimeError("D1-R endpoint does not split into one-second windows.")
    start_ms = (
        float(episode["simulation"]["block_start_ms"])
        + float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    )
    rows = []
    for index in range(central.size // samples):
        feature, _, _, _ = _feature_from_raw(
            central[index * samples:(index + 1) * samples],
            simulator_fs_hz=float(episode["simulator_fs_hz"]),
            start_ms=start_ms + 1000.0 * index,
            cfg=cfg,
        )
        alpha = float(feature["log10_alpha_power_8_12_hz"])
        rows.append({
            **{key: context[key] for key in (
                "context_id", "structure_seed", "hidden_frequency_hz", "label",
                "diffusion_rad2_per_s",
            )},
            "context_C1": float(screening["context_C1"]),
            "future_index": int(future_index + 1),
            "controller_mode": mode,
            "analysis_window_index": int(index + 1),
            "analysis_window_start_ms": float(start_ms + 1000.0 * index),
            "log10_alpha_power": alpha,
            "distance_to_B_log10": abs(alpha - float(target_alpha)),
        })
    return rows


def _metric_rows(
    *, context: dict[str, Any], screening: dict[str, Any], future_index: int,
    future_seed: int, episodes: dict[str, dict[str, Any]],
    baseline_reference: dict[str, Any], target: dict[str, Any], cfg: DictConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_alpha = float(target["outcome"]["mean_log10_alpha"])
    sham = episodes[SHAM]
    sham_feature, _, _ = _episode_feature(sham, "stimulation", cfg)
    sham_alpha = float(sham_feature["log10_alpha_power_8_12_hz"])
    sham_distance = abs(sham_alpha - target_alpha)
    sham_row = _epoch_row(sham, "stimulation")
    sham_baseline = _epoch_row(sham, "baseline")
    sham_washout = _epoch_row(sham, "washout")
    rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for mode in _controller_modes(cfg):
        episode = episodes[mode]
        feature, _, _ = _episode_feature(episode, "stimulation", cfg)
        outcome = _epoch_row(episode, "stimulation")
        baseline = _epoch_row(episode, "baseline")
        washout = _epoch_row(episode, "washout")
        alpha = float(feature["log10_alpha_power_8_12_hz"])
        distance = abs(alpha - target_alpha)
        baseline_error = _relative_rms_error(
            _epoch_raw(baseline_reference, "baseline"),
            _epoch_raw(episode, "baseline"),
        )
        if mode == SHAM:
            residual, tolerance, recovered = 0.0, 0.0, True
            rate_safe = _relative_rate_safe(sham_row, sham_row, cfg)
        else:
            residual = float(
                (sham_washout.log10_alpha_power_8_12_hz
                 - sham_baseline.log10_alpha_power_8_12_hz)
                - (washout.log10_alpha_power_8_12_hz
                   - baseline.log10_alpha_power_8_12_hz)
            )
            recovered, tolerance = _field_removal_status(
                effect_log10=sham_alpha - alpha,
                residual_log10=residual,
                cfg=cfg,
            )
            rate_safe = _relative_rate_safe(sham_row, outcome, cfg)
        updates = pd.DataFrame(episode["simulation"]["phase_updates"])
        active_updates = updates.iloc[1:] if len(updates) > 1 else updates
        rows.append({
            **{key: context[key] for key in context},
            "context_C1": float(screening["context_C1"]),
            "context_alpha_excess_log10": float(
                screening["context_alpha_excess_log10"]
            ),
            "EEG_selected_frequency_hz": float(
                screening["EEG_selected_frequency_hz"]
            ),
            "future_index": int(future_index + 1),
            "future_drive_seed": int(future_seed),
            "controller_mode": mode,
            "amplitude_v_per_m": (
                0.0 if mode == SHAM else float(cfg.analysis.actions.amplitude_v_per_m)
            ),
            "post_log10_alpha_power": alpha,
            "post_distance_to_B_log10": distance,
            "causal_distance_improvement_vs_sham_log10": sham_distance - distance,
            "causal_alpha_suppression_vs_sham_log10": sham_alpha - alpha,
            "post_E_firing_rate_hz": float(outcome.E_firing_rate_hz),
            "post_I_firing_rate_hz": float(outcome.I_firing_rate_hz),
            "rate_safe": bool(rate_safe),
            "washout_residual_log10": residual,
            "washout_tolerance_log10": tolerance,
            "field_removal_recovered": bool(recovered),
            "final_extracellular_residual_mV": float(
                episode["simulation"]["final_residual_mV"]
            ),
            "baseline_relative_rms_error": float(baseline_error),
            "phase_update_count": int(len(updates)),
            "mean_abs_phase_error_before_correction_rad": float(
                active_updates.phase_error_before_correction_rad.abs().mean()
            ),
            "maximum_abs_phase_error_before_correction_rad": float(
                active_updates.phase_error_before_correction_rad.abs().max()
            ),
            "mean_abs_frequency_correction_hz": float(
                active_updates.frequency_correction_hz.abs().mean()
            ),
            "maximum_abs_frequency_correction_hz": float(
                active_updates.frequency_correction_hz.abs().max()
            ),
            "maximum_field_boundary_discontinuity_v_per_m": float(
                updates.field_boundary_discontinuity_v_per_m.max()
            ),
            "all_phase_estimates_causal": bool(
                updates.estimate_is_strictly_causal.all()
            ),
            "policy_uses_hidden_state_or_spikes": False,
        })
        trajectory_rows.extend(_one_second_rows(
            episode,
            context=context,
            screening=screening,
            future_index=future_index,
            mode=mode,
            target_alpha=target_alpha,
            cfg=cfg,
        ))
        for update in episode["simulation"]["phase_updates"]:
            update_rows.append({
                **{key: context[key] for key in (
                    "context_id", "structure_seed", "hidden_frequency_hz", "label",
                    "diffusion_rad2_per_s",
                )},
                "context_C1": float(screening["context_C1"]),
                "future_index": int(future_index + 1),
                "future_drive_seed": int(future_seed),
                **update,
            })
    return rows, trajectory_rows, update_rows


def _expected_map(metrics: pd.DataFrame) -> pd.DataFrame:
    group = [
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "diffusion_rad2_per_s", "context_C1", "controller_mode",
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
            mean_abs_phase_error_rad=(
                "mean_abs_phase_error_before_correction_rad", "mean"
            ),
            maximum_field_boundary_discontinuity_v_per_m=(
                "maximum_field_boundary_discontinuity_v_per_m", "max"
            ),
        )
        .sort_values(group)
        .reset_index(drop=True)
    )


def _comparison_tables(
    expected: pd.DataFrame, metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    active = expected[expected.controller_mode.isin([ONE_TIME, REFRESHED])]
    pivot = active.pivot(index=[
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "diffusion_rad2_per_s", "context_C1",
    ], columns="controller_mode", values=[
        "expected_post_distance_to_B_log10", "future_sd_post_distance_log10",
        "mean_abs_phase_error_rad",
    ]).reset_index()
    pivot.columns = [
        "_".join(str(x) for x in column if str(x))
        if isinstance(column, tuple) else str(column) for column in pivot.columns
    ]
    pivot = pivot.rename(columns={
        "context_id_": "context_id",
        "structure_seed_": "structure_seed",
        "hidden_frequency_hz_": "hidden_frequency_hz",
        "label_": "label",
        "diffusion_rad2_per_s_": "diffusion_rad2_per_s",
        "context_C1_": "context_C1",
    })
    one = "expected_post_distance_to_B_log10_one_time"
    refreshed = "expected_post_distance_to_B_log10_phase_refreshed"
    pivot["refresh_advantage_over_one_time_log10"] = pivot[one] - pivot[refreshed]
    pivot["expected_active_winner"] = np.where(
        pivot[refreshed] < pivot[one], REFRESHED, ONE_TIME
    )

    realized = metrics[metrics.controller_mode.isin([ONE_TIME, REFRESHED])]
    agreement = []
    for row in pivot.itertuples():
        group = realized[realized.context_id.eq(row.context_id)]
        expected_winner = str(row.expected_active_winner)
        winners = []
        for _, future in group.groupby("future_index"):
            winners.append(str(future.sort_values(
                ["post_distance_to_B_log10", "controller_mode"]
            ).iloc[0].controller_mode))
        agreement.append(float(np.mean(np.asarray(winners) == expected_winner)))
    pivot["realized_winner_agreement_fraction"] = agreement
    structure = (
        pivot.groupby("structure_seed", as_index=False)
        .agg(
            context_count=("context_id", "nunique"),
            mean_refresh_advantage_log10=(
                "refresh_advantage_over_one_time_log10", "mean"
            ),
            mean_winner_agreement_fraction=(
                "realized_winner_agreement_fraction", "mean"
            ),
        )
    )
    audit = {
        "mean_refresh_advantage_over_one_time_log10": float(
            structure.mean_refresh_advantage_log10.mean()
        ),
        "positive_structure_fraction": float(
            np.mean(structure.mean_refresh_advantage_log10 > 0.0)
        ),
        "mean_realized_winner_agreement_fraction": float(
            pivot.realized_winner_agreement_fraction.mean()
        ),
        "mean_one_time_future_sd_log10": float(
            pivot.future_sd_post_distance_log10_one_time.mean()
        ),
        "mean_refreshed_future_sd_log10": float(
            pivot.future_sd_post_distance_log10_phase_refreshed.mean()
        ),
        "refresh_to_one_time_future_sd_ratio": float(
            pivot.future_sd_post_distance_log10_phase_refreshed.mean()
            / max(
                pivot.future_sd_post_distance_log10_one_time.mean(),
                np.finfo(float).tiny,
            )
        ),
        "mean_one_time_phase_error_rad": float(
            pivot.mean_abs_phase_error_rad_one_time.mean()
        ),
        "mean_refreshed_phase_error_rad": float(
            pivot.mean_abs_phase_error_rad_phase_refreshed.mean()
        ),
    }
    for label in (LOW, HIGH):
        subset = pivot[pivot.label.eq(label)]
        audit[f"mean_refresh_advantage_{label}_log10"] = float(
            subset.refresh_advantage_over_one_time_log10.mean()
        )
    return pivot, structure, audit


def _checks(
    *, calibration: pd.DataFrame, screening: pd.DataFrame, metrics: pd.DataFrame,
    expected: pd.DataFrame, comparison: pd.DataFrame, structure: pd.DataFrame,
    updates: pd.DataFrame, audit: dict[str, Any], sources: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    active = metrics[metrics.controller_mode.ne(SHAM)]
    reference_rates = {
        "E": float(calibration.E_firing_rate_hz.mean()),
        "I": float(calibration.I_firing_rate_hz.mean()),
    }
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    rate_matched = all(
        abs(float(getattr(row, f"baseline_{population}_firing_rate_hz"))
            - reference_rates[population])
        <= tolerance * max(reference_rates[population], np.finfo(float).tiny)
        for row in eligible.itertuples() for population in ("E", "I")
    )
    refreshed_updates = updates[
        updates.controller_mode.eq(REFRESHED) & updates.phase_refresh_applied
    ]
    checks = {
        "source_D1_hash_locked_with_future_instability": bool(
            sources["D1_failed_only_future_reproducibility"]
        ),
        "D1R_seeds_disjoint_from_D0b_F0_and_D1": True,
        "reference_target_calibrated_on_disjoint_B_seeds": len(calibration)
        >= int(criteria.minimum_reference_seeds) or bool(cfg.analysis.smoke_test),
        "four_second_duration_matched_EEG_endpoint": bool(
            np.isclose(
                float(cfg.analysis.timeline.stimulation_steps)
                * float(cfg.env.simulation.obs_win_len)
                - 2.0 * float(cfg.analysis.timeline.stimulation_analysis_trim_ms),
                4000.0,
            )
            or bool(cfg.analysis.smoke_test)
        ),
        "complete_crossed_screening_grid": len(screening) == len(
            _run_context_specs(cfg)
        ),
        "screening_uses_only_predecision_ideal_EEG": bool(
            screening.screen_uses_only_predecision_ideal_EEG.all()
        ),
        "minimum_eligible_contexts": len(eligible)
        >= int(criteria.minimum_eligible_contexts) or bool(cfg.analysis.smoke_test),
        "minimum_independent_structures": eligible.structure_seed.nunique()
        >= int(criteria.minimum_structure_seeds) or bool(cfg.analysis.smoke_test),
        "both_diffusion_levels_and_frequencies_enrolled": bool(
            eligible.label.nunique() == 2 and eligible.hidden_frequency_hz.nunique() == 2
        ) or bool(cfg.analysis.smoke_test),
        "frequency_identified_from_predecision_EEG": bool(
            len(eligible) and eligible.EEG_frequency_selection_correct.mean()
            >= float(criteria.minimum_frequency_detection_accuracy)
        ),
        "multiple_independent_postdecision_futures": bool(
            len(expected)
            and expected.n_futures.min() >= int(criteria.minimum_future_continuations)
        ) or bool(cfg.analysis.smoke_test),
        "identical_predecision_EEG_across_controllers_and_futures": bool(
            len(metrics)
            and metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "both_active_arms_use_identical_0p2_V_per_m": bool(
            len(active) and np.allclose(active.amplitude_v_per_m, 0.2)
        ),
        "phase_updates_use_only_preceding_EEG": bool(
            len(updates)
            and updates.estimate_is_strictly_causal.all()
            and (updates.estimate_stop_ms - updates.boundary_ms).max()
            <= float(criteria.maximum_causal_timing_error_ms)
        ),
        "phase_refreshed_arm_updates_after_onset": bool(
            len(refreshed_updates) > 0
        ),
        "phase_correction_is_frequency_bounded": bool(
            len(refreshed_updates)
            and refreshed_updates.frequency_correction_hz.abs().max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "field_waveform_continuous_across_update_boundaries": bool(
            len(active)
            and active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
        ),
        "all_actions_rate_safe": bool(len(metrics) and metrics.rate_safe.all()),
        "reference_rate_matched": bool(rate_matched),
        "field_removal_recovered": bool(
            len(metrics) and metrics.field_removal_recovered.all()
            and np.allclose(metrics.final_extracellular_residual_mV, 0.0)
        ),
        "refreshed_phase_error_lower_than_one_time": float(
            audit["mean_refreshed_phase_error_rad"]
        ) < float(audit["mean_one_time_phase_error_rad"]),
        "refreshed_controller_has_practical_mean_advantage": float(
            audit["mean_refresh_advantage_over_one_time_log10"]
        ) >= float(criteria.practical_refresh_advantage_log10),
        "refresh_advantage_positive_across_structures": float(
            audit["positive_structure_fraction"]
        ) >= float(criteria.minimum_positive_structure_fraction),
        "realized_active_winner_reproducible_across_futures": float(
            audit["mean_realized_winner_agreement_fraction"]
        ) >= float(criteria.minimum_realized_winner_agreement_fraction),
        "refresh_does_not_increase_future_variance": float(
            audit["refresh_to_one_time_future_sd_ratio"]
        ) <= float(criteria.maximum_refresh_to_one_time_future_sd_ratio),
        "policy_and_controller_exclude_hidden_state_and_spikes": bool(
            (~metrics.policy_uses_hidden_state_or_spikes.astype(bool)).all()
        ),
    }
    mechanism_gate = [
        "source_D1_hash_locked_with_future_instability",
        "D1R_seeds_disjoint_from_D0b_F0_and_D1",
        "reference_target_calibrated_on_disjoint_B_seeds",
        "four_second_duration_matched_EEG_endpoint",
        "complete_crossed_screening_grid",
        "screening_uses_only_predecision_ideal_EEG",
        "minimum_eligible_contexts",
        "minimum_independent_structures",
        "both_diffusion_levels_and_frequencies_enrolled",
        "frequency_identified_from_predecision_EEG",
        "multiple_independent_postdecision_futures",
        "identical_predecision_EEG_across_controllers_and_futures",
        "both_active_arms_use_identical_0p2_V_per_m",
        "phase_updates_use_only_preceding_EEG",
        "phase_refreshed_arm_updates_after_onset",
        "phase_correction_is_frequency_bounded",
        "field_waveform_continuous_across_update_boundaries",
        "all_actions_rate_safe",
        "reference_rate_matched",
        "field_removal_recovered",
        "refreshed_phase_error_lower_than_one_time",
        "refreshed_controller_has_practical_mean_advantage",
        "refresh_advantage_positive_across_structures",
        "realized_active_winner_reproducible_across_futures",
        "refresh_does_not_increase_future_variance",
        "policy_and_controller_exclude_hidden_state_and_spikes",
    ]
    passed = bool(all(checks[name] for name in mechanism_gate) and not bool(
        cfg.analysis.smoke_test
    ))
    return checks, {
        **audit,
        "phase_refresh_mechanism_feasible": passed,
        "ready_for_reliable_context_action_remapping": passed,
        "contextual_bandit_status": "NOT TRAINED OR TESTED",
        "claim_scope": "exploratory ideal-neural-EEG phase-maintenance audit",
    }


def _plots(
    *, root: Path, expected: pd.DataFrame, comparison: pd.DataFrame,
    structure: pd.DataFrame, updates: pd.DataFrame, trajectories: pd.DataFrame,
    cfg: DictConfig,
) -> None:
    modes = [SHAM, ONE_TIME, REFRESHED]
    colors = {SHAM: "0.5", ONE_TIME: "tab:orange", REFRESHED: "tab:blue"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, label in zip(axes, (LOW, HIGH)):
        subset = expected[expected.label.eq(label)]
        means = subset.groupby("controller_mode").expected_post_distance_to_B_log10.mean()
        axis.bar(
            range(len(modes)), [means.get(mode, np.nan) for mode in modes],
            color=[colors[mode] for mode in modes],
        )
        axis.set_xticks(range(len(modes)), ["sham", "one-time", "refreshed"])
        axis.set_title(label.replace("_", " "))
        axis.set_ylabel("Expected four-second distance to B (log10)")
    figure.suptitle("Causal phase-maintenance comparison")
    figure.tight_layout()
    figure.savefig(root / "figure_01_controller_outcomes.png", dpi=250)
    plt.close(figure)

    active = updates[updates.controller_mode.isin([ONE_TIME, REFRESHED])]
    summary = active.groupby([
        "controller_mode", "update_index"
    ]).phase_error_before_correction_rad.apply(
        lambda values: float(np.mean(np.abs(values)))
    ).reset_index(name="mean_abs_phase_error_rad")
    figure, axis = plt.subplots(figsize=(8, 4))
    for mode in (ONE_TIME, REFRESHED):
        subset = summary[summary.controller_mode.eq(mode)]
        axis.plot(
            subset.update_index * 0.25,
            subset.mean_abs_phase_error_rad,
            marker="o",
            label=mode.replace("_", " "),
            color=colors[mode],
        )
    axis.set(xlabel="Time since stimulation onset (s)", ylabel="Mean |phase error| (rad)")
    axis.legend(frameon=False)
    axis.set_title("Causal phase error before each correction")
    figure.tight_layout()
    figure.savefig(root / "figure_02_phase_tracking.png", dpi=250)
    plt.close(figure)

    summary = trajectories.groupby([
        "controller_mode", "analysis_window_index"
    ]).distance_to_B_log10.mean().reset_index()
    figure, axis = plt.subplots(figsize=(8, 4))
    for mode in modes:
        subset = summary[summary.controller_mode.eq(mode)]
        axis.plot(
            subset.analysis_window_index,
            subset.distance_to_B_log10,
            marker="o",
            label=mode.replace("_", " "),
            color=colors[mode],
        )
    axis.set(
        xlabel="One-second analysis window",
        ylabel="Mean distance to B (log10)",
        xticks=sorted(summary.analysis_window_index.unique()),
    )
    axis.legend(frameon=False)
    axis.set_title("Within-intervention EEG trajectory")
    figure.tight_layout()
    figure.savefig(root / "figure_03_eeg_trajectory.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(structure), dtype=float)
    axis.bar(
        positions,
        structure.mean_refresh_advantage_log10,
        color="tab:green",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(positions, structure.structure_seed.astype(str))
    axis.set(
        xlabel="Independent circuit structure",
        ylabel="Refreshed advantage over one-time (log10)",
        title="Structure-level directional audit",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_04_structure_advantage.png", dpi=250)
    plt.close(figure)

    # Reconstruct the exact analytical field command for one paired example.
    first = updates.iloc[0]
    example = updates[
        updates.context_id.eq(first.context_id)
        & updates.future_index.eq(first.future_index)
        & updates.controller_mode.isin([ONE_TIME, REFRESHED])
    ]
    refresh_ms = float(cfg.analysis.tacs.refresh_interval_ms)
    amplitude = float(cfg.analysis.actions.amplitude_v_per_m)
    block_start = float(example.boundary_ms.min())
    block_stop = float(example.boundary_ms.max() + refresh_ms)
    figure, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, sharey=True)
    for axis, mode in zip(axes, (ONE_TIME, REFRESHED)):
        group = example[example.controller_mode.eq(mode)].sort_values("update_index")
        time_parts, field_parts = [], []
        for row_index, row in enumerate(group.itertuples()):
            relative = np.arange(0.0, refresh_ms + 0.5, 0.5)
            time_ms = float(row.boundary_ms) + relative
            phase = (
                float(row.oscillator_phase_before_update_rad)
                + 2.0 * np.pi * float(row.command_frequency_hz)
                * relative / 1000.0
            )
            field = amplitude * np.sin(phase)
            if row_index < len(group) - 1:
                time_ms, field = time_ms[:-1], field[:-1]
            time_parts.append(time_ms)
            field_parts.append(field)
        time_ms = np.concatenate(time_parts)
        field = apply_raised_cosine_block_envelope(
            np.concatenate(field_parts),
            time_ms=time_ms,
            block_start_ms=block_start,
            block_stop_ms=block_stop,
            ramp_ms=float(cfg.analysis.timeline.block_ramp_ms),
        )
        axis.plot((time_ms - block_start) / 1000.0, field, linewidth=0.9)
        for boundary in group.boundary_ms.iloc[1:]:
            axis.axvline((float(boundary) - block_start) / 1000.0,
                         color="0.85", linewidth=0.5)
        axis.set_ylabel("Field (V/m)")
        axis.set_title(mode.replace("_", " "))
    axes[-1].set_xlabel("Time since stimulation onset (s)")
    figure.suptitle("Continuous field across causal controller windows")
    figure.tight_layout()
    figure.savefig(root / "figure_05_example_field_waveform.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "phase_refresh_audit"
    if rank == 0:
        exists = bool(root.exists() and any(root.iterdir()))
    else:
        exists = None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### D1-R causal phase-refresh audit")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    calibration_rows: list[dict[str, Any]] = []
    for seed in _reference_seeds(cfg):
        if rank == 0:
            print(f"B reference calibration seed={seed}")
        reference_cfg = _reference_cfg(cfg, seed)
        episode = _run_condition(
            condition_id="B_homogeneous_reference",
            condition_cfg=reference_cfg,
            action=_sham(reference_cfg, "B_homogeneous_reference"),
            stimulate=False,
            seed=int(seed),
            action_index=0,
            output_dir=root / "reference_B" / str(seed),
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(seed),
            drive_seed=int(seed),
            phase_seed=int(seed),
        )
        if rank == 0:
            baseline, _, _ = _episode_feature(episode, "baseline", cfg)
            outcome, _, _ = _episode_feature(episode, "stimulation", cfg)
            row = _epoch_row(episode, "stimulation")
            calibration_rows.append({
                "seed": int(seed),
                "baseline_log10_alpha_power": float(
                    baseline["log10_alpha_power_8_12_hz"]
                ),
                "outcome_log10_alpha_power": float(
                    outcome["log10_alpha_power_8_12_hz"]
                ),
                "E_firing_rate_hz": float(row.E_firing_rate_hz),
                "I_firing_rate_hz": float(row.I_firing_rate_hz),
            })
    if rank == 0:
        calibration = pd.DataFrame(calibration_rows)
        target = _fit_B_target(calibration, cfg)
        target["frozen_C1_threshold"] = float(sources["frozen_C1_threshold"])
        calibration.to_csv(root / "reference_B_calibration.csv", index=False)
        (root / "frozen_B_target.json").write_text(json.dumps(
            _json_ready(target), indent=2, allow_nan=False
        ))
    else:
        calibration, target = None, None
    target = comm.bcast(target, root=0)

    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    contexts = _run_context_specs(cfg)
    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"D={context['diffusion_rad2_per_s']:g} rad^2/s"
            )
        state_cfg = _with_diffusion_state(cfg, {
            "frequency_hz": float(context["hidden_frequency_hz"]),
            "phase_seed": int(context["phase_seed"]),
            "diffusion_rad2_per_s": float(context["diffusion_rad2_per_s"]),
        })
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
            screening = _context_features(baseline_reference, context, target, cfg)
            screening_rows.append(screening)
            eligible = bool(screening["eligible"])
            selected_frequency = float(screening["EEG_selected_frequency_hz"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'EXCLUDED'}; "
                f"C1={screening['context_C1']:.3f}; selected={selected_frequency:g} Hz; "
                f"reason={screening['exclusion_reasons']}"
            )
        else:
            screening, eligible, selected_frequency = None, None, None
        eligible = bool(comm.bcast(eligible, root=0))
        selected_frequency = float(comm.bcast(selected_frequency, root=0))
        if not eligible:
            continue
        action_cfg = _with_action_frequency(state_cfg, selected_frequency)
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations)):
            future_seed = _future_seed(cfg, context, future_index)
            episodes: dict[str, dict[str, Any]] | None = {} if rank == 0 else None
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
                metric_rows.extend(rows)
                trajectory_rows.extend(trajectories)
                update_rows.extend(updates)
        del baseline_reference

    if rank != 0:
        return
    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "prospective_screening.csv", index=False)
    if not metric_rows:
        conclusion = {
            "scope": "D1-R exploratory ideal-neural-EEG phase-maintenance audit",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "phase_refresh_mechanism_feasible": False,
                "ready_for_reliable_context_action_remapping": False,
                "contextual_bandit_status": "NOT TRAINED OR TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No context passed the prospective EEG screen.",
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(
            _json_ready(conclusion), indent=2, allow_nan=False
        ))
        print("\nNo eligible contexts; phase-refresh audit: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    updates = pd.DataFrame(update_rows)
    expected = _expected_map(metrics)
    comparison, structure, audit = _comparison_tables(expected, metrics)
    checks, conclusions = _checks(
        calibration=calibration,
        screening=screening,
        metrics=metrics,
        expected=expected,
        comparison=comparison,
        structure=structure,
        updates=updates,
        audit=audit,
        sources=sources,
        cfg=cfg,
    )
    metrics.to_csv(root / "context_controller_future_metrics.csv", index=False)
    expected.to_csv(root / "expected_context_controller_map.csv", index=False)
    comparison.to_csv(root / "context_refresh_comparison.csv", index=False)
    structure.to_csv(root / "structure_level_refresh_advantage.csv", index=False)
    trajectories.to_csv(root / "one_second_eeg_trajectories.csv", index=False)
    updates.to_csv(root / "causal_phase_updates.csv", index=False)
    provenance = {
        "experiment": "D1R_causal_phase_refresh_audit",
        "frozen_sources": {"roots": sources["roots"], "hashes": sources["hashes"]},
        "state_generator": {
            "frequencies_hz": [9.0, 11.0],
            "phase_diffusion_rad2_per_s": [0.5, 2.0],
            "modulation_depth": 0.04,
            "shared_latent_phase_private_Poisson_events": True,
        },
        "controllers": {
            "sham": "zero field",
            "one_time": "one pre-onset one-second EEG phase estimate; carrier runs open loop",
            "phase_refreshed": (
                "rolling one-second ideal EEG, updated every 250 ms; wrapped phase "
                "error corrected by bounded continuous frequency slew"
            ),
        },
        "fixed_active_amplitude_v_per_m": 0.2,
        "relative_phase_target_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
        "phase_estimation_history_ms": float(cfg.analysis.tacs.phase_estimation_history_ms),
        "refresh_interval_ms": float(cfg.analysis.tacs.refresh_interval_ms),
        "maximum_frequency_correction_hz": float(
            cfg.analysis.tacs.maximum_frequency_correction_hz
        ),
        "primary_endpoint": "four-second ideal-EEG absolute log-alpha distance to B",
        "statistical_unit": "independent circuit structure",
        "not_a_bandit_or_confirmatory_experiment": True,
        "not_a_disease_or_human_treatment_model": True,
        "concurrent_EEG_is_ideal_and_artifact_free": True,
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "D1-R exploratory ideal-neural-EEG causal phase-maintenance audit",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure; other axes are repeats",
        "inference_boundary": (
            "directional mechanism/reliability audit only; a positive result selects "
            "a controller for a new action-map discovery experiment"
        ),
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot):
        _plots(
            root=root,
            expected=expected,
            comparison=comparison,
            structure=structure,
            updates=updates,
            trajectories=trajectories,
            cfg=cfg,
        )
    print("\n### D1-R screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### D1-R causal phase-refresh checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nCausal phase-refresh mechanism: "
        f"{'PASSED' if conclusions['phase_refresh_mechanism_feasible'] else 'NOT PASSED'}"
    )
    print(
        "Ready for reliable context-action remapping: "
        f"{'YES' if conclusions['ready_for_reliable_context_action_remapping'] else 'NO'}"
    )
    print("Contextual bandit status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
