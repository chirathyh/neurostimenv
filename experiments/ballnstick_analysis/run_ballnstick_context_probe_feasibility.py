"""CL1-P common-probe feasibility map for EEG-contextual tACS dosing.

Every eligible active replay receives the same EEG-relative 10-Hz, 0.2-V/m
probe.  Paired episodes then diverge only at the decision boundary: one keeps
0.2 V/m and the other smoothly transitions to 0.4 V/m.  A frozen mechanistic
rule asks whether a duration-matched probe EEG that has reached or crossed the
population B alpha target should avoid escalation.

This is full-information system identification for a later contextual bandit.
It is not RL, a depression model, or evidence about human treatment.  The EEG
is the ideal neural-only simulated signal.
"""

from __future__ import annotations

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


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    A_HIGH,
    B_LOW,
    _action,
    _condition_for_seed,
    _estimate_relative_field_phase,
    _feature_from_raw,
    _plain,
    _reference_phase,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (  # noqa: E402
    _complex_response_decomposition,
    _field_removal_status,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_screened_confirmation import (  # noqa: E402
    _load_frozen_candidate,
    _screen_phase_quality,
    _screening_decision,
)
from experiments.ballnstick_analysis.run_ballnstick_context_dose_feasibility import (  # noqa: E402
    _add_context_metadata,
    _context_specs,
    _phase_tracking,
    _seed_values,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _collect_epoch_spikes,
    _environment_action,
    _mpi_variables,
    _phase_locking_metrics,
    _phase_rng,
    _relative_rate_safe,
    _validate_online_outputs,
    _zero_action,
)


SHAM = "A_sham"
MAINTAIN = "A_probe_maintain_0p2"
ESCALATE = "A_probe_escalate_0p4"
ANALYSIS_EPOCHS = ("baseline", "probe", "decision", "washout")
ACTIVE_ARMS = (MAINTAIN, ESCALATE)


def _probe_timeline(cfg: DictConfig) -> list[tuple[str, int]]:
    timeline = cfg.analysis.timeline
    result = [
        ("burn_in", int(timeline.burn_in_steps)),
        ("baseline", int(timeline.baseline_steps)),
        ("probe", int(timeline.probe_steps)),
        ("decision", int(timeline.decision_steps)),
        ("washout", int(timeline.washout_steps)),
    ]
    if any(count <= 0 for _, count in result):
        raise ValueError("Every CL1-P epoch must contain at least one window.")
    return result


def _probe_target_seeds(cfg: DictConfig) -> list[int]:
    block = cfg.analysis.probe_target_calibration
    first = int(cfg.experiment.seed) + int(block.seed_offset)
    return [first + index for index in range(int(block.n_seeds))]


def _phase_seed(cfg: DictConfig) -> int:
    return int(cfg.experiment.seed) + int(
        cfg.analysis.crossed_design.fixed_phase_seed_offset
    )


def _arm_dose(arm: str, cfg: DictConfig) -> float:
    if arm == SHAM:
        return 0.0
    if arm == MAINTAIN:
        return float(cfg.analysis.actions.maintain_dose_v_per_m)
    if arm == ESCALATE:
        return float(cfg.analysis.actions.escalate_dose_v_per_m)
    raise ValueError(f"Unknown CL1-P arm {arm!r}.")


def _probe_rule_arm(signed_probe_error: float, *, threshold: float) -> str:
    """Frozen mechanistic rule; lower/equal error prevents dose escalation."""
    return MAINTAIN if float(signed_probe_error) <= float(threshold) else ESCALATE


def _validate_design(cfg: DictConfig, frozen: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("CL1-P requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("Every CL1-P arm requires inhibition_scale=1.")
    _probe_timeline(cfg)

    probe = float(cfg.analysis.actions.probe_dose_v_per_m)
    maintain = float(cfg.analysis.actions.maintain_dose_v_per_m)
    escalate = float(cfg.analysis.actions.escalate_dose_v_per_m)
    maximum = float(cfg.analysis.maximum_field_v_per_m)
    if not (0.0 < probe <= maintain < escalate <= maximum):
        raise ValueError(
            "CL1-P requires 0 < probe <= maintain < escalate <= maximum field."
        )
    if not np.isclose(probe, maintain) or not np.isclose(probe, 0.2):
        raise ValueError("The common probe and maintain action must remain 0.2 V/m.")
    if not np.isclose(escalate, 0.4):
        raise ValueError("The frozen fixed comparator must remain 0.4 V/m.")

    candidate = frozen["candidate"]
    if not np.isclose(float(candidate["selected_dose_v_per_m"]), escalate):
        raise ValueError("The escalation comparator must equal the frozen candidate.")
    if not np.isclose(float(candidate["frequency_hz"]), 10.0):
        raise ValueError("CL1-P retains the frozen 10-Hz frequency.")
    if not np.isclose(float(candidate["relative_phase_offset_rad"]), np.pi):
        raise ValueError("CL1-P retains the frozen EEG-relative 180-degree phase.")
    if str(candidate["montage"]) != str(cfg.analysis.tacs.axial_montage):
        raise ValueError("CL1-P retains the frozen axial montage.")
    if not np.isclose(float(frozen["target"]["selected_modulation_depth"]), 0.04):
        raise ValueError("CL1-P requires the frozen 0.04 elevated-alpha state.")

    timeline = cfg.analysis.timeline
    window_ms = float(cfg.env.simulation.obs_win_len)
    if int(timeline.baseline_steps) < 4 or int(timeline.probe_steps) < 3:
        raise ValueError("CL1-P requires at least 4-s baseline and 3-s probe epochs.")
    probe_ms = int(timeline.probe_steps) * window_ms
    decision_ms = int(timeline.decision_steps) * window_ms
    probe_trim = float(timeline.probe_analysis_trim_start_ms)
    decision_trim = (
        float(timeline.decision_analysis_trim_start_ms)
        + float(timeline.decision_analysis_trim_end_ms)
    )
    if probe_trim < float(timeline.block_ramp_ms) or probe_trim >= probe_ms:
        raise ValueError("The probe trim must remove the onset ramp and leave data.")
    if decision_trim >= decision_ms:
        raise ValueError("Decision trimming must leave analysis data.")
    transition = float(timeline.amplitude_transition_ms)
    if transition <= 0.0 or transition > window_ms:
        raise ValueError("The escalation transition must fit in its first window.")
    if float(timeline.decision_analysis_trim_start_ms) < transition:
        raise ValueError("Decision analysis must exclude the escalation transition.")

    calibration = set(_probe_target_seeds(cfg))
    structures = set(_seed_values(cfg, kind="structure"))
    drives = set(_seed_values(cfg, kind="drive"))
    phase = {_phase_seed(cfg)}
    trials = {
        int(context["trial_seed"]) for context in _context_specs(cfg)
    }
    namespaces = (calibration, structures, drives, phase, trials)
    if any(not values for values in namespaces):
        raise ValueError("Every CL1-P seed namespace must be nonempty.")
    if any(
        namespaces[i].intersection(namespaces[j])
        for i in range(len(namespaces))
        for j in range(i + 1, len(namespaces))
    ):
        raise ValueError("CL1-P seed namespaces must be disjoint.")
    if max(set.union(*namespaces)) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("CL1-P seeds are too large for seed * 10,000.")


def _episode_config(
    base_cfg: DictConfig, *, seed: int, output_dir: Path
) -> DictConfig:
    run_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    OmegaConf.set_struct(run_cfg, False)
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    n_steps = sum(count for _, count in _probe_timeline(run_cfg))
    with open_dict(run_cfg):
        run_cfg.experiment.seed = int(seed)
        run_cfg.experiment.dir = str(output_dir)
        run_cfg.env.simulation.duration = n_steps * window_ms
        run_cfg.env.network.inhibition_scale = float(
            run_cfg.analysis.inhibition_scale
        )
        run_cfg.env.ts.apply = True
        run_cfg.env.online.temperature_mode = "configured"
        run_cfg.env.online.stimulation.parameterization = "uniform_field"
        run_cfg.env.stimAmplitude_max = float(
            run_cfg.analysis.maximum_field_v_per_m
        )
    return run_cfg


def _simulate_probe_episode(
    base_cfg: DictConfig,
    *,
    arm: str,
    seed: int,
    output_dir: Path,
    comm: Any,
    size: int,
    rank: int,
    structure_seed: int,
    drive_seed: int,
) -> dict[str, Any] | None:
    run_cfg = _episode_config(base_cfg, seed=seed, output_dir=output_dir)
    np.random.seed(int(structure_seed) * 10_000 + rank)
    random.seed(int(structure_seed) * 10_000 + rank)
    environment = OnlineNeuronEnv(
        run_cfg,
        _mpi_variables(
            comm,
            size,
            rank,
            seed,
            structure_seed=structure_seed,
            drive_seed=drive_seed,
        ),
        ENV_SEED=0,
    )
    schedule = _probe_timeline(run_cfg)
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    pre_steps = int(run_cfg.analysis.timeline.burn_in_steps) + int(
        run_cfg.analysis.timeline.baseline_steps
    )
    block_start_ms = pre_steps * window_ms
    block_stop_ms = block_start_ms + (
        int(run_cfg.analysis.timeline.probe_steps)
        + int(run_cfg.analysis.timeline.decision_steps)
    ) * window_ms
    block_envelope = {
        "start_ms": block_start_ms,
        "stop_ms": block_stop_ms,
        "ramp_ms": float(run_cfg.analysis.timeline.block_ramp_ms),
    }
    zero = _zero_action(run_cfg)
    probe_action = _action(
        run_cfg,
        identifier="common_probe_0p2",
        role="common_probe",
        amplitude=float(run_cfg.analysis.actions.probe_dose_v_per_m),
        montage=str(run_cfg.analysis.tacs.axial_montage),
        relative_offset=float(
            run_cfg.analysis.frozen_candidate.expected_relative_phase_offset_rad
        ),
    )
    decision_action = _action(
        run_cfg,
        identifier=arm,
        role="sham" if arm == SHAM else "post_probe_counterfactual",
        amplitude=_arm_dose(arm, run_cfg),
        montage=str(run_cfg.analysis.tacs.axial_montage),
        relative_offset=(
            None
            if arm == SHAM
            else float(
                run_cfg.analysis.frozen_candidate.expected_relative_phase_offset_rad
            )
        ),
    )
    outputs = (
        {name: [] for name, _ in schedule} if rank == 0 else None
    )
    diagnostics: dict[str, float] | None = None
    realized_probe = dict(probe_action)
    realized_decision = dict(decision_action)
    final_residual_mV = float("nan")
    try:
        for epoch, count in schedule:
            if epoch == "probe" and arm != SHAM:
                if rank == 0:
                    diagnostics = _estimate_relative_field_phase(
                        outputs["baseline"],
                        simulator_fs_hz=1000.0 / float(run_cfg.env.network.dt),
                        block_start_ms=block_start_ms,
                        relative_offset_rad=float(
                            probe_action["eeg_relative_phase_offset_rad"]
                        ),
                        cfg=run_cfg,
                    )
                diagnostics = comm.bcast(diagnostics, root=0)
                realized_probe["phase_rad"] = float(diagnostics["phase_rad"])
                realized_decision["phase_rad"] = float(diagnostics["phase_rad"])

            for epoch_step in range(count):
                active_epoch = arm != SHAM and epoch in ("probe", "decision")
                transition_from = None
                transition_ms = 0.0
                if epoch == "probe" and arm != SHAM:
                    step_action = _environment_action(realized_probe)
                    if epoch_step > 0:
                        step_action.pop("phase_rad", None)
                elif epoch == "decision" and arm != SHAM:
                    step_action = _environment_action(realized_decision)
                    step_action.pop("phase_rad", None)
                    if arm == ESCALATE and epoch_step == 0:
                        transition_from = float(
                            run_cfg.analysis.actions.probe_dose_v_per_m
                        )
                        transition_ms = float(
                            run_cfg.analysis.timeline.amplitude_transition_ms
                        )
                else:
                    step_action = dict(zero)

                output = environment.step_online(
                    step_action,
                    duration_ms=window_ms,
                    phase_continuous=True,
                    ramp_ms=0.0,
                    block_envelope=block_envelope if active_epoch else None,
                    transition_from_ac_amplitude_v_per_m=transition_from,
                    amplitude_transition_ms=transition_ms,
                )
                if rank == 0:
                    outputs[epoch].append(output)
        if rank == 0:
            final_residual_mV = (
                environment.stimulation_controller.max_abs_extracellular(
                    environment.network
                )
            )
    finally:
        environment.close()

    if rank != 0:
        return None
    _validate_online_outputs(outputs)
    if final_residual_mV != 0.0:
        raise RuntimeError(
            "CL1-P washout left residual extracellular voltage: "
            f"{final_residual_mV:g} mV."
        )
    return {
        "seed": int(seed),
        "structure_seed": int(structure_seed),
        "drive_seed": int(drive_seed),
        "arm": arm,
        "action": realized_decision,
        "probe_action": realized_probe,
        "block_start_ms": float(block_start_ms),
        "block_stop_ms": float(block_stop_ms),
        "outputs_by_epoch": outputs,
        "final_residual_mV": float(final_residual_mV),
        **({} if diagnostics is None else diagnostics),
    }


def _epoch_raw(simulation: dict[str, Any], epoch: str) -> np.ndarray:
    return np.concatenate([
        np.asarray(output["eeg_v"], dtype=np.float64).reshape(-1)
        for output in simulation["outputs_by_epoch"][epoch]
    ])


def _analysis_raw(
    simulation: dict[str, Any], epoch: str, cfg: DictConfig
) -> tuple[np.ndarray, float]:
    raw = _epoch_raw(simulation, epoch)
    outputs = simulation["outputs_by_epoch"][epoch]
    start_ms = float(outputs[0]["t_start_ms"])
    dt_ms = float(cfg.env.network.dt)
    start_trim_ms = 0.0
    end_trim_ms = 0.0
    if epoch == "probe":
        start_trim_ms = float(
            cfg.analysis.timeline.probe_analysis_trim_start_ms
        )
    elif epoch == "decision":
        start_trim_ms = float(
            cfg.analysis.timeline.decision_analysis_trim_start_ms
        )
        end_trim_ms = float(
            cfg.analysis.timeline.decision_analysis_trim_end_ms
        )
    start_samples = int(round(start_trim_ms / dt_ms))
    end_samples = int(round(end_trim_ms / dt_ms))
    stop = None if end_samples == 0 else -end_samples
    trimmed = raw[start_samples:stop]
    if trimmed.size == 0:
        raise RuntimeError(f"CL1-P trimming removed the complete {epoch} epoch.")
    return trimmed, start_ms + start_samples * dt_ms


def _analyze_probe_episode(
    simulation: dict[str, Any],
    *,
    cfg: DictConfig,
    action_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    arm = str(simulation["arm"])
    simulator_fs_hz = 1000.0 / float(cfg.env.network.dt)
    rows: list[dict[str, Any]] = []
    raw_by_epoch: dict[str, np.ndarray] = {}
    spike_payload: dict[str, np.ndarray] = {}
    arm_index = {SHAM: 0, MAINTAIN: 1, ESCALATE: 2}[arm]
    for epoch_index, epoch in enumerate(ANALYSIS_EPOCHS):
        raw = _epoch_raw(simulation, epoch)
        analysis_raw, start_ms = _analysis_raw(simulation, epoch, cfg)
        features, processed, frequencies, psd = _feature_from_raw(
            analysis_raw,
            simulator_fs_hz=simulator_fs_hz,
            start_ms=start_ms,
            cfg=cfg,
        )
        outputs = simulation["outputs_by_epoch"][epoch]
        duration_s = sum(
            float(output["t_stop_ms"] - output["t_start_ms"])
            for output in outputs
        ) / 1000.0
        applied_dose = (
            float(cfg.analysis.actions.probe_dose_v_per_m)
            if arm != SHAM and epoch == "probe"
            else (_arm_dose(arm, cfg) if epoch == "decision" else 0.0)
        )
        row: dict[str, Any] = {
            "seed": int(simulation["seed"]),
            "structure_seed": int(simulation["structure_seed"]),
            "drive_seed": int(simulation["drive_seed"]),
            "arm": arm,
            "epoch": epoch,
            "applied_dose_v_per_m": applied_dose,
            "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
            "montage": str(cfg.analysis.tacs.axial_montage),
            "phase_rad": float(simulation["action"].get("phase_rad", 0.0)),
            **features,
        }
        for population_index, population_name in enumerate(("E", "I")):
            times, population_size = _collect_epoch_spikes(
                outputs, population_name
            )
            locking = _phase_locking_metrics(
                times,
                frequency_hz=float(cfg.analysis.tacs.frequency_hz),
                phase_origin_ms=float(simulation["block_start_ms"]),
                n_surrogates=int(cfg.analysis.phase_null.n_surrogates),
                rng=_phase_rng(
                    int(simulation["seed"]),
                    action_index,
                    epoch_index,
                    population_index,
                    arm_index,
                ),
            )
            row[f"{population_name}_firing_rate_hz"] = float(
                times.size / (population_size * duration_s)
            )
            for name, value in locking.items():
                row[f"{population_name}_{name}"] = value
            spike_payload[f"{epoch}_{population_name}_times_ms"] = times
        rows.append(row)
        raw_by_epoch[epoch] = raw
        if bool(cfg.analysis.save_raw_eeg):
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_dir / f"{epoch}_signals.npz",
                eeg_raw_v=raw,
                eeg_analysis_v=analysis_raw,
                eeg_preprocessed_v=processed,
                frequencies_hz=frequencies,
                psd_v2_per_hz=psd,
            )
    if bool(cfg.analysis.save_spikes):
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "spikes.npz", **spike_payload)
    return {
        "simulation": simulation,
        "epoch_rows": rows,
        "raw_by_epoch": raw_by_epoch,
        "simulator_fs_hz": simulator_fs_hz,
    }


def _run_episode(
    *,
    condition_cfg: DictConfig,
    arm: str,
    seed: int,
    action_index: int,
    output_dir: Path,
    comm: Any,
    size: int,
    rank: int,
    structure_seed: int,
    drive_seed: int,
) -> dict[str, Any] | None:
    simulation = _simulate_probe_episode(
        condition_cfg,
        arm=arm,
        seed=seed,
        output_dir=output_dir,
        comm=comm,
        size=size,
        rank=rank,
        structure_seed=structure_seed,
        drive_seed=drive_seed,
    )
    if rank != 0:
        return None
    return _analyze_probe_episode(
        simulation,
        cfg=condition_cfg,
        action_index=action_index,
        output_dir=output_dir / "analysis",
    )


def _row(episode: dict[str, Any], epoch: str) -> pd.Series:
    return pd.Series(next(
        value for value in episode["epoch_rows"] if value["epoch"] == epoch
    ))


def _screen_view(episode: dict[str, Any]) -> dict[str, Any]:
    decision = dict(_row(episode, "decision"))
    decision["epoch"] = "stimulation"
    return {**episode, "epoch_rows": [decision]}


def _probe_context(
    episode: dict[str, Any], *, probe_target: dict[str, Any], cfg: DictConfig
) -> dict[str, float]:
    simulation = episode["simulation"]
    probe_raw, probe_start_ms = _analysis_raw(simulation, "probe", cfg)
    baseline_raw = _epoch_raw(simulation, "baseline")
    if baseline_raw.size < probe_raw.size:
        raise RuntimeError("The baseline is shorter than the probe analysis window.")
    matched_baseline = baseline_raw[-probe_raw.size:]
    baseline_stop_ms = float(
        simulation["outputs_by_epoch"]["baseline"][-1]["t_stop_ms"]
    )
    baseline_start_ms = baseline_stop_ms - (
        matched_baseline.size * float(cfg.env.network.dt)
    )
    baseline_features, _, _, _ = _feature_from_raw(
        matched_baseline,
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        start_ms=baseline_start_ms,
        cfg=cfg,
    )
    probe_features, _, _, _ = _feature_from_raw(
        probe_raw,
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        start_ms=probe_start_ms,
        cfg=cfg,
    )
    baseline_alpha = float(baseline_features["log10_alpha_power_8_12_hz"])
    probe_alpha = float(probe_features["log10_alpha_power_8_12_hz"])
    suppression = baseline_alpha - probe_alpha
    dose = float(cfg.analysis.actions.probe_dose_v_per_m)
    return {
        "context_baseline_matched_log10_alpha": baseline_alpha,
        "context_probe_log10_alpha": probe_alpha,
        "context_probe_signed_error_to_B_log10": probe_alpha
        - float(probe_target["B_probe_mean_log10_alpha"]),
        "context_probe_alpha_suppression_log10": suppression,
        "context_probe_gain_log10_per_vpm": suppression / dose,
        "context_baseline_10hz_resultant_v": float(
            baseline_features["eeg_10hz_resultant_v"]
        ),
        "context_probe_10hz_resultant_v": float(
            probe_features["eeg_10hz_resultant_v"]
        ),
        "context_probe_10hz_resultant_reduction_v": float(
            baseline_features["eeg_10hz_resultant_v"]
            - probe_features["eeg_10hz_resultant_v"]
        ),
        "context_probe_analysis_duration_s": float(
            probe_features["analysis_duration_s"]
        ),
    }


def _context_action_rows(
    *,
    context: dict[str, Any],
    episodes: dict[str, dict[str, Any]],
    screening: dict[str, Any],
    screen_phase: dict[str, Any],
    probe_target: dict[str, Any],
    target_model: dict[str, Any],
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    sham = episodes[SHAM]
    maintain = episodes[MAINTAIN]
    escalate = episodes[ESCALATE]
    sham_decision = _row(sham, "decision")
    sham_baseline = _row(sham, "baseline")
    sham_washout = _row(sham, "washout")
    target = float(target_model["B_mean_log10_alpha"])
    initial_distance = abs(
        float(sham_decision.log10_alpha_power_8_12_hz) - target
    )
    context_features = _probe_context(
        maintain, probe_target=probe_target, cfg=cfg
    )
    paired_baseline_error = _relative_rms_error(
        maintain["raw_by_epoch"]["baseline"],
        escalate["raw_by_epoch"]["baseline"],
    )
    paired_probe_error = _relative_rms_error(
        maintain["raw_by_epoch"]["probe"],
        escalate["raw_by_epoch"]["probe"],
    )
    base = {
        **context,
        **context_features,
        "screen_margin_toward_A_log10": float(
            screening["screen_margin_toward_A_log10"]
        ),
        "screen_phase_split_error_deg": float(
            screening["screen_phase_split_error_deg"]
        ),
        "frozen_B_mean_log10_alpha": target,
        "duration_matched_B_probe_mean_log10_alpha": float(
            probe_target["B_probe_mean_log10_alpha"]
        ),
        "pre_action_distance_to_B_log10": initial_distance,
        "paired_baseline_relative_rms_error": paired_baseline_error,
        "paired_probe_relative_rms_error": paired_probe_error,
        "paired_predecision_relative_rms_error": _relative_rms_error(
            np.concatenate([
                maintain["raw_by_epoch"]["baseline"],
                maintain["raw_by_epoch"]["probe"],
            ]),
            np.concatenate([
                escalate["raw_by_epoch"]["baseline"],
                escalate["raw_by_epoch"]["probe"],
            ]),
        ),
    }
    rows = []
    for arm in ACTIVE_ARMS:
        episode = episodes[arm]
        decision = _row(episode, "decision")
        baseline = _row(episode, "baseline")
        washout = _row(episode, "washout")
        post = float(decision.log10_alpha_power_8_12_hz)
        distance = abs(post - target)
        suppression = float(
            sham_decision.log10_alpha_power_8_12_hz - post
        )
        washout_residual = float(
            (sham_washout.log10_alpha_power_8_12_hz
             - sham_baseline.log10_alpha_power_8_12_hz)
            - (washout.log10_alpha_power_8_12_hz
               - baseline.log10_alpha_power_8_12_hz)
        )
        recovered, tolerance = _field_removal_status(
            effect_log10=suppression,
            residual_log10=washout_residual,
            cfg=cfg,
        )
        phase = _phase_tracking(
            active_episode=episode,
            screen_phase=screen_phase,
            relative_offset=float(
                cfg.analysis.frozen_candidate.expected_relative_phase_offset_rad
            ),
            phase_seed=int(context["phase_seed"]),
        )
        decomposition = _complex_response_decomposition(
            sham_cosine=float(sham_decision.eeg_10hz_cosine_v),
            sham_sine=float(sham_decision.eeg_10hz_sine_v),
            active_cosine=float(decision.eeg_10hz_cosine_v),
            active_sine=float(decision.eeg_10hz_sine_v),
        )
        rows.append({
            **base,
            "arm": arm,
            "decision_dose_v_per_m": _arm_dose(arm, cfg),
            "post_log10_alpha_power": post,
            "post_signed_error_to_B_log10": post - target,
            "post_distance_to_B_log10": distance,
            "reward_negative_distance": -distance,
            "target_distance_improvement_log10": initial_distance - distance,
            "alpha_suppression_vs_sham_log10": suppression,
            "finishes_below_frozen_B": bool(post < target),
            "coherent_10hz_suppression_v": float(
                sham_decision.eeg_10hz_resultant_v
                - decision.eeg_10hz_resultant_v
            ),
            "alpha_peak_prominence_reduction_db": float(
                sham_decision.alpha_peak_prominence_db
                - decision.alpha_peak_prominence_db
            ),
            "E_ppc_reduction": float(sham_decision.E_ppc - decision.E_ppc),
            "I_ppc_reduction": float(sham_decision.I_ppc - decision.I_ppc),
            "E_rate_change_hz": float(
                decision.E_firing_rate_hz - sham_decision.E_firing_rate_hz
            ),
            "I_rate_change_hz": float(
                decision.I_firing_rate_hz - sham_decision.I_firing_rate_hz
            ),
            "rate_safe": bool(_relative_rate_safe(decision, sham_decision, cfg)),
            "field_removal_residual_log10": washout_residual,
            "field_removal_tolerance_log10": tolerance,
            "field_removal_recovered": recovered,
            "baseline_relative_rms_error_vs_sham": _relative_rms_error(
                sham["raw_by_epoch"]["baseline"],
                episode["raw_by_epoch"]["baseline"],
            ),
            **decomposition,
            **phase,
        })
    return rows


def _counterfactual_summary(
    metrics: pd.DataFrame, *, cfg: DictConfig
) -> pd.DataFrame:
    practical = float(cfg.analysis.criteria.practical_advantage_log10)
    threshold = float(
        cfg.analysis.actions.probe_stop_escalation_threshold_log10
    )
    rows = []
    for context_id, group in metrics.groupby("context_id", sort=False):
        maintain = group[group.arm.eq(MAINTAIN)].iloc[0]
        escalate = group[group.arm.eq(ESCALATE)].iloc[0]
        oracle = group.sort_values(
            ["post_distance_to_B_log10", "decision_dose_v_per_m"]
        ).iloc[0]
        rule_arm = _probe_rule_arm(
            float(maintain.context_probe_signed_error_to_B_log10),
            threshold=threshold,
        )
        selected = maintain if rule_arm == MAINTAIN else escalate
        oracle_advantage = float(
            escalate.post_distance_to_B_log10
            - oracle.post_distance_to_B_log10
        )
        rule_advantage = float(
            escalate.post_distance_to_B_log10
            - selected.post_distance_to_B_log10
        )
        rows.append({
            **{
                name: maintain[name]
                for name in (
                    "context_id", "context_order", "trial_seed",
                    "structure_seed", "drive_seed", "phase_seed",
                    "structure_index", "drive_index",
                    "context_baseline_matched_log10_alpha",
                    "context_probe_log10_alpha",
                    "context_probe_signed_error_to_B_log10",
                    "context_probe_alpha_suppression_log10",
                    "context_probe_gain_log10_per_vpm",
                    "context_baseline_10hz_resultant_v",
                    "context_probe_10hz_resultant_v",
                    "context_probe_10hz_resultant_reduction_v",
                    "pre_action_distance_to_B_log10",
                    "paired_predecision_relative_rms_error",
                )
            },
            "maintain_distance_to_B_log10": float(
                maintain.post_distance_to_B_log10
            ),
            "escalate_distance_to_B_log10": float(
                escalate.post_distance_to_B_log10
            ),
            "maintain_minus_escalate_distance_log10": float(
                maintain.post_distance_to_B_log10
                - escalate.post_distance_to_B_log10
            ),
            "oracle_arm": str(oracle.arm),
            "oracle_distance_to_B_log10": float(
                oracle.post_distance_to_B_log10
            ),
            "oracle_advantage_over_escalate_log10": oracle_advantage,
            "oracle_practically_prefers_maintain": bool(
                str(oracle.arm) == MAINTAIN and oracle_advantage >= practical
            ),
            "probe_rule_arm": rule_arm,
            "probe_rule_distance_to_B_log10": float(
                selected.post_distance_to_B_log10
            ),
            "probe_rule_advantage_over_escalate_log10": rule_advantage,
            "probe_rule_matches_oracle": bool(rule_arm == str(oracle.arm)),
        })
    return pd.DataFrame(rows)


def _context_shuffle_null(
    summary: pd.DataFrame, *, cfg: DictConfig
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.experiment.seed), 1_000_003])
    )
    errors = summary.context_probe_signed_error_to_B_log10.to_numpy(float)
    maintain = summary.maintain_distance_to_B_log10.to_numpy(float)
    escalate = summary.escalate_distance_to_B_log10.to_numpy(float)
    threshold = float(
        cfg.analysis.actions.probe_stop_escalation_threshold_log10
    )
    observed = float(summary.probe_rule_advantage_over_escalate_log10.mean())
    rows = []
    for permutation in range(int(cfg.analysis.context_shuffle.n_permutations)):
        shuffled = rng.permutation(errors)
        selected = np.where(shuffled <= threshold, maintain, escalate)
        rows.append({
            "permutation": permutation + 1,
            "mean_advantage_over_escalate_log10": float(
                np.mean(escalate - selected)
            ),
        })
    null = pd.DataFrame(rows)
    values = null.mean_advantage_over_escalate_log10.to_numpy(float)
    p_value = float(
        (1 + np.count_nonzero(values >= observed)) / (values.size + 1)
    )
    return null, {
        "observed_mean_advantage_log10": observed,
        "shuffled_mean_advantage_log10": float(values.mean()),
        "shuffled_95th_percentile_log10": float(np.quantile(values, 0.95)),
        "context_shuffle_p_value": p_value,
    }


def _experiment_checks(
    *,
    calibration: pd.DataFrame,
    calibration_seeds: list[int],
    screening: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    shuffle: dict[str, float],
    cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any], pd.DataFrame]:
    criteria = cfg.analysis.criteria
    active = metrics
    structure_level = summary.groupby("structure_seed", as_index=False)[
        "probe_rule_advantage_over_escalate_log10"
    ].mean()
    positive_structure_fraction = float(
        (structure_level.probe_rule_advantage_over_escalate_log10 > 0.0).mean()
    )
    oracle_mean = float(summary.oracle_advantage_over_escalate_log10.mean())
    practical = summary[summary.oracle_practically_prefers_maintain]
    practical_structures = int(practical.structure_seed.nunique())
    rule_mean = float(summary.probe_rule_advantage_over_escalate_log10.mean())
    checks = {
        "minimum_probe_target_calibration_seeds": len(calibration)
        >= int(criteria.minimum_probe_target_calibration_seeds),
        "probe_target_calibration_disjoint_from_contexts": set(
            calibration_seeds
        ).isdisjoint(set(summary.structure_seed) | set(summary.drive_seed)),
        "complete_crossed_screening_grid": len(screening)
        == int(cfg.analysis.crossed_design.n_structure_seeds)
        * int(cfg.analysis.crossed_design.n_drive_seeds),
        "screening_precedes_and_excludes_stimulation_outcomes": bool(
            (~screening.screening_uses_stimulation_outcome).all()
        ),
        "screening_does_not_use_seed_specific_B": bool(
            (~screening.screening_uses_seed_specific_B).all()
        ),
        "minimum_eligible_contexts": len(summary)
        >= int(criteria.minimum_eligible_contexts),
        "eligible_structure_coverage": summary.structure_seed.nunique()
        >= int(criteria.minimum_eligible_structure_seeds),
        "eligible_drive_coverage": summary.drive_seed.nunique()
        >= int(criteria.minimum_eligible_drive_seeds),
        "paired_history_identical_through_probe": bool(
            active.paired_predecision_relative_rms_error.max()
            <= float(criteria.maximum_predecision_relative_rms_error)
        ),
        "all_enrolled_phase_actionable": bool(active.phase_quality_pass.all()),
        "action_phase_tracks_screen_estimate": bool(
            active.action_phase_tracking_error_rad.max()
            <= float(criteria.maximum_phase_tracking_error_rad)
        ),
        "all_actions_rate_safe": bool(active.rate_safe.all()),
        "field_removal_recovered": bool(active.field_removal_recovered.all()),
        "baseline_causality": bool(
            active.baseline_relative_rms_error_vs_sham.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "coherent_decomposition_exact": bool(np.allclose(
            active.coherent_net_change_v2,
            active.coherent_interference_cross_term_v2
            + active.coherent_induced_component_v2,
            rtol=1.0e-10,
            atol=1.0e-30,
        )),
        "oracle_has_practical_contextual_opportunity": bool(
            oracle_mean >= float(criteria.minimum_mean_oracle_advantage_log10)
            and int(summary.oracle_practically_prefers_maintain.sum())
            >= int(criteria.minimum_practical_maintain_contexts)
            and practical_structures
            >= int(criteria.minimum_practical_maintain_structures)
        ),
        "frozen_probe_rule_uses_multiple_actions": summary.probe_rule_arm.nunique()
        >= int(criteria.minimum_probe_rule_selected_action_count),
        "frozen_probe_rule_beats_fixed_escalation": bool(
            rule_mean >= float(criteria.minimum_mean_probe_rule_advantage_log10)
            and positive_structure_fraction
            >= float(criteria.minimum_positive_structure_fraction)
        ),
        "probe_context_beats_shuffled_context": bool(
            rule_mean > float(shuffle["shuffled_mean_advantage_log10"])
            and float(shuffle["context_shuffle_p_value"])
            <= float(criteria.maximum_context_shuffle_p_value)
        ),
        "fixed_comparator_is_frozen_0p4": np.isclose(
            float(cfg.analysis.actions.escalate_dose_v_per_m), 0.4
        ),
    }
    primary = (
        "minimum_probe_target_calibration_seeds",
        "probe_target_calibration_disjoint_from_contexts",
        "complete_crossed_screening_grid",
        "screening_precedes_and_excludes_stimulation_outcomes",
        "screening_does_not_use_seed_specific_B",
        "minimum_eligible_contexts",
        "eligible_structure_coverage",
        "eligible_drive_coverage",
        "paired_history_identical_through_probe",
        "all_enrolled_phase_actionable",
        "action_phase_tracks_screen_estimate",
        "all_actions_rate_safe",
        "field_removal_recovered",
        "baseline_causality",
        "oracle_has_practical_contextual_opportunity",
        "frozen_probe_rule_uses_multiple_actions",
        "frozen_probe_rule_beats_fixed_escalation",
        "probe_context_beats_shuffled_context",
        "fixed_comparator_is_frozen_0p4",
    )
    conclusions = {
        "probe_response_contextual_feasibility_gate_passed": all(
            checks[name] for name in primary
        ),
        "scope": "directional paired CL1-P system identification; not RL",
        "screened_context_count": int(len(screening)),
        "eligible_context_count": int(len(summary)),
        "screening_yield": float(screening.eligible.mean()),
        "oracle_mean_advantage_over_fixed_escalation_log10": oracle_mean,
        "practical_maintain_context_count": int(
            summary.oracle_practically_prefers_maintain.sum()
        ),
        "practical_maintain_structure_count": practical_structures,
        "probe_rule_mean_advantage_over_fixed_escalation_log10": rule_mean,
        "probe_rule_positive_structure_fraction": positive_structure_fraction,
        "probe_rule_selected_arms": sorted(summary.probe_rule_arm.unique()),
        **shuffle,
        "ready_for_contextual_bandit": all(checks[name] for name in primary),
        "policy_observes_only_ideal_EEG": True,
        "hidden_spikes_and_rates_used_only_for_mechanism_and_safety": True,
    }
    return checks, conclusions, structure_level


def _decision_psd(
    episode: dict[str, Any], cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray]:
    raw, start_ms = _analysis_raw(episode["simulation"], "decision", cfg)
    _, _, frequencies, psd = _feature_from_raw(
        raw,
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        start_ms=start_ms,
        cfg=cfg,
    )
    return frequencies, psd


def _plot_results(
    *,
    root: Path,
    frequencies: np.ndarray,
    psds: dict[str, list[np.ndarray]],
    summary: pd.DataFrame,
) -> None:
    colors = {SHAM: "#9467BD", MAINTAIN: "#1F77B4", ESCALATE: "#D62728"}
    labels = {SHAM: "A sham", MAINTAIN: "probe + maintain 0.2 V/m", ESCALATE: "probe + escalate 0.4 V/m"}
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for arm in (SHAM, MAINTAIN, ESCALATE):
        mean_psd = np.mean(np.asarray(psds[arm]), axis=0)
        axis.plot(
            frequencies,
            10.0 * np.log10(np.maximum(mean_psd, np.finfo(float).tiny)),
            label=labels[arm],
            color=colors[arm],
            linewidth=2.0,
        )
    axis.axvspan(8.0, 12.0, color="gold", alpha=0.14)
    axis.set_xlim(2.0, 25.0)
    axis.set(
        xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)",
        title="CL1-P decision-period ideal neural EEG",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_probe_decision_psd.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    axis.scatter(
        summary.context_probe_signed_error_to_B_log10,
        summary.maintain_minus_escalate_distance_log10,
        c=summary.structure_index,
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.5,
    )
    for row in summary.itertuples():
        axis.annotate(row.context_id, (row.context_probe_signed_error_to_B_log10, row.maintain_minus_escalate_distance_log10), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.axvline(0.0, color="0.4", linewidth=0.9)
    axis.axhline(0.0, color="0.4", linewidth=0.9)
    axis.set(
        xlabel="Probe signed alpha error to duration-matched B (log10)",
        ylabel="Distance maintain 0.2 − escalate 0.4 (log10)",
        title="Does the common probe reveal the better dose?",
    )
    figure.tight_layout()
    figure.savefig(root / "figure_02_probe_context_interaction.png", dpi=250)
    plt.close(figure)

    ordered = summary.sort_values("context_order")
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.plot(x, ordered.escalate_distance_to_B_log10, "o-", label="Fixed escalate 0.4 V/m")
    axis.plot(x, ordered.probe_rule_distance_to_B_log10, "o-", label="Frozen probe rule")
    axis.plot(x, ordered.oracle_distance_to_B_log10, "o--", label="Counterfactual oracle")
    axis.set_xticks(x, ordered.context_id, rotation=45, ha="right")
    axis.set(
        ylabel="Absolute distance to frozen B (log10)",
        title="CL1-P adaptive opportunity versus fixed escalation",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_03_probe_policy_comparison.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    frozen = _load_frozen_candidate(cfg)
    _validate_design(cfg, frozen)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "context_probe_feasibility"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### CL1-P common-probe contextual feasibility")
        print(json.dumps(_plain(frozen), indent=2))
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()

    started = time.perf_counter()
    target_model = frozen["target"]["target_model"]
    depth = float(frozen["target"]["selected_modulation_depth"])
    calibration_seeds = _probe_target_seeds(cfg)
    calibration_rows: list[dict[str, Any]] = []
    for seed in calibration_seeds:
        if rank == 0:
            print(f"probe-target B calibration seed={seed}")
        b_cfg = _condition_for_seed(cfg, seed=seed, modulation_depth=0.0)
        episode = _run_episode(
            condition_cfg=b_cfg,
            arm=SHAM,
            seed=seed,
            action_index=0,
            output_dir=root / "calibration" / str(seed),
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=seed,
            drive_seed=seed,
        )
        if rank == 0:
            probe = _row(episode, "probe")
            decision = _row(episode, "decision")
            calibration_rows.append({
                "seed": seed,
                "probe_log10_alpha_power": float(
                    probe.log10_alpha_power_8_12_hz
                ),
                "probe_analysis_duration_s": float(probe.analysis_duration_s),
                "decision_log10_alpha_power": float(
                    decision.log10_alpha_power_8_12_hz
                ),
                "decision_distance_to_frozen_B_log10": abs(
                    float(decision.log10_alpha_power_8_12_hz)
                    - float(target_model["B_mean_log10_alpha"])
                ),
            })

    if rank == 0:
        calibration = pd.DataFrame(calibration_rows)
        probe_target = {
            "n_seeds": int(len(calibration)),
            "B_probe_mean_log10_alpha": float(
                calibration.probe_log10_alpha_power.mean()
            ),
            "B_probe_sd_log10_alpha": float(
                calibration.probe_log10_alpha_power.std(ddof=1)
                if len(calibration) > 1 else 0.0
            ),
            "probe_analysis_duration_s": float(
                calibration.probe_analysis_duration_s.iloc[0]
            ),
            "calibration_uses_disjoint_B_population_seeds": True,
            "calibration_uses_seed_specific_B": False,
        }
        calibration.to_csv(root / "probe_target_calibration.csv", index=False)
        (root / "frozen_probe_target.json").write_text(
            json.dumps(_plain(probe_target), indent=2)
        )
    else:
        calibration = None
        probe_target = None
    probe_target = comm.bcast(probe_target, root=0)

    contexts = _context_specs(cfg)
    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    psds = {arm: [] for arm in (SHAM, MAINTAIN, ESCALATE)}
    frequencies = None
    phase_seed = _phase_seed(cfg)
    relative_offset = float(
        cfg.analysis.frozen_candidate.expected_relative_phase_offset_rad
    )

    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} "
                f"structure={context['structure_seed']} drive={context['drive_seed']}"
            )
        a_cfg = _condition_for_seed(
            cfg, seed=phase_seed, modulation_depth=depth
        )
        sham = _run_episode(
            condition_cfg=a_cfg,
            arm=SHAM,
            seed=int(context["trial_seed"]),
            action_index=0,
            output_dir=root / "episodes" / context["context_id"] / SHAM,
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(context["structure_seed"]),
            drive_seed=int(context["drive_seed"]),
        )
        if rank == 0:
            _add_context_metadata(sham, context)
            screen_episode = _screen_view(sham)
            screen_phase = _screen_phase_quality(
                episode=screen_episode,
                relative_offset=relative_offset,
                cfg=cfg,
            )
            screening = _screening_decision(
                seed=int(context["trial_seed"]),
                screening_order=int(context["context_order"]),
                a_episode=screen_episode,
                phase_quality=screen_phase,
                target_model=target_model,
                cfg=cfg,
            )
            screening.update(context)
            screening_rows.append(screening)
            epoch_rows.extend(sham["epoch_rows"])
            eligible = bool(screening["eligible"])
            print(
                f"screen {context['context_id']}: "
                f"{'ELIGIBLE' if eligible else 'EXCLUDED'} "
                f"({screening['exclusion_reasons']})"
            )
        else:
            eligible = None
        eligible = bool(comm.bcast(eligible, root=0))
        if not eligible:
            continue

        episodes = {SHAM: sham} if rank == 0 else None
        for action_index, arm in enumerate(ACTIVE_ARMS, start=1):
            episode = _run_episode(
                condition_cfg=a_cfg,
                arm=arm,
                seed=int(context["trial_seed"]),
                action_index=action_index,
                output_dir=root / "episodes" / context["context_id"] / arm,
                comm=comm,
                size=size,
                rank=rank,
                structure_seed=int(context["structure_seed"]),
                drive_seed=int(context["drive_seed"]),
            )
            if rank == 0:
                _add_context_metadata(episode, context)
                episodes[arm] = episode
                epoch_rows.extend(episode["epoch_rows"])

        if rank == 0:
            metric_rows.extend(_context_action_rows(
                context=context,
                episodes=episodes,
                screening=screening_rows[-1],
                screen_phase=screen_phase,
                probe_target=probe_target,
                target_model=target_model,
                cfg=cfg,
            ))
            if bool(cfg.experiment.plot):
                for arm, episode in episodes.items():
                    frequencies, psd = _decision_psd(episode, cfg)
                    psds[arm].append(psd)
            del episodes

    if rank != 0:
        return

    screening_frame = pd.DataFrame(screening_rows)
    screening_frame.to_csv(root / "screening_audit.csv", index=False)
    if not metric_rows:
        result = {
            "scope": "directional paired CL1-P system identification",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "probe_response_contextual_feasibility_gate_passed": False,
                "eligible_context_count": 0,
                "ready_for_contextual_bandit": False,
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No crossed context passed the frozen EEG/phase screen.",
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_plain(result), indent=2)
        )
        print("\nNo eligible CL1-P contexts; feasibility gate: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    epochs = pd.DataFrame(epoch_rows)
    summary = _counterfactual_summary(metrics, cfg=cfg)
    shuffle_null, shuffle = _context_shuffle_null(summary, cfg=cfg)
    checks, conclusions, structure_level = _experiment_checks(
        calibration=calibration,
        calibration_seeds=calibration_seeds,
        screening=screening_frame,
        metrics=metrics,
        summary=summary,
        shuffle=shuffle,
        cfg=cfg,
    )

    epochs.to_csv(root / "context_epoch_eeg_and_hidden_metrics.csv", index=False)
    metrics.to_csv(root / "context_action_metrics.csv", index=False)
    summary.to_csv(root / "context_counterfactual_summary.csv", index=False)
    structure_level.to_csv(root / "structure_level_policy_comparison.csv", index=False)
    shuffle_null.to_csv(root / "probe_context_shuffle_null.csv", index=False)
    provenance = {
        **frozen,
        "probe_target": probe_target,
        "probe_target_calibration_seeds": calibration_seeds,
        "crossed_contexts": contexts,
        "common_probe_v_per_m": float(cfg.analysis.actions.probe_dose_v_per_m),
        "post_probe_actions_v_per_m": [
            float(cfg.analysis.actions.maintain_dose_v_per_m),
            float(cfg.analysis.actions.escalate_dose_v_per_m),
        ],
        "frozen_probe_rule": (
            "maintain 0.2 V/m when duration-matched probe signed error to "
            "population B is <= 0; otherwise escalate to 0.4 V/m"
        ),
        "paired_history_identical_until_decision": True,
        "selection_performed": "none; directional paired system identification",
        "requires_disjoint_bandit_confirmation": True,
    }
    (root / "frozen_protocol_provenance.json").write_text(
        json.dumps(_plain(provenance), indent=2)
    )
    result = {
        "scope": "ideal neural-only EEG, screen-positive crossed toy contexts",
        "checks": checks,
        "conclusions": conclusions,
        "primary_comparator": "common probe followed by fixed escalation to 0.4 V/m",
        "context": (
            "duration-matched ideal-EEG signed alpha error and measured "
            "0.2-V/m probe response"
        ),
        "reward": "negative absolute log10 alpha-power distance to frozen B mean",
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "A pass supports implementing a small contextual bandit on disjoint "
            "structure seeds. A failure means the common probe did not expose "
            "reproducible action heterogeneity beyond fixed escalation."
        ),
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_plain(result), indent=2)
    )
    if bool(cfg.experiment.plot):
        _plot_results(
            root=root,
            frequencies=np.asarray(frequencies),
            psds=psds,
            summary=summary,
        )

    print("\n### CL1-P screening")
    print(f"crossed contexts screened: {len(screening_frame)}")
    print(f"eligible contexts: {int(screening_frame.eligible.sum())}")
    print(f"screening yield: {float(screening_frame.eligible.mean()):.3f}")
    print("\n### CL1-P feasibility checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nProbe-response contextual feasibility gate:",
        "PASSED" if conclusions["probe_response_contextual_feasibility_gate_passed"]
        else "NOT PASSED",
    )
    print("Contextual bandit status: NOT TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
