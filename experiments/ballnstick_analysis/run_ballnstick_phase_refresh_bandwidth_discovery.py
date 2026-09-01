"""H4-BW causal phase-tracker bandwidth discovery for BallAndStick.

The completed D1-R experiment found a lower EEG-derived phase error and a
practical mean benefit for causal phase refreshing, but its active winner was
not sufficiently reproducible across independent stochastic futures.  This
small, disjoint discovery study therefore compares the frozen D1-R tracker
with two higher-bandwidth variants before any 12-structure H4 confirmation.

All active arms have an identical one-second phase initialization, 0.2-V/m
field, EEG-selected 9/11-Hz carrier, axial direction, and pi-relative phase
target.  Only post-onset phase-history length and refresh interval differ.  A
fixed correction horizon decouples feedback gain from observation cadence.
The field oscillator is corrected by bounded frequency slew and never jumps
phase.  This is controller selection, not confirmatory inference or a bandit.
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
from omegaconf import DictConfig, OmegaConf


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _epoch_raw,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (  # noqa: E402
    HIGH,
    LOW,
    _context_features,
    _future_seed,
    _relative_rate_safe,
    _run_context_specs,
    _with_diffusion_state,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (  # noqa: E402
    ONE_TIME,
    SHAM,
    _analyze_controller_episode,
    _controller_modes,
    _json_ready,
    _load_sources as _load_upstream_sources,
    _metric_rows,
    _phase_slew_frequency,
    _sha256,
    _signed_phase_error,
    _tail_phase_estimate,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _episode_config,
    _mpi_variables,
    _timeline,
    _validate_online_outputs,
    _zero_action,
)


CURRENT = "refresh_1000ms_250ms"
SHORT = "refresh_500ms_250ms"
FAST = "refresh_500ms_125ms"
EXPECTED_MODES = [SHAM, ONE_TIME, CURRENT, SHORT, FAST]


def _profile(cfg: DictConfig, mode: str) -> dict[str, Any]:
    value = cfg.analysis.controller_profiles[mode]
    return {
        "adaptive": bool(value.adaptive),
        "history_ms": float(value.history_ms),
        "update_interval_ms": float(value.update_interval_ms),
    }


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    """Hash-lock D0b/F0/D1 and the failed D1-R result and frozen target."""
    sources = _load_upstream_sources(cfg)
    root = Path(to_absolute_path(str(cfg.analysis.source_d1r.result_dir)))
    files = {
        "conclusion": root / "experiment_conclusion.json",
        "provenance": root / "protocol_and_provenance.json",
        "target": root / "frozen_B_target.json",
        "calibration": root / "reference_B_calibration.csv",
        "screening": root / "prospective_screening.csv",
        "metrics": root / "context_controller_future_metrics.csv",
        "updates": root / "causal_phase_updates.csv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen D1-R sources: {missing}")
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {
        name: str(cfg.analysis.source_d1r.expected_sha256[name]) for name in files
    }
    if observed != expected:
        raise RuntimeError(
            f"D1-R source hash mismatch: expected={expected}, observed={observed}"
        )
    conclusion = json.loads(files["conclusion"].read_text())
    failed = [
        name for name, passed in conclusion["checks"].items() if not bool(passed)
    ]
    if failed != ["realized_active_winner_reproducible_across_futures"]:
        raise RuntimeError(
            "H4-BW requires D1-R to have failed only future-wise active-winner "
            f"reproducibility; observed failed checks={failed}."
        )
    d1r = conclusion["conclusions"]
    if bool(d1r["phase_refresh_mechanism_feasible"]):
        raise RuntimeError("Frozen D1-R unexpectedly reports a passed mechanism.")
    if float(d1r["mean_realized_winner_agreement_fraction"]) >= float(
        cfg.analysis.criteria.minimum_realized_candidate_win_fraction
    ):
        raise RuntimeError("Frozen D1-R no longer records the motivating failure.")

    target = json.loads(files["target"].read_text())
    calibration = pd.read_csv(files["calibration"])
    for table_path in (files["calibration"], files["screening"], files["metrics"]):
        table = pd.read_csv(table_path)
        for column in (
            "seed", "structure_seed", "history_seed", "phase_seed",
            "trial_seed", "future_drive_seed",
        ):
            if column in table:
                sources["source_seed_union"].update(
                    table[column].dropna().astype(int).tolist()
                )
    sources["roots"]["d1r"] = str(root)
    sources["hashes"]["d1r"] = observed
    sources["target"] = target
    sources["calibration"] = calibration
    sources["D1R_failed_only_future_reproducibility"] = True
    return sources


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H4-BW requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H4-BW may not change recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H4-BW freezes the 9/11-Hz generator grid.")
    levels = [
        (str(x.label), float(x.diffusion_rad2_per_s))
        for x in cfg.analysis.states.phase_diffusion_levels
    ]
    if levels != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("H4-BW freezes D to 0.5 and 2 rad^2/s.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H4-BW freezes afferent modulation depth to 0.04.")
    if _controller_modes(cfg) != EXPECTED_MODES:
        raise ValueError(f"H4-BW controller order must be {EXPECTED_MODES}.")
    if [str(x) for x in cfg.analysis.selection.candidates] != [SHORT, FAST]:
        raise ValueError("H4-BW freezes the short-history candidate set.")
    if str(cfg.analysis.selection.benchmark_controller) != CURRENT:
        raise ValueError("H4-BW benchmark must be the frozen D1-R tracker.")
    amplitude = float(cfg.analysis.actions.amplitude_v_per_m)
    if not np.isclose(amplitude, 0.2) or amplitude > float(
        cfg.analysis.maximum_field_v_per_m
    ):
        raise ValueError("H4-BW freezes every active controller to 0.2 V/m.")
    if not np.isclose(
        _wrap_phase(float(cfg.analysis.tacs.relative_phase_offset_rad)), np.pi
    ):
        raise ValueError("H4-BW freezes the EEG-relative phase target to pi.")
    if not np.isclose(float(cfg.analysis.tacs.initialization_history_ms), 1000.0):
        raise ValueError("All active arms require the same one-second initialization.")
    if not np.isclose(float(cfg.analysis.tacs.correction_horizon_ms), 250.0):
        raise ValueError("H4-BW freezes a 250-ms correction horizon.")
    if not np.isclose(
        float(cfg.analysis.tacs.maximum_frequency_correction_hz), 2.0
    ):
        raise ValueError("H4-BW freezes the frequency-slew limit to +/-2 Hz.")
    expected_profiles = {
        SHAM: (False, 1000.0, 250.0),
        ONE_TIME: (False, 1000.0, 250.0),
        CURRENT: (True, 1000.0, 250.0),
        SHORT: (True, 500.0, 250.0),
        FAST: (True, 500.0, 125.0),
    }
    for mode, expected in expected_profiles.items():
        profile = _profile(cfg, mode)
        observed = (
            profile["adaptive"], profile["history_ms"],
            profile["update_interval_ms"],
        )
        if observed != expected:
            raise ValueError(
                f"H4-BW profile {mode} changed: expected={expected}, observed={observed}."
            )

    window_ms = float(cfg.env.simulation.obs_win_len)
    if not np.isclose(window_ms, 1000.0):
        raise ValueError("H4-BW requires 1000-ms outer online windows.")
    dt_ms = float(cfg.env.network.dt)
    stimulation_ms = int(cfg.analysis.timeline.stimulation_steps) * window_ms
    for mode in EXPECTED_MODES:
        profile = _profile(cfg, mode)
        refresh = profile["update_interval_ms"]
        history = profile["history_ms"]
        if refresh <= 0 or not np.isclose(stimulation_ms / refresh, round(stimulation_ms / refresh)):
            raise ValueError(f"{mode} update interval must divide stimulation duration.")
        if not np.isclose(history / dt_ms, round(history / dt_ms)):
            raise ValueError(f"{mode} phase history must align with simulation dt.")
    timeline = cfg.analysis.timeline
    minimum_baseline = 4 if bool(cfg.analysis.smoke_test) else 12
    if int(timeline.baseline_steps) < minimum_baseline:
        raise ValueError(f"H4-BW requires at least {minimum_baseline} baseline seconds.")
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    if trim_ms < float(timeline.block_ramp_ms) or 2.0 * trim_ms >= stimulation_ms:
        raise ValueError("H4-BW trim must contain both ramps and leave an endpoint.")
    endpoint_ms = stimulation_ms - 2.0 * trim_ms
    if not bool(cfg.analysis.smoke_test) and not np.isclose(endpoint_ms, 4000.0):
        raise ValueError("Full H4-BW freezes the four-second EEG endpoint.")
    if not np.isclose(endpoint_ms / 1000.0, round(endpoint_ms / 1000.0)):
        raise ValueError("H4-BW endpoint must split into complete one-second windows.")
    if not bool(cfg.analysis.smoke_test):
        if int(cfg.analysis.crossed_design.n_structure_seeds) < 3:
            raise ValueError("Full H4-BW requires at least three structures.")
        if int(cfg.analysis.crossed_design.n_future_continuations) < 4:
            raise ValueError("Full H4-BW requires at least four futures per arm.")

    contexts = _run_context_specs(cfg)
    namespaces = [
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
        raise ValueError("Every H4-BW seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H4-BW seed namespaces overlap.")
    if set().union(*namespaces).intersection(sources["source_seed_union"]):
        raise ValueError("H4-BW seeds overlap D0b, F0, D1, or D1-R sources.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H4-BW structure seed exceeds the uint32 mapping range.")


def _controller_action(cfg: DictConfig, mode: str) -> dict[str, Any]:
    profile = _profile(cfg, mode)
    return {
        "id": mode,
        "role": "H4_controller_bandwidth_discovery",
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


def _fixed_horizon_phase_slew(
    cfg: DictConfig, *, carrier_hz: float, target_phase_rad: float,
    oscillator_phase_rad: float,
) -> dict[str, float]:
    """Return the common-gain slew command used at every refresh cadence."""
    return _phase_slew_frequency(
        carrier_hz=carrier_hz,
        target_phase_rad=target_phase_rad,
        oscillator_phase_rad=oscillator_phase_rad,
        update_interval_ms=float(cfg.analysis.tacs.correction_horizon_ms),
        maximum_correction_hz=float(
            cfg.analysis.tacs.maximum_frequency_correction_hz
        ),
    )


def _simulate_controller_episode(
    base_cfg: DictConfig, *, context: dict[str, Any], future_seed: int,
    future_index: int, mode: str, output_dir: Path, comm: Any, size: int,
    rank: int,
) -> dict[str, Any] | None:
    """Run a persistent paired episode with a mode-specific causal tracker."""
    run_cfg = _episode_config(
        base_cfg, seed=int(context["trial_seed"]), output_dir=output_dir
    )
    profile = _profile(run_cfg, mode)
    window_ms = float(run_cfg.env.simulation.obs_win_len)
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
    update_rows: list[dict[str, Any]] | None = [] if rank == 0 else None
    all_preceding_outputs: list[dict[str, Any]] | None = [] if rank == 0 else None
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
            estimate_history_ms = (
                initialization_history_ms if update_index == 0 else history_ms
            )
            if rank == 0:
                estimate = _tail_phase_estimate(
                    all_preceding_outputs,
                    boundary_ms=boundary_ms,
                    history_ms=estimate_history_ms,
                    simulator_fs_hz=1000.0 / float(run_cfg.env.network.dt),
                    relative_offset_rad=float(
                        run_cfg.analysis.tacs.relative_phase_offset_rad
                    ),
                    cfg=run_cfg,
                )
                oscillator_phase = (
                    float(estimate["desired_field_phase_rad"])
                    if update_index == 0 else float(environment.phase_rad)
                )
                audit_history_ms = float(run_cfg.analysis.tacs.get(
                    "common_audit_history_ms", estimate_history_ms
                ))
                common_audit = _tail_phase_estimate(
                    all_preceding_outputs,
                    boundary_ms=boundary_ms,
                    history_ms=audit_history_ms,
                    simulator_fs_hz=1000.0 / float(run_cfg.env.network.dt),
                    relative_offset_rad=float(
                        run_cfg.analysis.tacs.relative_phase_offset_rad
                    ),
                    cfg=run_cfg,
                )
                # The correction horizon is fixed across controller profiles.
                # Passing refresh_ms here would double the feedback gain at
                # 125-ms updates and confound observation cadence with gain.
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
                        common_audit["desired_field_phase_rad"]
                    ),
                    "common_audit_estimated_eeg_phase_at_boundary_rad": float(
                        common_audit["estimated_eeg_phase_at_boundary_rad"]
                    ),
                    "common_audit_resultant_to_rms": float(
                        common_audit["resultant_to_rms"]
                    ),
                    "common_audit_phase_error_before_correction_rad": (
                        _signed_phase_error(
                            float(common_audit["desired_field_phase_rad"]),
                            oscillator_phase,
                        )
                    ),
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
        raise RuntimeError("H4-BW washout left residual extracellular voltage.")
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


def _augment_metric_rows(
    rows: list[dict[str, Any]], episodes: dict[str, dict[str, Any]], cfg: DictConfig,
) -> None:
    by_mode = {str(row["controller_mode"]): row for row in rows}
    threshold = float(cfg.analysis.screening.minimum_recent_resultant_to_rms)
    for mode, episode in episodes.items():
        updates = pd.DataFrame(episode["simulation"]["phase_updates"])
        post = updates.iloc[1:] if len(updates) > 1 else updates
        applied = post[post.phase_refresh_applied.astype(bool)]
        row = by_mode[mode]
        profile = _profile(cfg, mode)
        row.update({
            "configured_phase_history_ms": float(profile["history_ms"]),
            "configured_update_interval_ms": float(profile["update_interval_ms"]),
            "correction_horizon_ms": float(cfg.analysis.tacs.correction_horizon_ms),
            "post_onset_refresh_count": int(len(applied)),
            "mean_phase_resultant_to_rms": float(post.resultant_to_rms.mean()),
            "phase_estimate_actionable_fraction": float(
                np.mean(post.resultant_to_rms.to_numpy(float) >= threshold)
            ),
            "frequency_correction_saturation_fraction": (
                float(applied.frequency_correction_saturated.mean())
                if len(applied) else 0.0
            ),
        })


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
            all_rate_safe=("rate_safe", "all"),
            all_field_removal_recovered=("field_removal_recovered", "all"),
            mean_abs_phase_error_rad=(
                "mean_abs_phase_error_before_correction_rad", "mean"
            ),
            mean_phase_resultant_to_rms=("mean_phase_resultant_to_rms", "mean"),
            phase_estimate_actionable_fraction=(
                "phase_estimate_actionable_fraction", "mean"
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


def _comparison_tables(
    expected: pd.DataFrame, metrics: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = [
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "diffusion_rad2_per_s", "context_C1",
    ]
    values = [
        "expected_post_distance_to_B_log10", "future_sd_post_distance_log10",
        "mean_abs_phase_error_rad", "phase_estimate_actionable_fraction",
        "correction_saturation_fraction",
    ]
    pivot = expected[expected.controller_mode.ne(SHAM)].pivot(
        index=keys, columns="controller_mode", values=values
    ).reset_index()
    pivot.columns = [
        "_".join(str(x) for x in column if str(x))
        if isinstance(column, tuple) else str(column) for column in pivot.columns
    ]
    pivot = pivot.rename(columns={f"{key}_": key for key in keys})

    realized = metrics[metrics.controller_mode.ne(SHAM)].pivot(
        index=["context_id", "future_index"],
        columns="controller_mode", values="post_distance_to_B_log10",
    ).reset_index()
    comparison_rows: list[dict[str, Any]] = []
    for row in pivot.to_dict("records"):
        context_id = str(row["context_id"])
        one_distance = float(
            row[f"expected_post_distance_to_B_log10_{ONE_TIME}"]
        )
        current_distance = float(
            row[f"expected_post_distance_to_B_log10_{CURRENT}"]
        )
        future = realized[realized.context_id.eq(context_id)]
        for mode in (CURRENT, SHORT, FAST):
            candidate_distance = float(
                row[f"expected_post_distance_to_B_log10_{mode}"]
            )
            wins = future[mode].to_numpy(float) < future[ONE_TIME].to_numpy(float)
            comparison_rows.append({
                **{key: row[key] for key in keys},
                "controller_mode": mode,
                "selection_candidate": mode in (SHORT, FAST),
                "advantage_over_one_time_log10": one_distance - candidate_distance,
                "advantage_over_current_log10": current_distance - candidate_distance,
                "realized_candidate_win_fraction": float(np.mean(wins)),
                "future_sd_ratio_to_one_time": float(
                    row[f"future_sd_post_distance_log10_{mode}"]
                    / max(
                        float(row[f"future_sd_post_distance_log10_{ONE_TIME}"]),
                        np.finfo(float).tiny,
                    )
                ),
                "mean_abs_phase_error_rad": float(
                    row[f"mean_abs_phase_error_rad_{mode}"]
                ),
                "current_mean_abs_phase_error_rad": float(
                    row[f"mean_abs_phase_error_rad_{CURRENT}"]
                ),
                "phase_estimate_actionable_fraction": float(
                    row[f"phase_estimate_actionable_fraction_{mode}"]
                ),
                "correction_saturation_fraction": float(
                    row[f"correction_saturation_fraction_{mode}"]
                ),
            })
    comparison = pd.DataFrame(comparison_rows)
    structure = (
        comparison.groupby(["controller_mode", "structure_seed"], as_index=False)
        .agg(
            context_count=("context_id", "nunique"),
            mean_advantage_over_one_time_log10=(
                "advantage_over_one_time_log10", "mean"
            ),
            mean_advantage_over_current_log10=(
                "advantage_over_current_log10", "mean"
            ),
            mean_realized_candidate_win_fraction=(
                "realized_candidate_win_fraction", "mean"
            ),
        )
    )

    criteria = cfg.analysis.criteria
    candidate_rows: list[dict[str, Any]] = []
    for mode in (CURRENT, SHORT, FAST):
        context = comparison[comparison.controller_mode.eq(mode)]
        structures = structure[structure.controller_mode.eq(mode)]
        diffusion = context.groupby("label").advantage_over_one_time_log10.mean()
        mean_advantage = float(
            structures.mean_advantage_over_one_time_log10.mean()
        )
        mean_vs_current = float(
            structures.mean_advantage_over_current_log10.mean()
        )
        positive_fraction = float(np.mean(
            structures.mean_advantage_over_one_time_log10 > 0.0
        ))
        win_fraction = float(context.realized_candidate_win_fraction.mean())
        sd_ratio = float(context.future_sd_ratio_to_one_time.mean())
        phase_error = float(context.mean_abs_phase_error_rad.mean())
        current_phase_error = float(context.current_mean_abs_phase_error_rad.mean())
        actionable = float(context.phase_estimate_actionable_fraction.mean())
        gate_values = {
            "gate_practical_advantage_over_one_time": mean_advantage
            >= float(criteria.practical_advantage_over_one_time_log10),
            "gate_positive_across_structures": positive_fraction
            >= float(criteria.minimum_positive_structure_fraction),
            "gate_nonadverse_in_both_diffusion_levels": bool(
                len(diffusion) == 2 and (diffusion >= 0.0).all()
            ),
            "gate_realized_winner_reproducible": win_fraction
            >= float(criteria.minimum_realized_candidate_win_fraction),
            "gate_future_variance_not_increased": sd_ratio
            <= float(criteria.maximum_candidate_to_one_time_future_sd_ratio) + 1.0e-12,
            "gate_noninferior_to_current_endpoint": mean_vs_current
            >= -float(criteria.current_tracker_noninferiority_margin_log10),
            "gate_phase_error_lower_than_current": phase_error < current_phase_error,
            "gate_short_phase_estimate_actionable": actionable
            >= float(criteria.minimum_phase_estimate_actionable_fraction),
        }
        candidate_rows.append({
            "controller_mode": mode,
            "selection_candidate": mode in (SHORT, FAST),
            "mean_advantage_over_one_time_log10": mean_advantage,
            "mean_advantage_over_current_log10": mean_vs_current,
            "positive_structure_fraction": positive_fraction,
            "minimum_diffusion_advantage_log10": float(diffusion.min()),
            "mean_realized_candidate_win_fraction": win_fraction,
            "mean_future_sd_ratio_to_one_time": sd_ratio,
            "mean_abs_phase_error_rad": phase_error,
            "current_mean_abs_phase_error_rad": current_phase_error,
            "mean_phase_estimate_actionable_fraction": actionable,
            "mean_correction_saturation_fraction": float(
                context.correction_saturation_fraction.mean()
            ),
            **gate_values,
            "passes_bandwidth_gate": bool(
                mode in (SHORT, FAST) and all(gate_values.values())
            ),
        })
    summary = pd.DataFrame(candidate_rows)
    selection = _select_controller(summary, cfg)
    return comparison, structure, summary, selection


def _select_controller(summary: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    """Apply the frozen gate, efficacy ranking, and parsimony tie-break."""
    passing = summary[
        summary.selection_candidate.astype(bool)
        & summary.passes_bandwidth_gate.astype(bool)
    ].copy()
    if passing.empty:
        return {
            "selected_controller": None,
            "bandwidth_candidate_found": False,
            "selection_rule": (
                "pass all gates; maximize structure-mean advantage over one-time; "
                "within the frozen tie margin prefer the slower update"
            ),
        }
    best = float(passing.mean_advantage_over_one_time_log10.max())
    margin = float(cfg.analysis.selection.practical_tie_log10)
    tied = set(passing[
        passing.mean_advantage_over_one_time_log10 >= best - margin
    ].controller_mode.astype(str))
    preference = [str(value) for value in cfg.analysis.selection.tie_preference]
    selected = next(mode for mode in preference if mode in tied)
    row = passing[passing.controller_mode.eq(selected)].iloc[0]
    return {
        "selected_controller": selected,
        "bandwidth_candidate_found": True,
        "selected_profile": _profile(cfg, selected),
        "selected_mean_advantage_over_one_time_log10": float(
            row.mean_advantage_over_one_time_log10
        ),
        "selected_mean_realized_candidate_win_fraction": float(
            row.mean_realized_candidate_win_fraction
        ),
        "selection_rule": (
            "pass all gates; maximize structure-mean advantage over one-time; "
            "within 0.01 log10 prefer refresh_500ms_250ms"
        ),
    }


def _common_initialization(updates: pd.DataFrame) -> bool:
    initial = updates[
        updates.update_index.eq(0) & updates.controller_mode.ne(SHAM)
    ]
    if initial.empty:
        return False
    for _, group in initial.groupby(["context_id", "future_index"]):
        phases = group.desired_field_phase_rad.to_numpy(float)
        if len(phases) != len(EXPECTED_MODES) - 1:
            return False
        errors = np.angle(np.exp(1j * (phases - phases[0])))
        if np.max(np.abs(errors)) > 1.0e-10:
            return False
        if not np.allclose(group.phase_history_ms.to_numpy(float), 1000.0):
            return False
    return True


def _checks(
    *, screening: pd.DataFrame, metrics: pd.DataFrame, expected: pd.DataFrame,
    updates: pd.DataFrame, summary: pd.DataFrame, selection: dict[str, Any],
    sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    active = metrics[metrics.controller_mode.ne(SHAM)]
    refreshed = updates[
        updates.controller_mode.isin([CURRENT, SHORT, FAST])
        & updates.phase_refresh_applied.astype(bool)
    ]
    short_updates = updates[
        updates.controller_mode.isin([SHORT, FAST])
        & updates.update_index.gt(0)
    ]
    reference_rates = {
        "E": float(sources["target"]["reference_E_firing_rate_hz"]),
        "I": float(sources["target"]["reference_I_firing_rate_hz"]),
    }
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    rate_matched = all(
        abs(float(getattr(row, f"baseline_{population}_firing_rate_hz"))
            - reference_rates[population])
        <= tolerance * max(reference_rates[population], np.finfo(float).tiny)
        for row in eligible.itertuples() for population in ("E", "I")
    )
    contexts = _run_context_specs(cfg)
    expected_groups_complete = bool(
        len(expected)
        and expected.groupby("context_id").controller_mode.nunique().min()
        == len(EXPECTED_MODES)
    )
    checks = {
        "source_D1R_hash_locked_with_reliability_gate_failed": bool(
            sources["D1R_failed_only_future_reproducibility"]
        ),
        "bandwidth_seeds_disjoint_from_D0b_F0_D1_and_D1R": True,
        "frozen_B_target_loaded_without_recalibration": bool(
            sources["target"].get("target_is_population_reference_not_seed_specific")
            and len(sources["calibration"]) >= 12
        ),
        "complete_crossed_screening_grid": len(screening) == len(contexts),
        "screening_uses_only_predecision_ideal_EEG": bool(
            len(screening)
            and screening.screen_uses_only_predecision_ideal_EEG.all()
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
        "complete_controller_grid_for_enrolled_contexts": expected_groups_complete,
        "multiple_independent_postdecision_futures": bool(
            len(expected)
            and expected.n_futures.min() >= int(criteria.minimum_future_continuations)
        ) or bool(cfg.analysis.smoke_test),
        "identical_predecision_EEG_across_controllers_and_futures": bool(
            len(metrics)
            and metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "all_active_arms_use_identical_0p2_V_per_m": bool(
            len(active) and np.allclose(active.amplitude_v_per_m, 0.2)
        ),
        "all_active_arms_share_one_second_initialization": _common_initialization(
            updates
        ),
        "phase_updates_use_only_preceding_EEG": bool(
            len(updates)
            and updates.estimate_is_strictly_causal.all()
            and (updates.estimate_stop_ms - updates.boundary_ms).max()
            <= float(criteria.maximum_causal_timing_error_ms)
        ),
        "correction_horizon_fixed_across_refresh_rates": bool(
            len(refreshed)
            and np.allclose(
                refreshed.correction_horizon_ms,
                float(cfg.analysis.tacs.correction_horizon_ms),
            )
        ),
        "phase_correction_is_frequency_bounded": bool(
            len(refreshed)
            and refreshed.frequency_correction_hz.abs().max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "field_waveform_continuous_across_update_boundaries": bool(
            len(active)
            and active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
        ),
        "half_second_phase_estimates_remain_actionable": bool(
            len(short_updates)
            and np.mean(
                short_updates.resultant_to_rms.to_numpy(float)
                >= float(cfg.analysis.screening.minimum_recent_resultant_to_rms)
            ) >= float(criteria.minimum_phase_estimate_actionable_fraction)
        ),
        "all_actions_rate_safe": bool(len(metrics) and metrics.rate_safe.all()),
        "reference_rate_matched": bool(rate_matched),
        "field_removal_recovered": bool(
            len(metrics) and metrics.field_removal_recovered.all()
            and np.allclose(metrics.final_extracellular_residual_mV, 0.0)
        ),
        "at_least_one_short_history_candidate_passes_frozen_gate": bool(
            selection["bandwidth_candidate_found"]
        ),
        "selection_uses_no_hidden_state_or_spikes": bool(
            len(metrics) and (~metrics.policy_uses_hidden_state_or_spikes.astype(bool)).all()
        ),
    }
    design_gate = [
        "source_D1R_hash_locked_with_reliability_gate_failed",
        "bandwidth_seeds_disjoint_from_D0b_F0_D1_and_D1R",
        "frozen_B_target_loaded_without_recalibration",
        "complete_crossed_screening_grid",
        "screening_uses_only_predecision_ideal_EEG",
        "minimum_eligible_contexts",
        "minimum_independent_structures",
        "both_diffusion_levels_and_frequencies_enrolled",
        "frequency_identified_from_predecision_EEG",
        "complete_controller_grid_for_enrolled_contexts",
        "multiple_independent_postdecision_futures",
        "identical_predecision_EEG_across_controllers_and_futures",
        "all_active_arms_use_identical_0p2_V_per_m",
        "all_active_arms_share_one_second_initialization",
        "phase_updates_use_only_preceding_EEG",
        "correction_horizon_fixed_across_refresh_rates",
        "phase_correction_is_frequency_bounded",
        "field_waveform_continuous_across_update_boundaries",
        "half_second_phase_estimates_remain_actionable",
        "all_actions_rate_safe",
        "reference_rate_matched",
        "field_removal_recovered",
        "at_least_one_short_history_candidate_passes_frozen_gate",
        "selection_uses_no_hidden_state_or_spikes",
    ]
    ready = bool(
        all(checks[name] for name in design_gate) and not bool(cfg.analysis.smoke_test)
    )
    return checks, {
        **selection,
        "ready_for_disjoint_12_structure_H4_confirmation": ready,
        "contextual_bandit_status": "NOT TRAINED OR TESTED",
        "claim_scope": "exploratory ideal-neural-EEG controller selection",
        "candidate_summaries": summary.to_dict("records"),
    }


def _plots(
    *, root: Path, expected: pd.DataFrame, comparison: pd.DataFrame,
    structure: pd.DataFrame, summary: pd.DataFrame, updates: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> None:
    modes = EXPECTED_MODES
    colors = {
        SHAM: "0.55", ONE_TIME: "tab:orange", CURRENT: "tab:blue",
        SHORT: "tab:green", FAST: "tab:purple",
    }
    labels = {
        SHAM: "sham", ONE_TIME: "one-time", CURRENT: "1 s / 250 ms",
        SHORT: "0.5 s / 250 ms", FAST: "0.5 s / 125 ms",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, diffusion in zip(axes, (LOW, HIGH)):
        subset = expected[expected.label.eq(diffusion)]
        means = subset.groupby("controller_mode").expected_post_distance_to_B_log10.mean()
        axis.bar(
            np.arange(len(modes)), [means.get(mode, np.nan) for mode in modes],
            color=[colors[mode] for mode in modes],
        )
        axis.set_xticks(np.arange(len(modes)), [labels[x] for x in modes], rotation=25)
        axis.set_title(diffusion.replace("_", " "))
        axis.set_ylabel("Expected 4-s distance to B (log10)")
    figure.suptitle("H4 controller-bandwidth discovery")
    figure.tight_layout()
    figure.savefig(root / "figure_01_controller_outcomes.png", dpi=250)
    plt.close(figure)

    active = updates[updates.controller_mode.ne(SHAM)].copy()
    active["time_since_onset_s"] = active.groupby(
        ["context_id", "future_index", "controller_mode"]
    ).boundary_ms.transform(lambda x: (x - x.min()) / 1000.0)
    phase = active.groupby([
        "controller_mode", "time_since_onset_s"
    ]).phase_error_before_correction_rad.apply(
        lambda values: float(np.mean(np.abs(values)))
    ).reset_index(name="mean_abs_phase_error_rad")
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for mode in modes[1:]:
        subset = phase[phase.controller_mode.eq(mode)]
        axis.plot(
            subset.time_since_onset_s, subset.mean_abs_phase_error_rad,
            label=labels[mode], color=colors[mode], linewidth=1.5,
        )
    axis.set(xlabel="Time since tACS onset (s)", ylabel="Mean |phase error| (rad)")
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Causal phase-tracking error")
    figure.tight_layout()
    figure.savefig(root / "figure_02_phase_tracking.png", dpi=250)
    plt.close(figure)

    candidates = summary[summary.selection_candidate].copy()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = np.arange(len(candidates))
    axes[0].bar(
        positions, candidates.mean_advantage_over_one_time_log10,
        color=[colors[x] for x in candidates.controller_mode],
    )
    axes[0].axhline(0.01, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Advantage over one-time (log10)")
    axes[1].bar(
        positions, candidates.mean_realized_candidate_win_fraction,
        color=[colors[x] for x in candidates.controller_mode],
    )
    axes[1].axhline(0.75, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Paired future win fraction")
    for axis in axes:
        axis.set_xticks(positions, [labels[x] for x in candidates.controller_mode], rotation=20)
    figure.suptitle("Frozen candidate advancement criteria")
    figure.tight_layout()
    figure.savefig(root / "figure_03_candidate_gate.png", dpi=250)
    plt.close(figure)

    trajectory = trajectories.groupby([
        "controller_mode", "analysis_window_index"
    ]).distance_to_B_log10.mean().reset_index()
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for mode in modes:
        subset = trajectory[trajectory.controller_mode.eq(mode)]
        axis.plot(
            subset.analysis_window_index, subset.distance_to_B_log10,
            marker="o", label=labels[mode], color=colors[mode],
        )
    axis.set(
        xlabel="One-second endpoint window", ylabel="Mean distance to B (log10)",
        xticks=sorted(trajectory.analysis_window_index.unique()),
    )
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Within-intervention EEG trajectory")
    figure.tight_layout()
    figure.savefig(root / "figure_04_eeg_trajectory.png", dpi=250)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    for mode in (CURRENT, SHORT, FAST):
        subset = structure[structure.controller_mode.eq(mode)]
        axis.scatter(
            np.full(len(subset), modes.index(mode)),
            subset.mean_advantage_over_one_time_log10,
            color=colors[mode], label=labels[mode], s=45,
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(
        [modes.index(x) for x in (CURRENT, SHORT, FAST)],
        [labels[x] for x in (CURRENT, SHORT, FAST)],
    )
    axis.set_ylabel("Structure-level advantage over one-time (log10)")
    axis.set_title("Independent-structure consistency")
    figure.tight_layout()
    figure.savefig(root / "figure_05_structure_advantage.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "phase_refresh_bandwidth_discovery"
    if rank == 0:
        exists = bool(root.exists() and any(root.iterdir()))
    else:
        exists = None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H4-BW causal controller-bandwidth discovery")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    target = sources["target"]
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
                _augment_metric_rows(rows, episodes, cfg)
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
            "scope": "H4-BW exploratory ideal-neural-EEG controller selection",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "selected_controller": None,
                "ready_for_disjoint_12_structure_H4_confirmation": False,
                "contextual_bandit_status": "NOT TRAINED OR TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No context passed the prospective EEG screen.",
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(
            _json_ready(conclusion), indent=2, allow_nan=False
        ))
        print("\nNo eligible contexts; H4 bandwidth discovery: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    updates = pd.DataFrame(update_rows)
    expected = _expected_map(metrics)
    comparison, structure, summary, selection = _comparison_tables(
        expected, metrics, cfg
    )
    checks, conclusions = _checks(
        screening=screening,
        metrics=metrics,
        expected=expected,
        updates=updates,
        summary=summary,
        selection=selection,
        sources=sources,
        cfg=cfg,
    )
    metrics.to_csv(root / "context_controller_future_metrics.csv", index=False)
    expected.to_csv(root / "expected_context_controller_map.csv", index=False)
    comparison.to_csv(root / "controller_comparison_by_context.csv", index=False)
    structure.to_csv(root / "structure_level_controller_advantage.csv", index=False)
    summary.to_csv(root / "controller_selection_summary.csv", index=False)
    trajectories.to_csv(root / "one_second_eeg_trajectories.csv", index=False)
    updates.to_csv(root / "causal_phase_updates.csv", index=False)
    frozen = {
        **selection,
        "selected_profile": (
            _profile(cfg, str(selection["selected_controller"]))
            if selection["selected_controller"] is not None else None
        ),
        "initialization_history_ms": float(cfg.analysis.tacs.initialization_history_ms),
        "correction_horizon_ms": float(cfg.analysis.tacs.correction_horizon_ms),
        "maximum_frequency_correction_hz": float(
            cfg.analysis.tacs.maximum_frequency_correction_hz
        ),
        "amplitude_v_per_m": float(cfg.analysis.actions.amplitude_v_per_m),
        "relative_phase_offset_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
        "montage": str(cfg.analysis.tacs.axial_montage),
        "requires_disjoint_H4_confirmation": True,
    }
    (root / "frozen_controller_candidate.json").write_text(json.dumps(
        _json_ready(frozen), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H4BW_causal_phase_tracker_bandwidth_discovery",
        "frozen_sources": {"roots": sources["roots"], "hashes": sources["hashes"]},
        "frozen_B_target_source": sources["roots"]["d1r"],
        "state_generator": {
            "frequencies_hz": [9.0, 11.0],
            "phase_diffusion_rad2_per_s": [0.5, 2.0],
            "modulation_depth": 0.04,
            "shared_latent_phase_private_Poisson_events": True,
        },
        "controller_profiles": {
            mode: _profile(cfg, mode) for mode in EXPECTED_MODES
        },
        "all_active_arms_share_one_second_initialization": True,
        "fixed_correction_horizon_decouples_gain_from_update_interval": True,
        "fixed_active_amplitude_v_per_m": 0.2,
        "relative_phase_target_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
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
        "scope": "H4-BW exploratory ideal-neural-EEG controller selection",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure; other axes are repeats",
        "inference_boundary": (
            "discovery only; a selected controller must be hash-frozen and tested "
            "once on disjoint 12-structure H4 confirmation seeds"
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
            summary=summary,
            updates=updates,
            trajectories=trajectories,
        )
    print("\n### H4-BW screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### H4-BW controller-bandwidth checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Candidate summaries")
    print(summary.to_string(index=False))
    print(
        "\nBandwidth candidate selected: "
        f"{conclusions['selected_controller'] or 'NONE'}"
    )
    print(
        "Ready for disjoint 12-structure H4 confirmation: "
        f"{'YES' if conclusions['ready_for_disjoint_12_structure_H4_confirmation'] else 'NO'}"
    )
    print("Contextual bandit status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
