"""Profile a persistent, full-circuit L23Net uniform-field tACS episode.

The runner writes a JSON checkpoint after network construction and after every
simulation window.  It is intended to establish technical correctness,
runtime, and memory scaling on NCI.  A single circuit realization cannot
establish entrainment, efficacy, or a statistically reliable MDD effect.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
from typing import Any

import hydra
from mpi4py import MPI
import numpy as np
from omegaconf import DictConfig, OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from env.models.neuron.env_online import OnlineNeuronEnv
from env.models.neuron.stimulation import apply_raised_cosine_block_envelope


KIB_PER_GIB = 1024.0 * 1024.0


def _mpi_variables(cfg: DictConfig) -> dict[str, Any]:
    comm = MPI.COMM_WORLD
    seed = int(cfg.experiment.seed)
    return {
        "COMM": comm,
        "SIZE": int(comm.Get_size()),
        "RANK": int(comm.Get_rank()),
        "GLOBALSEED": seed,
        "SEED": seed * 10_000,
    }


def _duration_samples(duration_ms: float, dt_ms: float) -> int:
    samples = int(round(float(duration_ms) / float(dt_ms)))
    if not np.isclose(samples * dt_ms, duration_ms, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Duration {duration_ms} ms is not an integer multiple of "
            f"dt={dt_ms} ms."
        )
    return samples


def _positive_integer_windows(duration_ms: float, window_ms: float, name: str) -> int:
    windows = int(round(float(duration_ms) / float(window_ms)))
    if windows <= 0 or not np.isclose(
        windows * window_ms, duration_ms, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            f"analysis.timeline.{name} must be a positive integer multiple "
            "of analysis.timeline.window_ms."
        )
    return windows


def _read_proc_status_kib() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmPeak:")):
                    name, raw_value, _unit = line.split()[:3]
                    values[name.rstrip(":")] = int(raw_value)
    except OSError:
        pass
    return values


def _local_network_counts(environment: OnlineNeuronEnv | None) -> tuple[int, int]:
    if environment is None or environment.network is None:
        return 0, 0
    diagnostics = environment.network.online_diagnostics()
    return (
        int(diagnostics["local_cell_count"]),
        int(diagnostics["local_segment_count"]),
    )


def _memory_snapshot(
    *,
    comm,
    environment: OnlineNeuronEnv | None,
    stage: str,
    simulated_ms: float,
    run_start_s: float,
) -> dict[str, Any] | None:
    comm.Barrier()
    proc = _read_proc_status_kib()
    cells, segments = _local_network_counts(environment)
    row = {
        "rank": int(comm.Get_rank()),
        "hostname": socket.gethostname(),
        "rss_kib": int(proc.get("VmRSS", 0)),
        "hwm_kib": int(
            max(
                proc.get("VmHWM", 0),
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            )
        ),
        "virtual_kib": int(proc.get("VmSize", 0)),
        "local_cells": cells,
        "local_segments": segments,
        "elapsed_s": float(time.perf_counter() - run_start_s),
    }
    gathered = comm.gather(row, root=0)
    if comm.Get_rank() != 0:
        return None

    def distribution(key: str) -> dict[str, float]:
        data = np.asarray([entry[key] for entry in gathered], dtype=np.float64)
        return {
            "minimum": float(np.min(data)),
            "median": float(np.median(data)),
            "maximum": float(np.max(data)),
            "sum": float(np.sum(data)),
        }

    by_node: dict[str, dict[str, float | int]] = {}
    for entry in gathered:
        node = by_node.setdefault(
            entry["hostname"],
            {
                "ranks": 0,
                "rss_gib": 0.0,
                "hwm_gib": 0.0,
                "cells": 0,
                "segments": 0,
            },
        )
        node["ranks"] += 1
        node["rss_gib"] += entry["rss_kib"] / KIB_PER_GIB
        node["hwm_gib"] += entry["hwm_kib"] / KIB_PER_GIB
        node["cells"] += entry["local_cells"]
        node["segments"] += entry["local_segments"]

    return {
        "stage": str(stage),
        "simulated_ms": float(simulated_ms),
        "elapsed_s": distribution("elapsed_s"),
        "rss_gib": {
            key: value / KIB_PER_GIB
            for key, value in distribution("rss_kib").items()
        },
        "process_high_water_gib": {
            key: value / KIB_PER_GIB
            for key, value in distribution("hwm_kib").items()
        },
        "virtual_memory_gib": {
            key: value / KIB_PER_GIB
            for key, value in distribution("virtual_kib").items()
        },
        "local_cells": distribution("local_cells"),
        "local_segments": distribution("local_segments"),
        "by_node": by_node,
        "interpretation": (
            "Summed process RSS is an approximate in-job measure and may "
            "double-count shared pages. The PBS epilogue Memory Used value "
            "is authoritative for job-level peak memory."
        ),
    }


def _write_json_checkpoint(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _validate_configuration(cfg: DictConfig, mpi_size: int) -> dict[str, int]:
    if str(cfg.env.name) != "hl23net":
        raise ValueError("This profiler requires env=hl23net.")
    if bool(cfg.analysis.require_full_network) and bool(cfg.experiment.debug):
        raise ValueError("The full-network profile requires experiment.debug=false.")
    if not bool(cfg.env.simulation.MDD):
        raise ValueError("This baseline profile requires env.simulation.MDD=true.")
    if bool(cfg.env.simulation.DRUG):
        raise ValueError("This baseline profile requires env.simulation.DRUG=false.")
    if not bool(cfg.env.network.syn_activity):
        raise ValueError("Use env.network.syn_activity=true for provenance parity.")
    if not bool(cfg.env.ts.apply):
        raise ValueError("The full-network tACS profile requires env.ts.apply=true.")
    if str(cfg.env.online.waveform) != "sinusoidal":
        raise ValueError("The full-network profile requires a sinusoidal waveform.")
    if str(cfg.env.online.stimulation.parameterization) != "uniform_field":
        raise ValueError("The full-network profile requires a uniform field.")
    if bool(cfg.analysis.require_full_network) and not np.isclose(
        float(cfg.env.network.dt), 0.025, rtol=0.0, atol=1e-12
    ):
        raise ValueError("The full-network profile requires dt=0.025 ms.")
    if not np.isclose(float(cfg.env.network.celsius), 34.0, rtol=0.0, atol=1e-12):
        raise ValueError("The L23Net production configuration requires 34 C.")

    configured_ranks = int(cfg.analysis.resource_request.mpi_ranks)
    if bool(cfg.analysis.require_full_network) and mpi_size != configured_ranks:
        raise ValueError(
            f"MPI size is {mpi_size}, but the frozen profile requests "
            f"{configured_ranks} ranks."
        )

    timeline = cfg.analysis.timeline
    window_ms = float(timeline.window_ms)
    if not np.isclose(
        float(cfg.env.simulation.obs_win_len), window_ms, rtol=0.0, atol=1e-9
    ):
        raise ValueError("env.simulation.obs_win_len must equal timeline.window_ms.")
    _duration_samples(window_ms, float(cfg.env.network.dt))
    windows = {
        name: _positive_integer_windows(
            float(timeline[f"{name}_ms"]), window_ms, f"{name}_ms"
        )
        for name in ("burn_in", "baseline", "stimulation", "washout")
    }
    total_ms = sum(
        float(timeline[f"{name}_ms"])
        for name in ("burn_in", "baseline", "stimulation", "washout")
    )
    if not np.isclose(
        float(cfg.env.simulation.duration), total_ms, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "env.simulation.duration must equal burn-in + baseline + "
            f"stimulation + washout ({total_ms} ms)."
        )
    ramp_ms = float(timeline.block_ramp_ms)
    stimulation_ms = float(timeline.stimulation_ms)
    if ramp_ms < 0.0 or 2.0 * ramp_ms >= stimulation_ms:
        raise ValueError("The block ramp must leave a non-empty central interval.")
    return windows


def _global_population_counts(environment: OnlineNeuronEnv, comm) -> dict[str, int]:
    local = {
        str(name): len(getattr(environment.network.populations[name], "gids", []))
        for name in environment.network.population_names
    }
    gathered = comm.gather(local, root=0)
    if comm.Get_rank() != 0:
        return {}
    return {
        name: int(sum(rank_values[name] for rank_values in gathered))
        for name in local
    }


def _distribution(values) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "maximum": float(np.max(array)) if array.size else None,
        "all_positive": bool(np.all(array > 0.0)) if array.size else False,
    }


def _window_summary(
    *,
    result: dict[str, Any],
    stage: str,
    stage_window: int,
    duration_ms: float,
    dt_ms: float,
    wall_s: float,
    active_peak_by_rank: list[float],
    current_extracellular_by_rank: list[float],
    expected_active_field: np.ndarray | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    expected_samples = _duration_samples(duration_ms, dt_ms)
    eeg = np.asarray(result["eeg_v"], dtype=np.float64)
    sample_times = np.asarray(result["sample_times_ms"], dtype=np.float64)
    stimulation = result["stimulation"]
    field = np.asarray(stimulation["field_v_per_m"], dtype=np.float64)

    if int(result["sample_count"]) != expected_samples:
        errors.append(
            f"{stage}[{stage_window}] returned {result['sample_count']} samples; "
            f"expected {expected_samples}."
        )
    if sample_times.size != expected_samples or eeg.shape[-1] != expected_samples:
        errors.append(f"{stage}[{stage_window}] output shapes are inconsistent.")
    if not np.all(np.isfinite(eeg)):
        errors.append(f"{stage}[{stage_window}] EEG contains non-finite values.")
    if sample_times.size > 1 and not np.all(np.diff(sample_times) > 0.0):
        errors.append(f"{stage}[{stage_window}] times are not strictly increasing.")

    before = result["diagnostics"]["before"]
    after = result["diagnostics"]["after"]
    if not np.isclose(
        float(after["configured_celsius"]),
        float(after["effective_h_celsius"]),
        rtol=0.0,
        atol=1e-12,
    ):
        errors.append(f"{stage}[{stage_window}] configured/effective temperature differs.")
    if not np.isclose(
        float(before["h_t_ms"]), float(result["t_start_ms"]), atol=1e-9
    ) or not np.isclose(
        float(after["h_t_ms"]), float(result["t_stop_ms"]), atol=1e-9
    ):
        errors.append(f"{stage}[{stage_window}] continuation diagnostics disagree.")

    if expected_active_field is None:
        if np.count_nonzero(field) != 0:
            errors.append(f"{stage}[{stage_window}] sham field is not exactly zero.")
        if not np.allclose(
            current_extracellular_by_rank, 0.0, rtol=0.0, atol=1e-15
        ):
            errors.append(
                f"{stage}[{stage_window}] left non-zero extracellular voltage."
            )
    elif not np.allclose(field, expected_active_field, rtol=0.0, atol=1e-12):
        errors.append(f"{stage}[{stage_window}] field does not match requested sine.")

    return (
        {
            "stage": str(stage),
            "stage_window": int(stage_window),
            "t_start_ms": float(result["t_start_ms"]),
            "t_stop_ms": float(result["t_stop_ms"]),
            "duration_ms": float(duration_ms),
            "sample_count": int(result["sample_count"]),
            "wall_s": float(wall_s),
            "simulated_ms_per_wall_s": float(duration_ms / wall_s),
            "eeg_finite": bool(np.all(np.isfinite(eeg))),
            "eeg_rms_v": float(np.sqrt(np.mean(eeg * eeg))),
            "field_peak_v_per_m": float(np.max(np.abs(field))),
            "peak_abs_extracellular_assigned_mV": _distribution(
                active_peak_by_rank
            ),
            "end_abs_extracellular_mV": _distribution(
                current_extracellular_by_rank
            ),
            "firing_rates": {
                key: float(value)
                for key, value in (result.get("firing_rates") or {}).items()
            },
            "errors": errors,
        },
        errors,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mpi_variables = _mpi_variables(cfg)
    comm = mpi_variables["COMM"]
    rank = mpi_variables["RANK"]
    mpi_size = mpi_variables["SIZE"]
    windows_by_stage = _validate_configuration(cfg, mpi_size)
    window_ms = float(cfg.analysis.timeline.window_ms)
    dt_ms = float(cfg.env.network.dt)

    output_directory = Path(str(cfg.experiment.dir)).resolve()
    report_path = output_directory / "l23net_tacs_full_scale_profile.json"
    trace_path = output_directory / "l23net_tacs_full_scale_trace.npz"
    if rank == 0:
        output_directory.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    run_start_s = time.perf_counter()
    report: dict[str, Any] | None = None
    if rank == 0:
        report = {
            "status": "building",
            "scope": (
                "Full-network technical correctness and resource profile; "
                "not an efficacy or entrainment result."
            ),
            "errors": [],
            "configuration": OmegaConf.to_container(cfg, resolve=True),
            "mpi": {"size": mpi_size},
            "windows": [],
            "memory_snapshots": [],
            "completed_simulated_ms": 0.0,
            "limitations": [
                "The ideal EEG excludes stimulation artifact and sensor noise.",
                "One circuit seed is not a statistical replicate.",
                "Changing MPI rank count changes the current rank-local circuit RNG realization.",
                "env.network.syn_activity is retained for command provenance but L23Net constructs its OU/background mechanisms unconditionally.",
                "The established full L23Net STIM_PARAM schedule delivers one-off subset events just after 4000 ms, so the 4-8 s reference is post-transient but not a purely stationary spontaneous baseline.",
                "Legacy env.ts.method/type/electrodeParameters fields are ignored by this OnlineNeuronEnv uniform-field path; env.online defines the actuator.",
            ],
        }
        _write_json_checkpoint(report_path, report)

    environment: OnlineNeuronEnv | None = None
    eeg_parts: list[np.ndarray] = []
    dipole_parts: list[np.ndarray] = []
    sample_time_parts: list[np.ndarray] = []
    field_parts: list[np.ndarray] = []
    field_time_parts: list[np.ndarray] = []
    stage_code_parts: list[np.ndarray] = []
    stage_codes = {"burn_in": 0, "baseline": 1, "stimulation": 2, "washout": 3}

    try:
        comm.Barrier()
        build_start_s = time.perf_counter()
        environment = OnlineNeuronEnv(cfg, mpi_variables, ENV_SEED=0)
        comm.Barrier()
        build_wall_s = comm.reduce(
            time.perf_counter() - build_start_s, op=MPI.MAX, root=0
        )
        population_counts = _global_population_counts(environment, comm)
        build_diagnostics = comm.gather(
            environment.network.online_diagnostics(), root=0
        )
        snapshot = _memory_snapshot(
            comm=comm,
            environment=environment,
            stage="after_build",
            simulated_ms=0.0,
            run_start_s=run_start_s,
        )

        if rank == 0:
            expected_sizes = (
                {
                    str(name): int(value)
                    for name, value in cfg.analysis.expected_population_sizes.items()
                }
                if bool(cfg.analysis.require_full_network)
                else {
                    f"HL23{name}": int(value)
                    for name, value in cfg.env.debug_n_neurons.items()
                }
            )
            if population_counts != expected_sizes:
                report["errors"].append(
                    f"Population counts {population_counts} do not match {expected_sizes}."
                )
            local_cell_counts = [
                int(values["local_cell_count"]) for values in build_diagnostics
            ]
            local_segment_counts = [
                int(values["local_segment_count"]) for values in build_diagnostics
            ]
            if any(value <= 0 for value in local_cell_counts):
                report["errors"].append("At least one MPI rank has no cells.")
            if any(value <= 0 for value in local_segment_counts):
                report["errors"].append("At least one MPI rank has no segments.")
            report["status"] = "running"
            report["build"] = {
                "wall_s": float(build_wall_s),
                "population_counts": population_counts,
                "total_cells": int(sum(population_counts.values())),
                "local_cells": _distribution(local_cell_counts),
                "local_segments": _distribution(local_segment_counts),
                "configured_celsius": float(cfg.env.network.celsius),
                "effective_celsius_by_rank": _distribution(
                    [values["effective_h_celsius"] for values in build_diagnostics]
                ),
                "fixed_dt_ms": dt_ms,
            }
            report["memory_snapshots"].append(snapshot)
            _write_json_checkpoint(report_path, report)

        stimulation_start_ms = (
            float(cfg.analysis.timeline.burn_in_ms)
            + float(cfg.analysis.timeline.baseline_ms)
        )
        stimulation_stop_ms = (
            stimulation_start_ms
            + float(cfg.analysis.timeline.stimulation_ms)
        )
        previous_sample_time_ms: float | None = None
        previous_active_phase_stop: float | None = None
        completed_ms = 0.0

        for stage in ("burn_in", "baseline", "stimulation", "washout"):
            for stage_window in range(windows_by_stage[stage]):
                if stage == "stimulation":
                    action: dict[str, Any] = {
                        "ac_amplitude_v_per_m": float(
                            cfg.analysis.stimulation.amplitude_v_per_m
                        ),
                        "frequency_hz": float(
                            cfg.analysis.stimulation.frequency_hz
                        ),
                        "montage": str(cfg.analysis.stimulation.montage),
                    }
                    if stage_window == 0:
                        action["phase_rad"] = float(
                            cfg.analysis.stimulation.phase_rad
                        )
                    block_envelope = {
                        "start_ms": stimulation_start_ms,
                        "stop_ms": stimulation_stop_ms,
                        "ramp_ms": float(cfg.analysis.timeline.block_ramp_ms),
                    }
                else:
                    action = {
                        "ac_amplitude_v_per_m": 0.0,
                        "frequency_hz": 0.0,
                    }
                    block_envelope = None

                comm.Barrier()
                window_start_s = time.perf_counter()
                result = environment.step_online(
                    action,
                    duration_ms=window_ms,
                    phase_continuous=True,
                    ramp_ms=0.0,
                    block_envelope=block_envelope,
                )
                comm.Barrier()
                window_wall_s = comm.reduce(
                    time.perf_counter() - window_start_s,
                    op=MPI.MAX,
                    root=0,
                )
                local_peak_assigned = (
                    environment.stimulation_controller
                    .peak_abs_extracellular_assigned_mV()
                )
                peak_by_rank = comm.gather(
                    local_peak_assigned,
                    root=0,
                )
                current_by_rank = comm.gather(
                    environment.stimulation_controller.max_abs_extracellular(
                        environment.network
                    ),
                    root=0,
                )
                completed_ms += window_ms

                if rank == 0:
                    stimulation = result["stimulation"]
                    waveform_time = np.asarray(
                        stimulation["time_ms"], dtype=np.float64
                    )
                    field = np.asarray(
                        stimulation["field_v_per_m"], dtype=np.float64
                    )
                    if stage == "stimulation":
                        base_sine = float(
                            cfg.analysis.stimulation.amplitude_v_per_m
                        ) * np.sin(
                            float(stimulation["phase_start_rad"])
                            + 2.0
                            * np.pi
                            * float(cfg.analysis.stimulation.frequency_hz)
                            * (waveform_time - waveform_time[0])
                            / 1000.0
                        )
                        expected_field = apply_raised_cosine_block_envelope(
                            base_sine,
                            time_ms=waveform_time,
                            block_start_ms=stimulation_start_ms,
                            block_stop_ms=stimulation_stop_ms,
                            ramp_ms=float(cfg.analysis.timeline.block_ramp_ms),
                        )
                        if previous_active_phase_stop is not None and not np.isclose(
                            float(stimulation["phase_start_rad"]),
                            previous_active_phase_stop,
                            rtol=0.0,
                            atol=1e-12,
                        ):
                            report["errors"].append(
                                "The active sinusoid phase was discontinuous at "
                                f"stimulation window {stage_window}."
                            )
                        previous_active_phase_stop = float(
                            stimulation["phase_stop_rad"]
                        )
                    else:
                        expected_field = None

                    summary, window_errors = _window_summary(
                        result=result,
                        stage=stage,
                        stage_window=stage_window,
                        duration_ms=window_ms,
                        dt_ms=dt_ms,
                        wall_s=float(window_wall_s),
                        active_peak_by_rank=peak_by_rank,
                        current_extracellular_by_rank=current_by_rank,
                        expected_active_field=expected_field,
                    )
                    current_times = np.asarray(
                        result["sample_times_ms"], dtype=np.float64
                    )
                    if (
                        previous_sample_time_ms is not None
                        and current_times[0] <= previous_sample_time_ms
                    ):
                        window_errors.append(
                            f"{stage}[{stage_window}] did not continue time strictly."
                        )
                    previous_sample_time_ms = float(current_times[-1])
                    report["errors"].extend(window_errors)
                    report["windows"].append(summary)

                    sample_time_parts.append(current_times.copy())
                    eeg_parts.append(
                        np.asarray(result["eeg_v"], dtype=np.float64).copy()
                    )
                    dipole_parts.append(
                        np.asarray(result["probe_data"][1], dtype=np.float64).copy()
                    )
                    field_time_parts.append(waveform_time[:-1].copy())
                    field_parts.append(field[:-1].copy())
                    stage_code_parts.append(
                        np.full(
                            current_times.size,
                            stage_codes[stage],
                            dtype=np.int8,
                        )
                    )

                del result
                gc.collect()
                snapshot = _memory_snapshot(
                    comm=comm,
                    environment=environment,
                    stage=f"{stage}_{stage_window + 1}",
                    simulated_ms=completed_ms,
                    run_start_s=run_start_s,
                )
                if rank == 0:
                    report["memory_snapshots"].append(snapshot)
                    report["completed_simulated_ms"] = float(completed_ms)
                    _write_json_checkpoint(report_path, report)

        final_times = comm.gather(environment.network.current_time_ms, root=0)
        final_zero = comm.gather(
            environment.stimulation_controller.max_abs_extracellular(
                environment.network
            ),
            root=0,
        )
        if rank == 0:
            total_ms = float(cfg.env.simulation.duration)
            expected_total_samples = _duration_samples(total_ms, dt_ms)
            actual_total_samples = int(
                sum(window["sample_count"] for window in report["windows"])
            )
            if actual_total_samples != expected_total_samples:
                report["errors"].append(
                    f"Total samples {actual_total_samples} != {expected_total_samples}."
                )
            if not np.allclose(final_times, total_ms, rtol=0.0, atol=1e-9):
                report["errors"].append(
                    f"MPI final times do not all equal {total_ms} ms."
                )
            if not np.allclose(final_zero, 0.0, rtol=0.0, atol=1e-15):
                report["errors"].append("The final washout field is not exactly zero.")

            active_peaks = [
                window["peak_abs_extracellular_assigned_mV"]
                for window in report["windows"]
                if window["stage"] == "stimulation"
            ]
            if not active_peaks or any(
                not values["all_positive"] for values in active_peaks
            ):
                report["errors"].append(
                    "At least one occupied MPI rank did not receive active "
                    "extracellular polarization in every stimulation window."
                )

            integration_wall_s = float(
                sum(window["wall_s"] for window in report["windows"])
            )
            simulated_s = total_ms / 1000.0
            requested_memory_gib = float(
                cfg.analysis.resource_request.memory_gb
            )
            build_rss_gib = float(
                report["memory_snapshots"][0]["rss_gib"]["sum"]
            )
            final_rss_gib = float(
                report["memory_snapshots"][-1]["rss_gib"]["sum"]
            )
            persistent_growth_gib_per_sim_s = max(
                0.0, (final_rss_gib - build_rss_gib) / simulated_s
            )
            memory_projection_s = None
            if persistent_growth_gib_per_sim_s > 0.0:
                memory_projection_s = max(
                    0.0,
                    (0.8 * requested_memory_gib - build_rss_gib)
                    / persistent_growth_gib_per_sim_s,
                )
            report["performance"] = {
                "build_wall_s": float(report["build"]["wall_s"]),
                "integration_wall_s": integration_wall_s,
                "simulated_s": simulated_s,
                "wall_s_per_simulated_s": integration_wall_s / simulated_s,
                "simulated_s_per_wall_hour": (
                    simulated_s / integration_wall_s * 3600.0
                ),
                "requested_memory_gib": requested_memory_gib,
                "approximate_build_aggregate_rss_gib": build_rss_gib,
                "approximate_final_aggregate_rss_gib": final_rss_gib,
                "approximate_persistent_rss_growth_gib_per_simulated_s": (
                    persistent_growth_gib_per_sim_s
                ),
                "rough_80_percent_memory_projection_simulated_s": (
                    memory_projection_s
                ),
                "projection_warning": (
                    "The duration projection is a linear extrapolation from "
                    "process RSS and must be checked against the PBS job-level "
                    "Memory Used value. It is not a safe production limit."
                ),
            }

            np.savez_compressed(
                trace_path,
                sample_time_ms=np.concatenate(sample_time_parts),
                eeg_v=np.concatenate(eeg_parts, axis=-1),
                dipole_nA_um=np.concatenate(dipole_parts, axis=-1),
                field_left_boundary_time_ms=np.concatenate(field_time_parts),
                field_left_boundary_v_per_m=np.concatenate(field_parts),
                stage_code=np.concatenate(stage_code_parts),
                stage_names=np.asarray(
                    ["burn_in", "baseline", "stimulation", "washout"]
                ),
            )
            report["artifacts"] = {
                "report": str(report_path),
                "trace": str(trace_path),
            }
            report["status"] = "passed" if not report["errors"] else "failed"
            _write_json_checkpoint(report_path, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            print(f"Saved {report_path}")
            print(f"Saved {trace_path}")
    finally:
        if environment is not None:
            environment.close()

    if rank == 0 and report["status"] != "passed":
        raise SystemExit("Full-scale L23Net tACS profile failed validation.")


if __name__ == "__main__":
    main()
