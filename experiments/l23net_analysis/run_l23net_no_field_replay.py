"""Run one independently constructed, bounded-memory L23Net no-field replay.

Two separate invocations of this program are compared by
``compare_l23net_no_field_replays.py``.  Reconstructing the Python/NEURON
process is intentional: replay validation must not reuse in-memory simulator
state.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import hydra
from mpi4py import MPI
import numpy as np
from omegaconf import DictConfig, OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from env.models.neuron.networkenv_online import (  # noqa: E402
    canonical_fixed_step_boundary,
    fixed_step_time_tolerance_ms,
    hoc_vector_to_numpy,
)
from env.models.neuron.streaming import OnlineTraceWriter  # noqa: E402
from experiments.l23net_analysis.profile_l23net_tacs_full_scale import (  # noqa: E402
    _distribution,
    _duration_samples,
    _global_population_counts,
    _maximum_recorded_vector_size,
    _memory_snapshot,
    _mpi_variables,
    _write_json_checkpoint,
)
from experiments.l23net_analysis.replay_validation import (  # noqa: E402
    REPORT_NAME,
    TRACE_NAME,
    canonical_json_sha256,
    trace_content_summary,
)


def _validate_configuration(cfg: DictConfig, mpi_size: int) -> tuple[int, int]:
    if str(cfg.env.name) != "hl23net":
        raise ValueError("The no-field replay gate requires env=hl23net.")
    require_full = bool(cfg.analysis.require_full_network)
    if require_full and bool(cfg.experiment.debug):
        raise ValueError("A full-network replay requires experiment.debug=false.")
    if str(cfg.analysis.condition) != "reference":
        raise ValueError("Gate G1A is frozen to analysis.condition=reference.")
    if bool(cfg.env.simulation.MDD):
        raise ValueError("The reference replay requires env.simulation.MDD=false.")
    if bool(cfg.env.simulation.DRUG):
        raise ValueError("The reference replay requires env.simulation.DRUG=false.")
    if not bool(cfg.env.network.syn_activity):
        raise ValueError("The historical L23Net protocol requires synaptic activity.")
    if bool(cfg.env.ts.apply):
        raise ValueError("The no-field replay requires env.ts.apply=false.")
    if str(cfg.env.online.waveform) != "sinusoidal":
        raise ValueError("The online path must retain waveform=sinusoidal.")
    if str(cfg.env.online.stimulation.parameterization) != "uniform_field":
        raise ValueError("The online path must retain uniform-field parameterization.")
    if str(cfg.env.online.temperature_mode) != "configured":
        raise ValueError("Scientific L23Net replay requires configured temperature.")
    if not np.isclose(float(cfg.env.network.celsius), 34.0, atol=1e-12):
        raise ValueError("Scientific L23Net replay requires 34 degrees C.")
    if require_full and not np.isclose(
        float(cfg.env.network.dt), 0.025, rtol=0.0, atol=1e-12
    ):
        raise ValueError("The full-network replay requires dt=0.025 ms.")

    configured_ranks = int(cfg.analysis.resource_request.mpi_ranks)
    if mpi_size != configured_ranks:
        raise ValueError(
            f"MPI size is {mpi_size}; the replay contract specifies "
            f"{configured_ranks}."
        )

    duration_ms = float(cfg.analysis.duration_ms)
    window_ms = float(cfg.analysis.window_ms)
    if not np.isclose(
        float(cfg.env.simulation.duration), duration_ms, rtol=0.0, atol=1e-9
    ):
        raise ValueError("env.simulation.duration must equal analysis.duration_ms.")
    if not np.isclose(
        float(cfg.env.simulation.obs_win_len), window_ms, rtol=0.0, atol=1e-9
    ):
        raise ValueError("env.simulation.obs_win_len must equal analysis.window_ms.")
    samples_per_window = _duration_samples(window_ms, float(cfg.env.network.dt))
    window_count = int(round(duration_ms / window_ms))
    if window_count <= 0 or not np.isclose(
        window_count * window_ms, duration_ms, rtol=0.0, atol=1e-9
    ):
        raise ValueError("duration_ms must be a positive multiple of window_ms.")
    return window_count, samples_per_window


def _package_versions() -> dict[str, str]:
    names = ("LFPy", "NEURON", "numpy", "scipy", "mpi4py", "h5py")
    return {name: importlib.metadata.version(name) for name in names}


def _replay_contract(cfg: DictConfig, mpi_size: int) -> dict[str, Any]:
    """Select only fields that can affect this trajectory or its interpretation."""
    return {
        "contract_version": 1,
        "circuit": str(cfg.env.name),
        "condition": str(cfg.analysis.condition),
        "experiment_seed": int(cfg.experiment.seed),
        "env_seed": int(cfg.analysis.env_seed),
        "debug": bool(cfg.experiment.debug),
        "mpi_ranks": int(mpi_size),
        "simulation": {
            "duration_ms": float(cfg.env.simulation.duration),
            "window_ms": float(cfg.env.simulation.obs_win_len),
            "MDD": bool(cfg.env.simulation.MDD),
            "DRUG": bool(cfg.env.simulation.DRUG),
        },
        "network": {
            "dt_ms": float(cfg.env.network.dt),
            "tstart_ms": float(cfg.env.network.tstart),
            "v_init_mV": float(cfg.env.network.v_init),
            "celsius": float(cfg.env.network.celsius),
            "syn_activity": bool(cfg.env.network.syn_activity),
        },
        "online": {
            "waveform": str(cfg.env.online.waveform),
            "temperature_mode": str(cfg.env.online.temperature_mode),
            "max_step_ms": float(cfg.env.online.max_step_ms),
            "parameterization": str(
                cfg.env.online.stimulation.parameterization
            ),
            "field_direction": [
                float(value)
                for value in cfg.env.online.stimulation.field_direction
            ],
        },
        "stimulation_enabled": bool(cfg.env.ts.apply),
        "eeg": OmegaConf.to_container(cfg.env.eeg, resolve=True),
        "package_versions": _package_versions(),
    }


def _update_text(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_array(
    digest: "hashlib._Hash",
    values,
    *,
    dtype,
) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))


def _local_structure_fingerprint(
    environment: OnlineNeuronEnv,
    mpi_variables: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint rank-local geometry, synapse layout, events and NetCons."""
    geometry = hashlib.sha256()
    synapse_layout = hashlib.sha256()
    afferent_events = hashlib.sha256()
    netcons = hashlib.sha256()
    local_cells = 0
    local_segments = 0
    local_synapses = 0
    local_afferent_vectors = 0
    local_afferent_events = 0

    network = environment.network
    for population_name in network.population_names:
        population = network.populations[population_name]
        for gid, cell in zip(population.gids, population.cells):
            local_cells += 1
            _update_text(geometry, str(population_name))
            _update_array(geometry, [int(gid)], dtype="<i8")
            for coordinate_name in ("x", "y", "z"):
                coordinates = np.asarray(getattr(cell, coordinate_name))
                local_segments = max(local_segments, 0)
                _update_text(geometry, coordinate_name)
                _update_array(geometry, coordinates, dtype="<f8")
            local_segments += int(getattr(cell, "totnsegs", len(cell.x)))

            synapse_indices = np.asarray(
                getattr(cell, "synidx", []), dtype=np.int64
            )
            local_synapses += int(synapse_indices.size)
            _update_text(synapse_layout, str(population_name))
            _update_array(synapse_layout, [int(gid)], dtype="<i8")
            _update_array(synapse_layout, synapse_indices, dtype="<i8")
            _update_array(
                synapse_layout,
                [len(getattr(cell, "netconsynapses", []))],
                dtype="<i8",
            )

            spike_time_vectors = list(getattr(cell, "_sptimeslist", []) or [])
            _update_text(afferent_events, str(population_name))
            _update_array(afferent_events, [int(gid)], dtype="<i8")
            _update_array(
                afferent_events, [len(spike_time_vectors)], dtype="<i8"
            )
            local_afferent_vectors += len(spike_time_vectors)
            for vector in spike_time_vectors:
                event_times = hoc_vector_to_numpy(vector)
                local_afferent_events += int(event_times.size)
                _update_array(afferent_events, event_times, dtype="<f8")

    hoc_netcons = list(getattr(network, "_hoc_netconlist", []) or [])
    for index, netcon in enumerate(hoc_netcons):
        try:
            source_gid = int(netcon.srcgid())
        except (AttributeError, TypeError, ValueError):
            source_gid = -1
        _update_array(netcons, [index, source_gid], dtype="<i8")
        _update_array(
            netcons,
            [float(netcon.weight[0]), float(netcon.delay)],
            dtype="<f8",
        )

    base_seed = int(mpi_variables["SEED"])
    rank = int(mpi_variables["RANK"])
    return {
        "rank": rank,
        "numpy_rank_seed": base_seed + rank,
        "local_cells": int(local_cells),
        "local_segments": int(local_segments),
        "local_synapses": int(local_synapses),
        "local_recurrent_netcons": int(len(hoc_netcons)),
        "local_afferent_vectors": int(local_afferent_vectors),
        "local_afferent_events": int(local_afferent_events),
        "geometry_sha256": geometry.hexdigest(),
        "synapse_layout_sha256": synapse_layout.hexdigest(),
        "afferent_events_sha256": afferent_events.hexdigest(),
        "recurrent_netcons_sha256": netcons.hexdigest(),
    }


def _global_structure_fingerprint(
    environment: OnlineNeuronEnv,
    mpi_variables: dict[str, Any],
) -> dict[str, Any] | None:
    comm = mpi_variables["COMM"]
    rows = comm.gather(
        _local_structure_fingerprint(environment, mpi_variables), root=0
    )
    if comm.Get_rank() != 0:
        return None
    rows.sort(key=lambda row: row["rank"])
    totals = {
        key: int(sum(row[key] for row in rows))
        for key in (
            "local_cells",
            "local_segments",
            "local_synapses",
            "local_recurrent_netcons",
            "local_afferent_vectors",
            "local_afferent_events",
        )
    }
    return {
        "global_sha256": canonical_json_sha256(rows),
        "totals": totals,
        "by_rank": rows,
    }


def _spike_summary(spikes: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    for population_name in sorted(spikes):
        values = spikes[population_name]
        times = np.asarray(values["times_ms"], dtype=np.float64)
        gids = np.asarray(values["gids"], dtype=np.int64)
        if times.shape != gids.shape:
            raise ValueError(f"Spike time/GID shapes differ for {population_name}.")
        _update_text(digest, population_name)
        _update_array(digest, gids, dtype="<i8")
        _update_array(digest, times, dtype="<f8")
        counts[population_name] = int(times.size)
    return {"counts": counts, "sha256": digest.hexdigest()}


def _seed_manifest(cfg: DictConfig, mpi_size: int) -> dict[str, Any]:
    experiment_seed = int(cfg.experiment.seed)
    env_seed = int(cfg.analysis.env_seed)
    global_seed = experiment_seed + env_seed
    base_seed = experiment_seed * 10_000 + env_seed
    return {
        "experiment_seed": experiment_seed,
        "env_seed": env_seed,
        "resolved_global_seed": global_seed,
        "resolved_base_seed": base_seed,
        "rank_numpy_seeds": [base_seed + rank for rank in range(mpi_size)],
        "warning": "Changing MPI rank count changes the rank-local random realization.",
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mpi_variables = _mpi_variables(cfg)
    comm = mpi_variables["COMM"]
    rank = int(mpi_variables["RANK"])
    mpi_size = int(mpi_variables["SIZE"])
    window_count, samples_per_window = _validate_configuration(cfg, mpi_size)
    dt_ms = float(cfg.env.network.dt)
    window_ms = float(cfg.analysis.window_ms)
    duration_ms = float(cfg.analysis.duration_ms)

    output_directory = Path(str(cfg.experiment.dir)).resolve()
    report_path = output_directory / REPORT_NAME
    trace_path = output_directory / TRACE_NAME
    if rank == 0:
        output_directory.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    contract = _replay_contract(cfg, mpi_size)
    run_start_s = time.perf_counter()
    report: dict[str, Any] | None = None
    if rank == 0:
        report = {
            "status": "building",
            "scope": (
                "Independent deterministic zero-field replay gate; not a "
                "reference-versus-reduced-inhibition or stimulation result."
            ),
            "errors": [],
            "configuration": OmegaConf.to_container(cfg, resolve=True),
            "replay_contract": contract,
            "replay_contract_sha256": canonical_json_sha256(contract),
            "seed_manifest": _seed_manifest(cfg, mpi_size),
            "mpi": {"size": mpi_size},
            "windows": [],
            "memory_snapshots": [],
            "completed_simulated_ms": 0.0,
            "limitations": [
                "This gate tests deterministic reconstruction only at the frozen MPI rank count.",
                "The ideal EEG excludes sensor noise, unrelated sources, and stimulation artifact.",
                "One reference circuit trajectory is not a biological or statistical replicate.",
                "The historical one-off subset stimulus just after 4000 ms is retained.",
            ],
        }
        _write_json_checkpoint(report_path, report)

    environment: OnlineNeuronEnv | None = None
    trace_writer: OnlineTraceWriter | None = None
    try:
        writer_error = None
        if rank == 0:
            try:
                trace_writer = OnlineTraceWriter(
                    trace_path,
                    stage_names=["no_field"],
                )
            except Exception as exc:
                writer_error = f"Could not create replay trace: {exc!r}"
        writer_error = comm.bcast(writer_error, root=0)
        if writer_error is not None:
            raise RuntimeError(writer_error)

        comm.Barrier()
        build_start_s = time.perf_counter()
        environment = OnlineNeuronEnv(
            cfg,
            mpi_variables,
            ENV_SEED=int(cfg.analysis.env_seed),
        )
        comm.Barrier()
        build_wall_s = comm.reduce(
            time.perf_counter() - build_start_s, op=MPI.MAX, root=0
        )
        population_counts = _global_population_counts(environment, comm)
        build_diagnostics = comm.gather(
            environment.network.online_diagnostics(), root=0
        )
        structure = _global_structure_fingerprint(environment, mpi_variables)
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
                    f"Population counts {population_counts} != {expected_sizes}."
                )
            local_cells = [
                int(values["local_cell_count"]) for values in build_diagnostics
            ]
            local_segments = [
                int(values["local_segment_count"])
                for values in build_diagnostics
            ]
            for rank_index, values in enumerate(build_diagnostics):
                if values["online_probe_names"] != ["current_dipole_moment"]:
                    report["errors"].append(
                        f"Rank {rank_index} evaluated unexpected probes "
                        f"{values['online_probe_names']}."
                    )
                if not np.isclose(
                    float(values["configured_celsius"]),
                    float(values["effective_h_celsius"]),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    report["errors"].append(
                        f"Rank {rank_index} configured/effective temperature differs."
                    )
                if _maximum_recorded_vector_size(
                    values, "soma_voltage_vector_sizes"
                ) != 0:
                    report["errors"].append(
                        f"Rank {rank_index} accumulated unused soma voltage."
                    )
            if bool(cfg.analysis.require_full_network) and any(
                value <= 0 for value in local_cells
            ):
                report["errors"].append("At least one full-network rank has no cells.")
            report["build"] = {
                "wall_s": float(build_wall_s),
                "population_counts": population_counts,
                "total_cells": int(sum(population_counts.values())),
                "local_cells": _distribution(local_cells),
                "local_segments": _distribution(local_segments),
                "configured_celsius": float(cfg.env.network.celsius),
                "effective_celsius_by_rank": _distribution(
                    [values["effective_h_celsius"] for values in build_diagnostics]
                ),
                "fixed_dt_ms": dt_ms,
            }
            report["structure"] = structure
            report["memory_snapshots"].append(snapshot)
            report["status"] = "running"
            _write_json_checkpoint(report_path, report)

        previous_sample_time: float | None = None
        integration_wall_s = 0.0
        for window_index in range(window_count):
            comm.Barrier()
            window_start_s = time.perf_counter()
            result = environment.step_online(
                {"ac_amplitude_v_per_m": 0.0, "frequency_hz": 0.0},
                duration_ms=window_ms,
                phase_continuous=True,
                ramp_ms=0.0,
            )
            comm.Barrier()
            window_wall_s = comm.reduce(
                time.perf_counter() - window_start_s,
                op=MPI.MAX,
                root=0,
            )
            current_extracellular = comm.gather(
                environment.stimulation_controller.max_abs_extracellular(
                    environment.network
                ),
                root=0,
            )
            diagnostics = comm.gather(
                environment.network.online_diagnostics(), root=0
            )
            completed_ms = float((window_index + 1) * window_ms)

            stream_error = None
            if rank == 0:
                integration_wall_s += float(window_wall_s)
                times = np.asarray(result["sample_times_ms"], dtype=np.float64)
                eeg = np.asarray(result["eeg_v"], dtype=np.float64)
                dipole = np.asarray(result["dipole_nA_um"], dtype=np.float64)
                stimulation = result["stimulation"]
                field_time = np.asarray(stimulation["time_ms"], dtype=np.float64)
                field = np.asarray(stimulation["field_v_per_m"], dtype=np.float64)
                window_errors: list[str] = []
                if int(result["sample_count"]) != samples_per_window:
                    window_errors.append(
                        f"window {window_index} returned {result['sample_count']} "
                        f"samples; expected {samples_per_window}."
                    )
                if times.shape != (samples_per_window,):
                    window_errors.append(
                        f"window {window_index} time shape is {times.shape}."
                    )
                if eeg.ndim != 2 or eeg.shape[-1] != samples_per_window:
                    window_errors.append(
                        f"window {window_index} EEG shape is {eeg.shape}."
                    )
                if dipole.shape != (3, samples_per_window):
                    window_errors.append(
                        f"window {window_index} dipole shape is {dipole.shape}."
                    )
                if not np.all(np.isfinite(eeg)) or not np.all(np.isfinite(dipole)):
                    window_errors.append(
                        f"window {window_index} contains non-finite output."
                    )
                if times.size > 1 and not np.all(np.diff(times) > 0.0):
                    window_errors.append(
                        f"window {window_index} time is not strictly increasing."
                    )
                if previous_sample_time is not None and times[0] <= previous_sample_time:
                    window_errors.append(
                        f"window {window_index} does not continue time strictly."
                    )
                previous_sample_time = float(times[-1])
                if bool(stimulation["enabled"]):
                    window_errors.append(
                        f"window {window_index} reports stimulation enabled."
                    )
                if np.count_nonzero(field) != 0:
                    window_errors.append(
                        f"window {window_index} generated a nonzero field."
                    )
                if not np.allclose(
                    current_extracellular, 0.0, rtol=0.0, atol=1e-15
                ):
                    window_errors.append(
                        f"window {window_index} left extracellular voltage."
                    )
                for rank_index, values in enumerate(diagnostics):
                    if _maximum_recorded_vector_size(
                        values, "spike_vector_sizes"
                    ) != 0:
                        window_errors.append(
                            f"window {window_index} rank {rank_index} did not "
                            "drain spike vectors."
                        )
                    if _maximum_recorded_vector_size(
                        values, "soma_voltage_vector_sizes"
                    ) != 0:
                        window_errors.append(
                            f"window {window_index} rank {rank_index} "
                            "accumulated unused soma voltage."
                        )
                before = result["diagnostics"]["before"]
                after = result["diagnostics"]["after"]
                if not np.isclose(
                    float(before["canonical_time_ms"]),
                    window_index * window_ms,
                    rtol=0.0,
                    atol=1e-12,
                ) or not np.isclose(
                    float(after["canonical_time_ms"]),
                    completed_ms,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    window_errors.append(
                        f"window {window_index} continuation boundary is incorrect."
                    )

                spikes = _spike_summary(result["spikes"])
                report["windows"].append(
                    {
                        "index": int(window_index),
                        "start_ms": float(window_index * window_ms),
                        "stop_ms": completed_ms,
                        "sample_count": int(result["sample_count"]),
                        "wall_s": float(window_wall_s),
                        "spikes": spikes,
                        "firing_rates": {
                            name: float(value)
                            for name, value in result["firing_rates"].items()
                        },
                        "errors": window_errors,
                    }
                )
                report["errors"].extend(window_errors)
                try:
                    trace_writer.append_window(
                        sample_time_ms=times,
                        eeg_v=eeg,
                        dipole_nA_um=dipole,
                        field_left_boundary_time_ms=field_time[:-1],
                        field_left_boundary_v_per_m=field[:-1],
                        stage_code=0,
                    )
                except Exception as exc:
                    stream_error = (
                        f"Could not stream replay window {window_index}: {exc!r}"
                    )

            stream_error = comm.bcast(stream_error, root=0)
            if stream_error is not None:
                raise RuntimeError(stream_error)
            del result
            gc.collect()
            snapshot = _memory_snapshot(
                comm=comm,
                environment=environment,
                stage=f"window_{window_index + 1}",
                simulated_ms=completed_ms,
                run_start_s=run_start_s,
            )
            if rank == 0:
                report["memory_snapshots"].append(snapshot)
                report["completed_simulated_ms"] = completed_ms
                _write_json_checkpoint(report_path, report)

        final_times = comm.gather(environment.network.current_time_ms, root=0)
        final_steps = comm.gather(environment.network.current_step_index, root=0)
        final_zero = comm.gather(
            environment.stimulation_controller.max_abs_extracellular(
                environment.network
            ),
            root=0,
        )
        if rank == 0:
            expected_samples = _duration_samples(duration_ms, dt_ms)
            if any(step != expected_samples for step in final_steps):
                report["errors"].append(
                    f"MPI final step counts differ from {expected_samples}: "
                    f"{final_steps}."
                )
            for rank_index, raw_time in enumerate(final_times):
                canonical_time, _ = canonical_fixed_step_boundary(
                    raw_time,
                    dt_ms,
                    name=f"rank {rank_index} final NEURON time",
                )
                if not np.isclose(
                    canonical_time, duration_ms, rtol=0.0, atol=1e-12
                ):
                    report["errors"].append(
                        f"Rank {rank_index} ended at {canonical_time} ms."
                    )
                if abs(float(raw_time) - duration_ms) > fixed_step_time_tolerance_ms(
                    duration_ms, dt_ms
                ):
                    report["errors"].append(
                        f"Rank {rank_index} raw clock drift is excessive."
                    )
            if not np.allclose(final_zero, 0.0, rtol=0.0, atol=1e-15):
                report["errors"].append("Final extracellular voltage is nonzero.")
            if trace_writer.committed_samples != expected_samples:
                report["errors"].append(
                    f"Committed samples {trace_writer.committed_samples} != "
                    f"{expected_samples}."
                )
            if trace_writer.committed_windows != window_count:
                report["errors"].append(
                    f"Committed windows {trace_writer.committed_windows} != "
                    f"{window_count}."
                )
            trace_writer.close()
            trace_summary = trace_content_summary(trace_path)
            report["performance"] = {
                "build_wall_s": float(build_wall_s),
                "integration_wall_s": float(integration_wall_s),
                "simulated_s": duration_ms / 1000.0,
                "wall_s_per_simulated_s": integration_wall_s
                / (duration_ms / 1000.0),
                "requested_ncpus": int(cfg.analysis.resource_request.ncpus),
                "requested_memory_gib": float(
                    cfg.analysis.resource_request.memory_gb
                ),
            }
            report["artifacts"] = {
                "report": str(report_path),
                "trace": str(trace_path),
                "trace_summary": trace_summary,
            }
            report["status"] = "passed" if not report["errors"] else "failed"
            _write_json_checkpoint(report_path, report)
            console = {
                "status": report["status"],
                "errors": report["errors"],
                "completed_simulated_ms": report["completed_simulated_ms"],
                "mpi": report["mpi"],
                "population_counts": report["build"]["population_counts"],
                "replay_contract_sha256": report["replay_contract_sha256"],
                "structure_sha256": report["structure"]["global_sha256"],
                "trace_content_sha256": trace_summary["content_sha256"],
                "performance": report["performance"],
                "artifacts": {
                    "report": str(report_path),
                    "trace": str(trace_path),
                },
            }
            print(json.dumps(console, indent=2, sort_keys=True), flush=True)

        status = comm.bcast(None if rank != 0 else report["status"], root=0)
        if status != "passed":
            raise RuntimeError("L23Net no-field replay failed validation.")
    except Exception as exc:
        if rank == 0:
            failure = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            if report is None:
                report = {"errors": [], "completed_simulated_ms": 0.0}
            report["status"] = "failed"
            report.setdefault("errors", []).append(
                f"{failure['exception_type']}: {failure['message']}"
            )
            report["failure"] = failure
            try:
                _write_json_checkpoint(report_path, report)
                _write_json_checkpoint(
                    output_directory / "failure_summary.json", failure
                )
                print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
            except Exception as checkpoint_error:
                print(
                    f"Could not persist replay failure: {checkpoint_error!r}",
                    flush=True,
                )
        raise SystemExit(1) from None
    finally:
        if trace_writer is not None:
            trace_writer.close()
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
