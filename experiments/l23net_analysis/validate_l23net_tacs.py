"""Minimal serial/MPI validation of sinusoidal uniform-field tACS in L23Net.

This is a technical smoke test, not an efficacy or entrainment experiment.  It
uses one persistent NEURON episode with baseline, active-field, and washout
windows and checks waveform, units, geometry, timing, MPI participation, EEG
finiteness, temperature, and exact field removal.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import hydra
import numpy as np
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from env.models.neuron.env_online import OnlineNeuronEnv


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


def _local_projection_rows(environment: OnlineNeuronEnv, direction) -> list[dict]:
    """Describe each local cell's coupling span along the field direction."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    rows: list[dict] = []
    for population_name in environment.network.population_names:
        population = environment.network.populations[population_name]
        cells = list(getattr(population, "cells", None) or [])
        gids = list(getattr(population, "gids", []))
        if len(cells) != len(gids):
            raise RuntimeError(
                f"Population {population_name}: cell/GID count mismatch."
            )
        for cell, gid in zip(cells, gids):
            midpoints_um = np.column_stack(
                (
                    np.asarray(cell.x, dtype=np.float64).mean(axis=-1),
                    np.asarray(cell.y, dtype=np.float64).mean(axis=-1),
                    np.asarray(cell.z, dtype=np.float64).mean(axis=-1),
                )
            )
            projected_um = midpoints_um @ direction
            rows.append(
                {
                    "population": str(population_name),
                    "gid": int(gid),
                    "segment_count": int(midpoints_um.shape[0]),
                    "projected_span_um": float(np.ptp(projected_um)),
                }
            )
    return rows


def _duration_samples(duration_ms: float, dt_ms: float) -> int:
    samples = int(round(float(duration_ms) / float(dt_ms)))
    if not np.isclose(samples * dt_ms, duration_ms, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Window {duration_ms} ms is not an integer multiple of dt={dt_ms} ms."
        )
    return samples


def _root_report(
    *,
    cfg: DictConfig,
    mpi_diagnostics: list[dict],
    projection_rows: list[dict],
    baseline: dict,
    active: dict,
    washout: dict,
    active_extracellular_mV_by_rank: list[float],
    washout_extracellular_mV_by_rank: list[float],
    final_times_ms: list[float],
) -> dict[str, Any]:
    dt_ms = float(cfg.env.network.dt)
    expected_by_epoch = {
        "baseline": _duration_samples(cfg.analysis.baseline_ms, dt_ms),
        "stimulation": _duration_samples(cfg.analysis.stimulation_ms, dt_ms),
        "washout": _duration_samples(cfg.analysis.washout_ms, dt_ms),
    }
    episodes = {
        "baseline": baseline,
        "stimulation": active,
        "washout": washout,
    }

    errors: list[str] = []
    for name, episode in episodes.items():
        expected = expected_by_epoch[name]
        if int(episode["sample_count"]) != expected:
            errors.append(
                f"{name}: sample_count={episode['sample_count']}, expected={expected}."
            )
        if np.asarray(episode["eeg_v"]).shape[-1] != expected:
            errors.append(f"{name}: EEG sample count is not {expected}.")
        if not np.all(np.isfinite(episode["eeg_v"])):
            errors.append(f"{name}: EEG contains non-finite values.")
        configured = float(episode["diagnostics"]["after"]["configured_celsius"])
        effective = float(episode["diagnostics"]["after"]["effective_h_celsius"])
        if not np.isclose(configured, effective, rtol=0.0, atol=1e-12):
            errors.append(
                f"{name}: configured/effective temperatures differ "
                f"({configured} vs {effective} C)."
            )

    all_sample_times = np.concatenate(
        [np.asarray(episodes[name]["sample_times_ms"]) for name in episodes]
    )
    if not np.all(np.diff(all_sample_times) > 0.0):
        errors.append("Sample times are not strictly increasing across windows.")

    stimulation = active["stimulation"]
    field = np.asarray(stimulation["field_v_per_m"], dtype=np.float64)
    waveform_time_ms = np.asarray(stimulation["time_ms"], dtype=np.float64)
    expected_field = float(cfg.analysis.amplitude_v_per_m) * np.sin(
        float(cfg.analysis.phase_rad)
        + 2.0
        * np.pi
        * float(cfg.analysis.frequency_hz)
        * (waveform_time_ms - waveform_time_ms[0])
        / 1000.0
    )
    if field.size != expected_by_epoch["stimulation"] + 1:
        errors.append("The active waveform does not contain intervals plus endpoint.")
    if not np.allclose(field, expected_field, rtol=0.0, atol=1e-12):
        errors.append("The installed active field is not the requested sinusoid.")
    if stimulation["parameterization"] != "uniform_field":
        errors.append("The active action did not use uniform-field parameterization.")

    if not all(value > 0.0 for value in active_extracellular_mV_by_rank):
        errors.append(
            "At least one MPI rank with cells had no active extracellular voltage."
        )
    if not np.allclose(
        washout_extracellular_mV_by_rank, 0.0, rtol=0.0, atol=1e-15
    ):
        errors.append("Washout left a non-zero extracellular voltage.")

    local_cell_counts = [int(row["local_cell_count"]) for row in mpi_diagnostics]
    if any(count <= 0 for count in local_cell_counts):
        errors.append("At least one MPI rank received no cells in this debug test.")
    expected_cells = sum(int(value) for value in cfg.env.debug_n_neurons.values())
    if sum(local_cell_counts) != expected_cells:
        errors.append(
            f"MPI cell count is {sum(local_cell_counts)}, expected {expected_cells}."
        )
    if not np.allclose(final_times_ms, final_times_ms[0], rtol=0.0, atol=1e-12):
        errors.append("MPI ranks ended at different NEURON times.")

    spans = np.asarray(
        [row["projected_span_um"] for row in projection_rows], dtype=np.float64
    )
    if spans.size != expected_cells or not np.all(np.isfinite(spans)):
        errors.append("Projected morphology spans are missing or non-finite.")
    if spans.size and np.any(spans <= 0.0):
        errors.append("At least one cell has zero extent along the selected field.")

    span_summary: dict[str, dict[str, float]] = {}
    for population_name in cfg.env.debug_n_neurons:
        population_rows = [
            row
            for row in projection_rows
            if row["population"] == f"HL23{population_name}"
        ]
        population_spans = np.asarray(
            [row["projected_span_um"] for row in population_rows],
            dtype=np.float64,
        )
        if population_spans.size:
            span_summary[str(population_name)] = {
                "cell_count": int(population_spans.size),
                "minimum_um": float(np.min(population_spans)),
                "median_um": float(np.median(population_spans)),
                "maximum_um": float(np.max(population_spans)),
            }

    report = {
        "status": "passed" if not errors else "failed",
        "scope": "technical smoke test; no efficacy or entrainment inference",
        "errors": errors,
        "mpi": {
            "size": int(len(mpi_diagnostics)),
            "local_cell_counts": local_cell_counts,
            "final_times_ms": [float(value) for value in final_times_ms],
        },
        "simulation": {
            "dt_ms": dt_ms,
            "expected_samples_by_epoch": expected_by_epoch,
            "configured_celsius": float(cfg.env.network.celsius),
            "debug_neurons": OmegaConf.to_container(
                cfg.env.debug_n_neurons, resolve=True
            ),
        },
        "stimulation": {
            "waveform": "sinusoidal",
            "parameterization": stimulation["parameterization"],
            "amplitude_v_per_m": float(cfg.analysis.amplitude_v_per_m),
            "frequency_hz": float(cfg.analysis.frequency_hz),
            "phase_rad": float(cfg.analysis.phase_rad),
            "montage": str(cfg.analysis.montage),
            "field_direction": np.asarray(
                stimulation["field_direction"], dtype=np.float64
            ).tolist(),
            "waveform_samples_including_left_endpoint": int(field.size),
            "active_extracellular_mV_by_rank": [
                float(value) for value in active_extracellular_mV_by_rank
            ],
            "washout_extracellular_mV_by_rank": [
                float(value) for value in washout_extracellular_mV_by_rank
            ],
        },
        "geometry": {
            "projected_span_by_population": span_summary,
            "note": (
                "Span is morphology after LFPy rotation projected onto the "
                "configured tissue-field direction; it is not a response metric."
            ),
        },
        "eeg": {
            name: {
                "finite": bool(np.all(np.isfinite(episode["eeg_v"]))),
                "rms_v": float(
                    np.sqrt(np.mean(np.asarray(episode["eeg_v"]) ** 2))
                ),
            }
            for name, episode in episodes.items()
        },
    }
    return report


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mpi_variables = _mpi_variables(cfg)
    comm = mpi_variables["COMM"]
    rank = mpi_variables["RANK"]

    if str(cfg.env.name) != "hl23net":
        raise ValueError("This validation requires env=hl23net.")
    if bool(cfg.analysis.require_debug) and not bool(cfg.experiment.debug):
        raise ValueError(
            "The local validation config requires experiment.debug=true."
        )
    if str(cfg.env.online.waveform) != "sinusoidal":
        raise ValueError("This validation requires online.waveform=sinusoidal.")
    if str(cfg.env.online.stimulation.parameterization) != "uniform_field":
        raise ValueError(
            "This validation requires online.stimulation.parameterization="
            "uniform_field."
        )
    if not bool(cfg.env.ts.apply):
        raise ValueError("This validation requires env.ts.apply=true.")

    total_duration_ms = sum(
        float(value)
        for value in (
            cfg.analysis.baseline_ms,
            cfg.analysis.stimulation_ms,
            cfg.analysis.washout_ms,
        )
    )
    if not np.isclose(
        float(cfg.env.simulation.duration), total_duration_ms, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "env.simulation.duration must equal baseline + stimulation + washout "
            f"({total_duration_ms} ms)."
        )

    environment: OnlineNeuronEnv | None = None
    report = None
    try:
        environment = OnlineNeuronEnv(cfg, mpi_variables, ENV_SEED=0)
        mpi_diagnostics = comm.allgather(environment.network.online_diagnostics())

        direction = environment.field_montages[str(cfg.analysis.montage)]
        local_projection = _local_projection_rows(environment, direction)
        gathered_projection = comm.gather(local_projection, root=0)

        baseline = environment.step_online(
            [0.0, 0.0], duration_ms=float(cfg.analysis.baseline_ms)
        )
        active = environment.step_online(
            {
                "ac_amplitude_v_per_m": float(cfg.analysis.amplitude_v_per_m),
                "frequency_hz": float(cfg.analysis.frequency_hz),
                "phase_rad": float(cfg.analysis.phase_rad),
                "montage": str(cfg.analysis.montage),
            },
            duration_ms=float(cfg.analysis.stimulation_ms),
            ramp_ms=float(cfg.analysis.ramp_ms),
        )
        active_extracellular = comm.allgather(
            environment.stimulation_controller.max_abs_extracellular(
                environment.network
            )
        )
        washout = environment.step_online(
            [0.0, 0.0], duration_ms=float(cfg.analysis.washout_ms)
        )
        washout_extracellular = comm.allgather(
            environment.stimulation_controller.max_abs_extracellular(
                environment.network
            )
        )
        final_times_ms = comm.allgather(environment.network.current_time_ms)

        if rank == 0:
            projection_rows = [
                row for rank_rows in gathered_projection for row in rank_rows
            ]
            report = _root_report(
                cfg=cfg,
                mpi_diagnostics=mpi_diagnostics,
                projection_rows=projection_rows,
                baseline=baseline,
                active=active,
                washout=washout,
                active_extracellular_mV_by_rank=active_extracellular,
                washout_extracellular_mV_by_rank=washout_extracellular,
                final_times_ms=final_times_ms,
            )
            output_dir = Path(str(cfg.experiment.dir)).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "l23net_tacs_smoke.json"
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
            print(f"Saved {output_path}", flush=True)

        status = comm.bcast(None if rank != 0 else report["status"], root=0)
        if status != "passed":
            raise AssertionError("L23Net sinusoidal uniform-field smoke test failed.")
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
