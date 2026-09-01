"""Test reversible weak-field entrainment in the BallAndStick network.

T1 is an acute A-B-A experiment in one persistent online episode:

    A: sham baseline
    B: tACS exposure
    A': stimulation-free washout

Every active episode has a same-seed sham trajectory.  The primary endpoint is
the active-minus-sham difference-in-differences in excitatory-population
pairwise phase consistency (PPC).  PPC is an unbiased transformation of PLV
with respect to spike count.  EEG power at the stimulation fundamental is
secondary and is accompanied by an observation-only synthetic-sine control.

Discovery maps a predeclared realistic-dose amplitude/frequency grid.  The
highest mean E-PPC action is frozen, then evaluated on disjoint validation
seeds against a transverse-field control, neighbouring frequencies, and the
discovery doses at the frozen frequency.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _analyze_eeg,
    _make_synthetic_epoch,
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick import (  # noqa: E402
    _benjamini_hochberg,
)


ANALYSIS_EPOCHS = ("baseline", "stimulation", "washout")
PRIMARY_METRIC = "E_ppc_difference_in_differences"


def _plain_copy(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _mpi_variables(
    comm,
    size: int,
    rank: int,
    seed: int,
    *,
    structure_seed: int | None = None,
    drive_seed: int | None = None,
    future_drive_seed: int | None = None,
    future_start_ms: float | None = None,
) -> dict[str, Any]:
    """Build MPI seeds, optionally separating structure from afferent drive.

    Existing callers retain the historical matched-seed behavior.  Scientific
    variance audits may hold the LFPy/NumPy structure stream fixed while
    changing only the per-synapse background-event stream.
    """
    structure = int(seed if structure_seed is None else structure_seed)
    drive = int(seed if drive_seed is None else drive_seed)
    result = {
        "COMM": comm,
        "SIZE": size,
        "RANK": rank,
        "GLOBALSEED": drive,
        "SEED": structure * 10_000,
    }
    if (future_drive_seed is None) != (future_start_ms is None):
        raise ValueError(
            "future_drive_seed and future_start_ms must be provided together."
        )
    if future_drive_seed is not None:
        result["FUTUREGLOBALSEED"] = int(future_drive_seed)
        result["FUTURESTARTMS"] = float(future_start_ms)
    return result


def _timeline(cfg: DictConfig) -> list[tuple[str, int]]:
    timeline = cfg.analysis.timeline
    values = [
        ("burn_in", int(timeline.burn_in_steps)),
        ("baseline", int(timeline.baseline_steps)),
        ("stimulation", int(timeline.stimulation_steps)),
        ("washout", int(timeline.washout_steps)),
    ]
    if any(count <= 0 for _, count in values):
        raise ValueError("Every T1 epoch must contain at least one window.")
    return values


def _episode_config(
    base_cfg: DictConfig,
    *,
    seed: int,
    output_dir: Path,
) -> DictConfig:
    run_cfg = _plain_copy(base_cfg)
    OmegaConf.set_struct(run_cfg, False)
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    n_steps = sum(count for _, count in _timeline(run_cfg))
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


def _action_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _build_discovery_actions(cfg: DictConfig) -> list[dict[str, Any]]:
    maximum = float(cfg.analysis.maximum_field_v_per_m)
    amplitudes = sorted(
        {float(value) for value in cfg.analysis.discovery.amplitudes_v_per_m}
    )
    frequencies = sorted(
        {float(value) for value in cfg.analysis.discovery.frequencies_hz}
    )
    if not amplitudes or not frequencies:
        raise ValueError("The T1 discovery grid cannot be empty.")
    if any(value <= 0.0 or value > maximum for value in amplitudes):
        raise ValueError(
            "Every discovery amplitude must be in (0, maximum_field_v_per_m]."
        )
    if any(value <= 0.0 for value in frequencies):
        raise ValueError("Every discovery frequency must be positive.")

    phase_rad = float(cfg.analysis.protocol.phase_rad)
    montage = str(cfg.analysis.protocol.axial_montage)
    return [
        {
            "id": f"axial_a{_action_token(amplitude)}_f{_action_token(frequency)}",
            "role": "discovery",
            "montage": montage,
            "dc_offset_v_per_m": 0.0,
            "ac_amplitude_v_per_m": amplitude,
            "frequency_hz": frequency,
            "phase_rad": phase_rad,
        }
        for amplitude in amplitudes
        for frequency in frequencies
    ]


def _environment_action(action: dict[str, Any]) -> dict[str, Any]:
    ignored = {"id", "role"}
    return {key: value for key, value in action.items() if key not in ignored}


def _zero_action(cfg: DictConfig) -> dict[str, Any]:
    return {
        "montage": str(cfg.analysis.protocol.axial_montage),
        "dc_offset_v_per_m": 0.0,
        "ac_amplitude_v_per_m": 0.0,
        "frequency_hz": 0.0,
        "phase_rad": 0.0,
    }


def _validation_actions(
    selected: dict[str, Any],
    discovery_actions: list[dict[str, Any]],
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    amplitude = float(selected["ac_amplitude_v_per_m"])
    frequency = float(selected["frequency_hz"])
    phase_rad = float(selected.get("phase_rad", 0.0))
    axial = str(cfg.analysis.protocol.axial_montage)
    transverse = str(cfg.analysis.protocol.transverse_montage)

    actions = [
        {
            **selected,
            "id": "selected_axial",
            "role": "selected",
            "montage": axial,
        },
        {
            **selected,
            "id": "selected_transverse",
            "role": "orientation_control",
            "montage": transverse,
        },
    ]

    if bool(cfg.analysis.validation.include_dose_controls):
        available_amplitudes = sorted(
            {
                float(action["ac_amplitude_v_per_m"])
                for action in discovery_actions
                if np.isclose(float(action["frequency_hz"]), frequency)
            }
        )
        for control_amplitude in available_amplitudes:
            if np.isclose(control_amplitude, amplitude):
                continue
            actions.append(
                {
                    "id": f"dose_control_a{_action_token(control_amplitude)}",
                    "role": "dose_control",
                    "montage": axial,
                    "dc_offset_v_per_m": 0.0,
                    "ac_amplitude_v_per_m": control_amplitude,
                    "frequency_hz": frequency,
                    "phase_rad": phase_rad,
                }
            )

    if bool(cfg.analysis.validation.include_frequency_neighbors):
        available = sorted(
            {
                float(action["frequency_hz"])
                for action in discovery_actions
                if np.isclose(
                    float(action["ac_amplitude_v_per_m"]), amplitude
                )
            }
        )
        lower = [value for value in available if value < frequency]
        upper = [value for value in available if value > frequency]
        neighbours: list[tuple[str, float]] = []
        if lower:
            neighbours.append(("lower_frequency_control", max(lower)))
        if upper:
            neighbours.append(("upper_frequency_control", min(upper)))
        for role, neighbour_frequency in neighbours:
            actions.append(
                {
                    "id": role,
                    "role": role,
                    "montage": axial,
                    "dc_offset_v_per_m": 0.0,
                    "ac_amplitude_v_per_m": amplitude,
                    "frequency_hz": neighbour_frequency,
                    "phase_rad": phase_rad,
                }
            )

    identifiers = [str(action["id"]) for action in actions]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Validation action identifiers must be unique.")
    return actions


def _candidate_analysis_config(
    base_cfg: DictConfig,
    action: dict[str, Any],
) -> DictConfig:
    cfg = _plain_copy(base_cfg)
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        cfg.analysis.protocol.frequency_hz = float(action["frequency_hz"])
        cfg.analysis.protocol.amplitude_v_per_m = float(
            action["ac_amplitude_v_per_m"]
        )
        cfg.analysis.protocol.block_ramp_ms = float(
            cfg.analysis.timeline.block_ramp_ms
        )
    return cfg


def _validate_online_outputs(
    outputs_by_epoch: dict[str, list[dict[str, Any]]],
) -> None:
    previous_stop_ms: float | None = None
    previous_sample_ms: float | None = None
    for epoch, outputs in outputs_by_epoch.items():
        for output in outputs:
            eeg = np.asarray(output["eeg_v"], dtype=np.float64).reshape(-1)
            times = np.asarray(output["sample_times_ms"], dtype=np.float64)
            if eeg.size != int(output["expected_sample_count"]):
                raise RuntimeError(f"Unexpected EEG sample count in {epoch}.")
            if eeg.size != times.size or not np.all(np.isfinite(eeg)):
                raise RuntimeError(f"Invalid EEG samples in {epoch}.")
            if previous_stop_ms is not None:
                if not np.isclose(float(output["t_start_ms"]), previous_stop_ms):
                    raise RuntimeError("T1 online windows are not contiguous.")
                if float(times[0]) <= float(previous_sample_ms):
                    raise RuntimeError("T1 online windows duplicate a boundary sample.")
            for population_name in ("E", "I"):
                spikes = output["spikes"][population_name]
                event_count = int(np.asarray(spikes["times_ms"]).size)
                cell_count = sum(len(values) for values in spikes["per_cell"].values())
                if event_count != cell_count:
                    raise RuntimeError(
                        f"{population_name} spike accounting mismatch in {epoch}."
                    )
            previous_stop_ms = float(output["t_stop_ms"])
            previous_sample_ms = float(times[-1])


def _simulate_episode(
    base_cfg: DictConfig,
    *,
    seed: int,
    action: dict[str, Any],
    stimulate: bool,
    output_dir: Path,
    comm,
    size: int,
    rank: int,
    structure_seed: int | None = None,
    drive_seed: int | None = None,
) -> dict[str, Any] | None:
    run_cfg = _episode_config(base_cfg, seed=seed, output_dir=output_dir)
    structure = int(seed if structure_seed is None else structure_seed)
    drive = int(seed if drive_seed is None else drive_seed)
    np.random.seed(structure * 10_000 + rank)
    random.seed(structure * 10_000 + rank)
    environment = OnlineNeuronEnv(
        run_cfg,
        _mpi_variables(
            comm,
            size,
            rank,
            seed,
            structure_seed=structure,
            drive_seed=drive,
        ),
        ENV_SEED=0,
    )

    schedule = _timeline(run_cfg)
    window_ms = float(run_cfg.env.simulation.obs_win_len)
    pre_stimulation_steps = int(run_cfg.analysis.timeline.burn_in_steps) + int(
        run_cfg.analysis.timeline.baseline_steps
    )
    block_start_ms = pre_stimulation_steps * window_ms
    block_stop_ms = block_start_ms + int(
        run_cfg.analysis.timeline.stimulation_steps
    ) * window_ms
    block_envelope = {
        "start_ms": block_start_ms,
        "stop_ms": block_stop_ms,
        "ramp_ms": float(run_cfg.analysis.timeline.block_ramp_ms),
    }
    zero = _zero_action(run_cfg)
    active = _environment_action(action)
    outputs_by_epoch: dict[str, list[dict[str, Any]]] | None = (
        {name: [] for name, _ in schedule} if rank == 0 else None
    )
    final_residual_mV = float("nan")
    try:
        for epoch, count in schedule:
            for epoch_step in range(count):
                is_active = bool(stimulate and epoch == "stimulation")
                step_action = dict(active if is_active else zero)
                if is_active and epoch_step > 0:
                    step_action.pop("phase_rad", None)
                output = environment.step_online(
                    step_action,
                    duration_ms=window_ms,
                    phase_continuous=True,
                    ramp_ms=0.0,
                    block_envelope=block_envelope if is_active else None,
                )
                if rank == 0:
                    outputs_by_epoch[epoch].append(output)
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
    _validate_online_outputs(outputs_by_epoch)
    if final_residual_mV != 0.0:
        raise RuntimeError(
            "The T1 washout left residual extracellular voltage: "
            f"{final_residual_mV:g} mV."
        )
    return {
        "seed": int(seed),
        "structure_seed": structure,
        "drive_seed": drive,
        "action": dict(action),
        "stimulate": bool(stimulate),
        "block_start_ms": float(block_start_ms),
        "outputs_by_epoch": outputs_by_epoch,
        "final_residual_mV": float(final_residual_mV),
    }


def _collect_epoch_spikes(
    outputs: list[dict[str, Any]],
    population_name: str,
) -> tuple[np.ndarray, int]:
    times = np.concatenate(
        [
            np.asarray(output["spikes"][population_name]["times_ms"])
            for output in outputs
        ]
    ).astype(np.float64, copy=False)
    population_size = len(outputs[0]["spikes"][population_name]["per_cell"])
    return times, int(population_size)


def _phase_locking_metrics(
    spike_times_ms: np.ndarray,
    *,
    frequency_hz: float,
    phase_origin_ms: float,
    n_surrogates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    times = np.asarray(spike_times_ms, dtype=np.float64)
    n_spikes = int(times.size)
    if n_spikes == 0:
        return {
            "spike_count": 0.0,
            "plv": float("nan"),
            "ppc": float("nan"),
            "mean_phase_rad": float("nan"),
            "rayleigh_z": float("nan"),
            "uniform_phase_null_plv_p95": float("nan"),
            "plv_above_uniform_null": 0.0,
        }

    phases = 2.0 * np.pi * float(frequency_hz) * (
        (times - float(phase_origin_ms)) / 1000.0
    )
    resultant = np.mean(np.exp(1j * phases))
    plv = float(np.abs(resultant))
    ppc = (
        float((n_spikes * plv * plv - 1.0) / (n_spikes - 1.0))
        if n_spikes > 1
        else float("nan")
    )
    if n_surrogates > 0:
        null_phases = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(int(n_surrogates), n_spikes),
        )
        null_plv = np.abs(np.mean(np.exp(1j * null_phases), axis=1))
        null_p95 = float(np.quantile(null_plv, 0.95))
    else:
        null_p95 = float("nan")
    return {
        "spike_count": float(n_spikes),
        "plv": plv,
        "ppc": ppc,
        "mean_phase_rad": float(np.angle(resultant)),
        "rayleigh_z": float(n_spikes * plv * plv),
        "uniform_phase_null_plv_p95": null_p95,
        "plv_above_uniform_null": float(
            np.isfinite(null_p95) and plv > null_p95
        ),
    }


def _phase_rng(
    seed: int,
    action_index: int,
    epoch_index: int,
    population_index: int,
    arm_index: int,
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [seed, action_index, epoch_index, population_index, arm_index, 104_729]
    )
    return np.random.default_rng(sequence)


def _analyze_episode(
    simulation: dict[str, Any],
    *,
    action: dict[str, Any],
    action_index: int,
    arm: str,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    candidate_cfg = _candidate_analysis_config(cfg, action)
    frequency_hz = float(action["frequency_hz"])
    seed = int(simulation["seed"])
    arm_index = 1 if arm == "active" else 0
    epoch_rows: list[dict[str, Any]] = []
    raw_by_epoch: dict[str, np.ndarray] = {}
    spike_payload: dict[str, np.ndarray] = {}
    simulator_fs_hz = 1000.0 / float(candidate_cfg.env.network.dt)

    for epoch_index, epoch in enumerate(ANALYSIS_EPOCHS):
        outputs = simulation["outputs_by_epoch"][epoch]
        raw = np.concatenate(
            [np.asarray(output["eeg_v"]).reshape(-1) for output in outputs]
        ).astype(np.float64, copy=False)
        features, processed, frequencies_hz, psd = _analyze_eeg(
            raw,
            simulator_fs_hz=simulator_fs_hz,
            cfg=candidate_cfg,
        )
        duration_s = sum(
            float(output["t_stop_ms"] - output["t_start_ms"])
            for output in outputs
        ) / 1000.0
        row: dict[str, Any] = {
            "seed": seed,
            "action_id": str(action["id"]),
            "action_role": str(action["role"]),
            "arm": arm,
            "epoch": epoch,
            "montage": str(action["montage"]),
            "ac_amplitude_v_per_m": float(
                action["ac_amplitude_v_per_m"]
            ),
            "frequency_hz": frequency_hz,
            "phase_rad": float(action.get("phase_rad", 0.0)),
            **features,
        }
        for population_index, population_name in enumerate(("E", "I")):
            times, population_size = _collect_epoch_spikes(
                outputs, population_name
            )
            metrics = _phase_locking_metrics(
                times,
                frequency_hz=frequency_hz,
                phase_origin_ms=float(simulation["block_start_ms"]),
                n_surrogates=int(cfg.analysis.phase_null.n_surrogates),
                rng=_phase_rng(
                    seed,
                    action_index,
                    epoch_index,
                    population_index,
                    arm_index,
                ),
            )
            row[f"{population_name}_firing_rate_hz"] = (
                times.size / (population_size * duration_s)
            )
            for name, value in metrics.items():
                row[f"{population_name}_{name}"] = value
            spike_payload[f"{epoch}_{population_name}_times_ms"] = times
        i_rate = float(row["I_firing_rate_hz"])
        row["E_I_firing_rate_ratio"] = (
            float(row["E_firing_rate_hz"]) / i_rate
            if i_rate > 0.0
            else float("nan")
        )
        epoch_rows.append(row)
        raw_by_epoch[epoch] = raw
        if bool(cfg.analysis.save_raw_eeg):
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_dir / f"{epoch}_signals.npz",
                eeg_raw_v=raw,
                eeg_preprocessed_v=processed,
                frequencies_hz=frequencies_hz,
                psd_v2_per_hz=psd,
            )

    if bool(cfg.analysis.save_spikes):
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "spikes.npz", **spike_payload)
    return epoch_rows, raw_by_epoch


def _synthetic_row(
    *,
    sham_rows: list[dict[str, Any]],
    sham_raw: np.ndarray,
    active_raw: np.ndarray,
    action: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    candidate_cfg = _candidate_analysis_config(cfg, action)
    simulator_fs_hz = 1000.0 / float(candidate_cfg.env.network.dt)
    synthetic, amplitude_v, achieved_power = _make_synthetic_epoch(
        sham_raw,
        active_raw,
        simulator_fs_hz=simulator_fs_hz,
        cfg=candidate_cfg,
    )
    features, _, _, _ = _analyze_eeg(
        synthetic,
        simulator_fs_hz=simulator_fs_hz,
        cfg=candidate_cfg,
    )
    source = next(row for row in sham_rows if row["epoch"] == "stimulation")
    copied = {
        key: value
        for key, value in source.items()
        if key.startswith("E_") or key.startswith("I_")
    }
    return {
        "seed": int(source["seed"]),
        "action_id": str(action["id"]),
        "action_role": str(action["role"]),
        "arm": "synthetic",
        "epoch": "stimulation",
        "montage": str(action["montage"]),
        "ac_amplitude_v_per_m": float(action["ac_amplitude_v_per_m"]),
        "frequency_hz": float(action["frequency_hz"]),
        "phase_rad": float(action.get("phase_rad", 0.0)),
        **features,
        **copied,
        "synthetic_added_peak_v": float(amplitude_v),
        "synthetic_achieved_driven_band_power": float(achieved_power),
    }


def _relative_rate_safe(
    active: pd.Series,
    sham: pd.Series,
    cfg: DictConfig,
) -> bool:
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    limits = cfg.analysis.rate_guardrails_hz
    for population_name in ("E", "I"):
        name = f"{population_name}_firing_rate_hz"
        active_value = float(active[name])
        sham_value = float(sham[name])
        low = float(limits[f"{population_name}_min"])
        high = float(limits[f"{population_name}_max"])
        relative_low = max(0.0, sham_value * (1.0 - tolerance))
        relative_high = sham_value * (1.0 + tolerance)
        if not (
            low <= active_value <= high
            and relative_low <= active_value <= relative_high
        ):
            return False
    return True


def _difference_in_differences(
    active_baseline: float,
    active_epoch: float,
    sham_baseline: float,
    sham_epoch: float,
) -> float:
    return float(
        (active_epoch - active_baseline) - (sham_epoch - sham_baseline)
    )


def _paired_action_row(
    *,
    seed: int,
    action: dict[str, Any],
    active_rows: list[dict[str, Any]],
    sham_rows: list[dict[str, Any]],
    synthetic: dict[str, Any],
    active_raw: dict[str, np.ndarray],
    sham_raw: dict[str, np.ndarray],
    cfg: DictConfig,
) -> dict[str, Any]:
    active = {row["epoch"]: pd.Series(row) for row in active_rows}
    sham = {row["epoch"]: pd.Series(row) for row in sham_rows}
    result: dict[str, Any] = {
        "seed": int(seed),
        "action_id": str(action["id"]),
        "action_role": str(action["role"]),
        "montage": str(action["montage"]),
        "ac_amplitude_v_per_m": float(action["ac_amplitude_v_per_m"]),
        "frequency_hz": float(action["frequency_hz"]),
        "phase_rad": float(action.get("phase_rad", 0.0)),
        "baseline_relative_rms_error_active_vs_sham": _relative_rms_error(
            sham_raw["baseline"], active_raw["baseline"]
        ),
        "final_field_residual_mV": 0.0,
    }
    for population_name in ("E", "I"):
        for metric in ("plv", "ppc"):
            name = f"{population_name}_{metric}"
            result[f"{name}_difference_in_differences"] = (
                _difference_in_differences(
                    float(active["baseline"][name]),
                    float(active["stimulation"][name]),
                    float(sham["baseline"][name]),
                    float(sham["stimulation"][name]),
                )
            )
            result[f"{name}_washout_difference_in_differences"] = (
                _difference_in_differences(
                    float(active["baseline"][name]),
                    float(active["washout"][name]),
                    float(sham["baseline"][name]),
                    float(sham["washout"][name]),
                )
            )
        result[f"{population_name}_active_stimulation_plv"] = float(
            active["stimulation"][f"{population_name}_plv"]
        )
        result[f"{population_name}_sham_stimulation_plv"] = float(
            sham["stimulation"][f"{population_name}_plv"]
        )
        result[f"{population_name}_active_plv_above_uniform_null"] = bool(
            active["stimulation"][
                f"{population_name}_plv_above_uniform_null"
            ]
        )
        result[f"{population_name}_rate_change_hz"] = float(
            active["stimulation"][f"{population_name}_firing_rate_hz"]
            - sham["stimulation"][f"{population_name}_firing_rate_hz"]
        )
        result[f"{population_name}_active_rate_hz"] = float(
            active["stimulation"][f"{population_name}_firing_rate_hz"]
        )
        result[f"{population_name}_sham_rate_hz"] = float(
            sham["stimulation"][f"{population_name}_firing_rate_hz"]
        )

    residual_fraction = float(cfg.analysis.criteria.maximum_washout_residual_fraction)
    primary_effect = float(result[PRIMARY_METRIC])
    primary_washout = float(result["E_ppc_washout_difference_in_differences"])
    result["E_washout_recovered"] = bool(
        primary_effect > 0.0
        and abs(primary_washout)
        <= residual_fraction * max(abs(primary_effect), np.finfo(float).eps)
    )
    result["rate_safe"] = _relative_rate_safe(
        active["stimulation"], sham["stimulation"], cfg
    )
    for feature_name in (
        "log10_total_power_1_80_excluding_stimulus",
        "relative_gamma_power_excluding_stimulus",
        "stimulus_frequency_power",
    ):
        active_value = float(active["stimulation"][feature_name])
        sham_value = float(sham["stimulation"][feature_name])
        synthetic_value = float(synthetic[feature_name])
        result[f"active_minus_sham_{feature_name}"] = active_value - sham_value
        result[f"active_minus_synthetic_{feature_name}"] = (
            active_value - synthetic_value
        )
    result["synthetic_added_peak_v"] = float(synthetic["synthetic_added_peak_v"])
    return result


def _bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(int(n_bootstrap), values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_p(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(np.mean(values)))
    signs = rng.choice([-1.0, 1.0], size=(int(n_permutations), values.size))
    null = np.abs(np.mean(signs * values, axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))


def _summarize_actions(
    rows: pd.DataFrame,
    *,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for action_id, group in rows.groupby("action_id", sort=False):
        primary = group[PRIMARY_METRIC].to_numpy(dtype=float)
        e_plv = group["E_plv_difference_in_differences"].to_numpy(dtype=float)
        i_ppc = group["I_ppc_difference_in_differences"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(
            primary,
            rng=rng,
            n_bootstrap=int(cfg.analysis.n_bootstrap),
        )
        first = group.iloc[0]
        summaries.append(
            {
                "action_id": str(action_id),
                "action_role": str(first["action_role"]),
                "montage": str(first["montage"]),
                "ac_amplitude_v_per_m": float(first["ac_amplitude_v_per_m"]),
                "frequency_hz": float(first["frequency_hz"]),
                "n_seeds": int(group["seed"].nunique()),
                "mean_E_ppc_gain": float(np.nanmean(primary)),
                "E_ppc_gain_ci_2.5": ci_low,
                "E_ppc_gain_ci_97.5": ci_high,
                "E_ppc_paired_sign_flip_p": _sign_flip_p(
                    primary,
                    rng=rng,
                    n_permutations=int(cfg.analysis.n_permutations),
                ),
                "mean_E_plv_gain": float(np.nanmean(e_plv)),
                "mean_I_ppc_gain": float(np.nanmean(i_ppc)),
                "positive_E_ppc_seed_fraction": float(np.mean(primary > 0.0)),
                "E_plv_above_null_seed_fraction": float(
                    group["E_active_plv_above_uniform_null"].mean()
                ),
                "washout_recovery_seed_fraction": float(
                    group["E_washout_recovered"].mean()
                ),
                "rate_safe_seed_fraction": float(group["rate_safe"].mean()),
                "maximum_baseline_relative_rms_error": float(
                    group[
                        "baseline_relative_rms_error_active_vs_sham"
                    ].max()
                ),
            }
        )
    return pd.DataFrame(summaries)


def _select_protocol(
    summary: pd.DataFrame,
    actions: list[dict[str, Any]],
    cfg: DictConfig,
) -> tuple[dict[str, Any], bool]:
    minimum_safe = float(cfg.analysis.discovery.minimum_rate_safe_fraction)
    eligible = summary[summary["rate_safe_seed_fraction"] >= minimum_safe]
    used_safe_subset = not eligible.empty
    ranked = eligible if used_safe_subset else summary
    selected_id = str(
        ranked.sort_values(
            ["mean_E_ppc_gain", "mean_E_plv_gain"],
            ascending=False,
        ).iloc[0]["action_id"]
    )
    return next(action for action in actions if action["id"] == selected_id), used_safe_subset


def _paired_control_summary(
    rows: pd.DataFrame,
    *,
    selected_id: str,
    control_id: str,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    pivot = rows.pivot(index="seed", columns="action_id", values=PRIMARY_METRIC)
    paired = pivot[[selected_id, control_id]].dropna()
    differences = (
        paired[selected_id] - paired[control_id]
    ).to_numpy(dtype=float)
    ci_low, ci_high = _bootstrap_ci(
        differences,
        rng=rng,
        n_bootstrap=int(cfg.analysis.n_bootstrap),
    )
    return {
        "selected_action_id": selected_id,
        "control_action_id": control_id,
        "n_seeds": int(differences.size),
        "mean_selected_minus_control_E_ppc_gain": float(
            np.mean(differences)
        ),
        "ci_2.5": ci_low,
        "ci_97.5": ci_high,
        "paired_sign_flip_p": _sign_flip_p(
            differences,
            rng=rng,
            n_permutations=int(cfg.analysis.n_permutations),
        ),
    }


def _dose_response(
    action_summary: pd.DataFrame,
    *,
    selected_frequency_hz: float,
) -> dict[str, float]:
    frequency_mask = np.isclose(
        action_summary["frequency_hz"].to_numpy(dtype=float),
        float(selected_frequency_hz),
    )
    role_mask = action_summary["action_role"].isin(["selected", "dose_control"])
    rows = action_summary[frequency_mask & role_mask].sort_values(
        "ac_amplitude_v_per_m"
    )
    if rows.empty:
        # Discovery actions all have the generic discovery role.
        rows = action_summary[frequency_mask].sort_values(
            "ac_amplitude_v_per_m"
        )
    if len(rows) < 2:
        return {"n_doses": int(len(rows)), "spearman_rho": float("nan")}
    rho = st.spearmanr(
        rows["ac_amplitude_v_per_m"].to_numpy(dtype=float),
        rows["mean_E_ppc_gain"].to_numpy(dtype=float),
    ).statistic
    return {"n_doses": int(len(rows)), "spearman_rho": float(rho)}


def _plot_discovery(summary: pd.DataFrame, output_dir: Path) -> None:
    table = summary.pivot(
        index="ac_amplitude_v_per_m",
        columns="frequency_hz",
        values="mean_E_ppc_gain",
    ).sort_index().sort_index(axis=1)
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    image = axis.imshow(table.to_numpy(), aspect="auto", origin="lower")
    axis.set_xticks(np.arange(table.shape[1]), labels=[f"{x:g}" for x in table.columns])
    axis.set_yticks(np.arange(table.shape[0]), labels=[f"{x:g}" for x in table.index])
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Field amplitude (V/m)")
    axis.set_title("Discovery mean E-PPC gain (active minus sham DID)")
    figure.colorbar(image, ax=axis, label="E-PPC gain")
    figure.tight_layout()
    figure.savefig(output_dir / "discovery_E_ppc_gain.png", dpi=250)
    plt.close(figure)


def _run_cohort(
    *,
    cohort: str,
    seeds: list[int],
    actions: list[dict[str, Any]],
    cfg: DictConfig,
    root: Path,
    comm,
    size: int,
    rank: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    epoch_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for seed in seeds:
        if rank == 0:
            print(f"{cohort} seed={seed}, arm=sham")
        sham_simulation = _simulate_episode(
            cfg,
            seed=seed,
            action=actions[0],
            stimulate=False,
            output_dir=root / cohort / "sham" / f"seed_{seed}",
            comm=comm,
            size=size,
            rank=rank,
        )
        for action_index, action in enumerate(actions):
            if rank == 0:
                print(
                    f"{cohort} seed={seed}, action={action['id']} "
                    f"({action['ac_amplitude_v_per_m']:g} V/m, "
                    f"{action['frequency_hz']:g} Hz, {action['montage']})"
                )
            active_simulation = _simulate_episode(
                cfg,
                seed=seed,
                action=action,
                stimulate=True,
                output_dir=root / cohort / "active" / str(action["id"]) / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank != 0:
                continue
            sham_action_rows, sham_raw = _analyze_episode(
                sham_simulation,
                action=action,
                action_index=action_index,
                arm="sham",
                cfg=cfg,
                output_dir=root / cohort / "sham_analysis" / str(action["id"]) / f"seed_{seed}",
            )
            active_action_rows, active_raw = _analyze_episode(
                active_simulation,
                action=action,
                action_index=action_index,
                arm="active",
                cfg=cfg,
                output_dir=root / cohort / "active" / str(action["id"]) / f"seed_{seed}",
            )
            synthetic = _synthetic_row(
                sham_rows=sham_action_rows,
                sham_raw=sham_raw["stimulation"],
                active_raw=active_raw["stimulation"],
                action=action,
                cfg=cfg,
            )
            epoch_rows.extend(sham_action_rows)
            epoch_rows.extend(active_action_rows)
            epoch_rows.append(synthetic)
            paired_rows.append(
                _paired_action_row(
                    seed=seed,
                    action=action,
                    active_rows=active_action_rows,
                    sham_rows=sham_action_rows,
                    synthetic=synthetic,
                    active_raw=active_raw,
                    sham_raw=sham_raw,
                    cfg=cfg,
                )
            )
    if rank != 0:
        return None, None
    return pd.DataFrame(epoch_rows), pd.DataFrame(paired_rows)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("The T1 entrainment experiment requires online mode.")

    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "tes_entrainment"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### BallAndStick T1 reversible tES entrainment")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    base_seed = int(cfg.experiment.seed)
    discovery_seeds = [
        base_seed + int(cfg.analysis.discovery.seed_offset) + index
        for index in range(int(cfg.analysis.discovery.n_seeds))
    ]
    validation_seeds = [
        base_seed + int(cfg.analysis.validation.seed_offset) + index
        for index in range(int(cfg.analysis.validation.n_seeds))
    ]
    if set(discovery_seeds).intersection(validation_seeds):
        raise ValueError("T1 discovery and validation seeds must be disjoint.")

    discovery_actions = _build_discovery_actions(cfg)
    discovery_epochs, discovery_pairs = _run_cohort(
        cohort="discovery",
        seeds=discovery_seeds,
        actions=discovery_actions,
        cfg=cfg,
        root=root,
        comm=comm,
        size=size,
        rank=rank,
    )
    if rank == 0:
        rng = np.random.default_rng(base_seed + 700_001)
        discovery_summary = _summarize_actions(
            discovery_pairs, cfg=cfg, rng=rng
        )
        selected, selected_from_safe_subset = _select_protocol(
            discovery_summary, discovery_actions, cfg
        )
        discovery_epochs.to_csv(root / "discovery_epoch_metrics.csv", index=False)
        discovery_pairs.to_csv(root / "discovery_action_seed_metrics.csv", index=False)
        discovery_summary.to_csv(root / "discovery_action_summary.csv", index=False)
        if bool(cfg.experiment.plot):
            _plot_discovery(discovery_summary, root)
    else:
        selected = selected_from_safe_subset = None
    selected = comm.bcast(selected, root=0)
    selected_from_safe_subset = comm.bcast(selected_from_safe_subset, root=0)
    validation_actions = _validation_actions(selected, discovery_actions, cfg)

    validation_epochs, validation_pairs = _run_cohort(
        cohort="validation",
        seeds=validation_seeds,
        actions=validation_actions,
        cfg=cfg,
        root=root,
        comm=comm,
        size=size,
        rank=rank,
    )
    if rank != 0:
        return

    rng = np.random.default_rng(base_seed + 800_001)
    validation_summary = _summarize_actions(
        validation_pairs, cfg=cfg, rng=rng
    )
    validation_epochs.to_csv(root / "validation_epoch_metrics.csv", index=False)
    validation_pairs.to_csv(root / "validation_action_seed_metrics.csv", index=False)
    validation_summary.to_csv(root / "validation_action_summary.csv", index=False)

    comparison_rows = []
    for control_id in validation_summary["action_id"]:
        if control_id == "selected_axial":
            continue
        comparison_rows.append(
            _paired_control_summary(
                validation_pairs,
                selected_id="selected_axial",
                control_id=str(control_id),
                cfg=cfg,
                rng=rng,
            )
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["fdr_q"] = _benjamini_hochberg(
        comparisons["paired_sign_flip_p"].to_numpy(dtype=float)
    )
    comparisons.to_csv(root / "validation_control_comparisons.csv", index=False)

    selected_summary = validation_summary[
        validation_summary["action_id"] == "selected_axial"
    ].iloc[0]
    orientation = comparisons[
        comparisons["control_action_id"] == "selected_transverse"
    ].iloc[0]
    frequency_comparisons = comparisons[
        comparisons["control_action_id"].str.contains("frequency_control")
    ]
    discovery_dose_response = _dose_response(
        discovery_summary,
        selected_frequency_hz=float(selected["frequency_hz"]),
    )
    validation_dose_response = _dose_response(
        validation_summary,
        selected_frequency_hz=float(selected["frequency_hz"]),
    )
    criteria = cfg.analysis.criteria
    checks = {
        "minimum_validation_seeds": int(selected_summary["n_seeds"])
        >= int(criteria.minimum_validation_seeds),
        "positive_primary_effect": (
            float(selected_summary["mean_E_ppc_gain"])
            >= float(criteria.minimum_mean_E_ppc_gain)
            and float(selected_summary["E_ppc_gain_ci_2.5"]) > 0.0
        ),
        "seed_consistency": float(
            selected_summary["positive_E_ppc_seed_fraction"]
        )
        >= float(criteria.minimum_positive_seed_fraction),
        "above_uniform_phase_null": float(
            selected_summary["E_plv_above_null_seed_fraction"]
        )
        >= float(criteria.minimum_above_null_seed_fraction),
        "rate_safe": float(selected_summary["rate_safe_seed_fraction"])
        >= float(criteria.minimum_rate_safe_seed_fraction),
        "washout_reversible": float(
            selected_summary["washout_recovery_seed_fraction"]
        )
        >= float(criteria.minimum_washout_recovery_seed_fraction),
        "baseline_causality": float(
            selected_summary["maximum_baseline_relative_rms_error"]
        )
        <= float(criteria.maximum_baseline_relative_rms_error),
        "orientation_specific": float(orientation["ci_2.5"]) > 0.0,
        "frequency_specific": bool(
            not frequency_comparisons.empty
            and (frequency_comparisons["ci_2.5"] > 0.0).all()
        ),
        "dose_response": bool(
            validation_dose_response["n_doses"]
            >= int(criteria.minimum_dose_levels)
            and np.isfinite(validation_dose_response["spearman_rho"])
            and validation_dose_response["spearman_rho"]
            >= float(criteria.minimum_dose_spearman_rho)
        ),
        "selected_from_rate_safe_discovery_set": bool(
            selected_from_safe_subset
        ),
    }
    conclusion = {
        "interpretation": (
            "Acute generic weak-field entrainment in the 40-cell BallAndStick "
            "network; not depression, treatment, or after-effect validation."
        ),
        "selected_protocol": selected,
        "discovery_dose_response": discovery_dose_response,
        "validation_dose_response": validation_dose_response,
        "primary_validation_summary": selected_summary.to_dict(),
        "checks": checks,
        "t1_reversible_entrainment_passed": bool(all(checks.values())),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    with (root / "selected_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    with (root / "experiment_conclusion.json").open("w", encoding="utf-8") as handle:
        json.dump(conclusion, handle, indent=2)

    print("\n### T1 selected protocol")
    print(json.dumps(selected, indent=2))
    print("\n### T1 mechanistic checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nReversible tES entrainment: "
        f"{'PASSED' if conclusion['t1_reversible_entrainment_passed'] else 'NOT PASSED'}"
    )
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
