"""Validate subthreshold polarization of the isolated BallAndStick cell.

This assay checks the numerical and biophysical coupling of a spatially
uniform vector field before network controllability is considered. It does not
claim that the same field can restore an I-to-E conductance perturbation.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import neuron
import numpy as np
import pandas as pd
from decouple import config
from hydra.utils import to_absolute_path
from LFPy import NetworkCell
from omegaconf import DictConfig, OmegaConf


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.extracellular_online import (  # noqa: E402
    OnlineExtracellularController,
)
from env.models.neuron.stimulation import (  # noqa: E402
    make_sinusoidal_electric_field,
)


def _cell_parameters(cfg: DictConfig, duration_ms: float) -> dict[str, Any]:
    circuit = Path(MAIN_PATH) / "setup" / "circuits" / "ballnstick"
    return {
        "morphology": str(circuit / "BallAndStick.hoc"),
        "templatefile": str(circuit / "BallAndStickTemplate.hoc"),
        "templatename": "BallAndStickTemplate",
        "templateargs": None,
        "delete_sections": False,
        "dt": float(cfg.env.network.dt),
        "tstop": float(duration_ms),
        "verbose": False,
    }


def _harmonic_amplitude(values: np.ndarray, time_ms: np.ndarray, frequency_hz: float):
    if frequency_hz <= 0.0:
        return 0.0
    phase = 2.0 * np.pi * float(frequency_hz) * np.asarray(time_ms) / 1000.0
    design = np.column_stack(
        [np.ones(phase.size), np.sin(phase), np.cos(phase)]
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.hypot(coefficients[1], coefficients[2]))


def _run_action(cfg: DictConfig, action: dict[str, Any]) -> dict[str, Any]:
    baseline_ms = float(cfg.analysis.baseline_ms)
    stimulation_ms = float(cfg.analysis.stimulation_ms)
    measurement_ms = float(cfg.analysis.measurement_ms)
    dt_ms = float(cfg.env.network.dt)
    total_ms = baseline_ms + stimulation_ms

    neuron.h("forall delete_section()")
    cell = NetworkCell(**_cell_parameters(cfg, total_ms))
    segments = [segment for section in cell.allseclist for segment in section]
    for section in cell.allseclist:
        section.insert("extracellular")

    midpoints_um = np.column_stack(
        [
            np.asarray(cell.x, dtype=np.float64).mean(axis=1),
            np.asarray(cell.y, dtype=np.float64).mean(axis=1),
            np.asarray(cell.z, dtype=np.float64).mean(axis=1),
        ]
    )
    field = make_sinusoidal_electric_field(
        amplitude_v_per_m=float(action.get("ac_amplitude_v_per_m", 0.0)),
        frequency_hz=float(action.get("frequency_hz", 0.0)),
        dc_offset_v_per_m=float(action.get("dc_offset_v_per_m", 0.0)),
        phase_rad=float(action.get("phase_rad", 0.0)),
        start_ms=baseline_ms,
        duration_ms=stimulation_ms,
        dt_ms=dt_ms,
        include_endpoint=True,
    )
    v_ext_mV = OnlineExtracellularController.uniform_field_potential_mV(
        midpoints_um=midpoints_um,
        field_v_per_m=field.field_v_per_m,
        field_direction=action["field_direction"],
    )

    neuron.h.dt = dt_ms
    neuron.h.celsius = float(cfg.env.network.celsius)
    neuron.h.finitialize(float(cfg.env.network.v_init))
    neuron.h.fcurrent()
    neuron.h.frecord_init()

    n_steps = int(round(total_ms / dt_ms))
    baseline_steps = int(round(baseline_ms / dt_ms))
    time_ms = np.empty(n_steps, dtype=np.float64)
    soma_v_mV = np.empty(n_steps, dtype=np.float64)
    distal_v_mV = np.empty(n_steps, dtype=np.float64)
    for step in range(n_steps):
        if step >= baseline_steps:
            field_index = step - baseline_steps
            for segment, value in zip(segments, v_ext_mV[:, field_index]):
                segment.e_extracellular = float(value)
        neuron.h.fadvance()
        time_ms[step] = float(neuron.h.t)
        soma_v_mV[step] = float(segments[0].v)
        distal_v_mV[step] = float(segments[-1].v)

    baseline_mask = (time_ms > baseline_ms - measurement_ms) & (
        time_ms <= baseline_ms
    )
    stimulation_mask = time_ms > total_ms - measurement_ms
    stimulus_time_ms = time_ms[stimulation_mask] - baseline_ms
    soma_baseline = float(np.mean(soma_v_mV[baseline_mask]))
    distal_baseline = float(np.mean(distal_v_mV[baseline_mask]))
    soma_shift = float(np.mean(soma_v_mV[stimulation_mask]) - soma_baseline)
    distal_shift = float(
        np.mean(distal_v_mV[stimulation_mask]) - distal_baseline
    )
    frequency_hz = float(action.get("frequency_hz", 0.0))
    ac_amplitude = float(action.get("ac_amplitude_v_per_m", 0.0))
    row = {
        "action_id": str(action["id"]),
        "dc_offset_v_per_m": float(action.get("dc_offset_v_per_m", 0.0)),
        "ac_amplitude_v_per_m": ac_amplitude,
        "frequency_hz": frequency_hz,
        "phase_rad": float(action.get("phase_rad", 0.0)),
        "field_x": float(action["field_direction"][0]),
        "field_y": float(action["field_direction"][1]),
        "field_z": float(action["field_direction"][2]),
        "soma_baseline_mV": soma_baseline,
        "distal_baseline_mV": distal_baseline,
        "soma_mean_shift_mV": soma_shift,
        "distal_mean_shift_mV": distal_shift,
        "soma_ac_amplitude_mV": _harmonic_amplitude(
            soma_v_mV[stimulation_mask], stimulus_time_ms, frequency_hz
        ),
        "distal_ac_amplitude_mV": _harmonic_amplitude(
            distal_v_mV[stimulation_mask], stimulus_time_ms, frequency_hz
        ),
        "peak_extracellular_span_mV": float(
            np.max(np.ptp(v_ext_mV, axis=0))
        ),
        "finite": bool(
            np.all(np.isfinite(soma_v_mV))
            and np.all(np.isfinite(distal_v_mV))
        ),
    }
    if ac_amplitude > 0.0:
        row["soma_ac_gain_mV_per_v_per_m"] = (
            row["soma_ac_amplitude_mV"] / ac_amplitude
        )
        row["distal_ac_gain_mV_per_v_per_m"] = (
            row["distal_ac_amplitude_mV"] / ac_amplitude
        )
    else:
        row["soma_ac_gain_mV_per_v_per_m"] = 0.0
        row["distal_ac_gain_mV_per_v_per_m"] = 0.0

    try:
        cell.__del__()
    finally:
        neuron.h("forall delete_section()")
    return row


def _summarize(rows: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    axial = rows[
        (rows["frequency_hz"] == 0.0)
        & (rows["dc_offset_v_per_m"] != 0.0)
        & (np.abs(rows["field_z"]) > 0.99)
    ]
    doses = axial["dc_offset_v_per_m"].to_numpy(dtype=np.float64)
    responses = axial["soma_mean_shift_mV"].to_numpy(dtype=np.float64)
    slope = float(np.dot(doses, responses) / np.dot(doses, doses))
    predicted = slope * doses
    residual_ss = float(np.sum((responses - predicted) ** 2))
    total_ss = float(np.sum((responses - np.mean(responses)) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0.0 else 0.0

    axial_0p8 = float(
        rows.loc[rows["action_id"] == "axial_dc_p0p8", "soma_mean_shift_mV"].iloc[0]
    )
    transverse_0p8 = float(
        rows.loc[
            rows["action_id"] == "transverse_dc_p0p8", "soma_mean_shift_mV"
        ].iloc[0]
    )
    transverse_ratio = (
        abs(transverse_0p8 / axial_0p8) if axial_0p8 != 0.0 else float("inf")
    )
    criteria = cfg.analysis.criteria
    checks = {
        "all_finite": bool(rows["finite"].all()),
        "axial_dc_polarizes_soma": bool(
            abs(slope)
            >= float(criteria.minimum_abs_axial_dc_soma_gain_mV_per_v_per_m)
        ),
        "axial_dc_is_linear": bool(
            r_squared >= float(criteria.minimum_axial_dc_linearity_r2)
        ),
        "transverse_negative_control": bool(
            transverse_ratio
            <= float(criteria.maximum_transverse_to_axial_gain_ratio)
        ),
        "polarity_reverses_response": bool(
            float(
                rows.loc[
                    rows["action_id"] == "axial_dc_m0p8",
                    "soma_mean_shift_mV",
                ].iloc[0]
            )
            * axial_0p8
            < 0.0
        ),
    }
    return {
        "axial_dc_soma_gain_mV_per_v_per_m": slope,
        "axial_dc_linearity_r2": r_squared,
        "transverse_to_axial_soma_response_ratio": transverse_ratio,
        "max_abs_realistic_soma_mean_shift_mV": float(
            rows["soma_mean_shift_mV"].abs().max()
        ),
        "max_axial_ac_soma_gain_mV_per_v_per_m": float(
            rows["soma_ac_gain_mV_per_v_per_m"].max()
        ),
        "checks": checks,
        "polarization_validation_passed": bool(all(checks.values())),
        "interpretation": (
            "This validates field-to-membrane coupling only; network-level "
            "A-like controllability requires the separate open-loop screen."
        ),
    }


def _plot(rows: pd.DataFrame, output_dir: Path) -> None:
    axial = rows[
        (rows["frequency_hz"] == 0.0)
        & (rows["dc_offset_v_per_m"] != 0.0)
        & (np.abs(rows["field_z"]) > 0.99)
    ].sort_values("dc_offset_v_per_m")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        axial["dc_offset_v_per_m"], axial["soma_mean_shift_mV"], "o-"
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Signed axial DC field (V/m)")
    axes[0].set_ylabel("Steady soma polarization (mV)")
    ac = rows[rows["ac_amplitude_v_per_m"] > 0.0]
    axes[1].scatter(ac["frequency_hz"], ac["soma_ac_amplitude_mV"])
    axes[1].set_xlabel("AC frequency (Hz)")
    axes[1].set_ylabel("Somatic harmonic amplitude (mV)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "cellular_polarization.png", dpi=220)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = (
        Path(to_absolute_path(str(cfg.experiment.dir))) / "cellular_polarization"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n### Isolated BallAndStick polarization validation")
    print(OmegaConf.to_yaml(cfg.analysis, resolve=True))

    actions = OmegaConf.to_container(cfg.analysis.actions, resolve=True)
    rows = pd.DataFrame([_run_action(cfg, dict(action)) for action in actions])
    summary = _summarize(rows, cfg)
    rows.to_csv(output_dir / "polarization_metrics.csv", index=False)
    with (output_dir / "polarization_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _plot(rows, output_dir)

    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
