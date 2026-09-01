"""Visualize one-time and causally refreshed phase control from D1-R outputs.

The D1-R result files deliberately store compact phase-controller diagnostics
rather than every raw simulator sample.  This script reconstructs two signals
that are sufficient to visualize the controller mechanics:

* the target-frequency EEG carrier component implied by the rolling one-second
  Fourier estimates saved at every controller boundary; and
* the exact phase-continuous tACS field command implied by the saved oscillator
  phase and frequency command in each controller interval.

The reconstructed EEG carrier is not the raw broadband EEG.  That distinction
is stated in the figure and its metadata so this pedagogical visualization
cannot be mistaken for an additional neural endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd

from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from env.models.neuron.stimulation import apply_raised_cosine_block_envelope


ONE_TIME = "one_time"
REFRESHED = "phase_refreshed"
MODES = (ONE_TIME, REFRESHED)


def _load_resolved_parameters(result_dir: Path) -> dict[str, float]:
    hydra_config = result_dir.parent / ".hydra" / "config.yaml"
    if not hydra_config.exists():
        raise FileNotFoundError(
            f"Resolved Hydra configuration not found: {hydra_config}"
        )
    cfg = OmegaConf.load(hydra_config)
    return {
        "amplitude_v_per_m": float(cfg.analysis.actions.amplitude_v_per_m),
        "refresh_interval_ms": float(cfg.analysis.tacs.refresh_interval_ms),
        "phase_history_ms": float(cfg.analysis.tacs.phase_estimation_history_ms),
        "relative_phase_offset_rad": float(
            cfg.analysis.tacs.relative_phase_offset_rad
        ),
        "block_ramp_ms": float(cfg.analysis.timeline.block_ramp_ms),
    }


def _choose_example(
    updates: pd.DataFrame,
    *,
    context_id: str | None,
    future_index: int | None,
) -> tuple[str, int, pd.DataFrame]:
    active = updates[updates.controller_mode.isin(MODES)].copy()
    if active.empty:
        raise ValueError("No one_time or phase_refreshed records were found.")

    available_modes = active.groupby(["context_id", "future_index"])[
        "controller_mode"
    ].nunique()
    complete = available_modes[available_modes.eq(len(MODES))].index
    active = active.set_index(["context_id", "future_index"]).loc[complete].reset_index()
    if active.empty:
        raise ValueError("No context/future pair contains both active controllers.")

    if context_id is not None:
        active = active[active.context_id.eq(str(context_id))]
    if future_index is not None:
        active = active[active.future_index.eq(int(future_index))]
    if active.empty:
        raise ValueError("The requested context/future pair is not available.")

    if context_id is None or future_index is None:
        # Select a representative, not an extreme, example. Rank complete pairs
        # by the refreshed-minus-one-time reduction in absolute phase error and
        # use the median pair. This selection is for display only.
        error = (
            active.assign(abs_error=active.phase_error_before_correction_rad.abs())
            .groupby(["context_id", "future_index", "controller_mode"])
            .abs_error.mean()
            .unstack("controller_mode")
        )
        error["refresh_error_reduction_rad"] = (
            error[ONE_TIME] - error[REFRESHED]
        )
        ranked = error.sort_values(
            ["refresh_error_reduction_rad", ONE_TIME], kind="mergesort"
        )
        selected_context, selected_future = ranked.index[len(ranked) // 2]
    else:
        selected_context, selected_future = str(context_id), int(future_index)

    selected = active[
        active.context_id.eq(str(selected_context))
        & active.future_index.eq(int(selected_future))
    ].copy()
    return str(selected_context), int(selected_future), selected


def _controller_signals(
    rows: pd.DataFrame,
    *,
    amplitude_v_per_m: float,
    block_ramp_ms: float,
    samples_per_second: float,
) -> dict[str, np.ndarray | float]:
    rows = rows.sort_values("update_index").reset_index(drop=True)
    if len(rows) < 2:
        raise ValueError("At least two controller intervals are required.")
    boundary_ms = rows.boundary_ms.to_numpy(dtype=float)
    intervals_ms = np.diff(boundary_ms)
    update_ms = float(np.median(intervals_ms))
    if not np.allclose(intervals_ms, update_ms, rtol=0.0, atol=1.0e-8):
        raise ValueError("Controller update intervals are not uniform.")

    start_ms = float(boundary_ms[0])
    stop_ms = float(boundary_ms[-1] + update_ms)
    sample_dt_ms = 1000.0 / float(samples_per_second)
    time_ms = np.arange(start_ms, stop_ms, sample_dt_ms, dtype=float)
    if time_ms.size == 0:
        raise ValueError("The reconstructed time axis is empty.")

    # Reconstruct the EEG carrier from the boundary estimates. Directly
    # unwrapping phase is ambiguous because a 9--11 Hz carrier completes more
    # than one cycle per 250-ms interval. Removing the known carrier first
    # leaves the slowly varying phase residual, which can be safely unwrapped.
    carrier_hz = float(rows.carrier_frequency_hz.iloc[0])
    if not np.allclose(rows.carrier_frequency_hz, carrier_hz):
        raise ValueError("Carrier frequency unexpectedly changes between updates.")
    boundary_s = boundary_ms / 1000.0
    phase_residual = np.unwrap(
        rows.estimated_eeg_phase_at_boundary_rad.to_numpy(dtype=float)
        - 2.0 * np.pi * carrier_hz * boundary_s
    )
    time_s = time_ms / 1000.0
    residual_t = np.interp(time_ms, boundary_ms, phase_residual)
    amplitude_t = np.interp(
        time_ms,
        boundary_ms,
        rows.eeg_resultant_v.to_numpy(dtype=float),
    )
    eeg_carrier_v = amplitude_t * np.cos(
        2.0 * np.pi * carrier_hz * time_s + residual_t
    )

    field_v_per_m = np.empty_like(time_ms)
    for row in rows.itertuples(index=False):
        left = float(row.boundary_ms)
        right = left + update_ms
        keep = (time_ms >= left) & (time_ms < right)
        relative_s = (time_ms[keep] - left) / 1000.0
        field_v_per_m[keep] = float(amplitude_v_per_m) * np.sin(
            float(row.oscillator_phase_before_update_rad)
            + 2.0 * np.pi * float(row.command_frequency_hz) * relative_s
        )
    field_v_per_m = apply_raised_cosine_block_envelope(
        field_v_per_m,
        time_ms=time_ms,
        block_start_ms=start_ms,
        block_stop_ms=stop_ms,
        ramp_ms=float(block_ramp_ms),
    )
    return {
        "time_s": (time_ms - start_ms) / 1000.0,
        "eeg_carrier_nV": eeg_carrier_v * 1.0e9,
        "field_v_per_m": field_v_per_m,
        "boundaries_s": (boundary_ms - start_ms) / 1000.0,
        "carrier_hz": carrier_hz,
        "update_interval_s": update_ms / 1000.0,
        "mean_abs_phase_error_rad": float(
            rows.phase_error_before_correction_rad.abs().mean()
        ),
    }


def _plot(
    *,
    signals: dict[str, dict[str, np.ndarray | float]],
    context_id: str,
    future_index: int,
    parameters: dict[str, float],
    output_path: Path,
    display_duration_s: float | None,
) -> None:
    labels = {
        ONE_TIME: "(1) One-time phase initialization",
        REFRESHED: "(2) Periodically refreshed phase control",
    }
    colors = {ONE_TIME: "#D97706", REFRESHED: "#2563EB"}
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 7.2),
        sharex="col",
        gridspec_kw={"height_ratios": [1.0, 1.0]},
        constrained_layout=False,
    )

    available_duration = min(
        float(np.max(signals[mode]["time_s"])) for mode in MODES
    )
    shown_duration = available_duration
    if display_duration_s is not None:
        shown_duration = min(float(display_duration_s), available_duration)
    if shown_duration <= 0.0:
        raise ValueError("display_duration_s must be positive.")

    eeg_limit = max(
        float(np.max(np.abs(signals[mode]["eeg_carrier_nV"]))) for mode in MODES
    )
    eeg_limit = max(1.05 * eeg_limit, np.finfo(float).eps)
    field_limit = 1.15 * float(parameters["amplitude_v_per_m"])

    for column, mode in enumerate(MODES):
        signal = signals[mode]
        time_s = np.asarray(signal["time_s"], dtype=float)
        shown = time_s <= shown_duration + 1.0e-12
        top, bottom = axes[0, column], axes[1, column]
        color = colors[mode]

        top.plot(
            time_s[shown],
            np.asarray(signal["eeg_carrier_nV"])[shown],
            color="black",
            linewidth=1.05,
        )
        bottom.plot(
            time_s[shown],
            np.asarray(signal["field_v_per_m"])[shown],
            color=color,
            linewidth=1.2,
        )
        for boundary in np.asarray(signal["boundaries_s"], dtype=float)[1:]:
            if boundary > shown_duration + 1.0e-12:
                continue
            line_color = color if mode == REFRESHED else "0.75"
            line_alpha = 0.35 if mode == REFRESHED else 0.22
            for axis in (top, bottom):
                axis.axvline(
                    boundary,
                    color=line_color,
                    alpha=line_alpha,
                    linewidth=0.8,
                    linestyle="--",
                    zorder=0,
                )

        top.set_title(labels[mode], fontsize=12, fontweight="bold", pad=10)
        top.set_ylim(-eeg_limit, eeg_limit)
        bottom.set_ylim(-field_limit, field_limit)
        top.grid(axis="y", alpha=0.18)
        bottom.grid(axis="y", alpha=0.18)
        bottom.axhline(0.0, color="0.5", linewidth=0.65)
        bottom.set_xlabel("Time from stimulation onset (s)")
        top.text(
            0.015,
            0.94,
            f"carrier = {float(signal['carrier_hz']):g} Hz",
            transform=top.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        bottom.text(
            0.015,
            0.94,
            (
                f"mean |phase error| = "
                f"{float(signal['mean_abs_phase_error_rad']):.2f} rad"
            ),
            transform=bottom.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        top.set_xlim(0.0, shown_duration)

    axes[0, 0].set_ylabel("Estimated EEG carrier (nV)")
    axes[1, 0].set_ylabel("Uniform electric field (V/m)")
    figure.suptitle(
        "Causal EEG phase tracking and the resulting continuous tACS waveform",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.012,
        (
            f"Context {context_id}, future {future_index}. EEG is the target-frequency "
            f"carrier reconstructed from rolling {parameters['phase_history_ms'] / 1000.0:g}-s "
            f"ideal-EEG Fourier estimates—not raw broadband EEG. Dashed lines mark "
            f"{parameters['refresh_interval_ms'] / 1000.0:g}-s controller boundaries; "
            "the refreshed field changes frequency without resetting phase."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.14, hspace=0.12, wspace=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_metadata(
    *,
    output_path: Path,
    source_dir: Path,
    context_id: str,
    future_index: int,
    parameters: dict[str, float],
    signals: dict[str, dict[str, np.ndarray | float]],
) -> None:
    metadata: dict[str, Any] = {
        "figure": str(output_path.resolve()),
        "source_result_directory": str(source_dir.resolve()),
        "context_id": context_id,
        "future_index": future_index,
        "parameters": parameters,
        "controller_summary": {
            mode: {
                "carrier_frequency_hz": float(signals[mode]["carrier_hz"]),
                "mean_absolute_phase_error_rad": float(
                    signals[mode]["mean_abs_phase_error_rad"]
                ),
            }
            for mode in MODES
        },
        "EEG_trace_scope": (
            "target-frequency carrier reconstructed from rolling causal Fourier "
            "estimates; not the raw broadband simulated EEG"
        ),
        "tACS_trace_scope": (
            "phase-continuous field command reconstructed from the saved oscillator "
            "phase, command frequency, amplitude, and global block envelope"
        ),
        "selection": "median phase-error-reduction pair unless explicitly requested",
        "not_an_efficacy_endpoint": True,
    }
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="D1-R phase_refresh_audit result directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. A matching PDF is also written.",
    )
    parser.add_argument("--context-id", default=None)
    parser.add_argument("--future-index", type=int, default=None)
    parser.add_argument(
        "--display-duration-s",
        type=float,
        default=2.0,
        help="Seconds shown from stimulation onset; use 0 for the full block.",
    )
    parser.add_argument("--samples-per-second", type=float, default=2000.0)
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    update_path = result_dir / "causal_phase_updates.csv"
    if not update_path.exists():
        raise FileNotFoundError(f"Phase-update table not found: {update_path}")
    updates = pd.read_csv(update_path)
    required = {
        "context_id",
        "future_index",
        "controller_mode",
        "update_index",
        "boundary_ms",
        "carrier_frequency_hz",
        "estimated_eeg_phase_at_boundary_rad",
        "eeg_resultant_v",
        "oscillator_phase_before_update_rad",
        "command_frequency_hz",
        "phase_error_before_correction_rad",
    }
    missing = required.difference(updates.columns)
    if missing:
        raise ValueError(f"Phase-update table is missing columns: {sorted(missing)}")

    parameters = _load_resolved_parameters(result_dir)
    context_id, future_index, selected = _choose_example(
        updates,
        context_id=args.context_id,
        future_index=args.future_index,
    )
    signals = {
        mode: _controller_signals(
            selected[selected.controller_mode.eq(mode)],
            amplitude_v_per_m=parameters["amplitude_v_per_m"],
            block_ramp_ms=parameters["block_ramp_ms"],
            samples_per_second=float(args.samples_per_second),
        )
        for mode in MODES
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else result_dir / "figure_06_phase_refresh_mechanics.png"
    )
    duration = None if float(args.display_duration_s) == 0.0 else float(
        args.display_duration_s
    )
    _plot(
        signals=signals,
        context_id=context_id,
        future_index=future_index,
        parameters=parameters,
        output_path=output_path,
        display_duration_s=duration,
    )
    _write_metadata(
        output_path=output_path,
        source_dir=result_dir,
        context_id=context_id,
        future_index=future_index,
        parameters=parameters,
        signals=signals,
    )
    print(f"Selected context: {context_id}, future: {future_index}")
    print(f"PNG saved to: {output_path}")
    print(f"PDF saved to: {output_path.with_suffix('.pdf')}")
    print(
        "EEG trace: reconstructed target-frequency carrier from causal phase "
        "estimates (not raw broadband EEG)."
    )


if __name__ == "__main__":
    main()
