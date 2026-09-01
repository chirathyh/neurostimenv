"""Waveform generation utilities for NeuroStimEnv.

This module is additive: it does not modify the existing ``prep_stim_seq``
(square/biphasic pulse) implementation.  It provides sinusoidal waveforms for
both precomputed rollouts and step-wise online control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


_MA_TO_NA = 1e6
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class StimulationWaveform:
    """A current waveform sampled on an absolute NEURON time axis."""

    time_ms: np.ndarray
    current_nA: np.ndarray
    final_phase_rad: float


@dataclass(frozen=True)
class ElectricFieldWaveform:
    """A spatially uniform electric-field waveform on an absolute time axis."""

    time_ms: np.ndarray
    field_v_per_m: np.ndarray
    final_phase_rad: float


def _parse_action(action: Sequence[float] | Mapping[str, float]) -> tuple[float, float]:
    """Return ``(amplitude_mA, frequency_hz)`` from an action."""
    if isinstance(action, Mapping):
        amplitude_mA = float(action["amplitude_mA"])
        frequency_hz = float(action["frequency_hz"])
    else:
        if len(action) != 2:
            raise ValueError("An action must contain [amplitude_mA, frequency_hz].")
        amplitude_mA = float(action[0])
        frequency_hz = float(action[1])

    if amplitude_mA < 0:
        raise ValueError(f"amplitude_mA must be non-negative, got {amplitude_mA}.")
    if frequency_hz < 0:
        raise ValueError(f"frequency_hz must be non-negative, got {frequency_hz}.")
    return amplitude_mA, frequency_hz


def _raised_cosine_envelope(
    relative_time_ms: np.ndarray,
    duration_ms: float,
    ramp_ms: float,
) -> np.ndarray:
    """Return a symmetric raised-cosine onset/offset envelope."""
    envelope = np.ones_like(relative_time_ms, dtype=np.float64)
    if ramp_ms <= 0:
        return envelope

    effective_ramp_ms = min(float(ramp_ms), float(duration_ms) / 2.0)
    onset = relative_time_ms < effective_ramp_ms
    offset = relative_time_ms > (duration_ms - effective_ramp_ms)

    envelope[onset] = 0.5 * (
        1.0 - np.cos(np.pi * relative_time_ms[onset] / effective_ramp_ms)
    )
    envelope[offset] = 0.5 * (
        1.0
        - np.cos(
            np.pi * (duration_ms - relative_time_ms[offset]) / effective_ramp_ms
        )
    )
    return envelope


def apply_raised_cosine_block_envelope(
    values: np.ndarray,
    *,
    time_ms: np.ndarray,
    block_start_ms: float,
    block_stop_ms: float,
    ramp_ms: float,
) -> np.ndarray:
    """Apply one absolute-time raised-cosine envelope across many windows.

    Unlike the per-waveform ``ramp_ms`` argument, this helper does not restart
    the ramp at each online decision boundary.  Supplying consecutive slices
    of one block therefore gives the same samples as enveloping the complete
    block at once.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    time_ms = np.asarray(time_ms, dtype=np.float64).reshape(-1)
    block_start_ms = float(block_start_ms)
    block_stop_ms = float(block_stop_ms)
    ramp_ms = float(ramp_ms)

    if values.size != time_ms.size:
        raise ValueError("values and time_ms must have equal length.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(time_ms)):
        raise ValueError("values and time_ms must contain only finite values.")
    if block_stop_ms <= block_start_ms:
        raise ValueError("block_stop_ms must exceed block_start_ms.")
    if ramp_ms < 0:
        raise ValueError("ramp_ms must be non-negative.")

    duration_ms = block_stop_ms - block_start_ms
    inside = (time_ms >= block_start_ms) & (time_ms <= block_stop_ms)
    relative_time_ms = np.clip(
        time_ms - block_start_ms,
        0.0,
        duration_ms,
    )
    envelope = _raised_cosine_envelope(
        relative_time_ms=relative_time_ms,
        duration_ms=duration_ms,
        ramp_ms=ramp_ms,
    )
    envelope[~inside] = 0.0
    return values * envelope


def make_sinusoidal_stimulation(
    *,
    amplitude_mA: float,
    frequency_hz: float,
    start_ms: float,
    duration_ms: float,
    dt_ms: float,
    phase_rad: float = 0.0,
    dc_offset_mA: float = 0.0,
    ramp_ms: float = 0.0,
    include_endpoint: bool = True,
) -> StimulationWaveform:
    """Generate a sinusoidal extracellular-current waveform.

    The waveform is returned in nA because this is the current unit used by the
    existing LFPy electrode code in NeuroStimEnv.  Time values are absolute
    NEURON times in milliseconds, which allows the waveform to be installed
    after the simulation has already advanced.

    ``phase_rad`` is the phase at ``start_ms``.  The returned final phase can be
    supplied to the next call to keep phase continuous across control windows.
    """
    amplitude_mA = float(amplitude_mA)
    frequency_hz = float(frequency_hz)
    start_ms = float(start_ms)
    duration_ms = float(duration_ms)
    dt_ms = float(dt_ms)
    phase_rad = float(phase_rad)
    dc_offset_mA = float(dc_offset_mA)

    if amplitude_mA < 0:
        raise ValueError("amplitude_mA must be non-negative.")
    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative.")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive.")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive.")

    n_intervals = int(round(duration_ms / dt_ms))
    if not np.isclose(n_intervals * dt_ms, duration_ms, rtol=0.0, atol=1e-9):
        raise ValueError(
            "duration_ms must be an integer multiple of dt_ms for fixed-step "
            f"simulation; got duration_ms={duration_ms}, dt_ms={dt_ms}."
        )

    sample_count = n_intervals + int(include_endpoint)
    relative_time_ms = np.arange(sample_count, dtype=np.float64) * dt_ms
    time_ms = start_ms + relative_time_ms

    phase = phase_rad + _TWO_PI * frequency_hz * (relative_time_ms / 1000.0)
    current_mA = dc_offset_mA + amplitude_mA * np.sin(phase)
    current_mA *= _raised_cosine_envelope(
        relative_time_ms=relative_time_ms,
        duration_ms=duration_ms,
        ramp_ms=float(ramp_ms),
    )

    final_phase_rad = float(
        np.mod(phase_rad + _TWO_PI * frequency_hz * duration_ms / 1000.0, _TWO_PI)
    )
    return StimulationWaveform(
        time_ms=time_ms,
        current_nA=current_mA * _MA_TO_NA,
        final_phase_rad=final_phase_rad,
    )


def make_zero_stimulation(
    *,
    start_ms: float,
    duration_ms: float,
    dt_ms: float,
    include_endpoint: bool = True,
) -> StimulationWaveform:
    """Generate an explicit zero-current waveform."""
    return make_sinusoidal_stimulation(
        amplitude_mA=0.0,
        frequency_hz=0.0,
        start_ms=start_ms,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        include_endpoint=include_endpoint,
    )


def make_sinusoidal_electric_field(
    *,
    amplitude_v_per_m: float,
    frequency_hz: float,
    start_ms: float,
    duration_ms: float,
    dt_ms: float,
    phase_rad: float = 0.0,
    dc_offset_v_per_m: float = 0.0,
    ramp_ms: float = 0.0,
    initial_amplitude_v_per_m: float | None = None,
    amplitude_transition_ms: float = 0.0,
    include_endpoint: bool = True,
) -> ElectricFieldWaveform:
    """Generate a spatially uniform DC-plus-AC electric-field waveform.

    Field strength is the biologically interpretable quantity at the modeled
    tissue.  It deliberately avoids pretending that a microscopic point-source
    current can be mapped directly to a scalp-current dose.
    """
    amplitude_v_per_m = float(amplitude_v_per_m)
    frequency_hz = float(frequency_hz)
    start_ms = float(start_ms)
    duration_ms = float(duration_ms)
    dt_ms = float(dt_ms)
    phase_rad = float(phase_rad)
    dc_offset_v_per_m = float(dc_offset_v_per_m)
    initial_amplitude_v_per_m = (
        amplitude_v_per_m
        if initial_amplitude_v_per_m is None
        else float(initial_amplitude_v_per_m)
    )
    amplitude_transition_ms = float(amplitude_transition_ms)

    if amplitude_v_per_m < 0:
        raise ValueError("amplitude_v_per_m must be non-negative.")
    if initial_amplitude_v_per_m < 0:
        raise ValueError("initial_amplitude_v_per_m must be non-negative.")
    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative.")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive.")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive.")
    if not np.all(
        np.isfinite(
            [
                amplitude_v_per_m,
                frequency_hz,
                start_ms,
                duration_ms,
                dt_ms,
                phase_rad,
                dc_offset_v_per_m,
                ramp_ms,
                initial_amplitude_v_per_m,
                amplitude_transition_ms,
            ]
        )
    ):
        raise ValueError("Electric-field waveform parameters must be finite.")
    if amplitude_transition_ms < 0.0:
        raise ValueError("amplitude_transition_ms must be non-negative.")
    if amplitude_transition_ms > duration_ms:
        raise ValueError(
            "amplitude_transition_ms cannot exceed the waveform duration."
        )

    n_intervals = int(round(duration_ms / dt_ms))
    if not np.isclose(n_intervals * dt_ms, duration_ms, rtol=0.0, atol=1e-9):
        raise ValueError(
            "duration_ms must be an integer multiple of dt_ms for fixed-step "
            f"simulation; got duration_ms={duration_ms}, dt_ms={dt_ms}."
        )

    sample_count = n_intervals + int(include_endpoint)
    relative_time_ms = np.arange(sample_count, dtype=np.float64) * dt_ms
    time_ms = start_ms + relative_time_ms
    phase = phase_rad + _TWO_PI * frequency_hz * (relative_time_ms / 1000.0)
    amplitude_profile = np.full_like(
        relative_time_ms, amplitude_v_per_m, dtype=np.float64
    )
    if amplitude_transition_ms > 0.0:
        transitioning = relative_time_ms < amplitude_transition_ms
        progress = relative_time_ms[transitioning] / amplitude_transition_ms
        interpolation = 0.5 * (1.0 - np.cos(np.pi * progress))
        amplitude_profile[transitioning] = (
            initial_amplitude_v_per_m
            + (amplitude_v_per_m - initial_amplitude_v_per_m) * interpolation
        )
    field_v_per_m = (
        dc_offset_v_per_m + amplitude_profile * np.sin(phase)
    )
    field_v_per_m *= _raised_cosine_envelope(
        relative_time_ms=relative_time_ms,
        duration_ms=duration_ms,
        ramp_ms=float(ramp_ms),
    )
    final_phase_rad = float(
        np.mod(
            phase_rad + _TWO_PI * frequency_hz * duration_ms / 1000.0,
            _TWO_PI,
        )
    )
    return ElectricFieldWaveform(
        time_ms=time_ms,
        field_v_per_m=field_v_per_m,
        final_phase_rad=final_phase_rad,
    )


def prepare_sinusoidal_stim_seq(
    *,
    actions: Iterable[Sequence[float] | Mapping[str, float]],
    step_size_ms: float,
    dt_ms: float,
    start_ms: float = 0.0,
    initial_phase_rad: float = 0.0,
    phase_continuous: bool = True,
    ramp_ms: float = 0.0,
) -> StimulationWaveform:
    """Precompute a multi-step sinusoidal sequence for legacy rollouts.

    This is the sinusoidal counterpart of the existing ``prep_stim_seq``.
    Boundary samples are de-duplicated when the windows are concatenated.
    """
    current_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    phase_rad = float(initial_phase_rad)
    window_start_ms = float(start_ms)

    actions_list = list(actions)
    if not actions_list:
        raise ValueError("actions must contain at least one action.")

    for action_index, action in enumerate(actions_list):
        amplitude_mA, frequency_hz = _parse_action(action)
        waveform = make_sinusoidal_stimulation(
            amplitude_mA=amplitude_mA,
            frequency_hz=frequency_hz,
            start_ms=window_start_ms,
            duration_ms=step_size_ms,
            dt_ms=dt_ms,
            phase_rad=phase_rad if phase_continuous else initial_phase_rad,
            ramp_ms=ramp_ms,
            include_endpoint=True,
        )

        if action_index < len(actions_list) - 1:
            time_parts.append(waveform.time_ms[:-1])
            current_parts.append(waveform.current_nA[:-1])
        else:
            time_parts.append(waveform.time_ms)
            current_parts.append(waveform.current_nA)

        phase_rad = waveform.final_phase_rad
        window_start_ms += float(step_size_ms)

    return StimulationWaveform(
        time_ms=np.concatenate(time_parts),
        current_nA=np.concatenate(current_parts),
        final_phase_rad=phase_rad,
    )
