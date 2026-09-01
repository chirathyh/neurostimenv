"""Step-wise extracellular stimulation for an in-memory NEURON simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import neuron
import numpy as np


@dataclass(frozen=True)
class AppliedExtracellularWaveform:
    """Metadata describing the waveform installed for one control window."""

    start_ms: float
    stop_ms: float
    sample_count: int
    peak_current_nA: float | None = None
    peak_field_v_per_m: float | None = None
    field_direction: tuple[float, float, float] | None = None
    parameterization: str = "point_source_current"
    integration_mode: str = "manual_fixed_step"


class OnlineExtracellularController:
    """Manage fixed-step LFPy-style extracellular inputs.

    The neural state is never reinitialised.  At each action boundary the old
    play vectors are detached, the next absolute-time field is cached, and the
    online integration loop assigns its value at each fixed-step boundary.
    """

    def __init__(
        self,
        *,
        electrode_index: int = 0,
        points_per_electrode: int = 5,
        field_model: str = "inf",
        field_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        if field_model not in {"inf", "semi"}:
            raise ValueError("field_model must be 'inf' or 'semi'.")
        if int(points_per_electrode) <= 0:
            raise ValueError("points_per_electrode must be positive.")
        self.electrode_index = int(electrode_index)
        self.points_per_electrode = int(points_per_electrode)
        self.field_model = field_model
        self.field_direction = self.normalize_field_direction(field_direction)

        self._active_time_ms = np.empty(0, dtype=np.float64)
        self._active_cell_fields: list[tuple[list, np.ndarray]] = []
        self._active_index = 0

    @staticmethod
    def normalize_field_direction(
        field_direction,
    ) -> np.ndarray:
        """Return a finite unit vector for a spatially uniform field."""
        direction = np.asarray(field_direction, dtype=np.float64).reshape(-1)
        if direction.size != 3 or not np.all(np.isfinite(direction)):
            raise ValueError("field_direction must contain three finite values.")
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("field_direction must be non-zero.")
        return direction / norm

    @classmethod
    def uniform_field_potential_mV(
        cls,
        *,
        midpoints_um: np.ndarray,
        field_v_per_m: np.ndarray,
        field_direction,
    ) -> np.ndarray:
        """Return segment extracellular voltage for one uniform vector field.

        Rows correspond to segment midpoints and columns to field samples.  A
        per-cell constant potential is removed because it cannot change the
        transmembrane polarization.
        """
        midpoints_um = np.asarray(midpoints_um, dtype=np.float64)
        field_v_per_m = np.asarray(field_v_per_m, dtype=np.float64).reshape(-1)
        if midpoints_um.ndim != 2 or midpoints_um.shape[1] != 3:
            raise ValueError("midpoints_um must have shape (n_segments, 3).")
        if not np.all(np.isfinite(midpoints_um)) or not np.all(
            np.isfinite(field_v_per_m)
        ):
            raise ValueError("Uniform-field geometry and waveform must be finite.")
        direction = cls.normalize_field_direction(field_direction)
        centered_um = midpoints_um - np.mean(midpoints_um, axis=0)
        projected_um = centered_um @ direction
        return (
            -projected_um[:, np.newaxis]
            * field_v_per_m[np.newaxis, :]
            * 1e-3
        )

    @staticmethod
    def _iter_cells(network) -> Iterator:
        if network is None:
            return
        populations = getattr(network, "populations", None)
        population_names = getattr(network, "population_names", None)
        if populations is None or population_names is None:
            return

        for population_name in list(population_names):
            population = populations.get(population_name)
            if population is None:
                continue
            cells = getattr(population, "cells", None)
            if cells is None:
                continue
            for cell in list(cells):
                if cell is not None:
                    yield cell

    @staticmethod
    def _iter_sections(cell) -> Iterator:
        sections = getattr(cell, "allseclist", None)
        if sections is None:
            return
        try:
            for section in sections:
                yield section
        except TypeError:
            # LFPy cleanup may already have invalidated the SectionList.
            return

    def prepare_network(self, network) -> None:
        """Insert ``extracellular`` before the one-time ``finitialize`` call."""
        for cell in self._iter_cells(network):
            for section in self._iter_sections(cell):
                section.insert("extracellular")
            cell.extracellular = True
            self._clear_cell_playback(cell, set_zero=True)

    def _clear_cell_playback(self, cell, *, set_zero: bool) -> None:
        old_vectors = getattr(cell, "v_ext", None)
        if old_vectors is not None:
            try:
                iterable = list(old_vectors)
            except TypeError:
                iterable = []
            for vector in iterable:
                try:
                    # NEURON documents play_remove() as detaching the Vector
                    # from both record and play lists.
                    vector.play_remove()
                except Exception:
                    pass

        cell.v_ext = []
        cell.t_ext = None

        if set_zero:
            for section in self._iter_sections(cell):
                for segment in section:
                    try:
                        segment.e_extracellular = 0.0
                    except (AttributeError, ReferenceError):
                        pass

    def clear(self, network) -> None:
        """Detach active stimulation vectors and set external voltage to zero."""
        self._active_time_ms = np.empty(0, dtype=np.float64)
        self._active_cell_fields = []
        self._active_index = 0
        for cell in self._iter_cells(network):
            self._clear_cell_playback(cell, set_zero=True)
        # Recompute assigned currents at the unchanged dynamic state.  This is
        # not a reinitialisation and does not clear the event queue.
        try:
            neuron.h.fcurrent()
        except Exception:
            pass

    def max_abs_extracellular(self, network) -> float:
        """Return the largest current extracellular voltage over local cells."""
        maximum = 0.0
        for cell in self._iter_cells(network):
            for section in self._iter_sections(cell):
                for segment in section:
                    try:
                        maximum = max(
                            maximum,
                            abs(float(segment.e_extracellular)),
                        )
                    except (AttributeError, ReferenceError):
                        continue
        return maximum

    def apply_waveform(
        self,
        *,
        network,
        electrode,
        current_nA: np.ndarray,
        time_ms: np.ndarray,
    ) -> AppliedExtracellularWaveform:
        """Install one absolute-time waveform for the next online interval."""
        current_nA = np.asarray(current_nA, dtype=np.float64).reshape(-1)
        time_ms = np.asarray(time_ms, dtype=np.float64).reshape(-1)

        if current_nA.size != time_ms.size:
            raise ValueError("current_nA and time_ms must have equal length.")
        if current_nA.size < 2:
            raise ValueError("At least two stimulation samples are required.")
        if not np.all(np.isfinite(current_nA)) or not np.all(np.isfinite(time_ms)):
            raise ValueError("Stimulation arrays must contain finite values.")
        if not np.all(np.diff(time_ms) > 0):
            raise ValueError("time_ms must be strictly increasing.")

        current_time_ms = float(neuron.h.t)
        tolerance = max(1e-8, float(getattr(network, "dt", 0.0)) * 1e-5)
        if abs(float(time_ms[0]) - current_time_ms) > tolerance:
            raise ValueError(
                "The online waveform must start at the current NEURON time: "
                f"waveform starts at {time_ms[0]} ms, NEURON is at "
                f"{current_time_ms} ms."
            )

        # This is the same probe current API used by the legacy implementation.
        electrode.probe.set_current(self.electrode_index, current_nA)

        zero_waveform = bool(np.allclose(current_nA, 0.0, rtol=0.0, atol=0.0))
        self._active_time_ms = time_ms.copy()
        self._active_cell_fields = []
        self._active_index = 0
        for cell in self._iter_cells(network):
            self._clear_cell_playback(cell, set_zero=True)
            if zero_waveform:
                continue

            cell_mid_points = np.column_stack(
                (
                    np.asarray(cell.x).mean(axis=-1),
                    np.asarray(cell.y).mean(axis=-1),
                    np.asarray(cell.z).mean(axis=-1),
                )
            )

            # Force a geometry-specific mapping for this cell, matching the
            # existing LFPy-based implementation.
            if hasattr(electrode.probe, "electrodes"):
                electrodes = electrode.probe.electrodes
                electrodes[self.electrode_index].mapping = None
            electrode.probe.points_per_electrode = self.points_per_electrode
            electrode.probe.model = self.field_model

            v_ext_mV = np.asarray(
                electrode.probe.compute_field(cell_mid_points),
                dtype=np.float64,
            )
            if v_ext_mV.ndim == 1:
                v_ext_mV = v_ext_mV[:, np.newaxis]
            expected_shape = (int(cell.totnsegs), int(current_nA.size))
            if v_ext_mV.shape != expected_shape:
                raise RuntimeError(
                    "Unexpected extracellular field shape: "
                    f"got {v_ext_mV.shape}, expected {expected_shape}."
                )

            segments = []
            for section in self._iter_sections(cell):
                for segment in section:
                    segments.append(segment)

            if len(segments) != int(cell.totnsegs):
                raise RuntimeError(
                    f"Prepared stimulation on {len(segments)} segments; "
                    f"cell reports {cell.totnsegs}."
                )
            self._active_cell_fields.append((segments, v_ext_mV))

        # Vector.play registered after finitialize does not begin playback in
        # NEURON 8.2.3.  The small online BallAndStick loop therefore assigns
        # these cached field values explicitly at each fixed step.
        self.set_time(float(time_ms[0]))
        neuron.h.fcurrent()

        return AppliedExtracellularWaveform(
            start_ms=float(time_ms[0]),
            stop_ms=float(time_ms[-1]),
            sample_count=int(time_ms.size),
            peak_current_nA=float(np.max(np.abs(current_nA))),
            parameterization="point_source_current",
        )

    def apply_uniform_field(
        self,
        *,
        network,
        field_v_per_m: np.ndarray,
        time_ms: np.ndarray,
        field_direction=None,
    ) -> AppliedExtracellularWaveform:
        """Install a spatially uniform field for the next online interval.

        For a uniform field ``E`` along unit vector ``d``, extracellular
        potential is ``phi(r) = -E d·r``.  With positions in micrometres and
        NEURON extracellular voltage in millivolts, the conversion factor is
        ``1e-3 mV / (V/m * um)``.  A constant reference potential is immaterial
        to transmembrane polarization, so coordinates are centered per cell to
        keep values numerically small.
        """
        field_v_per_m = np.asarray(field_v_per_m, dtype=np.float64).reshape(-1)
        time_ms = np.asarray(time_ms, dtype=np.float64).reshape(-1)
        self._validate_waveform_axis(
            values=field_v_per_m,
            time_ms=time_ms,
            network=network,
            value_name="field_v_per_m",
        )

        direction = self.normalize_field_direction(
            self.field_direction if field_direction is None else field_direction
        )
        zero_waveform = bool(
            np.allclose(field_v_per_m, 0.0, rtol=0.0, atol=0.0)
        )
        self._active_time_ms = time_ms.copy()
        self._active_cell_fields = []
        self._active_index = 0

        for cell in self._iter_cells(network):
            self._clear_cell_playback(cell, set_zero=True)
            if zero_waveform:
                continue

            midpoints_um = np.column_stack(
                (
                    np.asarray(cell.x).mean(axis=-1),
                    np.asarray(cell.y).mean(axis=-1),
                    np.asarray(cell.z).mean(axis=-1),
                )
            )
            v_ext_mV = self.uniform_field_potential_mV(
                midpoints_um=midpoints_um,
                field_v_per_m=field_v_per_m,
                field_direction=direction,
            )

            segments = [
                segment
                for section in self._iter_sections(cell)
                for segment in section
            ]
            if len(segments) != int(cell.totnsegs):
                raise RuntimeError(
                    f"Prepared stimulation on {len(segments)} segments; "
                    f"cell reports {cell.totnsegs}."
                )
            self._active_cell_fields.append((segments, v_ext_mV))

        self.set_time(float(time_ms[0]))
        neuron.h.fcurrent()
        return AppliedExtracellularWaveform(
            start_ms=float(time_ms[0]),
            stop_ms=float(time_ms[-1]),
            sample_count=int(time_ms.size),
            peak_field_v_per_m=float(np.max(np.abs(field_v_per_m))),
            field_direction=tuple(float(value) for value in direction),
            parameterization="uniform_field",
        )

    @staticmethod
    def _validate_waveform_axis(
        *,
        values: np.ndarray,
        time_ms: np.ndarray,
        network,
        value_name: str,
    ) -> None:
        if values.size != time_ms.size:
            raise ValueError(f"{value_name} and time_ms must have equal length.")
        if values.size < 2:
            raise ValueError("At least two stimulation samples are required.")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(time_ms)):
            raise ValueError("Stimulation arrays must contain finite values.")
        if not np.all(np.diff(time_ms) > 0):
            raise ValueError("time_ms must be strictly increasing.")

        current_time_ms = float(neuron.h.t)
        tolerance = max(1e-8, float(getattr(network, "dt", 0.0)) * 1e-5)
        if abs(float(time_ms[0]) - current_time_ms) > tolerance:
            raise ValueError(
                "The online waveform must start at the current NEURON time: "
                f"waveform starts at {time_ms[0]} ms, NEURON is at "
                f"{current_time_ms} ms."
            )

    def set_time(self, time_ms: float) -> None:
        """Set the extracellular value for one fixed-step left boundary."""
        if self._active_time_ms.size == 0 or not self._active_cell_fields:
            return

        time_ms = float(time_ms)
        dt_ms = (
            float(self._active_time_ms[1] - self._active_time_ms[0])
            if self._active_time_ms.size > 1
            else 0.0
        )
        tolerance = max(1e-8, abs(dt_ms) * 1e-5)

        # Calls are monotonic within a window, so advance from the prior index
        # and avoid a search at every 0.0625-ms step.
        while (
            self._active_index + 1 < self._active_time_ms.size
            and self._active_time_ms[self._active_index] < time_ms - tolerance
        ):
            self._active_index += 1

        scheduled_ms = float(self._active_time_ms[self._active_index])
        if abs(scheduled_ms - time_ms) > tolerance:
            raise RuntimeError(
                "No stimulation sample matches the NEURON integration "
                f"boundary: requested={time_ms}, scheduled={scheduled_ms}."
            )

        for segments, field_mV in self._active_cell_fields:
            values = field_mV[:, self._active_index]
            for segment, value in zip(segments, values):
                segment.e_extracellular = float(value)
