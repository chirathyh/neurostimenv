"""Step-wise LFPy/NEURON network execution.

This module is additive.  It leaves the legacy ``NetworkEnv.simulate`` path
unchanged and provides an in-process online path that initialises NEURON once,
then advances the same model state with a small fixed-step recording loop.
"""

from __future__ import annotations

from typing import Any

import neuron
import numpy as np
from mpi4py import MPI
from neuron import units

from env.models.neuron.networkenv import NetworkEnv


def hoc_vector_to_numpy(vector) -> np.ndarray:
    """Copy a NEURON ``Vector`` into a NumPy float64 array.

    ``np.asarray(h.Vector(), dtype=...)`` is not reliable with all NEURON
    8.2.x Python wrappers and can produce an object containing a
    ``hoc.HocObject``.  ``Vector.as_numpy()`` is the preferred zero-copy view;
    it is copied immediately because the NEURON vector may subsequently be
    detached or resized.
    """
    if vector is None:
        return np.empty(0, dtype=np.float64)

    as_numpy = getattr(vector, "as_numpy", None)
    if callable(as_numpy):
        try:
            return np.array(as_numpy(), dtype=np.float64, copy=True)
        except (TypeError, ValueError, RuntimeError):
            pass

    to_python = getattr(vector, "to_python", None)
    if callable(to_python):
        try:
            return np.asarray(to_python(), dtype=np.float64)
        except (TypeError, ValueError, RuntimeError):
            pass

    # Last-resort fallback for older wrappers.
    size_method = getattr(vector, "size", None)
    size = int(size_method()) if callable(size_method) else len(vector)
    return np.fromiter(
        (float(vector[index]) for index in range(size)),
        dtype=np.float64,
        count=size,
    )


class OnlineNetworkEnv(NetworkEnv):
    """``NetworkEnv`` variant supporting control-window-level online stepping."""

    def __init__(self, **network_parameters: Any) -> None:
        super().__init__(**network_parameters)
        self._online_initialized = False
        self._online_comm = None
        self._online_rank: int | None = None
        self._online_probes: list[Any] = []
        self._online_transforms: list[np.ndarray] = []
        self._online_segments: list[Any] = []
        self._online_representative_sites: list[dict[str, Any]] = []
        self._online_representative_stride_steps = 1
        self._online_cvode = None
        self._online_integration_method = "manual_fadvance"

    @property
    def current_time_ms(self) -> float:
        return float(neuron.h.t)

    def initialize_online(
        self,
        *,
        probes: list[Any],
        comm,
        max_step_ms: float = 10.0,
        temperature_mode: str = "configured",
        record_representative_state: bool = False,
        representative_state_interval_ms: float | None = None,
    ) -> None:
        """Initialise the network once without running the full episode.

        No later online step calls ``finitialize``; therefore membrane states,
        synaptic states and the event queue are preserved across decisions.
        """
        if self._online_initialized:
            raise RuntimeError("OnlineNetworkEnv has already been initialised.")
        if not probes:
            raise ValueError("At least one LFPy probe is required.")
        if float(max_step_ms) <= 0:
            raise ValueError("max_step_ms must be positive.")
        if temperature_mode not in {"configured", "legacy_default"}:
            raise ValueError(
                "temperature_mode must be 'configured' or 'legacy_default'."
            )

        self._online_comm = comm
        self._online_rank = int(comm.Get_rank())
        self._online_probes = list(probes)

        # Build the same local 'super-cell' geometry that LFPy uses internally
        # for network probe transformations.  This is a private LFPy API, so the
        # online path should remain pinned to the tested LFPy version.
        _, network_dummy_cell = self._Network__create_network_dummycell()
        self._online_transforms = []
        for probe in self._online_probes:
            if probe.cell is not None:
                raise ValueError(f"{probe.__class__.__name__}.cell must be None.")
            probe.cell = network_dummy_cell
            try:
                transform = np.asarray(
                    probe.get_transformation_matrix(),
                    dtype=np.float64,
                )
            finally:
                probe.cell = None
            self._online_transforms.append(transform)

        # Match the segment ordering used by LFPy's network dummy cell:
        # population order -> cell order -> section order -> segment order.
        self._online_segments = []
        for population_name in self.population_names:
            population = self.populations[population_name]
            cells = getattr(population, "cells", None) or []
            for cell in cells:
                sections = getattr(cell, "allseclist", None)
                if sections is None:
                    continue
                for section in sections:
                    for segment in section:
                        self._online_segments.append(segment)

        if not self._online_transforms:
            raise RuntimeError("No probe transformation matrices were created.")

        expected_local_segments = self._online_transforms[0].shape[1]
        if expected_local_segments != len(self._online_segments):
            raise RuntimeError(
                "Probe geometry and segment ordering disagree: "
                f"transform has {expected_local_segments} columns but "
                f"{len(self._online_segments)} local segments were found."
            )
        for transform in self._online_transforms[1:]:
            if transform.shape[1] != expected_local_segments:
                raise RuntimeError(
                    "Probe transformations disagree on local segment count."
                )

        self._online_representative_sites = []
        if bool(record_representative_state):
            requested_interval = (
                float(self.dt) if representative_state_interval_ms is None
                else float(representative_state_interval_ms)
            )
            stride = int(round(requested_interval / float(self.dt)))
            if stride < 1 or not np.isclose(
                stride * float(self.dt), requested_interval, atol=1.0e-12
            ):
                raise ValueError(
                    "representative_state_interval_ms must be a positive "
                    "integer multiple of the simulator dt."
                )
            self._online_representative_stride_steps = stride
            self._prepare_representative_sites()

        self.pc.set_maxstep(float(max_step_ms))
        neuron.h.dt = float(self.dt)
        # LFPy 2.3 Network.simulate stores Network.celsius but, in the version
        # pinned by this project, does not assign global h.celsius.  Make the
        # choice explicit: scientific runs use the configured model-native
        # temperature; exact regression against old runs can request the
        # legacy NEURON default.
        self._online_temperature_mode = str(temperature_mode)
        if temperature_mode == "configured":
            neuron.h.celsius = float(self.celsius)
        else:
            # Reproduce the effective value of the historical LFPy path even
            # when another episode in this Python process changed h.celsius.
            neuron.h.celsius = 6.3

        self._online_cvode = neuron.h.CVode()
        self._online_cvode.use_fast_imem(1)
        self._online_cvode.active(0)  # this implementation assumes fixed dt

        # Cell soma/spike recorders are constructed by LFPy while the network is
        # built, so this single frecord_init initialises them.  Do not attach
        # temporary Vector.record instances later: in NEURON 8.2.3 those
        # vectors remain empty until another frecord_init, which would also
        # reset all existing recorders.
        neuron.h.finitialize(float(self.v_init) * units.mV)
        neuron.h.fcurrent()
        neuron.h.frecord_init()
        neuron.h.t = float(self.tstart)

        # Match LFPy's Network.simulate ordering: load externally specified
        # Synapse event trains after finitialize/frecord_init and before the
        # first continuation call.
        for population_name in self.population_names:
            population = self.populations[population_name]
            for cell in (getattr(population, "cells", None) or []):
                cell._load_spikes()

        self._online_initialized = True

    def online_diagnostics(self) -> dict[str, Any]:
        """Return inexpensive state useful when diagnosing continuation."""
        local_cells = 0
        for population_name in self.population_names:
            population = self.populations[population_name]
            local_cells += len(getattr(population, "cells", None) or [])

        imem_refs_available = all(
            hasattr(segment, "_ref_i_membrane_")
            for segment in self._online_segments
        )
        spike_vector_sizes = {
            population_name: [
                int(vector.size())
                for vector in getattr(
                    self.populations[population_name],
                    "_hoc_spike_vectors",
                    [],
                )
            ]
            for population_name in self.population_names
        }
        return {
            "integration_method": self._online_integration_method,
            "h_t_ms": self.current_time_ms,
            "local_cell_count": int(local_cells),
            "local_segment_count": int(len(self._online_segments)),
            "local_representative_site_count": int(
                len(self._online_representative_sites)
            ),
            "representative_state_interval_ms": float(
                self._online_representative_stride_steps * self.dt
            ),
            "i_membrane_ref_available": bool(imem_refs_available),
            "spike_vector_sizes": spike_vector_sizes,
            "fast_imem_enabled": bool(self._online_cvode.use_fast_imem()),
            "fixed_dt_ms": float(self.dt),
            "configured_celsius": float(self.celsius),
            "effective_h_celsius": float(neuron.h.celsius),
            "temperature_mode": self._online_temperature_mode,
        }

    @staticmethod
    def _section_by_label(cell: Any, label: str):
        """Return the first cell section whose HOC name contains ``label``."""
        sections = getattr(cell, "allseclist", None)
        if sections is None:
            return None
        for section in sections:
            if str(label).lower() in str(section.name()).lower():
                return section
        return None

    def _prepare_representative_sites(self) -> None:
        """Select the globally lowest-GID E and I cell for optional auditing.

        The sites are sampled manually inside the fixed-step loop.  This avoids
        attaching new ``Vector.record`` objects after ``finitialize()``, which
        is unsafe for the persistent LFPy recorders under NEURON 8.2.3.
        """
        local_minima: dict[str, int | None] = {}
        for population_name in self.population_names:
            gids = list(getattr(self.populations[population_name], "gids", []))
            local_minima[str(population_name)] = (
                min(int(gid) for gid in gids) if gids else None
            )
        gathered = self._online_comm.allgather(local_minima)
        global_minima = {
            str(population_name): min(
                int(rank_values[str(population_name)])
                for rank_values in gathered
                if rank_values[str(population_name)] is not None
            )
            for population_name in self.population_names
        }

        for population_name in self.population_names:
            population = self.populations[population_name]
            cells = list(getattr(population, "cells", None) or [])
            gids = list(getattr(population, "gids", []))
            if len(cells) != len(gids):
                raise RuntimeError(
                    f"Population {population_name}: cell/GID count mismatch "
                    f"({len(cells)} vs {len(gids)})."
                )
            target_gid = int(global_minima[str(population_name)])
            for cell, gid in zip(cells, gids):
                if int(gid) != target_gid:
                    continue
                soma = self._section_by_label(cell, "soma")
                apic = self._section_by_label(cell, "apic")
                if soma is None or apic is None:
                    raise RuntimeError(
                        f"Representative {population_name} cell {target_gid} "
                        "does not expose soma and apic sections."
                    )
                soma_segment = soma(0.5)
                apic_segment = apic(0.9)
                self._online_representative_sites.append({
                    "population": str(population_name),
                    "gid": target_gid,
                    "soma_segment": soma_segment,
                    "apic_segment": apic_segment,
                    "soma_area_um2": float(soma_segment.area()),
                    "apic_area_um2": float(apic_segment.area()),
                })
                break

    @staticmethod
    def _density_current_nA(segment: Any, name: str, area_um2: float) -> float:
        """Convert an available NEURON density current to segment current."""
        value = getattr(segment, name, None)
        if value is None:
            return float("nan")
        # mA/cm2 * um2 = 1e-2 nA.
        return float(value) * float(area_um2) * 1.0e-2

    def _sample_representative_site(self, site: dict[str, Any]) -> dict[str, float]:
        soma = site["soma_segment"]
        apic = site["apic_segment"]
        soma_area = float(site["soma_area_um2"])
        apic_area = float(site["apic_area_um2"])
        values = {
            "soma_voltage_mV": float(soma.v),
            "apic_distal_voltage_mV": float(apic.v),
            "soma_total_membrane_current_nA": float(soma.i_membrane_),
            "apic_total_membrane_current_nA": float(apic.i_membrane_),
            "soma_capacitive_current_nA": self._density_current_nA(
                soma, "i_cap", soma_area
            ),
            "apic_capacitive_current_nA": self._density_current_nA(
                apic, "i_cap", apic_area
            ),
            "soma_passive_current_nA": self._density_current_nA(
                soma, "i_pas", soma_area
            ),
            "apic_passive_current_nA": self._density_current_nA(
                apic, "i_pas", apic_area
            ),
            "soma_sodium_current_nA": self._density_current_nA(
                soma, "ina", soma_area
            ),
            "soma_potassium_current_nA": self._density_current_nA(
                soma, "ik", soma_area
            ),
        }
        # A canonical HH soma has no passive-density mechanism, whereas the
        # apical cable does. Omit unavailable mechanism-specific currents
        # instead of writing NaNs that could be mistaken for numerical failure.
        return {name: value for name, value in values.items() if np.isfinite(value)}

    def _read_local_membrane_currents(self) -> np.ndarray:
        """Read fast membrane currents in cached LFPy dummy-cell order."""
        currents = np.empty(len(self._online_segments), dtype=np.float64)
        for index, segment in enumerate(self._online_segments):
            try:
                currents[index] = float(segment.i_membrane_)
            except (AttributeError, ReferenceError) as exc:
                raise RuntimeError(
                    "A cached online segment no longer exposes i_membrane_. "
                    "CVode.use_fast_imem(1) must remain enabled and cells must "
                    "stay alive for the entire episode."
                ) from exc
        return currents

    def _collect_window_spikes(self, *, start_ms: float, stop_ms: float):
        """Gather spikes occurring in ``(start_ms, stop_ms]`` onto MPI rank 0."""
        local_by_population: dict[str, list[tuple[int, list[float]]]] = {}
        tolerance = max(1e-9, float(self.dt) * 1e-6)

        for population_name in self.population_names:
            population = self.populations[population_name]
            local_entries: list[tuple[int, list[float]]] = []
            spike_vectors = getattr(population, "_hoc_spike_vectors", [])
            gids = getattr(population, "gids", [])

            if len(spike_vectors) != len(gids):
                raise RuntimeError(
                    f"Population {population_name}: spike-vector/GID count "
                    f"mismatch ({len(spike_vectors)} vs {len(gids)})."
                )

            for index, vector in enumerate(spike_vectors):
                all_times = hoc_vector_to_numpy(vector)
                selected = all_times[
                    (all_times > start_ms + tolerance)
                    & (all_times <= stop_ms + tolerance)
                ]
                local_entries.append((int(gids[index]), selected.tolist()))
            local_by_population[population_name] = local_entries

        gathered = self._online_comm.gather(local_by_population, root=0)
        if self._online_rank != 0:
            return None

        merged: dict[str, dict[str, Any]] = {}
        for population_name in self.population_names:
            event_gids: list[int] = []
            event_times: list[float] = []
            per_cell: dict[int, np.ndarray] = {}

            for rank_data in gathered:
                for gid, gid_times in rank_data[population_name]:
                    values = np.asarray(gid_times, dtype=np.float64)
                    per_cell[int(gid)] = values
                    event_gids.extend([int(gid)] * int(values.size))
                    event_times.extend(values.tolist())

            if event_times:
                # Stable sorting preserves GID order for simultaneous spikes,
                # matching LFPy's population/cell gathering convention.
                order = np.argsort(
                    np.asarray(event_times, dtype=np.float64),
                    kind="stable",
                )
                times_array = np.asarray(event_times, dtype=np.float64)[order]
                gids_array = np.asarray(event_gids, dtype=np.int64)[order]
            else:
                times_array = np.empty(0, dtype=np.float64)
                gids_array = np.empty(0, dtype=np.int64)

            merged[population_name] = {
                "times_ms": times_array,
                "gids": gids_array,
                "per_cell": per_cell,
            }
        return merged

    def _population_firing_rates(self, *, spikes, duration_ms: float):
        if self._online_rank != 0 or spikes is None:
            return None
        duration_s = float(duration_ms) / 1000.0
        if duration_s <= 0:
            raise ValueError("duration_ms must be positive.")

        rates: dict[str, float] = {}
        for population_name in self.population_names:
            # per_cell contains every GID, including silent cells, and therefore
            # gives a robust global population size after MPI gathering.
            population_size = len(spikes[population_name]["per_cell"])
            spike_count = int(spikes[population_name]["times_ms"].size)
            rates[f"{population_name}_spike_count"] = float(spike_count)
            rates[f"{population_name}_firing_rate_hz"] = (
                spike_count / (population_size * duration_s)
                if population_size > 0
                else np.nan
            )

        if "E_firing_rate_hz" in rates and "I_firing_rate_hz" in rates:
            i_rate = rates["I_firing_rate_hz"]
            rates["E_I_firing_rate_ratio"] = (
                rates["E_firing_rate_hz"] / i_rate if i_rate > 0 else np.nan
            )
        return rates

    def advance_online(
        self,
        *,
        stop_ms: float,
        before_advance=None,
    ) -> dict[str, Any] | None:
        """Advance the existing state and sample currents at each right boundary.

        Samples use the half-open convention ``(start_ms, stop_ms]``.  The
        input callback, when supplied, sets the extracellular value at the
        current left boundary immediately before each fixed integration step.
        """
        if not self._online_initialized:
            raise RuntimeError("Call initialize_online() before advance_online().")

        start_ms = self.current_time_ms
        stop_ms = float(stop_ms)
        if stop_ms <= start_ms:
            raise ValueError(
                f"stop_ms ({stop_ms}) must exceed current time ({start_ms})."
            )

        duration_ms = stop_ms - start_ms
        expected_steps = int(round(duration_ms / float(self.dt)))
        if not np.isclose(
            expected_steps * float(self.dt),
            duration_ms,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "Online window duration must be an integer multiple of dt: "
                f"duration={duration_ms}, dt={self.dt}."
            )

        local_probe_data = [
            np.empty((transform.shape[0], expected_steps), dtype=np.float64)
            for transform in self._online_transforms
        ]
        sample_times = np.empty(expected_steps, dtype=np.float64)
        representative_stride = int(self._online_representative_stride_steps)
        representative_samples = expected_steps // representative_stride
        if representative_stride > 1 and (
            expected_steps % representative_stride != 0
        ):
            raise RuntimeError(
                "Online window is not divisible by the representative-state "
                "sampling interval."
            )
        representative_state = {
            f"{site['population']}_gid_{site['gid']}": {
                key: np.empty(representative_samples, dtype=np.float64)
                for key in self._sample_representative_site(site)
            }
            for site in self._online_representative_sites
        }
        size = int(self._online_comm.Get_size())
        dt_ms = float(self.dt)
        tolerance = max(1e-8, dt_ms * 1e-5)
        diagnostics_before = self.online_diagnostics()

        for step_index in range(expected_steps):
            left_boundary_ms = start_ms + step_index * dt_ms
            target_ms = start_ms + (step_index + 1) * dt_ms
            if before_advance is not None:
                before_advance(left_boundary_ms)

            if size == 1:
                neuron.h.fadvance()
            else:
                # psolve is required for inter-rank NetCon delivery.  Sampling
                # after each fixed-dt absolute target keeps the same convention
                # as the single-rank fadvance loop.
                self.pc.psolve(target_ms)

            reached_step_ms = self.current_time_ms
            if abs(reached_step_ms - target_ms) > tolerance:
                raise RuntimeError(
                    "NEURON fixed-step continuation drifted from its target: "
                    f"step={step_index}, reached={reached_step_ms}, "
                    f"expected={target_ms}."
                )

            sample_times[step_index] = reached_step_ms
            local_imem = self._read_local_membrane_currents()
            for probe_index, transform in enumerate(self._online_transforms):
                local_probe_data[probe_index][:, step_index] = (
                    transform @ local_imem
                )
            if (step_index + 1) % representative_stride == 0:
                representative_index = (step_index + 1) // representative_stride - 1
                for site in self._online_representative_sites:
                    site_id = f"{site['population']}_gid_{site['gid']}"
                    for name, value in self._sample_representative_site(site).items():
                        representative_state[site_id][name][
                            representative_index
                        ] = value

        self._online_comm.Barrier()
        reached_time = self.current_time_ms
        if abs(reached_time - stop_ms) > tolerance:
            raise RuntimeError(
                f"NEURON stopped at {reached_time} ms; expected {stop_ms} ms."
            )

        global_probe_data: list[np.ndarray] | None = (
            [] if self._online_rank == 0 else None
        )
        for local_data in local_probe_data:
            if size == 1:
                reduced = local_data
            else:
                reduced = (
                    np.empty_like(local_data)
                    if self._online_rank == 0
                    else None
                )
                self._online_comm.Reduce(
                    local_data,
                    reduced,
                    op=MPI.SUM,
                    root=0,
                )
            if self._online_rank == 0:
                global_probe_data.append(reduced)

        spikes = self._collect_window_spikes(
            start_ms=start_ms,
            stop_ms=stop_ms,
        )
        firing_rates = self._population_firing_rates(
            spikes=spikes,
            duration_ms=duration_ms,
        )
        gathered_representative = self._online_comm.gather(
            representative_state, root=0
        )
        diagnostics_after = self.online_diagnostics()

        if self._online_rank != 0:
            return None

        representative_global: dict[str, dict[str, np.ndarray]] = {}
        for rank_values in gathered_representative:
            for site_id, values in rank_values.items():
                if site_id in representative_global:
                    raise RuntimeError(
                        f"Representative site {site_id} was recorded on multiple ranks."
                    )
                representative_global[site_id] = values

        return {
            "t_start_ms": start_ms,
            "t_stop_ms": reached_time,
            "time_ms": sample_times,
            "sample_times_ms": sample_times,
            "sample_boundary_convention": "(t_start_ms, t_stop_ms]",
            "probe_data": global_probe_data,
            "spikes": spikes,
            "firing_rates": firing_rates,
            "representative_state": representative_global,
            "representative_state_interval_ms": float(
                representative_stride * self.dt
            ),
            "sample_count": int(expected_steps),
            "expected_sample_count": int(expected_steps),
            "diagnostics": {
                "before": diagnostics_before,
                "after": diagnostics_after,
            },
        }

    def close_online(self) -> None:
        """Release references owned only by the online runner."""
        self._online_initialized = False
        self._online_probes = []
        self._online_transforms = []
        self._online_segments = []
        self._online_representative_sites = []
        self._online_representative_stride_steps = 1
        self._online_cvode = None
        self._online_comm = None
        self._online_rank = None
