"""True step-wise closed-loop NeuroStimEnv environment.

This module is additive and does not replace
``env.models.neuron.env.NeuronEnv``.  It is intentionally limited to the small
``ballnstick`` model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import gym
import neuron
import numpy as np
from lfpykit.eegmegcalc import FourSphereVolumeConductor

from env.eeg import features
from env.models.neuron.extracellular import ExtracellularModels
from env.models.neuron.extracellular_online import OnlineExtracellularController
from env.models.neuron.networkenv_online import OnlineNetworkEnv
from env.models.neuron.stimulation import (
    apply_raised_cosine_block_envelope,
    make_sinusoidal_electric_field,
    make_sinusoidal_stimulation,
)
from setup.circuits.ballnstick.utils import setup_network_ballnstick


class OnlineNeuronEnv(gym.Env):
    """A decision-window-level closed-loop environment backed by one NEURON run."""

    def __init__(self, args, MPI_VAR, ENV_SEED: int = 0):
        self.args = args
        self.MPI_VAR = dict(MPI_VAR)
        self.MPI_VAR["GLOBALSEED"] += int(ENV_SEED)
        self.MPI_VAR["SEED"] += int(ENV_SEED)
        self.rank = int(self.MPI_VAR["RANK"])
        self.comm = self.MPI_VAR["COMM"]
        self.sampling_rate = (1.0 / float(args.env.network.dt)) * 1000.0
        self.phase_rad = 0.0
        self.step_count = 0
        self._closed = False
        self._build_complete = False

        self.network = None
        self.four_sphere_top = None
        self.extracellular = None
        self.extracellular_models = None
        self.stimulation_controller = None

        self.reset_online()

    def _build(self) -> None:
        if self.args.env.name != "ballnstick":
            raise ValueError(
                "OnlineNeuronEnv is intentionally limited to env=ballnstick; "
                f"received env={self.args.env.name!r}."
            )
        if not bool(self.args.env.eeg.measure):
            raise ValueError("OnlineNeuronEnv currently requires env.eeg.measure=true.")

        self.network = OnlineNetworkEnv(**self.args.env.networkParameters)
        setup_network_ballnstick(self.network, self.args, self.MPI_VAR)

        self.four_sphere_top = FourSphereVolumeConductor(
            np.asarray(self.args.env.eeg.locations, dtype=np.float64),
            self.args.env.eeg.foursphereheadmodel["radii"],
            self.args.env.eeg.foursphereheadmodel["sigmas"],
        )
        self.extracellular = ExtracellularModels(self.args)
        self.extracellular_models = self.extracellular.get_probes()

        online_cfg = self.args.env.get("online", {})
        stimulation_cfg = online_cfg.get("stimulation", {})
        self.stimulation_parameterization = str(
            stimulation_cfg.get(
                "parameterization",
                "point_source_current",
            )
        )
        if self.stimulation_parameterization not in {
            "point_source_current",
            "uniform_field",
        }:
            raise ValueError(
                "online.stimulation.parameterization must be "
                "'point_source_current' or 'uniform_field'."
            )
        self.stimulation_controller = OnlineExtracellularController(
            electrode_index=0,
            points_per_electrode=int(online_cfg.get("points_per_electrode", 5)),
            field_model=str(online_cfg.get("field_model", "inf")),
            field_direction=tuple(
                stimulation_cfg.get("field_direction", [0.0, 0.0, 1.0])
            ),
        )
        default_direction = self.stimulation_controller.field_direction
        raw_montages = stimulation_cfg.get("montages", {})
        self.field_montages: dict[str, np.ndarray] = {
            "default": np.asarray(default_direction, dtype=np.float64)
        }
        for montage_name, montage_direction in raw_montages.items():
            self.field_montages[str(montage_name)] = (
                self.stimulation_controller.normalize_field_direction(
                    montage_direction
                )
            )
        self.default_montage = str(
            stimulation_cfg.get("default_montage", "default")
        )
        if self.default_montage not in self.field_montages:
            raise ValueError(
                "online.stimulation.default_montage must name an entry in "
                "online.stimulation.montages."
            )
        self.stimulation_controller.prepare_network(self.network)
        self.network.initialize_online(
            probes=self.extracellular_models,
            comm=self.comm,
            max_step_ms=float(online_cfg.get("max_step_ms", 10.0)),
            temperature_mode=str(
                online_cfg.get("temperature_mode", "configured")
            ),
        )

    def _parse_action(
        self,
        action: Sequence[float] | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalize legacy or scientific actions into an MPI-safe mapping.

        The legacy two-value uniform-field action remains
        ``[AC amplitude V/m, frequency Hz]``. Mapping actions additionally
        support signed DC, an explicit window-start phase, and either a named
        montage or a direct field direction.
        """
        if self.stimulation_parameterization == "point_source_current":
            if isinstance(action, Mapping):
                amplitude = float(action["amplitude_mA"])
                frequency_hz = float(action["frequency_hz"])
                phase_rad = action.get("phase_rad")
            else:
                if len(action) != 2:
                    raise ValueError("action must be [amplitude, frequency_hz].")
                amplitude = float(action[0])
                frequency_hz = float(action[1])
                phase_rad = None
            if amplitude < 0:
                raise ValueError("Stimulation amplitude must be non-negative.")
            if frequency_hz < 0:
                raise ValueError("Stimulation frequency must be non-negative.")
            return {
                "amplitude_mA": amplitude,
                "frequency_hz": frequency_hz,
                "phase_rad": None if phase_rad is None else float(phase_rad),
            }

        if isinstance(action, Mapping):
            amplitude = float(
                action.get(
                    "ac_amplitude_v_per_m",
                    action.get("amplitude_v_per_m", 0.0),
                )
            )
            dc_offset = float(
                action.get(
                    "dc_offset_v_per_m",
                    action.get("dc_field_v_per_m", 0.0),
                )
            )
            frequency_hz = float(action.get("frequency_hz", 0.0))
            phase_rad = action.get("phase_rad")
            montage_value = action.get(
                "montage",
                action.get("montage_id", self.default_montage),
            )
            direct_direction = action.get("field_direction")
            if direct_direction is not None and (
                "montage" in action or "montage_id" in action
            ):
                raise ValueError(
                    "Specify either field_direction or montage, not both."
                )
            if direct_direction is None:
                montage_name = str(montage_value)
                if montage_name not in self.field_montages:
                    available = ", ".join(sorted(self.field_montages))
                    raise ValueError(
                        f"Unknown field montage {montage_name!r}; available: "
                        f"{available}."
                    )
                direction = self.field_montages[montage_name]
            else:
                montage_name = "custom"
                direction = self.stimulation_controller.normalize_field_direction(
                    direct_direction
                )
        else:
            if len(action) != 2:
                raise ValueError("action must be [amplitude, frequency_hz].")
            amplitude = float(action[0])
            dc_offset = 0.0
            frequency_hz = float(action[1])
            phase_rad = None
            montage_name = self.default_montage
            direction = self.field_montages[montage_name]

        numeric_values = [amplitude, dc_offset, frequency_hz]
        if phase_rad is not None:
            numeric_values.append(float(phase_rad))
        if not np.all(np.isfinite(numeric_values)):
            raise ValueError("Stimulation action values must be finite.")
        if amplitude < 0:
            raise ValueError("AC field amplitude must be non-negative.")
        if frequency_hz < 0:
            raise ValueError("Stimulation frequency must be non-negative.")
        if amplitude > 0.0 and frequency_hz == 0.0:
            raise ValueError(
                "A non-zero AC amplitude requires frequency_hz > 0; use "
                "dc_offset_v_per_m for a DC field."
            )
        return {
            "ac_amplitude_v_per_m": amplitude,
            "dc_offset_v_per_m": dc_offset,
            "frequency_hz": frequency_hz,
            "phase_rad": None if phase_rad is None else float(phase_rad),
            "montage": montage_name,
            "field_direction": [float(value) for value in direction],
        }

    def step_online(
        self,
        action: Sequence[float] | Mapping[str, Any] | None,
        *,
        duration_ms: float | None = None,
        phase_continuous: bool = True,
        ramp_ms: float = 0.0,
        block_envelope: Mapping[str, float] | None = None,
        transition_from_ac_amplitude_v_per_m: float | None = None,
        amplitude_transition_ms: float = 0.0,
    ) -> dict[str, Any] | None:
        """Apply one action, advance one window, then return EEG and spikes."""
        if self._closed:
            raise RuntimeError("Cannot step a closed OnlineNeuronEnv.")
        if self.network is None:
            raise RuntimeError("OnlineNeuronEnv was not built successfully.")

        if self.rank == 0:
            action_to_send = self._parse_action(
                [0.0, 0.0] if action is None else action
            )
        else:
            action_to_send = None
        action_spec = self.comm.bcast(action_to_send, root=0)
        frequency_hz = float(action_spec["frequency_hz"])
        amplitude = float(
            action_spec.get(
                "ac_amplitude_v_per_m",
                action_spec.get("amplitude_mA", 0.0),
            )
        )

        start_ms = self.network.current_time_ms
        duration_ms = (
            float(self.args.env.simulation.obs_win_len)
            if duration_ms is None
            else float(duration_ms)
        )
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive.")
        episode_stop_ms = float(self.args.env.simulation.duration)
        if start_ms >= episode_stop_ms - 1e-9:
            raise RuntimeError(
                f"Episode already ended at t={start_ms} ms "
                f"(duration={episode_stop_ms} ms)."
            )
        if start_ms + duration_ms > episode_stop_ms + 1e-9:
            raise ValueError(
                "The next observation window exceeds simulation.duration: "
                f"start={start_ms}, window={duration_ms}, "
                f"duration={episode_stop_ms}."
            )

        if block_envelope is not None and float(ramp_ms) > 0.0:
            raise ValueError(
                "Use either per-window ramp_ms or block_envelope, not both."
            )
        if (
            transition_from_ac_amplitude_v_per_m is not None
            and self.stimulation_parameterization != "uniform_field"
        ):
            raise ValueError(
                "Amplitude transitions are supported only for uniform fields."
            )

        requested_phase_rad = action_spec.get("phase_rad")
        phase_start_rad = (
            float(requested_phase_rad)
            if requested_phase_rad is not None
            else (self.phase_rad if phase_continuous else 0.0)
        )
        if self.stimulation_parameterization == "uniform_field":
            dc_offset_v_per_m = float(action_spec["dc_offset_v_per_m"])
            field_direction = action_spec["field_direction"]
            waveform = make_sinusoidal_electric_field(
                amplitude_v_per_m=amplitude,
                frequency_hz=frequency_hz,
                start_ms=start_ms,
                duration_ms=duration_ms,
                dt_ms=float(self.network.dt),
                phase_rad=phase_start_rad,
                dc_offset_v_per_m=dc_offset_v_per_m,
                ramp_ms=float(ramp_ms),
                initial_amplitude_v_per_m=(
                    None
                    if transition_from_ac_amplitude_v_per_m is None
                    else float(transition_from_ac_amplitude_v_per_m)
                ),
                amplitude_transition_ms=float(amplitude_transition_ms),
                include_endpoint=True,
            )
        else:
            waveform = make_sinusoidal_stimulation(
                amplitude_mA=amplitude,
                frequency_hz=frequency_hz,
                start_ms=start_ms,
                duration_ms=duration_ms,
                dt_ms=float(self.network.dt),
                phase_rad=phase_start_rad,
                ramp_ms=float(ramp_ms),
                include_endpoint=True,
            )

        block_envelope_metadata = None
        if block_envelope is not None:
            block_start_ms = float(block_envelope["start_ms"])
            block_stop_ms = float(block_envelope["stop_ms"])
            block_ramp_ms = float(block_envelope.get("ramp_ms", 0.0))
            if self.stimulation_parameterization == "uniform_field":
                waveform = replace(
                    waveform,
                    field_v_per_m=apply_raised_cosine_block_envelope(
                        waveform.field_v_per_m,
                        time_ms=waveform.time_ms,
                        block_start_ms=block_start_ms,
                        block_stop_ms=block_stop_ms,
                        ramp_ms=block_ramp_ms,
                    ),
                )
            else:
                waveform = replace(
                    waveform,
                    current_nA=apply_raised_cosine_block_envelope(
                        waveform.current_nA,
                        time_ms=waveform.time_ms,
                        block_start_ms=block_start_ms,
                        block_stop_ms=block_stop_ms,
                        ramp_ms=block_ramp_ms,
                    ),
                )
            block_envelope_metadata = {
                "start_ms": block_start_ms,
                "stop_ms": block_stop_ms,
                "ramp_ms": block_ramp_ms,
            }

        if bool(self.args.env.ts.apply):
            if self.stimulation_parameterization == "uniform_field":
                applied = self.stimulation_controller.apply_uniform_field(
                    network=self.network,
                    field_v_per_m=waveform.field_v_per_m,
                    time_ms=waveform.time_ms,
                    field_direction=field_direction,
                )
            else:
                applied = self.stimulation_controller.apply_waveform(
                    network=self.network,
                    electrode=self.extracellular_models[0],
                    current_nA=waveform.current_nA,
                    time_ms=waveform.time_ms,
                )
        else:
            self.stimulation_controller.clear(self.network)
            applied = None

        result = self.network.advance_online(
            stop_ms=start_ms + duration_ms,
            before_advance=(
                self.stimulation_controller.set_time
                if bool(self.args.env.ts.apply)
                else None
            ),
        )
        self.phase_rad = waveform.final_phase_rad if phase_continuous else 0.0
        self.step_count += 1

        if self.rank != 0:
            return None
        if result is None:
            raise RuntimeError("Rank 0 did not receive an online simulation result.")

        # Probe 1 is CurrentDipoleMoment according to
        # ExtracellularModels.get_probes().  Its units are nA*um, and the
        # four-sphere helper returns mV, matching the legacy path.
        dipole_moment = result["probe_data"][1]
        eeg_mV = self.four_sphere_top.get_dipole_potential(
            dipole_moment,
            np.asarray(self.args.env.network.position, dtype=np.float64),
        )
        eeg_v = np.asarray(eeg_mV, dtype=np.float64) * 1e-3

        displayed_amplitude = (
            abs(float(action_spec["dc_offset_v_per_m"])) + amplitude
            if self.stimulation_parameterization == "uniform_field"
            else amplitude
        )
        action_array = np.asarray(
            [displayed_amplitude, frequency_hz],
            dtype=np.float64,
        )
        feature_dict = features.feature_space(
            eeg=eeg_v,
            fs=self.sampling_rate,
            ts=action_array,
        )
        observation_values = []
        for value in feature_dict.values():
            array = np.asarray(value)
            if array.size != 1:
                raise ValueError(
                    "Online feature_space must return scalar features; "
                    f"received shape {array.shape}."
                )
            observation_values.append(float(array.reshape(-1)[0]))
        observation = np.asarray(observation_values, dtype=np.float64)

        result.update(
            {
                "eeg_v": eeg_v,
                "observation": observation,
                "observation_features": feature_dict,
                "action": action_array,
                "action_spec": action_spec,
                "stimulation": {
                    "enabled": bool(self.args.env.ts.apply),
                    "parameterization": self.stimulation_parameterization,
                    "applied": applied,
                    "time_ms": waveform.time_ms,
                    "frequency_hz": frequency_hz,
                    "phase_start_rad": float(phase_start_rad),
                    "phase_requested_rad": (
                        None
                        if requested_phase_rad is None
                        else float(requested_phase_rad)
                    ),
                    "phase_stop_rad": float(waveform.final_phase_rad),
                    "phase_continuous": bool(phase_continuous),
                    "block_envelope": block_envelope_metadata,
                    "amplitude_transition": (
                        None
                        if transition_from_ac_amplitude_v_per_m is None
                        else {
                            "from_v_per_m": float(
                                transition_from_ac_amplitude_v_per_m
                            ),
                            "to_v_per_m": amplitude,
                            "duration_ms": float(amplitude_transition_ms),
                        }
                    ),
                    **(
                        {
                            "field_v_per_m": waveform.field_v_per_m,
                            "amplitude_v_per_m": amplitude,
                            "ac_amplitude_v_per_m": amplitude,
                            "dc_offset_v_per_m": dc_offset_v_per_m,
                            "field_direction": np.asarray(
                                field_direction, dtype=np.float64
                            ),
                            "montage": action_spec["montage"],
                        }
                        if self.stimulation_parameterization == "uniform_field"
                        else {
                            "current_nA": waveform.current_nA,
                            "amplitude_mA": amplitude,
                        }
                    ),
                },
                "done": self.network.current_time_ms
                >= episode_stop_ms - 1e-9,
            }
        )
        return result

    def analysis_rollout_online(
        self,
        policy_seq,
        *,
        phase_continuous: bool = True,
        ramp_ms: float = 0.0,
        block_envelope: Mapping[str, float] | None = None,
    ):
        """Execute a supplied sequence one genuine online step at a time."""
        outputs = [] if self.rank == 0 else None
        for action in policy_seq:
            result = self.step_online(
                action,
                phase_continuous=phase_continuous,
                ramp_ms=ramp_ms,
                block_envelope=block_envelope,
            )
            if self.rank == 0:
                outputs.append(result)
                if result["done"]:
                    break
        return outputs

    def _release_network_resources(self) -> None:
        """Best-effort release used by reset, close, and failed construction."""
        network = self.network
        controller = self.stimulation_controller

        if controller is not None and network is not None:
            try:
                controller.clear(network)
            except Exception:
                pass

        if network is not None:
            try:
                network.close_online()
            except Exception:
                pass
            try:
                network.pc.gid_clear()
            except Exception:
                pass

            populations = getattr(network, "populations", None)
            if populations is not None:
                for population in list(populations.values()):
                    cells = getattr(population, "cells", None)
                    if cells is None:
                        continue
                    for cell in list(cells):
                        if cell is None:
                            continue
                        try:
                            cell.__del__()
                        except Exception:
                            pass
                    try:
                        population.cells = None
                    except Exception:
                        pass

        self.network = None
        self.stimulation_controller = None
        self.extracellular_models = None
        self.extracellular = None
        self.four_sphere_top = None

        try:
            neuron.h("forall delete_section()")
        except Exception:
            pass

    def reset_online(self):
        """Build and initialise a fresh episode, returning initial metadata.

        This is the only supported way to restart.  It destroys any prior
        episode and performs exactly one new ``finitialize``; normal
        ``step_online`` calls never reinitialise.
        """
        self._release_network_resources()
        self._closed = False
        self._build_complete = False
        self.phase_rad = 0.0
        self.step_count = 0

        # LFPy uses NumPy's process-local RNG for positions, connectivity,
        # weights and delays.  Reset it at episode construction so independent
        # same-seed online runs are reproducible.
        np.random.seed(int(self.MPI_VAR["SEED"]) + int(self.rank))

        try:
            self._build()
            self._build_complete = True
        except Exception:
            self._release_network_resources()
            self._closed = True
            raise

        if self.rank != 0:
            return None
        return {
            "t_ms": self.network.current_time_ms,
            "step_count": self.step_count,
            "diagnostics": self.network.online_diagnostics(),
        }

    def close(self) -> None:
        """Idempotently release NEURON/LFPy resources."""
        if self._closed:
            return
        self._closed = True
        self._release_network_resources()
