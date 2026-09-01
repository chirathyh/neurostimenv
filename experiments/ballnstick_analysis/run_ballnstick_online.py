"""Run and validate a causal online BallAndStick episode."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from decouple import config
from mpi4py import MPI
from omegaconf import DictConfig

MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env_online import OnlineNeuronEnv
from utils.utils import setup_folders


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        cfg = setup_folders(cfg)
        print("\n### Online experiment configuration")
        print(cfg)
    comm.Barrier()

    mpi_variables = {
        "COMM": comm,
        "SIZE": size,
        "RANK": rank,
        "GLOBALSEED": int(cfg.experiment.seed),
        "SEED": int(cfg.experiment.seed) * 10_000,
    }

    env: OnlineNeuronEnv | None = None
    try:
        env = OnlineNeuronEnv(cfg, mpi_variables, ENV_SEED=0)
        if rank == 0:
            print(
                "Online environment initialised at "
                f"t={env.network.current_time_ms:.6f} ms"
            )

            print(f"Online diagnostics: {env.network.online_diagnostics()}")

        duration_ms = float(cfg.env.simulation.duration)
        window_ms = float(cfg.env.simulation.obs_win_len)
        n_windows = int(round(duration_ms / window_ms))
        if not np.isclose(n_windows * window_ms, duration_ms):
            raise ValueError(
                "simulation.duration must be an integer multiple of "
                "simulation.obs_win_len for this runner."
            )

        def select_action(step_index, previous_observation):
            """Example policy called only after the prior observation exists."""
            del previous_observation  # replace with a learned policy input
            if not bool(cfg.env.ts.apply):
                return [0.0, 0.0]
            parameterization = str(
                cfg.env.online.stimulation.parameterization
            )
            amplitude = (
                0.8 if parameterization == "uniform_field" else 0.002
            )
            return {
                0: [0.0, 0.0],
                1: [0.0, 0.0],
                2: [amplitude, 10.0],
                3: [amplitude, 20.0],
                4: [amplitude, 40.0],
                5: [0.0, 0.0],
            }.get(step_index, [0.0, 0.0])

        outputs = [] if rank == 0 else None
        previous_result = None
        for step_index in range(n_windows):
            # Select only the current action.  A learned policy can use
            # previous_result["observation"] here; no future stimulation
            # waveform is prepared by the environment.
            previous_observation = (
                None
                if previous_result is None
                else previous_result["observation"]
            )
            action = select_action(step_index, previous_observation)
            result = env.step_online(
                action=action,
                duration_ms=window_ms,
                phase_continuous=True,
                ramp_ms=float(
                    cfg.env.get("online", {}).get("ramp_ms", 0.0)
                ),
            )
            if rank == 0:
                outputs.append(result)
                previous_result = result
                if result["done"]:
                    break

        if rank == 0:
            output_dir = Path(cfg.experiment.dir) / "online"
            output_dir.mkdir(parents=True, exist_ok=True)

            if len(outputs) != n_windows:
                raise RuntimeError(
                    f"Expected {n_windows} windows, completed {len(outputs)}."
                )

            previous_stop_ms = None
            previous_last_sample_ms = None
            for step_index, result in enumerate(outputs):
                if result is None:
                    raise RuntimeError(f"Missing result for step {step_index}.")
                eeg = np.asarray(result["eeg_v"], dtype=np.float64)
                sample_times = np.asarray(
                    result["sample_times_ms"],
                    dtype=np.float64,
                )
                if eeg.size == 0 or not np.all(np.isfinite(eeg)):
                    raise RuntimeError(
                        f"Invalid EEG returned at step {step_index}: shape={eeg.shape}."
                    )
                if eeg.shape[-1] != sample_times.size:
                    raise RuntimeError(
                        "EEG/time sample count mismatch at step "
                        f"{step_index}: {eeg.shape[-1]} vs {sample_times.size}."
                    )
                if previous_stop_ms is not None:
                    if not np.isclose(result["t_start_ms"], previous_stop_ms):
                        raise RuntimeError("Online time did not advance contiguously.")
                    if sample_times[0] <= previous_last_sample_ms:
                        raise RuntimeError(
                            "Consecutive online windows duplicate a boundary sample."
                        )

                for population_name in ("E", "I"):
                    population_spikes = result["spikes"][population_name]
                    event_count = len(population_spikes["times_ms"])
                    per_cell_count = sum(
                        len(values)
                        for values in population_spikes["per_cell"].values()
                    )
                    if event_count != per_cell_count:
                        raise RuntimeError(
                            f"{population_name} spike accounting mismatch: "
                            f"events={event_count}, per_cell={per_cell_count}."
                        )

                np.save(output_dir / f"eeg_step_{step_index:02d}.npy", eeg)
                np.save(
                    output_dir / f"time_step_{step_index:02d}.npy",
                    sample_times,
                )
                np.savez(
                    output_dir / f"spikes_step_{step_index:02d}.npz",
                    E_times_ms=result["spikes"].get("E", {}).get("times_ms", []),
                    E_gids=result["spikes"].get("E", {}).get("gids", []),
                    I_times_ms=result["spikes"].get("I", {}).get("times_ms", []),
                    I_gids=result["spikes"].get("I", {}).get("gids", []),
                )
                stimulation = result["stimulation"]
                stimulation_arrays = {
                    "enabled": stimulation["enabled"],
                    "parameterization": stimulation["parameterization"],
                    "time_ms": stimulation["time_ms"],
                    "frequency_hz": stimulation["frequency_hz"],
                    "phase_start_rad": stimulation["phase_start_rad"],
                    "phase_stop_rad": stimulation["phase_stop_rad"],
                    "phase_continuous": stimulation["phase_continuous"],
                }
                if stimulation["parameterization"] == "uniform_field":
                    stimulation_arrays.update(
                        field_v_per_m=stimulation["field_v_per_m"],
                        amplitude_v_per_m=stimulation["amplitude_v_per_m"],
                    )
                else:
                    stimulation_arrays.update(
                        current_nA=stimulation["current_nA"],
                        amplitude_mA=stimulation["amplitude_mA"],
                    )
                np.savez(
                    output_dir / f"stimulation_step_{step_index:02d}.npz",
                    **stimulation_arrays,
                )
                print(
                    f"step={step_index}, action={result['action'].tolist()}, "
                    f"t={result['t_start_ms']:.3f}-{result['t_stop_ms']:.3f} ms, "
                    f"samples={result['sample_count']}/"
                    f"{result['expected_sample_count']}, "
                    f"eeg_shape={eeg.shape}, rates={result['firing_rates']}"
                )
                previous_stop_ms = float(result["t_stop_ms"])
                previous_last_sample_ms = float(sample_times[-1])

            final_residual_mV = env.stimulation_controller.max_abs_extracellular(
                env.network
            )
            print(
                f"completed_windows={len(outputs)}, "
                f"final_t_ms={env.network.current_time_ms:.6f}, "
                f"final_max_abs_e_extracellular_mV={final_residual_mV:.6g}"
            )
            if outputs[-1]["action"][0] == 0.0 and final_residual_mV != 0.0:
                raise RuntimeError(
                    "The final zero action left residual extracellular voltage."
                )
    finally:
        if env is not None:
            env.close()
            env = None
        comm.Barrier()


if __name__ == "__main__":
    main()
