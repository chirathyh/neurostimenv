"""Compare same-seed no-stimulation online and legacy BallAndStick EEG."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import hydra
import numpy as np
from decouple import config
from mpi4py import MPI
from omegaconf import DictConfig, open_dict


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from env.models.neuron.env import NeuronEnv  # noqa: E402
from env.models.neuron.env_online import OnlineNeuronEnv  # noqa: E402


def _mpi_variables(comm, size: int, rank: int, seed: int):
    return {
        "COMM": comm,
        "SIZE": size,
        "RANK": rank,
        "GLOBALSEED": seed,
        "SEED": seed * 10_000,
    }


def _seed(seed: int, rank: int) -> None:
    np.random.seed(seed * 10_000 + rank)
    random.seed(seed * 10_000 + rank)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    seed = int(cfg.experiment.seed)
    duration_ms = float(cfg.env.simulation.duration)
    window_ms = float(cfg.env.simulation.obs_win_len)
    n_windows = int(round(duration_ms / window_ms))
    if not np.isclose(n_windows * window_ms, duration_ms):
        raise ValueError("duration must be an integer multiple of obs_win_len.")

    with open_dict(cfg):
        cfg.env.ts.apply = False
        cfg.env.online.temperature_mode = "configured"

    _seed(seed, rank)
    legacy = NeuronEnv(
        cfg,
        _mpi_variables(comm, size, rank, seed),
        ENV_SEED=0,
    )
    try:
        legacy_eeg = legacy.step_n([], [], stim_elec=0)
    finally:
        legacy.close()

    _seed(seed, rank)
    online = OnlineNeuronEnv(
        cfg,
        _mpi_variables(comm, size, rank, seed),
        ENV_SEED=0,
    )
    try:
        outputs = online.analysis_rollout_online(
            [[0.0, 0.0] for _ in range(n_windows)]
        )
    finally:
        online.close()

    if rank != 0:
        return
    legacy_values = np.asarray(legacy_eeg, dtype=np.float64).reshape(-1)
    online_values = np.concatenate(
        [
            np.asarray(output["eeg_v"], dtype=np.float64).reshape(-1)
            for output in outputs
        ]
    )
    if legacy_values.size == online_values.size + 1:
        legacy_aligned = legacy_values[1:]
    elif legacy_values.size == online_values.size:
        legacy_aligned = legacy_values
    else:
        raise RuntimeError(
            f"Unexpected sample counts: legacy={legacy_values.size}, "
            f"online={online_values.size}."
        )

    difference = online_values - legacy_aligned
    reference_rms = float(np.sqrt(np.mean(np.square(legacy_aligned))))
    result = {
        "legacy_samples": int(legacy_values.size),
        "online_samples": int(online_values.size),
        "correlation": float(np.corrcoef(legacy_aligned, online_values)[0, 1]),
        "rms_error_v": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_rms_error": (
            float(np.sqrt(np.mean(np.square(difference))) / reference_rms)
            if reference_rms > 0
            else 0.0
        ),
        "max_abs_error_v": float(np.max(np.abs(difference))),
        "configured_celsius": float(cfg.env.network.celsius),
        "online_effective_celsius": float(
            outputs[0]["diagnostics"]["before"]["effective_h_celsius"]
        ),
        "sample_boundary_convention": "(start, stop] online; legacy t=0 dropped",
    }
    output_dir = Path(cfg.experiment.dir) / "online_legacy_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("\n### Online/legacy same-seed no-stimulation comparison")
    print(json.dumps(result, indent=2))
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
