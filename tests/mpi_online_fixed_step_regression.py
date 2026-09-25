"""MPI regression for the NEURON 8.2.3 long-horizon psolve boundary bug.

Run with::

    mpiexec -n 2 python tests/mpi_online_fixed_step_regression.py

The former absolute-grid psolve target reproducibly skipped one complete
0.025-ms step at 11636.95 ms.  This probe crosses that boundary and verifies
the exact one-step invariant used by ``OnlineNetworkEnv`` through 15 seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mpi4py import MPI
import numpy as np
from neuron import h

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from env.models.neuron.networkenv_online import (  # noqa: E402
    advance_one_fixed_step,
    canonical_fixed_step_boundary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-ms", type=float, default=15000.0)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        raise SystemExit("Run this regression with at least two MPI processes.")

    duration_ms, expected_steps = canonical_fixed_step_boundary(
        args.duration_ms,
        args.dt_ms,
        name="duration_ms",
    )
    section = h.Section(name=f"mpi_fixed_step_probe_rank_{comm.rank}")
    section.L = 20.0
    section.diam = 20.0
    section.insert("hh")
    pc = h.ParallelContext()
    h.dt = float(args.dt_ms)
    pc.set_maxstep(10.0)
    h.finitialize(-65.0)

    for _ in range(expected_steps):
        advance_one_fixed_step(
            pc=pc,
            dt_ms=args.dt_ms,
            parallel=True,
        )

    raw_times = comm.gather(float(h.t), root=0)
    summary = None
    if comm.rank == 0:
        canonical_times = [
            canonical_fixed_step_boundary(
                value,
                args.dt_ms,
                name=f"rank {rank} final raw NEURON time",
            )[0]
            for rank, value in enumerate(raw_times)
        ]
        if not np.allclose(canonical_times, duration_ms, rtol=0.0, atol=0.0):
            raise RuntimeError(
                f"MPI ranks did not reach {duration_ms} ms: {raw_times}"
            )
        summary = {
            "status": "passed",
            "mpi_ranks": int(comm.size),
            "dt_ms": float(args.dt_ms),
            "expected_steps": int(expected_steps),
            "canonical_duration_ms": float(duration_ms),
            "raw_final_time_min_ms": float(min(raw_times)),
            "raw_final_time_max_ms": float(max(raw_times)),
            "maximum_raw_clock_drift_ms": float(
                max(abs(value - duration_ms) for value in raw_times)
            ),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    summary = comm.bcast(summary, root=0)
    if summary["status"] != "passed":
        raise SystemExit(1)
    comm.Barrier()
    h.delete_section(sec=section)


if __name__ == "__main__":
    main()
