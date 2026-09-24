"""Tests for bounded-memory online recording and trace persistence."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import h5py
from mpi4py import MPI
from neuron import h
import numpy as np

from env.models.neuron.networkenv_online import OnlineNetworkEnv
from env.models.neuron.streaming import OnlineTraceWriter


class OnlineRecordingOptimizationTests(unittest.TestCase):
    def test_automatic_soma_voltage_recorders_are_replaced(self):
        section = h.Section(name="online_recording_test_soma")
        cell = SimpleNamespace(somav=h.Vector())
        cell.somav.record(section(0.5)._ref_v)

        network = OnlineNetworkEnv.__new__(OnlineNetworkEnv)
        network.population_names = ["E"]
        network.populations = {"E": SimpleNamespace(cells=[cell])}

        self.assertEqual(network._disable_unused_soma_voltage_recorders(), 1)
        h.dt = 0.1
        h.finitialize(-65.0)
        h.frecord_init()
        for _ in range(10):
            h.fadvance()

        self.assertEqual(int(cell.somav.size()), 0)
        h.delete_section(sec=section)

    def test_spike_vectors_are_drained_between_windows(self):
        vector = h.Vector([0.0, 1.0, 2.0, 3.0])
        network = OnlineNetworkEnv.__new__(OnlineNetworkEnv)
        network.dt = 0.1
        network.population_names = ["E"]
        network.populations = {
            "E": SimpleNamespace(_hoc_spike_vectors=[vector], gids=[17])
        }
        network._online_comm = MPI.COMM_SELF
        network._online_rank = 0

        first = network._collect_window_spikes(start_ms=0.0, stop_ms=2.0)
        np.testing.assert_array_equal(first["E"]["times_ms"], [1.0, 2.0])
        np.testing.assert_array_equal(first["E"]["gids"], [17, 17])
        self.assertEqual(int(vector.size()), 0)

        vector.append(2.5)
        vector.append(3.5)
        second = network._collect_window_spikes(start_ms=2.0, stop_ms=4.0)
        np.testing.assert_array_equal(second["E"]["times_ms"], [2.5, 3.5])
        self.assertEqual(int(vector.size()), 0)

    def test_trace_writer_appends_and_commits_complete_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.h5"
            writer = OnlineTraceWriter(
                path,
                stage_names=["baseline", "stimulation"],
            )
            writer.append_window(
                sample_time_ms=[0.1, 0.2],
                eeg_v=[[1.0, 2.0]],
                dipole_nA_um=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                field_left_boundary_time_ms=[0.0, 0.1],
                field_left_boundary_v_per_m=[0.0, 0.0],
                stage_code=0,
            )
            writer.append_window(
                sample_time_ms=[0.3, 0.4],
                eeg_v=[[3.0, 4.0]],
                dipole_nA_um=[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                field_left_boundary_time_ms=[0.2, 0.3],
                field_left_boundary_v_per_m=[0.5, -0.5],
                stage_code=1,
            )
            self.assertEqual(writer.committed_samples, 4)
            self.assertEqual(writer.committed_windows, 2)
            writer.close()
            writer.close()

            with h5py.File(path, "r") as trace:
                self.assertEqual(int(trace.attrs["committed_samples"]), 4)
                self.assertEqual(int(trace.attrs["committed_windows"]), 2)
                np.testing.assert_array_equal(
                    trace["sample_time_ms"][:], [0.1, 0.2, 0.3, 0.4]
                )
                np.testing.assert_array_equal(trace["stage_code"][:], [0, 0, 1, 1])
                self.assertEqual(trace["eeg_v"].shape, (1, 4))
                self.assertEqual(trace["dipole_nA_um"].shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
