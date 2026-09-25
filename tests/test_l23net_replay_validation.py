"""Tests for deterministic L23Net replay artifact comparison."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from env.models.neuron.streaming import OnlineTraceWriter
from experiments.l23net_analysis.replay_validation import (
    REPORT_NAME,
    TRACE_NAME,
    canonical_json_sha256,
    compare_replay_artifacts,
    trace_content_summary,
)


class L23NetReplayValidationTests(unittest.TestCase):
    def _write_run(self, directory: Path, *, eeg_offset: float = 0.0) -> None:
        writer = OnlineTraceWriter(directory / TRACE_NAME, stage_names=["no_field"])
        writer.append_window(
            sample_time_ms=[0.1, 0.2],
            eeg_v=[[1.0 + eeg_offset, 2.0]],
            dipole_nA_um=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            field_left_boundary_time_ms=[0.0, 0.1],
            field_left_boundary_v_per_m=[0.0, 0.0],
            stage_code=0,
        )
        writer.close()
        report = {
            "status": "passed",
            "replay_contract_sha256": canonical_json_sha256({"seed": 10}),
            "seed_manifest": {"seed": 10, "ranks": [100000, 100001]},
            "structure": {"global_sha256": "structure", "by_rank": []},
            "windows": [{"spikes": {"counts": {"E": 1}, "sha256": "spikes"}}],
        }
        (directory / REPORT_NAME).write_text(
            json.dumps(report) + "\n", encoding="utf-8"
        )

    def test_trace_summary_is_content_deterministic(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = directory / "first"
            second = directory / "second"
            first.mkdir()
            second.mkdir()
            self._write_run(first)
            self._write_run(second)
            self.assertEqual(
                trace_content_summary(first / TRACE_NAME),
                trace_content_summary(second / TRACE_NAME),
            )

    def test_identical_replays_pass(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = directory / "first"
            second = directory / "second"
            first.mkdir()
            second.mkdir()
            self._write_run(first)
            self._write_run(second)
            comparison = compare_replay_artifacts(first, second)
            self.assertEqual(comparison["status"], "passed")
            self.assertEqual(comparison["errors"], [])
            self.assertTrue(
                all(
                    row["exact"]
                    for row in comparison["dataset_comparisons"].values()
                )
            )

    def test_changed_trace_fails_with_difference(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = directory / "first"
            second = directory / "second"
            first.mkdir()
            second.mkdir()
            self._write_run(first)
            self._write_run(second, eeg_offset=0.25)
            comparison = compare_replay_artifacts(first, second)
            self.assertEqual(comparison["status"], "failed")
            eeg = comparison["dataset_comparisons"]["eeg_v"]
            self.assertFalse(eeg["exact"])
            self.assertEqual(eeg["maximum_absolute_difference"], 0.25)


if __name__ == "__main__":
    unittest.main()
