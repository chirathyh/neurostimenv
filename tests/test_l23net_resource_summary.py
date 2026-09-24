"""Unit tests for the NCI L23Net resource-summary parser."""

from pathlib import Path
import tempfile
import unittest

from experiments.l23net_analysis.summarize_l23net_resources import (
    parse_pbs_duration_seconds,
    parse_pbs_memory_gib,
    parse_qstat,
)
from experiments.l23net_analysis.profile_l23net_tacs_full_scale import (
    _aggregate_rss_trend,
    _compact_console_report,
    _stage_performance,
)


class L23NetResourceSummaryTests(unittest.TestCase):
    def test_duration_supports_cpu_hours_above_one_day(self):
        self.assertEqual(parse_pbs_duration_seconds("00:02:34"), 154.0)
        self.assertEqual(parse_pbs_duration_seconds("253:04:17"), 911057.0)

    def test_memory_units_are_binary(self):
        self.assertAlmostEqual(
            parse_pbs_memory_gib("36382016kb"),
            36382016.0 / 1024.0 / 1024.0,
        )
        self.assertEqual(parse_pbs_memory_gib("2GB"), 2.0)

    def test_qstat_extracts_resource_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qstat_at_exit.txt"
            path.write_text(
                """resources_used.cpupercent = 3068
resources_used.cput = 00:44:29
resources_used.mem = 36382016kb
resources_used.walltime = 00:02:34
Resource_List.ncpus = 624
""",
                encoding="utf-8",
            )
            values = parse_qstat(path)
        self.assertEqual(values["resources_used.cpupercent"], "3068")
        self.assertEqual(values["resources_used.mem"], "36382016kb")
        self.assertEqual(values["Resource_List.ncpus"], "624")

    def test_stage_performance_reports_stimulation_overhead(self):
        windows = [
            {"stage": "burn_in", "wall_s": 1.0, "duration_ms": 250.0},
            {"stage": "baseline", "wall_s": 1.0, "duration_ms": 250.0},
            {"stage": "stimulation", "wall_s": 2.0, "duration_ms": 250.0},
            {"stage": "washout", "wall_s": 1.0, "duration_ms": 250.0},
        ]
        summary = _stage_performance(windows)
        self.assertEqual(summary["by_stage"]["stimulation"]["window_count"], 1)
        self.assertEqual(
            summary["stimulation_to_inactive_median_window_wall_ratio"], 2.0
        )

    def test_rss_trend_excludes_build_and_first_window(self):
        snapshots = [
            {"simulated_ms": 0.0, "rss_gib": {"sum": 10.0}},
            {"simulated_ms": 250.0, "rss_gib": {"sum": 11.0}},
            {"simulated_ms": 500.0, "rss_gib": {"sum": 11.1}},
            {"simulated_ms": 750.0, "rss_gib": {"sum": 11.2}},
        ]
        trend = _aggregate_rss_trend(snapshots)
        self.assertEqual(trend["point_count"], 2)
        self.assertAlmostEqual(trend["slope_gib_per_simulated_s"], 0.4)
        self.assertAlmostEqual(trend["all_checkpoint_peak_gib"], 11.2)

    def test_console_report_collapses_per_rank_probe_names(self):
        report = {
            "status": "passed",
            "errors": [],
            "completed_simulated_ms": 500.0,
            "mpi": {"size": 2},
            "build": {
                "online_probe_names_by_rank": [
                    ["current_dipole_moment"],
                    ["current_dipole_moment"],
                ]
            },
            "windows": [{}, {}],
            "performance": {},
            "artifacts": {},
        }
        compact = _compact_console_report(report)
        self.assertEqual(
            compact["build"]["unique_online_probe_sets"],
            [["current_dipole_moment"]],
        )
        self.assertNotIn("online_probe_names_by_rank", compact["build"])


if __name__ == "__main__":
    unittest.main()
