"""Focused design tests for the H4-BW2 cadence discovery."""

import unittest
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir

from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_cadence_discovery import (
    FAST,
    NEW,
    _augment_common_audit,
    _controller_modes,
    _profile,
    _select_controller,
)


def _config():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                "env=ballnstick",
                "analysis=ballnstick_phase_refresh_cadence_discovery",
            ],
        )


class PhaseRefreshCadenceDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _config()

    def test_new_controller_changes_cadence_but_not_history(self):
        self.assertEqual(
            _controller_modes(self.cfg),
            [
                "sham",
                "one_time",
                "refresh_1000ms_250ms",
                "refresh_1000ms_125ms",
                "refresh_500ms_125ms",
            ],
        )
        profile = _profile(self.cfg, NEW)
        self.assertTrue(profile["adaptive"])
        self.assertEqual(profile["history_ms"], 1000.0)
        self.assertEqual(profile["update_interval_ms"], 125.0)
        self.assertEqual(float(self.cfg.analysis.tacs.correction_horizon_ms), 250.0)

    def test_common_audit_uses_common_250ms_boundaries(self):
        updates = []
        for index, boundary in enumerate((0.0, 125.0, 250.0, 375.0, 500.0)):
            updates.append({
                "update_index": index,
                "boundary_ms": boundary,
                "common_audit_phase_error_before_correction_rad": 0.2,
                "common_audit_resultant_to_rms": 1.0,
            })
        rows = [{"controller_mode": NEW}]
        episodes = {NEW: {"simulation": {"phase_updates": updates}}}
        _augment_common_audit(rows, episodes, self.cfg)
        # The onset is excluded; 250 and 500 ms are the two common boundaries.
        self.assertEqual(rows[0]["common_phase_audit_count"], 2)
        self.assertAlmostEqual(rows[0]["mean_abs_common_phase_error_rad"], 0.2)
        self.assertEqual(rows[0]["common_phase_estimate_actionable_fraction"], 1.0)

    def test_selection_prefers_one_second_estimator_within_tie(self):
        common = {
            "selection_candidate": True,
            "passes_cadence_gate": True,
            "mean_realized_candidate_win_fraction": 0.9,
        }
        summary = pd.DataFrame([
            {
                "controller_mode": NEW,
                "mean_advantage_over_one_time_log10": 0.050,
                **common,
            },
            {
                "controller_mode": FAST,
                "mean_advantage_over_one_time_log10": 0.058,
                **common,
            },
        ])
        selected = _select_controller(summary, self.cfg)
        self.assertTrue(selected["cadence_candidate_found"])
        self.assertEqual(selected["selected_controller"], NEW)


if __name__ == "__main__":
    unittest.main()
