"""Focused design and inference tests for H4-C."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from experiments.ballnstick_analysis.run_ballnstick_h4_confirmation import (
    EXPECTED_MODES,
    ONE_TIME,
    SELECTED,
    SHAM,
    _common_initialization,
    _controller_modes,
    _effect_tables,
    _load_frozen_h4bw2,
    _power_design,
    _profile,
    _validate_design,
)


def _config():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return compose(
            config_name="config",
            overrides=["env=ballnstick", "analysis=ballnstick_h4_confirmation"],
        )


class H4ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _config()
        cls.sources = _load_frozen_h4bw2(cls.cfg)

    def test_hash_locked_design_and_power_are_valid(self):
        power = _power_design(self.cfg)
        _validate_design(self.cfg, self.sources, power)
        self.assertEqual(_controller_modes(self.cfg), EXPECTED_MODES)
        self.assertEqual(power["planned_independent_structures"], 12)
        self.assertEqual(power["required_independent_structures"], 12)
        self.assertGreaterEqual(power["a_priori_t_approximation_power"], 0.8)

    def test_selected_controller_exactly_matches_frozen_profile(self):
        self.assertEqual(
            _profile(self.cfg, SELECTED),
            {"adaptive": True, "history_ms": 500.0, "update_interval_ms": 125.0},
        )
        candidate = self.sources["candidate"]
        self.assertEqual(candidate["selected_controller"], SELECTED)
        self.assertEqual(candidate["selected_profile"], _profile(self.cfg, SELECTED))

    def test_effect_table_uses_expected_futures_and_correct_direction(self):
        expected_rows = []
        metric_rows = []
        distances = {SHAM: 0.30, ONE_TIME: 0.35, SELECTED: 0.20}
        phase_errors = {SHAM: 1.4, ONE_TIME: 1.8, SELECTED: 0.6}
        for structure in (1, 2):
            for frequency in (9.0, 11.0):
                for label, diffusion in (("low_diffusion", 0.5), ("high_diffusion", 2.0)):
                    context = f"s{structure}_f{frequency}_{label}"
                    for mode in EXPECTED_MODES:
                        expected_rows.append({
                            "context_id": context,
                            "structure_seed": structure,
                            "hidden_frequency_hz": frequency,
                            "label": label,
                            "diffusion_rad2_per_s": diffusion,
                            "context_C1": 0.5,
                            "controller_mode": mode,
                            "expected_post_distance_to_B_log10": distances[mode],
                            "mean_abs_common_phase_error_rad": phase_errors[mode],
                        })
                        for future_index, jitter in enumerate((-0.01, 0.0, 0.01)):
                            metric_rows.append({
                                "context_id": context,
                                "future_index": future_index,
                                "controller_mode": mode,
                                "post_distance_to_B_log10": distances[mode] + jitter,
                            })
        context, structure, diffusion = _effect_tables(
            pd.DataFrame(expected_rows), pd.DataFrame(metric_rows)
        )
        self.assertTrue(np.allclose(
            context.primary_refresh_advantage_over_one_time_log10, 0.15
        ))
        self.assertTrue(np.allclose(
            context.secondary_refresh_advantage_over_sham_log10, 0.10
        ))
        self.assertTrue(np.allclose(
            structure.mean_phase_error_reduction_vs_one_time_rad, 1.2
        ))
        self.assertTrue(np.allclose(
            diffusion.mean_future_win_fraction, 1.0
        ))

    def test_common_initialization_requires_same_phase_and_one_second_history(self):
        rows = []
        for mode in (ONE_TIME, SELECTED):
            rows.append({
                "context_id": "c1", "future_index": 0,
                "controller_mode": mode, "update_index": 0,
                "desired_field_phase_rad": 0.7, "phase_history_ms": 1000.0,
            })
        self.assertTrue(_common_initialization(pd.DataFrame(rows)))
        rows[-1]["desired_field_phase_rad"] = 0.8
        self.assertFalse(_common_initialization(pd.DataFrame(rows)))


if __name__ == "__main__":
    unittest.main()
