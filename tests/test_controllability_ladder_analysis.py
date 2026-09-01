"""Unit tests for the causal-controllability ladder primitives."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_controllability_ladder import (
    FAMILY_INTERPOLATION,
    FAMILY_I_DRIVE,
    _episode_config,
    _interpolation_seed_metrics,
    _reachability_metrics,
    _select_i_drive,
)


class ControllabilityLadderAnalysisTests(unittest.TestCase):
    @staticmethod
    def _config():
        return OmegaConf.create(
            {
                "experiment": {"seed": 1, "dir": "unused"},
                "analysis": {
                    "timeline": {"burn_in_steps": 2, "analysis_steps": 3},
                },
                "env": {
                    "simulation": {"obs_win_len": 1000.0, "duration": 1.0},
                    "network": {
                        "inhibition_scale": 1.0,
                        "background": {"I": {"weight": 0.001}},
                    },
                    "ts": {"apply": False},
                    "online": {
                        "temperature_mode": "configured",
                        "stimulation": {"parameterization": "uniform_field"},
                    },
                },
            }
        )

    def test_episode_config_scales_only_copied_i_background_weight(self):
        base = self._config()
        run = _episode_config(
            base,
            inhibition_scale=0.5,
            i_background_weight_multiplier=1.2,
            seed=17,
            output_dir=Path("episode"),
        )

        self.assertAlmostEqual(run.env.network.background.I.weight, 0.0012)
        self.assertAlmostEqual(base.env.network.background.I.weight, 0.001)
        self.assertEqual(run.env.simulation.duration, 5000.0)
        self.assertEqual(run.env.network.inhibition_scale, 0.5)

    def test_reachability_halfway_point_has_half_closure(self):
        target = {"x": 1.0, "y": 1.0}
        sham = {"x": 0.0, "y": 0.0}
        candidate = {"x": 0.5, "y": 0.5}

        result = _reachability_metrics(
            target=target,
            sham=sham,
            candidate=candidate,
            feature_names=["x", "y"],
            center=np.zeros(2),
            scale=np.ones(2),
        )

        self.assertAlmostEqual(result["fractional_distance_improvement"], 0.5)
        self.assertAlmostEqual(result["target_shift_alignment"], 1.0)
        self.assertAlmostEqual(result["max_abs_target_error_z"], 0.5)

    def test_interpolation_reports_perfect_rank_monotonicity(self):
        rows = pd.DataFrame(
            {
                "cohort": ["validation"] * 4,
                "seed": [70001] * 4,
                "family": [FAMILY_INTERPOLATION] * 4,
                "inhibition_scale": [0.5, 0.7, 0.9, 1.0],
                "candidate_distance_to_A": [2.0, 1.2, 0.4, 0.0],
            }
        )

        result = _interpolation_seed_metrics(rows).iloc[0]

        self.assertAlmostEqual(result["scale_distance_spearman"], -1.0)
        self.assertTrue(result["negative_spearman"])
        self.assertTrue(result["strictly_nonincreasing_distance"])

    def test_i_drive_selection_excludes_sham_and_prefers_safe_effect(self):
        summary = pd.DataFrame(
            {
                "family": [FAMILY_I_DRIVE] * 3,
                "level_id": ["Ibg_1", "Ibg_1p1", "Ibg_1p2"],
                "i_background_weight_multiplier": [1.0, 1.1, 1.2],
                "all_rate_safe": [True, True, False],
                "mean_fractional_improvement": [0.0, 0.2, 0.9],
                "median_alignment": [0.0, 0.8, 1.0],
            }
        )

        selected = _select_i_drive(summary, top_k=1)

        self.assertEqual(selected[0]["level_id"], "Ibg_1p1")


if __name__ == "__main__":
    unittest.main()
