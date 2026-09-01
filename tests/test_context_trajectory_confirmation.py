"""Focused tests for the CL1-C frozen EEG-trajectory confirmation."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_context_probe_feasibility import (
    ESCALATE,
    MAINTAIN,
)
from experiments.ballnstick_analysis.run_ballnstick_context_trajectory_confirmation import (
    _baseline_severity_rule_arm,
    _confirmation_summary,
    _context_shuffle_null,
    _exact_one_sided_sign_flip,
    _trajectory_rule_arm,
)


class ContextTrajectoryConfirmationTests(unittest.TestCase):
    def test_frozen_trajectory_rule_uses_strict_positive_change(self):
        self.assertEqual(_trajectory_rule_arm(0.01), MAINTAIN)
        self.assertEqual(_trajectory_rule_arm(0.0), ESCALATE)
        self.assertEqual(_trajectory_rule_arm(-0.01), ESCALATE)

    def test_baseline_control_uses_frozen_a_mean(self):
        self.assertEqual(
            _baseline_severity_rule_arm(
                -20.6, frozen_a_mean_log10_alpha=-20.5
            ),
            MAINTAIN,
        )
        self.assertEqual(
            _baseline_severity_rule_arm(
                -20.4, frozen_a_mean_log10_alpha=-20.5
            ),
            ESCALATE,
        )

    def test_summary_selects_action_only_from_predecision_trajectory(self):
        common = {
            "context_id": "s1_d1",
            "context_order": 1,
            "trial_seed": 1,
            "structure_seed": 11,
            "drive_seed": 21,
            "phase_seed": 31,
            "structure_index": 1,
            "drive_index": 1,
            "screen_log10_alpha_power": -20.6,
            "screen_alpha_excess_to_B_log10": 0.2,
            "context_baseline_matched_log10_alpha": -20.4,
            "context_probe_log10_alpha": -20.7,
            "context_probe_signed_error_to_B_log10": 0.1,
            "context_probe_alpha_suppression_log10": 0.3,
            "context_probe_gain_log10_per_vpm": 1.5,
            "context_sham_trajectory_log10": -0.1,
            "context_causal_probe_suppression_log10": 0.4,
            "pre_action_distance_to_B_log10": 0.3,
            "paired_predecision_relative_rms_error": 0.0,
        }
        metrics = pd.DataFrame([
            {
                **common,
                "arm": MAINTAIN,
                "post_distance_to_B_log10": 0.02,
            },
            {
                **common,
                "arm": ESCALATE,
                "post_distance_to_B_log10": 0.20,
            },
        ])
        cfg = OmegaConf.create({
            "analysis": {
                "trajectory_rule": {"threshold_log10": 0.0},
                "criteria": {"practical_advantage_log10": 0.01},
            }
        })
        target = {
            "A_mean_log10_alpha": -20.5,
            "B_mean_log10_alpha": -20.8,
        }

        summary = _confirmation_summary(metrics, target_model=target, cfg=cfg)

        self.assertEqual(summary.loc[0, "trajectory_rule_arm"], MAINTAIN)
        self.assertEqual(summary.loc[0, "baseline_rule_arm"], MAINTAIN)
        self.assertEqual(summary.loc[0, "sham_trajectory_rule_arm"], ESCALATE)
        self.assertAlmostEqual(
            summary.loc[0, "trajectory_advantage_over_fixed_0p4_log10"],
            0.18,
        )

    def test_structure_sign_flip_is_exact(self):
        self.assertAlmostEqual(
            _exact_one_sided_sign_flip(np.ones(6)),
            1.0 / 64.0,
        )
        self.assertEqual(_exact_one_sided_sign_flip(np.array([])), 1.0)

    def test_context_shuffle_is_reproducible(self):
        summary = pd.DataFrame({
            "context_probe_alpha_suppression_log10": [0.2, -0.1, 0.3, -0.2],
            "maintain_distance_to_B_log10": [0.01, 0.4, 0.02, 0.3],
            "escalate_distance_to_B_log10": [0.2, 0.1, 0.2, 0.1],
            "trajectory_rule_distance_to_B_log10": [0.01, 0.1, 0.02, 0.1],
        })
        cfg = OmegaConf.create({
            "experiment": {"seed": 7},
            "analysis": {
                "trajectory_rule": {"threshold_log10": 0.0},
                "context_shuffle": {"n_permutations": 100},
            },
        })

        first, first_summary = _context_shuffle_null(
            summary, best_fixed_arm=MAINTAIN, cfg=cfg
        )
        second, second_summary = _context_shuffle_null(
            summary, best_fixed_arm=MAINTAIN, cfg=cfg
        )

        np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
        self.assertEqual(first_summary, second_summary)


if __name__ == "__main__":
    unittest.main()
