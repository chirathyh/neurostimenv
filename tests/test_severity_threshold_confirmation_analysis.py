"""Tests for CDM2-C frozen severity-threshold confirmation statistics."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_severity_threshold_confirmation import (
    _exact_sign_flip_p_value,
    _frozen_policy_table,
    _one_sample_t_power,
    _required_n_one_sample_t,
    _structure_summary,
)


class SeverityThresholdConfirmationTests(unittest.TestCase):
    def test_power_design_requires_twelve_structures_for_planned_effect(self):
        effect = 0.01 / 0.013
        required = _required_n_one_sample_t(
            effect_size=effect,
            alpha=0.05,
            target_power=0.80,
        )
        self.assertEqual(required, 12)
        self.assertGreaterEqual(
            _one_sample_t_power(n=12, effect_size=effect, alpha=0.05),
            0.80,
        )
        self.assertLess(
            _one_sample_t_power(n=11, effect_size=effect, alpha=0.05),
            0.80,
        )

    def test_frozen_rule_selects_exactly_one_constant_dose_per_context(self):
        cfg = OmegaConf.create(
            {
                "analysis": {
                    "frozen_discovery": {
                        "expected_low_dose_v_per_m": 0.2,
                        "expected_high_dose_v_per_m": 0.4,
                    },
                    "inference": {
                        "primary_comparator_dose_v_per_m": 0.4,
                        "secondary_comparator_dose_v_per_m": 0.2,
                    },
                }
            }
        )
        table = pd.DataFrame(
            {
                "context_id": ["low", "equal", "high"],
                "structure_seed": [1, 1, 2],
                "history_seed": [3, 4, 3],
                "state_label": ["moderate", "strong", "strong"],
                "context_alpha_excess_log10": [0.2, 0.38, 0.5],
                "low_expected_distance_to_B_log10": [0.03, 0.08, 0.20],
                "high_expected_distance_to_B_log10": [0.09, 0.04, 0.02],
                "expected_optimal_binary_dose_v_per_m": [0.2, 0.4, 0.4],
            }
        )
        policy = _frozen_policy_table(table, threshold=0.38, cfg=cfg)
        self.assertEqual(policy.selected_dose_v_per_m.tolist(), [0.2, 0.4, 0.4])
        np.testing.assert_allclose(
            policy.primary_advantage_over_fixed_0p4_log10,
            [0.06, 0.0, 0.0],
        )
        self.assertTrue(policy.matches_expected_binary_oracle.all())

    def test_structure_not_context_is_inferential_unit(self):
        policy = pd.DataFrame(
            {
                "context_id": ["a", "b", "c", "d"],
                "structure_seed": [10, 10, 20, 20],
                "selected_expected_distance_to_B_log10": [0.1, 0.2, 0.3, 0.4],
                "fixed_0p4_expected_distance_to_B_log10": [0.2, 0.3, 0.4, 0.5],
                "fixed_0p2_expected_distance_to_B_log10": [0.1, 0.3, 0.5, 0.7],
                "primary_advantage_over_fixed_0p4_log10": [0.1, 0.1, 0.1, 0.1],
                "secondary_advantage_over_fixed_0p2_log10": [0.0, 0.1, 0.2, 0.3],
                "selected_dose_v_per_m": [0.2, 0.4, 0.2, 0.4],
                "matches_expected_binary_oracle": [True, True, False, True],
            }
        )
        structure = _structure_summary(policy)
        self.assertEqual(len(structure), 2)
        self.assertEqual(structure.eligible_context_count.tolist(), [2, 2])
        np.testing.assert_allclose(
            structure.primary_advantage_over_fixed_0p4_log10,
            [0.1, 0.1],
        )

    def test_exact_sign_flip_test_is_one_sided_and_reproducible(self):
        values = np.array([0.01, 0.02, 0.03, 0.04])
        first = _exact_sign_flip_p_value(
            values,
            maximum_exact_n=20,
            monte_carlo_samples=100,
            rng=np.random.default_rng(7),
        )
        second = _exact_sign_flip_p_value(
            values,
            maximum_exact_n=20,
            monte_carlo_samples=100,
            rng=np.random.default_rng(99),
        )
        self.assertEqual(first, second)
        self.assertEqual(first[1], "exact")
        self.assertEqual(first[2], 16)
        self.assertAlmostEqual(first[0], 1.0 / 16.0)


if __name__ == "__main__":
    unittest.main()
