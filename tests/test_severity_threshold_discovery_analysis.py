"""Tests for CDM2-D monotone EEG-severity threshold discovery."""

import unittest

import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_severity_threshold_discovery import (
    _apply_threshold,
    _crossvalidated_threshold_policy,
    _fit_monotone_threshold,
)


def _cfg():
    return OmegaConf.create({
        "analysis": {
            "threshold_discovery": {
                "low_dose_v_per_m": 0.2,
                "high_dose_v_per_m": 0.4,
                "minimum_training_contexts_per_action": 2,
            }
        }
    })


def _table():
    rows = []
    for structure in (1, 2, 3):
        for index, excess in enumerate((0.10, 0.20, 0.40, 0.50), start=1):
            low_is_best = excess < 0.30
            rows.append({
                "context_id": f"s{structure}_c{index}",
                "structure_seed": structure,
                "history_seed": 10 + index % 2,
                "state_label": "lower" if low_is_best else "higher",
                "context_alpha_excess_log10": excess,
                "low_expected_distance_to_B_log10": 0.02 if low_is_best else 0.20,
                "high_expected_distance_to_B_log10": 0.20 if low_is_best else 0.02,
                "expected_optimal_binary_dose_v_per_m": 0.2 if low_is_best else 0.4,
            })
    return pd.DataFrame(rows)


class SeverityThresholdDiscoveryTests(unittest.TestCase):
    def test_threshold_is_fit_between_low_and_high_severity_contexts(self):
        model = _fit_monotone_threshold(_table(), cfg=_cfg())
        self.assertIsNotNone(model)
        self.assertAlmostEqual(model["threshold_log10_alpha_excess"], 0.30)
        actions, distances = _apply_threshold(
            _table(), threshold=model["threshold_log10_alpha_excess"], cfg=_cfg()
        )
        self.assertEqual(set(actions), {0.2, 0.4})
        self.assertTrue((distances == 0.02).all())

    def test_leave_one_structure_out_rule_generalizes(self):
        policy, folds = _crossvalidated_threshold_policy(_table(), cfg=_cfg())
        self.assertEqual(len(folds), 3)
        self.assertTrue(folds.threshold_available.all())
        self.assertEqual(len(policy), 12)
        self.assertTrue(policy.matches_expected_binary_oracle.all())
        self.assertGreater(policy.advantage_over_fixed_0p4_log10.mean(), 0.0)
        self.assertEqual(set(policy.selected_dose_v_per_m), {0.2, 0.4})

    def test_threshold_is_unavailable_without_both_action_support(self):
        table = _table().iloc[:3].copy()
        self.assertIsNone(_fit_monotone_threshold(table, cfg=_cfg()))


if __name__ == "__main__":
    unittest.main()
