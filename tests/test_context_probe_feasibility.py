"""Focused tests for the CL1-P probe-response feasibility analysis."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_context_probe_feasibility import (
    ESCALATE,
    MAINTAIN,
    _context_shuffle_null,
    _counterfactual_summary,
    _probe_rule_arm,
)


class ContextProbeFeasibilityTests(unittest.TestCase):
    def test_frozen_probe_rule_stops_escalation_at_or_below_target(self):
        self.assertEqual(
            _probe_rule_arm(-0.01, threshold=0.0), MAINTAIN
        )
        self.assertEqual(
            _probe_rule_arm(0.0, threshold=0.0), MAINTAIN
        )
        self.assertEqual(
            _probe_rule_arm(0.01, threshold=0.0), ESCALATE
        )

    def test_counterfactual_summary_uses_only_predecision_probe_for_rule(self):
        common = {
            "context_order": 1,
            "trial_seed": 1,
            "structure_seed": 11,
            "drive_seed": 21,
            "phase_seed": 31,
            "structure_index": 1,
            "drive_index": 1,
            "context_baseline_matched_log10_alpha": -20.4,
            "context_probe_log10_alpha": -20.8,
            "context_probe_signed_error_to_B_log10": -0.05,
            "context_probe_alpha_suppression_log10": 0.4,
            "context_probe_gain_log10_per_vpm": 2.0,
            "context_baseline_10hz_resultant_v": 2.0e-10,
            "context_probe_10hz_resultant_v": 1.0e-10,
            "context_probe_10hz_resultant_reduction_v": 1.0e-10,
            "pre_action_distance_to_B_log10": 0.3,
            "paired_predecision_relative_rms_error": 0.0,
        }
        metrics = pd.DataFrame([
            {
                **common,
                "context_id": "c1",
                "arm": MAINTAIN,
                "decision_dose_v_per_m": 0.2,
                "post_distance_to_B_log10": 0.02,
            },
            {
                **common,
                "context_id": "c1",
                "arm": ESCALATE,
                "decision_dose_v_per_m": 0.4,
                "post_distance_to_B_log10": 0.20,
            },
        ])
        cfg = OmegaConf.create({
            "analysis": {
                "criteria": {"practical_advantage_log10": 0.01},
                "actions": {"probe_stop_escalation_threshold_log10": 0.0},
            }
        })

        summary = _counterfactual_summary(metrics, cfg=cfg)

        self.assertEqual(summary.loc[0, "probe_rule_arm"], MAINTAIN)
        self.assertEqual(summary.loc[0, "oracle_arm"], MAINTAIN)
        self.assertAlmostEqual(
            summary.loc[0, "probe_rule_advantage_over_escalate_log10"],
            0.18,
        )

    def test_context_shuffle_null_is_reproducible(self):
        summary = pd.DataFrame({
            "context_probe_signed_error_to_B_log10": [-0.1, 0.1, 0.2],
            "maintain_distance_to_B_log10": [0.01, 0.4, 0.3],
            "escalate_distance_to_B_log10": [0.2, 0.1, 0.1],
            "probe_rule_advantage_over_escalate_log10": [0.19, 0.0, 0.0],
        })
        cfg = OmegaConf.create({
            "experiment": {"seed": 7},
            "analysis": {
                "actions": {"probe_stop_escalation_threshold_log10": 0.0},
                "context_shuffle": {"n_permutations": 100},
            },
        })

        first, first_summary = _context_shuffle_null(summary, cfg=cfg)
        second, second_summary = _context_shuffle_null(summary, cfg=cfg)

        np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
        self.assertEqual(first_summary, second_summary)


if __name__ == "__main__":
    unittest.main()
