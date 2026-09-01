"""Focused tests for CDM1-S conditional expected-action analysis."""

import unittest

import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_single_action_dose_map import (
    _expected_action_map,
    _exploratory_loso_policy,
)


def _cfg():
    return OmegaConf.create({
        "analysis": {
            "actions": {
                "sham_dose_v_per_m": 0.0,
                "active_doses_v_per_m": [0.1, 0.2, 0.3, 0.4],
            },
            "context": {
                "policy_features": [
                    "context_alpha_excess_log10",
                    "context_coherent_alpha_fraction",
                    "context_alpha_peak_prominence_db",
                    "context_alpha_temporal_sd_log10",
                ],
                "ridge_penalty": 1.0e-6,
            },
            "criteria": {"practical_advantage_log10": 0.01},
        }
    })


def _metrics():
    rows = []
    order = 0
    for structure in (11, 12):
        for state_index, (label, depth, excess, optimum) in enumerate((
            ("mild", 0.02, 0.1, 0.1),
            ("strong", 0.06, 0.5, 0.4),
        ), start=1):
            order += 1
            context_id = f"{label}_s{structure}"
            for future_index in (1, 2):
                for dose in (0.0, 0.1, 0.2, 0.3, 0.4):
                    if dose == 0.0:
                        distance = 0.30
                    elif dose == optimum:
                        distance = 0.01 + 0.002 * (future_index - 1)
                    else:
                        distance = 0.20 + 0.002 * (future_index - 1)
                    rows.append({
                        "context_id": context_id,
                        "context_order": order,
                        "state_label": label,
                        "state_index": state_index,
                        "modulation_depth": depth,
                        "structure_seed": structure,
                        "structure_index": structure - 10,
                        "history_seed": 21,
                        "history_index": 1,
                        "phase_seed": 31,
                        "future_index": future_index,
                        "dose_v_per_m": dose,
                        "context_alpha_excess_log10": excess,
                        "context_coherent_alpha_fraction": excess,
                        "context_alpha_peak_prominence_db": 10.0 * excess,
                        "context_alpha_temporal_sd_log10": 0.01,
                        "context_log10_alpha_power": -21.0 + excess,
                        "context_10hz_resultant_v": excess * 1.0e-10,
                        "context_alpha_temporal_slope_log10_per_window": 0.0,
                        "context_alpha_first_last_change_log10": 0.0,
                        "context_window_count": 3,
                        "screen_phase_split_error_deg": 5.0,
                        "screen_10hz_resultant_to_rms": 0.2,
                        "pre_action_distance_to_B_log10": excess,
                        "post_distance_to_B_log10": distance,
                        "causal_target_distance_improvement_vs_sham_log10": (
                            0.30 - distance
                        ),
                        "causal_alpha_suppression_vs_sham_log10": 0.30 - distance,
                        "coherent_10hz_suppression_vs_sham_v": dose * 1.0e-10,
                        "alpha_peak_prominence_reduction_vs_sham_db": dose,
                        "rate_safe": True,
                        "field_removal_recovered": True,
                    })
    return pd.DataFrame(rows)


class SingleActionDoseMapTests(unittest.TestCase):
    def test_expected_oracle_averages_independent_futures(self):
        expected, summary, audit = _expected_action_map(_metrics(), cfg=_cfg())
        self.assertEqual(len(expected), 4 * 5)
        selected = summary.set_index("state_label")[
            "expected_optimal_active_dose_v_per_m"
        ]
        self.assertTrue((selected.loc["mild"] == 0.1).all())
        self.assertTrue((selected.loc["strong"] == 0.4).all())
        self.assertEqual(audit["expected_optimal_active_doses"], [0.1, 0.4])
        self.assertEqual(audit["mean_realized_oracle_agreement_fraction"], 1.0)

    def test_loso_eeg_rule_uses_one_discrete_action_per_context(self):
        expected, summary, _ = _expected_action_map(_metrics(), cfg=_cfg())
        policy = _exploratory_loso_policy(expected, summary, cfg=_cfg())
        self.assertEqual(len(policy), 4)
        selected = policy.set_index("state_label").selected_dose_v_per_m
        self.assertTrue((selected.loc["mild"] == 0.1).all())
        self.assertTrue((selected.loc["strong"] == 0.4).all())
        self.assertGreater(
            policy.contextual_advantage_over_best_fixed_log10.mean(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
