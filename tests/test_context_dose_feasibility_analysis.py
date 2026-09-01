"""Tests for the crossed-seed CL0 context/dose feasibility analysis."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_context_dose_feasibility import (
    _counterfactual_summary,
    _cross_validated_context_policy,
    _two_way_variance,
    _wide_context_table,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (
    _mpi_variables,
)


def _cfg():
    return OmegaConf.create({
        "analysis": {
            "actions": {
                "active_doses_v_per_m": [0.2, 0.4],
                "low_dose_v_per_m": 0.2,
                "fixed_comparator_v_per_m": 0.4,
            },
            "criteria": {"practical_advantage_log10": 0.01},
            "context_model": {"ridge_penalty": 1.0e-6},
        }
    })


def _metric_rows():
    rows = []
    context_order = 0
    for structure in (1, 2, 3):
        for drive, alpha_excess in ((11, 0.1), (12, 0.4)):
            context_order += 1
            distances = {0.0: 0.30, 0.2: 0.01 if alpha_excess < 0.2 else 0.20,
                         0.4: 0.20 if alpha_excess < 0.2 else 0.01}
            for dose, distance in distances.items():
                rows.append({
                    "context_id": f"s{structure}_d{drive}",
                    "context_order": context_order,
                    "trial_seed": 1000 + context_order,
                    "structure_seed": structure,
                    "drive_seed": drive,
                    "phase_seed": 99,
                    "structure_index": structure,
                    "drive_index": 1 if drive == 11 else 2,
                    "context_alpha_excess_log10": alpha_excess,
                    "context_coherent_alpha_fraction": alpha_excess,
                    "context_log10_alpha_power": -20.5 + alpha_excess,
                    "context_10hz_resultant_v": 1.0e-10,
                    "context_alpha_peak_prominence_db": 5.0,
                    "screen_margin_toward_A_log10": 0.1,
                    "screen_phase_split_error_deg": 5.0,
                    "screen_10hz_resultant_to_rms": 0.2,
                    "pre_action_distance_to_B_log10": 0.3,
                    "dose_v_per_m": dose,
                    "post_distance_to_B_log10": distance,
                    "reward_negative_distance": -distance,
                    "alpha_suppression_log10": 0.3 - distance,
                    "induced_eeg_10hz_resultant_v": dose * 1.0e-10,
                    "E_ppc_reduction": dose * 0.01,
                })
    return pd.DataFrame(rows)


class ContextDoseFeasibilityTests(unittest.TestCase):
    def test_mpi_seed_namespaces_can_vary_independently(self):
        marker = object()
        values = _mpi_variables(
            marker,
            size=4,
            rank=2,
            seed=7,
            structure_seed=101,
            drive_seed=202,
        )
        self.assertIs(values["COMM"], marker)
        self.assertEqual(values["SEED"], 1_010_000)
        self.assertEqual(values["GLOBALSEED"], 202)

        split = _mpi_variables(
            marker,
            size=1,
            rank=0,
            seed=7,
            structure_seed=101,
            drive_seed=202,
            future_drive_seed=303,
            future_start_ms=5_000.0,
        )
        self.assertEqual(split["GLOBALSEED"], 202)
        self.assertEqual(split["FUTUREGLOBALSEED"], 303)
        self.assertEqual(split["FUTURESTARTMS"], 5_000.0)

        legacy = _mpi_variables(marker, size=1, rank=0, seed=7)
        self.assertEqual(legacy["SEED"], 70_000)
        self.assertEqual(legacy["GLOBALSEED"], 7)

    def test_counterfactual_oracle_uses_target_distance_not_suppression(self):
        metrics = _metric_rows()
        summary = _counterfactual_summary(metrics, cfg=_cfg())
        low_context = summary.iloc[0]
        high_context = summary.iloc[1]
        self.assertEqual(low_context.oracle_dose_v_per_m, 0.2)
        self.assertEqual(high_context.oracle_dose_v_per_m, 0.4)
        self.assertAlmostEqual(
            low_context.oracle_advantage_over_fixed_log10, 0.19
        )
        self.assertTrue(low_context.oracle_practically_beats_fixed)
        self.assertFalse(high_context.oracle_practically_beats_fixed)

    def test_leave_one_structure_out_rule_recovers_context_action_interaction(self):
        metrics = _metric_rows()
        summary = _counterfactual_summary(metrics, cfg=_cfg())
        table = _wide_context_table(metrics, summary, cfg=_cfg())
        policy = _cross_validated_context_policy(table, cfg=_cfg())
        self.assertEqual(len(policy), 6)
        selected = policy.set_index("context_id").selected_dose_v_per_m
        for context_id, dose in selected.items():
            expected = 0.2 if context_id.endswith("d11") else 0.4
            self.assertEqual(dose, expected)
        self.assertGreater(
            policy.contextual_advantage_over_fixed_log10.mean(), 0.0
        )

    def test_two_way_variance_separates_structure_and_drive_effects(self):
        frame = pd.DataFrame({
            "structure_seed": [1, 1, 2, 2],
            "drive_seed": [10, 20, 10, 20],
            "value": [0.0, 2.0, 1.0, 3.0],
        })
        result = _two_way_variance(frame, value="value")
        self.assertTrue(result["balanced_complete_grid"])
        self.assertAlmostEqual(result["structure_fraction"], 0.2)
        self.assertAlmostEqual(result["drive_fraction"], 0.8)
        self.assertAlmostEqual(result["interaction_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
