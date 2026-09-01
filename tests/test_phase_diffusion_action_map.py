"""Focused pure-analysis and lifecycle tests for the D1 action map."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (
    _phase_estimation_outputs,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (
    _context_action_summary,
    _context_specs,
    _diffusion_dose_interaction,
    _future_seed,
    _json_ready,
    _policy_table,
)


class PhaseDiffusionActionMapTests(unittest.TestCase):
    def test_json_outputs_replace_nonfinite_audits_with_null(self):
        self.assertEqual(
            _json_ready({"finite": 1.0, "missing": float("nan")}),
            {"finite": 1.0, "missing": None},
        )

    def test_recent_phase_tail_is_opt_in_and_backward_compatible(self):
        outputs = [{"index": index} for index in range(12)]
        historical = OmegaConf.create({"analysis": {"tacs": {}}})
        recent = OmegaConf.create({
            "analysis": {"tacs": {"phase_estimation_steps": 1}}
        })
        self.assertIs(_phase_estimation_outputs(outputs, historical), outputs)
        self.assertEqual(
            _phase_estimation_outputs(outputs, recent), [{"index": 11}]
        )
        invalid = OmegaConf.create({
            "analysis": {"tacs": {"phase_estimation_steps": 13}}
        })
        with self.assertRaises(ValueError):
            _phase_estimation_outputs(outputs, invalid)

    def test_context_grid_pairs_diffusion_and_future_random_numbers(self):
        cfg = OmegaConf.create({
            "experiment": {"seed": 1},
            "analysis": {
                "states": {
                    "frequencies_hz": [9.0, 11.0],
                    "phase_diffusion_levels": [
                        {"label": "low_diffusion", "diffusion_rad2_per_s": 0.5},
                        {"label": "high_diffusion", "diffusion_rad2_per_s": 2.0},
                    ],
                },
                "crossed_design": {
                    "n_structure_seeds": 3,
                    "n_history_seeds": 1,
                    "n_future_continuations": 2,
                    "structure_seed_offset": 421000,
                    "history_seed_offset": 440000,
                    "phase_seed_offset": 441000,
                    "trial_seed_offset": 442000,
                    "future_seed_offset": 443000,
                },
            },
        })
        specs = _context_specs(cfg)
        self.assertEqual(len(specs), 12)
        for _, pair in pd.DataFrame(specs).groupby("paired_diffusion_context_id"):
            self.assertEqual(set(pair.label), {"low_diffusion", "high_diffusion"})
            self.assertEqual(pair.structure_seed.nunique(), 1)
            self.assertEqual(pair.history_seed.nunique(), 1)
            self.assertEqual(pair.phase_seed.nunique(), 1)
            rows = pair.to_dict(orient="records")
            self.assertEqual(_future_seed(cfg, rows[0], 0), _future_seed(cfg, rows[1], 0))

    @staticmethod
    def _synthetic_tables():
        expected_rows = []
        metric_rows = []
        context_order = 0
        for structure in (1, 2, 3):
            for label, diffusion, c1 in (
                ("high_diffusion", 2.0, 0.2),
                ("low_diffusion", 0.5, 0.8),
            ):
                context_id = f"s{structure}_{label}"
                distances = (
                    {0.0: 0.30, 0.2: 0.10, 0.4: 0.20}
                    if c1 < 0.5 else
                    {0.0: 0.30, 0.2: 0.20, 0.4: 0.10}
                )
                for dose, distance in distances.items():
                    expected_rows.append({
                        "context_id": context_id,
                        "paired_diffusion_context_id": f"s{structure}",
                        "context_order": context_order,
                        "future_group_index": structure,
                        "structure_index": structure - 1,
                        "structure_seed": structure,
                        "history_index": 0,
                        "history_seed": 100 + structure,
                        "phase_seed": 200 + structure,
                        "hidden_frequency_hz": 9.0,
                        "label": label,
                        "diffusion_rad2_per_s": diffusion,
                        "context_C1": c1,
                        "context_C1_abs": abs(c1),
                        "context_C1_imag": 0.0,
                        "context_C1_temporal_sd": 0.1,
                        "context_spectral_concentration": c1,
                        "context_spectral_rms_width_hz": 1.0 - c1,
                        "context_log10_alpha_power": -20.0,
                        "context_alpha_excess_log10": 0.2,
                        "context_distance_to_B_log10": 0.2,
                        "EEG_selected_frequency_hz": 9.0,
                        "recent_resultant_to_rms": 0.1,
                        "dose_v_per_m": dose,
                        "action_id": str(dose),
                        "n_future_continuations": 2,
                        "expected_post_distance_to_B_log10": distance,
                    })
                    for future in (1, 2):
                        metric_rows.append({
                            "context_id": context_id,
                            "future_index": future,
                            "dose_v_per_m": dose,
                            "post_distance_to_B_log10": distance,
                        })
                context_order += 1
        return pd.DataFrame(expected_rows), pd.DataFrame(metric_rows)

    def test_expected_response_not_single_future_drives_action_and_policy(self):
        expected, metrics = self._synthetic_tables()
        cfg = OmegaConf.create({
            "analysis": {
                "criteria": {"practical_action_advantage_log10": 0.01},
                "context": {
                    "policy_features": ["context_C1"],
                    "ridge_penalty": 1.0e-3,
                },
                "actions": {
                    "sham_dose_v_per_m": 0.0,
                    "active_doses_v_per_m": [0.2, 0.4],
                },
            }
        })
        summary, _, audit = _context_action_summary(expected, metrics, cfg)
        self.assertEqual(set(summary.expected_optimal_dose_v_per_m), {0.2, 0.4})
        self.assertEqual(audit["practical_nonfixed_structure_count"], 3)
        policy = _policy_table(expected, summary, cfg)
        self.assertEqual(set(policy.selected_dose_v_per_m), {0.2, 0.4})
        self.assertGreater(
            policy.contextual_advantage_over_best_fixed_log10.mean(), 0.0
        )
        self.assertTrue(policy.policy_uses_only_predecision_C1.all())

    def test_diffusion_interaction_is_paired_and_has_expected_magnitude(self):
        expected, _ = self._synthetic_tables()
        cfg = OmegaConf.create({
            "analysis": {
                "actions": {"active_doses_v_per_m": [0.2, 0.4]}
            }
        })
        paired, audit = _diffusion_dose_interaction(expected, cfg)
        self.assertEqual(len(paired), 3)
        self.assertTrue(np.allclose(
            paired.absolute_diffusion_by_dose_interaction_log10, 0.2
        ))
        self.assertAlmostEqual(
            audit["mean_absolute_diffusion_by_dose_interaction_log10"], 0.2
        )


if __name__ == "__main__":
    unittest.main()
