"""Pure-analysis tests for the F0 frequency/phase feasibility map."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (
    SHAM,
    _action_specs,
    _fit_reference_target,
    _policy_comparison,
    _target_distance,
)


class FrequencyPhaseFeasibilityAnalysisTests(unittest.TestCase):
    def test_action_grid_fixes_amplitude_and_crosses_frequency_with_phase(self):
        cfg = OmegaConf.create({
            "analysis": {
                "tacs": {
                    "frequencies_hz": [9.0, 11.0],
                    "relative_phase_offsets_rad": [0.0, float(np.pi)],
                    "amplitude_v_per_m": 0.4,
                }
            }
        })
        actions = _action_specs(cfg)
        self.assertEqual(len(actions), 5)
        self.assertEqual(actions[0]["id"], SHAM)
        active = actions[1:]
        self.assertEqual({x["frequency_hz"] for x in active}, {9.0, 11.0})
        self.assertEqual(
            {round(x["relative_phase_offset_rad"], 12) for x in active},
            {0.0, round(float(np.pi), 12)},
        )
        self.assertTrue(all(np.isclose(x["ac_amplitude_v_per_m"], 0.4) for x in active))

    def test_reference_distance_uses_both_frozen_spectral_features(self):
        cfg = OmegaConf.create({
            "analysis": {
                "states": {"frequencies_hz": [9.0, 11.0]},
                "spectral_target": {
                    "minimum_scale_log10": 0.05,
                    "reference_quantile": 0.95,
                },
            }
        })
        rows = pd.DataFrame({
            "log10_power_9hz": [-10.0, -10.1, -9.9],
            "log10_power_11hz": [-11.0, -11.1, -10.9],
        })
        target = _fit_reference_target(rows, cfg)
        at_target = {
            "log10_power_9hz": target["means"]["log10_power_9hz"],
            "log10_power_11hz": target["means"]["log10_power_11hz"],
        }
        self.assertAlmostEqual(_target_distance(at_target, target), 0.0)
        displaced = dict(at_target)
        displaced["log10_power_11hz"] += target["scales"]["log10_power_11hz"]
        self.assertAlmostEqual(_target_distance(displaced, target), np.sqrt(0.5))

    def test_candidate_policy_uses_detected_eeg_frequency_not_hidden_label(self):
        action_rows = []
        contexts = [
            ("c0", 0, 100, 11.0, 9.0),  # deliberately wrong hidden label
            ("c1", 1, 101, 9.0, 11.0),
        ]
        distances = {
            "c0": {"f9_antiphase": 0.1, "f11_antiphase": 0.9,
                   "f9_inphase": 0.5, "f11_inphase": 0.8},
            "c1": {"f9_antiphase": 0.9, "f11_antiphase": 0.1,
                   "f9_inphase": 0.8, "f11_inphase": 0.5},
        }
        for context_id, order, structure, hidden, detected in contexts:
            for action_id, frequency, offset in (
                ("f9_inphase", 9.0, 0.0),
                ("f9_antiphase", 9.0, np.pi),
                ("f11_inphase", 11.0, 0.0),
                ("f11_antiphase", 11.0, np.pi),
            ):
                action_rows.append({
                    "context_id": context_id,
                    "context_order": order,
                    "structure_index": order,
                    "structure_seed": structure,
                    "hidden_frequency_hz": hidden,
                    "detected_frequency_hz": detected,
                    "action_id": action_id,
                    "action_frequency_hz": frequency,
                    "relative_phase_offset_rad": offset,
                    "expected_distance_to_B": distances[context_id][action_id],
                })
        expected = pd.DataFrame(action_rows)
        screening = pd.DataFrame([
            {
                "context_id": context_id,
                "context_order": order,
                "structure_index": order,
                "structure_seed": structure,
                "hidden_frequency_hz": hidden,
                "detected_frequency_hz": detected,
            }
            for context_id, order, structure, hidden, detected in contexts
        ])
        comparison, _, _ = _policy_comparison(expected, screening)
        selected = comparison.set_index("context_id").policy_action_id.to_dict()
        self.assertEqual(selected, {"c0": "f9_antiphase", "c1": "f11_antiphase"})
        self.assertTrue(comparison.policy_uses_only_detected_frequency_and_frozen_antiphase.all())


if __name__ == "__main__":
    unittest.main()
