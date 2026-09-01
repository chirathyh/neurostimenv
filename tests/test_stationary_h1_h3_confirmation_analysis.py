"""Pure-analysis tests for stationary H1--H3 confirmation."""

import unittest

import numpy as np
import pandas as pd

from experiments.ballnstick_analysis.run_ballnstick_stationary_h1_h3_confirmation import (
    _bh_fdr,
    _confirmation_policy_comparison,
    _one_sample_t_power,
    _structure_preserving_shuffle,
)


def _expected_action_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    screens = []
    # Both contexts share one independent structure. The deliberately false
    # hidden labels verify that the frozen rule reads detected EEG frequency.
    contexts = (
        ("c9", 0, 9001, 11.0, 9.0),
        ("c11", 1, 9001, 9.0, 11.0),
    )
    distances = {
        "c9": {
            "sham": 1.00,
            "f9_inphase": 0.70,
            "f9_antiphase": 0.20,
            "f11_inphase": 0.85,
            "f11_antiphase": 0.90,
        },
        "c11": {
            "sham": 1.00,
            "f9_inphase": 0.85,
            "f9_antiphase": 0.90,
            "f11_inphase": 0.70,
            "f11_antiphase": 0.20,
        },
    }
    actions = (
        ("sham", 9.0, 0.0),
        ("f9_inphase", 9.0, 0.0),
        ("f9_antiphase", 9.0, np.pi),
        ("f11_inphase", 11.0, 0.0),
        ("f11_antiphase", 11.0, np.pi),
    )
    for context_id, order, structure, hidden, detected in contexts:
        screens.append({
            "context_id": context_id,
            "structure_index": 0,
            "structure_seed": structure,
            "hidden_frequency_hz": hidden,
            "detected_frequency_hz": detected,
        })
        for action_id, frequency, offset in actions:
            rows.append({
                "context_id": context_id,
                "context_order": order,
                "structure_index": 0,
                "structure_seed": structure,
                "hidden_frequency_hz": hidden,
                "detected_frequency_hz": detected,
                "action_id": action_id,
                "action_frequency_hz": frequency,
                "relative_phase_offset_rad": offset,
                "expected_distance_to_B": distances[context_id][action_id],
            })
    return pd.DataFrame(rows), pd.DataFrame(screens)


class StationaryH1H3ConfirmationAnalysisTests(unittest.TestCase):
    def test_a_priori_power_matches_declared_sample_sizes(self):
        self.assertGreaterEqual(
            _one_sample_t_power(n=16, effect_size=0.70, alpha=0.05), 0.80
        )
        self.assertGreaterEqual(
            _one_sample_t_power(n=12, effect_size=0.80, alpha=0.05), 0.80
        )
        self.assertLess(
            _one_sample_t_power(n=11, effect_size=0.80, alpha=0.05), 0.80
        )

    def test_policy_uses_detected_eeg_and_random_is_uniform_action_expectation(self):
        expected, screening = _expected_action_rows()
        comparison, structures = _confirmation_policy_comparison(
            expected,
            screening,
            frozen_fixed="f9_antiphase",
            preferred_phase=float(np.pi),
        )
        selected = comparison.set_index("context_id").policy_action_id.to_dict()
        self.assertEqual(selected, {"c9": "f9_antiphase", "c11": "f11_antiphase"})
        # c9 active outcomes are {0.70, 0.20, 0.85, 0.90}.
        c9 = comparison[comparison.context_id.eq("c9")].iloc[0]
        self.assertAlmostEqual(
            c9.random_policy_expected_distance_to_B,
            np.mean([0.70, 0.20, 0.85, 0.90]),
        )
        self.assertTrue(comparison.policy_uses_only_predecision_EEG.all())
        self.assertTrue((~comparison.policy_uses_hidden_state_or_spikes).all())
        self.assertEqual(len(structures), 1)

    def test_structure_preserving_shuffle_breaks_frequency_assignment(self):
        expected, screening = _expected_action_rows()
        comparison, _ = _confirmation_policy_comparison(
            expected,
            screening,
            frozen_fixed="f9_antiphase",
            preferred_phase=float(np.pi),
        )
        null, p_value = _structure_preserving_shuffle(
            expected,
            comparison,
            frozen_fixed="f9_antiphase",
            preferred_phase=float(np.pi),
        )
        self.assertEqual(len(null), 2)
        observed = comparison.groupby("structure_seed").policy_advantage_over_fixed.mean().mean()
        self.assertAlmostEqual(null.iloc[0].shuffled_policy_advantage, observed)
        self.assertLess(null.iloc[1].shuffled_policy_advantage, observed)
        self.assertAlmostEqual(p_value, 0.5)

    def test_benjamini_hochberg_is_monotone_in_rank(self):
        raw = np.asarray([0.03, 0.001, 0.02, 0.8])
        adjusted = _bh_fdr(raw)
        ranked = adjusted[np.argsort(raw)]
        self.assertTrue(np.all(np.diff(ranked) >= -1.0e-15))
        self.assertTrue(np.all(adjusted >= raw))
        self.assertTrue(np.all(adjusted <= 1.0))


if __name__ == "__main__":
    unittest.main()
