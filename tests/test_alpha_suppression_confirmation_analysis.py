"""Tests for frozen alpha-suppression confirmation statistics."""

import unittest

import numpy as np

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_confirmation import (
    _circular_difference,
    _exact_sign_flip_p,
)


class AlphaSuppressionConfirmationTests(unittest.TestCase):
    def test_eight_same_direction_seeds_reach_exact_significance(self):
        values = np.arange(1.0, 9.0)
        self.assertAlmostEqual(_exact_sign_flip_p(values), 2.0 / 256.0)

    def test_mixed_sign_effect_is_not_significant(self):
        values = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        self.assertEqual(_exact_sign_flip_p(values), 1.0)

    def test_exact_test_preserves_small_voltage_units(self):
        values = np.arange(1.0, 9.0) * 1.0e-12
        self.assertAlmostEqual(_exact_sign_flip_p(values), 2.0 / 256.0)

    def test_circular_difference_wraps_at_two_pi(self):
        difference = _circular_difference(np.deg2rad(5.0), np.deg2rad(355.0))
        self.assertAlmostEqual(np.degrees(difference), 10.0)


if __name__ == "__main__":
    unittest.main()
