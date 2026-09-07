import unittest

from setup.circuits.ballnstick.utils import (
    exp2syn_conductance_time_area,
    i_to_e_kinetics_and_weight_scale,
)


class TestBallAndStickSynapseKinetics(unittest.TestCase):
    def test_exp2syn_area_requires_ordered_positive_time_constants(self):
        for tau1, tau2 in ((0.0, 9.0), (9.0, 9.0), (10.0, 9.0)):
            with self.assertRaises(ValueError):
                exp2syn_conductance_time_area(tau1, tau2)

    def test_area_preserving_i_to_e_scaling_is_exact(self):
        baseline = exp2syn_conductance_time_area(0.1, 9.0)
        for multiplier in (0.8, 1.0, 1.2):
            tau2, scale, ratio = i_to_e_kinetics_and_weight_scale(
                tau1_ms=0.1,
                tau2_ms=9.0,
                tau2_multiplier=multiplier,
                preserve_conductance_time_area=True,
            )
            perturbed = exp2syn_conductance_time_area(0.1, tau2)
            self.assertAlmostEqual(tau2, 9.0 * multiplier)
            self.assertAlmostEqual(scale * perturbed, baseline, places=12)
            self.assertAlmostEqual(ratio, 1.0, places=12)

    def test_disabled_normalization_preserves_peak_weight(self):
        tau2, scale, ratio = i_to_e_kinetics_and_weight_scale(
            tau1_ms=0.1,
            tau2_ms=9.0,
            tau2_multiplier=1.2,
            preserve_conductance_time_area=False,
        )
        self.assertAlmostEqual(tau2, 10.8)
        self.assertEqual(scale, 1.0)
        self.assertGreater(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
