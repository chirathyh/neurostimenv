"""Unit tests for the alpha-suppression dose/mechanism audit."""

import unittest

import numpy as np
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (
    _complex_response_decomposition,
    _dose_id,
    _field_removal_status,
    _fit_zero_intercept_quadratic,
)


class AlphaSuppressionDoseAuditTests(unittest.TestCase):
    def test_coherent_decomposition_is_exact(self):
        result = _complex_response_decomposition(
            sham_cosine=2.0,
            sham_sine=-1.0,
            active_cosine=1.0,
            active_sine=3.0,
        )
        self.assertAlmostEqual(
            result["coherent_net_change_v2"],
            result["coherent_interference_cross_term_v2"]
            + result["coherent_induced_component_v2"],
        )
        self.assertAlmostEqual(result["coherent_net_change_v2"], 5.0)

    def test_destructive_cross_term_can_be_outweighed_by_induced_component(self):
        result = _complex_response_decomposition(
            sham_cosine=1.0,
            sham_sine=0.0,
            active_cosine=-2.0,
            active_sine=0.0,
        )
        self.assertLess(result["coherent_interference_cross_term_v2"], 0.0)
        self.assertGreater(result["coherent_induced_component_v2"], 0.0)
        self.assertGreater(result["coherent_net_change_v2"], 0.0)

    def test_washout_recovery_does_not_depend_on_effect_sign(self):
        cfg = OmegaConf.create({
            "analysis": {
                "criteria": {
                    "maximum_washout_absolute_log10": 1.0e-6,
                    "maximum_washout_residual_fraction": 0.5,
                }
            }
        })
        positive, positive_tolerance = _field_removal_status(
            effect_log10=0.1, residual_log10=0.01, cfg=cfg
        )
        adverse, adverse_tolerance = _field_removal_status(
            effect_log10=-0.1, residual_log10=0.01, cfg=cfg
        )
        self.assertTrue(positive)
        self.assertTrue(adverse)
        self.assertEqual(positive_tolerance, adverse_tolerance)

    def test_quadratic_fit_recovers_interior_maximum(self):
        doses = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8])
        effects = 0.8 * doses - doses**2
        fit = _fit_zero_intercept_quadratic(doses, effects)
        self.assertAlmostEqual(fit["linear_coefficient"], 0.8)
        self.assertAlmostEqual(fit["quadratic_coefficient"], -1.0)
        self.assertAlmostEqual(fit["turning_dose_v_per_m"], 0.4)
        self.assertAlmostEqual(fit["r_squared"], 1.0)

    def test_dose_identifier_is_filesystem_stable(self):
        self.assertEqual(_dose_id(0.2), "A_tacs_180deg_0p2_vpm")


if __name__ == "__main__":
    unittest.main()
