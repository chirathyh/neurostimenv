import unittest

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.qualification import common, field_sensitivity as fs


class FieldSensitivityTests(unittest.TestCase):
    def setUp(self):
        with initialize_config_dir(version_base=None, config_dir=str(common.REPO / "configs")):
            self.cfg = compose(config_name="config", overrides=["env=ballnstick", "analysis=ballnstick_fs0"])

    def test_stable_voltage_limits_and_positive_slope(self):
        self.assertEqual(float(fs.exprel(0.)), 1.)
        self.assertAlmostEqual(float(fs.exprel(1e-8)), 1+5e-9)
        v = np.r_[np.linspace(-100, 60, 1000), -20., 10.]
        self.assertTrue(np.isfinite(fs.activation(v)).all())
        self.assertTrue(((fs.activation(v) > 0) & (fs.activation(v) < 1)).all())
        self.assertGreater(fs.unit_slope(-65.), 0)
        # Source-law equilibrium approximation is fast relative to alpha periods.
        self.assertLess(np.max(fs.inherited_relaxation_ms(v)), .1)

    def test_tonic_ratio_preserved_not_arbitrary_field_scaling(self):
        a = OmegaConf.to_container(self.cfg.analysis.tonic)
        c = fs.calibration(a)
        self.assertAlmostEqual(c["reference_gbar_S_per_cm2"], .000938*.0002/.0000954)
        toy_ratio = c["reference_gbar_S_per_cm2"]*fs.unit_slope(-65)/.0002
        self.assertAlmostEqual(toy_ratio, c["tonic_to_leak_slope_ratio_at_reference"])
        self.assertEqual(a["low_fraction"], .6)

    def test_grid_and_diagnostic_pairing(self):
        cases = fs.cases(OmegaConf.to_container(self.cfg.analysis))
        self.assertEqual(len(cases), 35)
        self.assertEqual(len({c["id"] for c in cases}), 35)
        self.assertEqual(sum(c["kind"] == "field" for c in cases), 18)
        self.assertEqual(sum(c["kind"] == "sham" for c in cases), 4)
        for c in cases:
            if c["kind"] in ("dt_refinement", "space_refinement", "transverse"):
                self.assertTrue(any(b["kind"] == "field" and all(b[k] == c[k] for k in
                    ("condition", "frequency_hz", "amplitude_v_per_m")) for b in cases))

    def test_harmonic_has_correct_phase_and_amplitude(self):
        t = np.arange(16000)*.125
        x = 4+2*np.cos(2*np.pi*10*t/1000+.83)
        expected = 2*np.exp(.83j)
        self.assertLess(abs(fs.harmonic(x, t, 10)-expected), 1e-12)

    def test_full_grid_protected(self):
        from experiments.ballnstick_analysis.run_ballnstick_fs0 import validate
        validate(self.cfg)
        self.cfg.analysis.frequencies_hz = [10.]
        with self.assertRaises(ValueError):
            validate(self.cfg)
        self.cfg.analysis.smoke = True
        validate(self.cfg)

    def test_modified_source_parameter_rejected(self):
        from experiments.ballnstick_analysis.run_ballnstick_fs0 import validate
        self.cfg.analysis.tonic.l23_apical_gbar_S_per_cm2 = .001
        with self.assertRaises(ValueError):
            validate(self.cfg)


if __name__ == "__main__":
    unittest.main()
