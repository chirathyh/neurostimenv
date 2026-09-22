import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from experiments.ballnstick_analysis.qualification import common
from experiments.ballnstick_analysis.qualification import phase_measurement as pm


def config():
    with initialize_config_dir(version_base=None, config_dir=str(common.REPO/"configs")):
        return compose(config_name="config", overrides=["env=ballnstick", "analysis=ballnstick_cf0b",
            "env.simulation.obs_win_len=1000"])


class PhaseMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.settings = OmegaConf.to_container(self.cfg.analysis.phase_audit)

    def test_known_phase_and_centered_reference_convention(self):
        fs, f, phi = 1000, 9.37, .731
        t = (np.arange(4000)+1)/fs
        x = 3+.002*t+np.cos(2*np.pi*f*t+phi)
        estimated, _ = pm.estimate_phase(x[:2000], fs, f, 2., .5, "raw_ols")
        reference, _ = pm.centered_reference(x, fs, f, 2., 1.)
        expected = pm.wrap(2*np.pi*f*2+phi)
        self.assertLess(abs(pm.wrap(estimated-expected)), 1e-10)
        self.assertLess(abs(pm.wrap(reference-expected)), 1e-10)
        self.assertTrue(np.isnan(pm.centered_reference(x, fs, f, 3.8, 1.)[0]))

    def test_h4_matches_historical_primitives_exactly(self):
        from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import _tail_phase_estimate
        cfg = self.cfg.copy()
        with open_dict(cfg.analysis):
            cfg.analysis.tacs = {"frequency_hz": 9.37}
        fs = 16000
        times = (np.arange(fs)+1)/fs
        x = np.cos(2*np.pi*9.37*times+.713)+.2*np.random.default_rng(3).normal(size=fs)
        expected = _tail_phase_estimate([{"eeg_v": x, "sample_times_ms": times*1000}],
            boundary_ms=1000, history_ms=500, simulator_fs_hz=fs,
            relative_offset_rad=np.pi, cfg=cfg)
        actual = pm.estimate_phase(x, fs, 9.37, 1., .5, "h4_tail")
        self.assertAlmostEqual(actual[0], expected["estimated_eeg_phase_at_boundary_rad"])
        self.assertAlmostEqual(actual[1], expected["resultant_to_rms"])

    def test_future_mutation_cannot_change_deployable_phase(self):
        fs = 1000
        t = (np.arange(4000)+1)/fs
        x = np.cos(2*np.pi*10.3*t)
        changed = x.copy(); changed[2200:] += 1e7
        first = pm.audit_trajectory(x, x, fs, 2000, 10.3, self.settings)
        second = pm.audit_trajectory(changed, changed, fs, 2000, 10.3, self.settings)
        keep = first.boundary_s <= 2.2
        np.testing.assert_array_equal(first.loc[keep, "phase_rad"], second.loc[keep, "phase_rad"])
        # Offline reference is explicitly allowed to change from future samples.
        self.assertFalse(np.array_equal(first.loc[keep, "common_reference_rad"], second.loc[keep, "common_reference_rad"]))
        self.assertTrue((first.latest_input_s <= first.boundary_s).all())

    def test_references_do_not_supply_observed_estimate(self):
        fs = 1000
        t = (np.arange(4000)+1)/fs
        neural = np.cos(2*np.pi*10*t)
        observed = np.cos(2*np.pi*10*t+.4)
        a = pm.audit_trajectory(neural, observed, fs, 2000, 10., self.settings)
        b = pm.audit_trajectory(-neural, observed, fs, 2000, 10., self.settings)
        np.testing.assert_array_equal(a.phase_rad, b.phase_rad)
        np.testing.assert_array_equal(a.confidence, b.confidence)
        self.assertGreater(np.nanmean(b.reference_error_deg), 100)

    def test_invalid_inputs(self):
        for x in (np.ones(10), np.full(1000, np.nan)):
            with self.assertRaises(ValueError):
                pm.estimate_phase(x, 1000, 10, 1., .5, "h4_tail")
        with self.assertRaises(ValueError):
            pm.estimate_phase(np.ones(1000), 1000, 10, 1., .5, "unknown")

    def test_rank_screen_strict_ties_and_insufficient_calibration(self):
        screen = pm.rank_threshold(np.arange(19), .05)
        self.assertEqual(screen["rank"], 19)
        self.assertEqual(screen["threshold_db"], 18)
        self.assertFalse(pm.rhythm_present(18, screen))
        self.assertTrue(pm.rhythm_present(18.001, screen))
        self.assertFalse(pm.rhythm_present(np.nan, screen))
        too_small = pm.rank_threshold(np.arange(18), .05)
        self.assertIsNone(too_small["threshold_db"])
        self.assertFalse(pm.rhythm_present(1e9, too_small))

    def make_rows(self):
        rows = []
        for s in range(3):
            for d in (.5, 2.):
                for profile in ("cf0_raw", "h4_original"):
                    for window in range(10):
                        rows.append(dict(profile=profile, structure_seed=s, context=f"{s}_{d}", D=d,
                            state="A", actionable=True, neural_actionable=True,
                            noise_error_deg=5., reference_error_deg=10.,
                            reference_valid=True, large_error=False, prediction_error_deg=20.))
        return pd.DataFrame(rows)

    def test_pass_and_abstain_all_failure_and_large_error(self):
        rows = self.make_rows()
        summaries, contexts, structures = pm.phase_summary(rows, self.settings, 3)
        self.assertEqual(len(structures), 6)
        self.assertTrue(all(s["passes"] for s in summaries.values()))
        rows["actionable"] = False
        summaries, _, _ = pm.phase_summary(rows, self.settings, 3)
        self.assertFalse(any(s["passes"] for s in summaries.values()))
        self.assertIsNone(pm.choose_profile(summaries, self.settings["candidates"]))
        rows = self.make_rows(); rows["reference_error_deg"] = 100
        summaries, _, _ = pm.phase_summary(rows, self.settings, 3)
        self.assertFalse(any(s["passes"] for s in summaries.values()))

    def test_repeat_windows_do_not_change_structure_unit(self):
        original = self.make_rows()
        extra = pd.concat([original, original[original.structure_seed == 0]]*5, ignore_index=True)
        a, _, sa = pm.phase_summary(original, self.settings, 3)
        b, _, sb = pm.phase_summary(extra, self.settings, 3)
        self.assertEqual(a, b)
        self.assertEqual(len(sa), len(sb))

    def test_frozen_design_and_disjoint_contexts(self):
        from experiments.ballnstick_analysis.run_ballnstick_cf0b import validate, specs
        from experiments.ballnstick_analysis.qualification.continuous_alpha import contexts
        validate(self.cfg)
        source = {"source_contexts": [contexts(OmegaConf.to_container(self.cfg.analysis.design), "discovery")]}
        calibration, validation = specs(self.cfg, source)
        self.assertEqual((len(calibration), len(validation)), (19, 30))
        self.assertTrue(all("." not in r["id"] for r in calibration+validation))
        self.assertEqual(len({r["structure_seed"] for r in validation}), 6)
        self.cfg.analysis.smoke = True
        smoke_cal, smoke_val = specs(self.cfg, source)
        self.assertFalse({r["structure_seed"] for r in calibration+validation} &
                         {r["structure_seed"] for r in smoke_cal+smoke_val})
        self.cfg.analysis.smoke = False
        self.cfg.analysis.phase_audit.maximum_reference_error_deg = 90.
        with self.assertRaises(ValueError):
            validate(self.cfg)

    def test_source_hash_lock_rejects_wrong_conclusion(self):
        from experiments.ballnstick_analysis.run_ballnstick_cf0b import load_source
        with tempfile.TemporaryDirectory() as tmp:
            self.cfg.analysis.source_cf0.result_dir = tmp
            for name in ("run_complete.json", "experiment_conclusion.json", "frozen_estimator.json",
                         "resolved_config.yaml", "context_metrics.csv", "prespecified_contexts.json"):
                (Path(tmp)/name).write_text("{}")
            with self.assertRaisesRegex(ValueError, "exact completed negative"):
                load_source(self.cfg)

    def test_small_B_interval_is_not_powered_specificity(self):
        low, high = pm.exact_binomial_interval(6, 6)
        self.assertLess(low, .6)
        self.assertEqual(high, 1.)

    def test_screening_fallback_cannot_use_hidden_frequency_or_spikes(self):
        from experiments.ballnstick_analysis.run_ballnstick_cf0b import apply_frozen_screen
        screen = pm.rank_threshold(np.arange(19), .05)
        phases = pd.DataFrame([dict(profile="h4_original", elapsed_s=0., actionable=True)])
        # There are deliberately no hidden labels or safety/spike columns.
        row, phase_rows = apply_frozen_screen({"evidence_db": 19., "accepted": True},
            phases.copy(), screen, {"name": "h4_original"})
        self.assertTrue(row["initial_measurement_eligible"])
        self.assertFalse(row["stimulation_actually_applied"])
        for evidence, carrier, phase in ((18., True, True), (19., False, True), (19., True, False)):
            phases["actionable"] = phase
            row, _ = apply_frozen_screen({"evidence_db": evidence, "accepted": carrier},
                phases.copy(), screen, {"name": "h4_original"})
            self.assertEqual(row["future_treatment_fallback"], "sham")


if __name__ == "__main__":
    unittest.main()
