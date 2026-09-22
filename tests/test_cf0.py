import unittest
import tempfile
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.qualification import common, continuous_alpha as ca


def config():
    with initialize_config_dir(version_base=None, config_dir=str(common.REPO / "configs")):
        return compose(config_name="config", overrides=["env=ballnstick", "analysis=ballnstick_cf0", "env.simulation.obs_win_len=1000"])


class ContinuousAlphaTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.params = OmegaConf.to_container(self.cfg.analysis.estimator.candidates[0])

    def test_stratified_continuous_disjoint_grid(self):
        d = OmegaConf.to_container(self.cfg.analysis.design)
        a, b = [ca.contexts(d, s) for s in ("discovery", "replication")]
        self.assertEqual(len(a), 15)
        self.assertEqual(len(b), 15)
        self.assertFalse({r["structure_seed"] for r in a} & {r["structure_seed"] for r in b})
        carriers = [r["carrier_hz"] for r in a if r["state"] == "A"]
        self.assertTrue(all(8 < f < 12 and not f.is_integer() for f in carriers))
        self.assertEqual(len(set(carriers)), 6)
        self.assertEqual(len({r["id"] for r in a+b}), 30)

    def test_continuous_estimation_including_band_edges(self):
        fs = 100.
        t = np.arange(3000)/fs
        for f in (8.07, 9.37, 10.63, 11.91):
            x = np.cos(2*np.pi*f*t+.7)+.25*np.random.default_rng(31).normal(size=len(t))
            result = ca.estimate(x, fs, self.params)
            self.assertTrue(result["accepted"])
            self.assertLess(abs(result["frequency_hz"]-f), .15)
            self.assertNotIn(result["frequency_hz"], [9., 11.])

    def test_equal_competing_peaks_abstain(self):
        t = np.arange(3000)/100.
        x = np.cos(2*np.pi*8.7*t)+np.cos(2*np.pi*11.3*t)
        result = ca.estimate(x, 100, self.params)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["fallback"], "sham")

    def test_zero_signal_abstains_and_nonfinite_rejected(self):
        self.assertFalse(ca.estimate(np.zeros(3000), 100, self.params)["accepted"])
        with self.assertRaises(ValueError):
            ca.estimate(np.full(3000, np.nan), 100, self.params)

    def test_noise_scaling_and_prefix_do_not_use_future(self):
        x = np.random.default_rng(91).normal(size=3000)
        obs, unit, scale = ca.baseline_scaled_noise(x, 2000, .95, .25, 32)
        altered = np.r_[x[:2000], x[2000:]*10000, np.zeros(500)]
        obs2, unit2, scale2 = ca.baseline_scaled_noise(altered, 2000, .95, .25, 32)
        np.testing.assert_array_equal(unit, unit2[:len(unit)])
        np.testing.assert_array_equal(obs[:2000], obs2[:2000])
        self.assertEqual(scale, scale2)
        self.assertAlmostEqual(np.sqrt(np.mean((obs[:2000]-x[:2000])**2))/np.std(x[:2000]), .25)

    def test_noise_rejects_unflattened_online_eeg(self):
        with self.assertRaises(ValueError):
            ca.baseline_scaled_noise(np.ones((1, 16000)), 16000, .95, .25, 32)

    def test_causal_phase_uses_only_tail(self):
        fs, frequency, phase = 1000, 9.372, .91
        t = (np.arange(4000)+1)/fs
        x = np.cos(2*np.pi*frequency*t+phase)+.01*t
        estimated, confidence = ca.phase_at_boundary(x[:2000], fs, frequency, 2., .5)
        expected = 2*np.pi*frequency*2+phase
        self.assertLess(abs(np.angle(np.exp(1j*(estimated-expected)))), 1e-10)
        self.assertGreater(confidence, 1.)
        altered = x[:2000].copy()
        altered[:1500] = 1e9
        self.assertEqual(ca.phase_at_boundary(altered, fs, frequency, 2., .5), (estimated, confidence))

    def test_generator_condition_is_separate_and_B_has_zero_depth(self):
        from experiments.ballnstick_analysis.run_ballnstick_cf0 import condition, validate
        stages = validate(self.cfg)
        cfg_a = condition(self.cfg, stages[0][0])
        cfg_b = condition(self.cfg, stages[0][-1])
        for pop in ("E", "I"):
            self.assertEqual(cfg_a.env.network.background[pop].rhythm.modulation_depth, .04)
            self.assertEqual(cfg_b.env.network.background[pop].rhythm.modulation_depth, 0.)
            self.assertEqual(cfg_a.env.network.background[pop].interval_ms, cfg_b.env.network.background[pop].interval_ms)
        self.assertFalse(self.cfg.env.network.background.E.rhythm.enabled)

    def test_shortened_full_protocol_rejected(self):
        from experiments.ballnstick_analysis.run_ballnstick_cf0 import validate
        self.cfg.analysis.timeline.baseline_steps = 4
        with self.assertRaises(ValueError):
            validate(self.cfg)
        self.cfg.analysis.smoke = True
        validate(self.cfg)

    def test_success_and_abstain_all_gates_are_structure_level(self):
        import pandas as pd
        from experiments.ballnstick_analysis.run_ballnstick_cf0 import summarize
        rows = []
        for structure in range(3):
            for d in (.5, 2.):
                for frequency in (8.6, 11.4):
                    rows.append(dict(structure_seed=structure, D=d, state="A",
                        carrier_hz=frequency, absolute_error_hz=.05, accepted=True,
                        phase_actionable_fraction=1., rate_safe=True,
                        field_residual_mV=0., neural_error_hz=.04))
            rows.append({**rows[-1], "state": "B", "accepted": False})
        frame = pd.DataFrame(rows)
        summary, structures = summarize(frame, self.cfg)
        self.assertTrue(summary["passes"])
        self.assertEqual(len(structures), 3)
        self.assertEqual(summary["B_rhythm_false_acceptance_audit"], 0)
        frame["accepted"] = False
        summary, _ = summarize(frame, self.cfg)
        self.assertFalse(summary["passes"])
        self.assertFalse(summary["checks"]["identification_coverage"])

    def test_completion_manifest_and_overwrite_protection(self):
        import time, json
        from experiments.ballnstick_analysis import run_ballnstick_cf0 as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"run"
            common.begin(root, self.cfg, runner.__file__)
            self.assertFalse((root/"run_complete.json").exists())
            with self.assertRaises(FileExistsError):
                common.begin(root, self.cfg, runner.__file__)
            common.finish(root, time.perf_counter(), {"scientific_gate": False})
            marker = json.loads((root/"run_complete.json").read_text())
            self.assertTrue(marker["completed"])
            self.assertTrue(all(common.sha256(root/k)==v for k,v in marker["files_sha256"].items()))


if __name__ == "__main__":
    unittest.main()
