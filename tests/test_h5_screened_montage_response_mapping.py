"""Frozen design, causal measurement, and honest residual-opportunity inference."""
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from experiments.ballnstick_analysis import run_ballnstick_h5_screened_montage_response_mapping as m


class ScreenedMontageMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
            cls.cfg = compose(config_name="config", overrides=[
                "env=ballnstick", "analysis=ballnstick_h5_screened_montage_response_mapping",
                "experiment.seed=1", "env.simulation.obs_win_len=1000",
            ])
            cls.old = compose(config_name="config", overrides=[
                "env=ballnstick", "analysis=ballnstick_h5_rhythm_screening_validation",
                "experiment.seed=1", "env.simulation.obs_win_len=1000",
            ])

    def clone(self):
        return OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))

    def source(self):
        return {"config": self.old, "seed_union": set(range(429351, 429431))}

    def test_design_24_contexts_288_outcomes(self):
        rows = m._validate(self.cfg, self.source())
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(rows)*3*self.cfg.analysis.crossed_design.n_future_continuations, 288)
        self.assertEqual(sorted(set(r["structure_seed"] for r in rows)), [429451, 429452, 429453])
        self.assertEqual(len(set(r["history_seed"] for r in rows)), 3)

    def test_frozen_network_and_estimators_reject_changes(self):
        for path, value in [("analysis.spatial_measurement.phenotype_minimum_excess_log10", .04),
                            ("analysis.states.modulation_depth", .05),
                            ("analysis.actions.amplitude_v_per_m", .4),
                            ("analysis.timeline.stimulation_steps", 8),
                            ("analysis.observation_noise.ar1_coefficient", .9)]:
            cfg = self.clone()
            OmegaConf.update(cfg, path, value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                m._validate(cfg, self.source())

    def test_forced_eligibility_cannot_pass_as_full_experiment(self):
        cfg = self.clone(); cfg.analysis.smoke_force_eligible = True
        with self.assertRaises(ValueError):
            m._validate(cfg, self.source())

    def test_seed_collisions_and_uint32_overflow_rejected(self):
        cfg = self.clone(); cfg.analysis.crossed_design.structure_seed_offset = 429500
        with self.assertRaises(ValueError):
            m._validate(cfg, self.source())
        source = self.source(); source["seed_union"].add(429451)
        with self.assertRaises(ValueError):
            m._validate(self.cfg, source)

    def test_noise_history_identical_futures_independent_all_sensors(self):
        cfg = self.clone()
        cfg.analysis.timeline.baseline_steps = 4
        cfg.analysis.timeline.stimulation_steps = 2
        cfg.analysis.timeline.washout_steps = 1
        context = m._contexts(cfg)[0]
        a, split, seeds_a = m._unit_noise(cfg, context, 0)
        b, _, seeds_b = m._unit_noise(cfg, context, 1)
        np.testing.assert_array_equal(a[:, :split], b[:, :split])
        for sensor in range(3):
            self.assertFalse(np.array_equal(a[sensor, split:], b[sensor, split:]))
            self.assertEqual(seeds_a[sensor, 0], seeds_b[sensor, 0])
            self.assertNotEqual(seeds_a[sensor, 1], seeds_b[sensor, 1])
            self.assertAlmostEqual(np.sqrt(np.mean(a[sensor, :split]**2)), 1.0)
        self.assertFalse(np.array_equal(a[0], a[1]))

    def test_vertex_noise_uses_actual_online_episode_seed(self):
        cfg = self.clone(); context = m._contexts(cfg)[0]
        cfg.experiment.seed = context["trial_seed"]
        for future in range(4):
            self.assertEqual(m._noise_seeds(self.cfg, context, future, 0),
                             m.online._noise_seeds(cfg, context, future))

    def test_ppc_recomputed_at_selected_carrier_with_endpoint_trim(self):
        cfg = self.clone()
        spikes = np.arange(500., 1501., 100.)
        episode = {"simulation": {"outputs_by_epoch": {"stimulation": [{
            "t_start_ms": 0., "t_stop_ms": 2000.,
            "spikes": {"E": {"times_ms": spikes}},
        }]}}}
        self.assertAlmostEqual(m._carrier_ppc(episode, "E", 10., cfg), 1.)
        self.assertLess(m._carrier_ppc(episode, "E", 9., cfg), 0.)

    def test_rate_guardrails_use_active_not_sham_as_tested_value(self):
        text = inspect.getsource(m._metric)
        self.assertIn('_relative_rate_safe(outcome, sham_outcome, cfg)', text)
        limits = self.cfg.analysis.rate_guardrails_hz
        sham = pd.Series({'E_firing_rate_hz': float(limits.E_max)*.95,
                          'I_firing_rate_hz': 8.})
        active = sham.copy(); active['E_firing_rate_hz'] = float(limits.E_max)*1.01
        self.assertFalse(m.montage._relative_rate_safe(active, sham, self.cfg))

    def test_rotation_pairs_share_noise_not_action_outcomes(self):
        rows = m._contexts(self.cfg)
        for i in range(4):
            self.assertEqual(rows[0]["history_seed"], rows[i]["history_seed"])
            self.assertEqual(rows[0]["phase_seed"], rows[i]["phase_seed"])
            for sensor in range(3):
                self.assertEqual(m._noise_seeds(self.cfg, rows[0], 0, sensor),
                                 m._noise_seeds(self.cfg, rows[i], 0, sensor))

    def test_reciprocal_field_projection(self):
        profiles = m.montage._profile_specs(self.cfg)
        rows = m._contexts(self.cfg)
        for row in rows[:4]:
            theta = row["rotation_y_rad"]
            actual = [m.montage._field_projection(row, p) for p in profiles]
            np.testing.assert_allclose(actual, [np.cos(theta), np.cos(theta-np.pi/3)])

    def test_neural_CSD_gain_normalization_rotation_invariant(self):
        model = m.spatial._forward_model(self.cfg)
        fs = 500
        time = np.arange(8*fs) / fs
        local = np.stack([np.zeros_like(time), np.zeros_like(time), np.sin(2*np.pi*9*time)])
        values = []
        for angle in [0, 20, 40, 60]:
            signal = m.spatial._project(local, angle, model)
            estimate, _, _ = m._measure(signal, fs, model, self.cfg)
            self.assertAlmostEqual(estimate["estimated_orientation_deg"], angle)
            values.append(estimate["geometry_normalized_log10_alpha"])
        np.testing.assert_allclose(values, values[0], atol=1e-10)

    def test_screen_interface_excludes_hidden_and_future_inputs(self):
        self.assertEqual(list(inspect.signature(m._screen).parameters), ["baseline", "source", "cfg"])
        text = inspect.getsource(m._screen)
        self.assertIn('hidden_frequency_hz=float("nan")', text)
        self.assertNotIn('["neural"]', text)
        self.assertNotIn('baseline["stimulation"]', text)

    def synthetic(self, flip_future=False, residual=False):
        rows = []
        for structure in range(3):
            for orientation in range(4):
                rule = m.ACTIVE[0] if orientation < 2 else m.ACTIVE[1]
                winner = m.ACTIVE[1 - m.ACTIVE.index(rule)] if residual and orientation in (1, 2) else rule
                for future in range(1, 5):
                    actual = m.ACTIVE[1-m.ACTIVE.index(winner)] if flip_future and future > 2 else winner
                    for action in m.ACTIONS:
                        rows.append(dict(context_id=f"s{structure}_o{orientation}", structure_seed=structure,
                            hidden_frequency_hz=9., orientation_label=str(orientation), rotation_y_rad=orientation*.3,
                            analytical_action=rule, future_index=future, montage_profile=action,
                            loss_log10=.8 if action == m.SHAM else (.2 if action == actual else .4)))
        return pd.DataFrame(rows)

    def test_geometry_success_is_not_residual_H5_opportunity(self):
        _, splits, _, summary = m._response_analysis(self.synthetic(), self.cfg)
        self.assertGreater(summary["analytical_advantage_vs_fixed"]["mean"], .09)
        self.assertAlmostEqual(summary["empirical_oracle_advantage_vs_analytical"], 0)
        self.assertAlmostEqual(splits.selected_advantage_vs_analytical.mean(), 0)

    def test_true_residual_opportunity_replication(self):
        _, splits, structures, summary = m._response_analysis(self.synthetic(residual=True), self.cfg)
        self.assertGreater(summary["empirical_oracle_advantage_vs_analytical"], .09)
        self.assertTrue((structures.selected_advantage_vs_analytical > .09).all())
        self.assertEqual(len(splits), 24)

    def test_future_overfit_does_not_pass_replication(self):
        _, splits, _, _ = m._response_analysis(self.synthetic(flip_future=True), self.cfg)
        self.assertTrue((splits.groupby("split").selected_advantage_vs_fixed.mean() < 0).all())

    def test_structure_weighting_not_pseudoreplication(self):
        table = pd.DataFrame({"structure_seed": [1]*100+[2], "value": [1.]*100+[0.]})
        self.assertEqual(m._structure_mean(table, "value"), .5)

    def test_exact_test_cannot_be_significant_with_three_structures(self):
        result = m._inference([.01, .02, .03], self.cfg)
        self.assertEqual(result["exact_one_sided_p"], .125)
        self.assertFalse(result["confirmatory"])

    def test_zero_enrollment_analysis_is_explicit(self):
        _, _, _, summary = m._response_analysis(pd.DataFrame(), self.cfg)
        self.assertFalse(summary["available"])

    def test_residual_opportunity_does_not_require_analytical_rule_to_win(self):
        analytical, residual = m._readiness({'safe': True}, {'geometry_wins': False},
                                           {'beats_fixed_analytical_and_sham': True}, smoke=False)
        self.assertFalse(analytical)
        self.assertTrue(residual)

    def test_smoke_or_integrity_failure_cannot_permit_policy_development(self):
        for integrity, smoke in [({'safe': True}, True), ({'safe': False}, False), ({}, False)]:
            self.assertEqual(m._readiness(integrity, {'works': True}, {'headroom': True}, smoke=smoke),
                             (False, False))


if __name__ == "__main__":
    unittest.main()
