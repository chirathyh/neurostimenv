"""B-only calibration, causal screening, independent-unit inference and design."""

import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from experiments.ballnstick_analysis import (
    run_ballnstick_h5_rhythm_screening_validation as m,
)


class RhythmScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        ):
            cls.cfg = compose(
                config_name="config",
                overrides=[
                    "env=ballnstick",
                    "analysis=ballnstick_h5_rhythm_screening_validation",
                    "experiment.seed=1",
                    "env.simulation.obs_win_len=1000",
                ],
            )
            cls.source_cfg = compose(
                config_name="config",
                overrides=[
                    "env=ballnstick",
                    "analysis=ballnstick_h5_spatial_measurement_audit",
                    "experiment.seed=1",
                    "env.simulation.obs_win_len=1000",
                ],
            )

    def clone(self):
        return OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))

    def source(self):
        return {"config": self.source_cfg, "seed_union": set()}

    def calibration(self, n=19):
        return pd.DataFrame(
            [
                {
                    "stage": "calibration",
                    "condition": "B",
                    "signal_view": view,
                    "estimator": estimator,
                    "structure_seed": 100 + seed,
                    "noise_repeat": repeat,
                    "true_orientation_deg": angle,
                    m.SCORE: seed / 10 + repeat / 100 if view == "observed" else 10000,
                    "outcome_geometry_normalized_log10_alpha": -20.0,
                }
                for seed in range(n)
                for repeat in range(3)
                for angle in (0, 20, 40, 60)
                for view in ("observed", "neural_audit")
                for estimator in m.spatial.METHODS
            ]
        )

    def test_full_design_109_episodes_and_one_matched_history_per_structure(self):
        specs = pd.DataFrame(m._validate_design(self.cfg, self.source()))
        self.assertEqual(len(specs), 109)
        self.assertEqual(specs.stage.eq("calibration").sum(), 19)
        cal = specs[specs.stage.eq("calibration")]
        validation = specs[specs.stage.eq("evaluation")]
        self.assertTrue(cal.condition.eq("B").all())
        self.assertEqual(validation.structure_seed.nunique(), 30)
        self.assertTrue(set(cal.structure_seed).isdisjoint(validation.structure_seed))
        for _, g in validation.groupby("structure_seed"):
            self.assertEqual(list(g.condition), ["B", "A", "A"])
            for key in ("history_seed", "phase_seed", "future_drive_seed"):
                self.assertEqual(g[key].nunique(), 1)

    def test_source_overlap_is_rejected(self):
        source = self.source()
        source["seed_union"] = {429351}
        with self.assertRaisesRegex(ValueError, "Seed namespaces"):
            m._validate_design(self.cfg, source)

    def test_physical_estimator_and_alpha_prerequisite_cannot_be_retuned(self):
        for key, value in (
            ("env.network.celsius", 36.5),
            ("analysis.spatial_measurement.phenotype_minimum_excess_log10", 0.01),
            ("analysis.multitaper.minimum_residual_evidence_db", 10.0),
            ("analysis.states.modulation_depth", 0.08),
        ):
            cfg = self.clone()
            OmegaConf.update(cfg, key, value)
            with self.assertRaisesRegex(ValueError, "Frozen source setting"):
                m._validate_design(cfg, self.source())

    def test_smoke_is_four_episodes_but_full_size_cannot_be_lowered(self):
        cfg = self.clone()
        cfg.analysis.measurement_design.evaluation_structures = 1
        with self.assertRaisesRegex(ValueError, "full design"):
            m._validate_design(cfg, self.source())
        cfg.analysis.smoke_test = True
        self.assertEqual(len(m._validate_design(cfg, self.source())), 4)

    def test_cluster_rank_cutoff_is_max_of_19_structure_maxima(self):
        rule = m._calibrate_null(self.calibration(), self.cfg)
        self.assertEqual(rule["calibration_structures"], 19)
        self.assertEqual(rule["order_statistic_rank"], 19)
        self.assertAlmostEqual(rule["cutoff_db"], 1.82)
        self.assertAlmostEqual(rule["marginal_rank_bound"], 0.05)
        # Noise views and rotations do not increase the calibration sample size.
        duplicated = pd.concat([self.calibration()] * 3, ignore_index=True)
        self.assertEqual(rule, m._calibrate_null(duplicated, self.cfg))

    def test_calibration_excludes_A_evaluation_and_ideal_score_information(self):
        for key, value in (("condition", "A"), ("stage", "evaluation")):
            rows = self.calibration()
            rows.loc[0, key] = value
            with self.assertRaisesRegex(ValueError, "B-only"):
                m._calibrate_null(rows, self.cfg)
        original = self.calibration()
        modified = original.copy()
        modified.loc[modified.signal_view.eq("neural_audit"), m.SCORE] = -1.0e10
        self.assertEqual(
            m._calibrate_null(original, self.cfg), m._calibrate_null(modified, self.cfg)
        )

    def test_nonfinite_null_data_is_rejected_not_silently_ignored_by_max(self):
        rows = self.calibration()
        index = rows[
            rows.signal_view.eq("observed") & rows.estimator.eq(m.spatial.PRIMARY)
        ].index[0]
        rows.loc[index, m.SCORE] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m._calibrate_null(rows, self.cfg)

    def test_insufficient_null_calibration_abstains_instead_of_loosening_bound(self):
        rule = m._calibrate_null(self.calibration(1), self.cfg)
        self.assertIsNone(rule["cutoff_db"])
        self.assertTrue(rule["abstain_all_for_insufficient_calibration"])
        self.assertFalse(
            m._screen(self.measurement(1.0e6), self.target(), rule, self.cfg)[
                "treatment_eligible"
            ]
        )

    def measurement(self, score=3.0):
        return {
            "geometry_normalized_log10_alpha": -20.0,
            "spatial_accepted": True,
            "carrier_identified": True,
            "recent_phase_actionable": True,
            m.SCORE: score,
        }

    def target(self):
        return {"baseline_mean_log10": -20.2}

    def test_strict_cutoff_ties_are_negative_and_alpha_prerequisite_is_retained(self):
        rule = m._calibrate_null(self.calibration(), self.cfg)
        self.assertFalse(
            m._screen(
                self.measurement(rule["cutoff_db"]), self.target(), rule, self.cfg
            )["rhythm_present"]
        )
        self.assertTrue(
            m._screen(self.measurement(), self.target(), rule, self.cfg)[
                "treatment_eligible"
            ]
        )
        low_alpha = {**self.measurement(), "geometry_normalized_log10_alpha": -21.0}
        result = m._screen(low_alpha, self.target(), rule, self.cfg)
        self.assertTrue(result["rhythm_present"])
        self.assertFalse(result["phenotype_positive"])
        self.assertEqual(result["fallback_action"], "sham")

    def test_confidence_abstention_does_not_hide_phenotype_false_positives(self):
        rule = m._calibrate_null(self.calibration(), self.cfg)
        for key in (
            "carrier_identified",
            "spatial_accepted",
            "recent_phase_actionable",
        ):
            result = m._screen(
                {**self.measurement(), key: False}, self.target(), rule, self.cfg
            )
            self.assertTrue(result["phenotype_positive"])
            self.assertFalse(result["treatment_eligible"])

    def test_screen_does_not_depend_on_hidden_labels_or_future_measurements(self):
        self.assertEqual(set(m.INPUTS), set(m.spatial.SCREEN_INPUT_FIELDS) | {m.SCORE})
        self.assertEqual(
            list(inspect.signature(m._screen).parameters),
            ["measurement", "target", "rule", "cfg"],
        )
        rule = m._calibrate_null(self.calibration(), self.cfg)
        original = m._screen(self.measurement(), self.target(), rule, self.cfg)
        other = {
            **self.measurement(),
            "condition": "B",
            "hidden_frequency_hz": 11.0,
            "true_orientation_deg": 60.0,
            "outcome_geometry_normalized_log10_alpha": 100.0,
        }
        self.assertEqual(original, m._screen(other, self.target(), rule, self.cfg))

    def test_exact_power_uses_30_independent_structures_and_28_clean_cutpoint(self):
        power = m._power_design(self.cfg)
        self.assertEqual(power["planned_structures"], 30)
        self.assertEqual(power["critical_clean_structure_count"], 28)
        self.assertAlmostEqual(power["anticipated_exact_power"], 0.81217881314696)
        passed = m._binomial_inference([True] * 28 + [False] * 2, self.cfg)
        failed = m._binomial_inference([True] * 27 + [False] * 3, self.cfg)
        self.assertLess(passed["one_sided_exact_binomial_p"], 0.05)
        self.assertGreater(passed["one_sided_95_lower_bound"], 0.8)
        self.assertGreater(failed["one_sided_exact_binomial_p"], 0.05)
        with self.assertRaises(ValueError):
            m._binomial_inference([], self.cfg)

    def test_new_summary_reject_all_cannot_pass_and_legacy_plotting_needs_no_exact_enumeration(
        self,
    ):
        rows = []
        trajectory_rows = []
        for seed in range(30):
            for condition, frequency in (("B", 9), ("A", 9), ("A", 11)):
                trajectory_rows.append(
                    {
                        "stage": "evaluation",
                        "condition": condition,
                        "structure_seed": seed,
                        "applied_amplitude_v_per_m": 0.0,
                        "final_extracellular_residual_mV": 0.0,
                        "E_firing_rate_hz": 4.0,
                        "I_firing_rate_hz": 8.0,
                    }
                )
                for method in m.spatial.METHODS:
                    for view in ("observed", "neural_audit"):
                        for angle in (0, 20, 40, 60):
                            for repeat in range(3):
                                rows.append(
                                    {
                                        "structure_seed": seed,
                                        "condition": condition,
                                        "hidden_frequency_hz": frequency,
                                        "estimator": method,
                                        "signal_view": view,
                                        "true_orientation_deg": angle,
                                        "noise_repeat": repeat,
                                        "true_matched_profile": (
                                            "montage_profile_0deg"
                                            if angle <= 30
                                            else "montage_profile_60deg"
                                        ),
                                        "predicted_profile": (
                                            "montage_profile_0deg"
                                            if angle <= 30
                                            else "montage_profile_60deg"
                                        ),
                                        "legacy_alpha_positive": True,
                                        "legacy_treatment_eligible": True,
                                        "phenotype_positive": False,
                                        "treatment_eligible": False,
                                        "absolute_angle_error_deg": 0.0,
                                        "profile_correct": True,
                                        "carrier_identified": True,
                                        "carrier_correct": True,
                                        "recent_phase_actionable": True,
                                        "geometry_normalized_log10_alpha": -20.0,
                                        "outcome_geometry_normalized_log10_alpha": -20.0,
                                        m.SCORE: 1.0,
                                        "achieved_vertex_noise_RMS_fraction": 0.25,
                                    }
                                )
        cal = self.calibration()
        rule = m._calibrate_null(cal, self.cfg)
        structures, _, inference, checks = m._summarize(
            pd.DataFrame(rows), cal, pd.DataFrame(trajectory_rows), {}, rule, self.cfg
        )
        self.assertTrue(checks["primary_structure_specificity_test_passes"])
        self.assertFalse(checks["A_sensitivity"])
        self.assertFalse(checks["A_treatment_coverage_in_both_carriers"])
        self.assertFalse(all(checks.values()))
        self.assertEqual(inference["independent_structure_count"], 30)
        plots = m._legacy_plot_metrics(structures)
        self.assertEqual(len(plots), 30 * 2 * 2)
        self.assertTrue(plots.balanced_accuracy.eq(1.0).all())
        self.assertIn("A_screen_sensitivity", plots)
        # One entirely abstained structure cannot be silently dropped from
        # the accepted-carrier accuracy average.
        missing = pd.DataFrame(rows)
        missing.loc[missing.structure_seed.eq(0), "carrier_identified"] = False
        _, _, _, incomplete_checks = m._summarize(
            missing, cal, pd.DataFrame(trajectory_rows), {}, rule, self.cfg
        )
        self.assertFalse(incomplete_checks["accepted_carrier_accuracy"])


if __name__ == "__main__":
    unittest.main()
