"""Fixed spatial estimator, causal screening, and paired-reference tests."""

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from experiments.ballnstick_analysis import (
    run_ballnstick_h5_spatial_measurement_audit as m,
)


class SpatialMeasurementTests(unittest.TestCase):
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
                    "analysis=ballnstick_h5_spatial_measurement_audit",
                    "experiment.seed=1",
                    "env.simulation.obs_win_len=1000",
                ],
            )
        cls.model = m._forward_model(cls.cfg)

    def clone(self):
        return OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))

    def covariance(self, theta, noise=1.0e-22):
        axis = np.array([np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))])
        g = self.model["leadfield_v_per_nA_um"] @ axis
        return 100 * np.outer(g, g) + noise * np.eye(3), 100 * np.sum(g * g)

    def test_exact_rank_one_plus_isotropic_noise_recovers_angles_and_energy(self):
        normalized = []
        for theta in (0, 20, 40, 60):
            covariance, energy = self.covariance(theta)
            result = m._estimate_spatial(covariance, self.model, self.cfg)[m.PRIMARY]
            self.assertAlmostEqual(result["estimated_orientation_deg"], theta)
            self.assertAlmostEqual(result["global_alpha_estimate_v2"] / energy, 1)
            self.assertAlmostEqual(
                result["estimated_noise_alpha_per_sensor_v2"] / 1.0e-22, 1
            )
            normalized.append(result["geometry_normalized_log10_alpha"])
        self.assertLess(np.ptp(normalized), 1.0e-12)

    def test_estimate_is_invariant_to_signal_polarity_and_absolute_scale(self):
        covariance, _ = self.covariance(40)
        for scale in (1.0e-4, 1.0, 1.0e4):
            result = m._estimate_spatial(scale * covariance, self.model, self.cfg)[
                m.PRIMARY
            ]
            self.assertEqual(result["estimated_orientation_deg"], 40)
            self.assertEqual(result["predicted_profile"], "montage_profile_60deg")

    def test_noise_only_abstains(self):
        result = m._estimate_spatial(np.eye(3) * 1.0e-22, self.model, self.cfg)[
            m.PRIMARY
        ]
        self.assertFalse(result["spatial_accepted"])
        self.assertEqual(result["global_alpha_estimate_v2"], 0)

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            m._estimate_spatial(np.full((3, 3), np.nan), self.model, self.cfg)

    def test_auto_power_noise_bias_is_not_used_as_ground_truth(self):
        covariance, _ = self.covariance(0)
        result = m._estimate_spatial(covariance, self.model, self.cfg)
        self.assertEqual(result[m.PRIMARY]["estimated_orientation_deg"], 0)
        self.assertGreater(result[m.AUTO]["estimated_orientation_deg"], 0)

    def test_pure_measurement_interfaces_exclude_hidden_labels(self):
        self.assertEqual(
            set(m.SCREEN_INPUT_FIELDS),
            {
                "geometry_normalized_log10_alpha",
                "spatial_accepted",
                "carrier_identified",
                "recent_phase_actionable",
            },
        )
        self.assertEqual(
            list(inspect.signature(m._estimate_spatial).parameters),
            ["covariance", "model", "cfg"],
        )
        self.assertEqual(
            list(inspect.signature(m._screen_measurement).parameters),
            ["measurement", "target", "cfg"],
        )
        measurement = dict(
            geometry_normalized_log10_alpha=-20,
            spatial_accepted=True,
            carrier_identified=True,
            recent_phase_actionable=True,
        )
        target = {"baseline_mean_log10": -20.1}
        original = m._screen_measurement(measurement, target, self.cfg)
        mutated = {
            **measurement,
            "true_orientation_deg": 60,
            "condition": "B",
            "hidden_frequency_hz": 11,
        }
        self.assertEqual(original, m._screen_measurement(mutated, target, self.cfg))
        self.assertTrue(original["treatment_eligible"])

    def test_confidence_and_phenotype_failures_map_to_sham(self):
        row = dict(
            geometry_normalized_log10_alpha=-20,
            spatial_accepted=True,
            carrier_identified=True,
            recent_phase_actionable=True,
        )
        for key in (
            "spatial_accepted",
            "carrier_identified",
            "recent_phase_actionable",
        ):
            result = m._screen_measurement(
                {**row, key: False}, {"baseline_mean_log10": -21}, self.cfg
            )
            self.assertFalse(result["treatment_eligible"])
            self.assertEqual(result["fallback_action"], "sham")
            self.assertTrue(result["phenotype_positive"])
        self.assertFalse(
            m._screen_measurement(row, {"baseline_mean_log10": -19}, self.cfg)[
                "phenotype_positive"
            ]
        )

    def test_one_B_reference_per_structure_and_shared_history_A_B(self):
        specs = pd.DataFrame(m._validate_design(self.cfg, {"seed_union": set()}))
        self.assertEqual(len(specs), 15)
        self.assertEqual(specs.stage.eq("calibration").sum(), 3)
        for _, group in specs[specs.stage.eq("evaluation")].groupby("structure_seed"):
            self.assertEqual(group.condition.eq("B").sum(), 1)
            for key in ("history_seed", "phase_seed", "future_drive_seed"):
                self.assertEqual(group[key].nunique(), 1)

    def test_smoke_cannot_replace_full_design(self):
        cfg = self.clone()
        with open_dict(cfg):
            cfg.analysis.measurement_design.evaluation_structures = 1
        with self.assertRaises(ValueError):
            m._validate_design(cfg, {"seed_union": set()})
        with open_dict(cfg):
            cfg.analysis.smoke_test = True
        self.assertEqual(len(m._validate_design(cfg, {"seed_union": set()})), 4)

    def test_seed_overlap_rejected(self):
        structure = m._trajectory_specs(self.cfg)[0]["structure_seed"]
        with self.assertRaises(ValueError):
            m._validate_design(self.cfg, {"seed_union": {structure}})

    def test_source_hash_lock_when_previous_results_are_present(self):
        source = Path(self.cfg.analysis.source_h5o1d.result_dir)
        if not source.exists():
            self.skipTest("Frozen source dataset is not installed on this machine")
        loaded = m._load_source(self.cfg)
        self.assertEqual(len(loaded["hashes"]), 7)
        m._validate_design(self.cfg, loaded)

    def test_unit_noise_history_independent_of_future_seed(self):
        args = dict(
            n_samples=10000, split_sample=7000, history_seed=30, coefficient=0.95
        )
        first = m._ar1_path(**args, future_seed=40)
        second = m._ar1_path(**args, future_seed=41)
        np.testing.assert_array_equal(first[:7000], second[:7000])
        self.assertFalse(np.array_equal(first[7000:], second[7000:]))
        self.assertAlmostEqual(np.mean(first[:7000] ** 2), 1)

    def test_reference_calibration_weights_structures_not_repeated_views(self):
        rows = [
            dict(
                signal_view="observed",
                estimator=m.PRIMARY,
                structure_seed=1,
                geometry_normalized_log10_alpha=2.0,
                outcome_geometry_normalized_log10_alpha=4.0,
            )
        ] * 10
        rows += [
            dict(
                signal_view="observed",
                estimator=m.PRIMARY,
                structure_seed=2,
                geometry_normalized_log10_alpha=4.0,
                outcome_geometry_normalized_log10_alpha=6.0,
            )
        ]
        target = m._calibrate_targets(pd.DataFrame(rows))[f"observed/{m.PRIMARY}"]
        self.assertEqual(target["baseline_mean_log10"], 3)
        self.assertEqual(target["outcome_mean_log10"], 5)

    def test_known_dipole_projection_and_psd_energy(self):
        fs = 250.0
        t = np.arange(4 * int(fs)) / fs
        dipole = np.stack(
            [np.zeros_like(t), np.zeros_like(t), np.sin(2 * np.pi * 9 * t)]
        )
        y = m._project(dipole, 40, self.model)
        frequencies, spectrum, covariance = m._cross_spectrum(y, fs, self.cfg)
        self.assertEqual(frequencies[np.argmax(spectrum[0, 0].real)], 9)
        self.assertGreater(np.linalg.eigvalsh(covariance)[-1], 0)
        result = m._estimate_spatial(covariance, self.model, self.cfg)[m.PRIMARY]
        self.assertEqual(result["estimated_orientation_deg"], 40)
        self.assertTrue(result["spatial_accepted"])

    def test_carrier_does_not_need_hidden_frequency(self):
        fs = 250.0
        t = np.arange(4 * int(fs)) / fs
        for frequency in (9.0, 11.0):
            result = m._carrier_and_phase(
                1.0e-9 * np.sin(2 * np.pi * frequency * t), fs, self.cfg
            )
            self.assertEqual(result["EEG_selected_frequency_hz"], frequency)
            self.assertTrue(result["carrier_identified"])

    def test_accepted_balanced_accuracy_requires_both_classes(self):
        group = pd.DataFrame(
            {"true_matched_profile": ["montage_profile_z"], "profile_correct": [True]}
        )
        self.assertTrue(np.isnan(m._balanced_accuracy(group)))

    def test_plot_orientation_matching_tolerates_floating_point_roundoff(self):
        rows, spectra, structures = [], [], []
        for view in ("observed", "neural_audit"):
            for method in m.METHODS:
                structures.append(
                    dict(
                        signal_view=view,
                        estimator=method,
                        structure_seed=1,
                        balanced_accuracy=1.0,
                        A_screen_sensitivity=1.0,
                        B_screen_specificity=1.0,
                    )
                )
                for angle in (0.0, np.degrees(np.pi / 3)):
                    for condition, carrier in (("B", 9), ("A", 9), ("A", 11)):
                        rows.append(
                            dict(
                                signal_view=view,
                                estimator=method,
                                structure_seed=1,
                                true_orientation_deg=angle,
                                estimated_orientation_deg=angle,
                                condition=condition,
                                hidden_frequency_hz=carrier,
                                absolute_angle_error_deg=0.0,
                                alpha_excess_over_B_log10=(
                                    0.1 if condition == "A" else 0.0
                                ),
                                outcome_geometry_normalized_log10_alpha=-20.0,
                            )
                        )
                        if method == m.PRIMARY:
                            for sensor in range(3):
                                for frequency in (8.0, 9.0, 10.0, 11.0, 12.0):
                                    spectra.append(
                                        dict(
                                            stage="evaluation",
                                            signal_view=view,
                                            condition=condition,
                                            hidden_frequency_hz=carrier,
                                            true_orientation_deg=angle,
                                            sensor_index=sensor,
                                            frequency_hz=frequency,
                                            PSD_v2_per_hz=1.0e-22,
                                        )
                                    )

        def check_figure(fig, root, name):
            if name == "figure_07_side_sensor_noise_PSD":
                self.assertEqual(len(fig.axes), 6)
                for axis in fig.axes:
                    self.assertEqual(len(axis.lines), 4)
                    for line in axis.lines:
                        self.assertGreater(len(line.get_xdata()), 0)
            m.plt.close(fig)

        with patch.object(m, "_save_figure", side_effect=check_figure) as save:
            m._plots(
                Path("unused"),
                pd.DataFrame(rows),
                pd.DataFrame(rows),
                pd.DataFrame(spectra),
                pd.DataFrame(structures),
                self.cfg,
            )
        self.assertEqual(save.call_count, 7)


if __name__ == "__main__":
    unittest.main()
