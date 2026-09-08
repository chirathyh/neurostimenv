"""Focused tests for the H5-O0 orientation--montage opportunity study."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from env.models.neuron.env_online import _parse_optional_dipole_rotation
from experiments.ballnstick_analysis.run_ballnstick_h5_montage_orientation_opportunity import (
    ACTIVE_PROFILES,
    PROFILE_60,
    PROFILE_Z,
    TOPOGRAPHY_FEATURES,
    _contexts,
    _field_projection,
    _head_from_local_rotation,
    _load_sources,
    _opportunity,
    _orientation_loso,
    _orientation_specs,
    _profile_specs,
    _reference_contexts,
    _validate_design,
    _with_orientation_state,
)


class H5MontageOrientationOpportunityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            version_base=None,
            config_dir=str((Path(__file__).parents[1] / "configs").resolve()),
        ):
            cls.cfg = compose(
                config_name="config",
                overrides=[
                    "experiment.seed=1",
                    "env=ballnstick",
                    "analysis=ballnstick_h5_montage_orientation_opportunity",
                    "env.simulation.obs_win_len=1000",
                ],
            )

    def test_source_lock_and_crossed_design(self) -> None:
        sources = _load_sources(self.cfg)
        _validate_design(self.cfg, sources)
        contexts = pd.DataFrame(_contexts(self.cfg))
        self.assertEqual(len(contexts), 12)
        self.assertEqual(contexts.structure_seed.nunique(), 3)
        self.assertEqual(set(contexts.hidden_frequency_hz), {9.0, 11.0})
        self.assertEqual(
            set(contexts.orientation_label),
            {"orientation_0deg", "orientation_60deg"},
        )
        self.assertTrue(
            contexts.groupby("paired_orientation_context_id")
            .orientation_label.nunique().eq(2).all()
        )
        references = pd.DataFrame(_reference_contexts(self.cfg))
        self.assertEqual(len(references), 6)
        self.assertTrue(
            set(contexts.structure_seed).isdisjoint(set(references.structure_seed))
        )
        self.assertTrue(
            set(contexts.structure_seed).isdisjoint(sources["source_seed_union"])
        )

    def test_projection_matrix_has_symmetric_half_dose_mismatch(self) -> None:
        orientations = _orientation_specs(self.cfg)
        profiles = _profile_specs(self.cfg)
        observed = {
            (orientation["orientation_label"], profile["montage_profile"]):
            _field_projection(orientation, profile)
            for orientation in orientations for profile in profiles
        }
        self.assertAlmostEqual(observed[("orientation_0deg", PROFILE_Z)], 1.0)
        self.assertAlmostEqual(observed[("orientation_0deg", PROFILE_60)], 0.5)
        self.assertAlmostEqual(observed[("orientation_60deg", PROFILE_Z)], 0.5)
        self.assertAlmostEqual(observed[("orientation_60deg", PROFILE_60)], 1.0)

    def test_orientation_changes_only_rotation_not_circuit_state(self) -> None:
        contexts = _contexts(self.cfg)
        zero = next(x for x in contexts if x["orientation_label"] == "orientation_0deg")
        tilted = next(
            x for x in contexts
            if x["paired_orientation_context_id"]
            == zero["paired_orientation_context_id"]
            and x["orientation_label"] == "orientation_60deg"
        )
        zero_cfg = _with_orientation_state(self.cfg, zero)
        tilted_cfg = _with_orientation_state(self.cfg, tilted)
        self.assertAlmostEqual(float(zero_cfg.env.network.population.rotation_y_rad), 0.0)
        # Both simulations retain one canonical local circuit. Orientation is
        # applied only by reciprocal head/local field and dipole transforms.
        self.assertAlmostEqual(
            float(tilted_cfg.env.network.population.rotation_y_rad), 0.0
        )
        self.assertTrue(np.allclose(
            np.asarray(zero_cfg.env.eeg.dipole_rotation_matrix, dtype=float),
            np.eye(3),
        ))
        self.assertTrue(np.allclose(
            np.asarray(tilted_cfg.env.eeg.dipole_rotation_matrix, dtype=float),
            _head_from_local_rotation(tilted),
        ))
        for population in ("E", "I"):
            left = zero_cfg.env.network.background[population].rhythm
            right = tilted_cfg.env.network.background[population].rhythm
            self.assertAlmostEqual(float(left.modulation_depth), 0.04)
            self.assertEqual(
                left.shared_modulated_fraction, right.shared_modulated_fraction
            )
            self.assertEqual(left.frequency_hz, right.frequency_hz)
            self.assertEqual(left.phase_diffusion_rad2_per_s,
                             right.phase_diffusion_rad2_per_s)
        self.assertEqual(
            zero_cfg.env.network.connection_probability,
            tilted_cfg.env.network.connection_probability,
        )

    def test_optional_online_eeg_rotation_is_proper_and_backward_compatible(self) -> None:
        self.assertIsNone(_parse_optional_dipole_rotation({}))
        rotation = _head_from_local_rotation({"rotation_y_rad": np.pi / 3})
        self.assertTrue(np.allclose(
            _parse_optional_dipole_rotation({
                "dipole_rotation_matrix": rotation.tolist()
            }),
            rotation,
        ))
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            _parse_optional_dipole_rotation({
                "dipole_rotation_matrix": np.diag([1.0, 1.0, -1.0]).tolist()
            })

    def test_loso_topography_recovers_separable_orientation(self) -> None:
        rows = []
        for structure in range(3):
            for frequency in (9.0, 11.0):
                for label, values in (
                    ("orientation_0deg", [0.70, 0.15, 0.15]),
                    ("orientation_60deg", [0.20, 0.70, 0.10]),
                ):
                    rows.append({
                        "context_id": f"{structure}_{frequency}_{label}",
                        "structure_seed": structure,
                        "orientation_label": label,
                        "eligible": True,
                        **dict(zip(TOPOGRAPHY_FEATURES, values)),
                    })
        predictions, audit = _orientation_loso(pd.DataFrame(rows))
        self.assertEqual(len(predictions), 12)
        self.assertAlmostEqual(audit["LOSO_balanced_accuracy"], 1.0)

    def test_opportunity_detects_geometric_crossover_and_future_replication(self) -> None:
        expected_rows, metric_rows = [], []
        for structure in range(3):
            for frequency in (9.0, 11.0):
                for orientation, matched, topo in (
                    ("orientation_0deg", PROFILE_Z, [0.7, 0.15, 0.15]),
                    ("orientation_60deg", PROFILE_60, [0.2, 0.7, 0.1]),
                ):
                    context_id = f"s{structure}_f{frequency}_{orientation}"
                    losses = {
                        PROFILE_Z: 0.10 if matched == PROFILE_Z else 0.14,
                        PROFILE_60: 0.10 if matched == PROFILE_60 else 0.14,
                    }
                    for profile in ACTIVE_PROFILES:
                        common = {
                            "context_id": context_id,
                            "paired_orientation_context_id": f"s{structure}_f{frequency}",
                            "structure_seed": structure,
                            "hidden_frequency_hz": frequency,
                            "orientation_label": orientation,
                            "rotation_y_rad": 0.0 if matched == PROFILE_Z else np.pi / 3,
                            "matched_profile": matched,
                            "EEG_selected_frequency_hz": frequency,
                            "phase_sensor_index": 0,
                            "phase_sensor_label": "vertex",
                            "montage_profile": profile,
                            "field_projection_fraction": 1.0 if profile == matched else 0.5,
                            "effective_axial_amplitude_v_per_m": (
                                0.2 if profile == matched else 0.1
                            ),
                            "profile_matches_orientation": profile == matched,
                            **dict(zip(TOPOGRAPHY_FEATURES, topo)),
                        }
                        expected_rows.append({
                            **common,
                            "expected_distance_to_B_log10": losses[profile],
                        })
                        for future in range(1, 5):
                            metric_rows.append({
                                **common,
                                "future_index": future,
                                "post_distance_to_orientation_B_log10": losses[profile],
                            })
        context_map, crossover, split, audit = _opportunity(
            pd.DataFrame(expected_rows), pd.DataFrame(metric_rows), self.cfg
        )
        self.assertEqual(context_map.expected_optimal_profile.nunique(), 2)
        self.assertTrue((crossover.matched_advantage_log10 > 0.03).all())
        self.assertGreater(audit["mean_oracle_advantage_over_best_fixed_log10"], 0.01)
        self.assertGreater(audit["future_split_mean_advantage_log10"], 0.01)
        self.assertAlmostEqual(audit["geometric_match_fraction"], 1.0)
        self.assertEqual(split.split_direction.nunique(), 2)


if __name__ == "__main__":
    unittest.main()
