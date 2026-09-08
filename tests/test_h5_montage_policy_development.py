"""Focused tests for H5-O1D noisy-EEG montage policy development."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from experiments.ballnstick_analysis.run_ballnstick_h5_montage_orientation_opportunity import (
    PROFILE_60,
    PROFILE_Z,
    _reference_contexts,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_montage_policy_development import (
    POLICY_FEATURES,
    PRIMARY_SPLIT,
    _base_contexts,
    _fit_ridge,
    _load_sources,
    _policy_crossvalidation,
    _predict_ridge,
    _run_contexts,
    _side_noise_seeds,
    _validate_design,
)


class H5MontagePolicyDevelopmentTests(unittest.TestCase):
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
                    "analysis=ballnstick_h5_montage_policy_development",
                    "env.simulation.obs_win_len=1000",
                ],
            )

    def test_source_lock_and_complete_disjoint_design(self) -> None:
        sources = _load_sources(self.cfg)
        _validate_design(self.cfg, sources)
        contexts = pd.DataFrame(_base_contexts(
            self.cfg, apply_smoke_limit=False
        ))
        self.assertEqual(len(contexts), 32)
        self.assertEqual(contexts.structure_seed.nunique(), 4)
        self.assertEqual(contexts.hidden_frequency_hz.nunique(), 2)
        self.assertEqual(contexts.orientation_label.nunique(), 4)
        self.assertEqual(contexts.matched_profile.nunique(), 2)
        references = pd.DataFrame(_reference_contexts(self.cfg))
        self.assertEqual(len(references), 12)
        self.assertTrue(set(contexts.structure_seed).isdisjoint(
            set(references.structure_seed)
        ))
        self.assertTrue(set(contexts.structure_seed).isdisjoint(
            sources["source_seed_union"]
        ))

    def test_smoke_selection_spans_structures_and_actions(self) -> None:
        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
        with open_dict(cfg):
            cfg.analysis.smoke_test = True
            cfg.analysis.smoke_context_limit = 4
        contexts = pd.DataFrame(_run_contexts(cfg))
        self.assertEqual(len(contexts), 4)
        self.assertEqual(contexts.structure_seed.nunique(), 2)
        self.assertEqual(set(contexts.matched_profile), {PROFILE_Z, PROFILE_60})

    def test_side_sensor_noise_seeds_are_paired_and_disjoint(self) -> None:
        context = _base_contexts(self.cfg, apply_smoke_limit=False)[0]
        sensor_one = _side_noise_seeds(self.cfg, context, 1)
        sensor_two = _side_noise_seeds(self.cfg, context, 2)
        self.assertEqual(sensor_one, _side_noise_seeds(self.cfg, context, 1))
        self.assertTrue(set(sensor_one).isdisjoint(sensor_two))

    def test_ridge_model_recovers_monotone_paired_effect(self) -> None:
        table = pd.DataFrame({
            "topography_vertex": [0.98, 0.90, 0.80, 0.70],
            "topography_right_minus_left": [0.00, 0.03, 0.08, 0.12],
            "paired_effect_z_minus_60_log10": [-0.04, -0.02, 0.02, 0.04],
        })
        model = _fit_ridge(
            table, response="paired_effect_z_minus_60_log10", penalty=1.0
        )
        prediction = _predict_ridge(model, table)
        self.assertTrue(np.all(prediction[:2] < 0))
        self.assertTrue(np.all(prediction[2:] > 0))
        self.assertEqual(model["feature_names"], POLICY_FEATURES)

    def test_crossvalidation_uses_only_context_and_heldout_outcomes(self) -> None:
        screening_rows, metric_rows = [], []
        for structure in range(4):
            for frequency in (9.0, 11.0):
                for angle, right_left, matched in (
                    (0, 0.00, PROFILE_Z),
                    (20, 0.03, PROFILE_Z),
                    (40, 0.08, PROFILE_60),
                    (60, 0.12, PROFILE_60),
                ):
                    context_id = f"s{structure}_f{frequency}_o{angle}"
                    screening_rows.append({
                        "context_id": context_id,
                        "structure_seed": structure,
                        "hidden_frequency_hz": frequency,
                        "orientation_label": f"orientation_{angle}deg",
                        "rotation_y_rad": np.radians(angle),
                        "matched_profile": matched,
                        "topography_vertex": 0.98 - 0.004 * angle,
                        "topography_right_minus_left": right_left,
                        "eligible": True,
                    })
                    for future in range(1, 5):
                        for profile in (PROFILE_Z, PROFILE_60):
                            matched_loss = 0.10 + 0.001 * structure
                            loss = matched_loss if profile == matched else matched_loss + 0.04
                            metric_rows.append({
                                "context_id": context_id,
                                "structure_seed": structure,
                                "future_index": future,
                                "montage_profile": profile,
                                "post_distance_to_orientation_B_log10": loss,
                            })
        evaluation, folds, frozen = _policy_crossvalidation(
            pd.DataFrame(metric_rows), pd.DataFrame(screening_rows), self.cfg
        )
        primary = evaluation[evaluation.split_direction.eq(PRIMARY_SPLIT)]
        self.assertEqual(primary.heldout_structure_seed.nunique(), 4)
        self.assertEqual(set(primary.learned_action), {PROFILE_Z, PROFILE_60})
        self.assertGreater(primary.learned_advantage_over_fixed_log10.mean(), 0.0)
        self.assertTrue(primary.policy_uses_only_predecision_observed_EEG.all())
        self.assertTrue(primary.hidden_labels_excluded_from_policy.all())
        self.assertEqual(len(folds["fold_models"]), 8)
        self.assertEqual(frozen["feature_names"], POLICY_FEATURES)


if __name__ == "__main__":
    unittest.main()
