"""Focused tests for the H5-P1 full-information response-mapping study."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (
    CONSERVATIVE,
    RESPONSIVE,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_response_mapping import (
    P1_CONTEXT_FEATURES,
    _bh_fdr,
    _context_specs,
    _feature_response_associations,
    _load_sources,
    _response_map,
    _run_context_specs,
    _validate_design,
)


class H5ResponseMappingTests(unittest.TestCase):
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
                    "analysis=ballnstick_h5_response_mapping",
                    "env.simulation.obs_win_len=1000",
                ],
            )

    def test_sources_hash_lock_and_full_design_is_disjoint(self) -> None:
        sources = _load_sources(self.cfg)
        _validate_design(self.cfg, sources)
        contexts = _context_specs(self.cfg)
        self.assertEqual(len(contexts), 48)
        self.assertEqual(len({row["structure_seed"] for row in contexts}), 6)
        self.assertEqual(
            set(row["hidden_frequency_hz"] for row in contexts), {9.0, 11.0}
        )
        self.assertEqual(
            set(row["diffusion_rad2_per_s"] for row in contexts), {0.5, 2.0}
        )
        self.assertEqual(
            set(row["shared_modulated_fraction"] for row in contexts), {0.5, 1.0}
        )
        new_seeds = {
            int(row[column])
            for row in contexts
            for column in (
                "structure_seed", "history_seed", "phase_seed", "trial_seed"
            )
        }
        self.assertTrue(new_seeds.isdisjoint(sources["source_seed_union"]))

    def test_smoke_selection_prioritizes_both_carriers(self) -> None:
        cfg = self.cfg.copy()
        with open_dict(cfg):
            cfg.analysis.smoke_test = True
            cfg.analysis.smoke_context_limit = 2
        selected = _run_context_specs(cfg)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {row["hidden_frequency_hz"] for row in selected}, {9.0, 11.0}
        )

    @staticmethod
    def _synthetic_expected_and_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
        expected_rows = []
        metric_rows = []
        for context_index, responsive_advantage in enumerate((-0.04, 0.05)):
            context_id = f"context_{context_index}"
            base = {
                "context_id": context_id,
                "paired_shared_drive_context_id": f"pair_{context_index}",
                "structure_seed": 700000 + context_index,
                "hidden_frequency_hz": 9.0 + 2.0 * context_index,
                "label": "low_diffusion" if context_index == 0 else "high_diffusion",
                "diffusion_rad2_per_s": 0.5 + 1.5 * context_index,
                "shared_drive_label": (
                    "partial_shared_drive" if context_index == 0
                    else "full_shared_drive"
                ),
                "shared_modulated_fraction": 0.5 + 0.5 * context_index,
                "EEG_selected_frequency_hz": 9.0 + 2.0 * context_index,
                **{feature: 0.1 + context_index for feature in P1_CONTEXT_FEATURES},
            }
            conservative = 0.20
            responsive = conservative - responsive_advantage
            for mode, distance in (
                ("sham", 0.30),
                (CONSERVATIVE, conservative),
                (RESPONSIVE, responsive),
            ):
                expected_rows.append({
                    **base,
                    "controller_mode": mode,
                    "n_futures": 4,
                    "expected_post_distance_to_B_log10": distance,
                    "future_sd_post_distance_log10": 0.01,
                    "mean_abs_common_phase_error_rad": 0.2,
                })
            for future in range(4):
                for mode, distance in (
                    (CONSERVATIVE, conservative),
                    (RESPONSIVE, responsive),
                ):
                    metric_rows.append({
                        "context_id": context_id,
                        "future_index": future + 1,
                        "controller_mode": mode,
                        "post_distance_to_B_log10": distance + 0.001 * future,
                    })
        return pd.DataFrame(expected_rows), pd.DataFrame(metric_rows)

    def test_response_map_uses_expected_paired_outcomes(self) -> None:
        expected, metrics = self._synthetic_expected_and_metrics()
        action_map, structure, opportunity = _response_map(
            expected, metrics, self.cfg
        )
        self.assertEqual(
            set(action_map.expected_optimal_profile), {CONSERVATIVE, RESPONSIVE}
        )
        self.assertTrue(
            np.allclose(
                action_map.realized_optimal_profile_agreement_fraction, 1.0
            )
        )
        self.assertEqual(len(structure), 2)
        self.assertGreater(
            opportunity["mean_oracle_advantage_over_best_fixed_log10"], 0.0
        )
        self.assertTrue(opportunity["oracle_is_post_hoc_full_information_and_not_deployable"])

    def test_structure_preserving_feature_mapping_detects_known_signal(self) -> None:
        rng = np.random.default_rng(9123)
        rows = []
        for structure in range(6):
            for condition in range(8):
                driver = float(condition)
                row = {
                    "structure_seed": 800000 + structure,
                    "responsive_advantage_over_conservative_log10": (
                        0.02 * driver + 0.001 * structure
                    ),
                }
                for feature in P1_CONTEXT_FEATURES:
                    row[feature] = (
                        driver if feature == "context_C1" else float(rng.normal())
                    )
                rows.append(row)
        cfg = self.cfg.copy()
        with open_dict(cfg):
            cfg.analysis.response_mapping.association_permutations = 199
        associations, audit = _feature_response_associations(
            pd.DataFrame(rows), cfg
        )
        c1 = associations[associations.feature.eq("context_C1")].iloc[0]
        self.assertGreater(c1.structure_centered_spearman_rho, 0.95)
        self.assertTrue(bool(c1.passes_exploratory_response_association_gate))
        self.assertEqual(audit["selected_candidate_response_feature"], "context_C1")

    def test_policy_features_exclude_hidden_state_and_absolute_phase(self) -> None:
        forbidden = ("hidden", "spike", "diffusion_rad2", "shared_modulated", "phase_rad")
        self.assertTrue(P1_CONTEXT_FEATURES)
        for feature in P1_CONTEXT_FEATURES:
            self.assertFalse(any(token in feature for token in forbidden), feature)

    def test_bh_fdr_is_monotone_in_ranked_p_values(self) -> None:
        p_values = np.asarray([0.001, 0.02, 0.04, 0.50])
        q_values = _bh_fdr(p_values)
        order = np.argsort(p_values)
        self.assertTrue(np.all(np.diff(q_values[order]) >= -1.0e-12))
        self.assertTrue(np.all((q_values >= p_values) & (q_values <= 1.0)))


if __name__ == "__main__":
    unittest.main()
