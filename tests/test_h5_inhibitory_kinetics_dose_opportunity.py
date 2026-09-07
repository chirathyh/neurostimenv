"""Focused tests for H5-K0 inhibitory-kinetics dose opportunity."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from experiments.ballnstick_analysis.run_ballnstick_h5_dose_opportunity import (
    _action_role,
    _contexts,
    _kinetics_baseline_pairs,
    _kinetics_metadata,
    _load_sources,
    _opportunity,
    _validate_design,
    _with_susceptibility_state,
)


class H5InhibitoryKineticsDoseOpportunityTests(unittest.TestCase):
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
                    "analysis=ballnstick_h5_inhibitory_kinetics_dose_opportunity",
                    "env.simulation.obs_win_len=1000",
                ],
            )

    def test_sources_are_locked_and_crossed_design_is_disjoint(self) -> None:
        sources = _load_sources(self.cfg)
        _validate_design(self.cfg, sources)
        contexts = _contexts(self.cfg)
        self.assertEqual(len(contexts), 12)
        self.assertEqual(len({row["structure_seed"] for row in contexts}), 3)
        self.assertEqual(
            {row["hidden_frequency_hz"] for row in contexts}, {9.0, 11.0}
        )
        self.assertEqual(
            {row["i_to_e_tau2_multiplier"] for row in contexts}, {0.8, 1.2}
        )
        pairs = pd.DataFrame(contexts).groupby(
            "paired_susceptibility_context_id"
        )
        self.assertTrue(pairs.size().eq(2).all())
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_group_index",
        ):
            self.assertTrue(pairs[column].nunique().eq(1).all())
        new_seeds = {
            int(row[column])
            for row in contexts
            for column in (
                "structure_seed", "history_seed", "phase_seed", "trial_seed"
            )
        }
        self.assertTrue(new_seeds.isdisjoint(sources["source_seed_union"]))

    def test_state_changes_only_declared_i_to_e_controls(self) -> None:
        contexts = _contexts(self.cfg)
        short = next(
            row for row in contexts if row["i_to_e_tau2_label"] == "short_decay"
        )
        state = _with_susceptibility_state(self.cfg, short)
        self.assertAlmostEqual(
            float(state.env.network.recurrent.i_to_e_tau2_multiplier), 0.8
        )
        self.assertTrue(
            bool(state.env.network.recurrent.i_to_e_preserve_conductance_time_area)
        )
        self.assertAlmostEqual(
            float(state.env.network.synapse_kinetics.inhibitory.tau2_ms), 9.0
        )
        self.assertAlmostEqual(
            float(state.env.network.background.E.rhythm.shared_modulated_fraction),
            1.0,
        )
        self.assertAlmostEqual(
            float(state.env.network.background.I.rhythm.shared_modulated_fraction),
            1.0,
        )
        self.assertEqual(
            _action_role(state),
            "H5_K0_I_to_E_kinetics_fast_controller_dose_map",
        )

    def test_resolved_kinetics_preserve_area_and_i_to_i_decay(self) -> None:
        for context in _contexts(self.cfg):
            metadata = _kinetics_metadata(context, self.cfg)
            self.assertAlmostEqual(
                metadata["i_to_e_normalized_conductance_time_area_ratio"], 1.0
            )
            self.assertAlmostEqual(metadata["i_to_i_tau2_ms"], 9.0)
        short = _kinetics_metadata(_contexts(self.cfg)[0], self.cfg)
        long = _kinetics_metadata(_contexts(self.cfg)[1], self.cfg)
        self.assertAlmostEqual(short["i_to_e_tau2_ms"], 7.2)
        self.assertAlmostEqual(long["i_to_e_tau2_ms"], 10.8)
        self.assertGreater(short["i_to_e_peak_weight_scale"], 1.0)
        self.assertLess(long["i_to_e_peak_weight_scale"], 1.0)

    def test_baseline_pair_audit_detects_unmatched_rates(self) -> None:
        rows = []
        for label, e_rate in (("short_decay", 3.0), ("long_decay", 4.0)):
            rows.append({
                "paired_susceptibility_context_id": "pair",
                "structure_seed": 1,
                "hidden_frequency_hz": 9.0,
                "i_to_e_tau2_label": label,
                "context_alpha_excess_log10": 0.2,
                "baseline_E_firing_rate_hz": e_rate,
                "baseline_I_firing_rate_hz": 8.0,
                "alpha_phenotype_present": True,
                "baseline_rates_safe": True,
            })
        pairs, audit = _kinetics_baseline_pairs(pd.DataFrame(rows))
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(audit["maximum_abs_E_rate_difference_hz"], 1.0)
        self.assertTrue(audit["all_pairs_retain_alpha_phenotype"])
        self.assertTrue(np.isfinite(pairs.select_dtypes(include=np.number)).all().all())

    def test_opportunity_requires_a_practical_state_level_crossover(self) -> None:
        expected_rows, metric_rows = [], []
        for structure in range(3):
            for label in ("short_decay", "long_decay"):
                context_id = f"s{structure}_{label}"
                losses = (
                    {0.1: 0.10, 0.2: 0.12, 0.4: 0.30}
                    if label == "short_decay"
                    else {0.1: 0.14, 0.2: 0.10, 0.4: 0.30}
                )
                common = {
                    "context_id": context_id,
                    "paired_shared_drive_context_id": f"s{structure}",
                    "structure_seed": 100 + structure,
                    "hidden_frequency_hz": 9.0,
                    "shared_drive_label": "full_shared_drive",
                    "shared_modulated_fraction": 1.0,
                    "i_to_e_tau2_label": label,
                    "i_to_e_tau2_multiplier": (
                        0.8 if label == "short_decay" else 1.2
                    ),
                    "i_to_e_tau2_ms": 7.2 if label == "short_decay" else 10.8,
                    "i_to_e_peak_weight_scale": 1.0,
                    "i_to_e_normalized_conductance_time_area_ratio": 1.0,
                    "i_to_i_tau2_ms": 9.0,
                    "carrier_maximum_residual_evidence_db": 3.0,
                    "context_C1_abs": 0.5,
                    "context_spectral_concentration": 0.5,
                    "context_spectral_rms_width_hz": 1.0,
                    "context_alpha_excess_log10": 0.2,
                    "recent_resultant_to_rms": 0.1,
                }
                for dose, loss in losses.items():
                    expected_rows.append({
                        **common,
                        "dose_v_per_m": dose,
                        "expected_post_distance_to_B_log10": loss,
                    })
                    for future in range(1, 5):
                        metric_rows.append({
                            "context_id": context_id,
                            "structure_seed": 100 + structure,
                            "dose_v_per_m": dose,
                            "future_index": future,
                            "post_distance_to_B_log10": loss,
                        })
        context_map, _, audit = _opportunity(
            pd.DataFrame(expected_rows), pd.DataFrame(metric_rows), self.cfg
        )
        self.assertEqual(audit["best_fixed_active_dose_v_per_m"], 0.2)
        self.assertEqual(audit["kinetics_state_preferred_dose_count"], 2)
        self.assertAlmostEqual(
            audit["kinetics_state_crossover_minimum_margin_log10"], 0.02
        )
        self.assertEqual(context_map.expected_optimal_dose_v_per_m.nunique(), 2)


if __name__ == "__main__":
    unittest.main()
