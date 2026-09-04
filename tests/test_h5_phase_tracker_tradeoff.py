"""Focused tests for the H5-P2A phase-tracker trade-off study."""

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
from experiments.ballnstick_analysis.run_ballnstick_h5_phase_tracker_tradeoff import (
    _candidate_selection,
    _context_specs,
    _load_sources,
    _run_context_specs,
    _split_normalized_ar1_path,
    _tracker_rows,
    _validate_design,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (
    _with_action_frequency,
)


class H5PhaseTrackerTradeoffTests(unittest.TestCase):
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
                    "analysis=ballnstick_h5_phase_tracker_tradeoff",
                    "env.simulation.obs_win_len=1000",
                ],
            )

    def test_sources_hash_lock_and_crossed_design_is_disjoint(self) -> None:
        sources = _load_sources(self.cfg)
        _validate_design(self.cfg, sources)
        contexts = _context_specs(self.cfg)
        self.assertEqual(len(contexts), 24)
        self.assertEqual(len({row["structure_seed"] for row in contexts}), 6)
        self.assertEqual(
            {row["hidden_frequency_hz"] for row in contexts}, {9.0, 11.0}
        )
        self.assertEqual(
            {row["diffusion_rad2_per_s"] for row in contexts}, {0.5, 2.0}
        )
        self.assertEqual(
            {row["shared_modulated_fraction"] for row in contexts}, {1.0}
        )
        new_seeds = {
            int(row[column])
            for row in contexts
            for column in (
                "structure_seed", "history_seed", "phase_seed", "trial_seed",
                "noise_seed",
            )
        }
        self.assertTrue(new_seeds.isdisjoint(sources["source_seeds"]))

    def test_smoke_selection_preserves_crossed_conditions(self) -> None:
        cfg = self.cfg.copy()
        with open_dict(cfg):
            cfg.analysis.smoke_test = True
            cfg.analysis.smoke_context_limit = 4
        selected = _run_context_specs(cfg)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {row["hidden_frequency_hz"] for row in selected}, {9.0, 11.0}
        )
        self.assertEqual({row["label"] for row in selected}, {
            "low_diffusion", "high_diffusion",
        })

    def test_ar1_path_is_paired_and_normalized_only_before_boundary(self) -> None:
        first = _split_normalized_ar1_path(
            n_samples=4000, normalization_samples=3000, seed=401,
            coefficient=0.95,
        )
        repeated = _split_normalized_ar1_path(
            n_samples=4000, normalization_samples=3000, seed=401,
            coefficient=0.95,
        )
        different = _split_normalized_ar1_path(
            n_samples=4000, normalization_samples=3000, seed=402,
            coefficient=0.95,
        )
        self.assertTrue(np.array_equal(first, repeated))
        self.assertFalse(np.array_equal(first, different))
        self.assertAlmostEqual(float(np.sqrt(np.mean(first[:3000] ** 2))), 1.0)

    def test_trackers_share_a_common_causal_audit_grid(self) -> None:
        dt_ms = float(self.cfg.env.network.dt)
        times_ms = np.arange(dt_ms, 5000.0 + 0.5 * dt_ms, dt_ms)
        frequency_hz = 9.0
        phase = 2.0 * np.pi * frequency_hz * times_ms / 1000.0
        neural = np.sin(phase)
        observed = neural + 0.05 * np.sin(2.0 * np.pi * 3.0 * times_ms / 1000.0)
        frequency_cfg = _with_action_frequency(self.cfg, frequency_hz)
        rows = _tracker_rows(
            observed=observed,
            neural=neural,
            times_ms=times_ms,
            latent_times_ms=times_ms,
            latent_phase_rad=phase,
            latent_transfer_offset_rad=0.0,
            evaluation_start_ms=2000.0,
            evaluation_stop_ms=4000.0,
            context={
                "context_id": "synthetic",
                "structure_seed": 1,
                "hidden_frequency_hz": frequency_hz,
                "label": "low_diffusion",
                "diffusion_rad2_per_s": 0.5,
            },
            noise_fraction=0.25,
            frequency_cfg=frequency_cfg,
            cfg=self.cfg,
        )
        table = pd.DataFrame(rows)
        self.assertEqual(len(table), 2 * 16)
        self.assertEqual(set(table.tracker_profile), {CONSERVATIVE, RESPONSIVE})
        self.assertTrue(table.estimate_uses_only_preceding_observed_EEG.all())
        self.assertFalse(table.latent_reference_used_by_tracker.any())
        self.assertTrue(np.isfinite(table.absolute_latent_reference_error_rad).all())
        updates = table.groupby("tracker_profile").profile_update_applied.sum()
        self.assertEqual(int(updates[CONSERVATIVE]), 8)
        self.assertEqual(int(updates[RESPONSIVE]), 16)

    def test_candidate_selection_freezes_smallest_passing_noise_pair(self) -> None:
        carrier_rows = []
        summary_rows = []
        advantage_rows = []
        for structure in range(6):
            for frequency in (9.0, 11.0):
                for label, diffusion in (
                    ("low_diffusion", 0.5), ("high_diffusion", 2.0)
                ):
                    context_id = f"s{structure}_f{frequency:g}_{label}"
                    for noise in (0.25, 0.5, 0.75):
                        carrier_rows.append({
                            "noise_fraction": noise,
                            "carrier_identified": True,
                            "carrier_selection_correct": True,
                            "carrier_usable_for_phase_audit": True,
                        })
                        for profile in (CONSERVATIVE, RESPONSIVE):
                            summary_rows.append({
                                "context_id": context_id,
                                "structure_seed": 900000 + structure,
                                "hidden_frequency_hz": frequency,
                                "label": label,
                                "diffusion_rad2_per_s": diffusion,
                                "noise_fraction": noise,
                                "tracker_profile": profile,
                                "tracker_actionable_fraction": 1.0,
                            })
                        fast_advantage = 0.0
                        if label == "high_diffusion" and np.isclose(noise, 0.25):
                            fast_advantage = 0.04
                        elif label == "low_diffusion" and np.isclose(noise, 0.5):
                            fast_advantage = -0.03
                        elif label == "low_diffusion" and np.isclose(noise, 0.75):
                            fast_advantage = -0.05
                        advantage_rows.append({
                            "context_id": context_id,
                            "structure_seed": 900000 + structure,
                            "hidden_frequency_hz": frequency,
                            "label": label,
                            "diffusion_rad2_per_s": diffusion,
                            "noise_fraction": noise,
                            "fast_advantage_latent_error_rad": fast_advantage,
                            "slow_advantage_observation_error_rad": (
                                0.02 if label == "low_diffusion" else 0.0
                            ),
                        })
        candidates, structures, frozen = _candidate_selection(
            pd.DataFrame(carrier_rows), pd.DataFrame(summary_rows),
            pd.DataFrame(advantage_rows), self.cfg,
        )
        self.assertTrue(candidates.passes_measurement_tradeoff_gate.all())
        self.assertEqual(len(structures), 12)
        self.assertTrue(frozen["candidate_selected"])
        self.assertAlmostEqual(frozen["selected_high_noise_fraction"], 0.5)
        self.assertTrue(frozen["selection_uses_no_stimulation_outcomes"])


if __name__ == "__main__":
    unittest.main()
