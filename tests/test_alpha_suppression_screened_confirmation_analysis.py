"""Tests for prospective alpha-phenotype screening and confirmation logic."""

import inspect
import unittest

from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_screened_confirmation import (
    _classify_alpha,
    _screening_decision,
    _screening_target_reached,
)


def _cfg():
    return OmegaConf.create({
        "analysis": {
            "rate_guardrails_hz": {
                "E_min": 0.1,
                "E_max": 30.0,
                "I_min": 0.1,
                "I_max": 60.0,
            }
        }
    })


def _episode(alpha=-20.5, e_rate=3.0, i_rate=8.0):
    return {
        "epoch_rows": [{
            "epoch": "stimulation",
            "log10_alpha_power_8_12_hz": alpha,
            "E_firing_rate_hz": e_rate,
            "I_firing_rate_hz": i_rate,
        }]
    }


def _model():
    return {
        "classification_threshold": -20.63,
        "A_is_above_threshold": True,
        "B_mean_log10_alpha": -20.75,
    }


def _phase(passed=True):
    return {
        "screen_phase_at_action_rad": 1.0,
        "screen_phase_split_error_deg": 10.0,
        "screen_10hz_resultant_v": 1.0e-10,
        "screen_eeg_rms_v": 4.0e-10,
        "screen_10hz_resultant_to_rms": 0.25,
        "screen_phase_quality_pass": passed,
    }


class AlphaSuppressionScreenedConfirmationTests(unittest.TestCase):
    def test_frozen_threshold_classifies_alpha_without_treatment_data(self):
        self.assertEqual(_classify_alpha(-20.5, _model()), "A")
        self.assertEqual(_classify_alpha(-20.8, _model()), "B")

    def test_elevated_phase_actionable_candidate_is_eligible(self):
        result = _screening_decision(
            seed=1,
            screening_order=1,
            a_episode=_episode(),
            phase_quality=_phase(),
            target_model=_model(),
            cfg=_cfg(),
        )
        self.assertTrue(result["eligible"])
        self.assertFalse(result["screening_uses_stimulation_outcome"])
        self.assertFalse(result["screening_uses_seed_specific_B"])

    def test_candidate_without_elevated_alpha_is_excluded(self):
        result = _screening_decision(
            seed=2,
            screening_order=2,
            a_episode=_episode(alpha=-20.8),
            phase_quality=_phase(),
            target_model=_model(),
            cfg=_cfg(),
        )
        self.assertFalse(result["eligible"])
        self.assertIn("not_elevated", result["exclusion_reasons"])

    def test_phase_unstable_candidate_is_excluded_even_when_alpha_is_high(self):
        result = _screening_decision(
            seed=3,
            screening_order=3,
            a_episode=_episode(),
            phase_quality=_phase(False),
            target_model=_model(),
            cfg=_cfg(),
        )
        self.assertFalse(result["eligible"])
        self.assertIn("unstable_or_weak_10hz_phase", result["exclusion_reasons"])

    def test_screening_api_cannot_receive_active_or_B_outcomes(self):
        parameters = inspect.signature(_screening_decision).parameters
        self.assertNotIn("active_episode", parameters)
        self.assertNotIn("b_episode", parameters)

    def test_rank_zero_enrollment_stop_is_broadcast_to_worker(self):
        class FakeComm:
            def __init__(self, root_value):
                self.root_value = root_value
                self.calls = []

            def bcast(self, value, root):
                self.calls.append((value, root))
                return self.root_value

        root_comm = FakeComm(True)
        worker_comm = FakeComm(True)

        self.assertTrue(_screening_target_reached(
            comm=root_comm,
            rank=0,
            enrolled_count=8,
            target_count=8,
        ))
        self.assertTrue(_screening_target_reached(
            comm=worker_comm,
            rank=1,
            enrolled_count=0,
            target_count=8,
        ))
        self.assertEqual(root_comm.calls, [(True, 0)])
        self.assertEqual(worker_comm.calls, [(None, 0)])


if __name__ == "__main__":
    unittest.main()
