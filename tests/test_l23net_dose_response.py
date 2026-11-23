"""Regression coverage for eligibility, qualification and paired dose analysis."""

import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.l23net_analysis import dose_response_protocol as p
from experiments.l23net_analysis.analyze_l23net_dose_response import (
    analyze_suite, isi_adjusted_ppc, separated_ppc, summarize_discovery,
)
from experiments.l23net_analysis.nci.submit_l23net_dose_response import check_budget, worker_command


class DoseResponseTests(unittest.TestCase):
    def test_prospective_eligibility_and_engineering_bypass(self):
        screen = {'eligible': False}
        for stage, safe, delivered in [('discovery', True, 'sham'), ('qualification', True, 'dose_08'),
                                       ('qualification', False, 'sham')]:
            self.assertEqual(p.decide('dose_08', stage, screen, safe, 28000)['delivered_arm'], delivered)
        for arm, amplitude in p.AMPLITUDES.items():
            d = p.decide(arm, 'discovery', {'eligible': True}, True, 28000)
            self.assertEqual(d['amplitude_v_per_m'], amplitude)
            self.assertEqual(d['frequency_hz'], 14 if amplitude else 0)
        self.assertFalse(p.baseline_rates_safe([{**p.RATE_LIMITS, 'HL23SST': float('nan')}]))

    def test_reservations_and_fa32_dependency_commands(self):
        self.assertEqual(check_budget('qualification', 200), 4.992)
        self.assertEqual(check_budget('discovery', 200), 59.904)
        with self.assertRaises(ValueError):
            check_budget('discovery', 60)
        jobs = p.jobs('discovery')
        self.assertEqual(len(jobs), 24)
        self.assertTrue(all(job['project'] == 'fa32' for job in jobs))
        self.assertFalse(set(p.SEEDS['qualification']) & set(p.SEEDS['discovery']))
        manifest = {'stage': 'discovery', 'max_concurrent': 8, 'jobs': jobs, 'commit': 'test'}
        jobs[0]['job_id'] = '123.gadi-pbs'
        command = worker_command(Path('/test/gadi'), manifest, 8)
        self.assertEqual(command[:3], ['qsub', '-P', 'fa32'])
        self.assertIn('depend=afterok:123.gadi-pbs', command)
        self.assertIn('ncpus=624,mem=256GB,walltime=02:00:00,jobfs=10GB,storage=gdata/ny83', command)
        self.assertNotIn('-W', worker_command(Path('/test/gadi'), manifest, 7))

    def test_phase_timing_controls_drop_local_pairs_and_preserve_identical_arms(self):
        phases = (np.arange(19) + 0.5) / 14 + np.arange(19)
        self.assertAlmostEqual(separated_ppc(phases, phases, 1), 0)
        self.assertIsNone(separated_ppc(np.array([.2, 1.2]), np.array([.4, 1.4]), 1))
        periodic = np.arange(1, 281) / 14.
        self.assertAlmostEqual(isi_adjusted_ppc(periodic, np.random.default_rng(13)), 0, places=12)

    def test_select_on_primary_and_keep_active_dose_contrasts(self):
        rows = []
        for seed in p.SEEDS['discovery']:
            for arm, benefit in [('dose_02', .05), ('dose_04', .35), ('dose_08', -.03)]:
                rows.append({'seed': seed, 'arm': arm, 'eligible': True, 'rate_safe': True,
                             'benefit': benefit, 'late_benefit': 1. if arm == 'dose_08' else 0.,
                             'excluded_benefit': benefit, 'washout_benefit': 0., 'shift_alignment': benefit,
                             'band_log10_sham_minus_active': [.01] * 3,
                             'timing': {pop: {metric: {'active_minus_sham': 0.}
                                              for metric in ('separated_1s', 'separated_2s', 'isi_adjusted')}
                                        for pop in p.POPULATIONS}})
        result = summarize_discovery(rows)
        self.assertEqual(result['candidate_for_independent_confirmation'], 'dose_04')
        self.assertAlmostEqual(result['dose_comparisons']['dose_04_minus_dose_02']['mean'], .30)
        rows[1]['rate_safe'] = False
        self.assertIsNone(summarize_discovery(rows)['candidate_for_independent_confirmation'])
        with self.assertRaises(ValueError):
            summarize_discovery(rows[:-1])

    def test_missing_workers_cannot_freeze_upper_dose_gate(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            jobs = [dict(job, job_id=f'{index}.gadi-pbs') for index, job in enumerate(p.jobs('qualification'))]
            manifest = {'stage': 'qualification', 'submission_status': 'submitted', 'jobs': jobs,
                        'commit': 'test', 'code_sha256': {}, 'mechanism_sha256': {}}
            with patch.object(p, 'check_manifest', return_value=(manifest, None, None, None)):
                result = analyze_suite(folder)
            self.assertEqual(result['status'], 'incomplete_or_failed')
            self.assertEqual(result['completed_valid_runs'], 0)
            self.assertFalse((folder / p.UPPER_GATE).exists())


if __name__ == '__main__':
    unittest.main()
