"""Guard the frequency comparison, independent histories and technical release."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis import repeatability_protocol as repeat
from experiments.l23net_analysis import theta_frequency_protocol as p
from experiments.l23net_analysis.analyze_l23net_theta_frequency import check_structures, summarize_combined
from experiments.l23net_analysis.nci.submit_l23net_theta_frequency import reservation_for_history, worker_command
from experiments.l23net_analysis.r1_protocol import save_json


def make_summary(history, theta_14, theta_10):
    structures, rows = {}, []
    for seed in p.SEEDS:
        structures[str(seed)] = {'structure_sha256': str(seed), 'build_sha256': f'build{seed}', 'ou_sites': 12,
                                 'history_seed': (seed, history), 'stream_sha256': f'{seed}-{history}',
                                 'prestimulation_sha256': f'prefix-{seed}-{history}',
                                 'sham_carrier_audit': {'baseline': {'accepted': False}, 'stable_across_epochs': False,
                                                        'phase_audit': [{'accepted': False}]}}
        for arm, theta in zip(p.ARMS[1:], (theta_14, theta_10)):
            rows.append({'seed': seed, 'arm': arm, 'eligible': True, 'rate_safe': True,
                         'band_log10_sham_minus_active': [theta, 0., -.03],
                         'benefit': 0., 'late_benefit': 0., 'washout_benefit': 0.,
                         'low_beta_excluding_14hz_log10_active_over_sham': .05})
    return {'structures': structures, 'rows': rows}


class ThetaFrequencyTests(unittest.TestCase):
    def test_new_cohort_and_matched_history_grid(self):
        self.assertTrue(set(p.SEEDS).isdisjoint(repeat.SEEDS))
        self.assertEqual(len(p.SEEDS), 8)
        one, two = (p.jobs(history) for history in p.HISTORIES)
        for wave in (one, two):
            self.assertEqual(len(wave), 24)
            self.assertEqual(len({j['history_seed'] for j in wave}), 8)
            for seed in p.SEEDS:
                triad = [j for j in wave if j['seed'] == seed]
                self.assertEqual(tuple(j['arm'] for j in triad), p.ARMS)
                self.assertEqual(len({j['history_seed'] for j in triad}), 1)
        self.assertTrue({j['history_seed'] for j in one}.isdisjoint(j['history_seed'] for j in two))

    def test_same_amplitude_different_frequency_and_exact_sham_fallback(self):
        yes, no = {'eligible': True}, {'eligible': False}
        for arm, frequency in zip(p.ARMS[1:], (14., 10.)):
            action = p.decide(arm, yes, True, 28000.)
            self.assertEqual((action['frequency_hz'], action['amplitude_v_per_m']), (frequency, .2))
            self.assertEqual(action['phase_rad_at_onset'], 0.)
            self.assertEqual(action['field_direction'], [0., 0., 1.])
            for screen, safe in ((no, True), (yes, False)):
                action = p.decide(arm, screen, safe, 28000.)
                self.assertEqual((action['delivered_arm'], action['frequency_hz'], action['amplitude_v_per_m']),
                                 ('sham', 0., 0.))

    def test_combined_effect_is_structure_level_and_frequency_contrast_is_paired(self):
        first, second = make_summary(1, .06, .02), make_summary(2, .08, .03)
        result = summarize_combined(first, second)
        main, frequency = result['primary_14hz_vs_sham'], result['frequency_14hz_vs_10hz']
        self.assertEqual(main['n_structures'], 8)
        self.assertEqual(main['permutation_draws'], 256)
        self.assertAlmostEqual(main['mean'], .07)
        self.assertAlmostEqual(frequency['mean'], .045)
        self.assertTrue(main['supported'])
        self.assertTrue(frequency['tested_after_primary'])
        self.assertTrue(frequency['supported'])
        self.assertEqual(result['eligible_histories'], 16)
        self.assertLess(result['secondary']['fixed_low_beta']['low_beta_vs_sham']['geometric_percent_reduction'], 0.)

    def test_effect_must_repeat_and_primary_failure_blocks_secondary_test(self):
        result = summarize_combined(make_summary(1, .12, -.02), make_summary(2, -.01, -.07))
        main, frequency = result['primary_14hz_vs_sham'], result['frequency_14hz_vs_10hz']
        self.assertTrue(main['checks']['practical_mean'])
        self.assertFalse(main['checks']['each_history_mean_positive'])
        self.assertFalse(main['supported'])
        self.assertFalse(frequency['tested_after_primary'])
        self.assertNotIn('p_one_sided', frequency)
        self.assertFalse(frequency['supported'])

    def test_structure_and_history_fingerprints(self):
        first, second = make_summary(1, .04, .02), make_summary(2, .06, .03)
        check_structures(second['structures'], first)
        seed = str(p.SEEDS[0])
        second['structures'][seed]['stream_sha256'] = first['structures'][seed]['stream_sha256']
        with self.assertRaisesRegex(ValueError, 'new realized OU history'):
            check_structures(second['structures'], first)

    def test_release_depends_on_technical_completeness_only(self):
        data = {'history': 1, 'status': 'history_complete', 'errors': [], 'completed_valid_runs': 24,
                'run_report_sha256': {job['name']: 'hash' for job in p.jobs(1)},
                'protocol': p.PROTOCOL, 'code_sha256': p.code_hashes(), 'mechanism_sha256': dose.mechanism_hashes(),
                **make_summary(1, -.1, -.2)}
        with TemporaryDirectory() as folder:
            path = Path(folder) / p.SUMMARY
            save_json(path, reference.signed(data))
            self.assertEqual(p.load_first_history(path)['status'], 'history_complete')
            data['completed_valid_runs'] = 23
            save_json(path, reference.signed(data))
            with self.assertRaisesRegex(ValueError, 'technical checks'):
                p.load_first_history(path)

    def test_budget_and_pbs_throttle(self):
        self.assertEqual(reservation_for_history(1, 60.904), 59.904)
        with self.assertRaisesRegex(ValueError, '60.904 KSU'):
            reservation_for_history(2, 60.90)
        manifest = {'history': 1, 'commit': 'commit', 'max_concurrent': 8, 'jobs': p.jobs(1)}
        manifest['jobs'][0]['job_id'] = '5678.gadi-pbs'
        first = worker_command(Path('/example'), manifest, 0)
        ninth = worker_command(Path('/example'), manifest, 8)
        self.assertNotIn('-W', first)
        self.assertIn('depend=afterok:5678.gadi-pbs', ninth)
        self.assertEqual(ninth[ninth.index('-P') + 1], 'fa32')


if __name__ == '__main__':
    unittest.main()
