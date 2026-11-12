"""Exact and Monte Carlo identities for H1, independent of simulator imports."""
import tempfile
import unittest
import numpy as np

from experiments.eeg_representation.analytical import oscillator_transition
from experiments.eeg_representation.h1.controls import (
    gaussian_information_model, gaussian_information_control, poisson_block_law,
    poisson_block_control, repeated_future_control,
)


class GaussianInformationTests(unittest.TestCase):
    def test_known_information_nested_and_four_term_identity(self):
        F,Q=oscillator_transition(.01,frequency_hz=9,damping_per_s=2,diffusion_variance_per_s=4)
        m=gaussian_information_model(F,Q,history_samples=8,target_steps=(1,4,10),observation_noise=.5)
        self.assertTrue(all(v>0 for v in m['exact_terms'].values()))
        self.assertAlmostEqual(sum(m['exact_terms'].values()),m['exact_model_risk'])
        identity=gaussian_information_model(F,Q,history_samples=8,representation=np.eye(8),observation_noise=.5)
        self.assertAlmostEqual(identity['exact_terms']['compression'],0,places=10)
        complete=gaussian_information_model(F,Q,history_samples=8,observation_noise=0)
        self.assertLess(complete['exact_terms']['hidden_state'],m['exact_terms']['hidden_state'])

    def test_mc_recovers_scalar_block_and_no_process_noise(self):
        for noise,process,steps in ((0,0,(1,4,10)),(0,4,(4,)),(.5,4,(1,4,10))):
            with self.subTest(noise=noise,process=process,steps=steps):
                result=gaussian_information_control(seed=923,n_trials=5000,
                    process_noise=process,observation_noise=noise,target_steps=steps)
                self.assertTrue(result['passed'],result['terms'])
                if not process: self.assertEqual(result['terms']['innovation']['analytic_risk'],0)

    def test_affine_scaling_changes_every_risk_by_square(self):
        a=gaussian_information_control(seed=19,n_trials=1024)
        b=gaussian_information_control(seed=19,n_trials=1024,affine_scale=-3,affine_offset=40)
        for key in a['terms']:
            self.assertAlmostEqual(b['terms'][key]['analytic_risk'],9*a['terms'][key]['analytic_risk'],places=10)
            self.assertAlmostEqual(b['terms'][key]['empirical_risk'],9*a['terms'][key]['empirical_risk'],places=10)

    def test_global_random_state_untouched(self):
        before=np.random.get_state()
        gaussian_information_control(n_trials=32)
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1]); self.assertEqual(before[2:],after[2:])


class PoissonInformationTests(unittest.TestCase):
    def test_block_covariance_and_projection_mc(self):
        raw=poisson_block_control(seed=142,n_futures=6000)
        projected=poisson_block_control(seed=142,n_futures=6000,projection=np.full((1,6),1/6))
        self.assertTrue(raw['passed']); self.assertTrue(projected['passed'])
        cov=np.asarray(raw['analytic_covariance']); empirical=np.asarray(raw['empirical_covariance'])
        np.testing.assert_allclose(empirical,cov,atol=.025)
        self.assertAlmostEqual(projected['analytic_risk'],cov.mean())
        self.assertAlmostEqual(raw['sample_variance'],raw['loo_corrected_mse'],places=12)
        self.assertGreater(cov[1,2],0)

    def test_zero_rate_and_affine_scaling(self):
        zero=poisson_block_control(n_futures=32,rate_hz=0)
        self.assertEqual(zero['analytic_risk'],0)
        self.assertLess(zero['sample_variance'],1e-28)
        a=poisson_block_control(seed=7,n_futures=128)
        b=poisson_block_control(seed=7,n_futures=128,affine_scale=4,affine_offset=-12)
        self.assertAlmostEqual(b['sample_variance'],16*a['sample_variance'])
        self.assertAlmostEqual(b['analytic_risk'],16*a['analytic_risk'])
        with self.assertRaises(ValueError): poisson_block_law([0,.1])

    def test_repeated_mean_variance_hidden_state_remainder(self):
        control=repeated_future_control(seed=167,n_repeats=18000,m_values=(1,2,8,32,201))
        for row in control['rows']:
            self.assertAlmostEqual(row['empirical_mean_variance_given_state'],row['analytic_mean_variance_given_state'],
                                   delta=.05*row['analytic_mean_variance_given_state'])
            self.assertAlmostEqual(row['empirical_mean_variance_given_visible_history'],row['analytic_mean_variance_given_visible_history'],
                                   delta=.05*row['analytic_mean_variance_given_visible_history'])
            if row['m']>1:
                self.assertAlmostEqual(row['empirical_sample_variance_variance'],row['analytic_sample_variance_variance'],
                                       delta=.09*row['analytic_sample_variance_variance'])
        self.assertEqual(control['rows'][0]['sample_variance_status'],'unavailable')
        self.assertEqual(control['gaussian_scalar_m_for_relative_sd_at_most_point_one'],201)
        self.assertGreater(control['rows'][-1]['empirical_mean_variance_given_visible_history'],.6)


if __name__=='__main__': unittest.main()
