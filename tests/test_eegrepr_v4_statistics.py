"""Risk identities, uncertain references and paired-regime inference controls."""
import unittest
import numpy as np
from experiments.eeg_representation.h1.scoring import fork_risk_components
from experiments.eeg_representation.h1.v4.scoring import (
    assert_risk_contract,gaussian_diagnostics,paired_regime_mc,hierarchical_circuit_inference)
from experiments.eeg_representation.h1.v4.replication import finite_reference_comparison,replication_curve


class StatisticsTests(unittest.TestCase):
    def test_finite_m_identity_retains_negative(self):
        y=np.array([[[-1.],[1.]]]);out=fork_risk_components(y,np.zeros((1,1)))
        self.assertLess(out['corrected_excess'][0],0)
        np.testing.assert_allclose(out['mean_risk'],out['state_floor']+out['corrected_excess'])
        one=fork_risk_components(y[:,:1],np.zeros((1,1)))
        self.assertTrue(np.isnan(one['state_floor'][0]))

    def test_reference_correction_and_pool_disjointness(self):
        x=np.array([[-1.],[1.]]);r=np.array([[-2.],[2.]])
        out=finite_reference_comparison(x,r,scoring_ids=['s0','s1'],reference_ids=['r0','r1'])
        self.assertEqual(out['reference_noise_corrected_mean_mse'],-4.)
        self.assertFalse(out['reference_is_exact_truth'])
        self.assertIn('mean_difference_interval95_low',out)
        with self.assertRaises(ValueError):finite_reference_comparison(x,r,scoring_ids=['s0','s1'],reference_ids=['s0','r1'])

    def test_reference_noise_correction_known_gaussian_expectation(self):
        rng=np.random.default_rng(981);values=[];raw=[]
        for _ in range(1500):
            x=rng.normal(0,2,(4,2));r=rng.normal(0,2,(16,2))
            out=finite_reference_comparison(x,r,scoring_ids=range(4),reference_ids=range(10,26),bootstrap=0)
            values.append(out['reference_noise_corrected_mean_mse']);raw.append(out['squared_mean_distance_to_finite_reference'])
        self.assertAlmostEqual(np.mean(values),4/4,delta=.10)
        self.assertAlmostEqual(np.mean(raw),4*(1/4+1/16),delta=.10)

    def test_full_pool_spread_is_not_mc_precision(self):
        x=np.arange(8.)[:,None]
        out=replication_curve(x,scoring_ids=[f's{i}' for i in range(8)],m_values=[1,2,4,8],repetitions=8)
        last=out['rows'][-1]
        self.assertEqual(last['mean_subsample_spread'],0.)
        self.assertFalse(last['mc_uncertainty_is_zero'])
        self.assertIn('unavailable',last['reference_status'])

    def test_gaussian_density_units_and_intervals(self):
        rng=np.random.default_rng(2);y=rng.normal(size=(8,1000,2));mu=np.zeros((8,2));v=np.ones_like(mu);mask=np.ones(y.shape[:2],bool)
        context,arrays=gaussian_diagnostics(y,mu,v,mask,y_scale=1e-6)
        self.assertAlmostEqual(context['coverage_95'].mean(),.95,delta=.01)
        np.testing.assert_allclose(arrays['marginal_nll_physical']-arrays['marginal_nll'],np.log(1e-6))
        np.testing.assert_allclose(arrays['joint_diagonal_nll_physical']-arrays['joint_diagonal_nll'],2*np.log(1e-6))
        self.assertEqual(int(arrays['pit_histogram'].sum()),y.size)

    def test_contract_mismatch_rejected(self):
        a=dict(targetphysicalhash='y',population_id='p',acquisitionhash='x',normalization_id='n',units='V',dimension=2,support=[2,32])
        assert_risk_contract(a,a)
        with self.assertRaises(ValueError):assert_risk_contract(a,dict(a,units='V2'))
        with self.assertRaises(ValueError):assert_risk_contract({}, {})

    def test_paired_regime_not_independent_context_mc(self):
        rng=np.random.default_rng(8);base=rng.normal(size=(2,16,1))
        # Duplicating a perfectly paired regime cannot improve precision.
        single=paired_regime_mc(base,np.zeros((2,1)),[0,1])
        duplicate=paired_regime_mc(np.repeat(base,4,axis=0),np.zeros((8,1)),np.repeat([0,1],4))
        self.assertAlmostEqual(single['pooled_conditional_mc_se']['state_floor'],duplicate['pooled_conditional_mc_se']['state_floor'])
        nested=hierarchical_circuit_inference(np.ones((8,16)),np.repeat([0,1],4),[f's{s}-p0000-R{r}' for s in [0,1] for r in range(4)],bootstrap=100)
        self.assertEqual(nested['n_circuits'],2)
        self.assertEqual(nested['bootstrap_95'],[1.,1.])


if __name__=='__main__':unittest.main()
