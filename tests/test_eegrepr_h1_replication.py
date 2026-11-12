"""Finite-replication statistics and honest independent-context coverage."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.eeg_representation.h1.replication import (
    finite_m_estimates, replication_curve, fixed_budget_plan, check_pool_independence,
    fit_pooled_one_future, predict_pooled_one_future, gaussian_calibration, fixed_budget_comparison, run_replication,
)


class ReplicationIdentityTests(unittest.TestCase):
    def test_negative_excess_is_not_clipped_and_m1_is_unavailable(self):
        r=finite_m_estimates([[-1.],[1.]],prediction=[0.])
        self.assertEqual(r['variance'],2.)
        self.assertEqual(r['bias_corrected_excess'],-1.)
        self.assertEqual(r['variance'],r['loo_corrected_mse'])
        one=finite_m_estimates([[3.]],prediction=[1.])
        self.assertIsNone(one['variance']); self.assertIsNone(one['bias_corrected_excess'])
        self.assertEqual(one['model_risk'],4.)

    def test_corrected_excess_and_independent_oracle_risk_mc(self):
        rng=np.random.default_rng(137)
        k,m=5000,8
        x=rng.normal(size=(k,m,2))*np.sqrt(2)
        predicted=np.full(2,.6)
        estimates=np.array([finite_m_estimates(a,predicted)['bias_corrected_excess'] for a in x])
        self.assertAlmostEqual(estimates.mean(),.36,delta=6*estimates.std(ddof=1)/np.sqrt(k))
        independent=rng.normal(size=(k,2))*np.sqrt(2)
        losses=((x.mean(axis=1)-independent)**2).mean(axis=1)
        self.assertAlmostEqual(losses.mean(),(1+1/m)*2,delta=6*losses.std(ddof=1)/np.sqrt(k))
        self.assertGreater(np.mean(estimates<0),.1)

    def test_disjoint_reference_ids_and_pending_reference(self):
        with self.assertRaises(ValueError):check_pool_independence(['a','b'],['b','c'])
        with self.assertRaises(ValueError):check_pool_independence(['a','a'],[])
        r=replication_curve(np.arange(16.)[:,None],m_values=[1,2,16,32],repetitions=8,bootstrap_samples=8,
                            scoring_ids=[f's{i}' for i in range(16)],reference_ids=[])
        self.assertTrue(r['reference']['status'].startswith('pending'))
        self.assertNotIn('mean_rmse_against_finite_reference',r['rows'][1])
        self.assertTrue(r['rows'][-1]['status'].startswith('pending'))
        self.assertIsNone(r['rows'][0]['variance_mean'])

    def test_nested_panel_mean_variance_and_exact_gaussian_coverage(self):
        rng=np.random.default_rng(912)
        x=rng.normal(size=(2048,1))*2+3
        ref=rng.normal(size=(4096,1))*2+3
        r=replication_curve(x,ref,m_values=[1,2,8,32,128],repetitions=1000,bootstrap_samples=64,seed=19,
            scoring_ids=[f's{i}' for i in range(len(x))],reference_ids=[f'r{i}' for i in range(len(ref))],
            truth_mean=[3.],truth_variance=4.,gaussian_scalar=True)
        row=r['rows'][3]
        self.assertAlmostEqual(row['variance_mean'],4.,delta=.3)
        self.assertGreater(row['mean_interval_exact_truth_coverage'],.91)
        self.assertLess(row['mean_interval_exact_truth_coverage'],.99)
        self.assertGreater(row['variance_interval_exact_truth_coverage'],.91)
        self.assertLess(row['variance_interval_exact_truth_coverage'],.99)
        self.assertGreater(r['reference']['variance_bootstrap_standard_error'],0)
        self.assertLess(r['rows'][-1]['mean_rmse_against_finite_reference'],r['rows'][1]['mean_rmse_against_finite_reference'])


class OneFutureLearningTests(unittest.TestCase):
    def test_pooled_visible_history_variance_includes_hidden_state(self):
        rng=np.random.default_rng(734)
        n=2400; ne=1600
        v=rng.normal(size=n); ve=rng.normal(size=ne)
        x=np.repeat(v[:,None],16,axis=1); xe=np.repeat(ve[:,None],16,axis=1)
        # C=2V+U, U~N(0,2); future innovation variance=1. A correct
        # EEG-only variance is 3, so scoring it against state variance 1
        # would incorrectly penalize the correct uncertainty prediction.
        y=2*v+rng.normal(size=n)*np.sqrt(2)+rng.normal(size=n)
        ye=2*ve+rng.normal(size=ne)*np.sqrt(2)+rng.normal(size=ne)
        model=fit_pooled_one_future(x,y[:,None],np.arange(n)%4)
        mean,var=predict_pooled_one_future(model,xe,structure_ids=10+np.arange(ne)%4)
        scores=gaussian_calibration(ye[:,None],mean,var)
        self.assertGreater(var.mean(),2.6); self.assertLess(var.mean(),3.5)
        self.assertGreater(scores['marginal_coverage95'],.92)
        self.assertLess(scores['marginal_coverage95'],.98)
        self.assertLess(np.mean((mean[:,0]-2*ve)**2),.04)
        self.assertIn('visible EEG',model['variance_conditioning'])
        with self.assertRaises(ValueError):predict_pooled_one_future(model,xe,structure_ids=np.zeros(ne,dtype=int))
        with self.assertRaises(ValueError):fit_pooled_one_future(x,y[:,None],np.arange(n)%2)

    def test_ridge_small_context_design_discloses_true_context_count(self):
        rng=np.random.default_rng(92);x=rng.normal(size=(12,16));y=rng.normal(size=(12,1))
        ids=np.array([f'c{i}' for i in range(12)]);groups=np.arange(12)%3
        model=fit_pooled_one_future(np.repeat(x,4,axis=0),np.repeat(y,4,axis=0),
            np.repeat(groups,4),context_ids=np.repeat(ids,4))
        q=model['context_qualification']
        self.assertEqual(q['independent_training_contexts'],12)
        self.assertTrue(all(f['independent_contexts']==8 and f['outcome_rows']==32 for f in q['folds']))
        self.assertTrue(all(f['regularized_underdetermined'] for f in q['folds']))

    def test_deterministic_targets_have_declared_positive_numerical_variance(self):
        rng=np.random.default_rng(902)
        x=rng.normal(size=(128,8))
        model=fit_pooled_one_future(x,np.zeros((128,1)),np.arange(128)%4)
        mean,var=predict_pooled_one_future(model,x,structure_ids=np.full(128,12))
        self.assertTrue(np.isfinite(var).all())
        self.assertTrue(np.all(var>=model['variance_floor']))
        self.assertLess(var.max(),1e-10)
        self.assertTrue(np.isfinite(gaussian_calibration(np.zeros_like(mean),mean,var)['nll_per_dimension']))

    def test_runner_preserves_missing_reference_and_numeric_artifacts(self):
        rng=np.random.default_rng(101)
        def split(n,m,offset):
            x=rng.normal(size=(n,16)); y=x[:,-1,None,None]+rng.normal(size=(n,m,2))
            return dict(X=x,Y=y,structure_id=offset+np.arange(n)%4,
                regime=np.array(['R0','Ralpha','Rbeta_low','Rbeta_high'])[np.arange(n)%4],
                context_id=np.array([f's{offset+i%4}-p{i//4:04d}-R' for i in range(n)]),
                future_mask=np.ones((n,m),dtype=bool),future_id=np.array([[f'{offset}/{i}/{j}' for j in range(m)] for i in range(n)]),
                reference_Y=np.zeros((n,0,2)),reference_mask=np.zeros((n,0),dtype=bool),
                reference_future_id=np.full((n,0),'',dtype='<U1'))
        train=split(96,1,0); dev=split(8,1,10); ev=split(8,4,20)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); files={}
            for name,values in [('train',train),('development',dev),('evaluation',ev)]:
                path=root/f'{name}.npz';np.savez(path,**values);files[name]={'path':path.name}
            manifest=dict(files=files,target_fingerprint='target',comparison_fingerprint='comparison',normalization={'y_scale':2.})
            panel=dict(train=train,development=dev,evaluation=ev,manifest=manifest)
            cfg=dict(replication={'m_values':[1,2,4,8],'repetitions':4,'bootstrap_samples':4},
                data={'structures':{'train':[0,1,2,3]},'regimes':[{'name':r} for r in ['R0','Ralpha','Rbeta_low','Rbeta_high']]})
            with patch('experiments.eeg_representation.h1.dataset.list_jobs',return_value=['primary']), \
                 patch('experiments.eeg_representation.h1.dataset.load_panel',return_value=panel), \
                 patch('experiments.eeg_representation.h1.artifacts.require_compatible'), \
                 patch('experiments.eeg_representation.h1.artifacts.resume_stage',return_value=None), \
                 patch('experiments.eeg_representation.h1.artifacts.stage_complete'):
                out=run_replication(cfg,root)
            job=out['jobs'][0]
            self.assertIn('pending reference',job['fixed_k_panels'][0]['reference_status'])
            self.assertEqual(job['pooled_one_future']['status'],'executed')
            self.assertIsNone(job['pooled_one_future']['state_fork_variance_rmse'])
            self.assertEqual(job['physical_mse_conversion'],4.)
            self.assertTrue((root/'replication/primary/estimates.npz').exists())
            self.assertTrue((root/'replication/primary/replication.pdf').exists())
            with np.load(root/'replication/primary/estimates.npz',allow_pickle=False) as arrays:
                self.assertIn('context0_m2_variance',arrays.files)

    def test_fixed_budget_union_and_honest_deficit(self):
        plan=fixed_budget_plan()
        self.assertEqual(plan['unique_future_union'],768)
        self.assertEqual(plan['required_candidate_contexts'],256)
        self.assertEqual(plan['additional_contexts_required'],256)
        existing=[dict(structure_id=c,regime=r,candidate_index=0,n_futures=16)
                  for c in range(4) for r in ['R0','Ralpha','Rbeta_low','Rbeta_high']]
        plan=fixed_budget_plan(available_contexts=existing)
        self.assertEqual(plan['additional_contexts_required'],240)
        self.assertEqual(plan['additional_union_futures_required'],512)
        self.assertEqual([r['k'] for r in plan['allocations']],[256,128,64,32,16])
        self.assertTrue(all(r['outcomes']==256 for r in plan['allocations']))

    def test_complete_allocations_use_same_heldout_panel_and_save_ids(self):
        rng=np.random.default_rng(724)
        regimes=['a','b','c','d']; rows=[(c,r,j) for c in range(4) for r in regimes for j in range(16)]
        n=len(rows); counts=np.asarray([max(m for m in (1,2,4,8,16) if j<16//m) for c,r,j in rows])
        x=rng.normal(size=(n,16)); y=2*x[:,-1,None,None]+rng.normal(size=(n,16,1))
        train=dict(X=x,Y=y,structure_id=np.array([r[0] for r in rows]),regime=np.array([r[1] for r in rows]),
            context_id=np.array([f's{c}-p{j:04d}-{r}' for c,r,j in rows]),
            future_mask=np.arange(16)[None,:]<counts[:,None],
            future_id=np.array([[f'{i}/{j}' for j in range(16)] for i in range(n)]))
        xe=rng.normal(size=(24,16)); ye=2*xe[:,-1,None,None]+rng.normal(size=(24,4,1))
        ev=dict(X=xe,Y=ye,structure_id=10+np.arange(24)%4,context_id=np.array([f'e{i}' for i in range(24)]),
                future_mask=np.ones((24,4),dtype=bool))
        r=fixed_budget_comparison(train,ev)
        self.assertTrue(all(a['status']=='executed' for a in r['allocations']))
        self.assertTrue(all(a['training_outcomes']==256 for a in r['allocations']))
        self.assertEqual([a['independent_training_contexts'] for a in r['allocations']],[256,128,64,32,16])
        for m in (1,2,4,8,16):
            self.assertEqual(r['arrays'][f'm{m}_mean'].shape,(24,1))
            self.assertEqual(len(set(r['arrays'][f'm{m}_training_future_id'].tolist())),256)
        train['future_mask'][:]=False;train['future_mask'][:,0]=True
        incomplete=fixed_budget_comparison(train,ev)
        self.assertEqual(incomplete['allocations'][0]['status'],'executed')
        self.assertTrue(all(a['status'].startswith('pending') for a in incomplete['allocations'][1:]))


if __name__=='__main__': unittest.main()
