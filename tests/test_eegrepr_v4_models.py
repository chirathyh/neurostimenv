"""Meaningful v4 qualification-path and development-selection controls."""
import unittest
import numpy as np
from experiments.eeg_representation.h1.v4.models import (
    DEFAULTS,settings,fit_ridge,predict_ridge,residual_shrinkage,
    calibrate_variance,matched_calibrated_variance,circuit_weights,known_learnable_control)


class ModelQualificationTests(unittest.TestCase):
    def test_complete_ridge_grid_cannot_be_truncated(self):
        c=settings({});self.assertEqual(len(c['ridge_lags'])*len(c['ridge_alphas']),14)
        with self.assertRaises(ValueError):settings({'models':{'ridge_alphas':[.01,.1,1]}})
        with self.assertRaises(ValueError):settings({'models':{'neural_seeds':[0]}})

    def test_ridge_known_linear_and_circuit_weights(self):
        rng=np.random.default_rng(18);x=rng.normal(size=(120,16));w=rng.normal(size=(16,3));y=x@w
        model=fit_ridge(x[:90],y[:90],.01);prediction=predict_ridge(model,x[90:])
        self.assertLess(np.mean((prediction-y[90:])**2),1e-5)
        weights=circuit_weights([1,1,1,2]);np.testing.assert_allclose(weights,[1/6,1/6,1/6,1/2])

    def test_shrinkage_and_calibration_do_not_need_evaluation(self):
        dev=dict(Y=np.array([[[1.]],[[3.]]]),future_mask=np.ones((2,1),bool),structure_id=np.array([1,2]))
        baseline=np.zeros(1);pred=np.array([[10.],[30.]])
        self.assertAlmostEqual(residual_shrinkage(dev,pred,baseline),.1)
        # Same variance family for baseline and neural variance, no special access.
        multiplier=calibrate_variance(dev,np.zeros((2,1)),np.ones((2,1)))
        np.testing.assert_allclose(multiplier,[5.])
        self.assertEqual(residual_shrinkage(dev,-pred,baseline),0.)

    def test_exact_matched_variance_family_is_development_only(self):
        # Two contexts in circuit 1 and one in circuit 2: circuit weights matter.
        dev=dict(Y=np.array([[[1.]],[[3.]],[[5.]]]),future_mask=np.ones((3,1),bool),structure_id=np.array([1,1,2]))
        mean=np.zeros((3,1));matched=matched_calibrated_variance(dev,mean)
        np.testing.assert_allclose(matched,[15.])
        raw_heteroscedastic=np.array([[1.],[4.],[9.]])
        multiplicative=raw_heteroscedastic*calibrate_variance(dev,mean,raw_heteroscedastic)
        self.assertGreater(np.ptp(multiplicative),0.)
        np.testing.assert_allclose(np.broadcast_to(matched,mean.shape),np.full((3,1),15.))
        # Evaluation rows are irrelevant and are not accepted by the estimator.
        dev['evaluation_Y']=np.full((100,1,1),1e50)
        np.testing.assert_array_equal(matched_calibrated_variance(dev,mean),matched)
        dev['evaluation_Y']*=1e30
        np.testing.assert_array_equal(matched_calibrated_variance(dev,mean),matched)

    def test_same_small_neural_path_learns_deterministic_control(self):
        report=known_learnable_control()
        self.assertTrue(report['passed'],report)
        self.assertLessEqual(report['accounting']['parameter_count'],50000)
        self.assertTrue(report['same_neural_path'])
        self.assertEqual(report['accounting']['training_circuits'],4)
        self.assertLessEqual(report['accounting']['selected_step'],report['accounting']['optimizer_updates'])


class PipelineIntegrationTests(unittest.TestCase):
    def test_numeric_fit_score_replication_and_resume(self):
        import tempfile
        from pathlib import Path
        from experiments.eeg_representation.h1.artifacts import atomic_npz
        from experiments.eeg_representation.storage import atomic_json,file_hash,read_json
        from experiments.eeg_representation.h1.v4.models import fit
        from experiments.eeg_representation.h1.v4.scoring import score
        from experiments.eeg_representation.h1.v4.replication import replication
        with tempfile.TemporaryDirectory(prefix='v4-test-') as temp:
            root=Path(temp);job='A1__T0_l2_d32';directory=root/'derived'/job;rng=np.random.default_rng(29);files={}
            for split,n,offset,m in [('train',32,0,1),('development',16,2,1),('evaluation',8,4,4)]:
                x=rng.normal(size=(n,64));coefficient=np.array([[[1.,.8,.4]]])
                y=x[:,-1,None,None]*coefficient+rng.normal(size=(n,m,3))*.1
                sid=np.repeat(np.arange(offset,offset+2),n//2)
                ids=np.array([f's{c}-p{i//4:04d}-R{i%4}' for i,c in enumerate(sid)])
                ref=x[:,-1,None,None]*coefficient+rng.normal(size=(n,4,3))*.1
                atomic_npz(directory/f'{split}.npz',X=x,X_physical=x,Y=y,Y_physical=y,
                    reference_Y=ref,reference_Y_physical=ref,future_mask=np.ones((n,m),bool),
                    reference_mask=np.ones((n,4),bool),structure_id=sid,context_id=ids,
                    regime=np.array([f'R{i%4}' for i in range(n)]),
                    future_id=np.array([[f'{cid}-score-{j}' for j in range(m)] for cid in ids]),
                    reference_future_id=np.array([[f'{cid}-ref-{j}' for j in range(4)] for cid in ids]))
                files[split]={'sha256':file_hash(directory/f'{split}.npz')}
            contract=dict(targetphysicalhash='x',population_id='p',acquisitionhash='a',normalization_id='n',units='V',dimension=3,support=[2,32])
            atomic_json(directory/'manifest.json',dict(normalization=dict(fit_split='train',y_offset=[0,0,0],y_scale=1),target_fingerprint='t',risk_contract=contract,files=files))
            atomic_json(root/'derived/manifest.json',dict(jobs=[dict(id=job,status='complete')]))
            config={'models':{'neural_enabled':False},'scoring':{'circuit_bootstrap':12,'hierarchical_bootstrap':12},'replication':{'repetitions':3,'bootstrap':4}}
            fit(config,root);score(config,root);replication(config,root)
            fitted=read_json(root/'models'/job/'manifest.json')
            self.assertEqual(fitted['complete_ridge_candidate_count'],14)
            self.assertEqual(len(fitted['candidates']),19)
            self.assertTrue(fitted['selection_does_not_use_evaluation'])
            for candidate in fitted['candidates']:
                self.assertNotIn('evaluation',candidate['split_scores'])
            self.assertTrue(fit(config,root,resume=True)['jobs'][0]['resumed'])
            self.assertTrue(score(config,root,resume=True)['jobs'][0]['resumed'])
            self.assertTrue(replication(config,root,resume=True)['jobs'][0]['resumed'])
            self.assertEqual(read_json(root/'replication'/job/'summary.json')['n_independent_reference_contexts'],8)
            scored=read_json(root/'scores'/job/'summary.json')
            self.assertEqual(sum(r['calibration']=='matched_calibrated' for r in scored['rows']),19)
            for candidate in fitted['candidates']:
                with np.load(root/'models'/job/candidate['id']/'development.npz',allow_pickle=False) as f:
                    dv_mean=f['mean'];dv_variance=f['variance_matched_calibrated']
                with np.load(directory/'development.npz',allow_pickle=False) as f:
                    dv={k:f[k] for k in f.files}
                expected=matched_calibrated_variance(dv,dv_mean)
                np.testing.assert_allclose(dv_variance,np.broadcast_to(expected,dv_mean.shape))
                with np.load(root/'models'/job/candidate['id']/'evaluation.npz',allow_pickle=False) as f:
                    np.testing.assert_allclose(f['variance_matched_calibrated'],np.broadcast_to(expected,f['mean'].shape))


if __name__=='__main__':unittest.main()
