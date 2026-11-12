import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from experiments.eeg_representation.h1.targets import TargetSpec
from experiments.eeg_representation.h1.scoring import fork_risk_components,gaussian_scores,score_panel,circuit_summary
from experiments.eeg_representation.h1.benchmark import (_fit_candidates,_make_network,_validate_panel,_one_future_panel,run_benchmark,get_representations,load_numeric_model,training_diagnostics,Predictor)


def fixture():
    rng=np.random.default_rng(12031);target=TargetSpec(cutoff_ms=1000,history_s=.064,lead_ms=2,duration_ms=16,output_grid_hz=500)
    manifest=dict(job_id='control',target_spec=target.to_dict(),target_fingerprint=target.fingerprint,comparison_fingerprint='matched-control-v1',operator=target.operator,lead_ms=2.,duration_ms=16.,output_grid_hz=500.,input=dict(sample_rate_hz=500.,history_s=.064),normalization=dict(x_offset=0.,x_scale=1.,y_offset=[0.]*8,y_scale=1.,fit_split='train',id='control-training-only'),files={})
    panel={'manifest':manifest}
    for split,n,m,structures in [('train',24,1,[1,2]),('development',12,1,[3,4]),('evaluation',8,8,[5,6])]:
        x=rng.normal(size=(n,32));mu=x[:,-1,None]*(.8**np.arange(1,9))[None];y=mu[:,None]+rng.normal(size=(n,m,8))*.03
        context=np.asarray([f'{split}-{i}' for i in range(n)]);future=np.asarray([[f'{split}-{i}-future-{j}' for j in range(m)] for i in range(n)])
        panel[split]=dict(X=x,X_physical=x.copy(),Y=y,Y_physical=y.copy(),future_mask=np.ones((n,m),bool),context_id=context,structure_id=np.repeat(structures,n//2),future_id=future,reference_Y=np.empty((n,0,8)),reference_mask=np.zeros((n,0),bool),regime=np.asarray(['Ralpha']*n),input_time_ms=np.arange(-31,1)*2.,target_time_ms=np.arange(1,9)*2+2.)
    return panel


def config():
    return dict(model=dict(embedding_dim=8,layers=1,heads=2,patch_samples=4),benchmark=dict(seeds=[0],ridge_alphas=[.1],lags=[8],state_ranks=[2],harmonic_frequencies_hz=[10.],harmonic_history_fractions=[1.],steps=12,batch_size=8,learning_rates=[.003],draws=8,diagnostic_steps=3,diagnostic_examples=4,representation_steps=3,threads=1),compression=dict(dimensions=[2,8]))


class MatchedScoringTests(unittest.TestCase):
    def test_finite_repeats_identity_negative_excess_and_m1(self):
        y=np.asarray([[[-1.],[1.]],[[2.],[0.]]]);mask=np.asarray([[1,1],[1,0]],bool)
        r=fork_risk_components(y,np.zeros((2,1)),mask)
        self.assertEqual(r['mean_risk'][0],1.)
        self.assertEqual(r['state_floor'][0],2.)
        self.assertEqual(r['corrected_excess'][0],-1.)
        self.assertEqual(r['corrected_loo_risk'][0],2.)
        self.assertTrue(np.isnan(r['state_floor'][1]));self.assertTrue(np.isnan(r['corrected_excess'][1]))

    def test_known_gaussian_risk_units_and_raw_reaggregation(self):
        rng=np.random.default_rng(41);y=rng.normal(size=(60,20,3))*2;mu=np.ones((60,3))*.5;var=np.full_like(mu,4.)
        report,a=score_panel(y,mu,var,np.zeros(3),np.repeat([1,2,3],20),target_fingerprint='A',prediction_fingerprint='A',y_scale=3.,draws=8)
        self.assertLess(abs(a['context__state_floor'].mean()-4),.25)
        self.assertLess(abs(a['context__corrected_excess'].mean()-.25),.15)
        np.testing.assert_allclose(a['context__mean_risk_physical'],9*a['context__mean_risk'])
        np.testing.assert_allclose(a['context__gaussian_nll_physical'],a['context__gaussian_nll']+np.log(3))
        rebuilt=circuit_summary({k[len('context__'):]:v for k,v in a.items() if k.startswith('context__')},np.repeat([1,2,3],20),bootstrap=0)
        self.assertEqual(rebuilt['summary']['mean_risk']['mean'],report['summary']['mean_risk']['mean'])
        with self.assertRaisesRegex(ValueError,'fingerprint'):score_panel(y,mu,var,np.zeros(3),np.repeat([1,2,3],20),target_fingerprint='A',prediction_fingerprint='B')

    def test_population_risk_ratio_is_not_mean_conditional_ratio(self):
        report=circuit_summary(dict(mean_risk=np.array([1.,10.]),r0=np.array([1.,100.]),context_risk_ratio=np.array([1.,.1]),context_skill=np.array([0.,.9])),[1,2],bootstrap=0)
        self.assertAlmostEqual(report['summary']['risk_over_r0']['mean'],11/101)
        self.assertAlmostEqual(report['summary']['skill']['mean'],1-11/101)
        self.assertAlmostEqual(report['summary']['context_risk_ratio']['mean'],.55)
        same_circuit=circuit_summary(dict(mean_risk=np.array([1.,10.]),r0=np.array([1.,100.])),[1,1],bootstrap=0)
        self.assertAlmostEqual(same_circuit['per_circuit'][0]['risk_over_r0'],11/101)

    def test_gaussian_proper_scores_and_calibration(self):
        rng=np.random.default_rng(3);y=rng.normal(size=(2,500,4));mean=np.zeros((2,4))
        good=gaussian_scores(y,mean,np.ones_like(mean),draws=32);bad=gaussian_scores(y,mean,np.ones_like(mean)*9,draws=32)
        self.assertLess(good['gaussian_nll'].mean(),bad['gaussian_nll'].mean())
        self.assertLess(good['marginal_crps'].mean(),bad['marginal_crps'].mean())
        self.assertLess(good['energy'].mean(),bad['energy'].mean())
        self.assertLess(abs(good['coverage_95'].mean()-.95),.02)
        self.assertLess(abs(good['pit'].mean()-.5),.02)
        self.assertEqual(good['draws'].shape,(32,2,4))


    def test_partial_reference_pool_remains_unavailable_where_missing(self):
        y=np.zeros((2,3,1));ref=np.ones((2,2,1));mask=np.array([[1,1],[0,0]],bool)
        report,arrays=score_panel(y,np.zeros((2,1)),np.ones((2,1)),np.zeros(1),[1,2],target_fingerprint='A',reference_futures=ref,reference_mask=mask,draws=4)
        self.assertEqual(arrays['context__independent_reference_mean_risk'][0],1.)
        self.assertTrue(np.isnan(arrays['context__independent_reference_mean_risk'][1]))
        with self.assertRaisesRegex(ValueError,'context/population'):
            score_panel(y,np.zeros((2,1)),np.ones((2,1)),np.zeros(1),[1,2],target_fingerprint='A',comparison_fingerprint='C1',prediction_comparison_fingerprint='C2')


class BenchmarkTests(unittest.TestCase):
    def test_simple_learnable_control_and_no_evaluation_target_tuning(self):
        p=fixture();c=config();a,record=_fit_candidates(c,p,'ridge_wiener');pred,_=a.predict(p['development'])
        mse=((pred-p['development']['Y'][:,0])**2).mean();r0=(p['development']['Y'][:,0]**2).mean()
        self.assertLess(mse/r0,.05)
        other=copy.deepcopy(p);other['evaluation']['Y'][:]=1e9;b,_=_fit_candidates(c,other,'ridge_wiener')
        np.testing.assert_array_equal(a.state['weight'],b.state['weight'])
        self.assertEqual(record['selection_split'],'development')
        p['evaluation']['structure_id'][0]=1
        with self.assertRaisesRegex(ValueError,'overlap'):_validate_panel(p)

    def test_persistence_operator_alignment_and_exact_harmonic_bin_means(self):
        from experiments.eeg_representation.h1.targets import operator_from_projected
        p=fixture();data=p['evaluation']
        for operator in ('native_waveform','bin_average_waveform','rms','log_total_power','full_complex_fft','magnitude_spectrum'):
            spec=TargetSpec.from_dict(dict(p['manifest']['target_spec'],operator=operator))
            manifest=dict(p['manifest'],target_spec=spec.to_dict(),normalization=dict(p['manifest']['normalization'],y_offset=[0.]*spec.dimension))
            model=Predictor('persistence',manifest=manifest);pred,_=model.predict(data)
            n=256 if operator=='native_waveform' else 8
            for i in range(len(data['X'])):np.testing.assert_allclose(pred[i],operator_from_projected(np.full(n,data['X_physical'][i,-1]),spec))
        data=dict(data);time=data['input_time_ms']/1000;data['X_physical']=np.tile(.2+np.sin(2*np.pi*10*time),(8,1))
        model=Predictor('harmonic',dict(frequencies=np.array([10.])),dict(history_fraction=1.),p['manifest']);pred,_=model.predict(data)
        midpoint=(2+np.arange(1,9)*2-1)/1000
        expected=.2+np.sin(2*np.pi*10*midpoint)*np.sinc(10*.002)
        np.testing.assert_allclose(pred,np.tile(expected,(8,1)),rtol=1e-12,atol=1e-12)

    def test_pair_same_encoder_and_mean_initialization(self):
        cfg=dict(architecture='transformer',channels=1,embedding_dim=8,layers=1,heads=2,patch_samples=4)
        a=_make_network(cfg,8,False,11);b=_make_network(cfg,8,True,11)
        for name,value in a.state_dict().items():np.testing.assert_array_equal(value.detach().numpy(),b.state_dict()[name].detach().numpy())
        import torch
        mean,var=b(torch.zeros((2,1,32)));self.assertTrue(bool((var>0).all()))
        self.assertEqual(mean.shape,(2,8))

    def test_first_future_policy_is_explicit(self):
        p=fixture();p['train']=copy.deepcopy(p['evaluation']);p['train']['future_mask'][0,0]=False
        q=_one_future_panel(p)
        np.testing.assert_array_equal(q['train']['Y'][0,0],p['train']['Y'][0,1]);self.assertEqual(q['train']['Y'].shape,(8,1,8))
        self.assertEqual(p['train']['Y'].shape,(8,8,8))

    def test_prespecified_unit_variance_reference_and_first_future_parity(self):
        p=fixture();c=config();d=p['train'];original=d['Y'].copy()
        d['Y']=np.concatenate([original,original+1000,original-2000],axis=1)
        d['Y_physical']=d['Y'].copy();d['future_mask']=np.ones((len(original),3),bool)
        d['future_mask'][0,0]=False;d['future_mask'][1,1:]=False
        d['future_id']=np.asarray([[f'train-{i}-{j}' for j in range(3)] for i in range(len(original))])
        expected=original[:,0].copy();expected[0]=d['Y'][0,1]
        empirical,empirical_fit=_fit_candidates(c,p,'train_mean')
        unit,unit_fit=_fit_candidates(c,p,'train_mean_unit_variance')
        np.testing.assert_array_equal(unit.state['mean'],expected.mean(0))
        np.testing.assert_array_equal(empirical.state['mean'],unit.state['mean'])
        np.testing.assert_allclose(empirical.state['variance'],expected.var(0,ddof=1))
        np.testing.assert_array_equal(unit.state['variance'],np.ones(8))
        self.assertEqual(unit_fit['accounting']['parameter_count'],8)
        self.assertEqual(empirical_fit['accounting']['parameter_count'],16)
        self.assertEqual(unit_fit['accounting']['fixed_variance_coordinates'],8)
        self.assertEqual(unit_fit['accounting']['calibration_development_contexts'],0)
        self.assertEqual(unit_fit['search_count'],1)
        p['development']['Y'][:]=1e12;p['evaluation']['Y'][:]=-1e12
        other,_=_fit_candidates(c,p,'train_mean_unit_variance')
        np.testing.assert_array_equal(unit.state['mean'],other.state['mean'])
        np.testing.assert_array_equal(unit.state['variance'],other.state['variance'])

    def test_all_models_numeric_save_reload_and_corrupt_resume(self):
        p=fixture();c=config()
        with tempfile.TemporaryDirectory() as d:
            with patch('experiments.eeg_representation.h1.dataset.list_jobs',return_value=['control']),patch('experiments.eeg_representation.h1.dataset.load_panel',return_value=p):
                summary=run_benchmark(c,d);self.assertEqual(summary['completed_jobs'],1)
                root=Path(d)/'benchmarks/control'
                with np.load(root/'predictions.npz',allow_pickle=False) as f:
                    self.assertEqual(f['mean'].shape,(8,8,8));self.assertEqual(len(f['model_names']),8)
                    unit_index=list(f['model_names']).index('train_mean_unit_variance')
                    empirical_index=list(f['model_names']).index('train_mean')
                    np.testing.assert_array_equal(f['variance'][unit_index],np.ones((8,8)))
                    np.testing.assert_array_equal(f['mean'][unit_index],f['mean'][empirical_index])
                with np.load(root/'scores/train_mean_unit_variance.npz',allow_pickle=False) as f:
                    self.assertTrue(np.isfinite(f['context__gaussian_nll']).all())
                    self.assertTrue(np.isfinite(f['context__marginal_crps']).all())
                for modeldir in (root/'models').iterdir():
                    model,record=load_numeric_model(modeldir);mean,var=model.predict(p['evaluation']);self.assertTrue(np.isfinite(mean).all());self.assertTrue((var>0).all())
                    with np.load(modeldir/'model.npz',allow_pickle=False) as f:self.assertTrue(all(f[k].dtype.kind!='O' for k in f.files))
                self.assertEqual(run_benchmark(c,d,resume=True),summary)
                with self.assertRaises(FileExistsError):run_benchmark(c,d)
                (root/'predictions.npz').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'corrupt'):run_benchmark(c,d,resume=True)

    def test_representation_banks_numeric_and_aligned(self):
        p=fixture();c=config()
        with tempfile.TemporaryDirectory() as d:
            bank=get_representations(c,d,p)
            self.assertEqual(set(bank),{'feature18','pca','untrained','masked_mse','forecast'})
            self.assertEqual(bank['feature18']['train'].shape,(24,18))
            self.assertEqual(bank['pca']['train'].shape,(24,8))
            again=get_representations(c,d,p,resume=True)
            for name in bank:np.testing.assert_array_equal(bank[name]['evaluation'],again[name]['evaluation'])

    def test_actual_encoder_few_example_overfit(self):
        p=fixture();c=config();c['benchmark']['diagnostic_steps']=250
        with tempfile.TemporaryDirectory() as d:
            result=training_diagnostics(c,p,d)
            self.assertTrue(result['exact_deterministic_control']['passed'])
            self.assertTrue(result['few_example_overfit']['passed'],result['few_example_overfit'])


if __name__=='__main__':unittest.main()
