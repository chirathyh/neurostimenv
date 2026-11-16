"""Cheap controls for frozen full-history means and independent variance fitting."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.eeg_representation.h1.v6 import models, calibration, retention, scoring
from experiments.eeg_representation.h1.targets import canonical_hash
from experiments.eeg_representation.storage import atomic_json,read_json,file_hash
from experiments.eeg_representation.h1.artifacts import atomic_npz


def panel_fixture(root,length=64,evaluation_shift=0.,calibration_shift=0.,futures=1):
    """Independent circuits, four conditions, two prefixes, known early-history signal."""
    rng=np.random.default_rng(891);panels={}
    for split,circuits in [('train',range(100,104)),('development',range(104,106)),('calibration',range(106,108)),('evaluation',range(108,110))]:
        rows=[(c,r,p) for c in circuits for r in range(4) for p in range(2)]
        n=len(rows);x=rng.normal(size=(n,length));y=.5*x[:,:length//2].mean(1)+.15*rng.normal(size=n)
        if split=='evaluation':y+=evaluation_shift
        if split=='calibration':y+=calibration_shift
        ids=np.array([f's{c}-p{p:04d}-R{r}' for c,r,p in rows]);fids=np.array([[v+'-score-0000'] for v in ids])
        panels[split]=dict(X=x,X_physical=x*2e-9+1e-9,Y=y[:,None,None],Y_physical=y[:,None,None]*3e-9,
            future_mask=np.ones((n,1),bool),future_id=fids,reference_future_id=np.empty((n,0),dtype='U1'),
            reference_mask=np.empty((n,0),bool),structure_id=np.array([c for c,r,p in rows]),context_id=ids,
            regime=np.array([f'R{r}' for c,r,p in rows]),task_carrier=np.array([r-1 for c,r,p in rows]),task_rhythm=np.array([r>0 for c,r,p in rows],int))
    if futures>1:
        for split in ('calibration','evaluation'):
            p=panels[split];n=len(p['X']);p['Y']=p['Y']+rng.normal(0,.1,(n,futures,1));p['Y_physical']=p['Y']*3e-9
            p['future_mask']=np.ones((n,futures),bool);p['future_id']=np.asarray([[v+f'-score-{j:04d}' for j in range(futures)] for v in p['context_id']])
    job='A0__T0_l10_d128';directory=Path(root)/'derived'/job
    contract=dict(dimension=1,units='V',population_id='fixture',normalization_id='fixture-training-only',target_physical_hash='fixture',acquisition_hash='fixture',support=[6010,6138])
    manifest=dict(target_fingerprint=canonical_hash(contract),risk_contract=contract,target_family='T0',target_spec=dict(family='T0',dimension=1),
        normalization=dict(x_offset=1e-9,x_scale=2e-9,y_offset=[0.],y_scale=3e-9,fit_split='train'),
        input=dict(sample_rate_hz=500.),files={})
    for split,data in panels.items():
        path=directory/(split+'.npz');atomic_npz(path,**data)
        manifest['files'][split]=models._record(Path(root),path)
    atomic_json(directory/'manifest.json',manifest)
    atomic_json(Path(root)/'derived/manifest.json',dict(jobs=[dict(id=job,status='complete',**models._record(Path(root),directory/'manifest.json'))]))
    config=dict(claim_status='engineering',selected_job_ids=[job],models=dict(neural_enabled=False,run_controls=False,full_history_samples=length,pca_width=8))
    return panels,manifest,config,job


class ModelTests(unittest.TestCase):
    def test_full_history_receptive_field_and_early_input(self):
        cfg=models.settings(dict(claim_status='prospective'))
        net=models.make_tcn(64,np.zeros(64),cfg,seed=3)
        self.assertLessEqual(sum(x.numel() for x in net.parameters()),50000)
        self.assertGreaterEqual(1+2*sum(cfg['neural_dilations']),2000)
        import torch
        with torch.no_grad():net.head.weight.fill_(.1)
        a=np.zeros((2,2000));b=a.copy();b[:,:100]+=1.
        self.assertGreater(np.max(np.abs(models.neural_predict(net,a,cfg,'cpu')-models.neural_predict(net,b,cfg,'cpu'))),1e-5)
        with self.assertRaisesRegex(ValueError,'complete 14'):models.settings(dict(models=dict(ridge_alphas=[1.])))
        with self.assertRaisesRegex(ValueError,'fixed to 2000'):models.settings(dict(models=dict(full_history_samples=64)))
        with self.assertRaisesRegex(ValueError,'five distinct'):models.settings(dict(models=dict(neural_seeds=[0,1,2])))

    def test_same_neural_path_learns_and_counts_actual_ids(self):
        cfg=models.settings(dict(claim_status='engineering',models=dict(full_history_samples=64,neural_width=4,neural_dilations=[1,2,4,8,16,32],control_steps=120,max_steps=60,eval_every=10,patience=60)))
        result=models.known_learnable_control(cfg)
        self.assertTrue(result['passed'],result)
        self.assertGreater(result['early_half_perturbation_max_change'],1e-6)
        with tempfile.TemporaryDirectory() as d:
            panels,_,_,_=panel_fixture(d)
            diag,arrays=models.few_example_diagnostic(panels['train'],cfg)
            self.assertEqual(diag['actual_context_count'],4)
            self.assertEqual(diag['actual_circuit_count'],1)
            self.assertEqual(diag['context_ids'],panels['train']['context_id'][:4].tolist())
            self.assertEqual(arrays['prediction'].dtype,np.float64)

    def test_dev_selection_complete_grid_numeric_inference_and_resume(self):
        with tempfile.TemporaryDirectory() as d,tempfile.TemporaryDirectory() as other:
            panel,manifest,config,job=panel_fixture(d)
            result=models.fit(config,d)
            fitted=read_json(Path(d)/'models'/job/'manifest.json')
            self.assertEqual(fitted['complete_ridge_candidate_count'],14)
            tails=[r for r in fitted['candidates'] if r['id'].startswith('ridge_tail')]
            self.assertEqual(len(tails),14)
            self.assertEqual({r['alpha'] for r in tails},set(models.DEFAULTS['ridge_alphas']))
            self.assertTrue(all(not r['evaluation_outcomes_accessed'] and not r['calibration_outcomes_accessed'] for r in fitted['candidates']))
            self.assertEqual(fitted['fp64_sensitivity'][0]['status'],'complete')
            output=models.predict_from_artifact(d,job,X_physical=panel['evaluation']['X_physical'])
            selected=fitted['selected_member_ids'][0]
            with np.load(Path(d)/'models'/job/selected/'evaluation.npz',allow_pickle=False) as z:
                np.testing.assert_allclose(output['mean_standardized'][0],z['mean'],rtol=1e-12,atol=1e-12)
                self.assertEqual(z['mean'].dtype,np.float64)
            self.assertTrue(models.fit(config,d,resume=True)['jobs'][0]['resumed'])
            altered=copy.deepcopy(config);altered['models']['dtype']='float64'
            with self.assertRaisesRegex(ValueError,'identity mismatch'):models.fit(altered,d,resume=True)
            panel2,_,config2,_=panel_fixture(other,evaluation_shift=999.,calibration_shift=-999.)
            models.fit(config2,other)
            fit2=read_json(Path(other)/'models'/job/'manifest.json')
            self.assertEqual(fit2['selected_candidate'],fitted['selected_candidate'])
            for a,b in zip(fitted['candidates'],fit2['candidates']):
                self.assertEqual(a['development_mse'],b['development_mse'])
                self.assertEqual(a['shrinkage'],b['shrinkage'])
                with np.load(Path(d)/a['state_file'],allow_pickle=False) as x,np.load(Path(other)/b['state_file'],allow_pickle=False) as y:
                    for key in x.files:np.testing.assert_array_equal(x[key],y[key])
            state=Path(d)/fitted['candidates'][0]['state_file'];state.write_bytes(state.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):models.predict_from_artifact(d,job,X=panel['evaluation']['X'],candidate='train_mean')

    def test_independent_variance_fits_frozen_means(self):
        with tempfile.TemporaryDirectory() as d:
            panel,manifest,config,job=panel_fixture(d,calibration_shift=2.)
            models.fit(config,d)
            selection=Path(d)/'models'/job/'selection.json';before=file_hash(selection)
            result=calibration.calibrate(config,d)
            self.assertEqual(before,file_hash(selection))
            out=read_json(Path(d)/'calibration'/job/'manifest.json')
            self.assertFalse(out['selection_uses_calibration'])
            self.assertEqual(out['calibration_circuit_ids'],[106,107])
            with np.load(Path(d)/'models'/job/'train_mean/calibration.npz',allow_pickle=False) as p,np.load(Path(d)/'calibration'/job/'train_mean/variance.npz',allow_pickle=False) as v:
                expected=models.matched_variance(panel['calibration'],p['mean'])
                np.testing.assert_array_equal(v['variance'],expected)
            self.assertTrue(calibration.calibrate(config,d,resume=True)['jobs'][0]['resumed'])
            conf=copy.deepcopy(config);conf['claim_status']='prospective';conf['models']['full_history_samples']=2000
            with self.assertRaisesRegex(ValueError,'circuit count'):calibration.calibrate(conf,d)
            leaked=copy.deepcopy(panel);leaked['calibration']['structure_id'][0]=leaked['train']['structure_id'][0]
            with patch('experiments.eeg_representation.h1.v6.derivation.load_panel',return_value=(leaked,manifest)):
                with self.assertRaisesRegex(ValueError,'leakage'):models.load_panel(d,job)


    def test_staged_split_arrival_never_refits_or_replaces_frozen_means(self):
        with tempfile.TemporaryDirectory() as d:
            panel,manifest,config,job=panel_fixture(d,futures=4);root=Path(d);all_records=copy.deepcopy(manifest['files'])
            def available(splits):
                manifest['files']={split:models._record(root,root/'derived'/job/(split+'.npz')) for split in splits}
                atomic_json(root/'derived'/job/'manifest.json',manifest)
                atomic_json(root/'derived/manifest.json',dict(jobs=[dict(id=job,status='complete',**models._record(root,root/'derived'/job/'manifest.json'))]))
            available(['train','development'])
            fit=models.fit(config,d)
            self.assertEqual(fit['inference']['jobs'][0]['status'],'planned')
            frozen_paths=[root/'models'/job/'manifest.json',root/'models'/job/'selection.json',root/'models'/job/'fit_identity.json']
            hashes=[file_hash(p) for p in frozen_paths]
            available(['train','development','calibration'])
            pred=models.predict(config,d,resume=True)
            self.assertEqual(pred['jobs'][0]['new_splits'],['calibration'])
            calibration.calibrate(config,d)
            available(['train','development','calibration','evaluation'])
            fit=models.fit(config,d,resume=True)
            self.assertEqual(fit['inference']['jobs'][0]['new_splits'],['evaluation'])
            self.assertEqual(hashes,[file_hash(p) for p in frozen_paths])
            with np.load(root/'models'/job/'train_mean/evaluation.npz',allow_pickle=False) as z:
                self.assertEqual(z['mean'].shape,(16,1))
            changed=copy.deepcopy(panel['evaluation']);changed['X']+=1.
            atomic_npz(root/'derived'/job/'evaluation.npz',**changed)
            available(['train','development','calibration','evaluation'])
            with self.assertRaisesRegex(ValueError,'split changed'):models.predict(config,d,resume=True)


    def test_physical_energy_persistence_uses_explicit_acquired_history(self):
        cfg=models.settings({});t=np.arange(2000)/500.;x=2e-6*np.cos(2*np.pi*10*t)
        data=dict(X=x[None],X_physical=x[None]);state=dict(baseline_mean=np.zeros(1))
        spec=dict(kind='spectral_persistence')
        manifest=dict(target_spec=dict(family='T4_energy',band_hz=[8,12],include_upper=False),
            normalization=dict(y_offset=[0.],y_scale=1.))
        prediction,_=models._predict(spec,state,data,cfg,manifest,'cpu')
        np.testing.assert_allclose(prediction,[[2e-12]],rtol=1e-12,atol=0.)
        manifest['target_spec'].update(family='T4_log_energy',log_floor_v2=1e-40,log_reference_v2=1.)
        prediction,_=models._predict(spec,state,data,cfg,manifest,'cpu')
        np.testing.assert_allclose(prediction,[[np.log(2e-12)]],rtol=1e-12,atol=0.)
        data=dict(X=np.zeros((1,2000)),X_physical=np.zeros((1,2000)))
        prediction,_=models._predict(spec,state,data,cfg,manifest,'cpu')
        np.testing.assert_allclose(prediction,[[np.log(1e-40)]],rtol=1e-12,atol=0.)

    def test_equal_circuit_condition_weighting_and_shrinkage(self):
        data=dict(structure_id=np.array([1,1,1,2]),context_id=np.array(['s1-p0-R0','s1-p1-R0','s1-p0-R1','s2-p0-R0']),
            Y=np.array([1,1,1,1.])[:,None,None],future_mask=np.ones((4,1),bool))
        np.testing.assert_allclose(models.context_weights(data),[.125,.125,.25,.5])
        self.assertEqual(models.residual_shrinkage(data,np.ones((4,1))*100,np.array([0.])),.01)
        self.assertEqual(models.residual_shrinkage(data,-np.ones((4,1)),np.array([0.])),0.)


class RetentionTests(unittest.TestCase):
    def test_class_order_missing_classes_and_float64_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            panel,manifest,_,_=panel_fixture(d);panel={s:panel[s] for s in retention.SPLITS};panel['manifest']=manifest
            for s in retention.SPLITS:panel[s]['task_four_class']=panel[s]['task_carrier']+1
            values={s:np.eye(4)[panel[s]['task_four_class']] for s in retention.SPLITS}
            dest=Path(d)/'class'
            result=retention.evaluate_readout(panel,values,'four_class','all',[.01,1.,100.],dest)
            self.assertEqual(result['evaluation']['class_order'],[0,1,2,3])
            self.assertEqual(result['evaluation']['balanced_accuracy'],1.)
            self.assertEqual(len(result['candidates']),3)
            with np.load(dest/'predictions.npz',allow_pickle=False) as z:
                self.assertEqual(z['evaluation_selected_prediction'].dtype,np.float64)
                np.testing.assert_array_equal(z['class_order'],[0,1,2,3])
                self.assertEqual(z['evaluation__alpha_1p0'].dtype,np.float64)
            for s in retention.SPLITS:panel[s]['task_four_class']=np.minimum(panel[s]['task_four_class'],2)
            unavailable=retention.evaluate_readout(panel,values,'four_class','all',[1.],Path(d)/'missing')
            self.assertEqual(unavailable['status'],'unavailable')

    def test_primary_payload_raw_denominator_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            panel,manifest,config,job=panel_fixture(d)
            config['retention']=dict(job_id=job,legacy_forecast=False,dimensions=[8],code_bits=[32],range_rules=['quantile_005_995'])
            out=retention.retention(config,d)
            self.assertEqual(out['status'],'complete')
            self.assertTrue(out['primary']['denominator_estimated_from_evaluation'])
            self.assertEqual(out['primary']['independent_circuits'],2)
            raw=out['primary']['per_circuit'];baseline=np.mean([r['reconstruction_baseline_mse'] for r in raw])
            self.assertAlmostEqual(baseline,out['primary']['reconstruction_denominator'])
            for row in raw:
                self.assertAlmostEqual((row['reconstruction_mse_feature']-row['reconstruction_mse_pca'])/baseline,row['reconstruction_difference_nmse'])
            identity=[r for r in out['deterministic_reconstruction'] if r['representation']=='raw_history'][0]
            self.assertEqual(identity['mse'],0.)
            self.assertTrue(any(r['representation']=='pca' and r['code_bits']==32 for r in out['deterministic_reconstruction']))
            with np.load(Path(d)/'retention/banks/feature18/d8_B32_quantile_005_995/codes.npz',allow_pickle=False) as z:
                self.assertEqual(z['evaluation_packed'].shape,(16,4))
            self.assertEqual(retention.retention(config,d,resume=True)['signature'],out['signature'])

    def test_score_and_confirm_use_raw_h3_ratio_and_frozen_seed_policy(self):
        with tempfile.TemporaryDirectory() as d:
            panel,manifest,config,job=panel_fixture(d,futures=5)
            config['retention']=dict(job_id=job,legacy_forecast=False,dimensions=[8],code_bits=[32],range_rules=['quantile_005_995'])
            config['protocol']=dict(primary_job_id=job,test_circuits=2)
            models.fit(config,d);calibration.calibrate(config,d);retention.retention(config,d)
            scoring.score(config,d)
            report=read_json(Path(d)/'scores'/job/'summary.json')
            selected=report['selected'];self.assertEqual(selected['fitting_repeats_averaged'],1)
            self.assertFalse(selected['selected_neural_policy'])
            self.assertEqual(len(selected['member_circuit_risks']),1)
            meanrow=next(r for r in report['rows'] if r['candidate']=='train_mean')
            self.assertEqual(set(meanrow['calibration']),{'variance_uncalibrated','variance_development_calibrated','variance_independent_calibration','variance_unit'})
            with np.load(Path(d)/'scores'/job/'train_mean__risk_components.npz',allow_pickle=False) as z:
                self.assertEqual(z['per_future_mse'].shape,(16,4))
                np.testing.assert_allclose(z['mean_risk'],z['state_floor']+z['corrected_excess'])
            # Pure unit fixture: no simulation, scientific confirmation, or protocol claim.
            atomic_json(Path(d)/'protocol.json',dict(status='frozen',protocol_hash='engineering-fixture'))
            confirmation_config=copy.deepcopy(config);confirmation_config['claim_status']='prospective_confirmation'
            result=scoring.confirm_statistics(confirmation_config,d)
            self.assertEqual(result['H1_primary_family_B']['fitting_repeats_averaged'],1)
            self.assertEqual(result['H3_primary_family_A']['n_circuits'],2)
            self.assertIn('denominator uncertainty included',result['H3_primary_family_A']['reconstruction_denominator_scope'])
            # Verify five member RISKS enter family B, rather than an already-averaged
            # vector incorrectly labelled as a one-repeat result.
            selected.update(selected_neural_policy=True,fitting_repeats_averaged=5,
                member_circuit_risks=[(np.asarray(selected['model_circuit_risks'])*(1+(i-2)*.01)).tolist() for i in range(5)])
            report['selected']=selected;atomic_json(Path(d)/'scores'/job/'summary.json',report)
            result=scoring.confirm_statistics(confirmation_config,d)
            self.assertEqual(result['H1_primary_family_B']['fitting_repeats_averaged'],5)
            np.testing.assert_allclose(result['H1_primary_family_B']['model_circuit_risks'],selected['model_circuit_risks'])


if __name__=='__main__':unittest.main()
