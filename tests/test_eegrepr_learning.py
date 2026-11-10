"""Small genuine analytical arrays exercise learning, objectives and persistence.

These tests do not impersonate a biophysical integration test. The CLI real
NEURON smoke uses the same fitting and evaluation entry points separately.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'Optional PyTorch ML dependency unavailable')
class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return dict(profile='smoke',model=dict(embedding_dim=16,layers=1,heads=2,patch_samples=16),
                    training=dict(steps=4,batch_size=4,lr=.003,source_tasks=['rhythm','carrier'],
                                  heldout_tasks=['baseline_alpha'],cpu_threads=1,checkpoint_every=1),
                    evaluation=dict(tasks=['rhythm'],label_budgets=[1,2],probes=['linear','nonlinear'],
                                    probe_steps=3,nonlinear_grid=[.01],linear_grid=[1.],shuffle_diagnostic=True))

    def examples(self,split):
        rng=np.random.default_rng({'train':17,'development':18,'test':19}[split])
        count=16 if split=='train' else 8
        y=np.arange(count)%2
        t=np.arange(128)/500
        x=(.1*rng.normal(size=(count,1,128))+(y[:,None,None]*2-1)*np.sin(2*np.pi*10*t)[None,None]).astype('float32')
        # Both sign and variance labels remain present in the normalized arrays.
        x*=np.where(y[:,None,None],2.,1.)
        return dict(x=x,future=x[:,:,-16:].copy(),labels={'rhythm':y,'carrier':y,'baseline_alpha':y},
                    metadata=[dict(parent_trial_id=f'{split}-{i}',trial_id=f'{split}-{i}',
                                   structure_id=f'{split}-{i//8}',family='analytical_fixture') for i in range(count)],
                    normalization={'scale_v':[1e-12]})

    def loader(self,root,config,split,protocol,**kwargs):
        return self.examples(split)

    def test_shared_models_and_parameter_ceiling(self):
        from experiments.eeg_representation.models import make_encoder,encoder_config
        for architecture in ('transformer','tcn'):
            model=make_encoder(encoder_config({'model':{'architecture':architecture}}))
            self.assertLessEqual(sum(p.numel() for p in model.parameters()),2_000_000)
            self.assertEqual(tuple(model(torch.randn(2,1,500)).shape),(2,128))

    def test_mask_loss_has_no_unmasked_gradient(self):
        from experiments.eeg_representation.models import gap_mask,masked_loss
        x=torch.randn(2,1,100)
        mask=gap_mask(x,.4,16)
        self.assertTrue(torch.all(mask.sum(-1)==40))
        prediction=torch.zeros_like(x,requires_grad=True)
        masked_loss(prediction,x,mask).backward()
        self.assertTrue(torch.all(prediction.grad[~mask]==0))
        x2=x.clone();x2[~mask]+=100
        self.assertEqual(float(masked_loss(prediction,x,mask)),float(masked_loss(prediction,x2,mask)))

    def test_gaussian_covariance_and_scores(self):
        from experiments.eeg_representation.models import GaussianForecast,make_encoder
        from experiments.eeg_representation.evaluation import forecast_scores
        for kind in ('lowrank','diagonal','ar'):
            model=GaussianForecast(make_encoder(dict(embedding_dim=16,layers=1,heads=2,patch_samples=16)),(1,8),kind)
            d=model(torch.randn(3,1,64))
            self.assertTrue(torch.all(d.variance>0))
            y=torch.randn(3,1,8)
            scores=forecast_scores(d,y,torch.zeros(1,8),draws=8)
            self.assertTrue(all(np.isfinite(v).all() for v in scores.values()))
            (-d.log_prob(y.flatten(1)).mean()).backward()
        mean=torch.zeros(1000,4)
        d=torch.distributions.Independent(torch.distributions.Normal(mean,torch.ones_like(mean)),1)
        torch.manual_seed(8);target=d.sample()
        score=forecast_scores(d,target,mean,draws=32)
        self.assertAlmostEqual(float(score['coverage_95'].mean()),.95,delta=.025)
        self.assertAlmostEqual(float(score['nll_per_dimension'].mean()),.5*np.log(2*np.pi)+.5,delta=.08)

    def test_trial_budgets_nested_and_order_exposures_exact(self):
        from experiments.eeg_representation.training import independent_budget_indices,exposure_schedule
        examples=self.examples('train')
        small,_=independent_budget_indices(examples['metadata'],examples['labels']['rhythm'],'classification',2,0)
        large,_=independent_budget_indices(examples['metadata'],examples['labels']['rhythm'],'classification',4,0)
        self.assertTrue(set(small)<=set(large))
        selections={'rhythm':np.arange(13),'carrier':np.arange(8)}
        summaries=[]
        for recipe in ('joint','progressive','reverse','random_order','reconstruction_supervised'):
            _,summary=exposure_schedule(selections,9,4,3,recipe)
            summaries.append(summary)
        self.assertEqual(len({s['ticket_hash'] for s in summaries}),1)
        self.assertEqual(len({s['examples_processed'] for s in summaries}),1)
        self.assertGreater(len({s['schedule_hash'] for s in summaries}),1)

    def test_paired_regime_siblings_cost_one_independent_label(self):
        from experiments.eeg_representation.training import independent_budget_indices
        metadata=[];labels=[]
        for family in range(4):
            for regime in range(4):
                metadata.append(dict(parent_trial_id=f'p{family}-r{regime}',paired_family_id=f'p{family}',structure_id='train'))
                labels.append(int(regime>0))
        indices,report=independent_budget_indices(metadata,np.asarray(labels),'classification',2,0)
        self.assertEqual(report['available_per_condition'],{'0':4,'1':4})
        self.assertEqual(len(indices),8)
        self.assertEqual(len(report['selected_independent_families']),2)
        self.assertEqual(len(report['selected_parents']),8)

    def test_variance_classifier_and_shuffled_label_control(self):
        from experiments.eeg_representation.training import feature_matrix
        from experiments.eeg_representation.evaluation import _linear_probe
        rng=np.random.default_rng(114)
        y=np.arange(800)%2;x=rng.normal(size=(800,1,128))*np.where(y[:,None,None],2.,.5)
        features=feature_matrix(x)
        model=_linear_probe(features[:400],y[:400],'classification',1.,0)
        self.assertGreater(np.mean(model.predict(features[400:])==y[400:]),.95)
        shuffled=y[:400].copy();rng.shuffle(shuffled)
        model=_linear_probe(features[:400],shuffled,'classification',1.,0)
        self.assertLess(np.mean(model.predict(features[400:])==y[400:]),.65)
        self.assertGreater(np.mean(model.predict(features[400:])==y[400:]),.35)

    def test_all_recipes_save_reload_evaluate_and_resume(self):
        from experiments.eeg_representation.training import RECIPES,fit,load_fitted
        from experiments.eeg_representation.evaluation import evaluate
        cfg=self.config()
        with tempfile.TemporaryDirectory() as temp,patch('experiments.eeg_representation.training._load_examples',side_effect=self.loader),patch('experiments.eeg_representation.evaluation._load_examples',side_effect=self.loader):
            accounting=[]
            for recipe in RECIPES:
                result=fit(cfg,Path(temp),recipe,0)
                if recipe not in {'features','specialist'}:
                    accounting.append(result['accounting'])
                self.assertEqual(result['status'],'complete')
                self.assertGreaterEqual(result['parameter_count'],0)
                if recipe!='features':
                    model,payload=load_fitted(result['run_dir'])
                    self.assertTrue(torch.isfinite(model.encoder(torch.tensor(self.examples('test')['x']))).all())
                    self.assertTrue((Path(result['run_dir'])/'encoder.pt').exists())
                again=fit(cfg,Path(temp),recipe,0,resume=True)
                self.assertEqual(result['config_hash'],again['config_hash'])
                with self.assertRaises(FileExistsError):
                    fit(cfg,Path(temp),recipe,0)
            self.assertEqual(len({r['ticket_hash'] for r in accounting}),1)
            self.assertEqual(len({r['examples_processed'] for r in accounting}),1)
            self.assertEqual(len({r['optimizer_updates'] for r in accounting}),1)
            result=evaluate(cfg,Path(temp))
            self.assertTrue(result['rows'])
            self.assertEqual({r['recipe'] for r in result['rows']},set(RECIPES))
            forecasts=json.loads((Path(temp)/'evaluations/leave-task-label-out/forecasts.json').read_text())
            self.assertEqual(len(forecasts),3)
            changed=copy.deepcopy(cfg);changed['training']['lr']=.01
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                fit(changed,Path(temp),'joint',0,resume=True)

    def test_missing_labels_masked_and_resume_after_interruption(self):
        from experiments.eeg_representation.training import fit,load_fitted
        cfg=self.config()
        original=torch.optim.AdamW.step
        calls=[0]
        def failing_step(optimizer,*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:
                raise RuntimeError('deliberate interruption fixture')
            return original(optimizer,*args,**kwargs)
        def missing(*args,**kwargs):
            values=self.loader(*args,**kwargs)
            values['labels']['carrier']=values['labels']['carrier'].copy()
            values['labels']['carrier'][::2]=-1
            return values
        with tempfile.TemporaryDirectory() as temp,patch('experiments.eeg_representation.training._load_examples',side_effect=missing):
            with patch.object(torch.optim.AdamW,'step',failing_step):
                with self.assertRaisesRegex(RuntimeError,'deliberate'):
                    fit(cfg,temp,'joint',0)
            result=fit(cfg,temp,'joint',0,resume=True)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['accounting']['unique_examples']['carrier'],8)
            model,payload=load_fitted(result['run_dir'])
            self.assertEqual(payload['completed'],result['optimizer_updates'])

    def test_optional_adaptation_and_order_repeats(self):
        from experiments.eeg_representation.training import fit_suite
        from experiments.eeg_representation.evaluation import evaluate
        cfg=self.config();cfg['training']['random_orders']=3
        cfg['evaluation']['adaptation_steps']=2
        cfg['evaluation']['variants']=['clean','snr_10'];cfg['evaluation']['gains']=[1.,2.]
        with tempfile.TemporaryDirectory() as temp,patch('experiments.eeg_representation.training._load_examples',side_effect=self.loader),patch('experiments.eeg_representation.evaluation._load_examples',side_effect=self.loader):
            results=fit_suite(cfg,temp,recipes=['random_order'],seeds=[0])
            self.assertEqual(len(results),2)
            self.assertEqual(len({r['accounting']['ticket_hash'] for r in results}),1)
            self.assertEqual(len({tuple(r['accounting']['order']) for r in results}),2)
            result=evaluate(cfg,temp)
            self.assertIn('random_init_specialist',{r['probe'] for r in result['rows']})
            self.assertIn('limited_label_finetune',{r['probe'] for r in result['rows']})
            normal=[r for r in result['rows'] if r['probe']=='linear']
            self.assertTrue(all(len(r['robustness'])==3 for r in normal))
            self.assertTrue((Path(temp)/'evaluations/leave-task-label-out/aggregate_transfer.json').exists())

    def test_losses_learn_deterministic_control(self):
        from experiments.eeg_representation.training import fit
        cfg=self.config();cfg['training']['steps']=50
        def deterministic(*args,**kwargs):
            data=self.loader(*args,**kwargs)
            # A deterministic two-variance rhythm permits reconstruction and forecasting.
            time=np.arange(128)/500
            y=data['labels']['rhythm']
            data['x']=(np.sin(2*np.pi*10*time)[None,None,:]*(1+y[:,None,None])).astype('float32')
            data['future']=data['x'][:,:,-16:].copy()
            return data
        with tempfile.TemporaryDirectory() as temp,patch('experiments.eeg_representation.training._load_examples',side_effect=deterministic):
            for recipe in ('specialist','masked_mse','masked_l1','spectral_hybrid','forecast_lowrank','forecast_diagonal','forecast_ar','momentum_contrastive'):
                result=fit(cfg,temp,recipe,7)
                curve=json.loads((Path(result['run_dir'])/'learning_curve.json').read_text())
                initial=np.median([r['loss'] for r in curve[:5]])
                final=np.median([r['loss'] for r in curve[-5:]])
                self.assertLess(final,initial,msg=recipe)

    def test_structure_leakage_refused(self):
        from experiments.eeg_representation.evaluation import evaluate
        with tempfile.TemporaryDirectory() as temp,patch('experiments.eeg_representation.evaluation._load_examples',return_value=self.examples('train')):
            with self.assertRaisesRegex(ValueError,'leakage'):
                evaluate(self.config(),temp)

    def test_temporal_probe_uses_final_causal_cutoff_only(self):
        from experiments.eeg_representation.evaluation import trial_pool,trial_waveforms
        features=np.asarray([[1.,2.],[3.,4.],[5.,6.]])
        y=np.asarray([9.,8.,7.])
        metadata=[dict(parent_trial_id='p',structure_id='s',cutoff_index=i) for i in [10,20,30]]
        z,target,_=trial_pool(features,y,metadata,'regression',causal_target=True)
        np.testing.assert_array_equal(z,features[-1:])
        np.testing.assert_array_equal(target,y[-1:])
        example=dict(x=features[:,None,:],labels={'envelope':y},metadata=metadata)
        waves,target,_=trial_waveforms(example,'envelope','regression')
        np.testing.assert_array_equal(waves[0],features[-1:,None,:])
        np.testing.assert_array_equal(target,y[-1:])

    def test_structure_cluster_area_and_fdr(self):
        from experiments.eeg_representation.evaluation import normalized_log_budget_area,paired_cluster_inference,fdr_bh
        self.assertAlmostEqual(normalized_log_budget_area([10,100,1000],[.5,.7,.9]),.7)
        result=paired_cluster_inference({'a':.6,'b':.7,'c':.8},{'a':.5,'b':.5,'c':.5})
        self.assertEqual(result['n_structures'],3)
        self.assertAlmostEqual(result['mean'],.2)
        self.assertTrue(all(v>=p for v,p in zip(fdr_bh([.01,.03,.9]),[.01,.03,.9])))


if __name__=='__main__':
    unittest.main()
