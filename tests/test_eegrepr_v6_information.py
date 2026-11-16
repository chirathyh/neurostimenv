"""Scientific controls for information accounting; synthetic arrays are not NEURON evidence."""
import copy
import unittest
import numpy as np
from experiments.eeg_representation.h1.v6.information import summarize,paired_contrast,future_jackknife,_policies
from experiments.eeg_representation.h1.v6.replay import compare_trials
from experiments.eeg_representation.simulator import array_hash


def panel(circuits=3,prefixes=2,regimes=2,m=4):
    keys=[(s,p,r) for s in range(1,circuits+1) for p in range(prefixes) for r in range(regimes)]
    return dict(structure_id=np.array([s for s,p,r in keys]),regime=np.array(['R'+str(r) for s,p,r in keys]),
        context_id=np.array([f's{s}-p{p:04d}-R{r}' for s,p,r in keys]),
        future_id=np.array([[f's{s}-p{p:04d}-R{r}-score-{j:04d}' for j in range(m)] for s,p,r in keys]))


class InformationAccountingTests(unittest.TestCase):
    def test_physical_units_unbiased_floor_negative_excess_and_seed_losses(self):
        p=panel();n=len(p['context_id']);y=np.tile(np.array([-3,-1,1,3.])[None,:,None],(n,1,2))*1e-6
        means=np.zeros((2,n,2));means[0]+=1e-6;means[1]-=1e-6
        row,comp=summarize(y,means,np.full(2,2e-6),p,method_id='control')
        self.assertAlmostEqual(row['physical_mse']/1e-12,6)
        self.assertAlmostEqual(row['state_variance']/1e-12,20/3)
        self.assertAlmostEqual(row['combined_unresolved_excess']/1e-12,-2/3)
        self.assertAlmostEqual(row['error_baseline100'],100*6/9)
        self.assertEqual(row['negative_corrected_context_excess'],n)
        self.assertEqual(row['fitting_repeats_averaged'],2)
        self.assertEqual(row['conditional_mc']['status'],'complete')
        self.assertAlmostEqual(row['finite_m_identity_residual'],0,places=24)

    def test_M1_floor_is_unavailable_not_zero(self):
        p=panel(m=1);n=len(p['context_id']);y=np.ones((n,1,1))
        row,_=summarize(y,np.zeros((n,1)),np.zeros(1),p,method_id='single')
        self.assertIsNone(row['state_variance']);self.assertIsNone(row['combined_unresolved_excess'])
        self.assertEqual(row['conditional_mc']['status'],'unavailable')

    def test_equal_circuit_risk_ratio_and_paired_cancellation(self):
        p=panel(prefixes=1,regimes=1);y=np.array([[[1],[2],[3],[4]],[[2],[4],[6],[8]],[[3],[6],[9],[12]]],float)
        means=np.array([[.1],[2],[4.]])
        a,_=summarize(y,means,np.zeros(1),p,method_id='first')
        self.assertAlmostEqual(a['risk_ratio'],np.mean(a['circuit_components']['model'])/np.mean(a['circuit_components']['baseline']))
        out=paired_contrast(a,copy.deepcopy(a),label='identical')
        self.assertAlmostEqual(out['gain_difference']['estimate'],0)
        self.assertAlmostEqual(out['gain_difference']['standard_error'],0)
        self.assertAlmostEqual(out['conditional_mc_gain_difference_se'],0)
        b=copy.deepcopy(a);b['context_ids'][0]='different'
        with self.assertRaises(ValueError):paired_contrast(a,b,label='invalid')

    def test_joint_regime_mc_keeps_shared_stream_covariance(self):
        p=panel(circuits=2,prefixes=1,regimes=2);n=len(p['context_id'])
        y=np.tile(np.arange(1.,5.)[None,:,None],(n,1,1))
        mc=future_jackknife(y,np.zeros((n,1)),np.zeros(1),p)
        single=panel(circuits=2,prefixes=1,regimes=1)
        one=future_jackknife(y[::2],np.zeros((2,1)),np.zeros(1),single)
        self.assertAlmostEqual(mc['pooled_standard_error']['model'],one['pooled_standard_error']['model'])
        broken=copy.deepcopy(p);broken['future_id'][1]=broken['future_id'][1,::-1]
        with self.assertRaises(ValueError):future_jackknife(y,np.zeros((n,1)),np.zeros(1),broken)

    def test_short_policy_uses_development_and_exact_64_samples(self):
        records=[dict(id='train_mean',kind='mean',development_mse=3),
            dict(id='tail16',kind='ridge',history_samples=16,development_mse=0),
            dict(id='tail64',kind='ridge',history_samples=64,development_mse=2),
            dict(id='full',kind='ridge',history_samples=2000,development_mse=1)]
        fit=dict(candidates=records,selected_member_ids=['full'],policies=[dict(id='full',members=['full'],development_mse=1),dict(id='train_mean',members=['train_mean'],development_mse=3)])
        rows=_policies(fit);short=next(r for r in rows if r['id']=='eeg_short_selected')
        self.assertEqual(short['members'],['tail64']);self.assertEqual(short['history_samples'],64)


class InformationPipelineTests(unittest.TestCase):
    def test_common_population_readouts_first_four_futures_and_resume(self):
        import tempfile
        from pathlib import Path
        from tests.test_eegrepr_v6_models import panel_fixture
        from experiments.eeg_representation.h1.v6 import information,information_models,models
        from experiments.eeg_representation.h1.v6.artifacts import read_json
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);panels,manifest,config,job=panel_fixture(root,length=2000,futures=5)
            config['information_models']=dict(forecast_enabled=False)
            models.fit(config,root)
            information_models.fit(config,root)
            result=information.run(config,root)
            self.assertEqual(result['status'],'complete')
            summary=read_json(root/'information'/job/'summary.json')
            rows={r['id']:r for r in summary['rows']}
            self.assertEqual(summary['population']['first_scoring_futures'],4)
            self.assertEqual(summary['population']['n_scoring_episodes'],len(panels['evaluation']['X'])*4)
            codes=[r for r in rows.values() if r.get('information_available')=='compressed EEG representation' and r['status']=='measured']
            self.assertEqual(len(codes),4)
            for row in codes:
                self.assertEqual(row['context_ids'],rows['train_mean']['context_ids'])
                self.assertEqual(row['future_ids'],rows['train_mean']['future_ids'])
                self.assertEqual(row['reference_mse'],rows['train_mean']['reference_mse'])
            self.assertIsNone(rows['eeg_ideal']['physical_mse'])
            self.assertIsNone(rows['deterministic_replay']['physical_mse'])
            self.assertEqual(rows['eeg_full_selected']['history_samples'],2000)
            self.assertEqual(rows['eeg_short_selected']['history_samples'],64)
            information.run(config,root,resume=True)
            # The real fit signature contains three analysis dependencies. A
            # qualification guard must reproduce it rather than rejecting every fit.
            from experiments.eeg_representation.h1.v6 import planning
            from experiments.eeg_representation.h1.v6.artifacts import atomic_json
            qualified=planning.qualify_phase0(config,root)
            self.assertFalse(any('model implementation/training inputs changed' in reason for reason in qualified['missing']),qualified)
            guardpath=root/'models'/job/'fit_identity.json'
            guard=read_json(guardpath);guard['analysis_dependencies']['training.py']='tampered'
            atomic_json(guardpath,guard)
            rejected=planning.qualify_phase0(config,root)
            self.assertTrue(any('model source dependencies changed' in reason for reason in rejected['missing']))


    def test_frozen_means_on_same_replay_population_and_saved_physical_arrays(self):
        import tempfile
        from pathlib import Path
        from tests.test_eegrepr_v6_models import panel_fixture
        from experiments.eeg_representation.h1.v6 import models,replay
        from experiments.eeg_representation.h1.v6.artifacts import atomic_json,atomic_npz,artifact_record
        from experiments.eeg_representation.h1.v6.targets import target_spec
        from experiments.eeg_representation.h1.v4.measurement import _design,coefficient_hash
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);panels,manifest,config,job=panel_fixture(root,length=2000)
            config.update(cutoff_ms=6000.,history_s=4.,acquisition_ids=['A0'])
            # This is an explicitly synthetic, 64-coordinate engineering fit;
            # no saved trial or measured-NEURON claim is manufactured.
            for split,panel_ in panels.items():
                for key in ('Y','Y_physical'):panel_[key]=np.repeat(panel_[key],64,axis=-1)
                path=root/'derived'/job/(split+'.npz');atomic_npz(path,**panel_)
                manifest['files'][split]=artifact_record(root,path)
            manifest['normalization']['y_offset']=[0.]*64
            manifest['target_spec']['dimension']=64
            manifest['risk_contract'].update(dimension=64,target_physical_hash=target_spec('T0_l10_d128').fingerprint)
            manifest['target_fingerprint']=replay.canonical_hash(manifest['risk_contract'])
            manifest['input'].update(acquisition_id='A0',coefficient_sha256=coefficient_hash(_design('A0')[1]))
            atomic_json(root/'derived'/job/'manifest.json',manifest)
            atomic_json(root/'derived/manifest.json',dict(jobs=[dict(id=job,status='complete',**artifact_record(root,root/'derived'/job/'manifest.json'))]))
            models.fit(config,root)
            binding=replay.qualification_binding(config,root)
            t=np.arange(0,6138/.0625+1)*.0625;trials=[];records=[]
            for circuit in (37510,37511):
                for regime in range(4):
                    context=f's{circuit}-p0000-R{regime}'
                    y=1e-9+2e-9*np.sin(2*np.pi*(10+regime)*t/1000+circuit)
                    trials.append(dict(eeg_v=y[:,None],time_ms=t,metadata=dict(regime=dict(name='R'+str(regime)))))
                    records.append(dict(context_id=context,structure_id=circuit,trial_id=context+'-score-0000',source={'engineering':'synthetic'},replica={'engineering':'identical synthetic trace'}))
            tables,files=replay._qualified_tables(config,trials,copy.deepcopy(trials),records,binding,root=root)
            self.assertEqual(len(tables),1);self.assertEqual(len(files),1)
            rows={r['id']:r for r in tables[0]['rows']}
            self.assertEqual(rows['deterministic_replay']['physical_mse'],0.)
            self.assertEqual(rows['train_mean']['error_baseline100'],100.)
            self.assertEqual(rows['eeg_full_selected']['context_ids'],rows['train_mean']['context_ids'])
            self.assertEqual(rows['eeg_short_selected']['n_circuits'],2)
            self.assertIsNone(rows['state_variance']['physical_mse'])
            self.assertIsNone(rows['compressed_future_readout']['physical_mse'])
            with np.load(root/files[0]['path'],allow_pickle=False) as saved:
                self.assertEqual(saved['Y_physical'].shape,(8,1,64))
                np.testing.assert_array_equal(saved['Y_physical'][:,0],saved['deterministic_replay'])
                self.assertEqual(saved['eeg_short_selected'].shape,(1,8,64))


def replay_fixture():
    """Explicit synthetic schema fixture exercising audits, never saved as evidence."""
    t=np.arange(1,1281,dtype=float)*.0625;y=1e-6*np.sin(t)
    privileged={k:np.array([1.,2.,3.]) for k in ('structure_geometry','synapse_parameters','recurrent_netcons','external_netcons','external_event_times_ms','external_event_offsets','cutoff_physical_state')}
    privileged.update(phase_time_ms=np.arange(81.),phase_rad=np.arange(81.)*.03,
        pending_recurrent_events=np.zeros((0,6)),pending_external_events=np.zeros((0,6)))
    fork=dict(cutoff_ms=30.,physical_state_hash=array_hash(privileged['cutoff_physical_state']),external_history_hash='fixture',pending_recurrent_hash='empty',pending_external_hash='empty',prefix_spike_hash='fixture')
    return dict(time_ms=t,eeg_v=y,dipole_na_um=np.column_stack((y,y,y)),privileged=privileged,
        metadata=dict(backend='neuron_lfpy_real',structure_hash='fixture-only',fork=fork))


class ReplayAuditTests(unittest.TestCase):
    def test_exact_replay_and_nonzero_future_error(self):
        a=replay_fixture();b=copy.deepcopy(a)
        result=compare_trials(a,b,cutoff_ms=30.,targets=['T0_l0_d32'])
        self.assertTrue(result['passed']);self.assertEqual(result['targets'][0]['physical_mse'],0)
        b['eeg_v'][b['time_ms']>30]+=1e-7
        result=compare_trials(a,b,cutoff_ms=30.,targets=['T0_l0_d32'])
        self.assertFalse(result['passed']);self.assertAlmostEqual(result['targets'][0]['physical_mse']/1e-14,1)

    def test_native_column_channel_and_multichannel_rejection(self):
        from experiments.eeg_representation.h1.v6.replay import acquisition_state
        a=replay_fixture();column=copy.deepcopy(a);column['eeg_v']=column['eeg_v'][:,None]
        for acquisition in ('A0','A1','A2'):
            self.assertEqual(acquisition_state(a,acquisition,30.),acquisition_state(column,acquisition,30.))
        result=compare_trials(column,copy.deepcopy(column),cutoff_ms=30.,targets=['T0_l0_d32'])
        self.assertTrue(result['passed'])
        column['eeg_v']=np.repeat(column['eeg_v'],2,axis=1)
        with self.assertRaises(ValueError):acquisition_state(column,'A0',30.)

    def test_incomplete_state_or_changed_forcing_cannot_be_replay(self):
        a=replay_fixture();b=copy.deepcopy(a);b['privileged']['external_event_times_ms'][0]+=1
        with self.assertRaises(ValueError):compare_trials(a,b,cutoff_ms=30.)
        b=copy.deepcopy(a);b['privileged']['cutoff_physical_state'][0]+=1
        with self.assertRaises(ValueError):compare_trials(a,b,cutoff_ms=30.)
        b=copy.deepcopy(a);b['metadata']['backend']='analytical'
        with self.assertRaises(ValueError):compare_trials(a,b,cutoff_ms=30.)

if __name__=='__main__':unittest.main()
