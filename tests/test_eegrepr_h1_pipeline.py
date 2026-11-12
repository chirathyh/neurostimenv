"""Integration guards for native timing, data isolation, execution and provenance."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import contextlib,io,json,unittest
from unittest.mock import patch
import numpy as np
from experiments.eeg_representation.h1.targets import TargetSpec,extract_target,require_same_target,physical_risk
from experiments.eeg_representation.h1.acquisition import causal_history,metadata
from experiments.eeg_representation.h1.config import load_config,resolve,context_specs
from experiments.eeg_representation.h1.artifacts import initialize_output,require_compatible,stage_complete,resume_stage,atomic_npz,ROOT
from experiments.eeg_representation.h1.dataset import generate,derive,load_panel,precut_normalization_target
from experiments.eeg_representation.h1.planning import plan

class TimingAndTargetTests(unittest.TestCase):
    def test_native_right_edges_and_box_projection(self):
        t=np.arange(1,16001)*.0625;s=TargetSpec(cutoff_ms=500,history_s=.25,lead_ms=2,duration_ms=32)
        actual=extract_target(t,t,s)
        np.testing.assert_array_equal(actual,502+np.arange(16)*2+1.03125)
        with self.assertRaises(ValueError):TargetSpec(lead_ms=2,output_grid_hz=100)
        with self.assertRaises(ValueError):TargetSpec(lead_ms=2.01)
    def test_future_perturbation_and_actual_delay(self):
        t=np.arange(1,16001)*.0625;x=np.sin(t*.02);other=x.copy();other[t>500]=1e50
        opts=dict(cutoff_ms=500,history_s=.25,rate_hz=100)
        a,at,m=causal_history(x,t,**opts);b,bt,_=causal_history(other,t,**opts)
        np.testing.assert_array_equal(a,b);self.assertEqual(at[-1],0);self.assertEqual(at[0],-240)
        self.assertEqual(m['support_ms'],128);self.assertEqual(m['group_delay_ms'],64)
    def test_predecision_normalizer_never_reads_future(self):
        t=np.arange(1,16001)*.0625;x=np.sin(t*.02);s=TargetSpec(cutoff_ms=500,history_s=.25)
        y=precut_normalization_target({'time_ms':t,'eeg_v':x},s);x[t>500]=1e50
        np.testing.assert_array_equal(y,precut_normalization_target({'time_ms':t,'eeg_v':x},s))
    def test_incompatible_context_or_scaler_rejected(self):
        a=TargetSpec();b=TargetSpec(normalization_id='different')
        with self.assertRaises(ValueError):require_same_target(a,b)
        with self.assertRaises(ValueError):require_same_target(a,TargetSpec(population_id='different'))
        self.assertEqual(physical_risk(2,3),18)

class ConfigAndArtifactTests(unittest.TestCase):
    def test_actual_job_counts_and_allocation(self):
        c=load_config(ROOT/'configs/eeg_representation/h1/scale_core.yaml');c['cost_gate']['required']=False
        self.assertEqual(plan(c)['actual_job_count'],7);self.assertEqual(plan(c)['final_fitted_model_count'],70)
        c=load_config(ROOT/'configs/eeg_representation/h1/fixed_budget_extension.yaml');c['cost_gate']['required']=False
        p=plan(c);self.assertEqual(p['continuations_by_split'],{'train':768,'development':64,'evaluation':3584})
    def test_timing_gate_and_split_rejections(self):
        gate=load_config(ROOT/'configs/eeg_representation/h1/timing_gate.yaml')
        self.assertEqual(plan(gate)['final_fitted_model_count'],0)
        c=load_config(ROOT/'configs/eeg_representation/h1/bounded_development.yaml')
        with self.assertRaisesRegex(ValueError,'timing audit'):plan(c)
        c['data']['structures']['evaluation'][0]=c['data']['structures']['train'][0]
        with self.assertRaisesRegex(ValueError,'Disjoint'):resolve(c)
    def test_immutable_source_config_and_checksum_resume(self):
        c=load_config(ROOT/'configs/eeg_representation/h1/analytic_smoke.yaml')
        with TemporaryDirectory() as td:
            r=Path(td);initialize_output(r,c);a=r/'x.npz';atomic_npz(a,x=np.arange(3));stage_complete(r,'test',[a])
            self.assertEqual(resume_stage(r,'test',True)['status'],'complete')
            with self.assertRaises(FileExistsError):initialize_output(r,c)
            different=deepcopy(c);different['benchmark']['steps']+=1
            with self.assertRaises(ValueError):require_compatible(r,different)
            with patch('experiments.eeg_representation.h1.artifacts.source_hash',return_value='changed'):
                with self.assertRaises(ValueError):require_compatible(r,c)
            a.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):resume_stage(r,'test',True)
    def test_pickle_arrays_rejected(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(ValueError):atomic_npz(Path(td)/'x.npz',x=np.array([{}],object))

class SmallDataIntegration(unittest.TestCase):
    def test_executed_jobs_training_only_scaler_and_split_pool_isolation(self):
        c=load_config(ROOT/'configs/eeg_representation/h1/analytic_smoke.yaml')
        c['data']['structures']={'train':[35100,35101],'development':[35103],'evaluation':[35105]}
        c['data']['prefixes_per_regime']={'train':1,'development':1,'evaluation':1}
        c['forks']['score_futures']=2;c['forks']['reference_futures']=2
        second=deepcopy(c['jobs'][0]);second.update(id='rate100',input_rate_hz=100)
        c['jobs'].append(second);c=resolve(c)
        with TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);initialize_output(root,c);generate(c,root);report=derive(c,root)
            self.assertEqual(report['completed_jobs'],plan(c)['actual_job_count'])
            a=load_panel(root,'primary');b=load_panel(root,'rate100')
            np.testing.assert_array_equal(a['evaluation']['Y_physical'],b['evaluation']['Y_physical'])
            self.assertEqual(b['evaluation']['X'].shape[-1],25)
            n=a['manifest']['normalization'];self.assertEqual(set(n['fit_contexts']),set(a['train']['context_id']))
            self.assertAlmostEqual(n['x_offset'],a['train']['X_physical'].mean())
            score=set(a['evaluation']['future_id'].ravel());ref=set(a['evaluation']['reference_future_id'].ravel())
            self.assertFalse(score&ref)
            for split in ('train','development','evaluation'):
                ids=set(a[split]['structure_id']);self.assertEqual(ids,set(c['data']['structures'][split]))
            # Corrupt a declared shape with a matching file checksum: semantic validation must still fail.
            from experiments.eeg_representation.h1.artifacts import atomic_json,artifact_record
            manifest_path=root/'panels/primary/manifest.json';m=json.loads(manifest_path.read_text())
            e=a['evaluation'];e['Y']=e['Y'][...,:-1];path=root/'panels/primary/evaluation.npz';atomic_npz(path,**e)
            m['files']['evaluation']=artifact_record(root,path);atomic_json(manifest_path,m)
            index=json.loads((root/'panels/manifest.json').read_text());index['jobs'][0].update(artifact_record(root,manifest_path));atomic_json(root/'panels/manifest.json',index)
            with self.assertRaisesRegex(ValueError,'dimension'):load_panel(root,'primary')


class TimingGateTrustTests(unittest.TestCase):
    def _gate(self,root):
        from experiments.eeg_representation.h1.artifacts import atomic_json
        from experiments.eeg_representation.h1.config import cutoff_ms,stop_ms
        c=load_config(ROOT/'configs/eeg_representation/h1/timing_gate.yaml')
        initialize_output(root,c)
        contexts=context_specs(c);paths=[];audits=[]
        for context in contexts:
            for future in range(2):
                path=root/'native/evaluation'/context['context_id']/f'score-{future:04d}.h5'
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test checksum fixture')
                paths.append(path)
            audits.append({'context_id':context['context_id'],'fork_audit_status':'passed','audit':{'passed':True}})
        timing={'backend':'neuron','population_e':32,'population_i':8,'complete':True,'contexts':4,
            'continuations':8,'mean_continuation_seconds':2.,'summed_generation_seconds':16.,
            'native_bytes':sum(path.stat().st_size for path in paths),'episode_stop_ms':stop_ms(c),
            'cutoff_ms':cutoff_ms(c),'peak_process_rss_bytes':10000}
        tp=root/'native/timing-generate-evaluation.json';ap=root/'native/prefix_validation-generate-evaluation.json'
        atomic_json(tp,timing);atomic_json(ap,{'contexts':audits})
        atomic_json(root/'timing_audit.json',timing)
        stage_complete(root,'generate-evaluation',paths+[tp,ap],timing)
        return c,tp

    def test_unbound_alias_cannot_change_projection_and_bound_tamper_rejected(self):
        from experiments.eeg_representation.h1.artifacts import atomic_json
        from experiments.eeg_representation.h1.planning import timing_reference
        with TemporaryDirectory() as td:
            root=Path(td);c,tp=self._gate(root)
            atomic_json(root/'timing_audit.json',{'mean_continuation_seconds':.000001})
            measured=timing_reference(c,root)
            self.assertEqual(measured['mean_continuation_seconds'],2.)
            self.assertEqual(measured['verified_timing_artifact'],'native/timing-generate-evaluation.json')
            tp.write_text('{}')
            with self.assertRaisesRegex(ValueError,'corrupt'):timing_reference(c,root)

    def test_design_and_configuration_binding(self):
        from experiments.eeg_representation.h1.artifacts import atomic_json
        from experiments.eeg_representation.h1.planning import timing_reference
        from experiments.eeg_representation.h1.targets import canonical_hash
        with TemporaryDirectory() as td:
            root=Path(td);c,_=self._gate(root)
            changed=deepcopy(c);changed['simulation']['population_e']=2
            atomic_json(root/'resolved_config.json',changed)
            with self.assertRaisesRegex(ValueError,'provenance'):timing_reference(c,root)
            provenance=json.loads((root/'provenance.json').read_text())
            provenance['config_hash']=canonical_hash(changed);atomic_json(root/'provenance.json',provenance)
            with self.assertRaisesRegex(ValueError,'32E/8I'):timing_reference(c,root)


@unittest.skipUnless(__import__('os').environ.get('EEGREPR_RUN_NEURON')=='1','Set EEGREPR_RUN_NEURON=1 for early-stop integration')
class NativeEarlyStopTests(unittest.TestCase):
    def test_early_stop_matches_longer_episode_through_required_sample(self):
        from experiments.eeg_representation.h1.dataset import adapter_config,trial_spec
        from experiments.eeg_representation.simulator import generate_trial
        from experiments.eeg_representation.forks import validate_fork_prefixes
        c=load_config(ROOT/'configs/eeg_representation/h1/simulator_smoke.yaml')
        c['data']['regimes'][1]['diffusion']=2.
        context=[x for x in context_specs(c) if x['split']=='evaluation' and x['regime']['name']=='Ralpha'][0]
        spec=trial_spec(c,context,'score',0);short=adapter_config(c);long=deepcopy(short)
        long['simulation']['duration_s']+=.1
        a=generate_trial(short,spec);b=generate_trial(long,spec)
        for key in ('time_ms','eeg_v','dipole_na_um'):
            np.testing.assert_array_equal(a[key],b[key][:len(a[key])])
        cropped=deepcopy(b)
        for key in ('time_ms','eeg_v','dipole_na_um'):cropped[key]=cropped[key][:len(a[key])]
        for key in ('phase_time_ms','phase_rad'):
            cropped['privileged'][key]=cropped['privileged'][key][:len(a['privileged'][key])]
        self.assertTrue(validate_fork_prefixes([a,cropped],spec['future_start_ms'])['passed'])
        self.assertEqual(a['metadata']['population_sizes'],{'E':2,'I':1})

if __name__=='__main__':unittest.main()
