"""Causality, full-spectrum PSD, timing, field, and fail-closed gate tests."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.replay_validation import canonical_json_sha256


class S0S1Tests(unittest.TestCase):
    def test_causal_filter_chunking_and_future_independence(self):
        x = np.random.default_rng(1).normal(size=40000)
        a=s0.CausalEEG(.025); t,y=a.append(x)
        b=s0.CausalEEG(.025); ta,ya=b.append(x[:12345]); tb,yb=b.append(x[12345:])
        np.testing.assert_array_equal(np.r_[ta,tb],t)
        np.testing.assert_array_equal(np.r_[ya,yb],y)
        changed=x.copy(); changed[20000:]*=100
        tc,yc=s0.CausalEEG(.025).append(changed)
        np.testing.assert_array_equal(y[t<=.5],yc[tc<=.5])

    def test_full_spectrum_power_units_and_resolution(self):
        t=np.arange(5000)/250
        x=sum(a*np.sin(2*np.pi*f*t) for a,f in [(1e-9,6),(2e-9,10),(3e-9,14)])
        f,p=s0.spectrum(x)
        self.assertAlmostEqual(f[1]-f[0],.25)
        np.testing.assert_allclose(10**s0.log_powers(x),np.array([.5,2,4.5])*1e-18,rtol=1e-8)
        excluded=10**s0.log_powers(x,(9.5,10.5))
        self.assertLess(excluded[1],1e-30)
        np.testing.assert_allclose(excluded[[0,2]],np.array([.5,4.5])*1e-18,rtol=1e-8)

    def test_transient_and_ramps_excluded(self):
        self.assertEqual(s0.PROTOCOL['outcome_ms'],[29000,49000])
        self.assertEqual(s0.PROTOCOL['excluded_ms'],8000)
        t=np.arange(1,15001)/250
        x=np.zeros_like(t);x[t<=8]=1e9
        self.assertEqual(len(s0.epoch(t,x,8,28)),5000)
        self.assertFalse(s0.epoch(t,x,8,28).any())
        with self.assertRaises(ValueError): s0.epoch(t[:-1],x[:-1],50,60)

    def test_carrier_abstains_for_dc_and_finds_known_sines(self):
        self.assertFalse(s0.carrier(np.ones(5000),'alpha')['accepted'])
        t=np.arange(5000)/250
        for band,f in s0.FIXED.items():
            pick=s0.stable_carrier(np.sin(2*np.pi*f*t),band)
            self.assertTrue(pick['accepted'])
            self.assertEqual(pick['frequency_hz'],f)

    def test_phase_convention(self):
        t=np.arange(1,501)/250+100
        phi=.37
        fitted=s0.phase_fit(t,1e-9*np.cos(2*np.pi*10*(t-102)+phi),10,102)
        self.assertAlmostEqual(float(s0.wrap_phase(fitted['phase_rad']-phi)),0,places=10)
        self.assertAlmostEqual(fitted['r2'],1)

    def make_gate(self):
        from experiments.l23net_analysis.run_l23net_no_field_replay import _replay_contract
        contract=_replay_contract(self.config(),624)
        contract.pop('experiment_seed');contract.pop('condition');contract['simulation'].pop('MDD')
        contract['simulation']['duration_ms']=28000.
        contract['stimulation_enabled']=False
        value={'protocol':s0.PROTOCOL,'technical_passed':True,
               'reference_contract':contract,
               'gates':{'full_spectrum_phenotype':True},
               'allowed_arms':['sham','fixed_theta','fixed_alpha','fixed_low_beta','transverse_alpha']}
        value['sha256']=canonical_json_sha256(value)
        return value

    def test_gate_detects_tampering_and_negative_phenotype(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'gate.json'; gate=self.make_gate()
            p.write_text(json.dumps(gate)); self.assertTrue(s0.load_gate(p)['technical_passed'])
            gate['technical_passed']=False;p.write_text(json.dumps(gate))
            with self.assertRaises(ValueError): s0.load_gate(p)
            gate=self.make_gate();gate['gates']['full_spectrum_phenotype']=False
            gate.pop('sha256');gate['sha256']=canonical_json_sha256(gate);p.write_text(json.dumps(gate))
            with self.assertRaises(ValueError): s0.load_gate(p)

    def config(self):
        from hydra import initialize_config_dir,compose
        root=Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
            return compose(config_name='config',overrides=['env=hl23net','analysis=l23net_s1',
                'experiment.seed=8451','experiment.debug=false','env.network.dt=0.025',
                'env.simulation.duration=60000','env.simulation.obs_win_len=1000','env.simulation.MDD=true','env.ts.apply=true'])

    def test_production_gate_rejects_unqualified_actions_and_timing(self):
        from experiments.l23net_analysis.run_l23net_s1 import validate_configuration
        cfg=self.config()
        with patch.object(s0,'load_gate',return_value=self.make_gate()):
            self.assertEqual(validate_configuration(cfg,624),(60,40000))
            cfg.analysis.arm='matched_alpha_anti'
            with self.assertRaises(ValueError):validate_configuration(cfg,624)
            cfg.analysis.arm='sham';cfg.analysis.excluded_ms=4000
            with self.assertRaises(ValueError):validate_configuration(cfg,624)

    def test_discovery_requires_long_run_qualification(self):
        from experiments.l23net_analysis.run_l23net_s1 import validate_configuration
        cfg=self.config();cfg.analysis.mode='discovery';cfg.experiment.seed=8401
        with patch.object(s0,'load_gate',return_value=self.make_gate()), \
             patch('experiments.l23net_analysis.s1_analysis.load_qualification',side_effect=ValueError('not qualified')):
            with self.assertRaises(ValueError):validate_configuration(cfg,624)

    def test_debug_cannot_qualify_production(self):
        from experiments.l23net_analysis.run_l23net_s1 import validate_configuration
        cfg=self.config();cfg.analysis.mode='debug'
        with self.assertRaises(ValueError):validate_configuration(cfg,624)

    def test_qualification_code_identity(self):
        from experiments.l23net_analysis.s1_analysis import load_qualification,code_hashes
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'q.json'
            q={'status':'qualified','s0_sha256':'same','code_sha256':code_hashes()}
            q['sha256']=canonical_json_sha256(q);p.write_text(json.dumps(q))
            self.assertEqual(load_qualification(p,'same')['status'],'qualified')
            with self.assertRaises(ValueError):load_qualification(p,'different')
            with patch('experiments.l23net_analysis.s1_analysis.code_hashes',return_value={}):
                with self.assertRaises(ValueError):load_qualification(p,'same')

    def test_s1_submission_size_budget_and_discovery_block(self):
        from experiments.l23net_analysis.nci.submit_l23net_s1 import jobs_for_stage,allocate_projects,main
        self.assertEqual(len(jobs_for_stage('qualification')),3)
        self.assertEqual(len(jobs_for_stage('discovery')),13)
        jobs=allocate_projects(jobs_for_stage('qualification'),{'sj53':13.63,'fa32':12.28,'ny83':73.26})
        self.assertTrue(all(j['project']=='sj53' for j in jobs))
        with patch.object(s0,'load_gate',return_value=self.make_gate()):
            with self.assertRaises(ValueError):main(['--stage','discovery'])

    def test_worker_mpirun_stdin_and_failure_dependencies(self):
        root=Path(__file__).resolve().parents[1]
        worker=(root/'experiments/l23net_analysis/nci/run_l23net_s1_worker.sh').read_text()
        self.assertIn('env.online.temperature_mode=configured < /dev/null',worker)
        submit=(root/'experiments/l23net_analysis/nci/submit_l23net_s1.py').read_text()
        self.assertIn('depend=afterok:',submit)

    def test_complete_60s_analysis_and_sparse_stream_without_simulation(self):
        import h5py
        from omegaconf import OmegaConf
        from experiments.l23net_analysis.run_l23net_s1 import FieldProtocol
        from experiments.l23net_analysis.run_l23net_no_field_replay import _replay_contract
        from experiments.l23net_analysis.replay_validation import trace_content_summary
        from experiments.l23net_analysis.s1_analysis import audit_run,paired_contrast
        from env.models.neuron.streaming import OnlineTraceWriter
        from env.models.neuron.stimulation import apply_raised_cosine_block_envelope
        cfg=self.config();cfg.analysis.arm='fixed_theta'
        root=Path(__file__).resolve().parents[1]
        cfg.analysis.s0_gate=str(root/'experiments/l23net_analysis/frozen/s0_full_spectrum_v1.json')
        with tempfile.TemporaryDirectory() as tmp:
            cfg.experiment.dir=tmp
            protocol=FieldProtocol(cfg)
            writer=OnlineTraceWriter(Path(tmp)/'l23net_s1_trace.h5',stage_names=protocol.stage_names)
            windows=[]
            for i in range(60):
                action,kwargs,stage=protocol.action(i)
                t=i+np.arange(1,40001)/40000
                eeg=1e-9*(np.sin(2*np.pi*6*t)+np.sin(2*np.pi*10*t)+np.sin(2*np.pi*14*t))
                ft=i*1000+np.arange(40001)*.025
                field=action['ac_amplitude_v_per_m']*np.sin(2*np.pi*action['frequency_hz']*(ft-28000)/1000)
                field=apply_raised_cosine_block_envelope(field,time_ms=ft,block_start_ms=28000,block_stop_ms=50000,ramp_ms=1000)
                result={'sample_times_ms':t*1000,'eeg_v':eeg[None,:],
                    'dipole_nA_um':np.zeros((3,40000)), 'firing_rates':{'HL23PYR_firing_rate_hz':1.,'HL23PYR_spike_count':800.},
                    'spikes':{'HL23PYR':{'times_ms':np.array([]),'gids':np.array([],dtype=int)}},
                    'stimulation':{'time_ms':ft,'field_v_per_m':field,'field_direction':np.array([0.,0.,1.])}}
                errors,metadata=protocol.observe(i,result,[0.],[.1 if 28<=i<50 else 0.])
                self.assertEqual(errors,[])
                writer.append_window(sample_time_ms=t*1000,eeg_v=result['eeg_v'],dipole_nA_um=result['dipole_nA_um'],
                                     field_left_boundary_time_ms=ft[:-1],field_left_boundary_v_per_m=field[:-1],stage_code=stage)
                windows.append({'start_ms':i*1000,'stop_ms':(i+1)*1000,'spikes':{'counts':{'HL23PYR':0}},
                                'firing_rates':result['firing_rates'],**metadata})
            final=protocol.summary();protocol.close();protocol.close()
            writer.close()
            self.assertEqual(final['analysis_storage_samples'],15000)
            self.assertEqual(final['outcomes']['stimulation']['interval_s'],[29,49])
            self.assertTrue(final['tissue_coupling_observed'])
            self.assertEqual(final['decision']['amplitude_v_per_m'],.4)
            with h5py.File(Path(tmp)/'l23net_s1_spikes.h5') as h:
                self.assertEqual(h.attrs['committed_windows'],60)
            contract=_replay_contract(cfg,624)
            report={'status':'passed','errors':[],'build_audit':{'errors':[]},'replay_contract':contract,
                    'replay_contract_sha256':canonical_json_sha256(contract),'window_protocol':final,
                    'artifacts':{'trace_summary':trace_content_summary(Path(tmp)/'l23net_s1_trace.h5')},
                    'windows':windows,'configuration':OmegaConf.to_container(cfg,resolve=True)}
            path=Path(tmp)/'l23net_s1_run.json';path.write_text(json.dumps(report))
            self.assertEqual(audit_run(tmp,protocol.gate)['status'],'passed')
            contrast=paired_contrast(8401,'transverse_alpha',report,report)
            self.assertEqual(contrast['benefit'],0.)
            self.assertEqual(contrast['excluded_benefit'],0.)
            self.assertTrue(contrast['paired_rate_safe'])
            report['window_protocol']['fundamental_excluded']['alpha']['distance']+=1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'Excluded-band target score'):
                audit_run(tmp,protocol.gate)

    def test_rate_guard_never_confuses_counts_with_hz(self):
        from experiments.l23net_analysis.s1_analysis import population_rates
        rates=population_rates({'HL23PYR_firing_rate_hz':1.2,'HL23PYR_spike_count':960.,
                                'HL23PV_firing_rate_hz':15.,'HL23PV_spike_count':1050.})
        self.assertEqual(rates,{'HL23PYR':1.2,'HL23PV':15.})
        with self.assertRaises(ValueError):population_rates({'HL23PYR_spike_count':960.})

    def test_submission_failure_is_recorded_without_resubmitting(self):
        from experiments.l23net_analysis.nci import submit_l23net_s1 as submit
        for fail_at in (None,2):
            with self.subTest(fail_at=fail_at),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);gate=root/'gate.json';gate.write_text('{}');calls=[]
                def output(command,**kwargs):
                    if command[0]=='git':return 'commit\n'
                    calls.append(command)
                    if len(calls)==fail_at:raise subprocess.CalledProcessError(1,command)
                    return f'{len(calls)}.gadi-pbs\n'
                with patch.object(submit,'ROOT',root),patch.object(s0,'load_gate',return_value=self.make_gate()), \
                     patch.object(submit,'code_hashes',return_value={}),patch.object(submit.shutil,'which',return_value='/qsub'), \
                     patch.object(submit.subprocess,'run'),patch.object(submit.subprocess,'check_output',side_effect=output), \
                     contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    args=['--stage','qualification','--s0',str(gate),'--max-concurrent','1',
                          '--available-ksu','sj53=13.63','fa32=12.28','ny83=73.26','--submit']
                    if fail_at:
                        with self.assertRaises(subprocess.CalledProcessError):submit.main(args)
                    else:self.assertEqual(submit.main(args),0)
                manifest=json.loads(next((root/'results').glob('*/submission.json')).read_text())
                self.assertEqual(manifest['submission_status'],'partial_failure' if fail_at else 'submitted')
                self.assertIn('depend=afterok:1.gadi-pbs',calls[1])
                self.assertEqual(len(calls),fail_at or 3)


if __name__=='__main__':unittest.main()
