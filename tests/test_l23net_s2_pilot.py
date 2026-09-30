"""Frozen target, prospective screening, paired inference and quota regressions."""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiments.l23net_analysis import s2_pilot_protocol as p
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis import analyze_l23net_s2_pilot as analysis
from experiments.l23net_analysis.nci.submit_l23net_s2_pilot import allocate, submit_jobs


class S2PilotTests(unittest.TestCase):
    def target(self):
        t={'mean_log10':[-19.,-19.,-19.],'scale_log10':[.1,.1,.1],'screen_cutoff':1.}
        return {'sha256':p.TARGET_SHA,'targets':{'plateau':t,'washout':t,
                'fundamental_excluded_plateau':{'low_beta':t}}}, {'targets':{'20s':t}}

    def report(self,seed=8501,arm='sham',eligible=True):
        target,gate=self.target(); power=[-18.7]*3 if eligible else [-19.]*3
        screen=s0.scores(power,gate['targets']['20s'])
        windows=[{'start_ms':1000*i,'stop_ms':1000*(i+1),
                  'firing_rates':{'HL23PYR_firing_rate_hz':1.,'HL23PV_firing_rate_hz':10.}}
                 for i in range(60)]
        amplitude=.4 if eligible and arm!='sham' else 0.
        return {'configuration':{'analysis':{'mode':'replication_pilot','condition':'mdd','arm':arm}},
                'seed_manifest':{'experiment_seed':seed},'replay_contract':{},'windows':windows,
                'memory_snapshots':[{'simulated_ms':8000,'rss_gib':{'sum':190.}}],
                'window_protocol':{'decision':{'requested_arm':arm,'delivered_arm':arm if eligible else 'sham',
                     'screen':screen,'baseline_rates_safe':True,'qualification_screen_bypass':False,
                     'amplitude_v_per_m':amplitude,'frequency_hz':14. if arm!='sham' else 0.,
                     'field_direction':[0.,0.,1.],'phase_rad_at_onset':0.,'decision_time_ms':28000.},
                     'nonzero_field_samples':10 if amplitude else 0,
                     'outcomes':{k:{'log_powers':power} for k in ('baseline','stimulation','washout')},
                     'fundamental_excluded':{'low_beta':{'log_powers':power}}}}

    def config(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(p.ROOT/'configs'),version_base=None):
            return compose(config_name='config',overrides=[
                'env=hl23net','analysis=l23net_s2_pilot','experiment.seed=8501','experiment.debug=false',
                'env.simulation.MDD=true','env.simulation.DRUG=false','env.ts.apply=true',
                'env.network.dt=0.025','env.simulation.duration=60000','env.simulation.obs_win_len=1000'])

    def test_frozen_seeds_arms_and_reservation(self):
        self.assertEqual(len(p.jobs()),20)
        self.assertEqual(p.PROTOCOL,json.loads(json.dumps(p.PROTOCOL)))
        self.assertEqual(set(p.SEEDS),set(range(8501,8511)))
        self.assertFalse(set(p.SEEDS)&set(p.reference.SEEDS))
        self.assertFalse(set(p.SEEDS)&set(s0.PROTOCOL['discovery_seeds']))
        self.assertEqual({j['run']['arm'] for j in p.jobs()},set(p.ARMS))
        self.assertAlmostEqual(p.RESERVATION_KSU,2.08)

    def test_full_walltime_quota_and_invalid_balances(self):
        jobs=allocate({'sj53':4.02,'fa32':4.19,'ny83':39.71})
        self.assertEqual([sum(j['project']==k for j in jobs) for k in ('sj53','fa32','ny83')],[1,1,18])
        self.assertAlmostEqual(len(jobs)*p.RESERVATION_KSU,41.6)
        for balances in ({'sj53':4.02,'fa32':4.19,'ny83':38.},
                         {'sj53':float('nan'),'fa32':4.,'ny83':40.},
                         {'sj53':-1.,'fa32':4.,'ny83':40.},{'ny83':100.}):
            with self.subTest(balances=balances),self.assertRaises(ValueError):allocate(balances)

    def test_production_configuration_is_exact(self):
        from experiments.l23net_analysis.run_l23net_s2_pilot import validate_configuration
        from experiments.l23net_analysis.run_l23net_no_field_replay import _replay_contract
        from omegaconf import OmegaConf
        cfg=self.config();contract=_replay_contract(cfg,624)
        contract.pop('experiment_seed');contract.pop('condition');contract['simulation'].pop('MDD')
        contract['simulation']['duration_ms']=28000.;contract['stimulation_enabled']=False
        gate={'sha256':'gate','reference_contract':contract}
        with patch.object(p,'load_prerequisites',return_value=(gate,{})),patch.object(s0,'load_gate',return_value=gate):
            self.assertEqual(validate_configuration(cfg,624),(60,40000))
            for key,value in [('experiment.seed',8401),('env.simulation.MDD',False),('env.simulation.DRUG',True),
                              ('analysis.arm','fixed_alpha'),('analysis.excluded_ms',4000),('analysis.mode','qualification'),
                              ('env.network.dt',.05),('env.network.celsius',36.5),('env.network.v_init',-70),
                              ('env.ts.apply',False),('experiment.debug',True),('analysis.env_seed',1)]:
                bad=copy.deepcopy(cfg);OmegaConf.update(bad,key,value)
                with self.subTest(key=key),self.assertRaises(ValueError):validate_configuration(bad,624)
            with self.assertRaises(ValueError):validate_configuration(cfg,2)

    def test_debug_cannot_enter_production(self):
        from experiments.l23net_analysis.run_l23net_s2_pilot import validate_configuration
        cfg=self.config();cfg.analysis.mode='debug'
        with self.assertRaises(ValueError):validate_configuration(cfg,624)

    def test_hash_locked_target_and_screen(self):
        target,gate=self.target();gate['sha256']='gate'
        target.update(code_sha256=p.reference.code_hashes(),s0_sha256='gate')
        target['targets']['baseline_screen_unchanged']=gate['targets']['20s']
        summary={'sha256':p.SUMMARY_SHA,'target_sha256':p.TARGET_SHA,'status':'reference_calibrated',
                 'errors':[],'completed_valid_runs':16}
        with patch.object(p.reference,'load_signed',side_effect=[target,summary]),patch.object(s0,'load_gate',return_value=gate),patch.object(p.s1,'load_qualification'):
            self.assertEqual(p.load_prerequisites('/unused'),(gate,target))
        for field,value in [('sha256','different'),('code_sha256',{}),('s0_sha256','changed')]:
            bad=copy.deepcopy(target);bad[field]=value
            with patch.object(p.reference,'load_signed',side_effect=[bad,summary]),patch.object(s0,'load_gate',return_value=gate):
                with self.assertRaises(ValueError):p.load_prerequisites('/unused')

    def test_eligibility_rejection_and_bypass_guards(self):
        target,gate=self.target()
        for eligible in (True,False):
            for arm in p.ARMS:
                r=self.report(arm=arm,eligible=eligible)
                self.assertEqual(analysis.audit_decision(r,gate,arm),eligible)
                r['window_protocol']['decision']['qualification_screen_bypass']=True
                with self.assertRaises(ValueError):analysis.audit_decision(r,gate,arm)
        r=self.report(arm='fixed_low_beta',eligible=False);r['window_protocol']['nonzero_field_samples']=1
        with self.assertRaises(ValueError):analysis.audit_decision(r,gate,'fixed_low_beta')

    def test_new_target_not_legacy_score_and_rate_safety(self):
        target,gate=self.target();a=self.report();b=self.report(arm='fixed_low_beta')
        b['window_protocol']['outcomes']['stimulation']['log_powers']=[-18.8]*3
        b['window_protocol']['fundamental_excluded']['low_beta']['log_powers']=[-18.8]*3
        row=analysis.contrast(8501,a,b,target)
        self.assertAlmostEqual(row['benefit'],1.)
        self.assertGreater(row['shift_alignment'],0)
        self.assertTrue(row['rate_safe'])
        for w in b['windows'][29:49]:w['firing_rates']['HL23PYR_firing_rate_hz']=2.
        self.assertFalse(analysis.contrast(8501,a,b,target)['rate_safe'])

    def test_runtime_summary_scores_frozen_epoch_target(self):
        from experiments.l23net_analysis.run_l23net_s2_pilot import PilotProtocol
        target,gate=self.target();gate['sha256']='gate'
        old=gate['targets']['20s']
        gate['targets'].update({'10s':old,'fundamental_excluded':{band:old for band in s0.BANDS}})
        # New epoch means deliberately differ; old diagnostic scores must not leak in.
        target=copy.deepcopy(target)
        target['targets']['plateau']['mean_log10']=[-20.]*3
        cfg=self.config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.experiment.dir=tmp
            with patch.object(p,'load_prerequisites',return_value=(gate,target)),patch.object(s0,'load_gate',return_value=gate):
                protocol=PilotProtocol(cfg)
            protocol.spike_path.write_bytes(b'unit-test sparse artifact hash')
            t=np.arange(1,15001)/250
            protocol.times=[t]
            protocol.eeg=[sum(a*np.sin(2*np.pi*f*t) for a,f in [(1e-9,6.1),(2e-9,10.1),(3e-9,14.1)])]
            protocol.completed=protocol.total
            value=protocol.summary();meta=value['s2_pilot']
            expected=s0.scores(value['outcomes']['stimulation']['log_powers'],target['targets']['plateau'])
            self.assertEqual(meta['outcomes']['plateau'],expected)
            self.assertNotEqual(meta['outcomes']['plateau']['distance'],value['outcomes']['stimulation']['distance'])
            self.assertEqual(meta['target_sha256'],p.TARGET_SHA)
            protocol.close();protocol.close()

    def test_manifest_cannot_change_arms_or_omit_pairs(self):
        target,gate=self.target();gate['sha256']='gate'
        manifest={'stage':'s2_pilot','protocol':p.PROTOCOL,'code_sha256':p.code_hashes(),
                  'target_sha256':p.TARGET_SHA,'s0_sha256':'gate','jobs':p.jobs()}
        with patch.object(p,'load_prerequisites',return_value=(gate,target)):
            self.assertEqual(p.check_manifest(manifest,'unused'),(gate,target))
            for key,value in [('jobs',p.jobs()[:-1]),('target_sha256','refit'),('code_sha256',{})]:
                changed=copy.deepcopy(manifest);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):p.check_manifest(changed,'unused')

    def rows(self):
        return [{'seed':s,'eligible':True,'benefit':.4+i*.01,'excluded_benefit':.3,
                 'washout_benefit':0.,'shift_alignment':.5,'rate_safe':True,
                 'band_log10_sham_minus_active':[.01,.02,.03]} for i,s in enumerate(p.SEEDS)]

    def test_complete_cohort_and_positive_negative_pilot(self):
        rows=self.rows();result=analysis.cohort_result(rows)
        self.assertTrue(result['pilot_supports_larger_confirmation'])
        self.assertEqual(result['primary']['n_structures'],10)
        self.assertAlmostEqual(result['primary']['p_one_sided'],1/1024)
        for row in rows:row['benefit']=-.4
        self.assertFalse(analysis.cohort_result(rows)['pilot_supports_larger_confirmation'])
        for incomplete in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):analysis.cohort_result(incomplete)

    def test_no_replacement_screen_yield_and_all_candidate_audit(self):
        rows=self.rows()
        for r in rows[:3]:r['eligible']=False;r['benefit']=0.
        result=analysis.cohort_result(rows)
        self.assertFalse(result['pilot_checks']['screen_coverage'])
        self.assertEqual(result['primary']['n_structures'],7)
        self.assertEqual(result['all_candidate_policy_audit']['n_structures'],10)
        for r in rows:r['eligible']=False
        self.assertFalse(analysis.cohort_result(rows)['pilot_supports_larger_confirmation'])

    def test_submission_wave_and_persisted_partial_failure(self):
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                manifest={'commit':'test','jobs':allocate({'sj53':4.02,'fa32':4.19,'ny83':39.71}),'submission_status':'submitting'}
                outputs=['1.gadi-pbs',subprocess.CalledProcessError(1,'qsub')] if fail else [f'{i}.gadi-pbs' for i in range(20)]
                with patch('subprocess.check_output',side_effect=outputs) as mock:
                    if fail:
                        with self.assertRaises(subprocess.CalledProcessError):submit_jobs(Path(tmp),manifest,20)
                    else:
                        submit_jobs(Path(tmp),manifest,20)
                        self.assertTrue(all('-W' not in c.args[0] for c in mock.call_args_list))
                saved=json.loads((Path(tmp)/'submission.json').read_text())
                self.assertEqual(saved['submission_status'],'partial_failure' if fail else 'submitted')
                self.assertIn('job_id',saved['jobs'][0])

    def test_analysis_never_freezes_partial_results(self):
        target,gate=self.target()
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);jobs=allocate({'sj53':4.02,'fa32':4.19,'ny83':39.71})
                (root/'submission.json').write_text(json.dumps({'jobs':jobs,'commit':'test','submission_status':'submitted'}))
                for j in jobs:
                    d=root/j['name'];d.mkdir()
                    (d/'worker_exit_code.txt').write_text('0');(d/'git_commit.txt').write_text('test')
                    (d/'l23net_s1_run.json').write_text('{}')
                    (d/'pbs.out').write_text('Service Units: 1600\nWalltime Used: 01:17:00\nMemory Used: 160GB\nExit Status: 0\n')
                def audit(directory,gate):
                    j=next(j for j in jobs if j['name']==directory.name)
                    if fail and directory.name==jobs[0]['name']:raise ValueError('failed raw trace')
                    r=self.report(seed=j['run']['seed'],arm=j['run']['arm'])
                    scores=s0.scores(r['window_protocol']['outcomes']['stimulation']['log_powers'],target['targets']['plateau'])
                    r['window_protocol']['s2_pilot']={'protocol_sha256':p.reference.canonical_json_sha256(p.PROTOCOL),
                      'code_sha256':p.code_hashes(),'target_sha256':p.TARGET_SHA,
                      'outcomes':{'plateau':scores,'washout':scores},'excluded_plateau':scores}
                    return r
                with patch.object(p,'check_manifest',return_value=(gate,target)),patch.object(p,'validate_contract'),patch.object(analysis.s1,'audit_run',side_effect=audit),patch.object(analysis.s1,'compare_prefix'):
                    check=analysis.analyze_suite(root,write=False)
                    self.assertFalse((root/'s2_pilot_status.json').exists())
                    self.assertFalse((root/'s2_pilot_summary.json').exists())
                    result=analysis.analyze_suite(root)
                    self.assertEqual(check,result)
                self.assertEqual(result['technical_passed'],not fail)
                self.assertEqual((root/'s2_pilot_summary.json').exists(),not fail)
                self.assertTrue((root/'s2_pilot_status.json').exists())
                if not fail:
                    self.assertFalse(result['pilot_supports_larger_confirmation'])
                    self.assertEqual(len(result['rows']),10)

    def test_worker_shell_and_resource_contract(self):
        path=p.ROOT/'experiments/l23net_analysis/nci/run_l23net_s2_pilot_worker.sh'
        subprocess.run(['bash','-n',str(path)],check=True)
        source=path.read_text()
        for token in ('walltime='+p.WALLTIME,'ncpus=624','mem=256GB','env.simulation.MDD=true',
                      'analysis.mode=replication_pilot','env.network.dt=0.025','< /dev/null'):
            self.assertIn(token,source)


if __name__=='__main__':
    unittest.main()
