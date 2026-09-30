"""Reference calibration guards, portable replay hashes, targets and quota accounting."""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from experiments.l23net_analysis import reference60_protocol as p
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.analyze_l23net_reference60 import fit_targets
from experiments.l23net_analysis.nci.submit_l23net_reference60 import allocate, submit_jobs


class Reference60Tests(unittest.TestCase):
    def config(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(p.ROOT/'configs'), version_base=None):
            return compose(config_name='config', overrides=[
                'env=hl23net','analysis=l23net_reference60','experiment.seed=8101',
                'experiment.debug=false','env.simulation.MDD=false','env.simulation.DRUG=false',
                'env.ts.apply=true','env.network.dt=0.025','env.simulation.duration=60000',
                'env.simulation.obs_win_len=1000'])

    def test_exact_protocol_and_healthy_only_guards(self):
        from experiments.l23net_analysis.run_l23net_reference60 import validate_configuration
        from experiments.l23net_analysis.run_l23net_no_field_replay import _replay_contract
        cfg = self.config()
        contract = _replay_contract(cfg,624)
        contract.pop('experiment_seed'); contract.pop('condition'); contract['simulation'].pop('MDD')
        contract['simulation']['duration_ms'] = 28000.
        contract['stimulation_enabled'] = False
        gate = {'sha256':'test', 'reference_contract':contract}
        with patch.object(s0,'load_gate', return_value=gate), patch(
                'experiments.l23net_analysis.run_l23net_reference60.load_qualification', return_value={}):
            self.assertEqual(validate_configuration(cfg,624),(60,40000))
            for key,value in [('env.simulation.MDD',True),('env.simulation.DRUG',True),
                              ('analysis.arm','fixed_low_beta'),('experiment.seed',8201),
                              ('analysis.mode','qualification'),('analysis.excluded_ms',4000),
                              ('env.network.dt',.05),('env.ts.apply',False),('experiment.debug',True)]:
                changed = copy.deepcopy(cfg)
                from omegaconf import OmegaConf
                OmegaConf.update(changed,key,value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_configuration(changed,624)
            with self.assertRaises(ValueError): validate_configuration(cfg,2)

    def test_qualification_failure_is_not_bypassed(self):
        from experiments.l23net_analysis.run_l23net_reference60 import validate_configuration
        # Invalid S0 also fails closed before any simulation.
        with patch.object(s0,'load_gate', side_effect=ValueError('invalid prerequisite')):
            with self.assertRaises(ValueError): validate_configuration(self.config(),624)

    def test_quota_allocation_and_fail_closed(self):
        jobs = allocate({'sj53':5.61,'fa32':7.37,'ny83':60.11})
        self.assertEqual([sum(j['project']==k for j in jobs) for k in ('sj53','fa32','ny83')],[1,2,13])
        self.assertAlmostEqual(p.RESERVATION_KSU,2.496)
        for balance in ({'sj53':1.,'fa32':1.,'ny83':1.}, {'sj53':float('nan'),'fa32':7.,'ny83':60.},
                        {'sj53':5.,'fa32':-1.,'ny83':60.}):
            with self.assertRaises(ValueError): allocate(balance)

    def test_exact_frozen_seed_set_and_manifest_roundtrip(self):
        self.assertEqual(p.PROTOCOL, json.loads(json.dumps(p.PROTOCOL)))
        self.assertEqual([j['run']['seed'] for j in p.jobs()], list(range(8101,8117)))
        self.assertTrue(all(j['run']['condition']=='reference' and j['run']['arm']=='sham' for j in p.jobs()))

    def test_prefix_hash_is_chunk_independent_and_detects_early_not_late_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/'a.h5',Path(tmp)/'b.h5'
            for path,n,chunk in [(a,40,7),(b,80,13)]:
                with h5py.File(path,'w') as h:
                    h.create_dataset('sample_time_ms',data=np.arange(n,dtype=float),chunks=(chunk,))
                    h.create_dataset('eeg_v',data=np.arange(n,dtype=float)[None],chunks=(1,chunk))
                    h.create_dataset('dipole_nA_um',data=np.tile(np.arange(n,dtype=float),(3,1)),chunks=(3,chunk))
            first = p.prefix_hashes(a,duration_ms=40,dt_ms=1,window_ms=10)
            self.assertEqual(first,p.prefix_hashes(b,40,1,10))
            with h5py.File(b,'a') as h: h['eeg_v'][0,60] += 1
            self.assertEqual(first,p.prefix_hashes(b,40,1,10))
            with h5py.File(b,'a') as h: h['eeg_v'][0,4] += 1
            self.assertNotEqual(first,p.prefix_hashes(b,40,1,10))
            with self.assertRaises(ValueError): p.prefix_hashes(a,81,1,10)

    def test_prefix_failure_blocks_calibration(self):
        source = {'prefix':{'structure_sha256':'expected'}}
        with patch.object(p,'prefix_record',return_value={'structure_sha256':'wrong'}):
            with self.assertRaisesRegex(ValueError,'prefix mismatch'): p.check_prefix({},'unused',source)

    def test_targets_use_all_reference_seeds_and_preserve_screen(self):
        gate = {'targets':{'20s':{'unchanged':'original screening'}}}
        rows = []
        for i,seed in enumerate(p.SEEDS):
            baseline = np.array([-19.,-19.5,-20.])+i*.01
            rows.append({'seed':seed,'condition':'reference',
                         'log_powers':{'baseline':baseline.tolist(),'baseline_10s':baseline.tolist(),
                                       'plateau':(baseline+.2).tolist(),'washout':(baseline-.1).tolist()},
                         'excluded_plateau':{b:baseline.tolist() for b in s0.BANDS}})
        targets,drift = fit_targets(rows,gate)
        self.assertEqual(targets['baseline_screen_unchanged'],gate['targets']['20s'])
        self.assertEqual(targets['plateau']['interval_s'],[29,49])
        self.assertEqual(targets['washout']['n_structures'],16)
        self.assertAlmostEqual(drift['plateau_theta']['mean'],.2)
        self.assertIn('q_bh_six_drift_tests',drift['washout_low_beta'])
        for changed in (rows[:-1],rows+[rows[0]], [dict(r,condition='mdd') for r in rows]):
            with self.assertRaises(ValueError): fit_targets(changed,gate)

    def test_self_hash_tampering_and_frozen_output(self):
        from experiments.l23net_analysis.r1_analysis import frozen_json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'value.json'; value = p.signed({'a':1})
            frozen_json(path,value); frozen_json(path,value)
            self.assertEqual(p.load_signed(path),value)
            with self.assertRaises(ValueError): frozen_json(path,p.signed({'a':2}))
            value['a']=2;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): p.load_signed(path)

    def test_analysis_orchestration_freezes_only_complete_valid_suite(self):
        # Isolate orchestration from the separately tested native-trace auditor
        # and causal filter. Use real epoch/PSD/target fitting and PBS parsing.
        from experiments.l23net_analysis import analyze_l23net_reference60 as analysis
        for fail_seed in (None,8102):
            with self.subTest(fail_seed=fail_seed), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);jobs=p.jobs();code=p.code_hashes()
                for j in jobs:j['project']='ny83'
                gate={'sha256':'gate','targets':{'20s':{'frozen':'screen'}}}
                sources=p.signed({'s0_sha256':'gate','sources':[{'seed':s} for s in p.SEEDS]})
                manifest={'stage':'reference60','protocol':p.PROTOCOL,'code_sha256':code,'s0_sha256':'gate',
                          'reference_sources_sha256':sources['sha256'],'jobs':jobs,'commit':'commit',
                          'submission_status':'submitted'}
                (root/'submission.json').write_text(json.dumps(manifest))
                (root/'reference_sources.json').write_text(json.dumps(sources))
                t=np.arange(1,15001)/250
                eeg=sum(a*np.sin(2*np.pi*f*t) for a,f in [(1e-9,6.1),(2e-9,10.1),(3e-9,14.1)])
                for j in jobs:
                    folder=root/j['name'];folder.mkdir()
                    (folder/'worker_exit_code.txt').write_text('0')
                    (folder/'git_commit.txt').write_text('commit')
                    (folder/'l23net_s1_run.json').write_text('{}')
                    (folder/'pbs.out').write_text('Service Units: 1600\nWalltime Used: 01:16:00\nMemory Used: 160GB\nExit Status: 0\n')
                    with h5py.File(folder/'l23net_s1_trace.h5','w') as h:
                        h['eeg_v']=eeg[None,:];h['field_left_boundary_v_per_m']=np.zeros(len(t))
                def audit(directory,gate):
                    seed=int(directory.name.split('_')[-1])
                    if seed==fail_seed:raise ValueError('deliberate raw audit failure')
                    return {'configuration':{'analysis':{'mode':'reference_calibration','arm':'sham'}},
                            'seed_manifest':{'experiment_seed':seed},'replay_contract':{'condition':'reference',
                                'simulation':{'MDD':False,'DRUG':False}},
                            'window_protocol':{'max_field_v_per_m':0,'nonzero_field_samples':0,
                                'reference60':{'code_sha256':code,'protocol_sha256':p.canonical_json_sha256(p.PROTOCOL)}},
                            'memory_snapshots':[{'simulated_ms':8000,'rss_gib':{'sum':190.}}],
                            'artifacts':{'trace_summary':{'content_sha256':'trace'}}}
                with patch.object(s0,'load_gate',return_value=gate), patch.object(analysis.s1,'load_qualification'), \
                     patch.object(analysis.s1,'audit_run',side_effect=audit), patch.object(p,'check_prefix'), \
                     patch.object(analysis.s1,'mean_rates',return_value={'HL23PYR':1.}), \
                     patch.object(s0.CausalEEG,'append',return_value=(t,eeg)):
                    result=analysis.analyze_suite(root)
                self.assertEqual(result['completed_valid_runs'],15 if fail_seed else 16)
                self.assertEqual((root/'reference60_target.json').exists(),not bool(fail_seed))
                self.assertEqual((root/'reference60_summary.json').exists(),not bool(fail_seed))
                self.assertTrue((root/'reference60_status.json').is_file())
                if not fail_seed:
                    self.assertEqual(result['status'],'reference_calibrated')
                    self.assertAlmostEqual(result['actual_ksu'],25.6)
                    self.assertEqual(p.load_signed(root/'reference60_target.json')['status'],'reference_calibrated')

    def test_submission_all_eligible_and_partial_failure_persisted(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            manifest = {'commit':'abc','jobs':allocate({'sj53':5.61,'fa32':7.37,'ny83':60.11}), 'submission_status':'submitting'}
            with patch('subprocess.check_output',side_effect=[f'{i}.gadi-pbs' for i in range(16)]) as mocked:
                submit_jobs(Path(tmp),manifest,16)
                self.assertTrue(all('-W' not in c.args[0] for c in mocked.call_args_list))
            self.assertEqual(json.loads((Path(tmp)/'submission.json').read_text())['submission_status'],'submitted')
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            manifest = {'commit':'abc','jobs':p.jobs()[:2], 'submission_status':'submitting'}
            for j in manifest['jobs']:j['project']='sj53'
            with patch('subprocess.check_output',side_effect=['1.gadi-pbs',subprocess.CalledProcessError(1,'qsub')]):
                with self.assertRaises(subprocess.CalledProcessError):submit_jobs(Path(tmp),manifest,1)
            saved=json.loads((Path(tmp)/'submission.json').read_text())
            self.assertEqual(saved['submission_status'],'partial_failure')
            self.assertEqual(saved['jobs'][0]['job_id'],'1.gadi-pbs')
            self.assertTrue(saved['jobs'][1]['submission_attempted'])

    def test_worker_shell_and_no_stdin_or_resource_regression(self):
        path=p.ROOT/'experiments/l23net_analysis/nci/run_l23net_reference60_worker.sh'
        subprocess.run(['bash','-n',str(path)],check=True)
        source=path.read_text()
        for text in ('walltime=02:00:00','ncpus=624','mem=256GB','env.simulation.MDD=false',
                     'analysis.mode=reference_calibration','analysis=l23net_reference60','< /dev/null'):
            self.assertIn(text,source)


if __name__ == '__main__':
    unittest.main()
