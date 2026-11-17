"""No-simulation recovery guards and isolated fitting orchestration."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager,redirect_stdout
from unittest.mock import patch,MagicMock
import importlib
import io
import json
import tempfile
import unittest

from experiments.eeg_representation.h1.v6_pilot import recovery,config,budget,generation,learning,cli
from experiments.eeg_representation.h1.artifacts import atomic_json,file_hash


class RecoveryTests(unittest.TestCase):
    def test_native_imports_and_every_native_entry_are_blocked_without_ledger_charge(self):
        with tempfile.TemporaryDirectory() as temp:
            budget.initialize(temp);before=recovery.protected_snapshot(temp)
            with recovery.native_disabled():
                for name in ('neuron','LFPy','mpi4py'):
                    with self.assertRaisesRegex(ImportError,'prohibited'):importlib.import_module(name)
                for fn in (budget.begin,generation._run,generation._worker):
                    with self.assertRaisesRegex(RuntimeError,'prohibits every native'):fn(temp,'not-allowed',{})
            recovery.assert_protected(temp,before)
            self.assertEqual(budget.summary(temp)['attempted_native_calls'],0)

    def test_verifier_rejects_corruption_and_escaping_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'data';path.write_bytes(b'frozen');v=recovery.Verifier()
            v.record(root,dict(path='data',sha256=file_hash(path),bytes=6))
            self.assertEqual(len(v.files),1)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'size mismatch'):v.record(root,dict(path='data',sha256='irrelevant',bytes=6))
            with self.assertRaisesRegex(ValueError,'escapes'):v.record(root,dict(path='../other',sha256='irrelevant'))

    def test_protected_accounting_cannot_change_even_if_count_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);budget.initialize(root);before=recovery.protected_snapshot(root)
            atomic_json(root/'pilot_provenance.json',{'identity':'changed'})
            with self.assertRaisesRegex(RuntimeError,'Protected native accounting'):recovery.assert_protected(root,before)

    def test_read_only_preflight_stops_at_missing_upstream_completion(self):
        c=config.resolve();adapter={'fixed':'capture'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.zip';source.write_bytes(b'archive')
            c['source_archive_sha256']=file_hash(source);atomic_json(root/'pilot_config.json',c)
            identity=dict(config_hash=cli.canonical_hash(c),device='cuda',dtype=c['dtype'],workers=8,threads_per_worker=1,generation_adapter=adapter)
            atomic_json(root/'pilot_provenance.json',{'identity':identity,'commit':'frozen'})
            atomic_json(root/'progress/preflight.json',{'status':'failed'})
            before={p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(generation,'adapter_contract',return_value=adapter),patch.object(generation,'_frozen',return_value={'selection':{'state_cohort':[]}}):
                with self.assertRaisesRegex(ValueError,'completed upstream stage: preflight'):
                    recovery.verify_upstream(root,source,'cuda')
            after={p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_existing_tcn_uses_original_signature_validation_without_child(self):
        c=config.resolve();job=c['neural_jobs'][0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dest=root/'matched/learners'/job/'tcn_64';atomic_json(dest/'selection.json',{'status':'complete'})
            original=MagicMock(return_value={'status':'complete'})
            with patch.object(recovery.subprocess,'Popen',side_effect=AssertionError('No new process for completed fit')):
                result=recovery.run_tcn_child(root,root/'recovery',job,64,'cuda',original,{}, {},dest,c)
            self.assertEqual(result['status'],'complete');original.assert_called_once_with({}, {},dest,c,64,'cuda',True)

    def test_aborted_child_is_recorded_and_never_retried(self):
        c=config.resolve();job=c['neural_jobs'][0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dest=root/'matched/learners'/job/'tcn_64';original=MagicMock()
            child=MagicMock(pid=771);child.wait.return_value=-6
            with patch.object(recovery.subprocess,'Popen',return_value=child) as spawn:
                with self.assertRaisesRegex(RuntimeError,'exit -6; no automatic retry'):
                    recovery.run_tcn_child(root,root/'recovery',job,64,'cuda',original,{}, {},dest,c)
            self.assertEqual(spawn.call_count,1);original.assert_not_called()
            command=spawn.call_args.args[0]
            self.assertIn('faulthandler',command);self.assertEqual(command[command.index('--device')+1],'cuda')
            record=json.loads((root/'recovery/tcn'/job/'h64/exit.json').read_text())
            self.assertEqual(record['returncode'],-6);self.assertEqual(record['native_calls_allowed'],0)

    def test_check_only_never_executes_or_changes_existing_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);budget.initialize(root);before=recovery.protected_snapshot(root)
            with patch.object(recovery,'verify_upstream',return_value={'status':'complete','native_calls_allowed':0}),patch.object(recovery,'execute') as execute,redirect_stdout(io.StringIO()):
                self.assertEqual(recovery.main(['--root',temp,'--source-archive','frozen.zip','--check-only']),0)
            execute.assert_not_called();recovery.assert_protected(root,before)

    def test_execute_only_ordinary_fit_and_downstream_preserves_original_progress(self):
        from experiments.eeg_representation.h1.v6_pilot import analysis,precision,reporting,exporting
        c=config.resolve()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);budget.initialize(root);atomic_json(root/'pilot_config.json',c);atomic_json(root/'pilot_provenance.json',{'identity':{'device':'cuda'}})
            atomic_json(root/'progress/ordinary_fit.json',{'status':'running','pid':123,'started_unix':99})
            (root/'report').mkdir();(root/'report/pilot_findings.pdf').write_bytes(b'pdf')
            atomic_json(root/'report/receipt.json',{'status':'complete'});atomic_json(root/'pilot_results.json',{'status':'fixture'})
            before=recovery.protected_snapshot(root);called=[]
            def fit(*args):called.append('ordinary');return {'status':'complete'}
            def scoring(*args):called.append('score');return {}
            def mc(*args):called.append('precision');return {}
            report={'status':'complete','pdf_verification':{'automated_checks':{'fixture':True}}}
            with recovery.native_disabled(),patch.object(analysis,'fit_ordinary',side_effect=fit),patch.object(analysis,'score_models',side_effect=scoring),patch.object(precision,'run',side_effect=mc),patch.object(cli,'_assemble',return_value={}),patch.object(reporting,'report',return_value=report),patch.object(cli,'verify_report'),patch.object(exporting,'export',return_value={'status':'complete','path':'output','sha256':'fake'}):
                result=recovery.execute(root,'source.zip',{'device':'cuda'})
            self.assertEqual(called,['ordinary','score','precision']);self.assertEqual(result['recovery']['native_calls'],0)
            saved=list((root/'recovery').glob('*/initial_progress/ordinary_fit.json'))
            self.assertEqual(len(saved),1);self.assertEqual(json.loads(saved[0].read_text())['pid'],123)
            recovery.assert_protected(root,before)

if __name__=='__main__':unittest.main()
