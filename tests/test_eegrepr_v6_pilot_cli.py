"""Orchestration gates, no native simulations or network calls."""
from contextlib import contextmanager,redirect_stdout
from copy import deepcopy
import io,json,shutil,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import numpy as np
from experiments.eeg_representation.h1.v6_pilot import cli,config,budget,learning
from experiments.eeg_representation.storage import atomic_json,file_hash
from tests.test_eegrepr_v6_pilot_learning import panel as engineering_panel


class PilotCliTests(unittest.TestCase):
    def test_initialization_debits_actual_implementation_calls_once_and_locks_capture(self):
        c=config.resolve()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'
            cli.initialize(c,root,'cpu',False)
            self.assertEqual(budget.summary(root)['implementation_engineering_calls'],21)
            self.assertEqual(budget.summary(root)['remaining_native_calls'],139)
            cli.initialize(c,root,'cpu',True)
            self.assertEqual(budget.summary(root)['attempted_native_calls'],21)
            with patch.object(cli,'generation_adapter_contract',return_value={'changed':'capture source'}):
                with self.assertRaisesRegex(ValueError,'identity changed'):cli.initialize(c,root,'cpu',True)

    def test_real_fitted_artifacts_required_for_every_state_setting(self):
        c=config.resolve();p,x,m=engineering_panel();cohorts={s:list(map(str,r['context_id'])) for s,r in p.items()};jobs=config.jobs()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);first=root/'first';learning.ridge(p,x,m,first,c)
            for job in jobs:shutil.copytree(first,root/'matched/learners'/job/'complete_state')
            fit=dict(status='complete',jobs=[dict(job=j) for j in jobs],actually_fitted_state_dependent_candidate=True,
                coverage={'input_coverage_complete':True},cohorts=cohorts)
            atomic_json(root/'state/fit.json',fit)
            with patch.object(cli,'verify_replay',return_value={'status':'complete'}),patch.object(cli.analysis,'load_panel',return_value=(p,m)):
                gate=cli.verify_state_fit(c,root,cohorts,jobs,fit)
                self.assertTrue(gate['held_out_evaluated']);self.assertEqual(gate['jobs'],60)
                (root/'matched/learners'/jobs[-1]/'complete_state/predictions.npz').unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):cli.verify_state_fit(c,root,cohorts,jobs,fit)

    def test_failed_state_gate_prevents_h2_native_allocation(self):
        c=config.resolve();cohort=[dict(context_id=f'{s}-context',split=s) for s in ('train','development','calibration','evaluation')]
        args=SimpleNamespace(source_archive='archive.zip',native_source='native',resume=True,device='cpu',destination=None)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(cli,'preflight',return_value={'status':'complete'}),patch.object(cli,'file_hash',return_value=c['source_archive_sha256']),patch.object(cli,'EvidenceSource'),patch.object(cli,'import_evidence'),patch.object(cli,'smoke',return_value={'status':'complete'}),patch.object(cli.generation,'prepare',return_value={'selection':{'state_cohort':cohort}}),patch.object(cli,'derive',return_value={'completed_jobs':60}),patch.object(cli.panels,'derive_matched'),patch.object(cli.generation,'concurrency_check'),patch.object(cli.generation,'measured_plan'),patch.object(cli,'verify_measured_plan'),patch.object(cli.generation,'replay_controls'),patch.object(cli,'verify_replay'),patch.object(cli.generation,'recover_states'),patch.object(cli.analysis,'fit_complete_state',return_value={'status':'complete','jobs':[]}),patch.object(cli.generation,'extend_h2') as h2:
                with self.assertRaisesRegex(ValueError,'60-setting'):cli.complete(c,root,args)
                h2.assert_not_called()
                self.assertEqual(json.loads((root/'progress/state_model_gate.json').read_text())['status'],'failed')

    def test_plan_requires_measured_eight_process_gate_and_total_capacity(self):
        c=config.resolve()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);budget.initialize(root);budget.record_implementation_debit(root,21,{'source':'journal'})
            atomic_json(root/'concurrency_check.json',dict(status='complete',actual_worker_processes=8,simultaneous_native_overlap_seconds=1,host=cli.platform.node()))
            plan=dict(status='measured_planned',workers=8,threads_per_worker=1,host=cli.platform.node(),projected_total_calls=127,prospective_remaining_calls=106)
            with patch.object(cli.generation,'preflight',return_value={'passed':True,'blockers':[]}):
                self.assertTrue(cli.verify_measured_plan(c,root,plan)['eligible_for_bounded_pilot'])
                with self.assertRaises(RuntimeError):cli.verify_measured_plan(c,root,{**plan,'projected_total_calls':161})
            atomic_json(root/'concurrency_check.json',dict(status='complete',actual_worker_processes=7,simultaneous_native_overlap_seconds=1,host=cli.platform.node()))
            with self.assertRaisesRegex(ValueError,'eight-process'):cli.verify_measured_plan(c,root,plan)

    def test_failed_replay_invariance_stops_before_main_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);atomic_json(root/'design.json',{'selection':{'h2_contexts':[{'context_id':'a'},{'context_id':'b'}]}})
            replay=dict(status='complete',all_exact=True,audits=[dict(context_id=x,full_trace_identical=True,complete_state_future_invariance={'passed':False}) for x in ('a','b')])
            with self.assertRaisesRegex(ValueError,'invariance'):cli.verify_replay(root,replay)

    def test_report_only_acquires_output_lock_and_never_fits_or_generates(self):
        c=config.resolve();events=[]
        @contextmanager
        def locked(root):
            events.append('enter');yield;events.append('exit')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);atomic_json(root/'pilot_config.json',c)
            def report(*a,**kw):
                self.assertEqual(events,['enter']);return {'status':'incomplete','experimental_completion_eligible':False}
            with patch.object(cli,'output_lock',side_effect=locked),patch.object(cli.reporting,'report',side_effect=report),patch.object(cli.generation,'concurrency_check') as native,patch.object(cli.analysis,'fit_complete_state') as fitting,redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['report-only','--root',str(root)]),0)
                native.assert_not_called();fitting.assert_not_called()
            self.assertEqual(events,['enter','exit'])

    def test_findings_include_measured_h2_and_matched_task_distinctions(self):
        tables=dict(h1_information=[],h2_precision=[dict(job='A0__T0_l0_d128',context_id=x,M=32,L=32,mean_error_corrected=-.1,mean_error_units='V^2',variance_error=-.02,variance_error_units='V^4') for x in ('context-a','context-b')],
            h3_retention=[dict(population='Matched two inspected circuits',method=x,past_reconstruction_nmse=.8,future_error_baseline100=102,carrier_cross_entropy_bits=1.2,rhythm_balanced_accuracy=.5,status='measured') for x in ('feature18_B32','pca_B32','forecast_B32')])
        findings=cli.scientific_findings(tables)
        self.assertEqual(len(findings),5)
        self.assertTrue(all(r['later_study'] for r in findings))
        self.assertIn('-0.1',findings[0]['observed'])
        self.assertTrue(all('carrier CE 1.2' in r['observed'] for r in findings[2:]))
        self.assertTrue(all('Two inspected' in r['limitation'] for r in findings[2:]))

    def test_pdf_receipt_checks_every_render_hash(self):
        c=config.resolve();results={'status':'ready_for_report','tables':{}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);atomic_json(root/'pilot_results.json',results);(root/'report/rendered').mkdir(parents=True)
            (root/'report/pilot_findings.pdf').write_bytes(b'fixture-pdf');(root/'report/rendered/page-1.png').write_bytes(b'fixture-image')
            receipt=dict(status='complete',experimental_completion_eligible=True,pdf_sha256=file_hash(root/'report/pilot_findings.pdf'),
                results_hash=cli.reporting._canonical(cli.reporting.normalize_results(results)),config_hash=cli.reporting._canonical(c),
                pdf_verification=dict(status='complete',all_pages_rendered=True,errors=[],pages=[dict(render='rendered/page-1.png',render_sha256=file_hash(root/'report/rendered/page-1.png'))]))
            with patch.object(cli.reporting,'completion_checks',return_value={'experimental_completion_eligible':True}):
                cli.verify_report(root,c,receipt)
                (root/'report/rendered/page-1.png').write_bytes(b'altered')
                with self.assertRaisesRegex(ValueError,'Rendered PDF'):cli.verify_report(root,c,receipt)

if __name__=='__main__':unittest.main()
