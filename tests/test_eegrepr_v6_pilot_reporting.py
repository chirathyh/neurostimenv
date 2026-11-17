"""Reporting contracts and a visibly synthetic engineering fixture."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.eeg_representation.h1.v6_pilot import reporting


def engineering_fixture(root):
    """Purely synthetic reporting values, never presented as simulator evidence."""
    root=Path(root);rng=np.random.default_rng(761)
    targets=[('waveform',h,w,'V','V^2') for h in (0,2,10) for w in (32,128)]
    targets += [('fourier',h,1000,'V','V^2') for h in (0,10)]
    targets += [(kind+'_'+band,h,1000,'V^2' if kind=='energy' else 'log(V^2 / 1 V^2)','V^4' if kind=='energy' else 'dimensionless^2') for kind in ('energy','logenergy') for band in ('alpha','beta_low','beta_high') for h in (0,10)]
    tables={name:[] for name in reporting.TABLES}
    for acq in ('A0','A1','A2'):
        for target,h,w,units,risk_units in targets:
            job=acq+'__'+('T0' if target=='waveform' else target)+f'_l{h}_d{w}'
            baseline=1e-18 if risk_units=='V^2' else (1e-36 if risk_units=='V^4' else 2.)
            full=101.2 if h==0 else 99.3
            for method,error in [('complete_state_candidate',115.3),('complete_state_selected',100.),('state_variance',89.2),('eeg_short',100.8),('eeg_full',full),('PCA_d8_B32',98.2),('training_mean',100.)]:
                tables['h1_information'].append(dict(job=job,acquisition=acq,target=target,h_ms=h,duration_ms=w,method=method,information=method.replace('_',' '),evidence_type='estimated_variance' if method=='state_variance' else 'fitted',risk_physical=baseline*error/100,reference_risk_physical=baseline,error_baseline100=error,state_variance_physical=baseline*.892,population='matched_state_engineering_cohort',circuits=2,contexts=8,futures=4,status='complete',uncertainty=[error-4,error+4],actually_trained=method=='complete_state_candidate'))
            for method,status,err in [('theoretical_zero','theoretical',0.),('ideal_eeg','unknown',None)]:
                tables['h1_information'].append(dict(job=job,acquisition=acq,target=target,h_ms=h,duration_ms=w,method=method,information=method,evidence_type=status,status=status,error_baseline100=err,population='mathematical_statement'))
            if target=='waveform':
                for history,error in [('short',100.8),('full',full)]:tables['h1_comparisons'].append(dict(job=job,acquisition=acq,target=target,h_ms=h,duration_ms=w,history=history,error_baseline100=error,relative_mse_reduction_pp=100-error,paired_difference=-.7,uncertainty='Descriptive engineering fixture only',population='matched_state_engineering_cohort',status='complete'))
            h4=dict(job=job,acquisition=acq,target=target,h_ms=h,duration_ms=w,physical_units=units,risk_units=risk_units,floor_reference_ratio=.892,error_baseline100=full,relative_mse_reduction_pp=100-full,uncertainty='Deferred: tiny pilot',population='matched_state_engineering_cohort',method='eeg_full',status='complete')
            tables['h4_targets'].append(h4);tables['full_grid'].append(h4.copy())
            for context in ('engineering-circuit1-R0','engineering-circuit2-R10'):
                for m in (1,2,4,8,16,32):tables['h2_precision'].append(dict(job=job,context_id=context,M=m,L=32,mean_error_corrected=.9/m-.9/32,variance_error=None if m==1 else .4/(m-1),uncertainty='Dependent-subset diagnostic',reference_uncertainty='Finite L32, engineering fixture',status='complete'))
    for rep,nmse,ce in [('Feature18',.98,.2),('PCA',.91,1.4),('forecast',1.01,1.1)]:
        for bits in (None,32):tables['h3_retention'].append(dict(representation=rep,projection='d8',budget_bits=bits,reconstruction_nmse=nmse+(.003 if bits else 0),future_error_baseline100=102 if rep=='forecast' else 99,carrier_ce_bits=ce+(.03 if bits else 0),accuracy=.75,balanced_accuracy=.5,population='engineering_fixture',uncertainty='Unavailable: template only',status='complete'))
    tables['findings']=[dict(question='Does the reporting template retain negative state-model results?',observed='State candidate error 115.3 versus mean 100; selected mean policy 100',interpretation='The nonmean fitted candidate remains displayed.',limitation='Synthetic engineering values, not scientific findings.',later_study='Estimate effects on new registered circuits.',status='engineering')]
    tables['run_status']=[dict(stage='fixture',category='engineering diagnostics',status='complete',count=0,details='Zero NEURON calls. No biological claim.')]
    root.mkdir(parents=True,exist_ok=True);manifest=[]
    for role in ('schema','fitted_state','predictions','selection','losses','training_curve'):
        for index in range(1 if role=='schema' else 60):
            p=root/('fixture_'+role+('' if index==0 else '_'+str(index))+'.json');p.write_text(json.dumps({'purpose':'engineering_fixture','role':role}));manifest.append(dict(path=p.name,sha256=reporting._hash(p),role=role))
    state=dict(status='complete',artifact_manifest=manifest)
    for key in ('input_coverage_complete','future_invariance_passed','fitted_state_dependent_candidate','dev_selection_complete','held_out_predictions_complete','target_population_matching_passed'):state[key]=True
    t=np.arange(1,65)*2.;future=rng.normal(0,1e-9,(8,64))
    return dict(schema_version=reporting.SCHEMA,purpose='engineering_fixture',status='ready_for_report',required_stages={'fixture':'complete'},tables=tables,
        complete_state_verification=state,completed_allocation={'new_NEURON_invocations':0,'contexts':'synthetic arrays only'},runtime={'actual_workers':'not run; template fixture','worker_hours':'unavailable'},
        coverage={'inventory':[dict(field='Synthetic field coverage fixture',units='arbitrary',shape='test only',identity='not a physical export',status='engineering')]},
        figures={'same_state_futures':dict(times_ms=t.tolist(),futures=future.tolist(),past_times_ms=(-t[::-1]).tolist(),past=np.zeros(64).tolist(),units='V (synthetic fixture)',context_id='engineering-only')})


class PilotReportTests(unittest.TestCase):
    def test_metrics_retain_physical_scale_negative_excess_and_nulls(self):
        result=reporting.normalize_results({'tables':{'h1_information':[dict(risk_physical=2.,reference_risk_physical=1.,state_variance_physical=3.,status='complete'),dict(evidence_type='unknown',status='unknown',risk_physical=None)]}})
        rows=result['tables']['h1_information'];self.assertEqual(rows[0]['error_baseline100'],200);self.assertEqual(rows[0]['relative_mse_reduction_pp'],-100);self.assertEqual(rows[0]['unresolved_excess_physical'],-1);self.assertIsNone(rows[1]['risk_physical'])
        with self.assertRaisesRegex(ValueError,'mismatch'):reporting.normalize_results({'tables':{'h1_information':[dict(risk_physical=2,reference_risk_physical=1,error_baseline100=100)]}})
        with self.assertRaisesRegex(ValueError,'M=1'):reporting.normalize_results({'tables':{'h2_precision':[dict(M=1,L=32,variance_error=0)]}})
        with self.assertRaisesRegex(ValueError,'theoretical'):reporting.normalize_results({'tables':{'h1_information':[dict(evidence_type='theoretical',status='complete',error_baseline100=0)]}})

    def test_complete_state_gate_cannot_be_satisfied_by_pdf_or_mean(self):
        with tempfile.TemporaryDirectory() as d:
            fixture=engineering_fixture(d);r=reporting.normalize_results(fixture);check=reporting.completion_checks(d,r)
            self.assertFalse(check['experimental_completion_eligible']);self.assertEqual(len(check['issues']),1)
            r['purpose']='exploratory_reporting_pilot'
            self.assertTrue(reporting.completion_checks(d,r)['experimental_completion_eligible'])
            r['tables']['h1_information']=[row for row in r['tables']['h1_information'] if row.get('method')!='complete_state_candidate']
            self.assertFalse(reporting.completion_checks(d,r)['experimental_completion_eligible'])
            r=reporting.normalize_results(fixture);r['purpose']='exploratory_reporting_pilot';Path(d,'fixture_predictions.json').write_text('changed')
            self.assertTrue(any('hash mismatch' in x for x in reporting.completion_checks(d,r)['issues']))

    def test_unrounded_exports_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as d:
            result=reporting.normalize_results({'tables':{'h1_information':[dict(risk_physical=1.234567891234e-18,reference_risk_physical=2e-18,status='complete'),dict(risk_physical=None,status='unavailable')]}})
            reporting.write_tables(d,result)
            with Path(d,'h1_information.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(float(rows[0]['risk_physical']),1.234567891234e-18);self.assertEqual(rows[1]['risk_physical'],'')
            with np.load(Path(d,'h1_information.npz'),allow_pickle=False) as npz:self.assertFalse(npz['risk_physical__available'][1]);self.assertEqual(npz['risk_physical'][0],1.234567891234e-18)
            self.assertTrue(Path(d,'h1_information.tex').exists())

    @unittest.skipUnless(reporting.dependencies()['status']=='complete','PDF dependencies unavailable')
    def test_empty_report_builds_but_does_not_claim_pilot_completion(self):
        with tempfile.TemporaryDirectory() as d:
            receipt=reporting.report(d,{})
            self.assertEqual(receipt['status'],'complete');self.assertFalse(receipt['experimental_completion_eligible'])
            self.assertTrue(receipt['pdf_verification']['all_pages_rendered']);self.assertEqual(len(receipt['figures']),1)
            self.assertTrue(Path(d,'report/pilot_findings.pdf').stat().st_size>10000)

    @unittest.skipUnless(reporting.dependencies()['status']=='complete','PDF dependencies unavailable')
    def test_full_engineering_report_all_pages_and_numerical_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            fixture=engineering_fixture(d);receipt=reporting.report(d,{},fixture)
            self.assertEqual(receipt['status'],'complete');self.assertFalse(receipt['experimental_completion_eligible'])
            self.assertTrue(receipt['pdf_verification']['all_pages_rendered']);self.assertEqual(receipt['pdf_verification']['errors'],[])
            self.assertEqual(receipt['tables']['full_grid']['rows'],60)
            self.assertEqual(len(receipt['figures']),7)
            self.assertIn('ENGINEERING',__import__('pypdf').PdfReader(Path(d,'report/pilot_findings.pdf')).pages[0].extract_text())
            displayed=json.loads(Path(d,'report/tables/h1_information.json').read_text())
            self.assertEqual(displayed[0]['risk_physical'],fixture['tables']['h1_information'][0]['risk_physical'])
            self.assertTrue(all(p.get('render_sha256') for p in receipt['pdf_verification']['pages']))

if __name__=='__main__':unittest.main()
