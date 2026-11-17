"""Eight-worker reporting pilot, isolated from all confirmation execution paths."""
from pathlib import Path
import argparse,copy,json,os,platform,subprocess,sys,time,traceback
import numpy as np
from ..artifacts import ROOT,atomic_json,file_hash,output_lock
from ..targets import canonical_hash
from ..v6.evidence import EvidenceSource,import_evidence
from ..v6.derivation import derive
from . import config as cfg,budget,generation,analysis,panels,reporting,exporting


def _read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default

def initialize(c,root,device,resume):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);p=root/'pilot_provenance.json'
    identity=dict(config_hash=canonical_hash(c),device=device,dtype=c['dtype'],workers=8,threads_per_worker=1,
        generation_adapter=generation_adapter_contract())
    if p.exists():
        old=_read(p)
        if not resume:raise FileExistsError('Pilot root exists; explicit resume required')
        if old['identity']!=identity:raise ValueError('Pilot config/device/precision/worker identity changed; choose new root')
    else:
        allowed={'logs','.eegrepr.lock'}
        if any(x.name not in allowed for x in root.iterdir()):raise ValueError('Nonempty unversioned pilot root refused')
        old=dict(schema_version=c['schema_version'],identity=identity,purpose=c['purpose'],created_unix=time.time(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),host=platform.node(),
            environment=dict(python=sys.version,executable=sys.executable),
            policy='Native generation, complete-state export, learner and report identities are separate; no confirmation or power-driven expansion.')
        atomic_json(p,old);atomic_json(root/'pilot_config.json',c)
    budget.initialize(root,c['max_new_invocations'])
    implementation=ROOT/'docs/eegrepr_v6_reporting_pilot/engineering_invocations.json'
    debit=_read(implementation)
    if debit is None or sum(int(x['actual_neuron_invocations']) for x in debit['steps'])!=int(debit['actual_neuron_invocations']):
        raise ValueError('Missing or inconsistent implementation NEURON invocation journal')
    budget.record_implementation_debit(root,debit['actual_neuron_invocations'],dict(
        path=str(implementation.relative_to(ROOT)),sha256=file_hash(implementation),scope=debit['scope'],host=debit['host']))
    atomic_json(root/'engineering/implementation_invocations.json',debit)
    return old


def generation_adapter_contract():
    """Freeze new complete-state capture/physics without relabeling old recordings."""
    return generation.adapter_contract()


def verify_measured_plan(c,root,plan):
    root=Path(root);gate=_read(root/'concurrency_check.json');hardware=generation.preflight(c,root)
    if plan.get('status')!='measured_planned' or plan.get('workers')!=8 or plan.get('threads_per_worker')!=1:
        raise ValueError('Measured eight-worker plan missing or unsupported')
    if not gate or gate.get('status')!='complete' or gate.get('actual_worker_processes')!=8 or gate.get('simultaneous_native_overlap_seconds',0)<=0:
        raise ValueError('Actual concurrent eight-process simulation check did not pass')
    if gate.get('host')!=platform.node() or plan.get('host')!=platform.node():raise ValueError('Timing measurements came from another host')
    if not hardware['passed']:raise RuntimeError('Measured allocation hardware blocker: '+str(hardware['blockers']))
    if plan['projected_total_calls']>c['max_new_invocations']:raise RuntimeError('Measured allocation exceeds complete 160-call budget')
    budget.require_capacity(root,plan['prospective_remaining_calls'],'measured remaining pilot allocation')
    result={**plan,'eligible_for_bounded_pilot':True,'hardware_recheck':hardware,
        'concurrency_sha256':file_hash(root/'concurrency_check.json'),'power_gate_applied':False}
    atomic_json(root/'measured_plan.json',result);return result


def verify_replay(root,replay):
    root=Path(root);planned={x['context_id'] for x in _read(root/'design.json')['selection']['h2_contexts']}
    rows=replay.get('audits',[])
    if replay.get('status')!='complete' or replay.get('all_exact') is not True or len(rows)!=2 or {r['context_id'] for r in rows}!=planned:
        raise ValueError('Registered exact-forcing replay gate failed; preserve outputs and stop before state recovery/H2 allocation')
    for row in rows:
        if row.get('full_trace_identical') is not True or row.get('complete_state_future_invariance',{}).get('passed') is not True:
            raise ValueError('Complete present-state future-invariance or deterministic replay failed')
        if file_hash(row['replica_path'])!=row['replay_native_sha256']:raise ValueError('Replay gate artifact checksum changed')
    return replay


def verify_state_fit(c,root,cohorts,jobs,statefit):
    """Verify actual 60-job candidate weights, predictions and held-out losses."""
    root=Path(root);expected=set(jobs);rows=statefit.get('jobs',[])
    if len(expected)!=60 or len(rows)!=60 or {r['job'] for r in rows}!=expected:
        raise ValueError('Complete-state learning gate requires every declared 60-setting fit')
    if statefit.get('status')!='complete' or statefit.get('actually_fitted_state_dependent_candidate') is not True or statefit.get('coverage',{}).get('input_coverage_complete') is not True:
        raise ValueError('Complete-state coverage or fitted candidate is unavailable')
    if statefit.get('cohorts')!=cohorts:raise ValueError('Fitted state cohort differs from registered allocation')
    verify_replay(root,_read(root/'replay_controls.json'))
    inputs=[]
    for job in jobs:
        directory=root/'matched/learners'/job/'complete_state';selection=_read(directory/'selection.json')
        if not selection or selection.get('status')!='complete' or not selection.get('actually_fitted_state_dependent'):
            raise ValueError('State-dependent fit missing: '+job)
        if selection.get('selection_split')!='development' or selection.get('evaluation_selection') is not False or selection.get('calibration_selection') is not False:
            raise ValueError('State learner selected with an unauthorized split: '+job)
        for name,digest in selection['files'].items():
            if file_hash(directory/name)!=digest:raise ValueError('State fitting artifact checksum mismatch: '+job+'/'+name)
        for name in ('state.npz','predictions.npz','learning_curve.json','losses.json'):
            if name not in selection['files']:raise ValueError('Missing required complete-state fitting evidence: '+name)
        candidate=selection['predeclared_candidate_id'];selected=selection['selected_id']
        declared=next((r for r in selection['candidates'] if r['id']==candidate),{})
        if declared.get('actually_trained') is not True or declared.get('state_dependent_candidate') is not True or declared.get('alpha')!=c['state_candidate_alpha']:
            raise ValueError('Predeclared state-dependent candidate was not actually trained')
        values=analysis.learning.numeric(directory/'state.npz');pred=analysis.learning.numeric(directory/'predictions.npz')
        if candidate+'__coef' not in values or not np.isfinite(values[candidate+'__coef']).all():raise ValueError('Actual fitted state coefficients absent/nonfinite')
        panel,manifest=analysis.load_panel(root/'matched',job)
        if selection['target_fingerprint']!=manifest['target_fingerprint'] or selection['risk_contract']!=manifest['risk_contract']:
            raise ValueError('Fitted state target/population contract mismatch')
        if json.loads(str(pred['risk_contract_json']))!=manifest['risk_contract']:raise ValueError('Prediction risk contract differs')
        losses=_read(directory/'losses.json')['candidates'];curve=_read(directory/'learning_curve.json')['rows']
        if {r['id'] for r in curve}!={r['id'] for r in selection['candidates']}:raise ValueError('State training/selection record incomplete')
        for split in ('train','development','calibration','evaluation'):
            actual=panel[split]
            if list(map(str,pred[split+'_context_id']))!=cohorts[split] or not np.array_equal(pred[split+'_context_id'],actual['context_id']):
                raise ValueError('Held-out state predictions missing or ID alignment changed')
            for tag,key in ((candidate,split+'_candidate_prediction'),(selected,split+'_prediction')):
                pp=pred[key]
                if pp.shape!=(len(cohorts[split]),actual['Y'].shape[-1]) or not np.isfinite(pp).all():raise ValueError('State prediction shape/value invalid')
                if not np.array_equal(pp,pred[split+'__'+tag]):raise ValueError('Selected/candidate state predictions differ from saved fit')
                observed=analysis.vm.circuit_loss(actual,pp)
                if not np.isclose(observed,losses[tag][split],rtol=1e-12,atol=1e-12):raise ValueError('Saved state loss differs from actual prediction/target')
        devloss={r['id']:r['development_mse'] for r in selection['candidates']}
        if not np.isfinite(list(devloss.values())).all() or devloss[selected]>min(devloss.values())+1e-12:raise ValueError('Development selector did not retain its declared best candidate')
        inputs.extend(analysis.record(root,directory/name) for name in ('selection.json','state.npz','predictions.npz','losses.json','learning_curve.json'))
    result=dict(status='complete',state_dependent_candidate_fitted=True,held_out_evaluated=True,
        jobs=len(jobs),contexts={s:len(ids) for s,ids in cohorts.items()},artifacts=inputs,
        fit_artifact=analysis.record(root,root/'state/fit.json'),selection_split='development',
        mean_fallback_is_not_candidate=True,no_effect_direction_gate=True)
    atomic_json(root/'state_model_gate.json',result);return result


def verify_report(root,c,receipt):
    root=Path(root);results=_read(root/'pilot_results.json')
    checked=reporting.completion_checks(root,results)
    if not checked['experimental_completion_eligible']:raise ValueError('Experimental completion gates failed: '+str(checked['issues']))
    if not receipt or receipt.get('status')!='complete' or receipt.get('experimental_completion_eligible') is not True:
        raise ValueError('PDF/report receipt does not establish completed pilot')
    qa=receipt.get('pdf_verification',{})
    if qa.get('status')!='complete' or qa.get('all_pages_rendered') is not True or qa.get('errors') or not qa.get('pages'):
        raise ValueError('Every-page PDF rendering/layout validation did not pass')
    if file_hash(root/'report/pilot_findings.pdf')!=receipt['pdf_sha256']:raise ValueError('Compiled PDF checksum mismatch')
    for page in qa['pages']:
        image=(root/'report'/page['render']).resolve()
        if not image.is_relative_to((root/'report').resolve()) or file_hash(image)!=page['render_sha256']:raise ValueError('Rendered PDF page missing or altered')
    if receipt['results_hash']!=reporting._canonical(reporting.normalize_results(results)) or receipt['config_hash']!=reporting._canonical(c):
        raise ValueError('PDF report was built from different numerical results or configuration')
    return receipt


def stage(root,name,fn):
    path=Path(root)/'progress'/(name+'.json');started=time.time()
    atomic_json(path,dict(stage=name,status='running',started_unix=started,pid=os.getpid()))
    try:
        result=fn();atomic_json(path,dict(stage=name,status='complete',started_unix=started,finished_unix=time.time(),elapsed_seconds=time.time()-started,result_status=result.get('status') if isinstance(result,dict) else None));return result
    except Exception as exc:
        record=dict(stage=name,status='failed',error=str(exc),type=type(exc).__name__,traceback=traceback.format_exc(),started_unix=started,finished_unix=time.time())
        atomic_json(path,record);atomic_json(Path(root)/'failures'/(name+'-'+str(time.time_ns())+'.json'),record);raise


def preflight(c,root,device):
    from ..v6.models import configure_torch
    deps=reporting.dependencies();hardware=generation.preflight(c,root);errors=list(hardware['blockers'])
    try:
        import neuron,LFPy,h5py
        versions=dict(neuron=neuron.__version__,LFPy=LFPy.__version__,h5py=h5py.__version__)
        torch,_=configure_torch(device,c['dtype'],1);versions['torch']=torch.__version__
    except Exception as exc:versions={};errors.append(str(exc))
    if deps.get('status')!='complete':errors.append('PDF dependency preflight failed: '+str(deps))
    result=dict(status='blocked' if errors else 'complete',blockers=errors,hardware=hardware,pdf_dependencies=deps,
        versions=versions,device=device,threads=1,simulation_workers=8,confirmation_enabled=False)
    atomic_json(Path(root)/'preflight.json',result);return result


def smoke(c,root,device):
    from .state import learnable_control
    result=learnable_control(Path(root)/'engineering/state-learnable')
    if not result['passed']:raise ValueError('Complete-state learnable engineering control failed')
    modules=['tests.test_eegrepr_v6_pilot_state','tests.test_eegrepr_v6_pilot_learning','tests.test_eegrepr_v6_pilot_generation','tests.test_eegrepr_v6_pilot_precision','tests.test_eegrepr_v6_pilot_cli','tests.test_eegrepr_v6_pilot_resume','tests.test_eegrepr_v6_extended_targets']
    # These modules contain no native simulations. Actual-host native concurrency,
    # prefix and replay checks are separately ledger-charged in complete mode.
    path=Path(root)/'engineering/smoke-tests.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as log:proc=subprocess.run([sys.executable,'-m','unittest','-v',*modules],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError('Pilot smoke failed; inspect '+str(path))
    out=dict(status='complete',state_control=result,tests=modules,test_log=str(path),test_log_sha256=file_hash(path),native_invocations=0,scope='Engineering only; no experiment completion')
    atomic_json(Path(root)/'engineering/smoke.json',out);return out


def _cohorts(plan):return {s:[r['context_id'] for r in plan['selection']['state_cohort'] if r['split']==s] for s in ('train','development','calibration','evaluation')}


def run_status_rows(root,plan,ledger):
    root=Path(root);rows=[];calls=ledger['calls_by_stage']
    categories={'preflight':'hardware/dependency diagnostic','engineering':'nonnative engineering diagnostic',
        'registration':'verified historical reuse','derive_existing':'native-derived targets','matched_targets':'matched cohort derivation',
        'concurrency':'new native simulations','measured_plan':'measured resource plan','replay':'new native replay diagnostic',
        'state_recovery':'new cutoff recovery simulations','state_fit':'actual fitted complete-state models',
        'state_model_gate':'verified fitting/held-out artifact gate','h2_generation':'new native H2 futures',
        'ordinary_fit':'actual EEG/code models and task readouts','scoring':'measured same-population errors',
        'h2_replay_scoring':'finite-reference/replay scoring'}
    for path in sorted((root/'progress').glob('*.json')):
        name=path.stem
        if name not in categories:continue
        actual=_read(path);count=None;details='Recorded stage status; no result inferred from a plan.'
        if name=='registration':count=len(plan['historical_rows']);details='Historical native episodes verified by original file/config/manifest hashes; no new simulation.'
        elif name=='derive_existing':count=len(_read(root/'broad/derived/manifest.json',{}).get('jobs',[]));details='Physical native targets derived across acquisitions; historical generation provenance retained.'
        elif name=='matched_targets':count=len(_read(root/'matched/derived/manifest.json',{}).get('jobs',[]));details='32 train / 8 development / 8 separate calibration / 8 evaluation contexts; disjoint circuit groups.'
        elif name=='concurrency':count=sum(v for k,v in calls.items() if k.startswith('concurrency-'));details='Actual attempted native calls, including retries; successful gate requires eight simultaneous independent one-thread processes.'
        elif name=='state_recovery':count=calls.get('cutoff-state-recovery',0);details='Actual attempted cutoff replays; eight compatible captured timing states are reused. Each recovery has an excluded 1-ms engineering tail.'
        elif name=='replay':count=calls.get('exact-forcing-replay',0);details='Actual native exact-forcing replay calls; separately checks complete-state invariance across different unknown future draws.'
        elif name=='h2_generation':count=calls.get('h2-extension',0)+calls.get('h2-fresh-fallback',0);details='Actual attempted additional futures. Completed H2 M32/L32 population is recorded separately; incompatible populations are never blended.'
        elif name=='state_fit':count=len(_read(root/'state/fit.json',{}).get('jobs',[]));details='Target-setting fits; each retains the predeclared trained alpha10 state-dependent candidate and development-selected policy.'
        elif name=='ordinary_fit':
            fitted=_read(root/'fit_ordinary.json',{});count=sum(len(fitted.get(k,{}).get('jobs',[])) for k in ('matched','broad'));details='Method-target fits in distinct matched/broad cohorts; one-seed TCN only on selected matched A1 h0/h10 W128 settings.'
        elif name=='engineering':count=0;details='Learnable complete-state control and numeric/unit tests; zero native calls in workstation smoke.'
        elif name=='measured_plan':details='Projected remaining generation is based on actual same-host eight-process measurements; projected values are not observations.'
        rows.append(dict(stage=name,category=categories[name],status=actual['status'],count=count,details=details,elapsed_seconds=actual.get('elapsed_seconds')))
    rows.extend([dict(stage='implementation_engineering',category='earlier native engineering diagnostics',status='recorded',count=ledger['implementation_engineering_calls'],details='Actual implementation calls, including failed attempts, debit the same 160-call budget; separate from workstation experimental observations.'),
        dict(stage='failed_or_interrupted_native_attempts',category='retained attempt accounting',status='recorded',count=ledger['failed_calls']+ledger['unresolved_calls'],details='Local failed/unresolved attempts remain charged. Counts here overlap the per-stage native attempts above and are not additional calls.'),
        dict(stage='large_confirmation',category='deferred',status='deferred',count=0,details='N192/N257 confirmation, 768-episode numerical panel and automatic power-driven expansion are not launched. A separate decision follows pilot review.')])
    return rows


def scientific_findings(tables):
    """Describe the measured pilot without converting engineering into confirmation."""
    result=[]
    def fmt(value):return 'unavailable' if value is None else format(float(value),'.6g')
    for job in ('A1__T0_l0_d128','A1__T0_l10_d128','A0__T4_energy_alpha_l0_d1000'):
        for r in tables['h1_information']:
            if r['job']!=job or r['method'] not in ('complete_state_candidate','eeg_full') or not r['population'].startswith('Matched'):continue
            result.append(dict(question='H1/H4 '+job,method=r['method'],observed=r['method']+': MSE '+fmt(r['error_baseline100'])+' (training mean=100)',
                interpretation='Achieved fitted pilot performance; full-state fitted error is not the state floor.',
                limitation='Two inspected evaluation circuits; ideal EEG and separate information losses unidentified.',
                later_study='Separate later confirmation decision; retain negative skill.',status='measured'))
    for r in tables.get('h2_precision',[]):
        if r['job']!='A0__T0_l0_d128' or r['M']!=32:continue
        result.append(dict(question='H2 finite futures: '+r['context_id'],
            observed='M32/L32 corrected mean error '+fmt(r['mean_error_corrected'])+' '+r['mean_error_units']+'; corrected squared variance error '+fmt(r['variance_error'])+' '+r['variance_error_units'],
            interpretation='Conditional sampling precision at one registered fixed state; both scoring and reference pools remain finite.',
            limitation='Two registered contexts if compatible, otherwise one fresh context. Negative corrections retained; signed variance-difference intervals are not squared-error intervals.',
            later_study='Assess precision using independent larger reference pools; do not count futures as new circuits.',status='measured'))
    for r in tables.get('h3_retention',[]):
        if not r['population'].startswith('Matched') or r['method'] not in ('feature18_B32','pca_B32','forecast_B32'):continue
        result.append(dict(question='H3 task retention: '+r['method'],
            observed='d8/B32 past NMSE '+fmt(r['past_reconstruction_nmse'])+'; future MSE '+fmt(r['future_error_baseline100'])+' (mean=100); carrier CE '+fmt(r['carrier_cross_entropy_bits'])+' bits; rhythm balanced accuracy '+fmt(r['rhythm_balanced_accuracy']),
            interpretation='Same payload evaluated on past reconstruction, native future prediction and carrier/rhythm tasks; these are distinct readout-dependent objectives.',
            limitation='Two inspected evaluation circuits. Carrier is conditional on rhythmic membership; no total information fraction or objective-superiority inference.',
            later_study='Choose one task/code contrast before a separate disjoint-circuit study; inspect clipping and decoder capacity.',status=r['status']))
    return result


def _assemble(c,root,tables,precision):
    root=Path(root);plan=_read(root/'design.json');statefit=_read(root/'state/fit.json');replay=_read(root/'replay_controls.json');ledger=budget.summary(root)
    tables['h1_information']+=precision.get('replay_information',[]);tables['h2_precision']=precision['h2_precision']
    full=[r for r in tables['h4_targets'] if r['population'].startswith('Matched')]
    tables['full_grid']=full
    tables['run_status']=run_status_rows(root,plan,ledger)
    tables['findings']=scientific_findings(tables)
    artifact=[]
    for job in cfg.jobs():
        dest=root/'matched/learners'/job/'complete_state'
        for filename,role in [('state.npz','fitted_state'),('predictions.npz','predictions'),('selection.json','selection'),('losses.json','losses'),('learning_curve.json','training_curve')]:artifact.append(analysis.record(root,dest/filename,role))
    artifact.append(analysis.record(root,root/'state/adapter/adapter.json','schema'))
    verified=dict(status='complete',input_coverage_complete=statefit['coverage']['input_coverage_complete'],
        future_invariance_passed=all(r['complete_state_future_invariance']['passed'] for r in replay['audits']),
        fitted_state_dependent_candidate=statefit['actually_fitted_state_dependent_candidate'],dev_selection_complete=True,
        held_out_predictions_complete=True,target_population_matching_passed=True,artifact_manifest=artifact)
    required_names=('preflight','engineering','registration','derive_existing','matched_targets','concurrency',
        'measured_plan','replay','state_recovery','state_fit','state_model_gate','h2_generation','ordinary_fit','scoring','h2_replay_scoring')
    required={name:_read(root/'progress'/(name+'.json'),{}).get('status','unrun') for name in required_names}
    results=dict(schema_version='eegrepr-v6-reporting-pilot-results-1',purpose=c['purpose'],status='ready_for_report',
        completed_allocation=dict(historical_episodes_reused=len(plan['historical_rows']),state_cohort=plan['selection']['state_counts'],
            simulation_ledger=ledger,h2=_read(root/'h2/generation.json'),ordinary_evaluation_circuits=8,matched_evaluation_circuits=2),
        runtime=dict(simulation=ledger,concurrency=_read(root/'concurrency_check.json'),measured_plan=_read(root/'measured_plan.json'),
            fitting=_read(root/'fit_ordinary.json'),state_fit_seconds=statefit['elapsed_seconds'],device=_read(root/'pilot_provenance.json')['identity']['device'],
            worker_count=8,cpu_threads_per_worker=1),coverage=statefit['coverage'],complete_state_verification=verified,
        required_stages=required,tables=tables,figures=precision.get('figures',{}),
        limitations=['Exploratory already-inspected circuits, not confirmation.','No power or effect-direction gate.','Full-state fitted error is not the state floor.','Unknown EEG ideal remains unknown.','L32 reference is finite and uncertain.'])
    atomic_json(root/'pilot_results.json',results);return results


def complete(c,root,args):
    root=Path(root);receipt=root/'pilot_completed.json'
    if receipt.exists():
        old=_read(receipt)
        if file_hash(old['export']['path'])!=old['export']['sha256']:raise ValueError('Completed pilot export changed')
        if file_hash(root/'pilot_results.json')!=old.get('pilot_results_sha256'):raise ValueError('Completed pilot numerical results changed')
        if canonical_hash(c)!=old.get('config_hash'):raise ValueError('Completed pilot configuration changed')
        current=verify_report(root,c,_read(root/'report/receipt.json'))
        revision=_read(root/'report_revision.json',old)
        if file_hash(root/'report/receipt.json')!=revision['report_receipt_sha256'] or current['pdf_sha256']!=revision['pdf_sha256']:
            raise ValueError('Completed PDF or report receipt changed outside a verified report-only rebuild')
        return {**old,'current_report':current}
    checked=stage(root,'preflight',lambda:preflight(c,root,args.device))
    if checked['status']!='complete':raise RuntimeError('Hardware/dependency blocker: '+str(checked['blockers']))
    if not args.source_archive or not args.native_source:raise ValueError('Complete pilot requires original archive and native source')
    if file_hash(args.source_archive)!=c['source_archive_sha256']:raise ValueError('Historical archive hash differs from frozen source')
    with EvidenceSource(args.source_archive) as source:import_evidence(source,root)
    stage(root,'engineering',lambda:smoke(c,root,args.device))
    plan=stage(root,'registration',lambda:generation.prepare(c,root,args.native_source));cohorts=_cohorts(plan)
    broad=root/'broad';broad.mkdir(exist_ok=True)
    deriv=stage(root,'derive_existing',lambda:derive(cfg.analysis_config(),broad,source=args.source_archive,native_source=args.native_source,resume=args.resume))
    if deriv['completed_jobs']!=60:raise RuntimeError('Pilot requires native support for all60 targets before simulations')
    stage(root,'matched_targets',lambda:panels.derive_matched(broad,root/'matched',cohorts,cfg.jobs()))
    stage(root,'concurrency',lambda:generation.concurrency_check(c,root,resume=args.resume))
    stage(root,'measured_plan',lambda:verify_measured_plan(c,root,generation.measured_plan(c,root)))
    stage(root,'replay',lambda:verify_replay(root,generation.replay_controls(c,root,resume=args.resume)))
    stage(root,'state_recovery',lambda:generation.recover_states(c,root,resume=args.resume))
    statefit=stage(root,'state_fit',lambda:analysis.fit_complete_state(c,root,cohorts,cfg.jobs(),args.resume))
    gate=stage(root,'state_model_gate',lambda:verify_state_fit(c,root,cohorts,cfg.jobs(),statefit))
    stage(root,'h2_generation',lambda:generation.extend_h2(c,root,resume=args.resume,state_model_gate=gate))
    stage(root,'ordinary_fit',lambda:analysis.fit_ordinary(c,root,args.source_archive,cfg.jobs(),args.resume,args.device))
    tables=stage(root,'scoring',lambda:analysis.score_models(c,root,cfg.jobs()))
    from .precision import run as precision
    extra=stage(root,'h2_replay_scoring',lambda:precision(c,root))
    results=_assemble(c,root,tables,extra)
    report=stage(root,'report',lambda:reporting.report(root,c,results))
    verify_report(root,c,report)
    dest=Path(args.destination) if args.destination else root.parent/(root.name+'-evidence.zip')
    exported=stage(root,'export',lambda:exporting.export(root,dest))
    result=dict(status='complete',purpose=c['purpose'],completed_unix=time.time(),simulation_ledger=budget.summary(root),
        report_receipt_sha256=file_hash(root/'report/receipt.json'),pdf_sha256=file_hash(root/'report/pilot_findings.pdf'),
        pilot_results_sha256=file_hash(root/'pilot_results.json'),config_hash=canonical_hash(c),
        pdf_all_pages_rendered=True,pdf_checks=report['pdf_verification']['automated_checks'],
        export=exported,confirmation='Not launched; separate user decision after pilot review')
    atomic_json(receipt,result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','doctor','plan','dry-run','smoke','complete','complete-pilot','resume','status','report-only'])
    p.add_argument('--config',default=str(ROOT/'configs/eeg_representation/v6_reporting_pilot/pilot.yaml'));p.add_argument('--root',required=True)
    p.add_argument('--source-archive');p.add_argument('--native-source');p.add_argument('--destination');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args(argv);root=Path(a.root).expanduser().resolve()
    if a.mode=='status':
        print(json.dumps(dict(ledger=budget.summary(root),completed=_read(root/'pilot_completed.json'),progress={p.stem:_read(p) for p in (root/'progress').glob('*.json')},process_liveness='Not inferred from saved artifacts'),indent=2));return 0
    if a.mode=='report-only':
        c=_read(root/'pilot_config.json')
        if c is None:raise ValueError('Initialized pilot root required')
        with output_lock(root):
            rebuilt=reporting.report(root,c)
            if (root/'pilot_completed.json').exists():
                verify_report(root,c,rebuilt)
                atomic_json(root/'report_revision.json',dict(report_receipt_sha256=file_hash(root/'report/receipt.json'),
                    pdf_sha256=file_hash(root/'report/pilot_findings.pdf'),rebuilt_unix=time.time(),
                    experiment_rerun=False,export_rebuilt=False,original_completion_sha256=file_hash(root/'pilot_completed.json')))
        print(json.dumps(rebuilt,indent=2));return 0
    c=cfg.load(a.config);a.resume=a.resume or a.mode=='resume'
    if a.mode in ('complete','complete-pilot','resume') and not a.execute:p.error('Expensive pilot execution requires explicit --execute')
    with output_lock(root):
        initialize(c,root,a.device,a.resume)
        print('PILOT config_hash='+canonical_hash(c)+' root='+str(root)+' workers=8 purpose='+c['purpose'],flush=True)
        if a.mode in ('preflight','doctor'):result=preflight(c,root,a.device)
        elif a.mode in ('plan','dry-run'):
            debit=budget.summary(root)['implementation_engineering_calls']
            result=dict(status='planned',purpose=c['purpose'],jobs=cfg.jobs(),workers=8,new_native_calls_default=106,
                implementation_engineering_calls=debit,total_invocations_default=debit+106,total_invocations_fresh_fallback=debit+122,new_native_hard_cap=160,
                allocation=dict(concurrency=8,remaining_state_recovery=48,h2_extension=48,exact_forcing_replay=2),
                reuse=dict(historical_episodes=832,matched_state_contexts=56),provisional_worker_hours=106*88.71645*(7.010/6.138)/3600,
                timing='Provisional same-host historical full-episode scale only. Actual eight-process concurrency and stage-specific prefix runtime required before allocation; fitting/GPU measured separately',
                simulation_ledger=budget.summary(root),
                prerequisites=preflight(c,root,a.device),confirmation=False,original_numerical_768_panel=False,power_expansion=False)
            atomic_json(root/'plan.json',result)
        elif a.mode=='smoke':result=smoke(c,root,a.device)
        else:result=complete(c,root,a)
    print(json.dumps(result,indent=2,default=str));return 2 if result.get('status')=='blocked' else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print('REPORTING PILOT ERROR: '+str(exc),file=sys.stderr);raise SystemExit(2)
