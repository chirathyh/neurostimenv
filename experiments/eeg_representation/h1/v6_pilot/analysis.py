"""Same-population pilot fitting and numerical H1--H4 accounting."""
from pathlib import Path
import json,time
import numpy as np
from ..artifacts import atomic_json,atomic_npz,file_hash
from ..targets import canonical_hash
from ..v6 import models as vm
from ..v6.derivation import load_panel
from ..v6.information import summarize,paired_contrast
from ..v4.scoring import assert_risk_contract
from ..v6.targets import timing_metadata,parse_target_id
from ..v6.retention import evaluate_readout
from . import learning,banks

def _read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def record(root,path,role=None):
    p=Path(path);r=dict(path=str(p.relative_to(root)),sha256=file_hash(p),bytes=p.stat().st_size)
    if role:r['role']=role
    return r

def state_features(config,root,cohorts):
    """Fit once, then verify and load an immutable complete-state input matrix.

    Field registration can use held-out field paths/shapes; numerical centering
    and scales use training values only. A changed state export, cohort, schema,
    implementation or saved matrix fails before any artifact is overwritten.
    """
    from . import state
    import inspect
    root=Path(root);directory=root/'state';manifest_path=directory/'manifest.json'
    manifest=_read(manifest_path)
    if manifest is None:raise ValueError('Complete-state recovery has not completed')
    entries=manifest.get('contexts',manifest.get('records',manifest.get('states',[])))
    if isinstance(entries,dict):entries=[dict(context_id=k,**v) for k,v in entries.items()]
    if len({r['context_id'] for r in entries})!=len(entries):raise ValueError('Duplicate complete-state context export')
    byid={r['context_id']:r for r in entries};payloads={};sources=[]
    for split,ids in cohorts.items():
        payloads[split]=[]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate complete-state cohort context')
        for cid in ids:
            if cid not in byid:raise ValueError('Complete-state export missing '+cid)
            r=byid[cid];path=Path(r.get('state_directory',r.get('state_path',r.get('path',root/'state'/cid))))
            if not path.is_absolute():path=root/path
            payload=state.load_state(path);state.validate(payload)
            if payload['binding'].get('context_id')!=cid:raise ValueError('Complete-state context binding mismatch')
            payloads[split].append(payload)
            sources.append(dict(split=split,context_id=cid,input_hash=payload['input_hash'],
                schema_hash=payload['schema_hash'],schema_sha256=file_hash(path/'schema.json'),
                fields_sha256=file_hash(path/'fields.npz'),binding=payload['binding']))
    details=dict(schema='eegrepr-pilot-state-features-freeze-1',cohorts=cohorts,sources=sources,
        recovery_population_signature=canonical_hash({k:manifest.get(k) for k in
            ('design_hash','context_counts','circuit_ids','target_binding','input_policy','future_invariance')}),
        state_implementation_sha256=file_hash(Path(state.__file__)),
        feature_implementation=canonical_hash(inspect.getsource(state_features)),
        adapter_version=state.ADAPTER_VERSION,arithmetic='float64',
        scaling='training values only; mechanism-aware lower scale bounds; no coordinate deletion',
        registry='declared field paths and shapes from the registered cohort; no held-out numerical statistics')
    contract=dict(signature=canonical_hash(details),details=details)
    guard=directory/'adapter_input_contract.json';marker=directory/'adapter_freeze.json'
    outputs=['adapter/adapter.json','adapter/scaling.npz','adapter_inputs.npz','coverage_audit.json']
    if guard.exists():
        if _read(guard)!=contract:raise ValueError('Immutable complete-state input/cohort/schema contract changed')
    elif marker.exists() or any((directory/p).exists() for p in outputs):
        raise ValueError('Unversioned complete-state adapter artifacts cannot be overwritten')
    else:atomic_json(guard,contract)
    if marker.exists():
        frozen=_read(marker)
        if frozen.get('signature')!=contract['signature']:raise ValueError('Complete-state adapter freeze changed')
        checksum=frozen.pop('manifest_content_sha256',None)
        if checksum!=canonical_hash(frozen):raise ValueError('Complete-state adapter manifest content changed')
        for name,digest in frozen['files'].items():
            if not (directory/name).is_file() or file_hash(directory/name)!=digest:
                raise ValueError('Complete-state adapter artifact checksum mismatch: '+name)
        adapter=state.load_adapter(directory/'adapter')
        with np.load(directory/'adapter_inputs.npz',allow_pickle=False) as f:
            values={split:f[split].copy() for split in cohorts}
            for split,ids in cohorts.items():
                if not np.array_equal(f[split+'_context_id'],np.asarray(ids)):
                    raise ValueError('Frozen complete-state input context ordering changed')
        if adapter['fingerprint']!=frozen['adapter_fingerprint']:raise ValueError('Complete-state adapter/scaling identity changed')
        return values,_read(directory/'coverage_audit.json')
    # If a prior run stopped between atomic outputs, verify any surviving adapter
    # against the deterministic fit before continuing; never silently overwrite.
    allp=[p for x in payloads.values() for p in x]
    expected=state.fit_adapter(payloads['train'],all_schema=allp)
    adapter_dir=directory/'adapter'
    if (adapter_dir/'adapter.json').exists():
        adapter=state.load_adapter(adapter_dir)
        if adapter['fingerprint']!=expected['fingerprint']:raise ValueError('Partial complete-state adapter differs from the registered fit')
    elif adapter_dir.exists() and any(adapter_dir.iterdir()):
        raise ValueError('Unfrozen partial complete-state adapter cannot be overwritten')
    else:state.save_adapter(adapter_dir,expected);adapter=expected
    values={split:state.transform_adapter(adapter,p) for split,p in payloads.items()}
    arrays={**values,**{split+'_context_id':np.asarray(ids) for split,ids in cohorts.items()}}
    bank_path=directory/'adapter_inputs.npz'
    if bank_path.exists():
        with np.load(bank_path,allow_pickle=False) as f:
            if set(f.files)!=set(arrays) or any(f[k].dtype!=np.asarray(v).dtype or not np.array_equal(f[k],v) for k,v in arrays.items()):
                raise ValueError('Partial complete-state input matrix differs; refusing overwrite')
    else:atomic_npz(bank_path,**arrays)
    audit=dict(status='complete',input_coverage_complete=True,contexts=sum(map(len,payloads.values())),
        fields=[state.validate(p) for p in allp],adapter_scope='registered structural coordinate union; only training values fit scaling',
        future_invariance_source='generation replay_controls.json; required before complete-state learning gate',
        input_coordinates=values['train'].shape[1],no_coordinate_truncation=True,
        input_signature=contract['signature'],adapter_fingerprint=adapter['fingerprint'])
    audit_path=directory/'coverage_audit.json'
    if audit_path.exists():
        if _read(audit_path)!=audit:raise ValueError('Partial complete-state coverage metadata changed; refusing overwrite')
    else:atomic_json(audit_path,audit)
    frozen=dict(status='complete',signature=contract['signature'],adapter_fingerprint=adapter['fingerprint'],
        files={name:file_hash(directory/name) for name in outputs+['adapter_input_contract.json']})
    frozen['manifest_content_sha256']=canonical_hash(frozen);atomic_json(marker,frozen)
    return values,audit

def fit_complete_state(config,root,cohorts,jobs,resume=False):
    root=Path(root);values,audit=state_features(config,root,cohorts);rows=[];start=time.perf_counter();design=None
    for job in jobs:
        panel,manifest=load_panel(root/'matched',job)
        for s,p in panel.items():
            if list(map(str,p['context_id']))!=cohorts[s]:raise ValueError('State/target ordering differs')
        if design is None:design=learning.fit_design(values['train'],vm.context_weights(panel['train']),already_scaled=True)
        dest=root/'matched/learners'/job/'complete_state'
        fitted=learning.ridge(panel,values,manifest,dest,config,resume=resume,mechanism_scaled=True,design=design)
        rows.append(dict(job=job,**fitted,selection_artifact=record(root,dest/'selection.json')))
    result=dict(status='complete',jobs=rows,coverage=audit,elapsed_seconds=time.perf_counter()-start,
        actually_fitted_state_dependent_candidate=True,selected_mean_may_be_fallback=True,
        full_state_not_state_floor=True,cohorts=cohorts)
    atomic_json(root/'state/fit.json',result);return result

def fit_ordinary(config,root,source_archive,jobs,resume=False,device='cpu'):
    root=Path(root);result={};start=time.perf_counter()
    for cohort in ('matched','broad'):
        stage=root/cohort;panel,manifest=load_panel(stage,'A0__T0_l10_d128')
        codes=banks.build(config,stage,panel,source_archive,device,resume);rows=[];designs={}
        for job in jobs:
            panel,manifest=load_panel(stage,job)
            for history in config['histories']:
                values={s:p['X'][:,-history:] for s,p in panel.items()};method='eeg_short' if history==64 else 'eeg_full'
                key=(job.split('__')[0],history)
                if key not in designs:designs[key]=learning.fit_design(values['train'],vm.context_weights(panel['train']))
                r=learning.ridge(panel,values,manifest,stage/'learners'/job/method,config,resume=resume,design=designs[key])
                rows.append(dict(job=job,method=method,history_samples=history,selection=r['selected_id']))
            if job.startswith('A0__'):
                for name,bank in codes.items():
                    if name not in designs:designs[name]=learning.fit_design(bank['values']['train'],vm.context_weights(panel['train']))
                    r=learning.ridge(panel,bank['values'],manifest,stage/'learners'/job/name,config,resume=resume,design=designs[name])
                    rows.append(dict(job=job,method=name,selection=r['selected_id'],metadata=bank['metadata']))
            if cohort=='matched' and job in config['neural_jobs']:
                for h in config['histories']:
                    r=learning.tcn(panel,manifest,stage/'learners'/job/('tcn_'+str(h)),config,h,device,resume)
                    rows.append(dict(job=job,method='tcn_'+str(h),selection=r['selected_id'],updates=r['optimizer_updates']))
        # Matched H3 tasks use the identical projected/coded features as future readouts.
        anchor,_=load_panel(stage,'A0__T0_l10_d128');h3=[]
        for name,bank in codes.items():
            for task in ('waveform_reconstruction','carrier','rhythm'):
                p={s:v for s,v in anchor.items() if s!='calibration'}
                r=evaluate_readout(p,{s:v for s,v in bank['values'].items() if s!='calibration'},task,'all',config['ridge_alphas'],stage/'retention'/name/task,resume)
                h3.append({**r,'method':name,'task':task,'metadata':bank['metadata']})
        atomic_json(stage/'retention/summary.json',dict(status='complete',rows=h3))
        result[cohort]=dict(status='complete',jobs=rows,neural_policy='one seed; only matched A1 h0/h10 W128 × two histories')
        atomic_json(stage/'learners/manifest.json',result[cohort])
    result['elapsed_seconds']=time.perf_counter()-start;atomic_json(root/'fit_ordinary.json',result);return result

def _prediction_row(config,panel,manifest,job,method,pred,baseline,cohort,source):
    p=panel['evaluation'];m=int(config['score_futures'])
    if np.any(p['future_mask'].sum(1)<m):raise ValueError('Incomplete pilot scoring futures')
    q={**p,'future_id':p['future_id'][:,:m]};y=p['Y_physical'][:,:m]
    n=manifest['normalization'];mu=pred*n['y_scale']+np.asarray(n['y_offset']);base=baseline*n['y_scale']+np.asarray(n['y_offset'])
    out,components=summarize(y,mu,base,q,method_id=method,method=method,information_available=method,
        source=source,confidence=config['descriptive_confidence'])
    family,h,w=parse_target_id(job.split('__')[1]);interval=out['error_baseline100_interval']
    # Two circuits cannot usefully establish pilot population precision. Retain the
    # raw circuit components and conditional MC, but defer the displayed interval.
    if out['n_circuits']<4:interval=dict(status='deferred',reason='Only two evaluation circuits; per-circuit effects and conditional MC retained, population precision deferred')
    row=dict(job=job,acquisition=job.split('__')[0],target=job.split('__')[1],h_ms=h,duration_ms=w,
        method=method,information=method,evidence_type='fitted',risk_physical=out['physical_mse'],
        reference_risk_physical=out['reference_mse'],error_baseline100=out['error_baseline100'],
        relative_mse_reduction=out['relative_mse_reduction_pp'],uncertainty=interval,
        circuits=out['n_circuits'],contexts=out['n_contexts'],futures=m,population=cohort,status='measured',
        physical_units='V^2' if family.startswith('T4_energy') else 'dimensionless log(V^2 / 1V^2)' if family.startswith('T4_log') else 'V',
        risk_units='V^4' if family.startswith('T4_energy') else 'dimensionless squared log units' if family.startswith('T4_log') else 'V^2',
        state_variance_physical=out['state_variance'],combined_unresolved_excess=out['combined_unresolved_excess'],
        circuit_components=out['circuit_components'],conditional_mc=out['conditional_mc'],source=source,
        full_numeric=out,actually_trained=method=='complete_state_candidate',
        interpretation='Fitted errors do not identify individual observation/compression/fitting losses')
    return row,components

def comparison(first,second,kind,config):
    """Paired effects preserve actual IDs, target denominators and MC streams."""
    if first['population']!=second['population']:raise ValueError('Paired population differs')
    if kind=='acquisition' and first['target']!=second['target']:
        raise ValueError('Acquisition contrast requires the same physical target')
    if kind in ('acquisition','history') and first['reference_risk_physical']!=second['reference_risk_physical']:
        raise ValueError('Fixed-target contrast uses different training-mean references')
    paired=paired_contrast(first['full_numeric'],second['full_numeric'],label=kind,confidence=config['descriptive_confidence'])
    interval=paired['gain_difference']
    if first['circuits']<4:
        interval=dict(status='deferred',reason='Two evaluation circuits; paired components and conditional MC retained, population precision deferred')
    return dict(job=first['job'],comparison_type=kind,first_job=first['job'],second_job=second['job'],
        first_method=first['method'],second_method=second['method'],population=first['population'],
        acquisition=first['acquisition'],h_ms=first['h_ms'],duration_ms=first['duration_ms'],
        error_baseline100=first['error_baseline100'],relative_mse_reduction=first['relative_mse_reduction'],
        paired_difference=paired['gain_difference']['estimate'],difference_definition='relative MSE reduction(first) minus relative MSE reduction(second), percentage points',
        uncertainty=interval,paired_descriptive=paired,circuits=first['circuits'],
        target_denominators='same target-specific mean reference' if kind in ('history','acquisition') else 'different target-specific mean references; different prediction problems',
        status='measured')


def score_models(config,root,jobs):
    root=Path(root);allrows=[];h3rows=[];contrasts=[];h4=[]
    for cohort in ('matched','broad'):
        stage=root/cohort;population=('Matched state cohort: two evaluation circuits / eight contexts' if cohort=='matched' else 'Historical inspected ordinary cohort: eight evaluation circuits / 32 contexts')
        for job in jobs:
            panel,manifest=load_panel(stage,job);baseline=vm.context_weights(panel['train'])@vm.first_targets(panel['train'])
            methods={};saved=[];components={}
            for path in sorted((stage/'learners'/job).glob('*/predictions.npz')):
                name=path.parent.name;pred=learning.numeric(path);ids=pred['evaluation_context_id']
                if not np.array_equal(ids,panel['evaluation']['context_id']):raise ValueError('Prediction population changed')
                selection=_read(path.parent/'selection.json')
                if not selection or selection.get('selection_split')!='development':raise ValueError('Missing development selection record')
                if selection['files']['predictions.npz']!=file_hash(path):raise ValueError('Prediction checksum changed')
                if 'risk_contract_json' not in pred:raise ValueError('Prediction risk contract missing')
                assert_risk_contract(manifest['risk_contract'],json.loads(str(pred['risk_contract_json'])))
                if name=='complete_state':
                    methods['complete_state_selected']=(pred['evaluation_prediction'],path)
                    methods['complete_state_candidate']=(pred['evaluation_candidate_prediction'],path)
                else:methods[name]=(pred['evaluation_prediction'],path)
            methods['train_mean']=(np.broadcast_to(baseline,(len(panel['evaluation']['context_id']),len(baseline))),None)
            for method,(pred,path) in methods.items():
                source=[] if path is None else [record(root,path)]
                row,comp=_prediction_row(config,panel,manifest,job,method,pred,baseline,population,source)
                if path is not None:
                    selected=_read(path.parent/'selection.json')
                    row['training_budget']={k:selected.get(k) for k in ('training_contexts','training_circuits','optimizer_budget','fitting_repeats','input_coordinates','selected_id','predeclared_candidate_id')}
                    row['selection_source']=record(root,path.parent/'selection.json')
                saved.append(row);components[method]=comp
            mean=next(r for r in saved if r['method']=='train_mean')
            common={k:mean[k] for k in ('job','acquisition','target','h_ms','duration_ms','circuits','contexts','futures','population','physical_units','risk_units','reference_risk_physical')}
            saved.extend([dict(common,method='state_variance',information='complete present state, future inputs unknown',evidence_type='estimated reference',status='estimated reference',risk_physical=mean['state_variance_physical'],error_baseline100=100*mean['state_variance_physical']/mean['reference_risk_physical'],uncertainty=mean['uncertainty'] if mean['circuits']<4 else mean['full_numeric']['state_variance_baseline100_interval']),
                dict(common,method='theoretical_future_input',information='complete state plus exact realised future inputs',evidence_type='theoretical',status='theoretical',risk_physical=0.,error_baseline100=0.,uncertainty=None),
                dict(common,method='ideal_eeg',information='specified EEG history',evidence_type='unknown',status='unavailable',risk_physical=None,error_baseline100=None,uncertainty='Ideal EEG risk is not identified')])
            atomic_json(stage/'analysis'/job/'rows.json',dict(status='complete',rows=saved,target_timing=timing_metadata(job.split('__')[1])))
            atomic_npz(stage/'analysis'/job/'loss_components.npz',**{m+'__'+k:v for m,c in components.items() for k,v in c.items() if v is not None})
            allrows.extend(saved)
            r=next(r for r in saved if r['method']=='eeg_full');h4.append({**r,'same_state_variance_ratio':r['state_variance_physical']/r['reference_risk_physical']})
            short=next(r for r in saved if r['method']=='eeg_short')
            contrast=comparison(r,short,'history',config)
            contrast.pop('error_baseline100')
            contrasts.append(dict(contrast,short_error=short['error_baseline100'],full_error=r['error_baseline100']))
        lookup={(r['job'],r['method']):r for r in allrows if r['population']==population}
        for job in jobs:
            acquisition,target=job.split('__');family,h,w=parse_target_id(target)
            current=lookup[(job,'eeg_full')]
            if acquisition!='A0' and ('A0__'+target,'eeg_full') in lookup:
                contrasts.append(comparison(current,lookup[('A0__'+target,'eeg_full')],'acquisition',config))
            if family=='T0' and h in (2,10):
                anchor=f'{acquisition}__T0_l0_d{int(w)}'
                if (anchor,'eeg_full') in lookup:contrasts.append(comparison(current,lookup[(anchor,'eeg_full')],'target offset',config))
            if family=='T0' and w==128:
                anchor=f'{acquisition}__T0_l{int(h)}_d32'
                if (anchor,'eeg_full') in lookup:contrasts.append(comparison(current,lookup[(anchor,'eeg_full')],'target duration',config))
        retention=_read(stage/'retention/summary.json',{'rows':[]})
        for method in sorted({r['method'] for r in retention['rows']}):
            rows=[r for r in retention['rows'] if r['method']==method];task={r['task']:r for r in rows};anchor=next((r for r in allrows if r['population']==population and r['job']=='A0__T0_l10_d128' and r['method']==method),None)
            rec=task.get('waveform_reconstruction',{}).get('evaluation',{});carrier=task.get('carrier',{}).get('evaluation',{});rhythm=task.get('rhythm',{}).get('evaluation',{})
            h3rows.append(dict(method=method,representation=method.split('_')[0],variant='B32' if method.endswith('B32') else 'projected unquantized',
                projection='d8',budget_bits=32 if method.endswith('B32') else None,
                population=population,cohort_budget=rows[0].get('metadata'),past_reconstruction_nmse=rec.get('normalized_by_train_mean_risk'),
                future_error_baseline100=None if anchor is None else anchor['error_baseline100'],future_target='A0 h10 W128',
                carrier_cross_entropy_bits=carrier.get('cross_entropy_bits'),accuracy=carrier.get('accuracy'),rhythm_balanced_accuracy=rhythm.get('balanced_accuracy'),
                descriptive_uncertainty='circuit/readout detailed records; no pilot primary decision',status='measured' if all(r['status']=='complete' for r in rows) else 'partial',
                source=str(stage/'retention'/method)))
    atomic_json(root/'analysis_summary.json',dict(status='complete',information=allrows,comparisons=contrasts,h3=h3rows,h4=h4))
    return dict(h1_information=allrows,h1_comparisons=contrasts,h3_retention=h3rows,h4_targets=h4)
