"""Privileged exact-forcing full-prefix replay on the eight registered gate trials.

This is a software diagnostic on its own population. It neither restores a
checkpoint nor trains a predictor from saved EEG or an incomplete state vector.
"""
from pathlib import Path
import time
import numpy as np
from scipy import signal
from .artifacts import atomic_json,atomic_npz,read_json,artifact_record,verify_record,canonical_hash,module_hash,file_hash
from .config import contexts,target_ids
from .generation import adapter,generation_contract,verify_row,trial_spec
from .simulator import generate_trial,coupling_fingerprint
from .targets import extract_physical_target,target_metadata,target_spec
from ...forks import _read_fork,validate_fork_prefixes
from ...simulator import array_hash
from ...storage import write_trial
from ..v4.measurement import _design,causal_history,coefficient_hash


def acquisition_state(trial,acquisition,cutoff_ms):
    t=np.asarray(trial['time_ms']);x=np.asarray(trial['eeg_v'],np.float64)
    if x.ndim==2 and x.shape[1]==1:x=x[:,0]
    if x.ndim!=1 or t.ndim!=1 or len(x)!=len(t):raise ValueError('Replay acquisition requires one native EEG channel aligned to timestamps')
    x=x[t<=cutoff_ms]
    if not len(x) or t[t<=cutoff_ms][-1]!=cutoff_ms:raise ValueError('Exact cutoff absent')
    kind,coef=_design(acquisition)
    if kind=='sos':_,state=signal.sosfilt(coef,x,zi=np.zeros((len(coef),2)))
    else:_,state=signal.lfilter(coef,[1.],x,zi=np.zeros(len(coef)-1))
    return dict(operator=acquisition,coefficient_hash=array_hash(coef),state_hash=array_hash(state),
        state_shape=list(state.shape),prefix_hash=array_hash(x),startup='zero state at original native prefix start; preserved through cutoff')


def compare_trials(original,replica,*,cutoff_ms,targets=(),history_s=4.):
    """Audit actual complete replay inputs/state and measure rather than insert zero."""
    if original['metadata'].get('backend')!='neuron_lfpy_real' or replica['metadata'].get('backend')!='neuron_lfpy_real':
        raise ValueError('Measured deterministic NEURON replay cannot use analytical fallback')
    fork=validate_fork_prefixes([original,replica],cutoff_ms)
    forcing=coupling_fingerprint(original)
    if forcing!=coupling_fingerprint(replica):raise ValueError('Exact realised external schedule/phase/parameters differ')
    required=('cutoff_physical_state','pending_recurrent_events','pending_external_events')
    states={k:array_hash(original['privileged'][k]) for k in required}
    acquisition={}
    for name in ('A0','A1','A2'):
        a=acquisition_state(original,name,cutoff_ms);b=acquisition_state(replica,name,cutoff_ms)
        if a!=b:raise ValueError('Acquisition state differs under replay')
        if cutoff_ms>=history_s*1000:
            x,_,_=causal_history(original['eeg_v'],original['time_ms'],cutoff_ms=cutoff_ms,history_s=history_s,acquisition_id=name)
            z,_,_=causal_history(replica['eeg_v'],replica['time_ms'],cutoff_ms=cutoff_ms,history_s=history_s,acquisition_id=name)
            if not np.array_equal(x,z):raise ValueError('Acquired prefix differs under replay')
            a['history_hash']=array_hash(x)
        acquisition[name]=a
    rows=[]
    for target in targets:
        a=extract_physical_target(original['eeg_v'],original['time_ms'],target,cutoff_ms=cutoff_ms)
        b=extract_physical_target(replica['eeg_v'],replica['time_ms'],target,cutoff_ms=cutoff_ms)
        rows.append(dict(target_id=target,physical_mse=float(np.mean((a-b)**2)),max_absolute_error=float(np.max(np.abs(a-b))),
            target_hash=array_hash(a),replay_target_hash=array_hash(b),target_spec=target_metadata(target,cutoff_ms=cutoff_ms)))
    full=bool(np.array_equal(original['eeg_v'],replica['eeg_v']) and np.array_equal(original['dipole_na_um'],replica['dipole_na_um']))
    return dict(status='complete',passed=full,full_trace_identical=full,
        native_eeg_mse=float(np.mean((np.asarray(original['eeg_v'])-np.asarray(replica['eeg_v']))**2)),
        fork_audit=fork,forcing_hashes=forcing,causal_state_hashes=states,acquisition_states=acquisition,targets=rows,
        verified_state='All segment v/vext, HH m/h/n, Exp2Syn A/B, caused pending deliveries, phase and physical parameters verified through full-prefix replay and saved audits.',
        missing_checkpoint_capability='No restoration from a snapshot is claimed; the simulator repeats the complete prefix using exactly the same realised forcing.',
        privilege='Exact external event times, full future phase path and RNG encodings stay in privileged audit artifacts, never learner inputs.')


def qualification_binding(config,qualification_root):
    if qualification_root is None:return dict(status='unavailable',reason='Frozen qualifying EEG models not supplied')
    path=Path(qualification_root).resolve();files=[];jobs={}
    from .information import _policies
    for job in config['selected_job_ids']:
        model=path/'models'/job/'manifest.json';fit=read_json(model)
        acquisition,target=job.split('__');derived=read_json(path/'derived'/job/'manifest.json')
        physical=target_spec(target,cutoff_ms=config['cutoff_ms']);physical_hash=physical['operator_hash'] if isinstance(physical,dict) else physical.fingerprint
        if derived['risk_contract']['target_physical_hash']!=physical_hash:raise ValueError('Replay predictor physical target/support differs')
        if derived['input'].get('acquisition_id')!=acquisition or derived['input'].get('coefficient_sha256')!=coefficient_hash(_design(acquisition)[1]):raise ValueError('Replay predictor acquisition differs')
        guard=read_json(path/'models'/job/'fit_identity.json');source_root=Path(__file__).parents[2]
        dependencies={str(q.relative_to(source_root)):file_hash(q) for q in [Path(__file__).with_name('models.py'),Path(__file__).parents[1]/'v4/models.py',source_root/'training.py']}
        if guard.get('analysis_dependencies')!=dependencies:raise ValueError('Replay frozen predictor implementation changed')
        if fit.get('selection_split')!='development':raise ValueError('Replay predictor was not frozen on development')
        policies=[p for p in _policies(fit) if p['id'] in ('eeg_full_selected','eeg_short_selected')]
        members=sorted(set(['train_mean']+[m for policy in policies for m in policy['members']]))
        paths=[model,path/'models'/job/'fit_identity.json',path/'models'/job/'selection.json',path/'derived'/job/'manifest.json']
        for candidate in fit['candidates']:
            if candidate['id'] in members:
                record=next(r for r in candidate['files'] if r['path']==candidate['state_file'])
                paths.append(verify_record(path,record))
        jobs[job]=dict(policies=policies,members=members)
        for item in paths:files.append(dict(path=str(item),sha256=artifact_record(path,item)['sha256'],bytes=item.stat().st_size))
    return dict(status='complete',root=str(path),jobs=jobs,files=files,evidence_hash=canonical_hash(files))


def _qualified_tables(config,trials,replicas,records,qualification,*,root=None):
    """Same eight original gate contexts for every empirical fitted/replay row."""
    from .models import predict_from_artifact
    from .information import summarize,_unknown
    tables=[];files=[];contexts_=[r['context_id'] for r in records];structures=np.asarray([r['structure_id'] for r in records])
    regimes=np.asarray([trial['metadata']['regime']['name'] for trial in trials])
    panel=dict(context_id=np.asarray(contexts_),structure_id=structures,regime=regimes,
        future_id=np.asarray([[r['trial_id']] for r in records]))
    histories={a:np.asarray([causal_history(t['eeg_v'],t['time_ms'],cutoff_ms=config['cutoff_ms'],history_s=config['history_s'],acquisition_id=a)[0] for t in trials]) for a in config['acquisition_ids']}
    for job in config['selected_job_ids']:
        acquisition,target=job.split('__');physical=np.asarray([extract_physical_target(t['eeg_v'],t['time_ms'],target,cutoff_ms=config['cutoff_ms']) for t in trials])
        if qualification['status']!='complete':raise ValueError('Frozen qualification required for same-population replay comparisons')
        qroot=qualification['root'];mean=predict_from_artifact(qroot,job,X_physical=histories[acquisition],candidate='train_mean')
        base=mean['baseline_mean_physical'];source=[r['source'] for r in records]+qualification['files']
        baseline,_=summarize(physical[:,None,:],mean['mean_physical'],base,panel,method_id='train_mean',information_available='No current recording',method='Frozen Phase0 training mean scored on gate contexts',source=source)
        replay_predictions=np.asarray([extract_physical_target(t['eeg_v'],t['time_ms'],target,cutoff_ms=config['cutoff_ms']) for t in replicas])
        arrays=dict(Y_physical=physical[:,None,:],baseline_mean_physical=np.asarray(base),train_mean=mean['mean_physical'],deterministic_replay=replay_predictions,**panel)
        replay,_=summarize(physical[:,None,:],replay_predictions,base,panel,method_id='deterministic_replay',information_available='Complete causal state, parameters and exact realised future external inputs',method='Privileged full-prefix replay',source=source+[r['replica'] for r in records])
        replay['uncertainty_scope']='Exact-input software diagnostic; zero observed spread across two gate circuits does not estimate population-general forecasting accuracy'
        rows=[dict(id='complete_state_exact_future_theory',status='theoretical',physical_mse=0.,reference_mse=baseline['reference_mse'],error_baseline100=0.,risk_ratio=0.,relative_mse_reduction_pp=100.,method='Deterministic-system ideal, not a measured zero'),replay,
              _unknown('state_variance','Complete causal state, unknown future inputs','Within-context variance','Unavailable at M=1; do not substitute a fitted variance model'),
              _unknown('fitted_partial_state','Partial present-state features','Fitted partial-state predictor','No verified arbitrary-input partial-feature inference adapter for this gate population'),
              _unknown('eeg_ideal','Specified observed EEG history','Best possible EEG predictor','EEG-history Bayes floor remains unidentified'),
              _unknown('compressed_future_readout','Compressed EEG history','Matched-code future decoder','No verified arbitrary-input code inference adapter for this gate population; full evaluation table reports code errors separately')]
        for policy in qualification['jobs'][job]['policies']:
            pred=[predict_from_artifact(qroot,job,X_physical=histories[acquisition],candidate=m)['mean_physical'][0] for m in policy['members']]
            row,_=summarize(physical[:,None,:],np.asarray(pred),base,panel,method_id=policy['id'],information_available=f'EEG history, {policy["history_samples"]} acquired samples',method='Frozen Phase0 '+policy['family']+' policy',source=source)
            row.update(policy);rows.append(row);arrays[policy['id']]=np.asarray(pred)
        rows.append(baseline)
        artifact=None
        if root is not None:
            output=Path(root)/'replay/predictions'/(job+'.npz');atomic_npz(output,**arrays);artifact=artifact_record(Path(root),output);files.append(artifact)
        tables.append(dict(predictions=artifact,job_id=job,target_id=target,acquisition=acquisition,target_spec=target_metadata(target,cutoff_ms=config['cutoff_ms']),
            population='Two registered timing circuits, four regimes each, prefix0, one scoring future; separate from inspected/evaluation populations',
            context_ids=contexts_,n_circuits=2,n_contexts=8,n_scoring_episodes=8,rows=rows,
            uncertainty='Descriptive two-circuit intervals conditional on frozen Phase0 fitting; M1 cannot identify within-context variance',
            baseline100_units='Presentation scaling by target-specific Phase0 training mean scored on exactly these eight contexts'))
    return tables,files


def _inputs(config,root,timing_root,qualification_root=None):
    root=Path(root);timing=Path(timing_root).resolve();gc=read_json(timing/'resolved_config.json')
    if gc['claim_status']!='timing_only' or gc['backend']!='neuron':raise ValueError('Replay requires the registered physical timing gate')
    planned=contexts(gc);specs=[trial_spec(gc,p,'score',0) for p in planned]
    if len(specs)!=8 or set(gc['data']['structures']['evaluation'])!={37510,37511} or any(p['score_futures']!=1 or p['reference_futures'] for p in planned):
        raise ValueError('Exactly eight registered timing contexts, one scoring future each, required')
    for key in ('simulation','cutoff_ms','stop_ms'):
        if gc[key]!=config[key]:raise ValueError('Replay physics/support differ from current protocol')
    manifest_path=timing/'generation/native/manifest.json';native=read_json(manifest_path)
    rows=native['trials'];lookup={r['trial_id']:r for r in rows}
    if native['status']!='complete' or len(rows)!=8 or set(lookup)!={s['trial_id'] for s in specs}:raise ValueError('Timing gate incomplete or allocation mismatch')
    contract=canonical_hash(generation_contract(gc));originals=[]
    for spec in specs:
        row=lookup[spec['trial_id']]
        if row['spec']!=spec or row['generation_contract_hash']!=contract:raise ValueError('Timing physical generation contract changed')
        path=verify_row(timing/'generation',row)
        originals.append(dict(row=row,path=path))
    qualification=qualification_binding(config,qualification_root)
    signature=canonical_hash(dict(config_targets=target_ids(config),qualification=qualification,timing_config=canonical_hash(gc),
        originals=[o['row']['sha256'] for o in originals],generation_contract=contract,
        analysis=module_hash('replay','targets'),cutoff_ms=gc['cutoff_ms']))
    return gc,originals,signature,qualification


def verify_summary(config,root,timing_root):
    root=Path(root);path=root/'replay/summary.json';summary=read_json(path)
    gc,originals,signature,qualification=_inputs(config,root,timing_root,summary.get('qualification_root'))
    if qualification['status']!='complete':raise ValueError('Same-population frozen EEG comparison missing from replay diagnostic')
    if summary.get('signature')!=signature or summary.get('status')!='complete' or summary.get('passed') is not True or summary.get('n_replays')!=8:
        raise ValueError('Replay summary incomplete, failed or stale')
    if len(summary['trials'])!=8 or {r['trial_id'] for r in summary['trials']}!={o['row']['trial_id'] for o in originals}:raise ValueError('Replay summary allocation mismatch')
    for record in summary['files']:verify_record(root,record)
    expected=set(target_ids(config))
    lookup={o['row']['trial_id']:o for o in originals}
    for row in summary['trials']:
        original=lookup[row['trial_id']]
        expected_source=dict(path=str(original['path']),sha256=original['row']['sha256'],bytes=original['row']['bytes'])
        if row['source']!=expected_source:raise ValueError('Replay comparison original changed')
        verify_record(root,row['replica'])
        actual=compare_trials(_read_fork(original['path']),_read_fork(root/row['replica']['path']),cutoff_ms=gc['cutoff_ms'],targets=target_ids(config))
        if canonical_hash(actual)!=canonical_hash(row['audit']):raise ValueError('Saved replay audit does not reproduce from actual native arrays')
        if set(r['target_id'] for r in row['audit']['targets'])!=expected:raise ValueError('Replay target support set differs')
        if not row['audit']['passed'] or not row['audit']['full_trace_identical']:raise ValueError('Replay failed exact-trace audit')
        for target in row['audit']['targets']:
            if target['target_spec']!=target_metadata(target['target_id'],cutoff_ms=gc['cutoff_ms']):raise ValueError('Replay target operator/support changed')
    if {t['job_id'] for t in summary['tables']}!=set(config['selected_job_ids']):raise ValueError('Replay ordinary comparison jobs incomplete')
    from .information import summarize
    for table in summary['tables']:
        if not table.get('predictions'):raise ValueError('Replay numeric predictions absent')
        predpath=verify_record(root,table['predictions'])
        with np.load(predpath,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        panel={k:arrays[k] for k in ('context_id','structure_id','regime','future_id')}
        if panel['context_id'].astype(str).tolist()!=table['context_ids']:raise ValueError('Replay table context IDs differ')
        for row in table['rows']:
            if row['status']!='measured':continue
            recomputed,_=summarize(arrays['Y_physical'],arrays[row['id']],arrays['baseline_mean_physical'],panel,method_id=row['id'])
            for key in ('physical_mse','reference_mse','error_baseline100','relative_mse_reduction_pp','circuit_components','state_variance'):
                if canonical_hash(recomputed[key])!=canonical_hash(row[key]):raise ValueError('Replay table does not reproduce from saved physical predictions')
    return dict(status='complete',passed=True,n_replays=8,summed_replay_seconds=summary['summed_replay_seconds'],
        wall_seconds=summary.get('command_wall_seconds',summary['summed_replay_seconds']),native_bytes=summary['native_bytes'],evidence_hash=artifact_record(root,path)['sha256'],summary=artifact_record(root,path),
        scope='Separate registered timing population, not confirmation predictor error')


def run(config,root,timing_root,*,execute=False,resume=False,qualification_root=None):
    root=Path(root)
    if qualification_root is None and (root/'plan.json').exists():qualification_root=read_json(root/'plan.json').get('phase0_root')
    gc,originals,signature,qualification=_inputs(config,root,timing_root,qualification_root);dest=root/'replay';marker=dest/'summary.json'
    if marker.exists():
        old=read_json(marker)
        if not resume or old['signature']!=signature:raise ValueError('Replay analysis dependency mismatch; preserve old output')
        verify_summary(config,root,timing_root);return old
    if qualification['status']!='complete':raise ValueError('Frozen Phase0 models required before replay generation')
    from .planning import require_execution
    require_execution(config,root,execute,numerical=True)
    records=[];files=[];total=0.;started_command=time.perf_counter()
    for original in originals:
        row=original['row'];tid=row['trial_id'];path=dest/'native'/(tid+'.h5');recordpath=dest/'records'/(tid+'.json')
        source_record=dict(path=str(original['path']),sha256=row['sha256'],bytes=row['bytes'])
        identity=dict(original=source_record,spec=row['spec'],generation_contract_hash=row['generation_contract_hash'])
        if recordpath.exists():
            saved=read_json(recordpath)
            if not resume or saved['identity']!=identity:raise ValueError('Replay physical record mismatch')
            verify_record(root,saved['artifact']);seconds=saved['generation_seconds']
        else:
            if path.exists():
                import h5py,json
                with h5py.File(path,'r') as f:attrs=json.loads(f.attrs['metadata_json'])
                if attrs.get('replay_identity')!=identity:raise ValueError('Orphan replay file belongs to another original')
                seconds=float(attrs['generation_seconds'])
            else:
                started=time.perf_counter();replica=generate_trial(adapter(gc),row['spec']);seconds=time.perf_counter()-started
                write_trial(path,replica,dict(metadata=replica['metadata'],spec=row['spec'],replay_identity=identity,generation_seconds=seconds))
            saved=dict(identity=identity,artifact=artifact_record(root,path),generation_seconds=seconds)
            atomic_json(recordpath,saved)
        audit=compare_trials(_read_fork(original['path']),_read_fork(path),cutoff_ms=gc['cutoff_ms'],targets=target_ids(config))
        records.append(dict(trial_id=tid,context_id=row['context_id'],structure_id=row['structure_id'],source=source_record,replica=saved['artifact'],audit=audit,generation_seconds=seconds))
        files.extend([saved['artifact'],artifact_record(root,recordpath)]);total+=seconds
        atomic_json(dest/'progress.json',dict(completed=len(records),planned=8,status='running'))
    originals_data=[_read_fork(o['path']) for o in originals];replicas=[_read_fork(root/r['replica']['path']) for r in records]
    tables,table_files=_qualified_tables(config,originals_data,replicas,records,qualification,root=root);files.extend(table_files)
    result=dict(status='complete',passed=all(r['audit']['passed'] for r in records),signature=signature,n_replays=8,n_circuits=2,n_contexts=8,
        trials=records,tables=tables,qualification_root=qualification.get('root'),qualification=qualification,files=files,summed_replay_seconds=total,command_wall_seconds=time.perf_counter()-started_command,
        execution_topology='serial full-prefix replays; independent generation workers are not assumed here',native_bytes=sum(r['replica']['bytes'] for r in records),
        cost_scope='Eight additional full-prefix physical episodes; original eight timing-gate episodes reused as comparison',
        full_checkpoint_restoration=False,ordinary_predictor_access_to_future_inputs=False,
        interpretation='Privileged exact-realised-forcing replay only; measured errors are not mixed into a table from different evaluation circuits.')
    atomic_json(marker,result)
    if not result['passed']:raise RuntimeError('Deterministic replay mismatch; measurements preserved in replay/summary.json')
    return result
