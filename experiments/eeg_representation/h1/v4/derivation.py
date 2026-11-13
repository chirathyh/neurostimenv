"""Read-only native reuse and target-specific, training-only v4 derivation.

Original generation provenance is retained in a content-addressed import record.
Changed operators receive new fingerprints. Missing native recordings make only
those specific derivations unavailable; saved-array work can still complete.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import io
import json
import hashlib
import numpy as np
from scipy import signal
from ..artifacts import atomic_json, atomic_npz, artifact_record, verify_record, source_hash
from ..targets import canonical_hash
from ...forks import validate_fork_prefixes
from . import SCHEMA_VERSION
from .evidence import EvidenceSource
from .measurement import causal_history, qualify, metadata
from .targets import TargetSpec, default_specs, extract_target, from_binned, axes, floor_activations

SPLITS = ('train','development','evaluation')


def _native(source,row):
    import h5py
    with source.open_binary(row['path']) as stream:blob=stream.read()
    if hashlib.sha256(blob).hexdigest()!=row['sha256'] or len(blob)!=row['bytes']:
        raise ValueError('Native import checksum mismatch: '+row['path'])
    with h5py.File(io.BytesIO(blob),'r') as h:
        attrs=json.loads(h.attrs['metadata_json'])
        p={k:v[()] for k,v in h['privileged'].items()}
        p.update({k:np.asarray(json.loads(v)) for k,v in h['privileged'].attrs.items()})
        for key in ('pending_recurrent_events','pending_external_events'):
            if key in p:p[key]=p[key].reshape(-1,6)
        return {**{k:h[k][:] for k in ('time_ms','eeg_v','dipole_na_um')},'privileged':p,'metadata':attrs['metadata']}


def _legacy(source):
    if not source.exists('panels/primary/manifest.json'):return None,{}
    source.verify('panels/primary/manifest.json')
    manifest=source.read_json('panels/primary/manifest.json');panels={}
    original_spec=manifest.get('target_spec',{})
    if original_spec and any(original_spec.get(k)!=v for k,v in {'operator':'bin_average_waveform','lead_ms':10.,'duration_ms':128.,'output_grid_hz':500.}.items()):
        raise ValueError('Saved primary arrays are not the frozen 10/128-ms native-box contract')
    for split in SPLITS:
        path=f'panels/primary/{split}.npz'
        if not source.exists(path):return manifest,{}
        source.verify(path);panels[split]=source.read_npz(path)
    return manifest,panels


def _labels(trial,context,history_s):
    t=trial['time_ms'];x=trial['eeg_v'].reshape(-1);cut=context['cutoff_ms']
    keep=(t>cut-history_s*1000)&(t<=cut);y=x[keep];times=(t[keep]-cut)/1000
    f,p=signal.periodogram(y,fs=1000/(t[1]-t[0]),detrend='constant',window='hann')
    band=(f>=4)&(f<=30);peak=float(f[band][np.argmax(p[band])])
    phase=np.angle(np.sum((y-y.mean())*np.exp(-2j*np.pi*peak*times)))
    regime=context['regime'];rhythm=int(regime.get('modulation_depth',0)>0)
    frequency=float(regime.get('frequency_hz',0));power=float(np.mean(y*y));ac=float(np.var(y))
    return {'task_rhythm':rhythm,'task_carrier':{10.:0,16.:1,24.:2}.get(frequency,-1) if rhythm else -1,
            'task_phase_sin':float(np.sin(phase)),'task_phase_cos':float(np.cos(phase)),
            'task_log_power':float(np.log(max(power,1e-40))),
            'task_log_mean_square':float(np.log(max(power,1e-40))),
            'task_log_ac_variance':float(np.log(max(ac,1e-40))),
            'task_diffusion':float(regime.get('diffusion',0)),'task_kinetics':float(regime.get('kinetics_scale',1.))}


def _legacy_contexts(panels):
    lookup={}
    for split,p in panels.items():
        for i,cid in enumerate(p['context_id']):lookup[str(cid)]=(split,i)
    return lookup


def _saved_records(panels,specs):
    """Only derive targets whose exact native block is already saved."""
    records=[]
    for split,p in panels.items():
        for i,cid in enumerate(p['context_id']):
            r={'context':{'context_id':str(cid),'structure_id':int(p['structure_id'][i]),'split':split,
                          'regime':{'name':str(p['regime'][i])},'cutoff_ms':6000.},
               'histories':{'A0':p['X_physical'][i]},'targets':{},'ids':{},'floor_counts':{},
               'tasks':{k:p[k][i].item() for k in p if k.startswith('task_')}}
            if 'task_log_power' in r['tasks']:r['tasks']['task_log_mean_square']=r['tasks']['task_log_power']
            for pool,yk,mk,ik in [('score','Y_physical','future_mask','future_id'),('reference','reference_Y_physical','reference_mask','reference_future_id')]:
                r['ids'][pool]=list(map(str,p[ik][i,p[mk][i]]))
                values=p[yk][i,p[mk][i]]
                for spec in specs:
                    if spec.family=='T3' or spec.lead_ms!=10. or spec.duration_ms>128.:continue
                    w=values[...,:round(spec.duration_ms/2)]
                    r['targets'].setdefault(spec.id,{})[pool]=from_binned(w,spec)
                    r['floor_counts'].setdefault(spec.id,{})[pool]=floor_activations(w,spec)
            records.append(r)
    return records


def _native_records(source,rows,specs,acquisitions,history_s,legacy_panels,cutoff):
    by_context={}
    for row in rows:by_context.setdefault(row['context_id'],[]).append(row)
    legacy_lookup=_legacy_contexts(legacy_panels)
    order=list(legacy_lookup) if legacy_lookup and set(legacy_lookup)==set(by_context) else list(by_context)
    records=[];audits=[];unavailable={};input_meta={};target_unavailable={}
    for count,cid in enumerate(order):
        group=sorted(by_context[cid],key=lambda r:(r['pool'],r['future_index']))
        first=_native(source,group[0]);spec0=group[0]['spec']
        context={'context_id':cid,'structure_id':int(group[0]['structure_id']),'split':group[0]['split'],
                 'regime':spec0['regime'],'cutoff_ms':float(spec0['future_start_ms'])}
        if context['cutoff_ms']!=cutoff:raise ValueError('Mixed cutoffs are incompatible risk contracts')
        r={'context':context,'histories':{},'targets':{},'ids':{'score':[],'reference':[]},
           'floor_counts':{},'tasks':_labels(first,context,history_s)}
        for acquisition in acquisitions:
            try:
                x,rel,meta=causal_history(first['eeg_v'],first['time_ms'],cutoff_ms=cutoff,history_s=history_s,acquisition_id=acquisition)
                input_meta[acquisition]=meta
                if acquisition=='A0' and cid in legacy_lookup:
                    split,i=legacy_lookup[cid];old=legacy_panels[split]['X_physical'][i]
                    if old.shape!=x.shape or not np.allclose(old,x,rtol=1e-12,atol=1e-24):
                        raise ValueError('Frozen A0 acquisition does not reproduce original arrays')
                    x=old.copy()  # preserve imported A0 values exactly, including library rounding.
                r['histories'][acquisition]=x
            except ValueError as exc:
                if acquisition!='A2':raise
                unavailable[acquisition]=str(exc)
        ref_ids=set();score_ids=set();reference_seeds=set();scoring_seeds=set();fork_audit=None
        for n,row in enumerate(group):
            trial=first if n==0 else _native(source,row)
            if n:fork_audit=validate_fork_prefixes([first,trial],cutoff)
            pool=row['pool']
            if pool not in r['ids']:raise ValueError('Unknown native future pool')
            r['ids'][pool].append(row['trial_id'])
            (score_ids if pool=='score' else ref_ids).add(row['trial_id'])
            (scoring_seeds if pool=='score' else reference_seeds).add(row['spec']['future_seed'])
            for spec in specs:
                if spec.id in target_unavailable:continue
                try:value=extract_target(trial['eeg_v'],trial['time_ms'],spec)
                except ValueError as exc:
                    if 'support is absent' not in str(exc):raise
                    target_unavailable[spec.id]=str(exc);continue
                r['targets'].setdefault(spec.id,{'score':[],'reference':[]})[pool].append(value)
                if 'log_' in spec.family:
                    rawspec=TargetSpec(**{**asdict(spec),'family':'T0'})
                    w=extract_target(trial['eeg_v'],trial['time_ms'],rawspec)
                    d=r['floor_counts'].setdefault(spec.id,{'score':0,'reference':0})
                    d[pool]+=floor_activations(w,spec)
            if cid in legacy_lookup:
                split,i=legacy_lookup[cid];p=legacy_panels[split]
                key='future_id' if pool=='score' else 'reference_future_id'
                inds=np.flatnonzero(p[key][i]==row['trial_id'])
                if len(inds)!=1:raise ValueError('Native/legacy future ID mismatch')
                rawspec=TargetSpec(cutoff_ms=cutoff)
                val=extract_target(trial['eeg_v'],trial['time_ms'],rawspec)
                old=p['Y_physical' if pool=='score' else 'reference_Y_physical'][i,inds[0]]
                if not np.array_equal(val,old):raise ValueError('Frozen original physical target changed')
        if ref_ids&score_ids or reference_seeds&scoring_seeds:
            raise ValueError('Scoring/reference future pools overlap')
        for spec_id,pools in r['targets'].items():
            d=next(s.dimension for s in specs if s.id==spec_id)
            for pool,values in pools.items():pools[pool]=np.asarray(values,dtype=float).reshape(-1,d)
        records.append(r)
        audits.append({'context_id':cid,'status':'complete' if fork_audit else 'single future',
                       'fork_audit':fork_audit,'future_pools_disjoint':True,
                       'score_futures':len(score_ids),'reference_futures':len(ref_ids)})
        if count%32==0:print(f'V4 derivation native contexts {count+1}/{len(order)}',flush=True)
    return records,audits,input_meta,unavailable,target_unavailable


def _normalization(records,acquisition,spec):
    training=[r for r in records if r['context']['split']=='train']
    x=np.stack([r['histories'][acquisition] for r in training])
    y=np.stack([r['targets'][spec.id]['score'].mean(0) for r in training])
    # Equal contexts, then equal circuits. This remains training-only with unequal designs.
    ids=np.asarray([r['context']['structure_id'] for r in training]);circuits=np.unique(ids)
    weights=np.asarray([1/(len(circuits)*np.sum(ids==s)) for s in ids])
    xo=float(np.sum(weights*x.mean(1)));xs=float(np.sqrt(np.sum(weights*np.mean((x-xo)**2,axis=1))))
    yo=np.sum(y*weights[:,None],axis=0)
    ys=float(np.sqrt(np.sum(weights*np.mean((y-yo)**2,axis=1))))
    if min(xs,ys)<=0 or not np.isfinite([xs,ys]).all():
        raise ValueError('Zero or nonfinite training normalization; no arbitrary data-dependent floor')
    result={'x_offset':xo,'x_scale':xs,'y_offset':yo.tolist(),'y_scale':ys,'fit_split':'train',
            'fit_contexts':[r['context']['context_id'] for r in training],
            'target_fit_support':'training scored futures only, one context mean per training context; no development/evaluation outcomes',
            'weighting':'equal contexts within each training circuit, then equal circuits'}
    result['id']=canonical_hash(result);return result


def derive(config,root,source,resume=False):
    root=Path(root);directory=root/'derived';index=directory/'manifest.json'
    own_source=not isinstance(source,EvidenceSource)
    source=EvidenceSource(source) if own_source else source
    try:
        provenance=source.provenance();analysis_hash=source_hash();config_hash=canonical_hash(config)
        if index.exists():
            if not resume:raise FileExistsError('Derived outputs exist; use --resume')
            prior=json.loads(index.read_text())
            if (prior['analysis_source_hash'],prior['config_hash'],prior['source_content_id'])!=(analysis_hash,config_hash,provenance['content_id']):
                raise ValueError('Derivation resume fingerprint mismatch')
            for expected in prior.get('input_artifacts',[]):
                actual=source.verify(expected['path'])
                if actual['sha256']!=expected['sha256'] or actual['bytes']!=expected['bytes']:
                    raise ValueError('Import dependency changed since derivation: '+expected['path'])
            for row in prior['jobs']:
                if row['status']=='complete':load_panel(root,row['id'])
            return prior
        qualify(root)
        original_manifest,old=_legacy(source)
        raw=source.read_json('native/manifest.json') if source.exists('native/manifest.json') else {}
        rows=raw.get('trials',[])
        cutoff=float(rows[0]['spec']['future_start_ms']) if rows else float(config.get('cutoff_ms',6000.))
        history=float(config.get('history_s',(original_manifest or {}).get('input',{}).get('history_s',4.)))
        specs=[TargetSpec(**{**v,'cutoff_ms':cutoff}) for v in config['target_specs']] if config.get('target_specs') else default_specs(cutoff)
        acquisitions=config.get('acquisition_ids',['A0','A1','A2'])
        native_complete=bool(rows) and all(source.exists(r['path']) for r in rows)
        if native_complete:
            records,audits,input_meta,acq_unavailable,target_unavailable=_native_records(source,rows,specs,acquisitions,history,old,cutoff)
        elif old:
            records=_saved_records(old,specs);audits=[];input_meta={'A0':metadata('A0')}
            acq_unavailable={a:'Native EEG unavailable in supplied archive; changed input cannot be recovered from filtered samples' for a in acquisitions if a!='A0'}
            target_unavailable={s.id:'Required native future support unavailable; existing arrays contain only (cutoff+10,cutoff+138] 2-ms bins' for s in specs if s.family=='T3' or s.lead_ms!=10 or s.duration_ms>128}
        else:
            raise ValueError('Neither complete native data nor the original saved primary panels are available')
        splits={r['context']['split'] for r in records}
        if not set(SPLITS)<=splits:raise ValueError('Derivation requires train, development and evaluation splits; a timing gate alone is not trainable')
        population_id=canonical_hash([r['context']['context_id'] for r in records if r['context']['split']=='evaluation'])
        atomic_json(directory/'fork_validation.json',{'native_directly_verified':native_complete,'contexts':audits,
            'interpretation':'Raw hashes plus exact cutoff EEG/dipole/state/pending-event checks; privileged arrays never become model inputs'})
        jobs=[]
        for acquisition in acquisitions:
            for spec in specs:
                jid=f'{acquisition}__{spec.id}';out=directory/jid
                reason=acq_unavailable.get(acquisition) or target_unavailable.get(spec.id)
                if reason:
                    jobs.append({'id':jid,'job_id':jid,'status':'unavailable','reason':reason});continue
                if any(acquisition not in r['histories'] or spec.id not in r['targets'] for r in records):
                    raise ValueError('Unexpected incomplete derivation '+jid)
                anchor=acquisition=='A0' and spec.family=='T0' and spec.lead_ms==10 and spec.duration_ms==128 and bool(old)
                norm=original_manifest['normalization'] if anchor else _normalization(records,acquisition,spec)
                if acquisition=='A0' and old and not anchor:
                    norm['x_offset']=original_manifest['normalization']['x_offset']
                    norm['x_scale']=original_manifest['normalization']['x_scale']
                    norm['input_normalization']='frozen original A0 training normalization preserved'
                    norm.pop('id',None);norm['id']=canonical_hash(norm)
                yo=np.asarray(norm['y_offset']);ys=float(norm['y_scale']);files={}
                for split in SPLITS:
                    selected=[r for r in records if r['context']['split']==split]
                    n=len(selected);m=max(len(r['ids']['score']) for r in selected);nr=max(len(r['ids']['reference']) for r in selected);d=spec.dimension
                    yp=np.zeros((n,m,d));rp=np.zeros((n,nr,d));mask=np.zeros((n,m),bool);rmask=np.zeros((n,nr),bool)
                    fi=np.full((n,m),'',dtype='U128');ri=np.full((n,nr),'',dtype='U128')
                    for k,r in enumerate(selected):
                        a=len(r['ids']['score']);b=len(r['ids']['reference'])
                        yp[k,:a]=r['targets'][spec.id]['score'];mask[k,:a]=True;fi[k,:a]=r['ids']['score']
                        if b:rp[k,:b]=r['targets'][spec.id]['reference'];rmask[k,:b]=True;ri[k,:b]=r['ids']['reference']
                    xp=np.stack([r['histories'][acquisition] for r in selected])
                    panel={'X':((xp-norm['x_offset'])/norm['x_scale']).astype(np.float32),'X_physical':xp,
                           'Y':(yp-yo)/ys,'Y_physical':yp,'reference_Y':(rp-yo)/ys,'reference_Y_physical':rp,
                           'future_mask':mask,'reference_mask':rmask,'future_id':fi,'reference_future_id':ri,
                           'context_id':np.asarray([r['context']['context_id'] for r in selected]),
                           'structure_id':np.asarray([r['context']['structure_id'] for r in selected],dtype=np.int64),
                           'regime':np.asarray([r['context']['regime']['name'] for r in selected]),
                           'input_time_ms':-history*1000+np.arange(1,xp.shape[1]+1)*2.,
                           'acquisition_clipping_fraction':np.zeros(n),**axes(spec)}
                    for key in selected[0]['tasks']:
                        if all(key in r['tasks'] for r in selected):panel[key]=np.asarray([r['tasks'][key] for r in selected])
                    if anchor:
                        for key,value in old[split].items():
                            if key in panel and not key.startswith('task_') and not np.array_equal(panel[key],value):
                                # Padding Unicode widths are serialization metadata, values remain exact.
                                raise ValueError('Frozen anchor array differs: '+key)
                            panel[key]=value.copy()
                    p=out/f'{split}.npz';atomic_npz(p,**panel);files[split]=artifact_record(root,p)
                contract={'target_physical_hash':spec.fingerprint,'population_id':population_id,
                          'acquisition_hash':input_meta[acquisition]['operator_hash'],'normalization_id':norm['id'],
                          'units':spec.units,'dimension':spec.dimension,'support':spec.to_dict()['physical_support_ms'],
                          'conditioning':'physical-state reference vs EEG-only predictions; exact acquisition contract retained'}
                fingerprint=canonical_hash(contract)
                manifest={'schema_version':SCHEMA_VERSION,'status':'complete','job_id':jid,
                    'target_spec':spec.to_dict(),'target_fingerprint':fingerprint,'comparison_fingerprint':fingerprint,
                    'physical_target_fingerprint':spec.fingerprint,'risk_contract':contract,
                    'input':{**input_meta[acquisition],'history_s':history},'normalization':norm,
                    'target_family':spec.family,'operator':spec.family,'lead_ms':spec.lead_ms,'duration_ms':spec.duration_ms,'output_grid_hz':500.,
                    'population_id':population_id,'files':files,'config_hash':config_hash,
                    'analysis_source_hash':analysis_hash,'source_content_id':provenance['content_id'],
                    'original_generation_provenance':provenance,'native_verified':native_complete,
                    'frozen_legacy_anchor_arrays_preserved':anchor,
                    'original_target_fingerprint':original_manifest.get('target_fingerprint') if anchor else None,
                    'log_floor_activations':{s:sum(sum(r['floor_counts'].get(spec.id,{}).values()) for r in records if r['context']['split']==s) for s in SPLITS},
                    'task_definitions':{'task_log_power':'legacy natural log of physical predecision mean square in V^2; offset-sensitive, not band power',
                                        'task_log_mean_square':'explicit alias of legacy task_log_power',
                                        'task_log_ac_variance':'natural log of physical predecision centered variance in V^2; unavailable if native history absent',
                                        'carrier':'10/16/24 Hz generator modulation class conditional on rhythmic membership; homogeneous=-1'},
                    'interpretation':'Existing inspected circuits are exploratory; changes of acquisition or target are new information/risk conditions.'}
                atomic_json(out/'manifest.json',manifest)
                jobs.append({'id':jid,'job_id':jid,'status':'complete',**artifact_record(root,out/'manifest.json'),
                             'actual_target_dimension':spec.dimension,'history_samples':round(history*500),
                             'target_fingerprint':fingerprint})
        input_artifacts=[{k:r[k] for k in ('path','sha256','bytes')} for r in rows] if native_complete else []
        if old:
            input_artifacts += [source.verify('panels/primary/manifest.json')]
            input_artifacts += [source.verify(f'panels/primary/{s}.npz') for s in SPLITS]
        result={'schema_version':SCHEMA_VERSION,'status':'complete','jobs':jobs,
                'input_artifacts':input_artifacts,
                'completed_jobs':sum(r['status']=='complete' for r in jobs),'planned_jobs':len(jobs),
                'analysis_source_hash':analysis_hash,'config_hash':config_hash,'source_content_id':provenance['content_id'],
                'native_verified':native_complete,'source':provenance,
                'original_anchor_reproduced':bool(old),'circuits':{s:len({r['context']['structure_id'] for r in records if r['context']['split']==s}) for s in SPLITS}}
        atomic_json(index,result);return result
    finally:
        if own_source:source.close()


def load_panel(root,job_id):
    root=Path(root);index=json.loads((root/'derived/manifest.json').read_text())
    entry=next((r for r in index['jobs'] if r['id']==job_id),None)
    if entry is None or entry['status']!='complete':raise ValueError('Unavailable derived job '+job_id)
    manifest=json.loads(verify_record(root,entry).read_text());panels={};seen_circuits=set();seen_contexts=set();seen_futures=set()
    if canonical_hash(manifest['risk_contract'])!=manifest['target_fingerprint']:
        raise ValueError('Target risk contract fingerprint mismatch')
    for split,record in manifest['files'].items():
        path=verify_record(root,record)
        with np.load(path,allow_pickle=False) as archive:panels[split]={k:archive[k] for k in archive.files}
        p=panels[split];circuits=set(p['structure_id']);contexts=set(p['context_id'])
        if circuits&seen_circuits or contexts&seen_contexts or len(contexts)!=len(p['context_id']):raise ValueError('Circuit/context split leakage')
        seen_circuits|=circuits;seen_contexts|=contexts
        ids=list(p['future_id'][p['future_mask']])+list(p['reference_future_id'][p['reference_mask']])
        if len(set(ids))!=len(ids) or set(ids)&seen_futures:raise ValueError('Future/reference pool leakage')
        seen_futures|=set(ids)
        if p['Y'].shape[-1]!=manifest['target_spec']['dimension']:raise ValueError('Wrong target dimension')
    return panels,manifest


def audit_existing(config,root,source,resume=False):
    """Derive supported operators, then reproduce frozen scores numerically.

    This reads evidence only. It never fits models, launches NEURON, edits the
    imported files or loads model pickles. Full native archives additionally get
    the exact fork/target audit performed by :func:`derive`.
    """
    from scipy.special import ndtr
    from scipy.stats import norm as gaussian
    root=Path(root);owned=not isinstance(source,EvidenceSource)
    src=EvidenceSource(source) if owned else source
    try:
        derived=derive(config,root,src,resume=resume)
        original,panels=_legacy(src)
        result={'status':'complete','native_verified':derived['native_verified'],
                'supported_jobs':derived['completed_jobs'],'total_jobs':derived['planned_jobs'],
                'unavailable_jobs':[r for r in derived['jobs'] if r['status']=='unavailable'],
                'source':src.provenance(),'analysis_source_hash':source_hash(),
                'original_arrays':{'status':'complete' if panels else 'unavailable',
                                   'reason':None if panels else 'No frozen primary panels supplied'},
                'frozen_scores':{'status':'unavailable','reason':'No saved prediction bundle supplied'},
                'claim':'Exploratory reanalysis of inspected circuits; physical-state reference is not the EEG-history Bayes risk'}
        if panels:
            p=panels['evaluation'];y=p['Y'];mask=p['future_mask'];circuits=p['structure_id']
            def circuit_mean(values):
                values=np.asarray(values)
                return float(np.mean([values[circuits==c].mean() for c in np.unique(circuits)]))
            floors=[];eligible=[]
            for i in range(len(y)):
                yi=y[i,mask[i]]
                if len(yi)>=2:floors.append(float(np.var(yi,axis=0,ddof=1).mean()));eligible.append(i)
            floor=None
            if len(eligible)==len(y):floor=circuit_mean(floors)
            training=panels['train'];tid=training['structure_id']
            conditional_train=np.asarray([training['Y'][i,training['future_mask'][i]].mean(0) for i in range(len(tid))])
            mu=np.mean([conditional_train[tid==c].mean(0) for c in np.unique(tid)],axis=0)
            risk=circuit_mean([np.mean((y[i,mask[i]]-mu)**2) for i in range(len(y))])
            scale=float(original['normalization']['y_scale'])
            result['original_arrays'].update({'train_mean_risk':risk,'state_reference':floor,
                'state_reference_over_mean_risk':floor/risk if floor is not None and risk>0 else None,
                'physical_mse_conversion':scale**2,'train_mean_physical_risk':risk*scale**2,
                'state_reference_physical':floor*scale**2 if floor is not None else None,
                'evaluation_contexts':len(y),'evaluation_circuits':len(np.unique(circuits)),
                'scoring_futures':int(mask.sum()),'independent_reference_futures':int(p['reference_mask'].sum()),
                'frozen_anchor_arrays_preserved':True,'raw_target_exact_reproduction':derived['native_verified'],
                'raw_target_verification_status':'complete' if derived['native_verified'] else 'unavailable without native arrays'})
            pred_path='benchmarks/primary/predictions.npz'
            if src.exists(pred_path):
                pred_record=src.verify(pred_path);pred=src.read_npz(pred_path)
                if str(pred['target_fingerprint'].item())!=original['target_fingerprint']:
                    raise ValueError('Original prediction/target contracts differ')
                if not np.array_equal(pred['context_id'],p['context_id']):raise ValueError('Original predictions have different context order')
                rows=[]
                for k,name in enumerate(pred['model_names']):
                    mean=pred['mean'][k];variance=pred['variance'][k]
                    if np.any(variance<=0):raise ValueError('Original predictive variance must be positive')
                    contexts=[]
                    for i in range(len(y)):
                        yi=y[i,mask[i]];err=yi-mean[i];sd=np.sqrt(variance[i]);z=err/sd
                        contexts.append({'mean_risk':float(np.mean(err**2)),
                            'gaussian_nll':float(np.mean(.5*(np.log(2*np.pi*variance[i])+err**2/variance[i]))),
                            'marginal_crps':float(np.mean(sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi)))),
                            'coverage_50':float(np.mean(abs(z)<=gaussian.ppf(.75))),
                            'coverage_80':float(np.mean(abs(z)<=gaussian.ppf(.9))),
                            'coverage_95':float(np.mean(abs(z)<=gaussian.ppf(.975)))})
                    metrics={key:circuit_mean([r[key] for r in contexts]) for key in contexts[0]}
                    score_path=f'benchmarks/primary/scores/{name}.json';errors={}
                    if src.exists(score_path):
                        src.verify(score_path);oldscore=src.read_json(score_path)['summary']
                        for key,value in metrics.items():
                            expected=float(oldscore[key]['mean']);errors[key]=abs(value-expected)
                            if not np.isclose(value,expected,rtol=1e-11,atol=1e-12):raise ValueError(f'Frozen score reproduction failed: {name}/{key}')
                    rows.append({'model':str(name),'metrics':metrics,'saved_metric_absolute_errors':errors,
                                 'comparison_status':'complete' if errors else 'unavailable old score table'})
                result['frozen_scores']={'status':'complete','models':rows,'prediction_artifact':pred_record,
                    'weighting':'equal contexts within each circuit, then equal circuits',
                    'note':'Original marginal scores reproduced; their old uncertainty summaries are not silently endorsed or relabeled'}
        atomic_json(root/'audit_existing.json',result)
        return result
    finally:
        if owned:src.close()
