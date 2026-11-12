"""Matched causal prefixes, private future pools, and training-only acquisition.

Simulation delegates to the already-validated adapters. Each episode stops at
its last required sample; no checkpoint cache is claimed. All outputs are new
H1 artifacts, never mutations of the pilot or COSYNE data.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import time, resource
import numpy as np
from scipy import signal
from . import SCHEMA_VERSION
from .config import context_specs,cutoff_ms,stop_ms
from .targets import TargetSpec,canonical_hash,extract_target,operator_from_projected
from .acquisition import causal_history,fit_range,quantize
from .artifacts import (atomic_json,read_json,atomic_npz,file_hash,rng_seed,artifact_record,
                        verify_record,stage_complete,resume_stage,source_hash)
from ..storage import write_trial
from ..forks import _read_fork,validate_fork_prefixes


def adapter_config(c):
    from ..schema import DEFAULTS
    out=deepcopy(DEFAULTS)
    out['simulation'].update(c['simulation'])
    out['simulation']['phase_grid_dt_ms']=c['simulation']['phase_grid_ms']
    out['simulation']['duration_s']=stop_ms(c)/1000-c['simulation']['burn_in_s']
    out['data']['backend']=c['backend'];out['data']['analytical_kind']=c['analytical']['kind']
    out['analytical']={k:v for k,v in c['analytical'].items() if k!='kind'}
    return out


def trial_spec(c,context,pool,future):
    ns=c['namespace'];sid=context['structure_id'];p=context['prefix']
    return {'trial_id':context['context_id']+f'-{pool}-{future:04d}',
      'history_id':context['context_id'],'structure_id':str(sid),'structure_seed':sid,
      'split':context['split'],'regime':context['regime'],'future_id':f'{pool}-{future:04d}',
      'trial_seed':rng_seed(ns,'history-events',sid,p),
      'phase_seed':rng_seed(ns,'history-phase',sid,p),
      'envelope_seed':rng_seed(ns,'history-envelope',sid,p),
      'future_seed':rng_seed(ns,'future-events',sid,p,pool,future),
      'future_phase_seed':rng_seed(ns,'future-phase',sid,p,pool,future),
      'future_burst_seed':rng_seed(ns,'future-envelope',sid,p,pool,future),
      'future_start_ms':cutoff_ms(c)}


def generate(c,root,resume=False,*,splits=None):
    root=Path(root);cfg=adapter_config(c)
    if c['backend']=='neuron':from ..simulator import generate_trial
    else:from ..analytical import generate_trial
    stage='generate-'+('-'.join(sorted(splits)) if splits else 'all')
    old=resume_stage(root,stage,resume)
    if old:return old['details']
    marker=root/'native/manifest.json'
    rows=read_json(marker)['trials'] if marker.exists() else []
    if rows and not resume:raise FileExistsError('Native shards exist; use --resume')
    for r in rows:verify_record(root,r)
    known={r['trial_id']:r for r in rows}
    contexts=context_specs(c)
    if splits:contexts=[x for x in contexts if x['split'] in splits]
    audits=[];started=time.perf_counter();generated=0;resumed=0;replay_check=None
    for context in contexts:
        reference=None;audit=None
        for pool,n in [('score',context['score_futures']),('reference',context['reference_futures'])]:
            for f in range(n):
                spec=trial_spec(c,context,pool,f);tid=spec['trial_id']
                if tid in known:
                    trial=_read_fork(root/known[tid]['path']);resumed+=1
                else:
                    t0=time.perf_counter()
                    try:trial=generate_trial(cfg,spec)
                    except Exception as exc:
                        atomic_json(root/'failures'/f'{tid}.json',{'status':'failed','spec':spec,'error':repr(exc),'schema_version':SCHEMA_VERSION})
                        raise
                    path=root/'native'/context['split']/context['context_id']/f'{pool}-{f:04d}.h5'
                    write_trial(path,trial,{'schema_version':SCHEMA_VERSION,'spec':spec,'metadata':trial['metadata']})
                    row={**artifact_record(root,path),'trial_id':tid,'context_id':context['context_id'],
                      'structure_id':context['structure_id'],'split':context['split'],'pool':pool,'future_index':f,
                      'spec':spec,'metadata':trial['metadata'],'generation_seconds':time.perf_counter()-t0}
                    rows.append(row);known[tid]=row;generated+=1
                    atomic_json(marker,{'schema_version':SCHEMA_VERSION,'config_hash':canonical_hash(c),'source_hash':source_hash(),'trials':rows})
                if reference is None:reference=trial
                else:audit=validate_fork_prefixes([reference,trial],cutoff_ms(c))
                # One genuine future-input replay is a numerical control, not a predictor.
                if replay_check is None and c['claim_status']=='engineering':
                    t0=time.perf_counter();replayed=generate_trial(cfg,spec)
                    validate_fork_prefixes([trial,replayed],cutoff_ms(c))
                    identical=all(np.array_equal(trial[k],replayed[k]) for k in ('time_ms','eeg_v','dipole_na_um'))
                    if not identical:raise RuntimeError('Future-input numerical replay failed')
                    replay_check={'trial_id':tid,'full_trace_identical':True,'seconds':time.perf_counter()-t0,
                      'interpretation':'privileged identical future-input replay; not realistic EEG prediction'}
                print(f'H1 native {tid} ({generated} generated, {resumed} resumed)',flush=True)
        audits.append({'context_id':context['context_id'],'split':context['split'],
          'audit':audit,'future_pools_disjoint':True,'fork_audit_status':'passed' if audit else 'single training/development outcome; no independent-future audit'})
        atomic_json(root/'native/prefix_validation.json',{'schema_version':SCHEMA_VERSION,'contexts':audits})
    relevant=[r for r in rows if not splits or r['split'] in splits]
    elapsed=sum(r['generation_seconds'] for r in relevant)
    result={'schema_version':SCHEMA_VERSION,'backend':c['backend'],'claim_status':c['claim_status'],
      'population_e':c['simulation']['population_e'],'population_i':c['simulation']['population_i'],
      'contexts':len(contexts),'continuations':len(relevant),'generated_this_call':generated,'resumed_this_call':resumed,
      'summed_generation_seconds':elapsed,'mean_continuation_seconds':elapsed/len(relevant) if relevant else None,
      'command_wall_seconds':time.perf_counter()-started,'workers':1,'cpu_concurrency_assumption':'one serial NEURON process; no MPI speedup assumed',
      'native_bytes':sum(r['bytes'] for r in relevant),'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      'cutoff_ms':cutoff_ms(c),'episode_stop_ms':stop_ms(c),'physical_future_stop_ms':max(j['lead_ms']+j['duration_ms'] for j in c['jobs'])+cutoff_ms(c),
      'early_stop':'full prefix replay stops at final required future sample, rounded up only to phase grid; no state caching',
      'future_input_replay_control':replay_check,'complete':len(relevant)==sum(x['score_futures']+x['reference_futures'] for x in contexts)}
    atomic_json(root/'timing_audit.json',result)
    own_audit=root/'native'/f'prefix_validation-{stage}.json'
    own_timing=root/'native'/f'timing-{stage}.json'
    atomic_json(own_audit,{'schema_version':SCHEMA_VERSION,'contexts':audits})
    atomic_json(own_timing,result)
    stage_complete(root,stage,[root/r['path'] for r in relevant]+[own_audit,own_timing],result)
    return result


def target_spec(c,j,*,normalization_id='physical-unfitted',population_id='panel',context_id='panel',split='evaluation'):
    from .acquisition import metadata
    unit='ln(V^2)' if j['operator'].startswith('log_') else ('V amplitude magnitude' if j['operator']=='magnitude_spectrum' else 'V')
    return TargetSpec(operator=j['operator'],cutoff_ms=cutoff_ms(c),history_s=j['history_s'],lead_ms=j['lead_ms'],duration_ms=j['duration_ms'],
      output_grid_hz=j['output_grid_hz'],native_dt_ms=c['simulation']['native_dt_ms'],units=unit,
      input_acquisition=metadata(j['input_rate_hz'],c['simulation']['native_dt_ms'],c['acquisition']['passband_hz'],c['acquisition']['fir_taps']),
      normalization_id=normalization_id,context_id=context_id,population_id=population_id,split_id=split,
      nuisance={'input_snr_db':j['snr_db'],'acquisition_bits':j['acquisition_bits'],'future_target':'clean'},code_version=source_hash())


def precut_normalization_target(trial,spec):
    # Apply target operator to a disjoint TRAINING PRE-CUTOFF block, never to a scored future.
    t=trial['time_ms'];x=trial['eeg_v'].reshape(-1)
    keep=(t>spec.cutoff_ms-spec.duration_ms+1e-8)&(t<=spec.cutoff_ms+1e-8)
    block=x[keep]
    if len(block)!=round(spec.duration_ms/spec.native_dt_ms):raise ValueError('Insufficient predecision normalizer support')
    if spec.operator=='native_waveform':return block
    width=round(1000/spec.output_grid_hz/spec.native_dt_ms)
    return operator_from_projected(block.reshape(-1,width).mean(1),spec)


def task_labels(trial,c,context,carrier_map):
    cutoff=cutoff_ms(c);t=trial['time_ms'];x=trial['eeg_v'].reshape(-1)
    keep=(t>cutoff-1000*max(j['history_s'] for j in c['jobs']))&(t<=cutoff)
    y=x[keep];times=(t[keep]-cutoff)/1000
    # Fixed causal, clean-history task. It is a measurement task, not a novel hidden mechanism.
    fs=1000/c['simulation']['native_dt_ms']
    f,p=signal.periodogram(y,fs=fs,detrend='constant',window='hann')
    band=(f>=4)&(f<=30)
    peak=float(f[band][np.argmax(p[band])]) if band.any() else 10.
    z=np.sum((y-y.mean())*np.exp(-2j*np.pi*peak*times))
    phase=np.angle(z)
    regime=context['regime'];freq=float(regime.get('frequency_hz',0))
    return {'task_rhythm':int(regime.get('modulation_depth',0)>0),
      'task_carrier':carrier_map.get(freq,-1) if regime.get('modulation_depth',0)>0 else -1,
      'task_phase_sin':float(np.sin(phase)),'task_phase_cos':float(np.cos(phase)),
      'task_log_power':float(np.log(max(np.mean(y*y),1e-40))),
      'task_diffusion':float(regime.get('diffusion',0)),'task_kinetics':float(regime.get('kinetics_scale',1.))}


def derive(c,root,resume=False):
    root=Path(root);old=resume_stage(root,'derive',resume)
    if old:return old['details']
    raw=read_json(root/'native/manifest.json')
    if raw['schema_version']!=SCHEMA_VERSION or raw['config_hash']!=canonical_hash(c) or raw['source_hash']!=source_hash():raise ValueError('Incompatible native schema/config/source')
    by_context={}
    for row in raw['trials']:
        verify_record(root,row);by_context.setdefault(row['context_id'],[]).append(row)
    contexts=context_specs(c)
    for x in contexts:
        if len(by_context.get(x['context_id'],[]))!=x['score_futures']+x['reference_futures']:raise ValueError('Generate every required split/pool before derivation')
    if not any(x['split']=='train' for x in contexts):raise ValueError('Timing gate has no training data; its target normalization/fitting is intentionally unavailable')
    carrier_map={freq:i for i,freq in enumerate(sorted({float(r.get('frequency_hz',0)) for r in c['data']['regimes'] if r.get('modulation_depth',0)>0}))}
    jobs=[];outputs=[]
    for job in c['jobs']:
        spec=target_spec(c,job);records=[]
        for context in contexts:
            rows=sorted(by_context[context['context_id']],key=lambda x:(x['pool'],x['future_index']))
            first=_read_fork(root/rows[0]['path'])
            history,relative_t,filter_meta=causal_history(first['eeg_v'],first['time_ms'],cutoff_ms=cutoff_ms(c),history_s=job['history_s'],rate_hz=job['input_rate_hz'],passband=c['acquisition']['passband_hz'],taps=c['acquisition']['fir_taps'])
            pools={'score':[],'reference':[]};ids={'score':[],'reference':[]}
            for row in rows:
                trial=first if row is rows[0] else _read_fork(root/row['path'])
                # Repeat the inherited prefix/state check while assembling the scored targets.
                if row is not rows[0]:validate_fork_prefixes([first,trial],cutoff_ms(c))
                pools[row['pool']].append(extract_target(trial['eeg_v'],trial['time_ms'],spec))
                ids[row['pool']].append(row['trial_id'])
            records.append({'context':context,'history':history,'pools':pools,'ids':ids,
                'normalizer_target':precut_normalization_target(first,spec),
                'tasks':task_labels(first,c,context,carrier_map)})
        training=[r for r in records if r['context']['split']=='train']
        tx=np.stack([r['history'] for r in training]);norm_y=np.stack([r['normalizer_target'] for r in training])
        noise_power=float(np.var(tx))
        if job['snr_db'] is not None:
            for r in records:
                rng=np.random.default_rng(rng_seed(c['namespace'],'observation-noise',r['context']['context_id'],job['input_rate_hz'],job['history_s']))
                r['history']=r['history']+rng.normal(size=len(r['history']))*np.sqrt(noise_power*10**(-job['snr_db']/10))
        acquisition_quantizer=None
        if job['acquisition_bits'] is not None:
            acquisition_quantizer=fit_range(np.concatenate([r['history'] for r in training]),job['acquisition_bits'])
            for r in records:r['history'],r['acquisition_codes'],r['clip_fraction']=quantize(r['history'],acquisition_quantizer)
        tx=np.stack([r['history'] for r in training]);x_offset=float(tx.mean());x_scale=float(tx.std())
        y_offset=norm_y.mean(0);y_scale=float(np.sqrt(np.mean((norm_y-y_offset)**2)))
        if min(x_scale,y_scale)<=0 or not np.isfinite([x_scale,y_scale]).all():raise ValueError('Zero/nonfinite training normalization; do not replace with an arbitrary floor')
        normalization={'x_offset':x_offset,'x_scale':x_scale,'y_offset':y_offset.tolist(),'y_scale':y_scale,
          'fit_split':'train','fit_contexts':[r['context']['context_id'] for r in training],
          'target_fit_support':'clean PRE-CUTOFF blocks only; no scored future enters normalization',
          'input_noise_reference_power_v2':noise_power,'acquisition_quantizer':acquisition_quantizer}
        normalization['id']=canonical_hash(normalization)
        population_id=canonical_hash([x['context_id'] for x in contexts if x['split']=='evaluation'])
        spec=target_spec(c,job,normalization_id=normalization['id'],population_id=population_id)
        directory=root/'panels'/job['id'];files={}
        for split in ('train','development','evaluation'):
            selected=[r for r in records if r['context']['split']==split]
            if not selected:raise ValueError(f'No {split} data for a trainable panel')
            n=len(selected);m=max(len(r['pools']['score']) for r in selected);nr=max(len(r['pools']['reference']) for r in selected);d=spec.dimension
            yp=np.zeros((n,m,d));rp=np.zeros((n,nr,d));mask=np.zeros((n,m),bool);rmask=np.zeros((n,nr),bool)
            future_ids=np.full((n,m),'',dtype='U128');ref_ids=np.full((n,nr),'',dtype='U128')
            for k,r in enumerate(selected):
                a=len(r['pools']['score']);b=len(r['pools']['reference']);yp[k,:a]=r['pools']['score'];mask[k,:a]=True;future_ids[k,:a]=r['ids']['score']
                if b:rp[k,:b]=r['pools']['reference'];rmask[k,:b]=True;ref_ids[k,:b]=r['ids']['reference']
            xp=np.stack([r['history'] for r in selected])
            data={'X':((xp-x_offset)/x_scale).astype(np.float32),'X_physical':xp,
              'Y':(yp-y_offset)/y_scale,'Y_physical':yp,
              'reference_Y':(rp-y_offset)/y_scale,'reference_Y_physical':rp,
              'future_mask':mask,'reference_mask':rmask,'future_id':future_ids,'reference_future_id':ref_ids,
              'context_id':np.array([r['context']['context_id'] for r in selected]),
              'structure_id':np.array([r['context']['structure_id'] for r in selected],np.int64),
              'regime':np.array([r['context']['regime']['name'] for r in selected]),'input_time_ms':relative_t,
              'target_time_ms':(np.arange(1,d+1)*(spec.native_dt_ms if spec.operator=='native_waveform' else 1000/spec.output_grid_hz)+spec.lead_ms) if spec.operator in ('native_waveform','bin_average_waveform') else np.array([],dtype=float),
              'acquisition_clipping_fraction':np.array([r.get('clip_fraction',0.) for r in selected])}
            for task in selected[0]['tasks']:data[task]=np.array([r['tasks'][task] for r in selected])
            if acquisition_quantizer:data['acquisition_codes']=np.stack([r['acquisition_codes'] for r in selected])
            file=directory/f'{split}.npz';atomic_npz(file,**data);files[split]=artifact_record(root,file);outputs.append(file)
        manifest={'schema_version':SCHEMA_VERSION,'job_id':job['id'],'target_spec':spec.to_dict(),
          'target_fingerprint':spec.fingerprint,'comparison_fingerprint':spec.fingerprint,'physical_target_fingerprint':spec.physical_fingerprint,
          'input':{'sample_rate_hz':job['input_rate_hz'],'history_s':job['history_s'],'acquisition_bits':job['acquisition_bits'],'snr_db':job['snr_db'],**filter_meta},
          'normalization':normalization,'operator':job['operator'],'output_grid_hz':job['output_grid_hz'],'lead_ms':job['lead_ms'],'duration_ms':job['duration_ms'],
          'files':files,'population_id':population_id,'context_weighting':'equal contexts inside each circuit, then equal circuits; future counts do not change context weights',
          'task_definitions':{'rhythm':'generator modulation present','carrier':'generator frequency labels among rhythmic contexts; homogeneous=-1',
            'phase_sin_cos':'phase of clean PRE-CUTOFF EEG at its dominant4-30Hz Fourier bin relative to cutoff, using maximum declared history; measurement diagnostic, not instantaneous latent phase',
            'log_power':'log mean-square clean PRE-CUTOFF EEG over maximum declared history',
            'absent_factors':'constant diffusion/kinetics labels cannot support mechanism discrimination'},
          'source_hash':source_hash(),'config_hash':canonical_hash(c)}
        atomic_json(directory/'manifest.json',manifest);outputs.append(directory/'manifest.json')
        jobs.append({'job_id':job['id'],**artifact_record(root,directory/'manifest.json'),'actual_target_dimension':spec.dimension,
          'history_samples':round(job['history_s']*job['input_rate_hz']),'target_fingerprint':spec.fingerprint})
    atomic_json(root/'panels/manifest.json',{'schema_version':SCHEMA_VERSION,'jobs':jobs,'completed_jobs':len(jobs),'planned_jobs':len(c['jobs'])})
    outputs.append(root/'panels/manifest.json')
    result={'completed_jobs':len(jobs),'planned_jobs':len(c['jobs']),'contexts':len(contexts),'training_only_normalization':True}
    stage_complete(root,'derive',outputs,result)
    return result


def list_jobs(root):
    manifest=read_json(Path(root)/'panels/manifest.json')
    if manifest['schema_version']!=SCHEMA_VERSION:raise ValueError('Stale panel schema')
    return [row['job_id'] for row in manifest['jobs']]


def load_panel(root,job_id,verify=True):
    root=Path(root);index=read_json(root/'panels/manifest.json')
    if index['schema_version']!=SCHEMA_VERSION:raise ValueError('Stale panel schema')
    row=next(x for x in index['jobs'] if x['job_id']==job_id)
    p=verify_record(root,row) if verify else root/row['path']
    manifest=read_json(p)
    if manifest['schema_version']!=SCHEMA_VERSION or TargetSpec.from_dict(manifest['target_spec']).fingerprint!=manifest['target_fingerprint']:raise ValueError('Target fingerprint mismatch')
    result={'manifest':manifest};seen=set();seen_contexts=set();seen_futures=set()
    for split,record in manifest['files'].items():
        p=verify_record(root,record) if verify else root/record['path']
        with np.load(p,allow_pickle=False) as f:result[split]={k:f[k] for k in f.files}
        data=result[split];ids=set(map(int,data['structure_id']))
        expected=TargetSpec.from_dict(manifest['target_spec']).dimension
        if data['Y'].shape[-1]!=expected or data['reference_Y'].shape[-1]!=expected:raise ValueError('Target array dimension differs from TargetSpec')
        if data['X'].shape[1]!=round(manifest['input']['history_s']*manifest['input']['sample_rate_hz']):raise ValueError('Input history dimension differs from acquisition metadata')
        context_ids=set(data['context_id'])
        if len(context_ids)!=len(data['context_id']) or context_ids&seen_contexts:raise ValueError('Duplicate/leaked context IDs')
        seen_contexts|=context_ids
        if ids&seen:raise ValueError('Circuit/variant split leakage')
        seen|=ids
        primary=set(data['future_id'][data['future_mask']]);reference=set(data['reference_future_id'][data['reference_mask']])
        if primary&reference:raise ValueError('Reference/scoring future leakage')
        active_count=int(data['future_mask'].sum()+data['reference_mask'].sum())
        if len(primary|reference)!=active_count or (primary|reference)&seen_futures:raise ValueError('Duplicate/leaked future IDs')
        seen_futures|=primary|reference
    return result
