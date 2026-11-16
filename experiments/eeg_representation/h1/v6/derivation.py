"""Exact saved-array reuse or verified native derivation, with distinct analysis hashes."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
from .evidence import EvidenceSource
from .config import ANCHORS,SCHEMA,contexts,target_ids
from .artifacts import atomic_json,atomic_npz,artifact_record,verify_record,canonical_hash,module_hash,resume_record
from ..v4.derivation import _native_records,_normalization,load_panel as verified_load_panel
from .targets import TargetSpec,axes,target_spec,timing_metadata,parse_target_id
from ..v4.measurement import qualify
from .targets import energy_from_coefficients,spec as energy_spec

class NativeComposite:
 def __init__(self,root):
  self.path=Path(root);self.manifest=json.loads((self.path/'generation/native/manifest.json').read_text());self.sources={};self.rows={r['path']:r for r in self.manifest['trials']}
  if len(self.rows)!=len(self.manifest['trials']):raise ValueError('Duplicate native logical paths')
 def open_binary(self,path):
  row=self.rows[path];directory=row.get('source_root',str(self.path/'generation'))
  if directory not in self.sources:self.sources[directory]=EvidenceSource(directory)
  return self.sources[directory].open_binary(path)
 def provenance(self):
  return {'content_id':canonical_hash(self.manifest),'manifest_hash':canonical_hash(self.manifest),'generation_origins':sorted(set(str(r.get('source_root','new_generation')) for r in self.rows.values())),'policy':'Imported rows retain their original source path/hash/metadata; no relabeling.'}
 def close(self):
  for s in self.sources.values():s.close()

def _save_job(root,jid,panels,manifest):
 root=Path(root);out=root/'derived'/jid;manifest=deepcopy(manifest);manifest['files']={}
 for split,panel in panels.items():
  path=out/(split+'.npz')
  if path.exists():
   with np.load(path,allow_pickle=False) as previous:
    if set(previous.files)!=set(panel) or any(not np.array_equal(previous[k],panel[k],equal_nan=True) if np.issubdtype(panel[k].dtype,np.number) else not np.array_equal(previous[k],panel[k]) for k in panel):
     raise ValueError('Previously derived split changed; never overwrite frozen train/dev '+str(path))
  else:atomic_npz(path,**panel)
  manifest['files'][split]=artifact_record(root,path)
 manifest['schema_version']=SCHEMA;manifest['analysis_dependency_hash']=module_hash('derivation','targets');atomic_json(out/'manifest.json',manifest)
 return {'id':jid,'job_id':jid,'status':'complete',**artifact_record(root,out/'manifest.json')}

def _saved_anchor(source,acquisition,target):
 jid=acquisition+'__'+target;name='derived/'+jid+'/manifest.json'
 if not source.exists(name):raise ValueError('Saved exact acquisition/target unavailable: '+jid+'; native source required')
 m=source.read_json(name);p={s:source.read_npz(r['path']) for s,r in m['files'].items()}
 for s,row in p.items():
  if row['X_physical'].shape[1]!=2000:raise ValueError('Full4s history absent')
  n=m['normalization'];expected=(row['Y_physical']-np.asarray(n['y_offset']))/n['y_scale']
  if not np.allclose(row['Y'],expected,rtol=1e-12,atol=1e-12):raise ValueError('Saved target normalization mismatch')
  # Preserve physical data exactly. Float64 normalization is intentional arithmetic,
  # not an acquisition change; original float32 X remains in original evidence.
  row['X']=(row['X_physical']-n['x_offset'])/n['x_scale']
  row['task_four_class']=np.where(row['task_rhythm']==0,0,row['task_carrier']+1)
 return p,m

def _energy_panel(root,acquisition,target,panels,manifest):
 family,h,w=parse_target_id(target);log=family.startswith('T4_log_');band=family.split('energy_',1)[1];ts=energy_spec(band,log,lead_ms=h)
 out=deepcopy(panels);floors={}
 for split,p in out.items():
  floors[split]=0
  for physical,normed,mask in [('Y_physical','Y','future_mask'),('reference_Y_physical','reference_Y','reference_mask')]:
   c=p[physical];e=energy_from_coefficients(c,p['target_frequency_hz'],p['target_component'],band,log=log)
   if log:
    raw=energy_from_coefficients(c,p['target_frequency_hz'],p['target_component'],band)
    floors[split]+=int(np.sum((raw[...,0]<1e-40)&p[mask]))
   e[~p[mask]]=0.;p[physical]=e
  p['target_time_ms']=np.array([],dtype=float);p.pop('target_frequency_hz',None);p.pop('target_component',None)
 train=out['train'];y=np.array([v[mask].mean(0) for v,mask in zip(train['Y_physical'],train['future_mask'])]);ids=train['structure_id'];uids=np.unique(ids)
 w=np.array([1/(len(uids)*np.sum(ids==i)) for i in ids]);offset=np.sum(y*w[:,None],axis=0);scale=float(np.sqrt(np.sum(w*np.mean((y-offset)**2,axis=1))))
 if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate training energy normalization')
 m=deepcopy(manifest);n={**m['normalization'],'y_offset':offset.tolist(),'y_scale':scale,'fit_split':'train','energy_normalization':'nonlinear target per future, then training-only equal-circuit context means'};n.pop('id',None);n['id']=canonical_hash(n)
 for p in out.values():
  p['Y']=(p['Y_physical']-offset)/scale;p['reference_Y']=(p['reference_Y_physical']-offset)/scale
 m.update(job_id=acquisition+'__'+target,target_spec=ts,target_family=ts['family'],normalization=n,physical_target_fingerprint=ts['operator_hash'],log_floor_activations=floors)
 m['risk_contract']={**m['risk_contract'],'target_physical_hash':ts['operator_hash'],'normalization_id':n['id'],'units':ts['units'],'dimension':1,'support':ts['physical_support_ms']}
 m['target_fingerprint']=m['comparison_fingerprint']=canonical_hash(m['risk_contract']);return out,m

def _validate_native_allocation(c,rows):
 planned={r['context_id']:r for r in contexts(c)};seen=set();splits=set();seeds={}
 for row in rows:
  cid=row['context_id'];ctx=planned.get(cid)
  if ctx is None:raise ValueError('Unregistered native context '+cid)
  if row['split']!=ctx['split'] or int(row['structure_id'])!=ctx['structure_id']:raise ValueError('Native split/circuit differs from frozen allocation')
  pool=row['pool'];future=int(row['future_index'])
  if pool not in ('score','reference') or not 0<=future<ctx[pool+'_futures']:raise ValueError('Native future outside registered pool')
  if row['trial_id']!=cid+f'-{pool}-{future:04d}' or row['trial_id'] in seen:raise ValueError('Duplicate/misidentified native future')
  normalized=lambda r:{'diffusion':0.,'kinetics_scale':1.,'shared_fraction':1.,'burst':False,**r}
  if normalized(row['spec']['regime'])!=normalized(ctx['regime']):raise ValueError('Native regime differs from registered acquisition population')
  if float(row['spec']['future_start_ms'])!=float(c['cutoff_ms']):raise ValueError('Native cutoff differs from risk contract')
  seed=row['spec']['future_seed']
  if seed in seeds.setdefault(cid,set()):raise ValueError('Duplicate RNG realization within native context future pools')
  seeds[cid].add(seed);seen.add(row['trial_id']);splits.add(row['split'])
 expected={r['context_id']+f'-{pool}-{future:04d}' for r in planned.values() if r['split'] in splits for pool in ('score','reference') for future in range(r[pool+'_futures'])}
 if seen!=expected:raise ValueError('Derivation requires complete registered splits; incomplete native allocation')
 return {'present_splits':sorted(splits),'complete_present_splits':True,'episodes':len(seen),'planned_episodes':sum(r['score_futures']+r['reference_futures'] for r in planned.values())}


def _open_source(value,root):
 if value is None:return NativeComposite(root),True,True
 if not isinstance(value,(str,Path)):return value,bool(getattr(value,'rows',None)),False
 path=Path(value)
 if path.is_dir() and (path/'generation/native/manifest.json').is_file():
  composite=NativeComposite(path)
  present=bool(composite.rows) and all((Path(r.get('source_root',str(path/'generation')))/r['path']).is_file() for r in composite.rows.values())
  if present:return composite,True,True
  composite.close()
 src=EvidenceSource(path)
 # Standard evidence exports retain the original native manifest but omit H5.
 # A provenance ledger is not a native recording and must not trigger a read.
 native_present=bool(src.records) and all(src.exists(r['path']) for r in src.records.values())
 return src,native_present,True


def _retarget_saved(panels,manifest,ts):
 """Recompute training normalization after an exact physical-bin restriction."""
 out=deepcopy(panels);m=deepcopy(manifest);train=out['train'];ids=train['structure_id'];unique=np.unique(ids)
 y=np.stack([v[mask].mean(0) for v,mask in zip(train['Y_physical'],train['future_mask'])])
 weights=np.asarray([1/(len(unique)*np.sum(ids==i)) for i in ids]);offset=np.sum(weights[:,None]*y,axis=0)
 scale=float(np.sqrt(np.sum(weights*np.mean((y-offset)**2,axis=1))))
 if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate saved-target training normalization')
 n={**m['normalization'],'y_offset':offset.tolist(),'y_scale':scale};n.pop('id',None);n['id']=canonical_hash(n)
 for p in out.values():
  p['Y']=(p['Y_physical']-offset)/scale;p['reference_Y']=(p['reference_Y_physical']-offset)/scale;p.update(axes(ts))
 m.update(target_spec=ts.to_dict(),target_family=ts.family,normalization=n,physical_target_fingerprint=ts.fingerprint)
 m['risk_contract']={**m['risk_contract'],'target_physical_hash':ts.fingerprint,'normalization_id':n['id'],'units':ts.units,'dimension':ts.dimension,'support':ts.to_dict()['physical_support_ms']}
 m['target_fingerprint']=m['comparison_fingerprint']=canonical_hash(m['risk_contract'])
 return out,m


def _saved_target(source,acquisition,target):
 if source.exists('derived/'+acquisition+'__'+target+'/manifest.json'):return _saved_anchor(source,acquisition,target)
 ts=target_spec(target)
 if not isinstance(ts,TargetSpec) or ts.family!='T0':raise ValueError('Exact saved target absent; native future samples required for '+target)
 candidates=[]
 for name in source.list_names():
  prefix='derived/'+acquisition+'__T0_'
  if name.startswith(prefix) and name.endswith('/manifest.json'):
   tid=name.split('__',1)[1].split('/')[0];p,m=_saved_anchor(source,acquisition,tid);candidates.append((p,m))
 wanted=axes(ts)['target_time_ms'];selected=[]
 for time in wanted:
  matches=[]
  for p,m in candidates:
   inds=np.flatnonzero(p['train']['target_time_ms']==time)
   if len(inds):matches.append((p,m,int(inds[0])))
  if not matches:raise ValueError(f'Exact native bin ending+{time:g}ms is absent from saved arrays; native recordings required for {target}')
  for candidate,_,index in matches[1:]:
   reference,_,reference_index=matches[0]
   if set(reference)!=set(candidate):raise ValueError('Saved overlap source populations differ')
   for split in reference:
    for key in ('Y_physical','reference_Y_physical'):
     if not np.array_equal(reference[split][key][...,reference_index],candidate[split][key][...,index]):raise ValueError('Conflicting overlapping physical target bins')
  selected.append(matches[0])
 if not selected:raise ValueError('Empty exact-bin reconstruction')
 panels=deepcopy(selected[0][0]);m=deepcopy(selected[0][1]);keys=('future_mask','reference_mask','future_id','reference_future_id','context_id','structure_id','regime','input_time_ms','X_physical')
 for source_panels,_,_ in selected:
  if set(source_panels)!=set(panels):raise ValueError('Saved bin source populations differ')
  for split,p in panels.items():
   for key in keys:
    if not np.array_equal(p[key],source_panels[split][key]):raise ValueError('Saved bin source identities/pools differ')
 for split,p in panels.items():
  for key in ('Y_physical','reference_Y_physical'):
   p[key]=np.stack([rows[split][key][...,i] for rows,_,i in selected],axis=-1)
 out,m=_retarget_saved(panels,m,ts);m['exact_saved_bin_reuse']={'source_targets':sorted(set(x[1]['target_spec'].get('id',x[1]['job_id'].split('__',1)[1]) for x in selected)),'target_bin_end_offsets_ms':wanted.tolist()}
 return out,m


def _add_timing(m,target):
 m=deepcopy(m);timing=timing_metadata(target)
 m['target_timing']=timing;m['target_spec']={**m['target_spec'],**{k:v for k,v in timing.items() if k!='physical_support_ms'}}
 m['input']['sample_availability']={'relative_ms':[-3998.,0.],'sampling_interval_ms':2.,'last_available_ms':6000.,'backdated':False,'operator_phase_and_memory':'See acquisition operator metadata; unchanged by target gap.'}
 n=m['normalization'];m['target_normalization']={'y_offset':n['y_offset'],'y_scale':n['y_scale'],'fit_split':'train','fit_contexts':n.get('fit_contexts',[]),'shared_across_acquisitions':True}
 m['target_normalization']['id']=canonical_hash(m['target_normalization'])
 return m


def derive(c,root,source=None,resume=False,native_source=None):
 """Derive feasible exact targets; absent native supports remain unavailable.

 An optional saved archive plus native_source preserves frozen saved anchors
 while deriving added supports. Source arrays never enter ordinary inputs.
 """
 root=Path(root);path=root/'derived/manifest.json';saved=None;owned_saved=False
 src,native,owned=_open_source(native_source if native_source is not None else source,root)
 if native_source is not None and source is not None:saved=EvidenceSource(source) if isinstance(source,(str,Path)) else source;owned_saved=isinstance(source,(str,Path))
 try:
  rows=list(src.rows.values()) if isinstance(src,NativeComposite) else list(src.records.values()) if native else []
  if native:_validate_native_allocation(c,rows)
  identity=src.provenance()
  if saved is not None:identity={'native':identity,'saved':saved.provenance(),'content_id':canonical_hash({'native':identity['content_id'],'saved':saved.provenance()['content_id']})}
  sig=canonical_hash({'config':c,'inputs':identity['content_id'],'analysis':module_hash('derivation','targets')});old=None
  if path.exists() and native and resume:
   prior=json.loads(path.read_text())
   if prior.get('signature')!=sig:
    current={r['trial_id']:r['sha256'] for r in rows};previous=prior.get('native_trial_hashes')
    if not previous or not set(previous)<set(current) or any(current[k]!=v for k,v in previous.items()) or prior.get('analysis_dependency_hash')!=module_hash('derivation','targets') or prior.get('config_hash')!=canonical_hash(c):
     raise ValueError('Native derive may only append registered splits under identical analysis/config and unchanged original native bytes')
   else:old=resume_record(path,sig,resume)
  else:old=resume_record(path,sig,resume)
  if old:
   for j in old['jobs']:verified_load_panel(root,j['id'])
   return old
  qualify(root);cache={};jobs=[];missing=[]
  wanted=[j.split('__',1)[1] for j in c['selected_job_ids']];bases=list(dict.fromkeys(t for t in wanted if not t.startswith('T4_')))
  for t in wanted:
   if t.startswith('T4_'):
    _,h,_=parse_target_id(t);base=f'T3_l{h:g}_d1000'
    if base not in bases:bases.append(base)
  if native:
   specs=[target_spec(t,c['cutoff_ms']) for t in bases]
   records,audits,input_meta,unavailable,target_unavailable=_native_records(src,rows,specs,c['acquisition_ids'],4.,{},6000.)
   atomic_json(root/'derived/native_audits.json',audits)
   for a in c['acquisition_ids']:
    for ts in specs:
     jid=a+'__'+ts.id
     if a in unavailable or ts.id in target_unavailable:missing.append({'id':jid,'status':'unavailable','reason':unavailable.get(a,target_unavailable.get(ts.id))});continue
     norm=_normalization(records,a,ts);yo=np.asarray(norm['y_offset']);ys=norm['y_scale'];panels={}
     for split in c['data']['structures']:
      sel=[r for r in records if r['context']['split']==split]
      if not sel:continue
      n=len(sel);m=max(len(r['ids']['score']) for r in sel);nr=max(len(r['ids']['reference']) for r in sel)
      yp=np.zeros((n,m,ts.dimension));rp=np.zeros((n,nr,ts.dimension));mask=np.zeros((n,m),bool);rm=np.zeros((n,nr),bool);fi=np.full((n,m),'',dtype='U128');ri=np.full((n,nr),'',dtype='U128')
      for i,r in enumerate(sel):
       ns=len(r['ids']['score']);nref=len(r['ids']['reference']);yp[i,:ns]=r['targets'][ts.id]['score'];rp[i,:nref]=r['targets'][ts.id]['reference'];mask[i,:ns]=1;rm[i,:nref]=1;fi[i,:ns]=r['ids']['score'];ri[i,:nref]=r['ids']['reference']
      x=np.stack([r['histories'][a] for r in sel]);panel={'X_physical':x,'X':(x-norm['x_offset'])/norm['x_scale'],'Y_physical':yp,'Y':(yp-yo)/ys,'reference_Y_physical':rp,'reference_Y':(rp-yo)/ys,'future_mask':mask,'reference_mask':rm,'future_id':fi,'reference_future_id':ri,'context_id':np.array([r['context']['context_id'] for r in sel]),'structure_id':np.array([r['context']['structure_id'] for r in sel]),'regime':np.array([r['context']['regime']['name'] for r in sel]),'input_time_ms':-4000+np.arange(1,2001)*2.,**axes(ts)}
      for k in sel[0]['tasks']:panel[k]=np.array([r['tasks'][k] for r in sel])
      panel['task_four_class']=np.where(panel['task_rhythm']==0,0,panel['task_carrier']+1);panels[split]=panel
     contract={'target_physical_hash':ts.fingerprint,'population_id':canonical_hash({s:sorted(r['context_id'] for r in contexts(c) if r['split']==s) for s in c['data']['structures']}),'acquisition_hash':input_meta[a]['operator_hash'],'normalization_id':norm['id'],'units':ts.units,'dimension':ts.dimension,'support':ts.to_dict()['physical_support_ms']}
     manifest={'status':'complete','job_id':jid,'input':{**input_meta[a],'history_s':4},'normalization':norm,'target_spec':ts.to_dict(),'target_family':ts.family,'physical_target_fingerprint':ts.fingerprint,'risk_contract':contract,'target_fingerprint':canonical_hash(contract),'comparison_fingerprint':canonical_hash(contract),'original_generation_provenance':identity}
     if saved is not None and saved.exists('derived/'+jid+'/manifest.json'):
      frozen,fm=_saved_anchor(saved,a,ts.id)
      if set(frozen)!=set(panels):raise ValueError('Saved/native cohort mismatch; cannot mix imported target panels')
      for split,p in frozen.items():
       for k in ('context_id','future_id','reference_future_id','future_mask','reference_mask','Y_physical','reference_Y_physical'):
        if not np.array_equal(p[k],panels[split][k]):raise ValueError('Native values/IDs failed frozen target reproduction: '+jid+' '+k)
      panels,manifest=frozen,fm
     cache[jid]=(panels,manifest)
  else:
   for a in c['acquisition_ids']:
    for t in bases:
     try:cache[a+'__'+t]=_saved_target(src,a,t)
     except ValueError as exc:
      if 'absent' not in str(exc) and 'required' not in str(exc):raise
      missing.append({'id':a+'__'+t,'status':'unavailable','reason':str(exc)})
  for jid in c['selected_job_ids']:
   a,t=jid.split('__',1)
   if t.startswith('T4_'):
    _,h,_=parse_target_id(t);base=a+f'__T3_l{h:g}_d1000'
    if base not in cache:missing.append({'id':jid,'status':'unavailable','reason':'Native one-second Fourier block unavailable: '+base});continue
    cache[jid]=_energy_panel(root,a,t,*cache[base])
   if jid not in cache:continue
   panels,m=cache[jid];m['job_id']=jid
   first=c['acquisition_ids'][0]+'__'+t
   if first in cache and jid!=first:
    other,om=cache[first]
    for split,p in panels.items():
     for key in ('Y_physical','future_mask','future_id','context_id','reference_Y_physical','reference_mask','reference_future_id'):
      if not np.array_equal(p[key],other[split][key]):raise ValueError('Cross-acquisition target/population mismatch '+key)
    for key in ('y_offset','y_scale'):
     if not np.array_equal(np.asarray(m['normalization'][key]),np.asarray(om['normalization'][key])):raise ValueError('Target training normalization differs across acquisitions '+key)
   m.update(source_content_id=identity['content_id'],source_provenance=identity,claim_status=c['claim_status']);m=_add_timing(m,t)
   jobs.append(_save_job(root,jid,panels,m))
  # Only requested endpoints appear in completion accounting; Fourier helpers
  # requested solely to derive energy are not extra completed study endpoints.
  missing=[r for r in missing if r['id'] in c['selected_job_ids']]
  result={'status':'complete' if not missing else 'partial','signature':sig,'jobs':jobs,'completed_jobs':len(jobs),'requested_jobs':len(c['selected_job_ids']),'unavailable_jobs':missing,'source':identity,'claim_status':c['claim_status'],'native_rederived':native,'all_fixed_target_acquisition_arrays_identical':True,'analysis_dependency_hash':module_hash('derivation','targets'),'config_hash':canonical_hash(c),'native_trial_hashes':{r['trial_id']:r['sha256'] for r in rows} if native else None}
  atomic_json(path,result)
  for j in jobs:verified_load_panel(root,j['id'])
  return result
 finally:
  if owned:src.close()
  if owned_saved:saved.close()

load_panel=verified_load_panel
