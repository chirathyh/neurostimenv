"""Exact saved-array reuse or verified native derivation, with distinct analysis hashes."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
from .evidence import EvidenceSource
from .config import ANCHORS,SCHEMA,contexts
from .artifacts import atomic_json,atomic_npz,artifact_record,verify_record,canonical_hash,module_hash,resume_record
from ..v4.derivation import _native_records,_normalization,load_panel as verified_load_panel
from ..v4.targets import TargetSpec,axes
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
 log=target.startswith('T4_log_');band=target.split('energy_',1)[1].split('_l10',1)[0];ts=energy_spec(band,log)
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


def derive(c,root,source=None,resume=False):
 root=Path(root);path=root/'derived/manifest.json';owned=source is None or isinstance(source,(str,Path))
 native=source is None
 src=NativeComposite(root) if native else EvidenceSource(source) if isinstance(source,(str,Path)) else source
 try:
  if native:_validate_native_allocation(c,list(src.rows.values()))
  identity=src.provenance();sig=canonical_hash({'config':c,'inputs':identity['content_id'],'analysis':module_hash('derivation','targets')})
  old=None
  if path.exists() and native and resume:
   prior=json.loads(path.read_text())
   if prior.get('signature')!=sig:
    current={r['trial_id']:r['sha256'] for r in src.rows.values()};previous=prior.get('native_trial_hashes')
    if not previous or not set(previous)<set(current) or any(current[k]!=v for k,v in previous.items()) or prior.get('analysis_dependency_hash')!=module_hash('derivation','targets') or prior.get('config_hash')!=canonical_hash(c):
     raise ValueError('Native derive may only append registered splits under identical analysis/config and unchanged original native bytes')
   else:old=resume_record(path,sig,resume)
  else:old=resume_record(path,sig,resume)
  if old:
   for j in old['jobs']:verified_load_panel(root,j['id'])
   return old
  qualify(root);cache={};jobs=[]
  if native:
   specs=[TargetSpec(family='T0',lead_ms=2.,duration_ms=32.),TargetSpec(),TargetSpec(family='T3',duration_ms=1000.)]
   records,audits,input_meta,unavailable,target_unavailable=_native_records(src,list(src.rows.values()),specs,c['acquisition_ids'],4.,{},6000.)
   if unavailable or target_unavailable:raise ValueError('Native derivation qualification failed '+str((unavailable,target_unavailable)))
   atomic_json(root/'derived/native_audits.json',audits)
   for a in c['acquisition_ids']:
    for ts in specs:
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
     jid=a+'__'+ts.id;manifest={'status':'complete','job_id':jid,'input':{**input_meta[a],'history_s':4},'normalization':norm,'target_spec':ts.to_dict(),'target_family':ts.family,'physical_target_fingerprint':ts.fingerprint,'risk_contract':contract,'target_fingerprint':canonical_hash(contract),'comparison_fingerprint':canonical_hash(contract),'original_generation_provenance':identity}
     cache[jid]=(panels,manifest)
  else:
   for a in c['acquisition_ids']:
    for t in ANCHORS:cache[a+'__'+t]=_saved_anchor(src,a,t)
  for a in c['acquisition_ids']:
   for t in ANCHORS:
    panels,m=cache[a+'__'+t]
    # A fixed target must really be identical across acquisitions.
    if a!=c['acquisition_ids'][0]:
     other=cache[c['acquisition_ids'][0]+'__'+t][0]
     for split,p in panels.items():
      for key in ['Y_physical','future_mask','future_id','context_id','reference_Y_physical']:
       if not np.array_equal(p[key],other[split][key]):raise ValueError('Cross-acquisition target/population mismatch '+key)
    m.update(source_content_id=identity['content_id'],source_provenance=identity,claim_status=c['claim_status'])
    jobs.append(_save_job(root,a+'__'+t,panels,m))
   p,m=cache[a+'__T3_l10_d1000']
   for jid in c['selected_job_ids']:
    acq,target=jid.split('__',1)
    if acq==a and target.startswith('T4_'):
     ep,em=_energy_panel(root,a,target,p,m);jobs.append(_save_job(root,jid,ep,em))
  result={'status':'complete','signature':sig,'jobs':jobs,'completed_jobs':len(jobs),'source':identity,'claim_status':c['claim_status'],'native_rederived':native,'all_fixed_target_acquisition_arrays_identical':True,'analysis_dependency_hash':module_hash('derivation','targets'),'config_hash':canonical_hash(c),'native_trial_hashes':{r['trial_id']:r['sha256'] for r in src.rows.values()} if native else None}
  atomic_json(path,result)
  for j in jobs:verified_load_panel(root,j['id'])
  return result
 finally:
  if owned:src.close()

load_panel=verified_load_panel
