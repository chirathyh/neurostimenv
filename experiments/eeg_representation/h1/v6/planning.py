"""Fail-closed execution gates binding current physics and immutable evidence."""
from pathlib import Path
import os,platform,shutil,math
from .artifacts import atomic_json,read_json,file_hash,canonical_hash,environment,verify_record,module_hash
from .config import counts,registry_check,contexts
from .models import settings
from .retention import settings as retention_settings
from ..v4.measurement import metadata


def _read(path):return read_json(path) if Path(path).is_file() else None


def _resources(root,workers,peak_rss,numerical_root=None):
 root=Path(root);memory={}
 try:
  for line in Path('/proc/meminfo').read_text().splitlines():
   name,value=line.split(':',1)
   if name in ('MemTotal','MemAvailable'):memory[name+'_bytes']=int(value.strip().split()[0])*1024
 except OSError:pass
 available=memory.get('MemAvailable_bytes');logical=os.cpu_count();reasons=[]
 # Two times the observed largest worker RSS reserves overhead and variability;
 # processes do not share the conservative per-worker resident allocation.
 bound=2*int(workers)*int(peak_rss or 0)
 if not logical or workers>logical:reasons.append('generation workers exceed observed logical CPUs')
 if not available:reasons.append('available RAM measurement absent')
 elif bound and bound>available:reasons.append('two-times measured worker RSS budget exceeds available RAM')
 folders=[root/'generation',root/'numerical']
 if numerical_root is not None:folders.append(Path(numerical_root)/'numerical')
 files={p.resolve() for folder in folders if folder.exists() for p in folder.rglob('*.h5')}
 # Credit only already-written allocation bytes on this filesystem; imported
 # external files are not silently copied or counted as newly allocated storage.
 device=root.stat().st_dev
 completed=sum(p.stat().st_size for p in files if p.stat().st_dev==device)
 return {'logical_cpus':logical,'workers':workers,'memory':memory,'measured_peak_worker_rss_bytes':int(peak_rss or 0),
         'conservative_worker_ram_budget_bytes':bound,'ram_safety_factor':2.,'completed_native_storage_credit_bytes':completed,
         'free_bytes':shutil.disk_usage(root).free,'blocked_reasons':reasons}


def _fit_hardware(device):
 if not device.startswith('cuda'):return {}
 import torch
 if not torch.cuda.is_available():raise ValueError('Measured CUDA fitting device unavailable')
 index=torch.cuda.current_device() if ':' not in device else int(device.split(':')[1]);p=torch.cuda.get_device_properties(index)
 return {'device_name':p.name,'total_memory_bytes':int(p.total_memory),'compute_capability':[int(p.major),int(p.minor)],'device_index':index}



def _record(root,path):
 p=Path(path);return {'path':str(p.relative_to(root)),'sha256':file_hash(p),'bytes':p.stat().st_size}


def _verify_hashes(root,records):
 for row in records:verify_record(root,row)


def qualify_phase0(c,path):
 p=Path(path);missing=[];records={}
 def bind(q):
  q=Path(q);records[str(q.relative_to(p))]=_record(p,q)
 m=_read(p/'models/manifest.json');d=_read(p/'derived/manifest.json');audit=_read(p/'audit_existing.json')
 measurement=_read(p/'measurement/qualification.json')
 if not audit or audit.get('status')!='complete':missing.append('verified original evidence audit')
 if not measurement or not measurement.get('passed'):missing.append('causal acquisition measurement qualification')
 elif measurement.get('conditions')!={a:metadata(a) for a in ('A0','A1','A2')}:missing.append('measurement design changed after qualification')
 required=set(c['selected_job_ids'])
 if not d or required-{r['id'] for r in d.get('jobs',[]) if r['status']=='complete'}:missing.append('complete fixed-target9cell and18energy/log derivations')
 if not m or not m.get('neural_path_control',{}).get('passed'):missing.append('same full-history neural path positive control')
 if not m or required-{r['id'] for r in m.get('jobs',[]) if r['status']=='complete'}:missing.append('complete bounded full-history qualification grid')
 if d:
  for j in d['jobs']:
   q=verify_record(p,j);bind(q);manifest=read_json(q)
   if manifest.get('analysis_dependency_hash')!=module_hash('derivation','targets'):missing.append('derivation implementation changed '+j['id'])
   for record in manifest['files'].values():bind(verify_record(p,record))
 if m:
  for j in m['jobs']:
   q=p/j['path'];fit=_read(q)
   if not fit or fit.get('full_history_sample_count')!=2000 or fit.get('complete_ridge_candidate_count')!=14 or not fit.get('mean_fallback_retained'):
    missing.append('invalid full-history/ridge/fallback '+j['id']);continue
   bind(q)
   for record in fit.get('files',[]):bind(verify_record(p,record))
   guardpath=p/'models'/j['id']/'fit_identity.json';guard=_read(guardpath)
   if not guard or guard.get('settings')!=settings(c):missing.append('qualification learner grid differs from prospective '+j['id'])
   else:
    bind(guardpath);dm=read_json(p/'derived'/j['id']/'manifest.json')
    expected=canonical_hash(dict(cfg=guard['settings'],device=guard['device'],environment=guard['numerical_environment'],
        source=file_hash(Path(__file__).with_name('models.py')),target=dm['target_fingerprint'],
        train=file_hash(p/'derived'/j['id']/'train.npz'),development=file_hash(p/'derived'/j['id']/'development.npz')))
    if expected!=guard['signature']:missing.append('model implementation/training inputs changed '+j['id'])
 for name in ['models/manifest.json','models/neural_path_control.json','derived/manifest.json','audit_existing.json','precision_inputs.json','measurement/qualification.json','measurement/coefficients.npz']:
  q=p/name
  if q.exists():bind(q)
 if not (p/'precision_inputs.json').exists():missing.append('phase0 circuit-level precision inputs')
 result={'status':'complete' if not missing else 'incomplete','passed':not missing,'missing':missing,'root':str(p.resolve()),'files':list(records.values())}
 result['evidence_hash']=canonical_hash(result);return result


def _physical(contract):
 # Registered timing/numerical namespaces intentionally differ. All other
 # physical dependencies, including the actual dt adapter, remain identical.
 return {k:v for k,v in contract.items() if k!='namespace'}


def measured_gate(c,path):
 from .generation import generation_contract,verify_row,cpu_topology
 p=Path(path);saved=read_json(p/'resolved_config.json');prov=read_json(p/'provenance.json');r=read_json(p/'generation/timing_audit.json')
 if saved['claim_status']!='timing_only' or r.get('continuations')!=8 or r.get('new_episodes')!=8 or not r.get('complete'):raise ValueError('Full8episode timing gate incomplete')
 if saved['backend']!='neuron' or saved['simulation']!=c['simulation'] or saved['stop_ms']!=7010 or saved['cutoff_ms']!=6000:raise ValueError('Gate physics/support differ; remeasure')
 if r.get('cpu_topology')!=cpu_topology():raise ValueError('Measured CPU model/affinity/topology changed; remeasure')
 now=environment();prior=prov['environment']
 for k in ('host','platform','python','versions','logical_cpus'):
  if now.get(k)!=prior.get(k):raise ValueError('Gate host/software changed; remeasure')
 manifest=read_json(p/'generation/native/manifest.json');rows=manifest['trials']
 if len(rows)!=8 or len({x['trial_id'] for x in rows})!=8:raise ValueError('Timing native allocation is not exactly8 unique episodes')
 expected=_physical(generation_contract(c));identities=[]
 for split in r['requested_splits']:
  directory=p/'generation'/split;identity=read_json(directory/'generation_contract.json')
  if identity.get('generation_contract_hash')!=canonical_hash(identity['generation_contract']) or _physical(identity['generation_contract'])!=expected:
   raise ValueError('Timing physical-source contract changed; remeasure actual implementation')
  if identity.get('allocation_hash')!=canonical_hash(identity['context_allocation']):raise ValueError('Timing seed allocation hash mismatch')
  identities.append(_record(p,directory/'generation_contract.json'))
 for row in rows:
  verify_row(p/'generation',row)
  if row.get('generation_contract_hash') not in {read_json(p/i['path'])['generation_contract_hash'] for i in identities}:raise ValueError('Timing row belongs to another generation contract')
 for sub in r['results']:
  if sub.get('integration_dt_ms')!=c['simulation']['native_dt_ms'] or not sub.get('full_prefix_replay') or sub.get('state_caching'):
   raise ValueError('Timing did not replay full prefixes at declared dt')
 fit=_read(p/'fit_throughput.json')
 if not fit or fit.get('host')!=platform.node() or fit.get('status')!='complete':raise ValueError('Same-host fit/calibration throughput gate absent')
 if settings(saved)!=settings(c):raise ValueError('Measured fitting grid differs from planned grid')
 hardware=_fit_hardware(fit['device']);fit_env=environment()
 # CUBLAS workspace policy is part of the measured software environment.
 if fit.get('environment')!=fit_env:raise ValueError('Fitting throughput environment changed')
 fit_signature=canonical_hash({'settings':settings(saved),'environment':fit_env,'device':fit['device'],
     'source':file_hash(Path(__file__).with_name('controls.py')),'models_source':file_hash(Path(__file__).with_name('models.py')),'hardware':hardware})
 if fit.get('signature')!=fit_signature or fit.get('hardware')!=hardware or fit.get('settings')!=settings(saved):raise ValueError('Fitting throughput source/settings/hardware changed')
 if not r.get('mean_continuation_seconds',0)>0 or not r.get('native_bytes',0)>0 or not r.get('peak_process_rss_bytes',0)>0:raise ValueError('Timing measurements invalid')
 files=identities+[_record(p,p/n) for n in ('resolved_config.json','provenance.json','generation/timing_audit.json','generation/native/manifest.json','fit_throughput.json')]
 return {**r,'root':str(p.resolve()),'environment':prior,'manifest_sha256':file_hash(p/'generation/native/manifest.json'),'fit_throughput':fit,'files':files,'evidence_hash':canonical_hash(files)}


def numerical_qualification(c,path,phase0_root):
 from .generation import generation_contract,verify_row
 from .numerical import panel_config
 p=Path(path);dest=p/'numerical';r=read_json(dest/'summary.json')
 if r.get('status')!='complete' or not r.get('passed') or not r.get('forcing_verified'):raise ValueError('Coupled-event numerical equivalence not qualified')
 if r.get('config_hash')!=canonical_hash(c) or r.get('episodes')!=counts(c)['numerical_episodes']:raise ValueError('Numerical config/allocation differs')
 if Path(r['qualification_root']).resolve()!=Path(phase0_root).resolve():raise ValueError('Numerical predictor used another qualification root')
 if r.get('output_grid_hz')!=16000 or not r.get('same_native_fourier_normalization'):raise ValueError('Numerical target output grid/normalization differs')
 for target in ('T0_l2_d32','T0_l10_d128','T3_l10_d1000'):
  if not r.get('reports',{}).get(target,{}).get('passed'):raise ValueError('Numerical anchor unresolved: '+target)
 for name,key in [('numerical_arrays.npz','arrays_sha256'),('coupled_forcing.json','coupled_forcing_sha256')]:
  if file_hash(dest/name)!=r.get(key):raise ValueError('Numerical evidence checksum mismatch '+name)
 forcing=read_json(dest/'coupled_forcing.json')
 if not forcing.get('passed') or not all(x.get('passed') and x.get('levels')==3 for x in forcing.get('trials',[])) or len(forcing.get('trials',[]))!=counts(c)['numerical_episodes']//3:
  raise ValueError('Incomplete same-forcing verification across all integration levels')
 for name,digest in r['predictor_identity']['files'].items():
  q=(Path(phase0_root)/name).resolve()
  if not q.is_relative_to(Path(phase0_root).resolve()) or file_hash(q)!=digest:raise ValueError('Numerical frozen predictor/normalization changed')
 if read_json(dest/'predictor_identity.json')!=r['predictor_identity']:raise ValueError('Numerical predictor identity record differs')
 files=[_record(p,dest/n) for n in ('summary.json','numerical_arrays.npz','coupled_forcing.json','predictor_identity.json')]
 for dt,sub in zip(c['numerical']['dts_ms'],r['generation']):
  directory=dest/('dt_'+str(dt).replace('.','p'));identity=read_json(directory/'generation_contract.json');q=panel_config(c,dt)
  expected=generation_contract(q)
  if identity.get('generation_contract')!=expected or identity.get('generation_contract_hash')!=canonical_hash(expected):raise ValueError('Numerical physical dependency changed')
  if identity.get('context_allocation')!=contexts(q):raise ValueError('Numerical context allocation changed')
  manifest=directory/'native/manifest.json'
  if file_hash(manifest)!=sub.get('native_manifest_sha256'):raise ValueError('Numerical native manifest changed')
  rows=read_json(manifest)['trials'];planned=sum(x['score_futures'] for x in contexts(q))
  if len(rows)!=planned or len({x['trial_id'] for x in rows})!=planned:raise ValueError('Numerical native allocation incomplete')
  for row in rows:verify_row(directory,row)
  files.extend([_record(p,manifest),_record(p,directory/'generation_contract.json')])
 if len(r['generation'])!=3:raise ValueError('Numerical generation levels incomplete')
 return {**r,'root':str(p.resolve()),'files':files,'evidence_hash':canonical_hash(files)}


def _reuse(c,native_source):
 from .generation import import_reused
 wanted=[r for r in contexts(c) if r.get('reuse')]
 if not wanted:return {'status':'not_required','reused_episodes':0}
 rows,record=import_reused(c,native_source,wanted)
 if len(rows)!=c['reuse']['expected_episodes']:raise ValueError('Reusable native allocation differs from expected320')
 return {'status':'complete',**record,'records_hash':canonical_hash(rows)}


def _precision_binding(c,root,source_archive,phase0_root,compute=False):
 path=Path(root)/'precision_certificate.json';bindingpath=Path(root)/'precision_binding.json'
 if source_archive is None:
  old=_read(bindingpath)
  if not old:raise ValueError('Precision archive binding absent; provide --source-archive')
  source_archive=old['source_archive']
 source=Path(source_archive).resolve()
 if not source.is_file():raise ValueError('Precision source must be the available standard numeric ZIP')
 digest=file_hash(source)
 if digest!=c['reuse']['source_archive_sha256']:raise ValueError('Historical numeric ZIP differs from registered source hash')
 phasehash=file_hash(Path(phase0_root)/'precision_inputs.json') if phase0_root and (Path(phase0_root)/'precision_inputs.json').exists() else None
 if compute:
  from .precision import certificate
  certificate(c,root,source,phase0_root)
 result=dict(source_archive=str(source),source_archive_sha256=digest,phase0_precision_sha256=phasehash,
             certificate_sha256=file_hash(path),protocol_hash=canonical_hash(c['protocol']))
 if compute:atomic_json(bindingpath,result)
 elif read_json(bindingpath)!=result:raise ValueError('Precision source, Phase-0 inputs or certificate changed; rebuild measured plan')
 cert=read_json(path)
 if not cert.get('eligible_to_freeze') or cert.get('planned_N')!=c['protocol']['test_circuits']:raise ValueError('Final worst-case precision certificate not eligible')
 return result


def plan(c,root,*,timing_root=None,phase0_root=None,numerical_root=None,source=None,source_archive=None,native_source=None):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);design=counts(c);registry=registry_check(c);reasons=[]
 if source is not None and source_archive is not None and Path(source).resolve()!=Path(source_archive).resolve():raise ValueError('Conflicting archive inputs')
 archive=source_archive or source
 phase={'passed':False,'missing':['phase0 root not supplied']};gate=None;numeric=None;reuse=None;precision=None
 if phase0_root:
  try:phase=qualify_phase0(c,phase0_root)
  except (ValueError,OSError,KeyError) as exc:phase={'passed':False,'missing':[str(exc)]}
 if not phase['passed']:reasons.extend(phase['missing'])
 if timing_root:
  try:gate=measured_gate(c,timing_root)
  except (ValueError,OSError,KeyError) as exc:reasons.append(str(exc))
 else:reasons.append('fresh same-host measured8episode gate not supplied')
 try:precision=_precision_binding(c,root,archive,phase0_root,compute=archive is not None)
 except (ValueError,OSError,KeyError) as exc:reasons.append(str(exc))
 try:reuse=_reuse(c,native_source)
 except (ValueError,OSError,KeyError) as exc:reasons.append(str(exc))
 if numerical_root:
  try:numeric=numerical_qualification(c,numerical_root,phase0_root)
  except (ValueError,OSError,KeyError,TypeError) as exc:reasons.append(str(exc))
 else:reasons.append('coupled-event dt convergence not qualified')
 seconds=gate['mean_continuation_seconds'] if gate else c['budget']['provisional_seconds'];perbytes=gate['native_bytes']/8 if gate else c['budget']['provisional_episode_bytes']
 core=design['new_episodes']*seconds/3600
 numeric_hours=sum(c['simulation']['native_dt_ms']/dt for dt in c['numerical']['dts_ms'])*design['numerical_episodes']/len(c['numerical']['dts_ms'])*seconds/3600
 if numeric:numeric_hours=sum(r['summed_generation_seconds'] for r in numeric['generation'])/3600
 native=math.ceil((design['new_episodes']+design['numerical_episodes'])*perbytes);summed=core+numeric_hours
 topology=_resources(root,c['data']['workers'],gate.get('peak_process_rss_bytes',0) if gate else 0,numerical_root);free=topology['free_bytes'];resource=list(topology['blocked_reasons'])
 if summed>c['budget']['max_generation_hours']:resource.append('summed generation exceeds500h cap')
 if native>c['budget']['max_native_bytes']:resource.append('native projection exceeds120GB cap')
 if free+topology['completed_native_storage_credit_bytes']<c['budget']['minimum_free_bytes']:resource.append('free plus already-written allocation is below250GB storage reserve')
 if gate and c['data']['workers']!=gate['workers']:resource.append('requested workers differ from measured gate concurrency')
 reasons.extend(resource)
 result={'status':'planned','config_hash':canonical_hash(c),'counts':design,'registry':registry,
  'phase0_root':str(Path(phase0_root).resolve()) if phase0_root else None,'qualification_root':str(Path(phase0_root).resolve()) if phase0_root else None,
  'timing_root':str(Path(timing_root).resolve()) if timing_root else None,'numerical_root':str(Path(numerical_root).resolve()) if numerical_root else None,
  'source_path':str(Path(archive).resolve()) if archive else precision['source_archive'] if precision else None,
  'native_source':str(Path(native_source).resolve()) if native_source else None,'native_reuse':reuse,
  'phase0':phase,'measured_gate':gate,'numerical_qualification':numeric,'precision_binding':precision,
  'precision_certificate_sha256':precision['certificate_sha256'] if precision else None,
  'generation_core_summed_hours':core,'generation_numerical_summed_hours':numeric_hours,'summed_generation_hours':summed,
  'estimated_generation_wall_hours':summed/c['data']['workers'],'workers':c['data']['workers'],
  'native_bytes_estimate':native,'lossless_export_bytes_reserve':native,'model_export_storage_reserve_bytes':max(0,c['budget']['minimum_free_bytes']-2*native),
  'free_bytes':free,'measured_topology':topology,'cost_status':'measured same-host extrapolation; fine dt cost observed' if numeric else 'same-host coarse extrapolation; fine dt scaled until observed' if gate else 'provisional only; no execution authorization',
  'fit_calibration_cost':gate['fit_throughput'] if gate else {'status':'unmeasured'},
  'eligible_numerical_execution':bool(gate and phase['passed'] and not resource),'eligible_for_generation':not reasons,
  'blocked_reasons':reasons,'numerical_gate_passed':bool(numeric),
  'positive_test_effect_required':False,'no_optional_experiments':True}
 if not reasons:
  payload={'status':'frozen','schema_version':c['schema_version'],'config_hash':canonical_hash(c),'resolved_config':c,'registry':registry,
   'operators':{a:metadata(a) for a in c['acquisition_ids']},'learner_grid':settings(c),'retention_grid':retention_settings(c),'inference':c['protocol'],
   'phase0_evidence_hash':phase['evidence_hash'],'precision_binding':precision,'native_reuse':reuse,
   'numerical_evidence_hash':numeric['evidence_hash'],
   'calibration':'24 separate circuits; equal-circuit constant diagonal family per frozen mean; no calibration model selection',
   'freeze_sequence':'Protocol precedes fresh data. Final means freeze before calibration; calibration policy frozen before evaluation.',
   'no_new_test_selection':True,'claim':'Two predeclared families conditional on training/model selection; no human EEG ceiling or H5/H6 result'}
  payload['protocol_hash']=canonical_hash(payload);old=_read(root/'protocol.json')
  if old and old!=payload:raise ValueError('Protocol immutable; changed qualification requires new root')
  atomic_json(root/'protocol.json',payload);result['protocol_hash']=payload['protocol_hash']
 result['plan_hash']=canonical_hash(result);atomic_json(root/'plan.json',result);return result


def _mean_freeze(c,root):
 root=Path(root);records={}
 for job in c['selected_job_ids']:
  directory=root/'models'/job;selection=read_json(directory/'selection.json');fit=read_json(directory/'manifest.json')
  if selection.get('status')!='frozen' or selection.get('selection_split')!='development' or fit['selected_candidate']!=selection['selected_candidate']:
   raise ValueError('Freeze training/development means before calibration/evaluation generation')
  for p in (directory/'selection.json',directory/'fit_identity.json'):
   records[str(p.relative_to(root))]=_record(root,p)
  for candidate in fit['candidates']:
   record=next(r for r in candidate['files'] if r['path']==candidate['state_file'])
   p=verify_record(root,record);records[record['path']]=_record(root,p)
 result={'status':'frozen','files':list(records.values()),'selection_split':'development',
         'policy':'All numeric mean candidates and selection fixed before calibration generation; no later refitting'}
 result['freeze_hash']=canonical_hash(result);path=root/'mean_freeze.json'
 if path.exists() and read_json(path)!=result:raise ValueError('Mean weights/selection changed after calibration freeze')
 if not path.exists():atomic_json(path,result)
 return result


def require_execution(c,root,execute,timing=False,numerical=False,split=None):
 if not execute:raise ValueError('Native execution requires explicit --execute after a saved measured plan')
 registry_check(c);root=Path(root)
 if timing:
  if c['claim_status']!='timing_only' or c['backend']!='neuron' or counts(c)['total_episodes']!=8:raise ValueError('Timing restricted to registered8episode full-network gate')
  ancestor=root
  while not ancestor.exists():ancestor=ancestor.parent
  if shutil.disk_usage(ancestor).free<2000000000:raise ValueError('Timing needs at least2GB free')
  return
 if c['claim_status']=='engineering':return
 p=read_json(root/'plan.json')
 if p.get('config_hash')!=canonical_hash(c) or p.get('plan_hash')!=canonical_hash({k:v for k,v in p.items() if k!='plan_hash'}):raise ValueError('Plan config/hash mismatch')
 key='eligible_numerical_execution' if numerical else 'eligible_for_generation'
 if not p.get(key):raise ValueError('Measured plan blocks execution: '+ '; '.join(p.get('blocked_reasons',[])))
 if measured_gate(c,p['timing_root'])['evidence_hash']!=p['measured_gate']['evidence_hash']:raise ValueError('Timing artifacts changed after plan')
 phase=qualify_phase0(c,p['phase0_root'])
 if not phase['passed'] or phase['evidence_hash']!=p['phase0']['evidence_hash']:raise ValueError('Phase0 artifacts changed after plan')
 topology=_resources(root,c['data']['workers'],p['measured_gate'].get('peak_process_rss_bytes',0),p.get('numerical_root'))
 if topology['blocked_reasons']:raise ValueError('Current worker topology blocks execution: '+ '; '.join(topology['blocked_reasons']))
 if topology['free_bytes']+topology['completed_native_storage_credit_bytes']<c['budget']['minimum_free_bytes']:raise ValueError('Free storage below measured allocation reserve')
 if numerical:return
 if _precision_binding(c,root,p['source_path'],p['phase0_root'])!=p['precision_binding']:raise ValueError('Precision certificate binding changed after plan')
 if numerical_qualification(c,p['numerical_root'],p['phase0_root'])['evidence_hash']!=p['numerical_qualification']['evidence_hash']:raise ValueError('Numerical evidence changed after plan')
 if _reuse(c,p['native_source'])!=p['native_reuse']:raise ValueError('Native reuse source changed after plan')
 protocol=read_json(root/'protocol.json')
 if protocol.get('protocol_hash')!=p.get('protocol_hash') or protocol.get('protocol_hash')!=canonical_hash({k:v for k,v in protocol.items() if k!='protocol_hash'}):raise ValueError('Frozen protocol changed/tampered')
 requested=set([split] if isinstance(split,str) else split or c['data']['structures'])
 if requested&{'calibration','evaluation'}:_mean_freeze(c,root)
 if 'evaluation' in requested:
  # Evaluation does not select its own interval family or refit means.
  calibration=read_json(root/'calibration/manifest.json')
  if calibration.get('status')!='complete' or set(c['selected_job_ids'])-{r['id'] for r in calibration.get('jobs',[]) if r['status']=='complete'}:
   raise ValueError('Complete separate calibration before evaluation generation')

  calibration_files=[]
  for job in c['selected_job_ids']:
   manifest=root/'calibration'/job/'manifest.json';r=read_json(manifest)
   if not r.get('calibrated_means_unchanged') or r.get('selection_uses_calibration') or r.get('selected_candidate')!=read_json(root/'models'/job/'selection.json')['selected_candidate']:
    raise ValueError('Calibration altered frozen means/selection')
   calibration_files.append(_record(root,manifest))
   for row in r['files']:calibration_files.append(_record(root,verify_record(root,row)))
  frozen={'status':'frozen','mean_freeze_sha256':file_hash(root/'mean_freeze.json'),'files':calibration_files}
  frozen['freeze_hash']=canonical_hash(frozen);path=root/'calibration_freeze.json'
  if path.exists() and read_json(path)!=frozen:raise ValueError('Calibration changed after evaluation freeze')
  if not path.exists():atomic_json(path,frozen)
