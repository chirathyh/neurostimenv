"""Frozen means, first-four primary futures and separate calibration accounting."""
from pathlib import Path
import json
import numpy as np
from .artifacts import atomic_json,atomic_npz,read_json,file_hash,canonical_hash,artifact_record,verify_record,module_hash
from .derivation import load_panel
from .models import context_weights,prediction_record
from .statistics import select_primary_futures,aggregate_circuit_losses,family_b,bootstrap_ratio_sensitivity
from ..scoring import fork_risk_components
from ..v4.scoring import assert_risk_contract,gaussian_diagnostics
from .replication import synchronized_jackknife


def circuit_values(context_values,panel):
 out=[]
 for sid in np.unique(panel['structure_id']):
  take=panel['structure_id']==sid;sub={k:v[take] for k,v in panel.items() if isinstance(v,np.ndarray) and len(v.shape)>0 and v.shape[0]==len(take)}
  out.append(float(context_weights(sub)@np.asarray(context_values)[take]))
 return np.asarray(out,dtype=np.float64)

def score(config,root,jobs=None,resume=False):
 root=Path(root);index=read_json(root/'models/manifest.json');ids=jobs or [j['id'] for j in index['jobs']];summaries=[];precision=[]
 for job in ids:
  panels,manifest=load_panel(root,job);fit=read_json(root/'models'/job/'manifest.json');assert_risk_contract(manifest['risk_contract'],fit['risk_contract'])
  ev=select_primary_futures(panels['evaluation']);scale=manifest['normalization']['y_scale'];calpath=root/'calibration'/job/'manifest.json'
  if config['claim_status']=='prospective_confirmation' and not calpath.exists():raise ValueError('Separate calibration must complete before primary scoring')
  sig=canonical_hash({'fit':file_hash(root/'models'/job/'manifest.json'),'inference':file_hash(root/'models'/job/'inference_manifest.json'),'evaluation':manifest['files']['evaluation']['sha256'],'calibration':file_hash(calpath) if calpath.exists() else None,'source':module_hash('scoring','statistics','replication')})
  out=root/'scores'/job;marker=out/'summary.json'
  if marker.exists():
   old=read_json(marker)
   if not resume or old['signature']!=sig:raise ValueError('Score dependency mismatch')
   for r in old['files']:verify_record(root,r)
   summaries.append(old);continue
  rows=[];files=[];losses={};baselines={};circuits=np.unique(ev['structure_id'])
  for candidate in fit['candidates']:
   name=candidate['id'];path=root/'models'/job/name/'evaluation.npz'
   expected=prediction_record(root,job,path);verify_record(root,expected)
   with np.load(path,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
   if not np.array_equal(pred['context_id'],ev['context_id']):raise ValueError('Prediction ID/order mismatch')
   assert_risk_contract(manifest['risk_contract'],json.loads(str(pred['risk_contract_json'])))
   base=np.broadcast_to(pred['baseline_mean'],pred['mean'].shape)
   comp=fork_risk_components(ev['Y'],pred['mean'],ev['future_mask']);b=fork_risk_components(ev['Y'],base,ev['future_mask'])['mean_risk']
   cr=circuit_values(comp['mean_risk'],ev);cb=circuit_values(b,ev);losses[name]=cr;baselines[name]=cb
   if cb.mean()<=0:raise ValueError('Positive fitted-mean baseline risk required for normalized improvement')
   component_path=out/(name+'__risk_components.npz')
   atomic_npz(component_path,**comp,baseline_context_risk=b,baseline_per_future_mse=np.mean((ev['Y']-base[:,None])**2,axis=-1),
       context_id=ev['context_id'],structure_id=ev['structure_id'],future_id=ev['future_id'],future_mask=ev['future_mask'])
   files.append(artifact_record(root,component_path))
   record={'candidate':name,'circuit_ids':circuits.tolist(),'circuit_risk':cr.tolist(),'baseline_circuit_risk':cb.tolist(),
           'model_risk':float(cr.mean()),'baseline_risk':float(cb.mean()),'gain_pp':float(100*(1-cr.mean()/cb.mean())),
           'physical_mse':float(cr.mean()*scale**2),'physical_risk_units':manifest['target_spec'].get('risk_units',manifest['risk_contract']['units']+' squared'),
           'state_variance':float(circuit_values(comp['state_floor'],ev).mean()),'negative_corrected_context_excess':int(np.sum(comp['corrected_excess']<0)),
           'calibration':{},'selection_eligible':candidate['primary_eligible'],'selected_member':name in fit['selected_member_ids']}
   families={k:pred[k] for k in ('variance_uncalibrated','variance_development_calibrated') if k in pred}
   cp=root/'calibration'/job/name/'variance.npz'
   if cp.exists():
    cm=read_json(calpath);verify_record(root,next(r for r in cm['files'] if r['path']==str(cp.relative_to(root))))
    with np.load(cp,allow_pickle=False) as z:families['variance_independent_calibration']=np.broadcast_to(z['variance'],pred['mean'].shape)
   # Explicit unit-variance baseline for probability scoring, same mean.
   if name=='train_mean':families['variance_unit']=np.ones_like(pred['mean'])
   for kind,var in families.items():
    var=np.broadcast_to(var,pred['mean'].shape);context,arrays=gaussian_diagnostics(ev['Y'],pred['mean'],var,ev['future_mask'],y_scale=scale)
    record['calibration'][kind]={k:float(circuit_values(v,ev).mean()) for k,v in context.items()}
    fp=out/(name+'__'+kind+'.npz');atomic_npz(fp,**arrays,context_id=ev['context_id'],structure_id=ev['structure_id'],future_id=ev['future_id']);files.append(artifact_record(root,fp))
   record['conditional_mc']=synchronized_jackknife(ev['Y'],pred['mean'],ev)
   rows.append(record)
  members=fit['selected_member_ids'];model=np.mean([losses[n] for n in members],axis=0);baseline=baselines[members[0]]
  selected_neural=all(next(r['kind'] for r in fit['candidates'] if r['id']==n)=='neural' for n in members)
  primary={'candidate':fit['selected_candidate'],'member_ids':members,'circuit_ids':circuits.tolist(),'model_circuit_risks':model.tolist(),'baseline_circuit_risks':baseline.tolist(),
           'model_risk':float(model.mean()),'baseline_risk':float(baseline.mean()),'gain_pp':float(100*(1-model.mean()/baseline.mean())),
           'member_circuit_risks':[losses[n].tolist() for n in members],'selected_neural_policy':selected_neural,'fitting_repeats_averaged':len(members),
           'mean_fallback_selected':members==['train_mean'],
           'mean_fallback_interpretation':'If the fallback is selected, equivalence to itself is algebraic and does not test absence of EEG predictability',
           'seed_aggregation':'member-specific risks averaged inside circuit, not ensemble-mean squared error'}
  fp64=[]
  for sensitivity in fit.get('fp64_sensitivity',[]):
   if 'state_file' not in sensitivity:
    fp64.append(sensitivity);continue
   name=sensitivity['candidate'];path=root/'models'/job/'fp64_sensitivity'/name/'evaluation.npz'
   verify_record(root,prediction_record(root,job,path))
   with np.load(path,allow_pickle=False) as z:
    if not np.array_equal(z['context_id'],ev['context_id']):raise ValueError('Float64 sensitivity prediction IDs mismatch')
    assert_risk_contract(manifest['risk_contract'],json.loads(str(z['risk_contract_json'])))
    context=np.mean((ev['Y']-z['mean'][:,None])**2,axis=-1).mean(1)
   cr=circuit_values(context,ev)
   fp64.append(dict(candidate=name,status='complete',circuit_ids=circuits.tolist(),model_circuit_risks=cr.tolist(),
       paired_float32_circuit_risks=losses[name].tolist(),fitting_seed=next(r['seed'] for r in fit['candidates'] if r['id']==name),
       interpretation='paired optimizer seed and data; both checkpoints selected on development; no primary reselection'))
  numeric=[r for r in fp64 if 'model_circuit_risks' in r]
  if numeric:
   from .statistics import paired_ratio_difference
   double=np.mean([r['model_circuit_risks'] for r in numeric],axis=0)
   single=np.mean([r['paired_float32_circuit_risks'] for r in numeric],axis=0)
   fp64_comparison=paired_ratio_difference(double,baseline,single,baseline)
   fp64_comparison.update(interpretation='G(float64 fit) minus G(float32 fit); matched seeds averaged as risks, descriptive numerical sensitivity',fitting_repeats=len(numeric))
  else:fp64_comparison={'status':'not_required_or_unrun','details':fp64}
  summary={'status':'complete','signature':sig,'job_id':job,'claim_status':config['claim_status'],'primary_future_policy':'first4registeredscorefutures, includingH2contexts',
           'selected':primary,'rows':rows,'fp64_sensitivity':fp64,'fp64_comparison':fp64_comparison,'files':files,'risk_contract':manifest['risk_contract'],'scope':'Formal family decision only in confirm-statistics; other jobs descriptive.'}
  atomic_json(marker,summary);summaries.append(summary)
 # Build planning input from predeclared eligible policies, never evaluation winners.
 primary_job=config['protocol']['primary_job_id']
 for summary in summaries:
  if summary['job_id']!=primary_job:continue
  fit=read_json(root/'models'/primary_job/'manifest.json');lookup={r['candidate']:r for r in summary['rows']}
  policies=fit['policies'];chosen={fit['selected_candidate']};family_best={}
  for p in policies:
   kind='neural' if len(p['members'])>1 else next(r['kind'] for r in fit['candidates'] if r['id']==p['members'][0])
   if kind not in family_best or p['development_mse']<family_best[kind]['development_mse']:family_best[kind]=p
  chosen.update(p['id'] for p in family_best.values())
  for p in policies:
   if p['id'] not in chosen:continue
   precision.append({'name':p['id'],'model_circuit_risks':np.mean([lookup[n]['circuit_risk'] for n in p['members']],axis=0).tolist(),
                     'baseline_circuit_risks':lookup[p['members'][0]]['baseline_circuit_risk'],'circuit_ids':lookup[p['members'][0]]['circuit_ids'],
                     'is_full_history':True,'status':'complete','selection_split':'development','target':primary_job,'policy':'selected overall or development-best within prespecified full-history feature family'})
 result={'status':'complete','jobs':[{'id':s['job_id'],'status':'complete','path':str((root/'scores'/s['job_id']/'summary.json').relative_to(root))} for s in summaries]}
 atomic_json(root/'scores/manifest.json',result)
 if config['claim_status']=='exploratory_existing':atomic_json(root/'precision_inputs.json',{'comparisons':precision,'claim_status':'exploratory_existing','selection':'development-only; all declared families preserved for precision sensitivity'})
 return result


def confirm_statistics(config,root):
 from .statistics import family_a
 root=Path(root)
 if config['claim_status']!='prospective_confirmation':raise ValueError('Inspected circuits cannot receive fresh confirmation status')
 protocol=read_json(root/'protocol.json')
 if protocol.get('status')!='frozen':raise ValueError('Protocol not frozen')
 model=read_json(root/'scores'/config['protocol']['primary_job_id']/'summary.json')['selected']
 if len(model['circuit_ids'])!=config['protocol']['test_circuits']:raise ValueError('Incomplete fixedN primary panel')
 a=read_json(root/'retention/primary_components.json');rows=a['per_circuit']
 task=np.asarray([r['task_difference_bits'] for r in rows],float)
 recon=np.asarray([r['reconstruction_mse_feature']-r['reconstruction_mse_pca'] for r in rows],float)
 reconstruction_baseline=np.asarray([r['reconstruction_baseline_mse'] for r in rows],float)
 if [r['structure_id'] for r in rows]!=model['circuit_ids']:raise ValueError('Primary families must use same registered circuits')
 if not a.get('denominator_estimated_from_evaluation'):raise ValueError('Unexpected H3 reconstruction denominator policy')
 fa=family_a(task,recon,reconstruction_baseline_risks=reconstruction_baseline)
 neural=bool(model.get('selected_neural_policy',False))
 risks=model['member_circuit_risks'] if neural else model['model_circuit_risks']
 fb=family_b(risks,model['baseline_circuit_risks'],neural=neural)
 result={'status':'complete','protocol_hash':protocol['protocol_hash'],'familywise_alpha':.05,'family_a_alpha':.025,'family_b_alpha':.025,
         'H3_primary_family_A':fa,'H1_primary_family_B':fb,'H1_bootstrap_sensitivity':bootstrap_ratio_sensitivity(model['model_circuit_risks'],model['baseline_circuit_risks']),
         'inference_scope':'Conditional on frozen training/development selection and fitted means; circuits independent, not future/optimizer repeats. H5/H6 deferred.'}
 atomic_json(root/'confirmation_statistics.json',result);return result
