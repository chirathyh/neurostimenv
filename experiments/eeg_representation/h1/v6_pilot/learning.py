"""Bounded fitted pilot learners, with an actually trained nonmean state control."""
from pathlib import Path
import json,time
import numpy as np
from ..artifacts import atomic_json,atomic_npz,file_hash
from ..targets import canonical_hash
from ..v6 import models as v6
from ..v6.information_models import fit_design,_digest

def numeric(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def signature(panel,values,manifest,settings,*,implementation_sha256=None):
    return canonical_hash(dict(inputs={s:_digest({**{k:p[k] for k in ('context_id','structure_id','Y','future_mask')},'features':values[s]}) for s,p in panel.items()},
        risk_contract=manifest['risk_contract'],settings=settings,implementation=file_hash(Path(__file__)) if implementation_sha256 is None else implementation_sha256))

def ridge(panel,values,manifest,destination,config,*,resume=False,mechanism_scaled=False,design=None,signature_implementation_sha256=None):
    """All candidate predictions saved; development alone chooses mean or ridge.

    The predeclared alpha10 candidate is retained even if selection prefers mean.
    Model inputs have already passed complete-state coverage when used for state.
    """
    dest=Path(destination);marker=dest/'selection.json'
    settings=dict(alphas=config['ridge_alphas'],predeclared_alpha=config['state_candidate_alpha'],mechanism_scaled=mechanism_scaled)
    sig=signature(panel,values,manifest,settings,implementation_sha256=signature_implementation_sha256)
    if marker.exists():
        if not resume:raise FileExistsError(marker)
        old=json.loads(marker.read_text())
        if old['signature']!=sig:raise ValueError('Pilot readout inputs/settings changed')
        for name,digest in old['files'].items():
            if file_hash(dest/name)!=digest:raise ValueError('Pilot readout checksum mismatch')
        return old
    started=time.perf_counter();train=panel['train'];dev=panel['development'];w=v6.context_weights(train)
    design=fit_design(values['train'],w,already_scaled=mechanism_scaled) if design is None else design
    baseline=w@v6.first_targets(train);residual=(v6.first_targets(train)-baseline)*design['sqrt_weights'][:,None]
    projected=design['u'].T@residual;arrays={};state=dict(offset=design['offset'],scale=design['scale'],baseline=baseline)
    candidates=[];best=float('inf');chosen=None
    for alpha in [None]+list(config['ridge_alphas']):
        tag='train_mean' if alpha is None else 'ridge_'+str(float(alpha)).replace('.','p')
        coef=None if alpha is None else design['vt'].T@((design['singular']/(design['singular']**2+alpha))[:,None]*projected)
        if coef is not None:state[tag+'__coef']=coef
        losses={}
        for split,p in panel.items():
            pred=np.broadcast_to(baseline,(len(p['context_id']),len(baseline))).copy()
            if coef is not None:pred+=((np.asarray(values[split],float)-design['offset'])/design['scale'])@coef
            arrays[split+'__'+tag]=pred;losses[split]=v6.circuit_loss(p,pred)
        candidate=dict(id=tag,alpha=alpha,training_mse=losses['train'],development_mse=losses['development'],
            state_dependent_candidate=coef is not None,actually_trained=coef is not None,
            nominal_parameters=int(len(baseline)+(0 if coef is None else coef.size)),
            effective_df=0. if alpha is None else float(np.sum(design['singular']**2/(design['singular']**2+alpha))))
        candidates.append(candidate)
        if candidate['development_mse']<best:best=candidate['development_mse'];chosen=tag
    predeclared='ridge_'+str(float(config['state_candidate_alpha'])).replace('.','p')
    for split,p in panel.items():
        arrays[split+'_prediction']=arrays[split+'__'+chosen]
        arrays[split+'_candidate_prediction']=arrays[split+'__'+predeclared]
        arrays[split+'_context_id']=p['context_id'];arrays[split+'_structure_id']=p['structure_id']
    # Fit one shared diagonal variance family only on the separate calibration cohort.
    calibration={}
    if 'calibration' in panel:
        for label in ('prediction','candidate_prediction'):
            var=v6.matched_variance(panel['calibration'],arrays['calibration_'+label]);arrays['calibration_'+label+'_variance']=var
            calibration[label]=dict(variance=var.tolist(),fit_split='calibration',contexts=len(panel['calibration']['context_id']),
                mean_retuned=False,family='context-independent diagonal residual variance',marginal_not_joint=True)
    arrays['risk_contract_json']=np.asarray(json.dumps(manifest['risk_contract'],sort_keys=True))
    atomic_npz(dest/'state.npz',**state);atomic_npz(dest/'predictions.npz',**arrays)
    atomic_json(dest/'learning_curve.json',dict(kind='closed-form ridge regularization path, not gradient epochs',rows=candidates))
    atomic_json(dest/'losses.json',dict(candidates={tag:{s:v6.circuit_loss(p,arrays[s+'__'+tag]) for s,p in panel.items()} for tag in [r['id'] for r in candidates]},calibration=calibration))
    result=dict(status='complete',signature=sig,selected_id=chosen,predeclared_candidate_id=predeclared,
        candidates=candidates,selection='equal circuit/regime weighted development MSE only',
        selection_split='development',evaluation_selection=False,calibration_selection=False,
        actually_fitted_state_dependent=True,input_coordinates=int(values['train'].shape[1]),
        input_coordinate_policy='all adapter coordinates retained; ridge coefficients are fitted internal compression',
        training_contexts=len(train['context_id']),training_circuits=len(np.unique(train['structure_id'])),
        target_fingerprint=manifest['target_fingerprint'],risk_contract=manifest['risk_contract'],
        elapsed_seconds=time.perf_counter()-started,arithmetic='float64 closed-form fitting/predictions',
        files={n:file_hash(dest/n) for n in ('state.npz','predictions.npz','learning_curve.json','losses.json')})
    atomic_json(marker,result);return result

def tcn(panel,manifest,destination,config,history,device='cpu',resume=False):
    dest=Path(destination);marker=dest/'selection.json'
    cfg={**v6.DEFAULTS,'neural_seeds':[config['neural_seed']],'max_steps':config['neural_max_steps'],
         'patience':config['neural_patience'],'eval_every':config['neural_eval_every'],'dtype':config['dtype'],'threads':1}
    sig=signature(panel,{s:p['X'][:,-history:] for s,p in panel.items()},manifest,dict(cfg=cfg,history=history,device=device))
    if marker.exists():
        old=json.loads(marker.read_text())
        if not resume or old['signature']!=sig:raise ValueError('Pilot TCN identity changed')
        for n,h in old['files'].items():
            if file_hash(dest/n)!=h:raise ValueError('Pilot TCN output changed')
        return old
    net,state,account=v6.train_tcn(panel['train'],panel['development'],cfg,history_samples=history,seed=config['neural_seed'],device=device)
    baseline=v6.context_weights(panel['train'])@v6.first_targets(panel['train'])
    dev=v6.neural_predict(net,panel['development']['X'][:,-history:],cfg,device)
    shrink=v6.residual_shrinkage(panel['development'],dev,baseline)
    use=v6.circuit_loss(panel['development'],baseline+shrink*(dev-baseline))<v6.circuit_loss(panel['development'],np.broadcast_to(baseline,dev.shape))
    arrays={}
    for s,p in panel.items():
        raw=v6.neural_predict(net,p['X'][:,-history:],cfg,device)
        arrays[s+'_raw_prediction']=np.asarray(raw,np.float64)
        arrays[s+'_prediction']=np.asarray(baseline+(shrink if use else 0)*(raw-baseline),np.float64)
        arrays[s+'_context_id']=p['context_id'];arrays[s+'_structure_id']=p['structure_id']
    arrays['risk_contract_json']=np.asarray(json.dumps(manifest['risk_contract'],sort_keys=True))
    state.update(baseline=baseline,residual_shrinkage=np.asarray(shrink),selected_mean_fallback=np.asarray(not use))
    atomic_npz(dest/'state.npz',**state);atomic_npz(dest/'predictions.npz',**arrays)
    atomic_json(dest/'learning_curve.json',account)
    result=dict(account,status='complete',signature=sig,selected_id='tcn' if use else 'train_mean',history_samples=history,
        device=device,optimizer_budget=cfg['max_steps'],fitting_repeats=1,residual_shrinkage=shrink,
        training_contexts=len(panel['train']['context_id']),training_circuits=len(np.unique(panel['train']['structure_id'])),
        selection_split='development',mean_fallback_retained=True,
        files={n:file_hash(dest/n) for n in ('state.npz','predictions.npz','learning_curve.json')})
    atomic_json(marker,result);return result
