"""Small EEG-only qualification models; selection is confined to development.

Every tried candidate is retained. Numeric NPZ states never require pickle.
Optimization seeds are fitting repeats, not independent circuit replicates.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from ..artifacts import atomic_npz
from ...storage import atomic_json, file_hash, read_json
from ..targets import canonical_hash

DEFAULTS = dict(ridge_lags=[16, 64], ridge_alphas=[.01, .1, 1, 10, 100, 1000, 10000],
    state_ranks=[2, 4], state_alpha=100., neural_hidden=[32], neural_lags=64,
    neural_seeds=[0, 1, 2], neural_losses=['mse', 'gaussian'], max_steps=300,
    patience=40, eval_every=10, learning_rate=.001, weight_decay=.01,
    max_parameters=50000, variance_floor=1e-6, threads=1, neural_enabled=True)


def settings(config):
    out=copy.deepcopy(DEFAULTS)
    out.update(config.get('v4', {}).get('models', {}))
    out.update(config.get('models', {}))
    if list(out['ridge_lags']) != [16, 64] or list(out['ridge_alphas']) != DEFAULTS['ridge_alphas']:
        raise ValueError('Qualification requires the complete 16/64 by seven-alpha ridge grid')
    if int(out['max_parameters']) > 50000 or int(out['max_parameters']) < 1:
        raise ValueError('Small neural qualification parameter ceiling must be <=50000')
    if out['neural_enabled'] and config.get('claim_status')!='engineering' and len(set(out['neural_seeds'])) < 3:
        raise ValueError('Prospective neural diagnostics require at least three optimizer seeds')
    return out


def load_panel(root, job_id):
    """Read derived numeric arrays and verify files; never read pickle archives."""
    directory=Path(root)/'derived'/job_id
    manifest=read_json(directory/'manifest.json')
    if manifest['normalization'].get('fit_split') != 'train':
        raise ValueError('Only training-fitted target normalization is supported')
    panel={}; seen_futures=set()
    for split in ('train', 'development', 'evaluation'):
        path=directory/f'{split}.npz'
        record=manifest.get('files', {}).get(split, {})
        if isinstance(record, dict) and record.get('sha256') and file_hash(path)!=record['sha256']:
            raise ValueError('Derived panel checksum mismatch: '+str(path))
        with np.load(path, allow_pickle=False) as f: panel[split]={k:f[k] for k in f.files}
        p=panel[split]
        if p['X'].ndim!=2 or p['Y'].ndim!=3 or len(p['X'])!=len(p['Y']):
            raise ValueError('Expected X[K,L] and Y[K,M,D]')
        if not np.isfinite(p['X']).all() or not np.isfinite(p['Y'][p['future_mask']]).all():
            raise ValueError('Nonfinite observed input or valid target')
        if np.any(p['future_mask'].sum(1)==0): raise ValueError('Every context needs a scoring future')
        if len(set(p['context_id'].tolist()))!=len(p['context_id']): raise ValueError('Duplicate context IDs')
        scoring=list(p['future_id'][p['future_mask']]); reference=list(p['reference_future_id'][p['reference_mask']])
        all_ids=scoring+reference
        if any(not str(v) for v in all_ids) or len(set(all_ids))!=len(all_ids) or seen_futures.intersection(all_ids):
            raise ValueError('Duplicated/leaked or missing future identities')
        seen_futures.update(all_ids)
    for name in ('structure_id','context_id'):
        groups=[set(panel[s][name].tolist()) for s in panel]
        if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError('Train/development/evaluation '+name+' overlap')
    return panel,manifest


def job_ids(root, selected=None, config=None):
    rows=read_json(Path(root)/'derived/manifest.json')['jobs']
    available=[r.get('id',r.get('job_id')) for r in rows if r.get('status','complete') in ('complete','supported')]
    chosen=selected
    if chosen is None and config: chosen=config.get('selected_job_ids',config.get('models',{}).get('selected_job_ids'))
    chosen=available if chosen is None else list(chosen)
    if set(chosen)-set(available): raise ValueError('Requested derived job unavailable: '+str(set(chosen)-set(available)))
    return chosen


def circuit_weights(ids):
    ids=np.asarray(ids); groups,counts=np.unique(ids,return_counts=True)
    return np.asarray([1./(len(groups)*counts[np.where(groups==s)[0][0]]) for s in ids])


def target_mean(data):
    mask=np.asarray(data['future_mask'],bool)
    return np.where(mask[...,None],data['Y'],0).sum(1)/mask.sum(1)[:,None]


def first_targets(data):
    return data['Y'][np.arange(len(data['Y'])),np.argmax(data['future_mask'],axis=1)]


def circuit_loss(data, mean, variance=None):
    err=(data['Y']-np.asarray(mean)[:,None])**2
    if variance is not None: err=.5*(np.log(2*np.pi*np.asarray(variance)[:,None])+err/np.asarray(variance)[:,None])
    mask=data['future_mask']; row=np.where(mask,err.mean(-1),0).sum(1)/mask.sum(1)
    return float(circuit_weights(data['structure_id'])@row)


def fit_ridge(x, y, alpha, weights=None):
    weights=np.ones(len(x))/len(x) if weights is None else np.asarray(weights,float)
    offset=weights@x; scale=np.sqrt(weights@((x-offset)**2)); scale=np.where(scale>1e-12,scale,1.)
    z=(x-offset)/scale; intercept=weights@y
    # Multiplication by N keeps alpha in the usual sum-of-squares convention.
    w=np.sqrt(weights*len(x)); u,s,vt=np.linalg.svd(z*w[:,None],full_matrices=False)
    coefficient=(vt.T*(s/(s*s+float(alpha))))@u.T@((y-intercept)*w[:,None])
    return dict(offset=offset,scale=scale,coefficient=coefficient,intercept=intercept)


def predict_ridge(state,x): return ((x-state['offset'])/state['scale'])@state['coefficient']+state['intercept']


def residual_shrinkage(development, prediction, baseline):
    delta=np.asarray(prediction)-baseline; y=target_mean(development)-baseline
    weights=circuit_weights(development['structure_id'])[:,None]
    denominator=float(np.sum(weights*delta**2))
    return float(np.clip(np.sum(weights*delta*y)/denominator,0.,1.)) if denominator>0 else 0.


def calibrate_variance(development, mean, raw_variance, floor=1e-6):
    """Identical multiplicative calibration ACCESS, not final variance expressivity.

    E_dev[(Y-mu)^2/raw_variance] per coordinate, equal weight per circuit.
    A heteroscedastic raw Gaussian remains heteroscedastic after multiplication;
    use matched_calibrated_variance for the exactly matched constant family.
    No evaluation values enter either fit.
    """
    mask=development['future_mask']; residual=(development['Y']-mean[:,None])**2
    per=np.where(mask[...,None],residual,0).sum(1)/mask.sum(1)[:,None]
    multiplier=circuit_weights(development['structure_id'])@(per/np.maximum(raw_variance,floor))
    return np.maximum(multiplier,floor)



def matched_calibrated_variance(development, mean, floor=1e-6):
    """Exactly matched context-independent diagonal family for EVERY mean.

    Fit E_dev[(Y-mu)^2] per coordinate using equal circuit/context weights.
    The fitted vector is broadcast to all contexts and never reads raw predicted
    variance, input covariates or evaluation outcomes. Temporal covariance is
    unmodeled. This is the primary matched-variance-family comparison.
    """
    mean=np.asarray(mean,float)
    if mean.shape!=(len(development['Y']),development['Y'].shape[-1]):
        raise ValueError('Matched variance mean dimensions differ from development targets')
    mask=np.asarray(development['future_mask'],bool)
    if np.any(mask.sum(1)==0):raise ValueError('Each development context needs an observed future')
    residual=(development['Y']-mean[:,None])**2
    per=np.where(mask[...,None],residual,0).sum(1)/mask.sum(1)[:,None]
    variance=circuit_weights(development['structure_id'])@per
    if not np.isfinite(variance).all():raise ValueError('Nonfinite matched development variance')
    return np.maximum(variance,floor)


def make_network(input_dimension, output_dimension, hidden, gaussian, baseline, seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from torch import nn
    torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.body=nn.Sequential(nn.Linear(input_dimension,hidden),nn.Tanh())
            self.mean_head=nn.Linear(hidden,output_dimension)
            self.variance_head=nn.Linear(hidden,output_dimension) if gaussian else None
            self.register_buffer('training_mean',torch.as_tensor(baseline,dtype=torch.float32))
            nn.init.zeros_(self.mean_head.weight); nn.init.zeros_(self.mean_head.bias)
            if gaussian:
                nn.init.zeros_(self.variance_head.weight)
                nn.init.constant_(self.variance_head.bias,float(np.log(np.expm1(1.))))
        def forward(self,x):
            z=self.body(x); mu=self.training_mean+self.mean_head(z)
            var=None if self.variance_head is None else torch.nn.functional.softplus(self.variance_head(z))+1e-6
            return mu,var
    return ResidualMLP()


def train_neural(train, development, cfg, *, hidden=32, gaussian=False, seed=0, device='cpu'):
    """One shared training path used by the engineering control and EEG fits."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(int(cfg['threads'])); torch.use_deterministic_algorithms(True)
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    lag=min(int(cfg['neural_lags']),train['X'].shape[1]); weights=circuit_weights(train['structure_id'])
    y=first_targets(train); base=weights@y
    net=make_network(lag,y.shape[1],hidden,gaussian,base,seed).to(device)
    parameters=sum(p.numel() for p in net.parameters())
    if parameters>int(cfg['max_parameters']): raise ValueError('Neural parameter budget exceeded')
    x=torch.as_tensor(train['X'][:,-lag:],dtype=torch.float32,device=device)
    yy=torch.as_tensor(y,dtype=torch.float32,device=device)
    w=torch.as_tensor(weights,dtype=torch.float32,device=device)
    dx=torch.as_tensor(development['X'][:,-lag:],dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=float(cfg['learning_rate']),weight_decay=float(cfg['weight_decay']))
    best=None; stale=0; history=[]; start=time.perf_counter(); actual_steps=0
    for step in range(int(cfg['max_steps'])+1):
        if step % int(cfg['eval_every'])==0 or step==int(cfg['max_steps']):
            net.eval()
            with torch.no_grad():
                dm,dv=net(dx); metric=circuit_loss(development,dm.cpu().numpy(),None if not gaussian else dv.cpu().numpy())
            history.append(dict(step=step,development_score=metric))
            if best is None or metric<best[0]-1e-10:
                best=(metric,step,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()});stale=0
            else: stale+=int(cfg['eval_every'])
            if stale>=int(cfg['patience']): break
        if step==int(cfg['max_steps']): break
        net.train();optimizer.zero_grad(set_to_none=True);mu,var=net(x)
        loss=(mu-yy).square() if not gaussian else .5*(torch.log(2*torch.pi*var)+(mu-yy).square()/var)
        loss=(loss.mean(-1)*w).sum()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite neural loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.); optimizer.step();actual_steps+=1
    net.load_state_dict(best[2]);net.eval()
    state={'tensor__'+k:v.numpy() for k,v in best[2].items()}
    account=dict(parameter_count=parameters,selected_step=best[1],optimizer_updates=actual_steps,
        development_history=history,selection_criterion='Gaussian marginal NLL' if gaussian else 'MSE',
        selection_split='development',elapsed_seconds=time.perf_counter()-start,lag=lag,
        training_contexts=len(y),training_circuits=len(np.unique(train['structure_id'])),
        full_batch_examples=actual_steps*len(y),weight_decay=cfg['weight_decay'],seed=seed,
        actual_training_context_ids=train['context_id'].tolist(),actual_training_structure_ids=np.unique(train['structure_id']).tolist())
    return net,state,account


def neural_predict(net,data,lag,device):
    import torch
    with torch.no_grad(): m,v=net(torch.as_tensor(data['X'][:,-lag:],dtype=torch.float32,device=device))
    return m.cpu().numpy().astype(float),None if v is None else v.cpu().numpy().astype(float)


def known_learnable_control(config=None,device='cpu'):
    """Deterministic hold-out regression through exactly train_neural's path."""
    cfg=copy.deepcopy(DEFAULTS); cfg.update(config or {})
    # Fixed control protocol, independent of EEG outcomes/candidate tuning.
    cfg.update(neural_lags=16,max_steps=400,patience=100,eval_every=10,learning_rate=.01,weight_decay=.001)
    rng=np.random.default_rng(90127); x=rng.uniform(-1,1,(192,16))
    y=(.8*x[:,-1]+.3*x[:,-2])[:,None]*np.asarray([[1.,.7,.4,.2]])
    def part(a,b,offset):
        return dict(X=x[a:b],Y=y[a:b,None],future_mask=np.ones((b-a,1),bool),
                    structure_id=np.repeat(np.arange(offset,offset+4), (b-a)//4),
                    context_id=np.asarray([f'control-{i}' for i in range(a,b)]))
    tr=part(0,128,0); dv=part(128,160,4); ev=part(160,192,8)
    net,state,account=train_neural(tr,dv,cfg,hidden=32,seed=19,device=device)
    p,_=neural_predict(net,ev,16,device); baseline=np.broadcast_to(y[:128].mean(0),p.shape)
    risk=circuit_loss(ev,p); r0=circuit_loss(ev,baseline)
    return dict(status='complete',passed=bool(risk<.10*r0),risk=risk,baseline_risk=r0,
        skill=1-risk/r0,criterion='hold-out MSE < 10% training-mean MSE',
        same_neural_path=True,accounting=account,interpretation='engineering path qualification, not biophysical evidence')



def few_example_diagnostic(train,cfg,device='cpu'):
    """Memorization diagnostic, with counts from actual sliced context IDs.

    This uses the first four examples by saved order, and selects a checkpoint on
    those same examples solely for optimization diagnostics. It is not validation.
    """
    indices=np.arange(min(4,len(train['X'])));n=len(train['X']);subset={}
    for name,value in train.items():
        subset[name]=value[indices] if name not in ('input_time_ms','target_time_ms') and np.asarray(value).ndim and len(value)==n else value
    settings=copy.deepcopy(cfg);settings.update(max_steps=100,patience=100,eval_every=10,learning_rate=.01)
    net,state,account=train_neural(subset,subset,settings,hidden=32,seed=97,device=device)
    prediction,_=neural_predict(net,subset,account['lag'],device)
    report=dict(status='complete',diagnostic='first up-to-four examples, training-only memorization',
        context_ids=subset['context_id'].tolist(),structure_ids=np.unique(subset['structure_id']).tolist(),
        actual_context_count=len(indices),actual_circuit_count=len(np.unique(subset['structure_id'])),
        training_mse=circuit_loss(subset,prediction),accounting=account,
        selection_split='same training subset, diagnostic only; no generalization claim')
    return report,dict(prediction=prediction,truth=first_targets(subset),context_id=subset['context_id'],structure_id=subset['structure_id'],**state)


def _state_fit(train,y,rank,alpha):
    lag=min(16,train['X'].shape[1]-1)
    sequences=np.lib.stride_tricks.sliding_window_view(train['X'],lag,axis=1)
    a=sequences[:,:-1].reshape(-1,lag);b=sequences[:,1:].reshape(-1,lag)
    ix=np.linspace(0,len(a)-1,min(2048,len(a)),dtype=int);a=a[ix];b=b[ix];offset=a.mean(0)
    _,_,vt=np.linalg.svd(a-offset,full_matrices=False);components=vt[:min(rank,len(vt))]
    z=(a-offset)@components.T;zz=(b-offset)@components.T
    transition=np.linalg.lstsq(z,zz,rcond=None)[0];radius=np.max(np.abs(np.linalg.eigvals(transition)))
    if radius>.999: transition*=.999/radius
    state=dict(delay_offset=offset,components=components,transition=transition,lag=np.asarray(lag))
    state.update(fit_ridge(_state_features(state,train['X']),y,alpha,circuit_weights(train['structure_id'])))
    return state


def _state_features(state,x):
    z=(x[:,-int(state['lag']):]-state['delay_offset'])@state['components'].T
    return np.concatenate([z,z@state['transition']],axis=1)


def _record(root,path):
    return dict(path=Path(path).relative_to(root).as_posix(),sha256=file_hash(path),bytes=Path(path).stat().st_size)


def _signature(config,manifest,root,job):
    return canonical_hash(dict(settings=settings(config),manifest=manifest,
        panels={s:file_hash(Path(root)/'derived'/job/f'{s}.npz') for s in ('train','development','evaluation')},
        implementation=file_hash(Path(__file__))))


def fit(config,root,job_ids=None,device='cpu',resume=False):
    root=Path(root);cfg=settings(config);jobs=globals()['job_ids'](root,job_ids,config);outputs=[]
    control_path=root/'models/neural_path_control.json'
    if cfg['neural_enabled']:
        control_signature=canonical_hash(dict(config=cfg,source=file_hash(Path(__file__)),device=device))
        if control_path.exists() and resume:
            control=read_json(control_path)
            if control.get('signature')!=control_signature:raise ValueError('Neural control resume source/config/device mismatch')
        else:
            control=known_learnable_control(cfg,device=device);control['signature']=control_signature
            atomic_json(control_path,control)
        if not control['passed']: raise RuntimeError('Shared neural path failed known learnable control; no EEG neural fit authorized')
    else: control=dict(status='unavailable',reason='neural fitting explicitly disabled')
    for job in jobs:
        panel,manifest=load_panel(root,job);directory=root/'models'/job;signature=_signature(config,manifest,root,job)
        marker=directory/'manifest.json'
        if marker.exists():
            if not resume: raise FileExistsError('Model job exists; use resume: '+job)
            old=read_json(marker)
            if old['signature']!=signature: raise ValueError('Model resume config/source/input mismatch')
            for record in old['files']:
                if file_hash(root/record['path'])!=record['sha256']: raise ValueError('Corrupt fitted artifact')
            outputs.append(dict(id=job,status='complete',path=str(marker.relative_to(root)),resumed=True));continue
        tr=panel['train'];dv=panel['development'];y=first_targets(tr);weights=circuit_weights(tr['structure_id']);base=weights@y
        candidates=[dict(id='train_mean',kind='mean'),dict(id='train_mean_unit_variance',kind='mean_unit')]
        family=manifest.get('target_family',manifest.get('target_spec',{}).get('family',''))
        if not family: family=job.split('__',1)[-1].split('_l')[0]
        if family in ('T0','T1'):candidates.append(dict(id='persistence',kind='persistence'))
        else: candidates.append(dict(id='persistence',kind='unsupported',reason='No predefined persistence mapping for scalar/Fourier target'))
        candidates += [dict(id=f'ridge_l{lag}_a{alpha:g}',kind='ridge',lag=lag,alpha=alpha) for lag in cfg['ridge_lags'] for alpha in cfg['ridge_alphas']]
        candidates += [dict(id=f'state_rank{rank}',kind='state',rank=rank,alpha=cfg['state_alpha']) for rank in cfg['state_ranks']]
        if cfg['neural_enabled']:
            candidates += [dict(id=f'mlp_{loss}_h{hidden}_seed{seed}',kind='neural',loss=loss,hidden=hidden,seed=seed)
                for loss in cfg['neural_losses'] for hidden in cfg['neural_hidden'] for seed in cfg['neural_seeds']]
        records=[];files=[];winner=None
        if cfg['neural_enabled']:
            few_report,few_arrays=few_example_diagnostic(tr,cfg,device)
            atomic_json(directory/'few_example_diagnostic.json',few_report)
            atomic_npz(directory/'few_example_diagnostic.npz',**few_arrays)
            files.extend([_record(root,directory/'few_example_diagnostic.json'),_record(root,directory/'few_example_diagnostic.npz')])
        for candidate in candidates:
            name=candidate['id'];kind=candidate['kind'];state={};account={};net=None
            if kind=='unsupported': records.append(dict(**candidate,status='unavailable'));continue
            if kind=='ridge':
                lag=min(candidate['lag'],tr['X'].shape[1]);state=fit_ridge(tr['X'][:,-lag:],y,candidate['alpha'],weights)
                account=dict(effective_lags=lag,declared_lags=candidate['lag'])
            elif kind=='state': state=_state_fit(tr,y,candidate['rank'],candidate['alpha'])
            elif kind=='neural':net,state,account=train_neural(tr,dv,cfg,hidden=candidate['hidden'],gaussian=candidate['loss']=='gaussian',seed=candidate['seed'],device=device)
            else: state={'training_mean':base}
            raw={};variances={}
            for split,data in panel.items():
                if kind in ('mean','mean_unit'):mu=np.broadcast_to(base,(len(data['X']),len(base))).copy();var=None
                elif kind=='ridge':mu=predict_ridge(state,data['X'][:,-account['effective_lags']:]);var=None
                elif kind=='state':mu=predict_ridge(state,_state_features(state,data['X']));var=None
                elif kind=='persistence':
                    physical=np.zeros((len(data['X']),len(base))) if family=='T1' else np.repeat(data['X_physical'][:,-1,None],len(base),axis=1)
                    norm=manifest['normalization'];mu=(physical-np.asarray(norm['y_offset']))/norm['y_scale'];var=None
                else:mu,var=neural_predict(net,data,account['lag'],device)
                raw[split]=mu;variances[split]=var
            shrink=1. if kind in ('mean','mean_unit','persistence') else residual_shrinkage(dv,raw['development'],base)
            means={s:base+shrink*(p-base) for s,p in raw.items()}
            raw_residual=weights@((y-means['train'])**2)
            raw_residual=np.maximum(raw_residual,cfg['variance_floor'])
            for split in panel:
                if variances[split] is None:
                    variances[split]=np.broadcast_to(np.ones(len(base)) if kind=='mean_unit' else raw_residual,means[split].shape).copy()
            multiplier=calibrate_variance(dv,means['development'],variances['development'],cfg['variance_floor'])
            matched_variance=matched_calibrated_variance(dv,means['development'],cfg['variance_floor'])
            state.update(residual_shrinkage=np.asarray(shrink),variance_multiplier=multiplier,training_residual_variance=raw_residual,matched_calibration_variance=matched_variance)
            candidate_dir=directory/name
            atomic_npz(candidate_dir/'state.npz',**state);files.append(_record(root,candidate_dir/'state.npz'))
            scores={}
            for split,data in panel.items():
                atomic_npz(candidate_dir/f'{split}.npz',mean=means[split],mean_raw=raw[split],
                    variance_uncalibrated=variances[split],variance_calibrated=np.maximum(variances[split]*multiplier,cfg['variance_floor']),
                    variance_matched_calibrated=np.broadcast_to(matched_variance,means[split].shape).copy(),
                    context_id=data['context_id'],structure_id=data['structure_id'],baseline_mean=base,
                    target_fingerprint=np.asarray(manifest['target_fingerprint']),risk_contract_json=np.asarray(json.dumps(manifest['risk_contract'],sort_keys=True)))
                files.append(_record(root,candidate_dir/f'{split}.npz'))
                if split!='evaluation':
                    scores[split]=dict(mse=circuit_loss(data,means[split]),raw_mse=circuit_loss(data,raw[split]))
            # Evaluation outcomes are not scored until the separate score stage.
            dev_score=scores['development']['mse']
            if winner is None or dev_score<winner[0]-1e-12: winner=(dev_score,name)
            record=dict(**candidate,status='complete',development_mse=dev_score,split_scores=scores,
                shrinkage=shrink,shrinkage_fit_split='development',accounting=account,
                state_file=str((candidate_dir/'state.npz').relative_to(root)),
                variance_calibration='calibrated: identical development-only multiplicative access, retaining raw heteroscedasticity; not identical final variance family',
                matched_variance_calibration='matched_calibrated: identical context-independent diagonal family for every mean, fitted on development only; primary matched-family comparison',
                readout_scope='EEG-only; no privileged simulator or generator state',
                persistence_scope='last acquired sample held constant, not native persistence unless input A1' if kind=='persistence' else None)
            atomic_json(candidate_dir/'candidate.json',record);files.append(_record(root,candidate_dir/'candidate.json'));records.append(record)
        record=dict(status='complete',signature=signature,job_id=job,source_sha256=file_hash(Path(__file__)),
            target_fingerprint=manifest['target_fingerprint'],risk_contract=manifest['risk_contract'],
            selected_candidate=winner[1],selection_split='development',selection_metric='equal-circuit MSE',
            selection_does_not_use_evaluation=True,mean_fallback_retained=True,candidates=records,files=files,
            complete_ridge_candidate_count=sum(x['kind']=='ridge' for x in records),
            model_qualification=dict(completed=True,known_neural_path_passed=control.get('passed'),positive_evaluation_skill_required=False),
            training_target_policy='first available scoring future per context; equal circuit then context weights',
            repeated_optimizer_seeds='fitting repeats, not independent N; no H5/H6 objective superiority claim',
            development_reuse='checkpoint, shrinkage, variance calibration and candidate selection reuse development; prospective final circuits remain separate',
            inspected_panel_scope='exploratory if these circuits were previously inspected')
        atomic_json(marker,record);outputs.append(dict(id=job,status='complete',path=str(marker.relative_to(root)),resumed=False))
    result=dict(status='complete',jobs=outputs,scientific_question='H1 prediction-risk accounting; H4 target/prediction scale',
                neural_path_control=control,requested_jobs=jobs)
    atomic_json(root/'models/manifest.json',result);return result
