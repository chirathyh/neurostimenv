"""Full-history EEG predictors with separate mean selection and calibration.

No evaluation or calibration outcomes choose a candidate/checkpoint/shrinkage.
Numeric states are pickle-free. A neural family comprises five fitted repeats;
its inference unit is the circuit after averaging seed-specific losses.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
from ..artifacts import atomic_npz
from ..targets import canonical_hash
from ...storage import atomic_json, file_hash, read_json
from ...training import feature_matrix
from ..v4.models import predict_ridge, job_ids as available_jobs
from ..v4.artifacts import environment

DEFAULTS = dict(ridge_lags=[16,64], ridge_alphas=[.01,.1,1.,10.,100.,1000.,10000.],
    summary_alphas=[.01,.1,1.,10.,100.,1000.,10000.], pca_width=64,
    fourier_frequencies_hz=np.arange(4.,30.01,.5).tolist(), sample_rate_hz=500.,
    full_history_samples=2000, short_history_samples=64, neural_width=16,
    neural_dilations=[1,2,4,8,16,32,64,128,256,512], neural_seeds=[0,1,2,3,4],
    neural_enabled=True, max_steps=10000, patience=500, eval_every=50,
    batch_size=32, learning_rate=.001, weight_decay=.01, max_parameters=50000,
    dtype='float32', threads=1, variance_floor=1e-6, fp64_sensitivity=True,
    run_controls=True, control_steps=300)


def settings(config):
    out=copy.deepcopy(DEFAULTS); out.update(config.get('models',{}))
    if out['ridge_lags'] != [16,64] or list(out['ridge_alphas']) != DEFAULTS['ridge_alphas']:
        raise ValueError('The complete 14-candidate tail ridge grid must be preserved')
    if out['dtype'] not in ('float32','float64'): raise ValueError('Only float32/float64 training supported')
    if not 1 <= int(out['max_steps']) <= 10000: raise ValueError('Optimization bound is 1..10000 updates')
    if not 1 <= int(out['max_parameters']) <= 50000: raise ValueError('TCN capacity must be <=50000')
    if config.get('claim_status')!='engineering' and (out['full_history_samples']!=2000 or out['short_history_samples']!=64):
        raise ValueError('Scientific history comparison is fixed to 2000 versus 64 samples')
    if out['neural_enabled'] and config.get('claim_status') != 'engineering' and len(set(out['neural_seeds'])) != 5:
        raise ValueError('The prospective neural policy requires exactly five distinct fitting repeats')
    if 1+2*sum(out['neural_dilations']) < out['full_history_samples']:
        raise ValueError('Convolutional receptive field cannot span the declared full history')
    if not out['summary_alphas'] or min(out['summary_alphas'])<=0:raise ValueError('Positive summary regularization required')
    if int(out['eval_every'])<1 or int(out['patience'])<1 or int(out['batch_size'])<1: raise ValueError('Invalid bounded optimizer settings')
    return out


def load_panel(root, job_id):
    from .derivation import load_panel as load
    panel,manifest=load(root,job_id)
    for split in ('train','development'):
        if split not in panel: raise ValueError('Missing panel split '+split)
    for a in panel:
        for b in panel:
            if a<b and (set(panel[a]['structure_id'])&set(panel[b]['structure_id']) or set(panel[a]['context_id'])&set(panel[b]['context_id'])):
                raise ValueError('Circuit/context leakage between '+a+' and '+b)
    return panel,manifest


def context_weights(data):
    """Equal circuits, then conditions, then available prefixes, never futures."""
    groups=np.asarray(data['structure_id']);ctx=np.asarray(data['context_id']).astype(str)
    conditions=np.asarray([s.rsplit('-R',1)[1] if '-R' in s else 'one_condition' for s in ctx])
    circuits=np.unique(groups);w=np.zeros(len(groups),float)
    for c in circuits:
        regimes=np.unique(conditions[groups==c])
        for r in regimes:
            take=(groups==c)&(conditions==r);w[take]=1/(len(circuits)*len(regimes)*take.sum())
    return w


def first_targets(data):
    mask=np.asarray(data['future_mask'],bool)
    if np.any(mask.sum(1)==0): raise ValueError('Empty scoring context')
    return np.asarray(data['Y'],float)[np.arange(len(mask)),np.argmax(mask,axis=1)]


def target_mean(data):
    mask=np.asarray(data['future_mask'],bool)
    return np.where(mask[...,None],data['Y'],0).sum(1)/mask.sum(1)[:,None]


def context_risk(data, mean):
    mask=np.asarray(data['future_mask'],bool)
    return np.where(mask,np.mean((np.asarray(data['Y'],float)-np.asarray(mean,float)[:,None])**2,-1),0).sum(1)/mask.sum(1)


def circuit_loss(data,mean): return float(context_weights(data)@context_risk(data,mean))


def residual_shrinkage(data,prediction,baseline):
    delta=np.asarray(prediction,float)-baseline; y=target_mean(data)-baseline;w=context_weights(data)[:,None]
    den=float(np.sum(w*delta**2))
    return float(np.clip(np.sum(w*delta*y)/den,0,1)) if den>0 else 0.


def matched_variance(data,mean,floor=1e-6):
    mask=np.asarray(data['future_mask'],bool);e=(np.asarray(data['Y'],float)-np.asarray(mean,float)[:,None])**2
    per=np.where(mask[...,None],e,0).sum(1)/mask.sum(1)[:,None]
    return np.maximum(context_weights(data)@per,float(floor))


def configure_torch(device='cpu',dtype='float32',threads=1):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no silent device substitution')
    torch.set_num_threads(int(threads));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if hasattr(torch,'set_float32_matmul_precision'):torch.set_float32_matmul_precision('highest')
    return torch, getattr(torch,dtype)


def make_tcn(output_dimension,baseline,cfg,seed=0,device='cpu'):
    torch,dtype=configure_torch(device,cfg['dtype'],cfg['threads']);nn=torch.nn
    torch.manual_seed(int(seed))
    if device.startswith('cuda'):torch.cuda.manual_seed_all(int(seed))
    class Block(nn.Module):
        def __init__(self,width,dilation):
            super().__init__();self.dilation=dilation;self.conv=nn.Conv1d(width,width,3,dilation=dilation)
        def forward(self,x):
            return x+.1*torch.nn.functional.gelu(self.conv(torch.nn.functional.pad(x,(2*self.dilation,0))))
    class ResidualTCN(nn.Module):
        def __init__(self):
            super().__init__();width=int(cfg['neural_width'])
            self.input=nn.Conv1d(1,width,1);self.blocks=nn.Sequential(*[Block(width,int(d)) for d in cfg['neural_dilations']])
            self.head=nn.Linear(width,output_dimension)
            self.register_buffer('training_mean',torch.tensor(baseline,dtype=dtype))
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        def forward(self,x):
            z=self.blocks(torch.nn.functional.gelu(self.input(x[:,None])))
            return self.training_mean+self.head(z.mean(-1))
    net=ResidualTCN().to(device=device,dtype=dtype)
    if sum(v.numel() for v in net.parameters())>cfg['max_parameters']:raise ValueError('TCN parameter budget exceeded')
    return net


def neural_predict(net,x,cfg,device):
    torch,dtype=configure_torch(device,cfg['dtype'],cfg['threads']);out=[]
    net.eval()
    with torch.no_grad():
        for i in range(0,len(x),int(cfg['batch_size'])):
            out.append(net(torch.as_tensor(x[i:i+cfg['batch_size']],dtype=dtype,device=device)).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def train_tcn(train,development,cfg,*,history_samples,seed=0,device='cpu'):
    """Same exact training path for synthetic controls and experimental candidates."""
    torch,dtype=configure_torch(device,cfg['dtype'],cfg['threads']);w=context_weights(train)
    y=first_targets(train);baseline=w@y;net=make_tcn(y.shape[-1],baseline,cfg,seed,device)
    x=np.asarray(train['X'])[:,-history_samples:];dx=np.asarray(development['X'])[:,-history_samples:]
    tx=torch.as_tensor(x,dtype=dtype,device=device);ty=torch.as_tensor(y,dtype=dtype,device=device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=float(cfg['learning_rate']),weight_decay=float(cfg['weight_decay']))
    rng=np.random.default_rng(seed);history=[];best=None;stale=0;updates=0;examples=0;t0=time.perf_counter()
    for step in range(int(cfg['max_steps'])+1):
        if step%int(cfg['eval_every'])==0 or step==cfg['max_steps']:
            pred=neural_predict(net,dx,cfg,device);metric=circuit_loss(development,pred)
            train_mse=circuit_loss(train,neural_predict(net,x,cfg,device))
            history.append(dict(step=step,development_mse=metric,training_mse=train_mse,examples_seen=examples,elapsed_seconds=time.perf_counter()-t0))
            if best is None or metric<best[0]-1e-12:
                best=(metric,step,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()});stale=0
            else:stale+=cfg['eval_every']
            if stale>=cfg['patience']:break
        if step==cfg['max_steps']:break
        indices=rng.choice(len(x),min(cfg['batch_size'],len(x)),replace=True,p=w)
        net.train();optimizer.zero_grad(set_to_none=True)
        loss=(net(tx[indices])-ty[indices]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TCN loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();updates+=1;examples+=len(indices)
    net.load_state_dict(best[2]);net.eval()
    state={'tensor__'+k:v.numpy() for k,v in best[2].items()}
    account=dict(parameter_count=sum(p.numel() for p in net.parameters()),selected_step=best[1],optimizer_updates=updates,
        learning_curve=history,examples_seen=examples,elapsed_seconds=time.perf_counter()-t0,history_samples=int(history_samples),
        convolution_receptive_field_samples=1+2*sum(cfg['neural_dilations']),pooling='mean over every available causal history position',
        actual_input_samples=int(x.shape[1]),selection_split='development',training_context_count=len(x),
        training_circuit_count=len(np.unique(train['structure_id'])),training_context_ids=np.asarray(train['context_id']).tolist(),
        training_structure_ids=np.unique(train['structure_id']).tolist(),seed=int(seed),device=device,dtype=cfg['dtype'],
        amp=False,tf32=False,weight_decay=cfg['weight_decay'])
    return net,state,account


def _slice(data,idx):
    n=len(data['X']);return {k:np.asarray(v)[idx] if k not in ('input_time_ms','target_time_ms') and np.asarray(v).ndim and len(v)==n else v for k,v in data.items()}


def known_learnable_control(cfg=None,device='cpu'):
    cfg=copy.deepcopy(DEFAULTS if cfg is None else cfg);cfg.update(max_steps=int(cfg['control_steps']),patience=int(cfg['control_steps']),eval_every=20,learning_rate=.01)
    rng=np.random.default_rng(90127);n=96;length=int(cfg['full_history_samples']);level=rng.uniform(-1,1,n)
    x=level[:,None]+rng.normal(0,.05,(n,length));y=(.8*x.mean(1))[:,None]
    def part(start,stop,offset):
        return dict(X=x[start:stop],Y=y[start:stop,None],future_mask=np.ones((stop-start,1),bool),
            structure_id=np.repeat(np.arange(offset,offset+2),(stop-start)//2),context_id=np.array([f'control-{i}' for i in range(start,stop)]))
    tr=part(0,64,0);dv=part(64,80,2);ev=part(80,96,4)
    net,_,account=train_tcn(tr,dv,cfg,history_samples=length,seed=19,device=device)
    pred=neural_predict(net,ev['X'],cfg,device);base=np.broadcast_to(first_targets(tr).mean(0),pred.shape)
    changed=ev['X'].copy();changed[:,:length//2]+=1.
    perturbation=float(np.max(np.abs(neural_predict(net,changed,cfg,device)-pred)))
    risk=circuit_loss(ev,pred);r0=circuit_loss(ev,base)
    return dict(status='complete',passed=bool(risk<.1*r0 and perturbation>1e-6),risk=risk,baseline_risk=r0,
        improvement=1-risk/r0,early_half_perturbation_max_change=perturbation,same_neural_path=True,accounting=account,
        interpretation='engineering deterministic control, not biophysical evidence')


def few_example_diagnostic(train,cfg,device='cpu'):
    sub=_slice(train,np.arange(min(4,len(train['X']))));opt=copy.deepcopy(cfg)
    opt.update(max_steps=min(300,opt['max_steps']),patience=min(300,opt['max_steps']),learning_rate=.01)
    net,state,account=train_tcn(sub,sub,opt,history_samples=cfg['full_history_samples'],seed=97,device=device)
    pred=neural_predict(net,sub['X'][:,-cfg['full_history_samples']:],opt,device)
    return dict(status='complete',actual_context_count=len(sub['X']),actual_circuit_count=len(np.unique(sub['structure_id'])),
        context_ids=sub['context_id'].tolist(),circuit_ids=np.unique(sub['structure_id']).tolist(),mse=circuit_loss(sub,pred),
        selection='same training subset; optimization diagnostic, not validation',accounting=account),dict(prediction=pred,truth=first_targets(sub),**state)


def _features(kind,x,state,cfg):
    if kind=='ridge':return np.asarray(x,float)[:,-int(state['history_samples']):]
    if kind=='feature18':return feature_matrix(np.asarray(x,float)[:,None],cfg['sample_rate_hz'])
    if kind=='pca':return (np.asarray(x,float)-state['pca_center'])@state['pca_components'].T
    if kind=='fourier':
        t=(np.arange(x.shape[1])-x.shape[1]+1)/float(cfg['sample_rate_hz']);f=np.asarray(cfg['fourier_frequencies_hz'])
        basis=np.concatenate([np.cos(2*np.pi*f[:,None]*t),np.sin(2*np.pi*f[:,None]*t)],axis=0)/np.sqrt(x.shape[1]/2)
        return np.asarray(x,float)@basis.T
    raise ValueError('Unknown feature '+kind)


def _record(root,path):return dict(path=Path(path).relative_to(root).as_posix(),sha256=file_hash(path),bytes=Path(path).stat().st_size)


def _predict(spec,state,data,cfg,manifest,device):
    kind=spec['kind'];base=state['baseline_mean'];x=data['X']
    if kind=='mean':raw=np.broadcast_to(base,(len(x),len(base))).copy()
    elif kind=='spectral_persistence':
        xx=np.asarray(data['X_physical'],float)[:,-int(cfg['sample_rate_hz']):]
        n=xx.shape[1];freq=np.fft.rfftfreq(n,1/cfg['sample_rate_hz']);target=manifest['target_spec']
        lo,hi=target['band_hz'];closed=target.get('include_upper',target.get('right_endpoint_included',False))
        mask=(freq>=lo)&((freq<=hi) if closed else (freq<hi))
        physical=(2*np.sum(np.abs(np.fft.rfft(xx,axis=1)[:,mask])**2,axis=1)/n**2)[:,None]
        if target['family']=='T4_log_energy':
            physical=np.log(np.maximum(physical,target['log_floor_v2'])/target['log_reference_v2'])
        norm=manifest['normalization'];raw=(physical-np.asarray(norm['y_offset']))/norm['y_scale']
    elif kind=='persistence':
        physical=np.repeat(data['X_physical'][:,-1,None],len(base),axis=1)
        norm=manifest['normalization'];raw=(physical-np.asarray(norm['y_offset']))/norm['y_scale']
    elif kind=='neural':
        torch,_=configure_torch(device,cfg['dtype'],cfg['threads']);net=make_tcn(len(base),base,cfg,spec['seed'],device)
        net.load_state_dict({k.removeprefix('tensor__'):torch.as_tensor(v) for k,v in state.items() if k.startswith('tensor__')})
        raw=neural_predict(net,x[:,-spec['history_samples']:],cfg,device)
    else:raw=predict_ridge(state,_features(kind,x,state,cfg))
    return base+float(state.get('residual_shrinkage',1.))*(raw-base),raw


def _save_prediction(path,data,mu,raw,state,manifest):
    shape=mu.shape
    atomic_npz(path,mean=np.asarray(mu,np.float64),mean_raw=np.asarray(raw,np.float64),
        variance_uncalibrated=np.broadcast_to(state['raw_variance'],shape).copy(),
        variance_development_calibrated=np.broadcast_to(state['development_variance'],shape).copy(),
        variance_calibrated=np.broadcast_to(state['development_variance'],shape).copy(),
        baseline_mean=state['baseline_mean'],context_id=data['context_id'],structure_id=data['structure_id'],
        target_fingerprint=np.asarray(manifest['target_fingerprint']),risk_contract_json=np.asarray(json.dumps(manifest['risk_contract'],sort_keys=True)))


def fit(config,root,jobs=None,device='cpu',resume=False):
    root=Path(root);cfg=settings(config);ids=available_jobs(root,jobs,config);outputs=[]
    dependencies={str(p.relative_to(Path(__file__).parents[2])):file_hash(p) for p in [Path(__file__),Path(__file__).parents[1]/'v4/models.py',Path(__file__).parents[2]/'training.py']}
    env=environment();numeric_environment={k:env[k] for k in ('python','versions')}
    numeric_environment['requested_device']=device
    if device.startswith('cuda'):
        torch,_=configure_torch(device,cfg['dtype'],cfg['threads']);numeric_environment['device_name']=torch.cuda.get_device_name(device)
        numeric_environment['torch_cuda_version']=torch.version.cuda
    control=dict(status='unavailable',reason='neural fitting disabled')
    if cfg['neural_enabled'] and cfg['run_controls']:
        controlpath=root/'models/neural_path_control.json';sig=canonical_hash(dict(cfg=cfg,device=device,environment=numeric_environment,dependencies=dependencies,source=file_hash(Path(__file__))))
        if resume and controlpath.exists():
            control=read_json(controlpath)
            if control.get('signature')!=sig:raise ValueError('Control resume device/dtype/source mismatch')
        else:
            control=known_learnable_control(cfg,device);control['signature']=sig;atomic_json(controlpath,control)
        if not control['passed']:raise RuntimeError('Known learnable full-history path failed; no EEG neural training')
    for job in ids:
        panel,manifest=load_panel(root,job);tr=panel['train'];dv=panel['development'];directory=root/'models'/job
        if tr['X'].shape[1]!=cfg['full_history_samples']:raise ValueError('Derived history differs from declared full-history samples')
        sig=canonical_hash(dict(cfg=cfg,device=device,environment=numeric_environment,dependencies=dependencies,source=file_hash(Path(__file__)),target=manifest['target_fingerprint'],
            train=file_hash(root/'derived'/job/'train.npz'),development=file_hash(root/'derived'/job/'development.npz')))
        marker=directory/'manifest.json';guard=directory/'fit_identity.json'
        if guard.exists() and (not resume or read_json(guard)['signature']!=sig):raise ValueError('Fit resume identity mismatch; source/device/dtype and training inputs must match')
        atomic_json(guard,dict(signature=sig,device=device,dtype=cfg['dtype'],settings=cfg,numerical_environment=numeric_environment,analysis_dependencies=dependencies))
        if marker.exists():
            saved=read_json(marker)
            if not resume or saved['signature']!=sig:raise ValueError('Existing model output requires matching resume')
            for rec in saved['files']:
                if file_hash(root/rec['path'])!=rec['sha256']:raise ValueError('Fitted artifact checksum mismatch')
            outputs.append(dict(id=job,status='complete',resumed=True,path=str(marker.relative_to(root))));continue
        weights=context_weights(tr);y=first_targets(tr);base=weights@y
        specs=[dict(id='train_mean',kind='mean',primary_eligible=True)]
        family=manifest.get('target_family','')
        if family in ('T4_energy','T4_log_energy'):
            specs.append(dict(id='acquired_history_spectral_persistence',kind='spectral_persistence',primary_eligible=False,
                interpretation='last 1-second acquired history band energy; affected by acquisition, not native-input oracle'))
        if family=='T0':specs.append(dict(id='persistence',kind='persistence',primary_eligible=False))
        specs += [dict(id=f'ridge_tail{lag}_a{a:g}',kind='ridge',history_samples=lag,alpha=a,primary_eligible=False) for lag in cfg['ridge_lags'] for a in cfg['ridge_alphas']]
        for kind in ('ridge','pca','fourier','feature18'):
            specs += [dict(id=f'{kind}_full_a{a:g}',kind=kind,history_samples=cfg['full_history_samples'],alpha=a,primary_eligible=True) for a in cfg['summary_alphas']]
        if cfg['neural_enabled']:
            specs += [dict(id=f'tcn_h{h}_seed{s}',kind='neural',history_samples=h,seed=s,primary_eligible=h==cfg['full_history_samples']) for h in (cfg['short_history_samples'],cfg['full_history_samples']) for s in cfg['neural_seeds']]
        records=[];files=[];feature_cache={}
        if cfg['neural_enabled']:
            diag,arrays=few_example_diagnostic(tr,cfg,device);atomic_json(directory/'few_example_diagnostic.json',diag);atomic_npz(directory/'few_example_diagnostic.npz',**arrays)
            files.extend([_record(root,directory/'few_example_diagnostic.json'),_record(root,directory/'few_example_diagnostic.npz')])
        for spec in specs:
            dest=directory/spec['id'];cm=dest/'candidate.json'
            if resume and cm.exists():
                row=read_json(cm)
                if row['signature']!=sig:raise ValueError('Candidate resume identity mismatch')
                for rec in row['files']:
                    if file_hash(root/rec['path'])!=rec['sha256']:raise ValueError('Candidate checksum mismatch')
                records.append(row);files.extend(row['files']);files.append(_record(root,cm));continue
            state=dict(baseline_mean=base,history_samples=np.asarray(spec.get('history_samples',cfg['full_history_samples'])));account={};t0=time.perf_counter()
            if spec['kind']=='neural':
                net,ns,account=train_tcn(tr,dv,cfg,history_samples=spec['history_samples'],seed=spec['seed'],device=device);state.update(ns)
            elif spec['kind'] not in ('mean','persistence','spectral_persistence'):
                feature_key=(spec['kind'],spec['history_samples'])
                if feature_key not in feature_cache:
                    fs={}
                    if spec['kind']=='pca':
                        center=weights@tr['X'];_,_,vt=np.linalg.svd((tr['X']-center)*np.sqrt(weights*len(y))[:,None],full_matrices=False)
                        width=min(cfg['pca_width'],len(y)-1,tr['X'].shape[1]);fs.update(pca_center=center,pca_components=vt[:width])
                    fs['history_samples']=np.asarray(spec['history_samples']);xx=_features(spec['kind'],tr['X'],fs,cfg)
                    offset=weights@xx;scale=np.sqrt(weights@((xx-offset)**2));scale=np.where(scale>1e-12,scale,1.)
                    zz=(xx-offset)/scale;intercept=weights@y;sw=np.sqrt(weights*len(y))
                    u,ss,vt=np.linalg.svd(zz*sw[:,None],full_matrices=False)
                    feature_cache[feature_key]=(fs,offset,scale,intercept,ss,vt,u.T@((y-intercept)*sw[:,None]),xx.shape[1])
                fs,offset,scale,intercept,ss,vt,uy,width=feature_cache[feature_key]
                state.update(fs,offset=offset,scale=scale,intercept=intercept,coefficient=(vt.T*(ss/(ss*ss+spec['alpha'])))@uy)
                account['feature_width']=width;account['training_only_factorization_reused_across_alphas']=True
            raw={s:_predict(spec,state,panel[s],cfg,manifest,device)[1] for s in ('train','development')}
            shrink=1. if spec['kind'] in ('mean','persistence','spectral_persistence') else residual_shrinkage(dv,raw['development'],base)
            state['residual_shrinkage']=np.asarray(shrink);means={s:base+shrink*(v-base) for s,v in raw.items()}
            state['raw_variance']=matched_variance(tr,means['train'],cfg['variance_floor']);state['development_variance']=matched_variance(dv,means['development'],cfg['variance_floor'])
            atomic_npz(dest/'state.npz',**state);cfiles=[_record(root,dest/'state.npz')]
            for split in ('train','development'):
                _save_prediction(dest/f'{split}.npz',panel[split],means[split],raw[split],state,manifest);cfiles.append(_record(root,dest/f'{split}.npz'))
            row=dict(**spec,status='complete',signature=sig,accounting=account,device=device,dtype=cfg['dtype'],
                development_mse=circuit_loss(dv,means['development']),train_mse=circuit_loss(tr,means['train']),
                shrinkage=shrink,selection_split='development',calibration_outcomes_accessed=False,evaluation_outcomes_accessed=False,
                elapsed_seconds=time.perf_counter()-t0,files=cfiles,state_file=str((dest/'state.npz').relative_to(root)))
            atomic_json(cm,row);records.append(row);files+=cfiles+[_record(root,cm)]
        policies=[dict(id=r['id'],members=[r['id']],development_mse=r['development_mse']) for r in records if r['primary_eligible'] and r['kind']!='neural']
        neural=[r for r in records if r['primary_eligible'] and r['kind']=='neural']
        if neural:policies.append(dict(id='tcn_full_five_seed_policy',members=[r['id'] for r in neural],development_mse=float(np.mean([r['development_mse'] for r in neural]))))
        selected=min(policies,key=lambda r:r['development_mse'])
        selection=dict(status='frozen',signature=sig,selected_candidate=selected['id'],selected_member_ids=selected['members'],
            policies=policies,selection='minimum equal-circuit development MSE; neural repeats averaged as risks, not ensemble predictions',
            selection_split='development',calibration_selection=False,evaluation_selection=False)
        atomic_json(directory/'selection.json',selection);files.append(_record(root,directory/'selection.json'))
        sensitivity=[]
        if cfg['fp64_sensitivity'] and cfg['dtype']=='float32':
            for rec in neural:
                if rec['id'] not in selected['members']:continue
                dcfg=copy.deepcopy(cfg);dcfg['dtype']='float64';net,state,account=train_tcn(tr,dv,dcfg,history_samples=cfg['full_history_samples'],seed=rec['seed'],device=device)
                dest=directory/'fp64_sensitivity'/rec['id']
                p=neural_predict(net,dv['X'],dcfg,device);sh=residual_shrinkage(dv,p,base)
                values={s:base+sh*(neural_predict(net,panel[s]['X'],dcfg,device)-base) for s in ('train','development')}
                state.update(baseline_mean=base,history_samples=np.asarray(cfg['full_history_samples']),residual_shrinkage=np.asarray(sh),
                    raw_variance=matched_variance(tr,values['train'],cfg['variance_floor']),development_variance=matched_variance(dv,values['development'],cfg['variance_floor']))
                atomic_npz(dest/'state.npz',**state)
                for split,value in values.items():atomic_npz(dest/f'{split}.npz',mean=value,context_id=panel[split]['context_id'],structure_id=panel[split]['structure_id'])
                sensitivity.append(dict(candidate=rec['id'],status='complete',dtype='float64',accounting=account,development_mse=circuit_loss(dv,values['development']),
                    selection_eligible=False,state_file=str((dest/'state.npz').relative_to(root)),settings=dcfg))
                files.extend(_record(root,p) for p in sorted(dest.glob('*.npz')))
        if not neural or not any(r['id'] in selected['members'] for r in neural):
            sensitivity=[dict(status='complete',reason='Selected linear/mean fits and all numerical scores already use float64')]
        elif cfg['dtype']=='float64':
            sensitivity=[dict(status='complete',reason='Primary neural training already uses float64')]
        elif not cfg['fp64_sensitivity']:
            sensitivity=[dict(status='unrun',reason='Paired float64 sensitivity disabled explicitly')]
        result=dict(status='complete',signature=sig,job_id=job,target_fingerprint=manifest['target_fingerprint'],risk_contract=manifest['risk_contract'],
            selected_candidate=selected['id'],selected_member_ids=selected['members'],selection_split='development',candidates=records,policies=policies,
            complete_ridge_candidate_count=14,full_history_sample_count=cfg['full_history_samples'],mean_fallback_retained=True,
            fp64_sensitivity=sensitivity,device=device,dtype=cfg['dtype'],numerical_environment=numeric_environment,files=files,
            calibration_policy='separate calibration stage; development variance is historical-access comparison only',
            primary_seed_policy='Average member-specific losses within circuit; do not average predictions then square',
            prediction_stage='calibration/evaluation predictions generated after immutable mean selection record')
        atomic_json(marker,result);outputs.append(dict(id=job,status='complete',resumed=False,path=str(marker.relative_to(root))))
    out=dict(status='complete',jobs=outputs,neural_path_control=control);atomic_json(root/'models/manifest.json',out)
    # This stage is separately fingerprinted: newly arriving calibration/test data
    # never change a selected mean or training/development artifact.
    out['inference']=predict(config,root,jobs=ids,device=device,resume=resume)
    return out



def prediction_record(root,job,path):
    """Resolve a numeric prediction under its immutable inference-stage manifest."""
    root=Path(root);relative=str(Path(path).relative_to(root));directory=root/'models'/job
    marker=directory/'inference_manifest.json'
    manifest=read_json(marker) if marker.exists() else read_json(directory/'manifest.json')
    record=next((r for r in manifest['files'] if r['path']==relative),None)
    if record is None:raise ValueError('Prediction not covered by frozen inference manifest')
    return record


def predict(config,root,jobs=None,device='cpu',resume=False):
    """Infer newly arrived calibration/evaluation panels using frozen numeric means.

    Append-only split arrival is allowed; replacing a previously inferred split,
    model, device, dtype, normalization or target contract is rejected.
    """
    root=Path(root);outputs=[]
    for job in available_jobs(root,jobs,config):
        panel,derived=load_panel(root,job);directory=root/'models'/job
        frozen=read_json(directory/'manifest.json');identity=read_json(directory/'fit_identity.json');cfg=identity['settings']
        if frozen['risk_contract']!=derived['risk_contract'] or frozen['target_fingerprint']!=derived['target_fingerprint']:
            raise ValueError('Frozen inference target/operator/normalization/population mismatch')
        if device!=frozen['device']:raise ValueError('Frozen inference device differs; use separately named sensitivity')
        for rec in frozen['files']:
            if file_hash(root/rec['path'])!=rec['sha256']:raise ValueError('Frozen mean file checksum mismatch')
        env=environment();current={k:env[k] for k in ('python','versions')}
        previous=identity.get('numerical_environment',{})
        if any(current[k]!=previous.get(k) for k in current):raise ValueError('Frozen inference numerical environment mismatch')
        inputs={split:file_hash(root/'derived'/job/(split+'.npz')) for split in ('calibration','evaluation') if split in panel}
        marker=directory/'inference_manifest.json'
        signature=canonical_hash(dict(fit=file_hash(directory/'manifest.json'),identity=file_hash(directory/'fit_identity.json'),
            source=file_hash(Path(__file__)),device=device,environment=current))
        old=None
        if marker.exists():
            old=read_json(marker)
            if not resume or old['signature']!=signature:raise ValueError('Frozen inference resume identity mismatch')
            if any(inputs.get(split)!=digest for split,digest in old['prediction_inputs'].items()):raise ValueError('Previously inferred split changed or disappeared')
            for record in old['files']:
                if file_hash(root/record['path'])!=record['sha256']:raise ValueError('Frozen prediction checksum mismatch')
        files=[] if old is None else old['files'];completed=set() if old is None else set(old['prediction_inputs'])
        candidates={r['id']:r for r in frozen['candidates']}
        for split in inputs:
            if split in completed:continue
            for rec in frozen['candidates']:
                with np.load(root/rec['state_file'],allow_pickle=False) as f:state={k:f[k] for k in f.files}
                mu,raw=_predict(rec,state,panel[split],cfg,derived,device)
                path=directory/rec['id']/(split+'.npz');_save_prediction(path,panel[split],mu,raw,state,derived);files.append(_record(root,path))
            for row in frozen['fp64_sensitivity']:
                if 'state_file' not in row:continue
                rec=candidates[row['candidate']]
                with np.load(root/row['state_file'],allow_pickle=False) as f:state={k:f[k] for k in f.files}
                mu,raw=_predict(rec,state,panel[split],row['settings'],derived,device)
                path=directory/'fp64_sensitivity'/rec['id']/(split+'.npz');_save_prediction(path,panel[split],mu,raw,state,derived);files.append(_record(root,path))
            completed.add(split)
            atomic_json(marker,dict(status='complete',signature=signature,files=files,prediction_inputs={k:inputs[k] for k in sorted(completed)},
                selected_candidate=frozen['selected_candidate'],selected_member_ids=frozen['selected_member_ids'],
                mean_manifest_sha256=file_hash(directory/'manifest.json'),mean_selection_changed=False,
                policy='append-only newly arrived split inference; no calibration/evaluation mean fitting'))
        outputs.append(dict(id=job,status='complete' if inputs else 'planned',splits=sorted(inputs),new_splits=sorted(set(inputs)-(set(old['prediction_inputs']) if old else set()))))
    out=dict(status='complete',jobs=outputs);atomic_json(root/'models/inference_manifest.json',out);return out


def predict_from_artifact(root,job_id,*,X_physical=None,X=None,candidate=None,device=None):
    """Apply frozen numeric means without future targets or refitting.

    Physical histories must already use the saved acquisition operator/grid.
    Retain the member axis: primary risks average seed losses.
    """
    root=Path(root);directory=root/'models'/job_id;fitmeta=read_json(directory/'manifest.json')
    identity=read_json(directory/'fit_identity.json');cfg=identity['settings']
    derived=read_json(root/'derived'/job_id/'manifest.json');norm=derived['normalization']
    if fitmeta['target_fingerprint']!=derived['target_fingerprint'] or fitmeta['risk_contract']!=derived['risk_contract']:
        raise ValueError('Frozen predictor risk contract differs from derived target')
    if (X is None)==(X_physical is None):raise ValueError('Supply exactly one of X or X_physical')
    raw=np.asarray(X_physical,float) if X_physical is not None else None
    x=(raw-np.asarray(norm['x_offset']))/float(norm['x_scale']) if raw is not None else np.asarray(X,float)
    if x.ndim!=2 or x.shape[1]!=cfg['full_history_samples'] or not np.isfinite(x).all():raise ValueError('Frozen predictor input shape/finite contract mismatch')
    if raw is None:raw=x*float(norm['x_scale'])+np.asarray(norm['x_offset'])
    name=fitmeta['selected_candidate'] if candidate is None else candidate
    policies={p['id']:p['members'] for p in fitmeta['policies']};members=policies.get(name,[name]);records={r['id']:r for r in fitmeta['candidates']}
    means=[];baseline=None;requested=fitmeta['device'] if device is None else device
    for member in members:
        if member not in records:raise ValueError('Unknown frozen candidate '+member)
        rec=records[member]
        if rec['kind']=='neural' and requested!=fitmeta['device']:raise ValueError('Frozen neural inference device differs; declare a separate sensitivity')
        state_record=next(r for r in rec['files'] if r['path']==rec['state_file'])
        if file_hash(root/rec['state_file'])!=state_record['sha256']:raise ValueError('Frozen numeric state checksum mismatch')
        with np.load(root/rec['state_file'],allow_pickle=False) as f:state={k:f[k] for k in f.files}
        mean,_=_predict(rec,state,dict(X=x,X_physical=raw),cfg,derived,requested);means.append(mean);baseline=state['baseline_mean']
    mean=np.stack(means).astype(np.float64);offset=np.asarray(norm['y_offset']);scale=float(norm['y_scale'])
    return dict(member_ids=members,mean_standardized=mean,mean_physical=mean*scale+offset,
        baseline_mean_physical=baseline*scale+offset,baseline_mean_standardized=baseline,
        normalization=norm,target_fingerprint=derived['target_fingerprint'],risk_contract=derived['risk_contract'],
        aggregation='member-specific risks averaged within circuit, not risk of ensemble mean')
