"""Bounded H1 baselines and matched neural objectives, with numeric artifacts.

Training/development select every fit before evaluation outcomes are scored.
No privileged simulator state, regime label or future enters a predictor.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import tempfile
import time
import warnings

import numpy as np
from ..storage import atomic_json,read_json,file_hash
from ..training import feature_matrix
from .targets import TargetSpec,operator_from_projected,canonical_hash
from .scoring import score_panel

DEFAULTS=dict(models=['train_mean','train_mean_unit_variance','persistence','ridge_wiener','harmonic','state_space','neural_mse','neural_gaussian'],seeds=[0,1],ridge_alphas=[.01,.1,1.],lags=[16,64],state_ranks=[2,4,8],harmonic_frequencies_hz=list(np.arange(4,30.01,.5)),harmonic_history_fractions=[.5,.75,1.],max_candidates=3,steps=500,batch_size=16,learning_rates=[.0003],weight_decay=.01,draws=64,diagnostic_steps=100,diagnostic_examples=4,representation_steps=200,parameter_tolerance=.15,threads=1)

MEAN_REFERENCES=('train_mean','train_mean_unit_variance')


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    clean={k:np.asarray(v) for k,v in arrays.items()}
    if any(a.dtype.kind=='O' for a in clean.values()):raise ValueError('Object arrays/pickle forbidden in primary artifacts')
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,**clean);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def _settings(config):
    out=copy.deepcopy(DEFAULTS);out.update(config.get('benchmark',config.get('h1',{}).get('benchmark',{})));return out


def _target_mean(data):
    y=np.asarray(data['Y'],float);m=np.asarray(data['future_mask'],bool)
    if np.any(m.sum(1)==0):raise ValueError('Training contexts require a future')
    return np.where(m[...,None],y,0).sum(1)/m.sum(1)[:,None]


def _circuit_mse(data,pred):
    m=np.asarray(data['future_mask'],bool);e=((data['Y']-pred[:,None])**2).mean(-1)
    context=np.where(m,e,0).sum(1)/m.sum(1)
    return float(np.mean([context[data['structure_id']==s].mean() for s in np.unique(data['structure_id'])]))


def _circuit_nll(data,mean,var):
    m=np.asarray(data['future_mask'],bool);v=var[:,None]
    e=.5*(np.log(2*np.pi*v)+(data['Y']-mean[:,None])**2/v).mean(-1)
    context=np.where(m,e,0).sum(1)/m.sum(1)
    return float(np.mean([context[data['structure_id']==s].mean() for s in np.unique(data['structure_id'])]))


def _validate_panel(panel):
    for split in ('train','development','evaluation'):
        d=panel[split]
        if d['X'].ndim!=2 or d['Y'].ndim!=3 or len(d['X'])!=len(d['Y']):raise ValueError('Panel shapes must be X[N,L],Y[N,M,D]')
        if not len(d['X']) or not np.isfinite(d['X']).all():raise ValueError('Nonempty finite EEG input required')
        if not np.isfinite(d['Y'][d['future_mask']]).all():raise ValueError('Nonfinite valid targets')
    for key in ('structure_id','context_id'):
        groups=[set(np.asarray(panel[s][key]).tolist()) for s in ('train','development','evaluation')]
        if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError(f'Circuit/context split overlap: {key}')
    if panel['manifest']['normalization'].get('fit_split')!='train':raise ValueError('Normalization must be fitted on training circuits only')


def _ridge_fit(x,y,alpha):
    x=np.asarray(x,float);y=np.asarray(y,float);offset=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-12,scale,1.)
    z=(x-offset)/scale;ym=y.mean(0)
    # SVD is stable for collinear lagged observations and N<D.
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    weight=(vt.T*(s/(s*s+float(alpha))))@u.T@(y-ym)
    return dict(offset=offset,scale=scale,weight=weight,intercept=ym)


def _ridge_predict(state,x):return ((x-state['offset'])/state['scale'])@state['weight']+state['intercept']


def _physical_wave(model,data,harmonic=False):
    spec=TargetSpec.from_dict(model.manifest['target_spec'])
    n=round(spec.duration_ms/spec.native_dt_ms) if spec.operator=='native_waveform' else round(spec.duration_ms*spec.output_grid_hz/1000)
    step=spec.native_dt_ms if spec.operator=='native_waveform' else 1000/spec.output_grid_hz
    target_t=(spec.lead_ms+np.arange(1,n+1)*step)/1000
    result=[]
    for x in data['X_physical']:
        if not harmonic:w=np.full(n,x[-1])
        else:
            fraction=model.settings.get('history_fraction',1.)
            length=max(4,round(len(x)*fraction));xx=x[-length:]
            t=np.asarray(data['input_time_ms'])[-length:]/1000
            best=None
            for f in model.state['frequencies']:
                basis=np.stack([np.ones(len(t)),np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t)],1)
                coef=np.linalg.lstsq(basis,xx,rcond=None)[0];error=float(((basis@coef-xx)**2).mean())
                if best is None or error<best[0]:best=(error,f,coef)
            _,f,coef=best
            if spec.operator=='native_waveform':
                cosine,sine=np.cos(2*np.pi*f*target_t),np.sin(2*np.pi*f*target_t)
            else:
                midpoint=target_t-step/2000
                cosine=np.cos(2*np.pi*f*midpoint)*np.sinc(f*step/1000)
                sine=np.sin(2*np.pi*f*midpoint)*np.sinc(f*step/1000)
            w=np.stack([np.ones(n),cosine,sine],1)@coef
        result.append(operator_from_projected(w,spec))
    norm=model.manifest['normalization']
    return (np.asarray(result)-np.asarray(norm['y_offset']))/norm['y_scale']


def _state_basis(state,x):
    lag=int(state['lags']);z=(x[:,-lag:]-state['delay_offset'])@state['components'].T
    horizon=int(state['lead_steps']);advanced=z@np.linalg.matrix_power(state['transition'],horizon)
    return np.concatenate([z,advanced],1)


class Predictor:
    def __init__(self,kind,state=None,settings=None,manifest=None,network=None):
        self.kind=kind;self.state=state or {};self.settings=settings or {};self.manifest=manifest or {};self.network=network

    def predict(self,data,device='cpu'):
        x=np.asarray(data['X'],float)
        if self.kind in MEAN_REFERENCES:mean=np.repeat(self.state['mean'][None],len(x),0)
        elif self.kind=='ridge_wiener':mean=_ridge_predict(self.state,x[:,-int(self.settings['lags']):])
        elif self.kind=='state_space':mean=_ridge_predict(self.state,_state_basis(self.state,x))
        elif self.kind=='persistence':mean=_physical_wave(self,data)
        elif self.kind=='harmonic':mean=_physical_wave(self,data,True)
        elif self.kind in ('neural_mse','neural_gaussian'):
            import torch
            means=[];variances=[];self.network.eval()
            with torch.no_grad():
                for start in range(0,len(x),64):
                    mu,var=self.network(torch.as_tensor(x[start:start+64,None],dtype=torch.float32,device=device));means.append(mu.cpu().numpy())
                    if var is not None:variances.append(var.cpu().numpy())
            mean=np.concatenate(means)
            if variances:return mean,np.concatenate(variances)
        else:raise ValueError(self.kind)
        return mean,np.broadcast_to(self.state.get('variance',np.ones(mean.shape[-1])),mean.shape).copy()

    def export(self,directory,metadata):
        directory=Path(directory);arrays={k:v for k,v in self.state.items()}
        if self.network is not None:arrays.update({'tensor__'+k:v.detach().cpu().numpy() for k,v in self.network.state_dict().items()})
        atomic_npz(directory/'model.npz',**arrays)
        record=dict(kind=self.kind,settings=self.settings,target_fingerprint=self.manifest.get('target_fingerprint'),comparison_fingerprint=self.manifest.get('comparison_fingerprint'),manifest=self.manifest,model_file='model.npz',model_sha256=file_hash(directory/'model.npz'),**metadata)
        atomic_json(directory/'model.json',record);return record


def _encoder_cfg(config,x):
    c=dict(_settings(config));c.update(config.get('model',{}));width=int(c.get('embedding_dim',128));heads=int(c.get('heads',4))
    if width%heads:raise ValueError('Encoder width must be divisible by heads')
    return dict(architecture=c.get('architecture','transformer'),channels=1,embedding_dim=width,heads=heads,layers=int(c.get('layers',4)),patch_samples=int(c.get('patch_samples',32)))


def _make_network(encoder_config,dimension,gaussian,seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from torch import nn
    from ..models import make_encoder
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    class SharedForecast(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=make_encoder(encoder_config)
            self.mean_head=nn.Linear(self.encoder.embedding_dim,dimension)
            self.variance_head=nn.Linear(self.encoder.embedding_dim,dimension) if gaussian else None
            if gaussian:nn.init.zeros_(self.variance_head.weight);nn.init.zeros_(self.variance_head.bias)
        def forward(self,x):
            z=self.encoder(x);mu=self.mean_head(z)
            return mu,(torch.nn.functional.softplus(self.variance_head(z))+1e-4 if self.variance_head is not None else None)
    return SharedForecast()


def _flops(encoder,input_length,dimension,gaussian):
    tokens=int(np.ceil(input_length/encoder['patch_samples']));e=encoder['embedding_dim'];layers=encoder['layers'];p=encoder['patch_samples']
    return int(2*tokens*p*e+layers*(24*tokens*e*e+4*tokens*tokens*e)+2*e*dimension*(2 if gaussian else 1))


def _fit_neural(config,panel,kind,seed,lr,steps=None,subset=None):
    import torch
    settings=_settings(config);torch.set_num_threads(int(settings['threads']));torch.use_deterministic_algorithms(True)
    device=settings.get('_device','cpu');train=panel['train'];x=train['X'];y=_target_mean(train)
    if subset is not None:x=x[subset];y=y[subset]
    ecfg=_encoder_cfg(config,x);net=_make_network(ecfg,y.shape[-1],kind=='neural_gaussian',seed).to(device)
    parameters=sum(p.numel() for p in net.parameters())
    if parameters>int(config.get('model',{}).get('max_parameters',2_000_000)):raise ValueError('Complete model exceeds configured parameter ceiling')
    optimizer=torch.optim.AdamW(net.parameters(),lr=float(lr),weight_decay=float(settings['weight_decay']))
    rng=np.random.default_rng(seed);batch=min(int(settings['batch_size']),len(x));steps=int(settings['steps'] if steps is None else steps)
    xx=torch.as_tensor(x[:,None],dtype=torch.float32,device=device);yy=torch.as_tensor(y,dtype=torch.float32,device=device)
    history=[];cpu=time.process_time();start=time.perf_counter()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    for step in range(steps):
        idx=rng.choice(len(x),batch,replace=False);optimizer.zero_grad(set_to_none=True);mu,var=net(xx[idx])
        loss=((mu-yy[idx])**2).mean() if var is None else .5*(torch.log(2*torch.pi*var)+(yy[idx]-mu).square()/var).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Neural loss nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();history.append(float(loss.detach().cpu()))
    if device.startswith('cuda'):torch.cuda.synchronize()
    model=Predictor(kind,{},dict(encoder=ecfg,dimension=y.shape[-1],seed=seed),panel['manifest'],net)
    forward=_flops(ecfg,x.shape[1],y.shape[1],kind=='neural_gaussian')
    account=dict(parameter_count=parameters,encoder_parameters=sum(p.numel() for p in net.encoder.parameters()),mean_head_parameters=sum(p.numel() for p in net.mean_head.parameters()),covariance_head_parameters=0 if net.variance_head is None else sum(p.numel() for p in net.variance_head.parameters()),optimizer_updates=steps,forward_examples=steps*batch,forward_passes=steps,approximate_training_flops=3*steps*batch*forward,approximate_inference_flops_per_context=forward,flops_method='analytic dense/attention multiply-add count; backward approximated as twice forward; excludes normalization/activations',elapsed_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else None,independent_training_contexts=len(x),independent_training_circuits=len(np.unique(train['structure_id'])),device=device,deterministic_algorithms=True,loss_history=history)
    return model,account


def _calibrate_variance(model,development,device):
    mean,_=model.predict(development,device);mask=development['future_mask'];r=(development['Y']-mean[:,None])**2
    # Every context contributes equally; dimensionwise residual variance, fit on dev only.
    variance=(np.where(mask[...,None],r,0).sum(1)/mask.sum(1)[:,None]).mean(0)
    model.state['variance']=np.maximum(variance,1e-6)


def _fit_candidates(config,panel,name,seed=0,device='cpu'):
    # The public benchmark and direct fitting API share the same outcome budget.
    panel=_one_future_panel(panel)
    cfg=_settings(config);train=panel['train'];dev=panel['development'];y=_target_mean(train);manifest=panel['manifest'];records=[];chosen=None
    cfgconfig=copy.deepcopy(config);cfgconfig['benchmark']=dict(cfg,_device=device)
    if name in MEAN_REFERENCES:candidates=[{}]
    elif name=='persistence':candidates=[{}]
    elif name=='ridge_wiener':candidates=[dict(lags=min(int(l),train['X'].shape[1]),alpha=a) for a,l in itertools.product(cfg['ridge_alphas'],cfg['lags'])]
    elif name=='harmonic':candidates=[dict(history_fraction=f) for f in cfg.get('harmonic_history_fractions',[1.])]
    elif name=='state_space':candidates=[dict(rank=r,alpha=cfg['ridge_alphas'][min(i,len(cfg['ridge_alphas'])-1)]) for i,r in enumerate(cfg['state_ranks'])]
    elif name in ('neural_mse','neural_gaussian'):candidates=[dict(lr=lr) for lr in cfg['learning_rates']]
    else:raise ValueError('Unknown H1 benchmark model '+name)
    candidates=candidates[:int(cfg['max_candidates'])]
    for candidate in candidates:
        begin=time.perf_counter();cpu=time.process_time();account={}
        if name in MEAN_REFERENCES:
            variance=np.ones(y.shape[1]) if name=='train_mean_unit_variance' else np.maximum(y.var(0,ddof=1 if len(y)>1 else 0),1e-6)
            model=Predictor(name,dict(mean=y.mean(0),variance=variance),manifest=manifest)
        elif name=='ridge_wiener':model=Predictor(name,_ridge_fit(train['X'][:,-candidate['lags']:],y,candidate['alpha']),candidate,manifest)
        elif name in ('persistence','harmonic'):model=Predictor(name,dict(frequencies=np.asarray(cfg['harmonic_frequencies_hz'])) if name=='harmonic' else {},candidate,manifest)
        elif name=='state_space':
            lag=min(16,train['X'].shape[1]-1);sequences=np.lib.stride_tricks.sliding_window_view(train['X'],lag,axis=1)
            before=sequences[:,:-1].reshape(-1,lag);after=sequences[:,1:].reshape(-1,lag)
            # Deterministic coverage cap, not extra independent contexts.
            idx=np.linspace(0,len(before)-1,min(len(before),4096),dtype=int);before=before[idx];after=after[idx];center=before.mean(0)
            _,_,vt=np.linalg.svd(before-center,full_matrices=False);rank=min(int(candidate['rank']),vt.shape[0]);components=vt[:rank]
            z=(before-center)@components.T;zz=(after-center)@components.T
            transition=np.linalg.lstsq(z,zz,rcond=None)[0];radius=float(np.max(np.abs(np.linalg.eigvals(transition))))
            if radius>=.999:transition*=.999/radius
            lead_steps=max(1,round(manifest['lead_ms']*manifest['input']['sample_rate_hz']/1000))
            state=dict(lags=np.asarray(lag),delay_offset=center,components=components,transition=transition,lead_steps=np.asarray(lead_steps))
            state.update(_ridge_fit(_state_basis(state,train['X']),y,candidate['alpha']));model=Predictor(name,state,candidate,manifest)
        else:model,account=_fit_neural(cfgconfig,panel,name,seed,candidate['lr'])
        if name not in (*MEAN_REFERENCES,'neural_gaussian'):_calibrate_variance(model,dev,device)
        pred,var=model.predict(dev,device);metric=_circuit_nll(dev,pred,var) if name=='neural_gaussian' else _circuit_mse(dev,pred)
        if not account:
            parameters=sum(np.asarray(v).size for k,v in model.state.items() if k not in {'frequencies','lags','lead_steps'} and np.asarray(v).dtype.kind in 'fc')
            if name=='train_mean_unit_variance':parameters-=model.state['variance'].size
            n=len(train['X']);p=train['X'].shape[-1];d=y.shape[-1]
            account=dict(parameter_count=parameters,parameter_definition='stored fitted floating coefficients/statistics; fixed frequency grid and horizon excluded',per_context_harmonic_fit_parameters=4 if name=='harmonic' else 0,optimizer_updates=0,forward_examples=len(train['X'])+len(dev['X']),forward_passes=2,approximate_training_flops=int(2*n*min(p,64)**2+min(p,64)**3+2*n*min(p,64)*d) if name in ('ridge_wiener','state_space') else None,approximate_inference_flops_per_context=None,flops_method='dense least-squares order estimate; null means not estimated, never zero-compute',elapsed_seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,independent_training_contexts=len(train['X']),independent_training_circuits=len(np.unique(train['structure_id'])),device='cpu')
        development_calls=1 if name in (*MEAN_REFERENCES,'neural_gaussian') else 2
        training_examples=account['forward_examples'] if name.startswith('neural_') else 0
        training_passes=account['forward_passes'] if name.startswith('neural_') else 0
        account.update(training_forward_examples=training_examples,development_forward_examples=development_calls*len(dev['X']),
            training_forward_passes=training_passes,development_forward_passes=development_calls*(int(np.ceil(len(dev['X'])/64)) if name.startswith('neural_') else 1),
            forward_examples=training_examples+development_calls*len(dev['X']),
            forward_passes=training_passes+development_calls*(int(np.ceil(len(dev['X'])/64)) if name.startswith('neural_') else 1),
            forward_definition='neural minibatches / vectorized prediction calls; numerical factorization cost reported separately',
            calibration_development_contexts=0 if name in (*MEAN_REFERENCES,'neural_gaussian') else len(dev['X']),
            candidate_total_seconds=time.perf_counter()-begin,
            fit_uses_evaluation_outcomes=False)
        if not name.startswith('neural_'):
            length=train['X'].shape[1];dimension=y.shape[1]
            if name in MEAN_REFERENCES:fit_flops=(1 if name=='train_mean_unit_variance' else 5)*len(y)*dimension;per_context=0
            elif name=='persistence':fit_flops=0;per_context=20*dimension
            elif name=='harmonic':fit_flops=0;per_context=30*length*len(cfg['harmonic_frequencies_hz'])+20*dimension
            elif name=='ridge_wiener':
                lag=candidate['lags'];fit_flops=4*len(y)*lag*min(len(y),lag)+2*len(y)*lag*dimension;per_context=2*lag*dimension+2*lag
            else:
                lag=int(model.state['lags']);rank=len(model.state['components']);fit_flops=4*min(4096,len(y)*length)*lag**2+2*len(y)*rank*dimension;per_context=2*lag*rank+2*rank*rank+4*rank*dimension
            account.update(approximate_training_flops=int(fit_flops),approximate_inference_flops_per_context=int(per_context),flops_method='declared dense-factorization/multiply-add order estimate; harmonic trigonometric evaluations excluded; zero for mean broadcast is arithmetic-only, not zero latency')
        account['approximate_development_flops']=account['development_forward_examples']*account['approximate_inference_flops_per_context']
        record=dict(hyperparameters=candidate,development_score=metric,criterion='Gaussian NLL' if name=='neural_gaussian' else 'MSE',accounting=account);records.append(record)
        if chosen is None or metric<chosen[0]:chosen=(metric,model,record)
    if chosen is None:raise ValueError('No candidate model')
    selected=chosen[2].copy();selected['candidates']=records;selected['search_count']=len(records)
    selected['selection_split']='none; prespecified reference, development score is diagnostic only' if name in MEAN_REFERENCES else 'development'
    selected['variance_model']='prespecified identity covariance in standardized target coordinates; no variance fitting or development calibration' if name=='train_mean_unit_variance' else ('training learned heteroscedastic diagonal' if name=='neural_gaussian' else ('unconditional training target diagonal variance' if name=='train_mean' else 'dimensionwise pooled development residual variance, frozen before scoring'))
    selected['training_outcome_policy']='first valid independent future per training context; equal context weights'
    if name=='train_mean_unit_variance':
        selected['accounting'].update(fixed_variance_coordinates=y.shape[1],parameter_definition='fitted training-mean coordinates only; stored prespecified unit variance excluded')
    selected['conditioning']='visible EEG only; state-space means fitted delay-embedding dynamics, not a privileged simulator-state oracle'
    if name=='state_space':selected['state_space_horizon_approximation']=dict(physical_target_lead_ms=manifest['lead_ms'],latent_transition_steps=int(chosen[1].state['lead_steps']),input_step_ms=1000/manifest['input']['sample_rate_hz'],interpretation='nearest positive integer latent transition count is a fitted model approximation; supervised multihorizon readout still predicts the exact unchanged target operator/grid')
    selected['harmonic_scope']='EEG-fitted sinusoid with unshifted availability timestamps; no hidden frequency; D=0 reference, delayed/noisy input can limit it' if name=='harmonic' else None
    return chosen[1],selected


def load_numeric_model(directory,device='cpu'):
    directory=Path(directory);record=read_json(directory/'model.json')
    if file_hash(directory/record['model_file'])!=record['model_sha256']:
        raise ValueError('Model checksum mismatch')
    with np.load(directory/record['model_file'],allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    state={k:v for k,v in arrays.items() if not k.startswith('tensor__')};network=None
    if record['kind'] in ('neural_mse','neural_gaussian'):
        import torch
        settings=record['settings'];network=_make_network(settings['encoder'],settings['dimension'],record['kind']=='neural_gaussian',settings['seed'])
        network.load_state_dict({k[len('tensor__'):]:torch.as_tensor(v) for k,v in arrays.items() if k.startswith('tensor__')});network.to(device)
    return Predictor(record['kind'],state,record['settings'],record['manifest'],network),record


def _stage_signature(config,manifest):
    return canonical_hash(dict(benchmark=_settings(config),model=config.get('model',{}),manifest=manifest,implementation={p.name:file_hash(p) for p in [Path(__file__),Path(__file__).with_name('scoring.py')]}))


def _slice(data,indices):
    n=len(data['X']);out={}
    for key,value in data.items():
        out[key]=value[indices] if key not in ('input_time_ms','target_time_ms') and getattr(value,'ndim',0)>0 and len(value)==n else value
    return out


def training_diagnostics(config,panel,directory,device='cpu'):
    """Train/dev-only diagnostics; failure is evidence, never a test-based rescue."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);cfg=_settings(config)
    train=panel['train'];dev=panel['development'];y=_target_mean(train);rng=np.random.default_rng(9197)
    permutation=rng.permutation(len(y));lag=min(32,train['X'].shape[1]);normal=_ridge_fit(train['X'][:,-lag:],y,1.);shuffled=_ridge_fit(train['X'][:,-lag:],y[permutation],1.)
    p=_ridge_predict(normal,dev['X'][:,-lag:]);q=_ridge_predict(shuffled,dev['X'][:,-lag:]);baseline=np.repeat(y.mean(0)[None],len(dev['X']),0)
    # A learnable exact deterministic signal verifies end-to-end regression/scoring.
    control_x=rng.normal(size=(192,16));control_y=control_x[:,-1,None]*(.8**np.arange(1,9))[None]
    fit=_ridge_fit(control_x[:128],control_y[:128],.01);control_pred=_ridge_predict(fit,control_x[128:]);control_mse=float(((control_pred-control_y[128:])**2).mean());control_r0=float(((control_y[128:]-control_y[:128].mean(0))**2).mean())
    report=dict(selection_use='diagnostic only; neither evaluated on test nor used to tune campaign settings',exact_deterministic_control=dict(mse=control_mse,r0=control_r0,skill=1-control_mse/control_r0,passed=control_mse<.05*control_r0),shuffled_training_targets=dict(development_mse=_circuit_mse(dev,q),unshuffled_ridge_mse=_circuit_mse(dev,p),training_mean_development_mse=_circuit_mse(dev,baseline),n_permutations=1,interpretation='one null diagnostic has finite-sample fluctuations; it is not a leakage certificate'))
    atomic_npz(directory/'null_predictions.npz',development_context_id=dev['context_id'],ridge_mean=p,shuffled_mean=q,baseline_mean=baseline,training_permutation=permutation,control_target=control_y[128:],control_prediction=control_pred)
    if int(cfg['diagnostic_steps'])>0:
        n=min(int(cfg['diagnostic_examples']),len(y));indices=np.arange(n);dc=copy.deepcopy(config);dc['benchmark']=dict(cfg,_device=device,weight_decay=0.)
        model,account=_fit_neural(dc,panel,'neural_mse',0,.003,steps=cfg['diagnostic_steps'],subset=indices)
        small=_slice(train,indices);pred,_=model.predict(small,device);mse=float(((pred-y[indices])**2).mean());r0=float(((y[indices]-y[indices].mean(0))**2).mean())
        report['few_example_overfit']=dict(n_contexts=n,training_mse=mse,training_mean_risk=r0,ratio=mse/r0 if r0>0 else None,passed=bool(r0>0 and mse<.1*r0),accounting=account,interpretation='actual shared encoder; failure blocks information-limit attribution, not scoring/reporting')
        atomic_npz(directory/'few_example_predictions.npz',context_id=small['context_id'],target=y[indices],mean=pred)
    else:report['few_example_overfit']=dict(status='not run: diagnostic_steps=0')
    atomic_json(directory/'diagnostics.json',report);return report


def run_benchmark(config,output_root,device='cpu',resume=False):
    from .dataset import load_panel,list_jobs
    root=Path(output_root);cfg=_settings(config);results=[]
    if (root/'provenance.json').exists():
        from .artifacts import require_compatible
        require_compatible(root,config)
    for job_id in list_jobs(root):
        panel=load_panel(root,job_id);_validate_panel(panel);panel=_one_future_panel(panel);manifest=panel['manifest'];signature=_stage_signature(config,manifest)
        directory=root/'benchmarks'/job_id;marker=directory/'summary.json'
        if marker.exists():
            prior=read_json(marker)
            if prior.get('signature')!=signature:raise ValueError('Incompatible benchmark resume; use an immutable new output root')
            if not resume:raise FileExistsError('Benchmark exists; --resume required')
            for path,digest in prior['artifacts'].items():
                if not (directory/path).is_file() or file_hash(directory/path)!=digest:raise ValueError('Missing/corrupt benchmark artifact '+path)
            results.append(prior);continue
        directory.mkdir(parents=True,exist_ok=True);models=[];model_records=[];captured=[]
        atomic_npz(directory/'fitting_contexts.npz',training_context_id=panel['train']['context_id'],training_structure_id=panel['train']['structure_id'],selected_training_future_id=panel['train']['future_id'],selected_training_Y=panel['train']['Y'],development_context_id=panel['development']['context_id'])
        # All fits and hyperparameter selection complete before evaluation X/Y are used.
        for name in cfg['models']:
            seeds=cfg['seeds'] if name in ('neural_mse','neural_gaussian') else [0]
            for seed in seeds:
                identity=name+(f'__seed_{seed}' if name.startswith('neural_') else '')
                modeldir=directory/'models'/identity
                if (modeldir/'model.json').exists():
                    record=read_json(modeldir/'model.json')
                    if not resume or record.get('signature')!=signature:raise ValueError('Existing/incompatible frozen model')
                    model,record=load_numeric_model(modeldir,device)
                else:
                    with warnings.catch_warnings(record=True) as warning_records:
                        warnings.simplefilter('always');model,fit=_fit_candidates(config,panel,name,seed,device)
                    messages=[dict(category=w.category.__name__,message=str(w.message),filename=w.filename,line=w.lineno) for w in warning_records]
                    captured.extend(dict(model=identity,**w) for w in messages)
                    audit_start=time.perf_counter()
                    train_mean,train_var=model.predict(panel['train'],device);dev_mean,dev_var=model.predict(panel['development'],device)
                    atomic_npz(modeldir/'fit_predictions.npz',training_context_id=panel['train']['context_id'],training_target=panel['train']['Y'],training_future_mask=panel['train']['future_mask'],training_mean=train_mean,training_variance=train_var,development_context_id=panel['development']['context_id'],development_target=panel['development']['Y'],development_future_mask=panel['development']['future_mask'],development_mean=dev_mean,development_variance=dev_var)
                    fit['postfit_prediction_audit']=dict(forward_examples=len(train_mean)+len(dev_mean),forward_passes=(int(np.ceil(len(train_mean)/64))+int(np.ceil(len(dev_mean)/64))) if name.startswith('neural_') else 2,elapsed_seconds=time.perf_counter()-audit_start,selection_use='none; export selected model predictions for independent audit')
                    record=model.export(modeldir,dict(signature=signature,identity=identity,fit=fit,warnings=messages,fit_predictions_sha256=file_hash(modeldir/'fit_predictions.npz')))
                models.append((identity,model));model_records.append(record)
        paired=[]
        byid={r['identity']:r for r in model_records}
        for seed in cfg['seeds']:
            a=byid.get(f'neural_mse__seed_{seed}');b=byid.get(f'neural_gaussian__seed_{seed}')
            if a and b:
                ac=a['fit']['accounting'];bc=b['fit']['accounting'];relative=(bc['parameter_count']-ac['parameter_count'])/ac['parameter_count']
                paired.append(dict(seed=seed,shared_encoder_and_mean_decoder=True,relative_parameter_overhead=relative,declared_tolerance=cfg['parameter_tolerance'],within_tolerance=relative<=cfg['parameter_tolerance'],updates_mse=ac['optimizer_updates'],updates_gaussian=bc['optimizer_updates'],mean_decoder_initialization='same seed and construction order; variance head allocated afterwards',compute_match='fixed updates; measured time/FLOPs differ and are reported'))
        frozen=dict(signature=signature,target_fingerprint=manifest['target_fingerprint'],comparison_fingerprint=manifest['comparison_fingerprint'],models=[dict(identity=r['identity'],sha256=r['model_sha256']) for r in model_records],frozen_before_scoring=True)
        atomic_json(directory/'frozen_models.json',frozen)
        diagnostics=training_diagnostics(config,panel,directory/'diagnostics',device)
        evaluate=panel['evaluation'];base=_target_mean(panel['train']).mean(0);means=[];variances=[];reports=[]
        for index,(identity,model) in enumerate(models):
            start=time.perf_counter();mean,variance=model.predict(evaluate,device)
            if device.startswith('cuda'):
                import torch
                torch.cuda.synchronize()
            inference=time.perf_counter()-start
            report,arrays=score_panel(evaluate['Y'],mean,variance,base,evaluate['structure_id'],future_mask=evaluate['future_mask'],target_fingerprint=manifest['target_fingerprint'],prediction_fingerprint=model.manifest['target_fingerprint'],y_scale=manifest['normalization']['y_scale'],y_offset=manifest['normalization']['y_offset'],draws=int(cfg['draws']),seed=1000+index,reference_futures=evaluate.get('reference_Y'),reference_mask=evaluate.get('reference_mask'),comparison_fingerprint=manifest['comparison_fingerprint'],prediction_comparison_fingerprint=model.manifest['comparison_fingerprint'])
            arrays.update(context_id=evaluate['context_id'],structure_id=evaluate['structure_id'],future_id=evaluate['future_id'],comparison_fingerprint=np.asarray(manifest['comparison_fingerprint']))
            path=directory/'scores'/f'{identity}.npz';atomic_npz(path,**arrays);report.update(model=identity,inference_seconds=inference,prediction_file=str(path.relative_to(directory)))
            atomic_json(path.with_suffix('.json'),report);reports.append(report);means.append(mean);variances.append(variance)
        atomic_npz(directory/'predictions.npz',model_names=np.asarray([name for name,_ in models]),mean=np.stack(means),variance=np.stack(variances),context_id=evaluate['context_id'],structure_id=evaluate['structure_id'],target_fingerprint=np.asarray(manifest['target_fingerprint']),comparison_fingerprint=np.asarray(manifest['comparison_fingerprint']),y_scale=np.asarray(manifest['normalization']['y_scale']),y_offset=np.asarray(manifest['normalization']['y_offset']))
        paired_risks=[]
        reports_by_model={r['model']:r for r in reports}
        for seed in cfg['seeds']:
            left=reports_by_model.get(f'neural_mse__seed_{seed}');right=reports_by_model.get(f'neural_gaussian__seed_{seed}')
            if left and right:
                a={r['structure_id']:r['mean_risk'] for r in left['per_circuit']};b={r['structure_id']:r['mean_risk'] for r in right['per_circuit']}
                paired_risks.extend(dict(optimizer_seed=seed,structure_id=s,gaussian_minus_mse_mean_risk=b[s]-a[s]) for s in sorted(set(a)&set(b)))
        atomic_json(directory/'paired_neural_risks.json',dict(rows=paired_risks,interpretation='paired evaluation circuits, same contexts/futures; optimizer repeats are not additional circuits'))
        artifacts={str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*') if p.is_file() and p!=marker}
        summary=dict(schema_version='eegrepr-h1-benchmark-v2',signature=signature,job_id=job_id,target_fingerprint=manifest['target_fingerprint'],comparison_fingerprint=manifest['comparison_fingerprint'],completed_models=len(models),models=reports,fits=model_records,neural_pair_accounting=paired,diagnostics=diagnostics,warnings=captured,artifacts=artifacts,status='complete',claim_scope='bounded H1 diagnostic; a poor fitted predictor alone cannot establish a physical information limit',future_pool='same histories and scoring futures for every frozen model',training_outcome_policy='first valid independent future per training context, saved in fitting_contexts.npz; equal context weights')
        atomic_json(marker,summary);results.append(summary)
    index=dict(schema_version='eegrepr-h1-benchmark-index-v2',completed_jobs=len(results),jobs=[dict(job_id=x['job_id'],completed_models=x['completed_models'],path=f"benchmarks/{x['job_id']}/summary.json") for x in results])
    atomic_json(root/'benchmarks'/'manifest.json',index);return index


def _one_future_panel(panel):
    panel=dict(panel);data=dict(panel['train']);mask=np.asarray(data['future_mask'],bool);first=mask.argmax(1);n=np.arange(len(first))
    for key in ('Y','Y_physical','future_id'):
        if key in data:data[key]=data[key][n,first][:,None]
    data['future_mask']=np.ones((len(first),1),bool);panel['train']=data;return panel


def _encode_network(encoder,x,device):
    import torch
    encoder.eval();rows=[]
    with torch.no_grad():
        for start in range(0,len(x),64):rows.append(encoder(torch.as_tensor(x[start:start+64,None],dtype=torch.float32,device=device)).cpu().numpy())
    return np.concatenate(rows)


def get_representations(config,output_root,data,device='cpu',resume=False):
    """Training-only diagnostic banks, aligned to panel context order.

    Returns {name:{train,development,evaluation,metadata}}. Native dimensions
    are honest; downstream rate sweeps may reject unsupported dimensions/ranks.
    All fitted model arrays and embeddings are NPZ, never arbitrary pickle.
    """
    panel=_one_future_panel(data);_validate_panel(panel);directory=Path(output_root)/'representations';directory.mkdir(parents=True,exist_ok=True)
    signature=canonical_hash(dict(stage=_stage_signature(config,panel['manifest']),compression=config.get('compression',config.get('h1',{}).get('compression',{}))))
    marker=directory/'manifest.json'
    if marker.exists():
        record=read_json(marker)
        if not resume or record['signature']!=signature:raise ValueError('Existing/incompatible representation bank')
        output={}
        for row in record['representations']:
            for name,digest in row['artifacts'].items():
                if file_hash(directory/name)!=digest:raise ValueError('Representation checksum mismatch')
            with np.load(directory/row['path'],allow_pickle=False) as f:output[row['name']]={s:f[s] for s in ('train','development','evaluation')}
            output[row['name']]['metadata']=row
        return output
    cfg=_settings(config);train=panel['train'];split_names=('train','development','evaluation');output={};records=[]
    fs=float(panel['manifest']['input']['sample_rate_hz']);compression=config.get('compression',config.get('h1',{}).get('compression',{}))
    dimensions=compression.get('dimensions',[8,32,128]);maxdim=max(dimensions)
    def save(name,values,metadata,model_arrays=None):
        arrays=dict(values);arrays.update({s+'_context_id':panel[s]['context_id'] for s in split_names})
        atomic_npz(directory/f'{name}.npz',**arrays);artifacts={f'{name}.npz':file_hash(directory/f'{name}.npz')}
        if model_arrays is not None:
            atomic_npz(directory/f'{name}_model.npz',**model_arrays);artifacts[f'{name}_model.npz']=file_hash(directory/f'{name}_model.npz')
        row=dict(name=name,path=f'{name}.npz',dimension=int(values['train'].shape[1]),normalization=panel['manifest']['normalization'],fit_split='train',context_alignment='exact panel split order; IDs saved',artifacts=artifacts,**metadata)
        atomic_json(directory/f'{name}.json',row);row['artifacts'][f'{name}.json']=file_hash(directory/f'{name}.json')
        output[name]=dict(values,metadata=row);records.append(row)
    save('feature18',{s:feature_matrix(panel[s]['X'][:,None],sample_hz=fs) for s in split_names},dict(method='18 phase-invariant spectral/autocorrelation/envelope features from available EEG',sample_rate_hz=fs))
    center=train['X'].mean(0);_,singular,vt=np.linalg.svd(train['X']-center,full_matrices=False);rank=int(np.sum(singular>max(singular[0] if len(singular) else 0,1.)*1e-10));dimension=min(maxdim,rank)
    if dimension<1:raise ValueError('Training EEG has zero PCA rank')
    components=vt[:dimension]
    save('pca',{s:(panel[s]['X']-center)@components.T for s in split_names},dict(method='training-only PCA',available_rank=rank),dict(offset=center,components=components,singular_values=singular))
    import torch
    from ..models import make_encoder,ReconstructionModel,gap_mask,masked_loss
    torch.set_num_threads(int(cfg['threads']));torch.use_deterministic_algorithms(True);ecfg=_encoder_cfg(config,train['X']);torch.manual_seed(0)
    encoder=make_encoder(ecfg).to(device)
    save('untrained',{s:_encode_network(encoder,panel[s]['X'],device) for s in split_names},dict(method='untrained frozen shared encoder',encoder_config=ecfg,seed=0),{'tensor__'+k:v.detach().cpu().numpy() for k,v in encoder.state_dict().items()})
    torch.manual_seed(0);reconstructor=ReconstructionModel(make_encoder(ecfg)).to(device);optimizer=torch.optim.AdamW(reconstructor.parameters(),lr=.001)
    xx=torch.as_tensor(train['X'][:,None],dtype=torch.float32,device=device);rng=np.random.default_rng(0);losses=[];begin=time.perf_counter();batch=min(int(cfg['batch_size']),len(xx));steps=int(cfg['representation_steps'])
    for step in range(steps):
        idx=rng.choice(len(xx),batch,replace=False);x=xx[idx];mask=gap_mask(x,.4,max(1,round(.128*fs)))
        optimizer.zero_grad(set_to_none=True);loss=masked_loss(reconstructor(x.masked_fill(mask,0)),x,mask);loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
    save('masked_mse',{s:_encode_network(reconstructor.encoder,panel[s]['X'],device) for s in split_names},dict(method='masked reconstruction of training EEG only',encoder_config=ecfg,seed=0,optimizer_updates=steps,forward_examples=steps*batch,elapsed_seconds=time.perf_counter()-begin,loss_history=losses,parameter_count=sum(p.numel() for p in reconstructor.parameters())),{'tensor__'+k:v.detach().cpu().numpy() for k,v in reconstructor.state_dict().items()})
    # Prefer the already frozen H1 mean-objective encoder for this same job.
    modelpath=None;job_id=panel['manifest'].get('job_id')
    for parent in [Path(output_root),*Path(output_root).parents]:
        candidate=parent/'benchmarks'/str(job_id)/'models'/'neural_mse__seed_0'
        if (candidate/'model.json').exists():modelpath=candidate;break
    if modelpath is not None:
        forecast,record=load_numeric_model(modelpath,device)
        if forecast.manifest['comparison_fingerprint']!=panel['manifest']['comparison_fingerprint']:raise ValueError('Forecast representation has different target/data fingerprint')
        provenance=dict(reused_model=str(modelpath),reused_model_sha256=record['model_sha256'],method='frozen H1 deterministic mean encoder')
    else:
        local=copy.deepcopy(config);local['benchmark']=dict(cfg,_device=device)
        forecast,account=_fit_neural(local,panel,'neural_mse',0,cfg['learning_rates'][0],steps=cfg['representation_steps'])
        provenance=dict(method='diagnostic forecast encoder fitted on the same training recordings; no H2 leaderboard',accounting=account)
    save('forecast',{s:_encode_network(forecast.network.encoder,panel[s]['X'],device) for s in split_names},dict(encoder_config=ecfg,**provenance),{'tensor__'+k:v.detach().cpu().numpy() for k,v in forecast.network.encoder.state_dict().items()})
    atomic_json(marker,dict(schema_version='eegrepr-h1-representations-v2',signature=signature,representations=records,supervision='forecast uses first training future per context; all other encoders use training histories only; no evaluation outcomes'))
    return output
