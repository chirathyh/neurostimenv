"""Frozen probes and proper forecast scoring with structure-level inference.

Only training/development labels fit probes or choose hyperparameters. Test
labels are read by scoring after each selected probe has been frozen. Windows,
noise copies and optimizer seeds are never treated as independent circuits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .training import (atomic_json,digest,feature_matrix,independent_budget_indices,
                       label_valid,load_fitted,task_specs,_load_examples)


def encode(model,x,device='cpu',batch_size=64):
    import torch
    if model is None:
        return feature_matrix(x)
    out=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            out.append(model.encoder(torch.as_tensor(x[start:start+batch_size],device=device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,model.encoder.embedding_dim))


def trial_pool(features,y,metadata,kind,causal_target=False):
    """One observation per independent parent for probe fit and inference.

    Averaging representations over windows is explicitly trial-level transfer.
    View-specific robustness is scored separately; pooling is never more labels.
    """
    groups={}
    valid=label_valid(y,kind)
    for i,m in enumerate(metadata):
        if valid[i]:
            groups.setdefault(str(m['parent_trial_id']),[]).append(i)
    z,targets,meta=[],[],[]
    for parent,indices in sorted(groups.items()):
        if causal_target:
            # A later history must never contain an earlier scored future target.
            indices=[max(indices,key=lambda i:metadata[i].get('cutoff_index',i))]
        values=np.asarray(y)[indices]
        if kind=='classification' and len(np.unique(values))!=1:
            raise ValueError('Classification label varies within parent trial')
        z.append(features[indices].mean(0));targets.append(values.mean())
        identities={str(metadata[i]['structure_id']) for i in indices}
        if len(identities)!=1:
            raise ValueError('Parent trial crosses structures')
        meta.append(dict(metadata[indices[0]],parent_trial_id=parent))
    return np.asarray(z),np.asarray(targets),meta


def structure_metrics(y,pred,metadata,kind,reference_variance=None):
    from sklearn.metrics import balanced_accuracy_score,f1_score,mean_squared_error,r2_score
    rows=[]
    structures=sorted({str(m['structure_id']) for m in metadata})
    for structure in structures:
        idx=np.array([str(m['structure_id'])==structure for m in metadata])
        if kind=='classification':
            values=dict(balanced_accuracy=float(balanced_accuracy_score(y[idx],pred[idx])),
                        macro_f1=float(f1_score(y[idx],pred[idx],average='macro',zero_division=0)))
        else:
            mse=float(mean_squared_error(y[idx],pred[idx]))
            values=dict(mse=mse,normalized_mse=mse/reference_variance if reference_variance and reference_variance>0 else None,
                        r2=float(r2_score(y[idx],pred[idx])) if idx.sum()>1 and np.var(y[idx])>0 else None)
        rows.append(dict(structure_id=structure,n_trials=int(idx.sum()),
            n_independent_families=len({str(m.get('paired_family_id',m['parent_trial_id'])) for m,k in zip(metadata,idx) if k}),**values))
    metrics={key:float(np.mean([r[key] for r in rows if r[key] is not None]))
             for key in (('balanced_accuracy','macro_f1') if kind=='classification' else ('mse','normalized_mse','r2'))
             if any(r[key] is not None for r in rows)}
    return metrics,rows


def _metric_score(y,pred,metadata,kind,variance):
    scores,_=structure_metrics(y,pred,metadata,kind,variance)
    return scores.get('balanced_accuracy',-scores.get('normalized_mse',scores.get('mse',float('inf'))))


def _linear_probe(z,y,kind,regularization,seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression,Ridge
    estimator=(LogisticRegression(C=regularization,max_iter=1000,random_state=seed,class_weight='balanced')
               if kind=='classification' else Ridge(alpha=regularization))
    return make_pipeline(StandardScaler(),estimator).fit(z,y)


class ShallowProbe:
    """One hidden layer; normalization and target scale fitted on train only."""
    def __init__(self,z,y,kind,regularization,seed,steps=100,hidden=64):
        import torch
        from sklearn.preprocessing import StandardScaler,LabelEncoder
        self.scaler=StandardScaler().fit(z)
        self.kind=kind
        self.label_encoder=LabelEncoder().fit(y) if kind=='classification' else None
        self.mean=float(y.mean());self.scale=float(y.std()) or 1.
        torch.manual_seed(seed)
        output=len(self.label_encoder.classes_) if self.label_encoder is not None else 1
        self.model=torch.nn.Sequential(torch.nn.Linear(z.shape[1],hidden),torch.nn.GELU(),torch.nn.Linear(hidden,output))
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=.003,weight_decay=regularization)
        xx=torch.tensor(self.scaler.transform(z),dtype=torch.float32)
        yy=torch.tensor(self.label_encoder.transform(y),dtype=torch.long) if kind=='classification' else torch.tensor((y-self.mean)/self.scale,dtype=torch.float32)
        # Full trial-batch avoids giving long recordings additional label weight.
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True);prediction=self.model(xx)
            loss=(torch.nn.functional.cross_entropy(prediction,yy) if kind=='classification' else
                  torch.nn.functional.mse_loss(prediction[:,0],yy))
            loss.backward();optimizer.step()
        self.model.eval()

    def predict(self,z):
        import torch
        with torch.no_grad():
            p=self.model(torch.tensor(self.scaler.transform(z),dtype=torch.float32)).numpy()
        return self.label_encoder.inverse_transform(p.argmax(-1)) if self.kind=='classification' else p[:,0]*self.scale+self.mean


def _select_probe(train,development,kind,probe,config,seed):
    z,y,m=train;dz,dy,dm=development
    candidates=config.get('linear_grid',[.1,1.,10.]) if probe=='linear' else config.get('nonlinear_grid',[.001,.01,.1])
    chosen=None;selection=[]
    variance=float(y.var())
    for strength in candidates:
        fitted=(_linear_probe(z,y,kind,strength,seed) if probe=='linear' else
                ShallowProbe(z,y,kind,strength,seed,int(config.get('probe_steps',100)),int(config.get('probe_hidden',64))))
        value=_metric_score(dy,fitted.predict(dz),dm,kind,variance) if len(dz) else None
        selection.append(dict(regularization=strength,development_score=value))
        if chosen is None or (value is not None and value>chosen[0]):
            chosen=(value if value is not None else -float('inf'),fitted,strength)
    return chosen[1],dict(selected_regularization=chosen[2],candidates=selection,
                         selection_split='development' if len(dz) else 'predeclared first value; development unavailable')


def normalized_log_budget_area(budgets,values):
    budgets=np.asarray(budgets,dtype=float);values=np.asarray(values,dtype=float)
    order=np.argsort(budgets);budgets=budgets[order];values=values[order]
    if len(budgets)<2 or np.any(budgets<=0) or len(np.unique(budgets))!=len(budgets):
        raise ValueError('Area requires at least two distinct positive independent-label budgets')
    return float(np.trapz(values,np.log(budgets))/(np.log(budgets[-1])-np.log(budgets[0])))


def cluster_interval(values,seed=0,replicates=2000):
    values=np.asarray(values,dtype=float)
    if len(values)<2:
        return dict(mean=float(values.mean()) if len(values) else None,lower=None,upper=None,n_structures=len(values),
                    limitation='At least two independent structures required for uncertainty')
    rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(replicates,len(values)))].mean(1)
    return dict(mean=float(values.mean()),lower=float(np.quantile(means,.025)),upper=float(np.quantile(means,.975)),n_structures=len(values))


def paired_cluster_inference(a,b,seed=0,replicates=2000):
    """Maps structure->score; average optimizer repeats BEFORE this function."""
    common=sorted(set(a)&set(b))
    differences=np.array([a[k]-b[k] for k in common])
    result=cluster_interval(differences,seed,replicates)
    if len(differences):
        if len(differences)<=16:
            signs=1-2*((np.arange(2**len(differences))[:,None]>>np.arange(len(differences)))&1)
            null=(signs*differences).mean(1)
            p=float(np.mean(np.abs(null)>=abs(differences.mean())-1e-15))
        else:
            rng=np.random.default_rng(seed)
            null=(rng.choice([-1,1],(replicates,len(differences)))*differences).mean(1)
            p=float((1+(np.abs(null)>=abs(differences.mean())-1e-15).sum())/(replicates+1))
        result['paired_sign_flip_p']=p
    result['structures']=common
    return result


def fdr_bh(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p)
    adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty(len(p));out[order]=np.minimum(adjusted,1.)
    return out.tolist()


def forecast_scores(distribution,target,unconditional_mean,draws=64):
    """Proper joint NLL and energy score; marginal calibration is supplementary."""
    import torch
    from scipy.stats import norm
    target=target.flatten(1)
    mean=distribution.mean
    variance=distribution.variance
    scale=variance.sqrt()
    mse=(mean-target).square().mean(-1)
    baseline=(target-unconditional_mean).square().mean(-1)
    nll=-distribution.log_prob(target)/target.shape[-1]
    samples=distribution.rsample(torch.Size([draws]))
    # Independent halves, never O(n^2) pseudo-replicates.
    first=(samples-target).square().mean(-1).sqrt().mean(0)
    half=draws//2
    second=(samples[:half]-samples[half:2*half]).square().mean(-1).sqrt().mean(0)
    energy=first-.5*second
    result=dict(mse=mse.detach().cpu().numpy(),baseline_mse=baseline.detach().cpu().numpy(),
                nll_per_dimension=nll.detach().cpu().numpy(),energy=energy.detach().cpu().numpy())
    z=((target-mean)/scale).detach().cpu().numpy()
    # Gaussian CRPS is a proper *marginal* score, reported separately from joint energy.
    crps=scale.detach().cpu().numpy()*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))
    result['marginal_crps']=crps.mean(1)
    for level in (.5,.8,.95):
        critical=float(norm.ppf((1+level)/2))
        result[f'coverage_{int(level*100)}']=(np.abs(z)<=critical).mean(1)
    return result


def _future_diagnostic_features(values,threshold,sample_hz=500):
    """Offline distribution diagnostics, never a training/selection objective."""
    values=np.asarray(values,dtype=float)
    width=min(values.shape[-1],max(1,round(.032*sample_hz)))
    windows=np.lib.stride_tricks.sliding_window_view(values,width,axis=-1)
    envelope=np.sqrt((windows**2).mean(-1))
    centered=values-values.mean(-1,keepdims=True)
    psd=np.abs(np.fft.rfft(centered,axis=-1))**2/(sample_hz*values.shape[-1])
    features=dict(envelope_mean=envelope.mean(axis=(-1,-2)),
        burst_proxy_occupancy=(envelope>threshold).mean(axis=(-1,-2)),psd=psd.mean(-2))
    for lag in (1,4,16):
        if lag>=values.shape[-1]:
            continue
        numerator=(centered[...,:-lag]*centered[...,lag:]).mean(-1)
        denominator=(centered**2).mean(-1)
        features[f'autocorrelation_lag_{lag}']=(numerator/np.maximum(denominator,np.finfo(float).tiny)).mean(-1)
    return features


def _forecast_evaluate(model,train,test,device,seed,config):
    import torch
    torch.manual_seed(seed)
    baseline=torch.as_tensor(train['future'].mean(0).reshape(1,-1),device=device)
    collected={}
    diagnostics={}
    width=min(train['future'].shape[-1],16)
    train_envelope=np.sqrt((np.lib.stride_tricks.sliding_window_view(train['future'],width,axis=-1)**2).mean(-1))
    burst_threshold=float(np.quantile(train_envelope,.9))
    batch=int(config.get('batch_size',32))
    with torch.no_grad():
        for start in range(0,len(test['x']),batch):
            x=torch.as_tensor(test['x'][start:start+batch],device=device)
            target=torch.as_tensor(test['future'][start:start+batch],device=device)
            distribution=model(x)
            metrics=forecast_scores(distribution,target,baseline,int(config.get('forecast_draws',64)))
            samples=distribution.rsample(torch.Size([min(16,int(config.get('forecast_draws',64)))])).cpu().numpy().reshape(-1,len(x),*target.shape[1:])
            observed=_future_diagnostic_features(target.cpu().numpy(),burst_threshold)
            sampled=_future_diagnostic_features(samples,burst_threshold)
            for name in observed:
                predicted=sampled[name].mean(0)
                if name=='psd':
                    entries={'log1p_psd_squared_error':((np.log1p(predicted)-np.log1p(observed[name]))**2).mean(-1)}
                else:
                    entries={name+'_sampled':predicted,name+'_observed':observed[name]}
                for key,values in entries.items():
                    diagnostics.setdefault(key,[]).extend(values.tolist())
            for key,values in metrics.items():
                collected.setdefault(key,[]).extend(values.tolist())
    structures=sorted(set(str(m['structure_id']) for m in test['metadata']))
    rows=[]
    for structure in structures:
        parents={}
        for i,m in enumerate(test['metadata']):
            if str(m['structure_id'])==structure:
                parents.setdefault(str(m['parent_trial_id']),[]).append(i)
        row=dict(structure_id=structure,n_trials=len(parents))
        for key,values in collected.items():
            row[key]=float(np.mean([np.mean(np.asarray(values)[indices]) for indices in parents.values()]))
        row['normalized_mse']=row['mse']/row['baseline_mse'] if row['baseline_mse']>0 else None
        rows.append(row)
    diagnostic_structures=[]
    for structure in structures:
        parents={}
        for i,m in enumerate(test['metadata']):
            if str(m['structure_id'])==structure:
                parents.setdefault(str(m['parent_trial_id']),[]).append(i)
        diagnostic_structures.append(dict(structure_id=structure,**{key:float(np.mean([np.mean(np.asarray(values)[idx]) for idx in parents.values()])) for key,values in diagnostics.items()}))
    summary={key:float(np.mean([r[key] for r in rows])) for key in collected}
    summary['normalized_mse']=float(np.mean([r['normalized_mse'] for r in rows if r['normalized_mse'] is not None]))
    # Only a scalar channel is primary; conversion is reported when supplied by loader.
    normalization=test.get('normalization',{})
    scale=normalization.get('scale_v',normalization.get('scale'))
    if scale is not None and np.asarray(scale).size==1:
        scale=float(np.asarray(scale).ravel()[0])
        summary['mse_v2']=summary['mse']*scale**2
        summary['nll_per_dimension_volts']=summary['nll_per_dimension']+math.log(scale)
    return dict(metrics=summary,structure_scores=rows,units='training-normalized EEG',
                distribution_diagnostics=dict(structure_scores=diagnostic_structures,burst_proxy_rms_threshold=burst_threshold,
                    threshold_fit='90th percentile of 32-ms RMS in pretraining training futures',
                    frequency_resolution_hz=500/test['future'].shape[-1],
                    interpretation='PSD/autocorrelation and RMS exceedance are diagnostics only, not proof of correct conditional distribution or latent burst recovery; short blocks resolve neither slow rhythms nor full burst dwell times; zero-variance autocorrelation convention is zero'),
                target='declared future causal-filtered EEG; see forecast_config.target for clean versus sensor-noisy targets',
                covariance_family='Gaussian: cannot represent arbitrary multimodal conditional futures',
                baseline='unconditional future mean fitted on pretraining train structures only')


def _save_probe(path,probe):
    import joblib,os
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');joblib.dump(probe,tmp);os.replace(tmp,path)


def evaluate(config,output_root,protocol='leave-task-label-out',device='cpu',recipes=None,seeds=None,budgets=None,resume=False):
    cfg=config.get('evaluation',{})
    budgets=sorted(set(budgets or cfg.get('label_budgets',config.get('training',{}).get('label_budgets',[10,50,250,500]))))
    directory=Path(output_root)/'evaluations'/protocol
    directory.mkdir(parents=True,exist_ok=True)
    data={split:_load_examples(output_root,config,split,protocol,stage='probe') for split in ('train','development','test')}
    stress_data={}
    primary_variant=config.get('observation',{}).get('variant','clean')
    primary_gain=config.get('observation',{}).get('gain',1.)
    for variant in cfg.get('variants',[primary_variant]):
        for gain in cfg.get('gains',[primary_gain]):
            if (variant,gain)!=(primary_variant,primary_gain):
                stress_data[(variant,gain)]=_load_examples(output_root,config,'test',protocol,stage='probe',variant=variant,gain=gain)
    # Defensive post-loader audit: prohibit structure/parent overlap irrespective of windowing.
    for key in ('structure_id','parent_trial_id'):
        sets={split:{str(m[key]) for m in values['metadata']} for split,values in data.items()}
        if any(sets[a]&sets[b] for a,b in (('train','development'),('train','test'),('development','test'))):
            raise ValueError(f'Cross-split {key} leakage')
    if not len(data['train']['x']) or not len(data['test']['x']):
        raise ValueError('Frozen transfer evaluation requires nonempty train and test splits')
    runs=sorted((Path(output_root)/'models'/protocol).glob('*/seed_*/budget_*/run.json'))
    rows=[];forecasts=[];pending=[];specialist_seeds=set()
    task_names=cfg.get('tasks',list(data['train']['labels']))
    specs=task_specs([name for name in task_names if name in data['train']['labels']],data['train']['labels'])
    evaluation_signature=digest(dict(config=config,protocol=protocol,budgets=budgets,recipes=recipes,seeds=seeds))
    existing=directory/'probe_results.json'
    if existing.exists():
        previous=json.loads(existing.read_text())
        if previous.get('config_hash')!=evaluation_signature:
            raise ValueError('Incompatible evaluation resume; use a distinct output root/config')
        if not resume:
            raise FileExistsError('Evaluation outputs already exist; use --resume')
    for path in runs:
        run=json.loads(path.read_text());recipe=run['recipe'];seed=run['seed']
        if run['status']!='complete' or (recipes and recipe not in recipes) or (seeds is not None and seed not in seeds):
            continue
        run_key=dict(recipe=recipe,optimizer_seed=seed,pretraining_budget=run.get('budget'),protocol=protocol)
        model=None if recipe=='features' else load_fitted(path.parent,device)[0]
        features={split:encode(model,values['x'],device) for split,values in data.items()}
        stress_features={key:encode(model,values['x'],device) for key,values in stress_data.items()}
        if recipe.startswith('forecast_'):
            train_pre=_load_examples(output_root,config,'train',protocol,stage='pretrain')
            forecasts.append(dict(**run_key,forecast_config=config.get('forecast',{}),**_forecast_evaluate(model,train_pre,data['test'],device,seed,cfg)))
        for task,spec in specs.items():
            pooled={split:trial_pool(features[split],values['labels'][task],values['metadata'],spec['kind'],causal_target=task in {'envelope','log_bandpower','waveform'}) for split,values in data.items()}
            train=pooled['train'];dev=pooled['development'];test=pooled['test']
            if not len(train[0]) or not len(test[0]) or (spec['kind']=='classification' and len(np.unique(train[1]))<2):
                pending.append(dict(**run_key,task=task,status='unavailable',reason='No valid labels or fewer than two training classes'))
                continue
            for budget in budgets:
                idx,account=independent_budget_indices(train[2],train[1],spec['kind'],budget,seed)
                reduced=(train[0][idx],train[1][idx],[train[2][i] for i in idx])
                if spec['kind']=='classification' and len(np.unique(reduced[1]))<2:
                    pending.append(dict(**run_key,task=task,budget=budget,status='unavailable',reason='Label budget does not include two classes'))
                    continue
                for probe_name in cfg.get('probe_types',cfg.get('probes',['linear','nonlinear'])):
                    fitted,selection=_select_probe(reduced,dev,spec['kind'],probe_name,cfg,seed)
                    # Freeze fitted downstream head before touching the scoring labels.
                    checkpoint=directory/'probes'/recipe/f'seed_{seed}'/f'pretrain_{run.get("budget")}'/task/f'{probe_name}_{budget}.joblib'
                    _save_probe(checkpoint,fitted)
                    prediction=fitted.predict(test[0])
                    metrics,per_structure=structure_metrics(test[1],prediction,test[2],spec['kind'],float(reduced[1].var()))
                    rows.append(dict(**run_key,task=task,kind=spec['kind'],probe=probe_name,budget=budget,
                        metrics=metrics,structure_scores=per_structure,label_budget=account,selection=selection,
                        training_target_variance=float(reduced[1].var()),constant_training_target=bool(reduced[1].var()==0),
                        heldout_label=task in set(config.get('training',{}).get('heldout_tasks',cfg.get('heldout_tasks',[]))),
                        status='capped' if account['capped'] else 'complete',
                        representation_pooling=('final causal cutoff per parent; future temporal target remains unseen' if task in {'envelope','log_bandpower','waveform'} else 'retrospective mean over history windows within parent trial'),
                        fitted_probe=str(checkpoint)))
                    if protocol=='leave-generator-family-out':
                        family_mask=np.array([m['family'] in cfg.get('heldout_families',[]) for m in test[2]])
                        if family_mask.any() and (spec['kind']!='classification' or len(np.unique(test[1][family_mask]))>=2):
                            fm,fs=structure_metrics(test[1][family_mask],prediction[family_mask],[m for m,k in zip(test[2],family_mask) if k],spec['kind'],float(reduced[1].var()))
                            rows[-1]['heldout_generator_only']=dict(metrics=fm,structure_scores=fs,status='complete')
                        else:
                            rows[-1]['heldout_generator_only']=dict(status='Pending: no labelled cases or fewer than two held-out-family classes; no discriminability claim')
                    robustness=[]
                    for (variant,gain),values in stress_data.items():
                        rz,ry,rm=trial_pool(stress_features[(variant,gain)],values['labels'][task],values['metadata'],spec['kind'],causal_target=task in {'envelope','log_bandpower','waveform'})
                        if len(rz):
                            met,scores=structure_metrics(ry,fitted.predict(rz),rm,spec['kind'],float(reduced[1].var()))
                            robustness.append(dict(variant=variant,gain=gain,metrics=met,structure_scores=scores,
                                fitting='frozen primary-view training/development probe; no stress-test adaptation'))
                    rows[-1]['robustness']=robustness
                    # A single frozen shuffled-label linear diagnostic; never used for model selection.
                    if probe_name=='linear' and budget==budgets[-1] and cfg.get('shuffle_diagnostic',True):
                        yy=reduced[1].copy();rng=np.random.default_rng(seed+90001)
                        rng.shuffle(yy)
                        shuffled=_linear_probe(reduced[0],yy,spec['kind'],selection['selected_regularization'],seed)
                        sm,ss=structure_metrics(test[1],shuffled.predict(test[0]),test[2],spec['kind'],float(reduced[1].var()))
                        rows.append(dict(**run_key,task=task,kind=spec['kind'],probe='shuffled_linear_diagnostic',budget=budget,
                            metrics=sm,structure_scores=ss,label_budget=account,status='diagnostic',
                            excluded_from_primary=True))
        if model is not None:
            rows.extend(adaptation_rows(config,data,model,run_key,directory,specs,budgets,device))
        if seed not in specialist_seeds:
            specialist_key=dict(run_key,recipe='downstream_specialist',pretraining_budget=None)
            rows.extend(adaptation_rows(config,data,None,specialist_key,directory,specs,budgets,device,random_init=True))
            specialist_seeds.add(seed)
    areas=[]
    keys=sorted({(r['recipe'],r['optimizer_seed'],str(r['pretraining_budget']),r['task'],r['probe']) for r in rows if r['kind']=='classification' and not r.get('excluded_from_primary')})
    for key in keys:
        subset=[r for r in rows if (r['recipe'],r['optimizer_seed'],str(r['pretraining_budget']),r['task'],r['probe'])==key]
        subset=sorted(subset,key=lambda r:r['budget'])
        valid=len(subset)==len(budgets) and len(budgets)>=2 and all(r['status']=='complete' for r in subset)
        by_structure={}
        if valid:
            structures=set.intersection(*({s['structure_id'] for s in r['structure_scores']} for r in subset))
            for structure in sorted(structures):
                values=[next(s['balanced_accuracy'] for s in r['structure_scores'] if s['structure_id']==structure) for r in subset]
                by_structure[structure]=normalized_log_budget_area(budgets,values)
        areas.append(dict(recipe=key[0],optimizer_seed=key[1],pretraining_budget=key[2],task=key[3],probe=key[4],
                          budgets=budgets,status='complete' if valid else 'pending: insufficient or capped budget axis',
                          normalized_log_budget_area=float(np.mean(list(by_structure.values()))) if by_structure else None,
                          structure_scores=by_structure))
    result=dict(schema_version='eegrepr-evaluation-v1',config_hash=evaluation_signature,protocol=protocol,
                status='functional smoke' if config.get('claim_status',config.get('profile'))=='smoke' else 'exploratory',
                rows=rows,pending=pending,
                optimization='train labels; development hyperparameter selection; frozen test scoring',
                independent_unit='circuit structure; trial-pooled probe examples',
                primary_task_weighting='equal across declared classification tasks; continuous tasks reported separately',
                minimum_meaningful_gain=cfg.get('minimum_gain',cfg.get('minimum_meaningful_gain',.02)))
    aggregate=aggregate_transfer_areas(areas,[t for t,s in specs.items() if s['kind']=='classification'],config)
    atomic_json(existing,result);atomic_json(directory/'forecasts.json',forecasts);atomic_json(directory/'transfer_areas.json',areas)
    atomic_json(directory/'aggregate_transfer.json',aggregate)
    return result


class TrialNeuralProbe:
    """End-to-end limited-label adaptation; each update samples trials uniformly.

    One random window per selected trial per update prevents long recordings
    from increasing label exposure. Prediction averages all window logits.
    """
    def __init__(self,waveforms,y,kind,encoder_configuration,seed,steps=100,batch_size=16,
                 regularization=.01,initial_encoder=None,device='cpu'):
        import copy
        import torch
        from sklearn.preprocessing import LabelEncoder
        from .models import make_encoder
        torch.manual_seed(seed)
        self.kind,self.device=kind,device
        self.encoder=copy.deepcopy(initial_encoder) if initial_encoder is not None else make_encoder(encoder_configuration)
        self.label_encoder=LabelEncoder().fit(y) if kind=='classification' else None
        self.target_mean=float(y.mean());self.target_scale=float(y.std()) or 1.
        self.head=torch.nn.Linear(self.encoder.embedding_dim,len(self.label_encoder.classes_) if self.label_encoder else 1)
        self.encoder.to(device);self.head.to(device)
        self.encoder.train();self.head.train()
        optimizer=torch.optim.AdamW(list(self.encoder.parameters())+list(self.head.parameters()),lr=.001,weight_decay=regularization)
        labels=self.label_encoder.transform(y) if self.label_encoder is not None else (y-self.target_mean)/self.target_scale
        rng=np.random.default_rng(seed)
        self.examples_processed=0
        for _ in range(steps):
            ids=rng.choice(len(waveforms),size=min(batch_size,len(waveforms)),replace=False)
            x=np.stack([waveforms[i][rng.integers(len(waveforms[i]))] for i in ids])
            xx=torch.as_tensor(x,device=device)
            yy=torch.as_tensor(labels[ids],device=device)
            prediction=self.head(self.encoder(xx))
            loss=(torch.nn.functional.cross_entropy(prediction,yy.long()) if kind=='classification' else
                  torch.nn.functional.mse_loss(prediction[:,0],yy.float()))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.head.parameters()),1.)
            optimizer.step();self.examples_processed+=len(ids)
        self.encoder.eval();self.head.eval()

    def predict(self,waveforms):
        import torch
        outputs=[]
        with torch.no_grad():
            for windows in waveforms:
                batches=[]
                for i in range(0,len(windows),32):
                    batches.append(self.head(self.encoder(torch.as_tensor(windows[i:i+32],device=self.device))).cpu().numpy())
                outputs.append(np.concatenate(batches).mean(0))
        out=np.asarray(outputs)
        return self.label_encoder.inverse_transform(out.argmax(-1)) if self.kind=='classification' else out[:,0]*self.target_scale+self.target_mean


def trial_waveforms(examples,task,kind):
    valid=label_valid(examples['labels'][task],kind)
    groups={}
    for i,m in enumerate(examples['metadata']):
        if valid[i]:
            groups.setdefault(str(m['parent_trial_id']),[]).append(i)
    waveforms=[];ys=[];metadata=[]
    for parent,indices in sorted(groups.items()):
        if task in {'envelope','log_bandpower','waveform'}:
            indices=[max(indices,key=lambda i:examples['metadata'][i].get('cutoff_index',i))]
        y=np.asarray(examples['labels'][task])[indices]
        if kind=='classification' and len(np.unique(y))!=1:
            raise ValueError('Inconsistent parent labels')
        waveforms.append(examples['x'][indices]);ys.append(float(y.mean()));metadata.append(examples['metadata'][indices[0]])
    return waveforms,np.asarray(ys),metadata


def adaptation_rows(config,data,model,run_key,directory,specs,budgets,device,random_init=False):
    """Optional full encoder adaptation plus a matched random-init specialist."""
    from .models import encoder_config
    cfg=config.get('evaluation',{})
    steps=int(cfg.get('adaptation_steps',0))
    if steps<=0:
        return []
    rows=[];seed=run_key['optimizer_seed']
    kind_name='random_init_specialist' if random_init else 'limited_label_finetune'
    for task,spec in specs.items():
        pooled={split:trial_waveforms(values,task,spec['kind']) for split,values in data.items()}
        tr,dev,te=(pooled[s] for s in ('train','development','test'))
        if not len(tr[0]) or not len(te[0]) or (spec['kind']=='classification' and len(np.unique(tr[1]))<2):
            continue
        for budget in budgets:
            idx,account=independent_budget_indices(tr[2],tr[1],spec['kind'],budget,seed)
            waves=[tr[0][i] for i in idx];y=tr[1][idx]
            selected=None;selection=[]
            for regularization in cfg.get('nonlinear_grid',[.001,.01,.1]):
                probe=TrialNeuralProbe(waves,y,spec['kind'],encoder_config(config,data['train']['x'].shape[1]),seed,
                    steps=steps,batch_size=int(config.get('training',{}).get('batch_size',16)),regularization=regularization,
                    initial_encoder=None if random_init else model.encoder,device=device)
                value=_metric_score(dev[1],probe.predict(dev[0]),dev[2],spec['kind'],float(y.var())) if len(dev[0]) else 0.
                selection.append(dict(regularization=regularization,development_score=value))
                if selected is None or value>selected[0]:
                    selected=(value,probe,regularization)
            probe=selected[1]
            path=directory/'probes'/run_key['recipe']/f'seed_{seed}'/task/f'{kind_name}_{budget}.joblib'
            _save_probe(path,probe)
            prediction=probe.predict(te[0]);metrics,structures=structure_metrics(te[1],prediction,te[2],spec['kind'],float(y.var()))
            rows.append(dict(**run_key,task=task,kind=spec['kind'],probe=kind_name,budget=budget,metrics=metrics,
                structure_scores=structures,label_budget=account,status='capped' if account['capped'] else 'complete',
                selection=dict(candidates=selection,selected_regularization=selected[2],split='development'),
                optimizer_updates=steps,examples_processed=probe.examples_processed,fitted_probe=str(path),
                initialization='random' if random_init else 'pretrained encoder',
                excluded_from_primary=True))
    return rows


def aggregate_transfer_areas(areas,declared_tasks,config):
    """Descriptive coverage only; aggregation.summarize owns scientific inference.

    Avoid a second inference path that could disagree about missing optimizer
    seeds or treat a functional smoke as a scientific hypothesis test.
    """
    summaries=[]
    expected_seeds=set(config.get('training',{}).get('seeds',[]))
    keys=sorted({(a['recipe'],a['pretraining_budget'],a['probe']) for a in areas})
    for key in keys:
        sub=[a for a in areas if (a['recipe'],a['pretraining_budget'],a['probe'])==key]
        seeds=sorted({a['optimizer_seed'] for a in sub})
        expected=set(declared_tasks)
        complete=bool(expected) and (not expected_seeds or set(seeds)==expected_seeds) and all(
            {a['task'] for a in sub if a['optimizer_seed']==seed and a['status']=='complete'}==expected for seed in seeds)
        by_structure={}
        if complete:
            structures=set.intersection(*(set(a['structure_scores']) for a in sub))
            by_structure={structure:float(np.mean([a['structure_scores'][structure] for a in sub])) for structure in structures}
        summaries.append(dict(recipe=key[0],pretraining_budget=key[1],probe=key[2],
            status='complete descriptive coverage' if complete else 'Pending: declared tasks, budgets or optimizer seeds incomplete',
            tasks=sorted(expected),optimizer_seeds=seeds,expected_optimizer_seeds=sorted(expected_seeds),
            structure_scores=by_structure,mean=float(np.mean(list(by_structure.values()))) if by_structure else None))
    return dict(summaries=summaries,descriptive_only=True,
        authoritative_scientific_inference='experiments.eeg_representation.aggregation.summarize',
        independent_unit='circuit structure; average optimizer repeats and equally weight declared tasks',
        empirical_claim='This artifact performs no hypothesis tests; smoke outcomes cannot establish H1/H2')
