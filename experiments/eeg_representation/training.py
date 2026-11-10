"""Budget-accounted recipes, atomic checkpoints and non-destructive resume."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from collections import Counter

import numpy as np

RECIPES = ('features', 'specialist', 'masked_mse', 'masked_l1', 'spectral_hybrid',
           'momentum_contrastive', 'forecast_lowrank', 'forecast_diagonal', 'forecast_ar',
           'joint', 'progressive', 'reverse', 'random_order', 'reconstruction_supervised')
SUPERVISED = {'joint', 'progressive', 'reverse', 'random_order', 'reconstruction_supervised', 'specialist'}
ALIASES = {'psd': 'features', 'masked-mse':'masked_mse', 'masked-l1':'masked_l1',
           'contrastive':'momentum_contrastive', 'probabilistic':'forecast_lowrank',
           'hybrid':'reconstruction_supervised', 'random':'random_order'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def run_directory(root, protocol, recipe, seed, budget=None):
    return Path(root) / 'models' / protocol / recipe / f'seed_{seed}' / f'budget_{budget if budget is not None else "all"}'


def label_valid(y, kind):
    y = np.asarray(y)
    return np.isfinite(y) & ((y >= 0) if kind == 'classification' else True)


def task_specs(names, labels):
    try:
        from .tasks import TASKS
    except ImportError:
        TASKS = {}
    result = {}
    for name in names:
        spec = TASKS.get(name, {})
        if not isinstance(spec, dict):
            spec = vars(spec)
        kind = spec.get('kind', 'classification')
        values = np.asarray(labels[name])
        result[name] = dict(kind=kind, n_classes=int(spec.get('n_classes') or
                               (int(values[np.isfinite(values)].max()) + 1 if kind == 'classification' else 1)))
    return result


def independent_budget_indices(metadata, y, kind, budget, seed):
    """Nested independent paired-family quotas per class, before windowing.

    Regime siblings sharing event/phase/noise seeds cost one family within each
    condition. All eligible parents/windows in a selected family are retained.
    Ordering is shared across classes to preserve common-random-number pairing.
    Fixtures without paired_family_id fall back to the parent trial identifier.
    """
    valid=label_valid(y,kind)
    strata={}
    for i,(m,keep) in enumerate(zip(metadata,valid)):
        if not keep:
            continue
        family=str(m.get('paired_family_id',m['parent_trial_id']))
        stratum=int(y[i]) if kind=='classification' else 0
        strata.setdefault(stratum,set()).add(family)
    selected={};available={}
    for stratum,families in sorted(strata.items()):
        ordered=sorted(families,key=lambda family:digest([seed,family]))
        available[str(stratum)]=len(ordered)
        selected[stratum]=set(ordered if budget is None else ordered[:budget])
    indices=[]
    for i,m in enumerate(metadata):
        if not valid[i]:
            continue
        stratum=int(y[i]) if kind=='classification' else 0
        family=str(m.get('paired_family_id',m['parent_trial_id']))
        if family in selected.get(stratum,set()):
            indices.append(i)
    parents=sorted({str(metadata[i]['parent_trial_id']) for i in indices})
    families=sorted({str(metadata[i].get('paired_family_id',metadata[i]['parent_trial_id'])) for i in indices})
    return np.asarray(indices,dtype=int),dict(requested_per_condition=budget,
        independent_unit='paired family, including dependent regime/noise/window siblings',
        available_per_condition=available,selected_parents=parents,
        selected_independent_families=families,
        selected_families_per_condition={str(k):len(v) for k,v in selected.items()},
        capped=budget is not None and any(v<budget for v in available.values()))


def unique_random_order(tasks,order_index):
    """Frozen order identity is independent of paired optimizer randomness."""
    if not 0 <= order_index < math.factorial(len(tasks)):
        raise ValueError('Random-order index exceeds the number of distinct task permutations')
    rng=np.random.default_rng(974741)
    seen=set()
    while len(seen)<=order_index:
        order=tuple(rng.permutation(tasks).tolist())
        if order not in seen:
            seen.add(order)
            if len(seen)==order_index+1:
                return list(order)


def exposure_schedule(task_indices, steps, batch_size, seed, recipe, order=None, replay_fraction=.25, order_index=0):
    """Identical labelled tickets, regrouped only in time for order comparisons.

    Each pool is fully covered at least once. Requested steps is a *minimum*
    because truncating a pass would break unique-label parity. Replay reserves
    a declared fraction of tickets for subsequent stages; nothing is dropped.
    Joint uses shuffled homogeneous-task minibatches (equal-weight task loss).
    """
    tasks = sorted(task_indices)
    if not tasks:
        raise ValueError('No available source labels')
    rng = np.random.default_rng(seed)
    requested_per_task = max(1, math.ceil(steps / len(tasks)))
    batches = {}
    for task in tasks:
        pool = np.asarray(task_indices[task], dtype=int)
        if not len(pool):
            raise ValueError(f'No labelled examples for {task}')
        repetitions = max(1, math.ceil(requested_per_task * batch_size / len(pool)))
        tickets = np.concatenate([rng.permutation(pool) for _ in range(repetitions)])
        batches[task] = [tickets[j:j+batch_size].tolist() for j in range(0, len(tickets), batch_size)]
        if len(batches[task])>1 and len(batches[task][-1])==1:
            batches[task][-2].extend(batches[task].pop())
    chosen = list(order or tasks)
    if sorted(chosen) != tasks:
        raise ValueError('Source order must contain each eligible source task exactly once')
    if recipe == 'reverse':
        chosen.reverse()
    elif recipe == 'random_order':
        chosen=unique_random_order(tasks,order_index)
    stages = [[] for _ in tasks]
    if recipe in {'progressive', 'reverse', 'random_order'}:
        for stage, task in enumerate(chosen):
            current = batches[task]
            n_replay = int(len(current)*replay_fraction) if stage < len(tasks)-1 else 0
            n_now = len(current)-n_replay
            stages[stage].extend((task, b) for b in current[:n_now])
            for j, b in enumerate(current[n_now:]):
                stages[stage+1+j % (len(tasks)-stage-1)].append((task,b))
        for stage in stages:
            rng.shuffle(stage)
    else:
        flat = [(task, b) for task in tasks for b in batches[task]]
        rng.shuffle(flat)
        for i, batch in enumerate(flat):
            stages[min(len(tasks)-1, i*len(tasks)//len(flat))].append(batch)
    schedule = [dict(task=task, indices=indices, stage=stage) for stage, group in enumerate(stages) for task, indices in group]
    exposure = Counter((item['task'], i) for item in schedule for i in item['indices'])
    accounting = dict(order=chosen, order_index=order_index, order_seed_independent_of_optimizer=True, replay_fraction=replay_fraction,
        replay_capacity='complete source pool; no eviction', sampling='one equal-weight task per minibatch',
        requested_minimum_updates=steps, optimizer_updates=len(schedule),
        examples_processed=sum(len(s['indices']) for s in schedule),
        label_exposures={task:sum(v for (t,_),v in exposure.items() if t == task) for task in tasks},
        unique_examples={task:len(task_indices[task]) for task in tasks},
        ticket_hash=digest(sorted((task,int(i),int(v)) for (task,i),v in exposure.items())),
        schedule_hash=digest(schedule))
    return schedule, accounting


def feature_matrix(x, sample_hz=500):
    """Phase-invariant PSD, bandpower, autocorrelation and RMS-envelope features.

    No per-window variance normalization: class-dependent variance is retained.
    Features are standardized by the downstream training-only pipeline.
    """
    from scipy.signal import welch
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=sample_hz, nperseg=min(x.shape[-1], 512), axis=-1)
    features = [x.mean(-1), np.log(np.maximum(x.var(-1), np.finfo(float).tiny))]
    for low, high in ((1,4),(4,8),(8,12),(12,20),(20,30),(30,80)):
        keep = (f >= low) & (f < high)
        power = p[...,keep].sum(-1) * (f[1]-f[0]) if keep.any() else np.zeros(x.shape[:2])
        features.append(np.log(np.maximum(power, np.finfo(float).tiny)))
    for lag in (1, 5, 10, 25):
        lag = min(lag, x.shape[-1]-1)
        features.append((x[...,:-lag]*x[...,lag:]).mean(-1))
    for ms in (32,128,512):
        n = min(x.shape[-1], max(1,round(sample_hz*ms/1000)))
        count = x.shape[-1]//n
        envelope = np.sqrt((x[...,:count*n].reshape(*x.shape[:2],count,n)**2).mean(-1))
        features.extend((envelope.mean(-1), envelope.std(-1)))
    return np.concatenate(features, axis=1).astype(np.float64)


def _load_examples(root, config, split, protocol, stage='pretrain', budget=None, seed=0, **view_kwargs):
    from .data import load_examples
    return load_examples(root, config, split, protocol, stage=stage, budget=budget, seed=seed, **view_kwargs)


def _atomic_torch(path, payload):
    import torch
    tmp = Path(path).with_suffix(f'.{os.getpid()}.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _construct(recipe, ecfg, specs, target_shape):
    from .models import make_encoder, MultiTaskModel, ReconstructionModel, MomentumContrast, GaussianForecast
    import torch
    encoder = make_encoder(ecfg)
    if recipe in SUPERVISED:
        model = MultiTaskModel(encoder, specs)
        if recipe == 'reconstruction_supervised':
            model.reconstructor = ReconstructionModel(encoder)
    elif recipe.startswith('forecast_'):
        model = GaussianForecast(encoder, target_shape, kind=recipe.split('_',1)[1], rank=int(ecfg.get('covariance_rank',4)))
    elif recipe == 'momentum_contrastive':
        model = MomentumContrast(encoder)
    else:
        model = ReconstructionModel(encoder)
    return model


def load_fitted(path, device='cpu'):
    import torch
    path = Path(path)
    payload = torch.load(path/'checkpoint.pt', map_location=device, weights_only=False)
    model = _construct(payload['recipe'], payload['encoder_config'], payload['task_specs'], payload['target_shape'])
    model.load_state_dict(payload['model'])
    return model.to(device).eval(), payload


def _retention(model, examples, specs, norms, device, stage, step, batch_size):
    import torch
    from sklearn.metrics import balanced_accuracy_score, mean_squared_error
    rows = []
    model.eval()
    with torch.no_grad():
        for task, spec in specs.items():
            if task not in examples['labels']:
                continue
            y = np.asarray(examples['labels'][task])
            mask = label_valid(y, spec['kind'])
            if not mask.any():
                continue
            values = []
            for i in range(0,len(y),batch_size):
                values.append(model(torch.as_tensor(examples['x'][i:i+batch_size], device=device),task).cpu().numpy())
            prediction = np.concatenate(values)
            if spec['kind'] == 'classification':
                score = balanced_accuracy_score(y[mask], prediction[mask].argmax(-1))
                metric = 'balanced_accuracy'
            else:
                mean, scale = norms[task]
                score = mean_squared_error(y[mask],prediction[mask,0]*scale+mean)/scale**2
                metric = 'normalized_mse'
            rows.append(dict(task=task,stage=stage,step=step,split='development',metric=metric,value=float(score)))
    model.train()
    return rows


def fit(config, output_root, recipe, seed, device='cpu', budget=None,
        protocol='leave-task-label-out', resume=False):
    recipe = ALIASES.get(recipe,recipe)
    recipe_id = recipe
    order_index = int(recipe.split('__order_')[1]) if recipe.startswith('random_order__order_') else 0
    if recipe.startswith('random_order__order_'):
        recipe = 'random_order'
    if recipe not in RECIPES:
        raise ValueError(f'Unknown recipe {recipe}; available: {RECIPES}')
    if protocol not in {'leave-task-label-out','leave-generator-family-out'}:
        raise ValueError('Unknown transfer protocol')
    import torch
    from .models import encoder_config, gap_mask, masked_loss, spectral_envelope_loss
    torch.set_num_threads(int(config.get('training',{}).get('threads',config.get('training',{}).get('cpu_threads',1))))
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu or a validated CUDA ML environment')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    data = _load_examples(output_root, config, 'train', protocol, seed=seed)
    if set(data.get('allowed_input_columns',['eeg'])) != {'eeg'}:
        raise ValueError('Only EEG may enter the model input')
    if not len(data['x']):
        raise ValueError('No pretraining examples after protocol exclusions')
    cfg = config.get('training',{})
    source = list(cfg.get('source_tasks',['rhythm','carrier']))
    heldout = set(cfg.get('heldout_tasks',config.get('evaluation',{}).get('heldout_tasks',[])))
    if protocol == 'leave-task-label-out' and heldout.intersection(source):
        raise ValueError('Held-out task labels appear in the source objective')
    if recipe == 'specialist':
        source = [cfg.get('specialist_task',source[0])]
        if source[0] in heldout and protocol == 'leave-task-label-out':
            raise ValueError('Specialist source label is held out')
    source = [task for task in source if task in data['labels']]
    specs = task_specs(source,data['labels'])
    specs = {task: spec for task,spec in specs.items() if label_valid(data['labels'][task],spec['kind']).any()}
    selections, budgets = {}, {}
    for task,spec in specs.items():
        selections[task], budgets[task] = independent_budget_indices(data['metadata'],data['labels'][task],spec['kind'],budget,seed)
    ecfg = encoder_config(config, data['x'].shape[1])
    directory = run_directory(output_root,protocol,recipe_id,seed,budget)
    record = dict(config=config,recipe=recipe_id,seed=seed,protocol=protocol,budget=budget,order_index=order_index,
                  training_parent_hash=digest(sorted(set(str(m['parent_trial_id']) for m in data['metadata']))),
                  input_hash=hashlib.sha256(np.ascontiguousarray(data['x']).tobytes()).hexdigest(),
                  target_hash=hashlib.sha256(np.ascontiguousarray(data['future']).tobytes()).hexdigest(),
                  label_hash=digest({k:np.asarray(v).tolist() for k,v in data['labels'].items()}))
    signature = digest(record)
    metadata_path = directory/'run.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous['config_hash'] != signature:
            raise ValueError('Incompatible resume: configuration or dataset content changed')
        if not resume:
            raise FileExistsError(f'Refusing to overwrite {directory}; use --resume')
        if previous.get('status') == 'complete':
            return previous
    directory.mkdir(parents=True,exist_ok=True)
    metadata = dict(schema_version='eegrepr-model-v1',config_hash=signature,recipe=recipe_id,seed=seed,order_index=order_index,device=device,
                    protocol=protocol,budget=budget,status='running',label_budgets=budgets,
                    random_orders_requested=cfg.get('random_orders',3),distinct_order_capacity=math.factorial(len(specs)),
                    encoder_config=ecfg, training_tasks=list(specs),requested_source_tasks=source,
                    unavailable_source_tasks=[t for t in source if t not in specs],run_dir=str(directory),
                    training_parent_hash=record['training_parent_hash'],
                    training_structures=sorted(set(str(m['structure_id']) for m in data['metadata'])),
                    units='training-normalized EEG; no per-window standardization',
                    comparison_scope='functional smoke' if config.get('claim_status',config.get('profile'))=='smoke' else 'pending empirical evaluation')
    atomic_json(metadata_path,metadata)
    start = time.perf_counter()
    cpu_start = time.process_time()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    if recipe == 'features':
        import joblib
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression, Ridge
        z = feature_matrix(data['x'])
        fitted = {}
        for task,spec in specs.items():
            idx = selections[task]; y = np.asarray(data['labels'][task])[idx]
            if spec['kind']=='classification' and len(np.unique(y))<2:
                continue
            head = LogisticRegression(max_iter=500,random_state=seed) if spec['kind']=='classification' else Ridge(alpha=1.)
            fitted[task] = make_pipeline(StandardScaler(),head).fit(z[idx],y)
        tmp = directory/'features.tmp'; joblib.dump(fitted,tmp); os.replace(tmp,directory/'features.joblib')
        metadata.update(status='complete',elapsed_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu_start,parameter_count=0,
                        examples_processed=len(data['x']),optimizer_updates=0,
                        feature_dimension=z.shape[1],unique_recordings=len({m['parent_trial_id'] for m in data['metadata']}))
        atomic_json(metadata_path,metadata)
        return metadata
    model = _construct(recipe,ecfg,specs,data['future'].shape[1:]).to(device)
    if sum(p.numel() for p in model.parameters()) > int(config.get('model',{}).get('max_parameters',2_000_000)):
        raise ValueError('Complete recipe model exceeds the declared parameter ceiling')
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(cfg.get('lr',3e-4)),weight_decay=float(cfg.get('weight_decay',.01)))
    steps, batch = int(cfg.get('steps',100)), int(cfg.get('batch_size',16))
    if recipe in SUPERVISED:
        order = cfg.get('source_order',list(specs))
        order = [t for t in order if t in specs]
        schedule, accounting = exposure_schedule(selections,steps,batch,seed,recipe,order,
                                                 float(cfg.get('replay_fraction',.25)),order_index)
    else:
        if selections:
            # Match the joint comparator's exact EEG tickets and optimizer updates.
            # Permitted source-label eligibility defines the pool; targets are discarded.
            schedule,accounting=exposure_schedule(selections,steps,batch,seed,'joint',list(specs),
                float(cfg.get('replay_fraction',.25)))
            schedule=[dict(item,task=None) for item in schedule]
            accounting['eligibility_label_exposures']=accounting['label_exposures']
            accounting['label_exposures']={}
            accounting['supervision']='source labels used for eligibility and balanced subset only; no labels enter loss'
        else:
            rng = np.random.default_rng(seed)
            schedule = []
            n = len(data['x'])
            for _ in range(max(1,math.ceil(steps*batch/n))):
                permutation = rng.permutation(n)
                schedule.extend(dict(indices=permutation[i:i+batch].tolist(),task=None,stage=0) for i in range(0,n,batch))
            accounting = dict(optimizer_updates=len(schedule),examples_processed=sum(len(s['indices']) for s in schedule),
                              label_exposures={},unique_examples=n,unique_recordings=len({m['parent_trial_id'] for m in data['metadata']}),
                              comparison_scope='no source labels available: pure unsupervised full passes')
    norms = {}
    for task,spec in specs.items():
        if spec['kind']=='regression':
            y = np.asarray(data['labels'][task])[selections[task]]
            scale = float(y.std())
            norms[task] = (float(y.mean()),scale if scale > 0 else 1.)
    # Calibrate auxiliary scales on training data only, with explicitly capped weights.
    probe = torch.as_tensor(data['x'][:min(32,len(data['x']))],device=device)
    power_scale = float(probe.square().mean().cpu())
    spectral_weights = [min(10.,1/max(0.1,math.log1p(power_scale)**2))]*3 + [min(10.,1/max(.1,power_scale**2))]*2
    seen_indices={i for item in schedule for i in item['indices']}
    accounting['unique_parent_trials_processed']=len({data['metadata'][i]['parent_trial_id'] for i in seen_indices})
    accounting['unique_independent_families_processed']=len({data['metadata'][i].get('paired_family_id',data['metadata'][i]['parent_trial_id']) for i in seen_indices})
    accounting['unique_recording_pool_hash']=digest(sorted({data['metadata'][i]['parent_trial_id'] for i in seen_indices}))
    accounting['encoder_forward_passes_per_batch']=2 if recipe in {'momentum_contrastive','reconstruction_supervised'} else 1
    accounting['encoder_forward_examples']=accounting['examples_processed']*accounting['encoder_forward_passes_per_batch']
    accounting['compute_match']='optimizer steps and EEG-ticket exposures matched; extra objective forwards explicitly counted; measured time is not constrained'
    metadata.update(accounting=accounting,parameter_count=sum(p.numel() for p in model.parameters()),
                    encoder_parameter_count=sum(p.numel() for p in model.encoder.parameters()),
                    task_loss_normalization='classification cross-entropy; regression training-label z-score; equal per-task minibatches',
                    regression_normalization=norms,spectral_weights=spectral_weights,
                    spectral_weight_cap=10.,auxiliary_weight=float(cfg.get('auxiliary_weight',1.)),
                    mask=dict(fraction=cfg.get('mask_fraction',.4),gap_ms=cfg.get('mask_gap_ms',128)),
                    source_order_selection=cfg.get('order_selected_on',cfg.get('source_order_selection','predeclared candidate; no test selection')))
    history,retention,completed = [],[],0
    checkpoint = directory/'checkpoint.pt'
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint,map_location=device,weights_only=False)
        if saved['config_hash'] != signature:
            raise ValueError('Incompatible checkpoint configuration hash')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        completed=saved['completed'];history=saved['history'];retention=saved['retention']
        torch.set_rng_state(saved['rng'].cpu())
        if device.startswith('cuda') and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    dev = None
    if recipe in SUPERVISED:
        dev = _load_examples(output_root,config,'development',protocol,seed=seed)
        if not retention and len(dev['x']):
            retention.extend(_retention(model,dev,specs,norms,device,-1,0,batch))
    model.train()
    for step in range(completed,len(schedule)):
        item=schedule[step];idx=item['indices']
        x=torch.as_tensor(data['x'][idx],device=device)
        optimizer.zero_grad(set_to_none=True)
        if recipe in SUPERVISED:
            task=item['task']; y=torch.as_tensor(np.asarray(data['labels'][task])[idx],device=device)
            prediction=model(x,task)
            if specs[task]['kind']=='classification':
                loss=torch.nn.functional.cross_entropy(prediction,y.long())
            else:
                mean,scale=norms[task]
                loss=torch.nn.functional.mse_loss(prediction[:,0],((y-mean)/scale).float())
            if recipe=='reconstruction_supervised':
                mask=gap_mask(x,float(cfg.get('mask_fraction',.4)),max(1,round(cfg.get('mask_gap_ms',128)*.5)))
                recon=model.reconstructor(x.masked_fill(mask,0))
                loss=loss+float(cfg.get('auxiliary_weight',1.))*masked_loss(recon,x,mask)
        elif recipe.startswith('forecast_'):
            target=torch.as_tensor(data['future'][idx],device=device).flatten(1)
            loss=-model(x).log_prob(target).mean()/target.shape[-1]
        elif recipe=='momentum_contrastive':
            if len(x)<2:
                x=torch.cat([x,x],0)
            loss=model.loss(x)
        else:
            mask=gap_mask(x,float(cfg.get('mask_fraction',.4)),max(1,round(cfg.get('mask_gap_ms',128)*.5)))
            recon=model(x.masked_fill(mask,0))
            loss=masked_loss(recon,x,mask,'l1' if recipe=='masked_l1' else 'mse')
            if recipe=='spectral_hybrid':
                completed_signal=torch.where(mask,recon,x)
                loss=loss+float(cfg.get('auxiliary_weight',1.))*spectral_envelope_loss(completed_signal,x,weights=spectral_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite training loss at step {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if recipe=='momentum_contrastive':
            model.update_teacher(float(cfg.get('teacher_momentum',.99)))
        history.append(dict(step=step+1,stage=item['stage'],task=item['task'],loss=float(loss.detach().cpu())))
        stage_end=step+1==len(schedule) or schedule[step+1]['stage']!=item['stage']
        if stage_end and dev is not None and len(dev['x']):
            retention.extend(_retention(model,dev,specs,norms,device,item['stage'],step+1,batch))
        if stage_end or (step+1)%int(cfg.get('checkpoint_every',50))==0:
            _atomic_torch(checkpoint,dict(config_hash=signature,recipe=recipe,encoder_config=ecfg,
                task_specs=specs,target_shape=data['future'].shape[1:],model=model.state_dict(),
                optimizer=optimizer.state_dict(),completed=step+1,history=history,retention=retention,
                regression_normalization=norms,rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None))
    _atomic_torch(directory/'encoder.pt',dict(encoder_config=ecfg,state_dict=model.encoder.state_dict(),config_hash=signature))
    atomic_json(directory/'learning_curve.json',history)
    atomic_json(directory/'retention.json',retention)
    import resource
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    metadata.update(status='complete',elapsed_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu_start,
                    peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,optimizer_updates=len(schedule),
                    examples_processed=accounting['examples_processed'],unique_recordings=len({m['parent_trial_id'] for m in data['metadata']}),
                    peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else None)
    atomic_json(metadata_path,metadata)
    return metadata


def fit_suite(config,output_root,recipes=None,seeds=None,budgets=None,device='cpu',protocol='leave-task-label-out',resume=False):
    cfg=config.get('training',{})
    recipes=recipes or cfg.get('recipes',['features','masked_mse','joint','progressive'])
    seeds=seeds if seeds is not None else cfg.get('seeds',[0,1])
    budgets=budgets if budgets is not None else cfg.get('pretraining_budgets',[None])
    expanded=[]
    for recipe in recipes:
        expanded.extend([recipe] + [f'random_order__order_{i}' for i in range(1,min(int(cfg.get('random_orders',3)),math.factorial(len(cfg.get('source_tasks',['rhythm','carrier'])))))]) if recipe=='random_order' else expanded.append(recipe)
    return [fit(config,output_root,r,s,device,b,protocol,resume) for s in seeds for b in budgets for r in expanded]
