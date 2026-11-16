"""V6 primary task/reconstruction disagreement and labelled legacy replication.

Primary banks are deterministic Feature18 and train-PCA, d8/B32. Legacy 200-update
forecast repeats remain a secondary procedural replication, not objective evidence.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import warnings
import numpy as np
from scipy.special import logsumexp
from ..artifacts import atomic_npz
from ..targets import canonical_hash
from ...storage import atomic_json,file_hash,read_json
from ..v4 import retention as legacy
from .models import load_panel,configure_torch,first_targets,context_weights
from ..v4.retention import balanced_label_subset,projection_fit,project,fit_range_rule,quantization_attribution,_classifier_score,_equal_circuit
SPLITS=('train','development','evaluation')
DEFAULTS=dict(job_id='A0__T0_l10_d128',dimensions=[8,32],code_bits=[32,128],
    range_rules=['quantile_005_995','robust_iqr'],regularization=[.01,.1,1.,10.,100.,1000.,10000.],
    label_budgets=['all'],tasks=['waveform_reconstruction','carrier','rhythm','four_class'],
    legacy_forecast=True,legacy_seeds=[0,1,2],qualified_banks=False,
    qualified_seeds=[0,1,2],qualified_max_steps=10000,qualified_patience=500,
    qualified_eval_every=50,qualified_batch_size=16,qualified_learning_rate=.001,
    dtype='float32',threads=1)


def settings(config):
    options=copy.deepcopy(DEFAULTS);options.update(config.get('retention',{}))
    if options['qualified_banks'] and len(set(options['qualified_seeds']))<3:raise ValueError('Qualified representation diagnostics require at least three fitting repeats')
    if not 1<=options['qualified_max_steps']<=10000:raise ValueError('Representation optimization exceeds 10000-step bound')
    if not set(DEFAULTS['regularization']).issubset(options['regularization']):raise ValueError('Frozen readout grid truncated')
    if 8 not in options['dimensions'] or 32 not in options['code_bits'] or 'quantile_005_995' not in options['range_rules']:raise ValueError('Frozen d8/B32 primary endpoint missing')
    if options['dtype'] not in ('float32','float64'):raise ValueError('Unsupported training arithmetic precision')
    if options['job_id']!=DEFAULTS['job_id'] and config.get('claim_status')!='engineering':raise ValueError('H3 primary acquisition is frozen to A0')
    if 'all' not in options['label_budgets'] or not set(DEFAULTS['tasks']).issubset(options['tasks']):raise ValueError('Primary carrier/reconstruction and four-class/rhythm controls required')
    return options


def evaluate_readout(panel, values, task, budget, alphas, destination, resume=False):
    """Dev-selected linear decoder, with explicit capacity and split predictions."""
    destination=Path(destination);marker=destination/'scores.json'
    if resume and marker.exists():
        previous=read_json(marker)
        for filename,digest in previous.get('artifact_hashes',{}).items():
            if file_hash(destination/filename)!=digest:raise ValueError('Readout resume checksum mismatch')
        if previous.get('artifact_hashes'):return previous
    classification = task in ('rhythm', 'carrier', 'four_class')
    key = ('X' if task == 'waveform_reconstruction' else 'Y' if task == 'future_mean' else 'task_' + task)
    if any(key not in panel[s] for s in SPLITS):
        return dict(status='unavailable', reason='task label unavailable', task=task, label_budget=budget)
    y, masks = {}, {}
    fitting_splits=('train','development')
    for s in fitting_splits:
        y[s] = np.asarray(panel[s][key])
        if task == 'future_mean':
            first = np.argmax(panel[s]['future_mask'], axis=1)
            y[s] = y[s][np.arange(len(first)), first]
        if not classification and y[s].ndim == 1:
            y[s] = y[s][:, None]
        masks[s] = np.isfinite(y[s]).all(1) if y[s].ndim > 1 else np.isfinite(y[s])
        if task == 'carrier':
            masks[s] &= y[s] >= 0
        y[s] = y[s][masks[s]]
    if any(not masks[s].any() for s in fitting_splits):
        return dict(status='unavailable', reason='empty eligible task population', task=task)
    groups = {s: np.asarray(panel[s]['structure_id'])[masks[s]] for s in fitting_splits}
    classes = np.asarray({'rhythm':[0,1], 'carrier':[0,1,2], 'four_class':[0,1,2,3]}[task]) if classification else np.asarray([0])
    if classification and any(set(y[s].tolist()) != set(classes.tolist()) for s in fitting_splits):
        return dict(status='unavailable', reason='required class coverage differs across splits', task=task)
    indices, sampling = balanced_label_subset(y['train'] if classification else np.zeros(len(y['train'])), groups['train'], budget, required_classes=classes)
    if indices is None:
        return dict(sampling, task=task, label_budget=budget)
    a = {s: np.asarray(values[s], float)[masks[s]] for s in fitting_splits}
    offset = a['train'][indices].mean(0); scale = a['train'][indices].std(0); scale = np.where(scale > 1e-12, scale, 1.)
    z = {s: (a[s] - offset) / scale for s in fitting_splits}; fitx = z['train'][indices]; fity = y['train'][indices]
    candidates = []; prediction_arrays = {}; model_arrays = dict(offset=offset, scale=scale, label_indices=indices)
    if not classification:
        ycenter = fity.mean(0); xc = fitx.mean(0); u, singular, vt = np.linalg.svd(fitx - xc, full_matrices=False)
    for alpha in alphas:
        alpha = float(alpha); predictions = {}; caught = []
        if classification:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1 / alpha, solver='lbfgs', max_iter=2000)
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always'); clf.fit(fitx, fity)
                caught = [str(w.message) for w in ww]
            if not np.array_equal(clf.classes_, classes):
                raise ValueError('Class order changed during readout fitting')
            for s in fitting_splits:
                logits = z[s] @ clf.coef_.T + clf.intercept_
                predictions[s] = np.c_[np.zeros(len(logits)), logits[:, 0]] if logits.shape[1] == 1 else logits
            devloss, _ = _classifier_score(y['development'], predictions['development'], classes)
            arrays = dict(coef=clf.coef_, intercept=clf.intercept_, classes=classes)
            effective = int(np.linalg.matrix_rank(fitx)); parameters = int(clf.coef_.size + clf.intercept_.size)
        else:
            # Save a dual numeric decoder to avoid O(input_width * output_width) storage.
            dual = (u * (1 / (singular ** 2 + alpha))) @ (u.T @ (fity - ycenter))
            for s in fitting_splits:
                predictions[s] = ((z[s] - xc) @ (fitx - xc).T) @ dual + ycenter
            if task == 'future_mean':
                dy = panel['development']['Y'][masks['development']]
                dm = panel['development']['future_mask'][masks['development']]
                devloss = np.where(dm, np.mean((predictions['development'][:, None] - dy) ** 2, axis=-1), 0).sum(1) / dm.sum(1)
            else:
                devloss = np.mean((predictions['development'] - y['development']) ** 2, axis=1)
            arrays = dict(fit_basis=fitx - xc, dual_coef=dual, xcenter=xc, intercept=ycenter)
            effective = float(np.sum(singular ** 2 / (singular ** 2 + alpha)))
            parameters = int((fitx.shape[1] + 1) * np.prod(fity.shape[1:]))
        score = _equal_circuit(devloss, groups['development'])
        candidates.append(dict(alpha=alpha, development_equal_circuit_loss=score, warnings=caught,
                               nominal_parameters=parameters, effective_input_df_or_rank=effective))
        tag = str(alpha).replace('.', 'p')
        for s in fitting_splits:
            prediction_arrays[s + '__alpha_' + tag] = np.asarray(predictions[s], dtype=np.float64)
        for array_key, value in arrays.items():
            model_arrays['candidate_' + tag + '__' + array_key] = value
        if len(candidates) == 1 or score < selected_score:
            selected_score, selected_alpha, selected, selected_arrays = score, alpha, predictions, arrays
    if not classification:
        # Float64 predictions retain full standard numeric precision; physical targets remain exact.
        selected = {s: selected[s].astype(np.float64) for s in fitting_splits}
    result = dict(prediction_storage='float64 predictions and numeric weights for every tried readout candidate, all splits',
        target_arrays='retention/task_targets.npz, shared exact target arrays and IDs; no repeated target copies per decoder', status='complete', task=task, label_budget=budget, sampling=sampling,
        selected_alpha=selected_alpha, candidates=candidates, decoder_family='linear logistic' if classification else 'linear ridge',
        selection='equal circuit-weighted development loss; evaluation never selects',
        cross_validation='none; genuinely circuit-disjoint train/development/evaluation; frozen-upstream diagnostic, not end-to-end CV',
        interpretation='decoder-accessible task information; cross entropy is not measured total mutual information',
        label_population='rhythmic contexts only' if task == 'carrier' else 'all eligible contexts')
    atomic_npz(destination / 'readout_weights.npz', **model_arrays, **selected_arrays)
    atomic_json(destination / 'selection.json', dict(status='frozen', selected_alpha=selected_alpha, selection_split='development', evaluation_selection=False))

    # No evaluation target or representation was used above. Freeze numeric
    # readout weights/selection before accessing held-out labels or predictions.
    s='evaluation';y[s]=np.asarray(panel[s][key])
    if task=='future_mean':
        first=np.argmax(panel[s]['future_mask'],axis=1);y[s]=y[s][np.arange(len(first)),first]
    if not classification and y[s].ndim==1:y[s]=y[s][:,None]
    masks[s]=np.isfinite(y[s]).all(1) if y[s].ndim>1 else np.isfinite(y[s])
    if task=='carrier':masks[s]&=y[s]>=0
    y[s]=y[s][masks[s]];groups[s]=np.asarray(panel[s]['structure_id'])[masks[s]]
    if not masks[s].any() or classification and set(y[s].tolist())!=set(classes.tolist()):
        result.update(status='unavailable',reason='required evaluation class/population coverage unavailable after freezing readout')
        atomic_json(destination/'scores.json',result);return result
    z[s]=(np.asarray(values[s],float)[masks[s]]-offset)/scale
    for alpha in alphas:
        tag=str(float(alpha)).replace('.','p');prefix='candidate_'+tag+'__'
        arrays={k[len(prefix):]:v for k,v in model_arrays.items() if k.startswith(prefix)}
        if classification:
            logits=z[s]@arrays['coef'].T+arrays['intercept']
            pred=np.c_[np.zeros(len(logits)),logits[:,0]] if logits.shape[1]==1 else logits
        else:pred=((z[s]-arrays['xcenter'])@arrays['fit_basis'].T)@arrays['dual_coef']+arrays['intercept']
        prediction_arrays[s+'__alpha_'+tag]=np.asarray(pred,np.float64)
        if float(alpha)==selected_alpha:selected[s]=np.asarray(pred,np.float64)
    for s in SPLITS:
        ids = np.asarray(panel[s]['context_id'])[masks[s]]
        prediction_arrays[s + '_context_id'] = ids; prediction_arrays[s + '_structure_id'] = groups[s]
        if classification:
            prediction_arrays[s + '_target'] = y[s]
        prediction_arrays[s + '_selected_prediction'] = selected[s]
        if classification:
            loss, score = _classifier_score(y[s], selected[s], classes)
            score['cross_entropy_bits'] = _equal_circuit(loss, groups[s])
        else:
            if task == 'future_mean':
                dy = panel[s]['Y'][masks[s]]; dm = panel[s]['future_mask'][masks[s]]
                loss = np.where(dm, np.mean((selected[s].astype(float)[:, None] - dy) ** 2, axis=-1), 0).sum(1) / dm.sum(1)
                baseline = np.where(dm, np.mean((dy - fity.mean(0)) ** 2, axis=-1), 0).sum(1) / dm.sum(1)
                prediction_arrays[s + '_future_mask'] = dm
            else:
                loss = np.mean((selected[s].astype(float) - y[s]) ** 2, axis=1)
                baseline = np.mean((y[s] - fity.mean(0)) ** 2, axis=1)
            score = dict(mse=_equal_circuit(loss, groups[s]), train_mean_baseline_mse=_equal_circuit(baseline, groups[s]))
            if task in ('waveform_reconstruction', 'future_mean') and 'normalization' in panel.get('manifest', {}):
                scale_key = 'x_scale' if task == 'waveform_reconstruction' else 'y_scale'
                target_scale = float(panel.get('manifest', {}).get('normalization', {}).get(scale_key, 1.))
                score['physical_mse'] = score['mse'] * target_scale ** 2
                score['physical_train_mean_baseline_mse'] = score['train_mean_baseline_mse'] * target_scale ** 2
                score['physical_unit'] = 'V^2' if task == 'waveform_reconstruction' else 'squared target units from panel manifest'
            score['normalized_by_train_mean_risk'] = score['mse'] / score['train_mean_baseline_mse'] if score['train_mean_baseline_mse'] > 0 else None
        prediction_arrays[s + '_selected_context_loss'] = loss
        score['per_circuit_loss'] = {str(c): float(np.mean(loss[groups[s] == c])) for c in np.unique(groups[s])}
        score['contexts'] = len(ids); score['circuits'] = len(np.unique(groups[s])); result[s] = score
    if classification:
        prediction_arrays['class_order'] = classes
    atomic_npz(destination / 'predictions.npz', **prediction_arrays)
    atomic_npz(destination / 'readout_weights.npz', **model_arrays, **selected_arrays)
    result['artifact_hashes']={name:file_hash(destination/name) for name in ('predictions.npz','readout_weights.npz','selection.json')}
    atomic_json(destination / 'scores.json', result)
    return result


def _qualified(panel,options,destination,device,resume=False):
    """Optional bounded same-architecture diagnostics, dev-selected; no objective claim."""
    from ...models import make_encoder,ReconstructionModel,gap_mask,masked_loss
    torch,dtype=configure_torch(device,options['dtype'],options['threads'])
    results={};x={s:torch.as_tensor(panel[s]['X'][:,None],dtype=dtype,device=device) for s in SPLITS}
    y={s:torch.as_tensor(first_targets(panel[s]),dtype=dtype,device=device) for s in ('train','development')}
    for seed in options['qualified_seeds']:
        for kind in ('forecast','masked_mse'):
            name=f'qualified_{kind}__seed_{seed}';frozen=destination/'banks'/(name+'_frozen.json')
            if resume and frozen.exists():
                record=read_json(frozen)
                for rec in record['files']:
                    if file_hash(destination/rec['path'])!=rec['sha256']:raise ValueError('Qualified bank checkpoint checksum mismatch')
                with np.load(destination/record['representations'],allow_pickle=False) as z:values={s:z[s] for s in SPLITS}
                results[name]=dict(values=values,metadata=record['metadata']);continue
            torch.manual_seed(seed);encoder=make_encoder(legacy.DEFAULTS['encoder_config']).to(device=device,dtype=dtype)
            model=(ReconstructionModel(encoder) if kind=='masked_mse' else torch.nn.Sequential(encoder,torch.nn.Linear(128,y['train'].shape[-1]))).to(device=device,dtype=dtype)
            rng=np.random.default_rng(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=options['qualified_learning_rate'],weight_decay=.01)
            torch.manual_seed(seed+937);devmask=gap_mask(x['development'],.4,64)
            history=[];best=None;stale=0;updates=0;start=time.perf_counter()
            weights=torch.as_tensor(context_weights(panel['development']),dtype=dtype,device=device)
            for step in range(options['qualified_max_steps']+1):
                if step%options['qualified_eval_every']==0 or step==options['qualified_max_steps']:
                    model.eval()
                    with torch.no_grad():
                        pieces=[]
                        for begin in range(0,len(x['development']),options['qualified_batch_size']):
                            end=begin+options['qualified_batch_size'];xx=x['development'][begin:end]
                            if kind=='forecast':part=((model(xx)-y['development'][begin:end])**2).mean(-1)
                            else:
                                mm=devmask[begin:end];output=model(xx.masked_fill(mm,0));part=((output-xx)**2*mm).sum((1,2))/mm.sum((1,2))
                            pieces.append(part)
                        loss=torch.cat(pieces);score=float((loss*weights).sum().cpu())
                    history.append(dict(step=step,development_loss=score,elapsed_seconds=time.perf_counter()-start))
                    if best is None or score<best[0]-1e-12:
                        best=(score,step,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()});stale=0
                    else:stale+=options['qualified_eval_every']
                    if stale>=options['qualified_patience']:break
                if step==options['qualified_max_steps']:break
                idx=rng.choice(len(x['train']),min(options['qualified_batch_size'],len(x['train'])),replace=True,p=context_weights(panel['train']))
                model.train();optimizer.zero_grad(set_to_none=True);a=x['train'][idx]
                if kind=='forecast':loss=(model(a)-y['train'][idx]).square().mean()
                else:
                    mask=gap_mask(a,.4,64);loss=masked_loss(model(a.masked_fill(mask,0)),a,mask)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite qualified representation loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();updates+=1
            model.load_state_dict(best[2]);model.eval();name=f'qualified_{kind}__seed_{seed}'
            atomic_npz(destination/'banks'/(name+'_model.npz'),**{'tensor__'+k:v.numpy() for k,v in best[2].items()})
            atomic_json(destination/'banks'/(name+'_selection.json'),dict(status='frozen',selection_split='development',selected_step=best[1],device=device,dtype=options['dtype']))
            values={}
            with torch.no_grad():
                for split in SPLITS:
                    values[split]=np.concatenate([encoder(x[split][i:i+16]).cpu().numpy() for i in range(0,len(x[split]),16)])
            results[name]=dict(values=values,metadata=dict(method='bounded dev-checkpoint diagnostic; H5/H6 deferred',native_width=128,
                optimization_seed=seed,parameter_count=sum(v.numel() for v in model.parameters()),optimizer_updates=updates,selected_step=best[1],
                history=history,device=device,dtype=options['dtype'],amp=False,tf32=False,
                exposure_caveat='early stopping can produce unequal exposure; no objective-superiority claim'))
            repfile=destination/'banks'/(name+'_representations.npz');atomic_npz(repfile,**values)
            paths=[repfile,destination/'banks'/(name+'_model.npz'),destination/'banks'/(name+'_selection.json')]
            atomic_json(frozen,dict(metadata=results[name]['metadata'],representations=str(repfile.relative_to(destination)),
                files=[dict(path=str(p.relative_to(destination)),sha256=file_hash(p)) for p in paths]))
    return results


def primary_components(rows,panel):
    """Raw circuit components; shared estimated reconstruction denominator stays explicit."""
    primary='d8_B32_quantile_005_995'
    lookup={(r['representation'],r.get('variant'),r.get('task'),str(r.get('label_budget'))):r for r in rows if r['status']=='complete'}
    selected={}
    for bank in ('feature18','pca'):
        for task in ('carrier','waveform_reconstruction'):
            key=(bank,primary,task,'all')
            if key not in lookup:raise ValueError('Missing frozen primary endpoint '+str(key))
            selected[(bank,task)]=lookup[key]
    data=panel['evaluation'];baseline=np.asarray(panel['train']['X'],float).mean(0)
    per_baseline=np.mean((np.asarray(data['X'],float)-baseline)**2,axis=1)
    circuits=np.unique(data['structure_id']);result=[]
    global_baseline=float(context_weights(data)@per_baseline)
    for c in circuits:
        key=str(c);take=data['structure_id']==c
        f=selected[('feature18','waveform_reconstruction')]['evaluation']['per_circuit_loss'][key]
        p=selected[('pca','waveform_reconstruction')]['evaluation']['per_circuit_loss'][key]
        task=selected[('pca','carrier')]['evaluation']['per_circuit_loss'][key]-selected[('feature18','carrier')]['evaluation']['per_circuit_loss'][key]
        result.append(dict(structure_id=int(c),task_difference_bits=task,reconstruction_mse_feature=f,reconstruction_mse_pca=p,
            reconstruction_baseline_mse=float(per_baseline[take].mean()),reconstruction_difference_nmse=(f-p)/global_baseline))
    return dict(status='complete',per_circuit=result,task='conditional carrier CE_PCA minus CE_feature18',
        reconstruction='difference of mean MSEs divided by shared equal-circuit evaluated training-mean baseline risk',
        reconstruction_denominator=global_baseline,denominator_estimated_from_evaluation=True,
        practical_margins=dict(task_bits=.20,reconstruction_nmse=.02),primary_family='A',one_sided_component_alpha=.025,
        inference='deferred to confirm-statistics; ratio influence must account for estimated reconstruction denominator',
        independent_circuits=len(circuits),training_conditioning='conditional on frozen training dataset and selection policy')


def retention(config,root,device='cpu',resume=False):
    root=Path(root);options=settings(config)
    job=options['job_id'];all_panel,manifest=load_panel(root,job);panel={s:dict(all_panel[s]) for s in SPLITS}
    for s in SPLITS:
        # Explicit generator four-class control: homogeneous=0, 10/16/24Hz=1/2/3.
        panel[s]['task_four_class']=np.asarray(panel[s]['task_carrier'])+1
    panel['manifest']=manifest;dest=root/'retention';marker=dest/'summary.json'
    sig=canonical_hash(dict(options=options,device=device,source=file_hash(Path(__file__)),legacy_source=file_hash(Path(legacy.__file__)),
        dependencies={p.name:file_hash(p) for p in [Path(__file__).parents[2]/'training.py',Path(__file__).parents[2]/'models.py',Path(__file__).parent/'models.py']},
        inputs={s:file_hash(root/'derived'/job/f'{s}.npz') for s in SPLITS}))
    guard=dest/'analysis_identity.json'
    if guard.exists() and (not resume or read_json(guard)['signature']!=sig):raise ValueError('Retention resume config/source/device/dtype mismatch')
    if marker.exists():
        old=read_json(marker)
        for r in old['files']:
            if file_hash(root/r['path'])!=r['sha256']:raise ValueError('Retention artifact checksum mismatch')
        return old
    atomic_json(guard,dict(signature=sig,options=options,device=device,dtype=options['dtype']))
    atomic_npz(dest/'task_targets.npz',**{s+'_'+k:v for s in SPLITS for k,v in panel[s].items()})
    legacy_options=copy.deepcopy(legacy.DEFAULTS);legacy_options.update(banks=['feature18','pca']+(['forecast'] if options['legacy_forecast'] else []),
        optimization_seeds=options['legacy_seeds'],representation_steps=200,pooling_banks=[],threads=options['threads'])
    configure_torch(device,options['dtype'],options['threads'])
    if options['dtype']=='float64' and options['legacy_forecast']:raise ValueError('Legacy exact procedural bank is float32; disable it for explicit all-fp64 training')
    cache=dest/'legacy_bank_cache.json'
    if resume and cache.exists():
        cached=read_json(cache)
        if cached['signature']!=sig:raise ValueError('Cached bank contract mismatch')
        for record in cached['files']:
            if file_hash(dest/record['path'])!=record['sha256']:raise ValueError('Cached bank checksum mismatch')
        banks={};architecture=cached['architecture']
        for name,meta in cached['banks'].items():
            with np.load(dest/meta['path'],allow_pickle=False) as z:
                banks[name]=dict(values={s:z[s] for s in SPLITS},metadata=meta['metadata'])
                if 'inverse__offset' in z.files:banks[name]['inverse']={k.removeprefix('inverse__'):z[k] for k in z.files if k.startswith('inverse__')}
    else:
        banks,architecture=legacy._new_banks(panel,dest,legacy_options,device)
        banks={('legacy200_'+k if k.startswith('forecast') else k):v for k,v in banks.items()}
        cache_rows={}
        for name,bank in banks.items():
            path=dest/'banks'/(name+'_cache.npz');atomic_npz(path,**bank['values'],**{'inverse__'+k:v for k,v in bank.get('inverse',{}).items()})
            cache_rows[name]=dict(path=str(path.relative_to(dest)),metadata=bank['metadata'])
        files=[dict(path=str(p.relative_to(dest)),sha256=file_hash(p)) for p in sorted((dest/'banks').glob('*.npz'))]
        atomic_json(cache,dict(signature=sig,architecture=architecture,banks=cache_rows,files=files))
    if options['qualified_banks']:banks.update(_qualified(panel,options,dest,device,resume=resume))
    direct={s:np.asarray(panel[s]['X'],float) for s in SPLITS}
    banks['raw_history']=dict(values=direct,metadata=dict(method='exact identity history; fitted ridge readout is not an information oracle'))
    rows=[];attribution=[];inverse_rows=[]
    for name,bank in banks.items():
        directory=dest/'banks'/name;values=bank['values'];width=values['train'].shape[1]
        atomic_npz(directory/'representations.npz',**values,**{s+'_context_id':panel[s]['context_id'] for s in SPLITS})
        atomic_json(directory/'metadata.json',dict(bank['metadata'],native_width=width))
        if 'inverse' in bank:atomic_npz(directory/'native_pca_inverse.npz',**bank['inverse'])
        variants=[('native_unquantized',values,dict(dimension=width,code_bits=None))]
        if name=='raw_history':inverse_rows.append(dict(representation=name,status='complete',endpoint='exact raw-history identity',mse=0.))
        if 'inverse' in bank:
            inv=bank['inverse'];inverse_rows.append(dict(representation=name,status='complete',dimension=width,code_bits=None,
                endpoint='deterministic native-width train-PCA inverse',scores={s:float(np.mean((values[s]@inv['components']+inv['offset']-direct[s])**2)) for s in SPLITS}))
        for dimension in ([] if name=='raw_history' else options['dimensions']):
            try:p=projection_fit(values['train'],dimension)
            except ValueError as exc:
                rows.append(dict(status='unsupported',representation=name,dimension=dimension,reason=str(exc)));continue
            z={s:project(values[s],p) for s in SPLITS};atomic_npz(directory/f'd{dimension}'/'projection.npz',**{k:v for k,v in p.items() if isinstance(v,np.ndarray)})
            projection_report={s:dict(native_coordinate_truncation_mse=float(np.mean((values[s]-(z[s]@p['components']+p['center']))**2))) for s in SPLITS}
            variants.append((f'd{dimension}_unquantized',z,dict(dimension=dimension,code_bits=None,projection=projection_report)))
            if 'inverse' in bank:
                inv=bank['inverse'];inverse_rows.append(dict(representation=name,status='complete',dimension=dimension,code_bits=None,
                    endpoint='deterministic projected train-PCA inverse',scores={s:float(np.mean(((z[s]@p['components']+p['center'])@inv['components']+inv['offset']-direct[s])**2)) for s in SPLITS}))
            for rule in options['range_rules']:
                for bits in options['code_bits']:
                    book=fit_range_rule(z['train'],bits,rule);decoded={};arrays={};reports={}
                    variant=f'd{dimension}_B{bits}_{rule}'
                    for s in SPLITS:
                        packed,decoded[s],reports[s]=quantization_attribution(z[s],book)
                        arrays.update({s+'_packed':packed,s+'_decoded':decoded[s],s+'_context_id':panel[s]['context_id']})
                    atomic_npz(directory/variant/'codes.npz',**arrays)
                    atomic_json(directory/variant/'codebook.json',dict(book,clipping=reports,shared_weights_excluded_from_payload=True))
                    variants.append((variant,decoded,dict(dimension=dimension,code_bits=bits,range_rule=rule,projection=projection_report)))
                    if 'inverse' in bank:
                        inv=bank['inverse'];inverse_rows.append(dict(representation=name,status='complete',dimension=dimension,code_bits=bits,range_rule=rule,
                            endpoint='deterministic quantized train-PCA inverse',scores={s:float(np.mean(((decoded[s]@p['components']+p['center'])@inv['components']+inv['offset']-direct[s])**2)) for s in SPLITS}))
                    attribution.append(dict(representation=name,variant=variant,clipping=reports))
        for variant,a,meta in variants:
            for task in options['tasks']:
                for budget in options['label_budgets']:
                    score=evaluate_readout(panel,a,task,budget,options['regularization'],directory/variant/task/('labels_'+str(budget)),resume=resume)
                    rows.append(dict(score,representation=name,variant=variant,**meta))
    primary=primary_components(rows,panel);atomic_json(dest/'primary_components.json',primary)
    secondary=[]
    for row in rows:
        if row['status']=='complete' and row['representation'].startswith('legacy200_forecast') and row['variant']=='d8_B32_quantile_005_995' and row['task']=='carrier' and row['label_budget']=='all':
            secondary.append(dict(bank=row['representation'],per_circuit_ce=row['evaluation']['per_circuit_loss'],scope='secondary legacy200 procedural replication, not an extra unadjusted primary'))
    files=[dict(path=str(p.relative_to(root)),sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(dest.rglob('*')) if p.is_file() and p!=marker]
    out=dict(status='complete',signature=sig,job_id=job,options=options,rows=rows,primary=primary,secondary_legacy_forecast=secondary,architecture=architecture,
        files=files,attribution=attribution,deterministic_reconstruction=inverse_rows,scope=config.get('claim_status','prospective'),objective_superiority='H5/H6 deferred',
        dtype=options['dtype'],prediction_storage='float64 numeric scores and all readout candidate predictions/weights; no pickle',
        primary_denominator='shared evaluated risk of frozen fitted training-mean history; infer its uncertainty jointly')
    atomic_json(marker,out);return out
