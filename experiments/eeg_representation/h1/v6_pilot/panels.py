"""Matched cohort slices with target-specific training normalization."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..artifacts import atomic_json
from ..targets import canonical_hash
from ..v6.derivation import load_panel,_save_job
from ..v6.models import context_weights,first_targets

AXES={'input_time_ms','target_time_ms','target_frequency_hz','target_component'}

def subset(data,indices):
    n=len(data['context_id'])
    return {k:np.asarray(v)[indices].copy() if k not in AXES and np.asarray(v).ndim and len(v)==n else np.asarray(v).copy() for k,v in data.items()}

def match(panel,cohorts,manifest):
    """Cohorts maps split->ordered context IDs, including independent calibration."""
    lookup={str(cid):(split,i) for split,p in panel.items() for i,cid in enumerate(p['context_id'])}
    out={}
    for split,ids in cohorts.items():
        if len(set(ids))!=len(ids):raise ValueError('Duplicate matched IDs')
        pieces=[]
        for cid in ids:
            if cid not in lookup:raise ValueError('Matched context absent '+cid)
            origin,i=lookup[cid];pieces.append(subset(panel[origin],[i]))
        first=pieces[0];out[split]={k:np.concatenate([p[k] for p in pieces],axis=0) if k not in AXES else first[k] for k in first}
    for a,p in out.items():
        for b,q in out.items():
            if a<b and set(p['structure_id'])&set(q['structure_id']):raise ValueError('Matched circuit split leakage')
    train=out['train'];w=context_weights(train)
    y=train['Y_physical'][np.arange(len(w)),np.argmax(train['future_mask'],axis=1)]
    yo=w@y;ys=float(np.sqrt(w@np.mean((y-yo)**2,axis=1)))
    x=train['X_physical'];xo=float(w@x.mean(1));xs=float(np.sqrt(w@np.mean((x-xo)**2,axis=1)))
    if min(xs,ys)<=0 or not np.isfinite([xs,ys,xo]).all():raise ValueError('Degenerate matched training normalization')
    n=dict(x_offset=xo,x_scale=xs,y_offset=yo.tolist(),y_scale=ys,fit_split='train',
        train_context_ids=list(cohorts['train']),policy='equal circuit/regime first registered training future; shared physical target normalization')
    n['id']=canonical_hash(n)
    for p in out.values():
        p['X']=(p['X_physical']-xo)/xs
        p['Y']=(p['Y_physical']-yo)/ys
        p['reference_Y']=(p['reference_Y_physical']-yo)/ys
    m=deepcopy(manifest);m['original_normalization']=manifest['normalization'];m['normalization']=n
    m['risk_contract']={**manifest['risk_contract'],'population_id':canonical_hash(cohorts),'normalization_id':n['id']}
    m['target_fingerprint']=m['comparison_fingerprint']=canonical_hash(m['risk_contract'])
    m['cohort']='matched complete-state reporting pilot';m['claim_status']='exploratory_reporting_pilot'
    return out,m

def derive_matched(broad_root,matched_root,cohorts,jobs):
    rows=[];target_norms={}
    for job in jobs:
        panel,m=load_panel(broad_root,job);panel,m=match(panel,cohorts,m)
        target=job.split('__')[1];key=(m['normalization']['y_offset'],m['normalization']['y_scale'])
        if target in target_norms and key!=target_norms[target]:raise ValueError('Matched normalization differs across acquisition')
        target_norms[target]=key;rows.append(_save_job(matched_root,job,panel,m))
    result=dict(status='complete',jobs=rows,cohorts=cohorts,source_root=str(Path(broad_root).resolve()),
        population='matched state cohort, separate from broader eight-circuit results')
    atomic_json(Path(matched_root)/'derived/manifest.json',result);return result
