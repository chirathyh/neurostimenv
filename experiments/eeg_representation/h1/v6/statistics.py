"""Prespecified v6 circuit-level inference, float64 throughout.

Primary A is one intersection-union family at alpha=.025. Primary B uses a
97.5% two-sided paired-circuit influence interval. Inference is conditional on
the frozen training/model-selection policy; optimizer seeds are fitting repeats.
"""
from __future__ import annotations
import itertools
import re
import numpy as np
from scipy.stats import t, binomtest

FAMILY_ALPHA = .025
FAMILY_A_MARGINS = {'task_bits': .20, 'reconstruction_nmse': .02}
FAMILY_B_MARGIN_PP = 1.


def _finite(values, name='values'):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError(f'{name} must be nonempty and finite')
    return a


def stable_mean(values, axis=None):
    a = _finite(values)
    scale = np.max(np.abs(a), axis=axis, keepdims=True)
    safe = np.where(scale > 0, scale, 1.)
    value = np.mean(a / safe, axis=axis, keepdims=True) * scale
    return np.squeeze(value, axis=axis) if axis is not None else float(value.squeeze())


def stable_sd(values, ddof=1):
    a = _finite(values).reshape(-1)
    if len(a) <= ddof:
        raise ValueError('More observations than ddof required')
    offset = stable_mean(a)
    with np.errstate(over='raise', invalid='raise'):
        centered = a - offset
        scale = np.max(np.abs(centered))
        if scale == 0: return 0.
        return float(scale * np.sqrt(np.sum((centered / scale) ** 2) / (len(a)-ddof)))


def average_fitting_repeats(values, *, neural=False, seed_axis=0, expected_seeds=5):
    """Average LOSSES/effects over fitting seeds, never average predictions here."""
    a = _finite(values)
    if neural:
        if a.ndim < 2 or a.shape[seed_axis] != expected_seeds:
            raise ValueError(f'Neural primary requires exactly {expected_seeds} fitting repeats')
        a = stable_mean(a, axis=seed_axis)
    if a.ndim != 1:
        raise ValueError('Expected one value per circuit after fitting-repeat averaging')
    if len(a) < 2: raise ValueError('At least two independent circuits required')
    return np.asarray(a, np.float64)


def validate_split_ids(split_ids, *, inspected_test_ids=()):
    """Validate circuit disjointness, not merely disjoint row/context IDs."""
    lists={str(k):list(map(str,v))for k,v in split_ids.items()}
    if any(len(v)!=len(set(v))for v in lists.values()):raise ValueError('Duplicate circuit IDs within a split')
    groups = {k:set(v)for k,v in lists.items()}
    for a, b in itertools.combinations(groups, 2):
        common = groups[a] & groups[b]
        if common: raise ValueError(f'Circuit leakage between {a} and {b}: {sorted(common)}')
    old = set(map(str, inspected_test_ids))
    for key in ('test', 'evaluation', 'calibration'):
        if groups.get(key, set()) & old:
            raise ValueError(f'Previously inspected test circuits cannot enter fresh {key}')
    return {'status': 'complete', 'n_circuits': {k: len(v) for k,v in groups.items()},
            'pairwise_disjoint': True, 'previous_test_circuits_excluded': True}


def aggregate_circuit_losses(losses, circuit_ids, condition_ids, prefix_ids,
                             *, expected_conditions=None, expected_prefixes=None):
    """Equal circuits, then conditions, then prefixes, then observed futures.

    `losses` is [context] or [context,future]; each context must identify exactly
    one circuit/condition/prefix. Caller averages exactly five fitting-repeat
    losses beforehand, or invokes this function separately for each seed.
    Missing conditions/prefixes fail when their frozen lists/counts are supplied.
    """
    loss = _finite(losses, 'losses')
    if loss.ndim == 2: loss = stable_mean(loss, axis=1)
    if loss.ndim != 1: raise ValueError('Expected [context] or [context,future] losses')
    sid = np.asarray(circuit_ids).astype(str)
    cond = np.asarray(condition_ids).astype(str)
    pre = np.asarray(prefix_ids).astype(str)
    if not (len(sid) == len(cond) == len(pre) == len(loss)):
        raise ValueError('Context metadata length mismatch')
    keys = list(zip(sid, cond, pre))
    if len(set(keys)) != len(keys): raise ValueError('Duplicate circuit/condition/prefix contexts')
    circuits = sorted(set(sid)); result=[]
    for s in circuits:
        conditions = sorted(set(cond[sid == s]))
        expected = conditions if expected_conditions is None else sorted(map(str, expected_conditions))
        if conditions != expected: raise ValueError(f'Missing/unexpected conditions for circuit {s}')
        values=[]
        for c in conditions:
            take=(sid==s)&(cond==c); prefixes=sorted(set(pre[take]))
            if isinstance(expected_prefixes, int) and len(prefixes) != expected_prefixes:
                raise ValueError(f'Wrong prefix count for {s}/{c}')
            if expected_prefixes is not None and not isinstance(expected_prefixes,int):
                if prefixes != sorted(map(str,expected_prefixes)):
                    raise ValueError(f'Wrong prefix set for {s}/{c}')
            values.append(stable_mean(loss[take]))
        result.append(stable_mean(values))
    return {'circuit_ids': circuits, 'values': np.asarray(result,np.float64),
            'weighting': 'equal circuit, condition, prefix; equal observed future within context'}


def select_primary_futures(panel, registered_ids=None, *, count=4):
    """Return a panel containing ONLY first four registered scoring futures.

    Explicit registry order wins. Without it, IDs must end in score-N and the
    required numeric ordinals are exactly 0..3; neither file order nor availability
    can replace a missing primary future. Reference pools remain separate.
    """
    y=np.asarray(panel['Y'],np.float64); ids=np.asarray(panel['future_id']).astype(str)
    mask=np.asarray(panel.get('future_mask',np.ones(ids.shape,bool)),bool)
    if y.ndim!=3 or ids.shape!=y.shape[:2] or mask.shape!=ids.shape:
        raise ValueError('Primary future shapes mismatch')
    if not len(y) or not y.shape[2] or not np.isfinite(y[mask]).all():
        raise ValueError('Active scoring futures must be nonempty and finite')
    if count!=4: raise ValueError('Frozen v6 primary uses exactly four scoring futures')
    registry=None if registered_ids is None else np.asarray(registered_ids).astype(str)
    if registry is not None and (registry.ndim!=2 or registry.shape[0]!=len(y) or registry.shape[1]<count):
        raise ValueError('Registry must have >=four scoring IDs per context')
    indices=[]
    for i,row in enumerate(ids):
        available=row[mask[i]]
        if len(set(available))!=len(available):raise ValueError('Duplicate scoring future IDs')
        if registry is not None: wanted=list(registry[i,:count])
        else:
            ordinals={}
            for j in np.flatnonzero(mask[i]):
                match=re.search(r'(?:^|-)score-(\d+)$',row[j])
                if not match: raise ValueError('Registered score-ordinal IDs or explicit registry required')
                ordinal=int(match.group(1))
                if ordinal in ordinals:raise ValueError('Duplicate scoring ordinal')
                ordinals[ordinal]=row[j]
            if any(j not in ordinals for j in range(count)):raise ValueError('Missing primary scoring future 0..3')
            wanted=[ordinals[j]for j in range(count)]
        if len(set(wanted))!=count or not set(wanted)<=set(available):
            raise ValueError('Required primary futures are missing or duplicated')
        indices.append([int(np.flatnonzero((row==w)&mask[i])[0])for w in wanted])
    index=np.asarray(indices);out=dict(panel)
    for key in ('Y','Y_physical','future_id','future_mask'):
        if key in panel:out[key]=np.asarray(panel[key])[np.arange(len(y))[:,None],index]
    out['primary_future_policy']='first four registered scoring futures; H2 extras excluded'
    out['selected_future_indices']=index
    return out


def _t_result(estimate, influence, *, confidence=.975, one_sided=False, margin=0.):
    z=_finite(influence).reshape(-1); n=len(z)
    if n<2:raise ValueError('Need >=two independent circuits')
    if not 0<confidence<1:raise ValueError('Invalid confidence')
    sd=stable_sd(z);se=sd/np.sqrt(n);q=float(t.ppf(confidence if one_sided else (1+confidence)/2,n-1))
    delta=float(estimate-margin)
    statistic=delta/se if se>0 else None
    p=float(t.sf(statistic,n-1))if se>0 else (0. if delta>0 else 1.)
    result={'estimate':float(estimate),'n_circuits':n,'df':n-1,'circuit_sd':sd,'standard_error':float(se),
            'confidence':confidence,'one_sided':one_sided,'critical_t':q,'lower':float(estimate-q*se),
            'upper':None if one_sided else float(estimate+q*se),'margin':float(margin),
            'one_sided_p_greater_than_margin':p,'t_statistic':statistic,
            'degenerate_observed_variance':bool(se==0),
            'method':'paired-circuit t / first-order influence; exact t only under normal circuit effects',
            'conditioning':'frozen training set and model-selection policy; no independent training-dataset variation'}
    return result


def ratio_of_means_interval(numerator, denominator, *, confidence=.975, one_sided=False,
                            margin=0., multiplier=1.):
    """Delta-method interval for multiplier * mean(a)/mean(b), with paired b."""
    a=_finite(numerator).reshape(-1);b=_finite(denominator).reshape(-1)
    if a.shape!=b.shape or len(a)<2 or np.any(b<0):raise ValueError('Paired nonnegative baseline risks required')
    bm=stable_mean(b)
    if bm<=0:raise ValueError('Baseline mean must be positive')
    ratio=stable_mean(a)/bm
    influence=float(multiplier)*(a-ratio*b)/bm
    out=_t_result(float(multiplier)*ratio,influence,confidence=confidence,one_sided=one_sided,margin=margin)
    out.update(numerator_mean=stable_mean(a),denominator_mean=bm,influence_values=influence.tolist())
    return out


def _sign_sensitivity(values):
    a=_finite(values).reshape(-1);nonzero=a[a!=0];n=len(nonzero);positive=int(np.sum(nonzero>0))
    return {'n_positive':positive,'n_negative':int(np.sum(nonzero<0)),'n_ties':len(a)-n,
            'two_sided_binomial_sign_p':float(binomtest(positive,n,.5).pvalue)if n else 1.,
            'scope':'direction probability / median under iid circuits; not an exact mean-effect test'}


def family_a(task_differences, reconstruction_differences, *, reconstruction_baseline_risks=None,
             neural=False, seed_axis=0, task_margin=.20, reconstruction_margin=.02, alpha=.025,
             bootstrap_draws=10000, bootstrap_seed=601):
    """H3 PCA minus feature CE, feature minus PCA reconstruction.

    With baseline risks supplied reconstruction differences are RAW paired MSE
    differences, divided by the shared mean baseline; its influence accounts for
    estimation of that denominator. Otherwise caller certifies a fixed denominator
    and supplies normalized per-circuit differences. Five neural repeats average
    within circuit before any inference.
    """
    if alpha!=FAMILY_ALPHA:raise ValueError('Primary family A alpha frozen at .025')
    if task_margin!=.20 or reconstruction_margin!=.02:raise ValueError('Primary practical margins are frozen')
    task=average_fitting_repeats(task_differences,neural=neural,seed_axis=seed_axis)
    rec=average_fitting_repeats(reconstruction_differences,neural=neural,seed_axis=seed_axis)
    if task.shape!=rec.shape:raise ValueError('Task/reconstruction circuits must align')
    tr=_t_result(stable_mean(task),task,confidence=1-alpha,one_sided=True,margin=task_margin)
    if reconstruction_baseline_risks is None:
        rr=_t_result(stable_mean(rec),rec,confidence=1-alpha,one_sided=True,margin=reconstruction_margin)
        scope='per-circuit NMSE differences using externally fixed shared denominator'
        normalized=rec
    else:
        base=_finite(reconstruction_baseline_risks).reshape(-1)
        if base.shape!=rec.shape:raise ValueError('Reconstruction denominator circuits must align')
        rr=ratio_of_means_interval(rec,base,confidence=1-alpha,one_sided=True,margin=reconstruction_margin)
        scope='ratio of equal-circuit raw MSE difference to same fitted-mean baseline risk; denominator uncertainty included'
        normalized=rec/stable_mean(base)
    p=max(tr['one_sided_p_greater_than_margin'],rr['one_sided_p_greater_than_margin'])
    passed=bool(p<=alpha and tr['lower']>task_margin and rr['lower']>reconstruction_margin)
    return {'family':'A','hypothesis':'BOTH E[D_task]>.20 bits AND reconstruction ratio>.02 NMSE',
            'alpha':alpha,'familywise_alpha_accounting':'A .025 + B .025 <= .05; conjunction consumes one family',
            'iut_p':p,'passed':passed,'decision':'practical disagreement supported'if passed else 'not established',
            'task':tr,'reconstruction':rr,'reconstruction_denominator_scope':scope,
            'per_circuit_task':task.tolist(),'per_circuit_reconstruction':normalized.tolist(),
            'task_direction_sensitivity':_sign_sensitivity(task-task_margin),
            'reconstruction_direction_sensitivity':_sign_sensitivity(normalized-reconstruction_margin),
            'fitting_repeats_averaged':5 if neural else 1,'n_circuits':len(task),
            'other_contrasts':'descriptive unless separately frozen and multiplicity-adjusted',
            'circuit_bootstrap_sensitivity':bootstrap_family_a_sensitivity(task,rec,
                reconstruction_baseline_risks=reconstruction_baseline_risks,draws=bootstrap_draws,seed=bootstrap_seed)}


def family_b(model_risks, baseline_risks, *, neural=False, seed_axis=0, margin_pp=1., confidence=.975,
             bootstrap_draws=10000, bootstrap_seed=601):
    """H1 G=100*(1-mean(model risk)/mean(mean-baseline risk)); not mean ratios."""
    if confidence!=.975 or margin_pp!=1.:raise ValueError('Primary B confidence and equivalence margin frozen')
    a=average_fitting_repeats(model_risks,neural=neural,seed_axis=seed_axis)
    b=_finite(baseline_risks).reshape(-1)
    if a.shape!=b.shape or np.any(a<0):raise ValueError('Aligned nonnegative risks required')
    out=ratio_of_means_interval(b-a,b,confidence=confidence,multiplier=100.)
    lo,hi=out['lower'],out['upper']
    decision=('meaningful improvement'if lo>margin_pp else 'meaningful worsening'if hi<-margin_pp else
              'practical equivalence'if lo>=-margin_pp and hi<=margin_pp else 'inconclusive')
    out.update(family='B',alpha=.025,decision=decision,equivalence_margin_pp=margin_pp,
               model_risk=stable_mean(a),baseline_risk=stable_mean(b),model_circuit_risks=a.tolist(),
               baseline_circuit_risks=b.tolist(),fitting_repeats_averaged=5 if neural else 1,
               estimand='100*(1-ratio of equal-circuit averaged risks)',
               interpretation='equivalence applies only to this frozen predictor/policy; nonsignificance is not equivalence',
               zero_direction_sensitivity=_sign_sensitivity(b-a),
               circuit_bootstrap_sensitivity=bootstrap_ratio_sensitivity(a,b,draws=bootstrap_draws,seed=bootstrap_seed))
    return out


def paired_ratio_difference(model_risks_a, baseline_risks_a, model_risks_b, baseline_risks_b,
                           *, confidence=.975, margin_pp=.1):
    """G_A - G_B on SAME circuits; shared-circuit influence preserves pairing."""
    a=_finite(model_risks_a).reshape(-1);aa=_finite(baseline_risks_a).reshape(-1)
    b=_finite(model_risks_b).reshape(-1);bb=_finite(baseline_risks_b).reshape(-1)
    if not(a.shape==aa.shape==b.shape==bb.shape)or np.any(np.concatenate([a,aa,b,bb])<0):
        raise ValueError('Paired nonnegative risks on identical circuits required')
    if stable_mean(aa)<=0 or stable_mean(bb)<=0:
        raise ValueError('Both paired baseline means must be positive')
    ra=stable_mean(a)/stable_mean(aa);rb=stable_mean(b)/stable_mean(bb)
    ia=-100*(a-ra*aa)/stable_mean(aa);ib=-100*(b-rb*bb)/stable_mean(bb)
    out=_t_result(100*(rb-ra),ia-ib,confidence=confidence)
    out.update(equivalence_margin_pp=margin_pp,within_tolerance=out['lower']>=-margin_pp and out['upper']<=margin_pp,
        interpretation='paired circuit interval; caller must separately quantify finite-future uncertainty and aligned forcings')
    return out


def bootstrap_ratio_sensitivity(model_risks,baseline_risks,*,draws=10000,seed=601):
    a=_finite(model_risks).reshape(-1);b=_finite(baseline_risks).reshape(-1)
    if a.shape!=b.shape or np.any(a<0)or np.any(b<0)or stable_mean(b)<=0:raise ValueError('Invalid paired risks')
    if draws<1:return {'status':'unrun','role':'optional circuit-bootstrap sensitivity disabled explicitly'}
    ix=np.random.default_rng(seed).integers(0,len(a),(int(draws),len(a)))
    den=b[ix].mean(1)
    if np.any(den<=0):raise ValueError('Zero bootstrap baseline; ratio sensitivity unavailable')
    values=100*(1-a[ix].mean(1)/den)
    return {'confidence':.975,'interval':np.quantile(values,[.0125,.9875]).tolist(),'draws':int(draws),
            'seed':seed,'role':'sensitivity only; draws are not additional circuits'}


def bootstrap_family_a_sensitivity(task_differences,reconstruction_differences,*,
                                   reconstruction_baseline_risks=None,draws=10000,seed=601):
    """Paired circuit percentile intervals; preserve paired estimated denominator.

    These sensitivities never replace the prespecified t/influence IUT. The same
    sampled circuit indices are used for the task and reconstruction components.
    """
    a=_finite(task_differences).reshape(-1);r=_finite(reconstruction_differences).reshape(-1)
    if a.shape!=r.shape or len(a)<2:raise ValueError('Paired circuit effects required')
    if draws<1:return {'status':'unrun','role':'optional circuit-bootstrap sensitivity disabled explicitly'}
    ix=np.random.default_rng(seed).integers(0,len(a),(int(draws),len(a)))
    task=stable_mean(a[ix],axis=1);rec=stable_mean(r[ix],axis=1)
    if reconstruction_baseline_risks is not None:
        b=_finite(reconstruction_baseline_risks).reshape(-1)
        if b.shape!=a.shape or np.any(b<0):raise ValueError('Paired baseline risks required')
        den=stable_mean(b[ix],axis=1)
        if np.any(den<=0):raise ValueError('Zero bootstrap denominator; reconstruction ratio sensitivity unavailable')
        rec=rec/den
    return {'status':'complete','confidence':.975,'one_sided':True,
            'task_lower':float(np.quantile(task,.025)),'reconstruction_lower':float(np.quantile(rec,.025)),
            'task_two_sided95':np.quantile(task,[.025,.975]).tolist(),
            'reconstruction_two_sided95':np.quantile(rec,[.025,.975]).tolist(),
            'draws':int(draws),'seed':seed,'n_circuits':len(a),
            'role':'sensitivity only; paired circuit draws are not new evidence or additional independent circuits'}
