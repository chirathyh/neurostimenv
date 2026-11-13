"""Optional shared *events*, distinct from shared rate phase; default is unchanged."""
from contextlib import contextmanager
from unittest.mock import patch
import hashlib
import numpy as np


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4],'little')


def split_common_events(original,kwargs,*,fraction,history_seed,future_seed,cache=None):
    if fraction==0:return original(**kwargs)
    if not 0<fraction<=1:raise ValueError('Common-event fraction outside[0,1]')
    # Unequal E/I rates MUST be different groups. Shared phase alone shares no events.
    group=(float(kwargs['interval_ms']),bool(kwargs.get('rhythm_enabled',False)),
           float(kwargs.get('modulation_depth',0)),float(kwargs.get('frequency_hz',0)))
    # Keep corresponding common RNG streams paired across compared regimes.
    # Modulation/frequency belongs to the intensity/cache key, never the seed key.
    seed_group=float(kwargs['interval_ms'])
    common=dict(kwargs,interval_ms=kwargs['interval_ms']/fraction,
                history_seed=_seed('common-history',history_seed,seed_group),future_seed=_seed('common-future',future_seed,seed_group) if kwargs.get('future_start_ms') is not None else None)
    key=group
    if cache is not None and key in cache:shared=cache[key]
    else:
        shared=np.asarray(original(**common),dtype=float)
        if cache is not None:cache[key]=shared
    if fraction==1:return shared.copy()
    private=dict(kwargs,interval_ms=kwargs['interval_ms']/(1-fraction),
                 history_seed=_seed('private',kwargs['history_seed']),future_seed=_seed('private',kwargs.get('future_seed')) if kwargs.get('future_start_ms') is not None else None)
    return np.sort(np.r_[shared,original(**private)])


@contextmanager
def common_event_driver(config):
    c=float(config['optional']['common_event_fraction'])
    if c==0:yield;return
    if config['backend']!='neuron':raise ValueError('Common-event sensitivity requires its named NEURON driver; no analytical substitution')
    from ... import simulator
    original_adapter=simulator._construction_adapter
    @contextmanager
    def adapted(utils,spec,total,phase_dt,record):
        original=utils.generate_split_background_spike_train;cache={}
        def events(**kwargs):
            return split_common_events(original,kwargs,fraction=c,history_seed=spec['trial_seed'],
                                       future_seed=spec['future_seed'],cache=cache)
        with patch.object(utils,'generate_split_background_spike_train',events):
            with original_adapter(utils,spec,total,phase_dt,record):yield
    with patch.object(simulator,'_construction_adapter',adapted):yield


def controls(repetitions=1500):
    # Actual event generator, not just algebra: disjoint private streams plus common counts.
    import os
    from .artifacts import ROOT
    with patch.dict(os.environ, {'MAIN_PATH':str(ROOT)}):
        from setup.circuits.ballnstick import utils
    original=utils.generate_split_background_spike_train
    pairs=[];c=.2;expected=25.
    for r in range(repetitions):
        cache={};pair=[]
        for j in (0,1):
            kw=dict(start_ms=0.,stop_ms=1000.,interval_ms=40.,history_seed=_seed(r,j))
            pair.append(len(split_common_events(original,kw,fraction=c,history_seed=r,future_seed=r+100000,cache=cache)))
        pairs.append(pair)
    pairs=np.asarray(pairs);cov=float(np.cov(pairs.T,ddof=1)[0,1]);mean=pairs.mean(0);var=pairs.var(0,ddof=1)
    kw=dict(start_ms=0.,stop_ms=2000.,interval_ms=40.,history_seed=123,future_start_ms=1000.,future_seed=456)
    a=split_common_events(original,kw,fraction=.02,history_seed=1,future_seed=2)
    b=split_common_events(original,dict(kw,future_seed=789),fraction=.02,history_seed=1,future_seed=3)
    prefix=np.array_equal(a[a<1000],b[b<1000])
    return {'status':'complete','passed':bool(np.all(abs(mean-expected)<1.5) and np.all(abs(var-expected)<4) and abs(cov-c*expected)<2.5 and prefix),
            'repetitions':repetitions,'expected_mean_and_variance':expected,'observed_mean':mean.tolist(),'observed_variance':var.tolist(),
            'expected_covariance':c*expected,'observed_covariance':cov,'prefix_future_split_identical':prefix,
            'scope':'Conditional common-event Poisson driver; no assertion of matched neural output rates or spectra'}
