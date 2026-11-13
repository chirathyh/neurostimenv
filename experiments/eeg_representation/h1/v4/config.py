"""Validated, bounded manuscript-v4 design; old pilot/COSYNE configs stay untouched."""
from copy import deepcopy
from pathlib import Path
import json,re
import yaml
from . import SCHEMA_VERSION
from ..targets import canonical_hash
from ..config import REGIMES
from ..artifacts import ROOT

DEFAULTS={
 'schema_version':SCHEMA_VERSION,'name':'evaluation_v4','claim_status':'prospective_bounded','backend':'neuron',
 'namespace':'eegrepr-evaluation-v4-prospective-20261113','history_s':4.,'cutoff_ms':6000.,'stop_ms':7010.,
 'simulation':{'native_dt_ms':.0625,'burn_in_s':2.,'celsius':6.3,'population_e':32,'population_i':8,'phase_grid_ms':1.,'external_synapses_per_cell':64},
 'data':{'structures':{'train':list(range(36000,36008)),'development':list(range(36008,36012)),'evaluation':list(range(36012,36020))},
         'prefixes_per_regime':{'train':8,'development':4,'evaluation':1},'regimes':REGIMES,'workers':1},
 'forks':{'score_futures':8,'reference_futures':32,'reference_circuits':[36012,36013,36014,36015],'reference_regimes':['R0','Ralpha']},
 'acquisition_ids':['A0','A1','A2'],'selected_job_ids':['A2__T3_l10_d1000','A0__T0_l10_d128','A1__T0_l2_d32'],
 'models':{'ridge_lags':[16,64],'ridge_alphas':[.01,.1,1.,10.,100.,1000.,10000.],
           'neural_hidden':[32],'neural_seeds':[0,1,2],'neural_losses':['mse','gaussian'],'max_steps':300,'patience':40,
           'learning_rate':.001,'weight_decay':.01,'eval_every':10,'max_parameters':50000,'threads':1},
 'replication':{'m_values':[1,2,4,8],'repetitions':32,'bootstrap_samples':256,'seed':8123},
 'retention':{'job_id':'A0__T0_l10_d128','banks':['feature18','pca','untrained','masked_mse','forecast'],
              'dimensions':[8,32],'code_bits':[32,128],'label_budgets':[8,'all'],
              'regularization':[.01,1.,100.],'range_rules':['quantile_005_995','robust_iqr'],
              'optimization_seeds':[0,1,2],'representation_steps':200,'tasks':['waveform_reconstruction','future_mean','rhythm','carrier','log_power','phase_sin','phase_cos']},
 'geometry':{'nonlinear_id':False,'max_state_coordinates':None,'seed':9131},
 'protocol':{'primary_job_id':'A2__T3_l10_d1000','matched_code_contrast':{'banks':['feature18','forecast'],'dimension':8,'bits':32,'readout':'linear','label_budget':'all','task':'carrier','range_rule':'quantile_005_995','pooling':'native pooled','job_id':'A0__T0_l10_d128'},
             'measurement_qualification_required':True,'neural_path_control_required':True,'practical_effect':None,
             'claim':'bounded prospective evaluation; not powered confirmation; H5/H6 deferred'},
 'optional':{'common_event_fraction':0.,'phase_diffusion':0.,'instrumentation':False},
 'budget':{'provisional_trial_seconds':88.71645,'historical_duration_ms':6138.,'provisional_native_bytes':4750000000,'require_measured_gate':True},
 'analytical':{'kind':'oscillator','amplitude_v':1e-6,'damping_per_s':3.},
 'target_specs':None,
}


def merge(base,override):
    result=deepcopy(base)
    for k,v in override.items():
        if k not in base:raise ValueError('Unknown v4 configuration key '+k)
        result[k]=merge(base[k],v) if isinstance(v,dict) and isinstance(base[k],dict) and k not in ('models','retention','geometry','replication') else deepcopy(v)
        if k in ('models','retention','geometry','replication') and isinstance(v,dict):result[k]={**deepcopy(base[k]),**deepcopy(v)}
    return result


def resolve(raw):
    c=merge(DEFAULTS,raw or {})
    if c['schema_version']!=SCHEMA_VERSION:raise ValueError('Unsupported v4 schema')
    if c['backend'] not in ('neuron','analytical'):raise ValueError('Backend must be explicit neuron or analytical')
    if c['claim_status'] not in ('engineering','exploratory_existing','prospective_bounded','timing_only'):raise ValueError('Unknown claim status')
    s=c['simulation']
    if s['native_dt_ms']!=.0625 or s['celsius']!=6.3:raise ValueError('Canonical toy integration and temperature are frozen')
    if c['backend']=='analytical' and c['claim_status']!='engineering':raise ValueError('Analytical data are engineering controls only, never NEURON fallback')
    if c['claim_status'] not in ('engineering',) and (s['population_e'],s['population_i'],s['external_synapses_per_cell'])!=(32,8,64):raise ValueError('Primary/timing network must be full32E/8I/64contacts')
    if c['cutoff_ms']!=1000*(s['burn_in_s']+c['history_s']):raise ValueError('Cutoff/history/burn-in conflict')
    if c['stop_ms']<c['cutoff_ms']+1010:raise ValueError('v4 new episodes must support full future-only1sT3 block')
    if not 0<c['history_s']<=4 or c['data']['workers']!=1:raise ValueError('Current runner supports explicit serial trials and up to4s history')
    if set(c['acquisition_ids'])-{'A0','A1','A2'}:raise ValueError('Unknown acquisition')
    ids=[i for v in c['data']['structures'].values() for i in v]
    if len(ids)!=len(set(ids)) or any(type(i)!=int or not 1<=i<=429495 for i in ids):raise ValueError('Disjoint valid circuit IDs required')
    if c['claim_status']=='prospective_bounded' and any(i in set(range(32000,32008))|set(range(35000,35400)) for i in ids):raise ValueError('Inspected/previouslyregisteredcircuits cannot be prospective')
    if c['forks']['score_futures']<2:raise ValueError('Evaluation state variance needsM>=2')
    if c['forks']['reference_futures'] not in (0,) and c['forks']['reference_futures']<2:raise ValueError('Finite independent reference needsL>=2')
    if set(c['forks']['reference_circuits'])-set(c['data']['structures']['evaluation']):raise ValueError('Reference circuits not in evaluation')
    if set(c['forks']['reference_regimes'])-set(r['name'] for r in c['data']['regimes']):raise ValueError('Reference regimes absent')
    if not 0<=c['optional']['common_event_fraction']<=1 or c['optional']['phase_diffusion']<0:raise ValueError('Invalid optional common events/phase diffusion')
    if c['optional']['instrumentation']:raise ValueError('Layer instrumentation is not part of the default campaign; explicit separate implementation/budget required')
    if c['models']['ridge_lags']!=[16,64] or c['models']['ridge_alphas']!=[.01,.1,1.,10.,100.,1000.,10000.]:raise ValueError('The declared complete ridge grid must not be truncated')
    if c['claim_status']=='prospective_bounded' and (len(set(c['models']['neural_seeds']))<3 or len(set(c['retention']['optimization_seeds']))<3):raise ValueError('Prospective fitting diagnostics require at least3 optimizer repeats')
    if c['models'].get('max_parameters',50000)>50000:raise ValueError('Small neural qualification budget is at most50k')
    if not re.fullmatch('[a-zA-Z0-9_-]+',c['name']):raise ValueError('Unsafe campaign name')
    return c


def load_config(path):return resolve(yaml.safe_load(Path(path).read_text()))


def contexts(c):
    out=[];refs=set(c['forks']['reference_circuits']);regrefs=set(c['forks']['reference_regimes'])
    for split,ids in c['data']['structures'].items():
        for sid in ids:
            for regime in c['data']['regimes']:
                for prefix in range(c['data']['prefixes_per_regime'][split]):
                    r=deepcopy(regime)
                    if r.get('modulation_depth',0)>0:r['diffusion']=c['optional']['phase_diffusion']
                    out.append({'context_id':f's{sid}-p{prefix:04d}-{r["name"]}','structure_id':sid,'prefix':prefix,'split':split,'regime':r,
                      'score_futures':c['forks']['score_futures'] if split=='evaluation' else 1,
                      'reference_futures':c['forks']['reference_futures'] if split=='evaluation' and sid in refs and r['name'] in regrefs and prefix==0 else 0})
    return out


def counts(c):
    rows=contexts(c)
    return {'contexts':len(rows),'contexts_by_split':{s:sum(r['split']==s for r in rows) for s in c['data']['structures']},
            'score_by_split':{s:sum(r['score_futures'] for r in rows if r['split']==s) for s in c['data']['structures']},
            'reference_contexts':sum(r['reference_futures']>0 for r in rows),'reference_episodes':sum(r['reference_futures'] for r in rows),
            'total_episodes':sum(r['score_futures']+r['reference_futures'] for r in rows)}
