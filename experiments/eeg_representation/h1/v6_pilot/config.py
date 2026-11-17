"""Independent, fixed pilot allocation and bounded learning policy."""
from copy import deepcopy
from pathlib import Path
import json
import yaml
from ..v6.config import load_config as load_v6, required_job_ids
from ..artifacts import ROOT

DEFAULTS = dict(schema_version='eegrepr-v6-reporting-pilot-1',
    name='eight-worker-reporting-pilot-v1', purpose='exploratory_reporting_pilot',
    workers=8, threads_per_worker=1, max_new_invocations=160,
    minimum_free_bytes=20000000000, minimum_available_ram_bytes=8000000000,
    source_archive_sha256='6818befa7ca70df3a0272bf69328f9e2d2316817493e49a13bac6e4b74a5817f',
    ridge_alphas=[.1,10.,1000.], state_candidate_alpha=10.,
    neural_seed=0, neural_max_steps=500, neural_patience=100, neural_eval_every=25,
    neural_jobs=['A1__T0_l0_d128','A1__T0_l10_d128'],
    histories=[64,2000], forecast_updates=200, forecast_seed=0,
    dtype='float32', statistics_dtype='float64', h2_m=[1,2,4,8,16,32], h2_l=32,
    score_futures=8, descriptive_confidence=.95, h2_bootstrap=100,
    allow_registered_fallback=True, fallback_circuit_id=37700,
    confirmation='deferred; separate decision after reporting pilot review')

def resolve(raw=None):
    raw=raw or {};unknown=set(raw)-set(DEFAULTS)
    if unknown:raise ValueError('Unknown pilot settings: '+str(sorted(unknown)))
    c={**deepcopy(DEFAULTS),**deepcopy(raw)}
    for key in ('schema_version','purpose','workers','threads_per_worker','max_new_invocations',
                'ridge_alphas','state_candidate_alpha','neural_seed','neural_jobs','histories',
                'forecast_seed','statistics_dtype','h2_m','h2_l','score_futures','fallback_circuit_id'):
        if c[key]!=DEFAULTS[key]:raise ValueError('Fixed reporting pilot field changed: '+key)
    if not 1<=c['neural_max_steps']<=500 or not 1<=c['forecast_updates']<=200:
        raise ValueError('Pilot optimizer budget exceeded')
    if c['dtype'] not in ('float32','float64'):raise ValueError('No AMP/TF32 pilot training')
    if c['minimum_free_bytes']<20000000000 or c['minimum_available_ram_bytes']<8000000000:
        raise ValueError('Do not lower hardware safety floors')
    return c

def load(path):
    text=Path(path).read_text();return resolve(json.loads(text) if text.lstrip().startswith('{') else yaml.safe_load(text))

def analysis_config():
    c=load_v6(ROOT/'configs/eeg_representation/v6_extended/existing.yaml')
    c['models']={'neural_enabled':False,'run_controls':False}
    return c

def jobs():return required_job_ids(analysis_config())
