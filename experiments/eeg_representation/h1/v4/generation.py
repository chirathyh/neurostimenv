"""Explicit v4 native generation through the unchanged, audited H1 replay engine."""
from copy import deepcopy
from pathlib import Path
from ..config import DEFAULTS as LEGACY
from ..dataset import generate as legacy_generate
from ..artifacts import initialize_output,atomic_json
from ..targets import canonical_hash
from .config import contexts


def adapter(c):
    out=deepcopy(LEGACY)
    out.update(name=c['name'],backend=c['backend'],namespace=c['namespace'],
               claim_status='engineering' if c['claim_status']=='engineering' else 'development',
               simulation=deepcopy(c['simulation']),data=deepcopy(c['data']),analytical=deepcopy(c['analytical']))
    for r in out['data']['regimes']:
        if r.get('modulation_depth',0)>0:r['diffusion']=c['optional']['phase_diffusion']
    out['forks'].update(score_futures=c['forks']['score_futures'],reference_futures=0,
        high_reference_futures=c['forks']['reference_futures'],high_score_futures=c['forks']['score_futures'],
        high_contexts=[r['context_id'] for r in contexts(c) if r['reference_futures']])
    out['jobs']=[dict(id='native_support',history_s=c['history_s'],lead_ms=10.,
        duration_ms=c['stop_ms']-c['cutoff_ms']-10.,operator='bin_average_waveform',output_grid_hz=500.,
        input_rate_hz=500.,acquisition_bits=None,snr_db=None)]
    out['v4_driver']=deepcopy(c['optional']) # checksum-bound extension: common events are distinct from shared phase.
    out['cost_gate']['required']=False # v4 verifies the stricter outer execution gate.
    return out


def generate(c,root,resume=False,*,execute=False,timing=False):
    # Direct Python callers receive the same fail-closed gate as the CLI.
    from .planning import require_execution
    require_execution(c,root,execute,timing=timing)
    from .driver import common_event_driver
    root=Path(root);old=adapter(c);directory=root/'generation'
    initialize_output(directory,old,resume)
    atomic_json(directory/'v4_parent.json',{'config_hash':canonical_hash(c),'namespace':c['namespace'],
        'optional':c['optional'],'interpretation':'New v4 generation, stored with backward-readable H1 native schema; no old data relabeling'})
    with common_event_driver(c):result=legacy_generate(old,directory,resume)
    atomic_json(root/'generation_result.json',result)
    return result
