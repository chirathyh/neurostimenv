"""Independent Healthy calibration at the absolute epochs of the 60-s S1 protocol."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis import s1_analysis as s1
from experiments.l23net_analysis.r1_protocol import COHORTS, sha256
from experiments.l23net_analysis.replay_validation import canonical_json_sha256, trace_content_summary

ROOT = Path(__file__).resolve().parents[2]
SEEDS = COHORTS['calibration']
RESERVATION_KSU = 624 * 2 * 2 / 1000
EPOCHS = {'baseline': [8, 28], 'baseline_10s': [18, 28], 'plateau': [29, 49], 'washout': [50, 60]}
PROTOCOL = {
    'version': 1, 'stage': 'reference60', 'seeds': list(SEEDS),
    'condition': 'reference', 'MDD': False, 'DRUG': False,
    'duration_ms': 60000, 'dt_ms': .025, 'mpi_ranks': 624, 'celsius': 34,
    'window_ms': 1000, 'env_seed': 0, 'field_v_per_m': 0,
    'epochs_s': EPOCHS, 'measurement': s0.PROTOCOL,
    'prefix_rule': 'Exact R1 0--28-s EEG, dipole, spikes, seed and construction identity.',
    'screening_rule': 'Keep the existing S0 8--28-s eligibility target unchanged.',
    'target_rule': 'Fit 29--49-s and 50--60-s targets using all 16 Healthy seeds only.',
    'interpretation': 'Reference calibration, not stimulation efficacy or clinical validation.',
}
# A manifest round trip must not turn tuple/list differences into a false failure.
PROTOCOL = json.loads(json.dumps(PROTOCOL))
NEW_CODE = [
    'configs/analysis/l23net_reference60.yaml',
    'experiments/l23net_analysis/reference60_protocol.py',
    'experiments/l23net_analysis/run_l23net_reference60.py',
    'experiments/l23net_analysis/analyze_l23net_reference60.py',
    'experiments/l23net_analysis/nci/submit_l23net_reference60.py',
    'experiments/l23net_analysis/nci/run_l23net_reference60_worker.sh',
]


def code_hashes():
    return {**s1.code_hashes(), **{p: sha256(ROOT/p) for p in NEW_CODE}}


def signed(value):
    if 'sha256' in value:
        raise ValueError('Do not sign an already signed object')
    return {**value, 'sha256': canonical_json_sha256(value)}


def load_signed(path):
    value = json.loads(Path(path).read_text())
    payload = dict(value)
    if payload.pop('sha256') != canonical_json_sha256(payload):
        raise ValueError(f'Changed artifact: {path}')
    return value


def prefix_hashes(path, duration_ms=28000, dt_ms=.025, window_ms=1000):
    """Bounded-memory, chunk-layout-independent hashes of exact native samples."""
    count, chunk = round(duration_ms/dt_ms), round(window_ms/dt_ms)
    result = {}
    with h5py.File(path) as h:
        for name in ('sample_time_ms', 'eeg_v', 'dipole_nA_um'):
            data = h[name]
            if data.shape[-1] < count:
                raise ValueError('Incomplete prefix: '+name)
            digest = hashlib.sha256()
            digest.update(json.dumps([str(data.dtype), list(data.shape[:-1])+[count]]).encode())
            for first in range(0, count, chunk):
                digest.update(np.ascontiguousarray(data[..., first:min(first+chunk, count)], dtype='<f8').tobytes())
            result[name] = digest.hexdigest()
    return result


def prefix_record(report, trace):
    return {
        'seed_manifest_sha256': canonical_json_sha256(report['seed_manifest']),
        'structure_sha256': report['structure']['global_sha256'],
        'build_sha256': report['build_audit']['invariant_sha256'],
        'spikes_sha256': canonical_json_sha256([w['spikes'] for w in report['windows'] if w['stop_ms'] <= 28000]),
        'trace_prefix': prefix_hashes(trace),
    }


def prepare_sources(core, gate):
    """Validate R1 against the S0-frozen summary before signing portable prefixes."""
    core = Path(core)
    expected = [v for k, v in gate['source_hashes'].items() if 'l23net_r1_core_' in k]
    if len(expected) != 1 or sha256(core/'r1_summary.json') != expected[0]:
        raise ValueError('R1 core summary does not match the source frozen in S0')
    summary = json.loads((core/'r1_summary.json').read_text())
    if not summary['technical_passed'] or summary['errors']:
        raise ValueError('R1 core was not technically complete')
    rows = [r for r in summary['rows'] if r['cohort'] == 'calibration']
    if sorted(r['seed'] for r in rows) != list(SEEDS):
        raise ValueError('R1 requires exactly the 16 frozen calibration seeds')
    sources = []
    for row in sorted(rows, key=lambda r: r['seed']):
        paths = list(core.glob(f"calibration_*/seed_{row['seed']}/reference/l23net_r1_run.json"))
        if len(paths) != 1:
            raise ValueError('Missing or ambiguous R1 source for '+str(row['seed']))
        report_path = paths[0]
        report = json.loads(report_path.read_text())
        trace = report_path.with_name('l23net_r1_trace.h5')
        if sha256(report_path) != row['report_sha256'] or report['status'] != 'passed':
            raise ValueError('Changed/failed R1 source report')
        if trace_content_summary(trace)['content_sha256'] != row['trace_content_sha256']:
            raise ValueError('Changed R1 source trace')
        sources.append({'seed': row['seed'], 'report_sha256': row['report_sha256'],
                        'trace_content_sha256': row['trace_content_sha256'],
                        'prefix': prefix_record(report, trace)})
    return signed({'core_summary_sha256': expected[0], 's0_sha256': gate['sha256'], 'sources': sources})


def find_core(gate):
    names = [Path(k).parent.name for k in gate['source_hashes'] if 'l23net_r1_core_' in k]
    if len(names) != 1:
        raise ValueError('Ambiguous S0 core source')
    for base in (ROOT/'results', ROOT/'results/r1_results'):
        candidate = base/names[0]
        if (candidate/'r1_summary.json').is_file():
            return candidate
    raise FileNotFoundError('Original R1 core not found; pass --core /full/path/to/core_suite')


def check_prefix(report, trace, source):
    actual = prefix_record(report, trace)
    changed = [k for k, v in source['prefix'].items() if actual.get(k) != v]
    if changed:
        raise ValueError('R1 prefix mismatch: '+', '.join(changed))


def jobs():
    return [{'name': f'reference_{s}', 'run': {'seed': s, 'condition': 'reference', 'arm': 'sham'}} for s in SEEDS]
