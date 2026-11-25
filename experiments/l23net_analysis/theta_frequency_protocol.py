"""Prespecified L23Net comparison of theta modulation at 14 and 10 Hz."""

import copy
import json
from pathlib import Path

from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis import repeatability_protocol as repeat
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.r1_protocol import sha256

ROOT = dose.ROOT
PROJECT = dose.PROJECT
HISTORIES = (1, 2)
SEEDS = tuple(range(8711, 8719))
FREQUENCIES = {'sham': 0., 'fixed_low_beta': s0.FIXED['low_beta'], 'fixed_alpha': s0.FIXED['alpha']}
ARMS = tuple(FREQUENCIES)
AMPLITUDE = .2
SUMMARY = 'theta_frequency_summary.json'
STATUS = 'theta_frequency_status.json'
FIRST_HISTORY = 'history1_summary.json'
PROTOCOL = {
    'name': 'l23net_theta_frequency', 'version': 1, 'project': PROJECT,
    'seeds': list(SEEDS), 'histories': list(HISTORIES), 'resources': dose.RESOURCES,
    'arm_frequency_hz': FREQUENCIES, 'active_amplitude_v_per_m': AMPLITUDE,
    'direction': dose.DIRECTION, 'phase_at_onset_rad': 0., 'dc_v_per_m': 0.,
    'timing_ms': list(dose.TIMING_MS), 'epochs_s': dose.EPOCHS_S,
    'noise': 'Random123(history_seed, gid, OU site). history_seed = 3470000 + 10*seed + history.',
    'wave1_lead_sha256': 'a9d9b09f53a5249f46f1ffab97aa73338e0e31a89a511cb3a0c6dc47d7e6169c',
    'reference60_target_sha256': dose.PROTOCOL['reference60_target_sha256'],
    'inference': 'Average both histories within structure, with screened-out histories retained as exact sham.',
    'primary': '29–49 s theta log10(sham power / 14 Hz power).',
    'frequency_contrast': '29–49 s theta log10(10 Hz power / 14 Hz power), tested after primary support.',
    'exploratory': '10 Hz vs sham, low beta with and without 13.5–14.5 Hz, three-band distances, late and washout.',
    'support_thresholds': {'theta_reduction_percent': 10., 'additional_reduction_vs_10hz_percent': 5.,
                           'alpha_one_sided': .05, 'minimum_eligible_histories': 12,
                           'minimum_positive_structures': 6, 'minimum_positive_both_histories': 4},
    'release_rule': 'Both histories are preplanned. Wave 2 requires a complete technical audit of wave 1.',
    'carrier_audit': 'Frozen sham-only S0 theta selector. Compare 8–28 s with 29–49 s and retain abstentions.',
}


def jobs(history):
    if history not in HISTORIES:
        raise ValueError(f'Invalid history {history}')
    return [dict(name=f's{seed}_{arm}', seed=seed, arm=arm, project=PROJECT,
                 history_seed=3470000 + 10 * seed + history) for seed in SEEDS for arm in ARMS]


def suite_path(history):
    if history not in HISTORIES:
        raise ValueError(f'Invalid history {history}')
    return ROOT / 'results' / f'l23net_theta_frequency_v1_history{history}'


def code_hashes():
    analyst = Path('experiments/l23net_analysis')
    files = [analyst / name for name in ('theta_frequency_protocol.py', 'run_l23net_theta_frequency.py',
                                        'analyze_l23net_theta_frequency.py', 'nci/submit_l23net_theta_frequency.py',
                                        'nci/run_l23net_theta_frequency_worker.sh')]
    files.append(Path('configs/analysis/l23net_theta_frequency.yaml'))
    return {**repeat.code_hashes(), **{str(file): sha256(ROOT / file) for file in files}}


def load_first_history(path):
    report = reference.load_signed(path)
    expected_names = {job['name'] for job in jobs(1)}
    if (report['history'] != 1 or report['status'] != 'history_complete' or report['errors'] or
            report['completed_valid_runs'] != len(expected_names) or set(report['run_report_sha256']) != expected_names or
            report['protocol'] != PROTOCOL or report['code_sha256'] != code_hashes() or
            report['mechanism_sha256'] != dose.mechanism_hashes() or
            set(report['structures']) != set(map(str, SEEDS))):
        raise ValueError('Wave 1 must pass technical checks before wave 2')
    return report


def check_manifest(directory):
    root = Path(directory)
    manifest = json.loads((root / 'submission.json').read_text())
    history = manifest['history']
    gate, target, late = dose.load_reference(root)
    specifications = [{key: job[key] for key in ('name', 'seed', 'arm', 'project', 'history_seed')}
                      for job in manifest['jobs']]
    if (manifest['protocol'] != PROTOCOL or specifications != jobs(history) or
            manifest['s0_sha256'] != gate['sha256'] or manifest['code_sha256'] != code_hashes() or
            manifest['mechanism_sha256'] != dose.mechanism_hashes()):
        raise ValueError('Submission differs from the frozen theta frequency protocol')
    first = None
    if history == 2:
        first = load_first_history(root / FIRST_HISTORY)
        if manifest['history1_sha256'] != first['sha256'] or manifest['commit'] != first['commit']:
            raise ValueError('History 2 must match the audited first history and its checkout')
    return manifest, gate, target, late, first


def validate_contract(contract, gate, seed):
    actual = copy.deepcopy(contract)
    if (actual.pop('experiment_seed') != seed or actual.pop('condition') != 'mdd' or
            actual['simulation'].pop('MDD') is not True):
        raise ValueError('Expected the planned MDD structure')
    expected = copy.deepcopy(gate['reference_contract'])
    expected['simulation']['duration_ms'] = dose.TIMING_MS[0]
    expected['stimulation_enabled'] = True
    if actual != expected:
        raise ValueError('Simulation, EEG or package settings differ from the qualified contract')


def decide(arm, screen, safe, start_ms):
    delivered = arm if screen['eligible'] and safe else 'sham'
    return {'requested_arm': arm, 'delivered_arm': delivered, 'screen': screen,
            'baseline_rates_safe': safe, 'qualification_screen_bypass': False,
            'amplitude_v_per_m': AMPLITUDE if delivered != 'sham' else 0.,
            'frequency_hz': FREQUENCIES[delivered], 'phase_rad_at_onset': 0.,
            'field_direction': dose.DIRECTION, 'decision_time_ms': start_ms}
