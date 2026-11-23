"""Prespecified L23Net amplitude study and worker/provenance contract."""

import copy
import json
from pathlib import Path

import numpy as np

from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis import s2_pilot_protocol as pilot
from experiments.l23net_analysis.r1_protocol import sha256

ROOT = Path(__file__).resolve().parents[2]
PROJECT = 'fa32'
STAGES = ('qualification', 'discovery')
AMPLITUDES = {'sham': 0., 'dose_02': .2, 'dose_04': .4, 'dose_08': .8}
SEEDS = {'qualification': (8601,), 'discovery': tuple(range(8611, 8617))}
FREQUENCY_HZ = 14.
DIRECTION = [0., 0., 1.]
TIMING_MS = (60000., 1000., 8000., 28000., 50000., 1000.)
EPOCHS_S = {'baseline': [8, 28], 'plateau': [29, 49], 'late': [39, 49], 'washout': [50, 60]}
PRIMARY_CARRIER_EXCLUSION_HZ = [13.5, 14.5]
POPULATIONS = {'HL23PYR': 800, 'HL23SST': 50, 'HL23PV': 70, 'HL23VIP': 80}
RATE_LIMITS = {'HL23PYR': 20., 'HL23SST': 100., 'HL23PV': 100., 'HL23VIP': 100.}
RESOURCES = {'queue': 'normal', 'ncpus': 624, 'memory_gb': 256, 'walltime': '02:00:00'}
RESERVATION_KSU = 2.496
LATE_TARGET = Path('experiments/l23net_analysis/frozen/l23net_dose_late_reference_v1.json')
LATE_SHA256 = 'e88ac472d769fb22cb76f882017c09f0dd5d20ca0d85f17525bb2b9a4667e7ba'
UPPER_GATE = 'upper_dose_qualification.json'
UPPER_CHECKS = {'matched_pre_action', 'upper_dose_delivered', 'baseline_rate_guards', 'paired_rate_guards'}
REFERENCE_SUITE_NAME = 'l23net_reference60_20260930_104958_dxtbj38r'
PROTOCOL = {
    'version': 1, 'name': 'l23net_dose_response', 'project': PROJECT,
    'resources': RESOURCES, 'mpi_ranks': 624, 'dt_ms': .025, 'celsius': 34.,
    'MDD': True, 'DRUG': False, 'env_seed': 0, 'populations': POPULATIONS,
    'seeds': {stage: list(seeds) for stage, seeds in SEEDS.items()},
    'amplitudes_v_per_m': AMPLITUDES, 'frequency_hz': FREQUENCY_HZ,
    'direction': DIRECTION, 'phase_at_onset_rad': 0., 'dc_v_per_m': 0.,
    'duration_ms': 60000, 'window_ms': 1000, 'excluded_ms': 8000,
    'stimulation_ms': [28000, 50000], 'ramp_ms': 1000, 'epochs_s': EPOCHS_S,
    'reference60_target_sha256': pilot.TARGET_SHA, 'late_target_sha256': LATE_SHA256,
    'prior_s2_sha256': '630e4a888e367644cc829a16dfbf06e0db63819ff3c1321719b52a11000ec672',
    'primary': 'Matched sham minus active three-band distance over (29,49] s.',
    'late_secondary': 'Matched distance benefit over (39,49] s with a separate Healthy target.',
    'mechanism': 'Per-cell phase consistency across separated 1-s blocks, with 2-s and circular ISI-permutation sensitivities.',
    'screen': 'Frozen S0 20-s EEG and absolute rate guards. Discovery rejections receive sham and remain in the cohort.',
    'qualification': 'Separate sham/0.8 V/m pair. Bypass phenotype only and require actual delivery, exact pairing, rate and technical checks.',
    'discovery_filter': {'minimum_eligible': 5, 'minimum_positive': 5, 'minimum_mean_benefit': .2,
                         'leave_one_structure_out_mean_positive': True,
                         'excluded_mean_positive': True, 'alignment_mean_positive': True,
                         'all_eligible_rate_safe': True},
    'multiplicity': 'Holm within prespecified outcome families. Candidate selection uses the discovery filter, not a significance claim.',
}


def jobs(stage):
    arms = ('sham', 'dose_08') if stage == 'qualification' else AMPLITUDES
    return [dict(name=f's{seed}_{arm}', seed=seed, arm=arm, project=PROJECT)
            for seed in SEEDS[stage] for arm in arms]


def suite_path(stage):
    if stage not in STAGES:
        raise ValueError(f'Unknown stage {stage}')
    return ROOT / 'results' / f'l23net_dose_response_v1_{stage}'


def code_hashes():
    files = ('dose_response_protocol.py', 'run_l23net_dose_response.py', 'analyze_l23net_dose_response.py',
             'nci/submit_l23net_dose_response.py', 'nci/run_l23net_dose_response_worker.sh')
    paths = [Path('experiments/l23net_analysis') / file for file in files]
    paths.extend([Path('configs/analysis/l23net_dose_response.yaml'), LATE_TARGET])
    return {**pilot.code_hashes(), **{str(path): sha256(ROOT / path) for path in paths}}


def mechanism_hashes():
    directory = ROOT / 'setup/circuits/L23Net/mod'
    return {path.name: sha256(path) for path in sorted(directory.glob('*.mod'))}


def load_reference(directory):
    gate, target = pilot.load_prerequisites(directory)
    late = reference.load_signed(ROOT / LATE_TARGET)
    if (late['sha256'] != LATE_SHA256 or late['reference60_target_sha256'] != target['sha256'] or
            late['interval_s'] != EPOCHS_S['late'] or
            [row['seed'] for row in late['rows']] != list(reference.SEEDS)):
        raise ValueError('Late reference differs from the frozen Healthy calibration')
    fitted = s0.fit_target([row['log_powers'] for row in late['rows']])
    for key in ('mean_log10', 'scale_log10'):
        if not np.allclose(fitted[key], late['target'][key], atol=1e-12, rtol=0):
            raise ValueError('Late reference does not reproduce from its calibration features')
    return gate, target, late


def load_upper_gate(path):
    result = reference.load_signed(path)
    if (result['stage'] != 'qualification' or result['status'] != 'qualified' or result['errors'] or
            result['completed_valid_runs'] != 2 or result['protocol'] != PROTOCOL or
            result['code_sha256'] != code_hashes() or result['mechanism_sha256'] != mechanism_hashes() or
            set(result['qualification_checks']) != UPPER_CHECKS or not all(result['qualification_checks'].values())):
        raise ValueError('Upper-dose qualification is incomplete, failed or from another implementation')
    return result


def check_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'submission.json').read_text())
    stage = manifest['stage']
    gate, target, late = load_reference(directory)
    specifications = [{key: entry[key] for key in ('name', 'seed', 'arm', 'project')}
                      for entry in manifest['jobs']]
    if (manifest['protocol'] != PROTOCOL or specifications != jobs(stage) or
            manifest['s0_sha256'] != gate['sha256'] or manifest['code_sha256'] != code_hashes() or
            manifest['mechanism_sha256'] != mechanism_hashes()):
        raise ValueError('The submission manifest differs from the frozen study')
    if stage == 'discovery':
        qualification = load_upper_gate(directory / UPPER_GATE)
        if manifest['upper_qualification_sha256'] != qualification['sha256']:
            raise ValueError('The qualification report differs from the submission')
    return manifest, gate, target, late


def validate_replay_contract(contract, gate, stage):
    value = copy.deepcopy(contract)
    if value.pop('experiment_seed') not in SEEDS[stage] or value.pop('condition') != 'mdd':
        raise ValueError('Unexpected seed or circuit state')
    if not value['simulation'].pop('MDD'):
        raise ValueError('Expected the MDD circuit')
    expected = copy.deepcopy(gate['reference_contract'])
    expected['simulation']['duration_ms'] = TIMING_MS[0]
    expected['stimulation_enabled'] = True
    if value != expected:
        raise ValueError('Simulation, package or EEG settings differ from the qualified contract')


def baseline_rates_safe(rates):
    for window in rates:
        if set(window) != set(RATE_LIMITS):
            raise ValueError('Baseline is missing population rates')
    return bool(rates) and all(0 <= rate <= RATE_LIMITS[pop] for window in rates for pop, rate in window.items())


def decide(arm, stage, screen, safe, start_ms):
    bypass = stage == 'qualification'
    delivered = arm if safe and (screen['eligible'] or bypass) else 'sham'
    return {'requested_arm': arm, 'delivered_arm': delivered, 'screen': screen,
            'baseline_rates_safe': safe, 'qualification_screen_bypass': bypass,
            'amplitude_v_per_m': AMPLITUDES[delivered],
            'frequency_hz': FREQUENCY_HZ if delivered != 'sham' else 0.,
            'phase_rad_at_onset': 0., 'field_direction': DIRECTION, 'decision_time_ms': start_ms}
