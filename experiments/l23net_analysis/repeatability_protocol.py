"""Frozen two-wave L23Net test of dose response across independent OU histories."""

import json
from pathlib import Path

from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis.r1_protocol import sha256

ROOT = dose.ROOT
PROJECT = dose.PROJECT
WAVES = (1, 2)
SEEDS = dose.SEEDS['discovery']
ARMS = ('sham', 'dose_02', 'dose_04')
ACTIVE_ARMS = ARMS[1:]
SUMMARY = 'repeatability_summary.json'
PREVIOUS_WAVE = 'history1_summary.json'
ARCHIVE = Path('experiments/l23net_analysis/frozen/l23net_repeatability_archive_v1.json')
ARCHIVE_SHA = '5d5a18f5f6b5477b27f082914dcb0fbaeb9bcfcff4a31cb786a6193fa85c7bc6'
NEXT_WAVE_RULE = {'minimum_eligible': 5, 'minimum_mean': .2, 'minimum_positive': 4,
                  'minimum_archived_positive_repeats': 3, 'minimum_leave_one_out_mean': 0.}
PROTOCOL = {
    'version': 1, 'name': 'l23net_ou_history_repeatability', 'project': PROJECT,
    'archive_sha256': ARCHIVE_SHA, 'structure_seeds': list(SEEDS), 'waves': list(WAVES),
    'arm_amplitudes_v_per_m': {arm: dose.AMPLITUDES[arm] for arm in ARMS},
    'frequency_hz': dose.FREQUENCY_HZ, 'direction': dose.DIRECTION, 'epochs_s': dose.EPOCHS_S,
    'resources': dose.RESOURCES, 'reference60_target_sha256': dose.PROTOCOL['reference60_target_sha256'],
    'late_target_sha256': dose.LATE_SHA256,
    'seed_rule': 'Random123(history_seed, cell_gid, OU_site_index); history_seed = 3260000 + 10*structure_seed + wave',
    'primary': 'Matched sham minus active 29–49-s distance. Count rejected histories as sham (zero benefit).',
    'next_wave_rule': NEXT_WAVE_RULE,
    'next_wave_extra_checks': 'Mean carrier-excluded benefit and target alignment positive, all paired rate guards.',
    'dose_preference_margin': .2,
    'interpretation': 'Repeatability within the six discovery structures. Average new histories within structure; archived outcomes supply the replication comparison.',
}


def jobs(wave):
    if wave not in WAVES:
        raise ValueError(f'Invalid history wave {wave}')
    return [{'name': f's{seed}_{arm}', 'seed': seed, 'arm': arm, 'project': PROJECT,
             'history_seed': 3260000 + 10 * seed + wave} for seed in SEEDS for arm in ARMS]


def suite_path(wave):
    if wave not in WAVES:
        raise ValueError(f'Invalid history wave {wave}')
    return ROOT / 'results' / f'l23net_repeatability_v1_history{wave}'


def code_hashes():
    files = ('repeatability_protocol.py', 'run_l23net_repeatability.py', 'analyze_l23net_repeatability.py',
             'nci/submit_l23net_repeatability.py', 'nci/run_l23net_repeatability_worker.sh')
    paths = [Path('experiments/l23net_analysis') / file for file in files]
    setup = Path('setup/circuits/L23Net')
    paths.extend([setup / name for name in ('utils.py', 'background_history.py', 'net_functions.hoc')])
    paths.append(ARCHIVE)
    return {**dose.code_hashes(), **{str(path): sha256(ROOT / path) for path in paths}}


def load_archive():
    archive = reference.load_signed(ROOT / ARCHIVE)
    if archive['sha256'] != ARCHIVE_SHA or [row['seed'] for row in archive['rows']] != list(SEEDS):
        raise ValueError('Archived dose study fingerprints differ from this protocol')
    return {row['seed']: row for row in archive['rows']}


def load_first_wave(path):
    report = reference.load_signed(path)
    if (report['wave'] != 1 or report['status'] != 'history_complete' or report['errors'] or
            report['completed_valid_runs'] != len(jobs(1)) or report['protocol'] != PROTOCOL or
            report['code_sha256'] != code_hashes() or report['mechanism_sha256'] != dose.mechanism_hashes()):
        raise ValueError('First-wave result is incomplete or from another implementation')
    if report['second_wave_allowed'] is not True:
        raise ValueError('Futility rule stopped the study after the first history wave')
    return report


def check_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'submission.json').read_text())
    wave = manifest['wave']
    gate, target, late = dose.load_reference(directory)
    specs = [{key: job[key] for key in ('name', 'seed', 'arm', 'project', 'history_seed')}
             for job in manifest['jobs']]
    if (manifest['protocol'] != PROTOCOL or specs != jobs(wave) or manifest['s0_sha256'] != gate['sha256'] or
            manifest['code_sha256'] != code_hashes() or manifest['mechanism_sha256'] != dose.mechanism_hashes()):
        raise ValueError('Repeatability manifest differs from the frozen protocol')
    if wave == 2:
        first = load_first_wave(directory / PREVIOUS_WAVE)
        if manifest['history1_sha256'] != first['sha256'] or manifest['commit'] != first['commit']:
            raise ValueError('Second wave must use the verified first wave and its checkout')
    return manifest, gate, target, late
