"""Frozen, quota-bounded replication pilot; not the powered final S2 study.

Ten new candidate structures, each sham and the S1-selected 14-Hz axial arm.
Eligibility uses only the unchanged S0 baseline screen. Rejections receive
sham and are retained, never replaced. The eligible-seed primary endpoint is
sham-minus-active three-band distance to the absolute-epoch Reference60 target.
Seeds are units; calibration uncertainty and biological variability are not
included in the conditional paired inference. Do not enlarge this cohort or
retune the action/target after looking at its outcomes. A later powered study
must use new seeds (or a prospectively specified sequential design).
"""
import copy
from pathlib import Path

from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis import s1_analysis as s1
from experiments.l23net_analysis.r1_protocol import sha256

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8501, 8511))
ARMS = ('sham', 'fixed_low_beta')
TARGET_SHA = 'e4c17b5dc211947762e3903997a7efeb2b87e5c6fe4885037c8daba0aca38f8c'
SUMMARY_SHA = 'c9f14ba268b2d827a86a2adf59866dab643dec3f433bb98ebd998dc6af1c2eff'
WALLTIME = '01:40:00'
RESERVATION_KSU = 624 * 2 * (100/60) / 1000
PREREQUISITES = ('reference60_summary.json', 'reference60_target.json', 's0_gate.json', 'qualification.json')
PROTOCOL = {
    'version': 1, 'stage': 's2_pilot', 'seeds': list(SEEDS), 'arms': list(ARMS),
    'duration_ms': 60000, 'mpi_ranks': 624, 'dt_ms': .025, 'celsius': 34,
    'MDD': True, 'DRUG': False, 'env_seed': 0,
    'excluded_ms': 8000, 'baseline_ms': [8000, 28000],
    'stimulation_ms': [28000, 50000], 'ramp_ms': 1000,
    'primary_ms': [29000, 49000], 'washout_ms': [50000, 60000],
    'frequency_hz': 14., 'amplitude_v_per_m': .4, 'dc_v_per_m': 0.,
    'field_direction': [0., 0., 1.], 'phase_rad_at_onset': 0.,
    'target_sha256': TARGET_SHA, 'reference_summary_sha256': SUMMARY_SHA,
    'selection': 'S1 seeds 8401--8403; freeze the previously selected 14-Hz arm, no reranking.',
    'selection_active_report_sha256': {
        's8401_fixed_low_beta': '4ab569337e5d3d0a7b0ddb3fc6ed28da9b11620083f19dffcb7effc7b53882a2',
        's8402_fixed_low_beta': '3e9fe3aa25c70adad126b0beec6b3e7e970e242a59c9a9fceaac47fe2b34090a',
        's8403_fixed_low_beta': '5cbdef7c7a52b86f420374e9b1ca699c0c29fdab7d486a8c5498fefac80f3052'},
    'screen': 'Unchanged S0 8--28-s phenotype and rate screen; rejection maps to sham; no replacement.',
    'primary': 'Eligible-seed mean(sham distance - active distance), 29--49 s, Reference60 three-band RMS z distance.',
    'pilot_support_rule': {'minimum_eligible': 8, 'minimum_mean_benefit': .2,
                           'minimum_positive_fraction': .7, 'one_sided_sign_flip_alpha': .05,
                           'excluded_mean_positive': True, 'alignment_mean_positive': True,
                           'all_eligible_rate_safe': True},
    'rate_guard': 'Plateau and washout population means within 20% of max(sham rate,0.1 Hz).',
    'secondary': 'Three bandwise log(sham/active) effects with BH correction; carrier-excluded distance; washout; all-candidate policy effects.',
    'interpretation': 'Small held-out replication pilot, not powered clinical/model efficacy confirmation or permission for bandits.',
    'margin_rationale': '0.2 standardized-distance units is a prospective model-scale pilot threshold, not a clinical effect size.',
    'historical_input': 'Keep the original synaptic event just after 4 s; exclude first 8 s from screening/outcomes.',
    'resources': {'queue': 'normal', 'ncpus': 624, 'memory_gb': 256, 'walltime': WALLTIME},
}
NEW_CODE = ['configs/analysis/l23net_s2_pilot.yaml',
            'experiments/l23net_analysis/s2_pilot_protocol.py',
            'experiments/l23net_analysis/run_l23net_s2_pilot.py',
            'experiments/l23net_analysis/analyze_l23net_s2_pilot.py',
            'experiments/l23net_analysis/nci/submit_l23net_s2_pilot.py',
            'experiments/l23net_analysis/nci/run_l23net_s2_pilot_worker.sh']


def code_hashes():
    # Keep every previous qualification hash valid: no simulation-kernel edits.
    return {**reference.code_hashes(), **{f: sha256(ROOT/f) for f in NEW_CODE}}


def load_prerequisites(directory):
    directory = Path(directory)
    target = reference.load_signed(directory/'reference60_target.json')
    summary = reference.load_signed(directory/'reference60_summary.json')
    gate = s0.load_gate(directory/'s0_gate.json')
    if (target['sha256'] != TARGET_SHA or summary['sha256'] != SUMMARY_SHA or
            summary['target_sha256'] != TARGET_SHA or summary['status'] != 'reference_calibrated' or
            summary['errors'] or summary['completed_valid_runs'] != 16 or
            target['code_sha256'] != reference.code_hashes() or target['s0_sha256'] != gate['sha256']):
        raise ValueError('Frozen Reference60 prerequisite differs or is incomplete; do not refit it')
    if target['targets']['baseline_screen_unchanged'] != gate['targets']['20s']:
        raise ValueError('Baseline eligibility target changed')
    s1.load_qualification(directory/'qualification.json', gate['sha256'])
    return gate, target


def latest_reference():
    paths = sorted(p for p in (ROOT/'results').glob('l23net_reference60_*')
                   if (p/'reference60_summary.json').is_file())
    if not paths:
        raise ValueError('Supply --reference with the completed Reference60 suite directory')
    return paths[-1]


def jobs():
    return [{'name': f's{seed}_{arm}', 'run': {'seed': seed, 'condition': 'mdd', 'arm': arm}}
            for seed in SEEDS for arm in ARMS]


def validate_contract(contract, gate):
    value = copy.deepcopy(contract)
    if (value['experiment_seed'] not in SEEDS or value['condition'] != 'mdd' or
            value['simulation']['MDD'] is not True or value['simulation']['DRUG'] is not False):
        raise ValueError('Wrong held-out seed/condition')
    value.pop('experiment_seed'); value.pop('condition'); value['simulation'].pop('MDD')
    expected = copy.deepcopy(gate['reference_contract'])
    expected['simulation']['duration_ms'] = 60000.
    expected['stimulation_enabled'] = True
    if value != expected:
        raise ValueError('Scientific/precision/environment contract differs from qualified S1')


def check_manifest(manifest, directory):
    gate, target = load_prerequisites(directory)
    if (manifest['stage'] != 's2_pilot' or manifest['protocol'] != PROTOCOL or
            manifest['code_sha256'] != code_hashes() or manifest['target_sha256'] != target['sha256'] or
            manifest['s0_sha256'] != gate['sha256'] or
            [{k:j[k] for k in ('name','run')} for j in manifest['jobs']] != jobs()):
        raise ValueError('Pilot submission identity or complete planned seed/arm set changed')
    return gate, target
