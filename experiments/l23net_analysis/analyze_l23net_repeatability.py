"""Verify raw L23Net histories and evaluate the predeclared repeatability gate."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis import repeatability_protocol as p
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.analyze_l23net_dose_response import REPORT, TRACE, adjust_family, audit_run
from experiments.l23net_analysis.analyze_l23net_s2_pilot import contrast, inference
from experiments.l23net_analysis.r1_analysis import frozen_json, pbs_epilogue
from experiments.l23net_analysis.r1_protocol import save_json, sha256
from experiments.l23net_analysis.replay_validation import canonical_json_sha256
from experiments.l23net_analysis.s1_analysis import compare_prefix


def prestimulation_hash(path, report):
    """Recompute the same window-interleaved EEG/dipole hash as the worker."""
    dt = report['replay_contract']['network']['dt_ms']
    stop = report['configuration']['analysis']['stim_start_ms']
    digest = hashlib.sha256()
    with h5py.File(path, 'r') as trace:
        for window in report['windows']:
            if window['stop_ms'] > stop:
                break
            start, end = (round(window[key] / dt) for key in ('start_ms', 'stop_ms'))
            for name in ('eeg_v', 'dipole_nA_um'):
                digest.update(np.ascontiguousarray(trace[name][..., start:end], dtype='<f8').tobytes())
    return digest.hexdigest()


def audit_history(report, job, archived):
    config = report['configuration']
    if (report['seed_manifest']['experiment_seed'], config['analysis']['mode'], config['analysis']['arm']) != (
            job['seed'], 'discovery', job['arm']):
        raise ValueError('Completed trajectory differs from its planned structure or arm')
    audit = report['build_audit']
    if (report['structure']['global_sha256'] != archived['structure_sha256'] or
            audit['invariant_sha256'] != archived['build_sha256']):
        raise ValueError('Structure or OU parameters differ from the archived dose experiment')
    history = audit['background_history']
    ranks = history['by_rank']
    seed = job['history_seed']
    if (config['env']['online']['l23net_history_seed'] != seed or history['history_seed'] != seed or
            history['generator'] != 'Random123' or history['global_index'] != 0 or
            len(ranks) != report['mpi']['size'] or len(ranks) != len(audit['by_rank']) or
            history['stream_sha256'] != canonical_json_sha256(ranks) or history['ou_sites'] <= 0 or
            history['ou_sites'] != sum(row['ou_sites'] for row in ranks) or
            any(row['history_seed'] != seed or row['generator'] != 'Random123' or row['global_index'] != 0 or
                row['ou_sites'] != build['counts']['ou_processes'] for row, build in zip(ranks, audit['by_rank']))):
        raise ValueError('OU history provenance or coverage differs from the planned stream')
    return history


def arm_summaries(rows, archive):
    keys = {(row['seed'], row['arm']) for row in rows}
    if len(keys) != len(rows) or keys != {(seed, arm) for seed in p.SEEDS for arm in p.ACTIVE_ARMS}:
        raise ValueError('Analysis requires the complete matched cohort')
    summaries = {}
    for arm in p.ACTIVE_ARMS:
        planned = [row for row in rows if row['arm'] == arm]
        selected = [row for row in planned if row['eligible']]
        primary = inference([row['benefit'] for row in planned])
        effects = [row['benefit'] for row in planned]
        loo = (sum(effects) - max(effects)) / (len(effects) - 1)
        repeats = sum(row['benefit'] > 0 and archive[row['seed']]['benefit'][arm] > 0 for row in planned)
        rule = p.NEXT_WAVE_RULE
        secondary = {key: inference([row[key] for row in selected]) for key in
                     ('late_benefit', 'excluded_benefit', 'shift_alignment', 'washout_benefit')}
        checks = {
            'coverage': len(selected) >= rule['minimum_eligible'],
            'practical_mean': primary['mean'] >= rule['minimum_mean'],
            'consistency': primary['positive_structures'] >= rule['minimum_positive'],
            'archived_positive_repeat': repeats >= rule['minimum_archived_positive_repeats'],
            'leave_one_out': loo > rule['minimum_leave_one_out_mean'],
            'excluded_direction': secondary['excluded_benefit'].get('mean', 0) > 0,
            'alignment': secondary['shift_alignment'].get('mean', 0) > 0,
            'rate_guards': bool(selected) and all(row['rate_safe'] for row in selected),
        }
        bands = {band: inference([row['band_log10_sham_minus_active'][index] for row in selected])
                 for index, band in enumerate(s0.BANDS)}
        summaries[arm] = {'all_candidate_benefit': primary,
                          'eligible_benefit': inference([row['benefit'] for row in selected]),
                          'eligible_seeds': [row['seed'] for row in selected],
                          'archived_positive_repeats': repeats, 'minimum_leave_one_out_mean': loo,
                          'repeatability_checks': checks, 'passes_repeatability_rule': all(checks.values()),
                          'band_log10_sham_minus_active': bands, **secondary}
    adjust_family([values['all_candidate_benefit'] for values in summaries.values()])
    return summaries


def compare_histories(first, second):
    first_rows = {(row['seed'], row['arm']): row for row in first['rows']}
    second_rows = {(row['seed'], row['arm']): row for row in second['rows']}
    summaries, mean_effects = {}, {}
    for arm in p.ACTIVE_ARMS:
        repeated, both_enrolled_differences = [], []
        for seed in p.SEEDS:
            a, b = first_rows[seed, arm], second_rows[seed, arm]
            values = [a['benefit'], b['benefit']]
            mean_effects[seed, arm] = sum(values) / 2
            if a['eligible'] and b['eligible']:
                both_enrolled_differences.append(values[0] - values[1])
            repeated.append({'seed': seed, 'benefits': values, 'eligible': [a['eligible'], b['eligible']],
                             'mean_benefit': mean_effects[seed, arm]})
        summaries[arm] = {
            'by_structure': repeated,
            'mean_of_two_new_histories': inference([row['mean_benefit'] for row in repeated]),
            'positive_both_histories': sum(min(row['benefits']) > 0 for row in repeated),
            'eligible_both_histories': len(both_enrolled_differences),
            'within_structure_sd_among_jointly_eligible': (
                float(np.sqrt(np.mean(np.square(both_enrolled_differences)) / 2))
                if both_enrolled_differences else None),
        }
    adjust_family([values['mean_of_two_new_histories'] for values in summaries.values()])
    low, high = p.ACTIVE_ARMS
    difference = inference([mean_effects[seed, low] - mean_effects[seed, high] for seed in p.SEEDS])
    advantages = [wave['arms'][low]['all_candidate_benefit']['mean'] - wave['arms'][high]['all_candidate_benefit']['mean']
                  for wave in (first, second)]
    margin = p.PROTOCOL['dose_preference_margin']
    preference = low if min(advantages) >= margin else high if max(advantages) <= -margin else None
    return {'by_dose': summaries,
            'dose_02_minus_dose_04': difference,
            'dose_preference_margin': margin, 'dose_02_advantage_by_wave': advantages,
            'same_practical_dose_preference_in_both_waves': preference,
            'interpretation': 'Descriptive repeatability conditional on the studied structures and first-wave release.'}


def pair_history(root, runs, seed, target, late, archived, first):
    sham, sham_psd = runs[seed, 'sham']
    sham_path = root / f's{seed}_sham'
    background = sham['build_audit']['background_history']
    prefix = prestimulation_hash(sham_path / TRACE, sham)
    if (prefix != sham['window_protocol']['prestimulation_eeg_dipole_sha256'] or
            prefix == archived['legacy_prefix_sha256']):
        raise ValueError('New prestimulation trace hash is invalid or duplicates the archived history')
    history = {'history_seed': background['history_seed'], 'ou_sites': background['ou_sites'],
               'stream_sha256': background['stream_sha256'], 'prestimulation_sha256': prefix,
               'structure_sha256': sham['structure']['global_sha256']}
    if first is not None:
        other = first['histories'][str(seed)]
        if (history['structure_sha256'] != other['structure_sha256'] or history['ou_sites'] != other['ou_sites'] or
                any(history[key] == other[key] for key in ('history_seed', 'stream_sha256', 'prestimulation_sha256'))):
            raise ValueError('Second wave must preserve structure and change the realized OU history')
    rows = []
    for arm in p.ACTIVE_ARMS:
        active, active_psd = runs[seed, arm]
        compare_prefix(sham_path, root / f's{seed}_{arm}', sham, active)
        if active['build_audit']['background_history'] != background:
            raise ValueError(f'{arm} did not use the matched OU streams')
        a, b = [report['window_protocol']['decision'] for report in (sham, active)]
        if any(a[key] != b[key] for key in ('screen', 'baseline_rates_safe')):
            raise ValueError(f'{arm} used a different pre-action screen')
        if b['delivered_arm'] == 'sham' and (
                sham['artifacts']['trace_summary'] != active['artifacts']['trace_summary'] or
                [w['spikes'] for w in sham['windows']] != [w['spikes'] for w in active['windows']]):
            raise ValueError(f'{arm} fallback differs from its matched sham')
        row = contrast(seed, sham, active, target)
        row.update(arm=arm, delivered_arm=b['delivered_arm'], archived_benefit=archived['benefit'][arm],
                   late_benefit=s0.scores(sham_psd['late'], late['target'])['distance'] -
                   s0.scores(active_psd['late'], late['target'])['distance'])
        rows.append(row)
    return history, rows


def analyze_suite(root, *, write=True):
    manifest, gate, target, late = p.check_manifest(root)
    wave = manifest['wave']
    archive = p.load_archive()
    first = p.load_first_wave(root / p.PREVIOUS_WAVE) if wave == 2 else None
    errors, runs, accounting, hashes = [], {}, [], {}
    ids = [job.get('job_id') for job in manifest['jobs']]
    if manifest['submission_status'] != 'submitted' or any(not job_id for job_id in ids) or len(set(ids)) != len(ids):
        errors.append('Worker submission is incomplete or contains duplicate PBS job IDs')
    for job in manifest['jobs']:
        name = job['name']
        folder = root / name
        try:
            if (folder / 'worker_exit_code.txt').read_text().strip() != '0':
                raise ValueError('Worker failed; inspect job_live.log and pbs.err')
            if (folder / 'git_commit.txt').read_text().strip() != manifest['commit']:
                raise ValueError('Worker commit differs from the submission')
            epilogue = pbs_epilogue(folder / 'pbs.out')
            if not epilogue['final_accounting_available']:
                raise ValueError('Final PBS epilogue is pending')
            accounting.append({'name': name, 'job_id': job['job_id'], 'project': p.PROJECT, **epilogue})
            if epilogue['exit_status'] != 0 or epilogue['memory_gb_pbs'] > .90 * dose.RESOURCES['memory_gb']:
                raise ValueError('PBS failure or insufficient memory headroom')
            report, powers, _ = audit_run(folder, gate)
            audit_history(report, job, archive[job['seed']])
            memory = [s['rss_gib']['sum'] for s in report['memory_snapshots'] if s['simulated_ms'] >= 8000]
            if not memory or max(memory) - min(memory) > max(2., .05 * memory[0]):
                raise ValueError('Post-recovery memory variation exceeds the prior tolerance')
            runs[job['seed'], job['arm']] = report, powers
            hashes[name] = sha256(folder / REPORT)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{name}: {exc}')

    result = {'wave': wave, 'status': 'incomplete_or_failed', 'errors': errors,
              'completed_valid_runs': len(runs), 'expected_runs': len(manifest['jobs']),
              'protocol': p.PROTOCOL, 'code_sha256': manifest['code_sha256'], 'commit': manifest['commit'],
              'mechanism_sha256': manifest['mechanism_sha256'], 'run_report_sha256': hashes,
              'history1_sha256': manifest.get('history1_sha256'), 'pbs': accounting}
    rows, histories = [], {}
    if not errors:
        for seed in p.SEEDS:
            try:
                history, pairs = pair_history(root, runs, seed, target, late, archive[seed], first)
                rows.extend(pairs)
                histories[str(seed)] = history
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f'Structure {seed} pairing: {exc}')
    if not errors:
        arms = arm_summaries(rows, archive)
        result.update(status='history_complete', rows=rows, histories=histories, arms=arms,
                      actual_ksu=sum(row['service_units'] for row in accounting) / 1000)
        if wave == 1:
            result['second_wave_allowed'] = any(arm['passes_repeatability_rule'] for arm in arms.values())
        else:
            result['repeatability_across_new_histories'] = compare_histories(first, result)
        if write:
            frozen_json(root / p.SUMMARY, reference.signed(result))
    if write:
        save_json(root / 'repeatability_status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', nargs='?', type=Path, help='Explicit suite directory, including transferred results')
    parser.add_argument('--wave', type=int, choices=p.WAVES, default=1)
    parser.add_argument('--verify-only', action='store_true', help='Audit without creating a release report')
    args = parser.parse_args()
    suite = (args.suite or p.suite_path(args.wave)).resolve()
    result = analyze_suite(suite, write=not args.verify_only)
    fields = ('wave', 'status', 'completed_valid_runs', 'expected_runs', 'errors', 'actual_ksu',
              'second_wave_allowed', 'repeatability_across_new_histories')
    overview = {key: result[key] for key in fields if key in result}
    if 'arms' in result:
        overview['arm_overview'] = {
            arm: {'mean_benefit': value['all_candidate_benefit']['mean'],
                  'eligible_seeds': value['eligible_seeds'], 'checks': value['repeatability_checks']}
            for arm, value in result['arms'].items()}
    print(json.dumps(overview, indent=2))
    print(f'SUITE_DIRECTORY={suite}')
    return 0 if result['status'] == 'history_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
