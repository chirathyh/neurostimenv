"""Audit native recordings and estimate theta suppression across two matched histories."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from env.models.neuron.stimulation import apply_raised_cosine_block_envelope
from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis import s1_analysis as s1
from experiments.l23net_analysis import theta_frequency_protocol as p
from experiments.l23net_analysis.analyze_l23net_dose_response import REPORT, TRACE, read_spikes
from experiments.l23net_analysis.analyze_l23net_repeatability import prestimulation_hash
from experiments.l23net_analysis.analyze_l23net_s2_pilot import contrast, inference
from experiments.l23net_analysis.r1_analysis import frozen_json, pbs_epilogue
from experiments.l23net_analysis.r1_protocol import save_json, sha256
from experiments.l23net_analysis.replay_validation import canonical_json_sha256, trace_content_summary


def audit_run(folder, gate=None):
    report = json.loads((folder / REPORT).read_text())
    config = report['configuration']['analysis']
    debug = config['mode'] == 'debug'
    contract = report['replay_contract']
    protocol = report['window_protocol']
    if (report['status'] != 'passed' or report['errors'] or report['build_audit']['errors'] or
            canonical_json_sha256(contract) != report['replay_contract_sha256'] or
            canonical_json_sha256(p.PROTOCOL) != protocol['theta_protocol_sha256'] or
            protocol['code_sha256'] != s1.code_hashes()):
        raise ValueError('Run failed or replay/protocol fingerprints differ')
    timing = tuple(config[key] for key in ('duration_ms', 'window_ms', 'excluded_ms',
                                            'stim_start_ms', 'stim_stop_ms', 'ramp_ms'))
    if not debug:
        p.validate_contract(contract, gate, report['seed_manifest']['experiment_seed'])
        if (config['mode'] != 'discovery' or timing != dose.TIMING_MS or
                report['build']['population_counts'] != dose.POPULATIONS or protocol['s0_sha256'] != gate['sha256']):
            raise ValueError('Full-network timing, populations or reference gate differ')
    total, interval, excluded, start, stop, ramp = timing
    dt = contract['network']['dt_ms']
    count, samples_per_window = round(total / dt), round(interval / dt)
    windows = report['windows']
    if report['completed_simulated_ms'] != total or len(windows) != round(total / interval):
        raise ValueError('Incomplete trajectory')
    for index, window in enumerate(windows):
        if (window['index'] != index or window['errors'] or window['sample_count'] != samples_per_window or
                window['start_ms'] != index * interval or window['stop_ms'] != (index + 1) * interval):
            raise ValueError(f'Invalid recording window {index}')

    trace_path = folder / TRACE
    summary = trace_content_summary(trace_path)
    if (summary != report['artifacts']['trace_summary'] or summary['committed_samples'] != count or
            summary['committed_windows'] != len(windows)):
        raise ValueError('Trace hash or committed extent differs')
    grid = np.arange(count) * dt
    with h5py.File(trace_path, 'r') as trace:
        eeg, dipole = trace['eeg_v'][:], trace['dipole_nA_um'][:]
        field = trace['field_left_boundary_v_per_m'][:]
        if (eeg.shape != (1, count) or dipole.shape != (3, count) or not np.isfinite(dipole).all() or
                not np.allclose(trace['field_left_boundary_time_ms'][:], grid, rtol=0, atol=1e-7) or
                not np.allclose(trace['sample_time_ms'][:], grid + dt, rtol=0, atol=1e-7)):
            raise ValueError('Invalid native trace or time grid')
    times, eeg = s0.CausalEEG(dt).append(eeg[0])
    powers, carrier = {}, None
    if debug:
        screen = {'eligible': True, 'debug_only': True}
    else:
        epochs = {name: s0.epoch(times, eeg, *bounds) for name, bounds in dose.EPOCHS_S.items()}
        powers = {name: s0.log_powers(values) for name, values in epochs.items()}
        powers['excluded'] = s0.log_powers(epochs['plateau'], dose.PRIMARY_CARRIER_EXCLUSION_HZ)
        screen = s0.scores(powers['baseline'], gate['targets']['20s'])
        for epoch, key in (('baseline', 'baseline'), ('plateau', 'stimulation'), ('washout', 'washout')):
            if not np.allclose(powers[epoch], protocol['outcomes'][key]['log_powers'], rtol=0, atol=1e-9):
                raise ValueError(f'Saved {epoch} spectrum differs from the raw trace')
        if not np.allclose(powers['excluded'], protocol['fundamental_excluded']['low_beta']['log_powers'],
                           rtol=0, atol=1e-9):
            raise ValueError('Saved carrier-excluded spectrum differs from the raw trace')
        if config['arm'] == 'sham':
            baseline = s0.stable_carrier(epochs['baseline'], 'theta')
            later = s0.stable_carrier(epochs['plateau'], 'theta')
            stable = baseline['accepted'] and later['accepted'] and abs(
                baseline['frequency_hz'] - later['frequency_hz']) <= s0.PROTOCOL['peak_half_difference_hz']
            carrier = {'baseline': baseline, 'later_sham': later, 'stable_across_epochs': stable,
                       'phase_audit': s0.phase_audit(times, eeg, 'theta')}

    rates = [s1.population_rates(w['firing_rates']) for w in windows
             if w['start_ms'] >= excluded and w['stop_ms'] <= start]
    decision = p.decide(config['arm'], screen, dose.baseline_rates_safe(rates), start)
    saved = protocol['decision']
    if (any(saved[key] != value for key, value in decision.items() if key != 'screen') or
            any(not np.isclose(saved['screen'][key], value, rtol=0, atol=1e-9) for key, value in screen.items())):
        raise ValueError('Saved field decision does not follow the prospective screen')
    amplitude = decision['amplitude_v_per_m']
    sine = amplitude * np.sin(2 * np.pi * decision['frequency_hz'] * (grid - start) / 1000)
    expected = apply_raised_cosine_block_envelope(sine, time_ms=grid, block_start_ms=start,
                                                  block_stop_ms=stop, ramp_ms=ramp)
    if (not np.allclose(field, expected, rtol=0, atol=1e-10) or np.any(field[(grid < start) | (grid >= stop)]) or
            any(w['field_direction'] != dose.DIRECTION for w in windows if start <= w['start_ms'] < stop)):
        raise ValueError('Raw field does not match the assigned sinusoid, orientation or envelope')
    if amplitude and (np.max(np.abs(field)) < .99 * amplitude or not protocol['tissue_coupling_observed']):
        raise ValueError('Active field did not reach amplitude or couple to the circuit')
    read_spikes(folder, report)
    return report, powers, carrier


def audit_assignment(report, job):
    cfg = report['configuration']
    if (cfg['analysis']['mode'], cfg['analysis']['arm'], report['seed_manifest']['experiment_seed']) != (
            'discovery', job['arm'], job['seed']):
        raise ValueError('Worker seed, arm or mode differs from the submission')
    audit = report['build_audit']
    history = audit['background_history']
    seed = job['history_seed']
    rows = history['by_rank']
    if (cfg['env']['online']['l23net_history_seed'] != seed or history['history_seed'] != seed or
            history['generator'] != 'Random123' or history['global_index'] != 0 or
            len(rows) != report['mpi']['size'] or len(rows) != len(audit['by_rank']) or
            history['stream_sha256'] != canonical_json_sha256(rows) or history['ou_sites'] <= 0 or
            history['ou_sites'] != sum(r['ou_sites'] for r in rows) or any(
                r['history_seed'] != seed or r['generator'] != 'Random123' or r['global_index'] != 0 or
                r['ou_sites'] != a['counts']['ou_processes'] for r, a in zip(rows, audit['by_rank']))):
        raise ValueError('OU streams differ from the assigned independent history')


def pair_structure(root, recorded, seed, target, late):
    sham, sham_power, carrier = recorded[seed, 'sham']
    path = root / f's{seed}_sham'
    background = sham['build_audit']['background_history']
    prefix = prestimulation_hash(path / TRACE, sham)
    if prefix != sham['window_protocol']['prestimulation_eeg_dipole_sha256']:
        raise ValueError('Sham EEG/dipole prefix differs from the saved fingerprint')
    structure = {'structure_sha256': sham['structure']['global_sha256'],
                 'build_sha256': sham['build_audit']['invariant_sha256'], 'history_seed': background['history_seed'],
                 'stream_sha256': background['stream_sha256'], 'prestimulation_sha256': prefix,
                 'ou_sites': background['ou_sites'], 'sham_carrier_audit': carrier}
    rows = []
    for arm in p.ARMS[1:]:
        active, active_power, _ = recorded[seed, arm]
        s1.compare_prefix(path, root / f's{seed}_{arm}', sham, active)
        if active['build_audit']['background_history'] != background:
            raise ValueError(f'{arm} has unmatched OU streams')
        a, b = [r['window_protocol']['decision'] for r in (sham, active)]
        if any(a[key] != b[key] for key in ('screen', 'baseline_rates_safe')):
            raise ValueError(f'{arm} has a different prospective screen')
        if b['delivered_arm'] == 'sham' and (
                sham['artifacts']['trace_summary'] != active['artifacts']['trace_summary'] or
                [w['spikes'] for w in sham['windows']] != [w['spikes'] for w in active['windows']]):
            raise ValueError(f'{arm} fallback was not an exact sham replay')
        row = contrast(seed, sham, active, target)
        row.update(arm=arm, delivered_arm=b['delivered_arm'],
                   low_beta_excluding_14hz_log10_active_over_sham=float(active_power['excluded'][2] - sham_power['excluded'][2]),
                   late_benefit=s0.scores(sham_power['late'], late['target'])['distance'] -
                   s0.scores(active_power['late'], late['target'])['distance'])
        rows.append(row)
    return structure, rows


def percent_reduction(summary):
    result = {**summary}
    convert = lambda x: float(-100 * np.expm1(-np.log(10) * x))
    result['geometric_percent_reduction'] = convert(summary['mean'])
    result['mean_t_ci95_percent_reduction'] = [convert(x) for x in summary['mean_t_ci95']]
    return result


def supported_contrast(effects, safe, eligible, practical_percent):
    # Effects has one row per independent structure and one column per history.
    estimate = percent_reduction(inference(effects.mean(axis=1)))
    rule = p.PROTOCOL['support_thresholds']
    checks = {'eligibility': eligible >= rule['minimum_eligible_histories'],
              'practical_mean': estimate['geometric_percent_reduction'] >= practical_percent,
              'one_sided_sign_flip': estimate['p_one_sided'] <= rule['alpha_one_sided'],
              'structure_consistency': estimate['positive_structures'] >= rule['minimum_positive_structures'],
              'both_histories_positive': int(np.sum((effects > 0).all(axis=1))) >= rule['minimum_positive_both_histories'],
              'each_history_mean_positive': bool((effects.mean(axis=0) > 0).all()), 'rate_guards': safe}
    return {**estimate, 'checks': checks, 'supported': all(checks.values()),
            'history_mean_log10': effects.mean(axis=0).tolist(),
            'positive_in_both_histories': int(np.sum((effects > 0).all(axis=1))),
            'by_structure_log10': {str(seed): values.tolist() for seed, values in zip(p.SEEDS, effects)}}


def summarize_combined(first, second):
    outcomes = []
    for report in (first, second):
        rows = {(row['seed'], row['arm']): row for row in report['rows']}
        if len(rows) != len(report['rows']) or set(rows) != {(seed, arm) for seed in p.SEEDS for arm in p.ARMS[1:]}:
            raise ValueError('The full preplanned structure-by-arm grid is required in both waves')
        outcomes.append(rows)

    def values(arm, get):
        return np.array([[get(rows[seed, arm]) for rows in outcomes] for seed in p.SEEDS])

    _, low_beta, alpha = p.ARMS
    theta = {arm: values(arm, lambda row: row['band_log10_sham_minus_active'][0]) for arm in (low_beta, alpha)}
    eligibility = values(low_beta, lambda row: row['eligible'])
    if not np.array_equal(eligibility, values(alpha, lambda row: row['eligible'])):
        raise ValueError('Screen eligibility disagreed between matched active arms')
    eligible = int(eligibility.sum())
    safety = {arm: bool(values(arm, lambda row: row['rate_safe']).all()) for arm in (low_beta, alpha)}
    rule = p.PROTOCOL['support_thresholds']
    primary = supported_contrast(theta[low_beta], safety[low_beta], eligible, rule['theta_reduction_percent'])
    frequency = supported_contrast(theta[low_beta] - theta[alpha], all(safety.values()), eligible,
                                    rule['additional_reduction_vs_10hz_percent'])
    frequency.update(tested_after_primary=primary['supported'],
                     supported=primary['supported'] and frequency['supported'])
    if not primary['supported']:
        frequency.pop('p_one_sided')
        frequency['checks']['one_sided_sign_flip'] = None
    secondary = {}
    for arm in (low_beta, alpha):
        secondary[arm] = {'theta_vs_sham': percent_reduction(inference(theta[arm].mean(axis=1))),
                          'low_beta_vs_sham': percent_reduction(inference(values(
                              arm, lambda row: row['band_log10_sham_minus_active'][2]).mean(axis=1))),
                          'low_beta_vs_sham_excluding_14hz': percent_reduction(inference(values(
                              arm, lambda row: -row['low_beta_excluding_14hz_log10_active_over_sham']).mean(axis=1)))}
        for key in ('benefit', 'late_benefit', 'washout_benefit'):
            secondary[arm][key] = inference(values(arm, lambda row: row[key]).mean(axis=1))
    carrier = [report['structures'][str(seed)]['sham_carrier_audit'] for seed in p.SEEDS for report in (first, second)]
    return {'n_structures': len(p.SEEDS), 'eligible_histories': eligible, 'planned_histories': int(eligibility.size),
            'primary_14hz_vs_sham': primary, 'frequency_14hz_vs_10hz': frequency, 'secondary': secondary,
            'sham_theta_carrier_audit': {'n_histories': len(carrier),
                                         'baseline_accepted': sum(c['baseline']['accepted'] for c in carrier),
                                         'stable_across_epochs': sum(c['stable_across_epochs'] for c in carrier),
                                         'possible_phase_decisions_per_horizon': len(carrier) * len(s0.PROTOCOL['phase_audit_boundaries_s']),
                                         'phase_predictions_accepted_by_horizon_s': {
                                             str(h): sum(r['accepted'] and r.get('horizon_s') == h
                                                         for c in carrier for r in c['phase_audit']) for h in (1, 4)}}}


def check_structures(structures, first):
    if len({row['structure_sha256'] for row in structures.values()}) != len(p.SEEDS):
        raise ValueError('Structure fingerprints must be unique across the planned seeds')
    if first is not None:
        for seed, new in structures.items():
            old = first['structures'][seed]
            if (any(new[key] != old[key] for key in ('structure_sha256', 'build_sha256', 'ou_sites')) or
                    any(new[key] == old[key] for key in ('history_seed', 'stream_sha256', 'prestimulation_sha256'))):
                raise ValueError(f'Structure {seed} must be unchanged with a new realized OU history')


def analyze_suite(root, *, write=True):
    manifest, gate, target, late, first = p.check_manifest(root)
    errors, runs, hashes, pbs, rows, structures = [], {}, {}, [], [], {}
    ids = [j.get('job_id') for j in manifest['jobs']]
    if manifest['submission_status'] != 'submitted' or None in ids or len(set(ids)) != len(ids):
        errors.append('Incomplete or duplicate PBS submission')
    for job in manifest['jobs']:
        name = job['name']
        folder = root / name
        try:
            if (folder / 'worker_exit_code.txt').read_text().strip() != '0':
                raise ValueError('Worker failed; check job_live.log and pbs.err')
            if (folder / 'git_commit.txt').read_text().strip() != manifest['commit']:
                raise ValueError('Worker checkout differs from the submission')
            accounting = pbs_epilogue(folder / 'pbs.out')
            if not accounting['final_accounting_available']:
                raise ValueError('Final PBS accounting is pending')
            pbs.append({'name': name, 'job_id': job['job_id'], **accounting})
            if accounting['exit_status'] != 0 or accounting['memory_gb_pbs'] > .90 * dose.RESOURCES['memory_gb']:
                raise ValueError('PBS failed or exceeded the qualified memory tolerance')
            run = audit_run(folder, gate)
            audit_assignment(run[0], job)
            rss = [s['rss_gib']['sum'] for s in run[0]['memory_snapshots'] if s['simulated_ms'] >= 8000]
            if not rss or max(rss) - min(rss) > max(2., .05 * rss[0]):
                raise ValueError('Post-recovery memory variation exceeds the qualified tolerance')
            runs[job['seed'], job['arm']] = run
            hashes[name] = sha256(folder / REPORT)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{name}: {exc}')

    if not errors:
        try:
            for seed in p.SEEDS:
                structure, pairs = pair_structure(root, runs, seed, target, late)
                structures[str(seed)] = structure
                rows.extend(pairs)
            check_structures(structures, first)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'Counterfactual pairing: {exc}')

    result = {'history': manifest['history'], 'status': 'incomplete_or_failed', 'errors': errors,
              'completed_valid_runs': len(runs), 'expected_runs': len(manifest['jobs']),
              'commit': manifest['commit'], 'protocol': p.PROTOCOL, 'code_sha256': manifest['code_sha256'],
              'mechanism_sha256': manifest['mechanism_sha256'], 'run_report_sha256': hashes, 'pbs': pbs}
    if not errors:
        result.update(status='history_complete' if first is None else 'study_complete', rows=rows,
                      structures=structures, actual_ksu=sum(row['service_units'] for row in pbs) / 1000)
        if first is not None:
            result.update(history1_sha256=first['sha256'], total_ksu=first['actual_ksu'] + result['actual_ksu'],
                          combined=summarize_combined(first, result))
        if write:
            frozen_json(root / p.SUMMARY, reference.signed(result))
    if write:
        save_json(root / p.STATUS, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', nargs='?', type=Path)
    parser.add_argument('--history', type=int, choices=p.HISTORIES, default=1)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = (args.suite or p.suite_path(args.history)).resolve()
    result = analyze_suite(root, write=not args.verify_only)
    keys = ('history', 'status', 'completed_valid_runs', 'expected_runs', 'errors', 'actual_ksu', 'total_ksu')
    output = {key: result[key] for key in keys if key in result}
    if 'combined' in result:
        for key in ('primary_14hz_vs_sham', 'frequency_14hz_vs_10hz', 'sham_theta_carrier_audit'):
            output[key] = result['combined'][key]
    print(json.dumps(output, indent=2))
    print(f'SUITE_DIRECTORY={root}')
    return 0 if result['status'] in ('history_complete', 'study_complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
