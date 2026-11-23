"""Audit complete dose suites, evaluate qualification and estimate paired effects."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from env.models.neuron.stimulation import apply_raised_cosine_block_envelope
from experiments.l23net_analysis import dose_response_protocol as p
from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis import s1_analysis as s1
from experiments.l23net_analysis.analyze_l23net_s2_pilot import contrast, inference
from experiments.l23net_analysis.r1_analysis import frozen_json, pbs_epilogue
from experiments.l23net_analysis.r1_protocol import save_json, sha256
from experiments.l23net_analysis.replay_validation import canonical_json_sha256, trace_content_summary

REPORT = 'l23net_s1_run.json'
TRACE = 'l23net_s1_trace.h5'
SPIKES = 'l23net_s1_spikes.h5'


def read_spikes(folder, report):
    path = folder / SPIKES
    if sha256(path) != report['window_protocol']['spike_file_sha256']:
        raise ValueError('Sparse spike file differs from the completed report')
    windows = report['windows']
    with h5py.File(path, 'r') as stream:
        if set(stream) != set(p.POPULATIONS) or stream.attrs['committed_windows'] != len(windows):
            raise ValueError('Sparse spike file is incomplete')
        output = {}
        for pop, group in stream.items():
            times, gids = group['times_ms'][:], group['gids'][:]
            if times.shape != gids.shape or not np.isfinite(times).all():
                raise ValueError(f'Invalid sparse spike arrays for {pop}')
            cursor = 0
            cells = report['build']['population_counts'][pop]
            if np.unique(gids).size > cells:
                raise ValueError(f'Too many distinct spike GIDs in {pop}')
            for window in windows:
                count = window['spikes']['counts'][pop]
                subset = times[cursor:cursor + count]
                duration_s = (window['stop_ms'] - window['start_ms']) / 1000
                rate = s1.population_rates(window['firing_rates'])[pop]
                if (len(subset) != count or np.any(subset < window['start_ms'] - 1e-7) or
                        np.any(subset > window['stop_ms'] + 1e-7) or
                        not np.isclose(rate, count / (cells * duration_s), rtol=0, atol=1e-10)):
                    raise ValueError(f'Spike count, times or rate mismatch for {pop}')
                cursor += count
            if cursor != len(times):
                raise ValueError(f'Uncommitted spike samples in {pop}')
            mask = (times > 29000) & (times <= 49000)
            times_s, cell_ids = (times[mask] - 29000) / 1000, gids[mask]
            order = np.lexsort((times_s, cell_ids))
            ids, starts = np.unique(cell_ids[order], return_index=True)
            ends = np.r_[starts[1:], len(order)]
            output[pop] = {int(gid): times_s[order[first:last]] for gid, first, last in zip(ids, starts, ends)}
        return output


def audit_run(folder, gate=None):
    """Read native trace/field/spike artifacts and recompute the causal EEG screen."""
    report = json.loads((folder / REPORT).read_text())
    windows = report['windows']
    if report['status'] != 'passed' or report['errors'] or report['build_audit']['errors']:
        raise ValueError('The worker did not complete its technical checks')
    protocol = report['window_protocol']
    contract = report['replay_contract']
    if (canonical_json_sha256(contract) != report['replay_contract_sha256'] or
            protocol['dose_protocol_sha256'] != canonical_json_sha256(p.PROTOCOL) or
            protocol['code_sha256'] != s1.code_hashes()):
        raise ValueError('Replay/protocol source identity differs')
    config = report['configuration']['analysis']
    stage, arm = config['mode'], config['arm']
    total, window_ms, excluded, start, stop, ramp = (config[key] for key in (
        'duration_ms', 'window_ms', 'excluded_ms', 'stim_start_ms', 'stim_stop_ms', 'ramp_ms'
    ))
    if stage != 'debug':
        p.validate_replay_contract(contract, gate, stage)
        if (report['build']['population_counts'] != p.POPULATIONS or protocol['s0_sha256'] != gate['sha256'] or
                (total, window_ms, excluded, start, stop, ramp) != p.TIMING_MS):
            raise ValueError('Full-network population, S0 or timing mismatch')
    dt = contract['network']['dt_ms']
    count = round(total / dt)
    samples_per_window = round(window_ms / dt)
    if len(windows) != round(total / window_ms) or report['completed_simulated_ms'] != total:
        raise ValueError('Incomplete window sequence')
    for index, window in enumerate(windows):
        if (window['index'] != index or window['errors'] or window['sample_count'] != samples_per_window or
                window['start_ms'] != index * window_ms or window['stop_ms'] != (index + 1) * window_ms):
            raise ValueError('Invalid recording window')

    summary = trace_content_summary(folder / TRACE)
    if (summary != report['artifacts']['trace_summary'] or summary['committed_samples'] != count or
            summary['committed_windows'] != len(windows)):
        raise ValueError('Native trace hash or committed extent differs')
    with h5py.File(folder / TRACE, 'r') as trace:
        eeg, dipole = trace['eeg_v'][:], trace['dipole_nA_um'][:]
        if eeg.shape != (1, count) or dipole.shape != (3, count):
            raise ValueError('Invalid EEG/dipole shape')
        if not np.isfinite(dipole).all():
            raise ValueError('Nonfinite dipole recording')
        field_time = trace['field_left_boundary_time_ms'][:]
        grid = np.arange(count) * dt
        if (not np.allclose(field_time, grid, atol=1e-7, rtol=0) or
                not np.allclose(trace['sample_time_ms'][:], grid + dt, atol=1e-7, rtol=0)):
            raise ValueError('Incorrect native sample/field time grid')
        times, eeg = s0.CausalEEG(dt).append(eeg[0])
        field = trace['field_left_boundary_v_per_m'][:]
    powers = {}
    if stage == 'debug':
        screen = {'eligible': True, 'debug_only': True}
    else:
        powers = {name: s0.log_powers(s0.epoch(times, eeg, *epoch)) for name, epoch in p.EPOCHS_S.items()}
        powers['excluded'] = s0.log_powers(s0.epoch(times, eeg, *p.EPOCHS_S['plateau']),
                                          p.PRIMARY_CARRIER_EXCLUSION_HZ)
        screen = s0.scores(powers['baseline'], gate['targets']['20s'])
        for epoch, saved in [('baseline', 'baseline'), ('plateau', 'stimulation'), ('washout', 'washout')]:
            if not np.allclose(powers[epoch], protocol['outcomes'][saved]['log_powers'], rtol=0, atol=1e-9):
                raise ValueError(f'Saved {epoch} spectral features differ from the raw trace')
        if not np.allclose(powers['excluded'], protocol['fundamental_excluded']['low_beta']['log_powers'],
                           rtol=0, atol=1e-9):
            raise ValueError('Saved carrier-excluded spectral features differ from the raw trace')

    rates = [s1.population_rates(w['firing_rates']) for w in windows
             if w['start_ms'] >= excluded and w['stop_ms'] <= start]
    decision = p.decide(arm, stage, screen, p.baseline_rates_safe(rates), start)
    saved = protocol['decision']
    if any(saved[key] != value for key, value in decision.items() if key != 'screen'):
        raise ValueError('Delivered arm or phase differs from the prospective decision')
    if any(not np.isclose(saved['screen'][key], value, rtol=0, atol=1e-9) for key, value in screen.items()):
        raise ValueError('Saved eligibility differs from the raw baseline')
    amplitude = decision['amplitude_v_per_m']
    sine = amplitude * np.sin(2 * np.pi * decision['frequency_hz'] * (field_time - start) / 1000)
    expected = apply_raised_cosine_block_envelope(sine, time_ms=field_time, block_start_ms=start,
                                                  block_stop_ms=stop, ramp_ms=ramp)
    if not np.allclose(field, expected, rtol=0, atol=1e-10) or np.any(field[(grid < start) | (grid >= stop)]):
        raise ValueError('Raw field differs from the analytic envelope or zero-field epochs')
    if amplitude and (np.max(np.abs(field)) < .99 * amplitude or not protocol['tissue_coupling_observed']):
        raise ValueError('An active field failed to reach dose or couple to the circuit')
    if any(w['field_direction'] != p.DIRECTION for w in windows if start <= w['start_ms'] < stop):
        raise ValueError('Unexpected field orientation')
    return report, powers, read_spikes(folder, report)


def separated_ppc(sham, active, width_s):
    count = round(20 / width_s)
    distant = np.abs(np.arange(count)[:, None] - np.arange(count)[None, :]) > 1
    phase_vectors = []
    occupied = []
    for times in (sham, active):
        block = np.clip(np.ceil(times / width_s).astype(int) - 1, 0, count - 1)
        phase = np.exp(2j * np.pi * p.FREQUENCY_HZ * times)
        n = np.bincount(block, minlength=count)
        total = (np.bincount(block, weights=phase.real, minlength=count) +
                 1j * np.bincount(block, weights=phase.imag, minlength=count))
        phase_vectors.append(np.divide(total, n, out=np.zeros_like(total), where=n > 0))
        occupied.append(n > 0)
    common = occupied[0] & occupied[1]
    keep = distant & common[:, None] & common[None, :]
    if not keep.any():
        return None
    values = [np.outer(v, v.conj()).real[keep].mean() for v in phase_vectors]
    return float(values[1] - values[0])


def isi_adjusted_ppc(times, rng):
    """Subtract a circular-ISI permutation expectation from one cell's PPC."""
    n = len(times)
    gaps = np.diff(np.r_[times, times[0] + 20])
    if n < 2 or np.any(gaps < 0):
        raise ValueError('Invalid circular spike intervals')
    permuted = rng.permuted(np.broadcast_to(gaps, (512, n)), axis=1)
    # The first spike is at 0 under the null. A constant phase rotation leaves PPC unchanged.
    phase = 2 * np.pi * p.FREQUENCY_HZ
    null = np.abs(1 + np.exp(1j * phase * np.cumsum(permuted[:, :-1], axis=1)).sum(axis=1)) ** 2
    observed = np.abs(np.exp(1j * phase * times).sum()) ** 2
    return float((observed - null.mean()) / (n * (n - 1)))


def timing_contrast(seed, sham, active):
    result = {}
    for index, pop in enumerate(p.POPULATIONS):
        values = {'separated_1s': [], 'separated_2s': [], 'isi_adjusted': []}
        for gid in sorted(sham[pop].keys() & active[pop].keys()):
            a, b = sham[pop][gid], active[pop][gid]
            for width in (1, 2):
                delta = separated_ppc(a, b, width)
                if delta is not None:
                    values[f'separated_{width}s'].append(delta)
            if min(len(a), len(b)) >= 2:
                rng = np.random.default_rng([20261124, seed, index, gid])
                values['isi_adjusted'].append(isi_adjusted_ppc(b, rng) - isi_adjusted_ppc(a, rng))
        result[pop] = {method: {'paired_cells': len(v), 'active_minus_sham': float(np.mean(v)) if v else None}
                       for method, v in values.items()}
    return result


def adjust_family(results, test_key='p_one_sided'):
    """Holm adjustment with missing comparisons retained in the family size."""
    ordered = sorted(results, key=lambda result: result.get(test_key, 1.))
    adjusted = 0.
    for index, result in enumerate(ordered):
        adjusted = max(adjusted, min(1., result.get(test_key, 1.) * (len(ordered) - index)))
        if test_key in result:
            result['p_holm'] = adjusted


def summarize_discovery(rows):
    arms = [arm for arm in p.AMPLITUDES if arm != 'sham']
    planned_keys = [(seed, arm) for seed in p.SEEDS['discovery'] for arm in arms]
    by_key = {(row['seed'], row['arm']): row for row in rows}
    if set(by_key) != set(planned_keys) or len(by_key) != len(rows):
        raise ValueError('Inference requires the complete planned cohort')
    eligible_seeds = sorted({row['seed'] for row in rows if row['eligible']})
    summaries = {}
    bands, mechanisms = [], []
    for arm in arms:
        planned = [row for row in rows if row['arm'] == arm]
        selected = [row for row in planned if row['eligible']]
        primary = np.asarray([row['benefit'] for row in selected])
        values = {metric: inference([row[metric] for row in selected])
                  for metric in ('benefit', 'late_benefit', 'excluded_benefit', 'washout_benefit', 'shift_alignment')}
        for index, band in enumerate(s0.BANDS):
            test = inference([row['band_log10_sham_minus_active'][index] for row in selected])
            if 'mean' in test:
                test['active_vs_sham_power_percent'] = float(100 * np.expm1(-np.log(10) * test['mean']))
            bands.append(test)
            values.setdefault('bands', {})[band] = test
        timing = {}
        for pop in p.POPULATIONS:
            timing[pop] = {}
            for method in ('separated_1s', 'separated_2s', 'isi_adjusted'):
                effects = [row['timing'][pop][method]['active_minus_sham'] for row in selected]
                test = inference(effects) if None not in effects else {'incomplete_paired_population': True}
                mechanisms.append(test)
                timing[pop][method] = test
        values['timing'] = timing
        loo = float((primary.sum() - primary.max()) / (len(primary) - 1)) if len(primary) > 1 else None
        values['minimum_leave_one_out_mean'] = loo
        rule = p.PROTOCOL['discovery_filter']
        checks = {'coverage': len(selected) >= rule['minimum_eligible'],
                  'practical_mean': bool(primary.size and primary.mean() >= rule['minimum_mean_benefit']),
                  'consistency': int(np.sum(primary > 0)) >= rule['minimum_positive'],
                  'leave_one_out': loo is not None and loo > 0,
                  'excluded_direction': values['excluded_benefit'].get('mean', 0) > 0,
                  'alignment': values['shift_alignment'].get('mean', 0) > 0,
                  'rate_guards': bool(selected) and all(row['rate_safe'] for row in selected)}
        values.update(discovery_checks=checks, candidate=all(checks.values()),
                      all_candidate_benefit=inference([row['benefit'] for row in planned]))
        summaries[arm] = values

    for metric in ('benefit', 'late_benefit', 'excluded_benefit', 'washout_benefit', 'shift_alignment'):
        adjust_family([values[metric] for values in summaries.values()])
    adjust_family(bands)
    adjust_family(mechanisms)
    dose_comparisons = {}
    for lower, higher in combinations(arms, 2):
        diff = np.array([by_key[seed, higher]['benefit'] - by_key[seed, lower]['benefit'] for seed in eligible_seeds])
        test = inference(diff)
        if 'p_one_sided' in test:
            reverse_p = inference(-diff)['p_one_sided']
            test['p_two_sided'] = min(1., 2 * min(test.pop('p_one_sided'), reverse_p))
        dose_comparisons[f'{higher}_minus_{lower}'] = test
    adjust_family(dose_comparisons.values(), test_key='p_two_sided')
    passing = [arm for arm in arms if summaries[arm]['candidate']]
    candidate = max(passing, key=lambda arm: summaries[arm]['benefit']['mean'], default=None)
    return {'eligible_seeds': eligible_seeds, 'arms': summaries, 'dose_comparisons': dose_comparisons,
            'candidate_for_independent_confirmation': candidate}


def analyze_suite(root, *, write=True):
    manifest, gate, target, late = p.check_manifest(root)
    stage = manifest['stage']
    errors, runs, accounting, hashes = [], {}, [], {}
    ids = [job.get('job_id') for job in manifest['jobs']]
    if manifest['submission_status'] != 'submitted' or None in ids or len(set(ids)) != len(ids):
        errors.append('The job submission is incomplete or contains duplicate IDs')
    for job in manifest['jobs']:
        name = job['name']
        folder = root / name
        try:
            if (folder / 'worker_exit_code.txt').read_text().strip() != '0':
                raise ValueError('Worker failed, see job_live.log and pbs.err')
            if (folder / 'git_commit.txt').read_text().strip() != manifest['commit']:
                raise ValueError('Worker commit differs from the submission')
            epilogue = pbs_epilogue(folder / 'pbs.out')
            if not epilogue['final_accounting_available']:
                raise ValueError('Final PBS epilogue is pending')
            accounting.append({'name': name, 'job_id': job['job_id'], 'project': job['project'], **epilogue})
            if epilogue['exit_status'] != 0 or epilogue['memory_gb_pbs'] > .90 * p.RESOURCES['memory_gb']:
                raise ValueError('PBS failure or insufficient memory headroom')
            report, powers, spikes = audit_run(folder, gate)
            config = report['configuration']['analysis']
            if (report['seed_manifest']['experiment_seed'], config['mode'], config['arm']) != (job['seed'], stage, job['arm']):
                raise ValueError('Completed trajectory differs from the planned seed/stage/arm')
            memory = [s['rss_gib']['sum'] for s in report['memory_snapshots'] if s['simulated_ms'] >= 8000]
            if not memory or max(memory) - min(memory) > max(2., .05 * memory[0]):
                raise ValueError('Post-recovery memory variation exceeds the prior tolerance')
            runs[(job['seed'], job['arm'])] = report, powers, spikes
            hashes[name] = sha256(folder / REPORT)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{name}: {exc}')

    result = {'stage': stage, 'status': 'incomplete_or_failed', 'errors': errors,
              'completed_valid_runs': len(runs), 'expected_runs': len(manifest['jobs']),
              'protocol': p.PROTOCOL, 'code_sha256': manifest['code_sha256'],
              'mechanism_sha256': manifest['mechanism_sha256'], 'run_report_sha256': hashes,
              'commit': manifest['commit'], 'pbs': accounting}
    rows, structures = [], {}
    if not errors:
        for job in manifest['jobs']:
            seed, arm = job['seed'], job['arm']
            if arm == 'sham':
                continue
            try:
                sham, sham_psd, sham_spikes = runs[(seed, 'sham')]
                active, active_psd, active_spikes = runs[(seed, arm)]
                s1.compare_prefix(root / f's{seed}_sham', root / job['name'], sham, active)
                decisions = [report['window_protocol']['decision'] for report in (sham, active)]
                if any(decisions[0][key] != decisions[1][key] for key in ('screen', 'baseline_rates_safe')):
                    raise ValueError('Pre-action eligibility differs across paired arms')
                if decisions[1]['delivered_arm'] == 'sham':
                    if sham['artifacts']['trace_summary'] != active['artifacts']['trace_summary']:
                        raise ValueError('The sham fallback differs from its full matched sham trace')
                    if [w['spikes'] for w in sham['windows']] != [w['spikes'] for w in active['windows']]:
                        raise ValueError('The sham fallback differs from its matched spike record')
                structures[seed] = sham['structure']['global_sha256']
                row = contrast(seed, sham, active, target)
                row.update(arm=arm, peak_v_per_m=active['window_protocol']['max_field_v_per_m'],
                           baseline_rates_safe=decisions[1]['baseline_rates_safe'], delivered_arm=decisions[1]['delivered_arm'])
                if stage == 'discovery':
                    row['late_benefit'] = (s0.scores(sham_psd['late'], late['target'])['distance'] -
                                           s0.scores(active_psd['late'], late['target'])['distance'])
                    if row['eligible']:
                        row['timing'] = timing_contrast(seed, sham_spikes, active_spikes)
                rows.append(row)
            except (ValueError, KeyError) as exc:
                errors.append(f"{job['name']} pairing: {exc}")
        if len(set(structures.values())) != len(structures):
            errors.append('Discovered repeated structure fingerprints across seeds')
        if stage == 'discovery':
            qualification = p.load_upper_gate(root / p.UPPER_GATE)
            if set(structures.values()) & set(qualification['structure_sha256'].values()):
                errors.append('A discovery structure repeats the qualification structure')

    if not errors:
        result.update(structure_sha256={str(k): v for k, v in structures.items()},
                      actual_ksu=sum(row['service_units'] for row in accounting) / 1000)
        if stage == 'qualification':
            row = rows[0]
            checks = {'matched_pre_action': True, 'upper_dose_delivered': row['delivered_arm'] == 'dose_08',
                      'baseline_rate_guards': row['baseline_rates_safe'], 'paired_rate_guards': row['rate_safe']}
            result.update(qualification_checks=checks, peak_v_per_m=row['peak_v_per_m'], rates=row['rates'],
                          status='qualified' if all(checks.values()) else 'qualification_failed')
            if write:
                frozen_json(root / p.UPPER_GATE, reference.signed(result))
        else:
            result.update(status='discovery_complete', rows=rows, **summarize_discovery(rows))
            if write:
                frozen_json(root / 'dose_response_summary.json', reference.signed(result))
    if write:
        save_json(root / 'dose_response_status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', nargs='?', type=Path, help='Explicit suite directory for transferred results')
    parser.add_argument('--stage', choices=p.STAGES, default='qualification')
    parser.add_argument('--verify-only', action='store_true', help='Audit without writing or creating a qualification gate')
    args = parser.parse_args()
    suite = (args.suite or p.suite_path(args.stage)).resolve()
    result = analyze_suite(suite, write=not args.verify_only)
    fields = ('status', 'completed_valid_runs', 'expected_runs', 'errors', 'actual_ksu',
              'qualification_checks', 'eligible_seeds', 'candidate_for_independent_confirmation')
    overview = {key: result[key] for key in fields if key in result}
    if 'arms' in result:
        overview['plateau_by_dose'] = {
            arm: {'v_per_m': p.AMPLITUDES[arm], 'mean_benefit': values['benefit'].get('mean'),
                  'positive_seeds': values['benefit'].get('positive_structures'),
                  'p_holm': values['benefit'].get('p_holm'), 'candidate': values['candidate']}
            for arm, values in result['arms'].items()
        }
    print(json.dumps(overview, indent=2))
    print(f'SUITE_DIRECTORY={suite}')
    return 0 if result['status'] in ('qualified', 'discovery_complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
