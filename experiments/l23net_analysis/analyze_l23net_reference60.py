"""Audit all 16 Healthy continuations before freezing absolute-epoch targets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from experiments.l23net_analysis import reference60_protocol as p
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis import s1_analysis as s1
from experiments.l23net_analysis.r1_protocol import save_json, sha256
from experiments.l23net_analysis.r1_analysis import frozen_json, paired_inference, bh_adjust, pbs_epilogue


def fit_targets(rows, gate):
    if sorted(r['seed'] for r in rows) != list(p.SEEDS):
        raise ValueError('All 16 unique Healthy calibration seeds are required; no subset fitting')
    if any(r['condition'] != 'reference' for r in rows):
        raise ValueError('Only Healthy data may fit reference targets')
    targets = {'baseline_screen_unchanged': gate['targets']['20s']}
    for name in ('plateau', 'washout'):
        data = np.asarray([r['log_powers'][name] for r in rows])
        if data.shape != (16, 3) or not np.isfinite(data).all():
            raise ValueError('Invalid target features')
        targets[name] = {'interval_s': p.EPOCHS[name], **s0.fit_target(data)}
    targets['fundamental_excluded_plateau'] = {
        band: {'interval_s': p.EPOCHS['plateau'], 'excluded_hz': [f-.5, f+.5],
               **s0.fit_target([r['excluded_plateau'][band] for r in rows])}
        for band, f in s0.FIXED.items()}
    # Drift is a diagnostic, not a gate that selects favorable seeds or epochs.
    drift = {}
    for epoch in ('plateau', 'washout'):
        baseline = 'baseline' if epoch == 'plateau' else 'baseline_10s'
        delta = np.array([r['log_powers'][epoch] for r in rows])-np.array([r['log_powers'][baseline] for r in rows])
        for i, band in enumerate(s0.BANDS):
            item = paired_inference(delta[:, i])
            reverse = paired_inference(-delta[:, i])
            item['p_two_sided'] = min(1., 2*min(item['p_one_sided'], reverse['p_one_sided']))
            item.pop('p_one_sided')
            item['geometric_mean_power_change_percent'] = float(100*np.expm1(np.log(10)*item['mean']))
            item['contrast'] = epoch+' minus '+baseline+' log10 power (duration matched)'
            item['uncertainty_scope'] = 'Paired variability across 16 model structures; not biological subjects.'
            drift[f'{epoch}_{band}'] = item
    for item, q in zip(drift.values(), bh_adjust([r['p_two_sided'] for r in drift.values()])):
        item['q_bh_six_drift_tests'] = q
    return targets, drift


def analyze_suite(root):
    root = Path(root)
    manifest = json.loads((root/'submission.json').read_text())
    gate = s0.load_gate(root/'s0_gate.json')
    sources = p.load_signed(root/'reference_sources.json')
    errors, rows, accounting = [], [], []
    if (manifest.get('stage') != 'reference60' or manifest.get('protocol') != p.PROTOCOL or
            manifest.get('code_sha256') != p.code_hashes() or manifest.get('s0_sha256') != gate['sha256']):
        raise ValueError('Submission protocol/code/S0 identity differs')
    if (sources['sha256'] != manifest['reference_sources_sha256'] or sources['s0_sha256'] != gate['sha256'] or
            sorted(s['seed'] for s in sources['sources']) != list(p.SEEDS)):
        raise ValueError('Source identity differs from submission')
    s1.load_qualification(root/'qualification.json', gate['sha256'])
    specifications = [{k: j[k] for k in ('name', 'run')} for j in manifest['jobs']]
    if specifications != p.jobs() or manifest.get('submission_status') != 'submitted':
        errors.append('Incomplete, duplicate or unexpected submission; no target will be frozen')
    by_seed = {s['seed']: s for s in sources['sources']}
    for job in manifest['jobs']:
        directory = root/job['name']
        try:
            if (directory/'worker_exit_code.txt').read_text().strip() != '0':
                raise ValueError('Worker failed')
            if (directory/'git_commit.txt').read_text().strip() != manifest['commit']:
                raise ValueError('Worker checkout differs from submission')
            report = s1.audit_run(directory, gate)
            cfg = report['configuration']
            if (report['seed_manifest']['experiment_seed'] != job['run']['seed'] or
                    report['replay_contract']['condition'] != 'reference' or
                    report['replay_contract']['simulation']['MDD'] or report['replay_contract']['simulation']['DRUG'] or
                    cfg['analysis']['mode'] != 'reference_calibration' or cfg['analysis']['arm'] != 'sham'):
                raise ValueError('Run is not the intended Healthy calibration')
            meta = report['window_protocol']['reference60']
            if meta['code_sha256'] != p.code_hashes() or meta['protocol_sha256'] != p.canonical_json_sha256(p.PROTOCOL):
                raise ValueError('Calibration implementation/protocol changed')
            if report['window_protocol']['max_field_v_per_m'] != 0 or report['window_protocol']['nonzero_field_samples'] != 0:
                raise ValueError('Reference field was not exactly zero')
            trace = directory/'l23net_s1_trace.h5'
            p.check_prefix(report, trace, by_seed[job['run']['seed']])
            pbs = pbs_epilogue(directory/'pbs.out')
            accounting.append({'job': job['name'], 'project': job['project'], **pbs})
            if not pbs.get('final_accounting_available'):
                raise ValueError('Final PBS accounting missing; rerun after all job files arrive')
            if pbs['exit_status'] != 0 or pbs['memory_gb_pbs'] > 230.4:
                raise ValueError('PBS failure or less than 10 percent memory headroom')
            memory = [s['rss_gib']['sum'] for s in report['memory_snapshots'] if s['simulated_ms'] >= 8000]
            if not memory or max(memory)-min(memory) > max(2., .05*memory[0]):
                raise ValueError('Persistent memory growth exceeds the qualified S1 tolerance')
            rates = {name: s1.mean_rates(report, a*1000, b*1000) for name, (a,b) in p.EPOCHS.items()}
            if any(not np.isfinite(v) or not 0 <= v <= (20 if 'PYR' in pop else 100)
                   for values in rates.values() for pop, v in values.items()):
                raise ValueError('Unsafe or nonfinite reference population firing rates')
            with h5py.File(trace) as h:
                if np.count_nonzero(h['field_left_boundary_v_per_m'][:]):
                    raise ValueError('Raw reference field is not zero')
                t, y = s0.CausalEEG(.025).append(h['eeg_v'][0])
            powers = {name: s0.log_powers(s0.epoch(t,y,a,b)).tolist() for name,(a,b) in p.EPOCHS.items()}
            excluded = {band: s0.log_powers(s0.epoch(t,y,29,49),(f-.5,f+.5)).tolist() for band,f in s0.FIXED.items()}
            rows.append({'seed': job['run']['seed'], 'condition': 'reference', 'prefix_exact': True,
                         'report_sha256': sha256(directory/'l23net_s1_run.json'),
                         'trace_content_sha256': report['artifacts']['trace_summary']['content_sha256'],
                         'log_powers': powers, 'excluded_plateau': excluded, 'rates_hz': rates})
        except Exception as exc:
            errors.append(f"{job['name']}: {exc}")
    result = {'stage': 'reference60', 'status': 'incomplete_or_failed', 'errors': errors,
              'completed_valid_runs': len(rows), 'expected_runs': 16, 'rows': rows, 'pbs': accounting,
              's0_sha256': gate['sha256'], 'code_sha256': p.code_hashes()}
    if not errors:
        targets, drift = fit_targets(rows, gate)
        target = p.signed({'protocol': p.PROTOCOL, 'targets': targets,
                           's0_sha256': gate['sha256'], 'code_sha256': p.code_hashes(),
                           'reference_sources_sha256': sources['sha256'],
                           'sources': [{k: r[k] for k in ('seed','report_sha256','trace_content_sha256')} for r in rows],
                           'status': 'reference_calibrated',
                           'limitations': ['Not evidence for tACS efficacy; use disjoint held-out efficacy seeds.',
                                           'Preserve S1 discovery results. Do not rerank its arms with the new target.',
                                           'Reference scales are estimated from 16 model structures.',
                                           'Baseline screening remains the original S0 target.']})
        frozen_json(root/'reference60_target.json', target)
        result.update(status='reference_calibrated', target_sha256=target['sha256'], paired_drift=drift,
                      actual_ksu=sum(row['service_units'] for row in accounting)/1000)
        frozen_json(root/'reference60_summary.json', p.signed(result))
    # A mutable status report is safe to re-run while jobs are incomplete.
    save_json(root/'reference60_status.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path, nargs='?')
    parser.add_argument('--latest', action='store_true')
    args = parser.parse_args(argv)
    if args.latest:
        if args.suite:
            parser.error('Use a suite path OR --latest')
        paths = sorted(p for p in (ROOT/'results').glob('l23net_reference60_*') if (p/'submission.json').is_file())
        if not paths:
            parser.error('No reference60 suite in results; give an explicit transferred suite path')
        args.suite = paths[-1]
    if args.suite is None:
        parser.error('Supply a suite path or --latest')
    result = analyze_suite(args.suite)
    print(json.dumps({k: result[k] for k in ('status','completed_valid_runs','expected_runs','errors')}, indent=2))
    print('SUITE_DIRECTORY='+str(args.suite.resolve()))
    return 0 if result['status'] == 'reference_calibrated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
