"""Offline full-spectrum S0 qualification of the frozen R1 recordings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.r1_protocol import COHORTS, save_json, sha256
from experiments.l23net_analysis.r1_analysis import paired_inference, bh_adjust
from experiments.l23net_analysis.replay_validation import canonical_json_sha256, trace_content_summary


def suite_runs(root):
    manifest = json.loads((root/'submission.json').read_text())
    summary = json.loads((root/'r1_summary.json').read_text())
    if not summary['technical_passed'] or summary['errors']:
        raise ValueError('Source R1 suite did not pass')
    indexed = {(r['seed'], r['condition']): r for r in summary['rows']}
    for job in manifest['jobs']:
        if (root/job['name']/'worker_exit_code.txt').read_text().strip() != '0':
            raise ValueError('Incomplete worker')
        for run in job['runs']:
            directory = root/job['name']/f"seed_{run['seed']}"/run['condition']
            report_path = directory/'l23net_r1_run.json'
            trace_path = directory/'l23net_r1_trace.h5'
            row = indexed[run['seed'], run['condition']]
            if sha256(report_path) != row['report_sha256']:
                raise ValueError('Source report changed')
            content = trace_content_summary(trace_path)
            if content['content_sha256'] != row['trace_content_sha256'] or content['committed_samples'] != 1120000:
                raise ValueError('Incomplete/changed source trace')
            with h5py.File(trace_path) as trace:
                if np.count_nonzero(trace['field_left_boundary_v_per_m'][:]):
                    raise ValueError('Source is stimulated')
                t, y = s0.CausalEEG(.025).append(trace['eeg_v'][0])
            yield row, t, y


def observations(row, t, y):
    x = s0.epoch(t, y, 8, 28)
    full = s0.log_powers(x)
    duration_error = {}
    # Shared-data descriptive discrepancies, not independent variance estimates.
    for length in [4, 8, 12]:
        errors = [s0.log_powers(s0.epoch(t, y, start, start+length))-full
                  for start in np.arange(8, 28-length+.01, 2)]
        duration_error[str(length)] = np.sqrt(np.mean(np.asarray(errors)**2, axis=0)).tolist()
    halves = [s0.log_powers(s0.epoch(t, y, start, start+10)) for start in [8, 18]]
    return {'seed': row['seed'], 'condition': row['condition'], 'cohort': row['cohort'],
            'source_trace_sha256': row['trace_content_sha256'],
            'log_powers': full.tolist(), 'half_log_ratio': (halves[1]-halves[0]).tolist(),
            'window_discrepancy_log10': duration_error,
            'carriers': {b: s0.stable_carrier(x, b) for b in s0.BANDS},
            'phase_audit': {b: s0.phase_audit(t, y, b) for b in s0.BANDS}}


def infer(rows, target):
    index = {(r['seed'], r['condition']): r for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    ref = np.array([index[s, 'reference']['log_powers'] for s in seeds])
    mdd = np.array([index[s, 'mdd']['log_powers'] for s in seeds])
    delta = mdd-ref
    composite = np.mean(delta/target['scale_log10'], axis=1)
    result = {'n': len(seeds), 'composite': paired_inference(composite),
              'bands': {b: paired_inference(delta[:, i]) for i, b in enumerate(s0.BANDS)},
              'sensitivity': float(np.mean([s0.scores(x, target)['eligible'] for x in mdd])),
              'specificity': float(np.mean([not s0.scores(x, target)['eligible'] for x in ref]))}
    for band,q in zip(s0.BANDS,bh_adjust([result['bands'][b]['p_one_sided'] for b in s0.BANDS])):
        result['bands'][band]['q_bh_three_bands'] = float(q)
    result['phenotype_passed'] = bool(result['composite']['mean_t_ci95'][0] > 0 and
        all(v['mean_t_ci95'][0] > 0 for v in result['bands'].values()) and
        result['sensitivity'] >= .8 and result['specificity'] >= .8)
    result['measurement'] = {}
    for band in s0.BANDS:
        mdd_rows = [r for r in rows if r['condition'] == 'mdd']
        coverage = float(np.mean([r['carriers'][band]['accepted'] for r in mdd_rows]))
        horizons = {}
        for horizon in s0.PROTOCOL['phase_horizons_s']:
            structure_errors = []
            accepted_count = 0
            for r in mdd_rows:
                errors = [abs(p['error_rad']) for p in r['phase_audit'][band]
                          if p.get('horizon_s') == horizon and p['accepted']]
                accepted_count += len(errors)
                if errors:
                    structure_errors.append(float(np.mean(errors)))
            horizons[str(horizon)] = {
                'accepted_predictions': accepted_count,
                'possible_predictions': len(mdd_rows)*len(s0.PROTOCOL['phase_audit_boundaries_s']),
                'structure_mean_absolute_error_rad': float(np.mean(structure_errors)) if structure_errors else None,
                'scored_structures': len(structure_errors),
                'uniform_phase_null_mae_rad': float(np.pi/2)}
        result['measurement'][band] = {'stable_peak_coverage': coverage,
            'frequency_passed': coverage >= s0.PROTOCOL['frequency_minimum_coverage'],
            'phase_horizons': horizons,
            'long_block_phase_passed': False,
            'phase_reason': '28-s sources cannot qualify 22-s open-loop phase prediction after the required baseline.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', required=True, type=Path)
    parser.add_argument('--extension', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Write the complete contract BEFORE reading any new measurement outcomes.
    save_json(args.output/'protocol_before_analysis.json', s0.PROTOCOL)
    manifest = json.loads((args.extension/'submission.json').read_text())
    for name, digest in manifest['core_hashes'].items():
        if sha256(args.core/name) != digest:
            raise ValueError('Frozen upstream core changed: '+name)
    rows, processed, spectra = [], {}, {}
    calibration = []
    for raw, t, y in suite_runs(args.core):
        row = observations(raw, t, y)
        rows.append(row)
        processed[f"s{row['seed']}_{row['condition']}"] = y
        f, psd = s0.spectrum(s0.epoch(t, y, 8, 28))
        spectra[f"s{row['seed']}_{row['condition']}"] = psd
        if row['cohort'] == 'calibration':
            calibration.append((row, t, y))
    if sorted(r['seed'] for r, t, y in calibration) != list(COHORTS['calibration']):
        raise ValueError('Wrong calibration cohort')
    target = s0.fit_target([r['log_powers'] for r, t, y in calibration])
    target10 = s0.fit_target([s0.log_powers(s0.epoch(t, y, 18, 28)) for r, t, y in calibration])
    excluded = {band: s0.fit_target([s0.log_powers(s0.epoch(t,y,8,28), (freq-.5,freq+.5))
                                     for r,t,y in calibration]) for band,freq in s0.FIXED.items()}
    targets = {'20s': target, '10s': target10, 'fundamental_excluded': excluded}
    save_json(args.output/'targets_frozen_before_extension.json', targets)
    discovery = infer([r for r in rows if r['cohort'] == 'primary'], target)
    save_json(args.output/'core_discovery_before_extension.json', discovery)
    print('Frozen core targets and measurement contract; now evaluating extension.', flush=True)
    for raw, t, y in suite_runs(args.extension):
        row = observations(raw, t, y)
        rows.append(row)
        processed[f"s{row['seed']}_{row['condition']}"] = y
        f, psd = s0.spectrum(s0.epoch(t, y, 8, 28))
        spectra[f"s{row['seed']}_{row['condition']}"] = psd
        print(f"S0 {row['seed']} {row['condition']}", flush=True)
    extension = [r for r in rows if r['cohort'] == 'extension']
    if len(extension) != 88 or sorted({r['seed'] for r in extension}) != list(COHORTS['extension']):
        raise ValueError('Incomplete extension')
    validation = infer(extension, target)
    phenotype = discovery['phenotype_passed'] and validation['phenotype_passed']
    bundle = {'protocol': s0.PROTOCOL, 'technical_passed': True,
              'reference_contract': json.loads((args.core/'core_data_frozen.json').read_text())['contract'],
              'gates': {'full_spectrum_phenotype': phenotype,
                        'fixed_frequency_discovery_allowed': phenotype,
                        'phase_matched_discovery_allowed': False},
              'targets': targets, 'core_discovery': discovery, 'extension_validation': validation,
              'allowed_arms': ['sham', 'fixed_theta', 'fixed_alpha', 'fixed_low_beta', 'transverse_alpha'] if phenotype else ['sham'],
              'source_hashes': {str(p): sha256(p) for p in [args.core/'r1_summary.json', args.extension/'r1_summary.json']},
              'analysis_code_sha256': {str(Path(p).relative_to(ROOT)): sha256(p) for p in [Path(__file__), Path(s0.__file__)]},
              'limitations': ['No latent neural phase ground truth; future sinusoid fit is a scoring convention only.',
                             'Longer active/sham and reference stationarity still require prospective checks.',
                             'Targets are matched in duration, not in absolute epoch; the late 60-s qualification reference audits transfer.',
                             'Core R1 phenotype outcomes were previously inspected; S0 is measurement qualification, not new blinded disease confirmation.',
                             'Short-window discrepancies share data with the 20-s estimate and are not independent noise variance.',
                             'No calibration/selection on active stimulation outcomes.']}
    bundle['sha256'] = canonical_json_sha256(bundle)
    save_json(args.output/'s0_gate.json', bundle)
    save_json(args.output/'s0_observations.json', {'rows': rows})
    np.savez_compressed(args.output/'processed_eeg.npz', time_s=t, **processed)
    np.savez_compressed(args.output/'spectra.npz', frequency_hz=f, **spectra)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11,4), layout='constrained')
    for condition, color in [('reference','#286792'),('mdd','#b65836')]:
        group = [r for r in extension if r['condition']==condition]
        values = np.array([spectra[f"s{r['seed']}_{condition}"] for r in group])
        axes[0].plot(f, values.mean(axis=0), color=color, label=condition)
        axes[0].fill_between(f,np.quantile(values,.25,axis=0),np.quantile(values,.75,axis=0),color=color,alpha=.15)
    axes[0].set(xlim=(2,20), yscale='log', xlabel='Frequency (Hz)', ylabel='EEG PSD (V²/Hz)', title='44-pair extension: mean and interquartile range')
    axes[0].set_ylim(2e-21, 2e-19)
    axes[0].legend(frameon=False)
    for i, band in enumerate(s0.BANDS):
        vals = [np.mean([r['window_discrepancy_log10'][str(n)][i] for r in extension]) for n in [4,8,12]]
        axes[1].plot([4,8,12],vals,'o-',label=band)
    axes[1].set(xlabel='Analysis duration (s)', ylabel='RMS log10 discrepancy from same-trace 20-s power',
                title='Descriptive window sensitivity; not independent error')
    axes[1].legend(frameon=False)
    fig.savefig(args.output/'s0_spectra.png',dpi=170)
    plt.close(fig)
    print(json.dumps({'gates': bundle['gates'], 'extension': validation},indent=2))


if __name__ == '__main__':
    main()
