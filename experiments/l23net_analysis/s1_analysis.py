"""Raw-trace S1 auditing and fail-closed 60-s qualification."""
import json
from pathlib import Path

import h5py
import numpy as np
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.replay_validation import canonical_json_sha256, trace_content_summary
from experiments.l23net_analysis.r1_protocol import sha256, save_json

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = ['experiments/l23net_analysis/run_l23net_s1.py',
              'experiments/l23net_analysis/s1_analysis.py',
              'experiments/l23net_analysis/s0_protocol.py',
              'experiments/l23net_analysis/run_l23net_no_field_replay.py',
              'experiments/l23net_analysis/run_l23net_g1b.py',
              'env/models/neuron/env_online.py', 'env/models/neuron/networkenv_online.py',
              'env/models/neuron/stimulation.py', 'env/models/neuron/streaming.py',
              'env/models/neuron/extracellular_online.py']


def code_hashes():
    return {name: sha256(ROOT/name) for name in CODE_FILES}


def population_rates(values):
    """The environment's firing_rates mapping ALSO contains spike counts."""
    rates={p.removesuffix('_firing_rate_hz'):float(v) for p,v in values.items() if p.endswith('_firing_rate_hz')}
    if not rates:
        raise ValueError('Missing population firing-rate fields')
    return rates


def latest_suite(stage):
    if stage not in ('qualification','discovery'):
        raise ValueError('Unknown stage')
    candidates=sorted(p for p in (ROOT/'results').glob(f'l23net_s1_{stage}_*') if (p/'submission.json').is_file())
    if not candidates:
        raise FileNotFoundError('No '+stage+' S1 suite exists in this checkout')
    return candidates[-1]


def load_qualification(path, s0_hash):
    report = json.loads(Path(path).read_text())
    value = dict(report)
    digest = value.pop('sha256')
    if canonical_json_sha256(value) != digest or report.get('status') != 'qualified':
        raise ValueError('60-s qualification is missing, changed or failed')
    if report['s0_sha256'] != s0_hash or report['code_sha256'] != code_hashes():
        raise ValueError('Qualification S0/code identity differs; requalify before discovery')
    return report


def audit_run(directory, gate, debug=False):
    directory = Path(directory)
    report = json.loads((directory/'l23net_s1_run.json').read_text())
    if report['status'] != 'passed' or report['errors'] or report['build_audit']['errors']:
        raise ValueError('Trajectory did not pass its technical checks')
    contract = report['replay_contract']
    if canonical_json_sha256(contract) != report['replay_contract_sha256']:
        raise ValueError('Contract hash mismatch')
    content = trace_content_summary(directory/'l23net_s1_trace.h5')
    if content != report['artifacts']['trace_summary']:
        raise ValueError('Raw trace hash/extent mismatch')
    duration = contract['simulation']['duration_ms']
    dt = contract['network']['dt_ms']
    if not debug or 'spike_file_sha256' in report['window_protocol']:
        spike_path = directory/'l23net_s1_spikes.h5'
        if sha256(spike_path) != report['window_protocol']['spike_file_sha256']:
            raise ValueError('Sparse spike artifact changed')
        with h5py.File(spike_path) as spike_file:
            if spike_file.attrs['committed_windows'] != len(report['windows']):
                raise ValueError('Sparse spikes incomplete')
            for population in spike_file:
                times=spike_file[population]['times_ms'][:]
                if not np.isfinite(times).all() or len(times)!=len(spike_file[population]['gids']):
                    raise ValueError('Invalid sparse spikes')
                if len(times)!=sum(w['spikes']['counts'][population] for w in report['windows']):
                    raise ValueError('Sparse spike count mismatch')
    if not debug:
        if contract['debug'] or contract['mpi_ranks'] != 624 or duration != 60000 or dt != .025:
            raise ValueError('Not the full 60-s / 624-rank scientific protocol')
        if report['window_protocol']['s0_sha256'] != gate['sha256']:
            raise ValueError('Different S0 target')
        if report['window_protocol']['code_sha256'] != code_hashes():
            raise ValueError('Source implementation changed since simulation')
    if len(report['windows']) != round(duration/contract['simulation']['window_ms']):
        raise ValueError('Incomplete windows')
    with h5py.File(directory/'l23net_s1_trace.h5') as h:
        count = round(duration/dt)
        if content['committed_samples'] != count or h['eeg_v'].shape != (1,count):
            raise ValueError('Incomplete saved EEG')
        for key in ('sample_time_ms','eeg_v','dipole_nA_um','field_left_boundary_v_per_m'):
            if not np.isfinite(h[key][:]).all():
                raise ValueError('Nonfinite saved '+key)
        if not np.allclose(h['sample_time_ms'][:],(np.arange(count)+1)*dt,rtol=0,atol=1e-7):
            raise ValueError('Wrong time grid')
        cfg = report['configuration']['analysis']
        start, stop, ramp = [cfg[k] for k in ('stim_start_ms','stim_stop_ms','ramp_ms')]
        field = h['field_left_boundary_v_per_m'][:]
        ft = h['field_left_boundary_time_ms'][:]
        d = report['window_protocol']['decision']
        from env.models.neuron.stimulation import apply_raised_cosine_block_envelope
        expected = d['amplitude_v_per_m']*np.sin(2*np.pi*d['frequency_hz']*(ft-start)/1000)
        expected = apply_raised_cosine_block_envelope(expected,time_ms=ft,block_start_ms=start,block_stop_ms=stop,ramp_ms=ramp)
        if not np.allclose(field,expected,rtol=0,atol=1e-10):
            raise ValueError('Saved field is not the intended waveform')
        if np.count_nonzero(field[(ft<start)|(ft>=stop)]):
            raise ValueError('Field outside stimulation epoch')
        if d['amplitude_v_per_m'] and np.max(np.abs(field)) < .39:
            raise ValueError('Active field never reached its prescribed amplitude')
        if d['amplitude_v_per_m'] and not report['window_protocol']['tissue_coupling_observed']:
            raise ValueError('Active waveform never produced nonzero extracellular coupling')
        direction = [1.,0.,0.] if cfg['arm']=='transverse_alpha' else [0.,0.,1.]
        for window in report['windows']:
            if start <= window['start_ms'] < stop and window['field_direction'] != direction:
                raise ValueError('Active-window field direction differs from the intended montage')
        if not debug:
            t,y = s0.CausalEEG(dt).append(h['eeg_v'][0])
            for name,a,b,target in [('baseline',8,28,'20s'),('stimulation',29,49,'20s'),('washout',50,60,'10s')]:
                powers = s0.log_powers(s0.epoch(t,y,a,b))
                saved = report['window_protocol']['outcomes'][name]
                if not np.allclose(powers,saved['log_powers'],rtol=0,atol=1e-9):
                    raise ValueError('Saved outcome fails raw-data recomputation')
                for key,value in s0.scores(powers,gate['targets'][target]).items():
                    if not np.allclose(value,saved[key],rtol=0,atol=1e-9):
                        raise ValueError('Saved target score fails raw-data recomputation')
            for band,freq in s0.FIXED.items():
                powers = s0.log_powers(s0.epoch(t,y,29,49),(freq-.5,freq+.5))
                saved = report['window_protocol']['fundamental_excluded'][band]
                if not np.allclose(powers,saved['log_powers'],rtol=0,atol=1e-9):
                    raise ValueError('Excluded-band power fails raw-data recomputation')
                for key,value in s0.scores(powers,gate['targets']['fundamental_excluded'][band]).items():
                    if not np.allclose(value,saved[key],rtol=0,atol=1e-9):
                        raise ValueError('Excluded-band target score fails raw-data recomputation')
    return report


def compare_prefix(first, second, a, b):
    if a['seed_manifest'] != b['seed_manifest'] or a['structure'] != b['structure']:
        raise ValueError('Counterfactual structure/random seed mismatch')
    if a['build_audit']['invariant_sha256'] != b['build_audit']['invariant_sha256']:
        raise ValueError('Counterfactual construction mismatch')
    if a['window_protocol']['prestimulation_eeg_dipole_sha256'] != b['window_protocol']['prestimulation_eeg_dipole_sha256']:
        raise ValueError('Counterfactual baseline fingerprint mismatch')
    stop = a['configuration']['analysis']['stim_start_ms']
    cutoff = round(stop/a['replay_contract']['network']['dt_ms'])
    with h5py.File(Path(first)/'l23net_s1_trace.h5') as x, h5py.File(Path(second)/'l23net_s1_trace.h5') as y:
        for name in ('eeg_v','dipole_nA_um'):
            if not np.array_equal(x[name][...,:cutoff],y[name][...,:cutoff]):
                raise ValueError('Raw counterfactual prestimulation traces differ')
    wa = [w['spikes'] for w in a['windows'] if w['stop_ms'] <= stop]
    wb = [w['spikes'] for w in b['windows'] if w['stop_ms'] <= stop]
    if wa != wb:
        raise ValueError('Counterfactual baseline spikes differ')


def mean_rates(report, start=29000, stop=49000):
    windows = [w for w in report['windows'] if w['start_ms'] >= start and w['stop_ms'] <= stop]
    if not windows:
        raise ValueError('Missing rate interval')
    rates=[population_rates(w['firing_rates']) for w in windows]
    return {p: float(np.mean([r[p] for r in rates])) for p in rates[0]}


def paired_contrast(seed, arm, sham, active):
    band = 'alpha' if arm=='transverse_alpha' else arm.removeprefix('fixed_')
    a,b = sham['window_protocol'],active['window_protocol']
    ra,rb = mean_rates(sham),mean_rates(active)
    safe = all(abs(rb[p]-ra[p])<=.2*max(ra[p],.1) for p in ra)
    return {'seed':seed,'arm':arm,'delivered':b['decision']['delivered_arm'],
            'benefit':a['outcomes']['stimulation']['distance']-b['outcomes']['stimulation']['distance'],
            'excluded_benefit':a['fundamental_excluded'][band]['distance']-b['fundamental_excluded'][band]['distance'],
            'band_log10_active_minus_sham':(np.array(b['outcomes']['stimulation']['log_powers'])-
                                          a['outcomes']['stimulation']['log_powers']).tolist(),
            'paired_rate_safe':safe,'rates_sham':ra,'rates_active':rb,
            'washout_distance_difference':a['outcomes']['washout']['distance']-b['outcomes']['washout']['distance']}


def analyze_suite(root):
    root = Path(root)
    manifest = json.loads((root/'submission.json').read_text())
    gate = s0.load_gate(root/'s0_gate.json')
    errors, reports, paths, accounting = [], {}, {}, []
    if manifest['stage']=='qualification':
        expected={(8451,'reference','sham'),(8451,'mdd','sham'),(8451,'mdd','fixed_alpha')}
    elif manifest['stage']=='discovery':
        expected={(seed,'mdd',arm) for seed in s0.PROTOCOL['discovery_seeds']
                  for arm in ('sham','fixed_theta','fixed_alpha','fixed_low_beta')}
        expected.add((8401,'mdd','transverse_alpha'))
    else:
        raise ValueError('Unknown submission stage')
    actual=[(j['run']['seed'],j['run']['condition'],j['run']['arm']) for j in manifest['jobs']]
    if len(actual)!=len(expected) or set(actual)!=expected:
        errors.append('Missing, duplicate or unplanned trajectories in submission manifest')
    from experiments.l23net_analysis.r1_analysis import pbs_epilogue
    for job in manifest['jobs']:
        folder = root/job['name']
        try:
            if (folder/'worker_exit_code.txt').read_text().strip() != '0':
                raise ValueError('Worker incomplete or failed')
            if (folder/'git_commit.txt').read_text().strip() != manifest['commit']:
                raise ValueError('Worker checkout differs from submission')
            report = audit_run(folder,gate)
            spec = job['run']
            if (report['seed_manifest']['experiment_seed'],report['replay_contract']['condition'],
                report['configuration']['analysis']['arm'],report['configuration']['analysis']['mode']) != (
                    spec['seed'],spec['condition'],spec['arm'],manifest['stage']):
                raise ValueError('Run does not match submission')
            reports[job['name']],paths[job['name']] = report, folder
            pbs = pbs_epilogue(folder/'pbs.out')
            accounting.append({'job':job['name'],**pbs})
            if not pbs.get('final_accounting_available'):
                raise ValueError('Final PBS epilogue missing; rerun analysis after final files arrive')
            if pbs['exit_status'] != 0 or pbs['memory_gb_pbs'] > 230.4:
                raise ValueError('PBS failure or <10% memory headroom; requalify resources')
        except Exception as exc:
            errors.append(f"{job['name']}: {exc}")
    result = {'stage':manifest['stage'],'status':'failed','errors':errors,'s0_sha256':gate['sha256'],
              'code_sha256':code_hashes(),'pbs':accounting,
              'run_report_hashes':{n:sha256(p/'l23net_s1_run.json') for n,p in paths.items()}}
    if not errors and manifest['stage']=='qualification':
        try:
            if set(reports) != {'q_reference_sham','q_mdd_sham','q_mdd_alpha'}:
                raise ValueError('Qualification requires all three frozen runs')
            ref, sham, active = [reports[n] for n in ('q_reference_sham','q_mdd_sham','q_mdd_alpha')]
            from experiments.l23net_analysis.g1b_analysis import pairing_errors
            if pairing_errors(ref,sham):
                raise ValueError('Reference/MDD qualification does not preserve intended paired construction')
            compare_prefix(paths['q_mdd_sham'],paths['q_mdd_alpha'],sham,active)
            if active['window_protocol']['decision']['amplitude_v_per_m'] != .4:
                raise ValueError('Qualification active arm did not deliver 0.4 V/m')
            if not active['window_protocol']['tissue_coupling_observed']:
                raise ValueError('No nonzero extracellular potential was observed')
            rates_sham, rates_active = mean_rates(sham), mean_rates(active)
            result['rates_sham_hz'],result['rates_active_hz']=rates_sham,rates_active
            if any(abs(rates_active[p]-rates_sham[p])>.2*max(rates_sham[p],.1) for p in rates_sham):
                errors.append('Active/sham population-rate change exceeds 20% guardrail')
            # Protect duration-matched target transfer without retuning it.
            drift = {label:(np.array(r['window_protocol']['outcomes']['stimulation']['log_powers'])-
                           r['window_protocol']['outcomes']['baseline']['log_powers']).tolist()
                     for label,r in [('reference',ref),('mdd_sham',sham)]}
            result['late_epoch_log10_drift'] = drift
            if any(np.any(np.abs(v)>np.log10(1.5)) for v in drift.values()):
                errors.append('Late no-field band-power drift exceeds predeclared 1.5-fold engineering tolerance; extend/calibrate before discovery')
            for name,r in reports.items():
                mem=[s['rss_gib']['sum'] for s in r['memory_snapshots'] if s['simulated_ms']>=8000]
                if max(mem)-min(mem)>max(2.,.05*mem[0]):
                    errors.append(f'{name}: persistent RSS growth exceeds engineering tolerance')
            result['status'] = 'qualified' if not errors else 'failed'
        except Exception as exc:
            errors.append(str(exc))
    elif not errors:
        try:
            contrasts=[]
            for seed in s0.PROTOCOL['discovery_seeds']:
                sham_name=f's{seed}_sham'
                sham=reports[sham_name]
                for arm in ['fixed_theta','fixed_alpha','fixed_low_beta']:
                    name=f's{seed}_{arm}'
                    active=reports[name]
                    compare_prefix(paths[sham_name],paths[name],sham,active)
                    contrasts.append(paired_contrast(seed,arm,sham,active))
            result['contrasts']=contrasts
            sham_name,orientation_name='s8401_sham','s8401_transverse_alpha'
            compare_prefix(paths[sham_name],paths[orientation_name],reports[sham_name],reports[orientation_name])
            result['orientation_control']=paired_contrast(8401,'transverse_alpha',reports[sham_name],reports[orientation_name])
            result['status']='completed_discovery'
            result['interpretation']='Three structures are discovery only. No automatic S2/bandit permission; inspect all bands, washout, safety, orientation, and excluded-space effects.'
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        result['status']='failed'
    result['sha256']=canonical_json_sha256(result)
    save_json(root/'s1_summary.json',result)
    return result
