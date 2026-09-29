"""Gated 60-s, full-spectrum L23Net field discovery; never a phase claim."""
import hashlib
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import hydra
import numpy as np
from omegaconf import OmegaConf
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.run_l23net_no_field_replay import run_replay, _validate_configuration, _replay_contract
from experiments.l23net_analysis.run_l23net_g1b import construction_audit
from env.models.neuron.stimulation import apply_raised_cosine_block_envelope

REPORT_NAME = 'l23net_s1_run.json'
TRACE_NAME = 'l23net_s1_trace.h5'


def validate_configuration(cfg, size):
    common = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    common.analysis.condition = 'reference'
    common.env.simulation.MDD = False
    common.env.ts.apply = False
    result = _validate_configuration(common, size)
    mode, arm = str(cfg.analysis.mode), str(cfg.analysis.arm)
    if mode not in ('debug', 'qualification', 'discovery'):
        raise ValueError('Invalid S1 mode')
    if cfg.analysis.condition not in ('reference', 'mdd') or bool(cfg.env.simulation.MDD) != (cfg.analysis.condition == 'mdd'):
        raise ValueError('Condition/MDD mismatch')
    if not cfg.env.ts.apply or cfg.env.network.tstart != 0 or cfg.env.network.v_init != -80:
        raise ValueError('S1 requires enabled uniform-field path, tstart=0 and v_init=-80')
    if mode == 'debug':
        if cfg.analysis.require_full_network or not cfg.experiment.debug or size > 2:
            raise ValueError('Debug is restricted to reduced, <=2-rank tests')
    else:
        gate = s0.load_gate(str(cfg.analysis.s0_gate))
        contract = _replay_contract(cfg, size)
        for key in ('experiment_seed','condition'):
            contract.pop(key)
        contract['simulation'].pop('MDD')
        expected_contract = copy.deepcopy(gate['reference_contract'])
        expected_contract['simulation']['duration_ms'] = 60000.
        expected_contract['stimulation_enabled'] = True
        if contract != expected_contract:
            raise ValueError('Scientific/EEG/environment contract differs from the frozen R1 reference')
        if arm not in gate['allowed_arms']:
            raise ValueError('Action not qualified by S0; phase/frequency-matched arms are blocked')
        if not cfg.analysis.require_full_network or cfg.experiment.debug or size != 624:
            raise ValueError('Production S1 requires full network / 624 ranks')
        expected = (60000., 1000., 8000., 28000., 50000., 1000.)
        actual = (float(cfg.analysis.duration_ms), float(cfg.analysis.window_ms),
                  float(cfg.analysis.excluded_ms), float(cfg.analysis.stim_start_ms),
                  float(cfg.analysis.stim_stop_ms), float(cfg.analysis.ramp_ms))
        if actual != expected or float(cfg.env.network.dt) != .025 or cfg.analysis.env_seed != 0:
            raise ValueError('S1 timing/dt/randomness differs from the frozen 60-s contract')
        if mode == 'qualification':
            if cfg.experiment.seed != 8451 or arm not in ('sham', 'fixed_alpha'):
                raise ValueError('Qualification is frozen to seed 8451 / sham and fixed_alpha')
        else:
            if cfg.experiment.seed not in s0.PROTOCOL['discovery_seeds'] or cfg.analysis.condition != 'mdd':
                raise ValueError('Discovery requires a fresh frozen MDD seed')
            from experiments.l23net_analysis.s1_analysis import load_qualification
            load_qualification(str(cfg.analysis.qualification), gate['sha256'])
    if arm not in ('sham', 'fixed_theta', 'fixed_alpha', 'fixed_low_beta', 'transverse_alpha'):
        raise ValueError('Unsupported action (no unqualified phase arms)')
    if cfg.analysis.condition == 'reference' and arm != 'sham':
        raise ValueError('Reference qualification must be unstimulated')
    a, b, r = map(float, [cfg.analysis.stim_start_ms, cfg.analysis.stim_stop_ms, cfg.analysis.ramp_ms])
    if not 0 <= cfg.analysis.excluded_ms < a < b < cfg.analysis.duration_ms or not 0 < r < (b-a)/2:
        raise ValueError('Invalid epoch/ramp configuration')
    if any(not np.isclose(x/float(cfg.analysis.window_ms), round(x/float(cfg.analysis.window_ms))) for x in [a,b]):
        raise ValueError('Stage boundaries must align with observation windows')
    return result


class FieldProtocol:
    stage_names = ['excluded', 'baseline', 'stimulation', 'washout']

    def __init__(self, cfg):
        self.cfg = cfg
        self.arm = str(cfg.analysis.arm)
        self.mode = str(cfg.analysis.mode)
        self.gate = None if self.mode == 'debug' else s0.load_gate(str(cfg.analysis.s0_gate))
        self.dt = float(cfg.env.network.dt)
        self.window = float(cfg.analysis.window_ms)
        self.start = float(cfg.analysis.stim_start_ms)
        self.stop = float(cfg.analysis.stim_stop_ms)
        self.ramp = float(cfg.analysis.ramp_ms)
        self.excluded = float(cfg.analysis.excluded_ms)
        self.total = float(cfg.analysis.duration_ms)
        self.filter = s0.CausalEEG(self.dt)
        self.times, self.eeg = [], []
        self.baseline_digest = hashlib.sha256()
        self.decision = None
        self.last_field_endpoint = 0.
        self.completed = 0.
        self.nonzero_samples = 0
        self.max_field = 0.
        self.baseline_rates = []
        self.tissue_coupling = False
        from experiments.l23net_analysis.s1_analysis import code_hashes
        self.code_sha256 = code_hashes()
        self.spike_file = None
        self.spike_path = Path(str(cfg.experiment.dir))/'l23net_s1_spikes.h5'

    def action(self, index):
        start = index*self.window
        if start >= self.start and self.decision is None:
            if self.mode == 'debug':
                screen = {'eligible': True, 'debug_only': True}
            else:
                values = s0.epoch(np.concatenate(self.times), np.concatenate(self.eeg), self.excluded/1000, self.start/1000)
                screen = s0.scores(s0.log_powers(values), self.gate['targets']['20s'])
            safe = all(np.isfinite(v) and 0 <= v <= (20 if 'PYR' in p else 100)
                       for rates in self.baseline_rates for p,v in rates.items())
            # Qualification is engineering only, including a no-field reference.
            eligible = safe and (screen['eligible'] or self.mode == 'qualification')
            band = self.arm.removeprefix('fixed_') if self.arm.startswith('fixed_') else 'alpha'
            frequency = s0.FIXED[band] if self.arm != 'sham' else 0.
            self.decision = {'requested_arm': self.arm, 'delivered_arm': self.arm if eligible else 'sham',
                             'screen': screen, 'baseline_rates_safe': safe,
                             'qualification_screen_bypass': self.mode == 'qualification',
                             'amplitude_v_per_m': .4 if eligible and self.arm != 'sham' else 0.,
                             'frequency_hz': frequency, 'phase_rad_at_onset': 0.,
                             'field_direction': [1.,0.,0.] if self.arm=='transverse_alpha' else [0.,0.,1.],
                             'decision_time_ms': start}
        stage = 0 if start < self.excluded else 1 if start < self.start else 2 if start < self.stop else 3
        action = {'ac_amplitude_v_per_m': 0., 'frequency_hz': 0.}
        kwargs = {}
        if stage == 2:
            action.update(ac_amplitude_v_per_m=self.decision['amplitude_v_per_m'],
                          frequency_hz=self.decision['frequency_hz'], field_direction=self.decision['field_direction'])
            if start == self.start:
                action['phase_rad'] = 0.
            kwargs['block_envelope'] = {'start_ms': self.start, 'stop_ms': self.stop, 'ramp_ms': self.ramp}
        return action, kwargs, stage

    def observe(self, index, result, extracellular, assigned_peaks):
        import h5py
        if self.spike_file is None:
            self.spike_file = h5py.File(self.spike_path, 'x')
            self.spike_file.attrs['format'] = 'l23net_s1_sparse_spikes_v1'
            for population in result['spikes']:
                group = self.spike_file.create_group(population)
                for name,dtype in [('times_ms','f8'),('gids','i8')]:
                    group.create_dataset(name,shape=(0,),maxshape=(None,),chunks=True,dtype=dtype)
        for population,spikes in result['spikes'].items():
            for name in ('times_ms','gids'):
                dataset = self.spike_file[population][name]
                values = np.asarray(spikes[name])
                old = len(dataset)
                dataset.resize((old+len(values),))
                dataset[old:] = values
        self.spike_file.attrs['committed_windows'] = index+1
        self.spike_file.flush()
        times = np.asarray(result['sample_times_ms'])
        eeg = np.asarray(result['eeg_v'])[0]
        t, y = self.filter.append(eeg)
        self.times.append(t)
        self.eeg.append(y)
        if (index+1)*self.window <= self.start:
            for x in (result['eeg_v'], result['dipole_nA_um']):
                self.baseline_digest.update(np.ascontiguousarray(x, dtype='<f8').tobytes())
            if index*self.window >= self.excluded:
                from experiments.l23net_analysis.s1_analysis import population_rates
                self.baseline_rates.append(population_rates(result['firing_rates']))
        stim = result['stimulation']
        ft, field = np.asarray(stim['time_ms']), np.asarray(stim['field_v_per_m'])
        active = self.start <= index*self.window < self.stop
        expected = np.zeros_like(field)
        if active:
            expected = self.decision['amplitude_v_per_m']*np.sin(2*np.pi*self.decision['frequency_hz']*(ft-self.start)/1000)
            expected = apply_raised_cosine_block_envelope(expected, time_ms=ft, block_start_ms=self.start,
                                                          block_stop_ms=self.stop, ramp_ms=self.ramp)
        errors = []
        if not np.isfinite(field).all() or not np.allclose(field, expected, atol=1e-10, rtol=0):
            errors.append('Field differs from the analytic absolute-time sinusoid/envelope')
        if not np.isclose(field[0], self.last_field_endpoint, atol=1e-10, rtol=0):
            errors.append('Field discontinuity at window boundary')
        if not np.any(expected) and not np.allclose(extracellular, 0, atol=1e-15, rtol=0):
            errors.append('Nonzero extracellular potential outside the active block')
        self.last_field_endpoint = float(field[-1])
        self.nonzero_samples += int(np.count_nonzero(np.abs(field[:-1]) > 1e-12))
        self.max_field = max(self.max_field, float(np.max(np.abs(field))))
        self.tissue_coupling |= bool(active and max(assigned_peaks)>1e-12)
        self.completed = (index+1)*self.window
        if self.completed == self.total:
            self.close()
        return errors, {'field_peak_v_per_m': float(np.max(np.abs(field))),
                        'assigned_extracellular_peak_mV': float(max(assigned_peaks)),
                        'field_error_v_per_m': float(np.max(np.abs(field-expected))),
                        'field_direction': stim['field_direction'].tolist()}

    def summary(self):
        value = {'mode': self.mode, 's0_sha256': None if self.gate is None else self.gate['sha256'],
                 'code_sha256': self.code_sha256, 'tissue_coupling_observed': self.tissue_coupling,
                 'decision': self.decision, 'prestimulation_eeg_dipole_sha256': self.baseline_digest.hexdigest(),
                 'nonzero_field_samples': self.nonzero_samples, 'max_field_v_per_m': self.max_field,
                 'transient_excluded_ms': self.excluded,
                 'analysis_storage_samples': int(sum(len(t) for t in self.times))}
        if self.completed == self.total:
            from experiments.l23net_analysis.r1_protocol import sha256
            value['spike_file_sha256'] = sha256(self.spike_path)
        if self.completed == self.total and self.mode != 'debug':
            times, eeg = np.concatenate(self.times), np.concatenate(self.eeg)
            outcomes = {}
            for name, start, stop, target in [('baseline',8,28,'20s'),('stimulation',29,49,'20s'),('washout',50,60,'10s')]:
                x = s0.epoch(times,eeg,start,stop)
                powers = s0.log_powers(x)
                outcomes[name] = {'interval_s': [start,stop], 'log_powers': powers.tolist(),
                                  **s0.scores(powers,self.gate['targets'][target])}
            value['outcomes'] = outcomes
            value['fundamental_excluded'] = {}
            x = s0.epoch(times,eeg,29,49)
            for band,freq in s0.FIXED.items():
                powers = s0.log_powers(x,(freq-.5,freq+.5))
                value['fundamental_excluded'][band] = {'log_powers': powers.tolist(),
                    **s0.scores(powers,self.gate['targets']['fundamental_excluded'][band])}
        return value

    def close(self):
        if self.spike_file is not None:
            self.spike_file.close()


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg):
    from mpi4py import MPI
    # Run validation before loading root-only analysis state, fail collectively.
    try:
        validate_configuration(cfg, MPI.COMM_WORLD.size)
        protocol = FieldProtocol(cfg)
        run_replay(cfg, validator=validate_configuration, report_name=REPORT_NAME, trace_name=TRACE_NAME,
                   build_audit=construction_audit, abort_on_failure=True, window_protocol=protocol,
                   scope='S1 full-spectrum, fixed-frequency discovery; technical pass is not efficacy.',
                   limitations=['No phase matching or adaptive phase claim.', 'Ideal neural-only EEG.',
                                'One matched full history per structure, not independently branched futures.',
                                'Duration-matched R1 target requires late-epoch stationarity qualification.'])
    except Exception as exc:
        import traceback
        from experiments.l23net_analysis.r1_protocol import save_json
        traceback.print_exc()
        try:
            directory = Path(str(cfg.experiment.dir)); directory.mkdir(parents=True,exist_ok=True)
            save_json(directory/f'failure_rank_{MPI.COMM_WORLD.rank}.json',
                      {'error':repr(exc),'traceback':traceback.format_exc()})
        except Exception:
            pass
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise


if __name__ == '__main__':
    main()
