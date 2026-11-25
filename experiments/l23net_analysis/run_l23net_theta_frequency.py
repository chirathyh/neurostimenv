"""Run one matched sham, 14-Hz, or 10-Hz trajectory with an independent OU history."""

import copy
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np

from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis import theta_frequency_protocol as p
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.replay_validation import canonical_json_sha256
from experiments.l23net_analysis.run_l23net_no_field_replay import _replay_contract, _validate_configuration, run_replay
from experiments.l23net_analysis.run_l23net_repeatability import audit_build
from experiments.l23net_analysis.run_l23net_s1 import FieldProtocol, REPORT_NAME, TRACE_NAME
from setup.circuits.L23Net.background_history import validate_history_seed


def validate_configuration(cfg, mpi_size):
    if cfg.analysis.condition != 'mdd' or not cfg.env.simulation.MDD or not cfg.env.ts.apply:
        raise ValueError('The theta frequency study requires MDD and the uniform-field interface')
    if cfg.analysis.arm not in p.ARMS:
        raise ValueError('Arm is outside the frozen three-arm comparison')
    validate_history_seed(cfg.env.online.get('l23net_history_seed'))
    common = copy.deepcopy(cfg)
    common.analysis.condition = 'reference'
    common.env.simulation.MDD = False
    common.env.ts.apply = False
    result = _validate_configuration(common, mpi_size)

    timing = tuple(float(cfg.analysis[key]) for key in (
        'duration_ms', 'window_ms', 'excluded_ms', 'stim_start_ms', 'stim_stop_ms', 'ramp_ms'))
    total, window, excluded, start, stop, ramp = timing
    if (not 0 <= excluded < start < stop < total or not 0 < ramp < (stop - start) / 2 or
            any(not np.isclose(t / window, round(t / window)) for t in (excluded, start, stop))):
        raise ValueError('Invalid epochs or ramps')
    if cfg.analysis.mode == 'debug':
        if cfg.analysis.require_full_network or not cfg.experiment.debug or mpi_size > 2:
            raise ValueError('Debug is restricted to a reduced serial/two-rank network')
    elif (cfg.analysis.mode != 'discovery' or cfg.experiment.seed not in p.SEEDS or
          cfg.experiment.debug or not cfg.analysis.require_full_network or mpi_size != 624 or
          dict(cfg.analysis.expected_population_sizes) != dose.POPULATIONS or timing != dose.TIMING_MS):
        raise ValueError('Expected the frozen full-network, 624-rank theta frequency protocol')
    return result


def validate_worker(cfg, mpi_size, history_seed):
    suite = Path(str(cfg.analysis.theta_suite)).resolve()
    manifest, gate, _, _, _ = p.check_manifest(suite)
    seed, arm = int(cfg.experiment.seed), str(cfg.analysis.arm)
    name = f's{seed}_{arm}'
    p.validate_contract(_replay_contract(cfg, mpi_size), gate, seed)
    if (arm not in gate['allowed_arms'] or
            not any(job['name'] == name and job['history_seed'] == history_seed for job in manifest['jobs']) or
            Path(str(cfg.experiment.dir)).resolve() != suite / name or
            Path(str(cfg.analysis.s0_gate)).resolve() != suite / 's0_gate.json'):
        raise ValueError('Worker differs from the audited seed, arm, or history assignment')


class ThetaProtocol(FieldProtocol):
    def __init__(self, cfg, rank):
        # The replay loop still follows the same collectives on all ranks.
        self.spike_file = None
        if rank == 0:
            super().__init__(cfg)

    def action(self, index):
        time_ms = index * self.window
        if time_ms >= self.start and self.decision is None:
            if self.mode == 'debug':
                screen = {'eligible': True, 'debug_only': True}
            else:
                baseline = s0.epoch(np.concatenate(self.times), np.concatenate(self.eeg),
                                    self.excluded / 1000, self.start / 1000)
                screen = s0.scores(s0.log_powers(baseline), self.gate['targets']['20s'])
            self.decision = p.decide(self.arm, screen, dose.baseline_rates_safe(self.baseline_rates), self.start)

        action = {'ac_amplitude_v_per_m': 0., 'frequency_hz': 0.}
        if time_ms < self.excluded:
            return action, {}, 0
        if time_ms < self.start:
            return action, {}, 1
        if time_ms >= self.stop:
            return action, {}, 3
        action.update(ac_amplitude_v_per_m=self.decision['amplitude_v_per_m'],
                      frequency_hz=self.decision['frequency_hz'], field_direction=dose.DIRECTION)
        if time_ms == self.start:
            action['phase_rad'] = 0.
        return action, {'block_envelope': {'start_ms': self.start, 'stop_ms': self.stop, 'ramp_ms': self.ramp}}, 2

    def summary(self):
        return {**super().summary(), 'theta_protocol_sha256': canonical_json_sha256(p.PROTOCOL)}


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    try:
        validate_configuration(cfg, comm.size)
        history_seed = cfg.env.online.l23net_history_seed
        if comm.rank == 0 and cfg.analysis.mode != 'debug':
            validate_worker(cfg, comm.size, history_seed)
        run_replay(cfg, validator=validate_configuration, report_name=REPORT_NAME, trace_name=TRACE_NAME,
                   build_audit=lambda env, conf: audit_build(env, conf, history_seed, None),
                   abort_on_failure=True, window_protocol=ThetaProtocol(cfg, comm.rank),
                   scope='Matched 14-Hz vs 10-Hz theta modulation in new full-network L23Net structures.',
                   limitations=['Ideal neural-only EEG. Fixed onset phase. OU histories restart the episode.',
                                'Theta modulation and three-band reference distance are separate outcomes.'])
    except Exception as exc:
        traceback.print_exc()
        try:
            folder = Path(str(cfg.experiment.dir))
            folder.mkdir(parents=True, exist_ok=True)
            save_json(folder / f'failure_rank_{comm.rank}.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        finally:
            if comm.size > 1:
                comm.Abort(1)
        raise


if __name__ == '__main__':
    main()
