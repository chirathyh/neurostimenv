"""Run a sham or fixed-14-Hz trajectory from the frozen L23Net dose suite."""

import copy
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np

from experiments.l23net_analysis import dose_response_protocol as p
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.replay_validation import canonical_json_sha256
from experiments.l23net_analysis.run_l23net_g1b import construction_audit
from experiments.l23net_analysis.run_l23net_no_field_replay import (
    _replay_contract, _validate_configuration, run_replay,
)
from experiments.l23net_analysis.run_l23net_s1 import FieldProtocol, REPORT_NAME, TRACE_NAME


def validate_configuration(cfg, size):
    if cfg.analysis.condition != 'mdd' or not cfg.env.simulation.MDD or not cfg.env.ts.apply:
        raise ValueError('Dose-response runs require the MDD circuit and the uniform-field interface')
    common = copy.deepcopy(cfg)
    common.analysis.condition = 'reference'
    common.env.simulation.MDD = False
    common.env.ts.apply = False
    result = _validate_configuration(common, size)

    stage, arm = str(cfg.analysis.mode), str(cfg.analysis.arm)
    if arm not in p.AMPLITUDES:
        raise ValueError('Arm is outside the predefined dose grid')
    if stage == 'debug':
        if size > 2 or not cfg.experiment.debug or cfg.analysis.require_full_network:
            raise ValueError('Debug is restricted to a reduced serial/two-rank network')
    else:
        if stage not in p.STAGES or not any(
            job['seed'] == cfg.experiment.seed and job['arm'] == arm for job in p.jobs(stage)
        ):
            raise ValueError('Stage, seed or arm is outside the frozen design')
        if size != 624 or cfg.experiment.debug or not cfg.analysis.require_full_network:
            raise ValueError('The production design requires the full network and 624 ranks')
        if dict(cfg.analysis.expected_population_sizes) != p.POPULATIONS:
            raise ValueError('Unexpected population contract')

    timing = tuple(float(cfg.analysis[k]) for k in (
        'duration_ms', 'window_ms', 'excluded_ms', 'stim_start_ms', 'stim_stop_ms', 'ramp_ms'
    ))
    total, window, excluded, start, stop, ramp = timing
    if not 0 <= excluded < start < stop < total or not 0 < ramp < (stop - start) / 2:
        raise ValueError('Invalid stimulation epochs or ramps')
    if any(not np.isclose(t / window, round(t / window)) for t in (excluded, start, stop)):
        raise ValueError('Epochs must align with observation windows')
    if stage != 'debug' and timing != p.TIMING_MS:
        raise ValueError('Production timing differs from the frozen 60-second protocol')
    return result


class DoseProtocol(FieldProtocol):
    def __init__(self, cfg, *, rank):
        # The replay loop performs the same collectives on every rank. Only root
        # owns the protocol state, analysis arrays and sparse spike file.
        self.spike_file = None
        if rank != 0:
            return
        super().__init__(cfg)
        self.dose_protocol_hash = canonical_json_sha256(p.PROTOCOL)

    def action(self, index):
        start = index * self.window
        if start >= self.start and self.decision is None:
            if self.mode == 'debug':
                screen = {'eligible': True, 'debug_only': True}
            else:
                baseline = s0.epoch(np.concatenate(self.times), np.concatenate(self.eeg),
                                    self.excluded / 1000, self.start / 1000)
                screen = s0.scores(s0.log_powers(baseline), self.gate['targets']['20s'])
            self.decision = p.decide(self.arm, self.mode, screen, p.baseline_rates_safe(self.baseline_rates), start)

        action = {'ac_amplitude_v_per_m': 0., 'frequency_hz': 0.}
        if start < self.excluded:
            return action, {}, 0
        if start < self.start:
            return action, {}, 1
        if start >= self.stop:
            return action, {}, 3
        action.update(ac_amplitude_v_per_m=self.decision['amplitude_v_per_m'],
                      frequency_hz=self.decision['frequency_hz'], field_direction=p.DIRECTION)
        if start == self.start:
            action['phase_rad'] = 0.
        envelope = {'start_ms': self.start, 'stop_ms': self.stop, 'ramp_ms': self.ramp}
        return action, {'block_envelope': envelope}, 2

    def summary(self):
        return {**super().summary(), 'dose_protocol_sha256': self.dose_protocol_hash}


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    try:
        validate_configuration(cfg, comm.size)
        if comm.rank == 0 and cfg.analysis.mode != 'debug':
            suite = Path(str(cfg.analysis.dose_suite)).resolve()
            manifest, gate, _, _ = p.check_manifest(suite)
            p.validate_replay_contract(_replay_contract(cfg, comm.size), gate, manifest['stage'])
            name = f's{cfg.experiment.seed}_{cfg.analysis.arm}'
            if (cfg.analysis.mode != manifest['stage'] or
                    not any(j['name'] == name for j in manifest['jobs']) or
                    Path(str(cfg.experiment.dir)).resolve() != suite / name or
                    Path(str(cfg.analysis.s0_gate)).resolve() != suite / 's0_gate.json'):
                raise ValueError('Worker configuration differs from its submission')

        run_replay(cfg, validator=validate_configuration, report_name=REPORT_NAME, trace_name=TRACE_NAME,
                   build_audit=construction_audit, abort_on_failure=True,
                   window_protocol=DoseProtocol(cfg, rank=comm.rank),
                   scope='Fixed 14-Hz L23Net dose qualification or matched dose discovery.',
                   limitations=['Ideal neural-only EEG and one matched input history per seed.',
                                'Tissue fields are simulator settings. Phase is fixed to stimulus onset.'])
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
