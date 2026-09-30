"""Healthy-only 60-second calibration, reusing the qualified S1 zero-field path."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import hydra
from omegaconf import OmegaConf
from experiments.l23net_analysis import reference60_protocol as protocol
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.run_l23net_s1 import FieldProtocol, REPORT_NAME, TRACE_NAME
from experiments.l23net_analysis.run_l23net_no_field_replay import _validate_configuration, _replay_contract, run_replay
from experiments.l23net_analysis.run_l23net_g1b import construction_audit
from experiments.l23net_analysis.s1_analysis import load_qualification


def validate_configuration(cfg, size):
    if cfg.analysis.condition != 'reference' or cfg.env.simulation.MDD or cfg.env.simulation.DRUG:
        raise ValueError('Reference calibration requires Healthy, MDD=false, DRUG=false')
    if cfg.analysis.arm != 'sham' or not cfg.env.ts.apply:
        raise ValueError('Reference calibration uses only sham through the qualified uniform-field path')
    common = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    common.env.ts.apply = False
    result = _validate_configuration(common, size)
    if cfg.env.network.tstart != 0 or cfg.env.network.v_init != -80:
        raise ValueError('Require tstart=0 and v_init=-80')
    if cfg.analysis.mode == 'debug':
        if cfg.analysis.require_full_network or not cfg.experiment.debug or size > 2:
            raise ValueError('Debug requires a reduced network and <=2 MPI ranks')
        # Keep debug epoch validation identical to S1.
        from experiments.l23net_analysis.run_l23net_s1 import validate_configuration as validate_debug
        return validate_debug(cfg, size)
    if cfg.analysis.mode != 'reference_calibration':
        raise ValueError('Wrong calibration mode')
    if cfg.experiment.seed not in protocol.SEEDS or cfg.analysis.env_seed != 0:
        raise ValueError('Use only the 16 independent R1 calibration seeds and env_seed=0')
    if not cfg.analysis.require_full_network or cfg.experiment.debug or size != 624:
        raise ValueError('Production requires the full 1000-cell circuit and 624 ranks')
    timing = tuple(float(cfg.analysis[k]) for k in
                   ('duration_ms', 'window_ms', 'excluded_ms', 'stim_start_ms', 'stim_stop_ms', 'ramp_ms'))
    if timing != (60000., 1000., 8000., 28000., 50000., 1000.):
        raise ValueError('Calibration must retain the exact S1 absolute epochs')
    gate = s0.load_gate(cfg.analysis.s0_gate)
    contract = _replay_contract(cfg, size)
    contract.pop('experiment_seed'); contract.pop('condition'); contract['simulation'].pop('MDD')
    expected = copy.deepcopy(gate['reference_contract'])
    expected['simulation']['duration_ms'] = 60000.
    expected['stimulation_enabled'] = True
    if contract != expected:
        raise ValueError('Scientific/package contract differs from frozen R1/S1')
    load_qualification(cfg.analysis.qualification, gate['sha256'])
    return result


class ReferenceProtocol(FieldProtocol):
    """Retain S1 formats and audit compatibility; mark old scores diagnostic only."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.calibration_code = protocol.code_hashes()

    def summary(self):
        value = super().summary()
        value['reference60'] = {'protocol_sha256': protocol.canonical_json_sha256(protocol.PROTOCOL),
                                'code_sha256': self.calibration_code,
                                'legacy_scores_are_diagnostic_only': True}
        return value


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg):
    from mpi4py import MPI
    try:
        validate_configuration(cfg, MPI.COMM_WORLD.size)
        run_replay(cfg, validator=validate_configuration, report_name=REPORT_NAME, trace_name=TRACE_NAME,
                   build_audit=construction_audit, abort_on_failure=True, window_protocol=ReferenceProtocol(cfg),
                   scope='Healthy absolute-epoch reference calibration; no active stimulation.',
                   limitations=['Same calibration structures as R1, not 16 new independent subjects.',
                                'All fields zero; does not establish stimulation efficacy.',
                                'Old S0 scores retained only for audit. New targets require all 16 verified trajectories.',
                                'Ideal neural-only EEG; seeds quantify model variability, not clinical uncertainty.'])
    except Exception as exc:
        import traceback
        from experiments.l23net_analysis.r1_protocol import save_json
        traceback.print_exc()
        directory = Path(str(cfg.experiment.dir)); directory.mkdir(parents=True, exist_ok=True)
        try:
            save_json(directory/f'failure_rank_{MPI.COMM_WORLD.rank}.json',
                      {'error': repr(exc), 'traceback': traceback.format_exc()})
        finally:
            if MPI.COMM_WORLD.size > 1:
                MPI.COMM_WORLD.Abort(1)
        raise


if __name__ == '__main__':
    main()
