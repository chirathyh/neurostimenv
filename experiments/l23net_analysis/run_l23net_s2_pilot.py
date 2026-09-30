"""Run one frozen S2-P sham/14-Hz trajectory using the unchanged S1 actuator."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import hydra
from experiments.l23net_analysis import s2_pilot_protocol as p
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.run_l23net_s1 import FieldProtocol, REPORT_NAME, TRACE_NAME
from experiments.l23net_analysis.run_l23net_no_field_replay import _validate_configuration, _replay_contract, run_replay
from experiments.l23net_analysis.run_l23net_g1b import construction_audit


def validate_configuration(cfg, size):
    if cfg.analysis.condition != 'mdd' or not cfg.env.simulation.MDD or cfg.env.simulation.DRUG:
        raise ValueError('Pilot requires MDD=true, DRUG=false')
    if cfg.analysis.arm not in p.ARMS or not cfg.env.ts.apply:
        raise ValueError('Only sham and the frozen 14-Hz field are allowed')
    if cfg.analysis.mode == 'debug':
        from experiments.l23net_analysis.run_l23net_s1 import validate_configuration as debug_validate
        return debug_validate(cfg, size)
    if cfg.analysis.mode != 'replication_pilot' or cfg.experiment.seed not in p.SEEDS:
        raise ValueError('Wrong pilot mode or seed namespace')
    common = copy.deepcopy(cfg)
    common.analysis.condition = 'reference'; common.env.simulation.MDD = False; common.env.ts.apply = False
    result = _validate_configuration(common, size)
    if not cfg.analysis.require_full_network or cfg.experiment.debug or size != 624:
        raise ValueError('Production requires full network and 624 ranks')
    timing = tuple(float(cfg.analysis[k]) for k in
                   ('duration_ms','window_ms','excluded_ms','stim_start_ms','stim_stop_ms','ramp_ms'))
    if timing != (60000.,1000.,8000.,28000.,50000.,1000.):
        raise ValueError('Pilot must preserve the exact S1 epochs')
    gate, _ = p.load_prerequisites(cfg.analysis.reference_suite)
    if s0.load_gate(cfg.analysis.s0_gate)['sha256'] != gate['sha256']:
        raise ValueError('Different runtime screening target')
    p.validate_contract(_replay_contract(cfg,size),gate)
    return result


class PilotProtocol(FieldProtocol):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pilot_code = p.code_hashes()
        self.target = None if self.mode == 'debug' else p.load_prerequisites(cfg.analysis.reference_suite)[1]

    def summary(self):
        value = super().summary()
        metadata = {'protocol_sha256': p.reference.canonical_json_sha256(p.PROTOCOL),
                    'code_sha256': self.pilot_code,
                    'target_sha256': None if self.target is None else self.target['sha256'],
                    'legacy_scores_are_diagnostic_only': True}
        if self.completed == self.total and self.target is not None:
            targets = self.target['targets']
            metadata['outcomes'] = {
                epoch: s0.scores(value['outcomes'][old]['log_powers'], targets[epoch])
                for epoch,old in [('plateau','stimulation'),('washout','washout')]}
            metadata['excluded_plateau'] = s0.scores(
                value['fundamental_excluded']['low_beta']['log_powers'],
                targets['fundamental_excluded_plateau']['low_beta'])
        value['s2_pilot'] = metadata
        return value


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg):
    from mpi4py import MPI
    try:
        validate_configuration(cfg, MPI.COMM_WORLD.size)
        run_replay(cfg, validator=validate_configuration, report_name=REPORT_NAME, trace_name=TRACE_NAME,
                   build_audit=construction_audit, abort_on_failure=True, window_protocol=PilotProtocol(cfg),
                   scope='S2-P held-out fixed 14-Hz replication pilot; technical pass is not efficacy.',
                   limitations=['Ten prespecified candidate structures; no outcome-dependent replacements.',
                                'Conditional on frozen 16-structure reference and baseline screening.',
                                'No EEG-relative phase, clinical safety, biological-subject or bandit claim.',
                                'Historical one-off synaptic input just after 4000 ms is retained; first 8 s excluded.',
                                'One stochastic trajectory per structure/arm; ideal neural-only EEG.'])
    except Exception as exc:
        import traceback
        from experiments.l23net_analysis.r1_protocol import save_json
        traceback.print_exc()
        directory = Path(str(cfg.experiment.dir)); directory.mkdir(parents=True,exist_ok=True)
        try:
            save_json(directory/f'failure_rank_{MPI.COMM_WORLD.rank}.json',
                      {'error':repr(exc),'traceback':traceback.format_exc()})
        finally:
            if MPI.COMM_WORLD.size > 1:
                MPI.COMM_WORLD.Abort(1)
        raise


if __name__ == '__main__':
    main()
