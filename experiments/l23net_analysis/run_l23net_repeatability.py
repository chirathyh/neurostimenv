"""Replay a frozen L23Net dose arm with a separately seeded OU history."""

from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra

from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import repeatability_protocol as p
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.replay_validation import canonical_json_sha256
from experiments.l23net_analysis.run_l23net_dose_response import DoseProtocol, validate_configuration
from experiments.l23net_analysis.run_l23net_g1b import construction_audit
from experiments.l23net_analysis.run_l23net_no_field_replay import (
    _global_structure_fingerprint, _replay_contract, run_replay,
)
from experiments.l23net_analysis.run_l23net_s1 import REPORT_NAME, TRACE_NAME
from setup.circuits.L23Net.background_history import validate_history_seed


def audit_build(environment, cfg, history_seed, archived):
    audit = construction_audit(environment, cfg)
    comm = environment.comm
    if history_seed is not None:
        local = getattr(environment.network, 'l23net_background_history', None)
        ranks = comm.gather(local, root=0)
    if cfg.analysis.mode != 'debug':
        structure = _global_structure_fingerprint(environment, environment.MPI_VAR)
    if comm.rank != 0:
        return None
    if history_seed is not None:
        valid = all(row is not None and row['history_seed'] == history_seed and
                    row['generator'] == 'Random123' and row['global_index'] == 0 and
                    row['ou_sites'] == build['counts']['ou_processes']
                    for row, build in zip(ranks, audit['by_rank'], strict=True))
        if not valid or sum(row['ou_sites'] for row in ranks) == 0:
            audit['errors'].append('OU reseeding did not cover all audited background sites')
        else:
            audit['background_history'] = {
                'history_seed': history_seed, 'generator': 'Random123', 'global_index': 0,
                'stream_sha256': canonical_json_sha256(ranks), 'by_rank': ranks,
                'ou_sites': sum(row['ou_sites'] for row in ranks),
            }
    if archived is not None:
        if (structure['global_sha256'] != archived['structure_sha256'] or
                audit['invariant_sha256'] != archived['build_sha256']):
            audit['errors'].append('Rebuilt network does not match its archived structure/OU-parameter fingerprints')
    return audit


def validate_worker(cfg, size, history_seed):
    if cfg.analysis.mode != 'discovery' or history_seed is None:
        raise ValueError('Production repeatability runs require discovery mode and an explicit OU history seed')
    suite = Path(str(cfg.analysis.dose_suite)).resolve()
    manifest, gate, _, _ = p.check_manifest(suite)
    dose.validate_replay_contract(_replay_contract(cfg, size), gate, 'discovery')
    name = f's{cfg.experiment.seed}_{cfg.analysis.arm}'
    if (not any(job['name'] == name and job['history_seed'] == history_seed for job in manifest['jobs']) or
            Path(str(cfg.experiment.dir)).resolve() != suite / name or
            Path(str(cfg.analysis.s0_gate)).resolve() != suite / 's0_gate.json'):
        raise ValueError('Worker settings differ from the planned seed, arm or OU history')
    return p.load_archive()[cfg.experiment.seed]


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    try:
        validate_configuration(cfg, comm.size)
        if cfg.analysis.arm not in p.ARMS:
            raise ValueError('Arm is outside the repeatability design')
        history_seed = cfg.env.online.get('l23net_history_seed')
        if history_seed is not None:
            validate_history_seed(history_seed)
        archived = None
        if comm.rank == 0 and cfg.analysis.mode != 'debug':
            archived = validate_worker(cfg, comm.size, history_seed)

        run_replay(cfg, validator=validate_configuration, report_name=REPORT_NAME, trace_name=TRACE_NAME,
                   build_audit=lambda env, conf: audit_build(env, conf, history_seed, archived),
                   abort_on_failure=True, window_protocol=DoseProtocol(cfg, rank=comm.rank),
                   scope='Fixed 14-Hz tACS repeatability on archived L23Net structures with new OU input histories.',
                   limitations=['Ideal neural-only EEG. OU histories restart from the episode beginning.',
                                'The archival comparison uses the legacy OU generator. Phase is fixed to onset.'])
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
