"""Opt-in independent Gaussian streams for the existing L23Net OU inputs."""

from hashlib import sha256
from struct import pack


def validate_history_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 < seed < 2**32:
        raise ValueError('L23Net background history seed must be a positive uint32')
    return seed


def bind_background_history(network, history_seed):
    """Rebind one retained Random per OU site after construction, before finitialize."""
    validate_history_seed(history_seed)
    fingerprint = sha256(pack('<I', history_seed))
    count = 0
    for name in network.population_names:
        population = network.populations[name]
        for gid, cell in zip(population.gids, population.cells):
            processes, generators = cell.template.OUprocess, cell.template.roulist
            if processes.count() != generators.count():
                raise ValueError(f'Cell {gid} has mismatched OU processes and random generators')
            for site in range(int(processes.count())):
                rng = generators.o(site)
                if rng.Random123_globalindex() != 0:
                    raise ValueError('Expected Random123 global index zero for L23Net background histories')
                # Explicit stream keys are independent of the order or rank of construction.
                rng.Random123(history_seed, int(gid), site)
                first_draw = rng.normal(0, 1)
                processes.o(site).noiseFromRandom(rng)
                fingerprint.update(pack('<IId', int(gid), site, first_draw))
                count += 1
    return {'history_seed': history_seed, 'generator': 'Random123', 'global_index': 0,
            'ou_sites': count, 'stream_sha256': fingerprint.hexdigest()}
