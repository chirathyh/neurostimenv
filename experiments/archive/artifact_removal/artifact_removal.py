import sys
import time
from mpi4py import MPI
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import warnings
warnings.simplefilter('ignore', Warning)

from env.models.neuron.env import NeuronEnv
from utils.utils import setup_folders
from utils.utils import prep_stim_seq

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # set up MPI variables:
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    GLOBALSEED = cfg.experiment.seed
    SEED = GLOBALSEED*10000  # Create new RandomState for each RANK
    MPI_VAR = {'COMM': COMM, 'SIZE': SIZE, 'RANK': RANK, 'GLOBALSEED': GLOBALSEED, 'SEED': SEED}

    if RANK==0:
        print("\n### Experiment Configuration:")
        print(cfg)
        print("\n")
        cfg = setup_folders(cfg)
    COMM.Barrier()

    tic_0 = time.perf_counter()
    N_TRIALS = 4
    for i in range(0, 1):

        ENVSEED = cfg.experiment.seed
        env = NeuronEnv(cfg, MPI_VAR, ENV_SEED=ENVSEED)

        policy_seq=[[2, 20], [2, 20]]

        i_stim, t_stim = prep_stim_seq(action=policy_seq, step_size=cfg.env.simulation.obs_win_len, steps=2, dt=cfg.env.network.dt)
        if RANK == 0:
            print("i_stim: ", i_stim)
            print("t_stim: ", t_stim)
        EEG = env.step_n(i_stim=i_stim, t_ext=t_stim, stim_elec=0) # no stimulation.
        env.close()
        COMM.Barrier()

    print('\n### Experiment run time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
    print("### Experiment completed.") if RANK == 0 else None


if __name__ == "__main__":
    main()


# ballnstick
# python artifact_removal.py experiment.name=test9 env=ballnstick env.network.syn_activity=True
# mpirun -np 2 python artifact_removal.py experiment.name=test9 env=ballnstick env.simulation.duration=1000 env.simulation.MDD=True env.ts.apply=True env.network.syn_activity=True experiment.tqdm=False

# hl23net
# mpirun -np 2 python artifact_removal.py experiment.name=test9 env=hl23net env.ts.apply=True env.network.syn_activity=True experiment.debug=True experiment.tqdm=False
# killall mpirun
