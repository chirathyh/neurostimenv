import sys
import time
from mpi4py import MPI
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import warnings
warnings.simplefilter('ignore', Warning)

from env.models.neuron.env import NeuronEnv
from agent.iql import IQL
from utils.utils import setup_folders
from utils.buffers import ReplayMemory

import numpy as np
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import json

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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

    for i in range(0, 2):
        env = NeuronEnv(cfg, MPI_VAR, ENV_SEED=i)
        EEG = env.step_n(i_stim=None, t_ext=None, stim_elec=None) # no stimulation.
        CIRCUIT = "_MDD_" if cfg.env.simulation.MDD else "_HEALTHY_"
        FILE = cfg.experiment.dir+"/EEG"+CIRCUIT+str(i)+".csv"
        np.savetxt(FILE, EEG, delimiter=",") if RANK == 0 else None
        env.close()

    print('\n### Experiment run time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
    print("### Experiment completed.") if RANK==0 else None


if __name__ == "__main__":
    main()


# ballnstick
# python run_simulations.py experiment.name=test9 env=ballnstick env.network.syn_activity=True
# mpirun -np 2 python run_simulations.py experiment.name=test9 env=ballnstick env.simulation.duration=1000 env.simulation.MDD=True env.ts.apply=False env.network.syn_activity=True experiment.tqdm=False

# hl23net
# mpirun -np 2 python run_simulations.py experiment.name=test9 env=hl23net env.network.syn_activity=True experiment.debug=True experiment.tqdm=False
# killall mpirun
