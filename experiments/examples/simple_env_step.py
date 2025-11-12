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
        print(cfg)
        cfg = setup_folders(cfg)
    COMM.Barrier()

    def run_experiment():
        env = NeuronEnv(cfg, MPI_VAR)
        EEG = env.step_n(i_stim=None, t_ext=None, stim_elec=None)
        env.close()

    run_experiment()
    print("done") if RANK==0 else None


if __name__ == "__main__":
    main()

# python simple_env_step.py experiment.name=test9 env=ballnstick env.network.syn_activity=True
# killall mpirun


