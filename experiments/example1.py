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
def remove_axis_junk(ax, lines=['right', 'top']):
    """remove chosen lines from plotting axis"""
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


@hydra.main(version_base=None, config_path="../configs", config_name="config")
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

    def run_experiment(action=None):
        tic_0 = time.perf_counter()
        env = NeuronEnv(cfg, MPI_VAR)
        obs, reward, done, info = env.step(action=action)
        env.close()
        print('Dominant Frequency:', str(info['dom_freq']), 'Hz') if RANK==0 else None
        print('Reward:', str(reward)) if RANK==0 else None
        print('simulation Time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
        return reward

    run_experiment(action=None)  # [mA, Hz]  -> 3000 nA
    print("done") if RANK==0 else None
    #run_experiment(action=[1e-3, 4])  # [mA, Hz]  -> 3000 nA


if __name__ == "__main__":
    main()

# python example1.py experiment.name=test9 env=ballnstick env.network.syn_activity=Tru
# killall mpirun

# mpirun -np 32 python example1.py experiment.name=test1 env=hl23net env.network.dt=0.025 env.simulation.obs_win_len=100 experiment.debug=True
