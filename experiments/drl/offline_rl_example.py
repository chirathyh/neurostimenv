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


import random
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import json

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Policy():
    def __init__(self, actions=2):
        self.n_actions=actions

    def get_action(self, cur_state=None):
        return random.choice([[6e-3, 4], [3e-3, 8]])


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

    def run_experiment(action=None):
        tic_0 = time.perf_counter()
        env = NeuronEnv(cfg, MPI_VAR)

        # policy = Policy()
        buffer = ReplayMemory(cfg.agent)
        iql_agent = IQL(cfg.agent)

        # action_sequence = [[6e-3, 4], [3e-3, 8], [3e-3, 8]]  # TODO: can be requested from policy. , [3e-3, 8], [6e-3, 4]
        # env.exploration_rollout(policy_seq=action_sequence, buffer=buffer, steps=3)
        # iql_agent.train(buffer, epochs=10)

        # on-line evaluation.
        env.evaluation_rollout(policy=iql_agent, buffer=buffer, steps=3)

        env.close()
        print('simulation Time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
        # return reward

    run_experiment(action=None)  # [mA, Hz]  -> 3000 nA
    print("done") if RANK==0 else None
    #run_experiment(action=[1e-3, 4])  # [mA, Hz]  -> 3000 nA


if __name__ == "__main__":
    main()

# python offline_rl_example.py experiment.name=test9 env=ballnstick env.network.syn_activity=True
# killall mpirun
