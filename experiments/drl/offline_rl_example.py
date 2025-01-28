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
        print(cfg)
        cfg = setup_folders(cfg)
    COMM.Barrier()

    def sample_random_actions(cfg, simulation_steps):
        amp_samples = np.random.uniform(cfg.env.stimAmplitude_min, cfg.env.stimAmplitude_min, size=simulation_steps)
        freq_samples = np.random.uniform(cfg.env.stimFreq_min, cfg.env.stimFreq_max, size=simulation_steps)
        return [[amp, freq] for amp, freq in zip(amp_samples, freq_samples)]

    def run_experiment():
        tic_0 = time.perf_counter()
        env = NeuronEnv(cfg, MPI_VAR)

        buffer = ReplayMemory(cfg.agent, bufferid="ballnstick_f0_r0")
        iql_agent = IQL(cfg)

        print(len(buffer))

        exploration_steps = 10
        evaluation_steps = 2

        rew = []

        for i in range(0, 1):  # collect data <S, A, R, S', Done>
            action_seq = sample_random_actions(cfg, exploration_steps)
            env.exploration_rollout(policy_seq=action_seq, buffer=buffer, steps=exploration_steps)
            iql_agent.train(buffer, epochs=50)
            reward = env.evaluation_rollout(policy=iql_agent, buffer=buffer, steps=evaluation_steps)
            print(reward)
            rew.append(reward)

        #iql_agent.train(buffer, epochs=40)

        # on-line evaluation.
        # reward = env.evaluation_rollout(policy=iql_agent, buffer=buffer, steps=evaluation_steps)
        # print(reward)

        env.close()
        buffer.close()

        plt.plot(rew)
        plt.show()

        print('simulation Time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
        # return reward

    run_experiment()  # [mA, Hz]  -> 3000 nA
    print("done") if RANK==0 else None


if __name__ == "__main__":
    main()

# python offline_rl_example.py experiment.name=test9 env=ballnstick env.network.syn_activity=True
# mpirun -np 8 python offline_rl_example.py experiment.name=test9 env=hl23net env.network.syn_activity=True experiment.debug=True
# killall mpirun
