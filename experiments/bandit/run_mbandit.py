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
from agent.dqn import DQN, ReplayBuffer
from agent.mbandit import EpsilonGreedyBandit

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import json

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils.utils import setup_folders



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

    def bandit_actions(cfg):
        amps = np.linspace(cfg.env.stimAmplitude_min, cfg.env.stimAmplitude_max, cfg.agent.n_arms)
        freqs = np.linspace(cfg.env.stimFreq_min, cfg.env.stimFreq_max, cfg.agent.n_arms)
        return amps, freqs

    bandit = EpsilonGreedyBandit(cfg.agent.n_arms, epsilon=0.1)
    amps, freqs = bandit_actions(cfg)

    rewards = np.zeros(cfg.agent.n_trials)
    optimal = np.zeros(cfg.agent.n_trials)
    true_means = np.random.rand(cfg.agent.n_arms)  # True mean rewards for each arm (hidden from the agent)

    for t in range(cfg.agent.n_trials):

        chosen_arm = bandit.select_arm()

        env = NeuronEnv(cfg, MPI_VAR)
        reward = env.exploration_rollout(policy_seq=[[0., 1.], [amps[chosen_arm], freqs[chosen_arm]]], buffer=None, steps=2)  # off-line
        env.close()

        bandit.update(chosen_arm, reward)
        rewards[t] = reward
        # optimal[t] = (chosen_arm == optimal_arm)

    if RANK==0:
        plt.figure()
        plt.plot(rewards)
        plt.title("Reward performance: Multi-arm Bandit Algorithm")
        plt.xlabel("Trials")        # Label for x-axis
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig("mbandit_result.png")
        plt.show()


if __name__ == "__main__":
    main()

# python run_bandit.py experiment.name=test9 env=ballnstick agent=mbandit env.network.syn_activity=True
# mpirun -np 2 python run_mbandit.py experiment.name=test9 env=ballnstick agent=mbandit env.network.syn_activity=True experiment.tqdm=False

# python run_bandit.py experiment.name=mtest9 env=hl23net agent=mbandit experiment.debug=True env.network.syn_activity=True
# mpirun -np 64 python run_mbandit.py experiment.name=mtest9 env=hl23net agent=mbandit experiment.debug=True env.network.syn_activity=True experiment.tqdm=False

# killall mpirun


