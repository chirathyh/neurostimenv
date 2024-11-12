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
        print('Dominant Frequency:', str(info['dom_freq']), 'Hz')
        print('Reward:', str(reward))
        print('simulation Time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
        return reward

    run_experiment(action=None)  # [mA, Hz]  -> 3000 nA
    #run_experiment(action=[3e-3, 1])  # [mA, Hz]  -> 3000 nA

    # Example Usage
    n_steps = 100
    n_arms = 8
    true_means = np.random.rand(n_arms)  # True mean rewards for each arm (hidden from the agent)
    bandit = EpsilonGreedyBandit(n_arms, epsilon=0.1)

    rewards = np.zeros(n_steps)
    optimal = np.zeros(n_steps)

    amp_min = 1e-3
    amp_max = 10e-3
    freq_min = 1
    freq_max = 100
    amps = np.linspace(amp_min, amp_max, n_arms)
    freqs = np.linspace(freq_min, freq_max, n_arms)

    for t in range(n_steps):
        chosen_arm = bandit.select_arm()
        arm_param = [amps[chosen_arm], freqs[chosen_arm]]
        reward = run_experiment(action=arm_param)  # Simulate reward from chosen arm
        bandit.update(chosen_arm, reward)
        rewards[t] = reward
        # optimal[t] = (chosen_arm == optimal_arm)

    print(rewards)
    plt.figure()
    plt.plot(rewards)
    plt.show()
    #run_experiment(action=[3e-3, 4])  # [mA, Hz]  -> 3000 nA
    exit()





if __name__ == "__main__":
    main()

# python example1.py experiment.name=test9 env=ballnstick env.network.syn_activity=Tru
# killall mpirun


