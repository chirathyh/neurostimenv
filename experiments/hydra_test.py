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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import json

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

        model = DQN(12, 5)  # TODO: automatically assign
        target_model = DQN(12, 5)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        buffer_capacity = 10000  # Replay buffer capacity
        replay_buffer = ReplayBuffer(buffer_capacity)

    COMM.Barrier()

    def run_experiment(action=None):
        env = NeuronEnv(cfg, MPI_VAR)
        state = np.random.rand(1, 12).astype(np.float32)
        d_freq = []
        COMM.Barrier()
        for step in range(0, 1):
            tic_0 = time.perf_counter()
            obs, reward, done, info = env.step(action=action)
            if RANK==0:
                #print('TS Setting, current', action[0], ' freq: ', freq)
                d_freq.append(info['dom_freq'])
                print('Dominant Frequency:', str(info['dom_freq']), 'Hz')
                print('Run time for simulation step: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes')
                print('n_step: ', str(info['sim_t']))
                # print('I_stim: ', str(info['I_stim']))
                # print('t_ext: ', str(info['t_ext']))
            COMM.Barrier()
            state = obs

            # if step > 3:
            #     current = current + 10

            #current = current + 10  # update the control variable
            # freq = freq + 5
        env.close()
        return d_freq

    dict = {}
    f1 = run_experiment(action=None)
    COMM.Barrier()
    # d2, f2 = run_experiment(current=1, freq=10)
    # COMM.Barrier()
    # d3, f3 = run_experiment(current=10, freq=20)
    # COMM.Barrier()
    f4 = run_experiment(action=[4, 10])
    COMM.Barrier()



    if RANK == 0:
        #dict.update(d1)
        # dict.update(d2)
        # dict.update(d3)
        # dict.update(d4)

        # with open('data.json', 'w') as json_file:
        #     json.dump(dict, json_file, indent=4)

        print('done')

        # plt.plot(range(len(f2)), f2, marker='o', linestyle='-', color='r')
        # plt.plot(range(len(f3)), f3, marker='o', linestyle='-', color='k')
        plt.plot(range(len(f4)), f4, marker='o', linestyle='-', color='g')
        plt.plot(range(len(f1)), f1, marker='o', linestyle='-', color='b')
        # Adding labels and title
        plt.xlabel('Iteration')
        plt.ylabel('Frequency')
        plt.title('Iteration vs Value')
        # Show the plot
        plt.grid(True)  # Optional: adds gridlines to the plot
        plt.show()


if __name__ == "__main__":
    main()

# mpirun -np 10 python hydra_test.py experiment.name=test33
# mpirun -np 10 python hydra_test.py experiment.name=test33 env=hl23pyrnet
# killall mpirun


