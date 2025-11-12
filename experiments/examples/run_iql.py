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

    def sample_random_actions(cfg, simulation_steps):
        # the first action is set to [0., 1.] for the transient.
        amp_samples = np.random.uniform(cfg.env.stimAmplitude_min, cfg.env.stimAmplitude_max, size=simulation_steps-1)
        freq_samples = np.random.uniform(cfg.env.stimFreq_min, cfg.env.stimFreq_max, size=simulation_steps-1)
        actions = [[amp, freq] for amp, freq in zip(amp_samples, freq_samples)]
        return [[0., 1.]] + actions

    def run_experiment():
        tic_0 = time.perf_counter()
        buffer = ReplayMemory(cfg.agent, bufferid=cfg.agent.bufferid, MPI_VAR=MPI_VAR)
        iql_agent = IQL(cfg)

        rew = []
        for i in range(0, 1):  # collect data <S, A, R, S', Done>
            ENVSEED = cfg.experiment.seed + i

            env = NeuronEnv(cfg, MPI_VAR, ENV_SEED=ENVSEED)
            action_seq = sample_random_actions(cfg, cfg.agent.n_expl_steps)
            env.exploration_rollout(policy_seq=action_seq, buffer=buffer, steps=cfg.agent.n_expl_steps)  # off-line
            env.close()

            print('\n### Exploration run time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
            tic_1 = time.perf_counter()

            COMM.Barrier()
            if RANK==0:
                print("\n==> Training RL agent...")
                iql_agent.train(buffer, epochs=25)
            COMM.Barrier()

            eval_env = NeuronEnv(cfg, MPI_VAR, ENV_SEED=ENVSEED)
            reward = eval_env.evaluation_rollout(policy=iql_agent, buffer=buffer, steps=cfg.agent.n_eval_steps)  # on-line
            eval_env.close()

            print('\n### Evaluation run time: ', str((time.perf_counter() - tic_1)/60)[:5], 'minutes') if RANK==0 else None

        if RANK==0:
            buffer.close()

        # plt.plot(rew)
        # plt.show()

        print('\n### Experiment run time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
        # return reward

    run_experiment()  # [mA, Hz]  -> 3000 nA
    print("### Experiment completed.") if RANK==0 else None


if __name__ == "__main__":
    main()

# ballnstick
# python run_iql.py experiment.name=test9 env=ballnstick env.network.syn_activity=True
# mpirun -np 2 python run_iql.py experiment.name=test9 env=ballnstick env.network.syn_activity=True experiment.tqdm=False

# hl23ney
# python run_iql.py experiment.name=test9 env=hl23net env.network.syn_activity=True experiment.debug=True experiment.tqdm=False
# mpirun -np 2 python run_iql.py experiment.name=test9 env=hl23net env.network.syn_activity=True experiment.debug=True experiment.tqdm=False
# mpirun -np 2 python run_iql.py experiment.name=test9 env=hl23net env.network.syn_activity=True experiment.debug=True experiment.tqdm=False experiment.plot=False

# killall mpirun
