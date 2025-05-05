import sys
import os
import time
import csv
import pandas as pd
from mpi4py import MPI
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import warnings
warnings.simplefilter('ignore', Warning)

from env.models.neuron.env import NeuronEnv
from agent.mbandit import EpsilonGreedyBandit

import numpy as np
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from utils.utils import setup_folders
import gc


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

    bandit_action_pairs = [[0.001, 2.],
                           [0.001, 10.],
                           [0.01, 10.],
                           [0.01, 20.],
                           [0.05, 4.],
                           [0.05, 8.],
                           [0.05, 10.],
                           [0.05, 12.],
                           [0.05, 16.],
                           [0.05, 20.],
                           [0.05, 40.],
                           [0.1, 50.],
                           [0.5, 75.]]

    if RANK==0:
        bandit = EpsilonGreedyBandit(cfg.agent.n_arms, epsilon=0.1,
                                     pretrain=cfg.agent.pretrain, checkpoint=cfg.agent.checkpoint)
    else:
        bandit = None
    COMM.Barrier()

    # best_arms = [1, 3, 10]
    # step_lens = [2, 5, 7]

    best_arms = [1, 3]
    step_lens = [2, 3]

    for best_arm in best_arms:
        for step_len_ in range(0, len(step_lens)):
            for i in range(cfg.agent.n_eval_trials):

                print("### Arm amplitude (mA): ", bandit_action_pairs[best_arm][0]) if RANK == 0 else None
                print("### Arm freq (Hz): ", bandit_action_pairs[best_arm][1]) if RANK == 0 else None
                print("### Step length (ms): ", step_lens[step_len_]) if RANK == 0 else None

                step_len = step_lens[step_len_]
                SUB_FOLDER = "ARM"+str(best_arm) + "_" + str(step_len) + "STEPS"
                os.makedirs(cfg.experiment.dir + '/testing/'+SUB_FOLDER, exist_ok=True)

                policy_seq = [[0., 1.]]
                for _ in range(0, step_len-1):
                    policy_seq.append([bandit_action_pairs[best_arm][0], bandit_action_pairs[best_arm][1]])
                print("### Sequence: ", policy_seq) if RANK == 0 else None

                ENVSEED = cfg.experiment.seed + i + 1000
                env = NeuronEnv(cfg, MPI_VAR, ENV_SEED=ENVSEED)
                reward = env.exploration_rollout(policy_seq=policy_seq, buffer=None, steps=step_len, save=True,
                                                 mode="testing/"+SUB_FOLDER, seed=ENVSEED)  # off-line
                env.close()

                COMM.Barrier()
                print("### Reward is: ", reward) if RANK == 0 else None

                if RANK == 0:
                    # saving
                    data = {"Reward": reward, "Arm": best_arm, "Amplitude": bandit_action_pairs[best_arm][0],
                            "Frequency": bandit_action_pairs[best_arm][1], "Seed": ENVSEED}
                    with open(cfg.experiment.dir+"/testing/"+SUB_FOLDER+"/STIM_BANDIT_"+str(ENVSEED)+".csv", "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(data.keys())  # Write headers
                        writer.writerow(data.values())  # Write values
                gc.collect()

    if RANK == 0:
        print('\n### Experiment run time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes')
        print("### Experiment completed.")


if __name__ == "__main__":
    main()

# python run_system_analysis.py experiment.name=test9 env=ballnstick agent=mbandit env.network.syn_activity=True
# mpirun -np 2 python run_system_analysis.py experiment.name=test9 env=ballnstick agent=mbandit env.network.syn_activity=True experiment.tqdm=False agent.pretrain=False agent.checkpoint=test6

# python run_system_analysis.py experiment.name=mtest9 env=hl23net agent=mbandit experiment.debug=True env.network.syn_activity=True
# mpirun -np 4 python run_system_analysis.py experiment.name=mtest9 env=hl23net agent=mbandit experiment.debug=True env.network.syn_activity=True experiment.tqdm=False

# killall mpirun


