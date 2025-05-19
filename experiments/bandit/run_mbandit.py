import sys
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

    # def bandit_actions(cfg):
    #     amps = np.linspace(cfg.env.stimAmplitude_min, cfg.env.stimAmplitude_max, cfg.agent.n_arms)
    #     freqs = np.linspace(cfg.env.stimFreq_min, cfg.env.stimFreq_max, cfg.agent.n_arms)
    #     return amps, freqs

    # bandit_action_pairs = [[0.5, 2.],
    #                        [0.5, 10.],
    #                        [1., 10.],
    #                        [1., 20.],
    #                        [2., 4.],
    #                        [2., 8.],
    #                        [2., 10.],
    #                        [2., 12.],
    #                        [2., 16.],
    #                        [2., 20.],
    #                        [2., 40.],
    #                        [4., 50.],
    #                        [15., 75.]]

    # bandit_action_pairs = [[0.001, 2.],
    #                        [0.001, 10.],
    #                        [0.01, 10.],
    #                        [0.01, 20.],
    #                        [0.05, 4.],
    #                        [0.05, 8.],
    #                        [0.05, 10.],
    #                        [0.05, 12.],
    #                        [0.05, 16.],
    #                        [0.05, 20.],
    #                        [0.05, 40.],
    #                        [0.1, 50.],
    #                        [0.5, 75.]]

    bandit_action_pairs = [[0.07304, 8.],
                           [0.14612, 8.],
                           [0.2922, 8.],
                           [0.14612, 10.],
                           [0.14612, 40.],
                           [1.0956, 77.5]]



    if RANK==0:
        bandit = EpsilonGreedyBandit(cfg.agent.n_arms, epsilon=1.0,
                                     pretrain=cfg.agent.pretrain, checkpoint=cfg.agent.checkpoint)
    else:
        bandit = None
    COMM.Barrier()

    rewards = np.zeros(cfg.agent.n_trials + cfg.agent.n_eval_trials)
    selected_freq = np.zeros(cfg.agent.n_trials + cfg.agent.n_eval_trials)
    selected_amp = np.zeros(cfg.agent.n_trials + cfg.agent.n_eval_trials)
    selected_arm = np.zeros(cfg.agent.n_trials + cfg.agent.n_eval_trials)
    experiment_seed = np.zeros(cfg.agent.n_trials + cfg.agent.n_eval_trials)

    for t in range(cfg.agent.n_trials):

        chosen_arm = bandit.select_arm() if RANK==0 else None
        chosen_arm = COMM.bcast(chosen_arm, root=0)
        COMM.Barrier()

        ENVSEED = cfg.experiment.seed + t
        env = NeuronEnv(cfg, MPI_VAR, ENV_SEED=ENVSEED)
        reward = env.exploration_rollout(policy_seq=[[0., 1.], [0., 1.], [0., 1.], [0., 1.],
                                                     [0., 1.],
                                                     [bandit_action_pairs[chosen_arm][0], bandit_action_pairs[chosen_arm][1]],
                                                     [0., 1.],
                                                     [bandit_action_pairs[chosen_arm][0], bandit_action_pairs[chosen_arm][1]]
                                                     ],
                                         buffer=None, steps=8, save=True, mode="training", seed=ENVSEED)  # off-line
        env.close()

        COMM.Barrier()
        if RANK==0:
            # update the bandit algorithm and save values and counts as checkpoints.
            bandit.update(chosen_arm, reward)
            np.save(cfg.experiment.dir+"/checkpoints/values.npy", bandit.values)  # Save as .npy file
            np.save(cfg.experiment.dir+"/checkpoints/counts.npy", bandit.counts)  # Save as .npy file

            # saving
            data = {"Reward": reward, "Arm": chosen_arm, "Amplitude": bandit_action_pairs[chosen_arm][0],
                    "Frequency": bandit_action_pairs[chosen_arm][1], "Seed": ENVSEED}
            with open(cfg.experiment.dir+"/training/STIM_BANDIT_"+str(ENVSEED)+".csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(data.keys())  # Write headers
                writer.writerow(data.values())  # Write values
            # save in bulk
            # rewards[t] = reward
            # selected_amp[t] = bandit_action_pairs[chosen_arm][0]
            # selected_freq[t] = bandit_action_pairs[chosen_arm][1]
            # selected_arm[t] = chosen_arm
            # experiment_seed[t] = ENVSEED
        # optimal[t] = (chosen_arm == optimal_arm)
        gc.collect()

    if RANK==0:
        plt.figure()
        plt.plot(rewards)
        plt.title("Reward performance: Multi-arm Bandit Algorithm")
        plt.xlabel("Trials")        # Label for x-axis
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(cfg.experiment.dir+"/mbandit_result.png")
        plt.close()

    # evaluate
    print("### Evaluating the best treatment...") if RANK == 0 else None
    best_arm = bandit.select_best_arm() if RANK == 0 else None
    best_arm = COMM.bcast(best_arm, root=0)
    COMM.Barrier()

    for i in range(cfg.agent.n_eval_trials):

        ENVSEED = cfg.experiment.seed + i + 1000
        env = NeuronEnv(cfg, MPI_VAR, ENV_SEED=ENVSEED)
        reward = env.exploration_rollout(policy_seq=[[0., 1.], [0., 1.], [0., 1.], [0., 1.],
                                                     [0., 1.],
                                                     [bandit_action_pairs[best_arm][0], bandit_action_pairs[best_arm][1]],
                                                     [0., 1.],
                                                     [bandit_action_pairs[best_arm][0], bandit_action_pairs[best_arm][1]]
                                                     ],
                                         buffer=None, steps=8, save=True, mode="testing", seed=ENVSEED)  # off-line
        env.close()

        COMM.Barrier()
        print("### Reward is: ", reward) if RANK == 0 else None
        print("### Best arm amplitude (mA): ", bandit_action_pairs[best_arm][0]) if RANK == 0 else None
        print("### Best arm freq (Hz): ", bandit_action_pairs[best_arm][1]) if RANK == 0 else None

        if RANK == 0:
            # saving
            data = {"Reward": reward, "Arm": best_arm,"Amplitude": bandit_action_pairs[best_arm][0],
                    "Frequency": bandit_action_pairs[best_arm][1], "Seed": ENVSEED}
            with open(cfg.experiment.dir+"/testing/STIM_BANDIT_"+str(ENVSEED)+".csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(data.keys())  # Write headers
                writer.writerow(data.values())  # Write values
            # save in bulk
            # rewards[cfg.agent.n_trials+i] = reward
            # selected_amp[cfg.agent.n_trials+i] = bandit_action_pairs[best_arm][0]
            # selected_freq[cfg.agent.n_trials+i] = bandit_action_pairs[best_arm][1]
            # selected_arm[cfg.agent.n_trials+i] = best_arm
            # experiment_seed[cfg.agent.n_trials+i] = ENVSEED

        gc.collect()

    if RANK == 0:
        # df = pd.DataFrame({
        #     'Reward': rewards,
        #     'Amplitude': selected_amp,
        #     'Frequency': selected_freq,
        #     'Seed': experiment_seed
        # })
        # df.to_csv(cfg.experiment.dir+'/experiment_summary.csv', index=False)
        print('\n### Experiment run time: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes')
        print("### Experiment completed.")


if __name__ == "__main__":
    main()

# python run_bandit.py experiment.name=test9 env=ballnstick agent=mbandit env.network.syn_activity=True
# mpirun -np 2 python run_mbandit.py experiment.name=test9 env=ballnstick agent=mbandit env.network.syn_activity=True experiment.tqdm=False agent.pretrain=True agent.checkpoint=test6

# python run_bandit.py experiment.name=mtest9 env=hl23net agent=mbandit experiment.debug=True env.network.syn_activity=True
# mpirun -np 64 python run_mbandit.py experiment.name=mtest9 env=hl23net agent=mbandit experiment.debug=True env.network.syn_activity=True experiment.tqdm=False

# killall mpirun


