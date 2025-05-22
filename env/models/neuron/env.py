import os
import re
import sys
from tqdm import tqdm
from decouple import config
import torch

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import neuron
import gym
from gym import spaces
import numpy as np
import scipy.signal as ss

from lfpykit.eegmegcalc import FourSphereVolumeConductor

from env.models.neuron.networkenv import NetworkEnv
#from LFPy import Network

from setup.circuits.hl23pyrnet.utils import create_populations_connect
from setup.circuits.L23Net.utils import setup_network
from setup.circuits.ballnstick.utils import setup_network_ballnstick

from env.models.neuron.extracellular import ExtracellularModels
from env.eeg import features
from utils.utils import prep_stim_seq
from visualiser.core import plot_episode
import matplotlib.pyplot as plt


class NeuronEnv(gym.Env):
    def __init__(self, args, MPI_VAR, ENV_SEED=0):
        self.args = args
        self.MPI_VAR = MPI_VAR
        self.MPI_VAR['GLOBALSEED'] = self.MPI_VAR['GLOBALSEED'] + ENV_SEED
        self.MPI_VAR['SEED'] = self.MPI_VAR['SEED'] + ENV_SEED
        self.sampling_rate = (1 / args.env.network.dt) * 1000
        self._reset()

    def evaluation_rollout(self, policy, buffer, steps):
        RANK = self.MPI_VAR['RANK']
        COMM = self.MPI_VAR['COMM']
        if RANK == 0:
            print("\n==> Running evaluation simulations...")
        # TODO: this is inefficient!. doing this way due to limitations of NEURON to pause/restart simulations when extracellular mechanisms are used.
        # NOTES:
        #  - Extracellular mechanisms are added at initialization (Play Vector).
        #  - SaveState() can't be used because extra-mech has dynamics objects.
        #  - Therefore in this approach have to ensure simulations are "deterministic". e.g., adding "Synapse Activity".
        eval_reward, cur_steps = 0, 0
        action, cur_state, i_stim, t_stim, cur_action, eeg = [], [], [], [], [], [[]]

        #for i in tqdm(range(0, steps), desc="Evaluation Progress", unit="step", disable=not self.args.experiment.tqdm):
        for i in range(0, steps):
            if RANK == 0:
                cur_action = [[0., 1.]] if i < 2 else policy.get_action([cur_state])  # first two actions are default.
                cur_action = cur_action[0]  # remove batch dimension
                action.append(cur_action)
                cur_steps = i+1
                i_stim, t_stim = prep_stim_seq(action=action, step_size=self.args.env.simulation.obs_win_len, steps=cur_steps, dt=self.network.dt)
            else:
                cur_action, i_stim, t_stim = None, None, None

            # Broadcast to all ranks
            i_stim = COMM.bcast(i_stim, root=0)
            t_stim = COMM.bcast(t_stim, root=0)
            cur_action = COMM.bcast(cur_action, root=0)

            self.network.tstart, self.network.tstop = 0., self.args.env.simulation.obs_win_len * (i+1)

            COMM.Barrier()
            # Redirect NEURON output to the log file and use tqdm progress bar.
            time_pattern = re.compile(r"t = (\d+\.\d+) ms")
            log_file_path = self.args.experiment.dir+'/neuron_evaluation_sim_output.log'
            with open(log_file_path, "a") as log_file:
                original_stdout = sys.stdout  # Save the original stdout
                sys.stdout = log_file
                try:
                    with tqdm(total=self.network.tstop, unit="ms", desc="Simulation Progress", disable=not self.args.experiment.tqdm) as pbar:
                        eeg = self.step_n(i_stim, t_stim, stim_elec=0)  # run the simulation
                        log_file.flush()  # Ensure file buffer is written
                        sys.stdout = original_stdout  # Restore stdout temporarily
                        with open(log_file_path, "r") as log_reader:
                            last_time = 0
                            for line in log_reader:
                                match = time_pattern.search(line)
                                if match:
                                    t_current = float(match.group(1))
                                    pbar.update(t_current - last_time)  # Update progress bar
                                    last_time = t_current
                        sys.stdout = log_file  # Redirect back to log file
                finally:
                    sys.stdout = original_stdout  # Restore original stdout
            #eeg = self.step_n(i_stim, t_stim, stim_elec=0)  # run the simulation
            COMM.Barrier()

            if RANK == 0:
                # slice cur_eeg; apply feature/obs calc ;obtain cur_state
                chunk_size = int(len(eeg[0]) / cur_steps)
                chunked_eeg = [eeg[0][i:i + chunk_size] for i in range(0, len(eeg[0])-1, chunk_size)]
                reward = features.reward_func_simple(np.array([chunked_eeg[-1]]), self.sampling_rate)  # reward for next state
                obs = features.feature_space(eeg=np.array([chunked_eeg[-1]]), fs=self.sampling_rate, ts=cur_action)  # obs of next state
                states = [obs[key].item() if isinstance(obs[key], np.ndarray) else obs[key] for key in obs]
                next_state = np.array(states)

                if i > 1 and buffer is not None:  # start saving transitions, avoid first and second. first is transient.
                    buffer.store(cur_state, cur_action, torch.tensor([reward]), next_state, torch.tensor([0.]))
                    eval_reward += reward

                cur_state = next_state

        if RANK == 0:
            print("### Evaluation rollout successfully completed.\n")
            plot_episode(self.args, eeg, i_stim, t_stim, cur_action, steps) if self.args.experiment.plot else None
        return eval_reward

    def exploration_rollout(self, policy_seq, buffer, steps, **kwargs):
        RANK = self.MPI_VAR['RANK']
        COMM = self.MPI_VAR['COMM']
        print("\n==> Running exploration simulations...") if RANK == 0 else None
        # TODO: this is inefficient. e.g., for an episilon-greedy algo exploration required "evaluation_rollout"
        # NOTE: Therefore; can explore offline RL algorithms which would be more suitable.
        self.network.tstart, self.network.tstop = 0., self.args.env.simulation.obs_win_len * steps
        reward = None
        if RANK == 0:
            i_stim, t_stim = prep_stim_seq(action=policy_seq, step_size=self.args.env.simulation.obs_win_len, steps=steps, dt=self.network.dt)
        else:
            i_stim, t_stim = None, None
        i_stim = COMM.bcast(i_stim, root=0)
        t_stim = COMM.bcast(t_stim, root=0)
        COMM.Barrier()

        # Redirect NEURON output to the log file and use tqdm progress bar.
        time_pattern = re.compile(r"t = (\d+\.\d+) ms")
        log_file_path = self.args.experiment.dir+'/neuron_exploration_sim_output.log'
        with open(log_file_path, "w") as log_file:
            original_stdout = sys.stdout  # Save the original stdout
            sys.stdout = log_file
            try:
                with tqdm(total=self.network.tstop, unit="ms", desc="Simulation Progress", disable=not self.args.experiment.tqdm) as pbar:
                    full_eeg = self.step_n(i_stim, t_stim, stim_elec=0)  # run the simulation
                    log_file.flush()  # Ensure file buffer is written
                    sys.stdout = original_stdout  # Restore stdout temporarily
                    with open(log_file_path, "r") as log_reader:
                        last_time = 0
                        for line in log_reader:
                            match = time_pattern.search(line)
                            if match:
                                t_current = float(match.group(1))
                                pbar.update(t_current - last_time)  # Update progress bar
                                last_time = t_current
                    sys.stdout = log_file  # Redirect back to log file
            finally:
                sys.stdout = original_stdout  # Restore original stdout
        #full_eeg = self.step_n(i_stim, t_stim, stim_elec=0)  # run the simulation

        if RANK == 0:
            # save the EEG signal for mbandit algorithm
            save = kwargs.get("save", False)
            save_seed = kwargs.get("seed", 0)
            save_mode = kwargs.get("mode", 'training')
            if self.args.agent.agent == 'mbandit' and save:
                FILE = self.args.experiment.dir+"/"+save_mode+"/EEG_BANDIT_"+str(save_seed)+".csv"
                np.savetxt(FILE, full_eeg, delimiter=",")
                print("### EEG Saved.")

            chunk_size = int(len(full_eeg[0]) / steps)
            chunked_eeg = [full_eeg[0][i:i + chunk_size] for i in range(0, len(full_eeg[0])-1, chunk_size)]

            cur_state, cur_action = [], []
            for c in range(0, len(chunked_eeg)):
                cur_action = policy_seq[c]
                reward = features.reward_func_simple(np.array([chunked_eeg[c]]), self.sampling_rate)
                obs = features.feature_space(eeg=np.array([chunked_eeg[c]]), fs=self.sampling_rate, ts=cur_action)
                states = [obs[key].item() if isinstance(obs[key], np.ndarray) else obs[key] for key in obs]
                next_state = np.array(states)

                if c > 1 and buffer is not None:  # start saving transitions, avoid first and second
                    buffer.store(cur_state, cur_action, torch.tensor([reward]), next_state, torch.tensor([0.]))
                    # print(reward, cur_action)
                cur_state = next_state
            print("Exploration rollout successfully completed.\n")
        return reward  # the returned reward is always the last segment.

    def step_n(self, i_stim, t_ext, stim_elec):
        COMM = self.MPI_VAR['COMM']
        RANK = self.MPI_VAR['RANK']
        eeg = None
        if self.args.env.ts.apply:
            self.extracellular_models[0].probe.set_current(stim_elec, i_stim)
            self.network.enable_extracellular_stimulation_mpi(self.extracellular_models[0], t_ext, n=5)  # using because issue with mpi and L23Net
            #self.network.enable_extracellular_stimulation(self.extracellular_models[0], t_ext, n=5)

        COMM.Barrier()
        SPIKES = self.network.simulate(probes=self.extracellular_models, **self.args.env.network.networkSimulationArguments)
        COMM.Barrier()

        if RANK == 0:
            P = self.extracellular_models[1].data['imem']  # numpy array <3, timesteps>
            pot_db_4s_top = self.four_sphere_top.get_dipole_potential(P, np.array(self.args.env.network.position))  # Units: mV
            eeg = np.array(pot_db_4s_top) * 1e-3  # convert units: V
            # FILE = self.args.experiment.dir+"/dipole"
            # np.save(FILE, P)
        # eeg = COMM.bcast(eeg, root=0)
        return eeg

    # # TODO: _step is old and broken, and follows old structure.
    # def _step(self, action, stim_elec=0):
    #     COMM = self.MPI_VAR['COMM']
    #     RANK = self.MPI_VAR['RANK']
    #     i_stim, t_ext, eeg, reward = [], [], [[]], -1
    #     cur_steps = 1
    #
    #     if RANK == 0:
    #         i_stim, t_stim = prep_stim_seq(action=[action], step_size=self.args.env.simulation.obs_win_len, steps=cur_steps, dt=self.network.dt)
    #
    #     COMM.Barrier()
    #     if self.args.env.ts.apply and action is not None:  # add transcranial stimulation
    #         self.extracellular_models[0].probe.set_current(stim_elec, i_stim)
    #         self.network.enable_extracellular_stimulation_mpi(self.extracellular_models[0], t_ext, n=5)
    #     SPIKES = self.network.simulate(probes=self.extracellular_models, **self.args.env.network.networkSimulationArguments)
    #     COMM.Barrier()
    #
    #     if RANK == 0:
    #         P = self.extracellular_models[1].data['imem']  # numpy array <3, timesteps>
    #         pot_db_4s_top = self.four_sphere_top.get_dipole_potential(P, np.array(self.args.env.network.position))  # Units: mV
    #         eeg = np.array(pot_db_4s_top) * 1e-3  # convert units: V
    #         reward = features.reward_func_simple(eeg, self.sampling_rate)
    #
    #     return reward
    #
    # def _step_old(self, action):
    #     COMM = self.MPI_VAR['COMM']
    #     RANK = self.MPI_VAR['RANK']
    #     reward, state, done, info = 0, 0, 0, 0
    #     I_stim, t_ext, eeg_top = [], [], []
    #
    #     if self.args.env.ts.apply and action is not None:  # add transcranial stimulation
    #         print("\nApplying tACS") if RANK == 0 else None
    #         # I_stim, t_ext = self.extracellular.electrode.probe.set_current_pulses(
    #         #         n_pulses=20, biphasic=True,  # width2=width1, amp2=-amp1
    #         #         width1=5, amp1=3000,  # nA
    #         #         dt=self.network.dt, t_stop=1200, interpulse=200, el_id=0, t_start=200)
    #         I_stim, t_ext = self.extracellular.set_pulse(self.network, self.step_count, action)
    #         self.network.enable_extracellular_stimulation_mpi(self.extracellular.electrode, self.step_count, t_ext, n=5)
    #         #self.network.enable_extracellular_stimulation(self.extracellular.electrode, t_ext, n=5)
    #     COMM.Barrier()
    #     SPIKES = self.network.simulate(probes=self.extracellular_models, **self.args.env.network.networkSimulationArguments)
    #     COMM.Barrier()
    #
    #     if RANK == 0:
    #         P = self.extracellular_models[1].data['imem']  # numpy array <3, timesteps>
    #         pot_db_4s_top = self.four_sphere_top.get_dipole_potential(P, np.array(self.args.env.network.position))  # Units: mV
    #         eeg_top = np.array(pot_db_4s_top) * 1e-3  # convert units: V
    #         print(eeg_top)
    #         plt.figure()
    #         plt.plot(eeg_top[0][-40000:])
    #         plt.savefig(self.args.experiment.dir+"/eeg_top_plot.png")  # You can specify the file format and path here
    #         plt.show()
    #         plt.close()
    #
    #     if RANK == 0:  # calc reward and observation space
    #         reward = features.reward_func_simple(eeg_top, self.sampling_rate)
    #         obs = features.feature_space(eeg=eeg_top, fs=self.sampling_rate)
    #         state = np.array([np.ravel(v) for v in obs.values()]).flatten()
    #         self.step_count += 1
    #         info = {'SPIKES': SPIKES, 'I_stim': I_stim, 't_ext': t_ext, 'sim_t': 0, 'dom_freq': obs['Dominant Frequency']}
    #
    #     self.step_count += 1
    #
    #     return state, reward, done, info

    def _close(self):
        ##########################################################################
        # customary cleanup of object references - the psection() function may not
        # write correct information if NEURON still has object references in memory
        # even if Python references has been deleted. It will also allow the script
        # to be run in successive fashion.
        ##########################################################################
        self.network.pc.gid_clear()  # allows assigning new gids to threads
        self.extracellular = None
        syn = None
        synapseModel = None
        for population in self.network.populations.values():
            for cell in population.cells:
                cell.__del__()
                cell = None
            population.cells = None
            population = None
        # pop = None
        self.network = None
        neuron.h('forall delete_section()')

    def _reset(self):
        if self.MPI_VAR['RANK'] == 0:
            print("\n==> Initialising microcircuit...")

        self.network = NetworkEnv(**self.args.env.networkParameters)
        #self.network = Network(**self.args.env.networkParameters)

        # create populations, setup synapses, and connections.
        if self.args.env.name == 'hl23pyrnet':
            create_populations_connect(self.network, self.args, self.MPI_VAR)
        if self.args.env.name == 'hl23net':
            setup_network(self.network, self.args, self.MPI_VAR)
        if self.args.env.name == 'ballnstick':
            setup_network_ballnstick(self.network, self.args, self.MPI_VAR)

        self.extracellular_models = None
        self.four_sphere_top = None
        if self.args.env.eeg.measure:
            self.four_sphere_top = FourSphereVolumeConductor(np.array(self.args.env.eeg.locations),
                                                             self.args.env.eeg.foursphereheadmodel['radii'],
                                                             self.args.env.eeg.foursphereheadmodel['sigmas'])

            self.extracellular = ExtracellularModels(self.args)
            self.extracellular_models = self.extracellular.get_probes()

        self.step_count = 0

    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1, 12))
