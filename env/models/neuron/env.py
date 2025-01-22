import os
import sys
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
from env.models.neuron.extracellular import ExtracellularModels
from env.eeg import features
from utils.utils import prep_stim_seq
import matplotlib.pyplot as plt


class NeuronEnv(gym.Env):
    def __init__(self, args, MPI_VAR):
        self.args = args
        self.MPI_VAR = MPI_VAR
        self.sampling_rate = (1 / args.env.network.dt) * 1000
        self._reset()

    def evaluation_rollout(self, policy, buffer, steps):
        # TODO: this is inefficient!. doing this way due to limitations of NEURON to pause/restart simulations when extracellular mechanisms are used.
        # NOTES:
        #  - Extracellular mechanisms are added at initialization (Play Vector).
        #  - SaveState() can't be used because extra-mech has dynamics objects.
        #  - Therefore in this approach have to ensure simulations are "deterministic". e.g., adding "Synapse Activity".
        sim_data = []
        action, cur_state = [], []
        for i in range(0, steps):
            cur_action = [0., 1.] if i == 0 else policy.get_action(cur_state)
            action.append(cur_action)
            cur_steps = i+1
            self.network.tstart, self.network.tstop = 0., self.args.env.simulation.obs_win_len * cur_steps
            i_stim, t_stim = prep_stim_seq(action=action, step_size=self.args.env.simulation.obs_win_len, steps=cur_steps, dt=self.network.dt)
            eeg = self.step_n(i_stim, t_stim, stim_elec=0)  # run the simulation

            # slice cur_eeg; apply feature/obs calc ;obtain cur_state
            chunk_size = int(len(eeg[0]) / cur_steps)
            chunked_eeg = [eeg[0][i:i + chunk_size] for i in range(0, len(eeg[0])-1, chunk_size)]
            reward = features.reward_func_simple(np.array([chunked_eeg[-1]]), self.sampling_rate)  # reward for next state
            obs = features.feature_space(eeg=np.array([chunked_eeg[-1]]), fs=self.sampling_rate)  # obs of next state
            states = [obs[key].item() if isinstance(obs[key], np.ndarray) else obs[key] for key in obs]
            next_state = np.array(states)

            if i > 0:  # start saving transitions, avoid first
                buffer.store(cur_state, cur_action, reward, next_state, 0.)

            cur_state = next_state

            sim_data.append(eeg)
            sim_data.append([i_stim])

        temp = buffer.get()
        print(temp)
        exit()

        # for testing only:
        # max_length = max(len(arr[0]) for arr in sim_data)
        # fig, axes = plt.subplots(len(sim_data), 1, figsize=(16, len(sim_data) * 2))
        # # Plot each array in a subplot
        # for i, arr in enumerate(sim_data):
        #     ax = axes[i] if len(sim_data) > 1 else axes  # Handle single subplot case
        #     ax.plot(range(1, len(arr[0]) + 1), arr[0], label=f"Array {i+1}")
        #     ax.set_xlim(1, max_length)  # Set x-axis limit to the maximum length
        #     ax.set_title(f"Plot of Array {i+1}")
        #     ax.set_xlabel("Index")
        #     ax.set_ylabel("Value")
        #     ax.legend()
        #     ax.grid(True)
        # plt.tight_layout()
        # plt.show()
        # exit()

    def exploration_rollout(self, policy_seq, buffer, steps):
        # TODO: this is inefficient. e.g., for an episilon-greedy algo exploration required "evaluation_rollout"
        # NOTE: Therefore; can explore offline RL algorithms which would be more suitable.
        self.network.tstart, self.network.tstop = 0., self.args.env.simulation.obs_win_len * steps
        i_stim, t_stim = prep_stim_seq(action=policy_seq, step_size=self.args.env.simulation.obs_win_len, steps=steps, dt=self.network.dt)
        full_eeg = self.step_n(i_stim, t_stim, stim_elec=0)  # run the simulation

        # for testing
        sim_data = []
        sim_data.append(full_eeg)
        sim_data.append([i_stim])

        chunk_size = int(len(full_eeg[0]) / steps)
        chunked_eeg = [full_eeg[0][i:i + chunk_size] for i in range(0, len(full_eeg[0])-1, chunk_size)]

        cur_state, cur_action = [], []
        for c in range(0, len(chunked_eeg)):
            cur_action = policy_seq[c]
            reward = features.reward_func_simple(np.array([chunked_eeg[c]]), self.sampling_rate)
            obs = features.feature_space(eeg=np.array([chunked_eeg[c]]), fs=self.sampling_rate)
            states = [obs[key].item() if isinstance(obs[key], np.ndarray) else obs[key] for key in obs]
            next_state = np.array(states)

            if c > 0:  # start saving transitions, avoid first
                buffer.store(cur_state, cur_action, torch.tensor([reward]), next_state, torch.tensor([0.]))
            cur_state = next_state

        # temp = buffer.get()
        # print(temp)
        # exit()

        # max_length = max(len(arr[0]) for arr in sim_data)
        # fig, axes = plt.subplots(len(sim_data), 1, figsize=(16, len(sim_data) * 2))
        # # Plot each array in a subplot
        # for i, arr in enumerate(sim_data):
        #     ax = axes[i] if len(sim_data) > 1 else axes  # Handle single subplot case
        #     ax.plot(range(1, len(arr[0]) + 1), arr[0], label=f"Array {i+1}")
        #     ax.set_xlim(1, max_length)  # Set x-axis limit to the maximum length
        #     ax.set_title(f"Plot of Array {i+1}")
        #     ax.set_xlabel("Index")
        #     ax.set_ylabel("Value")
        #     ax.legend()
        #     ax.grid(True)
        # plt.tight_layout()
        # plt.show()


    def step_n(self, I_stim, t_ext, stim_elec):
        COMM = self.MPI_VAR['COMM']
        RANK = self.MPI_VAR['RANK']
        if self.args.env.ts.apply:
            self.extracellular_models[0].probe.set_current(stim_elec, I_stim)
            self.network.enable_extracellular_stimulation(self.extracellular_models[0], t_ext, n=5)
        COMM.Barrier()
        SPIKES = self.network.simulate(probes=self.extracellular_models, **self.args.env.network.networkSimulationArguments)
        COMM.Barrier()
        if RANK == 0:
            P = self.extracellular_models[1].data['imem']  # numpy array <3, timesteps>
            pot_db_4s_top = self.four_sphere_top.get_dipole_potential(P, np.array([-10., 0., 0.]))  # Units: mV
            eeg = np.array(pot_db_4s_top) * 1e-3  # convert units: V
        return eeg

    def test_multiple_steps(self, action, steps):

        def customise_set_current_pulses(amp1, width1, interpulse, t_stop, dt,
                                 biphasic=False, interphase=None, n_pulses=None, n_bursts=None, interburst=None, amp2=None,
                                 width2=None, t_start=None):
            '''
            Computes and sets pulsed currents on specified electrode.

            Parameters
            ----------
            el_id: int
                Electrode index
            amp1: float
                Amplitude of stimulation. If 'biphasic', amplitude of the first phase.
            width1: float
                Duration of the pulse in ms. If 'biphasic', duration of the first phase.
            interpulse: float
                Interpulse interval in ms
            t_stop: float
                Stop time in ms
            dt: float
                Sampling period in ms
            biphasic: bool
                If True, biphasic pulses are used
            amp2: float
                Amplitude of second phase stimulation (when 'biphasic'). If None, this amplitude is the opposite of 'amp1'
            width2: float
                Duration of the pulse of the second phase in ms (when 'biphasic'). If None, it is the same as 'width1'
            interphase: float
                For biphasic stimulation, duration between two phases in ms
            n_pulses: int
                Number of pulses. If 'interburst' is given, this indicates the number of pulses in a burst
            interburst: float
                Inter burst interval in ms
            n_bursts: int
                Total number of burst (if None, no limit is used).
            t_start: float
                Start of the stimulation in ms

            Returns
            -------
            current: np.array
                Array of current values
            t_ext: np.array
                timestamps of computed currents
            '''
            if biphasic:
                if amp2 is None:
                    amp2 = -amp1
                if width2 is None:
                    width2 = width1
                if interphase is None:
                    interphase = 0
            if t_start is None:
                t_start = 0
            t_ext = np.arange(t_start, t_stop, dt)
            current_values = np.zeros(len(t_ext))

            if interburst is not None:
                assert n_pulses is not None, "For bursting pulses provide 'n_pulses' per burst"

            if n_pulses is None:
                n_pulses = np.inf
            if n_bursts is None:
                n_bursts = np.inf

            t_pulse = t_start
            n_p = 0
            n_b = 0
            while t_pulse < t_stop and n_p < n_pulses and n_b < n_bursts:
                pulse_idxs = np.where((t_ext > t_pulse) & (t_ext <= t_pulse + width1))
                current_values[pulse_idxs] = amp1
                if biphasic:
                    t_pulse2 = t_pulse + width1 + interphase
                    pulse2_idxs = np.where((t_ext > t_pulse2) & (t_ext <= t_pulse2 + width2))
                    current_values[pulse2_idxs] = amp2
                    t_pulse_end = t_pulse2 + width2
                else:
                    t_pulse_end = t_pulse + width1
                n_p += 1

                if interburst:
                    if n_p == n_pulses:
                        n_p = 0
                        t_pulse = t_pulse_end + interburst
                        n_b += 1
                    else:
                        t_pulse = t_pulse_end + interpulse
                else:
                    t_pulse = t_pulse_end + interpulse

            return current_values, t_ext

        stim_elec = 0
        step_size = 500.
        t_start_stim = 0.
        I_stim, t_ext = np.array([]), np.array([])

        self.network.tstart = 0.
        self.network.tstop = step_size

        sim_data = []

        for i in range(0, steps):

            I_stim_cur, t_ext_cur = customise_set_current_pulses(n_pulses=20, biphasic=True, width1=5, amp1=6000,
                                                                 dt=self.network.dt, interpulse=200, t_stop=self.network.tstop, t_start=t_start_stim)
            I_stim = np.concatenate((I_stim, I_stim_cur))
            t_ext = np.concatenate((t_ext, t_ext_cur))

            eeg = self.step_n(stim_elec, I_stim, t_ext)
            sim_data.append(eeg)
            sim_data.append([I_stim])

            t_start_stim += step_size
            self.network.tstop += step_size

        max_length = max(len(arr[0]) for arr in sim_data)
        fig, axes = plt.subplots(len(sim_data), 1, figsize=(16, len(sim_data) * 2))
        # Plot each array in a subplot
        for i, arr in enumerate(sim_data):
            ax = axes[i] if len(sim_data) > 1 else axes  # Handle single subplot case
            ax.plot(range(1, len(arr[0]) + 1), arr[0], label=f"Array {i+1}")
            ax.set_xlim(1, max_length)  # Set x-axis limit to the maximum length
            ax.set_title(f"Plot of Array {i+1}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()
        exit()

    def _step(self, action):
        COMM = self.MPI_VAR['COMM']
        RANK = self.MPI_VAR['RANK']
        reward, state, done, info = 0, 0, 0, 0
        I_stim, t_ext, eeg_top = [], [], []

        if self.args.env.ts.apply and action is not None:  # add transcranial stimulation
            print("\nApplying tACS") if RANK == 0 else None
            # I_stim, t_ext = self.extracellular.electrode.probe.set_current_pulses(
            #         n_pulses=20, biphasic=True,  # width2=width1, amp2=-amp1
            #         width1=5, amp1=3000,  # nA
            #         dt=self.network.dt, t_stop=1200, interpulse=200, el_id=0, t_start=200)
            I_stim, t_ext = self.extracellular.set_pulse(self.network, self.step_count, action)
            self.network.enable_extracellular_stimulation_mpi(self.extracellular.electrode, self.step_count, t_ext, n=5)
            #self.network.enable_extracellular_stimulation(self.extracellular.electrode, t_ext, n=5)
        COMM.Barrier()
        SPIKES = self.network.simulate(probes=self.extracellular_models, **self.args.env.network.networkSimulationArguments)
        COMM.Barrier()

        if RANK == 0:
            P = self.extracellular_models[1].data['imem']  # numpy array <3, timesteps>
            pot_db_4s_top = self.four_sphere_top.get_dipole_potential(P, np.array(self.args.env.network.position))  # Units: mV
            eeg_top = np.array(pot_db_4s_top) * 1e-3  # convert units: V
            print(eeg_top)
            plt.figure()
            plt.plot(eeg_top[0][-40000:])
            plt.savefig(self.args.experiment.dir+"/eeg_top_plot.png")  # You can specify the file format and path here
            plt.show()
            plt.close()

        if RANK == 0:  # calc reward and observation space
            reward = features.reward_func_simple(eeg_top, self.sampling_rate)
            obs = features.feature_space(eeg=eeg_top, fs=self.sampling_rate)
            state = np.array([np.ravel(v) for v in obs.values()]).flatten()
            self.step_count += 1
            info = {'SPIKES': SPIKES, 'I_stim': I_stim, 't_ext': t_ext, 'sim_t': 0, 'dom_freq': obs['Dominant Frequency']}

        self.step_count += 1

        return state, reward, done, info

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
        self.network = NetworkEnv(**self.args.env.networkParameters)
        # create populations, setup synapses, and connections.
        if self.args.env.name == 'hl23pyrnet':
            from setup.circuits.hl23pyrnet.utils import create_populations_connect
            create_populations_connect(self.network, self.args, self.MPI_VAR)
        if self.args.env.name == 'hl23net':
            from setup.circuits.L23Net.utils import setup_network
            setup_network(self.network, self.args, self.MPI_VAR)
        if self.args.env.name == 'ballnstick':
            from setup.circuits.ballnstick.utils import setup_network
            setup_network(self.network, self.args, self.MPI_VAR)

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
