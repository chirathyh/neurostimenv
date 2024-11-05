import os
import sys
from decouple import config

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


class NeuronEnv(gym.Env):
    def __init__(self, args, MPI_VAR):
        self.args = args
        self.MPI_VAR = MPI_VAR
        self.sampling_rate = (1 / args.env.network.dt) * 1000
        self._reset()

    def _step(self, action):
        COMM = self.MPI_VAR['COMM']
        RANK = self.MPI_VAR['RANK']
        reward, state, done, info = 0, 0, 0, 0
        I_stim, t_ext, eeg_top = [], [], []

        # if self.args.env.ts.apply and action is not None:  # add transcranial stimulation
        #     print("\nApplying tACS") if RANK == 0 else None
        #     I_stim, t_ext = self.extracellular.set_pulse(self.network, self.step_count, action)
        #     v_ext = self.network.enable_extracellular_stimulation_mpi(self.extracellular_models[0], self.step_count, t_ext, n=5)
        #
        #     if self.args.experiment.verbose:
        #         print("\n v_ext stimulation") if RANK == 0 else None
        #         print(v_ext) if RANK == 0 else None
        #     # if RANK == 0: # doesn't work.
        #     #     I_stim, t_ext = self.extracellular.set_pulse(self.network, self.step_count, action)
        #     #v_ext = self.network.enable_extracellular_stimulation(self.extracellular_models[0], t_ext, n=5) if RANK == 0 else None


        I_stim, t_ext = self.extracellular.electrode.probe.set_current_pulses(
            n_pulses=20,
            biphasic=True,  # width2=width1, amp2=-amp1
            width1=5,
            amp1=3000,  # nA
            dt=self.network.dt,
            t_stop=1200.,
            interpulse=200,
            el_id=0,
            t_start=200)
        print(t_ext)
        self.network.enable_extracellular_stimulation(self.extracellular.electrode, t_ext, n=5)


        # run simulation:
        SPIKES = self.network.step(
            probes=[self.extracellular.electrode],
            **self.args.env.network.networkSimulationArguments
        )

        #SPIKES, sim_t = self.network.step(probes=[electrode], **self.args.env.network.networkSimulationArguments)

        if RANK == 0:
            print(self.extracellular.electrode.data['imem'])
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(self.extracellular.electrode.data['imem'][0])
            plt.show()

        exit()



        # simulate the network
        COMM.Barrier()
        SPIKES, sim_t = self.network.step(probes=self.extracellular_models, **self.args.env.network.networkSimulationArguments)
        # Note: SPIKES provides the spike time grouped into populations e.g., [[0, 1, 2, 3] -> PYR, [4]->SST etc]
        COMM.Barrier()



        # if self.args.env.eeg.measure:  # calc EEG
        #     if RANK == 0:
        #         P = self.extracellular_models[1].data['imem']  # numpy array <3, timesteps>
        #         pot_db_4s_top = self.four_sphere_top.get_dipole_potential(P, np.array(self.args.env.network.position))  # Units: mV
        #         eeg_top = np.array(pot_db_4s_top) * 1e-3  # convert units: V
        #
        #         if self.args.experiment.verbose:
        #
        #             print("\n probes data")
        #             print("imem")
        #             print(self.extracellular_models[0].data['imem'])  # membrane current
        #             print("ipas")
        #             print(self.extracellular_models[0].data['ipas'])  # passive current
        #             print("icap")
        #             print(self.extracellular_models[0].data['icap'])  # capacitative current
        #             print("isyn_e")
        #             print(self.extracellular_models[0].data['isyn_e'])  # excitatory synaptic current
        #             print("isyn_i")
        #             print(self.extracellular_models[0].data['isyn_i'])  # inhibitory synaptic current
        #             print("####\n")
        #
        #             print('EEG: ')
        #             print(eeg_top)
        #             print(eeg_top.shape)
        #             print("####\n")
        #
        #             print("\n Cell vmem")
        #             for name in self.network.population_names:
        #                 for cell in self.network.populations[name].cells:
        #                     print(cell.vmem)
        #
        #             print("\n Cell istim")
        #             for name in self.network.population_names:
        #                 for cell in self.network.populations[name].cells:
        #                     if hasattr(cell, 'pointprocesses'):
        #                         print(f"Cell {cell}: pointprocesses = {cell.pointprocesses}")
        #                     else:
        #                         print(f"Cell {cell} has no pointprocesses attached.")

        if RANK == 0:  # calc reward and observation space
            reward = features.reward_func_simple(eeg_top, self.sampling_rate)
            obs = features.feature_space(eeg=eeg_top, fs=self.sampling_rate)
            # print(obs)
            state = np.array([np.ravel(v) for v in obs.values()]).flatten()
            # print(state)
            self.step_count += 1
            info = {'SPIKES': SPIKES, 'I_stim': I_stim, 't_ext': t_ext, 'sim_t': sim_t, 'dom_freq': obs['Dominant Frequency']}
            # print(SPIKES)

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

        self.network.init_simulation(probes=self.extracellular_models, obs_win_len=self.args.env.simulation.obs_win_len,
                                     **self.args.env.network.networkSimulationArguments)
        self.step_count = 0

    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1, 12))
