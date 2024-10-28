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
from env.networkenv import NetworkEnv
from env.electrodes import Electrodes


class NeuronEnv(gym.Env):
    def __init__(self, args, MPI_VAR):
        self.args = args
        self.MPI_VAR = MPI_VAR
        self.sampling_rate = (1 / args.network_dt) * 1000
        self._reset()

    def _step(self, action):
        COMM = self.MPI_VAR['COMM']
        RANK = self.MPI_VAR['RANK']
        COMM.Barrier()
        reward, state, done, info = 0, 0, 0, 0
        I_stim, t_ext, eeg_top = [], [], []

        if self.args.stim:
            I_stim, t_ext = self.electrodes.set_pulse(self.network, self.step_count, action)
            # MEAutility/core/py was changed to get extracellular working for different populations; line 127
            # this steps run into an error when executed through multiple processes.
            COMM.Barrier()
            # # for debugging
            # if RANK==0:
            #     for popname in self.network.populations.keys():
            #         cells = self.network.populations[popname].cells
            #         print(popname) if RANK==0 else None
            #         print(len(cells)) if RANK==0 else None
            self.network.enable_extracellular_stimulation_mpi(self.probes[0], t_ext, n=5)  # if RANK==0 else None
            COMM.Barrier()

        SPIKES, sim_t = self.network.step(probes=self.probes, **self.args.networkSimulationArguments)

        COMM.Barrier()
        if self.args.probes:
            #if RANK==0:
            P = self.probes[1].data['imem']  # numpy array <3, timesteps>
            pot_db_4s_top = self.four_sphere_top.get_dipole_potential(P, self.args.foursphereheadmodel['L23_pos'])
            eeg_top = np.array(pot_db_4s_top) * 1e9

        # state = self.probes[0].data['imem']
        # state = self.eeg_pre_processing(state)

        self.step_count += 1
        info = {'SPIKES': SPIKES, 'I_stim': I_stim, 't_ext': t_ext, 'sim_t': sim_t}
        return eeg_top, reward, done, info

    def eeg_pre_processing(self, data, filter_data=True, ztransform=True):
        # # data = ss.decimate(data, q=16, zero_phase=True)  # down sample?
        if filter_data:
            # b, a = ss.butter(N=2, Wn=0.02, btype='lowpass')
            b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=self.sampling_rate, output='ba')
            data = ss.filtfilt(b, a, data, axis=-1)
        # if ztransform:
        #     # substract the mean of a channel
        #     dataT = data.T - data.mean(axis=1)
        #     data = dataT.T
        return data

    def _close(self):
        ##########################################################################
        # customary cleanup of object references - the psection() function may not
        # write correct information if NEURON still has object references in memory
        # even if Python references has been deleted. It will also allow the script
        # to be run in successive fashion.
        ##########################################################################
        self.network.pc.gid_clear()  # allows assigning new gids to threads
        self.electrodes = None
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
        # init
        self.network = NetworkEnv(**self.args.networkParameters)

        if self.args.network_type == 'HL23PYR':
            from utils.HL23PYR.utils import create_populations_connect
            create_populations_connect(self.network, self.args)  # create populations, setup synapses, and connections.
            self.electrodes = Electrodes(self.args)
            self.probes = self.electrodes.get_probes()
            self.network.init_simulation(probes=self.probes, rl_args=self.args, **self.args.networkSimulationArguments)
            self.step_count = 0

        if self.args.network_type == 'L23Net':
            from utils.L23Net.utils import setup_network
            setup_network(self.network, self.args, self.MPI_VAR)
            self.probes = None
            self.four_sphere_top = None
            if self.args.probes:
                self.four_sphere_top = FourSphereVolumeConductor(self.args.foursphereheadmodel['EEG_sensor'],
                                                        self.args.foursphereheadmodel['radii'],
                                                        self.args.foursphereheadmodel['sigmas'])
                self.electrodes = Electrodes(self.args)
                self.probes = self.electrodes.get_probes()
            self.network.init_simulation(probes=self.probes, rl_args=self.args, **self.args.networkSimulationArguments)
            self.step_count = 0

    @property
    def action_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(16,))
