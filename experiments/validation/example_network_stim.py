import os
import sys
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.signal as ss
import scipy.stats as st
import h5py
from mpi4py import MPI
import neuron
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, \
    CurrentDipoleMoment
from neuron import units
from neuron import h
from lfpykit.eegmegcalc import FourSphereVolumeConductor
from utils.utils import generate_spike_train
# set up MPI variables:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell and synapse locations and random synapse
# activation times
GLOBALSEED = 1234
np.random.seed(GLOBALSEED + RANK)

##########################################################################
# Function declarations
##########################################################################
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


##########################################################################
# Set up shared and population-specific parameters
##########################################################################
# relative path for simulation output:
OUTPUTPATH = 'example_network_stim_outputx'

# class NetworkCell parameters:
cellParameters = dict(
    morphology='BallAndStick.hoc',
    templatefile='BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    delete_sections=False,
    dt=2**-4,
    tstop=2400,
)

# class NetworkPopulation parameters:
populationParameters = dict(
    Cell=NetworkCell,
    cell_args=cellParameters,
    pop_args=dict(
        radius=100.,
        loc=0.,
        scale=20.),
    rotation_args=dict(x=0., y=0.),
)

# class Network parameters:
networkParameters = dict(
    dt=2**-4,
    tstop=1200.,
    v_init=-65.,
    celsius=36.5,
    OUTPUTPATH=OUTPUTPATH
)

# class RecExtElectrode parameters:
electrodeParameters = dict(
    x=np.array([-100, 0, 100, -100, 0, 100, -100, 0, 100]),
    y=np.array([100, 100, 100, 0, 0, 0, -100, -100, -100]),
    z=np.zeros(9),
    N=np.array([[0., 0., 1.] for _ in range(9)]),
    r=20.,  # 5um radius
    n=50,  # nb of discrete point used to compute the potential
    sigma=1,  # conductivity S/m
    method="linesource"
)

# method Network.simulate() parameters:
networkSimulationArguments = dict(
    rec_pop_contributions=True,
    to_memory=True,
    to_file=False
)

# population names, sizes and connection probability:
population_names = ['E', 'I']
population_sizes = [32, 8]
connectionProbability = [[0.1, 0.1], [0.1, 0.1]]

# synapse model. All corresponding parameters for weights,
# connection delays, multapses and layerwise positions are
# set up as shape (2, 2) nested lists for each possible
# connection on the form:
# [["E:E", "E:I"],
#  ["I:E", "I:I"]].
synapseModel = neuron.h.Exp2Syn
# synapse parameters
synapseParameters = [[dict(tau1=0.2, tau2=1.8, e=0.),
                      dict(tau1=0.2, tau2=1.8, e=0.)],
                     [dict(tau1=0.1, tau2=9.0, e=-80.),
                      dict(tau1=0.1, tau2=9.0, e=-80.)]]
# synapse max. conductance (function, mean, st.dev., min.):
weightFunction = np.random.normal
weightArguments = [[dict(loc=0.001, scale=0.0001),
                    dict(loc=0.001, scale=0.0001)],
                   [dict(loc=0.01, scale=0.001),
                    dict(loc=0.01, scale=0.001)]]
minweight = 0.
# conduction delay (function, mean, st.dev., min.):
delayFunction = np.random.normal
delayArguments = [[dict(loc=1.5, scale=0.3),
                   dict(loc=1.5, scale=0.3)],
                  [dict(loc=1.5, scale=0.3),
                   dict(loc=1.5, scale=0.3)]]
mindelay = 0.3
multapseFunction = np.random.normal
multapseArguments = [[dict(loc=2., scale=.5), dict(loc=2., scale=.5)],
                     [dict(loc=5., scale=1.), dict(loc=5., scale=1.)]]
# method NetworkCell.get_rand_idx_area_and_distribution_norm
# parameters for layerwise synapse positions:
synapsePositionArguments = [[dict(section=['soma', 'apic'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=0., scale=100.),
                                           dict(loc=500., scale=100.)],
                                  funweights=[0.5, 1.]
                                  ) for _ in range(2)],
                            [dict(section=['soma', 'apic'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=0., scale=100.),
                                           dict(loc=100., scale=100.)],
                                  funweights=[1., 0.5]
                                  ) for _ in range(2)]]

if __name__ == '__main__':
    ##########################################################################
    # Main simulation
    ##########################################################################
    # create directory for output:
    if RANK == 0:
        if not os.path.isdir(OUTPUTPATH):
            os.mkdir(OUTPUTPATH)
        # remove old simulation output if directory exist
        else:
            for fname in os.listdir(OUTPUTPATH):
                os.unlink(os.path.join(OUTPUTPATH, fname))
    COMM.Barrier()

    # instantiate Network:
    network = Network(**networkParameters)

    # create E and I populations:
    for name, size in zip(population_names, population_sizes):
        network.create_population(name=name, POP_SIZE=size,
                                  **populationParameters)

        syn_activity = True
        if syn_activity:
            for cell in network.populations[name].cells:
                idx = cell.get_rand_idx_area_norm(section='allsec', nidx=64)
                for i in idx:
                    syn = Synapse(cell=cell, idx=i, syntype='Exp2Syn',
                                  weight=0.001,
                                  **dict(tau1=0.2, tau2=1.8, e=0.))
                    # syn.set_spike_times_w_netstim(interval=50.,
                    #                               seed=np.random.rand() * 2**32 - 1
                    #                               )
                    syn.set_spike_times(generate_spike_train(interval=50.))

    # create connectivity matrices and connect populations:
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            # boolean connectivity matrix between pre- and post-synaptic
            # neurons in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=connectionProbability[i][j]
            )

            # connect network:
            (conncount, syncount) = network.connect(
                pre=pre, post=post,
                connectivity=connectivity,
                syntype=synapseModel,
                synparams=synapseParameters[i][j],
                weightfun=weightFunction,
                weightargs=weightArguments[i][j],
                minweight=minweight,
                delayfun=delayFunction,
                delayargs=delayArguments[i][j],
                mindelay=mindelay,
                multapsefun=multapseFunction,
                multapseargs=multapseArguments[i][j],
                syn_pos_args=synapsePositionArguments[i][j],
                save_connections=False,
            )


    four_sphere_top = FourSphereVolumeConductor(np.array([[0., 0., 90000]] ), [79000., 80000., 85000., 90000.], [0.3, 1.5, 0.015, 0.3])
    electrode = RecExtElectrode(cell=None, **electrodeParameters)
    stim_elec = 4
    step_size = 500.
    t_start_stim = 0.
    I_stim, t_ext = np.array([]), np.array([])
    current_dipole_moment = CurrentDipoleMoment(cell=None)
    probes = [electrode, current_dipole_moment]
    stimulate = True

    def custom_simulations(stimulate, electrode, current_dipole_moment, four_sphere_top,
                           network, networkSimulationArguments, stim_elec, I_stim, t_ext):
        if stimulate:
            electrode.probe.set_current(stim_elec, I_stim)
            network.enable_extracellular_stimulation(electrode, t_ext, n=5)

        print(f"Rank {RANK} starting simulation", flush=True)
        SPIKES = network.simulate(probes=[electrode, current_dipole_moment],**networkSimulationArguments)
        print(f"Rank {RANK} completed simulation", flush=True)

        # exit()
        P = current_dipole_moment.data['imem']  # numpy array <3, timesteps>
        pot_db_4s_top = four_sphere_top.get_dipole_potential(P, np.array([-10., 0., 0.]))  # Units: mV
        eeg_top = np.array(pot_db_4s_top) * 1e-3  # convert units: V
        return eeg_top



    network.tstart = 0.
    network.tstop = 500.
    sim_data = []

    # test n loops
    for i in range(0, 2):

        I_stim_cur, t_ext_cur = customise_set_current_pulses(n_pulses=20, biphasic=True, width1=5, amp1=10000,
                                                             dt=network.dt, interpulse=200, t_stop=network.tstop, t_start=t_start_stim)
        I_stim = np.concatenate((I_stim, I_stim_cur))
        t_ext = np.concatenate((t_ext, t_ext_cur))
        eeg = custom_simulations(stimulate, electrode, current_dipole_moment, four_sphere_top,
                               network, networkSimulationArguments, stim_elec, I_stim, t_ext)
        sim_data.append(eeg)
        sim_data.append([I_stim])
        t_start_stim += step_size
        network.tstop += step_size


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



    # test reprodicibility.
    # for i in range(0, 2):
    #     I_stim, t_ext = np.array([]), np.array([])
    #     network.tstart = 0.
    #     network.tstop = 500.
    #     I_stim_cur, t_ext_cur = customise_set_current_pulses(n_pulses=20, biphasic=True, width1=5, amp1=10000,
    #                                                          dt=network.dt, interpulse=200, t_stop=network.tstop, t_start=t_start_stim)
    #     I_stim = np.concatenate((I_stim, I_stim_cur))
    #     t_ext = np.concatenate((t_ext, t_ext_cur))
    #     custom_simulations(stimulate, electrode, current_dipole_moment, four_sphere_top,
    #                            network, networkSimulationArguments, stim_elec, I_stim, t_ext)
