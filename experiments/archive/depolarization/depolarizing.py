import sys
import time
from mpi4py import MPI
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import os
import numpy as np
import matplotlib.pyplot as plt
import LFPy
from lfpykit import CellGeometry, RecExtElectrode

def simulate(stimulation=False):
    cellParameters = {
        'morphology': MAIN_PATH+'/setup/circuits/ballnstick/BallAndStick.hoc',
        'v_init': -65,                         # initial voltage
        'cm': 1.0,                             # membrane capacitance
        'Ra': 150,                             # axial resistivity
        'passive': True,                       # insert passive channels
        'passive_parameters': {"g_pas": 1./3E4, "e_pas": -65}, # passive params
        'dt': 2**-4,                           # simulation time res
        'tstart': 0.,                          # start t of simulation
        'tstop': 50.,                          # end t of simulation
    }
    cell = LFPy.Cell(**cellParameters)

    synapseParameters = {
        'idx': cell.get_closest_idx(x=0, y=0, z=800), # segment
        'e': 0,                                       # reversal potential
        'syntype': 'ExpSyn',                          # synapse type
        'tau': 2,                                     # syn. time constant
        'weight': 0.01,                               # syn. weight
        'record_current': True                        # syn. current record
    }
    synapse = LFPy.Synapse(cell, **synapseParameters)
    synapse.set_spike_times(np.array([20.]))

    N = np.empty((16, 3))
    for i in range(N.shape[0]):
        N[i,] = [1, 0, 0]  # normal vectors

    electrodeParameters = {  # parameters for RecExtElectrode class
        'sigma': 0.3,              # Extracellular potential
        'x': np.zeros(16) + 25,    # Coordinates of electrode contacts
        'y': np.zeros(16),
        'z': np.linspace(-500, 1000, 16),
        'n': 20,
        'r': 10,
        'N': N,
    }
    electrode = RecExtElectrode(cell, **electrodeParameters)

    # cell.simulate(probes=[electrode], rec_imem=True)
    # no_Stim = electrode.data[0]
    if stimulation:
        stim_elec = 4
        I_stim, t_ext = electrode.probe.set_current_pulses(
                n_pulses=1,
                biphasic=True,  # width2=width1, amp2=-amp1
                width1=5,
                amp1=3000,  # nA
                dt=cell.dt,
                t_stop=30,
                interpulse=1,
                el_id=stim_elec,
                t_start=10)
        cell.enable_extracellular_stimulation(electrode, t_ext, n=5)

    cell.simulate(probes=[electrode], rec_imem=True)
    del cell
    return electrode

electrode_no_stim = simulate(stimulation=False)
electrode = simulate(stimulation=True)
plt.figure()
plt.plot(electrode.data[0], 'b')
plt.plot(electrode_no_stim.data[0], 'r')
# plt.plot(no_Stim, 'r')
# plt.plot(electrode.data[2], 'k')
# plt.plot(electrode.data[3], 'm')
plt.show()

