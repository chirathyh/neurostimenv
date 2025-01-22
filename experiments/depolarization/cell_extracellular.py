import LFPy
import numpy as np
import matplotlib.pyplot as plt

import sys
import time
from mpi4py import MPI
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import neuron
from neuron import h


def simulate(stimulation=False, tstop = 200):
    print('Mechanisms found: ', MAIN_PATH+'/setup/circuits/L23Net/mod/x86_64/special')
    neuron.h('forall delete_section()')
    neuron.load_mechanisms(MAIN_PATH+'/setup/circuits/L23Net/mod/')
    h.load_file(MAIN_PATH+'/setup/circuits/L23Net/net_functions.hoc')
    h.load_file(MAIN_PATH+'/setup/circuits/L23Net/models/biophys_HL23PYR.hoc')
    cellParams = {
            'morphology': MAIN_PATH+'/setup/circuits/L23Net/morphologies/HL23PYR.swc',
            'v_init': -80,  # initial membrane potential, d=-65
            'passive': False,  # initialize passive mechs, d=T, should be overwritten by biophys
            'dt': 0.025,
            'tstart': 0.,
            'tstop': tstop,  # defaults to 100
            'nsegs_method': None,
            'pt3d': True,  # use pt3d-info of the cell geometries switch, d=F
            'delete_sections': False,
            'verbose': False}  # verbose output switch, for some reason doens't work, figure out why}

    cell = LFPy.Cell(**cellParams)
    # Add a synapse on the soma (index 0 of the cell)
    stim_params = {'tau_r_AMPA': 0.3, 'tau_d_AMPA': 3, 'tau_r_NMDA': 2, 'tau_d_NMDA': 65, 'e': 0, 'u0': 0, 'Dep': 670, 'Fac': 17, 'Use': 0.46, 'gmax': 0.004}
    idx = cell.get_rand_idx_area_norm(section='dend', nidx=5)
    for i in idx:
        syn = LFPy.Synapse(cell=cell, idx=i, syntype='ProbAMPANMDA', weight=1, **stim_params)
        syn.set_spike_times(np.array([100.]))

    LFPelectrodeParameters = dict(
        x=np.zeros(1),
        y=np.zeros(1),
        z=[5.],
        N=np.array([[0., 1., 0.] for _ in range(1)]),
        r=5.,
        n=50,
        sigma=0.3,
        method="pointsource")

    electrode = LFPy.RecExtElectrode(cell=cell, **LFPelectrodeParameters)
    if stimulation:
        stim_elec = 0
        I_stim, t_ext = electrode.probe.set_current_pulses(
            n_pulses=2,
            biphasic=True,  # width2=width1, amp2=-amp1
            width1=5,
            amp1=3,  # nA
            dt=cell.dt,
            t_stop=110.,
            interpulse=0,
            el_id=stim_elec,
            t_start=100.)
        cell.enable_extracellular_stimulation(electrode, t_ext, n=5)
    cell.simulate(probes=[electrode], rec_imem=True, rec_vmem=True)
    del cell
    return electrode

electrode_no_stim = simulate(stimulation=False, tstop = 200)
electrode = simulate(stimulation=True, tstop = 200)
plt.figure()
plt.plot(electrode.data[0], 'b', label="Extracellular Stimulation")
plt.plot(electrode_no_stim.data[0], 'r', label="No Stimulation")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Membrane Potential Over Time")
plt.legend()
plt.show()


# # Store the soma membrane potential with extracellular stimulation
# vmem_with_ext = cell.vmem[0, :]
#
# # --- Simulation without Extracellular Stimulation ---
# #create cell
# cell = LFPy.Cell(morphology=MAIN_PATH+'/setup/circuits/L23Net/morphologies/HL23PYR.swc',
#                  tstop=tstop, v_init=-80., celsius=34.)
# # Add a synapse on the soma (index 0 of the cell)
# synapse = LFPy.Synapse(cell, idx=0, syntype='ExpSyn', weight=1, **dict(tau1=0.2, tau2=1.8, e=0.))
# synapse.set_spike_times_w_netstim(noise=1., start=0., number=1E3, interval=10., seed=1234.)  # Time of spike events for the synapse
# #time vector and extracellular field for every segment:
# t_ext = np.arange(cell.tstop / cell.dt+ 1) * cell.dt
# cell.simulate(rec_imem=True, rec_vmem=True)
#
# # Store the soma membrane potential without extracellular stimulation
# vmem_without_ext = cell.vmem[0, :]
#
# # Define the start and end time for the range you want to plot
# start_time = 0  # start time in ms
# end_time = 200  # end time in ms
#
# # Create a mask for the time range
# mask = (t_ext >= start_time) & (t_ext <= end_time)
#
# # Apply the mask to time and voltage arrays
# t_ext_range = t_ext[mask]
# vmem_with_ext_range = vmem_with_ext[mask]
# vmem_without_ext_range = vmem_without_ext[mask]
#
# # Plot the soma membrane potential over the specified time range
# plt.figure(figsize=(8, 4))
# plt.plot(t_ext_range, vmem_with_ext_range, label="With Extracellular Stimulation", color="b")
# plt.plot(t_ext_range, vmem_without_ext_range, label="Without Extracellular Stimulation", color="r", linestyle='--')
# plt.xlabel("Time (ms)")
# plt.ylabel("Membrane Potential (mV)")
# plt.title("Soma Membrane Potential Over Time (With and Without Extracellular Stimulation)")
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot the soma membrane potential over time
# plt.figure(figsize=(8, 4))
# plt.plot(t_ext, cell.vmem[0, :], label="Soma Membrane Potential (mV)", color="b")
# plt.xlabel("Time (ms)")
# plt.ylabel("Membrane Potential (mV)")
# plt.title("Membrane Potential at Soma Over Time")
# plt.legend()
# plt.grid(True)
# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(311)
# ax2 = fig.add_subplot(312)
# ax3 = fig.add_subplot(313)
# eim = ax1.matshow(np.array(cell.v_ext))
# cb1 = fig.colorbar(eim, ax=ax1)
# cb1.set_label('v_ext')
# ax1.axis(ax1.axis('tight'))
# iim = ax2.matshow(cell.imem)
# cb2 = fig.colorbar(iim, ax=ax2)
# cb2.set_label('imem')
# ax2.axis(ax2.axis('tight'))
# vim = ax3.matshow(cell.vmem)
# ax3.axis(ax3.axis('tight'))
# cb3 = fig.colorbar(vim, ax=ax3)
# cb3.set_label('vmem')
# ax3.set_xlabel('tstep')
# plt.show()

