from simnibs import sim_struct, run_simnibs
import os
from copy import deepcopy


def run_tacs(amp=1, freq=2):
    OUTPUT = 'tacs_simu_' + str(amp) + 'mA_' + str(freq) + 'Hz'
    S = sim_struct.SESSION()
    S.subpath = 'm2m_ernie'  # head mesh
    S.pathfem = OUTPUT  # Directory for the simulation
    S.fields = 'eEjJ'
    S.map_to_surf=True
    S.open_in_gmsh=True # open results once they are ready (set to False if you are annoyed by the popup windows)

    ### Define Condition A@TACSchallenge
    ConditionA = S.add_tdcslist()
    ConditionA.currents   = [amp, -amp]
    ConditionA.frequency  = freq     # in Hz
    ConditionA.phase      = 0.0    # radians, optional

    # AF3
    AF3 = ConditionA.add_electrode()
    AF3.channelnr  = 1
    AF3.centre     = 'AF3'
    AF3.shape      = 'ellipse'
    AF3.dimensions = [25, 25]        # mm
    AF3.thickness  = [1.5, 1.5]      # gel, rubber mm

    # FC5
    FC5 = ConditionA.add_electrode()
    FC5.channelnr  = 2
    FC5.centre     = 'FC5'
    FC5.shape      = 'ellipse'
    FC5.dimensions = [25, 25]
    FC5.thickness  = [1.5, 1.5]

    run_simnibs(S)


run_tacs(amp=1, freq=100)
# simnibs_python run_simnibs_tacs.py
