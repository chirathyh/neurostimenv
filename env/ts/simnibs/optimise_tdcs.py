import simnibs

# Initialize structure
opt = simnibs.opt_struct.TDCSoptimize()
# Select the leadfield file
opt.leadfield_hdf = 'leadfield/ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5'
# Select a name for the optimization
opt.name = 'optimization/single_target'

# Select a maximum total current (in A)
opt.max_total_current = 2e-3
# Select a maximum current at each electrodes (in A)
opt.max_individual_current = 1e-3
# Select a maximum number of active electrodes (optional)
opt.max_active_electrodes = 2

# Define optimization target
target = opt.add_target()
# Position of target, in subject space!
# please see tdcs_optimize_mni.py for how to use MNI coordinates
target.positions = [0.0402640, -0.00945882, -1.4570916]
# Intensity of the electric field (in V/m)
target.intensity = 0.2

# Run optimization
simnibs.run_simnibs(opt)
