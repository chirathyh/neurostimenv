import simnibs

opt = simnibs.opt_struct.TDCSoptimize()  # Initialize structure
opt.leadfield_hdf = 'leadfield/ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5'  # Select the leadfield file
opt.name = 'optimization/single_target'  # Select a name for the optimization

opt.max_total_current = 2e-3  # Select a maximum total current (in A)
opt.max_individual_current = 1e-3  # Select a maximum current at each electrodes (in A)
opt.max_active_electrodes = 2  # Select a maximum number of active electrodes (optional)

target = opt.add_target()  # Define optimization target

target.positions = simnibs.mni2subject_coords([-41, 41, 16], 'm2m_ernie')  # Position of target, in subject space!
# I have used left DLPFC in MNI space based on below:
# https://thejournalofheadacheandpain.biomedcentral.com/articles/10.1186/s10194-018-0849-z/tables/2

target.directions = None
target.intensity = 0.2  # Intensity of the field based on leadfiled: could be E (V/m) or J (A/m2)

simnibs.run_simnibs(opt)  # Run optimization

# simnibs_python optimise_tdcs.py
