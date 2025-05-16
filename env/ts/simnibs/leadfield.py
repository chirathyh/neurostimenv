from simnibs import sim_struct, run_simnibs

tdcs_lf = sim_struct.TDCSLEADFIELD()
tdcs_lf.subpath = 'm2m_ernie'  # subject folder
tdcs_lf.pathfem = 'leadfield_J'  # output directory
tdcs_lf.field = 'J'

# Uncoment to use the pardiso solver
#tdcs_lf.solver_options = 'pardiso'
# This solver is faster than the default. However, it requires much more memory (~12 GB)

run_simnibs(tdcs_lf)
