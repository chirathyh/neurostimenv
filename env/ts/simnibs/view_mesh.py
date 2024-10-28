from simnibs import read_msh
import subprocess

# visualise a mesh file
#mesh_file = 'leadfield/ernie_electrodes_EEG10-10_UI_Jurak_2007.msh'
# mesh_file = 'leadfield/ernie_ROI.msh'
# mesh_file = 'optimization/single_target_el_currents.geo'
# mesh = read_msh(mesh_file)
# mesh.view().show()


# Path to the .geo file
# geo_file = 'optimization/single_target_el_currents.geo'
# msh_file = 'optimization/single_target.msh'
# # Run Gmsh to generate the mesh from the .geo file
# subprocess.run(['gmsh', geo_file, '-3', '-o', msh_file])
#
# # Load the generated .msh file in SimNIBS
# mesh = read_msh(msh_file)
#
# # Visualize the mesh
# mesh.view().show()


import os
import gmsh

# Initialize Gmsh
gmsh.initialize()

# Path to your .geo file
geo_file = 'tdcs_simu1/ernie_TDCS_1_el_currents.geo'

# Open the .geo file
gmsh.open(geo_file)

# Set options to optimize the visualization
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Geometry.PointNumbers", 1)
gmsh.option.setNumber("Geometry.LineNumbers", 1)

# Run the Gmsh GUI to visualize the geometry
gmsh.fltk.run()

# Finalize Gmsh (close the GUI)
gmsh.finalize()
