from simnibs import read_msh
import simnibs
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the head mesh and crop to gray matter
head_mesh = read_msh('tdcs_simu/ernie_TDCS_1_scalar.msh')
gray_matter = head_mesh.crop_mesh(simnibs.ElementTags.GM)

# Define dummy electrode coordinates (in subject space coordinates)
electrode_coords = np.array([
    [0.05, -0.01, -1.45],
    [0.03, 0.02, -1.46],
    [-0.04, -0.03, -1.44],
    # Add more electrodes as needed
])

# Convert the SimNIBS head mesh to PyVista format
def convert_to_pyvista(mesh):
    # Extract nodes and elements
    nodes = mesh.nodes
    elements = mesh.elm
    # Create PyVista PolyData
    poly_data = pv.PolyData(nodes)
    poly_data.faces = np.hstack([elements[:, 0], elements[:, 1], elements[:, 2]]).astype(np.int32)
    return poly_data

# Convert SimNIBS head mesh to PyVista format
head_mesh_pyvista = convert_to_pyvista(head_mesh)

# Create a PyVista plotter object
plotter = pv.Plotter()

# Add the head mesh to the plotter
plotter.add_mesh(head_mesh_pyvista, color='lightgray', opacity=0.7)

# Plot EEG electrodes
electrode_points = pv.PolyData(electrode_coords)
plotter.add_points(electrode_points, color='red', point_size=10, render_points_as_spheres=True)

# Add axes and labels
plotter.add_axes()

# Show the plot
plotter.show()
