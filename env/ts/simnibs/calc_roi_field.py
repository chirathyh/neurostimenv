from simnibs import read_msh
from simnibs import Msh
import simnibs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv


head_mesh = read_msh('tacs_simu_4mA_8Hz/ernie_TDCS_1_scalar.msh')
#head_mesh = read_msh('optimization/single_target.msh')
gray_matter = head_mesh.crop_mesh(simnibs.ElementTags.GM)
# scalp = head_mesh.crop_mesh(simnibs.ElementTags.SCALP)

# Coordinates of the target ROI
# ernie_coords = simnibs.mni2subject_coords([-41, 41, 16], 'm2m_ernie')  # F3: -58, 35, 50
ernie_coords = [-37.1796,  68.4849,  34.9607]  # the word/subject coordinate are present in the leadfield calculation

# Radius of the ROI
r = 10.0 # we will use a sphere of radius 10 mm
# Electric fields are defined in the center of the elements

elm_centers = gray_matter.elements_baricenters()[:]  # get element centers
roi = np.linalg.norm(elm_centers - ernie_coords, axis=1) < r  # determine the elements in the ROI
elm_vols = gray_matter.elements_volumes_and_areas()[:]  # get the element volumes, we will use those for averaging
num_elements_in_roi = np.sum(roi)
print(f'Number of elements in ROI at {ernie_coords}: {num_elements_in_roi}')
# scalp.add_element_field(roi, 'roi')
# scalp.view(visible_fields='roi').show()

# Plot the ROI
gray_matter.add_element_field(roi, 'roi')
gray_matter.view(visible_fields='roi').show()

print('\nfields available')
print(gray_matter.field)

field_name = 'magnJ'
field = gray_matter.field[field_name][:]
mean_magn = np.average(field[roi], weights=elm_vols[roi])  # Calculate the mean
print('mean ', field_name, ' in M1 ROI: ', mean_magn)

# calculating the current at the roi
# --- convert to total current in ROI -------------------------------
# radius in meters
r_m = r / 1000.0
# cross-sectional area (circle) perpendicular to current flow
cross_area = np.pi * r_m**2                          # m²
total_current = mean_magn * cross_area              # A

print(f"Estimated total current through a {r}mm‑radius ROI: "
      f"{total_current:.3e} A")

# simnibs_python calc_roi_field.py
