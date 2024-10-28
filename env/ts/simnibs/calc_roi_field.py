from simnibs import read_msh
from simnibs import Msh
import simnibs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

head_mesh = read_msh('tdcs_simu1/ernie_TDCS_1_scalar.msh')
gray_matter = head_mesh.crop_mesh(simnibs.ElementTags.GM)
# scalp = head_mesh.crop_mesh(simnibs.ElementTags.SCALP)
#ernie_coords = simnibs.mni2subject_coords([-58, 35, 50], 'm2m_ernie')  # F3: -58, 35, 50

#ernie_coords = simnibs.subject2mni_coords([0., 0., 78.2], 'm2m_ernie')  # F3: -58, 35, 50
ernie_coords = [78.2, 0, 0]

# Final
#ernie_coords = simnibs.mni2subject_coords([-41, 41, 16], 'm2m_ernie')  # F3: -58, 35, 50
# ernie_coords = [-39.05980886279742,85.04069743317085,59.190744338211545]  # F3 electrode for subject
# print(ernie_coords)


#ernie_coords = [0.0007629211490047538, 0.0005164314240723371, -1.0247522935997254] # cartesian coordinates
# ernie_coords = [0., 0., 90.2]




# Electric fields are defined in the center of the elements
r = 20.0 # we will use a sphere of radius 10 mm

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
# Calculate the mean
mean_magnE = np.average(field[roi], weights=elm_vols[roi])
print('mean ', field_name, ' in M1 ROI: ', mean_magnE)
