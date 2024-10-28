from simnibs import read_msh
from simnibs import Msh
import simnibs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

head_mesh = read_msh('tdcs_simu1/ernie_TDCS_1_scalar.msh')
#head_mesh = read_msh('results/tdcs/Fz_I2_0.1/ernie_TDCS_1_scalar.msh')
gray_matter = head_mesh.crop_mesh(simnibs.ElementTags.GM)

## Define the ROI

# Define M1 from MNI coordinates (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2034289/)
# the first argument is the MNI coordinates
# the second argument is the subject "m2m" folder

#ernie_coords = simnibs.mni2subject_coords([-37, -21, 58], 'm2m_ernie')
#print(ernie_coords)
#ernie_coords = simnibs.mni2subject_coords([0, 0, 78.2], 'm2m_ernie')
ernie_coords = [0.0402640, -0.00945882, -1.4570916] #[0, 0, 78.2]

# we will use a sphere of radius 10 mm
r = 15.

# Electric fields are defined in the center of the elements
# get element centers
elm_centers = gray_matter.elements_baricenters()[:]
# determine the elements in the ROI
roi = np.linalg.norm(elm_centers - ernie_coords, axis=1) < r
# get the element volumes, we will use those for averaging
elm_vols = gray_matter.elements_volumes_and_areas()[:]

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

field_name = 'J' #magnE' #'magnE'
field = gray_matter.field[field_name][:]
print(field[roi].shape)
# Extract field components for each direction
print('Field shape:', field.shape)
field_x = field[:, 0]
field_y = field[:, 1]
field_z = field[:, 2]
# Calculate mean for each direction within the ROI
mean_field_x = np.average(field_x[roi], weights=elm_vols[roi])
mean_field_y = np.average(field_y[roi], weights=elm_vols[roi])
mean_field_z = np.average(field_z[roi], weights=elm_vols[roi])
print('Mean J in x direction in ROI:', mean_field_x)
print('Mean J in y direction in ROI:', mean_field_y)
print('Mean J in z direction in ROI:', mean_field_z)


## Get field and calculate the mean
# get the field of interest
field_name = 'magnE'
field = gray_matter.field[field_name][:]
# Calculate the mean
mean_magnE = np.average(field[roi], weights=elm_vols[roi])
print('mean ', field_name, ' in M1 ROI: ', mean_magnE)

field_name = 'E' #magnE' #'magnE'
field = gray_matter.field[field_name][:]
print(field[roi].shape)
# Extract field components for each direction
print('Field shape:', field.shape)
field_x = field[:, 0]
field_y = field[:, 1]
field_z = field[:, 2]
# Calculate mean for each direction within the ROI
mean_field_x = np.average(field_x[roi], weights=elm_vols[roi])
mean_field_y = np.average(field_y[roi], weights=elm_vols[roi])
mean_field_z = np.average(field_z[roi], weights=elm_vols[roi])
print('Mean E in x direction in ROI:', mean_field_x)
print('Mean E in y direction in ROI:', mean_field_y)
print('Mean E in z direction in ROI:', mean_field_z)

exit()

# Calculate the mean
mean_magnE = np.average(field[roi], weights=elm_vols[roi])
print('mean ', field_name, ' in M1 ROI: ', mean_magnE)



exit()


M = Msh(fn='tdcs_simu/ernie_TDCS_1_scalar.msh')
print(M.nodedata)
print(M.nodes)

print('\nelementt info')
print(M.elm)

print('\nelement data')
print(M.elmdata)

print('\nfields available')
print(M.field)


for data in M.elmdata:  # two types of elements E, magnE
    print(data.field_name)
    print('values:')
    print(len(data.value))

print('\n bari centers')
bari_center = M.elements_baricenters()
print(bari_center)
print(bari_center.nr)
print(dir(bari_center))
