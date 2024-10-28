import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Load the NIfTI file
nifti_file = 'm2m_ernie/T1.nii.gz'
img = nib.load(nifti_file)

# Get the image data as a numpy array
img_data = img.get_fdata()

# Select a slice to visualize
slice_index = img_data.shape[2] // 2  # Take a slice from the middle of the z-axis
slice_data = img_data[:, :, slice_index]

# Define arrow properties
arrowprops = dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=2)

# Plot the slice
plt.figure(figsize=(8, 8))
plt.imshow(np.rot90(slice_data), cmap='gray')
plt.title(f'Slice {slice_index} of {nifti_file}')
plt.axis('off')

# Plot the x, y, z coordinate directions
plt.gca().add_patch(FancyArrowPatch((0.1 * img_data.shape[0], 0.1 * img_data.shape[1]), (0.4 * img_data.shape[0], 0.1 * img_data.shape[1]), **arrowprops))
plt.gca().add_patch(FancyArrowPatch((0.1 * img_data.shape[0], 0.1 * img_data.shape[1]), (0.1 * img_data.shape[0], 0.4 * img_data.shape[1]), **arrowprops))
plt.gca().add_patch(FancyArrowPatch((0.1 * img_data.shape[0], 0.1 * img_data.shape[1]), (0.1 * img_data.shape[0], 0.1 * img_data.shape[1] + 50), **arrowprops))

# Add labels for the coordinate directions
plt.text(0.5 * img_data.shape[0], 0.05 * img_data.shape[1], 'X', color='red', fontsize=12, ha='center')
plt.text(0.05 * img_data.shape[0], 0.5 * img_data.shape[1], 'Y', color='red', fontsize=12, ha='center')
plt.text(0.05 * img_data.shape[0], 0.05 * img_data.shape[1] + 50, 'Z', color='red', fontsize=12, ha='center')

plt.show()
