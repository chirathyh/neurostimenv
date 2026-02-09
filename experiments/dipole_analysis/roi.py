import numpy as np
import os.path as op
import mne
from scipy.spatial import cKDTree

# sample subject
subjects_dir = '/home/chirath/mne_data/MNE-sample-data/subjects'  # FreeSurfer subject dir (e.g. sample/subjects)
subject = 'sample' # subject name as in subjects_dir
trans = '/home/chirath/mne_data/MNE-sample-data/MEG/sample/sample_audvis_raw-trans.fif'

# transformations
head_to_mri = mne.read_trans(trans)
mri_to_mni = mne.read_talxfm(subject=subject, subjects_dir=subjects_dir)
head_to_mni = mne.transforms.combine_transforms(head_to_mri, mri_to_mni, "head", "mni_tal")
mni_to_head = mne.transforms.invert_transform(head_to_mni)

# example
head_coords = np.array([-0.02241, -0.02314, 0.06732])
mni_coords = mne.head_to_mni(head_coords, subject=subject, mri_head_t=head_to_mri, subjects_dir=subjects_dir)
print("Testing coordinate transforms")
print("head      (m) ", head_coords)
print("head->mni (mm)", mni_coords)
print("mni->head (m) ", mne.transforms.apply_trans(mni_to_head, mni_coords / 1000))

# Identify MNI coordinates; MNI is given in mm
# Brainnetome Atlas: https://pmc.ncbi.nlm.nih.gov/articles/PMC4961028/pdf/bhw157.pdf

dlpfc = np.array([-41., 41., 16.])
print('left-DLPFC')
print("mni->head (m) ", mne.transforms.apply_trans(mni_to_head, dlpfc / 1000))

right_spc = np.array([20.,-71.,50.])
print(' Right Superior Parietal Cortex (RSPC)')  #Nucleus Accumbens (NAcc)
print("mni->head (m) ", mne.transforms.apply_trans(mni_to_head, right_spc / 1000))
# The Neuroanatomical Basis for Posterior Superior Parietal Lobule Control Lateralization of Visuospatial Attention

# Sample output
# left-DLPFC
# mni->head (m)  [-0.04356633  0.08047897  0.05652746]
#  Right Superior Parietal Cortex (RSPC)
# mni->head (m)  [ 0.01388544 -0.02582726  0.11364826]
