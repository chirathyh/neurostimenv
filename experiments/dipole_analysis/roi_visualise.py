import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import mne
from mne.datasets import sample
import os.path as op


subjects_dir = '/home/chirath/mne_data/MNE-sample-data/subjects'  # FreeSurfer subject dir (e.g. sample/subjects)
sample_dir = '/home/chirath/mne_data/MNE-sample-data/MEG/sample'
subject = 'sample' # subject name as in subjects_dir

# Load transform
trans_fname = op.join(sample_dir, 'sample_audvis_raw-trans.fif')
trans = mne.read_trans(trans_fname)
head_to_mri = trans

brain_kwargs = dict(alpha=0.1, background="white", cortex="low_contrast")
brain = mne.viz.Brain("sample", subjects_dir=subjects_dir, **brain_kwargs)  # default mm

head_point_m = np.array([-0.04356633,  0.08047897,  0.05652746])  # example head coords (meters)
mri_point_m = mne.transforms.apply_trans(head_to_mri['trans'], head_point_m.reshape(1, 3)) #[0]
mri_point_m = mri_point_m * 1000

# print("head_point_m (m):", head_point_m)
# print("mri_point_m (mm):", mri_point_m)
# print("brain.units:", brain._units)

head_point_m2 = np.array([0.01388544, -0.02582726,  0.11364826])  # example head coords (meters)
mri_point_m2 = mne.transforms.apply_trans(head_to_mri['trans'], head_point_m2.reshape(1, 3)) #[0]
mri_point_m2 = mri_point_m2 * 1000

brain.add_foci(mri_point_m, hemi='lh', color='blue', scale_factor=0.5, coords_as_verts=False, name="FociA")
brain.plotter.add_point_labels(mri_point_m.tolist(), ['L-DLPFC'], font_size=24, point_size=12, text_color='blue', always_visible=True)
# brain.plotter.render()


brain.add_foci(mri_point_m2, hemi='rh', color='green', scale_factor=0.5, coords_as_verts=False, name="FociB")
brain.plotter.add_point_labels(mri_point_m2.tolist(), ['R-SPC'], font_size=24, point_size=12, text_color='green', always_visible=True)
brain.plotter.render()

# brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))
brain.show_view('lat')
# brain.save_image('brain_with_BA44.png')

input("Press Enter to close the 3D viewer and exit...")
