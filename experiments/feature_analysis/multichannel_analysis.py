import matplotlib.pyplot as plt
import numpy as np
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_anat
from scipy.spatial import cKDTree
import mne
from mne.evoked import combine_evoked
from mne.forward import make_forward_dipole
from mne.simulation import simulate_evoked


def roi_to_surface_frank(roi_xyzs,vtx_xyzs,faces):
    faces_xyz = np.mean(vtx_xyzs[faces],axis=1)
    face_nearestroi_inds = cKDTree(faces_xyz).query(roi_xyzs, k=1)[1]
    vtx_nearestroi_inds = cKDTree(vtx_xyzs).query(roi_xyzs, k=1)[1]

    return vtx_nearestroi_inds,face_nearestroi_inds

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
fname_ave = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"
fname_cov = data_path / "MEG" / "sample" / "sample_audvis-cov.fif"
#fname_bem = subjects_dir / "sample" / "bem" / "sample-5120-bem-sol.fif"
fname_trans = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
fname_surf_lh = subjects_dir / "sample" / "surf" / "lh.white"

raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"

# Load your dipole time series data (assumed to be shape (3, 32001))
dipole_data = np.load("dipole.npy")  # (3, 32001) where 3 channels are your dipoles


subject = 'sample'
# The transformation file obtained by coregistration
info = mne.io.read_info(raw_fname)

# source space is just a mesh of grid points of possible dipole origins
src = mne.setup_source_space(subject,
                             spacing='oct4', # increase to oct6, 8, etc for more realistic output
                             add_dist=True,
                             subjects_dir=subjects_dir,verbose=False)


### Make BEM ###

print('\n BEM model: ')
model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=(0.3, 0.006, 0.3),
                           subjects_dir=subjects_dir,verbose=False)

print('\n BEM Solution: ')
fname_bem = mne.make_bem_solution(model,verbose=False)
trans = mne.read_trans(fname_trans)

evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',baseline=(None, 0),verbose=False)
evoked.pick_types(meg=False, eeg=True)

evoked.info['sfreq'] = 400
evoked.info['lowpass'] = 200


# dip_locs = [[-0.07074824+.0131,0.05937758,0.0837867],
#             [-0.05889541+.0131,0.06404063-.0044,0.10122117-.01],
#             [-0.0378412+.0131,0.06372997-.017,0.11839715],
#             [-0.0506296+.0131,0.08445242-.007,  0.08830147],
#             [-0.04332023+.0131,  0.02930477-.014,  0.13122222-.002]]
coord = [-0.07074824+.0131,0.05937758,0.0837867]
coord = np.array([coord])
vtx_inds,face_inds = roi_to_surface_frank(coord,src[0]['rr'],src[0]['tris'])

ori = src[0]['nn'][vtx_inds]
ori = ori.reshape(1,3)
