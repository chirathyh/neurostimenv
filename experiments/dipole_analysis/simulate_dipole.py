"""
Simulate dipole data and visualize the results.
We use the sample OR fsaverage adult template MRI; place the dipole in a target location and model EEG using a standard 10-20 EEG montage.
"""

import numpy as np
import mne
import os.path as op
from mne.datasets import eegbci, fetch_fsaverage
from scipy.spatial import cKDTree
from mne.forward import make_forward_dipole
from mne.simulation import simulate_evoked
import matplotlib.pyplot as plt
import math

# average subject
# fs_dir = fetch_fsaverage(verbose=True)
# subjects_dir = fs_dir.parent
# subject = "fsaverage"
# trans = "fsaverage"  # MNE has a built-in fsaverage transformation

# sample subject
subjects_dir = '/home/chirath/mne_data/MNE-sample-data/subjects'  # FreeSurfer subject dir (e.g. sample/subjects)
subject = 'sample' # subject name as in subjects_dir
trans = '/home/chirath/mne_data/MNE-sample-data/MEG/sample/sample_audvis_raw-trans.fif'

# source-space / BEM surfaces: MRI
# head-space: EEG/MEG sensors
# trans file: head -> MRI

# The source-space (brain location where we add the dipoles - MRI space), BEM model (boundary element model descrbing the head as a set o flcosed surfaces),
# and the BEM solution (the numerical object ready to be used by forward solvers).
src = mne.setup_source_space(subject, spacing='oct4', add_dist=True, subjects_dir=subjects_dir, verbose=False) # spacing: increase to oct6, 8, etc for more realistic output
model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir, verbose=False)
fname_bem = mne.make_bem_solution(model, verbose=False)


# Setting up a dummy Evoked to hold EEG channel info.
sfreq = 100  # 100 Hz
montage = mne.channels.make_standard_montage("standard_1020")
exclude = ['T7', 'T8', 'P7', 'P8'] # 'T3', 'T5', 'T4', 'T6'  https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
keep_chs = [ch for ch in montage.ch_names if ch not in exclude]
print("EEG channels used:", keep_chs)
info = mne.create_info(keep_chs, sfreq, ch_types="eeg")
info.set_montage(montage)
evoked = mne.EvokedArray(np.zeros((len(keep_chs), 1)), info, tmin=0.0)
evoked.pick_types(meg=False, eeg=True)
# evoked.filter(l_freq=None, h_freq=199.9)

# Load dipole moments from file and place.
diplole_moments_file = "../../results/dipole/DIPOLE_HEALTHY_0.csv"  # 40,000 Hz
down_sample_factor = int(40000/sfreq)  # to get to sfreq
P = np.array(np.loadtxt(diplole_moments_file, delimiter=","))
P = P[:, 160000:]  # starting from 4seconds; since it;s the transient. should result in 24s of data at 100Hz.
P = P[:, ::down_sample_factor]
moments = P.T
# print(P.shape)

# Dipole placement: DLPFC dipole placed under F3.
# coord = np.array([[-0.05889541+.0131, 0.06404063-.0044, 0.10122117-.01]])  # original

coord = np.array([[-0.00772741,  0.05961728,  0.04007577]])

# mne.Dipole wants "pos" in headspace coordinates; the above coordinate is in headspace

pos_mri_coord = coord
# todo: roi_xyz should be in MRI coordinates?
# trans_file_path = op.join(fs_dir, 'bem', 'fsaverage-trans.fif')
# mri_head_t_object = mne.read_trans(trans_file_path)
# pos_mri_coord = mne.head_to_mri(coord, subject=subject, mri_head_t=mri_head_t_object)

roi_xyzs, vtx_xyzs, faces = pos_mri_coord, src[0]['rr'], src[0]['tris']
faces_xyz = np.mean(vtx_xyzs[faces], axis=1)
face_inds = cKDTree(faces_xyz).query(roi_xyzs, k=1)[1]
vtx_inds = cKDTree(vtx_xyzs).query(roi_xyzs, k=1)[1]

ori = src[0]['nn'][vtx_inds]
ori = ori.reshape(1, 3)  # dipole orientation.
# Notes: src[0] is the left hemisphere, src[1] is the right hemisphere.
# src[i]['rr'] — vertex coordinates (N×3) in meters, in the subject MRI coordinate frame.
# src[i]['tris'] — triangular faces (M×3 indices into rr) that form the mesh.
# src[i]['nn'] — normals at vertices (N×3), i.e. preferred orientations.

# prep the dipole time series and amplitudes.
n_times = len(moments)
times = np.arange(n_times) / sfreq
# Units of P is "nA µm" => mne.Dipole requires "Am".
raw_magnitude = np.linalg.norm(moments, axis=1)
amplitude_Am = raw_magnitude * 1e-15

# Simulate and save
dip = mne.Dipole(times=times, pos=np.full((len(moments), 3), coord),
                 ori=np.full((len(moments), 3), ori),
                 amplitude=amplitude_Am,
                 gof=np.full((len(moments)), 100))


fwd, stc = make_forward_dipole(dip, fname_bem, evoked.info, trans, verbose=False)
pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf,verbose=False)

out_fname = "simulation/pred_evoked-ave.fif"
pred_evoked.save(out_fname, overwrite=True)           # writes the FIF file
print("Saved evoked to", out_fname)


# multi-dipole simulation test.
# dip2 = mne.Dipole(times=times, pos=np.full((len(moments), 3), coord),
#                  ori=np.full((len(moments), 3), ori),
#                  amplitude=amplitude_Am,
#                  gof=np.full((len(moments)), 100))
#
#
# fwd, stc = make_forward_dipole([dip, dip2], fname_bem, evoked.info, trans, verbose=False)
# pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf,verbose=False)
#
# out_fname = "simulation/pred_evoked-ave.fif"
# pred_evoked.save(out_fname, overwrite=True)           # writes the FIF file
# print("Saved evoked to", out_fname)
