"""
.. _tut-eeg-fsaverage-source-modeling:

========================================
EEG forward operator with a template MRI
========================================

This tutorial explains how to compute the forward operator from EEG data
using the standard template MRI subject :ref:`fsaverage <fsaverage_background>`.

.. caution:: Source reconstruction without an individual T1 MRI from the
             subject will be less accurate. Do not over interpret activity
             locations which can be off by multiple centimeters.

Adult template MRI (fsaverage)
------------------------------
First we show how ``fsaverage`` can be used as a surrogate subject.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

import mne
from mne.datasets import eegbci, fetch_fsaverage
from scipy.spatial import cKDTree
from mne.forward import make_forward_dipole
from mne.simulation import simulate_evoked
import matplotlib.pyplot as plt
import math

def roi_to_surface_frank(roi_xyzs,vtx_xyzs,faces):
    faces_xyz = np.mean(vtx_xyzs[faces],axis=1)
    face_nearestroi_inds = cKDTree(faces_xyz).query(roi_xyzs, k=1)[1]
    vtx_nearestroi_inds = cKDTree(vtx_xyzs).query(roi_xyzs, k=1)[1]

    return vtx_nearestroi_inds,face_nearestroi_inds


# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation

src = mne.setup_source_space(subject, spacing='oct4',  add_dist=True, subjects_dir=subjects_dir,verbose=False)  # spacing: increase to oct6, 8, etc for more realistic output
model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir, verbose=False)  # Make BEM model and solution.
fname_bem = mne.make_bem_solution(model, verbose=False)

# load EEG data
# Read and set the EEG electrode locations, which are already in fsaverage's space (MNI space) for standard_1020:

# (raw_fname, ) = eegbci.load_data(subjects=1, runs=[2])
# raw = mne.io.read_raw_edf(raw_fname, preload=True)
# eegbci.standardize(raw)  # Clean channel names to be able to use a standard 1005 montage
# montage = mne.channels.make_standard_montage("standard_1020")
# raw.set_montage(montage)
# raw.set_eeg_reference(projection=True)  # needed for inverse modeling

# exit()

#Check that the locations of EEG electrodes is correct with respect to MRI
# fig = mne.viz.plot_alignment(
#     raw.info,
#     src=src,
#     eeg=["original", "projected"],
#     trans=trans,
#     show_axes=True,
#     mri_fiducials=True,
#     dig="fiducials",
# )
# fig.plotter.screenshot("alignment_plot.png")
# exit()

# You can ensure everything is as expected by plotting the result:
# fig = mne.viz.plot_alignment(
#     raw.info,
#     subject=subject,
#     subjects_dir=subjects_dir,
#     trans=trans,
#     src=src,
#     bem=fname_bem,
#     coord_frame="mri",
#     mri_fiducials=True,
#     show_axes=True,
#     surfaces=("white", "outer_skin", "inner_skull", "outer_skull"),
# )
# fig.plotter.screenshot("alignment_plot2.png")
# mne.viz.set_3d_view(fig, 25, 70, focalpoint=[0, -0.005, 0.01])
# exit()


# Create Dummy Evoked to hold dipole data
montage = mne.channels.make_standard_montage("standard_1020")
sfreq = 400
# ch_names = montage.ch_names

exclude = ['T7','T8','P7','P8','T3','T5','T4','T6']
keep_chs = [ch for ch in montage.ch_names if ch not in exclude]
print("keep_chs:", keep_chs)

info = mne.create_info(keep_chs, sfreq, ch_types="eeg")
info.set_montage(montage)
evoked = mne.EvokedArray(np.zeros((len(keep_chs), 1)), info, tmin=0.0)  # make a dummy evoked with zeros just to keep API consistent
evoked.pick_types(meg=False, eeg=True)
evoked.resample(400)
evoked.filter(l_freq=None, h_freq=199.9)

diplole_moments_file = "../../results/dipole/DIPOLE_HEALTHY_0.csv"
raw = np.loadtxt(diplole_moments_file, delimiter=",")
P = np.array(raw)
P= P[:, 80000:]
P = P[:, ::400]
moments = P.T
# print(moments.shape)

# cooordinate of the dipole placement
coord = [-0.05889541+.0131,0.06404063-.0044,0.10122117-.01]  # dipole placed under F3.
coord = np.array([coord])

# orientation of the dipole
vtx_inds,face_inds = roi_to_surface_frank(coord,src[0]['rr'],src[0]['tris'])
ori = src[0]['nn'][vtx_inds]
ori = ori.reshape(1, 3)


n_times = len(moments)
times = np.arange(n_times) / sfreq
print(times.shape)

# units unsdure yet?
raw_magnitude = np.linalg.norm(moments, axis=1)
amplitude_nAm = raw_magnitude * 1e-12
# need units nAm; lfpy gives nAm x um; so multiply by 10-6


#debug
# After raw.set_montage(montage)
# print("raw channels:", raw)
# print(montage.ch_names)

dip = mne.Dipole(times=times, pos = np.full((len(moments), 3), coord),
                 ori=np.full((len(moments), 3), ori),
                 amplitude=amplitude_nAm,
                 gof=np.full((len(moments)), 100))
fwd, stc = make_forward_dipole(dip, fname_bem, evoked.info, trans, verbose=False)
pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf,verbose=False)
data = pred_evoked.get_data()
# print(data)

out_fname = "pred_evoked-ave.fif"
pred_evoked.save(out_fname)           # writes the FIF file
print("Saved evoked to", out_fname)


print("complete")

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(121)
ax2 = plt.subplot(122)

# create mask to show source locations
electrodes = [9]
emask = []
for i in range(10):
    if i not in electrodes:
        emask.append([False])
    else:
        emask.append([True])

emask = np.array(emask)
mask_params = dict(markersize=10, markerfacecolor='white')

pred_evoked.plot_topomap(times=[0],
                         axes=[ax,ax2],
                         sensors=True,
                         ch_type="eeg",
                         show_names=True,
                         extrapolate="head",
                         # contours = np.arange(-.01,0.01,.001),
                         # mask = emask,
                         # mask_params=mask_params
                         )

fig.savefig('Figure6_B2.jpg', facecolor='white', edgecolor='none',bbox_inches = "tight",dpi=1000)


from plot_eeg import plot_eeg_stack
electrode_names = keep_chs[:10]
EEG_downsampled = data[:10, :]
plot_eeg_stack(EEG_downsampled, electrode_names, sfreq=400.0, unit='uV', figsize=(12, 8), savepath=None)
