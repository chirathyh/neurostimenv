import os.path as op
import numpy as np
import mne
from mne.forward import make_forward_dipole
from mne.simulation import simulate_evoked
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def roi_to_surface_frank(roi_xyzs,vtx_xyzs,faces):
    faces_xyz = np.mean(vtx_xyzs[faces],axis=1)
    face_nearestroi_inds = cKDTree(faces_xyz).query(roi_xyzs, k=1)[1]
    vtx_nearestroi_inds = cKDTree(vtx_xyzs).query(roi_xyzs, k=1)[1]

    return vtx_nearestroi_inds,face_nearestroi_inds


# Paths & files you must have
subjects_dir = '/home/chirath/mne_data/MNE-sample-data/subjects'       # FreeSurfer subject dir (e.g. sample/subjects)
subject = 'sample'                           # subject name as in subjects_dir
data_path = '/home/chirath/mne_data/MNE-sample-data/MEG/sample'

fname_trans = op.join(data_path, 'sample_audvis_raw-trans.fif')  # your trans file
fname_ave = op.join(data_path, 'sample_audvis-ave.fif')

# raw_fname = op.join(data_path, 'sample_audvis_raw.fif') # The raw MEG/EEG data
# info = mne.io.read_info(raw_fname)
# trans = mne.read_trans(fname_trans)

src = mne.setup_source_space(subject, spacing='oct4',  add_dist=True, subjects_dir=subjects_dir,verbose=False)  # spacing: increase to oct6, 8, etc for more realistic output
model = mne.make_bem_model(subject='sample', ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir,verbose=False)  # Make BEM model and solution.
fname_bem = mne.make_bem_solution(model,verbose=False)


# Create Dummy Evoked to hold dipole data
evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',baseline=(None, 0),verbose=False)
evoked.pick_types(meg=False, eeg=True)
evoked.resample(400)
evoked.filter(l_freq=None, h_freq=199.9)

diplole_moments_file = "../../results/dipole/DIPOLE_HEALTHY_0.csv"
raw = np.loadtxt(diplole_moments_file, delimiter=",")
P = np.array(raw)
P= P[:, 80000:]
P = P[:, ::400]
moments = P.T
print(moments.shape)

# cooordinate of the dipole placement
# coord = [-0.07074824+.0131,0.05937758,0.0837867]
coord = [-0.05889541+.0131,0.06404063-.0044,0.10122117-.01]  # dipole placed under F3.
coord = np.array([coord])

# orientation of the dipole
vtx_inds,face_inds = roi_to_surface_frank(coord,src[0]['rr'],src[0]['tris'])
ori = src[0]['nn'][vtx_inds]
ori = ori.reshape(1,3)

sfreq = 400
n_times = len(moments)
times = np.arange(n_times) / sfreq
print(times.shape)

# units unsdure yet?
raw_magnitude = np.linalg.norm(moments, axis=1)
amplitude_nAm = raw_magnitude * 1e-12
# need units nAm; lfpy gives nAm x um; so multiply by 10-6

dip = mne.Dipole(times=times, pos = np.full((len(moments),3),coord), ori = np.full((len(moments),3),ori),
                 amplitude = amplitude_nAm, gof = np.full((len(moments)),100))
fwd, stc = make_forward_dipole(dip, fname_bem, evoked.info, fname_trans, verbose=False)
pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf,verbose=False)
# pos: origin of each moment in head coordinates, stays same for all (fixed)
# ori: orientation of each dipole
# gof: goodness of fit, doesn't matter here

# get EEG data.
data = pred_evoked.get_data()


print("complete")
ch_names = pred_evoked.info['ch_names']   # list of 59 channel names in order
sfreq = pred_evoked.info['sfreq']         # sampling rate (Hz)
times = pred_evoked.times                 # time vector in seconds, length 2601

print(len(ch_names), sfreq, times.shape)
print(ch_names[:10])  # first 10 channel names


fig = plt.figure(figsize=(10,10))
ax = plt.subplot(121)
ax2 = plt.subplot(122)

# create mask to show source locations
electrodes = [9]
emask = []
for i in range(60):
    if i not in electrodes:
        emask.append([False])
    else:
        emask.append([True])

emask = np.array(emask)
mask_params = dict(markersize=10, markerfacecolor='white')

pred_evoked.plot_topomap(times = [0],
                         axes=[ax,ax2],
                         # sensors=True,
                         # names=ch_names,
                         # contours = np.arange(-.01,0.01,.001),
                         mask = emask,
                         mask_params=mask_params)

fig.savefig('Figure6_B1.jpg', facecolor='white', edgecolor='none',bbox_inches = "tight",dpi=1000)

# from plot_eeg import plot_eeg_stack
# electrode_names = ch_names[:10]
# EEG_downsampled = data[:10, :]
# plot_eeg_stack(EEG_downsampled, electrode_names, sfreq=400.0, unit='uV', figsize=(12, 8), savepath=None)
#
# print(data)


exit()











# ---------------------------
# Choose dipole position (meters in head frame)
# ---------------------------
# Example dipole coordinate in notebook:
roi_head = np.array([[-0.05889541 + 0.0131, 0.06404063 - 0.0044, 0.10122117 - 0.01]])  # already in meters?

# Verify coordinate unit: if it's in mm multiply by 1e-3. If in µm multiply by 1e-6.
# e.g., if your coord was in mm: roi_head = roi_head * 1e-3

# ---------------------------
# get orientation from sources (optional)
# ---------------------------
src = mne.setup_source_space(subject, spacing='oct4', subjects_dir=subjects_dir, add_dist=True, verbose=False)

print('\n BEM model: ')
model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3, 0.006, 0.3),
                           subjects_dir=subjects_dir,verbose=False)

print('\n BEM Solution: ')
fname_bem = mne.make_bem_solution(model,verbose=False)

trans = mne.read_trans(fname_trans)

print("complete")
