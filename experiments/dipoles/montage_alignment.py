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
from mne.datasets import sample  # or your dataset

data_path = '/home/chirath/mne_data/MNE-sample-data/MEG/sample'
raw_fname = op.join(data_path, 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=False)  # or load your raw/evoked

# choose montage: 'standard_1020' or 'standard_1005'
montage = mne.channels.make_standard_montage('standard_1020')

fname_ave = op.join(data_path, 'sample_audvis-ave.fif')
evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',baseline=(None, 0),verbose=False)
evoked.pick_types(meg=False, eeg=True)



fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection="3d")
raw.plot_sensors(ch_type="eeg", axes=ax2d)
raw.plot_sensors(ch_type="eeg", axes=ax3d, kind="3d")
ax3d.view_init(azim=70, elev=15)
plt.show()
exit()

# --- choose object to operate on: evoked or raw ---
# use evoked if you've already created it, or raw if you have raw.
# obj = evoked
obj = evoked  # or raw

# 1) get eeg channel picks, names and positions from your data
eeg_picks = mne.pick_types(obj.info, eeg=True, meg=False)
old_names = [obj.ch_names[p] for p in eeg_picks]

# positions are in meters in info['chs'][p]['loc'][:3]
old_pos = np.array([obj.info['chs'][p]['loc'][:3] for p in eeg_picks])
# print(old_pos)

# handle channels with missing/zero positions
zero_mask = np.all(np.isclose(old_pos, 0), axis=1)
if zero_mask.any():
    print("Warning: some EEG channels have zero/missing positions; they will be skipped for position-based mapping.")
    # keep only those with positions
    valid_idx = ~zero_mask
    old_names_valid = [n for i, n in enumerate(old_names) if valid_idx[i]]
    old_pos_valid = old_pos[valid_idx]
else:
    old_names_valid = old_names
    old_pos_valid = old_pos

# 2) montage positions
mont_pos = montage.get_positions()['ch_pos']   # dict name -> (x,y,z); this is in meters
# print(mont_pos)


mont_names = list(mont_pos.keys())
mont_array = np.array([mont_pos[n] for n in mont_names])

# 3) compute pairwise distances and find one-to-one assignment
# dist_matrix shape: (n_measured, n_mont)
dist_matrix = cdist(old_pos_valid, mont_array)  # meters

# Hungarian algorithm returns row->col assignment minimizing total distance
row_ind, col_ind = linear_sum_assignment(dist_matrix)

# construct mapping for only the valid channels we used
mapping = {}
assigned_distances = {}
for r, c in zip(row_ind, col_ind):
    old_name = old_names_valid[r]
    new_name = mont_names[c]
    mapping[old_name] = new_name
    assigned_distances[old_name] = dist_matrix[r, c]

# 4) sanity checks
# duplicates shouldn't happen due to Hungarian assignment, but check
if len(set(mapping.values())) != len(mapping.values()):
    print("Warning: duplicate target labels after assignment (unexpected)")

# warn about large distances (e.g. > 3 cm)
large = {n: d for n, d in assigned_distances.items() if d > 0.03}
if large:
    print("Warning — some assigned distances > 3 cm (check these channels):")
    for n, d in large.items():
        print(f"  {n} -> {mapping[n]}  ({d*1000:.1f} mm)")

# 5) apply mapping: rename and then set montage
# If some channels were zero-pos and were skipped, map them by order as fallback:
if zero_mask.any():
    # fallback: map remaining channels (zero-pos) by order to unused montage names
    unused_mont = [mn for mn in mont_names if mn not in mapping.values()]
    zero_names = [old_names[i] for i in np.where(zero_mask)[0]]
    if len(zero_names) <= len(unused_mont):
        fallback_map = dict(zip(zero_names, unused_mont[:len(zero_names)]))
        mapping.update(fallback_map)
        print("Applied fallback order-based mapping for channels without positions.")
    else:
        print("Not enough montage names for fallback mapping — manual check needed for zero-pos channels.")

# apply rename on the object (raw or evoked)
obj.rename_channels(mapping)

# attach the montage so channel positions are known
# (works because names now match the montage)
obj.set_montage(montage)

# verify
print("New channel names (first 30):", obj.ch_names[:30])
