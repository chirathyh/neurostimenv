import os
import mne
import pandas as pd
import numpy as np

from lfpykit.eegmegcalc import FourSphereVolumeConductor


electrode_names = ['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8',
                   'T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','Oz','O2']

montage = mne.channels.make_standard_montage('standard_1020')
pos_dict = montage.get_positions()['ch_pos']
print(pos_dict)
coords_m = []
missing = []
for name in electrode_names:
    if name in pos_dict:
        coords_m.append(np.asarray(pos_dict[name]))
    else:
        coords_m.append(np.array([np.nan, np.nan, np.nan]))  # or raise/skip as you prefer
        missing.append(name)
if missing:
    print("Warning: these electrodes were not found in the standard_1020 montage:", missing)

coords_m = np.array(coords_m)                # shape (n_channels, 3), in meters

# 4) convert to LFPy units: µm
locations = coords_m * 1e6                # meters -> micrometers

print(locations)

# exit()
#
# # 1) Read CSV and build ch_pos dict (meters)
# df = pd.read_csv("m2m_ernie/eeg_positions/EEG10-20_extended_SPM12.csv", header=None)
# df = df.iloc[:, :5]
# df.columns = ["tag", "x", "y", "z", "name"]
# df["name"] = df["name"].astype(str).str.strip()
# # numeric
# df["x"] = pd.to_numeric(df["x"], errors="coerce")
# df["y"] = pd.to_numeric(df["y"], errors="coerce")
# df["z"] = pd.to_numeric(df["z"], errors="coerce")
# df = df.dropna(subset=["x", "y", "z", "name"]).reset_index(drop=True)
# df[["x","y","z"]] = df[["x","y","z"]] * 1e3 # convert to µm from mm
#
# print(df)
#
#
# def get_locations(df, electrode_names, name_col='name', coords=('x','y','z'),
#                   case_insensitive=True, on_missing='skip'):
#     # Validate on_missing
#     if on_missing not in ('skip','nan','raise'):
#         raise ValueError("on_missing must be one of 'skip', 'nan', 'raise'")
#     # Prepare name mapping (lowercased) for robust matching
#     df_copy = df.copy()
#     if case_insensitive:
#         df_copy['_name_key'] = df_copy[name_col].astype(str).str.lower()
#         lookup = df_copy.set_index('_name_key')
#         keys = [n.lower() for n in electrode_names]
#     else:
#         df_copy['_name_key'] = df_copy[name_col].astype(str)
#         lookup = df_copy.set_index('_name_key')
#         keys = list(electrode_names)
#     locations = []
#     for orig_name, key in zip(electrode_names, keys):
#         if key in lookup.index:
#             # If there are multiple rows with same name, take the first one
#             row = lookup.loc[key]
#             if isinstance(row, pd.DataFrame):
#                 row = row.iloc[0]
#             try:
#                 loc = [float(row[c]) for c in coords]
#             except Exception as e:
#                 # If coords missing or not convertible
#                 raise ValueError(f"Failed to read coordinates {coords} for electrode '{orig_name}': {e}")
#             locations.append(loc)
#         else:
#             if on_missing == 'skip':
#                 # simply skip
#                 continue
#             elif on_missing == 'nan':
#                 locations.append([np.nan, np.nan, np.nan])
#             else: # raise
#                 raise KeyError(f"Electrode '{orig_name}' not found in DataFrame (name column '{name_col}')")
#     return locations
#
#
#
# electrode_names = ['Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7']
# locations = get_locations(df, electrode_names, on_missing='skip')
# print("locations (skip missing):", locations)
# locations = [[0., 0., 90000.]]  # µm

network_position = [0., 0., 78200] # µm
radii = [79000., 80000., 85000., 90000.]  # µm ["Brain", "CSF", "Skull", "Scalp"]
sigmas = [0.3, 1.5, 0.015, 0.3]  # conductivity: (S/m)

scalp_radius = radii[3]


sphere_center = np.array([0., 0., 0.])

coords_centered = locations - sphere_center  # positions relative to sphere center
norms = np.linalg.norm(coords_centered, axis=1)
max_norm = np.nanmax(norms)
median_norm = np.nanmedian(norms)


margin = 1000.0  # µm e.g. 1 mm margin
recommended_scalp_radius = max_norm + margin
print(f"Recommended scalp radius: {recommended_scalp_radius:.1f} µm (was {scalp_radius:.1f} µm)")

# apply it
radii_new = radii.copy()
radii_new[3] = recommended_scalp_radius
scalp_radius = radii_new[3]

print(f"scalp_radius = {scalp_radius:.1f} µm; max electrode radius = {max_norm:.1f} µm; median = {median_norm:.1f} µm")
outside_idx = np.where(norms > scalp_radius)[0]
print("Electrodes outside scalp radius:", [electrode_names[i] for i in outside_idx])




# # centering
# centroid = np.nanmean(locations, axis=0)
# print("electrode centroid (µm):", centroid)
# new_locations = locations - centroid + sphere_center
# new_norms = np.linalg.norm(new_locations - sphere_center, axis=1)
# print("max radius after recentering:", np.nanmax(new_norms))

# exit()


r_norms = np.linalg.norm(locations, axis=1)
bad = np.where(~np.isfinite(r_norms) | (r_norms > scalp_radius))[0]
if len(bad):
    print("Warning: electrodes outside scalp radius or missing:", [electrode_names[i] for i in bad])

# exit()


four_sphere_top = FourSphereVolumeConductor(np.array(locations), radii_new, sigmas)

# load the dipole moments
diplole_moments_file = "../../../results/dipole/DIPOLE_MDD_1.csv"
raw = np.loadtxt(diplole_moments_file, delimiter=",")
P = np.array(raw)

pot_db_4s_top = four_sphere_top.get_dipole_potential(P, np.array(network_position))  # Units: mV
eeg = np.array(pot_db_4s_top) * 1e-3  # convert units: V

print(eeg)


exit()

from mne.datasets import fetch_fsaverage
fs_dir = fetch_fsaverage(verbose=True)   # downloads fsaverage into MNE cache
subjects_dir = os.path.dirname(fs_dir)
subject = 'fsaverage'


# Heuristic: convert mm -> m if coordinates look like mm
r = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
med_r = np.median(r)
if 50 <= med_r <= 200:
    # convert mm -> m
    df[["x","y","z"]] = df[["x","y","z"]] * 1e-3
    print("Detected units ~ mm. Converted to meters for MNE.")
else:
    print("Coordinates appear to be in meters or ambiguous (median r = %.3f)." % med_r)

ch_pos = {row["name"]: np.array([row["x"], row["y"], row["z"]]) for _, row in df.iterrows()}



# 2) Create montage and attach to Info
montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')  # head frame
ch_names = list(ch_pos.keys())
info = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types='eeg')
info.set_montage(montage)


def roi_to_surface_frank(roi_xyzs,vtx_xyzs,faces):
    faces_xyz = np.mean(vtx_xyzs[faces],axis=1)
    face_nearestroi_inds = cKDTree(faces_xyz).query(roi_xyzs, k=1)[1]
    vtx_nearestroi_inds = cKDTree(vtx_xyzs).query(roi_xyzs, k=1)[1]

    return vtx_nearestroi_inds,face_nearestroi_inds



# print("Launching MNE coregistration GUI. In the GUI: load MRI (subject), align fiducials/points, then Save trans.")
# mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)

fname_trans = "/home/chirath/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-trans.fif"


src = mne.setup_source_space(subject,
                             spacing='oct4', # increase to oct6, 8, etc for more realistic output
                             add_dist=True,
                             subjects_dir=subjects_dir,verbose=False)


print('\n BEM model: ')
model = mne.make_bem_model(subject=subject, ico=4,
                           conductivity=(0.3, 0.006, 0.3),
                           subjects_dir=subjects_dir,verbose=False)

print('\n BEM Solution: ')
fname_bem = mne.make_bem_solution(model,verbose=False)

trans = mne.read_trans(fname_trans)


roi_loc = np.array([[-0.05889541+.0131,0.06404063-.0044,0.10122117-.01]])

mri_pos = mne.head_to_mni(pos = roi_loc,
               subject=subject,
               mri_head_t=trans,
               subjects_dir=subjects_dir)


vtx_inds,face_inds = roi_to_surface_frank(roi_loc,src[0]['rr'],src[0]['tris'])

diplole_moments_file = "results/dipole/DIPOLE_MDD_1.csv"
raw = np.loadtxt(diplole_moments_file, delimiter=",")
print(raw)


ori = src[0]['nn'][vtx_inds]
ori = ori.reshape(1,3)

dip = mne.Dipole(
    times = np.array([0]),
    pos = roi_loc,
    ori = ori,
    amplitude = np.array((moments[0,2]*0.000000001)**2).reshape(1,),
    gof=np.array([100]).reshape(1,),verbose=True)


print('\n Forward Model: ')
fwd, stc = make_forward_dipole(dip, fname_bem, evoked.info, fname_trans,verbose=False)

print('\n Predicting Activity:')
pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf,verbose=False)
