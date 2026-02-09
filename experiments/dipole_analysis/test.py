"""
Simulate dipole data and visualize the results.
We use the sample OR fsaverage adult template MRI; place the dipole in a target location and model EEG using a standard 10-20 EEG montage.
"""
import sys
import time
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import numpy as np
import mne
import os.path as op
from mne.datasets import eegbci, fetch_fsaverage
from scipy.spatial import cKDTree
from mne.forward import make_forward_dipole
from mne.simulation import simulate_evoked
import matplotlib.pyplot as plt
import math
import hydra
from omegaconf import DictConfig, OmegaConf


def setup_sourcespace_bem(subjects_dir, subject):
    # source-space / BEM surfaces: MRI
    # head-space: EEG/MEG sensors
    # trans file: head -> MRI
    # The source-space (brain location where we add the dipoles - MRI space), BEM model (boundary element model descrbing the head as a set o flcosed surfaces),
    # and the BEM solution (the numerical object ready to be used by forward solvers).
    src = mne.setup_source_space(subject, spacing='oct4', add_dist=True, subjects_dir=subjects_dir, verbose=False) # spacing: increase to oct6, 8, etc for more realistic output
    model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir, verbose=False)
    fname_bem = mne.make_bem_solution(model, verbose=False)
    return src, fname_bem


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    subjects_dir, subject, trans =  cfg.analysis.subjects_dir, cfg.analysis.subject, cfg.analysis.trans
    src, fname_bem = setup_sourcespace_bem(subjects_dir, subject)

    # Setting up a dummy Evoked to hold EEG channel info.
    sfreq = cfg.analysis.sfreq
    montage = mne.channels.make_standard_montage("standard_1020")
    exclude = ['T7', 'T8', 'P7', 'P8'] # 'T3', 'T5', 'T4', 'T6'  https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
    keep_chs = [ch for ch in montage.ch_names if ch not in exclude]
    print("EEG channels used:", keep_chs)
    info = mne.create_info(keep_chs, sfreq, ch_types="eeg")
    info.set_montage(montage)
    evoked = mne.EvokedArray(np.zeros((len(keep_chs), 1)), info, tmin=0.0)
    evoked.pick_types(meg=False, eeg=True)
    # evoked.filter(l_freq=None, h_freq=199.9)

    coords, coords_hemi, moments, amplitude_Am = [], [], [], []
    n_dipoles = len(cfg.analysis.dipoles)
    down_sample_factor = int(cfg.analysis.orig_sfreq/sfreq)  # to get to sfreq

    # todo: handling the files; combinations
    diplole_moments_files = ["../../results/dipole/DIPOLE_HEALTHY_0.csv", "../../results/dipole/DIPOLE_HEALTHY_1.csv"]

    # Load dipole information from configs and moments from file and calculate amplitudes.
    for i in range(n_dipoles):

        coords.append(np.array(cfg.analysis.dipoles[i].coord))
        coords_hemi.append(cfg.analysis.dipoles[i].hemisphere)

        P = np.array(np.loadtxt(diplole_moments_files[i], delimiter=","))
        P = P[:, 160000:]  # starting from 4seconds; since its the transient. should result in 24s of data at 100Hz.
        P = P[:, ::down_sample_factor]
        moments.append(P.T)
        raw_magnitude = np.linalg.norm(P.T, axis=1)
        amplitude_Am.append(raw_magnitude * 1e-15)  # existing moments -> scalar amplitude (nA·µm -> A)

    # prepare the dipole time series and amplitudes (same timecourse for both dipoles here)
    lengths = [len(m) for m in moments]
    assert all(l == lengths[0] for l in lengths), f"moment lengths differ: {lengths}"

    n_times = len(moments[0])
    times = np.arange(n_times) / sfreq

    # Build per-dipole arrays (list-of-arrays) then flatten consistently
    positions = []      # each entry (n_times, 3)
    orientations = []   # each entry (n_times, 3)
    amplitude_list = [] # each entry (n_times,)
    gof_list = []       # each entry (n_times,)

    for i in range(n_dipoles):
        coord = np.asarray(coords[i]).reshape(3,)
        hemi = coords_hemi[i]
        if hemi not in ('lh', 'rh'):
            raise ValueError(f"Unexpected hemisphere label '{hemi}' for dipole index {i}; expected 'lh' or 'rh'")

        hemi_ind = 0 if hemi == 'lh' else 1

        # find nearest vertex and get its normal (vertex normal is typically smoothed)
        roi_xyzs = coord.reshape(1, 3)
        vtx_xyzs = src[hemi_ind]['rr']
        vtx_ind = cKDTree(vtx_xyzs).query(roi_xyzs, k=1)[1].item()  # scalar index

        # get orientation vector and ensure normalization
        ori = np.array(src[hemi_ind]['nn'][vtx_ind], dtype=float).ravel()
        nrm = np.linalg.norm(ori)
        if nrm == 0:
            raise RuntimeError(f"Zero-length orientation at dipole {i}, vertex {vtx_ind}")
        ori = ori / nrm

        # Optional: enforce consistent sign (pointing roughly inward)
        # Compute vector from dipole pos to head center (approx 0,0,0 in MRI/head coords)
        # If dot(ori, pos-center) > 0 then ori points outward from center; flip to point inward.
        # This is optional — comment/uncomment depending on desired convention.
        # center = np.array([0.0, 0.0, 0.0])
        # if np.dot(ori, coord - center) > 0:
        #     ori = -ori

        # repeat orientation and position across time
        orientations.append(np.tile(ori.reshape(1, 3), (n_times, 1)))   # (n_times, 3)
        positions.append(np.full((n_times, 3), coord, dtype=float))

        # amplitude: amplitude_Am was computed as list of 1D arrays (n_times,)
        amplitude_list.append(np.asarray(amplitude_Am[i], dtype=float).reshape(n_times,))
        gof_list.append(np.full((n_times,), 100.0, dtype=float))

    # Now flatten in a consistent order: source0 times (all t), then source1 times, ...
    times_flat = np.tile(times, n_dipoles)                      # shape (n_dipoles * n_times,)
    amplitude_flat = np.concatenate(amplitude_list, axis=0)     # shape (n_dipoles * n_times,)
    gof_flat = np.concatenate(gof_list, axis=0)                 # shape (n_dipoles * n_times,)
    pos_flat = np.vstack(positions)                             # shape (n_dipoles * n_times, 3)
    ori_flat = np.vstack(orientations)                          # shape (n_dipoles * n_times, 3)

    # Sanity checks
    N = n_dipoles * n_times
    assert times_flat.ndim == 1 and times_flat.size == N
    assert amplitude_flat.ndim == 1 and amplitude_flat.size == N
    assert gof_flat.ndim == 1 and gof_flat.size == N
    assert pos_flat.shape == (N, 3)
    assert ori_flat.shape == (N, 3)

    # Build Dipole (use keywords to avoid positional mistakes)
    dip = mne.Dipole(times=times_flat,
                     pos=pos_flat,
                     amplitude=amplitude_flat,
                     ori=ori_flat,
                     gof=gof_flat)

    fwd, stc = make_forward_dipole(dip, fname_bem, evoked.info, trans, verbose=False)
    pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf,verbose=False)

    out_fname = "simulation/multi-dipole-pred_evoked-ave.fif"
    pred_evoked.save(out_fname, overwrite=True)           # writes the FIF file
    print("Saved evoked to", out_fname)

    # Helpful debug prints (remove later)
    print("Dipole summary:")
    print(" n_dipoles:", n_dipoles)
    print(" n_times:", n_times)
    print(" total_dipole_events:", N)
    print(" times_flat.shape:", times_flat.shape)
    print(" amplitude_flat min/max (A):", np.min(amplitude_flat), np.max(amplitude_flat))
    print(" pos_flat range (m): min", np.min(pos_flat, axis=0), "max", np.max(pos_flat, axis=0))
    print(" ori_flat norms: min/max", np.min(np.linalg.norm(ori_flat, axis=1)), np.max(np.linalg.norm(ori_flat, axis=1)))
    # ------------------- END REPLACEMENT -------------------


if __name__ == "__main__":
    main()
