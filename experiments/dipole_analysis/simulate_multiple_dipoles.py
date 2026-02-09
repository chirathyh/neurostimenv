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
    # SUBJECT = 'MDD_0'
    # diplole_moments_files = ["/home/chirath/Documents/NeuroLake/dipole/dipole_mdd1/DIPOLE_"+SUBJECT+".csv",
    #                          "/home/chirath/Documents/NeuroLake/dipole/dipole_mdd1/DIPOLE_"+SUBJECT+".csv"]


    SUBJECT = 'HEALTHY_0'
    diplole_moments_files = ["/home/chirath/Documents/NeuroLake/dipole/dipole_healthy1/DIPOLE_"+SUBJECT+".csv",
                             "/home/chirath/Documents/NeuroLake/dipole/dipole_healthy1/DIPOLE_"+SUBJECT+".csv"]

    #diplole_moments_files = ["../../results/dipole/DIPOLE_"+SUBJECT+".csv", "../../results/dipole/DIPOLE_"+SUBJECT+".csv"]

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

    # prep the dipole orientations and positions.
    orientations, positions = [], []
    for i in range(n_dipoles):
        coord, hemi, ori = coords[i], coords_hemi[i], None
        hemi_ind = 0 if hemi == 'lh' else 1
        roi_xyzs  = coord.reshape(1, 3)
        vtx_xyzs, faces = src[hemi_ind]['rr'], src[hemi_ind]['tris']  # vertex: single point(x, y, z) and faces: triangles
        # faces_xyz = np.mean(vtx_xyzs[faces], axis=1)
        # face_inds = cKDTree(faces_xyz).query(roi_xyzs, k=1)[1]
        vtx_inds = cKDTree(vtx_xyzs).query(roi_xyzs, k=1)[1]
        ori = src[hemi_ind]['nn'][vtx_inds]
        ori = ori.reshape(1, 3)
        orientations.append(ori)
        positions.append(np.full((n_times, 3), coord))

    dip1 = mne.Dipole(times=times, pos=np.full((len(moments[0]), 3), coords[0]),
                 ori=np.full((len(moments[0]), 3), orientations[0]),
                 amplitude=amplitude_Am[0],
                 gof=np.full((len(moments[0])), 100))

    dip2 = mne.Dipole(times=times, pos=np.full((len(moments[1]), 3), coords[1]),
                 ori=np.full((len(moments[1]), 3), orientations[1]),
                 amplitude=amplitude_Am[1],
                 gof=np.full((len(moments[1])), 100))

    fwd, stc = make_forward_dipole([dip1, dip2], fname_bem, evoked.info, trans, verbose=False)
    pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf,verbose=False)

    out_fname = "simulation/"+SUBJECT+".fif"
    pred_evoked.save(out_fname, overwrite=True)           # writes the FIF file
    print("Saved evoked to", out_fname)

    # build dipole object with both dipoles present at each timestamp
    # dip = mne.Dipole(times=np.tile(times, n_dipoles) , pos=np.vstack(positions), ori=np.vstack(orientations),
    #                  amplitude=np.concatenate(amplitude_Am), gof=np.full((n_dipoles * n_times,), 100.))
    # fwd, stc = make_forward_dipole(dip, fname_bem, evoked.info, trans, verbose=False)
    # pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf,verbose=False)
    #
    # out_fname = "simulation/multi-dipole-pred_evoked-ave.fif"
    # pred_evoked.save(out_fname, overwrite=True)           # writes the FIF file
    # print("Saved evoked to", out_fname)


if __name__ == "__main__":
    main()
