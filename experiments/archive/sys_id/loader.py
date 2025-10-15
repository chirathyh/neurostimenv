import os
import glob

import os
import re
import sys
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t
from scipy.signal import stft
from env.eeg import features
import pywt


from experiments.sys_id.configs import get_configs
dt, fs, nperseg, _, t1 = get_configs()


file = '../../data/feature_analysis/mdd/EEG_MDD_51.csv'


def extract_states(file):
    EEG = np.loadtxt(file, delimiter=",")

    # filter
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

    x1 = int(1000/dt)
    EEG_filt = EEG_filt[0:x1]

    # downsample
    new_fs = 500
    down_factor = int(fs / new_fs)
    EEG_down = ss.resample_poly(EEG_filt, up=1, down=down_factor)

    wavelet = 'db4'
    mode = 'periodization'
    coeffs = pywt.wavedec(EEG_down, wavelet=wavelet, mode=mode)

    lengths = [c.shape[0] for c in coeffs]
    flat = np.hstack(coeffs)                  # 1D array of all coeffs

    # 3) truncate to first K and zero‑pad the rest
    K = 64
    flat_trunc = np.zeros_like(flat)
    flat_trunc[:K] = flat[:K]

    # 4) unflatten back into list of arrays matching original shapes
    coeffs_trunc = []
    idx = 0
    for L in lengths:
        print(L)
        coeffs_trunc.append(flat_trunc[idx:idx+L])
        idx += L
