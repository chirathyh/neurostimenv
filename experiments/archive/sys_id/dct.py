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


from experiments.bandit.stats.configs import get_configs
dt, fs, nperseg, _, t1 = get_configs()


plt.rcParams.update({
    'font.size': 14,  # Base font size for small text (ticks, annotations)
    'axes.titlesize': 16,  # Axis titles
    'axes.labelsize': 16,  # Axis labels (xlabel, ylabel)
    'xtick.labelsize': 14,  # Tick labels
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,  # Figure title (if you ever use suptitle)
})


file = '../../data/feature_analysis/mdd/EEG_MDD_10.csv'
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


from scipy.fftpack import dct, idct
coeffs = dct(EEG_down, norm='ortho')
K=64
# keep only first K coefficients as features
features = coeffs[:K]
# reconstruct:
padded = np.zeros_like(coeffs)
padded[:K] = features
EEG_rec = idct(padded, norm='ortho')


f_orig, Pxx_orig = ss.welch(EEG_filt, fs=fs,       nperseg=fs//2)
f_down, Pxx_down = ss.welch(EEG_down, fs=new_fs,   nperseg=new_fs//2)
f_rec,  Pxx_rec  = ss.welch(EEG_rec,  fs=new_fs,   nperseg=new_fs//2)


# build time axes
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 6))

ax1.plot(EEG_filt, color='C0')
ax2.plot(EEG_down, color='C1')
ax3.plot(EEG_rec, color='C2')

ax4.plot(f_orig, Pxx_orig, color='C0')
ax4.plot(f_down, Pxx_down, color='C1')
ax4.plot(f_rec, Pxx_rec, color='C2')
ax4.set_xlim(4, 40)

plt.tight_layout()
plt.show()


