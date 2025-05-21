import os
import re
import sys
from tqdm import tqdm
from decouple import config
import torch

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import scipy.signal as ss
import numpy as np
from scipy.stats import t

from utils.utils import prep_stim_seq


dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)
transient = 4000  # ms; first 4s is removed from the EEG (triansient phase)
t1 = int(transient/dt)
print("Sampling Rate:", fs)
print("npserg", nperseg)


bandit_action_pairs = [[1, 8.],
                           [2, 8.],
                           [4, 8.],
                           [2, 10.],
                           [2, 40.],
                           [15, 77.5]]


chosen_arm = 1

# stim
policy_seq=[
            [0., 1.],
            [bandit_action_pairs[chosen_arm][0], bandit_action_pairs[chosen_arm][1]],
            [0., 1.],
            [bandit_action_pairs[chosen_arm][0], bandit_action_pairs[chosen_arm][1]]
            ]
i_stim, t_stim = prep_stim_seq(action=policy_seq, step_size=1000, steps=4, dt=0.025)
i_stim[:] = [x / 1e6 for x in i_stim]  # rescale to mA

# eeg
file="../../data/bandit/simnibsbandit2/testing/EEG_BANDIT_1010.csv"
EEG = np.loadtxt(file, delimiter=",")
b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)
EEG_filt = EEG_filt[1:]

EEG_COLOUR = '#000000'
TS_COLOUR = '#800000'

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.4])
ax2 = fig.add_subplot(gs[0, :])
divider = make_axes_locatable(ax2)
ax = divider.append_axes("top", size=3.0, pad=0.02, sharex=ax2)

ax.plot(t_stim, EEG_filt, color=EEG_COLOUR)
ax2.plot(t_stim, i_stim, color=TS_COLOUR)
eeg_line = mlines.Line2D([], [], color=EEG_COLOUR, label='EEG')
ts_a_line = mlines.Line2D([], [], color=TS_COLOUR, label=f'Amplitude: 2 mA')
ts_f_line = mlines.Line2D([], [], color=TS_COLOUR, label=f'Frequency: 8 Hz')
ax.legend(handles=[eeg_line, ts_a_line, ts_f_line], loc='upper right')

ax.set_ylabel('EEG (V)', color=EEG_COLOUR)
ax2.set_ylabel('Stimulation (mA)', color=TS_COLOUR)
ax2.set_xlabel('Time (ms)')
ax.set_title('Simulation: Transcranial Stimulation')


plt.show()
