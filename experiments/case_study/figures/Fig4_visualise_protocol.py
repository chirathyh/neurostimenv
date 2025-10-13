import os
import re
import sys
from decouple import config
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
from experiments.case_study.configs import get_configs


dt, fs, nperseg, _, t1 = get_configs()
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
})

chosen_arm = 1
bandit_action_pairs = [[1, 8.], [2, 8.], [4, 8.], [2, 10.], [2, 40.], [15, 77.5]]

# stim protocol
policy_seq=[[0., 1.],  # initial no-stim
            [bandit_action_pairs[chosen_arm][0], bandit_action_pairs[chosen_arm][1]],
            [0., 1.],
            [bandit_action_pairs[chosen_arm][0], bandit_action_pairs[chosen_arm][1]],
            [0., 1.],  # final no-stim
            ]
i_stim, t_stim = prep_stim_seq(action=policy_seq, step_size=1000, steps=5, dt=0.025)
i_stim[:] = [x / 1e6 for x in i_stim]  # rescale to mA

# eeg
file ="../data/bandit/testing/EEG_BANDIT_1010.csv"

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

ax.text(500, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"Segment-1", fontsize=14, color='black', ha='center', fontweight='bold')
ax.text(1500, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"Segment-2", fontsize=14, color=TS_COLOUR, ha='center', fontweight='bold')
ax.text(2500, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"Segment-3", fontsize=14, color=TS_COLOUR, ha='center', fontweight='bold')
ax.text(3500, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"Segment-4", fontsize=14, color=TS_COLOUR, ha='center', fontweight='bold')
ax.text(4500, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"Segment-5", fontsize=14, color='black', ha='center', fontweight='bold')


eeg_line = mlines.Line2D([], [], color=EEG_COLOUR, label='EEG')
ts_a_line = mlines.Line2D([], [], color=TS_COLOUR, label=f'Stimulation time-series (Amplitude: 2mA, Frequency: 8Hz)')
# ts_f_line = mlines.Line2D([], [], color=TS_COLOUR, label=f'Frequency: 8 Hz')
ax.legend(handles=[eeg_line, ts_a_line],
          bbox_to_anchor=(0.5, 0.25),   # center of the axes
          loc='center',                # place legend's center at bbox point
          bbox_transform=ax.transAxes,
          frameon=True,
          fontsize=14)
          # loc='lower right')
ax.xaxis.set_visible(False)
ax.set_ylabel('EEG (V)', color=EEG_COLOUR)
ax2.set_ylabel('Stimulation (mA)', color=TS_COLOUR)
ax2.set_xlabel('Time (ms)')
# ax.set_title('Simulation: Transcranial Stimulation')
plt.tight_layout()
fig.savefig('png/Fig4.png', dpi=600, bbox_inches='tight', pad_inches=0.02, facecolor='auto', transparent=False)
plt.show()
