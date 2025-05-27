import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t
from mne.time_frequency import psd_array_welch
from fooof import FOOOF
from fooof import FOOOFGroup


dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)
transient = 4000  # ms; first 4s is removed from the EEG (triansient phase)
t1 = int(transient/dt)
print("Sampling Rate:", fs)
print("npserg", nperseg)
sfreq = fs

plt.rcParams.update({
    # Base font size for small text (ticks, annotations)
    'font.size': 14,
    # Axis titles
    'axes.titlesize': 16,
    # Axis labels (xlabel, ylabel)
    'axes.labelsize': 16,
    # Tick labels
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    # Legend text
    'legend.fontsize': 14,
    # Figure title (if you ever use suptitle)
    'figure.titlesize': 18,
})


def get_bandit_eeg(folder_path, selected_arm=1, segment=4):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    eeg_data = []
    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        #print(file, reward_file)
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]
        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue
        if rew < -1.5:
            continue

        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        # different protocol stages
        x1 = int(1000/dt)
        if segment == 1:
            EEG_segment = EEG_filt[0:x1]
        elif segment == 2:
            EEG_segment = EEG_filt[ x1 : x1*2]
        elif segment == 3:
            EEG_segment = EEG_filt[x1*2 : x1*3]
        elif segment == 4:
            EEG_segment = EEG_filt[x1*3 : x1*4]
        elif segment == 5:
            EEG_segment = EEG_filt[x1*4 : ]
        elif segment == -1:
            EEG_segment = EEG_filt
        else:
            print("select segment")
            exit()
        eeg_data.append(EEG_segment)

    return np.array(eeg_data)


def get_offsets_exponents(mdd_eeg_data):

    spectra = []
    nperseg = fs/4
    for i in range(0, len(mdd_eeg_data)):
        freqs, powers = ss.welch(mdd_eeg_data[i], fs=fs, nperseg=nperseg)
        spectra.append(powers)
    spectra = np.array(spectra)
    print(freqs.shape)
    print(spectra.shape)
    fg = FOOOFGroup(peak_width_limits=[0.5, 8], min_peak_height=0.02, max_n_peaks=10)
    fg.fit(freqs, spectra, [4, 50])
    fg.print_results()
    aper_params = fg.get_params('aperiodic_params')    # shape (n_chans, 2)
    offsets   = aper_params[:, 0] #[:, None]
    exponents = aper_params[:, 1] #[:, None]
    return offsets, exponents

SELECTED_ARM = 1
SEGEMENT = 1

colours = ['r', 'b', 'k', 'y', 'm']
segments = [1, 5]
labels = ['Segment 1', 'Segment 2']
plt.figure(figsize=(10, 5))
for i in range(0, len(segments)):
    eeg_data = get_bandit_eeg(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=segments[i])
    offsets, exponents = get_offsets_exponents(eeg_data)
    plt.scatter(exponents, offsets, color=colours[i], label=labels[i])
plt.xlabel("Exponent")
plt.ylabel("Offset")
plt.legend()
plt.show()
exit()



EEG = np.loadtxt("../../data/bandit/simnibsbandit3/training/EEG_BANDIT_66.csv", delimiter=",")
b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

x1 = int(1000/dt)
segment1 = EEG_filt[0:x1]
segment2 = EEG_filt[x1 : x1*2]
segment3 = EEG_filt[x1*2 : x1*3]
segment4 = EEG_filt[x1*3 : x1*4]
segment5 = EEG_filt[x1*4 : ]

nperseg = fs/4
freqs1, powers1 = ss.welch(segment1, fs=fs, nperseg=nperseg)
freqs2, powers2 = ss.welch(segment2, fs=fs, nperseg=nperseg)
freqs3, powers3 = ss.welch(segment3, fs=fs, nperseg=nperseg)
freqs4, powers4 = ss.welch(segment4, fs=fs, nperseg=nperseg)
freqs5, powers5 = ss.welch(segment5, fs=fs, nperseg=nperseg)


freqs = freqs1 #np.array([freqs1, freqs2, freqs3, freqs4, freqs5])
spectra = np.array([powers1, powers2, powers3, powers4, powers5])

fg = FOOOFGroup(peak_width_limits=[0.5, 8], min_peak_height=0.02, max_n_peaks=10)
fg.fit(freqs, spectra, [4, 50])
fg.print_results()



aper_params = fg.get_params('aperiodic_params')    # shape (n_chans, 2)
print(aper_params)
offsets   = aper_params[:, 0]
exponents = aper_params[:, 1]
print(offsets)
print(exponents)


fg.plot()
plt.show()
# fm = fg.get_fooof(ind=4, regenerate=True)
# fm.print_results()
# fm.plot()
# plt.show()
#
