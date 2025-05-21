import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t

dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)
transient = 4000  # ms
t1 = int(transient/dt)


def import_data(folder_path):
    eeg_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    stim_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    filtered_eeg = []
    for eeg_file, stim_file in zip(eeg_files, stim_files):
        df = pd.read_csv(stim_file)
        arm = df['Arm'].values[0]
        if arm != 1:
            conti
        rew = df['Reward'].values[0]

    return eeg, stim


def filter(eeg, t1):
    EEG = np.loadtxt(eeg, delimiter=",")
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)
    return EEG_filt


def calc_psd(EEG_filt, start=None, end=None):
    EEG_freq, EEG_ps = ss.welch(EEG_filt[start:end], fs=fs, nperseg=nperseg)
    return EEG_freq, EEG_ps


def avg_psd(all_psd, all_freqs):
    avg_psd = np.mean(all_psd, axis=0)
    sem_psd = np.std(all_psd, axis=0, ddof=1) / np.sqrt(len(csv_files))
    ci_95 = t.ppf(0.975, df=len(csv_files)-1) * sem_psd
    return all_freqs, avg_psd, ci_95


def plot_psd(data_list):

    plt.figure(figsize=(10, 5))

    for item in data_list:
        freq, psd, color, label = item
        plt.plot(freq, psd, color=color, label=label)

    # Add vertical lines at 8 Hz and 12 Hz
    plt.axvline(x=8, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=12, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=16, color='gray', linestyle='--', alpha=0.7)

    # Add text annotations for frequency bands (Greek notation)
    plt.text(6.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\theta$",
             fontsize=14, color='black', ha='center', fontweight='bold')
    plt.text(10, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\alpha$",
             fontsize=14, color='black', ha='center', fontweight='bold')
    plt.text(20, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\beta$",
             fontsize=14, color='black', ha='center', fontweight='bold')
    plt.xlim(4, 50) # plt.ylim(0, 1e-19)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
    plt.title("Power Spectral Density (PSD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


eeg, stim = import_data(folder_path="../../data/bandit/simnibsbandit2/testing")




colors = ['royalblue', 'mediumseagreen', 'darkorchid', 'deepskyblue', 'limegreen', 'blueviolet']
labels = ['Rest 1', 'Stimulation 1', 'Rest 2', 'Stimulation 2']

plot_data = []

x1 = int(1000/dt)

starts = [0, x1, x1*2, x1*3]
ends = [x1, x1*2, x1*3, None]

EEG_filt = filter(eeg[0], t1)

i = 0
for start, end in zip(starts, ends):
    print(start, end)
    freq, psd = calc_psd(EEG_filt, start=start, end=end)
    plot_data.append([freq, psd, colors[i], labels[i]])
    i += 1

print(plot_data)

plot_psd(plot_data)



# fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
#
# # print(EEG_filt)
# # print(EEG_filt[t1:s1])
# # exit()
#
# axes[0].plot(EEG_filt)
# axes[0].set_ylabel('EEG 1 (µV)')
# axes[0].set_title('Filtered EEG Segment 1')
#
# # axes[1].plot(EEG_filt[s1:t2])
# # axes[1].set_ylabel('EEG 2 (µV)')
# # axes[1].set_title('Filtered EEG Segment 2')
# #
# # axes[2].plot(EEG_filt[t2:s2])
# # axes[2].set_xlabel('Time (s)')
# # axes[2].set_ylabel('EEG 3 (µV)')
# # axes[2].set_title('Filtered EEG Segment 3')
# #
# # axes[3].plot(EEG_filt[s2:])
# # axes[3].set_xlabel('Time (s)')
# # axes[3].set_ylabel('EEG 4 (µV)')
# # axes[3].set_title('Filtered EEG Segment 4')
#
# plt.tight_layout()
# plt.show()



