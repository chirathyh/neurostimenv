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
        # if rew < -1.5:
        #     continue

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


def get_aperiodic_period(mdd_eeg_data):

    spectra = []
    for i in range(0, len(mdd_eeg_data)):
        freqs, powers = ss.welch(mdd_eeg_data[i], fs=fs, nperseg=nperseg)
        spectra.append(powers)
    spectra = np.array(spectra)
    print(freqs.shape)
    print(spectra.shape)

    fg = FOOOFGroup(peak_width_limits=[0.5, 8], min_peak_height=0.02, max_n_peaks=10)
    fg.fit(freqs, spectra, [4, 50])
    fg.print_results()
    # fg.plot()
    # plt.show()

    # fm = fg.get_fooof(ind=4, regenerate=True)
    # fm.print_results()
    # fm.plot()
    # plt.show()


    aper_params = fg.get_params('aperiodic_params')    # shape (n_chans, 2)
    offsets   = aper_params[:, 0][:, None]
    exponents = aper_params[:, 1][:, None]
    aper_curves = 10**offsets * (freqs**(-exponents))

    # --- 5) Compute mean PSD, mean aperiodic, and mean periodic ---
    mean_psd       = np.log(spectra).mean(axis=0)

    #mean_aperiodic = aper_curves.mean(axis=0)
    # mean_periodic  = mean_psd - mean_aperiodic
    #
    # #log_mean_psd = np.log(mean_psd)
    # log_mean_aperiodic = np.log(mean_aperiodic)


    # --- 6) Plot the results ---
    plt.figure(figsize=(12, 5))

    # Panel 1: Mean PSD & 1/f fit
    plt.subplot(1, 2, 1)
    plt.plot(np.log(freqs), mean_psd,       label='Mean PSD',           lw=2)
    #plt.plot(freqs, log_mean_aperiodic, ls='--', label='Aperiodic (1/f)', lw=2)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.xlim(1, 50)
    plt.ylim(-50, -40)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (a.u.)')
    plt.title('Mean PSD with 1/f Aperiodic Fit')
    plt.legend()
    plt.grid(which='both', ls=':')

    # Panel 2: Mean Periodic Component
    # plt.subplot(1, 2, 2)
    # plt.loglog(freqs, mean_periodic.clip(min=1e-30),
    #            ls='-.', lw=2, label='Mean Periodic Residual')
    # plt.xlim(1, 50)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Residual (a.u.)')
    # plt.title('Mean Periodic Component (PSD - 1/f)')
    # plt.legend()
    # plt.grid(which='both', ls=':')

    plt.tight_layout()
    plt.show()

    exit()



SELECTED_ARM = 1
SEGEMENT = 1

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
offsets   = aper_params[:, 0][:, None]
exponents = aper_params[:, 1][:, None]

plt.figure(figsize=(10, 5))
plt.scatter(offsets, offsets)
plt.show()

exit()
fg.plot()
plt.show()
fm = fg.get_fooof(ind=4, regenerate=True)
fm.print_results()
fm.plot()
plt.show()

# log -log plot. Power vs frequency
# plt.figure(figsize=(10, 5))
# plt.loglog(freqs1, powers1, color='r', linestyle='--', label="Segment 1")
# plt.loglog(freqs2, powers2, color='b', linestyle='--', label="Segment 2")
# plt.loglog(freqs3, powers3, color='k', linestyle='--', label="Segment 3")
# plt.loglog(freqs4, powers4, color='y', linestyle='--', label="Segment 4")
# plt.loglog(freqs5, powers5, color='m', linestyle='--', label="Segment 5")
# plt.xlim(0, 100) # plt.ylim(0, 1e-19)
# plt.ylim(1e-23, 1e-18)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# normal plot. Power (dB) vs frequency
# plt.figure(figsize=(10, 5))
# plt.plot(freqs1, 10 * np.log10(powers1), color='r', linestyle='--', label="Segment 1")
# plt.plot(freqs2, 10 * np.log10(powers2), color='b', linestyle='--', label="Segment 2")
# plt.plot(freqs3, 10 * np.log10(powers3), color='k', linestyle='--', label="Segment 3")
# plt.plot(freqs4, 10 * np.log10(powers4), color='y', linestyle='--', label="Segment 4")
# plt.plot(freqs5, 10 * np.log10(powers5), color='m', linestyle='--', label="Segment 5")
# plt.xlim(0, 40) # plt.ylim(0, 1e-19)
# plt.ylim(-210, -185)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power (dB)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# exit()

# normal plot, freqs in log scale
# plt.figure(figsize=(10, 5))
# plt.plot(freqs1, powers1, color='r', linestyle='--', label="Segment 1")
# plt.plot(freqs2, powers2, color='b', linestyle='--', label="Segment 2")
# plt.plot(freqs3, powers3, color='k', linestyle='--', label="Segment 3")
# plt.plot(freqs4, powers4, color='y', linestyle='--', label="Segment 4")
# plt.plot(freqs5, powers5, color='m', linestyle='--', label="Segment 5")
# plt.xscale('log')
# plt.xlim(0, 50) # plt.ylim(0, 1e-19)
# # plt.ylim(1e-23, 1e-17)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
