import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t
from mne.time_frequency import psd_array_welch
from fooof import FOOOF



dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)
transient = 4000  # ms; first 4s is removed from the EEG (triansient phase)
t1 = int(transient/dt)
print("Sampling Rate:", fs)
print("npserg", nperseg)
sfreq = fs

def get_bandit_eeg(folder_path, selected_arm=1, segment=4):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    # print(sorted(csv_files))
    # print(sorted(reward_files))

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


def get_periodic_aperiodic(eeg_data):
    # --------------------------------
    # Parameters for spectral analysis
    # --------------------------------
    freq_range = [4, 30]  # Hz range for PSD
    n_boot = 1000          # number of bootstrap samples
    alpha = 0.05          # for 95% confidence interval

    # ---------------------------------------------------
    # 1) Compute power spectral densities (in dB) per epoch
    # ---------------------------------------------------
    psds, freqs = psd_array_welch(
        eeg_data,
        sfreq=sfreq,
        fmin=freq_range[0],
        fmax=freq_range[1],
        n_fft=int(sfreq),
        verbose=False
    )
    psds_db = np.log10(psds)
    log_freqs = np.log10(freqs)

    # ---------------------------------------------------
    # 2) Fit a 1/f "aperiodic" component via linear regression
    #    in log-log space, subtract to get periodic residuals
    # ---------------------------------------------------
    aper_psds = []
    periodic_psds = []

    for psd in psds_db:
        m, b = np.polyfit(log_freqs, psd, 1)
        ap_fit = m * log_freqs + b
        periodic = psd - ap_fit
        aper_psds.append(ap_fit)
        periodic_psds.append(periodic)

    aper_psds = np.array(aper_psds)
    periodic_psds = np.array(periodic_psds)

    # -------------------------------------------------------
    # 3) Bootstrap to estimate mean and 95% CI across epochs
    # -------------------------------------------------------
    boot_aper = []
    boot_per = []
    for _ in range(n_boot):
        idx = np.random.choice(len(aper_psds), size=len(aper_psds), replace=True)
        boot_aper.append(np.mean(aper_psds[idx], axis=0))
        boot_per.append(np.mean(periodic_psds[idx], axis=0))

    boot_aper = np.array(boot_aper)
    boot_per = np.array(boot_per)

    mean_aper = np.mean(aper_psds, axis=0)
    ci_lower_aper = np.percentile(boot_aper, 100 * (alpha / 2), axis=0)
    ci_upper_aper = np.percentile(boot_aper, 100 * (1 - alpha / 2), axis=0)

    mean_per = np.mean(periodic_psds, axis=0)
    ci_lower_per = np.percentile(boot_per, 100 * (alpha / 2), axis=0)
    ci_upper_per = np.percentile(boot_per, 100 * (1 - alpha / 2), axis=0)

    return freqs, ci_lower_aper, ci_upper_aper, mean_aper, mean_per, ci_lower_per, ci_upper_per


SELECTED_ARM = 1
SEGEMENT = 1

mdd_eeg_data = get_bandit_eeg(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=1)
freqs, ci_lower_aper, ci_upper_aper, mean_aper, mean_per, ci_lower_per, ci_upper_per = get_periodic_aperiodic(mdd_eeg_data)

stim_eeg_data = get_bandit_eeg(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=5)
stim_freqs, stim_ci_lower_aper, stim_ci_upper_aper, stim_mean_aper, stim_mean_per, stim_ci_lower_per, stim_ci_upper_per = get_periodic_aperiodic(stim_eeg_data)

# Aperiodic figure
plt.figure(figsize=(8, 5))
plt.fill_between(freqs, ci_lower_aper, ci_upper_aper, alpha=0.3)
plt.plot(freqs, mean_aper, label='MDD')

plt.fill_between(stim_freqs, stim_ci_lower_aper, stim_ci_upper_aper, alpha=0.3)
plt.plot(stim_freqs, stim_mean_aper, label='MDD + After Stimulation')


plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
plt.title('Aperiodic Component Mean ±95% CI')
plt.legend()
plt.tight_layout()

# Periodic figure
plt.figure(figsize=(8, 5))

plt.plot(freqs, mean_per, label='MDD')
plt.fill_between(freqs, ci_lower_per, ci_upper_per, alpha=0.3)

plt.plot(stim_freqs, stim_mean_per, label='MDD + After Stimulation')
plt.fill_between(stim_freqs, stim_ci_lower_per, stim_ci_upper_per, alpha=0.3)


plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$Power(\text{V}^2/\text{Hz})$')
plt.title('Periodic Component Mean ±95% CI')
plt.legend()
plt.tight_layout()

plt.show()

