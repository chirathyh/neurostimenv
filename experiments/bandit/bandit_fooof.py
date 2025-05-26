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


    aper_params = fg.get_params('aperiodic_params')    # shape (n_chans, 2)
    offsets   = aper_params[:, 0][:, None]
    exponents = aper_params[:, 1][:, None]
    aper_curves = 10**offsets * (freqs**(-exponents))

        # 4) Mean curves
    mean_psd       = spectra.mean(axis=0)
    mean_aperiodic = aper_curves.mean(axis=0)

    per_curves = np.log(spectra) - np.log(aper_curves)
    mean_periodic  = per_curves.mean(axis=0)

    # return freqs, mean_aperiodic, mean_periodic
    n_boot = 1000
    alpha = 0.05
    boot_aper = []
    boot_per = []
    for _ in range(n_boot):
        idx = np.random.choice(len(aper_curves), size=len(aper_curves), replace=True)
        boot_aper.append(np.mean(aper_curves[idx], axis=0))
        boot_per.append(np.mean(per_curves[idx], axis=0))

    boot_aper = np.array(boot_aper)
    ci_lower_aper = np.percentile(boot_aper, 100 * (alpha / 2), axis=0)
    ci_upper_aper = np.percentile(boot_aper, 100 * (1 - alpha / 2), axis=0)

    boot_per = np.array(boot_per)
    ci_lower_per = np.percentile(boot_per, 100 * (alpha / 2), axis=0)
    ci_upper_per = np.percentile(boot_per, 100 * (1 - alpha / 2), axis=0)

    return freqs, mean_aperiodic, ci_lower_aper, ci_upper_aper, mean_periodic, ci_lower_per, ci_upper_per




SELECTED_ARM = 1
SEGEMENT = 1

mdd_eeg_data = get_bandit_eeg(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=1)
freqs, mean_aperiodic, ci_lower_aper, ci_upper_aper, mean_periodic, ci_lower_per, ci_upper_per = get_aperiodic_period(mdd_eeg_data)

mdd_eeg_data_5 = get_bandit_eeg(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=5)
freqs_5, mean_aperiodic_5, ci_lower_aper_5, ci_upper_aper_5, mean_periodic_5, ci_lower_per_5, ci_upper_per_5 = get_aperiodic_period(mdd_eeg_data_5)


# aperiodic
plt.figure(figsize=(10,5))

plt.plot(freqs,  mean_aperiodic,   lw=2, label='Bandit stimulation: Segment 1')
plt.fill_between(freqs, ci_lower_aper, ci_upper_aper, alpha=0.3)

plt.plot(freqs_5,  mean_aperiodic_5, lw=2, color='g', label='Bandit stimulation: Segment 5')
plt.fill_between(freqs, ci_lower_aper_5, ci_upper_aper_5, color='g',alpha=0.3)

# Add vertical lines at 8 Hz and 12 Hz
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=12, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=16, color='gray', linestyle='--', alpha=0.7)

plt.text(6.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\theta$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.text(10, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\alpha$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.text(20, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\beta$",
         fontsize=14, color='black', ha='center', fontweight='bold')


plt.xlim(4, 50) #
plt.ylim(0, 0.8 * 1e-19)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# periodic
plt.figure(figsize=(10,5))

plt.plot(freqs,  mean_periodic,   lw=2, color='tab:blue', label='Bandit stimulation: Segment 1')
plt.fill_between(freqs, ci_lower_per, ci_upper_per, color='tab:blue', alpha=0.3)

plt.plot(freqs_5,  mean_periodic_5, lw=2, color='tab:green', label='Bandit stimulation: Segment 5')
plt.fill_between(freqs, ci_lower_per_5, ci_upper_per_5, color='tab:green',alpha=0.3)

# Add vertical lines at 8 Hz and 12 Hz
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=12, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=16, color='gray', linestyle='--', alpha=0.7)

plt.ylim(-0.4, 1.8)

plt.text(6.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\theta$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.text(10, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\alpha$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.text(20, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\beta$",
         fontsize=14, color='black', ha='center', fontweight='bold')


plt.xlim(4, 50) #

plt.xlabel("Frequency (Hz)")
plt.ylabel(r'$log(Power) - log(Power: Aperiodic)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
