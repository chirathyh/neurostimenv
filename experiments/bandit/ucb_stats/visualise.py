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



def bootstrap_ci(data, num_bootstraps=100, ci=95):
    """
    Compute the bootstrapped mean and confidence interval.

    Parameters:
    - data: NumPy array (each row is a PSD sample, columns are frequency bins)
    - num_bootstraps: Number of bootstrap resamples
    - ci: Confidence interval percentage (default: 95%)

    Returns:
    - mean_psd: Bootstrapped mean PSD
    - ci_lower: Lower bound of the confidence interval
    - ci_upper: Upper bound of the confidence interval
    """
    data = np.array(data)  # Ensure data is a NumPy array
    num_samples = data.shape[0]

    # Bootstrapped means
    boot_means = np.zeros((num_bootstraps, data.shape[1]))

    for i in range(num_bootstraps):
        # Resample with replacement
        resample_idx = np.random.choice(num_samples, num_samples, replace=True)
        resample = data[resample_idx, :]
        boot_means[i, :] = np.mean(resample, axis=0)

    # Compute bootstrapped mean and confidence interval
    mean_psd = np.mean(boot_means, axis=0)
    ci_lower = np.percentile(boot_means, (100 - ci) / 2, axis=0)
    ci_upper = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)

    return mean_psd, ci_lower, ci_upper


def process_eeg(file_path):
    file_list = []
    for i in range(10, 70):
        file_list.append(file_path+str(i)+".csv")

    # Filter coefficients
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    # Lists to store PSDs
    all_psd = []
    all_freqs = None

    for file in file_list:
        print(f"Processing {file}...")
        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        # test only using 1s
        x1 = int(1000/dt)
        EEG_filt = EEG_filt[0:x1]

        freqs, psd = ss.welch(EEG_filt, fs=fs, nperseg=nperseg, noverlap=noverlap)

        # Store results
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    avg_psd, ci_lower, ci_upper = bootstrap_ci(all_psd)
    return all_freqs, avg_psd, ci_lower, ci_upper


def process_bandit_testing(folder_path, selected_arm=1, segment=4):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    # print(sorted(csv_files))
    # print(sorted(reward_files))

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    all_psd = []
    all_freqs = None

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        #print(file, reward_file)
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]

        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue

        #print(file, reward_file)
        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        x1 = int(1000/dt)

        rew = features.reward_func_simple(np.array(EEG_filt[x1*4 : ]), fs)
        # if rew < -1.82668915:  # 75% 1.3929; 80%: 1.5272; 78% 1.5133; 90%: 2.0399; 85% 1.7268
        #     continue


        # different protocol stages
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

        freqs, psd = ss.welch(EEG_segment, fs=fs, nperseg=nperseg)  #, noverlap=noverlap
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    print("total examples: ", len(all_psd))

    ci_lower, ci_upper = [], []
    avg_psd, ci_lower, ci_upper = bootstrap_ci(all_psd)  # bootstrapped values
    #avg_psd = np.mean(np.array(all_psd), axis=0)  # simple mean

    return all_freqs, avg_psd, ci_lower, ci_upper


def save_calculated_psd_healthy_mdd(all_freqs, avg_psd, ci_lower, ci_upper,
                                    all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h):
    np.save('../../data/bandit/pre_calculated/mdd_all_freqs.npy', all_freqs)
    np.save('../../data/bandit/pre_calculated/mdd_avg_psd.npy', avg_psd)
    np.save('../../data/bandit/pre_calculated/mdd_ci_lower.npy', ci_lower)
    np.save('../../data/bandit/pre_calculated/mdd_ci_upper.npy', ci_upper)
    np.save('../../data/bandit/pre_calculated/healthy_all_freqs.npy', all_freqs_h)
    np.save('../../data/bandit/pre_calculated/healthy_avg_psd.npy', avg_psd_h)
    np.save('../../data/bandit/pre_calculated/healthy_ci_lower.npy', ci_lower_h)
    np.save('../../data/bandit/pre_calculated/healthy_ci_upper.npy', ci_upper_h)


def load_calculated_psd_healthy_mdd():
    all_freqs = np.load('../../../data/bandit/pre_calculated/mdd_all_freqs.npy')
    avg_psd = np.load('../../../data/bandit/pre_calculated/mdd_avg_psd.npy')
    ci_lower = np.load('../../../data/bandit/pre_calculated/mdd_ci_lower.npy')
    ci_upper = np.load('../../../data/bandit/pre_calculated/mdd_ci_upper.npy')
    all_freqs_h = np.load('../../../data/bandit/pre_calculated/healthy_all_freqs.npy')
    avg_psd_h = np.load('../../../data/bandit/pre_calculated/healthy_avg_psd.npy')
    ci_lower_h = np.load('../../../data/bandit/pre_calculated/healthy_ci_lower.npy')
    ci_upper_h = np.load('../../../data/bandit/pre_calculated/healthy_ci_upper.npy')
    return all_freqs, avg_psd, ci_lower, ci_upper, all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h

#### MAIN CODE ###


def plot_sfft_bandit_testing(folder_path, selected_arm=1, segment=4):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_66.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_66.csv"))
    print(sorted(csv_files))
    print(sorted(reward_files))

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    all_psd = []
    all_freqs = None

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        print(file, reward_file)
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]
        print(rew)
        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue


        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        x1 = int(1000/dt)

        selection_reward = features.reward_func_simple(np.array(EEG_filt[0:x1]), fs)
        if selection_reward >= -0.09264591737143694 :
            continue

        # different protocol stages

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

    nperseg  = fs       # ~0.1 s windows
    noverlap = 0       # 50% overlap

    # print(EEG_segment.shape)

    # 2) Compute STFT
    f, t, Zxx = stft(EEG_segment,
                     fs=fs,
                     window=('tukey', 0.1),  #'hann',
                     nperseg=nperseg,
                     noverlap=noverlap,
                     detrend=False,
                     return_onesided=True,
                     boundary='zeros',
                     padded=False )
    Sxx = np.abs(Zxx)
    Sxx = 10 * np.log10(np.abs(Zxx) + np.finfo(float).eps)

    # mask = f <= 30
    # vmin, vmax = np.percentile(Sxx[mask], [5, 95])

    # 3) Limit to frequencies ≤100 Hz
    freq_mask = f <= 30
    f_plot    = f[freq_mask]
    t_plot    = t
    Sxx_plot  = Sxx[freq_mask, :]

    # 4) Plot
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t_plot, f_plot, Sxx_plot, shading='gouraud', cmap='viridis')
    #plt.pcolormesh(t, f[mask], Sxx[mask], vmin=vmin, vmax=vmax, shading='gouraud', cmap='plasma')
    plt.colorbar(label='Power (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    # plt.title('Short-Term FFT (STFT) of 5 s EEG (fs=40 kHz)')
    plt.tight_layout()
    plt.show()


SELECTED_ARM = 1
SEGEMENT = 1
AMP = [1, 2, 4, 2, 2, 15]  # mA
FREQ = [8, 8, 8, 10, 40, 77.5]  # Hz


# plot_sfft_bandit_testing(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=-1)
# exit()


# all_freqs, avg_psd, ci_lower, ci_upper = process_eeg(file_path="../../data/feature_analysis/mdd/EEG_MDD_")
# all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h = process_eeg(file_path="../../data/feature_analysis/healthy/EEG_HEALTHY_")
all_freqs, avg_psd, ci_lower, ci_upper, all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h = load_calculated_psd_healthy_mdd()

all_freqs_b, avg_psd_b, ci_lower_b, ci_upper_b = process_bandit_testing(folder_path="../../../data/bandit/simnibsbandit3/training",
                                                                        selected_arm=SELECTED_ARM, segment=5)


all_freqs_seg1, avg_psd_seg1, ci_lower_seg1, ci_upper_seg1 = process_bandit_testing(folder_path="../../../data/bandit/simnibsbandit3/training",
                                                                        selected_arm=SELECTED_ARM, segment=1)


# testing whether somehow signal len has an effect: it does not
# all_freqs_b, avg_psd_b, ci_lower_b, ci_upper_b = process_bandit_testing(folder_path="../../data/bandit/hopefulbandit/testing",
#                                                                         selected_arm=5, segment=4)


# Plot EEG signal and PSD
plt.figure(figsize=(10, 5))
colors = ['royalblue', 'mediumseagreen', 'darkorchid', 'deepskyblue', 'limegreen', 'blueviolet']

# Depression group
plt.plot(all_freqs, avg_psd, color='r', linestyle='--', label="Depression Baseline")  # , linestyle='--'
# plt.fill_between(all_freqs, ci_lower, ci_upper, color='r', alpha=0.3)
#
# # Healthy group
plt.plot(all_freqs_h, avg_psd_h, color='k', linestyle='--', label="Healthy Baseline") #
#plt.fill_between(all_freqs_h, ci_lower_h, ci_upper_h, color='k', alpha=0.3)

# bandit results
#plt.plot(all_freqs_b, avg_psd_b, color='g', label=f"Bandit Stimulation: {AMP[SELECTED_ARM]}mA, {FREQ[SELECTED_ARM]}Hz")
plt.plot(all_freqs_b, avg_psd_b, color='tab:green', label=f"Bandit Stimulation: EEG Segment 5")
plt.fill_between(all_freqs_b, ci_lower_b, ci_upper_b, color='tab:green', alpha=0.3)


plt.plot(all_freqs_seg1, avg_psd_seg1, color='tab:blue', label=f"Bandit Stimulation: EEG Segment 1")
plt.fill_between(all_freqs_seg1, ci_lower_seg1, ci_upper_seg1, color='tab:blue', alpha=0.3)

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
# plt.title("Power Spectral Density (PSD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
