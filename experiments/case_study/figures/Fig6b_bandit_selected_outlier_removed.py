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
from experiments.case_study.configs import get_configs
from experiments.case_study import reward_func
from experiments.case_study.bootstrap import bootstrap_ci

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
        freqs, psd = ss.welch(EEG_filt, fs=fs, nperseg=nperseg)

        # Store results
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    avg_psd, ci_lower, ci_upper = bootstrap_ci(all_psd)
    return all_freqs, avg_psd, ci_lower, ci_upper


def process_bandit_testing(folder_path, selected_arm=1, segment=4):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    all_psd = []
    all_freqs = None

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        print(file, reward_file)
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]

        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue

        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        # different protocol stages
        x1 = int(1000/dt)

        rew = reward_func.reward_func_simple(np.array(EEG_filt[x1*4 : ]), fs)
        if rew < -1.82668915:  # 75% 1.3929; 80%: 1.5272; 78% 1.5133; 90%: 2.0399; 85% 1.7268
            continue

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

        freqs, psd = ss.welch(EEG_segment, fs=fs, nperseg=nperseg)
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    print("total examples: ", len(all_psd))

    avg_psd, ci_lower, ci_upper = bootstrap_ci(all_psd)
    return all_freqs, avg_psd, ci_lower, ci_upper


def save_calculated_psd_healthy_mdd(FOLDER, all_freqs, avg_psd, ci_lower, ci_upper,
                                    all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h):
    np.save(FOLDER + '/mdd_all_freqs.npy', all_freqs)
    np.save(FOLDER + 'mdd_avg_psd.npy', avg_psd)
    np.save(FOLDER + '/mdd_ci_lower.npy', ci_lower)
    np.save(FOLDER + 'mdd_ci_upper.npy', ci_upper)
    np.save(FOLDER + '/healthy_all_freqs.npy', all_freqs_h)
    np.save(FOLDER + '/healthy_avg_psd.npy', avg_psd_h)
    np.save(FOLDER + '/healthy_ci_lower.npy', ci_lower_h)
    np.save(FOLDER + '/healthy_ci_upper.npy', ci_upper_h)


def load_calculated_psd_healthy_mdd(FOLDER):
    all_freqs = np.load(FOLDER + '/mdd_all_freqs.npy')
    avg_psd = np.load(FOLDER + '/mdd_avg_psd.npy')
    ci_lower = np.load(FOLDER + '/mdd_ci_lower.npy')
    ci_upper = np.load(FOLDER + '/mdd_ci_upper.npy')
    all_freqs_h = np.load(FOLDER + '/healthy_all_freqs.npy')
    avg_psd_h = np.load(FOLDER + '/healthy_avg_psd.npy')
    ci_lower_h = np.load(FOLDER + '/healthy_ci_lower.npy')
    ci_upper_h = np.load(FOLDER + '/healthy_ci_upper.npy')
    return all_freqs, avg_psd, ci_lower, ci_upper, all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h


#### MAIN CODE ###
SELECTED_ARM = 1
SEGEMENT = 1
AMP = [1, 2, 4, 2, 2, 15]  # mA
FREQ = [8, 8, 8, 10, 40, 77.5]  # Hz


HEALTHY_EEG = "../data/healthy/EEG_HEALTHY_"
MDD_EEG = "../data/mdd/EEG_MDD_"
BANDIT = "../data/bandit/training"
FOLDER = "../data/pre-calculated"

# all_freqs, avg_psd, ci_lower, ci_upper = process_eeg(file_path=MDD_EEG)
# all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h = process_eeg(file_path=HEALTHY_EEG)
# save_calculated_psd_healthy_mdd(FOLDER, all_freqs, avg_psd, ci_lower, ci_upper, all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h)
all_freqs, avg_psd, ci_lower, ci_upper, all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h = load_calculated_psd_healthy_mdd(FOLDER)
all_freqs_b, avg_psd_b, ci_lower_b, ci_upper_b = process_bandit_testing(folder_path=BANDIT, selected_arm=SELECTED_ARM, segment=5)
all_freqs_seg1, avg_psd_seg1, ci_lower_seg1, ci_upper_seg1 = process_bandit_testing(folder_path=BANDIT, selected_arm=SELECTED_ARM, segment=1)


# Plot EEG signal and PSD
fig = plt.figure(figsize=(10, 5))
colors = ['royalblue', 'mediumseagreen', 'darkorchid', 'deepskyblue', 'limegreen', 'blueviolet']

# Depression group
plt.plot(all_freqs, avg_psd, color='r', linestyle='--', label="Depression Baseline")  # , linestyle='--'
# plt.fill_between(all_freqs, ci_lower, ci_upper, color='r', alpha=0.3)

# # Healthy group
plt.plot(all_freqs_h, avg_psd_h, color='k', linestyle='--', label="Healthy Baseline") #
# plt.fill_between(all_freqs_h, ci_lower_h, ci_upper_h, color='k', alpha=0.3)

# bandit results
plt.plot(all_freqs_seg1, avg_psd_seg1, color='tab:blue', label=f"EEG Segment-1 (pre-intervention)")
plt.fill_between(all_freqs_seg1, ci_lower_seg1, ci_upper_seg1, color='tab:blue', alpha=0.3)

#plt.plot(all_freqs_b, avg_psd_b, color='g', label=f"Bandit Stimulation: {AMP[SELECTED_ARM]}mA, {FREQ[SELECTED_ARM]}Hz")
plt.plot(all_freqs_b, avg_psd_b, color='tab:green', label=f"EEG Segment-5 (post-intervention)")
plt.fill_between(all_freqs_b, ci_lower_b, ci_upper_b, color='tab:green', alpha=0.3)

# Add vertical lines at 8 Hz and 12 Hz
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=12, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=16, color='gray', linestyle='--', alpha=0.7)

# Add text annotations for frequency bands (Greek notation)
plt.text(6.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\theta$", fontsize=14, color='black',
         ha='center', fontweight='bold')
plt.text(10, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\alpha$", fontsize=14, color='black',
         ha='center', fontweight='bold')
plt.text(20, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\beta$",fontsize=14, color='black',
         ha='center', fontweight='bold')

plt.xlim(4, 50) # plt.ylim(0, 1e-19)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
# plt.title("Power Spectral Density (PSD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig.savefig('png/Fig6b.png', dpi=600, bbox_inches='tight', pad_inches=0.02, facecolor='auto', transparent=False)
plt.show()
