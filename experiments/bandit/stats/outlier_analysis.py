import os
import glob
import sys
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t, shapiro, ttest_ind, mannwhitneyu
from scipy.signal import stft
from scipy import stats
from env.eeg import features  # if needed elsewhere

from experiments.bandit.stats.configs import get_configs
dt, fs, nperseg, _, t1 = get_configs()


def process_bandit_testing(folder_path, selected_arm=1, segment=4):
    """
    Process EEG files and return frequencies and list of PSD arrays for chosen segment.
    """
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    # Design bandpass filter
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    all_psd = []
    all_freqs = None

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]
        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue

        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        x1 = int(1000 / dt)
        # Filtering criteria; example: skip very negative rewards

        rew = features.reward_func_simple(np.array(EEG_filt[x1*4 : ]), fs)
        if rew < -1.82668915:  # 75% 1.3929; 80%: 1.5272; 78% 1.5133; 90%: 2.0399; 85% 1.7268
            plt.plot(EEG_filt, 'r')
            filename = os.path.join('g/', f"{abs(rew):.4f}.png")
            plt.savefig(filename)
            continue

        else:
            plt.plot(EEG_filt, 'b')
            filename = os.path.join('g/', f"{abs(rew):.4f}.png")
            plt.savefig(filename)

        # if rew < -2.00017065:  # 75% 1.3929; 80%: 1.5272; 78% 1.5133; 90%: 2.0399; 85% 1.7268
        #     continue

        # Select segment
        if segment == 1:
            EEG_segment = EEG_filt[0:x1]
        elif segment == 2:
            EEG_segment = EEG_filt[x1: x1*2]
        elif segment == 3:
            EEG_segment = EEG_filt[x1*2: x1*3]
        elif segment == 4:
            EEG_segment = EEG_filt[x1*3: x1*4]
        elif segment == 5:
            EEG_segment = EEG_filt[x1*4:]
        elif segment == -1:
            EEG_segment = EEG_filt
        else:
            raise ValueError("Invalid segment number")

        freqs, psd = ss.welch(EEG_segment, fs=fs, nperseg=nperseg)  #, noverlap=noverlap
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    return all_freqs, all_psd


if __name__ == "__main__":
    SELECTED_ARM = 1
    base_folder = "../../../data/bandit/simnibsbandit3/training"
    all_freqs_seg5, all_psdb_seg5 = process_bandit_testing(folder_path=base_folder, selected_arm=SELECTED_ARM, segment=5)
    all_freqs_seg1, all_psd_seg1 = process_bandit_testing(folder_path=base_folder, selected_arm=SELECTED_ARM, segment=1)


