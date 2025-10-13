import os
import glob

import os
import re
import sys
from decouple import config
import statistics

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)


import statsmodels
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t
from scipy.signal import stft

import reward_func
#from env.eeg import features


from experiments.case_study.configs import get_configs

dt, fs, nperseg, _, t1 = get_configs()


def process_bandit_testing(folder_path, selected_arm=1, segment=4, preprocessed_eeg=True, filter=False, filter_threshold=1):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    # print(sorted(csv_files))
    # print(sorted(reward_files))

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    reward_values = []
    reward_values_final_segment = []

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        #print(file, reward_file)
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]

        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue

        EEG = np.loadtxt(file, delimiter=",")

        if preprocessed_eeg:
            EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)
        else:
            EEG_filt = EEG[t1:]

        x1 = int(1000/dt)

        # filter
        if filter:
            rew = reward_func.reward_func_simple(np.array(EEG_filt[x1*4 : ]), fs)
            if rew < filter_threshold:  # 75% 1.3929; 80%: 1.5272; 78% 1.5133; 90%: 2.0399; 85% 1.7268  # best: -1.82668915
                continue
        # print(file, reward_file)

        reward_value = reward_func.reward_func_simple(np.array(EEG_filt[0:x1]), fs)
        reward_values.append(reward_value)
        reward_final = reward_func.reward_func_simple(np.array(EEG_filt[x1*4 : ]), fs)
        reward_values_final_segment.append(reward_final)


    return reward_values, reward_values_final_segment
