import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np

dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)
transient = 2000  # in seconds L23Net uses : 2000
t1 = int(transient/dt)
print("Sampling Rate:", fs)

# Load CSV file
EEG = np.loadtxt("../../data/feature_analysis/mdd/EEG_MDD_10.csv", delimiter=",")

print("Loaded data:\n", EEG)

# signal filtering
b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)


EEG_freq, EEG_ps = ss.welch(EEG_filt[t1:], fs=fs, nperseg=nperseg)

# Plot EEG signal and PSD
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
# Plot EEG filtered signal
axes[0].plot(np.arange(len(EEG_filt)) * dt, EEG_filt, color='b', label="Filtered EEG")
axes[0].set_ylabel("Amplitude (V)")
axes[0].set_title("Filtered EEG Signal")
axes[0].legend()
axes[0].grid(True)
# Plot Power Spectral Density (PSD)
axes[1].plot(EEG_freq, EEG_ps, color='r', label="PSD")
axes[1].set_xlim(4, 30)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power (dB/Hz)")
axes[1].set_title("Power Spectral Density (PSD)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
