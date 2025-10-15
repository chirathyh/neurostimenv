import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import scipy.signal as ss

# Create a synthetic EEG signal (sum of sinusoids with different frequencies)
# fs = 1000  # Sampling frequency (Hz)
# t = np.arange(0, 2, 1/fs)  # Time vector (2 seconds)
# f1 = 10  # Frequency of first signal (Hz)
# f2 = 50  # Frequency of second signal (Hz)
# f3 = 200  # Frequency of third signal (Hz)
#
# p1 = 0
# p2 = np.pi/4
# p3 = np.pi/2
#
# nperseg=256
# # Create a synthetic EEG signal as a sum of sinusoids
# signal = np.sin(2 * np.pi * f1 * t+p1) + 0.5 * np.sin(2 * np.pi * f2 * t+p2) + 0.3 * np.sin(2 * np.pi * f3 * t+p3)



dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)
transient = 4000  # in seconds L23Net uses : 2000
t1 = int(transient/dt)
EEG = np.loadtxt("../../data/bandit/nbandit2/testing/EEG_BANDIT_1060.csv", delimiter=",")
b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
eeg_data = ss.filtfilt(b, a, EEG[t1:], axis=-1)

t = np.arange(len(eeg_data)) / fs  # Time vector
signal = eeg_data


# Compute the Short-Time Fourier Transform (STFT)
f, t_spec, Zxx = spectrogram(signal, fs, nperseg=nperseg)

# Compute the instantaneous phase for each frequency
instantaneous_phase = np.angle(Zxx)

# Plot the signal and the instantaneous phase


# Limit frequency range to 1-50 Hz
freq_range = (f >= 1) & (f <= 50)  # Create a mask for frequencies 1-50 Hz
f_limited = f[freq_range]  # Frequencies in the range 1-50 Hz
instantaneous_phase_limited = instantaneous_phase[freq_range, :]  # Phase values for the selected frequencies


plt.figure(figsize=(12, 6))

# Plot the original signal
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Synthetic EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the instantaneous phase over time for each frequency
plt.subplot(2, 1, 2)
plt.pcolormesh(t_spec, f_limited, instantaneous_phase_limited, shading='auto')
plt.title('Instantaneous Phase of Individual Frequencies Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Phase (radians)')

plt.tight_layout()
plt.show()
