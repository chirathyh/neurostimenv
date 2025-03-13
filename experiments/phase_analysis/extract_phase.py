import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import pywt
import scipy.signal as ss


REAL = False

if REAL:
    dt = 0.025
    fs = (1 / dt) * 1000
    nperseg = int(fs/2)
    transient = 27000  # in seconds L23Net uses : 2000
    t1 = int(transient/dt)
    EEG = np.loadtxt("../../data/feature_analysis/mdd/EEG_MDD_10.csv", delimiter=",")
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    eeg_data = ss.filtfilt(b, a, EEG[t1:], axis=-1)

    t = np.arange(len(eeg_data)) / fs  # Time vector
    eeg_signal = eeg_data

else:
    # Step 1: Generate synthetic EEG signal
    fs = 1000  # Sampling frequency (Hz)
    t_stop = 2  # Duration of the signal (seconds)
    t = np.linspace(0, t_stop, fs*t_stop)  # Time vector (10 seconds)
    # Synthetic EEG signal with alpha (10 Hz), beta (20 Hz), and theta (5 Hz) waves
    alpha = np.sin(2 * np.pi * 10 * t)  # Alpha wave (10 Hz)
    beta = 0.5 * np.sin(2 * np.pi * 20 * t)  # Beta wave (20 Hz)
    theta = 0.8 * np.sin(2 * np.pi * 5 * t)  # Theta wave (5 Hz)
    # Combine the waves to create a synthetic EEG signal
    eeg_signal = alpha + beta + theta

# Step 2: Analyze phase using the Hilbert transform
analytic_signal = hilbert(eeg_signal)
instantaneous_phase_hilbert = np.angle(analytic_signal)

# Step 3: Analyze phase using Wavelet transform
# Use Continuous Wavelet Transform (CWT) with Morlet wavelet
wavelet = 'cmor'
scales = np.arange(1, 128)  # Scale range for CWT
coefficients, frequencies = pywt.cwt(eeg_signal, scales, wavelet, 1/fs)

# Extract the instantaneous phase from the CWT coefficients
instantaneous_phase_wavelet = np.angle(coefficients)

# Step 4: Plot the results

# Plot the synthetic EEG signal
plt.figure(figsize=(12, 8))

# Original EEG Signal
plt.subplot(3, 1, 1)
plt.plot(t, eeg_signal)
plt.title('EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Phase from Hilbert transform
plt.subplot(3, 1, 2)
plt.plot(t, instantaneous_phase_hilbert)
plt.title('Instantaneous Phase (Hilbert Transform)')
plt.xlabel('Time (s)')
plt.ylabel('Phase (radians)')

# Phase from Wavelet transform (using the frequency of the alpha band)
plt.subplot(3, 1, 3)
plt.plot(t, instantaneous_phase_wavelet[2, :])  # Extracting phase from alpha band (around 10 Hz)
plt.title('Instantaneous Phase (Wavelet Transform)')
plt.xlabel('Time (s)')
plt.ylabel('Phase (radians)')

plt.tight_layout()
plt.show()
