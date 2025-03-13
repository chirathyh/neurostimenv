import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
import scipy.signal as ss

REAL = True

if REAL:
    dt = 0.025
    fs = (1 / dt) * 1000
    nperseg = int(fs / 2)
    transient = 27000  # in seconds
    t1 = int(transient / dt)
    EEG = np.loadtxt("../../data/feature_analysis/mdd/EEG_MDD_10.csv", delimiter=",")
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    eeg_data = ss.filtfilt(b, a, EEG[t1:], axis=-1)

    eeg_signal = eeg_data
    sampling_rate = fs
    print("Sample Rate: ", sampling_rate)

else:
    nperseg = 256  # Window size for Fourier Transform
    EEG = np.random.randn(1000)
    sampling_rate = 1000  # Adjust according to your sampling rate (in Hz)

# Define parameters
frequencies_of_interest = [10, 20, 30]  # Example frequencies in Hz

t = np.arange(EEG.shape[0]) / sampling_rate  # Time vector (in seconds)

# Helper function for bandpass filtering
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    #b, a = butter(order, [low, high], btype='band')
    b, a = ss.butter(N=2, Wn=[lowcut, highcut], btype='bandpass', fs=fs, output='ba')
    return b, a

# Function to compute the Hilbert transform for each frequency of interest
def compute_phase_for_frequencies(signal, fs, frequencies_of_interest):
    phase_dict = {}

    for freq in frequencies_of_interest:
        # Design a bandpass filter for each frequency of interest
        lowcut = freq - 1
        highcut = freq + 1
        b, a = butter_bandpass(lowcut, highcut, fs)
        filtered_signal = filtfilt(b, a, signal)

        # Compute the Hilbert transform and phase
        analytic_signal = hilbert(filtered_signal)
        phase = np.angle(analytic_signal)

        phase_dict[freq] = phase

    return phase_dict

# Compute the synthetic signal (example with 10 Hz)
synthetic_signal = np.sin(2 * np.pi * 10 * t)  # Example synthetic signal with a 10 Hz frequency
synthetic_phase = np.angle(hilbert(synthetic_signal))  # Compute phase using Hilbert Transform

# Compute phases for the frequencies of interest in EEG
eeg_phases = compute_phase_for_frequencies(EEG, sampling_rate, frequencies_of_interest)

# Now, let's compute phase lags between the phases of EEG and synthetic signal
phase_lags = {}
for freq in frequencies_of_interest:
    # Calculate phase lag between EEG signal and synthetic signal
    phase_lag = eeg_phases[freq] - synthetic_phase  # Compute phase difference
    phase_lags[freq] = phase_lag


halfway_point = len(t) // 2


# Plotting Phase Lags for each frequency
plt.figure(figsize=(12, 6))
for freq in frequencies_of_interest:
    plt.subplot(len(frequencies_of_interest), 1, frequencies_of_interest.index(freq) + 1)
    plt.plot(t[:halfway_point], phase_lags[freq][:halfway_point], label=f'Phase Lag at {freq} Hz')
    plt.title(f'Phase Lag at {freq} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Phase Lag (radians)')
    plt.legend()

plt.tight_layout()
plt.show()
