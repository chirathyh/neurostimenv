import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import scipy.signal as ss

# Load EEG data and apply bandpass filter
dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs / 2)
transient = 26500  # in seconds
t1 = int(transient / dt)
EEG = np.loadtxt("../../data/feature_analysis/mdd/EEG_MDD_10.csv", delimiter=",")
b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
eeg_data = ss.filtfilt(b, a, EEG[t1:], axis=-1)

eeg_signal = eeg_data
sampling_rate = fs
print("Sample Rate: ", sampling_rate)
# exit()

# Frequencies of interest
frequencies_of_interest = [10, 20, 30]  # Example frequencies in Hz

# Function to get phase at a given frequency
def get_phase_at_frequency(eeg_signal, frequency, sampling_rate):
    nyquist = 0.5 * sampling_rate
    low_cutoff = frequency - 1
    high_cutoff = frequency + 1

    # Apply bandpass filter using scipy signal
    def butter_bandpass(lowcut, highcut, fs, order=4):
        b, a = ss.butter(N=2, Wn=[lowcut, highcut], btype='bandpass', fs=fs, output='ba')
        return b, a

    b, a = butter_bandpass(low_cutoff, high_cutoff, sampling_rate)
    filtered_signal = ss.filtfilt(b, a, eeg_signal)

    # Check for small values in the filtered signal
    if np.any(np.abs(filtered_signal) < 1e-10):
        print(f"Warning: Filtered signal at {frequency} Hz contains small values or zeros.")

    # Compute the analytic signal using Hilbert transform
    analytic_signal = hilbert(filtered_signal)

    # Check for NaN values in the analytic signal
    if np.any(np.isnan(analytic_signal)):
        print(f"Warning: Analytic signal at {frequency} Hz contains NaN values.")

    instantaneous_phase = np.angle(analytic_signal)
    return instantaneous_phase

# Plotting phases of multiple frequencies
plt.figure(figsize=(10, 6))
time = np.arange(len(eeg_signal)) / sampling_rate  # time in seconds

for freq in frequencies_of_interest:
    phase = get_phase_at_frequency(eeg_signal, freq, sampling_rate)
    plt.plot(time, phase, label=f'Frequency: {freq} Hz')

plt.title("Instantaneous Phase at Different Frequencies")
plt.xlabel("Time (s)")
plt.ylabel("Phase (radians)")
plt.legend()
plt.show()
