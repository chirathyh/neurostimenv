import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Define sampling parameters
fs = 1000  # Sampling frequency in Hz
t = np.linspace(0, 1, fs, endpoint=False)  # 1 second duration

# Define sine wave parameters
freqs = [5, 10, 20]  # Frequencies in Hz
amplitudes = [20.0, 10.5, 10.8]  # Amplitudes
phases = [0, np.pi/4, np.pi/2]  # Phases in radians

# Generate the signal
signal = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amplitudes, freqs, phases))

# Introduce a 4th sine wave with the same frequency and amplitude as the first wave but opposite phase
cancel_wave = (amplitudes[0]) * np.sin(2 * np.pi * freqs[0] * t + (phases[0]+np.pi))
cancel_wave2 = (amplitudes[1]/1.1) * np.sin(2 * np.pi * freqs[1] * t + (phases[1]+np.pi))
cancel_wave3 = (amplitudes[2]/2) * np.sin(2 * np.pi * freqs[2] * t + (phases[2]+np.pi))
stim_wave = cancel_wave + cancel_wave2 + cancel_wave3

# Add noise to simulate EEG signal
noise_level = 3  # Adjust noise level as needed
noise = noise_level * np.random.randn(len(t))
signal += noise

new_signal = signal + stim_wave # Summing cancels out the first sine wave

# Compute Power Spectral Density (PSD)
freqs_psd, psd = welch(signal, fs, nperseg=int(fs))
freqs_psd_new, psd_new = welch(new_signal, fs, nperseg=int(fs))

# Limit y-axis to the range of values within xlim
mask = (freqs_psd >= 0) & (freqs_psd <= 40)
psd_limited = psd[mask]
psd_limited_new = psd_new[mask]


# Plot the signals and their PSDs
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# Plot original time-domain signal
axs[0, 0].plot(t, signal, label='Original Signal')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].set_title('Original Simulated EEG')
axs[0, 0].grid()
axs[0, 0].legend()

# Plot original time-domain signal
axs[1, 0].plot(t, stim_wave, label='Cancelling Wave (Stimulation)')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Amplitude')
axs[1, 0].set_title('Cancelling Wave (Stimulation)')
axs[1, 0].grid()
axs[1, 0].legend()

# Plot new time-domain signal
axs[0, 1].plot(t, new_signal, label='New Signal')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 1].set_title('Simulated EEG with Stimulation')
axs[0, 1].grid()
axs[0, 1].legend()

# Plot original PSD
axs[1, 1].semilogy(freqs_psd, psd, label='Original PSD')
axs[1, 1].set_xlabel('Frequency [Hz]')
axs[1, 1].set_ylabel('Power/Frequency [V^2/Hz]')
axs[1, 1].set_title('Original Power Spectral Density (PSD)')


# Plot new PSD
axs[1, 1].semilogy(freqs_psd_new, psd_new, label='New PSD (stimulation)')
axs[1, 1].set_xlabel('Frequency [Hz]')
axs[1, 1].set_ylabel('Power/Frequency [V^2/Hz]')
axs[1, 1].set_title('New Power Spectral Density (PSD)')

axs[1, 1].grid()
axs[1, 1].legend()
axs[1, 1].set_xlim([0, 40])
axs[1, 1].set_ylim([psd_limited.min(), psd_limited.max()]) #psd_limited.max()

# axs[1, 1].grid()
# axs[1, 1].legend()
# axs[1, 1].set_xlim([0, 40])
# axs[1, 1].set_ylim([psd_limited_new.min(), 1])  #psd_limited_new.max()

plt.tight_layout()
plt.show()
