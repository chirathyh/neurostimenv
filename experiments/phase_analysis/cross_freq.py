import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Simulation parameters
fs = 1000  # Sampling frequency (Hz)
T = 5  # Duration (seconds)
t = np.arange(0, T, 1/fs)  # Time vector

# Define endogenous oscillation (target frequency to suppress)
f0 = 10  # Target brain oscillation frequency (Hz)
A0 = 1  # Amplitude
x = A0 * np.sin(2 * np.pi * f0 * t)

# Define tACS signal (anti-phase at the same frequency)
fs_tacs = 10  # tACS frequency (same as f0)
As = 0.5  # tACS amplitude
s = As * np.sin(2 * np.pi * fs_tacs * t + np.pi)  # 180-degree phase shift

# Simulated neural response with destructive interference
alpha = 0.5  # Linear modulation factor
beta = -0.8  # Nonlinear suppression factor (negative for cancellation)
y = x + alpha * s + beta * (x * s)  # Now suppressing instead of reinforcing

# Compute PSD using Welch's method
frequencies, psd_x = welch(x, fs, nperseg=1024)
frequencies, psd_y = welch(y, fs, nperseg=1024)

# Limit frequencies to 0-40 Hz
freq_mask = (frequencies >= 0) & (frequencies <= 40)

# Plot PSD
plt.figure(figsize=(10, 5))
plt.semilogy(frequencies[freq_mask], psd_x[freq_mask], label="Before tACS (Baseline)", linestyle="--")
plt.semilogy(frequencies[freq_mask], psd_y[freq_mask], label="After tACS (Anti-Phase)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Effect of Anti-Phase tACS on 10 Hz Neural Oscillation")
plt.legend()
plt.grid()
plt.show()
