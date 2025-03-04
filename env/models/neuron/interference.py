import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def four_sphere_potential(I, source_loc, electrode_loc, radii, sigmas, N_max=50):
    """
    Compute the EEG potential at electrode_loc (in m) due to a point current source
    I (in A) located at source_loc (in m) using a simplified series approximation of the
    four-sphere head model.

    Parameters:
      I            : Current amplitude (in A)
      source_loc   : 3-element array [x, y, z] in meters for the current source.
      electrode_loc: 3-element array [x, y, z] in meters for the EEG electrode.
      radii        : 4-element array [R1, R2, R3, R4] in meters for the four spheres.
      sigmas       : 4-element array of conductivities (S/m) for each layer.
      N_max        : Number of terms in the series expansion (default 50)

    Returns:
      V            : Potential at the electrode (in Volts)

    NOTE: This is a simplified placeholder.
    """
    # Compute distance between source and electrode
    d = np.linalg.norm(electrode_loc - source_loc)

    # Use the brain conductivity and brain radius as a base for this placeholder.
    sigma1 = sigmas[0]
    R1 = radii[0]
    R4 = radii[3]

    # Placeholder series sum (for demonstration)
    series_sum = 0.0
    for n in range(1, N_max + 1):
        coef = (2 * n + 1) / (n * (n + 1)) * (R1 / R4)**(n + 1)
        series_sum += coef

    # Simplified potential computation (ignoring the exact spatial dependence)
    V = I / (4 * np.pi * sigma1 * R1) * series_sum
    return V

def compute_eeg_from_tcs(i_stim, t_stim, source_loc, electrode_loc, radii, sigmas):
    """
    Compute the EEG potential at an electrode from a time-varying stimulation current
    using the four-sphere head model.

    Parameters:
      i_stim       : 1D numpy array of stimulation current amplitudes in mA.
      t_stim       : 1D numpy array of time stamps corresponding to i_stim.
      source_loc   : 3-element list/array [x, y, z] in µm for the stimulation source.
      electrode_loc: 3-element list/array [x, y, z] in µm for the EEG electrode.
      radii        : List of four radii [R1, R2, R3, R4] in µm.
      sigmas       : List of four conductivities in S/m.

    Returns:
      t_stim, eeg: The time stamps and the computed EEG potential (in Volts) at the electrode.
    """
    # Convert locations from µm to m and current from mA to A
    source_loc_m = np.array(source_loc) * 1e-6
    electrode_loc_m = np.array(electrode_loc) * 1e-6
    radii_m = np.array(radii) * 1e-6
    i_stim_A = i_stim * 1e-3

    eeg = np.zeros_like(i_stim_A)
    for idx, I in enumerate(i_stim_A):
        eeg[idx] = four_sphere_potential(I, source_loc_m, electrode_loc_m, radii_m, sigmas)
    return t_stim, eeg

if __name__ == "__main__":
    # Define source and electrode locations in µm
    source_loc = [0, 0, 78200]    # Stimulation electrode at 78,200 µm
    electrode_loc = [0, 0, 90000]   # EEG electrode at 90,000 µm

    # Four-sphere head model parameters (in µm and S/m)
    radii = [79000., 80000., 85000., 90000.]  # Brain, CSF, Skull, Scalp
    sigmas = [0.3, 1.5, 0.015, 0.3]            # Conductivities in S/m

    # Create a stimulation signal: 2 mA amplitude, 20 Hz sine wave over 1 second (1000 Hz sampling)
    t_stim = np.linspace(0, 1, 1000)
    freq = 20  # Hz
    i_stim = 2.0 * np.sin(2 * np.pi * freq * t_stim)  # in mA

    # Compute EEG potential using the four-sphere model
    t, eeg = compute_eeg_from_tcs(i_stim, t_stim, source_loc, electrode_loc, radii, sigmas)

    # Plotting the results
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Plot the EEG signal (potential)
    axs[0].plot(eeg, color='blue')
    axs[0].set_title("EEG Signal at Electrode")
    axs[0].set_ylabel("Potential (V)")

    # 2. Plot the stimulation signal (current)
    axs[1].plot(i_stim, color='red')
    axs[1].set_title("Stimulation Signal")
    axs[1].set_ylabel("Current (mA)")

    # 3. Compute and plot the PSD of the EEG signal using Welch's method
    fs = 1000  # Sampling frequency (Hz)
    f, Pxx = welch(eeg, fs=fs, nperseg=256)
    axs[2].semilogy(f, Pxx, color='green')
    axs[2].set_title("Power Spectral Density of EEG")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("PSD (V^2/Hz)")

    plt.tight_layout()
    plt.show()
