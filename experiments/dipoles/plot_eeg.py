import numpy as np
import matplotlib.pyplot as plt

def plot_eeg_stack(eeg, channel_names, sfreq=1000.0, unit='uV', figsize=(12, 8), savepath=None):
    """
    Plot stacked EEG channels (amplitude vs time) with channel labels.

    Parameters
    ----------
    eeg : ndarray, shape (n_channels, n_timepoints)
        EEG data (assumed in volts). Will be converted to microvolts for plotting.
    channel_names : list[str]
        Names of channels in the same order as eeg rows.
    sfreq : float
        Sampling frequency in Hz (default 1000.0). Used to create time axis.
    unit : str
        Unit string to show on y-axis ('uV' default).
    figsize : tuple
        Matplotlib figure size.
    savepath : str or None
        If given, save figure to this path (e.g., 'eeg_plot.png').
    """
    # basic checks
    eeg = np.asarray(eeg)
    if eeg.ndim != 2:
        raise ValueError("eeg must be a 2D array (n_channels, n_timepoints)")
    n_ch, n_t = eeg.shape
    if len(channel_names) != n_ch:
        raise ValueError(f"Number of channel names ({len(channel_names)}) "
                         f"does not match number of rows in eeg ({n_ch})")

    # Time axis (seconds)
    t = np.arange(n_t) / float(sfreq)

    # Convert to microvolts for readability
    eeg_uV = eeg * 1e6  # V -> µV

    # Choose vertical spacing robust to outliers
    amp99 = np.percentile(np.abs(eeg_uV), 99)
    spacing = amp99 * 2.5 if amp99 > 0 else 1.0

    offsets = np.arange(n_ch) * spacing
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_ch):
        ax.plot(t, eeg_uV[i] + offsets[i], linewidth=0.8)

    # label channels at left
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"EEG — stacked channels ({unit})")
    # show amplitude ticks? hide x minor clutter
    ax.set_xlim(t[0], t[-1])
    # invert y so first channel appears at top (common EEG convention)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


# ---------------------------
# Example usage:
# - Put your electrode_names list here (must match eeg rows order)
electrode_names = ['Fp1','Fpz','Fp2','F7','F3','Fz','F4','F8',
                   'T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','Oz','O2']

# - Assume `eeg` is your numpy array of shape (21, n_timepoints)
#   For example, if you've computed pot_db_4s_top above and have 'eeg' variable already:
# eeg = np.array(pot_db_4s_top)   # shape should be (21, n_timepoints)
#
# If your array currently has shape (n_timepoints, n_channels), transpose it:
# eeg = eeg.T

# Call the plotting function (adjust sfreq to your actual sampling rate)
# plot_eeg_stack(eeg, electrode_names, sfreq=1000.0, savepath='eeg_stacked.png')
