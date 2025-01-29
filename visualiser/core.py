import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import scipy.signal as ss
import numpy as np

EEG_COLOUR = '#000000'
TS_COLOUR = '#800000'
FREQ_MIN = 0
FREQ_MAX = 50


def plot_episode(cfg, eeg, ts, t, ts_params):
    # TODO: improve the handling of data. EEG output is 16001, when t is 16000
    # preprocessing
    eeg = eeg[0]  # remove multiple channels
    eeg = eeg[:-1]  # remove last element
    ts[:] = [x / 1e6 for x in ts]  # rescale to mA

    target_peak_freq = 8

    # calc psd
    eeg_mid = len(eeg) // 2
    eeg_pre = eeg[:eeg_mid]
    eeg_post = eeg[eeg_mid:]
    fs = (1 / cfg.env.network.dt) * 1000
    freq_pre, psd_pre = ss.welch(eeg_pre, fs=fs, nperseg=int(fs/4))
    freq_post, psd_post = ss.welch(eeg_post, fs=fs, nperseg=int(fs/4))
    psd = psd_pre.flatten()
    peak_index = np.argmax(psd)
    peak_freq_pre = freq_pre[peak_index]
    psd = psd_post.flatten()
    peak_index = np.argmax(psd)
    peak_freq_post = freq_post[peak_index]

    # figure specs
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.4])
    ax2 = fig.add_subplot(gs[0, :])
    ax3 = fig.add_subplot(gs[1, 0])  # Third plot: left part of the third row
    ax4 = fig.add_subplot(gs[1, 1])
    divider = make_axes_locatable(ax2)
    ax = divider.append_axes("top", size=3.0, pad=0.02, sharex=ax2)

    ax.plot(t, eeg, color=EEG_COLOUR)
    ax2.plot(t, ts, color=TS_COLOUR)
    ax3.plot(freq_pre, psd_pre)
    ax4.plot(freq_post, psd_post)
    ax3.axvline(x=8, color='g', linestyle='--', label=f'Target Freq: {target_peak_freq:.2f} Hz')
    ax4.axvline(x=8, color='g', linestyle='--', label=f'Target Freq: {target_peak_freq:.2f} Hz')
    ax3.axvline(x=peak_freq_pre, color='r', linestyle='--', label=f'Freq: {peak_freq_pre:.2f} Hz')
    ax4.axvline(x=peak_freq_post, color='r', linestyle='--', label=f'Freq: {peak_freq_post:.2f} Hz')
    eeg_line = mlines.Line2D([], [], color=EEG_COLOUR, label='EEG')
    ts_a_line = mlines.Line2D([], [], color=TS_COLOUR, label=f'Amplitude: {ts_params[0]:.4f} mA')
    ts_f_line = mlines.Line2D([], [], color=TS_COLOUR, label=f'Freq: {ts_params[1]:.4f} Hz')
    ax.legend(handles=[eeg_line, ts_a_line, ts_f_line], loc='upper right')
    ax3.legend()
    ax4.legend()

    ax.set_ylabel('EEG (V)', color=EEG_COLOUR)
    ax2.set_ylabel('Stimulation (mA)', color=TS_COLOUR)
    ax2.set_xlabel('Time (ms)')
    ax3.set_ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
    ax4.set_ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
    ax3.set_xlabel('Frequency (Hz)')
    ax4.set_xlabel('Frequency (Hz)')

    # axis limits
    ax3.set_xlim(FREQ_MIN, FREQ_MAX)
    ax4.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_title('Simulation: Transcranial Stimulation')
    ax3.set_title('No Stimulation')
    ax4.set_title('With Stimulation')
    ax.grid()
    plt.tight_layout()
    plt.show()


# def plot_n_episodes():
    # for testing only:
    # max_length = max(len(arr[0]) for arr in sim_data)
    # fig, axes = plt.subplots(len(sim_data), 1, figsize=(16, len(sim_data) * 2))
    # # Plot each array in a subplot
    # for i, arr in enumerate(sim_data):
    #     ax = axes[i] if len(sim_data) > 1 else axes  # Handle single subplot case
    #     ax.plot(range(1, len(arr[0]) + 1), arr[0], label=f"Array {i+1}")
    #     ax.set_xlim(1, max_length)  # Set x-axis limit to the maximum length
    #     ax.set_title(f"Plot of Array {i+1}")
    #     ax.set_xlabel("Index")
    #     ax.set_ylabel("Value")
    #     ax.legend()
    #     ax.grid(True)
    # plt.tight_layout()
    # plt.show()
