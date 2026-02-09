"""
Load a saved Evoked FIF, plot multiple topomaps with channel labels, and plot time-series figure with highlighted channels.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

SUBJECT = 'HEALTHY_0'
fname = "simulation/"+SUBJECT+".fif"        # path to your saved evoked file
#fname = "simulation/multi-dipole-pred_evoked-ave.fif"

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

out_dir2 = Path("EEG")
out_dir2.mkdir(exist_ok=True)
# -----------------------------

evoked = mne.read_evokeds(fname, verbose='error')[0]
print(f"Loaded evoked with {len(evoked.ch_names)} channels and {len(evoked.times)} time points.")
picks_eeg = mne.pick_types(evoked.info, eeg=True, meg=False)
evoked_eeg = evoked.copy().pick(picks=picks_eeg)

# Save the files to send Robin.
# data shape is (n_channels, n_times)
data = evoked_eeg.data  # numpy array
ch_names = evoked_eeg.ch_names
data_t = data.T  # shape -> (n_times, n_channels)
out_fname = out_dir2 / f"{Path(fname).stem}_eeg.csv"  # build output filename from original file stem
# Save CSV: first row = channel names, rows = time steps
# use comments='' so numpy doesn't prefix the header with '#'
np.savetxt(out_fname, data_t, delimiter=",", header=",".join(ch_names), comments='', fmt='%.8e')
print(f"Saved EEG CSV to: {out_fname} (shape: {data_t.shape}, header: {len(ch_names)} channels)")
#exit()

selected_channels = ['Fp1','Fpz','Fp2',      # frontal pole
                     'AF3','F7','F5','F3','F1','Fz','F4',   # frontal
                     'FC3','FC1',           # fronto-central
                     'C3','Cz','C4',        # central
                     'T9','T10',            # temporal (lateral)
                     'P3','Pz','P4','P6',   # parietal
                     'O1','Oz','O2','Iz']   # occipital / inferior


evoked_selected = evoked.copy().pick_channels(selected_channels)

# data_v = evoked_selected.get_data()            # shape (n_ch, n_times), in Volts
# print("V range: ", np.nanmin(data_v), np.nanmax(data_v))
# print("uV range:", np.nanmin(data_v) * 1e6, np.nanmax(data_v) * 1e6)
# exit()


all_times = np.arange(0, 25, 4)  # 24seconds.
fig = evoked_selected.plot_topomap(all_times, ch_type="eeg", time_unit='s', show_names=True, contours=10, colorbar=True,
                                   sphere=(-0.0014, 0.01, -0.01, 0.099), ncols=6, nrows="auto")

import matplotlib.ticker as ticker
import matplotlib.colorbar as mcb

mpl_fig = plt.gcf()
fmt = '%.3f'   # choose desired decimals
found = False
for i, ax in enumerate(mpl_fig.axes):
    # get the visible y-tick labels as strings
    ytick_texts = [t.get_text() for t in ax.get_yticklabels()]
    # get the tick *values*
    ytick_vals = ax.get_yticks()

    # Case A: tick labels are non-empty and all parse to floats -> likely a colorbar
    if len(ytick_texts) > 1 and all(txt.strip() != '' for txt in ytick_texts):
        print("Case A")
        parsable = True
        for s in ytick_texts:
            try:
                float(s.replace('−', '-'))  # handle weird minus sign
            except Exception:
                parsable = False
                break
        if parsable:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            found = True
            # we keep going to update all matching axes (in case there are multiple colorbars)

    # Case B: labels are empty but tick *values* look numeric and axis is narrow (likely a colorbar)
    elif len(ytick_vals) > 1:
        print("Case B")
        # heuristic: colorbar axes are usually tall & narrow; check aspect in figure coords
        bbox = ax.get_position()
        width, height = bbox.width, bbox.height
        if width < 0.12 and height > 0.2:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            found = True

# force redraw to update tick labels
plt.draw()

if not found:
    # helpful debug info for further troubleshooting
    print("Could not confidently detect a colorbar axis. Axes info (index, bbox, yticks, yticklabels):")
    for i, ax in enumerate(mpl_fig.axes):
        print(i, "bbox:", np.round(ax.get_position().bounds, 4),
              "yticks:", np.round(ax.get_yticks(), 6),
              "yticklabels:", [t.get_text() for t in ax.get_yticklabels()])

plt.show()
# save and show as before
fig.savefig('figures/topomap-full.jpg', facecolor='white', edgecolor='none',
            bbox_inches="tight", dpi=300)


# fig.savefig('figures/topomap-full.jpg', facecolor='white', edgecolor='none', bbox_inches="tight", dpi=300)
# plt.show()

# exit()

fig, anim = evoked_selected.animate_topomap(times=all_times, ch_type="eeg", blit=False, time_unit='s', frame_rate=10)
plt.show()
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=2, metadata=dict(artist='you'), bitrate=2000)
anim.save("figures/evoked_topomap.mp4", writer=writer, dpi=150)  # dpi controls output resolution
plt.close(fig)  # close the figure to free memory

from plot_eeg import plot_eeg_stack
electrode_names = evoked_selected.ch_names #[:20]
EEG = evoked_selected.get_data() #[:20, :]
plot_eeg_stack(EEG, electrode_names, sfreq=100.0, unit='V', figsize=(12, 8), savepath='figures/eeg.jpg')

# montage = mne.channels.make_standard_montage('standard_1020')
# montage.plot(sphere=(-0.0014, 0.01, -0.01, 0.099))
# plt.show()
# exit()
# f3_loc = evoked_eeg.info['chs']['F3']['loc'][:3]
# print(evoked_eeg.info.dig)
# exit()


# fig = evoked_selected.plot_topomap(times_s, ch_type="eeg", time_unit='s', show_names=True, sphere=(-0.0014, 0.01, -0.01, 0.099))  #, sphere='auto'
# fig.savefig('figures/topomap.jpg', facecolor='white', edgecolor='none', bbox_inches="tight", dpi=300)
# plt.show()


# fig = evoked_selected.plot_topomap(times_s, ch_type="eeg", time_unit='s', show_names=True, sphere=(-0.0014, 0.01, -0.01, 0.099))  #, sphere='auto'
# fig.savefig('figures/topomap.jpg', facecolor='white', edgecolor='none', bbox_inches="tight", dpi=300)
# plt.show()








#
#
#
# # 3) Convert requested times (ms -> seconds) and validate
# times_s = [t / 1000.0 for t in times_ms]
# # keep only times within available range
# times_s = [t for t in times_s if (t >= evoked_eeg.times.min() - 1e-12 and t <= evoked_eeg.times.max() + 1e-12)]
# if len(times_s) == 0:
#     raise ValueError("No requested times are within the range of the evoked times.")
# print("Plotting topomaps at (s):", times_s)
#
# # 4) Compute a common color scale (symmetric) based on the absolute 98th percentile
# abs_vals = np.abs(evoked_eeg.data)
# vmax = np.percentile(abs_vals, 98)
# vmin, vmax = -vmax, vmax
# print(f"Using vmin={vmin:.3e}, vmax={vmax:.3e} for topomaps")
#
# # 5) Create topomap figure with channel labels
# # n_maps = len(times_s)
# # fig_topo, axes = plt.subplots(1, n_maps+1, figsize=(3 * n_maps, 3))
# # axes = np.atleast_1d(axes)
#
# # Use evoked.plot_topomap with provided axes and without auto-show, then annotate
# evoked_eeg.plot_topomap(
#     times=times_s,
#     ch_type='eeg',
#     time_unit='s',
#     show=False,
#     extrapolate='box',
#     show_names=True,   # <-- ask MNE to annotate channel names safely
# )
# plt.show()


