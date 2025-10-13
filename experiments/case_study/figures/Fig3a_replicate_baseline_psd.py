import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from experiments.case_study.replication import process_eeg
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
})


HEALTHY_EEG = "../data/healthy/EEG_HEALTHY_"
MDD_EEG = "../data/mdd/EEG_MDD_"

# plot figure.
# Load / Process both depression and healthy EEG datasets
all_freqs, avg_psd, ci_lower, ci_upper = process_eeg(file_path=MDD_EEG)
all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h = process_eeg(file_path=HEALTHY_EEG)

# Plot the average PSD with bootstrapped 95% Confidence Interval
fig = plt.figure(figsize=(10, 5))

# Depression group
plt.plot(all_freqs, avg_psd, color='r', label="Depression Baseline")
plt.fill_between(all_freqs, ci_lower, ci_upper, color='r', alpha=0.3)

# Healthy group
plt.plot(all_freqs_h, avg_psd_h, color='k', label="Healthy Baseline")
plt.fill_between(all_freqs_h, ci_lower_h, ci_upper_h, color='k', alpha=0.3)

# Add vertical lines at 8 Hz and 12 Hz
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=12, color='gray', linestyle='--', alpha=0.7)

# Add text annotations for frequency bands (Greek notation)
plt.text(6.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\theta$",
         fontsize=14, color='black', ha='center', fontweight='bold')
plt.text(10, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\alpha$",
         fontsize=14, color='black', ha='center', fontweight='bold')
plt.text(20, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\beta$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.xlim(4, 50)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
# plt.title("Average Power Spectral Density (PSD) with Bootstrapped 95% CI")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig.savefig('png/Fig3a.png', dpi=600, bbox_inches='tight', pad_inches=0.02, facecolor='auto', transparent=False)
plt.show()
