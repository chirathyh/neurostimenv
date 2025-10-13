import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from experiments.case_study.replication import get_rewards
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

reward_values_mdd = get_rewards(file_path=MDD_EEG)
reward_values_healthy = get_rewards(file_path=HEALTHY_EEG)

fig = plt.figure(figsize=(10, 5))
plt.hist(reward_values_mdd, color='red', bins=10, label='Depression', orientation='horizontal')
plt.hist(reward_values_healthy, color='black', bins=10, label='Healthy', orientation='horizontal')
plt.axhline(y=-0.09264591737143694, color='green', label='Cutoff value', linestyle='--', alpha=0.7)
plt.legend()
plt.xlabel('Frequency')
plt.ylabel('Reward/Score')
plt.tight_layout()
fig.savefig('png/Fig3b.png', dpi=600, bbox_inches='tight', pad_inches=0.02, facecolor='auto', transparent=False)
plt.show()

