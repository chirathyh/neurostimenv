import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t
import seaborn as sns

dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)

transient = 4000  # in seconds L23Net uses : 2000

t1 = int(transient/dt)
print("Sampling Rate:", fs)


count = np.load("../../data/bandit/hbandit1/checkpoints/counts.npy")
values = np.load("../../data/bandit/hbandit1/checkpoints/values.npy")

print(count)
print(values)
print(np.argmax(values))
print(np.sum(count))

def read_rewards_from_csv(folder_path):
    reward_values = []
    arms = []

    # Find all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    print(sorted(csv_files))

    for file in sorted(csv_files):  # Sorting ensures sequential order
        df = pd.read_csv(file)
        reward_values.extend(df['Reward'].values)  # Append rewards to list
        arms.extend(df['Arm'].values)

    return reward_values, arms


def plot_rewards(reward_values):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_values, marker='o', linestyle='-', label='Reward')
    plt.xlabel('Trial Index')
    plt.ylabel('Reward')
    plt.title('Reward Values Over Trials')
    plt.legend()
    plt.grid()
    plt.show()

def plot_rewards_whisker(rewards, arms):
    plt.figure(figsize=(10, 5))
    df = pd.DataFrame({'Reward': rewards, 'Arm': arms})
    sns.boxplot(x='Arm', y='Reward', data=df)
    plt.xlabel('Arm')
    plt.ylabel('Reward')
    plt.title('Whisker Plot of Rewards for Each Arm')
    plt.grid()
    plt.show()

# Example usage
folder_path = "../../data/bandit/hopefulbandit/testing"  # Change this to the folder containing your CSV files
rewards, arms = read_rewards_from_csv(folder_path)
print(rewards)

# Plot histogram
plt.hist(rewards, bins=10, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of List Data')

# Show plot
plt.show()



exit()
plot_rewards_whisker(rewards, arms)
plot_rewards(rewards)

exit()

folder_path = "../../data/bandit/nbandit2/training"  # Change this to the folder containing your CSV files
rewards2, arms2 = read_rewards_from_csv(folder_path)

plot_rewards(rewards)
plot_rewards_whisker(rewards+rewards2, arms+arms2)
plot_rewards(rewards+rewards2)


# folder_path = "../../data/bandit/bandit2/training"  # Change this to the folder containing your CSV files
# rewards2, arms2 = read_rewards_from_csv(folder_path)
#
# folder_path = "../../data/bandit/bandit2/training"  # Change this to the folder containing your CSV files
# rewards3, _ = read_rewards_from_csv(folder_path)
#
# # plot_rewards(rewards)
# plot_rewards_whisker(rewards+rewards2, arms+arms2)
# plot_rewards(rewards+rewards2)
