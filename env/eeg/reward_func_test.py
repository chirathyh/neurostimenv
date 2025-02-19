import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

# Constants
DEPRESSION = 4.781662697628607e-19
HEALTHY = 3.3783662121573373e-19
TARGET_BAND = (4, 16)

# Reward function
def reward_func(calc_power):
    # Normalize power between [0, 1] relative to HEALTHY and DEPRESSION
    norm_power = (calc_power - DEPRESSION) / (HEALTHY - DEPRESSION)
    norm_power = np.clip(norm_power, 0, 1)  # Ensure within bounds

    # Compute reward (higher when closer to HEALTHY)
    reward = 1 - abs(norm_power - 1)
    return reward

# Generate range of calc_power values
calc_power_values = np.linspace(1e-19, 6e-19, 100)
rewards = [reward_func(cp) for cp in calc_power_values]

# Plot the reward function
plt.figure(figsize=(8, 5))
plt.plot(calc_power_values, rewards, label='Reward Function', color='b')
plt.axvline(DEPRESSION, color='r', linestyle='--', label='Depression Power')
plt.axvline(HEALTHY, color='g', linestyle='--', label='Healthy Power')
plt.xlabel("Calc Power")
plt.ylabel("Reward")
plt.title("Reward Function for EEG Band Power")
plt.legend()
plt.grid(True)
plt.show()
