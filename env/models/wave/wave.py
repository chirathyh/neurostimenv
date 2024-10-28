import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def damped_wave_equation_model(eeg_data, c, gamma, dx, dt):
    """
    Simulate the isotropic damped wave equation given initial conditions and parameters.
    """
    # Number of channels (spatial points) and time points
    num_channels, num_timepoints = eeg_data.shape

    # Initialize the model output (simulated EEG data)
    simulated_eeg = np.zeros((num_channels, num_timepoints))

    # Set initial conditions (e.g., use the first two time points of EEG data)
    simulated_eeg[:, 0] = eeg_data[:, 0]
    simulated_eeg[:, 1] = eeg_data[:, 1]

    # Time integration
    for t in range(1, num_timepoints-1):
        for x in range(1, num_channels-1):
            simulated_eeg[x, t+1] = (2 * simulated_eeg[x, t] - simulated_eeg[x, t-1] +
                                     (c**2 * dt**2 / dx**2) * (simulated_eeg[x+1, t] -
                                                               2 * simulated_eeg[x, t] +
                                                               simulated_eeg[x-1, t]) -
                                     gamma * dt * simulated_eeg[x, t]) / (1 + gamma * dt)

    return simulated_eeg


def plot_eeg(real_eeg, simulated_eeg, channels, start_time=0, end_time=1000):
    """
    Plot real and simulated EEG data.
    """
    time = np.arange(start_time, end_time)
    plt.figure(figsize=(15, 10))

    for i, channel in enumerate(channels):
        plt.subplot(len(channels), 1, i + 1)
        plt.plot(time, real_eeg[channel, start_time:end_time], label='Real EEG')
        plt.plot(time, simulated_eeg[channel, start_time:end_time], label='Simulated EEG', linestyle='--')
        plt.title(f'Channel {channel + 1}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.tight_layout()
    plt.show()


def objective(params, eeg_data, dx, dt):
    c, gamma = params
    simulated_eeg = damped_wave_equation_model(eeg_data.T, c, gamma, dx, dt)
    return np.mean((eeg_data.T - simulated_eeg)**2)


# Load EEG data from CSV file
eeg_data = pd.read_csv('simulated_eeg.csv')

eeg_data = eeg_data.values
eeg_data = eeg_data[:, 10000:-1]
# Normalize the data if needed
#eeg_data = (eeg_data - np.mean(eeg_data, axis=0)) / np.std(eeg_data, axis=0)
print(eeg_data.shape)


# Define the parameters
c = 1.0      # Wave speed
gamma = 0.1  # Damping coefficient
dx = 0.07     # Spatial step (channel spacing)
dt = 0.025   # Time step (sampling interval)

# Initial guess for the parameters
initial_guess = [1.0, 0.1]

# Optimize
result = minimize(objective, initial_guess, args=(eeg_data, dx, dt))
optimized_c, optimized_gamma = result.x

print(f'Optimized wave speed: {optimized_c}')
print(f'Optimized damping coefficient: {optimized_gamma}')

# plotting
c = 1.0      # Wave speed
gamma = 5.4 # Damping coefficient
simulated_eeg = damped_wave_equation_model(eeg_data.T, c, gamma, dx, dt)
mse = np.mean((eeg_data.T - simulated_eeg)**2)
print(f'Mean Squared Error: {mse}')

simulated_eeg = simulated_eeg.T
# Select channels to visualize (e.g., first 3 channels)
channels_to_plot = [0, 1, 2, 3, 4]
# Plot real vs. simulated EEG data
plot_eeg(eeg_data, simulated_eeg, channels_to_plot, start_time=1, end_time=40000)
