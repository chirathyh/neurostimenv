import numpy as np
import matplotlib.pyplot as plt

# Define insulin bolus events (units) at meal times
meal_times = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])      # hours of the day
bolus_units = np.array([1, 0.5, 0.5, 0.6, 0.3, 0.8, 1, 0.5, 0.6, 0.3, 0.1, 0.1, 1, 0.5, 0.5, 0.6, 0.3, 0.8, 1, 0.5, 0.6, 0.3, 0.1, 0.1])       # typical units for breakfast, lunch, dinner

# Optionally add a small basal rate as a constant thin bar
basal_rate = 0.5  # units per hour
basal_times = np.arange(0, 24, 1)       # hourly basal dosing
basal_units = np.full_like(basal_times, basal_rate)

# Plot
plt.figure(figsize=(10, 4))

# Basal insulin (thin gray bars)
plt.bar(basal_times, basal_units, width=1.0, color='red', edgecolor='none', label='Basal (units/h)')

# Bolus insulin (thicker blue bars)
plt.bar(meal_times, bolus_units, width=0.8, color='red', edgecolor='k', label='Bolus (units)')

plt.xlabel('Time of Day (hours)')
plt.ylabel('Insulin (units)')
plt.title('Simulated 24-Hour Insulin Delivery')
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 2))
# plt.legend()
plt.tight_layout()
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
#
# # Simulate a 24-hour glucose trajectory sampled every 5 minutes
# sampling_interval = 5  # minutes
# times = np.arange(0, 24*60 + sampling_interval, sampling_interval) / 60  # in hours
#
# # Baseline glucose around 90 mg/dL with small random fluctuations
# baseline = 90 + 5 * np.random.randn(len(times))
#
# # Add realistic meal-related spikes (breakfast at 8h, lunch at 13h, dinner at 19h)
# def add_meal_spike(data, times, meal_time, peak=50, width=0.5):
#     """
#     Adds a Gaussian-shaped spike around meal_time.
#     peak: height of the spike above baseline
#     width: standard deviation in hours
#     """
#     spike = peak * np.exp(-0.5 * ((times - meal_time) / width) ** 2)
#     return data + spike
#
# glucose = baseline.copy()
# for meal_time in [8, 13, 19]:
#     glucose = add_meal_spike(glucose, times, meal_time, peak=60, width=0.7)
#
# # Plot
# plt.figure(figsize=(10, 4))
# plt.plot(times, glucose, color='k')
# plt.xlabel('Time (hours)')
# plt.ylabel('Glucose (mg/dL)')
# plt.title('Simulated 24-Hour Glucose Trajectory')
# plt.xlim(0, 24)
# plt.xticks(np.arange(0, 25, 2))
# plt.tight_layout()
# plt.show()
