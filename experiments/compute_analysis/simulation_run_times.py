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

'''
Fig. Results for benchmark experiments performed for a 1,000ms simulation at 0.025ms resolution for a neural circuit with 1,000 neurons. 
The CPU time (left) and wall time (right) for experiments for different number of processes are presented.  
'''


def convert_to_minutes(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))  # Split the string by colon to get hours, minutes, and seconds
    total_minutes = hours * 60 + minutes + seconds / 60  # Convert hours and seconds to minutes and sum them up
    return total_minutes


no_stim = ["12:36:03", "11:12:26", "9:25:00","7:51:27","7:11:10","6:47:11","7:24:11", "10:15:09"]
stim = ["40:19:51","40:25:13", "39:36:43", "31:12:41","31:01:02","27:56:11","26:26:30","36:07:00"]
CPUtimes1 = [convert_to_minutes(x) for x in no_stim]  # no stim
CPUtimes2 = [convert_to_minutes(x) for x in stim] # stim

Np = [10, 16, 32, 72, 128, 256, 512, 756]  # Number of processes
walltimes1 = [63.95, 36.66, 15.99, 5.97, 3, 1.309,0.616, 0.561]  # Wall times for system 1 - No Stim
walltimes2 = [229.6, 145.8, 72.4, 25.42, 14.17, 6.251, 2.842, 2.636]  # Wall times for system 2 - with stim

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Define font sizes and bold formatting
title_font = {'size': 18} #, 'weight': 'bold'
label_font = {'size': 16} #, 'weight': 'bold'
legend_font = {'size': 14}
tick_font = 14

# Plot the CPU times on the first subplot
ax1.plot(Np, CPUtimes1, marker='^', label='No-Stimulation', color='g', markersize=8)
ax1.plot(Np, CPUtimes2, marker='d', label='Stimulation', color='m', markersize=8)
# ax1.set_title('CPU Times vs Number of Processes', **title_font)
ax1.set_xlabel('Number of Processes', **label_font)
ax1.set_ylabel('CPU Time (minutes)', **label_font)
ax1.legend(fontsize=legend_font['size'])
ax1.grid(True)
# ax1.tick_params(axis='both', which='major', labelsize=tick_font)

# Set x-axis to logarithmic scale
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax1.set_yscale('log')
# ax2.set_yscale('log')

# Plot the Wall times on the second subplot
ax2.plot(Np, walltimes1, marker='^', label='No-Stimulation', color='g', markersize=8)
ax2.plot(Np, walltimes2, marker='d', label='Stimulation', color='m', markersize=8)
# ax2.set_title('Wall Times vs Number of Processes', **title_font)
ax2.set_xlabel('Number of Processes', **label_font)
ax2.set_ylabel('Wall Time (minutes)', **label_font)
ax2.legend(fontsize=legend_font['size'])
ax2.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
fig.savefig('fig/Fig1.png', dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='auto', transparent=False)
plt.show()
