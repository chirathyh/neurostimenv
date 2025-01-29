import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create a figure
fig = plt.figure(figsize=(16, 8))

# Define the gridspec layout for 3 rows and 2 columns
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # 3 rows, 2 columns

# First and second plots (sharing x-axis)
# ax1 = fig.add_subplot(gs[0, :])  # First plot: spans the entire first row
ax2 = fig.add_subplot(gs[0, :])  # Second plot: spans the entire second row

# Third and fourth plots (left and right in the third row)
ax3 = fig.add_subplot(gs[1, 0])  # Third plot: left part of the third row
ax4 = fig.add_subplot(gs[1, 1])  # Fourth plot: right part of the third row

# Divider for ax2 to add a new axes above it
divider = make_axes_locatable(ax2)
ax_top = divider.append_axes("top", size=3.0, pad=0.02, sharex=ax2)

# Example plotting (replace with your actual data)
ax2.plot([0, 1], [1, 0], label="Plot 2")
ax3.plot([0, 1], [0, 0], label="Plot 3")
ax4.plot([0, 1], [1, 1], label="Plot 4")
ax_top.plot([0, 1], [0.5, 0.5], label="Top Plot (Above ax2)")

# Set labels for each plot
ax2.set_ylabel('Plot 2')
ax3.set_ylabel('Plot 3')
ax4.set_ylabel('Plot 4')
ax_top.set_ylabel('Top Plot')

# Adding titles and legends (if needed)
ax2.set_title('Main Plot 2')
ax3.set_title('Sub Plot 3')
ax4.set_title('Sub Plot 4')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
