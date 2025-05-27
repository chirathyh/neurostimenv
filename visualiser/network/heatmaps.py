import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

popnames = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
filename = 'cell_positions_and_rotations.h5'

fig, axes = plt.subplots(2, 2, figsize=(10,8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, pop in zip(axes, popnames):
    data = pd.read_hdf(filename, pop)
    hb = ax.hist2d(data['x'], data['y'], bins=80, cmap='magma')
    ax.set_title(pop)
    ax.set_xlabel('X'); ax.set_ylabel('Y')

fig.colorbar(hb[3], ax=axes.tolist(), label='Count')
plt.tight_layout()
plt.show()
