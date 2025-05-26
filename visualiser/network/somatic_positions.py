import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

popnames = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
filename = 'cell_positions_and_rotations.h5'

# dfs = {}
# for pop in popnames:
#     dfs[pop] = pd.read_hdf(filename, pop)
#     print(f"Loaded {pop}: {dfs[pop].shape[0]} cells")

fig = plt.figure(figsize=(9,9))
ax  = fig.add_subplot(111, projection='3d')

colors = {'HL23PYR':'tab:blue','HL23SST':'tab:orange',
          'HL23PV':'tab:green','HL23VIP':'tab:red'}
markers= {'HL23PYR':'o','HL23SST':'^','HL23PV':'s','HL23VIP':'d'}

for pop in popnames:
    data = pd.read_hdf(filename, pop)
    ax.scatter(data['x'], data['y'], data['z'],
               c=colors[pop], marker=markers[pop],
               label=pop, s=12, alpha=0.7)

ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)'); ax.set_zlabel('Z (µm)')
ax.legend()
plt.tight_layout()
plt.show()
