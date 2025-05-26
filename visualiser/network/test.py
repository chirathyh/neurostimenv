# import h5py
import matplotlib.pyplot as plt
import pandas as pd

# def print_h5_structure(h5obj, indent=0):
#     for key in h5obj.keys():
#         item = h5obj[key]
#         print('  ' * indent + f"{key} — {type(item)}; shape: {getattr(item, 'shape', '')}")
#         if isinstance(item, h5py.Group):
#             print_h5_structure(item, indent+1)
#
# with h5py.File('cell_positions_and_rotations.h5', 'r') as f:
#     print("Root groups/datasets:")
#     print_h5_structure(f)
pop_colors = {'HL23PYR':'k', 'HL23SST':'crimson', 'HL23PV':'green', 'HL23VIP':'darkorange'}
popnames = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']

filename = 'cell_positions_and_rotations.h5'


print(pd.read_hdf(filename,popnames[0]))
exit()

popDataArray = {}
popDataArray[popnames[0]] = pd.read_hdf(filename,popnames[0])
popDataArray[popnames[0]] = popDataArray[popnames[0]].sort_values('gid')
popDataArray[popnames[1]] = pd.read_hdf(filename,popnames[1])
popDataArray[popnames[1]] = popDataArray[popnames[1]].sort_values('gid')
popDataArray[popnames[2]] = pd.read_hdf(filename,popnames[2])
popDataArray[popnames[2]] = popDataArray[popnames[2]].sort_values('gid')
popDataArray[popnames[3]] = pd.read_hdf(filename,popnames[3])
popDataArray[popnames[3]] = popDataArray[popnames[3]].sort_values('gid')

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=5)
for pop in popnames:
	for i in range(0,len(popDataArray[pop]['gid'])):
		ax.scatter(popDataArray[pop]['x'][i],popDataArray[pop]['y'][i],popDataArray[pop]['z'][i], c=pop_colors[pop], s=5)
		ax.set_xlim([-300, 300])
		ax.set_ylim([-300, 300])
		ax.set_zlim([-1200, -400])
plt.show()
