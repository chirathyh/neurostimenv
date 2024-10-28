import pandas as pd
import os
import matplotlib.pyplot as plt

pop_colors = {'HL23PYR':'k', 'HL23SST':'crimson', 'HL23PV':'green', 'HL23VIP':'darkorange'}
popnames = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
# Plot soma positions
def plot_network_somas(OUTPUTPATH):
	filename = os.path.join(OUTPUTPATH,'cell_positions_and_rotations.h5')
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

	#plt.show()
	return fig
