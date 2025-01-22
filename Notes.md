# Implementation Notes:

### Todo:
- Refactor 

### Running SimNIBS using python interpreter:
<code>simnibs_python run_simnibs_simulation.py </code>

how to read info from SimNIBS simulation:
- load the mesh of the human head model
- crop the mesh to get the cortex (grey matter region)
- simnibs.ElementTags.GM: A constant representing the tag for gray matter. This line extracts the mesh elements that belong to the gray matter, isolating that part of the brain for analysis.
- define region of interest: simnibs.mni2subject_coords: Converts MNI coordinates (standard brain space coordinates) to subject-specific coordinates.
- NEURON is based on Cartesian coordinates while SImNIBS uses MNI coordinates
- Defines the radius of the spherical ROI to be 10 mm.
- get the elements in the region
- elements_baricenters: Computes the center points (barycenters) of all the elements in the gray matter mesh.

coordinate systems:
- NEURON - cartesian coordinates (micro meters)
- SimNIBS - world coordinates in subject space coordinates (mm)
- Units: https://simnibs.github.io/simnibs/build/html/faq.html#can-simnibs-simulate-tacs

**L23Net** was developed using <code>NEURON==7.7</code> and <code>LFPy==2.0</code>. There has been some changes in the LFPy library.
e.g., <br>
- soma_as_point was used to calculate the extracellular potential in LFPy2.0 in LFPy==2.3 this is replaced as root_as_point.
This shouldn't affect any of the implementations for tACS.
- Also, the EEG was calculated manually, where using the foursphere headmodel and the dipole moments were calculated within the simulation <code>rec_current_dipole_moment=False</code>. 
- In the latest version you can add a probe called CurrentDipoleMoment to calculate the dipole and then functions are available to calculate the EEG. <br>
  * Using probe RecExtElectrode and CurrentDipoleMoment: LFPy 2.3 Example > example_network.py
  * Calculating EEG example: LFPy 2.3 > example_EEG.py
  

EEG Calculation: We have to provide a reference dipole location, which is set to the location of the microcircuit.
Then we have to provide the x,y,z coordinates for the EEG placement along with some other parameterts.


<!---
<mpiexec> -n <processes> python example_mpi.py
--oversubscribe fore more processes and physical cores.
-->

nrniv -gdb python example_network_stim.py


Notes:
- Simulations tend to take a long time when recording LFP and DIPOLE.
- When simulation time is increased and at 0.025 dt, the required MEM increases exponentially, leading to crashes.


My Implementation on NCI (full network (1,000 neurons) - No probes):
- At 0.025 dt, 1000 ms simulation, 6.75GB MEM, runtime 30 min
- At 0.025 dt, 2000 ms simulation, 7.12GB, runtime 60 min
- At 0.025 dt, 5000 ms simulation, 8.11 GB, runtime 180 min 
- At 0.01 dt, 16,000 ms simulation, 8.22 GB, runtime 145 min 
- At 0.025 dt, 20,000 ms simulation, 15.22 GB, runtime 145 min 
- At 0.025 dt, 28,000 ms simulation, 26.59 GB, runtime 435 min ; (14:39:20 /2) - 700SU/experiment.

Running 


When I apply tACS the time increases by 5x.

Original Implementtaion:
- At 0.025 dt, 4500 ms simulation, 

Notes: 

- ## 0.025 ms (25 µs): This is a commonly used step size for detailed simulations that need to capture fine temporal dynamics, such as action potentials and synaptic events. It provides a high level of accuracy while maintaining a reasonable computational load.
- 0.1 ms (100 µs): This step size is often used for larger-scale network simulations where individual neuron dynamics are less critical, but computational efficiency is important. It still provides a good level of detail for many neuronal processes.
- Loss of Accuracy
- Failure to Capture Fast Dynamics
- Numerical Instability
- Inaccurate Synaptic Integration


## ReXExternal Electrode to add extracellular stimulation/
- Ran into an issue when using the library LFPY and MEAUtility.
- /home/chirath/Documents/depression-simulator/lib/python3.10/site-packages/MEAutility/core.py
- Had to comment two lines to get it working  comment 
- I'm computing a mapping each time now, to avoid the error, however might be inefficient.
- Original code seems to have a bug.

<code>
        recompute_mapping = False


        if self.mapping is not None:
            if self.mapping.shape != (len(points), npoints):
                # compute new mapping
                recompute_mapping = True
            # if np.all(np.array(points) == self.points):  # Todo: chirath, understand rational, i have commented to get it working
            #     recompute_mapping = True
        else:
            recompute_mapping = True

        if np.array(points).ndim == 1:
            points = np.array([points])
</code>
