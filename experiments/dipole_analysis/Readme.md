## Analysing simulated dipoles.

<code>run_simulation_dipoles.py</code> will run a simulation (healthy/MDD) and then save the dipole object.

The dipole can be placed on the "sample" subject in the MNE data and simulated to generate the EEG signals. We can place a single dipole (<code>simulate_dipole.py</code>) or place multiple dipoles (<code>simulate_multiple_dipoles.py</code>) and run the analysis.

You can select the EEG montage to be used. Currently using the 10-20 system.
Note: exclude = ['T7', 'T8', 'P7', 'P8'] # 'T3', 'T5', 'T4', 'T6'  https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)

The resulting EEG signals can be visualised (<code>visualise.py</code>) using simple timeseries or topographical maps (head models) and their simulations.


