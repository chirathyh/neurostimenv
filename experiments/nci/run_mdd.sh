#!/bin/bash
#PBS -P va80
#PBS -q normal
#PBS -l walltime=14:50:00
#PBS -l mem=380GB
#PBS -l jobfs=100GB
#PBS -l ncpus=96
#PBS -M chirath.hettiarachchi@anu.edu.au
#PBS -l storage=gdata/ny83
#PBS -o out_mdd.txt
#PBS -e err_mdd.txt
#PBS -l software=python

module load python3/3.10.4
module load openmpi/4.1.1
source /g/data/ny83/ch9972/NeuroStim/bin/activate
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv/experiments/feature_analysis

mpirun -np 32 python3 run_simulations.py experiment.name=test32111 env=hl23net env.simulation.duration=100 env.network.dt=0.025 env.simulation.MDD=True env.ts.apply=False env.network.syn_activity=True experiment.debug=False

wait
