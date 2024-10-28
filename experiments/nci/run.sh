#!/bin/bash
#PBS -P ny83
#PBS -q normal
#PBS -l walltime=15:50:00
#PBS -l mem=380GB
#PBS -l jobfs=100GB
#PBS -l ncpus=96
#PBS -M chirath.hettiarachchi@anu.edu.au
#PBS -l storage=gdata/ny83
#PBS -o out.txt
#PBS -e err.txt
#PBS -l software=python

module load python3/3.10.4
module load openmpi/4.1.1
source /g/data/ny83/ch9972/NeuroStim/bin/activate
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv

mpirun -np 72 python3 /g/data/ny83/ch9972/NeuroStim/neurostimenv/experiments/hydra_test.py experiment.name=test33 experiment.debug=False
wait
