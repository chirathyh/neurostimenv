#!/bin/bash
#PBS -P ny83
#PBS -q normal
#PBS -l walltime=5:50:00
#PBS -l mem=2090GB
#PBS -l jobfs=100GB
#PBS -l ncpus=528
#PBS -M chirath.hettiarachchi@anu.edu.au
#PBS -l storage=gdata/ny83
#PBS -o out.txt
#PBS -e err.txt
#PBS -l software=python

module load python3/3.10.4
module load openmpi/4.1.1
source /g/data/ny83/ch9972/NeuroStim/bin/activate
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv

mpirun -np 512 python3 /g/data/ny83/ch9972/NeuroStim/neurostimenv/experiments/hydra_test.py experiment.name=test33 experiment.debug=False
wait
