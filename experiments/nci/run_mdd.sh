#!/bin/bash
#PBS -P va80
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=2090GB
#PBS -l jobfs=100GB
#PBS -l ncpus=528
#PBS -M chirath.hettiarachchi@anu.edu.au
#PBS -l storage=gdata/ny83
#PBS -o out_mdd.txt
#PBS -e err_mdd.txt
#PBS -l software=python

module load python3/3.10.4
module load openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv/experiments/feature_analysis

mpirun -np 512 python3 run_simulations.py experiment.name=mdd experiment.seed=10 env=hl23net env.simulation.duration=28000.0 env.network.dt=0.025 env.simulation.MDD=True env.ts.apply=False env.network.syn_activity=True experiment.debug=False

wait
