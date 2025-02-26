#!/bin/bash
#PBS -P va80
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=2090GB
#PBS -l jobfs=100GB
#PBS -l ncpus=528
#PBS -M chirath.hettiarachchi@anu.edu.au
#PBS -l storage=gdata/ny83
#PBS -o out_bandit.txt
#PBS -e err_bandit.txt
#PBS -l software=python

module load python3/3.10.4
module load openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv/experiments/bandit

mpirun -np 512 python3 run_mbandit.py experiment.name=testbandit experiment.seed=10 env=hl23net env.simulation.obs_win_len=4000. env.simulation.duration=10000.0 env.network.dt=0.025 env.simulation.MDD=True env.ts.apply=True env.network.syn_activity=True experiment.debug=False experiment.tqdm=False experiment.plot=False agent=mbandit agent.n_arms=13 agent.n_trials=1 agent.n_eval_trials=1 agent.pretrain=False agent.checkpoint=test6
wait
