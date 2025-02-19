#!/bin/bash
#PBS -P va80
#PBS -l ncpus=1
#PBS -l mem=2GB
#PBS -l jobfs=2GB
#PBS -q copyq
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/ny83
#PBS -l wd
#PBS -o out11.txt
#PBS -e err11.txt

cd /g/data/ny83/ch9972/NeuroStim/neurostimenv/
zip -r results_nci.zip results/
