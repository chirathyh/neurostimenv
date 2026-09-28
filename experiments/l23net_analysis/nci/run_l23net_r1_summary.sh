#!/bin/bash
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/ny83
#PBS -l software=python
#PBS -l wd
#PBS -N l23_r1_sum
set -euo pipefail
: "${SUITE_DIRECTORY:?Missing suite}"
: "${EXPECTED_COMMIT:?Missing commit}"
cd "${PBS_O_WORKDIR}"
exec > >(tee -a "${SUITE_DIRECTORY}/summary_live.log") 2>&1
trap 'qstat -fx "${PBS_JOBID}" > "${SUITE_DIRECTORY}/summary_qstat_at_exit.txt" 2>&1 || true' EXIT
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || exit 2
git diff --quiet && git diff --cached --quiet || exit 2
module load python3/3.10.4
source /g/data/ny83/ch9972/NeuroStim/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/l23net_analysis/analyze_l23net_r1.py --suite "${SUITE_DIRECTORY}"
