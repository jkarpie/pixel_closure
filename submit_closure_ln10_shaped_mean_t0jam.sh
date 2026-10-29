#!/bin/bash
#SBATCH -o pixel_closure_ln10_shaped_mean_t0jam_%A
#SBATCH --job-name=pixel_closure_ln10_shaped_mean_t0jam
#SBATCH -A hadstruc24s
#SBATCH --partition=24s
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks=1

# The shaped-mean closure, but the preliminary DIS fits that set the t0
# covariance reference use the JAM-like mean M_f (x/0.1)^a_f ((1-x)/0.9)^b_f
# with M_f free and a_f, b_f fixed: a=0.5 convergent, -0.2 divergent; b=3
# quarks, 5 gluon. The argument is the GP width shared by all PDFs:
#   PIXEL_CLOSURE_Q=2 sbatch submit_closure_ln10_shaped_mean_t0jam.sh 0.3
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export PIXEL_T0_PRIOR=jam_shape
exec bash ./submit_closure_ln10_shaped_mean.sh "$@"
