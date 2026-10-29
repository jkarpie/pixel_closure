#!/bin/bash
#SBATCH -o pixel_closure_ln10_t0_mean_%A
#SBATCH --job-name=pixel_closure_ln10_t0_mean
#SBATCH -A hadstruc24s
#SBATCH --partition=24s
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks=1

# Production GP mean = the PDFs of the t0 fit. The preliminary DIS fits that
# set the t0 covariance reference use the JAM-like mean
# M_f (x/0.1)^a_f ((1-x)/0.9)^b_f with M_f free and a_f, b_f fixed: a=0.5
# convergent, -0.2 divergent; b=3 quarks, 5 gluon. The final t0 fit's PDFs are
# then the fixed production mean. The argument is the GP width shared by all
# PDFs, in both the t0 fits and production:
#   PIXEL_CLOSURE_Q=2 sbatch submit_closure_ln10_t0_mean.sh 0.3
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export PIXEL_CLOSURE_PRIOR_FORM=t0_logrbf
exec bash ./submit_closure_ln10_shaped_mean.sh "$@"
