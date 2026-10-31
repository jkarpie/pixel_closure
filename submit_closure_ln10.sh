#!/bin/bash
#SBATCH -o pixel_closure_ln10_%A
#SBATCH --job-name=pixel_closure_ln10
#SBATCH -A hadstruc24s
#SBATCH --partition=24s
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks=1

# JAM/union exp test: keep the small-prior amplitudes and jitter, but use
# ln(10) for the physical PDF log-RBF length. Nuisance fields retain ln(2).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export PIXEL_PDF_GP_LENGTH_LOG=2.302585092994046
export PIXEL_PRODUCTION_ROOT="${PIXEL_LN10_PRODUCTION_ROOT:-campaign/NNLO_RUNS/JAM_Q${PIXEL_CLOSURE_Q:-2}_exp_prod_smallprior_ln10}"

echo "PDF log-RBF length=ln(10); output=$PIXEL_PRODUCTION_ROOT"
exec bash ./submit_closure.sh
