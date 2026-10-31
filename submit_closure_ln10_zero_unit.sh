#!/bin/bash
#SBATCH -o pixel_closure_ln10_zero_unit_%A
#SBATCH --job-name=pixel_closure_ln10_zero_unit
#SBATCH -A hadstruc24s
#SBATCH --partition=24s
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks=1

# JAM/union exp: ln(10) PDF correlation length, and zero-mean unit-width
# priors for t3 and the valence-like fields; wider independent flat priors for
# t8, t15, sigma and g. Keep this campaign separate from earlier fits.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export PIXEL_VANISHING_PDF_PRIOR=zero_unit
export PIXEL_LN10_PRODUCTION_ROOT="${PIXEL_ZERO_UNIT_PRODUCTION_ROOT:-campaign/NNLO_RUNS/JAM_Q${PIXEL_CLOSURE_Q:-2}_exp_prod_smallprior_ln10_zero_unit}"

echo "PDF priors: vanishing=0±1, t8=1.5±2, t15=2±3, sigma=4±5, g=5±10; output=$PIXEL_LN10_PRODUCTION_ROOT"
exec bash ./submit_closure_ln10.sh
