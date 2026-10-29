#!/bin/bash
#SBATCH -o pixel_closure_ln10_shaped_mean_%A
#SBATCH --job-name=pixel_closure_ln10_shaped_mean
#SBATCH -A hadstruc24s
#SBATCH --partition=24s
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks=1

# Free M_f, a_f, b_f in every physical-PDF GP mean (x0=0.1), with the ln(10)
# GP length. The optional argument sets the GP width about that mean:
#   sbatch submit_closure_ln10_shaped_mean.sh            # zero-unit control widths
#   sbatch submit_closure_ln10_shaped_mean.sh x0.5       # halve every width
#   sbatch submit_closure_ln10_shaped_mean.sh 1          # width 1 for all nine
#   sbatch submit_closure_ln10_shaped_mean.sh x2,g=5     # double, then g=5
# PIXEL_SHAPED_GP_SIGMA works the same way. Each width gets its own output root
# and MAP cache, so several widths can run at once.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

gp_sigma="${1:-${PIXEL_SHAPED_GP_SIGMA:-}}"
root="campaign/JAM_Q${PIXEL_CLOSURE_Q:-2}_exp_prod_smallprior_ln10_shaped_mean"
if [[ -n "$gp_sigma" ]]; then
  export PIXEL_SHAPED_GP_SIGMA="$gp_sigma"
  tag="$(printf '%s' "$gp_sigma" | tr -d ' ' | tr '.=,' 'p-_')"
  root="${root}_gpsig_${tag}"
else
  unset PIXEL_SHAPED_GP_SIGMA
fi

export PIXEL_CLOSURE_PRIOR_FORM=shaped_logrbf
export PIXEL_VANISHING_PDF_PRIOR=zero_unit
export PIXEL_LN10_PRODUCTION_ROOT="${PIXEL_SHAPED_MEAN_PRODUCTION_ROOT:-$root}"

echo "Free anchored beta means (x0=0.1); GP width=${gp_sigma:-control}; output=$PIXEL_LN10_PRODUCTION_ROOT"
exec bash ./submit_closure_ln10.sh
