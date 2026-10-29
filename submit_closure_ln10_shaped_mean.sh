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
# GP length. The required argument is the GP width about that mean, one value
# shared by all nine PDFs:
#   PIXEL_CLOSURE_Q=2 sbatch submit_closure_ln10_shaped_mean.sh 0.3
# Each width gets its own output root and MAP cache, so several can run at once.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "usage: sbatch $0 <GP width, one positive number for all PDFs>" >&2
  exit 2
fi
gp_sigma="$1"
tag="$(printf '%s' "$gp_sigma" | tr '.' 'p')"
# PIXEL_T0_PRIOR=jam_shape gives the t0 fits their own JAM-like mean; see
# submit_closure_ln10_shaped_mean_t0jam.sh.
t0_prior="${PIXEL_T0_PRIOR:-same}"
case "$t0_prior" in
  same) ;;
  jam_shape) tag="${tag}_t0jam" ;;
  *) echo "PIXEL_T0_PRIOR must be same or jam_shape" >&2; exit 2 ;;
esac
export PIXEL_T0_PRIOR="$t0_prior"

export PIXEL_SHAPED_GP_SIGMA="$gp_sigma"
export PIXEL_CLOSURE_PRIOR_FORM=shaped_logrbf
export PIXEL_VANISHING_PDF_PRIOR=zero_unit
export PIXEL_LN10_PRODUCTION_ROOT="${PIXEL_SHAPED_MEAN_PRODUCTION_ROOT:-campaign/JAM_Q${PIXEL_CLOSURE_Q:-2}_exp_prod_smallprior_ln10_shaped_mean_gpsig_${tag}}"

echo "Free anchored beta means (x0=0.1); GP width=$gp_sigma for all PDFs; t0 prior=$t0_prior; output=$PIXEL_LN10_PRODUCTION_ROOT"
exec bash ./submit_closure_ln10.sh
