#!/bin/bash
# Submit one shaped-mean job per GP width, to study under/over-estimated errors.
#   bash submit_closure_ln10_shaped_mean_scan.sh                 # default scan
#   bash submit_closure_ln10_shaped_mean_scan.sh x0.5 x1 x2      # your own list
#   PIXEL_CLOSURE_Q=3 bash submit_closure_ln10_shaped_mean_scan.sh
# Widths use the same spec as submit_closure_ln10_shaped_mean.sh.
set -euo pipefail
cd "$(dirname "$0")"

widths=("$@")
(( ${#widths[@]} )) || widths=(x0.25 x0.5 x1 x2 x4)
for width in "${widths[@]}"; do
  # Fail before submitting anything if a spec does not parse.
  PIXEL_CLOSURE_PRIOR_FORM=shaped_logrbf PIXEL_VANISHING_PDF_PRIOR=zero_unit \
    PIXEL_SHAPED_GP_SIGMA="$width" ../pixel/venv/bin/python -c \
    "from closure_JAM_truth import config as c; print('$width:', {f: round(c.gp_sigma(f), 4) for f in c.ALL_FIELDS})"
done
for width in "${widths[@]}"; do
  sbatch --job-name="pixel_closure_shaped_gpsig_${width}" \
    submit_closure_ln10_shaped_mean.sh "$width"
done
