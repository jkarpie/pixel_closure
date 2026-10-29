#!/bin/bash
# Submit one shaped-mean job per GP width; each width is shared by all PDFs.
#   PIXEL_CLOSURE_Q=2 bash submit_closure_ln10_shaped_mean_scan.sh 0.1 0.3 1 3
#   PIXEL_CLOSURE_Q=2 bash submit_closure_ln10_shaped_mean_scan.sh --t0-mean 0.1 0.3 1 3
# --t0-mean uses the t0 fit's PDFs as the production mean (see
# submit_closure_ln10_t0_mean.sh).
set -euo pipefail
cd "$(dirname "$0")"

launcher=submit_closure_ln10_shaped_mean.sh
job=shaped
if [[ "${1:-}" == "--t0-mean" ]]; then
  launcher=submit_closure_ln10_t0_mean.sh
  job=t0mean
  shift
fi
if (( $# == 0 )); then
  echo "usage: bash $0 [--t0-mean] <GP width> [<GP width> ...]" >&2
  exit 2
fi
for width in "$@"; do
  # Fail before submitting anything if a width does not parse.
  ../pixel/venv/bin/python -c \
    "import sys; from closure_JAM_truth.config import parse_gp_sigma; parse_gp_sigma(sys.argv[1])" \
    "$width"
done
for width in "$@"; do
  sbatch --job-name="pixel_closure_${job}_gpsig_${width}" "$launcher" "$width"
done
