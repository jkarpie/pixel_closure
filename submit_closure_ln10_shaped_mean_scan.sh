#!/bin/bash
# Submit one shaped-mean job per GP width; each width is shared by all PDFs.
#   PIXEL_CLOSURE_Q=2 bash submit_closure_ln10_shaped_mean_scan.sh 0.1 0.3 1 3
set -euo pipefail
cd "$(dirname "$0")"

if (( $# == 0 )); then
  echo "usage: bash $0 <GP width> [<GP width> ...]" >&2
  exit 2
fi
for width in "$@"; do
  # Fail before submitting anything if a width does not parse.
  ../pixel/venv/bin/python -c \
    "import sys; from closure_JAM_truth.config import parse_gp_sigma; parse_gp_sigma(sys.argv[1])" \
    "$width"
done
for width in "$@"; do
  sbatch --job-name="pixel_closure_shaped_gpsig_${width}" \
    submit_closure_ln10_shaped_mean.sh "$width"
done
