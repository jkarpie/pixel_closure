#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

dataset=exp
truth=JAM
Q=2
[[ $# -le 1 ]] || { echo "usage: $0 [campaign-root]" >&2; exit 2; }
production_root="${1:-campaign/NNLO_RUNS/${truth}_Q${Q}_${dataset}_prod_smallprior}"
python_bin=../pixel/venv/bin/python
if [[ "$production_root" = /* ]]; then
  export PIXEL_Q_MAP_CACHE_ROOT="$production_root/_q_map_cache"
else
  export PIXEL_Q_MAP_CACHE_ROOT="$PWD/$production_root/_q_map_cache"
fi

"$python_bin" -m campaign.report_full_scale_closure \
  --root "$production_root" \
  --out "$production_root/CLOSURE_REPORT"
