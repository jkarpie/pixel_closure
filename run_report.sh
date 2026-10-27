#!/bin/bash


cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

dataset=exp
truth=JAM
Q=2
production_root="campaign/${truth}_Q${Q}_${dataset}_prod"
python_bin=../pixel/venv/bin/python
export PIXEL_Q_MAP_CACHE_ROOT="$PWD/$production_root/_q_map_cache"

../pixel/venv/bin/python -m campaign.report_full_scale_closure \
  --root ${production_root} \
  --out ${production_root}/CLOSURE_REPORT
