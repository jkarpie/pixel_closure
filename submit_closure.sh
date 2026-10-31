#!/bin/bash
#SBATCH -o pixel_closure_%A
#SBATCH --job-name=pixel_closure
#SBATCH -A hadstruc24s
#SBATCH --partition=24s
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --ntasks=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

dataset=exp
truth=JAM
Q="${PIXEL_CLOSURE_Q:-2}"
case "$Q" in
  mc|1|2|3|4|5) ;;
  *) echo "Unsupported PIXEL_CLOSURE_Q=$Q" >&2; exit 2 ;;
esac
# The small-closure-matched prior differs from earlier *_prod campaigns. Use a
# separate output root so neither completed streams nor checkpoints are mixed.
production_root="${PIXEL_PRODUCTION_ROOT:-campaign/NNLO_RUNS/${truth}_Q${Q}_${dataset}_prod_smallprior}"
python_bin=../pixel/venv/bin/python
if [[ "$production_root" = /* ]]; then
  export PIXEL_Q_MAP_CACHE_ROOT="$production_root/_q_map_cache"
else
  export PIXEL_Q_MAP_CACHE_ROOT="$PWD/$production_root/_q_map_cache"
fi
# Resume automatically when this campaign already has a manifest.
# Completed streams are skipped; unfinished streams use saved NUTS checkpoints.
export PIXEL_RESUME=0
if [[ -e "$production_root" || -L "$production_root" ]]; then
  if [[ ! -d "$production_root" || -L "$production_root" || \
        ( -e "$production_root/_q_map_cache" && ! -d "$production_root/_q_map_cache" ) || \
        -L "$production_root/_q_map_cache" ]]; then
    echo "Production output already exists: $production_root; choose a fresh path or archive it first." >&2
    exit 2
  fi
  if [[ -f "$production_root/manifest.json" ]]; then
    export PIXEL_RESUME=1
  elif [[ -n "$(find "$production_root" -mindepth 1 -maxdepth 1 ! -name _q_map_cache -print -quit)" ]]; then
    echo "Production output already exists: $production_root; choose a fresh path or archive it first." >&2
    exit 2
  fi
fi
resume="$PIXEL_RESUME"

echo "Batch job started"
export PIXEL_PROGRESS=1
mkdir -p "$(dirname "$production_root")"
if [[ "$resume" == 0 ]]; then
  : > "${production_root}.kern_data_gen.log"
  : > "${production_root}.map.log"
  : > "${production_root}.log"
fi

echo "Calling kernel maker"
PIXEL_PROGRESS=1 ./run_closure.sh --truth $truth --data union --Q $Q \
  --modes $dataset --kernels-only --cores 32 \
  --python "$python_bin" >> "${production_root}.kern_data_gen.log" 2>&1

# This command can also be run independently before submitting this job.
# A matching finite saved MAP is reused without another optimization.
echo "Preparing explicit-q MAP: L-BFGS -> Adam -> L-BFGS -> Newton (or loading validated MAP)"
"$python_bin" -m closure_full_q_integration.prepare_map \
  --truth "$truth" --data union --Q "$Q" --modes "$dataset" \
  --out "$production_root" \
  --lbfgs-maxiter "${PIXEL_MAP_LBFGS_MAXITER:-10000}" \
  --lbfgs-maxfun "${PIXEL_MAP_LBFGS_MAXFUN:-100000}" \
  --lbfgs-gtol "${PIXEL_MAP_LBFGS_GTOL:-1e-6}" \
  --lbfgs-maxls "${PIXEL_MAP_LBFGS_MAXLS:-100}" \
  --lbfgs-maxcor "${PIXEL_MAP_LBFGS_MAXCOR:-50}" \
  --adam-steps "${PIXEL_MAP_ADAM_STEPS:-10000,10000,10000}" \
  --adam-learning-rates "${PIXEL_MAP_ADAM_LEARNING_RATES:-0.01,0.003,0.001}" \
  --refine-lbfgs-maxiter "${PIXEL_MAP_REFINE_LBFGS_MAXITER:-10000}" \
  --refine-lbfgs-maxfun "${PIXEL_MAP_REFINE_LBFGS_MAXFUN:-100000}" \
  --refine-lbfgs-gtol "${PIXEL_MAP_REFINE_LBFGS_GTOL:-1e-10}" \
  --refine-lbfgs-maxls "${PIXEL_MAP_REFINE_LBFGS_MAXLS:-100}" \
  --refine-lbfgs-maxcor "${PIXEL_MAP_REFINE_LBFGS_MAXCOR:-100}" \
  --map-gradient-tol "${PIXEL_MAP_GRADIENT_TOL:-1e-1}" \
  --newton-gtol "${PIXEL_MAP_NEWTON_GTOL:-1e-4}" \
  --newton-maxiter "${PIXEL_MAP_NEWTON_MAXITER:-5}" \
  --report-every "${PIXEL_MAP_REPORT_EVERY:-100}" \
  >> "${production_root}.map.log" 2>&1

echo "Calling closure production run"
production_args=(--no-plots)
[[ "$resume" == 1 ]] && production_args+=(--resume)
./run_closure.sh \
  --truth $truth --data union --Q $Q --modes $dataset \
  --parameterization q --warmup 1000 --samples 25000 --streams 16 \
  --out "$production_root" \
  --python "$python_bin" "${production_args[@]}" \
  >> "${production_root}.log" 2>&1


echo "Batch job finished"
