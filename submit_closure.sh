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
Q=2
production_root="campaign/${truth}_Q${Q}_${dataset}_prod"
python_bin=../pixel/venv/bin/python
export PIXEL_Q_MAP_CACHE_ROOT="${PIXEL_Q_MAP_CACHE_ROOT:-$PWD/campaign/_q_map_cache}"

if [[ -e "$production_root" ]] && [[ ! -d "$production_root" || -n "$(ls -A "$production_root")" ]]; then
  echo "Production output already exists: $production_root; choose a fresh path or archive it first." >&2
  exit 2
fi

echo "Batch job started"
export PIXEL_PROGRESS=1
mkdir -p campaign

echo "Calling kernel maker"
PIXEL_PROGRESS=1 ./run_closure.sh --truth $truth --data union --Q $Q \
  --modes $dataset --kernels-only --cores 32 \
  --python "$python_bin" > "${production_root}.kern_data_gen.log" 2>&1

# This command can also be run independently before submitting this job.
# A matching converged saved MAP is reused without another optimization.
echo "Preparing explicit-q MAP (or loading an existing validated MAP)"
"$python_bin" -m closure_full_q_integration.prepare_map \
  --truth "$truth" --data union --Q "$Q" --modes "$dataset" \
  --lbfgs-maxiter "${PIXEL_MAP_LBFGS_MAXITER:-40000}" \
  --lbfgs-maxfun "${PIXEL_MAP_LBFGS_MAXFUN:-400000}" \
  --newton-maxiter "${PIXEL_MAP_NEWTON_MAXITER:-2000}" \
  > "${production_root}.map.log" 2>&1

echo "Calling closure production run"
./run_closure.sh \
  --truth $truth --data union --Q $Q --modes $dataset \
  --parameterization q --warmup 1000 --samples 25000 --streams 16 \
  --out "$production_root" --no-plots \
  --python "$python_bin" \
  > "${production_root}.log" 2>&1


echo "Batch job finished"
