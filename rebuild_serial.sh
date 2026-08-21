#!/usr/bin/env bash
# Serial full rebuild of the small closure suites: kernels, data, then every cell.
#
# One process at a time, on purpose -- the machine is shared.  Peak load is 4
# cores during kernel assembly and 1 during the MCMC that dominates wall clock,
# against 2 suites x 4 workers for the parallel driver.
#
# Order matters and is the point of this script.  ``both`` is lattice+DIS+DY, so
# it touches every dataset family, and kernel operators are Q-INDEPENDENT (the
# same forward operators are reused across truth members -- which is why six Q
# members share ~155 cached npz rather than needing six sets).  So a single
# ``both`` fit per suite builds the entire kernel cache, and every cell after it
# runs warm.  Doing it first also surfaces a kernel-side breakage in the first
# few minutes rather than an hour in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-/Users/jkarpie-admin/work/building/pixel/venv/bin/python}"
RESUME=0
if [[ "${1:-}" == "--resume" ]]; then RESUME=1; shift; fi
NAME="${1:-serial_$(date +%Y-%m-%d_%H%M)}"
LOG="$ROOT/closure_logs/${NAME}.log"
SUITES=(closure_NNPDF_truth_small closure_JAM_truth_small)
WARM_Q="2"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PIXEL_ALLOW_THREADED_BLAS=1
# "auto" OBEYS the memory guard; a NUMBER overrides it.  On a shared box that
# distinction is the whole ballgame: measured 2026-08-20 with five other Python
# jobs running, only 8 GiB was free and the guard put the safe kernel-worker count
# at 1, while an explicit 4 would have been honoured anyway ("honoring the
# explicit numeric override") and pushed everyone into swap.  Slower here is the
# correct trade -- the cap is re-evaluated per call, so this speeds up on its own
# when the machine frees up.
# Hard-pinned to ONE worker everywhere, by request: this is an overnight run on a
# shared machine.  Deliberately NOT "auto" -- auto obeys the memory guard but
# scales UP as the box frees up, which is the opposite of a promise to stay small.
# One worker plus the BLAS pinning above means one busy core, start to finish.
export PIXEL_KERNEL_WORKERS=1 PIXEL_ASSEMBLY_WORKERS=1 PIXEL_POSTERIOR_WORKERS=1
# JAX/XLA keeps its OWN intra-op thread pool, sized to the core count and
# untouched by the BLAS variables above or by PIXEL_KERNEL_WORKERS (which caps
# processes, not threads).  Measured 2026-08-20: with everything else pinned to 1
# the run still drew 302% CPU -- three cores -- from this pool alone.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PIXEL_KERNEL_PROCESS_START_METHOD=spawn MAX_ASSEMBLY_ENTRIES=128000000

mkdir -p "$(dirname "$LOG")"
PIXEL_ROOT="$(cd "$(dirname "$PYTHON")/../.." && pwd)"
{
  echo "=============================================================="
  echo " campaign : $NAME   (SERIAL)"
  echo " suites   : ${SUITES[*]}"
  echo " kernel fp: $("$PYTHON" -c 'from pixel.kernels._cache import kernel_code_fingerprint as f;print(f())' 2>/dev/null | cut -c1-16)"
  echo " pixel    : $(git -C "$PIXEL_ROOT" rev-parse --short HEAD 2>/dev/null) ($(git -C "$PIXEL_ROOT" status --porcelain -- "$PIXEL_ROOT/src" | grep -c . || true) uncommitted src files)"
  echo " started  : $(date '+%F %T')"
  echo "=============================================================="

  # -- 1. remove, do not overwrite ---------------------------------------
  if [[ "$RESUME" -eq 0 ]]; then
  for s in "${SUITES[@]}"; do
    n=$( { find "$s/data/_kernel_cache" -name '*.npz' 2>/dev/null || true; } | wc -l | tr -d ' ')
    rm -rf "$s/data/_kernel_cache" "$s"/data/truthQ_*
    echo "[clear] $s: $n kernel npz + all truthQ_* removed"
  done
  echo "[keep ] reference_pdfs/*.npz (raw LHAPDF dumps; kernel changes cannot alter them)"

  # -- 2. regenerate truth, one suite at a time --------------------------
  for s in "${SUITES[@]}"; do
    echo; echo "### GENERATE $s $(date '+%T')"
    nice -n 10 "$PYTHON" -m "${s}.generate" --all
  done

  # -- 3. kernel warm-up: 'both' first, per suite ------------------------
  for s in "${SUITES[@]}"; do
    echo; echo "### WARM (both, Q=$WARM_Q) $s $(date '+%T')"
    nice -n 10 "$PYTHON" -m campaign.run_campaign --suite "$s" --name "$NAME" \
        --rcond 1e-12 --q "$WARM_Q" --modes both --skip-existing
    echo "[kernels] $s now has $( { find "$s/data/_kernel_cache" -name '*.npz' 2>/dev/null || true; } | wc -l | tr -d ' ') cached npz"
  done
  else
    echo "[resume] keeping existing truth + kernels; running the campaign only"
  fi

  # -- 4. everything else, still serial ----------------------------------
  for s in "${SUITES[@]}"; do
    echo; echo "### CAMPAIGN $s $(date '+%T')"
    nice -n 10 "$PYTHON" -m campaign.run_campaign --suite "$s" --name "$NAME" \
        --rcond 1e-12 --modes lattice dis dy exp both --skip-existing
  done

  # -- 5. aggregate ------------------------------------------------------
  echo; echo "### COMPARISONS $(date '+%T')"
  nice -n 10 "$PYTHON" -m campaign.make_comparisons --name "$NAME" || true
  nice -n 10 "$PYTHON" -m campaign.combine_rows --name "$NAME" || true
  echo; echo "### ALL DONE $(date '+%T')"
} 2>&1 | tee "$LOG"
