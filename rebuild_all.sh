#!/usr/bin/env bash
# Full closure rebuild: kernels, truth data, and the campaign, from nothing.
#
# Use this after a correctness change in PIXEL that can move kernel values (the
# singlet sector, evolution, matching, coefficient functions).  Such a change
# invalidates BOTH halves of a closure test and they fail differently:
#
#   * the kernels, obviously -- but a cached kernel is only rejected if the
#     source fingerprint moved, so clearing the cache is the belt to that braces;
#   * the generated data, less obviously -- ``truth.json`` is read straight from
#     LHAPDF and may be byte-identical, while the ``.dat`` tables are the truth
#     FOLDED THROUGH the operators and do change.  Regenerating only when
#     ``truth.json`` differs would silently keep stale measurements.
#
# Everything is removed before it is rebuilt rather than overwritten in place, so
# a file the new code no longer emits cannot survive as a leftover.
#
# Usage:
#   ./rebuild_all.sh                      # small suites, all 6 Q, both priors
#   ./rebuild_all.sh --name my_run        # choose the campaign directory name
#   ./rebuild_all.sh --scale full         # the 13-ensemble realistic suites (slow)
#   ./rebuild_all.sh --modes lattice dis  # subset of modes
#   ./rebuild_all.sh --dry-run            # print the plan, touch nothing
#
# Concurrency is fixed at two suites x 4 workers: the kernel builder's memory
# guard puts the safe worker count at 3-4 on a 36 GB box, and a numeric
# PIXEL_KERNEL_WORKERS overrides that guard rather than obeying it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/Users/jkarpie-admin/work/building/pixel/venv/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/closure_logs}"

NAME="rebuild_$(date +%Y-%m-%d_%H%M)"
SCALE="small"
RCOND="1e-12"
MODES=(lattice dis dy exp both)
DRY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)    NAME="$2"; shift 2 ;;
        --scale)   SCALE="$2"; shift 2 ;;
        --rcond)   RCOND="$2"; shift 2 ;;
        --modes)   shift; MODES=(); while [[ $# -gt 0 && "$1" != --* ]]; do MODES+=("$1"); shift; done ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

case "$SCALE" in
    small) SUITES=(closure_JAM_truth_small closure_NNPDF_truth_small) ;;
    full)  SUITES=(closure_JAM_truth closure_NNPDF_truth) ;;
    *) echo "unknown scale '$SCALE' (expected small or full)" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PIXEL_ALLOW_THREADED_BLAS=1
export PIXEL_KERNEL_WORKERS=4 PIXEL_ASSEMBLY_WORKERS=4 PIXEL_POSTERIOR_WORKERS=4
export PIXEL_KERNEL_PROCESS_START_METHOD=spawn MAX_ASSEMBLY_ENTRIES=128000000

mkdir -p "$LOGDIR"
# bin -> venv -> repo root.  `git -C venv` alone is not enough: a `-- src/`
# pathspec is read relative to the working directory, so aiming it at venv/
# matched nothing and the dirty-tree warning silently never fired.
PIXEL_ROOT="$(cd "$(dirname "$PYTHON")/../.." && pwd)"
PIXEL_HEAD="$(git -C "$PIXEL_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
PIXEL_DIRTY="$(git -C "$PIXEL_ROOT" status --porcelain -- "$PIXEL_ROOT/src" 2>/dev/null | grep -c . || true)"
[[ "$PIXEL_DIRTY" == "0" ]] && PIXEL_DIRTY=""

FINGERPRINT="$("$PYTHON" -c 'from pixel.kernels._cache import kernel_code_fingerprint as f; print(f())' 2>/dev/null || echo unavailable)"

echo "=============================================================="
echo " campaign      : $NAME"
echo " scale         : $SCALE  (${SUITES[*]})"
echo " modes         : ${MODES[*]}"
echo " rcond         : $RCOND"
echo " kernel source : ${FINGERPRINT:0:16}"
 echo " pixel tree    : ${PIXEL_HEAD}${PIXEL_DIRTY:+  (${PIXEL_DIRTY} uncommitted src files)}"
echo "=============================================================="
if [[ -n "$PIXEL_DIRTY" ]]; then
    echo "NOTE: pixel/src has uncommitted changes.  The fingerprint above pins what"
    echo "      this run used, but the working tree can move under a long run --"
    echo "      already-imported modules do not reload, so the result would reflect"
    echo "      the code as of process start, not as of the edit."
    echo
fi
[[ "$DRY" -eq 1 ]] && { echo "(dry run: nothing removed or launched)"; exit 0; }

# -- 1. remove, do not overwrite -------------------------------------------
for suite in "${SUITES[@]}"; do
    n=$(find "$suite/data/_kernel_cache" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    rm -rf "$suite/data/_kernel_cache"
    rm -rf "$suite"/data/truthQ_*
    echo "cleared $suite: $n kernel npz + every truthQ_* member"
done
echo "kept: reference_pdfs/*.npz (raw LHAPDF replica dumps; a kernel change cannot alter them)"
echo

# -- 2. regenerate + fit, one chained job per suite -------------------------
pids=()
for suite in "${SUITES[@]}"; do
    log="$LOGDIR/${NAME}_${suite}.log"
    (
        echo "### STAGE 1: regenerate truth (all Q)"
        "$PYTHON" -m "${suite}.generate" --all
        echo "### STAGE 2: campaign"
        "$PYTHON" -m campaign.run_campaign --suite "$suite" --name "$NAME" \
            --rcond "$RCOND" --modes "${MODES[@]}" --skip-existing
        echo "### DONE"
    ) > "$log" 2>&1 &
    pids+=($!)
    echo "launched $suite -> $log"
done

echo
echo "waiting on ${#pids[@]} suite(s)..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

echo
"$PYTHON" -m campaign.make_comparisons --name "$NAME" || true
echo
"$PYTHON" -m campaign.combine_rows --name "$NAME" || true
[[ "$fail" -eq 0 ]] || { echo; echo "at least one suite exited non-zero; see $LOGDIR/${NAME}_*.log"; exit 1; }
