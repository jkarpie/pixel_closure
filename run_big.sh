#!/usr/bin/env bash
# Full-scale closure on a many-core machine.
#
#   ./run_big.sh -j 64 --data JAM --python /path/to/pixel/venv/bin/python
#
# A thin wrapper over run_closure.sh that fixes the things a big-box production run
# should not have to remember, and nothing else.  Everything it does is overridable.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CORES=""
PYTHON_BIN="${PYTHON:-}"
TRUTH="JAM"
DATA="union"
QS=""
MODES="both"
EXTRA=()

usage() {
    cat <<'USAGE'
Usage: ./run_big.sh -j N [options] [-- extra run_closure.sh args]

Full-scale closure, sized for a many-core box.  Defaults to the JAM truth on
the union dataset; --truth and --data select any of the eight suites.

  -j, --cores N       Concurrency for every parallel layer.  REQUIRED -- there is no
                      sensible default for someone else's machine, and the underlying
                      default (12) would waste a big box silently.
                      'auto' lets the kernel layer size itself from CPUs, task count
                      and free memory after a 4 GiB reserve.
      --python PATH   Interpreter with pixel importable.  This repo ships no venv.
      --truth T       Which PDF the synthetic truth is drawn from: JAM (default)
                      or NNPDF.
      --data D        Which datasets are fitted: union (default), JAM, NNPDF, mock.
                      Independent of --truth -- `--truth NNPDF --data JAM` is the
                      NNPDF4.0 truth on JAM24's dataset selection.
      --Q LIST        Comma-separated truth members, e.g. mc,1,2,3,4,5.  Default: all.
      --modes LIST    Default 'both' (lattice+DIS+DY).  See run_closure.sh --help.
      --dry-run       Print what would run.
  -h, --help          This message.

Everything after `--` is passed through to run_closure.sh untouched.

WHAT THIS SIZES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
Sizes: the task-level worker counts (kernel assembly, inner assembly, posterior
sweep), all from your -j.

Does NOT raise BLAS threads, and that is a speed decision rather than a purity one.
Measured in guides/performance_and_memory.md: kernel assembly is flat across 1/4/12
BLAS threads, the fit path saturates at ~1.6x by 2, while task-level workers give
~3x with BLAS pinned to 1.  Raising both oversubscribes the box and hands the
task-level win straight back.  It also forfeits kernel-cache reproducibility until
the next --remake-kernels.  Use run_closure.sh --blas-threads N if you want to
re-measure that on your own hardware.

Does NOT parallelise the NUTS trajectory.  There is a `multi_chain(workers=)` knob and
it will not make your fit faster: MAP-per-chain is SciPy and the NUTS trajectory loop
is Python-level, so both are GIL-bound.  That is a property of the interpreter, not of
your machine.  Its value is the R-hat diagnostic, not throughput.

MEMORY IS THE LIKELY BINDING CONSTRAINT, NOT CORES
---------------------------------------------------
The outer kernel layer is a *process* pool, so each worker carries its own Python and
JAX runtime and duplicates every temporary.  On a many-core box the core count is
usually not what runs out first.  If you are unsure, start with `-j auto`: it is the
only setting that consults free memory (after a 4 GiB reserve) instead of trusting a
number.  An explicit -j larger than that safe count still runs, with a loud warning.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--cores|--jobs) [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
                           CORES="$2"; shift 2 ;;
        --python)          [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
                           PYTHON_BIN="$2"; shift 2 ;;
        --truth)           TRUTH="$2"; shift 2 ;;
        --data|--scale)    DATA="$2"; shift 2 ;;
        --Q|--q)           QS="$2"; shift 2 ;;
        --modes)           MODES="$2"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        --)                shift; EXTRA+=("$@"); break ;;
        *)                 EXTRA+=("$1"); shift ;;
    esac
done

[[ -n "$CORES" ]] || { echo "error: -j/--cores is required (see --help for why)" >&2; exit 2; }
if [[ -n "$PYTHON_BIN" ]]; then
    [[ -x "$PYTHON_BIN" ]] || { echo "error: --python '$PYTHON_BIN' is not executable" >&2; exit 2; }
    export PYTHON="$PYTHON_BIN"
fi
"${PYTHON:-python3}" -c 'import pixel' 2>/dev/null || {
    echo "error: 'import pixel' failed under '${PYTHON:-python3}'." >&2
    echo "       This repo ships no venv; pass --python /path/to/pixel/venv/bin/python." >&2
    exit 2
}

# One process per physical task, spawned rather than forked: forking a process that has
# already built JAX's runtime threads can deadlock before the first task runs.
export PIXEL_KERNEL_PROCESS_START_METHOD="${PIXEL_KERNEL_PROCESS_START_METHOD:-spawn}"
# The three-line banner records what a multi-hour run actually used; every knob here is
# overridable from the environment, so without it the settings are unrecoverable later.
export PIXEL_RUNTIME_REPORT="${PIXEL_RUNTIME_REPORT:-1}"
# Progress checkpoints are worth their noise on a run measured in hours.
export PIXEL_PROGRESS="${PIXEL_PROGRESS:-1}"

args=(--truth "$TRUTH" --data "$DATA" -j "$CORES" --modes "$MODES")
if [[ -n "$QS" ]]; then args+=(--Q "$QS"); else args+=(--all); fi
args+=("${EXTRA[@]+"${EXTRA[@]}"}")

echo "run_big.sh: ./run_closure.sh ${args[*]}" >&2
exec ./run_closure.sh "${args[@]}"
