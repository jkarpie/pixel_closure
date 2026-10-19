#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./run_closure.sh [options]

Run a closure test package from a clean, explicit data/cache state.
Defaults to the JAM truth copy.

Options:
  --truth TRUTH            Truth source to use: JAM or NNPDF. Defaults to JAM.
                           Also accepts the full package names directly.
  --data DATA              Which dataset selection to fit.  Defaults to union.
                             union      everything either analysis uses (46 sets,
                                        4462 rows)          -> closure_<T>_truth
                             JAM        JAM24 only (29 sets) -> closure_<T>_truth_JAM_data
                             NNPDF      NNPDF4.0 only (45)   -> closure_<T>_truth_NNPDF_data
                             mock       small smoke-test     -> closure_<T>_truth_small
  --scale SCALE            The same axis under its older name, kept working:
                           full = union, small = mock, JAM_data, NNPDF_data.

                           **--truth and --data are independent**, and that is the
                           point of the 2x4 layout: --truth says which PDF set the
                           synthetic truth is drawn from, --data says which datasets
                           are fitted.  So

                             --truth NNPDF --data JAM

                           is the NNPDF4.0 truth fitted to JAM24's dataset selection,
                           which is a different experiment from either analysis.
  -j, --cores N            Concurrency limit: the max number of simultaneous workers
                           across every parallel layer (dataset kernel farm, inner
                           assembly, posterior sweep).  Equivalent to NPROC=N, which
                           still works.  `auto` lets the kernel layer pick from CPUs,
                           task count and free memory after a 4 GiB reserve -- the
                           safer choice on a shared or memory-tight box.
                           Defaults to 12.
                           Dataset workers are additionally capped by available
                           memory (4 GiB/worker, 8 GiB reserve by default).
                           Override with PIXEL_DATASET_WORKERS, or tune the
                           estimate via PIXEL_DATASET_WORKER_GIB and
                           PIXEL_DATASET_RESERVE_GIB. The log prints the actual
                           farm size when dataset assembly begins.
                           This does NOT raise BLAS threads: see --blas-threads.
  --blas-threads N         Threads *inside* each BLAS call.  Default 1, and that is a
                           speed choice, not only a reproducibility one -- kernel
                           assembly is flat across 1/4/12 threads and the fit path
                           saturates at ~1.6x by 2, while task-level workers give
                           ~3x. Raising both oversubscribes the box and hands the
                           task-level win back.  Raising it also forfeits kernel-cache
                           reproducibility until the next --remake-kernels.
  --Q Q                    Truth member to run: mc, 1, 2, 3, 4, 5. Also accepts Q2/truthQ_2.
                           Defaults to 2.
  --prior FORM             Prior form: const_logrbf (Const mean tied to LogRBF sigma,
                           an ``a +- a`` prior) or beta_envelope (Zero mean under a
                           x^alpha (1-x)^beta tapered LogRBF).  Exported as
                           PIXEL_CLOSURE_PRIOR_FORM.  Defaults to the package's own
                           config.py value.  This drives THIS suite only; a real
                           analysis builds the prior it wants directly.
  --all                    Run every configured truth member.
  --modes MODE[,MODE...]   Modes to fit: lattice (lattice only), dis (DIS only),
                           dy (Drell-Yan only), exp (all experiment = DIS+DY),
                           both (lattice+DIS+DY). Defaults to all configured modes.
  --data-only              Run dis_audit/generate as needed, then stop before fitting.
                           Use with --remake-data to force rebuilding existing data.
  --kernels-only           Ensure selected data exist, build all selected kernel
                           caches, then stop before fitting, sampling, or plotting.
                           Use with --remake-kernels to rebuild existing caches.
  --plots-only             Recompute posteriors in memory and rewrite only plot files.
                           Existing data and result summaries are preserved.
  --remake-data            Rebuild selected package data/truthQ_* inputs.
  --remake-kernels         Clear package data/_kernel_cache before generation/fitting.
  --keep-results           Keep existing package results outputs for the selected Q.
  --trust-kernel-cache     Set PIXEL_TRUST_KERNEL_CACHE=1 during fitting: accept a
                           fingerprint-matching cache with no content check at all.
  --verify-kernel-cache    Set PIXEL_VERIFY_KERNEL_ROWS=1 during fitting: always
                           re-derive sampled rows, even when the kernel source is
                           unchanged. Slow; the default already checks the saved
                           digests and the kernel source fingerprint.
  --no-plots               Pass --no-plots to the package run_closure module.
  --fail-fast              Stop at the first failed fit. By default an all-suite
                           run records the failure and continues other cases.
  --no-noise               Pass --no-noise to the package generate module.
  --seed SEED              Pass --seed to the package generate module.
  --python PATH            Python executable. Defaults to PYTHON env or venv/bin/python.
  --dry-run                Print the commands that would run, without running them.
  -h, --help               Show this help.

Examples:
  ./run_closure.sh --Q 2
  ./run_closure.sh --truth NNPDF --Q 2 --data-only --remake-data
  ./run_closure.sh --truth JAM --Q mc --modes both --kernels-only --remake-kernels
  ./run_closure.sh --truth NNPDF --Q 2 --remake-data
  ./run_closure.sh --truth JAM --Q 2 --modes both --plots-only
  ./run_closure.sh --Q 2 --remake-data --remake-kernels
  ./run_closure.sh --Q 2 --modes both --keep-results
  ./run_closure.sh --Q 2 --modes both --prior beta_envelope

  # The four dataset-variant suites (truth family x dataset selection):
  ./run_closure.sh --truth JAM   --scale JAM_data   --Q 2 --modes both
  ./run_closure.sh --truth JAM   --scale NNPDF_data --Q 2 --modes both
  ./run_closure.sh --truth NNPDF --scale JAM_data   --Q 2 --modes both
  ./run_closure.sh --truth NNPDF --scale NNPDF_data --Q 2 --modes both
  # ...or name the package directly:
  ./run_closure.sh --truth closure_NNPDF_truth_JAM_data --Q 2
USAGE
}

# Task-level parallelism.  NPROC is the one-knob concurrency limit for kernel,
# assembly, and posterior work.  The layer-specific variables remain available
# as expert overrides, but an ordinary run only needs e.g. ``NPROC=48``.
# Every one of these spreads *independent* units of work across cores; none of
# them raises the thread count inside a single BLAS/XLA call, which is deliberate
# on both correctness and speed grounds (see the BLAS pinning note below, and the
# measurement in the PIXEL_POSTERIOR_WORKERS comment).
#: Whether the caller exported a layer variable themselves.  Captured before the
#: defaulting below overwrites the distinction, so --cores can respect an explicit
#: override instead of silently stamping over it.
PIXEL_ASSEMBLY_WORKERS_SET="${PIXEL_ASSEMBLY_WORKERS+set}"
PIXEL_KERNEL_WORKERS_SET="${PIXEL_KERNEL_WORKERS+set}"
PIXEL_POSTERIOR_WORKERS_SET="${PIXEL_POSTERIOR_WORKERS+set}"
PIXEL_DATASET_WORKERS_SET="${PIXEL_DATASET_WORKERS+set}"

NPROC="${NPROC:-12}"
#: Set by --cores/-j.  Applied in apply_concurrency() *after* argument parsing, because
#: the flag has to win over this block while an explicit PIXEL_* export from the caller
#: still wins over the flag.
CORES_FLAG=""
export PIXEL_ASSEMBLY_WORKERS="${PIXEL_ASSEMBLY_WORKERS:-$NPROC}"
export PIXEL_ASSEMBLY_BACKEND=thread
export PIXEL_KERNEL_WORKERS="${PIXEL_KERNEL_WORKERS:-$NPROC}"
export PIXEL_DATASET_WORKERS="${PIXEL_DATASET_WORKERS:-$NPROC}"
# Never fork the multithreaded JAX runtime.  Spawned kernel workers import a
# clean interpreter and cannot inherit JAX locks from the parent process.
export PIXEL_KERNEL_PROCESS_START_METHOD="${PIXEL_KERNEL_PROCESS_START_METHOD:-spawn}"
# Threads for the Hubbard-Stratonovich posterior sweep, one dense complex
# n_grid x n_grid factorization per sample.  Measured 2.97x at 14 threads and
# agreeing with the serial sweep to 4e-16 (chunked weighted sums only
# reassociate).  Note this only bites when `bilinear_posterior_moments` actually
# runs: with the default VEGAS sampler the driver takes `samples.mean` /
# `.covariance` directly and never calls it.
export PIXEL_POSTERIOR_WORKERS="${PIXEL_POSTERIOR_WORKERS:-$NPROC}"
# Detailed library checkpoints are opt-in; the closure driver's own fit summaries
# and timing lines remain visible in the default, quieter mode.
export PIXEL_PROGRESS="${PIXEL_PROGRESS:-0}"
# The three-line runtime banner is ON here even though progress is off.  Every
# variable set in this block -- the worker counts above, the BLAS pinning below --
# is overridable from the caller's environment, so the settings a multi-hour run
# actually used are not recoverable from this file afterwards.  Three lines at the
# top of the log records them.  PIXEL_RUNTIME_REPORT=0 turns it off.
export PIXEL_RUNTIME_REPORT="${PIXEL_RUNTIME_REPORT:-1}"
# BLAS threads for every backend pixel.kernels._parallel knows about
# (BLAS_THREAD_ENV_VARS).  Set BLAS_THREADS=N to raise it.
#
# **Left at 1 because it measured faster, not only for reproducibility.**  BLAS
# threading does work here (a plain 3000x3000 matmul goes 0.135s -> 0.067s at 12
# threads), but neither workload in this script is BLAS-bound:
#   * kernel assembly is completely flat (0.20s at 1, 4 and 12 threads) -- it is
#     dominated by the Mellin contour and Python-level work;
#   * the evidence/fit path saturates at ~1.6x with 2 threads and is flat at 4
#     and 8.
# Meanwhile the task-level workers above gave 2.97x, measured with BLAS pinned to
# 1.  Raising both oversubscribes the box and gives back the task-level win, so
# one BLAS thread per worker is the right split.
#
# Raising it also forfeits kernel-cache reproducibility: a threaded BLAS
# reassociates the contraction inside a single call, cache hits are accepted on
# their saved digest, and the caches outlive the run -- so a file written that way
# stays trusted indefinitely.  Rebuild with --remake-kernels after going back.
BLAS_THREADS="${BLAS_THREADS:-1}"
BLAS_FLAG=""
export OMP_NUM_THREADS="$BLAS_THREADS"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export VECLIB_MAXIMUM_THREADS="$BLAS_THREADS"
export NUMEXPR_NUM_THREADS="$BLAS_THREADS"
# Acknowledge the threaded BLAS so cache writes do not warn on every run.
export PIXEL_ALLOW_THREADED_BLAS="${PIXEL_ALLOW_THREADED_BLAS:-1}"
export MAX_ASSEMBLY_ENTRIES=128000000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-venv/bin/python}"

#: Suite selection: (scale, truth_family) -> package.  The two axes are independent:
#: ``truth_family`` in {JAM, NNPDF} picks which PDF set the injected truth is read from,
#: and ``scale`` picks the size/dataset variant:
#:
#:   full        closure_<T>_truth              JAM n NNPDF dataset intersection
#:   small       closure_<T>_truth_small        mock smoke-test
#:   JAM_data    closure_<T>_truth_JAM_data     JAM24 (2501.00665) PDF datasets
#:   NNPDF_data  closure_<T>_truth_NNPDF_data   NNPDF4.0 (2109.02653) linear/bilinear
#:
#: So e.g. --truth NNPDF --scale JAM_data = closure_NNPDF_truth_JAM_data: the NNPDF4.0
#: truth fitted to JAM24's dataset selection.  Defaults: full + JAM.
scale="full"
truth_family="JAM"

run_all=0
q_key="2"
declare -a modes=()
declare -a generate_args=()
declare -a fit_args=()
remake_data=0
remake_kernels=0
keep_results=0
# default | trust (skip all content checks) | rows (always re-derive sampled rows)
kernel_cache_check="default"
data_only=0
kernels_only=0
plots_only=0
dry_run=0

die() {
    echo "error: $*" >&2
    exit 2
}

apply_concurrency() {
    # --cores/-j is one number for every parallel layer.  It is applied here rather
    # than in the export block above so that the precedence is the useful one:
    # an explicit PIXEL_KERNEL_WORKERS=... from the caller still beats the flag, the
    # flag beats NPROC, and NPROC beats the default.  Nothing here touches BLAS
    # threads; those are a separate knob for a separate reason (see --blas-threads).
    if [[ -n "$CORES_FLAG" ]]; then
        case "$CORES_FLAG" in
            auto|AUTO) ;;
            ''|*[!0-9]*) die "--cores expects a positive integer or 'auto', got '$CORES_FLAG'" ;;
            0) die "--cores 0 is not meaningful; use 1 for serial" ;;
        esac
        NPROC="$CORES_FLAG"
        [[ -n "${PIXEL_ASSEMBLY_WORKERS_SET:-}" ]] || export PIXEL_ASSEMBLY_WORKERS="$NPROC"
        [[ -n "${PIXEL_KERNEL_WORKERS_SET:-}" ]]   || export PIXEL_KERNEL_WORKERS="$NPROC"
        [[ -n "${PIXEL_POSTERIOR_WORKERS_SET:-}" ]] || export PIXEL_POSTERIOR_WORKERS="$NPROC"
        [[ -n "${PIXEL_DATASET_WORKERS_SET:-}" ]]   || export PIXEL_DATASET_WORKERS="$NPROC"
        # 'auto' is only meaningful to the kernel layer, which is the one that consults
        # free memory.  The thread pools take a number.
        if [[ "$NPROC" == auto || "$NPROC" == AUTO ]]; then
            # 'auto' is only meaningful to the kernel layer, which is the one that
            # consults free memory.  The two thread pools need a number, so give them
            # the CPU count.  Asked of the interpreter this script already requires --
            # nproc is absent on macOS and sysctl on Linux, and getting either wrong
            # silently degrades to serial, which is a slow failure rather than a loud
            # one.
            local cpus
            # getconf is POSIX and answers on both Linux and macOS; nproc and sysctl
            # are each absent on one of them, and $PYTHON is not guaranteed to exist
            # yet (it defaults to a venv path this repo does not ship).  Getting this
            # wrong degrades silently to serial, which is the slow kind of failure.
            cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null \
                    || nproc 2>/dev/null \
                    || sysctl -n hw.ncpu 2>/dev/null \
                    || echo 1)"
            [[ "$cpus" =~ ^[0-9]+$ ]] || cpus=1
            export PIXEL_ASSEMBLY_WORKERS="$cpus"
            export PIXEL_POSTERIOR_WORKERS="$cpus"
        fi
    fi
    if [[ -n "$BLAS_FLAG" ]]; then
        case "$BLAS_FLAG" in ''|*[!0-9]*) die "--blas-threads expects a positive integer" ;; esac
        BLAS_THREADS="$BLAS_FLAG"
        export OMP_NUM_THREADS="$BLAS_THREADS" OPENBLAS_NUM_THREADS="$BLAS_THREADS"
        export MKL_NUM_THREADS="$BLAS_THREADS" VECLIB_MAXIMUM_THREADS="$BLAS_THREADS"
        export NUMEXPR_NUM_THREADS="$BLAS_THREADS"
    fi
}

set_truth() {
    # Accepts a family (JAM/NNPDF) or a full package name (which also fixes scale).
    case "$1" in
        JAM|jam|Jam|JAM20|jam20) truth_family="JAM" ;;
        NNPDF|nnpdf|Nnpdf|NNPDF40|nnpdf40) truth_family="NNPDF" ;;
        closure_JAM_truth) truth_family="JAM"; scale="full" ;;
        closure_JAM_truth_small) truth_family="JAM"; scale="small" ;;
        closure_JAM_truth_JAM_data) truth_family="JAM"; scale="JAM_data" ;;
        closure_JAM_truth_NNPDF_data) truth_family="JAM"; scale="NNPDF_data" ;;
        closure_NNPDF_truth) truth_family="NNPDF"; scale="full" ;;
        closure_NNPDF_truth_small) truth_family="NNPDF"; scale="small" ;;
        closure_NNPDF_truth_JAM_data) truth_family="NNPDF"; scale="JAM_data" ;;
        closure_NNPDF_truth_NNPDF_data) truth_family="NNPDF"; scale="NNPDF_data" ;;
        *) die "unknown truth '$1' (expected JAM or NNPDF)" ;;
    esac
}

set_scale() {
    case "$1" in
        full|Full|FULL|big|large) scale="full" ;;
        small|Small|SMALL|mock|smoke) scale="small" ;;
        JAM_data|jam_data|JAMdata|jamdata) scale="JAM_data" ;;
        NNPDF_data|nnpdf_data|NNPDFdata|nnpdfdata) scale="NNPDF_data" ;;
        *) die "unknown scale '$1' (expected full, small, JAM_data or NNPDF_data)" ;;
    esac
}

set_data() {
    # The dataset axis, named for what it selects rather than for a size.  Bare "JAM"
    # and "NNPDF" are accepted here and NOT in set_scale: under --data they can only
    # mean a dataset selection, while under --scale they would sit ambiguously next to
    # --truth JAM.  Same reason "union" is the spelling rather than "full" -- the
    # headline suites stopped being the intersection on 2026-09-09 and "full" says
    # nothing about which.
    case "$1" in
        union|Union|UNION|both|all|full|Full|FULL) scale="full" ;;
        JAM|jam|JAM24|jam24|JAM_data|jam_data) scale="JAM_data" ;;
        NNPDF|nnpdf|NNPDF40|nnpdf40|NNPDF_data|nnpdf_data) scale="NNPDF_data" ;;
        mock|Mock|MOCK|small|Small|SMALL|smoke) scale="small" ;;
        *) die "unknown data selection '$1' (expected union, JAM, NNPDF or mock)" ;;
    esac
}

resolve_package() {
    local suffix=""
    case "$scale" in
        full) suffix="" ;;
        small) suffix="_small" ;;
        JAM_data) suffix="_JAM_data" ;;
        NNPDF_data) suffix="_NNPDF_data" ;;
        *) die "unknown scale '$scale'" ;;
    esac
    case "$truth_family" in
        JAM) CLOSURE_PACKAGE="closure_JAM_truth${suffix}" ;;
        NNPDF) CLOSURE_PACKAGE="closure_NNPDF_truth${suffix}" ;;
        *) die "unknown truth family '$truth_family'" ;;
    esac
    [[ -d "$CLOSURE_PACKAGE" ]] || die "no such suite directory: $CLOSURE_PACKAGE"
}

normalize_q() {
    local raw="$1"
    raw="${raw#truthQ_}"
    raw="${raw#truthq_}"
    raw="${raw#Q}"
    raw="${raw#q}"
    case "$raw" in
        mc|1|2|3|4|5) printf '%s\n' "$raw" ;;
        *) die "unknown Q '$1' (expected mc, 1, 2, 3, 4, or 5)" ;;
    esac
}

add_modes() {
    local value="$1"
    local part
    local old_ifs="$IFS"
    IFS=,
    for part in $value; do
        case "$part" in
            lattice|dis|dy|exp|both|synthetic_z_plumbing) modes+=("$part") ;;
            *) die "unknown mode '$part' (expected lattice, dis, dy, exp, or both)" ;;
        esac
    done
    IFS="$old_ifs"
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$dry_run" -eq 0 ]]; then
        "$@"
    fi
}

remove_path() {
    local path="$1"
    if [[ -e "$path" || -L "$path" ]]; then
        run rm -rf "$path"
    fi
}

truth_label() {
    printf 'truthQ_%s\n' "$1"
}

run_generate() {
    local -a cmd
    cmd=("$PYTHON" -m "${CLOSURE_PACKAGE}.generate" "$@")
    if [[ "${#generate_args[@]}" -gt 0 ]]; then
        cmd+=("${generate_args[@]}")
    fi
    run "${cmd[@]}"
}

run_fit() {
    local key="$1"
    local -a cmd
    case "$kernel_cache_check" in
        trust)
            cmd=(env PIXEL_TRUST_KERNEL_CACHE=1 "$PYTHON" -m "${CLOSURE_PACKAGE}.run_closure")
            ;;
        rows)
            cmd=(env PIXEL_VERIFY_KERNEL_ROWS=1 "$PYTHON" -m "${CLOSURE_PACKAGE}.run_closure")
            ;;
        *)
            cmd=("$PYTHON" -m "${CLOSURE_PACKAGE}.run_closure")
            ;;
    esac
    if [[ -n "$key" ]]; then
        cmd+=(--Q "$key")
    fi
    if [[ "${#modes[@]}" -gt 0 ]]; then
        cmd+=(--modes "${modes[@]}")
    fi
    if [[ "${#fit_args[@]}" -gt 0 ]]; then
        cmd+=("${fit_args[@]}")
    fi
    run "${cmd[@]}"
}

# Honor a preset CLOSURE_PACKAGE env var (seeds both scale and family).
if [[ -n "${CLOSURE_PACKAGE:-}" ]]; then
    set_truth "$CLOSURE_PACKAGE"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --truth|--truth-source)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            set_truth "$2"
            shift 2
            ;;
        --dataset|--truth-dataset)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            set_truth "$2"
            shift 2
            ;;
        --scale|--variant)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            set_scale "$2"
            shift 2
            ;;
        --data|--data-variant|--datasets)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            set_data "$2"
            shift 2
            ;;
        -j|--cores|--jobs)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            CORES_FLAG="$2"
            shift 2
            ;;
        --blas-threads)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            BLAS_FLAG="$2"
            shift 2
            ;;
        --prior|--prior-form)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            case "$2" in
                const_logrbf|beta_envelope) export PIXEL_CLOSURE_PRIOR_FORM="$2" ;;
                *) die "unknown prior '$2' (expected const_logrbf or beta_envelope)" ;;
            esac
            shift 2
            ;;
        --Q|--q|-Q)
            [[ $# -ge 2 ]] || die "--Q requires a value"
            q_key="$(normalize_q "$2")"
            run_all=0
            shift 2
            ;;
        --all)
            run_all=1
            shift
            ;;
        --modes|--mode)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            add_modes "$2"
            shift 2
            ;;
        --data-only|--generate-only|--make-data-only)
            data_only=1
            shift
            ;;
        --kernels-only|--kernel-only|--make-kernels-only)
            kernels_only=1
            fit_args+=("--kernels-only")
            shift
            ;;
        --plots-only|--plot-only)
            plots_only=1
            fit_args+=("--plots-only")
            shift
            ;;
        --remake-data)
            remake_data=1
            shift
            ;;
        --remake-kernels|--remake-kernel-cache)
            remake_kernels=1
            shift
            ;;
        --keep-results)
            keep_results=1
            shift
            ;;
        --trust-kernel-cache|--trust-kernels)
            kernel_cache_check="trust"
            shift
            ;;
        --verify-kernel-cache|--no-trust-kernel-cache)
            kernel_cache_check="rows"
            shift
            ;;
        --no-plots)
            fit_args+=("--no-plots")
            shift
            ;;
        --fail-fast)
            fit_args+=("--fail-fast")
            shift
            ;;
        --no-noise)
            generate_args+=("--no-noise")
            shift
            ;;
        --seed)
            [[ $# -ge 2 ]] || die "--seed requires a value"
            generate_args+=("--seed" "$2")
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || die "--python requires a value"
            PYTHON="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option '$1'"
            ;;
    esac
done

apply_concurrency

# What this run will actually use, printed before any of it starts.  A multi-hour run
# on a shared box is the wrong place to discover the concurrency was the 12-core
# default, and PIXEL's own runtime banner appears only once Python is up -- after the
# audit step has already gone by.
report_concurrency() {
    printf 'concurrency: dataset=%s kernel=%s assembly=%s posterior=%s | BLAS threads=%s\n' \
        "$PIXEL_DATASET_WORKERS" \
        "$PIXEL_KERNEL_WORKERS" "$PIXEL_ASSEMBLY_WORKERS" \
        "$PIXEL_POSTERIOR_WORKERS" "$OMP_NUM_THREADS" >&2
    if [[ "${OMP_NUM_THREADS:-1}" != 1 ]]; then
        printf '  note: BLAS threads > 1 oversubscribes against the task workers above,\n' >&2
        printf '        and forfeits kernel-cache reproducibility until --remake-kernels.\n' >&2
    fi
}
resolve_package
report_concurrency
data_label() {
    case "$scale" in
        full)       echo "union (JAM u NNPDF), 46 sets" ;;
        small)      echo "mock smoke-test" ;;
        JAM_data)   echo "JAM24, 29 sets" ;;
        NNPDF_data) echo "NNPDF4.0, 45 sets" ;;
        *)          echo "$scale" ;;
    esac
}
printf 'suite: %s  (truth=%s, data=%s)\n' \
    "$CLOSURE_PACKAGE" "$truth_family" "$(data_label)" >&2

DATA_DIR="$ROOT_DIR/$CLOSURE_PACKAGE/data"
RESULTS_DIR="$ROOT_DIR/$CLOSURE_PACKAGE/results"
DIS_MANIFEST="$DATA_DIR/dis_manifest.json"

if [[ "$plots_only" -eq 1 ]]; then
    [[ "$data_only" -eq 0 ]] || die "--plots-only cannot be combined with --data-only"
    [[ "$kernels_only" -eq 0 ]] || die "--plots-only cannot be combined with --kernels-only"
    [[ "$remake_data" -eq 0 ]] || die "--plots-only cannot be combined with --remake-data"
    [[ "$remake_kernels" -eq 0 ]] || die "--plots-only cannot be combined with --remake-kernels"
    for arg in "${fit_args[@]}"; do
        [[ "$arg" != "--no-plots" ]] || die "--plots-only cannot be combined with --no-plots"
    done
    keep_results=1
fi

if [[ "$data_only" -eq 1 && "$kernels_only" -eq 1 ]]; then
    die "--data-only and --kernels-only are separate stopping points; choose one"
fi

if [[ "$dry_run" -eq 0 && ! -x "$PYTHON" ]]; then
    die "expected executable Python at '$PYTHON'; set PYTHON or pass --python"
fi

mkdir -p "$DATA_DIR"
if [[ "$data_only" -eq 0 && "$kernels_only" -eq 0 ]]; then
    mkdir -p "$RESULTS_DIR"
fi

declare -a q_keys=()
if [[ "$run_all" -eq 1 ]]; then
    q_keys=(mc 1 2 3 4 5)
else
    q_keys=("$q_key")
fi

if [[ "$remake_kernels" -eq 1 ]]; then
    remove_path "$DATA_DIR/_kernel_cache"
fi

if [[ "$remake_data" -eq 1 ]]; then
    for key in "${q_keys[@]}"; do
        remove_path "$DATA_DIR/$(truth_label "$key")"
    done
fi

if [[ "$data_only" -eq 0 && "$kernels_only" -eq 0 && "$keep_results" -eq 0 ]]; then
    remove_path "$RESULTS_DIR/comparison"
    for key in "${q_keys[@]}"; do
        remove_path "$RESULTS_DIR/$(truth_label "$key")"
    done
fi

if [[ "$plots_only" -eq 1 ]]; then
    for key in "${q_keys[@]}"; do
        [[ -f "$DATA_DIR/$(truth_label "$key")/truth.json" ]] || \
            die "missing data for $(truth_label "$key"); run without --plots-only first"
    done
else
    need_generate="$remake_data"
    for key in "${q_keys[@]}"; do
        if [[ ! -f "$DATA_DIR/$(truth_label "$key")/truth.json" ]]; then
            need_generate=1
        fi
    done

    if [[ "$need_generate" -eq 1 || ! -f "$DIS_MANIFEST" ]]; then
        run "$PYTHON" -m "${CLOSURE_PACKAGE}.dis_audit"
    fi

    if [[ "$need_generate" -eq 1 ]]; then
        if [[ "$run_all" -eq 1 ]]; then
            run_generate --all
        else
            for key in "${q_keys[@]}"; do
                run_generate --Q "$key"
            done
        fi
    fi
fi

if [[ "$data_only" -eq 1 ]]; then
    exit 0
fi

if [[ "$run_all" -eq 1 ]]; then
    run_fit ""
else
    for key in "${q_keys[@]}"; do
        run_fit "$key"
    done
fi
