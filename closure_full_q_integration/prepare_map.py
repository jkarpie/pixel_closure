"""Prepare a saved full-scale explicit-q MAP without launching any sampling.

Example: python -m closure_full_q_integration.prepare_map --truth JAM \
    --data union --Q 2 --modes exp --lbfgs-maxiter 40000 --newton-maxiter 2000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SCALES = ("mc", "1", "2", "3", "4", "5")
MODES = ("lattice", "dis", "dy", "exp", "both")


def _choices(raw, choices, label):
    values = tuple(dict.fromkeys(part.strip() for part in raw.split(",")))
    if not values or any(value not in choices for value in values):
        raise ValueError(f"{label} must be a comma list drawn from {','.join(choices)}")
    return values


def prepare_cell(truth, data, q_key, mode, *, cache_root=None, options=None,
                 resume=False, check_only=False):
    """Build exactly the source runner's final t0 target, without run_fit/MCMC."""
    from .map_start import default_map_cache_root, load_q_map_start, prepare_q_map_start
    from .variant import activated

    cache_root = Path(cache_root) if cache_root is not None else default_map_cache_root()
    with activated(truth, data, map_cache_root=cache_root) as (package, cfg, fit, _runner):
        print(f"MAP target: {package} Q={q_key} mode={mode} cache={cache_root}", flush=True)
        analysis, _fields = fit.build_analysis(q_key, mode, use_kernel_cache=True)
        model = analysis.compile()
        t0, iterations = fit._iterate_t0(q_key, mode, model, use_kernel_cache=True)
        if t0:
            analysis, _fields = fit.build_analysis(
                q_key, mode, use_kernel_cache=True, t0=t0,
            )
            model = analysis.compile()
        print(f"MAP target ready: DIS t0 iterations={iterations}", flush=True)

        def progress(_point, info):
            print(
                f"MAP {info['stage']} iteration={info['iteration']} "
                f"action={info['action']:.12g} |gradient|={info['gradient_norm']:.6g} "
                f"elapsed={info['elapsed_seconds']:.1f}s status={info['status']}\n"
                f"  checkpoint: {info['checkpoint_path']}", flush=True,
            )

        kwargs = dict(prior_rcond=cfg.Q_INTEGRATION_PRIOR_RCOND, cache_root=cache_root)
        if check_only:
            _coordinates, _center, info = load_q_map_start(model, **kwargs)
        else:
            _coordinates, _center, info = prepare_q_map_start(
                model, **kwargs, options=options, resume=resume, progress=progress,
            )
        for stage, result in info.get("stages", {}).items():
            print(f"MAP {stage}: {result}", flush=True)
        print(
            f"MAP ready: action={info['action']:.12g} "
            f"|gradient|={info['gradient_norm']:.6g} cache={info['cache_status']}\n"
            f"  saved MAP: {info['cache_path']}", flush=True,
        )
        return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", choices=("JAM", "NNPDF"), default="JAM")
    parser.add_argument("--data", choices=("union", "JAM", "NNPDF"), default="union")
    parser.add_argument("--Q", default="2", help="comma list of truth Q values")
    parser.add_argument("--modes", default="exp", help="comma list of fit modes")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="default: PIXEL_Q_MAP_CACHE_ROOT or campaign/_q_map_cache")
    parser.add_argument("--lbfgs-maxiter", type=int, default=40000)
    parser.add_argument("--lbfgs-maxfun", type=int, default=400000)
    parser.add_argument("--newton-maxiter", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true",
                        help="restart optimization from a matching saved candidate")
    parser.add_argument("--check-only", action="store_true",
                        help="validate an existing MAP, without optimization or sampling")
    args = parser.parse_args(argv)
    try:
        scales = _choices(args.Q, SCALES, "--Q")
        modes = _choices(args.modes, MODES, "--modes")
        if args.resume and args.check_only:
            raise ValueError("--resume cannot be combined with --check-only")
        # Match the stream workers' numerical environment before importing PIXEL.
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            "PIXEL_ASSEMBLY_WORKERS", "PIXEL_KERNEL_WORKERS",
            "PIXEL_DATASET_WORKERS", "PIXEL_POSTERIOR_WORKERS",
        ):
            os.environ[name] = "1"
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
        from .map_start import MapOptions

        options = MapOptions(args.lbfgs_maxiter, args.newton_maxiter,
                             args.lbfgs_maxfun, args.checkpoint_every)
        for q_key in scales:
            for mode in modes:
                prepare_cell(args.truth, args.data, q_key, mode,
                             cache_root=args.cache_root, options=options,
                             resume=args.resume, check_only=args.check_only)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"MAP preparation failed: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
