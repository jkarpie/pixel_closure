"""Prepare a saved full-scale explicit-q MAP without launching any sampling.

Uses L-BFGS-B, then Adam passes, then L-BFGS-B refinement and Newton.
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


def _adam_schedule(steps_raw, rates_raw):
    try:
        steps = tuple(int(part.strip()) for part in steps_raw.split(","))
        rates = tuple(float(part.strip()) for part in rates_raw.split(","))
    except ValueError as exc:
        raise ValueError("Adam steps/rates must be comma-separated integers/numbers") from exc
    return (() if steps == (0,) else steps), rates


def _format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, whole_seconds = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {whole_seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _progress_reporter(report_every):
    last_report = {}

    def progress(_point, info):
        stage = info["stage"]
        iteration = int(info["iteration"])
        elapsed = float(info["elapsed_seconds"])
        result = info["stages"].get(stage, {})
        previous = last_report.get(stage)
        terminal = info["status"] != "running"
        if (previous is not None and not terminal
                and iteration - previous[0] < report_every):
            return

        recent = ""
        if previous is not None:
            steps = iteration - previous[0]
            seconds = elapsed - previous[1]
            if steps > 0 and seconds > 0:
                rate = steps / seconds
                recent = (f" recent={steps} steps/{_format_duration(seconds)} "
                          f"({rate:.2f} steps/s)")
                limit = result.get("maxiter")
                if not terminal and limit is not None:
                    remaining = max(int(limit) - iteration, 0)
                    recent += (f" remaining={remaining} steps "
                               f"eta_to_limit={_format_duration(remaining / rate)}")
        last_report[stage] = (iteration, elapsed)
        print(
            f"MAP {stage} iteration={iteration} "
            f"action={info['action']:.12g} |gradient|={info['gradient_norm']:.6g} "
            f"|gradient|_inf={info['gradient_inf_norm']:.6g} "
            f"elapsed={elapsed:.1f}s{recent} status={info['status']}\n"
            f"  checkpoint: {info['checkpoint_path']}", flush=True,
        )
        if "message" in result:
            print(f"  {stage}: {result['message']}", flush=True)

    return progress


def prepare_cell(truth, data, q_key, mode, *, cache_root=None, options=None,
                 resume=False, check_only=False, gradient_tol=None):
    """Build exactly the source runner's final t0 target, without run_fit/MCMC."""
    from .map_start import default_map_cache_root, load_q_map_start, prepare_q_map_start
    from .variant import activated

    cache_root = Path(cache_root) if cache_root is not None else default_map_cache_root()
    with activated(truth, data, map_cache_root=cache_root) as (package, cfg, fit, _runner):
        print(f"MAP target: {package} Q={q_key} mode={mode} cache={cache_root}", flush=True)
        analysis, _fields = fit.build_analysis(q_key, mode, use_kernel_cache=True)
        model = analysis.compile()
        print("MAP covariance preparation: preliminary DIS t0 fits", flush=True)
        t0, iterations = fit._iterate_t0(q_key, mode, model, use_kernel_cache=True)
        if t0:
            analysis, _fields = fit.build_analysis(
                q_key, mode, use_kernel_cache=True, t0=t0,
            )
            model = analysis.compile()
        print(f"MAP target ready: DIS t0 iterations={iterations}", flush=True)
        from .map_start import MapOptions
        selected = options or MapOptions()
        if not check_only:
            schedule = ", ".join(
                f"{steps} steps at lr={rate:g}" for steps, rate in zip(
                    selected.adam_steps, selected.adam_learning_rates,
                )
            ) or "disabled"
            print(f"MAP schedule: L-BFGS-B ({selected.lbfgs_maxiter}) -> "
                  f"Adam [{schedule}] -> "
                  f"L-BFGS-B refinement ({selected.refine_lbfgs_maxiter}, "
                  f"gtol={selected.refine_lbfgs_gtol:g}) -> "
                  f"Newton ({selected.newton_maxiter})", flush=True)

        kwargs = dict(prior_rcond=cfg.Q_INTEGRATION_PRIOR_RCOND, cache_root=cache_root)
        if check_only:
            _coordinates, _center, info = load_q_map_start(
                model, **kwargs, gradient_tol=gradient_tol,
            )
        else:
            _coordinates, _center, info = prepare_q_map_start(
                model, **kwargs, options=options, resume=resume,
                progress=_progress_reporter(selected.report_every),
            )
        for stage, result in info.get("stages", {}).items():
            print(f"MAP {stage}: {result}", flush=True)
        print(
            f"MAP ready: action={info['action']:.12g} "
            f"|gradient|={info['gradient_norm']:.6g} "
            f"|gradient|_inf={info['gradient_inf_norm']:.6g} "
            f"warning={info.get('map_gradient_warning', False)} "
            f"cache={info['cache_status']}\n"
            f"  saved MAP: {info['cache_path']}", flush=True,
        )
        return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", choices=("JAM", "NNPDF"), default="JAM")
    parser.add_argument("--data", choices=("union", "JAM", "NNPDF"), default="union")
    parser.add_argument("--Q", default="2", help="comma list of truth Q values")
    parser.add_argument("--modes", default="exp", help="comma list of fit modes")
    cache_location = parser.add_mutually_exclusive_group()
    cache_location.add_argument("--out", type=Path, default=None,
                                help="campaign output root; save MAPs in OUT/_q_map_cache")
    cache_location.add_argument("--cache-root", type=Path, default=None,
                                help="exact MAP directory; defaults to PIXEL_Q_MAP_CACHE_ROOT")
    parser.add_argument("--lbfgs-maxiter", type=int, default=40000)
    parser.add_argument("--lbfgs-maxfun", type=int, default=400000)
    parser.add_argument("--lbfgs-gtol", type=float, default=1.0e-8)
    parser.add_argument("--lbfgs-maxls", type=int, default=100)
    parser.add_argument("--lbfgs-maxcor", type=int, default=50)
    parser.add_argument("--refine-lbfgs-maxiter", type=int, default=10000,
                        help="post-Adam L-BFGS iterations; 0 disables this stage")
    parser.add_argument("--refine-lbfgs-maxfun", type=int, default=100000)
    parser.add_argument("--refine-lbfgs-gtol", type=float, default=1.0e-10)
    parser.add_argument("--refine-lbfgs-maxls", type=int, default=100)
    parser.add_argument("--refine-lbfgs-maxcor", type=int, default=100)
    parser.add_argument("--map-gradient-tol", type=float, default=None,
                        help="Euclidean MAP gradient warning threshold (default 1e-3)")
    parser.add_argument("--newton-gtol", type=float, default=1.0e-6,
                        help="Newton solver's gradient tolerance (default 1e-6)")
    parser.add_argument("--newton-maxiter", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--report-every", type=int, default=100,
                        help="terminal progress interval; checkpoints remain independent")
    parser.add_argument("--adam-steps", default="2000,2000,2000",
                        help="steps per Adam pass; comma list, or 0 to disable")
    parser.add_argument("--adam-learning-rates", default="0.01,0.003,0.001",
                        help="one positive learning rate per Adam pass")
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
        from .map_start import MapOptions, campaign_map_cache_root

        adam_steps, adam_rates = _adam_schedule(args.adam_steps, args.adam_learning_rates)
        cache_root = (campaign_map_cache_root(args.out) if args.out is not None
                      else args.cache_root)
        options = MapOptions(
            lbfgs_maxiter=args.lbfgs_maxiter, newton_maxiter=args.newton_maxiter,
            lbfgs_maxfun=args.lbfgs_maxfun,
            lbfgs_gtol=args.lbfgs_gtol,
            lbfgs_maxls=args.lbfgs_maxls,
            lbfgs_maxcor=args.lbfgs_maxcor,
            refine_lbfgs_maxiter=args.refine_lbfgs_maxiter,
            refine_lbfgs_maxfun=args.refine_lbfgs_maxfun,
            refine_lbfgs_gtol=args.refine_lbfgs_gtol,
            refine_lbfgs_maxls=args.refine_lbfgs_maxls,
            refine_lbfgs_maxcor=args.refine_lbfgs_maxcor,
            map_gradient_tol=(args.map_gradient_tol if args.map_gradient_tol is not None
                              else MapOptions.map_gradient_tol),
            newton_gtol=args.newton_gtol,
            checkpoint_every=args.checkpoint_every,
            report_every=args.report_every,
            adam_steps=adam_steps, adam_learning_rates=adam_rates,
        )
        for q_key in scales:
            for mode in modes:
                prepare_cell(args.truth, args.data, q_key, mode,
                             cache_root=cache_root, options=options,
                             resume=args.resume, check_only=args.check_only,
                             gradient_tol=args.map_gradient_tol)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"MAP preparation failed: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
