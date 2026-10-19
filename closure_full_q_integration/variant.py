"""Isolated q-integration overlay for each full-scale truth/data combination."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from threading import RLock

from . import config as policy


ROOT = Path(__file__).resolve().parent
SOURCE_SUITES = {
    ("JAM", "union"): "closure_JAM_truth",
    ("JAM", "JAM"): "closure_JAM_truth_JAM_data",
    ("JAM", "NNPDF"): "closure_JAM_truth_NNPDF_data",
    ("NNPDF", "union"): "closure_NNPDF_truth",
    ("NNPDF", "JAM"): "closure_NNPDF_truth_JAM_data",
    ("NNPDF", "NNPDF"): "closure_NNPDF_truth_NNPDF_data",
}
_LOCK = RLock()


@contextmanager
def activated(truth: str, data: str, *, output_root: Path | None = None):
    """Reuse source data/kernels while keeping q results and caches separate."""
    key = (truth.upper(), "union" if data.lower() == "union" else data.upper())
    try:
        package = SOURCE_SUITES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported full-scale truth/data selection: {key}") from exc
    with _LOCK:
        cfg = importlib.import_module(f"{package}.config")
        fit = importlib.import_module(f"{package}.fit")
        runner = importlib.import_module(f"{package}.run_closure")
        output_root = (Path(output_root) if output_root is not None else
                       ROOT / "outputs" / key[0].lower() / key[1].lower())
        overrides = {
            "Q_INTEGRATION_ENABLED": True,
            "Q_INTEGRATION_SAMPLER": policy.Q_INTEGRATION_SAMPLER,
            "Q_INTEGRATION_WARMUP": int(cfg.DIRECT_NUTS_WARMUP),
            "Q_INTEGRATION_INITIAL_STEP_SIZE": float(cfg.DIRECT_NUTS_INITIAL_STEP_SIZE),
            "Q_INTEGRATION_MAX_TREE_DEPTH": int(cfg.DIRECT_NUTS_MAX_TREE_DEPTH),
            "Q_INTEGRATION_PRIOR_RCOND": float(cfg.DIRECT_PRIOR_RCOND),
            "Q_INTEGRATION_METRIC": policy.Q_INTEGRATION_METRIC,
            "Q_INTEGRATION_TARGET_ACCEPT": policy.Q_INTEGRATION_TARGET_ACCEPT,
            "Q_INTEGRATION_ADAPT_MASS": policy.Q_INTEGRATION_ADAPT_MASS,
            "MCMC_SAMPLES": int(cfg.MCMC_SAMPLES),
            "MCMC_SEED": int(cfg.MCMC_SEED),
            "RESULTS_ROOT": output_root / "results",
            "INFERENCE_CACHE_ROOT": output_root / "inference_cache",
        }
        missing = object()
        original = {name: getattr(cfg, name, missing) for name in overrides}
        for name, value in overrides.items():
            setattr(cfg, name, value)
        try:
            yield package, cfg, fit, runner
        finally:
            for name, value in original.items():
                if value is missing:
                    delattr(cfg, name)
                else:
                    setattr(cfg, name, value)


def run_one(truth: str, data: str, q_key: str, mode: str, **kwargs):
    """Run one full-scale q-integration case from an existing source suite."""
    with activated(truth, data) as (_package, _cfg, _fit, runner):
        return runner.run_one(q_key, mode, **kwargs)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the full-scale explicit-q dense-NUTS closure variant.",
        epilog="Source-runner options such as --Q, --modes, --no-plots, and --fail-fast are forwarded.",
    )
    parser.add_argument("--truth", choices=("JAM", "NNPDF"), default="JAM")
    parser.add_argument("--data", choices=("union", "JAM", "NNPDF"), default="union")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="private root for results and inference caches")
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    with activated(args.truth, args.data, output_root=args.output_root) as (
        package, cfg, _fit, runner,
    ):
        if args.n_samples is not None:
            if args.n_samples < 1:
                parser.error("--n-samples must be positive")
            cfg.MCMC_SAMPLES = int(args.n_samples)
        if args.seed is not None:
            cfg.MCMC_SEED = int(args.seed)
        print(
            f"q-integration: source={package} metric={cfg.Q_INTEGRATION_METRIC} "
            f"target_accept={cfg.Q_INTEGRATION_TARGET_ACCEPT:.2f}", flush=True,
        )
        old_argv = sys.argv
        try:
            sys.argv = ["closure_full_q_integration.run_closure", *forwarded]
            runner.main()
        finally:
            sys.argv = old_argv
