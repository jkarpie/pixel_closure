"""Runtime isolation for the two small-closure contour variants.

The JAM and NNPDF small suites deliberately contain identical inference
plumbing but different stored truths.  Copying their generated data and kernel
caches would duplicate roughly two gigabytes without changing a calculation.
This module instead applies a complete, process-local configuration overlay
while a variant is running.  Inputs and kernel matrices stay shared; posterior
outputs, contour references, and Hessian metrics are variant-specific.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from threading import RLock


REPO = Path(__file__).resolve().parent
TRUTH_SUITES = {
    "JAM": "closure_JAM_truth_small",
    "NNPDF": "closure_NNPDF_truth_small",
}
GP_COVARIANCE_JITTER = 1.0e-6
FLOW_TIME = 0.2
FLOW_STEPS = 8
FLOW_REFERENCE_BUDGET_BYTES = 110 * 1024**3
HMC_INITIAL_STEP_SIZE = 0.05
HMC_N_LEAPFROG = 8
HMC_WARMUP = 100
HMC_TARGET_ACCEPT = 0.8
Q_INTEGRATION_SAMPLER = "nuts"
Q_INTEGRATION_PRIOR_RCOND = 0.0
Q_INTEGRATION_METRIC = "prior_whitened"
Q_INTEGRATION_WARMUP = 100
Q_INTEGRATION_INITIAL_STEP_SIZE = 0.1
Q_INTEGRATION_MAX_TREE_DEPTH = 6
Q_INTEGRATION_TARGET_ACCEPT = 0.8
Q_INTEGRATION_ADAPT_MASS = False


@dataclass(frozen=True)
class VariantSpec:
    """One fully specified small-closure contour policy."""

    name: str
    flow_time: float
    affine_reference_contour: bool
    flow_from_affine_reference: bool
    reference_budget_bytes: int | None


FLOWED = VariantSpec(
    name="flowed",
    flow_time=FLOW_TIME,
    affine_reference_contour=False,
    # Flow the real H-S plane itself.  The affine saddle contour remains a
    # separate T=0 control and is not used as the base map for this variant.
    flow_from_affine_reference=False,
    # This is only the contour implementation's conservative admission guard,
    # not a claim about host RAM.  Its projection is about 99.6 GiB for the
    # 1,314-coefficient Q2 cases while measured resident memory is ~12.7 GiB.
    # The campaign scheduler separately enforces the actual 32-GiB host policy.
    reference_budget_bytes=FLOW_REFERENCE_BUDGET_BYTES,
)

AFFINE = VariantSpec(
    name="affine",
    # T=0 is the affine steepest-descent contour for fixed/joint H-S routes.
    # The normalization-reference route additionally receives the explicit
    # affine-only switch below.
    flow_time=0.0,
    affine_reference_contour=True,
    flow_from_affine_reference=False,
    reference_budget_bytes=None,
)

_VARIANT_LOCK = RLock()


def _modules(truth: str):
    try:
        package = TRUTH_SUITES[truth.upper()]
    except KeyError as exc:
        raise ValueError(f"truth must be one of {tuple(TRUTH_SUITES)}, got {truth!r}") from exc
    cfg = importlib.import_module(f"{package}.config")
    fit = importlib.import_module(f"{package}.fit")
    runner = importlib.import_module(f"{package}.run_closure")
    return package, cfg, fit, runner


@contextmanager
def activated(spec: VariantSpec, truth: str, *, sampler: str = "nuts"):
    """Activate ``spec`` for one source suite and restore it on exit.

    The underlying suite modules read their shared ``config`` module at call
    time.  A lock makes the overlay safe against two variants being entered in
    the same process concurrently.  Separate CLI processes remain independent.
    """

    with _VARIANT_LOCK:
        if sampler not in ("nuts", "hmc"):
            raise ValueError("variant sampler must be 'nuts' or 'hmc'")
        package, cfg, fit, runner = _modules(truth)
        output_root = (
            REPO / f"closure_small_{spec.name}" / "outputs" / truth.lower() / sampler
        )
        overrides = {
            "GP_JITTER": GP_COVARIANCE_JITTER,
            "GP_ENVELOPE_JITTER": GP_COVARIANCE_JITTER,
            "SMALL_CLOSURE_SAMPLER": sampler,
            # Experimental scans retain failed diagnostic cases.  The source
            # suites remain fail-fast by default; these records carry an
            # explicit sampling_quality warning in their JSON summaries.
            "SMALL_CLOSURE_REJECT_UNRELIABLE_SAMPLES": False,
            "CONTOUR_FLOW_TIME": spec.flow_time,
            "CONTOUR_N_FLOW_STEPS": FLOW_STEPS,
            "CONTOUR_NUTS_AFFINE_REFERENCE": spec.affine_reference_contour,
            "CONTOUR_NUTS_FLOW_FROM_AFFINE_REFERENCE": spec.flow_from_affine_reference,
            "CONTOUR_NUTS_REFERENCE_BUDGET_BYTES": spec.reference_budget_bytes,
            "RESULTS_ROOT": output_root / "results",
            "INFERENCE_CACHE_ROOT": output_root / "inference_cache",
            # A Hessian metric makes this dimensionless step size meaningful
            # across datatypes.  Dual averaging during warmup supplies the
            # final case-specific correction rather than betting the scan on a
            # single unadapted value.
            "HMC_WARMUP": HMC_WARMUP,
            "HMC_STEP_SIZE": HMC_INITIAL_STEP_SIZE,
            "HMC_N_LEAPFROG": HMC_N_LEAPFROG,
            "HMC_INTEGRATOR": "leapfrog",
            "HMC_USE_HESSIAN_MASS": True,
            "HMC_ADAPT_STEP_SIZE": True,
            "HMC_ADAPT_MASS": False,
            "HMC_TARGET_ACCEPT": HMC_TARGET_ACCEPT,
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


def run_one(
    spec: VariantSpec,
    truth: str,
    q_key: str,
    mode: str,
    *,
    sampler: str = "nuts",
    **kwargs,
):
    """Run one variant closure case using the source suite's stored data."""

    with activated(spec, truth, sampler=sampler) as (_package, _cfg, _fit, runner):
        return runner.run_one(q_key, mode, **kwargs)


def run_cli(spec: VariantSpec, argv=None) -> None:
    """Run the original small-suite CLI under a variant overlay."""

    parser = argparse.ArgumentParser(
        description=(
            f"Run the {spec.name} small-closure copy. Remaining arguments are "
            "forwarded to each truth suite's run_closure CLI."
        )
    )
    parser.add_argument(
        "--truth", nargs="+", choices=tuple(TRUTH_SUITES),
        default=list(TRUTH_SUITES), help="truth suites to run; default: both",
    )
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    old_argv = sys.argv
    try:
        for truth in args.truth:
            with activated(spec, truth) as (package, _cfg, _fit, runner):
                print(
                    f"== {spec.name} contour copy: {truth} "
                    f"(source {package}, GP jitter={GP_COVARIANCE_JITTER:g}) ==",
                    flush=True,
                )
                sys.argv = [f"closure_small_{spec.name}.run_closure", *forwarded]
                runner.main()
    finally:
        sys.argv = old_argv


@contextmanager
def q_integration_activated(truth: str):
    """Activate the explicit retained-field sampler for one small truth suite.

    This overlay changes only inference policy and output locations.  The
    generated closure data and kernel caches remain owned by the JAM/NNPDF
    source suites, exactly as they do for the contour variants above.
    """

    with _VARIANT_LOCK:
        package, cfg, fit, runner = _modules(truth)
        output_root = (
            REPO / "closure_small_q_integration" / "outputs" / truth.lower()
        )
        overrides = {
            "GP_JITTER": GP_COVARIANCE_JITTER,
            "GP_ENVELOPE_JITTER": GP_COVARIANCE_JITTER,
            "SMALL_CLOSURE_SAMPLER": "q",
            "SMALL_CLOSURE_REJECT_UNRELIABLE_SAMPLES": False,
            "Q_INTEGRATION_SAMPLER": Q_INTEGRATION_SAMPLER,
            # Zero retains every numerically positive prior mode.  This makes
            # the q-integrated variant compare the same GP measure rather than
            # quietly turning the rank cutoff into an inference approximation.
            "Q_INTEGRATION_PRIOR_RCOND": Q_INTEGRATION_PRIOR_RCOND,
            "Q_INTEGRATION_METRIC": Q_INTEGRATION_METRIC,
            "Q_INTEGRATION_WARMUP": Q_INTEGRATION_WARMUP,
            "Q_INTEGRATION_INITIAL_STEP_SIZE": Q_INTEGRATION_INITIAL_STEP_SIZE,
            "Q_INTEGRATION_MAX_TREE_DEPTH": Q_INTEGRATION_MAX_TREE_DEPTH,
            "Q_INTEGRATION_TARGET_ACCEPT": Q_INTEGRATION_TARGET_ACCEPT,
            "Q_INTEGRATION_ADAPT_MASS": Q_INTEGRATION_ADAPT_MASS,
            # Included even at their source values so CLI overrides made while
            # the context is active are restored on exit.
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


def run_q_integration_one(truth: str, q_key: str, mode: str, **kwargs):
    """Run one explicit-q closure case using a source suite's stored data."""

    with q_integration_activated(truth) as (_package, _cfg, _fit, runner):
        return runner.run_one(q_key, mode, **kwargs)


def run_q_integration_cli(argv=None) -> None:
    """Run the small-suite CLI with explicit q integration for both truths."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the q-integration small-closure copy. Remaining arguments "
            "are forwarded to each truth suite's run_closure CLI."
        )
    )
    parser.add_argument(
        "--truth", nargs="+", choices=tuple(TRUTH_SUITES),
        default=list(TRUTH_SUITES), help="truth suites to run; default: both",
    )
    parser.add_argument(
        "--n-samples", type=int,
        help="override the source suite's production draw count",
    )
    parser.add_argument(
        "--seed", type=int,
        help="override the q-integration sampling seed",
    )
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    old_argv = sys.argv
    try:
        for truth in args.truth:
            with q_integration_activated(truth) as (package, _cfg, _fit, runner):
                if args.n_samples is not None:
                    _cfg.MCMC_SAMPLES = int(args.n_samples)
                if args.seed is not None:
                    _cfg.MCMC_SEED = int(args.seed)
                print(
                    f"== q-integration copy: {truth} "
                    f"(source {package}, GP jitter={GP_COVARIANCE_JITTER:g}) ==",
                    flush=True,
                )
                sys.argv = ["closure_small_q_integration.run_closure", *forwarded]
                runner.main()
    finally:
        sys.argv = old_argv
