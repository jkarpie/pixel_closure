#!/usr/bin/env python3
"""Fixed-prior hyperparameter scan for the small closure suites.

Stages (see ``campaign/prior_cells.py`` for the grids and ``prior_screen.py`` for the
no-sampling metrics)::

    expand   --stage s1 --out DIR                    -> DIR/cells_s1.json
    screen   --cells FILE --suite SUITE --out DIR    -> DIR/screen/<suite>__<cell>__<hash>.json
    select   --cells FILE --screen DIR --stage s2|s3|s4 --out FILE [--top N]
    fit      --cells FILE --suite SUITE --q Q --modes M... --out DIR [--sampler S]
    plan     --cells FILE --suite SUITE --q Q --modes M... [--sampler S...]   (dry run)
    report   --out DIR                               -> tables + figures

``fit`` runs every (cell, mode) in a fresh subprocess (``run-cell``), so a runaway fit is a
``timeout`` row instead of a hung scan and JAX memory never accumulates across cells.  Every
row carries the resolved prior and its ``cell_hash``; ``--skip-existing`` skips a row only
when the hash matches.  Nothing here edits a suite's ``config.py``.

``--sampler gaussian|nuts`` swaps the suite's VEGAS call for PIXEL's Gaussian or NUTS route
without editing ``fit.py``: ``run-cell`` rewrites the keyword arguments ``run_fit`` hands
to ``infer_mcmc`` (:func:`rewrite_infer_kwargs`) and applies that sampler's own gate to what
comes back (:func:`sampler_gate`).  ``plan`` prints the PIXEL route each request resolves to
without sampling anything.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import prior_cells as pc
from .proposal_width import parse_scales, patched_pilot_scales
from .saddle_strategy import SADDLE_STRATEGIES, patched_saddle_search

CAMPAIGN_ROOT = Path(__file__).resolve().parent
REPO = CAMPAIGN_ROOT.parent
SUITES = ("closure_JAM_truth_small", "closure_NNPDF_truth_small")
PHYSICS_MODES = ("lattice", "dis", "dy", "exp", "both")
_ESS_PATTERNS = {
    "vegas_ess_fraction": re.compile(r"VEGAS ESS fraction ([0-9.eE+-]+) is below"),
    "absolute_ess_fraction": re.compile(r"absolute outer ESS fraction ([0-9.eE+-]+) <"),
    "signed_ess_fraction": re.compile(r"signed outer ESS fraction ([0-9.eE+-]+) <"),
    "median_inner_ess_frac": re.compile(r"median inner ESS fraction ([0-9.eE+-]+) <"),
}

SAMPLERS = ("vegas", "gaussian", "nuts")
DEFAULT_TIMEOUT_S = {"vegas": 2400.0, "gaussian": 2400.0, "nuts": 10800.0}
#: ``run_fit``'s VEGAS-only keywords.  No Gaussian or NUTS route accepts them.
VEGAS_ONLY_OPTIONS = (
    "n_bins", "n_adapt_iterations", "n_eval_per_iteration", "alpha", "joint_grid",
    "min_adapt_phase_ess_frac", "max_extra_adapt_iterations", "n_adapt_inner_samples",
    "n_inner_samples", "covariance_inflation",
)
#: The weight-ESS trust threshold of ``pixel/guides/choosing_a_sampler.md``.
MIN_ESS_FRACTION = 0.10
_SADDLE_IMPLEMENTATIONS = ("normalization_reference_saddle_contour", "fixed_saddle_contour")
_CONTOUR_NUTS_IMPLEMENTATIONS = ("normalization_reference_contour_nuts", "fixed_contour_nuts",
                                 "joint_contour_nuts")
_TESTED_IMPLEMENTATIONS = {
    "gaussian": ("ordinary_gaussian", *_SADDLE_IMPLEMENTATIONS),
    "nuts": ("ordinary_nuts", *_CONTOUR_NUTS_IMPLEMENTATIONS),
}


class SamplerGateError(RuntimeError):
    """A Gaussian/NUTS result failed its ESS, saddle or chain gate; the row is ``gate_fail``."""


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-process temp name: concurrent scans sharing an --out write the same cell_json.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(path)


def _git(repo: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def provenance() -> dict:
    return {
        "pixel": _git(REPO.parent / "pixel"),
        "pixel_closure": _git(REPO),
        "env": {k: os.environ.get(k) for k in (
            "XLA_FLAGS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
            "PIXEL_KERNEL_WORKERS", "PIXEL_ASSEMBLY_WORKERS", "PIXEL_POSTERIOR_WORKERS")},
    }


def load_cells(path: Path) -> list[dict]:
    return [pc.validate(c) for c in json.loads(Path(path).read_text())["cells"]]


def ess_from_message(message: str) -> dict:
    out = {}
    for key, pattern in _ESS_PATTERNS.items():
        match = pattern.search(message or "")
        if match:
            out[key] = float(match.group(1))
    return out


# -- expand ---------------------------------------------------------------------


def cmd_expand(args) -> int:
    cfg = importlib.import_module(f"{SUITES[0]}.config")
    if args.stage != "s1":
        raise SystemExit("expand builds s1 only; later stages come from `select`")
    cells = pc.const_grid_s1(cfg) + pc.envelope_grid_s1(cfg)
    names = [c["name"] for c in cells]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate cell names in s1 grid")
    out = Path(args.out) / "cells_s1.json"
    _write_json(out, {"stage": "s1", "provenance": provenance(),
                      "cells": cells})
    counts = {p: sum(c["prior_form"] == p for c in cells) for p in pc.PRIOR_FORMS}
    print(f"[expand] {len(cells)} cells {counts} -> {out}")
    return 0


# -- screen ---------------------------------------------------------------------


def _screen_path(out: Path, suite: str, cell: dict) -> Path:
    return out / "screen" / f"{suite}__{cell['name']}__{pc.cell_hash(cell)}.json"


def cmd_screen(args) -> int:
    from .prior_screen import screen_cell

    cells = load_cells(args.cells)
    out = Path(args.out)
    started = time.perf_counter()
    for i, cell in enumerate(cells, start=1):
        path = _screen_path(out, args.suite, cell)
        if args.skip_existing and path.exists():
            continue
        row = screen_cell(args.suite, cell, q_key=args.q)
        row["provenance_run"] = provenance() if i == 1 else None
        _write_json(path, row)
        print(f"[screen {i}/{len(cells)}] {args.suite} {cell['name']} {row['status']} "
              f"info={row.get('dy_info_ratio_median', float('nan')):.3g} "
              f"cov={row.get('coverage_bulk_max_z_worst', float('nan')):.3g} "
              f"({row['elapsed_s']}s)", flush=True)
    print(f"[screen] done in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


def screen_table(screen_dir: Path) -> dict:
    """``{(cell_name, cell_hash): {suite: row}}`` from every screen row on disk."""
    table: dict = {}
    for path in sorted(Path(screen_dir).glob("*.json")):
        row = json.loads(path.read_text())
        table.setdefault((row["cell"], row["cell_hash"]), {})[row["suite"]] = row
    return table


def screen_score(rows_by_suite: dict) -> tuple | None:
    """Sort key for a screened cell over both suites, or ``None`` if it is disqualified.

    Disqualified: missing a suite, not factorable, or erroring.  Otherwise, in order:
    truth outside 3 prior sd anywhere in the bulk (a prior that cannot cover its truth),
    then the worst suite's median DY information ratio, then the worst prior-mean/data
    distance in log space.
    """
    if set(rows_by_suite) != set(SUITES):
        return None
    if any(r.get("status") != "ok" for r in rows_by_suite.values()):
        return None
    coverage = max(r["coverage_bulk_max_z_worst"] for r in rows_by_suite.values())
    info = max(r["dy_info_ratio_median"] for r in rows_by_suite.values())
    offset = max(abs(math.log10(max(r["dy_prior_mean_over_data_median"], 1e-300)))
                 for r in rows_by_suite.values())
    return (coverage > 3.0, math.log10(max(info, 1e-300)), offset)


# -- select ---------------------------------------------------------------------


def cmd_select(args) -> int:
    cells = {(c["name"], pc.cell_hash(c)): c for c in load_cells(args.cells)}
    chosen: list[dict] = []
    if args.stage in ("s2", "s3"):
        table = screen_table(Path(args.screen))
        for prior in pc.PRIOR_FORMS:
            scored = []
            for key, cell in cells.items():
                if cell["prior_form"] != prior or key not in table:
                    continue
                score = screen_score(table[key])
                if score is not None:
                    scored.append((score, cell))
            scored.sort(key=lambda item: item[0])
            baseline = "c_config" if prior == "const_logrbf" else "e_config"
            picks = [cell for _, cell in scored[: args.top]]
            if baseline not in {c["name"] for c in picks}:
                picks += [c for c in cells.values() if c["name"] == baseline]
            print(f"[select {args.stage}] {prior}: {len(scored)} qualified, keeping {len(picks)}")
            for score, cell in scored[: args.top]:
                print(f"    {cell['name']:<45} cov>3={score[0]} log10(info)={score[1]:.2f} "
                      f"|log10(mean/data)|={score[2]:.2f}")
            chosen += picks
        if args.stage == "s2":
            chosen = pc.refine_s2(chosen)
    elif args.stage == "s4":
        ranking = rank_fits(Path(args.rows), modes=("lattice", "dis"))
        for prior in pc.PRIOR_FORMS:
            ordered = [r for r in ranking if r["prior_form"] == prior]
            picks = [r["cell"] for r in ordered[: args.top]]
            baseline = "c_config" if prior == "const_logrbf" else "e_config"
            extra = [baseline] + [r["cell"] for r in ordered if r["cell"].startswith("ct_")][:1]
            names = list(dict.fromkeys(picks + extra))
            chosen += [c for c in cells.values() if c["name"] in names]
            print(f"[select s4] {prior}: {names}")
    _write_json(Path(args.out), {"stage": args.stage, "provenance": provenance(), "cells": chosen})
    print(f"[select] {len(chosen)} cells -> {args.out}")
    return 0


# -- fit ------------------------------------------------------------------------


def _row_path(out: Path, suite: str, q: str, mode: str, cell: dict, *, sampler: str) -> Path:
    return out / "rows" / f"{suite}__q{q}__{mode}__{cell['name']}__{sampler}__{pc.cell_hash(cell)}.json"


def default_auxiliary(sampler: str, has_bilinear: bool, requested: str | None = None) -> str:
    """The H-S policy the sampler test asks PIXEL for, unless one is given explicitly.

    NUTS must *sample* ``f`` (PIXEL refuses ``nuts`` + ``marginalize``); Gaussian asks for
    ``auto``, which PIXEL resolves to the saddle-contour route for bilinear data.
    """
    if requested:
        return requested
    return "sample" if sampler == "nuts" and has_bilinear else "auto"


def default_run_map(implementation: str, requested: str = "auto") -> bool:
    """MAP-centre only the ordinary Gaussian proposal, whose Laplace fit needs the mode."""
    if requested != "auto":
        return requested == "true"
    return implementation == "ordinary_gaussian"


def implementation_options(cfg, implementation: str, *, cache_stem: Path) -> dict:
    """Per-route sampler keywords, following the full suites' NUTS block in ``fit.py``.

    ``cache_stem`` is a per-(suite, Q, mode, cell, sampler) path prefix, so no two runs
    share a reference contour or Hessian metric.
    """
    nuts = {"n_warmup": cfg.NUTS_WARMUP, "step_size": cfg.NUTS_INITIAL_STEP_SIZE,
            "integrator": cfg.NUTS_INTEGRATOR, "max_tree_depth": cfg.NUTS_MAX_TREE_DEPTH,
            "adapt_mass": cfg.NUTS_ADAPT_MASS, "use_hessian_mass": True}
    reference = Path(f"{cache_stem}_reference.npz")
    hessian = Path(f"{cache_stem}_hessian_mass.npz")
    tol = cfg.CONTOUR_SADDLE_TOL
    table = {
        # Exact Laplace width: ``hessian_scale=2`` converts the Hessian of E^2 = -2 log L to a
        # covariance; ``covariance_inflation=1`` replaces PIXEL's default 3.
        "ordinary_gaussian": {"hessian_scale": 2.0, "covariance_inflation": 1.0},
        "normalization_reference_saddle_contour": {"saddle_tol": tol, "reference_cache_path": reference},
        "fixed_saddle_contour": {"saddle_tol": tol},
        "ordinary_nuts": nuts,
        "normalization_reference_contour_nuts": {**nuts, "saddle_tol": tol,
                                                 "reference_cache_path": reference,
                                                 "hessian_cache_path": hessian},
        "fixed_contour_nuts": {**nuts, "saddle_tol": tol, "hessian_cache_path": hessian},
        "joint_contour_nuts": {**nuts, "saddle_tol": tol},
    }
    if implementation not in table:
        raise ValueError(f"PIXEL route {implementation!r} is not part of the sampler test")
    return dict(table[implementation])


#: Pilot width grid for Gaussian runs when ``--proposal-scales`` is not given: exactly the
#: fitted Laplace width, neither PIXEL's 0.25/0.35-1 narrowing nor any inflation.
EXACT_LAPLACE_WIDTH = (1.0,)


def proposal_scales_for(sampler: str, text: str | None) -> tuple[float, ...] | None:
    """The saddle-contour pilot grid a run uses: explicit ``text``, else exact width for Gaussian."""
    scales = parse_scales(text)
    if scales is None and sampler == "gaussian":
        return EXACT_LAPLACE_WIDTH
    return scales


def rewrite_infer_kwargs(kw: dict, *, sampler: str, auxiliary: str, implementation: str,
                         run_map: bool, options: dict, extra: dict | None = None) -> dict:
    """Turn ``run_fit``'s VEGAS keywords into a Gaussian/NUTS ``infer_mcmc`` call.

    Keeps what every route shares (``n_samples``, ``seed``), drops the VEGAS-only
    options, sets the sampler/auxiliary/MAP choice, then layers the route's own options
    and finally any ``--sampler-options`` overrides.
    """
    if sampler not in _TESTED_IMPLEMENTATIONS:
        raise ValueError(f"no keyword rewrite for sampler {sampler!r}")
    if implementation not in _TESTED_IMPLEMENTATIONS[sampler]:
        raise ValueError(f"sampler {sampler!r} resolved to {implementation!r}, "
                         "which the sampler test does not run")
    new = {k: v for k, v in kw.items() if k not in VEGAS_ONLY_OPTIONS and k != "saddle_tol"}
    new.update(sampler=sampler, auxiliary=auxiliary, run_map=bool(run_map))
    new.update(options)
    new.update(extra or {})
    return new


def _moved_fraction(states) -> float | None:
    states = np.real(np.asarray(states))
    if states.ndim != 2 or states.shape[0] < 2:
        return None
    return float(np.mean(np.any(np.diff(states, axis=0) != 0.0, axis=1)))


def sampler_gate(samples, implementation: str) -> tuple[dict, list[str]]:
    """``(diagnostics, failures)`` for one Gaussian/NUTS result; empty failures means pass.

    Gaussian importance: weight ESS fraction.  Saddle contour: residual ESS fraction and a
    converged saddle.  NUTS: no divergences, a Hessian (not identity-fallback) metric and a
    reliable ``tau_int``; contour NUTS additionally a residual ESS fraction, a converged
    saddle and a chain that moved (``ess_frac`` reads at its ceiling on a frozen chain).
    """
    summary = dict(samples.summary()) if hasattr(samples, "summary") else {}
    meta = dict(getattr(samples, "metadata", None) or {})
    diag: dict = {"implementation": implementation}
    failures: list[str] = []

    def need_ess(value, label):
        if not (value is not None and np.isfinite(value) and value >= MIN_ESS_FRACTION):
            failures.append(f"{label} ESS fraction {value:.3g} < {MIN_ESS_FRACTION}"
                            if value is not None else f"{label} ESS fraction missing")

    def need_saddle():
        if not diag["saddle_converged"]:
            failures.append("H-S saddle did not converge")

    if implementation == "ordinary_gaussian":
        weights = np.asarray(samples.weights, dtype=float)
        diag.update(ess_fraction=float(samples.effective_sample_size) / samples.n_samples,
                    max_weight=float(np.max(weights)), n_invalid=int(samples.n_invalid))
        need_ess(diag["ess_fraction"], "Gaussian importance")
    elif implementation in _SADDLE_IMPLEMENTATIONS:
        saddle = samples.saddle
        diag.update(ess_frac=float(samples.ess_frac), saddle_converged=bool(saddle.converged),
                    saddle_n_iter=int(saddle.n_iter),
                    saddle_residual=float(np.max(np.abs(np.asarray(saddle.grad)))),
                    weight_magnitude_ess_fraction=summary.get("weight_magnitude_ess_fraction"),
                    phase_only_ess_fraction=summary.get("phase_only_ess_fraction"),
                    proposal_fit_success=meta.get("proposal_fit_success"),
                    proposal_scale=meta.get("proposal_scale"),
                    # PIXEL floors small proposal-precision eigenvalues, which narrows those
                    # directions below the fitted width; nonzero means the floor bound.
                    proposal_precision_eigenvalues_floored=meta.get(
                        "proposal_precision_eigenvalues_floored"))
        need_ess(diag["ess_frac"], "saddle-contour residual")
        need_saddle()
    elif implementation == "ordinary_nuts" or implementation in _CONTOUR_NUTS_IMPLEMENTATIONS:
        mass = meta.get("backend_metric_initialization", meta.get("mass_initialization"))
        diag.update(n_divergent=summary.get("n_divergent"), mass_initialization=mass,
                    tau_reliable=summary.get("tau_reliable"), ess_min=summary.get("ess_min"),
                    mean_accept_prob=summary.get("mean_accept_prob"),
                    n_max_depth=summary.get("n_max_depth"), step_size=summary.get("step_size"))
        if diag["n_divergent"] != 0:
            failures.append(f"NUTS divergences {diag['n_divergent']}")
        if mass in (None, "identity_fallback"):
            failures.append(f"NUTS mass initialization {mass!r}")
        if diag["tau_reliable"] is not True:
            failures.append("NUTS tau_int unreliable (chain shorter than 50 tau)")
        if implementation != "ordinary_nuts":
            moved = summary.get("moved_fraction")
            if moved is None:
                moved = _moved_fraction(samples.coordinates)
            diag.update(residual_ess_fraction=float(samples.ess_frac),
                        saddle_converged=bool(samples.saddle.converged), moved_fraction=moved,
                        branch_crossings=summary.get("branch_crossings"),
                        action_phase_jumps=summary.get("action_phase_jumps"),
                        phase_only_ess_fraction=summary.get("phase_only_ess_fraction"),
                        weight_magnitude_ess_fraction=summary.get("weight_magnitude_ess_fraction"))
            need_ess(diag["residual_ess_fraction"], "contour residual")
            need_saddle()
            if not (moved or 0.0) > 0.0:
                failures.append("contour chain never moved")
    else:
        raise ValueError(f"no gate for PIXEL route {implementation!r}")
    return diag, failures


def _vegas_diagnostics(samples) -> dict:
    diag = {}
    for name in ("effective_sample_size", "absolute_ess", "signed_ess", "n_samples"):
        if hasattr(samples, name):
            try:
                diag[name] = float(getattr(samples, name))
            except Exception:
                pass
    if hasattr(samples, "inner_ess_frac"):
        diag["median_inner_ess_frac"] = float(np.median(samples.inner_ess_frac))
    if diag.get("n_samples"):
        n = diag["n_samples"]
        for name in ("effective_sample_size", "absolute_ess", "signed_ess"):
            if name in diag:
                diag[name + "_fraction"] = diag[name] / n
    return diag


def classify_exception(exc: BaseException) -> str:
    message = str(exc)
    if isinstance(exc, MemoryError):
        # PIXEL's budget refusal names the "normalization-reference contour"; it is a
        # resource refusal before sampling, not a sampler-quality verdict.
        return "memory_refused"
    if isinstance(exc, (SamplerGateError, OverflowError)):
        return "gate_fail"
    if "ESS" in message or "saddle" in message or "normalization" in message:
        return "gate_fail"
    return "error"


def cmd_run_cell(args) -> int:
    """One (suite, cell, Q, mode, sampler) fit, in this process.  Always writes a row."""
    from pixel.infer import _resolve_mcmc_plan
    from .run_campaign import realized_constraints, save_posterior, _worst

    cell = pc.validate(json.loads(Path(args.cell_json).read_text()))
    out = Path(args.out)
    sampler = args.sampler
    extra_options = json.loads(args.sampler_options) if args.sampler_options else {}
    cfg = importlib.import_module(f"{args.suite}.config")
    runner = importlib.import_module(f"{args.suite}.run_closure")
    row_path = _row_path(out, args.suite, args.q, args.mode, cell, sampler=sampler)
    stem = f"{args.suite}__q{args.q}__{args.mode}__{cell['name']}__{sampler}"
    row = {"suite": args.suite, "q": args.q, "mode": args.mode, "cell": cell["name"],
           "cell_hash": pc.cell_hash(cell), "prior_form": cell["prior_form"], "sampler": sampler,
           "saddle_strategy": args.saddle_strategy,
           "proposal_scales": proposal_scales_for(sampler, args.proposal_scales),
           "sampler_options_override": extra_options,
           "resolved_prior": pc.resolved(cell), "provenance": provenance()}
    results_root = out / "results" / cell["name"] / sampler / args.suite
    started = time.perf_counter()
    captured: dict = {}
    original_fit = runner.fitmod.run_fit
    original_infer = runner.fitmod.infer_mcmc
    original_label = runner.fitmod._closure_sampler

    def capture_fit(*a, **kw):
        captured["result"] = original_fit(*a, **kw)
        return captured["result"]

    def capture_infer(model, **kw):
        if sampler == "vegas":
            samples = original_infer(model, **kw)
            captured["ess"] = _vegas_diagnostics(samples)
            return samples
        auxiliary = default_auxiliary(sampler, bool(model.has_bilinear), args.auxiliary)
        plan = _resolve_mcmc_plan(model, sampler=sampler, auxiliary=auxiliary)
        implementation = plan.resolved_implementation
        options = implementation_options(cfg, implementation,
                                         cache_stem=out / "inference_cache" / stem)
        new_kw = rewrite_infer_kwargs(kw, sampler=sampler, auxiliary=auxiliary,
                                      implementation=implementation,
                                      run_map=default_run_map(implementation, args.run_map),
                                      options=options, extra=extra_options)
        captured["plan"] = {"implementation": implementation,
                            "resolved_auxiliary": plan.resolved_auxiliary,
                            "n_ordinary": plan.n_ordinary, "n_normalization": plan.n_normalization,
                            "n_hubbard": plan.n_hubbard,
                            "kwargs": {k: (str(v) if isinstance(v, Path) else v)
                                       for k, v in new_kw.items()}}
        print(f"[run-cell] infer_mcmc {captured['plan']}", flush=True)
        samples = original_infer(model, **new_kw)
        diag, failures = sampler_gate(samples, implementation)
        captured["ess"] = diag
        if failures:
            raise SamplerGateError(f"{sampler} sampler gate (ESS/saddle/chain) failed: "
                                   + "; ".join(failures))
        return samples

    width_proxy = None
    try:
        with (pc.patched_prior(cfg, cell, results_root=results_root),
              patched_saddle_search(args.saddle_strategy),
              patched_pilot_scales(row["proposal_scales"]) as width_proxy):
            runner.fitmod.run_fit = capture_fit
            runner.fitmod.infer_mcmc = capture_infer
            if sampler != "vegas":
                runner.fitmod._closure_sampler = lambda model, mode=None: sampler
            try:
                summary = runner.run_one(args.q, args.mode, make_plots=not args.no_plots)
            finally:
                runner.fitmod.run_fit = original_fit
                runner.fitmod.infer_mcmc = original_infer
                runner.fitmod._closure_sampler = original_label
            constraints = realized_constraints(captured["result"]) if "result" in captured else None
            posterior = out / "posteriors" / f"{stem}.npz"
            save_posterior(captured["result"], runner.fitmod.load_truth(args.q), posterior)
        coverage = summary.get("coverage") or {}
        row.update({
            "status": "ok",
            "chi2": summary.get("chi2"),
            "n_data": summary.get("n_data"),
            "n_physical_data": summary.get("n_physical_data"),
            "chi2_per_physical": (summary.get("chi2") / summary["n_physical_data"]
                                  if summary.get("chi2") is not None and summary.get("n_physical_data") else None),
            "gate": captured.get("ess", {}),
            "plan": captured.get("plan"),
            "pulls": {f: {"rms": math.sqrt(c["pull_chi2_per_point"]),
                          "signed": c["mean_signed_pull"],
                          "frac_le_2": c.get("frac_pull_le_2", c.get("n_pull_le_2", 0) / max(c.get("n_points", 1), 1))}
                      for f, c in coverage.items()},
            "worst_abs_constraint_residual": _worst(constraints, "abs_residual"),
            "constraints": constraints,
            "posterior_npz": str(posterior.relative_to(out)),
            "summary": summary,
        })
    except Exception as exc:
        message = str(exc)
        row.update({"status": classify_exception(exc), "exception": type(exc).__name__,
                    "message": message, "plan": captured.get("plan"),
                    "gate": dict(captured.get("ess", {}), **ess_from_message(message)),
                    "traceback": traceback.format_exc(limit=8)})
    if width_proxy is not None:
        row["proposal_scale_grid_intercepted"] = width_proxy.intercepted
    row["elapsed_s"] = round(time.perf_counter() - started, 1)
    _write_json(row_path, row)
    print(f"[run-cell] {args.suite} q={args.q} {args.mode} {cell['name']} {sampler}: {row['status']} "
          f"({row['elapsed_s']} s)", flush=True)
    return 0


def _run_cell_command(args, cell_file: Path, mode: str) -> list[str]:
    command = [sys.executable, "-m", "campaign.prior_scan", "run-cell", "--suite", args.suite,
               "--cell-json", str(cell_file), "--q", args.q, "--mode", mode, "--out", str(args.out),
               "--sampler", args.sampler, "--run-map", args.run_map,
               "--saddle-strategy", args.saddle_strategy]
    if args.auxiliary:
        command += ["--auxiliary", args.auxiliary]
    if args.sampler_options:
        command += ["--sampler-options", args.sampler_options]
    if args.proposal_scales:
        command += ["--proposal-scales", args.proposal_scales]
    if args.no_plots:
        command.append("--no-plots")
    return command


def cmd_fit(args) -> int:
    cells = load_cells(args.cells)
    out = Path(args.out)
    tmp = out / "cell_json"
    tmp.mkdir(parents=True, exist_ok=True)
    timeout = args.timeout if args.timeout is not None else DEFAULT_TIMEOUT_S[args.sampler]
    jobs = [(cell, mode) for cell in cells for mode in args.modes]
    started = time.perf_counter()
    for i, (cell, mode) in enumerate(jobs, start=1):
        path = _row_path(out, args.suite, args.q, mode, cell, sampler=args.sampler)
        if args.skip_existing and path.exists():
            existing = json.loads(path.read_text())
            if existing.get("cell_hash") != pc.cell_hash(cell):
                raise SystemExit(f"{path} exists with a different cell_hash; refusing to mix cells")
            if existing.get("status") in ("ok", "gate_fail", "error") or not args.retry:
                continue
        cell_file = tmp / f"{cell['name']}__{pc.cell_hash(cell)}.json"
        _write_json(cell_file, cell)
        command = _run_cell_command(args, cell_file, mode)
        log = out / "logs" / f"{args.suite}__q{args.q}__{mode}__{cell['name']}__{args.sampler}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cell_started = time.perf_counter()
        try:
            with log.open("w") as handle:
                subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                               timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            _write_json(path, {"suite": args.suite, "q": args.q, "mode": mode, "cell": cell["name"],
                               "cell_hash": pc.cell_hash(cell), "prior_form": cell["prior_form"],
                               "sampler": args.sampler, "resolved_prior": pc.resolved(cell),
                               "status": "timeout", "elapsed_s": timeout})
        status = json.loads(path.read_text()).get("status") if path.exists() else "missing"
        print(f"[fit {i}/{len(jobs)}] {args.suite} q={args.q} {mode:<8} {cell['name']:<20} "
              f"{args.sampler:<8} {status} ({time.perf_counter() - cell_started:.0f} s)", flush=True)
    print(f"[fit] done in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


def cmd_plan(args) -> int:
    """Dry run: the PIXEL route and keywords each (cell, mode, sampler) would use; no sampling."""
    from pixel.infer import _resolve_mcmc_plan

    cfg = importlib.import_module(f"{args.suite}.config")
    fitmod = importlib.import_module(f"{args.suite}.fit")
    for cell in load_cells(args.cells):
        with pc.patched_prior(cfg, cell):
            for mode in args.modes:
                model = fitmod.build_analysis(args.q, mode)[0].compile()
                for sampler in args.samplers:
                    stem = f"{args.suite}__q{args.q}__{mode}__{cell['name']}__{sampler}"
                    try:
                        auxiliary = default_auxiliary(sampler, bool(model.has_bilinear), args.auxiliary)
                        if sampler == "vegas":
                            plan = _resolve_mcmc_plan(model, sampler=sampler,
                                                      auxiliary="marginalize" if model.has_bilinear else "auto")
                            detail = ""
                        else:
                            plan = _resolve_mcmc_plan(model, sampler=sampler, auxiliary=auxiliary)
                            impl = plan.resolved_implementation
                            options = implementation_options(cfg, impl, cache_stem=Path(args.out) / stem)
                            detail = (f" run_map={default_run_map(impl, args.run_map)} "
                                      f"options={sorted(options)}")
                        print(f"[plan] {args.suite} q={args.q} {cell['name']:<9} {mode:<21} {sampler:<8} "
                              f"-> {plan.resolved_implementation} aux={plan.resolved_auxiliary} "
                              f"(ord={plan.n_ordinary} norm={plan.n_normalization} "
                              f"hs={plan.n_hubbard}){detail}", flush=True)
                    except Exception as exc:
                        print(f"[plan] {args.suite} q={args.q} {cell['name']:<9} {mode:<21} {sampler:<8} "
                              f"REFUSED {type(exc).__name__}: {exc}", flush=True)
    return 0


# -- ranking and report ------------------------------------------------------------


def load_rows(rows_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(rows_dir).glob("*.json"))]


def rank_fits(rows_dir: Path, modes=PHYSICS_MODES, q: str | None = None) -> list[dict]:
    """Per-cell aggregate over suites x modes, sorted best first.

    Order: every gate passes on every (suite, mode) present; then the worst realized
    constraint residual is at or below 1e-4; then the smallest sum of per-field RMS pulls
    over the ``dis``/``exp``/``both`` rows present; then the smallest sum of |signed pulls|.
    Incomplete cells rank after complete ones.
    """
    by_cell: dict = {}
    for row in load_rows(rows_dir):
        if row["mode"] not in modes or (q is not None and row["q"] != q):
            continue
        by_cell.setdefault((row["cell"], row["cell_hash"]), []).append(row)
    ranking = []
    for (name, digest), rows in by_cell.items():
        expected = len(SUITES) * len(modes)
        passed = sum(r["status"] == "ok" for r in rows)
        pull_rows = [r for r in rows if r["status"] == "ok" and r["mode"] in ("dis", "exp", "both")]
        rms = sum(p["rms"] for r in pull_rows for p in r["pulls"].values())
        signed = sum(abs(p["signed"]) for r in pull_rows for p in r["pulls"].values())
        worst_constraint = max((r.get("worst_abs_constraint_residual") or 0.0) for r in rows if r["status"] == "ok") \
            if passed else float("inf")
        ranking.append({
            "cell": name, "cell_hash": digest, "prior_form": rows[0]["prior_form"],
            "n_rows": len(rows), "n_expected": expected, "n_pass": passed,
            "all_pass": passed == expected,
            "worst_abs_constraint_residual": worst_constraint,
            "sum_rms_pull": rms, "sum_abs_signed_pull": signed,
            "min_ess_fraction": min((v for r in rows for k, v in (r.get("gate") or {}).items()
                                     if k.endswith("fraction") or k.endswith("frac")), default=None),
        })
    ranking.sort(key=lambda r: (not r["all_pass"], r["n_expected"] - r["n_pass"],
                                r["worst_abs_constraint_residual"] > 1e-4,
                                r["sum_rms_pull"], r["sum_abs_signed_pull"]))
    return ranking


def cmd_report(args) -> int:
    from .prior_scan_report import write_report

    write_report(Path(args.out), q=args.q)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("expand"); p.add_argument("--stage", default="s1"); p.add_argument("--out", required=True)
    p = sub.add_parser("screen"); p.add_argument("--cells", required=True); p.add_argument("--suite", required=True)
    p.add_argument("--q", default="2"); p.add_argument("--out", required=True); p.add_argument("--skip-existing", action="store_true")
    p = sub.add_parser("select"); p.add_argument("--cells", required=True); p.add_argument("--screen")
    p.add_argument("--rows"); p.add_argument("--stage", required=True, choices=("s2", "s3", "s4"))
    p.add_argument("--top", type=int, default=10); p.add_argument("--out", required=True)
    def sampler_flags(p):
        p.add_argument("--auxiliary", default=None, choices=("auto", "sample", "marginalize"),
                       help="default: 'sample' for NUTS on bilinear data, else 'auto'")
        p.add_argument("--run-map", default="auto", choices=("auto", "true", "false"),
                       help="auto: MAP-centre the ordinary Gaussian proposal only")
        p.add_argument("--sampler-options", default=None,
                       help="JSON object layered over the per-route sampler options")
        p.add_argument("--saddle-strategy", default="pixel_default", choices=SADDLE_STRATEGIES,
                       help="experimental H-S saddle solver used only by this campaign run")
        p.add_argument("--proposal-scales", default=None,
                       help="comma list replacing PIXEL's 0.25/0.35..1 pilot width grid for the "
                            "normalization-reference saddle-contour proposal, e.g. '1,2,4,8'")

    p = sub.add_parser("fit"); p.add_argument("--cells", required=True); p.add_argument("--suite", required=True)
    p.add_argument("--q", default="2"); p.add_argument("--modes", nargs="+", default=list(PHYSICS_MODES))
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=float, default=None, help=f"seconds; default {DEFAULT_TIMEOUT_S}")
    p.add_argument("--skip-existing", action="store_true"); p.add_argument("--retry", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--sampler", default="vegas", choices=SAMPLERS); sampler_flags(p)
    p = sub.add_parser("run-cell"); p.add_argument("--suite", required=True); p.add_argument("--cell-json", required=True)
    p.add_argument("--q", required=True); p.add_argument("--mode", required=True); p.add_argument("--out", required=True)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--sampler", default="vegas", choices=SAMPLERS); sampler_flags(p)
    p = sub.add_parser("plan"); p.add_argument("--cells", required=True); p.add_argument("--suite", required=True)
    p.add_argument("--q", default="2"); p.add_argument("--modes", nargs="+", default=list(PHYSICS_MODES))
    p.add_argument("--samplers", nargs="+", default=list(SAMPLERS), choices=SAMPLERS)
    p.add_argument("--out", default="inference_cache"); sampler_flags(p)
    p = sub.add_parser("report"); p.add_argument("--out", required=True); p.add_argument("--q", default=None)
    args = parser.parse_args(argv)
    return {"expand": cmd_expand, "screen": cmd_screen, "select": cmd_select, "fit": cmd_fit,
            "run-cell": cmd_run_cell, "plan": cmd_plan, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
