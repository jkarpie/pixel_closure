#!/usr/bin/env python3
"""Small-suite closure campaign: every Q member x every mode x BOTH prior forms.

Reconstructs the driver behind ``campaign/full_2026-08-16/`` (144 cells), which did
not survive the move into this repo -- only its ``queue.txt`` and result rows did.

Why a driver exists at all: ``cfg.PRIOR_FORM`` is a module constant with no CLI or
environment override (``config.py:816``, dispatched at ``fit.py:232``), so the prior
axis cannot be swept from ``run_closure.py``'s argument parser. This sets it
in-process before each cell. The matching jitter needs no separate handling --
``fit._gp_prior`` reads ``cfg.GP_JITTER`` (1e-10) and ``fit._envelope_prior`` reads
``cfg.GP_ENVELOPE_JITTER`` (1e-2), each selected by ``PRIOR_FORM`` itself.

Output layout -- separable now, recombinable later:

    campaign/<name>/
        const_logrbf/<suite>/truthQ_*/...      <- per-prior result trees, so each
        beta_envelope/<suite>/truthQ_*/...        prior's plots stand alone
        rows/<suite>__<q>__<mode>__<prior>.json <- one flat row per cell

The per-prior trees are ordinary ``results/`` trees (``cfg.RESULTS_ROOT`` is
repointed), so every existing plotting path works inside one unchanged. The flat
``rows/`` directory carries the SAME filename and field schema as
``campaign/full_2026-08-16/rows/``, so old and new campaigns can be loaded together
and compared directly.

Usage:
    python -m campaign.run_campaign --suite closure_JAM_truth_small --name my_run
    python -m campaign.run_campaign --suite ... --priors const_logrbf --modes lattice
"""

from __future__ import annotations

import argparse
import importlib
import json
import time

import numpy as np
import traceback
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parent
PRIORS = ("const_logrbf", "beta_envelope")


def _row_path(rows_dir: Path, suite: str, q: str, mode: str, prior: str) -> Path:
    return rows_dir / f"{suite}__{q}__{mode}__{prior}.json"


def realized_constraints(result: dict) -> dict:
    """Contract each near-hard constraint's own row against the posterior.

    This is the "realized physics" check ``numerical_regularization.md`` calls the
    one that matters most, and ``fit.summarize`` does not compute it -- the driver
    that did was lost in the move, so it is rebuilt here from the same mechanism
    ``fit._dis_predictions`` (``fit.py:523``) uses: one
    ``model.posterior_predictive`` call, sliced per dataset through
    ``model._layout.datasets``.

    Stored **raw**, on purpose. The old rows carried a derived
    ``residual = |realized - target| / floor``; that identity reproduces exactly,
    and for ``cons_norm_*``/``cons_momentum`` the floor is simply ``|target|`` (the
    sum-rule value). For ``cons_endpoint_*`` the target is ~0 and the old floor was
    a per-field near-cancellation scale whose definition did not survive with the
    driver -- it is neither ``max|curve|`` nor the curve RMS (checked). Rather than
    guess it and silently publish a differently-normalized number under the same
    name, every row keeps ``target`` and ``realized``, from which ANY floor
    convention -- including the old one -- can be recomputed after the fact. The
    absolute residual is recorded too, since that is what compares directly against
    the ``CONSTRAINT_*_SIGMA = 1e-4`` standard deviation.
    """
    model = result["model"]
    prediction = np.asarray(model.posterior_predictive(result["vec_bar"])[0])
    out = {}
    for layout in model._layout.datasets:
        dataset = model.datasets[layout.dataset_index]
        if not dataset.name.startswith("cons_"):
            continue
        target = float(np.asarray(dataset.mean).ravel()[0])
        realized = float(prediction[layout.data_slice].ravel()[0])
        entry = {
            "target": target,
            "realized": realized,
            "abs_residual": abs(realized - target),
            "nominal_target": dataset.metadata.extras.get("nominal_target"),
        }
        # Only the SUM RULES get a relative residual, and `nominal_target` is what
        # separates them: it is 3.0 for `cons_norm_v`, 1.0 for `cons_momentum`, and
        # exactly 0.0 for every `cons_endpoint_*`/`cons_origin_*`.  Dividing an
        # endpoint by its own ~1e-6 target would manufacture an O(1) "residual" that
        # looks catastrophic and means nothing -- the endpoint's whole point is that
        # the target IS ~0, so only the absolute miss is interpretable there.
        # For the sum rules this floor is |target|, reproducing the old rows exactly.
        nominal = entry["nominal_target"]
        if nominal is not None and abs(float(nominal)) > 0.0 and abs(target) > 0.0:
            entry["floor"] = abs(target)
            entry["residual"] = abs(realized - target) / abs(target)
        out[dataset.name] = entry
    return out


def save_posterior(result: dict, truth: dict, path: Path) -> None:
    """Persist everything a plot or a re-analysis needs, so nothing needs a re-fit.

    The campaign's expensive product is the posterior, and until now it was thrown
    away: ``run_one`` surfaces it only through its optional ``run_cache`` argument,
    which the driver did not pass. Every downstream artefact that needs more than a
    scalar summary therefore required refitting -- most visibly
    ``plot_ratios.plot_mode_from_runs``, which wants ``(q_key, marginal, truth)``
    triples and so could not be built at all, and any change to plot limits, which
    forced a full re-fit to re-render.

    One ``.npz`` per cell, holding both halves of every comparison:

    * ``post_<field>_{x,mean,std}`` -- the posterior marginal on its own grid.
    * ``truth_x_nodes``, ``truth_<field>``, ``truth_std_<field>`` -- the injected
      truth and its replica spread, on the truth grid.

    That is exactly the input set of ``plot_reproduction`` and
    ``plot_ratios.ratio_curve``, so both can be regenerated offline, at any y-limit
    or styling, from the saved file alone. It costs ~30 KB per cell against fits
    that run 2-3 minutes each.
    """
    arrays: dict[str, np.ndarray] = {}
    for field, (x, mean, std) in result["marginal"].items():
        arrays[f"post_{field}_x"] = np.asarray(x, dtype=float)
        arrays[f"post_{field}_mean"] = np.asarray(mean, dtype=float)
        arrays[f"post_{field}_std"] = np.asarray(std, dtype=float)
    arrays["truth_x_nodes"] = np.asarray(truth["x_nodes"], dtype=float)
    for field, curve in truth["curves"].items():
        arrays[f"truth_{field}"] = np.asarray(curve, dtype=float)
    for field, spread in (truth.get("curve_std") or {}).items():
        arrays[f"truth_std_{field}"] = np.asarray(spread, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_posterior(path: Path) -> tuple[dict, dict]:
    """Inverse of :func:`save_posterior`: ``(marginal, truth)`` ready for the plotters.

    Returns the same shapes the live fit produced -- ``marginal[field] = (x, mean,
    std)`` and a ``truth`` dict with ``x_nodes``/``curves``/``curve_std`` -- so a
    caller can hand them straight to ``plot_reproduction`` or
    ``plot_ratios.plot_mode_from_runs`` without knowing they came off disk.
    """
    data = np.load(path)
    fields = sorted({
        key[len("post_"):-len("_mean")]
        for key in data.files
        if key.startswith("post_") and key.endswith("_mean")
    })
    marginal = {
        f: (data[f"post_{f}_x"], data[f"post_{f}_mean"], data[f"post_{f}_std"])
        for f in fields
    }
    truth = {
        "x_nodes": data["truth_x_nodes"],
        "curves": {f: data[f"truth_{f}"] for f in fields if f"truth_{f}" in data.files},
        "curve_std": {
            f: data[f"truth_std_{f}"] for f in fields if f"truth_std_{f}" in data.files
        },
    }
    return marginal, truth


def config_snapshot(cfg) -> dict:
    """The knobs that define a cell, recorded per run rather than assumed.

    Every one of these has moved at least once in this suite's history, and a row
    that does not carry them cannot be compared against a later row with any
    confidence -- the value has to be read back from whatever ``config.py`` happens
    to say today, which is exactly the assumption that makes an old campaign
    uninterpretable.
    """
    names = (
        "PRIOR_FORM", "RCOND", "GP_JITTER", "GP_ENVELOPE_JITTER", "GP_LENGTH_LOG",
        "CONSTRAINT_ENDPOINT_SIGMA", "CONSTRAINT_NORM_SIGMA",
        "CONSTRAINT_ORIGIN_SIGMA", "CONSTRAINT_MOMENTUM_SIGMA",
        "GRID_N", "GRID_SPACING", "X_MIN", "ELEMENT_TYPE", "ENDPOINT_X",
        "MCMC_SAMPLES", "MCMC_SEED", "SYNTHETIC_Z_MCMC_SAMPLES",
        "GP_AMPLITUDE_FREE", "GP_AMPLITUDE_FLOOR", "GP_AMPLITUDE_PRIOR_SIGMA",
        "T0_TOLERANCE", "T0_MAX_ITERATIONS", "N_HYPERPARAMS",
        "Q0", "NF", "ORDER", "ALPHAS_MZ", "VALENCE_NORMS",
        "GP_AMPLITUDES", "GP_ENVELOPE", "LOW_X_LINEAR_POWER",
    )
    out = {}
    for name in names:
        if not hasattr(cfg, name):
            continue
        value = getattr(cfg, name)
        try:
            json.dumps(value)
        except TypeError:
            value = repr(value)
        out[name] = value
    return out


def _worst(constraints: dict | None, key: str) -> float | None:
    if not constraints:
        return None
    values = [
        entry[key]
        for entry in constraints.values()
        if isinstance(entry, dict) and entry.get(key) is not None
    ]
    return max(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, help="e.g. closure_JAM_truth_small")
    ap.add_argument("--name", required=True, help="campaign directory name")
    ap.add_argument("--priors", nargs="+", default=list(PRIORS), choices=list(PRIORS))
    ap.add_argument("--q", nargs="+", default=None, help="default: every configured member")
    ap.add_argument("--modes", nargs="+", default=None, help="default: every cfg.TEST_MODES")
    ap.add_argument("--rcond", type=float, default=None,
                    help="override cfg.RCOND for every cell.  The shipped value is "
                         "1e-16, but campaign/full_2026-08-16 was run at 1e-12 -- and "
                         "the difference is not cosmetic: it moves the realized "
                         "endpoint residual by ~3 orders, i.e. across the 1e-4 bar. "
                         "Recorded per row, so the axis is never invisible again.")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-posteriors", action="store_true",
                    help="skip the per-cell posterior .npz.  They are ~30 KB each "
                         "against a 2-3 minute fit and are the ONLY thing that makes "
                         "ratio grids and any re-plot possible without refitting, so "
                         "this is almost never the right trade.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave cells whose row JSON already exists (resume a run)")
    args = ap.parse_args()

    cfg = importlib.import_module(f"{args.suite}.config")
    runner = importlib.import_module(f"{args.suite}.run_closure")

    q_keys = args.q or list(cfg.TRUTH_Q_CHOICES)
    modes = args.modes or list(cfg.TEST_MODES)

    base = CAMPAIGN_ROOT / args.name
    rows_dir = base / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    original_results_root = cfg.RESULTS_ROOT
    original_prior = cfg.PRIOR_FORM
    original_rcond = cfg.RCOND
    if args.rcond is not None:
        cfg.RCOND = args.rcond
        print(f"[campaign] RCOND override: {original_rcond:g} -> {cfg.RCOND:g}", flush=True)

    (base / "config_snapshot.json").write_text(json.dumps({
        "suite": args.suite,
        "priors": args.priors,
        "q_keys": q_keys,
        "modes": modes,
        "rcond_override": args.rcond,
        # PRIOR_FORM is swept per cell, so the snapshot records the module default
        # rather than implying a single value held for the whole run.
        "config": config_snapshot(cfg),
    }, indent=2))

    cells = [(p, q, m) for p in args.priors for q in q_keys for m in modes]
    print(f"[campaign] {args.suite}: {len(cells)} cells "
          f"({len(args.priors)} priors x {len(q_keys)} Q x {len(modes)} modes)",
          flush=True)

    started = time.perf_counter()
    n_ok = n_fail = n_skip = 0

    try:
        for i, (prior, q_key, mode) in enumerate(cells, start=1):
            row_path = _row_path(rows_dir, args.suite, q_key, mode, prior)
            if args.skip_existing and row_path.exists():
                n_skip += 1
                continue

            # The two knobs that define the cell.  RESULTS_ROOT is repointed so each
            # prior's summaries and plots land in their own tree instead of
            # overwriting one another under the package's own results/.
            # config.py validates PRIOR_FORM at import; assigning it here bypasses
            # that, and fit.gp_prior would only raise once it is next called --
            # mid-cell, after the kernels are built.  Re-check at the assignment so
            # a bad value fails before any work is done.
            if prior not in PRIORS:
                raise ValueError(
                    f"unknown prior form {prior!r}; expected one of {list(PRIORS)}"
                )
            cfg.PRIOR_FORM = prior
            cfg.RESULTS_ROOT = base / prior / args.suite

            print(f"[{i}/{len(cells)}] {args.suite} Q={q_key} mode={mode} "
                  f"prior={prior}", flush=True)
            cell_started = time.perf_counter()
            row = {
                "suite": args.suite,
                "q": q_key,
                "mode": mode,
                "prior": prior,
                "jitter": (cfg.GP_ENVELOPE_JITTER if prior == "beta_envelope"
                           else cfg.GP_JITTER),
                "rcond": cfg.RCOND,
                "constraint_sigma": cfg.CONSTRAINT_ENDPOINT_SIGMA,
            }
            try:
                # run_one is reused verbatim for the fit/summary/plots, but it does
                # not hand back the fitted model, so the realized-constraint pass
                # needs the result object.  Capture it as run_one produces it rather
                # than fitting twice.
                captured = {}
                original_run_fit = runner.fitmod.run_fit

                def _capture(*a, __fn=original_run_fit, **kw):
                    captured["result"] = __fn(*a, **kw)
                    return captured["result"]

                runner.fitmod.run_fit = _capture
                try:
                    summary = runner.run_one(q_key, mode, make_plots=not args.no_plots)
                finally:
                    runner.fitmod.run_fit = original_run_fit
            except Exception as exc:
                row.update({
                    "status": "fail",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                    "elapsed_s": round(time.perf_counter() - cell_started, 1),
                })
                n_fail += 1
                print(f"   [FAILED] {type(exc).__name__}: {exc}", flush=True)
            else:
                constraints = None
                posterior_path = None
                if "result" in captured:
                    try:
                        constraints = realized_constraints(captured["result"])
                    except Exception as exc:  # never lose a good fit to the audit
                        print(f"   [constraint audit skipped: {exc}]", flush=True)
                    if not args.no_posteriors:
                        try:
                            posterior_path = (
                                base / "posteriors"
                                / f"{args.suite}__{q_key}__{mode}__{prior}.npz"
                            )
                            save_posterior(
                                captured["result"],
                                runner.fitmod.load_truth(q_key),
                                posterior_path,
                            )
                            posterior_path = str(
                                posterior_path.relative_to(CAMPAIGN_ROOT)
                            )
                        except Exception as exc:
                            print(f"   [posterior save skipped: {exc}]", flush=True)
                            posterior_path = None
                row.update({
                    "status": "ok",
                    "elapsed_s": round(time.perf_counter() - cell_started, 1),
                    "chi2": summary.get("chi2"),
                    "chi2_components": summary.get("chi2_components"),
                    "n_data": summary.get("n_data"),
                    "ess": summary.get("effective_sample_size"),
                    "timing_seconds": summary.get("timing_seconds"),
                    "coverage": summary.get("coverage"),
                    "constraints": constraints,
                    "worst_abs_constraint_residual": _worst(constraints, "abs_residual"),
                    "worst_constraint_residual": _worst(constraints, "residual"),
                    "posterior_npz": posterior_path,
                    "summary": summary,
                })
                n_ok += 1
            row_path.write_text(json.dumps(row, indent=2))
    finally:
        cfg.PRIOR_FORM = original_prior
        cfg.RESULTS_ROOT = original_results_root
        cfg.RCOND = original_rcond

    elapsed = time.perf_counter() - started
    print(f"\n[campaign] {args.suite}: {n_ok} ok, {n_fail} failed, {n_skip} skipped "
          f"in {elapsed/60:.1f} min -> {base}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
