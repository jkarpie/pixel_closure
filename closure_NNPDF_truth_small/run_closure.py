"""Run the closure fits and compare across original-scale truth members.

For every generated NNPDF truth member (original ``Q``) and every test mode
(``lattice``, ``exp``, ``both``) this runs the closure fit, writes a per-run
summary plus PDF-space and data-space reproduction figures, and finally
aggregates cross-``Q`` comparison plots -- the proof that the code behaves
consistently whether the truth was NNPDF read at ``mc``, ``1``, ``2``, ``3``,
``4``, or ``5 GeV``.

Run standalone::

    python -m closure_NNPDF_truth_small.run_closure                 # every generated Q, all modes
    python -m closure_NNPDF_truth_small.run_closure --Q 2            # one member
    python -m closure_NNPDF_truth_small.run_closure --modes both     # one mode
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

from pixel.util.progress import Stopwatch, format_duration

from . import config as cfg
from . import fit as fitmod


# -- discovery ---------------------------------------------------------------


def generated_qs() -> list[str]:
    """Original-Q keys that have been generated (truth.json present)."""
    return [q for q in cfg.TRUTH_Q_CHOICES
            if (cfg.truth_dir(q) / "truth.json").exists()]


# -- plotting ----------------------------------------------------------------


def _prepare_matplotlib():
    """Use a writable Matplotlib config/cache directory inside closure results."""
    mpl_dir = cfg.RESULTS_ROOT / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


PLOT_SUFFIXES = (".png", ".pdf")


def save_figure_both(fig, path, *, dpi=150) -> list[Path]:
    """Save ``fig`` as both PNG and PDF, using ``path`` as the primary stem."""
    path = Path(path)
    if path.suffix.lower() in PLOT_SUFFIXES:
        paths = [path]
        paths.extend(
            path.with_suffix(suffix)
            for suffix in PLOT_SUFFIXES
            if suffix != path.suffix.lower()
        )
    else:
        paths = [path.with_suffix(suffix) for suffix in PLOT_SUFFIXES]
    path.parent.mkdir(parents=True, exist_ok=True)
    for out in paths:
        fig.savefig(out, dpi=dpi)
    return paths


def _hatched_band(ax, x, mean, std, color, label):
    import matplotlib.colors as mcolors
    rgba = mcolors.to_rgba(color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.30, facecolor=rgba,
                    edgecolor="none", linewidth=0.0)
    ax.plot(x, mean, color=rgba, lw=1.4, label=label)


def hybrid_xscale(ax, *, x_min=1.0e-4, x_max=1.0, split=0.1):
    """PDF-style x axis: log-x below ``split``, linear-x above it (C1 join).

    The forward map is ``log10(x)`` for ``x <= split`` and a linear continuation
    with matched slope above it, so the two regions meet smoothly at ``split``.
    """
    import math
    import numpy as np

    ln10 = math.log(10.0)
    t0 = math.log10(split)

    def fwd(x):
        x = np.clip(np.asarray(x, dtype=float), 1.0e-300, None)
        return np.where(x <= split, np.log10(x), t0 + (x - split) / (split * ln10))

    def inv(t):
        t = np.asarray(t, dtype=float)
        return np.where(t <= t0, np.power(10.0, t), split + (t - t0) * (split * ln10))

    ax.set_xscale("function", functions=(fwd, inv))
    ax.set_xlim(x_min, x_max)
    lo = int(math.floor(math.log10(x_min)))
    log_ticks = [10.0 ** k for k in range(lo, int(round(t0)) + 1)]   # ... 1e-2, 1e-1
    lin_ticks = [t for t in (0.2, 0.4, 0.6, 0.8, 1.0) if t > split + 1e-9]
    ticks = log_ticks + lin_ticks
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [rf"$10^{{{int(round(math.log10(v)))}}}$" if v <= split + 1e-9 else f"{v:g}"
         for v in ticks]
    )
    ax.axvline(split, color="0.85", lw=0.6, zorder=0)


def reproduction_ylim(
    x_truth,
    y_truth,
    truth_std,
    x_posterior,
    posterior_mean,
    posterior_std,
    *,
    focus_x=0.2,
    pad_fraction=0.12,
):
    """Scale a reproduction panel to show both bands for ``x >= focus_x``.

    Small-x closure posteriors can be effectively unconstrained and orders of
    magnitude wider than the phenomenologically useful moderate/large-x region.
    Those excursions should be visibly clipped rather than flattening every
    curve.  The selected limits contain the complete truth and posterior
    one-sigma bands in the focus region, plus zero and modest headroom.
    """
    values = []
    for x, center, spread in (
        (x_truth, y_truth, truth_std),
        (x_posterior, posterior_mean, posterior_std),
    ):
        x = np.asarray(x, dtype=float).ravel()
        center = np.asarray(center, dtype=float).ravel()
        spread = np.asarray(spread, dtype=float).ravel()
        mask = (x >= float(focus_x)) & np.isfinite(center) & np.isfinite(spread)
        if np.any(mask):
            values.extend((center[mask] - spread[mask], center[mask] + spread[mask]))
    finite = np.concatenate(values) if values else np.array([], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -0.08, 1.08
    ymin = min(0.0, float(np.min(finite)))
    ymax = max(0.0, float(np.max(finite)))
    span = ymax - ymin
    pad = float(pad_fraction) * (span if span > 0.0 else max(abs(ymin), 1.0e-3))
    return ymin - pad, ymax + pad


def plot_reproduction(q_key, mode, marginal, truth, path):
    """3x3 grid: injected NNPDF truth vs recovered posterior as ``x * PDF``."""
    _prepare_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xn = np.asarray(truth["x_nodes"])
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, name in zip(axes.ravel(), cfg.ALL_FIELDS):
        x, mean, std = marginal[name]
        tcurve = np.asarray(truth["curves"][name])
        # The fields ARE the momentum densities q = x*f, so plotting them
        # directly already gives "x * PDF" -- do NOT multiply by x again.
        y_truth = tcurve
        # True error band: the NNPDF replica-ensemble spread of the truth field.
        t_std = np.asarray(truth.get("curve_std", {}).get(name, np.zeros_like(tcurve)))
        y_mean = mean
        y_std = std
        ax.fill_between(xn, y_truth - t_std, y_truth + t_std,
                        alpha=0.20, facecolor="0.4", edgecolor="none",
                        label="x * NNPDF truth $\\pm$ replica error")
        ax.plot(xn, y_truth, "k--", lw=1.8)
        _hatched_band(ax, x, y_mean, y_std, "tab:red", "x * posterior")
        ax.set_ylim(*reproduction_ylim(xn, y_truth, t_std, x, y_mean, y_std))
        hybrid_xscale(ax, x_min=max(cfg.X_MIN, 1e-4), x_max=1.0, split=0.1)
        ax.set_title(f"{name}  ({cfg.SECTOR_LABEL.get(name, name)})", fontsize=9)
        ax.set_ylabel("x * PDF", fontsize=8)
        ax.axhline(0.0, color="0.7", lw=0.5)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Closure reproduction  Q={cfg.TRUTH_Q_CHOICES[q_key]} "
                 f"(input Q0=mc)  mode={mode}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure_both(fig, path, dpi=150)
    plt.close(fig)


def _pull_chi2_per_point(row):
    """Return pull chi2 per point, with old-summary fallback if needed."""
    value = row.get("pull_chi2_per_point")
    if value is not None:
        return value
    mean_abs = row.get("mean_abs_pull")
    return None if mean_abs is None else mean_abs * mean_abs


def plot_comparison(summaries, path):
    """Cross-Q pull-chi2-per-point comparison per field/mode."""
    _prepare_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = cfg.TEST_MODES
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 5), squeeze=False)
    for ci, mode in enumerate(modes):
        ax = axes[0][ci]
        qs = [s["q_key"] for s in summaries if s["mode"] == mode]
        for name in cfg.ALL_FIELDS:
            ys = [
                _pull_chi2_per_point(s["coverage"][name])
                for s in summaries if s["mode"] == mode
            ]
            ax.plot(range(len(qs)), ys, "o-", ms=4, lw=1, label=name)
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([f"Q={cfg.TRUTH_Q_CHOICES[q]}" for q in qs],
                           rotation=30, ha="right", fontsize=7)
        ax.axhline(1.0, color="0.6", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"mode={mode}", fontsize=10)
        ax.set_ylabel("bulk pull $\\chi^2$ / point")
    axes[0][-1].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("Cross-Q closure comparison: pull chi2 per field", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure_both(fig, path, dpi=150)
    plt.close(fig)


# -- driver ------------------------------------------------------------------


def build_kernels_only(q_key, mode):
    """Populate every kernel cache needed by one closure case, without fitting."""
    watch = Stopwatch()
    analysis, _fields = fitmod.build_analysis(
        q_key, mode, use_kernel_cache=True
    )
    model = analysis.compile()
    return {
        "n_fields": len(model.fields),
        "n_datasets": len(model.datasets),
        "n_data": model.n_data,
        "runtime_seconds": float(watch.elapsed),
    }


def run_one(q_key, mode, *, make_plots=True, write_outputs=True, run_cache=None):
    watch = Stopwatch()
    truth = fitmod.load_truth(q_key)
    result = fitmod.run_fit(q_key, mode)
    t_fit = watch.lap()
    summary = fitmod.summarize(q_key, mode, result, truth)
    t_summary = watch.lap()
    out = cfg.results_dir(q_key)
    out.mkdir(parents=True, exist_ok=True)
    if write_outputs:
        (out / f"summary_{mode}.json").write_text(json.dumps(summary, indent=2))
    if run_cache is not None:
        run_cache.setdefault(mode, []).append((q_key, result["marginal"], truth))
    t_pdf_plot = t_data_plot = 0.0
    if make_plots:
        try:
            plot_reproduction(q_key, mode, result["marginal"], truth,
                              out / f"reproduction_{mode}.png")
        except Exception as exc:  # plotting is optional
            print(f"   [PDF-space plot skipped: {exc}]")
        t_pdf_plot = watch.lap()
        try:
            from . import plot_datasets
            data_path = out / f"dataset_reproduction_{mode}.png"
            records = plot_datasets.build_records_from_result(q_key, mode, result, truth)
            if records:
                plot_datasets.plot_records(records, data_path, q_key=q_key, mode=mode)
                if write_outputs:
                    plot_datasets.write_summary(records, data_path, q_key=q_key, mode=mode)
            else:
                print("   [data-space plot skipped: no physical datasets in this mode]")
        except Exception as exc:  # plotting is optional
            print(f"   [data-space plot skipped: {exc}]")
        t_data_plot = watch.lap()
    parts = [f"fit {format_duration(t_fit)}",
             f"summary {format_duration(t_summary)}"]
    if make_plots:
        parts.append(f"pdf-plot {format_duration(t_pdf_plot)}")
        parts.append(f"data-plot {format_duration(t_data_plot)}")
    parts.append(f"run total {format_duration(watch.elapsed)}")
    print("   [time] " + "  ".join(parts), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run NNPDF closure fits.")
    ap.add_argument("--Q", dest="q", choices=list(cfg.TRUTH_Q_CHOICES),
                    help="single original-Q member (default: all generated)")
    ap.add_argument("--modes", nargs="+", choices=cfg.TEST_MODES,
                    default=list(cfg.TEST_MODES))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--plots-only", action="store_true",
                    help="rewrite plots without changing JSON result summaries")
    ap.add_argument("--kernels-only", action="store_true",
                    help="build selected kernel caches and stop before fitting")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first failed fit instead of recording and continuing")
    args = ap.parse_args()
    if args.plots_only and args.no_plots:
        ap.error("--plots-only cannot be combined with --no-plots")
    if args.plots_only and args.kernels_only:
        ap.error("--plots-only cannot be combined with --kernels-only")

    qs = [args.q] if args.q else generated_qs()
    if not qs:
        raise SystemExit("no generated truth members; run closure_NNPDF_truth_small.generate first")

    if args.kernels_only:
        failures = []
        suite_watch = Stopwatch()
        built = 0
        for q_key in qs:
            for mode in args.modes:
                print(
                    f"== kernels  Q={cfg.TRUTH_Q_CHOICES[q_key]} "
                    f"({cfg.truth_label(q_key)})  mode={mode} =="
                )
                try:
                    row = build_kernels_only(q_key, mode)
                except Exception as exc:
                    failures.append((q_key, mode, exc))
                    print(
                        f"   [FAILED: {type(exc).__name__}: {exc}]",
                        flush=True,
                    )
                    if args.fail_fast:
                        raise
                    continue
                built += 1
                print(
                    f"   kernels ready: fields={row['n_fields']} "
                    f"datasets={row['n_datasets']} data={row['n_data']} "
                    f"in {format_duration(row['runtime_seconds'])}",
                    flush=True,
                )
        print(
            f"\nDone. {built} kernel case(s) in "
            f"{format_duration(suite_watch.elapsed)}. No fits were run."
        )
        if failures:
            raise SystemExit(1)
        return

    all_summaries = []
    failures = []
    ratio_runs = {mode: [] for mode in args.modes}
    suite_watch = Stopwatch()
    for q_key in qs:
        for mode in args.modes:
            print(f"== fit  Q={cfg.TRUTH_Q_CHOICES[q_key]} ({cfg.truth_label(q_key)})"
                  f"  mode={mode} ==")
            case_watch = Stopwatch()
            try:
                s = run_one(
                    q_key, mode, make_plots=not args.no_plots,
                    write_outputs=not args.plots_only,
                    run_cache=ratio_runs if not args.no_plots else None,
                )
            except Exception as exc:
                failure = {
                    "q_key": q_key,
                    "mode": mode,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "runtime_seconds": float(case_watch.elapsed),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(f"   [FAILED after {format_duration(case_watch.elapsed)}: "
                      f"{type(exc).__name__}: {exc}]", flush=True)
                if args.fail_fast:
                    raise
                continue
            pull_chi2 = {
                k: None if v["pull_chi2_per_point"] is None
                else round(v["pull_chi2_per_point"], 3)
                for k, v in s["coverage"].items()
            }
            n_le_2 = {
                k: f"{v['n_pull_le_2']}/{v['n_points']}"
                for k, v in s["coverage"].items()
            }
            components = s.get("chi2_components") or {}
            print(f"   chi2/n={s['chi2_per_data']:.3g}  n={s['n_data']}  "
                  f"components={components}  mean pull^2={pull_chi2}  "
                  f"|pull|<=2={n_le_2}")
            exp_nuis = s.get("exp_nuisances") or {}
            norms = {k: v["normalization"] for k, v in exp_nuis.items()
                     if "normalization" in v}
            if norms:
                # Worst by absolute miss; z is conditional on the theory (see
                # exp_nuisance_report) and is None when the norm was fitted.
                worst = max(norms.items(),
                            key=lambda kv: abs(kv[1]["pull_minus_true"]))
                w = worst[1]
                tail = ("" if w.get("pull_z") is None
                        else f" (z_cond={w['pull_z']:+.1f})")
                kind = "fitted" if w.get("fitted") else "marginalized"
                print(f"   DIS norms ({kind}): {len(norms)} tables, worst "
                      f"{worst[0]} beta_true={w['beta_true']:+.2f} "
                      f"got={w['pull']:+.3f}{tail}")
            corr = {k: v["correlated"] for k, v in exp_nuis.items()
                    if "correlated" in v}
            if corr:
                n_src = sum(c["n_sources"] for c in corr.values())
                chi2 = [c["pull_chi2_per_source"] for c in corr.values()
                        if c["pull_chi2_per_source"] is not None]
                frac2 = [c["frac_within_2sigma"] for c in corr.values()
                         if c["frac_within_2sigma"] is not None]
                print(f"   DIS correlated systematics: {n_src} sources over "
                      f"{len(corr)} tables, pull chi2/src="
                      f"{np.mean(chi2):.2f}, |pull|<=2sigma={np.mean(frac2)*100:.0f}%")
            for name, row in (s.get("dy_normalization") or {}).items():
                z, unc = row["pull_z"], row["uncertainty"]
                kind = "fitted" if row.get("fitted") else "marginalized"
                got = (f"{row['pull']:+.3f}" if unc is None
                       else f"{row['pull']:+.3f}+-{unc:.3f}")
                print(f"   DY norm {name} ({kind}): beta_true={row['beta_true']:+.2f} "
                      f"got={got} (expect {row['expected_pull']:+.3f}"
                      + (f", {z:+.1f} sigma)" if z is not None else ")"))
            all_summaries.append(s)

    comp_dir = cfg.RESULTS_ROOT / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        (comp_dir / "summaries.json").write_text(json.dumps(all_summaries, indent=2))
        (comp_dir / "failures.json").write_text(json.dumps(failures, indent=2))
        manifest = {
            "requested": len(qs) * len(args.modes),
            "succeeded": len(all_summaries),
            "failed": len(failures),
            "runtime_seconds": float(suite_watch.elapsed),
            "failures": [
                {k: row[k] for k in (
                    "q_key", "mode", "exception", "message", "runtime_seconds"
                )}
                for row in failures
            ],
        }
        (comp_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not args.no_plots and all_summaries:
        try:
            plot_comparison(all_summaries, comp_dir / "cross_Q_coverage.png")
        except Exception as exc:
            print(f"[comparison plot skipped: {exc}]")
        try:
            from . import plot_ratios
            for mode, runs in ratio_runs.items():
                if runs:
                    plot_ratios.plot_mode_from_runs(
                        mode, runs, comp_dir / f"ratio_grid_{mode}.png"
                    )
        except Exception as exc:
            print(f"[ratio-grid plot skipped: {exc}]")
    total = suite_watch.elapsed
    mean = total / len(all_summaries) if all_summaries else 0.0
    print(f"\nDone.  {len(all_summaries)} fits in {format_duration(total)} "
          f"(mean {format_duration(mean)} per fit).  "
          f"Results under {cfg.RESULTS_ROOT}")
    if failures:
        print(f"{len(failures)} fit(s) failed; details are in "
              f"{comp_dir / 'failures.json'}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
