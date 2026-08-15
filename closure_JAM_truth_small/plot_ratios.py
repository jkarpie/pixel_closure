"""Cross-Q ratio grids: posterior / truth per flavor, all input Qs overlaid.

For each test mode (``lattice``, ``exp``, ``both``) this makes one 3x3 grid --
one panel per flavor -- where every generated truth member ``Q`` is drawn on top
of the others as the ratio ``posterior_mean(Q) / truth(Q)`` (a light band shows
``+/- posterior_std / |truth|``).  A perfect reproduction sits on the ``ratio = 1``
line, so the panels show at a glance how well each input scale is recovered and
where the flavors agree or drift.

Run standalone::

    python -m closure_JAM_truth_small.plot_ratios                    # every generated Q, all modes
    python -m closure_JAM_truth_small.plot_ratios --modes both       # one mode
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config as cfg
from . import fit as fitmod
from .run_closure import (
    _prepare_matplotlib,
    generated_qs,
    hybrid_xscale,
    save_figure_both,
)

#: Points where |truth| falls below this fraction of its peak are masked -- the
#: ratio is meaningless through a zero crossing (t3, v3, ...).
TRUTH_FLOOR_FRAC = 0.05
#: Only draw the band where the relative posterior width is modest; beyond this it
#: would span the whole panel and just muddy the overlay.
BAND_REL_CAP = 0.75
#: Panel y-range for the ratio.
Y_LO, Y_HI = 0.0, 2.0


def ratio_curve(x, mean, std, xn, tcurve):
    """Return ``(ratio, lo, hi)`` = posterior/truth with a +/- std/|truth| band.

    ``truth`` is interpolated onto the posterior grid ``x``.  The central ratio is
    masked through truth zero crossings; the band is additionally dropped where the
    relative width is large and clipped to the panel so it stays readable.
    """
    truth = np.interp(x, xn, tcurve)
    peak = float(np.nanmax(np.abs(tcurve))) or 1.0
    good = np.abs(truth) > TRUTH_FLOOR_FRAC * peak
    rel = np.where(good, std / np.abs(truth), np.inf)
    ratio = np.where(good, mean / truth, np.nan)
    band_ok = good & (rel <= BAND_REL_CAP)
    lo = np.where(band_ok, np.clip(ratio - rel, Y_LO, Y_HI), np.nan)
    hi = np.where(band_ok, np.clip(ratio + rel, Y_LO, Y_HI), np.nan)
    return ratio, lo, hi


def plot_mode_from_runs(mode: str, runs, path: Path):
    """Write the ratio grid for one mode from ``(q, marginal, truth)`` runs."""
    _prepare_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_keys = [q for q, _, _ in runs]
    cmap = plt.get_cmap("viridis")
    colors = {q: cmap(i / max(len(q_keys) - 1, 1)) for i, q in enumerate(q_keys)}

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, name in zip(axes.ravel(), cfg.ALL_FIELDS):
        for q, marginal, truth in runs:
            x, mean, std = marginal[name]
            xn = np.asarray(truth["x_nodes"], dtype=float)
            tcurve = np.asarray(truth["curves"][name], dtype=float)
            ratio, lo, hi = ratio_curve(x, mean, std, xn, tcurve)
            c = colors[q]
            ax.fill_between(x, lo, hi, color=c, alpha=0.10, linewidth=0.0)
            ax.plot(x, ratio, color=c, lw=1.4,
                    label=f"fit Q={cfg.TRUTH_Q_CHOICES[q]}")
        ax.axhline(1.0, color="0.35", ls="--", lw=0.9, zorder=0)
        ax.set_ylim(Y_LO, Y_HI)
        hybrid_xscale(ax, x_min=max(cfg.X_MIN, 1e-4), x_max=1.0, split=0.1)
        ax.set_title(f"{name}  ({cfg.SECTOR_LABEL.get(name, name)})", fontsize=9)
        ax.set_ylabel("fit / true PDF mean", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle(f"Closure reproduction ratio (fit / true PDF mean)  mode={mode}  "
                 f"-- input Q0=mc, truths read at Q in "
                 f"{[cfg.TRUTH_Q_CHOICES[q] for q in q_keys]} GeV", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure_both(fig, path, dpi=150)
    plt.close(fig)
    return path


def plot_mode(mode: str, q_keys, path: Path):
    """Fit every Q and write the ratio grid for one mode across ``q_keys``."""
    runs = []
    for q in q_keys:
        res = fitmod.run_fit(q, mode)
        runs.append((q, res["marginal"], fitmod.load_truth(q)))
    return plot_mode_from_runs(mode, runs, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-Q posterior/truth ratio grids.")
    ap.add_argument("--modes", nargs="+", choices=cfg.TEST_MODES,
                    default=list(cfg.TEST_MODES))
    ap.add_argument("--Q", dest="qs", nargs="+", choices=list(cfg.TRUTH_Q_CHOICES),
                    help="subset of Q members (default: all generated)")
    args = ap.parse_args()

    q_keys = args.qs or generated_qs()
    if not q_keys:
        raise SystemExit("no generated truth members; run closure_JAM_truth_small.generate first")

    out_dir = cfg.RESULTS_ROOT / "comparison"
    for mode in args.modes:
        p = plot_mode(mode, q_keys, out_dir / f"ratio_grid_{mode}.png")
        print(f"wrote {p} (+ .pdf)")


if __name__ == "__main__":
    main()
