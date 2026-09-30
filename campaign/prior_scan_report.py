"""Tables and figures for a prior scan directory (``campaign/prior_scan.py``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import prior_scan as ps

MODES = ps.PHYSICS_MODES + ("synthetic_z_plumbing",)
STATUS_COLORS = {"ok": "#4c9a5a", "gate_fail": "#e07b39", "timeout": "#7a4ea3", "error": "#b83b3b",
                 "missing": "#cccccc"}


def _matplotlib():
    from closure_JAM_truth_small.run_closure import _prepare_matplotlib, save_figure_both

    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    return plt, save_figure_both


def screen_figure(out: Path) -> Path | None:
    table = ps.screen_table(out / "screen")
    if not table:
        return None
    plt, save = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, prior in zip(axes, ("const_logrbf", "beta_envelope")):
        xs, ys, colors, labels = [], [], [], []
        for (name, _), rows in table.items():
            if not rows or next(iter(rows.values()))["prior_form"] != prior:
                continue
            ok = set(rows) == set(ps.SUITES) and all(r.get("status") == "ok" for r in rows.values())
            if not ok:
                continue
            xs.append(max(math.log10(max(r["dy_info_ratio_median"], 1e-300)) for r in rows.values()))
            ys.append(max(r["coverage_bulk_max_z_worst"] for r in rows.values()))
            colors.append("tab:red" if name.endswith("config") else "tab:blue")
            labels.append(name)
        ax.scatter(xs, ys, c=colors, s=14, alpha=0.8)
        for x, y, label in zip(xs, ys, labels):
            if label.endswith("config"):
                ax.annotate("shipped", (x, y), textcoords="offset points", xytext=(4, 4), color="tab:red")
        ax.axhline(3.0, color="0.5", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("log10 median DY prior-predictive sd / data sd (worst suite)")
        ax.set_ylabel("max bulk |truth - prior mean| / prior sd (worst suite, field)")
        ax.set_title(f"screen, prior = {prior}  ({len(xs)} factorable cells)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "figures" / "screen_scatter"
    path.parent.mkdir(parents=True, exist_ok=True)
    save(fig, str(path))
    plt.close(fig)
    return path


def status_figure(out: Path, q: str | None, cells_order: list[str]) -> Path | None:
    rows = ps.load_rows(out / "rows")
    if q is not None:
        rows = [r for r in rows if r["q"] == q]
    if not rows:
        return None
    plt, save = _matplotlib()
    from matplotlib.patches import Rectangle

    columns = [(s, m) for s in ps.SUITES for m in MODES]
    names = [c for c in cells_order if any(r["cell"] == c for r in rows)]
    lookup = {(r["cell"], r["suite"], r["mode"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(1.0 * len(columns) + 3, 0.34 * len(names) + 2))
    for i, name in enumerate(names):
        for j, (suite, mode) in enumerate(columns):
            row = lookup.get((name, suite, mode))
            status = row["status"] if row else "missing"
            ax.add_patch(Rectangle((j, len(names) - 1 - i), 1, 1, facecolor=STATUS_COLORS.get(status, "#999"),
                                   edgecolor="white"))
            if row and status == "ok" and row.get("chi2_per_physical") is not None:
                ax.text(j + 0.5, len(names) - 0.5 - i, f"{row['chi2_per_physical']:.2g}", ha="center",
                        va="center", fontsize=6, color="white")
    ax.set_xlim(0, len(columns)); ax.set_ylim(0, len(names))
    ax.set_xticks([j + 0.5 for j in range(len(columns))])
    ax.set_xticklabels([f"{s.split('_')[1]}\n{m.replace('synthetic_z_plumbing', 'synthZ')}" for s, m in columns],
                       fontsize=7)
    ax.set_yticks([len(names) - 0.5 - i for i in range(len(names))]); ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"fit status by cell (Q={q or 'all'}); number = chi2 / physical point")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLORS.values()]
    ax.legend(handles, list(STATUS_COLORS), loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=8)
    fig.tight_layout()
    path = out / "figures" / f"fit_status_q{q or 'all'}"
    path.parent.mkdir(parents=True, exist_ok=True)
    save(fig, str(path))
    plt.close(fig)
    return path


def pull_figure(out: Path, q: str | None, cells: list[str]) -> Path | None:
    rows = [r for r in ps.load_rows(out / "rows") if r["status"] == "ok" and r.get("pulls")
            and (q is None or r["q"] == q) and r["cell"] in cells and r["mode"] in ("dis", "exp", "both")]
    if not rows:
        return None
    plt, save = _matplotlib()
    fields = ("t3", "t8", "sigma", "t15", "g", "v3", "v8", "v", "v15")
    columns = [(s, m) for s in ps.SUITES for m in ("dis", "exp", "both")]
    fig, axes = plt.subplots(len(cells), 1, figsize=(10, 2.3 * len(cells) + 1), squeeze=False)
    for ax, name in zip(axes[:, 0], cells):
        grid = np.full((len(fields), len(columns)), np.nan)
        signed = np.full_like(grid, np.nan)
        for j, (suite, mode) in enumerate(columns):
            for row in rows:
                if row["cell"] == name and row["suite"] == suite and row["mode"] == mode:
                    for i, f in enumerate(fields):
                        grid[i, j] = row["pulls"][f]["rms"]
                        signed[i, j] = row["pulls"][f]["signed"]
        image = ax.imshow(grid, vmin=0, vmax=3, cmap="viridis", aspect="auto")
        for i in range(len(fields)):
            for j in range(len(columns)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{signed[i, j]:+.1f}", ha="center", va="center", fontsize=6, color="white")
        ax.set_yticks(range(len(fields))); ax.set_yticklabels(fields, fontsize=7)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([f"{s.split('_')[1]} {m}" for s, m in columns], fontsize=7)
        ax.set_title(f"{name}: RMS pull (colour), signed pull (text)", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.02)
    fig.tight_layout()
    path = out / "figures" / f"pulls_q{q or 'all'}"
    save(fig, str(path))
    plt.close(fig)
    return path


def write_report(out: Path, q: str | None = None) -> dict:
    report: dict = {}
    table = ps.screen_table(out / "screen") if (out / "screen").exists() else {}
    if table:
        screened = []
        for (name, digest), rows in table.items():
            score = ps.screen_score(rows)
            screened.append({"cell": name, "cell_hash": digest,
                             "prior_form": next(iter(rows.values()))["prior_form"],
                             "qualified": score is not None,
                             "score": None if score is None else list(score)})
        screened.sort(key=lambda r: (not r["qualified"], r["score"] or [True, 99, 99]))
        report["screen"] = screened
        report["screen_figure"] = str(screen_figure(out))
    if (out / "rows").exists():
        ranking = ps.rank_fits(out / "rows", modes=ps.PHYSICS_MODES, q=q)
        report["fit_ranking"] = ranking
        order = [r["cell"] for r in ranking]
        report["status_figure"] = str(status_figure(out, q, order))
        report["pull_figure"] = str(pull_figure(out, q, order[:6]))
    ps._write_json(out / f"report_q{q or 'all'}.json", report)
    for entry in report.get("fit_ranking", [])[:12]:
        print(f"{entry['cell']:<45} {entry['prior_form']:<14} pass {entry['n_pass']}/{entry['n_expected']} "
              f"constraint {entry['worst_abs_constraint_residual']:.2g} sumRMS {entry['sum_rms_pull']:.2f}")
    return report
