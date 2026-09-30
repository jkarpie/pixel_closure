#!/usr/bin/env python3
"""Kinematic coverage of the data each closure suite fits.

The kinematics come from each suite's **own** layout code -- ``generate.lattice_layout``,
``exp_layout``, ``nnpdf_layout`` and ``dy_layout`` (plus ``synthetic_z_layout`` in the
``_small`` suites) -- i.e. the rows generation folds, after the closure's Q2/W2 and DY
cuts.  Nothing is read from ``data/truthQ_*``, so a suite need not have been generated.
The audit those layouts trigger (DIS manifest, systematic sidecars) is redirected to a
temporary directory: the suites' ``data/`` trees are not touched.

The truth does not enter the kinematics, so the eight suites are four selections.
``extract`` runs all eight anyway and refuses to continue if a truth pair disagrees.

    extract  --out DIR     # writes DIR/kinematics.json (PIXEL venv, ~30 s)
    plot     --out DIR     # writes the figures and DIR/summary.md

Run from the repo root with PIXEL's venv::

    ../pixel/venv/bin/python -m campaign.kinematic_coverage extract --out campaign/kinematic_coverage_2026-09-16
    ../pixel/venv/bin/python -m campaign.kinematic_coverage plot --out campaign/kinematic_coverage_2026-09-16
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: ``(key, title, (JAM-truth suite, NNPDF-truth suite))`` in panel order.
SELECTIONS = (
    ("jam24", "JAM24", ("closure_JAM_truth_JAM_data", "closure_NNPDF_truth_JAM_data")),
    ("nnpdf40", "NNPDF4.0", ("closure_JAM_truth_NNPDF_data", "closure_NNPDF_truth_NNPDF_data")),
    ("union", "Union (JAM24 ∪ NNPDF4.0)", ("closure_JAM_truth", "closure_NNPDF_truth")),
    ("small", "Mock / smoke", ("closure_JAM_truth_small", "closure_NNPDF_truth_small")),
)


def _floats(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


# -- extraction (one suite per subprocess) -------------------------------------


def extract_suite(suite: str, sandbox: Path) -> dict:
    """Every fitted kinematic point of ``suite``, straight from its layout functions."""
    os.environ.pop("PIXEL_CLOSURE_ENSEMBLES", None)   # all ensembles, as generation runs
    cfg = importlib.import_module(f"{suite}.config")
    sandbox.mkdir(parents=True, exist_ok=True)
    cfg.DIS_MANIFEST_PATH = sandbox / "dis_manifest.json"
    cfg.DIS_SYS_DIR = sandbox / "dis_systematics"
    gen = importlib.import_module(f"{suite}.generate")
    dis_audit = importlib.import_module(f"{suite}.dis_audit")
    cfg.DIS_MANIFEST_PATH.write_text(json.dumps(dis_audit.build_manifest()))

    small = suite.endswith("_small")
    records = []

    # Lattice: every field folds its ensemble's one (z, p) grid, so keep one per ensemble.
    hbarc = float(cfg.HBARC_GEV_FM)
    lam = float(getattr(cfg, "PITD_MATCHING_LAMBDA", 1.0))
    seen = set()
    for rec in gen.lattice_layout():
        if rec["kind"] != "pseudoitd":
            continue
        if small:
            ens, n_sites, a_fm = "mock", int(cfg.LATTICE_L), float(cfg.LATTICE_A_FM)
        else:
            ens, n_sites, a_fm = rec["ensemble"], int(rec["L"]), float(rec["a"])
        if ens in seen:
            continue
        seen.add(ens)
        z, p = (np.asarray(c, dtype=float) for c in rec["_leading"])
        z_fm = z * a_fm
        records.append({
            "id": f"lattice.{ens}", "process": "lattice", "a_fm": a_fm, "L": n_sites,
            "z_fm": _floats(z_fm), "nu": _floats(2.0 * math.pi * p * z / n_sites),
            "mu2": _floats((lam * hbarc / z_fm) ** 2),
        })

    for rec in gen.exp_layout():
        x, q2 = rec["_leading"]
        records.append({"id": f"dis.{rec['label']}", "process": "dis", "kind": rec["kind"],
                        "x": _floats(x), "Q2": _floats(q2)})

    if hasattr(gen, "nnpdf_layout"):
        for rec in gen.nnpdf_layout():
            if rec.get("section") == "dy":
                records.append({"id": rec["contract_id"], "process": "dy", "kind": rec["kind"],
                                "reaction": rec.get("reaction"), "S": _floats(rec["S"]),
                                "Q2": _floats(rec["Q2"]), "Y": _floats(rec["Y"])})
            else:
                x, q2 = rec["_leading"]
                records.append({"id": rec["contract_id"], "process": "dis", "kind": rec["kind"],
                                "x": _floats(x), "Q2": _floats(q2)})

    dy = list(gen.dy_layout())
    if small:
        dy += gen.synthetic_z_layout()
    for rec in dy:
        records.append({"id": f"dy.{rec['label']}", "process": "dy", "kind": rec["kind"],
                        "reaction": rec.get("reaction"),
                        "synthetic": rec.get("classification") == "synthetic_proxy",
                        "S": _floats(rec["S"]), "Q2": _floats(rec["Q2"]), "Y": _floats(rec["Y"])})

    return {"suite": suite, "contract_mode": getattr(cfg, "CONTRACT_MODE", None),
            "Q0_2": float(cfg.Q0_2), "DIS_W2_MIN": float(cfg.DIS_W2_MIN), "records": records}


def extract(out: Path) -> Path:
    suites = [s for _, _, pair in SELECTIONS for s in pair]
    with tempfile.TemporaryDirectory(prefix="kinematic_coverage_") as tmp:
        tmp = Path(tmp)
        procs = {
            suite: subprocess.Popen(
                [sys.executable, "-m", "campaign.kinematic_coverage", "_suite", suite,
                 "--json", str(tmp / f"{suite}.json"), "--sandbox", str(tmp / suite)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for suite in suites
        }
        for suite, proc in procs.items():
            log, _ = proc.communicate()
            if proc.returncode:
                raise RuntimeError(f"{suite}: extraction failed\n{log}")
        dumps = {s: json.loads((tmp / f"{s}.json").read_text()) for s in suites}

    payload = {"selections": {}}
    for key, title, (jam, nnpdf) in SELECTIONS:
        if dumps[jam]["records"] != dumps[nnpdf]["records"]:
            raise RuntimeError(f"{jam} and {nnpdf} fit different kinematics; the truth "
                               "should not change the selection")
        payload["selections"][key] = {"title": title, "suites": [jam, nnpdf],
                                      **{k: v for k, v in dumps[jam].items() if k != "suite"}}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "kinematics.json"
    path.write_text(json.dumps(payload))
    return path


# -- plotting ------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
CONTEXT = "#dcdbd6"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"    # categorical slots 1-3
ORDINAL = ("#86b6ef", "#3987e5", "#256abf", "#104281")  # coarse -> fine lattice spacing

#: ``key -> (legend label, colour, marker, hollow)``, in legend order.
CATEGORIES = {
    "dis_ft": ("fixed-target e/μ DIS (SLAC, BCDMS, NMC, HERMES)", BLUE, "o", False),
    "dis_hera": ("HERA DIS (NC, CC, bottom)", BLUE, "^", False),
    "dis_nu": ("ν DIS (CHORUS, Pb)", AQUA, "o", False),
    "dy_ft": ("fixed-target Drell–Yan (E866 pp/pd, E605 Cu)", ORANGE, "o", False),
    "dy_collider": ("collider Z/γ* (CDF, ATLAS, LHCb)", ORANGE, "^", False),
    "dy_synthetic": ("synthetic Z row (plumbing only)", ORANGE, "^", True),
}


def category(rec: dict) -> str:
    if rec["process"] == "dis":
        if "chorus" in rec["id"]:
            return "dis_nu"
        return "dis_hera" if "hera" in rec["id"] else "dis_ft"
    if rec.get("synthetic"):
        return "dy_synthetic"
    return "dy_collider" if math.sqrt(rec["S"][0]) > 100.0 else "dy_ft"


def xq2_points(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """DIS ``(x, Q2)`` as given; Drell-Yan at LO, both ``x1,2 = sqrt(Q2/S) e^{+-y}``.

    A bin *centre* can sit past the LO limit even though the bin does not -- ATLAS 8 TeV
    high-mass, M = 1000 GeV at |y| = 2.3 gives x1 = 1.25 -- so ``x >= 1`` is dropped here
    and counted by :func:`n_unphysical`.
    """
    if rec["process"] == "dis":
        return np.asarray(rec["x"]), np.asarray(rec["Q2"])
    q2, s, y = (np.asarray(rec[k]) for k in ("Q2", "S", "Y"))
    tau = np.sqrt(q2 / s)
    x = np.concatenate([tau * np.exp(y), tau * np.exp(-y)])
    q2 = np.concatenate([q2, q2])
    keep = x < 1.0
    return x[keep], q2[keep]


def n_unphysical(sel: dict) -> int:
    """LO Drell-Yan momentum fractions at or past 1 that :func:`xq2_points` drops."""
    n = 0
    for rec in _experimental(sel):
        if rec["process"] == "dy":
            tau = np.sqrt(np.asarray(rec["Q2"]) / np.asarray(rec["S"]))
            n += int((tau * np.exp(np.abs(np.asarray(rec["Y"]))) >= 1.0).sum())
    return n


def _matplotlib():
    from closure_JAM_truth_small.run_closure import _prepare_matplotlib, save_figure_both

    _prepare_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": INK_3,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9,
    })
    return plt, save_figure_both


def _experimental(sel: dict) -> list[dict]:
    return [r for r in sel["records"] if r["process"] in ("dis", "dy")]


def _scatter(ax, x, q2, key, *, context=False):
    label, colour, marker, hollow = CATEGORIES[key]
    size = 13 if marker == "o" else 18
    if context:
        ax.scatter(x, q2, s=size, marker=marker, c=CONTEXT, edgecolors="none",
                   rasterized=True, zorder=1)
    elif hollow:
        ax.scatter(x, q2, s=46, marker=marker, facecolors="none", edgecolors=colour,
                   linewidths=1.4, zorder=4)
    else:
        ax.scatter(x, q2, s=size, marker=marker, c=colour, edgecolors=SURFACE,
                   linewidths=0.3, rasterized=True, zorder=3)


def plot_xq2(payload: dict, out: Path, plt, save) -> list[Path]:
    selections = payload["selections"]
    union = selections["union"]
    q0_2, w2_min = union["Q0_2"], union["DIS_W2_MIN"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.4), sharex=True, sharey=True)
    xs = np.logspace(-6, math.log10(0.995), 400)
    for ax, (key, *_rest) in zip(axes.ravel(), SELECTIONS):
        sel = selections[key]
        mine = {r["id"] for r in sel["records"]}
        if key != "union":
            for rec in _experimental(union):
                if rec["id"] not in mine:
                    _scatter(ax, *xq2_points(rec), category(rec), context=True)
        for cat in CATEGORIES:                         # fixed draw order, so layers agree
            for rec in _experimental(sel):
                if category(rec) == cat:
                    _scatter(ax, *xq2_points(rec), cat)

        # The DIS cuts every suite applies (Q2 >= Q0^2 = mc^2, W2 > 10 GeV^2).
        mp2 = 0.938272**2
        ax.axhline(q0_2, color=INK_3, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.plot(xs, (w2_min - mp2) * xs / (1.0 - xs), color=INK_3, lw=0.9, ls=(0, (4, 3)),
                zorder=2)
        ax.text(1.25e-5, q0_2 * 0.88, r"$Q^2 = m_c^2$", color=INK_2, fontsize=7.5,
                va="top")

        exp = _experimental(sel)
        n_dis = sum(r["process"] == "dis" for r in exp)
        n_dy = sum(r["process"] == "dy" for r in exp)
        n_rows = sum(len(r.get("x") or r["S"]) for r in exp)
        n_lat = sum(r["process"] == "lattice" for r in sel["records"])
        lattice = f"{n_lat} lattice ensemble{'s' * (n_lat != 1)}"
        for text, dy, style in (
            (sel["title"], 33, {"fontsize": 11.5, "fontweight": "bold", "color": INK}),
            (f"{n_dis} DIS + {n_dy} DY datasets · {n_rows} points · + {lattice}", 18,
             {"fontsize": 8.5, "color": INK_2}),
            (", ".join(sel["suites"]), 5,
             {"fontsize": 7.5, "color": INK_3, "family": "monospace"}),
        ):
            ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, dy),
                        textcoords="offset points", va="bottom", ha="left", **style)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-5, 1.0)
        ax.set_ylim(0.5, 3e6)
        ax.grid(True, which="major", color=INK_3, alpha=0.18, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(r"$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$Q^2$ [GeV$^2$]")

    handles = [
        plt.Line2D([], [], ls="none", marker=m, markersize=6.5 if m == "o" else 7.5,
                   markerfacecolor="none" if hollow else c, markeredgecolor=c,
                   markeredgewidth=1.3 if hollow else 0.0, label=label)
        for label, c, m, hollow in CATEGORIES.values()
    ]
    handles.append(plt.Line2D([], [], ls="none", marker="o", markersize=6.5,
                              markerfacecolor=CONTEXT, markeredgewidth=0.0,
                              label="in the union, not in this selection"))
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.06, 0.972), handletextpad=0.3, columnspacing=1.8)
    fig.suptitle("Kinematic coverage of the closure-test datasets", x=0.07, y=0.995,
                 ha="left", fontsize=13, fontweight="bold")
    dropped = n_unphysical(union)
    fig.text(0.07, 0.022,
             "Rows after the closure cuts, as generation folds them; dashed lines are the "
             rf"DIS cuts $Q^2 \geq m_c^2$ and $W^2 > {w2_min:g}$ GeV$^2$.  "
             "Each JAM/NNPDF truth pair fits identical kinematics.",
             fontsize=7.5, color=INK_3)
    fig.text(0.07, 0.006,
             r"Drell–Yan at LO: $x_{1,2} = (M/\sqrt{s})\,e^{\pm y}$ at $Q^2 = M^2$, bin "
             r"centres, both $x_1$ and $x_2$ drawn"
             + (rf"; {dropped} bin centre{'s' * (dropped != 1)} with LO $x \geq 1$ "
                "(ATLAS 8 TeV high-mass) not drawn." if dropped else "."),
             fontsize=7.5, color=INK_3)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.93), h_pad=4.2, w_pad=1.5)
    return save(fig, out / "kinematic_coverage_xQ2.png", dpi=170)


def _spacing_groups(records):
    """Lattice ensembles grouped by spacing to 0.01 fm (0.086/0.087 and 0.067/0.068 pair)."""
    groups: dict[float, list[dict]] = {}
    for rec in records:
        groups.setdefault(round(rec["a_fm"], 2), []).append(rec)
    return dict(sorted(groups.items(), reverse=True))


def plot_lattice(payload: dict, out: Path, plt, save) -> list[Path]:
    full = [r for r in payload["selections"]["union"]["records"] if r["process"] == "lattice"]
    mock = [r for r in payload["selections"]["small"]["records"] if r["process"] == "lattice"]
    hbarc = 0.1973269804

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    groups = _spacing_groups(full)
    if len(groups) > len(ORDINAL):
        raise RuntimeError(f"{len(groups)} spacing groups; the ordinal ramp has {len(ORDINAL)}")
    handles = []
    for colour, (a, recs) in zip(ORDINAL, groups.items()):
        z = np.concatenate([r["z_fm"] for r in recs])
        nu = np.concatenate([r["nu"] for r in recs])
        spacing = sorted({r["a_fm"] for r in recs})
        a_label = "/".join(f"{v:.3f}" for v in spacing)
        label = f"a = {a_label} fm  ({len(recs)} ens.)"
        ax.scatter(z, nu, s=30, c=colour, edgecolors=SURFACE, linewidths=0.5, label=label,
                   zorder=3)
    for rec in mock:
        ax.scatter(rec["z_fm"], rec["nu"], s=60, marker="D", facecolors="none",
                   edgecolors=INK, linewidths=1.3, zorder=4,
                   label=f"mock suites: a = {rec['a_fm']:.2f} fm, one z")
    ax.set_xlabel(r"Wilson-line separation $z$ [fm]")
    ax.set_ylabel(r"Ioffe time $\nu = p_z z$")
    ax.set_xlim(0.0, 0.37)
    ax.set_ylim(0.0, None)
    ax.grid(True, color=INK_3, alpha=0.18, lw=0.5)
    ax.spines["right"].set_visible(False)

    # mu(z) = lambda / z with lambda = 1: the same z axis, read as the matching scale.
    top = ax.secondary_xaxis(
        "top",
        functions=(lambda z: (hbarc / np.clip(z, 1e-3, None)) ** 2,
                   lambda m2: hbarc / np.sqrt(np.clip(m2, 1e-6, None))),
    )
    top.set_xticks([0.3, 0.5, 1, 2, 5, 10])
    top.set_xticklabels(["0.3", "0.5", "1", "2", "5", "10"])
    top.set_xlabel(r"matching scale $\mu^2 = 1/z^2$ [GeV$^2$]", color=INK_2)
    top.tick_params(colors=INK_2)

    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_title("Lattice pseudo-ITD kinematics", loc="left", fontsize=12, fontweight="bold",
                 pad=34)
    fig.text(0.015, 0.012,
             "Identical in all six full suites.  Every field folds its ensemble's grid; "
             "the Mellin moments sit at Q² = 4 GeV² and are not drawn.",
             fontsize=7.5, color=INK_3)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    return save(fig, out / "kinematic_coverage_lattice.png", dpi=170)


def summary_markdown(payload: dict) -> str:
    lines = ["| selection | category | datasets | points | x range | Q² range [GeV²] |",
             "|---|---|---:|---:|---|---|"]
    for key, *_ in SELECTIONS:
        sel = payload["selections"][key]
        for cat, (label, *_style) in CATEGORIES.items():
            recs = [r for r in _experimental(sel) if category(r) == cat]
            if not recs:
                continue
            x, q2 = (np.concatenate(v) for v in zip(*(xq2_points(r) for r in recs)))
            n = sum(len(r.get("x") or r["S"]) for r in recs)
            lines.append(f"| {sel['title']} | {label} | {len(recs)} | {n} | "
                         f"{x.min():.2g} – {x.max():.2g} | {q2.min():.3g} – {q2.max():.3g} |")
    return "\n".join(lines) + "\n"


def plot(out: Path) -> list[Path]:
    payload = json.loads((out / "kinematics.json").read_text())
    plt, save = _matplotlib()
    written = plot_xq2(payload, out, plt, save) + plot_lattice(payload, out, plt, save)
    (out / "summary.md").write_text(summary_markdown(payload))
    return written + [out / "summary.md"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "plot"):
        sub.add_parser(name).add_argument("--out", type=Path, required=True)
    one = sub.add_parser("_suite")
    one.add_argument("suite")
    one.add_argument("--json", type=Path, required=True)
    one.add_argument("--sandbox", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "_suite":
        args.json.write_text(json.dumps(extract_suite(args.suite, args.sandbox)))
    elif args.cmd == "extract":
        print(f"wrote {extract(args.out)}")
    else:
        for path in plot(args.out):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
