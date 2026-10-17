"""Pairwise evidence scans of the sampled objective, plus its Laplace summary.

Unlike the other plotters here, these are not views of a *fit* -- they are views
of the *objective a sampler has to cross*.  ``pixel.plotting.plot_evidence_scan``
scans the same scalar HMC and NUTS integrate against, holding every unscanned
coordinate at the MAP, so valleys, ridges and second minima are visible directly.

**The parameter list is split deliberately, not arbitrarily.**
``N`` parameters is ``N**2`` panels, so the cost is the square of a list length,
and eight per figure is the choice made here.  (``MAX_SCAN_PARAMETERS`` itself
was raised from 8 to 100 on 2026-10-16 for the explicit-``q`` field-space scan in
``closure_small_q_integration/plot_evidence_scans.py``, which needs 34 at once;
it no longer refuses a ninth, so the chunking below is this module's policy
rather than the library's.)
In ``exp`` the free vector is exactly nineteen scalars: twelve
Hubbard--Stratonovich auxiliaries (six per Drell--Yan table) and seven fitted
normalization amplitudes (five DIS, two DY).  The chunks below each keep
normalizations in the *same figure* as the auxiliaries they are meant to be read
against, because the whole question is how the two blocks relate; splitting them
into separate figures would answer nothing.

**On a bilinear model the MAP is only the point the slice is taken at.** PIXEL
warns that a MAP is not a valid posterior summary for a Hubbard--Stratonovich
posterior, and that is fine here: this figure is a view of the objective, and the
conditional slice has to be taken somewhere.

**Why a one-sided chunk exists.** Measured on this suite, ``E^2`` is exactly even
in ``f`` about the MAP (the 1-D scans are symmetric to ~1e-6 relative), so
``d2E2/df dN`` is *odd* in ``f`` and vanishes identically at ``f = 0``.  Every
``f``-against-normalization panel in the symmetric chunks is therefore
axis-aligned *by parity*, not because the coupling is absent.  The one-sided
chunk windows the auxiliaries over ``[0, +4 sigma]`` instead, which keeps the MAP
as the contour reference -- so ``Delta E^2`` stays non-negative and the contours
keep their usual meaning -- while exposing the half-plane where the coupling is
non-zero.

Run standalone::

    python -m closure_JAM_truth_small.plot_evidence_scans --Q 2 --mode exp
    python -m closure_JAM_truth_small.plot_evidence_scans --figures correlation
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from . import config as cfg
from . import fit as fitmod
from .run_closure import _prepare_matplotlib, save_figure_both


#: Nodes per axis.  The default is the coarse/fast grid: 26x26 keeps a full
#: 8-parameter chunk near half an hour on this suite, where the 51x51 default
#: would be roughly fifteen times the panel cost for contours this smooth.
DEFAULT_N_POINTS = 26

#: Half-width of a one-sided auxiliary window, in Laplace sigma.  Four, not six:
#: the scanned directions are flatter than Gaussian here, so 4 sigma lands near
#: ``Delta E^2 = 14..21`` and stays inside ``plot_evidence_scan``'s default 25
#: colour ceiling, while the measured coupling is already within a few percent
#: of its maximum.
ONESIDED_SIGMA = 4.0


def _dy_tables(labels) -> list[str]:
    """Drell-Yan table stems that own Hubbard-Stratonovich auxiliaries."""
    seen = []
    for label in labels:
        if " hubbard[" in label:
            owner = label.split(" hubbard[")[0]
            if owner not in seen:
                seen.append(owner)
    return seen


def chunk_plan(labels) -> list[tuple[str, list[str], str]]:
    """``(name, parameters, blurb)`` per figure, each at most eight parameters.

    Derived from the model's own labels rather than hard-coded, so a mode with a
    different table roster still produces a coherent split instead of a
    ``ValueError`` from a stale list.
    """
    hubbard = [lab for lab in labels if " hubbard[" in lab]
    norms = [lab for lab in labels if " normalization / " in lab]
    dy_norms = [n for n in norms if n.split(" normalization")[0] in _dy_tables(labels)]
    dis_norms = [n for n in norms if n not in dy_norms]

    plan: list[tuple[str, list[str], str]] = []
    for table in _dy_tables(labels):
        own = [lab for lab in hubbard if lab.startswith(f"{table} ")]
        # Six auxiliaries plus both DY normalizations is exactly the ceiling.
        params = (own + dy_norms)[:8]
        plan.append((
            f"{table.replace('dy_e866_', '')}_f_vs_dy_norms",
            params,
            f"{table} Hubbard-Stratonovich auxiliaries against the DY normalizations",
        ))
    if norms:
        plan.append((
            "all_normalizations", norms[:8],
            f"all {len(norms[:8])} fitted normalization amplitudes",
        ))
    if dis_norms and hubbard:
        # A few representative auxiliaries against the DIS block: enough to show
        # the cross-sector response without a fifth near-identical figure.
        picks = hubbard[:1] + hubbard[3:4] + [
            lab for lab in hubbard if lab.startswith(_dy_tables(labels)[-1])
        ][:1]
        params = (list(dict.fromkeys(picks)) + dis_norms)[:8]
        plan.append((
            "f_vs_dis_norms", params,
            "representative DY auxiliaries against the DIS normalizations",
        ))
    return plan


def build_target(q_key: str, mode: str, *, use_kernel_cache: bool = True):
    """``(analysis, model, result, t0)`` centred on the MAP, t0 converged.

    The DIS multiplicative uncertainties are referenced to theory rather than
    data, and that reference is exactly what the fitted DIS normalizations
    respond to -- so the scan is taken on the converged-t0 model, matching the
    production fit rather than a cheaper stand-in.
    """
    analysis, _fields = fitmod.build_analysis(
        q_key, mode, use_kernel_cache=use_kernel_cache
    )
    model = analysis.compile()
    t0, _iterations = fitmod._iterate_t0(
        q_key, mode, model, use_kernel_cache=use_kernel_cache
    )
    if t0:
        analysis, _fields = fitmod.build_analysis(
            q_key, mode, use_kernel_cache=use_kernel_cache, t0=t0
        )
        model = analysis.compile()
    with warnings.catch_warnings():
        # A bilinear MAP warns by design; see the module docstring.
        warnings.simplefilter("ignore")
        result = analysis.run_map()
    return analysis, model, result, t0


def _sigma(target, center, indices) -> dict:
    """Laplace sigma per free index, from ``2 H^-1`` as the scanner uses."""
    hessian = np.asarray(target.hessian(center), dtype=float)
    covariance = 2.0 * np.linalg.pinv(hessian)
    return {i: float(np.sqrt(covariance[i, i])) for i in indices}


def plot_chunks(q_key, mode, out_dir, *, n_points=DEFAULT_N_POINTS,
                use_kernel_cache=True, onesided=False):
    """Write one evidence-scan figure per chunk; returns the manifest dict."""
    from pixel.plotting.evidence_scan import (
        resolve_scan_target, scan_evidence, plot_evidence_scan,
    )

    analysis, model, result, t0 = build_target(
        q_key, mode, use_kernel_cache=use_kernel_cache
    )
    target, center = resolve_scan_target(analysis)
    labels = [str(lab) for lab in target.free_labels()]
    manifest = {
        "truth_Q": q_key, "mode": mode, "n_points": int(n_points),
        "n_free": len(labels), "free_labels": labels,
        "center_E2": float(getattr(result, "fun", float("nan"))),
        "t0_tables": sorted(t0), "chunks": {},
    }

    plan = chunk_plan(labels)
    if onesided:
        hubbard = [lab for lab in labels if " hubbard[" in lab]
        dy_norms = [
            lab for lab in labels if " normalization / " in lab
            and lab.split(" normalization")[0] in _dy_tables(labels)
        ]
        # The widest auxiliaries carry the coupling; pair them with both DY
        # normalizations and window them one-sided.
        indices = [labels.index(lab) for lab in hubbard]
        sigma = _sigma(target, center, indices)
        widest = sorted(hubbard, key=lambda lab: -sigma[labels.index(lab)])[:4]
        plan = [(
            "onesided_f_vs_dy_norms", widest + dy_norms,
            "widest auxiliaries against the DY normalizations, on one-sided "
            f"f windows [0, +{ONESIDED_SIGMA:g} sigma]",
        )]

    for name, params, blurb in plan:
        stem = out_dir / f"evidence_scan_{mode}_{name}"
        bounds = None
        if onesided:
            indices = [labels.index(p) for p in params]
            sigma = _sigma(target, center, indices)
            bounds = [
                (0.0, ONESIDED_SIGMA * sigma[labels.index(p)])
                if " hubbard[" in p else (-ONESIDED_SIGMA, ONESIDED_SIGMA)
                for p in params
            ]
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scan = scan_evidence(
                (target, center), params, bounds=bounds,
                n_points=n_points, verbose=True,
            )
            plot_evidence_scan(
                scan, path=str(stem), close=True,
                title=(
                    f"Evidence scan, small closure JAM truth Q={q_key}, "
                    f"mode {mode}: {blurb}\n"
                    f"physical coordinates, $E^2$, {scan.n_points}x"
                    f"{scan.n_points} per pair, conditional slice at the MAP "
                    "(not a profile)"
                ),
            )
        seconds = time.perf_counter() - started
        # Keep the numbers, not just the picture: a panel is re-readable from
        # this without a second multi-hour scan.
        np.savez_compressed(
            f"{stem}_scan.npz",
            labels=np.asarray(scan.labels), indices=np.asarray(scan.indices),
            axes=np.stack([np.asarray(a) for a in scan.axes]),
            center=np.asarray(scan.center),
            center_value=np.asarray(scan.center_value),
            diagonal=np.stack([np.asarray(d) for d in scan.diagonal]),
            **{f"panel_{a}_{b}": np.asarray(v) for (a, b), v in scan.panels.items()},
        )
        manifest["chunks"][name] = {
            "blurb": blurb, "labels": list(scan.labels), "seconds": seconds,
            "n_invalid": int(scan.n_invalid),
            "center_value": float(scan.center_value),
            "bounds_source": scan.metadata.get("bounds_source"),
            "variation": scan.metadata.get("variation"),
            "worst_dip_below_reference": scan.metadata.get(
                "worst_dip_below_reference"
            ),
            "warnings": [str(w.message) for w in caught],
        }
        print(f"wrote {stem}.png (+ .pdf) in {seconds:.1f}s "
              f"({scan.n_invalid} masked node(s))")
    return manifest


def _short(label: str) -> str:
    """Compact tick label: ``dy_e866_pp hubbard[3]`` -> ``pp f3``."""
    if " hubbard[" in label:
        owner, _, rest = label.partition(" hubbard[")
        return f"{owner.replace('dy_e866_', '')} f{rest.rstrip(']')}"
    if " normalization / " in label:
        return f"{label.split(' normalization / ')[0]} N"
    return label


def plot_correlation(q_key, mode, out_dir, *, use_kernel_cache=True):
    """Laplace correlation over the whole free vector, in one figure.

    The scans cannot show all nineteen coordinates at once -- the panel count is
    the square of the list length -- so the auxiliary-against-normalization block
    is only ever visible in pieces there.  This is its quadratic summary at the
    same centre, whole.

    ``E^2 = -2 log p``, so the Gaussian covariance is ``2 H^-1``: the same
    convention ``evidence_scan`` uses to size its windows, which is what makes
    the two commensurate.  **It is the Gaussian approximation and nothing more.**
    On a bilinear H-S posterior a single quadratic expansion is not a valid
    posterior summary -- that is why the scans exist -- so a correlation here
    that the matching scan panel does not show is the approximation failing.
    """
    import matplotlib.pyplot as plt

    _analysis, model, result, _t0 = build_target(
        q_key, mode, use_kernel_cache=use_kernel_cache
    )
    center = np.asarray(result.x, dtype=float).reshape(-1)
    labels = [str(lab) for lab in model.free_labels()]
    hessian = np.asarray(model.hessian(center), dtype=float)
    covariance = 2.0 * np.linalg.pinv(hessian)
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = covariance / np.outer(sigma, sigma)
    corr = np.where(np.isfinite(corr), corr, np.nan)

    n = len(labels)
    n_f = sum(1 for lab in labels if " hubbard[" in lab)
    names = [_short(lab) for lab in labels]

    fig, (ax, ax_s) = plt.subplots(
        1, 2, figsize=(13.0, 9.0), gridspec_kw={"width_ratios": [1.0, 0.34]}
    )
    mesh = ax.pcolormesh(
        np.arange(n + 1), np.arange(n + 1), np.ma.masked_invalid(corr),
        cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="flat",
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(names, fontsize=8)
    # The auxiliary/normalization split is the whole point, so rule it off.
    ax.axhline(n_f, color="black", lw=1.6)
    ax.axvline(n_f, color="black", lw=1.6)
    for i in range(n):
        for j in range(n):
            value = corr[i, j]
            if not np.isfinite(value) or abs(value) < 0.005:
                continue
            ax.text(
                j + 0.5, i + 0.5,
                f"{value:+.2f}".replace("+0.", ".").replace("-0.", "-."),
                ha="center", va="center", fontsize=5.5,
                color="white" if abs(value) > 0.55 else "0.15",
            )
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02).set_label(
        "Laplace correlation", fontsize=9
    )
    ax.set_title(
        f"H-S auxiliaries (left/top of the rule, {n_f}) "
        f"vs fitted normalizations ({n - n_f})", fontsize=10,
    )

    ax_s.barh(np.arange(n), sigma, color="tab:blue")
    ax_s.set_yticks(np.arange(n))
    ax_s.set_yticklabels(names, fontsize=8)
    ax_s.invert_yaxis()
    ax_s.axhline(n_f - 0.5, color="black", lw=1.6)
    ax_s.set_xlabel(r"Laplace $\sigma=\sqrt{2[H^{-1}]_{ii}}$", fontsize=9)
    ax_s.set_title("marginal width", fontsize=10)
    ax_s.grid(axis="x", ls=":", color="0.8")

    fig.suptitle(
        f"Laplace correlation at the MAP -- small closure, JAM truth "
        f"Q={q_key}, mode {mode} (GAUSSIAN APPROXIMATION; read against the "
        "evidence scans)", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stem = out_dir / f"evidence_scan_{mode}_laplace_correlation"
    paths = save_figure_both(fig, stem)
    plt.close(fig)

    cross = sorted(
        (
            (labels[i], labels[j], float(corr[i, j]))
            for i in range(n_f) for j in range(n_f, n)
            if np.isfinite(corr[i, j])
        ),
        key=lambda row: -abs(row[2]),
    )
    stem.with_suffix(".json").write_text(json.dumps({
        "truth_Q": q_key, "mode": mode, "labels": labels, "short_labels": names,
        "n_hubbard": n_f, "center_E2": float(result.fun),
        "sigma": [float(v) for v in sigma],
        "correlation": [
            [None if not np.isfinite(v) else float(v) for v in row] for row in corr
        ],
        "strongest_f_normalization_cross": [
            {"f": a, "normalization": b, "corr": c} for a, b, c in cross[:25]
        ],
        "caveat": (
            "Gaussian approximation at a bilinear H-S MAP; not a valid "
            "posterior summary on its own. Read against the evidence scans."
        ),
    }, indent=2) + "\n")
    print(f"wrote {paths[0]} (+ .pdf, .json)")
    return corr, sigma, labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pairwise evidence scans and their Laplace summary."
    )
    ap.add_argument("--Q", default="2", choices=list(cfg.TRUTH_Q_CHOICES))
    ap.add_argument("--mode", default="exp", choices=list(cfg.TEST_MODES))
    ap.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS,
                    help=f"nodes per axis (default {DEFAULT_N_POINTS})")
    ap.add_argument("--figures", default="all",
                    choices=["all", "chunks", "onesided", "correlation"])
    ap.add_argument("--no-kernel-cache", action="store_true",
                    help="rebuild kernels instead of reusing the cache")
    args = ap.parse_args()

    _prepare_matplotlib()
    import matplotlib
    matplotlib.use("Agg", force=True)

    out_dir = cfg.results_dir(args.Q)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_kernel_cache

    if args.figures in ("all", "chunks"):
        manifest = plot_chunks(
            args.Q, args.mode, out_dir,
            n_points=args.n_points, use_kernel_cache=use_cache,
        )
        path = out_dir / f"evidence_scan_{args.mode}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {path}")
    if args.figures in ("all", "onesided"):
        plot_chunks(
            args.Q, args.mode, out_dir, n_points=args.n_points,
            use_kernel_cache=use_cache, onesided=True,
        )
    if args.figures in ("all", "correlation"):
        plot_correlation(args.Q, args.mode, out_dir, use_kernel_cache=use_cache)


if __name__ == "__main__":
    main()
