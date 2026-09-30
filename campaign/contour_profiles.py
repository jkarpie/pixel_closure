#!/usr/bin/env python3
"""How Gaussian is the H-S integrand on the steepest-descent contour the Gaussian sampler uses?

For one (suite, Q, mode, cell) this rebuilds the model exactly as ``fit.run_fit`` does
(including the DIS t0 rebuild), takes the reference blocks at fitted normalizations ``A=0``
(what both saddle-contour samplers use), loads or finds the saddle, and builds PIXEL's
affine steepest-descent frame ``f = f* + M u`` in which the quadratic action is ``u^T u``.
It then records, with no sampling approximation:

* ``profiles``: ``Delta E^2(u e_k) = E^2(f* + M e_k u) - E^2(f*)`` along every frame axis;
  a Gaussian integrand gives ``Re = u^2`` and ``Im = 0``.
* ``draws``: ``u ~ N(0, I)`` (the fixed-contour proposal) and the log importance weight
  ``log w = |u|^2/2 - Delta E^2/2``.  For log-normal weights the magnitude ESS fraction is
  ``exp(-var(Re log w))``, so ``std(Re log w)`` above ~1.5 already means ESS below 0.1.

    run      --suite S --q Q --mode M --cell-json FILE --out DIR [--cache NPZ]
    plot     --out DIR
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from . import prior_cells as pc


def pipeline_model(suite: str, q: str, mode: str):
    fit = importlib.import_module(f"{suite}.fit")
    model = fit.build_analysis(q, mode)[0].compile()
    t0, _ = fit._iterate_t0(q, mode, model)
    if t0:
        model = fit.build_analysis(q, mode, t0=t0)[0].compile()
    return model


def saddle_frame(args):
    """``(cell, partition, saddle, cache_status, delta)`` with ``delta(u) = E^2(f* + M u) - E^2(f*)``.

    The reference point, saddle and frame are exactly the ones both saddle-contour samplers use.
    """
    import jax
    import jax.numpy as jnp
    from pixel.core.evidence import prepared_hubbard_evidence_complex_f

    cfg = importlib.import_module(f"{args.suite}.config")
    cell = pc.validate(json.loads(Path(args.cell_json).read_text()))
    with pc.patched_prior(cfg, cell):
        model = pipeline_model(args.suite, args.q, args.mode)
    p = model.contour_partition()
    v = jnp.asarray(model.free_vector()).reshape(-1)
    ordinary, _norm, f0 = model.split_contour_vector(v)
    ref = model.merge_contour_vector(ordinary, jnp.zeros(p.n_normalization, dtype=jnp.float64), f0)
    blocks = model.blocks(ref)
    tol = cfg.CONTOUR_SADDLE_TOL
    sad, status = None, "no_cache_given"
    if args.cache:
        fingerprint = c._reference_contour_fingerprint(blocks, rcond=model.rcond, saddle_tol=tol)
        sad, status = c._load_cached_saddle(args.cache, fingerprint)
    if sad is None:
        sad = c.find_saddle(blocks, rcond=model.rcond, tol=tol)
    if not sad.converged:
        raise SystemExit("saddle did not converge; nothing to profile")
    contour = c._build_affine_saddle_contour(blocks, rcond=model.rcond, saddle=sad, saddle_tol=tol)
    action = jax.jit(
        lambda z: prepared_hubbard_evidence_complex_f(contour.prepared, z)
    )
    f_star, frame = np.asarray(contour.f_star), np.asarray(contour.frame)
    e_star = complex(action(jnp.asarray(f_star)))

    def delta(u_vec):
        return complex(action(jnp.asarray(f_star + frame @ u_vec))) - e_star

    return cell, p, sad, status, e_star, delta


def cmd_run(args) -> int:
    started = time.perf_counter()
    cell, p, sad, status, e_star, delta = saddle_frame(args)
    n = p.n_hubbard
    grid = np.linspace(-args.umax, args.umax, args.n_grid)
    profiles = np.empty((n, grid.size), dtype=complex)
    for k in range(n):
        for j, u in enumerate(grid):
            vec = np.zeros(n)
            vec[k] = u
            profiles[k, j] = delta(vec)
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((args.n_draws, n))
    delta_draws = np.array([delta(d) for d in draws])
    log_w = 0.5 * np.sum(draws * draws, axis=1) - 0.5 * delta_draws
    w = np.exp(log_w - np.max(log_w.real))
    summary = {
        "suite": args.suite, "q": args.q, "mode": args.mode, "cell": cell["name"],
        "n_hubbard": int(n), "n_normalization": int(p.n_normalization), "cache_status": status,
        "saddle_iterations": int(sad.n_iter), "E_star": [e_star.real, e_star.imag],
        "std_re_log_w": float(np.std(log_w.real)),
        "magnitude_ess_fraction": float(np.sum(np.abs(w)) ** 2 / (w.size * np.sum(np.abs(w) ** 2))),
        "signed_ess_fraction": float(abs(np.sum(w)) ** 2 / (w.size * np.sum(np.abs(w) ** 2))),
        "phase_only_ess_fraction": float(abs(np.mean(np.exp(1j * log_w.imag))) ** 2),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.suite}__q{args.q}__{args.mode}__{cell['name']}"
    np.savez(out / f"{stem}.npz", grid=grid, profiles=profiles, draws=draws, delta_draws=delta_draws)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 0


def log_gaussian(u, scale):
    n = u.shape[1]
    return -0.5 * np.sum(u * u, axis=1) / scale ** 2 - n * np.log(scale) - 0.5 * n * np.log(2 * np.pi)


def log_student_t(u, nu, scale):
    from scipy.special import gammaln

    n = u.shape[1]
    return (gammaln(0.5 * (nu + n)) - gammaln(0.5 * nu) - 0.5 * n * np.log(nu * np.pi)
            - n * np.log(scale) - 0.5 * (nu + n) * np.log1p(np.sum(u * u, axis=1) / (nu * scale ** 2)))


def draw_student_t(rng, size, n, nu, scale):
    return scale * rng.standard_normal((size, n)) / np.sqrt(rng.chisquare(nu, size) / nu)[:, None]


def _logmeanexp(x):
    from scipy.special import logsumexp

    return logsumexp(x) - np.log(x.size)


def predicted_ess(log_p, log_q0, log_candidate) -> dict:
    """Asymptotic ESS fractions of a candidate proposal, estimated from draws of a wider ``q0``.

    ``signed = |int p|^2 / int |p|^2/q`` and ``magnitude = (int |p|)^2 / int |p|^2/q``, each
    integral written as an expectation over ``q0``.  Unbiased while ``q0`` is heavier-tailed
    than both the target and the candidate.  For a candidate lighter-tailed than the target the
    true denominator is infinite, so the asymptotic ESS is zero; a finite direct run of such a
    proposal still reports a small, sample-size-dependent number.
    """
    log_signed_num = 2.0 * np.real(_logmeanexp(log_p - log_q0))
    log_mag_num = 2.0 * _logmeanexp(log_p.real - log_q0)
    log_den = _logmeanexp(2.0 * log_p.real - log_candidate - log_q0)
    return {"signed": float(np.exp(log_signed_num - log_den)),
            "magnitude": float(np.exp(log_mag_num - log_den))}


def _direct_ess(log_w) -> dict:
    w = np.exp(log_w - np.max(log_w.real))
    den = w.size * np.sum(np.abs(w) ** 2)
    return {"signed": float(abs(np.sum(w)) ** 2 / den), "magnitude": float(np.sum(np.abs(w)) ** 2 / den)}


SHAPE_CANDIDATES = (
    [("gaussian", None, s) for s in (0.4, 1.0, 1.5, 2.0, 3.0)]
    + [("student_t", nu, s) for nu in (1.0, 2.0, 3.0, 5.0, 10.0) for s in (1.0, 1.5, 2.0)]
)


def cmd_shapes(args) -> int:
    """Predict ESS for Gaussian widths and Student-t proposals on the saddle frame; verify the best."""
    started = time.perf_counter()
    cell, p, sad, status, e_star, delta = saddle_frame(args)
    n = p.n_hubbard
    rng = np.random.default_rng(1)
    nu0, s0 = 2.0, 2.0
    u0 = draw_student_t(rng, args.n_draws, n, nu0, s0)
    log_p = -0.5 * np.array([delta(u) for u in u0])
    log_q0 = log_student_t(u0, nu0, s0)
    halves = (slice(0, args.n_draws // 2), slice(args.n_draws // 2, args.n_draws))
    rows = []
    for kind, nu, scale in SHAPE_CANDIDATES:
        log_c = log_gaussian(u0, scale) if kind == "gaussian" else log_student_t(u0, nu, scale)
        row = {"kind": kind, "nu": nu, "scale": scale, **predicted_ess(log_p, log_q0, log_c)}
        row["signed_halves"] = [predicted_ess(log_p[h], log_q0[h], log_c[h])["signed"] for h in halves]
        rows.append(row)
    stable = [r for r in rows if min(r["signed_halves"]) > 0.5 * max(r["signed_halves"])]
    best = max(stable or rows, key=lambda r: r["signed"])
    verify = {}
    for label, (kind, nu, scale) in {"laplace_gaussian": ("gaussian", None, 1.0),
                                     "best": (best["kind"], best["nu"], best["scale"])}.items():
        if kind == "gaussian":
            u = scale * rng.standard_normal((args.n_verify, n))
            log_q = log_gaussian(u, scale)
        else:
            u = draw_student_t(rng, args.n_verify, n, nu, scale)
            log_q = log_student_t(u, nu, scale)
        log_w = -0.5 * np.array([delta(x) for x in u]) - log_q
        verify[label] = {"kind": kind, "nu": nu, "scale": scale, "n": args.n_verify, **_direct_ess(log_w)}
    summary = {"suite": args.suite, "q": args.q, "mode": args.mode, "cell": cell["name"],
               "n_hubbard": int(n), "cache_status": status, "E_star": [e_star.real, e_star.imag],
               "q0": {"kind": "student_t", "nu": nu0, "scale": s0, "n": args.n_draws},
               "candidates": rows, "verify": verify,
               "elapsed_s": round(time.perf_counter() - started, 1)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"shapes__{args.suite}__q{args.q}__{args.mode}__{cell['name']}"
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("suite", "mode", "cell", "verify", "elapsed_s")}), flush=True)
    return 0


RADII = (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 45, 70, 100, 150, 250, 400, 700, 1000)


def cmd_radial(args) -> int:
    """``Delta E^2(r d)`` along random unit frame directions ``d`` out to ``r = 1000``.

    In ``n`` frame dimensions the integrand's radial mass is ``r^(n-1) exp(-Re Delta E^2/2)``;
    a Gaussian proposal draws ``r ~ sqrt(n)``, so where that mass peaks decides whether it can work.
    """
    cell, p, sad, status, e_star, delta = saddle_frame(args)
    n = p.n_hubbard
    rng = np.random.default_rng(7)
    dirs = rng.standard_normal((args.n_directions, n))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    radii = np.asarray(RADII, dtype=float)
    prof = np.array([[delta(r * d) for r in radii] for d in dirs])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"radial__{args.suite}__q{args.q}__{args.mode}__{cell['name']}"
    np.savez(out / f"{stem}.npz", radii=radii, profiles=prof, n_hubbard=n)
    log_mass = (n - 1) * np.log(radii) + np.mean(-0.5 * prof.real, axis=0)
    print(json.dumps({"suite": args.suite, "mode": args.mode, "cell": cell["name"], "n_hubbard": int(n),
                      "radial_mass_peak_r": float(radii[np.argmax(log_mass)])}), flush=True)
    return 0


def cmd_plot_radial(args) -> int:
    runner = importlib.import_module("closure_JAM_truth_small.run_closure")
    runner._prepare_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    files = sorted(out.glob("radial__*.npz"))
    if not files:
        raise SystemExit(f"no radial profiles in {out}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, path in enumerate(files):
        d = np.load(path)
        r, prof, n = d["radii"], d["profiles"], int(d["n_hubbard"])
        _, suite, _q, mode, cell = path.stem.split("__")
        label = f"{suite.split('_')[1]} {mode} {cell} (n_hs={n})"
        re = -0.5 * prof.real
        axes[0].plot(r, np.mean(re, axis=0), "o-", color=f"C{i}", ms=3, label=label)
        axes[0].fill_between(r, re.min(axis=0), re.max(axis=0), color=f"C{i}", alpha=0.2)
        axes[1].plot(r, np.mean(np.abs(0.5 * prof.imag), axis=0), "o-", color=f"C{i}", ms=3, label=label)
        log_mass = (n - 1) * np.log(r) + np.mean(re, axis=0)
        axes[2].plot(r, log_mass - log_mass.max(), "o-", color=f"C{i}", ms=3, label=label)
        n_draw = n
        radius = np.sqrt(np.random.default_rng(0).chisquare(n_draw, 20000))
        lo, hi = np.percentile(radius, [2.5, 97.5])
        axes[2].axvspan(lo, hi, color=f"C{i}", alpha=0.12)
    rr = np.geomspace(0.5, 1000, 200)
    axes[0].plot(rr, -0.5 * rr ** 2, "k--", lw=1.5, label=r"Gaussian $-r^2/2$")
    axes[0].set_ylim(-150, 5)
    axes[0].set_ylabel(r"$-\frac{1}{2}\mathrm{Re}\,\Delta E^2$  (log |integrand|)")
    axes[1].set_ylabel(r"mean $|\frac{1}{2}\mathrm{Im}\,\Delta E^2|$  (phase)")
    axes[2].set_ylabel(r"log radial mass $(n-1)\log r - \frac{1}{2}\mathrm{Re}\,\Delta E^2$ (max = 0)")
    axes[2].set_ylim(-40, 2)
    axes[2].set_title("shaded: 95% of Laplace-proposal draw radii", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("radius r in steepest-descent frame units")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Radial structure of the H-S integrand on the affine saddle contour (8 random directions)")
    fig.tight_layout()
    print(*runner.save_figure_both(fig, out / "radial_profiles.png"), sep="\n")
    return 0


def cmd_plot(args) -> int:
    runner = importlib.import_module("closure_JAM_truth_small.run_closure")
    runner._prepare_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    stems = sorted(p.stem for p in out.glob("*.json"))
    if not stems:
        raise SystemExit(f"no profiles in {out}")
    fig, axes = plt.subplots(3, len(stems), figsize=(3.4 * len(stems), 9.0), squeeze=False)
    for col, stem in enumerate(stems):
        s = json.loads((out / f"{stem}.json").read_text())
        d = np.load(out / f"{stem}.npz")
        grid, profiles = d["grid"], d["profiles"]
        title = (f"{s['suite'].split('_')[1]} {s['mode']} {s['cell']}\n"
                 f"n_hs={s['n_hubbard']} n_norm={s['n_normalization']}")
        ax = axes[0, col]
        for k in range(profiles.shape[0]):
            ax.plot(grid, -0.5 * profiles[k].real, lw=1)
        ax.plot(grid, -0.5 * grid ** 2, "k--", lw=2, label=r"Gaussian $-u^2/2$")
        ax.set_ylim(-0.5 * grid.max() ** 2 * 1.6, 2.0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("frame coordinate $u_k$")
        if col == 0:
            ax.set_ylabel(r"$-\frac{1}{2}\,\mathrm{Re}\,\Delta E^2$ (log magnitude)")
            ax.legend(fontsize=8)
        ax = axes[1, col]
        for k in range(profiles.shape[0]):
            ax.plot(grid, -0.5 * profiles[k].imag, lw=1)
        ax.axhline(0.0, color="k", ls="--", lw=2)
        ax.set_xlabel("frame coordinate $u_k$")
        if col == 0:
            ax.set_ylabel(r"$-\frac{1}{2}\,\mathrm{Im}\,\Delta E^2$ (phase)")
        ax = axes[2, col]
        log_w = 0.5 * np.sum(d["draws"] ** 2, axis=1) - 0.5 * d["delta_draws"].real
        ax.hist(log_w - np.max(log_w), bins=40, color="C0")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mathrm{Re}\,\log w - \max$  (u ~ N(0, I))")
        ax.set_title(f"std log|w| = {s['std_re_log_w']:.3g}\n"
                     f"ESS mag {s['magnitude_ess_fraction']:.3g}  phase {s['phase_only_ess_fraction']:.3g}",
                     fontsize=9)
        if col == 0:
            ax.set_ylabel("draws")
    fig.suptitle("H-S integrand on the steepest-descent saddle contour vs its Gaussian proposal", y=1.0)
    fig.tight_layout()
    paths = runner.save_figure_both(fig, out / "contour_profiles.png")
    print(*paths, sep="\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--suite", required=True); p.add_argument("--q", default="2")
    p.add_argument("--mode", required=True); p.add_argument("--cell-json", required=True)
    p.add_argument("--out", required=True); p.add_argument("--cache", default=None)
    p.add_argument("--umax", type=float, default=4.0); p.add_argument("--n-grid", type=int, default=33)
    p.add_argument("--n-draws", type=int, default=200)
    p = sub.add_parser("shapes")
    p.add_argument("--suite", required=True); p.add_argument("--q", default="2")
    p.add_argument("--mode", required=True); p.add_argument("--cell-json", required=True)
    p.add_argument("--out", required=True); p.add_argument("--cache", default=None)
    p.add_argument("--n-draws", type=int, default=1500); p.add_argument("--n-verify", type=int, default=600)
    p = sub.add_parser("radial")
    p.add_argument("--suite", required=True); p.add_argument("--q", default="2")
    p.add_argument("--mode", required=True); p.add_argument("--cell-json", required=True)
    p.add_argument("--out", required=True); p.add_argument("--cache", default=None)
    p.add_argument("--n-directions", type=int, default=8)
    p = sub.add_parser("plot"); p.add_argument("--out", required=True)
    p = sub.add_parser("plot-radial"); p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": cmd_run, "shapes": cmd_shapes, "radial": cmd_radial, "plot": cmd_plot,
            "plot-radial": cmd_plot_radial}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
