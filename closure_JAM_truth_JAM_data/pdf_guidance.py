"""Evaluate the JAM truth and project it into the nine closure basis fields.

The truth is the real ``JAM24_PDF_proton_nlo`` LHAPDF set.  There is no
python-LHAPDF binding here, so JAM is evaluated with the small C++ helper
``closure_JAM_truth/lhapdf_dump.cc`` (compiled once against the local LHAPDF install).

For each *original* scale ``Q`` we read JAM at ``Q`` on the closure x grid,
keep the LHAPDF ``x f(x,Q)`` momentum densities, and combine the per-flavor
values into the C-even ``{t3,t8,sigma,t15,g}`` and C-odd valence
``{v3,v8,v,v15}`` fields.  The
resulting curves are used *directly* as the truth (assigned at ``Q0 = mc``).  An
analytic two-term form is additionally fit to each field and saved as reference
metadata only -- it is never the truth.

Run standalone to cache every truth member and report fit quality::

    python -m closure_JAM_truth.pdf_guidance
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from . import config as cfg


# -- LHAPDF evaluation -------------------------------------------------------


def compile_lhapdf_dump() -> Path:
    """Compile ``lhapdf_dump.cc`` against the local LHAPDF install (cached)."""
    tool_dir = cfg.DATA_ROOT / "_tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    exe = tool_dir / "pixel_closure_lhapdf_dump"
    stamp = tool_dir / "pixel_closure_lhapdf_dump.build"
    src = cfg.ROOT / "lhapdf_dump.cc"
    compiler = os.environ.get("CXX", "c++")
    signature = "\n".join((
        str(src.stat().st_mtime_ns), compiler, str(cfg.LHAPDF_PREFIX),
        str(cfg.LHAPDF_LIB_DIR), "rpath-v1",
    ))
    if exe.exists() and stamp.exists() and stamp.read_text() == signature:
        return exe
    cmd = [
        compiler, "-std=c++17", "-O2",
        f"-I{cfg.LHAPDF_PREFIX / 'include'}",
        str(src), "-o", str(exe),
        f"-L{cfg.LHAPDF_LIB_DIR}",
        f"-Wl,-rpath,{cfg.LHAPDF_LIB_DIR}",
        "-lLHAPDF",
    ]
    subprocess.run(cmd, check=True)
    stamp.write_text(signature)
    return exe


def ensure_lhapdf_data() -> Path:
    """Return an LHAPDF data root containing the JAM grid set.

    Prefers a set already present in the LHAPDF install; otherwise extracts the
    repo-vendored tarball into :data:`config.JAM_LHAPDF_DATA_ROOT` once.

    Returns:
        Path: Directory to use as ``LHAPDF_DATA_PATH``.

    Raises:
        FileNotFoundError: If neither the installed set nor the tarball exists.
    """
    installed = cfg.LHAPDF_DATA_DIR / cfg.JAM_SET / f"{cfg.JAM_SET}.info"
    if installed.exists():
        return cfg.LHAPDF_DATA_DIR

    set_dir = cfg.JAM_LHAPDF_DATA_ROOT / cfg.JAM_SET
    info_path = set_dir / f"{cfg.JAM_SET}.info"
    if info_path.exists():
        return cfg.JAM_LHAPDF_DATA_ROOT

    if not cfg.JAM_TARBALL.exists():
        raise FileNotFoundError(f"JAM LHAPDF tarball not found: {cfg.JAM_TARBALL}")

    cfg.JAM_LHAPDF_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    root = cfg.JAM_LHAPDF_DATA_ROOT.resolve()
    with tarfile.open(cfg.JAM_TARBALL, "r:gz") as tar:
        for member in tar.getmembers():
            dest = (cfg.JAM_LHAPDF_DATA_ROOT / member.name).resolve()
            if dest != root and root not in dest.parents:
                raise ValueError(f"unsafe LHAPDF archive member: {member.name}")
        tar.extractall(cfg.JAM_LHAPDF_DATA_ROOT)

    if not info_path.exists():
        raise FileNotFoundError(
            f"JAM LHAPDF archive did not create expected info file: {info_path}"
        )
    return cfg.JAM_LHAPDF_DATA_ROOT


def dump_jam(nodes: np.ndarray, *, q: float, member: int = cfg.JAM_MEMBER) -> np.ndarray:
    """Run the C++ dumper: rows ``[x, g, u, ubar, d, dbar, s, sbar, c, cbar]`` (x*f)."""
    exe = compile_lhapdf_dump()
    env = os.environ.copy()
    lib = str(cfg.LHAPDF_LIB_DIR)
    data_roots = [str(ensure_lhapdf_data()), str(cfg.LHAPDF_DATA_DIR)]
    data_roots.extend(
        path for path in env.get("LHAPDF_DATA_PATH", "").split(os.pathsep) if path
    )
    env["LHAPDF_DATA_PATH"] = os.pathsep.join(dict.fromkeys(data_roots))
    env["DYLD_LIBRARY_PATH"] = f"{lib}:{env.get('DYLD_LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    payload = "\n".join(f"{x:.17g}" for x in np.asarray(nodes)) + "\n"
    proc = subprocess.run(
        [str(exe), cfg.JAM_SET, str(int(member)), f"{float(q):.17g}"],
        input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, env=env,
    )
    if proc.returncode:
        raise RuntimeError(
            f"LHAPDF helper failed with exit code {proc.returncode}; "
            f"library_dir={cfg.LHAPDF_LIB_DIR}; data_dir={env['LHAPDF_DATA_PATH']}\n"
            f"{proc.stderr.strip()}"
        )
    rows = [ln for ln in proc.stdout.splitlines()
            if ln.strip() and ln.strip()[0] in "+-.0123456789"]
    return np.loadtxt(rows)


def project_basis(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Combine per-flavor ``x f`` rows into the nine basis **momentum densities**.

    The closure fields are the momentum densities ``q = x f`` (O(1) for every
    flavor), so we use the LHAPDF ``x f(x)`` values directly -- no division by x.
    The gluon field is ``x g`` (which is what enters the gluon operator/ITD).
    """
    # Columns are already x*f from LHAPDF xfxQ.
    g, u, ubar, d, dbar, s, sbar, c, cbar = (raw[:, i] for i in range(1, 10))
    up, dp, sp, cp = u + ubar, d + dbar, s + sbar, c + cbar
    um, dm, sm, cm = u - ubar, d - dbar, s - sbar, c - cbar
    return {
        "t3": up - dp,
        "t8": up + dp - 2.0 * sp,
        "sigma": up + dp + sp + cp,
        "t15": up + dp + sp - 3.0 * cp,
        "g": g,
        "v3": um - dm,
        "v8": um + dm - 2.0 * sm,
        "v": um + dm + sm + cm,
        "v15": um + dm + sm - 3.0 * cm,
    }


def jam_truth_curves(nodes, *, q_key: str, member: int = cfg.JAM_MEMBER):
    """Return ``(curves, meta)`` for JAM at original scale ``q_key``.

    ``curves`` maps each field to its truth values on ``nodes`` (assigned at the
    input scale ``Q0 = mc``).  The requested scale is evaluated directly, even
    below the set's advertised ``QMin``.
    """
    q = float(cfg.TRUTH_Q_CHOICES[q_key])
    raw = dump_jam(np.asarray(nodes, dtype=float), q=q, member=member)
    curves = project_basis(raw)
    meta = {
        "set": cfg.JAM_SET, "member": int(member), "q_key": q_key,
        "q_original": q, "q_effective": q,
        "below_set_qmin": bool(cfg.JAM_QMIN is not None and q < cfg.JAM_QMIN),
        "q0_input": cfg.Q0, "nf": cfg.NF,
    }
    return curves, meta


# -- analytic reference fit (metadata only) ----------------------------------


def two_term(x, p):
    """``N x^a (1-x)^b (1 + c sqrt(x) + d x) + N' x^a' (1-x)^b'``."""
    N, a, b, c, d, Np, ap, bp = p
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 1.0 - 1e-12)
    t1 = N * x**a * (1.0 - x) ** b * (1.0 + c * np.sqrt(x) + d * x)
    t2 = Np * x**ap * (1.0 - x) ** bp
    return t1 + t2


def fit_two_term(nodes, curve, *, seed: int = 0, n_starts: int = 6):
    """Least-squares fit of :func:`two_term` to ``curve`` (reference only).

    Returns ``(params, r2)``.  Robustness is secondary -- this is metadata, not
    the truth.  Residuals are weighted by ``1/(|curve| + scale)`` so the fit
    tracks shape across the x decades rather than only the large-x tail.
    """
    from scipy.optimize import least_squares

    x = np.asarray(nodes, dtype=float)
    y = np.asarray(curve, dtype=float)
    mask = np.isfinite(y)
    x, y = x[mask], y[mask]
    scale = 0.05 * (np.max(np.abs(y)) + 1e-12)
    w = 1.0 / (np.abs(y) + scale)
    lo = np.array([-50.0, -0.9, 0.01, -50.0, -50.0, -50.0, -0.9, 0.01])
    hi = np.array([50.0, 5.0, 30.0, 50.0, 50.0, 50.0, 5.0, 30.0])

    def resid(p):
        return (two_term(x, p) - y) * w

    rng = np.random.default_rng(seed)
    best_p, best_cost = None, np.inf
    for k in range(n_starts):
        if k == 0:
            p0 = np.array([1.0, -0.2, 3.0, 0.0, 0.0, 0.1, 0.5, 6.0])
        else:
            p0 = lo + rng.random(lo.size) * (hi - lo)
            p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
        try:
            res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        if res.cost < best_cost:
            best_cost, best_p = res.cost, res.x
    if best_p is None:
        return None, 0.0
    pred = two_term(x, best_p)
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-30
    return best_p.tolist(), 1.0 - ss_res / ss_tot


PARAM_NAMES = ("N", "a", "b", "c", "d", "Np", "ap", "bp")


def reference_fits(nodes, curves, *, seed: int = 0):
    """Analytic reference fit for every field -> ``{field: {params, r2}}``."""
    out = {}
    for i, (field, curve) in enumerate(curves.items()):
        params, r2 = fit_two_term(nodes, curve, seed=seed + i)
        out[field] = {
            "params": None if params is None else dict(zip(PARAM_NAMES, params)),
            "r2": r2,
        }
    return out


# -- standalone cache / report ----------------------------------------------


def main() -> None:
    cfg.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    grid = cfg.make_grid()
    nodes = np.asarray(list(cfg.make_fields().values())[0].nodes).reshape(-1)
    compile_lhapdf_dump()
    print(f"JAM set: {cfg.JAM_SET}  (member {cfg.JAM_MEMBER})  input Q0=mc={cfg.Q0}")
    for q_key in cfg.TRUTH_Q_CHOICES:
        curves, meta = jam_truth_curves(nodes, q_key=q_key)
        fits = reference_fits(nodes, curves, seed=1234)
        rec = {
            "meta": meta,
            "x_nodes": nodes.tolist(),
            "curves": {k: v.tolist() for k, v in curves.items()},
            "reference_fits": fits,
        }
        path = cfg.REFERENCE_DIR / f"jam_{cfg.truth_label(q_key)}.json"
        path.write_text(json.dumps(rec, indent=2))
        r2s = {k: round(v["r2"], 3) for k, v in fits.items()}
        flag = (
            "  [LHAPDF extrapolation below set QMin]"
            if meta["below_set_qmin"] else ""
        )
        print(f"  Q={meta['q_original']:>4} (eff {meta['q_effective']:.3f}){flag}"
              f"  ref-fit R^2: {r2s}")
    print(f"cached -> {cfg.REFERENCE_DIR}")


if __name__ == "__main__":
    main()
