"""Audit the real fitpack DIS tables and write ``data/dis_manifest.json``.

The closure experimental fake data must use *real* kinematics and the *real*
point-by-point statistical precision.  This module scans every unpolarized-DIS
Excel workbook under :data:`closure_NNPDF_truth_small.config.FITPACK_IDIS`, records what it finds,
applies the closure DIS cuts, and marks each table's status:

* ``used``        -- an observable PIXEL models directly (F2, NC/CC sigma_r);
* ``unsupported`` -- ratios, heavy-flavor sigma_red, parity-violating, etc.;
* ``skipped_by_cuts`` -- no rows survive the DIS cuts.

The manifest distinguishes the raw table from the retained (post-cut) rows;
fake data are generated only at retained rows, but the full raw summary is kept.

Run standalone::

    python -m closure_NNPDF_truth_small.dis_audit
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pixel.data.experimental._dis_common import PROTON_MASS_GEV, dis_w2

from . import config as cfg


#: observable label -> closure kind, for tables PIXEL can model directly.
_DIRECT_OBS = {"F2": "f2", "sig_r": "sigma_r"}

def _col(df: pd.DataFrame, *names):
    """Return the first present column as a float array, else ``None``."""
    for n in names:
        if n in df.columns:
            return np.asarray(df[n], dtype=float)
    return None


def _str_field(df: pd.DataFrame, name: str) -> str:
    if name not in df.columns:
        return ""
    vals = [str(v) for v in df[name].dropna().unique()]
    return vals[0] if vals else ""


def _absolute_uncertainty(df: pd.DataFrame, column, value: np.ndarray) -> np.ndarray:
    """One ``*_c`` column as an absolute per-row uncertainty.

    Follows the fitpack/commondata convention (see
    :mod:`pixel.data.experimental.commondata`): a ``%`` anywhere in the column
    name means the entry is a **percentage of value**, otherwise it is already
    absolute.  Non-numeric cells become 0 so a source with blank rows simply does
    not act there.
    """
    raw = np.nan_to_num(pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float))
    if "%" in str(column):
        return value * raw / 100.0
    return raw


def correlated_columns(df: pd.DataFrame) -> "tuple[list, list]":
    """Split the ``*_c`` columns into ``(normalization, correlated)`` names.

    A ``*_c`` whose name contains ``norm`` is the overall multiplicative
    normalization; every other ``*_c`` is an ordinary correlated systematic
    source (HERA ships 169 of them per table, BCDMS 5, NMC 11).
    """
    cs = [c for c in df.columns if "_c" in str(c)]
    norm = [c for c in cs if "norm" in str(c).lower()]
    return norm, [c for c in cs if c not in norm]


def record_nuisances(rec, df, keep, value, rel) -> None:
    """Record a table's real normalization + correlated systematics into ``rec``.

    Both are stored *relative* to the real value so generation can rescale them
    onto the folded fake central, keeping the closure self-consistent.  The
    per-source vectors are far too bulky for JSON (HERA: 169 sources x ~500 rows
    per table), so they go to an ``.npz`` sidecar under
    :data:`config.DIS_SYS_DIR` and the manifest keeps the names plus a summary.

    Args:
        rec: The audit record to update in place.
        df: The full table.
        keep: Row indices retained by the cuts/subsample.
        value: Real measured values on the retained rows.
        rel: Relative statistical error on the retained rows.
    """
    norm_cols, corr_cols = correlated_columns(df)
    all_values = _col(df, "value")
    if norm_cols:
        if len(norm_cols) > 1:
            raise ValueError(f"{rec.get('idx')}: multiple norm columns {norm_cols}")
        norm_abs = _absolute_uncertainty(df, norm_cols[0], all_values)[keep]
        norm_rel = np.abs(norm_abs) / np.abs(value)
        finite = norm_rel[np.isfinite(norm_rel)]
        if finite.size:
            # A normalization is one scalar per table by construction.
            rec["rel_norm"] = float(np.median(finite))
            rec["norm_column"] = str(norm_cols[0])
            spread = float(np.max(finite) - np.min(finite))
            if spread > 1e-6:
                rec["rel_norm_spread"] = spread
    if corr_cols:
        rows = [
            np.abs(_absolute_uncertainty(df, c, all_values)[keep]) / np.abs(value)
            for c in corr_cols
        ]
        corr_rel = np.nan_to_num(np.vstack(rows))          # (n_sys, n_used)
        cfg.DIS_SYS_DIR.mkdir(parents=True, exist_ok=True)
        out = cfg.DIS_SYS_DIR / f"{rec['label']}.npz"
        np.savez_compressed(out, relative=corr_rel,
                            names=np.array([str(c) for c in corr_cols]))
        rec["n_correlated"] = int(corr_rel.shape[0])
        rec["correlated_file"] = out.name
        rec["correlated_names"] = [str(c) for c in corr_cols]
        # Quadrature size relative to the statistical error: how much these matter.
        quad = np.sqrt((corr_rel ** 2).sum(axis=0))
        rec["rel_corr_quadrature_median"] = float(np.median(quad))
        rec["corr_over_stat_median"] = float(np.median(quad) / np.median(rel))


def classify(df: pd.DataFrame, spec) -> str:
    """Return the audit status for a table given its preset spec."""
    obs = (_str_field(df, "obs") or spec.obs).strip()
    target = _str_field(df, "target").lower()
    # ratio targets like "d/p" or "n/d" are unsupported.
    if "/" in target:
        return "unsupported"
    if spec.kind == "sigma_r_cc" and _str_field(df, "current").strip().upper() != "CC":
        return "unsupported"
    if obs not in _DIRECT_OBS:
        return "unsupported"
    return "used"


def retained_rows(df: pd.DataFrame) -> np.ndarray:
    """Row indices surviving the closure DIS cuts (Q2 >= input, W2 > cut)."""
    x = _col(df, "X", "x")
    q2 = _col(df, "Q2", "q2")
    value = _col(df, "value")
    stat = _col(df, "stat_u")
    if x is None or q2 is None or value is None or stat is None:
        return np.array([], dtype=int)
    w2 = dis_w2(x, q2, m_h=PROTON_MASS_GEV)
    mask = (
        np.isfinite(x) & np.isfinite(q2) & np.isfinite(value) & np.isfinite(stat)
        & (x > 0.0) & (x < 1.0)
        & (q2 >= cfg.DIS_Q2_MIN) & (w2 > cfg.DIS_W2_MIN)
        & (value != 0.0) & (stat > 0.0)
    )
    q2_max = getattr(cfg, "DIS_Q2_MAX", None)
    if q2_max is not None and np.isfinite(q2_max):
        mask &= q2 <= q2_max
    return np.flatnonzero(mask)


def subsample(idx: np.ndarray, max_points: int, key: np.ndarray | None = None) -> np.ndarray:
    """Thin retained rows to at most ``max_points``, preserving the extremes.

    When ``key`` (one value per retained row, e.g. the row's ``x``) is given the
    rows are ordered by it before even thinning, so the lowest- and highest-key
    rows are always kept -- the fitpack tables are not x-sorted, so without this
    the interior minimum-x rows (e.g. NMC's low-x band) get dropped.  With no
    ``key`` the rows are thinned in their existing order.
    """
    if idx.size <= max_points:
        return idx
    order = np.argsort(key) if key is not None else np.arange(idx.size)
    ordered = idx[order]
    take = np.unique(np.rint(np.linspace(0, ordered.size - 1, max_points)).astype(int))
    return ordered[take]


def audit_table(spec) -> dict:
    """Audit one fitpack table into a manifest record."""
    source = cfg.FITPACK_IDIS / f"{spec.idx}.xlsx"
    rec = {
        "idx": spec.idx, "label": spec.label, "kind": spec.kind,
        "obs": spec.obs, "target": spec.target, "source": str(source),
    }
    if not source.exists():
        rec["status"] = "missing"
        return rec
    df = pd.read_excel(source)
    rec["n_raw"] = int(len(df))
    rec["current"] = _str_field(df, "current")
    rec["beam"] = _str_field(df, "lepton beam")
    rec["experiment"] = _str_field(df, "col")
    status = classify(df, spec)

    idx = retained_rows(df)
    if idx.size == 0 and status == "used":
        status = "skipped_by_cuts"
    rec["status"] = status
    rec["n_retained_precut"] = int(idx.size)

    if status == "used" and idx.size:
        # Order the surviving rows by x so the subsample preserves the low-x and
        # high-x endpoints (fitpack tables are x-binned, not globally x-sorted).
        x_all = _col(df, "X", "x")
        keep = subsample(idx, cfg.MAX_POINTS_PER_EXP_DATASET, key=x_all[idx])
        sel = df.iloc[keep]
        x = np.asarray(sel["X"], dtype=float)
        q2 = np.asarray(sel["Q2"], dtype=float)
        value = np.asarray(sel["value"], dtype=float)
        stat = np.asarray(sel["stat_u"], dtype=float)
        rs = _col(sel, "RS")
        rec["retained_rows"] = keep.tolist()
        rec["n_used"] = int(keep.size)
        rec["x"] = x.tolist()
        rec["Q2"] = q2.tolist()
        rec["W2"] = dis_w2(x, q2, m_h=PROTON_MASS_GEV).tolist()
        rec["real_value"] = value.tolist()
        rec["real_stat_u"] = stat.tolist()
        rec["rel_stat"] = (np.abs(stat) / np.abs(value)).tolist()
        rec["x_range"] = [float(x.min()), float(x.max())]
        rec["Q2_range"] = [float(q2.min()), float(q2.max())]
        rec["rel_stat_range"] = [
            float(np.min(np.abs(stat) / np.abs(value))),
            float(np.max(np.abs(stat) / np.abs(value))),
        ]
        record_nuisances(rec, df, keep, value, np.abs(stat) / np.abs(value))
        rec["error_column"] = "stat_u"
        if rs is not None:
            rec["RS"] = rs.tolist()
    return rec


def build_manifest() -> dict:
    """Audit every preset DIS table into a manifest dict."""
    records = [audit_table(spec) for spec in cfg.EXP_SPECS]
    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "cuts": {"Q2_min": cfg.DIS_Q2_MIN, "Q2_max": cfg.DIS_Q2_MAX,
                 "W2_min": cfg.DIS_W2_MIN,
                 "max_points_per_dataset": cfg.MAX_POINTS_PER_EXP_DATASET},
        "status_counts": counts,
        "tables": records,
    }


def main() -> None:
    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    cfg.DIS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"DIS audit -> {cfg.DIS_MANIFEST_PATH}")
    print(f"status counts: {manifest['status_counts']}")
    for r in manifest["tables"]:
        line = f"  {r['idx']} {r['label']:<24} {r['status']:<16}"
        if r.get("n_used"):
            line += (f" n={r['n_used']:>3}  x[{r['x_range'][0]:.3g},{r['x_range'][1]:.3g}]"
                     f"  Q2[{r['Q2_range'][0]:.3g},{r['Q2_range'][1]:.3g}]"
                     f"  rel_stat[{r['rel_stat_range'][0]*100:.2g}%,"
                     f"{r['rel_stat_range'][1]*100:.2g}%]")
        print(line)


if __name__ == "__main__":
    main()
