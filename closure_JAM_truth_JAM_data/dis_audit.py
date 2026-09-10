"""Audit the real fitpack DIS tables and write ``data/dis_manifest.json``.

The full-scale closure suite uses the **exact kinematic coverage of the real
analysis**: the DIS tables uncommented in ``fitpack_legacy/examples/input.py``
(:data:`closure_JAM_truth.config.ANALYSIS_IDIS_IDX`), with **all** retained rows
(no subsample) and the **real per-point statistical precision** ``rel_stat``.

This module also *scans every* unpolarized-DIS workbook under
:data:`closure_JAM_truth.config.FITPACK_IDIS` and records its status, so the
manifest is a full audit even though fake data are generated only for the
analysis tables:

* ``used``            -- an observable PIXEL models directly (F2, NC/CC sigma_r);
* ``unsupported``     -- ratios, R, parity-violating, heavy-flavor, etc.;
* ``skipped_by_cuts`` -- no rows survive the DIS cuts / no statistical error;
* ``missing``         -- the workbook is absent.

``manifest["tables"]`` holds the generated (analysis) records that
:mod:`closure_JAM_truth.generate` reads; ``manifest["audit"]`` holds the full
scan of every workbook.

Run standalone::

    python -m closure_JAM_truth.dis_audit
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pixel.data.experimental._dis_common import PROTON_MASS_GEV, dis_w2

from . import config as cfg


#: observable label (lower-cased) -> closure kind, for directly-modelled tables.
_DIRECT_OBS = {"f2": "f2", "sig_r": "sigma_r"}


def _col(df: pd.DataFrame, *names):
    """Return the first present column as a float array, else ``None``.

    Non-numeric cells (stray header/junk rows in some workbooks) are coerced to
    NaN so the downstream finite-mask drops them instead of raising.
    """
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").to_numpy(dtype=float)
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


def infer_kind(obs: str, current: str, idx) -> "str | None":
    """Closure ``kind`` for a table, or ``None`` if PIXEL cannot model it.

    ``obs`` sig_r with ``current`` CC is a charged-current reduced cross section;
    otherwise F2 -> ``f2`` and NC sig_r -> ``sigma_r``.
    """
    obs_l = (obs or "").strip().lower()
    cur_u = (current or "").strip().upper()
    is_cc_idx = isinstance(idx, int) and idx in cfg.CC_SIGMA_R_IDX
    if is_cc_idx or (obs_l == "sig_r" and cur_u == "CC"):
        return "sigma_r_cc"
    return _DIRECT_OBS.get(obs_l)


def classify(df: pd.DataFrame, idx) -> "tuple[str, str | None]":
    """Return ``(status, kind)`` for a table (before the DIS cuts)."""
    target = _str_field(df, "target").lower()
    if "/" in target:
        return "unsupported", None                     # ratio target (d/p, n/d, ...)
    kind = infer_kind(_str_field(df, "obs"), _str_field(df, "current"), idx)
    if kind is None:
        return "unsupported", None
    return "used", kind


def _stat_error(df: pd.DataFrame) -> "tuple[np.ndarray | None, str]":
    """Statistical error per row (``stat_u`` preferred; else quadrature of ``*_u``).

    Returns ``(sigma, column_label)``.  Correlated systematics (``*_c``) and the
    overall ``syst_u`` are excluded; only uncorrelated ``*_u`` columns feed the
    quadrature fallback.
    """
    if "stat_u" in df.columns:
        return _col(df, "stat_u"), "stat_u"
    u_cols = [c for c in df.columns
              if str(c).endswith("_u") and str(c) not in ("syst_u",)]
    if not u_cols:
        return None, "none"
    stack = np.column_stack([_col(df, c) for c in u_cols])
    return (np.sqrt(np.nansum(stack * stack, axis=1)),
            "quadrature(" + ",".join(str(c) for c in u_cols) + ")")


def retained_rows(df: pd.DataFrame, stat: "np.ndarray | None") -> np.ndarray:
    """Row indices surviving the closure DIS cuts (Q2 >= input, W2 > cut)."""
    x = _col(df, "X", "x")
    q2 = _col(df, "Q2", "q2")
    value = _col(df, "value")
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


def subsample(idx: np.ndarray, max_points, key: "np.ndarray | None" = None) -> np.ndarray:
    """Thin retained rows to at most ``max_points`` (no-op when it is ``None``)."""
    if max_points is None or idx.size <= max_points:
        return idx
    order = np.argsort(key) if key is not None else np.arange(idx.size)
    ordered = idx[order]
    take = np.unique(np.rint(np.linspace(0, ordered.size - 1, max_points)).astype(int))
    return ordered[take]


def _source_for(idx: int) -> Path:
    return cfg.FITPACK_IDIS / f"{idx}.xlsx"


def _label_for(idx: int, kind: str, target: str) -> str:
    override = cfg.EXP_LABEL_OVERRIDES.get(idx)
    if override:
        return override
    tgt = (target or "x").replace("/", "")
    return f"idis_{idx}_{tgt}_{kind}"


def audit_table(idx: int) -> dict:
    """Full audit + kinematics of one analysis table (generated when modelable)."""
    source = _source_for(idx)
    rec = {"idx": idx, "source": str(source)}
    if not source.exists():
        rec["status"] = "missing"
        return rec
    df = pd.read_excel(source)
    rec["n_raw"] = int(len(df))
    rec["obs"] = _str_field(df, "obs")
    rec["target"] = _str_field(df, "target")
    rec["current"] = _str_field(df, "current")
    rec["beam"] = _str_field(df, "lepton beam")
    rec["experiment"] = _str_field(df, "col")

    status, kind = classify(df, idx)
    rec["kind"] = kind
    rec["label"] = _label_for(idx, kind or "unsupported", rec["target"])

    if status != "used":
        rec["status"] = status
        return rec

    stat, err_col = _stat_error(df)
    keep = retained_rows(df, stat)
    if keep.size == 0:
        rec["status"] = "skipped_by_cuts"
        rec["error_column"] = err_col
        return rec

    x_all = _col(df, "X", "x")
    keep = subsample(keep, cfg.MAX_POINTS_PER_EXP_DATASET, key=x_all[keep])
    sel = df.iloc[keep]
    x = np.asarray(sel["X"] if "X" in sel else sel["x"], dtype=float)
    q2 = np.asarray(sel["Q2"] if "Q2" in sel else sel["q2"], dtype=float)
    value = np.asarray(sel["value"], dtype=float)
    sig = stat[keep]

    rec["status"] = status
    rec["error_column"] = err_col
    rec["retained_rows"] = keep.tolist()
    rec["n_used"] = int(keep.size)
    rec["x"] = x.tolist()
    rec["Q2"] = q2.tolist()
    rec["W2"] = dis_w2(x, q2, m_h=PROTON_MASS_GEV).tolist()
    rec["real_value"] = value.tolist()
    rec["real_stat_u"] = sig.tolist()
    rel = np.abs(sig) / np.abs(value)
    rec["rel_stat"] = rel.tolist()
    rec["x_range"] = [float(x.min()), float(x.max())]
    rec["Q2_range"] = [float(q2.min()), float(q2.max())]
    rec["rel_stat_range"] = [float(np.min(rel)), float(np.max(rel))]

    record_nuisances(rec, df, keep, value, rel)

    rs = _col(sel, "RS")
    if rs is not None:
        rec["RS"] = rs.tolist()
    y = _col(sel, "Y", "*Y")
    if y is not None and np.all(np.isfinite(y)):
        rec["Y"] = y.tolist()
    return rec


def scan_all() -> "list[dict]":
    """Light status scan of *every* workbook under FITPACK_IDIS (audit report)."""
    out = []
    for path in sorted(cfg.FITPACK_IDIS.glob("*.xlsx")):
        stem = path.stem
        entry = {"file": path.name, "stem": stem}
        try:
            df = pd.read_excel(path)
        except Exception as exc:  # noqa: BLE001 - record and move on
            entry["status"] = "read_error"
            entry["error"] = str(exc)[:120]
            out.append(entry)
            continue
        idx = int(stem) if stem.isdigit() else stem
        entry["idx"] = idx
        entry["obs"] = _str_field(df, "obs")
        entry["target"] = _str_field(df, "target")
        entry["current"] = _str_field(df, "current")
        entry["n_raw"] = int(len(df))
        status, kind = classify(df, idx)
        stat, _ = _stat_error(df)
        n_ret = int(retained_rows(df, stat).size)
        if status == "used" and n_ret == 0:
            status = "skipped_by_cuts"
        entry["kind"] = kind
        entry["status"] = status
        entry["n_retained"] = n_ret
        entry["in_analysis"] = (isinstance(idx, int) and idx in cfg.ANALYSIS_IDIS_IDX)
        out.append(entry)
    return out


def build_manifest() -> dict:
    """Audit the analysis tables (generated) and scan every workbook (report)."""
    tables = [audit_table(idx) for idx in cfg.ANALYSIS_IDIS_IDX]
    audit = scan_all()

    gen_counts: dict[str, int] = {}
    for r in tables:
        gen_counts[r["status"]] = gen_counts.get(r["status"], 0) + 1
    scan_counts: dict[str, int] = {}
    for r in audit:
        scan_counts[r["status"]] = scan_counts.get(r["status"], 0) + 1

    return {
        "cuts": {"Q2_min": cfg.DIS_Q2_MIN, "Q2_max": cfg.DIS_Q2_MAX,
                 "W2_min": cfg.DIS_W2_MIN,
                 "max_points_per_dataset": cfg.MAX_POINTS_PER_EXP_DATASET},
        "analysis_idx": list(cfg.ANALYSIS_IDIS_IDX),
        "generated_status_counts": gen_counts,
        "audit_status_counts": scan_counts,
        "tables": tables,          # generated set (analysis list); generate.py reads this
        "audit": audit,            # full scan of every workbook (report only)
    }


def main() -> None:
    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    cfg.DIS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"DIS audit -> {cfg.DIS_MANIFEST_PATH}")
    print(f"generated (analysis) status: {manifest['generated_status_counts']}")
    print(f"full-scan status ({len(manifest['audit'])} workbooks): "
          f"{manifest['audit_status_counts']}")
    for r in manifest["tables"]:
        line = f"  {r['idx']} {r.get('label', ''):<26} {r['status']:<16}"
        if r.get("n_used"):
            line += (f" n={r['n_used']:>4}  x[{r['x_range'][0]:.3g},{r['x_range'][1]:.3g}]"
                     f"  Q2[{r['Q2_range'][0]:.3g},{r['Q2_range'][1]:.3g}]"
                     f"  rel_stat[{r['rel_stat_range'][0]*100:.2g}%,"
                     f"{r['rel_stat_range'][1]*100:.2g}%]")
        print(line)


if __name__ == "__main__":
    main()
