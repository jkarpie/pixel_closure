"""Audit the fitpack fixed-target Drell-Yan tables into the closure manifest.

Drell-Yan constrains the light-quark sea (dbar-ubar) that inclusive DIS pins
only weakly.  This scans every workbook under
:data:`closure_NNPDF_truth_small.config.FITPACK_DY`, records each table's status, and for
the modelable analysis tables (E866 ``pp`` 10001 and ``pd`` 10002) converts the
kinematics to what :func:`pixel.data.DrellYan` wants and records the real
per-point statistical precision:

* ``used``        -- a Drell-Yan cross section PIXEL models (``pp``/``pd``);
* ``unsupported`` -- nuclear ``pA`` (no nuclear DY builder) or a cross-section
                     ratio (``pd/pp``, no ratio builder);
* ``skipped``     -- not a data table (e.g. an acceptance table);
* ``missing``     -- the workbook is absent.

Kinematics conversion (E866 fixed target): ``S = RS^2``, ``Q2 = Rtau * S``, and
``xF -> Y`` via ``x1,2 = (+/- xF + sqrt(xF^2 + 4 tau)) / 2``, ``Y = 0.5
ln(x1/x2)``; rows with ``x1 >= 1`` or ``x2 >= 1`` are cut.  ``rel_stat =
|stat_u / value|`` sizes the (diagonal) error, as in the DIS audit.

The records are written into ``data/dis_manifest.json`` under a ``"dy"`` key by
:func:`augment_manifest` (called from ``dis_audit`` / ``generate``), or the
module can be run standalone to preview the audit.

Run standalone::

    python -m closure_NNPDF_truth_small.dy_audit
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as cfg


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").to_numpy(dtype=float)
    return None


def _str_field(df, name):
    if name not in df.columns:
        return ""
    vals = [str(v) for v in df[name].dropna().unique()]
    return vals[0] if vals else ""


def _relative_normalization(df, value):
    """Relative overall normalization uncertainty from the table's norm column.

    E866 ships ``%norm_c`` (7, i.e. 7% of ``value``); the ``%`` marks a percentage
    exactly as in :mod:`pixel.data.experimental.commondata`, and a column without
    it is absolute and divided by ``value``.  Returns ``None`` when the table has
    no normalization column.
    """
    for col in df.columns:
        name = str(col).strip().lower()
        if "norm" not in name or "_c" not in name:
            continue
        raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if name.startswith("%") or "%" in name:
            rel = raw / 100.0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(raw) / np.abs(value)
        rel = rel[np.isfinite(rel)]
        if rel.size == 0:
            return None
        # One scalar per table: these columns are uniform by construction, so the
        # median is the value and a spread would mean the column is not a norm.
        return float(np.median(rel))
    return None


def convert_kinematics(rs, tau, xF):
    """(RS, Rtau=tau, xF) -> (S, Q2, Y, x1, x2) for fixed-target Drell-Yan."""
    rs = np.asarray(rs, dtype=float)
    tau = np.asarray(tau, dtype=float)
    xF = np.asarray(xF, dtype=float)
    S = rs * rs
    Q2 = tau * S
    root = np.sqrt(xF * xF + 4.0 * tau)
    x1 = 0.5 * (xF + root)
    x2 = 0.5 * (-xF + root)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = 0.5 * np.log(x1 / x2)
    return S, Q2, Y, x1, x2


def audit_table(idx: int) -> dict:
    """Audit one modelable DY table (kinematics conversion + rel_stat)."""
    reaction, label = cfg.DY_ANALYSIS[idx]
    source = cfg.FITPACK_DY / f"{idx}.xlsx"
    rec = {"idx": idx, "reaction": reaction, "label": label, "source": str(source),
           "kind": "drell_yan"}
    if not source.exists():
        rec["status"] = "missing"
        return rec
    df = pd.read_excel(source)
    rec["n_raw"] = int(len(df))
    rec["obs"] = _str_field(df, "obs")

    rs = _col(df, "RS")
    tau = _col(df, "Rtau", "rtau", "tau")
    xF = _col(df, "xF")
    value = _col(df, "value")
    stat = _col(df, "stat_u")
    if any(v is None for v in (rs, tau, xF, value, stat)):
        rec["status"] = "unsupported"
        rec["reason"] = "missing RS/Rtau/xF/value/stat_u column"
        return rec

    S, Q2, Y, x1, x2 = convert_kinematics(rs, tau, xF)
    mask = (
        np.isfinite(S) & np.isfinite(Q2) & np.isfinite(Y)
        & np.isfinite(value) & np.isfinite(stat)
        & (value != 0.0) & (stat > 0.0)
        & (x1 > 0.0) & (x1 < cfg.DY_X_MAX) & (x2 > 0.0) & (x2 < cfg.DY_X_MAX)
        & (Q2 >= cfg.DY_Q2_MIN)
    )
    keep = np.flatnonzero(mask)
    if keep.size == 0:
        rec["status"] = "skipped_by_cuts"
        return rec

    # Bilinear DY memory scales with the row count -> bin to DY_MAX_ROWS, ordered
    # by x1 so the subsample preserves the kinematic (sea x-fraction) spread.
    rec["n_retained_precut"] = int(keep.size)
    max_rows = getattr(cfg, "DY_MAX_ROWS", None)
    if max_rows is not None and keep.size > max_rows:
        order = keep[np.argsort(x1[keep])]
        take = np.unique(np.rint(np.linspace(0, order.size - 1, max_rows)).astype(int))
        keep = np.sort(order[take])

    rel = np.abs(stat[keep]) / np.abs(value[keep])
    rec["status"] = "used"
    rec["n_used"] = int(keep.size)
    rec["retained_rows"] = keep.tolist()
    rec["S"] = S[keep].tolist()
    rec["Q2"] = Q2[keep].tolist()
    rec["Y"] = Y[keep].tolist()
    rec["x1"] = x1[keep].tolist()
    rec["x2"] = x2[keep].tolist()
    rec["real_value"] = value[keep].tolist()
    rec["real_stat_u"] = stat[keep].tolist()
    rec["rel_stat"] = rel.tolist()
    # The real overall normalization uncertainty (E866: 7%).  Generation injects a
    # known multiple of it and the fit treats it as a nuisance -- see config.
    rel_norm = _relative_normalization(df, value)
    if rel_norm is not None:
        rec["rel_norm"] = rel_norm
    rec["x1_range"] = [float(x1[keep].min()), float(x1[keep].max())]
    rec["x2_range"] = [float(x2[keep].min()), float(x2[keep].max())]
    rec["Q2_range"] = [float(Q2[keep].min()), float(Q2[keep].max())]
    rec["rel_stat_range"] = [float(rel.min()), float(rel.max())]
    return rec


def scan_all() -> "list[dict]":
    """Status scan of every DY workbook (audit report)."""
    out = []
    for path in sorted(cfg.FITPACK_DY.glob("*.xlsx")):
        stem = path.stem
        entry = {"file": path.name, "stem": stem}
        try:
            df = pd.read_excel(path)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "read_error"
            entry["error"] = str(exc)[:120]
            out.append(entry)
            continue
        idx = int(stem) if stem.isdigit() else stem
        obs = _str_field(df, "obs")
        reaction = _str_field(df, "reaction")
        entry.update({"idx": idx, "obs": obs, "reaction": reaction,
                      "n_raw": int(len(df))})
        if isinstance(idx, int) and idx in cfg.DY_ANALYSIS:
            entry["status"] = "used"
        elif "/" in reaction or "/" in obs:
            entry["status"] = "unsupported"          # cross-section ratio
        elif reaction in ("pA", "pa") or _str_field(df, "target") not in ("", "p", "d"):
            entry["status"] = "unsupported"          # nuclear target
        elif not obs:
            entry["status"] = "skipped"              # not a data table
        else:
            entry["status"] = "unsupported"
        entry["in_analysis"] = isinstance(idx, int) and idx in cfg.DY_ANALYSIS
        out.append(entry)
    return out


def build_dy_manifest() -> dict:
    """Audit the analysis DY tables (generated) and scan every workbook."""
    tables = [audit_table(idx) for idx in cfg.DY_ANALYSIS]
    audit = scan_all()
    counts: dict[str, int] = {}
    for r in audit:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "cuts": {"x_max": cfg.DY_X_MAX, "Q2_min": cfg.DY_Q2_MIN},
        "analysis_idx": list(cfg.DY_ANALYSIS),
        # Marks a manifest written by an audit that reads the normalization
        # column; dy_layout re-audits when it is missing, so a cached manifest
        # from before normalizations were recorded refreshes itself.
        "has_norm_audit": True,
        "audit_status_counts": counts,
        "tables": tables,
        "audit": audit,
    }


def augment_manifest() -> dict:
    """Add the ``"dy"`` audit to the existing DIS manifest and rewrite it."""
    path = cfg.DIS_MANIFEST_PATH
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["dy"] = build_dy_manifest()
    path.write_text(json.dumps(manifest, indent=2))
    return manifest["dy"]


def main() -> None:
    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    dy = augment_manifest()
    print(f"DY audit -> {cfg.DIS_MANIFEST_PATH} (dy section)")
    print(f"full-scan status: {dy['audit_status_counts']}")
    for r in dy["tables"]:
        line = f"  {r['idx']} {r.get('label', ''):<14} {r['status']:<16}"
        if r.get("n_used"):
            line += (f" n={r['n_used']:>3}  x1[{r['x1_range'][0]:.3g},{r['x1_range'][1]:.3g}]"
                     f"  x2[{r['x2_range'][0]:.3g},{r['x2_range'][1]:.3g}]"
                     f"  Q2[{r['Q2_range'][0]:.3g},{r['Q2_range'][1]:.3g}]"
                     f"  rel_stat[{r['rel_stat_range'][0]*100:.2g}%,"
                     f"{r['rel_stat_range'][1]*100:.2g}%]")
        if r.get("rel_norm") is not None:
            line += f"  norm {r['rel_norm']*100:.3g}%"
        print(line)


if __name__ == "__main__":
    main()
