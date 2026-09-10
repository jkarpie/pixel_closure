"""Truth members: NNPDF curves (used directly) plus an analytic reference fit.

A truth member is the ``NNPDF40_nnlo_as_01180_1000`` result read at one original
scale ``Q`` and *assigned as the truth at* ``Q0 = mc``.  The NNPDF-projected
curves are the truth; the analytic two-term fit is reference metadata only.

This module is a thin layer over :mod:`closure_NNPDF_truth.pdf_guidance` that caches the NNPDF
dump per member and exposes the truth on the shared field nodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as cfg
from . import pdf_guidance


def _cache_path(q_key: str) -> Path:
    return cfg.REFERENCE_DIR / f"nnpdf_{cfg.truth_label(q_key)}.json"


def build_truth(q_key: str, nodes, *, member: int = cfg.NNPDF_MEMBER,
                use_cache: bool = True) -> dict:
    """Return a truth record for original scale ``q_key`` on ``nodes``.

    Keys: ``meta`` (NNPDF set/member/scale), ``x_nodes``, ``curves`` (field ->
    truth values, the actual truth), ``reference_fits`` (analytic, metadata).
    Uses the :mod:`closure_NNPDF_truth.pdf_guidance` cache when node counts match.
    """
    nodes = np.asarray(nodes, dtype=float).reshape(-1)
    cache = _cache_path(q_key)
    if use_cache and member == cfg.NNPDF_MEMBER and cache.exists():
        rec = json.loads(cache.read_text())
        cached_q = float(rec.get("meta", {}).get("q_effective", np.nan))
        expected_q = float(cfg.TRUTH_Q_CHOICES[q_key])
        if len(rec.get("x_nodes", [])) == nodes.size and np.allclose(
            np.asarray(rec["x_nodes"]), nodes, rtol=1e-9, atol=0.0
        ) and np.isclose(cached_q, expected_q, rtol=0.0, atol=1e-12):
            rec["curves"] = {k: np.asarray(v) for k, v in rec["curves"].items()}
            return rec
    curves, meta = pdf_guidance.nnpdf_truth_curves(nodes, q_key=q_key, member=member)
    fits = pdf_guidance.reference_fits(nodes, curves, seed=1234)
    return {
        "meta": meta,
        "x_nodes": nodes.tolist(),
        "curves": curves,
        "reference_fits": fits,
    }


def truth_nodes(truth: dict) -> dict:
    """Return ``{field: truth curve on the field nodes}`` as float arrays."""
    return {k: np.asarray(v, dtype=float).reshape(-1)
            for k, v in truth["curves"].items()}


def serializable(truth: dict) -> dict:
    """A JSON-serializable copy of a truth record (curves -> lists)."""
    out = dict(truth)
    out["curves"] = {k: np.asarray(v).tolist() for k, v in truth["curves"].items()}
    return out
