"""One-row NNPDF-truth closure for the complete photon DY NNLO block.

This is deliberately separate from the ordinary closure campaign, whose DY
configuration remains LO. It selects one real E866 pp kinematic row, folds the
**mean** of the cached NNPDF4.0 NNLO replica ensemble -- one curve per field, not
the replica stack; see :func:`_truth_curves` -- through the public NNLO Drell--Yan
builder, rebuilds the same dataset from those pseudo-data, and reports the
builder-to-kernel closure residual. The smoke quadrature is intentionally
small; a second refinement is reported rather than silently presented as a
production-accuracy prediction.

Run explicitly with ``python -m closure_NNPDF_truth_small.nnlo_dy_smoke``.
It does not run or modify the larger ``closure_*`` studies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pixel import data

from . import config as cfg
from .generate import dy_central


def _redy_source() -> Path:
    configured = os.environ.get("PIXEL_REDY_SOURCE")
    candidates = [] if configured is None else [Path(configured)]
    candidates.append(cfg.REPO_ROOT.parent.parent / "paper_repo/drell-yan/ReDY.tgz")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "set PIXEL_REDY_SOURCE to the user-supplied ReDY v0.3 archive"
    )


def _truth_curves():
    """One truth curve per field: the replica-ensemble **mean**, not the stack.

    ``dy_central`` folds a single PDF truth (``outer(q_A, q_B)``); handing it the
    ``(n_replicas, n_nodes)`` stack made it an ensemble *second* moment
    ``E[q_A q_B] = E[q_A] E[q_B] + Cov``, which no single PDF reproduces.  That was
    ``S0-05``, fixed 2026-08-14, and ``dy_central`` now refuses a 2-D curve rather
    than flattening it.  The mean is taken here, once.

    The closure residual this module reports is a *ratio* of two folds computed the
    same way, so it is insensitive to which law is used; what the change moves is
    the absolute ``smoke_prediction``.
    """
    path = cfg.REFERENCE_DIR / "nnpdf_ensemble_truthQ_mc.npz"
    with np.load(path) as blob:
        return {
            field: np.asarray(blob[field], dtype=float).mean(axis=0)
            for field in cfg.ALL_FIELDS
        }


def _point():
    manifest = json.loads(cfg.DIS_MANIFEST_PATH.read_text())
    record = next(
        item
        for item in manifest["dy"]["tables"]
        if item["label"] == "dy_e866_pp"
    )
    row = 2
    return {
        "Q2": [float(record["Q2"][row])],
        "S": [float(record["S"][row])],
        "Y": [float(record["Y"][row])],
    }


def _dataset(source, point, *, nodes, panels, mean=None, cov=None):
    fields_a, fields_b = cfg.dy_field_maps("pp")
    return data.DrellYan(
        name="nnlo_dy_e866_pp_smoke",
        fields_A=fields_a,
        fields_B=fields_b,
        order="NNLO",
        nnlo_source=source,
        alpha_s=cfg.DY_ALPHA_S,
        alpha_em=cfg.DY_ALPHA_EM,
        nf=cfg.DY_NF,
        quad_n_z=nodes,
        quad_n_y=nodes,
        nnlo_z_panels=panels,
        nnlo_y_panels=panels,
        mean=mean,
        cov=cov,
        source="NNPDF4.0 replica truth + E866 kinematics",
        component="pp_virtual_photon_dsigma_dQ2dY",
        **point,
    )


def run() -> dict:
    """Run only this one-row NNLO smoke and return closure/refinement metrics."""
    source = _redy_source()
    fields = cfg.make_fields()
    truth = _truth_curves()
    point = _point()

    generated = _dataset(source, point, nodes=8, panels=5)
    central = dy_central(generated, truth, fields, "pp")
    covariance = np.diag((0.02 * np.abs(central) + cfg.EXP_ABS_FLOOR) ** 2)
    closed = _dataset(
        source,
        point,
        nodes=8,
        panels=5,
        mean=central,
        cov=covariance,
    )
    prediction = dy_central(closed, truth, fields, "pp")

    refined = _dataset(source, point, nodes=12, panels=8)
    refined_prediction = dy_central(refined, truth, fields, "pp")
    return {
        "truth": "NNPDF40_nnlo_as_01180_1000 replica-ensemble MEAN at Q=mc "
                 "(one curve per field; not an ensemble second moment)",
        "theory_domain": "massless MSbar, fixed_nf=4, pp virtual photon",
        "observable": "d sigma / (d Q^2 dY)",
        "source_sha256": generated.metadata.meta["nnlo_source"]["sha256"],
        "kinematics": point,
        "smoke_prediction": float(central[0]),
        "smoke_quadrature": {"nodes": 8, "panels": 5},
        "max_abs_closure_residual": float(np.max(np.abs(prediction - central))),
        "max_rel_closure_residual": float(
            np.max(np.abs((prediction - central) / central))
        ),
        "refined_prediction": float(refined_prediction[0]),
        "refined_quadrature": {"nodes": 12, "panels": 8},
        "smoke_to_refined_relative_shift": float(
            refined_prediction[0] / central[0] - 1.0
        ),
        "production_defaults": {
            "quad_n_z": 24,
            "quad_n_y": 16,
            "nnlo_z_panels": 16,
            "nnlo_y_panels": 10,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
