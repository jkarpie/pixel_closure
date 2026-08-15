"""Small NNPDF-truth closure for the implemented massless NNLO DIS channels.

This is deliberately separate from the normal closure campaign.  It uses two
real F2, two real HERA NC, and two real HERA CC reduced-cross-section kinematic
points, folds the vendored NNPDF4.0 NNLO replica ensemble through Pixel's
fixed-nf=4 NNLO F2/FL/F3 operators, and rebuilds the public datasets from the
resulting pseudo-data.

Run explicitly with ``python -m closure_NNPDF_truth_small.nnlo_dis_smoke``.
It does not run the larger ``closure_*`` studies.  Generalized neutral-current
rows remain excluded because their three-loop FL ``fl11`` vector/axial charge
factor is not yet implemented.
"""

from __future__ import annotations

import json

import numpy as np

from pixel import data
from pixel.kernels.pqcd.contour import MellinContour

from . import config as cfg
from .generate import assemble_operator, fold_ensemble


def _maps(fields):
    return {basis: fields[field] for basis, field in cfg.EVEN_MAP.items()}


def _odd_maps(fields):
    return {basis: fields[field] for basis, field in cfg.ODD_MAP.items()}


def _ensemble():
    path = cfg.REFERENCE_DIR / "nnpdf_ensemble_truthQ_mc.npz"
    blob = np.load(path)
    return {field: np.asarray(blob[field], dtype=float) for field in cfg.ALL_FIELDS}


def _kinematics():
    manifest = json.loads(cfg.DIS_MANIFEST_PATH.read_text())["tables"]
    f2 = next(item for item in manifest if item["label"] == "slac_p_f2")
    sr = next(
        item for item in manifest if item["label"] == "hera_nc_ep_318_sigmar"
    )
    f2_rows = np.array([5, 12])
    sr_rows = np.array([9, 12])
    cc = next(item for item in manifest if item["label"] == "hera_cc_em_f3_proxy")
    cc_rows = np.array([2, 10])
    return {
        "f2": (
            np.asarray(f2["x"])[f2_rows],
            np.asarray(f2["Q2"])[f2_rows],
        ),
        "sigma_r": (
            np.asarray(sr["x"])[sr_rows],
            np.asarray(sr["Q2"])[sr_rows],
            float(np.asarray(sr["RS"])[sr_rows][0]),
        ),
        "sigma_r_cc": (
            np.asarray(cc["x"])[cc_rows],
            np.asarray(cc["Q2"])[cc_rows],
            float(np.asarray(cc["RS"])[cc_rows][0]),
            cc["beam"],
        ),
    }


def _theory(fields):
    return {
        "maps_to": _maps(fields),
        "target": "proton",
        "nf": 4,
        "Q20": cfg.Q0_2,
        "order": "NNLO",
        "matching_order": "NNLO",
        "mode": "truncated",
        "alphaS_MZ": cfg.ALPHAS_MZ,
        "mc": cfg.MC,
        "mb": cfg.MB,
        "momentum_density": True,
        "low_x_extensions": cfg.even_low_x_completions(),
        "flavour_mode": "fixed_nf",
        "fixed_nf": 4,
        "contour": MellinContour(c=1.1, zmax=80.0, panel=2.0, npts=16),
    }


def _close(builder, rows, ensemble, fields, theory, **observable):
    x, Q2 = rows[:2]
    placeholder = builder.from_arrays(
        x=x,
        Q2=Q2,
        mean=np.zeros_like(x),
        cov=np.eye(x.size),
        **observable,
        **theory,
    )
    operator = assemble_operator(placeholder, fields)
    replicas = fold_ensemble(operator, ensemble)
    central = replicas.mean(axis=0)
    covariance = np.atleast_2d(np.cov(replicas, rowvar=False))
    closed = builder.from_arrays(
        x=x,
        Q2=Q2,
        mean=central,
        cov=covariance,
        **observable,
        **theory,
    )
    prediction = fold_ensemble(assemble_operator(closed, fields), ensemble).mean(axis=0)
    residual = prediction - central
    return {
        "rows": int(x.size),
        "max_abs_closure_residual": float(np.max(np.abs(residual))),
        "matching_order": closed.metadata.meta["matching_order"],
        "coefficient_implementation": closed.metadata.meta[
            "dis_coefficient_function"
        ]["implementation"],
    }


def run() -> dict:
    """Run only the two-dataset NNLO smoke and return its closure metrics."""
    fields = cfg.make_fields()
    ensemble = _ensemble()
    points = _kinematics()
    theory = _theory(fields)
    return {
        "truth": "NNPDF40_nnlo_as_01180_1000 replica ensemble at Q=mc",
        "theory_domain": "massless MSbar, fixed_nf=4, F2/FL/F3 plus and minus",
        "f2": _close(data.F2, points["f2"], ensemble, fields, theory, matching=True),
        "sigma_r": _close(
            data.SigmaR,
            points["sigma_r"],
            ensemble,
            fields,
            theory,
            sqrt_s=points["sigma_r"][2],
        ),
        "sigma_r_cc": _close(
            data.ChargedCurrentSigmaR,
            points["sigma_r_cc"],
            ensemble,
            fields,
            theory,
            valence_maps_to=_odd_maps(fields),
            valence_low_x_extensions=cfg.odd_low_x_completions(),
            sqrt_s=points["sigma_r_cc"][2],
            beam_charge=points["sigma_r_cc"][3],
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
