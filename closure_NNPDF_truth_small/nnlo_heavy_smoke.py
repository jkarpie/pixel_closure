"""Two-row NNPDF-truth closure for exact two-loop heavy-pair DIS.

This deliberately small route selects two published HERA combined charm
reduced-cross-section rows from NNPDF commondata, folds the vendored NNPDF4.0
NNLO replica ensemble through Pixel's fixed-``nf=3`` FFNS
``sigma_r^ccbar`` operator, and rebuilds the public dataset from those
pseudo-data.  The light-flavour combinations needed by the two-loop massive
coefficients are reconstructed exactly from the suite's ``nf=4`` canonical
basis at ``Q0=mc`` before fixed-flavour evolution. The same two kinematic rows
also close the complete order-``alpha_s`` S-ACOT-chi and FONLL-A
``sigma_r^ccbar`` builders, including their independently assembled ``F2`` and
``FL`` components and the exact active-charm basis identity.

Run explicitly with ``python -m closure_NNPDF_truth_small.nnlo_heavy_smoke``.
It does not run the larger ``closure_*`` studies.  This is an internal closure
of the implemented FFNS operator, not NNPDF theory-200 FONLL-C/TMC parity.
"""

from __future__ import annotations

import json

import numpy as np

from pixel import data
from pixel.core.model import Field

from . import config as cfg
from .generate import assemble_operator, fold_ensemble


_COMMONDATA = cfg.NNPDF_COMMONDATA / "HERA_NC_318GEV_EAVG"
_OBSERVABLE = "CHARM-SIGMARED"
_ROWS = np.array([15, 45], dtype=int)


def _fields_and_ensemble():
    """Add exact three-light-flavour combinations to the closure basis."""
    fields = cfg.make_fields()
    with np.load(cfg.REFERENCE_DIR / "nnpdf_ensemble_truthQ_mc.npz") as blob:
        ensemble = {
            field: np.asarray(blob[field], dtype=float) for field in cfg.ALL_FIELDS
        }

    # nf=4 basis identities, applied replica by replica to the momentum
    # densities carried by this closure suite:
    #   L = u+ + d+ + s+ = (3 Sigma + T15)/4,
    #   sum_light e_q^2 q+ = 2 L/9 + T8/18 + T3/6.
    light = (3.0 * ensemble["sigma"] + ensemble["t15"]) / 4.0
    charge_weighted = (
        2.0 * light / 9.0 + ensemble["t8"] / 18.0 + ensemble["t3"] / 6.0
    )
    ensemble["light_singlet"] = light
    ensemble["charge_weighted_light"] = charge_weighted
    grid = fields["g"].grid
    fields["light_singlet"] = Field.create(
        "light_singlet", grid, element_type=cfg.ELEMENT_TYPE
    )
    fields["charge_weighted_light"] = Field.create(
        "charge_weighted_light", grid, element_type=cfg.ELEMENT_TYPE
    )
    return fields, ensemble


def _points():
    """Read the selected real HERA charm kinematics and measured values."""
    commondata = data.read_nnpdf_commondata(_COMMONDATA, _OBSERVABLE)
    x, Q2, y = commondata.require_kinematics("x", "Q2", "y")
    return {
        "x": x[_ROWS],
        "Q2": Q2[_ROWS],
        "y": y[_ROWS],
        "measurement": np.asarray(commondata.value, dtype=float)[_ROWS],
        "source": commondata.metadata["source_directory"],
    }


def _dataset(points, *, massive_order: int, mean=None, cov=None):
    n_rows = np.asarray(points["x"]).size
    return data.HeavyNCSigmaRFFNS.from_arrays(
        x=points["x"],
        Q2=points["Q2"],
        y=points["y"],
        mean=np.zeros(n_rows) if mean is None else mean,
        cov=np.eye(n_rows) if cov is None else cov,
        mass=cfg.MC,
        charge_sq=4.0 / 9.0,
        heavy_flavour="charm",
        maps_to={
            "gluon": "g",
            "singlet": "light_singlet",
            "charge_weighted_light": "charge_weighted_light",
        },
        massive_order=massive_order,
        nf=3,
        Q20=cfg.Q0_2,
        evolution_order="NNLO",
        mode="truncated",
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
        momentum_density=True,
        low_x_extensions={
            "gluon": "flat",
            "singlet": "flat",
            "charge_weighted_light": "flat",
        },
        points_per_interval=16,
        # This is only a two-row explicit smoke.  Avoid persistent caches so a
        # local correctness run cannot inherit matrices from an earlier kernel
        # revision or require the production BLAS reproducibility opt-in.
        cache_path=None,
        name=f"nnlo_heavy_hera_charm_m{massive_order}_smoke",
        source="NNPDF4.0 replica truth + HERA combined charm kinematics",
    )


def _gm_dataset(builder, points, *, mean=None, cov=None):
    """Build one active-charm reduced-cross-section route on the nf=4 basis."""
    n_rows = np.asarray(points["x"]).size
    kwargs = {}
    if builder is data.HeavyNCSigmaRFONLL:
        kwargs.update(variant="fonll_a", damping="standard")
    else:
        kwargs.update(evolution_order="NLO", mode="truncated")
    return builder.from_arrays(
        x=points["x"],
        Q2=points["Q2"],
        y=points["y"],
        mean=np.zeros(n_rows) if mean is None else mean,
        cov=np.eye(n_rows) if cov is None else cov,
        mass=cfg.MC,
        charge_sq=4.0 / 9.0,
        heavy_flavour="charm",
        maps_to={
            "gluon": "g",
            "singlet": "sigma",
            "heavy_nonsinglet": "t15",
        },
        nf=4,
        flavour_mode="vfns",
        Q20=cfg.Q0_2,
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
        momentum_density=True,
        points_per_interval=16,
        cache_path=None,
        name=f"nnlo_heavy_{builder.__name__}_sigma_r_smoke",
        source="NNPDF4.0 replica truth + HERA combined charm kinematics",
        **kwargs,
    )


def _gm_closure(builder, points, fields, ensemble):
    """Generate and rebuild one two-row GM-VFNS ``sigma_r`` pseudo-dataset."""
    generated = _gm_dataset(builder, points)
    replicas = fold_ensemble(assemble_operator(generated, fields), ensemble)
    central = replicas.mean(axis=0)
    covariance = np.atleast_2d(np.cov(replicas, rowvar=False))
    closed = _gm_dataset(builder, points, mean=central, cov=covariance)
    prediction = fold_ensemble(
        assemble_operator(closed, fields), ensemble
    ).mean(axis=0)
    residual = prediction - central
    settings = closed.meta["heavy_flavour"]
    return {
        "observable": "charm sigma_r pseudo-observable on HERA kinematics",
        "pseudo_data": central.tolist(),
        "max_abs_closure_residual": float(np.max(np.abs(residual))),
        "max_rel_closure_residual": float(
            np.max(np.abs(residual) / np.maximum(np.abs(central), 1.0e-300))
        ),
        "scheme": settings["scheme"],
        "complete_through_order": settings.get("complete_through_order", "alpha_s"),
        "heavy_basis_identity": settings["heavy_basis_identity"],
    }


def run() -> dict:
    """Run only this two-row massive-DIS smoke and return closure metrics."""
    fields, ensemble = _fields_and_ensemble()
    points = _points()
    generated = _dataset(points, massive_order=2)
    replicas = fold_ensemble(assemble_operator(generated, fields), ensemble)
    central = replicas.mean(axis=0)
    covariance = np.atleast_2d(np.cov(replicas, rowvar=False))

    closed = _dataset(
        points,
        massive_order=2,
        mean=central,
        cov=covariance,
    )
    prediction = fold_ensemble(
        assemble_operator(closed, fields), ensemble
    ).mean(axis=0)
    residual = prediction - central

    leading = _dataset(points, massive_order=1)
    leading_prediction = fold_ensemble(
        assemble_operator(leading, fields), ensemble
    ).mean(axis=0)
    result = {
        "truth": "NNPDF40_nnlo_as_01180_1000 replica ensemble at Q=mc",
        "theory_domain": (
            "photon NC charm-pair sigma_r, pole mass, fixed_nf=3 FFNS "
            "through alpha_s^2"
        ),
        "commondata": points["source"],
        "selected_rows": _ROWS.tolist(),
        "kinematics": {
            key: np.asarray(points[key], dtype=float).tolist()
            for key in ("x", "Q2", "y")
        },
        "published_measurement": points["measurement"].tolist(),
        "pseudo_data": central.tolist(),
        "max_abs_closure_residual": float(np.max(np.abs(residual))),
        "max_rel_closure_residual": float(
            np.max(np.abs(residual) / np.maximum(np.abs(central), 1.0e-300))
        ),
        "alpha_s2_relative_increment": (
            central / leading_prediction - 1.0
        ).tolist(),
        "matching_order": closed.meta["matching_order"],
        "coefficient_source": closed.meta["heavy_flavour"]["coefficient_source"],
        "closure_scope": "internal Pixel operator closure; not NNPDF FK parity",
    }
    result["s_acot_chi"] = _gm_closure(
        data.HeavyNCSigmaRSACOTChi, points, fields, ensemble
    )
    result["fonll_a"] = _gm_closure(
        data.HeavyNCSigmaRFONLL, points, fields, ensemble
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
