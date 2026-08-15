"""Build PIXEL datasets from a truth member's manifest.

Both generation (to fold the truth) and fitting call these builders with the
*same* scale/order/nf/mapping arguments, so the forward operator is identical on
both sides -- the property that makes the closure test meaningful.

All lattice and DIS datasets are read from the ASCII ``.dat`` files listed in the
member's ``manifest.json``; the per-file physics (component, CP channel, target,
reduced-cross-section ``y``) is recorded there by :mod:`closure_NNPDF_truth_small.generate`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pixel import data

from . import config as cfg


def _even_map(fields):
    return {basis: fields[name] for basis, name in cfg.EVEN_MAP.items()}


def _odd_map(fields):
    return {basis: fields[name] for basis, name in cfg.ODD_MAP.items()}


def _cache_path(section: str, stem: str, *, enabled: bool):
    if not enabled:
        return None
    # Shared across truth members: the forward operator (and hence its kernel
    # matrix) depends only on the dataset kinematics + Q0/nf/order, never on the
    # truth Q, so all truthQ_* reuse one cache.  Any accidental kinematic mismatch
    # is still caught by the cache metadata fingerprint + row sanity-check.
    return cfg.KERNEL_CACHE_ROOT / section / f"{stem}.npz"


def build_pseudoitd(rec, path, fields, cache):
    # LO (unevolved) reduced ITD of the momentum-density field.  Non-singlet quark
    # channels use alpha=-1 (int cos f = int (cos/x) q); the singlet/gluon channels
    # are the cosine transform of x*D itself (alpha=0).  NLO gluon matching is the
    # separate GluonPseudoITD path.
    return data.PseudoITD.from_file(
        path, L=cfg.LATTICE_L, a=cfg.LATTICE_A_FM,
        maps_to=fields[rec["field"]], component=rec["component"],
        momentum_density=bool(rec.get("momentum_density", True)),
        low_x_extension=cfg.low_x_completion(rec["field"]),
        pseudoitd_observable=cfg.PITD_OBSERVABLE,
        operator_scheme=cfg.PITD_OPERATOR_SCHEME,
        pseudoitd_kernel=cfg.PITD_KERNEL,
        pseudoitd_data_normalization=cfg.PITD_DATA_NORMALIZATION,
        pseudoitd_lorentz_component=cfg.PITD_LORENTZ_COMPONENT,
        m_pi=0.140, name=Path(rec["file"]).stem, cache_path=cache,
    )


def build_mellin(rec, path, fields, cache):
    cp_target = f"cp_{cfg.field_cparity(rec['field'])}"
    return data.MellinMoment.from_file(
        path, L=cfg.LATTICE_L, a=cfg.LATTICE_A_FM,
        maps_to={cp_target: fields[rec["field"]]},
        Q2=cfg.MOMENT_Q2, mu0_2=cfg.Q0_2, momentum_density=True,
        low_x_extension=cfg.low_x_completion(rec["field"]),
        name=Path(rec["file"]).stem, cache_path=cache,
    )


def _exp_nuisance_kwargs(rec, path, *, enabled=True, t0=None):
    """Normalization + correlated-systematic kwargs for a DIS record.

    The overall normalization is a scalar relative size.  The correlated sources
    are the ``%``-type columns, i.e. **multiplicative**: generation saved them as
    relative per-row vectors (one ``.npz`` per table, since HERA carries 169
    sources), and they are passed as ``multiplicative_systematics`` so the
    builder references them to ``t0`` when one is supplied and to the data
    otherwise -- the NNPDF t0 prescription.  ``t0`` is the current theory
    prediction for this dataset's rows, supplied by the iteration in
    :func:`fit.run_fit`; ``None`` is the first (data-referenced) iteration.

    ``enabled=False`` is the generation path: the forward operator must stay
    purely physical there (the nuisance realization is injected separately), and
    the sidecars do not exist yet when the operators are first assembled.
    """
    if not enabled:
        return {}
    kwargs = {}
    if rec.get("rel_norm") is not None:
        kwargs["normalization"] = float(rec["rel_norm"])
        kwargs["fit_normalization"] = cfg.DIS_FIT_NORMALIZATION
    sys_file = rec.get("correlated_file")
    if sys_file:
        blob = np.load(Path(path).parent.parent / "sys" / sys_file)
        relative = np.asarray(blob["relative"], dtype=float)
        names = [str(n) for n in blob["names"]]
        kwargs["multiplicative_systematics"] = dict(zip(names, relative))
    # t0 references every *marginalized* multiplicative quantity: the correlated
    # systematics always, and the normalization only when it is not being fitted
    # (a fitted normalization scales the prediction and needs no reference).
    if t0 is not None and ("multiplicative_systematics" in kwargs
                           or not kwargs.get("fit_normalization", False)):
        kwargs["t0"] = np.asarray(t0, dtype=float)
    return kwargs


def build_f2(rec, path, fields, cache, *, with_exp_nuisances=True, t0=None):
    return data.F2.from_file(
        path, target=rec["target"], maps_to=_even_map(fields), nf=cfg.NF,
        Q20=cfg.Q0_2, order=cfg.ORDER, mode=cfg.MODE, mc=cfg.MC, mb=cfg.MB,
        momentum_density=True,
        low_x_extensions=cfg.even_low_x_completions(),
        name=Path(rec["file"]).stem, cache_path=cache,
        **_exp_nuisance_kwargs(rec, path, enabled=with_exp_nuisances, t0=t0),
    )


def build_sigma_r(rec, path, fields, cache, *, with_exp_nuisances=True, t0=None):
    kwargs = dict(_exp_nuisance_kwargs(rec, path, enabled=with_exp_nuisances, t0=t0))
    if "y" in rec:
        kwargs["y"] = rec["y"]
    elif "sqrt_s" in rec:
        kwargs["sqrt_s"] = float(rec["sqrt_s"])
    return data.SigmaR.from_file(
        path, target=rec["target"], maps_to=_even_map(fields), nf=cfg.NF,
        Q20=cfg.Q0_2, order=cfg.ORDER, mode=cfg.MODE, mc=cfg.MC, mb=cfg.MB,
        momentum_density=True,
        low_x_extensions=cfg.even_low_x_completions(),
        name=Path(rec["file"]).stem, cache_path=cache, **kwargs,
    )


def build_sigma_r_cc(rec, path, fields, cache, *, with_exp_nuisances=True, t0=None):
    kwargs = dict(_exp_nuisance_kwargs(rec, path, enabled=with_exp_nuisances, t0=t0))
    if "y" in rec:
        kwargs["y"] = rec["y"]
    elif "sqrt_s" in rec:
        kwargs["sqrt_s"] = float(rec["sqrt_s"])
    return data.ChargedCurrentSigmaR.from_file(
        path, target=rec["target"], maps_to=_even_map(fields),
        valence_maps_to=_odd_map(fields), beam_charge=rec["beam"], nf=cfg.NF,
        Q20=cfg.Q0_2, order=cfg.ORDER, mode=cfg.MODE, mc=cfg.MC, mb=cfg.MB,
        momentum_density=True,
        low_x_extensions=cfg.even_low_x_completions(),
        valence_low_x_extensions=cfg.odd_low_x_completions(),
        name=Path(rec["file"]).stem, cache_path=cache,
        **kwargs,
    )



def build_drell_yan(rec, path, fields, cache):
    """Build a Drell-Yan bilinear dataset over the closure basis fields.

    Uses the parton -> basis-field maps (``config.dy_field_maps``) so the shared
    :func:`pixel.data.DrellYan` builder expands the bilinear luminosity over the
    nine basis fields.  Generation folds the same tensors; the fit reads the
    ``mean``/``cov`` back from the manifest (no linear ``.dat`` file).
    """
    from pixel.data import DrellYan
    from pixel.kernels.drell_yan import DrellYanElectroweak

    fields_A, fields_B = cfg.dy_field_maps(rec["reaction"])
    kwargs = {}
    if "mean" in rec and "cov" in rec:
        kwargs["mean"] = rec["mean"]
        kwargs["cov"] = rec["cov"]
        # The overall luminosity normalization is a nuisance only on the fit side:
        # during generation there is no measured vector to reference, and the
        # injected offset is applied to the folded central value instead.
        if rec.get("rel_norm") is not None:
            kwargs["normalization"] = float(rec["rel_norm"])
            kwargs["fit_normalization"] = cfg.DY_FIT_NORMALIZATION
    electroweak = (
        None if rec.get("boson") is None
        else DrellYanElectroweak(boson=rec["boson"])
    )
    # A non-photon provider normalizes its own weight table with a *running*
    # alpha_EM, and ``DYKernel._row_couplings`` refuses to pair that with a
    # different value in the hard prefactor.  The fixed Thomson ``DY_ALPHA_EM``
    # is right for the photon-only fixed-target tables and wrong here, so take
    # the coupling from the provider itself at the row's Q^2.  A photon provider
    # short-circuits before the comparison and keeps the fixed value.
    alpha_em = cfg.DY_ALPHA_EM
    if electroweak is not None and not electroweak.photon_only:
        # ``alpha_em`` is one scalar per dataset while the provider runs with
        # Q^2, so a multi-Q^2 boson table needs per-row couplings rather than
        # this branch.  Refuse it instead of silently normalizing every row at
        # one row's coupling.
        q2_values = {float(q) for q in rec["Q2"]}
        if len(q2_values) != 1:
            raise ValueError(
                f"{rec['label']}: a non-photon Drell-Yan table shares one scalar "
                f"alpha_EM, so it needs a single Q^2; got {sorted(q2_values)}"
            )
        alpha_em = float(electroweak.alpha_em(q2_values.pop()))
    return DrellYan(
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=cfg.DY_NF,
        alpha_em=alpha_em, channels=cfg.DY_CHANNELS,
        electroweak=electroweak,
        evolution_A=None, evolution_B=None,
        source=rec.get("classification", "closure"),
        component=rec.get("observable_contract", rec["reaction"]),
        cache_dir=str(cache.parent) if cache is not None else None,
        **kwargs,
    )


_LATTICE_BUILDERS = {"pseudoitd": build_pseudoitd, "mellin": build_mellin}
_EXP_BUILDERS = {
    "f2": build_f2,
    "sigma_r": build_sigma_r,
    "sigma_r_cc": build_sigma_r_cc,
}


def _experimental_builder(kind):
    if kind == "f3_proxy":
        raise RuntimeError(
            "stale HERA CC F3-proxy closure data; regenerate with --remake-data "
            "before fitting the physical sigma_r_cc operator"
        )
    try:
        return _EXP_BUILDERS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown experimental closure kind {kind!r}") from exc


def load_manifest(q_key: str) -> dict:
    return json.loads((cfg.truth_dir(q_key) / "manifest.json").read_text())


def build_datasets(q_key, fields, *, include_lattice=True, include_exp=True,
                   include_dy=False, include_synthetic_z=False, use_kernel_cache=True,
                   with_exp_nuisances=True, t0=None):
    """Build the PIXEL datasets for one truth member from its manifest.

    Returns a list of ``Dataset`` objects for the requested sources.
    """
    truth_dir = cfg.truth_dir(q_key)
    manifest = load_manifest(q_key)
    out = []
    if include_lattice:
        for rec in manifest.get("lattice", []):
            path = truth_dir / rec["file"]
            cache = _cache_path("lattice", path.stem, enabled=use_kernel_cache)
            out.append(_LATTICE_BUILDERS[rec["kind"]](rec, path, fields, cache))
    if include_exp:
        for rec in manifest.get("exp", []):
            path = truth_dir / rec["file"]
            cache = _cache_path("exp", path.stem, enabled=use_kernel_cache)
            label = rec.get("name", Path(rec["file"]).stem)
            out.append(_experimental_builder(rec["kind"])(
                rec, path, fields, cache,
                with_exp_nuisances=with_exp_nuisances,
                t0=None if t0 is None else t0.get(label)))
    if include_dy:
        for rec in manifest.get("dy", []):
            cache = _cache_path("dy", rec["label"], enabled=use_kernel_cache)
            out.append(build_drell_yan(rec, None, fields, cache))
    if include_synthetic_z:
        records = manifest.get("synthetic_z", [])
        if len(records) != 1:
            raise ValueError(
                "synthetic_z_plumbing requires exactly one manifest record; "
                "regenerate the small closure truth"
            )
        rec = records[0]
        if rec.get("classification") != "synthetic_proxy":
            raise ValueError("synthetic Z record must be classified synthetic_proxy")
        cache = _cache_path("synthetic_z", rec["label"], enabled=use_kernel_cache)
        out.append(build_drell_yan(rec, None, fields, cache))
    return out
