"""Build PIXEL datasets from a truth member's manifest.

Both generation (to fold the truth) and fitting call these builders with the
*same* scale/order/nf/mapping arguments, so the forward operator is identical on
both sides -- the property that makes the closure test meaningful.

All lattice and DIS datasets are read from the ASCII ``.dat`` files listed in the
member's ``manifest.json``; the per-file physics (ensemble spacing/extent/pion
mass, component, CP channel, moment order, target, reduced-cross-section ``y``,
and active systematic keys) is recorded there by
:mod:`closure_NNPDF_truth.generate`.

Lattice systematics
-------------------
Pseudo-ITD records wire each active key to a shared x-space nuisance curve
``nuis_f_k``.  Mellin records instead use independent scalar nuisance fields,
one per physical field, systematic key, and moment order.  A direct all-ones
row from the existing ``Unit`` kernel maps the scalar into data space, and the
registry coefficient supplies the
ensemble-dependent chiral or ``a/L`` scale.  Generation builds physical-only
datasets and injects the matching systematic shifts separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pixel import data, kernels
from pixel.core.model import Contribution

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
    # truth Q, so all truthQ_* reuse one cache.  The stem includes the ensemble id
    # for lattice data so different spacings/extents do not collide.  Any
    # accidental kinematic mismatch is still caught by the cache fingerprint.
    return cfg.KERNEL_CACHE_ROOT / section / f"{stem}.npz"


def _sys_map(rec, fields, *, enabled: bool):
    """Map pseudo-ITD keys to shared x-space nuisance curves."""
    if not enabled:
        return None
    keys = rec.get("systematics", [])
    field = rec["field"]
    out = {k: fields[cfg.nuisance_field_name(field, k)] for k in keys
           if cfg.nuisance_field_name(field, k) in fields}
    return out or None


def _moment_systematic_contributions(rec, dataset, fields, *, enabled: bool):
    """Independent scalar systematic terms for one Mellin power."""
    if not enabled:
        return []
    allowed = set(kernels.lattice_systematic_keys(kernels.MELLIN_MOMENT))
    out = []
    for key in rec.get("systematics", []):
        if key not in allowed:
            raise KeyError(
                f"systematic {key!r} is not available for Mellin moments; "
                f"available: {sorted(allowed)}"
            )
        name = cfg.moment_nuisance_field_name(
            rec["field"], key, int(rec["order"])
        )
        if name not in fields:
            continue
        _, factory = kernels.LATTICE_SYSTEMATICS[key]
        coefficient = factory(dataset.meta)
        out.append(
            Contribution(
                name,
                kernels.Unit(
                    coefficient=coefficient,
                    low_x_extension="flat",
                ),
            )
        )
    return out


def build_pseudoitd(rec, path, fields, cache, *, with_systematics=True):
    # Reduced ITD of the momentum-density field.  Non-singlet quark channels use
    # alpha=-1 (int cos f = int (cos/x) q); the singlet/gluon channels are the
    # cosine transform of x*D itself (alpha=0).  For a vector ITD, real/cosine
    # maps to C-odd q-qbar and imag/sine maps to C-even q+qbar.  Each ensemble
    # sets its own L (sites), a (fm), m_pi.
    #
    # With PITD_EVOLVE the physical PDF field is evolved from Q0 = mc to the
    # per-separation matching scale mu(z) = lam/z and matched at the configured
    # fixed order.  Lattice-artifact nuisance curves remain bare measurement-
    # level transforms; giving them DGLAP evolution would invent PDF scale
    # dependence for discretization, higher-twist, and chiral corrections.
    evolution = {}
    if cfg.PITD_EVOLVE:
        evolution = {
            "mu0_2": cfg.Q0_2,
            "evolution_lambda": cfg.PITD_MATCHING_LAMBDA,
            "order": cfg.ORDER,
            "mode": cfg.MODE,
            "matching": cfg.PITD_MATCHING_ORDER,
            "pseudoitd_observable": cfg.PITD_OBSERVABLE,
            "operator_scheme": cfg.PITD_OPERATOR_SCHEME,
            "pseudoitd_kernel": cfg.PITD_KERNEL,
            "pseudoitd_data_normalization": cfg.PITD_DATA_NORMALIZATION,
            "pseudoitd_lorentz_component": cfg.PITD_LORENTZ_COMPONENT,
        }
    return data.PseudoITD.from_file(
        path, L=rec["L"], a=rec["a"],
        maps_to=fields[rec["field"]], component=rec["component"],
        momentum_density=bool(rec.get("momentum_density", True)),
        low_x_extension=cfg.low_x_completion(rec["field"]),
        m_pi=rec.get("m_pi", 0.140),
        systematics=_sys_map(rec, fields, enabled=with_systematics),
        name=rec.get("name", Path(rec["file"]).stem), cache_path=cache,
        **evolution,
    )


def build_mellin(rec, path, fields, cache, *, with_systematics=True):
    cp_target = f"cp_{cfg.field_cparity(rec['field'])}"
    dataset = data.MellinMoment.from_file(
        path, L=rec["L"], a=rec["a"],
        maps_to={cp_target: fields[rec["field"]]},
        Q2=cfg.MOMENT_Q2, mu0_2=cfg.Q0_2, momentum_density=True,
        low_x_extension=cfg.low_x_completion(rec["field"]),
        m_pi=rec.get("m_pi", 0.140),
        name=rec.get("name", Path(rec["file"]).stem), cache_path=cache,
    )
    dataset.contributions.extend(
        _moment_systematic_contributions(
            rec, dataset, fields, enabled=with_systematics
        )
    )
    return dataset


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


def build_f2(rec, path, fields, cache, *, with_systematics=True,
             with_exp_nuisances=True, t0=None):
    return data.F2.from_file(
        path, target=rec["target"], maps_to=_even_map(fields), nf=cfg.NF,
        Q20=cfg.Q0_2, order=cfg.ORDER, mode=cfg.MODE, mc=cfg.MC, mb=cfg.MB,
        momentum_density=True,
        low_x_extensions=cfg.even_low_x_completions(),
        name=Path(rec["file"]).stem, cache_path=cache,
        **_exp_nuisance_kwargs(rec, path, enabled=with_exp_nuisances, t0=t0),
    )


def build_sigma_r(rec, path, fields, cache, *, with_systematics=True,
                  with_exp_nuisances=True, t0=None):
    kwargs = dict(_exp_nuisance_kwargs(rec, path, enabled=with_exp_nuisances, t0=t0))
    if "y" in rec:
        kwargs["y"] = rec["y"]
    elif "sqrt_s" in rec:
        kwargs["sqrt_s"] = float(rec["sqrt_s"])
    return data.SigmaR.from_file(
        path, target=rec["target"], maps_to=_even_map(fields),
        valence_maps_to=_odd_map(fields), nf=cfg.NF,
        neutral_current=True, beam_charge=rec["beam"],
        polarization=float(rec.get("polarization", 0.0)),
        Q20=cfg.Q0_2, order=cfg.ORDER, mode=cfg.MODE, mc=cfg.MC, mb=cfg.MB,
        momentum_density=True,
        low_x_extensions=cfg.even_low_x_completions(),
        valence_low_x_extensions=cfg.odd_low_x_completions(),
        name=Path(rec["file"]).stem, cache_path=cache, **kwargs,
    )


def build_sigma_r_cc(rec, path, fields, cache, *, with_systematics=True,
                     with_exp_nuisances=True, t0=None):
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


def dy_evolution_maps(rec, fields):
    """Row-wise flavour projection maps for one Drell-Yan record.

    Returns ``(fields_A, fields_B, evolution_A, evolution_B)``.  Each row is
    evolved by its own operator, because a fixed-target table's rows span a
    factor of ~4 in ``M^2`` and one matrix at an effective mass is a different
    operator, not an approximation of the right one.  Distinct scales -- not
    rows -- set the assembly cost.

    The projection replaces the static ``config.dy_field_maps`` coefficients:
    above the bottom threshold the parton content is five-flavour and the
    ``Nf=4`` input fields reach ``b``/``bbar`` only through the generated
    ``T24``-like combination, which a fixed coefficient table cannot express.
    """
    from pixel.kernels.evolution.flavor_transition import (
        conjugate_projection,
        isoscalar_projection,
        rowwise_parton_evolution_projection,
    )

    # Every closure field shares one basis; take it from whichever is present.
    basis = next(iter(fields.values())).basis
    cache_dir = cfg.dy_evolution_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    proton = rowwise_parton_evolution_projection(
        basis,
        Q2_rows=np.asarray(rec["Q2"], dtype=float),
        Q20=cfg.Q0_2,
        order=cfg.DY_EVOLUTION_ORDER,
        mode=cfg.DY_EVOLUTION_MODE,
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
        low_x_extensions=cfg.dy_low_x_completions(),
        cache_dir=str(cache_dir),
    )
    if int(proton.nf) != int(cfg.DY_NF):
        raise ValueError(
            f"DY record {rec.get('label')!r} projects to nf={proton.nf} but the "
            f"closure luminosity is configured for nf={cfg.DY_NF}; the hard "
            "kernel would then skip the flavours evolution just generated"
        )
    reaction = rec["reaction"]
    if reaction == "pp":
        target = proton
    elif reaction == "pd":
        target = isoscalar_projection(proton)
    elif reaction == "ppbar":
        target = conjugate_projection(proton)
    else:
        raise ValueError(f"unsupported DY reaction {reaction!r}")
    return proton.fields, target.fields, proton.evolution, target.evolution


def build_drell_yan(rec, path, fields, cache, *, with_systematics=True):
    """Build a Drell-Yan bilinear dataset over the closure basis fields.

    ``rec`` carries the E866 kinematics (S, Q2, Y) and the reaction (``pp``/``pd``).
    The parton -> basis-field maps (``config.dy_field_maps``) let the shared
    :func:`pixel.data.DrellYan` builder expand the bilinear luminosity over the
    nine basis fields; ``maps_to`` field *objects* are passed so the analysis
    binds to the same field instances.  ``cache``/``path`` are unused here (the
    kinematics live in the manifest, not a ``.dat`` file), kept for a uniform
    builder signature.
    """
    from pixel.data import DrellYan

    # parton -> {basis_field_name: coeff}; DrellYan binds contributions by name.
    fields_A, fields_B = cfg.dy_field_maps(rec["reaction"])
    evolution_A = evolution_B = None
    if cfg.DY_EVOLVE:
        fields_A, fields_B, evolution_A, evolution_B = dy_evolution_maps(
            rec, fields
        )
    kwargs = {}
    if "mean" in rec and "cov" in rec:              # fit path (generation fills these)
        kwargs["mean"] = rec["mean"]
        kwargs["cov"] = rec["cov"]
        # The overall luminosity normalization is a nuisance only on the fit side:
        # during generation there is no measured vector to reference, and the
        # injected offset is applied to the folded central value instead.
        if rec.get("rel_norm") is not None:
            kwargs["normalization"] = float(rec["rel_norm"])
            kwargs["fit_normalization"] = cfg.DY_FIT_NORMALIZATION
    return DrellYan(
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=cfg.DY_NF,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        source="closure", component=rec["reaction"],
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


def _lattice_cache_stem(rec, path) -> str:
    """Kernel-cache stem for a lattice record (namespaced by ensemble id)."""
    ens = rec.get("ensemble", "ens")
    return f"{ens}_{path.stem}"


def build_datasets(q_key, fields, *, include_lattice=True, include_exp=True,
                   include_dy=False, use_kernel_cache=True, with_systematics=True,
                   with_exp_nuisances=True, t0=None):
    """Build the PIXEL datasets for one truth member from its manifest.

    ``with_systematics`` adds the lattice nuisance contributions (the fit path);
    generation passes ``with_systematics=False`` so the folded operator carries
    only the physical field and the statistical covariance is the pure replica
    spread (the systematic shift is injected separately, see generate.py).
    ``with_exp_nuisances`` is the DIS counterpart: it folds each table's real
    normalization and correlated systematics into the covariance.  It is a
    *separate* switch because a DIS-only fit still wants them even though it
    creates no lattice nuisance fields.
    ``t0`` maps a DIS dataset label to the current theory prediction for its
    rows; multiplicative uncertainties are referenced to it instead of to the
    data (the NNPDF t0 prescription).  ``None`` is the first, data-referenced
    iteration -- see :func:`fit.run_fit`, which iterates it to convergence.
    ``include_dy`` adds the bilinear Drell-Yan datasets.

    Returns a list of ``Dataset`` objects for the requested sources (linear
    lattice/DIS first, then any bilinear Drell-Yan datasets).
    """
    truth_dir = cfg.truth_dir(q_key)
    manifest = load_manifest(q_key)
    out = []
    if include_lattice:
        for rec in manifest.get("lattice", []):
            path = truth_dir / rec["file"]
            cache = _cache_path("lattice", _lattice_cache_stem(rec, path),
                                enabled=use_kernel_cache)
            out.append(_LATTICE_BUILDERS[rec["kind"]](
                rec, path, fields, cache, with_systematics=with_systematics))
    if include_exp:
        for rec in manifest.get("exp", []):
            path = truth_dir / rec["file"]
            cache = _cache_path("exp", path.stem, enabled=use_kernel_cache)
            label = rec.get("name", Path(rec["file"]).stem)
            out.append(_experimental_builder(rec["kind"])(
                rec, path, fields, cache, with_systematics=with_systematics,
                with_exp_nuisances=with_exp_nuisances,
                t0=None if t0 is None else t0.get(label)))
    if include_dy:
        for rec in manifest.get("dy", []):
            cache = _cache_path("dy", rec["label"], enabled=use_kernel_cache)
            out.append(build_drell_yan(rec, None, fields, cache,
                                       with_systematics=with_systematics))
    return out
