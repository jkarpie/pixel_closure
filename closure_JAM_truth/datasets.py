"""Build PIXEL datasets from a truth member's manifest.

Both generation (to fold the truth) and fitting call these builders with the
*same* scale/order/nf/mapping arguments, so the forward operator is identical on
both sides -- the property that makes the closure test meaningful.

All lattice and DIS datasets are read from the ASCII ``.dat`` files listed in the
member's ``manifest.json``; the per-file physics (ensemble spacing/extent/pion
mass, component, CP channel, moment order, target, reduced-cross-section ``y``,
and active systematic keys) is recorded there by
:mod:`closure_JAM_truth.generate`.

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

from collections.abc import Mapping

import numpy as np

from pixel.util.flavor import NuclearTarget

from pixel.data.closure_datatypes import (
    DATATYPE_GATES,
    assert_datatype_ready,
)

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
        nuclear_projection,
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
    elif reaction == "pCu":
        target = nuclear_projection(proton, cfg.DY_COPPER_PROTON_FRACTION)
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

def _record_target(rec):
    """The record's target, reconstructing a nuclear one from its manifest mapping.

    JSON cannot carry a ``NuclearTarget``, so a nuclear record stores ``{"name",
    "mass_number", "atomic_number"}`` and it is rebuilt here.  The mapping form is what
    keeps A and Z *visible in the manifest*: ``normalize_target`` refuses a bare
    ``"Pb"`` precisely because natural lead and Pb-208 are different targets, and a
    manifest that named only "lead" would put that choice back out of sight.
    """
    target = rec["target"]
    if isinstance(target, Mapping):
        return NuclearTarget(
            str(target["name"]),
            float(target["mass_number"]),
            float(target["atomic_number"]),
        )
    return target


def build_drell_yan_binned(rec, path, fields, cache, *, with_systematics=True):
    """Build a **bin-integrated fiducial** Drell-Yan dataset (collider Z rapidity).

    The bilinear sibling of :func:`build_drell_yan`.  E866 is a *pointwise* fixed-target
    measurement: one (S, Q2, Y) per row.  A collider Z measurement is not -- each row is
    a cross section integrated over a rapidity bin (and, for the 2D sets, a mass bin)
    inside a *fiducial* lepton acceptance, so three things change and all three are
    settings on the same :func:`pixel.data.DrellYan` builder:

    * ``observable_mode="fiducial_dilepton"`` instead of the default ``"pointwise"``;
    * bin edges ``Y_low``/``Y_high`` (and ``Q2_low``/``Q2_high`` where the set is binned
      in mass).  ``_bin_kinematics`` falls back to the pointwise value per axis, so a
      Z-pole rapidity scan passes rapidity edges and leaves the mass axis degenerate at
      ``m_Z^2`` -- which is what the measurement is;
    * ``lepton_cuts``, resolved from ``nnpdf40_fiducial_cuts.yaml`` by the dataset's own
      NNPDF name.  This is **required**, not defaulted: the builder refuses
      ``fiducial_dilepton`` without it, because a wrong acceptance is a silent
      normalization error that a chi-squared cannot see.

    ``nf`` is 5 here regardless of ``cfg.DY_NF``.  At ``Q ~ m_Z`` the b quark is active,
    and the fields are evolved up from the ``Nf=4`` input basis exactly as the
    fixed-target path does when ``DY_EVOLVE`` is on.

    .. warning::

       **Structurally validated, not validated against NNPDF.**  This builds the
       operator, folds the truth through it and produces a dataset whose shapes,
       covariance and row count agree with the commondata.  It has *not* been compared
       row-by-row against NNPDF's own FK predictions -- ``assert_native_campaign_ready``
       still fails these sets on NNLO/theory-200 grounds (the closure runs NLO, and
       NNLO Drell-Yan needs the unvendored ReDY source).  Treat a recovered pull here as
       a statement about this closure's internal consistency, not as an NNPDF benchmark.
    """
    from pixel.data import DrellYan
    from pixel.data.fiducial_cuts import lepton_cuts_for

    fields_A, fields_B = cfg.dy_field_maps(rec["reaction"])
    evolution_A = evolution_B = None
    if cfg.DY_EVOLVE:
        fields_A, fields_B, evolution_A, evolution_B = dy_evolution_maps(rec, fields)

    kwargs = {}
    if "mean" in rec and "cov" in rec:               # fit path
        kwargs["mean"] = rec["mean"]
        kwargs["cov"] = rec["cov"]
        if rec.get("rel_norm") is not None:
            kwargs["normalization"] = float(rec["rel_norm"])
            kwargs["fit_normalization"] = cfg.DY_FIT_NORMALIZATION
    for edge in ("Q2_low", "Q2_high", "Y_low", "Y_high"):
        if rec.get(edge) is not None:
            kwargs[edge] = rec[edge]

    return DrellYan(
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=5,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        observable_mode="fiducial_dilepton",
        lepton_cuts=lepton_cuts_for(rec["nnpdf_legacy_name"]),
        source="closure", component=rec["reaction"],
        cache_dir=str(cache.parent) if cache is not None else None,
        **kwargs,
    )


def build_sigma_r_heavy(rec, path, fields, cache, *, with_systematics=True,
                        with_exp_nuisances=True, t0=None):
    """Build a **heavy-flavour-tagged** neutral-current reduced cross section.

    HERA's ``sigma_r^bbbar`` is not the inclusive ``sigma_r`` restricted to a region: it
    is the bottom-pair contribution alone, so it takes different coefficient functions
    (photon-gluon fusion into a massive pair) and a different builder.

    **Variant: FONLL-A, and that is the correct choice here rather than a fallback.**
    ``fonll_a`` is ``pdf_order: NLO, massless_order: 1, massive_order: 1``
    (``pixel.kernels.FONLL_VARIANTS``), which is exactly what this closure runs
    (``cfg.ORDER``).  ``fonll_b``/``fonll_c`` raise ``NotImplementedError`` from
    ``HeavyNCFONLL.from_arrays`` -- measured 2026-09-09 -- and NNPDF's theory-200 uses
    FONLL-C, so a *parity* comparison against NNPDF is out of reach.  A closure is not a
    parity comparison: it folds a truth through this operator and fits it back with the
    same one, and FONLL-A is self-consistent at NLO.

    **Flavour handling: ``flavour_mode="vfns"`` with ``nf=4``.**  The bottom PDF does not
    exist in this suite's input basis -- the nine fields are Nf=4 at ``Q0 = mc``.  The
    matched GM-VFNS path is the one that takes an Nf=4 input basis plus ``Q20`` and
    generates the heavy flavour by evolution across ``mb``; it *requires* ``nf == 4``
    (``heavy_flavour.py`` raises "matched GM-VFNS evolution requires an Nf=4 input
    basis" otherwise).  So the three fields supplied are the closure's own ``sigma``,
    ``g`` and ``t15``, and ``heavy_plus = (singlet - T24)/5`` is formed downstream from
    the evolved basis rather than read from a field that is not there.

    .. warning::

       Structurally validated only -- shapes, row counts and covariance against the
       commondata.  Not compared row-by-row against NNPDF's own predictions; see
       ``build_drell_yan_binned`` for the same caveat and why it applies.
    """
    from pixel.data import HeavyNCSigmaRFONLL

    kwargs = dict(_exp_nuisance_kwargs(rec, path, enabled=with_exp_nuisances, t0=t0))
    if "y" in rec:
        kwargs["y"] = rec["y"]
    elif "sqrt_s" in rec:
        kwargs["sqrt_s"] = float(rec["sqrt_s"])
    x, q2 = rec["_leading"]
    return HeavyNCSigmaRFONLL.from_arrays(
        x=np.asarray(x, dtype=float), Q2=np.asarray(q2, dtype=float),
        mean=rec["mean"], cov=rec["cov"],
        maps_to={"singlet": fields["sigma"], "gluon": fields["g"],
                 "heavy_nonsinglet": fields["t15"]},
        heavy_flavour=rec["heavy_flavour"], mass=float(rec["heavy_mass"]),
        charge_sq=float(rec["charge_sq"]),
        variant="fonll_a", nf=4, flavour_mode="vfns",
        Q20=cfg.Q0_2, alphaS_MZ=cfg.ALPHAS_MZ, mc=cfg.MC, mb=cfg.MB,
        target="proton", momentum_density=True,
        low_x_extensions=cfg.even_low_x_completions(),
        name=Path(rec["file"]).stem, cache_path=cache, **kwargs,
    )


def build_drell_yan_fiducial_lepton(rec, path, fields, cache, *, with_systematics=True):
    """Build a **W-type** collider Drell-Yan dataset: one *measured, charged* lepton.

    The third Drell-Yan arm. ``build_drell_yan`` is pointwise fixed-target and
    ``build_drell_yan_binned`` is a charge-blind dilepton pair; this one is neither,
    because a W decays to a lepton **and a neutrino** and only the lepton is measured.
    Three consequences, all of which PIXEL already handles and none of which the
    dilepton path can express:

    * ``observable_mode="fiducial_lepton"`` rather than ``"fiducial_dilepton"``;
    * ``lepton_cuts`` must be a :class:`LeptonLevelCuts`, which carries the lepton's
      **charge**. The builder refuses a charge-blind ``LeptonCuts`` outright -- W+ goes
      as ``u dbar`` and W- as ``d ubar``, so a lost charge is a silently wrong flavour
      decomposition rather than a normalisation error;
    * the cuts record's neutrino ``p_T`` and transverse-mass thresholds enter the
      acceptance, which is why these sets carry ``stage: B`` in
      ``nnpdf40_fiducial_cuts.yaml``.

    ``rec["lepton_charge"]`` is **required and never defaulted**. NNPDF's commondata does
    not carry the charge as a kinematic column; for the LHCb W/Z sets it is recoverable
    from the metadata's ``plotting.extra_labels.Boson`` (measured 2026-09-09: 8 ``W+``,
    8 ``W-``, 17 ``Z`` for LHCBWZMU7TEV), and where it is not recoverable the dataset
    must stay gated rather than have a charge guessed for it.
    """
    from pixel.data import DrellYan
    from pixel.data.fiducial_cuts import lepton_level_cuts_for

    charge = rec.get("lepton_charge")
    if charge not in (1, -1):
        raise ValueError(
            f"{rec.get('label')!r}: build_drell_yan_fiducial_lepton needs "
            f"rec['lepton_charge'] of +1 or -1, got {charge!r}. The W charge sets the "
            "flavour decomposition (W+ ~ u dbar, W- ~ d ubar); defaulting it would be a "
            "silent physics error, so a dataset whose per-row charge is not recorded "
            "must stay gated instead"
        )
    fields_A, fields_B = cfg.dy_field_maps(rec["reaction"])
    evolution_A = evolution_B = None
    if cfg.DY_EVOLVE:
        fields_A, fields_B, evolution_A, evolution_B = dy_evolution_maps(rec, fields)
    kwargs = {}
    if "mean" in rec and "cov" in rec:
        kwargs["mean"], kwargs["cov"] = rec["mean"], rec["cov"]
        if rec.get("rel_norm") is not None:
            kwargs["normalization"] = float(rec["rel_norm"])
            kwargs["fit_normalization"] = cfg.DY_FIT_NORMALIZATION
    for edge in ("Q2_low", "Q2_high", "Y_low", "Y_high"):
        if rec.get(edge) is not None:
            kwargs[edge] = rec[edge]
    return DrellYan(
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=5,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        observable_mode="fiducial_lepton",
        lepton_cuts=lepton_level_cuts_for(rec["nnpdf_legacy_name"], charge=int(charge)),
        source="closure", component=rec["reaction"],
        cache_dir=str(cache.parent) if cache is not None else None,
        **kwargs,
    )


def build_drell_yan_bin_integrated(rec, path, fields, cache, *, with_systematics=True):
    """Build a Drell-Yan dataset corrected by the experiment to **full phase space**.

    The distinction from :func:`build_drell_yan_binned` is one word in the cuts record
    and it inverts the acceptance treatment: an ``observable_level: full_phase_space``
    measurement has already been corrected for the fiducial acceptance by the
    experiment, so applying one here would **double-correct**. ``lepton_cuts_for``
    refuses such a dataset for exactly that reason, and this builder passes no cuts at
    all -- ``observable_mode="bin_integrated"``.

    Still bin-integrated: the rows are integrals over published mass/rapidity bins, so
    the edges are required. That is what currently gates its one candidate --
    ``CMSDY2D11``'s commondata carries ``rapidity``/``m_Z2``/``sqrts`` as bin *centres*
    with no ``_min``/``_max`` (measured 2026-09-09), so the edges must come from the
    paper or HEPData before it can be selected. The builder is ready for them.
    """
    from pixel.data import DrellYan

    missing = [e for e in ("Y_low", "Y_high") if rec.get(e) is None]
    if missing:
        raise ValueError(
            f"{rec.get('label')!r}: build_drell_yan_bin_integrated needs bin edges "
            f"{missing}; a bin-integrated observable evaluated at a bin centre is a "
            "different number, and silently smaller than the integral for a convex "
            "spectrum"
        )
    fields_A, fields_B = cfg.dy_field_maps(rec["reaction"])
    evolution_A = evolution_B = None
    if cfg.DY_EVOLVE:
        fields_A, fields_B, evolution_A, evolution_B = dy_evolution_maps(rec, fields)
    kwargs = {}
    if "mean" in rec and "cov" in rec:
        kwargs["mean"], kwargs["cov"] = rec["mean"], rec["cov"]
        if rec.get("rel_norm") is not None:
            kwargs["normalization"] = float(rec["rel_norm"])
            kwargs["fit_normalization"] = cfg.DY_FIT_NORMALIZATION
    for edge in ("Q2_low", "Q2_high", "Y_low", "Y_high"):
        if rec.get(edge) is not None:
            kwargs[edge] = rec[edge]
    return DrellYan(
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=5,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        observable_mode="bin_integrated",
        source="closure", component=rec["reaction"],
        cache_dir=str(cache.parent) if cache is not None else None,
        **kwargs,
    )


def build_drell_yan_mass_spectrum(rec, path, fields, cache, *, with_systematics=True):
    """Build a **rapidity-integrated** Drell-Yan mass spectrum.

    A different observable from a rapidity scan, not a variant of one: each row is
    integrated over the *whole* fiducial rapidity range at a given dilepton mass bin.
    It reuses ``fiducial_dilepton`` with the rapidity edges set to that fiducial range
    rather than to a per-row bin, which is what makes it a third arm rather than a
    special case -- the rapidity edges come from the **cuts record**, not the
    kinematics table.

    ``rec["Y_low"]``/``rec["Y_high"]`` must therefore be supplied by the layout from
    ``fiducial_cuts_for(...)``, and ``Q2_low``/``Q2_high`` from the mass bin edges.
    """
    from pixel.data import DrellYan
    from pixel.data.fiducial_cuts import lepton_cuts_for

    for edge in ("Q2_low", "Q2_high", "Y_low", "Y_high"):
        if rec.get(edge) is None:
            raise ValueError(
                f"{rec.get('label')!r}: build_drell_yan_mass_spectrum needs {edge}; the "
                "mass edges come from the kinematics table and the rapidity edges from "
                "the dataset's fiducial-cut record, and neither may be defaulted"
            )
    fields_A, fields_B = cfg.dy_field_maps(rec["reaction"])
    evolution_A = evolution_B = None
    if cfg.DY_EVOLVE:
        fields_A, fields_B, evolution_A, evolution_B = dy_evolution_maps(rec, fields)
    kwargs = {}
    if "mean" in rec and "cov" in rec:
        kwargs["mean"], kwargs["cov"] = rec["mean"], rec["cov"]
        if rec.get("rel_norm") is not None:
            kwargs["normalization"] = float(rec["rel_norm"])
            kwargs["fit_normalization"] = cfg.DY_FIT_NORMALIZATION
    return DrellYan(
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        Q2_low=rec["Q2_low"], Q2_high=rec["Q2_high"],
        Y_low=rec["Y_low"], Y_high=rec["Y_high"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=5,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        observable_mode="fiducial_dilepton",
        lepton_cuts=lepton_cuts_for(rec["nnpdf_legacy_name"]),
        source="closure", component=rec["reaction"],
        cache_dir=str(cache.parent) if cache is not None else None,
        **kwargs,
    )


def _dy_section_builder(builder):
    """Adapt a Drell-Yan builder to the ``datasets``-section calling convention.

    The ``datasets`` section dispatches through :data:`_EXP_BUILDERS`, which is a *DIS*
    convention: every builder there takes ``with_exp_nuisances`` and ``t0``, because a
    linear structure-function row carries a fitted normalization and a t0 reference.
    The Drell-Yan builders take neither -- ``build_drell_yan`` reads its normalization
    from ``rec["rel_norm"]`` and the DY t0 iteration is not wired -- so calling one
    through that path raises ``TypeError`` on the first keyword.

    Rather than widen two signatures with parameters they would have to ignore, this
    adapter drops exactly those two and is explicit about why. It is **not** a silent
    discard: a caller that passes a real ``t0`` for a DY row is asking for something
    that does not exist yet, and gets told so.
    """
    def build(rec, path, fields, cache, *, with_systematics=True,
              with_exp_nuisances=True, t0=None):
        if t0 is not None:
            raise NotImplementedError(
                f"t0 iteration is not implemented for the Drell-Yan builder "
                f"{builder.__name__}; the multiplicative-uncertainty reference for a "
                "bilinear row is not the same object as a DIS row's and must not be "
                "silently reused"
            )
        return builder(rec, path, fields, cache, with_systematics=with_systematics)

    build.__name__ = f"{builder.__name__}_datasets_section"
    return build


def _gated_builder(name):
    """A builder for a datatype the closure cannot build yet.

    Registered in :data:`_EXP_BUILDERS` exactly like a real one, so every datatype JAM24
    or NNPDF4.0 uses has an entry and the dispatch table is the full list rather than
    the working subset. Calling one raises
    :class:`pixel.data.closure_datatypes.DatatypeGateError` naming what blocks it and
    the plan section that says how to fix it.

    **The point is the failure mode.** Without an entry, a manifest record for a blocked
    kind dies on ``unknown experimental closure kind``, which says nothing about whether
    the datatype is unimplemented, out of scope, or simply misspelled. With one, the
    error is the reason.
    """
    def build(rec, path, fields, cache, **kwargs):
        assert_datatype_ready(name)
        raise AssertionError(f"unreachable: {name} is registered as ready but has no builder")

    build.__name__ = f"gated_{name}"
    build.__doc__ = f"Gated: see pixel.data.closure_datatypes for what blocks {name!r}."
    return build


_EXP_BUILDERS = {
    "f2": build_f2,
    "sigma_r": build_sigma_r,
    "sigma_r_cc": build_sigma_r_cc,
    "sigma_r_heavy": build_sigma_r_heavy,
    "drell_yan": _dy_section_builder(build_drell_yan),
    "drell_yan_binned": _dy_section_builder(build_drell_yan_binned),
    "drell_yan_fiducial_lepton": _dy_section_builder(build_drell_yan_fiducial_lepton),
    "drell_yan_bin_integrated": _dy_section_builder(build_drell_yan_bin_integrated),
    "drell_yan_mass_spectrum": _dy_section_builder(build_drell_yan_mass_spectrum),
}

# Every datatype JAM24 or NNPDF4.0 uses gets an entry, whether or not it builds today.
# The blocked ones raise with their own reason (see pixel.data.closure_datatypes); a
# missing entry would instead raise "unknown experimental closure kind", which cannot
# distinguish "not implemented" from "misspelled".
_EXP_BUILDERS.update({
    name: _gated_builder(name)
    for name, gate in DATATYPE_GATES.items()
    if gate.status != "ready" and name not in _EXP_BUILDERS
})


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
