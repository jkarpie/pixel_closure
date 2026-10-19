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
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from pixel.data.closure_datatypes import (
    DATATYPE_GATES,
    assert_datatype_ready,
)

from pixel import data, kernels
from pixel.data import families
from pixel.core.model import Contribution
from pixel.util.flavor import NuclearTarget

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


def build_sigma_r_cc(rec, path, fields, cache, *, with_systematics=True,
                     with_exp_nuisances=True, t0=None):
    kwargs = dict(_exp_nuisance_kwargs(rec, path, enabled=with_exp_nuisances, t0=t0))
    if "y" in rec:
        kwargs["y"] = rec["y"]
    elif "sqrt_s" in rec:
        kwargs["sqrt_s"] = float(rec["sqrt_s"])
    return data.ChargedCurrentSigmaR.from_file(
        path, target=_record_target(rec), maps_to=_even_map(fields),
        valence_maps_to=_odd_map(fields), beam_charge=rec["beam"], nf=cfg.NF,
        Q20=cfg.Q0_2, order=cfg.ORDER, mode=cfg.MODE, mc=cfg.MC, mb=cfg.MB,
        momentum_density=True,
        low_x_extensions=cfg.even_low_x_completions(),
        valence_low_x_extensions=cfg.odd_low_x_completions(),
        name=Path(rec["file"]).stem, cache_path=cache,
        **kwargs,
    )


def _target_transform(reaction):
    """The side-B projection for a DY reaction, as a function of the side-A one."""
    from pixel.kernels.evolution.flavor_transition import (
        conjugate_projection,
        isoscalar_projection,
        nuclear_projection,
    )

    if reaction == "pp":
        return lambda projection: projection
    if reaction == "pd":
        return isoscalar_projection
    if reaction == "ppbar":
        return conjugate_projection
    if reaction == "pCu":
        # Isospin only; the nuclear uncertainty rides in the _dw_ covariance variant.
        return lambda projection: nuclear_projection(
            projection, cfg.DY_COPPER_PROTON_FRACTION
        )
    raise ValueError(f"unsupported DY reaction {reaction!r}")


def _slot_structure(projection):
    """Slots grouped by shared operator object, plus every slot's weight.

    A slot is ``(parton, source, term_index)``.  Two slots sharing an operator at one
    scale share it at every scale (the projection is built deterministically), which is
    what lets a scale-callable projection keep the model's block-sparse grouping on
    ``id(evolution)`` instead of exploding it -- the same invariant
    ``rowwise_parton_evolution_projection`` preserves for the row-wise stack.
    """
    groups, order = {}, []
    for parton in sorted(projection.evolution):
        for source in sorted(projection.evolution[parton]):
            for index, (_, operator) in enumerate(projection.evolution[parton][source]):
                if id(operator) not in groups:
                    groups[id(operator)] = []
                    order.append(id(operator))
                groups[id(operator)].append((parton, source, index))
    weights = {
        (parton, source, index): float(weight)
        for parton, terms in projection.evolution.items()
        for source, values in terms.items()
        for index, (weight, _) in enumerate(values)
    }
    return tuple(tuple(groups[key]) for key in order), weights


class _ScaleOperator:
    """One evolution operator as a function of the hard scale ``Q2``.

    PIXEL's binned Drell-Yan kernels evaluate evolution at every mass quadrature node and
    accept a matrix or a callable of ``Q2`` there (``DYBinIntegratedKernel._evolution_at``).
    """

    def __init__(self, at, slot, at_many=None):
        self._at = at
        self._slot = slot
        self._at_many = at_many

    def __call__(self, Q2):
        parton, source, index = self._slot
        return self._at(Q2).evolution[parton][source][index][1]

    def prefetch(self, Q2s):
        """Build the projection at every scale in ``Q2s`` not built yet, in one batch.

        PIXEL's binned Drell-Yan kernels call this with every mass node before evaluating
        any (``DYBinIntegratedKernel.evolved_tensors``).  Projections at different scales
        share their contour work, so building them together is far cheaper than one
        ``__call__`` at a time; every operator of one ``_scale_evolution`` shares the
        cache, so the first ``prefetch`` serves them all.
        """
        if self._at_many is not None:
            self._at_many(Q2s)

    def __repr__(self):
        return f"_ScaleOperator{self._slot}"


def _scale_evolution(project, reference_Q2, project_many=None):
    """``(fields, evolution, nf)`` with every operator a cached callable of ``Q2``.

    ``project(Q2)`` builds the projection at one scale; ``project_many(Q2s)``, when given,
    builds several at once, which is what the operators' ``prefetch`` uses.  Each scale is
    built once and held; a scale whose flavour number or term structure differs from the
    reference's -- a bin straddling the bottom threshold -- raises rather than silently
    mixing a four- and a five-flavour parton set, exactly as the row-wise stack refuses one.
    """
    cache = {}

    def admit(key, projection):
        if cache and (
            int(projection.nf) != int(reference.nf)
            or _slot_structure(projection) != structure
        ):
            raise ValueError(
                f"Drell-Yan evolution at Q2={key:.6g} has a different flavour number "
                f"or term structure than at the reference Q2={float(reference_Q2):.6g}"
            )
        cache[key] = projection

    def at(Q2):
        key = float(Q2)
        if key not in cache:
            admit(key, project(key))
        return cache[key]

    def at_many(Q2s):
        keys = [float(Q2) for Q2 in np.asarray(Q2s, dtype=float).reshape(-1)]
        missing = [key for key in dict.fromkeys(keys) if key not in cache]
        if missing:
            built = (
                project_many(missing) if project_many is not None
                else [project(key) for key in missing]
            )
            for key, projection in zip(missing, built):
                admit(key, projection)

    reference = project(float(reference_Q2))
    cache[float(reference_Q2)] = reference
    structure = _slot_structure(reference)
    partition, weights = structure
    evolution = {
        parton: {source: [None] * len(values) for source, values in terms.items()}
        for parton, terms in reference.evolution.items()
    }
    for group in partition:
        operator = _ScaleOperator(at, group[0], at_many)
        for parton, source, index in group:
            evolution[parton][source][index] = (weights[(parton, source, index)], operator)
    return (
        {parton: dict(fields) for parton, fields in reference.fields.items()},
        {parton: {source: tuple(values) for source, values in terms.items()}
         for parton, terms in evolution.items()},
        int(reference.nf),
    )


def dy_evolution_maps(rec, fields):
    """Flavour projection maps for one Drell-Yan record.

    Returns ``(fields_A, fields_B, evolution_A, evolution_B)``.  Every row is evolved by
    the operator at its own hard scale -- one matrix at an effective mass is a different
    operator, not an approximation of the right one -- and **how** depends on the arm:

    * **pointwise** (``drell_yan``: E866, E605) -- one matrix per row, stacked row-wise
      (``rowwise_parton_evolution_projection``), which the pointwise kernel consumes;
    * **binned** (every other kind) -- a callable of ``Q2`` per operator.  These kernels
      integrate over mass and rapidity nodes and call the base kernel one node at a time,
      so a row-wise stack reaches them whole and is refused ("row-wise evolution_A must
      carry one matrix per kinematic row: n vs 1", measured 2026-09-10 on every binned
      arm).  A callable is evaluated at each node's own ``Q2``, which is also what makes
      a mass-integrated row exact rather than evolved at its bin centre.

    Above the bottom threshold the parton content is five-flavour and the ``Nf=4`` input
    fields reach ``b``/``bbar`` only through the generated ``T24``-like combination, which
    a fixed coefficient table (``config.dy_field_maps``) cannot express.
    """
    from pixel.kernels.evolution.flavor_transition import (
        nf4_parton_evolution_projection,
        nf4_parton_evolution_projections,
        rowwise_parton_evolution_projection,
    )

    # Every closure field shares one basis; take it from whichever is present.
    basis = next(iter(fields.values())).basis
    cache_dir = cfg.dy_evolution_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(
        Q20=cfg.Q0_2,
        order=cfg.DY_EVOLUTION_ORDER,
        mode=cfg.DY_EVOLUTION_MODE,
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
        low_x_extensions=cfg.dy_low_x_completions(),
        cache_dir=str(cache_dir),
    )
    transform = _target_transform(rec["reaction"])

    def require_nf(nf):
        if int(nf) != int(cfg.DY_NF):
            raise ValueError(
                f"DY record {rec.get('label')!r} projects to nf={nf} but the closure "
                f"luminosity is configured for nf={cfg.DY_NF}; the hard kernel would then "
                "skip the flavours evolution just generated"
            )

    if rec.get("kind", "drell_yan") == "drell_yan":
        proton = rowwise_parton_evolution_projection(
            basis, Q2_rows=np.asarray(rec["Q2"], dtype=float), **settings
        )
        require_nf(proton.nf)
        target = transform(proton)
        return proton.fields, target.fields, proton.evolution, target.evolution

    reference_Q2 = float(np.asarray(rec["Q2"], dtype=float)[0])
    proton_cache = {}

    def proton_at(Q2):
        if Q2 not in proton_cache:
            proton_cache[Q2] = nf4_parton_evolution_projection(basis, Q2=Q2, **settings)
        return proton_cache[Q2]

    def proton_many(Q2s):
        # Every mass node of a record at once: the scales share their contour work
        # (``nf4_parton_evolution_projections``), so this is far cheaper than node by node.
        missing = [Q2 for Q2 in dict.fromkeys(Q2s) if Q2 not in proton_cache]
        if missing:
            built = nf4_parton_evolution_projections(basis, Q2s=missing, **settings)
            proton_cache.update(zip(missing, built))
        return [proton_cache[Q2] for Q2 in Q2s]

    fields_A, evolution_A, nf = _scale_evolution(
        proton_at, reference_Q2, project_many=proton_many
    )
    require_nf(nf)
    if rec["reaction"] == "pp":
        return fields_A, fields_A, evolution_A, evolution_A
    fields_B, evolution_B, _ = _scale_evolution(
        lambda Q2: transform(proton_at(Q2)),
        reference_Q2,
        project_many=lambda Q2s: [transform(projection) for projection in proton_many(Q2s)],
    )
    return fields_A, fields_B, evolution_A, evolution_B


def build_drell_yan(rec, path, fields, cache, *, with_systematics=True):
    """Build a Drell-Yan bilinear dataset over the closure basis fields.

    ``rec`` carries the E866 kinematics (S, Q2, Y) and the reaction (``pp``/``pd``).
    The parton -> basis-field maps (``config.dy_field_maps``) let the shared
    :func:`pixel.data.DrellYan` builder expand the bilinear luminosity over the
    nine basis fields; ``maps_to`` field *objects* are passed so the analysis
    binds to the same field instances.  ``cache``/``path`` are unused here (the
    kinematics live in the manifest, not a ``.dat`` file), kept for a uniform
    builder signature.

    Photon exchange only -- no ``electroweak`` provider -- by owner decision D8
    (2026-09-11): E866 and E605 are fixed-target and stay photon-only, while every
    collider arm takes gamma*/Z (:func:`collider_exchange`).
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
        # Closure fields are momentum densities x*f (PIXEL fits x*f, never f).
        momentum_density=True,
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


def collider_exchange():
    """``{"electroweak": ..., "couplings": ...}`` for every collider Drell-Yan builder.

    **gamma*/Z exchange** -- owner decision D8, 2026-09-11 (PIXEL's
    ``plans/drell_yan_published_units.md``).  A collider dilepton row is gamma*/Z, not
    photon-only; fixed-target E866 and E605 stay photon-only in :func:`build_drell_yan`.

    The gamma*/Z weight table normalizes itself with a *running* ``alpha_EM(Q^2)``, and
    ``DYKernel._row_couplings`` refuses a hard prefactor using any other value, so the
    fixed ``cfg.DY_ALPHA_EM`` cannot pair with it.  PIXEL's per-row provider is
    :class:`~pixel.kernels.drell_yan.couplings.RunningCouplings`.  It takes ``alpha_EM``
    from the electroweak provider's own neutral-current object, so the two agree exactly,
    and ``alpha_s``/``nf`` from an :class:`~pixel.kernels.evolution.pqcd.alphas.AlphaS` configured as
    the DY evolution operators' coupling is (``dy_evolution_maps``).  **So these rows run
    ``alpha_s(Q^2)``, not the fixed ``cfg.DY_ALPHA_S``**: PIXEL has no provider pairing a
    fixed ``alpha_s`` with a running ``alpha_EM``, and the builders' fixed ``alpha_s``,
    ``nf`` and ``alpha_em`` arguments are ignored once ``couplings`` is given.

    The provider's Z pole also moves the bin integration's mass axis onto its
    Breit-Wigner panel (``DYBinIntegratedKernel._mass_resonance``).
    """
    from pixel.kernels.drell_yan import DrellYanElectroweak, RunningCouplings
    from pixel.kernels.evolution.pqcd.alphas import AlphaS

    electroweak = DrellYanElectroweak("gamma_z")
    couplings = RunningCouplings(
        alphas=AlphaS(cfg.DY_EVOLUTION_ORDER, alphaS_MZ=cfg.ALPHAS_MZ, mc=cfg.MC, mb=cfg.MB),
        eweak=electroweak.neutral,
    )
    return {"electroweak": electroweak, "couplings": couplings}


def build_drell_yan_binned(rec, path, fields, cache, *, with_systematics=True):
    """Build a **bin-integrated fiducial** Drell-Yan dataset (collider Z rapidity).

    The bilinear sibling of :func:`build_drell_yan`.  E866 is a *pointwise* fixed-target
    measurement: one (S, Q2, Y) per row.  A collider Z measurement is not -- each row is
    a cross section integrated over a rapidity bin (and, for the 2D sets, a mass bin)
    inside a *fiducial* lepton acceptance, so three things change and all three are
    settings on the same :func:`pixel.data.DrellYan` builder:

    * ``observable_mode="fiducial_dilepton"`` instead of the default ``"pointwise"``;
    * bin edges ``Y_low``/``Y_high`` and ``Q2_low``/``Q2_high``.  A rapidity scan passes
      its published mass window as the ``Q2`` edges: the measurement integrates
      ``60-120`` or ``66-116 GeV``, not the pole with zero width (owner decision D7,
      2026-09-11, enforced by ``generate._collider_dy_record``);
    * ``lepton_cuts``, resolved from ``nnpdf40_fiducial_cuts.yaml`` by the dataset's own
      NNPDF name.  This is **required**, not defaulted: the builder refuses
      ``fiducial_dilepton`` without it, because a wrong acceptance is a silent
      normalization error that a chi-squared cannot see.

    The exchange is gamma*/Z with running couplings, as on every collider arm
    (:func:`collider_exchange`).

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
        # Closure fields are momentum densities x*f (PIXEL fits x*f, never f).
        momentum_density=True,
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=5,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        observable_mode="fiducial_dilepton",
        **collider_exchange(),
        lepton_cuts=lepton_cuts_for(
            rec["nnpdf_legacy_name"], channel=rec.get("fiducial_channel")
        ),
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
    return HeavyNCSigmaRFONLL.from_file(
        path,
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
        # Closure fields are momentum densities x*f (PIXEL fits x*f, never f).
        momentum_density=True,
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=5,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        observable_mode="fiducial_lepton",
        lepton_cuts=lepton_level_cuts_for(
            rec["nnpdf_legacy_name"], charge=int(charge),
            channel=rec.get("fiducial_channel"),
        ),
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
        # Closure fields are momentum densities x*f (PIXEL fits x*f, never f).
        momentum_density=True,
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=5,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        observable_mode="bin_integrated",
        **collider_exchange(),
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
        # Closure fields are momentum densities x*f (PIXEL fits x*f, never f).
        momentum_density=True,
        name=rec.get("name", rec["label"]),
        Q2=rec["Q2"], S=rec["S"], Y=rec["Y"],
        Q2_low=rec["Q2_low"], Q2_high=rec["Q2_high"],
        Y_low=rec["Y_low"], Y_high=rec["Y_high"],
        fields_A=fields_A, fields_B=fields_B,
        order=cfg.DY_ORDER, alpha_s=cfg.DY_ALPHA_S, nf=5,
        alpha_em=cfg.DY_ALPHA_EM, channels=cfg.DY_CHANNELS,
        evolution_A=evolution_A, evolution_B=evolution_B,
        observable_mode="fiducial_dilepton",
        **collider_exchange(),
        lepton_cuts=lepton_cuts_for(
            rec["nnpdf_legacy_name"], channel=rec.get("fiducial_channel")
        ),
        source="closure", component=rec["reaction"],
        cache_dir=str(cache.parent) if cache is not None else None,
        **kwargs,
    )


_LATTICE_BUILDERS = {"pseudoitd": build_pseudoitd, "mellin": build_mellin}
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


#: Linear builders: the ``exp`` and ``datasets`` manifest sections.
_EXP_BUILDERS = {
    "f2": build_f2,
    "sigma_r": build_sigma_r,
    "sigma_r_cc": build_sigma_r_cc,
    # Heavy-flavour-tagged NC reduced cross section (HERA sigma_r^bbbar).
    "sigma_r_heavy": build_sigma_r_heavy,
}

#: Bilinear builders: the ``dy`` manifest section, one per Drell-Yan observable.
#:
#: **A separate table since 2026-09-10, and a separate section.**  Until then the NNPDF
#: Drell-Yan records sat in the generic ``datasets`` section beside CHORUS and reached
#: these builders through a DIS-convention adapter.  That section is gated as one unit
#: (``generic_section_family``), so a DIS + DY mix made every fit raise, and generation
#: sliced its operator list positionally across the two.  The family registry already
#: routes every ``dy.``/``z.``/``w.`` contract to the ``dy`` section; the records now
#: live there, and ``build_datasets`` dispatches that section on each record's kind.
_DY_BUILDERS = {
    "drell_yan": build_drell_yan,
    "drell_yan_binned": build_drell_yan_binned,
    "drell_yan_fiducial_lepton": build_drell_yan_fiducial_lepton,
    "drell_yan_bin_integrated": build_drell_yan_bin_integrated,
    "drell_yan_mass_spectrum": build_drell_yan_mass_spectrum,
}
DY_KINDS = tuple(_DY_BUILDERS)

# Every datatype JAM24 or NNPDF4.0 uses gets an entry, whether or not it builds today.
# The blocked ones raise with their own reason (see pixel.data.closure_datatypes); a
# missing entry would instead raise "unknown experimental closure kind", which cannot
# distinguish "not implemented" from "misspelled".
_EXP_BUILDERS.update({
    name: _gated_builder(name)
    for name, gate in DATATYPE_GATES.items()
    if gate.status != "ready" and name not in _EXP_BUILDERS
})


def dy_builder(kind):
    """The builder for one ``dy``-section record kind.

    A Drell-Yan datatype the registry blocks gets its gate, so the error names what
    blocks it; anything else unknown raises rather than falling back to the pointwise
    builder, which would build a different observable without complaint.
    """
    gate = DATATYPE_GATES.get(kind)
    if gate is not None and gate.status != "ready":
        # Checked first: a registered builder whose datatype the registry blocks must
        # not build -- its gate names why.
        return _gated_builder(kind)
    if kind in _DY_BUILDERS:
        return _DY_BUILDERS[kind]
    raise ValueError(f"unknown Drell-Yan closure kind {kind!r}")


def _experimental_builder(kind):
    if kind in _DY_BUILDERS:
        raise ValueError(
            f"{kind!r} is a Drell-Yan kind: its records belong in the 'dy' manifest "
            "section and are built by dy_builder, not by the linear sections"
        )
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


def generic_section_family(q_key: str) -> str | None:
    """The family owning the generic ``datasets`` manifest section, from its records.

    ``families.GENERIC_MANIFEST_SECTION`` exists precisely because these records name
    their own builder instead of a family, and until this function nothing routed a
    closure mode to it: ``fit.py`` aliased ``include_nnpdf = include_exp``.  That was
    correct only by accident -- every record in the section happens to be DIS today --
    and a DY or lattice record landing here would have been gated by the DIS flag with
    nothing to notice.

    Each record carries a ``contract_id``, and the registry already routes contract ids
    to families, so the answer is derived rather than assumed.

    Returns:
        The owning family key, or ``None`` when the section is absent or empty.

    Since 2026-09-10 the section holds linear DIS only: every Drell-Yan record lives in
    ``dy``, where its family routes it.  A Drell-Yan record found here is a generator
    regression, and the mixed-family raise below is what reports it.

    Raises:
        FamilyError: If a record's ``contract_id`` belongs to no family.
        ValueError: If the section spans more than one family.  Generation splits its
            operator list *positionally* against
            ``lattice_layout() + exp_layout() + nnpdf_layout()``, so the section is
            included or excluded as a unit; a mixed section has no correct gate, and
            guessing one is the failure this function exists to prevent.
    """
    records = load_manifest(q_key).get(families.GENERIC_MANIFEST_SECTION, [])
    keys = {families.family_for_contract_id(rec["contract_id"]) for rec in records}
    if not keys:
        return None
    if len(keys) > 1:
        raise ValueError(
            f"the {families.GENERIC_MANIFEST_SECTION!r} manifest section spans families "
            f"{sorted(keys)}; it is gated as a unit, so a mixed section cannot be routed. "
            "Split it, or give the second family its own section."
        )
    return keys.pop()


def _lattice_cache_stem(rec, path) -> str:
    """Kernel-cache stem for a lattice record (namespaced by ensemble id)."""
    ens = rec.get("ensemble", "ens")
    return f"{ens}_{path.stem}"


def build_datasets(q_key, fields, *, include_lattice=True, include_exp=True,
                   include_dy=False, include_nnpdf=False, use_kernel_cache=True,
                   with_systematics=True, with_exp_nuisances=True, t0=None):
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
    ``include_dy`` adds the ``dy`` section: the fitpack E866 tables and every
    NNPDF-commondata Drell-Yan record, fixed-target and collider, each built by
    :func:`dy_builder` for its kind.  ``include_nnpdf`` adds the NNPDF-commondata
    ``datasets`` section, which is linear DIS only since 2026-09-10 (CHORUS and the
    HERA bottom set).

    ``include_nnpdf`` defaults to **False**, opt-in like ``include_dy``, and that default
    is load-bearing rather than stylistic.  ``fit.py`` passes the other three flags
    explicitly from the mode; a fourth defaulting to ``True`` would put 1,040 rows of
    neutrino DIS into a ``lattice``-only fit, which is precisely the silent inclusion the
    contract machinery exists to prevent.  Generation asks for it explicitly.

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
    if include_nnpdf:
        # The NNPDF-commondata ``datasets`` section.  These are linear DIS operators
        # exactly like the ``exp`` ones and share their builders; they are a separate
        # manifest section only because their contracts select on ``commondata_name``
        # rather than a fitpack table id.  They must stay **after** ``exp`` and before
        # ``dy``: ``generate.generate_member`` pairs its operator list with
        # ``lattice_layout() + exp_layout() + nnpdf_linear`` and checks every pair by name.
        for rec in manifest.get("datasets", []):
            path = truth_dir / rec["file"]
            cache = _cache_path("datasets", path.stem, enabled=use_kernel_cache)
            label = rec.get("name", Path(rec["file"]).stem)
            out.append(_experimental_builder(rec["kind"])(
                rec, path, fields, cache, with_systematics=with_systematics,
                with_exp_nuisances=with_exp_nuisances,
                t0=None if t0 is None else t0.get(label)))
    if include_dy:
        for rec in manifest.get("dy", []):
            cache = _cache_path("dy", rec["label"], enabled=use_kernel_cache)
            out.append(dy_builder(rec["kind"])(rec, None, fields, cache,
                                               with_systematics=with_systematics))
    return out
