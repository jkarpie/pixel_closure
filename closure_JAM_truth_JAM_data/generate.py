"""Fold one fixed JAM truth through the PIXEL forward operators and save data.

For one original scale ``Q`` (``q_key``):

1. evaluate one frozen JAM member on the shared field nodes (assigned at
   ``Q0 = mc``), and evaluate the replica ensemble separately only to calibrate
   the fixed lattice covariance;
2. lay out the **per-ensemble** lattice pseudo-ITD / Mellin-moment kinematics
   (all thirteen NME ensembles, ``z = 1..ceil(0.3fm/a)``, ``p = 1..floor(L/6)``)
   and the analysis DIS kinematics (from ``dis_manifest.json``), writing
   placeholder ``.dat`` files;
3. build the PIXEL datasets with :mod:`closure_JAM_truth.datasets` (the *same*
   builders the fit uses, but **without** the systematic nuisance contributions)
   and fold the one representable truth through them;
4. **Lattice**: use the fixed-member central and the **full** replica covariance
   as a fixed error model; inject the property-enforced systematic shift
   (parity in nu, zero at nu->0, nulled counting moment, <=10% of the signal);
   inflate the covariance diagonal by 1% and draw a correlated multivariate
   observation.  **DIS**: fixed-member central with the **real** relative statistical
   error ``sigma_i = rel_stat_i |central_i|`` (diagonal), jittered.
5. write ``truth.json`` (fixed-member curves + external replica std + injected
   systematic curves + reference fit) and ``manifest.json``.

Run standalone::

    python -m closure_JAM_truth.generate --Q 2          # one member
    python -m closure_JAM_truth.generate --all          # every Q in config
    python -m closure_JAM_truth.generate --random        # a random Q, seeded
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

from collections.abc import Mapping

import numpy as np

from pixel.core.model import iter_bilinear_group_tensors
from pixel.util.progress import Stopwatch, format_duration

from pixel import kernels
from pixel.data.contracts import (
    annotate_manifest,
    closure_extension_roster,
    contract_artifact_provenance,
)
from pixel.data.closure_audit import read_nnpdf_source
from pixel.data.experimental._dis_common import PROTON_MASS_GEV, dis_w2
from pixel.kernels import LATTICE_SYSTEMATICS

from . import config as cfg
from . import datasets as dsets
from . import pdf_guidance

#: DIS observable kinds (real rel_stat diagonal error + jitter); everything else
#: is lattice (full replica covariance + 1% diagonal inflation + systematics).
#: ``sigma_r_heavy`` joined 2026-09-10: the HERA bottom set is linear DIS and its fold is
#: the DIS fold, and leaving it out of this tuple is what let the old positional operator
#: slice pair records with operators built for other datasets.
EXP_KINDS = ("f2", "sigma_r", "sigma_r_cc", "sigma_r_heavy")
LATTICE_KINDS = ("pseudoitd", "mellin")
#: The Drell-Yan uncertainty floor, as a fraction of each record's largest |prediction|.
#:
#: ``cfg.EXP_ABS_FLOOR`` is an ABSOLUTE floor sized for DIS observables of order one.
#: PIXEL's Drell-Yan predictions are in natural units -- about 1e-7 for E866 and 1e-16
#: for a collider Z row at the pole -- where an absolute 1e-6 swamps the published
#: relative errors.  Measured 2026-09-10 on the committed small-suite E866 data: median
#: sigma/|mean| 2.2 against rel_stat 0.15, so every E866 row was drawn with about fifteen
#: times its stated error.  A floor relative to the record's own scale keeps its one job
#: -- no zero variance where rel_stat is zero or a prediction vanishes -- at any units.
DY_RELATIVE_FLOOR = 1.0e-6
#: Drell-Yan kinds, folded by their own arm into the ``dy`` manifest section.
DY_KINDS = dsets.DY_KINDS

#: fitpack target label -> PIXEL target name.
#:
#: The fitpack spreadsheets this suite's ``dis_audit`` reads write the
#: abbreviations ``p``/``d``, so the manifest is expanded once here rather than
#: shipping abbreviations downstream.  This is *readability*, not correctness:
#: :func:`pixel.util.flavor.normalize_target` -- reached from every DIS builder
#: through ``_dis_common`` -- already accepts both forms and raises ``ValueError``
#: on anything outside its alias table, so an unexpanded label would still build
#: the right operator and a wrong one would still fail loudly.  Keep this a strict
#: subset of PIXEL's aliases; ``tests/test_closure_charged_current_wiring.py``
#: asserts the two agree key for key.
#:
#: The two ``_small`` suites define no ``TARGET_MAP`` and need none: their
#: ``dis_audit`` writes ``spec.target`` from ``config.EXP_SPECS``, which is
#: already canonical (measured: all 7 ``used`` tables carry
#: ``proton``/``deuteron``, against ``p``/``d`` on all 13 here).
TARGET_MAP = {"p": "proton", "d": "deuteron", "n": "neutron",
              "proton": "proton", "deuteron": "deuteron", "neutron": "neutron"}


# -- forward operator (shared with the fit by construction) ------------------


def forward(ds, fields, truth_nodes) -> np.ndarray:
    """``sum_contributions kernel.matrix(ds.nu, field.basis) @ truth_field``."""
    mean = np.zeros(np.asarray(ds.mean).shape, dtype=float)
    for c in ds.contributions:
        B = np.asarray(c.kernel.matrix(ds.nu, fields[c.field].basis), dtype=float)
        mean = mean + B @ truth_nodes[c.field]
    return mean.reshape(-1)


def assemble_operator(ds, fields) -> "list[tuple[str, np.ndarray]]":
    """Assemble a dataset's ``(field, B)`` kernel matrices **once**.

    ``kernel.matrix`` is the expensive call, so it must be invoked exactly once
    per kernel.  The dense arrays are reused for every truth member and replica.
    Datasets are built without systematics here, so a lattice operator has a
    single physical ``(field, B)`` entry.
    """
    return [
        (c.field, np.asarray(c.kernel.matrix(ds.nu, fields[c.field].basis), dtype=float))
        for c in ds.contributions
    ]


def fold_ensemble(operator, ens) -> np.ndarray:
    """Fold the replica ensemble through a pre-assembled ``operator`` -> ``(n_rep, n_pts)``."""
    field0, B0 = operator[0]
    Y = ens[field0] @ B0.T
    for field, B in operator[1:]:
        Y = Y + ens[field] @ B.T
    return Y


def fold_truth(operator, truth) -> np.ndarray:
    """Fold one representable truth through a pre-assembled linear operator."""
    field0, B0 = operator[0]
    result = B0 @ np.asarray(truth[field0], dtype=float)
    for field, B in operator[1:]:
        result = result + B @ np.asarray(truth[field], dtype=float)
    return np.asarray(result, dtype=float).reshape(-1)


# -- covariance + draws ------------------------------------------------------


def inflate_diagonal(cov, frac) -> np.ndarray:
    """Add a ``frac`` (e.g. 1%) increase to the diagonal: ``C_ii -> C_ii (1+frac)``.

    The full replica covariance of the folded smooth observables is
    rank-deficient / ill-conditioned; this nugget regularizes it to SPD.
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    cov = 0.5 * (cov + cov.T)
    return cov + np.diag(float(frac) * np.diag(cov))


def draw_mvn(mean, cov, rng) -> np.ndarray:
    """Correlated observation from ``MultivariateNormal(mean, cov)``."""
    return rng.multivariate_normal(np.asarray(mean, dtype=float),
                                   np.asarray(cov, dtype=float))


def draw_diag(mean, sigma, rng) -> np.ndarray:
    """Uncorrelated observation from ``Normal(mean, sigma)`` (diagonal jiggle)."""
    return np.asarray(mean) + np.asarray(sigma) * rng.standard_normal(len(mean))


# -- injected lattice systematics --------------------------------------------


def _pheno(x, N, a, b) -> np.ndarray:
    """Momentum-density shape ``N x^a (1-x)^b`` with ``a > 0`` (vanishes at x=0)."""
    x = np.clip(np.asarray(x, dtype=float), 1e-9, 1.0 - 1e-9)
    return N * x**a * (1.0 - x) ** b


def _mellin_n1_row(basis) -> np.ndarray:
    """The ``n = 1`` Mellin functional (``int f = int q/x``) on the field basis.

    Nulling a systematic curve against this row makes its counting moment (and
    the real cosine-ITD value at ``nu = 0``) vanish -- so the injected systematic
    never shifts the exactly-known ``int dx`` sum-rule moments, and starts from
    zero at ``nu = 0`` for the real component.
    """
    mell = kernels.Mellin(
        alpha=-1.0,
        low_x_extension={"kind": "power", "alpha": cfg.LOW_X_LINEAR_POWER},
    )
    return np.asarray(mell.matrix(np.array([1.0]), basis), dtype=float).reshape(-1)


def build_systematic_curves(nodes, basis, rng) -> "dict[str, dict[str, np.ndarray]]":
    """One nulled x-space systematic curve per (physical field, active key).

    Each curve is projected to **zero n=1 moment** and, for singlet/gluon fields
    whose ITD uses ``alpha=0``, independently to zero real ITD at ``nu=0``.
    (For quark fields using ``alpha=-1`` those two constraints are identical.)
    Amplitudes are normalized here; the <=10% cap is applied later once the
    folded signal is known.  Parity and large-``nu`` decay are automatic from
    reusing the field's cosine/sine transform for the nuisance contribution.
    """
    m1 = _mellin_n1_row(basis)
    itd_zero = np.asarray(
        kernels.PseudoITDReal(alpha=0.0, low_x_extension="flat").matrix(
            np.array([0.0]), basis
        ),
        dtype=float,
    ).reshape(-1)
    nulling_shapes = np.stack(
        (
            _pheno(nodes, 1.0, 0.7, 2.0),
            _pheno(nodes, 1.0, 1.1, 3.0),
        )
    )
    out: dict[str, dict[str, np.ndarray]] = {}
    for field in cfg.ALL_FIELDS:
        constraints = [m1]
        if not cfg.itd_momentum_density(field):
            constraints.append(itd_zero)
        C = np.stack(constraints)
        # Subtract smooth endpoint-vanishing shapes instead of projecting through
        # C.T directly.  The latter satisfies the moments but can introduce a
        # non-vanishing low-x endpoint into an alpha=-1 curve, spoiling the
        # intended Fourier decay.
        bases = nulling_shapes[: C.shape[0]]
        constraint_matrix = C @ bases.T
        out[field] = {}
        for key in cfg.ITD_SYSTEMATICS:
            N = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
            a = rng.uniform(0.2, 0.6)
            b = rng.uniform(1.0, 4.0)
            raw = _pheno(nodes, N, a, b)
            multipliers = np.linalg.solve(constraint_matrix, C @ raw)
            s = raw - multipliers @ bases
            out[field][key] = s
    return out


def build_moment_systematic_values(rng) -> "dict[str, float]":
    """Independent unit-normal scalar truth for every field/key/Mellin power."""
    return {
        cfg.moment_nuisance_field_name(field, key, order): float(
            rng.standard_normal()
        )
        for field, key, order in cfg.moment_nuisance_specs()
    }


def coefficient(key, meta) -> np.ndarray:
    """The exact registry systematic coefficient the fit will rebuild (per row)."""
    _, factory = LATTICE_SYSTEMATICS[key]
    return np.asarray(factory(meta).vector, dtype=float)


def lattice_meta(rec) -> dict:
    """Reconstruct the kinematic ``meta`` a systematic factory consumes."""
    if rec["kind"] == "pseudoitd":
        z = np.asarray(rec["_z"], dtype=float)
        p = np.asarray(rec["_p"], dtype=float)
        L = float(rec["L"])
        nu = (cfg.TWO_PI / L) * p * z
        return {"nu": nu, "z": z, "p": p, "z2": (z * float(rec["a"])) ** 2,
                "L": L, "a": float(rec["a"]), "m_pi": float(rec["m_pi"])}
    n = np.asarray(rec["_n"], dtype=float)
    return {"nu": n, "n": n, "L": float(rec["L"]), "a": float(rec["a"]),
            "m_pi": float(rec["m_pi"])}


class LatticeSystematicFolds(NamedTuple):
    """Pure result of folding lattice replicas and scaling injected systematics."""

    central: list[np.ndarray]
    covariance: list[np.ndarray]
    raw_shift: list[dict[str, np.ndarray]]
    max_ratio: dict[tuple[str, str], float]
    scale: dict[tuple[str, str], float]
    curves: dict[str, dict[str, np.ndarray]]
    moment_max_ratio: dict[str, float]
    moment_scale: dict[str, float]
    moment_values: dict[str, float]


def fold_lattice_systematics(
    records,
    operators,
    ens,
    sys_curves,
    moment_values=None,
    *,
    truth_curves=None,
    max_fraction: float = cfg.SYSTEMATIC_MAX_FRACTION,
) -> LatticeSystematicFolds:
    """Fold replicas and enforce the global per-field systematic-shift cap.

    This is the side-effect-free core of closure-data generation.  Keeping the
    cap calculation here lets tests exercise the exact production path without
    writing placeholder datasets or generated observations.
    """
    if len(records) != len(operators):
        raise ValueError("lattice records and operators must have equal length")
    if not 0.0 <= float(max_fraction) <= 1.0:
        raise ValueError("max_fraction must lie in [0, 1]")

    central_phys: list[np.ndarray] = []
    cov_phys: list[np.ndarray] = []
    raw_shift: list[dict[str, np.ndarray]] = []
    max_ratio: dict[tuple[str, str], float] = {}
    moment_max_ratio: dict[str, float] = {}
    moment_values = dict(moment_values or {})

    for rec, operator in zip(records, operators):
        if not operator:
            raise ValueError(f"{rec.get('name', 'lattice record')}: empty operator")
        field, B = operator[0]
        Y = fold_ensemble(operator, ens)
        central = Y.mean(axis=0) if truth_curves is None else fold_truth(operator, truth_curves)
        central_phys.append(central)
        cov_phys.append(np.atleast_2d(np.cov(Y, rowvar=False)))
        meta_k = lattice_meta(rec)
        shifts: dict[str, np.ndarray] = {}
        denom = np.where(np.abs(central) > 0.0, np.abs(central), 1.0)
        for key in rec["systematics"]:
            if rec["kind"] == "mellin":
                nname = cfg.moment_nuisance_field_name(
                    field, key, int(rec["order"])
                )
                if nname not in moment_values:
                    raise KeyError(f"missing Mellin systematic truth {nname!r}")
                shift = coefficient(key, meta_k) * moment_values[nname]
            else:
                shift = coefficient(key, meta_k) * (B @ sys_curves[field][key])
            shifts[key] = shift
            ratio = float(np.max(np.abs(shift) / denom))
            if rec["kind"] == "mellin":
                moment_max_ratio[nname] = max(
                    moment_max_ratio.get(nname, 0.0), ratio
                )
            else:
                pair = (field, key)
                max_ratio[pair] = max(max_ratio.get(pair, 0.0), ratio)
        raw_shift.append(shifts)

    scale = {
        pair: (float(max_fraction) / ratio if ratio > 0.0 else 1.0)
        for pair, ratio in max_ratio.items()
    }
    curves = {
        field: {
            key: sys_curves[field][key] * scale.get((field, key), 1.0)
            for key in cfg.ITD_SYSTEMATICS
        }
        for field in cfg.ALL_FIELDS
    }
    moment_scale = {
        name: (float(max_fraction) / ratio if ratio > 0.0 else 1.0)
        for name, ratio in moment_max_ratio.items()
    }
    final_moment_values = {
        name: value * moment_scale.get(name, 1.0)
        for name, value in moment_values.items()
    }
    return LatticeSystematicFolds(
        central=central_phys,
        covariance=cov_phys,
        raw_shift=raw_shift,
        max_ratio=max_ratio,
        scale=scale,
        curves=curves,
        moment_max_ratio=moment_max_ratio,
        moment_scale=moment_scale,
        moment_values=final_moment_values,
    )


# -- JAM replica ensemble ----------------------------------------------------


def _ensemble_cache_path(q_key: str) -> Path:
    return cfg.REFERENCE_DIR / f"jam_ensemble_{cfg.truth_label(q_key)}.npz"


def ensemble_curves(q_key: str, nodes, *, members=cfg.JAM_REPLICA_MEMBERS,
                    use_cache: bool = True):
    """Return ``(curves, meta)`` for the JAM replica ensemble at scale ``q_key``.

    ``curves`` maps each field to a ``(n_replicas, n_nodes)`` array (one row per
    replica).  Cached to ``reference_pdfs/jam_ensemble_<label>.npz``.
    """
    nodes = np.asarray(nodes, dtype=float).reshape(-1)
    members = tuple(int(m) for m in members)
    cache = _ensemble_cache_path(q_key)
    if use_cache and cache.exists():
        data = np.load(cache, allow_pickle=True)
        cached_nodes = np.asarray(data["x_nodes"], dtype=float)
        cached_meta = json.loads(str(data["meta"]))
        expected_q = float(cfg.TRUTH_Q_CHOICES[q_key])
        if (cached_nodes.size == nodes.size
                and np.allclose(cached_nodes, nodes, rtol=1e-9, atol=0.0)
                and int(data["n_members"]) == len(members)
                and np.isclose(
                    float(cached_meta.get("q_effective", np.nan)),
                    expected_q, rtol=0.0, atol=1e-12,
                )):
            curves = {f: np.asarray(data[f], dtype=float) for f in cfg.ALL_FIELDS}
            cached_meta.pop("truth_kind", None)
            cached_meta["uncertainty_kind"] = (
                "replica_ensemble_covariance_calibration"
            )
            return curves, cached_meta

    stacks = {f: [] for f in cfg.ALL_FIELDS}
    meta = None
    for m in members:
        curves_m, meta = pdf_guidance.jam_truth_curves(nodes, q_key=q_key, member=m)
        for f in cfg.ALL_FIELDS:
            stacks[f].append(np.asarray(curves_m[f], dtype=float).reshape(-1))
    curves = {f: np.asarray(stacks[f], dtype=float) for f in cfg.ALL_FIELDS}

    meta = dict(meta)
    meta.pop("member", None)
    meta["members"] = f"{min(members)}..{max(members)}"
    meta["n_members"] = len(members)
    meta["uncertainty_kind"] = "replica_ensemble_covariance_calibration"

    cfg.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x_nodes=nodes, n_members=len(members),
             meta=json.dumps(meta), **curves)
    return curves, meta


# -- file IO -----------------------------------------------------------------


def write_dat(path: Path, leading, mean, cov) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([*[np.asarray(c, dtype=float) for c in leading],
                            np.asarray(mean, dtype=float), np.asarray(cov, dtype=float)])
    np.savetxt(path, rows, fmt="%.12e")


# -- kinematics layout -------------------------------------------------------


def lattice_layout():
    """Per-ensemble lattice records (pseudo-ITD grid + Mellin moment) for all fields."""
    out = []
    for e in cfg.active_ensembles():
        zvals = np.asarray(e.z_values, dtype=float)
        pvals = np.asarray(e.p_values, dtype=float)
        zz, pp = np.meshgrid(zvals, pvals, indexing="ij")
        z = zz.ravel()
        p = pp.ravel()
        for field in cfg.ALL_FIELDS:
            even = cfg.field_cparity(field) == "even"
            comp = "imag" if even else "real"
            out.append({
                "kind": "pseudoitd", "ensemble": e.id, "field": field,
                "component": comp, "momentum_density": cfg.itd_momentum_density(field),
                "L": e.L_sites, "a": e.a_fm, "m_pi": e.m_pi_gev,
                "systematics": list(cfg.itd_systematics_for(field)),
                "file": f"lattice/{e.id}/itd_{field}_{comp}.dat",
                "name": f"{e.id}_itd_{field}_{comp}",
                "_leading": (z, p), "_z": z, "_p": p,
            })
            order = cfg.mellin_order_for(field)
            n_arr = np.array([float(order)])
            out.append({
                "kind": "mellin", "ensemble": e.id, "field": field,
                "order": int(order),
                "L": e.L_sites, "a": e.a_fm, "m_pi": e.m_pi_gev,
                "systematics": list(cfg.moment_systematics_for(field)),
                "file": f"lattice/{e.id}/moments_{field}.dat",
                "name": f"{e.id}_mom_{field}",
                "_leading": (n_arr,), "_n": n_arr,
            })
    return out


def dy_layout():
    """Drell-Yan records from the audited E866 tables (used only).

    Ensures the DIS manifest carries the ``dy`` audit, then returns one record per
    modelable table with its (S, Q2, Y) kinematics and real ``rel_stat``.
    """
    from . import dy_audit

    manifest = json.loads(cfg.DIS_MANIFEST_PATH.read_text())
    dy_manifest = manifest.get("dy", {})
    if (
        not dy_manifest.get("has_norm_audit")
        or dy_manifest.get("max_rows", object()) != cfg.DY_MAX_ROWS
        # A manifest audited before E866's Rtau was read as sqrt(tau) carries the wrong
        # (S, Q2, Y) and the wrong retained rows; missing the marker is enough to redo it.
        or dy_manifest.get("tau_convention") != dy_audit.TAU_CONVENTION
    ):
        dy_audit.augment_manifest()
        manifest = json.loads(cfg.DIS_MANIFEST_PATH.read_text())
    out = []
    for tab in manifest["dy"]["tables"]:
        if tab.get("status") != "used":
            continue
        rec = {
            "kind": "drell_yan", "reaction": tab["reaction"], "label": tab["label"],
            "idx": tab["idx"], "name": tab["label"], "file": f"dy/{tab['label']}.dat",
            "S": list(tab["S"]), "Q2": list(tab["Q2"]), "Y": list(tab["Y"]),
            "rel_stat": list(tab["rel_stat"]),
        }
        # The real overall normalization uncertainty plus the offset injected into
        # it; both travel in the manifest so the fit can treat it as a nuisance
        # and the report can compare the recovered pull against the truth.
        if tab.get("rel_norm") is not None:
            rec["rel_norm"] = float(tab["rel_norm"])
            rec["norm_beta"] = cfg.dy_norm_beta(tab["idx"])
        out.append(rec)
    return out


def _one_truth_curve(truth, field) -> np.ndarray:
    """Return ``truth[field]`` as a 1-D curve, refusing a replica stack.

    ``np.outer`` flattens its arguments, so handing this fold a
    ``(n_replicas, n_nodes)`` ensemble would not raise -- it would silently build
    an ``(n_rep*n_nodes, n_rep*n_nodes)`` outer product and contract a tensor
    slice of it, or reduce to the ensemble second moment ``E[qA qB]`` if someone
    "fixed" the shape.  Neither has a single PDF behind it, so neither can define
    coverage for a bilinear observable.  A bilinear central value takes **one**
    truth; the shape is where that is enforced.

    Args:
        truth: ``field name -> (n_nodes,)`` truth curves.
        field: Field name to read.

    Returns:
        np.ndarray: The 1-D truth curve.

    Raises:
        ValueError: If the stored curve is not 1-D.
    """
    curve = np.asarray(truth[field], dtype=float)
    if curve.ndim != 1:
        raise ValueError(
            f"dy_central needs one truth curve per field; truth[{field!r}] has "
            f"shape {curve.shape}. A replica ensemble is not a bilinear central "
            "value -- its second moment E[qA qB] = E[qA]E[qB] + Cov(qA, qB) is "
            "reproduced by no single PDF and cannot define coverage."
        )
    return curve


def dy_central(dataset, truth, fields, reaction) -> np.ndarray:
    """Exact DY cross section of one fixed truth -> ``(n_pts,)``.

    Folds the **fully assembled** operator of every bilinear contribution --
    ``BilinearContribution.assemble`` composes each side's evolution exactly as
    the fit does -- against the basis-field truth curves the contribution names.

    This genericity is the closure invariant, not a refactor.  The previous form
    pre-combined the truth into *parton* curves and folded ``bilinear_tensor``,
    the bare hard tensor.  Composing evolution on the fit side alone would then
    have left generation folding an unevolved operator: both sides internally
    consistent, the closure silently meaningless, and no test failing.  Reading
    the operator off the contribution makes the two sides structurally the same
    object, and covers the hard, evolved, and bin-integrated kernels without a
    second physics implementation.

    An ensemble second moment is deliberately forbidden here: it generally has no
    single PDF truth and therefore cannot define coverage for a bilinear
    observable.  ``_one_truth_curve`` makes that a *raise* rather than a comment --
    ``np.outer`` flattens, so a replica stack passed here used to be a silent
    reshape.  The two ``_small`` suites did fold the second moment until
    2026-08-14 (``S0-05``); all four copies now run this body, and
    ``tests/test_closure_truth_representable.py`` parametrizes over all four.

    **Which truth kind ``truth`` carries is the caller's decision** --
    ``fixed_lhapdf_member`` (``cfg.FIXED_TRUTH_MEMBER``, what this suite's
    ``generate_member`` passes) or ``replica_ensemble_mean`` (what the committed
    ``data/truthQ_*/truth.json`` files still record).  Nothing here depends on it:
    this folds whatever single curve the caller supplies.
    """
    del reaction  # the parton -> field map now travels on the contributions
    from dataclasses import replace

    # Every closure field shares one basis, so any of them supplies it; the
    # DY records may be built with a restricted field dict.
    a_basis = next(iter(fields.values())).basis
    # Group on the same key the block-sparse model uses.  Every field pair sharing
    # one (kernel, evolution) pair also shares one assembled operator, so the
    # weighted outer products are accumulated first and the expensive tensor is
    # built once per group rather than once per contribution.
    groups: dict = {}
    for c in dataset.bilinear_contributions:
        key = (id(c.kernel), id(c.evolution_A), id(c.evolution_B))
        curve_A = _one_truth_curve(truth, c.field_A)
        curve_B = _one_truth_curve(truth, c.field_B)
        outer = float(c.weight) * np.outer(curve_A, curve_B)
        if key in groups:
            groups[key][1] += outer
        else:
            groups[key] = [replace(c, weight=1.0), outer]

    central = np.zeros(int(dataset.n_data), dtype=float)
    # The iterator the fit's Model uses, so the two sides stay one object: every group
    # sharing a hard kernel is assembled in one pass, which builds that hard tensor once
    # instead of once per (kernel, evolution) group -- 30 instead of 680 on a collider
    # Z table at NLO (measured 2026-09-10).
    assembled = iter_bilinear_group_tensors(
        [contribution for contribution, _ in groups.values()],
        dataset.nu,
        lambda contribution: (a_basis, a_basis),
    )
    for (contribution, outer), tensor in zip(groups.values(), assembled):
        central += np.einsum(
            "rij,ij->r", np.asarray(tensor, dtype=float), outer, optimize=True
        )
    return central


def exp_layout():
    """DIS records from the audited analysis tables selected for closure."""
    manifest = json.loads(cfg.DIS_MANIFEST_PATH.read_text())
    out = []
    for tab in manifest["tables"]:
        if tab["status"] != "used":
            continue
        x = np.asarray(tab["x"], dtype=float)
        q2 = np.asarray(tab["Q2"], dtype=float)
        rel_stat = np.asarray(tab["rel_stat"], dtype=float)
        rec = {
            "kind": tab["kind"], "target": TARGET_MAP.get(tab["target"], tab["target"]),
            "label": tab["label"], "idx": tab["idx"], "file": f"exp/{tab['label']}.dat",
            "rel_stat": rel_stat.tolist(), "_leading": (x, q2),
        }
        # Both reduced-cross-section kinds need the beam charge, not just the
        # charged-current one.  ``build_sigma_r`` passes ``beam_charge=rec["beam"]``
        # for the NEUTRAL-current case too (datasets.py), because the beam charge
        # sets the sign of the gamma-Z interference term -- and this manifest's own
        # HERA NC tables really do carry both signs (idx 10030 is ``e_minus``, the
        # rest ``e_plus``).
        #
        # Writing it only for ``sigma_r_cc`` left ``build_sigma_r`` reading a key
        # its own generator never wrote: ``KeyError: 'beam'``, which aborted
        # full-suite generation entirely (measured 2026-08-16, both truths).  The
        # audit manifest carries ``beam`` for every table, so nothing else is
        # needed to supply it.
        if tab["kind"] in ("sigma_r", "sigma_r_cc"):
            rec["beam"] = tab["beam"]
        # Real experimental nuisances: the overall normalization (a scalar) and
        # the correlated systematic sources (a relative matrix in an .npz
        # sidecar).  Both are injected at generation and marginalized by the fit.
        if tab.get("rel_norm") is not None:
            rec["rel_norm"] = float(tab["rel_norm"])
            rec["norm_beta"] = cfg.dis_norm_beta(tab["idx"])
        if tab.get("correlated_file"):
            rec["correlated_file"] = tab["correlated_file"]
            rec["correlated_names"] = list(tab["correlated_names"])
            rec["_correlated_rel"] = np.asarray(
                np.load(cfg.DIS_SYS_DIR / tab["correlated_file"])["relative"],
                dtype=float,
            )
        if tab["kind"] in ("sigma_r", "sigma_r_cc"):
            if "Y" in tab:                                   # explicit inelasticity
                rec["y"] = list(tab["Y"])
            else:
                rs = np.asarray(tab.get("RS", []), dtype=float)
                if rs.size == x.size and np.all(rs > 0):
                    rec["y"] = (q2 / (x * rs * rs)).tolist()
        out.append(rec)
    return out


def nnpdf_layout():
    """Records for the ``datasets`` and ``dy`` manifest sections, from NNPDF commondata.

    **Two sections, one walk.**  A linear DIS record lands in ``datasets``; a Drell-Yan
    record -- fixed-target or collider -- lands in ``dy``, the section the family registry
    routes every ``dy.``/``z.``/``w.`` contract to, and is folded by its own arm in
    :func:`generate_member`.  Walking the selection once keeps the coverage guard seeing
    every selected id in one place.

    Walks the *contract selection* rather than a local list, so a dataset enters
    generation exactly when its contract says it should.  ``cfg.NNPDF_DATASET_SPECS``
    supplies only what the contract states as prose and the operator needs as numbers
    (the beam label and the target's A/Z); an id selected with no spec is left out
    here and caught by the coverage check in :func:`generate_member`.

    **Rows are cut, and the cut is the generator's own.**  CHORUS reaches down to
    ``Q2 = 0.263 GeV^2``, far below ``Q0_2 = 1.6384``, so evolving to it would mean
    running backwards out of the perturbative region.  The mask is the same one
    ``dis_audit.retained_rows`` applies to every other DIS table -- ``Q2 >= DIS_Q2_MIN``
    and ``W2 > DIS_W2_MIN`` -- and the retained count is what the contract's
    ``selected_rows`` records (**520 of 607 per beam**, 1,040 total).

    **Do not restore the 470 this docstring claimed until 2026-08-22.**  That count came
    from cutting at NNPDF's own ``Q0^2 = 1.65^2 = 2.7225`` instead of *this closure's*
    ``DIS_Q2_MIN = Q0_2 = mc^2 = 1.6384`` (``config.MC = 1.28``).  The generator's own
    coverage check rejected it and ``data/nnpdf_audit.json`` records 520; the prose
    outlived the fix.  Re-derive from the audit rather than quoting either number.

    **The uncertainty split is not the fitpack one and must not be forced into it.**
    A fitpack table's correlated sources are ``%``-type, i.e. multiplicative, which is
    why ``_exp_nuisance_kwargs`` passes them as ``multiplicative_systematics`` for t0
    referencing.  NNPDF's are typed individually, and CHORUS carries **1013 (nu) /
    113 (nb) additive** correlated sources plus exactly **one** multiplicative one.
    Additive sources are absolute numbers that no t0 iteration rescales, so they belong
    in the ``.dat`` covariance; only the genuine MULT source is handed to the nuisance
    machinery.  Routing the additive ones through ``multiplicative_systematics`` would
    silently rescale 75% of the nu variance by the theory prediction.

    One approximation is taken knowingly and is recorded per record as
    ``shared_multiplicative_sources``: ``global:CHORUSQEDRADCOR1`` is the *same* NNPDF
    nuisance in both beam records (correlated at r = 0.998 across them), but the DIS
    nuisance layer marginalizes each dataset's copy separately, so the cross-record
    block is dropped.  It carries a median 3-4% of each row's variance.
    """
    roster = closure_extension_roster(cfg.CONTRACT_MODE)
    out = []
    for entry in roster["selected"]:
        spec = cfg.NNPDF_DATASET_SPECS.get(entry["id"])
        if spec is None:
            continue
        common = read_nnpdf_source(cfg.NNPDF_COMMONDATA, spec["commondata_name"])
        # Drell-Yan is a different observable on the same commondata plumbing: bins in
        # rapidity or mass, not (x, Q2, y) DIS points, so it neither has nor wants the
        # W2 cut below.  Pointwise fixed-target first; every other DY kind is collider.
        if spec["kind"] == "drell_yan":
            out.append(_fixed_target_dy_record(entry, spec, common))
            continue
        if spec["kind"] in DY_KINDS:
            out.append(_collider_dy_record(entry, spec, common))
            continue
        x = np.asarray(common.kinematics["x"], dtype=float)
        q2 = np.asarray(common.kinematics["Q2"], dtype=float)
        y = np.asarray(common.kinematics["y"], dtype=float)
        value = np.asarray(common.value, dtype=float)
        w2 = dis_w2(x, q2, m_h=PROTON_MASS_GEV)
        keep = np.flatnonzero(
            np.isfinite(x) & np.isfinite(q2) & np.isfinite(y) & np.isfinite(value)
            & (x > 0.0) & (x < 1.0)
            & (q2 >= cfg.DIS_Q2_MIN) & (w2 > cfg.DIS_W2_MIN)
            & (value != 0.0)
        )
        if keep.size != int(entry["selected_rows"]):
            raise RuntimeError(
                f"{entry['id']}: the closure cut retains {keep.size} rows but the "
                f"contract records selected_rows={entry['selected_rows']}; one of the "
                "two has changed and generation must not paper over which"
            )

        # Additive sources: statistical (uncorrelated) plus correlated.  Both are
        # absolute in the commondata and both are stored here **relative to the
        # published central value**, because the closure's convention throughout is
        # "published relative error, applied to the folded truth" -- the folded truth
        # is not the published measurement and an absolute error bar carried over
        # unchanged would sit at the wrong scale.  ``rel_stat`` already works this way
        # for every fitpack table, so this keeps one convention rather than two.
        published = np.abs(value[keep])
        stat = np.sqrt(
            sum(np.asarray(v, dtype=float)[keep] ** 2 for v in common.uncorrelated.values())
        )
        correlated = [
            np.asarray(v, dtype=float)[keep] / published
            for v in common.correlated.values()
        ]
        additive = (
            np.stack(correlated, axis=1) if correlated
            else np.zeros((keep.size, 0), dtype=float)
        )
        rec = {
            "kind": spec["kind"],
            "section": "datasets",
            "label": spec["label"],
            "name": spec["label"],
            "contract_id": entry["id"],
            "commondata_name": spec["commondata_name"],
            # A nuclear target is a mapping (A/Z stay visible in the manifest); a
            # proton target is the plain string the builders already accept.
            "target": (dict(spec["target"]) if isinstance(spec["target"], Mapping)
                       else spec["target"]),
            "beam": spec["beam"],
            "file": f"datasets/{spec['label']}.dat",
            "y": y[keep].tolist(),
            "source_rows": int(x.size),
            "retained_rows": int(keep.size),
            "rel_stat": (stat / published).tolist(),
            "_leading": (x[keep], q2[keep]),
            "_additive_rel": additive,
        }
        # Heavy-flavour-tagged sets carry the quark they tag and its charge weight; the
        # inclusive arm has neither and the keys are simply absent there.
        for extra in ("heavy_flavour", "heavy_mass", "charge_sq"):
            if extra in spec:
                rec[extra] = spec[extra]
        multiplicative = {
            name: np.asarray(v, dtype=float)[keep]
            for name, v in common.multiplicative.items()
        }
        if multiplicative:
            rec["correlated_file"] = f"{spec['label']}.npz"
            rec["correlated_names"] = sorted(multiplicative)
            rec["_correlated_rel"] = np.stack(
                [multiplicative[name] for name in rec["correlated_names"]], axis=0
            )
            rec["shared_multiplicative_sources"] = [
                name for name in rec["correlated_names"] if name.startswith("global:")
            ]
        out.append(rec)
    return out


def _dy_uncertainties(common, rows, published):
    """Relative statistical, additive and multiplicative sources for ``rows``.

    The closure's one convention: stored **relative to the published central value**, so
    the fold applies them to the folded truth.  A source that vanishes on every selected
    row is dropped -- a per-point source belonging to another boson's rows of the same
    table would otherwise ride along as a zero column.

    Returns:
        tuple: ``(rel_stat, additive, names, multiplicative)`` with ``additive`` shaped
        ``(rows, sources)`` and ``multiplicative`` shaped ``(sources, rows)``.
    """
    stat = np.sqrt(sum(np.asarray(v, dtype=float)[rows] ** 2
                       for v in common.uncorrelated.values()))
    additive = [np.asarray(v, dtype=float)[rows] / published
                for v in common.correlated.values()]
    additive = [column for column in additive if np.any(column)]
    additive = (np.stack(additive, axis=1) if additive
                else np.zeros((rows.size, 0), dtype=float))
    multiplicative = {name: np.asarray(v, dtype=float)[rows]
                      for name, v in common.multiplicative.items()}
    names = sorted(name for name, v in multiplicative.items() if np.any(v))
    matrix = (np.stack([multiplicative[name] for name in names], axis=0) if names
              else np.zeros((0, rows.size), dtype=float))
    return np.asarray(stat / published, dtype=float), additive, names, matrix


def _attach_multiplicative(rec, names, matrix):
    """Record the multiplicative sources :func:`fold_dy_record` folds in."""
    if names:
        rec["multiplicative_names"] = list(names)
        rec["_multiplicative_rel"] = matrix
        rec["shared_multiplicative_sources"] = [
            name for name in names if name.startswith("global:")
        ]
    return rec


def _fixed_target_dy_record(entry, spec, common):
    """One pointwise fixed-target Drell-Yan record (E605 p+Cu) from NNPDF commondata.

    The same observable as the fitpack E866 tables the suite already carries -- one
    ``(S, Q2, Y)`` per row -- so it feeds ``build_drell_yan`` unchanged and the only new
    thing is the copper parton map (``config.nuclear_parton_basis``).  It lives in the
    ``dy`` section beside E866.

    The commondata gives ``rapidity``/``M2``/``sqrts``; the closure's DY cuts are
    ``Q2 >= DY_Q2_MIN`` and physical ``x1, x2 < DY_X_MAX``.  Measured 2026-09-09: all
    119 rows pass (``Q2`` spans 50.5-286.0, ``x1 <= 0.650``, ``x2 <= 0.482``), which is
    why the contract's ``selected_rows`` equals its ``source_rows``.  The equality is
    asserted rather than assumed, so a future cut change surfaces here.
    """
    value = np.asarray(common.value, dtype=float)
    kin = common.kinematics
    y  = np.asarray(kin["rapidity"], dtype=float)
    q2 = np.asarray(kin["M2"], dtype=float)
    s  = np.asarray(kin["sqrts"], dtype=float) ** 2
    tau = q2 / s
    x1, x2 = np.sqrt(tau) * np.exp(y), np.sqrt(tau) * np.exp(-y)
    keep = np.flatnonzero(
        np.isfinite(value) & (q2 >= cfg.DY_Q2_MIN)
        & (x1 < cfg.DY_X_MAX) & (x2 < cfg.DY_X_MAX)
    )
    if keep.size != int(entry["selected_rows"]):
        raise RuntimeError(
            f"{entry['id']}: the closure DY cut retains {keep.size} rows but the "
            f"contract records selected_rows={entry['selected_rows']}"
        )
    rel_stat, additive, names, matrix = _dy_uncertainties(
        common, keep, np.abs(value[keep])
    )
    rec = {
        "kind": spec["kind"],
        "section": "dy",
        "label": spec["label"],
        "name": spec["label"],
        "contract_id": entry["id"],
        "commondata_name": spec["commondata_name"],
        "reaction": spec["reaction"],
        "file": f"dy/{spec['label']}.dat",
        "Q2": q2[keep].tolist(), "S": s[keep].tolist(), "Y": y[keep].tolist(),
        "source_rows": int(value.size),
        "retained_rows": int(keep.size),
        "rel_stat": rel_stat.tolist(),
        "_additive_rel": additive,
    }
    return _attach_multiplicative(rec, names, matrix)


#: Rapidity-like kinematic columns, in the order they are looked for.  The legacy W/Z
#: concatenations label every row ``pseudorapidity`` even where a row is boson rapidity.
_Y_KEYS = ("rapidity", "pseudorapidity")


def _kinematic(kin, keys, what, label):
    for key in keys:
        if key in kin:
            return key
    raise RuntimeError(f"{label}: commondata carries no {what} column (have {sorted(kin)})")


def _mass_key(kin, label):
    """The mass column and whether it already holds a *squared* mass.

    Commondata spells it ``m_Z2``, ``m_W2``, ``m_V2``, ``m_ll2`` or, for two ATLAS mass
    spectra, a plain ``m_ll`` in GeV.
    """
    for key in sorted(kin):
        if key.startswith("m_") and not key.endswith(("_min", "_max")):
            return key, key.endswith("2")
    raise RuntimeError(f"{label}: commondata carries no mass column (have {sorted(kin)})")


def _hepdata_edges(table):
    """``(low, high)`` of the first independent variable of one HEPData table.

    ``table`` is ``(directory, number)`` under ``cfg.NNPDF_COMMONDATA``, read from that
    directory's ``rawdata/``.  Processed kinematics store bin centres, and the legacy W/Z
    concatenations have no rawdata of their own, so a record names the table its split
    sibling's own ``filter.py`` reads.  :func:`_check_centres` then refuses a table whose
    bins do not contain the commondata centres, which is what catches a wrong table.
    """
    import yaml

    directory, number = table
    paths = sorted(
        (cfg.NNPDF_COMMONDATA / directory / "rawdata").glob(f"HEPData-ins*-Table_{number}.yaml")
    )
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one HEPData Table_{number} under {directory}/rawdata, found {len(paths)}"
        )
    values = yaml.safe_load(paths[0].read_text())["independent_variables"][0]["values"]
    return (np.array([float(v["low"]) for v in values]),
            np.array([float(v["high"]) for v in values]))


def _csv_edges(how):
    """``(low, high)`` in GeV from a rawdata csv: ``("csv", directory, file, lo, hi)``."""
    import csv

    _, directory, filename, low_column, high_column = how
    path = cfg.NNPDF_COMMONDATA / directory / "rawdata" / filename
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (np.array([float(row[low_column]) for row in rows]),
            np.array([float(row[high_column]) for row in rows]))


def _rapidity_edges(spec, kin, rows, label):
    """Rapidity (or lepton-eta) bin edges for one record, per ``spec["y_edges"]``.

    ``"commondata"`` (default) reads ``<key>_min``/``<key>_max``; ``("hepdata", tables)``
    concatenates the named HEPData tables in order; ``"fiducial"`` is the whole fiducial
    window ``[-eta_max, +eta_max]`` of a rapidity-integrated mass spectrum, taken from the
    dataset's cut record rather than restated here.
    """
    how = spec.get("y_edges", "commondata")
    if how == "commondata":
        key = _kinematic(kin, _Y_KEYS, "rapidity", label)
        if f"{key}_min" not in kin or f"{key}_max" not in kin:
            raise RuntimeError(
                f"{label}: commondata carries {key} centres only; name the HEPData "
                "tables that hold the edges in the spec's y_edges"
            )
        return (np.asarray(kin[f"{key}_min"], dtype=float)[rows],
                np.asarray(kin[f"{key}_max"], dtype=float)[rows])
    if how == "fiducial":
        from pixel.data.fiducial_cuts import fiducial_cuts_for

        cuts = fiducial_cuts_for(
            spec["nnpdf_legacy_name"], channel=spec.get("fiducial_channel")
        )
        eta = float(cuts.lepton_eta_max)
        return np.full(rows.size, -eta), np.full(rows.size, eta)
    kind, tables = how
    if kind != "hepdata":
        raise RuntimeError(f"{label}: unknown y_edges source {kind!r}")
    lows, highs = zip(*(_hepdata_edges(table) for table in tables))
    return np.concatenate(lows), np.concatenate(highs)


def _mass_edges(spec, kin, rows, label):
    """Squared-mass bin edges per ``spec["mass_edges"]``, or ``(None, None)``.

    ``"commondata"`` reads ``<mass>_min``/``<mass>_max``; ``("csv", ...)`` a rawdata csv
    in GeV; ``("windows", ((first, stop, low_gev, high_gev), ...))`` a published mass
    interval per block of the record's rows.  ``"pole"`` (the default) evaluates each row
    at its commondata mass with zero ``Q^2`` width, which is not what any Z/gamma*
    measurement publishes -- each integrates a mass window -- so
    :func:`_collider_dy_record` refuses it for every kind but the gated W arm (owner
    decision D7, 2026-09-11).
    """
    how = spec.get("mass_edges", "pole")
    if how == "pole":
        return None, None
    if how == "commondata":
        key, squared = _mass_key(kin, label)
        low = np.asarray(kin[f"{key}_min"], dtype=float)[rows]
        high = np.asarray(kin[f"{key}_max"], dtype=float)[rows]
        return (low, high) if squared else (low ** 2, high ** 2)
    if how[0] == "csv":
        low, high = _csv_edges(how)
        return low ** 2, high ** 2
    if how[0] == "windows":
        low = np.full(rows.size, np.nan)
        high = np.full(rows.size, np.nan)
        for first, stop, low_gev, high_gev in how[1]:
            low[first:stop] = float(low_gev) ** 2
            high[first:stop] = float(high_gev) ** 2
        if np.any(np.isnan(low)):
            raise RuntimeError(f"{label}: mass windows leave rows {np.flatnonzero(np.isnan(low)).tolist()} uncovered")
        return low, high
    raise RuntimeError(f"{label}: unknown mass_edges source {how[0]!r}")


def _check_centres(label, what, centre, low, high):
    """Refuse edges that do not bracket their own row's centre.

    The one guard against the failure an edge table invites: the right numbers in the
    wrong order, or the W+ table read for the W- rows.  Nothing downstream could notice.
    """
    if low is None:
        return
    if low.shape != centre.shape or high.shape != centre.shape:
        raise RuntimeError(
            f"{label}: {low.size} {what} edges for {centre.size} rows"
        )
    tol = 1e-9 * (1.0 + np.abs(centre))
    bad = np.flatnonzero((low > centre + tol) | (high < centre - tol) | (high < low))
    if bad.size:
        raise RuntimeError(
            f"{label}: {what} centre outside its own bin on rows {bad.tolist()}; the edge "
            "source does not belong to these rows"
        )


def _collider_dy_record(entry, spec, common):
    """One bin-integrated collider Drell-Yan record for the ``dy`` manifest section.

    Handles every collider arm: a fiducial dilepton rapidity scan, a single-lepton W
    cross section, an unfolded full-phase-space distribution and a rapidity-integrated
    mass spectrum.  The spec in ``config.COLLIDER_DY_SPECS`` states what varies:

    * ``rows`` -- a ``(first, stop)`` slice of the commondata table.  The mixed W/Z tables
      carry one record per boson (and per Z mass slice), because each needs a different
      operator; the slice is fixed by the set's own ``filter.py`` and pinned in the
      contract's gate evidence, never read from metadata labels, one of which is wrong;
    * ``y_edges`` / ``mass_edges`` -- where the bin edges come from (see
      :func:`_rapidity_edges` and :func:`_mass_edges`);
    * ``lepton_charge`` and ``fiducial_channel`` -- carried through to the builder;
    * ``sqrts_gev`` -- only where the commondata omits ``sqrts``.

    **No kinematic cut is applied, and that is deliberate.**  Every row sits at
    ``Q >= 46 GeV``, far above the input scale, so the contract's ``selected_rows``
    equals its ``source_rows`` and both equal the slice; asserting that is the check.

    Uncertainties follow the closure's one convention, relative to the published
    central value (:func:`_dy_uncertainties`).  A split record keeps the sources its rows
    carry and drops their correlation with the other bosons' rows -- recorded in the
    contract's ``covariance_scope``.
    """
    label = entry["id"]
    value_all = np.asarray(common.value, dtype=float)
    kin = common.kinematics
    first, stop = (int(v) for v in spec.get("rows", (0, value_all.size)))
    if stop > value_all.size:
        raise RuntimeError(
            f"{label}: rows {first}-{stop} exceed the commondata's {value_all.size}"
        )
    rows = np.arange(first, stop)
    if not rows.size == int(entry["selected_rows"]) == int(entry["source_rows"]):
        raise RuntimeError(
            f"{label}: collider Drell-Yan applies no kinematic cut, so its slice "
            f"(commondata rows {first}-{stop}, {rows.size} rows) must equal the contract's "
            f"selected_rows={entry['selected_rows']} and source_rows={entry['source_rows']}"
        )
    value = value_all[rows]
    if "sqrts" in kin:
        sqrts = np.asarray(kin["sqrts"], dtype=float)[rows]
    else:
        sqrts = np.full(rows.size, float(spec["sqrts_gev"]))
    mass_key, squared = _mass_key(kin, label)
    mass = np.asarray(kin[mass_key], dtype=float)[rows]
    q2 = mass if squared else mass ** 2
    if spec.get("y_edges") == "fiducial":
        y = np.zeros(rows.size)
    else:
        y = np.asarray(kin[_kinematic(kin, _Y_KEYS, "rapidity", label)], dtype=float)[rows]
    y_low, y_high = _rapidity_edges(spec, kin, rows, label)
    q2_low, q2_high = _mass_edges(spec, kin, rows, label)
    if q2_low is None and spec["kind"] != "drell_yan_fiducial_lepton":
        raise RuntimeError(
            f"{label}: a Z/gamma* record must integrate its published mass window "
            "(spec mass_edges).  At the pole it is d sigma/dQ^2 dY with zero Q^2 width -- "
            "a different observable, in GeV^-4 rather than a bin integral's GeV^-2"
        )
    _check_centres(label, "rapidity", y, y_low, y_high)
    _check_centres(label, "mass^2", q2, q2_low, q2_high)

    rel_stat, additive, names, matrix = _dy_uncertainties(common, rows, np.abs(value))
    rec = {
        "kind": spec["kind"],
        "section": "dy",
        "label": spec["label"],
        "name": spec["label"],
        "contract_id": entry["id"],
        "commondata_name": spec["commondata_name"],
        "nnpdf_legacy_name": spec["nnpdf_legacy_name"],
        "reaction": spec["reaction"],
        "file": f"dy/{spec['label']}.dat",
        "Q2": q2.tolist(), "S": (sqrts ** 2).tolist(), "Y": y.tolist(),
        "Y_low": y_low.tolist(), "Y_high": y_high.tolist(),
        "Q2_low": None if q2_low is None else q2_low.tolist(),
        "Q2_high": None if q2_high is None else q2_high.tolist(),
        "row_range": [first, stop],
        "source_rows": int(rows.size),
        "retained_rows": int(rows.size),
        "rel_stat": rel_stat.tolist(),
        "_additive_rel": additive,
    }
    for extra in ("lepton_charge", "fiducial_channel"):
        if extra in spec:
            rec[extra] = spec[extra]
    return _attach_multiplicative(rec, names, matrix)


def fold_dy_record(rec, dataset, truth, fields, rng, *, noise=True):
    """Fold one Drell-Yan record: ``(central, measured, covariance)``.

    The central value is the fixed truth folded through the record's own operator
    (:func:`dy_central`), shifted by the configured normalization offset where the record
    carries one (the fitpack E866 tables).  The covariance is the statistical diagonal
    plus the outer products of every correlated source, each scaled by ``|central|``:

    * **additive** sources exactly as the DIS arm treats NNPDF's (``nnpdf_layout``);
    * **multiplicative** sources in the covariance too, referenced to the folded truth.
      The DIS arm hands its multiplicative sources to the fit as nuisances with a t0
      iteration; the Drell-Yan t0 iteration is not wired, so here they are fixed at the
      truth reference -- the converged-t0 limit, which is also NNPDF's own choice for
      closure tests.  The fit reads ``mean``/``cov`` back from the manifest, so it sees
      exactly this covariance.

    The statistical floor is :data:`DY_RELATIVE_FLOOR` of the record's largest
    |prediction|, not the DIS arm's absolute ``EXP_ABS_FLOOR`` -- see the constant.

    A record with no correlated source draws from the diagonal, as E866 always has.
    """
    central = dy_central(dataset, truth, fields, rec["reaction"])
    if rec.get("rel_norm") is not None:
        central = (1.0 + rec["norm_beta"] * rec["rel_norm"]) * central
    scale = np.abs(central)
    floor = DY_RELATIVE_FLOOR * float(scale.max()) if scale.size and scale.max() > 0.0 \
        else cfg.EXP_ABS_FLOOR
    sigma = np.abs(np.asarray(rec["rel_stat"], dtype=float)) * scale + floor
    blocks = []
    additive = rec.get("_additive_rel")
    if additive is not None and np.asarray(additive).size:
        blocks.append(np.asarray(additive, dtype=float) * scale[:, None])
    multiplicative = rec.get("_multiplicative_rel")
    if multiplicative is not None and np.asarray(multiplicative).size:
        blocks.append(np.asarray(multiplicative, dtype=float).T * scale[:, None])
    if blocks:
        vectors = np.concatenate(blocks, axis=1)
        cov = np.diag(sigma * sigma) + vectors @ vectors.T
        measured = draw_mvn(central, cov, rng) if noise else central
    else:
        cov = np.diag(sigma * sigma)
        measured = draw_diag(central, sigma, rng) if noise else central
    return central, measured, cov


def _split_sections(records):
    """``(datasets_records, dy_records)`` from one contract walk, by each record's section."""
    linear, bilinear = [], []
    for rec in records:
        section = rec.get("section")
        if section == "datasets":
            linear.append(rec)
        elif section == "dy":
            bilinear.append(rec)
        else:
            raise RuntimeError(
                f"{rec.get('contract_id')}: unknown manifest section {section!r}"
            )
    return linear, bilinear


def _operator_names(rec):
    """Names a builder may give this record's dataset: its manifest name or file stem."""
    return {name for name in (rec.get("name"), Path(rec["file"]).stem) if name}


def _pair_operators(records, named_operators):
    """The operators in record order, refusing anything but a one-to-one match by NAME.

    ``named_operators`` is ``[(dataset name, operator), ...]`` as ``build_datasets``
    emitted it.  The check used to be ``len(operators) == len(records)``, which a swap of
    equal-length lists passes; that is how the 2026-09-09 layout paired CHORUS with
    E605's operator.  Each pair is now checked by identity.
    """
    if len(named_operators) != len(records):
        raise RuntimeError(
            f"{len(named_operators)} operators for {len(records)} manifest records; a "
            "mismatch would fold each record through another dataset's operator"
        )
    out = []
    for index, (rec, (name, operator)) in enumerate(zip(records, named_operators)):
        expected = _operator_names(rec)
        if name not in expected:
            raise RuntimeError(
                f"operator {index} was built for {name!r} but record {index} is "
                f"{sorted(expected)}; the operator list and the record list are out of "
                "order, and folding would silently use the wrong operator"
            )
        out.append(operator)
    return out


def _leading_lattice_count(records):
    """How many records are lattice; they must all come first."""
    count = 0
    while count < len(records) and records[count]["kind"] in LATTICE_KINDS:
        count += 1
    if any(rec["kind"] in LATTICE_KINDS for rec in records[count:]):
        raise RuntimeError("lattice records must precede every experimental record")
    return count


# -- one member --------------------------------------------------------------


def generate_member(q_key: str, *, noise: bool = True, seed: int = 20260710,
                    fields=None, operators=None):
    """Generate and save all closure data for one original-scale truth member.

    Returns ``(truth_dir, fields, operators)`` so a driver can thread the shared
    fields/operators through every member (they are Q-independent).  ``operators`` is a
    list of ``(dataset name, operator)`` pairs; :func:`_pair_operators` checks each name
    against its record before anything is folded.
    """
    truth_dir = cfg.truth_dir(q_key)
    (truth_dir / "lattice").mkdir(parents=True, exist_ok=True)
    (truth_dir / "exp").mkdir(parents=True, exist_ok=True)
    (truth_dir / "datasets").mkdir(parents=True, exist_ok=True)

    if fields is None:
        fields = cfg.make_fields()
    nodes = np.asarray(list(fields.values())[0].nodes).reshape(-1)
    basis = list(fields.values())[0].basis

    # (1) One representable JAM truth.  The replica ensemble is retained only
    # as an external uncertainty source for the fixed lattice covariance.
    ens, covariance_meta = ensemble_curves(q_key, nodes)
    mean_curves, meta = pdf_guidance.jam_truth_curves(
        nodes, q_key=q_key, member=cfg.FIXED_TRUTH_MEMBER
    )
    mean_curves = {
        field: np.asarray(mean_curves[field], dtype=float).reshape(-1)
        for field in cfg.ALL_FIELDS
    }
    std_curves = {f: ens[f].std(axis=0, ddof=1) for f in cfg.ALL_FIELDS}
    meta = dict(meta)
    meta["truth_kind"] = "fixed_lhapdf_member"
    meta["member"] = int(cfg.FIXED_TRUTH_MEMBER)
    meta["covariance_source"] = "replica_ensemble"
    meta["covariance_members"] = covariance_meta.get("members")
    meta["covariance_n_members"] = covariance_meta.get("n_members")

    # Injected systematic curves (nulled n=1 moment); the <=10% cap is applied
    # below once the folded physical signal is known.  Same across Q (fixed seed).
    sys_rng = np.random.default_rng(cfg.SYSTEMATIC_SEED)
    sys_curves = build_systematic_curves(nodes, basis, sys_rng)
    moment_sys_rng = np.random.default_rng(cfg.MOMENT_SYSTEMATIC_SEED)
    moment_sys_values = build_moment_systematic_values(moment_sys_rng)

    ref_fits = pdf_guidance.reference_fits(nodes, mean_curves, seed=1234)
    truth = {
        "meta": meta,
        "x_nodes": nodes.tolist(),
        "curves": {f: mean_curves[f].tolist() for f in cfg.ALL_FIELDS},
        "curve_std": {f: std_curves[f].tolist() for f in cfg.ALL_FIELDS},
        "reference_fits": ref_fits,
    }

    # Order is load-bearing: ``datasets.build_datasets`` emits lattice, then exp, then
    # the generic ``datasets`` section, and ``_pair_operators`` checks that list against
    # ``records`` by name before anything is folded.  The Drell-Yan records from the same
    # contract walk go to the ``dy`` section and their own fold arm below.
    nnpdf_linear, nnpdf_dy = _split_sections(nnpdf_layout())
    records = lattice_layout() + exp_layout() + nnpdf_linear

    # (2) placeholder files so the PIXEL builders can read kinematics.
    for rec in records:
        n = len(rec["_leading"][0])
        write_dat(truth_dir / rec["file"], rec["_leading"], np.ones(n), np.eye(n))

    # (3) manifest (build kwargs live here; datasets.py reads it).
    manifest = {
        "q_key": q_key, "meta": meta,
        "provenance": contract_artifact_provenance(),
        "closure_extension": closure_extension_roster(cfg.CONTRACT_MODE),
        "ensembles": [e.id for e in cfg.active_ensembles()],
        "lattice": [_public(r) for r in records if r["kind"] in LATTICE_KINDS],
        "exp": [
            _public(r) for r in records
            if r["kind"] in EXP_KINDS and r.get("section", "exp") == "exp"
        ],
        "datasets": [_public(r) for r in nnpdf_linear],
    }
    # The guard is a *coverage* check over BOTH sections the walk feeds.  It raises when a
    # selected id has no record -- a contract promotion that generation quietly ignores --
    # and when a record has no selection behind it.  Sets, not counts, so a swap (one id
    # gained, another lost) cannot cancel out.
    selected_ids = {row["id"] for row in manifest["closure_extension"]["selected"]}
    materialized_ids = {r["contract_id"] for r in nnpdf_linear + nnpdf_dy}
    missing = selected_ids - materialized_ids
    if missing:
        raise RuntimeError(
            "observable contracts select records this generator has not materialized: "
            f"{sorted(missing)}; add a cfg.NNPDF_DATASET_SPECS entry (and a builder arm) "
            "or demote the contract -- refusing silent omission"
        )
    stray = materialized_ids - selected_ids
    if stray:
        raise RuntimeError(
            f"generated records {sorted(stray)} are not contract-selected; a dataset must "
            "not enter a closure fit without a contract authorizing it"
        )
    manifest = annotate_manifest(manifest)
    (truth_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # (4) assemble the forward operators once (physical only, no systematics),
    # then fold every replica.  Operators are Q-independent -> reused across Q.
    if operators is None:
        built = dsets.build_datasets(q_key, fields, use_kernel_cache=True,
                                     include_nnpdf=True,
                                     with_systematics=False,
                                     with_exp_nuisances=False)
        operators = [(ds.name, assemble_operator(ds, fields)) for ds in built]
    paired = _pair_operators(records, operators)
    q_offset = list(cfg.TRUTH_Q_CHOICES).index(q_key)
    rng = np.random.default_rng(seed + q_offset)

    # -- lattice pass 1: physical central/cov + raw systematic shift per record.
    n_lattice = _leading_lattice_count(records)
    lat_recs, exp_recs = records[:n_lattice], records[n_lattice:]
    unexpected = sorted({rec["kind"] for rec in exp_recs} - set(EXP_KINDS))
    if unexpected:
        raise RuntimeError(
            f"experimental records of kind {unexpected} have no DIS fold; a Drell-Yan "
            "record belongs in the dy section"
        )
    lat_ops, exp_ops = paired[:n_lattice], paired[n_lattice:]

    folds = fold_lattice_systematics(
        lat_recs,
        lat_ops,
        ens,
        sys_curves,
        moment_sys_values,
        truth_curves=mean_curves,
    )
    sys_scale = folds.scale
    sys_final = folds.curves
    truth["systematic_curves"] = {
        f: {k: sys_final[f][k].tolist() for k in cfg.ITD_SYSTEMATICS}
        for f in cfg.ALL_FIELDS
    }
    truth["systematic_scale"] = {f"{f}:{k}": float(v) for (f, k), v in sys_scale.items()}
    truth["moment_systematic_values"] = {
        name: float(value) for name, value in folds.moment_values.items()
    }
    truth["moment_systematic_scale"] = {
        name: float(value) for name, value in folds.moment_scale.items()
    }
    (truth_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    # -- lattice pass 2: central + scaled shift, inflate diagonal, correlated draw.
    for index, (rec, operator) in enumerate(zip(lat_recs, lat_ops)):
        field = operator[0][0]
        central = folds.central[index].copy()
        for key, shift in folds.raw_shift[index].items():
            if rec["kind"] == "mellin":
                nname = cfg.moment_nuisance_field_name(
                    field, key, int(rec["order"])
                )
                factor = folds.moment_scale.get(nname, 1.0)
            else:
                factor = sys_scale.get((field, key), 1.0)
            central = central + factor * shift
        cov = inflate_diagonal(folds.covariance[index], cfg.COV_DIAGONAL_INFLATE)
        measured = draw_mvn(central, cov, rng) if noise else central
        write_dat(truth_dir / rec["file"], rec["_leading"], measured, cov)

    # -- DIS: fixed-truth central + real rel_stat diagonal error, jittered, plus the
    # real experimental nuisances (overall normalization + correlated
    # systematics) injected as a *drawn* realization.  Drawing them -- rather
    # than only inflating the covariance -- is what keeps the closure honest:
    # the fit marginalizes exactly these directions, so the data must actually
    # scatter along them or the enlarged covariance would be unjustified and
    # coverage would come out artificially conservative.
    (truth_dir / "sys").mkdir(parents=True, exist_ok=True)
    sys_rng_exp = np.random.default_rng(cfg.EXP_SYSTEMATIC_SEED)
    exp_nuisance_truth = {}
    for rec, operator in zip(exp_recs, exp_ops):
        central = fold_truth(operator, mean_curves)
        scale = np.abs(central)
        shift = np.zeros_like(central)
        drawn = {}

        if rec.get("rel_norm") is not None:
            # A fixed, configured offset (not drawn) so the recovered pull can be
            # compared against a known truth, exactly as for Drell-Yan.
            beta = float(rec["norm_beta"])
            shift = shift + beta * rec["rel_norm"] * scale
            drawn["normalization"] = beta

        if "_correlated_rel" in rec:
            # Absolute per-source vectors on the *fake* central; saved so the fit
            # folds the identical directions into its covariance.
            vectors = np.asarray(rec["_correlated_rel"], dtype=float) * scale
            betas = (sys_rng_exp.standard_normal(vectors.shape[0]) if noise
                     else np.zeros(vectors.shape[0]))
            shift = shift + betas @ vectors
            # ``relative`` is what the fit re-references to its own t0 (the
            # NNPDF prescription); ``vectors`` records the absolute shift
            # actually injected and ``betas`` the drawn truth amplitudes.
            np.savez_compressed(truth_dir / "sys" / rec["correlated_file"],
                                relative=np.asarray(rec["_correlated_rel"], dtype=float),
                                vectors=vectors,
                                names=np.array(rec["correlated_names"]),
                                betas=betas)
            drawn["correlated"] = betas.tolist()
        if drawn:
            exp_nuisance_truth[rec["label"]] = drawn

        central = central + shift
        rel = np.asarray(rec["rel_stat"], dtype=float)
        sigma = np.abs(rel) * np.abs(central) + cfg.EXP_ABS_FLOOR
        # The .dat covariance is statistical only for a fitpack table; the fit's
        # builder re-adds the nuisance outer products from rel_norm + the saved sys
        # vectors.  An NNPDF record additionally carries *additive* correlated
        # sources, which no t0 iteration rescales and which therefore belong here
        # rather than in the multiplicative nuisance layer -- see ``nnpdf_layout``.
        additive = rec.get("_additive_rel")
        if additive is not None and np.asarray(additive).size:
            vectors = np.asarray(additive, dtype=float) * np.abs(central)[:, None]
            cov = np.diag(sigma * sigma) + vectors @ vectors.T
            measured = draw_mvn(central, cov, rng) if noise else central
        else:
            cov = np.diag(sigma * sigma)
            measured = draw_diag(central, sigma, rng) if noise else central
        write_dat(truth_dir / rec["file"], rec["_leading"], measured, cov)
    # truth.json was written above (before the lattice draw); rewrite it now that
    # the injected DIS nuisance realization is known, so the recovery report can
    # compare against it.
    truth["exp_nuisances"] = exp_nuisance_truth
    (truth_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    # -- Drell-Yan (bilinear): fold the same fixed truth through each record's own
    # operator.  The fitpack E866 tables and every NNPDF Drell-Yan record share this arm
    # and the ``dy`` section; their mean/cov travel in the manifest (no linear .dat), so
    # the fit's builders read them back directly.
    dy_recs = dy_layout() + nnpdf_dy
    if dy_recs:
        (truth_dir / "dy").mkdir(parents=True, exist_ok=True)
        dy_manifest = []
        for rec in dy_recs:
            ds = dsets.dy_builder(rec["kind"])(rec, None, fields, None,
                                               with_systematics=False)
            _, measured, cov = fold_dy_record(rec, ds, mean_curves, fields, rng,
                                              noise=noise)
            # record-keeping .dat: Q2, S, Y, mean, full cov
            write_dat(truth_dir / rec["file"],
                      (np.asarray(rec["Q2"]), np.asarray(rec["S"]), np.asarray(rec["Y"])),
                      measured, cov)
            out = _public(rec)
            out["mean"] = measured.tolist()
            out["cov"] = cov.tolist()
            dy_manifest.append(out)
        manifest["dy"] = dy_manifest
        manifest = annotate_manifest(manifest)
        (truth_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return truth_dir, fields, operators


def _public(rec: dict) -> dict:
    """Manifest-facing view of a record (drop private ``_`` layout keys)."""
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# -- driver ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate JAM closure data.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--Q", dest="q", choices=list(cfg.TRUTH_Q_CHOICES),
                   help="single original-Q member")
    g.add_argument("--all", action="store_true", help="every Q in config")
    g.add_argument("--random", action="store_true", help="one random Q (seeded)")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=20260710)
    args = ap.parse_args()

    if args.all:
        keys = list(cfg.TRUTH_Q_CHOICES)
    elif args.random:
        rng = np.random.default_rng(args.seed)
        keys = [rng.choice(list(cfg.TRUTH_Q_CHOICES))]
    else:
        keys = [args.q or cfg.DEFAULT_TRUTH_Q]

    fields, operators = None, None
    for q_key in keys:
        n_ens = len(cfg.active_ensembles())
        print(f"== generating truth member Q={cfg.TRUTH_Q_CHOICES[q_key]} "
              f"({cfg.truth_label(q_key)}), input Q0=mc, {n_ens} ensembles, "
              f"{len(cfg.JAM_REPLICA_MEMBERS)} JAM replicas ==")
        watch = Stopwatch()
        out, fields, operators = generate_member(
            q_key, noise=not args.no_noise, seed=args.seed,
            fields=fields, operators=operators)
        print(f"   saved -> {out}  ({format_duration(watch.elapsed)})")


if __name__ == "__main__":
    main()
