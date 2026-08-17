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

import numpy as np

from pixel.util.progress import Stopwatch, format_duration

from pixel import kernels
from pixel.data.contracts import (
    annotate_manifest,
    closure_extension_roster,
    contract_artifact_provenance,
)
from pixel.kernels import LATTICE_SYSTEMATICS

from . import config as cfg
from . import datasets as dsets
from . import pdf_guidance

#: DIS observable kinds (real rel_stat diagonal error + jitter); everything else
#: is lattice (full replica covariance + 1% diagonal inflation + systematics).
EXP_KINDS = ("f2", "sigma_r", "sigma_r_cc")

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
    manifest = json.loads(cfg.DIS_MANIFEST_PATH.read_text())
    dy_manifest = manifest.get("dy", {})
    if (
        not dy_manifest.get("has_norm_audit")
        or dy_manifest.get("max_rows", object()) != cfg.DY_MAX_ROWS
    ):
        from . import dy_audit
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
    for contribution, outer in groups.values():
        tensor = np.asarray(
            contribution.assemble(dataset.nu, a_basis, a_basis), dtype=float
        )
        central += np.einsum("rij,ij->r", tensor, outer, optimize=True)
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


# -- one member --------------------------------------------------------------


def generate_member(q_key: str, *, noise: bool = True, seed: int = 20260710,
                    fields=None, operators=None):
    """Generate and save all closure data for one original-scale truth member.

    Returns ``(truth_dir, fields, operators)`` so a driver can thread the shared
    fields/operators through every member (they are Q-independent).
    """
    truth_dir = cfg.truth_dir(q_key)
    (truth_dir / "lattice").mkdir(parents=True, exist_ok=True)
    (truth_dir / "exp").mkdir(parents=True, exist_ok=True)

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

    records = lattice_layout() + exp_layout()

    # (2) placeholder files so the PIXEL builders can read kinematics.
    for rec in records:
        n = len(rec["_leading"][0])
        write_dat(truth_dir / rec["file"], rec["_leading"], np.ones(n), np.eye(n))

    # (3) manifest (build kwargs live here; datasets.py reads it).
    manifest = {
        "q_key": q_key, "meta": meta,
        "provenance": contract_artifact_provenance(),
        "closure_extension": closure_extension_roster("jam_full"),
        "ensembles": [e.id for e in cfg.active_ensembles()],
        "lattice": [_public(r) for r in records if r["kind"] in ("pseudoitd", "mellin")],
        "exp": [_public(r) for r in records if r["kind"] in EXP_KINDS],
    }
    if manifest["closure_extension"]["selected_source_count"]:
        raise RuntimeError(
            "observable contracts select new EW/NNPDF records, but this generator "
            "has not materialized matching manifest datasets; refusing silent omission"
        )
    manifest = annotate_manifest(manifest)
    (truth_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # (4) assemble the forward operators once (physical only, no systematics),
    # then fold every replica.  Operators are Q-independent -> reused across Q.
    if operators is None:
        built = dsets.build_datasets(q_key, fields, use_kernel_cache=True,
                                     with_systematics=False,
                                     with_exp_nuisances=False)
        operators = [assemble_operator(ds, fields) for ds in built]
    q_offset = list(cfg.TRUTH_Q_CHOICES).index(q_key)
    rng = np.random.default_rng(seed + q_offset)

    # -- lattice pass 1: physical central/cov + raw systematic shift per record.
    lat_recs = [r for r in records if r["kind"] in ("pseudoitd", "mellin")]
    exp_recs = [r for r in records if r["kind"] in EXP_KINDS]
    lat_ops = operators[:len(lat_recs)]
    exp_ops = operators[len(lat_recs):]

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
        # The .dat covariance is statistical only; the fit's builder re-adds the
        # nuisance outer products from rel_norm + the saved sys vectors.
        cov = np.diag(sigma * sigma)
        measured = draw_diag(central, sigma, rng) if noise else central
        write_dat(truth_dir / rec["file"], rec["_leading"], measured, cov)
    # truth.json was written above (before the lattice draw); rewrite it now that
    # the injected DIS nuisance realization is known, so the recovery report can
    # compare against it.
    truth["exp_nuisances"] = exp_nuisance_truth
    (truth_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    # -- Drell-Yan (bilinear): fold the same fixed truth through the DY tensors, real
    # rel_stat diagonal error.  DY records carry their mean/cov in the manifest
    # (no linear .dat), so the fit's builder reads them back directly.
    dy_recs = dy_layout()
    if dy_recs:
        (truth_dir / "dy").mkdir(parents=True, exist_ok=True)
        dy_manifest = []
        for rec in dy_recs:
            ds = dsets.build_drell_yan(rec, None, fields, None, with_systematics=False)
            central = dy_central(ds, mean_curves, fields, rec["reaction"])
            # Inject the known normalization offset: every row of a table is
            # scaled by the same 1 + beta_true * rel_norm, exactly the shift the
            # fit's normalization nuisance is allowed to absorb.
            if rec.get("rel_norm") is not None:
                central = (1.0 + rec["norm_beta"] * rec["rel_norm"]) * central
            rel = np.asarray(rec["rel_stat"], dtype=float)
            sigma = np.abs(rel) * np.abs(central) + cfg.EXP_ABS_FLOOR
            cov = np.diag(sigma * sigma)
            measured = draw_diag(central, sigma, rng) if noise else central
            # record-keeping .dat: Q2, S, Y, mean, diag(cov)
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
