"""Fold the JAM replica ensemble through the PIXEL forward operators and save data.

For one original scale ``Q`` (``q_key``):

1. evaluate every JAM replica (members ``1..195``) on the shared field nodes
   (assigned at ``Q0 = mc``) and project each into the nine basis fields;
2. lay out the lattice pseudo-ITD / Mellin-moment kinematics and the real DIS
   kinematics (from ``dis_manifest.json``), writing placeholder ``.dat`` files;
3. build the PIXEL datasets with :mod:`closure_JAM_truth_small.datasets` -- the *same* builders
   the fit uses -- and fold **every replica** through them
   (``y_m = sum_contributions kernel.matrix(nu, basis) @ truth_field_m``);
4. take the **central value** (mean over replicas) and **covariance** (sample
   covariance over replicas) of the transformed output ``y``.  Both lattice and
   experimental data keep only the **diagonal** of that covariance
   (``sigma_i = sqrt(C_ii)``) and add Gaussian jiggle to the central values --
   the fake data are uncorrelated.
5. write ``truth.json`` (ensemble-mean curves + per-node std + reference fit) and
   ``manifest.json``.

Run standalone::

    python -m closure_JAM_truth_small.generate --Q 2          # one member
    python -m closure_JAM_truth_small.generate --all          # every Q in config
    python -m closure_JAM_truth_small.generate --random        # a random Q, seeded
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pixel.util.progress import Stopwatch, format_duration

from . import config as cfg
from . import datasets as dsets
from . import pdf_guidance

#: DIS observable kinds (diagonal covariance + jiggle); everything else is
#: lattice (full SVD-floored covariance, no jiggle).
EXP_KINDS = ("f2", "sigma_r", "sigma_r_cc")


# -- forward operator (shared with the fit by construction) ------------------


def forward(ds, fields, truth_nodes) -> np.ndarray:
    """``sum_contributions kernel.matrix(ds.nu, field.basis) @ truth_field``."""
    mean = np.zeros(np.asarray(ds.mean).shape, dtype=float)
    for c in ds.contributions:
        B = np.asarray(c.kernel.matrix(ds.nu, fields[c.field].basis), dtype=float)
        mean = mean + B @ truth_nodes[c.field]
    return mean.reshape(-1)


def assemble_operator(ds, fields) -> list[tuple[str, np.ndarray]]:
    """Assemble a dataset's ``(field, B)`` kernel matrices **once**.

    ``kernel.matrix`` is the expensive call (it assembles/loads and *verifies* the
    cached NLO matrices), so it must be invoked exactly once per kernel.  The
    returned dense arrays are reused for every truth member and every replica --
    the forward operator is identical across members (all at ``Q0 = mc``).
    """
    return [
        (c.field, np.asarray(c.kernel.matrix(ds.nu, fields[c.field].basis), dtype=float))
        for c in ds.contributions
    ]


def fold_ensemble(operator, ens) -> np.ndarray:
    """Fold the replica ensemble through a pre-assembled ``operator`` -> ``(n_rep, n_pts)``.

    ``operator`` is the ``[(field, B), ...]`` list from :func:`assemble_operator`.
    Linearity lets us apply each ``B`` to the stacked replica curves at once
    (``ens[field]`` is ``(n_rep, n_nodes)``); no kernel is re-assembled here.
    """
    field0, B0 = operator[0]
    Y = ens[field0] @ B0.T                  # (n_rep, n_nodes) @ (n_nodes, n_pts)
    for field, B in operator[1:]:
        Y = Y + ens[field] @ B.T
    return Y


# -- covariance helpers ------------------------------------------------------


def diag_cov(sigma) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    return np.diag(sigma * sigma)


def draw(mean, sigma, rng) -> np.ndarray:
    """Fake observation from ``Normal(mean, sigma)`` (diagonal jiggle)."""
    return np.asarray(mean) + np.asarray(sigma) * rng.standard_normal(len(mean))


# -- JAM replica ensemble ----------------------------------------------------


def _ensemble_cache_path(q_key: str) -> Path:
    return cfg.REFERENCE_DIR / f"jam_ensemble_{cfg.truth_label(q_key)}.npz"


def ensemble_curves(q_key: str, nodes, *, members=cfg.JAM_REPLICA_MEMBERS,
                    use_cache: bool = True):
    """Return ``(curves, meta)`` for the JAM replica ensemble at scale ``q_key``.

    ``curves`` maps each field to a ``(n_replicas, n_nodes)`` array of the
    projected truth values (one row per replica).  Cached to
    ``reference_pdfs/jam_ensemble_<label>.npz`` so re-generation does not re-run
    the C++ dumper once per replica.
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
    meta["truth_kind"] = "replica_ensemble_mean"

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
    """Placeholder lattice records + their leading columns, no truth yet."""
    z = np.full(len(cfg.P_VALUES), cfg.Z_LAT, dtype=float)
    p = np.asarray(cfg.P_VALUES, dtype=float)
    itd, mom = [], []
    for field in cfg.ALL_FIELDS:
        even = cfg.field_cparity(field) == "even"
        comp = "imag" if even else "real"
        itd.append({
            "kind": "pseudoitd", "field": field, "component": comp,
            "momentum_density": cfg.itd_momentum_density(field),
            "file": f"lattice/itd_{field}_{comp}.dat",
            "_leading": (z, p), "_p": p,
        })
        order = cfg.CP_EVEN_MELLIN_ORDER if even else cfg.CP_ODD_MELLIN_ORDER
        mom.append({
            "kind": "mellin", "field": field,
            "order": int(order), "file": f"lattice/moments_{field}.dat",
            "_leading": (np.array([float(order)]),),
        })
    return itd + mom



def dy_layout():
    """Drell-Yan records from the audited E866 tables (used only)."""
    manifest = json.loads(cfg.DIS_MANIFEST_PATH.read_text())
    if not manifest.get("dy", {}).get("has_norm_audit"):
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


def synthetic_z_layout():
    """One unevolved gamma/Z row for plumbing, never physics validation."""
    mz = 91.1876
    return [{
        "kind": "drell_yan",
        "reaction": "pp",
        "label": "synthetic_z_plumbing",
        "name": "synthetic_z_plumbing",
        "file": "synthetic_z/synthetic_z_plumbing.dat",
        "S": [7000.0**2],
        "Q2": [mz**2],
        "Y": [0.0],
        "rel_stat": [0.03],
        "boson": "gamma_z",
        "classification": "synthetic_proxy",
        "observable_contract": "inclusive_boson_level_pointwise_unevolved",
        "physics_coverage": False,
    }]


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
    """Exact DY cross section of the **one** truth on record -> ``(n_pts,)``.

    Folds the fully assembled operator of every bilinear contribution --
    ``BilinearContribution.assemble`` composes each side's evolution exactly as
    the fit does -- against the basis-field truth curves the contribution names.
    Term for term the same body as ``closure_JAM_truth/generate.py``'s, so the
    small and full suites cannot disagree about what a DY central value is.

    **An ensemble second moment is forbidden here, and used to be what this
    function computed.**  It folded ``C = qA.T @ qB / n_rep``, i.e.
    ``E[qA qB] = E[qA] E[qB] + Cov(qA, qB)``, while ``truth.json`` records the
    ensemble *mean* ``E[q]`` and every linear dataset's central value is
    ``E[A q] = A E[q]`` exactly (folding is linear).  A bilinear observable has
    no such identity, so the DY rows were generated from a central value no
    single PDF reproduces: the covariance term is a bias with no truth curve
    behind it, and coverage for the DY sector was undefined.

    Measured at ``Q = 2`` on the shipped replica ensembles, second moment vs.
    this fold, ``max|ratio - 1|`` over rows: JAM 791 replicas ``2.4e-03``
    (``dy_e866_pp``) and ``2.9e-03`` (``dy_e866_pd``); NNPDF 1000 replicas
    ``2.1e-03`` and ``6.4e-03``.  Small next to the ``7%`` rel_stat, but it is a
    systematic offset of the *truth*, not noise -- it does not average away and
    it is not in the covariance.

    Args:
        dataset: The built ``DrellYan`` dataset carrying the bilinear
            contributions.
        truth: ``field name -> (n_nodes,)`` truth curve.  **Which truth kind
            that is -- ``replica_ensemble_mean`` or ``fixed_lhapdf_member`` --
            is the caller's decision and nothing here depends on it.**  This
            fold takes whatever single curve the caller wrote to ``truth.json``;
            that independence is exactly what the second-moment version did not
            have, since it read a replica stack and could only ever produce an
            ensemble statistic.
        fields: Closure field objects, read only for their shared basis.
        reaction: Unused -- the parton -> field map travels on the
            contributions, and re-deriving it here is exactly how generation and
            the fit drift apart without anything failing.

    Returns:
        np.ndarray: Central value per data row.
    """
    del reaction  # the parton -> field map now travels on the contributions
    from dataclasses import replace

    # Every closure field shares one basis, so any of them supplies it.
    a_basis = next(iter(fields.values())).basis
    # Group on the same key the block-sparse model uses: every field pair sharing
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


def exp_layout(norm_betas=None):
    """(see below)  ``norm_betas`` overrides the injected normalization offsets.

    ``cfg.DIS_NORM_BETA`` is a table of *fixed constants*, so every truth member
    and both truth packages inject the identical offsets.  A replica campaign
    that varies only the statistical seed therefore re-uses one normalization
    realization throughout, and a pathological draw in that direction is
    indistinguishable from a structural bias.  Pass ``{idx: beta}`` to redraw it.
    """
    """Placeholder DIS records from the audited manifest selected for closure.

    **No ``TARGET_MAP`` here, deliberately -- the two full suites need one and
    this suite does not.**  Their ``dis_audit`` reads the target off the fitpack
    spreadsheet, which writes the abbreviations ``p``/``d``; this suite's
    ``dis_audit`` writes ``spec.target`` from :data:`config.EXP_SPECS`, which is
    already canonical.  Measured on the shipped manifests: all 13 ``used`` tables
    in each full suite carry ``p``/``d``, and all 7 here carry
    ``proton``/``deuteron``.  So the label passed through verbatim is the same
    label the full suites' map produces.

    Nothing downstream depends on which of the two forms arrives:
    :func:`pixel.util.flavor.normalize_target` -- reached from every DIS builder
    via ``_dis_common`` -- accepts ``p``/``proton``/``n``/``neutron``/``d``/
    ``deuteron``/``isoscalar``/``pn`` and raises ``ValueError`` on anything else,
    so a wrong label fails loudly rather than defaulting.  ``TARGET_MAP`` is
    manifest readability, not correctness; adding a dead copy here would be a
    second place for the alias table to drift from PIXEL's.
    """
    manifest = json.loads(cfg.DIS_MANIFEST_PATH.read_text())
    out = []
    for tab in manifest["tables"]:
        if tab["status"] != "used":
            continue
        x = np.asarray(tab["x"], dtype=float)
        q2 = np.asarray(tab["Q2"], dtype=float)
        rel_stat = np.asarray(tab["rel_stat"], dtype=float)
        rec = {
            "kind": tab["kind"], "target": tab["target"], "label": tab["label"],
            "idx": tab["idx"], "file": f"exp/{tab['label']}.dat",
            "rel_stat": rel_stat.tolist(), "_leading": (x, q2),
        }
        if tab["kind"] == "sigma_r_cc":
            rec["beam"] = tab["beam"]
        # Real experimental nuisances: the overall normalization (a scalar) and
        # the correlated systematic sources (a relative matrix in an .npz
        # sidecar).  Both are injected at generation and marginalized by the fit.
        if tab.get("rel_norm") is not None:
            rec["rel_norm"] = float(tab["rel_norm"])
            rec["norm_beta"] = (
                cfg.dis_norm_beta(tab["idx"]) if norm_betas is None
                else float(norm_betas.get(int(tab["idx"]), 0.0))
            )
        if tab.get("correlated_file"):
            rec["correlated_file"] = tab["correlated_file"]
            rec["correlated_names"] = list(tab["correlated_names"])
            rec["_correlated_rel"] = np.asarray(
                np.load(cfg.DIS_SYS_DIR / tab["correlated_file"])["relative"],
                dtype=float,
            )

        if tab["kind"] in ("sigma_r", "sigma_r_cc"):
            if "Y" in tab:
                rec["y"] = list(tab["Y"])
            else:
                rs = np.asarray(tab.get("RS", []), dtype=float)
                if rs.size == x.size and np.all(rs > 0):
                    rec["y"] = (q2 / (x * rs * rs)).tolist()
        out.append(rec)
    return out


# -- one member --------------------------------------------------------------


def generate_member(q_key: str, *, noise: bool = True, seed: int = 20260710,
                    fields=None, operators=None, sys_seed=None,
                    norm_betas=None):
    """Generate and save all closure data for one original-scale truth member.

    The truth field is the JAM replica-ensemble **mean**; the fake-data
    covariance is the ensemble covariance of the forward-folded output.

    The forward operators depend only on the (Q-independent) dataset kinematics
    and ``Q0 = mc``, never on the truth's original scale ``Q`` -- every member is
    treated as living at ``Q0``.  So ``fields`` and the assembled ``operators``
    (dense ``(field, B)`` matrices, one list per dataset) can be built once and
    reused across members: pass them in to skip the (only) expensive step --
    ``kernel.matrix`` assembly/verification.  Returns ``(truth_dir, fields,
    operators)`` so a driver can thread the shared operators through every member.
    """
    truth_dir = cfg.truth_dir(q_key)
    (truth_dir / "lattice").mkdir(parents=True, exist_ok=True)
    (truth_dir / "exp").mkdir(parents=True, exist_ok=True)

    if fields is None:
        fields = cfg.make_fields()
    nodes = np.asarray(list(fields.values())[0].nodes).reshape(-1)

    # (1) JAM replica ensemble -> mean truth curves + per-node spread.
    ens, meta = ensemble_curves(q_key, nodes)
    mean_curves = {f: ens[f].mean(axis=0) for f in cfg.ALL_FIELDS}
    std_curves = {f: ens[f].std(axis=0, ddof=1) for f in cfg.ALL_FIELDS}

    # Analytic two-term reference fit on the mean curve (metadata only).
    ref_fits = pdf_guidance.reference_fits(nodes, mean_curves, seed=1234)
    truth = {
        "meta": meta,
        "x_nodes": nodes.tolist(),
        "curves": {f: mean_curves[f].tolist() for f in cfg.ALL_FIELDS},
        "curve_std": {f: std_curves[f].tolist() for f in cfg.ALL_FIELDS},
        "reference_fits": ref_fits,
    }
    # Write the truth record up front (before the slow DIS fold) so downstream
    # tools can read it as soon as generation starts.
    (truth_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    records = lattice_layout() + exp_layout(norm_betas=norm_betas)

    # (2) placeholder files so the PIXEL builders can read kinematics.
    for rec in records:
        n = len(rec["_leading"][0])
        write_dat(truth_dir / rec["file"], rec["_leading"],
                  np.ones(n), np.eye(n))

    # (3) manifest (build kwargs live here; datasets.py reads it).
    manifest = {
        "q_key": q_key, "meta": meta,
        "lattice": [_public(r) for r in records if r["kind"] in ("pseudoitd", "mellin")],
        "exp": [_public(r) for r in records if r["kind"] in EXP_KINDS],
    }
    (truth_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # (4) assemble the forward operators once, then fold every replica.  The
    # kernels are Q-independent, so pre-assembled ``operators`` (dense matrices)
    # are reused as-is; only the per-member truth curves change.  Re-calling
    # ``kernel.matrix`` here would re-run the (slow) cache verification per member.
    if operators is None:
        built = dsets.build_datasets(q_key, fields, use_kernel_cache=True,
                                     with_exp_nuisances=False)
        operators = [assemble_operator(ds, fields) for ds in built]
    q_offset = list(cfg.TRUTH_Q_CHOICES).index(q_key)
    rng = np.random.default_rng(seed + q_offset)

    # Real DIS experimental nuisances (overall normalization + correlated
    # systematics) are injected as a *drawn* realization on the experimental
    # records only.  Drawing them -- rather than only inflating the covariance --
    # is what keeps the closure honest: the fit marginalizes exactly these
    # directions, so the data must actually scatter along them.
    (truth_dir / "sys").mkdir(parents=True, exist_ok=True)
    # cfg.EXP_SYSTEMATIC_SEED is one fixed constant shared by every member and
    # both packages, so the correlated-systematic realization is identical
    # everywhere unless a replica campaign overrides it here.
    sys_rng_exp = np.random.default_rng(
        cfg.EXP_SYSTEMATIC_SEED if sys_seed is None else int(sys_seed)
    )
    exp_nuisance_truth = {}
    for rec, operator in zip(records, operators):
        # Fold every replica; take the ensemble mean and covariance of y.
        Y = fold_ensemble(operator, ens)                   # (n_rep, n_pts)
        central = Y.mean(axis=0)
        cov = np.atleast_2d(np.cov(Y, rowvar=False))       # (n_pts, n_pts)

        if rec["kind"] in EXP_KINDS:
            scale = np.abs(central)
            shift = np.zeros_like(central)
            drawn = {}
            if rec.get("rel_norm") is not None:
                # A configured offset (not drawn), so the recovered pull can be
                # compared against a known truth, exactly as for Drell-Yan.
                beta = float(rec["norm_beta"])
                shift = shift + beta * rec["rel_norm"] * scale
                drawn["normalization"] = beta
            if "_correlated_rel" in rec:
                # Absolute per-source vectors on the *fake* central; saved so the
                # fit folds the identical directions into its covariance.
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

        # Both lattice and experimental data keep only the diagonal of the
        # replica covariance and are jiggled -- the fake data are uncorrelated.
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        cov = diag_cov(sigma)
        measured = draw(central, sigma, rng) if noise else central

        write_dat(truth_dir / rec["file"], rec["_leading"], measured, cov)

    # truth.json was written above; rewrite it now that the injected DIS nuisance
    # realization is known, so the recovery report can compare against it.
    truth["exp_nuisances"] = exp_nuisance_truth
    (truth_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    # -- Drell-Yan (bilinear): fold the ONE truth on record (the ensemble mean
    # written to truth.json above) through the DY tensors; real rel_stat diagonal
    # error.  Not a replica statistic -- see dy_central.  DY records carry
    # mean/cov in the manifest.
    dy_recs = dy_layout()
    if dy_recs:
        (truth_dir / "dy").mkdir(parents=True, exist_ok=True)
        dy_manifest = []
        for rec in dy_recs:
            ds = dsets.build_drell_yan(rec, None, fields, None)
            central = dy_central(ds, mean_curves, fields, rec["reaction"])
            # Inject the known normalization offset: every row of a table is
            # scaled by the same 1 + beta_true * rel_norm, exactly the shift the
            # fit's normalization nuisance is allowed to absorb.
            if rec.get("rel_norm") is not None:
                central = (1.0 + rec["norm_beta"] * rec["rel_norm"]) * central
            rel = np.asarray(rec["rel_stat"], dtype=float)
            sigma = np.abs(rel) * np.abs(central) + cfg.EXP_ABS_FLOOR
            cov = np.diag(sigma * sigma)
            measured = draw(central, sigma, rng) if noise else central
            write_dat(truth_dir / rec["file"],
                      (np.asarray(rec["Q2"]), np.asarray(rec["S"]), np.asarray(rec["Y"])),
                      measured, cov)
            out = _public(rec)
            out["mean"] = measured.tolist()
            out["cov"] = cov.tolist()
            dy_manifest.append(out)
        manifest["dy"] = dy_manifest
        (truth_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # The small suites intentionally do not evolve DY.  This single synthetic
    # gamma/Z point exists only to catch EW builder/manifest plumbing failures.
    # Keeping it in its own mode prevents it from masquerading as retained LHC
    # physics data or acceptance validation.
    synthetic_manifest = []
    (truth_dir / "synthetic_z").mkdir(parents=True, exist_ok=True)
    for rec in synthetic_z_layout():
        ds = dsets.build_drell_yan(rec, None, fields, None)
        central = dy_central(ds, mean_curves, fields, rec["reaction"])
        sigma = np.asarray(rec["rel_stat"], dtype=float) * np.abs(central)
        sigma = sigma + cfg.EXP_ABS_FLOOR
        cov = np.diag(sigma * sigma)
        measured = draw(central, sigma, rng) if noise else central
        write_dat(
            truth_dir / rec["file"],
            (np.asarray(rec["Q2"]), np.asarray(rec["S"]), np.asarray(rec["Y"])),
            measured,
            cov,
        )
        out = _public(rec)
        out["mean"] = measured.tolist()
        out["cov"] = cov.tolist()
        synthetic_manifest.append(out)
    manifest["synthetic_z"] = synthetic_manifest
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

    # The forward operators are identical for every member (all treated at
    # Q0 = mc), so assemble the fields + kernel matrices once and reuse them:
    # only the first member pays the kernel-assembly/verify cost, the rest just
    # fold new curves through the cached dense matrices.
    fields, operators = None, None
    for q_key in keys:
        print(f"== generating truth member Q={cfg.TRUTH_Q_CHOICES[q_key]} "
              f"({cfg.truth_label(q_key)}), input Q0=mc, "
              f"{len(cfg.JAM_REPLICA_MEMBERS)} JAM replicas ==")
        watch = Stopwatch()
        out, fields, operators = generate_member(
            q_key, noise=not args.no_noise, seed=args.seed,
            fields=fields, operators=operators)
        print(f"   saved -> {out}  ({format_duration(watch.elapsed)})")


if __name__ == "__main__":
    main()
