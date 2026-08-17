"""Momentum and valence sum-rule checks against the JAM24 LHAPDF closure input.

Exercises ``pixel.kernels.pqcd.evolution.NonSingletEvolution``/``SingletEvolution``
(via ``pixel.kernels.evolution.evolution_factors``/``singlet_evolution_factors`` in
Mellin space, and ``evolution_matrix``/``singlet_evolution_matrix`` in x space) at
``cfg.ORDER = "NLO"`` (``closure_JAM_truth_small/config.py:190``) only -- this file
never parametrizes over order, so it probes neither LO nor NNLO.  At NLO the
singlet/non-singlet splitting functions are direct closed-form Mellin expressions
(``src/pixel/kernels/pqcd/splitting.py:27-28``), not the Moch-Vermaseren-Vogt
compact parametrization used only at NNLO.  **This file is therefore not, and
cannot be, the NNLO-fit-floor witness** that CLAUDE.md's evolution section
describes (quark ``4.2e-05``/``1.8e-04``/``1.2e-04``, gluon
``5.6e-06``/``3.3e-05``/``2.9e-05`` for nf=3/4/5): that witness is
``test_momentum_sum_rule_measures_the_parametrization_accuracy`` in
``tests/test_nnlo_splitting.py``, cross-referenced directly from
``src/pixel/kernels/pqcd/nnlo_splitting.py:69``, and it probes the bare NNLO
matrix at one Mellin point, not the RGE-evolved operator checked here.  The
order-dependence of the same *kind* of check, with a toy input instead of JAM24,
is in ``tests/test_evolution.py``'s
``test_singlet_pdf_mellin_moments_satisfy_momentum_sum_rule_under_evolution`` and
``test_valence_first_moment_conserved`` (LO exact to ``1e-12``, NLO limited to
``~1e-9`` by polygamma floating-point precision -- the same floor measured below).

Oracles, in increasing order of what they touch:

* The two ``..._at_input_scale`` tests below check only the JAM24 LHAPDF dump
  and ``closure_JAM_truth_small.pdf_guidance.dump_jam``'s reading pipeline -- no
  ``src/pixel`` code runs -- and exist to gate that the realistic input the
  PIXEL tests use is itself sane, not to test evolution.
* The four ``pixel_...``/``xspace_...`` tests check momentum/quark-number
  conservation: an exact QCD identity independent of alpha_s, order, or any
  external fixture, and genuinely independent of the evolution code under test.
  Traced, not assumed, to be non-tautological: ``lo_singlet``/``nlo_singlet``
  (``src/pixel/kernels/pqcd/splitting.py:257-468``) build ``qq``, ``qg``,
  ``gq``, ``gg`` as four independently-coded closed-form Mellin expressions, so
  N=2 column-sum conservation is an emergent cancellation across them, not
  something imposed by construction -- confirmed by perturbing the LO ``pgq``
  term in memory (never touching ``src/`` on disk): a 1e-4 relative error moves
  the evolved-operator defect from its ``~1e-8`` floor to ``1.8e-5``/``1.8e-6``,
  comfortably above ``PIXEL_CONSERVATION_ATOL``.
* The two ``xspace_...`` tests additionally route through
  ``singlet_evolution_matrix``/``non_singlet_evolution_matrix``, which call the
  identical ``SingletEvolution.operator``/``NonSingletEvolution.sigma`` used by
  the Mellin-space tests above (confirmed by reading ``assembly.py``'s
  ``sigma_block_fn``/``sigma_fn``), so a splitting-function bug is common-mode
  between them.  Their own new territory is the finite-element basis,
  adaptive-contour, and low-x-completion pipeline: the comment above
  ``XSPACE_CONTOUR_TOL`` records a 3x node-count refinement that moved the
  Q=10/Q=100 GeV residuals by only ``6.5e-8``/``8.3e-8``, i.e. that bar has
  already survived refinement in the parameter under suspicion
  (``tests/README.md`` rule 1), and its dominant error is the unweighted low-x
  completion below ``x_min ~ 1.2e-6`` (comment above ``XSPACE_GRID_N``), not
  contour truncation.

Accuracy limits, measured directly (single Mellin-point calls,
``OMP_NUM_THREADS=1``): the Mellin-space ``PIXEL_CONSERVATION_ATOL`` check
achieves ``|defect|`` between ``1.0e-10`` and ``6.4e-9`` (N=2 quark column) and
``2.2e-8`` and ``5.5e-8`` (N=2 gluon column) across the file's five target
scales, and ``1.1e-9`` to ``2.9e-9`` (N=1 valence factor) -- set by
floating-point cancellation in the harmonic-sum/polygamma evaluation and the
VFNS threshold-matching chain, not by any perturbative approximation.  See
``PIXEL_CONSERVATION_ATOL``'s comment for the full coefficient-sensitivity scan.
The x-space ``XSPACE_MELLIN_ATOL``/``XSPACE_SUM_RULE_ATOL = 1.2e-2`` bars are
four orders looser, for the low-x-completion reason above, not because the
underlying evolution is any less accurate there.

Known blind spot (FIXED 2026-08-13 in ``test_pixel_singlet_evolution_preserves_jam24_momentum_sum_rule``,
weakness ``test_momentum_sum_rule-01``): the three original conservation
assertions downstream of ``_jam24_input_momentum()`` consume only
``evolved_momentum.sum()`` (and, in the x-space test, ``reconstructed.sum()``);
a swap of which JAM24 LHAPDF column is attributed to ``quark_singlet`` vs
``gluon`` would be undetected by them, since ``operator.sum(axis=0) @ v``
collapses to ``sum(v)``, which is exactly invariant to the order of ``v``'s two
entries -- demonstrated by evaluating the same evolution operator against the
real and the component-swapped JAM24 input: the totals differ by ``5.5e-9``,
far under both ``PIXEL_CONSERVATION_ATOL`` and ``JAM24_NORMALIZATION_ATOL``.
A fourth, order-sensitive assertion now closes this for the Mellin-space
singlet test: ``evolved_momentum[0] > evolved_momentum[1]`` (quark-singlet
exceeds gluon) holds at every target scale with a shrinking-but-never-crossing
margin (``0.204`` at the charm threshold down to ``0.053`` at 100 GeV), and a
component swap flips the inequality at every scale (measured).  The x-space
test (``test_xspace_reconstruction_preserves_momentum_sum_rule``) still shares
this gap by the identical structural argument (only ``.sum()`` is ever
compared there); not re-measured or fixed here -- see the JSON report's `M03`.
The valence tests do not share this gap: their assertions are elementwise over
four flavors against the asymmetric ``(2, 1, 0, 0)`` target, so a component
reorder (e.g. ``u`` <-> ``d``) fails immediately.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache

import numpy as np
import pytest

cfg = pytest.importorskip("closure_JAM_truth_small.config")
pdf_guidance = pytest.importorskip("closure_JAM_truth_small.pdf_guidance")

from pixel.geometry import Grid
from pixel.geometry.finite_elements import CubicSplineBasis
from pixel.kernels.evolution import (
    adaptive_contour,
    evolution_factors,
    evolution_matrix,
    singlet_evolution_factors,
    singlet_evolution_matrix,
)


INPUT_Q_GEV = 2.0
TARGET_Q_GEV = (
    pytest.param(cfg.MC, id="mc"),
    pytest.param(4.0, id="4gev"),
    pytest.param(5.0, id="5gev"),
    pytest.param(10.0, id="10gev"),
    pytest.param(100.0, id="100gev"),
)
# N=2 is the Mellin moment of int dx x f(x): the momentum fraction.  Momentum
# conservation makes each singlet/gluon column of the true anomalous-dimension
# matrix sum to zero here (to 1 after RGE exponentiation) -- see the module
# docstring's tautology-check paragraph for why that is not built in by
# construction.
MOMENTUM_N = 2.0
# N=1 is the Mellin moment of int dx (q - qbar): the valence (baryon/quark)
# number.  Conserved order by order in perturbation theory, so the minus-channel
# evolution factor at N=1 must equal 1 exactly for the true splitting functions.
VALENCE_N = 1.0
# (u, d, s, c) valence quantum numbers of the proton -- exact and independent of
# any fit.  Deliberately asymmetric (2, 1, 0, 0): a downstream component reorder
# (e.g. u <-> d) fails immediately against this target, unlike the momentum
# checks below, which reduce (Sigma, g) to a single sum -- see the module
# docstring's "known blind spot" paragraph.
EXPECTED_VALENCE_COUNTS = np.array([2.0, 1.0, 0.0, 0.0])

# The JAM24 LHAPDF central member integrates to 0.999961 with the local table and
# this quadrature -- a 3.9e-05 deviation (reproduced directly here: 3.8719e-05),
# where JAM20 gave 1.00305 (3.1e-03).  The tolerance tracks the set: it was 4.0e-3
# for JAM20 (~30% headroom), so 1.0e-4 keeps the same discriminating power for
# JAM24 rather than leaving a bound two orders of magnitude looser than the
# quantity it checks.  Tight enough to catch flavor omissions, while the PIXEL
# conservation check below is much stricter.
JAM24_NORMALIZATION_ATOL = 1.0e-4
# Measured against the same JAM24 dump used above: the (u, d, s, c) deviations
# from the exact (2, 1, 0, 0) target are (1.985e-3, 1.321e-3, 5.60e-4, 0.0) --
# 2.5x headroom under this bar, comparably tight to JAM24_NORMALIZATION_ATOL
# rather than a loose afterthought (previously unrecorded here).
JAM24_VALENCE_ATOL = 5.0e-3
# Bounds the N=2 singlet-column-sum and N=1 valence-factor conservation defects
# of PIXEL's own NLO evolution operator -- exact splitting functions, not the
# NNLO MVV fit (module docstring).  Measured directly at the five target scales
# below: 1.0e-10 to 6.4e-9 (N=2 quark column), 2.2e-8 to 5.5e-8 (N=2 gluon
# column), 1.1e-9 to 2.9e-9 (N=1 valence factor) -- set by polygamma/harmonic-sum
# floating-point cancellation and the VFNS threshold-matching chain, not by any
# perturbative approximation.  A relative perturbation of the LO singlet ``pgq``
# term (in-memory monkeypatch only) moves the N=2 defect from that floor to
# 1.8e-5/1.8e-6 at 1e-4 relative and to 1.8e-7/2.0e-8 at 1e-6 relative, so this
# bar reliably catches realistic (>=1e-4 relative) coefficient errors and loses
# sensitivity only below about one part in a million.  It does NOT catch a
# quark_singlet<->gluon component swap in the JAM24 input: both conservation
# assertions below consume only a sum, and a full swap changes that sum by just
# 5.5e-9 (measured) -- see the module docstring's "known blind spot" paragraph.
PIXEL_CONSERVATION_ATOL = 5.0e-7
XSPACE_MELLIN_ATOL = 1.2e-2
XSPACE_SUM_RULE_ATOL = 1.2e-2
# This test resolves a real singlet operator, so it must use the c>1 adaptive
# panel rule rather than the coarse exact-identity exception.  On this machine
# and grid, tol=1e-2 uses 67,968 singlet nodes and takes ~38 s at Q=10 GeV;
# tightening to 3e-3 uses 101,568 nodes and ~47 s while moving the Q=10 GeV
# momentum sum by 6.5e-8 and the largest valence sum by 1.2e-6.  At the harder
# Q=100 GeV endpoint, the 1.0733e-2 momentum discrepancy moves by only 8.3e-8
# under that refinement.  It is therefore the finite-element/JAM24 low-x floor,
# not contour error; the 1.2e-2 assertion keeps ~12% headroom over that measured
# maximum.  Do not pay the 214,464-node production-default cost here when it adds
# no discrimination, and do not lower the node count without repeating this scan.
XSPACE_CONTOUR_TOL = 1.0e-2

# Reconstruction grid for the x-space checks.  Its first positive node is
# determined by the point count (``x_min = 1.07 ** -n_points``), so the count is
# really a choice of how far down in x the sum-rule integrals reach.
#
# 201 points put x_min at ~1.2e-06.  That matters because the JAM24 sea rises
# steeply toward the origin and these integrals are unweighted (``int dx (q-qbar)``
# and ``int dx x f``), so whatever lies below x_min is simply lost -- the
# quadrature has no low-x completion.  At the old 65 points x_min was only 1.2e-02
# and the valence counts came out [1.81, 0.79] against an exact [2, 1]; from ~145
# points on they are correct to a part in 1e-3.
XSPACE_GRID_N = 201
XSPACE_X_CUTOFF = 0.2


def _split_quadrature(*, xmin: float, n_per_segment: int = 160):
    """Gauss quadrature nodes/weights on [xmin, 1] with extra low-x resolution."""
    segments = (
        (xmin, 1.0e-3, "log"),
        (1.0e-3, 1.0e-1, "log"),
        (1.0e-1, 1.0, "linear"),
    )
    xs = []
    ws = []
    nodes, weights = np.polynomial.legendre.leggauss(n_per_segment)
    for left, right, spacing in segments:
        if spacing == "log":
            t_left, t_right = np.log(left), np.log(right)
            t = 0.5 * (t_right - t_left) * nodes + 0.5 * (t_right + t_left)
            x = np.exp(t)
            w = 0.5 * (t_right - t_left) * weights * x
        else:
            x = 0.5 * (right - left) * nodes + 0.5 * (right + left)
            w = 0.5 * (right - left) * weights
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _dump_jam24_or_skip(x: np.ndarray, *, q: float) -> np.ndarray:
    """Evaluate the closure LHAPDF helper, skipping when JAM24 is unavailable."""
    if not cfg.LHAPDF_PREFIX.exists():
        pytest.skip(f"LHAPDF prefix not found: {cfg.LHAPDF_PREFIX}")
    try:
        return pdf_guidance.dump_jam(x, q=q, member=cfg.JAM_MEMBER)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"JAM24 LHAPDF dump is unavailable: {exc}")


@lru_cache(maxsize=1)
def _jam24_input_table():
    """Return ``(x, weights, raw_xfx)`` for JAM24 at Q = 2 GeV."""
    x, weights = _split_quadrature(xmin=cfg.X_MIN)
    raw = _dump_jam24_or_skip(x, q=INPUT_Q_GEV)
    return x, weights, raw


@lru_cache(maxsize=1)
def _jam24_input_momentum() -> np.ndarray:
    """Return ``(<x>_Sigma, <x>_g)`` for JAM24 at Q = 2 GeV."""
    _, weights, raw = _jam24_input_table()

    # Column layout trusted from pdf_guidance.dump_jam and never independently
    # re-verified by this file: raw[:, 1] is gluon, raw[:, 2:10] the 8 quark
    # flavors.  Every downstream conservation check consumes only the *sum* of
    # these two components, so a swap between them here would be undetected --
    # see the module docstring's "known blind spot" paragraph.
    gluon = raw[:, 1]
    quark_singlet = raw[:, 2:10].sum(axis=1)
    return np.array(
        [
            float(weights @ quark_singlet),
            float(weights @ gluon),
        ]
    )


@lru_cache(maxsize=1)
def _jam24_input_valence_counts() -> np.ndarray:
    """Return ``int dx (q - qbar)`` for ``(u, d, s, c)`` at Q = 2 GeV."""
    x, weights, raw = _jam24_input_table()

    return np.array(
        [
            float(weights @ ((raw[:, 2] - raw[:, 3]) / x)),
            float(weights @ ((raw[:, 4] - raw[:, 5]) / x)),
            float(weights @ ((raw[:, 6] - raw[:, 7]) / x)),
            float(weights @ ((raw[:, 8] - raw[:, 9]) / x)),
        ]
    )


@lru_cache(maxsize=1)
def _jam24_xspace_input():
    """Return PDF-valued JAM24 input fields and quadrature weights for x-space."""
    grid = Grid(
        n_points=XSPACE_GRID_N,
        spacing="log-linear",
        x_cutoff=XSPACE_X_CUTOFF,
    )
    x = grid.points
    basis = CubicSplineBasis(x)
    raw = _dump_jam24_or_skip(x, q=INPUT_Q_GEV)

    nodes, weights = basis.quadrature(16)
    basis_at_nodes = basis.evaluate(nodes)
    mass_weights = weights @ basis_at_nodes
    momentum_weights = (weights * nodes) @ basis_at_nodes
    # Real evolution amplifies c>1 panel errors.  Use the production contour
    # factory so the test exercises the same resolved-panel and singlet-zmax-cap
    # policy as closure generation; the explicit tolerance is justified above.
    singlet_contour = adaptive_contour(
        basis,
        tol=XSPACE_CONTOUR_TOL,
        min_c=1.05,
    )
    non_singlet_contour = adaptive_contour(
        basis,
        tol=XSPACE_CONTOUR_TOL,
    )

    return {
        "basis": basis,
        "mass_weights": mass_weights,
        "momentum_weights": momentum_weights,
        "singlet_pdf": raw[:, 2:10].sum(axis=1) / x,
        "gluon_pdf": raw[:, 1] / x,
        "valence_pdfs": np.column_stack(
            [
                (raw[:, 2] - raw[:, 3]) / x,
                (raw[:, 4] - raw[:, 5]) / x,
                (raw[:, 6] - raw[:, 7]) / x,
                (raw[:, 8] - raw[:, 9]) / x,
            ]
        ),
        "singlet_contour": singlet_contour,
        "non_singlet_contour": non_singlet_contour,
    }


@lru_cache(maxsize=None)
def _xspace_reconstructed_sum_rules(target_q_gev: float, order: str = cfg.ORDER):
    """Evolve in x space and re-integrate the sum-rule Mellin moments.

    ``order`` defaults to ``cfg.ORDER`` (NLO), which is what the closure
    pipeline runs; ``"LO"`` is used by
    ``test_xspace_reconstruction_preserves_momentum_sum_rule_at_lo`` to isolate
    the reconstruction error from the perturbative one.
    """
    rec = _jam24_xspace_input()
    basis = rec["basis"]
    mass_weights = rec["mass_weights"]
    momentum_weights = rec["momentum_weights"]
    singlet_pdf = rec["singlet_pdf"]
    gluon_pdf = rec["gluon_pdf"]
    valence_pdfs = rec["valence_pdfs"]
    q2 = float(target_q_gev) ** 2

    initial_momentum = np.array(
        [
            float(momentum_weights @ singlet_pdf),
            float(momentum_weights @ gluon_pdf),
        ]
    )
    singlet_blocks = {
        (target, source): singlet_evolution_matrix(
            basis,
            source=source,
            target=target,
            Q2=q2,
            Q20=INPUT_Q_GEV**2,
            order=order,
            mode=cfg.MODE,
            alphaS_MZ=cfg.ALPHAS_MZ,
            mc=cfg.MC,
            mb=cfg.MB,
            contour=rec["singlet_contour"],
            low_x_extension={"kind": "power", "alpha": 1.0},
        ).matrix
        for target in ("quark_singlet", "gluon")
        for source in ("quark_singlet", "gluon")
    }
    evolved_singlet_pdf = (
        singlet_blocks[("quark_singlet", "quark_singlet")] @ singlet_pdf
        + singlet_blocks[("quark_singlet", "gluon")] @ gluon_pdf
    )
    evolved_gluon_pdf = (
        singlet_blocks[("gluon", "quark_singlet")] @ singlet_pdf
        + singlet_blocks[("gluon", "gluon")] @ gluon_pdf
    )
    reconstructed_momentum = np.array(
        [
            float(momentum_weights @ evolved_singlet_pdf),
            float(momentum_weights @ evolved_gluon_pdf),
        ]
    )
    momentum_operator = singlet_evolution_factors(
        [MOMENTUM_N],
        Q2=q2,
        Q20=INPUT_Q_GEV**2,
        order=order,
        mode=cfg.MODE,
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
    )[:, :, 0]
    expected_momentum = momentum_operator @ initial_momentum

    initial_valence = mass_weights @ valence_pdfs
    valence_matrix, _ = evolution_matrix(
        basis,
        Q2=q2,
        Q20=INPUT_Q_GEV**2,
        order=order,
        channel="minus",
        mode=cfg.MODE,
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
        contour=rec["non_singlet_contour"],
        low_x_extension={"kind": "power", "alpha": 1.0},
    )
    reconstructed_valence = mass_weights @ (valence_matrix @ valence_pdfs)
    valence_factor = evolution_factors(
        [VALENCE_N],
        Q2=q2,
        Q20=INPUT_Q_GEV**2,
        order=order,
        channel="minus",
        mode=cfg.MODE,
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
    )[0]
    expected_valence = float(valence_factor) * initial_valence

    return (
        reconstructed_momentum,
        expected_momentum,
        reconstructed_valence,
        expected_valence,
    )


def test_jam24_momentum_sum_rule_at_input_scale():
    """JAM24 has the expected total parton momentum at the 2 GeV input scale.

    Fixture sanity gate, not a PIXEL test: exercises only the JAM24 LHAPDF dump and
    ``pdf_guidance.dump_jam``'s reading pipeline, no ``src/pixel`` code.  Oracle is
    the exact momentum sum rule (target ``1.0``); bar is
    ``JAM24_NORMALIZATION_ATOL = 1.0e-4``, measured hit at ``3.8719e-05`` (matches
    that constant's comment).  Would catch a wrong LHAPDF column, member, or unit
    in the reading pipeline, or a JAM24 grid whose own fit violates momentum
    conservation beyond quoted precision -- not an evolution bug, since evolution
    never runs here.
    """
    input_momentum = _jam24_input_momentum()

    assert input_momentum.sum() == pytest.approx(1.0, abs=JAM24_NORMALIZATION_ATOL)


@pytest.mark.parametrize("target_q_gev", TARGET_Q_GEV)
def test_pixel_singlet_evolution_preserves_jam24_momentum_sum_rule(target_q_gev):
    """PIXEL preserves the JAM24 N=2 singlet+gluon momentum at requested scales.

    NLO only (``cfg.ORDER``); the N=2 column-sum-to-1 identity is an emergent,
    non-tautological cancellation across the independently-coded ``qq``/``qg``/
    ``gq``/``gg`` entries of ``lo_singlet``/``nlo_singlet`` (module docstring).
    Bar is ``PIXEL_CONSERVATION_ATOL = 5e-7``; that constant's comment records the
    full measured margin and a coefficient-perturbation scan showing this bar
    catches a >=1e-4 relative error and loses sensitivity below ~1e-6 relative.

    FIX (weakness ``test_momentum_sum_rule-01``, S1): the three conservation
    assertions below all consume only a ``.sum()``, so a
    ``quark_singlet<->gluon`` component swap in ``input_momentum`` is invisible
    to them (measured difference ``5.5e-9``, far under ``PIXEL_CONSERVATION_ATOL``)
    -- see the module docstring's "known blind spot" paragraph. A fourth,
    order-sensitive assertion closes this: JAM24's quark-singlet momentum
    fraction exceeds its gluon fraction at every one of this test's five target
    scales (measured ``0.576`` vs ``0.424`` at the input scale, narrowing but
    never crossing, to ``0.526`` vs ``0.473`` at ``Q=100`` GeV -- a margin of
    ``0.053``, ~1e5x ``PIXEL_CONSERVATION_ATOL``). A component swap flips this
    order at every scale (measured: the swapped input evolves to ``[0.418,
    0.582]`` at ``Q=mc``, ``[0.461, 0.539]`` at ``Q=100``), so this assertion is
    a genuine, non-tautological, order-sensitive check, not a restatement of
    conservation.  The valence test below checks its four flavors elementwise
    and does not share this gap.
    """
    input_momentum = _jam24_input_momentum()
    input_total = float(input_momentum.sum())

    operator = singlet_evolution_factors(
        [MOMENTUM_N],
        Q2=float(target_q_gev) ** 2,
        Q20=INPUT_Q_GEV**2,
        order=cfg.ORDER,
        mode=cfg.MODE,
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
    )[:, :, 0]

    evolved_momentum = operator @ input_momentum

    np.testing.assert_allclose(
        operator.sum(axis=0),
        np.ones(2),
        atol=PIXEL_CONSERVATION_ATOL,
    )
    assert evolved_momentum.sum() == pytest.approx(
        input_total,
        abs=PIXEL_CONSERVATION_ATOL,
    )
    assert evolved_momentum.sum() == pytest.approx(
        1.0,
        abs=JAM24_NORMALIZATION_ATOL,
    )
    # Order-sensitive (see docstring): quark_singlet (index 0) exceeds gluon
    # (index 1) at every target scale, margin >=0.053 -- catches a
    # quark_singlet<->gluon column swap that the three sum-only checks above
    # cannot, since operator.sum(axis=0) @ v collapses to sum(v).
    assert evolved_momentum[0] > evolved_momentum[1]


# The LO ``N=2`` singlet anomalous-dimension matrix has a zero eigenvalue whose
# eigenvector is ``(<x>_Sigma, <x>_g) prop (3 nf, 16)``: momentum conservation
# forces the columns to sum to zero, so the surviving direction is fixed by the
# ratio ``P_qg / P_gq`` alone.  Every parton distribution therefore flows to
# ``<x>_g -> 16 / (16 + 3 nf)`` as ``Q -> infinity``, independently of its input.
# Textbook closed form (Altarelli-Parisi), independent of anything in this repo.
# ``nf = 5`` is this configuration's asymptotic flavour count: the only
# thresholds handed to ``singlet_evolution_factors`` below are ``mc`` and ``mb``.
ASYMPTOTIC_NF = 5
ASYMPTOTIC_GLUON_FRACTION = 16.0 / (16.0 + 3.0 * ASYMPTOTIC_NF)
ASYMPTOTIC_SINGLET_FRACTION = 3.0 * ASYMPTOTIC_NF / (16.0 + 3.0 * ASYMPTOTIC_NF)


def test_pixel_singlet_evolution_flows_to_the_asymptotic_momentum_partition():
    """Each momentum fraction separately approaches ``(3 nf, 16)/(16 + 3 nf)``.

    Oracle A2 (an exact asymptotic limit, stated above the constants and
    derivable from the ``N=2`` anomalous-dimension matrix alone), and the answer
    to missing-test item ``-M03``: it constrains ``<x>_Sigma`` and ``<x>_g``
    **individually**, which is what the rest of this file does not.  Every other
    momentum assertion here consumes ``.sum()`` -- the module docstring's "known
    blind spot" -- and the one order-sensitive check that exists
    (``evolved_momentum[0] > evolved_momentum[1]``) constrains a *sign*, not a
    value.  The originally proposed fix, a golden pin of each component, was not
    written: there is no source in this tree for JAM24's individual momentum
    fractions that is independent of ``dump_jam`` itself, so a pin would have
    been ``F2`` (freeze what the code produces) exactly where the item asked for
    an independent expectation.  The asymptotic fixed point is independent, and
    it is per-component.

    MEASURED this pass (JAM24 central, ``Q0 = 2`` GeV, ``cfg.ORDER``): the gluon
    fraction rises monotonically ``0.42366 -> 0.45934 -> 0.47348 -> 0.48129 ->
    0.48636 -> 0.49270 -> 0.49658`` across ``Q = 2 ... 1e8`` GeV, i.e. it
    approaches ``16/31 = 0.516129`` strictly **from below** and never crosses
    it, with the gap shrinking every step (``9.25e-02`` down to ``1.95e-02``).
    The approach is only logarithmic in ``Q``, and NLO shifts the fixed point at
    the percent level, so the bar below is on the *direction and monotonicity*
    of the flow plus a loose ``< 0.03`` cap on the remaining gap at ``1e8`` --
    deliberately not an equality with ``16/31``, which would be a claim this
    evolution does not make.

    **Why it is per-component.** Swapping the two input fractions (MEASURED:
    the swapped input's gluon fraction runs ``0.57630 -> 0.55742 -> 0.53936 ->
    ... -> 0.52053``) approaches the same fixed point strictly **from above** at
    every scale, so the sign assertion below flips; and it decreases with ``Q``,
    so the monotone-increase assertion flips too.  Neither the three ``.sum()``
    checks nor the ``[0] > [1]`` check in the test above distinguishes the two
    at the input scale in any way this one does not.
    """
    input_momentum = _jam24_input_momentum()
    scales = (2.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e6, 1.0e8)

    gluon_fractions = []
    for q_gev in scales:
        operator = singlet_evolution_factors(
            [MOMENTUM_N],
            Q2=float(q_gev) ** 2,
            Q20=INPUT_Q_GEV**2,
            order=cfg.ORDER,
            mode=cfg.MODE,
            alphaS_MZ=cfg.ALPHAS_MZ,
            mc=cfg.MC,
            mb=cfg.MB,
        )[:, :, 0]
        evolved = operator @ input_momentum
        gluon_fractions.append(float(evolved[1] / evolved.sum()))

    gluon_fractions = np.array(gluon_fractions)
    gaps = ASYMPTOTIC_GLUON_FRACTION - gluon_fractions

    # Approach from below, at every scale.  A component swap makes every one of
    # these negative (measured; see docstring).
    assert np.all(gaps > 0.0), gaps
    # Monotone flow toward the fixed point -- not merely "ends up near it".
    assert np.all(np.diff(gluon_fractions) > 0.0), gluon_fractions
    assert np.all(np.diff(gaps) < 0.0), gaps
    # MEASURED: 1.95e-02 remaining at Q = 1e8 GeV.  A cap, not an equality:
    # the approach is logarithmic and the NLO fixed point is not exactly 16/31.
    assert gaps[-1] < 3.0e-2, gaps[-1]
    # The singlet fraction is the complement by construction, so state the
    # per-component consequence explicitly rather than leaving it implied.
    assert 1.0 - gluon_fractions[-1] > ASYMPTOTIC_SINGLET_FRACTION


def test_conservation_atol_catches_a_realistic_splitting_coefficient_error(
    monkeypatch,
):
    """``PIXEL_CONSERVATION_ATOL``'s sensitivity boundary, asserted rather than
    described.

    That constant's comment records a coefficient-perturbation scan -- "catches
    a >=1e-4 relative error and loses sensitivity only below about one part in a
    million" -- but until this pass nothing executable held it, so a refactor
    that narrowed the bar's reach (a cancellation introduced upstream, a
    conservation check applied after a re-normalization step) would leave every
    assertion in this file passing and the recorded claim silently false.
    Oracle A2: the ``N=2`` column sum is exactly 1 for the true splitting
    functions, so its deviation reads out the injected error directly.

    The perturbation is applied **in memory** to ``lo_singlet``'s ``pgq`` entry
    (``P[1, 0]``; the Mellin moments sit on the trailing axis, so ``P[..., 1,
    0]`` would silently perturb ``pqg``/``pgg`` instead -- checked).
    ``_SPLITTING_CACHE`` is keyed on the contour nodes and ``(nf, order,
    channel)``, **not** on the splitting functions themselves, so it must be
    cleared around every call or the perturbation is a silent no-op -- measured:
    without the clear, all four perturbation sizes returned the clean defect
    bit-for-bit.  It is cleared again in ``finally`` so no perturbed entry
    outlives this test.  Nothing is written to disk: ``cache_path`` is ``None``
    on every call here.

    MEASURED this pass at ``Q = 10`` GeV from ``Q0 = 2`` GeV, quark column
    (gluon column in parentheses):

    ======================  ============================
    relative ``pgq`` error  ``|N=2 column sum - 1|``
    ======================  ============================
    ``0`` (clean)           ``1.95e-09`` (``3.78e-08``)
    ``1e-7``                ``1.62e-08`` (``3.60e-08``)
    ``1e-6``                ``1.80e-07`` (``1.99e-08``)
    ``1e-5``                ``1.82e-06`` (``1.41e-07``)
    ``1e-4``                ``1.82e-05`` (``1.75e-06``)
    ======================  ============================

    so ``PIXEL_CONSERVATION_ATOL = 5e-7`` sits between the ``1e-6`` and ``1e-5``
    rows -- both directions are pinned below, which is the point: a bar that
    only ever gets a "caught it" assertion can be tightened into uselessness (a
    false alarm on the clean case) without any test objecting.
    """
    import pixel.kernels.pqcd.splitting as splitting

    original = splitting.lo_singlet

    def column_defect() -> np.ndarray:
        splitting._SPLITTING_CACHE.clear()
        operator = singlet_evolution_factors(
            [MOMENTUM_N],
            Q2=100.0,
            Q20=INPUT_Q_GEV**2,
            order=cfg.ORDER,
            mode=cfg.MODE,
            alphaS_MZ=cfg.ALPHAS_MZ,
            mc=cfg.MC,
            mb=cfg.MB,
        )[:, :, 0]
        return np.abs(operator.sum(axis=0) - 1.0)

    def perturbed(relative):
        def _lo_singlet(N, nf):
            matrix = np.array(original(N, nf))
            matrix[1, 0] = matrix[1, 0] * (1.0 + relative)
            return matrix

        return _lo_singlet

    try:
        # Control: unperturbed, the defect is far under the bar.  Without this
        # the "caught" assertions would pass on a validator that rejects
        # everything.  MEASURED: 1.95e-09 / 3.78e-08.
        clean = column_defect()
        assert np.all(clean < PIXEL_CONSERVATION_ATOL), clean

        monkeypatch.setattr(splitting, "lo_singlet", perturbed(1.0e-5))
        caught = column_defect()
        # Premise guard: the perturbation must actually have landed (the cache
        # and the index are both easy to get wrong -- see the docstring).
        assert caught[0] > 10.0 * clean[0], (caught, clean)
        # MEASURED 1.82e-06 against the 5e-7 bar: a realistic coefficient typo
        # is caught with 3.6x margin.
        assert caught[0] > PIXEL_CONSERVATION_ATOL, caught

        monkeypatch.setattr(splitting, "lo_singlet", perturbed(1.0e-7))
        missed = column_defect()
        # MEASURED 1.62e-08: below the bar, so the bar is not so tight that the
        # floating-point floor trips it.  This half fails if the bar is ever
        # tightened past what the polygamma cancellation floor supports.
        assert np.all(missed < PIXEL_CONSERVATION_ATOL), missed
        assert missed[0] > clean[0]      # the perturbation did land, just small
    finally:
        splitting._SPLITTING_CACHE.clear()


def test_jam24_valence_flavor_sum_rules_at_input_scale():
    """JAM24 has the expected proton valence counts at the 2 GeV input scale.

    Fixture sanity gate, like the momentum version above: exercises the JAM24
    dump and its reading pipeline, not ``src/pixel``.  Oracle is the exact
    valence quantum numbers ``(2, 1, 0, 0)`` for ``(u, d, s, c)``; bar is
    ``JAM24_VALENCE_ATOL = 5.0e-3``, measured hit at a ``1.985e-3`` maximum
    deviation (``u`` quark) -- 2.5x headroom, tight rather than a loose
    afterthought.  Would catch a JAM24 grid, member, or reading bug that shifts
    a valence count past that margin.
    """
    input_valence = _jam24_input_valence_counts()

    np.testing.assert_allclose(
        input_valence,
        EXPECTED_VALENCE_COUNTS,
        atol=JAM24_VALENCE_ATOL,
        rtol=0.0,
    )


@pytest.mark.parametrize("target_q_gev", TARGET_Q_GEV)
def test_pixel_non_singlet_evolution_preserves_jam24_valence_sum_rules(target_q_gev):
    """PIXEL preserves ``int dx (q - qbar) = N_q`` at requested scales.

    NLO only.  At N=1, quark-number conservation requires ``lo_nonsinglet(1) = 0``
    (an emergent cancellation, ``4/3*(3 + 1 - 4)``, not a special-cased identity)
    and the NLO minus-channel piece to vanish there too; both are verified
    numerically here, not imposed.  Bar is ``PIXEL_CONSERVATION_ATOL = 5e-7``,
    measured hit at ``1.1e-9`` to ``2.9e-9`` across the five scales.  Unlike the
    singlet momentum test above, every assertion here is elementwise over the
    four flavors, so this does not share that test's swap-blind spot: a
    ``u<->d`` component reorder would fail immediately against the asymmetric
    ``EXPECTED_VALENCE_COUNTS = (2, 1, 0, 0)``.
    """
    input_valence = _jam24_input_valence_counts()

    factor = evolution_factors(
        [VALENCE_N],
        Q2=float(target_q_gev) ** 2,
        Q20=INPUT_Q_GEV**2,
        order=cfg.ORDER,
        channel="minus",
        mode=cfg.MODE,
        alphaS_MZ=cfg.ALPHAS_MZ,
        mc=cfg.MC,
        mb=cfg.MB,
    )[0]

    evolved_valence = float(factor) * input_valence

    assert factor == pytest.approx(1.0, abs=PIXEL_CONSERVATION_ATOL)
    np.testing.assert_allclose(
        evolved_valence,
        input_valence,
        atol=PIXEL_CONSERVATION_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        evolved_valence,
        EXPECTED_VALENCE_COUNTS,
        atol=JAM24_VALENCE_ATOL,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "target_q_gev",
    (
        pytest.param(cfg.MC, id="mc", marks=pytest.mark.slow),
        pytest.param(4.0, id="4gev"),
        pytest.param(5.0, id="5gev", marks=pytest.mark.slow),
        pytest.param(10.0, id="10gev"),
        pytest.param(100.0, id="100gev"),
    ),
)
def test_xspace_reconstruction_preserves_momentum_sum_rule(target_q_gev):
    """After x-space evolution, re-integrating ``int dx x f`` preserves momentum.

    Routes through ``singlet_evolution_matrix``, which calls the same
    ``SingletEvolution.operator`` as the Mellin-space test above (module
    docstring) -- common-mode with it on the splitting-function physics; this
    test's independent territory is the finite-element basis, adaptive contour,
    and low-x-completion pipeline instead.  Bar ``XSPACE_MELLIN_ATOL = 1.2e-2``
    is four orders looser than the Mellin-space bar, dominated by the unweighted
    low-x completion below ``x_min ~ 1.2e-6`` (comment above ``XSPACE_GRID_N``),
    not contour error -- ``XSPACE_CONTOUR_TOL``'s comment records a 3x
    node-count refinement moving the Q=10/Q=100 GeV residuals by only
    ``6.5e-8``/``8.3e-8``.  Shares the singlet Mellin-space test's sum-only
    swap blind spot by the identical argument (only ``.sum()`` is ever
    compared); not separately re-measured here since the argument is
    structural.  ``mc`` and ``5gev`` are hand-marked ``pytest.mark.slow``
    because they need the finest contour panel and cross a flavour threshold
    respectively -- the module docstring's ``XSPACE_CONTOUR_TOL`` cross-
    reference gives ~38-47s for the comparable Q=10 GeV case.
    """
    reconstructed, expected, _, _ = _xspace_reconstructed_sum_rules(
        float(target_q_gev)
    )

    assert reconstructed.sum() == pytest.approx(
        expected.sum(),
        abs=XSPACE_MELLIN_ATOL,
    )
    assert reconstructed.sum() == pytest.approx(
        1.0,
        abs=XSPACE_SUM_RULE_ATOL,
    )


@pytest.mark.parametrize("target_q_gev", TARGET_Q_GEV)
def test_xspace_reconstruction_preserves_valence_sum_rules(target_q_gev):
    """After x-space evolution, re-integrating ``int dx (q-qbar)`` preserves Nq.

    Routes through ``non_singlet_evolution_matrix``, which calls the same
    ``NonSingletEvolution.sigma`` as the Mellin-space valence test above --
    common-mode on the splitting-function physics, independent on the
    basis/contour/low-x pipeline, same division of labor as the momentum pair
    above.  Bar ``XSPACE_MELLIN_ATOL``/``XSPACE_SUM_RULE_ATOL = 1.2e-2``,
    dominated by the same low-x completion (module docstring).  Elementwise
    over the four flavors like the Mellin-space valence test, so it retains
    that test's robustness to a component reorder rather than the momentum
    pair's blind spot.
    """
    _, _, reconstructed, expected = _xspace_reconstructed_sum_rules(
        float(target_q_gev)
    )

    np.testing.assert_allclose(
        reconstructed,
        expected,
        atol=XSPACE_MELLIN_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reconstructed,
        EXPECTED_VALENCE_COUNTS,
        atol=XSPACE_SUM_RULE_ATOL,
        rtol=0.0,
    )


# At LO the Mellin-space momentum identity is exact to float64
# (``tests/test_evolution.py::test_lo_singlet_momentum_columns_conserve_n2``;
# reproduced here at ``2.2e-16``), so an LO x-space residual contains *no*
# perturbative component -- it is the finite-element basis, the contour, and the
# missing region below ``x_min`` and nothing else.  That makes an LO case a
# strictly tighter regression guard on the reconstruction pipeline than any NLO
# one, which is why these bars are ~8x under ``XSPACE_MELLIN_ATOL``.
#
# MEASURED this pass at ``XSPACE_GRID_N = 201`` over all five target scales:
# momentum ``|reconstructed - expected|`` runs 1.56e-05 (mc), 5.60e-05 (4),
# 8.10e-05 (5), 1.85e-04 (10), 4.28e-04 (100 GeV), and ``|reconstructed - 1|``
# peaks at 4.80e-04; the valence deviation from its own Mellin-space
# expectation peaks at 1.44e-03.  The bars below keep ~3x over those.
XSPACE_LO_MELLIN_ATOL = 1.5e-3
XSPACE_LO_VALENCE_ATOL = 4.0e-3


@pytest.mark.parametrize(
    "target_q_gev",
    (
        pytest.param(cfg.MC, id="mc", marks=pytest.mark.slow),
        pytest.param(4.0, id="4gev"),
        pytest.param(5.0, id="5gev", marks=pytest.mark.slow),
        pytest.param(10.0, id="10gev"),
        pytest.param(100.0, id="100gev"),
    ),
)
def test_xspace_reconstruction_preserves_sum_rules_at_lo(target_q_gev):
    """The same x-space reconstruction at ``order="LO"``, where the perturbative
    part of the residual is exactly zero.

    Oracle A2, and the point of the case: the NLO tests above cannot say how much
    of their ``1.2e-2`` bar is the reconstruction pipeline and how much is the
    evolution, because both contribute.  At LO the second term vanishes
    identically, so this runs the *identical* grid, basis, contour and low-x
    completion and asserts a bar 8x tighter.  A regression in
    ``_basis_mellin_moments``, the adaptive contour sizing, or the
    ``power``/``alpha=1`` completion would move this long before it moved the
    NLO bar.

    **This measurement contradicts one claim in ``XSPACE_CONTOUR_TOL``'s comment,
    and the correction is worth having on record.**  That comment reads the
    ``1.0733e-2`` residual at ``Q = 100`` GeV as "the finite-element/JAM24 low-x
    floor, not contour error".  It is neither, at that scale.  MEASURED this
    pass, three ways:

    * **Order.** At ``Q <= 10`` GeV, LO and NLO residuals agree to ~15%
      (``1.85e-04`` vs ``2.10e-04`` at 10 GeV), consistent with a shared
      reconstruction floor.  At ``Q = 100`` GeV they differ by **25x**
      (``4.28e-04`` LO vs ``1.08e-02`` NLO) on the identical pipeline, so at
      that scale the NLO residual is overwhelmingly *not* the reconstruction.
    * **Refinement in ``x_min``, the parameter actually under suspicion.**
      Raising ``XSPACE_GRID_N`` 161 -> 201 -> 241 -> 281 (``x_min`` 1.9e-05 ->
      5.5e-09) drives the ``Q = 10`` GeV LO residual 1.34e-03 -> 1.85e-04 ->
      3.70e-05 -> 1.47e-05, i.e. it converges like a genuine low-x truncation.
      At ``Q = 100`` GeV the same sweep gives 2.21e-02 -> 4.28e-04 -> 2.80e-04
      -> 7.36e-04 (LO) and 4.70e+00 -> 1.08e-02 -> 6.61e-03 -> 6.20e-03 (NLO):
      it stops converging and plateaus, an order of magnitude apart by order.
    * **Contour.** Tightening ``XSPACE_CONTOUR_TOL`` 1e-2 -> 3e-3 at ``Q = 100``
      GeV NLO changes the residual by exactly nothing (``1.0763e-02`` both), so
      it is not contour error either, consistent with that comment's own scan.

    Nothing here asserts on the ``Q = 100`` GeV NLO residual -- that stays the
    NLO test's business at its existing bar.  The measurements are recorded so
    the next reader does not re-derive them, and so the plateau is not mistaken
    for a converged floor.
    """
    reconstructed, expected, valence, valence_expected = (
        _xspace_reconstructed_sum_rules(float(target_q_gev), "LO")
    )

    # MEASURED: worst 4.28e-04 across the five scales, against this bar.
    assert reconstructed.sum() == pytest.approx(
        expected.sum(),
        abs=XSPACE_LO_MELLIN_ATOL,
    )
    # MEASURED: worst 4.80e-04.  The JAM24 input itself is only 3.87e-05 off 1.0
    # (``JAM24_NORMALIZATION_ATOL``'s comment), so this leg is dominated by the
    # reconstruction, not by the input set.
    assert reconstructed.sum() == pytest.approx(
        1.0,
        abs=XSPACE_LO_MELLIN_ATOL,
    )
    # Elementwise over the four flavors, so unlike the momentum legs this one
    # would also catch a flavor reorder.  MEASURED: worst 1.44e-03.
    np.testing.assert_allclose(
        valence,
        valence_expected,
        atol=XSPACE_LO_VALENCE_ATOL,
        rtol=0.0,
    )
