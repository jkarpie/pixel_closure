"""Property tests for the injected full-scale closure lattice systematics.

Exercises ``closure_JAM_truth.generate`` / ``closure_NNPDF_truth.generate`` (physically
duplicated twin modules -- ``build_systematic_curves``, ``fold_lattice_systematics``,
``inflate_diagonal``, ``_mellin_n1_row``, ``coefficient``, ``lattice_layout``,
``assemble_operator``, ``fold_truth``), ``closure_JAM_truth.config`` /
``closure_NNPDF_truth.config`` (``Ensemble.z_max``, ``Ensemble.p_max``,
``active_ensembles``, ``low_x_completion``, ``moment_nuisance_specs``), and
``closure_JAM_truth.datasets`` / ``closure_NNPDF_truth.datasets``
(``_moment_systematic_contributions``); and, past those closure-suite drivers,
``pixel.kernels.common`` (``Cosine``/``Sine``, aliased ``PseudoITDReal``/
``PseudoITDImag``, plus ``Mellin`` and ``Unit``), ``pixel.kernels.lowx``
(``low_x_fourier_head``, ``low_x_quadrature_correction``) and
``pixel.kernels.lattice`` (``LATTICE_SYSTEMATICS``).

These check the enforced properties of the injected lattice-systematic curves
(closure_extension.md / closure_extension_plan.md) *without* running a fit or the
slow DIS fold:

1. parity in nu -- real (cosine) even, imag (sine) odd;
2. convergence as nu -> infinity (Riemann-Lebesgue decay);
3. zero shift at nu = 0 for the real component AND zero n=1 counting moment
   (two constraints for the singlet/gluon ``alpha=0`` convention);
4. every field/systematic pair obeys the 10% cap in the generation fold;
5. covariance inflation makes the actual folded replica blocks SPD.

Also sanity-checks both halves of the ensemble table's kinematic grid -- the
ceil-based ``z`` range and the floor-based ``p`` range -- the ``fold_lattice_systematics``
input guards, ``inflate_diagonal`` in isolation, ``active_ensembles``'s override
branches, and the fixed-truth (``truth_curves=``) fold branch that ``generate_member``
actually uses in production.

Oracle independence, stated plainly (measured 2026-08-13; see each test's docstring
for the numbers).

Properties 3 (nulled ``n=1`` moment and vanishing ``nu=0`` real value) and 4 (the
10% cap) are now checked against oracles built **outside** the code that produced
them, which is what makes them able to fail:

- ``_independent_power_moment_row`` below re-derives ``int_0^1 x^w e_j(x) dx`` from
  the basis alone, with a composite Gauss-Legendre rule in ``u = log x`` and a
  closed-form ``[0, x_min)`` head -- a different quadrature, a different variable and
  a different low-x treatment from ``pixel.kernels``'s own composite rule plus
  ``low_x_quadrature_correction``/``low_x_fourier_head``. Agreement is
  ``1.2e-14`` relative on every basis column except column 0 (see
  ``test_nulled_counting_moment`` for the one measured discrepancy there).
- ``test_generation_fold_enforces_cap_for_every_field_key`` rebuilds the shift and
  the central value from ``sys_curves``, ``coefficient`` and the fixture's own
  operator/ensemble, so only ``folds.scale`` still comes out of the function under
  test.

Both column-0 bars (``1e-3`` in ``test_nulled_counting_moment``, ``head_bar`` in
``test_real_vanishes_at_nu0_imag_vanishes_at_nu0``) compare two evaluations of the
*same* low-x completion, so neither can see the completion being the wrong shape --
which is the larger error by ~1000x for a sublinear field. ``1e-3`` is
``1/(npts+1)**2`` at the shipped ``npts = 32``, and is ~independent of ``x_min``
(measured: 6% over four decades). ``test_low_x_head_error_converges_in_x_min`` carries
the claim those bars cannot: it sweeps ``x_min`` in ``{1e-6, 1e-7, 1e-8}`` at fixed
``n_points = 145``, states ``x_min`` explicitly instead of inheriting ``cfg.X_MIN``,
and asserts the per-decade **rate** -- ``10**a`` for the vanishing class, ``x100`` for
``sigma``/``g`` -- against 40-digit ``mpmath`` truths. Full measured diagnosis:
``plans/low_x_head_diagnosis.md``.

Property 1 (parity) remains a statement about the ``Cosine``/``Sine`` kernel rather
than about the injected curve -- it is exactly even/odd in ``nu`` for *any* input
vector -- and is labelled as such; it now at least covers both production
``alpha``/``low_x_extension`` branches instead of only the ``alpha=-1`` one.
Property 2 (decay) is measured against the real curves, with the bound split by
``alpha`` class because a single bound was ~300x looser for the ``alpha=0`` fields
than for the ``alpha=-1`` ones. Property 5 (SPD) is exercised on two replica
ensembles: the original rank-1-by-construction toy, and a full-rank noisy one whose
folded covariance is singular only for the emergent reason (``n_rep - 1 < n_pts``).

The systematic-*coefficient* formulas (``ht``, ``chiral``, ``inf_Lz``, ...) are
independently pinned in ``tests/test_coefficients.py``, not here -- this file treats
the ``LATTICE_SYSTEMATICS`` registry as a shared oracle for wiring only. There is no
separate tighter/production-size version of this file: the 13-ensemble lattice layout
exercised here already matches what ``closure_JAM_truth.generate`` writes to disk.

**Which suite each test runs against, since 2026-08-14.** The module-scoped ``suite``
fixture is NNPDF-truth only (see ``FULL_SUITES``); it used to carry a JAM leg as well,
re-running every property test against the physically duplicated twin package. The
owner dropped that as duplication. The two tests that genuinely compare the truth
families -- ``test_seeded_systematic_curves_match_between_truth_families`` and
``test_mellin_systematic_truths_are_independent_and_deterministic_per_order`` -- do
not use the fixture at all; they import ``jam_cfg``/``jam_gen`` and ``nnpdf_cfg``/
``nnpdf_gen`` directly and are untouched, which is why all four imports remain.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from pixel import kernels
from pixel.core.model import Contribution, Field
from pixel.geometry import Grid

from closure_JAM_truth import config as jam_cfg
from closure_JAM_truth import generate as jam_gen
from closure_NNPDF_truth import config as nnpdf_cfg
from closure_NNPDF_truth import generate as nnpdf_gen


def _low_x_gamma(low_x_extension) -> float:
    """Power ``gamma`` of the completion ``L(x) = (x/x_min)**gamma`` below ``x_min``.

    ``"flat"`` is ``gamma = 0``; a ``{"kind": "power", "alpha": g}`` spec is ``g``.
    Kept separate from ``pixel.kernels.lowx.normalize_low_x_extension`` on purpose:
    the whole point of the independent row below is not to route through the module
    whose head it is checking.
    """
    if low_x_extension == "flat":
        return 0.0
    assert low_x_extension["kind"] == "power", low_x_extension
    return float(low_x_extension["alpha"])


def _independent_power_moment_row(basis, nodes, weight, gamma, points_per_interval=24):
    r"""``row[j] = int_0^1 x**weight e_j(x) dx`` for a ``(x/x_min)**gamma`` low-x tail.

    Deliberately built without ``pixel.kernels``:

    * substitute ``x = exp(u)``, so ``int x^w e_j dx = int exp((w+1)u) e_j(exp(u)) du``
      -- a change of variable PIXEL's assembler does not make, and one that removes
      the ``1/x`` endpoint singularity of the ``w = -1`` (momentum-density) case
      instead of quadrature-ing through it;
    * composite Gauss-Legendre with ``points_per_interval`` nodes on each grid
      interval, rather than PIXEL's ``basis.quadrature`` composite rule;
    * the ``[0, x_min)`` head in closed form,
      ``int_0^x_min x^w (x/x_min)^gamma dx = x_min**(w+1) / (w + 1 + gamma)``, added
      to column 0 (the basis is nodal at ``x_min`` -- asserted by the tests that use
      this) rather than PIXEL's ``low_x_quadrature_correction`` Gauss-Jacobi rule or
      its ``low_x_fourier_head`` Taylor series.

    Converged in ``points_per_interval``: measured identical to ``8.6e-4`` (column 0)
    and ``1.2e-14`` (all other columns) against ``gen._mellin_n1_row`` at 8, 16, 24
    and 40 points per interval, so the residual is a real scheme difference and not
    this rule's truncation error.
    """
    x_min = float(basis.domain[0])
    u_edges = np.log(np.asarray(nodes, dtype=float))
    gl_x, gl_w = leggauss(int(points_per_interval))
    mids = 0.5 * (u_edges[1:] + u_edges[:-1])
    halves = 0.5 * (u_edges[1:] - u_edges[:-1])
    u = (mids[:, None] + halves[:, None] * gl_x[None, :]).ravel()
    w = (halves[:, None] * gl_w[None, :]).ravel()
    evaluated = np.asarray(basis.evaluate(np.exp(u)), dtype=float)
    row = (w * np.exp((float(weight) + 1.0) * u)) @ evaluated
    row[0] += x_min ** (float(weight) + 1.0) / (float(weight) + 1.0 + float(gamma))
    return row


#: The ``(config, generate)`` pairs the module-scoped ``suite`` fixture parametrizes over.
#:
#: **NNPDF truth only, on the owner's instruction 2026-08-14.** This was
#: ``(pytest.param((jam_cfg, jam_gen), id="jam"), pytest.param((nnpdf_cfg, nnpdf_gen),
#: id="nnpdf"))``, so every property test in this file ran twice against physically
#: duplicated twin packages whose ``generate.py`` differs only in which truth PDF set it
#: was written for -- and none of those properties (parity, decay, the nulled ``n=1``
#: moment, the 10% cap, SPD after inflation) reads the truth set. The comparison that
#: *does* is made directly by ``test_seeded_systematic_curves_match_between_truth_families``
#: and ``test_mellin_systematic_truths_are_independent_and_deterministic_per_order``,
#: which bypass this fixture and import ``jam_cfg``/``jam_gen`` themselves. **Those two
#: tests are the reason the JAM imports above must stay** even though no ``pytest.param``
#: here names them.
FULL_SUITES = (pytest.param((nnpdf_cfg, nnpdf_gen), id="nnpdf"),)


@pytest.fixture(scope="module", params=FULL_SUITES)
def suite(request):
    """One ``(config, generate)`` module pair -- see ``FULL_SUITES``.

    Every test below that takes this fixture runs once per pair. The two tests that
    import both truth families directly, to cross-check them against each other, do
    not take it.
    """
    return request.param


@pytest.fixture(scope="module")
def basis_nodes(suite):
    """The shared FE basis and its node array -- every closure field lives on it."""
    cfg, _ = suite
    fields = cfg.make_fields()
    f0 = list(fields.values())[0]
    return f0.basis, np.asarray(f0.nodes).reshape(-1)


@pytest.fixture(scope="module")
def sys_curves(suite, basis_nodes):
    """The real, seeded x-space systematic curves -- one per (field, ITD key).

    Built by :func:`~closure_JAM_truth.generate.build_systematic_curves`, which
    projects a random power-law shape to null its ``n=1`` Mellin moment (and, for
    ``alpha=0`` fields, its real ITD value at ``nu=0``) before any 10% cap is
    applied.  Several tests below substitute an unrelated vector for these curves
    to show which of their own assertions do, and do not, depend on this specific
    content.
    """
    cfg, gen = suite
    basis, nodes = basis_nodes
    rng = np.random.default_rng(cfg.SYSTEMATIC_SEED)
    return gen.build_systematic_curves(nodes, basis, rng)


@pytest.fixture(scope="module")
def independent_row(suite, basis_nodes):
    """``(weight, gamma) -> _independent_power_moment_row(...)``, memoized per suite.

    One cache per suite module pair so the three tests that need these rows
    (``n=1`` moment, ``nu=0`` real value, small-``nu`` imaginary slope) each pay for
    the basis evaluation once rather than per (field, key) pair.
    """
    basis, nodes = basis_nodes
    cache: dict[tuple[float, float], np.ndarray] = {}

    def get(weight, gamma):
        """Return the memoized independent row for ``(weight, gamma)``."""
        key = (float(weight), float(gamma))
        if key not in cache:
            cache[key] = _independent_power_moment_row(basis, nodes, *key)
        return cache[key]

    return get


def test_basis_is_nodal_at_x_min(basis_nodes):
    """``e_j(x_min) == delta_{j0}``, the premise both low-x heads are built on.

    ``pixel.kernels.base.base_matrix`` adds the whole ``[0, x_min)`` low-x integral
    into **column 0** of the kernel matrix, which is only correct if the field value
    at ``x_min`` is the zeroth basis coefficient. ``_independent_power_moment_row``
    makes the same assumption for its own closed-form head, so the two would agree
    with each other even if the premise were false. Asserted here once, exactly
    (measured: the row at ``x_min`` is bit-exactly ``[1, 0, ..., 0]``), so the other
    tests may rely on it.
    """
    basis, _ = basis_nodes
    row = np.asarray(basis.evaluate(np.array([float(basis.domain[0])])))[0]
    expected = np.zeros(row.size)
    expected[0] = 1.0
    np.testing.assert_array_equal(row, expected)


def test_z_range_is_ceil(suite):
    """``Ensemble.z_max == ceil(Z_PHYS_MAX_FM / a_fm)`` and ``z_values`` starts at 1.

    ``np.ceil`` here is an independent recomputation of the same closed-form
    ceiling division as ``Ensemble.z_max`` (``config.py``'s ``math.ceil``), so a
    wrong rounding mode (floor, or truncation via plain ``int()``) would be
    caught: none of the 13 configured ensembles has an exactly-integral
    ``Z_PHYS_MAX_FM / a_fm``, so ceil and floor/truncate disagree for every one
    of them. Exact integer equality, no tolerance. Its floor-based sibling
    ``p_max``/``p_values`` is covered by ``test_p_range_is_floor``.
    """
    cfg, _ = suite
    for ensemble in cfg.active_ensembles():
        assert ensemble.z_max == int(np.ceil(cfg.Z_PHYS_MAX_FM / ensemble.a_fm))
        assert np.array_equal(
            ensemble.z_values,
            np.arange(1, ensemble.z_max + 1, dtype=float),
        )


def test_p_range_is_floor(suite):
    """``Ensemble.p_max == floor(L_sites / P_INDEX_DIVISOR)`` and ``p_values`` from 1.

    The structural sibling of ``test_z_range_is_ceil``, and the other half of the
    ``(z, p)`` meshgrid ``lattice_layout`` builds -- an off-by-one in ``p_max``
    silently changes every ensemble's pseudo-ITD row count. Oracle is an independent
    ``np.floor`` recomputation against ``config.py``'s ``math.floor`` (``A1``), exact
    integer equality.

    Unlike the ``z`` sibling, the rounding mode is **not** discriminated by every
    ensemble: ``L_sites / 6`` is an exact integer for 8 of the 13 (``L = 48, 72,
    96``), where floor, ceil and truncation all coincide. The remaining five
    (``L = 32`` twice giving ``5`` vs ``6``, ``L = 64`` three times giving ``10`` vs
    ``11``) do separate them, and the second loop below asserts that this
    discriminating set is non-empty rather than assuming it -- if the ensemble table
    is ever changed to all-divisible extents, this test goes quiet and says so.
    """
    cfg, _ = suite
    discriminating = 0
    for ensemble in cfg.active_ensembles():
        assert ensemble.p_max == int(
            np.floor(ensemble.L_sites / cfg.P_INDEX_DIVISOR)
        )
        assert np.array_equal(
            ensemble.p_values,
            np.arange(1, ensemble.p_max + 1, dtype=float),
        )
        if int(np.ceil(ensemble.L_sites / cfg.P_INDEX_DIVISOR)) != ensemble.p_max:
            discriminating += 1
    # Measured 2026-08-13: 5 of the 13 ensembles separate floor from ceil.
    assert discriminating >= 1, "no ensemble distinguishes floor from ceil for p_max"


def test_finite_volume_systematics_are_enabled_for_both_lattice_data_types(suite):
    """Pseudo-ITD gets the z-dependent ``inf_Lz`` key; Mellin moments get ``inf_L``.

    Structural contract; constrains no numerics. Plain membership on the
    hardcoded ``ITD_SYSTEMATICS``/``MOMENT_SYSTEMATICS`` config tuples -- this
    is close to re-stating the config, but it does catch the one plausible
    mistake of reusing the same finite-volume key for both data kinds, which
    would conflate a per-separation and a separation-free nuisance.
    """
    cfg, _ = suite
    assert "inf_Lz" in cfg.ITD_SYSTEMATICS
    assert "inf_L" not in cfg.ITD_SYSTEMATICS
    assert "inf_L" in cfg.MOMENT_SYSTEMATICS
    assert "inf_Lz" not in cfg.MOMENT_SYSTEMATICS


def test_nulled_counting_moment(suite, basis_nodes, sys_curves, independent_row):
    """Every injected curve's ``n=1`` Mellin moment is ~0 -- against an independent
    quadrature, not against the projector's own constraint row.

    Three assertions of decreasing tautology and increasing power:

    1. ``m1 @ s ~ 0`` with ``m1 = gen._mellin_n1_row(basis)``. Kept, but it is an
       algebraic identity: ``build_systematic_curves``'s ``np.linalg.solve``
       projector makes ``C @ s == 0`` to solver precision for *whatever* rows ``C``
       is handed, and this re-derives one of those same rows through the same
       function. Measured 2026-08-13: monkeypatching ``_mellin_n1_row`` to the
       ``n=2`` functional and rebuilding the curves still passes it (residual
       ``2.1e-14``) while those curves' true ``n=1`` moment is ``51.6``. It earns
       its place only as a check that the projection ran at all.
    2. ``_mellin_n1_row`` itself against ``_independent_power_moment_row(-1, gamma=
       LOW_X_LINEAR_POWER)`` -- a different quadrature in a different variable with a
       closed-form low-x head (see that helper). This is what pins the Mellin order,
       the ``alpha`` weight and the low-x power that assertion 1 cannot see.
    3. the curves' moment under that independent row, normalized by ``max|s|``.

    Two measured numbers set the bars, both converged in the independent rule's
    order (identical at 8/16/24/40 points per interval):

    * columns ``j >= 1`` agree to ``1.19e-14`` relative; bar ``1e-12``.
    * column 0 disagrees by ``8.557e-04`` relative, and **this is PIXEL's error, not
      the oracle's**. Column 0 carries the ``[0, x_min)`` head. For ``alpha != 0``
      ``pixel.kernels.base.base_matrix`` disables the closed-form head and falls back
      to ``low_x_quadrature_correction``, whose Gauss-Jacobi weight ``t**gamma`` does
      not account for the ``x**alpha`` baked into the kernel, leaving a ``1/t``
      singularity in the sampled integrand. For this configuration the exact head is
      ``int_0^x_min x^-1 (x/x_min)^1 dx = 1`` exactly; PIXEL returns
      ``0.99908173`` (measured directly on ``low_x_quadrature_correction``), i.e.
      ``-9.18e-04`` relative. The bar is ``1e-3``, just above it, so the check still
      pins the order/alpha/power while tolerating the known head error; a tighter bar
      would be asserting a ``src/`` bug fixed. The sine head at the same settings is
      exact to ``2e-16`` -- ``sin(nu x)/x`` is regular at the origin -- so only the
      even-kernel/Mellin case is affected.

      **What that ``1e-3`` is a bar on, stated so it cannot be misread as a general
      head-accuracy guarantee.** It is the Gauss-Jacobi rule's own error, whose
      closed form is ``1/(npts+1)**2`` -- so it is set by
      ``LowXExtension.npts = 32``, and it is very nearly *independent of ``x_min``*.
      Measured 2026-08-13 by sweeping both axes at this fixture's ``n_points``:

      ==========  ==========  ==========  ==========
      ``x_min``   npts 16     npts 32     npts 64
      ==========  ==========  ==========  ==========
      ``1e-4``    3.3255e-03  8.8037e-04  2.2677e-04
      ``1e-6``    3.2320e-03  **8.5569e-04**  2.2041e-04
      ``1e-8``    3.1421e-03  8.3194e-04  2.1430e-04
      ==========  ==========  ==========  ==========

      Four decades of ``x_min`` move it by 6%; one doubling of ``npts`` moves it by
      3.8x (``17**2 / 33**2 = 0.2654``, measured ``0.2648``). So refining ``x_min``
      will not retire this bar and refining ``npts`` will -- the opposite of the
      *model* error that dominates the same column, which responds only to ``x_min``.
      Both effects are separated and asserted as rates in
      ``test_low_x_head_error_converges_in_x_min``; the full measurement is
      ``plans/low_x_head_diagnosis.md``.
    * the resulting curve moment ``|m1_ind @ s| / max|s|`` reaches ``6.32e-05``
      across all 45 (field, key) pairs of both suites; bar ``2e-4`` (3.2x headroom).
      For contrast, the ``n=2``-functional mutation above puts this ratio at ``O(70)``.
    """
    cfg, gen = suite
    basis, _ = basis_nodes
    m1 = gen._mellin_n1_row(basis)
    m1_ind = independent_row(-1.0, cfg.LOW_X_LINEAR_POWER)

    ratio = np.abs(m1_ind / m1 - 1.0)
    assert np.max(ratio[1:]) < 1e-12, f"cols>=1 max rel {np.max(ratio[1:]):.3e}"
    # 1e-3 is the Gauss-Jacobi head rule's own 1/(npts+1)**2 = 9.18e-4 at the shipped
    # npts=32, NOT a head-accuracy guarantee and NOT an x_min statement (measured:
    # 6% over four decades of x_min, 3.8x per doubling of npts -- see the docstring).
    # The x_min behaviour is asserted in test_low_x_head_error_converges_in_x_min.
    assert ratio[0] < 1e-3, f"col 0 rel {ratio[0]:.3e} (bar 1e-3 = npts=32 head rule)"

    for field in cfg.ALL_FIELDS:
        for key in cfg.ITD_SYSTEMATICS:
            s = sys_curves[field][key]
            # Assertion 1: the projector ran (tautological about which row it used).
            moment = float(m1 @ s)
            assert abs(moment) < 1e-9, f"{field}/{key} n=1 moment={moment}"
            # Assertion 3: the moment is really zero, measured outside pixel.kernels.
            scale = float(np.max(np.abs(s)))
            assert scale > 1e-3, f"{field}/{key} curve is degenerate: max|s|={scale}"
            independent = abs(float(m1_ind @ s)) / scale
            assert independent < 2e-4, (
                f"{field}/{key} independent n=1 moment/max|s| = {independent:.3e}"
            )


def test_real_even_imag_odd_in_nu(suite, basis_nodes, sys_curves):
    """``Real(nu) == Real(-nu)``, ``Imag(nu) == -Imag(-nu)``, bar ``atol=1e-10``.

    This is a property of the ``Cosine``/``Sine`` kernel (``pixel.kernels.common``),
    not of the systematic curve: ``kernel_func`` is ``cos(nu*x)``/``sin(nu*x)``,
    and the closed-form low-x head (``pixel.kernels.lowx.low_x_fourier_head``)
    depends on ``nu`` only through ``b = nu*x_min`` via ``b**2`` (cosine branch)
    or an odd power series starting at ``b**1`` (sine branch) -- both exactly
    even/odd for *any* vector dotted against the row, before ``s`` is ever
    involved. Measured this audit: substituting a random, unrelated vector for
    ``sys_curves[field][key]`` gives an identical ``0.000e+00`` residual, same as
    the real curve. So this test constrains the kernel's handling of negative
    ``nu`` (a real thing worth checking), not the specific injected curve, contra
    the "parity in nu" framing of property 1 in the module docstring.

    What it now also does, which it did not before: run each field through its own
    production ``alpha``/``low_x_extension`` (``cfg.itd_momentum_density``,
    ``cfg.low_x_completion``) as well as the fixed ``alpha=-1``/linear-power pair.
    The old form used ``alpha=-1``/linear for every field and so never entered the
    ``alpha=0``/``low_x_extension="flat"`` branch -- which is a *different* low-x
    code path (``low_x_fourier_head``'s closed-form series, versus the
    ``low_x_quadrature_correction`` fallback that ``alpha != 0`` forces), and the one
    production uses for ``sigma`` and ``g``. Measured 2026-08-13: the residual is
    exactly ``0.000e+00`` on both branches, real and imaginary, so the bar below is
    only a formality -- it is the branch coverage that is the fix here.
    """
    cfg, _ = suite
    basis, _ = basis_nodes
    linear = {"kind": "power", "alpha": 1.0}
    settings = {"fixed-alpha-minus-1": (-1.0, linear)}
    for field in cfg.ALL_FIELDS:
        alpha = -1.0 if cfg.itd_momentum_density(field) else 0.0
        low_x = cfg.low_x_completion(field)
        settings[f"production-alpha-{alpha:+.0f}"] = (alpha, low_x)
    # Both production branches must actually be present, or this loop silently
    # degenerates back to the single-branch form it is replacing.
    assert len(settings) == 3, sorted(settings)

    nu = np.linspace(0.5, 8.0, 12)
    for label, (alpha, low_x) in settings.items():
        real_k = kernels.PseudoITDReal(alpha=alpha, low_x_extension=low_x)
        imag_k = kernels.PseudoITDImag(alpha=alpha, low_x_extension=low_x)
        Br_pos = np.asarray(real_k.matrix(nu, basis))
        Br_neg = np.asarray(real_k.matrix(-nu, basis))
        Bi_pos = np.asarray(imag_k.matrix(nu, basis))
        Bi_neg = np.asarray(imag_k.matrix(-nu, basis))
        for field in cfg.ALL_FIELDS:
            for key in cfg.ITD_SYSTEMATICS:
                s = sys_curves[field][key]
                # atol is far looser than the achieved bit-level agreement (measured
                # 0.000e+00 on all three settings, for both the real curve and an
                # unrelated random vector); this bar constrains the kernel's own
                # nu-parity, not sys_curves.
                assert np.allclose(Br_pos @ s, Br_neg @ s, atol=1e-10), label
                assert np.allclose(Bi_pos @ s, -(Bi_neg @ s), atol=1e-10), label


def _apply(kernel, nu, basis, s):
    """Fold ``s`` through one row of ``kernel.matrix(nu, basis)`` -> ``(len(nu),)``."""
    return (np.asarray(kernel.matrix(np.atleast_1d(nu), basis)) @ s).reshape(-1)


def test_real_vanishes_at_nu0_imag_vanishes_at_nu0(
    suite, basis_nodes, sys_curves, independent_row
):
    """Real component ``~0`` at ``nu=0`` under an independent quadrature; imaginary
    kernel *row* at ``nu=0`` is exactly the zero vector.

    Uses each field's production ``alpha``/``low_x_extension``
    (``cfg.itd_momentum_density``, ``cfg.low_x_completion``). The two halves are
    different kinds of statement and are now labelled as such:

    - **real half** -- checked twice. ``r0`` is the kernel's own row against ``s``,
      which is the projector's tautology again (for ``alpha=-1`` fields the row
      agrees with ``gen._mellin_n1_row`` to ``5.0e-16``; for ``alpha=0`` fields it is
      bitwise the same construction as ``itd_zero`` inside
      ``build_systematic_curves``). ``r0_ind`` is
      ``_independent_power_moment_row(alpha, gamma)`` against ``s`` -- a different
      quadrature with a closed-form low-x head -- and that one can fail. Bars, both
      measured 2026-08-13 across all 45 (field, key) pairs of both suites:

      * ``alpha=0`` (``sigma``, ``g``, flat completion): the independent row agrees
        with the kernel row to ``4.4e-16`` on column 0 and ``1.3e-14`` elsewhere, and
        ``|r0_ind @ s| / max|s|`` reaches only ``4.26e-15``; bar ``1e-12``.
      * ``alpha=-1`` (the seven quark-type fields): the ``nu=0`` real value *is* the
        ``n=1`` moment, so it inherits ``test_nulled_counting_moment``'s
        ``6.32e-05`` ceiling and the same ``8.557e-04`` column-0 head discrepancy
        described there; bar ``2e-4`` on the normalized value.

    - **imaginary half** -- the old ``abs(i0) < 1e-12`` was content-free: ``sin(0*x)``
      is identically zero and the sine low-x head's leading coefficient is
      ``b = nu*x_min = 0``, so the row is the exact zero vector for *any*
      ``alpha``/``low_x_extension``/``s`` (measured: bit-for-bit ``0.0`` for both the
      real curve and a random vector). It is now asserted as what it actually is --
      an exact property of the kernel *row*, with ``assert_array_equal`` and no
      reference to ``s`` at all -- so no reader can mistake it for a statement about
      the injected curve. The imaginary component's real content near ``nu = 0`` is
      its *slope*, which ``test_imag_small_nu_slope_matches_independent_moment``
      covers against an independent oracle.
    """
    cfg, _ = suite
    basis, _ = basis_nodes
    for field in cfg.ALL_FIELDS:
        momentum_density = cfg.itd_momentum_density(field)
        alpha = -1.0 if momentum_density else 0.0
        low_x = cfg.low_x_completion(field)
        gamma = _low_x_gamma(low_x)
        real_k = kernels.PseudoITDReal(alpha=alpha, low_x_extension=low_x)
        imag_k = kernels.PseudoITDImag(alpha=alpha, low_x_extension=low_x)

        real_kernel_row = np.asarray(real_k.matrix(np.array([0.0]), basis)).reshape(-1)
        real_ind_row = independent_row(alpha, gamma)
        ratio = np.abs(real_ind_row / real_kernel_row - 1.0)
        # cols >= 1: 1.16e-14 (alpha=-1) / 1.31e-14 (alpha=0), measured.
        assert np.max(ratio[1:]) < 1e-12, f"{field} cols>=1 {np.max(ratio[1:]):.3e}"
        # col 0 is the low-x head: exact for the alpha=0 closed form (4.4e-16),
        # 8.56e-04 off for the alpha=-1 quadrature fallback -- see
        # test_nulled_counting_moment for the direct measurement of that error, and
        # for why 1e-3 is a bar on npts=32 rather than on x_min or on head accuracy.
        # Both sides here use the *same* completion, so this compares evaluations of
        # one head model and can never see the model error; that is
        # test_low_x_head_error_converges_in_x_min's job.
        head_bar = 1e-3 if momentum_density else 1e-12
        assert ratio[0] < head_bar, f"{field} col 0 {ratio[0]:.3e}"

        # The imaginary row at nu=0 is exactly zero -- a kernel identity, asserted
        # without s so it cannot read as a statement about the curve.
        imag_kernel_row = np.asarray(imag_k.matrix(np.array([0.0]), basis)).reshape(-1)
        np.testing.assert_array_equal(imag_kernel_row, np.zeros(imag_kernel_row.size))

        value_bar = 2e-4 if momentum_density else 1e-12
        for key in cfg.ITD_SYSTEMATICS:
            s = sys_curves[field][key]
            scale = float(np.max(np.abs(s)))
            r0 = _apply(real_k, 0.0, basis, s)[0]
            # Same projector-floor character as test_nulled_counting_moment's bar
            # (and, for singlet/gluon fields, bitwise the same constraint row).
            assert abs(r0) < 1e-9, f"real {field}/{key} nu=0 -> {r0}"
            independent = abs(float(real_ind_row @ s)) / scale
            assert independent < value_bar, (
                f"independent real {field}/{key} nu=0 -> {independent:.3e}"
            )


# -- the column-0 low-x head, measured as a function of x_min ------------------
#
# The two column-0 bars above (``1e-3`` in ``test_nulled_counting_moment``, and the
# ``head_bar`` in ``test_real_vanishes_at_nu0_imag_vanishes_at_nu0``) sit at a level
# that ``cfg.X_MIN`` chose for them.  They read as universal head-accuracy
# guarantees and are nothing of the kind: the head error scales as a power of
# ``x_min`` whose *exponent is set by the field class*, and the two classes in these
# suites differ by eleven orders of magnitude at the same ``x_min``.  Everything
# below states ``x_min`` explicitly and asserts the **rate**.
# Full measured diagnosis: ``plans/low_x_head_diagnosis.md``.

#: ``x_min`` ladder for :func:`test_low_x_head_error_converges_in_x_min`.  Written
#: out here and never read from ``cfg.X_MIN`` -- inheriting it is exactly how the
#: fixed bars above acquired their misleading air of universality.
LOW_X_SWEEP_X_MIN = (1e-6, 1e-7, 1e-8)

#: Grid size held fixed across the sweep, so ``x_min`` is the only thing that moves.
#: 145 is this repo's closure-grid floor (101 breaks four singlet tests) and is what
#: the diagnosis measured on.
LOW_X_SWEEP_N_POINTS = 145

#: Ioffe time of the swept pseudo-ITD row.  Deliberately **not** ``0``: at ``nu = 0``
#: the real ITD collapses onto the ``n = 1`` Mellin moment already covered by
#: ``test_nulled_counting_moment``, so ``cos(4x)`` keeps this a statement about the
#: ITD kernel.  Its ``[0, x_min)`` head still sees ``cos ~ 1``, which is why the
#: measured levels below track the Mellin ones to within a factor of order one.
LOW_X_SWEEP_NU = 4.0

#: Large-``x`` exponent of the swept truth family ``q(x) = x**a (1-x)**beta``.  Only
#: sets the overall normalization; the head sees ``(1 - x)**beta -> 1``.
LOW_X_TRUTH_BETA = 3.0

#: Small-``x`` power ``a`` of the *vanishing*-field truth, and the single number every
#: bar below is a function of.
#:
#: ``0.5`` is a **stand-in for a physical valence density**: the true small-``x``
#: power of the JAM/NNPDF closure curves is not known here, and it is the largest
#: open input in ``plans/low_x_head_diagnosis.md`` ("What is NOT determined" #1).  The
#: observable error goes as ``x_min**a`` and the per-decade rate as ``10**a``, so this
#: file writes both as functions of ``a`` rather than as literals; re-point ``a`` at
#: the measured power and the bars follow.
#:
#: It must stay **sub**linear.  At ``a = 1`` the linear head *is* the true shape, the
#: model error vanishes, and what is left is below the FE floor at ``x_min = 1e-6``
#: and *rises* under refinement (measured 6.15e-09, 1.22e-08, 1.72e-08 across this
#: ladder) -- a convergence test on a linear truth measures the floor, not the head.
LOW_X_TRUTH_POWER = 0.5

#: Error rate exponent for the non-vanishing (``sigma``/``g``) class.  A flat head
#: reproduces ``q(0) != 0`` exactly at leading order, so the whole residual is the
#: first-order mismatch, ``int_0^x_min (q(x_min) - q(x)) dx = |q'(0)| x_min**2 / 2 +
#: ...``: one power of ``x_min`` from the interval, one from the mismatch.  The
#: vanishing class instead uses ``LOW_X_TRUTH_POWER`` -- its linear head is wrong at
#: *leading* order against an ``x**a`` truth, so it only buys ``10**a`` per decade.
LOW_X_FLAT_HEAD_RATE = 2.0

#: Below this predicted level the measurement is FE-discretization and round-off, not
#: head model error, and neither its value nor its rate means anything.  Measured: the
#: ``alpha=0`` class's third rung is predicted at ``1.008e-15`` and lands on
#: ``8.882e-16`` -- four ulps of the observable.  Rungs below the floor are checked
#: only for still going down.
LOW_X_FE_FLOOR = 1e-14

#: Multiplicative band on the per-decade rate.  Measured deviations from ``10**rate``:
#: ``+0.33%`` and ``+1.91%`` (vanishing class), ``-0.29%`` (flat class, first step),
#: so 1.3 leaves >=15x headroom.  It is wide enough to accept a differently-derived
#: but still ``10**rate`` law and narrow enough to reject a neighbouring exponent:
#: the acceptance mutation (a linear head on ``sigma``/``g``) turns the flat class's
#: ``x100`` into ``x10.0000``, which misses ``[76.9, 130]`` by 7.7x.
LOW_X_RATE_BAND = 1.3

#: Relative band on the *level* against the closed-form head-model error.  Measured
#: worst ``|err/pred - 1|`` over both classes and every resolved rung: ``2.17%``
#: (the ``x_min = 1e-8`` vanishing rung, where FE discretization starts to show), so
#: 0.10 leaves 4.6x headroom.
LOW_X_LEVEL_BAND = 0.10

#: Working precision for the mpmath oracles below.
LOW_X_MP_DPS = 40


def _mp_itd(nu, alpha, a, lo, hi):
    r"""``int_lo^hi cos(nu x) x**alpha x**a (1-x)**LOW_X_TRUTH_BETA dx``, 40 digits.

    The oracle for the sweep, and independent of everything it measures: 40-digit
    ``mpmath`` tanh-sinh quadrature of the *whole* integral down to a true ``x = 0``,
    with no basis, no finite-element grid, no ``pixel`` import and -- the point --
    no low-x completion at all.  Nothing here shares code, quadrature rule or
    variable with ``pixel.kernels.lowx``.  Oracle ``B1``.
    """
    with mpmath.workdps(LOW_X_MP_DPS):
        power = mpmath.mpf(alpha) + mpmath.mpf(a)
        beta = mpmath.mpf(LOW_X_TRUTH_BETA)

        def integrand(x):
            """The weighted truth ``cos(nu x) x**(alpha+a) (1-x)**beta`` at ``x``."""
            return mpmath.cos(mpmath.mpf(nu) * x) * x ** power * (1 - x) ** beta

        lo, hi = mpmath.mpf(lo), mpmath.mpf(hi)
        # Split the panel where the interval spans decades, so the tanh-sinh rule
        # never has to resolve the endpoint singularity and the bulk at once.
        interior = []
        if hi > lo * 10:
            interior = [lo + (hi - lo) * mpmath.mpf(f) for f in (0.25, 0.5)]
        return float(mpmath.quad(integrand, [lo, *interior, hi]))


def _predicted_low_x_head_error(nu, alpha, gamma, a, x_min, observable):
    r"""``(the head PIXEL models) - (the head the truth has)``, over ``observable``.

    PIXEL continues the field below ``x_min`` as ``q(x_min) (x / x_min)**gamma``, so
    its ``[0, x_min)`` contribution -- evaluated *exactly*, i.e. with no quadrature
    error of its own -- is ``q(x_min) int_0^x_min cos(nu x) x**alpha (x/x_min)**gamma
    dx``.  The truth's own contribution over the same interval is ``int_0^x_min
    cos(nu x) x**alpha q(x) dx``.  Their difference is the **model** error: the
    completion shape not being the true shape.

    This is the prediction the level assertion compares against, and it is worth
    saying why it is allowed to *be* the whole answer.  The competing term is
    PIXEL's Gauss-Jacobi evaluation error on that same head, in closed form
    ``1/(npts+1)**2`` = ``9.18e-04`` relative at the shipped ``npts = 32``.  For any
    sublinear field the model error beats it by ~1000x
    (``plans/low_x_head_diagnosis.md``), which is exactly what the ``<= 2.2%``
    measured agreement below confirms -- and is the finding that makes lowering
    ``x_min``, not refining the head, the only lever these bars respond to.

    Both integrals at 40 digits in ``mpmath``; oracle ``A1``/``B1``.
    """
    with mpmath.workdps(LOW_X_MP_DPS):
        xm = mpmath.mpf(x_min)
        beta = mpmath.mpf(LOW_X_TRUTH_BETA)
        q_xmin = xm ** mpmath.mpf(a) * (1 - xm) ** beta

        def modelled(x):
            """PIXEL's continued field under the kernel weight, at ``x``."""
            return (mpmath.cos(mpmath.mpf(nu) * x) * x ** mpmath.mpf(alpha)
                    * (x / xm) ** mpmath.mpf(gamma))

        def actual(x):
            """The true field under the same weight, at ``x``."""
            return (mpmath.cos(mpmath.mpf(nu) * x) * x ** mpmath.mpf(alpha)
                    * x ** mpmath.mpf(a) * (1 - x) ** beta)

        head_model = q_xmin * mpmath.quad(modelled, [0, xm])
        head_truth = mpmath.quad(actual, [0, xm])
        return float((head_model - head_truth) / mpmath.mpf(observable))


def test_low_x_head_error_converges_in_x_min(suite):
    """The column-0 head error falls at ``10**rate`` per decade of ``x_min``, with
    ``rate`` set by the field class -- and it is the *model* error, not a tolerance.

    This is the test the two fixed column-0 bars above are not.  ``1e-3`` and
    ``head_bar`` are what ``cfg.X_MIN = 1e-6`` happens to buy on those fixtures; the
    honest statement is a rate, because the head error is a power of ``x_min`` and
    the two production field classes carry *different powers*:

    * ``alpha = -1`` (the seven quark-type momentum densities, linear head): the head
      continues ``q`` as ``q(x_min) x / x_min`` while a physical valence density goes
      as ``x**a`` with ``a < 1``, so the completion is wrong at **leading** order and
      the observable error is ``(1/a - 1) q(x_min) / observable`` -- ``x_min**a``, i.e.
      only ``10**a`` per decade.  Measured on ``q = x**0.5 (1-x)**3`` at ``nu = 4``:
      ``1.356e-03``, ``4.273e-04``, ``1.326e-04``; ratios ``3.172``, ``3.222`` against
      ``10**0.5 = 3.162``.  **Lowering ``x_min`` is a weak lever here** -- four
      decades buy two.
    * ``alpha = 0`` (``sigma``, ``g``, flat head): the flat head reproduces
      ``q(0) != 0`` exactly at leading order, leaving only the first-order mismatch,
      so the error is ``x_min**2`` -- ``x100`` per decade.  Measured on ``q =
      (1-x)**3``: ``1.008e-11``, ``1.011e-13``, ``8.882e-16``; first-step ratio
      ``99.7``.  Eleven orders of magnitude better than the other class **at the same
      x_min**, which is precisely what a single fixed bar hides.

    Three assertions, and what each one can catch:

    1. **Integrability.**  ``alpha + gamma > -1``, the condition for the
       ``[0, x_min)`` integral to exist at all -- ``base_matrix`` bakes ``x**alpha``
       into the kernel, so the head integrand is ``x**(alpha+gamma)`` at the origin.
       Asserted on the *config* here, and stated as a physical requirement rather than
       delegated: ``LowXExtension`` validates ``effective_power > -1`` without ever
       seeing ``alpha``, and until ``check_low_x_integrable``
       (``pixel.kernels.lowx``) landed on 2026-08-13 nothing rejected the divergent
       pairing at all -- it returned a finite number growing with ``npts``.  Failing
       *here* names the offending field class and completion; letting the kernel raise
       instead would surface as an assembly error several frames down, from whichever
       test happened to build a matrix first.  See
       ``tests/test_kernel_guards.py::test_low_x_integrability_guard_rejects_a_divergent_pairing``
       for the kernel-side guard, the helper it still leaves unguarded, and its
       measured over-rejection of convergent Mellin orders.
    2. **Level**, against the closed-form model error of the completion the config
       actually declares (:func:`_predicted_low_x_head_error`, 40-digit ``mpmath``).
       Measured ``|err/pred - 1|``: ``0.04%``/``0.29%``/``2.17%`` (vanishing class)
       and ``0.002%``/``0.29%`` (flat class); bar ``10%``.  This is what says the
       observed error *is* the head-model error and not something else of a similar
       size, and it is what would fail if ``low_x_quadrature_correction`` stopped
       evaluating the head it advertises.
    3. **Rate**, ``10**rate`` per decade within :data:`LOW_X_RATE_BAND`, on the rungs
       whose *predicted* level clears :data:`LOW_X_FE_FLOOR`.  That floor rule is what
       drops the flat class's third rung (predicted ``1.008e-15``, four ulps of the
       observable) automatically rather than by a hardcoded index.

    Acceptance, three in-memory mutations measured 2026-08-13, never touching
    ``src/`` on disk -- and each caught by a *different* one of the three assertions,
    which is why all three earn their place:

    * a **linear head on sigma/g** puts the flat class at ``3.362e-06`` -- ``3.3e5x``
      the correct level -- and turns its ``x100`` into ``x10.0000``, failing
      assertion 3.  Assertion 2 does **not** fire here: the prediction follows the
      completion the config declares, so it tracks the mutation to
      ``got/pred = 1.000000``.  The rate is the only discriminator.
    * a **flat head on the vanishing class** makes ``alpha + gamma = -1`` exactly and
      fails assertion 1.  Neither of the others sees it: its error still falls at
      ``x3.16`` per decade (measured ``3.1606``, ``3.1528``), only 6.1x too high.
    * **inflating ``low_x_quadrature_correction`` by 50%** (an ``src``-side head bug,
      scoped to the ``alpha != 0`` path that uses it) fails assertion 2 at
      ``got/pred = 0.5008`` while leaving the rate at ``3.1831``/``3.2860``, inside
      the band -- so among *this test's* three assertions only the closed-form level
      comparison catches it.

    The file's *pre-existing* tests do also fail under all three, and it is worth
    being exact about why, because none of those failures is a statement about the
    low-x head. Mutation 1 trips
    ``test_real_vanishes_at_nu0_imag_vanishes_at_nu0``'s projector-floor bar
    (``5.086e-09`` vs ``1e-09``) because ``build_systematic_curves`` nulls against
    ``generate.py``'s *hardcoded* flat ``itd_zero`` row -- what is detected is a
    config inconsistency. Mutation 2 makes ``_independent_power_moment_row``'s own
    closed-form head divide by zero (``weight + 1 + gamma = 0``) and surfaces as a
    bare ``ZeroDivisionError`` inside a helper rather than as a diagnosed finding --
    the oracle encodes the same integrability condition ``src/`` does not, and blows
    up on it. Mutation 3 trips the column-0 quadrature bars. What none of them
    asserts, before this test, is the convergence claim: the rate, the level, and
    which class an error belongs to.

    Not covered: this says nothing about which ``a`` the real closure curves have
    (see :data:`LOW_X_TRUTH_POWER`), and it deliberately does not run a closure fit --
    the head error cancels there, because the same operator writes the target and
    reads it back.
    """
    cfg, _ = suite

    classes: dict[tuple[float, float, bool], list[str]] = {}
    for field in cfg.ALL_FIELDS:
        alpha = -1.0 if cfg.itd_momentum_density(field) else 0.0
        gamma = _low_x_gamma(cfg.low_x_completion(field))
        key = (alpha, gamma, bool(cfg.vanishes_at_origin(field)))
        classes.setdefault(key, []).append(field)
    # Both production classes must be present, or the sweep below silently measures
    # one of them twice.  Measured: 7 fields at (-1, 1, True), 2 at (0, 0, False).
    assert len(classes) == 2, f"expected two low-x classes, got {sorted(classes)}"

    for (alpha, gamma, vanishing), members in sorted(classes.items()):
        label = f"alpha={alpha:+.0f} gamma={gamma:+.0f} [{','.join(members)}]"
        # Assertion 1 -- integrability, named here on the config rather than left to
        # surface as a kernel-assembly error several frames down.
        assert alpha + gamma > -1.0, (
            f"{label}: divergent low-x pairing. The head integrand is "
            f"x**{alpha + gamma:+.1f} at the origin, so int_0^x_min diverges. "
            "LowXExtension's own check (effective_power > -1) never sees alpha; "
            "kernels.lowx.check_low_x_integrable is the kernel-side guard -- see "
            "tests/test_kernel_guards.py::"
            "test_low_x_integrability_guard_rejects_a_divergent_pairing"
        )

        a = LOW_X_TRUTH_POWER if vanishing else 0.0
        rate = a if vanishing else LOW_X_FLAT_HEAD_RATE
        observable = _mp_itd(LOW_X_SWEEP_NU, alpha, a, 0.0, 1.0)
        assert abs(observable) > 1e-3, f"{label}: degenerate observable {observable}"

        errors, predicted = [], []
        for x_min in LOW_X_SWEEP_X_MIN:
            grid = Grid(
                n_points=LOW_X_SWEEP_N_POINTS,
                spacing=cfg.GRID_SPACING,
                x_min=x_min,
            )
            fld = Field.create(members[0], grid, element_type=cfg.ELEMENT_TYPE)
            nodes = np.asarray(fld.nodes, dtype=float).reshape(-1)
            # The head is an integral over [0, basis.domain[0]); if that is not the
            # x_min asked for, the sweep is not sweeping what it says it is.
            np.testing.assert_allclose(
                float(fld.basis.domain[0]), x_min, rtol=1e-15, atol=0.0
            )
            curve = nodes ** a * (1.0 - nodes) ** LOW_X_TRUTH_BETA
            kernel = kernels.PseudoITDReal(
                alpha=alpha, low_x_extension=cfg.low_x_completion(members[0])
            )
            row = np.asarray(
                kernel.matrix(np.array([LOW_X_SWEEP_NU]), fld.basis), dtype=float
            ).reshape(-1)
            errors.append(abs(float(row @ curve) / observable - 1.0))
            predicted.append(abs(_predicted_low_x_head_error(
                LOW_X_SWEEP_NU, alpha, gamma, a, x_min, observable
            )))

        resolved = [p > LOW_X_FE_FLOOR for p in predicted]
        assert resolved[0] and resolved[1], (
            f"{label}: sweep starts at the FE floor, predicted {predicted}"
        )
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"{label}: lowering x_min made it worse -- {errors}"
            )
        # Assertion 2 -- the measured error IS the declared head's model error.
        for x_min, err, pred, ok in zip(
            LOW_X_SWEEP_X_MIN, errors, predicted, resolved
        ):
            if not ok:
                continue
            assert abs(err / pred - 1.0) < LOW_X_LEVEL_BAND, (
                f"{label} x_min={x_min:.0e}: measured {err:.6e} vs predicted "
                f"head-model error {pred:.6e} (ratio {err / pred:.6f})"
            )
        # Assertion 3 -- the rate, on the rungs above the floor.
        expected = 10.0 ** rate
        for i in range(len(errors) - 1):
            if not (resolved[i] and resolved[i + 1]):
                continue
            ratio = errors[i] / errors[i + 1]
            assert expected / LOW_X_RATE_BAND < ratio < expected * LOW_X_RATE_BAND, (
                f"{label}: {LOW_X_SWEEP_X_MIN[i]:.0e} -> "
                f"{LOW_X_SWEEP_X_MIN[i + 1]:.0e} gained x{ratio:.4f}, expected "
                f"x{expected:.4f} (10**{rate}) within x{LOW_X_RATE_BAND}"
            )


def test_imag_small_nu_slope_matches_independent_moment(
    suite, basis_nodes, sys_curves, independent_row
):
    """``Imag(nu)/nu -> int x^(alpha+1) f dx`` as ``nu -> 0``, against an independent
    quadrature.

    The imaginary component's value *at* ``nu = 0`` is an exact zero of the kernel
    row and says nothing about the curve (see
    ``test_real_vanishes_at_nu0_imag_vanishes_at_nu0``). Its slope there does: from
    ``sin(nu x) = nu x + O(nu^3)``, ``Imag(nu)/nu`` tends to the ``x^(alpha+1)``
    moment of the field, low-x completion included. That limit is computed here by
    ``_independent_power_moment_row(alpha + 1, gamma)`` -- a different quadrature in
    ``u = log x`` with a closed-form ``[0, x_min)`` head -- so this is the first
    assertion in the file to pin the ``Sine`` kernel's small-``nu`` behaviour against
    something outside ``pixel.kernels``.

    ``nu = 1e-4`` is small enough that the ``O(nu^2)`` truncation of the limit is
    below the bar and large enough to stay clear of float cancellation: measured
    2026-08-13, ``max |slope/independent - 1| = 1.57e-09`` across all 45 (field, key)
    pairs of both suites, so the bar is ``1e-7`` (64x headroom). The comparison is on
    the ratio, not the difference, precisely because the slopes span several decades
    across fields.

    Non-degeneracy: the independent moment itself is asserted to be well away from
    zero (measured minimum ``|slope| / max|s|`` is ``5.12e-02``, bar ``1e-3``),
    because a ratio bar is vacuous where both sides are ~0 -- which is exactly what
    the *real* component is at ``nu = 0``, and why the same trick cannot be reused
    there.
    """
    cfg, _ = suite
    basis, _ = basis_nodes
    nu_small = 1e-4
    for field in cfg.ALL_FIELDS:
        alpha = -1.0 if cfg.itd_momentum_density(field) else 0.0
        low_x = cfg.low_x_completion(field)
        imag_k = kernels.PseudoITDImag(alpha=alpha, low_x_extension=low_x)
        slope_row = np.asarray(
            imag_k.matrix(np.array([nu_small]), basis)
        ).reshape(-1) / nu_small
        ind_row = independent_row(alpha + 1.0, _low_x_gamma(low_x))
        for key in cfg.ITD_SYSTEMATICS:
            s = sys_curves[field][key]
            scale = float(np.max(np.abs(s)))
            expected = float(ind_row @ s)
            assert abs(expected) / scale > 1e-3, (
                f"{field}/{key} slope oracle is ~0 ({expected:.3e}); ratio bar vacuous"
            )
            achieved = float(slope_row @ s)
            assert abs(achieved / expected - 1.0) < 1e-7, (
                f"{field}/{key} slope {achieved:.6e} vs independent {expected:.6e}"
            )


#: Decay bound for ``test_converges_at_large_nu``, per ITD ``alpha`` class.  A single
#: shared 0.7 was ~300x looser than the achieved ratio for the ``alpha=0`` fields, so
#: it could only have caught a kernel that does not decay in ``nu`` at all there.
#: Measured 2026-08-13 over all 45 (field, key) pairs of *both* suites (identical, as
#: the two packages share a seed and a construction):
#:   alpha=-1: 35 pairs, ratios in [1.4518e-01, 6.0700e-01] -> bar 0.7 (87% used)
#:   alpha= 0: 10 pairs, ratios in [1.2365e-03, 2.5946e-03] -> bar 5e-3 (52% used)
DECAY_BARS = {-1.0: 0.7, 0.0: 5e-3}


def test_converges_at_large_nu(suite, basis_nodes, sys_curves):
    """Tail (``nu=300``) below an ``alpha``-class bound times the peak (``nu`` in
    ``[1,40]``) -- Riemann-Lebesgue decay.

    Unlike the parity/nu=0 tests above, this one genuinely evaluates the real
    ``sys_curves`` content through the real per-field kernel, so it is not
    construction-invariant. The bound is now split by ``alpha`` class (see
    ``DECAY_BARS`` for the measured ranges behind each value) because the single
    shared ``0.7`` was doing nothing for the ``alpha=0`` fields: their achieved
    ratios sit at ``1.2e-3``-``2.6e-3``, ~300x below it. At ``5e-3`` the ``alpha=0``
    fields now use 52% of their bar and the ``alpha=-1`` fields 87% of theirs, so a
    broken decay in either branch -- a dropped low-x head, a lost oscillation -- has
    to move the ratio by less than a factor of 2 to stay hidden, rather than by a
    factor of 300.

    The bars are one-directional by nature (decay is an inequality), so the two
    guards below supply the other direction: ``peak`` must be a real signal rather
    than a near-zero that makes any tail "small", and the achieved ratio must not
    collapse far *under* its class bar, which would mean the measurement behind
    ``DECAY_BARS`` has gone stale and the bar should be retightened rather than left
    as decoration.
    """
    cfg, _ = suite
    # The cosine transform of an integrable f decays toward large nu (property 2).
    # It is not monotonic from nu=0 (it rises to a peak first), so compare the
    # peak over a moderate band to the far tail.
    basis, _ = basis_nodes
    moderate = np.linspace(1.0, 40.0, 40)
    seen = {-1.0: 0, 0.0: 0}
    for field in cfg.ALL_FIELDS:
        alpha = -1.0 if cfg.itd_momentum_density(field) else 0.0
        low_x = cfg.low_x_completion(field)
        kernel_cls = (
            kernels.PseudoITDReal
            if cfg.field_cparity(field) == "even"
            else kernels.PseudoITDImag
        )
        kernel = kernel_cls(alpha=alpha, low_x_extension=low_x)
        B_moderate = np.asarray(kernel.matrix(moderate, basis))
        B_tail = np.asarray(kernel.matrix(np.array([300.0]), basis))
        bar = DECAY_BARS[alpha]
        for key in cfg.ITD_SYSTEMATICS:
            s = sys_curves[field][key]
            peak = float(np.max(np.abs(B_moderate @ s)))
            tail = abs(float((B_tail @ s)[0]))
            seen[alpha] += 1
            # Non-degeneracy: a vanishing peak would make the ratio bound vacuous.
            # Measured minimum peak over both suites: 3.62e-03.
            assert peak > 1e-3, f"{field}/{key}: degenerate peak {peak:.3e}"
            assert tail < bar * peak, (
                f"{field}/{key} (alpha={alpha:+.0f}): tail/peak {tail / peak:.4e} "
                f"vs bar {bar}"
            )
            # Bar-is-still-honest guard: 0.145/0.7 = 21% and 1.24e-3/5e-3 = 25% are
            # the measured minima, so 5% flags a bar that has drifted loose.
            assert tail > 0.05 * bar * peak, (
                f"{field}/{key} (alpha={alpha:+.0f}): tail/peak {tail / peak:.4e} is "
                f"far under bar {bar} -- retighten DECAY_BARS, do not leave it loose"
            )
    assert seen == {-1.0: 35, 0.0: 10}, seen


#: Replica count of both synthetic lattice ensembles below.  It is < the row count
#: of every pseudo-ITD record (smallest is 15 = z_max 3 * p_max 5), which is the
#: *emergent* reason the folded covariance is singular -- see
#: ``test_actual_folded_covariance_blocks_become_spd``.
N_SYNTHETIC_REPLICAS = 6


def _lattice_operators(cfg, gen, basis):
    """Real pseudo-ITD ``(record, [(field, B)])`` pairs for all 13 ensembles."""
    records = [rec for rec in gen.lattice_layout() if rec["kind"] == "pseudoitd"]
    operators = []
    for rec in records:
        meta = gen.lattice_meta(rec)
        alpha = -1.0 if rec["momentum_density"] else 0.0
        low_x = cfg.low_x_completion(rec["field"])
        kernel_cls = (
            kernels.PseudoITDReal
            if rec["component"] == "real"
            else kernels.PseudoITDImag
        )
        B = np.asarray(
            kernel_cls(alpha=alpha, low_x_extension=low_x).matrix(meta["nu"], basis)
        )
        operators.append([(rec["field"], B)])
    return records, operators


@pytest.fixture(scope="module")
def actual_lattice_folds(suite, basis_nodes, sys_curves):
    """Real pseudo-ITD kernel operators (all 13 ensembles) folded via the real
    :func:`~closure_JAM_truth.generate.fold_lattice_systematics`.

    The synthetic ``ensemble`` built here is ``(index+1) * replica_scale[:,None]
    * shape[None,:]`` -- every one of the 6 replica rows is a scalar multiple of
    the *same* ``shape`` vector, i.e. exactly rank 1 by construction (measured
    2026-08-13: ``np.linalg.matrix_rank(ensemble["t3"]) == 1`` of 6 possible).
    Retained because its folded central value is readable by inspection, but it is
    **not** the fixture that establishes rank deficiency: see
    ``nondegenerate_lattice_folds`` for the full-rank companion, which
    ``test_actual_folded_covariance_blocks_become_spd`` runs alongside this one.
    """
    cfg, gen = suite
    basis, nodes = basis_nodes
    records, operators = _lattice_operators(cfg, gen, basis)

    shape = nodes**0.4 * (1.0 - nodes) ** 3
    replica_scale = np.linspace(0.8, 1.2, N_SYNTHETIC_REPLICAS)
    ensemble = {
        field: (index + 1.0) * replica_scale[:, None] * shape[None, :]
        for index, field in enumerate(cfg.ALL_FIELDS)
    }
    return records, operators, ensemble, gen.fold_lattice_systematics(
        records, operators, ensemble, sys_curves
    )


@pytest.fixture(scope="module")
def nondegenerate_lattice_folds(suite, basis_nodes, sys_curves):
    """The same fold, on a replica ensemble that is **full rank** by construction.

    ``actual_lattice_folds``'s replicas are all scalar multiples of one shape, so any
    linear fold of them is rank <= 1 whatever the fold does -- which made
    "the raw fold covariance is singular" a restatement of the fixture rather than a
    property of the fold. Here each replica gets its own i.i.d. perturbation
    (seeded, so the fixture is deterministic), giving
    ``matrix_rank(ensemble[field]) == 6`` of 6 (measured 2026-08-13). The folded
    covariance is *still* singular, now for the emergent reason that a covariance of
    6 replicas has rank at most 5 while every record has >= 15 rows: measured ranks
    3-5 across the 117 blocks, versus 1 for every block of the rank-1 fixture.
    """
    cfg, gen = suite
    basis, nodes = basis_nodes
    records, operators = _lattice_operators(cfg, gen, basis)

    shape = nodes**0.4 * (1.0 - nodes) ** 3
    replica_scale = np.linspace(0.8, 1.2, N_SYNTHETIC_REPLICAS)
    rng = np.random.default_rng(9091)
    ensemble = {}
    for index, field in enumerate(cfg.ALL_FIELDS):
        base = (index + 1.0) * replica_scale[:, None] * shape[None, :]
        jitter = 0.15 * rng.standard_normal(
            (N_SYNTHETIC_REPLICAS, nodes.size)
        ) * shape[None, :]
        ensemble[field] = base + jitter
    return records, operators, ensemble, gen.fold_lattice_systematics(
        records, operators, ensemble, sys_curves
    )


def test_generation_fold_enforces_cap_for_every_field_key(
    suite, sys_curves, actual_lattice_folds
):
    """Every ``(field, key)`` pair is present and saturates the 10% cap -- with the
    shift and the signal rebuilt from the fixture, not read back out of the fold.

    The old form recomputed ``observed`` entirely from ``folds.scale``,
    ``folds.raw_shift`` and ``folds.central``, i.e. from three outputs of the
    function under test, using the same ``scale = max_fraction / max_ratio``
    relation that produced ``folds.scale``. That is an algebraic identity, and it was
    measured to be one 2026-08-13: substituting ``1000*N(0,1)`` unnulled noise for
    the real ``sys_curves`` (true ``n=1`` moment ``4.9e2``, vs ``~1e-14`` for the real
    curves) gave a **bit-for-bit identical** recomputed residual,
    ``max|observed-cap| = 1.388e-17``.

    Now only ``folds.scale`` comes from the fold. ``raw`` is rebuilt as
    ``coefficient(key, lattice_meta(rec)) * (B @ sys_curves[field][key])`` from the
    fixture's own operator and the curves the fixture handed in, and ``central`` as
    ``(ensemble[field] @ B.T).mean(axis=0)`` in plain numpy -- so the cap statement
    becomes "the *actual* scaled shift is 10% of the *actual* folded signal", which a
    fold applying the wrong coefficient, the wrong field's curve, or the wrong
    operator would break.

    Three bars, all measured 2026-08-13 over the 117 records / 45 pairs of both suites:

    * ``folds.raw_shift`` and ``folds.central`` reproduce the independent rebuild
      **bit-exactly** (``rtol=0, atol=0`` passes), so they are asserted with
      ``assert_array_equal``. Both sides are the same numpy expressions on the same
      arrays, so this is determinism, not luck.
    * the independently-recomputed cap ratio hits ``SYSTEMATIC_MAX_FRACTION`` to
      ``2.22e-16`` relative; bar ``rtol=1e-10``, kept from the old form.
    * ``folds.curves == sys_curves * scale`` to ``0.0`` absolute.

    The coverage check (``set(observed) == expected``) is unchanged and was always
    real: it catches a field or key silently dropped from the layout/systematics
    wiring.
    """
    cfg, gen = suite
    records, operators, ensemble, folds = actual_lattice_folds
    observed = {}
    for index, (rec, operator) in enumerate(zip(records, operators)):
        field, B = operator[0]
        meta = gen.lattice_meta(rec)
        # Independent of folds: the fixture's own ensemble through the fixture's own
        # operator, and the fixture's own curves through the registry coefficient.
        central = (ensemble[field] @ B.T).mean(axis=0)
        np.testing.assert_array_equal(central, folds.central[index])
        denom = np.where(np.abs(central) > 0.0, np.abs(central), 1.0)
        assert set(folds.raw_shift[index]) == set(rec["systematics"])
        for key in rec["systematics"]:
            raw = gen.coefficient(key, meta) * (B @ sys_curves[field][key])
            np.testing.assert_array_equal(raw, folds.raw_shift[index][key])
            ratio = np.max(np.abs(folds.scale[(field, key)] * raw) / denom)
            pair = (field, key)
            observed[pair] = max(observed.get(pair, 0.0), float(ratio))

    expected = {
        (field, key)
        for field in cfg.ALL_FIELDS
        for key in cfg.ITD_SYSTEMATICS
    }
    assert set(observed) == expected
    assert all(value <= cfg.SYSTEMATIC_MAX_FRACTION * (1.0 + 1e-12)
               for value in observed.values())
    assert all(np.isclose(value, cfg.SYSTEMATIC_MAX_FRACTION, rtol=1e-10)
               for value in observed.values())
    # The scaled curves the caller actually writes out must carry the same scale.
    for field in cfg.ALL_FIELDS:
        for key in cfg.ITD_SYSTEMATICS:
            np.testing.assert_array_equal(
                folds.curves[field][key],
                sys_curves[field][key] * folds.scale[(field, key)],
            )


@pytest.mark.parametrize(
    "fixture_name", ["actual_lattice_folds", "nondegenerate_lattice_folds"]
)
def test_actual_folded_covariance_blocks_become_spd(suite, fixture_name, request):
    """Raw fold covariance is singular; ``inflate_diagonal`` at 1% makes it SPD.

    Run on **two** replica ensembles, because on the original one alone the premise
    was a restatement of the fixture: ``actual_lattice_folds`` builds every replica
    as a scalar multiple of one shape vector, so any single-field linear fold of it
    is rank <= 1 no matter what the fold does (measured 2026-08-13:
    ``matrix_rank(ensemble["t3"]) == 1`` of 6, and every folded block came back rank
    1). ``nondegenerate_lattice_folds`` gives each replica its own i.i.d.
    perturbation, so ``matrix_rank(ensemble[field]) == 6`` of 6 and the folded blocks
    come back rank 3-5 -- still singular, but now for the reason that generalizes:
    a covariance of ``n_rep`` replicas has rank at most ``n_rep - 1``, and every
    pseudo-ITD record here has at least 15 rows.

    Four things are now asserted about each of the 117 covariance blocks, where
    before there was only ``matrix_rank(cov) < cov.shape[0]``:

    * **shape** ``== (n_rows, n_rows)`` for that record's own operator. The old test
      never looked, and ``np.cov(Y, rowvar=True)`` -- a one-word slip -- returns a
      ``6 x 6`` replica-space covariance that is rank 1, inflates to SPD and passes
      every one of the old assertions. Measured 2026-08-13: the old form passes under
      exactly that mutation.
    * **rank bounds**, per fixture. The rank-1 toy gives exactly 1; the full-rank
      ensemble gives 3-5 (measured), bounded above by ``n_rep - 1 = 5`` structurally
      and asserted to be at least 2 -- which is what fails if the fold ever stops
      using the replica spread (taking a mean, or dropping replicas, before
      ``np.cov``).
    * **Cholesky succeeds after 1% inflation** -- the substantive claim, real on both
      fixtures.
    * **the diagonal formula**, a closed-form recomputation independent of calling
      ``inflate_diagonal`` again. Its bar was numpy's implicit ``atol=1e-8``, which
      sat only ~1.2% below the *smallest* diagonal entry in the fold (measured
      ``min diag(cov) = 8.27e-07``, max ``3.10e+00``) and so was a real, undeclared
      floor. Now an explicit ``rtol=1e-13, atol=0``; achieved ``2.22e-16``, 0.2% of
      the bar.
    """
    cfg, gen = suite
    _, operators, ensemble, folds = request.getfixturevalue(fixture_name)
    degenerate = fixture_name == "actual_lattice_folds"
    expected_rank = 1 if degenerate else N_SYNTHETIC_REPLICAS
    for values in ensemble.values():
        assert np.linalg.matrix_rank(values) == expected_rank, fixture_name
    assert len(folds.covariance) == len(operators)
    for operator, cov in zip(operators, folds.covariance):
        n_rows = operator[0][1].shape[0]
        assert n_rows >= N_SYNTHETIC_REPLICAS
        assert cov.shape == (n_rows, n_rows)
        rank = np.linalg.matrix_rank(cov)
        assert rank < cov.shape[0]
        assert rank <= N_SYNTHETIC_REPLICAS - 1
        # The rank-1 toy pins its own construction; the full-rank ensemble must
        # retain genuine replica spread through the fold (measured 3-5 of 5).
        if degenerate:
            assert rank == 1
        else:
            assert rank >= 2
        reg = gen.inflate_diagonal(cov, cfg.COV_DIAGONAL_INFLATE)
        np.linalg.cholesky(reg)
        np.testing.assert_allclose(
            np.diag(reg),
            np.diag(cov) * (1.0 + cfg.COV_DIAGONAL_INFLATE),
            rtol=1e-13,
            atol=0.0,
        )


def test_seeded_systematic_curves_match_between_truth_families():
    """Truth choice must not change the injected nuisance realization.

    ``closure_JAM_truth`` and ``closure_NNPDF_truth`` are physically duplicated
    packages (confirmed by diff: ``generate.py``'s systematic-curve code is
    identical between the two, differing only in one comment header), each with
    its own ``SYSTEMATIC_SEED = 20260721``. This calls both packages'
    ``build_systematic_curves`` independently and checks they agree to
    ``atol=1e-14``. Real, if narrow: it would catch one twin package's
    systematic-curve construction (RNG draw order, pheno shape parameters)
    drifting from the other's, but not a bug present identically in both copies
    of the shared algorithm.
    """
    jam_fields = jam_cfg.make_fields()
    nnpdf_fields = nnpdf_cfg.make_fields()
    jam_basis = next(iter(jam_fields.values())).basis
    nnpdf_basis = next(iter(nnpdf_fields.values())).basis
    jam_nodes = np.asarray(next(iter(jam_fields.values())).nodes).reshape(-1)
    nnpdf_nodes = np.asarray(next(iter(nnpdf_fields.values())).nodes).reshape(-1)
    assert np.array_equal(jam_nodes, nnpdf_nodes)

    jam_curves = jam_gen.build_systematic_curves(
        jam_nodes,
        jam_basis,
        np.random.default_rng(jam_cfg.SYSTEMATIC_SEED),
    )
    nnpdf_curves = nnpdf_gen.build_systematic_curves(
        nnpdf_nodes,
        nnpdf_basis,
        np.random.default_rng(nnpdf_cfg.SYSTEMATIC_SEED),
    )
    for field in jam_cfg.ALL_FIELDS:
        for key in jam_cfg.ITD_SYSTEMATICS:
            assert np.allclose(
                jam_curves[field][key],
                nnpdf_curves[field][key],
                rtol=0.0,
                atol=1e-14,
            )


def test_mellin_systematics_use_distinct_one_point_fields_and_unit(suite):
    """Each Mellin nuisance is its own one-point ``Field``, wired through ``Unit``.

    Builds a real one-point ``Field`` (``cfg.make_moment_nuisance_grid`` ->
    ``ProductGrid``) and calls the real
    ``<suite>.datasets._moment_systematic_contributions`` (not a shim -- the
    concrete implementation lives in ``closure_JAM_truth/datasets.py`` and its
    ``closure_NNPDF_truth`` twin), then checks the returned contribution's field
    name, kernel type, and both the all-ones ``base_matrix`` and the coefficient
    -weighted ``matrix``. The wiring checks are real: a wrong key's factory
    selected, a wrong ``meta`` threaded through, a wrong contribution count/type,
    or a broken row-scaling in ``Unit``/``Kernel.apply_coefficient`` would all be
    caught. The final numeric equality is not independent of the code under
    test, though: ``_moment_systematic_contributions`` builds
    ``kernels.Unit(coefficient=factory(dataset.meta), ...)`` where ``factory =
    kernels.LATTICE_SYSTEMATICS[key][1]``, and the oracle here,
    ``gen.coefficient(key, meta)``, calls that *same* factory with the *same*
    ``meta`` object -- so a wrong ``LATTICE_SYSTEMATICS[key]`` formula would be
    invisible to this specific assertion. That formula-level physics is
    independently covered by ``tests/test_coefficients.py`` (e.g.
    ``test_chiral_factory_is_constant_vector``), not by this file.
    """
    cfg, gen = suite
    dsets = importlib.import_module(f"{cfg.__package__}.datasets")
    specs = cfg.moment_nuisance_specs()
    assert len(specs) == len(cfg.ALL_FIELDS) * len(cfg.MOMENT_SYSTEMATICS)

    names = []
    for field, key, order in specs:
        name = cfg.moment_nuisance_field_name(field, key, order)
        names.append(name)
        assert name != cfg.nuisance_field_name(field, key)

        scalar = Field.create(
            name,
            cfg.make_moment_nuisance_grid(order),
            element_type="piecewise",
        )
        assert scalar.n_grid == 1
        assert np.asarray(scalar.nodes).reshape(-1).tolist() == [float(order)]

        rec = {
            "kind": "mellin",
            "field": field,
            "order": order,
            "systematics": [key],
        }
        meta = {
            "nu": np.array([float(order)]),
            "n": np.array([float(order)]),
            "L": 48.0,
            "a": 0.09,
            "m_pi": 0.22,
        }
        contributions = dsets._moment_systematic_contributions(
            rec,
            SimpleNamespace(meta=meta),
            {name: scalar},
            enabled=True,
        )
        assert len(contributions) == 1
        assert contributions[0].field == name
        assert isinstance(contributions[0].kernel, kernels.Unit)
        base = np.asarray(
            contributions[0].kernel.base_matrix([order], scalar.basis)
        )
        matrix = np.asarray(
            contributions[0].kernel.matrix([order], scalar.basis)
        )
        np.testing.assert_allclose(base, np.ones((1, 1)))
        # gen.coefficient calls the same kernels.LATTICE_SYSTEMATICS[key][1]
        # factory, with the same meta, that _moment_systematic_contributions
        # used to build the kernel -- so this checks the wiring/row-scaling, not
        # the factory formula (see the module-level test_coefficients.py suite).
        np.testing.assert_allclose(
            matrix,
            gen.coefficient(key, meta).reshape(-1, 1),
        )

    assert len(names) == len(set(names))


def test_mellin_systematic_truths_are_independent_and_deterministic_per_order():
    """Per ``(field, key, Mellin order)`` truths: cross-package-deterministic and
    pairwise distinct.

    Renamed 2026-08-13 from ``..._and_seeded_by_power``, which promised a
    relationship between the drawn value and the Mellin order that nothing here
    checks and that ``build_moment_systematic_values`` does not have:
    it takes one ``rng.standard_normal()`` per ``(field, key, order)`` spec, with the
    order entering only as text inside the dict key. "Deterministic per order" is the
    property actually asserted -- one independent draw per spec, reproducible from
    the seed.

    Same cross-package-consistency character as
    ``test_seeded_systematic_curves_match_between_truth_families`` (real, but
    only catches the JAM/NNPDF twins drifting apart, not a shared bug).
    ``len(set(jam.values())) == len(jam)`` would catch an RNG-threading bug
    (e.g. the RNG being reset per key instead of advanced, collapsing every
    draw to the same value), since duplicate values among iid
    ``standard_normal()`` draws are otherwise a probability-zero event. It would
    NOT catch a duplicate *name* bug in ``moment_nuisance_specs()`` itself: both
    the actual dict and the "expected" set on the next line are built from that
    same spec list, so a duplicated spec silently reduces both sides together.
    """
    jam = jam_gen.build_moment_systematic_values(
        np.random.default_rng(jam_cfg.MOMENT_SYSTEMATIC_SEED)
    )
    nnpdf = nnpdf_gen.build_moment_systematic_values(
        np.random.default_rng(nnpdf_cfg.MOMENT_SYSTEMATIC_SEED)
    )
    assert jam == nnpdf
    assert set(jam) == {
        jam_cfg.moment_nuisance_field_name(field, key, order)
        for field, key, order in jam_cfg.moment_nuisance_specs()
    }
    assert len(set(jam.values())) == len(jam)


def test_generation_fold_caps_mellin_scalars_separately_from_itd_curves(
    suite, basis_nodes, sys_curves
):
    """Mellin/scalar systematics cap independently of, and stay out of, the ITD
    curve cap machinery -- restricted to one ensemble so each nuisance name
    has exactly one contributing record.

    Mixed character. ``folds.scale == {}`` and ``set(folds.moment_scale) ==
    expected_names`` are real separation/coverage checks. The per-record
    ``expected_raw = gen.coefficient(key, meta) * raw_values[name]`` and
    ``folds.moment_values[name] == raw_values[name] * folds.moment_scale[name]``
    checks are real wiring checks too, since ``raw_values`` is supplied
    externally (from ``gen.build_moment_systematic_values``) rather than read
    back from ``fold_lattice_systematics``'s own return value -- unlike
    ``test_generation_fold_enforces_cap_for_every_field_key``'s cap check, these
    are not self-referential (``gen.coefficient`` is still common-mode with the
    source's own factory call, per the same caveat as
    ``test_mellin_systematics_use_distinct_one_point_fields_and_unit``). The
    final cap line has the same algebraic-identity character as
    ``test_generation_fold_enforces_cap_for_every_field_key``'s cap check (single
    contributing record per name means ``moment_scale = cap / ratio`` guarantees
    equality). It used to be asserted one-directionally (``<=``), strictly weaker
    than the ITD sibling's ``np.isclose``, so a ``moment_scale`` bug that *under*-uses
    the allowed cap -- a stray factor of 2 -- would have passed. It is now two-sided
    at ``rtol=1e-10`` to match: measured 2026-08-13,
    ``max|ratio/cap - 1| = 2.22e-16`` over every (field, key) pair of the first
    ensemble, so the equality holds to float noise and the one-sided form was leaving
    a factor of 2 unguarded for nothing.
    """
    cfg, gen = suite
    basis, nodes = basis_nodes
    ensemble_id = cfg.active_ensembles()[0].id
    records = [
        rec
        for rec in gen.lattice_layout()
        if rec["kind"] == "mellin" and rec["ensemble"] == ensemble_id
    ]
    operators = []
    for rec in records:
        B = np.asarray(
            kernels.Mellin(
                alpha=-1.0,
                low_x_extension=cfg.low_x_completion(rec["field"]),
            ).matrix(
                rec["_n"], basis
            )
        )
        operators.append([(rec["field"], B)])

    shape = nodes**0.4 * (1.0 - nodes) ** 3
    replica_scale = np.linspace(0.8, 1.2, 6)
    ensemble = {
        field: (index + 1.0) * replica_scale[:, None] * shape[None, :]
        for index, field in enumerate(cfg.ALL_FIELDS)
    }
    raw_values = gen.build_moment_systematic_values(
        np.random.default_rng(cfg.MOMENT_SYSTEMATIC_SEED)
    )
    folds = gen.fold_lattice_systematics(
        records, operators, ensemble, sys_curves, raw_values
    )

    assert folds.scale == {}
    expected_names = {
        cfg.moment_nuisance_field_name(field, key, order)
        for field, key, order in cfg.moment_nuisance_specs()
    }
    assert set(folds.moment_scale) == expected_names
    for index, (rec, operator) in enumerate(zip(records, operators)):
        field = operator[0][0]
        central = folds.central[index]
        denom = np.where(np.abs(central) > 0.0, np.abs(central), 1.0)
        meta = gen.lattice_meta(rec)
        for key, raw_shift in folds.raw_shift[index].items():
            name = cfg.moment_nuisance_field_name(field, key, rec["order"])
            expected_raw = gen.coefficient(key, meta) * raw_values[name]
            np.testing.assert_allclose(raw_shift, expected_raw)
            np.testing.assert_allclose(
                folds.moment_values[name],
                raw_values[name] * folds.moment_scale[name],
            )
            ratio = np.max(
                np.abs(folds.moment_scale[name] * raw_shift) / denom
            )
            # Algebraically guaranteed equality (moment_scale = cap / ratio for
            # the single record behind this name), now asserted two-sided to match
            # the ITD sibling; achieved 2.22e-16, bar 1e-10.
            assert ratio <= cfg.SYSTEMATIC_MAX_FRACTION * (1.0 + 1e-12)
            assert np.isclose(ratio, cfg.SYSTEMATIC_MAX_FRACTION, rtol=1e-10), (
                f"{name}: ratio {ratio!r} vs cap {cfg.SYSTEMATIC_MAX_FRACTION}"
            )


def test_systematic_curves_are_distinct_across_keys(suite, sys_curves):
    """Each ``(field, key)`` gets its own draw -- no key shares a curve with another.

    The scalar sibling
    (``test_mellin_systematic_truths_are_independent_and_deterministic_per_order``)
    checks ``len(set(values)) == len(values)`` for the Mellin nuisances, but nothing
    checked the analogous property for the x-space ITD curves. It is not implied by
    any other test here: parity, the nulled moment, the ``nu=0`` value, the cap and
    the decay bound all hold *per curve*, so they stay green if every key of a field
    were handed the identical curve -- which is exactly what hoisting
    ``build_systematic_curves``'s three ``rng.uniform`` draws out of the
    ``for key in cfg.ITD_SYSTEMATICS`` loop would do.

    Oracle ``A2`` (a symmetry/independence property of the construction), asserted as
    a *relative* separation so it does not depend on the curves' overall scale:
    ``max|a-b| / max(max|a|, max|b|)``. Measured 2026-08-13 over the 90 same-field key
    pairs of each suite: minimum separation ``2.55e-02``, maximum ``1.95e+00``; the
    180 same-key field pairs bottom out at ``1.38e-01``. Bar ``1e-3``, ~25x below the
    tightest real pair, so it flags collapse rather than policing the draw
    distribution.
    """
    cfg, _ = suite
    keys = list(cfg.ITD_SYSTEMATICS)
    fields = list(cfg.ALL_FIELDS)
    for field in fields:
        for i, key_a in enumerate(keys):
            for key_b in keys[i + 1:]:
                a, b = sys_curves[field][key_a], sys_curves[field][key_b]
                scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
                separation = float(np.max(np.abs(a - b))) / scale
                assert separation > 1e-3, (
                    f"{field}: {key_a} and {key_b} coincide ({separation:.3e})"
                )
    # The same draw must not be reused across fields either: build_systematic_curves
    # advances one rng through (field, key) in order, so a per-field reset would
    # collapse this axis instead.
    for key in keys:
        for i, field_a in enumerate(fields):
            for field_b in fields[i + 1:]:
                a, b = sys_curves[field_a][key], sys_curves[field_b][key]
                scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
                separation = float(np.max(np.abs(a - b))) / scale
                assert separation > 1e-3, (
                    f"{key}: {field_a} and {field_b} coincide ({separation:.3e})"
                )


def test_fold_lattice_systematics_rejects_malformed_inputs(
    suite, sys_curves, actual_lattice_folds
):
    """The three ``ValueError`` guards in ``fold_lattice_systematics`` are reached.

    ``F1`` (structural): these are defensive raises, not physics. All three were
    unreached by any test in the suite -- the happy path is the only path the cap and
    SPD tests take -- so a guard that had been deleted, or whose bound was inverted,
    would have been invisible.

    Each case corrupts a *copy* of the real fixture inputs and matches on the
    guard's **message**, not merely on ``ValueError``: all three raise the same class,
    and type-only matching lets a corruption trip the wrong guard and still "pass".
    The control at the end asserts the uncorrupted inputs raise nothing, so the cases
    cannot all be passing against a validator that rejects everything.

    ``max_fraction`` is probed at both ends and just outside them
    (``-1e-9`` and ``1 + 1e-9``), and the accepted boundary values ``0.0`` and ``1.0``
    are shown to be accepted -- the guard is ``0 <= f <= 1``, so an off-by-one-strictness
    slip to ``<`` is what those two control cases catch.
    """
    _, gen = suite
    records, operators, ensemble, _ = actual_lattice_folds
    # Two records is enough for every guard and keeps the run cheap.
    records = list(records[:2])
    operators = list(operators[:2])

    with pytest.raises(ValueError, match="records and operators must have equal length"):
        gen.fold_lattice_systematics(records, operators[:1], ensemble, sys_curves)

    for bad in (-1e-9, 1.0 + 1e-9, -1.0, 2.0):
        with pytest.raises(ValueError, match=r"max_fraction must lie in \[0, 1\]"):
            gen.fold_lattice_systematics(
                records, operators, ensemble, sys_curves, max_fraction=bad
            )

    empty = [list(operators[0]), []]
    with pytest.raises(ValueError, match="empty operator"):
        gen.fold_lattice_systematics(records, empty, ensemble, sys_curves)

    # Controls: the unmodified inputs, and both accepted endpoints of max_fraction.
    gen.fold_lattice_systematics(records, operators, ensemble, sys_curves)
    for good in (0.0, 1.0):
        gen.fold_lattice_systematics(
            records, operators, ensemble, sys_curves, max_fraction=good
        )


def test_inflate_diagonal_symmetrizes_and_scales_the_diagonal(suite):
    """``inflate_diagonal(C, f) == sym(C) + f * diag(diag(C))``, exactly.

    ``inflate_diagonal`` had only ever been exercised through
    ``test_actual_folded_covariance_blocks_become_spd``'s already-symmetric,
    mildly-singular ``np.cov`` output, so its ``0.5 * (cov + cov.T)`` line had never
    seen an asymmetric input and nothing recorded what happens when 1% is not enough
    to reach positive-definiteness.

    Oracle ``A1``: the closed form written out here in numpy. Exact equality
    (``assert_array_equal``) -- the implementation is two adds and a multiply on the
    same floats, and it was measured bit-exact, so there is no tolerance to choose.

    The last block records the real limitation, which is a *multiplicative* nugget's:
    ``C_ii -> C_ii (1 + f)`` cannot lift a zero diagonal entry, so a covariance with
    an exactly-zero variance row stays singular however large ``f`` is. Asserted
    directly (the Cholesky raises), so the boundary of what this regularizer can do is
    written down rather than assumed.
    """
    _, gen = suite

    asymmetric = np.array([[4.0, 1.0], [3.0, 9.0]])
    frac = 0.01
    symmetric = 0.5 * (asymmetric + asymmetric.T)
    expected = symmetric + frac * np.diag(np.diag(symmetric))
    np.testing.assert_array_equal(gen.inflate_diagonal(asymmetric, frac), expected)
    # Spelled out: the off-diagonal is averaged, the diagonal scaled by 1+f.
    np.testing.assert_array_equal(
        gen.inflate_diagonal(asymmetric, frac),
        np.array([[4.04, 2.0], [2.0, 9.09]]),
    )

    # A genuinely singular (rank-1) block is lifted to SPD by 1%.
    rank_one = np.ones((2, 2))
    regularized = gen.inflate_diagonal(rank_one, frac)
    np.linalg.cholesky(regularized)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(regularized), [frac, 2.0 + frac], rtol=1e-13, atol=0.0
    )

    # ...but a zero variance is not liftable by a multiplicative nugget, at any frac.
    zero_variance = np.array([[0.0, 0.0], [0.0, 1.0]])
    for big in (frac, 1.0, 1e6):
        inflated = gen.inflate_diagonal(zero_variance, big)
        assert inflated[0, 0] == 0.0
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(inflated)


def test_active_ensembles_honours_env_override_and_module_subset(suite, monkeypatch):
    """``active_ensembles()``'s env override and ``ACTIVE_ENSEMBLE_IDS`` subsetting.

    ``F1`` (structural): both branches are dead in every other test in the suite --
    ``ACTIVE_ENSEMBLE_IDS`` is ``None`` and ``PIXEL_CLOSURE_ENSEMBLES`` is unset, so
    the module-parametrized fixtures always see the full 13-ensemble tuple and the
    only covered path is the ``return LATTICE_ENSEMBLES`` default.

    Covered here: the default; the env var (including precedence over
    ``ACTIVE_ENSEMBLE_IDS``, whitespace stripping, and empty entries being dropped);
    an env var set to the empty string falling through to the module constant rather
    than returning nothing; the module constant alone; and an unknown id raising
    ``KeyError`` rather than silently yielding a short tuple. Order is asserted to
    follow the *requested* ids, not the table's, since ``lattice_layout`` emits
    records in this order.
    """
    cfg, _ = suite
    monkeypatch.delenv("PIXEL_CLOSURE_ENSEMBLES", raising=False)
    monkeypatch.setattr(cfg, "ACTIVE_ENSEMBLE_IDS", None)
    default = cfg.active_ensembles()
    assert default is cfg.LATTICE_ENSEMBLES
    assert len(default) == 13

    monkeypatch.setattr(cfg, "ACTIVE_ENSEMBLE_IDS", ("a067m135",))
    assert [e.id for e in cfg.active_ensembles()] == ["a067m135"]

    # Env var wins over the module constant, strips whitespace, drops empties, and
    # preserves the requested order rather than the table's.
    monkeypatch.setenv("PIXEL_CLOSURE_ENSEMBLES", " a053m230 ,, a117m310 ")
    assert [e.id for e in cfg.active_ensembles()] == ["a053m230", "a117m310"]

    # An empty override is falsy, so it must fall through to the module constant.
    monkeypatch.setenv("PIXEL_CLOSURE_ENSEMBLES", "")
    assert [e.id for e in cfg.active_ensembles()] == ["a067m135"]

    monkeypatch.setenv("PIXEL_CLOSURE_ENSEMBLES", "not_an_ensemble")
    with pytest.raises(KeyError):
        cfg.active_ensembles()


def test_fold_uses_fixed_truth_curves_when_given_and_replica_mean_otherwise(
    suite, basis_nodes, sys_curves, actual_lattice_folds
):
    """``fold_lattice_systematics(truth_curves=...)`` folds that truth, not the
    replica mean -- and ``assemble_operator`` builds the multi-entry operator it does.

    ``generate.py:280`` reads ``central = Y.mean(axis=0) if truth_curves is None else
    fold_truth(operator, truth_curves)``. ``generate_member`` -- the production path --
    always passes ``truth_curves=mean_curves``, so the ``else`` branch is the one that
    makes the lattice channel obey the same fixed-truth rule
    ``tests/test_closure_truth_representable.py`` pins for DIS and DY. Every previous
    call in this file omitted ``truth_curves``, so only the ``Y.mean(axis=0)`` branch
    had ever run.

    Oracle ``A1``: the expected central value is ``B @ truth[field]`` written in
    plain numpy here. The truth curve is deliberately **not** the replica mean --
    ``truth = 1.7 * shape + 0.3`` against a replica mean of ``(index+1) * shape``
    -- and the control asserts the two branches actually differ on this fixture
    (measured relative separation is reported in the failure message if it ever
    collapses), because a truth that happened to equal the replica mean would make
    the whole test tautological in exactly the way the ``a ± a`` and ``model ==
    truth`` fixtures elsewhere in this audit did.

    ``assemble_operator`` is called here for the first time in the suite, on a
    production-shaped dataset carrying a physical pseudo-ITD contribution plus a
    ``Unit`` nuisance contribution. That is the multi-entry operator shape
    ``fold_truth``'s ``for field, B in operator[1:]`` loop exists for, and the shape
    ``build_pseudoitd`` produces once ``_sys_map`` adds its nuisance fields.
    """
    cfg, gen = suite
    basis, nodes = basis_nodes
    records, operators, ensemble, _ = actual_lattice_folds
    records = list(records[:3])
    operators = list(operators[:3])

    shape = nodes**0.4 * (1.0 - nodes) ** 3
    truth_curves = {field: 1.7 * shape + 0.3 for field in cfg.ALL_FIELDS}

    with_truth = gen.fold_lattice_systematics(
        records, operators, ensemble, sys_curves, truth_curves=truth_curves
    )
    without = gen.fold_lattice_systematics(records, operators, ensemble, sys_curves)
    for index, operator in enumerate(operators):
        field, B = operator[0]
        np.testing.assert_allclose(
            with_truth.central[index], B @ truth_curves[field], rtol=1e-13, atol=0.0
        )
        replica_mean = (ensemble[field] @ B.T).mean(axis=0)
        np.testing.assert_array_equal(without.central[index], replica_mean)
        # Control: the two branches are separated by this fixture, so the assertion
        # above could have failed had the fold taken the wrong one.
        separation = float(
            np.max(np.abs(with_truth.central[index] - replica_mean))
            / np.max(np.abs(replica_mean))
        )
        assert separation > 1e-2, f"record {index}: branches coincide ({separation:.3e})"

    # The covariance is a replica statistic either way -- only the central value
    # switches branch.
    for a, b in zip(with_truth.covariance, without.covariance):
        np.testing.assert_array_equal(a, b)

    # assemble_operator on a production-shaped, multi-contribution dataset.
    fields = cfg.make_fields()
    field_name = records[0]["field"]
    nuisance_name = cfg.nuisance_field_name(field_name, cfg.ITD_SYSTEMATICS[0])
    nuisance = Field.create(nuisance_name, cfg.make_grid(), element_type=cfg.ELEMENT_TYPE)
    fields = {field_name: fields[field_name], nuisance_name: nuisance}
    nu = gen.lattice_meta(records[0])["nu"]
    alpha = -1.0 if records[0]["momentum_density"] else 0.0
    low_x = cfg.low_x_completion(field_name)
    kernel_cls = (
        kernels.PseudoITDReal
        if records[0]["component"] == "real"
        else kernels.PseudoITDImag
    )
    dataset = SimpleNamespace(
        nu=nu,
        contributions=[
            Contribution(field_name, kernel_cls(alpha=alpha, low_x_extension=low_x)),
            Contribution(nuisance_name, kernel_cls(alpha=alpha, low_x_extension=low_x)),
        ],
    )
    operator = gen.assemble_operator(dataset, fields)
    assert [entry[0] for entry in operator] == [field_name, nuisance_name]
    for (name, B), contribution in zip(operator, dataset.contributions):
        np.testing.assert_array_equal(
            B, np.asarray(contribution.kernel.matrix(nu, fields[name].basis), dtype=float)
        )
    # ...and fold_truth accumulates every entry of it, not just the first.
    two_field_truth = {
        field_name: truth_curves[field_name],
        nuisance_name: 0.4 * shape - 0.9,
    }
    np.testing.assert_allclose(
        gen.fold_truth(operator, two_field_truth),
        operator[0][1] @ two_field_truth[field_name]
        + operator[1][1] @ two_field_truth[nuisance_name],
        rtol=1e-13,
        atol=0.0,
    )
    assert not np.allclose(
        gen.fold_truth(operator, two_field_truth),
        operator[0][1] @ two_field_truth[field_name],
    )
