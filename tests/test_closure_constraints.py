"""Closure pseudo-constraints must contain the represented injected truth.

Exercises the repo-root ``closure_JAM_truth``, ``closure_NNPDF_truth``, and their
``_small`` counterparts -- specifically ``fit.constraint_datasets`` (endpoint,
valence-norm, and origin pseudo-data) and ``config.{make_grid, vanishes_at_origin,
low_x_completion, ALL_FIELDS, VALENCE_NORMS, SINGLET_GLUON_FIELDS,
CONSTRAINT_*_SIGMA, ENDPOINT_X, T0_MAX_ITERATIONS}``.  These are ``closure_*/``
driver packages, not ``src/pixel``: no formula native to ``pixel`` is under test
here.  ``pixel.core.model.{Field, Dataset, Contribution}``
(``src/pixel/core/model.py``), ``pixel.geometry.Grid``
(``src/pixel/geometry/grid.py``), and ``pixel.kernels.{Delta, Mellin}``
(``src/pixel/kernels/common.py``) appear only as generic, already-covered plumbing
that ``constraint_datasets`` calls.

**Two suite tuples, deliberately different lengths.**  ``ALL_SUITES`` is all four
packages ``(full_jam, full_nnpdf, small_jam, small_nnpdf)`` and is read by exactly one
test, ``test_the_four_suites_still_share_one_constraint_builder``, whose whole subject
is that the four copies have not diverged.  ``SUITES`` -- the NNPDF pair, since
2026-08-14 on the owner's instruction -- is what every ``@parametrize("suite", ...)``
below uses.

The reason the behavioural sweep does not need all four: ``constraint_datasets`` is
byte-identical across all four ``fit.py`` files, and every ``config.py`` constant this
file reads (grid geometry, field lists, ``CONSTRAINT_*_SIGMA``, ``ENDPOINT_X``,
``T0_MAX_ITERATIONS``) is identical too -- and that is *asserted*, by the test named
above, not taken on trust.  So running the numeric checks against all four was one code
path executed four times for one answer.  What the identity test cannot do is notice a
divergence in behaviour that both copies' source shares, which is why the ``_small`` leg
is kept alongside the full one: those two genuinely differ in grid size.

Most numeric checks recompute a prediction with ``contribution.kernel.matrix(...)``
-- the *same kernel object* ``constraint_datasets`` already used to build
``dataset.mean`` -- against a curve provably identical (exact-knot ``np.interp``)
to the one used internally.  That is close to ``assert f(x) == f(x)``: it shows the
``Dataset`` stores what ``constraint_datasets`` computed, not that the ``nu``,
``alpha``, or target formula it chose is correct.  Measured by mutation: swapping
the endpoint ``nu`` from the correct ``1.0`` to a wrong ``0.9`` moves the stored
target from ``~5.9e-20`` (numerically zero) to ``~1.5e-2``, a factor of ``~2.5e17``,
and the same recomputation still agrees with it to ``atol=1e-12``.  The one check a
wrong implementation could not fake this way is the one-hot ``Delta`` row in
``test_origin_constraint_switches_on_for_the_vanishing_fields``; see that test's
docstring.

Two tests added 2026-08-13 remove both of those blind spots and are the ones to
read first:

* ``test_constraint_targets_match_an_analytic_affine_truth`` supplies the closed
  form the recompute tests lack.  On the affine truth ``q(x) = A + B x`` -- which
  a cubic-spline element basis represents *exactly* -- both target families have
  elementary values (``q(1) = A + B``; ``int q/x = A(1 + ln 1/x_min) + B``)
  computed with no reference to ``pixel``, so a wrong ``ENDPOINT_X``, a wrong
  Mellin ``alpha``, or a wrong low-x completion changes the answer.
* ``test_constraint_targets_read_the_represented_truth_not_the_nominal_value``
  adds the offset curve ``x**0.3 (1-x)**2 + 0.25``, nonzero at *both* endpoints,
  so represented (``0.25``) and nominal (``0.0``) finally differ for the endpoint
  and origin families as they already did for the valence norms.

The affine truth's ``rtol=2e-4`` norm bar is what one ``x_min`` and one
*non-vanishing* truth buy, not a low-x guarantee: it is exactly
``9.18e-4 * q(x_min) / norm_exact`` and falls only as ``1/ln(1/x_min)``.
``test_valence_norm_low_x_error_converges_in_x_min`` carries the claim it cannot --
it sweeps ``x_min`` in ``{1e-6, 1e-7, 1e-8}`` at fixed ``n_points = 145``, states
``x_min`` explicitly rather than inheriting ``cfg.X_MIN``, and asserts the
``10**a``-per-decade **rate** of the completion's model error on a *vanishing*
truth ``x**a (1-x)**3``.  Measured diagnosis: ``plans/low_x_head_diagnosis.md``.

The original curve family ``x**0.3 * (1-x)**2`` is kept where it still earns its
place -- it is the one whose targets can be read off by inspection -- but it is
exactly zero at both ``x = 0`` and ``x = 1``, so on its own it could not tell a
correct represented-truth read from a silent fallback to nominal for the endpoint
and origin families (measured ``~5.9e-20`` and exactly ``0.0``); only the
valence-norm sub-case could (measured ``represented=3.29`` vs ``nominal=1.0`` for
``v3``).

Production use of these same near-hard constraints is an actual closure fit via
``closure_*/run_closure.py`` (see ``closure_JAM_truth/README.md``), which this
file does not run.
"""

import ast
import inspect
import pathlib
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest

import pixel
from pixel import kernels
from pixel.core.model import Field
from pixel.geometry import Grid

from closure_JAM_truth import fit as full_jam
from closure_NNPDF_truth import fit as full_nnpdf
from closure_JAM_truth_small import fit as small_jam
from closure_NNPDF_truth_small import fit as small_nnpdf


#: All four closure ``fit`` modules.
#:
#: **Consumed only by ``test_the_four_suites_still_share_one_constraint_builder``**
#: (``reference, *others = ALL_SUITES``), whose subject *is* the four copies: it asserts
#: their ``constraint_datasets`` source and 14 ``config`` constants agree.  Trimming this
#: tuple would make that test compare a suite to itself and assert nothing.  It looks like
#: a redundant superset of ``SUITES``.  It is not -- keep all four.
ALL_SUITES = (full_jam, full_nnpdf, small_jam, small_nnpdf)

#: The suites every ``@pytest.mark.parametrize("suite", SUITES)`` below runs against:
#: **the NNPDF pair only**, on the owner's instruction 2026-08-14.  This was
#: ``ALL_SUITES``; the JAM legs re-ran an identical ``constraint_datasets`` over identical
#: config constants -- identical as asserted by the test named above, not as assumed.  The
#: full/``_small`` pair is retained because those two really do differ (grid size).
SUITES = (full_nnpdf, small_nnpdf)

#: Quark counting, spelled out here rather than read back from
#: ``cfg.VALENCE_NORMS``: ``int V3 = 1`` (one more u than d in a proton) and
#: ``int V8 = int V = int V15 = 3`` (three valence quarks).  These are the
#: physical sum rules the ``cons_norm_*`` pseudo-data exist to impose, so a
#: typo in ``config.VALENCE_NORMS`` must be caught by an independent statement
#: of them, not by reading that table twice.
VALENCE_NUMBERS = {"v3": 1.0, "v8": 3.0, "v": 3.0, "v15": 3.0}

#: ``x = 1`` -- the momentum density vanishes at the elastic endpoint.  A
#: literal for the same reason as ``VALENCE_NUMBERS``: ``cfg.ENDPOINT_X`` is
#: what is under test.  Measured (audit ``m2``): moving it to ``0.9`` changes a
#: stored target by a factor ``2.5e17`` without disturbing any
#: same-kernel recompute in this file.
ENDPOINT_X = 1.0


@pytest.mark.parametrize("suite", SUITES)
def test_near_hard_constraints_match_truth_in_the_fit_basis(suite):
    """Every near-hard dataset's mean equals ``kernel.matrix(nu, basis) @ curve``.

    Builds a synthetic curve per field (``x**0.3 * (1-x)**2``, scaled by
    ``1 + 0.1*i``) on ``suite.cfg.make_grid()``, then checks that
    ``constraint_datasets`` returns one endpoint dataset per field plus one norm
    dataset per ``VALENCE_NORMS`` entry (``9 + 4 = 13``), and that every dataset's
    ``mean`` reproduces ``contribution.kernel.matrix(dataset.nu, basis) @ curve``
    to ``atol=1e-12`` (``rtol=0.0``).

    Not independent: ``contribution.kernel`` is the literal object
    ``constraint_datasets`` used to compute ``dataset.mean``
    (``closure_JAM_truth/fit.py:57-66``), and the curve is provably identical
    (``np.interp`` at its own knots is exact) to the one used internally -- this is
    ``assert f(x) == f(x)`` for the kernel/curve pairing.  It proves the ``Dataset``
    stores what ``constraint_datasets`` computed, not that the ``nu``, ``alpha``, or
    target formula chosen is physically right.  Measured: swapping the endpoint
    ``nu`` from the correct ``1.0`` to a wrong ``0.9`` moves the stored target from
    ``~5.9e-20`` to ``~1.5e-2`` and this same recomputation still passes at
    ``atol=1e-12`` -- a mistyped ``ENDPOINT_X`` would not be caught here.  The
    independent facts checked are the dataset *count* and that each dataset carries
    exactly one contribution (the tuple-unpack on the line below raises otherwise).
    """
    fields = {
        name: Field.create(
            name,
            suite.cfg.make_grid(),
            element_type=suite.cfg.ELEMENT_TYPE,
        )
        for name in suite.cfg.ALL_FIELDS
    }
    nodes = np.asarray(next(iter(fields.values())).nodes, dtype=float)
    curves = {
        name: (1.0 + 0.1 * i) * nodes ** 0.3 * (1.0 - nodes) ** 2
        for i, name in enumerate(suite.cfg.ALL_FIELDS)
    }
    truth = {
        "x_nodes": nodes.tolist(),
        "curves": {name: curve.tolist() for name, curve in curves.items()},
    }

    datasets = suite.constraint_datasets(fields, truth)
    # + 1 for cons_momentum (2026-08-15): endpoint per field, norm per valence
    # field, and one momentum sum rule over Sigma + g jointly.
    assert len(datasets) == (
        len(suite.cfg.ALL_FIELDS) + len(suite.cfg.VALENCE_NORMS) + 1
    )
    for dataset in datasets:
        # cons_momentum is the only constraint spanning more than one field
        # (Sigma + g), so it cannot be unpacked as a single contribution.  Skipped
        # by name, not by len(), so another multi-field constraint appearing here
        # fails loudly instead of being quietly ignored.
        if dataset.name == "cons_momentum":
            continue
        (contribution,) = dataset.contributions
        field = fields[contribution.field]
        prediction = np.asarray(
            contribution.kernel.matrix(dataset.nu, field.basis), dtype=float
        ) @ curves[contribution.field]
        # Exact float64 reproduction of the same kernel call constraint_datasets
        # made (module docstring); not a physics accuracy bound.
        np.testing.assert_allclose(prediction, dataset.mean, rtol=0.0, atol=1e-12)
        assert "nominal_target" in dataset.metadata.extras
        assert "closure_target" in dataset.metadata.extras


# -- an analytic truth, so the targets have values and not just provenance ----

#: ``q(x) = A + B x``.  A natural cubic spline through samples of an affine
#: function *is* that function, so this truth is represented exactly by the
#: element basis and both target families reduce to elementary integrals with
#: no discretization error left to bound.  ``A != 0`` is what makes the
#: ``1/x`` weight and the low-x completion visible in the norm target: it
#: contributes the whole ``A ln(1/x_min)`` term.
AFFINE_A, AFFINE_B = 0.4, 0.7


def _affine_scale(index):
    """Per-field scale, so a field mix-up cannot pass by symmetry."""
    return 1.0 + 0.1 * index


@pytest.mark.parametrize("suite", SUITES)
def test_constraint_targets_match_an_analytic_affine_truth(suite):
    """Both near-hard target families equal their closed form on ``q = A + B x``.

    The independent oracle this file otherwise lacks (oracle A1).  Nothing below
    calls ``pixel``: the expected values are elementary integrals of the injected
    curve, so unlike the recompute tests they are not invariant under a wrong
    ``nu``, a wrong Mellin ``alpha``, or a wrong low-x completion.

    * endpoint -- ``Delta`` at ``x = 1`` returns the interpolant there, and the
      spline reproduces an affine function exactly, so the target is ``q(1) =
      A + B``.  Measured ``|got/exact - 1| = 0.0`` for all nine fields, but
      asserted at ``rtol=1e-12`` rather than exactly: the row at ``x = 1`` has
      110 nonzeros (it is a global spline evaluation, not the one-hot row the
      ``x = 0`` node gives), and the summation order of a 110-term dot product
      is not contractual across BLAS builds.
    * valence norm -- ``Mellin(alpha=-1)`` at ``n = 1`` is ``int_0^1 q(x)/x dx``.
      Below ``x_min`` the field is not resolved and ``cfg.low_x_completion``
      continues it as ``q(x_min) (x/x_min)^1``, which contributes exactly
      ``q(x_min)``; above it ``int_{x_min}^1 (A/x + B) dx``.  Total
      ``A (1 + ln(1/x_min)) + B``.

    The norm bar is ``rtol=2e-4`` against a **measured** ``5.543e-5`` (27.7% of
    the bar used, worst of nine fields): PIXEL's
    ``[0, x_min)`` head is short by ``9.18e-4 * q(x_min)``, and that shortfall is
    real, not a resolution artefact -- it survives grid refinement (``N`` 65 ->
    513, unchanged to four digits) and quadrature refinement
    (``points_per_interval`` 8 -> 256, unchanged to 15 digits), because
    ``kernels.base.assemble`` disables the exact ``low_x_mellin_head`` whenever
    ``alpha != 0`` and the Gauss-Jacobi fallback then integrates a ``1/t``
    singularity against a ``t^gamma`` weight.  It cancels out of a closure fit
    (the same operator builds the target and reads it back) but it is not zero,
    so the bar is set above it deliberately rather than tuned down to it.

    **``2e-4`` is not a universal low-x guarantee, and this truth cannot be made
    into one.**  The achieved value is exactly ``9.18e-4 * q(x_min) / norm_exact``
    -- reproduced to every printed digit at five ``x_min`` values -- so it inherits
    ``cfg.X_MIN`` through ``1 / (1 + ln(1/x_min))`` and falls only *logarithmically*:
    measured ``7.679e-05``, ``6.438e-05``, ``5.543e-05``, ``4.867e-05``,
    ``4.338e-05`` at ``x_min`` ``1e-4 .. 1e-8``, a factor of 1.77 across four
    decades.  The reason is this truth's own class: ``q(0) = AFFINE_A = 0.4 != 0``,
    so ``int q/x`` is log-divergent in exact arithmetic and ``norm_exact`` above is
    finite only *because* the completion is what defines it.  There is no model
    error left for a rate to converge -- only the head's quadrature error.  A
    convergence test therefore needs a **vanishing, sublinear** truth, which is what
    ``test_valence_norm_low_x_error_converges_in_x_min`` uses; see
    ``plans/low_x_head_diagnosis.md`` for the full measurement of both classes.
    """
    cfg = suite.cfg
    fields = _fields_on(suite, cfg.make_grid())
    nodes = np.asarray(next(iter(fields.values())).nodes, dtype=float)
    x_min = float(nodes.min())
    curves = {
        name: _affine_scale(i) * (AFFINE_A + AFFINE_B * nodes)
        for i, name in enumerate(cfg.ALL_FIELDS)
    }
    truth = {
        "x_nodes": nodes.tolist(),
        "curves": {name: curve.tolist() for name, curve in curves.items()},
    }
    scale = {name: _affine_scale(i) for i, name in enumerate(cfg.ALL_FIELDS)}

    endpoint_exact = AFFINE_A + AFFINE_B
    norm_exact = AFFINE_A * (1.0 + np.log(1.0 / x_min)) + AFFINE_B

    seen = set()
    for dataset in suite.constraint_datasets(fields, truth):
        # cons_momentum is the only constraint spanning more than one field
        # (Sigma + g), so it cannot be unpacked as a single contribution.  Skipped
        # by name, not by len(), so another multi-field constraint appearing here
        # fails loudly instead of being quietly ignored.
        if dataset.name == "cons_momentum":
            continue
        (contribution,) = dataset.contributions
        name = contribution.field
        seen.add(dataset.name)
        if dataset.name.startswith("cons_endpoint_"):
            # x = 1 stated here, not read from cfg -- ENDPOINT_X is under test.
            np.testing.assert_allclose(dataset.nu, [ENDPOINT_X], rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                dataset.mean, [scale[name] * endpoint_exact], rtol=1e-12, atol=0.0
            )
        else:
            # n = 1 is the number sum rule; the alpha=-1 weight turns the
            # momentum density q = x f back into int f dx.
            np.testing.assert_allclose(dataset.nu, [1.0], rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                dataset.mean, [scale[name] * norm_exact], rtol=2.0e-4, atol=0.0
            )
            # Independent statement of the quark-counting rule (module header).
            assert dataset.metadata.extras["nominal_target"] == VALENCE_NUMBERS[name]
        assert dataset.metadata.extras["closure_target"] == pytest.approx(
            float(dataset.mean[0]), rel=1e-15, abs=0.0
        )
    assert seen == (
        {f"cons_endpoint_{n}" for n in cfg.ALL_FIELDS}
        | {f"cons_norm_{n}" for n in VALENCE_NUMBERS}
    )


# -- the valence norm's low-x head, measured as a function of x_min ------------
#
# The ``rtol=2e-4`` above is what one ``x_min`` and one *non-vanishing* truth buy.
# The claim it reads like -- "the valence-norm target is good to 2e-4" -- is a
# statement about the low-x completion, and the completion's error is a power of
# ``x_min`` set by the field's small-x behaviour.  Everything below states ``x_min``
# explicitly and asserts that power.  Measured diagnosis: ``plans/low_x_head_diagnosis.md``.

#: ``x_min`` ladder for :func:`test_valence_norm_low_x_error_converges_in_x_min`.
#: Written out, never read from ``cfg.X_MIN``: inheriting it is how the fixed bar
#: above acquired its air of universality.
NORM_SWEEP_X_MIN = (1e-6, 1e-7, 1e-8)

#: Grid size held fixed across the sweep so ``x_min`` is the only moving part.  145
#: is this repo's closure-grid floor and is what the diagnosis measured on.
NORM_SWEEP_N_POINTS = 145

#: Large-``x`` exponent of the swept truth ``q(x) = x**a (1-x)**beta``.  Only sets
#: the normalization; the ``[0, x_min)`` head sees ``(1-x)**beta -> 1``.
NORM_TRUTH_BETA = 3.0

#: Small-``x`` power ``a``, and the one number every bar below is a function of.
#:
#: ``0.5`` is a **stand-in for a physical valence density**.  The true small-``x``
#: power of the JAM/NNPDF closure curves is not known here and is the single largest
#: open input in ``plans/low_x_head_diagnosis.md``; the answer scales as ``x_min**a``
#: and the per-decade rate as ``10**a``, so both are written as functions of ``a``
#: rather than as literals -- re-point ``a`` and the bars follow.
#:
#: It must stay **sub**linear, and the affine truth used by the test above is the
#: reason this file needs a second one at all.  ``AFFINE_A + AFFINE_B x`` has
#: ``q(0) != 0``, so its ``int q/x`` is log-divergent and its head error falls only
#: as ``1/ln(1/x_min)``; a *linear* vanishing truth (``a = 1``) is the opposite
#: failure -- the linear head is then exactly right, the model error vanishes, and
#: the residual sits under the FE floor and *rises* with refinement (measured
#: 6.15e-09, 1.22e-08, 1.72e-08 across this ladder).  Only a sublinear vanishing
#: truth has a head error with a rate to converge.
NORM_TRUTH_POWER = 0.5

#: Multiplicative band on the per-decade rate.  Measured deviations from
#: ``10**0.5 = 3.1623``: ``+0.33%`` (1e-6 -> 1e-7) and ``+1.91%`` (1e-7 -> 1e-8), so
#: 1.3 leaves ~15x headroom while still rejecting a neighbouring exponent (``10**1``
#: and ``10**0.25`` both miss ``[2.43, 4.11]``).
NORM_RATE_BAND = 1.3

#: Relative band on the *level* against the closed-form head-model error.  Measured
#: ``|err/pred - 1|`` = ``0.03%``, ``0.29%``, ``2.17%`` (the last is FE
#: discretization starting to show at ``x_min = 1e-8``), so 0.10 leaves 4.6x headroom.
NORM_LEVEL_BAND = 0.10

#: Working precision for the closed-form oracles.
NORM_MP_DPS = 40


def _valence_norm_exact(a=NORM_TRUTH_POWER, beta=NORM_TRUTH_BETA):
    r"""``int_0^1 (1/x) x**a (1-x)**beta dx = B(a, beta+1)``, in closed form.

    The oracle, and independent of everything it measures: an Euler beta function
    evaluated in ``mpmath`` at 40 digits, over the **whole** ``[0, 1]`` with no
    finite-element grid, no ``pixel`` import and no low-x completion.  At the shipped
    ``a = 0.5``, ``beta = 3`` it is the exact rational ``32/35`` -- asserted as such
    by the test, so a mis-specified oracle cannot slip through unnoticed.  Oracle
    ``A1``.
    """
    with mpmath.workdps(NORM_MP_DPS):
        return float(mpmath.beta(mpmath.mpf(a), mpmath.mpf(beta) + 1))


def _valence_norm_head_prediction(gamma, x_min, a=NORM_TRUTH_POWER,
                                  beta=NORM_TRUTH_BETA):
    r"""``(the head PIXEL models) - (the head the truth has)``, over the observable.

    ``cons_norm_*`` is ``Mellin(alpha=-1)`` at ``n = 1``, so the ``[0, x_min)``
    integrand is ``q(x)/x``.  PIXEL continues ``q`` there as
    ``q(x_min) (x/x_min)**gamma``, whose exact contribution is
    ``q(x_min) / gamma`` (``= q(x_min)`` for the production linear head).  The truth's
    own contribution is the incomplete beta ``int_0^x_min x**(a-1) (1-x)**beta dx``.
    Their difference is the **model** error -- the completion shape not being the true
    shape -- and for any sublinear ``a`` it beats PIXEL's quadrature error on the same
    head (``1/(npts+1)**2 = 9.18e-04`` relative at ``npts = 32``) by ~1000x, which is
    why it is allowed to stand as the whole prediction.  To leading order it is
    ``-(1/a - 1) q(x_min) / B(a, beta+1)``, i.e. ``x_min**a``.

    Entirely closed form -- ``mpmath.betainc`` and ``mpmath.beta`` at 40 digits, no
    quadrature anywhere.  Oracle ``A1``.
    """
    with mpmath.workdps(NORM_MP_DPS):
        xm, a_, b_ = mpmath.mpf(x_min), mpmath.mpf(a), mpmath.mpf(beta)
        q_xmin = xm ** a_ * (1 - xm) ** b_
        head_model = q_xmin / mpmath.mpf(gamma)
        head_truth = mpmath.betainc(a_, b_ + 1, 0, xm)
        return float((head_model - head_truth) / mpmath.beta(a_, b_ + 1))


@pytest.mark.parametrize("suite", SUITES)
def test_valence_norm_low_x_error_converges_in_x_min(suite):
    """``cons_norm_*``'s low-x error falls at ``10**a`` per decade of ``x_min``, and
    the level is the completion's *model* error -- not a tolerance.

    The convergence statement ``rtol=2e-4`` above cannot make.  That bar is
    ``9.18e-4 * q(x_min) / norm_exact`` on a truth with ``q(0) != 0``, so it moves
    only as ``1/ln(1/x_min)`` (1.77x across four decades) and its whole content is
    the head's *quadrature* error.  Here the truth **vanishes** at the origin --
    ``q(x) = x**0.5 (1-x)**3``, the physically relevant shape for a valence density --
    which makes the far larger *model* error visible and gives it a rate.

    Runs at ``a = 0.5`` (:data:`NORM_TRUTH_POWER`), a stand-in for physical valence:
    the real closure curves' small-x power is not known here, so both bars are
    written as functions of ``a``.  Everything about them follows from one fact --
    the production completion continues a momentum density linearly
    (``cfg.low_x_completion`` -> ``{"kind": "power", "alpha": 1}``), while the truth
    goes as ``x**a`` with ``a < 1``, so the head is wrong at **leading** order and
    leaves ``(1/a - 1) q(x_min) / observable``:

    * **level** -- ``x_min**a`` exactly, ``1.0938e-03`` at ``a = 0.5``,
      ``x_min = 1e-6``.  Measured through ``constraint_datasets``: ``1.094118e-03``,
      ``3.448574e-04``, ``1.070048e-04``, agreeing with the closed form to
      ``0.03%/0.29%/2.17%`` (bar :data:`NORM_LEVEL_BAND` = 10%, 4.6x headroom; the
      2.17% is FE discretization showing at ``x_min = 1e-8``).
    * **rate** -- ``10**a`` per decade, ``3.1623`` at ``a = 0.5``.  Measured
      ``3.1727`` and ``3.2228`` (bar :data:`NORM_RATE_BAND` = 1.3x, ~15x headroom).
      **This is the practical message**: four decades of ``x_min`` buy two of
      accuracy, so ``x_min`` is a weak lever and a matched-power head is the only
      strong one (``plans/low_x_head_diagnosis.md``).

    Oracle: ``B(a, beta+1)`` in closed form -- exactly ``32/35`` at the shipped
    ``(0.5, 3)``, asserted as such below so a mis-stated oracle cannot pass -- over
    the whole ``[0, 1]``, with no grid and no completion.  ``A1``, sharing nothing
    with ``pixel.kernels.lowx``.

    Acceptance, two in-memory mutations measured 2026-08-13, never touching ``src/``
    on disk:

    * a **flat completion on the valence fields** makes ``gamma = 0``, i.e.
      ``alpha + gamma = -1`` -- a mathematically divergent ``[0, x_min)`` integral --
      and fails the integrability assertion, *by name and before any kernel is
      built*.  Until ``kernels.lowx.check_low_x_integrable`` landed on 2026-08-13
      nothing rejected that pairing: it returned a finite ``6.691e-03`` here, 6.1x
      the correct level and growing with ``npts``.  The kernel now raises too, so
      this line's remaining value is diagnosis -- it names the field family and the
      completion instead of surfacing as an assembly error inside
      ``constraint_datasets``; the kernel-side guard, and the two gaps it leaves,
      are in ``tests/test_kernel_guards.py::
      test_low_x_integrability_guard_rejects_a_divergent_pairing``.  The
      **rate assertion cannot see it** -- the mutated error still falls at
      ``3.1606``/``3.1528`` per decade -- which is exactly why the integrability
      line is asserted separately rather than left to the rate.
    * **inflating ``low_x_quadrature_correction`` by 50%** (an ``src``-side head bug)
      fails the level assertion at ``got/pred = 0.5008``, while the rate stays at
      ``3.1831``/``3.2860``, inside the band.  So the level and the rate are
      independent detectors, and neither subsumes the other.

    The pre-existing tests here do also fail under both, and for reasons that are not
    convergence statements: the affine test's ``norm_exact`` formula *bakes in* the
    linear head (``q(x_min)`` contributed exactly), so a flat completion breaks the
    formula rather than being measured by it, and its ``2e-4`` bar is a quadrature
    bar that a 50% head change trivially exceeds.  What was untested before is the
    rate, the level against a head model, and the integrability of the pairing.

    Not covered: which ``a`` the real closure curves have, and nothing is validated
    against a closure fit -- this error cancels there by construction, since the same
    operator writes the target and reads it back.
    """
    cfg = suite.cfg
    exact = _valence_norm_exact()
    # The oracle, independently anchored: B(1/2, 4) = 32/35 exactly.
    np.testing.assert_allclose(exact, 32.0 / 35.0, rtol=1e-15, atol=0.0)

    gammas = {name: _norm_low_x_gamma(cfg.low_x_completion(name))
              for name in cfg.VALENCE_NORMS}
    assert len(set(gammas.values())) == 1, f"mixed valence completions: {gammas}"
    gamma = next(iter(gammas.values()))
    # cons_norm_* is Mellin(alpha=-1), so base_matrix's head integrand is
    # x**(gamma-1) at the origin and the [0, x_min) integral exists only for
    # gamma > 0.  Named here, on the config, so a violation reports the field family
    # rather than surfacing as an assembly error inside constraint_datasets;
    # kernels.lowx.check_low_x_integrable is the kernel-side guard.
    assert gamma > 0.0, (
        f"divergent low-x pairing for the valence norms: alpha=-1 with gamma="
        f"{gamma}, so the head integrand is x**{gamma - 1:+.1f} at the origin. "
        "See tests/test_kernel_guards.py::"
        "test_low_x_integrability_guard_rejects_a_divergent_pairing"
    )

    errors, predicted = [], []
    for x_min in NORM_SWEEP_X_MIN:
        grid = Grid(
            n_points=NORM_SWEEP_N_POINTS,
            spacing=cfg.GRID_SPACING,
            x_min=x_min,
        )
        fields = _fields_on(suite, grid)
        nodes = np.asarray(next(iter(fields.values())).nodes, dtype=float)
        # The head is an integral over [0, basis.domain[0]); if that is not the
        # x_min asked for, this sweep is not sweeping what it says it is.
        np.testing.assert_allclose(float(nodes.min()), x_min, rtol=1e-15, atol=0.0)
        curves = {
            name: _affine_scale(i) * nodes ** NORM_TRUTH_POWER
            * (1.0 - nodes) ** NORM_TRUTH_BETA
            for i, name in enumerate(cfg.ALL_FIELDS)
        }
        truth = {
            "x_nodes": nodes.tolist(),
            "curves": {name: curve.tolist() for name, curve in curves.items()},
        }
        scale = {name: _affine_scale(i) for i, name in enumerate(cfg.ALL_FIELDS)}

        seen = {}
        for dataset in suite.constraint_datasets(fields, truth):
            if not dataset.name.startswith("cons_norm_"):
                continue
            # cons_momentum spans Sigma + g jointly and cannot be unpacked as a
            # single contribution.  Skipped by name so that a NEW multi-field
            # constraint fails loudly here rather than being silently ignored.
            if dataset.name == "cons_momentum":
                continue
            (contribution,) = dataset.contributions
            name = contribution.field
            seen[name] = abs(
                float(dataset.mean[0]) / scale[name] / exact - 1.0
            )
        assert set(seen) == set(VALENCE_NUMBERS), f"norm datasets {sorted(seen)}"
        # Every valence field shares one completion and one basis, so they agree
        # exactly; taking the max keeps a future divergence visible rather than
        # averaged away.  Measured spread at x_min=1e-6: 0.0.
        errors.append(max(seen.values()))
        predicted.append(abs(_valence_norm_head_prediction(gamma, x_min)))

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f"lowering x_min made the norm worse: {errors}"
        )
    for x_min, err, pred in zip(NORM_SWEEP_X_MIN, errors, predicted):
        assert abs(err / pred - 1.0) < NORM_LEVEL_BAND, (
            f"x_min={x_min:.0e}: measured {err:.6e} vs closed-form head-model "
            f"error {pred:.6e} (ratio {err / pred:.6f})"
        )
    expected = 10.0 ** NORM_TRUTH_POWER
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert expected / NORM_RATE_BAND < ratio < expected * NORM_RATE_BAND, (
            f"{NORM_SWEEP_X_MIN[i]:.0e} -> {NORM_SWEEP_X_MIN[i + 1]:.0e} gained "
            f"x{ratio:.4f}, expected x{expected:.4f} (10**{NORM_TRUTH_POWER}) "
            f"within x{NORM_RATE_BAND}"
        )


def _norm_low_x_gamma(low_x_extension) -> float:
    """Power ``gamma`` of the completion ``L(x) = (x/x_min)**gamma`` below ``x_min``.

    ``"flat"`` is ``gamma = 0``; a ``{"kind": "power", "alpha": g}`` spec is ``g``.
    Deliberately does **not** route through
    ``pixel.kernels.lowx.normalize_low_x_extension``: reading the exponent back out
    of the module whose head is being measured would make the divergence assertion
    in the test above agree with whatever that module thinks, which is the one thing
    it must not do.
    """
    if low_x_extension == "flat":
        return 0.0
    assert low_x_extension["kind"] == "power", low_x_extension
    return float(low_x_extension["alpha"])


@pytest.mark.parametrize("suite", SUITES)
def test_constraint_targets_read_the_represented_truth_not_the_nominal_value(suite):
    """A truth offset by ``+0.25`` moves every target off its nominal value.

    The non-degenerate partner to the curve used elsewhere in this file: adding a
    constant to ``x**0.3 (1-x)**2`` makes it nonzero at *both* ``x = 0`` and
    ``x = 1``, which is what separates ``represented_target`` reading the injected
    curve from it silently returning ``nominal``
    (``closure_JAM_truth/fit.py:57-66``).  Under the original curve the two
    coincide at both endpoints (measured ``5.9e-20`` and ``0.0``), so every other
    test here passes either way for those two families.

    The oracle is the offset itself and needs no kernel: the fields' basis is
    interpolating at its own nodes, so the ``Delta`` targets at ``x = 0`` and
    ``x = 1`` are the curve's node values there, both exactly ``OFFSET`` (the
    ``x**0.3 (1-x)**2`` part vanishes at both).  Asserted exactly (measured
    difference ``0.0``); the nominal target stays ``0.0``, so a fallback is a
    ``0.25`` discrepancy, not a rounding one.  The valence norms are checked only
    for being finite and far from nominal -- with a curve that does not vanish at
    the origin the ``int q/x`` moment is log-divergent in exact arithmetic, so it
    has no closed form to compare against; that family's closed form lives in
    ``test_constraint_targets_match_an_analytic_affine_truth``.
    """
    cfg = suite.cfg
    OFFSET = 0.25
    grid = Grid(
        n_points=cfg.GRID_N, spacing=cfg.GRID_SPACING, x_min=cfg.X_MIN,
        include_origin=True,
    )
    fields = _fields_on(suite, grid)
    nodes = np.asarray(next(iter(fields.values())).nodes, dtype=float)
    curves = {
        name: _affine_scale(i) * nodes ** 0.3 * (1.0 - nodes) ** 2 + OFFSET
        for i, name in enumerate(cfg.ALL_FIELDS)
    }
    truth = {
        "x_nodes": nodes.tolist(),
        "curves": {name: curve.tolist() for name, curve in curves.items()},
    }

    n_delta = 0
    for dataset in suite.constraint_datasets(fields, truth):
        nominal = dataset.metadata.extras["nominal_target"]
        if dataset.name.startswith("cons_norm_"):
            assert np.isfinite(dataset.mean[0])
            assert abs(float(dataset.mean[0]) - nominal) > 1.0  # never the nominal
            continue
        if dataset.name == "cons_momentum":
            # A third family (2026-08-15): an integral constraint whose nominal is
            # 1.0, not 0.0.  Same claim as cons_norm_ -- the stored target is the
            # *represented* integral of this test's analytic truth, not the physical
            # fallback -- but the margin cannot be 1.0 here, because the nominal is
            # itself 1.0 and the represented value of (1 + 0.1 i) x^0.3 (1-x)^2
            # summed over sigma and g lands near 0.5.
            assert np.isfinite(dataset.mean[0])
            assert abs(float(dataset.mean[0]) - nominal) > 1.0e-3
            continue
        n_delta += 1
        assert nominal == 0.0  # the physical value both families would fall back to
        np.testing.assert_allclose(dataset.mean, [OFFSET], rtol=0.0, atol=0.0)
    # 9 endpoint + 7 origin datasets, all of them Delta evaluations of the curve.
    assert n_delta == len(cfg.ALL_FIELDS) + 7


@pytest.mark.parametrize("suite", SUITES)
def test_constraint_datasets_rejects_a_truth_missing_a_field_curve(suite):
    """A truth dict short one field's curve raises, and names the field.

    ``represented_target`` indexes ``truth["curves"][name]`` with no validation
    (``closure_JAM_truth/fit.py:58-64``).  What matters is not the diagnostic --
    it is a bare ``KeyError`` -- but that the miss is *loud*: a ``.get(name,
    nominal)``-style fallback would quietly hand back the nominal sum rules and
    produce a closure test whose constraints the injected truth does not
    satisfy, which is the failure this whole file exists to prevent.  Matched on
    the message (the missing field name) rather than the type alone, since a
    dict lookup anywhere else in the call would raise the same class.  The
    control below asserts the un-corrupted copy raises nothing, so this cannot
    pass against a builder that rejects everything.  Oracle F1.
    """
    fields = _fields_on(suite, suite.cfg.make_grid())
    nodes = np.asarray(next(iter(fields.values())).nodes, dtype=float)
    curve = (nodes ** 0.3 * (1.0 - nodes) ** 2).tolist()
    truth = {
        "x_nodes": nodes.tolist(),
        "curves": {name: list(curve) for name in suite.cfg.ALL_FIELDS},
    }
    suite.constraint_datasets(fields, truth)  # control: intact truth is accepted

    missing = dict(truth, curves={
        k: v for k, v in truth["curves"].items() if k != "v3"
    })
    with pytest.raises(KeyError, match="v3"):
        suite.constraint_datasets(fields, missing)


@pytest.mark.parametrize("suite", SUITES)
def test_vanishes_at_origin_rejects_an_unknown_field(suite):
    """``cfg.vanishes_at_origin`` refuses a name outside ``ALL_FIELDS``.

    The predicate is the single statement of the small-x limit for these suites
    -- both ``low_x_completion`` and the ``cons_origin_*`` guard read it -- and
    it is written to raise rather than to answer ``True`` by falling through its
    ``field not in SINGLET_GLUON_FIELDS`` body, which is what an unvalidated
    version would do for *any* typo.  That fall-through is the failure mode
    worth pinning: a misspelt ``"sigmaa"`` would silently acquire a vanishing
    origin and a power low-x completion.  Matched on the message; the two
    controls fix the answers a typo would land between.  Oracle F1.
    """
    with pytest.raises(KeyError, match="unknown closure field"):
        suite.cfg.vanishes_at_origin("sigmaa")
    assert suite.cfg.vanishes_at_origin("sigma") is False  # a real singlet name
    assert suite.cfg.vanishes_at_origin("v3") is True


def _eval_with(node, bindings):
    """Evaluate a tiny arithmetic AST -- constants, bound names, ``+``/``-``, ``*``.

    Enough to turn ``range(max_iterations + 1)`` into a number once
    ``max_iterations`` is bound to the literal parsed out of the same function, and
    small enough that it cannot execute anything. Anything richer raises, which is
    the point: an unrecognised loop bound must stop the test rather than be guessed.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        assert node.id in bindings, f"unbound name {node.id!r} in the loop bound"
        return bindings[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_with(node.left, bindings)
        right = _eval_with(node.right, bindings)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    raise AssertionError(f"unsupported loop-bound expression {ast.dump(node)}")


def _mcp_t0_map_defaults():
    """``{"max_iterations": 8, "tolerance": 1e-4, "passes": 9, "comparisons": 8}``.

    The first two are literals read out of ``_run_t0_map``; the last two are the
    loop's own arithmetic, evaluated from its ``range(...)`` call rather than typed.

    PIXEL has two independent t0 loops -- the MCP ``t0_map`` job
    (``src/pixel/mcp/runner.py``) and the closure suites' ``_iterate_t0`` -- and
    per ``guides/experimental_data_and_t0.md`` they "share the convergence
    criterion and nothing else".  The MCP defaults live as literals inside
    ``options.pop(...)`` calls, so there is no importable symbol to read; they
    are extracted from the module's AST rather than re-typed here, because a
    re-typed copy is exactly the thing that goes stale without failing.

    Parses the file rather than importing it: no import side effects, and no
    dependency on the MCP extra being installed.  A missing literal is an
    ``AssertionError``, never a skip -- there is no CI here, so a skip runs for
    nobody.

    ``passes`` used to be a typed ``+ 1`` at the call site, standing in for the
    loop's shape (``range(max_iterations + 1)``).  That made the *floor* derived
    while the arithmetic turning a budget into a pass count stayed a copy: widening
    the MCP loop -- ``range(max_iterations + 6)``, say -- would leave the floor at 9
    with the real budget at 14, and nothing would fail.  It is now evaluated from
    the ``range`` call itself, with ``max_iterations`` bound to the literal parsed
    above, so both halves of the comparison come from the same source.
    ``comparisons`` is ``passes - 1``: the first pass has ``previous is None`` and
    nothing to compare against (``runner.py``'s ``if previous is None``).
    """
    source = pathlib.Path(pixel.__file__).parent / "mcp" / "runner.py"
    assert source.exists(), f"{source} is gone; the floor below has no source"
    function = next(
        (
            node
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name == "_run_t0_map"
        ),
        None,
    )
    assert function is not None, "_run_t0_map not found in mcp/runner.py"
    defaults = {}
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in ("max_iterations", "tolerance")
        ):
            defaults[node.args[0].value] = ast.literal_eval(node.args[1])
    assert set(defaults) == {"max_iterations", "tolerance"}, defaults

    # The loop that spends the budget: `for iteration in range(max_iterations + 1)`.
    loops = [
        node.iter
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    ]
    assert len(loops) == 1, f"expected one range() loop in _run_t0_map, found {len(loops)}"
    bounds = [_eval_with(arg, {"max_iterations": defaults["max_iterations"]})
              for arg in loops[0].args]
    defaults["passes"] = len(range(*bounds))
    # The first pass has nothing to compare against (`if previous is None`).
    defaults["comparisons"] = defaults["passes"] - 1
    return defaults


@pytest.mark.parametrize("suite", SUITES)
def test_t0_iteration_ceiling_is_at_least_the_mcp_t0_map_budget(suite):
    """``cfg.T0_MAX_ITERATIONS`` (12) is at least the MCP t0 job's own budget.

    PROVENANCE OF THE FLOOR, recovered 2026-08-14.  This test used to assert a
    bare ``>= 9`` and was named "...covers observed small suite convergence",
    implying 9 had been measured from a real fit.  It had not: ``cfg``,
    ``fit.py`` and ``closure_logs/`` record no t0 iteration count anywhere (both
    small-suite logs on disk are failed runs that never reach ``fit.py``'s
    ``"DIS t0 ready ... iterations="`` line), so nobody reading it could tell a
    measurement from a guess.

    The only in-repo quantity that ``9`` matches is the *sibling implementation's
    own budget*: ``_run_t0_map`` defaults ``max_iterations = 8`` and loops
    ``range(max_iterations + 1)``, i.e. **9 passes**, which is how
    ``guides/experimental_data_and_t0.md``'s side-by-side table prints it.  The
    floor is therefore derived from :func:`_mcp_t0_map_defaults` rather than
    typed, and the numeric bar is unchanged at 9 -- this records where the number
    comes from, it does not move it.

    The comparison is only meaningful while the two loops really do share their
    convergence criterion, which that guide states and this test now checks:
    ``T0_TOLERANCE`` must equal the MCP job's default ``tolerance``.  If someone
    tightens one and not the other, the budgets stop being comparable, and this
    fails loudly at that assertion instead of silently comparing apples to
    oranges.

    THE TWO READINGS OF "BUDGET", reconciled 2026-08-14 -- they were previously
    stated here as unresolved, and the resolution is that they do not disagree
    about the verdict, only about the count.  Reading both loops:

    * ``_run_t0_map`` (``runner.py:253``) runs ``range(max_iterations + 1)``, so
      **9 passes**: one initial fit plus 8 refits, and **8** convergence
      comparisons (the first pass has ``previous is None``).
    * ``_iterate_t0`` (``fit.py:481-492``) takes one prediction *before* the loop
      -- the analogue of the MCP job's first pass -- then runs
      ``range(1, max_iter + 1)``, so ``T0_MAX_ITERATIONS`` counts **refits**: 12
      refits, 12 comparisons, 13 predictions in all.

    So refits are 12 vs 8, comparisons 12 vs 8, and total passes 13 vs 9.  The
    closure loop is the more generous budget under *every* reading, and the bar
    asserted below is the strictest of them -- expressed as ``max(...)`` over the
    three MCP counts rather than as a claim in prose, so "the stricter one is
    asserted" is executable.  The numeric bar is unchanged at 9.

    WHAT THIS STILL DOES NOT ESTABLISH: that any real fit converges inside 12.
    Nothing here runs ``_iterate_t0``; the loop's control flow is exercised on a
    scripted sequence by
    ``test_t0_iteration_stops_on_tolerance_and_never_exceeds_the_ceiling``
    below, and the real-fit iteration count needs a full ``build_analysis`` and
    generated truth on disk (report ``M04``) -- a closure run, which is outside
    what may be run here.  A ceiling that is generous relative to a sibling
    implementation is not evidence that either ceiling is enough.
    """
    defaults = _mcp_t0_map_defaults()
    # The shared criterion, without which the budget comparison means nothing.
    assert suite.cfg.T0_TOLERANCE == defaults["tolerance"] == 1e-4
    # Every reading of the MCP job's budget, all derived from its own source:
    # refits (8), convergence comparisons (8), total passes (9).  The strictest is
    # the floor; taking the max makes that choice executable rather than prose.
    floor = max(
        defaults["max_iterations"], defaults["comparisons"], defaults["passes"]
    )
    # A floor of 0 would make the real assertion vacuous, and a budget parsed as 0
    # means the parse broke rather than that the sibling job stopped iterating.
    assert floor >= 1, defaults
    # Measured 2026-08-14: floor 9, T0_MAX_ITERATIONS 12 -- 33% headroom.
    assert suite.cfg.T0_MAX_ITERATIONS >= floor, (suite.cfg.T0_MAX_ITERATIONS, defaults)


def _t0_stub_suite(suite, monkeypatch, sequence):
    """Point ``_iterate_t0``'s two collaborators at a scripted sequence.

    ``_dis_predictions`` yields ``sequence`` one entry per call (the first before
    the loop, one per iteration after); ``build_analysis`` becomes a recorder.
    Returns the ``(q_key, mode, t0-was-passed)`` call log.
    """
    calls = []
    remaining = list(sequence)

    def fake_build(q_key, mode, *, use_kernel_cache=True, t0=None):
        calls.append((q_key, mode, t0 is not None))
        return SimpleNamespace(compile=lambda: "model"), None

    def fake_predictions(model):
        return {"dis_x": np.array([remaining.pop(0)])} if remaining else {}

    monkeypatch.setattr(suite, "build_analysis", fake_build)
    monkeypatch.setattr(suite, "_dis_predictions", fake_predictions)
    return calls


@pytest.mark.parametrize("suite", SUITES)
def test_t0_iteration_stops_on_tolerance_and_never_exceeds_the_ceiling(suite, monkeypatch):
    """The t0 loop stops at the first shift within tolerance, and caps at the ceiling.

    Drives ``_iterate_t0`` (``closure_JAM_truth/fit.py:460-491``) over a scripted
    prediction sequence ``t0_k = 1000 (1 + 10**-k)`` with its two expensive
    collaborators stubbed, so the loop's own control flow is what is measured --
    the ceiling test above only reads the constant.  The ``1000`` scale is not
    cosmetic: it separates the relative shift the loop must use from an absolute
    one, which on an ``O(1)`` sequence would stop at the same iteration and hide
    the difference.  The oracle is arithmetic on
    that sequence, not on the implementation: the successive relative shifts are
    ``4.500e-01, 8.182e-02, 8.911e-03, 8.991e-04, 8.999e-05``, so the *fifth*
    is the first at or below ``T0_TOLERANCE = 1e-4`` and the loop must return
    ``n_iterations = 5`` with ``t0 = 1000 (1 + 1e-5)``.  The tolerance's regime is
    asserted first, so changing ``T0_TOLERANCE`` fails here loudly instead of
    silently moving the expected index.  Would catch: an off-by-one in the
    ``range(1, max_iter + 1)`` bookkeeping, a ``<``/``>`` inversion on the
    convergence test, an absolute-instead-of-relative shift, the ceiling not
    being honoured, or the ``exp``/``both`` -> ``"dis"`` sub-fit rebuild being
    dropped (which would send the bilinear H-S sector through MAP).  Oracle A1
    on the loop; says nothing about how many iterations a *real* fit needs.
    """
    cfg = suite.cfg
    # The stop index below is only 5 while the tolerance sits between the 5th
    # and 4th shifts of this sequence.
    assert 8.9991e-05 <= cfg.T0_TOLERANCE < 8.9910e-04

    calls = _t0_stub_suite(
        suite, monkeypatch, [1000.0 * (1.0 + 10.0 ** -k) for k in range(8)]
    )
    t0, n_iterations = suite._iterate_t0("mc", "dis", "model")
    assert n_iterations == 5
    np.testing.assert_allclose(
        t0["dis_x"], [1000.0 * (1.0 + 1.0e-5)], rtol=0.0, atol=0.0
    )
    assert len(calls) == 5  # one analysis rebuild per iteration, none wasted
    assert all(mode == "dis" and passed_t0 for _, mode, passed_t0 in calls)

    # A sequence that never settles must stop at the ceiling, not run away.
    alternating = [1.0 + 0.5 * (-1) ** k for k in range(cfg.T0_MAX_ITERATIONS + 4)]
    calls = _t0_stub_suite(suite, monkeypatch, alternating)
    _, n_iterations = suite._iterate_t0("mc", "dis", "model")
    assert n_iterations == cfg.T0_MAX_ITERATIONS
    assert len(calls) == cfg.T0_MAX_ITERATIONS

    # A mixed mode derives t0 from the linear DIS sub-fit, never from the
    # bilinear model it was handed.
    calls = _t0_stub_suite(suite, monkeypatch, [1.0, 1.0])
    suite._iterate_t0("mc", "both", "bilinear-model")
    assert [mode for _, mode, _ in calls] == ["dis", "dis"]
    assert calls[0][2] is False  # the sub-fit rebuild carries no t0 yet

    # No DIS dataset carries multiplicative uncertainties -> no iteration at all.
    calls = _t0_stub_suite(suite, monkeypatch, [])
    assert suite._iterate_t0("mc", "dis", "model") == ({}, 0)
    assert calls == []


@pytest.mark.parametrize("suite", SUITES)
def test_mellin_moment_convention_forces_momentum_density_on_every_field(suite):
    """``build_mellin``'s unconditional ``momentum_density=True`` is required, not lax.

    The audit flagged ``datasets.build_mellin`` for passing
    ``momentum_density=True`` for *every* field while taking the per-field
    ``cfg.low_x_completion(field)``, and asked for a per-field decision.  Measured
    here: a per-field ``momentum_density`` would be **wrong physics**, and the
    completion it is paired with is **numerically irrelevant** at the moment
    orders this suite builds.  Both halves are asserted, because the first alone
    would not explain why the pairing is safe.

    **Why the flag must stay on for every field.**  The fitted fields are momentum
    densities ``q = x f``, and a record labelled order ``n`` means the moment of
    ``f``: ``<x^(n-1)>_f = int x^(n-1) f dx = int x^(n-2) q dx``, i.e. the
    ``alpha = -1`` weight ``momentum_density=True`` selects.  Nothing about that
    depends on the field's small-``x`` behaviour -- it is the storage convention,
    the same for ``sigma``/``g`` as for the valence combinations.  With
    ``momentum_density=False`` the same row computes ``<x^n>_f``: the right
    integral of the wrong moment, silently off by one power.

    Oracle ``A2``, closed form and independent of the kernel: for
    ``q = x^p (1-x)^b`` the moment is ``B(p + n - 1, b + 1)``, evaluated in
    ``mpmath``.  ``p = 0.5`` is the measured physical valence exponent
    (``plans/low_x_head_diagnosis.md``, pooled median ``0.475``), so the fixture is
    non-degenerate in the dimension under test: ``B(p+n-1, b+1)`` and
    ``B(p+n, b+1)`` -- the ``momentum_density=False`` answer -- differ by a factor
    measured below and asserted to be large, so an off-by-one power cannot hide.

    **Why the completion pairing is safe here.**  A Mellin row of order ``n``
    carries its own ``x^(n-1)``, so the effective low-x weight is ``n - 2``, not
    ``-1``; this suite builds ``n = 2`` (C-odd) and ``n = 3`` (C-even) only, never
    ``n = 1``.  Asserted by swapping the completion for two others -- ``flat``
    (what ``sigma``/``g`` actually get) and a *rising* ``power(alpha=-0.3)``
    (the shape ADDENDUM 2 of the low-x diagnosis measures for JAM ``t8``/``t15``)
    -- and requiring the folded moment to be unchanged.  Measured on this fixture:
    ``4.9e-09`` and ``9.1e-09`` at ``n = 2``, ``6.0e-15`` and ``9.1e-15`` at
    ``n = 3``.  The bar is set 100x above the largest of those, so this is a
    measurement with headroom, not a threshold tuned to pass.

    Does **not** establish that ``cfg.low_x_completion`` picks the right shape.
    It does not, for ``t8``/``t15`` -- see the low-x diagnosis -- but the exposure
    is in the ``alpha = -1`` pseudo-ITD and ``cons_norm_*`` rows, not here.
    """
    cfg = suite.cfg
    grid = cfg.make_grid()
    basis = Field.create("q", grid, element_type=cfg.ELEMENT_TYPE).basis
    nodes = np.asarray(grid.as_array(), dtype=float)

    p, b = 0.5, 3.0
    q_curve = nodes**p * (1.0 - nodes) ** b

    def moment_order(field):
        """Configured Mellin order, both spellings.

        The full suites expose ``cfg.mellin_order_for``; the ``_small`` pair has
        no such helper and inlines the same C-parity choice in
        ``generate.lattice_layout``.  Read both rather than skipping the ``_small``
        suites -- a package-level gap in a parametrization is exactly what let the
        DY second-moment bug live (see ``tests/test_closure_truth_representable.py``).
        """
        chooser = getattr(cfg, "mellin_order_for", None)
        if chooser is not None:
            return float(chooser(field))
        return float(
            cfg.CP_EVEN_MELLIN_ORDER
            if cfg.field_cparity(field) == "even"
            else cfg.CP_ODD_MELLIN_ORDER
        )

    for field in cfg.ALL_FIELDS:
        n = moment_order(field)
        assert n >= 2.0, (field, n)  # n = 1 is the one order this pairing forbids
        completion = cfg.low_x_completion(field)

        # momentum_density=True -> alpha=-1; False -> alpha=0.  Both spellings
        # written here, not read from the builder, so this is a real comparison.
        got = float(
            np.asarray(
                kernels.Mellin(alpha=-1.0, low_x_extension=completion).matrix(
                    np.array([n]), basis
                ),
                dtype=float,
            ).reshape(-1)
            @ q_curve
        )
        wrong = float(
            np.asarray(
                kernels.Mellin(alpha=0.0, low_x_extension=completion).matrix(
                    np.array([n]), basis
                ),
                dtype=float,
            ).reshape(-1)
            @ q_curve
        )

        exact = float(mpmath.beta(p + n - 1.0, b + 1.0))
        exact_off_by_one = float(mpmath.beta(p + n, b + 1.0))
        # 1e-4 is a quadrature bar on a 145-node grid, not a transcription bar.
        assert abs(got / exact - 1.0) < 1.0e-4, (field, got, exact)
        # Non-degeneracy: the two conventions are far apart, so the check above
        # could not have been satisfied by the wrong one.
        assert abs(exact_off_by_one / exact - 1.0) > 0.2, (field, n)
        assert abs(wrong / exact - 1.0) > 0.2, (field, wrong, exact)

        # The completion is immaterial at these orders: swap it for a flat and a
        # RISING one and the folded moment must not move.
        for alternative in ("flat", {"kind": "power", "alpha": -0.3}):
            swapped = float(
                np.asarray(
                    kernels.Mellin(alpha=-1.0, low_x_extension=alternative).matrix(
                        np.array([n]), basis
                    ),
                    dtype=float,
                ).reshape(-1)
                @ q_curve
            )
            assert abs(swapped / got - 1.0) < 1.0e-6, (field, n, alternative)


def test_the_four_suites_still_share_one_constraint_builder():
    """The four packages are one code path -- assert that, don't assume it.

    **This test is why ``ALL_SUITES`` exists and must keep all four entries.**  It is
    the only reader of that tuple; ``SUITES``, which every other test here is
    parametrized over, is the NNPDF pair (2026-08-14, owner's instruction).  Trimming
    ``ALL_SUITES`` to match would leave ``others`` holding one suite and this test
    comparing NNPDF-full to NNPDF-small -- it would still pass, and it would no longer
    be able to see a JAM copy diverge, which is the entire failure it exists for.

    That relationship also inverts the old argument for the 4x sweep.  It used to be
    that ``constraint_datasets`` was byte-identical and the config constants matched
    "checked by ``diff`` during the audit, then written into the module docstring as
    prose" -- and prose goes stale silently, so the wide parametrization was insurance.
    Now the identity is *asserted* here, on the source text of the builder plus the 14
    constants the file's assertions actually depend on, which is a stronger guard than
    running the same numbers through four copies and getting the same answer.  The
    behavioural sweep can therefore be narrow while this stays wide.

    Oracle A3 (identity between objects the suites themselves build); structural only,
    no numerics.
    """
    reference, *others = ALL_SUITES
    ref_source = inspect.getsource(reference.constraint_datasets)
    keys = (
        "GRID_N", "GRID_SPACING", "X_MIN", "ELEMENT_TYPE", "ALL_FIELDS",
        "SINGLET_GLUON_FIELDS", "VALENCE_NORMS", "ENDPOINT_X",
        "CONSTRAINT_ENDPOINT_SIGMA", "CONSTRAINT_NORM_SIGMA",
        "CONSTRAINT_ORIGIN_SIGMA", "LOW_X_LINEAR_POWER", "T0_MAX_ITERATIONS",
        "T0_TOLERANCE",
    )
    ref_cfg = {k: getattr(reference.cfg, k) for k in keys}
    for suite in others:
        assert inspect.getsource(suite.constraint_datasets) == ref_source
        assert {k: getattr(suite.cfg, k) for k in keys} == ref_cfg


# -- the x*f(x=0) = 0 origin constraint ---------------------------------------


def _fields_on(suite, grid):
    """Build one ``Field`` per ``suite.cfg.ALL_FIELDS`` name, all sharing ``grid``."""
    return {
        name: Field.create(name, grid, element_type=suite.cfg.ELEMENT_TYPE)
        for name in suite.cfg.ALL_FIELDS
    }


@pytest.mark.parametrize("suite", SUITES)
def test_origin_constraint_is_off_when_the_grid_never_reaches_zero(suite):
    """On today's ``x_min > 0`` closure grid, no ``cons_origin_*`` dataset exists.

    Exercises the guard at ``closure_JAM_truth/fit.py:100`` (only build origin
    constraints when the grid's minimum node is exactly ``0.0``) through its
    effect: the returned dataset list has no name starting ``"cons_origin_"``.  A
    key-set check on ``constraint_datasets``'s output, independent of the
    recompute-style checks elsewhere in this file.  Would catch the guard being
    removed or inverted, which would silently start emitting the all-zero
    ``Delta`` row at ``x = 0`` this suite exists to avoid
    (``pixel.kernels.common.Delta.base_matrix``,
    ``src/pixel/kernels/common.py:466-525``).  The ``low_x_completion`` dispatch
    check that follows is only partly independent: the "power" vs "flat" *kind* is
    a real per-field check, but the expected ``alpha`` reuses
    ``cfg.LOW_X_LINEAR_POWER``, the same constant the source returns, so a wrong
    ``alpha`` value would not be caught.
    """
    grid = suite.cfg.make_grid()
    assert float(np.asarray(grid.as_array()).min()) > 0.0  # today's closure grid
    fields = _fields_on(suite, grid)
    # The premise, measured rather than asserted in prose: on this grid a Delta
    # row at x = 0 is *identically* zero (measured max|row| = 0.0, nnz = 0), so
    # an origin dataset here would look like a constraint and constrain nothing.
    # That is why the builder must decline; if Delta ever started extrapolating
    # below x_min this test's rationale would change and this line says so.
    zero_row = np.asarray(
        kernels.Delta().matrix(np.array([0.0]), fields["t3"].basis), dtype=float
    )
    assert np.count_nonzero(zero_row) == 0
    datasets = suite.constraint_datasets(fields)
    assert not [d for d in datasets if d.name.startswith("cons_origin_")]
    # The vanishing limit is carried by the low-x completion instead.
    for name in suite.cfg.ALL_FIELDS:
        completion = suite.cfg.low_x_completion(name)
        if suite.cfg.vanishes_at_origin(name):
            # alpha = 1 stated here, not read back from cfg.LOW_X_LINEAR_POWER:
            # x*f -> 0 linearly is the physics claim, and reading the constant
            # the source returns could not catch it being changed.
            assert completion == {"kind": "power", "alpha": 1.0}
        else:
            assert completion == "flat"  # sigma/g may tend to a nonzero constant


@pytest.mark.parametrize("suite", SUITES)
def test_origin_constraint_switches_on_for_the_vanishing_fields(suite):
    """With ``include_origin=True``, exactly the 7 vanishing fields get an
    origin dataset pinned at ``nu = mean = 0``, and its ``Delta`` row is one-hot.

    The field set is checked against ``{n: vanishes_at_origin(n)}`` and against
    the independently spelled-out ``ALL_FIELDS - SINGLET_GLUON_FIELDS`` (``= 7``,
    sigma and g excluded).  ``vanishes_at_origin`` is itself ``field not in
    SINGLET_GLUON_FIELDS`` (``closure_JAM_truth/config.py``), so this mainly
    confirms ``constraint_datasets``'s per-field guard calls that predicate rather
    than a different or hardcoded list.  ``dataset.cov[0, 0] ==
    approx(CONSTRAINT_ORIGIN_SIGMA**2)`` is tautological (both sides read the same
    constant); ``nu``/``mean == 0`` is a real but narrow pin, since ``truth=None``
    here takes ``represented_target``'s early-return branch.  The strongest check
    is the row: ``count_nonzero(row) == 1`` and ``row.sum() == approx(1.0)``
    verify the cubic-spline basis evaluates to a unit vector at its own ``x = 0``
    node -- an independent, non-tautological property of ``ProductBasis`` -- and
    is what rules out the silent all-zero ``Delta`` row this suite exists to
    catch.
    """
    cfg = suite.cfg
    grid = Grid(
        n_points=cfg.GRID_N, spacing=cfg.GRID_SPACING, x_min=cfg.X_MIN,
        include_origin=True,
    )
    assert float(np.asarray(grid.as_array()).min()) == 0.0
    # Note the coupling: constraint_datasets keys off the *fields it is given*,
    # while cfg.low_x_completion keys off cfg.make_grid().  They agree in
    # build_analysis, which builds the fields from make_grid(); switching the
    # suite to include_origin therefore means editing make_grid, not just
    # passing a different grid here.
    assert cfg.low_x_completion("t3") is not None  # make_grid() still has a gap

    fields = _fields_on(suite, grid)
    datasets = suite.constraint_datasets(fields)
    origin = {
        d.name.replace("cons_origin_", ""): d
        for d in datasets
        if d.name.startswith("cons_origin_")
    }
    # 7 of the 9 fields: everything except the singlet and gluon momentum
    # densities, whose x*f may tend to a nonzero constant at small x.
    assert set(origin) == {n for n in cfg.ALL_FIELDS if cfg.vanishes_at_origin(n)}
    assert set(origin) == set(cfg.ALL_FIELDS) - set(cfg.SINGLET_GLUON_FIELDS)
    # 7 = len(ALL_FIELDS) - len(SINGLET_GLUON_FIELDS) = 9 - 2; redundant with the
    # set-equality line above, kept as a direct count.
    assert len(origin) == 7

    for name, dataset in origin.items():
        # nu/mean equal 0.0 by construction: constraint_datasets passes a literal
        # 0.0 for both, and truth=None here takes represented_target's
        # early-return branch -- an exact pin, not a derived bound.
        np.testing.assert_allclose(dataset.nu, [0.0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(dataset.mean, [0.0], rtol=0.0, atol=0.0)
        # 1e-4 in q units, stated here rather than read back from
        # CONSTRAINT_ORIGIN_SIGMA: "near-hard" is the claim, and reading the
        # same constant on both sides could not catch it being loosened.
        # 1e-8 in q units since 2026-08-15 (was 1e-4), still stated here rather
        # than read back from CONSTRAINT_ORIGIN_SIGMA: reading the same constant on
        # both sides could not catch it being loosened, which is the whole claim.
        # An UPPER BOUND, not equality.  The intent is right -- a literal, so a
        # loosening cannot be silently mirrored by reading
        # ``CONSTRAINT_ORIGIN_SIGMA`` on both sides -- but equality also fires on
        # *tightening*, which is not a regression, and it pinned 1e-8 through two
        # subsequent retunes (1e-8 -> 1e-6 -> 1e-4, each with its own measurement
        # in config.py).  The claim this test defends is "near-hard", so the bar
        # is that the variance is no looser than the loosest value ever measured
        # to work: SD 1e-4, from the 2026-08-16 exp scan (1e-6 FAIL, 1e-5 PASS,
        # 1e-4 PASS).
        assert dataset.cov[0, 0] <= 1.0e-4 ** 2, (
            f"origin constraint looser than the measured feasible bound: "
            f"variance {dataset.cov[0, 0]:.3e} > {1.0e-4 ** 2:.3e}"
        )
        # The Delta row must actually select the origin node, or the constraint
        # is the silent zero row this test exists to rule out.
        # cons_momentum spans Sigma + g jointly and cannot be unpacked as a
        # single contribution.  Skipped by name so that a NEW multi-field
        # constraint fails loudly here rather than being silently ignored.
        if dataset.name == "cons_momentum":
            continue
        (contribution,) = dataset.contributions
        row = np.asarray(
            contribution.kernel.matrix(dataset.nu, fields[name].basis), dtype=float
        )
        assert np.count_nonzero(row) == 1  # exactly the x = 0 basis element
        # Exact: the basis is interpolating at its own nodes (measured
        # row.sum() - 1 == 0.0), so this is a bit-level pin, not a tolerance.
        assert row.sum() == 1.0


@pytest.mark.parametrize("suite", SUITES)
def test_origin_constraint_targets_the_represented_truth(suite):
    """Same recompute check as the first test, extended to origin datasets.

    Claims (comment below) that the origin constraint is satisfied by the
    *represented* injected truth rather than the nominal ``0.0``.  As measured,
    this is unfalsifiable for the origin family specifically: the synthetic curve
    ``x**0.3 * (1-x)**2`` is exactly zero at ``x = 0``, so the represented and
    nominal targets coincide there (measured ``represented == nominal == 0.0``)
    regardless of whether ``represented_target`` reads the curve or silently
    falls back to ``nominal``.  Only the valence-norm sub-case, folded into the
    same loop, exercises a real represented-vs-nominal gap (measured
    ``represented=3.29`` vs ``nominal=1.0`` for ``v3``).  The closing assertion is
    vacuously true for every non-origin dataset; for origin datasets it checks
    ``nominal_target == 0.0``, which is hardcoded at
    ``closure_JAM_truth/fit.py:105`` and, again, not derived from the curve.  Same
    same-kernel-object caveat as
    ``test_near_hard_constraints_match_truth_in_the_fit_basis`` applies to the
    ``atol=1e-12`` recompute above.
    """
    # Same contract as the x = 1 endpoint: a closure constraint must be satisfied
    # by the *represented* injected truth, not by the nominal physical value.
    cfg = suite.cfg
    grid = Grid(
        n_points=cfg.GRID_N, spacing=cfg.GRID_SPACING, x_min=cfg.X_MIN,
        include_origin=True,
    )
    fields = _fields_on(suite, grid)
    nodes = np.asarray(next(iter(fields.values())).nodes, dtype=float)
    curves = {
        name: (1.0 + 0.1 * i) * nodes ** 0.3 * (1.0 - nodes) ** 2
        for i, name in enumerate(cfg.ALL_FIELDS)
    }
    truth = {
        "x_nodes": nodes.tolist(),
        "curves": {name: curve.tolist() for name, curve in curves.items()},
    }
    for dataset in suite.constraint_datasets(fields, truth):
        # cons_momentum spans Sigma + g jointly and cannot be unpacked as a
        # single contribution.  Skipped by name so that a NEW multi-field
        # constraint fails loudly here rather than being silently ignored.
        if dataset.name == "cons_momentum":
            continue
        (contribution,) = dataset.contributions
        prediction = np.asarray(
            contribution.kernel.matrix(dataset.nu, fields[contribution.field].basis),
            dtype=float,
        ) @ curves[contribution.field]
        # Same same-kernel-object recompute as the first test (module docstring);
        # exact float64 reproduction, not a physics accuracy bound.
        np.testing.assert_allclose(prediction, dataset.mean, rtol=0.0, atol=1e-12)
        assert dataset.metadata.extras["nominal_target"] == 0.0 or not dataset.name.startswith(
            "cons_origin_"
        )
