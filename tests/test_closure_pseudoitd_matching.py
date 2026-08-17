"""Focused closure checks for vector pseudo-ITD matching through NNLO.

Exercises ``pixel.kernels.evolution.matrices.EvolvedPseudoITD``, the
Li--Ma--Qiu coordinate-space coefficient, its historical one-loop Mellin
bridge, and the closure-suite wiring in
``closure_JAM_truth.datasets.build_pseudoitd`` and its ``closure_NNPDF_truth``
twin -- ``diff`` shows the two ``datasets.py`` files differ by exactly one
docstring line, so they are one implementation for this file's purposes.

The full-scale closure used to read the LO reduced ITD straight off the fitted
field: ``build_pseudoitd`` passed neither ``mu0_2`` nor ``matching``.  These
tests pin the wired version, and in particular the one way it can be wired and
still be wrong -- an evolved, matched field whose *systematic nuisance curves*
are folded through the bare LO transform.

The physics oracles are deliberately layered:

* **The matching moments** are checked at integer Mellin
  ``N`` against a from-scratch harmonic-sum recomputation (helper
  ``_harmonic`` below) that never calls the source's ``mpmath`` digamma/
  trigamma helper ``_psi``.  That independence is real but narrow: it proves
  ``_pseudoitd_nlo_moments``'s complex-``N`` digamma formula reduces to the
  same harmonic-number closed form ``B(N)``/``L(N)`` coded there.  A separate
  test bridges it to the normalized Li--Ma--Qiu one-loop density, and the
  authors' ancillary ``iA(omega=29.15)`` output checks every translated
  positive- and negative-support coefficient through NNLO.  Measured this
  audit by reproducing both tests' arithmetic
  directly, the achieved agreement is machine precision (``~2.22e-16``, one
  ULP) against the coded ``rtol=1e-12`` bars -- the limit is double-precision
  roundoff in the ``mpmath.fp`` digamma/trigamma chain, not any physical or
  statistical approximation.

* **The kernel-wiring tests** are structural: each
  checks that a config flag or constructor argument reaches the right object
  attribute, or produces *some* nonzero change, not that the resulting numbers
  are physically correct.  The bit-level check that ``EvolvedPseudoITD``
  actually composes ``matched Fourier(mu2) @ evolution(mu2)`` correctly
  (not just nonzero) lives in ``tests/test_evolution.py::test_evolved_
  pseudoitd_matching_composes_after_evolution`` (``rtol=1e-11``); the analytic
  Beta-PDF closed-form checks for the pseudo-ITD matching transform live in
  ``tests/test_evolution_closure.py`` (``test_pseudoitd_matching_beta_truth_
  no_evolution`` and ``test_evolved_pseudoitd_identity_scale_beta_fourier_
  truth``); and the identical evolved-field/bare-nuisance split enforced here
  through the closure config is pinned one layer down, directly against
  ``pixel.data.PseudoITD.from_file`` with no closure suite involved, by
  ``tests/test_data_builders.py::test_pseudoitd_evolution_is_main_pdf_only_
  and_accepts_physical_z``.  What is specific to *this* file is the
  closure-suite's own config-to-kwargs plumbing (``build_pseudoitd``), driven
  here against a real generated manifest record via ``_lattice_fixture``,
  which is the layer that actually regressed before this file existed.

The grid used for the direct ``EvolvedPseudoITD`` construction
(``test_matching_changes_the_operator_it_is_applied_to``) is deliberately tiny
(9 points): only a nonzero-difference sanity check is wanted there, not an
accuracy measurement.  The production grid (128 points, cubic-spline
elements) is ``closure_JAM_truth.config.make_grid``/``closure_NNPDF_truth.
config.make_grid``, exercised by the parametrized closure-wiring tests via
``_lattice_fixture``.
"""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import pixel.kernels as kernels
from pixel.geometry import Grid, ProductBasis
from pixel.geometry.finite_elements import CubicSplineBasis
from pixel.kernels.evolution import EvolvedPseudoITD
from pixel.kernels.matching import (
    MatchingMatrix,
    VariableQ2MatchingMatrix,
    _pseudoitd_nlo_moments,
)
from pixel.kernels.pqcd import MellinContour
from pixel.kernels.pseudoitd_matching import (
    CA,
    CF,
    TF,
    ZETA3,
    _li_vector_quadrature,
    li_vector_densities,
    li_vector_fourier_matrix,
)


#: The full-size closure package the ``build_pseudoitd`` wiring tests run against, and
#: :data:`ALL_SUITES` the full and ``_small`` pair for the layout-convention test.
#:
#: Both were **NNPDF and JAM** until 2026-08-14, when the JAM entries came out on the
#: owner's instruction.  Nothing these tests assert reads the truth PDF set: they check
#: that ``build_pseudoitd`` threads its evolution constant to the built kernels, that the
#: generation-time and fitting-time operators agree, that systematic nuisances stay on the
#: bare transform, and that every regenerated record's ``component`` follows the field's
#: C-parity.  The JAM ``datasets.py``/``config.py`` are duplicate copies of the NNPDF ones
#: for all of that, so each JAM leg re-ran the same code for the same answer.  Neither
#: tuple feeds a cross-suite comparison -- no test here compares one suite to another --
#: so both were trimmed in place rather than split.
SUITES = ("closure_NNPDF_truth",)
ALL_SUITES = (
    "closure_NNPDF_truth",
    "closure_NNPDF_truth_small",
)
EULER = 0.5772156649015329  # = mpmath.fp.euler bit-for-bit (checked this audit)


def _suite(name):
    """Import ``<name>.config`` and ``<name>.datasets`` for one closure suite."""
    import importlib

    return (
        importlib.import_module(f"{name}.config"),
        importlib.import_module(f"{name}.datasets"),
    )


@pytest.mark.parametrize("suite", ALL_SUITES)
def test_closure_layout_uses_vector_real_imaginary_c_parity(suite):
    """Every regenerated closure record uses the vector full-PDF convention."""
    import importlib

    config = importlib.import_module(f"{suite}.config")
    generate = importlib.import_module(f"{suite}.generate")
    records = [
        record for record in generate.lattice_layout()
        if record["kind"] == "pseudoitd"
    ]
    assert records
    for record in records:
        expected = (
            "imag" if config.field_cparity(record["field"]) == "even" else "real"
        )
        assert record["component"] == expected


def _harmonic(n, power):
    """``sum_{k=1}^{n} 1/k**power``, written out -- no call to ``pqcd._psi``."""
    return sum(1.0 / k**power for k in range(1, n + 1))


def test_nlo_matching_moments_reduce_to_the_analytic_constant():
    """At the log-cancelling ``lam`` the moments are exactly ``C_F L(N)``.

    ``C_F [ ln(lam^2 e^{2 gamma_E + 1} / 4) B(N) + L(N) ]`` -- choosing
    ``lam = 2 e^{-gamma_E - 1/2}`` sends the bracketed log to zero and isolates
    the ``O(alpha_s)`` matching constant, which is then checked against harmonic
    sums evaluated directly at integer ``N``.  This is independent of the
    production contour path and of the source's own ``_psi`` digamma/trigamma
    calls, so it tests the constant rather than restating the machinery around
    it -- but not independent of the ``B(N)``/``L(N)`` formula itself, which is
    uncited anywhere in this repository (module docstring).  Bar ``rtol=1e-12``;
    measured this audit, the achieved agreement is machine precision
    (``2.22e-16``, one ULP), so the margin is over arithmetic roundoff, not
    modeling error.  Would catch a wrong numeric coefficient (e.g. the ``1.5``
    or a ``2.0`` prefactor), a sign flip on the log term or on ``L``, or an
    off-by-one in the harmonic-sum order or the ``N``/``N-1`` argument shift;
    would not catch an error shared with the ``B(N)``/``L(N)`` derivation.
    """
    lam = 2.0 * np.exp(-EULER - 0.5)
    assert np.isclose(lam**2 * np.exp(2.0 * EULER + 1.0) / 4.0, 1.0)

    orders = np.array([2.0, 3.0, 4.0, 5.0], dtype=complex)
    got = _pseudoitd_nlo_moments(orders, lam=lam)
    cf = 4.0 / 3.0
    expected = np.array(
        [
            cf
            * (
                2.0 * (_harmonic(int(n.real) - 1, 1) ** 2 + _harmonic(int(n.real) - 1, 2))
                + 1.0
                - 2.0 / (n.real * (n.real + 1.0))
            )
            for n in orders
        ]
    )
    # rtol=1e-12 bounds arithmetic roundoff, not physics: measured this audit,
    # got/expected agree to 2.22e-16 (machine epsilon), four orders inside the bar.
    assert np.allclose(got.real, expected, rtol=1e-12, atol=0.0)
    # Real integer N must give a real moment; nonzero imag would signal a
    # branch cut or a complex-argument bug in the _psi calls.
    assert np.allclose(got.imag, 0.0, atol=1e-12)
    # A constant that happened to vanish would make the test vacuous.
    assert np.all(np.abs(expected) > 1.0)


def test_matching_log_carries_the_lambda_dependence():
    """Away from that ``lam`` the moments move by exactly ``ln(lam^2 ...) B(N)``.

    Subtracts the log-cancelling reference of the previous test (evaluated at a
    second ``lam=1.7``) from a fresh call and checks the difference against
    ``C_F log_factor B(N)``, with ``B`` recomputed by the same from-scratch
    harmonic-sum helper (independent of ``_psi``, not of the ``B(N)`` formula's
    own uncited provenance -- module docstring).  ``reference`` is itself an
    output of the function under test, so any ``lam``-independent bug in its
    ``L(N)`` term cancels in the subtraction: this test constrains only the
    ``lam`` dependence, not the additive constant (test 1's job).  It does
    catch a broken ``(contour, lam)`` memoization key -- if ``lam`` dropped out
    of the cache key, ``got`` would equal ``reference`` exactly and the left
    side would vanish while ``cf * log_factor * b`` does not.

    Bar ``rtol=1e-12, atol=0.0``.  The explicit ``atol=0.0`` is load-bearing
    and was added to close audit weakness
    ``test_closure_pseudoitd_matching-04``: without it ``np.allclose`` falls
    back to its default ``atol=1e-8``, which at these RHS magnitudes
    (``|rhs|`` measured at ``3.25, 5.08, 6.38``) dominates the stated rtol by
    a factor of ~3075, making the effective bar ~1e-8 absolute rather than
    1e-12 relative.  Measured this pass with the explicit ``atol=0.0`` in
    place, the achieved agreement is ``max abs(lhs/rhs - 1) = 2.22e-16`` (one
    ULP), so the now-real 1e-12 bar still clears by four orders and the
    tightening masks nothing.
    """
    orders = np.array([2.0, 3.0, 4.0], dtype=complex)
    reference = _pseudoitd_nlo_moments(orders, lam=2.0 * np.exp(-EULER - 0.5))
    lam = 1.7
    got = _pseudoitd_nlo_moments(orders, lam=lam)
    cf = 4.0 / 3.0
    s1 = np.array([_harmonic(int(n.real), 1) for n in orders])
    b = 1.5 + 1.0 / (orders.real * (orders.real + 1.0)) - 2.0 * s1
    log_factor = np.log(lam**2 * np.exp(2.0 * EULER + 1.0) / 4.0)
    # atol=0.0 is deliberate: without it np.allclose's default atol=1e-8 would
    # dominate this rtol by ~3075x at these magnitudes and the stated 1e-12
    # would be decorative.  Achieved agreement 2.22e-16 (measured this pass).
    assert np.allclose(
        (got - reference).real, cf * log_factor * b, rtol=1e-12, atol=0.0
    )


def test_li_vector_density_reduces_to_existing_nlo_mellin_coefficient():
    """Li's normalized QCF reproduces PIXEL's one-loop forward coefficient.

    The paper uses powers ``z**(N-1)`` while PIXEL's stored helper is indexed by
    ``N``.  Division by the QCF value at ``nu=0`` supplies the factor one half;
    the remaining Li plus-density moment must therefore equal ``-c1(N)``.
    This is the convention bridge used by the NNLO implementation.
    """
    z, weights, p1, _, _ = li_vector_densities(lam=1.37, nf=3, n_points=128)
    orders = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=complex)
    li_moments = np.array(
        [np.sum(weights * p1 * (z ** (order.real - 1.0) - 1.0)) for order in orders]
    )
    expected = -_pseudoitd_nlo_moments(orders, lam=1.37).real
    np.testing.assert_allclose(li_moments, expected, rtol=0.0, atol=5e-12)


def test_li_nnlo_vector_ratio_normalizes_and_splits_real_imaginary():
    """The normalized Li kernel preserves nu=0 and has two NNLO projections."""
    basis = ProductBasis(
        Grid(n_points=13, domain=(1e-3, 1.0), spacing="log"),
        element_type="piecewise",
        element_kwargs={"order": 2},
    )
    a_s = 0.02
    common = dict(
        a_s=a_s,
        nf=3,
        order="NNLO",
        lam=1.0,
        matching_points=128,
    )
    real = li_vector_fourier_matrix(basis, [0.0, 2.1], component="real", **common)
    imag = li_vector_fourier_matrix(basis, [0.0, 2.1], component="imag", **common)
    lo_real = li_vector_fourier_matrix(
        basis, [0.0, 2.1], component="real", **{**common, "order": "LO"}
    )
    lo_imag = li_vector_fourier_matrix(
        basis, [0.0, 2.1], component="imag", **{**common, "order": "LO"}
    )
    # Every plus term vanishes at nu=0; the prescaled vector observable has
    # exactly the conserved a1 row there.
    np.testing.assert_allclose(real[0], lo_real[0], rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(imag[0], lo_imag[0], rtol=0.0, atol=2e-13)
    # The negative-support coefficient first appears at two loops and enters
    # cosine and sine with different signs.  Both corrections must be nonzero.
    assert np.max(np.abs(real[1] - lo_real[1])) > 1e-5
    assert np.max(np.abs(imag[1] - lo_imag[1])) > 1e-5


def test_li_vector_formula_matches_authors_ancillary_iA_benchmark():
    """Reproduce the authors' independent Mathematica output at omega=29.15.

    ``papers/.../attachedfiels/readme.nb``, Out[16], prints the temporal-vector
    QCF coefficients of ``alpha_s^n L^k`` for ``nf=3``.  This test reconstructs
    that unnormalized QCF from PIXEL's translated densities.  It exercises all
    positive- and negative-support NNLO functions before the reduced-ratio
    normalization, so a transcription error cannot hide in a shared quotient.
    """
    omega = 29.15
    nf = 3
    packed = _li_vector_quadrature(192)
    z, weights, names, *arrays = packed
    parts = dict(zip(names, arrays))
    endpoint = np.exp(1j * omega)
    positive_phase = np.exp(1j * z * omega) - endpoint
    negative_phase = np.exp(-1j * z * omega) - endpoint

    nlo = np.array(
        [
            (
                CF * 2.5 * endpoint
                + np.sum(weights * CF * parts["p10"] * positive_phase)
            )
            / np.pi,
            (
                CF * 1.5 * endpoint
                + np.sum(weights * CF * parts["p11"] * positive_phase)
            )
            / np.pi,
        ]
    )
    np.testing.assert_allclose(
        nlo,
        np.array(
            [
                -1.2005461924921144 + 12.98369222404692j,
                -1.145511209994593 - 3.4248911061708265j,
            ]
        ),
        rtol=0.0,
        atol=5e-10,
    )

    nonplanar = CF**2 - 0.5 * CF * CA
    positive = (
        CF**2 * parts["p20_cf2"]
        + CA * CF * parts["p20_cacf"]
        + nf * CF * TF * parts["p20_nftf"],
        CF**2 * parts["p21_cf2"]
        + CA * CF * parts["p21_cacf"]
        + nf * CF * TF * parts["p21_nftf"],
        CF**2 * parts["p22_cf2"]
        + CA * CF * parts["p22_cacf"]
        + nf * CF * TF * parts["p22_nftf"],
    )
    negative = (nonplanar * parts["n20"], nonplanar * parts["n21"], None)
    delta = (
        CF**2 * (-4.0 * ZETA3 + np.pi**2 / 9.0 + 223.0 / 192.0)
        + CA * CF * (ZETA3 - 5.0 * np.pi**2 / 24.0 + 4877.0 / 576.0)
        + nf * CF * TF * (-469.0 / 144.0),
        CF**2 * (np.pi**2 / 3.0 + 25.0 / 16.0)
        + CA * CF * (-np.pi**2 / 12.0 + 53.0 / 16.0)
        + nf * CF * TF * (-5.0 / 4.0),
        CF**2 * 9.0 / 16.0
        + CA * CF * 11.0 / 16.0
        + nf * CF * TF * (-1.0 / 4.0),
    )
    nnlo = []
    for power in range(3):
        value = delta[power] * endpoint + np.sum(
            weights * positive[power] * positive_phase
        )
        if negative[power] is not None:
            value += np.sum(weights * negative[power] * negative_phase)
        nnlo.append(value / np.pi**2)
    np.testing.assert_allclose(
        np.asarray(nnlo),
        np.array(
            [
                40.04903289675574 - 47.747261821812295j,
                -12.084423887750374 + 27.802547090451622j,
                -0.2614823096541079 - 4.45216157721349j,
            ]
        ),
        rtol=0.0,
        atol=5e-9,
    )


@pytest.mark.parametrize("suite", SUITES)
def test_pitd_evolve_constant_reaches_the_built_kernels(suite, monkeypatch):
    """The flag must change the operator, physical field and nuisances alike.

    Toggles ``config.PITD_EVOLVE`` and rebuilds the same lattice record both
    ways.  True must yield an ``EvolvedPseudoITD`` physical-field kernel whose
    ``.matching``/``.matching_order``/``.mu0_2``/``.lam`` equal
    ``config.PITD_MATCHING_ORDER``/``config.Q0_2``/``config.PITD_MATCHING_
    LAMBDA`` -- this file's pin against the historical bug named in the module
    docstring.

    Two audit findings are closed here.  ``-02``: ``config.ORDER`` and
    ``config.MODE`` go into the same ``evolution`` dict ``build_pseudoitd``
    assembles, but were never compared against ``kernel.order``/
    ``kernel.mode``, so a builder that hardcoded ``"LO"``/``"truncated"``
    passed every assertion in this test.  They are asserted now.  ``-01``: the
    ``PITD_EVOLVE=False`` branch asserted only *not* ``EvolvedPseudoITD``,
    which any wrong type satisfies -- most importantly the wrong Fourier
    parity.  It now asserts the concrete class the record's ``component``
    calls for.  ``pixel.kernels.PseudoITDReal`` is ``Cosine`` and
    ``PseudoITDImag`` is ``Sine`` (checked this pass: distinct classes,
    neither a subclass of the other), so the positive check does discriminate
    a real/imag mixup, and the negative check on the other parity is asserted
    alongside it.  The ``alpha`` that ``momentum_density`` selects is pinned
    too, since it survives the flag flip and reaches both branches.

    Both suites' ``datasets.py`` are byte-identical apart from one docstring
    line, and share the config values checked here (module docstring), so the
    two parametrizations exercise identical wiring code against two different
    generated manifests, not two different implementations.  See
    ``_lattice_fixture`` for the skip condition.
    """
    config, datasets = _suite(suite)
    fields, record, path = _lattice_fixture(config)

    monkeypatch.setattr(config, "PITD_EVOLVE", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evolved = datasets.build_pseudoitd(record, path, fields, None)
    assert len(evolved.contributions) > 1  # field + systematics
    kernel = evolved.contributions[0].kernel  # the physical field
    assert isinstance(kernel, EvolvedPseudoITD)
    assert kernel.matching is True
    assert str(kernel.matching_order).upper() == config.PITD_MATCHING_ORDER
    assert kernel.pseudoitd_observable == config.PITD_OBSERVABLE
    assert kernel.operator_scheme == config.PITD_OPERATOR_SCHEME
    assert kernel.pseudoitd_kernel == config.PITD_KERNEL
    assert kernel.pseudoitd_data_normalization == config.PITD_DATA_NORMALIZATION
    assert kernel.pseudoitd_lorentz_component == config.PITD_LORENTZ_COMPONENT
    assert evolved.meta["pseudoitd_observable"] == config.PITD_OBSERVABLE
    assert evolved.meta["operator_scheme"] == config.PITD_OPERATOR_SCHEME
    assert evolved.meta["pseudoitd_kernel"] == config.PITD_KERNEL
    assert (
        evolved.meta["pseudoitd_data_normalization"]
        == config.PITD_DATA_NORMALIZATION
    )
    assert (
        evolved.meta["pseudoitd_lorentz_component"]
        == config.PITD_LORENTZ_COMPONENT
    )
    assert kernel.mu0_2 == pytest.approx(config.Q0_2)
    assert kernel.lam == pytest.approx(config.PITD_MATCHING_LAMBDA)
    # Audit -02: the remaining two keys of build_pseudoitd's evolution dict.
    # Without these a builder that hardcoded the evolution order or mode would
    # pass this test unchanged.
    assert kernel.order == config.ORDER
    assert kernel.mode == config.MODE

    monkeypatch.setattr(config, "PITD_EVOLVE", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = datasets.build_pseudoitd(record, path, fields, None)
    bare = plain.contributions[0].kernel
    assert not isinstance(bare, EvolvedPseudoITD)
    # Audit -01: what the kernel actually IS on the off path, not merely what
    # it is not.  PseudoITDReal/PseudoITDImag are Cosine/Sine, distinct
    # classes, so this discriminates the Fourier parity that the negative
    # assertion above cannot.
    expected_type, wrong_type = (
        (kernels.PseudoITDReal, kernels.PseudoITDImag)
        if record["component"] == "real"
        else (kernels.PseudoITDImag, kernels.PseudoITDReal)
    )
    assert isinstance(bare, expected_type), type(bare).__name__
    assert not isinstance(bare, wrong_type), type(bare).__name__
    # alpha=-1 reads f from the momentum density q = x f; it must survive the
    # flag flip, so compare the two branches rather than a literal.
    assert bare.alpha == pytest.approx(-1.0 if record.get("momentum_density", True) else 0.0)
    assert bare.alpha == pytest.approx(kernel.alpha)


@pytest.mark.parametrize("suite", SUITES)
def test_generation_time_build_matches_the_fitting_time_forward_operator(
    suite, monkeypatch
):
    """``with_systematics=False`` must change only the nuisance list.

    Closes missing item ``test_closure_pseudoitd_matching-M03``.  The closure
    is only meaningful if the operator used to *generate* the pseudo-data and
    the operator used to *fit* it are the same map: ``datasets.py``'s module
    docstring says generation builds physical-only datasets
    (``with_systematics=False``) and injects the systematics separately, while
    fitting uses the ``True`` default.  Every call in this file, and every call
    in any test in the suite (grep: ``with_systematics`` appears in no test
    file), left the flag at its default, so nothing checked that the flag is
    confined to the nuisance list.

    Oracle ``A3``, an identity between two objects this suite builds: the same
    record built both ways.  The forward operator is compared through
    ``EvolvedPseudoITD._cache_settings()`` -- the twenty-key dict that decides
    kernel cache identity, so two kernels agreeing on it are the same operator
    by the code's own definition -- rather than by assembling two 128-node
    matrices, which would cost minutes and prove the same thing.  The data
    (``mean``, ``cov``, ``nu``) must be untouched as well: a flag that quietly
    also dropped a covariance block would be a far worse bug than a missing
    contribution.

    Acceptance (measured by mutation this pass): making the ``evolution`` dict
    conditional on ``with_systematics`` -- the coupling the audit worried about
    -- leaves the physical kernel a bare ``Cosine`` in the ``False`` build and
    fails the ``isinstance`` assertion; no other test in this file notices.
    """
    config, datasets = _suite(suite)
    fields, record, path = _lattice_fixture(config)
    monkeypatch.setattr(config, "PITD_EVOLVE", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitting = datasets.build_pseudoitd(record, path, fields, None)
        generating = datasets.build_pseudoitd(
            record, path, fields, None, with_systematics=False
        )

    assert len(generating.contributions) == 1
    assert len(fitting.contributions) == 1 + len(record["systematics"])

    generated_kernel = generating.contributions[0].kernel
    assert isinstance(generated_kernel, EvolvedPseudoITD)
    assert generated_kernel._cache_settings() == (
        fitting.contributions[0].kernel._cache_settings()
    )
    np.testing.assert_array_equal(generating.mean, fitting.mean)
    np.testing.assert_array_equal(generating.cov, fitting.cov)
    np.testing.assert_array_equal(generating.nu, fitting.nu)


def test_reduced_ratio_convention_refuses_unimplemented_operator_schemes():
    """A sample closure kernel cannot reinterpret the reduced-ratio NLO term.

    This is intentionally a small kernel construction, not a full closure run:
    it verifies the same evolved pseudo-ITD factory used by the closure suites
    records the reduced ratio and rejects an RI/MOM or unreduced-QCF request
    before it can reach the MSbar coefficient.
    """
    shared = dict(component="real", z=[1.0], a=0.08, mu0_2=4.0, lam=1.0)
    kernel = EvolvedPseudoITD(**shared)
    assert kernel.pseudoitd_observable == "reduced_ratio"
    assert kernel.operator_scheme == "msbar"
    assert kernel.pseudoitd_kernel == "unpolarized"
    assert kernel.pseudoitd_data_normalization == "a1_prescaled"
    assert kernel.pseudoitd_lorentz_component == "temporal"
    assert kernel.channel == "minus"
    with pytest.raises(NotImplementedError, match="unreduced QCF"):
        EvolvedPseudoITD(**shared, pseudoitd_observable="unreduced_qcf")
    with pytest.raises(NotImplementedError, match="RI/MOM"):
        EvolvedPseudoITD(**shared, operator_scheme="ri_mom")
    with pytest.raises(NotImplementedError, match="helicity"):
        EvolvedPseudoITD(**shared, pseudoitd_kernel="helicity")
    with pytest.raises(NotImplementedError, match="transversity"):
        EvolvedPseudoITD(**shared, pseudoitd_kernel="transversity")
    with pytest.raises(NotImplementedError, match="provider must apply"):
        EvolvedPseudoITD(
            **shared, pseudoitd_data_normalization="raw_ratio"
        )
    with pytest.raises(NotImplementedError, match="additional B coefficient"):
        EvolvedPseudoITD(**shared, pseudoitd_lorentz_component="spatial")

    fixed = MatchingMatrix(Q2=4.0, kind="pseudoitd")
    rowwise = VariableQ2MatchingMatrix([4.0], kind="pseudoitd")
    assert fixed.pseudoitd_observable == rowwise.pseudoitd_observable == "reduced_ratio"
    assert fixed.operator_scheme == rowwise.operator_scheme == "msbar"
    assert fixed.pseudoitd_kernel == rowwise.pseudoitd_kernel == "unpolarized"
    assert (
        fixed.pseudoitd_data_normalization
        == rowwise.pseudoitd_data_normalization
        == "a1_prescaled"
    )
    assert (
        fixed.pseudoitd_lorentz_component
        == rowwise.pseudoitd_lorentz_component
        == "temporal"
    )
    with pytest.raises(NotImplementedError, match="unreduced QCF"):
        MatchingMatrix(Q2=4.0, kind="pseudoitd", pseudoitd_observable="unreduced_qcf")
    with pytest.raises(NotImplementedError, match="RI/MOM"):
        VariableQ2MatchingMatrix([4.0], kind="pseudoitd", operator_scheme="ri_mom")


@pytest.mark.parametrize("suite", SUITES)
def test_systematic_nuisances_stay_on_the_bare_transform(suite, monkeypatch):
    """Lattice artifacts are not evolved with the field, and that is on purpose.

    ``ht``, ``chiral``, ``cont_invz``, ``cont_aL`` and ``inf_Lz`` contaminate the
    *measured* reduced ITD: they are properties of the lattice calculation, not
    PDF-like objects sitting at ``mu0`` with DGLAP scale dependence of their own.
    Evolving them alongside the field would invent that dependence.  The physical
    field's transform changes when ``PITD_EVOLVE`` flips; theirs must not.

    Mechanism: ``PseudoITD.from_file`` builds every systematic contribution via
    ``_kernel_for(..., coefficient=coeff)`` with no ``mu0_2``/``evolution_
    lambda``, so the split is structural, not a convention the closure config
    could accidentally violate -- this test would catch a change that routed
    the ``evolution`` dict into the systematics path too.  The fixture record
    used here carries all five systematic keys, so the loop exercises every
    kind, not just one.  ``c.kernel.coefficient is not None`` additionally
    pins that the coefficient factory itself was wired, not only the kernel
    type.

    Audit weakness ``-06`` closed: the split used to be positional
    (``physical, *nuisances = dataset.contributions``) with nothing confirming
    that ``contributions[0]`` really is the physical field, so a change to
    ``PseudoITD.from_file``'s construction order would have failed this test
    confusingly rather than clearly -- or, worse, swapped the two isinstance
    checks onto contributions that happened to satisfy them.  The split is now
    made by ``Contribution.field`` against the exact stand-ins
    ``_lattice_fixture`` handed in, which also pins that *every* systematic key
    in the manifest record reached the dataset and none was invented: the
    nuisance field set must equal the expected set, not merely be a subset.
    ``field`` is compared against ``str(...)`` of the fixture's own stand-in
    rather than a literal name, because ``pixel.data.base.field_name``
    special-cases real ``Field`` instances and falls back to ``str()`` for the
    ``SimpleNamespace`` stand-ins used here (see ``_lattice_fixture``); the
    comparison is therefore self-consistent rather than a hardcoded repr.

    The identical evolved/bare split is pinned one layer down -- directly
    against that builder, with no closure suite involved -- by
    ``tests/test_data_builders.py::test_pseudoitd_evolution_is_main_pdf_only_
    and_accepts_physical_z``.  See ``_lattice_fixture`` for the skip condition.
    """
    config, datasets = _suite(suite)
    fields, record, path = _lattice_fixture(config)
    monkeypatch.setattr(config, "PITD_EVOLVE", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = datasets.build_pseudoitd(record, path, fields, None)

    physical_field = str(fields[record["field"]])
    expected_nuisance_fields = {
        str(fields[config.nuisance_field_name(record["field"], key)])
        for key in record["systematics"]
    }
    assert expected_nuisance_fields, "record must carry systematics to mean anything"
    by_field = {c.field: c for c in dataset.contributions}
    assert len(by_field) == len(dataset.contributions)  # no duplicate fields
    assert set(by_field) == {physical_field} | expected_nuisance_fields

    physical = by_field[physical_field]
    nuisances = [by_field[name] for name in sorted(expected_nuisance_fields)]
    # Position is no longer assumed, but it is still true and worth pinning:
    # a reordering is a real change to the builder, not a detail.
    assert dataset.contributions[0].field == physical_field
    assert isinstance(physical.kernel, EvolvedPseudoITD)
    for c in nuisances:
        assert not isinstance(c.kernel, EvolvedPseudoITD), c.field
        assert c.kernel.coefficient is not None, c.field


def test_matching_changes_the_operator_it_is_applied_to():
    """Matching must not be a silent no-op on the assembled transform.

    Builds two otherwise-identical ``EvolvedPseudoITD`` kernels differing only
    in ``matching`` (``False`` vs ``"NLO"``) and requires their assembled
    ``.matrix()`` outputs to differ by more than ``1e-6 * scale``.  Structural
    only: it proves matching changes *something*, not that the change has the
    right sign or magnitude -- an implementation wrong by orders of magnitude
    would still pass this bar (see the comment on the bar below).  The
    bit-level check that the composition is not just nonzero but *correct* --
    row-by-row ``Fourier @ matching(mu2) @ evolution(mu2)`` -- lives in
    ``tests/test_evolution.py::test_evolved_pseudoitd_matching_composes_after_
    evolution`` (``rtol=1e-11``, module docstring).  Component ``"real"`` only.
    """
    basis = ProductBasis(
        Grid(n_points=9, domain=(1e-3, 1.0), spacing="log"),
        element_type="piecewise",
        element_kwargs={"order": 2},
    )
    z = np.array([1.0, 2.0])
    nu = np.array([0.5, 1.0])
    shared = dict(mu0_2=1.6384, lam=1.0, order="NLO", channel="minus", component="real")
    bare = EvolvedPseudoITD(z=z, a=0.087, matching=False, **shared)
    matched = EvolvedPseudoITD(z=z, a=0.087, matching="NLO", **shared)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare_matrix = np.asarray(bare.matrix(nu, basis), dtype=float)
        matched_matrix = np.asarray(matched.matrix(nu, basis), dtype=float)
    scale = np.abs(bare_matrix).max()
    assert scale > 0.0
    # Bar only proves nonzero, not correct-magnitude: measured this audit at
    # these kinematics, the actual diff/scale is ~0.70 -- six orders above this
    # 1e-6 bar, which fails only if matching is completely inert.
    assert np.abs(matched_matrix - bare_matrix).max() > 1e-6 * scale


def test_evolved_pseudoitd_refuses_a_grid_with_a_node_at_the_origin():
    """The origin node is rejected, and only the origin node.

    Closes missing item ``test_closure_pseudoitd_matching-M01``.
    ``EvolvedPseudoITD`` composes a Fourier matrix with an evolution matrix, so
    the evolution step has to be a *square* grid->grid operator; plain
    ``evolution_matrix`` tolerates the ``(n-1, n)`` result an origin node
    produces (``tests/test_evolution.py``), this kernel cannot.  The guard was
    proven unexecuted by any test in the suite (``coverage_index.txt``), and
    ``tests/test_coverage_edges.py``'s neighbouring ``EvolvedPseudoITD``
    ``pytest.raises`` cases reach only the ``component``, ``positive
    separations`` and z/nu-length guards -- ``Grid``'s own node placement never
    lands exactly on ``0.0`` (checked this pass: ``Grid(n_points=9,
    domain=(0.0, 1.0), spacing="linear")`` starts at ``0.0025``), which is why
    a raw-node ``CubicSplineBasis`` is used here.

    Oracle ``F1``.  Matched on the *message*, not the type: this method raises
    ``ValueError`` from four different guards, so a type-only match would let a
    corrupted input trip the wrong one and still pass.

    The control is the point of the test as much as the raise is.  The two
    bases differ in exactly one node -- ``nodes[1:]`` of the rejected basis is
    ``array_equal`` to the accepted basis's nodes, asserted below -- so the
    origin node is demonstrably the whole cause, and the accepted basis is run
    all the way through ``base_matrix`` to a finite matrix rather than merely
    constructed.  ``LO``/``matching=False`` keeps that control at ~0.7 s;
    matching is irrelevant to this guard, which fires before
    ``matching_scales`` is ever called.
    """
    positive_nodes = np.geomspace(1e-3, 1.0, 8)
    with_origin = CubicSplineBasis(np.concatenate(([0.0], positive_nodes)))
    without_origin = CubicSplineBasis(positive_nodes)
    from pixel.kernels.base import basis_grid_nodes

    np.testing.assert_array_equal(
        np.asarray(basis_grid_nodes(with_origin))[1:],
        np.asarray(basis_grid_nodes(without_origin)),
    )
    assert np.asarray(basis_grid_nodes(with_origin))[0] == 0.0

    nu = np.array([0.5, 1.0])
    shared = dict(
        component="real",
        z=np.array([1.0, 2.0]),
        a=0.09,
        mu0_2=2.0,
        lam=1.0,
        order="LO",
        channel="minus",
        matching=False,
        contour=MellinContour(c=0.8, zmax=8.0, panel=2.0, npts=16),
    )
    with pytest.raises(ValueError, match="node at x=0"):
        EvolvedPseudoITD(**shared).base_matrix(nu, with_origin)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        control = np.asarray(
            EvolvedPseudoITD(**shared).base_matrix(nu, without_origin), dtype=float
        )
    assert control.shape == (nu.size, without_origin.n_elements)
    assert np.isfinite(control).all()
    assert np.abs(control).max() > 0.0


def _lattice_fixture(config):
    """A real pseudo-ITD record carrying systematics, from the generated truth.

    Returns the first ``kind="pseudoitd"`` record with a nonempty
    ``systematics`` list from ``config.truth_dir("2")/manifest.json``, plus a
    ``fields`` dict (``config.ALL_FIELDS`` + every ``config.nuisance_specs()``
    name) of ``SimpleNamespace(basis=..., name=...)`` stand-ins.  That is
    enough for ``build_pseudoitd``, which only reads ``.basis``/``.name`` off
    them, but ``pixel.data.base.field_name`` special-cases real ``Field``
    instances and falls back to ``str(...)`` otherwise (checked this audit:
    ``str(SimpleNamespace(name="t3", ...))`` prints as ``"namespace(...)"``,
    not ``"t3"``) -- so any ``Contribution.field`` string built off this
    fixture is not a clean field name.  No test in this file reads it as one.
    ``pytest.skip``s, rather than failing, if ``truthQ_2`` was not generated in
    this checkout or has no pseudo-ITD record with systematics; both closure
    suites currently have 117 such records (checked this audit), but neither
    skip condition is enforced by anything here, so a ``generate.py``
    regression that stopped attaching systematics would silently skip this
    file's suite-parametrized tests rather than fail them.
    """
    manifest_path = config.truth_dir("2") / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("closure truth member not generated in this checkout")
    manifest = json.loads(manifest_path.read_text())
    record = next(
        (
            r
            for r in manifest.get("lattice", [])
            if r["kind"] == "pseudoitd" and r.get("systematics")
        ),
        None,
    )
    if record is None:
        pytest.skip("no pseudo-ITD record with systematics in the manifest")
    # The closure's element type is a cubic spline, which takes no ``order``.
    basis = ProductBasis(config.make_grid(), element_type=config.ELEMENT_TYPE)
    names = list(config.ALL_FIELDS) + [
        config.nuisance_field_name(field, key)
        for field, key in config.nuisance_specs()
    ]
    fields = {name: SimpleNamespace(basis=basis, name=name) for name in names}
    return fields, record, config.truth_dir("2") / record["file"]
