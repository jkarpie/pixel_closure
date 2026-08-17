"""Drell-Yan observable-space closure plotting: the bilinear posterior band
and the per-dataset record/figure assembly built on top of it.

Exercises the plotting orchestration layer in the four closure driver
packages -- ``closure_JAM_truth``, ``closure_JAM_truth_small``,
``closure_NNPDF_truth``, ``closure_NNPDF_truth_small`` -- specifically each
package's ``plot_datasets.py``: ``_dataset_x``, ``_is_constraint_dataset``,
``_sample_vectors_and_weights``, ``_sampled_linear_predictive_band``,
``_bilinear_symmetric_action``, ``_bilinear_predictive_band``,
``build_records_from_result``, and ``plot_records``.
``closure_JAM_truth/plot_datasets.py`` and
``closure_NNPDF_truth/plot_datasets.py`` are byte-identical, and likewise the
two ``_small`` variants -- asserted, not assumed, by
``test_jam_and_nnpdf_plot_datasets_stay_byte_identical``; "full" and "small"
differ only inside ``_truth_predictions`` (its ``moment_systematic_values``
handling), which no test below reaches.

``RUNNERS`` used to be all four packages, and so ran each of those two distinct
code bodies twice under different names.  Since 2026-08-14, on the owner's
instruction, it is the NNPDF pair: one entry per distinct body, no repeats.  The
JAM copies are still guarded -- more strictly than the parametrization managed --
by ``test_jam_and_nnpdf_plot_datasets_stay_byte_identical``, which compares the
files on disk and so catches divergence anywhere in them, including in
``main``/``write_summary``/``plot_records`` details no test here calls.  That test
names its packages as literal path components and never reads ``RUNNERS``, so it
is unaffected by the trim.  ``closure_JAM_truth_small.plot_datasets`` is also
still imported and driven directly by the ``plot_records`` rendering tests below.

Through ``_bilinear_predictive_band`` these tests also drive the real
``pixel.core.evidence._bilinear_prediction_from_second`` (imported lazily,
not mocked), and ``_sampled_linear_predictive_band`` drives the real
``pixel.core.evidence.posterior``.  ``_bilinear_symmetric_action`` here is a
private, parallel *reimplementation* of the same half-gradient math as
``pixel.core.evidence.bilinear_symmetric_action`` -- compare
``closure_JAM_truth/plot_datasets.py:132-170`` to
``src/pixel/core/evidence.py:1199-1259``.
``test_closure_symmetric_action_matches_the_pixel_core_evidence_copy`` checks
the two copies against each other, on a two-block factorized operator with
overlapping row ranges and on a non-symmetric dense one; being an
implementation-vs-implementation identity it measures transcription, not the
half-gradient convention, which ``tests/test_joint_action.py``'s ``jax.jacfwd``
check pins instead.

Oracles: the dense-band test's expected mean/variance are closed-form (a
quadratic-form expectation and its delta-method gradient), rederived
independently in the test rather than called from source -- a real check of
the wrapper arithmetic, except that the "variance" side transcribes a
*documented approximation* (the source's own comment notes the post-DY
posterior is not exactly Gaussian), so agreement confirms the formula matches
spec, not that the approximation itself is accurate.  The factorized test's
oracle is an algebraically equivalent dense operator built by the test, but
both results are produced by the *same* wrapper call; it is independent for
the per-branch tensor contraction (a transposed-mixing mutant breaks
agreement by 3.4e-2 against the 1e-12 bar) and common-mode for everything the
two branches share.  The sampled-band test's oracle is the law of total
variance written out in the test over per-sample moments from the shared
``posterior``.  The record-level and rendering tests use hand-built
``SimpleNamespace``/dict stand-ins for the real ``Model``/``Result``/record
objects; no accuracy bar applies to them, and each establishes a specific
structural or behavioural contract (see the per-test docstrings).

No test here runs a real fit or loads a truth file from disk
(``fitmod.run_fit``/``fitmod.load_truth`` are never reached: every test that
calls ``build_records_from_result`` supplies its own ``truth`` dict).  The
full pipeline that calls this same ``build_records_from_result``/
``plot_records`` pair against a real fit result is
``closure_*/run_closure.py`` (e.g.
``closure_JAM_truth_small/run_closure.py:274-280``), not executed here.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from closure_JAM_truth_small import plot_datasets as jam_small
from closure_NNPDF_truth import plot_datasets as nnpdf_full
from closure_NNPDF_truth_small import plot_datasets as nnpdf_small
from pixel.core.evidence import (
    BilinearFactorBlock,
    EvidenceBlocks,
    bilinear_symmetric_action,
    posterior,
)


# The NNPDF pair, since 2026-08-14 (owner's instruction).  This was all four packages,
# but closure_JAM_truth/plot_datasets.py == closure_NNPDF_truth/plot_datasets.py and
# likewise the two _small variants, byte for byte -- so the 4-entry tuple ran each of
# the 2 distinct plot_datasets.py bodies twice under different names.  One entry per
# distinct body is what is left.
#
# The JAM copies are NOT unguarded as a result: test_jam_and_nnpdf_plot_datasets_stay_
# byte_identical below compares the four files on disk (by literal package name -- it
# never reads this tuple), which detects any divergence, including in functions no test
# here calls.  That is strictly stronger than what re-running these fixtures through a
# second copy of the same bytes could show.
RUNNERS = (nnpdf_full, nnpdf_small)


def _blocks(operator, *, scale=None, linear_rows=0):
    """Minimal stand-in for ``EvidenceBlocks``, carrying only the fields
    ``_bilinear_predictive_band``/``_bilinear_symmetric_action`` read.
    """
    return SimpleNamespace(
        g=np.zeros(4),
        B=np.zeros((linear_rows, 4)),
        param_pred=np.zeros(linear_rows),
        bilinear_Y=operator,
        bilinear_M=np.zeros(2),
        bilinear_active_indices=None,
        bilinear_scale=scale,
    )


def test_jam_and_nnpdf_plot_datasets_stay_byte_identical():
    """The JAM and NNPDF copies of ``plot_datasets.py`` are the same file, in both
    sizes -- asserted, not assumed.

    ``RUNNERS`` used to be a 4-way parametrization over 2 distinct code bodies; the
    module docstring said so, and an earlier audit pass measured it with ``diff``.  A
    docstring cannot notice one copy being hand-edited, and the 4-way parametrization --
    kept for exactly that drift -- only detected it if the divergence happened to change
    the handful of functions the other tests call.  Comparing the bytes detects
    *any* divergence, including in ``main``/``write_summary``/``plot_records`` details no
    test here reaches (oracle ``F1``: file identity, sourced from disk rather than from
    either module object).

    **Since 2026-08-14 this test is the whole of the JAM-side guard**: ``RUNNERS`` is the
    NNPDF pair, so no parametrized case here loads ``closure_JAM_truth/plot_datasets.py``
    at all.  This test does not read ``RUNNERS`` -- it names the four packages as literal
    path components below -- and must keep naming all four.

    Measured this pass: ``sha256`` of the JAM/NNPDF full pair is
    ``cf44fa450537e9af...`` and of the ``_small`` pair ``adbee46c46c646b0...``; full and
    small differ from each other only inside ``_truth_predictions``
    (``moment_systematic_values`` handling), which is why they are compared as two pairs
    and not all four together.

    If this fails, the two copies were meant to diverge or one was edited without the
    other; resolve which, rather than deleting the check.
    """
    pairs = (
        ("closure_JAM_truth", "closure_NNPDF_truth"),
        ("closure_JAM_truth_small", "closure_NNPDF_truth_small"),
    )
    root = Path(__file__).resolve().parent.parent
    for jam, nnpdf in pairs:
        jam_bytes = (root / jam / "plot_datasets.py").read_bytes()
        nnpdf_bytes = (root / nnpdf / "plot_datasets.py").read_bytes()
        assert jam_bytes == nnpdf_bytes, (
            f"{jam}/plot_datasets.py and {nnpdf}/plot_datasets.py have diverged; "
            "copy the change to both, or add the JAM package back to RUNNERS so the "
            "divergent copy is actually exercised"
        )


@pytest.mark.parametrize("runner", RUNNERS)
def test_dense_dy_band_uses_quadratic_mean_and_delta_variance(runner):
    """Dense-operator DY band: closed-form mean, delta-method variance.

    ``expected_prediction`` is ``E[q^T Y q] = sum_ij Y_ij (Sigma_ij +
    mu_i mu_j)``, the exact expectation of a quadratic form -- rederived
    here, not called from source (oracle ``A1``).  ``expected_variance``
    transcribes the source's own delta-method formula
    ``4 (sym(Y) mu)^T Sigma (sym(Y) mu)``: a documented approximation to
    the true (non-Gaussian) DY posterior variance, so agreement confirms
    the formula is implemented as specified, not that the approximation
    itself is accurate.  Catches: a wrong contraction index, a missing
    factor of 2 (unsymmetrized gradient) or 4, or a non-symmetrized ``Y``.
    """
    operator = np.array(
        [
            [[1.0, 2.0], [0.0, 3.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    blocks = SimpleNamespace(
        bilinear_Y=operator,
        bilinear_M=np.zeros(2),
        bilinear_active_indices=None,
        bilinear_scale=None,
    )
    mean = np.array([0.4, -0.2])
    covariance = np.array([[0.3, 0.1], [0.1, 0.2]])

    prediction, std = runner._bilinear_predictive_band(blocks, mean, covariance)

    second = covariance + np.outer(mean, mean)
    expected_prediction = np.einsum("rij,ij->r", operator, second)
    symmetric = 0.5 * (operator + operator.transpose(0, 2, 1))
    half_gradient = np.einsum("rij,j->ri", symmetric, mean)
    expected_variance = 4.0 * np.einsum(
        "ri,ij,rj->r", half_gradient, covariance, half_gradient
    )
    # Bare default rtol=1e-7.  A standalone pixel.core.evidence reproduction
    # of this exact input (module docstring) landed at 0.0 relative
    # difference, so this bar only rules out an O(1) formula error -- it has
    # no work left to do at the precision two independent typings achieve.
    np.testing.assert_allclose(prediction, expected_prediction)
    np.testing.assert_allclose(std, np.sqrt(expected_variance))


@pytest.mark.parametrize("runner", RUNNERS)
def test_factorized_dy_band_matches_equivalent_dense_operator(runner):
    """Factorized ``BilinearFactorBlock`` path agrees with an algebraically
    equivalent dense tensor, to float round-off.

    ``dense = einsum("ab,rij->raibj", mixing, tensor).reshape(rows, 4, 4)``
    is the expansion of the factorized contraction (oracle ``A3``): both
    sides go through the *same* ``_bilinear_predictive_band`` call, so this
    is independent only for the per-branch tensor contraction inside
    ``_bilinear_prediction_from_second``/``_bilinear_symmetric_action``,
    and common-mode for everything the two branches share (second-moment
    construction, clip, sqrt) -- test 1 covers that shared part with an
    unrelated oracle, so the two are complementary.  Only a single
    factorized block is exercised; the multi-block accumulation
    (``action[sl] +=``) is untested here (module docstring).  Demonstrated
    to catch a transposed mixing index: a scratch reproduction using
    ``einsum("ba,rij->...")`` in place of ``einsum("ab,rij->...")`` breaks
    agreement by 3.4e-2, ten orders above the bar below.
    """
    mixing = np.array([[1.0, -0.3], [0.4, 0.8]])
    tensor = np.array(
        [
            [[0.5, 0.2], [-0.1, 0.7]],
            [[0.3, -0.4], [0.6, 0.1]],
        ]
    )
    scale = np.array([1.2, 0.7])
    factor = BilinearFactorBlock(
        tensor=tensor, row_start=0, row_stop=2, mixing=mixing
    )
    dense = np.einsum("ab,rij->raibj", mixing, tensor).reshape(2, 4, 4)
    dense *= scale[:, None, None]
    mean = np.array([0.4, -0.2, 0.3, 0.1])
    covariance = np.array(
        [
            [0.30, 0.02, 0.01, 0.00],
            [0.02, 0.25, 0.03, 0.01],
            [0.01, 0.03, 0.20, 0.04],
            [0.00, 0.01, 0.04, 0.15],
        ]
    )

    factorized_result = runner._bilinear_predictive_band(
        _blocks((factor,), scale=scale), mean, covariance
    )
    dense_result = runner._bilinear_predictive_band(
        _blocks(dense), mean, covariance
    )

    # Float64 round-off for an algebraically exact reformulation (test
    # docstring) -- not a physics bar.  Measured to catch a transposed
    # mixing/tensor index at 3.4e-2, far above this floor.
    np.testing.assert_allclose(factorized_result[0], dense_result[0], rtol=1e-12)
    np.testing.assert_allclose(factorized_result[1], dense_result[1], rtol=1e-12)


def _two_block_bilinear_blocks(seed=7):
    """A real ``EvidenceBlocks`` carrying a **two-block** factorized DY operator.

    The two blocks' row ranges overlap on row 1, so the source's
    ``action[row_start:row_stop] += ...`` accumulation is exercised as an accumulation:
    with ``+=`` replaced by ``=`` the overlapped row changes.  ``bilinear_active_indices``
    selects 6 of 8 coefficients, so the active-subspace path is exercised too.
    """
    rng = np.random.default_rng(seed)
    n_fields, n_grid = 2, 3
    blocks_tuple = (
        BilinearFactorBlock(
            tensor=rng.normal(size=(2, n_grid, n_grid)),
            row_start=0,
            row_stop=2,
            mixing=rng.normal(size=(n_fields, n_fields)),
        ),
        BilinearFactorBlock(
            tensor=rng.normal(size=(2, n_grid, n_grid)),
            row_start=1,
            row_stop=3,
            mixing=rng.normal(size=(n_fields, n_fields)),
        ),
    )
    return EvidenceBlocks(
        M=np.zeros((0,)),
        C=np.zeros((0, 0)),
        B=np.zeros((0, 8)),
        g=np.zeros(8),
        K=np.eye(8),
        param_pred=np.zeros((0,)),
        bilinear_Y=blocks_tuple,
        bilinear_M=np.zeros(3),
        bilinear_active_indices=np.array([0, 1, 2, 4, 5, 6]),
        bilinear_scale=np.array([1.2, 0.7, 1.5]),
    )


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("layout", ["factorized_two_block", "dense"])
def test_closure_symmetric_action_matches_the_pixel_core_evidence_copy(runner, layout):
    """The closure runners' private ``_bilinear_symmetric_action`` agrees with
    ``pixel.core.evidence.bilinear_symmetric_action`` on the same blocks.

    ``closure_JAM_truth/plot_datasets.py:132-170`` is a hand-maintained NumPy
    reimplementation of ``src/pixel/core/evidence.py:1199-1259`` -- same half-gradient
    ``0.5 * grad_q(q^T Y q)``, same einsum patterns, never imported from the source
    module.  ``tests/test_joint_action.py`` checks the ``pixel.core.evidence`` copy
    against ``jax.jacfwd``; nothing compared the two copies to each other, so a fix
    applied to one (informed by that autodiff check) could silently miss the other.
    This closes that: oracle ``A3``, an identity between two independently typed
    implementations, run on a real ``EvidenceBlocks`` -- note both are *implementations*,
    so agreement is evidence about the transcription, not about whether the half-gradient
    convention is physically right (``test_joint_action.py``'s autodiff check is what
    pins the convention).

    Two layouts, because the copies branch on the operator type:

    * ``factorized_two_block`` -- **two** ``BilinearFactorBlock``s with overlapping row
      ranges (0:2 and 1:3), which is new coverage: every existing test of the closure
      copy passes a single block, so its ``action[sl] +=`` accumulation was never an
      accumulation.  ``bilinear_active_indices`` selects 6 of 8 coefficients and
      ``bilinear_scale`` is non-uniform, so both post-processing steps are live.
    * ``dense`` -- the ``0.5 * (forward + backward)`` explicit symmetrization against the
      source's ``0.5 * (Y + Y^T)`` contraction, on a deliberately non-symmetric ``Y``
      (a symmetric one would make the two forms agree for the wrong reason).

    Bar ``rtol=1e-12, atol=0``.  Measured on these fixtures as ``max|a/b - 1|``: 6.7e-16
    for both layouts, with the smallest entry compared 0.199 (factorized) / 0.257 (dense)
    -- float64 noise between a NumPy and a JAX evaluation of the same contractions, not
    an accuracy floor, so the bar carries ~3 orders of headroom.
    """
    blocks = _two_block_bilinear_blocks()
    rng = np.random.default_rng(19)
    q = rng.normal(size=8)
    if layout == "dense":
        active = np.asarray(blocks.bilinear_active_indices, dtype=int)
        dense = rng.normal(size=(3, active.size, active.size))
        # Deliberately not symmetric: sym(Y) != Y, so the two symmetrizations
        # are being compared on an input that can tell them apart.
        assert not np.allclose(dense, np.swapaxes(dense, 1, 2))
        blocks = blocks._replace(bilinear_Y=dense)

    from_closure = np.asarray(runner._bilinear_symmetric_action(blocks, q), dtype=float)
    from_pixel = np.asarray(bilinear_symmetric_action(blocks, q), dtype=float)

    assert from_closure.shape == from_pixel.shape == (3, 6)
    # Ratio, not eyeball: both sides are O(1) here and non-zero everywhere.
    assert np.all(np.abs(from_pixel) > 1e-6)
    np.testing.assert_allclose(from_closure, from_pixel, rtol=1e-12, atol=0.0)


def _dataset(name, mean, cov, independent, *, component):
    """Minimal stand-in for a ``pixel.core.model.Dataset``, carrying only
    the attributes ``build_records_from_result``/``_dataset_x`` read.
    """
    return SimpleNamespace(
        name=name,
        mean=np.asarray(mean, dtype=float),
        cov=np.asarray(cov, dtype=float),
        nu=np.arange(len(mean), dtype=float),
        n_data=len(mean),
        source="closure",
        component=component,
        metadata=SimpleNamespace(independent=independent),
    )


def _plain_dataset(independent, *, nu, n_data):
    """A dataset stand-in carrying only what ``_dataset_x`` reads."""
    return SimpleNamespace(
        metadata=SimpleNamespace(independent=independent),
        nu=np.asarray(nu, dtype=float),
        n_data=n_data,
    )


@pytest.mark.parametrize(
    "independent, nu, n_data, expected_x, expected_label, expected_log",
    [
        # Q2/S/Y complete and S > 0: the xF branch.  2*sqrt(25/100) = 1.0 exactly
        # in float64, so xF is bit-identical to sinh(Y).
        ({"Q2": [25.0, 25.0], "S": [100.0, 100.0], "Y": [0.4, -0.2]},
         [0.0, 1.0], 2, np.sinh([0.4, -0.2]), r"$x_F$", False),
        # S <= 0 fails the xF guard and falls through to the key loop, which
        # finds the same table's "Y" -- one case covering both the guard and
        # the bare-Y branch.
        ({"Q2": [25.0, 25.0], "S": [0.0, 0.0], "Y": [0.4, -0.2]},
         [0.0, 1.0], 2, [0.4, -0.2], "Y", False),
        # Shape mismatch fails the same guard.
        ({"Q2": [25.0], "S": [100.0, 100.0], "Y": [0.4, -0.2]},
         [0.0, 1.0], 2, [0.4, -0.2], "Y", False),
        # "x", strictly positive: the only branch that turns the log scale on.
        ({"x": [0.1, 0.3]}, [0.0, 1.0], 2, [0.1, 0.3], "x", True),
        # "x" reaching zero: same branch, log scale suppressed.
        ({"x": [0.0, 0.3]}, [0.0, 1.0], 2, [0.0, 0.3], "x", False),
        ({"nu": [1.5, 2.5]}, [0.0, 1.0], 2, [1.5, 2.5], "nu", False),
        ({"n": [3.0, 4.0]}, [0.0, 1.0], 2, [3.0, 4.0], "n", False),
        # No usable independent variable: fall back to a 1-D nu ...
        ({}, [7.0, 8.0], 2, [7.0, 8.0], "nu", False),
        # ... and finally to a row index when nu is not 1-D.
        ({}, [[1.0, 2.0], [3.0, 4.0]], 2, [0.0, 1.0], "point", False),
    ],
)
@pytest.mark.parametrize("runner", RUNNERS)
def test_dataset_x_selects_each_independent_variable_in_priority_order(
    runner, independent, nu, n_data, expected_x, expected_label, expected_log
):
    """``_dataset_x`` returns the right ``(x, label, use_log)`` for all six of its
    return paths, including the two guards that fall out of the Drell--Yan branch.

    ``closure_JAM_truth/plot_datasets.py:22-42`` picks a plotting abscissa by trying, in
    order: the DY ``(Q2, S, Y) -> xF`` branch (guarded on matching shapes and ``S > 0``),
    then the first present key among ``x``/``Y``/``nu``/``n``, then ``dataset.nu`` if it
    is 1-D, then a bare row index.  Before this test only the first branch was asserted
    on, and only for a well-formed DY table: the ``x`` branch was reached incidentally by
    the mixed-records test without being checked, and the other four paths plus both
    guards were unreached anywhere in ``tests/`` (the helper is private and is never
    referenced by name outside its own module).

    Oracle ``F1``/``A1``: the label and ``use_log`` flag are structural contracts, and the
    xF case is a closed form -- ``2 sqrt(Q2/S) sinh(Y)`` with ``2 sqrt(25/100) = 1.0``
    exactly in float64, so the expected array is ``sinh(Y)`` with no rounding slack (this
    is the same exactness the mixed-records test relies on).  Bar: bare
    ``assert_allclose`` default ``rtol=1e-7``, which has nothing to do here -- every case
    is either exact or a passthrough of the input array.

    Catches a reordered or dropped key in the fallback chain (e.g. ``"nu"`` shadowing
    ``"x"``, which would silently switch DIS panels to a linear abscissa), an inverted
    ``S > 0`` / shape guard, and ``use_log`` being set for the wrong key or for
    non-positive ``x`` (which matplotlib would then render as an empty axis).  Says
    nothing about whether xF is the right variable to plot against.
    """
    x, label, use_log = runner._dataset_x(_plain_dataset(independent, nu=nu, n_data=n_data))
    np.testing.assert_allclose(x, expected_x)
    assert label == expected_label
    assert use_log is expected_log


def test_mixed_records_include_linear_and_dy_without_rebuilding_short_samples(
    monkeypatch,
):
    """``model._rebuild`` must receive the full representative vector, not
    a short per-sample row; DY record numerics are checked only for
    sign/finiteness, not value.

    ``rebuild``'s embedded ``assert vec.size == 3`` is the real check: it
    mirrors ``fit._weighted_mean_vec``'s ``NestedVegasSamples`` branch
    (``closure_JAM_truth_small/fit.py:474-484``), which returns a short
    Hubbard-Stratonovich-free ``theta_mean`` when called without ``model=``
    and the zero-padded full vector when called with it -- so this guards a
    real, documented fallback, not a synthetic one.  ``{record["kind"] for
    r in records} == {"linear", "dy"}`` establishes presence only; see the
    inline comments below for what it and the DY-record checks do not
    cover.
    """
    linear = _dataset(
        "dis_table", [0.8, 1.1], np.diag([0.04, 0.09]),
        {"x": np.array([0.1, 0.3])}, component="F2",
    )
    dy = _dataset(
        "dy_table", [1.4, 1.0], np.diag([0.01, 0.04]),
        {
            "Q2": np.array([25.0, 25.0]),
            "S": np.array([100.0, 100.0]),
            "Y": np.array([0.4, -0.2]),
        },
        component="pp",
    )
    operator = np.array(
        [
            np.diag([0.5, 0.2, 0.1, 0.3]),
            np.diag([0.2, 0.4, 0.3, 0.1]),
        ]
    )
    blocks = _blocks(operator, linear_rows=2)
    blocks.B[:] = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    model = SimpleNamespace(
        datasets=[linear, dy],
        fields=[SimpleNamespace(name="f", nodes=np.linspace(0.1, 0.9, 4))],
        gp_indices=[0],
        has_bilinear=True,
        n_free=3,
        rcond=1e-12,
        _layout=SimpleNamespace(
            datasets=(SimpleNamespace(dataset_index=0, data_slice=slice(0, 2)),),
            bilinear_datasets=(
                SimpleNamespace(dataset_index=1, data_slice=slice(0, 2)),
            ),
        ),
    )

    def rebuild(vec):
        # Mixed nested-VEGAS samples omit H-S coordinates; the plotting path must
        # use the already merged representative vector rather than those rows.
        assert np.asarray(vec).size == 3
        return vec

    model._rebuild = rebuild
    model._blocks = lambda _theta: blocks
    samples = SimpleNamespace(
        samples=np.zeros((5, 1)),
        mean=np.array([0.7, 0.9, 0.4, 0.2]),
        covariance=np.diag([0.04, 0.03, 0.02, 0.01]),
    )
    monkeypatch.setattr(
        jam_small.fitmod, "_weighted_mean_vec",
        lambda samples, model=None: np.zeros(1 if model is None else 3),
    )
    truth = {
        "x_nodes": np.linspace(0.1, 0.9, 4),
        "curves": {"f": [0.6, 0.8, 0.5, 0.3]},
    }

    records = jam_small.build_records_from_result(
        "2", "both", {"model": model, "samples": samples}, truth
    )

    # A count as well as the kinds: the set alone was blind to a duplicated
    # record (3 or 4 records with these kinds looked identical to it).
    assert len(records) == 2
    assert {record["kind"] for record in records} == {"linear", "dy"}
    by_kind = {record["kind"]: record for record in records}
    linear_record = by_kind["linear"]
    # The linear record's own numbers, all hand-derivable from the fixture and
    # previously unchecked: B selects GP coefficients 0 and 1, so the
    # reproduction is the first two entries of samples.mean, the band is the
    # square root of the first two diagonal entries of samples.covariance, and
    # the truth fold is the first two entries of the truth curve.
    assert linear_record["xlabel"] == "x"
    assert linear_record["use_log"] is True
    np.testing.assert_allclose(linear_record["x"], [0.1, 0.3])
    np.testing.assert_allclose(linear_record["data"], [0.8, 1.1])
    np.testing.assert_allclose(linear_record["sigma"], [0.2, 0.3])
    np.testing.assert_allclose(linear_record["reproduction"], [0.7, 0.9])
    np.testing.assert_allclose(linear_record["fit_std"], np.sqrt([0.04, 0.03]))
    np.testing.assert_allclose(linear_record["truth"], [0.6, 0.8])
    # chi2 = 0.1^2/0.04 + 0.2^2/0.09 = 0.25 + 4/9, over 2 points.
    assert linear_record["chi2"] == pytest.approx(0.25 + 4.0 / 9.0, rel=1e-12)
    assert linear_record["chi2_per_data"] == pytest.approx(
        (0.25 + 4.0 / 9.0) / 2.0, rel=1e-12
    )
    dy_record = by_kind["dy"]
    assert dy_record["xlabel"] == r"$x_F$"
    # Exact, not approximate: Q2/S = 0.25 makes the xF prefactor
    # 2*sqrt(Q2/S) equal to 1.0 in float64, so dy_record["x"] is
    # bit-identical to np.sinh(Y); the bare rtol=1e-7 default has no work
    # to do.
    np.testing.assert_allclose(dy_record["x"], np.sinh([-0.2, 0.4]))
    # Sign only -- does not pin the DY variance's value.  That formula is
    # pinned by test_dense_dy_band_uses_quadratic_mean_and_delta_variance;
    # a uniform scaling bug in this record's band would pass here.
    assert np.all(dy_record["fit_std"] > 0.0)
    # Finiteness only, not the chi2 value itself.
    assert np.isfinite(dy_record["chi2_per_data"])


def _linear_only_result(runner, datasets, monkeypatch, *, n_grid=2):
    """``(model, samples, truth)`` driving ``build_records_from_result`` over a list of
    purely linear datasets.

    ``B`` and ``param_pred`` are zero, so every reproduction is zero and each dataset's
    chi2 is exactly ``mean^T cov^-1 mean`` -- chosen per dataset by the caller, which is
    what makes the ordering and exclusion assertions readable by inspection.  The
    bilinear operator is present but empty, keeping ``_posterior_predictive_bands`` on
    its stored-moments branch without adding any DY rows.
    """
    rows = sum(int(dataset.n_data) for dataset in datasets)
    blocks = SimpleNamespace(
        g=np.zeros(n_grid),
        B=np.zeros((rows, n_grid)),
        param_pred=np.zeros(rows),
        bilinear_Y=np.zeros((0, n_grid, n_grid)),
        bilinear_M=np.zeros(0),
        bilinear_active_indices=None,
        bilinear_scale=None,
    )
    layouts, start = [], 0
    for index, dataset in enumerate(datasets):
        stop = start + int(dataset.n_data)
        layouts.append(
            SimpleNamespace(dataset_index=index, data_slice=slice(start, stop))
        )
        start = stop
    model = SimpleNamespace(
        datasets=list(datasets),
        fields=[SimpleNamespace(name="f", nodes=np.linspace(0.1, 0.9, n_grid))],
        gp_indices=[0],
        has_bilinear=True,
        n_free=1,
        rcond=1e-12,
        _layout=SimpleNamespace(
            datasets=tuple(layouts), bilinear_datasets=(), M=np.zeros(rows)
        ),
        _rebuild=lambda vec: vec,
        _blocks=lambda _theta: blocks,
    )
    samples = SimpleNamespace(
        samples=np.zeros((3, 1)),
        mean=np.zeros(n_grid),
        covariance=np.eye(n_grid) * 0.01,
    )
    monkeypatch.setattr(
        runner.fitmod, "_weighted_mean_vec", lambda samples, model=None: np.zeros(1)
    )
    truth = {
        "x_nodes": np.linspace(0.1, 0.9, n_grid),
        "curves": {"f": np.zeros(n_grid)},
    }
    return model, samples, truth


def _chi2_dataset(name, value, *, source="closure"):
    """A one-row linear dataset whose chi2 against a zero prediction is ``value**2``."""
    return SimpleNamespace(
        name=name,
        mean=np.array([float(value)]),
        cov=np.array([[1.0]]),
        nu=np.array([0.0]),
        n_data=1,
        source=source,
        component="F2",
        metadata=SimpleNamespace(independent={"x": np.array([0.2])}),
    )


@pytest.mark.parametrize("runner", RUNNERS)
def test_constraint_pseudo_data_is_excluded_from_records(runner, monkeypatch):
    """``build_records_from_result`` drops both flavours of constraint pseudo-data.

    ``_is_constraint_dataset`` (``closure_JAM_truth/plot_datasets.py:50-55``) is an
    ``or`` of two independent conditions -- ``source == "constraint"`` and a ``cons_``
    name prefix -- and ``build_records_from_result`` applies it to ``linear_layouts``
    *and* ``bilinear_layouts`` before anything is built or sorted.  No test reached it:
    every dataset built in this file has ``source="closure"`` and an ordinary name, and
    ``tests/test_closure_constraints.py`` tests a different concern (that the constraint
    datasets exist and match the injected truth, via ``suite.constraint_datasets``), not
    that the plotting layer filters them back out.

    Both legs are fed here: ``cons_origin_f`` has the ordinary ``"closure"`` source and
    is caught only by the name prefix, while ``endpoint_valence`` has an ordinary name
    and is caught only by the source.  A physical dataset is kept alongside them, so a
    filter that rejected everything would fail too (oracle ``F1``, structural).

    Catches the filter being dropped (constraint pseudo-data appearing as physics panels,
    and -- because ``x = 0`` endpoint rows carry a chi2 of their own -- displacing real
    datasets from the top of the chi2 ordering), and either leg being removed.
    """
    datasets = [
        _chi2_dataset("dis_table", 3.0),
        _chi2_dataset("cons_origin_f", 2.0),  # caught by the cons_ name prefix
        _chi2_dataset("endpoint_valence", 2.0, source="constraint"),  # by source
    ]
    model, samples, truth = _linear_only_result(runner, datasets, monkeypatch)

    records = runner.build_records_from_result(
        "2", "both", {"model": model, "samples": samples}, truth
    )

    assert [record["name"] for record in records] == ["dis_table"]


@pytest.mark.parametrize("runner", RUNNERS)
def test_records_are_sorted_by_descending_chi2_with_nan_last(runner, monkeypatch):
    """Records come back ordered by chi2, largest first, with NaN-chi2 datasets last.

    ``plot_datasets.py:324`` sorts on ``np.nan_to_num(r["chi2"], nan=-inf)`` in reverse,
    and the module docstring advertises the result ("panels sorted by chi2 contribution
    so the largest tensions appear first").  No test had more than two records or looked
    at their order, and the NaN branch -- which exists so a dataset whose chi2 cannot be
    formed sinks to the bottom instead of floating to the top -- was unreached.

    Four one-row datasets with unit variance and a zero prediction give chi2 exactly
    ``9``, ``4``, ``1`` and ``NaN`` (oracle ``A1``: chi2 for a unit-variance single row is
    the residual squared).  The expected order is therefore fixed by construction, not
    read off the output.  Catches a sort dropped or reversed, a sort key reading the
    per-point ``chi2_per_data`` instead of the total, and ``nan_to_num`` being removed --
    with ``nan=-inf`` gone, a NaN comparison leaves the list in an implementation-defined
    order that Python's stable sort happens to render as *unmoved*, so the NaN row would
    surface above ``dis_c`` here.
    """
    datasets = [
        _chi2_dataset("dis_nan", np.nan),
        _chi2_dataset("dis_c", 1.0),
        _chi2_dataset("dis_a", 3.0),
        _chi2_dataset("dis_b", 2.0),
    ]
    model, samples, truth = _linear_only_result(runner, datasets, monkeypatch)

    records = runner.build_records_from_result(
        "2", "both", {"model": model, "samples": samples}, truth
    )

    assert [record["name"] for record in records] == [
        "dis_a", "dis_b", "dis_c", "dis_nan",
    ]
    assert [record["chi2"] for record in records[:3]] == pytest.approx([9.0, 4.0, 1.0])
    assert np.isnan(records[-1]["chi2"])


@pytest.mark.parametrize("runner", RUNNERS)
def test_sample_vectors_and_weights_handles_map_and_weighted_samples(runner):
    """``_sample_vectors_and_weights`` distinguishes a MAP result from weighted samples.

    ``plot_datasets.py:58-65`` has three shapes to get right: a MAP-like object (anything
    carrying ``.x``) becomes a single row with unit weight; a 2-D ``.samples`` array keeps
    its rows and takes normalized weights from ``fit._weights``; a 1-D ``.samples`` array
    is promoted to one row.  It is private, never referenced by name anywhere in
    ``tests/``, and reached only through the non-bilinear predictive band -- so all three
    were unexercised.

    Oracle ``F1``/``A1``: shapes are structural, and the weights are the closed-form
    normalization ``w / sum(w)`` of the weights handed in (``[1, 2, 1] -> [0.25, 0.5,
    0.25]``), computed here rather than read back from the source.  Catches the MAP branch
    returning unnormalized or wrong-shaped output and ``_weights`` being bypassed so every
    sample counted equally.

    **The 1-D ``.samples`` leg is now a raise, and
    ``test_flat_samples_are_rejected_instead_of_silently_scaling_the_band`` pins it.**
    The old behaviour was a reshape to one row, which disagreed with ``fit._weights``
    (it reads ``samples.samples.shape[0]`` independently and returned ``n`` weights);
    ``zip`` truncated and scaled the band by ``1/n``.  See that test for the
    measurement and for why rejecting beats picking a reading.
    """
    single, weights = runner._sample_vectors_and_weights(
        SimpleNamespace(x=np.array([1.0, 2.0]))
    )
    assert single.shape == (1, 2)
    np.testing.assert_allclose(weights, [1.0])

    drawn, weights = runner._sample_vectors_and_weights(
        SimpleNamespace(
            samples=np.arange(6.0).reshape(3, 2), weights=np.array([1.0, 2.0, 1.0])
        )
    )
    assert drawn.shape == (3, 2)
    np.testing.assert_allclose(weights, [0.25, 0.5, 0.25])

    # The zero-parameter contract shape every sampler spells np.zeros((n, 0)) is
    # 2-D and must survive: rejecting it would break the fixed-parameter fits.
    empty, empty_weights = runner._sample_vectors_and_weights(
        SimpleNamespace(samples=np.zeros((4, 0)), weights=np.ones(4))
    )
    assert empty.shape == (4, 0)
    np.testing.assert_allclose(empty_weights, np.full(4, 0.25))


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("shape", [(3,), (2, 2, 2)])
def test_flat_samples_are_rejected_instead_of_silently_scaling_the_band(runner, shape):
    """A non-2-D ``.samples`` raises instead of yielding a band scaled by ``1/n``.

    ``.samples`` carries the sampler contract shape ``(n_samples, n_params)``.  Every
    shipped sampler allocates exactly that -- ``src/pixel/infer/hmc.py:1101`` and
    ``src/pixel/infer/nuts.py:1100`` as ``np.empty((n_samples, n_params))``,
    ``infer/vegas.py:1036`` and ``infer/mcmc.py:383`` by stacking rows -- and all four
    write the zero-parameter case as ``np.zeros((n_samples, 0))``, which is still 2-D
    (pinned by the previous test).  So a 1-D ``.samples`` is *out of contract*, not an
    alternative encoding.

    **Measured before the fix**, ``.samples = np.array([1.0, 2.0, 3.0])``:
    ``_sample_vectors_and_weights`` reshaped it to a single row of length 3 while
    ``fit._weights`` -- reading ``samples.samples.shape[0]`` -- returned three weights of
    ``1/3``.  ``_sampled_linear_predictive_band``'s ``zip(vectors, weights)`` truncated to
    one pair, so ``mean_acc`` accumulated total weight ``0.333`` instead of ``1.0``.
    Measured against the same single vector carried at unit weight, on
    ``_sampled_band_model``: the reported mean came back at exactly ``1/n``
    (``0.33333333`` on all three rows) and the one-sigma band at ``1.20``/``1.36``/``1.09``
    times the correct value -- ``var = w*var + (w - w**2)*mean**2`` is not a clean
    rescaling, so the band was wrong in the *other* direction and nothing raised.
    Picking either reading would have frozen a guess; refusing the shape cannot.

    ``shape=(2, 2, 2)`` covers the other half: the old ``ndim == 1`` test let a 3-D
    ``.samples`` through untouched, so the guard is written ``!= 2``, not ``== 1``.

    Oracle: the contract itself (the four sampler allocations above), not this code.
    A control asserts the in-contract 2-D shape raises nothing, so a validator that
    rejected everything could not pass.
    """
    flat = SimpleNamespace(samples=np.arange(float(np.prod(shape))).reshape(shape))
    with pytest.raises(ValueError, match=r"must be 2-D"):
        runner._sample_vectors_and_weights(flat)

    # Control: the in-contract shape is accepted, and its weights sum to one.
    ok, ok_weights = runner._sample_vectors_and_weights(
        SimpleNamespace(samples=np.arange(6.0).reshape(3, 2))
    )
    assert ok.shape == (3, 2)
    assert abs(float(np.sum(ok_weights)) - 1.0) < 1e-12


@pytest.mark.parametrize("runner", RUNNERS)
def test_predictive_band_refuses_a_vector_weight_count_mismatch(runner, monkeypatch):
    """``_sampled_linear_predictive_band`` names a count mismatch rather than zipping it away.

    ``zip`` truncates to the shorter sequence, so any future divergence between the
    vector count and ``fit._weights``'s count reappears as a band renormalized by the
    wrong total -- the exact failure the 1-D reshape produced.  The guard is fed by
    replacing ``fit._weights`` with one returning five weights for a three-row
    ``.samples``; the error message must name both lengths so the report is actionable.

    Control: the same model and samples with the real ``_weights`` return finite
    moments and raise nothing, so this cannot pass on a band that rejects every input.
    """
    model, _ = _sampled_band_model()
    samples = SimpleNamespace(samples=np.array([[0.3, 0.0], [-0.7, 0.0], [1.1, 0.0]]))

    mean, std = runner._sampled_linear_predictive_band(model, samples)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))

    monkeypatch.setattr(runner.fitmod, "_weights", lambda s: np.full(5, 0.2))
    with pytest.raises(ValueError, match=r"3 vectors, 5 weights"):
        runner._sampled_linear_predictive_band(model, samples)


def _sampled_band_model(rcond=1e-12):
    """A model whose ``_blocks`` returns a *different* ``EvidenceBlocks`` per sample.

    Two blocks differing only in ``param_pred`` (offset by the sample's own value), so
    the mixture the band accumulates has a genuine between-sample spread: with identical
    blocks the law-of-total-variance term the test checks would be exactly zero and the
    oracle could not tell an accumulation bug from a correct one.
    """
    rng = np.random.default_rng(5)
    n_data, n_grid = 3, 2
    base = rng.normal(size=(n_data, n_data))
    prior = rng.normal(size=(n_grid, n_grid))
    template = EvidenceBlocks(
        M=np.asarray(rng.normal(size=n_data)),
        C=np.asarray(base @ base.T + 0.4 * np.eye(n_data)),
        B=np.asarray(rng.normal(size=(n_data, n_grid))),
        g=np.asarray(rng.normal(size=n_grid)),
        K=np.asarray(prior @ prior.T + 0.4 * np.eye(n_grid)),
        param_pred=np.asarray(rng.normal(size=n_data)),
    )

    def blocks_for(theta):
        offset = float(np.asarray(theta).reshape(-1)[0])
        return template._replace(param_pred=template.param_pred + offset)

    return SimpleNamespace(
        rcond=rcond,
        _rebuild=lambda vec: vec,
        _blocks=blocks_for,
        _layout=SimpleNamespace(M=np.zeros(n_data)),
    ), blocks_for


@pytest.mark.parametrize("runner", RUNNERS)
def test_sampled_linear_band_mixes_per_sample_posterior_moments(runner):
    """The non-bilinear predictive band is the weighted *mixture* of the per-sample
    Gaussian bands, not an average of their means.

    ``_sampled_linear_predictive_band`` (``plot_datasets.py:68-97``) is the branch
    ``_posterior_predictive_bands`` takes whenever the model has no bilinear rows or the
    sampler stored no marginalized GP moments -- i.e. every ordinary MCMC/NUTS closure
    run.  Nothing in ``tests/`` reached it: this file's only record-level test sets
    ``has_bilinear=True`` with valid stored moments, which takes the other branch.

    Oracle: the law of total variance, ``Var = E[Var_s] + Var[E_s]``, written out here as
    ``sum_s w_s (var_s + mean_s^2) - (sum_s w_s mean_s)^2`` (``A1`` for the mixture
    combination).  The per-sample moments come from the same ``pixel.core.evidence.
    posterior`` the source calls -- deliberately, since what is under test is the
    accumulation across samples, not the single-point posterior (which
    ``tests/test_joint_action.py`` covers against an independent Cholesky path).  Two
    samples with unequal weights (0.25/0.75) and a per-sample ``param_pred`` offset of 4.0
    make the between-sample term dominant rather than incidental: measured on this
    fixture as ``max|a/b - 1|``, dropping that term changes the reported standard
    deviation by 38.7%, and weighting the two samples equally changes it by 9.9%.  The
    closing assertion measures the same ratio (0.631 achieved against a 0.1 bar), so a
    later fixture edit that quietly flattened the mixture would fail rather than pass
    vacuously.

    Bar ``rtol=1e-12``.  Measured max relative difference exactly 0.0 for both the mean
    and the standard deviation -- test and source happen to accumulate in the same order,
    so the bar is headroom against a reordering, not an achieved accuracy floor, and this
    is a transcription pin rather than an accuracy claim.
    """
    model, blocks_for = _sampled_band_model()
    vectors = np.array([[0.0], [4.0]])
    samples = SimpleNamespace(samples=vectors, weights=np.array([1.0, 3.0]))

    mean, std = runner._sampled_linear_predictive_band(model, samples)

    expected_mean = 0.0
    expected_second = 0.0
    for weight, vector in zip([0.25, 0.75], vectors):
        blocks = blocks_for(vector)
        qbar, covariance = posterior(blocks, model.rcond)
        bmat = np.asarray(blocks.B, dtype=float)
        row_mean = np.asarray(blocks.param_pred, dtype=float) + bmat @ np.asarray(qbar)
        row_var = np.einsum("ij,jk,ik->i", bmat, np.asarray(covariance), bmat)
        expected_mean = expected_mean + weight * row_mean
        expected_second = expected_second + weight * (row_var + row_mean * row_mean)
    expected_std = np.sqrt(expected_second - expected_mean * expected_mean)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(std, expected_std, rtol=1e-12)
    # Non-degeneracy: the between-sample term is a real part of the answer here
    # (measured 0.631 against the 0.1 bar), so an implementation that dropped it
    # would not agree by accident.
    within_sample = np.sqrt(
        sum(
            weight
            * np.einsum(
                "ij,jk,ik->i",
                np.asarray(blocks_for(vector).B, dtype=float),
                np.asarray(posterior(blocks_for(vector), model.rcond)[1]),
                np.asarray(blocks_for(vector).B, dtype=float),
            )
            for weight, vector in zip([0.25, 0.75], vectors)
        )
    )
    assert np.max(np.abs(std / within_sample - 1.0)) > 0.1


@pytest.mark.parametrize("runner", RUNNERS)
def test_sampled_linear_band_reuses_stored_moments_without_rebuilding(runner):
    """A sampler that already stored linear predictive moments short-circuits the
    per-sample loop entirely.

    ``plot_datasets.py:70-76`` returns ``samples._pixel_linear_predictive_moments``
    directly when its two arrays match ``model._layout.M`` in size -- the fast path for a
    sampler that accumulated the band itself, and the reason a full closure run does not
    re-solve the posterior once per stored sample.  It is guarded on size precisely
    because a stale cache of the wrong length must not be trusted.

    Both legs are checked with a model whose ``_blocks`` raises: the cached case must
    return the stored arrays without ever touching it, and the wrong-size case must fall
    through to the loop (observed as that exception escaping).  Oracle ``F1``: identity
    of the returned values plus a reachability assertion.  Catches the size guard being
    dropped (a mismatched cache silently truncating or misaligning every plotted band)
    and the cache being ignored.
    """
    def explode(_theta):
        raise AssertionError("cached moments must not rebuild the blocks")

    model = SimpleNamespace(
        rcond=1e-12,
        _rebuild=lambda vec: vec,
        _blocks=explode,
        _layout=SimpleNamespace(M=np.zeros(3)),
    )
    stored = (np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    samples = SimpleNamespace(
        samples=np.zeros((2, 1)), _pixel_linear_predictive_moments=stored
    )

    mean, std = runner._sampled_linear_predictive_band(model, samples)
    np.testing.assert_array_equal(mean, stored[0])
    np.testing.assert_array_equal(std, stored[1])

    # Wrong length: the guard must reject the cache and fall through to the loop.
    samples._pixel_linear_predictive_moments = (np.zeros(2), np.zeros(2))
    with pytest.raises(AssertionError, match="must not rebuild"):
        runner._sampled_linear_predictive_band(model, samples)


def _record(name, kind, x, *, use_log=False):
    """A well-formed ``plot_records`` record with ``x.size`` rows."""
    x = np.asarray(x, dtype=float)
    return {
        "name": name,
        "kind": kind,
        "x": x,
        "xlabel": "Y" if kind == "dy" else "x",
        "use_log": use_log,
        "data": np.linspace(1.0, 1.3, x.size),
        "sigma": np.full(x.size, 0.1),
        "truth": np.linspace(1.05, 1.2, x.size),
        "reproduction": np.linspace(1.02, 1.24, x.size),
        "fit_std": np.full(x.size, 0.08),
        "chi2": 0.5,
        "chi2_per_data": 0.5 / x.size,
        "n_data": int(x.size),
    }


def test_plot_records_renders_the_band_the_log_scale_and_the_dy_footnote(
    tmp_path, monkeypatch
):
    """The rendered figure is inspected, not just its file: one panel per record, the
    ``fill_between`` band only where there are at least two points, the log scale only
    where the record asks for it, and the DY footnote only when a DY record is present.

    ``plot_records`` (``closure_JAM_truth/plot_datasets.py:334-396``) closes its figure
    before returning, so nothing downstream can inspect it; this test intercepts
    ``matplotlib.pyplot.close`` to keep the ``Figure`` alive and then reads the artists
    back, the pattern ``tests/test_plotting.py:381`` already uses for ``pixel.plotting``.
    Three branches that no test reached are covered here: ``x.size < 2`` (the errorbar
    fallback in place of the band), ``use_log`` (``set_xscale("log")``), and the
    ``any(kind == "dy")`` footnote.

    Oracle ``F1`` (artist presence/type and axis state).  The band is identified as a
    ``PolyCollection`` specifically: ``errorbar`` also populates ``ax.collections``, with
    a ``LineCollection``, so a count of ``ax.collections`` alone could not tell the two
    branches apart -- measured, **both** panels carry exactly 2 collections, and only the
    three-point one contains a ``PolyCollection``
    (``['LineCollection', 'FillBetweenPolyCollection']`` against
    ``['LineCollection', 'LineCollection']``).

    Catches: the band branch inverted or dropped (a fit band silently missing from every
    multi-point panel), ``set_xscale`` applied unconditionally or never, the footnote
    dropped or attached to non-DY figures, and a panel count that does not match the
    record count.  Says nothing about colours, data values plotted, or layout.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    captured = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)

    records = [
        _record("dy_table", "dy", [-0.3, 0.0, 0.4]),          # band branch
        _record("dis_table", "linear", [0.2], use_log=True),  # errorbar + log branch
    ]
    output = tmp_path / "dataset_reproduction_both.png"
    jam_small.plot_records(records, output, q_key="2", mode="both")

    assert len(captured) == 1
    figure = captured[0]
    try:
        assert len(figure.axes) == 2  # one panel per record, no spares
        band_axis, fallback_axis = figure.axes
        assert [c for c in band_axis.collections if isinstance(c, PolyCollection)]
        assert not [
            c for c in fallback_axis.collections if isinstance(c, PolyCollection)
        ]
        assert band_axis.get_xscale() == "linear"  # use_log=False
        assert fallback_axis.get_xscale() == "log"  # use_log=True
        assert "dy_table" in band_axis.get_title()
        assert "[DY]" in band_axis.get_title()
        assert "[DY]" not in fallback_axis.get_title()
        footnotes = [
            text.get_text() for text in figure.texts
            if "first-order propagation" in text.get_text()
        ]
        assert len(footnotes) == 1
    finally:
        real_close(figure)

    # The same call still writes both formats (previously the only assertions).
    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()


def test_linear_only_records_carry_no_dy_footnote(tmp_path, monkeypatch):
    """The DY footnote is conditional: a figure with no DY record does not get it.

    Companion to ``test_plot_records_renders_the_band_the_log_scale_and_the_dy_footnote``
    -- the ``if any(rec.get("kind") == "dy" ...)`` guard at
    ``closure_JAM_truth/plot_datasets.py:387`` cannot be shown to be a guard by a case
    that always satisfies it.  Without this control, a footnote written unconditionally
    would pass the positive test and mislabel every purely linear closure figure as using
    first-order DY propagation.  Oracle ``F1``.
    """
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)

    jam_small.plot_records(
        [_record("dis_table", "linear", [0.1, 0.2, 0.3])],
        tmp_path / "linear_only.png",
        q_key="2",
        mode="both",
    )
    figure = captured[0]
    try:
        assert not [
            text for text in figure.texts
            if "first-order propagation" in text.get_text()
        ]
    finally:
        real_close(figure)


def test_dy_records_render_png_and_pdf(tmp_path):
    """``plot_records`` writes both a PNG and a PDF sibling; file
    existence only -- no rendered content is inspected here.

    Checks ``output.is_file()`` and ``output.with_suffix(".pdf").is_file()``
    after a real, unmocked call.  This confirms ``save_figure_both``'s
    suffix derivation and that the per-panel matplotlib calls complete
    without raising for a well-formed record (``x.size == 3`` exercises
    the ``fill_between`` band branch, not the ``x.size < 2`` errorbar
    fallback; ``use_log=False`` never exercises ``set_xscale("log")``).
    The rendered content -- band artist, axis scale, DY footnote, panel
    count -- is asserted in
    ``test_plot_records_renders_the_band_the_log_scale_and_the_dy_footnote``
    above, which intercepts ``plt.close`` to keep the ``Figure`` alive;
    this test deliberately keeps the unmocked end-to-end write path, since
    that one does not (``plt.close`` is patched there, so the figure is
    never released by the code under test).
    Catches: a dropped output format, a wrong suffix substitution, or an
    exception from mismatched record-array lengths.  Does not catch: wrong
    data, wrong colors, or wrong values plotted.
    """
    record = {
        "name": "dy_table",
        "kind": "dy",
        "x": np.array([-0.3, 0.0, 0.4]),
        "xlabel": "Y",
        "use_log": False,
        "data": np.array([1.0, 1.3, 0.9]),
        "sigma": np.array([0.1, 0.12, 0.08]),
        "truth": np.array([1.05, 1.2, 0.95]),
        "reproduction": np.array([1.02, 1.24, 0.93]),
        "fit_std": np.array([0.08, 0.09, 0.07]),
        "chi2": 0.5,
        "chi2_per_data": 1.0 / 6.0,
        "n_data": 3,
    }
    output = tmp_path / "dataset_reproduction_dy.png"

    jam_small.plot_records([record], output, q_key="2", mode="dy")

    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
