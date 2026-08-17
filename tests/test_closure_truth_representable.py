"""Generation must use one PDF truth for linear and bilinear observables.

Exercises ``closure_JAM_truth.generate``/``closure_NNPDF_truth.generate`` --
``fold_truth`` and ``dy_central`` are byte-identical between the two copies
(confirmed by diff), so the two parametrized cases run the same mechanism twice,
guarding against the copies drifting apart rather than against two different
implementations -- plus the real ``pixel.core.model.BilinearContribution.assemble``
``dy_central`` calls (not a shim).

Both oracles are plain ``numpy`` written directly in the test, independent of
``gen``: a matrix-vector product for the linear fold, an outer-product bilinear
contraction for the DY fold.  The bar is ``assert_allclose``'s default (rtol=1e-7,
atol=0): the path under test is exact linear algebra with no interpolation or
quadrature, so this is a transcription-level pin, not a physics accuracy bound.

The property unique to this file -- not covered by the far deeper
``tests/test_closure_dy_evolution.py`` (real kernels, evolution dispatch,
multi-field grouping, a reference "written differently on purpose") -- is that
``dy_central`` folds ``outer(truth, truth)``, not a replica-ensemble second
moment.  Measured on this file's own fixture the two laws differ by 7-8%
(``[28., 72.1]`` fixed-truth vs. ``[30., 77.7]`` second-moment, the latter from
``replicas`` whose mean is exactly the truth vector used for the former).

**That alternative was not hypothetical, and this file's scope gap is why it
survived.**  Until 2026-08-14 it was, term for term, what
``closure_JAM_truth_small.generate.dy_central`` and
``closure_NNPDF_truth_small.generate.dy_central`` computed
(``C = qA.T @ qB / n_rep``) -- while both ``_small`` suites record the ensemble
*mean* in ``truth.json`` (``truth_kind: "replica_ensemble_mean"``, confirmed on
all six ``truthQ_*`` members of both), so the DY rows were generated from a
central value no single PDF reproduces.  Both ``_small`` copies now run the same
body as the full suites, and ``ALL_DY_PACKAGES`` below names all *four* so
``test_all_four_dy_central_bodies_are_the_same_source`` can assert they have not
drifted apart again.  Measured on
the shipped ensembles at ``Q = 2``, second moment vs. fixed truth: JAM
``2.4e-03``/``2.9e-03``, NNPDF ``2.1e-03``/``6.4e-03`` -- far smaller than this
file's synthetic 7-8%, but a truth-level bias rather than noise.

**The behavioural parametrizations are NNPDF-only since 2026-08-14** (owner's
instruction): ``DY_PACKAGES`` and ``LINEAR_PACKAGES`` name only the NNPDF
packages, while ``ALL_DY_PACKAGES`` keeps all four for the identity test.  The
guard against the S0-05 recurrence is therefore the executable-body assertion,
not the width of the sweep -- which is the stronger of the two: the sweep would
catch a divergent ``_small`` copy only if the divergence happened to change what
these fixtures measure, whereas the identity test sees any change to the body at
all.  That distinction is exactly what let S0-05 survive originally.

The ``fold_truth`` (linear) checks stay on the full-size package: the ``_small``
suites have no ``fold_truth``, folding their ensemble through
``fold_ensemble``/``assemble_operator`` instead.  Linearity makes
``mean(A q_m) == A mean(q_m)`` exactly there, which is precisely the identity a
bilinear observable does not have -- the reason the bug could only live in the
DY fold.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pixel.core.model import BilinearContribution

#: **Every** package defining ``dy_central`` -- all four.
#:
#: Read by exactly one test, ``test_all_four_dy_central_bodies_are_the_same_source``,
#: whose subject *is* the four copies: it asserts their parsed executable bodies collapse
#: to one.  Trimming this list would make that assertion compare fewer copies and quietly
#: stop guarding the two JAM ones -- and a silently divergent ``_small`` copy is precisely
#: the S0-05 bug this file was written for.  It looks like a redundant superset of
#: ``DY_PACKAGES`` below.  It is not -- keep all four.
ALL_DY_PACKAGES = [
    "closure_JAM_truth",
    "closure_NNPDF_truth",
    "closure_JAM_truth_small",
    "closure_NNPDF_truth_small",
]

#: The package defining ``fold_truth`` that the linear tests run against.  The
#: ``_small`` suites fold their replica ensemble instead and have no such function;
#: ``closure_JAM_truth`` was dropped 2026-08-14 (owner's instruction) -- its
#: ``fold_truth`` is the NNPDF one's twin and nothing here reads a truth PDF set.
LINEAR_PACKAGES = ["closure_NNPDF_truth"]

#: The packages the two bilinear tests are parametrized over: one full, one ``_small``,
#: NNPDF truth only since 2026-08-14 (owner's instruction).  One entry per *size*, which
#: is the axis that used to differ; the JAM/NNPDF axis is covered -- more strictly -- by
#: the identity test over ``ALL_DY_PACKAGES``.
DY_PACKAGES = ["closure_NNPDF_truth", "closure_NNPDF_truth_small"]


def test_all_four_dy_central_bodies_are_the_same_source():
    """``dy_central`` is one implementation, copied -- asserted, not assumed.

    The ``_small`` copies previously diverged into a replica second moment and
    nothing failed, because no test called them.

    **This test is why ``ALL_DY_PACKAGES`` exists and must keep all four entries.**
    It is that list's only reader.  Since 2026-08-14 the two bilinear tests are
    parametrized over the NNPDF pair (``DY_PACKAGES``) rather than all four, so this
    is the *only* thing standing between a hand-edited
    ``closure_JAM_truth*/generate.dy_central`` and silence.  It is also the stronger
    guard of the two: a wide behavioural sweep catches a divergent copy only if the
    divergence happens to change what these fixtures measure, which is exactly the
    reason S0-05 survived in the first place, while this sees any change to the body.

    Compared as the parsed **executable** body -- ``ast.dump`` after dropping the
    leading docstring -- not as source text.  The four docstrings deliberately
    differ (the ``_small`` pair records the second-moment history and its
    measurements), and a text comparison would either force those to be identical
    or, worse, be silently satisfied by a shared prose edit that missed the code.

    Non-degeneracy: the same comparison run on ``fold_truth`` (a different
    function) must *not* match ``dy_central``, so this cannot pass on a
    normalization that flattens everything to the same string.
    """
    import ast
    import inspect
    import textwrap

    def executable_body(func):
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        (fn,) = tree.body
        if (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        ):
            fn.body = fn.body[1:]
        return ast.dump(fn, annotate_fields=True)

    bodies = {}
    for package in ALL_DY_PACKAGES:
        gen = __import__(f"{package}.generate", fromlist=["generate"])
        bodies[package] = executable_body(gen.dy_central)
    distinct = set(bodies.values())
    assert len(distinct) == 1, (
        "dy_central copies have drifted: "
        + ", ".join(sorted(bodies)) + f" give {len(distinct)} distinct bodies"
    )

    # Control: the normalization is not so aggressive that any two functions match.
    full = __import__("closure_JAM_truth.generate", fromlist=["generate"])
    assert executable_body(full.fold_truth) not in distinct


@pytest.mark.parametrize("package", DY_PACKAGES)
def test_bilinear_fold_uses_the_ensemble_mean_not_the_ensemble_second_moment(package):
    """``dy_central`` folds ``outer(E[q], E[q])``, never ``E[q q^T]``.

    This is the S0-05 property, and it is the one the two ``_small`` packages
    used to get wrong: they folded ``C = qA.T @ qB / n_rep`` over the replica
    stack, i.e. ``E[qA] E[qB] + Cov(qA, qB)``, while writing ``E[q]`` to
    ``truth.json`` as the truth.  A bilinear observable has no
    ``mean-of-folds == fold-of-mean`` identity, so the extra covariance term is a
    bias with no truth curve behind it and coverage for the DY sector is
    undefined.

    Fixture: a real ``replicas`` stack, deliberately **non-degenerate in the
    dimension under test** -- its per-node spread is nonzero, so ``E[q q^T]``
    genuinely differs from ``outer(E[q], E[q])`` (asserted below, 7-8% on this
    fixture; measured 2.1e-03..6.4e-03 on the shipped JAM/NNPDF ensembles).  The
    truth handed to ``dy_central`` is exactly ``replicas.mean(axis=0)``, so the
    two laws are being asked about the *same* ensemble and only the law differs.

    Oracle ``A1``: both contractions are written out here in plain ``numpy``,
    independent of ``gen``.  The second-moment control is what makes the first
    assertion mean something -- without it a fold that ignored its ``truth``
    argument entirely could still pass.
    """
    gen = __import__(f"{package}.generate", fromlist=["generate"])

    tensor = np.arange(18.0).reshape(2, 3, 3) / 10.0
    replicas = np.array([[0.5, 1.5, 3.0], [1.5, 2.5, 5.0]])
    mean = replicas.mean(axis=0)
    assert np.all(replicas.std(axis=0) > 0.0)  # non-degenerate in the test's own axis

    class Kernel:
        """Stub bilinear kernel: fixed tensor, ignores ``nu``/``basis``."""

        parton_A = "u"
        parton_B = "u"

        @staticmethod
        def bilinear_tensor(nu, basis_a, basis_b):
            """Ignore every argument; return the enclosing ``tensor``."""
            return tensor

    dataset = SimpleNamespace(
        n_data=2,
        nu=np.array([0.2, 0.4]),
        bilinear_contributions=[BilinearContribution("q", "q", Kernel())],
    )
    fields = {"q": SimpleNamespace(basis=object())}

    got = gen.dy_central(dataset, {"q": mean}, fields, "pp")
    fixed_truth = np.einsum("rij,ij->r", tensor, np.outer(mean, mean))
    np.testing.assert_allclose(got, fixed_truth, rtol=1e-13)

    # Control: the law this replaced.  E[q q^T] over the same replicas, which is
    # what the _small copies used to contract.
    second_moment = np.einsum(
        "rij,ij->r", tensor, replicas.T @ replicas / replicas.shape[0]
    )
    ratio = np.abs(second_moment / fixed_truth - 1.0)
    assert ratio.min() > 1e-2, f"fixture is degenerate: max ratio-1 = {ratio.max():.3e}"
    assert not np.allclose(got, second_moment)

    # Handing the fold the ensemble itself must RAISE, not silently reduce.
    # np.outer flattens, so without the shape guard a (n_rep, n_nodes) stack
    # builds a (6, 6) outer product and the einsum either mis-contracts or --
    # with a "helpful" reshape -- lands back on the second moment.  Matched on
    # the message, not the type: several things here raise ValueError.
    with pytest.raises(ValueError, match=r"one truth curve per field"):
        gen.dy_central(dataset, {"q": replicas}, fields, "pp")
    # Control for the guard: the 1-D call above already returned, so this cannot
    # be passing on a dy_central that rejects everything.


@pytest.mark.parametrize("package", LINEAR_PACKAGES)
def test_linear_and_bilinear_generation_fold_the_same_fixed_truth(package, monkeypatch):
    """Linear ``fold_truth`` and bilinear ``dy_central`` fold one truth vector, not
    a replica statistic.

    Pins ``cfg.FIXED_TRUTH_MEMBER == 1`` -- config.py's designated closure-truth
    member; unrelated in mechanism to the checks below, since it fixes *which*
    member counts as truth, not *whether* a single member or an ensemble statistic
    is used.  Then checks two closed-form identities against oracles written
    directly in ``numpy``, independent of ``gen``: ``fold_truth(operator, truth)
    == matrix @ truth["q"]`` (linear) and ``dy_central(dataset, truth, fields,
    "pp") == einsum("rij,i,j->r", tensor, truth["q"], truth["q"])`` (bilinear, via
    a real ``BilinearContribution`` whose ``kernel.bilinear_tensor`` stub ignores
    ``nu``/``basis`` and returns a fixed tensor).  Both use ``assert_allclose``'s
    default rtol=1e-7: exact arithmetic, so the achieved residual is float noise.

    The closing ``assert not np.allclose(exact, second_moment)`` is not a further
    call into ``gen``: by that point ``exact`` already equals the formula above,
    so this reduces to checking that the replica-second-moment law disagrees with
    the fixed-truth law on this fixture (measured 7-8%, see module docstring) --
    i.e. it certifies the *earlier* assertion would have failed had ``dy_central``
    implemented that law, which is what makes that earlier pass mean something.

    Does not catch: ``field_A == field_B == "q"`` makes ``outer(truth, truth)``
    symmetric, so a transposed contraction (``"rji,ij->r"`` for ``"rij,ij->r"``)
    or a swapped ``field_A``/``field_B`` gives the identical number and would
    still pass -- verified directly on this fixture.  The stub kernel also ignores
    ``nu`` and ``basis``, so neither is actually exercised here.  Real kernels,
    multi-field contributions, evolution dispatch and weight scaling are covered
    independently in ``tests/test_closure_dy_evolution.py``.
    """
    cfg = __import__(f"{package}.config", fromlist=["config"])
    gen = __import__(f"{package}.generate", fromlist=["generate"])
    # Which member counts as "the" truth (config.py: "never used as a nonlinear
    # central value"); a config pin, not a check of the single-truth-vs-ensemble
    # property the rest of this test targets.
    assert cfg.FIXED_TRUTH_MEMBER == 1

    truth = {"q": np.array([1.0, 2.0, 4.0])}
    matrix = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]])
    np.testing.assert_allclose(gen.fold_truth([("q", matrix)], truth), matrix @ truth["q"])
    # rtol=1e-7 (assert_allclose default).  Single-field operator only -- the
    # ``for field, B in operator[1:]`` accumulation in fold_truth for a
    # multi-field operator is not exercised by this call.

    tensor = np.arange(18.0).reshape(2, 3, 3) / 10.0

    class Kernel:
        """Stub bilinear kernel: ignores ``nu``/``basis_a``/``basis_b``, always
        returns the fixed ``tensor`` -- so this fixture cannot show whether
        ``dy_central`` threads the right kinematics or basis through, only that
        whatever tensor comes back is contracted with the right truth values.
        """

        parton_A = "u"
        parton_B = "u"

        @staticmethod
        def bilinear_tensor(nu, basis_a, basis_b):
            """Ignore every argument; always return the module-level ``tensor``."""
            return tensor

    # Generation reads the operator off the contribution rather than re-deriving
    # the parton -> field map from config: with evolution composed on the fit
    # side, a second derivation here is how the closure invariant breaks without
    # anything failing.  So the fixture is a real contribution, not a bare kernel.
    dataset = SimpleNamespace(
        n_data=2,
        nu=np.array([0.2, 0.4]),
        bilinear_contributions=[BilinearContribution("q", "q", Kernel())],
    )
    fields = {"q": SimpleNamespace(basis=object())}
    exact = gen.dy_central(dataset, truth, fields, "pp")
    expected = np.einsum("rij,i,j->r", tensor, truth["q"], truth["q"])
    np.testing.assert_allclose(exact, expected)
    # rtol=1e-7 default; exact algebra, not a physics bound.  field_A == field_B
    # == "q" makes this blind to an index-transposed contraction
    # (einsum("rji,ij->r", ...) for "rij,ij->r") or a field_A/field_B swap: both
    # give this same number, because outer(truth["q"], truth["q"]) is symmetric.
    # test_bilinear_fold_is_index_weight_and_kinematics_faithful below carries the
    # asymmetric fixture that does separate those; this single-contribution case
    # is kept because its expected value is readable by inspection.

    # This does not call gen.dy_central again: by this point ``exact`` already
    # equals ``expected`` (the assert above), so the check below reduces to
    # ``expected != second_moment`` on this fixture -- proof that the earlier
    # assert_allclose *would* have failed had dy_central implemented the
    # replica-second-moment law instead of the fixed-truth law.  Measured 7-8%
    # apart; replicas.mean(axis=0) is exactly truth["q"].  That alternative law is
    # not hypothetical -- see the module docstring for where it actually ships.
    replicas = np.array([[0.5, 1.5, 3.0], [1.5, 2.5, 5.0]])
    second_moment = np.mean(
        [np.einsum("rij,i,j->r", tensor, replica, replica) for replica in replicas],
        axis=0,
    )
    assert not np.allclose(exact, second_moment)
    # np.allclose default rtol=1e-5, atol=1e-8 -- far below the 7-8% gap measured
    # here, not a tight bound.


@pytest.mark.parametrize("package", LINEAR_PACKAGES)
def test_linear_generation_fold_accumulates_a_multi_field_operator(package):
    """``fold_truth`` sums ``B_k @ truth[field_k]`` over *every* operator entry.

    The sibling single-field call in
    ``test_linear_and_bilinear_generation_fold_the_same_fixed_truth`` never enters
    ``fold_truth``'s ``for field, B in operator[1:]`` accumulation loop, which is
    the shape a production lattice operator actually has (physical kernel plus one
    ``Unit`` row per enabled nuisance field, assembled by ``assemble_operator``).
    Oracle is plain ``numpy`` written here (``A1``): three separate matrix-vector
    products summed by hand, with per-field matrices and truths chosen so that no
    two terms coincide and no term is negligible.

    Controls below make the fixture's discriminating power explicit rather than
    assumed: dropping any single term, or pairing a matrix with the wrong field's
    truth, each changes the answer by a measured factor -- so a fold that silently
    ignored ``operator[1:]``, or that read every ``B`` against ``operator[0]``'s
    field, would fail here.
    """
    gen = __import__(f"{package}.generate", fromlist=["generate"])

    truth = {
        "q": np.array([1.0, 2.0, 4.0]),
        "g": np.array([-0.5, 3.0, 1.25]),
        "s": np.array([2.5, -1.0, 0.75]),
    }
    mats = {
        "q": np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]]),
        "g": np.array([[1.5, 0.25, -2.0], [-1.0, 0.5, 4.0]]),
        "s": np.array([[-0.75, 1.0, 3.5], [2.0, -2.5, 0.125]]),
    }
    operator = [(name, mats[name]) for name in ("q", "g", "s")]
    expected = sum(mats[name] @ truth[name] for name, _ in operator)
    np.testing.assert_allclose(gen.fold_truth(operator, truth), expected, rtol=1e-13)
    # rtol=1e-13 against a hand-summed numpy oracle; the path is three matvecs and
    # two adds, so the achieved residual is float noise (measured 0.0 exactly).

    # Non-degeneracy of the fixture, in the two dimensions this test claims to
    # cover.  Without these a fold that dropped a term, or mismatched matrices to
    # truths, could still land on `expected` by arithmetic accident.
    for dropped in ("q", "g", "s"):
        partial = sum(
            mats[name] @ truth[name] for name, _ in operator if name != dropped
        )
        assert not np.allclose(partial, expected), f"dropping {dropped} is invisible"
    mismatched = sum(mats[name] @ truth["q"] for name, _ in operator)
    assert not np.allclose(mismatched, expected)


@pytest.mark.parametrize("package", DY_PACKAGES)
def test_bilinear_fold_is_index_weight_and_kinematics_faithful(package):
    """``dy_central`` contracts ``outer(A, B)`` in that order, honours ``weight``,
    groups by kernel identity, and threads the dataset's ``nu``/basis through.

    The sibling test's only bilinear case is ``field_A == field_B == "q"``, which
    makes ``outer(truth, truth)`` symmetric and therefore cannot separate
    ``einsum("rij,ij->r")`` from ``einsum("rji,ij->r")``, nor ``outer(A, B)`` from
    ``outer(B, A)`` -- both realistic slips in ``generate.py``'s grouping code.
    This fixture is deliberately asymmetric in every axis under test: a non-symmetric
    ``tensor``, three contributions with ``field_A != field_B``, distinct truths per
    field, and three distinct non-unit weights.  Two kernel *instances* are used so
    the ``(id(kernel), id(evolution_A), id(evolution_B))`` grouping must build two
    groups and accumulate the first group's two outer products into one.

    Oracle (``A1``): the summed outer products and the two ``einsum`` contractions
    are written out here in plain ``numpy``, independent of ``gen``.  Bar
    ``rtol=1e-13``, tighter than ``assert_allclose``'s default, because the path is
    exact linear algebra.

    The four ``assert not np.allclose`` controls at the end measure that this
    fixture actually separates the four bug classes named above; each is the
    "would the earlier assertion have failed?" half that makes the pass mean
    something, in the same style as the sibling's second-moment control.
    """
    gen = __import__(f"{package}.generate", fromlist=["generate"])

    truth = {
        "q": np.array([1.0, 2.0, 4.0]),
        "g": np.array([-0.5, 3.0, 1.25]),
    }
    # +1 so no entry is zero; reshaped ranges are asymmetric under i <-> j.
    tensor_a = (np.arange(18.0).reshape(2, 3, 3) + 1.0) / 10.0
    tensor_b = (np.arange(18.0, 0.0, -1.0).reshape(2, 3, 3)) / 7.0
    nu = np.array([0.2, 0.4])
    seen = []

    class Kernel:
        """Stub bilinear kernel returning a fixed tensor, recording its call args.

        Unlike the sibling test's stub this one keeps ``(nu, basis_a, basis_b)``
        so the test can assert the dataset's kinematics and the shared field basis
        were actually threaded through rather than ignored.
        """

        def __init__(self, tensor):
            self.tensor = tensor

        def bilinear_tensor(self, nu_, basis_a, basis_b):
            """Record ``(nu_, basis_a, basis_b)``; return this kernel's tensor."""
            seen.append((nu_, basis_a, basis_b))
            return self.tensor

    kernel_a = Kernel(tensor_a)
    kernel_b = Kernel(tensor_b)
    contributions = [
        BilinearContribution("q", "q", kernel_a, weight=0.5),
        BilinearContribution("q", "g", kernel_a, weight=2.0),
        BilinearContribution("g", "q", kernel_b, weight=-1.5),
    ]
    basis = object()
    dataset = SimpleNamespace(
        n_data=2, nu=nu, bilinear_contributions=contributions
    )
    fields = {"q": SimpleNamespace(basis=basis), "g": SimpleNamespace(basis=basis)}

    q, g = truth["q"], truth["g"]
    outer_a = 0.5 * np.outer(q, q) + 2.0 * np.outer(q, g)
    outer_b = -1.5 * np.outer(g, q)
    expected = np.einsum("rij,ij->r", tensor_a, outer_a) + np.einsum(
        "rij,ij->r", tensor_b, outer_b
    )
    exact = gen.dy_central(dataset, truth, fields, "pp")
    np.testing.assert_allclose(exact, expected, rtol=1e-13)

    # The stub is called once per *group*, not once per contribution, and with the
    # dataset's own nu and the shared field basis.
    assert len(seen) == 2
    for nu_seen, basis_a, basis_b in seen:
        assert np.array_equal(np.asarray(nu_seen), nu)
        assert basis_a is basis and basis_b is basis

    # Controls: this fixture separates each bug class from the correct answer.
    transposed = np.einsum("rji,ij->r", tensor_a, outer_a) + np.einsum(
        "rji,ij->r", tensor_b, outer_b
    )
    assert not np.allclose(transposed, expected)
    swapped_a = 0.5 * np.outer(q, q) + 2.0 * np.outer(g, q)
    swapped_b = -1.5 * np.outer(q, g)
    swapped = np.einsum("rij,ij->r", tensor_a, swapped_a) + np.einsum(
        "rij,ij->r", tensor_b, swapped_b
    )
    assert not np.allclose(swapped, expected)
    unweighted = np.einsum(
        "rij,ij->r", tensor_a, np.outer(q, q) + np.outer(q, g)
    ) + np.einsum("rij,ij->r", tensor_b, np.outer(g, q))
    assert not np.allclose(unweighted, expected)
    # One group instead of two (kernel identity ignored) would fold both outer
    # products through whichever tensor was assembled first.
    merged = np.einsum("rij,ij->r", tensor_a, outer_a + outer_b)
    assert not np.allclose(merged, expected)
