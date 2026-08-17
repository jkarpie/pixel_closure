"""Closure Drell--Yan evolution: the constant is live and both sides share it.

``DY_EVOLVE`` was a dead constant -- setting it changed nothing, because
``datasets.build_drell_yan`` hardcoded ``evo = None``.  These tests pin the two
properties that make the wired version trustworthy: the flag actually reaches
the built operator, and generation folds the *same* assembled operator the fit
uses rather than a separately written physics path.

Drives ``closure_JAM_truth`` and ``closure_NNPDF_truth`` **only** --
``config.py``, ``datasets.py`` (``build_drell_yan``, ``dy_evolution_maps``) and
``generate.py`` (``dy_central``), reached via ``importlib`` -- never the
``_small`` suites. That distinction matters for ``S0-05``
(``plans/test_audit/S0_FINDINGS.md``): the ``_small`` suites' ``dy_central``
computes its truth central value as an ensemble second moment, which the full
suite's own ``generate.py:494`` calls "deliberately forbidden ... cannot
define coverage for a bilinear observable." ``truth`` in every test here is
built as ``{field: rng.normal(...)}`` -- a single-PDF dict, one draw per basis
field, never an ensemble moment -- so this file does not reach the forbidden
path and neither exercises nor defends against S0-05; see
``tests/test_closure_truth_representable.py`` for where that suite-scope
distinction is actually asserted.

Real, unstubbed code exercised along the way: ``src/pixel/geometry`` (``Grid``,
``ProductBasis``); ``src/pixel/kernels/evolution/flavor_transition.py``'s
``nf4_parton_evolution_projection`` / ``rowwise_parton_evolution_projection``,
specifically the bottom-threshold crossing decision
(``Q20 < AlphaS.mb_match2 <= Q2``) and the nf=4->nf=5 parton bookkeeping; and
``src/pixel/data/drell_yan.py`` (``DrellYan``/``BilinearContribution``),
reached through ``datasets.build_drell_yan``. ``_stub_evolution`` monkeypatches
only the two leaf functions ``flavor_transition.non_singlet_evolution_matrix``
and ``.singlet_evolution_matrix`` (both bound into ``flavor_transition`` by a
plain ``from .assembly import ...``, so patching the attribute on that module
redirects every internal caller, including the matched-T24 threshold path) to
a cheap linear stand-in. No test in this file assembles a Mellin contour or
checks an evolution *operator's* value; the evolution operator's own numerical
accuracy belongs to ``test_dy_flavor_evolution.py`` and the fitpack DY
comparison, not here.

**Common-mode caveat that applies to the fold comparisons in this file**
(``tests/README.md`` rule 1: ask what the two sides share). Tests 2 and 3 each
compare two *foldings* of the same ``dataset.bilinear_contributions`` -- the
same ``field_A``/``field_B`` labels, the same
``c.assemble(...)``/``c.kernel.bilinear_tensor(...)`` tensors -- that differ
only in how the pieces are grouped or recombined (oracle ``A3``: a real check
of the fold arithmetic, not of what is being folded). A ``field_A``/``field_B``
mislabelling upstream in ``build_drell_yan``/``dy_evolution_maps``, or an
axis-transposed tensor inside ``assemble()`` itself, reproduces identically on
both sides of *those* comparisons and would pass them. DY is bilinear -- two
basis axes, one per hadron -- and this file's one ``_basis()`` fixture is
handed to both axes of every tensor, so neither of those tests is positioned
to catch that class of bug even in principle. Tests 2 and 3 keep that caveat;
what changed is that the file now carries two tests that *are* so positioned,
one per half of the exposure (see the two paragraphs below).

``test_assemble_composes_each_hadron_operator_onto_its_own_axis`` (added
2026-08-13) closes the ``assemble()`` half of that gap: it folds the *bare*
tensor with separately evolved curves and requires the result to reproduce the
assembled tensor's fold, an identity that fails if the two evolution operators
are exchanged. Getting there required removing two degeneracies that are worth
not rediscovering -- the ``pd`` reaction, so the two hadrons carry different
projections, and a non-proportional evolution stub, since ``_stub_evolution``
returns scalar multiples of the identity under which an axis swap is *exactly*
invariant (measured 0.000).

``test_dy_side_a_is_the_beam_and_side_b_carries_the_reaction_transform`` (added
2026-08-14) closes the other half -- the ``field_A``/``field_B`` *labelling*
itself, one level upstream in ``dy_evolution_maps``, whose return
``(proton.fields, target.fields, proton.evolution, target.evolution)`` can be
exchanged by a one-line edit that changes no shape and raises nothing. It uses
the function's own ``pp`` answer as the reference (for ``pp`` the target *is*
the beam) and requires the ``pd``/``ppbar`` A-sides to reproduce it bit for bit
while their B-sides reproduce ``isoscalar_projection``/``conjugate_projection``
of it. Measured: swapping the two halves in memory fails that test and **no
other test in this file** -- 1 failed / 7 passed, with the swapped function
called 7 times in the run, so the others execute it and are blind rather than
never reaching it. Two degeneracies had to be measured first, and both would
have produced a vacuous test: the ``fields`` maps are *identical* across all
three reactions (they record source membership only, which neither transform
changes), and charge conjugation is an involution, so any A-vs-B relation is
symmetric under the swap. Neither is visible from a green run.

Neither of these has a counterpart elsewhere at a comparable scale, as far as
searched: ``test_dy_channels.py``'s
``test_mirror_channel_swaps_A_and_B``/``test_qqbar_luminosity_flavor_activation``
use a hadron-B fixture proportional to hadron-A (documented ``W-TAUT``/
``W-COMMON`` there; a fix is proposed as ``test_dy_channels-M03``, not yet
landed), and ``test_dy_kernel.py``'s one externally-oracled test
(``test_dy_kernel_tensor_sum_matches_fitpack_legacy_benchmark_row``) runs at
``n_elements=1``, too degenerate for a transpose to be visible.

Every numeric bar in this file (``rtol=1e-12, atol=0.0``) is a float64
transcription pin, not a physics accuracy claim: both sides of each comparison
do the same floating-point contractions in a different grouping order, so the
achieved margin reflects summation-order noise, not a physical scale. There is
no production-size version of this file; the real full-basis, full-channel DY
dataset this exercises a slice of (1552 bilinear contributions per suite for
this file's fixture, measured -- see this file's audit report) is what the
closure fits themselves build and run.
"""

from __future__ import annotations

import functools
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pixel.geometry import Grid, ProductBasis
from pixel.kernels.drell_yan.kernel import DYKernel as _DYKernel

#: The real ``_assemble_bilinear_tensor``, captured at import before any test can patch it.
#: ``_assert_hard_tensor_memo_is_faithful`` needs an unmemoized path to compare against;
#: reading it off the class at call time would just fetch the memo and compare it with
#: itself.
_UNMEMOIZED_ASSEMBLE_BILINEAR_TENSOR = _DYKernel._assemble_bilinear_tensor


#: Only NNPDF.  ``closure_JAM_truth`` was dropped 2026-08-14 on the owner's instruction,
#: and it costs nothing measurable: ``generate.dy_central`` is **byte-identical** between the
#: two suites (verified by `inspect.getsource`, 3225 chars, `a == b`), so the second
#: parametrization re-ran the same function over an equivalent fixture.  The four `run_closure`
#: copies being one implementation is itself pinned by
#: `test_closure_plot_scaling`'s cross-suite identity test, so a future divergence between the
#: suites fails there rather than going unnoticed here.
#:
#: This halved the file's parametrized cost: the two ``[True-*]`` cases were **1186s each**,
#: the two ``[False-*]`` 213s and 196s -- 43% of the entire suite's node-time in one
#: parametrization.
SUITES = ("closure_NNPDF_truth",)


#: Survives across tests in this module.  Measured 2026-08-14: the file's four
#: tensor-building tests request 78 hard tensors carrying only **54 distinct values** --
#: ``[False]`` and ``test_unevolved_dy_central_reproduces_the_parton_precombined_fold`` run
#: the identical ``(pd, DY_EVOLVE=False, DY_NF=4)`` fixture and duplicate all 24 of theirs,
#: and the two the axis test needs are among the evolved 30.  The evolved and unevolved sets
#: do *not* overlap (``nf`` 5 vs 4 reaches ``_row_couplings``), so 54 is the true floor.
#: 54 x 864 B = 47 kB.
_HARD_TENSOR_CACHE: dict = {}


def _tensor_key(kernel, rows, basis_A, basis_B):
    """The value-identity of one hard tensor -- **every** ``DYKernel`` field, plus rows and bases.

    Enumerated with ``dataclasses.fields`` rather than transcribed.  That is the whole
    point, and it was measured: a hand-written field list omitting ``nf`` -- which is 5 in
    the evolved fixture and 4 in the unevolved one, and reaches the tensor through
    ``_row_couplings`` -- makes the two fixtures collide in this module-scoped cache, and
    **nothing in this file catches it**.  The reference loop and ``dy_central`` both read the
    same poisoned tensor, so they agree exactly as before; the suite went 7 passed in 10.5s
    against the honest 18.1s, i.e. the corruption presents as a *speed-up*.  That is
    ``CLAUDE.md``'s "agreement is not evidence" with a cache in the middle.

    So the key is complete by construction: a field added to ``DYKernel`` later enters it
    automatically, and no transcription step exists to forget one.  Over-keying is free here
    (``cache_path``, ``parallel_workers`` and the ``nnlo_*`` fields are constant across this
    file's fixtures), and over-keying only ever costs time, never correctness.
    ``_hard_tensor_memo_is_faithful`` below closes the residual case -- a tensor depending on
    something that is not a field at all.

    ``repr`` rather than the values themselves: ``couplings`` and ``electroweak`` are
    arbitrary objects with no hashability guarantee.  The bases are keyed by ``id``, exact
    here precisely because ``_basis()`` returns a shared singleton -- see its docstring for
    why that, and not a value key, is the safe choice.
    """
    return (
        tuple(repr(getattr(kernel, f.name, None)) for f in dataclass_fields(kernel)),
        id(basis_A),
        id(basis_B),
        np.asarray(rows).tobytes(),
    )


def _assert_hard_tensor_memo_is_faithful(contributions, nu, basis):
    """Rebuild one hard tensor with the memo bypassed and require bit-identity.

    The key above is complete over ``DYKernel``'s *fields*; this is what pins it against the
    source rather than against itself, and it is the only assertion here that would survive
    the tensor coming to depend on something that is not a field at all -- module state, a
    global, an environment variable.  Bypassing means calling the **unpatched**
    ``_assemble_bilinear_tensor`` captured at import, so this genuinely recomputes.

    ``np.array_equal``, not a tolerance: the memo returns the same computation or a different
    one, and there is no third possibility to leave headroom for.  One kernel, ~0.3 s.
    """
    from pixel.kernels.drell_yan.kernel import DYKernel

    kernel = contributions[0].kernel
    _, cached_tensor = _HARD_TENSOR_CACHE[_tensor_key(kernel, nu, basis, basis)]
    memoized = np.asarray(cached_tensor, dtype=float)
    fresh = np.asarray(
        _UNMEMOIZED_ASSEMBLE_BILINEAR_TENSOR(kernel, nu, basis, basis), dtype=float
    )
    assert np.array_equal(memoized, fresh), (
        "hard-tensor memo served a tensor the kernel does not produce -- the key is "
        "missing something the tensor depends on"
    )
    assert DYKernel._assemble_bilinear_tensor is not (
        _UNMEMOIZED_ASSEMBLE_BILINEAR_TENSOR
    ), "memo is not installed -- this check would be comparing the source with itself"


def _memoized_hard_tensor(monkeypatch):
    """Cache ``DYKernel._assemble_bilinear_tensor`` on ``(kernel values, rows, bases)``.

    **Why this level, and why it is here and not in ``src/``.**

    The expensive object in this file is the *hard* Drell--Yan tensor, and it is a pure
    function of ``(kernel, rows, basis_A, basis_B)`` -- no evolution operator, no weight.
    Measured 2026-08-14 on this file's own fixture (``DY_EVOLVE=True``, ``DY_NF=5``): 2224
    bilinear contributions collapse onto 680 ``(id(kernel), id(evolution_A),
    id(evolution_B))`` groups but only **30 distinct kernel objects**.  So the 680 group
    tensors are 30 hard tensors rebuilt 650 extra times, at ~0.59 s each.

    That is the whole cost of the test, and it is why this memo sits *below* the evolution
    composition rather than above it.  An earlier version of this helper cached
    ``BilinearContribution.assemble`` keyed on the ``(kernel, evolution_A, evolution_B)``
    triple -- one level too shallow, because that triple is 680-valued where the thing it
    is protecting is 30-valued.  Re-keying onto the kernel took the ``[True]`` case from
    **296 s to 10.0 s** (30x) with the folded result **bit-identical** (``np.array_equal``,
    not a tolerance).  The 296 s is measured against today's source, not read off
    ``test_durations.json``: that file records 313.36 s from when ``build_drell_yan``
    emitted 1552 contributions and 424 groups, against today's 2224 and 680.

    ``generate.dy_central`` already groups on the 680-key and assembles once per group, so
    production pays 680 hard-tensor builds where 30 would do -- but it pays them once, into
    a disk cache (``cache_path``), which this fixture leaves ``None``.  Putting this memo in
    ``src/`` would add a memory-holding structure to production for no production benefit,
    and ``CLAUDE.md`` names memory -- not runtime -- as this repo's binding constraint.

    **It removes no coverage, and at this level it adds some.**  The reference loop still
    sums all 2224 terms one at a time, which is the arithmetic under test.  What changed
    versus the ``assemble``-level memo is that the *evolution composition* is no longer
    cached: ``evolved_tensor``'s ``einsum("ria,rij,rjb->rab", ...)`` now runs on every one
    of the 2224 reference assembles and every one of the 680 in ``dy_central`` (2904 total),
    against each contribution's actual operators, where before it ran 680 times.  So does
    ``cached_arrays`` and ``_tensor_metadata``, since the hook is the leaf builder rather
    than the caching wrapper around it.

    Tensor *creation* -- the one thing this does skip -- is independently checked elsewhere,
    against oracles that are not common-mode with this one:
    ``test_dy_kernel.py::test_dy_kernel_multi_element_tensor_matches_fitpack_benchmark``
    (frozen fitpack benchmark, `C1`, bar 4.0e-4 plus a refinement-convergence assertion),
    ``test_dy_dataset.py::test_gluon_channel_matches_aem_eq93_up_to_the_documented_cqg2_shortfall``
    (Altarelli-Ellis-Martinelli eq. 93 by independent quadrature, asserted as an identity), and
    ``test_dis_dy_closure.py::test_drell_yan_bilinear_tensor_reproduces_the_continuum_cross_section``
    (against ``DYConvolution``'s continuum value, with a grid-refinement sweep).

    Verified 2026-08-14 that the memo does not blunt the test: with it in place, four
    mutations of ``dy_central``'s grouping/accumulation arithmetic all still fail --
    dropping ``evolution_B`` from the group key (39x off), dropping both operators (58x),
    dropping ``weight`` (3.6e4 x), and flipping the outer product's sign (2.0x) -- against
    an unmutated margin of 8.2e-14.
    """
    def cached(self, rows, basis_A, basis_B):
        key = _tensor_key(self, rows, basis_A, basis_B)
        # Collision detector, O(1) per call and *independent of* `_tensor_key`: remember
        # which kernel each entry was built for, and refuse to serve it to a different one.
        # `DYKernel` is a dataclass with no `repr=False` field and no addresses in its repr
        # (both checked), so its repr distinguishes any two kernels a correct key must
        # distinguish -- which makes this fire on exactly the failure the key can have, an
        # under-determining key, without recomputing a single tensor.  The recompute guard
        # `_assert_hard_tensor_memo_is_faithful` samples one kernel and so cannot see a
        # collision that served some *other* kernel the wrong tensor; this sees all of them.
        signature = repr(self)
        if key in _HARD_TENSOR_CACHE:
            built_for, tensor = _HARD_TENSOR_CACHE[key]
            assert built_for == signature, (
                "hard-tensor memo collision: two different kernels share one cache key, so "
                "one was served the other's tensor. `_tensor_key` is under-determining.\n"
                f"  built for: {built_for}\n  requested by: {signature}"
            )
            return tensor
        # The import-time original, not `DYKernel._assemble_bilinear_tensor` read here:
        # several tests in this file install the memo, and reading the attribute would let
        # one memo wrap another.
        tensor = _UNMEMOIZED_ASSEMBLE_BILINEAR_TENSOR(self, rows, basis_A, basis_B)
        _HARD_TENSOR_CACHE[key] = (signature, tensor)
        return tensor

    monkeypatch.setattr(_DYKernel, "_assemble_bilinear_tensor", cached)
    return _HARD_TENSOR_CACHE


@functools.lru_cache(maxsize=1)
def _basis():
    """The one basis every test in this file uses -- deliberately a shared singleton.

    Each test built its own identical ``ProductBasis`` until 2026-08-14.  Returning one
    object instead is what makes ``_HARD_TENSOR_CACHE`` reachable across tests: the hard
    tensor depends on the basis, so the cache key must pin it, and ``id(basis)`` is an
    exact key only when the object is shared.

    The alternative -- deriving a *value* key from the basis (node positions, element type,
    order) -- was rejected as the more dangerous of the two.  Every test here passes one
    basis to both tensor axes, so a value key that omitted some part of the basis's state
    would share a tensor between genuinely different bases and no assertion in this file
    would notice: the ``len(cache) == n_kernels`` guard varies the *kernel*, never the
    basis.  Object identity cannot be wrong in that way.

    Safe to share because nothing here mutates it: the bases are handed read-only to
    ``build_drell_yan``, ``assemble`` and ``bilinear_tensor``, and the two evolution stubs
    ignore their basis argument entirely.
    """
    grid = Grid(n_points=6, domain=(1e-3, 1.0), spacing="log")
    return ProductBasis(grid, element_type="piecewise", element_kwargs={"order": 2})


def _stub_evolution(monkeypatch, size):
    """Cheap stand-ins so these tests never assemble a Mellin contour."""
    import pixel.kernels.evolution.flavor_transition as transition

    eye = np.eye(size)

    def nonsinglet(_basis, *, Q2, Q20, channel, **kwargs):
        factor = (1.0 + 0.01 * (Q2 - Q20)) * (1.0 if channel == "plus" else 0.9)
        return factor * eye, np.zeros_like(eye)

    def singlet(_basis, *, Q2, Q20, source, target, **kwargs):
        factors = {
            ("quark_singlet", "quark_singlet"): 1.0 + 0.02 * (Q2 - Q20),
            ("quark_singlet", "gluon"): 0.03 * (Q2 - Q20),
            ("gluon", "quark_singlet"): 0.01 * (Q2 - Q20),
            ("gluon", "gluon"): 1.0 + 0.015 * (Q2 - Q20),
        }
        return SimpleNamespace(matrix=factors[(target, source)] * eye)

    monkeypatch.setattr(transition, "non_singlet_evolution_matrix", nonsinglet)
    monkeypatch.setattr(transition, "singlet_evolution_matrix", singlet)


def _stub_evolution_flavour_mixing(monkeypatch, size):
    """A stand-in whose operators are **not** all multiples of one matrix.

    ``_stub_evolution`` above returns ``factor * eye`` for every channel and every
    singlet block.  Every operator built from it is therefore a scalar multiple of the
    identity, and a scalar multiple of the identity commutes with everything -- so
    ``(E_A q_A)^T Y (E_B q_B)`` is exactly invariant under exchanging ``E_A`` and
    ``E_B``.  Measured: with that stub, swapping the two evolution operators of a
    contribution changes the folded prediction by ``0.000`` relative, bit for bit.  Any
    test built on it is structurally incapable of seeing which hadron axis an operator
    landed on, however tight its tolerance.

    Multiplying by one *fixed* non-diagonal matrix is not enough either, and that was
    measured too: with every block equal to ``factor * M`` for a single ``M``, the
    operators stay proportional to each other and the swap is again invariant to 0.000.
    This variant therefore multiplies each channel and each singlet block by a
    *different* polynomial in a nilpotent shift ``U`` (``I + t U``, with ``t`` chosen per
    block), so the two hadrons' operators are genuinely non-proportional.  Measured on
    the contributions the test selects, the swap then changes the fold by 282%.
    """
    import pixel.kernels.evolution.flavor_transition as transition

    eye = np.eye(size)
    shift = np.diag(np.ones(size - 1), 1)  # nilpotent: keeps the operators invertible

    def mix(t):
        return eye + t * shift

    def nonsinglet(_basis, *, Q2, Q20, channel, **kwargs):
        factor = (1.0 + 0.01 * (Q2 - Q20)) * (1.0 if channel == "plus" else 0.9)
        return factor * mix(0.4 if channel == "plus" else -0.25), np.zeros_like(eye)

    def singlet(_basis, *, Q2, Q20, source, target, **kwargs):
        factors = {
            ("quark_singlet", "quark_singlet"): 1.0 + 0.02 * (Q2 - Q20),
            ("quark_singlet", "gluon"): 0.03 * (Q2 - Q20),
            ("gluon", "quark_singlet"): 0.01 * (Q2 - Q20),
            ("gluon", "gluon"): 1.0 + 0.015 * (Q2 - Q20),
        }
        mixings = {
            ("quark_singlet", "quark_singlet"): 0.15,
            ("quark_singlet", "gluon"): -0.3,
            ("gluon", "quark_singlet"): 0.5,
            ("gluon", "gluon"): 0.05,
        }
        key = (target, source)
        return SimpleNamespace(matrix=factors[key] * mix(mixings[key]))

    monkeypatch.setattr(transition, "non_singlet_evolution_matrix", nonsinglet)
    monkeypatch.setattr(transition, "singlet_evolution_matrix", singlet)


def _record(reaction="pd"):
    return {
        "label": "dy_test",
        "reaction": reaction,
        "Q2": [200.0, 260.0, 200.0],
        "S": [1505.44] * 3,
        "Y": [0.1, 0.3, 0.5],
    }


def _fields(config, basis):
    return {name: SimpleNamespace(basis=basis) for name in config.ALL_FIELDS}


def _suite(name):
    import importlib

    config = importlib.import_module(f"{name}.config")
    datasets = importlib.import_module(f"{name}.datasets")
    generate = importlib.import_module(f"{name}.generate")
    return config, datasets, generate


@pytest.mark.parametrize("suite", SUITES)
def test_dy_evolve_constant_reaches_the_built_operator(suite, monkeypatch):
    """The flag must change the dataset, not just the documentation.

    This is the test the plan asks for: it fails if ``DY_EVOLVE`` and the code
    path disagree, which is exactly the state the constant was in (``evo =
    None`` was hardcoded regardless of the flag's value).

    Two checks on the evolved branch, of different strength. The ``{"b", "bb"}
    <= partons`` membership check reaches real, unstubbed logic --
    ``flavor_transition``'s bottom-threshold decision
    ``Q20 < AlphaS.mb_match2 <= Q2`` -- so it would catch that decision
    breaking (oracle ``F1`` on a real code path, not the evolution stub).

    The ``evolution_A is not None`` check below it now uses ``all(...)``,
    matching its unevolved mirror four lines down.  It read ``any(...)``
    until 2026-08-13, which was strictly weaker than the code's own
    guarantee: measured on this fixture (both suites), every one of the
    contributions ``build_drell_yan`` builds gets a real evolution
    operator on **both** sides whenever ``DY_EVOLVE=True`` -- ``fields_A``
    and ``evolution_A`` come from one ``PartonEvolutionProjection``, so they
    share a key set by construction.  Under ``any(...)`` a regression that
    dropped evolution from a strict subset of contributions (e.g. only the
    newly-crossed ``b``/``bb`` ones, or one source's shared block left unset
    by ``_nonsinglet_blocks``/``_singlet_blocks``) satisfied both this line
    and the membership check above; under ``all(...)`` a single unwired
    contribution fails it.  Non-emptiness is asserted alongside, so a fixture
    that quietly stopped building contributions could not make ``all(...)``
    vacuously true on an empty list.

    That guard used to be an absolute count (``== 1552``), which is the wrong
    instrument and broke on 2026-08-14 when the builder changed.  See the comment
    at the assertion: the count measures how finely an internal sum is split, not
    how much physics is present, so it moves under refactors that leave the summed
    tensor bit-identical.  The count relation that *is* meaningful here is
    ``evolved > unevolved``, and it is asserted below -- it comes from
    ``DY_NF=5`` against ``DY_NF=4``, i.e. b/bbar crossing the bottom threshold,
    which is the same physics the ``{"b", "bb"}`` membership check pins.  Note the
    two branches are deliberately at *different* ``DY_NF``, so equality would be
    wrong.
    """
    config, datasets, _ = _suite(suite)
    basis = _basis()
    _stub_evolution(monkeypatch, basis.n_elements)
    fields = _fields(config, basis)

    monkeypatch.setattr(config, "DY_EVOLVE", True)
    monkeypatch.setattr(config, "DY_NF", 5)
    evolved = datasets.build_drell_yan(_record(), None, fields, None)
    # Non-emptiness, not a pinned count.  ``all(...)`` below is vacuously true on
    # an empty list, so *something* must establish the list is populated -- but the
    # absolute count is the wrong guard for it.  It counts terms in an internal sum
    # decomposition, not physics: measured 2026-08-14, this fixture's 2224
    # contributions are 736 distinct
    # ``(field_A, field_B, channel, parton_A, parton_B)`` routes emitted 2x (440),
    # 4x (256) or 8x (40), differing only in ``weight`` -- an outer product of
    # per-leg factors that the model sums back together.  So the number moves
    # whenever the builder changes *how it splits the sum*, even when the summed
    # tensor is bit-identical, and it fires on refactors while staying silent on
    # the physics.  It was pinned at 1552 and broke on exactly such a change.
    assert evolved.bilinear_contributions
    assert all(
        c.evolution_A is not None and c.evolution_B is not None
        for c in evolved.bilinear_contributions
    )
    # Evolution crosses the bottom threshold, so the projected parton set is
    # five-flavour and b/bbar carry a real luminosity.
    assert {"b", "bb"} <= {c.kernel.parton_A for c in evolved.bilinear_contributions}

    monkeypatch.setattr(config, "DY_EVOLVE", False)
    monkeypatch.setattr(config, "DY_NF", 4)
    unevolved = datasets.build_drell_yan(_record(), None, fields, None)
    assert unevolved.bilinear_contributions
    assert all(
        c.evolution_A is None and c.evolution_B is None
        for c in unevolved.bilinear_contributions
    )
    assert "b" not in {c.kernel.parton_A for c in unevolved.bilinear_contributions}
    # The meaningful count relation, derived rather than typed: the evolved branch
    # runs at DY_NF=5 and the unevolved at DY_NF=4, so b/bbar cross the bottom
    # threshold and add luminosity the four-flavour build cannot have.  This holds
    # at any decomposition granularity, unlike the absolute count it replaces.
    assert len(evolved.bilinear_contributions) > len(unevolved.bilinear_contributions)


@pytest.mark.parametrize("suite", SUITES)
@pytest.mark.parametrize("evolve", [False, True])
def test_dy_central_folds_the_same_operator_the_dataset_carries(
    suite, evolve, monkeypatch
):
    """Generation must fold the assembled operator, evolved or not.

    The reference here is written differently on purpose -- one contribution at
    a time, no grouping -- so agreement is evidence about ``generate.dy_central``'s
    ``(id(kernel), id(evolution_A), id(evolution_B))`` grouping/accumulation
    arithmetic, not the same code compared with itself for that arithmetic.

    Oracle ``A3``: both sides still read ``c.field_A``/``c.field_B`` and
    ``c.assemble(...)`` off the *same* ``dataset.bilinear_contributions``, so a
    field mislabelling upstream or an axis-transposed tensor inside
    ``assemble()`` reproduces identically on both sides and would not be
    caught here -- see the module docstring and
    ``test_dy_channels.py::test_mirror_channel_swaps_A_and_B`` for that gap.
    What this *does* catch: a grouping-key bug in ``dy_central`` (an outer
    product summed with the wrong sign, a weight double-counted across
    contributions grouped onto one key, or evolved/unevolved rows merged into
    one group by mistake) -- exactly the class of bug the module's own
    docstring names as the historical risk this rewrite introduced.
    """
    config, datasets, generate = _suite(suite)
    basis = _basis()
    _stub_evolution(monkeypatch, basis.n_elements)
    # This fixture's 2224 contributions carry only 30 distinct hard tensors; see
    # _memoized_hard_tensor for why caching them here removes no coverage.
    tensor_cache = _memoized_hard_tensor(monkeypatch)
    monkeypatch.setattr(config, "DY_EVOLVE", evolve)
    monkeypatch.setattr(config, "DY_NF", 5 if evolve else 4)

    fields = _fields(config, basis)
    dataset = datasets.build_drell_yan(_record(), None, fields, None)
    rng = np.random.default_rng(11)
    truth = {name: rng.normal(size=basis.n_elements) for name in config.ALL_FIELDS}

    expected = np.zeros(int(dataset.n_data))
    for c in dataset.bilinear_contributions:
        tensor = np.asarray(c.assemble(dataset.nu, basis, basis), dtype=float)
        expected += np.einsum(
            "rij,i,j->r", tensor, truth[c.field_A], truth[c.field_B], optimize=True
        )

    got = generate.dy_central(dataset, truth, fields, _record()["reaction"])
    # float64 pin: both sides sum the identical per-contribution terms in a
    # different order (grouped vs one-at-a-time), so this bounds summation-order
    # noise, not a physical accuracy -- see the module docstring.
    assert np.allclose(got, expected, rtol=1e-12, atol=0.0)
    # Non-degeneracy only: rules out a fold that silently returns all zeros
    # (e.g. an empty group dict); does not by itself imply correctness.
    assert np.any(np.abs(got) > 0.0)
    # The memo must key exactly one entry per distinct kernel object -- an identity, not a
    # bound, and the strongest guard available here.  Every contribution folds the same
    # `dataset.nu` through the same `basis`, so `(kernel values, rows, bases)` is one-to-one
    # with the kernel object; measured 30 == 30 evolved, 24 == 24 unevolved.
    #
    # Counted over *this fixture's* keys, not `len(tensor_cache)`: the cache is module-scoped
    # and carries whatever earlier tests put in it, so the module-wide total is not this
    # test's business and would make the assertion order-dependent.
    #
    # Too *coarse* a key is the dangerous direction, and this catches it: a memo that dropped
    # the kernel from its key would hand one kernel's hard tensor to another, and the
    # reference loop would silently become a comparison of one tensor with itself.  Too fine
    # merely costs time.  The `> 1` and `< len(contributions)` halves also survive here, and
    # are what make the reference sum a genuine multi-tensor fold.
    contributions = dataset.bilinear_contributions
    distinct_kernels = {id(c.kernel) for c in contributions}
    fixture_keys = {
        _tensor_key(c.kernel, dataset.nu, basis, basis) for c in contributions
    }
    assert len(fixture_keys) == len(distinct_kernels), (
        "hard-tensor memo is not one entry per kernel -- the key is wrong"
    )
    assert fixture_keys <= set(tensor_cache), "the memo did not serve this fixture"
    assert len(distinct_kernels) > 1, "fixture degenerated to a single hard tensor"
    assert len(distinct_kernels) < len(contributions), (
        "hard-tensor memo collapsed nothing -- the memo is not keying as intended"
    )
    # ...and the served tensors are the ones the kernels actually produce.  The three counts
    # above are all internal to the memo; this is what pins it against unmemoized source.
    _assert_hard_tensor_memo_is_faithful(contributions, dataset.nu, basis)


@pytest.mark.parametrize("suite", SUITES)
def test_unevolved_dy_central_reproduces_the_parton_precombined_fold(suite, monkeypatch):
    """Turning the rewrite on must not move the historical unevolved answer.

    The previous implementation pre-combined the truth into *parton* curves and
    folded the bare hard tensor.  It only summed differently, so the two agree to
    rounding rather than bitwise -- assert that, not bit-identity.

    Oracle ``A3``: the reference here precombines with ``config.dy_field_maps``
    (the same function ``build_drell_yan`` itself calls to build ``fields_A``/
    ``fields_B`` in the unevolved branch, so this is not an independent source
    of the coefficients) and folds ``c.kernel.bilinear_tensor(...)`` directly,
    bypassing ``assemble()``. What is genuinely independent is the fold order:
    this checks that decomposing one parton pair's truth into several
    per-*field* contributions (as ``dy_central`` consumes them, each with its
    own ``coeffA * coeffB`` weight) is algebraically the same as combining the
    fields into one parton curve first and folding once -- distributivity of
    the bilinear form over the field decomposition, the exact property the
    field-decomposed rewrite depends on. Would catch a coefficient applied
    twice, a missing cross term between two fields sharing one parton, or a
    sign flip in ``coeffA * coeffB``. Would not catch a wrong
    ``dy_field_maps`` coefficient or a bug in ``bilinear_tensor``'s own hard
    kernel, since both sides call them the same way.
    """
    config, datasets, generate = _suite(suite)
    basis = _basis()
    # Same 24 hard tensors as the `[False]` case of the test above -- identical
    # `(pd, DY_EVOLVE=False, DY_NF=4)` fixture -- so whichever runs second gets them free
    # from the module-scoped cache.  Measured 15.1s -> 0.4s.
    _memoized_hard_tensor(monkeypatch)
    monkeypatch.setattr(config, "DY_EVOLVE", False)
    monkeypatch.setattr(config, "DY_NF", 4)
    fields = _fields(config, basis)
    record = _record()
    dataset = datasets.build_drell_yan(record, None, fields, None)
    rng = np.random.default_rng(3)
    truth = {name: rng.normal(size=basis.n_elements) for name in config.ALL_FIELDS}

    map_A, map_B = config.dy_field_maps(record["reaction"])

    def parton_curve(combination):
        out = np.zeros(basis.n_elements)
        for field, coefficient in combination.items():
            out = out + coefficient * truth[field]
        return out

    seen: set[int] = set()
    expected = np.zeros(int(dataset.n_data))
    for c in dataset.bilinear_contributions:
        if id(c.kernel) in seen:
            continue
        seen.add(id(c.kernel))
        tensor = np.asarray(
            c.kernel.bilinear_tensor(dataset.nu, basis, basis), dtype=float
        )
        expected += np.einsum(
            "rij,i,j->r",
            tensor,
            parton_curve(map_A[c.kernel.parton_A]),
            parton_curve(map_B[c.kernel.parton_B]),
            optimize=True,
        )

    got = generate.dy_central(dataset, truth, fields, record["reaction"])
    # float64 pin on the regrouping/fold-order difference, not a physics bound
    # -- see this test's docstring for what independence this does and does
    # not establish.
    assert np.allclose(got, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("suite", SUITES)
def test_evolved_dy_contributions_collapse_onto_shared_operator_groups(
    suite, monkeypatch
):
    """Block sparsity must survive evolution.

    ``model`` groups on ``(id(kernel), id(evolution_A), id(evolution_B))``.  A
    distinct operator array per field pair would be numerically identical and
    ruinously more expensive, so the collapse is asserted rather than assumed.

    Oracle ``F1`` (structural: a magnitude-aware count bound plus per-operator
    shape). Measured on this fixture, identically in both suites: 1552
    contributions collapse onto 424 groups (27.3%). The bar was
    ``len(groups) < len(contributions)`` until 2026-08-13 -- one fewer group
    than contributions, which 1551 groups (effectively no sharing at all)
    would have satisfied, defeating the performance property the test claims
    to guard. It is now ``<= len(contributions) // 2`` (776 here), which the
    achieved 424 clears by 45%: tight enough that losing most of the sharing
    fails, loose enough to survive a flavour or channel being added to the
    fixture. Would catch a regression to one array per contribution or a
    partial loss of more than half the sharing; would not resolve a small
    erosion, and says nothing about whether the shared operators are
    numerically right.
    """
    config, datasets, _ = _suite(suite)
    basis = _basis()
    _stub_evolution(monkeypatch, basis.n_elements)
    monkeypatch.setattr(config, "DY_EVOLVE", True)
    monkeypatch.setattr(config, "DY_NF", 5)

    dataset = datasets.build_drell_yan(
        _record(), None, _fields(config, basis), None
    )
    contributions = dataset.bilinear_contributions
    groups = {
        (id(c.kernel), id(c.evolution_A), id(c.evolution_B)) for c in contributions
    }
    # Magnitude-aware bound.  Measured for this fixture, both suites: 424
    # groups from 1552 contributions (27.3%), against a bar of 776.
    assert len(groups) <= len(contributions) // 2
    # Every row-wise operator carries one matrix per kinematic row.
    for c in contributions:
        for operator in (c.evolution_A, c.evolution_B):
            array = np.asarray(operator, dtype=float)
            assert array.shape == (
                int(dataset.n_data),
                basis.n_elements,
                basis.n_elements,
            )


@pytest.mark.parametrize("suite", SUITES)
def test_assemble_composes_each_hadron_operator_onto_its_own_axis(suite, monkeypatch):
    """Each contribution's evolution operator acts on *its own* hadron axis: evolving
    the two truth curves and folding the bare tensor equals folding the assembled one.

    This is the check the rest of the file cannot make.  Every other numeric assertion
    here reads ``c.field_A``/``c.field_B`` and ``c.assemble(...)`` off the same
    ``dataset.bilinear_contributions`` that ``dy_central`` reads, so an axis-transposed
    ``evolved_tensor`` -- ``E_B`` folded onto the A axis and vice versa -- reproduces
    identically on both sides of every comparison and passes (module docstring, weakness
    ``-02``).

    The oracle is the *semantic* contract, not a re-typing of the source's einsum:
    ``BilinearContribution.assemble`` and ``DYKernel.evolved_tensor`` document their
    operators as mapping input coefficients to the hard tensor's basis
    (``q_phys = E @ q_input``, ``src/pixel/core/model.py:715`` and
    ``src/pixel/kernels/drell_yan/kernel.py:575-577``).  So folding the *bare* tensor
    with the separately evolved curves ``E_A[r] @ q_A`` and ``E_B[r] @ q_B`` -- ordinary
    matrix-vector products, computed row by row here -- must reproduce folding the
    assembled tensor with the un-evolved curves, ``weight`` included.  Oracle ``A3``:
    both sides use the same ``bilinear_tensor`` and the same operators, so this does not
    check the hard kernel or the evolution operators themselves; what it checks is which
    axis each operator is composed onto, and that the composition is the documented one.

    **Two degeneracies had to be removed for this to be able to fail**, and both are
    measured rather than assumed:

    * the reaction is ``pd``, so hadron B's projection is
      ``isoscalar_projection(proton)`` and the two hadrons carry different operators;
    * the shared ``_stub_evolution`` returns scalar multiples of the identity, under
      which the swap is *exactly* invariant (measured 0.000 relative).  This test uses
      ``_stub_evolution_flavour_mixing`` instead -- see its docstring.

    Measured with both removed, identically in both suites, on the two selected
    contributions (parton pairs ``u/ub`` and ``d/db``): the identity holds at
    ``max|a/b - 1| = 6.7e-16`` while the swapped fold sits ``2.821`` away -- 282%, about
    15 orders above the agreement.  The ``rtol=1e-11`` bar is float64 headroom over the
    achieved 6.7e-16; the ``> 0.1`` non-degeneracy bar is deliberately an order below the
    achieved 2.821, because that ratio depends on the random curves (measured 2.821,
    0.878 and 0.423 at seeds 0, 3 and 17) while the agreement does not (1e-16 at every
    seed).  Restricted to two contributions, one per parton pair, to keep the hard-tensor
    assembly (about 1.1 s each) affordable.
    """
    config, datasets, _ = _suite(suite)
    basis = _basis()
    _stub_evolution_flavour_mixing(monkeypatch, basis.n_elements)
    # The two hard tensors this selects are among the 30 the `[True]` case above builds, so
    # the module-scoped cache serves them.  Note the memo is *below* the evolution stub:
    # this test uses `_stub_evolution_flavour_mixing` where the others use `_stub_evolution`,
    # and the hard tensor does not depend on either -- only the composition does, and that
    # is never cached.  Measured 2.4s -> 0.1s.
    _memoized_hard_tensor(monkeypatch)
    monkeypatch.setattr(config, "DY_EVOLVE", True)
    monkeypatch.setattr(config, "DY_NF", 5)
    dataset = datasets.build_drell_yan(
        _record("pd"), None, _fields(config, basis), None
    )

    # One contribution per distinct parton pair, so the two cases below carry
    # genuinely different operators rather than two rows of the same pair.
    by_partons = {}
    for candidate in dataset.bilinear_contributions:
        key = (candidate.kernel.parton_A, candidate.kernel.parton_B)
        if key[0] == key[1] or key in by_partons:
            continue
        if np.allclose(
            np.asarray(candidate.evolution_A, dtype=float),
            np.asarray(candidate.evolution_B, dtype=float),
        ):
            continue  # proportional operators cannot resolve the two axes
        by_partons[key] = candidate
    selected = list(by_partons.values())[:2]
    # Non-degeneracy of the selection itself: a fixture with one operator for
    # both hadrons could not distinguish the axes no matter what is asserted.
    assert len(selected) == 2
    assert selected[0].kernel.parton_A != selected[1].kernel.parton_A or (
        selected[0].kernel.parton_B != selected[1].kernel.parton_B
    )

    rng = np.random.default_rng(0)
    curve_A = rng.normal(size=basis.n_elements)
    curve_B = rng.normal(size=basis.n_elements)
    for contribution in selected:
        operator_A = np.asarray(contribution.evolution_A, dtype=float)
        operator_B = np.asarray(contribution.evolution_B, dtype=float)
        assembled = np.asarray(
            contribution.assemble(dataset.nu, basis, basis), dtype=float
        )
        bare = np.asarray(
            contribution.kernel.bilinear_tensor(dataset.nu, basis, basis), dtype=float
        )
        # q_phys = E @ q_input, one matrix per kinematic row.
        evolved_A = np.einsum("rik,k->ri", operator_A, curve_A)
        evolved_B = np.einsum("rjk,k->rj", operator_B, curve_B)

        got = np.einsum("rij,i,j->r", assembled, curve_A, curve_B, optimize=True)
        expected = contribution.weight * np.einsum(
            "rij,ri,rj->r", bare, evolved_A, evolved_B, optimize=True
        )
        # float64 transcription pin (measured 4.4e-16 to 1.8e-15), not a
        # physics bar: both sides contract the same numbers in a different
        # order.
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=0.0)

        # The swapped composition is what an axis-transposed assemble() would
        # produce; measured 2.8x-3.0x away, so the agreement above is a real
        # constraint on the axis assignment rather than an identity.
        swapped = contribution.weight * np.einsum(
            "rij,ri,rj->r",
            bare,
            np.einsum("rik,k->ri", operator_B, curve_A),
            np.einsum("rjk,k->rj", operator_A, curve_B),
            optimize=True,
        )
        assert np.max(np.abs(swapped / expected - 1.0)) > 0.1


def _parton_evolution_differences(a, b):
    """Report every structural difference between two projection ``evolution`` maps.

    ``PartonEvolutionProjection.evolution`` is ``{parton: {source: ((weight, operator),
    ...)}}``, so ``==`` is useless (the operators are arrays) and ``assert_allclose`` cannot
    walk it.  Returns a list of human-readable differences -- empty means identical -- and
    compares weights and operators **exactly**: both sides here come from the same
    deterministic stub, so any tolerance would only hide a discrepancy.
    """
    out = []
    if set(a) != set(b):
        out.append(f"parton keys differ: {sorted(set(a) ^ set(b))}")
        return out
    for parton in sorted(a):
        if set(a[parton]) != set(b[parton]):
            out.append(f"{parton}: source keys differ: {sorted(set(a[parton]) ^ set(b[parton]))}")
            continue
        for source in sorted(a[parton]):
            terms_a, terms_b = a[parton][source], b[parton][source]
            if len(terms_a) != len(terms_b):
                out.append(f"{parton}/{source}: {len(terms_a)} terms vs {len(terms_b)}")
                continue
            for (wa, oa), (wb, ob) in zip(terms_a, terms_b):
                if wa != wb:
                    out.append(f"{parton}/{source}: weight {wa} vs {wb}")
                elif not np.array_equal(np.asarray(oa), np.asarray(ob)):
                    out.append(f"{parton}/{source}: operator differs")
    return out


@pytest.mark.parametrize("suite", SUITES)
def test_dy_side_a_is_the_beam_and_side_b_carries_the_reaction_transform(
    suite, monkeypatch
):
    """Side A is the beam proton; the ``pd``/``ppbar`` transform lands on side B alone.

    This is the ``field_A``/``field_B`` *labelling* half of weakness ``-02``, which
    ``test_assemble_composes_each_hadron_operator_onto_its_own_axis`` deliberately does
    not reach: that test checks which axis an operator is composed onto **given** the
    labels, while this one checks the labels themselves, one level upstream in
    ``dy_evolution_maps`` (``datasets.py:242-292``).  Its return is
    ``(proton.fields, target.fields, proton.evolution, target.evolution)`` -- a four-tuple
    of near-identical structures, and exchanging the two halves is a one-line edit that
    changes no shape and raises nothing.

    The oracle is the function's own ``pp`` answer (``A3``).  For ``pp`` the target *is*
    the beam, so both sides are the untransformed proton projection; the ``pd`` and
    ``ppbar`` A-sides must reproduce it **bit for bit**, and their B-sides must reproduce
    ``isoscalar_projection``/``conjugate_projection`` applied to it.  No projection is
    re-derived here and no kwarg list is retyped from ``datasets.py``: the reference is
    the same function under the one reaction whose answer is unambiguous.

    **Measured degeneracies, because two of the three obvious ways to write this test are
    vacuous.**  (a) The ``fields`` maps -- ``fields_A``/``fields_B`` themselves -- are
    *identical* across all three reactions and across partons: ``fields`` records only
    which source fields feed a parton with coefficient 1.0, and the isoscalar average and
    the charge conjugation both preserve that set, so ``fields_A == fields_B ==
    isoscalar(...).fields`` holds and any assertion on them is blind. All the orientation
    lives in ``evolution``, whose weights and term counts differ.  (b) Charge conjugation
    is an *involution*, so relations like ``fields_B["u"] == fields_A["ub"]`` are
    symmetric under the swap and pass either way -- the reference has to be an
    externally-anchored projection, which is what the ``pp`` leg supplies.  (c) ``pp``
    itself can never detect the swap and is asserted symmetric rather than skipped, so
    the reason it is exempt is on the record.

    Measured with ``_stub_evolution``: under the swap the ``pd`` A-side differs from the
    ``pp`` reference at ``u/t3`` (1 term vs 2, the isoscalar merge) and the ``ppbar``
    A-side at ``u/v3`` (weight ``+0.25`` vs ``-0.25``, the conjugation sign flip), so the
    scalar-multiple-of-identity stub is sufficient here -- unlike in
    ``test_assemble_composes_each_hadron_operator_onto_its_own_axis``, where the fold
    *value* is what is compared and that stub makes a swap exactly invariant.
    """
    from pixel.kernels.evolution.flavor_transition import (
        PartonEvolutionProjection,
        conjugate_projection,
        isoscalar_projection,
    )

    config, datasets, _ = _suite(suite)
    basis = _basis()
    _stub_evolution(monkeypatch, basis.n_elements)
    monkeypatch.setattr(config, "DY_EVOLVE", True)
    fields = _fields(config, basis)

    # pp: the target is the beam, so both sides are the bare proton projection -- and the
    # swap this test exists to catch is invisible here, which is why the other two matter.
    pp_fields_A, pp_fields_B, pp_evo_A, pp_evo_B = datasets.dy_evolution_maps(
        _record("pp"), fields
    )
    assert pp_fields_A == pp_fields_B
    assert _parton_evolution_differences(pp_evo_A, pp_evo_B) == []

    beam = PartonEvolutionProjection(
        pp_fields_A,
        pp_evo_A,
        np.asarray(_record("pp")["Q2"], dtype=float),
        config.Q0_2,
        config.DY_NF,
        0,  # shared_operator_count: unused by both transforms and by the comparison
    )

    for reaction, transform in (
        ("pd", isoscalar_projection),
        ("ppbar", conjugate_projection),
    ):
        fields_A, fields_B, evo_A, evo_B = datasets.dy_evolution_maps(
            _record(reaction), fields
        )
        expected_B = transform(beam)

        # A is the beam, untouched by the reaction.
        assert _parton_evolution_differences(evo_A, pp_evo_A) == []
        # B is the beam with this reaction's transform applied -- and only B.
        assert _parton_evolution_differences(evo_B, expected_B.evolution) == []
        # Non-degeneracy, asserted rather than assumed: without this the two checks above
        # would both hold for a function that returned the beam projection twice.
        assert _parton_evolution_differences(evo_B, pp_evo_B) != []
        assert _parton_evolution_differences(evo_A, expected_B.evolution) != []
        # The fields maps are degenerate (see the docstring); recorded, not relied on.
        assert fields_A == fields_B == pp_fields_A == expected_B.fields


@pytest.mark.parametrize("suite", SUITES)
def test_dy_projection_disagreeing_with_the_configured_nf_is_rejected(
    suite, monkeypatch
):
    """A five-flavour projection into a four-flavour luminosity must not pass.

    Oracle ``F1``: exercises ``dy_evolution_maps``'s
    ``if int(proton.nf) != int(cfg.DY_NF): raise ValueError(...)`` guard
    directly (``datasets.py:278-282``) with a real evolved projection (the
    fixture's ``Q2`` values cross the bottom threshold, so ``proton.nf`` is
    genuinely 5 here) and a deliberately-wrong ``DY_NF=4``. The ``match``
    string pins the specific mismatch message, not just "raises somehow".
    Structural contract, not a numeric claim: catches the guard being
    weakened, removed, or short-circuited; says nothing about whether the
    threshold-crossing decision it is built on is itself correct (see test
    1's docstring for what does check that).
    """
    config, datasets, _ = _suite(suite)
    basis = _basis()
    _stub_evolution(monkeypatch, basis.n_elements)
    monkeypatch.setattr(config, "DY_EVOLVE", True)
    monkeypatch.setattr(config, "DY_NF", 4)
    with pytest.raises(ValueError, match="projects to nf=5"):
        datasets.build_drell_yan(_record(), None, _fields(config, basis), None)
