"""Moderate/large-x scaling for closure reproduction panels.

Exercises ``reproduction_ylim`` in
``closure_{JAM,NNPDF}_truth{,_small}/run_closure.py``, a pure-numpy helper that
picks the ``(ymin, ymax)`` for a reproduction-panel y-axis from the truth and
posterior one-sigma bands; ``plot_reproduction`` calls it as
``ax.set_ylim(*reproduction_ylim(...))``.  Nothing here builds a Matplotlib
figure or inspects rendered output -- both tests call the helper directly and
check the numbers it returns, never ``plot_reproduction`` itself or what
actually lands on an axis.

The four ``reproduction_ylim`` bodies -- closure_JAM_truth/run_closure.py,
closure_JAM_truth_small/run_closure.py, closure_NNPDF_truth/run_closure.py,
closure_NNPDF_truth_small/run_closure.py -- are the same implementation, so
running the behavioural tests against more than one of them is a drift guard
between copies, not four independent implementations.  **That is now asserted,
not diffed by hand**:
``test_the_duplicated_run_closure_helpers_are_one_implementation`` compares the
parsed executable bodies (2026-08-14), and
``test_the_four_plot_reproduction_copies_differ_only_by_their_suite_name`` pins the
one callable that is one implementation *plus a literal*.  The duplication is much
wider than these two functions: of the 12 top-level callables all four
``run_closure.py`` copies share, **9 are one implementation copied four times**
(``_hatched_band``, ``_prepare_matplotlib``, ``_pull_chi2_per_point``,
``build_kernels_only``, ``generated_qs``, ``hybrid_xscale``, ``plot_comparison``,
``reproduction_ylim``, ``save_figure_both``), and only ``run_one`` and ``main``
genuinely differ between the full and ``_small`` suites -- the full pair also saves
posterior moments, draws a kinematic-coverage plot and prints a nuisance-coverage
summary.  De-duplicating the copies outright was considered and not done: the shared
home would have to be ``src/pixel`` or a new top-level package, the four suites are
deliberately self-contained data-generation trees with no cross-imports, and the
change is not one to make while a full-suite run is using them.  The executable
identity check is the part that makes the duplication safe to leave -- a fix landing
in one copy and not the other three now fails a test.  That
parametrization does cover all four packages, unlike
``test_closure_truth_representable.py``, whose ``package`` list excludes the
two ``_small`` packages and, per that file's own audit (S0-05), missed a real
methodology bug in ``closure_*_small/generate.py``'s ``dy_central`` as a
direct result.

**Two tuples, and they are deliberately different lengths.**  ``ALL_RUNNERS`` is all
four packages and is what the two executable-identity tests iterate; ``RUNNERS`` is the
NNPDF pair, and is what every behavioural ``@parametrize("runner", ...)`` below uses.
The JAM legs came out of the behavioural sweep on the owner's instruction 2026-08-14:
the identity tests establish that all four ``reproduction_ylim`` bodies are one
implementation, so running the same numeric fixture through four copies of it produced
the same number four times.  ``ALL_RUNNERS`` must keep all four -- see its own comment.

There is no independent physics oracle here; the "truth"/"posterior" values
are literals invented for this file, and the expected numbers are
hand-derived from that arithmetic (closed-form, stated in each test's
docstring), not pinned against any external reference.  This is a
UI/scaling contract, not a physics test; the reproduction panels themselves
are produced by the closure suites under ``closure_*/``.  One test does drive
``plot_reproduction`` itself -- the real ``ax.set_ylim(*reproduction_ylim(
...))`` call site -- and reads the limits back off the rendered Axes; the
rest call the helper directly.

Every ``reproduction_ylim`` branch is now reached by some test here, and each
branch's test is written so that deleting the branch changes the number it
asserts (checked by mutation, 2026-08-13):

* the ``x >= focus_x`` cutoff and both band edges (tests 1 and 3),
* the ``min(0.0, ...)``/``max(0.0, ...)`` zero-inclusion clamp (test 4),
* the ``span > 0.0`` degenerate-span fallback (test 6),
* the ``finite.size == 0`` empty-focus fallback (test 2),
* the ``np.isfinite`` filtering (test 7),
* the ``focus_x``/``pad_fraction`` keyword-only parameters (test 5).

One asymmetry is worth stating because it bounds what test 7 can prove: the
per-point ``np.isfinite(center) & np.isfinite(spread)`` terms in ``mask`` are
*provably redundant* with the later ``finite = finite[np.isfinite(finite)]``
filter.  Any non-finite ``center`` or ``spread`` makes both ``center-spread``
and ``center+spread`` non-finite, so the concatenated filter removes it
anyway; measured this pass, a mutant that drops the per-point terms returns
byte-identical limits on every fixture in this file.  The converse is not
true -- finite inputs whose sum overflows reach the concatenated filter and
nothing else -- so test 7 pins that filter and says so.
"""

import numpy as np
import pytest

from closure_JAM_truth import run_closure as jam_full
from closure_JAM_truth_small import run_closure as jam_small
from closure_NNPDF_truth import run_closure as nnpdf_full
from closure_NNPDF_truth_small import run_closure as nnpdf_small


#: All four ``run_closure.py`` copies.
#:
#: **Consumed only by the two cross-package identity tests** --
#: ``test_the_duplicated_run_closure_helpers_are_one_implementation`` (which asserts
#: ``len(set(bodies.values())) == 1`` over these) and
#: ``test_the_four_plot_reproduction_copies_differ_only_by_their_suite_name`` (which
#: asserts ``len(set(raw.values())) > 1``).  **Both are about the four copies, so this
#: tuple must keep all four**; trimming it would make the first vacuous and turn the
#: second RED, since two NNPDF ``plot_reproduction`` bodies do not differ by a suite
#: name.  It looks like a redundant duplicate of ``RUNNERS`` below.  It is not.
ALL_RUNNERS = (jam_full, jam_small, nnpdf_full, nnpdf_small)

#: The copies every behavioural ``@parametrize("runner", RUNNERS)`` below runs against:
#: **the NNPDF pair only**, on the owner's instruction 2026-08-14.  This was
#: ``ALL_RUNNERS``.  ``reproduction_ylim`` is a pure-numpy helper with no notion of a
#: truth PDF set, and the identity test above proves all four copies of it are one
#: implementation, so the two JAM legs pushed the same fixture through the same code for
#: the same answer.  Keeping the ``_small`` leg alongside the full one is what still
#: covers every branch in more than one copy, including the empty-focus fallback.
RUNNERS = (nnpdf_full, nnpdf_small)


def _executable_body(func):
    """``ast.dump`` of ``func`` with its leading docstring dropped.

    Source text would make the four copies look different for reasons that are
    not code -- each ``plot_reproduction`` names its own suite in a docstring and
    a legend label -- and a plain ``diff`` cannot separate those from a real
    divergence.  Comparing the parsed body does.
    """
    import ast
    import inspect
    import textwrap

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


#: The ``run_closure.py`` callables that are ONE implementation copied four times.
#: Measured 2026-08-14 over the 12 top-level callables all four packages share:
#: these 9 have byte-identical parsed bodies; ``plot_reproduction`` has 2 raw
#: bodies collapsing to 1 once the suite name is substituted; and ``run_one``
#: (3 raw / 2 agnostic) and ``main`` (4 raw / 2 agnostic) genuinely differ between
#: the full and ``_small`` suites -- the full pair also saves posterior moments,
#: draws a kinematic-coverage plot, and prints a nuisance-coverage summary.
IDENTICAL_RUN_CLOSURE_CALLABLES = (
    "_hatched_band",
    "_prepare_matplotlib",
    "_pull_chi2_per_point",
    "build_kernels_only",
    "generated_qs",
    "hybrid_xscale",
    "plot_comparison",
    "reproduction_ylim",
    "save_figure_both",
)


@pytest.mark.parametrize("name", IDENTICAL_RUN_CLOSURE_CALLABLES)
def test_the_duplicated_run_closure_helpers_are_one_implementation(name):
    """Each of these is the same code in all four packages -- asserted, not diffed.

    The module docstring's "byte-identical (``diff``, confirmed this audit)"
    claim is what licenses every ``RUNNERS`` parametrization below to be read as
    a drift guard rather than four independent implementations.  Until now that
    claim was prose: a hand-run ``diff`` at audit time, with nothing to re-check
    it.  Four near-identical copies is the shape where a fix lands in one and
    stays broken in three -- measured twice elsewhere in these packages this same
    session -- so the claim is now executable.

    **The duplication is much wider than ``reproduction_ylim``.**  Measured over
    the 12 top-level callables all four ``run_closure.py`` copies share: **9 are
    one implementation copied four times** (this parametrization),
    ``plot_reproduction`` is one implementation plus a suite name, and only
    ``run_one`` and ``main`` genuinely differ -- the full pair additionally saves
    posterior moments, draws a kinematic-coverage plot and prints a
    nuisance-coverage summary.

    Compared as the parsed body with the leading docstring dropped, so a
    per-suite docstring cannot make identical code look different -- and, more
    importantly, so a shared *prose* edit cannot make divergent code look the
    same.

    Non-degeneracy: ``_executable_body`` must distinguish two genuinely different
    functions, or this would pass on a normalization that flattens everything.
    """
    bodies = {
        runner.__name__: _executable_body(getattr(runner, name))
        for runner in ALL_RUNNERS
    }
    assert len(set(bodies.values())) == 1, (
        f"{name} copies have drifted across " + ", ".join(sorted(bodies))
    )
    # Control: the comparison is not so coarse that any two functions match.
    assert _executable_body(jam_full.hybrid_xscale) != _executable_body(
        jam_full.reproduction_ylim
    )


def test_the_four_plot_reproduction_copies_differ_only_by_their_suite_name():
    """``plot_reproduction`` is one implementation plus a literal -- both halves pinned.

    Each copy names its own suite in the legend label (``"x * JAM truth ..."`` vs
    ``"x * NNPDF truth ..."``).  That string is asserted to be the *only*
    divergence: substituting it out must collapse the four bodies to one, and
    *not* substituting it must leave more than one -- so the divergence is
    measured rather than assumed, and a second one fails here.

    De-duplicating these copies outright was considered and not done: the shared
    home would have to be ``src/pixel`` or a new top-level package, and the four
    suites are deliberately self-contained data-generation trees with no
    cross-imports.  This check is what makes leaving the duplication safe.
    """
    def suite_agnostic(runner):
        return (
            _executable_body(runner.plot_reproduction)
            .replace("NNPDF", "SUITE")
            .replace("JAM", "SUITE")
        )

    plot_bodies = {runner.__name__: suite_agnostic(runner) for runner in ALL_RUNNERS}
    assert len(set(plot_bodies.values())) == 1, (
        "plot_reproduction copies differ by more than their suite name: "
        + ", ".join(sorted(plot_bodies))
    )

    # The suite name really is a difference, not an assumption: without the
    # substitution the JAM and NNPDF copies must NOT match.  This is why the loop
    # above iterates ALL_RUNNERS and not the NNPDF-only RUNNERS -- with the JAM
    # copies dropped there would be no suite-name divergence left to find and this
    # assertion would fail, not pass vacuously.
    raw = {r.__name__: _executable_body(r.plot_reproduction) for r in ALL_RUNNERS}
    assert len(set(raw.values())) > 1

    # Control: the substitution is not so aggressive that any two functions
    # compare equal after it.
    assert suite_agnostic(jam_full) != _executable_body(jam_full.hybrid_xscale)


@pytest.mark.parametrize("runner", RUNNERS)
def test_reproduction_ylim_contains_both_bands_above_point_two(runner):
    """``reproduction_ylim`` keeps the x >= 0.2 one-sigma bands on-panel with headroom.

    Run once per closure package (``jam_full``, ``jam_small``, ``nnpdf_full``,
    ``nnpdf_small``); the four ``reproduction_ylim`` bodies are byte-identical
    (confirmed by ``diff`` this audit), so this is a drift guard across the four
    copies, not four independent implementations -- it would catch any one of them
    being edited in isolation, not a shared bug.

    Oracle: hand-derived from this fixture's own arithmetic (``A1``) -- with the
    default ``focus_x=0.2, pad_fraction=0.12`` the true return is exactly
    ``(-0.192, 1.792)`` (checked by direct computation this audit).  The bounds
    asserted below are a loose superset of that value, not the value itself.

    Catches: dropping the *upper* (``center + spread``) contribution to either
    band, or relaxing the ``x >= focus_x`` cutoff to ``x > focus_x`` -- both drop
    ``ymax`` well under 1.6 and fail the second assert (checked by mutation this
    audit).  Does NOT catch dropping the *lower* (``center - spread``)
    contribution: on this fixture the minimum is pinned by an exact zero at
    ``x=1.0`` (``truth=posterior=0``, ``std=0``) regardless of whether the lower
    band is computed at all, so that mutant returns the identical
    ``(-0.192, 1.792)`` and passes (checked by mutation this audit) -- the name
    promises "both bands" but only the upper one is actually load-bearing here.
    ``test_reproduction_ylim_floor_comes_from_the_lower_band`` below carries a
    fixture built specifically so that mutant *does* fail, and
    ``test_reproduction_ylim_clamps_a_one_signed_band_to_include_zero`` covers
    the ``min(0.0, ...)``/``max(0.0, ...)`` clamp, which is likewise inert on
    this fixture (the same exact-zero point already sits at the true min/max).

    The four one-sided bounds below are kept because each names a *distinct*
    failure direction in prose, but they are a loose superset of the true
    return, so the exact value is now asserted alongside them at
    ``rtol=1e-12, atol=0.0``.  Measured this pass, the hand-derived
    ``(-0.192, 1.792)`` reproduces bit-for-bit (``abs(got/expected - 1) == 0.0``
    in both components), so that bar is over decimal-literal round-tripping,
    not over any modelling slack.
    """
    x = np.array([0.01, 0.1, 0.2, 0.5, 1.0])
    truth = np.array([100.0, 20.0, 1.0, 0.5, 0.0])
    truth_std = np.array([50.0, 10.0, 0.1, 0.1, 0.0])
    posterior = np.array([-200.0, -40.0, 1.2, 0.4, 0.0])
    posterior_std = np.array([100.0, 30.0, 0.4, 0.2, 0.0])

    ymin, ymax = runner.reproduction_ylim(
        x, truth, truth_std, x, posterior, posterior_std
    )

    # The value itself, not just a box around it.  Hand-derived: in the focus
    # region (x >= 0.2) the band edges are truth {0.9, 1.1, 0.4, 0.6, 0.0} and
    # posterior {0.8, 1.6, 0.2, 0.6, 0.0}, so min=0.0 (clamped no-op) and
    # max=1.6; span=1.6, pad=0.12*1.6=0.192.  Agreement measured at 0.0
    # relative difference in both components.
    np.testing.assert_allclose(
        (ymin, ymax), (-0.192, 1.792), rtol=1e-12, atol=0.0
    )

    # True value -0.192 (see docstring); this bound only requires the sign, so a
    # pad as small as 1e-9 would pass exactly as readily as the real
    # pad_fraction=0.12 -- see the docstring for what that misses.
    assert ymin < -0.0
    # True value 1.792; 1.6 is the un-padded posterior upper edge (1.2 + 0.4) at
    # x=0.2, so this line mainly checks pad > 0 together with the x >= 0.2 cutoff.
    assert ymax > 1.6
    # Loose ceiling (true value -0.192): only rules out a grossly oversized pad
    # (pad_fraction gtrsim 6.25, vs. the real 0.12).
    assert ymin > -10.0
    # Loose ceiling (true value 1.792): only rules out a grossly oversized pad
    # (pad_fraction gtrsim 5.25, vs. the real 0.12).
    assert ymax < 10.0


@pytest.mark.parametrize("runner", RUNNERS)
def test_reproduction_ylim_has_stable_fallback_for_missing_focus_data(runner):
    """When nothing falls in the x >= focus_x window, returns the hardcoded default.

    Both ``x`` values (0.01, 0.1) are below the default ``focus_x=0.2``, and both
    curves are identically zero, so the ``x >= focus_x`` mask is all-False for the
    truth leg and the posterior leg alike; ``reproduction_ylim`` takes its
    early-return branch (``if finite.size == 0``) before ever touching
    ``min``/``max``/pad arithmetic.

    Oracle: ``F2``, a regression pin -- ``(-0.08, 1.08)`` is transcribed verbatim
    from the source's own hardcoded fallback literal
    (``closure_JAM_truth_small/run_closure.py:148``), not derived independently;
    there is no physics to check here, it is a fixed default axis range chosen so
    an empty panel still shows something sane.

    Bar: exact tuple equality.  Catches the fallback literal drifting, or the
    guard being removed so an empty ``finite`` reaches ``np.min``/``np.max`` and
    raises instead of returning (``ValueError: zero-size array``).  Audit item
    ``-M01`` was that this ran only ``jam_small``, so the byte-identical
    ``jam_full``/``nnpdf_full``/``nnpdf_small`` bodies never took this branch
    anywhere in the suite.  It was widened to all four; since 2026-08-14 it runs
    the ``RUNNERS`` pair (``nnpdf_full``, ``nnpdf_small``), which still covers both
    the full and ``_small`` layouts and so still closes ``-M01`` -- what it no
    longer does is re-enter the branch in the two JAM copies, whose bodies
    ``test_the_duplicated_run_closure_helpers_are_one_implementation`` pins as the
    same implementation.

    Distinct from ``test_reproduction_ylim_pads_a_degenerate_all_zero_band``
    below: there the focus mask *does* select points, they are simply all zero,
    which reaches the ``span > 0.0`` fallback instead of this one.
    """
    x = np.array([0.01, 0.1])
    zeros = np.zeros_like(x)
    assert runner.reproduction_ylim(x, zeros, zeros, x, zeros, zeros) == (
        -0.08,
        1.08,
    )
    # Transcribed verbatim from the source's own hardcoded fallback literal;
    # no independent derivation -- see docstring.


@pytest.mark.parametrize("runner", RUNNERS)
def test_reproduction_ylim_floor_comes_from_the_lower_band(runner):
    """A fixture where ``center - spread`` alone sets ``ymin``.

    Closes audit weakness ``test_closure_plot_scaling-01`` / missing item
    ``-M03``.  ``test_reproduction_ylim_contains_both_bands_above_point_two``
    cannot see the lower band being dropped, because its own minimum is an
    exact zero contributed by a point with ``std=0``.  Here nothing in the
    focus region is zero, and the unique extremes are

    * ``min = -0.25`` -- the *truth lower* edge at ``x=0.9`` (``0.20 - 0.45``),
      strictly negative so the ``min(0.0, ...)`` clamp is a no-op and cannot
      supply the floor instead;
    * ``max = +1.30`` -- the *truth upper* edge at ``x=0.3`` (``1.0 + 0.3``).

    Both are unique (no ties: the sorted focus-region values are ``-0.25,
    0.13, 0.2, 0.37, 0.4, 0.55, 0.65, 0.7, 0.8, 0.8, 1.25, 1.3``), so each
    band edge is individually load-bearing.  Oracle ``A1``: hand-derived from
    the literals, ``span = 1.55``, ``pad = 0.12 * 1.55 = 0.186``, giving
    ``(-0.436, 1.486)``.

    Acceptance (measured by mutation this pass): a copy that appends only
    ``center + spread`` returns ``(-0.156, 1.456)`` -- its lowest value becomes
    ``+0.37`` and the clamp pulls ``ymin`` to ``0.0``; a copy that appends only
    ``center - spread`` returns ``(-0.364, 0.814)``.  Both fail here.  The
    upper-only one, which is the mutant the audit found nothing could see,
    **passes** ``test_reproduction_ylim_contains_both_bands_above_point_two``
    -- that is the original finding, demonstrated.  (The lower-only mutant does
    fail that older test, because dropping the upper edge also lowers ``ymax``
    below its ``1.6`` bound; only the lower edge was invisible.)
    """
    x = np.array([0.01, 0.1, 0.3, 0.6, 0.9])
    truth = np.array([50.0, 8.0, 1.0, 0.6, 0.20])
    truth_std = np.array([25.0, 4.0, 0.3, 0.2, 0.45])
    posterior = np.array([-90.0, -15.0, 0.9, 0.5, 0.25])
    posterior_std = np.array([45.0, 20.0, 0.35, 0.3, 0.12])

    got = runner.reproduction_ylim(
        x, truth, truth_std, x, posterior, posterior_std
    )

    # rtol=1e-12 with an explicit atol=0.0: a pure relative bar, over decimal
    # round-tripping only.  Measured this pass, both components agree with the
    # hand-derived value at 0.0 relative difference.
    np.testing.assert_allclose(got, (-0.436, 1.486), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("sign", (+1.0, -1.0))
@pytest.mark.parametrize("runner", RUNNERS)
def test_reproduction_ylim_clamps_a_one_signed_band_to_include_zero(runner, sign):
    """Zero stays on-panel even when no band edge comes near it.

    Closes audit weakness ``test_closure_plot_scaling-02`` / missing item
    ``-M02``.  The source docstring promises the limits contain the bands
    "plus zero and modest headroom", implemented as
    ``ymin = min(0.0, ...)`` / ``ymax = max(0.0, ...)``; before this test no
    fixture in the suite needed either clamp, because an exact-zero data point
    already sat at the extremes.

    Both fixtures here are strictly one-signed in the focus region, so the
    clamp is the only thing that can put zero inside the returned interval:

    * ``sign=+1``: edges span ``[+0.4, +1.1]``.  Unclamped the answer would be
      ``(0.316, 1.184)`` -- an axis on which the ``ax.axhline(0.0)`` drawn by
      ``plot_reproduction`` would fall off the bottom.  Clamped: ``ymin=0.0``,
      ``ymax=1.1``, ``span=1.1``, ``pad=0.132`` -> ``(-0.132, 1.232)``.
    * ``sign=-1``: the exact mirror, ``[-1.1, -0.4]`` -> ``(-1.232, 0.132)``.

    Oracle ``A1`` (hand-derived from the literals).  Acceptance (measured by
    mutation this pass): replacing ``min(0.0, ...)``/``max(0.0, ...)`` with the
    bare ``float(np.min(finite))``/``float(np.max(finite))`` returns
    ``(0.316, 1.184)`` and ``(-1.184, -0.316)`` respectively, failing all eight
    of this test's items.  It also fails
    ``test_plot_reproduction_sets_each_panel_ylim_from_its_own_bands``, whose
    expectation is a closed form rather than a re-call of the helper and which
    therefore sees a broken helper too; every other test in this file passes.
    Before this test existed, that clamp mutation was invisible to the whole
    suite.
    """
    x = np.array([0.3, 0.6, 0.9])
    truth = sign * np.array([1.0, 0.8, 0.5])
    truth_std = np.array([0.1, 0.1, 0.1])
    posterior = sign * np.array([0.9, 0.7, 0.6])
    posterior_std = np.array([0.2, 0.1, 0.1])

    ymin, ymax = runner.reproduction_ylim(
        x, truth, truth_std, x, posterior, posterior_std
    )

    expected = (-0.132, 1.232) if sign > 0.0 else (-1.232, 0.132)
    np.testing.assert_allclose((ymin, ymax), expected, rtol=1e-12, atol=0.0)
    # The property the numbers encode, stated directly: zero is inside, and
    # strictly inside on the side the data never reaches.
    assert ymin < 0.0 < ymax
    interior = 0.4 if sign > 0.0 else -0.4
    assert ymin < interior < ymax


@pytest.mark.parametrize("runner", RUNNERS)
def test_reproduction_ylim_honours_explicit_focus_x_and_pad_fraction(runner):
    """The two keyword-only parameters are never passed anywhere in the repo.

    Closes missing item ``test_closure_plot_scaling-M04``.  ``focus_x`` and
    ``pad_fraction`` are keyword-only with defaults ``0.2``/``0.12``, and grep
    finds no call site in ``tests/`` or ``closure_*/`` that overrides either --
    ``plot_reproduction`` itself passes six positionals and nothing else.  So
    the non-default path of both was dead from a test standpoint, and a body
    that ignored its arguments and used the literals ``0.2``/``0.12`` directly
    would have been indistinguishable.

    Same fixture as ``test_reproduction_ylim_floor_comes_from_the_lower_band``,
    whose default-argument answer is ``(-0.436, 1.486)``.  Oracle ``A1``, each
    case hand-derived:

    * ``focus_x=0.6`` drops the ``x=0.3`` column, which owned the maximum
      ``+1.30``; the extremes become ``(-0.25, +0.80)``, ``span=1.05``,
      ``pad=0.126`` -> ``(-0.376, 0.926)``.
    * ``pad_fraction=0.0`` isolates the pad term: the return must be the
      clamped extremes themselves, ``(-0.25, 1.3)``, with no headroom at all.
      This is the strongest of the three, since it pins the un-padded value
      exactly rather than through the pad arithmetic.
    * ``pad_fraction=0.25`` scales that headroom: ``0.25 * 1.55 = 0.3875``
      -> ``(-0.6375, 1.6875)``.

    Acceptance (measured by mutation this pass): a body that ignores the
    incoming ``focus_x`` (uses the literal ``0.2``) returns the default answer
    ``(-0.436, 1.486)`` for the ``focus_x=0.6`` call and fails there; one that
    ignores ``pad_fraction`` (uses the literal ``0.12``) returns the same
    default for both ``pad_fraction`` calls and fails on each.  The
    ``focus_x``-ignoring mutation is invisible to every other test in this
    file.
    """
    x = np.array([0.01, 0.1, 0.3, 0.6, 0.9])
    truth = np.array([50.0, 8.0, 1.0, 0.6, 0.20])
    truth_std = np.array([25.0, 4.0, 0.3, 0.2, 0.45])
    posterior = np.array([-90.0, -15.0, 0.9, 0.5, 0.25])
    posterior_std = np.array([45.0, 20.0, 0.35, 0.3, 0.12])
    args = (x, truth, truth_std, x, posterior, posterior_std)

    # Control: the defaults, so a mutant that simply broke the function is not
    # mistaken for one that ignores its keywords.
    np.testing.assert_allclose(
        runner.reproduction_ylim(*args), (-0.436, 1.486), rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        runner.reproduction_ylim(*args, focus_x=0.6),
        (-0.376, 0.926),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        runner.reproduction_ylim(*args, pad_fraction=0.0),
        (-0.25, 1.3),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        runner.reproduction_ylim(*args, pad_fraction=0.25),
        (-0.6375, 1.6875),
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("runner", RUNNERS)
def test_reproduction_ylim_pads_a_degenerate_all_zero_band(runner):
    """A zero-width band still gets a nonzero axis range.

    Not an audit item -- found while closing ``-M02``: ``pad = pad_fraction *
    (span if span > 0.0 else max(abs(ymin), 1.0e-3))`` has a second branch that
    no test in the suite reached.  After the zero clamp, ``ymin <= 0 <= ymax``,
    so ``span == 0`` forces ``ymin == ymax == 0``: the branch is live exactly
    when every selected band edge is zero, which is what a field that has been
    completely switched off looks like on a reproduction panel.

    Distinct from ``test_reproduction_ylim_has_stable_fallback_for_missing_
    focus_data``: there the focus mask selects nothing and the function returns
    early; here it selects two points that happen to be zero, so control
    reaches the clamp and the pad.

    Oracle ``A1``: ``pad = 0.12 * max(0.0, 1e-3) = 1.2e-4`` ->
    ``(-1.2e-4, +1.2e-4)``, and ``0.5 * 1e-3 = 5e-4`` -> ``(-5e-4, +5e-4)``,
    which also shows the floor is a *floor* and not a hardcoded output.
    Acceptance (measured by mutation this pass): simplifying the expression to
    ``pad = pad_fraction * span`` returns ``(0.0, 0.0)`` -- a degenerate ylim
    Matplotlib would have to expand for itself -- and fails both assertions
    while every other test in this file still passes.
    """
    x = np.array([0.3, 0.9])
    zeros = np.zeros_like(x)

    np.testing.assert_allclose(
        runner.reproduction_ylim(x, zeros, zeros, x, zeros, zeros),
        (-1.2e-4, 1.2e-4),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        runner.reproduction_ylim(x, zeros, zeros, x, zeros, zeros, pad_fraction=0.5),
        (-5.0e-4, 5.0e-4),
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("runner", RUNNERS)
def test_reproduction_ylim_drops_nonfinite_points_instead_of_propagating_them(runner):
    """NaN/Inf entries are removed, not clipped and not propagated.

    Closes missing item ``test_closure_plot_scaling-M06``.  No test anywhere in
    the suite passed a non-finite value to this helper, although tolerating
    missing/undefined curve points is the stated reason both ``np.isfinite``
    filters exist.

    First fixture: ``truth`` is ``NaN`` at ``x=0.6`` and ``posterior_std`` is
    ``+inf`` at ``x=0.6``, both inside the focus region.  The surviving focus
    values are ``-0.2, 0.15, 0.25, 0.55, 0.7, 1.0, 1.25, 1.3``, so
    ``(-0.2, 1.3)``, ``span=1.5``, ``pad=0.18`` -> ``(-0.38, 1.48)``
    (oracle ``A1``, hand-derived).

    Second fixture pins the *concatenated* filter specifically, which is the
    only one of the two that is not redundant (module docstring): ``center``
    and ``spread`` are both the finite ``1e308``, so the per-point mask passes
    them through, and only ``center + spread`` overflows to ``+inf``.  The
    surviving values are ``1e308 - 1e308 = 0.0`` and the ``x=0.9`` column's
    ``0.4``/``0.6``, giving ``(0.0, 0.6)``, ``span=0.6``, ``pad=0.072`` ->
    ``(-0.072, 0.672)``.

    Acceptance, all three measured by mutation this pass:

    * deleting ``finite = finite[np.isfinite(finite)]`` leaves the first
      fixture *unchanged* at ``(-0.38, 1.48)`` -- the per-point mask has
      already removed the same points -- but returns ``(-inf, inf)`` on the
      second, so the overflow case is what makes this test able to fail on
      that mutation at all;
    * deleting only the per-point ``np.isfinite`` terms from ``mask`` is
      byte-identical on both fixtures, and on every other fixture in this
      file.  That is the redundancy stated in the module docstring, and no
      test here or elsewhere in the suite can catch it;
    * deleting *both* returns ``(-1.2e-4, 1.2e-4)`` on the first fixture and
      ``(-inf, inf)`` on the second.  The first is worth noting: ``NaN`` does
      not propagate to the caller, because Python's ``min(0.0, nan)`` and
      ``max(0.0, nan)`` both answer ``0.0``, so an unfiltered ``NaN`` silently
      collapses the panel onto the degenerate all-zero axis instead of raising
      -- a failure that would look like a blank plot, not like an error.
    """
    x = np.array([0.01, 0.3, 0.6, 0.9])
    truth = np.array([10.0, 1.0, np.nan, 0.4])
    truth_std = np.array([5.0, 0.3, 0.2, 0.6])
    posterior = np.array([-8.0, 0.9, 0.5, 0.2])
    posterior_std = np.array([4.0, 0.35, np.inf, 0.05])

    np.testing.assert_allclose(
        runner.reproduction_ylim(x, truth, truth_std, x, posterior, posterior_std),
        (-0.38, 1.48),
        rtol=1e-12,
        atol=0.0,
    )

    huge_x = np.array([0.3, 0.9])
    huge = np.array([1e308, 0.5])
    huge_std = np.array([1e308, 0.1])
    with np.errstate(over="ignore"):  # the overflow is the point of the case
        got = runner.reproduction_ylim(
            huge_x, huge, huge_std, huge_x, huge, huge_std
        )
    np.testing.assert_allclose(got, (-0.072, 0.672), rtol=1e-12, atol=0.0)


def _reproduction_inputs(runner, *, with_truth_std=True):
    """Truth/marginal dicts for ``plot_reproduction``, one distinct scale per field.

    Field ``i`` gets every curve multiplied by ``i + 1``.  ``reproduction_ylim``
    is homogeneous of degree one in its ``y`` arguments (min, max, span and pad
    all scale, and the zero clamp commutes with a positive scaling), so panel
    ``i``'s limits are panel 0's times ``i + 1`` -- which both makes all nine
    answers distinct and gives them a closed form.

    The truth leg is sampled on five nodes and the posterior leg on four
    *different* ones, so the two legs are not interchangeable: crossing the
    ``x`` arrays raises ``ValueError: operands could not be broadcast together
    with shapes (4,) (5,)`` (measured) rather than quietly returning a
    plausible number.
    """
    fields = runner.cfg.ALL_FIELDS
    x_nodes = np.array([0.05, 0.15, 0.3, 0.6, 0.9])
    x_post = np.array([0.05, 0.25, 0.5, 0.95])
    curve = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    curve_std = np.array([0.2, 0.15, 0.1, 0.08, 0.3])
    post = np.array([1.0, 0.6, 0.4, 0.15])
    post_std = np.array([0.3, 0.2, 0.1, 0.05])

    truth = {
        "x_nodes": x_nodes.tolist(),
        "curves": {n: (curve * (i + 1)).tolist() for i, n in enumerate(fields)},
    }
    if with_truth_std:
        truth["curve_std"] = {
            n: (curve_std * (i + 1)).tolist() for i, n in enumerate(fields)
        }
    marginal = {
        n: (x_post, post * (i + 1), post_std * (i + 1))
        for i, n in enumerate(fields)
    }
    return truth, marginal


@pytest.mark.parametrize("runner", RUNNERS)
def test_plot_reproduction_sets_each_panel_ylim_from_its_own_bands(runner, tmp_path):
    """The real ``ax.set_ylim(*reproduction_ylim(...))`` call site, per panel.

    Closes missing item ``test_closure_plot_scaling-M05``.  Every other test in
    this file calls the helper in isolation; nothing anywhere in the suite
    called ``plot_reproduction``, the function that actually renders the
    reproduction grid and is the helper's only caller.  A correct helper wired
    to the wrong arguments, applied once instead of per panel, or not applied
    at all was therefore invisible.

    Note on scope, since the audit plan mis-billed this: the ``plot_
    reproduction`` under test is ``closure_{JAM,NNPDF}_truth{,_small}/
    run_closure.py``'s, signature ``(q_key, mode, marginal, truth, path)``.
    The two same-named functions in ``examples/closure_test.py`` and
    ``examples/closure_test_nme.py`` take ``(records, fits, path)``, are a
    different function, and never call ``reproduction_ylim``.

    Method: ``save_figure_both`` is wrapped (not replaced) so the nine Axes can
    be read at the moment the figure is complete -- ``plot_reproduction``
    returns ``None`` and closes the figure, so there is no other capture point
    -- then the real save runs and the PNG/PDF must appear under ``tmp_path``.

    Oracle ``A1``, not a re-call of the helper: with this fixture the focus
    region (``x >= 0.2``) has band edges ``{0.4, 0.6, 0.22, 0.38, -0.2, 0.4}``
    on the truth leg and ``{0.4, 0.8, 0.3, 0.5, 0.10, 0.20}`` on the posterior
    leg, so ``ymin=-0.2`` comes from the *truth* lower edge at ``x=0.9`` and
    ``ymax=0.8`` from the *posterior* upper edge at ``x=0.25``.  Both legs are
    individually load-bearing.  ``span=1.0``, ``pad=0.12``, giving
    ``(-0.32, 0.92)`` for the first field and ``(i+1)`` times that for the
    rest.  Dropping ``curve_std`` moves the truth edges to the bare curve, so
    ``ymin`` clamps to ``0`` and ``ymax`` stays ``0.8``: ``(-0.096, 0.896)``
    times ``(i+1)``.

    Bar ``rtol=1e-12, atol=0.0``; measured this pass the largest relative
    difference across all 18 limits is ``2.22e-16`` (one ULP), so the margin is
    over ``set_ylim``/decimal round-tripping, which stores the floats verbatim.

    Acceptance (measured by mutation this pass, each against a mutated copy of
    ``plot_reproduction`` installed on all four modules; an identity copy
    passes all 36 items in this file, so the harness itself is not what
    fails): using the posterior's ``x`` for the truth leg raises
    ``ValueError``; substituting zeros for ``t_std`` returns the
    ``no-curve_std`` limits and fails the first case; computing the limits once
    from the first field and applying them to all nine fails panels 2-9;
    reversing the tuple into ``set_ylim`` mismatches 18/18 limits; and deleting
    the ``set_ylim`` call leaves Matplotlib's autoscaled limits and fails all
    nine.  Each of the five fails only this test.
    """
    cfg = runner.cfg
    # The panel grid is a literal 3x3 and the field list is zipped against it,
    # so a tenth field would be dropped silently.  Pin the assumption here.
    assert len(cfg.ALL_FIELDS) == 9, cfg.ALL_FIELDS

    captured = {}
    real_save = runner.save_figure_both

    def capture(fig, path, **kwargs):
        captured["ylim"] = [tuple(ax.get_ylim()) for ax in fig.axes]
        return real_save(fig, path, **kwargs)

    for with_std, expected_unit in ((True, (-0.32, 0.92)), (False, (-0.096, 0.896))):
        truth, marginal = _reproduction_inputs(runner, with_truth_std=with_std)
        out = tmp_path / f"repro_{with_std}.png"
        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(runner, "save_figure_both", capture)
            runner.plot_reproduction("2", "closure", marginal, truth, out)
        finally:
            monkey.undo()

        assert out.exists() and out.with_suffix(".pdf").exists()
        assert len(captured["ylim"]) == len(cfg.ALL_FIELDS)
        expected = [
            ((i + 1) * expected_unit[0], (i + 1) * expected_unit[1])
            for i in range(len(cfg.ALL_FIELDS))
        ]
        np.testing.assert_allclose(
            captured["ylim"], expected, rtol=1e-12, atol=0.0
        )
        # Non-degeneracy: nine identical panels would satisfy a per-panel check
        # vacuously, so require the nine answers to be nine different answers.
        assert len(set(captured["ylim"])) == len(cfg.ALL_FIELDS)
