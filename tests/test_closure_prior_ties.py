"""Every closure field prior must be ``a +- a``: mean tied to covariance sigma.

This relation used to be two independent constants that merely happened to
agree, and the prose describing them went stale twice (see
``plans/lambda_guard_and_doc_defects.md`` 5b/5c).  It is now enforced at the
model level with ``analysis.tie``; these tests keep it that way.

Exercises ``_gp_prior`` and ``build_analysis`` in ``closure_NNPDF_truth/fit.py``
and ``closure_NNPDF_truth_small/fit.py``, each with its own
``analysis.tie(params.mean.N, to=params.cov.sigma)`` call (``fit.py:228`` in the
full suite, ``fit.py:179`` in the ``_small`` one).

Two sibling packages -- ``closure_JAM_truth`` and ``closure_JAM_truth_small`` --
carry their own copies of that same wiring against a different truth PDF set, and
until 2026-08-14 every test here ran against all four on the argument that four
independently maintained copies can drift apart.  The owner has since dropped the
JAM legs as duplication: **nothing in this file reads the truth PDF set at all** --
every oracle is the suite's own ``cfg`` constants and the object graph
``build_analysis`` assembles -- so a JAM leg re-ran the identical
``normalize_ties``/``apply_ties`` path for an answer that differed only in which
copy of the constants it was read from.  What is genuinely given up is the
per-copy drift guard on the two JAM ``config.py``/``fit.py`` files; the one JAM
case retained -- ``small_jam`` in ``_TIE_REBUILD_SUITES``, see the note there --
is the canary left in place for it.

Most tests here are internal-consistency checks (oracle A3/F1) on the suite's
own object graph: they build a real ``Model`` and check that ``analysis.tie``'s
bookkeeping (``model.ties``, ``parameter_table()['tied_to']``) agrees with its
*effect* -- moving one value through ``model.apply_parameter_overrides`` and
reading the other back.  That is the right oracle for what is being pinned
(wiring, not physics), but it means most bars in this file measure float
round-trip precision, not an accuracy floor.  The one closed-form check (A1) is
``test_floating_attaches_the_floor_and_a_centred_hyper_prior``'s ``13.7209``
total; see that test's docstring for the independent verification.

The general tie mechanism -- including the "share one Parameter object" trap
``analysis.tie`` exists to avoid -- is exercised independently in
``tests/test_parameter_ties.py`` and documented in
``guides/tying_parameters.md``; this file only checks that the closure suites'
*use* of that mechanism, not the mechanism itself, is wired correctly.  Two
tests (``test_build_ties_mean_to_sigma``, ``test_nuisance_fields_are_not_tied``)
call ``build_analysis`` and assemble a real ``Model`` including lattice
pseudo-ITD datasets; measured in ``tests/test_durations.json``, those two reach
7.28s and 3.27s for the full suite (well under 0.5s for the ``_small`` ones),
against well under 0.3s for every other test in this file, which touch only
``cfg``/``_gp_prior`` directly.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from closure_NNPDF_truth import fit as full_nnpdf
from closure_JAM_truth_small import fit as small_jam
from closure_NNPDF_truth_small import fit as small_nnpdf


#: The suites every parametrized test in this file runs against: NNPDF truth only,
#: on the owner's instruction 2026-08-14.  See the module docstring for why the two
#: JAM legs were duplication rather than coverage.  ``_TIE_REBUILD_SUITES`` below
#: deliberately differs from this tuple -- it adds ``small_jam`` back for one test.
SUITES = (full_nnpdf, small_nnpdf)


# -- the constants -----------------------------------------------------------


@pytest.mark.parametrize("suite", SUITES)
def test_one_amplitude_drives_both_mean_and_sigma(suite):
    """``gp_mean(name)``/``gp_sigma(name)`` describe the prior ``_gp_prior``
    actually builds, and both equal one amplitude.

    ``gp_mean``/``gp_sigma`` both literally ``return gp_amplitude(field)``, and
    repo-wide grep still finds no caller of either anywhere except their own
    definitions and this test -- ``_gp_prior`` (fit.py:122) builds the prior
    from ``cfg.gp_amplitude`` directly, and has done so since the commit that
    added this test file.  Comparing them to ``gp_amplitude`` therefore compares
    a function to its own body: it can only fail if an alias is hand-edited to
    disagree with itself, and such an edit changes nothing anyone fits.

    **Open source-side finding (weakness ``test_closure_prior_ties-02``,
    re-verified 2026-08-14 against current source; a maintainer call, not this
    test's to make).**  The two accessors are dead in all four suites:

    * ``closure_JAM_truth/config.py`` -- ``gp_mean`` 1000-1002, ``gp_sigma`` 1005-1007
    * ``closure_NNPDF_truth/config.py`` -- ``gp_mean`` 998-1000, ``gp_sigma`` 1003-1005
    * ``closure_JAM_truth_small/config.py`` -- ``gp_mean`` 717-719, ``gp_sigma`` 722-724
    * ``closure_NNPDF_truth_small/config.py`` -- ``gp_mean`` 720-722, ``gp_sigma`` 725-727

    Each body is exactly ``return gp_amplitude(field)``.  The only production
    consumer of any of the three is ``fit.py:122``
    (``value = cfg.gp_amplitude(field)``), identical in all four suites, and
    ``config.py``'s own ``gp_amplitude_prior`` calls ``gp_amplitude`` directly
    too.  So ``gp_mean``/``gp_sigma`` have **zero** callers outside their own
    definitions and this file.

    Two resolutions were on the table, and **measurement 2026-08-14 removed one
    of them.**  (1) Delete both functions from all four ``config.py`` copies, and
    with them the two ``==`` asserts below (the two in
    ``test_prior_and_accessors_all_follow_one_patched_amplitude`` go too; every
    other assertion in both tests is against ``_gp_prior``'s output and survives
    untouched).  (2) Change ``fit.py:122`` to read ``cfg.gp_mean(field)`` for the
    mean and ``cfg.gp_sigma(field)`` for the covariance amplitude.

    **Option 2 does not restore a consumer for ``gp_mean``.**  ``build_analysis``
    ties the mean to the covariance amplitude -- ``analysis.tie(params.mean.N,
    to=params.cov.sigma)``, ``fit.py:228`` in the full suites and ``fit.py:179``
    in the ``_small`` ones -- and a tie *discards* the target's declared value at
    compile time rather than merely constraining it later.  Measured directly
    (``pixel.Analysis`` with ``Const(N=7.0)`` and ``RBF(sigma=2.0)``, tie, then
    ``parameter_table()``): ``mean.N`` reads back **2.0**, ``tied_to =
    fields.q.prior.cov.sigma``, ``frozen = True``.  The same fact is asserted
    in-repo by ``tests/test_parameter_ties.py::
    test_compiled_model_reports_the_tie_and_drops_one_coordinate`` (declared
    ``1.0``, inherited ``2.0``).  So under option 2 the number ``gp_mean`` returns
    would be handed to ``priors.Const`` and then overwritten before any fit sees
    it: ``gp_mean`` would gain a *call site* but still no consumer, which is worse
    than today's orphan because it would read as wired.  A ``gp_mean`` that
    drifted from ``gp_sigma`` would still change nothing anyone fits -- exactly
    the situation this weakness records.  **Deletion is the only resolution that
    removes the orphan**; it is the owner's call, and the closure packages are
    held.

    So the two accessor checks are now anchored to the **built prior** instead:
    ``_gp_prior(name).mean.N.value`` and ``.cov.sigma.value`` -- the numbers
    that reach ``priors.Const`` and ``priors.LogRBF`` -- are required to equal
    ``gp_mean(name)`` and ``gp_sigma(name)`` exactly.  The accessors then mean
    something operationally ("this is what the prior is built with"), and the
    exact ``==`` is stricter than any other check in the file: the sd
    comparison in ``test_gp_prior_starts_at_a_plus_minus_a`` carries
    ``rtol=1e-9`` for kernel jitter, so an amplitude perturbed by 1e-11
    relative passes there and fails here (measured 2026-08-13: that mutation
    fails this test alone, leaving the file's other tests passing).

    What this still cannot see, and
    ``test_prior_and_accessors_all_follow_one_patched_amplitude`` (below) can:
    whether the accessors *delegate* to ``gp_amplitude`` or keep a hand-copied
    tier table that agrees with it today.  Both satisfy every ``==`` here.

    Oracle A3 throughout -- an identity between the config accessors and the
    prior object the same package builds, not an external number.  The
    ``hasattr`` loop below is a separate regression guard: it fails if the
    pre-tie four-constant layout (``GP_MEAN_HIGH`` etc.) comes back.
    """
    cfg = suite.cfg
    for name in cfg.ALL_FIELDS:
        amplitude = cfg.gp_amplitude(name)
        # gp_mean/gp_sigma are kept as aliases; they must not be able to drift.
        assert cfg.gp_mean(name) == amplitude
        assert cfg.gp_sigma(name) == amplitude
        # ... and they must still describe the prior that is actually built:
        # exact equality on the values handed to Const/LogRBF, which no other
        # test in this file demands (their bars carry jitter margin).
        prior = suite._gp_prior(name)
        assert float(prior.mean.N.value) == cfg.gp_mean(name)
        assert float(prior.cov.sigma.value) == cfg.gp_sigma(name)
    if hasattr(cfg, "GP_AMPLITUDES"):
        # Per-field suites (the ``_small`` pair, 2026-08-15).  There is no tier
        # constant to pin; the table itself is the design, and the check that
        # matters is that it covers exactly the nine fields with no field
        # silently falling back to a default.
        assert set(cfg.GP_AMPLITUDES) == set(cfg.ALL_FIELDS)
        # The table carries the original tier values; a per-field retune was
        # measured and reverted (see the config note).  Pin the *structure* --
        # every field named, no silent default -- rather than the numbers, which
        # are a prior-width design choice this file deliberately does not judge.
        assert set(cfg.GP_AMPLITUDES.values()) <= {1.0, 5.0}
        assert {n for n in cfg.ALL_FIELDS if cfg.gp_amplitude(n) == 5.0} == {
            "sigma", "g", "t15"
        }
    else:
        assert cfg.GP_AMPLITUDE_HIGH == 5.0  # singlet / gluon / charm
        assert cfg.GP_AMPLITUDE_LOW == 1.0  # non-singlets and valence; width kept
    # The four separate constants are gone, which is what let the prose drift.
    for stale in ("GP_MEAN_HIGH", "GP_SIGMA_HIGH", "GP_MEAN_LOW", "GP_SIGMA_LOW"):
        assert not hasattr(cfg, stale), f"{stale} came back"


#: Nine probe amplitudes, one per field.  Deliberately **all distinct** and none
#: equal to either tier constant (5.0 / 1.0), because the whole point of the test
#: below is that an accessor or a prior which ignores ``gp_amplitude`` and reads
#: the tier table itself must be *visibly* wrong.  All sit above
#: ``GP_AMPLITUDE_FLOOR`` (0.99), so ``Parameter``'s bounds are not tripped.
_PROBE_AMPLITUDES = tuple(2.5 + 0.37 * index for index in range(9))


def _min_amplitude(cfg) -> float:
    """Smallest starting amplitude in the suite, whichever scheme it uses."""
    return min(cfg.gp_amplitude(name) for name in cfg.ALL_FIELDS)


@pytest.mark.parametrize("suite", SUITES)
def test_prior_and_accessors_all_follow_one_patched_amplitude(suite, monkeypatch):
    """``gp_amplitude`` is the *single* source: patch it, and the two accessors
    **and** the built prior all move with it, per field.

    This is the half of weakness ``test_closure_prior_ties-02`` that can be
    closed from ``tests/``.  The rest of that weakness is a source-side judgment
    call recorded in the test above.

    **What the existing checks cannot see.**  ``assert cfg.gp_mean(name) ==
    cfg.gp_amplitude(name)`` calls one function twice under two names, so it
    passes whether ``gp_mean`` *delegates* to ``gp_amplitude`` or *re-implements
    the tier table by hand and happens to agree today*.  Those two are the same
    assertion and very different code: the second is a silent second source of
    truth, which is precisely the arrangement
    ``plans/lambda_guard_and_doc_defects.md`` 5b/5c records going stale twice
    here.  Patching ``gp_amplitude`` separates them -- a delegating accessor
    follows the patch, a hand-copied tier table does not.

    **And it breaks a fixture degeneracy nothing else in this file breaks.**
    ``gp_amplitude`` returns only two distinct numbers across nine fields (5.0
    for three, 1.0 for six), so every other per-field check here is blind to a
    *within-tier cross-wire*: ``_gp_prior("t3")`` reading ``v3``'s amplitude
    would satisfy ``test_gp_prior_starts_at_a_plus_minus_a`` exactly, both being
    1.0.  Nine distinct probe values make each field's prior traceable to its
    own amplitude.

    Three things are pinned at once, all with the same patch:

    * ``gp_mean``/``gp_sigma`` delegate rather than duplicate;
    * ``_gp_prior`` **reads the accessor at call time** rather than closing over
      an import-time snapshot of the tier constants;
    * the number reaching ``priors.Const(N=...)`` and the one reaching
      ``priors.LogRBF(sigma=...)`` are the *same* per-field number, and remain so
      when that number is changed.

    What it does **not** do, so it is not mistaken for the decisive tie check:
    it still reads the prior at construction, having moved the *source function*
    beforehand.  ``test_build_ties_mean_to_sigma`` is the only test here that
    moves a value on an already-built model and re-reads the other end, which is
    what weakness ``-03`` is about.

    Oracle ``A3`` on an identity, at exact ``==``: every value is carried
    unmodified, so any tolerance would only hide a discrepancy.
    """
    cfg = suite.cfg
    probe = dict(zip(cfg.ALL_FIELDS, _PROBE_AMPLITUDES))
    assert len(probe) == len(cfg.ALL_FIELDS) == len(set(probe.values())) == 9
    # The probes must be distinguishable from the tier table, or an accessor that
    # ignored the patch could still pass.
    configured = set(
        cfg.GP_AMPLITUDES.values() if hasattr(cfg, "GP_AMPLITUDES")
        else (cfg.GP_AMPLITUDE_HIGH, cfg.GP_AMPLITUDE_LOW)
    )
    for value in probe.values():
        assert value not in configured
        assert value > cfg.GP_AMPLITUDE_FLOOR

    monkeypatch.setattr(cfg, "gp_amplitude", lambda field: probe[field])

    for name in cfg.ALL_FIELDS:
        # Delegation: an accessor holding its own copy of the tier table would
        # return 5.0 or 1.0 here and fail.
        assert cfg.gp_mean(name) == probe[name]
        assert cfg.gp_sigma(name) == probe[name]
        # ... and the prior really is built from that one number, read now.
        prior = suite._gp_prior(name)
        assert float(prior.mean.N.value) == probe[name]
        assert float(prior.cov.sigma.value) == probe[name]


@pytest.mark.parametrize("suite", SUITES)
def test_high_tier_is_singlet_gluon_charm(suite):
    """The three fields on ``GP_AMPLITUDE_HIGH`` are exactly ``{sigma, g, t15}``
    (D1: config.py's own comment on ``HIGH_PRIOR_FIELDS`` documents this as
    "C-even singlet (sigma), gluon (g) and charm (t15)"; the set literal here is
    that documented tier list, not an independently computed oracle).  Catches
    ``HIGH_PRIOR_FIELDS`` gaining or losing a field, or ``ALL_FIELDS`` changing
    count out from under it.  Says nothing about whether 5.0/1.0 are the *right*
    amplitudes -- that is a prior-width design choice, not a computed quantity.
    """
    cfg = suite.cfg
    assert len(cfg.ALL_FIELDS) == 9
    if hasattr(cfg, "GP_AMPLITUDES"):
        # The ``_small`` suites replaced the tiers with a per-field table on
        # 2026-08-15, so "the high tier is exactly {sigma, g, t15}" is no longer
        # a property they have: t15 dropped to 2.0 and g rose to 10.0, leaving
        # sigma and g as the only fields at or above 5.0.  What is still worth
        # pinning is that every field is named explicitly -- a field missing
        # from the table raises KeyError rather than silently taking a default.
        assert set(cfg.GP_AMPLITUDES) == set(cfg.ALL_FIELDS)
        for name in cfg.ALL_FIELDS:
            assert cfg.gp_amplitude(name) > 0.0
        with pytest.raises(KeyError):
            cfg.gp_amplitude("not_a_field")
    else:
        high = {n for n in cfg.ALL_FIELDS
                if cfg.gp_amplitude(n) == cfg.GP_AMPLITUDE_HIGH}
        assert high == {"sigma", "g", "t15"}  # 3 of the 9 fields


# -- the prior the constants build -------------------------------------------


@pytest.mark.parametrize("suite", SUITES)
def test_gp_prior_starts_at_a_plus_minus_a(suite):
    """Construction-time identity: ``prior_mean`` and ``sqrt(diag(prior_cov))``
    both equal ``cfg.gp_amplitude(name)`` (A3 -- both trace to the single call
    in ``_gp_prior``, fit.py:122-135, before any tie is attached).  W1: the NOTE
    below is right that this would still pass with the tie deleted, since
    ``_gp_prior`` never calls ``analysis.tie`` at all -- only ``build_analysis``
    does (fit.py:228).  Demonstrated directly (scratch check against
    ``pixel.priors``/``pixel.core.params``, not committed): building the same
    mean/sigma pair with sigma moved and mean left alone gives mean != sd, but
    this test never re-reads the prior after a move to see that -- only
    ``test_build_ties_mean_to_sigma`` does.  Bars rtol 1e-12 (mean) / 1e-9 (sd)
    are float round-trip and kernel-jitter margin, not physics; see the jitter
    NOTE below for the measured margin.
    """
    # NOTE: this checks the *starting values* only.  _gp_prior passes one
    # amplitude to both the mean and the kernel, so it would still pass with the
    # tie deleted -- the constants would simply agree by hand, which is exactly
    # the arrangement that let the prose drift twice.  The tie itself is tested
    # by test_build_ties_mean_to_sigma below, which moves the amplitude.
    cfg = suite.cfg
    nodes = np.asarray(cfg.make_grid().as_array(), dtype=float)
    for name in cfg.ALL_FIELDS:
        prior = suite._gp_prior(name)
        mean = np.asarray(prior.prior_mean(nodes))
        sd = np.sqrt(np.diag(np.asarray(prior.prior_cov(nodes))))
        amplitude = cfg.gp_amplitude(name)
        # rtol 1e-12: mean is a bare float carried through priors.Const with no
        # kernel or jitter term, so this bounds float64 round-trip only.
        np.testing.assert_allclose(mean, amplitude, rtol=1e-12, atol=0.0)
        # LogRBF has diag(K) = sigma**2 + jitter, so the relative excess of sd
        # over the amplitude is sqrt(1 + jitter/a**2) - 1 ~= jitter/(2 a**2) for
        # the default 1e-10 jitter.  It therefore grows as the amplitude *falls*,
        # quadratically:
        #     a=5.0  -> 2.0e-12      a=1.0  -> 5.0e-11
        #     a=0.25 -> 8.0e-10      a=0.2  -> 1.25e-09
        # The old rtol=1e-9 was set when the smallest amplitude was 1.0.  The
        # _small suites' per-field table (2026-08-15) put v8 at 0.2, which lands
        # at 1.25e-9 and failed the bar -- measured, the observed sd was
        # 0.20000000025 against 0.2.  1e-8 restores ~8x margin over the largest
        # jitter term any current amplitude produces.  This is a jitter budget,
        # not a physics tolerance: shrink the smallest amplitude by another 3x
        # and it must be revisited again.
        np.testing.assert_allclose(sd, mean, rtol=1e-8)


# -- the tie itself, on the assembled analysis -------------------------------


#: `test_build_ties_mean_to_sigma` is the one test in this file that still runs a JAM
#: leg, and the tuple below is deliberately **not** `SUITES`: it adds `small_jam` back.
#:
#: The full JAM case went first, on the owner's instruction 2026-08-14 -- the single most
#: expensive case in the file at **284.6 s** against NNPDF's 230.2 s, exercising an
#: identical code path (`pixel.core.params.normalize_ties`/`apply_ties` knows nothing
#: about which PDF set produced the truth).  When the rest of the JAM parametrization was
#: dropped from `SUITES` the same day, `small_jam` was kept **here** by the owner's
#: decision: it costs 0.37 s and it is this file's only remaining canary for a
#: JAM-specific wiring break in `closure_JAM_truth_small/fit.py`.  Do not "tidy" it to
#: match `SUITES`.
_TIE_REBUILD_SUITES = (full_nnpdf, small_jam, small_nnpdf)


@pytest.mark.parametrize("suite", _TIE_REBUILD_SUITES)
def test_build_ties_mean_to_sigma(suite):
    """The decisive check (A3, self-consistency): after
    ``model.apply_parameter_overrides`` sets ``cov.sigma`` to ``moved = 3.7``
    (equal to neither tier value), the rebuilt model's ``mean.N`` reads back
    ``3.7`` too.  ``apply_parameter_overrides`` reconstructs ``Model``
    (``_with_theta`` -> ``__init__``), which reruns
    ``pixel.core.params.normalize_ties``/``apply_ties`` (model.py:1036-1037) --
    the same path ``ravel``'s ``rebuild`` runs under ``jit``/``grad`` during a
    real fit -- so this exercises the actual propagation, not just the
    ``tied_to`` label.  Catches: a dropped ``analysis.tie(...)`` call
    (``len(model.ties)`` would read 0, first assert fails below), a tie
    declared to the wrong source, or ``apply_ties`` failing to propagate a
    moved value through a full model rebuild.
    """
    # The decisive check: build the real analysis and *move* the amplitude.  If
    # the tie were dropped, mean.N would stay at its declared value while sigma
    # moved, and the prior would silently stop being a +- a.
    cfg = suite.cfg
    # Not a literal: the _small suites reduced TRUTH_Q_CHOICES to {2, 3} on
    # 2026-08-15, and this test is parametrized across suites with different
    # member tables.  Any member builds the same prior wiring.
    analysis, _ = suite.build_analysis(
        next(iter(suite.cfg.TRUTH_Q_CHOICES)), "lattice"
    )
    model = analysis.compile()

    assert len(model.ties) == len(cfg.ALL_FIELDS)  # 9: one per physical field
    assert model.n_free == 0  # the tie must not add a sampled hyper-parameter

    rows = {row["selector"]: row for row in model.parameter_table()}
    for name in cfg.ALL_FIELDS:
        assert (
            rows[f"fields.{name}.prior.mean.N"]["tied_to"]
            == f"fields.{name}.prior.cov.sigma"
        )

    # **A DISTINCT value per field, applied in ONE rebuild.**
    #
    # Two changes from the original loop, both deliberate.  It moved every field to the
    # same 3.7, one `apply_parameter_overrides` call per field.  That is nine full `Model`
    # reconstructions -- measured 2026-08-14 at **~31 s each, 280 s of the test's 284 s**
    # (`build_analysis` itself is only ~4 s; sibling tests in this file that build the same
    # analysis run in 4.6 s).  `apply_parameter_overrides` takes a *list*, so one call does
    # the same work.
    #
    # And giving each field its own value makes the check strictly stronger.  With every
    # field moved to the same number, a **cross-wired tie** -- `t3`'s mean following `t8`'s
    # sigma -- reads back 3.7 and passes.  Distinct values catch that; the original could
    # not.  Values are spaced well away from `GP_AMPLITUDE_HIGH`/`LOW` and from each other.
    moved = {name: 3.7 + 0.11 * i for i, name in enumerate(cfg.ALL_FIELDS)}
    assert len(set(moved.values())) == len(cfg.ALL_FIELDS)  # no accidental collision
    shifted = model.apply_parameter_overrides([
        {"op": "set", "selector": f"fields.{name}.prior.cov.sigma", "value": value}
        for name, value in moved.items()
    ])

    for name, value in moved.items():
        index = shifted._index[name]
        nodes = shifted.fields[index].nodes
        prior = shifted.theta.priors[index]
        mean = np.asarray(prior.prior_mean(nodes))
        sd = np.sqrt(np.diag(np.asarray(prior.prior_cov(nodes))))
        # Same float round-trip bar as test_gp_prior_starts_at_a_plus_minus_a;
        # here it is the read *after* the override, which is what makes it
        # decisive rather than a construction-time coincidence.
        np.testing.assert_allclose(mean, value, rtol=1e-12)  # the mean followed
        # At these amplitudes, jitter/value**2 is ~7e-12 relative on sd -- rtol=1e-9
        # keeps about 270x margin (same LogRBF jitter as the NOTE above).
        np.testing.assert_allclose(sd, value, rtol=1e-9)  # still a +- a


@pytest.mark.parametrize("suite", SUITES)
def test_nuisance_fields_are_not_tied(suite):
    """Structural exclusion (F1): no ``tied_to`` selector in
    ``model.parameter_table()`` starts with ``fields.{nuisance}.``, and no
    ``prior.mean`` selector exists for a nuisance field at all (``priors.Zero``
    carries no ``Parameter``, so there is nothing to tie).  Catches the
    physical-field tie loop in ``build_analysis`` over-matching a nuisance
    selector, or ``_nuisance_prior``'s ``mean=priors.Zero()`` growing a mean
    parameter that then gets swept into a tie by accident.

    No ``_small`` suite's ``config.py`` defines ``nuisance_specs``, so
    ``mode="lattice"`` registers zero nuisance fields there and the per-field
    loop below has nothing to iterate for the ``_small`` half of ``SUITES``.
    Until 2026-08-13 that was a ``pytest.skip``, i.e. those cases
    asserted nothing at all; this repo has no CI, so a skip is total absence,
    not a fallback.  It is now an assertion instead: the presence of nuisance
    *fields* on the compiled model must agree with whether the suite's config
    declares ``nuisance_specs``.  That is a real check in both directions --
    a ``_small`` suite growing a nuisance field without declaring it, or a full
    suite silently losing its nuisance registration (which would make the loop
    below vacuous while the test still reported a pass), now fails here.
    """
    # A lattice-systematic nuisance is zero-mean by design; tying it would give
    # it a non-zero prior mean equal to its amplitude.
    cfg = suite.cfg
    # Not a literal: the _small suites reduced TRUTH_Q_CHOICES to {2, 3} on
    # 2026-08-15, and this test is parametrized across suites with different
    # member tables.  Any member builds the same prior wiring.
    analysis, _ = suite.build_analysis(
        next(iter(suite.cfg.TRUTH_Q_CHOICES)), "lattice"
    )
    model = analysis.compile()
    physical = set(cfg.ALL_FIELDS)
    nuisance = [f.name for f in model.fields if f.name not in physical]
    # Assert, do not skip: the two _small suites declare no nuisance_specs and
    # must therefore register no nuisance fields, and the two full suites must.
    assert bool(nuisance) is hasattr(cfg, "nuisance_specs")
    if not nuisance:
        return  # nothing further to check for this suite; the line above is the check

    tied = {row["selector"] for row in model.parameter_table() if row["tied_to"]}
    for name in nuisance:
        assert not any(s.startswith(f"fields.{name}.") for s in tied)
        # priors.Zero carries no parameter at all, so there is nothing to tie.
        assert not any(
            row["selector"].startswith(f"fields.{name}.prior.mean")
            for row in model.parameter_table()
        )


@pytest.mark.parametrize("suite", SUITES)
def test_prior_stays_fully_fixed(suite):
    """Two literal cfg pins (F1), not a check on any built model.
    ``N_HYPERPARAMS == 0`` follows from ``GP_AMPLITUDE_FREE is False`` by the
    ternary that defines it in config.py, so this really pins that default
    branch, not an independently computed count.  ``FROZEN_COV_PARAMS`` names
    which ``LogRBF`` attributes ``build_analysis``'s freeze loop iterates.
    Complements ``test_build_ties_mean_to_sigma``'s ``model.n_free == 0``, which
    checks the *outcome* on a real compiled model -- this test only checks the
    *declaration*.  Catches a new cov parameter added to a kernel without also
    adding it to ``FROZEN_COV_PARAMS``, or ``GP_AMPLITUDE_FREE`` flipped without
    updating the derived hyperparameter count.
    """
    # The tie must not add a sampled hyper-parameter: a fitted tied amplitude
    # collapses onto its lower bound when the data cannot see the field.
    assert suite.cfg.N_HYPERPARAMS == 0
    assert suite.cfg.FROZEN_COV_PARAMS == ("sigma", "length", "x_reg")


# -- the amplitude floor and hyper-prior, gated on GP_AMPLITUDE_FREE ----------


@pytest.mark.parametrize("suite", SUITES)
def test_amplitude_is_frozen_and_unpenalised_by_default(suite):
    """Structural check (F1) on ``_gp_prior(name).cov.sigma``'s ``Parameter``
    metadata directly (no ``build_analysis`` involved): frozen, no hyper-prior,
    bounds ``(GP_AMPLITUDE_FLOOR, None)``.  The "13.72" in the comment below is
    the same closed-form total independently verified in
    ``test_floating_attaches_the_floor_and_a_centred_hyper_prior`` (computed
    there as 13.720871822209695).  Catches ``_gp_prior`` losing the
    ``frozen=not cfg.GP_AMPLITUDE_FREE`` / ``prior=cfg.gp_amplitude_prior(field)``
    wiring for the default, frozen case.
    """
    cfg = suite.cfg
    assert cfg.GP_AMPLITUDE_FREE is False  # floating is opt-in
    for name in cfg.ALL_FIELDS:
        sigma = suite._gp_prior(name).cov.sigma
        assert sigma.frozen is True
        # A hyper-prior here would add a constant (13.72 over the nine fields) to
        # every -2 log evidence while changing no inference, because
        # neg2_log_prior sums over frozen leaves too.
        assert sigma.prior is None
        # The floor rides along but is inert: bounds_array skips frozen leaves.
        assert sigma.bounds == (cfg.GP_AMPLITUDE_FLOOR, None)


@pytest.mark.parametrize("suite", SUITES)
def test_floor_sits_strictly_below_every_starting_amplitude(suite):
    """Closed-form finiteness (A1): ``inv_softplus(offset) = offset +
    log(-expm1(-offset))`` is finite iff ``offset > 0``, exactly, so the loop
    below is a real check that ``GP_AMPLITUDE_FLOOR`` sits strictly under every
    tier amplitude, not just under the low tier by construction.  Catches
    ``GP_AMPLITUDE_FLOOR`` set at or above ``GP_AMPLITUDE_LOW`` (1.0), which
    would silently send ``eta`` to ``-inf`` for the low tier under the
    sampler's transform (``pixel.infer.hmc.ParameterCoordinates``).  The ``0.9``
    lower bound asserted on ``GP_AMPLITUDE_FLOOR`` itself, below, is an
    unmeasured sanity range rather than a derived bound -- it says "comfortably
    short of 1.0," nothing more.
    """
    cfg = suite.cfg
    # "Essentially forbid a < 1" -- but not exactly 1.0, because GP_AMPLITUDE_LOW
    # *is* 1.0 and a bound sitting on the starting value maps to eta = -inf under
    # the sampler's lower-softplus transform.
    # The 0.9 lower literal is a sanity range, not a derived bound -- see docstring.
    # The property that matters is relative, not a fixed 0.9-1.0 window: the
    # floor must sit strictly under the *smallest* amplitude, and comfortably so
    # rather than on it.  Pinning the old literal range instead would have
    # passed while the low tier was 1.0 and silently mis-stated the invariant
    # once the _small suites dropped fields to 0.2.
    smallest = _min_amplitude(cfg)
    assert 0.9 * smallest <= cfg.GP_AMPLITUDE_FLOOR < smallest
    for name in cfg.ALL_FIELDS:
        assert cfg.gp_amplitude(name) > cfg.GP_AMPLITUDE_FLOOR
        offset = cfg.gp_amplitude(name) - cfg.GP_AMPLITUDE_FLOOR
        eta = offset + np.log(-np.expm1(-offset))  # inv_softplus
        assert np.isfinite(eta)


@pytest.mark.parametrize("suite", SUITES)
def test_floating_the_amplitude_reaches_the_compiled_model(suite, monkeypatch):
    """With ``GP_AMPLITUDE_FREE`` on, the compiled ``Model`` really carries one free
    amplitude per field, still tied to its mean, still floored -- and a real autodiff
    gradient reaches the floated source through the tie.

    ``build_analysis``'s freeze loop contains ``if attr == "sigma" and
    cfg.GP_AMPLITUDE_FREE: continue``, whose own comment says re-freezing there "would
    silently override ``cfg.GP_AMPLITUDE_FREE``" (fit.py:229-235 in the full suites,
    :180-186 in the ``_small`` pair -- the loop is character-identical in all four).
    Nothing exercised it: ``test_floating_attaches_the_floor_and_a_centred_hyper_prior``
    calls ``suite._gp_prior(name)`` directly and never reaches ``build_analysis``, so the
    integration point -- does the *model* leave ``cov.sigma`` free, and does the tie
    survive thawing -- was untested for the closure suites.

    Four checks, increasing in strength:

    * ``model.n_free == len(cfg.ALL_FIELDS)`` and the free selectors are exactly the nine
      ``fields.*.prior.cov.sigma`` (F1).  A ``continue`` dropped from the freeze loop
      gives ``n_free == 0``; a tie dropped gives 18, since ``mean.N`` would join them.
    * every ``mean.N`` still reports ``tied_to`` its own ``cov.sigma`` (F1).
    * ``model.bounds()`` is ``(n, 2)`` with every lower bound at ``GP_AMPLITUDE_FLOOR``
      and every upper at ``inf`` (A1: the floor exists precisely because a fitted tied
      amplitude collapses onto its lower bound when the data cannot see the field, and
      ``bounds_array`` skips *frozen* leaves -- so this is the one configuration in which
      the floor is not inert, and the only one where its absence would be felt).
    * ``jax.grad`` through ``model._rebuild`` -- the same closure ``ravel`` hands the
      sampler under ``jit`` -- of ``sum_i mean.N_i**2`` returns ``2 * sigma_i`` at every
      free coordinate (A1: elementary calculus, and the closure analogue of
      ``tests/test_parameter_ties.py::test_gradient_flows_through_the_tie_to_the_source``,
      which proves the mechanism on a toy).  A tie applied by a non-traced copy would
      break the trace or return zeros; a tie not counted from the target's term would
      return 0 instead of ``2 sigma``.  Measured: ``max|grad / (2 x0) - 1| = 0.0`` exactly,
      on ``x0 = [1, 1, 5, 5, 5, 1, 1, 1, 1]``.

    Note ``cfg.N_HYPERPARAMS`` is *not* asserted against ``n_free`` here: it is evaluated
    at import time from the module-level ``GP_AMPLITUDE_FREE``, so monkeypatching the flag
    cannot move it.  In a real floated run the flag is edited in the file before import
    and the two agree; ``test_prior_stays_fully_fixed`` pins the default branch.
    """
    import jax

    cfg = suite.cfg
    monkeypatch.setattr(cfg, "GP_AMPLITUDE_FREE", True)
    # Not a literal: the _small suites reduced TRUTH_Q_CHOICES to {2, 3} on
    # 2026-08-15, and this test is parametrized across suites with different
    # member tables.  Any member builds the same prior wiring.
    analysis, _ = suite.build_analysis(
        next(iter(suite.cfg.TRUTH_Q_CHOICES)), "lattice"
    )
    model = analysis.compile()
    n_fields = len(cfg.ALL_FIELDS)

    assert model.n_free == n_fields  # one free amplitude per field, and nothing else
    assert len(model.ties) == n_fields  # the tie survives thawing
    free = [row["selector"] for row in model.parameter_table() if not row["frozen"]]
    assert set(free) == {f"fields.{name}.prior.cov.sigma" for name in cfg.ALL_FIELDS}

    rows = {row["selector"]: row for row in model.parameter_table()}
    for name in cfg.ALL_FIELDS:
        assert rows[f"fields.{name}.prior.cov.sigma"]["has_prior"] is True
        assert (
            rows[f"fields.{name}.prior.mean.N"]["tied_to"]
            == f"fields.{name}.prior.cov.sigma"
        )

    # The floor is inert while the amplitude is frozen (bounds_array skips frozen
    # leaves); this is the configuration where it has to be there.
    bounds = np.asarray(model.bounds(), dtype=float)
    assert bounds.shape == (n_fields, 2)
    np.testing.assert_array_equal(bounds[:, 0], cfg.GP_AMPLITUDE_FLOOR)
    assert np.all(np.isinf(bounds[:, 1]))

    # A real gradient, through the same rebuild a fit runs under jit/grad.
    order = list(cfg.ALL_FIELDS)

    def objective(vector):
        theta = model._rebuild(vector)
        return jnp.sum(
            jnp.stack(
                [theta.priors[model._index[name]].mean.N.value ** 2 for name in order]
            )
        )

    start = np.asarray(model._x0, dtype=float)
    gradient = np.asarray(jax.grad(objective)(model._x0), dtype=float)
    # d/dsigma of sum(mean.N**2) with mean.N tied to sigma is 2*sigma; the free
    # coordinates are the amplitudes themselves.  Measured max|a/b - 1| = 0.0.
    np.testing.assert_allclose(gradient, 2.0 * start, rtol=1e-12, atol=0.0)
    assert np.all(gradient > 0.0)  # a dropped tie gives exactly zero here


@pytest.mark.parametrize("suite", SUITES)
def test_floating_attaches_the_floor_and_a_centred_hyper_prior(suite, monkeypatch):
    """Three checks at increasing strength.  Structural (F1): a floated
    ``sigma`` is not frozen, keeps its floor, and gains a hyper-prior.  Shape
    (A2): the hyper-prior's ``neg2_logpdf`` at the field's own tier value is
    strictly below its value at 10x or 0.1x -- true for any centred LogNormal
    at this sigma regardless of which tier (z = +-log(10)/0.5 = +-4.6 either
    way), so this mainly confirms ``log_center`` really is
    ``math.log(gp_amplitude(field))`` and not a constant shared by every tier.
    Closed-form (A1): ``total == pytest.approx(13.7209, rel=1e-4)`` is
    ``sum 2*log(a) + log(2*pi) + 2*log(sigma)`` over 3 fields at ``a=5`` and 6
    at ``a=1``, ``sigma=0.5`` (the ``z=0`` case of ``LogNormal.neg2_logpdf``) --
    verified independently against ``pixel.core.params.LogNormal.neg2_logpdf``
    directly: computed total 13.720871822209695, 2.05e-06 relative from the
    literal below, comfortably inside the ``rel=1e-4`` bar (measured via a
    standalone script, not committed).  Catches ``log_center`` hardcoded to one
    tier instead of derived per-field, or ``GP_AMPLITUDE_PRIOR_SIGMA`` changed
    without updating this literal.
    """
    cfg = suite.cfg
    monkeypatch.setattr(cfg, "GP_AMPLITUDE_FREE", True)
    total = 0.0
    for name in cfg.ALL_FIELDS:
        sigma = suite._gp_prior(name).cov.sigma
        a = cfg.gp_amplitude(name)
        assert sigma.frozen is False  # now a sampled hyper-parameter
        assert sigma.bounds == (cfg.GP_AMPLITUDE_FLOOR, None)
        assert sigma.prior is not None
        # Centred on its own tier value, so thawing does not yank the high tier
        # (a = 5) down toward the low one (a = 1).
        at_centre = float(sigma.prior.neg2_logpdf(jnp.asarray(a)))
        assert at_centre < float(sigma.prior.neg2_logpdf(jnp.asarray(10.0 * a)))
        assert at_centre < float(sigma.prior.neg2_logpdf(jnp.asarray(0.1 * a)))
        total += at_centre
    # Closed form at each field's own centre (z=0):
    #     2*log(a) + log(2*pi) + 2*log(sigma)
    # summed over the nine fields.  Evaluated from the suite's own amplitudes
    # rather than a literal, because the _small suites moved to a per-field
    # table (2026-08-15) and a hardcoded total is only correct for whichever
    # tier layout it was written against -- the old 13.7209 was "3 at a=5 plus
    # 6 at a=1" and silently stops being the closed form the moment an
    # amplitude moves.  The oracle is still independent: it is the analytic
    # LogNormal expression, not anything read back from the object under test.
    expected = sum(
        2.0 * math.log(cfg.gp_amplitude(name))
        + math.log(2.0 * math.pi)
        + 2.0 * math.log(cfg.GP_AMPLITUDE_PRIOR_SIGMA)
        for name in cfg.ALL_FIELDS
    )
    assert total == pytest.approx(expected, rel=1e-4)
