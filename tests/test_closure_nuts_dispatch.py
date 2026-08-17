"""Sampler-dispatch policy of the closure-suite drivers.

Exercises the private dispatch helpers of ``closure_NNPDF_truth.fit`` and
``closure_NNPDF_truth_small.fit`` (aliased below as ``full_nnpdf`` and
``small_nnpdf``), plus the ``main`` /
``build_kernels_only`` entry points of their sibling ``run_closure``
modules. None of these live under ``src/pixel`` -- this file imports nothing
from ``pixel`` directly -- but every driver it drives does (``import pixel``,
``pixel.infer``, ``pixel.map``), so a dispatch bug here changes which real
``pixel`` sampler a closure fit reaches, not that sampler's own numerics.

The ``closure_JAM_truth`` and ``closure_JAM_truth_small`` drivers were
parametrized alongside these until 2026-08-14; see the note on ``SUITES``
below for why they came out.

Two dispatch shapes are covered here, and they are genuinely different:

* ``full_nnpdf`` (``FULL_SUITES``) -- ``_closure_sampler(model)`` inspects
  ``model.has_nonlinear`` / ``model.has_bilinear`` /
  ``model.contour_partition()`` and returns one of ``"direct_nuts"``,
  ``"hmc"``, ``"nuts"``. Real, three-way branch dispatch.
* ``small_nnpdf`` (``SMALL_SUITES``) -- ``_closure_sampler(model, mode=None)``
  unconditionally ``return "vegas"``; it reads neither argument (confirmed
  directly: ``suite._closure_sampler(None, None) == "vegas"``). The actual
  per-mode VEGAS configuration lives in ``run_fit``'s ``sampler_options``
  construction, **not** in the dispatcher's return value.

Because of that second point, every ``_closure_sampler(...) == "vegas"``
assertion is true by construction, and until 2026-08-13 the small-suite claims
rested entirely on ``inspect.getsource`` substrings. They now rest on
``test_small_run_fit_configures_vegas_from_the_model_not_the_mode`` and
``test_full_run_fit_routes_each_model_to_its_own_sampler_options``, which drive
``run_fit`` for real up to its inference call and assert on the arguments that
arrive there. ``test_small_mixed_mode_samples_explicit_normalizations_with_
joint_vegas`` was removed in the same pass: both its ``_closure_sampler`` lines
were tautologies and both its source substrings were already asserted verbatim
by ``test_small_bilinear_closures_configure_joint_affine_vegas`` on the same
source object, so it added nothing that survived either check -- the modes it
named ("exp"/"both") are now driven for real by the first of those two tests.

None of the oracles here are physics oracles; there is no accuracy floor to
state. Four kinds of check recur:

1. branch dispatch on a hand-built ``SimpleNamespace`` standing in for
   ``Model``, checked against the real conditional read out of the source;
2. literal substrings of ``inspect.getsource(suite.run_fit)`` /
   ``inspect.getsource(runner.main)`` -- these pin the *text* of a call site,
   so they catch a changed literal or a deleted call, but cannot tell live
   code from the same text sitting in a comment or a dead branch;
3. direct reads of ``suite.cfg.<NAME>`` against ``closure_*_truth[_small]/
   config.py``, which is where any recorded provenance for those numbers
   lives -- this file repeats it inline only where a test's own docstring
   says so;
4. driving ``run_fit`` with its collaborators stubbed until it calls the
   sampler, then asserting on the real call (``_capture_sampler_call``). This
   is the only kind here that distinguishes live code from matching text.

The one test with a genuine numeric bar tied to an analytic property (AR(1)
autocorrelation inflation) is ``test_autocorrelation_ess_falls_below_the_
sample_count_when_mixing_is_slow``, below; the Wolff estimator it calls is
pinned against independent references in ``tests/test_gamma_method.py``,
which this file does not duplicate. The production-scale version of every
dispatch path here is the full closure run itself (``python -m
closure_JAM_truth.run_closure``), which this file does not execute.
"""

import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from closure_NNPDF_truth import fit as full_nnpdf
from closure_NNPDF_truth_small import fit as small_nnpdf
from closure_NNPDF_truth import run_closure as full_nnpdf_runner
from closure_NNPDF_truth_small import run_closure as small_nnpdf_runner


# Full vs. small is the axis that matters here: they differ in sampler family
# (NUTS/HMC vs. VEGAS) and in the whole shape of ``run_fit``'s ``sampler_options``.
#
# JAM vs. NNPDF is not such an axis, and stopped being parametrized on 2026-08-14
# (owner's instruction).  ``diff closure_JAM_truth/fit.py closure_NNPDF_truth/fit.py``
# shows two docstring lines and nothing executable, and the same holds for the
# ``_small`` pair and for the four ``run_closure.py`` copies -- so every JAM case in
# this file re-ran an identical dispatch branch, an identical ``inspect.getsource``
# substring match, or an identical stubbed ``run_fit`` call, for an identical answer.
# Nothing here reads a truth PDF set; no test here compares one suite to another.
#
#: Both sizes -- what a test asserts for every closure driver.
SUITES = (full_nnpdf, small_nnpdf)
#: Full size only -- NUTS/HMC dispatch, which the ``_small`` drivers do not have.
FULL_SUITES = (full_nnpdf,)
#: Small size only -- the VEGAS path.
SMALL_SUITES = (small_nnpdf,)
#: The matching ``run_closure`` modules, for the two runner-level tests below.
RUNNERS = (full_nnpdf_runner, small_nnpdf_runner)


@pytest.mark.parametrize("suite", FULL_SUITES)
def test_bilinear_closures_use_public_nuts_family(suite):
    """A pure H-S model (bilinear, zero ordinary coordinates) gets plain NUTS.

    ``_closure_sampler`` is ``if has_nonlinear: direct_nuts; elif has_bilinear
    and n_ordinary: hmc; else: nuts`` -- read directly out of
    ``closure_NNPDF_truth/fit.py`` (and its byte-identical JAM twin).
    ``n_ordinary=0`` makes the middle branch
    false, so this exercises the fallthrough specifically, paired against
    ``test_full_mixed_linear_normalization_and_hs_closures_use_joint_hmc``
    below, which flips only ``n_ordinary`` and lands on ``"hmc"`` instead.
    Would catch: an inverted or dropped ``n_ordinary`` truthiness check, or a
    fallthrough that returns ``"hmc"``/``"direct_nuts"`` regardless of input.
    Structural dispatch only; no numerics.
    """
    model = SimpleNamespace(
        has_bilinear=True,
        contour_partition=lambda: SimpleNamespace(n_ordinary=0),
    )
    assert suite._closure_sampler(model) == "nuts"


@pytest.mark.parametrize("suite", FULL_SUITES)
def test_exact_ratio_closures_use_explicit_direct_nuts(suite):
    """``has_nonlinear`` models dispatch to the explicit retained-(theta,q) path.

    ``model`` here has no ``contour_partition`` attribute at all, so if
    dispatch order were wrong and ``has_bilinear`` were checked before
    ``has_nonlinear``, ``_closure_sampler`` would raise ``AttributeError``
    (``SimpleNamespace`` has no such method) rather than silently returning
    the wrong string -- this fixture proves the nonlinear check runs first
    and alone. ``DIRECT_PRIOR_RCOND`` is pinned exactly rather than bounded:
    the previous ``<= 1.0e-10`` sat *on* the value it bounded, so it could
    catch only a loosening and not drift to a different tight value -- and the
    cutoff is load-bearing (``config.py``: "physics promotion additionally
    requires the cutoff sweep"), because the near-hard closure constraints are
    very precise pseudo-data that a larger rcond would truncate out of the
    prior. The ``"run_direct_mcmc" in source`` line is a
    text-substring pin on ``run_fit`` (oracle F2): it catches the call being
    deleted or renamed, not a wrong argument to it -- for the arguments see
    ``test_full_run_fit_routes_each_model_to_its_own_sampler_options``.
    """
    model = SimpleNamespace(has_nonlinear=True, has_bilinear=True)
    assert suite._closure_sampler(model) == "direct_nuts"
    assert suite.cfg.DIRECT_PRIOR_RCOND == pytest.approx(1.0e-10, rel=1e-12, abs=0.0)
    source = inspect.getsource(suite.run_fit)
    assert "run_direct_mcmc" in source


@pytest.mark.parametrize("suite", FULL_SUITES)
def test_full_mixed_linear_normalization_and_hs_closures_use_joint_hmc(suite):
    """Bilinear *plus* ordinary coordinates (``n_ordinary=5``) route to joint HMC.

    Complementary fixture to ``test_bilinear_closures_use_public_nuts_family``
    above: same ``has_bilinear=True``, only ``n_ordinary`` flips (0 -> 5), and
    the dispatch result flips with it (``"nuts"`` -> ``"hmc"``). Together the
    pair exercises both sides of ``model.has_bilinear and
    model.contour_partition().n_ordinary`` in ``_closure_sampler``. Would
    catch a truthiness check that ignores ``n_ordinary``'s value entirely.
    Structural dispatch only.
    """
    model = SimpleNamespace(
        has_bilinear=True,
        contour_partition=lambda: SimpleNamespace(n_ordinary=5),
    )
    assert suite._closure_sampler(model) == "hmc"


@pytest.mark.parametrize("suite", SMALL_SUITES)
def test_small_dy_mode_uses_joint_affine_vegas(suite):
    """The small dispatcher is constant by design, and the VEGAS constants hold.

    Two claims, kept apart because only one of them is about dispatch.

    First, ``_closure_sampler(model, mode=None)`` is ``return "vegas"``
    unconditionally -- it reads neither argument.  That is asserted here the
    only way it can be asserted honestly: against deliberately invalid input
    (``None`` model, a mode string no suite defines).  Before 2026-08-13 this
    test instead called it with an elaborate ``model`` mock and a real mode
    string, which read as a dispatch check but could not fail for any input;
    the mock (``contour_partition``, ``_bilinear_normalization_design``) was
    never consulted by anything the test called.  Written this way the line
    still fails if the small dispatcher ever *becomes* mode-dependent, which is
    the point at which every small-suite test here needs revisiting.

    Second, the four ``cfg.VEGAS_*`` literals (oracle F2, config-drift pins).
    They confirm the constants have these values, not that DY mode wires them
    into ``run_fit`` -- that is
    ``test_small_run_fit_configures_vegas_from_the_model_not_the_mode``, which
    drives ``run_fit`` to its real ``infer_mcmc`` call for all three modes.
    """
    # Intentionally invalid input: the contract being pinned is that the
    # answer does not depend on it.
    assert suite._closure_sampler(None, None) == "vegas"
    assert suite._closure_sampler(None, "not-a-real-mode") == "vegas"
    assert suite.cfg.VEGAS_N_ADAPT_INNER_SAMPLES == 128  # pilot inner draws/outer point
    assert suite.cfg.VEGAS_N_INNER_SAMPLES == 1  # production inner draws/outer point
    assert suite.cfg.VEGAS_JOINT_GRID is True  # one adapted grid shared by outer+inner
    assert suite.cfg.VEGAS_MIN_SIGNED_ESS_FRAC == pytest.approx(
        0.05, rel=1e-12, abs=0.0
    )  # ESS floor


@pytest.mark.parametrize("suite", FULL_SUITES)
def test_full_linear_closures_use_nuts_without_a_contour(suite):
    """A model with no bilinear sector at all dispatches to ordinary NUTS.

    ``model`` has no ``contour_partition`` attribute, so -- as in
    ``test_exact_ratio_closures_use_explicit_direct_nuts`` above -- a
    dispatch-order bug that touched ``model.contour_partition()`` before
    checking ``has_bilinear`` would raise ``AttributeError`` here rather than
    silently misrouting. The five ``cfg.NUTS_*``/``MCMC_SAMPLES`` lines pin
    literal values read from ``config.py`` (100 / 0.1 / "leapfrog" / 6 /
    False / 1000); only ``NUTS_WARMUP`` gets a loose ``> 0`` bound instead of
    an exact pin, so it catches zero/negative warmup but not a value drifting
    away from 100. These are config-drift pins (oracle F2), not physics.
    """
    model = SimpleNamespace(has_bilinear=False)
    assert suite._closure_sampler(model) == "nuts"
    assert suite.cfg.NUTS_WARMUP > 0  # loose: nonzero warmup, not pinned to 100
    assert suite.cfg.NUTS_INITIAL_STEP_SIZE == pytest.approx(0.1)
    assert suite.cfg.NUTS_INTEGRATOR == "leapfrog"
    assert suite.cfg.NUTS_MAX_TREE_DEPTH == 6
    assert suite.cfg.NUTS_ADAPT_MASS is False  # Hessian mass is fixed; no re-adaptation
    assert suite.cfg.MCMC_SAMPLES == 1000


@pytest.mark.parametrize("suite", SMALL_SUITES)
def test_small_linear_closures_use_ordinary_vegas(suite):
    """Small-suite ordinary-grid VEGAS tuning: bins, adaptation, samples.

    As in the other small-suite dispatch tests, ``_closure_sampler(model) ==
    "vegas"`` cannot fail -- the function ignores ``model`` entirely for this
    suite (``return "vegas"`` unconditionally; see the module docstring). The
    ``model`` fixture here is at least minimal (no unused mock attributes),
    unlike the "dy"/"mixed" tests above. The four ``cfg.VEGAS_*``/
    ``MCMC_SAMPLES`` lines are real literal pins against
    ``closure_NNPDF_truth_small/config.py`` (16 / 5 / 128 / 1000; the JAM
    ``_small`` copy carries the same four values); they do not
    establish that a linear model actually reaches ordinary (non-nested)
    VEGAS at runtime, only that these constants have these values.
    """
    model = SimpleNamespace(has_bilinear=False)
    assert suite._closure_sampler(model) == "vegas"  # unconditional; see docstring
    assert suite.cfg.VEGAS_N_BINS == 16  # per-dimension adaptive-grid resolution
    assert suite.cfg.VEGAS_N_ADAPT_ITERATIONS == 5  # grid-refinement passes/production
    assert suite.cfg.VEGAS_N_EVAL_PER_ITERATION == 128  # evaluations/refinement pass
    assert suite.cfg.MCMC_SAMPLES == 1000


@pytest.mark.parametrize("suite", SUITES)
def test_contour_saddle_tolerance_covers_observed_float_stall(suite):
    """The saddle-acceptance tolerance clears the documented float-stall floor.

    ``config.py`` records the provenance directly: "Some closure likelihoods
    stall just below a 3e-3 Newton decrement at floating-point precision.
    Keep a small margin so a numerically stationary H-S saddle is accepted
    instead of aborting contour-NUTS initialization." The lower bound here
    (``3.0e-3``) is that documented floor; the upper bound (``1.0e-2``) caps
    looseness so a non-converged saddle cannot be silently waved through.
    Both suites currently set ``CONTOUR_SADDLE_TOL = 5.0e-3``, inside the
    range. This test does not itself re-measure the "stalls at 3e-3" claim --
    it only checks the constant sits in a range whose floor matches the
    comment; if that comment's number were wrong, this would not know.
    Oracle D1 (provenance recorded, but in ``config.py``, not here).
    """
    assert 3.0e-3 <= suite.cfg.CONTOUR_SADDLE_TOL <= 1.0e-2  # floor = stall point


@pytest.mark.parametrize("suite", FULL_SUITES)
def test_full_bilinear_closures_enable_only_the_two_setup_caches(suite):
    """Contour NUTS caches only the saddle reference and Hessian mass, nothing else.

    All checks read ``inspect.getsource(suite.run_fit)`` as text (oracle F2):
    a substring match proves the text is present somewhere in the function
    body, not that it executes on the bilinear branch specifically, and would
    not notice the same text moved into a comment or a dead branch. The two
    negative checks (``"chain_cache"``/``"summary_cache" not in source``) are
    a regression guard against reintroducing cache kinds this suite
    deliberately does not use; they would also pass if those strings simply
    never existed for unrelated reasons. ``INFERENCE_CACHE_ROOT.name`` is a
    direct, non-textual attribute read (not a source-substring check).
    """
    source = inspect.getsource(suite.run_fit)
    assert suite.cfg.INFERENCE_CACHE_ROOT.name == "_inference_cache"
    assert '"step_size": cfg.NUTS_INITIAL_STEP_SIZE' in source
    assert '"reference_cache_path"' in source
    assert '"hessian_cache_path"' in source
    assert "chain_cache" not in source
    assert "summary_cache" not in source


@pytest.mark.parametrize("suite", SMALL_SUITES)
def test_small_bilinear_closures_configure_joint_affine_vegas(suite):
    """``run_fit``'s bilinear branch builds nested-VEGAS options, not MCMC ones.

    This is the test that actually establishes the small-suite "bilinear ->
    joint affine VEGAS" claim: unlike the ``_closure_sampler``-based tests
    above, every check here reads real content out of
    ``inspect.getsource(suite.run_fit)`` (oracle F2, substring-in-text, so it
    cannot tell live code from the same text in a comment or dead branch, but
    there is no such branch here to confuse it with). The seven positive
    substrings together pin the ``if model.has_bilinear:`` block's option
    dict; ``'"n_warmup"' not in source`` is a negative check that this VEGAS
    path never grew an MCMC-style warmup option. ``VEGAS_JOINT_COVARIANCE_INFLATION
    == pytest.approx(1.0)`` is a direct (non-textual) literal read from
    ``config.py``. Would catch: a key renamed or dropped from
    ``sampler_options`` in the bilinear branch, or an MCMC-only option
    leaking into it.
    """
    source = inspect.getsource(suite.run_fit)
    assert '"n_bins": cfg.VEGAS_N_BINS' in source
    assert '"n_inner_samples"' in source
    assert 'auxiliary = "marginalize"' in source
    assert '"joint_grid"' in source
    assert '"saddle_tol"' in source
    assert '"min_adapt_phase_ess_frac"' in source
    assert '"max_extra_adapt_iterations"' in source
    assert suite.cfg.VEGAS_JOINT_COVARIANCE_INFLATION == pytest.approx(1.0)
    assert '"n_warmup"' not in source


# -- what actually reaches the sampler ---------------------------------------


class _SamplerCalled(Exception):
    """Sentinel: ``run_fit`` reached inference; everything after it is not under test."""


def _capture_sampler_call(suite, monkeypatch, model, mode="both"):
    """Drive ``run_fit`` as far as its inference call and return that call.

    Stubs the four collaborators between the entry point and the sampler
    (``build_analysis``, ``_iterate_t0``, and both inference entry points), then
    stops at the sampler with a sentinel exception -- the posterior
    marginalization after it needs a real ``Model`` and is not what this
    measures.  ``options`` holds the sampler options only; the three run-level
    kwargs every branch passes identically are split into ``run``.
    """
    captured = {}
    RUN_KWARGS = ("n_samples", "run_map", "seed")

    def fake_build(q_key, mode, *, use_kernel_cache=True, t0=None):
        return SimpleNamespace(compile=lambda: model), {}

    def record(entry):
        def inner(model_, *, sampler=None, auxiliary=None, **kwargs):
            captured.update(
                entry=entry, sampler=sampler, auxiliary=auxiliary,
                run={k: kwargs.pop(k, None) for k in RUN_KWARGS},
                options=kwargs,
            )
            raise _SamplerCalled

        return inner

    monkeypatch.setattr(suite, "build_analysis", fake_build)
    monkeypatch.setattr(suite, "_iterate_t0", lambda *a, **k: ({}, 0))
    monkeypatch.setattr(suite, "infer_mcmc", record("infer_mcmc"))
    monkeypatch.setattr(suite.pixel, "run_direct_mcmc", record("run_direct_mcmc"))
    with pytest.raises(_SamplerCalled):
        suite.run_fit(
            next(iter(suite.cfg.TRUTH_Q_CHOICES)), mode,
            use_kernel_cache=False, n_samples=4,
        )
    return captured


def _fake_model(**kwargs):
    """The attributes ``run_fit`` reads before it reaches the sampler."""
    return SimpleNamespace(
        datasets=[], free_vector=lambda: np.zeros(3), **kwargs
    )


@pytest.mark.parametrize("suite", SMALL_SUITES)
@pytest.mark.parametrize("mode", ("dy", "exp", "both"))
def test_small_run_fit_configures_vegas_from_the_model_not_the_mode(suite, mode, monkeypatch):
    """Nested VEGAS options appear for a bilinear model and for no other reason.

    The small suite's ``_closure_sampler(model, mode)`` is ``return "vegas"``
    unconditionally -- it reads neither argument -- so every
    ``_closure_sampler(...) == "vegas"`` assertion in this file is true by
    construction and the per-mode claims in those tests' names rest on
    ``inspect.getsource`` substrings, which cannot tell live code from the same
    text in a comment.  This drives ``run_fit`` for real instead, to the point
    where it calls ``infer_mcmc``, and asserts on the arguments that actually
    arrive: sampler name, ``auxiliary``, and the option dict.

    That makes the claim falsifiable in the way the source-text pins are not --
    the nested-VEGAS keys must be present for a bilinear model and *absent* for
    a linear one, and ``auxiliary`` must flip ``"marginalize"``/``"auto"`` with
    it.  Running all three modes shows the same thing the docstrings assert in
    prose: this dispatch is a function of ``model.has_bilinear`` alone, and no
    mode changes it.  Oracle F1 (structural, on the real call); the VEGAS
    numerics themselves are ``pixel.infer``'s and are not touched here.
    """
    cfg = suite.cfg
    ordinary_keys = {
        "n_bins", "n_adapt_iterations", "n_eval_per_iteration", "alpha",
        "covariance_inflation",
    }
    nested_keys = {
        "n_inner_samples", "n_adapt_inner_samples", "joint_grid", "saddle_tol",
        "min_adapt_phase_ess_frac", "max_extra_adapt_iterations",
    }

    call = _capture_sampler_call(suite, monkeypatch, _fake_model(has_bilinear=True), mode)
    assert (call["entry"], call["sampler"]) == ("infer_mcmc", "vegas")
    assert call["auxiliary"] == "marginalize"  # the H-S sector is integrated out
    assert call["run"] == {"n_samples": 4, "run_map": False, "seed": cfg.MCMC_SEED}
    assert set(call["options"]) == ordinary_keys | nested_keys
    assert call["options"]["n_inner_samples"] == cfg.VEGAS_N_INNER_SAMPLES
    assert call["options"]["n_adapt_inner_samples"] == cfg.VEGAS_N_ADAPT_INNER_SAMPLES
    assert call["options"]["joint_grid"] is cfg.VEGAS_JOINT_GRID
    assert call["options"]["covariance_inflation"] == pytest.approx(
        cfg.VEGAS_JOINT_COVARIANCE_INFLATION, rel=1e-12, abs=0.0
    )
    assert "n_warmup" not in call["options"]  # no MCMC option leaks into VEGAS

    call = _capture_sampler_call(suite, monkeypatch, _fake_model(has_bilinear=False), mode)
    assert (call["entry"], call["sampler"]) == ("infer_mcmc", "vegas")
    assert call["auxiliary"] == "auto"
    assert set(call["options"]) == ordinary_keys  # ordinary grid, no inner loop
    assert call["options"]["n_bins"] == cfg.VEGAS_N_BINS
    assert call["options"]["n_adapt_iterations"] == cfg.VEGAS_N_ADAPT_ITERATIONS
    assert call["options"]["n_eval_per_iteration"] == cfg.VEGAS_N_EVAL_PER_ITERATION


# (model kwargs, expected sampler, expected inference entry point).
FULL_DISPATCH_CASES = (
    ({"has_nonlinear": False, "has_bilinear": False}, "nuts", "infer_mcmc"),
    (
        {"has_nonlinear": False, "has_bilinear": True,
         "contour_partition": lambda: SimpleNamespace(n_ordinary=0)},
        "nuts", "infer_mcmc",
    ),
    (
        {"has_nonlinear": False, "has_bilinear": True,
         "contour_partition": lambda: SimpleNamespace(n_ordinary=5)},
        "hmc", "infer_mcmc",
    ),
    (
        {"has_nonlinear": True, "has_bilinear": True},
        "direct_nuts", "run_direct_mcmc",
    ),
)


@pytest.mark.parametrize("suite", FULL_SUITES)
@pytest.mark.parametrize(("model_kwargs", "sampler", "entry"), FULL_DISPATCH_CASES)
def test_full_run_fit_routes_each_model_to_its_own_sampler_options(
    suite, model_kwargs, sampler, entry, monkeypatch
):
    """Each dispatch branch reaches inference with its own option block.

    The runtime counterpart to the ``_closure_sampler``-only tests above: those
    stop at the returned string, and the option dicts beside them are pinned as
    ``inspect.getsource`` substrings (oracle F2), which pass on text sitting in
    a comment or a dead branch.  Here ``run_fit`` runs until it calls the
    sampler and the real call is inspected -- including *which* entry point it
    used, since ``direct_nuts`` alone goes through ``pixel.run_direct_mcmc``
    rather than ``infer_mcmc``, a split no source-substring check in this file
    covers.

    The two caches are asserted as the caller sees them: exactly
    ``reference_cache_path`` and ``hessian_cache_path``, only on the bilinear
    NUTS branch, under ``cfg.INFERENCE_CACHE_ROOT``.  Would catch: an option
    block wired to the wrong branch, a cache path leaking onto the linear or
    HMC branch, ``auxiliary`` not following ``has_bilinear``, or the
    ``direct_nuts`` call going through the ordinary entry point.  Oracle F1.
    """
    cfg = suite.cfg
    model = _fake_model(**model_kwargs)
    call = _capture_sampler_call(suite, monkeypatch, model, "both")
    assert call["entry"] == entry
    options = call["options"]
    if entry == "run_direct_mcmc":
        # A retained-(theta, q) run is a different API: sampler is pinned to
        # "nuts" there and the branch is identified by the entry point.
        assert call["sampler"] == "nuts"
        assert options["prior_rcond"] == cfg.DIRECT_PRIOR_RCOND
        assert options["n_warmup"] == cfg.DIRECT_NUTS_WARMUP
        assert "use_hessian_mass" not in options  # no contour to build a mass on
        return
    assert call["sampler"] == sampler
    assert call["auxiliary"] == ("sample" if model_kwargs["has_bilinear"] else "auto")
    assert options["use_hessian_mass"] is True
    caches = {"reference_cache_path", "hessian_cache_path"}
    if sampler == "hmc":
        assert options["n_leapfrog"] == cfg.HMC_N_LEAPFROG
        assert options["saddle_tol"] == cfg.CONTOUR_SADDLE_TOL
        assert not caches & set(options)  # the caches are NUTS-branch only
    elif model_kwargs["has_bilinear"]:
        assert options["max_tree_depth"] == cfg.NUTS_MAX_TREE_DEPTH
        assert options["saddle_tol"] == cfg.CONTOUR_SADDLE_TOL
        assert caches <= set(options)
        for key in caches:
            assert options[key].parent == cfg.INFERENCE_CACHE_ROOT
    else:
        assert not caches & set(options)
        assert "saddle_tol" not in options  # no contour without a bilinear sector


@pytest.mark.parametrize("suite", SUITES)
def test_no_dis_t0_datasets_skip_map(suite, monkeypatch):
    """DY-only/constraint-only layouts must never enter bilinear MAP.

    ``_dis_predictions`` collects datasets whose
    ``metadata.extras["correlated_systematics"]`` is truthy; here the one
    dataset has ``extras={}``, so that collection is empty and the function
    returns ``{}`` before ever calling ``pixel.map``. The trap is
    ``monkeypatch.setattr(suite.pixel, "map", unexpected_map)``: if the
    skip-when-empty guard were dropped, ``pixel.map`` would be called, the
    monkeypatched replacement would raise ``AssertionError``, and this test
    would fail loudly rather than merely returning a coincidentally-empty
    dict. Would catch: the ``if not layouts: return {}`` short circuit being
    removed or its condition inverted. Structural / call-avoidance only
    (oracle F1); no MAP numerics are exercised.
    """
    dataset = SimpleNamespace(
        metadata=SimpleNamespace(extras={}),
    )
    layout = SimpleNamespace(dataset_index=0)
    model = SimpleNamespace(
        datasets=[dataset],
        _layout=SimpleNamespace(datasets=[layout]),
    )

    def unexpected_map(*args, **kwargs):
        raise AssertionError("MAP should not run without DIS t0 datasets")

    monkeypatch.setattr(suite.pixel, "map", unexpected_map)
    assert suite._dis_predictions(model) == {}


@pytest.mark.parametrize("suite", SUITES)
def test_dis_t0_predictions_are_sliced_per_dataset(suite, monkeypatch):
    """The other branch: MAP runs once and each table gets its own slice.

    ``test_no_dis_t0_datasets_skip_map`` above drives only the skip branch, so
    the half of ``_dis_predictions`` that does the work
    (``closure_JAM_truth/fit.py:433-457``, byte-identical in the NNPDF twin this
    test now runs against) -- call ``pixel.map`` once, evaluate
    the posterior predictive there, hand each dataset its own
    ``layout.data_slice`` -- had no test at all.  That slicing is what defines
    each DIS table's t0 reference, and a wrong slice does not raise: it
    silently references one table's multiplicative uncertainties to another
    table's theory, which is a biased covariance, not a crash.

    ``pixel.map`` and ``posterior_predictive`` are stubbed, so the oracle is
    structural (F1): a prediction vector whose entries are their own index,
    which makes each expected slice readable and distinct.  The middle dataset
    carries no ``correlated_systematics`` and must be absent from the result
    *and* absent from the slicing -- a filter applied to the datasets but not
    to the layouts would shift every slice by one and is caught here.  MAP is
    asserted to run exactly once for the whole model, not once per table.
    """
    names = ("dis_a", "lattice_b", "dis_c")
    extras = ({"correlated_systematics": [1.0]}, {}, {"correlated_systematics": [2.0]})
    slices = (slice(0, 2), slice(2, 5), slice(5, 9))
    model = SimpleNamespace(
        datasets=[
            SimpleNamespace(name=n, metadata=SimpleNamespace(extras=e))
            for n, e in zip(names, extras)
        ],
        _layout=SimpleNamespace(datasets=[
            SimpleNamespace(dataset_index=i, data_slice=s)
            for i, s in enumerate(slices)
        ]),
        posterior_predictive=lambda vec: (np.arange(9.0), None, None),
    )
    map_calls = []

    def fake_map(m, **kwargs):
        map_calls.append(kwargs)
        return SimpleNamespace(x=np.zeros(3))

    monkeypatch.setattr(suite.pixel, "map", fake_map)
    out = suite._dis_predictions(model)

    assert len(map_calls) == 1  # one MAP solve for the model, not one per table
    assert map_calls[0]["method"] == suite.cfg.MAP_METHOD
    assert set(out) == {"dis_a", "dis_c"}  # the lattice table has no t0 reference
    np.testing.assert_array_equal(out["dis_a"], [0.0, 1.0])
    np.testing.assert_array_equal(out["dis_c"], [5.0, 6.0, 7.0, 8.0])


# -- autocorrelation ESS policy ----------------------------------------------


@pytest.mark.parametrize("suite", SUITES)
def test_autocorrelation_ess_routes_through_the_real_markov_history(suite):
    """Contour artifacts must be analyzed via ``.chain``, not their complex ``f``.

    ``ContourHMCSamples.samples`` is the *complex* flowed auxiliary field; the
    trajectory that was actually integrated is the real chain underneath it.
    Both assertions below are load-bearing, not coincidental attribute reads:
    a ``_markov_chain`` that inspected ``contour`` directly instead of
    preferring ``.chain`` would return ``None`` here (its own complex/weights
    guard rejects ``contour.samples``), which fails the ``is chain`` identity
    check -- verified directly by substituting such a mutant in isolation.
    And ``contour`` itself carries no ``param_labels``, so any path that
    skipped ``.chain`` and still somehow returned real-shaped data would
    report a ``"p0"``/``"p1"`` fallback label, not ``"t0"``/``"t1"`` -- so
    ``slowest`` cannot pass by accident either.

    What this does *not* establish: ``_markov_chain``'s own
    ``np.iscomplexobj``/weights guard is never exercised on a *chainless*
    complex object by this fixture, because ``contour.chain`` is always
    present, so ``candidate`` is always ``chain`` before that guard would
    matter -- substituting a version of ``_markov_chain`` with the guard
    deleted entirely returns the identical result on this input (verified
    directly). No sampler type in ``pixel.infer`` currently reaches
    ``_markov_chain`` with complex ``.samples`` and no ``.chain`` (every
    contour-family dataclass carries both), so that branch currently guards
    against a case with no known caller. Nor is "silently meaningless" quite
    right for what happens if the guard *were* bypassed today: handing the
    complex array straight to ``pixel.infer.gamma_method.autocorrelation_summary``
    raises ``ValueError`` there instead (an independent ``np.iscomplexobj``
    check inside ``_resolve_chain``, verified directly) -- the failure this
    guards against is now doubly caught and would be loud either way.
    """
    real = np.random.default_rng(0).standard_normal((600, 2))
    chain = SimpleNamespace(
        samples=real, param_labels=("t0", "t1"), weights=np.full(600, 1.0 / 600)
    )
    contour = SimpleNamespace(
        samples=real[:, :1] + 1j * real[:, 1:],  # complex: must not be used
        ess_frac=0.5,
        n_samples=600,
        chain=chain,
    )
    assert suite._markov_chain(contour) is chain

    report = suite._autocorrelation_ess(contour)
    assert report["ess"] > 0.0  # finiteness/positivity only, not a magnitude check
    assert report["slowest"] in ("t0", "t1")  # labels come from the real chain


@pytest.mark.parametrize("suite", SUITES)
def test_autocorrelation_ess_declines_non_markov_inputs(suite):
    """Importance draws and fixed posteriors have no autocorrelation to report.

    Reporting ``tau_int`` for independent weighted draws would be meaningless;
    the weight/phase ESS beside it is already the correct measure there. Two
    distinct guard clauses in ``_markov_chain`` are exercised: ``weighted``
    has non-uniform ``weights`` and no ``.chain``, so ``candidate is weighted``
    and the ``np.allclose(weights, weights.flat[0])`` check rejects it; ``fixed``
    is caught earlier still, by the ``isinstance(samples, PosteriorResult)``
    check, before any attribute lookup. ``PosteriorResult.__new__`` sidesteps
    ``__init__`` (which needs a real ``Model``) since only the class identity
    matters here. The all-``None`` dict for ``weighted`` also pins
    ``_autocorrelation_ess``'s exact placeholder shape (oracle F1). Would
    catch: the weights-uniformity check being dropped or its tolerance
    inverted, or the ``PosteriorResult`` short-circuit being removed.
    """
    rng = np.random.default_rng(1)
    weighted = SimpleNamespace(
        samples=rng.standard_normal((600, 2)), weights=rng.random(600)
    )
    assert suite._markov_chain(weighted) is None
    assert suite._autocorrelation_ess(weighted) == {
        "ess": None, "tau_int_max": None, "slowest": None, "reliable": None,
    }

    fixed = suite.PosteriorResult.__new__(suite.PosteriorResult)
    assert suite._markov_chain(fixed) is None


@pytest.mark.parametrize("suite", SUITES)
def test_markov_chain_rejects_a_history_it_cannot_analyze(suite):
    """``_markov_chain``'s own shape/complex guards, fed inputs that trip them.

    Three rejection clauses in ``_markov_chain`` are unreachable through the
    fixtures used elsewhere in this file, because those always supply an object
    with a real ``.chain``: a **complex** history on an object with no
    ``.chain`` (the contour case with its routing removed), a 1-D history, and
    a history with a single row.  ``test_autocorrelation_ess_routes_through_
    the_real_markov_history`` states in its own docstring that deleting the
    complex guard changes nothing on its fixture -- so the guard is asserted
    there and exercised here.

    No sampler in ``pixel.infer`` reaches this function that way today (every
    contour-family dataclass carries both ``.samples`` and ``.chain``), so this
    is defensive completeness rather than a live path, and it is cheap: the
    guard is what keeps a complex array out of the Gamma method, and the
    control below fixes that the same shape with a *real* history is accepted,
    so this cannot pass against a version that rejects everything.  Oracle F1.
    """
    real = np.random.default_rng(3).standard_normal((600, 2))
    # Control: identical object, real history -- accepted, and returns itself
    # (no `.chain` to prefer), so the rejections below are about the history.
    assert suite._markov_chain(SimpleNamespace(samples=real)) is not None

    chainless_complex = SimpleNamespace(samples=real[:, :1] + 1j * real[:, 1:])
    assert suite._markov_chain(chainless_complex) is None
    assert suite._markov_chain(SimpleNamespace(samples=real[:, 0])) is None  # 1-D
    assert suite._markov_chain(SimpleNamespace(samples=real[:1])) is None  # one row


@pytest.mark.parametrize("suite", SUITES)
def test_effective_sample_size_dispatches_on_the_sample_type(suite):
    """The weight/phase ESS picks its definition from the artifact it is given.

    ``_effective_sample_size`` has no direct test anywhere -- it is reached only
    through ``run_fit`` (which this file does not run) and through
    ``summarize``'s source-text pin, which checks the key name and not the
    branch.  Its siblings ``_markov_chain`` and ``_weights`` both get direct
    tests here; this closes the gap for the third.

    Oracle A1 on each branch's arithmetic, computed from the fixture: a fixed
    ``PosteriorResult`` is a single point (``1.0``); an ``ess_frac`` carrier is
    ``n_samples * ess_frac = 600 * 0.25 = 150``; and the fallback is the
    Kish-style ``1 / sum |w|**2`` on weights that ``_weights`` first normalizes
    to sum 1 -- ``[0.5, -0.25, 0.75]`` gives ``1 / 0.875 = 1.142857...``.  That
    last fixture carries a *negative* weight on purpose: it is a signed
    phase-space weight, and ``abs()`` inside the sum is what keeps the answer
    from exceeding the sample count.
    """
    fixed = suite.PosteriorResult.__new__(suite.PosteriorResult)
    assert suite._effective_sample_size(fixed) == 1.0

    contour = SimpleNamespace(n_samples=600, ess_frac=0.25, samples=np.zeros((600, 1)))
    assert suite._effective_sample_size(contour) == pytest.approx(
        150.0, rel=1e-12, abs=0.0
    )

    weighted = SimpleNamespace(
        samples=np.zeros((3, 1)), weights=np.array([0.5, -0.25, 0.75])
    )
    assert suite._effective_sample_size(weighted) == pytest.approx(
        1.0 / 0.875, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("suite", SMALL_SUITES)
def test_small_effective_sample_size_prefers_the_nested_vegas_signed_ess(suite):
    """A ``NestedVegasSamples`` reports its signed ESS, not a generic ``ess_frac``.

    The small suite's ``_effective_sample_size`` inserts a
    ``NestedVegasSamples`` branch *before* the generic ``hasattr(samples,
    "ess_frac")`` one; the full suite has no such branch.  Order is the whole
    claim, so the fixture carries **both** attributes: the correct answer is
    ``signed_ess = 12.5`` and the wrong one is ``n_samples * ess_frac = 90.0``,
    which is what a reordered dispatch would return.  Signed and absolute ESS
    differ exactly when the inner integrand changes sign, which is the regime
    nested VEGAS exists for, so silently using the unsigned number would
    overstate the sample size precisely where it matters.  Oracle A1 on the
    fixture; ``__new__`` sidesteps the 30-field dataclass constructor since
    only the class identity and two attributes matter.
    """
    nested = suite.NestedVegasSamples.__new__(suite.NestedVegasSamples)
    nested.signed_ess = 12.5
    nested.samples = np.zeros((100, 1))  # n_samples is a derived property
    nested.ess_frac = 0.9  # the generic branch's answer, deliberately different
    assert nested.n_samples * nested.ess_frac == pytest.approx(90.0)  # the wrong one
    assert suite._effective_sample_size(nested) == pytest.approx(
        12.5, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("suite", SMALL_SUITES)
def test_small_closure_reads_nested_vegas_signed_weights(suite):
    """``_weights`` prefers ``normalized_signed_weights`` over the generic names.

    The small suite's ``_weights`` checks ``("normalized_signed_weights",
    "residual_weights", "phase_weights", "weights")`` in that order (the full
    suite's list has only the last three -- this attribute is
    ``NestedVegasSamples``-specific). ``samples`` here exposes only
    ``normalized_signed_weights``; if that name were dropped from the small
    suite's priority list, the loop would fall through to the uniform
    ``np.ones(n)/n`` default (``[1/3, 1/3, 1/3]``), which would fail this
    ``assert_allclose`` against ``[0.5, -0.25, 0.75]``. What this does *not*
    establish: the fixture's weights already sum to exactly ``1.0``, so
    ``w / total`` is a no-op here -- a broken renormalization (wrong
    denominator, or skipped entirely) would not be caught, only a wrong
    *choice* of attribute. Default ``rtol``/``atol`` on ``assert_allclose`` is
    appropriate for this exact-echo comparison.
    """
    samples = SimpleNamespace(
        samples=np.array([[0.0], [1.0], [2.0]]),
        normalized_signed_weights=np.array([0.5, -0.25, 0.75]),
    )
    np.testing.assert_allclose(
        suite._weights(samples), np.array([0.5, -0.25, 0.75])
    )


@pytest.mark.parametrize("suite", SUITES)
def test_autocorrelation_ess_falls_below_the_sample_count_when_mixing_is_slow(suite):
    """The headline number must actually respond to a badly-mixing chain.

    ``series`` is an AR(1) process with ``decay = rho = 0.95``. Wolff's
    ``tau_int = 1/2 + sum_t rho(t)``; for AR(1), ``rho(t) = rho**t``, so
    ``tau_int = 1/2 + rho/(1-rho) = 0.5 + 19 = 19.5`` and the inflation factor
    ``N/ESS = 2*tau_int = 39`` -- the ``~40x`` in the inline comment below.
    This is a closed-form property of the generating process (oracle A1), not
    a value read back from the implementation. The bar (``ess < 0.2*n``,
    ``tau_int_max > 5.0``) is deliberately looser than the analytic value (the
    true ESS is ``~0.026*n``, roughly 8x under the ``0.2*n`` threshold; the
    true ``tau_int`` is ~4x the ``5.0`` floor): it is a one-sided regression
    guard against a Gamma-method call that ignores correlation altogether
    (which would report ``ess == n``, ``tau_int_max ~= 0.5``, and fail both
    bounds), not a tight pin on the inflation factor itself.
    """
    rng = np.random.default_rng(2)
    decay, n = 0.95, 2_000
    series = np.empty((n, 1))
    series[0] = 0.0
    for i in range(1, n):
        series[i] = np.sqrt(1 - decay**2) * rng.standard_normal(1) + decay * series[i - 1]
    chain = SimpleNamespace(
        samples=series, param_labels=("slow",), weights=np.full(n, 1.0 / n)
    )

    report = suite._autocorrelation_ess(chain)
    assert report["ess"] < 0.2 * n  # ~40x inflation at rho(1) = 0.95
    assert report["tau_int_max"] > 5.0  # true ~19.5; floor is loose, see docstring
    assert report["slowest"] == "slow"


@pytest.mark.parametrize("suite", SUITES)
def test_summary_reports_both_effective_sample_sizes(suite):
    """The report keeps the phase ESS beside the new autocorrelation headline.

    The two key-name/source-expression pairs are checked verbatim against
    ``inspect.getsource(suite.summarize)`` (oracle F2): they prove
    ``"effective_sample_size"`` is wired to ``result["autocorr"]["ess"]``
    (the Wolff/Gamma-method ESS) and ``"phase_effective_sample_size"`` to
    ``result["ess"]`` (the weight/phase ESS) specifically, not to the same
    source expression under both keys. If a future edit accidentally
    duplicated one under both keys, the literal substring for the other key
    would vanish from the source and this would fail. As with every
    ``inspect.getsource`` check in this file, it cannot distinguish live code
    from the identical text inside a comment. ``_finite_or_none(None) is
    None`` is a one-line direct unit check of the NaN/None-tolerance helper,
    unrelated to the source-text checks around it.
    """
    assert suite._finite_or_none(None) is None  # the report tolerates "n/a"
    source = inspect.getsource(suite.summarize)
    assert '"effective_sample_size": _finite_or_none(result["autocorr"]["ess"])' in source
    assert '"phase_effective_sample_size": _finite_or_none(result["ess"])' in source
    assert '"tau_int_max"' in source and '"tau_reliable"' in source


@pytest.mark.parametrize("runner", RUNNERS)
def test_suite_runner_records_case_failures_and_continues(runner):
    """``run_closure.main`` records per-case failures and keeps going by default.

    Four literal substrings of ``inspect.getsource(runner.main)`` (oracle F2):
    ``main`` catches per-case exceptions (``except Exception as exc``, present
    twice in the real source -- once in the ``--kernels-only`` loop, once in
    the main fit loop -- either occurrence satisfies this `in` check), writes
    both ``failures.json`` and ``run_manifest.json`` into the comparison
    directory, and only re-raises when ``args.fail_fast`` is set. This proves
    the text exists in the function body, not that a failure in one case
    actually leaves later cases unaffected at runtime -- that behaviour is
    exercised for real (not just its source text) by
    ``test_suite_kernel_only_mode_builds_without_fitting`` below, for the
    kernels-only path only.
    """
    source = inspect.getsource(runner.main)
    assert "except Exception as exc" in source
    assert 'comp_dir / "failures.json"' in source
    assert 'comp_dir / "run_manifest.json"' in source
    assert "if args.fail_fast" in source


@pytest.mark.parametrize("runner", RUNNERS)
def test_suite_kernel_only_mode_builds_without_fitting(runner, monkeypatch):
    """``--kernels-only`` builds every ``(Q, mode)`` kernel cache and skips fitting.

    Unlike every other test in this file, this one actually *executes* the
    dispatcher (``runner.main()`` with a real ``sys.argv`` and ``argparse``
    parse) rather than only inspecting source text or calling a small helper
    directly -- the closest thing here to an integration test. ``run_one`` is
    monkeypatched to raise ``AssertionError`` if called at all, so a
    regression that fell through to the fitting path would fail loudly, not
    just leave ``calls`` looking different. ``calls == [(q_key, "both")]``
    pins both the call's arguments and that it happened exactly once (no
    silent double-build). Would catch: ``--kernels-only`` no longer short-
    circuiting before the fit loop, or the ``(q_key, mode)`` loop order
    being swapped.
    """
    calls = []
    # The member argparse will accept, read from the suite under test rather than
    # hardcoded: TRUTH_Q_CHOICES dropped "mc" and "1" on 2026-08-15 and --Q
    # validates against it, so a literal breaks whenever that table changes.
    q_key = next(iter(runner.cfg.TRUTH_Q_CHOICES))

    def fake_build(q_key, mode):
        calls.append((q_key, mode))
        return {
            "n_fields": 9,
            "n_datasets": 3,
            "n_data": 12,
            "runtime_seconds": 0.01,
        }

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("kernel-only mode must not run a fit")

    monkeypatch.setattr(runner, "build_kernels_only", fake_build)
    monkeypatch.setattr(runner, "run_one", unexpected_fit)
    monkeypatch.setattr(
        sys,
        "argv",
        # Not a literal: TRUTH_Q_CHOICES is suite-specific and argparse validates
        # against it, so a hardcoded member breaks whenever the table changes.
        ["run_closure", "--Q", q_key, "--modes", "both", "--kernels-only"],
    )
    runner.main()
    assert calls == [(q_key, "both")]
