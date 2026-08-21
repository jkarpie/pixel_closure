"""``GP_ENVELOPE``: the physics invariants, and cross-suite consistency.

The companion to ``pixel``'s ``tests/test_beta_tapered_prior.py``.  That file
guards the *kernel* -- that ``BetaTaperedLogRBF`` really implements
``sigma^2 |x|^a |x'|^a |1-x|^b |1-x'|^b * logRBF`` -- over a range of exponents.
This file guards the *configuration*: that what the four suites actually ship
stays inside that range and keeps the properties it was derived to have.

The split exists because the closure packages moved out of the ``pixel`` checkout
on 2026-08-16.  A test in ``pixel`` can no longer import ``closure_*/config.py``,
so the sync assertion has to live here.

What is asserted, and why each one is a real invariant rather than a restatement
of the file:

* **``alpha >= 0`` for every field.**  The prior must not diverge as ``x -> 0``.
  This is an owner decision (2026-08-16) and it caught a real error: an earlier
  unconstrained minimax gave ``t8`` ``alpha = -0.30``, a *diverging* envelope for
  a field whose truth plainly converges -- ``x f`` is 0.03 at ``x = 1e-6`` against
  a bulk peak of 1.09, at every scale.  A regression here is a physics error, not
  a style one.
* **The four suites agree.**  ``fit.constraint_datasets`` is byte-identical across
  the packages and the priors are meant to be too; a drift between NNPDF and JAM
  copies would make any cross-truth comparison meaningless without announcing
  itself.
* **The exponents stay inside the range ``pixel``'s kernel tests exercise.**
  Otherwise the kernel is verified over one region and used in another.
* **``beta > 0``**, so the prior vanishes at ``x = 1`` and ``cons_endpoint_*`` is
  supported by the prior rather than fought by it.
"""

import importlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUITES = (
    "closure_NNPDF_truth_small",
    "closure_JAM_truth_small",
    "closure_NNPDF_truth",
    "closure_JAM_truth",
)

#: The (alpha, beta) region ``pixel``'s ``tests/test_beta_tapered_prior.py``
#: parametrizes over.  Keep in step if that file's ``SHIPPED`` changes.
ALPHA_RANGE = (0.0, 0.8)
BETA_RANGE = (0.5, 4.0)


def _cfg(suite):
    if str(ROOT / suite) not in sys.path:
        sys.path.insert(0, str(ROOT / suite))
    return importlib.import_module(f"{suite}.config")


@pytest.mark.parametrize("suite", SUITES)
def test_every_envelope_alpha_is_non_negative(suite):
    """The prior may not diverge as ``x -> 0``, for any field.

    Mutation that fails this: restoring any negative ``alpha`` in
    ``GP_ENVELOPE`` -- e.g. ``t8``'s pre-2026-08-16 ``-0.30``.
    """
    cfg = _cfg(suite)
    negative = {f: v["alpha"] for f, v in cfg.GP_ENVELOPE.items() if v["alpha"] < 0}
    assert not negative, (
        f"{suite}: diverging envelope for {negative}; alpha must be >= 0 so the "
        "prior converges as x -> 0"
    )


@pytest.mark.parametrize("suite", SUITES)
def test_every_envelope_beta_vanishes_at_the_endpoint(suite):
    """``beta > 0``, so the prior supports ``cons_endpoint_*`` instead of fighting it.

    Mutation that fails this: ``beta = 0`` for any field (the envelope then does
    not vanish at ``x = 1`` and the near-hard endpoint constraint has to remove a
    finite prior width).
    """
    cfg = _cfg(suite)
    bad = {f: v["beta"] for f, v in cfg.GP_ENVELOPE.items() if v["beta"] <= 0}
    assert not bad, f"{suite}: non-vanishing endpoint for {bad}"


@pytest.mark.parametrize("suite", SUITES)
def test_envelope_exponents_stay_inside_the_tested_kernel_range(suite):
    """Do not use the kernel outside the region its own tests exercise.

    ``pixel``'s ``test_beta_tapered_prior.py`` parametrizes a fixed set of
    ``(alpha, beta)`` pairs.  If the configuration drifts outside that box the
    kernel is verified in one region and used in another, which is the failure
    this pair of files exists to prevent.

    Mutation that fails this: any exponent outside ALPHA_RANGE / BETA_RANGE.
    """
    cfg = _cfg(suite)
    out = {
        f: (v["alpha"], v["beta"])
        for f, v in cfg.GP_ENVELOPE.items()
        if not (ALPHA_RANGE[0] <= v["alpha"] <= ALPHA_RANGE[1]
                and BETA_RANGE[0] <= v["beta"] <= BETA_RANGE[1])
    }
    assert not out, (
        f"{suite}: {out} outside the kernel-tested box alpha{ALPHA_RANGE} "
        f"beta{BETA_RANGE}; widen the parametrization in pixel's "
        "tests/test_beta_tapered_prior.py in the same change"
    )


def test_all_four_suites_ship_the_same_envelope():
    """A drift between the NNPDF and JAM copies would silently break comparisons.

    Mutation that fails this: changing ``GP_ENVELOPE`` in one suite only, which is
    exactly what happens when a retune is applied by hand to the package someone
    is working in.
    """
    ref = _cfg(SUITES[0]).GP_ENVELOPE
    for suite in SUITES[1:]:
        got = _cfg(suite).GP_ENVELOPE
        assert got == ref, (
            f"{suite} ships a different GP_ENVELOPE than {SUITES[0]}: "
            f"{ {k: (got.get(k), ref.get(k)) for k in set(got) | set(ref) if got.get(k) != ref.get(k)} }"
        )


def test_all_four_suites_share_the_numerical_settings():
    """Constraint sigmas, rcond, jitters and GP length must not drift either.

    These carry measurements in their config comments (the constraint cliff at
    1e-6, the rcond plateau, the two jitter boundaries).  A per-suite drift makes
    those comments wrong somewhere without saying where.
    """
    keys = ("CONSTRAINT_ENDPOINT_SIGMA", "CONSTRAINT_NORM_SIGMA",
            "CONSTRAINT_ORIGIN_SIGMA", "CONSTRAINT_MOMENTUM_SIGMA",
            "RCOND", "GP_JITTER", "GP_ENVELOPE_JITTER", "GP_LENGTH_LOG",
            "PRIOR_FORM")
    ref = {k: getattr(_cfg(SUITES[0]), k) for k in keys}
    for suite in SUITES[1:]:
        cfg = _cfg(suite)
        got = {k: getattr(cfg, k) for k in keys}
        diff = {k: (got[k], ref[k]) for k in keys if got[k] != ref[k]}
        assert not diff, f"{suite} differs from {SUITES[0]}: {diff}"
