"""Closure-package tests moved out of ``pixel`` on 2026-08-16.

These parametrize over ``closure_*/config.py`` and read closure manifests, so
they stopped resolving the moment the packages left that checkout -- 14 of the
19 failures in ``pixel``'s suite that day were these.  They are not skipped
there and duplicated here: they were *removed* from ``pixel``, because a test
that silently stops covering anything is worse than one that has moved.

The helpers below are copied rather than imported, so this file does not depend
on a sibling checkout's test layout.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pixel import kernels
from pixel.api.manifest import expand_manifest
from pixel.kernels.common import low_x_fourier_head
from pixel.kernels.lowx import high_x_mellin_head
from scipy.integrate import quad

#: Fields whose momentum density may tend to a non-zero constant as x -> 0 --
#: the singlet and the gluon.  Copied from pixel's test_no_low_x_truncation.py
#: with the tests that use it.
CONSTANT_AT_ORIGIN_FIELDS = ("sigma", "g")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _suite in ("closure_NNPDF_truth_small", "closure_JAM_truth_small",
               "closure_NNPDF_truth", "closure_JAM_truth"):
    _p = ROOT / _suite
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

def _closure_options():
    """Shared closure-profile options for the tests below.

    ``dy_deuteron_map`` (finding ``test_manifest_expansion-01``, fixed
    2026-08-13) is a genuinely distinct map from ``dy_proton_map``: the
    isospin-averaged nucleon under exact isospin symmetry.  See the module
    docstring for the derivation; ``_deuteron_map`` below builds it from
    ``proton`` directly so the two can never accidentally drift back into the
    same object.
    """
    even = {
        "u_minus_d": "t3",
        "u_plus_d_minus_2s": "t8",
        "singlet": "sigma",
        "u_plus_d_plus_s_minus_3c": "t15",
        "gluon": "g",
    }
    odd = {
        "u_minus_d": "v3",
        "u_plus_d_minus_2s": "v8",
        "singlet": "v",
        "u_plus_d_plus_s_minus_3c": "v15",
    }
    proton = {
        "u": {"sigma": 0.25, "t3": 0.5},
        "ub": {"sigma": 0.25, "v3": -0.5},
        "d": {"sigma": 0.25, "t3": -0.5},
        "db": {"sigma": 0.25, "v3": 0.5},
        "g": {"g": 1.0},
    }
    return {
        "sections": ["lattice", "exp", "dy"],
        "even_map": even,
        "odd_map": odd,
        "low_x_extensions": {},
        "nf": 4,
        "Q20": 1.28**2,
        "order": "NLO",
        "mode": "truncated",
        "mc": 1.28,
        "mb": 4.18,
        "lattice_L": 48,
        "lattice_a": 0.06,
        "moment_Q2": 4.0,
        "fit_normalization": True,
        "dy_proton_map": proton,
        "dy_deuteron_map": _deuteron_map(proton),
        "dy_fit_normalization": True,
    }


def _deuteron_map(proton):
    """The isospin-averaged nucleon field map, built from ``proton`` directly.

    ``u_deuteron = 0.5*(u_proton + d_proton)``, and symmetrically for the
    other three quark legs; ``g`` is isospin-blind.  See the module docstring
    for why the non-singlet (``t3``/``v3``) coefficients cancel exactly under
    this averaging, for any ``proton`` map built the same way this fixture's
    is.
    """
    def half_sum(a, b):
        keys = set(a) | set(b)
        return {k: 0.5 * (a.get(k, 0.0) + b.get(k, 0.0)) for k in keys}

    return {
        "u": half_sum(proton["u"], proton["d"]),
        "ub": half_sum(proton["ub"], proton["db"]),
        "d": half_sum(proton["d"], proton["u"]),
        "db": half_sum(proton["db"], proton["ub"]),
        "g": dict(proton["g"]),
    }


def _q_vanishing(x):
    """A momentum density ``x*f(x)`` that vanishes linearly at the origin.

    Cubic, so an ``order=3`` nodal basis reproduces it exactly and the interior
    quadrature contributes no interpolation error to anything measured here.
    """
    x = np.asarray(x, dtype=float)
    return x * (1.0 - x) ** 2


LOW_X_XMIN_LADDER = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


PRODUCTION_X_MIN = 1e-6


def _q_singlet(x):
    """A momentum density ``x*f(x)`` that tends to a nonzero constant at the origin."""
    x = np.asarray(x, dtype=float)
    return (1.0 - x) ** 3


@pytest.mark.parametrize(
    "module_name",
    (
        "closure_JAM_truth.config",
        "closure_NNPDF_truth.config",
        "closure_JAM_truth_small.config",
        "closure_NNPDF_truth_small.config",
    ),
)
def test_closure_field_policy_is_flat_or_linear(module_name):
    """Singlet/gluon fields get 'flat'; every other field gets 'power' alpha=1.0.

    Routes ``low_x_completion(field)`` through the real
    ``normalize_low_x_extension`` / ``.settings()`` pipeline, so a spec must
    actually construct a valid ``LowXExtension``, not just look right as a raw dict.
    The ``kind`` branch's oracle, ``SINGLET_GLUON_FIELDS``, is the same constant the
    source reads -- common-mode for a wrong physics classification, not for a logic
    bug in how it is used (module docstring). ``alpha == approx(1.0)`` is an
    independent literal, not a read of ``cfg.LOW_X_LINEAR_POWER``; measured exact
    (0.0 deviation, all 4 modules -- 0% of the default ``rel=1e-6`` bar used). Never
    reaches ``low_x_completion``'s ``None`` branch: all four grids measured starting
    at ``1e-6``, not ``0.0``, today.
    """
    module = __import__(module_name, fromlist=["config"])
    for field in module.ALL_FIELDS:
        settings = kernels.low_x_extension_settings(module.low_x_completion(field))
        if field in module.SINGLET_GLUON_FIELDS:
            assert settings["kind"] == "flat"
        else:
            assert settings["kind"] == "power"
            # Default pytest.approx rel=1e-6; this constrains the closure-suite's linear
            # low-x power alpha=1.0, an independent literal (not cfg.LOW_X_LINEAR_POWER)
            # -- measured exact (0.0 deviation) on all 4 modules, so the bar is never
            # used.
            assert settings["alpha"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "module_name",
    (
        "closure_JAM_truth.config",
        "closure_NNPDF_truth.config",
        "closure_JAM_truth_small.config",
        "closure_NNPDF_truth_small.config",
    ),
)
def test_closure_completion_accuracy_converges_in_x_min(module_name):
    """The closure config's own choice, judged as a number and refined in ``x_min``.

    Oracle ``B1``: ``scipy.integrate.quad`` of the **true** momentum density over
    ``[0, x_min)``.  Every other check on this policy -- here, in
    ``tests/test_dis_builder_equivalence.py``, in
    ``tests/test_closure_extension_systematics.py`` -- stops at a settings dict or a
    ``kind`` string.  This one asks the question those cannot: given the completion the
    config actually selected, **how wrong is the missing integral, and does the error
    shrink as the resolved region grows?**

    **Refinement is in ``x_min``, deliberately, because that is the parameter under
    suspicion.**  A previous investigation of the low-x head refined the grid (``N`` 65
    to 513) and the panel count (8 to 256), saw no change, and concluded the formula was
    wrong.  It was refining the wrong knob: the completion models ``[0, x_min)``, a
    region the grid does not touch, so grid refinement cannot move its error by
    construction.  ``x_min`` can, and does.

    **The two field classes are reported separately because they converge differently**,
    which is the physics:

    * ``sigma``/``g`` -- ``x*f(x)`` tends to a nonzero constant, so the ``flat``
      completion is the right model and the head tends to a constant times the interval;
    * every other field -- ``x*f(x)`` vanishes, so ``power alpha=1`` ("linear") is the
      right model and the head shrinks as ``x_min^2``.

    MEASURED over the ladder ``1e-4 ... 1e-8``, worst relative error of the modelled
    head against ``quad`` of the true density, at ``nu = 1, 5, 20``:

    ======================  =========  =========  =========  =========  =========
    class / parity          ``1e-4``   ``1e-5``   ``1e-6``   ``1e-7``   ``1e-8``
    ======================  =========  =========  =========  =========  =========
    vanishing, cos          6.667e-05  6.667e-06  6.667e-07  6.667e-08  6.667e-09
    vanishing, sin          5.000e-05  5.000e-06  5.000e-07  5.000e-08  5.000e-09
    singlet/gluon, cos      1.500e-04  1.500e-05  1.500e-06  1.500e-07  1.500e-08
    ======================  =========  =========  =========  =========  =========

    Exactly ``10x`` per decade in all three rows -- first order in ``x_min``, as the
    leading neglected term requires.  At the production ``x_min = 1e-6`` the completion
    is accurate to ``1.5e-06`` or better, three to four decades below this repo's
    recorded low-x completion outliers (``3.1e-2`` sigma, ``8.2e-2`` t15), which are
    therefore **not** attributable to the head model at this ``x_min``.

    **The wrong pairing does not converge at all**, which is what makes the assertion
    below discriminating rather than decorative.  Measured, flat-vs-linear swapped:
    ``1.0`` (flat head on a vanishing field, cosine), ``0.5`` (same, sine), ``0.5``
    (linear head on a singlet) -- identical at every ``x_min`` on the ladder, to four
    figures.  That swap is this test's acceptance mutation, not a case it asserts on:
    the suite carries no claims about configurations nobody would run.

    **Only ``spec`` comes from the config.**  Which fields are constant at the origin is
    this file's own literal ``CONSTANT_AT_ORIGIN_FIELDS``, not
    ``cfg.SINGLET_GLUON_FIELDS`` -- the constant the config's own selection reads.  That
    matters: ``test_closure_field_policy_is_flat_or_linear`` rebuilds its expectation
    from that constant and its own docstring records the consequence, that a wrong
    *physics* classification baked into it would be agreed on by both sides.  Measured
    here: adding ``"t15"`` to a config's ``SINGLET_GLUON_FIELDS`` in memory leaves the
    policy test passing (it now expects ``flat`` and gets ``flat``) while this test
    fails, because ``t15``'s density really does vanish and a flat head really is
    ``100%`` wrong about its missing integral.  The model density, the parity-to-trig
    mapping and the reference integral are likewise written out here.
    """
    module = __import__(module_name, fromlist=["config"])
    nu = np.array([1.0, 5.0, 20.0])
    seen = {}

    for field in module.ALL_FIELDS:
        spec = module.low_x_completion(field)
        # The field's own small-x limit, from this file's independent literal -- NOT
        # from cfg.SINGLET_GLUON_FIELDS, which is the constant the config's own
        # selection reads, and would otherwise supply its own justification.
        constant_at_origin = field in CONSTANT_AT_ORIGIN_FIELDS
        q = _q_singlet if constant_at_origin else _q_vanishing
        use_sin = module.field_cparity(field) == "odd"
        key = (repr(kernels.low_x_extension_settings(spec)), use_sin, constant_at_origin)
        if key in seen:
            continue

        errors = []
        for x_min in LOW_X_XMIN_LADDER:
            head = np.asarray(low_x_fourier_head(nu, x_min, spec, use_sin=use_sin))
            modeled = head * float(q(x_min))
            oscillator = np.sin if use_sin else np.cos
            truth = np.array([
                quad(lambda x, v=v: oscillator(v * x) * float(q(x)), 0.0, x_min,
                     limit=400, epsabs=1e-300, epsrel=1e-12)[0]
                for v in nu
            ])
            # A truncated head returns zeros: that is a relative error of 1, which the
            # convergence assertion below rejects outright.
            assert np.all(np.abs(truth) > 0.0), f"{field}: degenerate reference"
            errors.append(float(np.max(np.abs(modeled / truth - 1.0))))
        seen[key] = errors

        label = f"{module_name} {field} ({'flat/const' if constant_at_origin else 'linear/vanishing'})"
        # Convergence in x_min: measured exactly 10x per decade, so 5x is a wide bar
        # that a plateau (the mismatched-pairing signature, 0.5 or 1.0 flat across the
        # whole ladder) cannot clear even once.
        for coarse, fine in zip(errors, errors[1:]):
            assert fine < coarse / 5.0, f"{label}: completion stopped converging: {errors}"
        production = errors[LOW_X_XMIN_LADDER.index(PRODUCTION_X_MIN)]
        # Measured 6.667e-07 / 5.000e-07 / 1.500e-06 by class; bar 3e-6 is ~2x the
        # worst of them, and ~3e5 times below either mismatched-pairing plateau.
        assert production < 3e-6, f"{label}: x_min=1e-6 completion error {production:.3e}"

    # Both classes must actually have been exercised -- a config that returned the same
    # completion for every field would otherwise satisfy everything above -- and both
    # parities, since the head's cosine branch is the one carrying the known open
    # column-0 defect (``plans/test_audit/IN_FLIGHT.md``) and the sine branch is exact.
    assert {k[2] for k in seen} == {True, False}
    assert {k[1] for k in seen} == {True, False}
    assert set(CONSTANT_AT_ORIGIN_FIELDS) <= set(module.ALL_FIELDS)


@pytest.mark.parametrize(
    "module_name",
    (
        "closure_JAM_truth.config",
        "closure_NNPDF_truth.config",
        "closure_JAM_truth_small.config",
        "closure_NNPDF_truth_small.config",
    ),
)
def test_closure_grids_reach_their_upper_domain(module_name):
    """No closure may rely on a large-x completion.

    A grid whose last node sits below its domain leaves ``(grid[-1], hi]`` to be
    *modelled* rather than resolved.  Every PDF grid here runs to ``x = 1``, so the
    domain rule rejects a high-x completion outright -- this pins that.

    Oracle ``F1``: reads each real production closure config directly (not a
    synthetic fixture) and checks a real geometric property against it.  Catches any
    of the 4 registered configs' ``make_grid()`` drifting to a grid that no longer
    reaches ``x = 1`` (e.g. a stray ``x_cutoff``/``domain`` misconfiguration), which
    would silently reintroduce a large-x completion into production.  The
    ``pytest.skip`` fallback for a config without ``make_grid`` is not currently
    exercised: all 4 registered modules define it today.
    """
    module = __import__(module_name, fromlist=["config"])
    grid = module.make_grid() if hasattr(module, "make_grid") else None
    if grid is None:  # configs expose the grid through different helpers
        pytest.skip(f"{module_name} has no make_grid helper")
    nodes = grid.as_array()
    assert nodes[-1] == pytest.approx(grid.integration_domain[1])
    assert grid.high_x_completion is False


@pytest.mark.parametrize("suite", ["closure_NNPDF_truth_small"])
def test_small_closure_manifest_expands_to_closed_dataset_recipes(suite):
    """The real small-closure manifest expands to exactly its own entry count.

    ``29`` is counted directly against ``manifest.json``'s section lists (9
    pseudoitd + 9 mellin + 5 f2 + 2 sigma_r + 2 f3_proxy + 2 drell_yan),
    independent of this function's own output.  The builder-set check is
    ``>=``: presence per kind, not exhaustiveness.  ``bcdms_p_f2`` spot-checks
    one record's correlated-systematics sidecar and an absent-array default.

    Catches a record silently dropped or duplicated in translation, or a kind
    that stopped reaching its builder.  Does not check most records'
    field-level content.  ``dy_e866_pd`` (the fixture's one genuine ``"pd"``
    reaction record) now spot-checks the deuteron field map (finding
    ``test_manifest_expansion-01``, fixed 2026-08-13 -- see the module
    docstring and ``_closure_options``); before that fix this record's
    ``fields_B`` was silently the proton map, and nothing here or elsewhere
    in the suite could have told the difference.
    """
    root = Path(__file__).resolve().parents[1] / suite / "data" / "truthQ_2"
    manifest = json.loads((root / "manifest.json").read_text())
    options = _closure_options()

    records = expand_manifest(
        manifest,
        profile="closure",
        options=options,
        manifest_root=root,
        data_root=root,
    )

    # 9 pseudoitd + 9 mellin (lattice) + 5 f2 + 2 sigma_r + 2 sigma_r_cc (exp) +
    # 2 drell_yan (dy) = 29, counted directly against manifest.json's own
    # section lists, not against expand_manifest's output.
    #
    # The two charged-current records were ``f3_proxy`` (builder ``f3.file``)
    # until 2026-08-15, when the suite moved to the physical reduced cross
    # section (``sigma_r_cc`` -> ``sigma_r.cc.file``).  The *count* is unchanged
    # because it is a one-for-one replacement, which is exactly why the count
    # alone did not catch the switch -- the builder set did.
    assert len(records) == 29
    assert {record["builder"] for record in records} >= {
        "pseudoitd.file",
        "mellin.file",
        "f2.file",
        "sigma_r.file",
        "sigma_r.cc.file",
        "drell_yan",
    }
    # The CC records must carry the valence map and their beam charge; without
    # the second field map the builder silently loses the parity-odd piece.
    cc_records = [r for r in records if r["builder"] == "sigma_r.cc.file"]
    assert len(cc_records) == 2
    for record in cc_records:
        assert record["field_maps"]["maps_to"] == options["even_map"]
        assert record["field_maps"]["valence_maps_to"] == options["odd_map"]
        assert record["params"]["charged_current"] is True
        assert record["params"]["beam_charge"] in {"e_minus", "e_plus"}
    assert {r["params"]["beam_charge"] for r in cc_records} == {"e_minus", "e_plus"}
    bcdms = next(record for record in records if record["name"] == "bcdms_p_f2")
    assert bcdms["arrays"]["multiplicative_systematics"]
    assert bcdms["arrays"]["t0"] is None
    assert all(record["builder_schema_version"] == 1 for record in records)

    dy_pd = next(record for record in records if record["name"] == "dy_e866_pd")
    assert dy_pd["field_maps"]["fields_A"] == options["dy_proton_map"]
    assert dy_pd["field_maps"]["fields_B"] == options["dy_deuteron_map"]
    assert dy_pd["field_maps"]["fields_B"] != options["dy_proton_map"]
