"""The closure's sampler choice, against the registry that says which sampler applies.

Exercises: ``closure_*/fit._closure_sampler`` and ``pixel.core.model.Model.has_nonlinear``
    against ``pixel.data.nnpdf.native``'s ``requires_direct_sampler`` and
    ``pixel.data.dataset_prediction_class``/``dataset_nonlinear_kind``.
Claim: the suite dispatches to the direct sampler for exactly the prediction classes the
    registry says cannot be marginalized -- no more, and no fewer.
Oracle: ``F1``. Cross-module consistency; no physics and no sampling.
Cost: four config imports plus two synthetic datasets. ~2 s.
Not covered: whether a direct fit converges, mixes, or is affordable. Routing says which
    sampler is *applicable*, never that it works.

WHY THIS FILE EXISTS
====================
``pixel`` now derives the routing fact -- marginalized evidence needs a prediction
polynomial in ``q``, so every exact ratio is direct-only -- from the prediction class, in
``nnpdf40_native._SAMPLER_ROUTES``. The closure decides the same thing independently, from
``model.has_nonlinear``, in ``_closure_sampler``. Two derivations of one fact in two repos
is exactly the shape that drifts, and the failure would be quiet in the dangerous
direction: a ratio dataset routed to ``nuts`` does not crash, it marginalizes a likelihood
it may not marginalize.

The cross-repo half of the same argument is why ``evidence_value`` and
``Model._require_gaussian_likelihood`` are pinned against each other in
``pixel/tests/test_sampler_routing.py``; those two had in fact drifted, with
``mixed_ratio`` missing from one of them.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from pixel.core.model import dataset_nonlinear_kind
from pixel.data.nnpdf.native import _SAMPLER_ROUTES, load_nnpdf40_native_contracts

#: Suites whose ``_closure_sampler`` chooses between the marginalized and direct routes.
FULL_SUITES = ("closure_NNPDF_truth", "closure_JAM_truth")

#: The smoke suites, whose ``_closure_sampler`` is a *different function* -- it takes a
#: ``mode`` and returns ``"vegas"``/``"gaussian"``, never consulting ``has_nonlinear``.
#: That is safe only because they cannot build a ratio at all (see the test below), and it
#: is pinned rather than assumed: discovered 2026-08-22 when this file first parametrized
#: over all four and the two smoke suites failed.
SMALL_SUITES = ("closure_NNPDF_truth_small", "closure_JAM_truth_small")

SUITES = FULL_SUITES + SMALL_SUITES

#: Nonlinear kinds, i.e. every kind ``Model.has_nonlinear`` reports True for.
NONLINEAR_KINDS = ("linear_ratio", "bilinear_ratio", "mixed_ratio")


class _FakeLayout:
    def __init__(self, nonlinear=(), bilinear_ratio=(), mixed_ratio=()):
        self.nonlinear_datasets = nonlinear
        self.bilinear_ratio_datasets = bilinear_ratio
        self.mixed_ratio_datasets = mixed_ratio


class _FakeModel:
    """Only what ``_closure_sampler`` reads."""

    def __init__(self, *, has_nonlinear, has_bilinear=False, n_ordinary=0):
        self.has_nonlinear = has_nonlinear
        self.has_bilinear = has_bilinear
        self._n_ordinary = n_ordinary

    def contour_partition(self):
        return type("P", (), {"n_ordinary": self._n_ordinary})()


@pytest.mark.parametrize("suite", FULL_SUITES)
def test_a_nonlinear_model_goes_to_the_direct_sampler(suite):
    """The dispatch itself, for every suite that has one."""
    fit = importlib.import_module(f"{suite}.fit")
    chooser = getattr(fit, "_closure_sampler", None)
    if chooser is None:
        pytest.skip(f"{suite} has no _closure_sampler")
    assert chooser(_FakeModel(has_nonlinear=True)) == "direct_nuts"
    # And the two marginalized routes stay reachable, so this is not a guard that
    # answers "direct" to everything.
    assert chooser(_FakeModel(has_nonlinear=False)) == "nuts"
    assert chooser(
        _FakeModel(has_nonlinear=False, has_bilinear=True, n_ordinary=3)
    ) == "hmc"


@pytest.mark.parametrize("suite", FULL_SUITES)
def test_the_dispatch_reads_has_nonlinear_and_nothing_narrower(suite):
    """``has_nonlinear`` covers all three nonlinear kinds; a narrower test would not.

    Parsed rather than restated. A dispatch written against, say,
    ``model._layout.nonlinear_datasets`` would route ``linear_ratio`` correctly and send
    ``bilinear_ratio`` and ``mixed_ratio`` to ``nuts`` -- the quiet failure.
    """
    fit = importlib.import_module(f"{suite}.fit")
    chooser = getattr(fit, "_closure_sampler", None)
    if chooser is None:
        pytest.skip(f"{suite} has no _closure_sampler")
    source = inspect.getsource(chooser)
    assert "has_nonlinear" in source
    assert "nonlinear_datasets" not in source, (
        "dispatching on one layout tuple misses the other two nonlinear kinds; "
        "Model.has_nonlinear is the property that covers all three"
    )


def test_model_has_nonlinear_covers_every_direct_only_kind():
    """``has_nonlinear`` must be True for exactly the kinds the registry routes direct."""
    from pixel.core.model import Model

    source = inspect.getsource(Model.has_nonlinear.fget)
    named = {
        name.removesuffix("_datasets")
        for name in __import__("re").findall(r"self\._layout\.(\w+)", source)
    }
    assert named == set(NONLINEAR_KINDS) - {"linear_ratio"} | {"nonlinear"}, sorted(named)
    # ``linear_ratio`` datasets land in the ``nonlinear_datasets`` tuple -- see
    # ``infer/direct._direct_prediction_layout``, which maps the two -- so the layout name
    # and the prediction class deliberately differ here.


def test_registry_direct_only_classes_are_the_nonlinear_kinds():
    """The two vocabularies -- roster prediction class and runtime nonlinear kind -- agree.

    ``_SAMPLER_ROUTES`` speaks in prediction classes; ``dataset_nonlinear_kind`` speaks in
    runtime kinds. They must name the same set, or a dataset could be direct-only in one
    vocabulary and marginalizable in the other.
    """
    direct_only = {
        prediction for prediction, routes in _SAMPLER_ROUTES.items()
        if "marginalized" not in routes
    }
    assert direct_only == set(NONLINEAR_KINDS) | {"nonlinear"}, sorted(direct_only)


def test_no_roster_dataset_the_closure_can_build_is_misrouted():
    """Every direct-only roster dataset must be one the closure sends to direct_nuts.

    The bridge test: it walks the *real* roster, not a synthetic model, so a dataset
    promoted into a class the closure does not route correctly fails here.
    """
    fit = importlib.import_module("closure_NNPDF_truth.fit")
    for contract in load_nnpdf40_native_contracts():
        if not contract.requires_direct_sampler:
            continue
        # A model carrying this dataset reports has_nonlinear, so the closure must
        # choose the direct sampler for it.
        assert fit._closure_sampler(_FakeModel(has_nonlinear=True)) == "direct_nuts", (
            contract.legacy_name
        )
        assert contract.prediction in set(NONLINEAR_KINDS) | {"nonlinear"}


def test_the_f2_ratio_dataset_reports_the_kind_the_registry_assigns_it():
    """NMCPD's class, end to end from a built dataset rather than from the roster text.

    The roster says ``prediction: linear_ratio``; this asserts a real ``F2Ratio`` object
    reports the same kind, so the roster is describing the thing that gets built.
    """
    import numpy as np

    from pixel.data import F2Ratio

    dataset = F2Ratio.from_arrays(
        x=np.array([0.1, 0.2, 0.3]), Q2=np.array([5.0, 8.0, 12.0]),
        mean=np.ones(3), cov=np.eye(3),
        maps_to={
            "u_minus_d": "t3", "u_plus_d_minus_2s": "t8",
            "u_plus_d_plus_s_minus_3c": "t15", "singlet": "sigma", "gluon": "g",
        },
        nf=4, Q20=1.65 ** 2, order="NLO", matching_order="NLO",
    )
    assert dataset_nonlinear_kind(dataset) == "linear_ratio"
    contract = next(
        c for c in load_nnpdf40_native_contracts() if c.legacy_name == "NMCPD_dw_ite"
    )
    assert contract.prediction == "linear_ratio"
    assert contract.requires_direct_sampler


@pytest.mark.parametrize("suite", SMALL_SUITES)
def test_the_smoke_suites_cannot_build_a_dataset_their_sampler_cannot_route(suite):
    """Their ``_closure_sampler`` never looks at ``has_nonlinear`` -- so nothing may need it.

    The smoke suites answer ``"vegas"`` (or ``"gaussian"`` for the synthetic-Z mode) for
    every model. That is fine while no dataset they can build is direct-only, and this
    pins exactly that precondition: their builder tables carry only linear DIS kinds, none
    of which is a ratio. Adding a ratio builder to a smoke suite without teaching its
    sampler about ``has_nonlinear`` would fail here.

    The consequence if it ever slipped through is at least *loud*, not silent:
    ``evidence_value`` refuses any nonlinear block outright.
    """
    datasets = importlib.import_module(f"{suite}.datasets")
    table = dict(getattr(datasets, "_EXP_BUILDERS", {}))
    # Since 2026-09-09 the table lists **every** datatype JAM24 or NNPDF4.0 uses, most
    # of them behind a gate that raises (``pixel.data.closure_datatypes``). A gated
    # entry cannot build anything, so it cannot build "a dataset their sampler cannot
    # route" -- the claim is about what is *buildable*, and that is what is compared.
    buildable = {k for k, v in table.items() if not v.__name__.startswith("gated_")}
    gated = set(table) - buildable
    assert buildable == {"f2", "sigma_r", "sigma_r_cc"}, sorted(buildable)
    for name in buildable:
        assert "ratio" not in name and "asym" not in name

    # And the stronger statement the gate now makes available: the ratio and asymmetry
    # kinds are not merely absent from the table, they are present and *refuse*. Before
    # the registry this could only be asserted by their absence, which a future edit
    # could have undone silently by adding a working one.
    from pixel.data.closure_datatypes import DatatypeGateError

    for name in gated:
        if "ratio" in name or "asym" in name:
            with pytest.raises(DatatypeGateError):
                table[name]({}, None, None, None)

    fit = importlib.import_module(f"{suite}.fit")
    source = inspect.getsource(fit._closure_sampler)
    assert "has_nonlinear" not in source, (
        "this suite now consults has_nonlinear; move it to FULL_SUITES so the real "
        "dispatch assertions apply to it"
    )
