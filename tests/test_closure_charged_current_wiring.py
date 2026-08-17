"""Wiring checks for HERA charged-current DIS data, on both closure suite sizes.

Exercises the closure drivers directly, not ``src/pixel``:
``{suite}.generate.exp_layout`` (the manifest -> per-record parser), ``{suite}.config``
(``EVEN_MAP``, ``ODD_MAP``, ``ORDER``, ``TEST_MODES``, ``CC_SIGMA_R_IDX``,
``EXP_SPECS``), and ``{suite}.datasets`` (``_EXP_BUILDERS``, ``_experimental_builder``,
``build_sigma_r_cc``, ``_even_map``, ``_odd_map``), for every ``suite`` in ``SUITES``
below.

``SUITES`` had a JAM/NNPDF axis as well as the full/small one until 2026-08-14, when the
JAM entries were dropped on the owner's instruction.  The two axes were never equivalent.
For a given size, ``datasets.py``'s charged-current logic is byte-identical between the
JAM and NNPDF variant (diffed), so the JAM/NNPDF axis only ever ran the same code twice;
it was kept as drift protection against one copy being hand-edited without the other.
The full/small axis, which stays, exercises genuinely different code:
``build_sigma_r_cc`` takes a ``with_systematics`` kwarg only in the full suites, and CC
dataset registration is the ``CC_SIGMA_R_IDX`` tuple in the full config against per-entry
``EXP_SPECS.kind`` in the small one.  Every behavioural asymmetry this file pins is on
that axis, including the ``TARGET_MAP`` one described below, so all of them survive the
trim.

The one ``pixel`` symbol referenced here, ``pixel.data.ChargedCurrentSigmaR.from_file``
(``src/pixel/data/experimental/charged_current.py``, the real implementation, not a
shim), is monkeypatched out in the test that touches it, so no ``src/pixel`` code
actually executes in this file -- it is pure closure-driver wiring.  The physics tests
of ``ChargedCurrentSigmaR`` itself (the LO cross-section formula, the fitpack weight
convention, minus-vs-singlet valence routing) live in
``tests/test_charged_current_dis.py`` and are not repeated here.

Most assertions are structural (dict/tuple membership, exception type, string/list
equality) and carry no accuracy bar.  The one numeric check -- the rowwise inelasticity
``y`` in ``test_cc_layout_keeps_beam_and_rowwise_inelasticity`` -- computes the same
closed-form kinematic identity ``y = Q2 / (x * RS**2)`` the source uses, just regrouped,
so its ``atol=1e-15`` bar is a float-reassociation floor rather than an independent
measurement.

Two degeneracies documented here by a previous audit pass were fixed on 2026-08-13 and
their fixtures are now measured non-degenerate in the dimension each test names: the
layout fixture's rows no longer share a ``Q2/x`` ratio or an ``RS`` (a row-0 broadcast
mutant now misses row 1 by 0.259, and a mean broadcast by 0.130, against an ``atol`` of
1e-15), and the ``maps_to``/``valence_maps_to`` checks now compare the *tag of the field
object bound to each basis key*, not the key set, so a map binding every key to one
field fails.

One suite asymmetry found while writing those checks, and pinned below rather than
assumed away: the two full suites expand an abbreviated fitpack target label through
``generate.TARGET_MAP`` (``"p" -> "proton"``), and the two ``_small`` suites do **not**
-- their ``exp_layout`` copies ``tab["target"]`` through verbatim and define no
``TARGET_MAP`` at all.  Both behaviours are asserted per suite family in
``test_cc_layout_keeps_beam_and_rowwise_inelasticity``.

**Settled 2026-08-14: the asymmetry is intentional, and harmless.**  The full suites'
``dis_audit`` reads the target off the fitpack spreadsheet, which writes ``p``/``d``;
the ``_small`` suites' writes ``spec.target`` from ``config.EXP_SPECS``, which is
already canonical -- measured on the shipped manifests, ``p``/``d`` on all 13 ``used``
tables of each full suite and ``proton``/``deuteron`` on all 7 of each ``_small`` one.
Downstream nothing can tell the two apart: ``pixel.util.flavor.normalize_target``,
reached from every DIS builder through ``_dis_common``, maps both to the same canonical
target and raises on anything outside its alias table.
``test_target_labels_reaching_the_builders_agree_with_pixels_alias_table`` pins all
three of those facts, including that ``TARGET_MAP`` stays a strict subset of PIXEL's
aliases -- so a second, disagreeing convention in the closure copy fails here.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


#: One full-size and one ``_small`` closure package -- the two sizes really do differ
#: (see the module docstring).  The JAM pair was removed 2026-08-14; their
#: charged-current ``datasets.py`` logic is byte-identical to the NNPDF copy of the same
#: size, so those two cases re-ran the same code for the same answer.
SUITES = (
    "closure_NNPDF_truth",
    "closure_NNPDF_truth_small",
)


def _tagged_fields(cfg):
    """One stand-in field object per physical field, each tagged with its own name.

    Distinguishable sentinels, deliberately: a bare ``object()`` per field makes the
    basis-key -> field binding unreadable from the captured kwargs, which is what let a
    key-set-only assertion pass while every key was bound to the wrong field.
    """
    names = set(cfg.EVEN_MAP.values()) | set(cfg.ODD_MAP.values())
    return {name: SimpleNamespace(tag=name) for name in names}


def _tags(mapping):
    """``{basis_key: field.tag}`` for a captured ``maps_to``/``valence_maps_to``."""
    return {basis: field.tag for basis, field in mapping.items()}


def _capture_from_file(datasets, monkeypatch):
    """Replace ``ChargedCurrentSigmaR.from_file`` with a kwarg-capturing fake.

    Returns ``(captured, sentinel)``; the fake returns ``sentinel`` so a builder that
    silently constructs something else instead of forwarding is visible.
    """
    captured = {}
    sentinel = object()

    def fake_from_file(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        datasets.data.ChargedCurrentSigmaR, "from_file", fake_from_file
    )
    return captured, sentinel


def _builder_kwargs(suite, *, with_exp_nuisances):
    """Builder kwargs for one suite: only the full suites take ``with_systematics``."""
    kwargs = {"with_exp_nuisances": with_exp_nuisances, "t0": None}
    if not suite.endswith("_small"):
        kwargs["with_systematics"] = False
    return kwargs


@pytest.mark.parametrize("suite", SUITES)
def test_target_labels_reaching_the_builders_agree_with_pixels_alias_table(suite):
    """The ``TARGET_MAP`` asymmetry is safe because PIXEL normalizes both forms.

    The full suites expand ``p``/``d`` through ``generate.TARGET_MAP``; the
    ``_small`` suites define none and pass ``tab["target"]`` through.  This test
    is what makes that an *established* scope difference rather than an
    unexamined one, by pinning the three facts it rests on:

    1. **``TARGET_MAP`` is a strict subset of PIXEL's own alias table.**  Every
       key maps to exactly what :func:`pixel.util.flavor.normalize_target`
       returns, so the closure copy cannot drift into a second, disagreeing
       convention.  ``normalize_target`` is the real function reached from every
       DIS builder through ``_dis_common``; nothing is mocked.
    2. **Both label forms normalize to the same canonical target**, so passing an
       abbreviation through is not a defect -- ``normalize_target("p") ==
       normalize_target("proton")``.
    3. **Every target actually shipped in each suite's ``dis_manifest.json``
       normalizes without raising.**  Measured: the two full suites carry
       ``p``/``d`` on all 13 ``used`` tables, the two ``_small`` suites carry
       ``proton``/``deuteron`` on all 7 -- which is why only the full suites need
       a map.

    Control: an unknown label must raise, so this cannot be passing on a
    normalizer that accepts anything.
    """
    from pixel.util import flavor

    generate = importlib.import_module(f"{suite}.generate")
    target_map = getattr(generate, "TARGET_MAP", None)
    if suite.endswith("_small"):
        assert target_map is None, "the _small suites define no TARGET_MAP"
    else:
        assert target_map is not None
        for raw, expanded in target_map.items():
            assert flavor.normalize_target(raw) == expanded
            # The expansion is idempotent under PIXEL's own normalization.
            assert flavor.normalize_target(expanded) == expanded

    # Both forms are the same target as far as any builder is concerned.
    assert flavor.normalize_target("p") == flavor.normalize_target("proton")
    assert flavor.normalize_target("d") == flavor.normalize_target("deuteron")

    # Every label the shipped manifest actually carries is accepted.
    manifest = json.loads(generate.cfg.DIS_MANIFEST_PATH.read_text())
    used = [t for t in manifest["tables"] if t.get("status") == "used"]
    assert used, f"{suite}: no used DIS tables to check"
    seen = set()
    for table in used:
        raw = table["target"]
        canonical = flavor.normalize_target(raw)
        seen.add(canonical)
        if target_map is not None:
            assert flavor.normalize_target(target_map.get(raw, raw)) == canonical
    assert seen <= {"proton", "neutron", "deuteron"}

    # Control: the normalizer is not a pass-through that accepts everything.
    with pytest.raises(ValueError, match="proton, neutron, or deuteron"):
        flavor.normalize_target("carbon")


@pytest.mark.parametrize("suite", SUITES)
def test_cc_layout_keeps_beam_and_rowwise_inelasticity(suite, tmp_path, monkeypatch):
    """``exp_layout()`` keeps ``kind``/``beam``, skips retired tables, expands the
    target label in the full suites, and computes ``y`` per row -- from the explicit
    ``Y`` column when the table has one, otherwise from that row's own ``RS``.

    Builds a three-table ``sigma_r_cc`` manifest -- one ``status="used"`` table carrying
    only ``RS``, one ``status="retired"`` table, one ``status="used"`` table carrying
    *both* ``Y`` and ``RS`` -- and checks the parsed records against
    ``closure_NNPDF_truth/generate.py:527-565`` (the other three suites' ``exp_layout``
    implement the same branches; the ``_small`` pair omits ``TARGET_MAP``, see below).

    Four branches, each with the fixture built to make it observable:

    * **rowwise ``y = Q2 / (x * RS**2)``.**  ``x``/``Q2``/``RS`` all differ row to row
      (``Q2/x`` is 3e4 then 5e4; ``RS`` is 318 then 300), so the two rows' ``y`` are
      0.2967 and 0.5556.  Measured this pass, in-process against the real
      ``exp_layout`` re-exec'd from mutated source: a row-0 broadcast misses by
      ``2.5889e-01`` and a row-mean broadcast by ``1.2945e-01``, both ~14 orders above
      the ``atol=1e-15`` bar.  That bar itself stays a float-reassociation floor --
      ``(Q2/x)/RS**2`` here against ``Q2/(x*RS*RS)`` in the source, the same arithmetic
      regrouped, not an independent oracle, and measured to agree at exactly ``0.0`` on
      these inputs, so the bar is pure headroom.  It is the *fixture*, not the bar, that
      now carries "rowwise".
    * **explicit ``Y`` wins over ``RS``** (``generate.py:559-560``).  The third table
      supplies both; its ``Y`` is ``[0.65, 0.42]`` while its ``RS`` would give
      ``[0.2967, 0.2225]``, so a regression preferring ``RS`` (or ignoring ``Y``, which
      leaves the record with no ``y`` key at all) fails here.
    * **``status != "used"`` is skipped** (``generate.py:533-534``): the retired table is
      the only one whose label is absent from the parsed labels, and ``len(records)``
      pins that it was dropped rather than merely reordered.
    * **target expansion**: the manifest says ``"p"``.  The full suites map it through
      ``TARGET_MAP`` to ``"proton"``; the ``_small`` suites have no ``TARGET_MAP`` and
      pass ``"p"`` through unchanged (verified by reading all four ``exp_layout``
      copies).  Asserted per family, so either behaviour changing fails.

    Structural apart from the one ``y`` comparison.  Says nothing about whether the
    manifest's ``RS``/``Y`` values are themselves right, nor about anything downstream of
    the record dict.
    """
    generate = importlib.import_module(f"{suite}.generate")
    manifest_path = tmp_path / "dis_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "status": "used",
                        "kind": "sigma_r_cc",
                        "target": "p",  # abbreviated: TARGET_MAP input, full suites only
                        "label": "hera_cc_ep_318_sigmar",
                        "idx": 10031,
                        "beam": "e+",
                        # every row-varying input differs, so a broadcast is visible
                        "x": [0.01, 0.10],
                        "Q2": [300.0, 5000.0],
                        "RS": [318.0, 300.0],
                        "rel_stat": [0.1, 0.2],
                    },
                    {
                        "status": "retired",  # must be dropped entirely
                        "kind": "sigma_r_cc",
                        "target": "p",
                        "label": "hera_cc_retired_sigmar",
                        "idx": 10099,
                        "beam": "e-",
                        "x": [0.01],
                        "Q2": [300.0],
                        "RS": [318.0],
                        "rel_stat": [0.1],
                    },
                    {
                        "status": "used",
                        "kind": "sigma_r_cc",
                        "target": "p",
                        "label": "hera_cc_em_318_sigmar",
                        "idx": 10032,
                        "beam": "e-",
                        "x": [0.02, 0.04],
                        "Q2": [600.0, 900.0],
                        # both columns present: the explicit Y must win over RS,
                        # which would give [0.2967, 0.2225] instead.
                        "Y": [0.65, 0.42],
                        "RS": [318.0, 318.0],
                        "rel_stat": [0.1, 0.2],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(generate.cfg, "DIS_MANIFEST_PATH", manifest_path)

    records = generate.exp_layout()
    # A count, not just membership: pins that the retired table was dropped.
    assert len(records) == 2
    assert [rec["label"] for rec in records] == [
        "hera_cc_ep_318_sigmar",
        "hera_cc_em_318_sigmar",
    ]
    from_rs, from_y = records
    assert from_rs["kind"] == "sigma_r_cc"
    assert from_rs["beam"] == "e+"
    assert from_y["beam"] == "e-"
    # Full suites expand "p" through generate.TARGET_MAP; the _small suites have no
    # TARGET_MAP and pass the manifest label through unchanged.
    expected_target = "p" if suite.endswith("_small") else "proton"
    assert from_rs["target"] == expected_target
    assert from_y["target"] == expected_target
    np.testing.assert_allclose(
        from_rs["y"],
        np.array([300.0 / 0.01, 5000.0 / 0.10]) / np.array([318.0, 300.0]) ** 2,
        rtol=0.0,
        atol=1.0e-15,  # float-reassociation floor: (q2/x)/rs**2 here vs q2/(x*rs*rs) in source
    )
    # Explicit Y is passed through verbatim, not recomputed from RS.
    assert from_y["y"] == [0.65, 0.42]


@pytest.mark.parametrize("suite", SUITES)
def test_cc_builder_is_in_ordinary_dis_route(suite):
    """``sigma_r_cc`` is a first-class experimental builder, not a synthetic special case.

    Checks three registrations directly: ``"sigma_r_cc" in datasets._EXP_BUILDERS`` (the
    ordinary experimental-dataset dispatch, ``closure_NNPDF_truth/datasets.py:340-344``),
    ``"synthetic_cc" not in config.TEST_MODES`` (the obsolete synthetic-CC test mode stays
    removed -- ``plans/closure_ew_lhc_extension.md:1248`` records it was retired when real
    HERA CC data replaced it), and that dataset indices 10031/10032 carry
    ``kind == "sigma_r_cc"`` in each suite's own registry (``config.EXP_SPECS`` for the
    ``_small`` suites, ``config.CC_SIGMA_R_IDX`` for the full ones -- the two suite families
    use different registries, so the branch below is not optional).

    Purely structural: dict/tuple membership, function identity, and attribute equality,
    no numeric bar.  Would catch ``sigma_r_cc`` falling out of ``_EXP_BUILDERS`` (silently
    unrouted to the ordinary path), a regression reintroducing ``"synthetic_cc"`` into
    ``TEST_MODES``, or 10031/10032 losing their ``sigma_r_cc`` tag in one suite's config
    while keeping it in another's.

    The membership check alone could not catch the key being *bound to the wrong builder*
    (a copy-paste of the ``"sigma_r"`` line in the dict literal would route every closure
    CC dataset through the neutral-current builder, dropping the W3/valence physics, while
    ``"sigma_r_cc" in _EXP_BUILDERS`` stayed true), so the two ``is`` checks below pin the
    bound object: once on the dict literal (``datasets.py:340-344``) and once through the
    ``_experimental_builder`` dispatch that reads it, since a special case added ahead of
    the lookup could divert the kind before the dict is ever consulted.
    """
    config = importlib.import_module(f"{suite}.config")
    datasets = importlib.import_module(f"{suite}.datasets")
    assert "sigma_r_cc" in datasets._EXP_BUILDERS
    # Identity, not just presence: `in` on a dict never inspects the bound value.
    assert datasets._EXP_BUILDERS["sigma_r_cc"] is datasets.build_sigma_r_cc
    assert datasets._experimental_builder("sigma_r_cc") is datasets.build_sigma_r_cc
    assert "synthetic_cc" not in config.TEST_MODES
    if suite.endswith("_small"):  # small suites register kinds per-entry in EXP_SPECS
        specs = {spec.idx: spec for spec in config.EXP_SPECS}
        assert specs[10031].kind == "sigma_r_cc"
        assert specs[10032].kind == "sigma_r_cc"
    else:  # full suites carry a separate CC-only index tuple instead
        assert config.CC_SIGMA_R_IDX == (10031, 10032)


@pytest.mark.parametrize("suite", SUITES)
def test_stale_f3_proxy_manifest_is_rejected_with_regeneration_instruction(suite):
    """A manifest still tagged the retired ``"f3_proxy"`` kind fails loudly, not silently.

    ``"f3_proxy"`` was the synthetic HERA-CC stand-in used before real ``sigma_r_cc`` data was
    wired in (module docstring; ``plans/closure_ew_lhc_extension.md:1248``).
    ``datasets._experimental_builder`` special-cases it ahead of the ordinary
    ``_EXP_BUILDERS`` lookup (``closure_NNPDF_truth/datasets.py:347-352``) and raises
    ``RuntimeError`` naming the fix.  This calls it directly and checks both that it raises
    and that the message still contains ``"--remake-data"``.

    Would catch ``_experimental_builder`` silently falling through to a stale or default
    builder for ``"f3_proxy"`` instead of raising, or the raised message losing its
    regeneration instruction.  Structural (exception type + message substring); no numeric
    bar.  Scope: only this closure-driver path is checked.  ``src/pixel/api/manifest.py``'s
    separate closure-manifest translator (``_closure_record``, ``api/manifest.py:171``) still
    lists ``"f3_proxy"`` and has no ``"sigma_r_cc"`` branch at all -- a different, apparently
    un-migrated code path this test does not touch.
    """
    datasets = importlib.import_module(f"{suite}.datasets")
    with pytest.raises(RuntimeError, match="--remake-data"):
        datasets._experimental_builder("f3_proxy")


@pytest.mark.parametrize("suite", SUITES)
def test_cc_dataset_builder_receives_even_valence_beam_and_y(
    suite, tmp_path, monkeypatch
):
    """``build_sigma_r_cc`` forwards beam, y, the basis maps, and order to ``from_file``.

    Replaces ``ChargedCurrentSigmaR.from_file`` (``src/pixel/data/experimental/
    charged_current.py``) with a kwarg-capturing fake and calls ``datasets.build_sigma_r_cc``
    (``closure_NNPDF_truth/datasets.py:223-239``) on a hand-built record.  ``beam_charge`` and
    ``y`` are checked by value, so a wrong-value passthrough on either fails here.  ``order``
    is checked against ``cfg.ORDER`` (a plain string passthrough, no transformation).

    ``maps_to``/``valence_maps_to`` are checked **per key**: every field handed in is a
    ``SimpleNamespace`` tagged with its own name, so
    ``{basis: field.tag for basis, field in captured["maps_to"].items()}`` reconstructs the
    binding ``_even_map`` actually built (``datasets.py:37-42``) and is compared against
    ``cfg.EVEN_MAP`` entry by entry.  The earlier form of this check compared *key sets*
    (``set(captured["maps_to"]) == set(cfg.EVEN_MAP)``), which was measured blind to a map
    binding all five basis keys to one field object and to a reversed key/value binding --
    both keep the key set identical while corrupting every CC dataset's flavour content.
    The ``is`` checks below additionally pin that the objects passed through are the very
    field instances the analysis binds to, not copies.

    The two ``EVEN_MAP``/``ODD_MAP`` bindings are genuinely distinguishable here: they share
    four basis keys, and ``"singlet"`` maps to ``sigma`` in the even map but to ``v`` in the
    odd one, so swapping ``maps_to`` and ``valence_maps_to`` fails on both the key set and
    the per-key tags.

    Scope: this record always supplies ``y`` and carries no ``rel_norm``, so the
    ``elif "sqrt_s" in rec`` branch and the ``with_exp_nuisances`` gate are covered by the
    two tests below instead.  Because the fake absorbs arbitrary kwargs, this test cannot
    catch a kwarg renamed in the real ``from_file``/``from_arrays`` signature; that path is
    exercised unmocked, but not through this builder, in ``tests/test_charged_current_dis.py``.
    """
    datasets = importlib.import_module(f"{suite}.datasets")
    fields = _tagged_fields(datasets.cfg)
    captured, sentinel = _capture_from_file(datasets, monkeypatch)
    rec = {
        "kind": "sigma_r_cc",
        "target": "proton",
        "beam": "e-",
        "file": "exp/hera_cc_em_318_sigmar.dat",
        "y": [0.25],
    }
    result = datasets.build_sigma_r_cc(
        rec, tmp_path / "cc.dat", fields, None,
        **_builder_kwargs(suite, with_exp_nuisances=False),
    )

    assert result is sentinel
    assert captured["beam_charge"] == "e-"
    assert captured["y"] == [0.25]
    # Per key, not per key set: which physical field landed under each basis key.
    assert _tags(captured["maps_to"]) == dict(datasets.cfg.EVEN_MAP)
    assert _tags(captured["valence_maps_to"]) == dict(datasets.cfg.ODD_MAP)
    # The field *instances* are forwarded, so the analysis binds to the same objects.
    for basis, name in datasets.cfg.EVEN_MAP.items():
        assert captured["maps_to"][basis] is fields[name]
    for basis, name in datasets.cfg.ODD_MAP.items():
        assert captured["valence_maps_to"][basis] is fields[name]
    assert captured["order"] == datasets.cfg.ORDER


@pytest.mark.parametrize("suite", SUITES)
def test_cc_builder_falls_back_to_sqrt_s_when_the_record_has_no_y(
    suite, tmp_path, monkeypatch
):
    """A record without ``y`` reaches ``from_file`` as ``sqrt_s``, cast to ``float``.

    ``build_sigma_r_cc``'s ``if "y" in rec: ... elif "sqrt_s" in rec:``
    (``closure_NNPDF_truth/datasets.py:226-229``, byte-identical in all four suites) has a
    second leg no test reached before: every builder test here supplied ``y``.  This calls
    the builder on a record carrying only ``sqrt_s`` -- as a *string*, so the ``float(...)``
    cast is observable and not merely a passthrough -- and checks that ``sqrt_s`` arrives as
    ``318.0`` and that no ``y`` kwarg is invented.

    Structural (kwarg presence, one exact float); no numeric bar.  Catches the ``elif``
    branch being dropped (``sqrt_s`` silently not forwarded, leaving the builder to guess an
    inelasticity), the cast being dropped (a string reaching the real ``from_file``), or the
    two branches being made non-exclusive.  Says nothing about how ``ChargedCurrentSigmaR``
    then converts ``sqrt_s`` into per-row kinematics -- that is
    ``tests/test_charged_current_dis.py``'s subject, and it is mocked out here.
    """
    datasets = importlib.import_module(f"{suite}.datasets")
    captured, sentinel = _capture_from_file(datasets, monkeypatch)
    rec = {
        "kind": "sigma_r_cc",
        "target": "proton",
        "beam": "e+",
        "file": "exp/hera_cc_ep_318_sigmar.dat",
        "sqrt_s": "318.0",  # a string, so the float() cast in the source is observable
    }
    result = datasets.build_sigma_r_cc(
        rec, tmp_path / "cc.dat", _tagged_fields(datasets.cfg), None,
        **_builder_kwargs(suite, with_exp_nuisances=False),
    )

    assert result is sentinel
    assert "y" not in captured
    assert captured["sqrt_s"] == 318.0
    assert isinstance(captured["sqrt_s"], float)


@pytest.mark.parametrize("suite", SUITES)
@pytest.mark.parametrize("with_exp_nuisances", [False, True])
def test_cc_builder_forwards_the_normalization_only_with_exp_nuisances(
    suite, with_exp_nuisances, tmp_path, monkeypatch
):
    """``with_exp_nuisances`` gates the normalization kwargs, on a record that has one.

    ``_exp_nuisance_kwargs`` (``closure_NNPDF_truth/datasets.py:154-188``) returns ``{}``
    immediately when ``enabled`` is false -- the generation path, where the forward operator
    must stay purely physical -- and otherwise turns ``rec["rel_norm"]`` into
    ``normalization``/``fit_normalization``.  On a record with no ``rel_norm`` and no
    ``correlated_file`` the two settings are indistinguishable: measured, the identical
    (empty) captured-kwargs dict comes back either way, which is why the neighbouring
    builder tests cannot cover this gate.  This record carries ``rel_norm=0.015``, so the
    two parametrized cases now differ.

    Structural (kwarg presence plus one exact float); no numeric bar.  Catches the
    ``enabled`` early return being dropped (a nuisance leaking into the generation-side
    operator, which would make the injected truth and the fitted model disagree by
    construction), ``rel_norm`` being read from the wrong key, or ``fit_normalization``
    being hardcoded instead of read from ``cfg.DIS_FIT_NORMALIZATION``.  The
    ``correlated_file`` leg is still uncovered here: it loads an ``.npz`` sidecar from disk,
    which this record deliberately does not have.
    """
    datasets = importlib.import_module(f"{suite}.datasets")
    captured, sentinel = _capture_from_file(datasets, monkeypatch)
    rec = {
        "kind": "sigma_r_cc",
        "target": "proton",
        "beam": "e-",
        "file": "exp/hera_cc_em_318_sigmar.dat",
        "y": [0.25],
        "rel_norm": 0.015,  # the only input that makes the two cases differ
    }
    result = datasets.build_sigma_r_cc(
        rec, tmp_path / "cc.dat", _tagged_fields(datasets.cfg), None,
        **_builder_kwargs(suite, with_exp_nuisances=with_exp_nuisances),
    )

    assert result is sentinel
    if with_exp_nuisances:
        assert captured["normalization"] == 0.015
        assert captured["fit_normalization"] == datasets.cfg.DIS_FIT_NORMALIZATION
    else:
        assert "normalization" not in captured
        assert "fit_normalization" not in captured
