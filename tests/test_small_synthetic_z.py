"""The small EW smoke mode is isolated and unmistakably synthetic.

``z`` names the Z boson here (the manifest record's ``"boson": "gamma_z"``), not
the lattice Wilson-line separation this repo spells the same letter in
``kernels/lattice.py`` (``_separation_fm``, ``finite_volume_z``) and
``data/lattice/`` -- the two senses of ``z`` never collide in this file because it
builds only a Drell-Yan dataset and never constructs a lattice one.

Exercises the ``synthetic_z_plumbing`` branch of
``closure_JAM_truth_small.datasets.build_datasets`` and
``closure_NNPDF_truth_small.datasets.build_datasets`` -- past that, the real
``pixel.data.drell_yan.DrellYan``/``DrellYanData`` factory (imported by
``datasets.build_drell_yan``), with ``source``/``component`` read back off
``DatasetMetadata`` (``pixel.core.model``).

Both tests write ``manifest.json`` by hand inside ``tmp_path`` and monkeypatch
``config.truth_dir`` to read it, so neither calls ``generate.py`` at all: this file
says nothing about ``closure_JAM_truth_small.generate.dy_central`` /
``closure_NNPDF_truth_small.generate.dy_central``, which compute the DY truth
central value as an ensemble second moment (``C = qA.T @ qB / n_rep``,
``generate.py:265`` in both copies) -- a law ``closure_JAM_truth/generate.py:494``
documents as "deliberately forbidden" for a bilinear observable. See
``tests/test_closure_truth_representable.py`` for the measured 7-8% consequence;
this file's manifest ``mean``/``cov`` are literals the test itself writes, not
``generate.py`` output, so it is silent on that defect rather than evidence either
way.

The real generator is not a pure bystander, though: ``generate.py``'s own
``synthetic_z_layout()`` (JAM/NNPDF-identical, line 217) is the production source
of the record this file hand-writes, and ``generate_member``'s synthetic_z block
(``generate.py:469-493``) folds it through this same ``build_drell_yan`` and the
``dy_central`` the paragraph above flags, then writes the result back into
``manifest.json`` under the identical ``"synthetic_z"`` key. No test calls
``synthetic_z_layout`` or that block (grepped repo-wide: only the definition and
its one call site in each copy of ``generate.py``). This file's hand-written
record matches that function's current field set by inspection, not by a shared
check, so a schema change on either side -- a renamed or dropped manifest key --
would drift silently rather than fail here or there.

Every assertion here is exact string/int equality or a ``pytest.raises`` message
match against literals the test wrote a few lines earlier -- there is no numeric
physics claim and no independent oracle. This is plumbing/contract coverage (a
manifest field reaches the right dataset attribute; a malformed manifest raises),
not DY kernel or evolution physics -- see ``tests/test_closure_dy_evolution.py`` and
``tests/test_dy_dataset.py`` for that, neither of which touches the
closure_*_small packages or this file's synthetic_z manifest path.

The first test was parametrized over ``(jam_config, jam_datasets)`` and
``(nnpdf_config, nnpdf_datasets)`` until 2026-08-14, when the JAM case was dropped
on the owner's instruction.  It ran byte-identical code both times: ``build_datasets``
and ``build_drell_yan`` are unchanged between ``closure_JAM_truth_small/datasets.py``
and ``closure_NNPDF_truth_small/datasets.py`` (diffed directly; the only
differences anywhere in the file are a docstring cross-reference and an unrelated
kwarg-line reorder inside ``build_pseudoitd``), and every DY-relevant ``config.py``
constant (``DY_ORDER``, ``DY_ALPHA_S``, ``DY_NF``, ``DY_CHANNELS``, the parton/basis
maps) matches too -- so the second case was the same wiring check run twice, and the
test's own docstring already told the reader to "read this as one case, not two".
The JAM ``_small`` copy is still reached from this file: the second test (the
manifest-validation guards) is not parametrized and drives ``jam_config``/
``jam_datasets`` directly, which is why both imports remain.

``synthetic_z_plumbing``/``TEST_MODE_SYNTHETIC_Z`` exist only in the ``_small``
suites: ``closure_JAM_truth`` and ``closure_NNPDF_truth`` (the full, non-``_small``
packages) define neither, so this file's silence on those two packages is total
exclusion by construction, not a scoping gap -- there is no synthetic_z mode there
for it to miss.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from closure_JAM_truth_small import config as jam_config
from closure_JAM_truth_small import datasets as jam_datasets
from closure_JAM_truth_small import generate as jam_generate
from closure_NNPDF_truth_small import config as nnpdf_config
from closure_NNPDF_truth_small import datasets as nnpdf_datasets
from closure_NNPDF_truth_small import generate as nnpdf_generate

#: The two ``_small`` packages, for the source-identity test below and nothing
#: else.  ``synthetic_z_plumbing`` exists ONLY in the ``_small`` suites (the full
#: ``closure_JAM_truth``/``closure_NNPDF_truth`` define neither the mode nor
#: ``synthetic_z_layout``), so "all the copies" is two here, not four.
#:
#: **Must keep both entries.**  Its one reader,
#: ``test_both_small_suites_share_one_synthetic_z_code_path``, is the *replacement*
#: for the JAM/NNPDF behavioural sweep the owner stripped on 2026-08-14: rather
#: than running byte-identical ``build_datasets`` twice for the same answer, the
#: identity is asserted directly on the source text.  Trimming this to one entry
#: would leave that test comparing a package to itself -- still passing, and blind
#: to the divergence it exists to catch.
ALL_SMALL_SUITES = (
    (jam_config, jam_datasets, jam_generate),
    (nnpdf_config, nnpdf_datasets, nnpdf_generate),
)


#: Base manifest record for the tests added below, hand-written on purpose: the
#: whole point of
#: ``test_generation_time_synthetic_z_layout_is_what_the_manifest_reader_consumes``
#: is to compare a hand-written schema against ``generate.synthetic_z_layout()``,
#: so deriving this from the generator would make that comparison vacuous.  Each
#: test below copies it and changes exactly the keys it is about.
_SYNTHETIC_Z_RECORD = {
    "kind": "drell_yan",
    "reaction": "pp",
    "label": "synthetic_z_plumbing",
    "name": "synthetic_z_plumbing",
    "S": [7000.0 ** 2],
    "Q2": [91.1876 ** 2],
    "Y": [0.0],
    "boson": "gamma_z",
    "classification": "synthetic_proxy",
    "observable_contract": "inclusive_boson_level_pointwise_unevolved",
    "physics_coverage": False,
    "mean": [1.0],
    "cov": [[0.01]],
}


def _write_manifest(tmp_path, monkeypatch, config, record, *, key="synthetic_z"):
    """Point ``config.truth_dir`` at a ``tmp_path`` manifest holding ``record``.

    ``record`` may be a dict (wrapped in a one-element list) or an explicit list,
    so the "exactly one" guard can be fed 0, 1 or 2 records.
    """
    truth_dir = tmp_path / config.truth_label("mc")
    truth_dir.mkdir(exist_ok=True)
    records = record if isinstance(record, list) else [record]
    (truth_dir / "manifest.json").write_text(
        json.dumps({"lattice": [], "exp": [], "dy": [], key: records})
    )
    monkeypatch.setattr(config, "truth_dir", lambda _q: truth_dir)
    return truth_dir


def _build_synthetic_z(datasets, config):
    """Run ``build_datasets`` in synthetic_z-only mode against the written manifest."""
    return datasets.build_datasets(
        "mc",
        config.make_fields(),
        include_lattice=False,
        include_exp=False,
        include_dy=False,
        include_synthetic_z=True,
        use_kernel_cache=False,
    )


# NNPDF only since 2026-08-14 (owner's instruction); the JAM `_small` case ran
# byte-identical `build_datasets`/`build_drell_yan` over matching config constants --
# see the module docstring.  Kept as a one-entry parametrize rather than inlined, so
# re-adding a suite is a one-line change.
@pytest.mark.parametrize(
    ("config", "datasets"),
    [
        (nnpdf_config, nnpdf_datasets),
    ],
)
def test_small_synthetic_z_mode_builds_only_one_labeled_proxy(
    tmp_path, monkeypatch, config, datasets
):
    """One hand-written manifest record survives ``build_datasets`` labeled and alone.

    Checks ``built[0].name``/``.source``/``.component``/``.n_data`` against the
    literal ``name``/``classification``/``observable_contract``/row-count this test
    wrote into the manifest below -- the oracle is the fixture itself, not an
    independent computation (``D1``) -- plus ``len(built) == 1`` and a nonempty
    ``bilinear_contributions``. Catches a wiring bug that drops or swaps a manifest
    field in ``build_datasets``/``build_drell_yan``, or an emptied channel/field map.

    Does NOT check ``built[0].mean``/``.cov``: building the identical dataset with
    ``mean=None, cov=None`` in place of the manifest's ``[1.0]``/``[[0.01]]`` passes
    every assertion below unchanged (measured directly against
    ``pixel.data.DrellYan``; see the audit report's ``measurements``). This ran as
    two parametrized cases over byte-identical code until 2026-08-14, with the
    instruction "read this as one case, not two"; it is now literally one.

    The manifest's ``"name"`` and ``"label"`` are both the literal string
    ``"synthetic_z_plumbing"``, so ``built[0].name == "synthetic_z_plumbing"``
    cannot tell ``build_drell_yan``'s ``rec.get("name", rec["label"])`` apart from
    a bug that read ``label`` where ``name`` was intended (or vice versa) -- the
    fixture is degenerate in exactly the dimension this assertion probes
    (tests/README.md rule 9's second failure class).

    Seven assertions run in sequence and the first failure hides the rest. The
    first, ``TEST_MODE_SYNTHETIC_Z in TEST_MODES``, checks a static ``config.py``
    fact that does not depend on ``built`` at all -- it would fail identically
    whether the manifest plumbing above it is entirely broken or entirely working,
    so a failure there says nothing about the five field checks and the
    ``bilinear_contributions`` check that never ran. A standalone
    ``test_synthetic_z_mode_is_a_registered_test_mode`` (no ``tmp_path`` or
    monkeypatch needed) plus separate tests for ``len(built) == 1``, the four
    field pass-throughs, and ``bilinear_contributions`` would let each fail
    independently instead of in one bundle.
    """
    truth_dir = tmp_path / config.truth_label("mc")
    truth_dir.mkdir()
    record = {
        "kind": "drell_yan",
        "reaction": "pp",
        "label": "synthetic_z_plumbing",
        "name": "synthetic_z_plumbing",
        "S": [7000.0**2],
        "Q2": [91.1876**2],
        "Y": [0.0],
        "boson": "gamma_z",
        "classification": "synthetic_proxy",
        "observable_contract": "inclusive_boson_level_pointwise_unevolved",
        "physics_coverage": False,
        # Never asserted on the built dataset below (see docstring): a build that
        # dropped or corrupted mean/cov would still pass every check in this test.
        "mean": [1.0],
        "cov": [[0.01]],
    }
    (truth_dir / "manifest.json").write_text(
        json.dumps({"lattice": [], "exp": [], "dy": [], "synthetic_z": [record]})
    )
    monkeypatch.setattr(config, "truth_dir", lambda _q: truth_dir)

    built = datasets.build_datasets(
        "mc",
        config.make_fields(),
        include_lattice=False,
        include_exp=False,
        include_dy=False,
        include_synthetic_z=True,
        use_kernel_cache=False,
    )

    # TEST_MODE_SYNTHETIC_Z and TEST_MODES are defined a few lines apart in
    # config.py; this also gates fit.py's mode guard and the run_closure.py /
    # plot_ratios.py / plot_datasets.py --mode(s) CLI choices.
    assert config.TEST_MODE_SYNTHETIC_Z in config.TEST_MODES
    assert len(built) == 1
    assert built[0].name == "synthetic_z_plumbing"
    assert built[0].n_data == 1
    assert built[0].source == "synthetic_proxy"
    assert built[0].component == "inclusive_boson_level_pointwise_unevolved"
    # Nonempty only: confirms some channel/field-map combination assembled, not
    # that the full DY_CHANNELS x field-pair cross product is complete or correct.
    assert built[0].bilinear_contributions


def test_small_synthetic_z_rejects_missing_or_unclassified_manifest(tmp_path, monkeypatch):
    """Both ``build_datasets`` synthetic_z guard clauses raise their documented text.

    Zero records raises ``ValueError`` matching "exactly one" (the
    ``len(records) != 1`` check); a record with ``classification="physics"`` raises
    matching "synthetic_proxy" (the ``!= "synthetic_proxy"`` check). Oracle is the
    literal message text in ``closure_JAM_truth_small/datasets.py`` (``F1``,
    structural) -- catches either guard being removed or reworded.

    Does NOT catch the "exactly one" guard weakening to "at least one" (e.g.
    ``if not records``): only the 0-record arm is exercised, never 2+ records. Not
    parametrized over NNPDF, unlike the sibling test above --
    ``closure_NNPDF_truth_small.datasets`` carries the identical two guard clauses
    with no test coverage at all.

    The two ``pytest.raises`` blocks are independent guard checks run back to
    back. If the first regressed -- stopped raising, or started raising some
    other exception type -- the ``with pytest.raises(ValueError, match="exactly
    one")`` block itself would fail the test right there, and the second guard
    (the "physics"-classification case) would never run in that session. Splitting
    into ``test_small_synthetic_z_rejects_empty_manifest`` and
    ``test_small_synthetic_z_rejects_unclassified_manifest`` would let each guard
    fail on its own.
    """
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    monkeypatch.setattr(jam_config, "truth_dir", lambda _q: truth_dir)
    fields = jam_config.make_fields()

    # Empty list hits `len(records) != 1`; a manifest missing the "synthetic_z" key
    # entirely would hit the identical branch via .get("synthetic_z", []).
    (truth_dir / "manifest.json").write_text(json.dumps({"synthetic_z": []}))
    with pytest.raises(ValueError, match="exactly one"):
        jam_datasets.build_datasets(
            "mc", fields, include_lattice=False, include_exp=False,
            include_synthetic_z=True, use_kernel_cache=False,
        )

    # classification="physics" (present, wrong) exercises the same branch an absent
    # key would. The record is otherwise minimal: the guard raises before
    # build_drell_yan would need "name"/"S"/"Q2"/"Y"/"boson".
    (truth_dir / "manifest.json").write_text(json.dumps({"synthetic_z": [{
        "label": "bad", "reaction": "pp", "classification": "physics"
    }]}))
    with pytest.raises(ValueError, match="synthetic_proxy"):
        jam_datasets.build_datasets(
            "mc", fields, include_lattice=False, include_exp=False,
            include_synthetic_z=True, use_kernel_cache=False,
        )


def test_synthetic_z_rejects_two_records(tmp_path, monkeypatch):
    """Two records hit the *same* "exactly one" guard as zero records.

    The sibling guard test only ever builds a 0-record or 1-record manifest, so a
    regression weakening ``len(records) != 1`` to ``if not records`` -- which
    accepts 2+ and then silently builds only ``records[0]`` -- passes every other
    test in this file.  This is the other arm of that guard.

    Oracle ``F1``, matched on the message text rather than the type: this reader
    raises ``ValueError`` from a second guard a few lines below
    (``classification != "synthetic_proxy"``), and both records here are
    *correctly* classified, so a type-only match would have accepted the wrong
    guard firing.  The control is the second half: the identical first record on
    its own builds one dataset without raising, so the refusal above is about the
    count and nothing else.
    """
    good = dict(_SYNTHETIC_Z_RECORD)
    other = dict(good, label="synthetic_z_second", name="synthetic_z_second")

    _write_manifest(tmp_path, monkeypatch, nnpdf_config, [good, other])
    with pytest.raises(ValueError, match="exactly one"):
        _build_synthetic_z(nnpdf_datasets, nnpdf_config)

    # Control: drop the second record and the identical call succeeds.
    _write_manifest(tmp_path, monkeypatch, nnpdf_config, [good])
    assert len(_build_synthetic_z(nnpdf_datasets, nnpdf_config)) == 1


def test_synthetic_z_mean_and_cov_reach_the_built_dataset(tmp_path, monkeypatch):
    """The manifest's measured vector and covariance survive into the ``Dataset``.

    ``build_drell_yan`` forwards ``mean``/``cov`` only when *both* keys are
    present (``closure_*_small/datasets.py``'s ``if "mean" in rec and "cov" in
    rec``).  MEASURED for the audit: rebuilding the identical record with those
    two keys removed -- ``DrellYan`` then defaults ``mean=None``, ``cov=None`` --
    leaves ``name``/``source``/``component``/``n_data`` and the 360-entry
    ``bilinear_contributions`` list bit-for-bit unchanged, so every assertion in
    the sibling plumbing test passes on a build that dropped the data entirely.

    Oracle ``D1``: the values are literals this test wrote into the manifest a
    few lines earlier -- but at values (``[2.5]``, ``[[0.16]]``) that are neither
    ``DrellYan``'s defaults nor the sibling test's ``[1.0]``/``[[0.01]]``, so a
    hardcoded or stale-cached vector fails.  ``assert_array_equal``: these are
    JSON round-tripped literals, not computed numbers, so exact equality is the
    right bar and a tolerance would only hide a corruption.

    Also asserts the ``mean``/``cov``-absent branch is what the guard says it is
    (both ``None``), so the "both keys or neither" contract is pinned rather than
    assumed.
    """
    record = dict(_SYNTHETIC_Z_RECORD, mean=[2.5], cov=[[0.16]])
    _write_manifest(tmp_path, monkeypatch, nnpdf_config, record)
    built = _build_synthetic_z(nnpdf_datasets, nnpdf_config)[0]
    np.testing.assert_array_equal(np.asarray(built.mean), np.array([2.5]))
    np.testing.assert_array_equal(np.asarray(built.cov), np.array([[0.16]]))

    # The other side of `if "mean" in rec and "cov" in rec`: no data at all.
    bare = {k: v for k, v in record.items() if k not in ("mean", "cov")}
    _write_manifest(tmp_path, monkeypatch, nnpdf_config, bare)
    empty = _build_synthetic_z(nnpdf_datasets, nnpdf_config)[0]
    assert empty.mean is None and empty.cov is None


def test_synthetic_z_manifest_keys_map_to_distinct_dataset_attributes(
    tmp_path, monkeypatch
):
    """``name``/``label`` and ``observable_contract``/``reaction`` are read apart.

    ``build_drell_yan`` reads ``name=rec.get("name", rec["label"])`` and
    ``component=rec.get("observable_contract", rec["reaction"])``.  Every other
    fixture in this file sets ``name`` and ``label`` to the *same* literal, so
    ``built[0].name == "synthetic_z_plumbing"`` cannot tell the correct read from
    one that took the wrong key of the pair -- the fixture is degenerate in
    exactly the dimension the assertion probes (``tests/README.md`` rule 9's
    second failure class).  Here all four values differ.

    Oracle ``D1`` (fixture echo), but a non-degenerate one: the record's
    ``label`` and ``reaction`` are asserted to be *different* from the values
    expected on the dataset, so a swapped read fails rather than coinciding.
    ``reaction`` must stay ``"pp"`` -- ``cfg.dy_field_maps`` dispatches on it --
    which is why the contrast is made on ``observable_contract`` instead.
    """
    record = dict(
        _SYNTHETIC_Z_RECORD,
        label="synthetic_z_LABEL",
        name="synthetic_z_NAME",
        observable_contract="contract_under_test",
    )
    assert record["name"] != record["label"]          # the fixture is non-degenerate
    assert record["observable_contract"] != record["reaction"]

    _write_manifest(tmp_path, monkeypatch, nnpdf_config, record)
    built = _build_synthetic_z(nnpdf_datasets, nnpdf_config)[0]
    assert built.name == "synthetic_z_NAME"           # not "synthetic_z_LABEL"
    assert built.component == "contract_under_test"   # not "pp"


def test_synthetic_z_optional_keys_fall_back_to_their_partners(tmp_path, monkeypatch):
    """With ``name``/``observable_contract`` absent, the ``.get`` fallbacks fire.

    The second half of the pair above.  ``build_drell_yan``'s two ``rec.get(a,
    rec[b])`` defaults are never taken by any other test in this file, because
    every record it writes supplies both optional keys -- so a regression turning
    either into a plain ``rec[a]`` (``KeyError`` on a real generated manifest that
    omits it) or into a wrong-partner fallback goes unnoticed.

    Oracle ``F1``/``D1``: the two fallback partners are asserted to be *different
    strings* from the keys just removed, so "fell back correctly" is
    distinguishable from "read the removed key from somewhere else".  Catches the
    fallback being dropped, or pointed at the wrong partner key.
    """
    record = {
        k: v for k, v in _SYNTHETIC_Z_RECORD.items()
        if k not in ("name", "observable_contract")
    }
    record["label"] = "fallback_label"
    assert "name" not in record and "observable_contract" not in record
    assert record["label"] != record["reaction"]      # the two fallbacks differ

    _write_manifest(tmp_path, monkeypatch, nnpdf_config, record)
    built = _build_synthetic_z(nnpdf_datasets, nnpdf_config)[0]
    assert built.name == "fallback_label"             # name <- label
    assert built.component == "pp"                    # component <- reaction


def test_generation_time_synthetic_z_layout_is_what_the_manifest_reader_consumes(
    tmp_path, monkeypatch
):
    """``generate.synthetic_z_layout()`` and ``build_datasets`` agree on one schema.

    ``synthetic_z_layout()`` is the *production* source of the record every other
    test in this file hand-writes, and ``generate_member`` folds it through the
    same ``build_drell_yan`` before writing ``_public(rec)`` plus ``mean``/``cov``
    into ``manifest.json`` under the identical ``"synthetic_z"`` key
    (``closure_*_small/generate.py:563-582``).  Nothing called
    ``synthetic_z_layout`` or that block, so the hand-written literals could keep
    exercising an old schema forever while the generator moved on: a renamed or
    dropped key on either side would drift silently rather than fail anywhere.

    This test closes that without running the generator: it takes the real layout
    record, applies ``_public`` and the two keys ``generate_member`` adds, and
    drives it through the reader.  A key the generator stopped emitting, or one
    the reader started requiring, now fails here.

    Oracle ``A3`` -- an identity between two objects the suite itself builds, on
    the two routes that must not diverge:

    * **generation time** ``build_drell_yan(rec, None, fields, None)``, the exact
      call at ``generate.py:566``, on the raw layout record; and
    * **fitting time** ``build_datasets(..., include_synthetic_z=True)`` reading
      the manifest form of the same record.

    Both must produce one forward operator.  MEASURED on this record: 360
    bilinear contributions, whose ``(field_A, field_B)`` pairs and weights agree
    **exactly** between the two routes, so ``assert_array_equal`` on the weights
    is the right bar rather than a tolerance.  The pair list is compared as an
    ordered sequence, so a reordering that changed which weight attaches to which
    field pair also fails.

    The route-identity half alone would NOT see a generator-side rename, since
    both routes read the same record and would fall back together -- so the
    layout's own contract values are asserted too, and that is where the
    detection against the generator sits.  Two of them have an external oracle
    (``C1``) rather than being fixture echoes: ``sqrt(Q2)`` is the PDG Z mass
    ``91.1876 GeV`` and ``sqrt(S)`` is the 7 TeV LHC energy, which is what makes
    the mode a *Z* plumbing point at all.  MEASURED: both round-trip to
    ``<1e-12`` relative.  ``classification``/``physics_coverage``/``boson`` are
    the "unmistakably synthetic" claim in the module docstring, stated as
    assertions instead of prose.

    NOT covered here: ``dy_central``, which ``generate_member`` uses to build the
    ``mean`` this test supplies as a literal -- that is the ``S0-05``
    ensemble-second-moment question, recorded as the owner's decision (see the
    module docstring and ``tests/test_closure_truth_representable.py``).
    """
    layout = nnpdf_generate.synthetic_z_layout()
    assert len(layout) == 1, layout      # the reader's "exactly one" is satisfiable
    rec = layout[0]
    # _public drops leading-underscore layout keys; this record has none, so the
    # manifest form is the layout form plus mean/cov.  Pin that, since a future
    # private key would silently change what the reader sees.
    assert nnpdf_generate._public(rec) == rec, sorted(rec)
    manifest_rec = dict(nnpdf_generate._public(rec), mean=[1.5], cov=[[0.04]])

    # The layout's own contract, asserted rather than assumed: a generator-side
    # rename or value drift fails here (KeyError or a wrong value), where the
    # route-identity check below cannot see it.
    assert rec["classification"] == "synthetic_proxy"
    assert rec["physics_coverage"] is False
    assert rec["boson"] == "gamma_z"
    assert rec["observable_contract"] == "inclusive_boson_level_pointwise_unevolved"
    # C1: the PDG Z mass and the 7 TeV LHC energy -- external constants, not
    # echoes of anything this test wrote.  Measured: both agree to < 1e-12.
    assert abs(float(np.sqrt(rec["Q2"][0])) / 91.1876 - 1.0) < 1e-12, rec["Q2"]
    assert abs(float(np.sqrt(rec["S"][0])) / 7000.0 - 1.0) < 1e-12, rec["S"]

    fields = nnpdf_config.make_fields()
    # Generation-time route, exactly as generate.py:566 calls it.
    gen_ds = nnpdf_datasets.build_drell_yan(rec, None, fields, None)

    # Fitting-time route, through the manifest and both guard clauses.
    _write_manifest(tmp_path, monkeypatch, nnpdf_config, manifest_rec)
    fit_ds = _build_synthetic_z(nnpdf_datasets, nnpdf_config)[0]

    assert (fit_ds.name, fit_ds.source, fit_ds.component) == (
        gen_ds.name, gen_ds.source, gen_ds.component
    )
    assert fit_ds.n_data == gen_ds.n_data == 1
    # One forward operator, term by term (360 measured), order included.
    gen_terms = [(c.field_A, c.field_B) for c in gen_ds.bilinear_contributions]
    fit_terms = [(c.field_A, c.field_B) for c in fit_ds.bilinear_contributions]
    assert len(gen_terms) > 100 and fit_terms == gen_terms
    np.testing.assert_array_equal(
        np.array([c.weight for c in fit_ds.bilinear_contributions]),
        np.array([c.weight for c in gen_ds.bilinear_contributions]),
    )


def test_both_small_suites_share_one_synthetic_z_code_path():
    """The two ``_small`` copies are one code path -- assert it, don't assume it.

    **This test is why ``ALL_SMALL_SUITES`` exists and must keep both entries.**
    It is that tuple's only reader; every behavioural test in this file drives one
    suite, NNPDF for the parametrized case and JAM for the guard case.

    It is also the *replacement* for the JAM/NNPDF double run the owner stripped
    on 2026-08-14, and a stronger guard than what it replaces.  The old argument
    for the 2x sweep was that ``build_datasets``/``build_drell_yan`` were
    byte-identical between the copies -- established by ``diff`` during the audit
    and then written into the module docstring as *prose*, which goes stale
    silently.  Running identical code twice for the same answer does not detect a
    divergence either; only comparing the copies does.  That comparison is here,
    on the source text of all three functions in play plus the DY-relevant config
    constants this file's assertions actually depend on.

    Oracle ``A3`` (identity between objects the suites themselves define);
    structural only, no numerics.  ``synthetic_z_layout``'s *record* is compared
    as data, not as text, so a reformatting of the literal does not fail while a
    changed key or value does.
    """
    (ref_cfg, ref_ds, ref_gen), *others = ALL_SMALL_SUITES
    ref_src = {
        "build_datasets": inspect.getsource(ref_ds.build_datasets),
        "build_drell_yan": inspect.getsource(ref_ds.build_drell_yan),
        "synthetic_z_layout": inspect.getsource(ref_gen.synthetic_z_layout),
    }
    keys = (
        "DY_ORDER", "DY_ALPHA_S", "DY_NF", "DY_ALPHA_EM", "DY_CHANNELS",
        "DY_FIT_NORMALIZATION", "TEST_MODE_SYNTHETIC_Z", "TEST_MODES",
    )
    ref_consts = {k: getattr(ref_cfg, k) for k in keys}
    ref_record = ref_gen.synthetic_z_layout()
    ref_maps = ref_cfg.dy_field_maps("pp")

    assert others, "ALL_SMALL_SUITES must keep both entries -- see its comment"
    for cfg_, ds_, gen_ in others:
        assert inspect.getsource(ds_.build_datasets) == ref_src["build_datasets"]
        assert inspect.getsource(ds_.build_drell_yan) == ref_src["build_drell_yan"]
        assert (
            inspect.getsource(gen_.synthetic_z_layout) == ref_src["synthetic_z_layout"]
        )
        assert {k: getattr(cfg_, k) for k in keys} == ref_consts
        assert gen_.synthetic_z_layout() == ref_record
        assert cfg_.dy_field_maps("pp") == ref_maps
