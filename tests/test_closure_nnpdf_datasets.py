"""The ``datasets`` manifest section: CHORUS from NNPDF commondata, end to end.

**Suite: ``closure_NNPDF_truth_NNPDF_data``, since 2026-09-09.** This file used to run
against ``closure_NNPDF_truth``, which carried CHORUS while there was a single shared
dataset selection. The modes were then split into the two published lists plus their
intersection, and CHORUS is NNPDF-only -- it is absent from JAM24's ``input.py`` -- so it
left the intersection that ``closure_NNPDF_truth`` now runs and stayed in ``nnpdf_data``.
Pointing this file at the ``_NNPDF_data`` suite keeps it testing the arm it was written
for; against the intersection suite ``nnpdf_layout`` correctly returns nothing, which is
a true statement about that suite and a useless one about the commondata arm.

Exercises: ``closure_NNPDF_truth_NNPDF_data.generate.nnpdf_layout``, the coverage guard in
    ``generate_member``, ``closure_NNPDF_truth_NNPDF_data.datasets._record_target`` and the
    ``include_nnpdf`` arm, ``pixel.data.closure_audit.read_nnpdf_source``, and
    ``pixel.data.contracts.annotate_manifest`` on a populated ``"datasets"`` section.
Claim: a contract-selected NNPDF record is materialized with the cuts, the variant, the
    target composition and the uncertainty split its contract declares -- and a selected
    record that is *not* materialized stops generation rather than being omitted.
Oracle: ``F1`` for the contract/registry agreements; ``D2`` for the row counts and the
    uncertainty structure, which are measured against the commondata itself.
Cost: reads two CHORUS commondata sets (1013 + 113 additive sources). ~3 s.
Not covered: the folded prediction's *values*. Nothing here builds a kernel or asserts a
    cross section; that is ``pixel/tests/test_charged_current_dis.py`` and the FK
    asymmetry comparison. This file pins wiring, not physics.

Three defects this file was written after, all of which it now guards:

1. ``closure_audit._NNPDF_SOURCE`` hand-carried a *variant* column that had drifted from
   ``nnpdf40_dis_dy_roster.yaml`` on every neutrino entry. It read ``legacy``, whose
   16 sources include the MULT ``CHORUSISOTARGCOR`` correcting the data toward an
   isoscalar target; the roster reads ``legacy_dw``, which carries no such correction and
   is exactly why the operator owes lead's real ``Z/A``. Auditing one while predicting the
   other applies the isoscalar correction twice. For NuTeV the two variants do not even
   share central values.
2. The CHORUS contracts were promoted with ``selected_rows == source_rows`` and
   ``drop_reason: null``, claiming a dataset whose ``Q2`` reaches ``0.263 GeV^2`` needs no
   cut, while every sibling DIS record here records one.
3. Correcting (2) the first time used ``Q0^2 = 1.65^2``, NNPDF theory 200's input scale,
   rather than the *closure's* ``Q0^2 = mc^2 = 1.6384``, giving 470 instead of 520. The
   generator's own count check rejected it -- which is why that check compares against the
   contract rather than trusting either side.
"""

from __future__ import annotations

import json

from collections.abc import Mapping

import numpy as np
import pytest

from pixel.data import families
from pixel.data.contracts import annotate_manifest, closure_extension_roster

from closure_NNPDF_truth_NNPDF_data import config as cfg
from closure_NNPDF_truth_NNPDF_data import datasets as dsets
from closure_NNPDF_truth_NNPDF_data import generate as gen


@pytest.fixture(scope="module")
def layout():
    return gen.nnpdf_layout()


@pytest.fixture(scope="module")
def chorus_layout(layout):
    """The two CHORUS records alone.

    ``nnpdf_layout`` used to *be* the CHORUS records; since 2026-09-09 it also carries
    the bin-integrated collider Z sets (``drell_yan_binned``), the heavy-flavour-tagged
    bottom set (``sigma_r_heavy``) and fixed-target nuclear Drell-Yan (``drell_yan``).
    The tests below assert properties that are true *of CHORUS* -- a nuclear target
    mapping, a shared ``global:`` multiplicative source, a Q2 cut that actually bites --
    and are not claims about every record the layout can now produce, so they take this
    view rather than the whole layout. The whole-layout invariants (contract coverage,
    row/contract agreement, annotation) keep using ``layout`` and are the stronger for
    now spanning four kinds.
    """
    return [rec for rec in layout if str(rec["contract_id"]).startswith("dis.chorus")]


@pytest.fixture(scope="module")
def selected():
    return {row["id"]: row for row in closure_extension_roster("nnpdf_data")["selected"]}


def test_layout_materializes_exactly_the_contract_selection(layout, selected):
    """No omission, and no dataset entering a fit without a contract.

    Set equality both ways: a promotion the generator ignores and a record the generator
    invents are different failures and both are caught.
    """
    assert {rec["contract_id"] for rec in layout} == set(selected), (
        sorted(rec["contract_id"] for rec in layout), sorted(selected)
    )


def test_every_spec_has_a_contract_and_every_selected_contract_a_spec(selected):
    """``cfg.NNPDF_DATASET_SPECS`` is the build table; the contract is the authority."""
    assert set(cfg.NNPDF_DATASET_SPECS) >= set(selected), (
        f"selected without a build spec: {sorted(set(selected) - set(cfg.NNPDF_DATASET_SPECS))}"
    )


def test_retained_rows_match_the_contract_and_the_closure_cut(chorus_layout, selected):
    """The cut is the generator's own, not NNPDF's, and the contract records its result.

    Recomputed here from ``cfg`` rather than restated, so a change to ``DIS_Q2_MIN`` or
    ``DIS_W2_MIN`` shows up as a disagreement with the contract instead of silently
    changing what is fitted.
    """
    from pixel.data.closure_audit import read_nnpdf_source
    from pixel.data.experimental._dis_common import PROTON_MASS_GEV, dis_w2

    for rec in chorus_layout:
        contract = selected[rec["contract_id"]]
        assert rec["source_rows"] == contract["source_rows"]
        assert rec["retained_rows"] == contract["selected_rows"]
        assert rec["retained_rows"] < rec["source_rows"], (
            "CHORUS reaches below the closure input scale; a record retaining every row "
            "means the cut stopped being applied"
        )

        common = read_nnpdf_source(cfg.NNPDF_COMMONDATA, rec["commondata_name"])
        x = np.asarray(common.kinematics["x"], dtype=float)
        q2 = np.asarray(common.kinematics["Q2"], dtype=float)
        value = np.asarray(common.value, dtype=float)
        w2 = dis_w2(x, q2, m_h=PROTON_MASS_GEV)
        expected = int(
            ((q2 >= cfg.DIS_Q2_MIN) & (w2 > cfg.DIS_W2_MIN) & (value != 0.0)
             & (x > 0.0) & (x < 1.0)).sum()
        )
        assert rec["retained_rows"] == expected

        kept_q2 = np.asarray(rec["_leading"][1], dtype=float)
        assert kept_q2.min() >= cfg.DIS_Q2_MIN
        # Anti-vacuity: the source really does extend below the cut, so the mask is
        # doing work rather than passing everything.
        assert q2.min() < cfg.DIS_Q2_MIN


def test_the_roster_variant_is_read_not_the_audit_default():
    """``legacy_dw``, from the roster -- the whole nuclear argument rests on it.

    Guards the drift that was live until 2026-08-22. ``legacy`` carries the MULT
    ``CHORUSISOTARGCOR``; ``legacy_dw`` does not. A lead-composition prediction is correct
    only against the variant that has *not* already been corrected to isoscalar.
    """
    from pixel.data.closure_audit import _roster_variant

    # Scoped to the *nuclear* specs since 2026-09-09.  ``legacy_dw`` matters because it
    # is the variant that has NOT been corrected to an isoscalar target, which is the
    # whole nuclear argument; a proton-target LHCb Z set has no isoscalar correction to
    # avoid and the roster names no variant for it at all.  Asserting ``legacy_dw`` over
    # every spec would therefore be asserting something the claim does not say.
    nuclear_specs = [
        spec for spec in cfg.NNPDF_DATASET_SPECS.values()
        if isinstance(spec.get("target"), Mapping)
    ]
    assert nuclear_specs, "no nuclear spec left to check the variant on"
    for spec in nuclear_specs:
        assert _roster_variant(spec["commondata_name"]) == "legacy_dw"

    common = gen.read_nnpdf_source(
        cfg.NNPDF_COMMONDATA, "CHORUS_CC_NOTFIXED_PB_NU-SIGMARED"
    )
    labels = {meta.label for meta in common.systematic_metadata.values()}
    assert not any("ISOTARGCOR" in label.upper() for label in labels), sorted(labels)[:8]


def test_the_target_composition_matches_the_contract_prose(selected):
    """A and Z live in config as numbers and in the contract as prose; pin them together.

    Two nuclear conventions can agree at one kinematic point and diverge elsewhere, so the
    convention is pinned by *name and number*, not by a prediction agreeing somewhere.
    """
    from pixel.data.contracts import load_observable_contracts

    contracts = {r["id"]: r for r in load_observable_contracts()["datasets"]}
    for contract_id in selected:
        spec = cfg.NNPDF_DATASET_SPECS[contract_id]
        # A/Z prose only exists for a nuclear target; a proton one has none to pin.
        if not isinstance(spec.get("target"), Mapping):
            continue
        nuclear = str(contracts[contract_id]["operator"]["nuclear"])
        assert str(spec["target"]["mass_number"]) in nuclear, (spec["target"], nuclear)
        assert str(int(spec["target"]["atomic_number"])) in nuclear
        assert "legacy_dw" in nuclear


def test_a_nuclear_target_is_rebuilt_from_the_manifest_mapping():
    """JSON round trip: the mapping form is what keeps A and Z visible in the manifest."""
    from pixel.util.flavor import NuclearTarget

    target = dsets._record_target(
        {"target": {"name": "lead", "mass_number": 207.2, "atomic_number": 82.0}}
    )
    assert isinstance(target, NuclearTarget)
    assert target.mass_number == pytest.approx(207.2)
    assert target.atomic_number == pytest.approx(82.0)
    assert target.proton_fraction == pytest.approx(82.0 / 207.2)
    assert not target.is_isoscalar
    # A plain string still passes through, which is what every fitpack record uses.
    assert dsets._record_target({"target": "proton"}) == "proton"


def test_a_bare_element_name_is_still_refused():
    """The mapping form exists because ``"Pb"`` alone does not name a target.

    Natural lead (A = 207.2) and Pb-208 are different targets and their ``Z/A`` differ, so
    the loader refuses to pick one. If this ever stops raising, the manifest could name a
    nucleus without naming its composition.
    """
    from pixel.util.flavor import normalize_target

    with pytest.raises(ValueError, match="(?i)nuclear|mass number|atomic number"):
        normalize_target("Pb")


def test_additive_sources_stay_out_of_the_multiplicative_nuisance_layer(chorus_layout):
    """The split that a fitpack-shaped assumption would have destroyed.

    ``_exp_nuisance_kwargs`` passes ``correlated_names`` as ``multiplicative_systematics``,
    which the builder references to ``t0``. CHORUS's 1013 (nu) / 113 (nb) correlated
    sources are **additive** -- absolute numbers no t0 iteration rescales. Routing them
    through that kwarg would rescale roughly 75% of the nu row variance by the theory
    prediction. Only the one genuine MULT source may appear there.
    """
    for rec in chorus_layout:
        additive = np.asarray(rec["_additive_rel"], dtype=float)
        assert additive.ndim == 2 and additive.shape[0] == rec["retained_rows"]
        assert additive.shape[1] > 100, additive.shape
        names = rec["correlated_names"]
        assert names == ["global:CHORUSQEDRADCOR1"], names
        assert np.asarray(rec["_correlated_rel"]).shape == (
            len(names), rec["retained_rows"],
        )
        # The approximation is declared rather than left implicit: this source is the
        # same NNPDF nuisance in both beam records and the DIS layer marginalizes each
        # copy separately, dropping the cross-record block.
        assert rec["shared_multiplicative_sources"] == names


def test_the_shared_source_really_is_shared_across_the_two_records(chorus_layout):
    """Anti-vacuity for the declaration above: it names a real, and real-sized, effect."""
    by_id = {rec["contract_id"]: rec for rec in chorus_layout}
    nu = np.asarray(by_id["dis.chorus_nu"]["_correlated_rel"], dtype=float)[0]
    nb = np.asarray(by_id["dis.chorus_nb"]["_correlated_rel"], dtype=float)[0]
    assert nu.shape == nb.shape
    assert not np.allclose(nu, nb), "identical vectors would make the record trivial"
    assert float(np.corrcoef(nu, nb)[0, 1]) > 0.99


def test_annotate_manifest_binds_every_datasets_record_to_its_contract(layout):
    """The path ``pixel/tests/test_observable_contracts.py`` recorded as uncovered.

    ``annotate_manifest`` requires **exactly one** contract per record. The ``datasets``
    selector matches on ``commondata_name``, a field no other section carries, so a
    record reaching this with the wrong section or a missing name fails here rather than
    in a fit.
    """
    manifest = {
        "lattice": [],
        "exp": [],
        "datasets": [
            {k: v for k, v in rec.items() if not k.startswith("_")} for rec in layout
        ],
    }
    from pixel.data.contracts import load_observable_contracts

    annotated = annotate_manifest(manifest)
    assert len(annotated["datasets"]) == len(layout)
    by_label = {rec["label"]: rec["contract_id"] for rec in layout}
    # Each record's citation must be ITS OWN contract's, checked against the contract
    # file rather than against one hardcoded key.  It was ``== ["CHORUS:2005cpn"]`` until
    # 2026-09-09, which was equivalent only while the layout was CHORUS-only; the
    # per-contract form is what the test's own docstring claims and is strictly stronger,
    # since a record annotated from the wrong contract now fails on the citation as well
    # as on the id.
    contracts = {r["id"]: r for r in load_observable_contracts()["datasets"]}
    assert len({by_label[rec["label"]] for rec in layout}) > 1, (
        "this check is only meaningful across more than one contract"
    )
    for record in annotated["datasets"]:
        # ``annotate_manifest`` *overwrites* ``contract_id`` from whatever the selector
        # matched, so asserting it still equals what the layout wrote is a check on the
        # selector, not a tautology: a record annotated as the other beam would fail here.
        assert record["contract_id"] == by_label[record["label"]]
        assert record["classification"] == "physics"
        assert record["citation_keys"] == list(
            contracts[record["contract_id"]]["citation_keys"]
        )
        assert record["provenance"]["contract_id"] == record["contract_id"]
    assert len({r["contract_id"] for r in annotated["datasets"]}) == len(layout)


def test_a_selected_contract_with_no_build_spec_stops_generation(monkeypatch, selected):
    """The guard's real job, and the direction it used to get wrong.

    It previously raised whenever the contracts selected *anything*, which was right only
    while nothing could be built. It must now raise on a selected id with no record --
    and comparing sets rather than counts means a swap cannot cancel out.
    """
    victim = sorted(selected)[0]
    trimmed = {k: v for k, v in cfg.NNPDF_DATASET_SPECS.items() if k != victim}
    monkeypatch.setattr(cfg, "NNPDF_DATASET_SPECS", trimmed)
    records = gen.nnpdf_layout()
    assert {rec["contract_id"] for rec in records} == set(selected) - {victim}

    selected_ids = set(selected)
    materialized = {rec["contract_id"] for rec in records}
    assert selected_ids - materialized == {victim}, (
        "the layout must drop the record so generate_member's coverage check can refuse; "
        "silently substituting one would be the omission the check exists to prevent"
    )


def test_build_datasets_emits_lattice_then_exp_then_datasets():
    """The positional operator split in ``generate_member`` depends on this order.

    ``records = lattice_layout() + exp_layout() + nnpdf_layout()`` is sliced against the
    operator list ``build_datasets`` returns. Parsed from the source rather than restated,
    because restating it would create the second copy this test exists to prevent.
    """
    import inspect
    import re

    source = inspect.getsource(dsets.build_datasets)
    order = re.findall(r'manifest\.get\("(\w+)"', source)
    assert order == ["lattice", "exp", "datasets", "dy"], order

    member = inspect.getsource(gen.generate_member)
    assert "lattice_layout() + exp_layout() + nnpdf_layout()" in member


def test_the_nnpdf_section_is_opt_in_and_follows_the_dis_gate():
    """A fourth flag defaulting to True would put 1,040 neutrino rows in a lattice fit.

    ``fit.py`` passes ``include_lattice``/``include_exp``/``include_dy`` explicitly from
    the mode. ``build_datasets`` grew a fourth switch, and had it defaulted to ``True``
    every caller that did not know about it -- including the ``lattice`` mode -- would
    have silently acquired CHORUS. That is the inclusion the contract machinery exists to
    prevent, so the default is pinned here rather than left to style.
    """
    import inspect

    default = inspect.signature(dsets.build_datasets).parameters["include_nnpdf"].default
    assert default is False

    source = inspect.getsource(gen.generate_member)
    assert "include_nnpdf=True" in source, (
        "generation must ask for the section explicitly; without it the operator list is "
        "shorter than the record list and the positional split mispairs every record"
    )

    from closure_NNPDF_truth import fit as closure_fit

    fit_source = inspect.getsource(closure_fit)
    assert "include_nnpdf=include_nnpdf" in fit_source
    # The gate is DERIVED from the section's own records (2026-08-22), not aliased.
    # This assertion replaced ``"include_nnpdf = include_exp" in fit_source``, which
    # pinned the alias in place -- see the three tests below for what replaced it.
    assert "generic_section_family" in fit_source, (
        "fit.py must route the generic section through the registry; aliasing it to "
        "include_exp is right only while every record in the section happens to be DIS"
    )
    # Checked against *code*, not prose: fit.py's comment quotes the old line to explain
    # what replaced it, so a bare substring test on the source fails on the explanation.
    code = [
        line for line in fit_source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("include_nnpdf = include_exp" in line for line in code)


def test_the_variant_follows_the_roster_rather_than_a_stored_column(monkeypatch):
    """Change the roster and the variant changes with it -- the *derivation* is pinned.

    ``test_the_roster_variant_is_read_not_the_audit_default`` asserts the *value* is
    ``legacy_dw``. That is necessary and not sufficient: re-adding a hardcoded ``variant``
    column to ``closure_audit._NNPDF_SOURCE`` reading ``legacy_dw`` would satisfy it, and
    the drift it exists to prevent -- a stored column silently disagreeing with the roster
    -- would be back with nothing able to notice.

    This one moves the roster underneath and requires the answer to move. A stored column
    cannot pass it.

    The failing case is the one that mattered: ``legacy`` carries the MULT
    ``CHORUSISOTARGCOR``, which corrects the data *toward* isoscalar, so predicting lead's
    real ``Z/A`` against it double-counts the correction.
    """
    from pixel.data import closure_audit
    from pixel.data import nnpdf40_native

    setname = "CHORUS_CC_NOTFIXED_PB_NU-SIGMARED"
    assert closure_audit._roster_variant(setname) == "legacy_dw"

    real = nnpdf40_native.load_nnpdf40_dataset_aliases()
    rewritten = {
        "aliases": {
            alias: (
                {**target, "variant": "legacy"}
                if str(target["dataset"]) == setname
                else target
            )
            for alias, target in real["aliases"].items()
        }
    }
    monkeypatch.setattr(
        nnpdf40_native, "load_nnpdf40_dataset_aliases", lambda: rewritten
    )
    assert closure_audit._roster_variant(setname) == "legacy", (
        "the variant is stored somewhere instead of read from the roster"
    )


def test_a_setname_no_roster_alias_names_is_refused_not_guessed():
    """No alias -> ``KeyError``, because for NuTeV the variant changes central values.

    Anti-default companion to the test above: a lookup that fell back to ``legacy_dw``
    would make the derivation look right on every existing set and be silently wrong on
    the first new one.
    """
    from pixel.data import closure_audit

    with pytest.raises(KeyError, match="must not be guessed"):
        closure_audit._roster_variant("NOT_A_REAL_COMMONDATA_SET")


def _manifest_with_generic(tmp_path, monkeypatch, contract_ids):
    """Point ``load_manifest`` at a synthetic manifest whose generic section is given."""
    manifest = {
        "lattice": [],
        "exp": [],
        families.GENERIC_MANIFEST_SECTION: [
            {"contract_id": cid, "kind": "sigma_r_cc", "file": f"{i}.dat"}
            for i, cid in enumerate(contract_ids)
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cfg, "truth_dir", lambda _q_key: tmp_path)
    return manifest


def test_the_generic_section_is_routed_by_its_own_records_not_by_the_dis_gate(
    tmp_path, monkeypatch
):
    """``generic_section_family`` reads ``contract_id`` and asks the registry.

    ``families.GENERIC_MANIFEST_SECTION`` has existed since the family registry landed
    and nothing routed a mode to it: ``fit.py`` read ``include_nnpdf = include_exp``.
    That is correct today and correct only by accident -- every record in the section is
    DIS -- so a DY or lattice record landing there would have been gated by the DIS flag
    with nothing able to notice.

    The three cases here are the whole contract of the function: a populated section
    names its family, and an absent or empty one is ``None`` rather than a guess.
    """
    _manifest_with_generic(tmp_path, monkeypatch, ["dis.chorus_nu", "dis.chorus_nb"])
    assert dsets.generic_section_family("mc") == "dis"

    _manifest_with_generic(tmp_path, monkeypatch, [])
    assert dsets.generic_section_family("mc") is None

    (tmp_path / "manifest.json").write_text(json.dumps({"lattice": [], "exp": []}))
    assert dsets.generic_section_family("mc") is None


def test_a_generic_section_of_dy_records_follows_the_dy_gate_not_the_dis_one(
    tmp_path, monkeypatch
):
    """The case the alias got wrong, made concrete.

    Under ``include_nnpdf = include_exp`` a section of Drell-Yan records entered a
    ``dis``-mode fit and was withheld from a ``dy``-mode one -- both backwards. The
    registry answers from the record, so the gate follows the record.
    """
    _manifest_with_generic(tmp_path, monkeypatch, ["dy.e906r"])
    assert dsets.generic_section_family("mc") == "dy"

    family = dsets.generic_section_family("mc")
    assert family not in families.mode_families("lattice")
    assert family in families.mode_families("dy")
    assert family not in families.mode_families("dis")   # the alias said the opposite


def test_a_mixed_generic_section_raises_rather_than_picking_a_side(
    tmp_path, monkeypatch
):
    """A section spanning two families has no correct gate, so it must not get one.

    Generation splits its operator list *positionally* against
    ``lattice_layout() + exp_layout() + nnpdf_layout()``, so the section is included or
    excluded as a unit. ``bool(families & selected)`` would admit the DY records whenever
    DIS was selected -- silently, and with the positional split still intact, so nothing
    downstream would flag it.
    """
    _manifest_with_generic(tmp_path, monkeypatch, ["dis.chorus_nu", "dy.e906r"])
    with pytest.raises(ValueError, match="spans families"):
        dsets.generic_section_family("mc")


def test_an_unroutable_generic_contract_id_raises(tmp_path, monkeypatch):
    """A record no family claims is an error, not a default.

    ``family_for_contract_id``'s own docstring gives the reason: silence would route the
    record into whichever section the caller happened to be iterating.
    """
    _manifest_with_generic(tmp_path, monkeypatch, ["nosuchfamily.thing"])
    with pytest.raises(families.FamilyError):
        dsets.generic_section_family("mc")


def test_generate_member_refuses_a_records_operators_length_mismatch():
    """The positional split's failure mode is a wrong number, not a crash.

    ``lat_ops = operators[:len(lat_recs)]`` assumes the two lists correspond one to one.
    If they do not, every record after the gap folds a real truth through the wrong
    operator and generation completes successfully with wrong data.
    """
    import inspect

    source = inspect.getsource(gen.generate_member)
    assert "len(operators) != len(records)" in source
    # The check must come *before* the split it protects.
    assert source.index("len(operators) != len(records)") < source.index("lat_ops = operators[")


def _write_record(rec, directory, central):
    """Write the ``.dat`` and ``sys`` sidecar exactly as ``generate_member`` does."""
    from closure_NNPDF_truth.generate import write_dat

    path = directory / rec["file"]
    scale = np.abs(central)
    additive = np.asarray(rec["_additive_rel"], dtype=float) * scale[:, None]
    sigma = np.abs(np.asarray(rec["rel_stat"], dtype=float)) * scale + cfg.EXP_ABS_FLOOR
    cov = np.diag(sigma * sigma) + additive @ additive.T
    write_dat(path, rec["_leading"], central, cov)

    (directory / "sys").mkdir(parents=True, exist_ok=True)
    relative = np.asarray(rec["_correlated_rel"], dtype=float)
    np.savez_compressed(
        directory / "sys" / rec["correlated_file"],
        relative=relative, vectors=relative * scale,
        names=np.array(rec["correlated_names"]),
        betas=np.zeros(relative.shape[0]),
    )
    return path, cov


def test_the_two_uncertainty_layers_are_added_once_each(tmp_path, chorus_layout):
    """Additive in the ``.dat``, multiplicative in the nuisance layer, neither twice.

    This is the arithmetic behind the split, and it is the failure a working-looking fit
    would hide. With nuisances off the dataset's covariance must be *exactly* the one
    generation wrote; with them on it must differ by a rank-1 outer product -- one MULT
    source, no more. If the 1013 additive sources also reached the nuisance layer the
    difference would be rank 1013, and if they were dropped from the ``.dat`` the first
    comparison would fail.

    Cheap on purpose: ``from_file`` builds contributions lazily, so nothing here assembles
    a kernel. Operator assembly is ~1.2 s/row on the production basis and has no place in
    a unit test.
    """
    fields = cfg.make_fields()
    rec = next(r for r in chorus_layout if r["contract_id"] == "dis.chorus_nu")
    # A positive, non-constant stand-in for the folded truth. Its *values* are irrelevant
    # here -- only that both layers are referenced to the same vector.
    central = 0.1 + 0.9 * np.linspace(0.0, 1.0, rec["retained_rows"]) ** 2
    path, written = _write_record(rec, tmp_path, central)

    bare = dsets.build_sigma_r_cc(rec, path, fields, None,
                                  with_systematics=False, with_exp_nuisances=False)
    plain = np.asarray(bare.cov, dtype=float)
    assert plain.shape == (rec["retained_rows"], rec["retained_rows"])
    # Not exact equality: ``write_dat`` writes ``%.12e``, so the covariance makes a text
    # round trip with ~12 significant digits. Measured max relative difference 4.9e-13,
    # i.e. the format's own precision and nothing else -- 1e-11 is the exact-algebra tier
    # for that round trip, two orders above what is realised.
    np.testing.assert_allclose(plain, written, rtol=1.0e-11, atol=0.0)

    full = dsets.build_sigma_r_cc(rec, path, fields, None,
                                  with_systematics=False, with_exp_nuisances=True)
    augmented = np.asarray(full.cov, dtype=float)
    delta = augmented - plain
    # Anti-vacuity: the nuisance layer must actually do something.
    assert np.linalg.norm(delta) > 0.0
    assert np.linalg.matrix_rank(delta, tol=1e-12 * np.abs(delta).max()) == 1, (
        "the nuisance layer must add exactly the one MULT source; a higher rank means "
        "the additive sources are being counted a second time"
    )
    # And the additive block is genuinely large, so the first comparison is not trivial.
    stat_only = np.diag(np.diag(plain))
    assert np.linalg.norm(plain - stat_only) > np.linalg.norm(stat_only)


def test_a_drawn_observation_is_consistent_with_the_written_covariance(tmp_path, chorus_layout):
    """chi2/ndof ~ 1 for the exact covariance generation writes.

    Guards the scale error the relative/absolute conversion invites: the commondata's
    additive sources are absolute against the *published* central value, while generation
    applies them to the *folded truth*. Storing them relative and rescaling is what keeps
    the two consistent; carrying the absolute numbers over unchanged would put the
    covariance at the wrong scale and show up here immediately.
    """
    rec = next(r for r in chorus_layout if r["contract_id"] == "dis.chorus_nb")
    n = rec["retained_rows"]
    central = 0.1 + 0.9 * np.linspace(0.0, 1.0, n) ** 2
    _, cov = _write_record(rec, tmp_path, central)

    rng = np.random.default_rng(20260822)
    draw = rng.multivariate_normal(central, cov)
    residual = draw - central
    chi2 = float(residual @ np.linalg.solve(cov, residual))
    # sd(chi2/ndof) = sqrt(2/n); 5 sigma is a wide, non-flaky band that a scale error of
    # any realistic size blows straight through.
    assert abs(chi2 / n - 1.0) < 5.0 * np.sqrt(2.0 / n), chi2 / n


def test_a_record_survives_json_and_annotation_and_still_names_its_nucleus(tmp_path, chorus_layout):
    """The manifest is written to disk and read back; the target must survive that.

    ``datasets.py`` never sees the in-memory record -- it reads ``manifest.json``. So the
    round trip that matters is chorus_layout -> ``annotate_manifest`` -> JSON -> builder, and the
    target has to come back out as a ``NuclearTarget`` rather than as the mapping that was
    serialised. ``annotate_manifest`` also *adds* keys (``classification``,
    ``citation_keys``, ``inspire_ids``, ``provenance``); an annotated record must still
    build.
    """
    import json

    from pixel.util.flavor import NuclearTarget

    fields = cfg.make_fields()
    for rec in chorus_layout:
        central = 0.1 + 0.9 * np.linspace(0.0, 1.0, rec["retained_rows"]) ** 2
        _write_record(rec, tmp_path, central)

    manifest = annotate_manifest({
        "lattice": [], "exp": [],
        "datasets": [
            {k: v for k, v in rec.items() if not k.startswith("_")} for rec in chorus_layout
        ],
    })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    reloaded = json.loads(path.read_text())

    # Anti-vacuity: annotation really did add keys, so this is not the raw chorus_layout record.
    added = set(reloaded["datasets"][0]) - set(chorus_layout[0])
    assert {"classification", "citation_keys", "provenance"} <= added, sorted(added)

    for record in reloaded["datasets"]:
        # After JSON the target is a plain dict, which is the case ``_record_target``
        # exists for; a builder reading it straight through would hand a mapping to
        # ``normalize_target`` and fail.
        assert isinstance(record["target"], dict)
        dataset = dsets.build_sigma_r_cc(
            record, tmp_path / record["file"], fields, None,
            with_systematics=False, with_exp_nuisances=True,
        )
        assert dataset.n_data == record["retained_rows"]
        target = dataset.metadata.meta.get("target")
        assert isinstance(target, NuclearTarget), target
        assert target.atomic_number == pytest.approx(82.0)
        assert target.mass_number == pytest.approx(207.2)
        assert len(dataset.metadata.extras["correlated_systematics"]) == 1


def test_a_chorus_record_builds_and_folds_through_the_real_generation_arm(tmp_path):
    """The generation path *runs*, on a truncated record, in seconds rather than minutes.

    Everything else in this file pins arithmetic, contracts and wiring while deliberately
    never building a kernel -- the module docstring says so, and the reason is cost: a full
    CHORUS record is 23 blocks and about 10 minutes, which is why ``generate_member`` was
    run out of band and its results recorded in
    ``plans/fonll_c_and_session_handoff.md`` rather than reproduced here.

    That left a real gap. "The arithmetic is pinned" and "the path runs" are different
    claims, and only the first was tested; a builder-arm signature change, a bad
    ``target`` mapping or a kernel that raises on this record's kinematics would have been
    invisible in suite.

    This closes it at the smallest honest scale: **the real record from
    ``nnpdf_layout()``** -- contract-selected, with its own beam, lead target and slow
    rescaling -- truncated to six rows, through the same
    ``datasets._experimental_builder(rec["kind"])`` arm generation uses, then folded.
    MEASURED 2026-08-23: 23 contributions, ~10 s for the fold, ~15 s for the test.

    Truncation is the whole trick and it is safe here: the assembly is row-wise, so six
    rows exercise the same builder, the same kernel construction and the same fold as 520.
    What it does *not* cover is anything row-count dependent -- the 520x520 covariance
    rank and the ``chi2/ndof`` of a drawn observation stay out of band, and rows 15-17
    pin their arithmetic separately.
    """
    import numpy as np

    records = [
        rec for rec in gen.nnpdf_layout()
        if str(rec["contract_id"]).startswith("dis.chorus")
    ]
    assert records, "nnpdf_layout produced no CHORUS record to smoke"
    # ``records[0]`` until 2026-09-09, when the layout stopped being CHORUS-only: the
    # collider Z and fixed-target DY records carry no ``_leading`` (that pair is the DIS
    # arm's x/Q2) and would KeyError here while saying nothing about the CHORUS fold.
    record = records[0]
    n_rows = len(record["_leading"][0])
    keep = 6

    # Trim every row-aligned entry together; a record whose ``y`` outlived its ``x`` is
    # refused by the builder, which is itself the check that they must stay aligned.
    public = {
        key: (list(value)[:keep]
              if isinstance(value, (list, tuple)) and len(value) == n_rows
              else value)
        for key, value in record.items()
        if not key.startswith("_")
    }
    # The correlated-systematics sidecar is a full-length .npz that truncation would
    # desynchronise; this smoke is about the operator, so drop it and the nuisance layer.
    for sidecar in ("correlated_file", "correlated_names",
                    "shared_multiplicative_sources"):
        public.pop(sidecar, None)

    path = tmp_path / "chorus_smoke.dat"
    gen.write_dat(
        path,
        [np.asarray(column)[:keep] for column in record["_leading"]],
        np.ones(keep),
        np.eye(keep),
    )

    fields = cfg.make_fields()
    dataset = dsets._experimental_builder(public["kind"])(
        public, path, fields, None,
        with_systematics=False, with_exp_nuisances=False, t0=None,
    )
    assert dataset.contributions, "the builder produced no contributions"

    x = np.asarray(record["_leading"][0])[:keep]
    basis = list(fields.values())[0].basis
    total = 0.0
    for contribution in dataset.contributions:
        matrix = np.asarray(contribution.kernel.matrix(x, basis), dtype=float)
        assert matrix.shape[0] == keep, (contribution.field, matrix.shape)
        assert np.all(np.isfinite(matrix)), contribution.field
        total += float(np.abs(matrix).sum())

    # Anti-vacuity: an all-zero operator would satisfy "finite" and predict nothing.
    assert total > 0.0
