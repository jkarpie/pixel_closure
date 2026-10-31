"""The ``datasets`` and ``dy`` manifest sections, from NNPDF commondata, end to end.

**Since 2026-09-10 the walk feeds two sections.**  Linear DIS (CHORUS, the HERA bottom set)
lands in ``datasets``; every Drell-Yan record -- E605 and the collider W/Z sets -- lands in
``dy``, beside the fitpack E866 tables, with its own fold arm.  The tests at the end of this
file pin that split, the per-boson row slices of the mixed W/Z tables, the edge sources, and
the name-checked operator pairing that replaced a positional slice.

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
import importlib

from collections.abc import Mapping

import numpy as np
import pytest

from pixel.data import families
from pixel.data.contracts import annotate_manifest, closure_extension_roster

from closure_NNPDF_truth_NNPDF_data import config as cfg
from closure_NNPDF_truth_NNPDF_data import datasets as dsets
from closure_NNPDF_truth_NNPDF_data import generate as gen


def test_new_nnpdf_collider_energy_provenance():
    """The 2026 commondata omits per-bin sqrt(s); closure recovers its source value."""
    from pixel.data.closure_audit import read_nnpdf_source

    cases = (
        ("LHCB_Z0_8TEV_DIELECTRON_Y", 8000.0, "hepdata_rawdata_qualifier"),
        ("ATLAS_DY_7TEV_46FB_CC", 7000.0, "legacy_commondata_pin_90d9edc63c8e"),
        ("CMS_Z0_7TEV_DIMUON_2D", 7000.0, "legacy_commondata_pin_90d9edc63c8e"),
    )
    for name, energy, source in cases:
        common = read_nnpdf_source(cfg.NNPDF_COMMONDATA, name)
        assert np.all(np.asarray(common.kinematics["sqrts"]) == energy)
        assert common.metadata["sqrts_source"] == source
    assert "m_Z" in read_nnpdf_source(
        cfg.NNPDF_COMMONDATA, "CMS_Z0_7TEV_DIMUON_2D"
    ).kinematics


def test_collider_energy_disagreement_is_refused():
    from dataclasses import replace

    spec = cfg.COLLIDER_DY_SPECS["z.atlas7_himass"]
    common = gen.read_nnpdf_source(cfg.NNPDF_COMMONDATA, spec["commondata_name"])
    wrong = replace(
        common,
        kinematics={**common.kinematics, "sqrts": np.full(common.n, 8000.0)},
    )
    entry = {
        "id": "z.atlas7_himass", "selected_rows": common.n, "source_rows": common.n,
    }
    with pytest.raises(RuntimeError, match="SQRT\\(S\\) disagrees"):
        gen._collider_dy_record(entry, spec, wrong)


@pytest.mark.parametrize("suite", (
    "closure_JAM_truth", "closure_JAM_truth_JAM_data",
    "closure_JAM_truth_NNPDF_data", "closure_NNPDF_truth",
    "closure_NNPDF_truth_JAM_data", "closure_NNPDF_truth_NNPDF_data",
))
def test_heavy_builder_reads_the_public_manifest_file(suite, tmp_path, monkeypatch):
    """The manifest drops ``_leading``; heavy DIS must read its written .dat file."""
    from pixel.data import HeavyNCSigmaRFONLL
    from pixel.data.experimental._dis_common import read_sf_columns

    path = tmp_path / "heavy.dat"
    np.savetxt(path, [
        [0.01, 100.0, 1.0, 0.2, 0.0],
        [0.02, 200.0, 2.0, 0.0, 0.3],
    ])
    rec = {
        "file": "datasets/heavy.dat", "y": [0.2, 0.3],
        "heavy_flavour": "bottom", "heavy_mass": 4.5, "charge_sq": 1 / 9,
    }
    called = []
    sentinel = object()

    def fake_from_file(cls, source, **kwargs):
        x, q2, mean, cov = read_sf_columns(source)
        np.testing.assert_array_equal(x, [0.01, 0.02])
        np.testing.assert_array_equal(q2, [100.0, 200.0])
        np.testing.assert_array_equal(mean, [1.0, 2.0])
        np.testing.assert_array_equal(cov, [[0.2, 0.0], [0.0, 0.3]])
        assert kwargs["y"] == rec["y"]
        assert kwargs["variant"] == "fonll_c"
        assert kwargs["flavour_mode"] == "vfns"
        assert kwargs["nf"] == 4
        called.append(source)
        return sentinel

    monkeypatch.setattr(HeavyNCSigmaRFONLL, "from_file", classmethod(fake_from_file))
    module = importlib.import_module(f"{suite}.datasets")
    fields = {"sigma": "sigma", "g": "gluon", "t15": "t15"}
    result = module.build_sigma_r_heavy(
        rec, path, fields, None, with_exp_nuisances=False,
    )
    assert result is sentinel
    assert called == [path]


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


def test_annotate_manifest_binds_every_record_in_its_own_section(layout):
    """Every record matches exactly one contract, in the section it is written to.

    ``annotate_manifest`` compares sections before anything else.  Since 2026-09-10 the
    Drell-Yan records are written to ``dy`` and their contracts select there, so a record
    in the wrong section matches zero contracts -- the last assertion shows that failure.
    Each citation is checked against the record's OWN contract, so a record annotated from
    a neighbour's contract fails on the citation as well as on the id.
    """
    from pixel.data.contracts import ContractError, load_observable_contracts

    linear, bilinear = gen._split_sections(layout)
    public = lambda records: [  # noqa: E731
        {k: v for k, v in rec.items() if not k.startswith("_")} for rec in records
    ]
    manifest = {"lattice": [], "exp": [], "datasets": public(linear), "dy": public(bilinear)}
    annotated = annotate_manifest(manifest)
    contracts = {r["id"]: r for r in load_observable_contracts()["datasets"]}
    by_label = {rec["label"]: rec["contract_id"] for rec in layout}
    seen = set()
    for section in ("datasets", "dy"):
        for record in annotated[section]:
            # ``annotate_manifest`` *overwrites* ``contract_id`` from whatever the selector
            # matched, so this is a check on the selector, not a tautology.
            assert record["contract_id"] == by_label[record["label"]]
            assert record["classification"] == "physics"
            assert record["citation_keys"] == list(
                contracts[record["contract_id"]]["citation_keys"]
            )
            assert record["provenance"]["contract_id"] == record["contract_id"]
            seen.add(record["contract_id"])
    assert seen == {rec["contract_id"] for rec in layout}

    misplaced = {"lattice": [], "exp": [], "datasets": [manifest["dy"][0]]}
    with pytest.raises(ContractError, match="matches 0 contracts"):
        annotate_manifest(misplaced)


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
    # Drell-Yan records left this list on 2026-09-10 for their own ``dy`` section, and the
    # pairing is checked by name before the first slice rather than trusted by position.
    assert "records = lattice_layout() + exp_layout() + nnpdf_linear" in member
    assert member.index("_pair_operators(records, operators)") < member.index("lat_ops, exp_ops =")


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


def test_the_operator_pairing_is_checked_by_identity_not_length():
    """The positional split's failure mode is a wrong number, not a crash.

    Until 2026-09-10 the guard compared ``len(operators)`` with ``len(records)``. A swap
    of two equal-length lists passes that, and it is how the 2026-09-09 layout would have
    folded CHORUS through E605's operator: ``EXP_KINDS`` filtered the records while the
    operator list stayed unfiltered.  ``_pair_operators`` checks every pair by name, and is
    shown here able to fail on a swap, a short list and a wrong name.
    """
    import inspect

    records = [
        {"kind": "pseudoitd", "name": "a117m310_itd_t3_real",
         "file": "lattice/a117m310/itd_t3_real.dat"},
        {"kind": "sigma_r", "label": "hera", "file": "exp/hera.dat"},
        {"kind": "sigma_r_cc", "name": "chorus_nu", "file": "datasets/chorus_nu.dat"},
    ]
    good = [("a117m310_itd_t3_real", "L"), ("hera", "H"), ("chorus_nu", "C")]
    assert gen._pair_operators(records, good) == ["L", "H", "C"]
    with pytest.raises(RuntimeError, match="out of order"):
        gen._pair_operators(records, [good[0], good[2], good[1]])
    with pytest.raises(RuntimeError, match="3 manifest records"):
        gen._pair_operators(records, good[:2])
    # A fitpack DIS record carries no ``name``; its builder names the dataset by the file
    # stem, which is accepted, and nothing else is.
    with pytest.raises(RuntimeError, match="out of order"):
        gen._pair_operators(records[1:2], [("not_hera", "H")])

    member = inspect.getsource(gen.generate_member)
    assert member.index("_pair_operators(records, operators)") < member.index(
        "lat_ops, exp_ops ="
    )


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


# ---------------------------------------------------------------------------
# The ``dy`` section, 2026-09-10
# ---------------------------------------------------------------------------


def test_every_drell_yan_record_is_written_to_the_dy_section(layout):
    """Kind decides the section, and the generic section is left linear.

    The generic ``datasets`` section is gated as one unit, so a Drell-Yan record there made
    ``generic_section_family`` raise on every fit.  That was the third of the three
    defects that stopped the 2026-09-09 promotions from generating.
    """
    linear, bilinear = gen._split_sections(layout)
    assert linear and bilinear
    assert {rec["kind"] for rec in bilinear} <= set(dsets.DY_KINDS)
    assert not {rec["kind"] for rec in linear} & set(dsets.DY_KINDS)
    assert all(rec["file"].startswith("dy/") for rec in bilinear)
    assert all(rec["file"].startswith("datasets/") for rec in linear)
    assert {families.family_for_contract_id(r["contract_id"]) for r in linear} == {"dis"}
    assert {families.family_for_contract_id(r["contract_id"]) for r in bilinear} == {"dy"}
    with pytest.raises(RuntimeError, match="unknown manifest section"):
        gen._split_sections([{"contract_id": "z.x", "section": "datasets_typo"}])


def test_a_drell_yan_kind_is_built_only_by_the_dy_table():
    """A DY kind reaching the linear table is refused by name; a blocked one names its gate."""
    from pixel.data.closure_datatypes import DatatypeGateError

    for kind in dsets.DY_KINDS:
        assert callable(dsets.dy_builder(kind))
        with pytest.raises(ValueError, match="'dy' manifest section"):
            dsets._experimental_builder(kind)
    with pytest.raises(DatatypeGateError):
        dsets.dy_builder("drell_yan_total")({}, None, None, None)
    # A registered builder whose datatype the registry blocks does not build: the W arm
    # constructs a Dataset, but assembling one raises because its kernel composes no
    # evolution, so the gate answers first.
    with pytest.raises(DatatypeGateError, match="evolved_tensor"):
        dsets.dy_builder("drell_yan_fiducial_lepton")({}, None, None, None)
    with pytest.raises(ValueError, match="unknown Drell-Yan closure kind"):
        dsets.dy_builder("drell_yan_typo")


def test_the_six_suites_carry_one_spec_table_and_one_layout():
    """One spec block and one layout in every full suite, so the six cannot drift.

    The JAM-truth suites carried a pre-2026-09-09 ``generate_member`` that never
    materialized an NNPDF record, which is exactly the drift this pins against.  The spec
    block is compared as text between its markers and the layout functions as source;
    ``generate_member`` itself is not, because it names its own truth.
    """
    import importlib
    import inspect
    from pathlib import Path

    suites = (
        "closure_NNPDF_truth_NNPDF_data", "closure_NNPDF_truth", "closure_JAM_truth",
        "closure_JAM_truth_NNPDF_data", "closure_NNPDF_truth_JAM_data",
        "closure_JAM_truth_JAM_data",
    )
    root = Path(__file__).resolve().parents[1]
    blocks = set()
    for suite in suites:
        text = (root / suite / "config.py").read_text()
        blocks.add(text[text.index("# >>> canonical NNPDF specs"):
                        text.index("# <<< canonical NNPDF specs")])
    assert len(blocks) == 1, "the canonical spec blocks differ between suites"
    for name in ("nnpdf_layout", "_collider_dy_record", "_fixed_target_dy_record",
                 "fold_dy_record", "_pair_operators", "_split_sections",
                 "_rapidity_edges", "_mass_edges", "_check_centres"):
        sources = {
            inspect.getsource(getattr(importlib.import_module(f"{suite}.generate"), name))
            for suite in suites
        }
        assert len(sources) == 1, f"{name} differs between the six suites"


def test_split_tables_are_partitioned_exactly_and_by_the_filter():
    """A W/Z table split across records covers every row once, as its filter orders them.

    Read from the spec table rather than the layout: the W halves are contracted but not
    selected (the lepton-level kernel composes no evolution), and their wiring -- rows,
    edges, charges -- is kept built so that promoting them is a contract edit.  Each
    record is materialized through ``_collider_dy_record`` with its own synthetic entry.

    The slices come from each set's ``filter.py``, not from metadata labels:
    ``LHCB_DY_8TEV_MUON``'s labels say 16 Z / 9 W+ / 9 W- and are wrong (the filter and the
    kinematics say 18 / 8 / 8).  So each record is checked against the kinematics -- W
    rows sit at ``m_W^2`` and no Z row does -- and the 8 TeV labels are shown to disagree
    while the 7 TeV ones agree.
    """
    from collections import Counter, defaultdict

    import yaml

    tables = defaultdict(list)
    for contract_id, spec in cfg.COLLIDER_DY_SPECS.items():
        if "rows" in spec:
            tables[spec["commondata_name"]].append((contract_id, spec))
    assert set(tables) == {
        "CMS_WPWM_8TEV_MUON_Y", "LHCB_DY_7TEV_MUON_Y", "LHCB_DY_8TEV_MUON_Y",
        "ATLAS_DY_7TEV_46FB_CC", "ATLAS_DY_7TEV_36PB_ETA",
    }
    for name, members in tables.items():
        common = gen.read_nnpdf_source(cfg.NNPDF_COMMONDATA, name)
        covered = sorted(i for _, spec in members for i in range(*spec["rows"]))
        assert covered == list(range(len(common.value))), name
        for contract_id, spec in members:
            n = spec["rows"][1] - spec["rows"][0]
            rec = gen._collider_dy_record(
                {"id": contract_id, "selected_rows": n, "source_rows": n}, spec, common
            )
            is_w = contract_id.startswith("w.")
            assert (rec.get("lepton_charge") in (1, -1)) == is_w, contract_id
            at_w_mass = np.allclose(np.asarray(rec["Q2"]), 80.398 ** 2, rtol=1e-3)
            assert at_w_mass == is_w, contract_id
        charges = sorted(spec["lepton_charge"] for _, spec in members if "lepton_charge" in spec)
        assert charges == [-1, 1], name

    def labels(directory):
        text = (cfg.NNPDF_COMMONDATA / directory / "metadata.yaml").read_text()
        observable = yaml.safe_load(text)["implemented_observables"][0]
        return Counter(observable["plotting"]["extra_labels"]["Boson"])

    assert labels("LHCB_DY_8TEV_MUON") != Counter({"$Z$": 18, "$W^+$": 8, "$W^-$": 8})
    assert labels("LHCB_DY_7TEV_MUON") == Counter({"$Z$": 17, "$W^+$": 8, "$W^-$": 8})


def test_each_atlas_w_record_reads_the_edge_table_of_its_own_charge():
    """The one swap an edge check cannot see: W+ and W- share their eta binning.

    ``_check_centres`` refuses a table whose bins do not bracket the rows, which catches a
    wrong-size or wrong-range table.  A swapped W+/W- pair passes it.  Each HEPData table
    names its own reaction, so this reads that qualifier for every ATLAS W record and
    requires it to match the record's charge.
    """
    import yaml

    checked = 0
    for contract_id, spec in cfg.COLLIDER_DY_SPECS.items():
        how = spec.get("y_edges")
        if not contract_id.startswith("w.") or not isinstance(how, tuple):
            continue
        sign, other = ("W+", "W-") if spec["lepton_charge"] == 1 else ("W-", "W+")
        for directory, number in how[1]:
            path = next((cfg.NNPDF_COMMONDATA / directory / "rawdata").glob(
                f"HEPData-ins*-Table_{number}.yaml"))
            qualifiers = yaml.safe_load(path.read_text())["dependent_variables"][0]["qualifiers"]
            reaction = next(str(q["value"]) for q in qualifiers if q["name"] == "RE")
            assert sign in reaction and other not in reaction, (contract_id, reaction)
            checked += 1
    assert checked == 4


def test_an_edge_source_that_does_not_bracket_its_rows_is_refused():
    """Shown able to fail: a wrong mass window, and a wrong-size rapidity table."""
    common = gen.read_nnpdf_source(cfg.NNPDF_COMMONDATA, "ATLAS_DY_7TEV_46FB_CC")

    spec = dict(cfg.COLLIDER_DY_SPECS["z.atlas7_46fb_cc_low"])
    spec["mass_edges"] = ("windows", ((0, 6, 116.0, 150.0),))  # the HIGH slice's window
    entry = {"id": "z.atlas7_46fb_cc_low", "selected_rows": 6, "source_rows": 6}
    with pytest.raises(RuntimeError, match="centre outside its own bin"):
        gen._collider_dy_record(entry, spec, common)

    spec = dict(cfg.COLLIDER_DY_SPECS["z.atlas7_46fb_cc_peak"])
    spec["y_edges"] = ("hepdata", (("ATLAS_Z0_7TEV_46FB", 11),))  # the LOW slice's 6 bins
    entry = {"id": "z.atlas7_46fb_cc_peak", "selected_rows": 12, "source_rows": 12}
    with pytest.raises(RuntimeError, match="6 rapidity edges for 12 rows"):
        gen._collider_dy_record(entry, spec, common)


def test_the_dy_fold_adds_each_source_once_and_keeps_the_fitpack_stream(monkeypatch):
    """Additive and multiplicative sources enter the covariance once each, at ``|central|``.

    With neither present the draw is ``draw_diag``, exactly as E866 always drew, so the
    fitpack tables' random stream does not move when NNPDF records join the section.
    ``dy_central`` is replaced by a fixed vector: this pins the arithmetic, not a kernel.
    """
    central = np.array([2.0, -3.0, 5.0])
    monkeypatch.setattr(gen, "dy_central", lambda *args: central.copy())
    base = {"reaction": "pp", "rel_stat": [0.1, 0.2, 0.05]}
    additive = np.array([[0.01, 0.0], [0.02, 0.01], [0.0, 0.03]])  # rows x sources
    multiplicative = np.array([[0.05, 0.05, 0.05]])                # sources x rows
    record = dict(base, _additive_rel=additive, _multiplicative_rel=multiplicative)
    _, measured, cov = gen.fold_dy_record(record, None, None, None,
                                          np.random.default_rng(1), noise=False)
    scale = np.abs(central)
    sigma = np.asarray(base["rel_stat"]) * scale + gen.DY_RELATIVE_FLOOR * scale.max()
    a = additive * scale[:, None]
    m = multiplicative.T * scale[:, None]
    np.testing.assert_allclose(cov, np.diag(sigma ** 2) + a @ a.T + m @ m.T,
                               rtol=1e-14, atol=0.0)
    np.testing.assert_array_equal(measured, central)

    plain = dict(base, _additive_rel=np.zeros((3, 0)))
    _, drawn, cov_plain = gen.fold_dy_record(plain, None, None, None,
                                             np.random.default_rng(5))
    np.testing.assert_array_equal(drawn, gen.draw_diag(central, sigma, np.random.default_rng(5)))
    np.testing.assert_allclose(cov_plain, np.diag(sigma ** 2), rtol=1e-15, atol=0.0)

    # The floor is relative, so a prediction in natural units keeps its stated errors.
    # With the DIS arm's absolute 1e-6 this row -- a Z-pole scale -- came out at a
    # relative error of 1e9 (measured 2026-09-10); E866 at about 2.2 against 0.15.
    tiny = central * 1.0e-16
    monkeypatch.setattr(gen, "dy_central", lambda *args: tiny.copy())
    _, _, cov_tiny = gen.fold_dy_record(plain, None, None, None, np.random.default_rng(5),
                                        noise=False)
    relative = np.sqrt(np.diag(cov_tiny)) / np.abs(tiny)
    np.testing.assert_allclose(relative, np.asarray(base["rel_stat"]), rtol=1e-4, atol=0.0)


def test_a_collider_record_builds_and_folds_through_the_real_dy_arm():
    """The generation path runs on a truncated real record, through the real code.

    Layout, builder table and fold are each pinned above in isolation; this runs them in
    sequence on ``z.cdf_zrap`` -- p pbar, full phase space, the arm with no acceptance --
    truncated to one row, and folds the suite's own fixed truth through it.  The assembly
    is row-wise, so one row exercises the same evolution, kernel and fold as twenty-eight.
    Measured 2026-09-10 with the batched assembly: 135 s per row at NLO (43 min before).
    """
    record = next(r for r in gen.nnpdf_layout() if r["contract_id"] == "z.cdf_zrap")
    keep, n = 1, record["retained_rows"]
    trimmed = {
        key: (list(value)[:keep] if isinstance(value, list) and len(value) == n else value)
        for key, value in record.items()
    }
    trimmed["_additive_rel"] = np.asarray(record["_additive_rel"])[:keep]
    trimmed["_multiplicative_rel"] = np.asarray(record["_multiplicative_rel"])[:, :keep]

    fields = cfg.make_fields()
    nodes = np.asarray(list(fields.values())[0].nodes).reshape(-1)
    truth, _ = gen.pdf_guidance.nnpdf_truth_curves(nodes, q_key="2",
                                                   member=cfg.FIXED_TRUTH_MEMBER)
    truth = {f: np.asarray(truth[f], dtype=float).reshape(-1) for f in cfg.ALL_FIELDS}
    dataset = dsets.dy_builder(trimmed["kind"])(trimmed, None, fields, None,
                                                 with_systematics=False)
    assert dataset.n_data == keep
    central, measured, cov = gen.fold_dy_record(trimmed, dataset, truth, fields,
                                                np.random.default_rng(3))
    assert np.all(np.isfinite(central)) and np.all(central > 0.0), central
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)
    assert not np.array_equal(measured, central), "noise=True must draw"
