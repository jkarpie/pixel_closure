"""CHORUS in the closure selection, and the duplicate-contract hazard that hides it.

Exercises: `observable_contracts.yaml` as consumed by `pixel.data.contracts`, and the
    `pixel_closure` copy of the same file.
Claim: the two CHORUS records are selected as `physics`, carry a *named* nuclear convention
    rather than a deferred one, and the two on-disk copies of the contract file agree.
Oracle: `F1`/`A3` — contract and file-identity checks, no physics.
Cost: pure data.
Not covered: whether the closure can *generate* CHORUS. Selection is necessary and not
    sufficient; `generate.py` still has no layout for it and `datasets.py` no builder arm.

The duplicate-file test is not hygiene. `pixel/observable_contracts.yaml` and
`pixel_closure/observable_contracts.yaml` were **byte-identical**, and only the first is read
— `contracts.load_observable_contracts` resolves `repository_root()/observable_contracts.yaml`
against the *pixel* checkout. Editing the closure copy therefore changes nothing, silently.
That is exactly what happened on 2026-08-22: a promotion applied to the closure copy left the
selection at 28 records with no error anywhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pixel.data.contracts import full_closure_selection, load_observable_contracts

CLOSURE_ROOT = Path(__file__).resolve().parents[1]
CLOSURE_CONTRACTS = CLOSURE_ROOT / "observable_contracts.yaml"

CHORUS_IDS = ("dis.chorus_nu", "dis.chorus_nb")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authoritative_path() -> Path:
    """Where ``load_observable_contracts`` actually reads from."""
    from pixel.data.contracts import repository_root

    return repository_root() / "observable_contracts.yaml"


def test_the_two_contract_copies_are_identical():
    """Only one is read; a divergence makes an edit to the other a silent no-op.

    Measured 2026-08-22: they were byte-identical, and a promotion applied to the closure copy
    changed nothing while reporting success. If this ever fails, the question is not "which is
    right" but "which one did the last edit go to".
    """
    authoritative = _authoritative_path()
    assert authoritative.exists(), authoritative
    assert CLOSURE_CONTRACTS.exists(), CLOSURE_CONTRACTS
    if authoritative.resolve() == CLOSURE_CONTRACTS.resolve():
        pytest.skip("the closure copy IS the authoritative one in this layout")
    assert _digest(authoritative) == _digest(CLOSURE_CONTRACTS), (
        f"{authoritative} and {CLOSURE_CONTRACTS} have diverged; only the first is read by "
        "load_observable_contracts, so an edit to the second does nothing"
    )


@pytest.mark.parametrize("contract_id", CHORUS_IDS)
def test_chorus_is_selected_as_physics_with_its_retained_rows(contract_id):
    # ``nnpdf_data``, not ``nnpdf_full``, since 2026-09-09.  CHORUS is absent from JAM24's
    # own dataset list (``fitpack_legacy/examples/input.py`` has no CHORUS entry), so when
    # the modes were split into the two published selections plus their intersection,
    # CHORUS stayed in the NNPDF4.0 selection and left the intersection that the two
    # headline suites run.  It is still selected, still physics, still 520 retained rows.
    selection = {record["id"]: record for record in full_closure_selection("nnpdf_data")}
    assert contract_id in selection, sorted(selection)
    # CHORUS is NNPDF-only, and the two mode families say that differently:
    # it is in ``nnpdf_data`` and (since the modes became the UNION on 2026-09-09) in
    # ``jam_full``/``nnpdf_full`` too, but it must never appear in ``jam_data`` --
    # JAM24's input.py has no CHORUS entry, and that is the fact being pinned.
    assert contract_id not in {
        r["id"] for r in full_closure_selection("jam_data")
    }, "CHORUS is absent from JAM24's dataset list and must not enter jam_data"
    assert contract_id in {
        r["id"] for r in full_closure_selection("nnpdf_full")
    }, "the full modes are the union, so an NNPDF-only record belongs in them"
    record = selection[contract_id]
    assert record["classification"] == "physics"
    assert record["selectable"] is True
    # 607 rows per beam sign in the commondata; 520 retained under the closure's own
    # DIS cuts.  Pinning both, and pinning that they DIFFER, is the point: the record
    # was first promoted with selected == source and drop_reason null, which claimed a
    # dataset reaching Q2 = 0.263 GeV^2 needed no cut while every sibling here records
    # one.  87 rows fail: 67 below Q0^2 = mc^2 = 1.6384, 20 below W2 = 10.
    assert record["source_rows"] == 607
    assert record["selected_rows"] == 520
    assert record["drop_reason"] == "87 rows fail Q2/W2 closure cuts"
    assert record["likelihood_group"] == contract_id


@pytest.mark.parametrize("contract_id", CHORUS_IDS)
def test_the_nuclear_convention_is_named_not_deferred(contract_id):
    """``operator.nuclear`` must state the composition, not promise one later.

    The field previously read "lead nuclear model required", which is a deferral. A record
    that says *required* while being `selectable: true` is the failure this guards: it would
    let a dataset into a fit with the convention still open.
    """
    contracts = load_observable_contracts()
    record = next(r for r in contracts["datasets"] if r["id"] == contract_id)
    nuclear = str(record["operator"]["nuclear"])
    assert "required" not in nuclear.lower(), nuclear
    # The composition itself, so a silent change of isotope convention is visible.
    assert "207.2" in nuclear and "82" in nuclear, nuclear
    assert "legacy_dw" in nuclear, nuclear
    assert "unimplemented" not in str(record["operator"]["kind"]).lower()


def _by_prefix(mode):
    counts: dict[str, int] = {}
    for record in full_closure_selection(mode):
        key = str(record["id"]).split(".")[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_the_three_selections_have_their_measured_compositions():
    """Pinned so a promotion or a regression is acknowledged rather than absorbed.

    Was 30 records = 13 lattice + 15 DIS + 2 DY for a single shared selection (measured
    2026-08-22; 28 before CHORUS).  Split 2026-09-09 into the two published dataset lists
    and their intersection, so there are now three compositions to pin, and the two
    differences between them are the whole point of the split:

    * ``dis.hermes_p_10007`` is JAM-only -- HERMES sigma_r is not on the NNPDF4.0 roster;
    * ``dis.chorus_nu``/``dis.chorus_nb`` are NNPDF-only -- CHORUS is not in JAM's
      ``input.py``;
    * the 2026-09-10 collider Drell-Yan wiring added ten Z records, two of which JAM24
      also fits (so they are in both lists, and in neither's difference); the W records
      it contracted stay reader_only;
    * ``dy.e866_pd_10002`` left *all three*.  Verified from the workbooks: JAM fits
      E866 pp absolute + the pd/2pp ratios, NNPDF4.0 fits ``DYE886P`` (pp absolute) +
      ``DYE886R`` (the ratio).  Neither fits E866 pd absolute, which is why the DY count
      is 1 and not 2.
    """
    # 2026-09-10: the collider Drell-Yan wiring.  jam_data gains its first two Z records
    # (JAM24's zrap/1000 and wzhep ATLAS7C_Z, realized from NNPDF's copies of the same
    # measurements); nnpdf_data gains ten.  No W record is selected anywhere: the
    # lepton-level kernel composes no evolution (closure_datatypes, 2026-09-10).
    assert _by_prefix("jam_data") == {"lattice": 13, "dis": 13, "dy": 1, "z": 2}
    assert _by_prefix("nnpdf_data") == {"lattice": 13, "dis": 15, "dy": 2, "z": 15}
    # **UNION since 2026-09-09**, at the owner's request -- "the larger set which
    # contains everything used in both or either". It was the intersection for one
    # afternoon; both are defensible and the union is what the suites run.
    union = {"lattice": 13, "dis": 16, "dy": 2, "z": 15}
    assert _by_prefix("jam_full") == union
    assert _by_prefix("nnpdf_full") == union

    jam = {r["id"] for r in full_closure_selection("jam_data")}
    nnpdf = {r["id"] for r in full_closure_selection("nnpdf_data")}
    full = {r["id"] for r in full_closure_selection("jam_full")}
    # The headline suites run exactly the UNION -- asserted, not asserted-by-count.
    # Both still run the SAME dataset, so a JAM-vs-NNPDF truth comparison stays
    # unconfounded by any dataset difference; the union just makes it on more data.
    assert full == jam | nnpdf
    assert full != jam & nnpdf, "a union that equals the intersection means the two selections stopped differing"
    assert full == {r["id"] for r in full_closure_selection("nnpdf_full")}
    assert jam - nnpdf == {"dis.hermes_p_10007"}
    # JAM's two Z records are NNPDF records, not twins: a fitpack twin in the union would
    # count the same measurement twice.
    assert {"z.cdf_zrap", "z.atlas7_46fb_cc_peak"} <= jam & nnpdf
    assert not {"z.cdf_1000", "z.atlas7c_peak", "w.atlas7c_wplus", "w.atlas7c_wminus"} & full
    assert not any(i.startswith("w.") for i in full), "a W record cannot be folded yet"
    assert nnpdf - jam == {
        "dis.chorus_nu", "dis.chorus_nb",
        # Promoted 2026-09-09 with their builders; all NNPDF-only.
        "dis.heracomb_bottom", "dy.e605_copper",
        "z.lhcb7_dielectron", "z.lhcb8_dielectron", "z.lhcb13_dimuon",
        "z.lhcb13_dielectron", "z.atlas8_lomass_2d",
        # 2026-09-10, NNPDF-only.
        "z.atlas7_himass", "z.atlas8_himass_2d", "z.atlas7_46fb_cf",
        "z.lhcb7_muon", "z.lhcb8_muon",
        "z.atlas7_46fb_cc_low", "z.atlas7_46fb_cc_high", "z.atlas7_36pb",
    }
    assert not any("e866_pd" in i for i in jam | nnpdf)


def test_nutev_is_not_promoted_alongside_chorus():
    """NuTeV shares CHORUS's target machinery and beam, so it is the likeliest to slip in.

    Its observable is massive charm tagged by a muonic decay, not inclusive CC, and no
    coefficient function for it exists. It must stay `reader_only`.
    """
    contracts = load_observable_contracts()
    for record in contracts["datasets"]:
        if str(record["id"]).startswith("dis.nutev"):
            assert record["classification"] == "reader_only", record["id"]
            assert record["selectable"] is False
            assert record["drop_reason"]
    selected = {record["id"] for record in full_closure_selection("nnpdf_full")}
    assert not {i for i in selected if i.startswith("dis.nutev")}
