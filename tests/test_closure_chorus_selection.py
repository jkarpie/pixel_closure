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
    selection = {record["id"]: record for record in full_closure_selection("nnpdf_full")}
    assert contract_id in selection, sorted(selection)
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


def test_the_closure_selection_grew_by_exactly_the_two_chorus_records():
    """Pinned so a promotion or a regression is acknowledged rather than absorbed.

    Measured 2026-08-22: 30 records = 13 lattice + 15 DIS + 2 DY. It was 28 before CHORUS.
    """
    selection = full_closure_selection("nnpdf_full")
    by_prefix: dict[str, int] = {}
    for record in selection:
        by_prefix[str(record["id"]).split(".")[0]] = (
            by_prefix.get(str(record["id"]).split(".")[0], 0) + 1
        )
    assert by_prefix == {"lattice": 13, "dis": 15, "dy": 2}, by_prefix
    assert len(selection) == 30


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
