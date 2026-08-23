"""The closure half of the family-registry contract.

Exercises: the four `closure_*/config.py` ``TEST_MODES`` tuples and ``run_closure.sh``'s
    ``case`` arm against :mod:`pixel.data.families`.
Claim: the process taxonomy is stated once, in the registry. A mode offered by a config but
    rejected by the runner, or a family added to the registry and never surfaced here, is a
    test failure rather than a run that quietly does less.
Oracle: F1 -- internal consistency across three independently maintained lists (a Python
    tuple per suite, a shell ``case`` arm, and the registry). No physics.
Cost: pure data; imports the four configs, reads one shell script.
Not covered: whether a mode, once accepted, actually selects the datasets the registry says
    it should. That is the call-site migration and is not done yet -- ``fit.py`` still
    derives ``include_lattice``/``include_exp``/``include_dy`` from its own literals.

``pixel/tests`` may not import ``closure_*``; this is the mirror of
``pixel/tests/test_family_registry.py``, which pins ``families.MODES`` literally and names
this gap in its own audit entry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pixel.data import families

ROOT = Path(__file__).resolve().parents[1]
RUN_CLOSURE = ROOT / "run_closure.sh"

FULL_SUITES = ("closure_NNPDF_truth", "closure_JAM_truth")
SMALL_SUITES = ("closure_NNPDF_truth_small", "closure_JAM_truth_small")
SUITES = FULL_SUITES + SMALL_SUITES


def _config(suite: str):
    return __import__(f"{suite}.config", fromlist=["config"])


def _runner_modes() -> frozenset[str]:
    """Modes ``run_closure.sh``'s ``case`` arm accepts.

    Parsed rather than restated: restating it here would create a fourth copy of exactly the
    list this file exists to keep down to one.
    """
    text = RUN_CLOSURE.read_text(encoding="utf-8")
    match = re.search(r"^\s*([a-z_|]+)\)\s*modes\+=\(", text, re.MULTILINE)
    assert match, "could not find the mode 'case' arm in run_closure.sh"
    return frozenset(match.group(1).split("|"))


@pytest.mark.parametrize("suite", FULL_SUITES)
def test_full_suite_modes_are_exactly_the_registry_modes(suite):
    """A full suite offers the registry's modes and nothing suite-local."""
    assert tuple(_config(suite).TEST_MODES) == families.MODES


@pytest.mark.parametrize("suite", SMALL_SUITES)
def test_small_suite_modes_are_the_registry_modes_plus_named_synthetic_ones(suite):
    """The smoke suites add synthetic plumbing modes, and only those.

    The registry deliberately does not know about ``synthetic_z_plumbing``: it is a wiring
    probe, not a process family, and giving it a ``FamilySpec`` would put it into the derived
    ``exp`` and ``both`` of every suite.
    """
    cfg = _config(suite)
    modes = tuple(cfg.TEST_MODES)
    assert modes[: len(families.MODES)] == families.MODES
    extra = modes[len(families.MODES):]
    assert extra == (cfg.TEST_MODE_SYNTHETIC_Z,)
    assert set(extra).isdisjoint(families.FAMILIES)


def test_runner_accepts_every_mode_every_suite_offers():
    """`run_closure.sh` rejecting a configured mode is a silent loss of coverage.

    It does not fail the suite -- the runner ``die``s on that argument, so whoever asked for
    the mode simply does not get it. This has drifted before.
    """
    accepted = _runner_modes()
    offered = {m for suite in SUITES for m in _config(suite).TEST_MODES}
    missing = sorted(offered - accepted)
    assert not missing, f"run_closure.sh rejects configured modes: {missing}"


def test_runner_accepts_no_mode_no_suite_offers():
    """The other direction: an arm left behind after a mode was retired."""
    accepted = _runner_modes()
    offered = {m for suite in SUITES for m in _config(suite).TEST_MODES}
    stale = sorted(accepted - offered)
    assert not stale, f"run_closure.sh accepts modes no suite offers: {stale}"


def test_reserved_registry_families_are_not_offered_by_any_suite():
    """A reserved family has no builder, so a suite offering it would build nothing."""
    reserved = set(families.reserved_families())
    assert reserved
    for suite in SUITES:
        assert reserved.isdisjoint(_config(suite).TEST_MODES), suite


def test_all_four_suites_agree_on_the_shared_modes():
    """The four configs are meant to differ only in truth set and smoke scale."""
    shared = [tuple(_config(s).TEST_MODES)[: len(families.MODES)] for s in SUITES]
    assert len(set(shared)) == 1, dict(zip(SUITES, shared))


# -- the call-site migration, 2026-08-22 --------------------------------------
#
# Until this date ``families.py`` was declared and not consumed: every ``fit.py`` still
# derived ``include_lattice``/``include_exp``/``include_dy`` from its own
# ``mode in ("dis", "exp", "both")`` literals, and ``api/manifest.py`` and
# ``data/contracts.py`` each carried the section tuple inline. Four copies of the mode
# taxonomy and two of the section list, all of which the registry already held.
#
# The tests above pin that the *lists agree*. These pin that the call sites **read the
# registry**, which is the only thing that keeps them agreeing without anyone remembering.

ALL_SUITES = ("closure_NNPDF_truth", "closure_JAM_truth",
              "closure_NNPDF_truth_small", "closure_JAM_truth_small")


@pytest.mark.parametrize("suite", ALL_SUITES)
def test_fit_derives_its_mode_gates_from_the_registry(suite):
    """No ``fit.py`` may restate ``mode in (...)`` for a process family.

    The failure a restated chain produces is silent: a family the registry knows about and
    the chain does not is simply never selected, and the fit reports success having fitted
    less data.
    """
    import importlib
    import inspect
    import re

    fit = importlib.import_module(f"{suite}.fit")
    source = inspect.getsource(fit)
    assert "families.mode_families(" in source, (
        f"{suite}/fit.py does not consult pixel.data.families"
    )
    # The literal the migration removed, in either shape it appeared in.
    for literal in ('mode in ("dis", "exp", "both")',
                    'mode in ("dy", "exp", "both")',
                    'mode in ("lattice", "both")'):
        assert literal not in source, (
            f"{suite}/fit.py still restates {literal!r}; that is the copy families.py exists "
            "to remove"
        )
    # Anti-vacuity: something must actually branch on the selection.
    assert re.search(r'"(lattice|dis|dy)" in selected', source), source[:0]


@pytest.mark.parametrize("suite", ALL_SUITES)
def test_the_registry_gates_select_what_the_old_literals_did(suite):
    """Behaviour is unchanged: the derivation reproduces the retired chain exactly.

    A refactor that quietly changed which datasets a mode selects would pass the test above
    and be a much worse bug than the drift it was fixing. This recomputes both sides.
    """
    from pixel.data.families import mode_families

    legacy = {
        "lattice": {"lattice"},
        "dis": {"dis"},
        "dy": {"dy"},
        "exp": {"dis", "dy"},
        "both": {"lattice", "dis", "dy"},
    }
    for mode, expected in legacy.items():
        assert set(mode_families(mode)) == expected, mode


def test_manifest_sections_come_from_the_registry():
    """``annotate_manifest`` and ``api.manifest`` must not carry their own section tuple.

    An unannotated section is silent in the worst way: ``matching_contracts`` is never asked
    about those records, so they enter a fit with no contract behind them -- which is the
    exact gate the contract machinery exists to be.
    """
    import inspect

    from pixel.api import manifest as api_manifest
    from pixel.data import contracts
    from pixel.data.families import manifest_sections

    assert manifest_sections() == ("lattice", "exp", "dy", "datasets")
    assert (
        inspect.signature(contracts.annotate_manifest).parameters["sections"].default
        is None
    )
    for module in (contracts, api_manifest):
        source = inspect.getsource(module)
        assert '("lattice", "exp", "dy", "datasets")' not in source.replace(
            '``("lattice", "exp", "dy", "datasets")``', ""
        ), f"{module.__name__} still restates the section tuple"
    # And the default really is applied, not just accepted.
    annotated = contracts.annotate_manifest(
        {section: [] for section in manifest_sections()}
    )
    assert set(annotated) == set(manifest_sections())
