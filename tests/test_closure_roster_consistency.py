"""Moved from pixel/tests/test_nnpdf40_roster.py 2026-08-16.

The only test in that file that read the closure packages: it asserts all four
closure configs share the frozen NNPDF4.0 roster.  It is about the closure
suites, not about pixel, so it travels with them -- pixel/tests must import no
closure_* module.
"""

import pytest


def test_all_four_closure_configs_share_the_frozen_roster():
    """The four closure configs' ``nnpdf40_datasets_for_mode`` agree on counts and caps.

    Reaches much further than its explicit assertions below: all four
    ``closure_{JAM,NNPDF}_truth[_small]/config.py`` modules implement
    ``nnpdf40_datasets_for_mode`` as a thin wrapper around
    ``pixel.data.nnpdf40_native.nnpdf40_native_contracts_for_mode`` (confirmed by
    reading all four; none call this module's ``nnpdf40_mode_datasets`` directly). That
    function requires, and raises ``NNPDF40NativeError``/``NNPDF40RosterError`` before
    any assertion here runs if violated: the alias file's identity set equals the
    roster's exactly, in both directions (``nnpdf40_native.py``'s ``set(aliases) !=
    roster_names``); every roster entry maps to exactly one native family; and
    ``_RAW_NDATA``'s key set equals the resolved contract identities exactly, also both
    directions. This is the closest thing in the whole nnpdf40 area to a "does every
    roster entry resolve to code" check spanning all four hand-maintained tables at once
    -- and it is entirely implicit, carried by exception propagation, not by any
    assertion below (oracle F1 for the explicit ``len``/``max_rows`` checks only).

    RESOLVED test_nnpdf40_roster-04: despite the name, this used to compare only
    *counts* -- two configs returning two different, non-overlapping 18-dataset
    subsets would have passed. Now also asserts the actual ``legacy_name`` SETS
    returned by all four configs are identical (for both ``dis`` and ``dy``,
    full scale), sourced independently for each config rather than compared
    only to a shared integer. Currently unexploitable only because all four
    wrappers call the identical underlying function with identical arguments
    (verified by reading all four ``config.py`` files); this pins that as an
    enforced contract rather than an accident of the current implementation.

    Not a duplicate of ``test_nnpdf40_mode_semantics_are_process_complete``: that test
    calls ``nnpdf40_mode_datasets`` (this module) directly, so it cannot see a fault in
    the alias/family/raw-ndata cross-referencing this test's underlying wrapper
    performs. See ``tests/TEST_EXPLANATION.md``.
    """
    from closure_JAM_truth import config as jam_full
    from closure_JAM_truth_small import config as jam_small
    from closure_NNPDF_truth import config as nnpdf_full
    from closure_NNPDF_truth_small import config as nnpdf_small

    def _names(config, mode):
        return {row["legacy_name"] for row in config.nnpdf40_datasets_for_mode(mode)}

    for config in (jam_full, nnpdf_full):
        assert len(config.nnpdf40_datasets_for_mode("dis")) == 18
        assert len(config.nnpdf40_datasets_for_mode("dy")) == 25
        assert all(
            row["max_rows"] is None
            for row in config.nnpdf40_datasets_for_mode("exp")
        )
    for config in (jam_small, nnpdf_small):
        assert len(config.nnpdf40_datasets_for_mode("dis")) == 18
        assert len(config.nnpdf40_datasets_for_mode("dy")) == 25
        assert all(
            row["max_rows"] == 2
            for row in config.nnpdf40_datasets_for_mode("exp")
        )

    # RESOLVED test_nnpdf40_roster-04: the four configs must agree on WHICH
    # datasets, not just how many -- sourced independently per config, not
    # against a shared literal, so two disjoint 18/25-element subsets (which
    # would previously have passed every assertion above) now fail here.
    dis_reference = _names(jam_full, "dis")
    dy_reference = _names(jam_full, "dy")
    for config in (jam_full, jam_small, nnpdf_full, nnpdf_small):
        assert _names(config, "dis") == dis_reference
        assert _names(config, "dy") == dy_reference


