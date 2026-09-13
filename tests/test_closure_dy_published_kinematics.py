"""The Drell-Yan rows are the published measurement: E866's mass, and each Z set's window.

Three owner decisions of 2026-09-11 (PIXEL's ``plans/drell_yan_published_units.md``, W7.7),
each of which changed what the closure folds:

* **D6** -- the fitpack workbooks' ``Rtau`` column is ``sqrt(tau) = M/sqrt(s)``, not ``tau``.
  Read as ``tau``, row 0 of table 10001 sat at ``M = 13.44 GeV`` instead of the published
  ``<M> = 4.64 GeV``, and the resulting unphysical ``x1``/``x2`` cut real rows away.
* **D7** -- every selectable collider Z record integrates its published mass window
  (LHCb ``60-120``, CDF and ATLAS ``66-116 GeV``) instead of being evaluated at ``m_Z^2``
  with zero ``Q^2`` width, which is a different observable in a different unit.
* **D8** -- collider Drell-Yan is ``gamma*/Z``; fixed-target E866 and E605 stay photon-only.

Exercises: every suite's ``dy_audit`` (``tau_from_rtau``, ``audit_table``,
    ``build_dy_manifest``) and ``generate.dy_layout``; the six full suites'
    ``config.COLLIDER_DY_SPECS``, ``generate._mass_edges`` / ``_collider_dy_record`` and
    ``datasets.collider_exchange`` with the three collider builders; and, through them,
    ``pixel.data.DrellYan``, ``DYBinIntegratedKernel`` and
    ``pixel.kernels.drell_yan.electroweak``/``couplings``.
Claim: each row carries the kinematics, the mass window and the exchange its publication
    states, on the generation side and on the fit side alike.
Oracle: ``D2`` for the kinematics and the windows -- E866's own published table (the LANL
    ``pp`` release NNPDF ships as rawdata, which carries ``Min``/``Max``/``<M>`` per row)
    and each set's HEPData mass qualifier, both read here rather than restated; ``F1`` for
    the wiring assertions.
Cost: reads two fitpack workbooks per suite and the NNPDF commondata of every selected
    record once (module-scoped ``layout`` fixture). No kernel tensor is assembled.
Not covered: the *value* of a windowed prediction. Nothing here folds a truth or asserts a
    cross section -- the window changes the number by roughly the width of the window, and
    no independent prediction of it exists in this repository. What is pinned is that the
    operator integrates the published interval and uses the published exchange.
    The ``pd`` table (10002) has no published-table copy in the NNPDF commondata, so its
    rows are checked against E866's overall published mass range rather than per-bin.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from pixel.data.contracts import contract_index, load_observable_contracts

#: Every suite carries its own copy of ``dy_audit``/``generate``; the E866 reading has to
#: be right in all eight, which is what D6 says ("in every closure package").
SUITES = (
    "closure_JAM_truth", "closure_JAM_truth_JAM_data", "closure_JAM_truth_NNPDF_data",
    "closure_JAM_truth_small", "closure_NNPDF_truth", "closure_NNPDF_truth_JAM_data",
    "closure_NNPDF_truth_NNPDF_data", "closure_NNPDF_truth_small",
)

#: The suite the collider tests drive, as in ``test_closure_nnpdf_datasets.py``: its
#: ``CONTRACT_MODE`` selects every collider Z record the closure can build.
COLLIDER_SUITE = "closure_NNPDF_truth_NNPDF_data"

#: E866's published ``pp`` table, as NNPDF ships it under the set's ``rawdata/`` -- the
#: LANL release of NuSea:2003qoe (hep-ex/0302019).  Columns, from its own header:
#: ``xF bin, M Min, M Max, <M>, <x_F>, <p_T>, sigma_uncorrected, stat, syst, rad.corr.,
#: sigma_corrected, stat, syst``.  The fitpack workbook carries the *radiatively
#: corrected* column, which is why the identity check below reads column 10.
_E866_TABLE = ("DYE866_Z0_800GEV", "rawdata", "table.csv")
_M_MIN, _M_MAX, _M_MEAN, _XF_MEAN, _SIGMA_CORRECTED = 1, 2, 3, 4, 10

#: Row 0 of table 10001 as E866 published it: the ``4.2-4.7 GeV`` bin at ``<M> = 4.64``.
#: Quoted from that table rather than derived, so this is the paper's number and not the
#: audit's own.  ``Rtau`` is stored to three decimals, so ``Rtau * sqrt(s)`` reproduces
#: ``<M>`` only to about ``0.0005 * 38.8 = 0.02 GeV``.
_ROW0_BIN_GEV = (4.2, 4.7)
_ROW0_MEAN_M_GEV = 4.64
_ROW0_MEAN_TOLERANCE_GEV = 0.05

#: Contract id -> (commondata directory, rawdata table file, the HEPData qualifier that
#: states the dilepton mass window).  The *values* are read from HEPData in the test; only
#: the table and the qualifier's name are named here, because a qualifier cannot be found
#: without them.  Each is the table the set's own record reads its rows from.
_PUBLISHED_MASS_QUALIFIER = {
    "z.lhcb7_dielectron": ("LHCB_Z0_7TEV_DIELECTRON", "Table2.yaml", "LEPTONS MASS"),
    "z.lhcb8_dielectron": ("LHCB_Z0_8TEV_DIELECTRON", "Table2.yaml", "M(E+ E-)"),
    "z.lhcb13_dimuon": ("LHCB_Z0_13TEV", "Table_5a.yaml", "M(LEPTON+ LEPTON-)"),
    "z.lhcb13_dielectron": ("LHCB_Z0_13TEV", "Table_5b.yaml", "M(LEPTON+ LEPTON-)"),
    "z.lhcb7_muon": (
        "LHCB_DY_7TEV_MUON", "HEPData-ins1373300-v1-Table_1.yaml", "M(MU+ MU-)",
    ),
    "z.lhcb8_muon": (
        "LHCB_DY_8TEV_MUON", "HEPData-ins1406555-v1-Table_2.yaml", "M(MU+ MU-)",
    ),
    "z.cdf_zrap": (
        "CDF_Z0_1P96TEV", "HEPData-ins856131-v1-Table_2.yaml", " M(P=3_4) in GEV",
    ),
    "z.atlas7_46fb_cc_peak": (
        "ATLAS_Z0_7TEV_46FB", "HEPData-ins1502620-v1-Table_12.yaml", "MLL",
    ),
    "z.atlas7_36pb": (
        "ATLAS_Z0_7TEV_36PB", "HEPData-ins928289-v1-Table_1.yaml", "M(C=LEPTON+LEPTON-)",
    ),
}

#: The builder kinds that take a dilepton pair, i.e. everything but the gated W arm.
_Z_KINDS = ("drell_yan_binned", "drell_yan_bin_integrated", "drell_yan_mass_spectrum")


def _suite(name):
    return (
        importlib.import_module(f"{name}.config"),
        importlib.import_module(f"{name}.dy_audit"),
        importlib.import_module(f"{name}.generate"),
    )


def _workbook(cfg, dya, idx):
    """``(RS, Rtau, xF, value)`` columns of one fitpack Drell-Yan workbook."""
    import pandas as pd

    df = pd.read_excel(cfg.FITPACK_DY / f"{idx}.xlsx")
    return (dya._col(df, "RS"), dya._col(df, "Rtau"), dya._col(df, "xF"),
            dya._col(df, "value"))


def _published_table(cfg):
    return np.loadtxt(cfg.NNPDF_COMMONDATA.joinpath(*_E866_TABLE), delimiter=",")


def _published_window_gev(cfg, contract_id):
    """The dilepton mass window one HEPData table states, as ``(low, high)`` in GeV."""
    import yaml

    directory, filename, qualifier = _PUBLISHED_MASS_QUALIFIER[contract_id]
    path = cfg.NNPDF_COMMONDATA / directory / "rawdata" / filename
    document = yaml.safe_load(path.read_text())
    for variable in document["dependent_variables"]:
        for entry in variable.get("qualifiers", []):
            if str(entry["name"]) == qualifier:
                low, high = str(entry["value"]).split("-")
                return float(low), float(high)
    raise AssertionError(f"{contract_id}: no {qualifier!r} qualifier in {path}")


@pytest.fixture(scope="module")
def collider():
    cfg, _, gen = _suite(COLLIDER_SUITE)
    datasets = importlib.import_module(f"{COLLIDER_SUITE}.datasets")
    return cfg, gen, datasets


@pytest.fixture(scope="module")
def layout(collider):
    _, gen, _ = collider
    return {rec["contract_id"]: rec for rec in gen.nnpdf_layout()}


@pytest.mark.parametrize("suite", SUITES)
def test_e866_row_zero_sits_in_its_published_mass_bin(suite):
    """``Rtau`` is ``sqrt(tau)``: row 0 of 10001 is E866's ``4.2-4.7 GeV`` bin, not 13.4 GeV.

    The discriminating measurement of D6, on the one row the inventory names. The bin and
    ``<M>`` come from E866's own published table, so the oracle is the experiment's, not
    this audit's own arithmetic.

    The wrong reading is asserted to be wrong as well, which is what makes this test
    unable to pass on the old code: with ``tau = Rtau`` the same row lands at
    ``M = 13.44 GeV``, outside the published bin by a factor of three.
    """
    cfg, dya, _ = _suite(suite)
    rs, rtau, xF, _ = _workbook(cfg, dya, 10001)

    _, q2, _, _, _ = dya.convert_kinematics(rs, dya.tau_from_rtau(rtau), xF)
    mass = float(np.sqrt(q2[0]))
    assert _ROW0_BIN_GEV[0] <= mass <= _ROW0_BIN_GEV[1], mass
    assert abs(mass - _ROW0_MEAN_M_GEV) < _ROW0_MEAN_TOLERANCE_GEV, mass

    # Anti-vacuity, and the old behaviour named: reading the column as tau puts the row
    # outside its own bin, so the assertion above is doing work.
    _, wrong_q2, _, _, _ = dya.convert_kinematics(rs, rtau, xF)
    wrong_mass = float(np.sqrt(wrong_q2[0]))
    assert not _ROW0_BIN_GEV[0] <= wrong_mass <= _ROW0_BIN_GEV[1], wrong_mass


@pytest.mark.parametrize("suite", SUITES)
def test_every_audited_e866_pp_row_lands_in_its_published_mass_bin(suite):
    """All 184 ``pp`` rows, against the published table row by row -- not just row 0.

    The audit's retained-row indices address the workbook, and the workbook is in the
    published table's order; that ordering is *checked* rather than assumed, by requiring
    each row's central value to equal the table's radiatively corrected cross section for
    the same index. Without it a per-row bin comparison could be comparing a row with some
    other row's bin and still pass.

    ``n_retained_precut`` is read rather than ``n_used`` because the ``_small`` suites
    subsample to ``DY_MAX_ROWS`` after the cut; the kinematics being checked are the same.
    """
    cfg, dya, _ = _suite(suite)
    table = _published_table(cfg)
    record = dya.audit_table(10001)
    _, _, _, value = _workbook(cfg, dya, 10001)

    assert record["n_retained_precut"] == table.shape[0], record["n_retained_precut"]
    np.testing.assert_allclose(value, table[:, _SIGMA_CORRECTED], rtol=1e-12, atol=0.0)

    mass = np.sqrt(np.asarray(record["Q2"], dtype=float))
    rows = np.asarray(record["retained_rows"], dtype=int)
    assert np.all(mass >= table[rows, _M_MIN]), mass.min()
    assert np.all(mass <= table[rows, _M_MAX]), mass.max()
    # The bins are 0.5 GeV wide and the column is stored to three decimals, so each row
    # also reproduces its own published <M>.
    assert np.max(np.abs(mass - table[rows, _M_MEAN])) < 0.05


@pytest.mark.parametrize("suite", SUITES)
def test_the_audited_e866_row_counts_are_the_contract_counts(suite):
    """No physical row is cut any more, and the contract records the count it keeps.

    Under the old reading the unphysical ``tau`` pushed ``x1``/``x2`` past 1 on rows that
    are perfectly physical, and the contract's ``selected_rows`` had been written from that
    cut. Recomputed here from the workbook rather than restated, so a cut change shows up
    as a disagreement with the contract instead of silently changing what is fitted.
    """
    cfg, dya, _ = _suite(suite)
    contracts = contract_index(load_observable_contracts())
    ids = {10001: "dy.e866_pp_10001", 10002: "dy.e866_pd_10002"}

    for idx in cfg.DY_ANALYSIS:
        record = dya.audit_table(idx)
        contract = contracts[ids[idx]]
        rs, rtau, xF, value = _workbook(cfg, dya, idx)
        assert record["n_retained_precut"] == int(contract["selected_rows"])
        assert int(contract["source_rows"]) == len(value)
        # Every row of the table survives the closure's DY cuts once tau is right.
        assert record["n_retained_precut"] == len(value)

        # Anti-vacuity: the cut is capable of removing rows -- it removed these.
        _, q2, _, x1, x2 = dya.convert_kinematics(rs, rtau, xF)
        wrong = int(((x1 > 0.0) & (x1 < cfg.DY_X_MAX) & (x2 > 0.0) & (x2 < cfg.DY_X_MAX)
                     & (q2 >= cfg.DY_Q2_MIN)).sum())
        assert wrong < len(value), "the old reading cut nothing; the counts cannot differ"


@pytest.mark.parametrize("suite", SUITES)
def test_a_dy_manifest_audited_under_the_old_reading_is_rebuilt(suite, monkeypatch, tmp_path):
    """A cached manifest cannot carry the old kinematics into a new generation.

    The manifest is the only thing ``dy_layout`` reads, and it is written once and reused:
    the marker is what makes a stale one visible. Without it, everything else in this file
    could pass while generation went on folding ``M = 13.44 GeV`` rows out of
    ``data/dis_manifest.json``.

    Both halves are asserted, because a re-audit that fires unconditionally would make the
    first half pass and cost every generation a full re-audit.
    """
    cfg, dya, gen = _suite(suite)
    stale = {
        "dy": {
            "has_norm_audit": True,
            "max_rows": getattr(cfg, "DY_MAX_ROWS", None),
            "tables": [{
                "idx": 10001, "status": "used", "reaction": "pp", "label": "dy_e866_pp",
                "S": [1505.44], "Q2": [180.66], "Y": [0.066], "rel_stat": [0.7],
            }],
        }
    }
    path = tmp_path / "dis_manifest.json"
    path.write_text(json.dumps(stale))
    monkeypatch.setattr(cfg, "DIS_MANIFEST_PATH", path)

    records = gen.dy_layout()
    # Re-audited: the stale single row is gone and the kinematics are the fixed ones.
    assert float(np.sqrt(records[0]["Q2"][0])) == pytest.approx(4.656, abs=5e-3)
    assert json.loads(path.read_text())["dy"]["tau_convention"] == dya.TAU_CONVENTION

    # And a manifest that already carries the marker is not audited again.
    calls = []
    original_augment = dya.augment_manifest
    monkeypatch.setattr(
        dya, "augment_manifest", lambda: calls.append(1) or original_augment()
    )
    gen.dy_layout()
    assert calls == [], "a current manifest was re-audited"


def test_every_collider_z_record_integrates_its_published_mass_window(collider, layout):
    """Each Z set's ``Q2`` edges are its HEPData mass qualifier, squared.

    The windows are read from HEPData here and written in ``config.COLLIDER_DY_SPECS``
    there, so this is the published interval against the one the operator integrates,
    not the spec compared with itself.

    The second half covers the Z records this table does not name -- the 2D sets, the mass
    spectrum, the CF and the off-peak CC slices -- by requiring every dilepton record in
    the layout to carry a mass window of non-zero width, whatever its source.
    """
    cfg, _, _ = collider
    checked = 0
    for contract_id in _PUBLISHED_MASS_QUALIFIER:
        record = layout[contract_id]
        low, high = _published_window_gev(cfg, contract_id)
        np.testing.assert_allclose(
            np.asarray(record["Q2_low"], dtype=float), low ** 2, rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(record["Q2_high"], dtype=float), high ** 2, rtol=1e-12, atol=0.0
        )
        checked += 1
    assert checked == len(_PUBLISHED_MASS_QUALIFIER)

    dilepton = [rec for rec in layout.values() if rec["kind"] in _Z_KINDS]
    assert len(dilepton) > checked, "no Z record beyond the nine named above"
    for record in dilepton:
        low = np.asarray(record["Q2_low"], dtype=float)
        high = np.asarray(record["Q2_high"], dtype=float)
        assert record["Q2_low"] is not None, record["contract_id"]
        assert np.all(high > low), record["contract_id"]


def test_a_z_record_left_at_the_pole_is_refused(collider, layout):
    """Shown able to fail: the pole route, which is what every Z set used until D7.

    A zero-width ``Q^2`` row is ``d sigma / dQ^2 dY`` -- ``GeV^-4`` -- where the
    measurement is a bin integral in ``GeV^-2``. PIXEL's own unit machinery would refuse
    the barn conversion later; this refuses the record at the point it is built, which is
    where the information about which observable it is still exists.
    """
    cfg, gen, _ = collider
    spec = dict(cfg.COLLIDER_DY_SPECS["z.cdf_zrap"])
    spec.pop("mass_edges")
    entry = {"id": "z.cdf_zrap", "selected_rows": 28, "source_rows": 28}
    common = gen.read_nnpdf_source(cfg.NNPDF_COMMONDATA, spec["commondata_name"])
    with pytest.raises(RuntimeError, match="must integrate its published mass window"):
        gen._collider_dy_record(entry, spec, common)

    # The gated W arm keeps the pole: it has no dilepton mass window of its own, and
    # refusing it would stop the wiring that is kept built for its promotion.
    w_spec = dict(cfg.COLLIDER_DY_SPECS["w.lhcb7_wplus"])
    assert "mass_edges" not in w_spec
    w_common = gen.read_nnpdf_source(cfg.NNPDF_COMMONDATA, w_spec["commondata_name"])
    w_record = gen._collider_dy_record(
        {"id": "w.lhcb7_wplus", "selected_rows": 8, "source_rows": 8}, w_spec, w_common
    )
    assert w_record["Q2_low"] is None


def test_collider_builders_take_gamma_z_and_fixed_target_stays_photon(collider, layout,
                                                                      monkeypatch):
    """D8 at the built operator: the exchange, its couplings, and the resonance it implies.

    Four things are asserted together because each is invisible to the others:

    * the provider is ``gamma_z`` on all three collider arms, and absent (photon) on the
      fixed-target arm;
    * ``alpha_EM`` in the hard prefactor is the provider's own running value -- checked by
      calling ``_row_couplings``, which *raises* when the two disagree, so a fixed
      ``alpha_em`` paired with a ``gamma_z`` table cannot reach a fit;
    * the mass axis picks up the Z pole, which is what makes the window integrable: the
      photon route returns one uniform panel where this returns the Breit-Wigner split;
    * the declared unit is the bin integral's ``GeV^-2`` rather than the pole's ``GeV^-4``.

    ``DY_EVOLVE`` is switched off so no evolution operator is assembled; the exchange and
    the kinematics are independent of it.
    """
    cfg, _, datasets = collider
    monkeypatch.setattr(cfg, "DY_EVOLVE", False)

    exchange = datasets.collider_exchange()
    assert exchange["electroweak"].boson == "gamma_z"
    # One provider, one alpha_EM: the couplings read the electroweak object's own neutral
    # current rather than a second implementation that could drift from it.
    assert exchange["couplings"].eweak is exchange["electroweak"].neutral

    seen = set()
    for contract_id in ("z.cdf_zrap", "z.lhcb7_dielectron", "z.atlas7_himass"):
        record = {k: (list(v)[:1] if isinstance(v, list) else v)
                  for k, v in layout[contract_id].items()}
        dataset = datasets.dy_builder(record["kind"])(
            record, None, None, None, with_systematics=False
        )
        seen.add(record["kind"])
        assert dataset.metadata.meta["units"] == "GeV^-2", contract_id
        kernel = dataset.bilinear_contributions[0].kernel
        base = kernel.base
        assert base.electroweak.boson == "gamma_z", contract_id
        q2 = float(record["Q2"][0])
        _, _, alpha_em = base._row_couplings(q2, q2)
        assert alpha_em == pytest.approx(base.electroweak.alpha_em(q2), rel=1e-13)
        assert kernel._mass_resonance() == (base.electroweak.mz, base.electroweak.gamma_z)
        low, high = float(record["Q2_low"][0]), float(record["Q2_high"][0])
        nodes, _, _ = kernel._nodes(low, high, kernel.mass_nodes, kernel._mass_resonance())
        uniform, _, _ = kernel._nodes(low, high, kernel.mass_nodes, None)
        # The panel is spent where the pole is and nowhere else: a peak window is split
        # around m_Z and gets more nodes than the photon route, while the 116-1500 GeV
        # mass spectrum sits entirely above the pole and correctly keeps uniform nodes.
        # Asserting both directions stops this from reading as "gamma_z always panels".
        contains_pole = low <= base.electroweak.mz ** 2 <= high
        if contains_pole:
            assert nodes.size > uniform.size, contract_id
        else:
            assert nodes.size == uniform.size, contract_id
        assert contains_pole == (contract_id != "z.atlas7_himass"), contract_id
    assert seen == set(_Z_KINDS), seen

    fixed_target = datasets.build_drell_yan(
        {"label": "e866", "reaction": "pp", "Q2": [180.7], "S": [1505.44], "Y": [0.19]},
        None, None, None,
    )
    kernel = fixed_target.bilinear_contributions[0].kernel
    assert kernel.electroweak is None
    assert kernel.alpha_em == cfg.DY_ALPHA_EM
    assert fixed_target.metadata.meta["units"] == "GeV^-4"


def test_the_window_survives_the_manifest_into_the_fit_builder(collider, layout, tmp_path,
                                                               monkeypatch):
    """Generation writes the window; the fit reads it back and builds the same operator.

    The fit never sees the in-memory record -- it reads ``manifest.json`` -- so a window
    that lives only in the layout would leave the two sides integrating different things,
    with nothing to say so. This runs the real round trip: layout record, ``_public``,
    JSON, builder, with ``mean``/``cov`` present as they are on the fit side.
    """
    cfg, gen, datasets = collider
    # The edges are what is under test; no evolution operator is needed to carry them.
    monkeypatch.setattr(cfg, "DY_EVOLVE", False)
    source = layout["z.cdf_zrap"]
    rows = len(source["Q2"])
    record = {k: (list(v)[:2] if isinstance(v, list) and len(v) == rows else v)
              for k, v in gen._public(source).items()}
    record["mean"] = [1.0, 1.1]
    record["cov"] = [[1.0, 0.0], [0.0, 1.0]]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(record))
    restored = json.loads(path.read_text())

    dataset = datasets.dy_builder(restored["kind"])(
        restored, None, None, None, with_systematics=False
    )
    low = dataset.nu[:, 3]
    high = dataset.nu[:, 4]
    window = _published_window_gev(cfg, "z.cdf_zrap")
    np.testing.assert_allclose(low, window[0] ** 2, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(high, window[1] ** 2, rtol=1e-12, atol=0.0)
    assert dataset.metadata.meta["units"] == "GeV^-2"
