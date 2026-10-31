"""NNLO closure defaults reach PIXEL's DIS and coherent DY builders."""

import importlib

import pytest


FULL_SUITES = tuple(
    f"closure_{truth}_truth{selection}"
    for truth in ("JAM", "NNPDF")
    for selection in ("", "_JAM_data", "_NNPDF_data")
)
SUITES = FULL_SUITES + ("closure_JAM_truth_small", "closure_NNPDF_truth_small")


@pytest.mark.parametrize("suite", SUITES)
def test_nnlo_dis_and_dy_configuration(suite, monkeypatch, tmp_path):
    cfg = importlib.import_module(f"{suite}.config")
    datasets = importlib.import_module(f"{suite}.datasets")
    assert cfg.ORDER == cfg.DY_ORDER == "NNLO"
    assert cfg.DY_CHANNELS == ("nnlo_full",)
    if suite in FULL_SUITES:
        assert cfg.DY_EVOLUTION_ORDER == "NNLO"
        assert cfg.PITD_MATCHING_ORDER == "NLO"

    # Exercise real builder argument validation, including PIXEL's special NNLO
    # channel/source contract, without assembling production-sized tensors.
    from pixel.kernels.drell_yan import ReDYNNLOSource

    monkeypatch.setattr(cfg, "DY_EVOLVE", False)
    source = object.__new__(ReDYNNLOSource)
    source.provenance = {"source_kind": "wiring_test"}
    monkeypatch.setattr(cfg, "DY_NNLO_SOURCE", source)
    record = {
        "reaction": "pp", "label": "nnlo_order_check",
        "Q2": [25.0], "S": [1501.0], "Y": [0.3],
    }
    dataset = datasets.build_drell_yan(record, None, {}, None)
    assert dataset.metadata.meta["hard_order"] == "NNLO"
    assert dataset.metadata.meta["hard_channels"] == ("nnlo_full",)
    assert dataset.bilinear_contributions
    assert all(c.kernel.order == "NNLO" for c in dataset.bilinear_contributions)

    captured = {}

    def capture(path, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(datasets.data.F2, "from_file", capture)
    fields = {name: name for name in cfg.ALL_FIELDS}
    datasets.build_f2(
        {"target": "proton", "file": "test.dat"}, tmp_path / "test.dat",
        fields, None, with_exp_nuisances=False,
    )
    assert captured["order"] == "NNLO"
