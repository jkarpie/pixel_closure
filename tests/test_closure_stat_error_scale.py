"""Statistical precision couples noise while preserving injected nuisances."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("package", ["closure_NNPDF_truth_small",
                                     "closure_JAM_truth_small"])
def test_scaled_member_preserves_nuisances_and_default_bytes(tmp_path, monkeypatch,
                                                              package):
    import importlib

    generator = importlib.import_module(f"{package}.generate")
    cfg = generator.cfg
    field = cfg.ALL_FIELDS[0]
    fields = {field: SimpleNamespace(nodes=np.array([0.2, 0.8]))}
    replica = np.array([[1., 2.], [2., 3.], [3., 4.]])
    monkeypatch.setattr(generator, "ensemble_curves",
                        lambda *_args: ({name: replica for name in cfg.ALL_FIELDS},
                                        {"mock": True}))
    monkeypatch.setattr(generator.pdf_guidance, "reference_fits",
                        lambda *_args, **_kwargs: {})
    lattice = {"kind": "pseudoitd", "file": "lattice/l.dat",
               "_leading": (np.array([1., 2.]),)}
    exp = {"kind": next(iter(generator.EXP_KINDS)), "file": "exp/e.dat",
           "label": "e", "_leading": (np.array([1., 2.]),),
           "rel_norm": 0.1, "norm_beta": 0.5,
           "_correlated_rel": np.array([[0.02, 0.03]]),
           "correlated_file": "e.npz", "correlated_names": ["source"]}
    monkeypatch.setattr(generator, "lattice_layout", lambda: [lattice])
    monkeypatch.setattr(generator, "exp_layout", lambda **_kwargs: [exp])
    dy = {"file": "dy/d.dat", "reaction": "pp", "rel_norm": 0.1,
          "norm_beta": 0.25, "rel_stat": [0.1, 0.2],
          "Q2": [10., 11.], "S": [100., 100.], "Y": [0., 0.1]}
    z = dict(dy, file="synthetic_z/z.dat")
    monkeypatch.setattr(generator, "dy_layout", lambda: [dy])
    monkeypatch.setattr(generator, "synthetic_z_layout", lambda: [z])
    monkeypatch.setattr(generator.dsets, "build_drell_yan",
                        lambda *_args: object())
    monkeypatch.setattr(generator, "dy_central",
                        lambda *_args: np.array([1., 2.]))
    operators = [[(field, np.eye(2))], [(field, np.eye(2))]]

    def member(label, **kwargs):
        root = tmp_path / label
        monkeypatch.setattr(cfg, "truth_dir", lambda _q: root)
        generator.generate_member("2", fields=fields, operators=operators,
                                  seed=7, **kwargs)
        return root

    default = member("default")
    explicit = member("explicit", stat_error_scale=1.0, stat_noise=None)
    scaled = member("scaled", stat_error_scale=3.0)
    center = member("center", noise=True, stat_noise=False)

    def digest(root, relative):
        return hashlib.sha256((root / relative).read_bytes()).hexdigest()

    for relative in ("lattice/l.dat", "exp/e.dat", "dy/d.dat",
                     "synthetic_z/z.dat", "manifest.json", "truth.json"):
        assert digest(default, relative) == digest(explicit, relative)

    for relative, leading in (("lattice/l.dat", 1), ("exp/e.dat", 1),
                              ("dy/d.dat", 3), ("synthetic_z/z.dat", 3)):
        one = np.loadtxt(default / relative)
        three = np.loadtxt(scaled / relative)
        central = np.loadtxt(center / relative)
        measured = leading
        assert np.allclose(three[:, measured] - central[:, measured],
                           3 * (one[:, measured] - central[:, measured]),
                           rtol=1e-11, atol=1e-11)
        cov_one = one[:, measured + 1:]
        cov_three = three[:, measured + 1:]
        assert np.allclose(cov_three, 9 * cov_one, rtol=1e-11, atol=1e-11)

    with np.load(default / "sys/e.npz") as one, np.load(scaled / "sys/e.npz") as three:
        for key in ("relative", "vectors", "names", "betas"):
            assert np.array_equal(one[key], three[key])
    assert digest(default, "truth.json") == digest(scaled, "truth.json")
    with pytest.raises(ValueError, match="stat_error_scale"):
        member("invalid", stat_error_scale=0)
