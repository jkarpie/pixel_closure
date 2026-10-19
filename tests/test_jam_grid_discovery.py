"""The full-scale JAM suites must use the grid tracked in this checkout."""

import pytest

from closure_JAM_truth import config as union_config
from closure_JAM_truth import pdf_guidance as union_guidance
from closure_JAM_truth_JAM_data import config as jam_config
from closure_JAM_truth_JAM_data import pdf_guidance as jam_guidance
from closure_JAM_truth_NNPDF_data import config as nnpdf_config
from closure_JAM_truth_NNPDF_data import pdf_guidance as nnpdf_guidance


@pytest.mark.parametrize(
    ("config", "guidance"),
    (
        (union_config, union_guidance),
        (jam_config, jam_guidance),
        (nnpdf_config, nnpdf_guidance),
    ),
)
def test_tracked_jam_grid_works_without_installed_set_or_tarball(
    config, guidance, monkeypatch, tmp_path
):
    monkeypatch.setattr(config, "JAM_LHAPDF_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(config, "LHAPDF_DATA_DIR", tmp_path / "install")
    monkeypatch.setattr(config, "JAM_TARBALL", tmp_path / "missing.tar.gz")

    info = config.JAM_REPO_DATA_ROOT / config.JAM_SET / f"{config.JAM_SET}.info"
    assert info.is_file()
    assert config._read_set_info()["NumMembers"] == "792"
    assert guidance.ensure_lhapdf_data() == config.JAM_REPO_DATA_ROOT
