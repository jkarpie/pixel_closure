"""The production shell launcher must not conflate stream and worker controls."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROOT / "run_closure.sh"), *arguments],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


def test_q_cli_routes_isolated_streams_and_warmup_to_sweep(tmp_path):
    output = tmp_path / "new-q-run"
    result = _run(
        "--truth", "JAM", "--data", "union", "--Q", "2", "--modes", "exp",
        "--parameterization", "q", "--warmup", "500", "--samples", "2500",
        "--streams", "4", "--seed-base", "123", "--out", str(output),
        "--dry-run", "--no-plots",
    )
    assert result.returncode == 0, result.stderr
    assert "campaign.run_full_scale_stream_sweep sweep" in result.stdout
    assert "closure_JAM_truth.run_closure --Q 2 --modes exp --kernels-only" in result.stdout
    for option in (
        "--parameterizations q", "--truths JAM", "--data union",
        "--scales 2", "--modes exp", "--warmup 500", "--samples 2500",
        "--stream-counts 4", "--seed-base 123", "--out",
    ):
        assert option in result.stdout
    assert "rm -rf" not in result.stdout
    assert not output.exists()


def test_q_cli_requires_private_output_and_rejects_mock(tmp_path):
    missing = _run("--parameterization", "q", "--Q", "2", "--dry-run")
    assert missing.returncode == 2
    assert "requires --out" in missing.stderr

    mock = _run("--parameterization", "q", "--data", "mock",
                "--out", str(tmp_path / "mock"), "--dry-run")
    assert mock.returncode == 2
    assert "full-scale data" in mock.stderr

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("must remain")
    blocked = _run("--parameterization", "q", "--Q", "2",
                   "--out", str(occupied))
    assert blocked.returncode == 2
    assert "not empty" in blocked.stderr
    assert (occupied / "keep.txt").read_text() == "must remain"


def test_old_hs_cli_rejects_q_sampler_controls(tmp_path):
    result = _run("--samples", "500", "--out", str(tmp_path / "unused"),
                  "--dry-run")
    assert result.returncode == 2
    assert "require --parameterization q" in result.stderr


def test_old_hs_cli_still_uses_source_runner():
    result = _run("--truth", "JAM", "--data", "union", "--Q", "2",
                  "--modes", "exp", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "closure_JAM_truth.run_closure --Q 2 --modes exp" in result.stdout
    assert "campaign.run_full_scale_stream_sweep" not in result.stdout
