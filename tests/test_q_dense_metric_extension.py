"""The saved Q=2 dense-NUTS streams extend without a second warmup."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from campaign import run_q_dense_metric_scan as scan
from campaign import report_q_dense_scan as report


def _saved_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "MODES", ("exp",))
    monkeypatch.setattr(scan, "STREAMS", 1)
    monkeypatch.setattr(scan, "DRAW_COUNT", 4)
    monkeypatch.setattr(scan, "WARMUP", 3)
    monkeypatch.setattr(scan, "CHUNK", 2)
    root = tmp_path / "scan"
    cell = scan._cell_dir(root, "JAM", "exp")
    directory = cell / "stream_01"
    directory.mkdir(parents=True)
    metric = {"metric": "dense_inverse_hessian_at_map",
              "target_fingerprint": "fingerprint", "dimension": 2}
    (cell / "dense_metric.json").write_text(json.dumps(metric))
    for index in range(2):
        samples = np.asarray([[2 * index, 2 * index],
                              [2 * index + 1, 2 * index + 1]], dtype=float)
        summary = {"n_samples": 2, "n_warmup": 3 if index == 0 else 0,
                   "step_size": 0.25}
        np.savez_compressed(directory / f"block_{index:04d}.npz",
                            samples=samples, summary=np.asarray(json.dumps(summary)))
    np.savez_compressed(directory / "state.npz", position=np.asarray([3., 3.]),
                        step_size=np.asarray(0.25), blocks_done=np.asarray(2),
                        draws_done=np.asarray(4))
    stream = {
        "truth": "JAM", "q_key": "2", "mode": "exp", "stream": 1,
        "seed": 19, "n_samples": 4, "n_warmup": 3, "chunk": 2,
        "sampler": "q_nuts", "metric": "dense_inverse_hessian_at_map",
        "adapt_mass": False, "max_tree_depth": scan.MAX_TREE_DEPTH,
        "target_accept": 0.9, "target_fingerprint": "fingerprint",
        "status": "complete", "draws_done": 4, "blocks_done": 2,
        "step_size": 0.25, "elapsed_s": 10.0,
    }
    (directory / "stream.json").write_text(json.dumps(stream))
    manifest = {
        "q_key": "2", "truths": ["JAM"], "modes": ["exp"],
        "streams_per_cell": 1, "warmup_per_stream": 3, "chunk": 2,
        "retained_per_stream": 4, "target_accept": 0.9,
        "metric": "dense inverse Hessian at MAP; no mass adaptation",
        "jobs": [{"truth": "JAM", "mode": "exp", "stream": 1,
                  "seed": 19, "n_samples": 4, "status": "complete"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, directory


def test_extension_dry_run_validates_without_writing(tmp_path, monkeypatch):
    root, _directory = _saved_scan(tmp_path, monkeypatch)
    before = {path.relative_to(root): path.read_bytes()
              for path in root.rglob("*") if path.is_file()}
    assert scan.extend(root, draws=6, dry_run=True) == 0
    after = {path.relative_to(root): path.read_bytes()
             for path in root.rglob("*") if path.is_file()}
    assert after == before


def test_extension_appends_from_last_position_with_no_new_warmup(
    tmp_path, monkeypatch,
):
    root, directory = _saved_scan(tmp_path, monkeypatch)
    original_blocks = [(directory / f"block_{index:04d}.npz").read_bytes()
                       for index in range(2)]
    monkeypatch.setattr(scan, "_build_cell_target", lambda *args, **kwargs: (
        None, SimpleNamespace(dimension=2), None, "fingerprint",
        np.zeros(2), None, {},
    ))
    monkeypatch.setattr(scan, "_load_metric", lambda *args, **kwargs: (
        np.zeros(2), np.eye(2), {"metric": "dense_inverse_hessian_at_map"},
    ))
    import pixel.infer

    calls = []

    def fake_run_q_mcmc(_model, **kwargs):
        calls.append(kwargs)
        samples = np.asarray([[4., 4.], [5., 5.]])
        chain = SimpleNamespace(
            tree_depth=np.ones(2), n_leapfrog=np.ones(2),
            accept_prob=np.full(2, 0.9), energy_error=np.zeros(2),
            divergent=np.zeros(2, dtype=bool),
            reached_max_depth=np.zeros(2, dtype=bool),
        )
        return SimpleNamespace(
            samples=samples, theta_samples=np.zeros((2, 0)),
            action_values=np.zeros(2), chain=chain,
            summary=lambda: {"n_samples": 2, "n_warmup": 0,
                             "step_size": 0.25, "mean_accept_prob": 0.9,
                             "n_divergent": 0},
        )

    monkeypatch.setattr(pixel.infer, "run_q_mcmc", fake_run_q_mcmc)
    result = scan.run_stream(root, "JAM", "exp", 1, draws=6, warmup=0,
                             chunk=2, seed=19, target_accept=0.9,
                             extend_existing=True)
    assert len(calls) == 1
    call = calls[0]
    assert np.array_equal(np.asarray(call["initial"]), [3., 3.])
    assert call["n_warmup"] == 0
    assert call["adapt_step_size"] is False
    assert call["adapt_mass"] is False
    assert call["step_size"] == 0.25
    assert call["seed"] == 19 + 2 * 1_000_003
    assert result["draws_done"] == 6
    assert result["n_warmup"] == 3
    assert result["warmup_this_invocation"] == 0
    assert result["adapt_step_size_this_invocation"] is False
    assert result["elapsed_s"] >= 10.0
    assert [(directory / f"block_{index:04d}.npz").read_bytes()
            for index in range(2)] == original_blocks
    assert (directory / "block_0002.npz").is_file()
    with np.load(directory / "state.npz") as state:
        assert int(state["draws_done"]) == 6
        assert int(state["blocks_done"]) == 3
        assert np.array_equal(state["position"], [5., 5.])


def test_extension_rejects_checkpoint_not_at_last_draw(tmp_path, monkeypatch):
    root, directory = _saved_scan(tmp_path, monkeypatch)
    np.savez_compressed(directory / "state.npz", position=np.asarray([99., 99.]),
                        step_size=np.asarray(0.25), blocks_done=np.asarray(2),
                        draws_done=np.asarray(4))
    with pytest.raises(ValueError, match="not the last draw"):
        scan.extend(root, draws=6, dry_run=True)
    assert not (root / "extension.lock").exists()


def test_original_scan_cannot_reset_an_extended_chain(tmp_path, monkeypatch):
    root, directory = _saved_scan(tmp_path, monkeypatch)
    np.savez_compressed(directory / "state.npz", position=np.asarray([5., 5.]),
                        step_size=np.asarray(0.25), blocks_done=np.asarray(3),
                        draws_done=np.asarray(6))
    with pytest.raises(ValueError, match="longer than this run target"):
        scan.run_stream(root, "JAM", "exp", 1, draws=4, warmup=3,
                        chunk=2, seed=19, target_accept=0.9)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["retained_per_stream"] = 6
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="extended chains"):
        scan.scan(root, dry_run=True, target_accept=0.9, truths=("JAM",))


def test_interrupted_extension_finalizes_without_extra_trajectories(
    tmp_path, monkeypatch,
):
    root, directory = _saved_scan(tmp_path, monkeypatch)
    np.savez_compressed(
        directory / "block_0002.npz",
        samples=np.asarray([[4., 4.], [5., 5.]]),
        summary=np.asarray(json.dumps({"n_samples": 2, "n_warmup": 0,
                                       "step_size": 0.25})),
    )
    np.savez_compressed(directory / "state.npz", position=np.asarray([5., 5.]),
                        step_size=np.asarray(0.25), blocks_done=np.asarray(3),
                        draws_done=np.asarray(6), elapsed_s=np.asarray(12.0))
    monkeypatch.setattr(scan, "_build_cell_target", lambda *args, **kwargs: (
        None, SimpleNamespace(dimension=2), None, "fingerprint",
        np.zeros(2), None, {},
    ))
    monkeypatch.setattr(scan, "_load_metric", lambda *args, **kwargs: (
        np.zeros(2), np.eye(2), {"metric": "dense_inverse_hessian_at_map"},
    ))
    import pixel.infer
    monkeypatch.setattr(pixel.infer, "run_q_mcmc", lambda *args, **kwargs:
                        pytest.fail("an already complete checkpoint must not sample"))
    result = scan.run_stream(root, "JAM", "exp", 1, draws=6, warmup=0,
                             chunk=2, seed=19, target_accept=0.9,
                             extend_existing=True)
    assert result["draws_done"] == 6
    assert result["elapsed_s"] >= 12.0
    assert result["blocks_this_invocation"] == []


def test_campaign_report_uses_extended_retained_count(tmp_path, monkeypatch):
    root, _directory = _saved_scan(tmp_path, monkeypatch)
    monkeypatch.setattr(report, "MODES", ("exp",))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["retained_per_stream"] = 6
    manifest_path.write_text(json.dumps(manifest))
    cell = scan._cell_dir(root, "JAM", "exp")
    row = {
        "tau_int_max": 1.0, "ess_min": 3.0, "ess_fraction_min": 0.5,
        "acceptance_rate": 0.9, "max_abs_delta_h": 1.0, "step_size": 0.25,
        "average_depth": 2.0, "wall_time_s": 20.0,
        "steps_per_trajectory": 3.0, "depth_source": "trajectory",
    }
    result = {
        "sampler": {"total_draws": 6, "total_divergences": 0,
                    "split_rhat_max": 1.0, "streams": [row]},
        "physics": {"chi2": 1.0, "chi2_per_npt": 1.0, "npt": 1,
                    "mean_abs_pull": 0.0, "max_abs_pull": 0.0,
                    "frac_pull_lt_1": 1.0, "frac_pull_lt_2": 1.0,
                    "worst_dataset_chi2": None,
                    "worst_true_pdf_pull": {"pdf": "t3", "pull": 0.0, "x": 0.1}},
    }
    (cell / "REPORT.json").write_text(json.dumps(result))
    rendered = report.render(root)
    assert "6 retained trajectories per stream" in rendered
    assert "`6/(2 tau_int)`" in rendered
