"""MAP command routing, source-target agreement, and batch failure propagation."""

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from closure_full_q_integration import map_start, prepare_map, variant
from campaign import run_full_scale_stream_sweep as sweep


@pytest.mark.parametrize("check_only", (False, True))
@pytest.mark.parametrize("selection", variant.SOURCE_SUITES)
def test_map_driver_builds_same_final_t0_target(monkeypatch, tmp_path, selection, check_only):
    truth, data = selection
    with variant.activated(truth, data) as (_, cfg, fit, _):
        original, final = object(), object()
        t0 = {"table": [1.2]}
        calls = []

        def build(q, mode, **kwargs):
            calls.append((q, mode, kwargs))
            model = final if "t0" in kwargs else original
            return SimpleNamespace(compile=lambda: model), {}

        def iterate(q, mode, model, **kwargs):
            assert (q, mode, model) == ("2", "exp", original)
            assert kwargs == {"use_kernel_cache": True}
            return t0, 3

        def use_map(model, **kwargs):
            assert model is final
            assert kwargs["prior_rcond"] == 0
            assert kwargs["cache_root"] == tmp_path
            return None, None, {"action": 1, "gradient_norm": 0,
                                "cache_status": "hit", "cache_path": "test.npz"}

        def forbidden(*a, **k):
            pytest.fail("MAP-only driver entered sampling or the wrong MAP path")

        monkeypatch.setattr(fit, "build_analysis", build)
        monkeypatch.setattr(fit, "_iterate_t0", iterate)
        monkeypatch.setattr(fit, "run_fit", forbidden)
        monkeypatch.setattr(map_start, "load_q_map_start", use_map if check_only else forbidden)
        monkeypatch.setattr(map_start, "prepare_q_map_start", forbidden if check_only else use_map)
        prepare_map.prepare_cell(truth, data, "2", "exp", cache_root=tmp_path,
                                 check_only=check_only)
        assert calls == [("2", "exp", {"use_kernel_cache": True}),
                         ("2", "exp", {"use_kernel_cache": True, "t0": t0})]


def test_cli_passes_large_custom_budgets_and_multiple_cells(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(prepare_map, "prepare_cell", lambda *a, **k: calls.append((a, k)))
    assert prepare_map.main([
        "--truth", "NNPDF", "--data", "JAM", "--Q", "2,3", "--modes", "dis,exp",
        "--cache-root", str(tmp_path), "--lbfgs-maxiter", "80000",
        "--newton-maxiter", "8000", "--lbfgs-maxfun", "900000", "--resume",
        "--adam-steps", "100,200", "--adam-learning-rates", "0.02,0.005",
    ]) == 0
    assert [call[0][2:] for call in calls] == [("2", "dis"), ("2", "exp"), ("3", "dis"), ("3", "exp")]
    for _, kwargs in calls:
        assert kwargs["options"].lbfgs_maxiter == 80000
        assert kwargs["options"].newton_maxiter == 8000
        assert kwargs["options"].lbfgs_maxfun == 900000
        assert kwargs["options"].adam_steps == (100, 200)
        assert kwargs["options"].adam_learning_rates == (0.02, 0.005)
        assert kwargs["resume"]
    assert prepare_map.main(["--newton-maxiter", "-1"]) == 2


def test_cli_can_disable_adam_and_rejects_bad_schedules(monkeypatch):
    calls = []
    monkeypatch.setattr(prepare_map, "prepare_cell", lambda *a, **k: calls.append(k))
    assert prepare_map.main(["--adam-steps", "0"]) == 0
    assert calls[0]["options"].adam_steps == ()
    calls.clear()
    assert prepare_map.main(["--adam-steps", "1,2"]) == 2
    assert prepare_map.main(["--adam-steps", "bad"]) == 2
    assert not calls


def test_map_cache_environment_is_shared_by_preparation_and_old_runners(monkeypatch, tmp_path):
    monkeypatch.setenv("PIXEL_Q_MAP_CACHE_ROOT", str(tmp_path / "maps"))
    assert map_start.default_map_cache_root() == tmp_path / "maps"
    with variant.activated("JAM", "union", output_root=tmp_path / "streams") as (_, cfg, _, _):
        assert cfg.Q_INTEGRATION_MAP_CACHE_ROOT == tmp_path / "maps"
    args = SimpleNamespace(out=tmp_path / "new", parameterizations="q", truths="JAM",
                           data="union", scales="2", modes="exp", stream_counts="1", seed_base=1)
    assert sweep.plan(args)[0]["map_cache_root"] == str(tmp_path / "maps")


def test_sweep_map_failure_stops_before_any_stream_or_output(monkeypatch, tmp_path):
    source = tmp_path / "closure_JAM_truth" / "data" / "truthQ_2" / "truth.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    monkeypatch.setattr(sweep, "REPO", tmp_path)
    commands = []

    def check(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(sweep.subprocess, "run", check)
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **k: pytest.fail("launched a stream"))
    args = SimpleNamespace(out=tmp_path / "new", parameterizations="q", truths="JAM",
                           data="union", scales="2", modes="exp", stream_counts="1,2",
                           seed_base=1, samples=5, warmup=1, python="python", dry_run=False)
    with pytest.raises(ValueError, match="MAP preflight failed"):
        sweep.run_sweep(args)
    assert len(commands) == 1
    assert "closure_full_q_integration.prepare_map" in commands[0]
    assert "--check-only" in commands[0]
    assert not args.out.exists()


@pytest.mark.parametrize("failure", ("none", "kernel", "map"))
def test_submit_orders_preparation_and_production_and_propagates_failure(tmp_path, failure):
    repo = tmp_path / "closure"
    repo.mkdir()
    script = repo / "submit_closure.sh"
    shutil.copyfile(Path(__file__).parents[1] / "submit_closure.sh", script)
    launcher = repo / "run_closure.sh"
    launcher.write_text(
        '#!/bin/bash\nprintf "launcher %s\\n" "$*" >> "$CALL_LOG"\n'
        'if [[ "$*" == *--kernels-only* && "$TEST_FAIL" == kernel ]]; then exit 7; fi\n'
    )
    launcher.chmod(0o755)
    python = tmp_path / "pixel" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/bash\nprintf "python %s\\n" "$*" >> "$CALL_LOG"\n'
        'if [[ "$TEST_FAIL" == map ]]; then exit 8; fi\n'
    )
    python.chmod(0o755)
    log = tmp_path / "calls.txt"
    env = {**os.environ, "CALL_LOG": str(log), "TEST_FAIL": failure,
           "SLURM_SUBMIT_DIR": str(repo)}
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    commands = log.read_text().splitlines()
    assert "--kernels-only" in commands[0]
    if failure == "kernel":
        assert result.returncode == 7
        assert len(commands) == 1
    else:
        assert "closure_full_q_integration.prepare_map" in commands[1]
        assert "--lbfgs-maxiter 40000" in commands[1]
        assert "--newton-maxiter 2000" in commands[1]
        assert "--adam-steps 2000,2000,2000" in commands[1]
        assert "--adam-learning-rates 0.01,0.003,0.001" in commands[1]
        if failure == "map":
            assert result.returncode == 8
            assert len(commands) == 2
        else:
            assert result.returncode == 0
            assert len(commands) == 3
            assert "--parameterization q --warmup 1000 --samples 25000 --streams 16" in commands[2]
    assert ("Batch job finished" in result.stdout) == (failure == "none")
    occupied = repo / "campaign" / "JAM_Q2_exp_prod"
    occupied.mkdir()
    (occupied / "keep").write_text("existing data")
    blocked = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True)
    assert blocked.returncode == 2
    assert (occupied / "keep").read_text() == "existing data"
    assert log.read_text().splitlines() == commands
