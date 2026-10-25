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
                                "gradient_inf_norm": 0,
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
        "--report-every", "150",
        "--adam-steps", "100,200", "--adam-learning-rates", "0.02,0.005",
    ]) == 0
    assert [call[0][2:] for call in calls] == [("2", "dis"), ("2", "exp"), ("3", "dis"), ("3", "exp")]
    for _, kwargs in calls:
        assert kwargs["options"].lbfgs_maxiter == 80000
        assert kwargs["options"].newton_maxiter == 8000
        assert kwargs["options"].lbfgs_maxfun == 900000
        assert kwargs["options"].report_every == 150
        assert kwargs["options"].adam_steps == (100, 200)
        assert kwargs["options"].adam_learning_rates == (0.02, 0.005)
        assert kwargs["resume"]
    assert prepare_map.main(["--newton-maxiter", "-1"]) == 2
    assert prepare_map.main(["--report-every", "0"]) == 2
    assert prepare_map.main(["--report-every", "125"]) == 2


def test_cli_out_places_map_cache_inside_campaign(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(prepare_map, "prepare_cell", lambda *a, **k: calls.append(k))
    output = tmp_path / "study"
    assert prepare_map.main(["--out", str(output)]) == 0
    assert calls[0]["cache_root"] == output / "_q_map_cache"
    with pytest.raises(SystemExit):
        prepare_map.main(["--out", str(output), "--cache-root", str(tmp_path)])


def test_map_progress_reports_recent_stage_rate_and_inf_norm(capsys):
    report = prepare_map._progress_reporter(100)
    base = {
        "stage": "lbfgs", "action": -20.0, "gradient_norm": 10.0,
        "gradient_inf_norm": 7.0, "status": "running",
        "checkpoint_path": "checkpoint.npz",
        "stages": {"lbfgs": {"maxiter": 400}},
    }
    for iteration, elapsed in ((0, 0.0), (50, 2.0), (100, 5.0),
                               (150, 9.0), (200, 15.0)):
        report(None, {**base, "iteration": iteration, "elapsed_seconds": elapsed})
    output = capsys.readouterr().out
    assert "iteration=50" not in output and "iteration=150" not in output
    assert "iteration=100" in output and "iteration=200" in output
    assert "recent=100 steps/5.0s (20.00 steps/s)" in output
    assert "remaining=300 steps eta_to_limit=15.0s" in output
    assert "recent=100 steps/10.0s (10.00 steps/s)" in output
    assert "remaining=200 steps eta_to_limit=20.0s" in output
    assert "|gradient|_inf=7" in output

    report(None, {**base, "iteration": 225, "elapsed_seconds": 17.5,
                  "status": "stage_complete",
                  "stages": {"lbfgs": {"maxiter": 400, "message": "done"}}})
    output = capsys.readouterr().out
    assert "iteration=225" in output and "lbfgs: done" in output
    assert "eta_to_limit" not in output

    report(None, {**base, "stage": "adam_1", "iteration": 0,
                  "elapsed_seconds": 17.5, "stages": {"adam_1": {"maxiter": 200}}})
    report(None, {**base, "stage": "adam_1", "iteration": 100,
                  "elapsed_seconds": 19.5, "stages": {"adam_1": {"maxiter": 200}}})
    assert "recent=100 steps/2.0s (50.00 steps/s)" in capsys.readouterr().out


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
    monkeypatch.delenv("PIXEL_Q_MAP_CACHE_ROOT", raising=False)
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
    prepared_cache = args.out / "_q_map_cache"
    prepared_cache.mkdir(parents=True)
    (prepared_cache / "candidate.npz").write_bytes(b"saved")
    with pytest.raises(ValueError, match="MAP preflight failed"):
        sweep.run_sweep(args)
    assert len(commands) == 1
    assert "closure_full_q_integration.prepare_map" in commands[0]
    assert "--check-only" in commands[0]
    assert (prepared_cache / "candidate.npz").read_bytes() == b"saved"
    assert not (args.out / "manifest.json").exists()


@pytest.mark.parametrize("failure", ("none", "kernel", "map"))
def test_submit_orders_preparation_and_production_and_propagates_failure(tmp_path, failure):
    repo = tmp_path / "closure"
    repo.mkdir()
    script = repo / "submit_closure.sh"
    shutil.copyfile(Path(__file__).parents[1] / "submit_closure.sh", script)
    launcher = repo / "run_closure.sh"
    launcher.write_text(
        '#!/bin/bash\nprintf "launcher %s map=%s\\n" "$*" "$PIXEL_Q_MAP_CACHE_ROOT" >> "$CALL_LOG"\n'
        'if [[ "$*" == *--kernels-only* && "$TEST_FAIL" == kernel ]]; then exit 7; fi\n'
    )
    launcher.chmod(0o755)
    python = tmp_path / "pixel" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/bash\nprintf "python %s map=%s\\n" "$*" "$PIXEL_Q_MAP_CACHE_ROOT" >> "$CALL_LOG"\n'
        'if [[ "$TEST_FAIL" == map ]]; then exit 8; fi\n'
    )
    python.chmod(0o755)
    log = tmp_path / "calls.txt"
    env = {**os.environ, "CALL_LOG": str(log), "TEST_FAIL": failure,
           "SLURM_SUBMIT_DIR": str(repo)}
    occupied = repo / "campaign" / "JAM_Q2_exp_prod"
    prepared_cache = occupied / "_q_map_cache"
    prepared_cache.mkdir(parents=True)
    (prepared_cache / "candidate.npz").write_bytes(b"saved")
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    commands = log.read_text().splitlines()
    expected_cache = str(prepared_cache)
    assert all(f"map={expected_cache}" in command for command in commands)
    assert "--kernels-only" in commands[0]
    if failure == "kernel":
        assert result.returncode == 7
        assert len(commands) == 1
    else:
        assert "closure_full_q_integration.prepare_map" in commands[1]
        assert "--out campaign/JAM_Q2_exp_prod" in commands[1]
        assert "--lbfgs-maxiter 10000" in commands[1]
        assert "--newton-maxiter 2000" in commands[1]
        assert "--adam-steps 4000,4000,4000" in commands[1]
        assert "--adam-learning-rates 0.01,0.003,0.001" in commands[1]
        assert "--report-every 100" in commands[1]
        if failure == "map":
            assert result.returncode == 8
            assert len(commands) == 2
        else:
            assert result.returncode == 0
            assert len(commands) == 3
            assert "--parameterization q --warmup 1000 --samples 25000 --streams 16" in commands[2]
    assert ("Batch job finished" in result.stdout) == (failure == "none")
    (occupied / "keep").write_text("existing data")
    blocked = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True)
    assert blocked.returncode == 2
    assert (occupied / "keep").read_text() == "existing data"
    assert log.read_text().splitlines() == commands


def test_top_level_q_launcher_allows_only_prepared_campaign_map(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    script = repo / "run_closure.sh"
    shutil.copyfile(Path(__file__).parents[1] / "run_closure.sh", script)
    source = repo / "closure_JAM_truth" / "data"
    (source / "truthQ_2").mkdir(parents=True)
    (source / "truthQ_2" / "truth.json").write_text("{}")
    (source / "dis_manifest.json").write_text("{}")
    python = repo / "fake_python"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    output = repo / "campaign" / "study"
    cache = output / "_q_map_cache"
    cache.mkdir(parents=True)
    (cache / "candidate.npz").write_bytes(b"saved")
    command = ["bash", str(script), "--truth", "JAM", "--data", "union",
               "--Q", "2", "--modes", "exp", "--parameterization", "q",
               "--out", str(output), "--python", str(python), "--no-plots"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (cache / "candidate.npz").read_bytes() == b"saved"
    (output / "manifest.json").write_text("{}")
    blocked = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert blocked.returncode == 2
    assert "production artifacts" in blocked.stderr
