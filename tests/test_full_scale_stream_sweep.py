"""Plan and isolate full-scale closure scaling streams without sampling."""

from __future__ import annotations

from argparse import Namespace
import importlib
import json

import pytest

from campaign import run_full_scale_stream_sweep as sweep


def _args(tmp_path, **overrides):
    values = dict(
        out=tmp_path / "scan", parameterizations="hs,q", truths="JAM",
        data="union,JAM", scales="2", modes="dis,exp",
        stream_counts="1,2", samples=5, seed_base=42,
        python="python", dry_run=True,
    )
    values.update(overrides)
    return Namespace(**values)


def test_plan_comma_lists_and_repeated_stream_seeds(tmp_path):
    cells = sweep.plan(_args(tmp_path))
    assert len(cells) == 2 * 2 * 2 * 2
    assert sum(len(cell["streams"]) for cell in cells) == 24
    first_one = cells[0]["streams"][0]
    first_two = cells[8]["streams"][0]
    assert first_one["seed"] == first_two["seed"]
    assert first_one["output"] != first_two["output"]
    assert cells[0]["parameterization"] == "hs"
    assert cells[8]["stream_count"] == 2


def test_sweep_dry_run_does_not_write(tmp_path):
    args = _args(tmp_path, parameterizations="q", data="union",
                 modes="exp", stream_counts="1,2")
    assert sweep.run_sweep(args) == 0
    assert not args.out.exists()


@pytest.mark.parametrize("parameterization", ("hs", "q"))
def test_worker_isolates_outputs_and_restores_config(
    tmp_path, monkeypatch, parameterization,
):
    cfg = importlib.import_module("closure_JAM_truth.config")
    runner = importlib.import_module("closure_JAM_truth.run_closure")
    original = (cfg.RESULTS_ROOT, cfg.INFERENCE_CACHE_ROOT,
                cfg.MCMC_SAMPLES, cfg.MCMC_SEED)
    observed = {}

    def fake_run_one(scale, mode, **kwargs):
        observed.update(
            scale=scale, mode=mode, kwargs=kwargs,
            results=cfg.RESULTS_ROOT, cache=cfg.INFERENCE_CACHE_ROOT,
            samples=cfg.MCMC_SAMPLES, seed=cfg.MCMC_SEED,
            q_enabled=getattr(cfg, "Q_INTEGRATION_ENABLED", False),
        )

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    output = tmp_path / parameterization
    args = Namespace(parameterization=parameterization, truth="JAM",
                     data="union", scale="2", mode="exp", samples=17,
                     seed=19, output=output)
    assert sweep.run_worker(args) == 0
    assert observed == {
        "scale": "2", "mode": "exp", "kwargs": {"make_plots": False},
        "results": output / "results",
        "cache": output / "inference_cache",
        "samples": 17, "seed": 19,
        "q_enabled": parameterization == "q",
    }
    assert (cfg.RESULTS_ROOT, cfg.INFERENCE_CACHE_ROOT,
            cfg.MCMC_SAMPLES, cfg.MCMC_SEED) == original
    assert json.loads((output / "stream.json").read_text())["status"] == "complete"
    assert (output / "inference_cache").is_dir()
