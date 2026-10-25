"""Plan and isolate full-scale closure scaling streams without sampling."""

from __future__ import annotations

from argparse import Namespace
import importlib
import json

import numpy as np
import pytest

from campaign import run_full_scale_stream_sweep as sweep
from campaign import report_full_scale_stream_sweep as report


def _args(tmp_path, **overrides):
    values = dict(
        out=tmp_path / "scan", parameterizations="hs,q", truths="JAM",
        data="union,JAM", scales="2", modes="dis,exp",
        stream_counts="1,2", samples=5, seed_base=42,
        python="python", dry_run=True, warmup=None,
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
                 modes="exp", stream_counts="1,2", warmup=500)
    assert sweep.run_sweep(args) == 0
    assert not args.out.exists()
    command = sweep.worker_command("python", sweep.plan(args)[0],
                                   sweep.plan(args)[0]["streams"][0], 5, 500)
    assert command[-2:] == ["--warmup", "500"]
    assert command[command.index("--map-cache-root") + 1] == str(
        sweep.default_map_cache_root()
    )


def test_warmup_override_rejects_hs_and_negative_values(tmp_path):
    with pytest.raises(ValueError, match="only for q-integration"):
        sweep.run_sweep(_args(tmp_path, warmup=5))
    with pytest.raises(ValueError, match="nonnegative"):
        sweep.run_sweep(_args(tmp_path, parameterizations="q", warmup=-1))


@pytest.mark.parametrize("parameterization", ("hs", "q"))
def test_worker_isolates_outputs_and_restores_config(
    tmp_path, monkeypatch, parameterization,
):
    from pixel.infer.samplers.nuts import NUTSSamples

    cfg = importlib.import_module("closure_JAM_truth.config")
    runner = importlib.import_module("closure_JAM_truth.run_closure")
    original = (cfg.RESULTS_ROOT, cfg.INFERENCE_CACHE_ROOT,
                cfg.MCMC_SAMPLES, cfg.MCMC_SEED,
                getattr(cfg, "Q_INTEGRATION_WARMUP", None))
    observed = {}

    def fake_run_one(scale, mode, **kwargs):
        artifact_hook = kwargs.pop("artifact_hook")
        observed.update(
            scale=scale, mode=mode, kwargs=kwargs,
            results=cfg.RESULTS_ROOT, cache=cfg.INFERENCE_CACHE_ROOT,
            samples=cfg.MCMC_SAMPLES, seed=cfg.MCMC_SEED,
            q_enabled=getattr(cfg, "Q_INTEGRATION_ENABLED", False),
            q_warmup=getattr(cfg, "Q_INTEGRATION_WARMUP", None),
        )
        draws = np.arange(17, dtype=float).reshape(-1, 1)
        chain = NUTSSamples(
            samples=draws, evidence_values=np.zeros(17),
            weights=np.full(17, 1 / 17),
            tree_depth=np.full(17, 2), n_leapfrog=np.full(17, 3),
            accept_prob=np.full(17, 0.9), energy_error=np.zeros(17),
            divergent=np.zeros(17, dtype=bool),
            reached_max_depth=np.zeros(17, dtype=bool),
            initial=np.zeros(1), step_size=0.1, n_warmup=5,
            inv_mass=np.eye(1), param_labels=("p",),
            metadata={"metric_structure": "dense"},
        )
        artifact_hook({"samples": chain, "timing_seconds": {"inference": 1.25}})

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    output = tmp_path / parameterization
    args = Namespace(parameterization=parameterization, truth="JAM",
                     data="union", scale="2", mode="exp", samples=17,
                     seed=19, output=output,
                     warmup=5 if parameterization == "q" else None,
                     map_cache_root=tmp_path / "shared_q_map_cache")
    assert sweep.run_worker(args) == 0
    assert observed == {
        "scale": "2", "mode": "exp", "kwargs": {"make_plots": False},
        "results": output / "results",
        "cache": output / "inference_cache",
        "samples": 17, "seed": 19,
        "q_enabled": parameterization == "q",
        "q_warmup": 5 if parameterization == "q" else None,
    }
    assert (cfg.RESULTS_ROOT, cfg.INFERENCE_CACHE_ROOT,
            cfg.MCMC_SAMPLES, cfg.MCMC_SEED,
            getattr(cfg, "Q_INTEGRATION_WARMUP", None)) == original
    assert json.loads((output / "stream.json").read_text())["status"] == "complete"
    assert json.loads((output / "stream.json").read_text())["warmup_requested"] == args.warmup
    assert (output / "chain.npz").is_file()
    assert (output / "inference_summary.json").is_file()
    assert (output / "inference_cache").is_dir()


def test_report_uses_pixel_nuts_chains_and_pixel_split_rhat(tmp_path):
    from pixel.infer.diagnostics.multichain import split_r_hat
    from pixel.infer.samplers.nuts import NUTSSamples

    streams = []
    chains = []
    for number in (1, 2):
        output = tmp_path / "exp" / f"stream_{number:02d}"
        output.mkdir(parents=True)
        draws = np.random.default_rng(number).normal(size=(30, 2))
        energy = np.zeros(30)
        energy[0] = -4.0
        chain = NUTSSamples(
            samples=draws, evidence_values=np.zeros(30),
            weights=np.full(30, 1 / 30), tree_depth=np.full(30, 3),
            n_leapfrog=np.full(30, 7), accept_prob=np.full(30, 0.9),
            energy_error=energy, divergent=np.zeros(30, dtype=bool),
            reached_max_depth=np.zeros(30, dtype=bool),
            initial=np.zeros(2), step_size=0.05, n_warmup=10,
            inv_mass=np.eye(2), param_labels=("a", "b"),
            metadata={"metric_structure": "dense",
                      "hessian_mass_mode": "dense"},
        )
        chain.save_npz(output / "chain.npz")
        (output / "inference_summary.json").write_text(
            json.dumps(sweep._json_safe(chain.summary()))
        )
        (output / "stream.json").write_text(json.dumps({
            "status": "complete", "chain_kind": "nuts", "samples": 30,
            "wall_seconds": 20.0, "sampler_wall_seconds": 15.0,
        }))
        streams.append({"output": str(output)})
        chains.append(chain)
    cell = dict(parameterization="q", truth="JAM", data="union",
                scale="2", mode="exp", stream_count=2, streams=streams)
    path, result = report.render_cell(cell)
    assert path.is_file()
    assert result["split_rhat_max"] == pytest.approx(
        float(np.max(split_r_hat([chain.samples for chain in chains])))
    )
    assert result["streams"][0]["tau_int_max"] == pytest.approx(
        chains[0].summary()["tau_int_max"]
    )
    assert result["streams"][0]["max_abs_delta_h"] == 4.0
    assert result["streams"][0]["average_depth"] == 3.0
    assert result["streams"][0]["steps_per_trajectory"] == 7.0
    assert result["streams"][0]["acceptance_statistic"] == pytest.approx(0.9)
    assert result["streams"][0]["acceptance_rate"] is None
    assert result["streams"][0]["metric_structure"] == "dense"
    assert result["streams"][0]["dense_metric_realized"] is True


def test_report_distinguishes_hmc_rate_invalid_and_missing_rhat(tmp_path):
    from pixel.infer.samplers.hmc import HMCSamples

    output = tmp_path / "hmc" / "stream_01"
    output.mkdir(parents=True)
    chain = HMCSamples(
        samples=np.random.default_rng(9).normal(size=(20, 2)),
        evidence_values=np.zeros(20), weights=np.full(20, 1 / 20),
        accepted=np.array([True] * 15 + [False] * 5),
        delta_hamiltonian=np.array([-5.0] + [0.1] * 19),
        accept_prob=np.full(20, 0.8), initial=np.zeros(2),
        step_size=0.05, n_leapfrog=8, n_warmup=10,
        inv_mass=np.eye(2), param_labels=("a", "b"), n_invalid=2,
        metadata={"metric_structure": "dense"},
    )
    chain.save_npz(output / "chain.npz")
    (output / "inference_summary.json").write_text(
        json.dumps(sweep._json_safe(chain.summary()))
    )
    (output / "stream.json").write_text(json.dumps({
        "status": "complete", "chain_kind": "hmc", "samples": 20,
        "wall_seconds": 12.0, "sampler_wall_seconds": 9.0,
    }))
    cell = dict(parameterization="hs", truth="JAM", data="union",
                scale="2", mode="exp", stream_count=1,
                streams=[{"output": str(output)}])
    _path, result = report.render_cell(cell)
    row = result["streams"][0]
    assert result["split_rhat_max"] is None
    assert row["acceptance_rate"] == 0.75
    assert row["acceptance_statistic"] == pytest.approx(0.8)
    assert row["num_invalid"] == 2
    assert row["num_divergence"] is None
    assert row["average_depth"] is None
    assert row["max_abs_delta_h"] == 5.0
    assert row["steps_per_trajectory"] == 8


def test_inference_summary_json_keeps_complex_and_nonfinite_values():
    assert sweep._json_safe({"z": 1 + 2j, "bad": np.nan}) == {
        "z": {"real": 1.0, "imag": 2.0}, "bad": None,
    }
