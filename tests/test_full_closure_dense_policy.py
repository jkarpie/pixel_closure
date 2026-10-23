"""Exercise the full-scale HS and explicit-q inference policies at dispatch."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from closure_full_q_integration import variant
from closure_full_q_integration.variant import SOURCE_SUITES, activated


class _ReachedSampler(Exception):
    pass


def _capture_call(monkeypatch, fit, *, has_bilinear, has_nonlinear=False,
                  n_ordinary=0, entry="infer_mcmc", n_samples=3):
    model = SimpleNamespace(
        datasets=[], free_vector=lambda: np.zeros(3),
        has_bilinear=has_bilinear, has_nonlinear=has_nonlinear,
        contour_partition=lambda: SimpleNamespace(
            n_ordinary=n_ordinary, n_normalization=0, n_hubbard=3,
        ),
    )
    captured = {}
    monkeypatch.setattr(
        fit, "build_analysis",
        lambda *args, **kwargs: (SimpleNamespace(compile=lambda: model), {}),
    )
    monkeypatch.setattr(fit, "_iterate_t0", lambda *args, **kwargs: ({}, 0))

    def record(_model, **kwargs):
        captured.update(kwargs)
        raise _ReachedSampler

    monkeypatch.setattr(
        fit if entry == "infer_mcmc" else fit.pixel,
        entry if entry == "infer_mcmc" else "run_q_mcmc",
        record,
    )
    with pytest.raises(_ReachedSampler):
        kwargs = {"use_kernel_cache": False}
        if n_samples is not None:
            kwargs["n_samples"] = n_samples
        fit.run_fit("2", "exp", **kwargs)
    return captured


@pytest.mark.parametrize("package", SOURCE_SUITES.values())
def test_hs_metric_is_dense_and_nuts_uses_library_acceptance_default(
    package, monkeypatch,
):
    cfg = importlib.import_module(f"{package}.config")
    fit = importlib.import_module(f"{package}.fit")
    assert cfg.HS_HESSIAN_MASS_MODE == "dense"
    nuts = _capture_call(monkeypatch, fit, has_bilinear=True)
    assert nuts["sampler"] == "nuts"
    assert nuts["hessian_mass_mode"] == "dense"
    assert nuts["adapt_mass"] is False
    assert "target_accept" not in nuts
    hmc = _capture_call(
        monkeypatch, fit, has_bilinear=True, n_ordinary=2,
    )
    assert hmc["sampler"] == "hmc"
    assert hmc["hessian_mass_mode"] == "dense"


@pytest.mark.parametrize(("truth", "data", "package"), [
    (*key, package) for key, package in SOURCE_SUITES.items()
])
def test_full_q_variant_routes_dense_nuts_with_ninety_percent_target(
    truth, data, package, monkeypatch,
):
    import closure_full_q_integration.map_start as map_start

    cfg = importlib.import_module(f"{package}.config")
    fit = importlib.import_module(f"{package}.fit")
    map_calls = []
    coordinates = object()
    center = np.array([0.25, -0.5, 0.75])

    def fake_map_start(model, *, prior_rcond, cache_root):
        map_calls.append((model, prior_rcond, cache_root))
        return coordinates, center, {
            "action": 1.0, "gradient_norm": 1.0e-7,
            "cache_status": "miss", "fingerprint": "test-map",
        }

    monkeypatch.setattr(map_start, "prepare_q_map_start", fake_map_start)
    previous_results = cfg.RESULTS_ROOT
    previous_cache = cfg.INFERENCE_CACHE_ROOT
    assert not hasattr(cfg, "Q_INTEGRATION_ENABLED")
    with activated(truth, data) as (selected, active_cfg, _fit, _runner):
        assert selected == package
        assert active_cfg is cfg
        assert cfg.Q_INTEGRATION_ENABLED is True
        assert cfg.RESULTS_ROOT != previous_results
        assert cfg.INFERENCE_CACHE_ROOT != previous_cache
        options = _capture_call(
            monkeypatch, fit, has_bilinear=True, has_nonlinear=True,
            entry="run_q_mcmc",
        )
        assert options["sampler"] == "nuts"
        assert options["metric"] == "legacy_hessian_dense"
        assert options["target_accept"] == pytest.approx(0.90)
        assert options["adapt_mass"] is False
        assert options["prior_rcond"] == cfg.DIRECT_PRIOR_RCOND == 0.0
        assert options["coordinates"] is coordinates
        np.testing.assert_array_equal(options["initial"], center)
        assert options["metadata"]["closure_q_map_start"]["fingerprint"] == "test-map"
        assert len(map_calls) == 1
        assert map_calls[0][1] == 0.0
        assert map_calls[0][2] == cfg.Q_INTEGRATION_MAP_CACHE_ROOT
        cfg.MCMC_SAMPLES = 17
        cfg.MCMC_SEED = 19
        defaults = _capture_call(
            monkeypatch, fit, has_bilinear=False,
            entry="run_q_mcmc", n_samples=None,
        )
        assert defaults["n_samples"] == 17
        assert defaults["seed"] == 19
        assert len(map_calls) == 2
    assert not hasattr(cfg, "Q_INTEGRATION_ENABLED")
    assert cfg.RESULTS_ROOT == previous_results
    assert cfg.INFERENCE_CACHE_ROOT == previous_cache


def test_full_q_cli_forwards_source_options_and_restores_config(monkeypatch):
    cfg = importlib.import_module("closure_JAM_truth.config")
    runner = importlib.import_module("closure_JAM_truth.run_closure")
    original = (cfg.MCMC_SAMPLES, cfg.MCMC_SEED, tuple(sys.argv))
    observed = {}

    def fake_main():
        observed.update(
            argv=tuple(sys.argv), draws=cfg.MCMC_SAMPLES, seed=cfg.MCMC_SEED,
            metric=cfg.Q_INTEGRATION_METRIC,
            target=cfg.Q_INTEGRATION_TARGET_ACCEPT,
        )

    monkeypatch.setattr(runner, "main", fake_main)
    variant.main([
        "--truth", "JAM", "--data", "union",
        "--n-samples", "17", "--seed", "19",
        "--Q", "2", "--modes", "both", "--no-plots",
    ])
    assert observed == {
        "argv": ("closure_full_q_integration.run_closure", "--Q", "2",
                 "--modes", "both", "--no-plots"),
        "draws": 17, "seed": 19,
        "metric": "legacy_hessian_dense", "target": 0.90,
    }
    assert (cfg.MCMC_SAMPLES, cfg.MCMC_SEED, tuple(sys.argv)) == original


def test_full_q_variant_accepts_private_output_root(tmp_path):
    cfg = importlib.import_module("closure_JAM_truth.config")
    original = (cfg.RESULTS_ROOT, cfg.INFERENCE_CACHE_ROOT)
    with activated("JAM", "union", output_root=tmp_path / "stream"):
        assert cfg.RESULTS_ROOT == tmp_path / "stream" / "results"
        assert cfg.INFERENCE_CACHE_ROOT == tmp_path / "stream" / "inference_cache"
        assert cfg.Q_INTEGRATION_MAP_CACHE_ROOT == tmp_path / "_map_cache"
    assert (cfg.RESULTS_ROOT, cfg.INFERENCE_CACHE_ROOT) == original


@pytest.mark.parametrize("package", SOURCE_SUITES.values())
def test_full_runner_exposes_pixel_inference_result_to_artifact_hook(
    package, monkeypatch, tmp_path,
):
    cfg = importlib.import_module(f"{package}.config")
    fit = importlib.import_module(f"{package}.fit")
    runner = importlib.import_module(f"{package}.run_closure")
    result = {"samples": object()}
    seen = []
    monkeypatch.setattr(cfg, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(fit, "load_truth", lambda _q: {})
    monkeypatch.setattr(fit, "run_fit", lambda _q, _mode: result)
    monkeypatch.setattr(fit, "summarize", lambda *_args: {"ok": True})
    summary = runner.run_one(
        "2", "exp", make_plots=False, write_outputs=False,
        artifact_hook=seen.append,
    )
    assert summary == {"ok": True}
    assert seen == [result]
