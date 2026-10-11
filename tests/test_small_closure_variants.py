"""Configuration contract for the flowed and affine small-suite copies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from small_closure_variant import (
    AFFINE,
    FLOWED,
    GP_COVARIANCE_JITTER,
    activated,
    q_integration_activated,
)
from campaign.run_hmc_variant_scan import (
    DEFAULT_STREAMS,
    HOST_MEMORY_LIMIT_GIB,
    MAX_EXPENSIVE_SAMPLERS,
    MAX_PARALLEL_SAMPLERS,
    MEMORY_ADMISSION_GIB,
    MODE_MEMORY_GIB,
)


TRUTHS = ("JAM", "NNPDF")
VARIANTS = (FLOWED, AFFINE)
SOURCE_PACKAGES = {
    "JAM": "closure_JAM_truth_small",
    "NNPDF": "closure_NNPDF_truth_small",
}


def _normalization_model():
    return SimpleNamespace(
        has_bilinear=True,
        contour_partition=lambda: SimpleNamespace(
            n_ordinary=0, n_normalization=7, n_hubbard=12
        ),
    )


def _fixed_hs_model():
    return SimpleNamespace(
        has_bilinear=True,
        contour_partition=lambda: SimpleNamespace(
            n_ordinary=0, n_normalization=0, n_hubbard=1
        ),
    )


@pytest.mark.parametrize("truth", TRUTHS)
@pytest.mark.parametrize("spec", VARIANTS, ids=lambda spec: spec.name)
def test_every_variant_uses_one_em6_for_both_gp_covariance_forms(spec, truth):
    with activated(spec, truth) as (_package, cfg, fit, _runner):
        assert cfg.GP_JITTER == pytest.approx(1.0e-6)
        assert cfg.GP_ENVELOPE_JITTER == pytest.approx(1.0e-6)
        assert fit._gp_prior("t3").cov.jitter == pytest.approx(GP_COVARIANCE_JITTER)
        assert fit._envelope_prior("t3").cov.jitter == pytest.approx(GP_COVARIANCE_JITTER)


@pytest.mark.parametrize("truth", TRUTHS)
def test_flowed_copy_routes_every_bilinear_case_from_real_axis_to_t_point_two(truth):
    with activated(FLOWED, truth) as (_package, cfg, fit, _runner):
        auxiliary, options = fit._nuts_sampler_options("2", "both", _normalization_model())
        assert auxiliary == "sample"
        assert options["flow_time"] == pytest.approx(0.2)
        assert options["n_flow_steps"] == 8
        assert options["affine_reference_contour"] is False
        assert options["flow_from_affine_reference"] is False
        assert options["reference_budget_bytes"] == 110 * 1024**3
        assert options["reference_cache_path"].parent == cfg.INFERENCE_CACHE_ROOT
        auxiliary, fixed = fit._nuts_sampler_options(
            "2", cfg.TEST_MODE_SYNTHETIC_Z, _fixed_hs_model()
        )
        assert auxiliary == "sample"
        assert fixed["flow_time"] == pytest.approx(0.2)
        assert fixed["flow_from_affine_reference"] is False
        assert "affine_reference_contour" not in fixed


@pytest.mark.parametrize("truth", TRUTHS)
def test_affine_copy_routes_every_bilinear_case_to_unflowed_affine_map(truth):
    with activated(AFFINE, truth) as (_package, cfg, fit, _runner):
        auxiliary, options = fit._nuts_sampler_options("2", "both", _normalization_model())
        assert auxiliary == "sample"
        assert options["flow_time"] == pytest.approx(0.0)
        assert options["n_flow_steps"] == 8
        assert options["affine_reference_contour"] is True
        assert options["flow_from_affine_reference"] is False
        assert options["reference_budget_bytes"] is None
        assert options["reference_cache_path"].parent == cfg.INFERENCE_CACHE_ROOT
        auxiliary, fixed = fit._nuts_sampler_options(
            "2", cfg.TEST_MODE_SYNTHETIC_Z, _fixed_hs_model()
        )
        assert auxiliary == "sample"
        assert fixed["flow_time"] == pytest.approx(0.0)
        assert fixed["flow_from_affine_reference"] is False
        assert "affine_reference_contour" not in fixed


@pytest.mark.parametrize("truth", TRUTHS)
@pytest.mark.parametrize("spec", VARIANTS, ids=lambda spec: spec.name)
def test_variant_outputs_are_isolated_and_source_configuration_is_restored(spec, truth):
    package = SOURCE_PACKAGES[truth]
    module = __import__(f"{package}.config", fromlist=["config"])
    original = {
        "GP_JITTER": module.GP_JITTER,
        "GP_ENVELOPE_JITTER": module.GP_ENVELOPE_JITTER,
        "RESULTS_ROOT": module.RESULTS_ROOT,
        "INFERENCE_CACHE_ROOT": module.INFERENCE_CACHE_ROOT,
        "SMALL_CLOSURE_REJECT_UNRELIABLE_SAMPLES": (
            module.SMALL_CLOSURE_REJECT_UNRELIABLE_SAMPLES
        ),
    }
    with activated(spec, truth) as (_package, cfg, _fit, _runner):
        assert cfg.DATA_ROOT == module.DATA_ROOT
        assert cfg.RESULTS_ROOT == Path(__file__).parents[1] / f"closure_small_{spec.name}" / "outputs" / truth.lower() / "nuts" / "results"
        assert cfg.INFERENCE_CACHE_ROOT.parent == cfg.RESULTS_ROOT.parent
        assert cfg.SMALL_CLOSURE_REJECT_UNRELIABLE_SAMPLES is False
    for name, value in original.items():
        assert getattr(module, name) == value


@pytest.mark.parametrize("truth", TRUTHS)
def test_q_integration_copy_retains_unreliable_experimental_samples(truth):
    with q_integration_activated(truth) as (_package, cfg, _fit, _runner):
        assert cfg.SMALL_CLOSURE_REJECT_UNRELIABLE_SAMPLES is False


@pytest.mark.parametrize("truth", TRUTHS)
@pytest.mark.parametrize("spec", VARIANTS, ids=lambda spec: spec.name)
def test_hmc_scan_uses_hessian_metric_and_adapts_a_conservative_step(spec, truth):
    with activated(spec, truth, sampler="hmc") as (_package, cfg, fit, _runner):
        assert cfg.SMALL_CLOSURE_SAMPLER == "hmc"
        assert cfg.HMC_USE_HESSIAN_MASS is True
        assert cfg.HMC_STEP_SIZE == pytest.approx(0.05)
        assert cfg.HMC_N_LEAPFROG == 8
        assert cfg.HMC_WARMUP == 100
        assert cfg.HMC_ADAPT_STEP_SIZE is True
        assert cfg.HMC_ADAPT_MASS is False
        auxiliary, options = fit._hmc_sampler_options(
            "2", "both", _normalization_model()
        )
        assert auxiliary == "sample"
        assert options["use_hessian_mass"] is True
        assert options["step_size"] == pytest.approx(0.05)
        assert options["n_leapfrog"] == 8
        assert options["adapt_step_size"] is True
        assert options["target_accept"] == pytest.approx(0.8)
        assert options["flow_time"] == pytest.approx(spec.flow_time)
        assert options["affine_reference_contour"] is spec.affine_reference_contour
        assert options["flow_from_affine_reference"] is spec.flow_from_affine_reference
        assert options["hessian_cache_path"].parent == cfg.INFERENCE_CACHE_ROOT


def test_hmc_scan_four_worker_memory_policy():
    assert DEFAULT_STREAMS == 1
    assert HOST_MEMORY_LIMIT_GIB == pytest.approx(36.0)
    assert MEMORY_ADMISSION_GIB < HOST_MEMORY_LIMIT_GIB
    assert MAX_PARALLEL_SAMPLERS == 4
    assert MAX_EXPENSIVE_SAMPLERS == 2
    assert (
        2 * MODE_MEMORY_GIB["both"] + 2 * MODE_MEMORY_GIB["dis"]
        <= MEMORY_ADMISSION_GIB
    )
    assert 3 * MODE_MEMORY_GIB["both"] > MEMORY_ADMISSION_GIB
