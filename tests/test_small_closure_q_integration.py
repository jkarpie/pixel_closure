"""Contracts for the explicit retained-field small-closure copy."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pixel.infer import DirectMCMCSamples

from small_closure_variant import (
    GP_COVARIANCE_JITTER,
    q_integration_activated,
)


TRUTHS = ("JAM", "NNPDF")
SOURCE_PACKAGES = {
    "JAM": "closure_JAM_truth_small",
    "NNPDF": "closure_NNPDF_truth_small",
}


@pytest.mark.parametrize("truth", TRUTHS)
def test_q_copy_selects_prior_whitened_explicit_field_nuts(truth):
    with q_integration_activated(truth) as (_package, cfg, fit, _runner):
        assert cfg.SMALL_CLOSURE_SAMPLER == "q"
        assert fit._closure_sampler(SimpleNamespace(), "both") == "q"
        engine, options = fit._q_sampler_options()
        assert engine == "nuts"
        assert options["prior_rcond"] == 0.0
        assert options["metric"] == "prior_whitened"
        assert options["n_warmup"] == 100
        assert options["step_size"] == pytest.approx(0.1)
        assert options["max_tree_depth"] == 6
        assert options["adapt_mass"] is False
        assert cfg.GP_JITTER == pytest.approx(GP_COVARIANCE_JITTER)
        assert cfg.GP_ENVELOPE_JITTER == pytest.approx(GP_COVARIANCE_JITTER)


@pytest.mark.parametrize("truth", TRUTHS)
def test_q_copy_isolates_outputs_and_restores_source_configuration(truth):
    package = SOURCE_PACKAGES[truth]
    source = __import__(f"{package}.config", fromlist=["config"])
    original = {
        name: getattr(source, name, None)
        for name in (
            "SMALL_CLOSURE_SAMPLER",
            "RESULTS_ROOT",
            "INFERENCE_CACHE_ROOT",
        )
    }
    with q_integration_activated(truth) as (_package, cfg, _fit, _runner):
        expected = (
            Path(__file__).parents[1]
            / "closure_small_q_integration"
            / "outputs"
            / truth.lower()
        )
        assert cfg.RESULTS_ROOT == expected / "results"
        assert cfg.INFERENCE_CACHE_ROOT == expected / "inference_cache"
        assert cfg.DATA_ROOT == source.DATA_ROOT
    for name, value in original.items():
        assert getattr(source, name, None) == value


def _direct_fixture():
    partition = SimpleNamespace(
        n_hubbard=1,
        theta_labels=("table normalization / amp",),
    )
    model = SimpleNamespace(
        fields=(SimpleNamespace(name="q", nodes=np.asarray([0.2, 0.8])),),
        q_layout=lambda: (("q", 0, 2),),
        free_partition=lambda: partition,
        free_labels=lambda: (
            "table normalization / amp",
            "hubbard[0]",
        ),
        merge_free_vector=lambda theta, hubbard: np.concatenate((theta, hubbard)),
        _layout=SimpleNamespace(M=np.zeros(2)),
    )
    coordinates = SimpleNamespace(
        dimension=3,
        prior_subspace=SimpleNamespace(full_dimension=2, rank=2),
    )
    chain = SimpleNamespace(
        samples=np.zeros((3, 3)),
        weights=np.ones(3) / 3.0,
        summary=lambda: {},
    )
    samples = DirectMCMCSamples(
        model=model,
        chain=chain,
        coordinates=coordinates,
        samples=np.zeros((3, 3)),
        theta_samples=np.asarray([[-1.0], [0.0], [2.0]]),
        q_samples=np.asarray([[1.0, 3.0], [2.0, 5.0], [6.0, 7.0]]),
        action_values=np.zeros(3),
        weights=np.ones(3) / 3.0,
        param_labels=("theta", "q0", "q1"),
        metadata={"sampler": "q_nuts"},
    )
    return model, samples


@pytest.mark.parametrize("truth", TRUTHS)
def test_q_samples_are_reconstructed_as_physical_fields_not_model_vectors(
    monkeypatch, truth
):
    with q_integration_activated(truth) as (_package, _cfg, fit, _runner):
        model, samples = _direct_fixture()
        predictive_mean = np.asarray([10.0, 20.0, 30.0])
        predictive_cov = np.diag([1.0, 4.0, 9.0])
        monkeypatch.setattr(
            DirectMCMCSamples,
            "posterior_predictive_moments",
            lambda self: {
                "mean": predictive_mean,
                "cov": predictive_cov,
                "datasets": [],
            },
        )

        marginal = fit.q_integration_marginal_posterior(model, samples)
        np.testing.assert_allclose(marginal["q"][1], [3.0, 5.0])
        np.testing.assert_allclose(
            marginal["q"][2],
            np.std(samples.q_samples, axis=0, ddof=0),
        )
        np.testing.assert_allclose(
            fit._weighted_mean_vec(samples, model=model),
            [1.0 / 3.0, 0.0],
        )
        normalization = fit.fitted_normalization_posterior(model, samples)
        assert normalization["table"]["mean"] == pytest.approx(1.0 / 3.0)
        assert normalization["table"]["std"] == pytest.approx(
            np.std(samples.theta_samples[:, 0], ddof=0)
        )
        cached_mean, cached_cov = samples._pixel_direct_predictive_moments
        np.testing.assert_array_equal(cached_mean, predictive_mean)
        np.testing.assert_array_equal(cached_cov, predictive_cov)


@pytest.mark.parametrize("truth", TRUTHS)
def test_q_predictive_plot_band_uses_matched_draw_cache(truth):
    with q_integration_activated(truth) as (package, _cfg, _fit, _runner):
        plotting = __import__(f"{package}.plot_datasets", fromlist=["plot_datasets"])
        blocks = SimpleNamespace(
            M=np.zeros(2),
            bilinear_M=np.zeros(1),
            g=np.zeros(2),
        )
        model = SimpleNamespace(
            _rebuild=lambda vec: vec,
            _blocks=lambda theta: blocks,
        )
        samples = SimpleNamespace(
            _pixel_direct_predictive_moments=(
                np.asarray([1.0, 2.0, 3.0]),
                np.diag([4.0, 9.0, 16.0]),
            )
        )
        linear, linear_std, dy, dy_std, returned = (
            plotting._posterior_predictive_bands(model, samples, np.zeros(0))
        )
        np.testing.assert_array_equal(linear, [1.0, 2.0])
        np.testing.assert_array_equal(linear_std, [2.0, 3.0])
        np.testing.assert_array_equal(dy, [3.0])
        np.testing.assert_array_equal(dy_std, [4.0])
        assert returned is blocks
