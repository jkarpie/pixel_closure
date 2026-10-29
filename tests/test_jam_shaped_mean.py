"""Focused checks for free anchored beta means in the JAM closure."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from campaign.run_full_scale_stream_sweep import prior_identity
from closure_JAM_truth import config as cfg, fit


@pytest.fixture
def shaped(monkeypatch):
    amplitudes = {**cfg.GP_AMPLITUDES, **{name: 0.3 for name in cfg.ALL_FIELDS}}
    monkeypatch.setattr(cfg, "PRIOR_FORM", "shaped_logrbf")
    monkeypatch.setattr(cfg, "SHAPED_GP_SIGMA", 0.3)
    monkeypatch.setattr(cfg, "VANISHING_PDF_PRIOR", "zero_unit")
    monkeypatch.setattr(cfg, "GP_AMPLITUDES", amplitudes)
    monkeypatch.setattr(cfg, "PDF_GP_LENGTH_LOG", float(np.log(10)))


def test_shaped_means_have_three_gaussian_parameters_and_anchor(shaped):
    x0 = cfg.SHAPED_MEAN_X0
    for field in cfg.ALL_FIELDS:
        prior = fit.gp_prior(field)
        mean = prior.mean
        expected = cfg.shaped_mean_priors(field)
        assert float(mean(jnp.array([x0]))[0]) == pytest.approx(1.0)
        for name in ("M", "a", "b"):
            param = getattr(mean, name)
            assert param.frozen is False
            assert float(param.value) == expected[name][0]
            assert float(param.prior.center) == expected[name][0]
            assert float(param.prior.sigma) == expected[name][1]
        assert np.all(np.isfinite(np.asarray(mean(jnp.array([0., 1e-6, .1, .9, 1.])))))
        gradient = jax.grad(lambda m: jnp.sum(m(jnp.array([1e-6, .1, .9]))))(mean)
        assert all(np.isfinite(float(getattr(gradient, name).value))
                   for name in ("M", "a", "b"))


def test_shaped_campaign_identity_records_all_gaussian_priors(shaped):
    identity = prior_identity("JAM", "union")
    assert identity["form"] == "shaped_logrbf"
    assert identity["length_log"] == pytest.approx(np.log(10))
    assert identity["shaped_mean"]["x0"] == 0.1
    assert identity["shaped_mean"]["priors"] == {
        field: cfg.shaped_mean_priors(field) for field in cfg.ALL_FIELDS
    }
    priors = identity["shaped_mean"]["priors"]
    assert all(priors[f]["M"] == [1.0, 1.0] for f in cfg.ALL_FIELDS)
    assert all(priors[f]["a"] == [0.5, 0.4] for f in ("t3", "v3", "v8", "v", "v15"))
    assert all(priors[f]["a"] == [-0.2, 0.15] for f in ("t8", "t15", "sigma", "g"))
    assert all(priors[f]["b"] == [3.0, 2.0] for f in cfg.ALL_FIELDS if f != "g")
    assert priors["g"]["b"] == [5.0, 3.0]


def test_gp_sigma_is_one_positive_number():
    assert cfg.parse_gp_sigma("0.3") == 0.3
    assert cfg.parse_gp_sigma("1e-1") == pytest.approx(0.1)
    for bad in ("", "0", "-1", "x0.5", "g=3", "nan", "inf"):
        with pytest.raises(ValueError):
            cfg.parse_gp_sigma(bad)


def test_one_gp_width_reaches_every_prior_and_identity(shaped):
    for field in cfg.ALL_FIELDS:
        assert float(fit.gp_prior(field).cov.sigma.value) == pytest.approx(0.3)
    assert prior_identity("JAM", "union")["amplitudes"] == {
        field: pytest.approx(0.3) for field in cfg.ALL_FIELDS}


def test_shaped_prior_requires_a_gp_width(shaped, monkeypatch):
    monkeypatch.setattr(cfg, "SHAPED_GP_SIGMA", None)
    with pytest.raises(ValueError, match="PIXEL_SHAPED_GP_SIGMA"):
        fit.gp_prior("g")


def test_shaped_means_remain_free_in_analysis(shaped, monkeypatch):
    class TinyAnalysis:
        def __init__(self, _name):
            self.priors = {}
            self.ties = []
            self.frozen = []

        def field(self, name, *, prior, **_kwargs):
            self.priors[name] = prior
            return name

        def add(self, _dataset):
            raise AssertionError("focused test must not assemble data")

        def compile(self, **_kwargs):
            return self

        def prior_params(self, field):
            return self.priors[field]

        def tie(self, target, *, to):
            self.ties.append((target, to))

        def freeze(self, *params):
            self.frozen.extend(params)

    monkeypatch.setattr(fit.pixel, "Analysis", TinyAnalysis)
    monkeypatch.setattr(fit.dsets, "generic_section_family", lambda _q: None)
    monkeypatch.setattr(fit.dsets, "build_datasets", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(fit, "load_truth", lambda _q: {})
    monkeypatch.setattr(fit, "constraint_datasets", lambda *_args: [])

    analysis, _fields = fit.build_analysis("2", "exp")
    assert analysis.ties == []
    for field in cfg.ALL_FIELDS:
        mean = analysis.priors[field].mean
        assert all(not any(getattr(mean, name) is frozen for frozen in analysis.frozen)
                   for name in ("M", "a", "b"))
