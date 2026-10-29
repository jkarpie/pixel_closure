"""Focused checks for the JAM-like prior mean of the t0 covariance fits."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from campaign.run_full_scale_stream_sweep import prior_identity
from closure_JAM_truth import config as cfg, fit


@pytest.fixture
def t0_jam(monkeypatch):
    monkeypatch.setattr(cfg, "PRIOR_FORM", "shaped_logrbf")
    monkeypatch.setattr(cfg, "VANISHING_PDF_PRIOR", "zero_unit")
    monkeypatch.setattr(cfg, "SHAPED_GP_SIGMA", 0.3)
    monkeypatch.setattr(cfg, "GP_AMPLITUDES",
                        {**cfg.GP_AMPLITUDES, **{n: 0.3 for n in cfg.ALL_FIELDS}})
    monkeypatch.setattr(cfg, "PDF_GP_LENGTH_LOG", float(np.log(10)))
    monkeypatch.setattr(cfg, "T0_PRIOR", "jam_shape")


EXPECTED = {**{f: (0.5, 3.0) for f in ("t3", "v3", "v8", "v", "v15")},
            **{f: (-0.2, 3.0) for f in ("t8", "t15", "sigma")},
            "g": (-0.2, 5.0)}


def test_t0_mean_frees_only_the_normalization(t0_jam):
    for field in cfg.ALL_FIELDS:
        prior = fit.gp_prior(field, t0=True)
        mean = prior.mean
        a, b = EXPECTED[field]
        assert mean.M.frozen is False
        assert float(mean.M.prior.center) == 1.0 and float(mean.M.prior.sigma) == 1.0
        assert mean.a.frozen is True and float(mean.a.value) == a
        assert mean.b.frozen is True and float(mean.b.value) == b
        assert mean.a.prior is None and mean.b.prior is None
        assert float(prior.cov.sigma.value) == pytest.approx(0.3)
        x = jnp.array([1e-4, 0.5])
        expected = (np.asarray(x) / 0.1) ** a * ((1 - np.asarray(x)) / 0.9) ** b
        np.testing.assert_allclose(np.asarray(mean(x)), expected, rtol=1e-12)


def test_production_prior_keeps_free_exponents(t0_jam):
    mean = fit.gp_prior("g").mean
    assert mean.a.frozen is False and mean.b.frozen is False


def test_same_t0_prior_is_the_production_prior(t0_jam, monkeypatch):
    monkeypatch.setattr(cfg, "T0_PRIOR", "same")
    assert fit.gp_prior("g", t0=True).mean.a.frozen is False


def test_every_t0_fit_uses_the_t0_prior(t0_jam, monkeypatch):
    calls = []

    class Built:
        def compile(self):
            return self

    def build(q_key, mode, *, use_kernel_cache=True, t0=None, t0_prior=False):
        calls.append((mode, t0 is not None, t0_prior))
        return Built(), {}

    predictions = iter([{"d": np.array([1.0])}, {"d": np.array([1.0])}])
    monkeypatch.setattr(fit, "build_analysis", build)
    monkeypatch.setattr(fit, "_dis_predictions", lambda _model: next(predictions))
    t0, iterations = fit._iterate_t0("2", "exp", object())
    assert iterations == 1
    assert calls == [("dis", False, True), ("dis", True, True)]


def test_t0_prior_reaches_campaign_identity(t0_jam):
    identity = prior_identity("JAM", "union")["t0_prior"]
    assert identity["form"] == "jam_shape"
    assert identity["M"] == [1.0, 1.0]
    assert identity["exponents"] == {f: list(EXPECTED[f]) for f in cfg.ALL_FIELDS}


def test_default_t0_prior_leaves_identity_unchanged(t0_jam, monkeypatch):
    monkeypatch.setattr(cfg, "T0_PRIOR", "same")
    assert "t0_prior" not in prior_identity("JAM", "union")
