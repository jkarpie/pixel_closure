"""Focused checks for the optional independent-mean JAM PDF priors."""

from __future__ import annotations

import pytest

from campaign.run_full_scale_stream_sweep import prior_identity
from closure_JAM_truth import config as cfg, fit


@pytest.fixture
def zero_unit(monkeypatch):
    amplitudes = {**cfg.GP_AMPLITUDES,
                  **{name: 1.0 for name in cfg.VANISHING_PDF_FIELDS},
                  **{name: width for name, (_mean, width)
                     in cfg.ZERO_UNIT_NONVANISHING_PRIORS.items()}}
    monkeypatch.setattr(cfg, "VANISHING_PDF_PRIOR", "zero_unit")
    monkeypatch.setattr(cfg, "GP_AMPLITUDES", amplitudes)
    monkeypatch.setattr(cfg, "GP_AMPLITUDE_FLOOR", .99 * min(amplitudes.values()))


def test_zero_unit_prior_values_and_campaign_identity(zero_unit):
    assert set(cfg.VANISHING_PDF_FIELDS) == {"t3", "v3", "v8", "v", "v15"}
    assert cfg.ZERO_UNIT_NONVANISHING_PRIORS == {
        "t8": (1.5, 2.0), "t15": (2.0, 3.0),
        "sigma": (4.0, 5.0), "g": (5.0, 10.0),
    }
    for name in cfg.ALL_FIELDS:
        prior = fit._gp_prior(name)
        expected_mean = (0.0 if name in cfg.VANISHING_PDF_FIELDS
                         else cfg.ZERO_UNIT_NONVANISHING_PRIORS[name][0])
        assert float(prior.mean.N.value) == expected_mean
        assert float(prior.cov.sigma.value) == cfg.GP_AMPLITUDES[name]
    identity = prior_identity("JAM", "union")
    assert identity["means"] == {name: cfg.gp_mean(name) for name in cfg.ALL_FIELDS}
    assert identity["amplitudes"] == {name: cfg.gp_sigma(name) for name in cfg.ALL_FIELDS}


def test_all_variant_means_are_frozen_and_untied(zero_unit, monkeypatch):
    class TinyAnalysis:
        def __init__(self, _name):
            self.priors = {}
            self.ties = []
            self.frozen = []

        def field(self, name, *, prior, **_kwargs):
            self.priors[name] = prior
            return name

        def add(self, _dataset):
            raise AssertionError("the focused prior test must not assemble data")

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
    for name in cfg.ALL_FIELDS:
        prior = analysis.priors[name]
        tied = any(target is prior.mean.N and source is prior.cov.sigma
                   for target, source in analysis.ties)
        frozen = any(param is prior.mean.N for param in analysis.frozen)
        assert (frozen, tied) == (True, False)
