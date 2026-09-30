"""The fixed-prior scan harness applies the prior it records, and ranks what it ran.

Covers ``campaign/prior_cells.py`` (grids, validation, hashing, :func:`patched_prior`) and
the parts of ``campaign/prior_scan.py`` that decide what a row means (gate parsing, the
hash guard on ``--skip-existing``, the fit ranking).  No fit is run: the one property that
needs a real suite -- that a patched value actually reaches the prior object ``fit.py``
builds -- is read off ``fit.gp_prior`` directly.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp

from campaign import prior_cells as pc
from campaign import prior_scan as ps
from campaign import saddle_strategy as ss

SUITE = "closure_JAM_truth_small"


@pytest.fixture(scope="module")
def cfg():
    return importlib.import_module(f"{SUITE}.config")


@pytest.fixture(scope="module")
def fitmod():
    return importlib.import_module(f"{SUITE}.fit")


def _cell(cfg, prior="const_logrbf"):
    return pc.base_cell(cfg, prior, "t")


def test_patched_prior_reaches_the_built_prior_and_restores_the_original_objects(cfg, fitmod):
    """Inside the context ``fit.gp_prior`` carries the cell's values; after it, the originals."""
    cell = _cell(cfg)
    cell["GP_AMPLITUDES"]["g"] = 1.75
    cell["GP_AMPLITUDES"]["v8"] = 0.05
    cell["GP_LENGTH_LOG"] = math.log(3.0)
    cell["GP_JITTER"] = 1e-7
    originals = {name: getattr(cfg, name) for name in pc.PATCHED}
    with pc.patched_prior(cfg, cell, results_root=Path("/tmp/prior_scan_test")):
        prior = fitmod.gp_prior("g")
        assert float(prior.cov.sigma.value) == 1.75
        assert float(prior.mean.N.value) == 1.75
        assert float(prior.cov.length.value) == math.log(3.0)
        assert prior.cov.jitter == 1e-7
        assert cfg.GP_AMPLITUDE_FLOOR == pytest.approx(0.99 * 0.05)
        assert cfg.results_dir("2") == Path("/tmp/prior_scan_test") / "truthQ_2"
    for name, value in originals.items():
        assert getattr(cfg, name) is value


def test_patched_prior_restores_even_when_the_body_raises(cfg):
    originals = {name: getattr(cfg, name) for name in pc.PATCHED}
    cell = _cell(cfg, "beta_envelope")
    with pytest.raises(RuntimeError):
        with pc.patched_prior(cfg, cell):
            raise RuntimeError("boom")
    for name, value in originals.items():
        assert getattr(cfg, name) is value


def test_envelope_cells_reach_the_envelope_prior(cfg, fitmod):
    cell = _cell(cfg, "beta_envelope")
    cell["GP_ENVELOPE"]["g"] = {"alpha": 0.3, "beta": 2.5, "sigma": 0.4}
    cell["GP_ENVELOPE_JITTER"] = 1e-3
    with pc.patched_prior(cfg, cell):
        prior = fitmod.gp_prior("g")
        assert type(prior.cov).__name__ == "BetaTaperedLogRBF"
        assert float(prior.cov.sigma.value) == 0.4
        assert float(prior.cov.alpha.value) == 0.3
        assert float(prior.cov.beta.value) == 2.5
        assert prior.cov.jitter == 1e-3


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.update(extra=1), "unknown keys"),
        (lambda c: c.update(prior_form="flat"), "prior_form"),
        (lambda c: c["GP_ENVELOPE"]["t3"].update(alpha=0.9), "alpha"),
        (lambda c: c["GP_ENVELOPE"]["t3"].update(beta=0.1), "beta"),
        (lambda c: c["GP_AMPLITUDES"].update(g=0.0), "amplitude"),
        (lambda c: c["GP_AMPLITUDES"].pop("g"), "exactly"),
        (lambda c: c.update(GP_JITTER=0.0), "GP_JITTER"),
    ],
)
def test_validate_refuses_incomplete_or_out_of_range_cells(cfg, mutate, match):
    cell = copy.deepcopy(_cell(cfg))
    mutate(cell)
    with pytest.raises(ValueError, match=match):
        pc.validate(cell)


def test_cell_hash_ignores_name_and_provenance_but_not_any_value(cfg):
    a = _cell(cfg)
    b = copy.deepcopy(a)
    b["name"], b["provenance"] = "other", {"x": 1}
    assert pc.cell_hash(a) == pc.cell_hash(b)
    b["GP_AMPLITUDES"]["t3"] = a["GP_AMPLITUDES"]["t3"] * (1 + 1e-12)
    assert pc.cell_hash(a) != pc.cell_hash(b)


def test_stage_one_grids_are_valid_distinct_and_include_the_shipped_prior(cfg):
    const = pc.const_grid_s1(cfg)
    envelope = pc.envelope_grid_s1(cfg)
    cells = const + envelope
    assert len({c["name"] for c in cells}) == len(cells)
    assert len({pc.cell_hash(c) for c in cells}) == len(cells)
    shipped_const = pc.base_cell(cfg, "const_logrbf", "c_config")
    shipped_env = pc.base_cell(cfg, "beta_envelope", "e_config")
    hashes = {pc.cell_hash(c) for c in cells}
    assert pc.cell_hash(shipped_const) in hashes and pc.cell_hash(shipped_env) in hashes
    assert all(c["prior_form"] == "const_logrbf" for c in const)
    assert all(c["prior_form"] == "beta_envelope" for c in envelope)


def test_gate_values_are_read_back_from_both_failure_messages():
    plain = "small-closure VEGAS ESS fraction 0.0243 is below 0.1; refusing an unreliable fit"
    nested = ("small-closure nested VEGAS diagnostics failed: absolute outer ESS fraction 0.00138 < 0.1; "
              "signed outer ESS fraction 0.00129 < 0.05; median inner ESS fraction 0.001 < 0.1")
    assert ps.ess_from_message(plain) == {"vegas_ess_fraction": 0.0243}
    assert ps.ess_from_message(nested) == {"absolute_ess_fraction": 0.00138,
                                           "signed_ess_fraction": 0.00129,
                                           "median_inner_ess_frac": 0.001}
    assert ps.ess_from_message("H-S saddle search did not converge") == {}


def test_skip_existing_refuses_a_row_written_for_a_different_prior(cfg, tmp_path):
    cell = _cell(cfg)
    other = copy.deepcopy(cell)
    other["GP_AMPLITUDES"]["g"] = 2.0
    cells_file = tmp_path / "cells.json"
    cells_file.write_text(json.dumps({"cells": [cell]}))
    path = ps._row_path(tmp_path, SUITE, "2", "dis", cell, sampler="gaussian")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"cell_hash": pc.cell_hash(other), "status": "ok"}))
    args = SimpleNamespace(cells=cells_file, out=tmp_path, modes=["dis"], suite=SUITE, q="2",
                           skip_existing=True, retry=False, no_plots=True, timeout=1,
                           sampler="gaussian", auxiliary=None, run_map="auto",
                           sampler_options=None, saddle_strategy="pixel_default")
    with pytest.raises(SystemExit, match="different cell_hash"):
        ps.cmd_fit(args)


def _vegas_kwargs():
    return {
        "n_samples": 1000,
        "seed": 17,
        "n_bins": 12,
        "n_adapt_iterations": 3,
        "n_eval_per_iteration": 256,
        "alpha": 0.5,
        "joint_grid": True,
        "min_adapt_phase_ess_frac": 0.1,
        "max_extra_adapt_iterations": 2,
        "n_adapt_inner_samples": 64,
        "n_inner_samples": 128,
        "covariance_inflation": 3.0,
        "saddle_tol": 1e-3,
    }


@pytest.mark.parametrize(
    "sampler, implementation, auxiliary, run_map, options",
    [
        ("gaussian", "ordinary_gaussian", "auto", True, {}),
        ("gaussian", "normalization_reference_saddle_contour", "auto", False,
         {"saddle_tol": 5e-3, "reference_cache_path": Path("ref.npz")}),
        ("nuts", "ordinary_nuts", "auto", False,
         {"n_warmup": 100, "step_size": 0.1, "use_hessian_mass": True}),
        ("nuts", "normalization_reference_contour_nuts", "sample", False,
         {"n_warmup": 100, "step_size": 0.1, "saddle_tol": 5e-3,
          "reference_cache_path": Path("ref.npz"),
          "hessian_cache_path": Path("hessian.npz")}),
    ],
)
def test_sampler_rewrite_drops_vegas_kwargs_and_keeps_only_the_requested_route(
        sampler, implementation, auxiliary, run_map, options):
    rewritten = ps.rewrite_infer_kwargs(
        _vegas_kwargs(), sampler=sampler, auxiliary=auxiliary,
        implementation=implementation, run_map=run_map, options=options,
        extra={"n_samples": 321},
    )
    assert all(name not in rewritten for name in ps.VEGAS_ONLY_OPTIONS)
    if "saddle_tol" not in options:
        assert "saddle_tol" not in rewritten
    assert rewritten["sampler"] == sampler
    assert rewritten["auxiliary"] == auxiliary
    assert rewritten["run_map"] is run_map
    assert rewritten["seed"] == 17
    assert rewritten["n_samples"] == 321  # explicit overrides are layered last
    for name, value in options.items():
        assert rewritten[name] == value


def test_sampler_rewrite_refuses_a_sampler_route_mismatch():
    with pytest.raises(ValueError, match="resolved to"):
        ps.rewrite_infer_kwargs(
            _vegas_kwargs(), sampler="gaussian", auxiliary="auto",
            implementation="ordinary_nuts", run_map=False, options={},
        )


def test_sampler_defaults_distinguish_linear_and_bilinear_routes():
    assert ps.default_auxiliary("nuts", has_bilinear=False) == "auto"
    assert ps.default_auxiliary("nuts", has_bilinear=True) == "sample"
    assert ps.default_auxiliary("gaussian", has_bilinear=True) == "auto"
    assert ps.default_run_map("ordinary_gaussian") is True
    assert ps.default_run_map("normalization_reference_saddle_contour") is False
    assert ps.default_run_map("ordinary_gaussian", "false") is False


def _saddle(*, converged=True):
    return SimpleNamespace(converged=converged, n_iter=4, grad=np.array([2e-6, -1e-6]))


def test_gaussian_and_saddle_contour_gates_record_diagnostics_and_fail_low_ess():
    gaussian = SimpleNamespace(
        weights=np.full(20, 1 / 20), effective_sample_size=20.0,
        n_samples=20, n_invalid=0, metadata={}, summary=lambda: {},
    )
    diag, failures = ps.sampler_gate(gaussian, "ordinary_gaussian")
    assert failures == []
    assert diag == {
        "implementation": "ordinary_gaussian", "ess_fraction": 1.0,
        "max_weight": 0.05, "n_invalid": 0,
    }
    gaussian.effective_sample_size = 1.0
    diag, failures = ps.sampler_gate(gaussian, "ordinary_gaussian")
    assert diag["ess_fraction"] == 0.05
    assert failures == ["Gaussian importance ESS fraction 0.05 < 0.1"]

    contour = SimpleNamespace(
        ess_frac=0.2, saddle=_saddle(),
        metadata={"proposal_fit_success": True, "proposal_scale": 1.5},
        summary=lambda: {"weight_magnitude_ess_fraction": 0.4,
                         "phase_only_ess_fraction": 0.5},
    )
    diag, failures = ps.sampler_gate(contour, "normalization_reference_saddle_contour")
    assert failures == []
    assert diag["saddle_converged"] is True
    assert diag["saddle_residual"] == pytest.approx(2e-6)
    assert diag["proposal_fit_success"] is True
    contour.ess_frac = 0.01
    contour.saddle = _saddle(converged=False)
    _, failures = ps.sampler_gate(contour, "fixed_saddle_contour")
    assert failures == [
        "saddle-contour residual ESS fraction 0.01 < 0.1",
        "H-S saddle did not converge",
    ]


def _nuts(*, summary=None, metadata=None, coordinates=None, ess_frac=None,
          saddle=None):
    values = {
        "n_divergent": 0,
        "tau_reliable": True,
        "ess_min": 120.0,
        "mean_accept_prob": 0.82,
        "n_max_depth": 0,
        "step_size": 0.08,
    }
    values.update(summary or {})
    return SimpleNamespace(
        metadata=metadata or {"mass_initialization": "hessian"},
        coordinates=np.array([[0.0], [1.0]]) if coordinates is None else coordinates,
        ess_frac=0.2 if ess_frac is None else ess_frac,
        saddle=_saddle() if saddle is None else saddle,
        summary=lambda: values,
    )


def test_nuts_gates_cover_chain_quality_and_contour_non_degeneracy():
    ordinary = _nuts()
    diag, failures = ps.sampler_gate(ordinary, "ordinary_nuts")
    assert failures == []
    assert diag["mass_initialization"] == "hessian"
    assert diag["ess_min"] == 120.0

    broken = _nuts(
        summary={"n_divergent": 2, "tau_reliable": False},
        metadata={"mass_initialization": "identity_fallback"},
    )
    _, failures = ps.sampler_gate(broken, "ordinary_nuts")
    assert failures == [
        "NUTS divergences 2",
        "NUTS mass initialization 'identity_fallback'",
        "NUTS tau_int unreliable (chain shorter than 50 tau)",
    ]

    contour = _nuts(metadata={"backend_metric_initialization": "hessian_cache"})
    diag, failures = ps.sampler_gate(contour, "normalization_reference_contour_nuts")
    assert failures == []
    assert diag["residual_ess_fraction"] == 0.2
    assert diag["moved_fraction"] == 1.0

    frozen = _nuts(
        coordinates=np.zeros((3, 1)), ess_frac=0.01,
        saddle=_saddle(converged=False),
        metadata={"backend_metric_initialization": "hessian"},
    )
    _, failures = ps.sampler_gate(frozen, "fixed_contour_nuts")
    assert failures == [
        "contour residual ESS fraction 0.01 < 0.1",
        "H-S saddle did not converge",
        "contour chain never moved",
    ]


def test_sampler_gate_exception_is_classified_as_a_gate_failure():
    assert ps.classify_exception(ps.SamplerGateError("bad ESS")) == "gate_fail"
    assert ps.classify_exception(RuntimeError("H-S saddle did not converge")) == "gate_fail"
    assert ps.classify_exception(TypeError("wrong keyword")) == "error"


def test_memory_budget_refusal_is_not_a_gate_failure():
    """PIXEL's refusal message contains "normalization", which the text match would misfile."""
    refusal = MemoryError("normalization-reference contour projected peak 106939725136 bytes "
                          "(99.60 GiB) exceeds the 17179869184 byte (16.00 GiB) budget")
    assert ps.classify_exception(refusal) == "memory_refused"


def test_row_paths_distinguish_samplers(cfg, tmp_path):
    cell = _cell(cfg)
    gaussian = ps._row_path(tmp_path, SUITE, "2", "dy", cell, sampler="gaussian")
    nuts = ps._row_path(tmp_path, SUITE, "2", "dy", cell, sampler="nuts")
    assert gaussian != nuts
    assert "__gaussian__" in gaussian.name
    assert "__nuts__" in nuts.name


def test_diagonal_newton_finds_a_badly_scaled_complex_stationary_point():
    root = jnp.array([1.0 + 0.5j, -2.0 + 3.0j])

    def action(z):
        delta = z - root
        return 0.5 * (1e12 * delta[0] ** 2 + 1e-4 * delta[1] ** 2)

    point, gradient, n_iter, converged = ss.diagonal_newton(
        action, np.zeros(2, dtype=complex), tol=1e-10,
        max_iter=10, max_backtracks=10,
    )
    assert converged is True
    assert n_iter <= 3
    np.testing.assert_allclose(point, np.asarray(root), rtol=0, atol=1e-11)
    np.testing.assert_allclose(gradient, 0.0, rtol=0, atol=1e-8)


def test_experimental_saddle_patch_is_scoped_and_covers_nested_vegas():
    import pixel.infer.contour as contour
    import pixel.infer.contour.normalization as contour_normalization
    import pixel.infer.samplers.nested_vegas as nested_vegas

    original_contour = contour.find_saddle
    original_normalization = contour_normalization.find_saddle
    original_nested = nested_vegas.find_saddle
    with ss.patched_saddle_search("shifted_diagonal"):
        assert contour.find_saddle is ss.shifted_diagonal_find_saddle
        assert contour_normalization.find_saddle is ss.shifted_diagonal_find_saddle
        assert nested_vegas.find_saddle is ss.shifted_diagonal_find_saddle
    assert contour.find_saddle is original_contour
    assert contour_normalization.find_saddle is original_normalization
    assert nested_vegas.find_saddle is original_nested


def _row(cell, suite, mode, status="ok", rms=1.0, signed=0.0, constraint=1e-5):
    pulls = {f: {"rms": rms, "signed": signed, "frac_le_2": 1.0} for f in pc.FIELDS}
    return {"cell": cell, "cell_hash": cell, "prior_form": "const_logrbf", "suite": suite, "q": "2",
            "mode": mode, "status": status, "pulls": pulls if status == "ok" else None,
            "worst_abs_constraint_residual": constraint if status == "ok" else None, "gate": {}}


def test_ranking_orders_gates_then_constraints_then_pulls(tmp_path):
    rows = tmp_path / "rows"
    rows.mkdir()
    specs = {
        "a_all_pass_low_pulls": {},
        "b_all_pass_high_pulls": {"rms": 2.0},
        "c_constraint_miss": {"constraint": 1e-3, "rms": 0.1},
        "d_one_gate_fail": {"fail": ("closure_NNPDF_truth_small", "dy"), "rms": 0.01},
    }
    for name, spec in specs.items():
        for suite in ps.SUITES:
            for mode in ps.PHYSICS_MODES:
                status = "gate_fail" if spec.get("fail") == (suite, mode) else "ok"
                row = _row(name, suite, mode, status=status, rms=spec.get("rms", 1.0),
                           constraint=spec.get("constraint", 1e-5))
                (rows / f"{suite}__{mode}__{name}.json").write_text(json.dumps(row))
    order = [r["cell"] for r in ps.rank_fits(rows)]
    assert order == ["a_all_pass_low_pulls", "b_all_pass_high_pulls", "c_constraint_miss", "d_one_gate_fail"]


def test_pilot_scale_patch_replaces_only_pixels_two_width_grids_and_restores_numpy():
    """Inside the patch PIXEL's contour module sees the chosen grid for its two pilot calls only."""
    import pixel.infer.contour.normalization as contour
    from campaign import proposal_width as pw

    assert contour.np is np
    with pw.patched_pilot_scales((1.0, 2.0, 4.0)) as proxy:
        assert contour.np.geomspace(0.35, 1.0, 4).tolist() == [1.0, 2.0, 4.0]
        assert contour.np.geomspace(0.25, 1.0, 7).tolist() == [1.0, 2.0, 4.0]
        np.testing.assert_array_equal(contour.np.geomspace(0.25, 1.0, 5), np.geomspace(0.25, 1.0, 5))
        assert contour.np.asarray is np.asarray
        assert proxy.intercepted == 2
    assert contour.np is np
    with pw.patched_pilot_scales(None) as proxy:
        assert proxy is None and contour.np is np


def test_pilot_scale_patch_restores_numpy_when_the_body_raises():
    import pixel.infer.contour.normalization as contour
    from campaign import proposal_width as pw

    with pytest.raises(RuntimeError):
        with pw.patched_pilot_scales((2.0,)):
            raise RuntimeError("boom")
    assert contour.np is np


@pytest.mark.parametrize("text", ["0,1", "-1", "nan", "1,,2"])
def test_proposal_scales_refuse_non_positive_or_malformed_values(text):
    from campaign import proposal_width as pw

    with pytest.raises(ValueError):
        pw.parse_scales(text)


def test_gaussian_runs_use_the_exact_laplace_width_unless_told_otherwise(cfg, tmp_path):
    """No inflation on the ordinary route, no narrowing on the saddle-contour route, by default."""
    options = ps.implementation_options(cfg, "ordinary_gaussian", cache_stem=tmp_path / "s")
    assert options == {"hessian_scale": 2.0, "covariance_inflation": 1.0}
    kw = ps.rewrite_infer_kwargs(_vegas_kwargs(), sampler="gaussian", auxiliary="auto",
                                 implementation="ordinary_gaussian", run_map=True, options=options)
    assert kw["covariance_inflation"] == 1.0  # the suite's VEGAS value must not leak through
    assert ps.proposal_scales_for("gaussian", None) == (1.0,)
    assert ps.proposal_scales_for("gaussian", "2,4") == (2.0, 4.0)
    assert ps.proposal_scales_for("vegas", None) is None
    assert ps.proposal_scales_for("nuts", None) is None


def test_saddle_contour_gate_records_whether_pixel_floored_proposal_eigenvalues():
    contour = SimpleNamespace(
        ess_frac=0.2, saddle=_saddle(),
        metadata={"proposal_fit_success": True, "proposal_scale": 1.0,
                  "proposal_precision_eigenvalues_floored": 3},
        summary=lambda: {"weight_magnitude_ess_fraction": 0.4, "phase_only_ess_fraction": 0.5},
    )
    diag, _ = ps.sampler_gate(contour, "normalization_reference_saddle_contour")
    assert diag["proposal_precision_eigenvalues_floored"] == 3
