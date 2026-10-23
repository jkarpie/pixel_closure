"""Check report arithmetic against PIXEL and the earlier Q-dense procedure."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from campaign import report_full_scale_closure as report
from campaign.run_full_scale_stream_sweep import _json_safe


def test_streamed_rhat_matches_pixel_and_explicit_split_formula():
    from pixel.infer.diagnostics.multichain import split_r_hat

    chains = [np.random.default_rng(i).normal(loc=i*.1, size=(81, 7)) for i in range(4)]
    measured = report.split_rhat([report.split_statistics(c) for c in chains], tuple(map(str, range(7))))
    expected = split_r_hat(chains)
    # Independent array calculation used by the earlier Q-dense report.
    half = len(chains[0]) // 2
    split = np.concatenate([np.stack(chains)[:, :half], np.stack(chains)[:, half:2*half]])
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half*split.mean(axis=1).var(axis=0, ddof=1)
    previous = np.sqrt((((half-1)/half)*within + between/half)/within)
    index = int(np.argmax(previous))
    assert measured["max"] == pytest.approx(expected.max(), rel=1e-14)
    assert measured["max"] == pytest.approx(previous[index], rel=1e-14)
    assert measured["parameter"] == str(index)


def test_frozen_disagreeing_chains_are_not_reported_as_converged():
    chains = [np.full((20, 1), value) for value in (0., 1.)]
    result = report.split_rhat([report.split_statistics(c) for c in chains], ("frozen",))
    assert result["max"] is None
    assert "frozen" in result["issue"]


def test_pull_statistics_matches_suite_coverage_and_old_finite_node_policy():
    from closure_JAM_truth.fit import coverage_report

    nodes = np.array([.001, .01, .03, .2, .5, .9, .99])
    truth = np.array([0., .2, .3, .2, .4, .5, 0.])
    mean = np.array([1e5, .25, .25, .6, .1, .9, 1e5])
    std = np.array([1., .12, .13, .23, .21, .31, 1.])
    layout = [{"name": "g", "start": 0, "stop": 7}]
    measured = report.pull_statistics(mean, std**2, truth, nodes, layout)
    coverage = coverage_report({"g": (nodes, mean, std)}, {"x_nodes": nodes, "curves": {"g": truth}})["g"]
    assert measured["mean_abs_pull"] == pytest.approx(coverage["mean_abs_pull"])
    assert measured["frac_pull_lt_1"] == coverage["frac_pull_le_1"]
    assert measured["frac_pull_lt_2"] == coverage["frac_pull_le_2"]
    signed = (mean-truth)/std
    valid = np.isfinite(signed) & (nodes >= .01) & (nodes <= .9)
    assert measured["max_abs_pull"] == np.max(np.abs(signed[valid]))
    assert measured["worst_true_pdf_pull"] == {"pdf": "g", "x": .2, "pull": signed[3]}


def _make_campaign(tmp_path):
    from pixel.diagnostics import save_posterior_moments
    from pixel.infer.moments import PosteriorMoments
    from pixel.infer.samplers.nuts import NUTSSamples

    root = tmp_path / "relocated"
    cell = dict(parameterization="q", truth="JAM", data="union", scale="2", mode="exp", stream_count=2, streams=[])
    datasets = [
        SimpleNamespace(name="dis", n_data=2, mean=np.array([1., 2.]), cov=np.array([[1., .2], [.2, .5]])),
        SimpleNamespace(name="cons_sumrule", n_data=1, mean=np.array([0.]), cov=np.array([[.1]])),
    ]
    model = SimpleNamespace(datasets=datasets)
    layout = [{"name": d.name, "dataset_index": i, "start": start, "stop": start+d.n_data, "prediction_kind": "linear"}
              for i, (start, d) in enumerate(zip((0, 2), datasets))]
    q_layout = ({"name": "g", "start": 0, "stop": 2},)
    raw_q, raw_pred, chains = [], [], []
    for number in (1, 2):
        output = report.stream_path(root, cell, number)
        output.mkdir(parents=True)
        chain = NUTSSamples(
            samples=np.random.default_rng(number).normal(size=(40, 2)),
            evidence_values=np.zeros(40), weights=np.full(40, 1/40),
            tree_depth=np.full(40, 3), n_leapfrog=np.full(40, 7),
            accept_prob=np.full(40, .91), energy_error=np.array([-4.]+[.1]*39),
            divergent=np.array([True]+[False]*39), reached_max_depth=np.zeros(40, dtype=bool),
            initial=np.zeros(2), step_size=.12, n_warmup=10,
            inv_mass=np.eye(2), param_labels=("a", "b"),
            metadata={"metric_structure": "dense", "mass_initialization": "hessian"},
        )
        chains.append(chain)
        chain.save_npz(output / "chain.npz")
        (output / "inference_summary.json").write_text(json.dumps(_json_safe(chain.summary())))
        record = {**{k: cell[k] for k in report.IDENTITY}, "seed": number,
                  "status": "complete", "samples": 40, "chain_kind": "nuts", "warmup_requested": 10,
                  "wall_seconds": 15., "sampler_wall_seconds": 10.}
        (output / "stream.json").write_text(json.dumps(record))
        # Different stream means make the between-stream variance essential.
        q = chain.samples + number*2
        pred = np.column_stack([q[:, 0], q[:, 1], np.zeros(40)])
        raw_q.append(q)
        raw_pred.append(pred)
        result_path = output / "results" / "truthQ_2"
        result_path.mkdir(parents=True)
        chi2 = report.prediction_chi2(pred.mean(0), layout, model, 1e-10)["chi2"]
        fit = {"q_key": "2", "mode": "exp", "sampler": "q_nuts", "n_data": 3,
               "n_physical_data": 2, "n_constraint_data": 1, "chi2": chi2,
               "chi2_per_data": chi2/3, "coverage": {}}
        (result_path / "summary_exp.json").write_text(json.dumps(fit))
        moments = PosteriorMoments(q.mean(0), np.cov(q, rowvar=False, ddof=0), q_layout,
                                   pred.mean(0), np.cov(pred, rowvar=False, ddof=0), tuple(layout),
                                   {"matched_nonlinear_draws": True})
        save_posterior_moments(result_path / "posterior_moments_exp.npz", moments,
                               truth=np.array([1., 2.]), nodes=np.array([.1, .3]),
                               metadata={"q_key": "2", "mode": "exp"})
        # Deliberately use nonexistent remote paths in the copied manifest.
        cell["streams"].append({"number": number, "seed": number, "output": f"/remote/elsewhere/stream_{number:02d}"})
    (root / "manifest.json").write_text(json.dumps({"cells": [cell], "status": "complete", "samples_per_stream": 40, "warmup_per_stream_requested": 10}))
    return root, cell, model, np.concatenate(raw_q), np.concatenate(raw_pred), chains


def test_complete_cli_pools_moments_matches_direct_draws_and_pixel_sampler(tmp_path, monkeypatch, capsys):
    root, cell, model, q, prediction, chains = _make_campaign(tmp_path)
    monkeypatch.setattr(report, "build_source_model", lambda _: (model, 1e-10))
    out = tmp_path / "transfer" / "report"
    assert report.main(["--root", str(root), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Pooled posterior: 80 retained draws" in text
    assert "NUTS acceptance" in text
    saved = json.loads(out.with_suffix(".json").read_text())["cells"][0]
    physics = saved["pooled_physics"]
    expected = report.pull_statistics(q.mean(0), q.var(0), np.array([1., 2.]), np.array([.1, .3]), [{"name": "g", "start": 0, "stop": 2}])
    assert physics["mean_abs_pull"] == pytest.approx(expected["mean_abs_pull"], rel=1e-14)
    assert physics["max_abs_pull"] == pytest.approx(expected["max_abs_pull"], rel=1e-14)
    residual = prediction.mean(0)[:2]-model.datasets[0].mean
    expected_chi2 = float(residual @ np.linalg.pinv(model.datasets[0].cov, rcond=1e-10) @ residual)
    assert physics["chi2"] == pytest.approx(expected_chi2)
    stream_chi2s = [r["fit"]["chi2"] for r in saved["streams"]]
    assert not np.isclose(physics["chi2"], np.mean(stream_chi2s))
    assert physics["worst_dataset_chi2"]["dataset"] == "dis"
    assert physics["npt"] == 3
    assert physics["n_physical_data"] == 2
    # Check sampler values against PIXEL and the actual trajectory arrays.
    chain = chains[0]
    previous = chain.summary()
    current = saved["streams"][0]["sampler"]
    for key in ("tau_int_max", "ess_min", "ess_fraction_min"):
        assert current[key] == pytest.approx(previous[key])
    assert current["max_abs_delta_h"] == float(np.max(np.abs(chain.energy_error)))
    assert current["average_depth"] == float(np.mean(chain.tree_depth))
    assert current["num_divergence"] == int(np.sum(chain.divergent))
    assert current["steps_per_trajectory"] == float(np.mean(chain.n_leapfrog))
    assert current["acceptance_statistic"] == pytest.approx(float(np.mean(chain.accept_prob)))


def test_changed_source_covariance_refuses_pooled_physics(tmp_path, monkeypatch):
    root, cell, model, *_ = _make_campaign(tmp_path)
    model.datasets[0].cov *= 2
    monkeypatch.setattr(report, "build_source_model", lambda _: (model, 1e-10))
    result = report.report_cell(root, cell)
    assert result["pooled_physics"] is None
    assert any("do not reproduce saved chi2" in s for s in result["unavailable"])


def test_partial_transfer_reports_missing_inputs_and_preserves_launch_records(tmp_path, capsys):
    root, cell, *_ = _make_campaign(tmp_path)
    for entry in cell["streams"]:
        path = report.stream_path(root, cell, entry["number"])
        (path / "stream.json").write_text(json.dumps({"status": "running", "seed": entry["seed"], "samples": 25000}))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert report.main(["--root", str(root)]) == 2
    printed = capsys.readouterr().out
    assert "running launch record" in printed
    assert "Pooled posterior" not in printed
    assert all(p.read_bytes() == content for p, content in before.items())


def test_json_transfer_reuses_only_matching_cached_report(tmp_path):
    from campaign.report_full_scale_stream_sweep import render_cell

    root, cell, *_ = _make_campaign(tmp_path)
    local = {**cell, "streams": [{"output": str(report.stream_path(root, cell, s["number"]))} for s in cell["streams"]]}
    _, previous = render_cell(local)
    for s in cell["streams"]:
        (report.stream_path(root, cell, s["number"]) / "chain.npz").unlink()
    saved = report.report_cell(root, cell, summary_only=True)
    assert saved["rhat"]["max"] == previous["split_rhat_max"]
    assert saved["streams"][0]["sampler"]["average_depth"] == 3
    assert saved["streams"][0]["sampler"]["n_max_depth"] == 0
    # Same draw count but a different diagnostic summary must invalidate cache.
    path = report.stream_path(root, cell, 1) / "inference_summary.json"
    summary = json.loads(path.read_text())
    summary["tau_int_max"] = 999.
    path.write_text(json.dumps(summary))
    changed = report.report_cell(root, cell, summary_only=True)
    assert changed["rhat"] is None
    assert changed["streams"][0]["sampler"]["average_depth"] is None


def test_retained_count_mismatch_is_an_error(tmp_path):
    root, cell, *_ = _make_campaign(tmp_path)
    path = report.stream_path(root, cell, 1) / "inference_summary.json"
    summary = json.loads(path.read_text())
    summary["n_samples"] = 41
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="retained-draw mismatch"):
        report.report_cell(root, cell)
