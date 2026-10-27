"""Print a full-scale closure report without launching inference.

Run where the original artifacts live for a complete report::

    python -m campaign.report_full_scale_closure --root campaign/JAM_Q2_exp_prod --out campaign/JAM_Q2_exp_prod/CLOSURE_REPORT

JSON-only transfers are supported; missing diagnostics are explicitly reported.
Exit status 2 means the requested full report has unavailable inputs.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from campaign.report_full_scale_stream_sweep import load_stream, stream_diagnostics


IDENTITY = ("parameterization", "truth", "data", "scale", "mode")


def read_json(path):
    return json.loads(Path(path).read_text())


def stream_path(root, cell, number):
    """Use the supplied root, never absolute paths from another machine."""
    return (root / f"streams_{cell['stream_count']}" / cell["parameterization"]
            / cell["truth"].lower() / cell["data"].lower()
            / f"truthQ_{cell['scale']}" / cell["mode"] / f"stream_{number:02d}")


def split_statistics(samples):
    """Sufficient statistics for PIXEL split R-hat, one chain at a time."""
    half = len(samples) // 2
    if half < 2:
        raise ValueError("split R-hat needs at least four draws per stream")
    return (half, np.stack([samples[:half].mean(0), samples[half:2*half].mean(0)]),
            np.stack([samples[:half].var(0, ddof=1), samples[half:2*half].var(0, ddof=1)]))


def split_rhat(statistics, labels):
    """Same arithmetic as PIXEL split_r_hat, without stacking full chains."""
    halves = {s[0] for s in statistics}
    if len(halves) != 1:
        raise ValueError("stream lengths differ; refusing to truncate for R-hat")
    half = statistics[0][0]
    means = np.concatenate([s[1] for s in statistics])
    within = np.concatenate([s[2] for s in statistics]).mean(0)
    between = means.var(0, ddof=1)
    # PIXEL reports constant coordinates as 1; flag disagreements explicitly.
    frozen_disagreement = (within == 0) & (between > 0)
    estimates = np.sqrt(((half-1)/half*within + between)
                        / np.where(within > 0, within, 1.0))
    estimates = np.where(within > 0, estimates, 1.0)
    if np.any(frozen_disagreement) or not np.all(np.isfinite(estimates)):
        return {"max": None, "parameter": None,
                "issue": "frozen disagreeing or nonfinite coordinates in R-hat"}
    index = int(np.argmax(estimates)) if estimates.size else None
    return {"max": float(estimates[index]) if index is not None else 1.0,
            "parameter": labels[index] if index is not None and labels else str(index),
            "source": "PIXEL split-R-hat formula on saved chains (streamed half statistics)"}


def summary_diagnostics(record, summary):
    """Only fields actually persisted in the PIXEL inference summary."""
    nuts = record.get("chain_kind") == "nuts" or summary.get("sampler") == "q_nuts"
    return {
        "draws": summary.get("n_samples"), "warmup": summary.get("n_warmup"),
        "tau_int_max": summary.get("tau_int_max"),
        "slowest_parameter": summary.get("slowest_param"),
        "ess_min": summary.get("ess_min"),
        "ess_fraction_min": summary.get("ess_fraction_min"),
        "tau_reliable": summary.get("tau_reliable"),
        "acceptance_statistic": summary.get("mean_accept_prob"),
        "acceptance_rate": None if nuts else summary.get("acceptance_rate"),
        "max_abs_delta_h": summary.get("max_abs_energy_error"),
        "step_size": summary.get("step_size"),
        # A stored median must not be passed off as a trajectory mean.
        "average_depth": None, "steps_per_trajectory": summary.get("mean_n_leapfrog"),
        "num_divergence": summary.get("n_divergent") if nuts else None,
        "num_invalid": summary.get("n_invalid") if not nuts else None,
        "n_max_depth": summary.get("n_max_depth"),
        "wall_seconds": record.get("wall_seconds"),
        "sampler_wall_seconds": record.get("sampler_wall_seconds"),
        "dense_metric_realized": None,
        "source": "saved PIXEL inference_summary.json",
    }


def coverage_totals(coverage):
    rows = [r for r in coverage.values() if r.get("n_points", 0) > 0]
    n = sum(r["n_points"] for r in rows)
    if not n:
        return {}
    result = {"n_points": n,
              "mean_abs_pull": sum(r["n_points"]*r["mean_abs_pull"] for r in rows)/n}
    for limit in (1, 2):
        if all(r.get(f"n_pull_le_{limit}") is not None for r in rows):
            result[f"frac_pull_le_{limit}"] = sum(r[f"n_pull_le_{limit}"] for r in rows)/n
    return result


def pull_statistics(mean, variance, truth, nodes, layout):
    """The old report's finite PDF pulls on 0.01 <= x <= 0.9."""
    with np.errstate(divide="ignore", invalid="ignore"):
        signed = (mean-truth)/np.sqrt(np.clip(variance, 0, None))
    valid = np.isfinite(signed) & (nodes >= .01) & (nodes <= .9)
    values = signed[valid]
    if not values.size:
        raise ValueError("no finite PDF pulls in 0.01 <= x <= 0.9")
    indices = np.flatnonzero(valid)
    worst_index = int(indices[np.argmax(np.abs(values))])
    name = next(str(r["name"]) for r in layout
                if int(r["start"]) <= worst_index < int(r["stop"]))
    per_pdf = {}
    for field in layout:
        sl = slice(int(field["start"]), int(field["stop"]))
        v = signed[sl][valid[sl]]
        if v.size:
            per_pdf[str(field["name"])] = {
                "n_points": int(v.size), "mean_signed_pull": float(v.mean()),
                "mean_abs_pull": float(np.abs(v).mean()),
                "max_abs_pull": float(np.abs(v).max()),
                "pull_chi2_per_point": float(np.mean(v*v)),
                "frac_pull_lt_1": float(np.mean(np.abs(v) < 1)),
                "frac_pull_lt_2": float(np.mean(np.abs(v) < 2)),
            }
    return {
        "n_points": int(values.size), "mean_abs_pull": float(np.abs(values).mean()),
        "max_abs_pull": float(np.abs(values).max()),
        "frac_pull_lt_1": float(np.mean(np.abs(values) < 1)),
        "frac_pull_lt_2": float(np.mean(np.abs(values) < 2)),
        "worst_true_pdf_pull": {"pdf": name, "x": float(nodes[worst_index]),
                                "pull": float(signed[worst_index])},
        "coverage_by_pdf": per_pdf,
    }


def read_moments(path):
    """Read compact sufficient moments; predictive covariance is unnecessary."""
    with np.load(path, allow_pickle=False) as d:
        if str(d["artifact_kind"].item()) != "pixel_posterior_moments" or int(d["artifact_version"].item()) != 1:
            raise ValueError(f"unsupported posterior-moments artifact: {path}")
        result = {k: np.asarray(d[k], dtype=float) for k in ("mean", "truth", "nodes", "predictive_mean")}
        result["variance"] = np.diag(d["covariance"]).copy()
        result.update({key: json.loads(str(d[source].item())) for key, source in (
            ("q_layout", "q_layout_json"), ("prediction_layout", "prediction_layout_json"),
            ("metadata", "metadata_json"))})
    n = result["mean"].shape
    if len(n) != 1 or any(result[k].shape != n for k in ("variance", "truth", "nodes")):
        raise ValueError(f"misaligned PDF moments: {path}")
    if not np.all(np.isfinite(result["mean"])) or not np.all(np.isfinite(result["variance"])):
        raise ValueError(f"nonfinite PDF moments: {path}")
    return result


def prediction_chi2(prediction, layout, model, rcond, precisions=None):
    """Use the suite's dataset covariance, excluding constraints only in rankings."""
    rows = []
    components = {"linear": 0.0, "bilinear": 0.0, "nonlinear": 0.0}
    covered = np.zeros(len(prediction), dtype=bool)
    seen = set()
    for entry in layout:
        index = int(entry["dataset_index"])
        dataset = model.datasets[index]
        start, stop = int(entry["start"]), int(entry["stop"])
        sl = slice(start, stop)
        if index in seen or start < 0 or stop > len(prediction) or stop-start != dataset.n_data or np.any(covered[sl]):
            raise ValueError("prediction layout does not match source datasets")
        if entry.get("name", dataset.name) != dataset.name:
            raise ValueError("prediction dataset name differs from source")
        seen.add(index)
        covered[sl] = True
        residual = prediction[sl]-np.asarray(dataset.mean)
        inverse = (precisions[index] if precisions is not None else
                   np.linalg.pinv(np.asarray(dataset.cov), rcond=rcond))
        value = float(residual @ inverse @ residual)
        kind = entry["prediction_kind"]
        components[kind if kind in ("linear", "bilinear") else "nonlinear"] += value
        rows.append({"dataset": dataset.name, "npt": int(dataset.n_data),
                     "chi2": value, "chi2_per_npt": value/dataset.n_data,
                     "constraint": dataset.name.startswith("cons_") or getattr(dataset, "source", None) == "constraint"})
    if not covered.all() or len(seen) != len(model.datasets):
        raise ValueError("prediction layout does not cover every source data row")
    npt = sum(r["npt"] for r in rows)
    total = sum(r["chi2"] for r in rows)
    physical = [r for r in rows if not r["constraint"]]
    return {"chi2": total, "chi2_per_npt": total/npt, "npt": npt,
            "n_physical_data": sum(r["npt"] for r in physical),
            "n_constraint_data": npt-sum(r["npt"] for r in physical),
            "chi2_components": components,
            "physical_chi2": sum(r["chi2"] for r in physical),
            "worst_dataset_chi2": max(physical, key=lambda r: r["chi2"], default=None),
            "worst_dataset_chi2_per_npt": max(physical, key=lambda r: r["chi2_per_npt"], default=None),
            "per_dataset": rows}


def build_source_model(cell):
    """Restore the same deterministic t0 data preparation as run_fit; no MCMC."""
    from closure_full_q_integration.variant import SOURCE_SUITES
    package = SOURCE_SUITES[(cell["truth"], cell["data"])]
    fit = importlib.import_module(f"{package}.fit")
    cfg = importlib.import_module(f"{package}.config")
    q, mode = cell["scale"], cell["mode"]
    analysis, _ = fit.build_analysis(q, mode, use_kernel_cache=True)
    model = analysis.compile()
    t0, _ = fit._iterate_t0(q, mode, model, use_kernel_cache=True)
    if t0:
        analysis, _ = fit.build_analysis(q, mode, use_kernel_cache=True, t0=t0)
        model = analysis.compile()
    return model, cfg.RCOND


def pooled_physics(cell, streams, model_builder=None):
    """Pool raw posterior moments, including between-stream mean variance.

    Verify every source chi2 against its saved run summary before accepting
    reconstructed dataset covariances. Never average stream chi2/pull values.
    """
    if cell["parameterization"] != "q":
        raise ValueError("pooled moments currently require explicit-q streams")
    model, rcond = (model_builder or build_source_model)(cell)
    precisions = [np.linalg.pinv(np.asarray(d.cov), rcond=rcond) for d in model.datasets]
    reference = None
    mean = variance = prediction = None
    total = 0
    checked = []
    for stream in streams:
        moment = read_moments(Path(stream["moments_path"]))
        if not moment["metadata"].get("matched_nonlinear_draws"):
            raise ValueError("moment artifact does not declare matched nonlinear draws")
        for key, expected in (("q_key", cell["scale"]), ("mode", cell["mode"])):
            if str(moment["metadata"].get(key)) != str(expected):
                raise ValueError(f"moment artifact has different {key}")
        if reference is None:
            reference = moment
        else:
            for key in ("q_layout", "prediction_layout"):
                if moment[key] != reference[key]:
                    raise ValueError(f"streams have different {key}")
            for key in ("truth", "nodes"):
                if not np.array_equal(moment[key], reference[key], equal_nan=True):
                    raise ValueError(f"streams have different {key}")
        measured = prediction_chi2(moment["predictive_mean"], moment["prediction_layout"], model, rcond, precisions)
        saved = stream["fit"]
        if saved.get("sampler") != "q_nuts" or saved["n_data"] != measured["npt"] or saved.get("chi2") is None or not np.isclose(saved["chi2"], measured["chi2"], rtol=1e-8, atol=1e-7):
            raise ValueError(f"stream {stream['number']}: source covariance/predictions do not reproduce saved chi2")
        checked.append({"stream": stream["number"], "saved_chi2": saved["chi2"],
                        "recomputed_chi2": measured["chi2"]})
        count = int(stream["sampler"]["draws"])
        if mean is None:
            mean, variance = moment["mean"].copy(), moment["variance"].copy()
            prediction = moment["predictive_mean"].copy()
        else:
            weight = count/(total+count)
            delta = moment["mean"]-mean
            variance = (1-weight)*variance + weight*moment["variance"] + weight*(1-weight)*delta*delta
            mean += weight*delta
            prediction += weight*(moment["predictive_mean"]-prediction)
        total += count
    result = prediction_chi2(prediction, reference["prediction_layout"], model, rcond, precisions)
    result.update(pull_statistics(mean, variance, reference["truth"], reference["nodes"], reference["q_layout"]))
    result.update(draws=total, stream_numbers=[s["number"] for s in streams], validation=checked,
                  source="saved matched-draw posterior moments, weighted by retained draw count; source chi2 verified per stream")
    return result


def aggregate_sampler(streams):
    """Summarize independent chains without pretending they form one chain."""
    rows = [(r["number"], r["sampler"]) for r in streams if r.get("sampler")]
    if not rows:
        return None
    draws = sum(int(s["draws"]) for _, s in rows)
    tau = [(number, s["tau_int_max"], s.get("slowest_parameter")) for number, s in rows
           if s.get("tau_int_max") is not None and np.isfinite(s["tau_int_max"])]
    worst = max(tau, key=lambda item: item[1]) if tau else None
    # The slowest coordinate may differ by stream. Weighting each stream's
    # maximum tau gives a conservative proxy for the pooled-mean variance
    # when the independent chains sample the same stationary distribution.
    pooled_tau = (sum(int(s["draws"])*s["tau_int_max"] for _, s in rows)/draws
                  if len(tau) == len(rows) else None)
    accept = [(int(s["draws"]), s.get("acceptance_statistic")) for _, s in rows]
    divergence = [s.get("num_divergence") for _, s in rows]
    return {
        "stream_numbers": [number for number, _ in rows],
        "draws": draws,
        "worst_stream_tau_int": worst[1] if worst else None,
        "worst_tau_stream": worst[0] if worst else None,
        "worst_tau_parameter": worst[2] if worst else None,
        "pooled_tau_upper_proxy": pooled_tau,
        "pooled_ess_lower_proxy": draws/(2*pooled_tau) if pooled_tau and pooled_tau > 0 else None,
        "mean_acceptance_statistic": (sum(n*v for n, v in accept)/draws
                                      if all(v is not None for _, v in accept) else None),
        "num_divergence": sum(divergence) if all(v is not None for v in divergence) else None,
        "tau_reliable": all(s.get("tau_reliable") is True for _, s in rows),
    }


def report_cell(root, cell, *, summary_only=False, physics_builder=None):
    if len(cell["streams"]) != cell["stream_count"]:
        raise ValueError("manifest stream count disagrees with stream list")
    numbers = [int(s["number"]) for s in cell["streams"]]
    if sorted(numbers) != list(range(1, cell["stream_count"]+1)):
        raise ValueError("manifest has missing or duplicate stream numbers")
    destination = stream_path(root, cell, 1).parent
    cached_path = destination / "REPORT.json"
    cached = read_json(cached_path) if cached_path.is_file() else None
    result = {"cell": {k: cell[k] for k in (*IDENTITY, "stream_count")},
              "streams": [], "unavailable": [], "rhat": None,
              "sampler_totals": None, "pooled_physics": None}
    stats, stat_numbers, labels, summaries = [], [], None, []
    for entry in sorted(cell["streams"], key=lambda s: s["number"]):
        number = entry["number"]
        path = stream_path(root, cell, number)
        row = {"number": number, "path": str(path), "status": "missing"}
        result["streams"].append(row)
        if not (path / "stream.json").is_file():
            result["unavailable"].append(f"stream {number}: missing stream.json")
            continue
        record = read_json(path / "stream.json")
        row.update(status=record.get("status", "unknown"), record=record)
        for key in IDENTITY:
            if key in record and str(record[key]) != str(cell[key]):
                raise ValueError(f"stream {number}: {key} differs from manifest")
        if record.get("seed") != entry.get("seed"):
            raise ValueError(f"stream {number}: seed differs from manifest")
        if row["status"] != "complete":
            result["unavailable"].append(f"stream {number}: {row['status']} launch record; no completed diagnostics")
            continue
        inference = path / "inference_summary.json"
        if inference.is_file():
            summary = read_json(inference)
            if summary.get("n_samples") != record["samples"] or int(record["samples"]) < 1:
                raise ValueError(f"stream {number}: retained-draw mismatch")
            if record.get("warmup_requested") is not None and summary.get("n_warmup") != record["warmup_requested"]:
                raise ValueError(f"stream {number}: warmup mismatch")
            summaries.append(summary)
            row["sampler"] = summary_diagnostics(record, summary)
        else:
            result["unavailable"].append(f"stream {number}: missing inference_summary.json")
        if not summary_only and (path / "chain.npz").is_file() and inference.is_file():
            _record, chain = load_stream(path)
            chain_labels = tuple(chain.param_labels)
            if labels is not None and chain_labels != labels:
                raise ValueError("parameter labels differ between streams")
            labels = chain_labels
            row["sampler"] = {**stream_diagnostics(record, chain),
                              "source": "recomputed from saved PIXEL chain"}
            row["sampler"]["n_max_depth"] = int(np.sum(chain.reached_max_depth)) if hasattr(chain, "reached_max_depth") else None
            stats.append(split_statistics(chain.samples))
            stat_numbers.append(number)
            del chain
        elif not summary_only and not (path / "chain.npz").is_file():
            result["unavailable"].append(f"stream {number}: missing chain.npz")
        fit_path = path / "results" / f"truthQ_{cell['scale']}" / f"summary_{cell['mode']}.json"
        row["moments_path"] = str(fit_path.with_name(f"posterior_moments_{cell['mode']}.npz"))
        if not summary_only and not Path(row["moments_path"]).is_file():
            result["unavailable"].append(f"stream {number}: missing posterior_moments_{cell['mode']}.npz")
        if fit_path.is_file():
            fit = read_json(fit_path)
            if str(fit["q_key"]) != str(cell["scale"]) or fit["mode"] != cell["mode"]:
                raise ValueError(f"stream {number}: fit summary belongs to another cell")
            # The sampler section already carries the needed diagnostics.
            # Direct diagnostics can include million-entry dense mass matrices;
            # copying them into every report makes a portable JSON hundreds of MB.
            fit.pop("direct_diagnostics", None)
            row["fit"] = fit
            row["pull_summary"] = coverage_totals(fit.get("coverage", {}))
        else:
            result["unavailable"].append(f"stream {number}: missing results/truthQ_{cell['scale']}/summary_{cell['mode']}.json")
    # A saved cross-stream statistic is reusable only with the identical saved
    # inference summaries and cell identity; otherwise it may belong to old draws.
    cache_valid = (cached is not None and cached.get("cell") == result["cell"]
                   and len(summaries) == cell["stream_count"]
                   and cached.get("pixel_inference_summaries") == summaries
                   and len(cached.get("streams", [])) == cell["stream_count"]
                   and all(r["status"] == "complete" for r in result["streams"])
                   and all(all(old.get(k) == row["sampler"].get(k) for k in (
                       "draws", "warmup", "wall_seconds", "sampler_wall_seconds"))
                       for old, row in zip(cached["streams"], result["streams"])))
    if cache_valid:
        for row, old in zip(result["streams"], cached["streams"]):
            if row["sampler"]["source"] != "recomputed from saved PIXEL chain":
                row["sampler"] = {**row["sampler"], **old,
                                  "source": "saved cell REPORT.json, matched inference summaries"}
    result["sampler_totals"] = aggregate_sampler(result["streams"])
    if len(stats) >= 2:
        try:
            result["rhat"] = {**split_rhat(stats, labels), "stream_numbers": stat_numbers}
        except ValueError as exc:
            result["unavailable"].append(f"cross-stream R-hat: {exc}")
    elif cache_valid and cached.get("split_rhat_max") is not None:
        result["rhat"] = {"max": cached["split_rhat_max"], "parameter": cached["split_rhat_parameter"],
                          "source": "saved cell REPORT.json, matched inference summaries",
                          "stream_numbers": [r["number"] for r in result["streams"]]}
    elif cell["stream_count"] > 1:
        result["unavailable"].append("cross-stream R-hat: need at least two completed chains or a matching full cell REPORT.json")
    physics_streams = [r for r in result["streams"] if r["status"] == "complete"
                       and r.get("fit") and r.get("sampler")
                       and Path(r["moments_path"]).is_file()]
    if cell["parameterization"] == "q" and not summary_only and physics_streams:
        try:
            # Libraries' setup messages go to stderr; stdout remains the report.
            with redirect_stdout(sys.stderr):
                result["pooled_physics"] = (physics_builder or pooled_physics)(cell, physics_streams)
        except (ValueError, OSError, KeyError, ImportError, RuntimeError) as exc:
            result["unavailable"].append(f"pooled physics: {type(exc).__name__}: {exc}")
    else:
        result["unavailable"].append("pooled chi2, PDF pulls and worst datasets: need at least one completed stream with posterior moments and source data (omitted with --summary-only)")
    for row in result["streams"]:
        sampler = row.get("sampler")
        if sampler and (sampler.get("average_depth") is None or sampler.get("dense_metric_realized") is None):
            result["unavailable"].append(f"stream {row['number']}: exact mean depth/metric metadata need chain.npz or matching cell REPORT.json")
    if result.get("rhat", {}) and result["rhat"].get("issue"):
        result["unavailable"].append(result["rhat"]["issue"])
    return result


def fmt(value):
    return "unavailable" if value is None else f"{value:.6g}"


def stream_list(numbers):
    return ", ".join(str(n) for n in numbers) if numbers else "none"


def render(report):
    lines = [f"# Full-scale closure report: {report['root']}", "",
             f"Manifest status: {report['manifest_status']} (saved snapshot).",
             f"Requested per stream: {report['requested_samples']} retained, {report['requested_warmup'] if report['requested_warmup'] is not None else 'source-default'} warmup.", ""]
    for result in report["cells"]:
        c = result["cell"]
        lines += [f"## {c['parameterization']} {c['truth']}/{c['data']} Q={c['scale']} {c['mode']}: {c['stream_count']} streams", ""]
        samplers = [r["sampler"] for r in result["streams"] if r.get("sampler")]
        keys = ("draws", "warmup", "tau_int_max", "ess_min", "ess_fraction_min", "tau_reliable")
        if samplers:
            lines += ["| stream | status | draws | warmup | max tau | min ESS | ESS fraction | reliable tau |", "|"+"---|"*8]
        else:
            lines += ["| stream | status |", "|---|---|"]
        for row in result["streams"]:
            s = row.get("sampler", {})
            values = " | ".join(str(s[k]) if isinstance(s.get(k), bool) else fmt(s.get(k)) for k in keys)
            lines.append(f"| {row['number']} | {row['status']} |" + (f" {values} |" if samplers else ""))
        if samplers:
            lines += ["", "| stream | accept stat | max abs dH | step | mean depth | divergences | depth limit hits | steps/traj |", "|"+"---|"*8]
            for row in result["streams"]:
                s = row.get("sampler", {})
                lines.append(f"| {row['number']} | " + " | ".join(fmt(s.get(k)) for k in (
                    "acceptance_statistic", "max_abs_delta_h", "step_size", "average_depth", "num_divergence", "n_max_depth", "steps_per_trajectory")) + " |")
        rhat = result["rhat"] or {}
        lines += ["", f"Worst split R-hat: {fmt(rhat.get('max'))} ({rhat.get('parameter') or 'unavailable'}); streams: {stream_list(rhat.get('stream_numbers'))}.", ""]
        totals = result.get("sampler_totals")
        if totals:
            lines.append(f"Completed streams with sampler diagnostics: {stream_list(totals['stream_numbers'])} ({len(totals['stream_numbers'])}/{c['stream_count']}); {totals['draws']} retained draws.")
            lines.append(f"Aggregate tau upper proxy: {fmt(totals['pooled_tau_upper_proxy'])}; pooled ESS lower proxy: {fmt(totals['pooled_ess_lower_proxy'])}. Worst per-stream tau: {fmt(totals['worst_stream_tau_int'])} (stream {totals['worst_tau_stream'] or 'unavailable'}, {totals['worst_tau_parameter'] or 'unavailable'}).")
            lines.append(f"Draw-weighted NUTS acceptance statistic: {fmt(totals['mean_acceptance_statistic'])}; total divergences: {fmt(totals['num_divergence'])}.")
        if samplers:
            draws = sum(s["draws"] for s in samplers)
            divergence = [s.get("num_divergence") for s in samplers]
            if all(v is not None for v in divergence):
                lines.append(f"Divergences: {sum(divergence)}/{draws} ({sum(divergence)/draws:.3%}), over {len(samplers)}/{c['stream_count']} streams with diagnostics.")
            if any(s.get("tau_reliable") is False for s in samplers):
                lines.append("CAUTION: one or more autocorrelation estimates are flagged unreliable by PIXEL.")
            if any((s.get("num_divergence") or 0) > 0 for s in samplers) or (rhat.get("max") or 0) > 1.01:
                lines.append("CAUTION: divergences or split R-hat > 1.01; pooled fit quantities are provisional.")
            timings = [s["sampler_wall_seconds"]/(s["draws"]+s["warmup"]) for s in samplers
                       if s.get("sampler_wall_seconds") is not None and s.get("warmup") is not None]
            if timings:
                lines.append(f"Mean sampler seconds per warmup+retained trajectory: {np.mean(timings):.6g} (amortized inference-call time).")
        if any(r.get("fit") for r in result["streams"]):
            lines += ["", "| stream | chi2 | chi2/npt | npt incl constraints | physical npt | mean abs PDF pull | frac <=1 | frac <=2 |", "|"+"---|"*8]
            for row in result["streams"]:
                f, p = row.get("fit", {}), row.get("pull_summary", {})
                lines.append(f"| {row['number']} | " + " | ".join(fmt(v) for v in (
                    f.get("chi2"), f.get("chi2_per_data"), f.get("n_data"), f.get("n_physical_data"),
                    p.get("mean_abs_pull"), p.get("frac_pull_le_1"), p.get("frac_pull_le_2"))) + " |")
        physics = result["pooled_physics"]
        if physics:
            lines += ["", f"Pooled posterior: {physics['draws']} retained draws from streams {stream_list(physics.get('stream_numbers'))}; chi2={fmt(physics['chi2'])}, chi2/npt={fmt(physics['chi2_per_npt'])}, npt={physics['npt']} ({physics['n_physical_data']} physical + {physics['n_constraint_data']} constraints).",
                      f"Pooled PDF pulls: mean abs={fmt(physics['mean_abs_pull'])}, max abs={fmt(physics['max_abs_pull'])}, fraction <1={fmt(physics['frac_pull_lt_1'])}, fraction <2={fmt(physics['frac_pull_lt_2'])}.",
                      f"Chi2 components: {json.dumps(physics['chi2_components'], sort_keys=True)}."]
            worst = physics["worst_true_pdf_pull"]
            lines.append(f"Worst PDF pull: {worst['pdf']}, x={fmt(worst['x'])}, signed pull={fmt(worst['pull'])}.")
            for key in ("worst_dataset_chi2", "worst_dataset_chi2_per_npt"):
                d = physics[key]
                if d:
                    lines.append(f"{key}: {d['dataset']}, chi2={fmt(d['chi2'])}, chi2/npt={fmt(d['chi2_per_npt'])}, npt={d['npt']}.")
            lines += ["", "| pooled PDF | mean signed pull | mean abs pull | max abs pull | frac <1 | frac <2 |", "|"+"---|"*6]
            for name, p in physics["coverage_by_pdf"].items():
                lines.append(f"| {name} | " + " | ".join(fmt(p[k]) for k in ("mean_signed_pull", "mean_abs_pull", "max_abs_pull", "frac_pull_lt_1", "frac_pull_lt_2")) + " |")
        if samplers:
            lines += ["", "| stream | dense realized | initialization | fallback | slowest coordinate |", "|"+"---|"*5]
            for row in result["streams"]:
                s = row.get("sampler", {})
                dense = s.get("dense_metric_realized")
                fallback = s.get("metric_fallback_reason") or ("none recorded" if dense is not None else "unavailable")
                lines.append(f"| {row['number']} | {str(dense) if dense is not None else 'unavailable'} | {s.get('metric_initialization') or 'unavailable'} | {fallback} | {s.get('slowest_parameter', 'unavailable')} |")
            lines += ["", "| stream | sampler seconds | total seconds | sampler s/trajectory |", "|"+"---|"*4]
            for row in result["streams"]:
                s = row.get("sampler", {})
                denominator = (s.get("draws") or 0)+(s.get("warmup") or 0)
                per_traj = s["sampler_wall_seconds"]/denominator if denominator and s.get("warmup") is not None and s.get("sampler_wall_seconds") is not None else None
                lines.append(f"| {row['number']} | {fmt(s.get('sampler_wall_seconds'))} | {fmt(s.get('wall_seconds'))} | {fmt(per_traj)} |")
            lines += ["", "Diagnostic source(s): " + "; ".join(sorted({s["source"] for s in samplers})) + "."]
            if rhat.get("source"):
                lines.append("R-hat source: " + rhat["source"] + ".")
        if result["unavailable"]:
            lines += ["", "Unavailable inputs/statistics:", ""]
            lines += [f"- {item}" for item in result["unavailable"]]
        lines.append("")
    lines += ["Tau uses Wolff's convention: independent draws have tau=0.5; ESS=N/(2*tau). The aggregate tau upper proxy is the draw-weighted mean of each independent stream's maximum tau; its ESS lower proxy assumes a shared stationary distribution. Check split R-hat before interpreting either as pooled precision.",
              "NUTS acceptance is a mean transition acceptance statistic. It is not a binary acceptance rate.",
              "Chi2/npt is not chi2/degrees of freedom; totals include constraint rows. Dataset rankings exclude constraints.",
              "PDF pulls use posterior standard deviations and finite nodes in 0.01 <= x <= 0.9. Nodes are correlated; fractions are descriptive.",
              "Pooled moments include between-stream mean variation and are weighted by draws, not by ESS. Stream chi2/pulls are never averaged into pooled statistics.", ""]
    return "\n".join(lines)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--summary-only", action="store_true", help="read small JSON files only; omit chain recomputation and pooled moments")
    parser.add_argument("--out", type=Path, help="also write <out>.md and <out>.json for transfer")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        manifest = read_json(root / "manifest.json")
        report = {"root": str(root), "manifest_status": manifest.get("status", "unknown"),
                  "requested_samples": manifest.get("samples_per_stream"),
                  "requested_warmup": manifest.get("warmup_per_stream_requested"),
                  "cells": [report_cell(root, cell, summary_only=args.summary_only) for cell in manifest["cells"]]}
        report = json_safe(report)
        body = render(report)
        print(body)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.with_suffix(".md").write_text(body)
            args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return 2 if any(c["unavailable"] for c in report["cells"]) else 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"Cannot report: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
