"""Observable-space closure plots: fake data, injected truth, reproduction.

This is a compact diagnostic companion to ``run_closure``.  Each panel shows one
generated PIXEL dataset in observable space, with panels sorted by chi2
contribution so the largest tensions appear first.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import config as cfg
from . import fit as fitmod
from .run_closure import _prepare_matplotlib, save_figure_both


def _dataset_x(dataset):
    """Return ``(x, label, use_log)`` for plotting a dataset."""
    independent = getattr(dataset, "independent", None)
    if not independent:
        independent = getattr(getattr(dataset, "metadata", None), "independent", {})
    if all(key in independent for key in ("Q2", "S", "Y")):
        q2 = np.asarray(independent["Q2"], dtype=float).reshape(-1)
        s = np.asarray(independent["S"], dtype=float).reshape(-1)
        rapidity = np.asarray(independent["Y"], dtype=float).reshape(-1)
        if q2.shape == s.shape == rapidity.shape and np.all(s > 0.0):
            x_f = 2.0 * np.sqrt(q2 / s) * np.sinh(rapidity)
            return x_f, r"$x_F$", False
    for key, label in (("x", "x"), ("Y", "Y"), ("nu", "nu"), ("n", "n")):
        if key in independent:
            x = np.asarray(independent[key], dtype=float).reshape(-1)
            use_log = label == "x" and x.size and np.nanmin(x) > 0.0
            return x, label, bool(use_log)
    nu = np.asarray(dataset.nu, dtype=float)
    if nu.ndim == 1:
        return nu.reshape(-1), "nu", False
    return np.arange(dataset.n_data, dtype=float), "point", False


def _diag_sigma(dataset):
    cov = np.asarray(dataset.cov, dtype=float)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def _is_constraint_dataset(dataset) -> bool:
    """Return True for internal hard-constraint pseudo-data."""
    return (
        getattr(dataset, "source", None) == "constraint"
        or str(getattr(dataset, "name", "")).startswith("cons_")
    )


def _sample_vectors_and_weights(samples):
    """Return ``(vectors, weights)`` for either MAP-like or MCMC samples.

    A MAP-like result (anything carrying ``.x``) is one row with unit weight.
    Otherwise ``.samples`` must be the sampler contract shape ``(n_samples,
    n_params)``: every shipped sampler allocates exactly that -- ``infer/hmc.py``
    and ``infer/nuts.py`` via ``np.empty((n_samples, n_params))``,
    ``infer/vegas.py`` and ``infer/mcmc.py`` by stacking per-draw rows, and all
    four spell the zero-parameter case ``np.zeros((n_samples, 0))`` rather than
    collapsing it to 1-D.

    **A 1-D ``.samples`` is rejected, not reshaped.**  It is ambiguous -- one
    draw of ``n`` parameters or ``n`` draws of one parameter -- and the two
    readings were split across this function and :func:`fit._weights`, which
    counts ``samples.samples.shape[0]`` independently.  For ``shape == (3,)``
    the old reshape returned *one* vector while ``_weights`` returned *three*
    weights of ``1/3``; the caller's ``zip`` then truncated to a single pair and
    the reported band came out scaled by ``1/n`` -- a silently wrong uncertainty,
    with nothing raised.  Since no sampler can produce that shape, refusing it is
    strictly better than picking a reading.
    """
    if hasattr(samples, "x"):
        return np.asarray(samples.x, dtype=float).reshape(1, -1), np.ones(1)
    vectors = np.asarray(samples.samples, dtype=float)
    if vectors.ndim != 2:
        raise ValueError(
            "closure plotting: samples.samples must be 2-D "
            f"(n_samples, n_params); got shape {vectors.shape}"
        )
    return vectors, fitmod._weights(samples)


def _sampled_linear_predictive_band(model, samples):
    """Linear-data moments from ordinary parameter samples."""
    cached = getattr(samples, "_pixel_linear_predictive_moments", None)
    if cached is not None:
        mean, std = cached
        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.asarray(std, dtype=float).reshape(-1)
        if mean.size == int(np.asarray(model._layout.M).size) and std.size == mean.size:
            return mean, std

    from pixel.core.evidence import posterior

    vectors, weights = _sample_vectors_and_weights(samples)
    # zip() below truncates to the shorter sequence, so a count mismatch would
    # silently drop samples and renormalize the band by the wrong total instead
    # of failing.  Name both lengths.
    if len(vectors) != len(weights):
        raise ValueError(
            "closure plotting: sample-vector count and weight count disagree "
            f"({len(vectors)} vectors, {len(weights)} weights); the predictive "
            "band would be silently truncated by zip()"
        )
    mean_acc = None
    second_acc = None
    for vec, weight in zip(vectors, weights):
        blocks = model._blocks(model._rebuild(vec))
        qbar, cov = posterior(blocks, model.rcond)
        bmat = np.asarray(blocks.B, dtype=float)
        qbar = np.asarray(qbar, dtype=float)
        cov = np.asarray(cov, dtype=float)
        mean = np.asarray(blocks.param_pred, dtype=float) + bmat @ qbar
        var = np.clip(np.einsum("ij,jk,ik->i", bmat, cov, bmat), 0.0, None)
        if mean_acc is None:
            mean_acc = np.zeros_like(mean)
            second_acc = np.zeros_like(mean)
        mean_acc += weight * mean
        second_acc += weight * (var + mean * mean)
    var = np.clip(second_acc - mean_acc * mean_acc, 0.0, None)
    return mean_acc, np.sqrt(var)


def _physical_gp_moments(samples, n_gp):
    """Return marginalized physical GP moments when the sampler stored them."""
    mean = getattr(samples, "mean", None)
    covariance = getattr(samples, "covariance", None)
    if mean is None or covariance is None or callable(mean) or callable(covariance):
        return None
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if mean.size != n_gp or covariance.shape != (n_gp, n_gp):
        return None
    return mean, covariance


def _representative_vector(model, samples):
    """Return a complete model free vector across old and new closure runners."""
    vec = np.asarray(fitmod._weighted_mean_vec(samples), dtype=float).reshape(-1)
    if vec.size == model.n_free:
        return vec
    try:
        vec = np.asarray(
            fitmod._weighted_mean_vec(samples, model=model), dtype=float
        ).reshape(-1)
    except TypeError:
        pass
    if vec.size != model.n_free:
        raise ValueError(
            f"posterior representative has {vec.size} parameters; "
            f"compiled model expects {model.n_free}"
        )
    return vec


def _bilinear_symmetric_action(blocks, vector):
    """Apply each symmetric DY operator to one GP vector without densifying it."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    indices = blocks.bilinear_active_indices
    if indices is not None:
        vector = vector[np.asarray(indices, dtype=int)]

    operators = blocks.bilinear_Y
    if isinstance(operators, tuple):
        if not operators:
            return np.zeros((0, vector.size), dtype=float)
        n_rows = int(np.asarray(blocks.bilinear_M).size)
        n_fields = int(operators[0].mixing.shape[0])
        n_grid = int(operators[0].tensor.shape[1])
        if vector.size != n_fields * n_grid:
            raise ValueError("factorized DY operator does not match GP moment size")
        q = vector.reshape(n_fields, n_grid)
        action = np.zeros((n_rows, n_fields, n_grid), dtype=float)
        for block in operators:
            mixing = np.asarray(block.mixing, dtype=float)
            tensor = np.asarray(block.tensor, dtype=float)
            forward = np.einsum(
                "ab,rij,bj->rai", mixing, tensor, q, optimize=True
            )
            backward = np.einsum(
                "ab,rij,ai->rbj", mixing, tensor, q, optimize=True
            )
            sl = slice(int(block.row_start), int(block.row_stop))
            action[sl] += 0.5 * (forward + backward)
        action = action.reshape(n_rows, -1)
    else:
        tensor = np.asarray(operators, dtype=float)
        forward = np.einsum("rij,j->ri", tensor, vector, optimize=True)
        backward = np.einsum("rji,j->ri", tensor, vector, optimize=True)
        action = 0.5 * (forward + backward)

    if blocks.bilinear_scale is not None:
        action *= np.asarray(blocks.bilinear_scale, dtype=float)[:, None]
    return action


def _bilinear_predictive_band(blocks, mean, covariance):
    """DY posterior mean and a first-order propagated standard deviation."""
    from pixel.core.evidence import _bilinear_prediction_from_second

    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    second = covariance + np.outer(mean, mean)
    prediction = np.asarray(
        _bilinear_prediction_from_second(blocks, second), dtype=float
    )

    indices = blocks.bilinear_active_indices
    if indices is not None:
        active = np.asarray(indices, dtype=int)
        covariance = covariance[np.ix_(active, active)]
    gradient_half = _bilinear_symmetric_action(blocks, mean)
    # grad(q^T Y q) = 2 sym(Y) q.  The physical posterior is not exactly
    # Gaussian after a DY likelihood, so this deliberately reports the stable
    # delta-method band available from its stored first two GP moments.
    variance = 4.0 * np.einsum(
        "ri,ij,rj->r", gradient_half, covariance, gradient_half, optimize=True
    )
    return prediction, np.sqrt(np.clip(variance, 0.0, None))


def _posterior_predictive_bands(model, samples, vec):
    """Return linear and DY posterior means/standard deviations in data space."""
    blocks = model._blocks(model._rebuild(vec))
    moments = _physical_gp_moments(samples, int(np.asarray(blocks.g).size))
    if model.has_bilinear and moments is not None:
        mean, covariance = moments
        bmat = np.asarray(blocks.B, dtype=float)
        linear_mean = np.asarray(blocks.param_pred, dtype=float) + bmat @ mean
        linear_var = np.einsum(
            "ij,jk,ik->i", bmat, covariance, bmat, optimize=True
        )
        dy_mean, dy_std = _bilinear_predictive_band(blocks, mean, covariance)
        return (
            linear_mean,
            np.sqrt(np.clip(linear_var, 0.0, None)),
            dy_mean,
            dy_std,
            blocks,
        )

    linear_mean, linear_std = _sampled_linear_predictive_band(model, samples)
    return linear_mean, linear_std, np.zeros(0), np.zeros(0), blocks


def _truth_predictions(model, truth, blocks):
    """Fold saved truth curves through both linear and bilinear operators."""
    from pixel.core.evidence import _bilinear_prediction_from_second

    saved_x = np.asarray(truth["x_nodes"], dtype=float)
    truth_curves = dict(truth["curves"])
    for field, curves in (truth.get("systematic_curves") or {}).items():
        for key, curve in curves.items():
            truth_curves[cfg.nuisance_field_name(field, key)] = curve
    moment_truth = truth.get("moment_systematic_values") or {}
    pieces = []
    for index in model.gp_indices:
        field = model.fields[index]
        nodes = np.asarray(field.nodes, dtype=float).reshape(-1)
        if field.name in moment_truth:
            pieces.append(np.full(nodes.size, float(moment_truth[field.name])))
        else:
            curve = np.asarray(
                truth_curves.get(field.name, np.zeros_like(saved_x)), dtype=float
            )
            pieces.append(np.interp(nodes, saved_x, curve))
    q_truth = np.concatenate(pieces) if pieces else np.zeros((0,), dtype=float)
    linear = (
        np.asarray(blocks.param_pred, dtype=float)
        + np.asarray(blocks.B, dtype=float) @ q_truth
    )
    if model.has_bilinear:
        bilinear = np.asarray(
            _bilinear_prediction_from_second(blocks, np.outer(q_truth, q_truth)),
            dtype=float,
        )
    else:
        bilinear = np.zeros((0,), dtype=float)
    return linear, bilinear


def _record_chi2(dataset, prediction, rcond):
    """Chi2 for one plotted dataset using the same pseudo-inverse tolerance."""
    residual = np.asarray(dataset.mean, dtype=float) - np.asarray(prediction, dtype=float)
    cov = np.asarray(dataset.cov, dtype=float)
    inverse = np.linalg.pinv(cov, rcond=float(rcond), hermitian=True)
    return float(residual @ inverse @ residual)


def build_records_from_result(q_key: str, mode: str, result: dict,
                              truth: dict | None = None):
    """Build sorted per-dataset plot records from an already-computed fit.

    Constraint rows are omitted.  Linear datasets and bilinear Drell--Yan rows
    use their separate layout slices, but are returned in one chi2-sorted list.
    """
    model = result["model"]
    samples = result["samples"]
    linear_layouts = [
        layout for layout in model._layout.datasets
        if not _is_constraint_dataset(model.datasets[layout.dataset_index])
    ]
    bilinear_layouts = [
        layout for layout in model._layout.bilinear_datasets
        if not _is_constraint_dataset(model.datasets[layout.dataset_index])
    ]
    if not linear_layouts and not bilinear_layouts:
        return []

    truth = truth if truth is not None else fitmod.load_truth(q_key)
    vec = _representative_vector(model, samples)
    pred, pred_std, dy_pred, dy_std, blocks = _posterior_predictive_bands(
        model, samples, vec
    )
    truth_pred, dy_truth = _truth_predictions(model, truth, blocks)

    records = []
    for kind, layouts, prediction, prediction_std, truth_prediction in (
        ("linear", linear_layouts, pred, pred_std, truth_pred),
        ("dy", bilinear_layouts, dy_pred, dy_std, dy_truth),
    ):
        for layout in layouts:
            dataset = model.datasets[layout.dataset_index]
            sl = layout.data_slice
            x, xlabel, use_log = _dataset_x(dataset)
            order = np.argsort(x) if x.size == dataset.n_data else np.arange(dataset.n_data)
            reproduction = np.asarray(prediction[sl], dtype=float)
            dataset_chi2 = _record_chi2(dataset, reproduction, model.rcond)
            records.append({
                "name": dataset.name,
                "source": dataset.source,
                "component": dataset.component,
                "kind": kind,
                "x": x[order],
                "xlabel": xlabel,
                "use_log": use_log,
                "data": np.asarray(dataset.mean, dtype=float)[order],
                "sigma": _diag_sigma(dataset)[order],
                "truth": np.asarray(truth_prediction[sl], dtype=float)[order],
                "reproduction": reproduction[order],
                "fit_std": np.asarray(prediction_std[sl], dtype=float)[order],
                "chi2": dataset_chi2,
                "chi2_per_data": (
                    dataset_chi2 / dataset.n_data if dataset.n_data else np.nan
                ),
                "n_data": int(dataset.n_data),
            })
    records.sort(key=lambda r: np.nan_to_num(r["chi2"], nan=-np.inf), reverse=True)
    return records


def build_records(q_key: str, mode: str):
    """Run one closure fit and build sorted per-dataset plot records."""
    result = fitmod.run_fit(q_key, mode)
    return build_records_from_result(q_key, mode, result, fitmod.load_truth(q_key))


def plot_records(records, path: Path, *, q_key: str, mode: str):
    """Write the combined dataset-reproduction plot.

    A no-op when ``records`` is empty.
    """
    if not records:
        return None
    _prepare_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows),
                             squeeze=False)
    for ax, rec in zip(axes.ravel(), records):
        x = rec["x"]
        ax.errorbar(x, rec["data"], yerr=rec["sigma"], fmt="ko", ms=3.0,
                    lw=0.7, capsize=2, label="fake data")
        ax.plot(x, rec["truth"], color="tab:blue", marker="s", ms=3.0,
                lw=1.2, label="truth mean")
        fit_lo = rec["reproduction"] - rec["fit_std"]
        fit_hi = rec["reproduction"] + rec["fit_std"]
        if x.size >= 2:
            ax.fill_between(x, fit_lo, fit_hi, color="tab:red", alpha=0.18,
                            linewidth=0.0, label="fit band")
        else:
            ax.errorbar(x, rec["reproduction"], yerr=rec["fit_std"], fmt="o",
                        color="tab:red", ms=3.0, lw=0.8, capsize=2,
                        label="fit band")
        ax.plot(x, rec["reproduction"], color="tab:red", marker="o", ms=2.8,
                lw=1.2, label="fit mean")
        if rec["use_log"]:
            ax.set_xscale("log")
        ax.axhline(0.0, color="0.75", lw=0.6)
        ax.grid(alpha=0.2, lw=0.5)
        kind = " [DY]" if rec.get("kind") == "dy" else ""
        title = (f"{rec['name']}{kind}  chi2/n={rec['chi2_per_data']:.2g}"
                 f"  n={rec['n_data']}")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(rec["xlabel"], fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8, loc="best")
    title = (
        f"Closure observable reproduction: {cfg.truth_label(q_key)} "
        f"(Q={cfg.TRUTH_Q_CHOICES[q_key]} GeV)"
    )
    title += (", " if ncols == 3 else "\n") + f"mode={mode}; panels sorted by chi2"
    fig.suptitle(title, fontsize=13)
    if any(rec.get("kind") == "dy" for rec in records):
        fig.text(
            0.5, 0.005,
            "DY fit bands use first-order propagation of the marginalized PDF covariance.",
            ha="center", va="bottom", fontsize=8, color="0.35",
        )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92 if ncols < 3 else 0.985))
    save_figure_both(fig, path, dpi=180)
    plt.close(fig)
    return path


def write_summary(records, out: Path, *, q_key: str, mode: str):
    """Write the JSON companion summary for a dataset-reproduction plot."""
    payload = {
        "q_key": q_key,
        "mode": mode,
        "output": str(out),
        "top_chi2": [
            {
                "name": r["name"],
                "kind": r.get("kind", "linear"),
                "n_data": r["n_data"],
                "chi2": r["chi2"],
                "chi2_per_data": r["chi2_per_data"],
            }
            for r in records[:10]
        ],
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(payload, indent=2))
    return payload, summary_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot closure data/truth/reproduction.")
    ap.add_argument("--Q", dest="q", choices=list(cfg.TRUTH_Q_CHOICES),
                    default=cfg.DEFAULT_TRUTH_Q)
    ap.add_argument("--mode", choices=cfg.TEST_MODES, default="both")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    records = build_records(args.q, args.mode)
    out = args.output or (
        cfg.results_dir(args.q) / f"dataset_reproduction_{args.mode}.png"
    )
    if not records:
        # Do not report files that were never written.
        print(f"no physical datasets to plot for mode={args.mode}; nothing written")
        return
    plot_records(records, out, q_key=args.q, mode=args.mode)
    payload, summary_path = write_summary(records, out, q_key=args.q, mode=args.mode)
    print(f"wrote {out} (+ sibling PNG/PDF)")
    print(f"wrote {summary_path}")
    for row in payload["top_chi2"][:5]:
        print(f"  {row['name']:<28} chi2={row['chi2']:.4g} "
              f"chi2/n={row['chi2_per_data']:.4g} n={row['n_data']}")


if __name__ == "__main__":
    main()
