"""No-sampling screen of one fixed prior against one suite's Q=2 data.

Everything here is a closed-form Gaussian-prior quantity evaluated on the compiled
``dy``-mode model -- no MCMC, no VEGAS, no saddle search -- so a few hundred cells can be
ranked before any fit is spent on them:

* **factorability** -- every field's prior covariance block must be Cholesky-factorable;
* **DY prior predictive** -- ``q^T Y_r q`` for ``q ~ N(g, K)`` has mean
  ``tr(Y_r K) + g^T Y_r g`` and variance ``2 tr(Y_r K Y_r K) + 4 g^T Y_r K Y_r g`` (``Y_r``
  symmetrized).  ``z_pred = |E - m| / sqrt(sigma^2 + Var)`` says how far the prior sits from
  each datum *in the prior's own predictive units*; ``z_data`` uses the data error alone;
* **truth fold** -- the injected truth contracted through the same operators must reproduce
  the pseudo-data at noise level, or the scan is measuring a broken dataset;
* **constraint tension** -- each ``cons_*`` row's prior predictive pull;
* **prior coverage** -- ``max |truth - g| / sqrt(diag K)`` over the bulk ``x in [0.01, 0.9]``
  per field, and for the envelope the all-``x`` ratio ``max |truth| / sqrt(diag K)`` (the
  owner's derivation rule wants it <= 1).
"""

from __future__ import annotations

import importlib
import time
import traceback

import numpy as np

BULK = (0.01, 0.9)


def _dense_symmetric_rows(blocks) -> np.ndarray:
    """``Ys[r]`` over the active GP subspace, from dense or factorized operators."""
    operators = blocks.bilinear_Y
    n_rows = int(np.asarray(blocks.bilinear_M).size)
    if isinstance(operators, tuple):
        n_fields = int(operators[0].mixing.shape[0])
        n_grid = int(operators[0].tensor.shape[1])
        n = n_fields * n_grid
        Y = np.zeros((n_rows, n, n), dtype=float)
        for block in operators:
            part = np.einsum(
                "ab,rij->raibj",
                np.asarray(block.mixing, dtype=float),
                np.asarray(block.tensor, dtype=float),
                optimize=True,
            ).reshape(-1, n, n)
            Y[int(block.row_start):int(block.row_stop)] += part
    else:
        Y = np.array(np.asarray(operators, dtype=float))
    scale = getattr(blocks, "bilinear_scale", None)
    if scale is not None:
        Y *= np.asarray(scale, dtype=float).reshape(-1, 1, 1)
    return 0.5 * (Y + np.swapaxes(Y, 1, 2))


def _field_layout(model, n_total):
    names = [model.fields[i].name for i in model.gp_indices]
    n_grid = n_total // len(names)
    return names, n_grid


def screen_model(model, truth, plot_datasets) -> dict:
    """All screen metrics for one compiled model."""
    vec = np.asarray(model.free_vector(), dtype=float)
    blocks = model._blocks(model._rebuild(vec))
    g = np.asarray(blocks.g, dtype=float)
    K = np.asarray(blocks.K, dtype=float)
    names, n_grid = _field_layout(model, g.size)
    out: dict = {"n_gp": int(g.size)}

    # Factorability, per field block.
    min_eig = {}
    factorable = True
    for k, name in enumerate(names):
        sl = slice(k * n_grid, (k + 1) * n_grid)
        block = 0.5 * (K[sl, sl] + K[sl, sl].T)
        min_eig[name] = float(np.linalg.eigvalsh(block)[0])
        try:
            np.linalg.cholesky(block)
        except np.linalg.LinAlgError:
            factorable = False
    out["factorable"] = factorable
    out["min_eig"] = min_eig

    # Truth on the field nodes, in the model's GP ordering.
    saved_x = np.asarray(truth["x_nodes"], dtype=float)
    pieces, nodes = [], None
    for index in model.gp_indices:
        field = model.fields[index]
        nodes = np.asarray(field.nodes, dtype=float).reshape(-1)
        pieces.append(np.interp(nodes, saved_x, np.asarray(truth["curves"][field.name], dtype=float)))
    q_truth = np.concatenate(pieces)
    sd_prior = np.sqrt(np.clip(np.diag(K), 0.0, None))
    bulk = (nodes >= BULK[0]) & (nodes <= BULK[1])
    coverage, bound = {}, {}
    for k, name in enumerate(names):
        sl = slice(k * n_grid, (k + 1) * n_grid)
        z = np.abs(q_truth[sl] - g[sl]) / sd_prior[sl]
        coverage[name] = float(np.max(z[bulk]))
        bound[name] = float(np.max(np.abs(q_truth[sl]) / sd_prior[sl]))
    out["coverage_bulk_max_z"] = coverage
    out["coverage_bulk_max_z_worst"] = max(coverage.values())
    out["envelope_bound_ratio"] = bound
    out["envelope_bound_ratio_worst"] = max(bound.values())

    # Constraint prior tension.
    B = np.asarray(blocks.B, dtype=float)
    M = np.asarray(blocks.M, dtype=float)
    C = np.asarray(blocks.C, dtype=float)
    pred = np.asarray(blocks.param_pred, dtype=float) + B @ g
    var = np.einsum("ij,jk,ik->i", B, K, B, optimize=True) + np.diag(C)
    tension = {}
    for layout in model._layout.datasets:
        dataset = model.datasets[layout.dataset_index]
        sl = layout.data_slice
        z = np.abs(pred[sl] - M[sl]) / np.sqrt(np.clip(var[sl], 1e-300, None))
        tension[dataset.name] = float(np.max(z))
    out["linear_prior_z"] = tension
    out["constraint_prior_z_worst"] = max(
        (v for k, v in tension.items() if k.startswith("cons_")), default=0.0
    )

    # Drell-Yan prior predictive and truth fold.
    if model.has_bilinear:
        m = np.asarray(blocks.bilinear_M, dtype=float)
        sd = np.sqrt(np.clip(np.diag(np.asarray(blocks.bilinear_C, dtype=float)), 0.0, None))
        idx = blocks.bilinear_active_indices
        ga, Ka = (g, K) if idx is None else (g[np.asarray(idx)], K[np.ix_(np.asarray(idx), np.asarray(idx))])
        Ys = _dense_symmetric_rows(blocks)
        YK = np.einsum("rij,jk->rik", Ys, Ka, optimize=True)
        mean = np.einsum("rii->r", YK) + np.einsum("i,rij,j->r", ga, Ys, ga, optimize=True)
        variance = 2.0 * np.einsum("rij,rji->r", YK, YK, optimize=True) + 4.0 * np.einsum(
            "i,rij,j->r", ga, np.einsum("rij,rjk->rik", YK, Ys, optimize=True), ga, optimize=True
        )
        z_data = np.abs(mean - m) / sd
        z_pred = np.abs(mean - m) / np.sqrt(sd**2 + np.clip(variance, 0.0, None))
        _, dy_truth = plot_datasets._truth_predictions(model, truth, blocks)
        truth_pull = (np.asarray(dy_truth, dtype=float) - m) / sd
        out.update({
            "dy_rows": int(m.size),
            "dy_prior_mean_over_data_median": float(np.median(mean / m)),
            "dy_z_data_median": float(np.median(z_data)),
            "dy_z_data_max": float(np.max(z_data)),
            "dy_z_pred_median": float(np.median(z_pred)),
            "dy_z_pred_max": float(np.max(z_pred)),
            "dy_prior_rel_width_median": float(np.median(np.sqrt(np.clip(variance, 0, None)) / np.abs(m))),
            # How much each datum narrows the prior: prior predictive sd over data sd.
            # A value of 1e3 means the posterior is a 1e-3 sliver of the prior along that
            # row, which is what collapses importance weights and stiffens the H-S saddle.
            "dy_info_ratio_median": float(np.median(np.sqrt(np.clip(variance, 0, None)) / sd)),
            "dy_info_ratio_max": float(np.max(np.sqrt(np.clip(variance, 0, None)) / sd)),
            "dy_truth_pull_rms": float(np.sqrt(np.mean(truth_pull**2))),
            "dy_truth_pull_max": float(np.max(np.abs(truth_pull))),
        })
    return out


def screen_cell(suite: str, cell: dict, q_key: str = "2", mode: str = "dy") -> dict:
    """Build ``suite`` at ``q_key``/``mode`` under ``cell``'s fixed prior and screen it."""
    from .prior_cells import cell_hash, patched_prior

    cfg = importlib.import_module(f"{suite}.config")
    fitmod = importlib.import_module(f"{suite}.fit")
    plot_datasets = importlib.import_module(f"{suite}.plot_datasets")
    row = {"suite": suite, "q": q_key, "mode": mode, "cell": cell["name"],
           "cell_hash": cell_hash(cell), "prior_form": cell["prior_form"],
           "provenance": cell.get("provenance")}
    started = time.perf_counter()
    try:
        with patched_prior(cfg, cell):
            analysis, _ = fitmod.build_analysis(q_key, mode)
            model = analysis.compile()
            row.update(screen_model(model, fitmod.load_truth(q_key), plot_datasets))
        row["status"] = "ok" if row.get("factorable") else "factor_error"
    except Exception as exc:
        row.update({"status": "error", "exception": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(limit=6)})
    row["elapsed_s"] = round(time.perf_counter() - started, 2)
    return row
