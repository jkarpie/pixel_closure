"""MAP-centred numerical evidence scan of full JAM q-fit normalizations.

Run on a machine with the already-built closure kernel cache. This does not
generate truth, run NUTS, or run posterior marginalization. Cache misses inside
PIXEL can still build missing kernels, so a minimal cache-size guard prevents
accidental execution against an empty or barely started cache. It rebuilds the
compiled target, finds/loads its MAP, and reuses PIXEL's evidence-scan evaluator.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def _number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.5g}" if np.isfinite(value) else str(value)


def _minimum(values: np.ndarray):
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, None
    masked = np.where(finite, values, np.inf)
    location = np.unravel_index(np.argmin(masked), values.shape)
    return float(values[location]), location


def summarize_scan(scan) -> dict:
    """Measure finite-grid shape without assuming a Gaussian landscape."""
    middle = scan.n_points // 2
    if scan.n_points % 2 != 1:
        raise ValueError("an odd number of grid points is required")
    center = float(scan.center_value)
    axes = []
    for label, axis, values in zip(scan.labels, scan.axes, scan.diagonal):
        axis = np.asarray(axis, dtype=float)
        delta = np.asarray(values, dtype=float) - center
        spacing = float(axis[middle + 1] - axis[middle])
        curvature = float(
            (delta[middle - 1] - 2 * delta[middle] + delta[middle + 1])
            / spacing**2
        )
        minimum, location = _minimum(delta)
        axes.append({
            "label": label,
            "map_coordinate": float(scan.center[scan.indices[len(axes)]]),
            "delta_minus_one": float(delta[0]),
            "delta_plus_one": float(delta[-1]),
            "range_delta": float(np.ptp(delta[np.isfinite(delta)]))
            if np.any(np.isfinite(delta)) else None,
            "curvature_at_map": curvature,
            "conditional_laplace_sigma": (
                float(np.sqrt(2.0 / curvature)) if curvature > 0 else None
            ),
            "lowest_grid_delta": minimum,
            "lowest_grid_offset": (
                float(axis[location[0]] - axis[middle])
                if location is not None else None
            ),
            "asymmetry_plus_minus": float(delta[-1] - delta[0]),
        })
    pairs = []
    for (i, j), values in scan.panels.items():
        grid = np.asarray(values, dtype=float) - center
        interaction_pp = float(
            grid[-1, -1] - grid[-1, middle] - grid[middle, -1]
            + grid[middle, middle]
        )
        interaction_pm = float(
            grid[-1, 0] - grid[-1, middle] - grid[middle, 0]
            + grid[middle, middle]
        )
        minimum, min_index = _minimum(grid)
        pairs.append({
            "labels": [scan.labels[i], scan.labels[j]],
            "lowest_grid_delta": minimum,
            "lowest_grid_offsets": [
                float(scan.axes[i][min_index[0]] - scan.axes[i][middle]),
                float(scan.axes[j][min_index[1]] - scan.axes[j][middle]),
            ] if min_index is not None else None,
            "interaction_plus_plus": interaction_pp,
            "interaction_plus_minus": interaction_pm,
            "range_delta": float(np.ptp(grid[np.isfinite(grid)]))
            if np.any(np.isfinite(grid)) else None,
        })
    return {
        "center_action": center,
        "n_points": scan.n_points,
        "n_invalid": int(scan.n_invalid),
        "coordinate_system": scan.coordinates,
        "slice": "conditional: all unscanned coordinates fixed at MAP",
        "axes": axes,
        "pairs": pairs,
    }


def _grid_markdown(scan, summary: dict) -> str:
    middle = scan.n_points // 2
    offsets = np.asarray(scan.axes[0]) - scan.axes[0][middle]
    lines = [
        "# Full q normalization evidence-grid values",
        "",
        "Each entry is ΔS = S(node) − S(MAP). A unit offset is one standard",
        "deviation of that normalization amplitude's N(0,1) prior. The other",
        "coordinates are held at the MAP; these are conditional slices, not",
        "profile likelihoods or marginal posteriors.",
        "",
        "## One-dimensional slices",
        "",
        "| parameter | " + " | ".join(_number(x) for x in offsets) + " |",
        "| --- | " + " | ".join("---:" for _ in offsets) + " |",
    ]
    for label, values in zip(scan.labels, scan.diagonal):
        lines.append(
            "| " + label + " | " + " | ".join(
                _number(float(x - scan.center_value)) for x in values
            ) + " |"
        )
    lines += [
        "",
        "## Fine grid around the MAP",
        "",
        "Each row has its own offset axis, selected to span up to three local",
        "conditional Hessian widths without exceeding ±1 prior width.",
        "",
    ]
    for item in summary["local_geometry"]["fine_slices"]:
        lines += [
            f"### {item['label']}", "",
            "| offset | ΔS |", "| ---: | ---: |",
        ]
        for offset, delta in zip(item["offsets"], item["delta_action"]):
            lines.append(f"| {_number(offset)} | {_number(delta)} |")
        lines.append("")
    lines += ["", "## Pairwise conditional grids", ""]
    for (i, j), values in scan.panels.items():
        lines += [
            f"### {scan.labels[i]} × {scan.labels[j]}",
            "",
            f"Rows: {scan.labels[i]} offset; columns: {scan.labels[j]} offset.",
            "",
            "| offset | " + " | ".join(_number(x) for x in offsets) + " |",
            "| ---: | " + " | ".join("---:" for _ in offsets) + " |",
        ]
        for offset, row in zip(offsets, values):
            lines.append(
                "| " + _number(offset) + " | " + " | ".join(
                    _number(float(x - scan.center_value)) for x in row
                ) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _summary_markdown(summary: dict, *, map_info: dict, chain_info: dict | None) -> str:
    lines = [
        "# Full q normalization evidence scan",
        "",
        f"MAP action S = {_number(summary['center_action'])}; "
        f"gradient norm = {_number(map_info['gradient_norm'])}; "
        f"MAP cache = {map_info['cache_status']}.",
        "",
        "The objective is the sampler's normalized joint action S = −2 log p.",
        "Each normalization amplitude has an N(0,1) prior; ±1 means one prior",
        "width, not one posterior width. All q modes and other amplitudes are",
        "held at MAP in the 1D/2D scans. Large ΔS means a steep conditional",
        "direction; a low ΔS or strong pair interaction suggests flatness/ridges.",
        "A conditional slice alone cannot rule out a curved ridge involving q.",
        "",
        "| amplitude | MAP A | ΔS(−1) | ΔS(+1) | exact MAP S'' | conditional σ | lowest ΔS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(summary["axes"]):
        exact_curvature = summary["local_geometry"]["hessian"][index][index]
        exact_sigma = summary["local_geometry"]["fine_slices"][index][
            "conditional_sigma_from_hessian"
        ]
        lines.append(
            "| " + item["label"] + " | " + " | ".join(
                _number(value) if value is not None else "n/a"
                for value in (
                    item["map_coordinate"], item["delta_minus_one"],
                    item["delta_plus_one"], exact_curvature, exact_sigma,
                    item["lowest_grid_delta"],
                )
            ) + " |"
        )
    lines += [
        "",
        "Exact six-by-six normalization conditional Hessian eigenvalues at MAP: "
        + ", ".join(
            _number(x) for x in summary["local_geometry"]["eigenvalues"]
        ) + ".",
        "",
        "Pair interactions below are ΔS(+1,+1) − ΔS(+1,0) − ΔS(0,+1);",
        "a large negative value means moving together is much cheaper than",
        "the two separate moves. The second column changes the latter sign.",
        "",
        "| pair | lowest ΔS | offset at lowest | interaction ++ | interaction +− |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["pairs"]:
        lines.append(
            f"| {item['labels'][0]} × {item['labels'][1]} | "
            f"{_number(item['lowest_grid_delta'])} | "
            f"{','.join(_number(x) for x in item['lowest_grid_offsets']) if item['lowest_grid_offsets'] is not None else 'n/a'} | "
            f"{_number(item['interaction_plus_plus'])} | "
            f"{_number(item['interaction_plus_minus'])} |"
        )
    lines += [
        "",
        f"Simultaneous six-amplitude ±1 corners: min ΔS = "
        f"{_number(summary['corners']['minimum_delta'])}, "
        f"max ΔS = {_number(summary['corners']['maximum_delta'])}; "
        f"{summary['corners']['below_map_count']} of 64 below MAP.",
        "",
        f"Non-finite grid nodes: {summary['n_invalid']}.",
        "Full numerical grids are in NORMALIZATION_GRID_VALUES.md and scan.npz;",
        "all 64 simultaneous shifts are in corners.csv.",
    ]
    if chain_info is not None:
        lines += ["", "Reference production-chain model fingerprint matched: "
                  + chain_info["model_fingerprint"] + "."]
    return "\n".join(lines) + "\n"


def _check_chain(chain: Path, model, coordinates) -> dict:
    from pixel.infer.q_integration._legacy import _direct_model_fingerprint

    with np.load(chain, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        labels = [str(x) for x in saved["param_labels"][:6]]
    fingerprint = _direct_model_fingerprint(model, coordinates)
    if metadata.get("model_fingerprint") != fingerprint:
        raise ValueError(
            "reconstructed target differs from the reference chain; refusing "
            f"to compare evidence grids ({fingerprint} != "
            f"{metadata.get('model_fingerprint')})"
        )
    return {"model_fingerprint": fingerprint, "normalization_labels": labels}


def _local_geometry(target, center, indices, labels, center_action) -> dict:
    """Use six HVPs for MAP curvature, then inspect fine numerical slices."""
    import jax
    import jax.numpy as jnp

    point = jnp.asarray(center)
    gradient = lambda x: target.action_and_grad(x)[1]
    hvp = jax.jit(lambda direction: jax.jvp(gradient, (point,), (direction,))[1])
    action = jax.jit(target.action)
    hessian = np.empty((len(indices), len(indices)), dtype=float)
    for col, index in enumerate(indices):
        direction = np.zeros_like(center)
        direction[index] = 1.0
        hessian[:, col] = np.asarray(hvp(jnp.asarray(direction)))[list(indices)]
    hessian = 0.5 * (hessian + hessian.T)
    eigvals = np.linalg.eigvalsh(hessian)
    fine_slices = []
    for col, (index, label) in enumerate(zip(indices, labels)):
        curvature = hessian[col, col]
        sigma = float(np.sqrt(2.0 / curvature)) if curvature > 0 else None
        half_width = min(1.0, 3.0 * sigma) if sigma is not None else 1.0
        offsets = np.linspace(-half_width, half_width, 9)
        values = []
        for offset in offsets:
            shifted = np.array(center, copy=True)
            shifted[index] += offset
            values.append(float(action(jnp.asarray(shifted))) - center_action)
        fine_slices.append({
            "label": label,
            "conditional_sigma_from_hessian": sigma,
            "offsets": offsets.tolist(),
            "delta_action": values,
        })
    return {
        "hessian": hessian.tolist(),
        "eigenvalues": eigvals.tolist(),
        "fine_slices": fine_slices,
        "note": "normalization conditional block only; q coordinates fixed",
    }


def run(args) -> None:
    from closure_full_q_integration.variant import activated

    with activated(
        "JAM", "union", output_root=Path(args.out),
        map_cache_root=Path(args.map_cache_root),
    ) as (_package, cfg, fit, _runner):
        _run_active(args, cfg, fit)


def _run_active(args, cfg, fit) -> None:
    import jax
    import jax.numpy as jnp
    from pixel.infer.q_integration._legacy import DirectCoordinates
    from pixel.infer.q_integration.target import QTarget
    from pixel.plotting.evidence_scan import plot_evidence_scan, scan_evidence

    from closure_full_q_integration.map_start import prepare_q_map_start

    if not cfg.KERNEL_CACHE_ROOT.is_dir():
        raise FileNotFoundError(
            f"{cfg.KERNEL_CACHE_ROOT} is absent. Run on the production machine "
            "with its existing kernel cache."
        )
    cached_matrices = sum(1 for _ in cfg.KERNEL_CACHE_ROOT.rglob("*.npz"))
    if cached_matrices < args.min_cached_matrices:
        raise RuntimeError(
            f"only {cached_matrices} kernel cache matrices found under "
            f"{cfg.KERNEL_CACHE_ROOT}; at least {args.min_cached_matrices} "
            "are required by this safety check. PIXEL can generate missing "
            "kernels on a cache miss, so verify the production cache before "
            "lowering --min-cached-matrices."
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Building cached full-scale target and converged DIS t0 covariance", flush=True)
    analysis, _ = fit.build_analysis(args.q, args.mode, use_kernel_cache=True)
    model = analysis.compile()
    t0, _ = fit._iterate_t0(args.q, args.mode, model, use_kernel_cache=True)
    if t0:
        analysis, _ = fit.build_analysis(
            args.q, args.mode, use_kernel_cache=True, t0=t0
        )
        model = analysis.compile()
    # Reject a target/version mismatch before the potentially long MAP solve.
    chain_info = None
    if args.chain:
        reference_coordinates = DirectCoordinates.from_model(
            model, prior_rcond=cfg.Q_INTEGRATION_PRIOR_RCOND
        )
        chain_info = _check_chain(Path(args.chain), model, reference_coordinates)
    coordinates, center, map_info = prepare_q_map_start(
        model,
        prior_rcond=cfg.Q_INTEGRATION_PRIOR_RCOND,
        cache_root=cfg.Q_INTEGRATION_MAP_CACHE_ROOT,
    )
    target = QTarget(coordinates)
    labels = tuple(
        label for label in target.free_labels()
        if label.endswith(" normalization / amp")
    )
    if len(labels) != 6:
        raise ValueError(f"expected six exp normalizations; found {labels}")
    if chain_info is not None and list(labels) != chain_info["normalization_labels"]:
        raise ValueError("reference chain normalization labels do not match")
    indices = tuple(target.free_labels().index(label) for label in labels)
    bounds = [(float(center[i] - 1), float(center[i] + 1)) for i in indices]
    print(f"MAP ready; scanning {len(labels)} amplitudes at {args.points} points", flush=True)
    scan = scan_evidence(
        (target, center), labels, bounds=bounds,
        n_points=args.points, chunk_size=args.chunk_size, verbose=True,
    )
    summary = summarize_scan(scan)
    print("Computing exact MAP normalization Hessian block and fine slices", flush=True)
    summary["local_geometry"] = _local_geometry(
        target, center, indices, labels, scan.center_value
    )
    action = jax.jit(target.action)
    corner_rows = []
    for signs in itertools.product((-1, 1), repeat=len(indices)):
        point = np.array(center, copy=True)
        point[list(indices)] += np.asarray(signs, dtype=float)
        delta = float(action(jnp.asarray(point))) - scan.center_value
        corner_rows.append((signs, delta))
    corner_values = np.asarray([delta for _, delta in corner_rows])
    corner_minimum, corner_minimum_index = _minimum(corner_values)
    corner_maximum = (
        float(np.max(corner_values[np.isfinite(corner_values)]))
        if np.any(np.isfinite(corner_values)) else None
    )
    summary["corners"] = {
        "minimum_delta": corner_minimum,
        "maximum_delta": corner_maximum,
        "below_map_count": int(np.count_nonzero(corner_values < -1e-4)),
        "minimum_signs": list(corner_rows[corner_minimum_index[0]][0])
        if corner_minimum_index is not None else None,
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(corner_values))),
    }
    summary["map"] = map_info
    summary["reference_chain"] = chain_info
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "REPORT.md").write_text(
        _summary_markdown(summary, map_info=map_info, chain_info=chain_info)
    )
    (out / "NORMALIZATION_GRID_VALUES.md").write_text(_grid_markdown(scan, summary))
    with (out / "corners.csv").open("w") as stream:
        stream.write(",".join(f"amp_{i+1}" for i in range(len(indices))) + ",delta_action\n")
        for signs, delta in corner_rows:
            stream.write(",".join(map(str, signs)) + f",{delta:.17g}\n")
    np.savez_compressed(
        out / "scan.npz", center=center, center_action=scan.center_value,
        labels=np.asarray(labels), indices=np.asarray(indices),
        axes=np.stack(scan.axes), diagonal=np.stack(scan.diagonal),
        pair_indices=np.asarray(list(scan.panels), dtype=int),
        pair_grids=np.stack(list(scan.panels.values())),
    )
    plot_evidence_scan(scan, path=out / "evidence_scan", close=True)
    print(f"Numerical tables, arrays, and PNG/PDF ready in {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", default="2")
    parser.add_argument("--mode", default="exp", choices=("exp",))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--map-cache-root", type=Path,
        default=Path("campaign/JAM_Q2_exp_prod/_q_map_cache"),
        help="shared MAP cache; independent of report output",
    )
    parser.add_argument("--chain", type=Path, help="reference chain for target-fingerprint check")
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--min-cached-matrices", type=int, default=100)
    args = parser.parse_args()
    if args.points < 3 or args.points % 2 != 1:
        parser.error("--points must be an odd integer >= 3")
    if args.chunk_size < 1 or args.min_cached_matrices < 0:
        parser.error("--chunk-size must be positive and --min-cached-matrices nonnegative")
    run(args)


if __name__ == "__main__":
    main()
