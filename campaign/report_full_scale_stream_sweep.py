"""Report full-scale stream diagnostics from saved PIXEL inference results.

The only non-PIXEL quantities are elapsed times, recorded by the closure
worker around the inference call and entire run. No sampler statistic is
inferred from console output or approximated from block medians.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}g}"


def _max_abs(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else None


def load_stream(path: Path) -> tuple[dict, object]:
    """Load the exact PIXEL chain and its separate worker timing record."""
    from pixel.infer.samplers.hmc import HMCSamples
    from pixel.infer.samplers.nuts import NUTSSamples

    record = json.loads((path / "stream.json").read_text())
    if record.get("status") != "complete":
        raise ValueError(f"stream is not complete: {path}")
    kind = record.get("chain_kind")
    if kind == "nuts":
        chain = NUTSSamples.load_npz(path / "chain.npz")
    elif kind == "hmc":
        chain = HMCSamples.load_npz(path / "chain.npz")
    else:
        raise ValueError(f"missing or unknown PIXEL chain kind in {path}")
    if chain.n_samples != int(record["samples"]):
        raise ValueError(f"retained-draw mismatch in {path}")
    if not (path / "inference_summary.json").is_file():
        raise ValueError(f"missing PIXEL inference summary in {path}")
    return record, chain


def stream_diagnostics(record: dict, chain) -> dict:
    """Use PIXEL's summary and trajectory arrays, without log parsing."""
    from pixel.infer.samplers.hmc import HMCSamples
    from pixel.infer.samplers.nuts import NUTSSamples

    measured = chain.summary()
    shared = {
        "draws": int(chain.n_samples),
        "warmup": int(chain.n_warmup),
        "tau_int_max": _finite(measured["tau_int_max"]),
        "slowest_parameter": measured["slowest_param"],
        "ess_min": _finite(measured["ess_min"]),
        "ess_fraction_min": _finite(measured["ess_fraction_min"]),
        "tau_reliable": bool(measured["tau_reliable"]),
        "step_size": float(chain.step_size),
        "metric_structure": chain.metadata.get("metric_structure"),
        "metric_initialization": chain.metadata.get(
            "mass_initialization", chain.metadata.get("metric_initialization")
        ),
        "metric_fallback_reason": chain.metadata.get(
            "mass_fallback_reason", chain.metadata.get("metric_fallback_reason")
        ),
        "hessian_mass_mode": chain.metadata.get("hessian_mass_mode"),
        "dense_metric_realized": (
            chain.metadata.get("metric_structure") == "dense"
            and chain.metadata.get("mass_fallback_reason",
                                   chain.metadata.get("metric_fallback_reason")) is None
        ),
        "wall_seconds": _finite(record.get("wall_seconds")),
        "sampler_wall_seconds": _finite(record.get("sampler_wall_seconds")),
    }
    if isinstance(chain, NUTSSamples):
        return {
            **shared, "kind": "nuts",
            "acceptance_statistic": _finite(chain.mean_accept_prob),
            "acceptance_rate": None,
            "max_abs_delta_h": _max_abs(chain.energy_error),
            "average_depth": _finite(np.mean(chain.tree_depth)),
            "num_divergence": int(chain.n_divergent),
            "num_invalid": None,
            "steps_per_trajectory": _finite(np.mean(chain.n_leapfrog)),
        }
    if isinstance(chain, HMCSamples):
        return {
            **shared, "kind": "hmc",
            "acceptance_statistic": _finite(np.mean(chain.accept_prob)),
            "acceptance_rate": _finite(chain.acceptance_rate),
            "max_abs_delta_h": _max_abs(chain.delta_hamiltonian),
            "average_depth": None,
            "num_divergence": None,
            "num_invalid": int(chain.n_invalid),
            "steps_per_trajectory": int(chain.n_leapfrog),
        }
    raise TypeError(f"unsupported PIXEL chain: {type(chain).__name__}")


def render_cell(cell: dict) -> tuple[Path, dict]:
    """Write a cell report only when all independently saved streams finished."""
    from pixel.infer.diagnostics.multichain import split_r_hat

    records_and_chains = [load_stream(Path(s["output"])) for s in cell["streams"]]
    rows = [stream_diagnostics(record, chain)
            for record, chain in records_and_chains]
    chains = [chain for _record, chain in records_and_chains]
    labels = [tuple(chain.param_labels) for chain in chains]
    if len(set(labels)) != 1:
        raise ValueError("PIXEL parameter labels differ between streams")
    kinds = {row["kind"] for row in rows}
    if len(kinds) != 1:
        raise ValueError("PIXEL sampler kinds differ between streams")
    rhat = None
    worst_label = None
    if len(chains) >= 2:
        estimates = split_r_hat([chain.samples for chain in chains])
        if estimates.size and np.all(np.isfinite(estimates)):
            index = int(np.argmax(estimates))
            rhat = float(estimates[index])
            worst_label = labels[0][index] if labels[0] else str(index)
    result = {
        "cell": {key: cell[key] for key in (
            "parameterization", "truth", "data", "scale", "mode",
            "stream_count",
        )},
        "streams": rows,
        "pixel_inference_summaries": [
            json.loads((Path(s["output"]) / "inference_summary.json").read_text())
            for s in cell["streams"]
        ],
        "split_rhat_max": rhat,
        "split_rhat_parameter": worst_label,
        "provenance": {
            "tau_ess": "PIXEL chain.summary() (Wolff Gamma method)",
            "rhat": "PIXEL split_r_hat() on saved chain.samples",
            "trajectory": "PIXEL NUTSSamples/HMCSamples trajectory arrays",
            "metric": "PIXEL chain.metadata",
            "inference_summary": "PIXEL inference-result summary()",
            "wall_time": "closure worker wall clock and fit inference timer",
        },
    }
    destination = Path(cell["streams"][0]["output"]).parent
    lines = [
        f"# {cell['parameterization']} {cell['truth']}/{cell['data']} "
        f"Q={cell['scale']} {cell['mode']} — {cell['stream_count']} stream(s)",
        "",
        "| stream | draws | warmup | max τ_int | min ESS | min ESS fraction | "
        "accept stat | accept rate | max abs ΔH | step size | mean depth | "
        "divergences | invalid | sampler wall (s) | total wall (s) | steps/trajectory |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['draws']} | {row['warmup']} | "
            f"{_fmt(row['tau_int_max'])} | {_fmt(row['ess_min'])} | "
            f"{_fmt(row['ess_fraction_min'])} | "
            f"{_fmt(row['acceptance_statistic'])} | "
            f"{_fmt(row['acceptance_rate'])} | "
            f"{_fmt(row['max_abs_delta_h'])} | {_fmt(row['step_size'])} | "
            f"{_fmt(row['average_depth'])} | "
            f"{_fmt(row['num_divergence'])} | {_fmt(row['num_invalid'])} | "
            f"{_fmt(row['sampler_wall_seconds'])} | "
            f"{_fmt(row['wall_seconds'])} | "
            f"{_fmt(row['steps_per_trajectory'])} |"
        )
    lines.extend([
        "",
        f"PIXEL split R-hat (worst parameter): {_fmt(rhat, 4)}"
        + (f" ({worst_label})" if worst_label else " (needs at least two streams)"),
        "",
        "τ_int and ESS use PIXEL's Wolff Gamma method; ESS fraction is "
        "1/(2τ_int). NUTS accept stat is the mean transition acceptance "
        "probability, not an accepted/rejected rate. HMC accept rate is the "
        "fraction of accepted proposals. HMC has no tree depth or NUTS "
        "divergence flag; its invalid count is reported separately.",
        "",
        "All sampler values come from the saved PIXEL chain and its metadata. "
        "Sampler and total wall times come from closure timing records. "
        "A dense *request* is not proof of a dense metric: inspect the recorded "
        "metric structure and fallback reason below.",
        "Dense metric realized in every stream: "
        + ("yes" if all(row["dense_metric_realized"] for row in rows) else "NO"),
        "",
        "Full PIXEL inference-object diagnostics (including contour/phase "
        "diagnostics when applicable): " + ", ".join(
            f"[stream {index}](stream_{index:02d}/inference_summary.json)"
            for index in range(1, len(rows) + 1)
        ) + ".",
        "",
        "| stream | actual metric | initialization | fallback reason |",
        "|---:|---|---|---|",
    ])
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['metric_structure'] or '—'} | "
            f"{row['metric_initialization'] or '—'} | "
            f"{row['metric_fallback_reason'] or '—'} |"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "REPORT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    report = destination / "REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    return report, result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads((args.root / "manifest.json").read_text())
    for cell in manifest["cells"]:
        if any(stream.get("exit_code") != 0 for stream in cell["streams"]):
            continue
        path, _result = render_cell(cell)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
