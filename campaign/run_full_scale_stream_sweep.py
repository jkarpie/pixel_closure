"""Launch isolated full-scale closure streams on a fixed external CPU budget.

Cells are serial; within each cell, the requested number of independent
streams run simultaneously.  Use ``taskset -c`` on Linux to pin the *whole*
sweep to the same cores for every stream-count arm.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from closure_full_q_integration.variant import SOURCE_SUITES, activated
from closure_full_q_integration.map_start import default_map_cache_root


REPO = Path(__file__).resolve().parents[1]
TRUTHS = ("JAM", "NNPDF")
DATA = ("union", "JAM", "NNPDF")
SCALES = ("mc", "1", "2", "3", "4", "5")
MODES = ("lattice", "dis", "dy", "exp", "both")
PARAMETERIZATIONS = ("hs", "q")
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "PIXEL_ASSEMBLY_WORKERS", "PIXEL_KERNEL_WORKERS",
    "PIXEL_POSTERIOR_WORKERS", "PIXEL_DATASET_WORKERS",
)


def csv_choices(raw: str, choices: tuple[str, ...], label: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(part.strip() for part in raw.split(",")))
    if not values or any(value not in choices for value in values):
        raise ValueError(f"{label} must be a comma list drawn from {','.join(choices)}")
    return values


def stream_counts(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(part.strip()) for part in raw.split(",")))
    except ValueError as exc:
        raise ValueError("--stream-counts requires positive comma-separated integers") from exc
    if not values or any(value < 1 for value in values):
        raise ValueError("--stream-counts requires positive comma-separated integers")
    return values


def stream_seed(base: int, cell: tuple[str, ...], stream: int) -> int:
    """Keep a cell's first N seeds identical across stream-count arms."""
    digest = hashlib.sha256("/".join(cell).encode()).digest()
    return (base + int.from_bytes(digest[:4], "big") + stream) % (2**32)


def plan(args) -> list[dict]:
    cells = (
        (parameterization, truth, data, scale, mode)
        for parameterization in csv_choices(
            args.parameterizations, PARAMETERIZATIONS, "--parameterizations"
        )
        for truth in csv_choices(args.truths, TRUTHS, "--truths")
        for data in csv_choices(args.data, DATA, "--data")
        for scale in csv_choices(args.scales, SCALES, "--scales")
        for mode in csv_choices(args.modes, MODES, "--modes")
    )
    selected_cells = tuple(cells)
    return [
        {
            "parameterization": cell[0], "truth": cell[1], "data": cell[2],
            "scale": cell[3], "mode": cell[4], "stream_count": count,
            "map_cache_root": str(
                Path(getattr(args, "map_cache_root", None) or default_map_cache_root()).resolve()
            ),
            "streams": [
                {
                    "number": number,
                    "seed": stream_seed(args.seed_base, cell, number),
                    "output": str(
                        args.out / f"streams_{count}" / cell[0] /
                        cell[1].lower() / cell[2].lower() /
                        f"truthQ_{cell[3]}" / cell[4] /
                        f"stream_{number:02d}"
                    ),
                }
                for number in range(1, count + 1)
            ],
        }
        for count in stream_counts(args.stream_counts)
        for cell in selected_cells
    ]


def worker_command(python: str, cell: dict, stream: dict, samples: int,
                   warmup: int | None = None) -> list[str]:
    command = [
        python, "-m", "campaign.run_full_scale_stream_sweep", "worker",
        "--parameterization", cell["parameterization"],
        "--truth", cell["truth"], "--data", cell["data"],
        "--scale", cell["scale"], "--mode", cell["mode"],
        "--samples", str(samples), "--seed", str(stream["seed"]),
        "--output", stream["output"],
    ]
    if cell["parameterization"] == "q":
        command.extend(("--map-cache-root", cell["map_cache_root"]))
    if warmup is not None:
        command.extend(("--warmup", str(warmup)))
    return command


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _json_safe(value):
    """Keep every PIXEL summary field, including NumPy/complex diagnostics."""
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": _json_safe(value.real), "imag": _json_safe(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


@contextmanager
def hs_activated(truth: str, data: str, output: Path, samples: int, seed: int):
    package = SOURCE_SUITES[(truth, data)]
    cfg = importlib.import_module(f"{package}.config")
    runner = importlib.import_module(f"{package}.run_closure")
    overrides = {
        "RESULTS_ROOT": output / "results",
        "INFERENCE_CACHE_ROOT": output / "inference_cache",
        "MCMC_SAMPLES": samples,
        "MCMC_SEED": seed,
    }
    old = {key: getattr(cfg, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(cfg, key, value)
        yield runner
    finally:
        for key, value in old.items():
            setattr(cfg, key, value)


def run_worker(args) -> int:
    if args.warmup is not None and args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.warmup is not None and args.parameterization != "q":
        raise ValueError("--warmup is currently supported only for q-integration")
    output = args.output.resolve()
    (output / "inference_cache").mkdir(parents=True, exist_ok=True)
    metadata = {
        "parameterization": args.parameterization, "truth": args.truth,
        "data": args.data, "scale": args.scale, "mode": args.mode,
        "samples": args.samples, "seed": args.seed,
        "warmup_requested": args.warmup,
        "status": "running", "started_at": time.time(),
    }
    write_json(output / "stream.json", metadata)

    def save_inference(result) -> None:
        """Persist PIXEL's actual trajectory result for post-run diagnostics."""
        from pixel.infer.samplers.hmc import HMCSamples
        from pixel.infer.samplers.nuts import NUTSSamples

        samples = result["samples"]
        chain = getattr(samples, "chain", samples)
        if not isinstance(chain, (HMCSamples, NUTSSamples)):
            raise TypeError(
                f"cannot save trajectory diagnostics from {type(chain).__name__}"
            )
        if args.warmup is not None and chain.n_warmup != args.warmup:
            raise RuntimeError(
                f"requested {args.warmup} warmup trajectories but PIXEL saved "
                f"{chain.n_warmup}"
            )
        chain.save_npz(output / "chain.npz")
        write_json(output / "inference_summary.json", _json_safe(samples.summary()))
        metadata["chain_kind"] = "nuts" if isinstance(chain, NUTSSamples) else "hmc"
        metadata["sampler_wall_seconds"] = float(
            result["timing_seconds"]["inference"]
        )

    try:
        if args.parameterization == "q":
            with activated(
                args.truth, args.data, output_root=output,
                map_cache_root=getattr(args, "map_cache_root", None),
            ) as (
                _package, cfg, _fit, runner,
            ):
                cfg.MCMC_SAMPLES = args.samples
                cfg.MCMC_SEED = args.seed
                if args.warmup is not None:
                    cfg.Q_INTEGRATION_WARMUP = args.warmup
                runner.run_one(args.scale, args.mode, make_plots=False,
                               artifact_hook=save_inference)
        else:
            with hs_activated(args.truth, args.data, output, args.samples,
                              args.seed) as runner:
                runner.run_one(args.scale, args.mode, make_plots=False,
                               artifact_hook=save_inference)
        if not (output / "chain.npz").is_file():
            raise RuntimeError("source runner did not provide a PIXEL chain artifact")
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        metadata["status"] = "complete"
        return 0
    finally:
        metadata["wall_seconds"] = time.time() - metadata["started_at"]
        write_json(output / "stream.json", metadata)


def run_sweep(args) -> int:
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.warmup is not None and args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    args.out = args.out.resolve()
    cells = plan(args)
    if args.warmup is not None and any(
        cell["parameterization"] != "q" for cell in cells
    ):
        raise ValueError("--warmup is currently supported only for q-integration")
    print(f"{len(cells)} cells; {sum(len(c['streams']) for c in cells)} streams")
    for cell in cells:
        label = (f"{cell['parameterization']} {cell['truth']}/{cell['data']} "
                 f"Q={cell['scale']} {cell['mode']} streams={cell['stream_count']}")
        print(label, flush=True)
        if args.dry_run:
            for stream in cell["streams"]:
                print("  " + " ".join(worker_command(
                    args.python, cell, stream, args.samples, args.warmup
                )))
    if args.dry_run:
        return 0
    output_root = args.out
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory is not empty: {output_root}")
    for cell in cells:
        for stream in cell["streams"]:
            source = (REPO / SOURCE_SUITES[(cell["truth"], cell["data"])] /
                      "data" / f"truthQ_{cell['scale']}" / "truth.json")
            if not source.is_file():
                raise FileNotFoundError(
                    f"missing {source}; prepare data with run_closure.sh first"
                )
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_VARS})
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    # Validate each target once before creating output or launching any worker.
    # A missing/failed MAP never triggers optimization in the sampling process.
    checked = set()
    for cell in cells:
        key = (cell["truth"], cell["data"], cell["scale"], cell["mode"])
        if cell["parameterization"] != "q" or key in checked:
            continue
        command = [
            args.python, "-m", "closure_full_q_integration.prepare_map",
            "--truth", cell["truth"], "--data", cell["data"],
            "--Q", cell["scale"], "--modes", cell["mode"],
            "--cache-root", cell["map_cache_root"], "--check-only",
        ]
        print("[MAP preflight] " + " ".join(command), flush=True)
        if subprocess.run(command, cwd=REPO, env=env, check=False).returncode != 0:
            raise ValueError("q MAP preflight failed; prepare this target's MAP first")
        checked.add(key)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"samples_per_stream": args.samples,
                "warmup_per_stream_requested": args.warmup, "cells": cells,
                "status": "running", "started_at": time.time()}
    write_json(output_root / "manifest.json", manifest)
    failures = 0
    completed = False
    try:
        for index, cell in enumerate(cells, 1):
            active = []
            try:
                for stream in cell["streams"]:
                    output = Path(stream["output"])
                    output.mkdir(parents=True)
                    log = (output / "run.log").open("w")
                    command = worker_command(
                        args.python, cell, stream, args.samples, args.warmup
                    )
                    process = subprocess.Popen(
                        command, cwd=REPO, env=env, stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    active.append((process, log, stream))
                for process, log, stream in active:
                    code = process.wait()
                    log.close()
                    stream["exit_code"] = code
                    failures += code != 0
                    print(f"[{index}/{len(cells)}] {stream['output']}: "
                          f"{'ok' if code == 0 else f'FAILED ({code})'}", flush=True)
            finally:
                for process, log, _stream in active:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                    log.close()
            if all(stream.get("exit_code") == 0 for stream in cell["streams"]):
                try:
                    from .report_full_scale_stream_sweep import render_cell

                    report_path, _report = render_cell(cell)
                    cell["report"] = str(report_path)
                    print(f"[report] {report_path}", flush=True)
                except Exception as exc:
                    cell["report_error"] = f"{type(exc).__name__}: {exc}"
                    failures += 1
                    print(f"[report FAILED] {cell['report_error']}", flush=True)
            write_json(output_root / "manifest.json", manifest)
        completed = True
    finally:
        manifest["wall_seconds"] = time.time() - manifest["started_at"]
        manifest["status"] = (
            "interrupted" if not completed else "failed" if failures else "complete"
        )
        write_json(output_root / "manifest.json", manifest)
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep = subparsers.add_parser("sweep", help="run the comma-list sweep")
    sweep.add_argument("--out", required=True, type=Path)
    sweep.add_argument("--parameterizations", default="q")
    sweep.add_argument("--truths", default="JAM")
    sweep.add_argument("--data", default="union")
    sweep.add_argument("--scales", default="2")
    sweep.add_argument("--modes", default="exp")
    sweep.add_argument("--stream-counts", default="1,2,4")
    sweep.add_argument("--samples", type=int, default=1000)
    sweep.add_argument("--warmup", type=int, default=None,
                       help="q-integration warmup trajectories per stream")
    sweep.add_argument("--seed-base", type=int, default=20261018)
    sweep.add_argument("--python", default=sys.executable)
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--map-cache-root", type=Path, default=None,
                       help="prepared MAP directory; defaults to PIXEL_Q_MAP_CACHE_ROOT or campaign/_q_map_cache")
    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--parameterization", choices=PARAMETERIZATIONS, required=True)
    worker.add_argument("--truth", choices=TRUTHS, required=True)
    worker.add_argument("--data", choices=DATA, required=True)
    worker.add_argument("--scale", choices=SCALES, required=True)
    worker.add_argument("--mode", choices=MODES, required=True)
    worker.add_argument("--samples", type=int, required=True)
    worker.add_argument("--warmup", type=int, default=None)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--map-cache-root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        return run_sweep(args) if args.command == "sweep" else run_worker(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
