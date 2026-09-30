"""Freeze a closure cell's ``EvidenceBlocks`` so the H-S benchmark can run on it.

`EXT:pixel/plans/hs_saddle_contour_algorithms.md` benchmarks saddle solvers and integration
contours.  Its C0-C2 tiers are analytic actions, PIXEL's own small fixtures and a twelve-row
Drell-Yan toy; none of them is a *closure* block, and the cells this whole investigation is
about are exactly the ones no fixture reproduces.  This writes those cells out.

**What is frozen and why it is enough.**  A benchmark case is an ``EvidenceBlocks`` at one
parameter point -- precisely the object ``find_saddle`` is handed -- so freezing it removes the
model build (minutes, and a closure checkout) from every benchmark run, while changing nothing
about the arithmetic the arms see.  The reference point is the same one both saddle-contour
samplers use: fitted normalizations zeroed, ``f`` at its stored value, t0 applied if the suite
iterates it.

**What can go stale, and how that is detected.**  A frozen block is a photograph of a kernel,
a prior and a t0 iteration.  The manifest therefore carries
``contour._reference_contour_fingerprint`` for the exported blocks, so a case exported against
one kernel and loaded against another is a mismatch a loader can see rather than a silent
difference in the numbers.  Per-array sha256 digests catch corruption or a partial write.

**Size.**  ``bilinear_Y`` is the problem: a dense ``(n_bilin, n_grid, n_grid)`` tensor on a
closure grid is hundreds of megabytes and must never be committed.  Where the model carries the
*factorized* form -- one hard tensor per parton channel plus a mixing matrix -- that is what is
written, and it is small.  Where it is dense, it is still written, and the manifest says so;
the intended workflow is that the ``.npz`` files live in an uncommitted scratch directory and
only the manifest is committed, with the regeneration command recorded in it.

Usage::

    ../pixel/venv/bin/python -m campaign.export_hs_cases \\
        --suite closure_JAM_truth_small --q 2 --mode dy \\
        --cell-json campaign/.../cell_json/c_config__c82a403ad1a9.json \\
        --out ../pixel/benchmarks/results/_scratch_hs_cases
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import prior_cells as pc
from .contour_profiles import pipeline_model

#: Array-valued fields of ``EvidenceBlocks`` that are written verbatim.  ``bilinear_Y`` is
#: handled separately because it is either a dense tensor or a tuple of factor blocks, and
#: the tuple fields are refused rather than half-exported.
PLAIN_FIELDS = (
    "M", "C", "B", "g", "K", "param_pred",
    "bilinear_M", "bilinear_C", "hubbard_f",
    "bilinear_active_indices", "bilinear_scale", "bilinear_offset", "bilinear_B",
)

#: Block kinds this exporter does not know how to freeze.  They are refused loudly: a case
#: silently missing one of them would be a *different* target wearing the same name.
REFUSED_FIELDS = ("nonlinear", "bilinear_ratio", "mixed_ratio")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:32]


def _record(store: dict, meta: dict, name: str, value) -> None:
    array = np.asarray(value)
    store[name] = array
    meta[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                  "sha256": _digest(array), "bytes": int(array.nbytes)}


def collect_arrays(blocks) -> tuple[dict, dict]:
    """``(arrays, metadata)`` for one ``EvidenceBlocks``.

    Refuses the block kinds it cannot represent instead of dropping them, and records whether
    ``bilinear_Y`` came out dense or factorized, because that decides how large the result is
    and whether it can be committed.
    """
    for field in REFUSED_FIELDS:
        if getattr(blocks, field, ()):
            raise SystemExit(
                f"export refuses {field!r}: this cell carries block kinds the H-S benchmark "
                "does not model, and exporting the rest would quietly change the target")

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for field in PLAIN_FIELDS:
        value = getattr(blocks, field, None)
        if value is not None:
            _record(arrays, meta, field, value)

    y = blocks.bilinear_Y
    if y is None:
        layout = "none"
    elif isinstance(y, (tuple, list)):
        layout = "factorized"
        for index, block in enumerate(y):
            _record(arrays, meta, f"bilinear_Y.{index}.tensor", block.tensor)
            _record(arrays, meta, f"bilinear_Y.{index}.mixing", block.mixing)
            meta[f"bilinear_Y.{index}.rows"] = {"row_start": int(block.row_start),
                                                "row_stop": int(block.row_stop)}
        meta["bilinear_Y.n_blocks"] = {"count": len(y)}
    else:
        layout = "dense"
        _record(arrays, meta, "bilinear_Y", y)
    meta["bilinear_Y.layout"] = {"layout": layout}
    return arrays, meta


def export_case(*, suite: str, q: str, mode: str, cell_json: Path, out_dir: Path,
                name: str | None = None) -> dict:
    """Build one cell, freeze its blocks, and return the manifest entry."""
    from pixel.infer.contour.saddle import _reference_contour_fingerprint

    cfg = importlib.import_module(f"{suite}.config")
    cell = pc.validate(json.loads(Path(cell_json).read_text()))
    started = time.perf_counter()
    with pc.patched_prior(cfg, cell):
        model = pipeline_model(suite, q, mode)
    # The reference point both saddle-contour samplers use: fitted normalizations zeroed.
    import jax.numpy as jnp

    partition = model.contour_partition()
    vector = jnp.asarray(model.free_vector()).reshape(-1)
    ordinary, _normalization, f0 = model.split_contour_vector(vector)
    reference = model.merge_contour_vector(
        ordinary, jnp.zeros(partition.n_normalization, dtype=jnp.float64), f0)
    blocks = model.blocks(reference)

    arrays, meta = collect_arrays(blocks)
    tol = float(cfg.CONTOUR_SADDLE_TOL)
    fingerprint = _reference_contour_fingerprint(
        blocks, rcond=model.rcond, saddle_tol=tol)
    case_name = name or f"{suite}-{q}-{mode}-{cell['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case_name}.npz"
    np.savez_compressed(path, **arrays)

    return {
        "name": case_name,
        "file": path.name,
        "suite": suite,
        "q": q,
        "mode": mode,
        "cell": cell["name"],
        "cell_json": str(cell_json),
        "prior_form": cell["prior_form"],
        "n_bilinear": int(np.asarray(blocks.bilinear_M).size),
        "n_grid_gp": int(np.asarray(blocks.g).size),
        "rcond": float(model.rcond),
        "saddle_tol": tol,
        "reference_contour_fingerprint": str(fingerprint),
        "arrays": meta,
        "total_bytes": int(sum(entry.get("bytes", 0) for entry in meta.values()
                               if isinstance(entry, dict))),
        "build_seconds": float(time.perf_counter() - started),
        "regenerate": (
            f"cd pixel_closure && ../pixel/venv/bin/python -m campaign.export_hs_cases "
            f"--suite {suite} --q {q} --mode {mode} --cell-json {cell_json} --out {out_dir}"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--q", default="2",
                        help="Q key *without* the truthQ_ prefix the suite adds: 2, 3, mc")
    parser.add_argument("--mode", required=True,
                        help="dy | exp | both | dis — the cell's dataset mode")
    parser.add_argument("--cell-json", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    entry = export_case(suite=args.suite, q=args.q, mode=args.mode,
                        cell_json=args.cell_json, out_dir=args.out, name=args.name)
    manifest_path = args.out / "manifest.json"
    manifest = {"cases": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("cases", {})
    manifest["cases"][entry["name"]] = entry
    manifest["schema"] = 1
    temporary = manifest_path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(temporary, manifest_path)
    print(f"wrote {args.out / entry['file']} "
          f"({entry['total_bytes'] / 1e6:.1f} MB, n_bilinear={entry['n_bilinear']}, "
          f"layout={entry['arrays']['bilinear_Y.layout']['layout']}) "
          f"in {entry['build_seconds']:.0f}s")
    print(f"manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
