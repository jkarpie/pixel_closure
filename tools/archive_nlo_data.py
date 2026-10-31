#!/usr/bin/env python3
"""Move existing closure data/caches aside before generating NNLO inputs.

Run once at the NLO-to-NNLO transition, before generating any NNLO data. This
archives the current files; it does not infer their order from their contents.
Uses only the Python standard library, so it also runs on remote checkouts.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="pixel_closure checkout (defaults to this script's checkout)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print moves only")
    args = parser.parse_args()
    root = args.root.resolve()

    # These are source inputs and reusable build tools, independent of the
    # perturbative order. Keep the active data directory and its placeholder.
    keep = {".gitkeep", "_tools", "lhapdf"}
    moves = []
    for truth in ("JAM", "NNPDF"):
        for suffix in ("", "_JAM_data", "_NNPDF_data", "_small"):
            suite = root / f"closure_{truth}_truth{suffix}"
            data = suite / "data"
            if not data.is_dir():
                continue
            if data.is_symlink():
                parser.error(f"refusing to archive a symlinked data directory: {data}")
            for source in sorted(data.iterdir()):
                if source.name not in keep:
                    moves.append((source, suite / "NLO_RUNS" / "data" / source.name))

    # Preflight the entire move before touching anything. A repeated invocation
    # with no new data is a no-op; later inputs cannot replace the NLO archive.
    for source, target in moves:
        if target.exists() or target.is_symlink():
            parser.error(f"archive already exists; refusing to overwrite: {target}")

    for source, target in moves:
        print(f"{source.relative_to(root)} -> {target.relative_to(root)}", flush=True)
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)

    action = "Would archive" if args.dry_run else "Archived"
    print(f"{action} {len(moves)} entries across {len({s.parent for s, _ in moves})} suites.")


if __name__ == "__main__":
    main()
