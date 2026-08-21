#!/usr/bin/env python3
"""Build the per-prior cross-Q ``comparison/`` trees a campaign does not write itself.

``run_closure.main()`` aggregates across Q members at the end of a suite run
(``run_closure.py:434-465``); ``run_campaign`` drives ``run_one`` cell by cell and
never reaches that block, so each per-prior result tree has its per-Q outputs but no
``comparison/``. This fills it in from the summaries already on disk.

``cross_Q_coverage`` is rebuilt from the saved ``summary_<mode>.json`` files.
``ratio_grid_<mode>`` is rebuilt from the per-cell posterior ``.npz`` that
``run_campaign.save_posterior`` now writes -- it needs ``(q_key, marginal, truth)``
triples, which used to exist only in memory and made the ratio grids unbuildable
without a re-fit. A campaign run before posteriors were saved (or with
``--no-posteriors``) simply skips that half and says so.

Usage:
    python -m campaign.make_comparisons --name rcond_1e-12
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .run_campaign import load_posterior

CAMPAIGN_ROOT = Path(__file__).resolve().parent
PRIORS = ("const_logrbf", "beta_envelope")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    base = CAMPAIGN_ROOT / args.name
    rows = [json.loads(p.read_text()) for p in (base / "rows").glob("*.json")]
    suites = sorted({r["suite"] for r in rows})

    for suite in suites:
        runner = importlib.import_module(f"{suite}.run_closure")
        for prior in PRIORS:
            tree = base / prior / suite
            if not tree.exists():
                continue
            summaries = []
            for path in sorted(tree.glob("truthQ_*/summary_*.json")):
                summaries.append(json.loads(path.read_text()))
            if not summaries:
                continue

            failures = [
                {"q_key": r["q"], "mode": r["mode"], "exception": r.get("exception"),
                 "message": r.get("message"), "runtime_seconds": r.get("elapsed_s")}
                for r in rows
                if r["suite"] == suite and r["prior"] == prior and r["status"] != "ok"
            ]
            comp = tree / "comparison"
            comp.mkdir(parents=True, exist_ok=True)
            (comp / "summaries.json").write_text(json.dumps(summaries, indent=2))
            (comp / "failures.json").write_text(json.dumps(failures, indent=2))
            (comp / "run_manifest.json").write_text(json.dumps({
                "campaign": args.name,
                "suite": suite,
                "prior": prior,
                "rcond": next((r.get("rcond") for r in rows if r["suite"] == suite), None),
                "requested": len(summaries) + len(failures),
                "succeeded": len(summaries),
                "failed": len(failures),
                "failures": failures,
            }, indent=2))

            try:
                runner.plot_comparison(summaries, comp / "cross_Q_coverage.png")
                status = "cross_Q_coverage"
            except Exception as exc:
                status = f"cross_Q FAILED: {exc}"

            # Ratio grids, one per mode, from the saved posteriors.
            plot_ratios = importlib.import_module(f"{suite}.plot_ratios")
            modes = sorted({r["mode"] for r in rows if r["suite"] == suite})
            n_grid = 0
            for mode in modes:
                runs = []
                for r in sorted(rows, key=lambda r: r["q"]):
                    if (r["suite"], r["prior"], r["mode"]) != (suite, prior, mode):
                        continue
                    npz = r.get("posterior_npz")
                    if not npz:
                        continue
                    path = CAMPAIGN_ROOT / npz
                    if path.exists():
                        marginal, truth = load_posterior(path)
                        runs.append((r["q"], marginal, truth))
                if runs:
                    try:
                        plot_ratios.plot_mode_from_runs(
                            mode, runs, comp / f"ratio_grid_{mode}.png"
                        )
                        n_grid += 1
                    except Exception as exc:
                        print(f"   [ratio_grid_{mode} failed: {exc}]")
            status += f", {n_grid} ratio grid(s)"
            print(f"{suite:28s} {prior:14s} {len(summaries):3d} summaries, "
                  f"{len(failures)} failed -> {status}")
    print("\nRatio grids come from the saved posteriors; a run made before those "
          "were persisted reports 0 of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
