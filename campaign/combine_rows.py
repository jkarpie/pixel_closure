#!/usr/bin/env python3
"""Combine campaign rows across truths, Q members, modes and BOTH prior forms.

Reads the flat ``rows/`` directory a campaign writes and produces the cross-cutting
view that no single per-prior result tree can show: the two priors side by side, over
every scale, for both truths.

Row schema note. The old ``campaign/full_2026-08-16/rows/`` carried a single derived
``worst_constraint_residual`` normalized by a per-constraint ``floor``. For the sum
rules that floor is ``|target|`` and reproduces exactly; for ``cons_endpoint_*`` the
old floor was a near-cancellation scale that did not survive the move, so newer rows
store raw ``target``/``realized`` plus ``abs_residual`` instead of guessing it. This
script therefore reports the **absolute** residual, which is defined identically in
both generations and is the quantity that compares against ``CONSTRAINT_*_SIGMA``.
Endpoint and sum-rule constraints are reported separately, because they are held to
the same 1e-4 SD but are not the same kind of number.

Usage:
    python -m campaign.combine_rows --name small_all_2026-08-17
    python -m campaign.combine_rows --name small_all_2026-08-17 --compare full_2026-08-16
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parent
QS = ["mc", "1", "2", "3", "4", "5"]
MODES = ["lattice", "dis", "dy", "synthetic_z_plumbing", "exp", "both"]
PRIORS = ["const_logrbf", "beta_envelope"]
SIGMA = 1.0e-4


def load(name: str) -> list[dict]:
    rows_dir = CAMPAIGN_ROOT / name / "rows"
    if not rows_dir.exists():
        rows_dir = CAMPAIGN_ROOT / name / "rows"
    return [json.loads(p.read_text()) for p in sorted(rows_dir.glob("*.json"))]


def _split_residuals(row: dict) -> tuple[float | None, float | None]:
    """Worst absolute residual over (endpoint-like, sum-rule-like) constraints.

    Falls back to the older rows' representation when ``abs_residual`` is absent, so
    a pre-move campaign and a post-move one can be tabulated together.
    """
    constraints = row.get("constraints") or {}
    endpoint, sumrule = [], []
    for name, entry in constraints.items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("abs_residual")
        if value is None and entry.get("residual") is not None:
            # Old schema: residual is relative to `floor`; recover the absolute miss.
            value = entry["residual"] * abs(entry.get("floor", 1.0))
        if value is None:
            continue
        (sumrule if name.startswith(("cons_norm", "cons_momentum")) else endpoint).append(value)
    return (max(endpoint) if endpoint else None, max(sumrule) if sumrule else None)


def _fmt(value: float | None) -> str:
    return "     --  " if value is None else f"{value:9.2e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--compare", default=None, help="a second campaign to tabulate beside it")
    args = ap.parse_args()

    campaigns = {args.name: load(args.name)}
    if args.compare:
        campaigns[args.compare] = load(args.compare)

    for name, rows in campaigns.items():
        ok = [r for r in rows if r.get("status") == "ok"]
        bad = [r for r in rows if r.get("status") != "ok"]
        print(f"\n{'='*78}\n{name}: {len(rows)} rows, {len(ok)} ok, {len(bad)} failed")

        for row in bad:
            print(f"   FAIL  {row['suite']:26s} Q={row['q']:<3s} {row['mode']:22s} "
                  f"{row['prior']:14s} {row.get('exception','?')}")

        # -- survival and residual, by suite x prior -------------------------
        print(f"\n{'suite':28s}{'prior':16s}{'ran':>6s}{'endpoint worst':>16s}{'sumrule worst':>15s}")
        print("-" * 78)
        for suite in sorted({r["suite"] for r in rows}):
            for prior in PRIORS:
                cells = [r for r in rows if r["suite"] == suite and r["prior"] == prior]
                good = [r for r in cells if r.get("status") == "ok"]
                if not cells:
                    continue
                ep = [v for v, _ in map(_split_residuals, good) if v is not None]
                sr = [v for _, v in map(_split_residuals, good) if v is not None]
                print(f"{suite:28s}{prior:16s}{len(good):>3d}/{len(cells):<2d}"
                      f"{_fmt(max(ep) if ep else None):>16s}{_fmt(max(sr) if sr else None):>15s}")

        # -- how many cells clear the 1e-4 bar -------------------------------
        print(f"\ncells whose worst ABSOLUTE constraint residual exceeds "
              f"the {SIGMA:.0e} constraint SD:")
        for prior in PRIORS:
            good = [r for r in ok if r["prior"] == prior]
            if not good:
                continue
            over = [r for r in good
                    if max((v for v in _split_residuals(r) if v is not None), default=0.0) > SIGMA]
            print(f"   {prior:16s} {len(over):>3d}/{len(good):<3d}", end="")
            if over:
                by_suite = defaultdict(int)
                for r in over:
                    by_suite[r["suite"]] += 1
                print("   " + ", ".join(f"{k.split('_')[1]}:{v}" for k, v in sorted(by_suite.items())))
            else:
                print()

        # -- median cost per mode -------------------------------------------
        cost = defaultdict(list)
        for row in ok:
            if row.get("elapsed_s"):
                cost[row["mode"]].append(row["elapsed_s"])
        if cost:
            print(f"\n{'mode':24s}{'n':>4s}{'median s':>10s}{'total min':>11s}")
            for mode in MODES:
                if mode in cost:
                    v = cost[mode]
                    print(f"{mode:24s}{len(v):>4d}{statistics.median(v):>10.1f}"
                          f"{sum(v)/60:>11.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
