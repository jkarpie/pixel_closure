# Audit reports handed over from `pixel`

These thirteen `reports/*.json` were written by the `pixel` test audit
(`plans/test_audit/`) before commit `f8439d6` (2026-08-16) moved the test files they
describe out of that repository and into this one. They are the machine-readable half of
the record; the prose half is the sibling `tests/audit_entries/` directory, which was moved
over at the same time.

Retired from `pixel` on 2026-08-24 at `pixel` rev `1e9fbbf`, as `WI-L8-05` of
`plans/wt_node-audit-recount.md`. `pixel`'s `stats.py` was summing them, so every audit
headline there counted work that no longer lived in that repository.

## What the record carries, measured at handover

| | recorded | **open** |
|---|---:|---:|
| `S1` false assurance | 19 | **0** |
| `S2` mislabelled | 10 | **1** |
| `S3` under-specified | 18 | **0** |
| `S4` hygiene | 10 | **0** |
| proven coverage gaps | 53 | **4** |
| tolerances annotated | 65 | — |

86 tests documented, 112 standalone measurements. `recorded` counts everything ever filed,
closed items included; the **open** column is the backlog. The single open `S2` is in
`test_closure_prior_ties.json`; the four open coverage gaps are in
`test_closure_constraints.json` (1), `test_closure_pseudoitd_matching.json` (1) and
`test_closure_truth_representable.json` (2).

An earlier `pixel` note said these files held "15 of the then-20 open `S1`". That was true
in 2026-08-14 terms and is not true now — those fifteen were closed before the move. Do not
carry the figure forward.

## Two things that came with them

- **`test_momentum_sum_rule.py` was `pixel`'s only NLO conservation-law oracle.**
  `plans/test_audit/pqcd_audit_order.md` there ordered wave 1 around it, "because it is the
  only test in the area whose oracle is a conservation law rather than another
  implementation". That claim now has no test in `pixel`: every remaining check of its
  evolution stack compares against another implementation (fitpack, Pegasus, EKO) or against
  a parametrization both codes share. If the sum rule is to guard `pixel`'s evolution again,
  a test has to be written there — this one guards it here.
- **`test_small_synthetic_z.py`** carries the `saddle_contour_importance_sample` memory
  defect record (quadratic in sample count).

## Tooling

`pixel`'s `plans/test_audit/tools/` (`stats.py`, `assemble.py`, `verify_entries.py`,
`entry_from_report.py`, `promote_stranded.py`) reads a directory holding `reports/` and
`entries/` subdirectories. These two directories are named `audit_reports/` and
`audit_entries/` here and are not laid out that way, so the tools do not run against them
as-is. They are kept as evidence, not as a live audit tree; wiring them into an audit here
means creating that layout, not editing the tools.
