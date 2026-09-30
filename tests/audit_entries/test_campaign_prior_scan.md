### `tests/test_campaign_prior_scan.py`

**Exercises** `campaign/prior_cells.py` (`base_cell`, `const_grid_s1`, `envelope_grid_s1`,
`validate`, `cell_hash`, `patched_prior`) against the real `closure_JAM_truth_small.config`
and `closure_JAM_truth_small.fit.gp_prior`; and from `campaign/prior_scan.py` the gate-message
parser `ess_from_message`, the `--skip-existing` hash guard in `cmd_fit`, and `rank_fits`.
No fit, kernel or sampler runs.

**Claim** A scan cell's fixed prior is what the fit actually builds (amplitude, tied mean,
length, jitter, envelope alpha/beta/sigma, the recomputed amplitude floor, the redirected
results directory), every patched attribute is restored by identity afterwards -- including
when the body raises -- malformed or out-of-range cells are refused, the cell hash separates
any value change and ignores labels, the stage-1 grids are distinct and contain the shipped
priors, gate values are recovered from both sampler failure messages, a row written for a
different prior is never silently skipped, and the ranking orders gates, then constraint
residual, then pulls.

**Oracles** `F1` throughout: values read back off the prior object `fit.py` builds, object
identity after the context, exceptions, and a hand-built synthetic row set whose ranking is
fixed by construction.

**Cost** 9 test functions, 15 cases; under one second (2026-09-13).

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_patched_prior_reaches_the_built_prior_and_restores_the_original_objects` | inside: `gp_prior("g")` sigma, tied mean, length, jitter, floor, results dir carry the cell; after: every patched attribute `is` the original | `F1` | exact | `S3` |
| 2 | `test_patched_prior_restores_even_when_the_body_raises` | restoration on exception | `F1` | identity | `S3` |
| 3 | `test_envelope_cells_reach_the_envelope_prior` | envelope sigma/alpha/beta/jitter reach `BetaTaperedLogRBF` | `F1` | exact | `S3` |
| 4 | `test_validate_refuses_incomplete_or_out_of_range_cells` | unknown key, bad prior form, alpha 0.9, beta 0.1, zero amplitude, missing field, zero jitter raise | `F1` | raises | `S3` |
| 5 | `test_cell_hash_ignores_name_and_provenance_but_not_any_value` | hash stable under relabelling, moves on a 1e-12 relative amplitude change | `F1` | exact | `S3` |
| 6 | `test_stage_one_grids_are_valid_distinct_and_include_the_shipped_prior` | unique names and hashes; shipped priors present; forms separated | `F1` | exact | `S3` |
| 7 | `test_gate_values_are_read_back_from_both_failure_messages` | VEGAS and nested-VEGAS messages parse to their fractions; saddle failure parses to nothing | `F1` | exact | `S3` |
| 8 | `test_skip_existing_refuses_a_row_written_for_a_different_prior` | an existing row with another hash raises instead of being skipped | `F1` | raises | `S3` |
| 9 | `test_ranking_orders_gates_then_constraints_then_pulls` | four synthetic cells come back in the documented order | `F1` | exact order | `S3` |

**Mutation (2026-09-13).** With a throwaway plugin: a `patched_prior` that never restores fails
tests 1 and 2; one that leaves `GP_AMPLITUDE_FLOOR` stale fails test 1; one that leaves both
jitters unpatched fails tests 1 and 3.

**Weak spots** The screen metrics in `campaign/prior_screen.py` and the subprocess fit path
are not unit-tested here; they are exercised by the scan's own smoke run.
