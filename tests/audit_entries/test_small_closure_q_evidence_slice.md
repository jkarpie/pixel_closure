### `tests/test_small_closure_q_evidence_slice.py`

**Exercises** `closure_small_q_integration/plot_evidence_scans.py`
(`build_slice`, `_slice_residuals`, `FieldSliceTarget`, `resolve_windows`,
`DEFAULT_X_VALUES`, `DEFAULT_SIGMA_HALF_WIDTH`) and, through
it, `src/pixel/infer/q_integration/_legacy.py`'s `PriorSubspace`
(`from_covariances`, `q_from_zeta`, `blocks[i].Q/L0/rank`) plus the value of
`src/pixel/plotting/evidence_scan.py`'s `MAX_SCAN_PARAMETERS`. It does **not**
exercise `scan_evidence`, `plot_evidence_scan`, `find_center`, `estimate_cost` or
any figure: the scan machinery is pinned in `pixel`'s own
`tests/test_evidence_scan.py`, and the rest needs a compiled model.

**Claim** The explicit-`q` evidence scan's axes are the physical quantities their
labels name. A fitted normalization axis is that normalization; a `field(x=...)`
axis is that distribution's value at that *snapped grid node*; a unit step along
either moves exactly that quantity by exactly one; the default `"prior"`
direction's leakage into the other scanned `x` of the same field is exactly
`K[k', k] / K[k, k]` and never crosses fields; `"node"` has no leakage; an
unknown direction raises; a scan window is sized by the *conditional* width
`sqrt(2/H_ii)` of the slice being drawn rather than the scanner's marginal
`sqrt(2[H^-1]_ii)`, with five of them landing the axis endpoint exactly on the
`Delta S = 25` colour ceiling; and the 34-coordinate figure the module is built
around fits inside the raised scan ceiling.

**Oracles** One kind, and it carries the correctness claim: `_dense_covariance`
reassembles `Q L0 L0' Q'` for one field from the subspace's own matrices in plain
numpy in this file, and every expected number is read off it. The fixture's
covariance is a hand-written smooth log-`x` squared-exponential on four nodes,
`scale = 1` for `t3` and `scale = 4` for `g` — the factor of four is what rejects
reading the wrong block, and the smoothness is what makes the cross-response
assertion non-vacuous (asserted directly: `max_abs_cross_response > 0.05`, so a
near-diagonal kernel that turned the expected matrix into the identity would fail
the guard rather than pass the test). `_slice_residuals` is measured through the
fixture's own `physical`, which is the real `PriorSubspace.q_from_zeta` lift, so
the "unit step moves one field value by one" claim is checked against the lift
rather than against the formula that built the column.

**Cost** 13 test functions, **13 cases**; measured `13 passed in 0.97 s`
(2026-10-16, serial). Nothing compiles a model, builds a kernel, or runs a
JAX-jitted path other than one 5x5 `jax.hessian` of a quadratic.

The window tests use a second fixture, `_CorrelatedTarget`: `S = k(a^2 + 2c ab +
b^2)`, whose conditional width is `1/sqrt(k)` and marginal width
`1/sqrt(k(1-c^2))`, so the ratio is `1/sqrt(1-c^2)` and nothing else. At
`c^2 = 0.99` that ratio is exactly 10, which is asserted — a fixture whose two
widths coincided would make every window assertion vacuous.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_every_axis_is_the_quantity_its_label_names` | label text and order (normalizations first, then `(field, x)` in grid order); labels carry the **snapped** node, not the request; the centre is the physical state at `x_center`, not the coordinate origin | exact label list + `_MEAN` entries | exact list; `rtol 1e-12` | `S2` |
| 2 | `test_the_recorded_prior_width_is_the_kernels_own_diagonal` | `prior_variance == K[k, k]` for every axis, per field | `_dense_covariance` | `rel 1e-10`; the 4x ratio at `rel 1e-8` | `S2` |
| 3 | `test_a_unit_step_moves_its_own_field_value_by_exactly_one` | self-response is 1 for **both** directions | measured through `q_from_zeta` | `atol 1e-12` | `S1` |
| 4 | `test_the_prior_direction_leaks_into_the_other_x_by_exactly_the_kernel_ratio` | the whole 4x4 cross-response equals `K[k', k] / K[k, k]` within a field and 0 across fields; the fixture can see a leak at all | `_dense_covariance` | `atol 1e-10`; `> 0.05`; `atol 1e-14` | `S1` |
| 5 | `test_the_node_direction_moves_one_grid_point_and_nothing_else` | `--direction node` gives the identity response | measured through `q_from_zeta` | `atol 1e-10` | `S2` |
| 6 | `test_an_unknown_direction_is_refused_rather_than_defaulted` | a typo raises instead of falling back | `ValueError` match | raises | `S3` |
| 7 | `test_the_slice_target_exposes_the_scanners_protocol_and_the_same_scalar` | `support_bounds()`/`bounds()` are `None`; no `values` attribute; the adapter returns the target's own value at the centre unshifted; the reduced Hessian is square and symmetric | identity + echo | `is None`; `rel 1e-12`; `atol 1e-10` | `S2` |
| 8 | `test_the_conditional_window_is_the_slice_width_not_the_marginal_one` | both widths match closed form; the ratio is exactly 10; the bounds use the conditional one | `_CorrelatedTarget` algebra | `rtol 1e-12` / `1e-8`; `rel 1e-6` | `S1` |
| 9 | `test_the_endpoint_of_a_default_window_lands_on_the_colour_ceiling` | five conditional sigma is `Delta S = 25`, checked on the objective itself rather than the algebra that chose it | evaluate `_raw_value` at the endpoint | `rel 1e-10` | `S1` |
| 10 | `test_the_marginal_window_defers_to_the_scanner_and_still_reports_both` | `kind="marginal"` returns `None` bounds; both widths are reported anyway | identity | `is None` | `S3` |
| 11 | `test_a_flat_or_inverted_axis_refuses_a_conditional_window` | a negative Hessian diagonal raises with the flag that fixes it; the same centre is still scannable as marginal | `ValueError` match + control | raises; `is None` | `S2` |
| 12 | `test_an_unknown_window_kind_is_refused` | a typo raises instead of defaulting | `ValueError` match | raises | `S3` |
| 13 | `test_the_full_field_scan_fits_inside_the_raised_scan_ceiling` | `7 + 9*3 == 34` and `MAX_SCAN_PARAMETERS >= 34` | arithmetic + the library constant | exact | `S3` |

**Why #7 asserts the absence of `values`.** `pixel.plotting.evidence_scan._batched_objective`
prefers `target.values(batch)` over `lax.map(target._raw_value, ...)` when a
`values` method exists. `FieldSliceTarget` has no batched path, so acquiring a
`values` attribute — by inheritance, or by someone adding one — would route every
node through something untested. The negative assertion is cheaper than finding
that out from a scan's numbers.

**Weak spots**

- **The scaling claim is untested.** The module's reason for choosing `"prior"`
  over `"node"` is that a single-node direction's prior cost is `[K^-1]_kk`,
  astronomical on the real 146-node kernel, so its Laplace window collapses.
  Nothing here measures that: #5 asserts only that `"node"` is what it says.
  Checking the stiffness would need the real covariance, and on the four-node
  fixture the ratio is far too mild to be a bar. It is visible instead in the
  `--estimate` output's `windows` block, which prints both sigmas for every axis
  before a scan launches.
- **Only two `x` per field, and both inside one block.** The cross-response
  matrix is 4x4, so a bug that mixed up *three or more* coordinates of one field
  — a wrong stride in the retained-offset arithmetic, say — has fewer places to
  show itself than in the 27-coordinate production slice. The offsets are
  exercised across two fields of unequal covariance, which is the part that
  actually moved.
- **`max_field_drift_per_unit_normalization` is structurally zero here.** The
  fixture's `physical` builds `q` from a constant `_MEAN`, so the measurement
  that the GP mean does not depend on the normalizations cannot fail in this
  file. On the real Q=2 `exp` model it is measured and recorded in the manifest,
  where it came out exactly `0.0`.

**Not covered here**

- Everything downstream of the slice: `scan_evidence`'s windows and panels,
  `plot_evidence_scan`'s layout, `save_large_figure`'s dpi budget, and the
  figures themselves. The first two are pinned in `pixel`.
- `find_center`. The two-stage L-BFGS-B/Newton-CG MAP is only reachable with a
  real 1,321-dimensional target; its measured behaviour (`S` from `1.85e10` to
  `~861`, `|grad S|` to `1e-6` in ~50 Newton-CG iterations) is recorded in the
  module docstring and in `closure_small_q_integration/README.md`, not asserted.
- The cache in `_load_center`/`_store_center`, including its `QTarget.metadata()`
  fingerprint.
- That a marginal window actually saturates the colour ceiling on the real
  model. The measurement that motivated `--window conditional` — `sigma(x=0.1)`
  and `g(x=0.1)` drawing blank on a 16-coordinate pilot, their marginal width
  being sixteen times their conditional one — is recorded in the module
  docstring and the README, and is visible in
  `outputs/jam/results/truthQ_2/q_evidence_scan_exp_pilot16cond.png` against its
  predecessor, but no test reproduces it.
