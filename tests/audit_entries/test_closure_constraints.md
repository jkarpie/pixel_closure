### `tests/test_closure_constraints.py`

**Exercises** the repo-root `closure_NNPDF_truth` and `closure_NNPDF_truth_small`
behaviourally, and all four packages (those two plus `closure_JAM_truth` and
`closure_JAM_truth_small`) structurally in one identity test -- specifically
`fit.constraint_datasets` (byte-identical
across all four packages) and `config.{make_grid, vanishes_at_origin,
low_x_completion, ALL_FIELDS, VALENCE_NORMS, SINGLET_GLUON_FIELDS,
CONSTRAINT_*_SIGMA, ENDPOINT_X, T0_MAX_ITERATIONS}`. These are `closure_*/` driver
packages, not `src/pixel`. `pixel.core.model.{Field, Dataset, Contribution}`
(`src/pixel/core/model.py`), `pixel.geometry.Grid` (`src/pixel/geometry/grid.py`),
and `pixel.kernels.{Delta, Mellin}` (`src/pixel/kernels/common.py`) appear only as
generic, already-covered plumbing that `constraint_datasets` calls; no formula
native to `pixel` is under test here.

**Claim** Near-hard closure pseudo-constraints (field vanishing at `x=1`, valence
quark-counting norms, and the `x*f(x=0)=0` origin limit) are satisfied by the
*represented* injected closure truth, not the nominal physical value, and the
origin constraint switches on only when the grid genuinely reaches `x=0` --
otherwise the vanishing limit is left to the low-x completion.

**Oracles** As of 2026-08-13 there is one independent closed form:
`test_constraint_targets_match_an_analytic_affine_truth` injects `q(x) = A + B x`
-- represented *exactly* by the cubic-spline element basis -- so the endpoint
target is `q(1) = A + B` and the valence-norm target is
`A (1 + ln 1/x_min) + B`, neither computed with any reference to `pixel` (`A1`).
The older numeric checks recompute a prediction with
`contribution.kernel.matrix(...)` -- the *same kernel object*
`constraint_datasets` already used to build `dataset.mean` -- against a curve
provably identical (exact-knot `np.interp`) to the internal one; this proves
`Dataset` storage is self-consistent, not that the chosen `nu`/`alpha`/target
formula is correct (demonstrated by mutation, below), and they are retained
alongside rather than in place of the closed form. Two further non-tautological
checks: the one-hot `Delta`-row test confirming the finite-element basis actually
selects the `x=0` node, and its complement -- that on the production `x_min > 0`
grid a `Delta` row at `x=0` is *identically* zero (measured `max|row| = 0.0`,
`nnz = 0`), which is the premise the origin guard exists for.

**Two suite tuples since 2026-08-14, and the split is load-bearing.** `SUITES` --
what every `@parametrize("suite", ...)` uses -- is the NNPDF pair
(`full_nnpdf`, `small_nnpdf`), trimmed from all four on the owner's instruction.
`ALL_SUITES` is the four-entry tuple and has exactly one reader,
`test_the_four_suites_still_share_one_constraint_builder`
(`reference, *others = ALL_SUITES`), whose subject *is* the four copies. Trimming
`ALL_SUITES` would leave `others` holding a single suite: the test would still pass and
would no longer be able to see a JAM copy diverge. That test is also what makes the
narrow behavioural sweep defensible -- `constraint_datasets` and the 14 `config.py`
constants this file reads are identical across all four packages **as asserted, not as
assumed**, which is a stronger guard than running the same numbers through four copies
and getting the same answer. The full/`_small` pair is kept in `SUITES` because those
two genuinely differ (grid size).

**Cost** 13 tests, **25 cases**; no `@pytest.mark.slow`; all operate on the closure
grid (128 points, plus a 3-rung 145-point `x_min` sweep in #12) with no
MCMC/fitting -- measured **25 passed in 1.60 s** for the file (2026-08-14, serial),
against **49 passed in 2.06 s** immediately before the JAM legs came out of `SUITES`.
Nothing
here runs an actual closure fit; the `t0` loop is driven over a scripted
prediction sequence with `build_analysis` stubbed.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_near_hard_constraints_match_truth_in_the_fit_basis` | Every near-hard dataset's mean equals a kernel/curve recompute; dataset count is `13` | Same-kernel-object recompute (`A3`) | `atol=1e-12` | `W1`, partnered by #2 |
| 2 | `test_constraint_targets_match_an_analytic_affine_truth` | On `q = A + Bx`, endpoint target `= A+B` and valence-norm target `= A(1+ln 1/x_min)+B`; `nu` and the quark-counting nominals are literals | Closed form, no `pixel` call (`A1`) | endpoint `rtol=1e-12` (measured `0.0`); norm `rtol=2e-4` (measured `5.543e-5`, 27.7% used) -- and that bar is `9.18e-4 * q(x_min)/norm`, falling only as `1/ln(1/x_min)` (measured `7.679e-5 .. 4.338e-5` over `x_min` `1e-4..1e-8`), **not** a low-x guarantee; #12 carries that claim | `S1` |
| 3 | `test_constraint_targets_read_the_represented_truth_not_the_nominal_value` | A truth offset by `+0.25` moves every `Delta` target off its nominal `0.0` | Node value of the injected curve (`A1`) | `atol=0.0` exact | `S1` |
| 4 | `test_constraint_datasets_rejects_a_truth_missing_a_field_curve` | A missing field curve raises, naming the field; the intact truth is accepted | `pytest.raises` on the message + control (`F1`) | message match | `S2` |
| 5 | `test_vanishes_at_origin_rejects_an_unknown_field` | A name outside `ALL_FIELDS` raises rather than falling through to `True` | `pytest.raises` on the message + two controls (`F1`) | message match | `S2` |
| 6 | `test_the_four_suites_still_share_one_constraint_builder` | The four suites' builder source and 14 config constants are identical | Identity between suite-built objects (`A3`) | exact equality | `S3` |
| 7 | `test_t0_iteration_ceiling_is_at_least_the_mcp_t0_map_budget` (renamed 2026-08-14) | `T0_MAX_ITERATIONS` is at least the MCP `t0_map` job's own budget under **every** reading of it, **and** the two loops still share their convergence tolerance | budget *and* loop shape both derived from `src/pixel/mcp/runner.py`'s AST -- the `options.pop` defaults and the `range(...)` call -- rather than re-typed (`F1`) | `T0_TOLERANCE == 1e-4 == the MCP default`; `>= max(refits 8, comparisons 8, passes 9)` = `>= 9` (configured `12`, 33% headroom, bar unmoved) | `W2` resolved, partnered by #8 |
| 8 | `test_t0_iteration_stops_on_tolerance_and_never_exceeds_the_ceiling` | The loop stops at the first relative shift `<= T0_TOLERANCE` (the 5th), caps at `T0_MAX_ITERATIONS`, rebuilds `exp`/`both` as `"dis"`, and no-ops on an empty `t0` | Arithmetic on a scripted sequence (`A1`) | exact iteration counts | `S1` |
| 9 | `test_origin_constraint_is_off_when_the_grid_never_reaches_zero` | No `cons_origin_*` dataset on the default grid; `Delta(0)` there is an all-zero row; low-x completion dispatch with `alpha` as a literal | Key-set check + direct row measurement (`F1`/`A1`) | `nnz == 0`; exact dict equality | `S3` |
| 10 | `test_origin_constraint_switches_on_for_the_vanishing_fields` | Exactly 7 fields get an origin dataset at `nu=mean=0`; `Delta` row is one-hot with `sum == 1.0` exactly; `cov` pinned to a literal `1e-4**2` | Nodal-basis identity (`A1`) for the row | `atol=0.0` exact; `rel=1e-12` on `cov` | `S2` |
| 11 | `test_origin_constraint_targets_the_represented_truth` | Origin (and other) constraints target the represented curve, not nominal | Same-kernel-object recompute (`A3`) | `atol=1e-12` | `W1`, partnered by #3 |
| 12 | `test_valence_norm_low_x_error_converges_in_x_min` | On a **vanishing** truth `x**0.5 (1-x)**3` swept over `x_min` in `{1e-6, 1e-7, 1e-8}` at fixed `n_points=145`: the `cons_norm_*` error is the completion's closed-form **model** error, and falls at `10**a` per decade; plus `alpha + gamma > -1`, named on the config (`LowXExtension`'s own check never sees `alpha`; `check_low_x_integrable` landed kernel-side 2026-08-13) | `B(a, beta+1)` = exactly `32/35` and `mpmath.betainc`, 40 digits, no grid and no completion (`A1`) | level `10%` (measured `0.03/0.29/2.17%`); rate `1.3x` around `10**0.5` (measured `3.1727`, `3.2228`); `gamma > 0` exact | `S1` |

**Weak spots**

- **RESOLVED 2026-08-14 — `build_mellin`'s unconditional `momentum_density=True` is
  REQUIRED, and the audit's per-field proposal would have been wrong physics.** The
  fields are momentum densities `q = x f`, so a record labelled order `n` means
  `<x^(n-1)>_f = int x^(n-2) q dx` — the `alpha = -1` weight. That is a storage
  convention, identical for `sigma`/`g` and the valence combinations; with
  `momentum_density=False` the same row computes `<x^n>_f`, the right integral of the
  wrong moment.
  `test_mellin_moment_convention_forces_momentum_density_on_every_field` pins it
  against the closed form `mpmath.beta(p+n-1, b+1)` with a measured non-degeneracy
  control (the `alpha=0` answer is `0.385` away, far outside the `1e-4` quadrature
  bar), and pins **why the pairing with `cfg.low_x_completion` is safe**: a Mellin row
  of order `n` carries its own `x^(n-1)`, so the effective low-x weight is `n-2`, and
  these suites build `n = 2`/`n = 3` only (asserted). Swapping the completion for
  `flat` and for a *rising* `power(-0.3)` moves the folded moment by at most `9.1e-09`
  against a `1e-6` bar. Acceptance: a plugin downgrading `Mellin(alpha=-1)` to
  `alpha=0` for `nu >= 2` only — scoped so the `nu = 1` `cons_norm_*` row is untouched
  — fails exactly the 4 new parametrizations, 45 other tests pass. **Does not**
  establish that `cfg.low_x_completion` picks the right *shape*; it does not for
  `t8`/`t15`, but the exposure is in the `alpha = -1` pseudo-ITD and `cons_norm_*`
  rows (see `plans/low_x_head_diagnosis.md` ADDENDUM 2).

- **RESOLVED 2026-08-13 by `test_constraint_targets_match_an_analytic_affine_truth`.**
  `test_near_hard_constraints_match_truth_in_the_fit_basis`'s core check
  (`tests/test_closure_constraints.py:110-118`) recomputes with the identical
  `kernel` object and a provably identical curve, so it is `assert f(x) == f(x)`
  in the kernel/curve pairing. Demonstrated by mutation: swapping the endpoint
  `nu` from the correct `1.0` to a wrong `0.9` moves the stored target from
  `5.865125278789004e-20` to `0.01453329096320881` (factor `2.478e17`) and the
  identical `assert_allclose(..., atol=1e-12)` still passes -- a mistyped
  `ENDPOINT_X` would sail through undetected. The new test supplies the closed
  form instead: an affine truth `q = A + Bx` is represented exactly by the
  spline basis, so the endpoint target is `A + B` and the valence-norm target
  `A(1 + ln 1/x_min) + B`, both computed without calling `pixel`. Acceptance:
  under `ENDPOINT_X = 0.9` the new test fails on all four suites while the four
  pre-existing tests still pass (`8 failed, 33 passed`); the same holds for
  `alpha=-1 -> 0` on the valence Mellin kernel. The recompute test is kept: it
  is the one whose answer can be read off by inspection.
- **RESOLVED 2026-08-13 by `test_constraint_targets_read_the_represented_truth_not_the_nominal_value`.**
  `test_origin_constraint_targets_the_represented_truth`
  (`tests/test_closure_constraints.py:254-301`) explicitly names and comments
  that it distinguishes "represented" from "nominal" targets, but as measured
  this is unfalsifiable for the origin family: the file's synthetic curve
  `x**0.3 * (1-x)**2` is exactly zero at `x=0`, so represented and nominal
  targets coincide (measured `0.0 == 0.0`) regardless of whether
  `represented_target` reads the curve or silently falls back to nominal. Only
  the valence-norm sub-case, folded into the same loop, exercised a real gap
  (measured `represented=3.28899164907155` vs `nominal=1.0` for `v3`). The new
  test adds the offset curve `x**0.3 (1-x)**2 + 0.25`, nonzero at both `x=0`
  and `x=1`. Acceptance: a `represented_target` that silently returns `nominal`
  **for Delta kernels only** (leaving the Mellin path correct) is invisible to
  every pre-existing test and fails the new one -- measured `9 failed, 32
  passed`, where the failures are the two new tests on four suites plus the
  source-identity guard, which reads `inspect.getsource` and is perturbed by
  the recompiling harness rather than by the bug.
- **RESOLVED 2026-08-13 by `test_the_four_suites_still_share_one_constraint_builder`.**
  The `SUITES = (full_jam, full_nnpdf, small_jam, small_nnpdf)` parametrization
  runs one code path four times: `constraint_datasets` is byte-identical across
  all four `fit.py` files, and every `config.py` constant this file reads is
  identical too. That was recorded only as prose, which goes stale silently. It
  is now an assertion over the builder's source text and 14 config constants.
  Acceptance: setting `closure_NNPDF_truth.config.LOW_X_LINEAR_POWER = 2.0`
  fails the guard plus exactly the two NNPDF-suite value assertions that depend
  on it (`3 failed, 38 passed`); before the fix nothing failed.
- **PARTLY RESOLVED 2026-08-13 by `test_t0_iteration_stops_on_tolerance_and_never_exceeds_the_ceiling`.**
  `test_t0_iteration_ceiling_covers_observed_small_suite_convergence`
  (`tests/test_closure_constraints.py:124-136`) asserts `T0_MAX_ITERATIONS >= 9`
  but never runs `_iterate_t0`; the literal `9`'s claimed provenance ("observed
  small suite convergence") was not found anywhere in `cfg.py`, `fit.py`, or
  `closure_logs/`. The loop itself is now driven, over a scripted prediction
  sequence `t0_k = 1000 (1 + 10**-k)` with `build_analysis`/`_dis_predictions`
  stubbed, and its stopping index (5), its ceiling behaviour, its
  `exp`/`both -> "dis"` rebuild and its empty-`t0` no-op are all asserted.
  Acceptance: three separate mutations -- `range(1, max_iter + 2)`, an absolute
  instead of relative shift, and `t0_mode = mode` -- each fail only this test
  (`4 failed, 37 passed`).
- **THE REMAINING HALF RESOLVED 2026-08-14** by
  `test_t0_iteration_ceiling_is_at_least_the_mcp_t0_map_budget` (renamed from
  `test_t0_iteration_ceiling_covers_observed_small_suite_convergence`). The `9`
  is **not** an observed iteration count and the old name should not have said it
  was; the audit's grep of `cfg`, `fit.py` and `closure_logs/` is confirmed --
  nothing records one. The only in-repo quantity `9` matches is the sibling
  implementation's own budget: `_run_t0_map` (`src/pixel/mcp/runner.py:243,253`)
  defaults `max_iterations = 8` and loops `range(max_iterations + 1)`, i.e. **9
  passes**, which is exactly how `guides/experimental_data_and_t0.md`'s
  side-by-side table prints it -- and that guide's own "upper bound on the
  setting" cell says only "none; a test asserts `>= 9`", i.e. the guide was
  pointing back at the test. The floor is now *derived* from that function's AST
  instead of typed, and the numeric bar is unchanged at 9: this records where the
  number comes from, it does not move it. Added alongside: the two loops must
  still share their convergence criterion (`T0_TOLERANCE == the MCP default
  tolerance == 1e-4`), without which comparing budgets is meaningless.
  *Acceptance*, two mutations, each against the pre-fix copy and the fixed file:
  (a) `cfg.T0_TOLERANCE = 2e-4` on all four suites -- deliberately chosen to stay
  inside the regime window `test_t0_iteration_stops_on_tolerance_...` asserts and
  to leave its expected stop index at 5, so nothing else in the file can notice
  the divergence: pre-fix `49 passed`; fixed `4 failed, 45 passed`. (b) the MCP
  default read back as `max_iterations = 20` (intercepted at `Path.read_text`, so
  `src/` is untouched): pre-fix `49 passed` **with the plugin self-check
  reporting `runner.py` rewritten 0 times** -- the pre-fix test never opened the
  file at all, which is the finding -- and fixed `4 failed, 45 passed` on
  `assert 12 >= (20 + 1)`, self-check 4 rewrites. (Both mutation counts are against
  the pre-2026-08-14 49-case parametrization and were **not** re-measured when `SUITES`
  was trimmed to the NNPDF pair; the mutations patch the packages, not the
  parametrization, so they still bite -- only the arithmetic changes, the two JAM cases
  of each failing test no longer being collected. Re-measure before quoting a
  failed/passed pair from this item.) **Still open, unchanged**: no
  real-fit iteration count exists anywhere; that needs a full `build_analysis`
  and generated truth on disk (report item `M04`), which the owner's hold on
  closure pipeline runs forbids. ~~Two readings of "budget" are also left
  unreconciled in the docstring~~ -- reconciled 2026-08-14, see the next bullet.
- **THE LAST TWO PIECES CLOSED 2026-08-14; the weakness is now RESOLVED and only the
  missing test `M04` remains.** *(a) The typed `+ 1`.* The floor read
  `T0_MAX_ITERATIONS >= defaults["max_iterations"] + 1`, so the sibling's *literal*
  was derived from its AST while the arithmetic turning it into a pass count -- the
  loop shape `range(max_iterations + 1)` -- was still a copy at the call site. The
  half left typed was the half that encodes **how the two loops compare**.
  `_mcp_t0_map_defaults` now evaluates the loop's own `range(...)` call (via
  `_eval_with`, an arithmetic-only AST walker that *raises* on anything it does not
  recognise rather than guessing) and returns `passes` and `comparisons` beside the
  two literals. *(b) The two readings.* Reading both loops settles them:
  `_run_t0_map` gets 9 passes = 1 initial fit + 8 refits, with 8 comparisons (the
  first has `previous is None`); `_iterate_t0` takes one prediction **before** its
  loop -- the analogue of that first MCP pass -- then runs `range(1, max_iter + 1)`,
  so `T0_MAX_ITERATIONS` counts *refits*: 12 refits, 12 comparisons, 13 predictions.
  Refits 12 vs 8, comparisons 12 vs 8, passes 13 vs 9 -- the closure budget wins
  under every reading, so the ambiguity never affected the verdict. The assertion
  takes `max()` over the three MCP counts, which makes "the stricter is asserted"
  executable rather than prose, and `assert floor >= 1` stops a broken parse from
  making it vacuous. **Bar unchanged at 9, achieved 12 (33% headroom), recorded both
  ways.** *Acceptance*: mutation `mcp_loop_widened` rewrites the parsed text to
  `range(max_iterations + 6)` (14 passes) at `Path.read_text`, leaving the
  `max_iterations = 8` literal alone -- **2 failed / 23 passed**, both failures being
  this test at `assert 12 >= 14`, self-check `runner.py` rewritten 3 times; the
  pre-fix arithmetic on the identical mutated text is `8 + 1 = 9`, so `12 >= 9`
  **passed** -- the old bar could not see the sibling loop widened by any amount,
  because the only thing it read from that loop was a literal the widening does not
  touch. Clean: **25 passed in 1.58 s**.
- **RESOLVED 2026-08-13 by `test_valence_norm_low_x_error_converges_in_x_min`.**
  `test_constraint_targets_match_an_analytic_affine_truth`'s valence-norm bar
  `rtol=2e-4` read as a low-x accuracy guarantee and was neither that nor
  `x_min`-independent. Measured: the achieved value is *exactly*
  `9.18e-4 * q(x_min) / norm_exact` -- reproduced to every printed digit at five
  `x_min` values -- so it is the head's **quadrature** error only, inherited from
  `cfg.X_MIN` through `1/(1 + ln(1/x_min))` and falling merely logarithmically
  (`7.679e-05`, `6.438e-05`, `5.543e-05`, `4.867e-05`, `4.338e-05` at `x_min`
  `1e-4 .. 1e-8`; a factor of 1.77 over four decades). The cause is the truth's
  own class: `AFFINE_A = 0.4`, so `q(0) != 0`, `int q/x` is log-divergent in exact
  arithmetic, and `norm_exact` is finite only *because* the completion defines it
  -- there is no model error left for a rate to converge. The new test supplies a
  **vanishing, sublinear** truth `x**0.5 (1-x)**3` instead, states `x_min`
  explicitly rather than inheriting `cfg.X_MIN`, and asserts a rate: level against
  the closed-form head-model error (`10%` bar, measured `0.03/0.29/2.17%`) and
  `10**a` per decade (`1.3x` bar, measured `3.1727`, `3.2228` against `3.1623`).
  `a = 0.5` is a stand-in for physical valence and both bars are written as
  functions of `a`, since the real closure curves' small-x power is unknown
  (`plans/low_x_head_diagnosis.md`). Acceptance, two mutations, each caught by a
  different assertion: a **flat completion on the valence fields** makes
  `alpha + gamma = -1` (divergent) and fails the integrability line, while its
  error still falls at `3.1606`/`3.1528` per decade so the rate cannot see it;
  **inflating `low_x_quadrature_correction` by 50%** fails the level at
  `got/pred = 0.5008` while leaving the rate at `3.1831`/`3.2860`, inside the
  band. The pre-existing tests do also fail under both, but not as convergence
  statements: `test_constraint_targets_match_an_analytic_affine_truth`'s `norm_exact`
  formula *bakes in* the linear head (the `[0, x_min)` interval contributes exactly
  `q(x_min)`), so a flat completion breaks the formula rather than being measured by it,
  and its `2e-4` quadrature bar is trivially exceeded by a 50% head change. The old test and
  its `2e-4` bar are kept -- its affine truth is still the one whose targets can be
  read off by inspection -- with the docstring rewritten to say what the bar is.
- The two `S4` hygiene items are closed in the same pass: `dataset.cov[0,0]` is
  now compared against a literal `1e-4**2` at `rel=1e-12` instead of reading
  `cfg.CONSTRAINT_ORIGIN_SIGMA` on both sides (acceptance: loosening the config
  constant 10x fails it), and the low-x completion's expected `alpha` is the
  literal `1.0` instead of `cfg.LOW_X_LINEAR_POWER`. The `Delta` row sum at the
  origin node is measured exact (`row.sum() - 1 == 0.0`) and is now asserted as
  such rather than at `pytest.approx`'s default `rel=1e-6`.

**Not covered here**

- `tests/test_closure_extension_systematics.py` also drives
  `closure_JAM_truth`/`closure_NNPDF_truth` `config.low_x_completion`, on the
  generation/systematics side rather than the constraint-datasets side exercised
  here.
- `tests/test_closure_truth_representable.py` checks that linear and bilinear
  *generation* fold the same fixed truth member -- a related but distinct
  truth-representability question from this file's constraint-time check.
- `_iterate_t0` (`closure_JAM_truth/fit.py:460-491`) and `T0_TOLERANCE` are now
  driven by `test_t0_iteration_stops_on_tolerance_and_never_exceeds_the_ceiling`,
  but only over a *scripted* prediction sequence with `build_analysis` stubbed.
  How many iterations a real fit needs is still measured nowhere (proposal
  `-M04`, left open: it needs a full `build_analysis` and generated truth on
  disk).
- The low-x head PIXEL uses for a `Mellin(alpha != 0)` kernel is short by a
  measured `9.18e-4 * q(x_min)` -- `kernels.base.assemble` disables the exact
  `low_x_mellin_head` whenever `alpha != 0`, and the Gauss-Jacobi fallback then
  integrates a `1/t` singularity against a `t^gamma` weight. It survives both
  grid refinement (`N` 65 -> 513) and quadrature refinement
  (`points_per_interval` 8 -> 256). It is `src/pixel` behaviour, not closure
  code, and it cancels out of a closure fit because the same operator writes and
  reads the target; it is why this file's valence-norm bar is `2e-4` rather than
  machine precision. Not filed as a weakness of this file. **Superseded in part
  2026-08-13**: that `9.18e-4` is the Gauss-Jacobi rule's closed-form
  `1/(npts+1)**2` at the shipped `npts = 32`, and it is *not* the dominant error
  on a physical (vanishing, sublinear) field -- the completion being the wrong
  shape beats it by ~1000x. Both are now measured and separated by #12; the full
  diagnosis is `plans/low_x_head_diagnosis.md`.
- **The missing integrability validation is `src/`, and stays open.** `LowXExtension`
  checks `effective_power > -1` without ever seeing the caller's integrand weight
  `alpha`, so `flat` paired with a momentum density's `alpha = -1` is accepted
  everywhere and returns a finite, `npts`-dependent, `x_min`-independent number
  instead of raising. #12 asserts `gamma > 0` on the production config, but that is
  a test-side guard on one config; the `src/` fix is proposed as a diff in
  `plans/test_audit/reports/test_kernel_guards.json` and the behaviour is pinned by
  `tests/test_kernel_guards.py::test_low_x_integrability_guard_rejects_a_divergent_pairing`.
