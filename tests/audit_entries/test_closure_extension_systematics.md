### `tests/test_closure_extension_systematics.py`

**Exercises** `closure_JAM_truth/generate.py` and its `closure_NNPDF_truth` twin
(`_mellin_n1_row`, `build_systematic_curves`, `coefficient`, `fold_lattice_systematics`,
`inflate_diagonal`, `assemble_operator`, `fold_truth`, `lattice_layout`),
`closure_JAM_truth/config.py` (`Ensemble.z_max`, `Ensemble.p_max`, `active_ensembles`,
`low_x_completion`, `moment_nuisance_specs`), `closure_JAM_truth/datasets.py`
(`_moment_systematic_contributions`), and past those drivers `src/pixel/kernels/common.py`
(`Cosine`/`Sine` aliased `PseudoITDReal`/`PseudoITDImag`, `Mellin`, `Unit`),
`src/pixel/kernels/lowx.py` (`low_x_fourier_head`, `low_x_quadrature_correction`),
`src/pixel/kernels/base.py` (`Kernel.matrix`/`apply_coefficient`), and
`src/pixel/kernels/lattice.py` (`LATTICE_SYSTEMATICS`).

**Claim** the injected lattice-systematic curves obey five enforced properties (parity in
`nu`, `nu->0`/`nu->inf` limits, a nulled `n=1` counting moment, a 10% amplitude cap, 1%
covariance-diagonal inflation) plus the ceil-based `z` and floor-based `p` ranges for the
ensemble table.

**Oracles** Properties 3 (nulled moment / vanishing `nu=0` real value) and 4 (the cap) are
now checked against oracles built outside the code that produced them, which is what makes
them able to fail. `_independent_power_moment_row` (module-level helper, added
2026-08-13) re-derives `int_0^1 x^w e_j(x) dx` from the basis alone: substitute
`x = exp(u)`, composite Gauss-Legendre per grid interval, and a closed-form `[0, x_min)`
head `x_min**(w+1)/(w+1+gamma)` — a different quadrature, a different variable, and a
different low-x treatment from PIXEL's own composite rule plus
`low_x_quadrature_correction`/`low_x_fourier_head`. The cap test rebuilds both the shift
(`coefficient(key, meta) * (B @ sys_curves[field][key])`) and the folded signal
(`(ensemble[field] @ B.T).mean(0)`) from the fixture, leaving only `folds.scale` sourced
from the function under test. Property 1 (parity) remains a property of the `Cosine`/`Sine`
kernel rather than of the curve — exactly even/odd in `nu` for *any* vector — and is
labelled as such, but now covers both production `alpha`/`low_x_extension` branches instead
of only `alpha=-1`. Property 2 (decay) is measured against the real curves with the bound
split by `alpha` class. Property 5 (SPD) runs on two replica ensembles: the original
rank-1-by-construction toy, and a full-rank noisy companion whose folded covariance is
singular only for the emergent reason `n_rep - 1 < n_pts`. The two cross-package tests are
real but narrow: JAM and NNPDF are physically duplicated source files, so they catch the
copies drifting apart, not a shared bug. The systematic-coefficient *formulas* (`ht`,
`chiral`, `inf_Lz`, ...) are independently covered by `tests/test_coefficients.py`, not by
this file.

**Cost** 21 test functions, 19 parametrized over `suite` and one of those also
over two fold fixtures, for **22 collected cases**; measured **22 passed in 3.87 s / 3.97 s**
serial (2026-08-14, two runs).
`actual_lattice_folds` and `nondegenerate_lattice_folds` each build real kernel operators
across all 13 ensembles (117 pseudo-ITD records); test 21 adds six 145-point grids and eight
40-digit `mpmath` integrals, ~0.6 s of that total.

`FULL_SUITES` is **NNPDF truth only** as of 2026-08-14, on the owner's instruction. It was
`(jam, nnpdf)` and the file collected 42 cases, measured **42 passed in 4.92 s** serial on
the same box immediately beforehand. None of the properties this file checks — parity,
Riemann-Lebesgue decay, the nulled `n=1` moment, the 10% cap, SPD after inflation — reads
the truth PDF set, so the JAM leg re-ran them against a physically duplicated twin package
for the same answer. **The saving is ~1.0 s, not the 3.4 s the JAM cases sum to**: `suite`
is module-scoped, so whichever param ran first absorbed the fixture warm-up (basis
construction, the memoized independent rows, the two fold ensembles) and the second param
inherited it. Tests 12 and 13 are untouched — they never took the `suite` fixture, and they
are the two that actually compare the truth families; the module keeps all four
`jam_cfg`/`jam_gen`/`nnpdf_cfg`/`nnpdf_gen` imports for them.

Note the "measured ... across all 45 (field, key) pairs of **both suites**" phrasing in
several test docstrings and in the rows below is a record of what was measured on
2026-08-13, when both legs ran. Those measurements are not invalidated by the trim; they
are simply now broader than what the file re-runs each time.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_basis_is_nodal_at_x_min` | `e_j(x_min) == delta_{j0}` | direct basis evaluation (`F1`) | exact | `S2` |
| 2 | `test_z_range_is_ceil` | `z_max == ceil(0.3fm/a)`, `z_values == arange(1,z_max+1)` | independent `np.ceil` recomputation vs `math.ceil` (`A1`) | exact int | `S1` |
| 3 | `test_p_range_is_floor` | `p_max == floor(L_sites/6)`, `p_values == arange(1,p_max+1)` | independent `np.floor` recomputation (`A1`), plus a count of the ensembles that separate floor from ceil | exact int | `S1` |
| 4 | `test_finite_volume_systematics_are_enabled_for_both_lattice_data_types` | `inf_Lz`/`inf_L` used by exactly one of ITD/Mellin each | membership on hardcoded config tuples (`F1`) | n/a | `S3` |
| 5 | `test_nulled_counting_moment` | `_mellin_n1_row` matches an independent quadrature; every curve's `n=1` moment under *that* row is ~0 | `_independent_power_moment_row(-1, gamma=1)` (`B1`), plus the original same-row check | `1e-12` cols>=1, `1e-3` col 0, `2e-4` on moment/`max|s|`. The `1e-3` is the Gauss-Jacobi head rule's `1/(npts+1)**2` at the shipped `npts=32`, **not** a head-accuracy or `x_min` bar -- measured 6% over four decades of `x_min` vs 3.8x per doubling of `npts` (table in the docstring); #21 carries the `x_min` claim | `S1` |
| 6 | `test_real_even_imag_odd_in_nu` | real component even, imag odd in `nu`, on **three** kernel settings incl. both production branches | `Cosine`/`Sine` kernel-matrix parity (`A2`) | `atol=1e-10` | `S2` |
| 7 | `test_real_vanishes_at_nu0_imag_vanishes_at_nu0` | real component 0 at `nu=0` under an independent row; imaginary kernel *row* at `nu=0` is exactly zero | `_independent_power_moment_row(alpha, gamma)` (`B1`) + `assert_array_equal` on the row | `1e-12`/`1e-3` on the row, `2e-4`/`1e-12` on the value. Both sides use the *same* completion, so the `1e-3` column-0 bar compares two evaluations of one head model and can never see the model error; see #5 and #21 | `S1` |
| 8 | `test_imag_small_nu_slope_matches_independent_moment` | `Imag(nu)/nu -> int x^(alpha+1) f dx` as `nu -> 0` | `_independent_power_moment_row(alpha+1, gamma)` (`B1`) | `1e-7` on the ratio | `S1` |
| 9 | `test_converges_at_large_nu` | tail (`nu=300`) below an `alpha`-class bound times peak (`nu` in `[1,40]`) | numeric evaluation of the real curves (`A2`) | `0.7` (`alpha=-1`), `5e-3` (`alpha=0`), plus a peak floor and a bar-still-honest lower guard | `S2` |
| 10 | `test_generation_fold_enforces_cap_for_every_field_key` | every `(field,key)` pair present and saturating the 10% cap, with shift and signal rebuilt from the fixture | independent rebuild of `raw_shift`/`central`; only `folds.scale` from the fold (`A3`) | `assert_array_equal` on the rebuild; `rtol 1e-10` on the cap | `S1` |
| 11 | `test_actual_folded_covariance_blocks_become_spd` | block shape, rank bounds, Cholesky after 1% inflation, diagonal formula — on a rank-1 **and** a full-rank replica ensemble | structural + real Cholesky/formula checks (`A1`/`F1`) | exact shape/rank; `rtol=1e-13, atol=0` on the diagonal | `S1` |
| 12 | `test_seeded_systematic_curves_match_between_truth_families` | JAM and NNPDF build bit-close curves from the same seed | cross-package comparison of two duplicated files (`A3`) | `atol=1e-14` | `S3` |
| 13 | `test_mellin_systematics_use_distinct_one_point_fields_and_unit` | Mellin nuisance is a distinct one-point field wired through `Unit` | wiring check + shared-factory coefficient equality (`A3`) | `assert_allclose` default | `S2` |
| 14 | `test_mellin_systematic_truths_are_independent_and_deterministic_per_order` | per-`(field,key,order)` truths distinct and cross-package-deterministic | structural coverage + float distinctness (`A2`) | exact | `S3` |
| 15 | `test_generation_fold_caps_mellin_scalars_separately_from_itd_curves` | Mellin cap kept separate from ITD cap; wiring correct; cap saturated two-sided | external-`raw_values` wiring check + `np.isclose` on the cap (`A3`) | `assert_allclose` default; `rtol=1e-10` on the cap | `S2` |
| 16 | `test_systematic_curves_are_distinct_across_keys` | no two `(field,key)` curves coincide, across keys and across fields | relative separation `max|a-b|/max(max|a|,max|b|)` (`A2`) | `1e-3` (measured min `2.55e-02`) | `S1` |
| 17 | `test_fold_lattice_systematics_rejects_malformed_inputs` | the three `ValueError` guards fire, matched on their **messages**, with a clean-input control and both accepted `max_fraction` endpoints | message match (`F1`) | n/a | `S1` |
| 18 | `test_inflate_diagonal_symmetrizes_and_scales_the_diagonal` | `sym(C) + f*diag(diag(C))` exactly; a rank-1 block is lifted to SPD; a zero variance is **not** liftable at any `f` | closed form in numpy (`A1`) | `assert_array_equal`; `rtol=1e-13` on eigenvalues | `S1` |
| 19 | `test_active_ensembles_honours_env_override_and_module_subset` | default, `ACTIVE_ENSEMBLE_IDS`, `PIXEL_CLOSURE_ENSEMBLES` precedence/parsing/order, empty-string fallthrough, unknown id -> `KeyError` | direct calls with `monkeypatch` (`F1`) | exact | `S1` |
| 20 | `test_fold_uses_fixed_truth_curves_when_given_and_replica_mean_otherwise` | `truth_curves=` folds that truth (`generate_member`'s production branch), not the replica mean; `assemble_operator` builds a multi-entry operator; `fold_truth` accumulates all of it | `B @ truth[field]` in numpy (`A1`) + a branch-separation control | `rtol=1e-13, atol=0` | `S1` |
| 21 | `test_low_x_head_error_converges_in_x_min` | Sweeping `x_min` in `{1e-6, 1e-7, 1e-8}` at fixed `n_points=145`, the column-0 head error is the completion's closed-form **model** error and falls at `10**rate` per decade, with `rate` set by the field class (`10**a` for the seven `alpha=-1` fields, `x100` for `sigma`/`g`); plus `alpha + gamma > -1`, named on the config (`LowXExtension`'s own check never sees `alpha`; `check_low_x_integrable` landed kernel-side 2026-08-13) | 40-digit `mpmath` quadrature of the whole `[0,1]` integral with no grid and no completion, and a closed-form head-model prediction (`B1`/`A1`) | integrability exact; level `10%` (measured `0.002 .. 2.17%`); rate `1.3x` around `10**rate` (measured `3.1724`, `3.2215`, `99.71`) | `S1` |

**Weak spots**

- **RESOLVED 2026-08-13 by `test_low_x_head_error_converges_in_x_min` (#21).** The two
  column-0 bars (`1e-3` in test 5, `head_bar` in test 7) read as universal head-accuracy
  guarantees and are neither universal nor about head accuracy. Both compare two evaluations
  of the *same* low-x completion, so **neither can see the completion being the wrong
  shape** — which is the larger error by ~1000x on any sublinear field
  (`plans/low_x_head_diagnosis.md`). And `1e-3` is not an `x_min` bar at all: it is the
  Gauss-Jacobi rule's closed form `1/(npts+1)**2` at the shipped `npts = 32`. Measured by
  sweeping both axes at this file's fixture size — `x_min` `1e-4/1e-6/1e-8` x `npts`
  16/32/64 gives `3.3255e-03 / 8.8037e-04 / 2.2677e-04`, `3.2320e-03 / 8.5569e-04 /
  2.2041e-04`, `3.1421e-03 / 8.3194e-04 / 2.1430e-04`: four decades of `x_min` move it 6%,
  one doubling of `npts` moves it 3.8x (`17**2/33**2 = 0.2654`, measured `0.2648`). So
  refining `x_min` would never retire that bar, which is the opposite of the model error
  dominating the same column. **This corrects the framing that both bars were "silently
  inherited from `x_min`" — only `test_closure_constraints.py`'s `2e-4` is.** Test 21 states
  `x_min` explicitly, sweeps `{1e-6, 1e-7, 1e-8}` at fixed `n_points = 145`, and asserts the
  per-decade **rate** per field class against 40-digit `mpmath` truths, plus the
  integrability condition `alpha + gamma > -1` (asserted on the config: `LowXExtension`
  validates `effective_power > -1` without seeing `alpha`, and nothing rejected the
  divergent pairing at all until `check_low_x_integrable` landed kernel-side on
  2026-08-13). Acceptance, three
  in-memory `cfg`/`src` mutations, each caught by a *different* assertion (so all three earn
  their place): a **linear head on `sigma`/`g`** turns `x100` into `x10.0000` and fails the
  rate (the level does *not* fire — the prediction follows the declared completion and
  tracks the mutation to `got/pred = 1.000000`); a **flat head on the vanishing class**
  makes `alpha + gamma = -1` and fails integrability (the rate does *not* fire — the mutated
  error still falls at `3.1606`/`3.1528`); **inflating `low_x_quadrature_correction` by
  50%** fails the level at `got/pred = 0.5008` while the rate stays at `3.1831`/`3.2860`,
  inside the band. The pre-existing tests do also fail under all three, but never as a
  head-accuracy statement: (i) trips test 7's projector-floor bar (`5.086e-09` vs `1e-09`)
  because `build_systematic_curves` nulls against `generate.py`'s *hardcoded* flat
  `itd_zero` row -- a config-inconsistency detection; (ii) makes
  `_independent_power_moment_row`'s own closed-form head divide by zero
  (`weight + 1 + gamma = 0`) and surfaces as a bare `ZeroDivisionError` inside a helper,
  the oracle encoding the very integrability condition `src/` does not; (iii) trips the
  column-0 quadrature bars. None of them asserted the rate, the level against a head
  model, or which class an error belongs to. Both original bars are kept — they still pin the order/`alpha`/power —
  with docstrings rewritten to say what each is a bar *on*.
- **RESOLVED 2026-08-13** — tests 5, 7 and 10 no longer re-derive their reference from the
  function under test. Test 5 now compares `_mellin_n1_row` against
  `_independent_power_moment_row` and folds the curves through *that* row; test 7 does the
  same at `nu = 0` per field; test 10 rebuilds `raw_shift` and `central` from the fixture.
  Acceptance: patching `_mellin_n1_row` to the `n=2` functional, and to `alpha=0`, each
  makes test 5 fail — while the pre-fix assertion passed under both (measured
  `max|m1 @ s| = 2.111e-14` and `2.109e-14`, bar `1e-9`). Zeroing the flat-branch cosine
  low-x head makes test 7 fail, where the pre-fix form passed at `max|r0| = 9.159e-16`.
  A fold applying the first key's coefficient to every systematic makes test 10 fail; the
  pre-fix recompute passed it at `max|observed/cap - 1| = 2.220e-16`.
- **RESOLVED 2026-08-13** — `test_real_even_imag_odd_in_nu` now runs each field through its
  own production `alpha`/`low_x_extension` as well as the old fixed `alpha=-1`/linear pair,
  so the `alpha=0`/`"flat"` branch (`low_x_fourier_head`'s closed-form series, used by
  `sigma` and `g`) is exercised for the first time. It remains a statement about the kernel,
  not the curve, and says so. Acceptance: a parity-breaking factor `(1 + 0.5*nu)` applied to
  the flat cosine head fails it, where the old `alpha=-1`-only form passed with residual
  `0.000e+00` — that branch was never reached.
- **RESOLVED 2026-08-13** — the content-free `abs(i0) < 1e-12` is gone. The exact zero of the
  imaginary row at `nu = 0` is now asserted as what it is, an identity of the kernel row with
  no reference to `s` (`assert_array_equal`), and the imaginary component's real near-zero
  content — its slope — is covered by the new
  `test_imag_small_nu_slope_matches_independent_moment` against an independent quadrature.
  Acceptance: distorting the sine kernel to `sin(nu x) * (1 + 0.01 x)` fails the new test and
  nothing else in the file (it preserves parity and `sin(0) = 0`).
- **RESOLVED 2026-08-13** — `test_converges_at_large_nu`'s single `0.7` bound is split by
  `alpha` class into `DECAY_BARS = {-1.0: 0.7, 0.0: 5e-3}`, from a full sweep of both suites:
  `alpha=-1` 35 pairs in `[1.4518e-01, 6.0700e-01]`, `alpha=0` 10 pairs in
  `[1.2365e-03, 2.5946e-03]`. Bar utilisation is now 87% and 52% instead of 87% and 0.4%.
  Acceptance: the odd-flat-head mutation above lifts the `alpha=0` ratios to a maximum of
  `7.0843e-03` (the first pair the test trips on is `sigma`/`ht` at `6.3955e-03`), which
  fails the new `5e-3` bar and passed the old `0.7` one.
- **RESOLVED 2026-08-13** — `test_actual_folded_covariance_blocks_become_spd` runs on a second,
  `nondegenerate_lattice_folds` fixture whose replicas are full rank (`matrix_rank == 6` of 6,
  versus 1 for the original toy), so the singularity it asserts is the emergent
  `rank <= n_rep - 1 < n_pts` rather than a restatement of the fixture. Block **shape** and
  rank bounds are now asserted too. Acceptance: `np.cov(Y, rowvar=True)` — a `6 x 6`
  replica-space covariance — fails the new test and **passed every assertion of the old one**;
  building the covariance from only two replicas fails the new full-rank case and passed the
  old one. The original rank-1 fixture is kept alongside, since its folded central value is
  readable by inspection.
- **RESOLVED 2026-08-13** — `test_generation_fold_caps_mellin_scalars_separately_from_itd_curves`'s
  final cap check is now two-sided (`np.isclose(..., rtol=1e-10)`), matching its ITD sibling.
  Acceptance: halving `moment_scale` fails it; the old one-directional `<=` passed at
  `max|ratio/cap - 1| = 5.000e-01`.
- **RESOLVED 2026-08-13** — the `np.allclose` on the inflated diagonal now carries an explicit
  `rtol=1e-13, atol=0`. The implicit `atol=1e-8` sat only ~1.2% below the *smallest* diagonal
  entry in the fold (measured `min diag(cov) = 8.27e-07`, max `3.10e+00`), so it was a real
  undeclared floor; the achieved residual is `2.22e-16`.
- **RESOLVED 2026-08-13** — `test_mellin_systematic_truths_are_independent_and_seeded_by_power`
  is renamed `..._and_deterministic_per_order`. `build_moment_systematic_values` takes one
  `rng.standard_normal()` per spec with no functional dependence on the order's value, so the
  old name promised a relationship nothing asserted.
- **Open, and outside this file to fix**: PIXEL's `[0, x_min)` low-x head is wrong by
  `-9.18e-04` relative for the `alpha != 0` even-kernel case. `pixel.kernels.base.base_matrix`
  disables the closed-form head when `alpha != 0` and falls back to
  `low_x_quadrature_correction`, whose Gauss-Jacobi weight `t**gamma` does not absorb the
  `x**alpha` baked into the kernel, leaving a `1/t` singularity in the sampled integrand. For
  `Mellin(alpha=-1, low_x=power(1))` at `nu=1` the exact head is `1` and PIXEL returns
  `0.99908173` (measured directly on `low_x_quadrature_correction`). Every other basis column
  agrees with the independent rule to `1.19e-14`. The sine head at the same settings is exact
  to `2e-16` (`sin(nu x)/x` is regular at the origin), so only the even-kernel/Mellin case is
  affected. `test_nulled_counting_moment`'s column-0 bar is `1e-3`, just above the known
  error, so it still pins the Mellin order/alpha/power without asserting a `src/` bug fixed.
- `test_mellin_systematics_use_distinct_one_point_fields_and_unit`'s coefficient equality is
  still common-mode (both sides call `kernels.LATTICE_SYSTEMATICS[key][1]` with the same
  `meta`). That is a caveat rather than a defect — the test's target is the `datasets.py`
  wiring, and the factory formulas are independently pinned in `tests/test_coefficients.py`.
  The module docstring states the cross-reference.

- **RED, and not this file's defect: `test_generation_fold_caps_mellin_scalars_separately_from_itd_curves`.**
  Measured 2026-08-13, after `check_low_x_integrable` landed in
  `src/pixel/kernels/lowx.py` from a concurrent session. That guard is deliberately
  kernel-agnostic, so it refuses `Mellin(alpha=-1, low_x_extension="flat")` at **every**
  order -- including `n = 2` and `n = 3`, where the kernel's own `x**(n-1)` leaves a
  perfectly convergent head `int_0^x_min x**(n-2) dx = x_min**(n-1)/(n-1)` (measured
  `5.0e-13` at `n = 3`, `x_min = 1e-6`). Test 15 builds exactly that pairing because
  production does: `closure_*_truth/datasets.py::build_mellin` passes
  `momentum_density=True` for every field while taking `low_x_completion(field)`, and 26
  of the 117 Mellin lattice records in both full suites are `sigma`/`g` at `n = 3` with
  the flat completion. So the fixture mirrors production correctly and the guard has
  surfaced a **live production break**, not a test bug. Deliberately left failing: the
  resolution is either a kernel-aware criterion (the head weight at Mellin order `n` is
  `alpha + gamma + n - 1`) or `build_mellin` passing `momentum_density` per field --
  `src/` is another session's active file and `closure_*_truth/` is under the owner's
  hold. Recorded in `plans/test_audit/reports/test_kernel_guards.json`
  (`test_kernel_guards-02`).

**Not covered here**

- The systematic-coefficient *formulas* (`ht`, `chiral`, `inf_Lz`, `cont_invz`, `cont_aL`,
  `inf_L`) are independently tested in `tests/test_coefficients.py`
  (e.g. `test_chiral_factory_is_constant_vector`); this file only checks that the formulas are
  wired to the right key/meta/kernel, using the same factory on both sides of its own check.
- `lattice_layout()`'s real/imag component assignment by field C-parity is independently
  asserted in `tests/test_closure_pseudoitd_matching.py::test_closure_layout_uses_vector_real_imaginary_c_parity`,
  not in this file.
- `generate_member` itself — the orchestration that threads one truth source into the
  lattice, DIS and DY folds alike — is still never called by any test. Test 20 covers the
  `truth_curves=` branch it relies on, and `assemble_operator`, as pure functions; it does not
  cover `generate_member` choosing the right truth to pass them.
- **The small-`x` power of the real JAM/NNPDF closure curves.** Test 21's `a = 0.5` is a
  stand-in for a physical valence density, and the observable error goes as `x_min**a`, so
  that exponent *is* the answer for the seven `alpha=-1` fields. Both of test 21's bars are
  written as functions of `a` so re-pointing it is a one-line change, but nobody has read the
  power off the truth files. Recorded as the single largest open input in
  `plans/low_x_head_diagnosis.md`.
- **The missing integrability validation in `src/`.** `LowXExtension` checks
  `effective_power > -1` without ever seeing the caller's integrand weight `alpha`, so
  `flat` + `alpha = -1` is accepted everywhere and returns a finite `npts`-dependent number.
  Test 21 asserts `alpha + gamma > -1` on the production config, but that is a test-side
  guard on one config; the behaviour is pinned by
  `tests/test_kernel_guards.py::test_low_x_integrability_guard_rejects_a_divergent_pairing`
  and the proposed `src/` diff is in `plans/test_audit/reports/test_kernel_guards.json`.
