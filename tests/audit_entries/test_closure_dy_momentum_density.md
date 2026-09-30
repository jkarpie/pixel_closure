### `tests/test_closure_dy_momentum_density.py`

**Exercises** every `closure_*/*.py` module that calls `pixel.data.DrellYan` (a static AST
scan of the eight suites: the five `datasets.py` builders in each full / `_data` suite, the
one `build_drell_yan` in each `_small` suite, and `closure_NNPDF_truth_small/nnlo_dy_smoke.py`);
and, for the two `_small` suites, the real `config.make_fields`, `config.dy_field_maps`,
`datasets.build_drell_yan` and `generate.dy_central`, reaching
`pixel.kernels.drell_yan.kernel.DYKernel` with `momentum_density=True`.

**Claim** Closure fields are momentum densities `x f`, and every closure Drell-Yan builder
says so; the `_small` suites' generated DY central value is the leading-order cross section
of the number densities `f = (x f)/x`. Before the PIXEL kernel fix (2026-09-13) every closure
DY row was `x1 x2` times that, identically on the generation and fit sides, so no closure
fit could see it.

**Oracles**
- Test 1 is `F1`: the keyword `momentum_density=True` appears literally in each call.
- Test 2 is `A1`: the closed-form LO formula
  `4 pi alpha^2/(9 Q^2 S) sum_q e_q^2 [f_q(x1) fbar_q(x2) + fbar_q(x1) f_q(x2)]` at
  `x1,2 = sqrt(tau) e^{+-Y}`. Its parton densities come from the suite's own
  `dy_field_maps` and the same field interpolants the fit uses, but none of the kernel,
  channel or contraction code. It holds only while the suites are LO and unevolved, which
  the test asserts instead of skipping.

**Cost** 2 test functions, 10 cases; about one second on 2026-09-13 (LO Born rows need no
`z` integral).

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_every_closure_drell_yan_builder_declares_momentum_densities` | each `DrellYan(...)` call in the suite passes `momentum_density=True` | AST scan (`F1`) | literal `True` | `S3` |
| 2 | `test_small_suite_fold_is_the_leading_order_cross_section_of_number_densities` | `dy_central(build_drell_yan(E866-like pp rows))` == closed-form LO cross section, both `_small` suites | `A1` | `abs(r-1) < 1e-9` | `S1` |

**Mutation (2026-09-13).** A throwaway plugin forcing `momentum_density=False` into every
`DrellYan` call makes test 2 fail with ratios `0.016656` and `0.039973`, exactly `Q^2/S` on
the two rows. Test 1 is static and unaffected by that mutation; deleting the keyword from
any builder fails it.

**Weak spots**
- Test 2 covers the `_small` suites' pointwise photon rows only. The full suites'
  collider (gamma*/Z, running couplings) and evolved rows are covered only by test 1's
  keyword check, not by a value.
- Whether the full suites' DY evolution projections map `x f(Q0)` to `x f(Q)` is not
  established here.
