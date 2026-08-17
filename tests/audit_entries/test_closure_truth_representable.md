### `tests/test_closure_truth_representable.py`

**Exercises** `closure_JAM_truth.generate`/`closure_NNPDF_truth.generate` --
`fold_truth` and `dy_central` (byte-identical between the two copies) -- and, since
2026-08-14, `closure_JAM_truth_small.generate`/`closure_NNPDF_truth_small.generate`'s
`dy_central` too, plus the real `pixel.core.model.BilinearContribution.assemble`
`dy_central` calls (not a shim).

**Claim** Generation folds linear (`fold_truth`) and bilinear (`dy_central`)
observables from the same single PDF truth -- whichever `truth_kind` that is -- never a
replica-ensemble statistic — and folds it with the right index order, the right weights, and every
operator entry.

**Oracles** All three tests compare against plain `numpy` written directly in the
test — matrix-vector products for the linear folds, outer-product contractions for
the DY folds — independent of `gen`. Every one carries explicit
`assert not np.allclose(...)` controls measuring that its fixture actually separates
the bug class it names, rather than assuming it.

**Cost** 5 tests, **7 collected cases**; measured `7 passed in 0.71 s` (2026-08-14,
serial), against `13 passed in 0.73 s` immediately before.

**Three package lists since 2026-08-14, and the split is load-bearing.**
`LINEAR_PACKAGES` is `["closure_NNPDF_truth"]` (the `_small` suites have no
`fold_truth`); `DY_PACKAGES` is the NNPDF pair, one entry per *size*; and
`ALL_DY_PACKAGES` keeps all four packages for
`test_all_four_dy_central_bodies_are_the_same_source`, its only reader. The JAM legs came
out of the two behavioural parametrizations on the owner's instruction: all four packages
execute identical `dy_central` code, so those cases were the same fold run twice.
Trimming `ALL_DY_PACKAGES` to match would leave the two JAM copies unguarded, and a
silently divergent copy is exactly the S0-05 bug this file exists for -- so it must keep
all four. The identity assertion is also the *stronger* of the two guards: a behavioural
sweep sees a divergent copy only if the divergence changes what these fixtures measure,
which is why S0-05 survived a wide `package` list in the first place, while the AST
comparison sees any change to the body.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_linear_and_bilinear_generation_fold_the_same_fixed_truth` | `fold_truth(op, truth) == matrix @ truth["q"]`; `dy_central(...) == einsum("rij,i,j->r", tensor, truth["q"], truth["q"])`; the fixed-truth law differs from a replica-second-moment law on this fixture | direct numpy formulas (`A1`) | `assert_allclose`/`allclose` defaults (rtol 1e-7 / 1e-5) | `S2` |
| 2 | `test_linear_generation_fold_accumulates_a_multi_field_operator` | `fold_truth` sums `B_k @ truth[field_k]` over **every** operator entry, not just the first | three hand-summed matvecs (`A1`), plus drop-a-term and wrong-field controls | `rtol=1e-13` | `S1` |
| 3 | `test_bilinear_fold_is_index_weight_and_kinematics_faithful` | `dy_central` contracts `outer(A, B)` in that order, honours `weight`, groups by kernel identity, and threads `dataset.nu` and the shared basis into the kernel | summed outer products + `einsum` in numpy (`A1`), plus transpose / swap / unweighted / merged-group controls | `rtol=1e-13` | `S1` |
| 4 | `test_bilinear_fold_uses_the_ensemble_mean_not_the_ensemble_second_moment` | `dy_central` folds `outer(E[q], E[q])`, never `E[q q^T]`; and a 2-D (replica-stack) truth curve RAISES rather than being silently reduced | `einsum` on `replicas.mean(axis=0)` in numpy (`A1`); second-moment control on the same replicas, asserted `> 1e-2` apart so the fixture is non-degenerate; message-matched raise with a 1-D control | `rtol=1e-13`; guard matched on message | `S2` |
| 5 | `test_all_four_dy_central_bodies_are_the_same_source` | the four `dy_central` copies are one implementation -- `ast.dump` of the executable body, docstrings dropped | parsed-AST identity (`A3`), with a `fold_truth` control proving the normalization does not flatten distinct functions | exact | `S3` |

**Weak spots**

- **RESOLVED 2026-08-14 — `S0-05` is closed**, by
  `test_bilinear_fold_uses_the_ensemble_mean_not_the_ensemble_second_moment` and
  `test_all_four_dy_central_bodies_are_the_same_source`. Both `_small` copies of `dy_central`
  now run the full suites' fixed-truth fold verbatim (AST-identical across all four
  packages), their call sites pass `mean_curves` instead of the replica stack, and a
  new `_one_truth_curve` helper in **all four** copies refuses a non-1-D truth curve
  rather than letting `np.outer` flatten it. `DY_PACKAGES` parametrized both bilinear
  tests over all four packages — closing the package-level gap that let the bug live —
  and `test_all_four_dy_central_bodies_are_the_same_source` keeps the copies in
  lockstep. **Since 2026-08-14 that second guard carries it alone**: `DY_PACKAGES` is the
  NNPDF pair and `ALL_DY_PACKAGES` is the four-entry list the drift test reads. Measured law difference on the shipped ensembles at `Q = 2`: JAM
  `2.4e-03`/`2.9e-03`, NNPDF `2.1e-03`/`6.4e-03` `max|ratio-1|` (the file's synthetic
  7-8% is its own fixture, not the real ensembles). Acceptance, two mutations: the
  **verbatim** shipped `_small` `dy_central` restored in memory fails exactly the four
  `_small` parametrizations plus the drift guard while the full packages pass; a
  signature-preserving mutation dropping only the shape guard fails all four
  parametrizations of the new test on `DID NOT RAISE`. (Both mutation counts are against
  the pre-2026-08-14 4-entry `DY_PACKAGES` and were not re-measured; the mutations patch
  the packages, not the parametrization, and the first would now be caught by the drift
  test rather than by the `_small` parametrizations. Re-measure before quoting.) The fix is **truth-kind
  agnostic** — `dy_central` folds whatever single curve the caller recorded, so
  switching between `replica_ensemble_mean` and `fixed_lhapdf_member` needs no change
  here.

- **RESOLVED 2026-08-13** by `test_bilinear_fold_is_index_weight_and_kinematics_faithful`.
  Test 1's only bilinear case is `field_A == field_B == "q"`, which makes
  `outer(truth, truth)` symmetric and therefore blind to a transposed contraction
  (`"rji,ij->r"` for `"rij,ij->r"`) or a swapped `field_A`/`field_B`. The new test
  uses a deliberately asymmetric fixture — non-symmetric tensor, three contributions
  with `field_A != field_B`, distinct truths, three distinct non-unit weights, and two
  kernel *instances* so the `(id(kernel), id(evolution_A), id(evolution_B))` grouping
  must build two groups. Acceptance: in-memory mutations of `dy_central` implementing
  (a) `einsum("rji,ij->r")`, (b) `np.outer(curve_B, curve_A)`, (c) `weight = 1.0`, and
  (d) a single merged group each make it fail, while test 1 passes under all four —
  which is the original finding, demonstrated. Test 1 is kept because its expected
  value is readable by inspection.
- **RESOLVED 2026-08-13** by `test_linear_generation_fold_accumulates_a_multi_field_operator`
  for the recorded gap that `fold_truth`'s `for field, B in operator[1:]` loop was never
  entered. Acceptance: a `fold_truth` that returns only `operator[0]`'s term, and one that
  pairs every matrix with `operator[0]`'s field, each make it fail; test 1 passes under both.
  (`assemble_operator`, the production builder of that multi-entry operator, is covered in
  `tests/test_closure_extension_systematics.py::test_fold_uses_fixed_truth_curves_when_given_and_replica_mean_otherwise`.)
- The closing `assert not np.allclose(exact, second_moment)` in test 1 does not call
  `gen.dy_central` again: given the prior `assert_allclose(exact, expected)` already
  passed, it reduces to a fact about the fixture arrays (`expected != second_moment`),
  certifying that the earlier assertion *would* have failed under the alternative law
  rather than independently re-checking the implementation. The combined mechanism is
  sound; only the original comment's attribution overstated what this one line does in
  isolation — corrected in the docstring. See
  `plans/test_suite_hardening.md#test_closure_truth_representable-02`.
- Tests 1 and 3 both use stub bilinear kernels. Test 3's stub records and the test
  asserts `(nu, basis_a, basis_b)`, so the kinematics and basis are at least shown to be
  threaded; neither exercises a real kernel's tensor. Real kernels, evolution dispatch and
  weight scaling are covered in `tests/test_closure_dy_evolution.py`.

**Not covered here**

- ~~`closure_JAM_truth_small`/`closure_NNPDF_truth_small` are excluded from this
  file's `package` parametrization, and their `dy_central` computes the
  replica-second-moment law.~~ **Closed 2026-08-14** — see the first bullet under
  **Weak spots**. The owner lifted the hold on the four `closure_*` packages, both
  `_small` copies now run the full suites' fixed-truth fold, and `ALL_DY_PACKAGES`
  keeps the drift test over all four (`DY_PACKAGES`, which the bilinear tests use, is the
  NNPDF pair since 2026-08-14).
- **Neither truth *kind* is pinned by this file, deliberately.** All 24 committed
  `truthQ_*/truth.json` files carry `truth_kind: "replica_ensemble_mean"`, while the
  full suites' `generate.py` today writes `fixed_lhapdf_member`
  (`FIXED_TRUTH_MEMBER = 1`) — so the committed full-suite artifacts predate that
  change, and `plans/low_x_head_diagnosis.md` ADDENDUM 2 measures the ensemble-mean
  artifact as *unphysical* below `x ~ 1e-4` for JAM (396/791 replicas have a negative
  `x(q+qbar)` at `x = 1e-6`; the mean and median of `x*t8` have opposite signs).
  Choosing between the two kinds requires regenerating committed data and is the
  owner's call. Nothing here depends on it: `dy_central` folds whatever single curve
  the caller recorded.
- `generate_member` itself — the orchestration that actually wires one truth source
  into lattice, DIS and DY alike — is still never called by any test. Its
  `fold_lattice_systematics(truth_curves=...)` branch and `assemble_operator` are now
  covered as pure functions in
  `tests/test_closure_extension_systematics.py::test_fold_uses_fixed_truth_curves_when_given_and_replica_mean_otherwise`;
  what remains uncovered is `generate_member` choosing a consistent truth source to hand
  to all three folds, which needs `tmp_path` plumbing and a minimal manifest to redirect
  `cfg.truth_dir`.

**Phase-2 missing-test pass, 2026-08-14 — nothing added; both remaining items stay open.**

- `-M01` **is** `S0-05`, which `plans/test_audit/RESUME.md` records as the **owner's
  decision** rather than a defect and which the missing-test brief keeps inside the hold on
  the `closure_*_small` packages. Re-confirmed against this batch's own dispatch, which names
  it explicitly. Writing a test would bar one of the two competing central-value laws as
  correct and pre-empt that decision either way, so none was written. The "Not covered here"
  section above keeps the gap visible.
- `-M03` (an end-to-end `generate_member` test) stays open at effort `L`, but the remaining
  work is now narrower. A sibling batch closed the *same shape* of gap for the `_small`
  suites' `synthetic_z` block **without running the generator or writing any files** —
  `tests/test_small_synthetic_z.py::test_generation_time_synthetic_z_layout_is_what_the_manifest_reader_consumes`
  takes the real layout record, applies exactly the transform `generate_member` applies
  (`_public` plus `mean`/`cov`), and drives it through the reader, comparing the
  generation-time and fitting-time build routes as an `A3` identity. The same shape should
  work here: call `fold_lattice_systematics` / `fold_truth` / `dy_central` directly on the
  **same** `mean_curves` that `generate_member` hands each of them (`generate.py:668, 713,
  767`) and assert the three agree on their truth source. That drops the `tmp_path`/
  `cfg.truth_dir` plumbing which made this an `L`, and stays clear of the standing
  prohibition on running a closure pipeline.
