### `tests/test_closure_prior_ties.py`

**Exercises** `_gp_prior`/`build_analysis` in `closure_NNPDF_truth/fit.py` and
`closure_NNPDF_truth_small/fit.py` (plus `closure_JAM_truth_small` for one test -- see
**Cost**), each with its own `analysis.tie(params.mean.N, to=params.cov.sigma)` call.
Through those, `pixel.api.analysis.Analysis.tie`,
`pixel.core.model.Model.tie`/`parameter_table`/
`apply_parameter_overrides`, and `pixel.core.params.normalize_ties`/`apply_ties`/`ravel`.
The file imports no `pixel` symbol directly -- only `closure_*_truth.fit` modules.

**2026-08-15 update -- the `_small` suites left the two-tier scheme.**
`closure_*_truth_small/config.py` replaced `GP_AMPLITUDE_HIGH`/`LOW`/`HIGH_PRIOR_FIELDS`
with a per-field `GP_AMPLITUDES` table (`t3` 0.25, `t8` 0.5, `sigma` 5.0, `t15` 2.0, `g`
10.0, `v3` 0.25, `v8` 0.2, `v` 0.5, `v15` 0.5), after `t3` measured as the worst-covered
field in the suite (mean pull chi2/point 1.67 over 24 exp/both cases, 18 above 1, 6/6
positive in every Q member of both truths) while carrying a prior mean ~6x its own truth.
Three assertions here were tier-shaped and are now scheme-aware via `hasattr(cfg,
"GP_AMPLITUDES")`; the full suites still take the tier branch and are unchanged.

Two consequences worth recording, both measured rather than assumed:

* `GP_AMPLITUDE_FLOOR` is now `0.99 * min(GP_AMPLITUDES.values())` = 0.198, not the literal
  0.99, which would otherwise sit **above** six of the nine amplitudes -- the exact
  condition test 10 proves sends `eta` to `-inf`. A per-field *fractional* floor was tried
  first and rejected: 0.99 of each field's own value is a 1% leash that made three tie
  tests raise "value is below its lower bound" the moment they perturbed an amplitude.
* Test 3's sd bar moved `rtol=1e-9 -> 1e-8`. `diag(K) = sigma**2 + jitter` makes the
  relative excess `~jitter/(2 a**2)`, which grows quadratically as the amplitude falls:
  2.0e-12 at a=5.0, 5.0e-11 at a=1.0, 1.25e-9 at a=0.2. The old bar was set when the
  smallest amplitude was 1.0; at `v8` = 0.2 the observed sd was 0.20000000025 against 0.2
  and failed it. This is a jitter budget, not a physics tolerance.

Both changed assertions were mutation-checked: raising the floor to 0.99 fails test 10
alone, and pinning the hyper-prior `log_center` to a constant fails test 9 alone.

**Claim** The closure suites' `a +- a` field prior (mean tied to kernel sigma) is wired
correctly in the fit drivers, stays fully fixed by default, and
-- when floated -- gains a sane floor/hyper-prior, reaches the compiled `Model` as exactly one
free amplitude per field, and still carries an autodiff gradient through the tie to that
floated source.

**Oracles** Mostly A3 self-consistency: build a real `Model`, move one tied value through
`apply_parameter_overrides`, and check the tied value follows -- appropriate for pinning
wiring rather than physics. One test's total is closed-form (A1), independently verified
here against `pixel.core.params.LogNormal.neg2_logpdf` directly (agrees to 2.05e-06
relative). The general tie mechanism, including the specific "share one `Parameter` object"
trap this file's tie exists to avoid, is proven independently in `tests/test_parameter_ties.py`
(`test_shared_parameter_object_does_not_survive_ravel`) and `guides/tying_parameters.md`;
this file only checks the closure suites' *use* of that mechanism.

**Cost** 11 tests, **23 collected cases** (verified 2026-08-14 by `--collect-only -q`): ten
tests x 2 `SUITES`, plus `test_build_ties_mean_to_sigma` over 3.

`SUITES` is **NNPDF truth only** as of 2026-08-14, on the owner's instruction. It was
`(full_jam, full_nnpdf, small_jam, small_nnpdf)` and the file ran every test against all
four independently maintained copies. Nothing in this file reads the truth PDF set -- every
oracle is the suite's own `cfg` constants and the object graph `build_analysis` assembles --
so each JAM leg re-ran the identical `normalize_ties`/`apply_ties` path for an answer that
differed only in which copy of the constants it came from. What is given up is the per-copy
drift guard on the two JAM `config.py`/`fit.py` files. One JAM case is kept as the canary
for that: `small_jam` in `_TIE_REBUILD_SUITES`, which is deliberately **not** `SUITES` and
carries a comment saying so. (The *full* JAM leg had already been dropped from that one test
on 2026-08-14 as the file's single most expensive case, 284.6 s against NNPDF's 230.2 s.)

Whole file **14.93 s warm** (measured 2026-08-14, serial, `23 passed`), against
**21.05 s / `43 passed`** on the same box immediately before the JAM legs came out.
(Two intervening runs read 224.35 s and 222.87 s with 1,638 `kernel source fingerprint
mismatch` warnings apiece -- a peer session had clobbered the shared kernel cache. Both
still reported `23 passed`; neither is a duration measurement. This is the same trap as
the cold-cache paragraph below, and it can open *between* two runs of the same file.)
No markers hand-set
(duration-derived). Three tests build a real `Model` via `build_analysis`; measured
2026-08-13 with a warm kernel cache they reach 3.6-5.7s each for a full suite and 0.1-0.8s
for a `_small` one. (`tests/test_durations.json` records
264s/217s for `test_build_ties_mean_to_sigma`; that was a cold-cache run and is not what the
test costs in a normal session -- measured 5.69s warm. That file still carries entries for
the removed JAM node ids; `conftest.py` looks durations up by node id and simply misses
them, so they are inert until the next full regeneration.) Every other test touches only
`cfg`/`_gp_prior` and runs under 0.4s.

**A cold-cache run is 25x the warm one, and it is not a regression.** Measured 2026-08-14:
the same file took **over 27 minutes** and was still short of the end when the run was
killed, against 20.07 s warm immediately afterwards. The cause is named in the run's own
output -- `LOUD WARNING: kernel cache ... could not be reused (kernel source fingerprint
mismatch); recomputing and overwriting it`, 1,297 of them -- i.e. a concurrent session had
edited `src/` kernel math, so every cached matrix was rebuilt once. Cold case costs in that
window: `test_nuisance_fields_are_not_tied[closure_JAM_truth.fit]` ~3.8 min,
`test_floating_the_amplitude_reaches_the_compiled_model[closure_JAM_truth.fit]` 156.39 s
against 0.5 s for the `_small` pair. (Both node ids are from the pre-2026-08-14
parametrization and no longer exist; the NNPDF cases they pair with behave the same way,
as the 224 s runs above re-confirmed.) **Never take a duration measurement during a
rebuild**, and never read a
killed run as a pass: that one exited **144 with no summary line**, 35 of 43 cases having
reported (0 failed), and the remaining 8 were measured separately.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_one_amplitude_drives_both_mean_and_sigma` | `gp_mean`/`gp_sigma`/`gp_amplitude` agree, **and both describe the values `_gp_prior` really hands to `Const`/`LogRBF`**; 4 retired constants are gone | aliases against the built prior object (`A3`) | exact `==` -- the strictest bar in the file; catches a 1e-11 amplitude drift that the `rtol=1e-8` sd bar in test 3 absorbs | `S3` |
| 2 | `test_high_tier_is_singlet_gluon_charm` | `{sigma, g, t15}` is the high-amplitude tier; 9 fields total | documented tier list (`D1`) | `==` | `S3` |
| 3 | `test_gp_prior_starts_at_a_plus_minus_a` | `_gp_prior`'s mean/sd both equal the amplitude, at construction | same shared call, no tie in this path (`A3`) | rtol 1e-12 / 1e-9 | `W1` |
| 4 | `test_build_ties_mean_to_sigma` | moving `cov.sigma` through the real model moves `mean.N` with it | rebuild reruns `apply_ties` (`A3`) | rtol 1e-12 / 1e-9 | `S3` |
| 5 | `test_nuisance_fields_are_not_tied` | lattice nuisance fields carry no tie and no mean parameter, **and their presence matches whether the suite declares `nuisance_specs`** (no more unconditional skip) | structural key-set (`F1`) | exact | `S3` |
| 6 | `test_prior_stays_fully_fixed` | `N_HYPERPARAMS==0`, `FROZEN_COV_PARAMS` pinned | cfg constants (`F1`) | `==` | `S3` |
| 7 | `test_amplitude_is_frozen_and_unpenalised_by_default` | default amplitude is frozen, prior-free, floored | `Parameter` metadata (`F1`) | `==` | `S3` |
| 8 | `test_floor_sits_strictly_below_every_starting_amplitude` | floor sits under every tier; inv-softplus stays finite | closed form (`A1`) | finite | `S3` |
| 9 | `test_floating_attaches_the_floor_and_a_centred_hyper_prior` | floated sigma gets floor + centred hyper-prior; total matches closed form | structural + shape + closed form (`F1`/`A1`) | closed form evaluated from the suite's own amplitudes (was the literal `13.7209`) | `S3` |
| 10 | `test_floating_the_amplitude_reaches_the_compiled_model` | with `GP_AMPLITUDE_FREE` on, `build_analysis` yields `n_free == 9` whose free selectors are exactly the nine `cov.sigma`, the tie survives, `model.bounds()` carries the floor on every row, and `jax.grad` through `model._rebuild` of `sum mean.N**2` returns `2*sigma` | structural (`F1`) + closed form (`A1`) on the bounds and the gradient | `rtol=1e-12`; measured `max\|grad/(2 x0) - 1\| = 0.0` | `S2` |
| 11 | `test_prior_and_accessors_all_follow_one_patched_amplitude` | with `cfg.gp_amplitude` monkeypatched to nine **distinct** per-field probe values, `gp_mean`/`gp_sigma` **and** both numbers inside the built prior (`mean.N`, `cov.sigma`) all follow it, per field | one patched source, exact `==` (`A3`) | exact `==`; probes asserted distinct from each other **and** from both tier constants, so an accessor ignoring the patch cannot pass | `S2` |

(Rows are in the order they were added, not file order; test 11 sits directly after test 1
in `tests/test_closure_prior_ties.py`.)

**Weak spots**

- PARTIALLY RESOLVED 2026-08-13. `gp_mean`/`gp_sigma` are still orphaned -- re-verified by
  repo-wide grep: no caller anywhere except their own definitions in the four `config.py`
  copies and this test, while `_gp_prior` (fit.py:122) calls `cfg.gp_amplitude` directly.
  Deleting them (or wiring `_gp_prior` to call them) is a source change inside the owner's
  held closure packages, so it stays open. What the test no longer does is pin the two orphans
  against each other *only*: it now requires `_gp_prior(name).mean.N.value` and
  `.cov.sigma.value` to equal them exactly, so the accessors describe the prior that is
  actually built. Acceptance: a 1e-11 relative perturbation of the amplitude in `_gp_prior`
  fails this test alone (`assert 1.00000000001 == 1.0`), while test 3's `rtol=1e-8` sd bar
  absorbs it. See `plans/test_suite_hardening.md#test-one-amplitude-drives-both-mean-and-sigma`.
- **2026-08-14, `test_closure_prior_ties-02`: the `tests/`-side half is now CLOSED by test 11;
  the source-side half is RE-CONFIRMED against current source and stays open as a maintainer
  edit.**
  *Re-verification.* `gp_mean` and `gp_sigma` are each still exactly
  `return gp_amplitude(field)`, at `closure_JAM_truth/config.py:1000-1002` and `1005-1007`,
  `closure_NNPDF_truth/config.py:998-1000` and `1003-1005`,
  `closure_JAM_truth_small/config.py:717-719` and `722-724`, and
  `closure_NNPDF_truth_small/config.py:720-722` and `725-727`. A repo-wide grep for
  `gp_amplitude(` finds exactly two kinds of production caller: `fit.py:122`
  (`value = cfg.gp_amplitude(field)`, identical in all four suites) and each `config.py`'s own
  `gp_amplitude_prior`. Neither accessor is called by either. **Zero production callers,
  unchanged.** Re-verified a third time on 2026-08-14 (evening) with every line number
  identical, and this pass the choice of resolution was *measured* rather than argued --
  see the next bullet, which retires one of the two options.
  *What test 11 closes, and it is not cosmetic.* Every existing check compares
  `cfg.gp_mean(name)` to `cfg.gp_amplitude(name)` -- one function called twice under two
  names -- so it passes whether the accessor *delegates* or *re-implements the tier table by
  hand and happens to agree today*. Those are the same assertion and very different code; the
  second is a silent second source of truth, which is the arrangement
  `plans/lambda_guard_and_doc_defects.md` 5b/5c records going stale twice in these very
  files. Patching `cfg.gp_amplitude` separates them. *And it breaks a fixture degeneracy
  nothing else in the file breaks*: `gp_amplitude` returns only **two** distinct numbers
  across nine fields in the **full** suites (5.0 for three, 1.0 for six), so every other
  per-field check there is blind to a within-tier cross-wire -- `_gp_prior("t3")` reading
  `v3`'s amplitude satisfies test 3 exactly, both being 1.0.  The `_small` pair no longer
  has that degeneracy (per-field table, 2026-08-15) but `t3` and `v3` are both 0.25, so the
  cross-wire it was written for survives there. Nine distinct probe values make each field's prior traceable
  to its own amplitude, and the same patch simultaneously pins that `_gp_prior` reads the
  accessor *at call time* rather than closing over an import-time snapshot.
  *Acceptance*: two in-memory mutations, each preserving every value the suites produce today
  and changing only where the value comes from -- so the file's pre-existing assertions all
  still pass and only test 11 can see the difference. `gp_mean_duplicates_tier_table`
  (`gp_mean`/`gp_sigma` replaced by a hand-copied `GP_AMPLITUDE_HIGH if field in
  HIGH_PRIOR_FIELDS else GP_AMPLITUDE_LOW`) and `gp_prior_inlines_tier_table` (`_gp_prior`
  rebuilt with that same inlined tier table in place of `cfg.gp_amplitude(field)`), both
  applied to all four suites, with a plugin self-check that a probe patch of `cfg.gp_amplitude`
  is genuinely *not* followed afterwards. **Measured 2026-08-14: both give `4 failed, 39
  passed`** — only the four cases of test 11 fail, every other test in the file passes
  including the three `build_analysis` ones (`gp_prior_inlines_tier_table` 21.82 s,
  `gp_mean_duplicates_tier_table` 20.57 s; unmutated baseline on identical file content
  `43 passed in 20.07 s`). The failures are the intended ones: `assert 1.0 == 2.5` at line 204
  (the prior's `mean.N` carries the inlined tier value rather than the patched amplitude) and
  `cfg.gp_mean('t3') == 1.0` against a probe of 2.5 at line 200. **The decisive half**: under
  `gp_mean_duplicates_tier_table`, test 1 *passes* — its `cfg.gp_mean(name) == amplitude` and
  `prior.mean.N.value == cfg.gp_mean(name)` cannot tell a delegating accessor from a
  hand-copied tier table. That is the blindness test 11 removes, demonstrated rather than
  argued. (Every count in this item is against the pre-2026-08-14 43-case parametrization
  and was **not** re-measured when `SUITES` was trimmed to the NNPDF pair. The mutations
  patch the closure packages, not the parametrization, so they still bite; what changes is
  the arithmetic -- the two JAM cases of test 11 that used to fail are no longer collected.
  Re-measure before quoting a failed/passed pair from this item.)
- **2026-08-14 (evening), the source-side half of `-02`: one of the two options is measured
  to be a non-fix, so this is now a one-option decision for the owner.** Option 2 -- point
  `fit.py:122` at the accessors -- **does not restore a consumer for `gp_mean`**.
  `build_analysis` ties the mean to the covariance amplitude
  (`analysis.tie(params.mean.N, to=params.cov.sigma)`, `fit.py:228` in both full suites,
  `fit.py:179` in both `_small` ones), and a tie **discards** the target's declared value at
  compile time rather than constraining it afterwards. Measured
  (`pixel.Analysis` with `priors.Const(N=7.0)` and `priors.RBF(sigma=2.0, length=0.3)`, tie,
  read back through `model.parameter_table()`): `mean.N` = **2.0**, `cov.sigma` = 2.0,
  `tied_to = "fields.q.prior.cov.sigma"`, `frozen = True` -- the declared 7.0 is gone. The
  same behaviour is already asserted in-repo by `tests/test_parameter_ties.py::
  test_compiled_model_reports_the_tie_and_drops_one_coordinate` (declared 1.0, inherited
  2.0). So under option 2 the number `gp_mean` returns would be handed to `priors.Const` and
  overwritten before any fit saw it: the accessor would gain a **call site** and still have no
  consumer -- worse than today's orphan, because it would *read* as wired while a `gp_mean`
  that drifted from `gp_sigma` still changed nothing anyone fits. That is the same
  non-detectability this weakness records, and it turns the earlier design objection (option 2
  re-creates the two-constants arrangement `analysis.tie` replaced) into a measured one.
  **So: delete both accessors**, at the eight line ranges above. Collateral, checked: in
  `tests/`, drop the two `==` asserts in test 1 (`:129-130`) and the two in test 11
  (`:211-212`), and re-point test 1's `prior.mean.N.value == cfg.gp_mean(name)` /
  `.cov.sigma.value == cfg.gp_sigma(name)` pair (`:135-136`) at `cfg.gp_amplitude(name)`.
  **Nothing operational is lost**: every remaining assertion in both tests is against
  `_gp_prior`'s output, and test 11's other three claims -- call-time reading, per-field
  traceability, and one number reaching both `Const` and `LogRBF` -- are asserted through the
  built prior, not through the accessors. Outside `tests/`, two prose references go stale in
  the same pass: `plans/closure_plan.md:307` and the `gp_mean`/`gp_sigma` columns of the table
  at `plans/lambda_guard_and_doc_defects.md:258-280`. No docs impact -- the closure packages
  have no `docs/source/apidoc` entry and no `.rst` names either accessor. Test 1's docstring
  now carries this measurement, so the next reader meets it at the assertion rather than in a
  report. **The item stays open**: the edit is inside the owner's hold. File after the
  docstring update: **23 passed in 15.78 s**, and 78 passed with `tests/test_parameter_ties.py`
  alongside.
- **W-COMMON** (S3, documented not fixed) `test_gp_prior_starts_at_a_plus_minus_a` checks
  `_gp_prior`'s mean and sd once, at construction, before `analysis.tie` is ever attached
  (`_gp_prior` never calls it): both values share one construction-time source and nothing in
  that test enforces they stay equal afterwards. The companion `test_build_ties_mean_to_sigma`
  is the file's decisive answer, and that is now measured rather than argued: deleting the
  `analysis.tie` call from `build_analysis` in all four suites fails
  `test_build_ties_mean_to_sigma` (`assert 0 == 9` on `len(model.ties)`) and the new test 10,
  while `test_gp_prior_starts_at_a_plus_minus_a` passes. The construction-time test keeps its
  place -- it is the only check on the starting values -- and its docstring says what it does
  not establish. See `plans/test_suite_hardening.md#test-gp-prior-starts-at-a-plus-minus-a`.
- RESOLVED 2026-08-13 (was: `test_nuisance_fields_are_not_tied` unconditionally `pytest.skip`ed
  for 2 of its 4 suites). The skip is now
  `assert bool(nuisance) is hasattr(cfg, "nuisance_specs")` followed by an early return, so
  the `_small` cases assert something real and a full suite that silently lost its nuisance
  registration fails instead of turning green. This repo has no CI, so a permanent skip was
  total absence, not a fallback. Acceptance: stubbing `cfg.nuisance_specs` to `[]` fails the
  two full-suite cases (`assert False is True`); under the old form they would have skipped.
- Every oracle in this file remains internal to the closure suites plus `pixel.core.params`.
  Nothing here checks that `5.0`/`1.0` are the *right* prior widths, or that the `a +- a`
  convention is the right one -- only that it is wired as documented.

**Not covered here**

- RESOLVED 2026-08-13: floating `GP_AMPLITUDE_FREE` through `build_analysis` is now covered by
  test 10, including the freeze loop's own guard against re-freezing the amplitude
  (`fit.py:229-235` in the full suites, `:180-186` in the `_small` pair -- character-identical
  in all four) and a real autodiff gradient reaching the floated source through the tie.
  One thing test 10 deliberately does **not** assert: `n_free == cfg.N_HYPERPARAMS`.
  `N_HYPERPARAMS` is evaluated at import time from the module-level `GP_AMPLITUDE_FREE`, so a
  monkeypatched flag cannot move it; that assertion would have been unable to fail. It is
  compared to `len(cfg.ALL_FIELDS)` instead, and `test_prior_stays_fully_fixed` pins the
  default branch.
- Rejecting `op: "set"`/`"thaw"` directly on a tie target (`fields.NAME.prior.mean.N`) is
  covered generically in `tests/test_parameter_ties.py::test_thaw_and_set_on_a_tie_target_raise`,
  not re-tested here on a real closure model -- reasonable, since this file's
  `test_build_ties_mean_to_sigma` already proves `apply_parameter_overrides` works correctly
  on the tie *source* for a real model, and the generic test proves the target-side guard.
- No test here runs a fit or evaluates the evidence, so the *consequence* the frozen amplitude
  exists to prevent -- a fitted tied amplitude collapsing onto its lower bound when the data
  cannot see the field, measured at 24% of draws at low S/N -- is not reproduced anywhere in
  this file. Test 10 checks that the floor is present in `model.bounds()` when floated, which
  is the wiring half of that story, not the statistical half.
