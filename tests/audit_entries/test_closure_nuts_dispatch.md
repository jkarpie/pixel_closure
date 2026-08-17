### `tests/test_closure_nuts_dispatch.py`

**Exercises** the private sampler-dispatch helpers of the NNPDF closure-suite drivers --
`closure_NNPDF_truth.fit` and
`closure_NNPDF_truth_small.fit` (`_closure_sampler`, `_markov_chain`, `_autocorrelation_ess`,
`_weights`, `_dis_predictions`, `_finite_or_none`, `summarize`, `run_fit`'s source text) plus
`main`/`build_kernels_only` in their two sibling `run_closure` modules. The two
`closure_JAM_truth*` drivers were parametrized alongside them until 2026-08-14 (see
**Cost**). None of these live
under `src/pixel` -- this file imports nothing from `pixel` directly -- but every driver it
drives does (`import pixel`, `pixel.infer`, `pixel.map`), and `_autocorrelation_ess` calls
straight into `src/pixel/infer/gamma_method.py`'s `autocorrelation_summary`/`_resolve_chain`.

**Claim** Full-suite closures (`full_nnpdf`) dispatch bilinear/nonlinear models to
the right one of NUTS/HMC/direct-NUTS by inspecting the model; small-suite closures
(`small_nnpdf`) always use a VEGAS family, configured differently for linear vs.
bilinear/DY models; the reported autocorrelation ESS comes from the real Markov chain, never a
contour's complex `f`; DY-only layouts never enter bilinear MAP; and the JSON summary keeps two
distinct ESS figures under distinct keys.

**Oracles** Not physics oracles -- there is no accuracy floor to state. Four kinds recur:
(1) branch dispatch on a hand-built `SimpleNamespace` standing in for `Model`, checked against
the real conditional read out of the source; (2) literal substrings of
`inspect.getsource(suite.run_fit)`/`inspect.getsource(runner.main)`, which pin the *text* of a
call site and cannot tell live code from the same text in a comment or dead branch; (3) direct
reads of `suite.cfg.<NAME>` against `closure_*_truth[_small]/config.py`, where provenance for
those numbers is recorded; and, added 2026-08-13, (4) driving `run_fit` with its collaborators
stubbed until it calls the sampler and asserting on the real call (`_capture_sampler_call`) --
the only kind here that distinguishes live code from matching text, and the one that makes the
small-suite claims falsifiable at all. The one test with a genuine analytic bar
(`test_autocorrelation_ess_falls_below_the_sample_count_when_mixing_is_slow`) uses the closed-form
AR(1) autocorrelation time; the Wolff estimator itself is independently pinned in
`tests/test_gamma_method.py`, not duplicated here.

**Cost** 23 test functions, **39 cases**; measured `39 passed in 0.72 s` (2026-08-14,
serial), against `78 passed in 0.76 s` immediately before.

The JAM legs came out on the owner's instruction 2026-08-14, removing exactly half the
cases -- 39 of 78, the largest case count of any file in that change and, at ~0.04 s,
essentially none of its time. `diff closure_JAM_truth/fit.py closure_NNPDF_truth/fit.py`
shows two docstring lines and nothing executable; the same holds for the `_small` pair and
the four `run_closure.py` copies. Every JAM case therefore re-ran an identical dispatch
branch, an identical `inspect.getsource` substring match, or an identical stubbed `run_fit`
call. No test in this file compares one suite to another, so nothing needed a second tuple.
The 12 inline `(full_jam, full_nnpdf)` / `(small_jam, small_nnpdf)` parametrize tuples were
replaced by two named module constants, `FULL_SUITES` and `SMALL_SUITES`, so the
full-vs-small axis -- the one that is a real difference -- is now visible at each call site
instead of being spelled out twelve times. No recorded duration in
`tests/test_durations.json` under the current filename -- the file's rename (see below)
orphaned whatever durations existed under the old name. Every test builds lightweight mocks,
calls `inspect.getsource`, or drives `run_fit` with `build_analysis`/`_iterate_t0`/the two
inference entry points stubbed; none compiles a model, builds a kernel, or runs JAX.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_bilinear_closures_use_public_nuts_family` | pure-H-S (`n_ordinary=0`) full-suite model dispatches to plain NUTS | branch dispatch on a `SimpleNamespace` model (`F1`) | exact string | `S3` |
| 2 | `test_exact_ratio_closures_use_explicit_direct_nuts` | `has_nonlinear` dispatches to `direct_nuts`; `DIRECT_PRIOR_RCOND` pinned exactly; `run_direct_mcmc` named in the source | dispatch + cfg read + source substring (`F2`) | exact string; `approx(1e-10, rel=1e-12)`; substring | `S3` |
| 3 | `test_full_mixed_linear_normalization_and_hs_closures_use_joint_hmc` | bilinear + `n_ordinary=5` dispatches to `hmc` | branch dispatch, paired with #1 (`F1`) | exact string | `S3` |
| 4 | `test_small_dy_mode_uses_joint_affine_vegas` | the small dispatcher is constant *by design* (asserted against deliberately invalid input); 4 VEGAS constants | explicit constant-contract + direct `cfg` reads (`D1`) | `128`/`1`/`True`/`approx(0.05, rel=1e-12)` | `S3` |
| 5 | *(removed 2026-08-13)* `test_small_mixed_mode_samples_explicit_normalizations_with_joint_vegas` -- both `_closure_sampler` lines were tautologies and both source substrings were already asserted verbatim by #10 on the same source object; the modes it named are now driven for real by #19 | -- | -- | -- |
| 6 | `test_full_linear_closures_use_nuts_without_a_contour` | bilinear-free full-suite model dispatches to `nuts`, with tuning pins | dispatch + 6 `cfg` reads (`F2`) | `100(loose)`/`0.1`/leapfrog/`6`/`False`/`1000` | `S3` |
| 7 | `test_small_linear_closures_use_ordinary_vegas` | bilinear-free small-suite model uses ordinary VEGAS, with tuning pins | tautological dispatch + 4 real `cfg` reads (`D1`) | `16`/`5`/`128`/`1000` | `W1` |
| 8 | `test_contour_saddle_tolerance_covers_observed_float_stall` | `CONTOUR_SADDLE_TOL` clears the documented 3e-3 float-stall floor | direct `cfg` read vs. a hardcoded range whose floor matches a `config.py` comment (`D1`) | `3.0e-3 <= x <= 1.0e-2` | `S3` |
| 9 | `test_full_bilinear_closures_enable_only_the_two_setup_caches` | contour NUTS caches only saddle reference + Hessian mass | 6 source substrings, 2 of them negative (`F2`) | substring presence/absence | `S3` |
| 10 | `test_small_bilinear_closures_configure_joint_affine_vegas` | `run_fit`'s bilinear branch builds nested-VEGAS options, not MCMC ones | 8 source substrings + 1 direct `cfg` read (`F2`) | substring presence/absence; `approx(1.0)` | `S3` |
| 11 | `test_no_dis_t0_datasets_skip_map` | empty `correlated_systematics` layouts skip `pixel.map` entirely | monkeypatch trap raises if `pixel.map` is called (`F1`) | return `{}`; zero trapped calls | `S3` |
| 12 | `test_autocorrelation_ess_routes_through_the_real_markov_history` | ESS is measured on `.chain`, never a contour's complex `f` | identity (`is chain`) + label provenance, both demonstrated discriminating (`A3`) | identity; label membership | `S3` |
| 13 | `test_autocorrelation_ess_declines_non_markov_inputs` | importance draws / fixed posteriors report no autocorrelation | two distinct `_markov_chain` guard clauses (`F1`) | identity; exact dict | `S3` |
| 14 | `test_small_closure_reads_nested_vegas_signed_weights` | `_weights` prefers `normalized_signed_weights` first | exact-echo `assert_allclose` (`A3`) | default rtol/atol | `S3` |
| 15 | `test_autocorrelation_ess_falls_below_the_sample_count_when_mixing_is_slow` | ESS/`tau_int_max` respond to a badly-mixing AR(1) chain | closed-form AR(1) autocorrelation time (`A1`) | `ess<0.2n`; `tau_int_max>5.0` | `S1` |
| 16 | `test_summary_reports_both_effective_sample_sizes` | summary keeps 2 ESS figures under distinct keys reading distinct sources | 2 source substrings pin the exact dict-value expressions (`F2`) | substring presence | `S3` |
| 17 | `test_suite_runner_records_case_failures_and_continues` | `main` catches per-case exceptions, writes 2 output files, honors `--fail-fast` | 4 source substrings (`F2`) | substring presence | `S3` |
| 18 | `test_suite_kernel_only_mode_builds_without_fitting` | `--kernels-only` builds every `(Q, mode)` cache and never fits | real `runner.main()` execution with monkeypatch traps (`F1`) | exact list; zero trapped calls | `S3` |
| 19 | `test_small_run_fit_configures_vegas_from_the_model_not_the_mode` | for all three modes, `run_fit` reaches `infer_mcmc` with `sampler="vegas"`, `auxiliary` following `has_bilinear`, and exactly the nested option keys for a bilinear model / exactly the ordinary ones for a linear one | `run_fit` driven to its real inference call (`F1`) | exact key sets and values | `S1` |
| 20 | `test_full_run_fit_routes_each_model_to_its_own_sampler_options` | each of the four full-suite dispatch branches reaches inference with its own option block, through the right entry point (`direct_nuts` alone via `pixel.run_direct_mcmc`), and the two contour caches appear on the bilinear-NUTS branch only | `run_fit` driven to its real inference call (`F1`) | exact entry point, keys and values | `S1` |
| 21 | `test_dis_t0_predictions_are_sliced_per_dataset` | MAP runs once; each DIS table gets its own `layout.data_slice`; a table without `correlated_systematics` is absent from both the result and the slicing | index-valued prediction vector (`F1`) | exact arrays | `S2` |
| 22 | `test_markov_chain_rejects_a_history_it_cannot_analyze` | a chainless **complex** history, a 1-D history and a single-row history are all rejected; a real history of the same shape is accepted | direct calls on the guard's own inputs + control (`F1`) | `is None` / `is not None` | `S3` |
| 23 | `test_effective_sample_size_dispatches_on_the_sample_type` | `PosteriorResult` -> `1.0`; `ess_frac` carrier -> `600*0.25 = 150`; fallback -> `1/sum|w|^2 = 1/0.875` | arithmetic on the fixture (`A1`) | `rel=1e-12` | `S2` |
| 24 | `test_small_effective_sample_size_prefers_the_nested_vegas_signed_ess` | a `NestedVegasSamples` carrying **both** `signed_ess` and `ess_frac` reports `signed_ess` -- the branch-order claim | arithmetic on the fixture (`A1`) | `rel=1e-12` | `S2` |

**Weak spots**

- **RESOLVED 2026-08-13 by `test_small_run_fit_configures_vegas_from_the_model_not_the_mode`.**
  `test_small_dy_mode_uses_joint_affine_vegas`, the now-removed
  `test_small_mixed_mode_samples_explicit_normalizations_with_joint_vegas`, and
  `test_small_linear_closures_use_ordinary_vegas` each asserted
  `suite._closure_sampler(model, mode) == "vegas"` for the small-suite drivers, but
  `closure_JAM_truth_small/fit.py:194-202` defines `_closure_sampler(model, mode=None): return
  "vegas"` unconditionally -- verified directly, `suite._closure_sampler(None, None) ==
  "vegas"` holds for garbage input too. Those assertions could not fail for any model or mode,
  and the elaborate mocks built for two of them (`contour_partition`,
  `_bilinear_normalization_design`) were read by nothing. Both halves of the recorded fix are
  taken: the dispatcher's constancy is now asserted *as* a constant contract, against
  deliberately invalid input, and the real per-mode claim has moved onto `run_fit`, driven to
  its actual `infer_mcmc` call. Acceptance: a small dispatcher that returns `"nuts"` for mode
  `"dy"` -- a pure function replacement, no source recompilation, so `inspect.getsource` is
  untouched -- fails **only** the new test (`2 failed, 76 passed`; counts predate the
  2026-08-14 JAM trim -- re-measure before quoting); all four original
  small-suite tests pass under it. A second mutation dropping the `if model.has_bilinear:`
  gate in `run_fit` also fails it.
- **RESOLVED 2026-08-13 by deletion.**
  `test_small_mixed_mode_samples_explicit_normalizations_with_joint_vegas`'s two real
  assertions (`inspect.getsource(run_fit)` substrings) were a strict subset of
  `test_small_bilinear_closures_configure_joint_affine_vegas`'s eight checks on the same
  source object, so with the tautology set aside the test added nothing independent. It is
  removed; the "exp"/"both" modes it named are parametrized cases of the new runtime test.
- **RESOLVED 2026-08-13 by `test_markov_chain_rejects_a_history_it_cannot_analyze`.**
  `test_autocorrelation_ess_routes_through_the_real_markov_history`'s docstring originally
  framed "handing the complex array to the Gamma method" as the one mistake it catches;
  measured directly, that framing overreached two ways: (a) `_markov_chain`'s own
  `np.iscomplexobj`/weights guard is never exercised on a *chainless* complex object by this
  fixture -- a mutant with that guard deleted but `.chain`-preference intact returns the
  identical result -- and (b) feeding the complex array straight to
  `pixel.infer.gamma_method.autocorrelation_summary` raises `ValueError` today (an independent
  guard in `_resolve_chain`), so the failure mode is currently loud, not silent. The docstring
  was corrected in the audit pass; the guard itself is now fed the inputs that trip it.
  Acceptance: deleting the `ndim != 2 or shape[0] < 2 or iscomplexobj` line fails only the new
  test (`4 failed, 74 passed`).
- The `DIRECT_PRIOR_RCOND` bound was one-sided and sat exactly on the value it bounded, so it
  could catch a loosening but not drift to a different tight value. Now pinned with
  `pytest.approx(1.0e-10, rel=1e-12)`. Acceptance: `DIRECT_PRIOR_RCOND = 1e-11` fails it
  (`2 failed, 76 passed`); the old `<= 1e-10` bound passes that value by inspection.

**Not covered here**

- Everything after the sampler call. The two runtime tests stop at `infer_mcmc` /
  `pixel.run_direct_mcmc` with a sentinel exception, so `run_fit`'s posterior marginalization,
  its ESS-floor guards and its coverage bookkeeping are still untested here -- they need a real
  `Model`.
- The `_weights` renormalization. `test_small_closure_reads_nested_vegas_signed_weights`'s
  fixture already sums to `1.0`, so `w / total` is a no-op there and a broken denominator would
  not be caught; only the *choice* of attribute is. (Pre-existing; recorded in that test's own
  docstring.)
- Real sampler behaviour. Every dispatch claim here is about which sampler and which options
  are selected, never about what that sampler then does; the VEGAS/NUTS/HMC numerics live in
  `src/pixel/infer` and are tested there.
