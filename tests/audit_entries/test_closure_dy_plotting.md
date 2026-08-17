### `tests/test_closure_dy_plotting.py`

**Exercises** `closure_JAM_truth/plot_datasets.py` and `closure_JAM_truth_small/plot_datasets.py`
(`_dataset_x`, `_is_constraint_dataset`, `_sample_vectors_and_weights`,
`_sampled_linear_predictive_band`, `_bilinear_symmetric_action`, `_bilinear_predictive_band`,
`_physical_gp_moments`, `_representative_vector`, `_posterior_predictive_bands`,
`_record_chi2`, `build_records_from_result`, `plot_records`) plus their
`closure_NNPDF_truth(_small)` aliases, which are byte-identical to the JAM copies -- now
asserted by `test_jam_and_nnpdf_plot_datasets_stay_byte_identical` rather than only measured
once with `diff`; `closure_JAM_truth_small/run_closure.py` (`_prepare_matplotlib`,
`save_figure_both`), reached through `plot_records`; and `src/pixel/core/evidence.py`
(`BilinearFactorBlock`, `EvidenceBlocks`, `_bilinear_prediction_from_second`,
`bilinear_symmetric_action`, `posterior`), called for real, not mocked.

**Claim** The dense and factorized DY-band code paths compute the documented quadratic-form
mean and delta-method variance, and the closure copy of the half-gradient agrees with
`pixel.core.evidence`'s; the sampled (non-bilinear) band is the weighted mixture of its
per-sample Gaussians; `build_records_from_result` picks the right abscissa, drops constraint
pseudo-data, sorts by descending chi2, and assembles linear and DY records using the correct,
fully-merged representative vector even for short (nested-VEGAS-shaped) samples; and
`plot_records` writes both a PNG and a PDF and renders the band, the log scale and the DY
footnote it claims to.

**Oracles** `test_dense_dy_band_uses_quadratic_mean_and_delta_variance`'s expected
mean/variance are closed-form, rederived independently in the test rather than called from
source -- genuinely independent of the wrapper arithmetic, except that the "variance" side
transcribes a *documented approximation* (the source's own comment notes the post-DY posterior
is not exactly Gaussian), so agreement confirms the formula matches spec, not that the
approximation is itself accurate.
`test_factorized_dy_band_matches_equivalent_dense_operator`'s oracle is an algebraically
equivalent dense operator built by the test, but both results are produced by the *same*
`_bilinear_predictive_band` call: independent for the per-branch tensor contraction,
common-mode for everything the branches share (second-moment construction, clip, sqrt) --
which is exactly what the dense-band test covers, so the two are complementary rather than
either being weak alone. `closure_*/plot_datasets.py`'s `_bilinear_symmetric_action` is a
private, parallel reimplementation of `pixel.core.evidence.bilinear_symmetric_action` (compare
`closure_JAM_truth/plot_datasets.py:132-170` to `src/pixel/core/evidence.py:1199-1259`);
`test_closure_symmetric_action_matches_the_pixel_core_evidence_copy` checks the two copies
against each other (`A3`) on a two-block factorized operator with overlapping row ranges and
on a non-symmetric dense one -- an implementation-vs-implementation identity, so it measures
transcription rather than the half-gradient convention, which `tests/test_joint_action.py`'s
`jax.jacfwd` check pins instead. `test_sampled_linear_band_mixes_per_sample_posterior_moments`
uses the law of total variance written out in the test (`A1`) over per-sample moments from the
shared `pixel.core.evidence.posterior` (`A3`, deliberately common-mode: the accumulation
across samples is what is under test). The record-level and rendering tests use hand-built
`SimpleNamespace`/dict stand-ins for the real `Model`/`Result`/record objects; no accuracy bar
applies, and each establishes a specific structural or behavioural contract instead.

**Cost** 16 test functions, most parametrized over the same **2-entry** `RUNNERS` tuple --
one entry per distinct `plot_datasets.py` body, since JAM and NNPDF share source text in both
the "full" and the "small" size (asserted by test 1, not assumed). **47 pytest items,
measured `47 passed in 2.06 s`** (2026-08-14, serial). No
`slow`-marked tests; three tests do real matplotlib work (one writes to `tmp_path`, two
intercept `plt.close` to read the `Figure` back).

`RUNNERS` was `(jam_full, jam_small, nnpdf_full, nnpdf_small)` and collected **89 items in
2.11 s**; the two JAM entries came out on the owner's instruction 2026-08-14, removing **42**
items. (The change inventory predicted 43 and 0.57 s; measured, it is 42 items and ~0.05 s,
which is inside run-to-run noise -- these are cheap tests and the saving here is duplication,
not time.) Test 1 is untouched and still names all four packages: it reads them as literal
path components off disk, never through `RUNNERS`. `closure_JAM_truth_small.plot_datasets` is
also still imported, and driven directly by the three `plot_records` rendering tests.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_jam_and_nnpdf_plot_datasets_stay_byte_identical` | the JAM and NNPDF copies of `plot_datasets.py` are the same file, in both sizes | `read_bytes()` on the two pairs (`F1`) | exact bytes | `S3` |
| 2 | `test_dense_dy_band_uses_quadratic_mean_and_delta_variance` | Dense-operator DY band mean equals `E[q^T Y q]`; std equals the delta-method propagation `4(sym(Y)mu)^T Sigma (sym(Y)mu)`, both rederived by hand in the test. | Independent closed-form recomputation vs. the real, unmocked `runner._bilinear_predictive_band` (`A1`) | bare `assert_allclose` default (rtol=1e-7); measured 0.0 relative diff in a standalone reproduction | `S1` |
| 3 | `test_factorized_dy_band_matches_equivalent_dense_operator` | The factorized `BilinearFactorBlock` path agrees with an algebraically equivalent dense tensor built by the test. | Both sides call the *same* `_bilinear_predictive_band` wrapper, differing only in operator type (`A3`) | rtol=1e-12; demonstrated to catch a transposed mixing index at 3.4e-2 | `S2` |
| 4 | `test_closure_symmetric_action_matches_the_pixel_core_evidence_copy` | the closure `_bilinear_symmetric_action` equals `pixel.core.evidence.bilinear_symmetric_action` on one real `EvidenceBlocks`, for a **two-block** factorized operator with overlapping rows and for a non-symmetric dense one | identity between two independently typed implementations (`A3`) | rtol=1e-12, atol=0; measured `max|a/b-1| = 6.7e-16` both layouts | `S2` |
| 5 | `test_dataset_x_selects_each_independent_variable_in_priority_order` | all six `_dataset_x` return paths and both guards on the xF branch, with the right label and `use_log` | closed form for xF (`2 sqrt(25/100) = 1.0` exactly, so expected = `sinh(Y)`), structural for the rest (`A1`/`F1`) | bare default; xF case exact | `S3` |
| 6 | `test_mixed_records_include_linear_and_dy_without_rebuilding_short_samples` | `model._rebuild` receives the full `n_free`-sized representative vector, never a short per-sample row; record **count**; and the linear record's `x`/`data`/`sigma`/`reproduction`/`fit_std`/`truth`/`chi2` by value | embedded `assert vec.size == 3` in a `rebuild` stub mirroring `fit._weighted_mean_vec`'s real `NestedVegasSamples` branch, plus hand-derived values (`F1` + `A1`) | exact size check; `rel=1e-12` on chi2; bare default elsewhere | `S2` |
| 7 | `test_constraint_pseudo_data_is_excluded_from_records` | `cons_*`-named and `source=="constraint"` datasets are dropped, a physical one is kept | returned record names, both `or` legs fed independently (`F1`) | none (structural) | `S3` |
| 8 | `test_records_are_sorted_by_descending_chi2_with_nan_last` | records come back 9, 4, 1, NaN | unit-variance one-row datasets give chi2 = residual^2 exactly (`A1`) | `pytest.approx` on the three finite chi2 | `S3` |
| 9 | `test_sample_vectors_and_weights_handles_map_and_weighted_samples` | MAP-like `.x` becomes one row at unit weight; a 2-D `.samples` keeps its rows and normalizes `[1,2,1] -> [0.25,0.5,0.25]` | closed-form normalization (`A1`/`F1`) | bare default | `S3` |
| 10 | `test_sampled_linear_band_mixes_per_sample_posterior_moments` | the non-bilinear band is the weighted mixture of the per-sample Gaussians, not an average of means | law of total variance written out in the test over `pixel.core.evidence.posterior` moments (`A1` + `A3`) | rtol=1e-12 (measured 0.0); non-degeneracy asserted at `> 0.1` with 0.631 achieved | `S2` |
| 11 | `test_sampled_linear_band_reuses_stored_moments_without_rebuilding` | a stored moment cache short-circuits the per-sample loop, and a wrong-size cache does not | a `_blocks` that raises, so reachability is observable (`F1`) | none (structural) | `S3` |
| 12 | `test_plot_records_renders_the_band_the_log_scale_and_the_dy_footnote` | panel count, `fill_between` band as a `PolyCollection` only where `x.size >= 2`, `set_xscale("log")` only where asked, `[DY]` in the right title, exactly one footnote | `plt.close` intercepted to keep the `Figure`; artist type and axis state (`F1`) | none (structural) | `S3` |
| 13 | `test_linear_only_records_carry_no_dy_footnote` | the footnote guard is a guard: no DY record, no footnote | same interception, negative control (`F1`) | none (structural) | `S3` |
| 14 | `test_dy_records_render_png_and_pdf` | `plot_records` writes both a `.png` and a sibling `.pdf`, unmocked | real call; `Path.is_file()` on both outputs (`F1`) | none numeric -- file existence only | `S3` |

**Weak spots**

- **RESOLVED 2026-08-14 — the 1-D `.samples` leg is now a raise, not an unpinned
  ambiguity**, pinned by
  `test_flat_samples_are_rejected_instead_of_silently_scaling_the_band` and
  `test_predictive_band_refuses_a_vector_weight_count_mismatch`. All four `plot_datasets.py` copies changed identically (the JAM/NNPDF
  and `_small` byte-identity pairs are preserved):
  `_sample_vectors_and_weights` refuses `vectors.ndim != 2` — written `!= 2`, not
  `== 1`, because the old reshape let a 3-D `.samples` through untouched — and
  `_sampled_linear_predictive_band` names a vector/weight count mismatch instead of
  letting `zip()` truncate. The justification is the sampler contract, not a
  preference: `infer/hmc.py:1101` and `infer/nuts.py:1100` allocate
  `np.empty((n_samples, n_params))`, `infer/vegas.py:1036` and `infer/mcmc.py:383`
  stack per-draw rows, and all four spell the zero-parameter case
  `np.zeros((n_samples, 0))` — still 2-D, and now pinned by a control. Measured
  pre-fix damage on `_sampled_band_model`: accumulated weight `0.333` instead of
  `1.0`, mean scaled by exactly `1/n`, band wrong by `1.09`-`1.36x` in the *other*
  direction (`var = w*var + (w - w**2)*mean**2` is not a clean rescaling).
  Acceptance: a plugin restoring the old reshape and removing the `zip` guard makes
  exactly the 12 new parametrized cases fail while the file's other 77 pass. (Counts are
  against the pre-2026-08-14 4-entry `RUNNERS`; the mutation patches the packages, not the
  parametrization, so it still bites -- re-measure the failed/passed pair before quoting it.)

- RESOLVED 2026-08-13 (was: **W-DEAD** S4, `RUNNERS` parametrizing 4 aliases over 2 distinct
  code bodies, with the byte-identity claimed only in prose). Test 1 now *asserts* the
  byte-identity of both pairs, so a hand-edit to one copy fails loudly instead of quietly
  voiding the parametrization's purpose. The same comparison applied to a genuinely different
  pair (full vs `_small`) returns False, so the assertion is not vacuous.
  **Closed the other way 2026-08-14** (owner's instruction): with the identity asserted, the
  4-entry tuple was the dead half of the arrangement, so `RUNNERS` is now one entry per
  distinct body. Test 1 carries the whole JAM-side guard, and does it better -- it sees a
  divergence anywhere in the file, where the parametrization saw one only if it happened to
  land in the handful of functions these tests call.
- **W-COMMON** (S3) tests 3's factorized and dense results are both produced by the same
  `_bilinear_predictive_band` call, so a bug in the shared wrapper (second-moment construction,
  clip, sqrt) affects both sides identically. Inherent to what that test targets, not a defect;
  test 2 covers the shared wrapper with an independent oracle, and test 4 (added 2026-08-13)
  now covers the half-gradient against a genuinely separate implementation.
  See `plans/test_suite_hardening.md#test_closure_dy_plotting-02`.
- RESOLVED 2026-08-13 (was: **W-LOOSE** S3, a kind-*set* blind to a duplicate record, with the
  "linear" record never inspected). Test 6 now asserts `len(records) == 2` and seven values on
  the linear record, all hand-derivable from the fixture (`reproduction == [0.7, 0.9]`,
  `fit_std == sqrt([0.04, 0.03])`, `truth == [0.6, 0.8]`, `chi2 == 0.25 + 4/9`).
- RESOLVED 2026-08-13 (was: **W-NAME** S3, `test_dy_records_render_png_and_pdf` checking only
  file existence while its name implied rendering). That test is unchanged and still covers the
  unmocked write path; tests 12-13 added the content assertions by intercepting `plt.close`.
  Note the band artist must be identified by *type*: `errorbar` also populates
  `ax.collections`, and measured, both panels carry exactly 2 collections.
- Tests 6-8 and 10-11 drive `build_records_from_result`/`_sampled_linear_predictive_band`
  through hand-built `SimpleNamespace` stand-ins for `Model`/`Result`. A change to the real
  `Model`'s attribute names or layout structure would not be caught here -- only a change in
  the plotting module's use of them.

**Not covered here**

- **FINDING, source not test** (2026-08-13): for a 1-D `samples.samples` array,
  `_sample_vectors_and_weights` returns **one** vector of length `n` while `fit._weights`
  returns **n** weights (it reads `samples.samples.shape[0]` independently), and
  `_sampled_linear_predictive_band`'s `zip(vectors, weights)` then iterates once with weight
  `1/n`, scaling the whole reported band. Which reading is intended -- one sample of `n`
  parameters or `n` samples of one -- is a source decision, so test 9 exercises that leg but
  deliberately does not pin either answer.
- `_truth_predictions`'s `moment_systematic_values` branch, the only place the full and
  `_small` copies of `plot_datasets.py` differ, is still unreached by any test here.
- `build_records` (which runs a real fit through `fitmod.run_fit`) and `fitmod.load_truth` are
  still never called; every test supplies its own `truth` dict and result.
- `write_summary` and `main` have no coverage in this file.
- The `bilinear_ratio`/nonlinear dataset kinds never appear in these layouts.
