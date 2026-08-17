### `tests/test_closure_plot_scaling.py`

**Exercises** `reproduction_ylim` in `closure_JAM_truth/run_closure.py:116-154`,
`closure_JAM_truth_small/run_closure.py:115-153`, `closure_NNPDF_truth/run_closure.py:116-154`
and `closure_NNPDF_truth_small/run_closure.py:115-153` -- a pure-numpy helper that picks the
`(ymin, ymax)` for a reproduction-panel y-axis from the truth and posterior one-sigma bands --
and, since 2026-08-13, `plot_reproduction` in the same four files (`:156-157`), its only
caller. The four bodies are byte-identical (`diff`, confirmed this audit; re-confirmed
2026-08-13 by `inspect.getsource` length, 1527 chars in all four). No `pixel` import: this
file drives only these four closure-driver copies, never `pixel.plotting` or any other
`src/pixel` module.

**Claim** The returned y-limits keep both the truth's and the posterior's one-sigma bands
visible with headroom for `x >= 0.2` without letting the much larger low-x excursions flatten
the useful region; zero stays on-panel; every guard (empty focus window, degenerate span,
non-finite entries) returns something sane; the two keyword-only parameters are read from the
arguments; and `plot_reproduction` puts the resulting limits on each of its nine Axes.

**Oracles** No independent physics oracle: the "truth"/"posterior" arrays are arbitrary
literals invented for this file, and the expected values are hand-derived from that literal
arithmetic (closed-form, elementary min/max/pad). Test 2's expected fallback tuple is
transcribed directly from the source's own hardcoded literal -- a regression pin, not an
independent derivation. Test 8 does build a Matplotlib figure and read the rendered
`ax.get_ylim()` back off all nine Axes; its expectation is a closed form derived from the
helper's degree-one homogeneity, *not* a second call of the helper, so it is not
self-referential.

**Cost** 10 test functions -- the 8 numeric ones parametrized over `RUNNERS`
(one also over the sign of the band), plus the two source-identity guards added 2026-08-14
(one parametrized over 9 duplicated `run_closure.py` callables) -- **28 pytest items,
measured `28 passed in 3.09 s`** (2026-08-14, serial). Essentially all of it is the Matplotlib
renders in test 8 (0.6 s each with a warm font cache; the very first call in a fresh
environment pays ~9 s for the font-cache build). The `reproduction_ylim` arithmetic itself is
trivial (numpy over arrays of length <= 5). No test here appears in
`tests/test_durations.json`, so all are unmarked by the `fast`/`slow` duration-based marker.

**Two runner tuples since 2026-08-14, and the split is load-bearing.** `RUNNERS` -- what
every behavioural `@parametrize("runner", ...)` uses -- is the NNPDF pair
(`nnpdf_full`, `nnpdf_small`), trimmed from all four on the owner's instruction:
`reproduction_ylim` has no notion of a truth PDF set, and test 9 proves all four copies of it
are one implementation, so the JAM legs pushed the same fixture through the same code.
`ALL_RUNNERS` is the four-entry tuple, kept for tests 9 and 10 only. **Test 10 asserts
`len(set(raw.values())) > 1`, so trimming it would turn that test RED rather than vacuous** --
measured directly 2026-08-14 with `_executable_body` outside pytest: 2 distinct raw
`plot_reproduction` bodies over `ALL_RUNNERS`, **1** over the NNPDF pair, because the raw
divergence *is* the JAM/NNPDF legend string. Before the trim: **46 pytest items in 5.72 s**.
The 18 removed items are worth ~2.6 s, most of it the two JAM Matplotlib renders in test 8.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_reproduction_ylim_contains_both_bands_above_point_two` | The exact return `(-0.192, 1.792)`, plus the four original one-sided bounds `ymin<0`, `ymax>1.6`, `ymin>-10`, `ymax<10`, run once per `RUNNERS` entry. | Hand-derived closed form on this fixture (`A1`); the exact-value assertion was added 2026-08-13 because none of the four bounds is tight against it. | `rtol=1e-12, atol=0.0` (measured `0.0` relative difference) plus the four bounds | `S2` |
| 2 | `test_reproduction_ylim_has_stable_fallback_for_missing_focus_data` | Exact fallback `(ymin, ymax) == (-0.08, 1.08)` when no point falls in the `x >= focus_x` window (both legs' masks all-False). Runs for both `RUNNERS` entries -- the full and `_small` layouts -- which is what closed audit item `-M01` (it originally ran `jam_small` alone). | Regression pin against the source's own hardcoded literal (`F2`); real, distinct early-return branch, not a weak check. | exact tuple equality | `S3` |
| 3 | `test_reproduction_ylim_floor_comes_from_the_lower_band` | The exact return `(-0.436, 1.486)` on a fixture with no exact zero in the focus region, where `min=-0.25` is the truth *lower* edge and `max=+1.30` the truth *upper* edge, both unique. | Hand-derived (`A1`): `span=1.55`, `pad=0.186`. The strictly-negative minimum stops the zero clamp from supplying the floor instead. | `rtol=1e-12, atol=0.0` | `S2` |
| 4 | `test_reproduction_ylim_clamps_a_one_signed_band_to_include_zero` | `(-0.132, 1.232)` for an all-positive focus region and `(-1.232, 0.132)` for the mirror, plus `ymin < 0 < ymax` and the data's own extreme strictly inside. | Hand-derived (`A1`) from edges spanning `[+0.4, +1.1]` / `[-1.1, -0.4]`, where the clamp moves an endpoint *and* the span. | `rtol=1e-12, atol=0.0` | `S2` |
| 5 | `test_reproduction_ylim_honours_explicit_focus_x_and_pad_fraction` | Four returns on one fixture: defaults `(-0.436, 1.486)`, `focus_x=0.6` -> `(-0.376, 0.926)`, `pad_fraction=0.0` -> `(-0.25, 1.3)`, `pad_fraction=0.25` -> `(-0.6375, 1.6875)`. | Hand-derived (`A1`); the `pad_fraction=0.0` case pins the un-padded clamped extremes with no pad arithmetic in the way. | `rtol=1e-12, atol=0.0` | `S2` |
| 6 | `test_reproduction_ylim_pads_a_degenerate_all_zero_band` | `(-1.2e-4, 1.2e-4)` at the default pad and `(-5e-4, 5e-4)` at `pad_fraction=0.5`, when every selected band edge is exactly zero. | Hand-derived (`A1`): after the clamp `ymin <= 0 <= ymax`, so `span == 0` forces `ymin == ymax == 0` and `pad = pad_fraction * 1e-3`. A different branch from test 2's early return. | `rtol=1e-12, atol=0.0` | `S2` |
| 7 | `test_reproduction_ylim_drops_nonfinite_points_instead_of_propagating_them` | `(-0.38, 1.48)` with a `NaN` truth centre and an `+inf` posterior spread inside the focus region; `(-0.072, 0.672)` when `center = spread = 1e308` so only the *sum* overflows. | Hand-derived (`A1`) from the surviving points. The second fixture is the only input class that reaches the concatenated `np.isfinite` filter. | `rtol=1e-12, atol=0.0` | `S2` |
| 8 | `test_plot_reproduction_sets_each_panel_ylim_from_its_own_bands` | All nine rendered `ax.get_ylim()` equal `(i+1)` times the first panel's limits, all nine differ, the PNG and PDF are written, and `len(cfg.ALL_FIELDS) == 9`. Run twice, with and without `curve_std`. | `A1`: field `i`'s curves are field 0's times `i+1` and `reproduction_ylim` is homogeneous of degree one, so the expectation is closed-form rather than a second call of the helper. `save_figure_both` is *wrapped*, not replaced, so the real save still runs. | `rtol=1e-12, atol=0.0` (measured worst case `2.22e-16` over all 18 limits) | `S2` |

**Weak spots**

- **RESOLVED 2026-08-14 — the module docstring's byte-identity claim is now
  executable, and the duplication turns out to be far wider than
  `reproduction_ylim`.** Every `RUNNERS` parametrization here is licensed by "the four
  `reproduction_ylim` bodies are byte-identical (`diff`, confirmed this audit)" — a
  hand-run diff with nothing to re-check it, in exactly the shape where a fix lands in
  one copy and stays broken in three.
  `test_the_duplicated_run_closure_helpers_are_one_implementation` asserts it on the
  parsed executable body (docstrings dropped), parametrized over **9** callables, and
  `test_the_four_plot_reproduction_copies_differ_only_by_their_suite_name` pins that
  `plot_reproduction` is one implementation plus a literal — in both directions, so the
  divergence is measured rather than assumed. Measured scope: of the 12 top-level
  callables all four `run_closure.py` copies share, 9 are one implementation copied
  four times, `plot_reproduction` is 2 raw bodies collapsing to 1, and only `run_one`
  (3 raw / 2 agnostic) and `main` (4 raw / 2 agnostic) genuinely differ — the full pair
  additionally saves posterior moments, draws a kinematic-coverage plot and prints a
  nuisance-coverage summary. Acceptance: a *behaviour-preserving* rewrite of one copy —
  self-checked to return identical numbers on a random fixture and on both fallback
  branches — fails `[reproduction_ylim]` **alone** (1 failed, 45 passed), so it sees
  drift no other test in the file can. De-duplicating the copies outright was
  considered and not done: the shared home would have to be `src/pixel` (off-limits) or
  a new top-level package, and the four suites are deliberately self-contained trees
  with no cross-imports.

- **W-LOOSE** (S1) **RESOLVED 2026-08-13** by `test_reproduction_ylim_floor_comes_from_the_lower_band`
  (added, all four runners as they then were; the NNPDF pair since 2026-08-14). The original fixture's minimum was an exact zero at `x=1.0`
  (`truth=posterior=0, std=0`), so a mutant that silently dropped the lower (`center - spread`)
  contribution returned the identical `(-0.192, 1.792)` and passed all four assertions --
  despite the test's own name promising "both bands". The new fixture has no exact zero in the
  focus region and both extremes are unique, `min=-0.25` being a strictly negative lower band
  edge so the zero clamp cannot supply the floor instead. *Acceptance:* an upper-only mutant
  returns `(-0.156, 1.456)` and a lower-only mutant `(-0.364, 0.814)`; both fail the new test,
  and the upper-only mutant -- the one the audit found nothing could see -- still **passes** the
  original test, which is the finding reproduced. (The lower-only mutant does fail the original
  test, because dropping the upper edge also pushes `ymax` under its `1.6` bound; only the lower
  edge was invisible.) The original test also gained its exact value `(-0.192, 1.792)` at
  `rtol=1e-12, atol=0.0`.
  See `plans/test_suite_hardening.md#test_closure_plot_scaling-01`.
- **W-NEVER** (S1) **RESOLVED 2026-08-13** by
  `test_reproduction_ylim_clamps_a_one_signed_band_to_include_zero` (4 runners x 2 signs when added; 2 x 2 since the 2026-08-14 `RUNNERS` trim).
  Removing the `min(0.0, ...)`/`max(0.0, ...)` zero-inclusion clamp (source docstring: "plus
  zero and modest headroom") used to change nothing on either fixture -- test 1's finite array
  already contained an exact `0.0`, test 2 returned from the `finite.size==0` branch first.
  Both new fixtures are strictly one-signed in the focus region, so the clamp is the only thing
  that can put zero inside the interval. *Acceptance:* a bare
  `float(np.min(finite))`/`float(np.max(finite))` mutant returns `(0.316, 1.184)` and
  `(-1.184, -0.316)`, failing all 8 of that test's items -- and also the 4
  `test_plot_reproduction_...` items, whose expectation is a closed form rather than a re-call
  of the helper (12 failed / 24 passed in total). Every other test in the file passes, and
  before this pass the clamp mutation was invisible to the entire suite. A neighbouring
  never-reached branch found while doing this -- the
  `span > 0.0` fallback in `pad = pad_fraction * (span if span > 0.0 else max(abs(ymin), 1e-3))`
  -- is now covered by `test_reproduction_ylim_pads_a_degenerate_all_zero_band`, which fails on
  the simplification `pad = pad_fraction * span` (it returns a degenerate `(0.0, 0.0)`).
  See `plans/test_suite_hardening.md#test_closure_plot_scaling-02`.
- **W-DEAD** (S4) tests/test_closure_plot_scaling.py `RUNNERS`: the 4-way `runner`
  parametrization ran one identical `reproduction_ylim` body four times (all four closure
  packages agree byte-for-byte; re-confirmed 2026-08-13 via `inspect.getsource` length, 1527
  chars each). It was kept as insurance against the copies drifting apart, same pattern
  as `test_closure_dy_plotting.py`'s `RUNNERS` tuple, and deliberately extended 2026-08-13 so
  every branch of the helper got the drift guard rather than only the main one.
  **CLOSED 2026-08-14 by the owner, the other way**: `RUNNERS` is now the NNPDF pair, and the
  drift guard moved to where it can be asserted instead of implied -- test 9
  (`test_the_duplicated_run_closure_helpers_are_one_implementation`) compares the parsed
  bodies of 9 shared callables across `ALL_RUNNERS`, which is strictly stronger than running
  the same fixture through four copies and getting the same number.
  See `plans/test_suite_hardening.md#test_closure_plot_scaling-03`.
- **W-NEVER, new and irreducible** (S4, recorded 2026-08-13): the per-point
  `np.isfinite(center) & np.isfinite(spread)` terms in `mask` are **provably redundant** with
  the later `finite = finite[np.isfinite(finite)]`. Any non-finite `center` or `spread` makes
  both `center-spread` and `center+spread` non-finite, so the concatenated filter removes it
  one step later; measured, a mutant that deletes the per-point terms is byte-identical on
  every fixture in this file. No test can catch their removal, and none should be written
  claiming to. The converse does *not* hold -- finite inputs whose sum overflows reach the
  concatenated filter and nothing else -- which is why test 7 carries a `1e308` fixture.

- **RESOLVED 2026-08-13, documentation-only** (`test_closure_plot_scaling-03`) — the four-runner parametrization ran one byte-identical `reproduction_ylim` body under four names, and the module docstring recorded that it was kept deliberately as a drift guard against the failure mode `S0-05` went undetected under. **Superseded 2026-08-14**: the behavioural parametrization is the NNPDF pair, and the drift guard is now the executable identity assertion in tests 9 and 10 over `ALL_RUNNERS`.

**Not covered here**

- [closure-suite] The `reproduction_ylim` branch coverage is now complete (empty focus window,
  degenerate span, zero clamp, both band edges, the `x >= focus_x` cutoff, both keyword-only
  parameters, and the concatenated `np.isfinite` filter), with the single measured exception
  recorded as the last weak spot above. What remains untested is everything *else*
  `plot_reproduction` does: the hatched posterior band, the hybrid x-scale, the titles, the
  legend and the suptitle are exercised (the figure renders and saves) but nothing is asserted
  about them. `tests/test_plotting.py:381` has the pattern if that is ever wanted.
- [closure-suite] Nothing here drives the closure runners' *other* figure producers
  (`plot_comparison` and the rest of `run_closure.py`), or the `run_closure` driver itself.
