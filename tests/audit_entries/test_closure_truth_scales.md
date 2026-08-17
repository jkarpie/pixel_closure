### `tests/test_closure_truth_scales.py`

**Exercises** `jam_truth_curves`/`nnpdf_truth_curves` and `project_basis` in
`closure_{JAM,NNPDF}_truth{,_small}/pdf_guidance.py`, and `ensemble_curves`/
`_ensemble_cache_path` in the matching `generate.py` -- eight real source files (no
re-export shim). The file imports nothing from `pixel`; it drives only the four
`closure_*_truth*/` driver packages.

**Claim** The requested truth scale reaches the LHAPDF dumper and the returned
metadata unclamped, and a replica-ensemble cache file written under the old
clamped-scale regime is rejected rather than trusted.

**Oracles** As of 2026-08-13 one is independent physics:
`test_project_basis_builds_the_su4_flavor_combinations` pins all nine flavor
combinations against hand-computed integers (`A1`), so the step that turns LHAPDF
flavor columns into the injected truth is no longer checked by key set alone. The
scale/cache oracles remain internal to the suite: the expected `q` is read from the
same `cfg.TRUTH_Q_CHOICES` table the source itself reads, and the "stale" vs. "fresh"
cache values are per-field sentinels the tests construct themselves. What makes those
non-trivial is git history, not an external reference: commit `6c5857b` deleted a real
`clamp_scale` helper that silently substituted `Q -> QMin` below a set's advertised
minimum and mislabeled the result under a `clamped_to_qmin` key. The two `clamp`-named
tests are precise regression guards against that specific, previously-shipped bug --
one in the live evaluation path, one in a cache file written while the bug was live --
and each is now paired with the branch it could not reach (`below_set_qmin`'s `False`
side; the cache's *accept* half including its legacy-key migration).

**Cost** 5 test functions, 22 cases; measured 0.75 s for the file.
`dump_jam`/`dump_nnpdf` (the
only functions that shell out to LHAPDF or compile the C++ helper) are monkeypatched
in every case that needs them, so nothing here runs a subprocess or reads a physical
PDF value; the only I/O is a few-KB `.npz` round trip through `tmp_path` in the two
cache tests.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_truth_curves_use_requested_scale_without_qmin_clamp` | The `q` forwarded to the (mocked) LHAPDF dumper, and `meta["q_original"]`/`["q_effective"]`, equal `cfg.TRUTH_Q_CHOICES[q_key]` unclamped; `below_set_qmin` is `True`; `clamped_to_qmin` is absent; `clamp_scale` no longer exists on the module | data-flow identity vs. the same config table the source reads (`A3`) | exact `==` | `S3` |
| 2 | `test_ensemble_cache_rejects_old_clamped_scale` | An on-disk cache whose `q_effective` matches the historical clamped value (everything else matching) is not served: the mocked truth-curve function is called exactly once, the returned array is the fresh sentinel not the cached one, and `meta["q_effective"]` is corrected | real cache-validity gate in `generate.py`; two suite-built sentinel objects (`A3`) | exact `==` on call record and sentinel arrays | `S3` |
| 3 | `test_truth_curves_report_a_scale_above_the_set_minimum` | `below_set_qmin` is `False` for a `q` above the set's QMin (JAM `mc`=1.28 > 1.14; NNPDF `2`=2.0 > 1.65), and `False` again -- not `None`, no raise -- when the set advertises no QMin | comparison re-derived from `cfg.TRUTH_Q_CHOICES` and the vendored `.info` QMin (`A3`) | exact `is False` | `S3` |
| 4 | `test_project_basis_builds_the_su4_flavor_combinations` | All nine combinations equal hand-computed integers on a fixture whose outputs are pairwise distinct in magnitude; a second row scaled by `-2` catches a transposition | standard SU(4) definitions, evaluated by hand (`A1`) | `rtol=0, atol=0` exact | `S2` |
| 5 | `test_ensemble_cache_accepts_a_matching_scale_and_migrates_legacy_meta` | A cache whose `q_effective` matches is served without calling the recompute at all; per-field sentinels come back under their own keys; full suites drop a legacy `truth_kind` in favour of `uncertainty_kind`, `_small` suites do not | suite-built sentinels + a recompute stub that raises (`A3`) | exact `==`; `rel=1e-12` on arrays | `S2` |

**Weak spots**

- **RESOLVED 2026-08-13.** `test_ensemble_cache_rejects_old_clamped_scale` checked only
  `curves[cfg.ALL_FIELDS[0]]`, and its mock returned bit-identical content for all 9
  fields, so a bug that mixed up which stacked replica array lands under which field
  key would pass every assertion in it. The mock now gives each field its own value
  (`100 + index`) and the test asserts the full key set plus every field's values.
  Acceptance: rotating the field keys by one in `ensemble_curves` (both the fresh-compute
  and the cache-hit dict comprehensions) fails it; the pre-fix body, run verbatim under
  the same mutation, **passes** -- measured directly rather than argued.

**Not covered here**

- Physical PDF values. Every dumper is mocked, so no test here reads LHAPDF; what
  `project_basis` is checked against is its own arithmetic on a synthetic block, not
  a physics reference.
- The ensemble cache's *other* validity fields. `q_effective` is isolated by
  construction in both cache tests (node array and member count always match), so a
  gate that stopped comparing nodes or member count is not covered.
- The closure fit itself (`run_closure.py`/`fit.py` in the same directories) --
  out of scope for this file.
- Not a defect, but worth knowing when reading this file: the full and `_small`
  closure suites build the truth by genuinely different methods (full: one fixed
  LHAPDF member, ensemble only for covariance calibration; `_small`: the ensemble
  *mean* itself, documented at `closure_JAM_truth_small/README.md:37`) even though
  `ensemble_curves`'s cache-validity code is otherwise shared -- this file's cache
  test does not distinguish or assert which is in effect, which is correct for its
  stated scope but easy to misread as "the same truth construction, tested twice."
