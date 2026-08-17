### `tests/test_small_synthetic_z.py`

**Exercises** the `synthetic_z_plumbing` branch of
`closure_JAM_truth_small.datasets.build_datasets`/`build_drell_yan` and the
byte-identical NNPDF copy -- past that, the real
`pixel.data.drell_yan.DrellYan`/`DrellYanData` factory (`.name`, `.source`,
`.component`, `.n_data`, `.bilinear_contributions`, `.mean`, `.cov`), with
`source`/`component` read back off `DatasetMetadata` (`pixel.core.model`). Does
**not** exercise `generate.py` in either package -- both tests write
`manifest.json` by hand and monkeypatch `config.truth_dir`.

**Claim** One hand-written manifest record, run through `build_datasets` with
only `include_synthetic_z=True`, produces exactly one correctly-labeled
`DrellYanData`; malformed synthetic_z manifests raise the guard clauses'
documented messages.

**Oracles** Every assertion compares against a literal the test itself wrote a
few lines earlier (`D1`), or matches a `pytest.raises` message substring against
`closure_JAM_truth_small/datasets.py`'s literal guard text (`F1`). There is no
numeric tolerance and no independent computation anywhere in this file -- it is
plumbing/contract coverage (a manifest field reaches the right attribute; a
malformed manifest raises), not DY kernel or evolution physics. See
`tests/test_closure_dy_evolution.py` and `tests/test_dy_dataset.py` for kernel-
and evolution-level DY coverage, neither of which touches the `closure_*_small`
packages or this file's synthetic_z path.

**S0-05 (open, owner decision):** `closure_*_small/generate.py`'s `dy_central`
computes the DY truth central value as a replica-ensemble second moment
(`generate.py:265`), which `closure_JAM_truth/generate.py:494` documents as
"deliberately forbidden" for a bilinear observable -- measured 7-8% off in
`tests/test_closure_truth_representable.py`. Confirmed by import inspection that
this file cannot touch it: `closure_JAM_truth_small/datasets.py` imports only
`json`, `pathlib`, `numpy`, `pixel.data` and its own `config` -- never
`generate` -- so neither test in this file calls `dy_central` transitively.
This file's manifest `mean`/`cov` are literals the test wrote itself, not
`generate.py` output; it is silent on the defect, neither masking nor exposing
it, not evidence either way. (`generate.py`'s own synthetic_z path *does* call
the same `dy_central` in production -- see **Not covered here**.)

**"z" is the Z boson**, not this repo's other use of the letter: lattice
matrix elements spell the Wilson-line separation `z` in
`src/pixel/kernels/lattice.py` (`_separation_fm`, `finite_volume_z`) and
`src/pixel/data/lattice/`. The two senses never collide in this file -- it
builds only a Drell-Yan dataset -- but the bare filename is genuinely ambiguous
out of context, now stated explicitly in the module docstring.

**Cost** 2 `def test_*`, **2 collected cases** (test 1 NNPDF-only, test 2 JAM-only).
All exact-equality/`pytest.raises` checks, no float
comparisons. Measured: **`2 passed in 0.68 s`** (2026-08-14, serial), against
**`3 passed in 0.61 s`** immediately before. Not slow -- neither test reaches the closure
fit or generation pipeline.

Test 1's JAM case was dropped 2026-08-14 on the owner's instruction. It ran
byte-identical code (see **Weak spots**), and the test's own docstring already said to
"read this as one case, not two"; it is now literally one. Test 2 is untouched and still
drives the JAM `_small` copy, so both packages are still imported and reached.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_small_synthetic_z_mode_builds_only_one_labeled_proxy` | `built[0].name`/`.source`/`.component`/`.n_data` equal the manifest literals; `len(built)==1`; `TEST_MODE_SYNTHETIC_Z in TEST_MODES`; `bilinear_contributions` nonempty | field read-back against literals the test wrote itself (`D1`) | exact `==`/`in`/truthiness, no tolerance | `S3` |
| 2 | `test_small_synthetic_z_rejects_missing_or_unclassified_manifest` | 0-record and misclassified-record manifests each raise `ValueError` matching the guard's literal text | `pytest.raises` message-substring match (`F1`) | exception type + substring | `S3` |

**Weak spots**

- **CLOSED 2026-08-14 by removal.** The JAM/NNPDF parametrization on test 1 ran
  byte-identical code both times:
  `closure_JAM_truth_small/datasets.py` and the NNPDF copy differ only in a
  docstring cross-reference and an unrelated kwarg-line reorder inside
  `build_pseudoitd` (diffed directly), and every DY-relevant `config.py`
  constant matches too. It guarded the two copies against drifting apart from
  each other, not against two different implementations -- disclosed in
  the docstring, and then dropped on the owner's instruction as duplication.
  Note what is *not* left behind in its place: this file has no executable
  assertion that the two `_small` `datasets.py` copies agree, the way
  `test_closure_truth_representable.py` and `test_closure_plot_scaling.py` do for
  theirs. The diff above is prose. See `plans/test_suite_hardening.md#test_small_synthetic_z-01`.
- The manifest's `"name"` and `"label"` are both the literal
  `"synthetic_z_plumbing"`, so `built[0].name == "synthetic_z_plumbing"` cannot
  tell `build_drell_yan`'s `rec.get("name", rec["label"])` apart from a bug
  that read the wrong one of the two -- the fixture is degenerate in exactly
  the dimension this assertion probes (tests/README.md rule 9's second failure
  class). See `plans/test_suite_hardening.md#test_small_synthetic_z-02`.
- `built[0].mean`/`.cov` are never asserted. Measured directly: rebuilding the
  identical dataset with the manifest's `mean`/`cov` keys removed entirely
  (`DrellYan` then defaults both to `None`) leaves every assertion in the test
  passing unchanged -- `name`, `source`, `component`, `n_data` and
  `bilinear_contributions`'s truthiness are bit-for-bit identical with or
  without them (0.84s, see the audit report's `measurements`). See
  `plans/test_suite_hardening.md#test_small_synthetic_z-03`.
- Both tests bundle independent checks with no way for one to fail without
  hiding the rest. Test 1 runs seven assertions in sequence; the first,
  `TEST_MODE_SYNTHETIC_Z in TEST_MODES`, is a static `config.py` fact
  unrelated to `built` and would fail identically whether the manifest
  plumbing below it works or is completely broken, burying the five field
  checks and the `bilinear_contributions` check that never ran. Test 2's two
  `pytest.raises` blocks are independent guards run back to back; if the first
  regressed, the second (misclassified-record) guard would never execute in
  that session. See `plans/test_suite_hardening.md#test_small_synthetic_z-04`.

- **RESOLVED 2026-08-13, documentation-only** (`test_small_synthetic_z-01`) — the JAM/NNPDF parametrization runs byte-identical code with identical DY constants, so the second case cannot catch anything the first misses. Verified this pass that `:29-38` and `:49` record exactly that, leaving it an explicit drift guard rather than two independent cases.

**Not covered here**

- `generate.py`'s own `synthetic_z_layout()` (line 217, JAM/NNPDF-identical) is
  the production source of the record this file hand-writes, and
  `generate_member`'s synthetic_z block (`generate.py:469-493`) folds it
  through this same `build_drell_yan` and through `dy_central`, then writes it
  into `manifest.json` under the identical `"synthetic_z"` key. No test calls
  `synthetic_z_layout` or that block (confirmed repo-wide: only the definition
  and one call site per package). This file's hand-written record matches that
  function's current field set by inspection only, not by a shared check, so a
  schema change on either side would drift silently.
- `closure_NNPDF_truth_small.datasets`'s two synthetic_z guard clauses (byte-
  identical code to JAM) have zero test coverage -- the guard-clause test is
  JAM-only. Since 2026-08-14 the sibling test is NNPDF-only, so the two tests
  now cover one package each rather than overlapping; the gap named here is
  unchanged in substance.
- The "exactly one" guard's 2+-records arm is never exercised (only 0 and 1).
- `build_drell_yan`'s `component=rec.get("observable_contract", rec["reaction"])`
  fallback default is never exercised here -- this file's record always
  supplies `observable_contract`. Distinct from `tests/test_observable_contracts.py`,
  which drives an unrelated YAML contract registry in `pixel.data.contracts`.
- DY kernel and evolution physics for Drell-Yan datasets: see
  `tests/test_closure_dy_evolution.py` and `tests/test_dy_dataset.py`.

**Added by the Phase-2 missing-test pass, 2026-08-14** — closes
`test_small_synthetic_z-M01` through `-M06`. Plugins in `scratchpad/mut_n1/`; each
mutation below leaves the file at 1 failed / 7 passed.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| + | `test_synthetic_z_rejects_two_records` | 2 records hit the same "exactly one" guard as 0 records, and 1 record still builds | guard fed, matched on the **message** (a second `ValueError` guard sits below it), plus a 1-record control (`F1`) | `pytest.raises`; control `len(built) == 1` | `S3` |
| + | `test_synthetic_z_mean_and_cov_reach_the_built_dataset` | the manifest's `mean`/`cov` survive into the `Dataset`, and both-or-neither holds | fixture echo at `[2.5]`/`[[0.16]]`, values that are neither `DrellYan`'s defaults nor the sibling test's (`D1`) | `assert_array_equal` (JSON round-tripped literals, not computed) | `S2` |
| + | `test_synthetic_z_manifest_keys_map_to_distinct_dataset_attributes` | `name` vs `label` and `observable_contract` vs `reaction` are read apart | non-degenerate fixture: all four values differ, and that is asserted before the reads (`D1`) | exact string equality | `S1` |
| + | `test_synthetic_z_optional_keys_fall_back_to_their_partners` | with `name`/`observable_contract` absent, the two `rec.get(a, rec[b])` fallbacks take the right partner | the two partners are asserted to be different strings first (`F1`/`D1`) | exact string equality | `S1` |
| + | `test_generation_time_synthetic_z_layout_is_what_the_manifest_reader_consumes` | `generate.synthetic_z_layout()`'s record is exactly what `build_datasets` consumes, and both build routes give one forward operator | `A3` between the generation-time `build_drell_yan(rec, ...)` call at `generate.py:566` and the fitting-time manifest route (360 bilinear terms, ordered); plus `C1` on the PDG Z mass and the 7 TeV LHC energy carried by the layout | `assert_array_equal` on the 360 weights; `< 1e-12` relative on `sqrt(Q2)/91.1876` and `sqrt(S)/7000` | `S1` |
| + | `test_both_small_suites_share_one_synthetic_z_code_path` | the two `_small` copies of `build_datasets`, `build_drell_yan`, `synthetic_z_layout`, the 8 DY-relevant config constants, the layout record and `dy_field_maps("pp")` are identical | `A3`, source text plus data (`inspect.getsource`) | exact equality | `S3` |

**Weak spots** (of the six above)

- **`-M01` was not written the way it was proposed, and the proposal is superseded.** It
  asked to parametrize the guard test over `closure_NNPDF_truth_small` as well — which is
  precisely the byte-identical double run the owner stripped from this file's *other* test
  on 2026-08-14. `test_both_small_suites_share_one_synthetic_z_code_path` asserts the
  identity instead, which is strictly stronger: running the same code twice for the same
  answer cannot detect a divergence, only comparing the copies can. This follows the
  `ALL_SUITES` / `ALL_RUNNERS` convention already used in
  `tests/test_closure_constraints.py` and `tests/test_closure_plot_scaling.py`, and
  `ALL_SMALL_SUITES` must keep both entries for the same reason those must keep four.
- The source-identity test is textual: a semantically identical reformatting of either copy
  fails it. That is the intended trade (a false alarm is cheap, a silent divergence is not),
  but it is not a behavioural equivalence proof.
- `test_generation_time_...`'s route-identity half cannot see a **generator-side** rename,
  because both routes read the same record and fall back together. The layout's own
  contract values are asserted separately for exactly that reason, and that is where the
  detection against the generator sits.
- Still silent on `dy_central` (the `S0-05` ensemble-second-moment question): the `mean`
  this file supplies is always a literal, never `generate.py` output. Left with the owner.

**Acceptance mutations** (each: 1 failed, 7 passed)

- `mut_sz_guard_accepts_many` — the "exactly one" guard weakens to `if not records`
  (simulated at `load_manifest`, so `build_datasets`'s source text is untouched and the
  identity test stays out of it).
- `mut_sz_drops_mean_cov` — `build_drell_yan` stops forwarding `mean`/`cov`.
- `mut_sz_reads_label_for_name` — `name=rec.get("name", rec["label"])` reads `label`.
- `mut_sz_wrong_fallback_partner` — the two fallbacks point at the wrong partner key
  (`name <- reaction`, `component <- label`); fires only when the optional key is absent.
- `mut_sz_layout_renames_a_key` — the generator renames `observable_contract`, applied to
  **both** copies so the identity test is not dragged in.
- `mut_sz_suites_diverge` — one copy's `DY_NF` drifts from 4 to 5.

The three `build_drell_yan` mutations are applied to both `_small` copies deliberately: on a
one-sided patch `test_both_small_suites_share_one_synthetic_z_code_path` also fails, which
is a true positive rather than noise, but it muddies the per-item evidence.
