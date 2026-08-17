### `tests/test_closure_charged_current_wiring.py`

**Exercises** the closure drivers directly, not `src/pixel`: `{suite}.generate.exp_layout`
(`closure_NNPDF_truth/generate.py:528-566`, byte-identical RS-branch in the other three),
`{suite}.config` (`EVEN_MAP`, `ODD_MAP`, `ORDER`, `TEST_MODES`, `CC_SIGMA_R_IDX`, `EXP_SPECS`),
and `{suite}.datasets` (`_EXP_BUILDERS`, `_experimental_builder`, `build_sigma_r_cc`,
`_even_map`, `_odd_map`), for every `suite` in `SUITES` (`closure_NNPDF_truth`,
`closure_NNPDF_truth_small` -- see **Cost**). The one `pixel`
symbol referenced, `pixel.data.ChargedCurrentSigmaR.from_file`
(`src/pixel/data/experimental/charged_current.py`, real implementation, not a shim), is
monkeypatched out in the one test that touches it, so no `src/pixel` code actually executes
here.

**Claim** `exp_layout()` keeps a manifest record's `kind`/`beam`, drops non-`used` tables,
expands the target label in the full suites, and computes `y` per row -- from an explicit `Y`
column when present, otherwise from that row's own `RS`; `sigma_r_cc` is wired through the
ordinary experimental-dataset path (and is bound to the *right* builder there), not a synthetic
special case; a manifest still tagged the retired `"f3_proxy"` kind fails loudly; and
`build_sigma_r_cc` forwards beam charge, `y` or `sqrt_s`, the basis maps *per key*, the
normalization nuisance when enabled, and order to `from_file`.

**Oracles** Test 1's expected `y` is the source's own `q2/(x*rs*rs)` formula regrouped as
`(q2/x)/rs**2` on the same numbers -- a float-reassociation floor (measured bit-identical,
0.0, between the two groupings), not an independent computation. Tests 2-4 are purely
structural (`F1`): dict/tuple membership, exception type + message substring, and
captured-kwarg equality against a monkeypatched fake -- none carry a numeric bar or an
independent oracle in the physics sense. For a given suite size, `datasets.py`'s
charged-current logic is byte-identical between the JAM and NNPDF variant (diffed), so the
JAM/NNPDF axis of the parametrization is drift protection, not independent coverage; the
full/small axis does exercise genuinely different code (`with_systematics` only exists on the
full builders; CC registration is `CC_SIGMA_R_IDX` in the full config vs per-entry
`EXP_SPECS.kind` in the small one).

**Cost** 7 tests over the same **2-entry** `SUITES` tuple, one of them also parametrized over
`with_exp_nuisances` (**16 cases total**, measured `16 passed in 0.70 s`, 2026-08-14 serial).

`SUITES` was `(closure_NNPDF_truth, closure_JAM_truth, closure_NNPDF_truth_small,
closure_JAM_truth_small)` and collected **32 cases in 0.75 s**; the JAM entries came out on
the owner's instruction 2026-08-14. The two axes in that tuple were never equivalent: for a
given size the charged-current logic in `datasets.py` is byte-identical between the JAM and
NNPDF variant, so the JAM/NNPDF axis ran the same code twice, while the full/small axis --
which stays -- exercises genuinely different code (`with_systematics` kwarg, `CC_SIGMA_R_IDX`
vs per-entry `EXP_SPECS.kind`). Every behavioural asymmetry this file pins, including the
`TARGET_MAP` one, is on the surviving axis. All fast: JSON-fixture parsing and
dict/attribute lookups on live-imported closure config modules, no kernel construction, no
evolution, no numerical integration.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_cc_layout_keeps_beam_and_rowwise_inelasticity` | parsed record's `kind`/`beam`/`target`, the retired table dropped, explicit `Y` winning over `RS`, and `y = Q2/(x*RS**2)` per row on rows that differ in `x`, `Q2` *and* `RS` | real `exp_layout()` call vs the same formula regrouped (`A1`), plus record counts/labels (`F1`) | atol 1e-15, rtol 0.0 (measured achieved 0.0; broadcast mutants miss by 2.6e-1 / 1.3e-1) | `S2` |
| 2 | `test_cc_builder_is_in_ordinary_dis_route` | `"sigma_r_cc" in _EXP_BUILDERS` **and** the key `is` `build_sigma_r_cc`, through the dict and through `_experimental_builder`; `"synthetic_cc" not in TEST_MODES`; 10031/10032 tagged `sigma_r_cc` | dict/tuple membership + function identity + attribute equality (`F1`) | none (structural) | `S3` |
| 3 | `test_stale_f3_proxy_manifest_is_rejected_with_regeneration_instruction` | `_experimental_builder("f3_proxy")` raises `RuntimeError` naming `--remake-data` | exception type + message substring (`F1`) | none (structural) | `S3` |
| 4 | `test_cc_dataset_builder_receives_even_valence_beam_and_y` | `build_sigma_r_cc` forwards `beam_charge`/`y`/`order` by value, and the basis maps **per key** -- which physical field landed under each basis key, by tag and by object identity | captured kwargs from a monkeypatched fake `from_file`, against `dict(cfg.EVEN_MAP)`/`dict(cfg.ODD_MAP)` (`F1`) | none (structural) | `S3` |
| 5 | `test_cc_builder_falls_back_to_sqrt_s_when_the_record_has_no_y` | a record with no `y` reaches `from_file` as `sqrt_s`, cast to `float` (supplied as a string so the cast is observable), and no `y` is invented | captured kwargs from the same fake (`F1`) | none (structural) | `S3` |
| 6 | `test_cc_builder_forwards_the_normalization_only_with_exp_nuisances` | on a record carrying `rel_norm`, `with_exp_nuisances` gates `normalization`/`fit_normalization` | captured kwargs, parametrized True/False (`F1`) | none (structural) | `S3` |

**Weak spots**

- **RESOLVED 2026-08-14 — the `TARGET_MAP` asymmetry is settled, and is NOT a
  defect.** The full suites' `dis_audit` reads the target off the fitpack spreadsheet
  (`p`/`d` on all 13 `used` tables); the `_small` suites' writes `spec.target` from
  `config.EXP_SPECS` (`proton`/`deuteron` on all 7), so the label passed through
  verbatim already equals what the full suites' map produces. Nothing downstream can
  tell them apart: `pixel.util.flavor.normalize_target`, reached from every DIS builder
  through `_dis_common`, maps both to the same canonical target and raises on anything
  outside its alias table.
  `test_target_labels_reaching_the_builders_agree_with_pixels_alias_table` pins
  `TARGET_MAP` as a strict subset of PIXEL's aliases, the equivalence of both forms,
  and that every label in each shipped manifest normalizes — with a control asserting
  an unknown label raises. Acceptance: setting `TARGET_MAP['d'] = 'neutron'` fails
  exactly the two full-suite parametrizations.

- RESOLVED 2026-08-13 (was: test 1's fixture had `Q2/x=1e5` and `RS=318.0` on both rows, so a
  single-row broadcast reproduced the array bit-for-bit). The two rows now differ in `x`, `Q2`
  and `RS`; measured against the real `exp_layout` re-exec'd from mutated source, a row-0
  broadcast misses by `2.5889e-01` and a row-mean broadcast by `1.2945e-01`, ~14 orders above
  the `atol=1e-15` bar. The bar itself is unchanged and remains pure headroom: the regrouped
  reference agrees with the source formula at exactly `0.0`.
- RESOLVED 2026-08-13 (was: test 4 compared *key sets*, so `_even_map`/`_odd_map` could bind
  every basis key to the wrong field and still pass). The fields handed in are now tagged
  sentinels and the assertion compares `{basis: field.tag}` against `dict(cfg.EVEN_MAP)` entry
  by entry, plus an `is` check per key. Acceptance: a mutant binding all five even keys to the
  gluon field now fails it, with the file's other 24 cases passing.
- RESOLVED 2026-08-13 (was: test 2 checked key presence only). `_EXP_BUILDERS["sigma_r_cc"] is
  build_sigma_r_cc` is now asserted, and again through `_experimental_builder`, so a
  copy-paste routing CC data through `build_sigma_r` fails.
- Still structural throughout. Tests 2, 4, 5 and 6 read captured kwargs from a fake that
  absorbs anything, so a kwarg *renamed* in the real `from_file`/`from_arrays` signature is
  invisible here; that path is exercised unmocked, but not through this builder, in
  `tests/test_charged_current_dis.py`.
- Test 4's `_capture_from_file` fake returns a sentinel, so nothing downstream of the builder
  is exercised: no dataset is ever constructed, and no kinematics are checked.

**Not covered here**

- Suite asymmetry found while closing `-M06` and now pinned rather than assumed: only the two
  **full** suites expand an abbreviated fitpack target label through `generate.TARGET_MAP`
  (`"p" -> "proton"`); the two `_small` suites copy `tab["target"]` through verbatim and define
  no `TARGET_MAP` at all. Test 1 asserts both behaviours per family. Whether the `_small` pair
  *should* expand is an owner question, not a test one: a manifest using `p`/`d`/`n` would
  reach their dataset builders unexpanded today.
- `_exp_nuisance_kwargs`'s `correlated_file` leg (`datasets.py:177-182`) still has no coverage
  -- it loads an `.npz` sidecar from a path derived from the dataset file, which none of these
  fixtures has. Only the `rel_norm` leg is exercised (test 6).
- The `t0` reference branch (`datasets.py:184-188`) is passed `None` by every test here.
- The physics of `ChargedCurrentSigmaR` itself (LO cross-section formula, fitpack weight
  convention, minus-vs-singlet valence routing) is covered in
  `tests/test_charged_current_dis.py`, not repeated here.
- `src/pixel/api/manifest.py`'s separate closure-manifest translator still recognizes the
  retired `"f3_proxy"` kind and has no `"sigma_r_cc"` branch -- outside this file's scope
  (flagged separately, not a gap in this file's own coverage).
