### `tests/test_closure_nnpdf_datasets.py`

**Exercises** the `datasets` manifest section end to end: `generate.nnpdf_layout`, the
coverage guard in `generate_member`, `datasets._record_target` and the `include_nnpdf`
dispatch arm, `pixel.data.closure_audit.read_nnpdf_source`, and
`pixel.data.contracts.annotate_manifest` on a populated `datasets` section. Reads two
real CHORUS commondata sets; builds no kernel and asserts no cross section.

**Claim** a contract-selected NNPDF record is materialized with the cuts, the commondata
variant, the target composition and the uncertainty split its contract declares — and a
selected record that is *not* materialized stops generation rather than being omitted.

**Oracle** `F1` for the contract/registry agreements; `D2` for the row counts and the
uncertainty structure, both recomputed from the commondata rather than restated.

**Cost** 17 cases, ~17 s, dominated by reading CHORUS ν's 1013 additive sources twice.
Nothing here assembles a kernel: `from_file` builds contributions lazily, and operator
assembly is ~1.2 s/row on the production basis (~7 min per record), which has no place in
a unit test.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_layout_materializes_exactly_the_contract_selection` | Layout ids == selected ids. | Set equality both ways. (`F1`) | exact | `S1` |
| 2 | `test_every_spec_has_a_contract_and_every_selected_contract_a_spec` | No selected id lacks a build spec. | Set containment. (`F1`) | exact | `S1` |
| 3 | `test_retained_rows_match_the_contract_and_the_closure_cut` | 520 of 607, recomputed from `cfg.DIS_Q2_MIN`/`DIS_W2_MIN`. | Mask rebuilt from commondata. (`D2`) | exact | `S1` |
| 4 | `test_the_roster_variant_is_read_not_the_audit_default` | `legacy_dw`, and no `ISOTARGCOR` source survives. | Roster lookup + source labels. (`F1`) | exact | `S1` |
| 5 | `test_the_target_composition_matches_the_contract_prose` | Config's A/Z appear in the contract's `nuclear` text. | Substring on 207.2 / 82. (`F1`) | exact | `S1` |
| 6 | `test_a_nuclear_target_is_rebuilt_from_the_manifest_mapping` | JSON mapping → `NuclearTarget`; `Z/A` = 82/207.2. | Direct. (`F1`) | exact | `S1` |
| 7 | `test_a_bare_element_name_is_still_refused` | `normalize_target("Pb")` raises. | `pytest.raises`. (`F1`) | exact | `S2` |
| 8 | `test_additive_sources_stay_out_of_the_multiplicative_nuisance_layer` | `correlated_names == ["global:CHORUSQEDRADCOR1"]` only. | Shape + name equality. (`D2`) | exact | `S1` |
| 9 | `test_the_shared_source_really_is_shared_across_the_two_records` | The declared shared source is real and `r > 0.99`. | Correlation. (`D2`) | 0.99 | `S2` |
| 10 | `test_annotate_manifest_binds_every_datasets_record_to_its_contract` | Each record annotates to the contract the layout named. | `annotate_manifest`. (`F1`) | exact | `S1` |
| 11 | `test_a_selected_contract_with_no_build_spec_stops_generation` | Dropping a spec leaves the id unmaterialized. | `monkeypatch` on `cfg`. (`F1`) | exact | `S1` |
| 12 | `test_build_datasets_emits_lattice_then_exp_then_datasets` | The positional operator split's ordering. | Parsed from source. (`F1`) | exact | `S1` |
| 13 | `test_the_nnpdf_section_is_opt_in_and_follows_the_dis_gate` | `include_nnpdf` defaults False; generation opts in; `fit.py` ties it to the DIS gate. | Signature + parsed source. (`F1`) | exact | `S1` |
| 14 | `test_generate_member_refuses_a_records_operators_length_mismatch` | The length check exists and precedes the split. | Parsed source, order compared. (`F1`) | exact | `S2` |
| 15 | `test_the_two_uncertainty_layers_are_added_once_each` | `.dat` cov == what generation wrote; nuisance layer adds **rank 1**. | Built dataset, matrix rank. (`D2`) | 1e-11 (`%.12e` round trip; realised 4.9e-13) | `S1` |
| 16 | `test_a_drawn_observation_is_consistent_with_the_written_covariance` | `chi2/ndof` ~ 1 on the written covariance. | Draw + solve. (`D2`) | 5 sigma = 0.31 | `S1` |
| 17 | `test_a_record_survives_json_and_annotation_and_still_names_its_nucleus` | layout → annotate → JSON → builder; target returns as `NuclearTarget`. | Real round trip through a file. (`F1`) | exact | `S1` |

**Three defects this file was written after.** Each was found by measurement during the
plumbing work, not by review, and each is now guarded.

1. **`closure_audit._NNPDF_SOURCE` carried a stale variant column.** It read `legacy` for
   all four neutrino entries; `nnpdf40_dis_dy_roster.yaml` reads `legacy_dw`. Not
   cosmetic: `legacy`'s 16 sources include the MULT `CHORUSISOTARGCOR`, which corrects the
   data *toward an isoscalar target*, while `legacy_dw` (1015 sources, `experiment:
   NUCLEAR`) carries no such correction — which is the whole reason the operator owes
   lead's real `Z/A`. Auditing one while predicting the other applies the isoscalar
   correction twice and nothing downstream says so. For NuTeV the two variants do not even
   share central values (`04aa6c5c98` vs `3be1e978ab` on the ν rows) and `legacy` has
   **one** uncertainty source for 45 rows against `legacy_dw`'s 1002. Fixed by deleting the
   column and deriving it from the roster; test 4 pins the result.
2. **The CHORUS contracts claimed no rows were cut.** They were promoted with
   `selected_rows == source_rows == 607` and `drop_reason: null`, while every sibling DIS
   record here records a cut (`dis.slac_p_10010` is 661 → 222) and CHORUS reaches down to
   `Q2 = 0.263 GeV²`. Test 3 recomputes the retained count from `cfg` so the contract
   cannot drift from the rule again.
3. **The first correction used the wrong `Q0`.** It assumed NNPDF theory 200's `1.65²`
   and got 470; the *closure's* input scale is `Q0² = mc² = 1.6384`, giving 520. The
   generator's own count check rejected 470 — which is precisely why that check compares
   layout against contract instead of trusting either alone.

**The guard's direction changed, and that is the point of test 11.** `generate_member`
previously raised whenever `selected_source_count` was nonzero — correct only while
nothing could be materialized, and it would have blocked generation permanently once
anything was. It now compares the selected id *set* against the materialized id set in
both directions: a promotion the generator ignores raises, and so does a record with no
contract authorizing it. Comparing sets rather than counts means a swap cannot cancel out.

**The uncertainty split is the subtle one (test 8).** `_exp_nuisance_kwargs` hands
`correlated_names` to the builder as `multiplicative_systematics`, which are referenced to
`t0`. That is right for a fitpack table, whose correlated sources are `%`-type. NNPDF types
each source individually and CHORUS carries **1013 (ν) / 113 (ν̄) additive** correlated
sources plus exactly **one** multiplicative one. Additive sources are absolute numbers no
t0 iteration rescales, so they go into the `.dat` covariance; routing them through that
kwarg would have rescaled a median 75% of the ν row variance by the theory prediction —
and would have looked like a working fit.

**A declared approximation, sized rather than waved at.** `global:CHORUSQEDRADCOR1` is the
same NNPDF nuisance in both beam records (`r = 0.998` across them, test 9), but the DIS
nuisance layer marginalizes each dataset's copy separately, so the cross-record block is
dropped. It carries a median 3–4% of each row's variance. Records declare it as
`shared_multiplicative_sources` rather than leaving it implicit.

**Weak spots / not covered**

- **No physics.** Nothing here folds a truth or asserts a cross section. The operator's
  correctness is `pixel/tests/test_charged_current_dis.py` (neutrino beams, nuclear
  targets, convergence at the production grid) and the FK asymmetry comparison. This file
  would pass with a numerically wrong kernel.
- **Generation has not been run at full scale.** `generate_member` itself is not called
  with these records. The path it takes *was* run out of band on 2026-08-22 with a toy
  truth curve, and it completed: `chorus_nb` assembled 23 blocks in 416 s, the fold came
  out finite and strictly positive (`ptp = 0.1056`), the 520x520 covariance was full rank,
  and the drawn observation gave `chi2/ndof = 0.9691` against an expected sd of
  `sqrt(2/520) = 0.062`. That is a one-off measurement, not a test -- tests 15 and 16 pin
  the covariance arithmetic it exercised, without the 7-minute assembly.

  The run completed for both records: `chorus_nu` assembled in 396 s, folded finite and
  strictly positive (`ptp = 0.178`), covariance full rank at `kappa = 3.6e+04`, and
  `chi2/ndof = 0.9311` (1.1 sigma low on an expected sd of 0.062). The manifest annotated
  to both contract ids. Test 17 covers the round trip its final step exercised -- layout,
  annotation, JSON, builder -- which matters because `datasets.py` never sees the
  in-memory record, only `manifest.json`, where the target is a plain dict.
- **Mutation-checked 2026-08-22, twice, both in memory via throwaway pytest plugins.**

  1. *`nnpdf_layout` stops applying the `W2` cut, and its own count check is removed.* The
     check was removed deliberately: leaving it in proves the **layout** guards itself and
     says nothing about whether the test would notice. Test 3 failed at **540 vs 520** —
     exactly the 20 rows `W2 > 10` removes.
  2. *The 1013/113 additive correlated sources are routed through
     `multiplicative_systematics` as well.* This is the fitpack-shaped assumption the split
     exists to prevent, and it does not crash — the fit runs, having rescaled a median 75%
     of the ν row variance by the theory prediction. **Three** tests failed: 8 (the name
     list), 15 (the rank of the nuisance delta: **114 instead of 1**) and 17.

  Tests 1, 3, 11 and 15 were additionally observed failing during development — test 3
  rejected the wrong 470 row count, and test 15 first failed on exact equality, which
  located `write_dat`'s `%.12e` round trip rather than a defect. Tests 4, 9 and 16 remain
  un-mutated.
- Test 12 parses `build_datasets` with a regex over `manifest.get("...")`. A refactor to a
  dispatch table would make it find nothing and fail — safe direction, misleading message.
