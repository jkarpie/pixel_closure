### `tests/test_closure_dy_published_kinematics.py`

**Exercises** every suite's `dy_audit` (`tau_from_rtau`, `audit_table`, `build_dy_manifest`)
and `generate.dy_layout`; the six full suites' `config.COLLIDER_DY_SPECS`,
`generate._mass_edges` / `_collider_dy_record` and `datasets.collider_exchange` with the
three collider builders; and through them `pixel.data.DrellYan`, `DYBinIntegratedKernel`
and `pixel.kernels.drell_yan.electroweak`/`couplings`. Reads two fitpack workbooks per
suite and each selected record's NNPDF commondata once. No kernel tensor is assembled.

**Claim** each Drell-Yan row carries the kinematics, the mass window and the exchange its
publication states — on the generation side and on the fit side alike.

**Oracle** `D2` for the kinematics and the windows, both read rather than restated: E866's
own published table (the LANL `pp` release NNPDF ships as `DYE866_Z0_800GEV/rawdata`, which
carries `Min`/`Max`/`<M>` per row) and each set's HEPData mass qualifier. `F1` for the
wiring assertions.

**Cost** 36 cases, seconds. Record the command, not the wall time:
`venv/bin/python -m pytest tests/test_closure_dy_published_kinematics.py -n 2 -q`.

Written for the three owner decisions of 2026-09-11 (PIXEL's
`plans/drell_yan_published_units.md`, W7.7): **D6** the E866 `Rtau` reading, **D7** the
published mass window in place of the Z pole, **D8** `gamma*/Z` for collider Drell-Yan.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_e866_row_zero_sits_in_its_published_mass_bin` | Row 0 of 10001 is in E866's `4.2–4.7 GeV` bin at `<M> = 4.64`, and the old `tau = Rtau` reading is **not**. | Published table, both readings computed. (`D2`) | bin containment; `0.05 GeV` of `<M>` | `S1` |
| 2 | `test_every_audited_e866_pp_row_lands_in_its_published_mass_bin` | All 184 `pp` rows lie in their own published bins and reproduce their `<M>`. | Per-row, with the workbook↔table ordering *checked* by central-value identity. (`D2`) | bin containment; `0.05 GeV`; `rtol=1e-12` on the identity | `S1` |
| 3 | `test_the_audited_e866_row_counts_are_the_contract_counts` | Retained rows equal the contract's `selected_rows` and the table's full length; the old reading cut fewer. | Mask recomputed from the workbook. (`D2`) | exact | `S1` |
| 4 | `test_a_dy_manifest_audited_under_the_old_reading_is_rebuilt` | A manifest without `tau_convention` is re-audited; one carrying it is **not**. | Stale manifest written to `tmp_path`; `augment_manifest` call-counted. (`F1`) | exact | `S1` |
| 5 | `test_every_collider_z_record_integrates_its_published_mass_window` | The nine windows equal their HEPData qualifier squared; every other dilepton record has non-zero `Q2` width. | Qualifier parsed from rawdata. (`D2`) | `rtol=1e-12` | `S1` |
| 6 | `test_a_z_record_left_at_the_pole_is_refused` | A dilepton spec without `mass_edges` raises; the gated W arm keeps the pole. | `pytest.raises` + a real W record. (`F1`) | exact | `S1` |
| 7 | `test_collider_builders_take_gamma_z_and_fixed_target_stays_photon` | All three collider arms are `gamma_z` with the provider's own running `alpha_EM`, the mass axis takes the Z pole, the unit is `GeV^-2`; E866 stays photon-only at `GeV^-4`. | Built datasets; `_row_couplings` *raises* on an inconsistent `alpha_EM`. (`F1`) | `rel=1e-13` on `alpha_EM` | `S1` |
| 8 | `test_the_window_survives_the_manifest_into_the_fit_builder` | Layout → `_public` → JSON → builder keeps the window and the unit. | Real round trip through a file. (`F1`) | `rtol=1e-12` | `S1` |

**The premise each test was written against, and why it is not self-referential.** The two
numbers that decide D6 and D7 — E866's mass bins and each set's mass window — are read here
from the experiment's own published table and from HEPData, not from the closure's config.
So test 5 compares the interval PIXEL integrates against the interval the experiment
published, rather than comparing `COLLIDER_DY_SPECS` with itself. Test 2's ordering check is
what makes its per-row comparison meaningful: the audit addresses workbook rows by index, and
without requiring each row's central value to match the published table at the same index, a
per-row bin comparison could be pairing a row with another row's bin and still pass.

**Mutation-checked 2026-09-11, five arms, all in memory via throwaway pytest plugins**
(`w77_mut_*.py`; baseline `36 passed`):

1. *`tau_from_rtau` returns `Rtau` unsquared* — the pre-2026-09-11 reading. **32 failed, 4
   passed.** Every E866 case in every suite fails.
2. *`TAU_CONVENTION = None`*, i.e. no marker, which is how a stale manifest used to be
   reused. **8 failed** — test 4 in all eight suites, and nothing else.
3. *The nine specs lose `mass_edges`* (back to the pole). **4 errors**: the new guard in
   `_collider_dy_record` refuses at layout time, so the collider tests never run. Evidence
   that the guard fires, but *not* evidence about the tests — which is why arm 4 exists.
4. *Records built with the window stripped afterwards*, reproducing exactly what the old code
   returned, with the guard left in place. **3 failed** — tests 5, 7 and 8.
5. *`collider_exchange` returns `{}`* (photon-only again). **1 failed** — test 7, and only it.

**Weak spots / not covered**

- **No physics value.** Nothing here folds a truth or asserts a cross section. Integrating
  the window instead of evaluating the pole changes the prediction by roughly the width of
  the window, and no independent prediction of that number exists in this repository; what
  is pinned is that the operator integrates the published interval with the published
  exchange. A wrong `gamma*/Z` weight table would pass every test in this file.
- **`pd` (10002) is checked more weakly than `pp`.** The NNPDF commondata ships the published
  `pp` table only, so the `pd` rows are covered by tests 3 and 4 (counts, re-audit) but not
  per-bin. Its row 0 lands at `M = 4.97 GeV`; the published `pd` table would place it.
- **Test 7 builds with `DY_EVOLVE` off**, so no evolution operator is composed. The exchange
  and the edges are independent of it, but this file therefore says nothing about the
  interaction between callable evolution and the Breit-Wigner mass nodes — which is where
  `DYBinIntegratedKernel.evolved_tensor`'s recorded node-set defect lives (see its docstring
  in `pixel`; `evolved_tensors`, the path the closure takes, is the arm that closes it).
- **The windows are pinned per table, the qualifier name per set.** A set whose HEPData table
  is replaced by a different table with the same qualifier name would not be noticed.
