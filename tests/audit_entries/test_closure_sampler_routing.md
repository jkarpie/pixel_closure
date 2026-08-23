### `tests/test_closure_sampler_routing.py`

**Exercises** `closure_*/fit._closure_sampler` and `Model.has_nonlinear` against
`pixel.data.nnpdf40_native`'s `_SAMPLER_ROUTES` / `requires_direct_sampler` and
`core.model.dataset_nonlinear_kind`. Four config imports and one small `F2Ratio`; no
sampling, no kernels beyond that dataset's lazy construction.

**Claim** the suite dispatches to the direct sampler for exactly the prediction classes the
registry says cannot be marginalized — no more, and no fewer.

**Oracle** `F1`. Cross-module and cross-repo consistency; there is no external reference
for "which sampler applies".

**Cost** 10 cases, ~1.5 s.

**Why it exists.** `pixel` derives the routing fact — marginalized evidence needs a
prediction polynomial in `q`, so every exact ratio is direct-only — from the prediction
class. The closure derives the same fact independently from `model.has_nonlinear`. Two
derivations of one fact in two repos is the shape that drifts, and the failure is quiet in
the dangerous direction: a ratio routed to `nuts` does not crash, it marginalizes a
likelihood it may not marginalize. The in-repo half of the same argument already caught a
real drift — `evidence_value` and `Model._require_gaussian_likelihood` disagreed on
`mixed_ratio` (see `pixel/plans/test_audit/entries/test_sampler_routing.md`).

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_a_nonlinear_model_goes_to_the_direct_sampler[×2]` | `has_nonlinear` → `direct_nuts`; `nuts` and `hmc` still reachable. | Synthetic model with only what the chooser reads. (`F1`) | exact | `S1` |
| 2 | `test_the_dispatch_reads_has_nonlinear_and_nothing_narrower[×2]` | Not dispatched on one layout tuple. | Parsed source. (`F1`) | exact | `S1` |
| 3 | `test_model_has_nonlinear_covers_every_direct_only_kind` | The property names all three layout tuples. | Parsed source. (`F1`) | exact | `S1` |
| 4 | `test_registry_direct_only_classes_are_the_nonlinear_kinds` | Prediction-class and runtime-kind vocabularies name the same set. | Set equality. (`F1`) | exact | `S1` |
| 5 | `test_no_roster_dataset_the_closure_can_build_is_misrouted` | Every direct-only roster dataset routes to `direct_nuts`. | Walks the real roster. (`F1`) | exact | `S1` |
| 6 | `test_the_f2_ratio_dataset_reports_the_kind_the_registry_assigns_it` | A built `F2Ratio` reports `linear_ratio`, matching the roster. | Real dataset. (`F1`) | exact | `S1` |
| 7 | `test_the_smoke_suites_cannot_build_a_dataset_their_sampler_cannot_route[×2]` | The smoke suites' builder tables carry no ratio. | Builder-table equality + source scan. (`F1`) | exact | `S2` |

**What the first run found, and it is the reason test 7 exists.** Parametrizing the
dispatch assertions over all four suites failed on both `_small` ones: their
`_closure_sampler` is a *different function* — it takes a `mode`, returns
`"vegas"`/`"gaussian"`, and never consults `has_nonlinear`. That is safe today only because
those suites cannot build a ratio at all (`_EXP_BUILDERS` is exactly
`{f2, sigma_r, sigma_r_cc}`), and the precondition is now pinned rather than assumed. If a
ratio ever did reach one, the consequence would at least be loud: `evidence_value` refuses
any nonlinear block outright.

**A vocabulary mismatch worth knowing about, deliberately preserved.** A `linear_ratio`
dataset lands in the layout tuple named `nonlinear_datasets` — `infer/direct.
_direct_prediction_layout` maps the two explicitly. So the *prediction class* and the
*layout tuple name* differ for that one class, and test 3 accounts for it rather than
asserting a naming symmetry that does not hold.

**Weak spots / not covered**

- **Routing is not convergence.** Nothing here runs a sampler. That a ratio model is sent
  to `run_direct_mcmc` says nothing about whether it mixes or is affordable.
- Test 5 checks the dispatch's *answer* for a synthetic nonlinear model, not that building
  each roster dataset actually produces `has_nonlinear == True`. Test 6 does that for
  NMCPD only; the six `bilinear_ratio` datasets are not built here.
- Tests 2, 3 and 7 parse source with regexes and substring checks. A refactor into a helper
  would fail them — safe direction, misleading message.
