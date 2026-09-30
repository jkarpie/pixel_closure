# Plan + handoff: "stiff" closure fits — sampler test, VEGAS docs, truth-derived errors (2026-09-14)

Links are relative to the `pixel_closure` repo root; `../pixel/` is the PIXEL checkout.
This file doubles as the handoff and is copied to `pixel_closure/HANDOFF_precision_2026-09-14.md`
(step 0) and updated as each step lands.

## Context

- **The problem:** in the small closure suites (`closure_{JAM,NNPDF}_truth_small`), every Drell–Yan fit (`dy`, `exp`, `both`) fails at every Q, under both prior forms. NNPDF `dis` fails with `const_logrbf`.
- **The symptoms:**
  - nested-VEGAS ESS 0.001–0.01 against gates of 0.1 / 0.05 / 0.1;
  - "H-S saddle search did not converge";
  - float64 overflow.
- **Diagnosis:** this is the **"stiff" regime** PIXEL's plans already describe. The data, DY with realistic errors, are far narrower than the prior, and importance sampling collapses.
- **Owner decisions (2026-09-14):**
  1. **Highest priority:** test the Gaussian sampler and NUTS on every failing case.
  2. Add a "stiff regime" warning to the VEGAS docs.
  3. Make every pseudo-data statistical error come from the truth: **diagonal** replica spread, **keeping** the injected experimental nuisances (DIS normalizations and correlated systematics, DY 7% normalization). The closure then tests whether the final uncertainty is under- or over-estimated.
  4. Run the sampler test at **Q=2 first**, then Q=mc and Q=3 for whichever sampler works.
- **Repo state:**
  - PIXEL on branch `constants-migration-plan`, uncommitted DY momentum-density fix, all tests green.
  - pixel_closure on `main`, uncommitted: closure `momentum_density=True`, the prior-scan harness, the comment fix.
  - **No commits unless asked.**

---

## Step 0 — store the handoff
Copy this file to `pixel_closure/HANDOFF_precision_2026-09-14.md`, then update its status table and results sections after each step.

## Step 1 (highest priority) — Gaussian sampler and NUTS on every failing case, Q=2

**Cases.** Failing cells from the post-fix baseline (`baseline_fixed/summary.tsv`); 16 cells in all.
- **Priors:** shipped `c_config` (`const_logrbf`) and `e_config` (`beta_envelope`).
- **Failing cell:** a gate failure, saddle failure, overflow or error, or synthetic Z with χ²/n ≥ 3.

| prior | JAM | NNPDF |
|---|---|---|
| `const_logrbf` | dy, exp, both, synthetic_z | dis, dy, exp, both, synthetic_z |
| `beta_envelope` | dy, exp, both | dy, exp, both, synthetic_z |

**Sampler routes** (from `pixel.infer._resolve_mcmc_plan`; confirm with a dry run before fitting):
- **Gaussian:**
  - `dis`: `sampler="gaussian", auxiliary="auto", run_map=True` (MAP, so the Laplace proposal is centred) → `GaussianImportanceSamples`.
  - `dy`/`exp`/`both`: `sampler="gaussian", auxiliary="auto"` → `normalization_reference_saddle_contour_moments` → `SaddleContourSamples`. Only `n_samples`, `seed`, `saddle_tol`, `reference_cache_path` are accepted.
  - `synthetic_z`: already Gaussian (`fixed_saddle_contour`).
- **NUTS:**
  - `dis`: `sampler="nuts"` → `NUTSSamples`.
  - `dy`/`exp`/`both`: `sampler="nuts", auxiliary="sample"` → `normalization_reference_contour_nuts` → `JointContourHMCSamples`.
  - `synthetic_z`: `fixed_contour_nuts`.
  - Options follow the full suites ([`closure_JAM_truth/fit.py`](closure_JAM_truth/fit.py) NUTS block): `n_warmup=100`, `step_size=0.1`, `integrator="leapfrog"`, `max_tree_depth=6`, `adapt_mass=False`, `use_hessian_mass=True`, `saddle_tol=5e-3`, and per-run `reference_cache_path` / `hessian_cache_path` under the scan `--out`.
- **Refused by PIXEL** (not run): `nuts`+`marginalize`, `hmc`+`marginalize`.

**Harness change** ([`campaign/prior_scan.py`](campaign/prior_scan.py) `cmd_run_cell` / `cmd_fit`, no suite code edits):
- **New flags:** `--sampler {vegas,gaussian,nuts}`, `--auxiliary`, `--sampler-options JSON`, `--run-map`.
  - `cmd_fit` forwards them and gets a per-sampler `--timeout` (NUTS 3 h).
- **Kwarg rewrite:** the existing `capture_infer` wrapper rewrites `run_fit`'s VEGAS kwargs.
  - Drop `n_bins`, `n_adapt_iterations`, `n_eval_per_iteration`, `alpha`, `joint_grid`, `min_adapt_phase_ess_frac`, `max_extra_adapt_iterations`, `n_adapt_inner_samples`, `n_inner_samples`, `covariance_inflation`.
  - Set `sampler`, `auxiliary`, `run_map`/`map_kwargs` and the per-sampler options.
- **Label:** patch `fitmod._closure_sampler` so `summary["sampler"]` is correct; restore it in `finally`.
- **Gates after the call** (they raise with "ESS" or "saddle" so rows classify as `gate_fail`; every diagnostic is stored in the row):
  - Gaussian `dis`: `ess_fraction ≥ 0.10`; record `max_weight`, `n_invalid`.
  - Gaussian bilinear: `ess_frac ≥ 0.10`, `saddle_converged`; record `residual`, `weight_magnitude` and `phase_only_ess_fraction`, `proposal_fit_success`, `proposal_scale`.
  - NUTS: `n_divergent == 0`, `mass_initialization != identity_fallback`, `tau_reliable`; record `ess_min`, `mean_accept_prob`, `n_max_depth`. For bilinear also `residual_ess_fraction ≥ 0.10`, `moved_fraction > 0`, `saddle_converged`; record `branch_crossings`, `action_phase_jumps`.
- **Row paths** include the sampler: `rows/<suite>__q<Q>__<mode>__<cell>__<sampler>__<hash>.json`.
- **Tests** in [`tests/test_campaign_prior_scan.py`](tests/test_campaign_prior_scan.py) (mutation-checked):
  - the kwarg rewrite for each sampler and mode;
  - gate classification on stub sample objects;
  - the row path includes the sampler.

**Schedule** (two streams, JAM ∥ NNPDF, 7 cores each; output `campaign/sampler_test_2026-09-14/`):
1. **Dry run:** `_resolve_mcmc_plan` for every (case, sampler); print the resolved plan.
2. **Gaussian, all 16 cases** (~3–8 min each, about 1–1.5 h wall).
3. **NUTS on `dy`** (4 cases), then `synthetic_z` and `dis`. Pre-fix contour NUTS on `dy` projected about 45 min per fit.
4. **NUTS on `exp`/`both`** (8 cases, more expensive per step; about 1–3 h each, possibly 8–12 h wall).
5. **Report:**
   - status grid per sampler, gate diagnostics, χ²/n, 9-field pull tables;
   - dataset-reproduction and PDF-reproduction plots for every passing case (reuse `run_closure.plot_reproduction` / `plot_datasets`);
   - side-by-side comparison against the VEGAS baseline grid.

**Stop and ask:** after steps 2 and 3, show the results before spending the NUTS `exp`/`both` hours, and before Q=mc/3.

**Then:** rerun Q=mc and Q=3 for the failing cases with the sampler(s) that passed.

## Step 2 — "stiff regime" note in the VEGAS docs (PIXEL, doc-only)

Add a short warning, with evidence and links, in three places:
- [`guides/choosing_a_sampler.md`](../pixel/guides/choosing_a_sampler.md) `### "vegas"` section ([:179](../pixel/guides/choosing_a_sampler.md:179)).
  - A paragraph "VEGAS fails in stiff regimes". Case: Drell–Yan with realistic errors, where the data are much narrower than the GP prior (prior-predictive sd over data sd ~10²–10⁴), so outer and inner ESS collapse to 0.001–0.01.
  - Measured 2026-09-13/14 on the small closure suites after the DY momentum-density fix.
  - Link the stiff-regime definition ([`plans/contour_deformation_plan.md:17`](../pixel/plans/contour_deformation_plan.md:17)), the "Gentle regime only" limitation ([`plans/vegas_adaptive_importance_sampler.md:611`](../pixel/plans/vegas_adaptive_importance_sampler.md:611)) and the H-S sign-problem section ([`guides/hubbard_stratonovich_contour.md:227`](../pixel/guides/hubbard_stratonovich_contour.md:227)).
  - Say what to use instead once step 1 has measured it.
- Module docstring of [`src/pixel/infer/vegas.py`](../pixel/src/pixel/infer/vegas.py): one paragraph pointing to the guide section.
- Module docstring of [`src/pixel/infer/nested_vegas.py`](../pixel/src/pixel/infer/nested_vegas.py) and the `nested_vegas_importance_sample` docstring: same, naming DY with realistic errors as the known stiff case.

**Constraints:**
- Both modules are **outside** the kernel fingerprint closure (checked: the `infer` members are `__init__`, `_cache`, `contour`, `gamma_method`, `hmc`, `kl`, `mcmc`, `nuts`), so no cache invalidation.
- Do **not** touch `infer/__init__.py` or `contour.py`.
- Keep measured numbers in the guide only, not in docstrings (PIXEL CLAUDE.md).

**Verification:** `./run_tests.sh tests/test_guide_references.py tests/test_docstring_quality.py tests/test_readme_completeness.py`.

## Step 3 — truth-derived diagonal statistical errors for DY and synthetic Z (small suites)

**Decided:** diagonal replica spread; experimental nuisances kept. DIS and lattice already use it, so only DY and synthetic Z change.
- **New function** `dy_replica_sigma(dataset, ensemble, fields)` in both `closure_*_small/generate.py`. It computes the per-replica bilinear prediction `q_rᵀ Y q_r` over `iter_bilinear_group_tensors` and returns `std(ddof=1)` per row.
  - It is kept **separate** from `dy_central`: [`tests/test_closure_truth_representable.py`](tests/test_closure_truth_representable.py) requires the four `dy_central` bodies to be AST-identical.
  - The S0-05 ruling constrains only the *central* value ([`../pixel/plans/test_suite_hardening.md:693`](../pixel/plans/test_suite_hardening.md:693)).
  - The truth law allows replica moments as uncertainty ([`../pixel/plans/closed/closure_ew_lhc_extension.md`](../pixel/plans/closed/closure_ew_lhc_extension.md) §"truth law").
  - Measured cost about 1 s per E866 table and 0.02 s for synthetic Z.
- **Generation:**
  - DY σ = `dy_replica_sigma` + `_dy_floor`, replacing `rel_stat·|central|`; synthetic Z likewise, replacing the invented 3%.
  - The `rel_norm` injection and marginalization stay.
- **Measured replica σ/|central| at Q=2:**

| table | replica σ/\|central\| | today's real `rel_stat` |
|---|---|---|
| JAM pp | 0.034–0.51 | 0.042–1.0 |
| JAM pd | 0.044–1.31 | 0.039–1.0 |
| NNPDF pp | 0.031–0.33 | same as JAM pp |
| NNPDF pd | 0.031–0.56 | same as JAM pd |
| synthetic Z | JAM 4.3%, NNPDF 2.2% | invented 3% |

  - So DY mostly gets **tighter**, i.e. stiffer.
  - One edge row has σ > 100%: flag it in the report, keep the Gaussian draw.
- **Tests** (new `tests/test_closure_truth_derived_errors.py` + audit entry):
  - `dy_replica_sigma` equals an independent per-replica Python loop on a replica subset;
  - generated DY/synthetic-Z σ equals `dy_replica_sigma`;
  - DIS and lattice data are unchanged.
- **Docs:** fix [`closure_*_small/README.md:122`](closure_JAM_truth_small/README.md:122) (errors from truth; systematics *are* injected) and [`../pixel/guides/running_closure_tests.md:158`](../pixel/guides/running_closure_tests.md:158) "The truth" item 4.
- **Regenerate:** `--remake-data` for Q = mc, 2, 3. Hash check: only DY and synthetic-Z files plus `manifest.json` change.
- **Rerun** the failing cases with the sampler(s) that passed step 1, then run the coverage report (does the fit's uncertainty match the truth spread?).

---

## Handoff notes (what was tested, what was done before, what can be done, what needs study)

### A. Tested this session (measured)
1. **Unit rescale (PIXEL W6) is not the cause.**
   - A/B test: the H-S failures and synthetic-Z χ² are identical with it on and off.
   - Off, NNPDF `exp` overflows (log magnitude 838).
   - Rescaling and unit changes are exact changes of units, so they cannot fix stiffness.
2. **DY kernel treated x·f fields as number densities — fixed.**
   - LO predictions were τ too small.
   - Record: [`../pixel/plans/dy_momentum_density.md`](../pixel/plans/dy_momentum_density.md), [`../pixel/tests/test_dy_momentum_density.py`](../pixel/tests/test_dy_momentum_density.py), [`tests/test_closure_dy_momentum_density.py`](tests/test_closure_dy_momentum_density.py).
   - Full PIXEL suite green (6912 passed); closure suite 419 passed / 10 skipped.
   - Data regenerated: only DY and synthetic Z changed; synthetic Z now matches hand LO to 0.24%.
   - Pass/fail unchanged in every cell (expected at LO).
3. **Post-fix baseline** (2 priors × 2 truths × Q mc/2/3; scratchpad `baseline_fixed/`):
   - lattice passes everywhere;
   - DIS: `const_logrbf` fails for NNPDF, `beta_envelope` passes both;
   - `dy`/`exp`/`both` fail everywhere.
4. **Prior vs data** (Q=2, `dy`, no sampling): truth reproduces the data (truth-pull RMS 1.16), but the prior is far wider and higher than the data.

| prior | prior-predictive sd / data sd | prior-mean / data |
|---|---|---|
| `const_logrbf` | ~1,600 | ~100 |
| `beta_envelope` | ~15,000 | ~3,500 |

5. **Fixed-prior scan** (harness in `campaign/prior_*.py`, [`tests/test_campaign_prior_scan.py`](tests/test_campaign_prior_scan.py); rows in `campaign/prior_scan_2026-09-13/`):
   - **Stage-1 screen, 395 cells:** best covering `const_logrbf` (sea/gluon ≈ 1, e.g. `ct_k0p5`) cuts the ratio ~1,600 → ~75. `beta_envelope` stays ≥ ~3,000.
   - **Stage 2:** length/jitter barely matter.
   - **DY probe on the two best cells:** the saddle converges, but ESS is still 0.007–0.009 (signed 0.001–0.002, inner ~0.001).
   - **Stage 3, lattice + DIS, 29 cells:** lattice passes all.
     - DIS: all 14 `beta_envelope` cells pass both suites (ESS 0.19–0.66).
     - All narrowed `const_logrbf` cells fail both suites (ESS 0.005–0.10); shipped `c_config` passes JAM only (0.113 / 0.024).
6. **Pseudo-data precision vs real data** (Q=2):

| data | pseudo σ/value vs real stat |
|---|---|
| DIS | 0.5–1.4% vs 1.4–7.2% (2–10× tighter; replica spread) |
| DY | rel_stat 24–57% (real) |
| lattice | replica spread 0.3–5% |

### B. What was done before (guides read, quotes checked with `grep -n`)
- [`running_closure_tests.md:158`](../pixel/guides/running_closure_tests.md:158) "The truth" item 4: "the fake-data covariance is the replica covariance of the folded output". True for small-suite DIS and lattice only.
- [`hubbard_stratonovich_contour.md:227`](../pixel/guides/hubbard_stratonovich_contour.md:227): "For precise bilinear data … its effective sample size collapses." See also [:347](../pixel/guides/hubbard_stratonovich_contour.md:347), [:653](../pixel/guides/hubbard_stratonovich_contour.md:653), [:694](../pixel/guides/hubbard_stratonovich_contour.md:694).
- [`choosing_a_sampler.md`](../pixel/guides/choosing_a_sampler.md):
  - [:51](../pixel/guides/choosing_a_sampler.md:51) ESS 0.1 trust threshold;
  - [:475](../pixel/guides/choosing_a_sampler.md:475) and [:516](../pixel/guides/choosing_a_sampler.md:516) the "vegas" choice "worth revisiting";
  - [:617](../pixel/guides/choosing_a_sampler.md:617) nested VEGAS on `dy` never completed, and proposal inflation biased the answer;
  - [:634](../pixel/guides/choosing_a_sampler.md:634) earlier agreement was prior-dominated; "A sampler cannot fix a badly posed target".
- [`numerical_regularization.md`](../pixel/guides/numerical_regularization.md):
  - [:52](../pixel/guides/numerical_regularization.md:52) jitter;
  - [:155](../pixel/guides/numerical_regularization.md:155) and [:164](../pixel/guides/numerical_regularization.md:164) a constraint σ of 1e-6 gives ESS 0.001;
  - [:199](../pixel/guides/numerical_regularization.md:199) diagnosis order.
- [`experimental_data_and_t0.md`](../pixel/guides/experimental_data_and_t0.md) [:503](../pixel/guides/experimental_data_and_t0.md:503), [:567](../pixel/guides/experimental_data_and_t0.md:567): t0 corrects normalization bias, not precision.
- [`tying_parameters.md`](../pixel/guides/tying_parameters.md) [:139](../pixel/guides/tying_parameters.md:139), [:155](../pixel/guides/tying_parameters.md:155), [:194](../pixel/guides/tying_parameters.md:194): amplitude collapse.
- [`normalizing_flow_surrogate.md:432`](../pixel/guides/normalizing_flow_surrogate.md:432): direct (θ, q) sampling has no H-S phase (early).
- [`units_and_normalization.md:106`](../pixel/guides/units_and_normalization.md:106): the closure uncertainty floor.

### C. What was done before (plans and history)
- **"Stiff regime" definition:** [`contour_deformation_plan.md:17`](../pixel/plans/contour_deformation_plan.md:17).
  - VEGAS is "Gentle regime only" ([`vegas_adaptive_importance_sampler.md:611`](../pixel/plans/vegas_adaptive_importance_sampler.md:611)); the closure DY block had ESS 0.01–0.08 with a non-converged saddle ([:512](../pixel/plans/vegas_adaptive_importance_sampler.md:512)).
  - ESS falls 0.043 → 0.001 as the datum moves away from the prior prediction ([`closed/wt_node-bilinear-tensor-truth.md:389`](../pixel/plans/closed/wt_node-bilinear-tensor-truth.md:389)).
- **Small-suite DIS switched `rel_stat` → replica spread** in PIXEL `6c6377d` (2026-07-12); [`closed/closure_plan.md:247`](../pixel/plans/closed/closure_plan.md:247), [:267](../pixel/plans/closed/closure_plan.md:267).
- **Full suites reverted to real `rel_stat`** at the owner's request, `5ec0ae9` (2026-07-22): [`closed/closure_extension.md:3`](../pixel/plans/closed/closure_extension.md:3), [`closed/closure_extension_plan.md:180`](../pixel/plans/closed/closure_extension_plan.md:180). The small suites were never aligned.
  - DY got `rel_stat` then for a computational reason: the old second-moment central collapsed the replica stack.
- **DY errors were ~15× too large until 2026-09-10:** [`closure_missing_observables.md:155`](../pixel/plans/closure_missing_observables.md:155), lesson [§9.14](../pixel/plans/closure_missing_observables.md:771).
  - So every earlier DY/exp/both pass (71/71 campaign, sampler benchmark) had near-uninformative DY. That link is an inference; no plan states it.
- **Constraint σ cliff:** 1e-6 gives ESS 0.001, 1e-4 passes ([`closed/closure_config_derivable_handoff.md:244`](../pixel/plans/closed/closure_config_derivable_handoff.md:244)).
  - Same file: g width ceiling ([:160](../pixel/plans/closed/closure_config_derivable_handoff.md:160)); "No amplitude choice fixes a prior-dominated field" ([:186](../pixel/plans/closed/closure_config_derivable_handoff.md:186)); envelope "more robust, less accurate" ([:487](../pixel/plans/closed/closure_config_derivable_handoff.md:487)); overflow at 760 ([:649](../pixel/plans/closed/closure_config_derivable_handoff.md:649)).
- **Samplers:**
  - joint affine grid decided ([`sampler_comparison_benchmark.md:70`](../pixel/plans/sampler_comparison_benchmark.md:70));
  - flow-time tempering "NOT MET" ([`joint_theta_f_contour_hmc.md:1246`](../pixel/plans/joint_theta_f_contour_hmc.md:1246));
  - automatic low-ESS fallback refused ([`closed/sampler_interface_refactor.md:523`](../pixel/plans/closed/sampler_interface_refactor.md:523)).
  - Pre-fix `dy`: gaussian ESS 1.0 (prior-dominated), contour HMC 0.0014, contour NUTS never run. `exp`/`both` never measured with gaussian or NUTS.

### D. What can be done
- **Step 1:** Gaussian sampler / NUTS instead of VEGAS.
- **Step 3:** truth-derived errors.
- **`beta_envelope` for DIS-containing modes** (stage-3 evidence).
- **Narrow sea/gluon amplitudes:** only helps DY past the saddle search.
- **Log-space evidence normalization** to avoid overflow.
- **Direct (θ, q) sampling or flow tempering:** PIXEL work.
- **Not a fix:** rescaling or units.

### E. Needs study
- ESS as a function of an error multiplier per sampler, and the information ratio each sampler tolerates.
- Re-measure the old DY sampler conclusions now that DY is informative.
- The DY evolution-projection density convention for the full suites ([`dy_momentum_density.md`](../pixel/plans/dy_momentum_density.md) rows 6–7).
- Synthetic Z under `const_logrbf` (fit mean ≈ 0 while truth fits the datum).
- Coverage after step 3: under- or over-estimated uncertainty.
- The stale docs listed in step 3.

## Step 1 results so far (2026-09-14, Q=2)

Harness: `campaign/prior_scan.py --sampler {vegas,gaussian,nuts}` (+ `--saddle-strategy`, experimental);
tests `tests/test_campaign_prior_scan.py`. Diagnostics: `campaign/contour_profiles.py`
(`run`/`shapes`/`radial`/`plot`/`plot-radial`), output `campaign/contour_profiles_2026-09-14/`.

### F1. Gaussian, all 16 cells (`campaign/sampler_test_2026-09-14/rows`)
- NNPDF `dis` passes under both priors (ESS 0.24; χ²/n 0.55 `c_config`, 0.30 `e_config`). JAM synthetic Z `c_config` passes the gate (ESS 0.9997) but χ²/n = 1126.
- Every DY-bearing cell fails, in two distinct ways:
  - **Reference saddle not found** (5): JAM `dy` c, NNPDF `dy` c, NNPDF `both` c, NNPDF `exp` e; NNPDF synthetic Z c ("complex importance normalization vanished").
  - **Saddle converged, ESS ≈ 1/N** (7): JAM `dy` e, `exp` c/e, `both` c; NNPDF `dy` e, `exp` c, `both` e; also NNPDF synthetic Z e (0.026). The collapse is in the weight **magnitude** (magnitude ESS 0.001–0.017); phase ESS is 0.88–0.99 in the `dy`/`both` `e_config` cells.
- So fixing the saddle search alone cannot rescue the Gaussian sampler: 7 of 12 DY-bearing cells already have a converged saddle and still fail.

### F2. The saddle search (`scratchpad saddle_starts.py`, pipeline model incl. DIS t0)
- `find_saddle`'s fallback start `blocks.hubbard_f` is **zero** in these models, so it duplicates the zero start.
- **t0 matters**: JAM `exp` c converges from zero with the t0 covariance (56 it.) and fails without it (decrement 2.06). The earlier standalone "exp saddle failure" diagnostics built the model without t0.
- Tally over both suites × both priors × {`dy`; `exp`/`both` with and without t0}, PIXEL Newton at tol 5e-3: the zero start converges in 2/4 `dy` and 10/16 `exp`/`both` configurations; μ converges in 3/4 `dy` and **0/16** `exp`/`both`. t0 flips the outcome in both directions (JAM `exp` needs it; NNPDF `exp` `e_config` converges only without it).
- **μ = iC⁻¹m is not a safe start**: it converges JAM `dy` c (14 it.), but fails NNPDF `dy` c (decrement 4.2) and runs away on NNPDF `exp` c (|f|→337) and NNPDF `both` c, JAM `exp` c, where zero is fine or better. The `shifted_diagonal` pilot broke JAM `exp` c, which converges under the default. **Not promoted.**
- A decrement-converged saddle can be on the wrong sheet: JAM `exp` c has Im E* = π and frame profiles that *rise* away from it (`contour_profiles.png`); nothing in the pipeline checks this.

### F3. Why a converged saddle still gives ESS ≈ 1/N (`contour_profiles.png`, `radial_profiles.png`)
- On PIXEL's affine steepest-descent contour the DY integrand is Gaussian only for |u| ≲ 2. Beyond that −½ Re ΔE² falls roughly like −3√r (31 at r = 100, 95 at r = 1000). The phase stays small (|½ Im ΔE²| ≈ 1 at r = 30), so nothing cancels.
- In 12 frame dimensions the radial mass r¹¹ e^{−½ReΔE²} peaks at r ≈ 45 (JAM `dy` e) and r ≈ 8 (NNPDF `both` e). The Laplace proposal draws r ∈ [2.1, 5]. Control: JAM synthetic Z stays exactly −r²/2 to r = 1000.
- Proposal changes tested on the same frame, none reaching the gate in 12-D: Gaussian widths 0.4–20, Student-t ν = 1–10 at scales 1–10 (best direct: t(2, 10) magnitude 0.072, signed 0.016), and 8 rounds of adaptive full-covariance t fitting (signed ≤ 0.02, unstable). In 1-D (NNPDF synthetic Z e) t(3, 1.5) does fix it (signed 0.093 → 0.62).
- PIXEL's own width pilot scans only 0.25–1× Laplace, i.e. the wrong direction for this integrand.

### F4. Contour NUTS
- Refused before sampling in every DY-bearing mode: projected peak 99.6 GiB vs the 16 GiB budget, dominated by `NORMALIZATION_REFERENCE_ACTION_TANGENT_COPIES · n_f · n_q²` at n_q = 1314, a constant calibrated only to n_q = 64. The projection does not depend on `n_samples`.
- With the budget raised (JAM `dy` e): the saddle converged, then the dense Hessian inverse-mass setup exceeded 22 GiB within seconds (killed by watchdog). With an explicit unit inverse mass the chain runs at ≈ 11 GiB, but warmup did not complete 10 iterations in 26 min (≥ 2 days for 1100). Stopped.
- `actual_flow_time = 9.3e-16`: `_contour_scale_and_flow_rate` divides the flow time by the largest singular value of the stiff saddle Hessian, so the holomorphic flow is effectively off and every contour route uses the affine tangent plane of F3.
- Harness fix: a `MemoryError` refusal is now classified `memory_refused`; it had been misfiled as `gate_fail` because PIXEL's message contains "normalization".

### F4b. Inflating the saddle-contour Gaussian proposal (`campaign/proposal_inflation_2026-09-14/`)
- `--proposal-scales` (`campaign/proposal_width.py`) replaces PIXEL's 0.25/0.35–1× pilot width grid without editing `contour.py`. Each row records `proposal_scale_grid_intercepted = 1`.
- Pilot over widths 1–25× on the 7 converged-saddle cells: all fail the gate. The pilot's best estimate was 1–2 draws of 48 at every width, so its choice is noise; it picked 1× in four cells. Signed ESS 0.001–0.003, not better than PIXEL's own grid (0.001–0.015).
- Fixed ladder on JAM `dy` `e_config` (`width_ladder_JAM_dy_e_config.png`): signed ESS 0.0044 (0.40×, PIXEL), 0.0021 (1×), 0.0038 (2×), 0.0030 (4×), 0.0033 (8×), 0.0021 (16×). Magnitude ESS peaks at 4× (0.018) while phase ESS falls 0.99 → 0.44 → 0.34 → NaN. Wider draws buy magnitude coverage but reach the region where the phase varies. Here the normalization directions are inflated too, unlike the frame-only study in F3.

### F4b′. Exact-width Gaussian rerun, Q=2 (`campaign/sampler_exact_width_2026-09-15/`, 16 cells)
- Harness default since 2026-09-15: `covariance_inflation = 1` on the DIS route, pilot grid exactly `(1.0,)` on the saddle-contour route (`EXACT_LAPLACE_WIDTH`).
- NNPDF `dis` `c_config`: ESS 0.998 at exact width (0.244 at PIXEL's default ×3 inflation); χ²/n 0.551 unchanged.
- Every DY-bearing cell still fails. Saddle-converged signed ESS is 0.001–0.003, at or below the narrowed-grid values (e.g. NNPDF `both` e 0.0146 → 0.0030). The five saddle failures and NNPDF synthetic Z `c_config` are unchanged, as expected, since they never reach the proposal.
- **PIXEL's proposal-precision floor still binds**: in 3–7 of the 7–9 joint (A, u) directions of every `exp`/`both` cell (0 in `dy`). There the fitted curvature is below 0.25, possibly negative, and PIXEL narrows the width to the floor. This can't be removed without editing `contour.py`.

### F4c. Plain (direct θ, q) NUTS: `pixel.infer.run_direct_mcmc`, no H-S auxiliary (JAM `dy` `e_config`, Q=2)
- Coordinates: 1268 dims (2 normalizations plus 1266 retained GP coefficients, `prior_rcond = 1e-10`), prior-whitened.
- Default Hessian mass: `jax.hessian` over 1268 dims passed 22 GiB in about 20 s (watchdog kill). This is the same metric-setup wall as contour NUTS.
- Unit starting mass plus diagonal adaptation (full-suite settings: step 0.03, depth 8): by iteration 25, 9 divergences, step size 10⁻⁶–10⁻⁵, trajectories at max depth 8 (255 gradients, about 40 s each). A step near 10⁻⁵ against a prior scale of 1 means roughly 10⁵ gradients per independent draw. Stopped.
- A/B on the near-hard constraint σ (shipped 1e-4; in-memory patch, `config.py` untouched), warmup step size over the first 8–22 iterations:
  - ×1: 2×10⁻⁶ to 3×10⁻⁵
  - ×100: 3×10⁻⁵ to 7×10⁻⁴
  - ×10⁴: 4×10⁻⁵ to 1.6×10⁻⁴, still hitting depth 8
- So the constraints set the worst stiffness; a second source, not yet attributed (DY or lattice), holds the step near 10⁻⁴ once they are loose. Plain NUTS needs a dense, posterior-informed metric (e.g. whitening by the Gauss–Newton curvature at the MAP), not diagonal adaptation: the stiff directions are integrals over the grid, not single coefficients.

### F5. What this points to (not yet tested)
- The binding problem for every sampler is F3: on the affine contour, the DY H-S integral's mass lies 10–20 Laplace widths out, anisotropically. Candidates that address it:
  - a contour that follows the thimble in the soft directions, i.e. flow time set in whitened frame units rather than divided by the stiffest eigenvalue;
  - a sampler that targets |integrand| directly on that contour (phase ESS in the body is estimated at 0.25–0.35), with a metric scaled to the body rather than to the saddle curvature.
- Contour NUTS needs a Hessian-free (or HVP-based) metric and a large per-gradient speed-up before it is usable at n_q = 1314.

## Status
| # | Task | Status | Closed against |
|---|---|---|---|
| 0 | Store handoff in pixel_closure | LANDED 2026-09-14 | `HANDOFF_precision_2026-09-14.md` |
| 1 | Gaussian + NUTS on failing cases, Q=2 | OPEN — Gaussian done (F1); NUTS blocked by memory and cost (F4) | `campaign/sampler_test_2026-09-14/rows`, `campaign/contour_profiles_2026-09-14/*.png` |
| 1s | Shifted-center + diagonal Newton as the default H-S saddle search | REFUSED 2026-09-14 | F2: μ start diverges on NNPDF `exp`/`both` c and breaks JAM `exp` c; saddle is not the binding failure (F1, F3) |
| 1b | Same, Q=mc and Q=3 for the passing sampler(s) | OPEN | same |
| 2 | "Stiff regime" note in VEGAS docs | OPEN | guide section + two docstrings; guide/docstring tests green |
| 3 | Truth-derived diagonal DY/synthetic-Z errors; regenerate; rerun | OPEN | `dy_replica_sigma` + new closure test; hash check |

## Verification
- **Step 1:** dry-run plans printed; harness tests pass with mutation checks; every case has a row per sampler with gate diagnostics; figures sent.
- **Step 2:** PIXEL guide/docstring/README tests pass; the kernel fingerprint is unchanged (re-run the `_kernel_source_paths` check).
- **Step 3:** new closure test passes and fails under mutation; regeneration hash check shows only DY, synthetic Z and `manifest.json` changed; closure test suite green.
