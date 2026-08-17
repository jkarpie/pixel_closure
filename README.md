# PIXEL closure-test suites

Four packages that generate a synthetic "truth", fit it with PIXEL, and ask whether
the fit recovers it. Two truth sources x two scales:

| package | truth | scale |
|---|---|---|
| `closure_NNPDF_truth_small` | `NNPDF40_nnlo_as_01180_1000` | mock / smoke |
| `closure_JAM_truth_small` | JAM | mock / smoke |
| `closure_NNPDF_truth` | `NNPDF40_nnlo_as_01180_1000` | full, 13 lattice ensembles |
| `closure_JAM_truth` | JAM | full |

Run with the PIXEL repo's venv -- this repo has none of its own, and it takes
`pixel` (including `priors.BetaTaperedLogRBF`) from that checkout's `src/`:

```bash
PYTHON=/Users/jkarpie-admin/work/building/pixel/venv/bin/python ./run_closure.sh --Q 2
```

Everything below is **measured**, on 2026-08-16, mostly on
`closure_NNPDF_truth_small`. Re-measure before quoting; several of these numbers
superseded earlier values that looked equally solid.

---

## 1. The tests (modes)

`TEST_MODES` in each `config.py`. Each folds a different subset of data against
the same fields, priors and constraints:

| mode | data | what it exercises |
|---|---|---|
| `lattice` | pseudo-ITD only | analytic posterior, 0 free params |
| `dis` | DIS structure functions | the linear DIS operator |
| `dy` | Drell-Yan only | the **bilinear** path + Hubbard-Stratonovich saddle |
| `synthetic_z_plumbing` | one synthetic Z row | H-S plumbing in isolation |
| `exp` | DIS + DY | the discriminating mode -- see below |
| `both` | lattice + DIS + DY | everything at once |

**`exp` and `both` are the modes that matter for feasibility.** `dy` is LO
Drell-Yan in the `_small` suites and has **no gluon channel**, so it is
structurally blind to anything entering through the DIS gluon coefficient
function. Scanning a config knob on `dy` and concluding it is safe has given a
wrong answer at least twice: the constraint floor was set to `1e-6` on the
strength of a `dy` scan, and every `exp`/`both` fit then failed.

Physics constraints (section 4) are added in **every** mode.

---

## 2. Grid, basis, and what that costs

```
GRID_N        128
GRID_SPACING  log-linear
X_MIN         1e-6            (x = 0 is NOT a node)
ELEMENT_TYPE  cubic_spline
```

Splines rather than C0 piecewise: PIXEL's x-space inverse-Mellin evolution needs a
far longer contour for piecewise bases, so splines are both faster *and* more
accurate for the DIS kernels.

**Two consequences worth knowing before reading any coverage number:**

* `x = 0` is not a grid node, so `kernels.Delta` there returns an all-**zero**
  row -- a silent no-constraint, not an error. The `x*f(0) = 0` limit is carried
  by the low-x completion instead, and `cons_origin_*` is only emitted when the
  grid reaches the origin (it does not here).
* **The 69 "bulk" nodes used for coverage are worth `n_eff = 2.65` independent
  points**, not 69. The GP correlation length is `ln 2` and the bulk spans 4.38
  e-folds, so `n_eff = n^2 / sum_ij rho_ij^2 = 2.65`. `pull_chi2_per_point`
  divides by 69 and looks far more precise than it is. **The sampling band on
  pull^2 is `sqrt(2/n_eff) = +-0.87`** -- a per-field spread from 0.24 to 1.79 is
  entirely consistent with perfect calibration, and a single fit cannot rank two
  priors on coverage. `points_are_correlated: True` in the coverage report is a
  hardcoded warning about exactly this, which the number then ignores.

---

## 3. Priors: two forms, one switch

`cfg.PRIOR_FORM` selects; `fit.gp_prior` dispatches.

### `"const_logrbf"` -- the incumbent, and the current production choice

`priors.Const(N=a)` mean **tied** by `analysis.tie` to `priors.LogRBF(sigma=a)`,
so the prior is `a +- a`: one number sets the mean *and* the width.

```
t3 0.25   t8 0.5   sigma 5.0   t15 2.0   g 5.0
v3 0.25   v8 0.2   v 0.5       v15 0.5
```

Known defect, not fixed: `sigma` and `g` sit **8.3x and 12.5x above** what the
momentum sum rule permits. Their constant mean asserts `int x(Sigma+g) dx = 10.0`
against `cons_momentum`'s `1.0 +- 1e-4`. Setting them to the permitted values does
not help -- the tie then drops the *width* to 0.6 where the truth singlet reaches
6.3 at `x = 1e-6`. That conflict is unresolvable within this form, and is why the
second prior exists.

### `"beta_envelope"` -- zero mean under a PDF-shaped envelope

`priors.Zero()` mean under `priors.BetaTaperedLogRBF`:

```
k(x,x') = sigma^2 |x|^a |x'|^a |1-x|^b |1-x'|^b exp(-(log x - log x')^2 / 2 l^2)
```

A zero mean asserts no integral at all, which dissolves the tie: measured, the
`cons_momentum` tension goes from **9e6 constraint-sigma to -0.29 prior-sigma**.

**How `alpha`, `beta`, `sigma` were derived** (not guessed):

1. **Minimax**, not a slope fit. Among all `(alpha, beta)` whose envelope bounds
   the truth wherever it is *resolved* (`|xf|` above 1e-3 of that field's peak),
   take the one with the smallest *slack* -- the factor by which the envelope
   over-states the error at its loosest point.
2. **`alpha >= 0` enforced.** The prior may not diverge as `x -> 0`. This caught a
   real error: the unconstrained fit gave `t8` `alpha = -0.30`, a diverging
   envelope for a field whose truth plainly converges (`xf` = 0.03 at `x = 1e-6`
   against a 1.09 peak, at every scale).
3. **Dynamic range capped at 1e6.** Not cosmetic: at sd range 3.6e+08 *all 36
   points* of a `(jitter, rcond)` stability scan on `both` failed.
4. **Tuned on Q=2 only.** Q=4/5 read an already-evolved gluon (`xg` 3.61 at Q=2 to
   30.9 at Q=5); including them inflated the gluon amplitude to 85 with 59,213x
   slack, against 8 with 193x on Q=2 alone.
5. `sigma` from requiring the truth inside one prior sd at every node, x2 margin,
   rounded **up**.

Both truth packages are used. Shipped values are in `GP_ENVELOPE`.

---

## 4. Constraints: 14 near-hard pseudo-data

Built by `fit.constraint_datasets`, **byte-identical across all four suites**:

| constraint | count | kernel | target |
|---|---|---|---|
| `cons_endpoint_*` | 9 | `kernels.Delta` at x=1 | 0 |
| `cons_norm_*` | 4 | `kernels.Mellin(alpha=-1)`, n=1 | `int V3 = 1`, `int V8 = int V = int V15 = 3` |
| `cons_momentum` | 1 | `kernels.Mellin(alpha=0)`, n=1 | `int x(Sigma+g) dx = 1` |

**Targets are the *represented* truth, not the nominal value** -- the saved curve
evaluated through the same kernel and basis the fit uses -- so the injected truth
lies in each constraint's support by construction. This matters: `int x(Sigma+g)`
is 1.0005 at Q=2 but **0.821 for NNPDF at `mc`**, an 18% violation caused by
LHAPDF extrapolating below its 1.65 QMin. A nominal 1.0 would have injected that
18% as a fresh bias.

`CONSTRAINT_*_SIGMA` is a **standard deviation**; it is squared into the dataset
covariance.

### Why 1e-4

Measured on `exp` at Q=2, one fit per row:

| sigma | abs ESS | result |
|---|---|---|
| 1e-6 | 0.001 | FAIL -- one effective sample |
| 1e-5 | 0.980 | PASS |
| 1e-4 | 0.989 | PASS |
| 1e-3 | -- | runs, misses the 4-digit bar, ~1/3 of cells fail |

A cliff, not a gradient. **The realized residual scales with the constraint**
(1.78e-05 at SD 1e-4, 2.365e-04 at SD 1e-3, i.e. ~0.2x the SD), so loosening buys
nothing and costs accuracy. It is *not* set by a kernel accuracy floor: at SD
1e-4 the fit reaches below the `mellin`/`spline_loglinear` plateau of ~2.5e-05
quoted in `benchmarks/kernel_stress.py`.

---

## 5. Numerics: jitter and the SVD cutoff

```
RCOND               1e-16   (plateau is 1e-18 .. 1e-12)
GP_JITTER           1e-10   constant prior
GP_ENVELOPE_JITTER  1e-2    envelope prior
```

**They are not two knobs for one job. They act on different matrices**, and this
took a day to establish:

* `rcond` cuts `W = C + B K B^T` **and** conditions the H-S saddle contour
  (`infer/nested_vegas.py:358,411`).
* the jitter enters `K`, and `H = K - K B^T W^-1 B K` is a near-cancellation
  inverted with a **bare `jnp.linalg.inv`** (`core/evidence.py:675`) -- no cutoff
  on that path at all.

So truncating `W` cannot make `H` definite. Measured: 9/9 envelope fits fail
across rcond 1e-10 to 1e-6 at jitter <= 1e-10, **including at rcond 1e-6 where the
cut discards more than half the modes** (rank 74/158, 94/230).

**Jitter is load-bearing, not hygiene.** The log-RBF correlation `R` is itself
indefinite on this grid (min eig -3.4e-15, still -6.7e-16 at length 0.10, so
shortening does not help). The constant prior is positive definite *only* because
the jitter lifts it -- its `min eig K` is 9.99e-11, i.e. the jitter. At
`jitter = 0` every `both` fit fails in the saddle.

The envelope needs ~1e5x more jitter. That is arithmetic: the jitter is
`R + lambda I` with `R` carrying `sigma^2`, so the *relative* floor is
`lambda/sigma^2`, and `alpha >= 0` forced flat envelopes with much larger
amplitudes.

**One real interaction:** a jitter set *above* the rcond cut stops the cut from
firing. At 1e-10 the jitter is 1.4e3-2.3e4 times above a cut at 4.40e-15, so all
128 eigenvalues clear it, the Cholesky fast path is taken, and ~35 unresolvable
directions get solved instead of dropped.

`GP_LENGTH_LOG = ln(2)`. `ln(10)` was tried and reverted: more accurate at its
working point (9.90e-06 vs 1.78e-05) but it demands ~1e3x more jitter, survives
only in the top cell or two of the scanned range instead of a 4-6 order plateau,
and halves `n_eff` to 1.38.

---

## 6. Perturbative order and integration choices

| | `_small` | full |
|---|---|---|
| `ORDER` (evolution, DIS) | NLO | NLO |
| `DY_ORDER` | **LO** | **NLO** |
| `PITD_MATCHING_ORDER` | -- | NLO |
| `DY_EVOLUTION_ORDER` | -- | `ORDER` (NLO) |

Verified at the call sites in `datasets.py`, not just in config: every `order=`
kwarg reads from `cfg`. No hardcoded LO or NNLO anywhere in the fit path.

**The NNPDF truth is an NNLO LHAPDF grid used with NLO kernels.** Fine for a
closure test -- generation and fit share the operator -- but the truth is not "an
NLO PDF", and the config header says so.

**`DY_ORDER` differs between scales.** The H-S saddle lives in the DY path, so the
numerical conclusions in section 5 are on **LO** Drell-Yan and should be
re-probed at full scale rather than transferred.

Other integration choices:

* Fields are momentum densities `q = x f`, so `int f dx` is a Mellin `n=1` moment
  with an `alpha=-1` integrand weight.
* **Low-x completion** for `[0, x_min)`: `power(alpha=1)` where
  `cfg.vanishes_at_origin`, `flat` for the singlet and gluon. The C-parity split
  is *measurably wrong* for `t8`/`t15`, recorded rather than fixed -- see the
  warning in `config.vanishes_at_origin`. Roughly a fifth to two fifths of the
  `t8`/`t15` bare pseudo-ITD signal comes from that region.
* Bilinear (DY) data are marginalized through a Hubbard-Stratonovich auxiliary,
  one per DY point, sampled by nested VEGAS.
* Truth members `mc` and `1` sit **below the sets' advertised QMin** (NNPDF 1.65,
  JAM 1.14), so LHAPDF extrapolates; `4` and `5` read an already-evolved gluon and
  call it an input. Prefer 2 and 3 for anything about a realistic input scale.

---

## 7. What was measured, 2026-08-16

144-cell campaign: both priors x 6 scales x 6 modes x 2 truths, constraints 1e-4,
rcond 1e-12, 1000 MCMC samples, each prior at its own jitter.

| prior | ran | met the 4-digit bar |
|---|---|---|
| `const_logrbf` | 71/72 | **71/71** |
| `beta_envelope` | **72/72** | **36/72** |

Mode-matched median realized residual -- the constant prior is better everywhere:

| mode | const | envelope | ratio |
|---|---|---|---|
| lattice | 5.52e-06 | 1.38e-04 | 25x |
| dis | 4.44e-07 | 1.11e-04 | 251x |
| dy | 2.70e-07 | 8.69e-05 | 322x |
| synthetic_z | 2.70e-07 | 8.68e-05 | 322x |
| exp | 4.44e-07 | 1.11e-04 | 251x |
| both | 2.65e-06 | 1.39e-04 | 53x |

**Recommendation: `const_logrbf` for production.** The envelope is more robust
(never failed to run) but less accurate.

**Every envelope failure binds on an *endpoint* constraint** -- `cons_endpoint_t15`
(16), `_v3` (12), `_sigma` (8) -- never an integral one. The envelope's width at
`x=1` is `sigma * endpoint_reg^beta`, and `alpha >= 0` forced large amplitudes, so
`endpoint_reg = 1e-3` leaves residual width the near-hard endpoint constraint
cannot suppress. **Shrinking `endpoint_reg` toward 1e-5 is the obvious next fix**
and does not disturb `alpha >= 0` or the Q=2 tuning.

Results and figures ship under `results/` (per suite) and `campaign/`.

---

## 8. Open issues

* **`endpoint_reg` for the envelope** -- above. One parameter, ~15 min to test.
* **One unexplained ESS failure**: `const_logrbf` / NNPDF / Q=1 / `both`. Same
  signature as a non-monotonic `rcond = 1e-10` failure seen at constraint 1e-4 but
  not 1e-3 -- an interaction between constraint strength and the sampler, cause
  unknown.
* **Ranking the priors on calibration needs replicas.** With `n_eff = 2.65` a
  single fit cannot do it; the physics residual is what discriminates today.
* **`kernel_stress` has never been run in this tree** -- `benchmarks/results/
  kernel_stress/` does not exist, so the `CASE_ACCURACY` bars are inherited, not
  verified at N=128.
* **`SADDLE_MIN_ESS_FRAC` is a dead knob** -- defined, no consumer.
* **`nested_vegas` never forwards `saddle_tol`**, so `CONTOUR_SADDLE_TOL = 5e-3`
  does not reach the nested-VEGAS path; it falls back to `1e-9`.

## 9. Judging a run

1. **Physics first**: the realized constraint residual, to 4 digits. Not chi2, not
   ESS. At `rcond = 1e-8` the ESS *improves* (0.955 vs 0.214) while the physics
   degrades to 1.19e-04 -- sampler health and correctness point opposite ways.
2. **`N passed`, never the exit code.** A killed pytest exits 0 with no summary.
3. **`abs(a/b - 1)`, never by eye.** A 3-decimal print has hidden a 3.8e-7
   difference here.
4. **Plot it.** See each package's `CLAUDE.md`; reuse `run_closure.py`'s
   `_prepare_matplotlib`, `save_figure_both` and `hybrid_xscale`.
5. **~2 concurrent `both`/`exp` fits**, or 3 if heavy and light modes are
   interleaved. One `both` fit peaks at ~6.6 GB and ~550% CPU. Always
   `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` -- without it BLAS threads
   deadlock the suite in XLA's Eigen pool.
