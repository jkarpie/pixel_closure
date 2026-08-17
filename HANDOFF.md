# Closure suites: make configuration *derivable* instead of guessed

**Handoff plan.** The previous agent ran out of context. Everything needed to continue is
here; nothing important lives only in the old transcript.

---

## 1. Context — why this exists

The small closure suites (`closure_{NNPDF,JAM}_truth_small`) kept producing fits that either
failed outright or showed biases, and every fix so far has been *hand-picking a number and
running a 40-minute sweep to find out*. That loop is the actual problem. Two of the three
failures below were caused by a value chosen without a way to know in advance it was
infeasible.

**Current state, measured 2026-08-16 13:27:**

```
amplitudes  t3 0.25  t8 0.5  sigma 5.0  t15 2.0  g 5.0
            v3 0.25  v8 0.2  v   0.5    v15 0.5          <- owner-specified, keep
constraints CONSTRAINT_{ENDPOINT,NORM,ORIGIN,MOMENTUM}_SIGMA = 1e-6
Q members   {2, 3}          RCOND 1e-16       MCMC_SAMPLES 1000
```

| mode | NNPDF | JAM | note |
|---|---|---|---|
| `lattice` | 2/2 | 2/2 | 0 free params, analytic |
| `dis` | 2/2 | 2/2 | |
| `dy` | 2/2 | 2/2 | passes at 1e-6, failed at 1e-8 |
| `synthetic_z` | 2/2 | — | |
| **`exp`** | **0/2** | **0/2** | `nested VEGAS ESS 0.001 < 0.1` |
| **`both`** | **0/2** | running | same |

**The open failure is `exp`/`both` with an ESS collapse, and the live suspect is the
constraint tightening, not the amplitudes.** Evidence: the identical nine amplitudes with
`g=5.0` ran `exp` successfully for **24/24** replica cases while constraints were at their
original `1e-4/1e-3`. Only the constraints changed since.

### What has already been settled (do not re-litigate)

* **The `t3` "systematic over-estimate" is not real.** 24 replicas with statistical noise,
  correlated systematics and `DIS_NORM_BETA` all redrawn: `t3` mean signed pull
  `+0.040 +- 0.090` (0.4 sigma, 12/24 positive) against a single-draw value of `+1.55..+1.78`
  that sits **3.7 sd out**.
* **Real, deterministic biases do exist**, found by separating *signed* from *RMS* pull:
  `v15 -0.545` (44 sigma), `v8 -0.415` (29 sigma), `t8 +0.138` (25 sigma), `v -0.399`
  (8.9 sigma). Their scatter across replicas is tiny (0.027-0.070) -- machinery, not noise.
* **Mechanism, seen in the reproduction plots**: `t8`'s posterior sits flat at exactly its
  prior amplitude for all `x < 1e-2` (no data there), and the valence fields dip **negative**
  at large `x` to buy back an integral pinned by `cons_norm_*`.
* **`g = 10` failed on WIDTH, not mean.** Tie broken for the gluon alone:
  `mean 10 / width 5` PASS, `mean 5 / width 10` FAIL. Ceiling between 6 and 7.
* **Constraints at 1e-8 break the H-S saddle** (`dy`, `exp`, `both`, `synthetic_z`).
  Bisected to the tightening, not to the new momentum constraint. Loosening the saddle
  tolerance 20x and dropping `RCOND` to 1e-20 both fail to rescue it -- so the earlier
  "it sits on the RCOND truncation floor" explanation is **wrong and should not be repeated**.

---

## 2. Lessons that must survive the handoff

These each cost hours. They are the reason this plan is shaped the way it is.

1. **Pick the probe that can exhibit the failure, not the fastest one.** Scanning the
   constraint floor on `dy` gave "1e-6 works" -- but `dy` is LO Drell-Yan with **no gluon
   channel**, so it cannot see any failure that enters through the DIS gluon coefficient
   function. `exp` is the discriminating mode. This same mistake was made twice.
2. **A single-seed closure test cannot separate bias from noise.** `EXP_SYSTEMATIC_SEED` was
   one fixed constant shared by every Q member *and* both truth packages, and `DIS_NORM_BETA`
   were hardcoded, so "6/6 unanimous in both truths" was **one draw counted twelve times**.
   `generate_member` now takes `sys_seed=` and `norm_betas=` (defaults unchanged).
3. **Signed pull and RMS pull answer different questions.** Same `pull_chi2_per_point` can
   mean a one-directional bias or a symmetric band that is too narrow. `coverage_report` now
   stores `mean_signed_pull` and `signed_pull_bins`; use them.
4. **`a +- a` ties the mean to the width.** For a prior-dominated field the offset *and* the
   posterior width both scale with `a`, so the pull is nearly invariant -- measured
   `1.47 -> 1.40` for `a` going `1.0 -> 0.25`. **No amplitude choice fixes a prior-dominated
   field.** It also means "make the prior wide" and "stop the posterior sitting at a wrong
   non-zero value where there is no data" are in direct conflict.
5. **Exit code 0 is not success, and a passing test is not a working test.** A `timeout`-killed
   run reported 0 because `head` swallowed exit 124; `echo "$(date) rc=$?"` always prints 0
   because the command substitution resets `$?`. Judge by the `N passed` line and by a
   completed-case count.
6. **Plot it.** See `closure_*/CLAUDE.md`. The `t3` question survived several rounds of tables
   and was settled by one figure; the *mechanism* for the real biases was only ever visible in
   the reproduction grid.
7. **Compare with `abs(a/b - 1)`, never by eye.** A 3-decimal print hid a `3.8e-7` difference
   and nearly produced a phantom finding.
8. **Do not edit config or `fit.py` while a run is in flight.** Python snapshots imports at
   process start, so an edit silently splits a sweep into two different configurations.
9. **Machine limits.** ~2 concurrent `exp`/`both` fits maximum -- each peaks at **4-6 GB**
   (not the 1-3 GB quoted earlier) and ~300% CPU even with `OMP_NUM_THREADS=1`, because XLA
   threads independently. Do not stack pytest or one-off fits on top of a running sweep.
10. **`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`** -- without it, BLAS threads > 1
    deadlock the suite in XLA's Eigen pool (all threads in `__psynch_cvwait`). It is also
    *faster* here. Run detached (`fork`+`setsid`; macOS has no `setsid` binary) because a
    harness background task is capped at a 10 minute timeout.

---

## 3. Tests to run

Order matters: T1 can make T2 unnecessary.

### T1. Constraint-strength scan **on `exp`** (do this first)
Scan `CONSTRAINT_*_SIGMA` over `{1e-4, 1e-5, 1e-6}` running `exp` at Q=2, one truth. Report
ESS fractions, not just pass/fail. Expected outcome: a mode-dependent feasibility floor, with
`exp` needing a looser constraint than `dy`. **If `1e-4` passes and `1e-6` fails, the
constraint is the cause and T2 is unnecessary.**

Reuse: `scan_sigma.py` / `probe_field.py` pattern from the session scratchpad -- set the four
`cfg.CONSTRAINT_*` attributes, call `fitmod.run_fit("2", "exp")`, catch and report.

### T2. Amplitude bisect on `exp` (only if T1 does not explain it)
One `exp` fit per row, each reverting one group to the old tier values: `t15 -> 5.0`;
valence `v3/v8/v/v15 -> 1.0`; `t3,t8 -> 1.0`; plus an all-tiers control. Whichever passes
names the culprit. If none does, say so -- it is a combination, not a single value.

### T3. Per-mode replica campaign, now with signed pull stored
Re-run the A0 machinery (`a0_replicas.py` pattern) separately for `lattice`, `dis` and `exp`,
12+ replicas each, varying all three noise sources. This localises the remaining deterministic
biases (`v15`, `v8`, `t8`) to a mode. `lattice` was clean in the single canonical draw, which
points away from the prior/constraint machinery and toward the DIS operator -- but that is one
draw and needs replica confirmation.

### T4. Kernel re-verification (still open, was Part D)
**A closure test cannot detect a wrong kernel** -- generation and fit share the operator, so a
bug cancels. Verify against oracles that are independent of PIXEL: the exact sum rules
(`N=2` momentum, `N=1` valence) which need no external code, and NNPDF FK tables
(`parse_fktable`, **not** `load_fktable`; theory-200 is legacy `.dat`; pinned commit
`90d9edc`). fitpack agreement can be **common-mode** where PIXEL was transcribed from it --
record for each comparison whether the oracle is independent. Mutation-test the existing
kernel tests before trusting them.

---

## 4. The better mechanism — stop picking values by hand

Four pieces, in increasing order of ambition. M1 and M2 are cheap and would have prevented
both failures in this session.

### M1. A feasibility pre-flight (highest value per effort)
A small script that, for a candidate configuration, runs **one `exp` fit at one Q** and reports
pass/fail plus the three ESS fractions -- *before* anyone launches a 40-minute sweep. Roughly
3 minutes against 40. Make it the documented first step of any config change.

Why `exp`: it is the cheapest mode that exercises DIS + DY + the gluon coefficient function
together, which is where every failure so far has appeared. `dy` and `lattice` are structurally
blind to them.

### M2. Config guardrails with the measurement attached
Assertions at import time in `closure_*/config.py` that reject known-infeasible regions with
the measurement cited, so a bad value fails in 0.1 s rather than 40 minutes:

```python
assert GP_AMPLITUDES["g"] <= 6.0, (
    "gluon prior WIDTH above ~6 collapses nested-VEGAS ESS to 0.001 in exp; "
    "measured 6 PASS / 7 FAIL 2026-08-15.  The mean is irrelevant -- tie broken, "
    "mean 10 with width 5 passes."
)
assert CONSTRAINT_NORM_SIGMA >= 1e-6, (
    "constraints at 1e-8 break the H-S saddle in every DY-containing mode; "
    "measured 1e-8 FAIL / 1e-7 FAIL / 1e-6 PASS on dy 2026-08-15."
)
```

Every bound must carry its measurement and date. A bound without evidence is indistinguishable
from a bug being absorbed. Update the number when a new measurement supersedes it.

### M3. Derive amplitudes from the truth instead of guessing
Replace hand-picked amplitudes with a documented procedure that reads the truth the suite
actually generates:

1. Measure each field's scale in the coverage bulk (`mean |x*f|` over `0.01 <= x <= 0.9`) --
   this is a few lines against the stored `truth.json`, no fitting.
2. Set `a` from that scale with a **stated margin**, never below the field's own magnitude
   (priors should be wide -- owner's rule -- and never `a < 0.1`).
3. Emit the table plus the measurement that produced it, so the config records *why*.

For reference, measured at Q=2: `t3` 0.163, `v3` 0.190, `t8` 0.512, `v/v8/v15` ~0.45,
`sigma/t15/g` ~0.65-0.76.

### M4. Break the `a +- a` tie — the structural fix
M1-M3 make guessing safer; only this removes the conflict. A wide prior whose mean vanishes as
`x -> 0` is impossible while one number sets both. The pieces are already built and tested:

* **`priors.PowerTaperedLogRBF`** (`src/pixel/priors/covariances.py`, exported, 13 tests in
  `tests/test_power_tapered_prior.py`, all mutation-verified):
  `k = sigma^2 |x|^alpha |x'|^alpha exp(-(log x - log x')^2 / 2 l^2)`. Stationary log-RBF
  correlation untouched; only the amplitude is non-stationary. `alpha = 0` reduces to `LogRBF`
  bit-for-bit. `|x|` not `x` -- `(x x')^alpha` goes complex for one negative coordinate.
* **`priors.Pheno`** (`src/pixel/priors/means.py:485`) for the tapered mean -- it keeps the
  `N` leaf that `analysis.tie` needs. `PhenoSeries`/`Polynomial`/`MLP` do not and would break
  the tie.
* **Apply only where `cfg.vanishes_at_origin`** -- the seven non-singlet fields. `sigma` and
  `g` keep `Const` + `LogRBF`.

**Choosing `alpha` (owner's rule):** the prior must always be *less convergent, or more
divergent, than the truth*, so it never understates the error. With `x f ~ x^p`, that is the
single inequality `alpha < p`, with margin, in both regimes. Measured minima over both truths
at Q = mc,1,2 (Regge groupings):

| group | fields | truth min `p` | proposed `alpha` |
|---|---|---|---|
| non-singlet (a2/rho, `alpha_R ~ 0.5`) | t3, v3, v8, v, v15 | **+0.179** | **+0.10** |
| singlet + charm (Pomeron) | sigma, t15 | **-0.441** | **-0.60** |
| hard/model-divergent | t8, g | **-0.847** | **-0.90** |

`alpha <= -1` makes the prior's own momentum integral divergent, so `-1` is a boundary, not a
conservative choice. Two known limits: the taper uses the *regulated* coordinate
`(|x| + x_reg)^alpha`, and `diag(K) = sigma^2 taper^2 + jitter` means **the jitter floors the
variance** -- a strongly convergent `alpha` cannot push the prior below `sqrt(jitter)`.

**Expected effect is a tighter posterior at small x. Verify, do not assume** -- a tighter
posterior with unchanged bias makes pulls *worse*, so judge on pull and coverage, never on
error size.

---

## 5. Files

| file | role |
|---|---|
| `closure_*_truth_small/config.py` | amplitudes, constraint sigmas, `TRUTH_Q_CHOICES`; add M2 guardrails |
| `closure_*/fit.py` | `constraint_datasets` (now has `cons_momentum`), `_gp_prior` (M4 wiring), `coverage_report` (signed pull, done) |
| `src/pixel/priors/covariances.py` | `PowerTaperedLogRBF`, beside `CDGP` |
| `tests/test_power_tapered_prior.py` | 13 mutation-verified tests |
| new: a feasibility pre-flight script | M1 |

The constraint builder is **byte-identical across all four suites** and
`tests/test_closure_constraints.py` enforces that -- change all four together.

**Known-failing test to fix**: `test_origin_constraint_switches_on_for_the_vanishing_fields`
pins `1.0e-4 ** 2` and now sees `1e-12`. Its intent is right (a literal, so loosening cannot
be silently mirrored) but equality is wrong -- it fires on tightening too. Convert to an upper
bound `assert dataset.cov[0,0] <= 1.0e-4 ** 2`. Two other tests in that file pin literals the
same way.

---

## 6. Verification

1. **Every candidate config passes the M1 pre-flight on `exp` before any sweep.**
2. Report `chi2/n` **and** the three ESS fractions together (gates: abs >= 0.1, signed >= 0.05,
   inner >= 0.1). A crash and a bad chi2 are different failures; the `g=10` and 1e-8 episodes
   both produced crashes with no chi2 at all.
3. Full **9-field signed + RMS** table for any change -- an earlier retune fixed `t3` and broke
   `v8` to 4 sigma. Net scorecard was 8 better / 17 worse.
4. **Plots**, per `closure_*/CLAUDE.md`. Send the files.
5. `pytest tests/ -q -k "closure or prior or constraint"`, judged by the `N passed` line.
6. Any new kernel test: **mutation-check it**. A test that cannot fail is not a test -- one
   written this session passed 13/13 against its own mutation until it was rewritten to call
   the class instead of reimplementing it.

---

## 7. Repo note

`/Users/jkarpie-admin/work/building/pixel_closure` is the new home for the closure packages.
Source is synced as of 2026-08-16 (momentum constraint, signed pull, 1e-6 constraints,
`CLAUDE.md`, `run_closure.sh`, `.gitignore`); its `TRUTH_Q_CHOICES` is deliberately the full
six members while `pixel` is at `{2,3}`. It has **no venv** -- run with
`PYTHON=/Users/jkarpie-admin/work/building/pixel/venv/bin/python`. Its `.git` is already 609 MB
because ~3,586 extracted LHAPDF `.dat` members are tracked; the new `.gitignore` stops future
ones but untracking the existing blobs is an open decision.

Also open, out of scope but worth not losing: `covariances._param` silently accepts a
**non-scalar** `sigma` and builds a non-symmetric matrix that flows into `svd_factor`; and
`nested_vegas.py:810` exponentiates `log|norm|` into float64, giving `inf` for one member
(`log|norm| = 760` against a 709.78 ceiling) under an `errstate(over="ignore")`.
