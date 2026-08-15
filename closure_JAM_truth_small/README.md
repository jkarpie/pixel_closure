# Full-flavor closure test: lattice + DIS + Drell-Yan (JAM truth)

A closure test draws a known **truth** set of PDFs, folds it through the exact
forward operators the fit will use, adds realistic noise, and checks that the
fit recovers the truth within its posterior band.  This suite runs several tests
over one shared truth so we can see what each information source constrains:

1. `lattice` — pseudo-ITD + Mellin moments only;
2. `dis` — DIS structure functions only;
3. `dy` — fixed-target Drell-Yan (E866 pp/pd) only;
4. `exp` — **all experiment** (DIS + DY);
5. `both` — everything (lattice + DIS + DY);
6. `synthetic_z_plumbing` — one unevolved, boson-level gamma/Z point used only
   to smoke-test EW dataset plumbing.

The synthetic-Z mode is classified `synthetic_proxy`.  It is not an LHC
dataset, has no fiducial acceptance or DY evolution, and must not be cited as
physics Z coverage.

This is the **small/mock** suite (a handful of invented points, for fast
smoke-testing).  The realistic full-scale counterpart is
[`closure_JAM_truth/`](../closure_JAM_truth) — 13 real lattice ensembles, full DIS
coverage, and injected lattice systematics.  Both scales run through the same
wrapper: `./run_closure.sh --scale small|full`.

**Golden rule:** generation and fitting use the *identical* PIXEL forward
operators (input scale `Q0 = mc`, NLO, VFNS, `mc = 1.28`).  That is what makes
closure meaningful, independent of scheme realism.

See the repo-root `closure_plan.md` for the full design narrative.

## Truth = real JAM, read across scales

The truth is the real `JAM24_PDF_proton_nlo` global-analysis result
(NLO, VFNS, 196 replicas), used **directly**: every replica (members `1..195`) is
projected into the nine C-even/C-odd basis fields and folded through the
operators as-is.  The **truth field is the ensemble mean** over replicas and the
**fake-data covariance is the replica covariance** of the folded output — the
real JAM uncertainty propagated through the forward model, not an invented error
budget.

> **The ensemble is a covariance source, never a central value for a bilinear
> observable.**  For every *linear* dataset the two readings coincide exactly —
> folding is linear, so `mean(A q_m) == A mean(q_m)` — but Drell-Yan is bilinear
> and has no such identity.  Until 2026-08-14 `generate.dy_central` folded the
> ensemble *second* moment `E[q_A q_B] = E[q_A]E[q_B] + Cov(q_A, q_B)`, which no
> single PDF reproduces, so DY coverage was measured against a central value with
> no truth curve behind it (`S0-05`).  It now folds the one truth curve written to
> `truth.json`, exactly as the two full suites do, and refuses a replica stack
> rather than flattening it.  Measured size of the change at `Q = 2`:
> `2.4e-03`/`2.9e-03` (JAM) and `2.1e-03`/`6.4e-03` (NNPDF) `max|ratio-1|` over
> rows.  **The committed `data/truthQ_*/` DY rows predate the fix and still carry
> the second-moment central value; regenerating them is a separate decision and
> has not been done.**  An analytic two-term form `N x^a(1-x)^b(1+c√x+dx)+N'x^a'(1-x)^b'` is also
fit per field (to the mean curve) and saved as **reference metadata only** — it
is not the truth.  Each `truth.json` also stores `curve_std`, the per-node replica
spread, which `run_closure` draws as the truth error band.

We build several truth members by reading JAM at different **original scales**
`Q ∈ {mc, 1, 2, 3, 4, 5} GeV`, then treating each curve as the truth at the
common input scale `Q0 = mc` (where the distributions are parameterized and the
fake data are generated).  Reading JAM at different `Q` yields genuinely
different but realistic PDF-shaped truths.  Truths and results are saved keyed by
`Q`, and `run_closure` aggregates every `Q` that has been run into a cross-`Q`
comparison — the proof that the code behaves consistently across inputs.
`Q = 1 GeV` is evaluated directly through LHAPDF; the truth record notes that
this is an extrapolation below JAM's advertised `QMin = 1.14`.

## Flavor basis (nf = 4)

Nine fields on a shared 128-point log-linear grid:

| sector | C-even (q+q̄) | C-odd valence (q−q̄) |
|---|---|---|
| isovector | `t3` = T3 | `v3` = V3 |
| strange (SU(4)) | `t8` = T8 | `v8` = V8 |
| isosinglet | `sigma` = Σ | `v` = total valence |
| charm (SU(4)) | `t15` = T15 | `v15` = V15 |
| gluon | `g` | — (gluon is C-even only) |

The requested lattice "strange"/"charm" sectors are the SU(4) basis
combinations `T8/V8` and `T15/V15`, **not** isolated `s`/`c` PDFs.  DIS
charge-weighted combinations and lattice flavor combinations both project from
these same nine fields, so the combined fit is self-consistent.

## Data

All fake-data errors come from the **JAM replica spread of the folded observable**
(see above).  For **both** lattice and experimental data we keep only the
**diagonal** of the replica covariance (`sigma_i = sqrt(C_ii)`) and jiggle the
central values by `Normal(central, sigma)` — the fake data are uncorrelated.

- **Lattice pseudo-ITD:** `L = 48`, `a = 0.06 fm`, one separation `z = 0.3 fm`,
  momenta `pz ≲ 3 GeV` (integer `p = 1..7`).  For the vector convention,
  real/cosine → C-odd `q-qbar` and imaginary/sine → C-even `q+qbar`.
  This small suite keeps the bare Fourier transform; the full suite exercises
  row-wise evolution and matching.  Vector input arrays are interpreted as
  provider-prescaled `a1 * R` and are never rescaled by the loader.
- **Mellin moments at 2 GeV:** `⟨x⟩` (`n=2`) routes to the C-odd field and `⟨x²⟩`
  (`n=3`) to the C-even field under PIXEL's CP convention; NLO evolution
  `mc → 2 GeV`.
- **Drell-Yan:** fixed-target E866 `pp`/`pd` from
  `fitpack_legacy/database/dy/expdata/` (audited by `dy_audit.py`; kinematics
  `RS, Rtau, xF -> S, Q², Y`), sized by the real `rel_stat`.  DY is **bilinear**
  in the PDFs, so those fits use Hessian-preconditioned joint affine VEGAS for
  the Hubbard-Stratonovich auxiliary integral rather than the closed-form linear
  posterior.  Dataset normalizations are fitted real parameters; the H-S
  coordinates lie on a fixed-reference complex affine saddle contour.  The
  joint proposal uses this suite's `5e-3` saddle tolerance, unit covariance
  inflation, up to five extra pilot passes when phase ESS is below `0.25`, and
  freezes the highest-ESS pilot grid for production. Documented
  closure approximations: **LO** DY and no PDF evolution (identical on the
  generation and fit sides, so the closure stays self-consistent).
- **Synthetic Z plumbing:** exactly one `pp`, 7 TeV, `Y=0` gamma/Z point at
  `Q²=M_Z²`, isolated in `synthetic_z_plumbing`.  It deliberately retains the
  small-suite no-evolution approximation and tests only manifest, EW-weight,
  cache, bilinear-assembly, and inference wiring.
- **DIS:** real `(x, Q²)` from the fitpack tables in
  `fitpack_legacy/database/idis/expdata/` (audited by `dis_audit.py`).  Modelled
  directly: F2 (p/d), NC `σ_r` (HERA), and the inclusive HERA CC
  `σ_{r,CC}` (`10031/10032`) assembled from `W2`, `WL`, and `xW3`.  The real
  `rel_stat` is still audited but no longer sizes the error.  No systematic
  nuisances.

## Fit, priors, and constraints

- **Grid:** 128-point log-linear grid, cubic-spline finite elements.
- **Prior:** each field is a log-RBF GP (`length = ln 2`, fixed `x_reg`) with a
  **constant, x-independent mean** (`priors.Const`):
  - the C-even singlet (`sigma`), gluon (`g`) and charm (`t15`) use **mean 5,
    σ 5**;
  - the C-even non-singlets (`t3`, `t8`) and every C-odd valence field
    (`v3`, `v8`, `v`, `v15`) use **mean 0.5, σ 1**.

  The mean `N` is **tied** to σ (`analysis.tie`), so each prior is `a ± a`; the
  tied amplitude, length and `x_reg` are all **frozen** (set
  `config.GP_AMPLITUDE_FREE = True` to sample the amplitudes instead, which
  attaches a hard floor at `GP_AMPLITUDE_FLOOR` and a log-normal hyper-prior
  centred on each tier), and the 1,152 GP field
  coefficients are marginalized analytically.  Lattice-only fits therefore
  have zero sampled parameters; DIS/DY fits sample their fitted table
  normalizations and, for DY, the 12 mathematical H-S coordinates.  When
  inference is requested on a model with no free parameters, PIXEL returns the
  closed-form posterior as a `pixel.infer.PosteriorResult`.  (Grouping lives in
  `config.HIGH_PRIOR_FIELDS` / `gp_amplitude`.)
- **Physics constraints** (near-hard pseudo-data, every mode):
  - every distribution vanishes at `x = 1` (a `Delta` evaluation `= 0`);
  - `x*f(x) -> 0` at the origin, for every field but `sigma`/`g`
    (`config.vanishes_at_origin`), is imposed by the **low-x completion**, not by
    a constraint row: `x = 0` is not a grid node and `kernels.Delta` there is an
    all-zero row.  The `cons_origin_*` rows activate only on a grid built with
    `include_origin=True`;
  - valence quark-counting normalizations from `∫(u−ū)=2, ∫(d−d̄)=1,
    ∫(q−q̄)=0` for `s,c` → `∫V3 = 1`, `∫V8 = ∫V = ∫V15 = 3` (a Mellin `n=1`
    moment).  The JAM truth already satisfies these, so they are
    closure-consistent.

## Modules

| module | role |
|---|---|
| `config.py` | all shared constants: 9-field basis, lattice/DIS kinematics, prior amplitudes (tied mean = sigma), presets |
| `dis_audit.py` | scan fitpack DIS tables → `data/dis_manifest.json` |
| `pdf_guidance.py` | evaluate JAM (via `lhapdf_dump.cc`), project to 9 fields, reference fit |
| `truth.py` | assemble a truth member (JAM curves + reference fit) |
| `datasets.py` | build PIXEL datasets from a member's manifest (shared by gen + fit) |
| `generate.py` | fold every JAM replica through the operators, take ensemble mean/covariance, write data; DY folds the single ensemble-mean truth curve, not a replica statistic |
| `fit.py` | build the GP-prior analysis, MAP fit, coverage/pulls |
| `run_closure.py` | run all configured modes per `Q`; writes PDF-space, data-space, cross-`Q`, and ratio-grid diagnostics as PNG/PDF pairs |
| `plot_datasets.py` | observable-space plot: fake data (error bars) + truth mean + posterior fit band, per dataset. Draws *linear* rows only, so a `dy`-only fit produces no data-space figure (reported as `[data-space plot skipped: no linear datasets in this mode]`) |
| `plot_ratios.py` | cross-`Q` ratio grids: `fit / true PDF mean` per flavor, all input `Q` overlaid, one grid per mode |

## Reported effective sample sizes

The small suite uses independent VEGAS importance draws, not a Markov chain, so
`effective_sample_size` and the `autocorr` fields are `null`.  Weight quality is
reported under `sampler_diagnostics`:

| Field | Meaning |
|---|---|
| `absolute_ess_fraction` | Kish ESS from absolute real importance weights, divided by the production draw count. |
| `signed_ess_fraction` | ESS after real-sign cancellation; this can be much smaller than absolute ESS. |
| `average_sign` | Real signed weight sum divided by absolute weight sum. |
| `median_inner_ess_frac` | For the joint affine grid, the complex phase ESS fraction; for the nested fallback, the median inner-contour ESS. |
| `n_nonconverged_saddles` | Number of failed affine saddle constructions (required to be zero). |

The compatibility field `phase_effective_sample_size` contains the absolute
importance ESS.  The driver refuses fits below its configured absolute, signed,
or phase/inner ESS gates instead of silently writing an unreliable summary.

## Usage

The repo-level wrapper is the usual entry point.  It defaults to the JAM truth
suite, so `--truth JAM` is optional:

```bash
./run_closure.sh --Q 2 --modes both
./run_closure.sh --Q 2 --remake-data --remake-kernels
./run_closure.sh --all --modes lattice,dis,dy,exp,both
```

To generate or regenerate fake data without running any fits, add `--data-only`.
This runs `dis_audit` as needed, runs `generate`, and then exits before
`run_closure.py`.  In data-only mode the wrapper does not clear existing
`results/` outputs:

```bash
./run_closure.sh --Q 2 --data-only --remake-data
./run_closure.sh --all --data-only --remake-data
```

To populate the selected kernel caches without fitting or plotting, use
`--kernels-only`. Existing data are reused; missing data are generated first,
and existing results are preserved:

```bash
./run_closure.sh --Q 2 --modes both --kernels-only
./run_closure.sh --all --kernels-only --remake-kernels
```

`--remake-data` removes the selected `data/truthQ_*` inputs before generation;
without it, existing `truth.json` files are reused.  `--remake-kernels` clears
the shared `data/_kernel_cache/`.  `--trust-kernel-cache` and
`--verify-kernel-cache` set the kernel-cache checking level during fitting
only (see below).  Full fit runs remove the
selected `results/` outputs first unless `--keep-results` is supplied.

The underlying modules are still useful when debugging a specific stage:

```bash
venv/bin/python -m closure_JAM_truth_small.dis_audit                 # 1. audit real DIS tables
venv/bin/python -m closure_JAM_truth_small.pdf_guidance              # 2. cache JAM truth per Q
venv/bin/python -m closure_JAM_truth_small.generate --all            # 3. write data for every Q
#   or a single member:  python -m closure_JAM_truth_small.generate --Q 2
venv/bin/python -m closure_JAM_truth_small.run_closure               # 4. fit all modes + compare
#   or one:              python -m closure_JAM_truth_small.run_closure --Q 2 --modes both
```

Outputs land under `closure_JAM_truth_small/data/truthQ_<label>/` (truth + fake data) and
`closure_JAM_truth_small/results/{truthQ_<label>,comparison}/` (summaries + PNG/PDF figures).
Generated `data/`, `results/`, and `reference_pdfs/` contents are ignored by Git;
only `.gitkeep` placeholders keep the cache directories present in fresh clones.

The transformation kernels go `Q0² = mc² → data scale` and are **identical for
every truth member** (all treated at `Q0 = mc`; the original `Q` only sets the
LHAPDF input curve, applied after the kernel).  So `generate --all` assembles the
forward operators **once** and reuses them in-memory for every `Q` — only the
first member pays the kernel-assembly cost.  The kernels are also cached on disk
in the shared `closure_JAM_truth_small/data/_kernel_cache/` for reuse across processes (e.g. the
fits).

**Cache re-verification is no longer something you skip.**  A cache hit is
accepted on three checks: the metadata fingerprint, the per-array digest recorded
when the file was written, and a `kernel_code` fingerprint over `pixel/kernels`
and `pixel/geometry`.  That last one is what catches a stale cache left over from
an *older version of the kernel math* — the classic closure-cache-skew bug — so
editing any evolution/matching code no longer needs a manual
`closure_JAM_truth_small/data/_kernel_cache/` wipe: the next fit re-derives sampled rows, and
re-stamps the sidecar when they still agree.

On a warm cache the default checks are effectively free, and the build is
dominated by loading the arrays rather than validating them.  Two flags remain
for the edges:

```bash
./run_closure.sh --truth JAM --Q 2 --trust-kernel-cache   # accept the fingerprint with no content check at all
./run_closure.sh --truth JAM --Q 2 --verify-kernel-cache  # always re-derive sampled rows; the strongest and by far the slowest check
```

Run each once on your own hardware if you need the actual cost; `PIXEL_PROGRESS=1`
prints the per-kernel breakdown.

## Documented approximations

- The singlet (`sigma`) and gluon (`g`) pseudo-ITD/moment channels are evolved
  with the non-singlet operator PIXEL exposes for these builders; this is
  self-consistent for closure (generation and fit share it) but not the full
  singlet-mixing physics.
- The `t15/v15` "charm" and `v8` "strange-valence" JAM inputs are small/near-zero
  (symmetric sea), so DIS constrains them only weakly — expected, and visible in
  the coverage report.
- HERA CC uses the Fitpack-validated massless NLO VFNS operator.  It is not a
  substitute for the still-gated NNPDF theory-200 NNLO FONLL-C/TMC prediction.

## Legacy fixture

`closure_JAM_truth_small/data/standard/` plus `standard.py` / `generate_standard.py` are an
earlier LHAPDF-guided single-truth fixture (`Q0 = 2`, LO), kept for
reference.  Its historical CC F3 proxy is not used by the suite above.
