# Full-scale closure test: lattice + DIS (+ Drell-Yan) — JAM truth

The **full-scale** counterpart of `closure_JAM_truth_small/`: a realistic closure
dataset rather than a handful of mocked points.  A closure test draws a known
**truth** set of PDFs (the real `JAM24_PDF_proton_nlo` replica ensemble), folds
it through the exact forward operators the fit will use, adds realistic noise,
and checks that the fit recovers the truth within its posterior band.

Test modes (what each information source constrains):

1. `lattice` — pseudo-ITD + Mellin moments over **all thirteen** NME ensembles
   (arXiv:2601.10857), with injected-and-recovered lattice systematics;
2. `dis` — DIS structure functions at the **real analysis kinematic coverage**;
3. `dy` — fixed-target Drell-Yan (E866 pp/pd) only;
4. `exp` — **all experiment** (DIS + DY);
5. `both` — everything (lattice + DIS + DY).

**Golden rule:** generation and fitting use the *identical* PIXEL forward
operators (`Q0 = mc`, NLO, VFNS, `mc = 1.28`).  That is what makes closure
meaningful, independent of scheme realism.

See the repo-root `closure_extension_plan.md` for the full-scale design and
`closure_plan.md` for the original narrative.

## What differs from the small suite

| aspect | small suite | this (full) suite |
|---|---|---|
| lattice ensembles | 1 mock (`L=48`, `a=0.06`) | 13 real NME ensembles |
| pseudo-ITD kinematics | one `z`, `p=1..7` | `z=1..ceil(0.3fm/a)`, `p=1..floor(L/6)` per ensemble |
| lattice covariance | diagonal only | **full** replica covariance + 1% diagonal inflation |
| lattice systematics | none | injected **and** recovered (`ht, chiral, cont_invz, cont_aL`) |
| DIS coverage | 9 tables, 16 pts each | analysis-used tables, **all** retained rows (~2.5k pts) |
| DIS error | replica spread | **real** per-point `rel_stat` |
| Drell-Yan | E866 pp/pd, 6 rows/table | all accepted E866 pp/pd rows (179 + 183 = 362) |

## Low-x completion

The fitted fields are momentum densities `x*f(x)` on a positive grid beginning
at `x_min = 1e-6`.  Every lattice and DIS forward operator completes the interval
below that node.  The full closure uses a flat continuation for `sigma` and `g`,
whose `x*f` limits may be nonzero, and a power-one continuation for all valence
and flavor-nonsinglet fields, so those combinations approach zero linearly.
There is no omitted-interval mode.  The policy is single-sourced by
`config.vanishes_at_origin` -- read by both `config.low_x_completion` and the
constraint builder, so the two cannot disagree about the physics -- and shared by
generation, fitting, and constraints.

**This completion is where `x*f(x) -> 0` is enforced.**  `x = 0` is deliberately
not a grid node, and `kernels.Delta` there returns an all-**zero** row rather than
raising, so a fixed-point constraint at the origin would look imposed and do
nothing.  `constraint_datasets` therefore emits `cons_origin_*` rows only when the
grid actually reaches zero (`Grid(..., include_origin=True)`) and only for the
vanishing fields: none on this grid, 7 on an origin grid.  See
[`running_closure_tests.md`](../guides/running_closure_tests.md) for the
measurements behind leaving `x_min = 1e-6` alone.

Each field's prior is `a ± a`: one amplitude sets both the constant log-RBF mean
and its sigma, pinned by `analysis.tie` rather than by two constants agreeing.
Both ends are frozen.  Set `config.GP_AMPLITUDE_FREE = True` to sample the
amplitudes instead; that attaches a hard floor (`GP_AMPLITUDE_FLOOR = 0.99`, on
`Parameter.bounds` so it binds MAP as well as the samplers) and a log-normal
hyper-prior centred on each tier.  While the flag is off nothing is attached, so
the evidence is unshifted.

The full closure compiles Drell--Yan with the `block-sparse` bilinear layout.
Only the 9 physical PDF fields enter the H--S precision; lattice nuisance fields
are restored afterward from their exact Gaussian conditional.  Each physical
hard tensor is retained once with a small flavour-mixing matrix instead of being
expanded into a dense `(362, 5760, 5760)` tensor.

## Injected lattice systematics

Each active systematic key gets a shared x-space nuisance curve per physical
field, folded through the **bare** cosine/sine transform and scaled by the
registry coefficient `coeff_k(meta)`; every ensemble sees the same latent curve
through its own coefficient, so the multi-ensemble `a`/`L`/`Mπ` spread lets the
fit separate PDF from artifact.

The nuisance transform stays bare *even though the physical field is now evolved
and matched*, and that asymmetry is deliberate. These keys — higher twist,
chiral extrapolation, `a/z` and `a/L` discretization — are artifacts of the
lattice calculation: they contaminate the measured reduced ITD directly. They
are not PDF-like objects sitting at `Q0` with DGLAP scale dependence of their
own, and evolving them alongside the field would invent one.
`tests/test_data_builders.py` and `tests/test_closure_pseudoitd_matching.py`
both pin it.  The injected curves are property-enforced in
generation:

1. **parity in ν** — real (cosine) even, imag (sine) odd (automatic);
2. **converge as ν→∞** — automatic (Riemann–Lebesgue);
3. **zero at ν=0 (real) + no shift to the known ∫dx counting moments** — a single
   condition: the systematic curve has a **nulled n=1 Mellin moment**;
4. **≤ 10% of the leading PDF signal** — rescaled to this cap.

`tests/test_closure_extension_systematics.py` checks all four.

## Pseudo-ITD evolution and matching

`PITD_EVOLVE` is on. The fitted field is evolved from `Q0 = mc` to the
per-separation matching scale and matched at `PITD_MATCHING_ORDER`, rather than
reading the LO reduced ITD straight off the field.  The temporal-vector input
is provider-prescaled `a1 * R`: generation supplies the transformed mean and
covariance, and the loader leaves both unchanged.  Real/cosine rows use C-odd
`q-qbar`; imaginary/sine rows use C-even `q+qbar`:

```
mu^2(z) = (lam * hbarc / (a z))^2 ,   lam = PITD_MATCHING_LAMBDA = 1
```

`PITD_MATCHING_ORDER = "NLO"` is the `O(alpha_s)` matching,
`C_F [ ln(lam^2 e^(2 gamma_E + 1)/4) B(N) + L(N) ]` — the Altarelli-Parisi log
**and** the finite matching constant `L`. With `lam = 1` the log's argument is
`e^(2 gamma_E + 1)/4 = 2.40`, so neither term is negligible.
`tests/test_closure_pseudoitd_matching.py` bridges this historical one-loop
form to Li--Ma--Qiu and checks the complete NNLO temporal-vector translation
against the authors' ancillary output, independently of the production contour.

**The accuracy ceiling here is `alpha_s`, not the contour.** `mu^2(z)` *falls*
as the separation grows. Measured across all thirteen NME ensembles with
`z <= 0.3 fm`:

| | |
|---|---|
| `mu^2` range | `[0.32, 13.8] GeV^2` |
| `alpha_s` at the softest `mu^2` | **0.88** |
| Landau pole `Lambda_QCD(nf=3)^2` | `0.154 GeV^2` |

So the largest separations sit above the pole — the operator builds and
`EvolvedPseudoITD.matching_scales` does not refuse — but at `alpha_s ~ 0.9` the
`O(alpha_s)` term is not a small correction to the leading one, and the closure
should not be read as testing a converged perturbative matching there. Raising
`lam` to ~1.95 puts `mu^2 >= Q0^2` everywhere (`alpha_s <= 0.42`) at the cost of
a `ln ~ 2.2` matching log; tightening the `z` cut trades statistics for the same
thing. Both are physics choices, so `PITD_MATCHING_LAMBDA` is a named constant
rather than a buried default.

Generation needs no separate change: it already folds each dataset's own
contributions (`sum_c kernel.matrix(nu, basis) @ truth`), so it picks up the
evolved, matched operator automatically — the property `generate.dy_central` had
to be rewritten to acquire.

## Injected DIS experimental nuisances

The fitpack DIS tables carry the real nuisances alongside the statistical error,
and the closure suite now uses both:

* an overall multiplicative normalization (`*norm_c`) — SLAC 2.1%/1.7%, BCDMS 3%,
  NMC 2%, HERMES 7.6%;
* correlated systematic sources (`%*_c`) — **169 per HERA table**, 5 for BCDMS,
  11 for NMC; in quadrature these are 0.3–0.9× the statistical error, so they are
  a significant part of the error budget rather than a correction.

`dis_audit` records both *relative to the real value* (the per-source vectors go
to `.npz` sidecars under `data/dis_systematics/`, far too bulky for JSON).
Generation rescales them onto the folded fake central, applies a configured
offset to each normalization (`config.DIS_NORM_BETA`) and a **drawn**
`beta_k ~ N(0,1)` to every correlated source, then saves the absolute vectors to
`truthQ_*/sys/`.  The fit folds the identical directions into each table's
covariance, and `fit.exp_nuisance_report` inverts them via `Model.nuisance_pulls`.

Drawing the systematics rather than only inflating the covariance is what keeps
the closure honest: the fit marginalizes exactly those directions, so the data
must actually scatter along them — otherwise the enlarged covariance would be
unjustified and coverage would come out artificially conservative.

Reported per table: the normalization pull against its injected truth, and for
the correlated block a pull chi2 per source plus a coverage fraction (169 × 7
individual rows would be unreadable).

**Interpreting the reported z.** The uncertainty `nuisance_pulls` quotes is
`1 - vᵀC⁺v`, the width of `beta` with the *theory held fixed*. It does not
include the PDF's freedom to absorb an overall normalization, and that
degeneracy is real — at this scale the fitted PDF soaks up much of the injected
shift, so the residual only partly lies along `v` (measured projections
0.08–0.77). `z_cond` is therefore a diagnostic, not a significance: a large
value means the offset went into the PDF rather than that the machinery failed.
Exact recovery is asserted in `tests/test_normalization_nuisances.py`, which
pins the fields to break the degeneracy. The correlated block behaves as
expected in the opposite regime — with 169 sources against far fewer rows it is
prior-dominated (`median_uncertainty` ≈ 1), so its `pull chi2/src ≈ 1` and ~95%
coverage confirm consistency rather than demonstrate recovery.

## Drell-Yan evolution

`DY_EVOLVE` is on.  Every DY hard tensor now has the flavour-resolved evolution
operator from `Q0 = mc` composed into it, on the fit side **and** in generation.

This was previously off, and the constant was *dead*: `datasets.build_drell_yan`
hardcoded `evo = None`, so setting it changed nothing.  The approximation it
documented was self-consistent -- generation and fit shared one unevolved
operator -- but E866 sits one to five e-folds above the input scale, so the
absolute cross section was not physical.

Three things make the wired version correct rather than merely on:

* **Per-row scales.** An E866 table spans `Q^2 = 172-646`, and one matrix at an
  effective mass is a different operator, not an approximation of the right one.
  Each row carries its own operator.  Cost scales with *distinct scales*, not
  rows: 46 and 49 for the two tables, ~31 s each on the 128-node basis, so about
  50 minutes once.  The operators depend on kinematics and `Q0`, never on the
  truth, so all six `truthQ_*` members share one cache.  Pin
  `OMP_NUM_THREADS=1` while assembling it.
* **Five flavours.** Evolving from `mc` crosses bottom, so the projection is
  five-flavour and `b`/`bbar` carry a real luminosity through a generated
  `T24`-like combination the static `Nf=4` coefficient table cannot express.
  `DY_NF` follows `DY_EVOLVE`, and `datasets` refuses to build if the projection
  and the configured luminosity disagree about it.
* **Shared operator objects.** `model` groups block-sparse factors on
  `id(evolution_A)`.  Eight shared operators serve all 296 contributions of a
  table; one array per field pair would be numerically identical and ruinously
  more expensive.  `tests/test_closure_dy_evolution.py` asserts the collapse.

**`generate.dy_central` had to change with it.**  It used to pre-combine the
truth into *parton* curves and fold `bilinear_tensor`, the bare hard tensor.
Composing evolution on the fit side alone would have left generation folding an
unevolved operator: both sides internally consistent, the closure invariant
broken, and nothing failing.  It now folds the assembled operator off each
contribution, so the two sides are structurally the same object.  Unevolved
results are unchanged to rounding (the sum order differs).

**Still unevolved: the lattice pseudo-ITD.**  `build_pseudoitd` builds the LO
reduced ITD with no `mu0_2` and no matching.  `pixel.data.PseudoITD` supports
both, so this is a wiring gap rather than a missing operator, but it is a real
one -- the Mellin-moment records already evolve to `MOMENT_Q2`.  Every DIS
observable (F2, NC and CC reduced cross sections) has always evolved from `Q0`.

## Injected Drell-Yan normalization

Each E866 table carries one overall luminosity uncertainty — `%norm_c` = 7%,
recorded per table by `dy_audit` as `rel_norm`.  Generation scales every row of a
table by `1 + beta_true * rel_norm` (`config.DY_NORM_BETA`: `+1.0` for pp,
`-0.6` for pd, so a recovered pull cannot be a global fluke) and the fit
marginalizes the same `rel_norm` into that table's covariance.  The equivalent
best-fit offset comes back out of `Model.nuisance_pulls` and is reported as
`dy_normalization` in each summary (and printed per run).

Because the marginalized covariance references the *data* rather than the unknown
truth, the recoverable pull is `beta_true / (1 + beta_true * rel_norm)`; the
report quotes that expectation alongside the recovered pull.  Setting
`config.DY_FIT_NORMALIZATION = True` switches to a fitted amplitude instead —
correct machinery, but the amplitude only sees the data through the H-S `f`
integral, so the marginalized route is the default.

`tests/test_dy_normalization.py` covers both treatments, including recovery of a
known injected offset.

## Reported effective sample sizes

Each fit reports **two** different effective sample sizes, because a chain can
fail in two unrelated ways and neither number catches the other:

| Report field | What it measures |
|---|---|
| `effective_sample_size` | **Mixing.** Wolff's Gamma method (`UWerr`, [hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017)) over the real Markov history: how many effectively independent draws the chain produced, taken over its slowest direction. Accompanied by `tau_int_max`, `slowest_param`, and `tau_reliable`. |
| `phase_effective_sample_size` | **Phase/weight degeneracy.** The Kish ratio over residual phase weights — the sign-problem diagnostic for a contour fit. |

A contour fit can have a healthy phase ESS and still be worthless because the
chain never moved, and a well-mixed chain can still be dominated by a handful of
phase weights.  Read both.

> **The meaning of `effective_sample_size` changed.**  It used to be the
> weight/phase number, which for an ordinary (non-contour) HMC fit was
> `1/sum(w^2)` over uniform weights — i.e. exactly the sample count, every time.
> It could never signal a problem.  It is now the autocorrelation ESS; the old
> quantity is still reported under `phase_effective_sample_size`.  Numbers do not
> compare against reports generated before this change.

Conventions worth stating, because they are easy to misread: `tau_int` follows
Wolff, so **`1/2` means uncorrelated draws** and the ESS is `N / (2 tau_int)`.
A well-tuned chain on a smooth target can be *anti*-correlated and legitimately
report an ESS above the sample count.  `tau_reliable` is false when the chain is
shorter than 50 autocorrelation times, in which case `tau_int` is itself noise.

Fixed-parameter fits (`PosteriorResult`) and importance-weighted draws report
`null` here: neither is a Markov chain, so there is no autocorrelation to
measure.  See [`guides/autocorrelation.md`](../guides/autocorrelation.md).

## Usage

```bash
# generate + fit, full scale (default)
./run_closure.sh --Q 2 --modes both
# data only (audits + folds, no fit)
./run_closure.sh --Q 2 --data-only --remake-data
# kernels only (uses existing/generated data, no fit or plots)
./run_closure.sh --Q 2 --modes both --kernels-only --remake-kernels
# a single ensemble for a tractable run
PIXEL_CLOSURE_ENSEMBLES=a067m135 ./run_closure.sh --Q 2 --modes both
# the small mock suite, same wrapper
./run_closure.sh --scale small --Q 2
```

The first fit after the kernel cache was written by *older* PIXEL code pays a
one-time verification pass: the sidecars carry a `kernel_code` fingerprint, and a
mismatch re-derives sampled rows before re-stamping them.  With ~89 top-level
matching matrices here, budget roughly ten minutes for that once — every run
afterwards accepts the whole cache in about a second.  See the caching notes in
the [repo README](../README.md#how-a-kernel-cache-is-trusted).

`config.ACTIVE_ENSEMBLE_IDS` (or `PIXEL_CLOSURE_ENSEMBLES=id1,id2,...`) restricts
the ensemble set for quick runs.  The full 13-ensemble × 6-`Q` fit is large — see
the performance/execution-gate notes in `closure_extension_plan.md`; run the
resource estimate before attempting a full fit.

## Modules

| module | role |
|---|---|
| `config.py` | 13-ensemble table, kinematics rules, systematics, DIS/DY presets, priors |
| `dis_audit.py` | scan every fitpack IDIS workbook; full-coverage analysis tables |
| `dy_audit.py` | scan fitpack DY workbooks; E866 pp/pd kinematics → S/Q²/Y |
| `pdf_guidance.py` | evaluate JAM (via `lhapdf_dump.cc`), project to 9 fields |
| `truth.py` | assemble a truth member |
| `datasets.py` | per-ensemble lattice + DIS + Drell-Yan builders, systematics wiring |
| `generate.py` | fold every replica; full cov + 1% inflation; inject systematics |
| `fit.py` | GP-prior analysis + shared nuisance fields; ordinary HMC or joint contour HMC over flowed Hubbard-Stratonovich coordinates; coverage + nuisance recovery |
| `run_closure.py` | run modes per `Q`; reproduction/comparison plots |
