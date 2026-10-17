# Explicit-q small closure copy

This runnable copy covers both `closure_JAM_truth_small` and
`closure_NNPDF_truth_small`. It shares their generated data and kernel caches,
but writes independent outputs under `closure_small_q_integration/outputs/`.

Inference uses PIXEL's `run_q_mcmc` path for every mode. The chain evolves the
ordinary fitted parameters and prior-whitened retained GP grid fields together.
For Drell--Yan and the synthetic-Z plumbing case, predictions are evaluated
directly as degree-two functions of the physical grid fields. No
Hubbard--Stratonovich auxiliary is included in the sampled state, and there is
no complex phase or residual phase reweighting.

The default engine is NUTS with 100 warmup draws, a `0.1` initial step size,
tree depth 6, and the scalable prior-whitened identity metric. The prior cutoff
is zero, retaining every numerically positive covariance mode. Both shipped GP
covariance forms use jitter `1e-6`, matching the other isolated small-suite
variants. Results record the retained rank and whether the subspace is exact.

This route exchanges the H-S sign problem for a high-dimensional, potentially
multimodal real target. Posterior-quality runs should therefore inspect
divergences, maximum-depth hits, autocorrelation ESS, and independent-chain
agreement.

Like the other isolated development copies, this route retains experimental
results that miss a configured sampler-quality floor.  The JSON summary marks
them through `sampling_quality` rather than treating them as production-ready.

Run both truths and all cases:

```console
../pixel/venv/bin/python -m closure_small_q_integration.run_closure --Q 2
```

Run only the bilinear cases for one truth:

```console
../pixel/venv/bin/python -m closure_small_q_integration.run_closure \
  --truth NNPDF --Q 2 --modes dy exp both --n-samples 1000 --seed 7 --no-plots
```

The usual small-suite flags, including `--kernels-only`, `--plots-only`, and
`--fail-fast`, are forwarded unchanged.

## Evidence scans in physical field coordinates

`plot_evidence_scans.py` scans the scalar this variant's NUTS chain actually
integrates against -- `QTarget`'s joint action `S = -2 log p` over
`x = [u, zeta]` -- and draws it as a matrix of pairwise heatmaps, the same
figure `closure_JAM_truth_small/plot_evidence_scans.py` produces for the
Hubbard--Stratonovich route.

The two cannot share a parameter list. On the H-S route the free vector is
nineteen scalars and the GP fields are integrated out, so "the PDF at `x`" is
not a coordinate at all. Here the fields *are* the chain: seven fitted
normalizations followed by 1,314 prior-whitened retained modes, nine fields at
full rank 146 each because `Q_INTEGRATION_PRIOR_RCOND = 0`. A retained mode is
not a readable axis -- nobody can say what a fit looks like at
`v8 / retained[66] = 2`, which is the production run's slowest-mixing
coordinate -- so this scan reparametrizes: **one coordinate per
`(field, x)` pair, labelled `t3(x=0.0103)`, whose axis is that distribution's
value at that `x` in the fit's own units.** Thirty-four coordinates on the Q=2
`exp` case: seven normalizations and three `x` per field. That is why
`pixel.plotting.MAX_SCAN_PARAMETERS` was raised from 8 to 100.

The reparametrization is affine and exact. `x(z) = x_center + B (z - z_center)`
with `B` constant, and the suite measures rather than assumes its two claims:
moving an axis by one moves its own field value by one to `4e-16`, and the GP
mean and covariance are *exactly* independent of the seven normalizations
(measured drift `0.0`), so a normalization axis never moves a field value.

**Which way a field axis moves the distribution is a physics choice.** The
obvious one is useless: moving the single grid node and nothing else is the
exact conditional slice in physical `q`, but it is a delta spike whose prior
cost is `[K^-1]_kk`, astronomical on a smooth 146-node kernel, so the Laplace
window collapses to ~`1e-8` of the field and every panel draws the same
featureless parabola. The default is instead the GP's own conditional response,
`dq = K[:, k] / K[k, k]`, normalized so the value at `x_k` moves by exactly one
per unit of the axis -- "raise the PDF here" as the prior allows it, a smooth
bump, with prior cost `1 / K[k, k]` so the natural window is the prior width of
the field at that `x`. The price is that the field axes are not mutually
orthogonal: a bump at `x = 0.01` also moves `x = 0.1` by `K[k', k] / K[k, k]`.
The manifest stores the whole cross-response matrix at the scanned nodes, so
the leakage is auditable rather than assumed small. `--direction node` gives
the stiff alternative.

Unlike the H-S case the centre is a genuine mode: `QTarget` is a real positive
density, so `pixel.map`'s warning about bilinear MAPs does not apply. It is not
free. The production chain starts at the prior-mean origin, `S = 1.85e10`, and
L-BFGS-B alone does not get there -- the curvature's condition number is ~5e10,
and 4,000 iterations reach `S = 875` with `|grad S| = 1356` and still falling.
The MAP is therefore L-BFGS-B followed by Hessian-vector Newton-CG, which
reaches `|grad S| = 1e-6` in about fifty iterations. No dense 1,321 x 1,321
Hessian is formed. The centre is cached under `inference_cache/`, keyed on
`QTarget.metadata()` so a regenerated dataset invalidates it.

**A conditional slice needs a conditional window, and the scanner's default is
not one.** `scan_evidence` sizes an unbounded axis from the *marginal* Laplace
width `sqrt(2 [H^-1]_ii)`. Measured on a 16-coordinate pilot of this scan, that
left `sigma(x=0.1)` and `g(x=0.1)` blank -- every node past the `Delta S = 25`
colour ceiling -- because they sit on correlations of `-0.47` and `-0.99` with
their neighbours, which makes their marginal width sixteen times their
conditional one. The default here is therefore `--window conditional`:
`sqrt(2 / H_ii)`, the width of the slice actually being drawn, at five of which
the axis endpoint lands exactly on `Delta S = 25`, the outermost contour and the
colour ceiling. `--window marginal` restores the scanner's own rule.

**Estimate before launching.** `--estimate` builds everything, finds the centre,
prints both widths of every axis in physical units and times one real panel on
the real windows through the same jitted `lax.map` path the scan uses, then
stops without scanning:

```console
../pixel/venv/bin/python -m closure_small_q_integration.plot_evidence_scans \
    --truth JAM --Q 2 --mode exp --estimate
```

Then drop `--estimate` to launch. Measured on the Q=2 `exp` case, 2026-10-16:
34 coordinates, 561 pairs, 380,120 nodes at `n_points=26`; one real panel 3.06 s
warm, **4.53 ms per node, 28.8 min** for the whole scan, plus 9.2 min for the
MAP the first time (116 s L-BFGS-B, 435 s Newton-CG, ending at `S = 859.12` with
`|grad S| = 1.5e-06`) and ~35 s for the figure. The reduced 34 x 34 Hessian at
that centre has no negative eigenvalues, and a 16-coordinate pilot found the
centre to be the minimum of every scanned region (`worst_dip_below_reference`
exactly `0.0`) -- the two checks that it really is a mode. `n_points=51` is
1.46 million nodes, about 2.2 hours.

Use `--suffix` for a pilot, so a cut-down `--x`/`--n-points` run cannot overwrite
a production figure whose filename would look identical:

```console
../pixel/venv/bin/python -m closure_small_q_integration.plot_evidence_scans \
    --truth JAM --Q 2 --mode exp --x 0.1 --n-points 8 --figures scan \
    --suffix _pilot16cond
```

`plot_evidence_scan` lays out 3.1 x 3.0 inches per panel and
`pixel.plotting.style.save_figure` rasters at 180 dpi, which at 34 parameters is
19,000 x 18,000 px -- measured, that figure was killed by the OS on this 36 GB
machine. So this module draws with `path=None`, shrinks the figure to
`--figure-inches` (default 60) and picks its own dpi from a 40-megapixel budget.
It deliberately does **not** rasterize the meshes: that looks like the way to
keep a 1,156-panel PDF small, but matplotlib's PDF backend allocates a
full-figure Agg buffer per rasterized artist, and 1,156 of them measured 62.8 s
at 22.4 GB peak for a 6.0 MB file against 23.2 s at 1.49 GB for an 18.1 MB
vector one. As it stands the 34-parameter figure takes 5 s to draw and 27 s to
write, peaks at 1.6 GB, and comes out 10.5 MB PNG plus 18.1 MB PDF.
