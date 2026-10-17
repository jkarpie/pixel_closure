# Evidence scan — small closure, JAM truth Q=2, mode `exp`, **explicit-`q` route**

Generated 2026-10-17. Figures are PNG + PDF, with the raw panels kept as
`q_evidence_scan_exp_scan.npz` so any panel is re-readable without a second
scan. `q_evidence_scan_exp_manifest.json` carries the centre, windows, bounds
provenance, per-axis variation, the slice's own residuals and timings.

Kernel caches and the DIS t0 reference (9 tables, 8 iterations) were reused. The
MAP is cached under `inference_cache/truthQ_2_exp_q_map.npz`, keyed on
`QTarget.metadata()`.

This is the counterpart of
`closure_JAM_truth_small/results/truthQ_2/EVIDENCE_SCAN_exp.md`, which scans the
Hubbard–Stratonovich route on the same truth, Q and mode. **The two scan
different objectives in different coordinates and the numbers are not directly
comparable** — see "Against the H-S scan" at the end, which says exactly where
they can and cannot be read against each other.

## What was scanned

`QTarget`'s joint action `S = -2 log p` over `x = [u, zeta]`: **7 fitted
normalization amplitudes** (5 DIS, 2 DY) followed by **1,314 prior-whitened
retained field modes** — nine fields at full rank 146 each, because
`Q_INTEGRATION_PRIOR_RCOND = 0` retains every positive prior mode
(`approximate_subspace: false`). No H-S auxiliary is in the sampled state.

A retained mode is not a readable axis — nobody can say what a fit looks like at
`v8 / retained[66] = 2`, which is the production run's slowest-mixing
coordinate — so the scan is reparametrized onto **34 physical coordinates**: the
7 normalizations, plus the value of each of the 9 distributions at 3 values of
`x`. That is why `pixel.plotting.MAX_SCAN_PARAMETERS` was raised from 8 to 100
on 2026-10-16; at eight per figure, no split answers "how do the normalizations
trade against the fields" in one picture.

| figure | parameters | per-pair grid |
|---|---|---|
| `q_evidence_scan_exp` | all 34 at once | 26×26 |
| `q_evidence_scan_exp_laplace_correlation` | all 34 (exact reduced Hessian) | — |
| `q_evidence_scan_exp_pilot16cond` | 7 norms + 9 fields at `x=0.1` | 8×8 |

561 pairs, 1,156 panels, 380,120 nodes. **Zero masked nodes and zero warnings.**
Scan 1,820 s (30.3 min) against a 1,702 s projection from one timed panel — 7%
high. Figure 5 s to draw, 27 s to write, 1.6 GB peak, 10.5 MB PNG + 18.1 MB PDF.

### The reparametrization is exact, and measured rather than asserted

`x(z) = x_center + B (z - z_center)` with `B` constant. Three claims, all
measured through `coordinates.physical` and recorded in the manifest:

* moving an axis by one moves its own quantity by one to **8.9e-16**;
* the GP mean and covariance are **exactly** independent of the seven
  normalizations (`max_field_drift_per_unit_normalization` = `0.0`), so a
  normalization axis never moves a field value;
* field axes move along the GP conditional response `K[:, k] / K[k, k]`, so they
  are not mutually orthogonal. The largest leak in the whole slice: moving
  `sigma` at `x = 0.1` by one moves `sigma` at `x = 0.495` by **0.070**. The
  full 27×27 cross-response matrix is in the manifest.

The single-grid-node alternative (`--direction node`) is the exact conditional
slice in physical `q` and is useless: its prior cost is `[K^-1]_kk`,
astronomical on a smooth 146-node kernel, so its window collapses to ~`1e-8` of
the field and every panel draws the same featureless parabola.

### The centre is a genuine mode, and that is checkable here

Unlike the H-S case, `QTarget` is a real positive density, so `pixel.map`'s
warning about bilinear MAPs does not apply. The production chain starts at the
prior-mean origin, `S = 1.85e10`; L-BFGS-B alone does not get off it (4,000
iterations reach `S = 872.6` with `|grad S| = 755` and still falling — the
curvature's condition number is ~5e10), so the centre is L-BFGS-B followed by
Hessian-vector Newton-CG: 57 iterations, 435 s, **`S = 859.1166`,
`|grad S| = 1.5e-06`**. No dense 1,321×1,321 Hessian is formed.

Two independent confirmations that it is a mode:

* the reduced 34×34 Hessian has **no negative eigenvalues** (range
  `1.29` … `1.66e+09`);
* `worst_dip_below_reference` is **exactly `0.0`** — across all 561 panels and
  all 380,120 nodes, nothing beat the centre.

## Finding 1 — there is no second minimum

Nothing in a ±5 conditional-sigma box around the MAP, over any pair of the 34,
falls below the reference by any amount at all. **The explicit-`q` route's
sampling problem is not multimodality.** That is a load-bearing negative result:
tempering, replica exchange and contour flow all address a barrier, and there is
no barrier here to address.

The caveat the method itself imposes: a slice is not a profile, and a minimum
further out than 5 conditional sigma along some scanned axis is outside the
window and invisible at any resolution. What is excluded is a second minimum
*inside the box*, in any of the 561 scanned planes.

## Finding 2 — the problem is anisotropy, and it is inside the PDFs

Fitting each panel's 2×2 curvature and ranking by the basin's aspect ratio
`sqrt(lambda_max / lambda_min)`. `r` below is the **partial** (conditional)
correlation of the pair, which is what a slice shows:

| aspect | partial `r` | pair |
|---|---|---|
| **578** | −1.0000 | `v8(x=0.0107) × v8(x=0.1)` |
| **575** | −1.0000 | `v15(x=0.0107) × v15(x=0.1)` |
| **421** | −1.0000 | `v(x=0.0107) × v(x=0.1)` |
| **259** | −1.0000 | `v3(x=0.0107) × v3(x=0.1)` |
| **107** | −0.9998 | `g(x=0.0107) × g(x=0.1)` |
| 19.9 | +0.9949 | `t3(x=0.0107) × t15(x=0.0107)` |
| 16.0 | +0.9922 | `t8(x=0.0107) × t15(x=0.0107)` |
| 14.6 | −0.9906 | `t3(x=0.0107) × t8(x=0.0107)` |
| 13.2 | −0.9886 | `sigma(x=0.1) × g(x=0.1)` |
| 5.5 | −0.9367 | `sigma(x=0.0107) × g(x=0.1)` |

**Median aspect ratio over all 561 pairs: `1.000`.** Only 21 pairs exceed 3, ten
exceed 10, five exceed 100 — and **14 of those 21 are one field at two `x`**.
The posterior is round almost everywhere and knife-edged in a handful of
within-field directions.

The four valence distributions are the extreme case: `v3`, `v8`, `v` and `v15`
each have a low-`x`/mid-`x` pair at partial `r = −1.0000` to four decimals.
Those are not "strongly correlated"; they are degenerate to the precision of the
fit. The marginal correlations agree (`−1.0000` for all four).

## Finding 3 — the normalizations are not the problem

| block | max \|aspect\| in the slice | max \|rho\| marginal |
|---|---|---|
| normalization × normalization | **1.01** | 0.469 (`slac_d × bcdms_d`) |
| normalization × field | **1.92** | 0.665 (`slac_d × t3(x=0.1)`) |
| field × field | 578 | 1.000 |

The 7×7 normalization block is essentially a perfect sphere in the slice. The
strongest normalization-against-field panel in the entire figure is
`slac_p N × t3(x=0.1)` at aspect **1.92**. Among the marginal correlations,
`dy_e866_pd` does reach ±0.54…0.56 against `v15` and `t15` — the DY normalization
trades against the valence and `t15` shapes, which is the physically expected
channel — but none of it produces a ridge a sampler has to crawl along.

**If the question is how the normalizations integrate against the rest, the
explicit-`q` answer is that they don't fight anything.** The pathology is
entirely between a distribution's own values at different `x`.

## Finding 4 — conditional and marginal widths differ by up to 2,200×

Windows here are **conditional**, `sqrt(2 / H_ii)`, not the scanner's default
marginal `sqrt(2 [H^-1]_ii)`. This was forced by the data, not chosen for
tidiness: on a 16-coordinate pilot sized the scanner's way, `sigma(x=0.1)` and
`g(x=0.1)` drew completely blank, every node past the `Delta S = 25` colour
ceiling.

| axis | conditional | marginal | ratio |
|---|---|---|---|
| `v8(x=0.1)` | 5.76e-05 | 0.129 | **2,236** |
| `v15(x=0.1)` | 5.76e-05 | 0.103 | **1,782** |
| `v3(x=0.1)` | 5.76e-05 | 0.0443 | 769 |
| `sigma(x=0.1)` | 4.49e-04 | 0.0145 | 32 |
| `bcdms_p N` | 0.150 | 0.159 | **1.06** |

The split is the same one as Findings 2 and 3: every normalization is within
20% of its own marginal width, while the degenerate valence coordinates are a
factor of a thousand narrower conditionally than marginally. **That ratio *is*
the degeneracy**, read one axis at a time.

Along each held axis `Delta S = H_ii d^2 / 2`, so a 5-conditional-sigma window
puts the axis endpoint at `Delta S = 25` exactly. Measured: 32 of 34 diagonals
vary by `24.86` across their window. The two exceptions are the DY
normalizations at `28.15` and `28.43` — a genuine, mild anharmonicity, the only
non-quadratic direction in the slice. Panel corners reach `Delta S = 99.6` and
26.9% of nodes sit past the colour ceiling; that is the 5σ×5σ corner, by
construction, not saturation hiding structure.

## What that implies for the sampler

* **A prior-whitened identity metric cannot work.** `Q_INTEGRATION_METRIC =
  "prior_whitened"` is identity mass in a frame chosen by the *prior*, and the
  anisotropy here is put there by the *likelihood*. An axis ratio of 578 forces
  the step size down to the stiff direction while the soft direction needs
  ~578× more steps to cross.
* **It corroborates the production run exactly.** `summary_exp.json` reports
  `tau_int_max = 117`, `ESS = 4.3` of 1,000, `slowest_param =
  "v8 / retained[66]"`, and a dual-averaged step size of `1.79e-05` — and `v8`
  is the single worst pair in this scan. The chain was not stuck behind a
  barrier; it was crawling along a ridge.
* **The right fix is a metric, not a sampler family.** `legacy_hessian_dense`
  already exists in this variant (`Q_INTEGRATION_METRIC` in
  `small_closure_variant.py`) and is the direct answer to a unimodal,
  ill-conditioned target. A dense 1,321×1,321 Hessian cost **61.5 s** to build
  at the prior-mean origin (`streams/q2_exp_5k_2026-10-11/hessian_metric_q2_exp.json`)
  — negligible against the 2,623 s that chain spent sampling. It was *not* built
  at the MAP, and at the origin 37 of its 1,321 eigenvalues are negative, so that
  artifact is a cost measurement and not a usable metric.
* **A block structure would probably be enough**, and is much cheaper than a
  dense 1,321² metric: identity on the normalizations, which need nothing, and
  per-field 146×146 blocks, since 14 of the 21 anisotropic pairs are within one
  field and no normalization pair exceeds aspect 1.01. The cross-field
  exceptions are `t3/t8/t15` at low `x` and `sigma/g`, which share a prior
  block only through the data.
* **Reducing the retained rank would not help and might hurt.** The degeneracy
  is between physical field values, not between prior modes; `rcond = 0` is not
  what put it there.

## Against the H-S scan

The same truth, Q and mode, scanned on the other route, is in
`closure_JAM_truth_small/results/truthQ_2/EVIDENCE_SCAN_exp.md`. Four things are
worth reading across, and one comparison is a trap.

* **Different objectives.** That scan is of `Model._raw_value`, `Re E^2` on the
  undeformed real H-S plane, with the GP fields integrated analytically. This
  one is of `QTarget`'s normalized joint action over explicit fields. They are
  different scalars over different vectors.
* **The H-S route has no field coordinates at all**, so Findings 1–3 here have
  no counterpart there, and its f-parity argument has no counterpart here.
* **Both find the normalizations to be the well-behaved block**, by different
  routes — there via exact decoupling from the auxiliaries by parity, here via
  an aspect ratio of 1.01.
* **The trap: `slac_d × bcdms_d` is `+0.934` there and `+0.469` here, and that
  is not the same quantity.** Over there it is marginal over the whole 19-vector
  with the fields analytically integrated. Here it is marginal only within the
  34-coordinate reduced block, holding the other 1,287 retained field modes
  fixed — and holding the fields is precisely what suppresses the
  PDF-absorbs-a-normalization degeneracy that `fit.py` warns about. **The
  degeneracy has not gone away; this slice conditions on the thing that causes
  it.** Measuring it on this route needs the normalization block of
  `2 H^-1` over the full 1,321 coordinates, which this scan does not compute.
* **The two routes' hard problems are different in kind.** There the dominant
  term is the sign problem — phase ESS `0.0066`, with the metric worth a factor
  of a few and the sign problem a factor of ~150. Here there is no sign problem
  and no second minimum; the entire deficit is conditioning. That is the real
  trade the explicit-`q` route makes, now measured on both sides.

## Reproducing

```console
../pixel/venv/bin/python -m closure_small_q_integration.plot_evidence_scans \
    --truth JAM --Q 2 --mode exp --n-points 26
```

`--estimate` stops after the centre, printing both widths of every axis and the
projected cost from one timed panel. `--window marginal` restores the scanner's
own window rule, `--direction node` the stiff single-node field axes, and
`--suffix` keeps a cut-down pilot from overwriting these files. The MAP is
cached, so a re-run costs the scan alone.
