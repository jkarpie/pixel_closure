# Flowed small closure copy

This runnable copy covers both `closure_JAM_truth_small` and
`closure_NNPDF_truth_small`. It shares their generated data and kernel caches,
but writes independent outputs under `closure_small_flowed/outputs/`. Sampler
families have separate result and inference-cache directories.

Every GP covariance, including the constant log-RBF and beta-envelope forms,
uses jitter `1e-6`. Every bilinear H-S route flows the real H-S plane to
`T=0.2` with eight RK4 steps; it does not start from the affine saddle contour.
The normalization-reference route has an explicit 110 GiB admission ceiling
because its calibrated projection is about 99.6 GiB on the Q2 small-suite grid.

Run both truths and all cases:

```console
../pixel/venv/bin/python -m closure_small_flowed.run_closure --Q 2
```

Run selected cases:

```console
../pixel/venv/bin/python -m closure_small_flowed.run_closure \
  --truth JAM --Q 2 --modes exp both --no-plots
```
