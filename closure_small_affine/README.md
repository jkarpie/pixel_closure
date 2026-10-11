# Affine-contour small closure copy

This runnable copy covers both `closure_JAM_truth_small` and
`closure_NNPDF_truth_small`. It shares their generated data and kernel caches,
but writes independent outputs under `closure_small_affine/outputs/`. Sampler
families have separate result and inference-cache directories.

Every GP covariance, including the constant log-RBF and beta-envelope forms,
uses jitter `1e-6`. Bilinear H-S sectors remain on the affine saddle contour:
the normalization-reference route selects its affine-only map explicitly, and
fixed/joint routes use `T=0` with no positive-flow refinement.

This development copy serializes a run even when an importance or residual-ESS
quality floor is missed.  In that event `summary_<mode>.json` contains
`"sampling_quality": {"passed": false, "warnings": [...]}` alongside the
usual sampler diagnostics; such records are experimental results, not
production-quality fits.

Run both truths and all cases:

```console
../pixel/venv/bin/python -m closure_small_affine.run_closure --Q 2
```

Run selected cases:

```console
../pixel/venv/bin/python -m closure_small_affine.run_closure \
  --truth NNPDF --Q 2 --modes exp both --no-plots
```
