# Full-scale q-integration closure

This variant samples the retained physical `q` coefficients explicitly with
NUTS, a full dense inverse-Hessian mass metric, and target acceptance 0.90.
Its other initial sampler settings follow the selected full-scale source suite.
Warmup adapts the step size while keeping the dense mass fixed.

The source suites own generated truth, pseudodata, and kernel caches. This
variant writes results and inference caches under `outputs/<truth>/<data>/`,
so its outputs do not replace Hubbard-Stratonovich results.

```bash
../pixel/venv/bin/python -m closure_full_q_integration.run_closure \
  --truth JAM --data union --Q 2 --modes both --no-plots
```

`--truth` accepts `JAM` or `NNPDF`; `--data` accepts `union`, `JAM`, or
`NNPDF`. The source runner's `--Q`, `--modes`, `--no-plots`, `--kernels-only`,
and `--fail-fast` options are forwarded. Optional `--n-samples` and `--seed`
override the source configuration for this invocation. `--output-root` writes
results and inference caches to a private directory, which is required when
running independent streams concurrently. Prepare the selected
source suite's pseudodata and kernels first with `run_closure.sh` if absent.
No scan is launched by creating this variant.

For a comma-list sweep of full-scale H-S and/or explicit-q streams, see
[the full-scale stream sweep](../campaign/README_full_scale_stream_sweep.md).
