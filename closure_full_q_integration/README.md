# Full-scale q-integration closure

This variant samples the retained physical `q` coefficients explicitly with
NUTS, a full dense inverse-Hessian mass metric, and target acceptance 0.90.
Its other initial sampler settings follow the selected full-scale source suite.
For every full-scale truth/data suite and mode, the explicit-q action is first
optimized from the prior-mean origin to a local MAP. All streams for the same
target start at that MAP, and the dense inverse-Hessian metric is computed
there, not at the origin. The MAP is fingerprinted against the numerical fit
target and cached under a shared directory; concurrent streams wait for one
optimization rather than doing it separately. A nonconverged MAP stops the
run instead of silently reverting to the origin. Warmup then adapts the step
size while keeping the dense mass fixed.

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

For one production q cell with independent streams, the top-level launcher
prepares missing source data and checks/builds shared kernels once before
using the same isolated stream worker:

```bash
./run_closure.sh --truth JAM --data union --Q 2 --modes exp \
  --parameterization q --warmup 500 --samples 2500 --streams 4 \
  --seed-base 20261018 --out campaign/jam_q2_exp_4streams --no-plots \
  --python ../pixel/venv/bin/python
```

`--warmup` and `--samples` apply **to each stream**. Omit `--warmup` to use
the selected source suite's q warmup (250 for JAM/union). The output root
must be new or empty; the launcher never merges into or clears an existing q
run. Each stream gets its own seed, results, cache, PIXEL chain and timing
record; the MAP cache is shared across streams and stream-count arms. A
completed cell gets `REPORT.json` and `REPORT.md`. This path always
skips plots. `--cores` controls data/kernel preparation, whereas the q sweep
pins each sampler worker's inner pools to one thread; use `taskset` to constrain
the total CPU affinity on Linux.
