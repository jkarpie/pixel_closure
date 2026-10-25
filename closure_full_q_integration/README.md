# Full-scale q-integration closure

This variant samples the retained physical `q` coefficients explicitly with
NUTS, a full dense inverse-Hessian mass metric, and target acceptance 0.90.
Its other initial sampler settings follow the selected full-scale source suite.
For every full-scale truth/data suite and mode, prepare a local MAP in a
separate command before sampling. All streams for that target load the saved
MAP, and the dense inverse-Hessian metric is computed there. The launcher
validates each target before launching any workers; workers never optimize.
A missing or invalid MAP stops the run. A finite point above the requested
gradient tolerance produces a warning and sampling continues. Warmup adapts
the step size while keeping the dense mass fixed.

Prepare source pseudodata and kernels first, then run:

```bash
../pixel/venv/bin/python -m closure_full_q_integration.prepare_map \
  --truth JAM --data union --Q 2 --modes exp \
  --out campaign/JAM_Q2_exp_prod \
  --lbfgs-maxiter 40000 --lbfgs-maxfun 400000 --lbfgs-gtol 1e-8 \
  --adam-steps 2000,2000,2000 --adam-learning-rates 0.01,0.003,0.001 \
  --refine-lbfgs-maxiter 10000 --refine-lbfgs-gtol 1e-10 \
  --map-gradient-tol 1e-3 --newton-gtol 1e-6 \
  --newton-maxiter 2000
```

The default search is up to 40,000 L-BFGS-B iterations, then three Adam passes
(2,000 steps each, at learning rates 0.01, 0.003, and 0.001), then up to
10,000 refinement L-BFGS-B iterations, then up to 2,000 Newton trust-region
iterations with exact Hessian-vector products.
L-BFGS-B first locates a promising region, Adam can continue along narrow
valleys, the second L-BFGS-B pass can settle from Adam's best point, and
Newton refines the result toward stationarity.
Adam uses PIXEL's bias-corrected implementation on the same explicit-q action
and analytic gradient, with beta1=0.9, beta2=0.999, and epsilon=1e-8. Each pass
starts from the best finite candidate so far and resets its moment estimates.
An overshoot cannot worsen the starting point handed to the second L-BFGS-B
pass. Reaching an Adam step budget is expected; it does not certify convergence.
The schedule is configurable and has not been benchmarked on a completed full
JAM search.

`--adam-steps` and `--adam-learning-rates` accept equal-length comma lists.
The first L-BFGS-B pass accepts `--lbfgs-maxiter`, `--lbfgs-maxfun`,
`--lbfgs-gtol`, `--lbfgs-maxls`, and `--lbfgs-maxcor`. The post-Adam pass has
independent `--refine-lbfgs-maxiter`,
`--refine-lbfgs-maxfun`, `--refine-lbfgs-gtol`, `--refine-lbfgs-maxls`, and
`--refine-lbfgs-maxcor` options. Its defaults are 10,000 iterations, 100,000
function evaluations, gradient tolerance `1e-10`, 100 line-search steps, and
100 correction pairs; the first pass uses 40,000 iterations, 400,000
evaluations, gradient tolerance `1e-8`, 100 line-search steps, and 50 pairs.
Set `--refine-lbfgs-maxiter 0` to skip the second pass; `--adam-steps 0`
skips Adam. The function-evaluation budget for L-BFGS is also raised, and the
L-BFGS relative-action stopping tolerance is zero to avoid early termination
from a small relative action change. `--map-gradient-tol` sets the Euclidean
gradient norm at which the result is considered satisfactory (default `1e-3`).
`--newton-gtol` sets Newton's internal stopping tolerance (default `1e-6`);
Newton uses the stricter of this and `--map-gradient-tol`. The batch script
accepts `PIXEL_MAP_GRADIENT_TOL` and `PIXEL_MAP_NEWTON_GTOL` for these values,
plus `PIXEL_MAP_LBFGS_*` and `PIXEL_MAP_REFINE_LBFGS_*` for the two L-BFGS-B
passes.
A point above the requested MAP gradient tolerance is still saved and sampled
with a warning; it may yield a less useful Hessian metric. A nonfinite or
mismatched saved point is rejected. Larger budgets cannot guarantee convergence
or a global optimum. The final DIS t0 target is built exactly as
in the sampling runners. This command does not sample or reconstruct PDFs.

With `--out campaign/RUN`, MAP files live in
`campaign/RUN/_q_map_cache` alongside that campaign's stream outputs. This
is the batch submission default, and the stream sweep reads the same path.
Without `--out`, the standalone preparation command retains its default
`campaign/_q_map_cache`. For each numerical target fingerprint:

- `q_map_<hash>.npz` contains the best finite MAP candidate and provenance used by streams.
- `q_map_<hash>.attempt.json` records status, action, gradient norm, budgets,
  elapsed time, stopping messages, and iteration/evaluation counts for every stage,
  including failures.
- `q_map_<hash>.checkpoint.npz` contains the best recorded finite candidate.
- `q_map_<hash>.history.jsonl` retains progress across attempts.

Checkpoints are saved every 50 iterations and at stage boundaries; change
this with `--checkpoint-every`. Terminal progress defaults to every 100
iterations (`--report-every`) without changing the checkpoint frequency.
The report interval must be a multiple of the checkpoint interval.
Each progress line gives both the Euclidean gradient norm and its infinity norm
(largest absolute component) at the best-action point. It also reports the
measured time and step rate since the previous line, plus remaining iterations
and an ETA to the current stage's iteration limit. That ETA is an extrapolation
of recent speed, not a convergence prediction; the stage can stop earlier.
On interruption, the last saved checkpoint survives. To restart optimization
from that point with fresh optimizer state, repeat the preparation command with
`--resume` and optionally larger budgets.
This restarts the selected L-BFGS/Adam/L-BFGS/Newton sequence with fresh
optimizer state when the saved point exceeds the requested tolerance. A saved
finite point is otherwise reused, with a warning if above tolerance; pass
`--resume` to try to improve it. Use `--resume --adam-steps 0
--refine-lbfgs-maxiter 0` to recover the first L-BFGS/Newton sequence from the
saved candidate. A matching
point within tolerance is always reused. Use `--check-only` to
validate without optimization.

The fingerprint includes numerical data, covariance, prior and coordinates;
iteration budgets do not change it. Loading also recomputes the actual action
and gradient. Changing the target requires fresh preparation. Changing the stage
sequence does not change the fingerprint: existing validated MAPs and matching
candidate checkpoints remain usable. The requested warning tolerance is saved
with the MAP so stream preflight and workers use the same threshold.

Logs label the full search as `lbfgs`, `adam_1`, `adam_2`, `adam_3`,
`lbfgs_refine`, and `newton`, and record each stage's starting action and
gradient. The preliminary DIS t0 covariance fits retain their existing policy.

To prepare a campaign independently, pass `--out campaign/RUN` to MAP
preparation, then the same `--out campaign/RUN` to `run_closure.sh` or the
stream sweep. The directory may contain `_q_map_cache` before sampling, but
no prior production artifacts. For a custom cache outside the campaign, use
`--cache-root` for preparation and `--map-cache-root` for the sweep, or export
`PIXEL_Q_MAP_CACHE_ROOT` for both stages. The standalone preparation command
accepts either `--out` or `--cache-root`, not both.
`--Q` and `--modes` accept comma lists; repeat for each truth/data selection.

The source suites own generated truth, pseudodata, and kernel caches. This
variant writes results and inference caches under `outputs/<truth>/<data>/`,
so its outputs do not replace Hubbard-Stratonovich results.

```bash
../pixel/venv/bin/python -m closure_full_q_integration.prepare_map \
  --truth JAM --data union --Q 2 --modes both

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
must be new or contain only the prepared `_q_map_cache`; the launcher never
merges into or clears an existing q run. Each stream gets its own seed, results,
cache, PIXEL chain and timing
record; the prepared MAP cache is shared across streams and stream-count arms. A
completed cell gets `REPORT.json` and `REPORT.md`. This path always
skips plots. `--cores` controls data/kernel preparation, whereas the q sweep
pins each sampler worker's inner pools to one thread; use `taskset` to constrain
the total CPU affinity on Linux.

The sampling arguments in `submit_closure.sh` are unchanged. The batch script
now performs kernel preparation, runs the standalone MAP command once (or
reuses its validated result), and launches production only on success. A MAP
prepared independently with `--out campaign/JAM_Q2_exp_prod` is therefore
reused by the submission script. Its cache is
`campaign/JAM_Q2_exp_prod/_q_map_cache`; the batch script overrides any
previous `PIXEL_Q_MAP_CACHE_ROOT` value so the streams use that campaign's MAP.
Its MAP terminal output is saved to `campaign/JAM_Q2_exp_prod.map.log`.
`PIXEL_MAP_ADAM_STEPS`, `PIXEL_MAP_ADAM_LEARNING_RATES`,
`PIXEL_MAP_LBFGS_MAXITER`, `PIXEL_MAP_NEWTON_MAXITER`, and
`PIXEL_MAP_LBFGS_MAXFUN` override the batch preparation budgets.
`PIXEL_MAP_REPORT_EVERY` changes terminal progress cadence; checkpointing
still defaults to every 50 iterations. Existing production results are
preserved; select a fresh output path or archive the
old directory explicitly before another production run.
