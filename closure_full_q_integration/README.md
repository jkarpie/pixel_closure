# Full-scale q-integration closure

This variant samples the retained physical `q` coefficients explicitly with
NUTS, a full dense inverse-Hessian mass metric, and target acceptance 0.90.
Its other initial sampler settings follow the selected full-scale source suite.
For every full-scale truth/data suite and mode, prepare a local MAP in a
separate command before sampling. All streams for that target load the saved
MAP, and the dense inverse-Hessian metric is computed there. The launcher
validates each target before launching any workers; workers never optimize.
A missing or nonconverged MAP stops the run. Warmup adapts the step size while
keeping the dense mass fixed.

Prepare source pseudodata and kernels first, then run:

```bash
../pixel/venv/bin/python -m closure_full_q_integration.prepare_map \
  --truth JAM --data union --Q 2 --modes exp \
  --lbfgs-maxiter 40000 --lbfgs-maxfun 400000 \
  --adam-steps 2000,2000,2000 --adam-learning-rates 0.01,0.003,0.001 \
  --newton-maxiter 2000
```

The default search is up to 40,000 L-BFGS-B iterations, then three Adam passes
(2,000 steps each, at learning rates 0.01, 0.003, and 0.001), then up to
2,000 Newton trust-region iterations with exact Hessian-vector products.
L-BFGS-B first locates a promising region, Adam can continue along narrow
valleys, and Newton refines the resulting point toward stationarity.
Adam uses PIXEL's bias-corrected implementation on the same explicit-q action
and analytic gradient, with beta1=0.9, beta2=0.999, and epsilon=1e-8. Each pass
starts from the best finite candidate so far and resets its moment estimates.
An overshoot cannot worsen the starting point handed to Newton. Reaching an
Adam step budget is expected; it does not certify convergence. The schedule
is configurable and has not been benchmarked on a completed full JAM search.

`--adam-steps` and `--adam-learning-rates` accept equal-length comma lists.
Use `--adam-steps 0` to recover the L-BFGS/Newton-only search. The function-
evaluation budget for L-BFGS is also raised, and the
L-BFGS relative-action stopping tolerance is zero to avoid early termination
from a small relative action change. Finite action and a Euclidean action
gradient norm at most `1e-3` remain mandatory; larger budgets cannot guarantee
convergence or a global optimum. The final DIS t0 target is built exactly as
in the sampling runners. This command does not sample or reconstruct PDFs.

By default files live in `campaign/_q_map_cache`, independent of production
output directories. For each numerical target fingerprint:

- `q_map_<hash>.npz` contains the converged MAP and provenance used by streams.
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
This restarts the selected L-BFGS/Adam/Newton sequence with fresh optimizer
state. Use `--resume --adam-steps 0` to go straight to L-BFGS/Newton from the
saved candidate. A matching converged MAP is always reused. Use `--check-only` to
validate without optimization.

The fingerprint includes numerical data, covariance, prior and coordinates;
iteration budgets do not change it. Loading also recomputes the actual action
and gradient. Changing the target requires fresh preparation. Adding Adam does
not change the fingerprint: existing validated MAPs and matching candidate
checkpoints from the L-BFGS/Newton driver remain usable.

Logs label the full search as `lbfgs`, `adam_1`, `adam_2`, `adam_3`, and
`newton`, and record each stage's starting action and gradient. Adam is added
to the full explicit-q search; the preliminary DIS t0 covariance fits retain
their existing policy.

To choose another shared location, export `PIXEL_Q_MAP_CACHE_ROOT` before both
commands. The preparation command also accepts `--cache-root`; the variant
runner and sweep accept `--map-cache-root`. Keep these paths consistent.
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
must be new or empty; the launcher never merges into or clears an existing q
run. Each stream gets its own seed, results, cache, PIXEL chain and timing
record; the prepared MAP cache is shared across streams and stream-count arms. A
completed cell gets `REPORT.json` and `REPORT.md`. This path always
skips plots. `--cores` controls data/kernel preparation, whereas the q sweep
pins each sampler worker's inner pools to one thread; use `taskset` to constrain
the total CPU affinity on Linux.

The sampling arguments in `submit_closure.sh` are unchanged. The batch script
now performs kernel preparation, runs the standalone MAP command once (or
reuses its validated result), and launches production only on success. A MAP
prepared independently is therefore reused by the existing submission script.
Its MAP terminal output is saved to `campaign/JAM_Q2_exp_prod.map.log`.
`PIXEL_MAP_ADAM_STEPS`, `PIXEL_MAP_ADAM_LEARNING_RATES`,
`PIXEL_MAP_LBFGS_MAXITER`, `PIXEL_MAP_NEWTON_MAXITER`, and
`PIXEL_MAP_LBFGS_MAXFUN` override the batch preparation budgets.
`PIXEL_MAP_REPORT_EVERY` changes terminal progress cadence; checkpointing
still defaults to every 50 iterations. Existing production results are
preserved; select a fresh output path or archive the
old directory explicitly before another production run.
