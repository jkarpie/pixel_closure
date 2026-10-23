# Full-scale stream and core scaling sweep

Prepare each requested truth/data/Q/mode cell's data and kernel cache once,
before launching streams. For example:

```bash
./run_closure.sh --truth JAM --data union --Q 2 --modes exp,dis \
  --kernels-only --cores 16 --python ../pixel/venv/bin/python
```

Repeat preparation for each truth/data selection and requested Q. Here
`--scales` means truth Q values (`mc,1,2,3,4,5`), while `--data` chooses the
full-scale dataset collection (`union,JAM,NNPDF`).

Then, on Linux, pin the *entire sweep* to the same 16 cores. The launcher runs
each cell serially and runs its 1-, 2-, and 4-stream arms in sequence; streams
within an arm run simultaneously. This makes the core budget identical across
arms rather than accidentally running all cells at once.

```bash
taskset -c 0-15 ../pixel/venv/bin/python -m campaign.run_full_scale_stream_sweep sweep \
  --out campaign/full_scale_core16 \
  --parameterizations hs,q --truths JAM,NNPDF --data union,JAM,NNPDF \
  --scales 2,3 --modes exp,dis --stream-counts 1,2,4 \
  --samples 2500 --dry-run
```

Remove `--dry-run` to execute. For an SSH session, wrap the same command in
`nohup ... > full_scale_core16.log 2>&1 &`, then record the PID with `echo $!`.
The requested `--out` directory must be new or empty; every stream gets a
private results directory, inference cache, `stream.json` timing/seed record,
PIXEL `chain.npz` and full `inference_summary.json`, and `run.log`. The root
`manifest.json` records the sweep status. Every completed cell gets `REPORT.md`
and `REPORT.json` from the saved
PIXEL chain objects. A completed stream-count arm does not overwrite another
arm. The first N seeds for a given cell are the same across the N-stream arms,
so scaling comparisons use the same independent chains.
For explicit-q cells, the launcher also shares a fingerprinted MAP cache across
all streams and stream-count arms. Each stream starts at that fitted center,
and its fixed dense Hessian metric is evaluated there. A MAP optimization that
does not converge stops the affected cell rather than starting at the prior
mean.

The report gives PIXEL's Wolff τ_int and ESS, split R-hat across streams,
trajectory acceptance statistics, energy error, step size, depth, divergences,
steps per trajectory, and the *actual* metric/fallback metadata. HMC's binary
acceptance rate and invalid transitions are distinguished from NUTS's mean
acceptance statistic and divergence flag. Sampler and total wall times are
recorded by the closure worker, not inferred from sampler logs. A one-stream
arm has no between-stream R-hat. The report does not invent worst-dataset or
worst-PDF statistics; the existing `summary_<mode>.json` retains the standard
fit and pull summaries.

To regenerate reports from completed streams without rerunning inference:

```bash
../pixel/venv/bin/python -m campaign.report_full_scale_stream_sweep \
  --root campaign/full_scale_core16
```

For a combined **terminal report**, including fit quality and pooled PDF pulls,
run the following on the machine holding the completed chains, posterior
moments and source pseudodata/kernel caches:

```bash
../pixel/venv/bin/python -m campaign.report_full_scale_closure \
  --root campaign/JAM_Q2_exp_prod \
  --out campaign/JAM_Q2_exp_prod/CLOSURE_REPORT
```

The command prints the report and optionally saves `CLOSURE_REPORT.md` and
`CLOSURE_REPORT.json`. It does not launch sampling or modify launch records.
It resolves stream paths relative to `--root`, so copied manifests containing
absolute paths from another machine work. It reads chains one at a time for
PIXEL's Wolff autocorrelation/ESS diagnostics and keeps only split-chain
statistics for R-hat, rather than holding every full-scale chain in memory.

For explicit-q streams, it pools the saved `posterior_moments_<mode>.npz`
artifacts by retained draw count, including the variance between stream means.
It uses their matched-draw predictive means for pooled chi2; averaging per-stream
chi2 or pulls would be incorrect. It rebuilds the source datasets and their
deterministic DIS t0 references, and refuses pooled chi2 if they do not reproduce
**each stream's saved chi2**. Source preparation can read/populate kernel caches;
it never calls the MCMC runner. The report includes mean and maximum PDF pulls,
fractions below 1 and 2, per-PDF signed pulls, chi2 components, and the worst
physical datasets by chi2 and chi2/npt. Chi2 totals include constraint rows;
dataset rankings exclude them. Pulls use finite nodes in `0.01 <= x <= 0.9`,
as in the small-scale Q-dense report; these nodes are correlated.

For a **small JSON-only transfer**, copy the updated root `manifest.json`, each
stream's `stream.json`, `inference_summary.json` and
`results/truthQ_<Q>/summary_<mode>.json`, and the existing cell `REPORT.json`.
Then add `--summary-only` to print saved statistics without reading chains or
rebuilding source data. A saved cell report is reused only if its cell identity,
inference summaries, draw counts and timings match the transferred stream
records. This mode reports per-stream fit/pull summaries; full pooled physics
requires the moments and source data on the original machine. The generated
`CLOSURE_REPORT.json` itself is small and contains the full pooled results when
the command succeeds, so that is the preferred file to bring back for analysis.

Exit status is `0` for a complete report and `2` when inputs are missing,
inconsistent or explicitly omitted with `--summary-only`. A `running` launch
record is only a saved snapshot, not evidence that the remote job is still
running. Requested sample counts are never treated as completed trajectories.

Each worker pins BLAS and PIXEL's internal task pools to one thread. `taskset`
is what fixes the *total* CPU affinity; `--cores` in `run_closure.sh` controls
internal workers, not a process-wide CPU cap. Dense full-scale chains can use
substantial memory, so start with a small single-cell pilot and check the peak
RSS before launching many streams. The sweep has no resume mode: use a fresh
output directory for each invocation. It does not generate data or prebuild
kernels while streams are running.

For a single production q cell, `run_closure.sh` exposes the same isolated
stream machinery directly, including explicit per-stream warmup:

```bash
./run_closure.sh --truth JAM --data union --Q 2 --modes exp \
  --parameterization q --warmup 500 --samples 2500 --streams 4 \
  --out campaign/jam_q2_exp_4streams --no-plots \
  --python ../pixel/venv/bin/python
```

The q path prepares missing source data, preserves H-S result directories,
and refuses a nonempty `--out`; it does not resume or overwrite a scan.
`--warmup` on the lower-level sweep is also available for q cells only; omit
it to retain each source suite's configured default.
