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
