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
