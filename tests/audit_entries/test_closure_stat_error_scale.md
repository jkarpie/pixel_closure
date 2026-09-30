# `tests/test_closure_stat_error_scale.py`

**Exercises** `closure_NNPDF_truth_small.generate.generate_member` and the JAM small-suite copy, including lattice, DIS, E866 DY, and synthetic-Z statistical draws. Operators and truth curves are mocked so the generator's write path is exercised without kernel assembly.

**Claim** Every statistical standard deviation scales by `k`, covariance by `k²`, and the same standard-normal draw is applied about a nuisance-injected, statistically noise-free center. The old default call is byte-identical to explicit `stat_error_scale=1, stat_noise=None`.

**Oracle** The `d_k = c_nuisance + k(d_1-c_nuisance)` and `C_k = k² C_1` identities are algebraic (`A3`) but both data files share generator code. The test separately checks correlated-systematic sidecars and truth bytes remain unchanged, then rejects zero scale. It is a generator-wiring check, not a DY physics oracle.

**Cost** One parametrized test, two cases (NNPDF and JAM), mocked arrays and temporary files. `2 passed in 0.80s` on 2026-09-15.

| Test | Assertion | Bar | Strength |
|---|---|---|---|
| `test_scaled_member_preserves_nuisances_and_default_bytes` | Default and explicit-k1 files hash equally; each of four data files obeys nuisance-centered scaling and covariance scaling; sidecars and truth hash equally; zero scale raises. | exact SHA equality, `allclose` at `rtol=1e-11`, `atol=1e-11`, `ValueError` | `S2` |

**Weak spots** The expected center comes from the same generator at `stat_noise=False`. A generator bug common to all three precision calls could satisfy the identity. Mocked operators cannot reveal errors in real closure kernels or truth construction.

**Not covered here** Full NNPDF/JAM closure target assembly is measured by PIXEL's sampled-contour stiffness benchmark. Its fixed data seed and shared kernel cache keep sampler seed and algorithm out of mock-data generation.

**Mutation check (2026-09-15)** A temporary pytest plugin replaced only the two generators' in-memory `draw` binding so statistical draws ignored the requested standard deviation. Unmodified: **2 passed**. Mutated: **2 failed** at the nuisance-centered data identity. Repository source was unchanged.
