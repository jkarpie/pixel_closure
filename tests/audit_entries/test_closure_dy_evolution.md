### `tests/test_closure_dy_evolution.py`

**Exercises** `src/pixel/geometry` (`Grid`, `ProductBasis`);
`src/pixel/kernels/evolution/flavor_transition.py`'s
`nf4_parton_evolution_projection` / `rowwise_parton_evolution_projection`
(real, unstubbed bottom-threshold decision `Q20 < AlphaS.mb_match2 <= Q2`);
`src/pixel/data/drell_yan.py` (`DrellYan`, `BilinearContribution`); and, via
`importlib`, `closure_JAM_truth`'s and `closure_NNPDF_truth`'s `config.py`,
`datasets.py` (`build_drell_yan`, `dy_evolution_maps`) and `generate.py`
(`dy_central`) -- never the `_small` suites. `_stub_evolution` monkeypatches
only the two leaf functions `flavor_transition.non_singlet_evolution_matrix`
and `.singlet_evolution_matrix`, so no test here assembles a Mellin contour or
checks an evolution operator's numeric value. A second stub,
`_stub_evolution_flavour_mixing`, exists because the first one cannot resolve
the two hadron axes: it returns `factor * eye`, and a scalar multiple of the
identity makes the two-sided fold *exactly* invariant under exchanging the two
operators (measured swap ratio 0.000, bit for bit).

**Claim** `DY_EVOLVE` (previously a dead constant -- `build_drell_yan`
hardcoded `evo = None` regardless) actually reaches the built dataset and every
contribution it builds; `generate.dy_central` folds the *same* assembled
operator the dataset carries -- evolved or not -- rather than a separately
written physics path; each hadron's evolution operator is composed onto *its
own* tensor axis; and the shared-operator block sparsity survives evolution by
a measured margin.

**Oracles** The two fold comparisons are `A3`: two foldings of the same
`dataset.bilinear_contributions`, differing only in grouping/recombination
order. Per `tests/README.md` rule 1, both sides of those read the same
`field_A`/`field_B` labels and the same composed tensor
(`c.assemble(...)` / `c.kernel.bilinear_tensor(...)`) off the same contribution
objects, so a field mislabelling upstream would reproduce identically on both
sides and pass -- DY is bilinear (one basis axis per hadron) and this file hands
one `_basis()` fixture to both axes of every tensor.
`test_assemble_composes_each_hadron_operator_onto_its_own_axis` (added
2026-08-13) is the one comparison that is *not* symmetric under exchanging the
two hadrons: it folds the bare tensor with separately evolved curves against the
assembled tensor's fold, so an axis-transposed `evolved_tensor` breaks it (still
`A3` -- same tensor, same operators -- but sensitive to which axis each operator
lands on). Three structural/error-path tests (`F1`) exercise real guard code
directly: the nf-consistency check, the bottom-threshold parton-membership
decision, and the operator-sharing group count.

**Cost** 6 tests, 14 parametrized cases (`suite` x2 throughout, `evolve` x2 on
one test), no explicit markers. The file's own recorded cost is **12 passed in
186.74s** (previous audit) and `tests/test_durations.json` puts the two evolved
`dy_central` cases at 87.5s/88.7s. **Neither is what it costs today.** Measured
2026-08-13: the 10 items that exclude `test_dy_central_folds_the_same_operator_the_dataset_carries`
run in 24s, while a single evolved case of that test exceeds **20 CPU-minutes**.
The cause is one call: `DYKernel.bilinear_tensor` on this file's 3-row, 6-element
fixture measures **0.58 s per call** and is not memoized in-process, and that test
needs ~2000 of them (1552 in its own reference loop, 424 inside `dy_central`) --
0.58 x 1976 = 19.5 min, which matches the observed wall time. The recorded 88s
implies ~0.045 s per call, so the DY hard-tensor assembly is ~13x slower than when
the durations were taken. That is a **source-side change, not a test change**:
`src/pixel/kernels/drell_yan/kernel.py` and `src/pixel/data/drell_yan.py` carry
uncommitted modifications timestamped the same day, and the DY kernel package has
newly added `nnlo_convolution.py`/`nnlo_source.py`/`_redy_bridge.cpp` (the closure
config runs `DY_ORDER = "LO"`, so the NNLO path should not be engaged). Flagged for
whoever owns that edit; nothing in this file changed the code being timed.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_dy_evolve_constant_reaches_the_built_operator` | `DY_EVOLVE` toggles evolution attachment and the nf=4->nf=5 parton set | real threshold logic + stub attachment (`F1`) | membership only | `S2` |
| 2 | `test_dy_central_folds_the_same_operator_the_dataset_carries` | `dy_central`'s grouped fold matches a one-contribution-at-a-time reference, evolved or not | both read the same contributions (`A3`) | rtol 1e-12 | `S2` |
| 3 | `test_unevolved_dy_central_reproduces_the_parton_precombined_fold` | field-decomposed fold equals the historical parton-precombined fold | distributivity check, shared `dy_field_maps` (`A3`) | rtol 1e-12 | `S2` |
| 4 | `test_evolved_dy_contributions_collapse_onto_shared_operator_groups` | shared-operator contributions collapse to **at most half** as many groups; evolution operators are row-wise | magnitude-aware count + shape (`F1`) | `len(groups) <= len(contributions)//2` (776); measured 424 of 1552 | `S3` |
| 5 | `test_assemble_composes_each_hadron_operator_onto_its_own_axis` | folding the *bare* tensor with separately evolved curves reproduces folding the *assembled* tensor -- so `E_A` is composed onto the A axis and `E_B` onto the B axis, `weight` included | the documented `q_phys = E @ q_input` contract, expressed as matrix-vector products rather than the source's einsum (`A3`) | rtol 1e-11 (measured 6.7e-16); swapped composition measured 2.821 away, asserted `> 0.1` | `S2` |
| 6 | `test_dy_projection_disagreeing_with_the_configured_nf_is_rejected` | a real nf=5 projection against a configured `DY_NF=4` raises | real guard, message match (`F1`) | exception + message substring | `S3` |
| 7 | `test_dy_side_a_is_the_beam_and_side_b_carries_the_reaction_transform` | side A of `dy_evolution_maps` is the beam proton and the `pd`/`ppbar` transform lands on side B alone | the function's own `pp` answer as the anchor, then `isoscalar_projection`/`conjugate_projection` of it (`A3`) | exact -- `np.array_equal` on operators, `==` on weights; three degeneracies asserted, not assumed | `S3` |

**Weak spots**

- RESOLVED 2026-08-13 (was: test 1's evolved-branch check used `any(...)` where its unevolved
  mirror used `all(...)`, so a regression dropping evolution from a strict subset of
  contributions passed). It is now `all(...)`, with `len(...) == 1552` asserted beside it so
  the quantifier cannot go vacuously true on an empty list. Re-measured for both suites: 1552
  contributions, every one wired on both sides.
- **W-COMMON** (S3, partly closed 2026-08-13, **fully RESOLVED 2026-08-14**) Tests 2 and 3
  still read `field_A`/`field_B` and the composed tensor off the *same*
  `dataset.bilinear_contributions` the code under test reads, and that caveat stands --
  it is correct for what they target, the fold arithmetic. The exposure is now covered by
  two tests instead, one per half. Test 5 (2026-08-13) closes the
  `assemble()`/`evolved_tensor` half: an axis transpose there fails it.
  `test_dy_side_a_is_the_beam_and_side_b_carries_the_reaction_transform` (2026-08-14)
  closes the **labelling** half, upstream in `dy_evolution_maps`, whose return
  `(proton.fields, target.fields, proton.evolution, target.evolution)` can have its two
  halves exchanged by a one-line edit that changes no shape and raises nothing. It anchors
  on the function's *own* `pp` answer -- for `pp` the target **is** the beam -- and then
  requires the `pd`/`ppbar` A-sides to reproduce that reference bit for bit while their
  B-sides reproduce `isoscalar_projection`/`conjugate_projection` of it, so nothing is
  re-derived and no kwarg list is retyped from `datasets.py`. Acceptance: the swap applied
  in memory to both suites' `datasets.dy_evolution_maps` gives **1 failed / 7 passed**, the
  failure being only the new test (66 structural differences, first `b/g: 4 terms vs 2`),
  and the plugin's call counter shows the swapped function was called **7 times** -- so the
  four pre-existing `DY_EVOLVE=True` tests executed the mutated code and passed anyway.
- **Three degeneracies had to be measured before that test could mean anything**, and the
  first is the sharpest: the `fields` maps -- the very objects the weakness names -- are
  **identical** across `pp`/`pd`/`ppbar` and across partons, because
  `PartonEvolutionProjection.fields` records only *which* source fields feed a parton, with
  coefficient 1.0, and neither the isoscalar average nor the charge conjugation changes
  that set (measured: `fields_A == fields_B == isoscalar(...).fields ==
  conjugate(...).fields`, and `fields['u'] == fields['d'] == fields['ub']`). All the
  orientation lives in `.evolution`. Second, `conjugate_projection` is an **involution**,
  so any A-vs-B relation such as `fields_B["u"] == fields_A["ub"]` is symmetric under the
  swap and passes either way. Third, `pp` cannot detect the swap at all; it is asserted
  symmetric rather than skipped. Note the shared `_stub_evolution` *is* sufficient for this
  test -- unlike for test 5 -- because what is compared is term structure, not a folded
  value: the swap shows as `u/t3: 1 term vs 2` (`pd`) and `u/v3: weight +0.25 vs -0.25`
  (`ppbar`). Comparisons are exact (`np.array_equal` on operators, `==` on weights); both
  sides come from one deterministic stub, so a tolerance could only hide a discrepancy.
- RESOLVED 2026-08-13 (was: test 4's bare strict inequality, which 1551 groups would have
  satisfied). The bar is now `len(groups) <= len(contributions) // 2` (776), against a
  measured 424 of 1552 (27.32%) in both suites -- 45% headroom, tight enough to fail a loss of
  most of the sharing and loose enough to survive a flavour or channel being added.
- Two degeneracies had to be removed before test 5 could fail at all, and both are worth not
  rediscovering: the reaction must be `pd` (so hadron B's projection is
  `isoscalar_projection(proton)` rather than the same object), and the evolution stub must not
  be proportional to the identity -- nor to any *single* fixed matrix, since all operators
  would still be proportional to each other. Both were measured at a swap ratio of exactly
  0.000 before the fixture was changed.
- The evolution operators are still stubs throughout, so nothing in this file constrains the
  *values* an operator takes -- only where it is composed and whether it is attached.

**Not covered here**

- ~~The `field_A`/`field_B` *labelling* in `build_drell_yan`/`dy_evolution_maps`~~ --
  covered 2026-08-14 by
  `test_dy_side_a_is_the_beam_and_side_b_carries_the_reaction_transform`, see the weak-spot
  entry above. What remains uncovered is the *other* consumer of those names: nothing here
  drives `cfg.dy_field_maps` (the `DY_EVOLVE=False` static coefficient table) for the same
  question, and `build_drell_yan`'s own `fields_A=`/`fields_B=` keyword pass-through into
  `DrellYan` is checked only through `dy_evolution_maps`' output. The sibling files still
  cannot help: `test_dy_channels.py`'s scalar-luminosity swap tests use a hadron-B fixture
  proportional to hadron-A and are documented there as unable to detect a swap for that reason
  (fix proposed as `test_dy_channels-M03`, not yet landed); `test_dy_kernel.py`'s one
  externally-oracled test runs at `n_elements=1`, too degenerate for a transpose to be visible.
- Whether the `_small` closure suites' DY truth central value is meaningful
  (`S0-05`, `plans/test_audit/S0_FINDINGS.md`) -- this file's `SUITES` tuple
  only names the two full suites and never imports the `_small` packages, so
  it neither exercises nor defends against that finding; see
  `tests/test_closure_truth_representable.py`, which documents the `_small`
  exclusion explicitly.
- The evolution operator's own numerical accuracy (contour convergence,
  Mellin-space Wilson coefficients) -- entirely stubbed out here by
  `_stub_evolution`/`_stub_evolution_flavour_mixing`; covered by
  `test_dy_flavor_evolution.py` and the fitpack DY comparison.

**Cost, and a contract discovered while reducing it (2026-08-14)**

The file went **2641s -> 265s** (10x): `SUITES` dropped to NNPDF only, and the reference loop
memoizes the kernel tensor. `[True]` 1186s -> 248s, `[False]` 196s -> 4.6s. It had been **43%
of the whole suite's node-time**.

Dropping JAM costs nothing measurable: `generate.dy_central` is **byte-identical** between the
two suites (`inspect.getsource`, 3225 chars), so that parametrization re-ran the same function.
A future divergence fails `test_closure_plot_scaling`'s cross-suite identity test instead.

**`BilinearContribution.assemble` returns the tensor ALREADY SCALED BY `weight`**, so
`(kernel, evolution_A, evolution_B)` — the triple `generate.dy_central` groups by — is **not**
a valid cache key for its output. That grouping is valid for `dy_central`, which handles each
contribution's weight inside the group; reusing it to cache `assemble` hands every contribution
whichever weight was computed first. Measured: the identity failed at `rtol=1e-12` with `got`
~`1e-13` against `expected` ~`1e-11`. The memo therefore caches the *unweighted* tensor and
applies the weight per call — correct, and a better cache, since the weight varies where the
tensor does not.

Two further traps hit on the way, both already documented in `FIX_AGENT_BRIEF.md` and both
walked into anyway: `BilinearContribution` is a **frozen dataclass**, so `self.weight = 1.0`
raises rather than applies (mode 0; `object.__setattr__` is the idiom), and while it was raising
the file "ran" in 12.63s — a 209x speed-up that was entirely tests failing instantly. **Check
the per-test durations, not just the summary line**: a suspiciously fast pass is a suspiciously
fast pass.

The memo is guarded so it cannot hollow the test out: `1 < len(cache) < len(contributions)`.
Without the lower bound, a future change collapsing every contribution onto one key would turn
the reference loop into a comparison of one tensor with itself, and the test would still pass.

**The memo was one level too shallow — re-keyed 2026-08-14 (30x)**

The reduction above left `[True]` at 248s (recorded 313s), still the file's whole cost and
still the tail xdist schedules the suite around. The residue was the memo's *key*, not the
memo: caching `BilinearContribution.assemble` on `(kernel, evolution_A, evolution_B)` protects
a **680-valued** thing, while the object that is actually expensive — the hard tensor from
`DYKernel._assemble_bilinear_tensor` — is pure in `(kernel, rows, bases)` and therefore only
**30-valued**. The 680 group tensors were 30 hard tensors rebuilt 650 extra times at ~0.59s
each.

`_memoized_assemble` is now `_memoized_hard_tensor`, hooked on the leaf builder and shared
across the module. Measured on this fixture (2224 contributions, 680 groups, 30 kernels):

| | before | after |
|---|---|---|
| `[True]` case | **296.2s** | **10.8s** |
| `[False]` case | 8.1s | 8.2s |
| `test_unevolved_dy_central_reproduces_the_parton_precombined_fold` | 15.1s | **0.01s** |
| `test_assemble_composes_each_hadron_operator_onto_its_own_axis` | 2.4s | **0.01s** |
| whole file (7 items) | ~322s | **19.7s** |
| folded result | — | **bit-identical** (`np.array_equal`, not a tolerance) |

**The 296.2s baseline is measured, not read off `test_durations.json`.** That file records
313.36s, taken when `build_drell_yan` emitted 1552 contributions and 424 groups against
today's 2224 and 680 — so quoting it would have been quoting a stale fixture. It was
re-measured against today's source by restoring the old `assemble`-level memo through a
plugin, which is also the cheapest way to time a reverted implementation without touching
the working tree.

**Module scope is where the last 15s came from, and it needed a value key.** Four tests in
this file build hard tensors; between them they request 78 and only **54 are distinct**.
`[False]` and `test_unevolved_dy_central_reproduces_the_parton_precombined_fold` run the
identical `(pd, DY_EVOLVE=False, DY_NF=4)` fixture and duplicate all 24 of theirs, and the
two the axis test needs are among the evolved 30. The evolved and unevolved sets do not
overlap (`nf` 5 vs 4 reaches `_row_couplings`), so 54 is the floor. Realising it meant two
changes beyond the re-key: `_basis()` became an `lru_cache`d singleton (kernel objects are
minted fresh per `build_drell_yan`, but so were bases, and an `id(basis)` key cannot hit
across tests otherwise), and the three other tensor-building tests install the memo too.
Note this is a *serial* win — under `-n` the cache is per-worker, so the file measures 19.6s
at `-n 3` against 19.0s serial.

Two reasons the hook is `_assemble_bilinear_tensor` and not `bilinear_tensor`. First, it is
the only one that works: `evolved_tensor` reaches the hard tensor through the *private*
`_cached_bilinear_tensor`, and calls `bilinear_tensor` **only on the unevolved branch**
(`kernel.py:601`) — a memo on the public method measured 0.592s/call on the evolved path,
i.e. no hit at all, and looks like a working cache. Second, hooking the leaf keeps
`cached_arrays` and `_tensor_metadata` inside the exercised path.

**Re-keying this low raised coverage rather than lowering it.** The `assemble`-level memo
also cached the *evolution composition*, so `evolved_tensor`'s
`einsum("ria,rij,rjb->rab", ...)` ran 680 times against 680 operator pairs. It now runs on
every reference assemble and every `dy_central` assemble — **2904 times**, each against that
contribution's actual operators. The only thing skipped is hard-tensor creation, which is
oracled independently and non-common-mode by `test_dy_kernel.py` (frozen fitpack benchmark),
`test_dy_dataset.py` (Altarelli–Ellis–Martinelli eq. 93 by quadrature) and
`test_dis_dy_closure.py` (`DYConvolution` continuum value).

**A module-scoped cache can be corrupt and PRESENT AS A SPEED-UP. This is the finding.**

Going module-scoped means the key must be complete, and the first attempt hand-wrote the
field list. Dropping one field from it — `nf`, which is 5 in the evolved fixture and 4 in the
unevolved one and reaches the tensor through `_row_couplings` — makes the two fixtures
collide, and **every assertion in the file still passed**. The reference loop and `dy_central`
both read the same poisoned tensor, so they agree exactly as before; the only visible effect
was the suite running in **10.5s against the honest 18.1s**. A corrupted cache looks like a
better cache. That is `CLAUDE.md`'s "agreement is not evidence" with a memo in the middle, and
it is why none of the three counting guards below is sufficient on its own — all three are
internal to the memo.

Two changes closed it, and both generalise to any value-keyed test cache:

1. **Enumerate the key, never transcribe it.** `_tensor_key` walks
   `dataclasses.fields(kernel)` rather than a hand-written tuple, so a field added to
   `DYKernel` later enters the key automatically and there is no transcription step to forget
   one. The hand-written list had in fact already missed `muR2`. Over-keying (`cache_path`,
   `parallel_workers`, the `nnlo_*` fields) is free — it costs time, never correctness.
2. **Detect collisions directly, independently of the key.** Each entry records `repr(kernel)`
   alongside its tensor, and a hit whose requesting kernel has a different repr raises.
   `DYKernel` is a dataclass with no `repr=False` field and no addresses in its repr (both
   checked), so this fires on exactly the failure a key can have — under-determination — at
   O(1) per call and without recomputing anything.

`_assert_hard_tensor_memo_is_faithful` rebuilds one tensor through the import-time-captured
unpatched builder and demands bit-identity. It is the only check that would survive the tensor
coming to depend on something that is not a field at all, but note what it cannot do: it
samples **one** kernel, and a collision only harms the kernels that were served someone
else's tensor. It passed the `nf` mutation. The collision detector is the guard that works.

The three counting guards remain, now measured over this fixture's own keys rather than
`len(cache)` (which is module-wide and would make the assertion order-dependent):
`len(fixture_keys) == len({id(c.kernel)})` — 30 == 30 evolved, 24 == 24 unevolved —
plus `fixture_keys <= set(cache)` and `1 < n_kernels < n_contributions`.

Verified by mutation rather than by passing, per the standing rule:

| mutation | result |
|---|---|
| `dy_central` group key drops `evolution_B` | **fails** the fold (`got` ~-9.3e-11 vs `expected` ~2.5e-12) |
| memo key drops the kernel entirely | **fails** `fixture_keys <= set(cache)` |
| `_tensor_key` drops `nf` (module-scope collision) | **fails** the collision detector, which names the `nf=4` / `nf=5` pair |
| unmutated | passes, margin 8.2e-14 against the 1e-12 bar |

One trap, and it is mode 3 of `FIX_AGENT_BRIEF.md`. The first attempt at the memo mutation
did `import tests.test_closure_dy_evolution` and reported **"2 passed"** — there is no
`tests/__init__.py`, so pytest imports the file as top-level `test_closure_dy_evolution` and
the plugin had patched a second, unused module object. A no-op'd mutation is indistinguishable
from a blind test. Reach the module through `item.module` in
`pytest_collection_modifyitems` and assert something was patched.

Noticed in passing, not acted on: `_assemble_bilinear_tensor`'s **parallel** path builds a
`settings` dict of the fields a worker needs (`kernel.py:441-460`) and omits `parity`, which
the serial path does pass to `_BasisPairConvolution`. If that is not deliberate, the two paths
disagree whenever `parity != "even"` and `workers > 1`. This fixture runs `parity="even"`
throughout, so nothing here would see it.
