### `tests/test_closure_pseudoitd_matching.py`

**Exercises** `src/pixel/kernels/matching.py (_pseudoitd_nlo_moments, matching_coefficients kind="pseudoitd")`, `src/pixel/kernels/pqcd/splitting.py (_psi, CF, ZETA2 -- used inside _pseudoitd_nlo_moments)`, `src/pixel/kernels/evolution/matrices.py (EvolvedPseudoITD, re-exported via pixel.kernels.evolution)`, `src/pixel/geometry/grid.py (Grid)`, `src/pixel/geometry/finite_elements.py (ProductBasis)`, `src/pixel/data/lattice/pseudoitd.py (PseudoITD.from_file, _kernel_for -- reached through build_pseudoitd)`, `src/pixel/data/base.py (field_name, systematic_contributions -- reached through PseudoITD.from_file)`, `closure_JAM_truth/config.py, closure_JAM_truth/datasets.py (build_pseudoitd)`, `closure_NNPDF_truth/config.py, closure_NNPDF_truth/datasets.py (build_pseudoitd, byte-identical to the JAM twin apart from one docstring line)`.

**Cost** 12 test functions, **13 pytest items**, measured `13 passed in 2.26 s`
(2026-08-14, serial; the `~21 s` recorded earlier was a cold-kernel-cache run). `SUITES`
and `ALL_SUITES` are NNPDF-only as of 2026-08-14, on the owner's instruction -- they were
`("closure_JAM_truth", "closure_NNPDF_truth")` and all four packages respectively, and the
file collected **18 items in 2.32 s**. The five removed items are the JAM legs of four
tests; none of those tests reads the truth PDF set, and no test in this file compares one
suite to another, so both tuples were trimmed in place rather than split.
`test_closure_layout_uses_vector_real_imaginary_c_parity` keeps both a full and a `_small`
case. Five of the twelve tests were added by the
concurrent NNLO session and are still carried in
`tests/test_explanation_coverage.py::_UNEXPLAINED_BACKLOG`; the two of those that weakness
`-03`'s resolution rests on are documented in the table below and were removed from that
backlog in the same pass, per its shrink-only rule.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_nlo_matching_moments_reduce_to_the_analytic_constant` | Computes _pseudoitd_nlo_moments(orders, lam=that lam) for N=2..5 and compares to a from-scratch harmonic-sum recomputation of L(N) (helper _harmonic, no call to the source's _psi). | digamma/trigamma-based B(N),L(N) formula (source, mpmath.fp.psi) vs explicit-summation harmonic numbers (test); real and imaginary parts checked separately; a magnitude floor guards against a vacuous zero target. (`A1`) | rtol=1e-12, atol=0.0 (real part); atol=1e-12 (imag part) | `S2` |
| 2 | `test_matching_log_carries_the_lambda_dependence` | Subtracts the log-cancelling reference (lam from test 1) from a second call at lam=1.7 and compares the difference to CF*log_factor*B(N), B recomputed via the same from-scratch harmonic-sum helper. | reference and got are both outputs of the function under test at two different lam; the lam-independent L(N) term cancels in the subtraction, isolating the log(lam)*B(N) piece against an independently recomputed B(N). (`A1`) | rtol=1e-12, no atol= -> np.allclose's default atol=1e-8 applies and dominates (measured this audit: atol=1e-8 exceeds the rtol contribution by ~3075x at these RHS magnitudes, \|rhs\| in [3.3, 6.4]); true effective bar is ~1e-8 absolute, not 1e-12 relative | `S2` |
| 3 | `test_pitd_evolve_constant_reaches_the_built_kernels` | Monkeypatches PITD_EVOLVE True then False and rebuilds the same lattice record both times; on True, checks isinstance(EvolvedPseudoITD) plus four attribute equalities against config constants; on False, checks NOT isinstance(EvolvedPseudoITD). | closure_JAM_truth/closure_NNPDF_truth's build_pseudoitd -> pixel.data.PseudoITD.from_file -> _kernel_for, driven by a real generated manifest record via _lattice_fixture. Parametrized over `SUITES`, which was both full-size suites until 2026-08-14 and is `closure_NNPDF_truth` alone since: their datasets.py are byte-identical and the four checked config constants (PITD_EVOLVE, PITD_MATCHING_ORDER, PITD_MATCHING_LAMBDA, Q0_2) match, so the JAM leg exercised identical code against a second generated manifest, not a second implementation. (`F1`) | exact isinstance/equality checks; pytest.approx (default rel) on two O(1) float constants | `S3` |
| 4 | `test_systematic_nuisances_stay_on_the_bare_transform` | Builds the dataset with PITD_EVOLVE=True, splits contributions[0] (physical) from the rest, checks the physical kernel IS EvolvedPseudoITD and every nuisance kernel is NOT, with a non-None coefficient. | Same build_pseudoitd path; PseudoITD.from_file routes systematics through _kernel_for(..., coefficient=coeff) with no mu0_2/evolution_lambda, so the split is structural in the source, not merely conventional. (`F1`) | exact isinstance / is-not-None checks | `S3` |
| 5 | `test_matching_changes_the_operator_it_is_applied_to` | Builds bare (matching=False) and matched (matching="NLO") EvolvedPseudoITD kernels, otherwise identical, and requires their .matrix() outputs to differ by more than 1e-6*scale. | Direct kernel construction on a 9-point grid + .matrix() call; pure nonzero-difference structural check, no comparison to any expected value. (`F1`) | diff.max() > 1e-6 * scale, scale = max\|bare_matrix\| | `S3` |
| 6 | `test_li_vector_density_reduces_to_existing_nlo_mellin_coefficient` | Li's normalized one-loop QCF plus-density moment equals `-_pseudoitd_nlo_moments(N, lam=1.37)` for N=1..5. | Two independent representations of one constant: a z-space HPL density integrated numerically (translated from Li-Ma-Qiu, arXiv:2401.16281) versus the closed-form complex-N digamma expression written years earlier. Neither is derived from the other. (`A3`) | rtol=0.0, atol=5e-12 | `S1` |
| 7 | `test_li_vector_formula_matches_authors_ancillary_iA_benchmark` | The translated temporal-vector QCF densities reproduce the authors' own Mathematica output for `iA(omega=29.15)`, nf=3, at one and two loops (two NLO and three NNLO `alpha_s^n L^k` coefficients). | External oracle: literals copied from `paper_repo/arxiv_sources/Li-Ma-Qiu_.../attachedfiels/readme.nb`, `Out[16]` (file verified present on disk 2026-08-13). Exercises every positive- and negative-support function *before* the reduced-ratio normalization, so an error cannot cancel in a shared quotient. (`D1`) | rtol=0.0, atol=5e-10 (NLO) / 5e-9 (NNLO) | `S1` |
| 8 | `test_generation_time_build_matches_the_fitting_time_forward_operator` | `build_pseudoitd(..., with_systematics=False)` changes only the nuisance list: 1 contribution vs 1+5, same `EvolvedPseudoITD`, equal `_cache_settings()`, array-equal `mean`/`cov`/`nu`. | `A3`, the same manifest record built both ways. `_cache_settings()` is the dict the code itself uses for kernel cache identity, so agreement on it *is* operator identity by the code's own definition. | dict equality; `assert_array_equal` on the data | `S3` |
| 9 | `test_evolved_pseudoitd_refuses_a_grid_with_a_node_at_the_origin` | `base_matrix` raises `ValueError` matching "node at x=0" on a basis whose first node is exactly `0.0`, and the origin node is demonstrably the whole cause. | `F1`, matched on the **message** not the type (this method raises `ValueError` from four guards). The control basis differs in exactly one node (asserted `array_equal` after the origin) and is run through to a finite, nonzero matrix. | exception type + message substring; control asserts shape, finiteness, nonzero max | `S3` |

Rows 6 and 7 were written by the concurrent NNLO session; they are documented here because
weakness `-03`'s resolution rests on them, and their independence from each other was measured
by mutation in that pass (see the weak spot below). Three further tests from that session --
the closure-layout C-parity check, the normalized-NNLO-ratio check, and the reduced-ratio
convention guard -- are still unexplained and remain listed by name in
`tests/test_explanation_coverage.py::_UNEXPLAINED_BACKLOG`. They are referred to
descriptively rather than by name here on purpose: that backlog's staleness guard treats any
occurrence of a listed name inside this entry as "now explained", so naming them without
writing their rows would break it.

**Weak spots**

- **W-NAME** (S2) **RESOLVED 2026-08-13** by `test_pitd_evolve_constant_reaches_the_built_kernels`.
  The `PITD_EVOLVE=False` branch now asserts the concrete class the record's `component` calls
  for -- `pixel.kernels.PseudoITDReal` is `Cosine` and `PseudoITDImag` is `Sine`, checked this
  pass to be distinct classes with neither a subclass of the other -- plus the negative
  assertion on the *other* parity and the `alpha` that `momentum_density` selects (compared
  against the evolved branch's `alpha`, not a literal). *Acceptance:* an in-memory `_kernel_for`
  wrapper that flips `component` on the non-evolved path only fails both items of the hardened
  test, while the pre-fix assertions run verbatim standalone under the same mutation all
  **pass** -- the original finding reproduced. Original text: tests/test_closure_pseudoitd_matching.py:224 `assert not isinstance(plain.contributions[0].kernel, EvolvedPseudoITD)` is the entire PITD_EVOLVE=False check -- no assertion of what the kernel actually is. — the test's docstring claims the flag changes the operator 'alike' on both settings; a real/imag mixup, or any kernel type other than the expected PseudoITDReal/PseudoITDImag, would also satisfy this negative-only assertion on the False branch. *Fix:* also assert isinstance(plain.contributions[0].kernel, kernels.PseudoITDReal) (or the component-appropriate concrete type). See `plans/test_suite_hardening.md#test_closure_pseudoitd_matching-01`.
- **W-NAME** (S2) **RESOLVED 2026-08-13** by `test_pitd_evolve_constant_reaches_the_built_kernels`:
  `assert kernel.order == config.ORDER` and `assert kernel.mode == config.MODE`, closing the
  last two keys of `build_pseudoitd`'s evolution dict. *Acceptance:* `_kernel_for` wrappers
  forcing `order="LO"` and `mode="iterated"` each fail both items of the hardened test, and the
  pre-fix assertions run standalone **pass** under both. Original text: closure_JAM_truth/datasets.py:108-116 builds the evolution dict with five keys (mu0_2, evolution_lambda, order, mode, matching); tests/test_closure_pseudoitd_matching.py:211-215 checks only matching/matching_order/mu0_2/lam. config.ORDER and config.MODE (-> EvolvedPseudoITD.order/.mode, src/pixel/kernels/evolution/matrices.py:1259,1261) are never compared. — a build_pseudoitd edit that hardcoded evolution order or mode (e.g. always "LO"/"truncated" regardless of config) would pass every assertion in this test. *Fix:* add assert kernel.order == config.ORDER and assert kernel.mode == config.MODE. See `plans/test_suite_hardening.md#test_closure_pseudoitd_matching-02`.
- **W-TRANS** (S3) **RESOLVED 2026-08-13** -- not by an edit here, but by two tests the
  concurrent NNLO session added to this file, whose combination this audit predates. The claim
  "there is currently no independent physics oracle for this formula anywhere in the
  repository" is now **false**. The chain is
  `test_li_vector_formula_matches_authors_ancillary_iA_benchmark` (literals copied from the
  authors' own Mathematica ancillary `readme.nb`, `Out[16]`) -> the `p10`/`p11` densities ->
  `p1 = CF*(p10 + ell*p11)` in `li_vector_densities` ->
  `test_li_vector_density_reduces_to_existing_nlo_mellin_coefficient` ->
  `_pseudoitd_nlo_moments`. *Acceptance,* both links measured by in-memory mutation: corrupting
  the `L(N)` constant inside `_pseudoitd_nlo_moments` fails the bridge test **and** the
  harmonic-sum test while the ancillary benchmark keeps passing (so the benchmark is
  independent of the function under test); scaling the Li `p10` density by `1.000001` fails the
  ancillary benchmark **and** the bridge test while the harmonic-sum tests keep passing (so the
  bridge is what transfers the anchor). **Still open, and out of scope for a test file:**
  `matching.py:382-414` carries no literature citation, unlike its neighbour
  `_pitd_gluon_finite_moments` -- that is a `src/` docstring edit. Original text: src/pixel/kernels/matching.py:382-414 (_pseudoitd_nlo_moments) carries no literature citation, unlike its neighbor _pitd_gluon_finite_moments (matching.py:421-459), which cites Balitsky-Morris-Radyushkin PRD 105, 014008 (2022) Eq. (4.13) by equation number. Checked this audit: neither fitpack_legacy nor pdf_fitter implement a pseudo-ITD matching kernel (grep for pseudoitd/pseudo_itd/reduced_itd in both repos, 0 hits each). — both tests recompute the identical B(N)/L(N) algebraic combination via harmonic sums, independent of the source's digamma calls but not of the underlying formula; a transcription error in the physics content (as opposed to the digamma bookkeeping) would reproduce identically in the harmonic-sum recomputation and pass both tests. There is currently no independent physics oracle for this formula anywhere in the repository. *Fix:* cite the source paper/equation for the non-singlet pseudo-ITD one-loop matching constant in matching.py's docstring (out of scope for this file); once a citation exists, add a value copied directly from the paper as a third check. See `plans/test_suite_hardening.md#test_closure_pseudoitd_matching-03`.
- **W-LOOSE** (S3) **RESOLVED 2026-08-13** by adding the explicit `atol=0.0`, so the coded
  `rtol=1e-12` is now the operative bar rather than numpy's default `atol=1e-8`. Measured with
  the tightening in place: `|rhs| = 3.25, 5.08, 6.38` and `max abs(lhs/rhs - 1) = 2.22e-16`
  (one ULP), so the real bar clears by four orders and the change masks nothing. The
  docstring's prose describing the looser *effective* bar was replaced rather than left to
  contradict the code. Original text: tests/test_closure_pseudoitd_matching.py:178 `np.allclose((got - reference).real, cf * log_factor * b, rtol=1e-12)` has no atol=, so numpy's default atol=1e-8 applies. Measured this audit: for orders=[2,3,4], lam=1.7, \|cf*log_factor*b\| is in [3.3, 6.4], so the rtol=1e-12 term contributes only ~3.3-6.4e-12 while atol=1e-8 dominates by a factor of ~3075. — a reader would take 'rtol=1e-12' at face value as a 12-significant-figure bar; the true effective bar is ~1e-8 absolute (~3000x looser). Not currently masking anything -- the achieved agreement is 2.22e-16, four orders inside even the true floor -- but the discriminating power is far weaker than the literal parameter suggests. *Fix:* add atol=0.0 explicitly to state the intended pure-relative bar. See `plans/test_suite_hardening.md#test_closure_pseudoitd_matching-04`.
- **W-LOOSE** (S3) tests/test_closure_pseudoitd_matching.py:301 `assert np.abs(matched_matrix - bare_matrix).max() > 1e-6 * scale`. Measured this audit by reproducing the test's construction directly: diff.max()=0.483, scale=0.686, diff/scale=0.704 -- six orders above the 1e-6 bar. — this test cannot distinguish correctly-wired matching from matching wired at a wildly wrong magnitude; it only catches matching being completely inert. Mitigated: tests/test_evolution.py::test_evolved_pseudoitd_matching_composes_after_evolution independently checks the exact composition to rtol=1e-11, and this file's docstring now cross-references it. *Fix:* none required given the cross-referenced tight check exists elsewhere; documented this pass. See `plans/test_suite_hardening.md#test_closure_pseudoitd_matching-05`.
- **W-ORDER** (S4) **RESOLVED 2026-08-13** by `test_systematic_nuisances_stay_on_the_bare_transform`.
  The positional split is replaced by a lookup keyed on `Contribution.field`, compared against
  `str()` of the exact stand-ins `_lattice_fixture` handed in (self-consistent rather than a
  hardcoded repr, since `field_name` falls back to `str()` for `SimpleNamespace` stand-ins).
  The check is set **equality**, so it also pins that every systematic key in the manifest
  record reached the dataset and none was invented; `contributions[0]` being the physical field
  is still asserted, but as a separate, clearly-failing claim rather than an unstated
  assumption. *Acceptance:* an in-memory `systematic_contributions` returning one fewer entry
  fails the hardened test (and the new `with_systematics` test), while the pre-fix assertions
  run standalone **pass** under the same mutation. Original text: tests/test_closure_pseudoitd_matching.py:259 `physical, *nuisances = dataset.contributions` relies on src/pixel/data/lattice/pseudoitd.py:441-468 always list-constructing the main Contribution first, with no assertion (e.g. on contribution.field) confirming that positionally here. — if PseudoITD.from_file's contribution order ever changed, this test would fail confusingly (isinstance checks on the wrong contribution) rather than clearly; low risk in practice since the source's list-literal construction makes the order structurally obvious to a reader of that file. *Fix:* match by contribution.field instead of position, or leave as is given the low practical risk. See `plans/test_suite_hardening.md#test_closure_pseudoitd_matching-06`.
- **W-DEAD** (S4) tests/test_closure_pseudoitd_matching.py:319,330 (_lattice_fixture) pytest.skip on a missing manifest or a manifest with no systematic-bearing pseudo-ITD record; used by all 4 parametrized cases of these two tests. Checked this audit: both closure_JAM_truth/data/truthQ_2/manifest.json and closure_NNPDF_truth/data/truthQ_2/manifest.json currently carry 117 qualifying records each, so no case is currently skipped. — a generate.py regression that stopped attaching systematics to the pseudo-ITD records, or a checkout without generated truth, would silently skip these 4 cases rather than fail them; per this repo's own tests/README.md rule, a skip is easy to miss in a summary line the way a hard failure is not. *Fix:* none required now; the skip conditions and current non-triggering are documented in _lattice_fixture's docstring this pass. See `plans/test_suite_hardening.md#test_closure_pseudoitd_matching-07`.

- **RESOLVED 2026-08-13, documentation-only** (`test_closure_pseudoitd_matching-05`) — `test_matching_changes_the_operator_it_is_applied_to`'s `> 1e-6 * scale` bar sits six orders below the measured `diff/scale = 0.704`, so it only catches matching being wholly inert. Verified this pass that both `:38-41` and `:639-640` name `tests/test_evolution.py::test_evolved_pseudoitd_matching_composes_after_evolution` (`rtol=1e-11`) as the tight check this defers to.
- **RESOLVED 2026-08-13, documentation-only** (`test_closure_pseudoitd_matching-07`) — `_lattice_fixture`'s two `pytest.skip`s (`:752`, `:763`) would silently drop 4 parametrized cases on a checkout without generated truth. Verified this pass that `:743` documents both conditions and records that both manifests currently carry 117 qualifying records, so nothing skips today.

**Not covered here**

- [lattice-pitd] **RESOLVED 2026-08-13** by
  `test_evolved_pseudoitd_refuses_a_grid_with_a_node_at_the_origin`, written in this file
  rather than `tests/test_coverage_edges.py` (out of the batch's edit scope). The guard has
  also moved: it is now `matrices.py:1545-1558`, not `:1463-1476`. Original proposal: Exercise EvolvedPseudoITD's grid-node-at-x=0 guard — src/pixel/kernels/evolution/matrices.py:1463-1476 raises ValueError when the basis's first node is exactly 0.0, because EvolvedPseudoITD needs a square grid->grid evolution operator (unlike plain evolution_matrix, which tolerates a non-square (n-1,n) result for an origin node per tests/test_evolution.py:914-925). Line 1471 is proven never executed by any test in the suite (coverage_index.txt: src/pixel/kernels/evolution/matrices.py uncovered lines 1471,1661,1663,1677). (proposed oracle `F1`)
- [kernels-parallel] **OBSOLETE 2026-08-13, deliberately not written.** The proposal targets a
  matching branch inside `_evolved_pseudoitd_scale_task` that **no longer exists**: the
  concurrent NNLO refactor moved projection-dependent Li matching into the parent process
  (`matrices.py:1763-1783`, whose docstring now says so), and the worker returns only the
  evolution matrix. Both the serial and parallel branches then run the identical line
  `out[local_rows] = projected_fourier(local_rows, scale) @ np.asarray(T)`, so matching sits
  *outside* the parallel/serial fork and a `matching=True` identity test would exercise
  strictly less per second than the existing
  `tests/test_evolution.py::test_parallel_rowwise_evolution_wrappers_match_serial`, which
  already covers that fork. Two things worth passing on: the worker's `settings` dict still
  carries `matching`/`matching_order`/`pseudoitd_*`/`lam`/`alphas` keys that the worker never
  reads (dead payload -- harmless, but a reader will assume the worker matches); and the
  `coverage_index` claim below about lines 1661/1663/1677 refers to code that is gone.
  Original proposal: Exercise EvolvedPseudoITD's parallel worker path with matching enabled — _evolved_pseudoitd_scale_task's matching branch (src/pixel/kernels/evolution/matrices.py:1660-1677) is reached only when workers>1 and more than one distinct mu2 is requested. tests/test_evolution.py:1337-1394 (test_parallel_rowwise_evolution_wrappers_match_serial) exercises the parallel path for EvolvedPseudoITD with 3 distinct z (hence 3 distinct mu2) but leaves matching at its default False, so the matching branch inside the worker is never entered; tests/test_kernel_parallel.py's dedicated worker-invariance tests (lines 474-546) cover VariableQ2MatchingMatrix/VariableQ2NonSingletEvolutionMatrix/VariableQ2SingletEvolutionMatrix but never construct EvolvedPseudoITD at all. Coverage confirms lines 1661/1663/1677 (inside the matching branch) are never executed by any test in the suite. (proposed oracle `A3`)
- [closure-suite] **RESOLVED 2026-08-13** by
  `test_generation_time_build_matches_the_fitting_time_forward_operator` (both full suites).
  Original proposal: Exercise build_pseudoitd(..., with_systematics=False), the generation-time invocation — closure_JAM_truth/datasets.py's module docstring (lines 1-22) states generation builds physical-only datasets (with_systematics=False) and injects systematics separately, while fitting uses with_systematics=True; both must produce the identical forward operator for the closure to be meaningful. Every call in this file (e.g. tests/test_closure_pseudoitd_matching.py line 205's datasets.build_pseudoitd(record, path, fields, None)) leaves with_systematics at its True default. Reading the source shows the evolution dict construction (closure_JAM_truth/datasets.py:108-116) is unconditional on with_systematics, so no coupling bug currently exists, but that decoupling is not verified by any test. (proposed oracle `F1`)

<!-- entry regenerated from reports/test_closure_pseudoitd_matching.json by entry_from_report.py -->

**Phase-2 missing-test pass, 2026-08-14 — nothing added, and why.**

`test_closure_pseudoitd_matching-M02` ("exercise `EvolvedPseudoITD`'s parallel worker path
with matching enabled") is **OBSOLETE**, re-verified against the current source rather than
inherited from its 2026-08-13 note. `_evolved_pseudoitd_scale_task`
(`src/pixel/kernels/evolution/matrices.py`) has no matching branch at all: its docstring now
states that projection-dependent Li matching is assembled in the parent process and that the
worker returns only the evolution matrix, and its body calls `evolution_matrix` and returns
`(scale, T)`. The parallel arm (`matrices.py:1665`) and the serial arm (`:1693`) then run the
byte-identical line `out[local_rows] = projected_fourier(local_rows, scale) @ np.asarray(T)`,
so matching sits **outside** the parallel/serial fork entirely — a `matching=True`
serial-vs-parallel identity test would exercise strictly less per second than
`tests/test_evolution.py::test_parallel_rowwise_evolution_wrappers_match_serial`, which
already covers that fork. The line numbers the item cites for the "uncovered" matching branch
(1661/1663/1677) refer to code that no longer exists. `-M01` and `-M03` were already resolved
on 2026-08-13; this file has no open missing-test items.
