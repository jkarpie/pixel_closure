### `tests/test_momentum_sum_rule.py`

**Exercises** `pixel.kernels.pqcd.evolution.NonSingletEvolution`/`SingletEvolution` (via
`pixel.kernels.evolution.evolution_factors`/`singlet_evolution_factors` in Mellin space and
`evolution_matrix`/`singlet_evolution_matrix` in x space), `pixel.kernels.pqcd.splitting`'s
`lo_singlet`/`nlo_singlet`/`lo_nonsinglet`/`nlo_nonsinglet`, the VFNS threshold-matching
operators in `pixel.kernels.pqcd.heavy_threshold`, `pixel.kernels.evolution.contour.adaptive_contour`,
and `pixel.geometry.finite_elements.CubicSplineBasis`. Two of its six tests (see table) exercise
only `closure_JAM_truth_small.pdf_guidance.dump_jam`, not `src/pixel`.

**Claim** PIXEL's evolution operator conserves total quark+gluon momentum and per-flavor valence
number, both in Mellin space at `N=2`/`N=1` and after a full x-space reconstruction, when evolving
a realistic external PDF (JAM24) rather than a toy input. The file runs exclusively at
`cfg.ORDER = "NLO"` (`closure_JAM_truth_small/config.py:190`) — it never varies order, so it
probes neither LO nor NNLO. At NLO, PIXEL's singlet/non-singlet splitting functions are direct
closed-form Mellin expressions (`src/pixel/kernels/pqcd/splitting.py:27-28`), not the
Moch-Vermaseren-Vogt compact parametrization used only at NNLO — so **this file is not, and
cannot be, the NNLO-fit-floor witness** that CLAUDE.md's evolution section and
`plans/test_audit/pqcd_audit_order.md` describe. That witness is
`tests/test_nnlo_splitting.py::test_momentum_sum_rule_measures_the_parametrization_accuracy`,
cross-referenced directly from `src/pixel/kernels/pqcd/nnlo_splitting.py:69`, and it probes a
different, much larger quantity (the bare NNLO matrix's own ~2e-4/~3e-5 parametrization error at
one Mellin point) than anything measured here (an NLO floating-point floor around 1e-8, five
orders tighter). The LO/NLO order-dependence of the same *kind* of check, with a toy input, lives
in `tests/test_evolution.py::test_singlet_pdf_mellin_moments_satisfy_momentum_sum_rule_under_evolution`
and `::test_valence_first_moment_conserved`.

**Oracles** The two `jam24_..._at_input_scale` tests check only the JAM24 LHAPDF dump and its
reading pipeline (`closure_JAM_truth_small.pdf_guidance.dump_jam`) against the exact physical
targets `1.0` (momentum) and `(2,1,0,0)` (valence) — independent of `src/pixel`, and exist to
gate that the realistic input the PIXEL tests below use is itself sane, not to test evolution.
The four `pixel_...`/`xspace_...` tests check momentum/quark-number conservation, an exact QCD
identity independent of alpha_s, order, or any external fixture (`A2`) — genuinely independent
of the evolution code under test, and traced (not assumed) to be non-tautological: `lo_singlet`/
`nlo_singlet` build `qq`, `qg`, `gq`, `gg` as four independently-coded closed-form Mellin
expressions, so N=2 column-sum conservation is an emergent cancellation across them, confirmed by
an in-memory coefficient-perturbation test (see Weak spots). The two `xspace_...` tests
additionally route through `singlet_evolution_matrix`/`non_singlet_evolution_matrix`, which call
the identical `SingletEvolution.operator`/`NonSingletEvolution.sigma` used by the Mellin-space
tests (confirmed by reading `assembly.py`) — common-mode with those tests on the
splitting-function physics; their genuinely new territory is the finite-element basis, adaptive
contour, and low-x-completion pipeline, whose dominant error (low-x completion, not contour
truncation) the file's own comments already document as refined-and-converged.

**Cost** 6 test functions, 20 parametrized cases across four of them (5 target scales each: `mc`,
4, 5, 10, 100 GeV) plus 2 unparametrized fixture-sanity checks. 2 of the 20 x-space-momentum cases
(`mc`, `5gev`) are hand-marked `pytest.mark.slow` directly in the file's own parametrize
decorator — the finest contour panel and a threshold crossing respectively; the file's own comment
records ~38-47s for the Q=10 GeV x-space singlet case. All tests skip cleanly if the JAM24 LHAPDF
prefix is absent.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_jam24_momentum_sum_rule_at_input_scale` | JAM24's own momentum sum at Q=2 GeV is 1.0 | JAM24 LHAPDF dump + quadrature, no pixel code (`A1`) | abs 1.0e-4 | `S1` |
| 2 | `test_pixel_singlet_evolution_preserves_jam24_momentum_sum_rule` | N=2 singlet operator columns sum to 1; evolved JAM24 momentum total is preserved; **(added)** evolved quark-singlet exceeds evolved gluon | `SingletEvolution.operator`, NLO only (`A2`) | atol 5.0e-7; **(added)** strict `>` | `S1` (was `W2`) |
| 3 | `test_jam24_valence_flavor_sum_rules_at_input_scale` | JAM24's own valence counts at Q=2 GeV are (2,1,0,0) | JAM24 LHAPDF dump + quadrature, no pixel code (`A1`) | atol 5.0e-3 | `S1` |
| 4 | `test_pixel_non_singlet_evolution_preserves_jam24_valence_sum_rules` | N=1 non-singlet factor is 1; evolved valence counts preserved, elementwise | `NonSingletEvolution.sigma`, NLO only (`A2`) | atol 5.0e-7 / 5.0e-3 | `S1` |
| 5 | `test_xspace_reconstruction_preserves_momentum_sum_rule` | x-space-reconstructed momentum matches the Mellin prediction and 1.0 | `singlet_evolution_matrix`, shares the operator with test 2 (`A3`) | abs 1.2e-2 | `W2` |
| 6 | `test_xspace_reconstruction_preserves_valence_sum_rules` | x-space-reconstructed valence counts match the Mellin prediction and (2,1,0,0), elementwise | `non_singlet_evolution_matrix`, shares the operator with test 4 (`A3`) | atol 1.2e-2 | `S2` |
| 7 | `test_pixel_singlet_evolution_flows_to_the_asymptotic_momentum_partition` | the gluon **and** singlet momentum fractions each approach `16/(16+3nf)` / `3nf/(16+3nf)` — monotonically, from below, never crossing | zero eigenvector of the LO `N=2` anomalous-dimension matrix, `(3nf, 16)`; textbook, per-component (`A2`) | strict monotone + gap `< 3e-2` at `Q=1e8`; **achieved 1.95e-2** | `S1` |
| 8 | `test_conservation_atol_catches_a_realistic_splitting_coefficient_error` | `PIXEL_CONSERVATION_ATOL` catches a `1e-5` relative `pgq` error and does *not* fire on a `1e-7` one or on clean code | in-memory perturbation of `lo_singlet`, read out by the exact `N=2` column-sum identity (`A2`) | two-sided; **achieved 1.82e-06 caught / 1.62e-08 missed / 1.95e-09 clean against 5e-7** | `S1` |
| 9 | `test_xspace_reconstruction_preserves_sum_rules_at_lo` (5 scales) | the same x-space reconstruction at `order="LO"`, where the perturbative residual is identically zero | LO Mellin identity exact to 2.2e-16, so the residual is basis/contour/low-x only (`A2`) | abs 1.5e-3 / atol 4.0e-3; **achieved 4.8e-4 / 1.44e-3** | `S1` |

**Weak spots**

- **Missing tests `M01`, `M02`, `M03` all CLOSED 2026-08-14 by tests 9, 8 and 7.** Each is accepted
  by a mutation that **only** it catches:
  - test 7 (`M03`): a plugin exchanging `Q2` and `Q20` inside `singlet_evolution_factors` — evolution
    running *down* in scale — fails only test 7; the other 13 non-x-space tests pass, because momentum
    conservation is scale-pair-agnostic (column sums stay 1 to `8.5e-08`) and quark still exceeds gluon.
    That bug was invisible to the entire file. The proposed `D1` golden pin was **withdrawn**: no source
    in this tree gives JAM24's individual momentum fractions independently of `dump_jam`, so a pin would
    have been `F2` precisely where the item asked for independence.
  - test 8 (`M02`): a plugin normalizing each `N=2` operator column to sum to 1 — a "helpful" fix-up —
    fails only test 8, while the three conservation assertions in test 2 pass **vacuously**. Two traps
    are recorded in the test's own docstring: the Mellin moments sit on the *trailing* axis (so
    `P[..., 1, 0]` perturbs `pqg`/`pgg`, not `pgq`), and `_SPLITTING_CACHE` is not keyed on the
    splitting functions, so without clearing it every perturbation is a silent no-op — measured, all
    four sizes returned the clean defect bit-for-bit.
  - test 9 (`M01`): a plugin putting a `2e-3` relative normalization error on the assembled singlet
    x-space matrix fails all five new LO cases while the five NLO cases at `1.2e-2` pass.
- **`XSPACE_CONTOUR_TOL`'s comment is wrong at `Q = 100` GeV, measured three ways** (2026-08-14). It
  reads the `1.0733e-2` NLO residual as "the finite-element/JAM24 low-x floor, not contour error"; it is
  neither. (i) LO and NLO agree to ~15% at `Q <= 10` GeV but differ by **25x** at 100 GeV
  (`4.28e-04` vs `1.08e-02`) on the identical pipeline. (ii) Refining `x_min` — the parameter under
  suspicion — via `XSPACE_GRID_N` 161/201/241/281 converges cleanly at `Q=10` GeV
  (`1.34e-03 -> 1.47e-05`, LO) but **plateaus** at 100 GeV (`2.21e-02 -> 4.28e-04 -> 2.80e-04 ->
  7.36e-04` LO; `4.70e+00 -> 1.08e-02 -> 6.61e-03 -> 6.20e-03` NLO). (iii) Tightening the contour
  `1e-2 -> 3e-3` there moves it by **exactly 0** (`1.0763e-02` both). So that residual is an
  unexplained, order-dependent plateau, not a converged floor. Nothing asserts on it; recorded so the
  next reader does not mistake a plateau for convergence.
- **Also measured, and currently untested:** forcing `low_x_extension="flat"` on both x-space assembly
  entry points changes **nothing** in this file at either order — at `x_min = 1.24e-06` the head is
  inert for these unweighted integrals. The `power`/`alpha=1` keyword in
  `_xspace_reconstructed_sum_rules` is therefore not exercised by any assertion here.

- RESOLVED 2026-08-13 for test 2 (`test_pixel_singlet_evolution_preserves_jam24_momentum_sum_rule`)
  by an added order-sensitive assertion; test 5
  (`test_xspace_reconstruction_preserves_momentum_sum_rule`) still shares the gap, unfixed (effort M,
  see `M03` below). Tests 2 and 5 reduce every assertion to `.sum(axis=0)` or `.sum()` of a
  2-component (quark-singlet, gluon) vector, so neither can detect a `quark_singlet<->gluon`
  component swap in the JAM24 reading. Demonstrated directly (no edits to `src/`): evaluating the
  same NLO evolution operator against the real JAM24 `input_momentum = [0.5763, 0.4237]` and its
  reversed pair gives evolved-total `0.99996126...` vs `0.99996126...` differing by only `5.5e-9`
  — 91x under `PIXEL_CONSERVATION_ATOL=5e-7` and over 18,000x under `JAM24_NORMALIZATION_ATOL=1e-4`.
  The fix for test 2 adds `assert evolved_momentum[0] > evolved_momentum[1]`: quark-singlet exceeds
  gluon at every one of the 5 target scales, margin shrinking from `0.204` (charm threshold) to
  `0.053` (100 GeV) but never crossing — still ~1e5x `PIXEL_CONSERVATION_ATOL`. A column swap flips
  the inequality at every scale (measured: `[0.418, 0.582]` at the charm threshold, `[0.461, 0.539]`
  at 100 GeV). Acceptance: a plugin swapping `_jam24_input_momentum()`'s two return entries (via
  `pytest_collection_modifyitems`, since the test module is imported by pytest's rootdir mechanism
  rather than as an installed package) makes exactly the 5 parametrized cases of test 2 fail
  (`evolved_momentum[0] > evolved_momentum[1]` becomes false at every scale, e.g.
  `0.4606 > 0.5394` at `Q=mc`), while all other 17 tests in the file — including the two
  `..._at_input_scale` fixture gates and all four valence-sum-rule tests — pass unaffected; before
  the fix, all 6 selected cases (test 1 and the 5 parametrized cases of test 2) passed under this
  mutation. The sibling valence tests (3, 4, 6) never shared this gap: their assertions are
  elementwise over four flavors against the asymmetric target `(2,1,0,0)`, so a `u<->d` swap fails
  immediately. See `plans/test_suite_hardening.md#test_momentum_sum_rule-01`.

**Not covered here**

- Order-parametrized momentum/valence conservation with a toy input, both Mellin-space and at LO
  (exact to `1e-12`) vs NLO (floating-point floor `~1e-9`): `tests/test_evolution.py::
  test_lo_singlet_momentum_columns_conserve_n2`, `::test_singlet_pdf_mellin_moments_satisfy_momentum_sum_rule_under_evolution`,
  `::test_valence_first_moment_conserved`.
- The actual NNLO-parametrization-error witness (bare N=2 splitting matrix trace, ~2e-4 quark /
  ~3e-5 gluon): `tests/test_nnlo_splitting.py::test_momentum_sum_rule_measures_the_parametrization_accuracy`.
- Momentum/number conservation of the VFNS threshold-matching operators in isolation (as opposed
  to composed with segment evolution, which tests 2/4/5/6 here do at 4 of their 5 target scales):
  `tests/test_nnlo_heavy_threshold.py::test_threshold_omes_obey_number_and_momentum_sum_rules`.
- An LO case for the x-space reconstruction tests (5, 6), which would isolate the basis/contour/
  low-x-completion error from the NLO floating-point floor — not found anywhere in the suite.
- A permanent regression guard tying `PIXEL_CONSERVATION_ATOL`'s catch/miss boundary
  (`>=1e-4` relative coefficient error caught, `<1e-6` missed, both measured here) to an explicit
  assertion, rather than leaving it implicit in this audit's measurement.
- Test 5 (`test_xspace_reconstruction_preserves_momentum_sum_rule`) still has the sum-only
  component-swap blind spot fixed for test 2 above — an order-sensitive assertion there was left
  open (see `M03`), since the x-space value differs from the Mellin-space one (low-x completion)
  and would need its own margin measurement, not a reuse of test 2's numbers.
- An individual (non-summed) check of JAM24's quark-singlet vs. gluon momentum fraction against an
  independently published figure, which would close the swap blind spot structurally rather than
  by an order-sensitive sanity inequality (test 2's fix uses the latter, cheaper, form).
