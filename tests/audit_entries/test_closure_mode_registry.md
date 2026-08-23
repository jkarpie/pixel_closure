### `tests/test_closure_mode_registry.py`

**Exercises** the four `closure_*/config.py` `TEST_MODES` tuples and `run_closure.sh`'s mode
`case` arm (`:205`) against `pixel.data.families`. No `src/pixel` execution beyond importing
the registry, which is pure data; no kernels, no fits.

**Claim** the process taxonomy is stated once, in `pixel.data.families`. Three lists have to
agree — a Python tuple per suite, a shell `case` arm, and the registry — and every previous
mechanism for keeping them together was "remember to". The specific failure this guards is
*quiet*: `run_closure.sh` `die`s on an argument it does not recognise, so a mode a config
offers but the runner rejects is not a suite failure, it is a run that silently does less.

**Oracle** `F1` throughout. These are internal-consistency contracts across independently
maintained lists; there is no external reference and none would be appropriate. The runner's
accepted set is **parsed** from `run_closure.sh` rather than restated, because restating it
would create a fourth copy of exactly the list this file exists to keep down to one.

**Cost** 8 cases, ~0.9 s, dominated by importing the four configs.

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 1 | `test_full_suite_modes_are_exactly_the_registry_modes` (×2) | Full suites offer `families.MODES` and nothing suite-local. | Tuple equality, order included. (`F1`) | exact | `S1` |
| 2 | `test_small_suite_modes_are_the_registry_modes_plus_named_synthetic_ones` (×2) | Smoke suites prefix-match the registry then add exactly `TEST_MODE_SYNTHETIC_Z`, which must not be a registry family. | Prefix + suffix comparison, disjointness against `FAMILIES`. (`F1`) | exact | `S1` |
| 3 | `test_runner_accepts_every_mode_every_suite_offers` | No configured mode is rejected by the runner. | Parsed `case` arm minus the union of all four `TEST_MODES`. (`F1`) | exact | `S1` |
| 4 | `test_runner_accepts_no_mode_no_suite_offers` | No stale arm left after a mode is retired. | The same comparison, other direction. (`F1`) | exact | `S2` |
| 5 | `test_reserved_registry_families_are_not_offered_by_any_suite` | A reserved family (no builder) is never a selectable mode. | Disjointness. (`F1`) | exact | `S1` |
| 6 | `test_all_four_suites_agree_on_the_shared_modes` | The four configs differ only in truth set and smoke scale, not in taxonomy. | Set of the four prefix tuples has size 1. (`F1`) | exact | `S1` |

**Why the parse is self-checking.** Tests 3 and 4 pin the runner's accepted set in *both*
directions, so a regex that silently mis-parsed would fail one of them rather than pass
vacuously. Measured at time of writing: the arm yields
`{lattice, dis, dy, exp, both, synthetic_z_plumbing}`, the union of all four `TEST_MODES` is
the same set, and both tests pass — i.e. there is no drift today.

**A stale claim this retires.** `plans/hadronic_process_families.md:0.5` (in the `pixel`
repo) states that `run_closure.sh` "**has already drifted**, accepting only
`lattice|dis|dy|exp|both` and `die`ing on `synthetic_cc` and `synthetic_z_plumbing`, both of
which are in `cfg.TEST_MODES`". Measured 2026-08-21: the arm *does* accept
`synthetic_z_plumbing`, and no config carries `synthetic_cc` at all. The drift was fixed at
some point without the plan being updated. This file is now what would catch it recurring.

**Weak spots / not covered**

- ~~**Acceptance is not selection.**~~ **Closed 2026-08-22** by four new cases (see the
  section below). What remains uncovered is narrower: nothing runs a fit and checks that
  the *datasets actually built* match the families the registry names — the tests assert
  the gate values, not the resulting dataset list.
- The regex assumes the runner keeps one `case` arm of the form `a|b|c) modes+=(`. A
  refactor into several arms would make the parse find only the first, and tests 3/4 would
  fail — loudly, which is the right direction, but the message would point at drift rather
  than at the parse.
- `TEST_MODE_SYNTHETIC_Z` is the only suite-local mode contemplated. A second one would fail
  test 2, again in the safe direction.

---

## Companion: `test_closure_chorus_selection.py`, 2026-08-22

Seven cases pinning that the two CHORUS records reach the closure selection as `physics` with
all 607 rows each, that their `operator.nuclear` field **names** the composition
(`A = 207.2, Z = 82`, `legacy_dw`) rather than saying "required", that the selection is
exactly 30 records (13 lattice + 15 DIS + 2 DY), and that NuTeV does *not* come along —
it shares CHORUS's target machinery and beam, so it is the likeliest accidental promotion.

**The first test is the one that earned its place.** `pixel/observable_contracts.yaml` and
`pixel_closure/observable_contracts.yaml` were byte-identical, and only the first is read:
`contracts.load_observable_contracts` resolves `repository_root()/observable_contracts.yaml`
against the *pixel* checkout. The CHORUS promotion was first applied to the closure copy and
changed nothing — selection stayed at 28 with no error anywhere. The test digests both files
and fails on divergence, with a message saying which one is read.

Oracle `F1`/`A3` throughout; no physics. Not covered: whether the closure can *generate*
CHORUS — `generate.py` has no layout and `datasets.py` no builder arm, so selection is
necessary and not sufficient.


---

## The call-site migration, 2026-08-22

Until this date `families.py` was **declared and not consumed**. Six places restated what
it already held: four `closure_*/fit.py` `include_* = mode in (...)` chains, and the
manifest section tuple inline in both `api/manifest.py` and `data/contracts.annotate_manifest`.
The tests above pinned that the lists *agreed*; nothing made them agree.

Four cases were added:

| # | Test | What is asserted | How / oracle | Bar | S |
|---|---|---|---|---|---|
| 7 | `test_fit_derives_its_mode_gates_from_the_registry` (×4) | Every `fit.py` calls `families.mode_families`, and none of the three retired literals survives. | Parsed source, both directions. (`F1`) | exact | `S1` |
| 8 | `test_the_registry_gates_select_what_the_old_literals_did` (×4) | The derivation reproduces the retired chain **exactly**. | Recomputes both sides. (`F1`) | exact | `S1` |
| 9 | `test_manifest_sections_come_from_the_registry` | Neither module restates the section tuple; the `None` default really resolves. | Signature + source + a live call. (`F1`) | exact | `S1` |

**Test 8 is the one that stops this being a worse bug than the drift.** A refactor that
quietly changed which datasets a mode selects would satisfy test 7 — the call sites *would*
be reading the registry — while silently fitting a different dataset set. It recomputes the
retired mapping (`exp → {dis, dy}`, `both → {lattice, dis, dy}`, each family → itself) and
compares.

**One documented exception, in the smoke suites only.** `synthetic_z_plumbing` is not a
process family; `mode_families` raises on it, which is correct for the registry and wrong at
that call site. Both `_small` suites short-circuit to `frozenset()` before asking. The
alternative — registering a fake family — would have put a non-physics mode into the
taxonomy that `run_closure.sh`, the contract section enum and the plot-kind tables all read.

**Mutation-checked 2026-08-22.** `families.mode_families` was wrapped in memory to drop
`"dy"` from `exp` and `both`. This is the migration's real risk and it is *worse* than the
drift it replaced: the call sites now read the registry, so a wrong registry is obeyed
everywhere at once, nothing crashes, and the fit reports success having fitted less data.
Test 8 failed on all four suites. Test 7 passed — correctly, since the call sites really
were still consulting the registry — which is exactly why test 8 is the one that matters.

**Weak spots**

- Tests 7 and 9 scan source text. A `fit.py` that imported `mode_families` and then ignored
  it would pass 7; the anti-vacuity regex (`"lattice"|"dis"|"dy" in selected`) narrows that
  but does not close it. Test 8 is the real defence.
- Test 9's source scan on `contracts.py` has to exempt the docstring occurrence of the
  tuple, so it strips one literal spelling before checking. A differently-formatted
  docstring mention would make it fail spuriously.
- `include_nnpdf` (the generic `datasets` section) is still derived as `include_exp` rather
  than from the registry, because `datasets` is a *generic* section rather than a family.
  `GENERIC_MANIFEST_SECTION` exists but nothing routes modes to it.
