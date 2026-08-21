"""Regression coverage for exact closure-truth scale evaluation.

Exercises ``jam_truth_curves`` / ``nnpdf_truth_curves`` in
``closure_{JAM,NNPDF}_truth{,_small}/pdf_guidance.py`` (the LHAPDF-scale-to-truth-curve
step) and ``ensemble_curves`` / ``_ensemble_cache_path`` in the matching ``generate.py``
(the on-disk cache for the replica ensemble used to calibrate the fixed lattice
covariance).  No re-export shim is involved -- these are the real implementations,
imported directly from the four ``closure_*_truth*`` driver packages, and this file
imports nothing from ``pixel`` at all.

Every test here monkeypatches the one function that shells out (``dump_jam`` /
``dump_nnpdf``) before calling in, so no subprocess runs, no C++ helper gets compiled,
and no physical PDF value is ever read.  Most are pure data-flow and
cache-invalidation regression tests; they say nothing about the closure fit itself,
which lives in ``run_closure.py`` / ``fit.py`` alongside these modules.

The one arithmetic claim is
``test_project_basis_builds_the_su4_flavor_combinations`` (added 2026-08-13), which
pins all nine flavor combinations (``t3 = (u+ubar)-(d+dbar)`` etc.) against
hand-computed integers.  Before it, every test checked only ``project_basis``'s key
set (``set(curves) == set(cfg.ALL_FIELDS)``), so a wrong sign or coefficient there --
in the step that turns LHAPDF flavor columns into the injected truth -- passed
untouched.

The scale/cache oracles are internal to the suite, not independent physics: the
expected ``q``
is read from the same ``cfg.TRUTH_Q_CHOICES`` table the source itself reads, and the
"stale" vs. "fresh" cache values are sentinels the tests construct
themselves.  What makes them worth having anyway is git history, not an external
reference: commit ``6c5857b`` deleted a real ``clamp_scale`` helper that silently
substituted ``Q -> QMin`` whenever a requested truth scale undershot an LHAPDF set's
advertised minimum, and mislabeled the result under a ``clamped_to_qmin`` meta key
(``git log -p -S"clamp_scale" -- closure_JAM_truth/pdf_guidance.py`` shows the exact
diff).  The two ``clamp``-named tests are precise regression guards against that
specific, previously-shipped bug reappearing -- one in the live evaluation path, one
in a cache file written while the bug was live -- and each is now paired with the
branch it could not reach: ``below_set_qmin``'s ``False`` side
(``test_truth_curves_report_a_scale_above_the_set_minimum``) and the cache's
*accept* half including its legacy-key migration
(``test_ensemble_cache_accepts_a_matching_scale_and_migrates_legacy_meta``).

The two LHAPDF sets (``JAM24_PDF_proton_nlo``, ``QMin=1.14`` GeV;
``NNPDF40_nnlo_as_01180_1000``, ``QMin=1.65`` GeV -- both read from the vendored
``lhapdf/*.tar.gz`` ``.info`` members) are identical between a suite's "full" and
"_small" variant, and ``pdf_guidance.py`` is byte-identical across all four driver
packages modulo names -- so the full/small parametrization here exercises each config
module's own wiring, not a second code path.  ``generate.py`` is *not* identical,
though: the full suites build the truth from one fixed LHAPDF member
(``cfg.FIXED_TRUTH_MEMBER``) and use the replica ensemble only to calibrate the
covariance, while the ``_small`` suites still use the ensemble *mean* itself as the
truth (``closure_JAM_truth_small/README.md:37``) -- the cache test below exercises
``ensemble_curves`` for both without asserting which methodology is in effect, which is
correct for its scope (cache-staleness rejection) but worth knowing when reading it.
"""

import json

import numpy as np
import pytest

from closure_JAM_truth import config as full_jam_cfg
from closure_JAM_truth import generate as full_jam_generate
from closure_JAM_truth import pdf_guidance as full_jam_guidance
from closure_JAM_truth_small import config as small_jam_cfg
from closure_JAM_truth_small import generate as small_jam_generate
from closure_JAM_truth_small import pdf_guidance as small_jam_guidance
from closure_NNPDF_truth import config as full_nnpdf_cfg
from closure_NNPDF_truth import generate as full_nnpdf_generate
from closure_NNPDF_truth import pdf_guidance as full_nnpdf_guidance
from closure_NNPDF_truth_small import config as small_nnpdf_cfg
from closure_NNPDF_truth_small import generate as small_nnpdf_generate
from closure_NNPDF_truth_small import pdf_guidance as small_nnpdf_guidance


# (cfg, guidance, dump-function name, curves-function name, q_key).  Every q_key
# chosen here is below that module's own QMin (JAM 1.0 < 1.14; NNPDF 1.0 and
# 1.28 < 1.65), so `below_set_qmin` is asserted `True` in every case -- the `False`
# side (e.g. JAM's own "mc" = 1.28, which sits *above* its 1.14 QMin) is never
# exercised anywhere in this suite; see the missing-coverage note filed for this file.
TRUTH_SCALE_CASES = (
    (full_jam_cfg, full_jam_guidance, "dump_jam", "jam_truth_curves", "1"),
    (
        full_nnpdf_cfg,
        full_nnpdf_guidance,
        "dump_nnpdf",
        "nnpdf_truth_curves",
        "1",
    ),
    (
        full_nnpdf_cfg,
        full_nnpdf_guidance,
        "dump_nnpdf",
        "nnpdf_truth_curves",
        "mc",
    ),
)
# The ``_small`` suites were dropped from this list on 2026-08-15, when their
# ``TRUTH_Q_CHOICES`` was reduced to {2, 3} and neither member sat below any set's
# QMin (NNPDF 1.65, JAM 1.14).  **That reduction was reverted on 2026-08-16** --
# all six members are back, so the ``_small`` suites once again have below-QMin
# members (``mc`` = 1.28 and ``1`` for NNPDF; ``1`` for JAM) and the note below no
# longer holds.  They are still absent from this parametrization, so the
# below-QMin path is exercised only through the full suites; adding the two
# ``_small`` cases would close that gap (open coverage item, 2026-08-17).


@pytest.mark.parametrize(
    ("cfg", "guidance", "dump_name", "curves_name", "q_key"),
    TRUTH_SCALE_CASES,
)
def test_truth_curves_use_requested_scale_without_qmin_clamp(
    cfg, guidance, dump_name, curves_name, q_key, monkeypatch
):
    """The requested truth scale reaches the LHAPDF dumper unclamped.

    ``dump_name`` (``dump_jam``/``dump_nnpdf``, the only piece of
    ``jam_truth_curves``/``nnpdf_truth_curves`` that shells out) is mocked; the rest
    of the function runs for real.  Checks the recorded ``q`` argument, both
    ``meta["q_original"]``/``["q_effective"]``, and the absence of
    ``clamp_scale``/``clamped_to_qmin`` -- three independent guards against the
    ``clamp_scale`` helper removed in commit ``6c5857b``, which silently substituted
    ``Q -> QMin`` below a set's advertised minimum (see module docstring).  Oracle is
    internal (``cfg.TRUTH_Q_CHOICES``, the same table the source reads) -- a
    data-flow identity, not an independent physics reference; bar is exact ``==``.
    Does not check ``project_basis``'s flavor-combination arithmetic, only the key
    set of ``curves`` -- a wrong linear combination in ``project_basis`` would pass
    this test untouched.
    """
    nodes = np.array([1.0e-3, 0.1, 0.7])
    calls = []

    def fake_dump(x, *, q, member):
        """Record the call and return a synthetic 10-column ``[x, g, u, ...]`` block."""
        calls.append((np.asarray(x), q, member))
        return np.column_stack([x, *[(i + 1) * x for i in range(9)]])

    monkeypatch.setattr(guidance, dump_name, fake_dump)
    curves, meta = getattr(guidance, curves_name)(nodes, q_key=q_key)

    requested_q = float(cfg.TRUTH_Q_CHOICES[q_key])
    # The scale forwarded to LHAPDF must be the raw requested value -- this is the
    # direct check that no `clamp_scale`-style substitution happened before the call.
    assert calls[0][1] == requested_q
    # `pytest.approx` here is a float-safety net for an exact passthrough (`nodes` is
    # never transformed before reaching the dumper), not a physics tolerance; default
    # rel=1e-6/abs=1e-12 is far tighter than any plausible transformation bug.
    assert calls[0][0] == pytest.approx(nodes)
    # Keys only: confirms `project_basis` returns the 9 expected fields, not that its
    # per-field arithmetic (t3 = up-dp, etc.) is correct.
    assert set(curves) == set(cfg.ALL_FIELDS)
    assert meta["q_original"] == requested_q
    assert meta["q_effective"] == requested_q
    # Real production logic (not mocked): read from each set's vendored `.info` file.
    # JAM QMin=1.14, NNPDF QMin=1.65 -- every q_key used by this parametrization sits
    # below its set's QMin, so this assertion is only ever exercised on the True side;
    # the False side and the QMin-is-None guard are covered by
    # test_truth_curves_report_a_scale_above_the_set_minimum below.
    assert meta["below_set_qmin"] is True
    # Guards against the exact stale meta key the pre-fix `clamp_scale` code wrote.
    assert "clamped_to_qmin" not in meta
    # Guards against the helper itself being reintroduced.
    assert not hasattr(guidance, "clamp_scale")


# (cfg, guidance, dump-function name, curves-function name, q_key, QMin attribute).
# The complement of TRUTH_SCALE_CASES: every q_key here sits *above* its set's
# advertised QMin (JAM 1.28 > 1.14; NNPDF 2.0 > 1.65), so `below_set_qmin` must
# come back False.  JAM's "mc" is the naturally-occurring case -- it is this
# suite's own input scale Q0 = mc.
ABOVE_QMIN_CASES = (
    (full_jam_cfg, full_jam_guidance, "dump_jam", "jam_truth_curves", "mc", "JAM_QMIN"),
    # was "mc"; the _small suites now start at 2, which is likewise above JAM QMin 1.14.
    (small_jam_cfg, small_jam_guidance, "dump_jam", "jam_truth_curves", "2", "JAM_QMIN"),
    (
        full_nnpdf_cfg, full_nnpdf_guidance, "dump_nnpdf", "nnpdf_truth_curves",
        "2", "NNPDF_QMIN",
    ),
    (
        small_nnpdf_cfg, small_nnpdf_guidance, "dump_nnpdf", "nnpdf_truth_curves",
        "2", "NNPDF_QMIN",
    ),
)


@pytest.mark.parametrize(
    ("cfg", "guidance", "dump_name", "curves_name", "q_key", "qmin_attr"),
    ABOVE_QMIN_CASES,
)
def test_truth_curves_report_a_scale_above_the_set_minimum(
    cfg, guidance, dump_name, curves_name, q_key, qmin_attr, monkeypatch
):
    """``below_set_qmin`` is False above QMin, and False again when QMin is unknown.

    The False side of a flag every other test in this file sees only as True
    (oracle A3: the comparison is re-derived here from ``cfg.TRUTH_Q_CHOICES``
    and the set's own ``QMin``, both read off the vendored ``.info``, rather
    than from the meta dict under test).  Without it, ``below_set_qmin`` could be
    hardcoded ``True`` -- or be the wrong comparison entirely -- and the whole
    file would still pass; the flag exists to record whether LHAPDF is
    extrapolating, so a stuck value misreports every truth built at ``Q >=
    QMin``, which is most of ``TRUTH_Q_CHOICES``.

    Also drives the ``cfg.<SET>_QMIN is not None`` guard, which is otherwise
    unreachable in a test: a set whose ``.info`` carries no ``QMin`` (or one
    that is not installed) must report ``False``, not ``None`` and not raise.
    ``q_original``/``q_effective`` are re-checked here because clamping and this
    flag are the two halves of the same removed helper.
    """
    nodes = np.array([1.0e-3, 0.1, 0.7])

    def fake_dump(x, *, q, member):
        """Record nothing; the scale flag, not the payload, is under test here."""
        return np.column_stack([x, *[(i + 1) * x for i in range(9)]])

    monkeypatch.setattr(guidance, dump_name, fake_dump)
    requested_q = float(cfg.TRUTH_Q_CHOICES[q_key])
    # The premise of the case, asserted rather than assumed: this q really is
    # above the set's advertised minimum.
    assert requested_q > getattr(cfg, qmin_attr)

    _, meta = getattr(guidance, curves_name)(nodes, q_key=q_key)
    assert meta["below_set_qmin"] is False
    assert meta["q_original"] == requested_q
    assert meta["q_effective"] == requested_q  # still no clamp on this side

    # An uninstalled set (no QMin in the .info) must not turn the flag into
    # None or an exception.
    monkeypatch.setattr(cfg, qmin_attr, None)
    # Was "1"; the _small suites no longer define it.  With QMin set to None the
    # flag must be False regardless of which member is asked for.
    _, meta = getattr(guidance, curves_name)(
        nodes, q_key=next(iter(cfg.TRUTH_Q_CHOICES))
    )
    assert meta["below_set_qmin"] is False


# (cfg, guidance, dump-name, curves-name).  `project_basis` is byte-identical in
# all four packages, so this exercises each module's own import wiring rather
# than four implementations.
PROJECT_BASIS_CASES = (
    (full_jam_guidance,), (small_jam_guidance,),
    (full_nnpdf_guidance,), (small_nnpdf_guidance,),
)


@pytest.mark.parametrize("guidance", [c[0] for c in PROJECT_BASIS_CASES])
def test_project_basis_builds_the_su4_flavor_combinations(guidance):
    """The nine basis momentum densities equal their hand-computed values.

    The one arithmetic claim in this file.  Every other test checks only
    ``set(curves) == set(cfg.ALL_FIELDS)``, so a wrong sign or coefficient in
    any of the nine combinations passes untouched -- and ``project_basis`` is
    where the LHAPDF flavor columns become the fitted fields, so an error there
    silently rewrites the injected truth itself.

    Oracle A1: the standard SU(4) definitions, evaluated by hand on the fixture
    below and written out as literal integers, so a coefficient change in the
    source cannot propagate into the expectation.  With
    ``(g, u, ubar, d, dbar, s, sbar, c, cbar) = (23, 11, 3, 7, 5, 13, 2, 17,
    19)`` the C-even sums are ``u+ = 14, d+ = 12, s+ = 15, c+ = 36`` and the
    C-odd differences ``u- = 8, d- = 2, s- = 11, c- = -2``, giving

        t3 = u+ - d+                = 2      v3  = u- - d-               = 6
        t8 = u+ + d+ - 2 s+         = -4     v8  = u- + d- - 2 s-        = -12
        sigma = u+ + d+ + s+ + c+   = 77     v   = u- + d- + s- + c-     = 19
        t15 = u+ + d+ + s+ - 3 c+   = -67    v15 = u- + d- + s- - 3 c-   = 27
        g = g                       = 23

    The nine values are pairwise distinct in magnitude, so no swap between two
    outputs and no sign flip can land on another output's value.  The second row
    scales the first by ``-2``, which is linear in every combination and so
    catches a row/column transposition without weakening any of the above.
    """
    flavors = np.array([23.0, 11.0, 3.0, 7.0, 5.0, 13.0, 2.0, 17.0, 19.0])
    raw = np.column_stack([
        np.array([0.1, 0.9]),                       # the x column, never read
        *[np.array([v, -2.0 * v]) for v in flavors],
    ])
    expected = {
        "t3": 2.0, "t8": -4.0, "sigma": 77.0, "t15": -67.0, "g": 23.0,
        "v3": 6.0, "v8": -12.0, "v": 19.0, "v15": 27.0,
    }
    curves = guidance.project_basis(raw)
    assert set(curves) == set(expected)
    for field, value in expected.items():
        np.testing.assert_allclose(
            curves[field], [value, -2.0 * value], rtol=0.0, atol=0.0
        )


# (cfg, generate, guidance, curves-function name).  `generate.py` is not
# byte-identical across full/small the way `pdf_guidance.py` is -- see module
# docstring -- but `ensemble_curves`'s cache-validity gate (the thing under test
# here) is the same code in all four, differing only in the meta key it writes on a
# *fresh* compute (`uncertainty_kind` for full, `truth_kind` for small).
CACHE_CASES = (
    (full_jam_cfg, full_jam_generate, full_jam_guidance, "jam_truth_curves"),
    (small_jam_cfg, small_jam_generate, small_jam_guidance, "jam_truth_curves"),
    (
        full_nnpdf_cfg,
        full_nnpdf_generate,
        full_nnpdf_guidance,
        "nnpdf_truth_curves",
    ),
    (
        small_nnpdf_cfg,
        small_nnpdf_generate,
        small_nnpdf_guidance,
        "nnpdf_truth_curves",
    ),
)


@pytest.mark.parametrize(
    ("cfg", "generate", "guidance", "curves_name"),
    CACHE_CASES,
)
def test_ensemble_cache_rejects_old_clamped_scale(
    cfg, generate, guidance, curves_name, tmp_path, monkeypatch
):
    """A cache written under the old ``clamp_scale`` regime is not trusted.

    Writes an adversarial ``.npz`` cache whose ``q_effective`` equals the historical
    *clamped* value (standing in for what the removed ``clamp_scale`` would have
    produced for this request) while every other cache-validity field (node array,
    member count) matches the live call exactly -- isolating ``q_effective`` as the
    one field that must fail the match.  ``ensemble_curves``'s cache gate
    (``closure_JAM_truth/generate.py:356-373``, identical logic in the other three
    modules) must then reject the cache and recompute via the mocked
    ``curves_name``.  Proven three ways: the mock is called exactly once with the
    right ``(q_key, member)``; the returned array is the mock's ``1``s, not the
    cached ``0``s; and ``meta["q_effective"]`` is the corrected value.  Oracle is
    internal (two sentinel objects this test builds itself: a stale-cache write and
    a fresh-compute stub) -- not independent physics, a cache-invalidation contract.
    Catches a cache-validity check that stops comparing ``q_effective`` (e.g. keying
    only on node count and member count).  Does not check that a *valid* cache is
    ever accepted, so the full-suite-only legacy-key migration on a cache hit
    (``generate.py:369-372``, popping a stale ``truth_kind`` in favour of
    ``uncertainty_kind``) is never exercised by this suite.  Also checks only
    ``cfg.ALL_FIELDS[0]``, and the mock returns identical content for every field, so
    a bug that mixed up which stacked array lands under which field key would pass
    silently -- see the weakness filed for this test.
    """
    monkeypatch.setattr(cfg, "REFERENCE_DIR", tmp_path)
    nodes = np.array([0.1, 0.2])
    # Not a literal: the member tables differ per suite and have moved ("1" was
    # dropped from the _small pair on 2026-08-15 and restored on 2026-08-16).  Any
    # member exercises the cache.
    q_first = next(iter(cfg.TRUTH_Q_CHOICES))
    requested_q = float(cfg.TRUTH_Q_CHOICES[q_first])
    # Stand-in for the pre-6c5857b `clamp_scale(q)` output: the set's QMin, not the
    # actually-requested `q` -- exactly the value the removed helper would have
    # substituted.  `cfg` is JAM's or NNPDF's module depending on the parametrized
    # case, hence the `hasattr` branch rather than a single attribute name.
    stale_q = float(cfg.JAM_QMIN if hasattr(cfg, "JAM_QMIN") else cfg.NNPDF_QMIN)
    stale_meta = {"q_original": requested_q, "q_effective": stale_q}
    # Zeros are the sentinel for "stale, must not reach the caller".
    stale_curves = {field: np.zeros((1, nodes.size)) for field in cfg.ALL_FIELDS}
    np.savez(
        generate._ensemble_cache_path(q_first),
        x_nodes=nodes,
        n_members=1,
        meta=json.dumps(stale_meta),
        **stale_curves,
    )

    calls = []

    def exact_curves(x, *, q_key, member):
        """Stand in for a fresh, correctly-scaled recompute.

        Each field gets its **own** value (``100 + index``), not a shared ``1``:
        identical content per field is exactly the degeneracy that let the
        original version of this test miss a field mix-up, since every key then
        carried the same array.  ``100 + index`` also keeps the fresh values
        clear of the stale cache's ``0``s.
        """
        calls.append((q_key, member))
        curves = {
            field: np.full(nodes.size, 100.0 + i)
            for i, field in enumerate(cfg.ALL_FIELDS)
        }
        return curves, {
            "member": member,
            "q_original": requested_q,
            "q_effective": requested_q,
        }

    monkeypatch.setattr(guidance, curves_name, exact_curves)
    q_first = next(iter(cfg.TRUTH_Q_CHOICES))  # not a literal: the table is suite-specific
    curves, meta = generate.ensemble_curves(q_first, nodes, members=(1,))

    # A cache hit would never call the mock at all -- this alone proves rejection.
    assert calls == [(q_first, 1)]
    # Every field, not just the first, and each with its own value: a stack that
    # landed under the wrong key would keep the right key set and the right
    # shapes, so only per-field content can catch it.
    assert set(curves) == set(cfg.ALL_FIELDS)
    for i, field in enumerate(cfg.ALL_FIELDS):
        # Fresh values, not the cached `0`s: the payload came from recomputation.
        assert curves[field] == pytest.approx(
            np.full((1, nodes.size), 100.0 + i), rel=1e-12, abs=0.0
        )
    assert meta["q_effective"] == requested_q


@pytest.mark.parametrize(
    ("cfg", "generate", "guidance", "curves_name"),
    CACHE_CASES,
)
def test_ensemble_cache_accepts_a_matching_scale_and_migrates_legacy_meta(
    cfg, generate, guidance, curves_name, tmp_path, monkeypatch
):
    """A cache whose scale matches is used, and its legacy meta key is normalized.

    The accept half of the gate ``test_ensemble_cache_rejects_old_clamped_scale``
    only ever drives from the reject side.  A gate that rejected *everything*
    would pass that test perfectly while making the cache dead weight -- the
    ensemble is 791 LHAPDF member dumps, so an always-miss is a silent
    regeneration of all of them, not a wrong number.  Here the recompute stub
    raises instead of returning, so any miss fails loudly.

    Oracle A3, mirroring its sibling: the cache is written by this test with
    ``q_effective`` equal to the value the live request will compute, and each
    field gets its own sentinel row so a hit cannot be confirmed by shape alone.

    The full suites additionally run a one-way migration on a hit
    (``generate.py:369-372``): a cache written before the ``truth_kind`` ->
    ``uncertainty_kind`` rename must come back carrying only the new key.  That
    branch had no test at all -- it is reachable *only* through a cache hit, so
    the reject-side test could never have entered it.  The ``_small`` suites do
    not migrate, and this asserts that difference explicitly rather than
    skipping it, since the two halves are one parametrization.
    """
    monkeypatch.setattr(cfg, "REFERENCE_DIR", tmp_path)
    nodes = np.array([0.1, 0.2])
    # Not a literal: the member tables differ per suite and have moved ("1" was
    # dropped from the _small pair on 2026-08-15 and restored on 2026-08-16).  Any
    # member exercises the cache.
    q_first = next(iter(cfg.TRUTH_Q_CHOICES))
    requested_q = float(cfg.TRUTH_Q_CHOICES[q_first])
    # A cache written by an older build: right scale, but the pre-rename meta key.
    cached_meta = {
        "q_original": requested_q,
        "q_effective": requested_q,
        "truth_kind": "replica_ensemble_mean",
    }
    cached_curves = {
        field: np.full((1, nodes.size), 10.0 + i)
        for i, field in enumerate(cfg.ALL_FIELDS)
    }
    np.savez(
        generate._ensemble_cache_path(q_first),
        x_nodes=nodes,
        n_members=1,
        meta=json.dumps(cached_meta),
        **cached_curves,
    )

    def never_called(x, *, q_key, member):
        raise AssertionError("a valid cache must not be recomputed")

    monkeypatch.setattr(guidance, curves_name, never_called)
    q_first = next(iter(cfg.TRUTH_Q_CHOICES))  # not a literal: the table is suite-specific
    curves, meta = generate.ensemble_curves(q_first, nodes, members=(1,))

    assert set(curves) == set(cfg.ALL_FIELDS)
    for i, field in enumerate(cfg.ALL_FIELDS):
        assert curves[field] == pytest.approx(
            np.full((1, nodes.size), 10.0 + i), rel=1e-12, abs=0.0
        )
    assert meta["q_effective"] == requested_q
    if "_small" in generate.__name__:
        assert meta["truth_kind"] == "replica_ensemble_mean"  # no migration here
    else:
        # The rename is applied on read, so an old cache cannot smuggle the
        # stale key back into a fresh manifest.
        assert "truth_kind" not in meta
        assert meta["uncertainty_kind"] == "replica_ensemble_covariance_calibration"
