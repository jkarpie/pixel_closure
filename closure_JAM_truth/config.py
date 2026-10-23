"""Shared configuration for the JAM-anchored full-flavor closure suite.

Every constant that generation *and* fitting must agree on lives here, so the
two can never silently drift.  See ``closure_plan.md`` (repo root) for the
physics narrative, ``closure_extension.md`` / ``closure_extension_plan.md`` for
the full-scale design, and ``closure_JAM_truth/README.md`` for usage.

Truth
-----
The truth is the real ``JAM24_PDF_proton_nlo`` global-analysis result
(NLO, VFNS).  We read JAM at several *original* scales ``Q`` and treat each
resulting curve as the truth at the common input scale ``Q0 = mc`` -- the scale
at which the closure distributions are parameterized and the fake data are
generated.  Reading JAM at different ``Q`` gives different but realistic
PDF-shaped truths; comparing the recovered fits across ``Q`` is the proof that
the code works across inputs.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

TWO_PI = 2.0 * math.pi

# -- locations ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_ROOT = ROOT / "data"
RESULTS_ROOT = ROOT / "results"
REFERENCE_DIR = ROOT / "reference_pdfs"          # cached JAM dumps / ref fits
DIS_MANIFEST_PATH = DATA_ROOT / "dis_manifest.json"
#: Per-table DIS correlated-systematic vectors, stored *relative* to the real
#: value.  Too bulky for the JSON manifest (HERA ships 169 sources per table),
#: so dis_audit writes one .npz per table here and the manifest references it.
DIS_SYS_DIR = DATA_ROOT / "dis_systematics"

#: Shared kernel-matrix cache.  The lattice/DIS forward operators depend only on
#: the (Q-independent) dataset kinematics + Q0/nf/order, not on which JAM truth
#: scale Q is being generated, so every truth member reuses one cache instead of
#: rebuilding identical matrices per truthQ_* directory.
KERNEL_CACHE_ROOT = DATA_ROOT / "_kernel_cache"
#: Fingerprint-validated contour-reference and Hessian setup checkpoints.
INFERENCE_CACHE_ROOT = DATA_ROOT / "_inference_cache"

#: The fitpack checkout is expected beside this PIXEL checkout, e.g.
#: ``/work/pixel`` and ``/work/fitpack_legacy``.
FITPACK_ROOT = REPO_ROOT.parent / "fitpack_legacy"
#: fitpack unpolarized-DIS Excel tables (real kinematics + statistical errors).
FITPACK_IDIS = FITPACK_ROOT / "database" / "idis" / "expdata"
#: fitpack fixed-target Drell-Yan Excel tables.
FITPACK_DY = FITPACK_ROOT / "database" / "dy" / "expdata"
FITPACK_ZRAP = FITPACK_ROOT / "database" / "zrap" / "expdata"
FITPACK_WZHEP = FITPACK_ROOT / "database" / "wzhep" / "expdata"
FITPACK_WASYM = FITPACK_ROOT / "database" / "wasym" / "expdata"
NNPDF_COMMONDATA = REPO_ROOT.parent / "nnpdf" / "nnpdf_data" / "nnpdf_data" / "commondata"

# The canonical NNPDF4.0 comparison scope is the frozen published-baseline
# DIS/DY roster.  Generation must not infer this list by globbing a mutable
# NNPDF checkout.  Native builders land family by family; until every selected
# record passes its readiness gate, the contracts remain an explicit target
# rather than silently entering with a wrong inclusive proxy.
from pixel.data.nnpdf.native import (
    assert_native_campaign_ready as _assert_native_campaign_ready,
    nnpdf40_native_contracts_for_mode as _nnpdf40_native_contracts_for_mode,
)
NNPDF40_ROSTER_PATH = REPO_ROOT / "nnpdf40_dis_dy_roster.yaml"


def nnpdf40_datasets_for_mode(mode: str):
    """Published NNPDF4.0 DIS/DY native contracts targeted by one full mode."""
    return _nnpdf40_native_contracts_for_mode(mode, scale="full")


def assert_nnpdf40_native_ready(mode: str, *, validated_datasets=()):
    """Block canonical generation until every selected native operator passes NNPDF parity."""
    _assert_native_campaign_ready(
        nnpdf40_datasets_for_mode(mode), validated_datasets=validated_datasets
    )

# Contract-derived selections.  These are views of observable_contracts.yaml,
# never a second hand-maintained keep-list.  They remain empty while every EW
# or NNPDF candidate is scientifically gated as reader_only.
from pixel.data.contracts import full_closure_selection as _full_closure_selection
#: The one place this suite names its dataset selection.  ``generate`` and the
#: manifest read it from here rather than repeating the string, because a mode
#: stated twice is a mode that drifts: ``nnpdf_layout`` hardcoded "nnpdf_full"
#: and silently produced an empty layout when the modes were split on 2026-09-09.
CONTRACT_MODE = "jam_full"
_CONTRACT_SELECTION = _full_closure_selection(CONTRACT_MODE)
Z_ANALYSIS = tuple(r for r in _CONTRACT_SELECTION if str(r["id"]).startswith("z."))
W_ASYM_ANALYSIS = tuple(r for r in _CONTRACT_SELECTION if str(r["id"]).startswith("w."))
NNPDF_ANALYSIS = tuple(
    r for r in _CONTRACT_SELECTION
    if str(r["id"]).startswith(("dy.e906", "dy.e605", "dis.chorus", "dis.nutev"))
)
NNPDF_DY_MAX_ROWS = 40

# -- LHAPDF / JAM truth ------------------------------------------------------

#: Discover an LHAPDF source/install tree beside the PIXEL checkout, e.g.
#: ``/work/pixel`` and ``/work/LHAPDF-6.5.6/lhapdf-install``.  Sorting by the
#: numeric version makes the choice deterministic if several builds coexist.
_LHAPDF_INSTALLS = sorted(
    REPO_ROOT.parent.glob("LHAPDF-*/lhapdf-install"),
    key=lambda path: tuple(
        int(piece) for piece in path.parent.name.removeprefix("LHAPDF-").split(".")
    ),
)
LHAPDF_PREFIX = (
    _LHAPDF_INSTALLS[-1]
    if _LHAPDF_INSTALLS
    else REPO_ROOT.parent / "lhapdf-install"
)
_LHAPDF_LIB_DIRS = (LHAPDF_PREFIX / "lib", LHAPDF_PREFIX / "lib64")
LHAPDF_LIB_DIR = next(
    (path for path in _LHAPDF_LIB_DIRS if any(path.glob("libLHAPDF.*"))),
    LHAPDF_PREFIX / "lib",
)
JAM_SET = "JAM24_PDF_proton_nlo"
#: The full JAM grid is tracked here and shared by the full-scale JAM suites.
#: A checkout can generate truth without copying or linking it into ``data``.
JAM_REPO_DATA_ROOT = REPO_ROOT / "closure_JAM_truth" / "lhapdf"
#: Optional tarball fallback for checkouts that provide one instead of the grid.
JAM_TARBALL = ROOT / "lhapdf" / f"{JAM_SET}.tar.gz"
JAM_LHAPDF_DATA_ROOT = DATA_ROOT / "lhapdf"
JAM_MEMBER = 0            # 0 = central; used by pdf_guidance's standalone cache

#: LHAPDF's data directory for the configured prefix.
LHAPDF_DATA_DIR = LHAPDF_PREFIX / "share" / "LHAPDF"


def _parse_info(text: str) -> "dict[str, str]":
    """Parse the scalar ``key: value`` entries of an LHAPDF ``.info`` body.

    The ``.info`` files are YAML, but every field we need is a plain scalar on
    one line, so a dependency-free reader is enough (list-valued keys such as
    ``Flavors`` are skipped).

    Args:
        text: Contents of a ``.info`` file.

    Returns:
        dict: Scalar ``key -> value`` pairs.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep or key.startswith(("#", " ")):
            continue
        value = value.strip()
        if value and not value.startswith("["):
            out[key.strip()] = value
    return out


def _read_set_info(set_name: str = None) -> "dict[str, str]":
    """Read a set's ``.info`` metadata, wherever the grid currently lives.

    Prefer the tracked grid, then an extracted or installed copy, and finally
    the optional tarball.  The member count must not depend on the local
    LHAPDF installation carrying a separate copy of the set.

    Args:
        set_name: LHAPDF set name; defaults to :data:`JAM_SET`.

    Returns:
        dict: Scalar ``key -> value`` pairs, empty if the set cannot be found.
    """
    name = JAM_SET if set_name is None else set_name
    leaf = f"{name}.info"
    for root in (JAM_REPO_DATA_ROOT, JAM_LHAPDF_DATA_ROOT, LHAPDF_DATA_DIR):
        info_path = root / name / leaf
        if info_path.is_file():
            return _parse_info(info_path.read_text())

    if JAM_TARBALL.exists():
        import tarfile

        try:
            with tarfile.open(JAM_TARBALL, "r:gz") as tar:
                handle = tar.extractfile(f"{name}/{leaf}")
                if handle is not None:
                    return _parse_info(handle.read().decode())
        except (tarfile.TarError, KeyError, OSError):
            return {}
    return {}


#: Raw ``.info`` metadata for :data:`JAM_SET`, empty when the set is missing.
JAM_SET_INFO = _read_set_info()

#: Member count and QMin are properties *of the set*, so they are read from its
#: ``.info`` file rather than hardcoded -- hardcoding silently desynchronizes
#: whenever the truth set is swapped (JAM20 had 196 members, JAM24 has 792).
#: Both are 0/None when the set is not installed; callers skip in that case.
JAM_N_MEMBERS = int(JAM_SET_INFO.get("NumMembers", 0))

#: Advertised set QMin, recorded for metadata.  Requested truth scales are
#: evaluated directly; LHAPDF extrapolates outside the advertised grid.
JAM_QMIN = float(JAM_SET_INFO["QMin"]) if "QMin" in JAM_SET_INFO else None

#: One representable closure truth.  All linear and nonlinear fake data are
#: folded from this same member; L1 universes vary only measurement/systematic
#: noise.  Replica-ensemble moments are used only to define the fixed lattice
#: covariance and are never used as a nonlinear central value.
FIXED_TRUTH_MEMBER = 1

#: Replica members used solely to calibrate the fixed lattice covariance.
#: Member 0 is the set's central value and is excluded.
JAM_REPLICA_MEMBERS = tuple(range(1, JAM_N_MEMBERS))

# -- perturbative scheme -----------------------------------------------------

MC = 1.28                # charm mass = input scale
MB = 4.18
Q0 = MC                  # input / parameterization / generation scale
Q0_2 = Q0 * Q0
NF = 4                   # active flavors at the input scale (u,d,s,c)
ORDER = "NLO"
MODE = "truncated"
ALPHAS_MZ = 0.118


#: Original JAM scales used to build distinct truth members.  Each is read from
#: JAM at this Q, then assigned as the truth at ``Q0 = mc``.  ``Q = 1`` is
#: evaluated directly through LHAPDF even though it is below ``JAM_QMIN``.
TRUTH_Q_CHOICES = {
    "mc": MC,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
}
DEFAULT_TRUTH_Q = "2"

# -- PDF grid ----------------------------------------------------------------

# Log-linear grid, 146 points, piecewise-quintic (pw5) elements.  Changed 2026-09-09
# from 128-point cubic_spline.
#
# **Representation gain, measured against the real JAM24 truth** (Q=2, member 1;
# sampled at the nodes, read back on a dense x set, worst field, max |basis - JAM|
# relative to that field's peak):
#
# .. code-block:: text
#
#     128 cubic_spline   4.20e-04   <- previous setting
#     146 pw5            1.23e-04   <- 3.4x better; cubic needs n ~ 200-240 to match
#
# The gain is *not* uniform, and that matters when reading a per-field result: it is
# concentrated entirely in the steep small-x singlet-class fields -- t8 3.4x, sigma
# 3.7x, t15 3.3x, g 4.8x -- while t3 and all four valence fields (v3, v8, v, v15) are
# unchanged at 1.0-1.1x.  Those five are already resolved at 128 nodes, so a finer
# basis buys them nothing.
#
# **146, not 145.**  A quintic tiles the grid only when ``(n - 1) % 5 == 0``:
# ``PiecewisePolynomial`` groups the intervals into cells of ``order`` and drops the
# leftover ``(n - 1) % order`` intervals into a single ragged final cell of *lower*
# degree.  145 leaves ``144 % 5 = 4`` -- a degree-4 cell at large x, where the DIS and
# DY rows carry the most weight.  146 gives 145 intervals = 29 exact quintic cells.
# Measured on the polynomial-exactness probe (interpolate ``x^5 - 2x^3 + 1``, which a
# tiling quintic reproduces to roundoff): 146 -> 8.9e-16, 141 -> 1.0e-15,
# 145 -> 1.1e-09, i.e. the ragged cell costs ~6 orders.  On the *JAM* truth above the
# 145/146 difference is nil (1.226e-04 both), because there the error is set by
# small-x resolution rather than by the large-x cell -- so this is cheap insurance,
# not a measured rescue.  146 is also PIXEL's own pw5 rung for log-linear grids
# (``benchmarks/kernel_stress.py::PW5_LONG_SIZE``); its snapper maps the shared cubic
# anchor 145 to exactly 146 for a quintic.  141 also tiles, if a smaller grid is
# wanted.
#
# **The cubic-spline argument was outweighed, not refuted -- and it still costs.**
# The previous note recorded that PIXEL's x-space inverse-Mellin evolution needs a
# far longer (slower) contour for C0 "piecewise" bases.  pw5 is C0, so it still
# applies, and ``kernels.evolution.contour`` now takes its piecewise branch:
#
# .. code-block:: text
#
#     128 cubic_spline   basis_kind=spline                 target 1e-8   zmax 1622
#     146 pw5            basis_kind=piecewise-polynomial   target 1e-5   zmax 9111
#
# So the evolution/inverse-Mellin path is asking for a target three orders looser on
# a contour 5.6x longer, and PIXEL emits its "LOUD WARNING" on every adaptive_contour
# call.  Nothing here overrides that contour; the suites take the adaptive policy.
# The representation gain above is a property of the *basis*, measured without any
# evolution -- it does not speak to the evolved DIS/DY kernels.  **Measure those
# before treating this change as settled.**
GRID_N = 146
GRID_SPACING = "log-linear"
X_MIN = 1.0e-6
ELEMENT_TYPE = "piecewise_quintic"

# -- flavor basis (nf = 4) ---------------------------------------------------
# C-even (q+qbar) and C-odd valence (q-qbar) fields.  "strange" = T8/V8 and
# "charm" = T15/V15 are the SU(4) basis combinations, not isolated s / c PDFs.

EVEN_FIELDS = ("t3", "t8", "sigma", "t15", "g")
ODD_FIELDS = ("v3", "v8", "v", "v15")
ALL_FIELDS = EVEN_FIELDS + ODD_FIELDS

#: field -> its PIXEL canonical basis key (C-even and C-odd share basis keys).
FIELD_TO_BASIS = {
    "t3": "u_minus_d",
    "t8": "u_plus_d_minus_2s",
    "sigma": "singlet",
    "t15": "u_plus_d_plus_s_minus_3c",
    "g": "gluon",
    "v3": "u_minus_d",
    "v8": "u_plus_d_minus_2s",
    "v": "singlet",
    "v15": "u_plus_d_plus_s_minus_3c",
}

#: basis key -> field, for the DIS charge-weighted maps.
EVEN_MAP = {
    "u_minus_d": "t3",
    "u_plus_d_minus_2s": "t8",
    "singlet": "sigma",
    "u_plus_d_plus_s_minus_3c": "t15",
    "gluon": "g",
}
ODD_MAP = {
    "u_minus_d": "v3",
    "u_plus_d_minus_2s": "v8",
    "singlet": "v",
    "u_plus_d_plus_s_minus_3c": "v15",
}

#: (C-even, C-odd) quark channel pairs (gluon is C-even only).
FIELD_PAIRS = (("t3", "v3"), ("t8", "v8"), ("sigma", "v"), ("t15", "v15"))

#: human-readable lattice sector labels.
SECTOR_LABEL = {
    "t3": "isovector", "v3": "isovector",
    "t8": "strange (T8)", "v8": "strange (V8)",
    "sigma": "isosinglet", "v": "isosinglet valence",
    "t15": "charm (T15)", "v15": "charm (V15)",
    "g": "gluon",
}


def field_cparity(field: str) -> str:
    """Return ``"even"`` or ``"odd"`` for a field name."""
    return "even" if field in EVEN_FIELDS else "odd"


#: Fields are the momentum densities q = x*f.  The pseudo-ITD integrand weight
#: depends on the channel: the non-singlet quark reduced ITD is int cos(nu x) f
#: = int (cos/x) q  (alpha=-1), while the singlet and gluon ITDs are the cosine
#: transform of the momentum density itself, I = int cos(nu x) x*D = int cos q
#: (alpha=0; Balitsky-Morris-Radyushkin Eqs 4.8/2.6).  So only sigma and g use
#: alpha=0 for their ITD.
SINGLET_GLUON_FIELDS = ("sigma", "g")

# Every positive-grid forward operator must complete the unresolved interval
# 0 <= x < X_MIN.  The fitted fields store x*f(x): phenomenologically the
# singlet and gluon momentum densities may tend to a nonzero constant, whereas
# valence and flavor-nonsinglet combinations have a convergent zeroth moment and
# therefore vanish.  Use the minimal endpoint assumptions consistent with those
# limits: constant continuation for sigma/g, and a straight line to zero for all
# other fields.
LOW_X_LINEAR_POWER = 1.0


def vanishes_at_origin(field: str) -> bool:
    """True when theory predicts ``x*f(x) -> 0`` as ``x -> 0`` for this field.

    The single statement of the small-x limit for this suite: the singlet and
    gluon momentum densities may tend to a nonzero constant, while the valence
    and flavour-nonsinglet combinations have a convergent zeroth moment and so
    vanish.  Both the low-x completion and the near-hard origin constraint read
    it, so the two cannot disagree about what the physics says.

    .. warning::

       **The C-parity split is measurably wrong for ``t8`` and ``t15``, and this
       is recorded rather than fixed.**  See ``plans/low_x_head_diagnosis.md``
       ADDENDUM 2.  The *valence* half is a theorem -- ``VALENCE_NORMS`` are
       finite, so ``int (q/x) dx`` exists only if ``a > 0``, and ``a > 0`` was
       measured in all 96 (set, truth, Q) combinations.  The *flavour-nonsinglet*
       half is not: ``t8 = u+ + d+ - 2 s+`` vanishes only once ``s+`` has
       equilibrated with the light sea at that ``Q``, and ``t15`` likewise needs
       ``c+`` to catch up.  Measured at ``x ~ 1.1e-6``: NNPDF ``s+/u+ = 0.995``
       (so its ``t8`` *is* a genuine non-singlet, ``a = +0.33``) but
       ``c+/u+ = 0.215``, giving ``a(t15) = -0.080`` against ``a(sigma) = -0.119``
       -- ``t15`` is a *singlet-class* field there.  JAM member 1 has
       ``s+/u+ = 0.232``, so both ``t8`` (``a = -0.283``) and ``t15``
       (``a = -0.234``) track ``sigma`` (``a = -0.208``).

       Consequences, measured, before anyone "fixes" this:

       * these fields get ``low_x_completion -> power(alpha=1)``, a *vanishing*
         head, where the truth *rises*;
       * the physically-correct completion cannot simply be substituted, because
         ``itd_momentum_density`` splits on the same C-parity rule and gives them
         the pseudo-ITD weight ``a_k = -1``.  Measured through
         ``kernels.lowx.check_low_x_integrable``: ``power(alpha)`` RAISES at
         ``a_k = -1`` for every ``alpha <= 0`` and passes for ``alpha >= 0.3``.
         So correcting the completion for ``t8``/``t15`` requires moving them to
         ``a_k = 0`` in the same change -- ``int f dx`` genuinely diverges for a
         rising field and there is no number sum rule to converge to;
       * the Mellin-moment operators are **not** exposed: a row of order ``n``
         carries its own ``x^(n-1)``, and this suite builds ``n = 2``/``n = 3``
         only.  Measured on the 145-node grid, swapping the completion between
         ``power(1)``, ``flat`` and a rising ``power(-0.3)`` moves the folded
         moment by at most ``9.1e-09`` (``n = 2``) and ``9.1e-15`` (``n = 3``) --
         see ``tests/test_closure_constraints.py::
         test_mellin_moment_convention_forces_momentum_density_on_every_field``;
       * **the bare pseudo-ITD rows are exposed, and heavily.**  Fraction of the
         ``PseudoITDReal`` row value supplied by the ``[0, x_min)`` completion,
         measured at ``nu = 1, 2, 4, 8`` by differencing against a
         head-suppressing ``power(50)`` completion, on the ``_small`` suites'
         bare (unevolved) transform:

         .. code-block:: text

            field     replica-ensemble mean      JAM member 1 (FIXED_TRUTH_MEMBER)
            t8        0.409 .. 0.404             0.198 .. 0.212
            t15       0.290 .. 0.298             0.174 .. 0.184
            t3        8.8e-03 .. 3.5e-02         2.8e-03 .. 1.3e-02
            v3/v8/v/v15  2.3e-04 .. 3.5e-03      6.3e-04 .. 4.9e-03
            sigma/g   4.4e-06 .. 9.0e-04         1.7e-05 .. 3.2e-04

         So roughly a fifth to two fifths of the ``t8``/``t15`` lattice signal
         comes from a region whose assumed shape contradicts the measured local
         slope there.  Choosing ``FIXED_TRUTH_MEMBER`` over the ensemble mean
         roughly halves it but does not remove it.  The full suites run this
         transform through Mellin space (``PITD_EVOLVE = True``) and are far less
         exposed; the ``_small`` suites do not set ``PITD_EVOLVE`` at all.

       Changing this rule changes every forward operator in the suite,
       invalidates the kernel cache, and requires regenerating committed data.
       It is an owner decision, not a cleanup.
    """
    if field not in ALL_FIELDS:
        raise KeyError(f"unknown closure field {field!r}")
    return field not in SINGLET_GLUON_FIELDS


def low_x_completion(field: str):
    """Return the full-closure low-x completion for one ``x*f`` field.

    ``None`` when the grid reaches the origin: there is then no ``[0, x_min)`` gap to
    complete, the field is resolved there by its own basis elements, and passing a
    completion anyway is rejected by the domain rule.  In that regime the
    vanishing limit is instead imposed by the ``cons_origin_*`` pseudo-datum.
    """
    vanishes = vanishes_at_origin(field)  # validates the field name first
    if make_grid().as_array()[0] == 0.0:
        return None
    if not vanishes:
        return "flat"
    return {"kind": "power", "alpha": LOW_X_LINEAR_POWER}


def even_low_x_completions() -> dict:
    """Canonical C-even DIS component -> completion settings."""
    return {basis: low_x_completion(field) for basis, field in EVEN_MAP.items()}


def odd_low_x_completions() -> dict:
    """Canonical C-odd DIS component -> completion settings."""
    return {basis: low_x_completion(field) for basis, field in ODD_MAP.items()}


def itd_momentum_density(field: str) -> bool:
    """True (alpha=-1) for non-singlet quark ITD; False (alpha=0) for sigma/g."""
    return field not in SINGLET_GLUON_FIELDS


# -- pseudo-ITD evolution and short-distance matching ------------------------
#: Evolve the fitted field from ``Q0 = mc`` to the separation-dependent matching
#: scale and apply short-distance matching at ``PITD_MATCHING_ORDER``, rather
#: than reading the LO reduced ITD directly off the field.  Generation and the
#: fit share the physical transform.  Lattice-artifact nuisance fields remain
#: bare measurement-level transforms and are not assigned DGLAP evolution.
PITD_EVOLVE = True

#: Order of the short-distance matching coefficients.  ``"NLO"`` is the
#: ``O(alpha_s)`` matching: ``C_F [ ln(lam^2 e^{2 gamma_E + 1}/4) B(N) + L(N) ]``,
#: i.e. the Altarelli-Parisi log *and* the finite matching constant ``L``.
PITD_MATCHING_ORDER = "NLO"

#: Lattice pseudo-ITDs in this closure are reduced ratios, with short-distance
#: coefficients defined in the MSbar operator scheme.  Keep this explicit so a
#: future RI/MOM or hybrid closure cannot accidentally reuse these coefficients.
PITD_OBSERVABLE = "reduced_ratio"
PITD_OPERATOR_SCHEME = "msbar"
PITD_KERNEL = "unpolarized"
PITD_DATA_NORMALIZATION = "a1_prescaled"
PITD_LORENTZ_COMPONENT = "temporal"

#: Dimensionless matching-scale factor: ``mu(z) = PITD_MATCHING_LAMBDA / z`` in
#: physical units, so ``z^2 mu^2 = lam^2`` and the matching log is constant
#: across separations while the z-dependence sits in the evolution operator.
#:
#: **``mu^2`` falls as z grows, and this is the accuracy ceiling of the lattice
#: sector.**  Measured over the thirteen NME ensembles at ``lam = 1`` with
#: ``z <= 0.3 fm``: ``mu^2`` spans ``[0.32, 13.8] GeV^2`` and ``alpha_s`` reaches
#: **0.88** at the largest separation -- above the Landau pole
#: (``Lambda_QCD(nf=3)^2 = 0.154``), so the operator builds, but the
#: ``O(alpha_s)`` term there is not a small correction to the leading one.
#: Raising ``lam`` to ~1.95 keeps ``mu^2 >= Q0^2`` everywhere (``alpha_s <=
#: 0.42``) at the cost of a ``ln ~ 2.2`` matching log; that trade is a physics
#: choice, so it is a named constant here rather than a buried default.
PITD_MATCHING_LAMBDA = 1.0

# -- lattice ensembles (arXiv:2601.10857 Table I) ----------------------------
# The full-scale suite mocks the thirteen 2+1-flavor Wilson-clover NME ensembles.
# Each ensemble sets its own pseudo-ITD kinematics and forward/systematic
# metadata; the four lattice spacings, the volume pairs, and the four pion masses
# are exactly the continuum / finite-volume / chiral axes the systematics probe.

HBARC_GEV_FM = 0.1973269804

#: pseudo-ITD kinematic rules (per closure_extension.md):
#:   z (lattice units) = 1 .. ceil(Z_PHYS_MAX_FM / a)
#:   p (integer index) = 1 .. floor((L/a) / P_INDEX_DIVISOR)
Z_PHYS_MAX_FM = 0.3
P_INDEX_DIVISOR = 6


class Ensemble:
    """One lattice ensemble: geometry + spacing + pion mass (Table I row)."""

    __slots__ = ("id", "L_sites", "T_sites", "beta", "a_fm", "m_pi_mev")

    def __init__(self, id, L_sites, T_sites, beta, a_fm, m_pi_mev):
        self.id = id
        self.L_sites = int(L_sites)          # spatial extent in sites (= L/a)
        self.T_sites = int(T_sites)
        self.beta = float(beta)
        self.a_fm = float(a_fm)              # lattice spacing [fm]
        self.m_pi_mev = float(m_pi_mev)

    @property
    def m_pi_gev(self) -> float:
        return self.m_pi_mev / 1000.0

    @property
    def L_fm(self) -> float:
        return self.L_sites * self.a_fm

    @property
    def z_max(self) -> int:
        """Largest separation index: ceil(0.3 fm / a)."""
        return int(math.ceil(Z_PHYS_MAX_FM / self.a_fm))

    @property
    def p_max(self) -> int:
        """Largest momentum index: floor((L/a) / 6)."""
        return int(math.floor(self.L_sites / P_INDEX_DIVISOR))

    @property
    def z_values(self) -> tuple:
        return tuple(range(1, self.z_max + 1))

    @property
    def p_values(self) -> tuple:
        return tuple(range(1, self.p_max + 1))

    @property
    def pz_unit_gev(self) -> float:
        """Physical momentum of unit index: pz = 2*pi*hbarc / (L_sites * a) [GeV]."""
        return TWO_PI * HBARC_GEV_FM / self.L_fm

    def __repr__(self):
        return (f"Ensemble({self.id!r}, L={self.L_sites}, a={self.a_fm}, "
                f"m_pi={self.m_pi_mev} MeV, z<= {self.z_max}, p<= {self.p_max})")


#: The thirteen NME ensembles (id, L_sites, T_sites, beta, a_fm, m_pi_mev).
LATTICE_ENSEMBLES = (
    Ensemble("a117m310", 32, 96, 6.1, 0.117, 310),
    Ensemble("a087m290", 32, 64, 6.3, 0.087, 289),
    Ensemble("a087m290L", 48, 128, 6.3, 0.087, 289),
    Ensemble("a087m230", 48, 128, 6.3, 0.087, 233),
    Ensemble("a087m230X", 48, 128, 6.3, 0.087, 228),
    Ensemble("a086m180", 48, 96, 6.3, 0.086, 179),
    Ensemble("a086m180L", 64, 128, 6.3, 0.086, 179),
    Ensemble("a068m290", 48, 128, 6.5, 0.068, 290),
    Ensemble("a068m230", 64, 192, 6.5, 0.068, 234),
    Ensemble("a068m175", 72, 192, 6.5, 0.068, 175),
    Ensemble("a067m135", 96, 192, 6.5, 0.067, 135),
    Ensemble("a053m295", 64, 192, 6.7, 0.053, 295),
    Ensemble("a053m230", 72, 192, 6.7, 0.053, 226),
)

ENSEMBLES_BY_ID = {e.id: e for e in LATTICE_ENSEMBLES}

#: Optional restriction to a subset of ensembles for tractable dev runs.  Set to
#: a tuple of ensemble ids (e.g. ("a067m135",)) or None for all thirteen.  The
#: PIXEL_CLOSURE_ENSEMBLES env var (comma-separated ids) overrides this.
ACTIVE_ENSEMBLE_IDS = None


def active_ensembles() -> tuple:
    """The ensembles to generate/fit, honouring ACTIVE_ENSEMBLE_IDS + env override."""
    import os

    override = os.environ.get("PIXEL_CLOSURE_ENSEMBLES")
    if override:
        ids = [s.strip() for s in override.split(",") if s.strip()]
    elif ACTIVE_ENSEMBLE_IDS:
        ids = list(ACTIVE_ENSEMBLE_IDS)
    else:
        return LATTICE_ENSEMBLES
    return tuple(ENSEMBLES_BY_ID[i] for i in ids)


# -- Mellin moments ----------------------------------------------------------
#: Mellin moments at 2 GeV.  The data builder fixes CP from exponent parity:
#: <x> (stored order N=2) uses the C-odd field and <x^2> (N=3) the C-even
#: field. Thus "moments up to N=3" measures <x> on the valence fields and <x^2>
#: on the C-even fields (N=1 is the exactly-known counting moment, kept as a
#: constraint rather than lattice data).
MOMENT_Q2 = 4.0                                   # (2 GeV)^2
CP_EVEN_MELLIN_ORDER = 3                          # <x^2> -> C-even field
CP_ODD_MELLIN_ORDER = 2                           # <x>   -> C-odd field


def mellin_order_for(field: str) -> int:
    """Configured Mellin power for one physical closure field."""
    return (
        CP_EVEN_MELLIN_ORDER
        if field_cparity(field) == "even"
        else CP_ODD_MELLIN_ORDER
    )

# -- lattice covariance ------------------------------------------------------
#: The lattice fake-data covariance is the FULL replica covariance of the folded
#: observable ("integrate each true sample"): fold every JAM replica, take the
#: sample covariance (with off-diagonals).  That covariance is rank-deficient /
#: ill-conditioned (the folded smooth observables are collinear), so inflate the
#: diagonal by 1% to regularize it: C_ii -> C_ii * (1 + COV_DIAGONAL_INFLATE).
COV_DIAGONAL_INFLATE = 0.01

# -- injected lattice systematics --------------------------------------------
#: Active systematic keys (LATTICE_SYSTEMATICS registry).  Pseudo-ITD finite
#: volume uses the separation-dependent ``inf_Lz`` coefficient, while Mellin
#: moments use the separation-free ``inf_L`` coefficient.  Their nuisance truths
#: are independent: pseudo-ITD keys get x-space curves and each Mellin power gets
#: a scalar field.  Every ensemble sees the appropriate truth through its own
#: coefficient ``coeff_k(meta)``.
ITD_SYSTEMATICS = ("ht", "chiral", "inf_Lz", "cont_invz", "cont_aL")
MOMENT_SYSTEMATICS = ("chiral", "inf_L", "cont_aL")

#: Enforced systematic-shape properties (see closure_extension_plan.md):
#:  * parity in nu  -> automatic (real=cosine even, imag=sine odd);
#:  * converge nu->inf -> automatic (Riemann-Lebesgue);
#:  * zero shift at nu=0 (real) AND no error on the known counting moment ->
#:    null the n=1 Mellin moment of every systematic curve;
#:  * <= 10% of the leading PDF signal -> rescale to this cap.
SYSTEMATIC_NULL_MOMENT_ORDER = 1                  # zero this Mellin moment (alpha=-1)
SYSTEMATIC_MAX_FRACTION = 0.10                    # |sys shift| <= 10% |PDF signal|
SYSTEMATIC_SEED = 20260721
MOMENT_SYSTEMATIC_SEED = SYSTEMATIC_SEED + 1

#: Zero-mean log-RBF GP prior for the nuisance fields (hyperparameters frozen).
NUISANCE_GP_SIGMA = 1.0


# -- DIS experimental preset -------------------------------------------------
# The full-scale suite audits EVERY fitpack IDIS table (dis_audit globs the
# directory) and keeps the full retained kinematic coverage -- no subsample --
# so the fake data have the exact real (x, Q2) coverage.  F2 (p/n/d) and NC
# sigma_r (p/n/d) and HERA CC sigma_r are modelled directly.  The HERA CC rows
# use the massless NLO W2/WL/W3 operator validated against Fitpack; this closure
# remains distinct from the gated NNLO FONLL-C/TMC NNPDF-theory-200 campaign.

class ExpSpec:
    """A DIS-table kind/label hint for a specific table index.

    dis_audit discovers every table by globbing the directory and infers the
    closure ``kind`` from each sheet's ``obs``/``current`` columns.  These specs
    only supply preferred labels and explicit charged-current identities.
    """

    __slots__ = ("idx", "obs", "target", "kind", "label")

    def __init__(self, idx, obs, target, kind, label):
        self.idx = idx
        self.obs = obs
        self.target = target
        self.kind = kind          # "f2" | "sigma_r" | "sigma_r_cc"
        self.label = label

    def __repr__(self):
        return f"ExpSpec({self.idx}, {self.kind}, {self.target}, {self.label!r})"


#: **Union selection.**  The full suites fit everything either analysis uses, so this
#: is JAM's own idis list (the NNPDF DIS choices are a subset of it apart from CHORUS
#: and the bottom set, which enter through the contract selection).  10003 and 10021
#: are audited but generate no rows: every 10003 row fails the Q2/W2 cuts and 10021
#: is a ratio with no builder.
#:
#: The DIS tables the *analysis* uses, i.e. the exact kinematic coverage of the
#: real fit (uncommented IDIS entries in fitpack_legacy/examples/input.py).
#: dis_audit still scans every workbook and records its status, but fake data are
#: generated only for these (per closure_extension.md: "exact kinematic coverage
#: used in the analysis").  10021 is a nonlinear d/p F2 ratio whose multirow
#: flow domain is pending, so it is audited but not generated.
ANALYSIS_IDIS_IDX = (
    10010, 10016, 10020,                          # proton F2  (SLAC, BCDMS, NMC)
    10011, 10017,                                 # deuteron F2 (SLAC, BCDMS)
    10003, 10007,                                 # proton sig_r (JLab Hall C, HERMES)
    10026, 10027, 10028, 10029, 10030,            # proton sig_r (HERA II NC e+/e-)
    10031, 10032,                                 # proton sig_r (HERA II CC)
    10021,                                        # d/p F2 ratio (NMC) -> unsupported
)

#: HERA charged-current reduced-cross-section table identities.
CC_SIGMA_R_IDX = (10031, 10032)

#: Optional per-idx label overrides (else dis_audit builds a label from the sheet).
EXP_LABEL_OVERRIDES = {
    10010: "slac_p_f2", 10016: "bcdms_p_f2", 10020: "nmc_p_f2",
    10011: "slac_d_f2", 10017: "bcdms_d_f2",
    10003: "jlab_hallc_p_sigmar", 10007: "hermes_p_sigmar",
    10026: "hera_nc_ep_318_sigmar_1", 10027: "hera_nc_ep_300_sigmar",
    10028: "hera_nc_ep_251_sigmar", 10029: "hera_nc_ep_225_sigmar",
    10030: "hera_nc_em_318_sigmar",
    10031: "hera_cc_ep_318_sigmar", 10032: "hera_cc_em_318_sigmar",
    10021: "nmc_dp_f2_ratio",
}

#: DIS selection cuts (match the legacy unpolarized preset; keep evolution
#: upward from the input scale).  ``None`` means there is no upper-Q2 cap, so the
#: HERA high-Q2 DIS points remain eligible.
DIS_Q2_MIN = Q0_2                                 # >= input scale
DIS_Q2_MAX = None                                 # no upper-Q2 cap
DIS_W2_MIN = 10.0
#: Full coverage: keep EVERY retained row (no subsample).  None disables thinning.
MAX_POINTS_PER_EXP_DATASET = None
EXP_ABS_FLOOR = 1.0e-6

# -- DIS experimental nuisances ----------------------------------------------
# The fitpack tables carry the real nuisances alongside the statistical error:
# an overall multiplicative normalization (``*norm_c``: 1.7%-7.6% depending on
# the experiment) and correlated systematic sources (``%*_c``: HERA ships 169
# per table, BCDMS 5, NMC 11).  dis_audit records both; generation injects a
# realization and the fit marginalizes the identical directions, so this is the
# DIS analogue of the injected-and-recovered lattice systematics.

#: Injected normalization offsets per DIS table idx, in prior-sigma
#: (``rel_norm``) units.  Mixed signs/sizes so a recovered pull cannot be a
#: global fluke; tables not listed here get no offset.
DIS_NORM_BETA = {
    10010: 1.0,      # SLAC p F2      (2.1%)
    10011: -0.8,     # SLAC d F2      (1.7%)
    10016: 0.5,      # BCDMS p F2     (3%)
    10017: -1.2,     # BCDMS d F2     (3%)
    10020: 1.5,      # NMC p F2       (2%)
    10007: -0.7,     # HERMES p sig_r (7.6%)
}

#: NNPDF t0 iteration: multiplicative uncertainties are referenced to the theory
#: prediction rather than the data (referencing the data biases the fit low --
#: D'Agostini).  The reference is unknown before fitting, so it is iterated to
#: convergence.  0 disables the iteration (data-referenced, biased).
T0_MAX_ITERATIONS = 12
T0_TOLERANCE = 1.0e-4

#: Seed for the drawn correlated-systematic amplitudes (one N(0,1) per source).
EXP_SYSTEMATIC_SEED = 20260724

#: Fit the DIS normalization as a multiplicative parameter (row factor
#: ``1 + A rel``, ``A ~ N(0, 1)``) instead of marginalizing it into the
#: covariance.  ON: the suites fit normalizations, matching DY_FIT_NORMALIZATION
#: so both experimental families are treated identically.  The correlated
#: systematics stay marginalized either way -- only the normalization is fitted.
DIS_FIT_NORMALIZATION = True


def dis_norm_beta(idx) -> float:
    """Injected normalization offset (in ``rel_norm`` units) for a DIS table."""
    return float(DIS_NORM_BETA.get(int(idx), 0.0))

# -- Drell-Yan experimental preset -------------------------------------------
# Fixed-target Drell-Yan (E866/NuSea) constrains the light-quark SEA (dbar-ubar)
# that inclusive DIS pins only weakly.  10001 (pp) and 10002 (pd) are modelable;
# the nuclear pA (12881-3) and pd/pp ratio (20001-2) tables are unsupported, and
# SQ_Acceptance is not data.  dy_audit converts (RS, Rtau, xF) -> (S, Q2, Y) and
# records the real rel_stat, exactly like the DIS audit.

#: idx -> (reaction, label) for the modelable DY tables (the analysis-used set).
#: **E866 pp absolute only.**  Verified from the workbooks: 10001 is E866 ``pp``
#: (M^3 dsig/dM dxF, 184 rows), 10002 is E866 ``pd`` (191 rows), and 20001/20002 are the
#: E866/E906 ``sigpd/2sigpp`` ratios.  JAM's ``dy`` list is 10001/20001/20002 and
#: NNPDF4.0 carries E866 as ``DYE886P`` (pp absolute) + ``DYE886R`` (the ratio), so
#: **neither analysis fits E866 pd absolute**.  It is therefore in neither selection nor
#: in their UNION, which is what the full suites now run.  E605 (copper) enters through
#: the contract selection instead, not through this fitpack list.
DY_ANALYSIS = {
    10001: ("pp", "dy_e866_pp"),
}

#: DY kinematic cuts.  Require physical parton momentum fractions x1, x2 < 1 and
#: keep the perturbative Q2 (fixed-target DY is already Q2 >> Q0^2).
DY_X_MAX = 1.0
DY_Q2_MIN = Q0_2

#: Drell-Yan is *bilinear*: the model materializes a dense (n_rows, n, n) tensor
#: per contribution, and the full-basis x full-channel luminosity expands to
#: ~460 contributions, so memory scales with the DY row count.  The full closure
#: deliberately keeps every row surviving the physics cuts -- every row of each E866
#: table since ``Rtau`` is read as ``sqrt(tau)`` (``dy_audit.tau_from_rtau``,
#: 2026-09-11); the contract's ``selected_rows`` records the count.  Set an integer only
#: for an explicitly thinned run.
DY_MAX_ROWS = None

#: DY hard-kernel scheme.  **NLO as of 2026-08-14** (was ``"LO"``).  The reason
#: LO stood -- that NLO off-scale corrections overshoot (large/negative at edge
#: kinematics) -- was an artefact of evaluating the hard kernel at ``Q^2`` while
#: the fields sat unevolved at ``mc``, and those logarithms are what
#: :data:`DY_EVOLVE` now resums, so the objection no longer applies in this suite.
#: (It still applies in ``closure_*_truth_small``, which run ``DY_EVOLVE = False``
#: and are therefore left at LO.)  Both sides use this identically, so the closure
#: stays self-consistent either way.
#:
#: Three consequences follow, none of which any test asserts, so they are recorded
#: here rather than left to be rediscovered:
#:
#: * :data:`DY_CHANNELS` below now activates the gluon-initiated channels, and
#:   ``Cqg2`` is a known open defect: **+82% on the channel total, 1.8-4.3% on the
#:   observable** (``plans/drell_yan_qg_bug.md``), shared with fitpack so the usual
#:   oracle cannot see it.  A closure test still *closes* -- generation and fit use
#:   the identical wrong kernel -- but the generated truth stops being physical at
#:   that level, and any statement about DY absolute normalization inherits it.
#: * **Cost: the DY bilinear tensor is ~14x dearer.**  Measured 2026-08-14 on the
#:   kernel path, 3 rows on a 7-node quadratic basis at ``quad_n=20``: LO 0.001 s,
#:   NLO 0.632 s.  At LO the node count was both free and bitwise irrelevant
#:   (identical tensor at ``quad_n`` 5/20/40); neither holds at NLO, where 5 -> 20
#:   moves the tensor by ``3.9e-02``.
#: * **Accuracy: ``z_integration`` starts to matter, and is deliberately not
#:   changed here.**  The shipped ``legacy_linear`` at ``quad_n=20`` carries
#:   ``~5e-2`` on the finite-element tensor at NLO and does not converge in
#:   ``quad_n`` (still ``6.2e-3`` at 96); ``z_integration="element_aligned"`` with
#:   ``quad_n=8`` reaches ``9.2e-4`` at the same wall cost
#:   (``plans/dy_quad_n_convergence.md`` section 7).  That switch is opt-in because
#:   fitpack parity and the E866 goldens are pinned to ``legacy_linear``; at LO it
#:   changed nothing, at NLO it is the dominant quadrature choice.
DY_ORDER = "NLO"
#: Fixed couplings for the photon-only fixed-target arm (E866, E605).  Collider records
#: are gamma*/Z and take running couplings instead -- ``datasets.collider_exchange``
#: (owner decision D8, 2026-09-11) -- so these two do not reach them.
DY_ALPHA_EM = 1.0 / 137.035999
DY_ALPHA_S = 0.20                                 # fixed alpha_s; fixed-target DY only -- collider rows run alpha_s(Q^2), see collider_exchange

#: Compose the flavour-resolved PDF evolution operator into every DY hard tensor,
#: on **both** the generation and the fit side.  The fitted fields live at
#: ``Q0 = mc``; an E866 row sits at ``Q^2 ~ 170-650``, one to five e-folds above
#: it, and a weak boson would be eight.  Leaving it off is a documented
#: approximation that stays self-consistent (generation and fit share the same
#: wrong operator) but makes the absolute cross section unphysical, so it is now
#: on by default.  Turning it off restores the historical fixed-scale operator.
DY_EVOLVE = True

#: Perturbative order/mode of the *evolution* operator.  Independent of
#: :data:`DY_ORDER`, which is the hard-kernel order.  The two are now both NLO,
#: but they are still separate knobs: the off-scale NLO hard corrections that
#: once motivated ``DY_ORDER = "LO"`` were an artefact of *not* evolving, so it
#: is :data:`DY_EVOLVE` -- not this constant -- that made NLO hard matching the
#: consistent choice here.
DY_EVOLUTION_ORDER = ORDER
DY_EVOLUTION_MODE = MODE

#: Active flavours in the DY luminosity.  Evolving a fixed-target or boson row
#: from ``Q0 = mc`` crosses the bottom threshold, so the projected parton set is
#: five-flavour and ``b``/``bbar`` carry a real luminosity; without evolution the
#: fields stay in their Nf=4 input basis.  ``datasets`` asserts the projection
#: agrees with this before building.
DY_NF = 5 if DY_EVOLVE else 4
DY_CHANNELS = (("qA,qbB", "qbA,qB") if DY_ORDER == "LO" else
               ("qA,qbB", "qbA,qB", "qA,gB", "gA,qB"))


def dy_low_x_completions():
    """Per-source-field low-x completions for the DY evolution operators.

    Single-sourced from :func:`low_x_completion`, so the evolution operators
    complete ``[0, x_min)`` exactly the way every other forward operator in this
    suite does.  Fields sharing a completion share one assembled block.
    """
    from pixel.kernels.evolution.flavor_transition import NF4_SOURCE_FIELDS

    return {name: low_x_completion(name) for name in NF4_SOURCE_FIELDS}


def dy_evolution_cache_dir():
    """Cache directory for the truth-independent DY evolution operators.

    Namespaced by the settings that change the matrices but do not appear in the
    per-scale filenames, so two orders cannot land on one path.  The operators
    depend on kinematics and the input scale, never on which truth is folded, so
    every ``truthQ_*`` member reuses this.
    """
    return KERNEL_CACHE_ROOT / "dy_evolution" / f"{DY_EVOLUTION_ORDER}_{DY_EVOLUTION_MODE}"

# -- Drell-Yan normalization nuisance ----------------------------------------
# Each E866 table carries one overall luminosity uncertainty (``%norm_c``, 7%),
# recorded per table by dy_audit as ``rel_norm``.  Generation multiplies the
# folded central values by ``1 + beta_true * rel_norm`` and the fit marginalizes
# the same relative size into the covariance, so the injected offset must come
# back out of ``Model.nuisance_pulls`` -- the DY analogue of the injected-and-
# recovered lattice systematics.

#: Injected truth offsets per DY table idx, in prior-sigma (``rel_norm``) units.
#: Different signs/sizes per table so a recovered pull cannot be a global fluke.
DY_NORM_BETA = {10001: 1.0, 10002: -0.6}

#: Treat the DY normalization as a fitted parameter (row factor ``1 + A rel``)
#: instead of marginalizing it into the covariance.  ON, matching
#: DIS_FIT_NORMALIZATION.  Note the Hubbard-Stratonovich caveat: a bilinear
#: dataset's tensor reaches the evidence only through ``T = 2i f Y``, so the
#: amplitude is informed by the ``f`` integral rather than by the MAP gradient
#: (which vanishes at ``f = 0``).  Expect the DY normalization to be far less
#: constrained than the DIS ones; see the README.
DY_FIT_NORMALIZATION = True


def dy_norm_beta(idx) -> float:
    """Injected normalization offset (in ``rel_norm`` units) for a DY table."""
    return float(DY_NORM_BETA.get(int(idx), 0.0))


def _parton_basis_maps():
    """Compute the proton and deuteron ``parton -> {basis_field: coeff}`` maps.

    Each physical parton momentum density is a fixed linear combination of the
    nine closure basis fields.  The C-even part (``f + fbar``) projects onto
    ``t3,t8,sigma,t15`` and the C-odd part (``f - fbar``) onto ``v3,v8,v,v15``;
    ``f = (qp + qm)/2``, ``fbar = (qp - qm)/2``.  The deuteron is the isoscalar
    average (``u <-> d`` swapped), used for the ``pd`` side.
    """
    import numpy as np

    even = ["t3", "t8", "sigma", "t15"]
    odd = ["v3", "v8", "v", "v15"]
    # rows t3,t8,sigma,t15 in terms of (qp_u, qp_d, qp_s, qp_c):
    m = np.array([[1, -1, 0, 0], [1, 1, -2, 0],
                  [1, 1, 1, 1], [1, 1, 1, -3]], dtype=float)
    minv = np.linalg.inv(m)
    flav = ["u", "d", "s", "c"]

    def combo(row, fields, sign):
        return {fields[j]: sign * minv[i, j] for j in range(4)
                if abs(minv[i, j]) > 1e-12}
    proton = {}
    for i, f in enumerate(flav):
        qp = combo(i, even, 1.0)
        qm = combo(i, odd, 1.0)
        proton[f] = _merge(qp, qm, 0.5, 0.5)        # f    = (qp + qm)/2
        proton[f + "b"] = _merge(qp, qm, 0.5, -0.5)  # fbar = (qp - qm)/2
    proton["g"] = {"g": 1.0}

    partner = {"u": "d", "d": "u", "ub": "db", "db": "ub"}
    deuteron = {}
    for p, cmap in proton.items():
        q = partner.get(p)
        deuteron[p] = _merge(cmap, proton[q], 0.5, 0.5) if q else dict(cmap)
    return proton, deuteron


def _merge(a, b, wa, wb):
    out = {}
    for k, v in a.items():
        out[k] = out.get(k, 0.0) + wa * v
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + wb * v
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


#: parton -> {basis_field: coeff} maps for the proton and (isoscalar) deuteron.
DY_PARTON_BASIS_PROTON, DY_PARTON_BASIS_DEUTERON = _parton_basis_maps()
_DY_CONJUGATE = {
    "u": "ub", "ub": "u", "d": "db", "db": "d", "s": "sb", "sb": "s",
    "c": "cb", "cb": "c", "b": "bb", "bb": "b", "g": "g",
}
DY_PARTON_BASIS_ANTIPROTON = {
    parton: dict(DY_PARTON_BASIS_PROTON[_DY_CONJUGATE[parton]])
    for parton in DY_PARTON_BASIS_PROTON
}

#: Copper, for E605 (``DYE605_dw_ite``).  Natural Cu: A = 63.546, Z = 29.
DY_COPPER_A, DY_COPPER_Z = 63.546, 29.0
DY_COPPER_PROTON_FRACTION = DY_COPPER_Z / DY_COPPER_A


def nuclear_parton_basis(proton_fraction: float) -> dict:
    r"""``Z/A``-weighted nuclear parton map, the static counterpart of the evolved one.

    ``q_A = (Z/A) q_p + (N/A) q_n`` with ``q_n = q_p`` under ``u <-> d``.  The deuteron
    map above is this at ``1/2``; nothing else about the nucleus enters.

    **Isospin only, deliberately.**  No shadowing, antishadowing or EMC effect. These
    datasets are read at their NNPDF ``_dw_`` (deweighted) variant, which carries the
    nuclear uncertainty in the *covariance* rather than as a structure-function
    correction -- the same convention CHORUS already uses here.
    """
    z = float(proton_fraction)
    if not 0.0 <= z <= 1.0:
        raise ValueError("proton_fraction must be Z/A in [0, 1]")
    partner = {"u": "d", "d": "u", "ub": "db", "db": "ub"}
    out = {}
    for parton, cmap in DY_PARTON_BASIS_PROTON.items():
        other = partner.get(parton)
        out[parton] = (_merge(cmap, DY_PARTON_BASIS_PROTON[other], z, 1.0 - z)
                       if other else dict(cmap))
    return out


DY_PARTON_BASIS_COPPER = nuclear_parton_basis(DY_COPPER_PROTON_FRACTION)


def dy_field_maps(reaction: str):
    """Return ``(fields_A, fields_B)`` parton->basis maps for a DY reaction.

    ``pp`` uses the proton map on both sides; ``pd`` uses proton on side A and
    the isoscalar deuteron map on side B.
    """
    if reaction == "pp":
        return DY_PARTON_BASIS_PROTON, DY_PARTON_BASIS_PROTON
    if reaction == "pd":
        return DY_PARTON_BASIS_PROTON, DY_PARTON_BASIS_DEUTERON
    if reaction == "ppbar":
        return DY_PARTON_BASIS_PROTON, DY_PARTON_BASIS_ANTIPROTON
    if reaction == "pCu":
        return DY_PARTON_BASIS_PROTON, DY_PARTON_BASIS_COPPER
    raise ValueError(f"unsupported DY reaction {reaction!r}")


TEST_MODE_DY = "dy"                               # DY-only test mode

# -- fit priors + sampling ---------------------------------------------------
# Every field prior is log-RBF with a constant (x-independent) mean, fixed
# correlation length and a fixed per-field sigma.  mean, sigma, length and the
# x_reg regularizer are all frozen, so the GP prior is fully specified and the GP
# field values are marginalized analytically -- no free hyperparameters to sample.

#: One amplitude per field sets **both** the constant prior mean and the log-RBF
#: sigma, so every physical field's prior is ``a +- a``: a 100% relative prior
#: uncertainty about its own mean.  The C-even singlet (sigma), gluon (g) and
#: charm (t15) fields use GP_AMPLITUDE_HIGH; every other field -- the C-even
#: non-singlets (t3, t8) and all C-odd valence fields (v3, v8, v, v15) -- uses
#: GP_AMPLITUDE_LOW.  The mean is a constant function of x (priors.Const).
#:
#: ``fit.py`` enforces the relation with ``analysis.tie(mean.N, to=cov.sigma)``
#: rather than trusting two constants to stay equal.  Both stay **frozen**: a
#: fitted tied amplitude collapses onto its lower bound when the data cannot
#: distinguish the field from zero (see guides/tying_parameters.md).
HIGH_PRIOR_FIELDS = ("sigma", "g", "t15")
GP_AMPLITUDE_HIGH = 5.0
GP_AMPLITUDE_LOW = 1.0


def gp_amplitude(field: str) -> float:
    """Tied prior mean *and* log-RBF sigma (HIGH for singlet/gluon/charm)."""
    return GP_AMPLITUDE_HIGH if field in HIGH_PRIOR_FIELDS else GP_AMPLITUDE_LOW


#: Floating the tied amplitude.  Off: both ends of every tie stay frozen and the
#: hyper-prior contributes nothing.  On: each field's amplitude becomes a free
#: parameter with a hard floor and a log-normal hyper-prior centred on its own
#: tier value -- the mitigation for the ``a -> 0`` collapse that tying the mean to
#: the width introduces (see guides/tying_parameters.md).
GP_AMPLITUDE_FREE = False
#: Hard floor on a *fitted* amplitude: essentially forbid ``a < 1``, so the prior
#: width can never shrink below 1.  Deliberately a hair under 1.0 rather than
#: exactly 1.0, because GP_AMPLITUDE_LOW *is* 1.0 and a bound sitting exactly on
#: the starting value maps to ``eta = -inf`` under the sampler's lower-softplus
#: transform (pixel.infer.samplers.hmc.ParameterCoordinates).
GP_AMPLITUDE_FLOOR = 0.99
#: Log-normal width of that hyper-prior, in e-folds about the tier amplitude.
GP_AMPLITUDE_PRIOR_SIGMA = 0.5


def gp_amplitude_prior(field: str):
    """Hyper-prior for a *floating* tied amplitude; ``None`` while it is frozen.

    Returning ``None`` when frozen is what keeps the evidence unshifted:
    ``neg2_log_prior`` sums over frozen leaves too, so an always-on hyper-prior
    would add a constant (measured 13.72 for these nine fields) to every
    ``-2 log evidence`` while changing no inference at all.
    """
    if not GP_AMPLITUDE_FREE:
        return None
    import math

    from pixel.core.params import LogNormal

    return LogNormal(
        log_center=math.log(gp_amplitude(field)), sigma=GP_AMPLITUDE_PRIOR_SIGMA
    )


def gp_mean(field: str) -> float:
    """Constant log-RBF prior mean; tied to the amplitude, so == gp_sigma."""
    return gp_amplitude(field)


def gp_sigma(field: str) -> float:
    """Log-RBF prior amplitude; tied to the mean, so == gp_mean."""
    return gp_amplitude(field)



# -- alternative prior: zero mean under a x^alpha (1-x)^beta envelope ---------

#: Which prior form ``fit._gp_prior`` builds.  Two choices, and the difference is
#: structural rather than a retune:
#:
#: ``"const_logrbf"``
#:     The original.  ``Const(N=a)`` mean **tied** to ``LogRBF(sigma=a)``, so the
#:     prior is ``a +- a`` and one number sets both the mean and the width.
#:
#: ``"beta_envelope"``
#:     ``Zero()`` mean under :class:`pixel.priors.BetaTaperedLogRBF`,
#:     ``k = sigma^2 (x x')^alpha ((1-x)(1-x'))^beta * logRBF``.  The mean is zero,
#:     so there is nothing to tie and the envelope alone says where the field is
#:     allowed to be large.
#:
#: Why the alternative exists, measured 2026-08-16 on this suite at Q=2: the
#: constant mean asserts ``int x (Sigma + g) dx = 10.0`` against
#: ``cons_momentum``'s ``1.0 +- 1e-4``, an ~9e6-sigma conflict, because ``a = 5``
#: for both fields while the sum rule permits 0.601 and 0.399.  Setting ``a`` to
#: those values does not fix it -- the tie then drops the prior WIDTH to 0.60
#: where the truth singlet reaches 6.3 at ``x = 1e-6``, understating the error
#: exactly where the data is weakest.  A zero mean removes the conflict outright:
#: the prior no longer asserts any integral.
#: **Environment override, for driving THIS suite only.**
#: ``PIXEL_CLOSURE_PRIOR_FORM=beta_envelope`` selects the alternative form without
#: editing this file, which is what makes a prior sweep possible from a campaign
#: driver -- ``fit.gp_prior`` dispatches on a module constant, so before this the
#: only way to cross the prior axis was to monkeypatch ``cfg.PRIOR_FORM`` in
#: process, and a driver that did so was lost once and took the axis with it.
#:
#: It is deliberately NOT the intended user interface. A real analysis selects its
#: prior by constructing the one it wants and handing it to ``analysis.gp_prior``
#: (see ``guides/custom_prior_mean_covariance.md``); this variable exists to sweep
#: the two forms these closure packages ship, nothing more. Anything outside this
#: suite should ignore it.
PRIOR_FORM = os.environ.get("PIXEL_CLOSURE_PRIOR_FORM", "const_logrbf")
if PRIOR_FORM not in ("const_logrbf", "beta_envelope"):
    raise ValueError(
        f"PIXEL_CLOSURE_PRIOR_FORM={PRIOR_FORM!r} is not a known prior form; "
        "expected 'const_logrbf' or 'beta_envelope'"
    )

#: Envelope exponents and amplitude per field, **derived, not guessed**.
#:
#: The criterion is a **minimax**: among all (alpha, beta) whose envelope bounds
#: every truth at every node -- both truth packages, all six Q members -- take the
#: one whose *slack* is smallest, where slack is the factor by which the envelope
#: over-states the error at its loosest point.  That keeps the owner's rule (the
#: prior is never more convergent than the truth, so it never understates) without
#: paying for it in dynamic range.
#:
#: The first derivation did NOT do this and it is instructive.  It took each
#: exponent from the worst pointwise secant anywhere in a window and applied it
#: globally, then verified only ``max |truth| / envelope <= 1`` -- the tight side.
#: The loose side went unmeasured and was where the damage was: the envelope ran
#: 1.9e+05 at ``x = 1e-6`` where the truth singlet is 6.3, over-wide by ~3e+04.
#: That dynamic range, not any exponent, is what made the covariance unfactorable
#: and killed 10/10 Drell-Yan fits.  **Check both sides of a bound.**
#:
#: ``sigma`` is fixed by requiring the truth inside ONE prior standard deviation at
#: every node of every member, times a factor 2 margin, then rounded UP to the next
#: integer (owner's call, 2026-08-16).  Every rounding widens the prior, so the
#: bound can only improve.  Slack and the pre-rounding sigma are quoted per row.
GP_ENVELOPE = {
    "t3": {"alpha": 0.25, "beta": 2.1, "sigma": 3},   # slack 14.2x
    "t8": {"alpha": 0.0, "beta": 1.35, "sigma": 139},   # slack 2481.3x
    "sigma": {"alpha": 0.0, "beta": 1.95, "sigma": 53},   # slack 108.6x
    "t15": {"alpha": 0.0, "beta": 1.95, "sigma": 49},   # slack 118.5x
    "g": {"alpha": 0.0, "beta": 1.95, "sigma": 8},   # slack 95.4x
    "v3": {"alpha": 0.5, "beta": 2.05, "sigma": 3},   # slack 17.3x
    "v8": {"alpha": 0.4, "beta": 2.15, "sigma": 7},   # slack 21.6x
    "v": {"alpha": 0.4, "beta": 2.15, "sigma": 7},   # slack 22.1x
    "v15": {"alpha": 0.4, "beta": 2.15, "sigma": 7},   # slack 22.1x
}


#: Diagonal jitter for the envelope covariance.  **0 by design.**
#:
#: Regularisation belongs to the data covariance and the SVD rcond cut, not to a
#: second additive term that changes the prior.  Measured 2026-08-16: a 1e-10
#: jitter contributes 2.5e-11 to 1.0e-10 of the preconditioned spectrum while the
#: rcond cut sits at 4.40e-15 -- 1.4e3 to 2.3e4 times *above* it -- so it lifts
#: every eigenvalue over the cut and the truncation never fires at all.  Two
#: knobs for one job, with the wrong one winning.
#:
#:
#: **1e-10, not 0 — corrected 2026-08-16 after measuring.**  "Regularisation
#: belongs to the SVD cut, so set the jitter to zero" is wrong here, and the
#: measurement is unambiguous: the log-RBF correlation `R` is *itself* indefinite
#: on this grid (min eig -3.4e-15 at the shipped length, and still -6.7e-16 at
#: length 0.10, so shortening it does not help).  The constant prior is positive
#: definite ONLY because the jitter lifts it -- its `min eig K` is 9.99e-11, i.e.
#: the jitter itself.
#:
#: The jitter and the rcond cut are **not** two knobs for one job; they act on
#: different matrices.  `rcond` cuts `W = C + B K B^T` and conditions the saddle
#: contour.  The jitter enters `K`, and `H = K - K B^T W^-1 B K` is built from `K`
#: and inverted with a bare `jnp.linalg.inv` (`core/evidence.py:675`) -- no cutoff
#: on that path at all.  Measured on `both` with the envelope prior: 9/9 fail
#: across rcond 1e-10 to 1e-6 at jitter <= 1e-10, including at rcond 1e-6 where the
#: cut discards more than half the modes (rank 74/158, 94/230).  Truncating `W`
#: cannot make `H` definite.
#:
#: Required jitter scales with system size: `dy` (n=14) runs at 1e-10, while
#: `exp`/`both` (n=158/230) need 1e-4 for the envelope prior.
#: **1e-8, not 1e-10 — raised 2026-09-22.**  At 1e-10 roughly 900 of `H`'s 1314
#: eigenvalues sit *at the jitter* rather than decaying, so `Hinv = inv(H)` carries
#: ~900 directions of size 1e10 holding no information and `cond(H) = 6.96e12` is
#: inherited whole by the H-S matrix `A = Hinv + T(f)`.  The consequence is a wrong
#: number, not slow convergence: `force^T A^-1 force` has a forward error of ~1.8e6
#: and must cancel against `constant` (both 2.9e9) to leave ~66.  Measured on the
#: frozen JAM `dy`/`c_config` block at Q=2, `E^2` carries 0.17 peak-to-peak of noise
#: against a true variation of 0.0025, and its value is wrong by **85**.  One decade
#: of jitter takes the jaggedness to 5.27e-5 (**3200x**) with the frame-pinned
#: quadratic coefficient still right to 1.6%.
#:
#: **Carried here to hold `test_all_four_suites_share_the_numerical_settings`, and
#: measured only on the small suites.**  The mechanism is a property of the shared
#: GP prior and evidence algebra, but this suite's own grid and data volume differ;
#: confirm before relying on it at full scale.
#:
#: Full derivation and the ladder: `EXT:pixel/plans/hs_saddle_contour_algorithms.md`.
GP_JITTER = 1.0e-8


#: Jitter for the ENVELOPE prior, which needs far more than the constant one.
#:
#: Measured on `both` at Q=2, constraints 1e-4, rcond 1e-12: the envelope fails
#: (saddle below 1e-6, ESS from 1e-6 to 1e-4) and runs from 1e-3 upward, with a
#: flat plateau -- 1.355e-05 / 1.342e-05 / 1.263e-05 at 1e-3 / 1e-2 / 1e-1.  The
#: constant prior over the same range runs from 1e-10 at a steady 1.780e-05.
#:
#: The gap is arithmetic, not a second pathology.  The jitter enters as
#: ``R + lambda I`` where ``R`` already carries ``sigma^2``, so the *relative*
#: floor is ``lambda / sigma^2``.  Enforcing ``alpha >= 0`` forced flat envelopes
#: that must clear the small-x peak, raising ``sigma`` from 3 to 139 for ``t8``
#: and 3 to 53 for ``sigma`` -- roughly 300x in ``sigma^2`` -- so the same
#: ``lambda`` is a ~300x smaller relative floor.
#:
#: 1e-2 is the plateau centre.  At these amplitudes it is a relative floor of
#: ~1e-6 (t8) to ~1e-3 (t3, v3).
GP_ENVELOPE_JITTER = 1.0e-2

def gp_envelope(field: str) -> dict:
    """Envelope ``{alpha, beta, sigma}`` for one field; see :data:`GP_ENVELOPE`."""
    return GP_ENVELOPE[field]

GP_LENGTH_LOG = 0.6931471805599453           # ln(2)
# ln(10) was tried on 2026-08-16 and REVERTED, for robustness rather than
# accuracy.  At its working point ln(10) is more accurate -- the constant prior
# reached 9.90e-06 against ln(2)'s 1.78e-05, and the envelope 1.33e-05 against
# 2.92e-05 -- but it demands ~1e3x more jitter (const boundary 1e-10 -> 1e-4,
# envelope 1e-5 -> 1e-2) because the log-RBF correlation matrix becomes more
# indefinite (min eig -3.4e-15 -> -1.25e-14), and it survived in only the top
# cell or two of the scanned range instead of ln(2)'s 4-6 order plateau.  It
# also halves n_eff over the coverage bulk, 2.65 -> 1.38, widening the pull^2
# sampling band from +-0.87 to +-1.21.
FROZEN_COV_PARAMS = ("sigma", "length", "x_reg")  # all frozen
# Matches pixel.util.linalg.DEFAULT_RCOND.  svd_factor preconditions W to unit
# diagonal before applying this cutoff, so it is a cut on the *correlation*
# matrix and is invariant to the units each dataset is stored in.
RCOND = 1.0e-16

MAP_METHOD = "L-BFGS-B"
MAP_OPTIONS = {"maxiter": 200}
MCMC_SAMPLES = 1000
MCMC_SEED = 0
# H-S parameter inference uses ordinary NUTS for linear modes and contour
# NUTS for normalization-only bilinear modes; mixed ordinary/H-S modes use
# joint HMC. All initialize from a dense inverse Hessian.
NUTS_WARMUP = 100
NUTS_INITIAL_STEP_SIZE = 0.1
NUTS_INTEGRATOR = "leapfrog"
NUTS_MAX_TREE_DEPTH = 6
# The Hessian metric is already tailored to the local target. Avoid rebuilding
# NUTS steppers and repeating costly step-size searches at mass-window updates.
NUTS_ADAPT_MASS = False
# Shared by the H-S NUTS and joint-HMC paths.
HS_HESSIAN_MASS_MODE = "dense"

# Exact bilinear-ratio modes cannot use marginalized/H-S inference.  These
# settings govern the explicit (theta,q) NUTS path. Every positive prior
# mode is sampled; prior_rcond remains zero for compatibility with older configs.
DIRECT_PRIOR_RCOND = 0.0
DIRECT_NUTS_WARMUP = 250
DIRECT_NUTS_INITIAL_STEP_SIZE = 0.03
DIRECT_NUTS_MAX_TREE_DEPTH = 8
DIRECT_NUTS_ADAPT_MASS = True

#: Generic mixed linear-normalization + H-S models cannot use the optimized
#: fixed-reference NUTS contour.  They run the validated joint-contour HMC path.
HMC_WARMUP = 100
HMC_STEP_SIZE = 0.03
HMC_N_LEAPFROG = 12
HMC_INTEGRATOR = "omelyan"
#: Deform every sampled Hubbard-Stratonovich sector with PIXEL's generalized
#: thimble flow.  The affine-whitened flow scan in
#: ``pixel/plans/hs_saddle_contour_algorithms.md`` found ``T=0.2`` to be the
#: deepest consistently stable/improving point, with the 8-step RK4 result
#: converged against the 2/32-step checks.  Normalization-only contours can
#: start from the affine saddle reference; generic joint contours use PIXEL's
#: theta-dependent flowed map with the same time and resolution.
CONTOUR_FLOW_TIME = 0.2
CONTOUR_N_FLOW_STEPS = 8
CONTOUR_FLOW_FROM_AFFINE_REFERENCE = True
# Some closure likelihoods stall just below a 3e-3 Newton decrement at
# floating-point precision.  Keep a small margin so a numerically stationary
# H-S saddle is accepted instead of aborting contour-NUTS initialization.
CONTOUR_SADDLE_TOL = 5.0e-3

N_HYPERPARAMS = len(ALL_FIELDS) if GP_AMPLITUDE_FREE else 0  # 0 = fully-fixed prior

#: Test modes.  ``dy`` is Drell-Yan only; ``both`` = lattice + DIS + Drell-Yan.
TEST_MODES = ("lattice", "dis", "dy", "exp", "both")

# -- physics constraints (near-hard pseudo-data added to every fit) ----------
# 1. every distribution vanishes at x = 1;
# 2. valence normalizations from quark counting: int(u-ubar)=2, int(d-dbar)=1,
#    int(q-qbar)=0 for s,c.  In the basis fields this is
#    int V3 = 1, int V8 = 3, int V = 3, int V15 = 3.

ENDPOINT_X = 1.0
#: **1e-4, raised from 1e-6 on 2026-08-16.**  T1 was finally run on `exp` -- the mode
#: that exercises DIS + DY + the gluon coefficient function together -- instead of on
#: `dy`, which has no gluon channel and is structurally blind here.  One fit per row,
#: closure_NNPDF_truth_small at Q=2:
#:
#:     sigma    abs ESS   signed   inner    result
#:     1e-6     0.001     0.001    0.001    FAIL (172 s)
#:     1e-5     0.980     0.980    0.971    PASS (96 s)
#:     1e-4     0.989     0.989    0.984    PASS (135 s)
#:
#: A cliff, not a gradient, and 0.001 is exactly 1/MCMC_SAMPLES -- one effective
#: sample, a saturated floor.  Ruled out as causes in the same session: the momentum
#: row (dropped at 1e-6, still 0.001) and the GP amplitudes (sigma/g set to their
#: sum-rule values, and all nine fields set to theirs, both still 0.001 -- unchanged
#: to three digits under a 13x amplitude change).
#:
#: The earlier "1e-6 is the tightest value that works everywhere" was measured on
#: `dy` alone and is superseded.  The 1e-5/1e-4 pull tables agree to three decimals,
#: so nothing is bought by the extra decade; 1e-4 is taken for margin.
#: Do not tighten without re-running `exp`.
CONSTRAINT_ENDPOINT_SIGMA = 1.0e-4                # ~hard: f(x=1)=0
CONSTRAINT_NORM_SIGMA = 1.0e-4                    # ~hard: valence integrals
CONSTRAINT_ORIGIN_SIGMA = 1.0e-4                  # ~hard: x*f(x=0)=0
#: ~hard: <x^1>, the momentum sum rule int x (Sigma + g) dx = 1.  Added 2026-08-15
#: alongside the same constraint in the _small suites; `sigma` and `g` previously
#: carried no integral constraint at all.  The target is the *represented* truth,
#: not the nominal 1.0 -- see `constraint_datasets.represented_sum_target`.
CONSTRAINT_MOMENTUM_SIGMA = 1.0e-4
VALENCE_NORMS = {"v3": 1.0, "v8": 3.0, "v": 3.0, "v15": 3.0}

# -- per-member paths --------------------------------------------------------


def truth_label(q_key: str) -> str:
    """Directory-safe label for an original-Q truth member, e.g. ``truthQ_2``."""
    return f"truthQ_{q_key}"


def truth_dir(q_key: str) -> Path:
    return DATA_ROOT / truth_label(q_key)


def results_dir(q_key: str) -> Path:
    return RESULTS_ROOT / truth_label(q_key)


def make_grid():
    """The shared 146-point log-linear PDF grid (piecewise-quintic elements)."""
    from pixel.geometry import Grid

    return Grid(n_points=GRID_N, spacing=GRID_SPACING, x_min=X_MIN)


def make_fields():
    """Create all nine closure fields on the shared grid (name -> Field)."""
    from pixel.core.model import Field

    grid = make_grid()
    return {
        name: Field.create(name, grid, element_type=ELEMENT_TYPE)
        for name in ALL_FIELDS
    }


def make_moment_nuisance_grid(order: int):
    """One-active-point grid for a scalar nuisance attached to Mellin order ``N``."""
    from pixel.geometry import ProductGrid

    order = float(order)
    # Finite elements require a two-point underlying axis.  Mask the guard node;
    # the width-2 interval makes the active endpoint element's Unit integral 1.
    axis = [order, order + 2.0]
    return ProductGrid((axis,), support=lambda nodes: nodes[:, 0] == order)


# -- lattice systematics: shared nuisance fields -----------------------------


def nuisance_field_name(field: str, key: str) -> str:
    """Name of a pseudo-ITD nuisance curve for ``(physical field, key)``."""
    return f"nuis_{field}_{key}"


def moment_nuisance_field_name(field: str, key: str, order: int) -> str:
    """Name of an independent scalar nuisance for one Mellin power."""
    return f"nuis_moment_{field}_{key}_n{int(order)}"


def nuisance_specs() -> list:
    """Pseudo-ITD ``(physical_field, key)`` nuisance-curve specifications.

    Mellin moments deliberately do not appear here: their systematics are
    independent scalar fields returned by :func:`moment_nuisance_specs`.
    """
    return [(f, k) for f in ALL_FIELDS for k in ITD_SYSTEMATICS]


def moment_nuisance_specs() -> list:
    """``(physical_field, key, Mellin order)`` scalar nuisance specifications."""
    return [
        (field, key, mellin_order_for(field))
        for field in ALL_FIELDS
        for key in moment_systematics_for(field)
    ]


def itd_systematics_for(field: str) -> tuple:
    """Active pseudo-ITD systematic keys for a physical field."""
    return ITD_SYSTEMATICS


def moment_systematics_for(field: str) -> tuple:
    """Active Mellin-moment systematic keys for a physical field."""
    return MOMENT_SYSTEMATICS


# >>> canonical NNPDF specs -- identical in all six full suites; edit all six together
# -- NNPDF-commondata dataset specs ------------------------------------------------
#
# ``generate.nnpdf_layout`` walks the CONTRACT selection (``CONTRACT_MODE``) and looks each
# selected id up here, so a spec a suite's mode does not select costs nothing, and one
# table in every suite is what keeps the six from drifting.  Promoting a contract without
# a spec stops generation ("refusing silent omission"); a spec never selects anything.
# ``tests/test_closure_nnpdf_datasets.py`` pins the six copies identical.
#
# The nuclear composition is restated because the operator needs numbers and the contract
# states prose.  A and Z are *natural* lead: NNPDF's CHORUS target is the emulsion's
# natural isotopic mix, not Pb-208, and ``NuclearTarget`` refuses to default either.
NNPDF_LEAD_TARGET = {"name": "lead", "mass_number": 207.2, "atomic_number": 82.0}

_CHORUS_SPECS = {
    "dis.chorus_nu": {
        "label": "chorus_nu", "kind": "sigma_r_cc", "beam": "nu",
        "commondata_name": "CHORUS_CC_NOTFIXED_PB_NU-SIGMARED",
        "target": NNPDF_LEAD_TARGET,
    },
    "dis.chorus_nb": {
        "label": "chorus_nb", "kind": "sigma_r_cc", "beam": "nubar",
        "commondata_name": "CHORUS_CC_NOTFIXED_PB_NB-SIGMARED",
        "target": NNPDF_LEAD_TARGET,
    },
}

#: Heavy-flavour-tagged NC reduced cross section (HERACOMB_SIGMARED_B), FONLL-A at this
#: closure's NLO; see ``datasets.build_sigma_r_heavy``.
HEAVY_DIS_SPECS = {
    "dis.heracomb_bottom": {
        "label": "heracomb_bottom_sigmar", "kind": "sigma_r_heavy",
        "commondata_name": "HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED",
        "beam": "ep", "target": "proton",
        "heavy_flavour": "bottom", "heavy_mass": MB, "charge_sq": 1.0 / 9.0,
    },
}

#: Fixed-target nuclear Drell-Yan (DYE605_dw_ite, p+Cu): the pointwise E866 observable
#: with a copper parton map (``nuclear_parton_basis``).
FIXED_TARGET_DY_SPECS = {
    "dy.e605_copper": {
        "label": "dy_e605_copper", "kind": "drell_yan", "reaction": "pCu",
        "commondata_name": "DYE605_Z0_38P8GEV_DW_PXSEC",
        "beam": "p", "target": "copper",
    },
}

#: Collider Drell-Yan, read by ``generate._collider_dy_record``.  Fields beyond
#: label/kind/reaction/commondata_name/nnpdf_legacy_name:
#:
#:   rows              (first, stop) slice of the commondata table.  A mixed W/Z table is
#:                     one record per boson (and per Z mass slice), each fixed by the set's
#:                     own filter.py and pinned in its contract -- never by metadata labels
#:                     (LHCB_DY_8TEV_MUON's are wrong).
#:   lepton_charge     +1 / -1 for a W record; required by the fiducial_lepton builder.
#:   fiducial_channel  "w" / "z" for the two ATLAS concatenations, whose cut records are
#:                     read per channel (fiducial_cuts_for(..., channel=)).
#:   y_edges           "commondata" (default) | ("hepdata", ((directory, table), ...)) |
#:                     "fiducial" (the whole cut window, for a mass spectrum).
#:   mass_edges        "commondata" | ("csv", directory, file, low_column, high_column) |
#:                     ("windows", ((first, stop, low_gev, high_gev), ...)) | "pole".
#:                     Every Z/gamma* record names its published mass window (owner
#:                     decision D7, 2026-09-11): a measurement integrated over 60-120 or
#:                     66-116 GeV is not the cross section at m_Z^2 with zero width, and
#:                     generate._collider_dy_record refuses a Z record left at the default
#:                     "pole".  Only the gated W records still take it.  Each window is
#:                     commented with the HEPData qualifier it is read from.
#:   sqrts_gev         only where the commondata omits sqrts.
#:
#: Not here, each blocked in its contract: ATLASLOMASSDY11EXT (physics: Born acceptance
#: zero below 18 GeV), CMSDY2D11 (data: mass-bin edges), ATLAS_WZ_TOT_13TEV
#: (closure_builder: drell_yan_total).
#:
#: The ten W specs below ARE here and their contracts are reader_only: PIXEL's
#: DYLeptonEtaKernel composes no evolution, so the drell_yan_fiducial_lepton datatype is
#: gated (measured 2026-09-10) and dy_builder answers with the gate.  Rows, edges, charges
#: and cuts are kept built so that promoting them is a contract edit.
_Z = "drell_yan_binned"
_W = "drell_yan_fiducial_lepton"
COLLIDER_DY_SPECS = {
    # -- Z-peak rapidity scans and the 8 TeV low-mass 2D set, 2026-09-09 ----------------
    "z.lhcb7_dielectron": {
        "label": "dy_lhcb7_z_ee", "kind": _Z, "reaction": "pp",
        "commondata_name": "LHCB_Z0_7TEV_DIELECTRON_Y", "nnpdf_legacy_name": "LHCBZ940PB",
        # HEPData Table2 (LHCB_Z0_7TEV_DIELECTRON/rawdata): LEPTONS MASS 60.0-120.0.
        "mass_edges": ("windows", ((0, 9, 60.0, 120.0),)),
    },
    "z.lhcb8_dielectron": {
        "label": "dy_lhcb8_z_ee", "kind": _Z, "reaction": "pp",
        "commondata_name": "LHCB_Z0_8TEV_DIELECTRON_Y", "nnpdf_legacy_name": "LHCBZEE2FB_40",
        # HEPData Table2 (LHCB_Z0_8TEV_DIELECTRON/rawdata): M(E+ E-) 60.0-120.0.
        "mass_edges": ("windows", ((0, 17, 60.0, 120.0),)),
    },
    "z.lhcb13_dimuon": {
        "label": "dy_lhcb13_z_mumu", "kind": _Z, "reaction": "pp",
        "commondata_name": "LHCB_Z0_13TEV_DIMUON-Y",
        "nnpdf_legacy_name": "LHCB_Z_13TEV_DIMUON",
        # HEPData Table_5a (LHCB_Z0_13TEV/rawdata): M(LEPTON+ LEPTON-) 60.0-120.0.
        "mass_edges": ("windows", ((0, 18, 60.0, 120.0),)),
    },
    "z.lhcb13_dielectron": {
        "label": "dy_lhcb13_z_ee", "kind": _Z, "reaction": "pp",
        "commondata_name": "LHCB_Z0_13TEV_DIELECTRON-Y",
        "nnpdf_legacy_name": "LHCB_Z_13TEV_DIELECTRON",
        # HEPData Table_5b (LHCB_Z0_13TEV/rawdata): M(LEPTON+ LEPTON-) 60.0-120.0.
        "mass_edges": ("windows", ((0, 17, 60.0, 120.0),)),
    },
    "z.atlas8_lomass_2d": {
        "label": "dy_atlas8_z_lowmass_2d", "kind": _Z, "reaction": "pp",
        "commondata_name": "ATLAS_Z0_8TEV_LOWMASS_M-Y",
        "nnpdf_legacy_name": "ATLAS_DY_2D_8TEV_LOWMASS", "mass_edges": "commondata",
    },
    # -- 2026-09-10 --------------------------------------------------------------------
    "z.atlas8_himass_2d": {
        "label": "dy_atlas8_z_himass_2d", "kind": _Z, "reaction": "pp",
        "commondata_name": "ATLAS_Z0_8TEV_HIMASS_M-Y", "nnpdf_legacy_name": "ATLASDY2D8TEV",
        "mass_edges": "commondata",
    },
    "z.atlas7_himass": {
        "label": "dy_atlas7_z_himass", "kind": "drell_yan_mass_spectrum", "reaction": "pp",
        "commondata_name": "ATLAS_Z0_7TEV_49FB_HIMASS",
        "nnpdf_legacy_name": "ATLASZHIGHMASS49FB", "sqrts_gev": 7000.0,
        "y_edges": "fiducial",
        "mass_edges": ("csv", "ATLAS_Z0_7TEV_49FB", "ATLAS-49fb-Zhighmass.csv",
                       "mee_xmin", "mee_xmax"),
    },
    "z.atlas7_46fb_cf": {
        "label": "dy_atlas7_46fb_z_cf", "kind": _Z, "reaction": "pp",
        "commondata_name": "ATLAS_Z0_7TEV_46FB_CF-Y", "nnpdf_legacy_name": "ATLASWZRAP11CF",
        "y_edges": ("hepdata", (("ATLAS_Z0_7TEV_46FB", 14), ("ATLAS_Z0_7TEV_46FB", 15))),
        "mass_edges": ("windows", ((0, 9, 66.0, 116.0), (9, 15, 116.0, 150.0))),
    },
    "z.cdf_zrap": {
        "label": "dy_cdf_z_rap", "kind": "drell_yan_bin_integrated", "reaction": "ppbar",
        "commondata_name": "CDF_Z0_1P96TEV_ZRAP", "nnpdf_legacy_name": "CDFZRAP_NEW",
        # HEPData ins856131 Table_2 (CDF_Z0_1P96TEV/rawdata): M(P=3_4) 66.0-116.0 GeV.
        "mass_edges": ("windows", ((0, 28, 66.0, 116.0),)),
    },
    "w.cms8_wplus": {
        "label": "dy_cms8_wp_mu", "kind": _W, "reaction": "pp",
        "commondata_name": "CMS_WPWM_8TEV_MUON_Y", "nnpdf_legacy_name": "CMSWMU8TEV",
        "rows": (0, 11), "lepton_charge": 1,
    },
    "w.cms8_wminus": {
        "label": "dy_cms8_wm_mu", "kind": _W, "reaction": "pp",
        "commondata_name": "CMS_WPWM_8TEV_MUON_Y", "nnpdf_legacy_name": "CMSWMU8TEV",
        "rows": (11, 22), "lepton_charge": -1,
    },
    "z.lhcb7_muon": {
        "label": "dy_lhcb7_z_mumu", "kind": _Z, "reaction": "pp",
        "commondata_name": "LHCB_DY_7TEV_MUON_Y", "nnpdf_legacy_name": "LHCBWZMU7TEV",
        "rows": (0, 17),
        # HEPData ins1373300 Table_1 (LHCB_DY_7TEV_MUON/rawdata): M(MU+ MU-) 60.0-120.0.
        "mass_edges": ("windows", ((0, 17, 60.0, 120.0),)),
    },
    "w.lhcb7_wplus": {
        "label": "dy_lhcb7_wp_mu", "kind": _W, "reaction": "pp",
        "commondata_name": "LHCB_DY_7TEV_MUON_Y", "nnpdf_legacy_name": "LHCBWZMU7TEV",
        "rows": (17, 25), "lepton_charge": 1,
    },
    "w.lhcb7_wminus": {
        "label": "dy_lhcb7_wm_mu", "kind": _W, "reaction": "pp",
        "commondata_name": "LHCB_DY_7TEV_MUON_Y", "nnpdf_legacy_name": "LHCBWZMU7TEV",
        "rows": (25, 33), "lepton_charge": -1,
    },
    "z.lhcb8_muon": {
        "label": "dy_lhcb8_z_mumu", "kind": _Z, "reaction": "pp",
        "commondata_name": "LHCB_DY_8TEV_MUON_Y", "nnpdf_legacy_name": "LHCBWZMU8TEV",
        "rows": (0, 18),
        # HEPData ins1406555 Table_2 (LHCB_DY_8TEV_MUON/rawdata): M(MU+ MU-) 60.0-120.0.
        "mass_edges": ("windows", ((0, 18, 60.0, 120.0),)),
    },
    "w.lhcb8_wplus": {
        "label": "dy_lhcb8_wp_mu", "kind": _W, "reaction": "pp",
        "commondata_name": "LHCB_DY_8TEV_MUON_Y", "nnpdf_legacy_name": "LHCBWZMU8TEV",
        "rows": (18, 26), "lepton_charge": 1,
    },
    "w.lhcb8_wminus": {
        "label": "dy_lhcb8_wm_mu", "kind": _W, "reaction": "pp",
        "commondata_name": "LHCB_DY_8TEV_MUON_Y", "nnpdf_legacy_name": "LHCBWZMU8TEV",
        "rows": (26, 34), "lepton_charge": -1,
    },
    "w.atlas7_46fb_wplus": {
        "label": "dy_atlas7_46fb_wp", "kind": _W, "reaction": "pp",
        "commondata_name": "ATLAS_DY_7TEV_46FB_CC", "nnpdf_legacy_name": "ATLASWZRAP11CC",
        "rows": (0, 11), "lepton_charge": 1, "fiducial_channel": "w",
        "y_edges": ("hepdata", (("ATLAS_WPWM_7TEV_46FB", 9),)),
    },
    "w.atlas7_46fb_wminus": {
        "label": "dy_atlas7_46fb_wm", "kind": _W, "reaction": "pp",
        "commondata_name": "ATLAS_DY_7TEV_46FB_CC", "nnpdf_legacy_name": "ATLASWZRAP11CC",
        "rows": (11, 22), "lepton_charge": -1, "fiducial_channel": "w",
        "y_edges": ("hepdata", (("ATLAS_WPWM_7TEV_46FB", 10),)),
    },
    "z.atlas7_46fb_cc_low": {
        "label": "dy_atlas7_46fb_z_cc_low", "kind": _Z, "reaction": "pp",
        "commondata_name": "ATLAS_DY_7TEV_46FB_CC", "nnpdf_legacy_name": "ATLASWZRAP11CC",
        "rows": (22, 28), "fiducial_channel": "z",
        "y_edges": ("hepdata", (("ATLAS_Z0_7TEV_46FB", 11),)),
        "mass_edges": ("windows", ((0, 6, 46.0, 66.0),)),
    },
    "z.atlas7_46fb_cc_peak": {
        "label": "dy_atlas7_46fb_z_cc_peak", "kind": _Z, "reaction": "pp",
        "commondata_name": "ATLAS_DY_7TEV_46FB_CC", "nnpdf_legacy_name": "ATLASWZRAP11CC",
        "rows": (28, 40), "fiducial_channel": "z",
        "y_edges": ("hepdata", (("ATLAS_Z0_7TEV_46FB", 12),)),
        # HEPData ins1502620 Table_12 (ATLAS_Z0_7TEV_46FB/rawdata): MLL 66-116.
        "mass_edges": ("windows", ((0, 12, 66.0, 116.0),)),
    },
    "z.atlas7_46fb_cc_high": {
        "label": "dy_atlas7_46fb_z_cc_high", "kind": _Z, "reaction": "pp",
        "commondata_name": "ATLAS_DY_7TEV_46FB_CC", "nnpdf_legacy_name": "ATLASWZRAP11CC",
        "rows": (40, 46), "fiducial_channel": "z",
        "y_edges": ("hepdata", (("ATLAS_Z0_7TEV_46FB", 13),)),
        "mass_edges": ("windows", ((0, 6, 116.0, 150.0),)),
    },
    "w.atlas7_36pb_wplus": {
        "label": "dy_atlas7_36pb_wp", "kind": _W, "reaction": "pp",
        "commondata_name": "ATLAS_DY_7TEV_36PB_ETA", "nnpdf_legacy_name": "ATLASWZRAP36PB",
        "rows": (0, 11), "lepton_charge": 1, "fiducial_channel": "w",
        "y_edges": ("hepdata", (("ATLAS_WPWM_7TEV_36PB", 5),)),
    },
    "w.atlas7_36pb_wminus": {
        "label": "dy_atlas7_36pb_wm", "kind": _W, "reaction": "pp",
        "commondata_name": "ATLAS_DY_7TEV_36PB_ETA", "nnpdf_legacy_name": "ATLASWZRAP36PB",
        "rows": (11, 22), "lepton_charge": -1, "fiducial_channel": "w",
        "y_edges": ("hepdata", (("ATLAS_WPWM_7TEV_36PB", 3),)),
    },
    "z.atlas7_36pb": {
        "label": "dy_atlas7_36pb_z", "kind": _Z, "reaction": "pp",
        "commondata_name": "ATLAS_DY_7TEV_36PB_ETA", "nnpdf_legacy_name": "ATLASWZRAP36PB",
        "rows": (22, 30), "fiducial_channel": "z",
        "y_edges": ("hepdata", (("ATLAS_Z0_7TEV_36PB", 1),)),
        # HEPData ins928289 Table_1 (ATLAS_Z0_7TEV_36PB/rawdata): M(C=LEPTON+LEPTON-) 66.0-116.0.
        "mass_edges": ("windows", ((0, 8, 66.0, 116.0),)),
    },
}

#: One registry for everything ``nnpdf_layout`` materializes from NNPDF commondata.
NNPDF_DATASET_SPECS = {
    **_CHORUS_SPECS, **HEAVY_DIS_SPECS, **FIXED_TARGET_DY_SPECS, **COLLIDER_DY_SPECS,
}
# <<< canonical NNPDF specs
