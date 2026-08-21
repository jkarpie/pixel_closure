"""Shared configuration for the NNPDF-anchored full-flavor closure suite.

Every constant that generation *and* fitting must agree on lives here, so the
two can never silently drift.  See ``closure_plan.md`` (repo root) for the
physics narrative and ``closure_NNPDF_truth_small/README.md`` for usage.

Truth
-----
The truth is the real ``NNPDF40_nnlo_as_01180_1000`` global-analysis result
(NNLO LHAPDF grid, used with PIXEL's NLO closure kernels).  We read NNPDF at
several *original* scales ``Q`` and treat each
resulting curve as the truth at the common input scale ``Q0 = mc`` -- the scale
at which the closure distributions are parameterized and the fake data are
generated.  Reading NNPDF at different ``Q`` gives different but realistic
PDF-shaped truths; comparing the recovered fits across ``Q`` is the proof that
the code works across inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

# -- locations ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_ROOT = ROOT / "data"
RESULTS_ROOT = ROOT / "results"
REFERENCE_DIR = ROOT / "reference_pdfs"          # cached NNPDF dumps / ref fits
DIS_MANIFEST_PATH = DATA_ROOT / "dis_manifest.json"
#: Per-table DIS correlated-systematic vectors, stored *relative* to the real
#: value.  Too bulky for the JSON manifest (HERA ships 169 sources per table),
#: so dis_audit writes one .npz per table here and the manifest references it.
DIS_SYS_DIR = DATA_ROOT / "dis_systematics"

#: Shared kernel-matrix cache.  The lattice/DIS forward operators depend only on
#: the (Q-independent) dataset kinematics + Q0/nf/order, not on which NNPDF truth
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
NNPDF_COMMONDATA = REPO_ROOT.parent / "nnpdf" / "nnpdf_data" / "nnpdf_data" / "commondata"

# Small keeps every published NNPDF4.0 DIS/DY dataset identity but limits each
# table to a deterministic kinematic-extrema-preserving smoke subset.
from pixel.data.nnpdf40_native import (
    assert_native_campaign_ready as _assert_native_campaign_ready,
    nnpdf40_native_contracts_for_mode as _nnpdf40_native_contracts_for_mode,
)
NNPDF40_ROSTER_PATH = REPO_ROOT / "nnpdf40_dis_dy_roster.yaml"


def nnpdf40_datasets_for_mode(mode: str):
    """Published NNPDF4.0 DIS/DY native contracts targeted by one small mode."""
    return _nnpdf40_native_contracts_for_mode(mode, scale="small")


def assert_nnpdf40_native_ready(mode: str, *, validated_datasets=()):
    """Block canonical generation until every selected native operator passes NNPDF parity."""
    _assert_native_campaign_ready(
        nnpdf40_datasets_for_mode(mode), validated_datasets=validated_datasets
    )

# -- LHAPDF / NNPDF truth ------------------------------------------------------

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
NNPDF_SET = "NNPDF40_nnlo_as_01180_1000"
#: The LHAPDF grid tarball is vendored in the repo so a fresh checkout can
#: rebuild the truth without a separate LHAPDF set download, and so rebuilding
#: LHAPDF (which does not carry sets across) cannot silently disable the closure
#: suite again.  Extracted on demand into :data:`NNPDF_LHAPDF_DATA_ROOT`.
NNPDF_TARBALL = ROOT / "lhapdf" / f"{NNPDF_SET}.tar.gz"
NNPDF_LHAPDF_DATA_ROOT = DATA_ROOT / "lhapdf"
NNPDF_MEMBER = 0            # 0 = central; used by pdf_guidance's standalone cache

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

    The NNPDF grid is extracted lazily from :data:`NNPDF_TARBALL` by
    ``pdf_guidance.ensure_lhapdf_data``, so at import time it may not be on
    disk yet.  Look in the local extraction root and the LHAPDF install, then
    fall back to reading the ``.info`` member straight out of the tarball so
    the member count is correct even before the first extraction.

    Args:
        set_name: LHAPDF set name; defaults to :data:`NNPDF_SET`.

    Returns:
        dict: Scalar ``key -> value`` pairs, empty if the set cannot be found.
    """
    name = NNPDF_SET if set_name is None else set_name
    leaf = f"{name}.info"
    for root in (NNPDF_LHAPDF_DATA_ROOT, LHAPDF_DATA_DIR):
        info_path = root / name / leaf
        if info_path.is_file():
            return _parse_info(info_path.read_text())

    if NNPDF_TARBALL.exists():
        import tarfile

        try:
            with tarfile.open(NNPDF_TARBALL, "r:gz") as tar:
                handle = tar.extractfile(f"{name}/{leaf}")
                if handle is not None:
                    return _parse_info(handle.read().decode())
        except (tarfile.TarError, KeyError, OSError):
            return {}
    return {}


#: Raw ``.info`` metadata for :data:`NNPDF_SET`, empty when the set is missing.
NNPDF_SET_INFO = _read_set_info()

#: Member count and QMin are properties *of the set*, so they are read from its
#: ``.info`` file rather than hardcoded -- hardcoding silently desynchronizes
#: whenever the truth set is swapped or updated.  Both are 0/None when the set
#: cannot be located; callers skip in that case.
NNPDF_N_MEMBERS = int(NNPDF_SET_INFO.get("NumMembers", 0))

#: Advertised set QMin, recorded for metadata.  Requested truth scales are
#: evaluated directly; LHAPDF extrapolates outside the advertised grid.
NNPDF_QMIN = float(NNPDF_SET_INFO["QMin"]) if "QMin" in NNPDF_SET_INFO else None

#: Replica members used to build the closure fake data.  Generation folds every
#: replica through the forward operators and takes the ensemble mean (central
#: value) and covariance (error) of the transformed output.  Member 0 is the
#: central value and is excluded; 1..NumMembers-1 are the replicas.
NNPDF_REPLICA_MEMBERS = tuple(range(1, NNPDF_N_MEMBERS))

# -- perturbative scheme -----------------------------------------------------

MC = 1.28                # charm mass = input scale
MB = 4.18
Q0 = MC                  # input / parameterization / generation scale
Q0_2 = Q0 * Q0
NF = 4                   # active flavors at the input scale (u,d,s,c)
ORDER = "NLO"
MODE = "truncated"
ALPHAS_MZ = 0.118

#: Truth members.  Restored to the full six on 2026-08-16 at the owner's request, to
#: compare the constant prior against the beta-envelope prior across every scale.
#: The two caveats that motivated the 2026-08-15 reduction to {2, 3} still hold and
#: are recorded rather than repaired:
#:
#: * ``mc`` (1.28) and ``1`` sit **below the sets' advertised QMin** -- NNPDF 1.65,
#:   JAM 1.14 -- so LHAPDF extrapolates and the curve is not a physical PDF there.
#:   Measured: ``int x (Sigma + g) dx`` is **0.821 for NNPDF at mc**, an 18%
#:   violation, against 1.0005 at Q=2.  ``cons_momentum`` targets the *represented*
#:   truth rather than a nominal 1.0, so the closure stays self-consistent -- but any
#:   statement about absolute normalization at those two members inherits the 18%.
#: * ``4`` and ``5`` read an already-evolved gluon and call it an *input*.  At
#:   x = 1e-4 the truth gluon runs 0.11 (Q=1) to 17.3 (Q=5), a factor of ~150, so
#:   their low-x behaviour reflects evolution, not a 1-2 GeV starting distribution.
#:
#: Prefer 2 and 3 when the question is about a realistic input scale.
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

# Log-linear grid, 128 points (per spec).  Cubic-spline elements: PIXEL's x-space
# inverse-Mellin evolution needs a far longer (slower) contour for C0 "piecewise"
# bases, so splines are both faster and more accurate for the DIS kernels.
GRID_N = 128
GRID_SPACING = "log-linear"
X_MIN = 1.0e-6
ELEMENT_TYPE = "cubic_spline"

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
    """Return the closure low-x completion for one ``x*f`` field.

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
    return {basis: low_x_completion(field) for basis, field in EVEN_MAP.items()}


def odd_low_x_completions() -> dict:
    return {basis: low_x_completion(field) for basis, field in ODD_MAP.items()}


def itd_momentum_density(field: str) -> bool:
    """True (alpha=-1) for non-singlet quark ITD; False (alpha=0) for sigma/g."""
    return field not in SINGLET_GLUON_FIELDS


# -- lattice kinematics ------------------------------------------------------

#: The small suite does not enable short-distance matching, but it still names
#: the observable it folds so that switching matching on cannot silently turn
#: an unreduced-QCF fixture into a reduced-ratio closure.
PITD_OBSERVABLE = "reduced_ratio"
PITD_OPERATOR_SCHEME = "msbar"
PITD_KERNEL = "unpolarized"
PITD_DATA_NORMALIZATION = "a1_prescaled"
PITD_LORENTZ_COMPONENT = "temporal"

HBARC_GEV_FM = 0.1973269804
LATTICE_L = 48
LATTICE_A_FM = 0.06
Z_PHYS_FM = 0.3                                   # single separation: z = 0.3 fm
Z_LAT = Z_PHYS_FM / LATTICE_A_FM                  # separation in lattice units (=5)
P_VALUES = (1, 2, 3, 4, 5, 6, 7)                  # integer momenta; pz <~ 3 GeV
Z_INV_GEV = 1.0                                   # legacy: standard-path separation

#: physical momentum of unit index: pz = p * 2*pi*hbarc / (L*a) [GeV].
PZ_UNIT_GEV = 2.0 * 3.141592653589793 * HBARC_GEV_FM / (LATTICE_L * LATTICE_A_FM)

#: Mellin moments at 2 GeV.  The data builder fixes CP from exponent parity:
#: <x> (stored order N=2) uses the C-odd field and <x^2> (N=3) the C-even field.
MOMENT_Q2 = 4.0                                   # (2 GeV)^2
CP_EVEN_MELLIN_ORDER = 3                          # <x^2> -> C-even field
CP_ODD_MELLIN_ORDER = 2                           # <x>   -> C-odd field

# -- DEPRECATED error-model constants ---------------------------------------
# The closure fake-data covariance now comes entirely from the NNPDF replica
# spread of the transformed output (see closure_NNPDF_truth_small/generate.py): both lattice and
# experimental data keep the diagonal of that covariance and are jiggled.  These
# hand-tuned relative-error ramps / floors are no longer used by generation.
# Kept only to avoid breaking any incidental readers.
ITD_REL_MIN = 0.01
ITD_REL_MAX = 0.10
ITD_ABS_FLOOR = 1.0e-5
MOMENT_REL_EVEN = 0.10
MOMENT_REL_ODD = 0.01
MOMENT_ABS_FLOOR = 1.0e-5


def itd_rel_error(p: float) -> float:
    """DEPRECATED: relative pseudo-ITD error ramp (unused by generation now)."""
    p_min, p_max = float(min(P_VALUES)), float(max(P_VALUES))
    if p_max == p_min:
        return ITD_REL_MIN
    frac = (float(p) - p_min) / (p_max - p_min)
    return ITD_REL_MIN + (ITD_REL_MAX - ITD_REL_MIN) * frac


# -- DIS experimental preset -------------------------------------------------
# Real fitpack tables supplying kinematics + statistical precision.  F2, NC
# sigma_r, and the massless NLO HERA CC reduced cross sections are modelled
# directly; small differs only by its common per-table row cap.

class ExpSpec:
    """One DIS table used for closure fake data."""

    __slots__ = ("idx", "obs", "target", "kind", "label")

    def __init__(self, idx, obs, target, kind, label):
        self.idx = idx
        self.obs = obs
        self.target = target
        self.kind = kind          # "f2" | "sigma_r" | "sigma_r_cc"
        self.label = label

    def __repr__(self):
        return f"ExpSpec({self.idx}, {self.kind}, {self.target}, {self.label!r})"


EXP_SPECS = (
    ExpSpec(10010, "F2", "proton", "f2", "slac_p_f2"),
    ExpSpec(10016, "F2", "proton", "f2", "bcdms_p_f2"),
    ExpSpec(10020, "F2", "proton", "f2", "nmc_p_f2"),
    ExpSpec(10011, "F2", "deuteron", "f2", "slac_d_f2"),
    ExpSpec(10017, "F2", "deuteron", "f2", "bcdms_d_f2"),
    ExpSpec(10026, "sig_r", "proton", "sigma_r", "hera_nc_ep_318_sigmar"),
    ExpSpec(10030, "sig_r", "proton", "sigma_r", "hera_nc_em_318_sigmar"),
    ExpSpec(10031, "sig_r", "proton", "sigma_r_cc", "hera_cc_ep_318_sigmar"),
    ExpSpec(10032, "sig_r", "proton", "sigma_r_cc", "hera_cc_em_318_sigmar"),
)

#: DIS selection cuts (match the legacy unpolarized preset; keep evolution
#: upward from the input scale).  ``None`` means there is no upper-Q2 cap, so the
#: HERA high-Q2 DIS points remain eligible before the per-table subsampling.
DIS_Q2_MIN = Q0_2                                 # >= input scale
DIS_Q2_MAX = None                                 # no upper-Q2 cap
DIS_W2_MIN = 10.0
MAX_POINTS_PER_EXP_DATASET = 16                   # subsample (few distinct Q2)
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
    # 10007 (HERMES p sig_r, 7.6% -- the largest normalization in the set) was
    # listed here but is NOT in EXP_SPECS, so nothing generated the table, the
    # offset was never injected and no pull was ever recovered: it read as
    # normalization coverage and was dead.  Removed rather than promoted, because
    # adding the table would change what every default small run fits.  The full
    # suite does carry 10007 (closure_*_truth/config.py ANALYSIS_IDIS_IDX), and it
    # is the natural first addition if this suite ever takes more DIS tables.
    # Guarded by tests/test_closure_small_config_consistency.py.
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
#: covariance.  This matches the JAM nuisance-parameter methodology for every
#: experimental dataset; correlated systematics remain marginalized.
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
DY_ANALYSIS = {
    10001: ("pp", "dy_e866_pp"),
    10002: ("pd", "dy_e866_pd"),
}

#: DY kinematic cuts.  Require physical parton momentum fractions x1, x2 < 1 and
#: keep the perturbative Q2 (fixed-target DY is already Q2 >> Q0^2).
DY_X_MAX = 1.0
DY_Q2_MIN = Q0_2

#: Drell-Yan is *bilinear*: the model materializes a dense (n_rows, n, n) tensor
#: per contribution, and the full-basis x full-channel luminosity expands to
#: ~460 contributions, so memory scales with the DY row count.  We therefore bin
#: each E866 table to at most DY_MAX_ROWS rows (subsampled to preserve the x1/x2
#: spread).  Set to None for the full ~180-row coverage (needs a large machine).
DY_MAX_ROWS = 6

#: DY hard-kernel scheme.  LO (Born luminosity) is positive-definite and stable;
#: at the closure's no-evolution mc scale the NLO off-scale corrections overshoot
#: (large/negative at edge kinematics), so LO is the clean closure choice.  Both
#: sides use it identically, so the closure stays self-consistent.
DY_ORDER = "LO"
DY_ALPHA_EM = 1.0 / 137.035999
DY_ALPHA_S = 0.20                                 # fixed alpha_s at the DY scale
DY_NF = 4                                         # active flavors (u,d,s,c)
DY_CHANNELS = (("qA,qbB", "qbA,qB") if DY_ORDER == "LO" else
               ("qA,qbB", "qbA,qB", "qA,gB", "gA,qB"))

#: Documented closure approximation: the DY hard tensor uses per-row muF^2 = Q^2
#: but no PDF evolution matrix is composed (fields stay at Q0 = mc).  Generation
#: and fit share this exactly, so the closure is self-consistent; the absolute DY
#: normalization is not physical but the sea-constraining information is.
DY_EVOLVE = False

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


def dy_field_maps(reaction: str):
    """Return ``(fields_A, fields_B)`` parton->basis maps for a DY reaction.

    ``pp`` uses the proton map on both sides; ``pd`` uses proton on side A and
    the isoscalar deuteron map on side B.
    """
    if reaction == "pp":
        return DY_PARTON_BASIS_PROTON, DY_PARTON_BASIS_PROTON
    if reaction == "pd":
        return DY_PARTON_BASIS_PROTON, DY_PARTON_BASIS_DEUTERON
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
#: Per-field tied amplitude.  The two-tier `GP_AMPLITUDE_HIGH/LOW` constants were
#: replaced by this table on 2026-08-15 so individual fields can be tuned, but the
#: **values here are the original tiers** (5.0 for singlet/gluon/charm, 1.0 for the
#: rest): a retune to the observed field scales -- `t3` 0.25, `t8` 0.5, `t15` 2.0,
#: valence 0.2-0.5 -- was measured and **reverted**.
#:
#: Why it was reverted, so it is not retried blind:
#:
#: * It could not work for `t3` in `exp`.  The prior is `a +- a` with the mean tied
#:   to the width, and `t3` is prior-dominated there (it is a 1-6% perturbation on
#:   proton F2 across the coverage bulk, and deuteron F2 carries *zero* `t3` charge
#:   weight).  Offset and posterior width both scale with `a`, so their ratio -- the
#:   pull -- is nearly invariant: measured RMS pull 1.47 -> 1.40 for `a` 1.0 -> 0.25.
#: * It helped only where the data already pins the width.  In `both`, 8 lattice
#:   points see `t3` and nothing else, fixing sigma from data, so dropping the mean
#:   removed bias: 1.26 -> 0.61.
#: * It broke the valence sector.  `v8` RMS pull went 0.51 -> 2.65, a 4-sigma
#:   one-directional miss, because `v8 = 0.2 +- 0.2` cannot satisfy the hard
#:   quark-counting constraint `int V8 = 3`.  Net scorecard 8 better / 17 worse.
#:
#: Amplitudes are to be re-derived only *after* a prior mean that vanishes as
#: x -> 0 exists; until then a single positive constant cannot track a `t3` truth
#: running 0.005 at x=1e-4 to 0.2 at x=0.3, and tuning it just moves the problem.
GP_AMPLITUDES = {
    "t3": 0.25,
    "t8": 0.5,
    "sigma": 5.0,
    "t15": 2.0,
    "g": 5.0,
    "v3": 0.25,
    "v8": 0.2,
    "v": 0.5,
    "v15": 0.5,
}


def gp_amplitude(field: str) -> float:
    """Tied prior mean *and* log-RBF sigma, one value per field."""
    return GP_AMPLITUDES[field]


#: Floating the tied amplitude.  Off: both ends of every tie stay frozen and the
#: hyper-prior contributes nothing.  On: each field's amplitude becomes a free
#: parameter with a hard floor and a log-normal hyper-prior centred on its own
#: tier value -- the mitigation for the ``a -> 0`` collapse that tying the mean to
#: the width introduces (see guides/tying_parameters.md).
GP_AMPLITUDE_FREE = False
#: Hard floor on a *fitted* amplitude, so the prior width can never collapse.
#: Derived from the smallest amplitude in the table rather than hardcoded: the
#: old literal 0.99 was written against ``GP_AMPLITUDE_LOW = 1.0`` and would now
#: sit *above* six of the nine per-field amplitudes.  That is not merely inert
#: while ``GP_AMPLITUDE_FREE`` is off -- a bound at or above the starting value
#: makes ``inv_softplus(a - floor)`` non-finite and sends ``eta`` to ``-inf``
#: under the sampler's lower-softplus transform
#: (pixel.infer.hmc.ParameterCoordinates).
#:
#: It stays a single **absolute** floor, which is what the 0.99 literal always
#: was: 99% of the low tier but only 20% of the high tier, i.e. permissive for
#: the large-amplitude fields.  A per-field *fraction* was tried first and is
#: wrong -- 0.99 of each field's own value is a 1% leash that rejects any
#: meaningful excursion, and it made three tie tests raise "value is below its
#: lower bound" as soon as they perturbed an amplitude.
#: The 0.99 factor keeps it a hair under the smallest amplitude rather than on
#: it, for the same eta-finiteness reason.
GP_AMPLITUDE_FLOOR = 0.99 * min(GP_AMPLITUDES.values())
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
GP_JITTER = 1.0e-10


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

#: Sample cap for ``synthetic_z_plumbing`` only.  That mode samples its single
#: Hubbard-Stratonovich auxiliary through ``saddle_contour_importance_sample``,
#: whose cost and footprint both grow quadratically in the sample count -- for a
#: one-row dataset.  Measured on this box, solo, nothing else running:
#:
#:     n=384 -> 32.5s    n=512 -> 58.9s    n=640 -> 114.3s    n=1000 -> SIGKILL
#:
#: The 1000-sample run was killed twice at a peak RSS of ~23 GB, so the
#: configured ``MCMC_SAMPLES`` cannot run this mode at all.  512 is the largest
#: count measured to complete with headroom left for a shared machine.  This is
#: a workaround, not a fix: ~23 MB per sample for a single data point is a
#: memory defect in that sampler and is worth chasing separately.  It is
#: tolerable here only because this mode is plumbing, never physics validation.
SYNTHETIC_Z_MCMC_SAMPLES = 512

# Small-closure inference uses VEGAS for every nontrivial sampled fit.  In a
# linear DIS fit this is ordinary adaptive-grid importance sampling over the
# explicit table normalizations.  In DY-containing fits one joint VEGAS grid
# covers those normalizations and the coordinates on a fixed-reference affine
# H-S contour.  The normalization parameters are never marginalized.
VEGAS_N_BINS = 16
VEGAS_N_ADAPT_ITERATIONS = 5
VEGAS_N_EVAL_PER_ITERATION = 128
# Keep the usual five passes for well-conditioned cases.  Only extend a hard
# joint grid, by at most five cheap pilot passes, when its independent phase-ESS
# proxy is still too small to support the production-quality gate below.
VEGAS_MIN_ADAPT_PHASE_ESS_FRAC = 0.25
VEGAS_MAX_EXTRA_ADAPT_ITERATIONS = 5
# These are the bounded-memory nested fallback settings.  The public nested
# VEGAS default is 128 inner samples; the small suite selects the faster joint
# grid below, for which each joint point contains one H-S coordinate draw.
VEGAS_N_ADAPT_INNER_SAMPLES = 128
VEGAS_N_INNER_SAMPLES = 1
VEGAS_JOINT_GRID = True
VEGAS_ALPHA = 0.5
VEGAS_COVARIANCE_INFLATION = 1.5
# The joint normalization/H-S proposal is already built from the exact local
# magnitude Hessian in 19 dimensions.  Inflating every direction by 1.5 makes
# its tails collectively much too broad; use the Laplace width itself.
VEGAS_JOINT_COVARIANCE_INFLATION = 1.0
VEGAS_MIN_OUTER_ESS_FRAC = 0.10
VEGAS_MIN_SIGNED_ESS_FRAC = 0.05
VEGAS_MIN_MEDIAN_INNER_ESS_FRAC = 0.10

# Retained for the public HMC/NUTS examples and regression comparisons; the
# small closure dispatch itself no longer selects either Markov-chain sampler.
NUTS_WARMUP = 100
NUTS_INITIAL_STEP_SIZE = 0.1
NUTS_INTEGRATOR = "leapfrog"
NUTS_MAX_TREE_DEPTH = 6
# The Hessian metric is already tailored to the local target. Avoid rebuilding
# NUTS steppers and repeating costly step-size searches at mass-window updates.
NUTS_ADAPT_MASS = False

#: Mixed DIS+DY fits keep every normalization explicit.  A short zero-flow HMC
#: trajectory is both exact and stable after the retained-subspace derivative
#: fix; avoiding a dense Hessian setup saves several minutes per small fit.
HMC_WARMUP = 100
HMC_STEP_SIZE = 0.25
HMC_N_LEAPFROG = 3
HMC_INTEGRATOR = "leapfrog"
HMC_USE_HESSIAN_MASS = False
HMC_FLOW_TIME = 0.0
HMC_N_FLOW_STEPS = 1
HMC_AFFINE_REFERENCE_CONTOUR = True
# Some closure likelihoods stall just below a 3e-3 Newton decrement at
# floating-point precision.  Keep a small margin so a numerically stationary
# H-S saddle is accepted instead of aborting contour-NUTS initialization.
CONTOUR_SADDLE_TOL = 5.0e-3
# The DY-only small closure uses independent importance draws on the fixed A=0
# saddle contour.  Below this residual complex-weight ESS fraction the proposal
# is not resolving the integral reliably, so the fit fails instead of publishing
# misleading closure bands.
SADDLE_MIN_ESS_FRAC = 0.10

N_HYPERPARAMS = len(ALL_FIELDS) if GP_AMPLITUDE_FREE else 0  # 0 = fully-fixed prior

#: ``synthetic_z_plumbing`` is deliberately a one-row, unevolved, boson-level
#: gamma/Z proxy.  It tests EW dataset plumbing only and must never be reported
#: as physics Z coverage; all audited collider tables remain in the full suites.
TEST_MODE_SYNTHETIC_Z = "synthetic_z_plumbing"
#: Test modes: ``lattice`` (lattice only), ``dis`` (DIS only), ``dy`` (Drell-Yan
#: only), ``exp`` (all experiment = DIS + DY), ``both`` (lattice + DIS + DY),
#: and the explicitly non-physics EW smoke mode above.
TEST_MODES = ("lattice", "dis", "dy", "exp", "both", TEST_MODE_SYNTHETIC_Z)

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
CONSTRAINT_NORM_SIGMA = 1.0e-4                    # ~hard: <x^0>, valence integrals
CONSTRAINT_ORIGIN_SIGMA = 1.0e-4                  # ~hard: x*f(x=0)=0
#: ~hard: <x^1>, the momentum sum rule int x (Sigma + g) dx = 1.  Added 2026-08-15;
#: ``sigma`` and ``g`` were previously the only fields carrying no integral
#: constraint at all, and ``g`` showed a deterministic -0.16 signed-pull bias
#: (3.5 sigma over 24 noise replicas, 19/24 negative).
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
    """The shared 128-point log-linear PDF grid."""
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
