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
from pixel.data.nnpdf40_native import (
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
_CONTRACT_SELECTION = _full_closure_selection("jam_full")
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
#: The LHAPDF grid tarball is vendored in the repo so a fresh checkout can
#: rebuild the truth without a separate LHAPDF set download, and so rebuilding
#: LHAPDF (which does not carry sets across) cannot silently disable the closure
#: suite again.  Extracted on demand into :data:`JAM_LHAPDF_DATA_ROOT`.
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

    Prefer the installed LHAPDF data directory, then fall back to reading the
    ``.info`` member straight out of the download tarball so the member count
    stays correct even when the LHAPDF install has been rebuilt without its
    sets (which is exactly how these constants went stale before).

    Args:
        set_name: LHAPDF set name; defaults to :data:`JAM_SET`.

    Returns:
        dict: Scalar ``key -> value`` pairs, empty if the set cannot be found.
    """
    name = JAM_SET if set_name is None else set_name
    leaf = f"{name}.info"
    for root in (JAM_LHAPDF_DATA_ROOT, LHAPDF_DATA_DIR):
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
#: ~460 contributions, so memory scales with the DY row count.  The full closure
#: deliberately keeps every row surviving the physics cuts (currently 179 pp +
#: 183 pd E866 rows).  Set an integer only for an explicitly thinned run.
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
DY_ALPHA_EM = 1.0 / 137.035999
DY_ALPHA_S = 0.20                                 # fixed alpha_s at the DY scale

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
#: transform (pixel.infer.hmc.ParameterCoordinates).
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


GP_LENGTH_LOG = 0.6931471805599453                # log(2); fixed length (log-x)
FROZEN_COV_PARAMS = ("sigma", "length", "x_reg")  # all frozen
# Matches pixel.util.linalg.DEFAULT_RCOND.  svd_factor preconditions W to unit
# diagonal before applying this cutoff, so it is a cut on the *correlation*
# matrix and is invariant to the units each dataset is stored in.
RCOND = 1.0e-16

MAP_METHOD = "L-BFGS-B"
MAP_OPTIONS = {"maxiter": 200}
MCMC_SAMPLES = 1000
MCMC_SEED = 0
# Closure inference uses NUTS throughout. Bilinear Drell-Yan modes dispatch to
# contour NUTS, while linear modes use ordinary real-parameter NUTS. Both start
# from a Hessian-derived diagonal inverse mass.
NUTS_WARMUP = 100
NUTS_INITIAL_STEP_SIZE = 0.1
NUTS_INTEGRATOR = "leapfrog"
NUTS_MAX_TREE_DEPTH = 6
# The Hessian metric is already tailored to the local target. Avoid rebuilding
# NUTS steppers and repeating costly step-size searches at mass-window updates.
NUTS_ADAPT_MASS = False

# Exact bilinear-ratio modes cannot use marginalized/H-S inference.  These
# settings govern the explicit retained-(theta,q) NUTS path; physics promotion
# additionally requires the cutoff sweep and four-chain feasibility report.
DIRECT_PRIOR_RCOND = 1.0e-10
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
#: Tightened 1e-4/1e-3 -> 1e-8 on 2026-08-15, together with the _small suites, because
#: tests/test_closure_constraints.py enforces that all four packages share these values
#: and the argument applies equally: these are the ``lambda`` of gp_paper.tex
#: Eq.(constraints), and "in the limit that these lambda parameters go to 0, their
#: respective pieces of prior information are constrained exactly".  Verified to bind:
#: the posterior reproduces the represented momentum target to 5e-12, so RCOND = 1e-16
#: is not truncating the constraint row despite its 1e-16 variance.
CONSTRAINT_ENDPOINT_SIGMA = 1.0e-8                # ~hard: f(x=1)=0
CONSTRAINT_NORM_SIGMA = 1.0e-8                    # ~hard: valence integrals
CONSTRAINT_ORIGIN_SIGMA = 1.0e-8                  # ~hard: x*f(x=0)=0
#: ~hard: <x^1>, the momentum sum rule int x (Sigma + g) dx = 1.  Added 2026-08-15
#: alongside the same constraint in the _small suites; `sigma` and `g` previously
#: carried no integral constraint at all.  The target is the *represented* truth,
#: not the nominal 1.0 -- see `constraint_datasets.represented_sum_target`.
CONSTRAINT_MOMENTUM_SIGMA = 1.0e-8
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
