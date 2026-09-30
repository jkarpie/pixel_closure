"""Fixed-prior scan cells: grid definitions, validation, hashing, and the in-process patch.

A *cell* is one complete, fixed choice of prior hyperparameters for one prior form.  Every
value is resolved to an absolute table before anything runs -- a cell never says
"x2 the config value", because the config value can change between the day a scan is
expanded and the day its rows are read.  The resolved dict is what gets hashed, so two
rows with the same ``cell_hash`` were fitted with the same prior, byte for byte.

Nothing here edits ``config.py``.  :func:`patched_prior` assigns module attributes that
``fit.py`` reads at prior-build time (``cfg.gp_amplitude`` -> ``GP_AMPLITUDES``,
``cfg.gp_envelope`` -> ``GP_ENVELOPE``, ``GP_LENGTH_LOG``, both jitters,
``GP_AMPLITUDE_FLOOR`` as the amplitude bound) and restores the original objects on exit.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIELDS = ("t3", "t8", "sigma", "t15", "g", "v3", "v8", "v", "v15")
PRIOR_FORMS = ("const_logrbf", "beta_envelope")
GROUPS = {
    "sea_gluon": ("sigma", "g"),
    "t15": ("t15",),
    "nonsinglet": ("t3", "t8"),
    "valence": ("v3", "v8", "v", "v15"),
    "flat": ("t8", "sigma", "t15", "g"),        # envelope fields whose shipped alpha is 0
    "tapered": ("t3", "v3", "v8", "v", "v15"),
}
#: Bounds the closure's own config tests enforce on a shipped envelope
#: (``tests/test_closure_envelope_config.py``); a scan stays inside them.
ALPHA_RANGE = (0.0, 0.8)
BETA_RANGE = (0.5, 4.0)
#: Keys a resolved cell may carry.  Anything else is refused: a knob that is not wired
#: still produces results, which is how a scan silently measures nothing.
CELL_KEYS = {
    "name", "prior_form", "GP_AMPLITUDES", "GP_ENVELOPE", "GP_LENGTH_LOG",
    "GP_JITTER", "GP_ENVELOPE_JITTER", "provenance",
}
PATCHED = (
    "PRIOR_FORM", "GP_AMPLITUDES", "GP_AMPLITUDE_FLOOR", "GP_ENVELOPE",
    "GP_LENGTH_LOG", "GP_JITTER", "GP_ENVELOPE_JITTER", "RESULTS_ROOT",
)


def _fmt(value) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def validate(cell: dict) -> dict:
    """Refuse a cell that is incomplete, unknown-keyed, or outside the tested ranges."""
    unknown = set(cell) - CELL_KEYS
    if unknown:
        raise ValueError(f"cell {cell.get('name')!r}: unknown keys {sorted(unknown)}")
    if cell.get("prior_form") not in PRIOR_FORMS:
        raise ValueError(f"cell {cell.get('name')!r}: prior_form must be one of {PRIOR_FORMS}")
    for key in ("GP_LENGTH_LOG", "GP_JITTER", "GP_ENVELOPE_JITTER"):
        value = float(cell[key])
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"cell {cell['name']!r}: {key} must be positive, got {value}")
    amplitudes = cell["GP_AMPLITUDES"]
    if set(amplitudes) != set(FIELDS):
        raise ValueError(f"cell {cell['name']!r}: GP_AMPLITUDES must name exactly {FIELDS}")
    for field, value in amplitudes.items():
        if not (math.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"cell {cell['name']!r}: amplitude {field}={value} must be positive")
    envelope = cell["GP_ENVELOPE"]
    if set(envelope) != set(FIELDS):
        raise ValueError(f"cell {cell['name']!r}: GP_ENVELOPE must name exactly {FIELDS}")
    for field, env in envelope.items():
        alpha, beta, sigma = float(env["alpha"]), float(env["beta"]), float(env["sigma"])
        if not ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]:
            raise ValueError(f"cell {cell['name']!r}: {field} alpha={alpha} outside {ALPHA_RANGE}")
        if not BETA_RANGE[0] <= beta <= BETA_RANGE[1]:
            raise ValueError(f"cell {cell['name']!r}: {field} beta={beta} outside {BETA_RANGE}")
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError(f"cell {cell['name']!r}: {field} sigma={sigma} must be positive")
    return cell


def resolved(cell: dict) -> dict:
    """The part of a cell that defines the fit: everything except name and provenance."""
    return {k: cell[k] for k in sorted(cell) if k not in ("name", "provenance")}


def cell_hash(cell: dict) -> str:
    text = json.dumps(resolved(cell), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def base_cell(cfg, prior_form: str, name: str) -> dict:
    """A cell holding exactly the suite's shipped values."""
    return {
        "name": name,
        "prior_form": prior_form,
        "GP_AMPLITUDES": {f: float(cfg.GP_AMPLITUDES[f]) for f in FIELDS},
        "GP_ENVELOPE": {f: {k: float(v) for k, v in cfg.GP_ENVELOPE[f].items()} for f in FIELDS},
        "GP_LENGTH_LOG": float(cfg.GP_LENGTH_LOG),
        "GP_JITTER": float(cfg.GP_JITTER),
        "GP_ENVELOPE_JITTER": float(cfg.GP_ENVELOPE_JITTER),
        "provenance": {"base": "config"},
    }


def truth_bulk_max(q_key: str = "2", x_range=(0.01, 0.9)) -> dict:
    """``max |x f|`` per field over the bulk, maximized over both small suites' truths."""
    out = {f: 0.0 for f in FIELDS}
    for suite in ("closure_JAM_truth_small", "closure_NNPDF_truth_small"):
        truth = json.loads((REPO / suite / "data" / f"truthQ_{q_key}" / "truth.json").read_text())
        x = np.asarray(truth["x_nodes"], dtype=float)
        mask = (x >= x_range[0]) & (x <= x_range[1])
        for f in FIELDS:
            out[f] = max(out[f], float(np.max(np.abs(np.asarray(truth["curves"][f])[mask]))))
    return out


def const_grid_s1(cfg) -> list[dict]:
    """Stage-1 screen for ``const_logrbf`` (Const mean tied to LogRBF sigma, ``a +- a``)."""
    cells = []
    base = base_cell(cfg, "const_logrbf", "c_config")
    cells.append(base)

    def with_amplitudes(name, amplitudes, provenance):
        cell = copy.deepcopy(base)
        cell["name"] = name
        cell["GP_AMPLITUDES"] = {f: float(amplitudes[f]) for f in FIELDS}
        cell["provenance"] = provenance
        return cell

    shipped = dict(base["GP_AMPLITUDES"])
    # C1: sea/gluon plane x t15, everything else as shipped.
    for s, g, t15 in itertools.product((0.5, 1, 2, 3.5, 5), (0.5, 1, 2, 3.5, 5, 6.5), (0.5, 1, 2, 4)):
        amps = dict(shipped, sigma=s, g=g, t15=t15)
        cells.append(with_amplitudes(
            f"c1_s{_fmt(s)}_g{_fmt(g)}_t15{_fmt(t15)}", amps,
            {"grid": "C1", "sigma": s, "g": g, "t15": t15}))
    # C2: nonsinglet x valence multipliers x t15 at three sea/gluon anchors.
    ns_base = {"t3": 0.25, "t8": 0.5}
    val_base = {"v3": 0.25, "v8": 0.5, "v": 0.5, "v15": 0.5}
    for (s, g), ns, val, t15 in itertools.product(
        ((5, 5), (2, 2), (1, 1)), (0.5, 1, 2), (0.5, 1, 2), (0.5, 1, 2, 4)
    ):
        amps = dict(shipped, sigma=s, g=g, t15=t15)
        amps.update({f: ns * v for f, v in ns_base.items()})
        amps.update({f: val * v for f, v in val_base.items()})
        cells.append(with_amplitudes(
            f"c2_s{_fmt(s)}_g{_fmt(g)}_ns{_fmt(ns)}_val{_fmt(val)}_t15{_fmt(t15)}", amps,
            {"grid": "C2", "sigma": s, "g": g, "nonsinglet": ns, "valence": val, "t15": t15}))
    # Truth-derived: k x bulk max |x f| over both truths at Q=2.
    bulk = truth_bulk_max("2")
    for k in (0.5, 1, 2, 4):
        cells.append(with_amplitudes(
            f"ct_k{_fmt(k)}", {f: k * bulk[f] for f in FIELDS},
            {"grid": "truth", "k": k, "stat": "bulk_max", "x_range": [0.01, 0.9]}))
    # Control: the momentum-sum-rule amplitudes for sigma and g.
    cells.append(with_amplitudes(
        "c_sumrule", dict(shipped, sigma=0.601, g=0.399), {"grid": "control", "sigma": 0.601, "g": 0.399}))
    return _unique([validate(c) for c in cells])


def _unique(cells: list[dict]) -> list[dict]:
    """Drop cells whose resolved prior repeats an earlier one (the first name wins)."""
    seen, out = set(), []
    for cell in cells:
        digest = cell_hash(cell)
        if digest not in seen:
            seen.add(digest)
            out.append(cell)
    return out


def envelope_grid_s1(cfg) -> list[dict]:
    """Stage-1 screen for ``beta_envelope`` (Zero mean, BetaTaperedLogRBF)."""
    base = base_cell(cfg, "beta_envelope", "e_config")
    cells = []
    for scale, flat, da, db in itertools.product(
        (0.003, 0.01, 0.03, 0.1, 0.3, 1.0), (0.1, 0.3, 1.0), (0.0, 0.2, 0.4), (-0.5, 0.0, 0.5)
    ):
        cell = copy.deepcopy(base)
        cell["name"] = f"e_sc{_fmt(scale)}_flat{_fmt(flat)}_da{_fmt(da)}_db{_fmt(db)}"
        if scale == 1.0 and flat == 1.0 and da == 0.0 and db == 0.0:
            cell["name"] = "e_config"
        for f in FIELDS:
            env = cell["GP_ENVELOPE"][f]
            factor = scale * (flat if f in GROUPS["flat"] else 1.0)
            env["sigma"] = float(env["sigma"]) * factor
            env["alpha"] = min(ALPHA_RANGE[1], float(env["alpha"]) + da)
            env["beta"] = min(BETA_RANGE[1], max(BETA_RANGE[0], float(env["beta"]) + db))
        cell["provenance"] = {"grid": "E1", "sigma_scale": scale, "flat_scale": flat,
                              "alpha_shift": da, "beta_shift": db}
        cells.append(validate(cell))
    return cells


def refine_s2(parents: list[dict]) -> list[dict]:
    """Length x jitter refinement around each parent cell (jitter axis per prior form)."""
    lengths = (math.log(1.5), math.log(2.0), math.log(3.0))
    cells = []
    for parent in parents:
        jitters = ((1e-10, 1e-7, 1e-4) if parent["prior_form"] == "const_logrbf" else (1e-3, 1e-2, 1e-1))
        key = "GP_JITTER" if parent["prior_form"] == "const_logrbf" else "GP_ENVELOPE_JITTER"
        for length, jitter in itertools.product(lengths, jitters):
            cell = copy.deepcopy(parent)
            cell["GP_LENGTH_LOG"] = float(length)
            cell[key] = float(jitter)
            cell["name"] = f"{parent['name']}__L{_fmt(round(math.exp(length), 3))}_j{_fmt(jitter)}"
            cell["provenance"] = dict(parent.get("provenance", {}), parent=parent["name"],
                                      length=math.exp(length), jitter=jitter)
            cells.append(validate(cell))
    return cells


@contextlib.contextmanager
def patched_prior(cfg, cell: dict, results_root: Path | None = None):
    """Apply one cell's fixed prior to a suite's ``config`` module, and restore it after."""
    validate(cell)
    saved = {name: getattr(cfg, name) for name in PATCHED}
    try:
        cfg.PRIOR_FORM = cell["prior_form"]
        cfg.GP_AMPLITUDES = {f: float(cell["GP_AMPLITUDES"][f]) for f in FIELDS}
        cfg.GP_AMPLITUDE_FLOOR = 0.99 * min(cfg.GP_AMPLITUDES.values())
        cfg.GP_ENVELOPE = copy.deepcopy(cell["GP_ENVELOPE"])
        cfg.GP_LENGTH_LOG = float(cell["GP_LENGTH_LOG"])
        cfg.GP_JITTER = float(cell["GP_JITTER"])
        cfg.GP_ENVELOPE_JITTER = float(cell["GP_ENVELOPE_JITTER"])
        if results_root is not None:
            cfg.RESULTS_ROOT = Path(results_root)
        yield cfg
    finally:
        for name, value in saved.items():
            setattr(cfg, name, value)
