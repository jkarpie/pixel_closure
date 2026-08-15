"""Helpers for the saved standard closure-test mock dataset."""

from __future__ import annotations

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "standard"
TRUTH_PATH = DATA_ROOT / "truth.json"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
LATTICE_DIR = DATA_ROOT / "lattice"
EXP_DIR = DATA_ROOT / "exp"
KERNEL_CACHE_DIR = DATA_ROOT / "kernel_cache"

Q0 = 2.0
Q0_2 = Q0**2
NF = 4
GRID_N = 513
GRID_SPACING = "log"
X_MIN = 1.0e-6
ELEMENT_TYPE = "piecewise"

LATTICE_L = 48
LATTICE_A_FM = 0.06
Z_INV_GEV = 1.0
PITD_OBSERVABLE = "reduced_ratio"
PITD_OPERATOR_SCHEME = "msbar"

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

EVEN_FIELDS = ("t3", "t8", "sigma", "t15", "g")
ODD_FIELDS = ("v3", "v8", "v", "v15")


def field_cparity(field: str) -> str:
    if field in EVEN_FIELDS:
        return "even"
    if field in ODD_FIELDS:
        return "odd"
    raise KeyError(f"unknown standard-closure field {field!r}")


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


def standard_paths() -> dict[str, object]:
    """Return the saved standard closure dataset paths."""
    return {
        "root": DATA_ROOT,
        "truth": TRUTH_PATH,
        "manifest": MANIFEST_PATH,
        "lattice": sorted(LATTICE_DIR.glob("*.dat")),
        "exp": sorted(EXP_DIR.glob("*.dat")),
        "kernel_cache": KERNEL_CACHE_DIR,
    }


def load_manifest() -> dict:
    """Read the standard fixture manifest."""
    return json.loads(MANIFEST_PATH.read_text())


def load_truth() -> dict:
    """Read the standard fixture truth record."""
    return json.loads(TRUTH_PATH.read_text())


def make_grid():
    """Create the standard 513-point logarithmic PDF grid."""
    from pixel.geometry import Grid

    return Grid(n_points=GRID_N, spacing=GRID_SPACING, x_min=X_MIN)


def make_fields():
    """Create all standard closure fields on the shared PDF grid."""
    from pixel.core.model import Field

    grid = make_grid()
    return {
        name: Field.create(name, grid, element_type=ELEMENT_TYPE)
        for name in FIELD_TO_BASIS
    }


def _cache_path(section: str, stem: str, *, enabled: bool = True):
    if not enabled:
        return None
    return KERNEL_CACHE_DIR / section / f"{stem}.npz"


def exp_maps(fields):
    """Return C-even and C-odd DIS maps for a standard field dictionary."""
    even = {basis: fields[field] for basis, field in EVEN_MAP.items()}
    odd = {basis: fields[field] for basis, field in ODD_MAP.items()}
    return even, odd


def load_exp_datasets(*, fields=None, use_kernel_cache: bool = True):
    """Load saved experimental datasets with stable kernel-cache paths.

    This only constructs PIXEL Dataset objects.  The `.npz` cache files are
    created later by PIXEL when a kernel matrix is actually evaluated.
    """
    from pixel import data

    fields = make_fields() if fields is None else fields
    even, odd = exp_maps(fields)
    datasets = []
    for rec in load_manifest()["exp"]:
        if "file" not in rec:
            continue
        path = DATA_ROOT / rec["file"]
        cache_path = _cache_path("exp", Path(rec["file"]).stem, enabled=use_kernel_cache)
        if rec["kind"] == "f2":
            dataset = data.F2.from_file(
                path,
                target=rec["target"],
                maps_to=even,
                nf=NF,
                order="LO",
                cache_path=cache_path,
            )
        elif rec["kind"] == "sigma_r":
            kwargs = {}
            if "y" in rec:
                kwargs["y"] = rec["y"]
            elif "sqrt_s" in rec:
                values = rec["sqrt_s"]
                first = float(values[0])
                if all(abs(float(v) - first) < 1e-12 for v in values):
                    kwargs["sqrt_s"] = first
                else:
                    x = rec["x"]
                    q2 = rec["Q2"]
                    kwargs["y"] = [float(q) / (float(xx) * float(rs) ** 2) for xx, q, rs in zip(x, q2, values)]
            dataset = data.SigmaR.from_file(
                path,
                target=rec["target"],
                maps_to=even,
                nf=NF,
                order="LO",
                cache_path=cache_path,
                **kwargs,
            )
        elif rec["kind"] == "f3_proxy":
            dataset = data.F3.from_file(
                path,
                target=rec["target"],
                maps_to=odd,
                nf=NF,
                order="LO",
                cache_path=cache_path,
            )
        else:
            raise ValueError(f"unknown experimental closure kind {rec['kind']!r}")
        datasets.append(dataset)
    return datasets


def load_lattice_datasets(*, fields=None, use_kernel_cache: bool = True):
    """Load saved lattice datasets with stable kernel-cache paths.

    This only constructs PIXEL Dataset objects.  The `.npz` cache files are
    created later by PIXEL when a kernel matrix is actually evaluated.
    """
    from pixel import data

    fields = make_fields() if fields is None else fields
    datasets = []
    for rec in load_manifest()["lattice"]:
        path = DATA_ROOT / rec["file"]
        cache_path = _cache_path("lattice", Path(rec["file"]).stem, enabled=use_kernel_cache)
        if rec["kind"] == "pseudoitd":
            dataset = data.PseudoITD.from_file(
                path,
                L=LATTICE_L,
                a=LATTICE_A_FM,
                maps_to=fields[rec["field"]],
                component=rec["component"],
                cache_path=cache_path,
                pseudoitd_observable=PITD_OBSERVABLE,
                operator_scheme=PITD_OPERATOR_SCHEME,
                pseudoitd_kernel=PITD_KERNEL,
                pseudoitd_data_normalization=PITD_DATA_NORMALIZATION,
                pseudoitd_lorentz_component=PITD_LORENTZ_COMPONENT,
            )
        elif rec["kind"] == "mellin" and "cp_even_field" in rec:
            dataset = data.MellinMoment.from_file(
                path,
                L=LATTICE_L,
                a=LATTICE_A_FM,
                maps_to={
                    "cp_even": fields[rec["cp_even_field"]],
                    "cp_odd": fields[rec["cp_odd_field"]],
                },
                cache_path=cache_path,
            )
        elif rec["kind"] == "mellin":
            dataset = data.MellinMoment.from_file(
                path,
                L=LATTICE_L,
                a=LATTICE_A_FM,
                maps_to={
                    f"cp_{field_cparity(rec['field'])}": fields[rec["field"]]
                },
                cache_path=cache_path,
            )
        else:
            raise ValueError(f"unknown lattice closure kind {rec['kind']!r}")
        datasets.append(dataset)
    return datasets


def load_datasets(*, fields=None, include_exp: bool = True, include_lattice: bool = True, use_kernel_cache: bool = True):
    """Load the standard closure datasets selected by source."""
    fields = make_fields() if fields is None else fields
    out = []
    if include_exp:
        out.extend(load_exp_datasets(fields=fields, use_kernel_cache=use_kernel_cache))
    if include_lattice:
        out.extend(load_lattice_datasets(fields=fields, use_kernel_cache=use_kernel_cache))
    return out
