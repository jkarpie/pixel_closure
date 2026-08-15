"""Generate the saved standard closure-test mock dataset.

LHAPDF is used only here, to create input-scale truth curves with realistic
shape.  The closure test itself reads the saved ASCII/JSON files and does not
need LHAPDF.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pixel import kernels
from pixel.core.model import Field
from pixel.data.experimental._dis_common import PROTON_MASS_GEV, dis_w2
from pixel.geometry import Grid
from pixel.util import flavor

from . import config as _paths
from . import pdf_guidance as _pdf_guidance
from . import standard as cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
# Single-sourced from config so a moved fitpack tree or a rebuilt LHAPDF
# install only has to be updated in one place.
FITPACK_IDIS = _paths.FITPACK_IDIS
LHAPDF_PREFIX = _paths.LHAPDF_PREFIX
# Shape guidance for this legacy fixture comes from the suite's own truth set
# (vendored under ``lhapdf/``), so it needs no separately installed PDF set.
PDF_SET = _paths.NNPDF_SET
PDF_MEMBER = _paths.NNPDF_MEMBER
SEED = 20260710
MAX_POINTS_PER_EXP_DATASET = 10
EXP_ABS_FLOOR = 1e-5
LATTICE_ABS_FLOOR = 1e-5
HBARC_GEV_FM = 0.1973269804


@dataclass(frozen=True)
class ExpSpec:
    idx: int
    obs: str
    target: str
    kind: str
    label: str


EXP_SPECS = (
    ExpSpec(10010, "F2", "proton", "f2", "slac_p_f2"),
    ExpSpec(10016, "F2", "proton", "f2", "bcdms_p_f2"),
    ExpSpec(10020, "F2", "proton", "f2", "nmc_p_f2"),
    ExpSpec(10011, "F2", "deuteron", "f2", "slac_d_f2"),
    ExpSpec(10017, "F2", "deuteron", "f2", "bcdms_d_f2"),
    ExpSpec(10003, "sig_r", "proton", "sigma_r", "jlab_p_sigmar"),
    ExpSpec(10026, "sig_r", "proton", "sigma_r", "hera_nc_ep_318_sigmar"),
    ExpSpec(10030, "sig_r", "proton", "sigma_r", "hera_nc_em_318_sigmar"),
    ExpSpec(10007, "sig_r", "proton", "sigma_r", "hermes_p_sigmar"),
    ExpSpec(10031, "F3", "proton", "f3_proxy", "hera_cc_ep_f3_proxy"),
    ExpSpec(10032, "F3", "proton", "f3_proxy", "hera_cc_em_f3_proxy"),
)


FIELD_PAIRS = (
    ("t3", "v3"),
    ("t8", "v8"),
    ("sigma", "v"),
    ("t15", "v15"),
)


def make_grid() -> Grid:
    return Grid(n_points=cfg.GRID_N, spacing=cfg.GRID_SPACING, x_min=cfg.X_MIN)


def make_fields() -> dict[str, Field]:
    grid = make_grid()
    return {
        name: Field.create(name, grid, element_type=cfg.ELEMENT_TYPE)
        for name in (*cfg.EVEN_FIELDS, *cfg.ODD_FIELDS)
    }


def compile_lhapdf_dump() -> Path:
    return _pdf_guidance.compile_lhapdf_dump()


def dump_lhapdf(nodes: np.ndarray) -> np.ndarray:
    exe = compile_lhapdf_dump()
    env = os.environ.copy()
    lib = str(cfg.LHAPDF_LIB_DIR)
    data_roots = [
        str(_pdf_guidance.ensure_lhapdf_data()), str(cfg.LHAPDF_DATA_DIR)
    ]
    data_roots.extend(
        path for path in env.get("LHAPDF_DATA_PATH", "").split(os.pathsep) if path
    )
    env["LHAPDF_DATA_PATH"] = os.pathsep.join(dict.fromkeys(data_roots))
    env["DYLD_LIBRARY_PATH"] = f"{lib}:{env.get('DYLD_LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    payload = "\n".join(f"{x:.17g}" for x in nodes) + "\n"
    proc = subprocess.run(
        [str(exe), PDF_SET, str(PDF_MEMBER), str(cfg.Q0)],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env=env,
    )
    lines = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped[0] not in "+-.0123456789":
            continue
        lines.append(stripped)
    return np.loadtxt(lines)


def truth_from_lhapdf(nodes: np.ndarray) -> dict[str, np.ndarray]:
    raw = dump_lhapdf(nodes)
    x = raw[:, 0]
    x_safe = np.clip(x, 1e-12, None)
    g = raw[:, 1] / x_safe
    u = raw[:, 2] / x_safe
    ubar = raw[:, 3] / x_safe
    d = raw[:, 4] / x_safe
    dbar = raw[:, 5] / x_safe
    s = raw[:, 6] / x_safe
    sbar = raw[:, 7] / x_safe
    c = raw[:, 8] / x_safe
    cbar = raw[:, 9] / x_safe

    up = u + ubar
    dp = d + dbar
    sp = s + sbar
    cp = c + cbar
    um = u - ubar
    dm = d - dbar
    sm = s - sbar
    cm = c - cbar
    return {
        "t3": up - dp,
        "t8": up + dp - 2.0 * sp,
        "sigma": up + dp + sp + cp,
        "t15": up + dp + sp - 3.0 * cp,
        "g": g,
        "v3": um - dm,
        "v8": um + dm - 2.0 * sm,
        "v": um + dm + sm + cm,
        "v15": um + dm + sm - 3.0 * cm,
    }


def write_table(path: Path, leading: list[np.ndarray], mean: np.ndarray, cov: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([*leading, mean, cov])
    np.savetxt(path, rows, fmt="%.12e")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def choose_rows(df: pd.DataFrame, max_points: int) -> np.ndarray:
    x = np.asarray(df["X"], dtype=float)
    q2 = np.asarray(df["Q2"], dtype=float)
    value = np.asarray(df["value"], dtype=float)
    stat = np.asarray(df["stat_u"], dtype=float)
    w2 = dis_w2(x, q2, m_h=PROTON_MASS_GEV)
    mask = (
        np.isfinite(x)
        & np.isfinite(q2)
        & np.isfinite(value)
        & np.isfinite(stat)
        & (x > 0.0)
        & (x < 1.0)
        & (q2 > cfg.Q0_2)
        & (w2 > 10.0)
        & (value != 0.0)
        & (stat > 0.0)
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    if idx.size <= max_points:
        return idx
    take = np.unique(np.rint(np.linspace(0, idx.size - 1, max_points)).astype(int))
    return idx[take]


def interp_truth(truth: dict[str, np.ndarray], nodes: np.ndarray, x: np.ndarray, field: str) -> np.ndarray:
    return np.interp(x, nodes, truth[field])


def f2_mean(nodes: np.ndarray, truth: dict[str, np.ndarray], x: np.ndarray, target: str) -> np.ndarray:
    coeffs = flavor.f2_charge_coefficients(target, nf=cfg.NF)
    total = np.zeros_like(x, dtype=float)
    for basis_name, weight in coeffs.items():
        field = cfg.EVEN_MAP[basis_name]
        total += weight * interp_truth(truth, nodes, x, field)
    return x * total


def f3_proxy_mean(nodes: np.ndarray, truth: dict[str, np.ndarray], x: np.ndarray, target: str) -> np.ndarray:
    coeffs = flavor.f3_valence_coefficients(target, nf=cfg.NF)
    total = np.zeros_like(x, dtype=float)
    for basis_name, weight in coeffs.items():
        if basis_name == "gluon":
            continue
        field = cfg.ODD_MAP[basis_name]
        total += weight * interp_truth(truth, nodes, x, field)
    return x * total


def exp_mean(spec: ExpSpec, nodes: np.ndarray, truth: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    if spec.kind == "f3_proxy":
        return f3_proxy_mean(nodes, truth, x, spec.target)
    return f2_mean(nodes, truth, x, spec.target)


def generate_exp(nodes: np.ndarray, truth: dict[str, np.ndarray], rng: np.random.Generator) -> list[dict]:
    records = []
    for spec in EXP_SPECS:
        source = FITPACK_IDIS / f"{spec.idx}.xlsx"
        df = pd.read_excel(source)
        rows = choose_rows(df, MAX_POINTS_PER_EXP_DATASET)
        if rows.size == 0:
            records.append({"idx": spec.idx, "label": spec.label, "status": "skipped_by_cuts"})
            continue
        selected = df.iloc[rows]
        x = np.asarray(selected["X"], dtype=float)
        q2 = np.asarray(selected["Q2"], dtype=float)
        real_value = np.asarray(selected["value"], dtype=float)
        real_stat = np.asarray(selected["stat_u"], dtype=float)
        rel_stat = np.abs(real_stat / real_value)
        mean = exp_mean(spec, nodes, truth, x)
        sigma = rel_stat * np.abs(mean) + EXP_ABS_FLOOR
        cov = np.diag(sigma * sigma)
        obs = rng.normal(mean, sigma)
        filename = f"{spec.label}_{spec.idx}.dat"
        path = cfg.EXP_DIR / filename
        write_table(path, [x, q2], obs, cov)
        meta = {
            "idx": spec.idx,
            "label": spec.label,
            "kind": spec.kind,
            "obs": spec.obs,
            "target": spec.target,
            "source": str(source),
            "row_indices": rows.astype(int).tolist(),
            "x": x.tolist(),
            "Q2": q2.tolist(),
            "real_value": real_value.tolist(),
            "real_stat": real_stat.tolist(),
            "rel_stat": rel_stat.tolist(),
            "truth_mean": mean.tolist(),
            "observed": obs.tolist(),
            "sigma": sigma.tolist(),
            "theory_level": "input_scale_fixed_LO_fixture",
        }
        if "Y" in selected:
            meta["y"] = np.asarray(selected["Y"], dtype=float).tolist()
        if "*Y" in selected:
            meta["y"] = np.asarray(selected["*Y"], dtype=float).tolist()
        if "RS" in selected:
            meta["sqrt_s"] = np.asarray(selected["RS"], dtype=float).tolist()
        write_json(path.with_suffix(".json"), meta)
        records.append({**meta, "file": str(path.relative_to(cfg.DATA_ROOT))})
    return records


def diagonal_cov(mean: np.ndarray, rel: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    sigma = rel * np.abs(mean) + floor
    return np.diag(sigma * sigma), sigma


def generate_itd(
    fields: dict[str, Field],
    truth: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> list[dict]:
    records = []
    z_latt = HBARC_GEV_FM / (cfg.Z_INV_GEV * cfg.LATTICE_A_FM)
    p = np.arange(1, 5, dtype=float)
    z = np.full_like(p, z_latt)
    rel = np.linspace(0.01, 0.10, p.size)
    for field_name in (*cfg.EVEN_FIELDS, *cfg.ODD_FIELDS):
        # q(-x)=-qbar(x): real/cosine measures C-odd q-qbar, while
        # imaginary/sine measures C-even q+qbar.
        component = "imag" if field_name in cfg.EVEN_FIELDS else "real"
        kernel = kernels.PseudoITDReal() if component == "real" else kernels.PseudoITDImag()
        field = fields[field_name]
        nu = (2.0 * np.pi / cfg.LATTICE_L) * p * z
        mean = np.asarray(kernel.matrix(nu, field.basis), dtype=float) @ truth[field_name]
        cov, sigma = diagonal_cov(mean, rel, LATTICE_ABS_FLOOR)
        obs = rng.normal(mean, sigma)
        filename = f"itd_{field_name}_{component}.dat"
        path = cfg.LATTICE_DIR / filename
        write_table(path, [z, p], obs, cov)
        pz_gev = (2.0 * np.pi * p / (cfg.LATTICE_L * cfg.LATTICE_A_FM)) * HBARC_GEV_FM
        meta = {
            "kind": "pseudoitd",
            "field": field_name,
            "component": component,
            "file": str(path.relative_to(cfg.DATA_ROOT)),
            "L": cfg.LATTICE_L,
            "a_fm": cfg.LATTICE_A_FM,
            "z_lattice": z.tolist(),
            "z_physical_fm": (z * cfg.LATTICE_A_FM).tolist(),
            "p_lattice": p.tolist(),
            "pz_GeV": pz_gev.tolist(),
            "rel_error": rel.tolist(),
            "truth_mean": mean.tolist(),
            "observed": obs.tolist(),
            "sigma": sigma.tolist(),
        }
        write_json(path.with_suffix(".json"), meta)
        records.append(meta)
    return records


def generate_moments(
    fields: dict[str, Field],
    truth: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> list[dict]:
    records = []
    kernel = kernels.Mellin()
    for even, odd in FIELD_PAIRS:
        even_field = fields[even]
        odd_field = fields[odd]
        m_x = (np.asarray(kernel.matrix(np.array([2.0]), odd_field.basis)) @ truth[odd])[0]
        m_x2 = (np.asarray(kernel.matrix(np.array([3.0]), even_field.basis)) @ truth[even])[0]
        n = np.array([2.0, 3.0])
        mean = np.array([m_x, m_x2])
        rel = np.array([0.01, 0.10])
        cov, sigma = diagonal_cov(mean, rel, LATTICE_ABS_FLOOR)
        obs = rng.normal(mean, sigma)
        filename = f"moments_{even}_{odd}.dat"
        path = cfg.LATTICE_DIR / filename
        write_table(path, [n], obs, cov)
        meta = {
            "kind": "mellin",
            "cp_even_field": even,
            "cp_odd_field": odd,
            "file": str(path.relative_to(cfg.DATA_ROOT)),
            "Q2": cfg.Q0_2,
            "n": n.tolist(),
            "moment_labels": ["<x> cp-odd", "<x^2> cp-even"],
            "rel_error": rel.tolist(),
            "truth_mean": mean.tolist(),
            "observed": obs.tolist(),
            "sigma": sigma.tolist(),
        }
        write_json(path.with_suffix(".json"), meta)
        records.append(meta)

    field = fields["g"]
    # The gluon is CP-even, so only even exponent moments are physical.  Stored
    # Mellin order N=3 corresponds to the even exponent <x^2>.
    n = np.array([3.0])
    mean = np.asarray(kernel.matrix(n, field.basis), dtype=float) @ truth["g"]
    rel = np.array([0.10])
    cov, sigma = diagonal_cov(mean, rel, LATTICE_ABS_FLOOR)
    obs = rng.normal(mean, sigma)
    path = cfg.LATTICE_DIR / "moments_g.dat"
    write_table(path, [n], obs, cov)
    meta = {
        "kind": "mellin",
        "field": "g",
        "file": str(path.relative_to(cfg.DATA_ROOT)),
        "Q2": cfg.Q0_2,
        "n": n.tolist(),
        "moment_labels": ["<x^2> cp-even gluon"],
        "rel_error": rel.tolist(),
        "truth_mean": mean.tolist(),
        "observed": obs.tolist(),
        "sigma": sigma.tolist(),
    }
    write_json(path.with_suffix(".json"), meta)
    records.append(meta)
    return records


def main() -> None:
    rng = np.random.default_rng(SEED)
    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cfg.EXP_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LATTICE_DIR.mkdir(parents=True, exist_ok=True)
    fields = make_fields()
    nodes = np.asarray(next(iter(fields.values())).nodes, dtype=float)
    truth = truth_from_lhapdf(nodes)

    truth_payload = {
        "name": "standard",
        "seed": SEED,
        "pdf_guidance": {
            "lhapdf_prefix": str(LHAPDF_PREFIX),
            "set": PDF_SET,
            "member": PDF_MEMBER,
            "Q": cfg.Q0,
            "used_only_for_generation": True,
        },
        "grid": {
            "n_points": cfg.GRID_N,
            "spacing": cfg.GRID_SPACING,
            "x_min": cfg.X_MIN,
            "element_type": cfg.ELEMENT_TYPE,
        },
        "x_nodes": nodes.tolist(),
        "fields": {name: values.tolist() for name, values in truth.items()},
        "basis": cfg.FIELD_TO_BASIS,
    }
    write_json(cfg.TRUTH_PATH, truth_payload)

    exp_records = generate_exp(nodes, truth, rng)
    lattice_records = [*generate_itd(fields, truth, rng), *generate_moments(fields, truth, rng)]
    manifest = {
        "name": "standard",
        "seed": SEED,
        "truth": str(cfg.TRUTH_PATH.relative_to(cfg.DATA_ROOT)),
        "q0": cfg.Q0,
        "q0_2": cfg.Q0_2,
        "nf": cfg.NF,
        "theory_level": "input_scale_fixed_LO_fixture",
        "notes": [
            "LHAPDF was used only to create realistic input-scale truth curves.",
            "Experimental kinematics and relative statistical errors come from fitpack IDIS tables.",
            "CC tables are stored as F3 proxy data at the real CC kinematics.",
        ],
        "exp": exp_records,
        "lattice": lattice_records,
    }
    write_json(cfg.MANIFEST_PATH, manifest)
    print(f"Wrote {cfg.DATA_ROOT}")
    print(f"  exp files: {sum(1 for r in exp_records if 'file' in r)}")
    print(f"  lattice files: {len(lattice_records)}")


if __name__ == "__main__":
    main()
