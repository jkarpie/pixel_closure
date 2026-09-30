"""Closure Drell--Yan builders declare momentum densities, and the fold is the LO cross section.

Every closure field is a momentum density ``x f(x)``.  PIXEL's Drell--Yan kernel used
to consume its fields as number densities, so every closure DY row -- generated and
fitted alike -- was ``x1 x2`` times the physical cross section: self-consistent, and
therefore invisible to a closure test.  ``pixel.data.DrellYan(momentum_density=True)``
fixes the kernel side; these tests pin the closure side:

* every ``DrellYan(...)`` call in every suite names ``momentum_density=True`` (an AST
  scan, so a new builder cannot silently fall back to a default);
* a ``_small`` suite's real ``build_drell_yan`` folded through its own
  ``generate.dy_central`` reproduces the textbook leading-order cross section
  ``4 pi alpha^2/(9 Q^2 S) sum_q e_q^2 [f_q(x1) fbar_q(x2) + fbar_q(x1) f_q(x2)]``
  with ``f = (x f)/x`` built from the suite's own parton maps.
"""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SUITES = (
    "closure_JAM_truth",
    "closure_NNPDF_truth",
    "closure_JAM_truth_JAM_data",
    "closure_JAM_truth_NNPDF_data",
    "closure_NNPDF_truth_JAM_data",
    "closure_NNPDF_truth_NNPDF_data",
    "closure_JAM_truth_small",
    "closure_NNPDF_truth_small",
)


def _drell_yan_calls(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "DrellYan":
            yield node


@pytest.mark.parametrize("suite", SUITES)
def test_every_closure_drell_yan_builder_declares_momentum_densities(suite):
    """Each ``DrellYan(...)`` call in the suite passes ``momentum_density=True`` literally."""
    paths = sorted((REPO / suite).glob("*.py"))
    calls = [(path, node) for path in paths for node in _drell_yan_calls(path)]
    assert calls, f"{suite}: no DrellYan builder found"
    for path, node in calls:
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        value = keywords.get("momentum_density")
        assert isinstance(value, ast.Constant) and value.value is True, (
            f"{path.relative_to(REPO)}:{node.lineno} builds DrellYan without "
            "momentum_density=True"
        )


def _parton_momentum_density(maps, parton, curves, basis, x):
    """``x f_parton(x)`` from the suite's own parton -> field map and the field interpolants."""
    row = np.asarray(basis.evaluate([x]), dtype=float)[0]
    return sum(coeff * float(row @ curves[field]) for field, coeff in maps[parton].items())


@pytest.mark.parametrize("suite", ["closure_JAM_truth_small", "closure_NNPDF_truth_small"])
def test_small_suite_fold_is_the_leading_order_cross_section_of_number_densities(suite, monkeypatch):
    """``build_drell_yan`` + ``dy_central`` on smooth ``x f`` curves equals the closed-form LO rows.

    The oracle (``A1``) is the leading-order Drell--Yan formula in number densities,
    evaluated here from the same field interpolants the fit uses and the suite's own
    ``dy_field_maps``; it shares no code with the kernel.  With the builders forced to
    number densities the ratio is exactly ``tau = Q^2/S`` on each row, not ``1``.
    """
    monkeypatch.setenv("PIXEL_KERNEL_WORKERS", "1")
    cfg = importlib.import_module(f"{suite}.config")
    datasets = importlib.import_module(f"{suite}.datasets")
    generate = importlib.import_module(f"{suite}.generate")
    # The closed form below is leading order and unevolved; if the suite ever changes
    # either, this test must be rewritten rather than silently skipped.
    assert str(cfg.DY_ORDER).upper() == "LO" and not getattr(cfg, "DY_EVOLVE", False)

    fields = cfg.make_fields()
    basis = next(iter(fields.values())).basis
    nodes = np.asarray(next(iter(fields.values())).nodes, dtype=float).reshape(-1)
    shapes = {
        "t3": (0.6, 3.0, 0.2), "t8": (0.4, 4.0, 0.5), "sigma": (0.1, 5.0, 1.5),
        "t15": (0.2, 6.0, 0.8), "g": (0.05, 5.0, 2.0), "v3": (0.7, 3.0, 0.3),
        "v8": (0.6, 3.5, 0.9), "v": (0.6, 3.5, 1.0), "v15": (0.6, 3.5, 1.0),
    }
    curves = {name: n * nodes**a * (1.0 - nodes) ** b for name, (a, b, n) in shapes.items()}

    record = {
        "kind": "drell_yan", "reaction": "pp", "label": "momentum_density_check",
        "Q2": [25.0, 60.0], "S": [1501.0, 1501.0], "Y": [0.3, -0.2],
    }
    dataset = datasets.build_drell_yan(record, None, fields, None)
    got = np.asarray(generate.dy_central(dataset, curves, fields, "pp"), dtype=float)

    maps_A, maps_B = cfg.dy_field_maps("pp")
    charges = {"u": 4.0 / 9.0, "c": 4.0 / 9.0, "d": 1.0 / 9.0, "s": 1.0 / 9.0}
    expected = []
    for Q2, S, Y in zip(record["Q2"], record["S"], record["Y"]):
        tau = Q2 / S
        x1, x2 = math.sqrt(tau) * math.exp(Y), math.sqrt(tau) * math.exp(-Y)

        def f(maps, parton, x):
            return _parton_momentum_density(maps, parton, curves, basis, x) / x

        lum = sum(
            e2 * (f(maps_A, q, x1) * f(maps_B, q + "b", x2) + f(maps_A, q + "b", x1) * f(maps_B, q, x2))
            for q, e2 in charges.items()
            if q in maps_A and q + "b" in maps_B
        )
        expected.append(4.0 * math.pi * cfg.DY_ALPHA_EM**2 / (9.0 * Q2 * S) * lum)
    ratio = got / np.asarray(expected)
    assert np.all(np.abs(ratio - 1.0) < 1e-9), ratio
