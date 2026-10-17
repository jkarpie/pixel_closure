"""Contracts for the explicit-q evidence scan's physical-field reparametrization.

The scan itself is pinned in ``pixel``'s ``tests/test_evidence_scan.py``.  What
is new here, and the only thing that can be silently wrong, is the affine map
from readable coordinates -- a fitted normalization, or one distribution's value
at one ``x`` -- onto ``QTarget``'s ``[u, zeta]`` vector.  Every claim below is
checked against the closed form of the GP lift ``q = g + Q L0 zeta`` computed in
plain numpy in this file, on a real :class:`~pixel.infer.PriorSubspace` built
from a hand-written covariance.  No model is compiled and no kernel is built.
"""

import numpy as np
import pytest

from pixel.infer import PriorSubspace
from pixel.plotting import MAX_SCAN_PARAMETERS

from closure_small_q_integration.plot_evidence_scans import (
    DEFAULT_SIGMA_HALF_WIDTH,
    DEFAULT_X_VALUES,
    FieldSliceTarget,
    build_slice,
    resolve_windows,
)
from types import SimpleNamespace


#: Two fields, four grid nodes each, on a deliberately *smooth* kernel: the
#: whole point of the ``"prior"`` direction is that a single-node spike is stiff,
#: and a near-diagonal covariance would hide that.
_NODES = np.asarray([0.01, 0.05, 0.2, 0.6])
_X_WANTED = (0.01, 0.2)


def _covariance(scale):
    distance = np.log(_NODES)[:, None] - np.log(_NODES)[None, :]
    return scale * np.exp(-0.5 * (distance / 1.5) ** 2) + 1.0e-10 * np.eye(4)


_COVARIANCES = (_covariance(1.0), _covariance(4.0))
_MEAN = np.asarray([0.3, 0.2, 0.1, 0.02, 1.0, 0.8, 0.5, 0.1])
_THETA_CENTER = np.asarray([0.97])


def _fixture():
    """A model/coords/target triple exposing exactly what ``build_slice`` reads."""
    subspace = PriorSubspace.from_covariances(_COVARIANCES, names=("t3", "g"))
    n_theta = _THETA_CENTER.size
    dimension = n_theta + subspace.rank

    model = SimpleNamespace(
        fields=(
            SimpleNamespace(name="t3", nodes=_NODES),
            SimpleNamespace(name="g", nodes=_NODES),
        ),
        q_layout=lambda: (("t3", 0, 4), ("g", 4, 4)),
        free_partition=lambda: SimpleNamespace(
            n_theta=n_theta, n_hubbard=0,
            theta_labels=("table normalization / amp",),
        ),
    )

    def physical(x):
        x = np.asarray(x, dtype=float)
        theta = _THETA_CENTER + x[:n_theta]
        return theta, np.asarray(
            subspace.q_from_zeta(_MEAN, x[n_theta:]), dtype=float
        )

    coords = SimpleNamespace(
        model=model, dimension=dimension, n_theta=n_theta,
        prior_subspace=subspace, x0=np.zeros(dimension),
        physical=physical,
    )
    # A quadratic standing in for the action: enough for _raw_value to be a real
    # differentiable function of the reduced coordinates without a model.  It has
    # to be written in `jnp`, because `FieldSliceTarget.hessian` traces it.
    import jax.numpy as jnp

    target = SimpleNamespace(
        _raw_value=lambda x: jnp.sum(jnp.asarray(x, dtype=float) ** 2)
    )
    return model, coords, target, subspace


def _dense_covariance(subspace, index):
    """``Q L0 L0' Q'`` for one block, from the subspace's own matrices."""
    block = subspace.blocks[index]
    basis = np.asarray(block.Q, dtype=float)
    scale = np.asarray(block.L0, dtype=float)
    return basis @ scale @ scale.T @ basis.T


# -- 1. the coordinates are what their labels say ----------------------------


def test_every_axis_is_the_quantity_its_label_names():
    """Labels, order, and centre values all come from the physical state.

    The normalizations come first in the order the model partitions them, then
    one coordinate per ``(field, x)`` pair in grid order, each labelled with the
    *snapped* node rather than the requested ``x`` -- so a label can never claim
    an ``x`` the scan did not use.
    """
    model, coords, target, _subspace = _fixture()
    slice_target, diagnostics = build_slice(
        model, coords, target, coords.x0, x_values=_X_WANTED, verbose=False
    )

    assert list(slice_target.free_labels()) == [
        "table normalization / amp",
        "t3(x=0.01)", "t3(x=0.2)", "g(x=0.01)", "g(x=0.2)",
    ]
    assert diagnostics["n_normalizations"] == 1
    assert diagnostics["n_parameters"] == 5

    # The centre is the physical state at x_center, not the coordinate origin.
    centre = slice_target.free_vector()
    assert centre[0] == pytest.approx(_THETA_CENTER[0])
    np.testing.assert_allclose(
        centre[1:], [_MEAN[0], _MEAN[2], _MEAN[4], _MEAN[6]], rtol=1e-12
    )
    assert [row["node_index"] for row in diagnostics["field_coordinates"]] == [
        0, 2, 0, 2
    ]
    assert [row["requested_x"] for row in diagnostics["field_coordinates"]] == [
        0.01, 0.2, 0.01, 0.2
    ]


def test_the_recorded_prior_width_is_the_kernels_own_diagonal():
    """``prior_variance`` is ``K[k, k]`` of that field, not a stand-in.

    This is the number that sets the natural window of a field axis, so it has
    to be the covariance the fit actually uses -- and the two fields differ by
    the factor of four built into the fixture, which is what rejects reading the
    wrong block.
    """
    model, coords, target, subspace = _fixture()
    _slice_target, diagnostics = build_slice(
        model, coords, target, coords.x0, x_values=_X_WANTED, verbose=False
    )
    rows = diagnostics["field_coordinates"]
    for row in rows:
        block = 0 if row["field"] == "t3" else 1
        dense = _dense_covariance(subspace, block)
        assert row["prior_variance"] == pytest.approx(
            dense[row["node_index"], row["node_index"]], rel=1e-10
        )
    assert rows[2]["prior_variance"] == pytest.approx(
        4.0 * rows[0]["prior_variance"], rel=1e-8
    )


# -- 2. a unit step does what it claims --------------------------------------


def test_a_unit_step_moves_its_own_field_value_by_exactly_one():
    """The axis is the field value, to float precision, in both directions.

    Both halves matter. Without this the label is a guess; and because the
    ``"prior"`` direction is ``K[:, k] / K[k, k]``, a dropped or doubled
    ``K[k, k]`` would rescale the axis without changing anything else visible.
    """
    model, coords, target, _subspace = _fixture()
    for direction in ("prior", "node"):
        _slice_target, diagnostics = build_slice(
            model, coords, target, coords.x0, x_values=_X_WANTED,
            direction=direction, verbose=False,
        )
        np.testing.assert_allclose(
            diagnostics["self_response"], np.ones(4), atol=1e-12
        )
        assert diagnostics["max_self_response_error"] < 1.0e-12


def test_the_prior_direction_leaks_into_the_other_x_by_exactly_the_kernel_ratio():
    """Cross-response is ``K[k', k] / K[k, k]``, measured, not assumed small.

    This is the price of a readable axis and the reason the manifest stores the
    whole matrix. On this smooth kernel the leak between ``x = 0.01`` and
    ``x = 0.2`` is far from negligible, which is exactly why asserting it is
    worth more than asserting it is tiny.
    """
    model, coords, target, subspace = _fixture()
    _slice_target, diagnostics = build_slice(
        model, coords, target, coords.x0, x_values=_X_WANTED, verbose=False
    )
    response = np.asarray(diagnostics["cross_response"], dtype=float)

    expected = np.zeros((4, 4))
    for column, (block, node) in enumerate(((0, 0), (0, 2), (1, 0), (1, 2))):
        dense = _dense_covariance(subspace, block)
        for row, (other_block, other_node) in enumerate(
            ((0, 0), (0, 2), (1, 0), (1, 2))
        ):
            if other_block != block:
                continue
            expected[row, column] = dense[other_node, node] / dense[node, node]
    np.testing.assert_allclose(response, expected, atol=1e-10)

    # The fixture has to be able to see a leak at all, or the assertion above is
    # vacuous: a near-diagonal kernel would make `expected` the identity.
    assert diagnostics["max_abs_cross_response"] > 0.05
    # Fields never mix: the prior is block diagonal per field.
    np.testing.assert_allclose(response[:2, 2:], 0.0, atol=1e-14)
    np.testing.assert_allclose(response[2:, :2], 0.0, atol=1e-14)


def test_the_node_direction_moves_one_grid_point_and_nothing_else():
    """``--direction node`` is the exact conditional slice in physical ``q``.

    It is the honest alternative to the default, and the one whose window
    collapses: its prior cost is ``[K^-1]_kk``. The assertion here is only that
    it is what it says -- zero cross-response -- since the stiffness is a
    property of the kernel, not of this code.
    """
    model, coords, target, _subspace = _fixture()
    _slice_target, diagnostics = build_slice(
        model, coords, target, coords.x0, x_values=_X_WANTED,
        direction="node", verbose=False,
    )
    response = np.asarray(diagnostics["cross_response"], dtype=float)
    np.testing.assert_allclose(response, np.eye(4), atol=1e-10)
    assert diagnostics["max_abs_cross_response"] < 1.0e-10


def test_an_unknown_direction_is_refused_rather_than_defaulted():
    """A typo must not silently fall back to one of the two real choices."""
    model, coords, target, _subspace = _fixture()
    with pytest.raises(ValueError, match="direction must be"):
        build_slice(
            model, coords, target, coords.x0, direction="smooth", verbose=False
        )


# -- 3. the adapter the scanner consumes -------------------------------------


def test_the_slice_target_exposes_the_scanners_protocol_and_the_same_scalar():
    """``FieldSliceTarget`` reports the underlying action, unshifted.

    ``support_bounds()`` returning ``None`` is load bearing rather than lazy:
    it is what makes ``ParameterCoordinates`` the identity, so a ``"physical"``
    scan scans the reduced coordinates themselves. At the centre the adapter
    must return exactly the target's own value there.
    """
    model, coords, target, _subspace = _fixture()
    slice_target, _diagnostics = build_slice(
        model, coords, target, coords.x0, x_values=_X_WANTED, verbose=False
    )
    assert isinstance(slice_target, FieldSliceTarget)
    assert slice_target.support_bounds() is None
    assert slice_target.bounds() is None
    # Not `values`: `_batched_objective` would prefer it, and the adapter has no
    # batched path of its own.
    assert not hasattr(slice_target, "values")

    centre = slice_target.free_vector()
    assert float(slice_target._raw_value(centre)) == pytest.approx(
        float(target._raw_value(coords.x0)), rel=1e-12, abs=1e-12
    )
    hessian = np.asarray(slice_target.hessian(centre), dtype=float)
    assert hessian.shape == (centre.size, centre.size)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-10)


# -- 4. the window the slice is drawn in -------------------------------------


class _CorrelatedTarget:
    """Two coordinates whose marginal width is ten times their conditional one.

    ``S = x' A x`` with ``A = [[1, c], [c, 1]] * k`` has
    ``H = 2A``, conditional ``sqrt(2 / H_ii) = 1 / sqrt(k)`` and marginal
    ``sqrt(2 [H^-1]_ii) = 1 / sqrt(k (1 - c^2))``, so the ratio is
    ``1 / sqrt(1 - c^2)`` and is set by the correlation alone.
    """

    def __init__(self, *, k=4.0, c=np.sqrt(0.99)):
        self.k, self.c = float(k), float(c)

    def free_labels(self):
        return ("a", "b")

    def free_vector(self):
        return np.zeros(2)

    def support_bounds(self):
        return None

    def _raw_value(self, z):
        import jax.numpy as jnp

        z = jnp.asarray(z, dtype=float)
        return self.k * (z[0] ** 2 + 2.0 * self.c * z[0] * z[1] + z[1] ** 2)

    def hessian(self, z):
        return 2.0 * self.k * np.asarray(
            [[1.0, self.c], [self.c, 1.0]], dtype=float
        )


def test_the_conditional_window_is_the_slice_width_not_the_marginal_one():
    """Both widths are reported, and the conditional one sizes the window.

    On a correlation of `sqrt(0.99)` the marginal width is ten times the
    conditional one, so a window sized the scanner's way is ten times too wide
    for the slice it draws. The closed forms are in `_CorrelatedTarget`.
    """
    target = _CorrelatedTarget()
    centre = np.zeros(2)
    bounds, info = resolve_windows(target, centre, half_width=5.0)

    conditional = 1.0 / np.sqrt(target.k)
    marginal = conditional / np.sqrt(1.0 - target.c ** 2)
    np.testing.assert_allclose(info["conditional_sigma"], [conditional] * 2,
                               rtol=1e-12)
    np.testing.assert_allclose(info["marginal_sigma"], [marginal] * 2, rtol=1e-8)
    assert marginal / conditional == pytest.approx(10.0, rel=1e-6)

    np.testing.assert_allclose(
        bounds, [(-5.0 * conditional, 5.0 * conditional)] * 2, rtol=1e-12
    )
    assert info["kind"] == "conditional"
    assert info["n_negative_eigenvalues"] == 0


def test_the_endpoint_of_a_default_window_lands_on_the_colour_ceiling():
    """Five conditional sigma is `Delta S = 25`: the outermost contour exactly.

    This is the whole reason the default is five rather than the scanner's six.
    Checked against the objective itself, not against the algebra that chose it.
    """
    target = _CorrelatedTarget()
    centre = np.zeros(2)
    bounds, _info = resolve_windows(
        target, centre, half_width=DEFAULT_SIGMA_HALF_WIDTH
    )
    assert DEFAULT_SIGMA_HALF_WIDTH == 5.0
    edge = np.asarray([bounds[0][1], 0.0])
    delta = float(target._raw_value(edge)) - float(target._raw_value(centre))
    assert delta == pytest.approx(25.0, rel=1e-10)


def test_the_marginal_window_defers_to_the_scanner_and_still_reports_both():
    """`kind="marginal"` returns no bounds, so `scan_evidence` applies its rule."""
    bounds, info = resolve_windows(
        _CorrelatedTarget(), np.zeros(2), half_width=6.0, kind="marginal"
    )
    assert bounds is None
    assert info["kind"] == "marginal"
    assert len(info["conditional_sigma"]) == 2
    assert len(info["marginal_sigma"]) == 2


def test_a_flat_or_inverted_axis_refuses_a_conditional_window():
    """A non-positive Hessian diagonal has no conditional width to use.

    It happens for real: sliced at the prior-mean origin rather than the MAP,
    the 34-coordinate block of the Q=2 `exp` case has negative curvature. The
    message has to say what to do, because the fix is a different flag.
    """
    class _Flat(_CorrelatedTarget):
        def hessian(self, z):
            return np.asarray([[2.0, 0.0], [0.0, -1.0]], dtype=float)

    with pytest.raises(ValueError, match="conditional windows need"):
        resolve_windows(_Flat(), np.zeros(2), half_width=5.0)
    # The same centre is still scannable the scanner's own way.
    bounds, _info = resolve_windows(
        _Flat(), np.zeros(2), half_width=5.0, kind="marginal"
    )
    assert bounds is None


def test_an_unknown_window_kind_is_refused():
    with pytest.raises(ValueError, match="window kind must be"):
        resolve_windows(
            _CorrelatedTarget(), np.zeros(2), half_width=5.0, kind="laplace"
        )


# -- 5. the ceiling this module depends on -----------------------------------


def test_the_full_field_scan_fits_inside_the_raised_scan_ceiling():
    """Nine fields at three ``x`` plus seven normalizations is 34, and 34 <= 100.

    The dependency runs the other way from the usual one: this module exists in
    its current shape *because* ``MAX_SCAN_PARAMETERS`` was raised from 8 on
    2026-10-16. If it is ever lowered back, the Q=2 ``exp`` figure stops being
    constructible, and that should fail here rather than at the end of a
    forty-minute scan.
    """
    assert len(DEFAULT_X_VALUES) == 3
    assert 7 + 9 * len(DEFAULT_X_VALUES) == 34
    assert MAX_SCAN_PARAMETERS >= 34
