"""Evidence scans of the explicit-``q`` small closure, in physical field coordinates.

This is the ``q``-integration counterpart of
``closure_JAM_truth_small/plot_evidence_scans.py``.  Both scan the *same scalar
the sampler integrates against*, holding every unscanned coordinate at the
centre, so valleys, ridges and second minima are visible directly.  What differs
is the target and therefore what a scan axis can mean.

**The H-S route has no field coordinates to scan; this one has 1,314 of them.**
On the Hubbard--Stratonovich route the free vector is nineteen scalars -- twelve
auxiliaries and seven normalizations -- and the GP fields are integrated
analytically, so "the PDF at ``x``" is not a coordinate at all.  Here
``QTarget`` samples ``x = [u, zeta]``: seven normalizations followed by 1,314
prior-whitened retained field modes (nine fields at full rank 146 each, because
``Q_INTEGRATION_PRIOR_RCOND = 0``).  The fields are *in* the chain, which is
exactly why this variant exists -- and exactly why its scan should be read in
field values rather than in mode amplitudes.

**A retained mode is not a readable axis, so the scan is reparametrized.**
``v8 / retained[66]`` -- the slowest-mixing coordinate of the production run --
is an eigenvector of the ``v8`` prior covariance.  Nobody can say what a fit
looks like at ``zeta_66 = 2``.  This module therefore scans physical field
values: one coordinate per ``(field, x)`` pair, labelled ``t3(x=0.0103)``, whose
axis is the value of that distribution at that ``x`` in the fit's own units.

**Which direction "the PDF at x" moves along is a physics choice, and the
obvious one is useless.**  Moving the single grid node ``k`` and nothing else is
the exact conditional slice in physical ``q``, but it is a delta spike: its
prior cost is ``[K^-1]_kk``, which on a smooth 146-node kernel is astronomical,
so the Laplace window collapses to ~``1e-8`` of the field and every panel draws
the same featureless parabola.  The default direction here is instead the GP's
own conditional response, ``dq = K[:, k] / K[k, k]``, normalized so the value at
``x_k`` moves by exactly one per unit of the axis.  That is "raise the PDF here"
as the prior allows it -- a smooth bump -- and its prior cost is ``1 / K[k, k]``,
so the natural window is the prior width of the field at that ``x``.  Pass
``--direction node`` for the stiff alternative; the manifest records which was
used either way.

**The cost of that choice: the field axes are not mutually orthogonal.**  A bump
centred at ``x = 0.01`` also moves ``x = 0.1`` a little, by exactly
``K[k', k] / K[k, k]``.  Scanning two of them is therefore not a pure
conditional slice in the pair, and the manifest stores the full cross-response
matrix at the scanned nodes so the leakage is auditable rather than assumed
small.  ``--direction node`` has no leakage and no legibility.

**The centre is a real mode here, unlike the H-S case.**  ``QTarget`` is a real
positive density, so a MAP is a legitimate point to slice at -- ``pixel.map``'s
warning about bilinear H-S MAPs does not apply.  It is not free: the production
chain starts at the prior-mean origin (``S = 1.85e10``) and the MAP is found by
L-BFGS-B followed by Hessian-vector Newton-CG, measured at about eight and a
half minutes on this suite for ``|grad S| ~ 1e-6``.  It is cached.

**A conditional slice needs a conditional window, and the scanner's default is
not one.**  ``scan_evidence`` sizes an unbounded axis from the *marginal*
Laplace width ``sqrt(2 [H^-1]_ii)``.  Measured on a 16-coordinate pilot of this
scan, that left ``sigma(x=0.1)`` and ``g(x=0.1)`` blank -- every node past the
``Delta S = 25`` colour ceiling -- because they sit on correlations of ``-0.47``
and ``-0.99`` with their neighbours, which makes their marginal width sixteen
times their conditional one.  The default here is therefore ``--window
conditional``: ``sqrt(2 / H_ii)``, the width of the slice being drawn, at five
of which the axis endpoint lands exactly on ``Delta S = 25``.  Pass ``--window
marginal`` for the scanner's own rule.

**Always estimate before launching.**  ``--estimate`` builds everything, finds
the centre, prints the Laplace windows and times one real panel, then stops
without scanning.  Measured on the Q=2 ``exp`` case: 34 coordinates, 561 pairs,
380,120 nodes at ``n_points=26``, one real panel 3.06 s warm, **4.53 ms per
node, 29 min** for the whole scan, plus 9.2 min for the MAP the first time and
~35 s for the figure.  ``n_points=51`` is 1.46 million nodes, about 2.2 hours.

Run standalone::

    python -m closure_small_q_integration.plot_evidence_scans --estimate
    python -m closure_small_q_integration.plot_evidence_scans --truth JAM --Q 2 \\
        --mode exp --x 0.01 0.1 0.5 --n-points 26
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

from small_closure_variant import TRUTH_SUITES, q_integration_activated


#: Nodes per axis.  The coarse/fast grid, as on the H-S side: the panel count
#: here is 561 rather than 28, so 26x26 is what keeps a 34-parameter figure
#: inside an hour.  See ``--estimate`` before raising it.
DEFAULT_N_POINTS = 26

#: Default ``x`` values per field.  Three, as asked for, spread across the
#: regimes the small suite actually distinguishes: ``0.01`` sits at the low edge
#: of the fixed-target DIS/DY coverage where ``t8`` is known to sit flat at its
#: prior amplitude, ``0.1`` is inside the data, and ``0.5`` is the valence
#: region where the fields dip negative to buy back a pinned integral.  Snapped
#: to the nearest grid node in ``log x``.
DEFAULT_X_VALUES = (0.01, 0.1, 0.5)

#: Half-width of a scan axis, in whichever sigma ``--window`` selects.  Five on
#: the default conditional rule, because along a held axis
#: ``Delta S = H_ii d^2 / 2``, so five conditional sigma puts the axis endpoint
#: at ``Delta S = 25`` -- exactly the outermost drawn contour and exactly
#: ``plot_evidence_scan``'s colour ceiling.  The corners of a panel reach
#: ``50`` and saturate, which is the intended reading: a corner two sigma out in
#: both coordinates is not where the posterior is.  ``--estimate`` prints every
#: window, and both sigmas, before anything is scanned.
DEFAULT_SIGMA_HALF_WIDTH = 5.0

#: Total figure size budget in inches per side.  ``plot_evidence_scan`` lays out
#: 3.1 x 3.0 inches per panel, which is 106 x 102 inches at 34 parameters, and
#: ``pixel.plotting.style.save_figure`` rasters at 180 dpi -- a 1.4 GB RGBA
#: buffer.  The figure is drawn and then shrunk to this before saving.
DEFAULT_FIGURE_INCHES = 60.0

#: Raster budget for the shrunk figure, and the reason this module saves its own
#: figures instead of calling ``save_figure``.  At 34 parameters the layout is
#: 106 x 102 inches, which ``save_figure``'s 180 dpi turns into 19,000 x 18,000
#: px; measured, that figure was killed by the OS on a 36 GB machine.  At this
#: budget the 34-parameter PNG costs ~74 bytes of peak RSS per output pixel, so
#: 40 megapixels is about 3 GB and still ~185 px per panel for a 26 x 26 mesh.
MAX_FIGURE_PIXELS = 4.0e7

#: L-BFGS-B iterations before the Newton-CG refinement.  Measured on this suite:
#: 200 reaches ``S = 2982``, 1,000 reaches 913, 4,000 reaches 875 with
#: ``|grad| = 1356`` and still falling -- the target's condition number is ~5e10,
#: so the quasi-Newton phase is a starting point, not a solution.
DEFAULT_LBFGS_MAXITER = 4000

#: Newton-CG iterations. Measured: 53 iterations, 367 s, ``|grad| = 1.05e-06``.
DEFAULT_NEWTON_MAXITER = 200


# -- target assembly ---------------------------------------------------------


def build_target(cfg, fit, q_key, mode, *, use_kernel_cache=True, verbose=True):
    """``(model, coords, target, t0)`` on the converged-t0 model.

    The DIS multiplicative uncertainties are referenced to theory rather than
    data, and the fitted DIS normalizations are exactly what that reference
    moves -- so the scan is taken on the same converged-t0 model the production
    ``q`` fit uses, not a cheaper stand-in.  ``run_fit`` iterated it eight times
    on the Q=2 ``exp`` case.

    Must be called inside :func:`small_closure_variant.q_integration_activated`,
    whose overlay owns ``Q_INTEGRATION_PRIOR_RCOND`` and the output locations.
    """
    from pixel.infer import DirectCoordinates, QTarget

    started = time.perf_counter()
    analysis, _fields = fit.build_analysis(
        q_key, mode, use_kernel_cache=use_kernel_cache
    )
    model = analysis.compile()
    t0, iterations = fit._iterate_t0(
        q_key, mode, model, use_kernel_cache=use_kernel_cache
    )
    if t0:
        analysis, _fields = fit.build_analysis(
            q_key, mode, use_kernel_cache=use_kernel_cache, t0=t0
        )
        model = analysis.compile()
    rcond = float(getattr(cfg, "Q_INTEGRATION_PRIOR_RCOND", 0.0))
    coords = DirectCoordinates.from_model(model, prior_rcond=rcond)
    target = QTarget(coords)
    if verbose:
        print(
            f"target ready in {time.perf_counter() - started:.1f}s: "
            f"dimension={coords.dimension} n_theta={coords.n_theta} "
            f"retained_rank={coords.prior_subspace.rank} "
            f"prior_rcond={rcond:g} t0_tables={len(t0)} "
            f"t0_iterations={int(iterations)}",
            flush=True,
        )
    return model, coords, target, t0


def target_fingerprint(target, coords, *, q_key, mode, t0) -> str:
    """A cache key over everything that would move the MAP.

    Built from ``QTarget.metadata()`` -- which carries a numerical fingerprint
    per compiled likelihood term -- rather than from the case label alone, so a
    regenerated dataset or a changed kernel cache invalidates the stored centre
    instead of silently reusing it.
    """
    payload = json.dumps(
        {
            "metadata": target.metadata(),
            "q_key": str(q_key),
            "mode": str(mode),
            "t0_tables": sorted(t0),
            "dimension": int(coords.dimension),
            "n_theta": int(coords.n_theta),
            "block_ranks": list(coords.prior_subspace.block_ranks),
            "prior_rcond": float(coords.prior_subspace.rcond),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# -- the centre --------------------------------------------------------------


def find_center(target, coords, *, fingerprint, cache_path=None,
                lbfgs_maxiter=DEFAULT_LBFGS_MAXITER,
                newton_maxiter=DEFAULT_NEWTON_MAXITER, verbose=True):
    """``(x, info)`` at the MAP of the explicit-``q`` action.

    Two stages, because one does not work.  L-BFGS-B falls from ``S = 1.85e10``
    to ``875`` in 4,000 iterations and then crawls -- the Hessian at the origin
    has condition number 5.4e10 -- so the quasi-Newton point is handed to
    Newton-CG with exact Hessian-vector products, which reaches
    ``|grad S| = 1e-6`` in about fifty iterations.  No dense 1,321 x 1,321
    Hessian is formed at any point.

    Returns the cached centre when ``cache_path`` holds one for this
    ``fingerprint``.
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    if cache_path is not None:
        cached = _load_center(cache_path, fingerprint, coords.dimension)
        if cached is not None:
            if verbose:
                print(
                    f"centre: cache hit {cache_path} "
                    f"S={cached[1]['action']:.10g} "
                    f"|grad|={cached[1]['gradient_norm']:.3g}",
                    flush=True,
                )
            return cached

    value_and_grad = jax.jit(jax.value_and_grad(target._raw_value))
    hvp = jax.jit(
        lambda x, v: jax.jvp(jax.grad(target._raw_value), (x,), (v,))[1]
    )

    def fun_and_jac(x):
        value, gradient = value_and_grad(jnp.asarray(x))
        return float(np.asarray(value)), np.asarray(gradient, dtype=float)

    def hessp(x, p):
        return np.asarray(hvp(jnp.asarray(x), jnp.asarray(p)), dtype=float)

    start = np.asarray(coords.x0, dtype=float)
    start_value, start_gradient = fun_and_jac(start)
    stages = []

    clock = time.perf_counter()
    quasi = minimize(
        fun_and_jac, start, jac=True, method="L-BFGS-B",
        options={"maxiter": int(lbfgs_maxiter), "gtol": 1.0e-8,
                 "ftol": 1.0e-12, "maxls": 50, "maxcor": 50},
    )
    stages.append({
        "method": "L-BFGS-B", "seconds": time.perf_counter() - clock,
        "iterations": int(quasi.nit), "evaluations": int(quasi.nfev),
        "action": float(quasi.fun),
        "gradient_norm": float(np.linalg.norm(quasi.jac)),
        "success": bool(quasi.success), "message": str(quasi.message),
    })
    if verbose:
        print(f"centre: {stages[-1]}", flush=True)

    clock = time.perf_counter()
    newton = minimize(
        fun_and_jac, np.asarray(quasi.x, dtype=float), jac=True, hessp=hessp,
        method="trust-ncg",
        options={"maxiter": int(newton_maxiter), "gtol": 1.0e-6},
    )
    stages.append({
        "method": "trust-ncg", "seconds": time.perf_counter() - clock,
        "iterations": int(newton.nit), "evaluations": int(newton.nfev),
        "action": float(newton.fun),
        "gradient_norm": float(np.linalg.norm(newton.jac)),
        "success": bool(newton.success), "message": str(newton.message),
    })
    if verbose:
        print(f"centre: {stages[-1]}", flush=True)

    # Keep whichever stage actually got lowest; trust-ncg can report failure
    # ("A bad approximation caused failure to predict improvement") at a point
    # whose gradient norm is nonetheless 1e-6, which is a converged minimum.
    candidates = [
        (start_value, start), (float(quasi.fun), np.asarray(quasi.x, float)),
        (float(newton.fun), np.asarray(newton.x, float)),
    ]
    action, x = min(candidates, key=lambda row: row[0])
    final_value, final_gradient = fun_and_jac(x)
    info = {
        "method": "lbfgs_then_trust_ncg",
        "start_action": float(start_value),
        "start_gradient_norm": float(np.linalg.norm(start_gradient)),
        "action": float(final_value),
        "gradient_norm": float(np.linalg.norm(final_gradient)),
        "stages": stages,
        "fingerprint": fingerprint,
        "cache_status": "miss" if cache_path is not None else "disabled",
    }
    if cache_path is not None:
        _store_center(cache_path, x, info)
    return x, info


def _load_center(cache_path, fingerprint, dimension):
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        stored = np.load(path, allow_pickle=False)
        if str(stored["fingerprint"].item()) != str(fingerprint):
            return None
        x = np.asarray(stored["x"], dtype=float)
        if x.shape != (int(dimension),) or not np.all(np.isfinite(x)):
            return None
        info = json.loads(str(stored["info"].item()))
    except Exception:
        # A corrupt or legacy cache is a reason to recompute, not to fail.
        return None
    info["cache_status"] = "hit"
    return x, info


def _store_center(cache_path, x, info):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, x=np.asarray(x, dtype=float),
        fingerprint=np.asarray(str(info["fingerprint"])),
        info=np.asarray(json.dumps(info)),
    )


# -- the physical-field slice ------------------------------------------------


class FieldSliceTarget:
    """``QTarget`` seen through an affine slice in physically readable coordinates.

    Exposes exactly the protocol :mod:`pixel.plotting.evidence_scan` consumes --
    ``free_labels``, ``free_vector``, ``support_bounds``, ``_raw_value`` and an
    optional ``hessian`` -- over ``m`` coordinates instead of 1,321, with

    ``x(z) = x_center + B @ (z - z_center)``

    and ``B`` constant.  The scanned scalar is the sampler's own action ``S`` at
    the corresponding full point; no Jacobian is introduced for the
    reparametrization, exactly as the H-S scan reports ``E^2`` in physical
    parameters rather than the potential in ``eta``.

    ``support_bounds`` returns ``None`` deliberately, and truthfully: the
    ``q``-target's own coordinates are unconstrained, which is what makes
    ``ParameterCoordinates`` the identity here so a ``"physical"`` scan scans
    ``z`` itself.
    """

    def __init__(self, target, x_center, columns, labels, z_center):
        import jax
        import jax.numpy as jnp

        self._target = target
        self._x_center = jnp.asarray(np.asarray(x_center, dtype=float))
        self._columns = jnp.asarray(np.asarray(columns, dtype=float))
        self._z_center_j = jnp.asarray(np.asarray(z_center, dtype=float))
        self._z_center = np.asarray(z_center, dtype=float)
        self._labels = tuple(str(label) for label in labels)
        self._hessian = jax.jit(jax.hessian(self._raw_value))

    def free_labels(self):
        """Readable labels: normalization amplitudes, then ``field(x=...)``."""
        return self._labels

    def free_vector(self):
        """The centre in reduced coordinates."""
        return self._z_center.copy()

    def support_bounds(self):
        """``None``: the underlying ``q`` coordinates are unconstrained."""
        return None

    def bounds(self):
        """``None``, for the same reason."""
        return None

    def _raw_value(self, z):
        import jax.numpy as jnp

        z = jnp.asarray(z, dtype=float)
        delta = z - self._z_center_j
        return self._target._raw_value(self._x_center + self._columns @ delta)

    def hessian(self, z):
        """The exact reduced ``m x m`` Hessian, for the scanner's windows."""
        import jax.numpy as jnp

        return self._hessian(jnp.asarray(z, dtype=float))


def build_slice(model, coords, target, x_center, *, x_values=DEFAULT_X_VALUES,
                direction="prior", verbose=True):
    """``(FieldSliceTarget, diagnostics)`` over normalizations and field values.

    One coordinate per fitted normalization, then one per ``(field, x)`` pair --
    seven plus twenty-seven on the Q=2 ``exp`` case, which is thirty-four and the
    reason ``MAX_SCAN_PARAMETERS`` had to be raised past eight.

    ``direction``:
        ``"prior"`` moves the field along the GP's own conditional response
        ``K[:, k] / K[k, k]``, so the axis is the field value at ``x_k`` and the
        field moves as a smooth bump.  ``"node"`` moves the single grid node and
        nothing else -- the exact conditional slice in physical ``q``, and a
        delta spike whose window collapses.  See the module docstring.

    ``diagnostics`` carries the snapped nodes, the per-axis prior width, the
    cross-response matrix between scanned field coordinates, and the measured
    residual of the claim that moving axis ``i`` by one moves its own field
    value by exactly one.
    """
    import jax.numpy as jnp

    if direction not in ("prior", "node"):
        raise ValueError(
            f"direction must be 'prior' or 'node', got {direction!r}"
        )
    subspace = coords.prior_subspace
    dimension = int(coords.dimension)
    n_theta = int(coords.n_theta)
    theta_labels = [str(lab) for lab in model.free_partition().theta_labels]
    nodes_by_field = {
        str(field.name): np.asarray(field.nodes, dtype=float)
        for field in model.fields
    }
    q_offsets = {str(name): int(offset) for name, offset, _dim in model.q_layout()}

    theta_center, q_center = coords.physical(jnp.asarray(x_center))
    theta_center = np.asarray(theta_center, dtype=float)
    q_center = np.asarray(q_center, dtype=float)

    column_list, labels, z_center = [], [], []
    for index, label in enumerate(theta_labels):
        column = np.zeros(dimension, dtype=float)
        column[index] = 1.0
        column_list.append(column)
        labels.append(label)
        z_center.append(float(theta_center[index]))

    field_rows = []
    retained_offset = 0
    for block in subspace.blocks:
        name = str(block.name)
        basis = np.asarray(block.Q, dtype=float)
        scale = np.asarray(jnp.diagonal(block.L0), dtype=float)
        eigenvalues = scale ** 2
        nodes = nodes_by_field[name]
        if nodes.size != basis.shape[0]:
            raise ValueError(
                f"field {name!r} has {nodes.size} grid nodes but a "
                f"{basis.shape[0]}-dimensional prior block"
            )
        for wanted in x_values:
            node_index = int(
                np.argmin(np.abs(np.log(nodes) - np.log(float(wanted))))
            )
            # K[k, k] = sum_a Q[k, a]^2 lambda_a on the retained basis, which is
            # exact at the centre theta because the basis *is* the eigensystem
            # of K(theta_center).
            variance = float(np.sum(basis[node_index] ** 2 * eigenvalues))
            if direction == "prior":
                weights = basis[node_index] * scale / variance
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    weights = np.where(scale > 0.0, basis[node_index] / scale, 0.0)
            column = np.zeros(dimension, dtype=float)
            section = slice(
                n_theta + retained_offset,
                n_theta + retained_offset + int(block.rank),
            )
            column[section] = weights
            column_list.append(column)
            labels.append(f"{name}(x={nodes[node_index]:.3g})")
            z_center.append(float(q_center[q_offsets[name] + node_index]))
            field_rows.append({
                "field": name,
                "requested_x": float(wanted),
                "node_x": float(nodes[node_index]),
                "node_index": node_index,
                "prior_variance": variance,
                "prior_sigma": float(np.sqrt(variance)),
                "global_q_index": int(q_offsets[name] + node_index),
                "center_value": float(q_center[q_offsets[name] + node_index]),
            })
        retained_offset += int(block.rank)

    columns = np.stack(column_list, axis=1)
    z_center = np.asarray(z_center, dtype=float)
    slice_target = FieldSliceTarget(
        target, x_center, columns, labels, z_center
    )

    diagnostics = {
        "direction": direction,
        "n_parameters": len(labels),
        "labels": list(labels),
        "n_normalizations": len(theta_labels),
        "field_coordinates": field_rows,
        "center_action": float(target._raw_value(jnp.asarray(x_center))),
        **_slice_residuals(coords, columns, z_center, field_rows, x_center),
    }
    if verbose:
        print(
            f"slice: {len(labels)} coordinates "
            f"({len(theta_labels)} normalizations + {len(field_rows)} field "
            f"values), direction={direction}, "
            f"max self-response error "
            f"{diagnostics['max_self_response_error']:.3g}, "
            f"max cross-response "
            f"{diagnostics['max_abs_cross_response']:.3g}",
            flush=True,
        )
    return slice_target, diagnostics


def _slice_residuals(coords, columns, z_center, field_rows, x_center):
    """Measure, rather than assume, what a unit step along each axis does.

    Two claims are checked numerically through ``coords.physical``: that moving
    axis ``i`` by one moves its own field value by exactly one, and how much it
    moves the *other* scanned field values.  The second is the leakage the
    ``"prior"`` direction trades legibility for; it is recorded instead of
    waved away.
    """
    import jax.numpy as jnp

    n_field = len(field_rows)
    n_norm = columns.shape[1] - n_field
    indices = np.asarray([row["global_q_index"] for row in field_rows], dtype=int)
    base = np.asarray(coords.physical(jnp.asarray(x_center))[1], dtype=float)

    response = np.zeros((n_field, n_field), dtype=float)
    for position in range(n_field):
        step = np.zeros(columns.shape[1], dtype=float)
        step[n_norm + position] = 1.0
        moved = np.asarray(x_center, dtype=float) + columns @ step
        values = np.asarray(coords.physical(jnp.asarray(moved))[1], dtype=float)
        response[:, position] = values[indices] - base[indices]

    # Does a normalization step move the fields?  It should not: the GP mean and
    # covariance depend on shape hyperparameters, and in this mode every theta
    # coordinate is a data normalization amplitude.  If this is not ~0 then
    # "the field value at x" is not an exact coordinate along those axes.
    normalization_leak = 0.0
    for position in range(n_norm):
        step = np.zeros(columns.shape[1], dtype=float)
        step[position] = 1.0
        moved = np.asarray(x_center, dtype=float) + columns @ step
        values = np.asarray(coords.physical(jnp.asarray(moved))[1], dtype=float)
        normalization_leak = max(
            normalization_leak, float(np.max(np.abs(values - base)))
        )

    self_response = np.diag(response)
    off_diagonal = response - np.diag(self_response)
    return {
        "self_response": [float(v) for v in self_response],
        "max_self_response_error": float(np.max(np.abs(self_response - 1.0))),
        "cross_response": [[float(v) for v in row] for row in response],
        "max_abs_cross_response": float(np.max(np.abs(off_diagonal))),
        "max_field_drift_per_unit_normalization": float(normalization_leak),
    }


# -- windows -----------------------------------------------------------------


def resolve_windows(slice_target, z_center, *, half_width, kind="conditional"):
    """``(bounds, info)``: per-axis scan window, and both widths that could set it.

    **A conditional slice needs a conditional window, and that is not what the
    scanner picks by default.**  ``scan_evidence`` sizes an unbounded axis from
    ``sigma_i = sqrt(2 [H^-1]_ii)``, the *marginal* Laplace width -- correct for
    a marginal posterior, and far too wide for a slice through a correlated
    block.  Measured on the 16-parameter pilot of this scan: ``sigma(x=0.1)``
    and ``g(x=0.1)`` came out blank, every node past the ``Delta S = 25`` colour
    ceiling, because they sit on correlations of ``-0.99`` and ``-0.47`` with
    their neighbours.  Along a *held* axis the objective rises as
    ``Delta S = H_ii d^2 / 2``, so the width that puts the ``n``-sigma contour
    at ``n`` units is ``sqrt(2 / H_ii)`` -- the conditional width, and with a
    strong correlation it is orders of magnitude smaller.

    ``kind="marginal"`` returns ``None``, letting the scanner apply its own
    rule, which is what to use when the question is "how wide is this
    parameter" rather than "what does the sampler cross".
    """
    center = np.asarray(z_center, dtype=float)
    hessian = np.asarray(slice_target.hessian(center), dtype=float)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    diagonal = np.diag(hessian)
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = np.sqrt(np.where(diagonal > 0.0, 2.0 / diagonal, np.nan))
    marginal = np.sqrt(
        np.clip(np.diag(2.0 * np.linalg.pinv(hessian)), 0.0, None)
    )
    bounds = None
    if kind == "conditional":
        if not np.all(np.isfinite(conditional)):
            raise ValueError(
                "conditional windows need a positive Hessian diagonal; "
                f"{int(np.count_nonzero(~np.isfinite(conditional)))} axis/axes "
                "are flat or inverted at this centre. Re-run with "
                "--window marginal, or with --center map if you sliced at the "
                "origin."
            )
        bounds = [
            (float(c - half_width * w), float(c + half_width * w))
            for c, w in zip(center, conditional)
        ]
    elif kind != "marginal":
        raise ValueError(
            f"window kind must be 'conditional' or 'marginal', got {kind!r}"
        )
    info = {
        "kind": kind,
        "half_width": float(half_width),
        "hessian_eigenvalue_min": float(eigenvalues.min()),
        "hessian_eigenvalue_max": float(eigenvalues.max()),
        "n_negative_eigenvalues": int(np.count_nonzero(eigenvalues < 0.0)),
        "conditional_sigma": [
            None if not np.isfinite(v) else float(v) for v in conditional
        ],
        "marginal_sigma": [float(v) for v in marginal],
    }
    return bounds, info


# -- cost estimate -----------------------------------------------------------


def estimate_cost(slice_target, z_center, *, n_points, sigma_half_width,
                  window="conditional", verbose=True):
    """Time one real panel and project the whole figure from it.

    The projection is measured, not guessed: a batch of ``n_points**2`` nodes is
    pushed through the same jitted ``lax.map`` path ``scan_evidence`` uses, after
    a warm-up call so the compile does not land in the number.  The panel is
    taken on the *real* windows, so its reported ``Delta S`` span is the one the
    figure will show -- which is how a saturated colour ceiling is caught before
    a 35-minute scan rather than after it.
    """
    import jax
    import jax.numpy as jnp
    from pixel.plotting.evidence_scan import _resolve_bounds, resolve_parameters

    labels = [str(lab) for lab in slice_target.free_labels()]
    indices, _resolved = resolve_parameters(slice_target, labels)
    center = np.asarray(z_center, dtype=float)

    clock = time.perf_counter()
    bounds, window_info = resolve_windows(
        slice_target, center, half_width=sigma_half_width, kind=window
    )
    hessian_seconds = time.perf_counter() - clock
    eigenvalues = np.asarray(
        [window_info["hessian_eigenvalue_min"],
         window_info["hessian_eigenvalue_max"]], dtype=float
    )
    sigma = np.asarray(
        window_info["conditional_sigma" if window == "conditional"
                    else "marginal_sigma"], dtype=float
    )

    clock = time.perf_counter()
    windows, sources = _resolve_bounds(
        slice_target, center, indices, tuple(labels), bounds, sigma_half_width
    )
    bounds_seconds = time.perf_counter() - clock

    objective = jax.jit(
        lambda batch: jax.lax.map(slice_target._raw_value, batch)
    )
    n_nodes = int(n_points) * int(n_points)
    axis_a = np.linspace(*windows[0], int(n_points))
    axis_b = np.linspace(*windows[1], int(n_points))
    grid_a, grid_b = np.meshgrid(axis_a, axis_b, indexing="ij")
    batch = np.repeat(center.reshape(1, -1), n_nodes, axis=0)
    batch[:, indices[0]] = grid_a.reshape(-1)
    batch[:, indices[1]] = grid_b.reshape(-1)

    clock = time.perf_counter()
    first = np.asarray(objective(jnp.asarray(batch)))
    compile_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    warm = np.asarray(objective(jnp.asarray(batch)))
    panel_seconds = time.perf_counter() - clock
    center_value = float(slice_target._raw_value(center))

    n = len(indices)
    n_pairs = n * (n - 1) // 2
    total_nodes = n_pairs * n_nodes + n * int(n_points)
    per_node = panel_seconds / n_nodes
    projected = per_node * total_nodes + compile_seconds

    report = {
        "n_parameters": n,
        "n_pairs": n_pairs,
        "n_panels": n * n,
        "n_points": int(n_points),
        "nodes_per_panel": n_nodes,
        "total_nodes": int(total_nodes),
        "panel_compile_seconds": compile_seconds,
        "panel_warm_seconds": panel_seconds,
        "seconds_per_node": per_node,
        "projected_scan_seconds": projected,
        "projected_scan_hours": projected / 3600.0,
        "reduced_hessian_seconds": hessian_seconds,
        "bounds_seconds": bounds_seconds,
        "reduced_hessian_eigenvalue_min": window_info["hessian_eigenvalue_min"],
        "reduced_hessian_eigenvalue_max": window_info["hessian_eigenvalue_max"],
        "reduced_hessian_n_negative": window_info["n_negative_eigenvalues"],
        "sigma_half_width": float(sigma_half_width),
        "window": window_info,
        "windows": {
            label: {
                "center": float(center[index]),
                "sigma": float(sigma[index]),
                "conditional_sigma": window_info["conditional_sigma"][index],
                "marginal_sigma": window_info["marginal_sigma"][index],
                "lo": float(lo), "hi": float(hi),
                "bounds_source": sources[label],
            }
            for label, index, (lo, hi) in zip(labels, indices, windows)
        },
        "panel_delta_min": float(np.nanmin(warm) - center_value),
        "panel_delta_max": float(np.nanmax(warm) - center_value),
        "panel_n_non_finite": int(np.count_nonzero(~np.isfinite(warm))),
        "center_action": center_value,
        "first_warm_bitwise_identical": bool(np.array_equal(first, warm)),
        "timed_panel": f"{labels[0]} x {labels[1]}",
    }
    if verbose:
        print(
            "\n-- estimate ------------------------------------------------\n"
            f"  {n} parameters -> {n_pairs} pairs, {n * n} panels\n"
            f"  {n_points}x{n_points} per pair -> {total_nodes:,} nodes\n"
            f"  timed panel '{report['timed_panel']}': "
            f"{panel_seconds:.2f}s warm ({compile_seconds:.2f}s first, "
            f"including compile)\n"
            f"  {per_node * 1e3:.2f} ms per node\n"
            f"  projected scan: {projected:.0f}s = "
            f"{projected / 60.0:.1f} min = {projected / 3600.0:.2f} h\n"
            f"  reduced Hessian: {hessian_seconds:.1f}s, eigenvalues "
            f"[{eigenvalues.min():.4g}, {eigenvalues.max():.4g}], "
            f"{report['reduced_hessian_n_negative']} negative\n"
            f"  windows: {window} at {sigma_half_width:g} sigma\n"
            f"  timed panel span: dS in "
            f"[{report['panel_delta_min']:.4g}, "
            f"{report['panel_delta_max']:.4g}], "
            f"{report['panel_n_non_finite']} non-finite\n"
            "------------------------------------------------------------",
            flush=True,
        )
        print(
            "  windows (physical units; 'cond' is the slice width "
            "sqrt(2/H_ii), 'marg' the marginal sqrt(2[H^-1]_ii)):"
        )
        for label, entry in report["windows"].items():
            conditional = entry["conditional_sigma"]
            print(
                f"    {label:<34} centre={entry['center']:+.6g} "
                f"cond={float('nan') if conditional is None else conditional:.4g} "
                f"marg={entry['marginal_sigma']:.4g} "
                f"window=[{entry['lo']:+.6g}, {entry['hi']:+.6g}] "
                f"({entry['bounds_source']})",
                flush=True,
            )
    return report


# -- figures -----------------------------------------------------------------


def save_large_figure(fig, stem, *, inches=DEFAULT_FIGURE_INCHES,
                      max_pixels=MAX_FIGURE_PIXELS):
    """Write PNG and PDF of a figure too large for ``save_figure``'s 180 dpi.

    ``plot_evidence_scan`` sizes itself at 3.1 x 3.0 inches per panel, which is
    fine at eight parameters and 106 x 102 inches at thirty-four.  This shrinks
    the figure to a budget and picks a dpi from the pixel budget.

    **It deliberately does not rasterize the meshes, and that is the measured
    opposite of the obvious choice.**  Rasterizing looks like the way to keep a
    1,156-panel PDF small, but matplotlib's PDF backend allocates a
    *full-figure* Agg buffer for every rasterized artist, so 1,156 of them
    allocate and free an 80 MB buffer each: measured on the 34-parameter figure,
    the rasterized PDF took 62.8 s and peaked at **22.4 GB**, while the plain
    vector PDF took 23.2 s at 1.49 GB and came out 18.1 MB instead of 6.0 MB.
    The PNG is unaffected either way (7.9 s, 1.48 GB).  Twelve megabytes is a
    cheap price for not being killed.
    """
    import matplotlib.pyplot as plt

    width, height = fig.get_size_inches()
    scale = min(1.0, float(inches) / max(width, height))
    fig.set_size_inches(width * scale, height * scale)
    width, height = fig.get_size_inches()
    dpi = min(180.0, float(np.sqrt(max_pixels / max(width * height, 1.0))))
    fig.set_dpi(dpi)

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    # No ``bbox_inches="tight"``: it renders the figure an extra time to measure
    # the ink, which on a 1,156-panel matrix is the difference between finishing
    # and being killed.  The default subplot margins are fractional, so on a
    # 60-inch figure they leave inches of room for the edge labels anyway.
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf, dpi=dpi)
    plt.close(fig)
    return [png, pdf], {
        "figure_inches": [float(width), float(height)],
        "dpi": float(dpi),
        "megapixels": float(width * height * dpi * dpi / 1.0e6),
    }


def plot_scan(slice_target, z_center, out_stem, *, title, bounds=None,
              n_points=DEFAULT_N_POINTS,
              sigma_half_width=DEFAULT_SIGMA_HALF_WIDTH,
              figure_inches=DEFAULT_FIGURE_INCHES):
    """Scan every pair of the reduced coordinates and draw the matrix.

    Returns ``(manifest, scan)``.  The numbers are written next to the figure as
    a ``.npz``: a scan this long should be re-readable without re-running it.
    """
    from pixel.plotting.evidence_scan import plot_evidence_scan, scan_evidence

    labels = [str(lab) for lab in slice_target.free_labels()]
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan = scan_evidence(
            (slice_target, np.asarray(z_center, dtype=float)), labels,
            bounds=bounds, n_points=int(n_points),
            sigma_half_width=float(sigma_half_width), verbose=True,
        )
        figures = plot_evidence_scan(scan, path=None, close=False, title=title)
    scan_seconds = time.perf_counter() - started

    paths, figure_info = save_large_figure(
        figures[0], out_stem, inches=figure_inches
    )
    np.savez_compressed(
        f"{out_stem}_scan.npz",
        labels=np.asarray(scan.labels), indices=np.asarray(scan.indices),
        axes=np.stack([np.asarray(a) for a in scan.axes]),
        center=np.asarray(scan.center),
        center_value=np.asarray(scan.center_value),
        diagonal=np.stack([np.asarray(d) for d in scan.diagonal]),
        **{f"panel_{a}_{b}": np.asarray(v) for (a, b), v in scan.panels.items()},
    )
    manifest = {
        "labels": list(scan.labels),
        "n_points": int(scan.n_points),
        "scan_seconds": scan_seconds,
        "center_value": float(scan.center_value),
        "n_invalid": int(scan.n_invalid),
        "bounds_source": scan.metadata.get("bounds_source"),
        "variation": scan.metadata.get("variation"),
        "worst_dip_below_reference": scan.metadata.get(
            "worst_dip_below_reference"
        ),
        "worst_dip_panel": scan.metadata.get("worst_dip_panel"),
        "delta_max_drawn": float(
            max(
                (float(np.nanmax(np.asarray(v))) for v in scan.panels.values()),
                default=float("nan"),
            ) - scan.center_value
        ),
        "fraction_past_colour_ceiling": float(
            np.mean([
                float(np.mean(np.asarray(v) - scan.center_value > 25.0))
                for v in scan.panels.values()
            ]) if scan.panels else float("nan")
        ),
        "figure": figure_info,
        "paths": [str(path) for path in paths],
        "warnings": [str(w.message) for w in caught],
    }
    print(
        f"wrote {paths[0]} (+ .pdf, _scan.npz) in {scan_seconds:.0f}s "
        f"({scan.n_invalid} masked node(s))",
        flush=True,
    )
    return manifest, scan


def plot_reduced_correlation(slice_target, z_center, out_stem, *, title):
    """The Laplace correlation of the reduced block, whole, in one figure.

    The scan matrix shows every pair but reads one panel at a time; this is its
    quadratic summary at the same centre.  ``S = -2 log p``, so the Gaussian
    covariance is ``2 H^-1`` -- the convention ``scan_evidence`` sizes its own
    windows with, which is what makes the two commensurate.

    Unlike its H-S counterpart this is a Gaussian approximation to a genuinely
    real, positive density at a genuine mode, so it is a *weak* summary rather
    than an invalid one.  A correlation here that the matching scan panel does
    not show is still the approximation failing.
    """
    import matplotlib.pyplot as plt

    labels = [str(lab) for lab in slice_target.free_labels()]
    center = np.asarray(z_center, dtype=float)
    hessian = np.asarray(slice_target.hessian(center), dtype=float)
    hessian = 0.5 * (hessian + hessian.T)
    covariance = 2.0 * np.linalg.pinv(hessian)
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / np.outer(sigma, sigma)
    correlation = np.where(np.isfinite(correlation), correlation, np.nan)

    n = len(labels)
    n_norm = sum(1 for label in labels if " normalization / " in label)
    names = [_short(label) for label in labels]

    fig, (ax, ax_sigma) = plt.subplots(
        1, 2, figsize=(15.0, 10.0), gridspec_kw={"width_ratios": [1.0, 0.34]}
    )
    mesh = ax.pcolormesh(
        np.arange(n + 1), np.arange(n + 1), np.ma.masked_invalid(correlation),
        cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="flat",
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(names, fontsize=7)
    # The normalization/field split is the whole question, so rule it off.
    ax.axhline(n_norm, color="black", lw=1.6)
    ax.axvline(n_norm, color="black", lw=1.6)
    for row in range(n):
        for col in range(n):
            value = correlation[row, col]
            if not np.isfinite(value) or abs(value) < 0.05:
                continue
            ax.text(
                col + 0.5, row + 0.5,
                f"{value:+.2f}".replace("+0.", ".").replace("-0.", "-."),
                ha="center", va="center", fontsize=4.0,
                color="white" if abs(value) > 0.55 else "0.15",
            )
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02).set_label(
        "Laplace correlation", fontsize=9
    )
    ax.set_title(
        f"fitted normalizations (left/top of the rule, {n_norm}) vs "
        f"field values at x ({n - n_norm})", fontsize=10,
    )

    # Points, not bars: the normalization widths and the field widths differ by
    # orders of magnitude, so the axis has to be logarithmic -- and a bar drawn
    # from zero on a log axis has no left edge.
    ax_sigma.plot(
        np.where(sigma > 0.0, sigma, np.nan), np.arange(n),
        marker="o", ls="none", color="tab:blue",
    )
    ax_sigma.set_yticks(np.arange(n))
    ax_sigma.set_yticklabels(names, fontsize=7)
    ax_sigma.invert_yaxis()
    ax_sigma.axhline(n_norm - 0.5, color="black", lw=1.6)
    ax_sigma.set_xscale("log")
    ax_sigma.set_xlabel(r"Laplace $\sigma=\sqrt{2[H^{-1}]_{ii}}$", fontsize=9)
    ax_sigma.set_title("marginal width", fontsize=10)
    ax_sigma.grid(axis="x", ls=":", color="0.8")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths, _info = save_large_figure(fig, out_stem, inches=20.0)

    cross = sorted(
        (
            (labels[i], labels[j], float(correlation[i, j]))
            for i in range(n_norm) for j in range(n_norm, n)
            if np.isfinite(correlation[i, j])
        ),
        key=lambda row: -abs(row[2]),
    )
    print(f"wrote {paths[0]} (+ .pdf)", flush=True)
    return {
        "labels": labels, "short_labels": names, "n_normalizations": n_norm,
        "sigma": [float(v) for v in sigma],
        "correlation": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in correlation
        ],
        "strongest_normalization_field_cross": [
            {"normalization": a, "field": b, "corr": c} for a, b, c in cross[:25]
        ],
        "caveat": (
            "Gaussian approximation at a real-density mode; weak, not invalid. "
            "Read against the evidence scan."
        ),
        "paths": [str(path) for path in paths],
    }


def _short(label: str) -> str:
    """Compact tick label: ``slac_p_f2 normalization / amp`` -> ``slac_p N``."""
    if " normalization / " in label:
        owner = label.split(" normalization / ")[0]
        return f"{owner.replace('_f2', '')} N"
    return label.replace("(x=", " @").replace(")", "")


# -- CLI ---------------------------------------------------------------------


def run(truth, q_key, mode, *, x_values=DEFAULT_X_VALUES,
        n_points=DEFAULT_N_POINTS, direction="prior",
        sigma_half_width=DEFAULT_SIGMA_HALF_WIDTH, window="conditional",
        figure_inches=DEFAULT_FIGURE_INCHES, use_kernel_cache=True,
        center_kind="map", estimate_only=False, figures="all", suffix=""):
    """Build, centre, estimate and (unless ``estimate_only``) scan and draw."""
    with q_integration_activated(truth) as (package, cfg, fit, runner):
        runner._prepare_matplotlib()
        import matplotlib

        matplotlib.use("Agg", force=True)

        if mode not in cfg.TEST_MODES:
            raise ValueError(
                f"mode must be one of {tuple(cfg.TEST_MODES)}, got {mode!r}"
            )
        model, coords, target, t0 = build_target(
            cfg, fit, q_key, mode, use_kernel_cache=use_kernel_cache
        )
        fingerprint = target_fingerprint(
            target, coords, q_key=q_key, mode=mode, t0=t0
        )
        out_dir = cfg.results_dir(q_key)
        out_dir.mkdir(parents=True, exist_ok=True)

        if center_kind == "origin":
            x_center = np.asarray(coords.x0, dtype=float)
            center_info = {
                "method": "prior_mean_origin",
                "action": float(target._raw_value(x_center)),
                "note": (
                    "the point the production q chain starts from, not a mode"
                ),
            }
            print(f"centre: origin S={center_info['action']:.6g}", flush=True)
        else:
            cache_path = (
                Path(cfg.INFERENCE_CACHE_ROOT)
                / f"{cfg.truth_label(q_key)}_{mode}_q_map.npz"
            )
            x_center, center_info = find_center(
                target, coords, fingerprint=fingerprint, cache_path=cache_path
            )

        slice_target, slice_diagnostics = build_slice(
            model, coords, target, x_center,
            x_values=x_values, direction=direction,
        )
        z_center = slice_target.free_vector()

        manifest = {
            "truth": truth, "truth_Q": q_key, "mode": mode,
            "source_package": package,
            "route": "explicit_q (run_q_mcmc), no Hubbard-Stratonovich auxiliary",
            "sampler_dimension": int(coords.dimension),
            "n_theta": int(coords.n_theta),
            "retained_rank": int(coords.prior_subspace.rank),
            "approximate_subspace": bool(
                coords.prior_subspace.approximate_subspace
            ),
            "prior_rcond": float(coords.prior_subspace.rcond),
            "t0_tables": sorted(t0),
            "fingerprint": fingerprint,
            "center_kind": center_kind,
            "center": center_info,
            "slice": slice_diagnostics,
            "requested_x": [float(v) for v in x_values],
            "n_points": int(n_points),
            "sigma_half_width": float(sigma_half_width),
        }

        estimate = estimate_cost(
            slice_target, z_center, n_points=n_points,
            sigma_half_width=sigma_half_width, window=window,
        )
        manifest["estimate"] = estimate
        manifest["window"] = window
        bounds, _window_info = resolve_windows(
            slice_target, z_center, half_width=sigma_half_width, kind=window
        )

        # A pilot must not overwrite a production figure: the two differ only
        # in --n-points and --x, which the filename would not show.
        stem = out_dir / f"q_evidence_scan_{mode}{suffix}"
        if estimate_only:
            path = out_dir / f"q_evidence_scan_{mode}{suffix}_estimate.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"\nwrote {path}", flush=True)
            print(
                "estimate only: nothing scanned, nothing drawn. Re-run without "
                "--estimate to launch.",
                flush=True,
            )
            return manifest

        label = (
            f"small closure {truth} truth Q={q_key}, mode {mode}, "
            f"explicit-q route"
        )
        if figures in ("all", "correlation"):
            manifest["correlation"] = plot_reduced_correlation(
                slice_target, z_center, f"{stem}_laplace_correlation",
                title=(
                    f"Laplace correlation at the explicit-q MAP -- {label} "
                    "(GAUSSIAN APPROXIMATION; read against the evidence scan)"
                ),
            )
        if figures in ("all", "scan"):
            axis_note = (
                "field axes move along the GP conditional response "
                "$K[:,k]/K[k,k]$; see cross_response in the manifest"
                if direction == "prior" else
                "field axes move one grid node only (exact slice, stiff)"
            )
            centred_at = "the MAP" if center_kind == "map" else "the prior-mean origin"
            window_note = (
                "axes windowed at "
                f"{sigma_half_width:g} conditional sigma $\\sqrt{{2/H_{{ii}}}}$"
                if window == "conditional" else
                f"axes windowed at {sigma_half_width:g} marginal Laplace sigma"
            )
            scan_manifest, _scan = plot_scan(
                slice_target, z_center, stem, bounds=bounds,
                title=(
                    f"Evidence scan, {label}: "
                    f"{slice_diagnostics['n_normalizations']} fitted "
                    f"normalizations against "
                    f"{len(slice_diagnostics['field_coordinates'])} field values "
                    f"at x\nphysical coordinates, joint action "
                    f"$S=-2\\log p$, {int(n_points)}x{int(n_points)} per pair, "
                    f"conditional slice at {centred_at} (not a profile); "
                    f"{window_note}; {axis_note}"
                ),
                n_points=n_points, sigma_half_width=sigma_half_width,
                figure_inches=figure_inches,
            )
            manifest["scan"] = scan_manifest

        path = out_dir / f"q_evidence_scan_{mode}{suffix}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {path}", flush=True)
        return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evidence scans of the explicit-q small closure over the fitted "
            "normalizations and three x values of every PDF."
        )
    )
    parser.add_argument("--truth", default="JAM", choices=tuple(TRUTH_SUITES))
    parser.add_argument("--Q", default="2", dest="q_key")
    parser.add_argument("--mode", default="exp")
    parser.add_argument(
        "--x", nargs="+", type=float, default=list(DEFAULT_X_VALUES),
        dest="x_values",
        help=(
            "x values per field, snapped to the nearest grid node in log x "
            f"(default {' '.join(str(v) for v in DEFAULT_X_VALUES)})"
        ),
    )
    parser.add_argument(
        "--n-points", type=int, default=DEFAULT_N_POINTS,
        help=f"nodes per axis (default {DEFAULT_N_POINTS})",
    )
    parser.add_argument(
        "--direction", default="prior", choices=("prior", "node"),
        help=(
            "how a field axis moves the distribution: 'prior' along the GP "
            "conditional response K[:,k]/K[k,k] (readable), 'node' the single "
            "grid node (exact slice, collapsed window)"
        ),
    )
    parser.add_argument(
        "--sigma-half-width", type=float, default=DEFAULT_SIGMA_HALF_WIDTH,
        help=f"axis half-width in sigma (default {DEFAULT_SIGMA_HALF_WIDTH})",
    )
    parser.add_argument(
        "--window", default="conditional", choices=("conditional", "marginal"),
        help=(
            "which sigma sizes an axis: 'conditional' sqrt(2/H_ii), the width "
            "of the slice being drawn, or 'marginal' sqrt(2[H^-1]_ii), the "
            "scanner's own default -- which saturates the colour ceiling on a "
            "strongly correlated block"
        ),
    )
    parser.add_argument(
        "--figure-inches", type=float, default=DEFAULT_FIGURE_INCHES,
        help="longest figure side before saving",
    )
    parser.add_argument(
        "--center", default="map", choices=("map", "origin"), dest="center_kind",
        help="slice at the MAP, or at the prior-mean origin the chain starts from",
    )
    parser.add_argument(
        "--figures", default="all", choices=("all", "scan", "correlation"),
    )
    parser.add_argument(
        "--suffix", default="",
        help=(
            "appended to every output stem, so a pilot (--x/--n-points cut "
            "down) cannot overwrite a production figure"
        ),
    )
    parser.add_argument(
        "--estimate", action="store_true",
        help="build, centre, print the projected cost and the windows, then stop",
    )
    parser.add_argument("--no-kernel-cache", action="store_true")
    args = parser.parse_args(argv)

    run(
        args.truth, args.q_key, args.mode,
        x_values=tuple(args.x_values), n_points=args.n_points,
        direction=args.direction, sigma_half_width=args.sigma_half_width,
        window=args.window, figure_inches=args.figure_inches,
        use_kernel_cache=not args.no_kernel_cache,
        center_kind=args.center_kind, estimate_only=args.estimate,
        figures=args.figures, suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
