"""Experimental H-S saddle-search strategies for closure sampler comparisons.

This module deliberately lives in ``pixel_closure`` until an end-to-end sampler test
shows that the shifted, equilibrated search produces a trustworthy contour.  It patches
only the two PIXEL module globals that dispatch :func:`find_saddle`; the physical action,
its derivatives, and every downstream contour calculation remain PIXEL's implementation.
"""

from __future__ import annotations

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np


SADDLE_STRATEGIES = ("pixel_default", "shifted_diagonal")


def diagonal_newton(
    action,
    start,
    *,
    tol: float,
    max_iter: int,
    max_backtracks: int,
):
    """Damped holomorphic Newton in fixed diagonally equilibrated coordinates.

    If ``z = D f`` and ``T = D^-1``, then the transformed derivatives are
    ``g_z = T.T @ g`` and ``H_z = T.T @ H @ T``.  The Newton step is therefore
    the same physical step as ``solve(H, -g)`` in exact arithmetic.  Only the
    linear solve and the gradient-norm merit used by backtracking see the better
    coordinates; the invariant PIXEL Newton decrement remains the convergence test.
    """
    from pixel.infer.contour.saddle import _newton_decrement

    grad_action = jax.grad(action, holomorphic=True)
    hess_action = jax.jacfwd(grad_action, holomorphic=True)

    def gradient(point):
        return np.asarray(grad_action(jnp.asarray(point, dtype=complex)))

    def hessian(point):
        return np.asarray(hess_action(jnp.asarray(point, dtype=complex)))

    f = np.asarray(start, dtype=complex)
    g = gradient(f)
    initial_hessian = hessian(f)
    diagonal = np.abs(np.diag(initial_hessian))
    finite = diagonal[np.isfinite(diagonal)]
    largest = float(np.max(finite)) if finite.size else 1.0
    floor = max(largest * np.finfo(float).eps, np.finfo(float).tiny)
    inverse_scale = 1.0 / np.sqrt(np.where(np.isfinite(diagonal),
                                           np.maximum(diagonal, floor), floor))
    transform = np.diag(inverse_scale)
    merit = float(np.linalg.norm(transform.T @ g))
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        H = hessian(f)
        if _newton_decrement(H, g) < tol:
            converged = True
            break
        transformed_hessian = transform.T @ H @ transform
        transformed_gradient = transform.T @ g
        try:
            transformed_step = -np.linalg.solve(
                transformed_hessian, transformed_gradient
            )
        except np.linalg.LinAlgError:
            transformed_step = -np.linalg.lstsq(
                transformed_hessian, transformed_gradient, rcond=None
            )[0]
        step = transform @ transformed_step
        alpha = 1.0
        improved = False
        for _ in range(max_backtracks):
            f_try = f + alpha * step
            g_try = gradient(f_try)
            merit_try = float(np.linalg.norm(transform.T @ g_try))
            if np.all(np.isfinite(g_try)) and merit_try < merit:
                f, g, merit = f_try, g_try, merit_try
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
    else:
        H = hessian(f)
        converged = _newton_decrement(H, g) < tol
    return f, g, it, converged


def shifted_diagonal_find_saddle(
    blocks,
    f0=None,
    *,
    rcond: float = 1e-12,
    tol: float = 1e-9,
    max_iter: int = 100,
    max_backtracks: int = 40,
):
    """PIXEL ``find_saddle`` with a natural center and diagonal coordinates.

    The default start is the normalized auxiliary Gaussian center
    ``mu = i C^-1 m``.  An explicit ``f0`` (used by warm-started contour
    reductions) is respected.  Returned derivatives are transformed back to
    physical ``f`` units exactly as in PIXEL's production implementation.
    """
    from pixel.core.evidence import (
        _has_bilinear,
        prepare_hubbard_evidence,
        prepared_hubbard_evidence_complex_f,
    )
    from pixel.infer.contour import saddle as contour_saddle

    if not _has_bilinear(blocks):
        raise ValueError("find_saddle requires bilinear (Hubbard-Stratonovich) data")
    scale = contour_saddle.bilinear_scale(blocks)
    work = contour_saddle._normalized(blocks, scale)
    prepared = prepare_hubbard_evidence(work, rcond)
    action = lambda z: prepared_hubbard_evidence_complex_f(prepared, z)
    start = (np.asarray(prepared.mu, dtype=complex)
             if f0 is None else np.asarray(f0, dtype=complex) / scale)
    f, g, it, converged = diagonal_newton(
        action,
        start,
        tol=tol,
        max_iter=max_iter,
        max_backtracks=max_backtracks,
    )
    f_j = jnp.asarray(f)
    grad_action = jax.grad(action, holomorphic=True)
    hessian = jax.jacfwd(grad_action, holomorphic=True)(f_j)
    return contour_saddle.SaddleResult(
        f_star=f_j * scale,
        grad=jnp.asarray(g) / scale,
        hessian=hessian / scale ** 2,
        evidence=action(f_j),
        n_iter=int(it),
        converged=bool(converged),
    )


@contextmanager
def patched_saddle_search(strategy: str):
    """Temporarily install one experimental saddle strategy in PIXEL dispatch."""
    if strategy not in SADDLE_STRATEGIES:
        raise ValueError(f"unknown saddle strategy {strategy!r}")
    if strategy == "pixel_default":
        yield
        return

    import pixel.infer.contour as contour_api
    from pixel.infer.contour import chains, frames, importance, normalization, quadrature
    import pixel.infer.samplers.nested_vegas as nested_vegas

    targets = (contour_api, chains, frames, importance, normalization, quadrature, nested_vegas)
    originals = {target: target.find_saddle for target in targets}
    for target in targets:
        target.find_saddle = shifted_diagonal_find_saddle
    try:
        yield
    finally:
        for target, original in originals.items():
            target.find_saddle = original
