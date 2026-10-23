"""Reproducible, shared MAP starts for full-scale explicit-q closure streams."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np


_ALGORITHM_VERSION = 1
_LBFGS_MAXITER = 4000
_NEWTON_MAXITER = 200
_MAX_GRADIENT_NORM = 1.0e-3


def _cache_key(model, coordinates, target) -> str:
    """Include numerical target data, coordinates and optimizer policy."""
    from pixel.infer.q_integration._legacy import _direct_model_fingerprint

    payload = {
        "algorithm_version": _ALGORITHM_VERSION,
        "model_fingerprint": _direct_model_fingerprint(model, coordinates),
        "target_metadata": target.metadata(),
        "lbfgs_maxiter": _LBFGS_MAXITER,
        "newton_maxiter": _NEWTON_MAXITER,
        "max_gradient_norm": _MAX_GRADIENT_NORM,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_center(path: Path, fingerprint: str, dimension: int):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["fingerprint"].item()) != fingerprint:
                return None
            center = np.asarray(saved["center"], dtype=float)
            info = json.loads(str(saved["info"].item()))
        if center.shape != (dimension,) or not np.all(np.isfinite(center)):
            return None
        if not np.isfinite(float(info["action"])):
            return None
        gradient_norm = float(info["gradient_norm"])
        if not np.isfinite(gradient_norm) or gradient_norm > _MAX_GRADIENT_NORM:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return center, {**info, "cache_status": "hit"}


def _save_center(path: Path, center: np.ndarray, info: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            center=np.asarray(center, dtype=float),
            fingerprint=np.asarray(info["fingerprint"]),
            info=np.asarray(json.dumps(info, sort_keys=True)),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _optimize_center(target, initial) -> tuple[np.ndarray, dict]:
    """Minimize the explicit-q action, then require a stationary finite center."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    # QTarget's explicit pullback avoids materializing a data-row x grid-node
    # Jacobian during every high-dimensional MAP objective evaluation.
    value_and_grad = jax.jit(target.action_and_grad)
    hessian_vector_product = jax.jit(
        lambda x, direction: jax.jvp(
            lambda point: target.action_and_grad(point)[1],
            (x,), (direction,)
        )[1]
    )

    def objective(x):
        value, gradient = value_and_grad(jnp.asarray(x))
        return float(value), np.asarray(gradient, dtype=float)

    def hessp(x, direction):
        return np.asarray(
            hessian_vector_product(jnp.asarray(x), jnp.asarray(direction)),
            dtype=float,
        )

    origin = np.asarray(initial, dtype=float)
    initial_action, initial_gradient = objective(origin)
    if not np.isfinite(initial_action) or not np.all(np.isfinite(initial_gradient)):
        raise ValueError("full-scale q MAP has a non-finite initial action or gradient")
    quasi = minimize(
        objective, origin, jac=True, method="L-BFGS-B",
        options={"maxiter": _LBFGS_MAXITER, "gtol": 1.0e-8,
                 "ftol": 1.0e-12, "maxls": 50, "maxcor": 50},
    )
    newton = minimize(
        objective, np.asarray(quasi.x, dtype=float), jac=True, hessp=hessp,
        method="trust-ncg",
        options={"maxiter": _NEWTON_MAXITER, "gtol": 1.0e-6},
    )
    candidates = [origin, np.asarray(quasi.x, dtype=float),
                  np.asarray(newton.x, dtype=float)]
    evaluated = []
    for point in candidates:
        action, gradient = objective(point)
        if np.isfinite(action) and np.all(np.isfinite(gradient)):
            evaluated.append((action, point, float(np.linalg.norm(gradient))))
    if not evaluated:
        raise RuntimeError("full-scale q MAP optimizer produced no finite point")
    action, center, gradient_norm = min(evaluated, key=lambda item: item[0])
    if gradient_norm > _MAX_GRADIENT_NORM:
        raise RuntimeError(
            "full-scale q MAP did not converge: "
            f"|gradient|={gradient_norm:.6g} > {_MAX_GRADIENT_NORM:g}; "
            "sampling from the prior-mean origin is refused"
        )
    info = {
        "method": "lbfgs_then_trust_ncg",
        "action": action,
        "gradient_norm": gradient_norm,
        "start_action": initial_action,
        "start_gradient_norm": float(np.linalg.norm(initial_gradient)),
        "lbfgs_success": bool(quasi.success),
        "lbfgs_message": str(quasi.message),
        "newton_success": bool(newton.success),
        "newton_message": str(newton.message),
    }
    return np.asarray(center, dtype=float), info


def prepare_q_map_start(model, *, prior_rcond: float, cache_root: Path):
    """Return exact q coordinates and a cached MAP for the compiled fit target.

    The lock makes simultaneous streams of one cell wait for one optimization.
    No truth curve is used; the center depends only on the fit's target.
    """
    from pixel.infer.q_integration._legacy import DirectCoordinates
    from pixel.infer.q_integration.target import QTarget

    coordinates = DirectCoordinates.from_model(model, prior_rcond=prior_rcond)
    target = QTarget(coordinates)
    fingerprint = _cache_key(model, coordinates, target)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"q_map_{fingerprint}.npz"
    lock_path = cache_root / f"q_map_{fingerprint}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _load_center(path, fingerprint, coordinates.dimension)
        if cached is not None:
            center, info = cached
        else:
            center, info = _optimize_center(target, coordinates.x0)
            info = {**info, "fingerprint": fingerprint, "cache_status": "miss"}
            _save_center(path, center, info)
    return coordinates, center, {**info, "cache_path": str(path)}
