"""Explicit, persistent MAP preparation and load-only starts for full-scale q fits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


_ALGORITHM_VERSION = 2
_LBFGS_MAXITER = 40000
_NEWTON_MAXITER = 2000
_MAX_GRADIENT_NORM = 1.0e-3


@dataclass(frozen=True)
class MapOptions:
    lbfgs_maxiter: int = _LBFGS_MAXITER
    newton_maxiter: int = _NEWTON_MAXITER
    lbfgs_maxfun: int = 400000
    checkpoint_every: int = 50
    report_every: int = 100
    adam_steps: tuple[int, ...] = (2000, 2000, 2000)
    adam_learning_rates: tuple[float, ...] = (0.01, 0.003, 0.001)

    def __post_init__(self):
        if min(self.lbfgs_maxiter, self.newton_maxiter) < 0:
            raise ValueError("MAP iteration limits must be nonnegative")
        if self.lbfgs_maxfun < 1 or self.checkpoint_every < 1 or self.report_every < 1:
            raise ValueError("MAP maxfun, checkpoint and report intervals must be positive")
        if self.report_every % self.checkpoint_every:
            raise ValueError("MAP report interval must be a multiple of checkpoint interval")
        if self.adam_steps and len(self.adam_steps) != len(self.adam_learning_rates):
            raise ValueError("Adam steps and learning rates need the same number of passes")
        if any(steps < 1 for steps in self.adam_steps):
            raise ValueError("Adam steps must be positive; use an empty schedule to disable")
        if any(not np.isfinite(rate) or rate <= 0 for rate in self.adam_learning_rates):
            raise ValueError("Adam learning rates must be finite and positive")


def default_map_cache_root() -> Path:
    """Independent of stream output, so a prepared MAP survives new campaigns."""
    return Path(os.environ.get(
        "PIXEL_Q_MAP_CACHE_ROOT",
        str(Path(__file__).resolve().parents[1] / "campaign" / "_q_map_cache"),
    )).expanduser().resolve()


def campaign_map_cache_root(output_root: Path) -> Path:
    """Keep a campaign's prepared MAPs with its stream artifacts."""
    return Path(output_root).expanduser().resolve() / "_q_map_cache"


def _cache_key(model, coordinates, target) -> str:
    """Identify the numerical target, independently of optimization budgets."""
    from pixel.infer.q_integration._legacy import _direct_model_fingerprint

    payload = {
        "algorithm_version": _ALGORITHM_VERSION,
        "model_fingerprint": _direct_model_fingerprint(model, coordinates),
        "target_metadata": target.metadata(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_point(path: Path, fingerprint: str, dimension: int):
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
        if not np.isfinite(float(info["gradient_norm"])):
            return None
        return center, info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _load_center(path: Path, fingerprint: str, dimension: int):
    cached = _load_point(path, fingerprint, dimension)
    if cached is None or float(cached[1]["gradient_norm"]) > _MAX_GRADIENT_NORM:
        return None
    return cached[0], {**cached[1], "cache_status": "hit"}


def _save_center(path: Path, center: np.ndarray, info: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary, center=np.asarray(center, dtype=float),
            fingerprint=np.asarray(info["fingerprint"]),
            info=np.asarray(json.dumps(info, sort_keys=True)),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_json(path: Path, info: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class MapConvergenceError(RuntimeError):
    def __init__(self, info):
        self.info = info
        super().__init__(
            "full-scale q MAP did not converge: "
            f"|gradient|={info['gradient_norm']:.6g} > {_MAX_GRADIENT_NORM:g}; "
            "sampling is refused. See the saved MAP attempt diagnostics."
        )


def _optimize_center(target, initial, *, options=None, progress=None):
    """L-BFGS, Adam passes, and exact-HVP Newton; preserve failure diagnostics."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from pixel.infer.optimize.adam import adam_minimize

    options = options or MapOptions(
        lbfgs_maxiter=_LBFGS_MAXITER, newton_maxiter=_NEWTON_MAXITER,
    )
    value_and_grad = jax.jit(target.action_and_grad)
    hessian_vector_product = jax.jit(
        lambda x, direction: jax.jvp(
            lambda point: target.action_and_grad(point)[1], (x,), (direction,)
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
    best = (initial_action, origin.copy(), float(np.linalg.norm(initial_gradient)),
            float(np.max(np.abs(initial_gradient))) if initial_gradient.size else 0.0)
    started = time.monotonic()
    info = {
        "method": ("lbfgs_then_adam_then_trust_ncg" if options.adam_steps
                   else "lbfgs_then_trust_ncg"),
        "options": asdict(options),
        "start_action": initial_action,
        "start_gradient_norm": best[2],
        "start_gradient_inf_norm": best[3], "stages": {},
    }

    def record(point, stage, iteration, status="running"):
        nonlocal best
        action, gradient = objective(point)
        norm = float(np.linalg.norm(gradient))
        inf_norm = float(np.max(np.abs(gradient))) if gradient.size else 0.0
        if np.isfinite(action) and np.isfinite(norm) and action <= best[0]:
            best = (action, np.asarray(point, dtype=float).copy(), norm, inf_norm)
        info.update(action=best[0], gradient_norm=best[2],
                    gradient_inf_norm=best[3], stage=stage,
                    iteration=iteration, status=status,
                    elapsed_seconds=time.monotonic() - started)
        if progress is not None:
            progress(best[1], dict(info))

    record(origin, "initial", 0)
    point = origin

    def run_scipy_stage(stage, method, limit, extra):
        nonlocal point
        if limit == 0:
            info["stages"][stage] = {"nit": 0, "message": "disabled", "success": False}
            return
        iteration = 0
        start_action, start_gradient = objective(point)
        info["stages"][stage] = {
            "method": method, "maxiter": limit, "start_action": start_action,
            "start_gradient_norm": float(np.linalg.norm(start_gradient)),
        }
        record(point, stage, 0)

        def callback(x):
            nonlocal iteration
            iteration += 1
            if iteration % options.checkpoint_every == 0:
                record(x, stage, iteration)

        result = minimize(
            objective, point, jac=True, method=method, callback=callback,
            **({"hessp": hessp} if stage == "newton" else {}),
            options={"maxiter": limit, **extra},
        )
        info["stages"][stage].update({
            name: int(result[name]) for name in ("nit", "nfev", "njev", "nhev", "status")
            if name in result
        })
        end_action, end_gradient = objective(result.x)
        info["stages"][stage].update(
            success=bool(result.success), message=str(result.message),
            action=end_action if np.isfinite(end_action) else None,
            gradient_norm=(float(np.linalg.norm(end_gradient))
                           if np.all(np.isfinite(end_gradient)) else None),
        )
        record(result.x, stage, int(result.nit), "stage_complete")
        # Pass the lowest finite action to the next optimizer even if the last
        # step was uphill or non-finite.
        point = best[1]

    run_scipy_stage("lbfgs", "L-BFGS-B", options.lbfgs_maxiter, {
        # Action cancellation can stop this fit well before stationarity.
        "gtol": 1.0e-8, "ftol": 0.0, "maxls": 100, "maxcor": 50,
        "maxfun": options.lbfgs_maxfun,
    })
    # Reuse PIXEL's bias-corrected Adam on the same explicit-q action/gradient.
    # Q coordinates already encode physical support, so no extra box clipping
    # or posterior approximation is introduced here.
    adam_target = SimpleNamespace(value_and_grad=value_and_grad, bounds=lambda: None)
    for number, (steps, rate) in enumerate(zip(
        options.adam_steps, options.adam_learning_rates,
    ), 1):
        stage = f"adam_{number}"
        start_action, start_gradient = objective(point)
        stage_info = {
            "method": "adam", "learning_rate": rate, "maxiter": steps,
            "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
            "start_action": start_action,
            "start_gradient_norm": float(np.linalg.norm(start_gradient)),
        }
        info["stages"][stage] = stage_info
        record(point, stage, 0)

        def adam_callback(iteration, x, _action, _norm):
            if iteration and iteration % options.checkpoint_every == 0:
                record(x, stage, iteration)

        result = adam_minimize(
            adam_target, point, learning_rate=rate, max_iter=steps,
            beta1=0.9, beta2=0.999, eps=1e-8,
            gtol=1e-6, ftol=0.0, callback=adam_callback, verbose=False,
        )
        norm = float(np.linalg.norm(result.jac))
        stage_info.update(
            nit=int(result.nit), nfev=int(result.nfev),
            success=bool(result.success), message=str(result.message),
            action=float(result.fun) if np.isfinite(result.fun) else None,
            gradient_norm=norm if np.isfinite(norm) else None,
        )
        record(result.x, stage, int(result.nit), "stage_complete")
        # Reset moments for the next learning rate; retain the lowest finite
        # action so an overshoot cannot worsen the Newton start.
        point = best[1]

    run_scipy_stage("newton", "trust-ncg", options.newton_maxiter,
                    {"gtol": 1.0e-6})
    converged = best[2] <= _MAX_GRADIENT_NORM
    record(best[1], "finished", 0, "complete" if converged else "failed")
    if not converged:
        raise MapConvergenceError(info)
    return best[1], info


def _target(model, prior_rcond):
    from pixel.infer.q_integration._legacy import DirectCoordinates
    from pixel.infer.q_integration.target import QTarget

    coordinates = DirectCoordinates.from_model(model, prior_rcond=prior_rcond)
    target = QTarget(coordinates)
    return coordinates, target, _cache_key(model, coordinates, target)


def _validated_center(path, fingerprint, coordinates, target):
    cached = _load_center(path, fingerprint, coordinates.dimension)
    if cached is None:
        return None
    center, info = cached
    action, gradient = target.action_and_grad(center)
    action, norm = float(action), float(np.linalg.norm(np.asarray(gradient)))
    if (not np.isfinite(action) or not np.isfinite(norm)
            or norm > _MAX_GRADIENT_NORM
            or not np.isclose(action, info["action"], rtol=1e-9, atol=1e-8)):
        return None
    inf_norm = float(np.max(np.abs(np.asarray(gradient)))) if coordinates.dimension else 0.0
    return center, {**info, "action": action, "gradient_norm": norm,
                    "gradient_inf_norm": inf_norm,
                    "cache_path": str(path)}


def load_q_map_start(model, *, prior_rcond: float, cache_root: Path):
    """Load and re-evaluate a stationary MAP; never optimize in a stream."""
    coordinates, target, fingerprint = _target(model, prior_rcond)
    path = Path(cache_root) / f"q_map_{fingerprint}.npz"
    cached = _validated_center(path, fingerprint, coordinates, target)
    if cached is None:
        raise FileNotFoundError(
            f"no validated full-scale q MAP for this target: {path}. "
            "Run python -m closure_full_q_integration.prepare_map with the same "
            "--truth, --data, --Q and --modes first (and the same MAP cache root)."
        )
    return coordinates, *cached


def prepare_q_map_start(model, *, prior_rcond: float, cache_root: Path,
                        options=None, resume=False, progress=None):
    """Explicit preparation only; save checkpoints and optimizer diagnostics."""
    coordinates, target, fingerprint = _target(model, prior_rcond)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = cache_root / f"q_map_{fingerprint}"
    path = stem.with_suffix(".npz")
    checkpoint = stem.with_suffix(".checkpoint.npz")
    attempt_path = stem.with_suffix(".attempt.json")
    with stem.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _validated_center(path, fingerprint, coordinates, target)
        if cached is not None:
            return coordinates, *cached
        initial = coordinates.x0
        if resume:
            saved = _load_point(checkpoint, fingerprint, coordinates.dimension)
            if saved is None:
                raise FileNotFoundError(f"no matching finite MAP checkpoint: {checkpoint}")
            initial = saved[0]
        attempt_id = time.time_ns()
        latest = {
            "fingerprint": fingerprint, "attempt_id": attempt_id,
            "status": "starting", "resumed": bool(resume),
            "options": asdict(options or MapOptions()), "updated_at": time.time(),
        }
        _save_json(attempt_path, latest)

        def persist(center, info):
            nonlocal latest
            latest = {**info, "fingerprint": fingerprint, "attempt_id": attempt_id,
                      "resumed": bool(resume), "updated_at": time.time(),
                      "checkpoint_path": str(checkpoint)}
            _save_center(checkpoint, center, latest)
            _save_json(attempt_path, latest)
            with stem.with_suffix(".history.jsonl").open("a") as history:
                history.write(json.dumps(latest, sort_keys=True) + "\n")
            if progress is not None:
                progress(center, latest)

        try:
            center, info = _optimize_center(
                target, initial, options=options, progress=persist,
            )
        except BaseException as exc:
            if latest:
                _save_json(attempt_path, {
                    **latest, "status": "interrupted" if isinstance(
                        exc, (KeyboardInterrupt, SystemExit)) else "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            raise
        info = {**latest, **info, "fingerprint": fingerprint, "cache_status": "miss"}
        _save_center(path, center, info)
    return coordinates, center, {**info, "cache_path": str(path)}
