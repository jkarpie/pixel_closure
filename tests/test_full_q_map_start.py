"""MAP-start convergence and cache checks without closure data generation."""

from __future__ import annotations

import jax.numpy as jnp
import jax
import numpy as np
import pytest
import json
from types import SimpleNamespace
from pathlib import Path

from closure_full_q_integration import map_start


class QuadraticTarget:
    def _raw_value(self, x):
        center = jnp.array([1.5, -2.0])
        offset = x - center
        return 3.0 * offset[0] ** 2 + 7.0 * offset[1] ** 2

    def action_and_grad(self, x):
        return jax.value_and_grad(self._raw_value)(x)


def test_optimizer_finds_map_not_origin():
    center, info = map_start._optimize_center(
        QuadraticTarget(), np.zeros(2),
        options=map_start.MapOptions(adam_steps=()),
    )
    np.testing.assert_allclose(center, [1.5, -2.0], atol=1.0e-6)
    assert info["gradient_norm"] < 1.0e-4
    assert info["start_action"] > info["action"]


def test_map_cache_rejects_stale_and_nonstationary_entries(tmp_path):
    path = tmp_path / "center.npz"
    info = {
        "fingerprint": "current", "action": 2.0,
        "gradient_norm": 1.0e-7,
    }
    map_start._save_center(path, np.array([1.0, 2.0]), info)
    center, loaded = map_start._load_center(path, "current", 2)
    np.testing.assert_array_equal(center, [1.0, 2.0])
    assert loaded["cache_status"] == "hit"
    assert map_start._load_center(path, "other", 2) is None
    assert map_start._load_center(path, "current", 3) is None
    map_start._save_center(
        path, np.array([1.0, 2.0]), {**info, "gradient_norm": 1.0},
    )
    assert map_start._load_center(path, "current", 2) is None
    map_start._save_center(
        path, np.array([1.0, 2.0]), {**info, "gradient_norm": float("nan")},
    )
    assert map_start._load_center(path, "current", 2) is None


def test_nonconverged_map_refuses_sampling():
    with pytest.raises(RuntimeError, match="MAP did not converge"):
        map_start._optimize_center(QuadraticTarget(), np.zeros(2), options=map_start.MapOptions(
            lbfgs_maxiter=0, newton_maxiter=0, adam_steps=(),
        ))


def test_failure_diagnostics_and_resume_preserve_candidate(tmp_path, monkeypatch):
    coordinates = SimpleNamespace(x0=np.zeros(2), dimension=2)
    monkeypatch.setattr(map_start, "_target", lambda *_a: (
        coordinates, QuadraticTarget(), "quadratic",
    ))
    with pytest.raises(map_start.MapConvergenceError):
        map_start.prepare_q_map_start(
            object(), prior_rcond=0, cache_root=tmp_path,
            options=map_start.MapOptions(lbfgs_maxiter=1, newton_maxiter=0, adam_steps=()),
        )
    stem = tmp_path / "q_map_quadratic"
    attempt = json.loads(stem.with_suffix(".attempt.json").read_text())
    assert attempt["status"] == "failed"
    assert attempt["stages"]["lbfgs"]["nit"] == 1
    assert "LIMIT" in attempt["stages"]["lbfgs"]["message"].upper()
    assert attempt["stages"]["lbfgs"]["nfev"] > 0
    assert attempt["action"] < attempt["start_action"]
    assert not stem.with_suffix(".npz").exists()
    saved_point, saved_info = map_start._load_point(
        stem.with_suffix(".checkpoint.npz"), "quadratic", 2,
    )
    assert saved_info["gradient_norm"] > 1e-3
    with pytest.raises(FileNotFoundError, match="prepare_map"):
        map_start.load_q_map_start(object(), prior_rcond=0, cache_root=tmp_path)
    _, point, info = map_start.prepare_q_map_start(
        object(), prior_rcond=0, cache_root=tmp_path, resume=True,
        options=map_start.MapOptions(lbfgs_maxiter=50, newton_maxiter=20, adam_steps=()),
    )
    np.testing.assert_allclose(point, [1.5, -2], atol=1e-6)
    assert info["resumed"]
    assert info["start_action"] == pytest.approx(saved_info["action"])
    assert not np.array_equal(saved_point, coordinates.x0)
    assert info["gradient_norm"] <= 1e-3
    history = [json.loads(line) for line in stem.with_suffix(".history.jsonl").read_text().splitlines()]
    assert len({row["attempt_id"] for row in history}) == 2


class GaussianModel:
    """q~N(0,1), y~N(q,0.5): analytic MAP q=2*y/3."""
    use_param_priors = False
    n_free = 0
    fields = (SimpleNamespace(name="q", nodes=np.array([0.5])),)

    def __init__(self, mean=1.0):
        self.mean = mean

    def free_partition(self):
        return SimpleNamespace(theta_indices=(), hubbard_indices=(),
                               theta_labels=(), hubbard_labels=(),
                               n_theta=0, n_hubbard=0)

    def free_vector(self):
        return jnp.zeros(0)

    def split_free_vector(self, vec=None):
        return jnp.zeros(0), jnp.zeros(0)

    def support_bounds(self):
        return None

    def q_layout(self):
        return (("q", 0, 1),)

    def _blocks_with_external_hubbard(self, _theta, _hubbard):
        from pixel.core.evidence import EvidenceBlocks
        return EvidenceBlocks(
            M=jnp.array([self.mean]), C=jnp.array([[0.5]]), B=jnp.ones((1, 1)),
            g=jnp.zeros(1), K=jnp.eye(1), param_pred=jnp.zeros(1),
        ), None


def test_real_qtarget_map_roundtrip_and_changed_target_rejection(tmp_path, monkeypatch):
    model = GaussianModel()
    _, point, info = map_start.prepare_q_map_start(
        model, prior_rcond=0, cache_root=tmp_path,
        options=map_start.MapOptions(lbfgs_maxiter=50, newton_maxiter=20,
                                     adam_steps=(50, 30), adam_learning_rates=(0.03, 0.01)),
    )
    np.testing.assert_allclose(point, [2 / 3], atol=1e-7)
    assert tuple(info["stages"]) == ("adam_1", "adam_2", "lbfgs", "newton")
    assert info["stages"]["lbfgs"]["start_action"] <= info["stages"]["adam_2"]["action"]
    monkeypatch.setattr(map_start, "_optimize_center", lambda *a, **k: pytest.fail(
        "cached/stream calls must never repeat optimization"
    ))
    # Changing iteration budgets does not change the target identity.
    _, cached, loaded = map_start.prepare_q_map_start(
        model, prior_rcond=0, cache_root=tmp_path,
        options=map_start.MapOptions(lbfgs_maxiter=99, newton_maxiter=33),
    )
    assert loaded["cache_status"] == "hit"
    assert loaded["fingerprint"] == info["fingerprint"]
    for _ in range(3):
        _, cached, loaded = map_start.load_q_map_start(model, prior_rcond=0, cache_root=tmp_path)
        np.testing.assert_array_equal(cached, point)
    with pytest.raises(FileNotFoundError):
        map_start.load_q_map_start(GaussianModel(2), prior_rcond=0, cache_root=tmp_path)
    # A plausible saved summary cannot hide a nonstationary actual point.
    map_start._save_center(Path(info["cache_path"]), np.zeros(1), info)
    with pytest.raises(FileNotFoundError):
        map_start.load_q_map_start(model, prior_rcond=0, cache_root=tmp_path)


def test_newton_budget_is_used_after_short_bfgs():
    _, info = map_start._optimize_center(
        QuadraticTarget(), np.zeros(2),
        options=map_start.MapOptions(lbfgs_maxiter=1, newton_maxiter=20,
                                     checkpoint_every=1, adam_steps=()),
    )
    assert info["stages"]["lbfgs"]["nit"] == 1
    assert info["stages"]["newton"]["nit"] > 0
    assert info["stages"]["newton"]["nhev"] > 0
    assert info["gradient_norm"] <= 1e-3


def test_adam_schedule_refines_high_dimensional_target_and_hands_off_best_point():
    class ScaledTarget:
        def action_and_grad(self, x):
            center = jnp.linspace(-1.5, 1.5, 128)
            weights = jnp.logspace(-2, 2, 128)
            offset = x - center
            return jnp.sum(weights * offset**2), 2 * weights * offset

    events = []
    point, info = map_start._optimize_center(
        ScaledTarget(), np.zeros(128),
        options=map_start.MapOptions(
            adam_steps=(150, 100, 50), adam_learning_rates=(0.03, 0.01, 0.003),
            lbfgs_maxiter=200, newton_maxiter=50, checkpoint_every=25,
        ), progress=lambda x, row: events.append((x.copy(), row.copy())),
    )
    np.testing.assert_allclose(point, np.linspace(-1.5, 1.5, 128), atol=1e-3)
    assert info["gradient_norm"] <= 1e-3
    assert tuple(info["stages"]) == ("adam_1", "adam_2", "adam_3", "lbfgs", "newton")
    assert info["stages"]["adam_3"]["action"] < info["start_action"] * 0.01
    assert info["stages"]["lbfgs"]["start_action"] < info["start_action"] * 0.01
    actions = [row["action"] for _, row in events]
    assert all(after <= before for before, after in zip(actions, actions[1:]))
    assert any(row["stage"] == "adam_1" and row["iteration"] == 25 for _, row in events)


def test_adam_overshoot_cannot_worsen_lbfgs_start():
    point, info = map_start._optimize_center(
        QuadraticTarget(), np.zeros(2),
        options=map_start.MapOptions(
            adam_steps=(3,), adam_learning_rates=(1e6,),
            lbfgs_maxiter=50, newton_maxiter=20,
        ),
    )
    assert info["stages"]["lbfgs"]["start_action"] == info["start_action"]
    np.testing.assert_allclose(point, [1.5, -2], atol=1e-6)


def test_nonfinite_adam_step_falls_back_to_finite_candidate():
    class FiniteRegion:
        def action_and_grad(self, x):
            valid = jnp.all(jnp.abs(x) < 10)
            return (jnp.where(valid, jnp.sum((x - 1) ** 2), jnp.nan),
                    jnp.where(valid, 2 * (x - 1), jnp.nan))

    point, info = map_start._optimize_center(
        FiniteRegion(), np.zeros(2),
        options=map_start.MapOptions(adam_steps=(10,), adam_learning_rates=(100,),
                                     lbfgs_maxiter=50, newton_maxiter=20),
    )
    assert "Non-finite" in info["stages"]["adam_1"]["message"]
    assert info["stages"]["lbfgs"]["start_action"] == info["start_action"]
    np.testing.assert_allclose(point, np.ones(2), atol=1e-6)


def test_adam_failure_checkpoint_can_restart_through_full_pipeline(tmp_path, monkeypatch):
    coordinates = SimpleNamespace(x0=np.zeros(2), dimension=2)
    monkeypatch.setattr(map_start, "_target", lambda *_a: (
        coordinates, QuadraticTarget(), "adam-restart",
    ))
    with pytest.raises(map_start.MapConvergenceError):
        map_start.prepare_q_map_start(
            object(), prior_rcond=0, cache_root=tmp_path,
            options=map_start.MapOptions(adam_steps=(10,), adam_learning_rates=(0.01,),
                                         lbfgs_maxiter=0, newton_maxiter=0, checkpoint_every=2),
        )
    stem = tmp_path / "q_map_adam-restart"
    attempt = json.loads(stem.with_suffix(".attempt.json").read_text())
    assert attempt["status"] == "failed"
    assert attempt["stages"]["adam_1"]["nit"] == 10
    assert attempt["action"] < attempt["start_action"]
    _, point, info = map_start.prepare_q_map_start(
        object(), prior_rcond=0, cache_root=tmp_path, resume=True,
        options=map_start.MapOptions(adam_steps=(10,), adam_learning_rates=(0.003,),
                                     lbfgs_maxiter=50, newton_maxiter=20),
    )
    assert info["resumed"]
    assert info["start_action"] == pytest.approx(attempt["action"])
    np.testing.assert_allclose(point, [1.5, -2], atol=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"adam_steps": (1, 2), "adam_learning_rates": (0.01,)},
    {"adam_steps": (-1,), "adam_learning_rates": (0.01,)},
    {"adam_learning_rates": (0.01, 0.003, float("nan"))},
    {"adam_learning_rates": (0.01, 0.003, -0.001)},
])
def test_invalid_adam_schedules_are_rejected(kwargs):
    with pytest.raises(ValueError, match="Adam"):
        map_start.MapOptions(**kwargs)
