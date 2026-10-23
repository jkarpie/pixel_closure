"""MAP-start convergence and cache checks without closure data generation."""

from __future__ import annotations

import jax.numpy as jnp
import jax
import numpy as np
import pytest

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


def test_nonconverged_map_refuses_sampling(monkeypatch):
    monkeypatch.setattr(map_start, "_LBFGS_MAXITER", 0)
    monkeypatch.setattr(map_start, "_NEWTON_MAXITER", 0)
    with pytest.raises(RuntimeError, match="MAP did not converge"):
        map_start._optimize_center(QuadraticTarget(), np.zeros(2))
