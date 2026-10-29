"""An anchored beta-shaped GP mean with three Gaussian prior parameters."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from pixel import priors
from pixel.core.params import Normal, Parameter


class AnchoredBetaMean(priors.MeanFunction):
    """M (x/x0)^a ((1-x)/(1-x0))^b, with M equal to the mean at x0."""

    M: Parameter
    a: Parameter
    b: Parameter
    x0: float = eqx.field(static=True)

    def __init__(self, *, x0: float, M_prior: tuple[float, float],
                 a_prior: tuple[float, float],
                 b_prior: tuple[float, float]):
        if not 0 < x0 < 1:
            raise ValueError("x0 must be inside (0, 1)")
        self.M = Parameter(M_prior[0], prior=Normal(*M_prior), tags=("prior_mean",))
        self.a = Parameter(a_prior[0], prior=Normal(*a_prior), tags=("prior_mean",))
        self.b = Parameter(b_prior[0], prior=Normal(*b_prior), tags=("prior_mean",))
        self.x0 = float(x0)
        self.floor = 1e-6
        self.ndim = 1

    def __call__(self, x):
        nodes = jnp.asarray(x, dtype=jnp.float64)
        if nodes.ndim == 2:
            if nodes.shape[1] != 1:
                raise ValueError("AnchoredBetaMean expects one x coordinate")
            nodes = nodes[:, 0]
        nodes = jnp.clip(nodes, self.floor, 1.0 - 1e-12)
        return self.M.value * jnp.exp(
            self.a.value * jnp.log(nodes / self.x0)
            + self.b.value * jnp.log((1.0 - nodes) / (1.0 - self.x0))
        )
