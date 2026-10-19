"""Inference policy for the full-scale q-integration closure variant."""

from __future__ import annotations

Q_INTEGRATION_SAMPLER = "nuts"
Q_INTEGRATION_METRIC = "legacy_hessian_dense"
Q_INTEGRATION_TARGET_ACCEPT = 0.90
Q_INTEGRATION_ADAPT_MASS = False

# The source suite supplies its established direct-q warmup, initial step size,
# tree-depth cap, and retained-prior cutoff.  This variant changes only the
# sampling representation, mass metric, and requested NUTS acceptance target.
