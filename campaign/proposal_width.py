"""Experimental width grid for PIXEL's normalization-reference saddle-contour proposal.

``normalization_reference_saddle_contour_moments`` chooses one scalar proposal width by a
pilot over a hard-coded grid -- ``np.geomspace(0.35, 1.0, 4)`` with fitted linear (DIS)
normalizations, ``np.geomspace(0.25, 1.0, 7)`` without -- so it can only *narrow* the
fitted Laplace proposal.  The closure DY integrand on the affine saddle contour is far
broader than that Laplace width (``campaign/contour_profiles.py radial``), so this module
substitutes a caller-chosen grid, e.g. ``1..20``, without editing PIXEL: the contour
normalization module is
inside the kernel fingerprint closure.

The substitution swaps ``pixel.infer.contour.normalization.np`` for a proxy that delegates
every attribute to numpy and intercepts ``geomspace`` only for exactly those two argument
tuples. Scaling the proposal covariance changes nothing in the estimator: the importance
weights use the matching precision.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

#: The two pilot grids PIXEL hard-codes, as ``np.geomspace`` positional arguments.
PIXEL_PILOT_GRIDS = ((0.35, 1.0, 4), (0.25, 1.0, 7))


def parse_scales(text: str | None) -> tuple[float, ...] | None:
    if text is None or text == "":
        return None
    scales = tuple(float(s) for s in text.split(","))
    if not scales or any(not np.isfinite(s) or s <= 0.0 for s in scales):
        raise ValueError(f"proposal scales must be positive numbers, got {text!r}")
    return scales


class _NumpyWithPilotGrid:
    def __init__(self, scales):
        self._scales = np.asarray(scales, dtype=float)
        self.intercepted = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def geomspace(self, *args, **kwargs):
        if not kwargs and tuple(float(a) for a in args) in PIXEL_PILOT_GRIDS:
            self.intercepted += 1
            return self._scales.copy()
        return np.geomspace(*args, **kwargs)


@contextmanager
def patched_pilot_scales(scales):
    """Install ``scales`` in PIXEL's contour-normalization module; restore after.

    Yields the proxy (or ``None`` when ``scales`` is ``None``) so a caller can confirm the
    grid was actually used via ``proxy.intercepted``.
    """
    if scales is None:
        yield None
        return
    import pixel.infer.contour.normalization as contour_normalization

    original = contour_normalization.np
    proxy = _NumpyWithPilotGrid(scales)
    contour_normalization.np = proxy
    try:
        yield proxy
    finally:
        contour_normalization.np = original
