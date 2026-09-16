"""Backward-compatible re-export for horizontal density-gradient diagnostics.

These functions now live in :mod:`croco_tools.eos`, alongside the rest of
the density-related code (the CROCO equation of state). Existing imports
from ``croco_tools.stratification`` continue to work unchanged.
"""

from __future__ import annotations

from .eos import (
    density_gradient_magnitude,
    horizontal_density_gradient,
)

__all__ = [
    "density_gradient_magnitude",
    "horizontal_density_gradient",
]
