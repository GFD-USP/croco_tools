"""Backward-compatible imports for the former monolithic xgcm_tools module.

New code may import grid preparation from :mod:`croco_tools.grid`, velocity
diagnostics from :mod:`croco_tools.kinematics`, and scalar-gradient diagnostics
from :mod:`croco_tools.stratification`. Existing imports from xgcm_tools remain
valid.
"""

from .grid import (
    GridLocation,
    add_horizontal_metrics,
    build_xgcm_grid,
    merge_grid,
    normalize_croco_coords,
    prepare_xgcm,
)
from .kinematics import (
    horizontal_divergence,
    relative_vorticity,
    strain_rate,
    vorticity_at_rho,
)
from .stratification import (
    density_gradient_magnitude,
    horizontal_density_gradient,
)

__all__ = [
    "GridLocation",
    "normalize_croco_coords",
    "merge_grid",
    "add_horizontal_metrics",
    "build_xgcm_grid",
    "prepare_xgcm",
    "relative_vorticity",
    "horizontal_divergence",
    "strain_rate",
    "vorticity_at_rho",
    "horizontal_density_gradient",
    "density_gradient_magnitude",
]
