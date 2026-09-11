"""Modern xarray-based tools for CROCO output."""

from . import accessor as _accessor
from .grid import (
    psi_to_rho,
    rho_to_psi,
    rho_to_u,
    rho_to_v,
    rotate_velocity,
    u_to_rho,
    v_to_rho,
)
from .io import attach_grid, open_croco
from .section import Transect
from .vertical import compute_depths, interpolate_to_depth

from .eos import (
    buoyancy_frequency,
    density_anomaly,
)

from .xgcm_tools import (
    density_gradient_magnitude,
    horizontal_density_gradient,
    horizontal_divergence,
    prepare_xgcm,
    relative_vorticity,
    strain_rate,
    vorticity_at_rho,
)

__all__ = [
    "attach_grid",
    "compute_depths",
    "horizontal_divergence",
    "interpolate_to_depth",
    "open_croco",
    "prepare_xgcm",
    "psi_to_rho",
    "relative_vorticity",
    "rho_to_psi",
    "rho_to_u",
    "rho_to_v",
    "rotate_velocity",
    "strain_rate",
    "Transect",
    "u_to_rho",
    "vorticity_at_rho",
    "v_to_rho",
    "croco_buoyancy_frequency",
    "croco_density",
    "density_gradient_magnitude",
    "horizontal_density_gradient",
]

__version__ = "0.1.0"