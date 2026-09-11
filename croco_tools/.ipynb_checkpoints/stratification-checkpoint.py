"""Horizontal scalar and density-gradient diagnostics on the CROCO C-grid."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from xgcm import Grid

from .grid import GridLocation, _move_to_grid, _target_location

def horizontal_density_gradient(
    density: xr.DataArray,
    xgrid: Grid,
    *,
    grid: Literal["native", "rho", "psi", "u", "v"] = "rho",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return horizontal density-anomaly gradients."""

    if "xi_rho" not in density.dims or "eta_rho" not in density.dims:
        raise ValueError(
            "density must be defined at CROCO rho points."
        )

    # Native derivatives:
    #   dρ/dx: rho -> u
    #   dρ/dy: rho -> v
    #
    # Use edge extension to avoid differencing against an artificial
    # zero value outside the model domain.
    drho_dx = xgrid.derivative(
        density,
        "X",
        boundary="extend",
    )

    drho_dy = xgrid.derivative(
        density,
        "Y",
        boundary="extend",
    )

    if grid == "native":
        return drho_dx, drho_dy

    if grid == "u":
        drho_dy = xgrid.interp(
            drho_dy,
            "X",
            boundary="extend",
        )

    elif grid == "v":
        drho_dx = xgrid.interp(
            drho_dx,
            "Y",
            boundary="extend",
        )

    elif grid == "psi":
        drho_dx = xgrid.interp(
            drho_dx,
            "Y",
            boundary="extend",
        )
        drho_dy = xgrid.interp(
            drho_dy,
            "X",
            boundary="extend",
        )

    elif grid == "rho":
        drho_dx = xgrid.interp(
            drho_dx,
            "X",
            boundary="extend",
        )
        drho_dy = xgrid.interp(
            drho_dy,
            "Y",
            boundary="extend",
        )

        # Centered rho-point gradients are undefined at the external
        # boundary. Mask only the edge relevant to each component.
        drho_dx = drho_dx.where(
            (drho_dx["xi_rho"] != drho_dx["xi_rho"].isel(xi_rho=0))
            & (drho_dx["xi_rho"] != drho_dx["xi_rho"].isel(xi_rho=-1))
        )

        drho_dy = drho_dy.where(
            (drho_dy["eta_rho"] != drho_dy["eta_rho"].isel(eta_rho=0))
            & (drho_dy["eta_rho"] != drho_dy["eta_rho"].isel(eta_rho=-1))
        )

    else:
        raise ValueError(
            "grid must be one of 'native', 'rho', 'psi', 'u', or 'v'."
        )

    drho_dx.name = "density_anomaly_gradient_x"
    drho_dy.name = "density_anomaly_gradient_y"

    return drho_dx, drho_dy

def density_gradient_magnitude(
    density: xr.DataArray,
    xgrid: Grid,
    *,
    grid: Literal["rho", "psi", "u", "v"] = "rho",
) -> xr.DataArray:
    """Return the magnitude of the horizontal density-anomaly gradient."""

    drho_dx, drho_dy = horizontal_density_gradient(
        density,
        xgrid,
        grid=grid,
    )

    magnitude = np.hypot(drho_dx, drho_dy)
    magnitude.name = "density_anomaly_gradient_magnitude"
    magnitude.attrs.update(
        {
            "long_name": "horizontal density anomaly gradient magnitude",
            "units": "kg m-4",
        }
    )

    return magnitude
