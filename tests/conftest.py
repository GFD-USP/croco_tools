"""Shared fixtures for croco_tools tests.

These build small, fully synthetic CROCO-like datasets so tests run in
milliseconds and do not depend on any real model output on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from croco_tools.grid import prepare_xgcm


def _uniform_grid_dataset(
    nx: int = 6,
    ny: int = 5,
    dx: float = 1000.0,
    dy: float = 1000.0,
    f0: float = 1.0e-4,
    h0: float = 100.0,
) -> xr.Dataset:
    """A bare CROCO-like grid: constant spacing, constant Coriolis, flat bottom.

    Dimensions already use croco_tools' internal names (xi_rho, xi_u,
    eta_rho, eta_v), so normalize_croco_coords is a no-op on it.
    """
    ds = xr.Dataset(
        coords={
            "xi_rho": np.arange(nx),
            "eta_rho": np.arange(ny),
            # Present from the start (even with no data variable on them
            # yet) so normalize_croco_coords' C-grid dimension check passes
            # even before u/v are assigned.
            "xi_u": np.arange(nx - 1),
            "eta_v": np.arange(ny - 1),
        }
    )

    ds["pm"] = xr.DataArray(
        np.full((ny, nx), 1.0 / dx),
        dims=("eta_rho", "xi_rho"),
    )
    ds["pn"] = xr.DataArray(
        np.full((ny, nx), 1.0 / dy),
        dims=("eta_rho", "xi_rho"),
    )
    ds["f"] = xr.DataArray(
        np.full((ny, nx), f0),
        dims=("eta_rho", "xi_rho"),
    )
    ds["h"] = xr.DataArray(
        np.full((ny, nx), h0),
        dims=("eta_rho", "xi_rho"),
    )
    ds["mask_rho"] = xr.DataArray(
        np.ones((ny, nx)),
        dims=("eta_rho", "xi_rho"),
    )
    ds["angle"] = xr.DataArray(
        np.zeros((ny, nx)),
        dims=("eta_rho", "xi_rho"),
    )

    return ds


@pytest.fixture
def uniform_grid() -> xr.Dataset:
    """Bare grid dataset, before prepare_xgcm."""
    return _uniform_grid_dataset()


@pytest.fixture
def prepared_uniform_flow(uniform_grid: xr.Dataset):
    """A grid with a spatially uniform velocity field, prepared with xgcm.

    Any horizontal derivative of a uniform field is exactly zero, so this
    is a robust check for relative_vorticity / horizontal_divergence /
    strain_rate without depending on hand-traced xgcm index arithmetic.
    """
    ds = uniform_grid.copy()
    ny = ds.sizes["eta_rho"]
    nx = ds.sizes["xi_rho"]

    ds["u"] = xr.DataArray(
        np.full((ny, nx - 1), 0.3),
        dims=("eta_rho", "xi_u"),
    )
    ds["v"] = xr.DataArray(
        np.full((ny - 1, nx), -0.2),
        dims=("eta_v", "xi_rho"),
    )

    prepared, xgrid = prepare_xgcm(ds)
    return prepared, xgrid


@pytest.fixture
def prepared_sheared_flow(uniform_grid: xr.Dataset):
    """A grid with a horizontally linear (sheared) velocity field.

    u depends linearly on eta (so du/dy is a nonzero constant) and v is
    zero, giving a nonzero, spatially uniform vorticity/strain field.
    Because the field is spatially uniform, normalized and unnormalized
    outputs are related by an exact, known factor (1/f or 1/|f|) even
    without hand-deriving the raw vorticity value.
    """
    ds = uniform_grid.copy()
    ny = ds.sizes["eta_rho"]
    nx = ds.sizes["xi_rho"]

    alpha = 4.0e-5  # s^-1, shear rate
    eta_u = np.arange(ny, dtype=float).reshape(ny, 1)
    ds["u"] = xr.DataArray(
        np.broadcast_to(alpha * eta_u * dy_for(ds), (ny, nx - 1)).copy(),
        dims=("eta_rho", "xi_u"),
    )
    ds["v"] = xr.DataArray(
        np.zeros((ny - 1, nx)),
        dims=("eta_v", "xi_rho"),
    )

    prepared, xgrid = prepare_xgcm(ds)
    return prepared, xgrid, alpha


def dy_for(ds: xr.Dataset) -> float:
    """Grid spacing in metres, assuming a uniform pn."""
    return float(1.0 / ds["pn"].isel(eta_rho=0, xi_rho=0))
