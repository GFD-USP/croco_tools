"""Tests for croco_tools.stratification.

As of this refactor, horizontal_density_gradient / density_gradient_magnitude
are implemented in croco_tools.eos; this module just re-exports them for
backward compatibility. These tests exercise that stratification.py import
path specifically -- see test_eos.py for the canonical-location tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from croco_tools.grid import prepare_xgcm
from croco_tools.stratification import (
    density_gradient_magnitude,
    horizontal_density_gradient,
)


def _prepared_rho_field(uniform_grid: xr.Dataset, values: np.ndarray):
    ds = uniform_grid.copy()
    ny, nx = values.shape
    ds["density"] = xr.DataArray(values, dims=("eta_rho", "xi_rho"))
    prepared, xgrid = prepare_xgcm(ds)
    return prepared, xgrid


def test_gradient_zero_for_uniform_density(uniform_grid: xr.Dataset) -> None:
    ny = uniform_grid.sizes["eta_rho"]
    nx = uniform_grid.sizes["xi_rho"]
    prepared, xgrid = _prepared_rho_field(
        uniform_grid, np.full((ny, nx), 3.0)
    )

    dx, dy = horizontal_density_gradient(
        prepared["density"], xgrid, grid="native"
    )
    np.testing.assert_allclose(dx.values, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(dy.values, 0.0, atol=1.0e-12)


def test_gradient_x_matches_known_linear_slope(uniform_grid: xr.Dataset) -> None:
    ny = uniform_grid.sizes["eta_rho"]
    nx = uniform_grid.sizes["xi_rho"]
    dx_m = float(1.0 / uniform_grid["pm"].isel(eta_rho=0, xi_rho=0))

    slope = 0.002  # kg m-3 per metre
    x_index = np.arange(nx, dtype=float)
    values = np.broadcast_to(slope * x_index * dx_m, (ny, nx)).copy()

    prepared, xgrid = _prepared_rho_field(uniform_grid, values)

    dx, dy = horizontal_density_gradient(
        prepared["density"], xgrid, grid="native"
    )

    np.testing.assert_allclose(dx.values, slope, rtol=1.0e-8)
    np.testing.assert_allclose(dy.values, 0.0, atol=1.0e-12)


def test_density_gradient_magnitude_matches_hypot(uniform_grid: xr.Dataset) -> None:
    ny = uniform_grid.sizes["eta_rho"]
    nx = uniform_grid.sizes["xi_rho"]
    prepared, xgrid = _prepared_rho_field(
        uniform_grid, np.random.default_rng(0).normal(size=(ny, nx))
    )

    dx, dy = horizontal_density_gradient(prepared["density"], xgrid, grid="rho")
    magnitude = density_gradient_magnitude(
        prepared["density"], xgrid, grid="rho"
    )

    np.testing.assert_allclose(
        magnitude.values,
        np.hypot(dx.values, dy.values),
        equal_nan=True,
    )


@pytest.mark.parametrize("grid", ["rho", "psi", "u", "v"])
def test_horizontal_density_gradient_grid_locations(
    uniform_grid: xr.Dataset, grid: str
) -> None:
    ny = uniform_grid.sizes["eta_rho"]
    nx = uniform_grid.sizes["xi_rho"]
    prepared, xgrid = _prepared_rho_field(
        uniform_grid, np.full((ny, nx), 1.0)
    )

    dx, dy = horizontal_density_gradient(prepared["density"], xgrid, grid=grid)
    assert dx.shape == dy.shape


def test_horizontal_density_gradient_requires_rho_point_field(
    uniform_grid: xr.Dataset,
) -> None:
    prepared, xgrid = _prepared_rho_field(
        uniform_grid,
        np.full(
            (
                uniform_grid.sizes["eta_rho"],
                uniform_grid.sizes["xi_rho"],
            ),
            1.0,
        ),
    )
    not_at_rho = prepared["density"].rename({"xi_rho": "xi_u"})

    with pytest.raises(ValueError):
        horizontal_density_gradient(not_at_rho, xgrid, grid="rho")


def test_horizontal_density_gradient_invalid_grid_raises(
    uniform_grid: xr.Dataset,
) -> None:
    prepared, xgrid = _prepared_rho_field(
        uniform_grid,
        np.full(
            (
                uniform_grid.sizes["eta_rho"],
                uniform_grid.sizes["xi_rho"],
            ),
            1.0,
        ),
    )
    with pytest.raises(ValueError):
        horizontal_density_gradient(prepared["density"], xgrid, grid="not-a-location")
