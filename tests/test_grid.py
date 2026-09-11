"""Tests for C-grid interpolation and velocity rotation."""

import numpy as np
import xarray as xr

from croco_tools.grid import (
    rho_to_psi,
    rho_to_u,
    rho_to_v,
    rotate_velocity,
)


def test_rho_to_u() -> None:
    field = xr.DataArray(
        [[0.0, 2.0, 4.0]],
        dims=("eta_rho", "xi_rho"),
    )

    result = rho_to_u(field)

    np.testing.assert_allclose(
        result.values,
        [[1.0, 3.0]],
    )


def test_rho_to_v() -> None:
    field = xr.DataArray(
        [[0.0], [2.0], [4.0]],
        dims=("eta_rho", "xi_rho"),
    )

    result = rho_to_v(field)

    np.testing.assert_allclose(
        result.values,
        [[1.0], [3.0]],
    )


def test_rho_to_psi_uses_four_points() -> None:
    field = xr.DataArray(
        [[0.0, 2.0], [4.0, 6.0]],
        dims=("eta_rho", "xi_rho"),
    )

    result = rho_to_psi(field)

    np.testing.assert_allclose(
        result.values,
        [[3.0]],
    )


def test_rotate_velocity_in_degrees() -> None:
    u_velocity = xr.DataArray(
        np.ones((2, 1)),
        dims=("eta_rho", "xi_u"),
    )
    v_velocity = xr.DataArray(
        np.zeros((1, 2)),
        dims=("eta_v", "xi_rho"),
    )
    angle = xr.DataArray(
        np.full((2, 2), 90.0),
        dims=("eta_rho", "xi_rho"),
    )

    eastward, northward = rotate_velocity(
        u_velocity,
        v_velocity,
        angle,
        angle_units="degrees",
    )

    np.testing.assert_allclose(
        eastward.values,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        northward.values,
        1.0,
        atol=1.0e-12,
    )
