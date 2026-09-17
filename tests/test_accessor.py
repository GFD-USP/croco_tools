"""Tests for the `.croco` xarray Dataset accessor.

Importing croco_tools registers the accessor as a side effect, so these
tests check that the accessor's methods are thin, correctly-wired
wrappers around the standalone functions -- not that the underlying
math is right again (that is covered in test_vertical.py / test_grid.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import croco_tools  # noqa: F401 -- registers the .croco accessor
from croco_tools.grid import rotate_velocity
from croco_tools.vertical import compute_depths, interpolate_to_depth


def _dataset_with_depths_and_velocity() -> xr.Dataset:
    return xr.Dataset(
        {
            "h": xr.DataArray([[100.0]], dims=("eta_rho", "xi_rho")),
            "zeta": xr.DataArray([[0.0]], dims=("eta_rho", "xi_rho")),
            "hc": xr.DataArray(20.0),
            "Vtransform": xr.DataArray(2),
            "sc_r": xr.DataArray([-0.75, -0.25], dims=("s_rho",)),
            "sc_w": xr.DataArray([-1.0, -0.5, 0.0], dims=("s_w",)),
            "Cs_r": xr.DataArray([-0.75, -0.25], dims=("s_rho",)),
            "Cs_w": xr.DataArray([-1.0, -0.5, 0.0], dims=("s_w",)),
            "temp": xr.DataArray(
                [[[1.0, 2.0]]], dims=("eta_rho", "xi_rho", "s_rho")
            ),
            "u": xr.DataArray(np.ones((1, 1)), dims=("eta_rho", "xi_u")),
            "v": xr.DataArray(np.zeros((1, 1)), dims=("eta_v", "xi_rho")),
            "angle": xr.DataArray(np.zeros((1, 1)), dims=("eta_rho", "xi_rho")),
        }
    )


def test_accessor_is_registered() -> None:
    ds = _dataset_with_depths_and_velocity()
    assert hasattr(ds, "croco")


def test_accessor_depths_matches_compute_depths() -> None:
    ds = _dataset_with_depths_and_velocity()

    z_rho_acc, z_w_acc = ds.croco.depths()
    z_rho_fn, z_w_fn = compute_depths(ds)

    np.testing.assert_allclose(z_rho_acc.values, z_rho_fn.values)
    np.testing.assert_allclose(z_w_acc.values, z_w_fn.values)


def test_accessor_velocity_matches_rotate_velocity() -> None:
    ds = _dataset_with_depths_and_velocity()

    u_east_acc, v_north_acc = ds.croco.velocity()
    u_east_fn, v_north_fn = rotate_velocity(ds["u"], ds["v"], ds["angle"])

    np.testing.assert_allclose(u_east_acc.values, u_east_fn.values)
    np.testing.assert_allclose(v_north_acc.values, v_north_fn.values)


def test_accessor_to_depth_matches_manual_pipeline() -> None:
    ds = _dataset_with_depths_and_velocity()

    result_acc = ds.croco.to_depth("temp", depths=-75.0)

    z_rho, _ = compute_depths(ds)
    result_fn = interpolate_to_depth(ds["temp"], z_rho, -75.0)

    np.testing.assert_allclose(
        result_acc.values, result_fn.values, equal_nan=True
    )


def test_accessor_velocity_uses_custom_variable_names() -> None:
    ds = _dataset_with_depths_and_velocity().rename(
        {"u": "u_custom", "v": "v_custom"}
    )

    u_east, v_north = ds.croco.velocity(u_name="u_custom", v_name="v_custom")
    assert u_east is not None
    assert v_north is not None
