"""Tests for CROCO vertical coordinates."""

import numpy as np
import xarray as xr

from croco_tools.vertical import (
    compute_depths,
    interpolate_to_depth,
)


def make_dataset(
    vtransform: int,
) -> xr.Dataset:
    return xr.Dataset(
        {
            "h": xr.DataArray(
                [[100.0]],
                dims=("eta_rho", "xi_rho"),
            ),
            "zeta": xr.DataArray(
                [[[0.0]]],
                dims=("time", "eta_rho", "xi_rho"),
            ),
            "hc": xr.DataArray(20.0),
            "Vtransform": xr.DataArray(vtransform),
            "sc_r": xr.DataArray(
                [-0.75, -0.25],
                dims=("s_rho",),
            ),
            "sc_w": xr.DataArray(
                [-1.0, -0.5, 0.0],
                dims=("s_w",),
            ),
            "Cs_r": xr.DataArray(
                [-0.75, -0.25],
                dims=("s_rho",),
            ),
            "Cs_w": xr.DataArray(
                [-1.0, -0.5, 0.0],
                dims=("s_w",),
            ),
        }
    )


def test_compute_depth_shapes() -> None:
    dataset = make_dataset(vtransform=2)

    z_rho, z_w = compute_depths(dataset)

    assert z_rho.sizes["s_rho"] == 2
    assert z_w.sizes["s_w"] == 3


def test_interpolate_to_depth_is_linear() -> None:
    depths = xr.DataArray(
        [-100.0, -50.0, 0.0],
        dims=("s_rho",),
    )
    values = xr.DataArray(
        [0.0, 5.0, 10.0],
        dims=("s_rho",),
    )

    result = interpolate_to_depth(
        values,
        depths,
        [-75.0, -25.0],
    )

    np.testing.assert_allclose(
        result.values,
        [2.5, 7.5],
    )
