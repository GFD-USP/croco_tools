"""Tests for C-grid interpolation and velocity rotation."""

import numpy as np
import pytest
import xarray as xr

from croco_tools.grid import (
    add_horizontal_metrics,
    build_xgcm_grid,
    merge_grid,
    normalize_croco_coords,
    prepare_xgcm,
    psi_to_rho,
    rho_to_psi,
    rho_to_u,
    rho_to_v,
    rotate_velocity,
    u_to_rho,
    v_to_rho,
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


# ---------------------------------------------------------------------------
# u_to_rho / v_to_rho (inverse of rho_to_u / rho_to_v, with edge extension)
# ---------------------------------------------------------------------------


def test_u_to_rho_interior_matches_average() -> None:
    field = xr.DataArray(
        [[1.0, 3.0]],
        dims=("eta_rho", "xi_u"),
    )

    result = u_to_rho(field)

    # 3 rho points from 2 u points: edges duplicate the nearest u value,
    # the interior point is their average.
    np.testing.assert_allclose(
        result.values,
        [[1.0, 2.0, 3.0]],
    )


def test_v_to_rho_interior_matches_average() -> None:
    field = xr.DataArray(
        [[1.0], [3.0]],
        dims=("eta_v", "xi_rho"),
    )

    result = v_to_rho(field)

    np.testing.assert_allclose(
        result.values,
        [[1.0], [2.0], [3.0]],
    )


def test_rho_to_u_then_u_to_rho_interior_is_consistent() -> None:
    field = xr.DataArray(
        [[0.0, 2.0, 4.0, 6.0]],
        dims=("eta_rho", "xi_rho"),
    )

    roundtrip = u_to_rho(rho_to_u(field))

    # The interior points of a linear field are exactly recovered;
    # only the boundary points (nearest-neighbor filled) may differ.
    np.testing.assert_allclose(
        roundtrip.values[:, 1:-1],
        field.values[:, 1:-1],
    )


# ---------------------------------------------------------------------------
# psi_to_rho
# ---------------------------------------------------------------------------


def test_psi_to_rho_pads_to_original_rho_shape() -> None:
    # psi_to_rho's own interior averaging step needs a psi grid of at
    # least 2x2 to produce a non-empty interior before padding -- a
    # real CROCO grid is always far larger than this, but a bare 2x2
    # rho grid (1x1 psi) is a genuine degenerate edge case, not
    # something to test here. Use a 4x4 rho grid instead.
    #
    # Size bookkeeping: rho_to_psi shrinks n -> n-1 (4-point average).
    # psi_to_rho shrinks that by 1 again for its own interior average
    # (n-1 -> n-2), then pads by 1 on each side (n-2 -> n). Net effect:
    # psi_to_rho(rho_to_psi(field)) has exactly the same shape as field.
    field = xr.DataArray(
        np.arange(16.0).reshape(4, 4),
        dims=("eta_rho", "xi_rho"),
    )
    psi = rho_to_psi(field)

    back = psi_to_rho(psi)

    assert back.sizes["eta_rho"] == field.sizes["eta_rho"]
    assert back.sizes["xi_rho"] == field.sizes["xi_rho"]


# ---------------------------------------------------------------------------
# Dimension validation
# ---------------------------------------------------------------------------


def test_rho_to_u_missing_dimension_raises() -> None:
    field = xr.DataArray([1.0, 2.0], dims=("eta_rho",))
    with pytest.raises(ValueError):
        rho_to_u(field)


# ---------------------------------------------------------------------------
# normalize_croco_coords
# ---------------------------------------------------------------------------


def test_normalize_croco_coords_renames_xios_dims() -> None:
    ds = xr.Dataset(
        {
            "temp": xr.DataArray(
                np.zeros((1, 2, 2)),
                dims=("time_counter", "y_rho", "x_rho"),
            ),
            "u": xr.DataArray(
                np.zeros((1, 2, 1)),
                dims=("time_counter", "y_rho", "x_u"),
            ),
            "v": xr.DataArray(
                np.zeros((1, 1, 2)),
                dims=("time_counter", "y_v", "x_rho"),
            ),
        }
    )

    out = normalize_croco_coords(ds)

    assert "xi_rho" in out.dims
    assert "eta_rho" in out.dims
    assert "xi_u" in out.dims
    assert "eta_v" in out.dims
    assert "time" in out.dims


def test_normalize_croco_coords_renames_nav_lon_lat() -> None:
    ds = xr.Dataset(
        {
            "temp": xr.DataArray(
                np.zeros((2, 2)),
                dims=("eta_rho", "xi_rho"),
            ),
        },
        coords={
            "nav_lon": (("eta_rho", "xi_rho"), np.zeros((2, 2))),
            "nav_lat": (("eta_rho", "xi_rho"), np.zeros((2, 2))),
            # Present so the C-grid dimension check in
            # normalize_croco_coords is satisfied.
            "xi_u": np.arange(1),
            "eta_v": np.arange(1),
        },
    )

    out = normalize_croco_coords(ds)
    assert "lon" in out.variables
    assert "lat" in out.variables


def test_normalize_croco_coords_missing_required_dims_raises() -> None:
    ds = xr.Dataset(
        {"temp": xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))}
    )
    with pytest.raises(ValueError):
        normalize_croco_coords(ds)


# ---------------------------------------------------------------------------
# merge_grid
# ---------------------------------------------------------------------------


def test_merge_grid_adds_static_variables() -> None:
    history = xr.Dataset(
        coords={
            "xi_rho": np.arange(2),
            "eta_rho": np.arange(2),
            "xi_u": np.arange(1),
            "eta_v": np.arange(1),
        }
    )
    history["temp"] = xr.DataArray(
        np.zeros((2, 2)), dims=("eta_rho", "xi_rho")
    )

    grid = history.copy()
    grid["pm"] = xr.DataArray(np.full((2, 2), 1.0e-3), dims=("eta_rho", "xi_rho"))
    grid["pn"] = xr.DataArray(np.full((2, 2), 1.0e-3), dims=("eta_rho", "xi_rho"))
    grid["h"] = xr.DataArray(np.full((2, 2), 50.0), dims=("eta_rho", "xi_rho"))

    out = merge_grid(history, grid)

    assert "pm" in out
    assert "pn" in out
    assert "h" in out
    assert "temp" in out


# ---------------------------------------------------------------------------
# add_horizontal_metrics / build_xgcm_grid / prepare_xgcm
# ---------------------------------------------------------------------------


def test_add_horizontal_metrics_derives_dx_dy_from_pm_pn(uniform_grid) -> None:
    out = add_horizontal_metrics(uniform_grid)

    expected_dx = 1.0 / uniform_grid["pm"]
    expected_dy = 1.0 / uniform_grid["pn"]

    np.testing.assert_allclose(out["dx_rho"].values, expected_dx.values)
    np.testing.assert_allclose(out["dy_rho"].values, expected_dy.values)
    np.testing.assert_allclose(
        out["area_rho"].values,
        (expected_dx * expected_dy).values,
    )


def test_add_horizontal_metrics_requires_pm_pn() -> None:
    ds = xr.Dataset(
        coords={
            "xi_rho": np.arange(2),
            "eta_rho": np.arange(2),
            "xi_u": np.arange(1),
            "eta_v": np.arange(1),
        }
    )
    with pytest.raises(KeyError):
        add_horizontal_metrics(ds)


def test_build_xgcm_grid_requires_metrics_first(uniform_grid) -> None:
    with pytest.raises(KeyError):
        build_xgcm_grid(uniform_grid)


def test_prepare_xgcm_returns_dataset_and_grid(uniform_grid) -> None:
    ds, xgrid = prepare_xgcm(uniform_grid)

    assert "dx_rho" in ds
    assert "area_psi" in ds
    assert hasattr(xgrid, "diff")
    assert hasattr(xgrid, "interp")
