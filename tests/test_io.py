"""Tests for croco_tools.io."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from croco_tools.io import GRID_VARIABLES, attach_grid, open_croco, promote_coordinates


def _history_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "temp": xr.DataArray(
                np.zeros((2, 2, 2)),
                dims=("time", "eta_rho", "xi_rho"),
            ),
        }
    )


def _grid_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "h": xr.DataArray(np.full((2, 2), 50.0), dims=("eta_rho", "xi_rho")),
            "pm": xr.DataArray(np.full((2, 2), 1.0e-3), dims=("eta_rho", "xi_rho")),
            "pn": xr.DataArray(np.full((2, 2), 1.0e-3), dims=("eta_rho", "xi_rho")),
            "mask_rho": xr.DataArray(np.ones((2, 2)), dims=("eta_rho", "xi_rho")),
            "lon_rho": xr.DataArray(np.zeros((2, 2)), dims=("eta_rho", "xi_rho")),
            "lat_rho": xr.DataArray(np.zeros((2, 2)), dims=("eta_rho", "xi_rho")),
        }
    )


def test_attach_grid_adds_missing_variables() -> None:
    history = _history_dataset()
    grid = _grid_dataset()

    out = attach_grid(history, grid)

    for name in ("h", "pm", "pn", "mask_rho"):
        assert name in out.variables


def test_attach_grid_does_not_overwrite_by_default() -> None:
    history = _history_dataset()
    history["h"] = xr.DataArray(
        np.full((2, 2), 999.0), dims=("eta_rho", "xi_rho")
    )
    grid = _grid_dataset()

    out = attach_grid(history, grid, overwrite=False)

    np.testing.assert_allclose(out["h"].values, 999.0)


def test_attach_grid_overwrite_true_replaces_existing() -> None:
    history = _history_dataset()
    history["h"] = xr.DataArray(
        np.full((2, 2), 999.0), dims=("eta_rho", "xi_rho")
    )
    grid = _grid_dataset()

    out = attach_grid(history, grid, overwrite=True)

    np.testing.assert_allclose(out["h"].values, 50.0)


def test_attach_grid_promotes_grid_variables_to_coordinates() -> None:
    history = _history_dataset()
    grid = _grid_dataset()

    out = attach_grid(history, grid)

    assert "h" in out.coords
    assert "pm" in out.coords
    assert "temp" not in out.coords  # data variables stay data variables


def test_attach_grid_ignores_variables_absent_from_grid_file() -> None:
    history = _history_dataset()
    grid = _grid_dataset().drop_vars("mask_rho")

    out = attach_grid(history, grid)

    assert "mask_rho" not in out.variables


def test_promote_coordinates_is_noop_without_grid_variables() -> None:
    history = _history_dataset()
    out = promote_coordinates(history)
    assert set(out.coords) == set(history.coords)


def test_grid_variables_constant_matches_expected_names() -> None:
    # Documents the intended set of variables attach_grid understands;
    # guards against an accidental typo/removal going unnoticed.
    assert "h" in GRID_VARIABLES
    assert "pm" in GRID_VARIABLES
    assert "angle" in GRID_VARIABLES
    assert "f" in GRID_VARIABLES


def test_open_croco_single_file(tmp_path) -> None:
    netcdf = pytest.importorskip("netCDF4")  # noqa: F841 -- skip if unavailable

    history_path = tmp_path / "his.nc"
    grid_path = tmp_path / "grd.nc"

    _history_dataset().to_netcdf(history_path)
    _grid_dataset().to_netcdf(grid_path)

    ds = open_croco(history_path, grid=grid_path)

    assert "temp" in ds
    assert "h" in ds.coords
