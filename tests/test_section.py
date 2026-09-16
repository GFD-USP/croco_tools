"""Tests for croco_tools.section."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from croco_tools.section import Transect, interpolate_section


def test_transect_from_coordinates_due_east() -> None:
    # A due-east transect at fixed latitude.
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 0.0, 0.0])

    transect = Transect.from_coordinates(lon, lat)

    assert transect.longitude.size == 3
    np.testing.assert_allclose(transect.distance.values[0], 0.0, atol=1e-9)
    assert np.all(np.diff(transect.distance.values) > 0)

    # Due east: bearing ~90 deg, tangent_east ~1, tangent_north ~0.
    np.testing.assert_allclose(transect.bearing.values, 90.0, atol=0.5)
    np.testing.assert_allclose(transect.tangent_east.values, 1.0, atol=1e-2)
    np.testing.assert_allclose(transect.tangent_north.values, 0.0, atol=1e-2)

    # Left-normal of a due-east transect points due north.
    np.testing.assert_allclose(transect.normal_east.values, 0.0, atol=1e-2)
    np.testing.assert_allclose(transect.normal_north.values, 1.0, atol=1e-2)


def test_transect_requires_at_least_two_points() -> None:
    with pytest.raises(ValueError):
        Transect.from_coordinates([0.0], [0.0])


def test_transect_requires_matching_lengths() -> None:
    with pytest.raises(ValueError):
        Transect.from_coordinates([0.0, 1.0], [0.0, 1.0, 2.0])


def test_transect_requires_1d_input() -> None:
    with pytest.raises(ValueError):
        Transect.from_coordinates([[0.0, 1.0]], [[0.0, 1.0]])


def test_transect_repr_contains_npoints_and_length() -> None:
    transect = Transect.from_coordinates([0.0, 1.0], [0.0, 0.0])
    text = repr(transect)
    assert "npoints=2" in text
    assert "length=" in text


def test_transect_to_dataset_has_expected_variables() -> None:
    transect = Transect.from_coordinates([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    ds = transect.to_dataset()

    for name in (
        "bearing",
        "tangent_east",
        "tangent_north",
        "normal_east",
        "normal_north",
    ):
        assert name in ds


def test_transect_rotate_velocity_due_east_is_identity() -> None:
    transect = Transect.from_coordinates([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    dimension = transect.longitude.dims[0]
    distance = transect.distance.rename({dimension: "distance"})

    u_east = xr.DataArray(
        [1.0, 2.0, 3.0], dims=("distance",), coords={"distance": distance}
    )
    v_north = xr.DataArray(
        [10.0, 20.0, 30.0], dims=("distance",), coords={"distance": distance}
    )

    rotated = transect.rotate_velocity(u_east, v_north)

    # Due east: along-track ~= u_east, across-track ~= v_north.
    np.testing.assert_allclose(
        rotated["velocity_along"].values, u_east.values, atol=1e-2
    )
    np.testing.assert_allclose(
        rotated["velocity_across"].values, v_north.values, atol=1e-2
    )


def test_transect_from_endpoints_uses_grid_spacing(uniform_grid) -> None:
    transect = Transect.from_endpoints(
        uniform_grid,
        start=(0.0, 0.0),
        end=(0.0, 0.1),
    )
    assert transect.longitude.size >= 2


def test_transect_from_endpoints_rejects_invalid_sampling(uniform_grid) -> None:
    with pytest.raises(ValueError):
        Transect.from_endpoints(
            uniform_grid,
            start=(0.0, 0.0),
            end=(0.0, 0.1),
            sampling=0.0,
        )


def test_interpolate_section_recovers_grid_node_values() -> None:
    # A simple regular lon/lat grid and a field linear in longitude.
    lon_1d = np.linspace(-10.0, -8.0, 5)
    lat_1d = np.linspace(-20.0, -19.0, 4)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    field_values = lon2d - lon2d.min()  # simple, known scalar field
    field = xr.DataArray(field_values, dims=("eta_rho", "xi_rho"))
    longitude = xr.DataArray(lon2d, dims=("eta_rho", "xi_rho"))
    latitude = xr.DataArray(lat2d, dims=("eta_rho", "xi_rho"))

    # Track exactly along one grid row, so each target point coincides
    # with a source grid node.
    track = Transect.from_coordinates(lon_1d, np.full_like(lon_1d, lat_1d[1]))

    section = interpolate_section(
        field, track, longitude=longitude, latitude=latitude
    )

    expected = lon_1d - lon_1d.min()
    np.testing.assert_allclose(section.values, expected, atol=1e-6)


def test_interpolate_section_raises_for_mismatched_lon_lat_dims() -> None:
    field = xr.DataArray(np.zeros((3, 3)), dims=("eta_rho", "xi_rho"))
    longitude = xr.DataArray(np.zeros((3, 3)), dims=("eta_rho", "xi_rho"))
    latitude = xr.DataArray(np.zeros((3, 4)), dims=("eta_rho", "xi_other"))
    track = Transect.from_coordinates([0.0, 1.0], [0.0, 0.0])

    with pytest.raises(ValueError):
        interpolate_section(field, track, longitude=longitude, latitude=latitude)
