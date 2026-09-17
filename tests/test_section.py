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


def test_interpolate_section_recovers_linear_field_exactly() -> None:
    # A regular, axis-aligned lon/lat grid is a classic trigger for
    # scipy/Qhull's "flat initial simplex" precision error during
    # Delaunay triangulation -- real CROCO curvilinear grids never look
    # this regular, so a tiny deterministic jitter (well under the grid
    # spacing) makes this a realistic, robust test rather than a
    # pathological one.
    #
    # The field is chosen to be *exactly* linear in longitude
    # (field = lon - lon.min()). Barycentric/linear interpolation over
    # any valid triangulation reproduces an affine function exactly
    # everywhere inside the convex hull -- so this checks the real
    # interpolation logic without needing target points to coincide
    # with source grid nodes (which is what triggered the degenerate
    # triangulation in the first place).
    rng = np.random.default_rng(0)

    lon_1d = np.linspace(-10.0, -8.0, 5)
    lat_1d = np.linspace(-20.0, -19.0, 4)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    jitter = 0.02  # << 0.5 deg / 0.33 deg grid spacing
    lon2d = lon2d + rng.uniform(-jitter, jitter, size=lon2d.shape)
    lat2d = lat2d + rng.uniform(-jitter, jitter, size=lat2d.shape)

    field_values = lon2d - lon2d.min()
    field = xr.DataArray(field_values, dims=("eta_rho", "xi_rho"))
    longitude = xr.DataArray(lon2d, dims=("eta_rho", "xi_rho"))
    latitude = xr.DataArray(lat2d, dims=("eta_rho", "xi_rho"))

    # Target points chosen independently of any grid node, safely
    # inside the domain interior.
    target_lon = np.array([-9.7, -9.3, -8.9, -8.5, -8.3])
    target_lat = np.array([-19.85, -19.7, -19.5, -19.3, -19.15])
    track = Transect.from_coordinates(target_lon, target_lat)

    # padding=1.0: interpolate_section's own default padding (0.1 deg)
    # crops the source grid to the track's bounding box, which -- given
    # the track sits close to this small test grid's own edges -- was
    # excluding the outermost grid rows and leaving some target points
    # outside the cropped convex hull. A generous explicit padding
    # keeps the whole (tiny) test grid in play.
    section = interpolate_section(
        field, track, longitude=longitude, latitude=latitude, padding=1.0
    )

    expected = target_lon - lon2d.min()
    np.testing.assert_allclose(section.values, expected, atol=1e-6)


def test_interpolate_section_raises_for_mismatched_lon_lat_dims() -> None:
    field = xr.DataArray(np.zeros((3, 3)), dims=("eta_rho", "xi_rho"))
    longitude = xr.DataArray(np.zeros((3, 3)), dims=("eta_rho", "xi_rho"))
    latitude = xr.DataArray(np.zeros((3, 4)), dims=("eta_rho", "xi_other"))
    track = Transect.from_coordinates([0.0, 1.0], [0.0, 0.0])

    with pytest.raises(ValueError):
        interpolate_section(field, track, longitude=longitude, latitude=latitude)
