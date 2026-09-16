"""Tests for croco_tools.geo.

geodesic_distance/bearing wrap pyproj.Geod directly, so these tests check
invariants (symmetry, additivity, zero self-distance, known landmark
values with a generous tolerance) rather than re-deriving WGS84 geodesy
by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from croco_tools.geo import along_track_distance, bearing, geodesic_distance


def test_geodesic_distance_zero_for_same_point() -> None:
    d = geodesic_distance(-45.0, -23.0, -45.0, -23.0)
    assert d == pytest.approx(0.0, abs=1.0e-9)


def test_geodesic_distance_symmetric() -> None:
    d_forward = geodesic_distance(-45.0, -23.0, -43.0, -22.0)
    d_backward = geodesic_distance(-43.0, -22.0, -45.0, -23.0)
    assert d_forward == pytest.approx(d_backward, rel=1.0e-9)


def test_geodesic_distance_one_degree_latitude_is_about_111_km() -> None:
    # One degree of latitude is close to 111.2 km almost everywhere on
    # the WGS84 ellipsoid; loose tolerance since it varies slightly with
    # latitude.
    d = geodesic_distance(0.0, 0.0, 0.0, 1.0, units="km")
    assert d == pytest.approx(111.2, rel=0.01)


def test_geodesic_distance_units_m_vs_km() -> None:
    d_km = geodesic_distance(0.0, 0.0, 0.0, 1.0, units="km")
    d_m = geodesic_distance(0.0, 0.0, 0.0, 1.0, units="m")
    assert d_m == pytest.approx(d_km * 1000.0, rel=1.0e-9)


def test_geodesic_distance_invalid_units_raises() -> None:
    with pytest.raises(ValueError):
        geodesic_distance(0.0, 0.0, 1.0, 1.0, units="furlongs")


def test_along_track_distance_starts_at_zero() -> None:
    lon = xr.DataArray([-45.0, -44.0, -43.0], dims=("s",))
    lat = xr.DataArray([-23.0, -23.0, -23.0], dims=("s",))

    distance = along_track_distance(lon, lat)

    assert float(distance.isel(s=0)) == pytest.approx(0.0, abs=1.0e-9)


def test_along_track_distance_is_monotonically_increasing() -> None:
    lon = xr.DataArray([-45.0, -44.0, -43.0, -42.5], dims=("s",))
    lat = xr.DataArray([-23.0, -23.2, -23.5, -23.6], dims=("s",))

    distance = along_track_distance(lon, lat).values
    assert np.all(np.diff(distance) > 0)


def test_along_track_distance_is_additive() -> None:
    lon = xr.DataArray([-45.0, -44.0, -43.0], dims=("s",))
    lat = xr.DataArray([-23.0, -23.0, -23.0], dims=("s",))

    distance = along_track_distance(lon, lat).values
    leg_1 = geodesic_distance(-45.0, -23.0, -44.0, -23.0)
    leg_2 = geodesic_distance(-44.0, -23.0, -43.0, -23.0)

    assert distance[-1] == pytest.approx(leg_1 + leg_2, rel=1.0e-9)


def test_along_track_distance_origin_end_reverses() -> None:
    lon = xr.DataArray([-45.0, -44.0, -43.0], dims=("s",))
    lat = xr.DataArray([-23.0, -23.0, -23.0], dims=("s",))

    from_start = along_track_distance(lon, lat, origin="start").values
    from_end = along_track_distance(lon, lat, origin="end").values

    assert from_end[-1] == pytest.approx(0.0, abs=1.0e-9)
    assert from_start[-1] == pytest.approx(from_end[0], rel=1.0e-9)


def test_along_track_distance_requires_matching_dims() -> None:
    lon = xr.DataArray([0.0, 1.0], dims=("s",))
    lat = xr.DataArray([0.0, 1.0], dims=("t",))
    with pytest.raises(ValueError):
        along_track_distance(lon, lat)


def test_along_track_distance_requires_1d() -> None:
    lon = xr.DataArray([[0.0, 1.0]], dims=("a", "b"))
    lat = xr.DataArray([[0.0, 1.0]], dims=("a", "b"))
    with pytest.raises(ValueError):
        along_track_distance(lon, lat)


def test_bearing_due_east_is_ninety_degrees() -> None:
    az = bearing(0.0, 0.0, 1.0, 0.0)
    assert az == pytest.approx(90.0, abs=0.5)


def test_bearing_due_north_is_zero_degrees() -> None:
    az = bearing(0.0, 0.0, 0.0, 1.0)
    assert az == pytest.approx(0.0, abs=0.5)
