"""Geographic utilities.

These functions are independent of CROCO and operate on
longitude and latitude coordinates.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import Geod


_WGS84 = Geod(ellps="WGS84")


def geodesic_distance(
    longitude_1,
    latitude_1,
    longitude_2,
    latitude_2,
    *,
    units: str = "km",
):
    """Return geodesic distance between two geographic points.

    Parameters
    ----------
    longitude_1, latitude_1
        Coordinates of the first point.
    longitude_2, latitude_2
        Coordinates of the second point.
    units
        Output units, either ``"m"`` or ``"km"``.

    Returns
    -------
    array-like
        Geodesic distance between the points.
    """
    _, _, distance = _WGS84.inv(
        longitude_1,
        latitude_1,
        longitude_2,
        latitude_2,
    )

    if units == "m":
        return distance

    if units == "km":
        return distance / 1_000.0

    message = "units must be either 'm' or 'km'."
    raise ValueError(message)


def along_track_distance(
    longitude: xr.DataArray,
    latitude: xr.DataArray,
    *,
    dimension: str | None = None,
    units: str = "km",
    origin: str = "start",
) -> xr.DataArray:
    """Return cumulative distance along a geographic track.

    Parameters
    ----------
    longitude, latitude
        One-dimensional longitude and latitude coordinates.
    dimension
        Dimension along the track. When omitted, the sole dimension
        of ``longitude`` is used.
    units
        Output units, either ``"m"`` or ``"km"``.
    origin
        Distance origin. Use ``"start"`` for zero at the first point
        or ``"end"`` for zero at the last point.

    Returns
    -------
    xarray.DataArray
        Cumulative distance along the track.
    """
    if longitude.ndim != 1 or latitude.ndim != 1:
        message = (
            "longitude and latitude must both be "
            "one-dimensional DataArrays."
        )
        raise ValueError(message)

    if longitude.dims != latitude.dims:
        message = (
            "longitude and latitude must have "
            "the same dimension."
        )
        raise ValueError(message)

    if dimension is None:
        dimension = longitude.dims[0]

    if dimension not in longitude.dims:
        message = (
            f"Dimension {dimension!r} was not found "
            f"in longitude dimensions {longitude.dims}."
        )
        raise ValueError(message)

    lon_1 = longitude.isel(
        {dimension: slice(None, -1)}
    )
    lon_2 = longitude.isel(
        {dimension: slice(1, None)}
    )
    lat_1 = latitude.isel(
        {dimension: slice(None, -1)}
    )
    lat_2 = latitude.isel(
        {dimension: slice(1, None)}
    )

    segment_distance = geodesic_distance(
        lon_1.values,
        lat_1.values,
        lon_2.values,
        lat_2.values,
        units=units,
    )

    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(segment_distance),
        ]
    )

    if origin == "end":
        cumulative = cumulative[-1] - cumulative
    elif origin != "start":
        message = (
            "origin must be either 'start' or 'end'."
        )
        raise ValueError(message)

    distance = xr.DataArray(
        cumulative,
        dims=(dimension,),
        coords={
            dimension: longitude[dimension],
        },
        name="distance",
        attrs={
            "long_name": "along-track distance",
            "units": units,
        },
    )

    return distance


def bearing(
    longitude_1,
    latitude_1,
    longitude_2,
    latitude_2,
):
    """Return forward azimuth from the first point to the second.

    The result is expressed in degrees clockwise from geographic north.
    """
    forward_azimuth, _, _ = _WGS84.inv(
        longitude_1,
        latitude_1,
        longitude_2,
        latitude_2,
    )

    return forward_azimuth