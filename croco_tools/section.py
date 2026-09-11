"""Tools for defining and extracting model transects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay
from pyproj import Geod

from .geo import along_track_distance


_WGS84 = Geod(ellps="WGS84")


@dataclass
class Transect:
    """Geographic transect used for section extraction.

    A transect may be defined either by two endpoints or by arrays of
    longitude and latitude coordinates.

    Parameters
    ----------
    longitude, latitude
        One-dimensional arrays defining the transect path.
    """

    longitude: xr.DataArray
    latitude: xr.DataArray

    def __post_init__(self) -> None:
        """Validate coordinates and calculate transect geometry."""
        if self.longitude.ndim != 1 or self.latitude.ndim != 1:
            raise ValueError(
                "longitude and latitude must be one-dimensional."
            )

        if self.longitude.sizes != self.latitude.sizes:
            raise ValueError(
                "longitude and latitude must contain the same number "
                "of points."
            )

        if self.longitude.size < 2:
            raise ValueError(
                "A transect must contain at least two points."
            )

        dimension = self.longitude.dims[0]

        if self.latitude.dims[0] != dimension:
            self.latitude = self.latitude.rename(
                {self.latitude.dims[0]: dimension}
            )

        self.longitude = self.longitude.rename("longitude")
        self.latitude = self.latitude.rename("latitude")

        self.distance = along_track_distance(
            self.longitude,
            self.latitude,
            dimension=dimension,
            units="km",
        )

        self.bearing = self._calculate_bearing()
        self.tangent_east, self.tangent_north = (
            self._calculate_tangent()
        )
        self.normal_east, self.normal_north = (
            self._calculate_left_normal()
        )

    @classmethod
    def from_endpoints(
        cls,
        dataset: xr.Dataset,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        sampling: float = 1.0,
        npoints: int | None = None,
        dimension: str = "s",
    ) -> "Transect":
        """Create a straight transect between two geographic points.

        Parameters
        ----------
        dataset
            CROCO dataset containing ``pm`` and ``pn`` grid metrics.
        start, end
            Transect endpoints as ``(longitude, latitude)``.
        sampling
            Sampling interval relative to the model grid spacing.
            The default, 1.0, generates approximately one transect
            point per model grid cell. A value of 0.5 oversamples by
            a factor of two.
        npoints
            Explicit number of transect points. When provided, this
            overrides automatic sampling.
        dimension
            Name of the transect dimension.

        Returns
        -------
        Transect
            Transect geometry object.
        """
        if sampling <= 0:
            raise ValueError("sampling must be greater than zero.")

        start_lon, start_lat = start
        end_lon, end_lat = end

        forward_azimuth, _, length_m = _WGS84.inv(
            start_lon,
            start_lat,
            end_lon,
            end_lat,
        )

        if npoints is None:
            grid_spacing = cls._estimate_grid_spacing(dataset)
            target_spacing = sampling * grid_spacing

            number_of_segments = max(
                1,
                int(np.ceil(length_m / target_spacing)),
            )
            npoints = number_of_segments + 1

        if npoints < 2:
            raise ValueError("npoints must be at least 2.")

        intermediate = _WGS84.npts(
            start_lon,
            start_lat,
            end_lon,
            end_lat,
            npoints - 2,
        )

        coordinates = [
            (start_lon, start_lat),
            *intermediate,
            (end_lon, end_lat),
        ]

        longitude = xr.DataArray(
            [point[0] for point in coordinates],
            dims=(dimension,),
            name="longitude",
            attrs={
                "long_name": "transect longitude",
                "units": "degrees_east",
            },
        )

        latitude = xr.DataArray(
            [point[1] for point in coordinates],
            dims=(dimension,),
            name="latitude",
            attrs={
                "long_name": "transect latitude",
                "units": "degrees_north",
            },
        )

        transect = cls(
            longitude=longitude,
            latitude=latitude,
        )

        transect.attrs = {
            "start_longitude": start_lon,
            "start_latitude": start_lat,
            "end_longitude": end_lon,
            "end_latitude": end_lat,
            "length_km": length_m / 1_000.0,
            "mean_bearing_degrees": forward_azimuth,
            "sampling": sampling,
            "estimated_grid_spacing_m": (
                cls._estimate_grid_spacing(dataset)
            ),
        }

        return transect

    @classmethod
    def from_coordinates(
        cls,
        longitude,
        latitude,
        *,
        dimension: str = "s",
    ) -> "Transect":
        """Create a transect from longitude and latitude arrays."""
        longitude = cls._as_dataarray(
            longitude,
            dimension=dimension,
            name="longitude",
        )
        latitude = cls._as_dataarray(
            latitude,
            dimension=dimension,
            name="latitude",
        )

        transect = cls(
            longitude=longitude,
            latitude=latitude,
        )

        transect.attrs = {
            "length_km": float(transect.distance.isel(
                {dimension: -1}
            )),
        }

        return transect

    @staticmethod
    def _estimate_grid_spacing(dataset: xr.Dataset) -> float:
        """Estimate representative horizontal grid spacing in metres."""
        missing = [
            name
            for name in ("pm", "pn")
            if name not in dataset
        ]

        if missing:
            raise KeyError(
                "Automatic transect sampling requires the CROCO "
                f"grid metrics {missing}. Provide npoints explicitly "
                "when pm or pn is unavailable."
            )

        dx = 1.0 / dataset["pm"]
        dy = 1.0 / dataset["pn"]

        spacing = np.sqrt(dx * dy)
        spacing_m = float(spacing.median(skipna=True).compute())

        if not np.isfinite(spacing_m) or spacing_m <= 0:
            raise ValueError(
                "Could not determine a valid model grid spacing."
            )

        return spacing_m

    @staticmethod
    def _as_dataarray(
        values,
        *,
        dimension: str,
        name: str,
    ) -> xr.DataArray:
        """Convert one-dimensional values to a DataArray."""
        if isinstance(values, xr.DataArray):
            if values.ndim != 1:
                raise ValueError(
                    f"{name} must be one-dimensional."
                )
            return values.rename(name)

        values = np.asarray(values)

        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        return xr.DataArray(
            values,
            dims=(dimension,),
            name=name,
        )

    def _calculate_bearing(self) -> xr.DataArray:
        """Calculate local forward bearing along the transect."""
        dimension = self.longitude.dims[0]

        lon = self.longitude.values
        lat = self.latitude.values

        segment_bearing, _, _ = _WGS84.inv(
            lon[:-1],
            lat[:-1],
            lon[1:],
            lat[1:],
        )

        point_bearing = np.empty(self.longitude.size)
        point_bearing[:-1] = segment_bearing
        point_bearing[-1] = segment_bearing[-1]

        return xr.DataArray(
            point_bearing,
            dims=(dimension,),
            coords={
                dimension: self.longitude[dimension],
            },
            name="bearing",
            attrs={
                "long_name": "local transect bearing",
                "units": "degrees clockwise from north",
            },
        )

    def _calculate_tangent(
        self,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Calculate eastward and northward unit tangent vectors."""
        angle = np.deg2rad(self.bearing)

        tangent_east = np.sin(angle).rename("tangent_east")
        tangent_north = np.cos(angle).rename("tangent_north")

        tangent_east.attrs = {
            "long_name": "eastward transect tangent component",
            "units": "1",
        }
        tangent_north.attrs = {
            "long_name": "northward transect tangent component",
            "units": "1",
        }

        return tangent_east, tangent_north

    def _calculate_left_normal(
        self,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Calculate unit normal pointing left of the transect."""
        normal_east = (-self.tangent_north).rename(
            "normal_east"
        )
        normal_north = self.tangent_east.rename(
            "normal_north"
        )

        normal_east.attrs = {
            "long_name": "eastward left-normal component",
            "units": "1",
        }
        normal_north.attrs = {
            "long_name": "northward left-normal component",
            "units": "1",
        }

        return normal_east, normal_north

    def rotate_velocity(
        self,
        u_east: xr.DataArray,
        v_north: xr.DataArray,
    ) -> xr.Dataset:
        """Rotate geographic velocity into along- and across-track components.

        Across-track velocity is positive toward the left side of the
        transect when looking from its first point toward its final point.
        """
        tangent_east = self.tangent_east
        tangent_north = self.tangent_north
        
        if "s" in tangent_east.dims:
            tangent_east = tangent_east.rename({"s": "distance"})
        
        if "s" in tangent_north.dims:
            tangent_north = tangent_north.rename({"s": "distance"})
        
        # Ensure the coordinates exactly match the section coordinate.
        tangent_east = tangent_east.assign_coords(
            distance=u_east.distance
        )
        
        tangent_north = tangent_north.assign_coords(
            distance=u_east.distance
        )
        
        velocity_along = (
            u_east * tangent_east
            + v_north * tangent_north
        )
        
        velocity_across = (
            -u_east * tangent_north
            + v_north * tangent_east
        )

        velocity_along.attrs = {
            **u_east.attrs,
            "long_name": "along-track velocity",
            "long_name": "from first to last point of the transect",
        }
        velocity_across.attrs = {
            **u_east.attrs,
            "long_name": "across-track velocity",
            "positive": "left of transect",
        }

        
        return xr.Dataset(
            {
                "velocity_along": velocity_along,
                "velocity_across": velocity_across,
            }
        )

    def to_dataset(self) -> xr.Dataset:
        """Return the transect geometry as an xarray Dataset."""
        dataset = xr.Dataset(
            coords={
                "longitude": self.longitude,
                "latitude": self.latitude,
                "distance": self.distance,
            },
            data_vars={
                "bearing": self.bearing,
                "tangent_east": self.tangent_east,
                "tangent_north": self.tangent_north,
                "normal_east": self.normal_east,
                "normal_north": self.normal_north,
            },
        )

        if hasattr(self, "attrs"):
            dataset.attrs.update(self.attrs)

        return dataset

    def __repr__(self) -> str:
        """Return a compact description of the transect."""
        dimension = self.longitude.dims[0]
        length = float(self.distance.isel({dimension: -1}))

        return (
            f"Transect(npoints={self.longitude.size}, "
            f"length={length:.2f} km, "
            f"start=({float(self.longitude[0]):.4f}, "
            f"{float(self.latitude[0]):.4f}), "
            f"end=({float(self.longitude[-1]):.4f}, "
            f"{float(self.latitude[-1]):.4f}))"
        )


def _build_linear_weights(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build reusable linear interpolation weights using a Delaunay
    triangulation.

    Parameters
    ----------
    source_points
        Array with shape (n_source, 2), containing longitude and latitude.
    target_points
        Array with shape (n_target, 2), containing the transect points.

    Returns
    -------
    vertices
        Indices of the three source vertices surrounding each target point.
    weights
        Barycentric weights for those vertices.
    """
    triangulation = Delaunay(source_points)
    simplex = triangulation.find_simplex(target_points)

    n_target = target_points.shape[0]

    vertices = np.full((n_target, 3), -1, dtype=np.int64)
    weights = np.full((n_target, 3), np.nan, dtype=float)

    inside = simplex >= 0

    if not np.any(inside):
        return vertices, weights

    simplex_inside = simplex[inside]

    vertices[inside] = triangulation.simplices[simplex_inside]

    transform = triangulation.transform[simplex_inside, :2]
    offset = triangulation.transform[simplex_inside, 2]

    delta = target_points[inside] - offset

    barycentric = np.einsum(
        "nij,nj->ni",
        transform,
        delta,
    )

    weights[inside, :2] = barycentric
    weights[inside, 2] = 1.0 - barycentric.sum(axis=1)

    return vertices, weights


def _apply_linear_weights(
    values: np.ndarray,
    vertices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Apply previously calculated interpolation weights.
    """
    output = np.full(vertices.shape[0], np.nan, dtype=float)

    inside = vertices[:, 0] >= 0

    if not np.any(inside):
        return output

    vertex_values = values[vertices[inside]]

    valid = np.all(np.isfinite(vertex_values), axis=1)

    interpolated = np.full(vertex_values.shape[0], np.nan, dtype=float)

    interpolated[valid] = np.einsum(
        "ni,ni->n",
        vertex_values[valid],
        weights[inside][valid],
    )

    output[inside] = interpolated

    return output

def interpolate_section(
    field: xr.DataArray,
    track,
    *,
    longitude: xr.DataArray,
    latitude: xr.DataArray,
    padding: float = 0.1,
) -> xr.DataArray:
    """
    Interpolate a CROCO field horizontally along a transect.

    The Delaunay triangulation is calculated only once and reused for all
    non-horizontal dimensions, including vertical levels.

    Parameters
    ----------
    field
        CROCO DataArray. Its final two dimensions must correspond to the
        horizontal grid represented by longitude and latitude.
    track
        Transect object with longitude, latitude, and distance arrays.
    longitude, latitude
        Two-dimensional horizontal grid coordinates.
    padding
        Geographic padding, in degrees, around the transect bounding box.
        Only source-grid points within this box are triangulated.

    Returns
    -------
    xarray.DataArray
        Field interpolated along the transect.
    """
    if longitude.ndim != 2 or latitude.ndim != 2:
        raise ValueError("longitude and latitude must be two-dimensional")

    if longitude.dims != latitude.dims:
        raise ValueError(
            "longitude and latitude must have identical dimensions"
        )

    horizontal_dims = longitude.dims

    for dim in horizontal_dims:
        if dim not in field.dims:
            raise ValueError(
                f"Horizontal dimension {dim!r} is missing from field"
            )

    track_lon = np.asarray(track.longitude, dtype=float)
    track_lat = np.asarray(track.latitude, dtype=float)
    track_distance = np.asarray(track.distance, dtype=float)

    if not (
        track_lon.shape == track_lat.shape == track_distance.shape
    ):
        raise ValueError(
            "track longitude, latitude, and distance must have the same shape"
        )

    lon_min = np.nanmin(track_lon) - padding
    lon_max = np.nanmax(track_lon) + padding
    lat_min = np.nanmin(track_lat) - padding
    lat_max = np.nanmax(track_lat) + padding

    lon_values = np.asarray(longitude)
    lat_values = np.asarray(latitude)

    source_mask = (
        np.isfinite(lon_values)
        & np.isfinite(lat_values)
        & (lon_values >= lon_min)
        & (lon_values <= lon_max)
        & (lat_values >= lat_min)
        & (lat_values <= lat_max)
    )

    if np.count_nonzero(source_mask) < 3:
        raise ValueError(
            "Too few source-grid points near the transect. "
            "Increase padding."
        )

    source_points = np.column_stack(
        (
            lon_values[source_mask],
            lat_values[source_mask],
        )
    )

    target_points = np.column_stack(
        (
            track_lon,
            track_lat,
        )
    )

    vertices, weights = _build_linear_weights(
        source_points,
        target_points,
    )

    # Put all non-horizontal dimensions first and the two horizontal
    # dimensions last.
    extra_dims = [
        dim for dim in field.dims
        if dim not in horizontal_dims
    ]

    transposed = field.transpose(
        *extra_dims,
        *horizontal_dims,
    )

    field_values = np.asarray(transposed)

    extra_shape = field_values.shape[:-2]
    horizontal_shape = field_values.shape[-2:]

    if horizontal_shape != longitude.shape:
        raise ValueError(
            "The field horizontal shape does not match longitude/latitude: "
            f"{horizontal_shape} versus {longitude.shape}"
        )

    # Combine all leading dimensions so the same weights can be applied
    # efficiently to every level/time/member.
    flattened = field_values.reshape(
        (-1,) + horizontal_shape
    )

    section_values = np.empty(
        (flattened.shape[0], track_lon.size),
        dtype=float,
    )

    for index, horizontal_slice in enumerate(flattened):
        nearby_values = horizontal_slice[source_mask]

        section_values[index] = _apply_linear_weights(
            nearby_values,
            vertices,
            weights,
        )

    section_values = section_values.reshape(
        extra_shape + (track_lon.size,)
    )

    coords = {
        dim: transposed.coords[dim]
        for dim in extra_dims
        if dim in transposed.coords
    }

    coords.update(
        {
            "distance": ("distance", track_distance),
            "longitude": ("distance", track_lon),
            "latitude": ("distance", track_lat),
        }
    )

    result = xr.DataArray(
        section_values,
        dims=extra_dims + ["distance"],
        coords=coords,
        name=field.name,
        attrs=field.attrs.copy(),
    )

    result["distance"].attrs.update(
        {
            "long_name": "distance along transect",
            "units": getattr(track.distance, "attrs", {}).get(
                "units",
                "km",
            ),
        }
    )

    result["longitude"].attrs["units"] = "degrees_east"
    result["latitude"].attrs["units"] = "degrees_north"

    return result