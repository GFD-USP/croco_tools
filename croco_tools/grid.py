"""C-grid interpolation and vector rotation."""

from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Iterable, Literal
from xgcm import Grid


def _require_dimension(
    array: xr.DataArray,
    dimension: str,
) -> None:
    if dimension not in array.dims:
        message = (
            f"Expected dimension {dimension!r}; "
            f"found dimensions {array.dims}."
        )
        raise ValueError(message)


def _strip_horizontal_coordinates(
    array: xr.DataArray,
    dimensions: tuple[str, ...],
) -> xr.DataArray:
    """Remove coordinates tied to staggered horizontal dimensions."""
    output = array

    coordinate_names = [
        name
        for name, coordinate in output.coords.items()
        if any(
            dimension in coordinate.dims
            for dimension in dimensions
        )
        and name not in dimensions
    ]

    if coordinate_names:
        output = output.reset_coords(
            coordinate_names,
            drop=True,
        )

    for dimension in dimensions:
        if dimension in output.dims:
            output = output.assign_coords(
                {
                    dimension: np.arange(
                        output.sizes[dimension]
                    )
                }
            )

    return output


def rho_to_u(
    array: xr.DataArray,
    *,
    x_dimension: str = "xi_rho",
) -> xr.DataArray:
    """Average rho-point values to u points."""
    _require_dimension(array, x_dimension)

    clean = _strip_horizontal_coordinates(
        array,
        (x_dimension,),
    )

    west = clean.isel(
        {x_dimension: slice(None, -1)}
    )
    east = clean.isel(
        {x_dimension: slice(1, None)}
    )

    east = east.assign_coords(
        {
            x_dimension: west[x_dimension],
        }
    )

    result = 0.5 * (west + east)
    result = result.rename(
        {
            x_dimension: "xi_u",
        }
    )

    return result


def rho_to_v(
    array: xr.DataArray,
    *,
    y_dimension: str = "eta_rho",
) -> xr.DataArray:
    """Average rho-point values to v points."""
    _require_dimension(array, y_dimension)

    clean = _strip_horizontal_coordinates(
        array,
        (y_dimension,),
    )

    south = clean.isel(
        {y_dimension: slice(None, -1)}
    )
    north = clean.isel(
        {y_dimension: slice(1, None)}
    )

    north = north.assign_coords(
        {
            y_dimension: south[y_dimension],
        }
    )

    result = 0.5 * (south + north)
    result = result.rename(
        {
            y_dimension: "eta_v",
        }
    )

    return result


def rho_to_psi(
    array: xr.DataArray,
    *,
    x_dimension: str = "xi_rho",
    y_dimension: str = "eta_rho",
) -> xr.DataArray:
    """Average the four surrounding rho points to psi points."""
    _require_dimension(array, x_dimension)
    _require_dimension(array, y_dimension)

    clean = _strip_horizontal_coordinates(
        array,
        (
            x_dimension,
            y_dimension,
        ),
    )

    southwest = clean.isel(
        {
            y_dimension: slice(None, -1),
            x_dimension: slice(None, -1),
        }
    )
    southeast = clean.isel(
        {
            y_dimension: slice(None, -1),
            x_dimension: slice(1, None),
        }
    )
    northwest = clean.isel(
        {
            y_dimension: slice(1, None),
            x_dimension: slice(None, -1),
        }
    )
    northeast = clean.isel(
        {
            y_dimension: slice(1, None),
            x_dimension: slice(1, None),
        }
    )

    target_coordinates = {
        y_dimension: southwest[y_dimension],
        x_dimension: southwest[x_dimension],
    }

    southeast = southeast.assign_coords(
        target_coordinates
    )
    northwest = northwest.assign_coords(
        target_coordinates
    )
    northeast = northeast.assign_coords(
        target_coordinates
    )

    result = 0.25 * (
        southwest
        + southeast
        + northwest
        + northeast
    )

    result = result.rename(
        {
            x_dimension: "xi_psi",
            y_dimension: "eta_psi",
        }
    )

    return result


def u_to_rho(
    array: xr.DataArray,
    *,
    x_dimension: str = "xi_u",
) -> xr.DataArray:
    """Average u-point values to rho points.

    Boundary rho points are filled using the nearest u value.
    """
    _require_dimension(array, x_dimension)

    clean = _strip_horizontal_coordinates(
        array,
        (x_dimension,),
    )

    west = clean.isel(
        {x_dimension: slice(None, -1)}
    )
    east = clean.isel(
        {x_dimension: slice(1, None)}
    )

    east = east.assign_coords(
        {
            x_dimension: west[x_dimension],
        }
    )

    interior = 0.5 * (west + east)

    first = clean.isel(
        {x_dimension: slice(0, 1)}
    )
    last = clean.isel(
        {x_dimension: slice(-1, None)}
    )

    first = first.assign_coords(
        {
            x_dimension: [-1],
        }
    )
    interior = interior.assign_coords(
        {
            x_dimension: np.arange(
                interior.sizes[x_dimension]
            )
        }
    )
    last = last.assign_coords(
        {
            x_dimension: [
                interior.sizes[x_dimension]
            ],
        }
    )

    result = xr.concat(
        [
            first,
            interior,
            last,
        ],
        dim=x_dimension,
    )

    result = result.assign_coords(
        {
            x_dimension: np.arange(
                result.sizes[x_dimension]
            )
        }
    )

    return result.rename(
        {
            x_dimension: "xi_rho",
        }
    )


def v_to_rho(
    array: xr.DataArray,
    *,
    y_dimension: str = "eta_v",
) -> xr.DataArray:
    """Average v-point values to rho points.

    Boundary rho points are filled using the nearest v value.
    """
    _require_dimension(array, y_dimension)

    clean = _strip_horizontal_coordinates(
        array,
        (y_dimension,),
    )

    south = clean.isel(
        {y_dimension: slice(None, -1)}
    )
    north = clean.isel(
        {y_dimension: slice(1, None)}
    )

    north = north.assign_coords(
        {
            y_dimension: south[y_dimension],
        }
    )

    interior = 0.5 * (south + north)

    first = clean.isel(
        {y_dimension: slice(0, 1)}
    )
    last = clean.isel(
        {y_dimension: slice(-1, None)}
    )

    first = first.assign_coords(
        {
            y_dimension: [-1],
        }
    )
    interior = interior.assign_coords(
        {
            y_dimension: np.arange(
                interior.sizes[y_dimension]
            )
        }
    )
    last = last.assign_coords(
        {
            y_dimension: [
                interior.sizes[y_dimension]
            ],
        }
    )

    result = xr.concat(
        [
            first,
            interior,
            last,
        ],
        dim=y_dimension,
    )

    result = result.assign_coords(
        {
            y_dimension: np.arange(
                result.sizes[y_dimension]
            )
        }
    )

    return result.rename(
        {
            y_dimension: "eta_rho",
        }
    )


def psi_to_rho(
    array: xr.DataArray,
    *,
    x_dimension: str = "xi_psi",
    y_dimension: str = "eta_psi",
) -> xr.DataArray:
    """Average psi-point values to rho points."""
    _require_dimension(array, x_dimension)
    _require_dimension(array, y_dimension)

    clean = _strip_horizontal_coordinates(
        array,
        (
            x_dimension,
            y_dimension,
        ),
    )

    southwest = clean.isel(
        {
            y_dimension: slice(None, -1),
            x_dimension: slice(None, -1),
        }
    )
    southeast = clean.isel(
        {
            y_dimension: slice(None, -1),
            x_dimension: slice(1, None),
        }
    )
    northwest = clean.isel(
        {
            y_dimension: slice(1, None),
            x_dimension: slice(None, -1),
        }
    )
    northeast = clean.isel(
        {
            y_dimension: slice(1, None),
            x_dimension: slice(1, None),
        }
    )

    target_coordinates = {
        y_dimension: southwest[y_dimension],
        x_dimension: southwest[x_dimension],
    }

    southeast = southeast.assign_coords(
        target_coordinates
    )
    northwest = northwest.assign_coords(
        target_coordinates
    )
    northeast = northeast.assign_coords(
        target_coordinates
    )

    interior = 0.25 * (
        southwest
        + southeast
        + northwest
        + northeast
    )

    interior = interior.rename(
        {
            x_dimension: "xi_rho",
            y_dimension: "eta_rho",
        }
    )

    result = interior.pad(
        {
            "xi_rho": (1, 1),
            "eta_rho": (1, 1),
        },
        mode="edge",
    )

    result = result.assign_coords(
        {
            "xi_rho": np.arange(
                result.sizes["xi_rho"]
            ),
            "eta_rho": np.arange(
                result.sizes["eta_rho"]
            ),
        }
    )

    return result


def rotate_velocity(
    u_velocity: xr.DataArray,
    v_velocity: xr.DataArray,
    angle: xr.DataArray,
    *,
    angle_units: str = "radians",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate grid-aligned velocity to east and north.

    Velocity components are first interpolated to rho points.
    """
    if "xi_u" in u_velocity.dims:
        u_rho = u_to_rho(u_velocity)
    else:
        u_rho = u_velocity

    if "eta_v" in v_velocity.dims:
        v_rho = v_to_rho(v_velocity)
    else:
        v_rho = v_velocity

    u_rho = _strip_horizontal_coordinates(
        u_rho,
        (
            "eta_rho",
            "xi_rho",
        ),
    )
    v_rho = _strip_horizontal_coordinates(
        v_rho,
        (
            "eta_rho",
            "xi_rho",
        ),
    )
    clean_angle = _strip_horizontal_coordinates(
        angle,
        (
            "eta_rho",
            "xi_rho",
        ),
    )

    common_coordinates = {
        "eta_rho": np.arange(
            clean_angle.sizes["eta_rho"]
        ),
        "xi_rho": np.arange(
            clean_angle.sizes["xi_rho"]
        ),
    }

    u_rho = u_rho.assign_coords(
        common_coordinates
    )
    v_rho = v_rho.assign_coords(
        common_coordinates
    )
    clean_angle = clean_angle.assign_coords(
        common_coordinates
    )

    if angle_units == "degrees":
        theta = np.deg2rad(clean_angle)
    elif angle_units == "radians":
        theta = clean_angle
    else:
        message = (
            "angle_units must be either "
            "'degrees' or 'radians'."
        )
        raise ValueError(message)

    cosine = np.cos(theta)
    sine = np.sin(theta)

    eastward = (
        u_rho * cosine
        - v_rho * sine
    )
    northward = (
        u_rho * sine
        + v_rho * cosine
    )

    eastward = eastward.rename("u_east")
    northward = northward.rename("v_north")

    eastward.attrs.update(
        {
            "long_name": "eastward velocity",
            "units": u_velocity.attrs.get(
                "units",
                "m s-1",
            ),
        }
    )
    northward.attrs.update(
        {
            "long_name": "northward velocity",
            "units": v_velocity.attrs.get(
                "units",
                "m s-1",
            ),
        }
    )

    return eastward, northward


# -----------------------------------------------------------------------------
# xgcm grid infrastructure
# -----------------------------------------------------------------------------

_XGCM_COORDS = {
    "X": {"center": "xi_rho", "inner": "xi_u"},
    "Y": {"center": "eta_rho", "inner": "eta_v"},
}


def normalize_croco_coords(ds: xr.Dataset) -> xr.Dataset:
    """Normalize common CROCO/XIOS dimension and coordinate names.

    Parameters
    ----------
    ds
        CROCO history/average dataset.

    Returns
    -------
    xr.Dataset
        A shallowly transformed dataset using the conventional CROCO names
        ``xi_rho``, ``xi_u``, ``eta_rho``, and ``eta_v``.
    """
    rename: dict[str, str] = {}

    # XIOS time and horizontal dimensions.
    candidates = {
        "time_counter": "time",
        "x_rho": "xi_rho",
        "x_u": "xi_u",
        "y_rho": "eta_rho",
        "y_v": "eta_v",
        "x_v": "xi_rho",
        "y_u": "eta_rho",
        "x_w": "xi_rho",
        "y_w": "eta_rho",
        # Standard CROCO psi dimensions are collocated with u/v dimensions.
        "xi_psi": "xi_u",
        "eta_psi": "eta_v",
    }

    for old, new in candidates.items():
        if old in ds.dims and old != new:
            if new not in ds.dims:
                rename[old] = new
            elif ds.sizes[old] == ds.sizes[new]:
                # Redundant dimension names occasionally occur in XIOS output.
                rename[old] = new

    if rename:
        ds = ds.rename(rename)

    # Convert XIOS navigation variables to standard names.
    coord_rename: dict[str, str] = {}
    for name in tuple(ds.variables):
        new = name.replace("nav_lon", "lon").replace("nav_lat", "lat")
        if new != name and new not in ds.variables:
            coord_rename[name] = new
    if coord_rename:
        ds = ds.rename(coord_rename)

    horizontal_coords = [
        name
        for name in ds.variables
        if name.startswith(("lon_", "lat_"))
    ]
    if horizontal_coords:
        ds = ds.set_coords(horizontal_coords)

    required_dims = ("xi_rho", "xi_u", "eta_rho", "eta_v")
    missing = [dim for dim in required_dims if dim not in ds.dims]
    if missing:
        raise ValueError(
            "Dataset is not recognizable as a CROCO C-grid. "
            f"Missing dimensions: {missing}. Present dimensions: {tuple(ds.dims)}"
        )

    return ds


def merge_grid(
    ds: xr.Dataset,
    grd: xr.Dataset,
    variables: Iterable[str] = (
        "pm",
        "pn",
        "mask_rho",
        "lon_rho",
        "lat_rho",
        "lon_u",
        "lat_u",
        "lon_v",
        "lat_v",
        "lon_psi",
        "lat_psi",
        "angle",
        "f",
        "h",
    ),
) -> xr.Dataset:
    """Merge selected grid variables into an output dataset."""
    ds = normalize_croco_coords(ds)
    grd = normalize_croco_coords(grd)

    available = [name for name in variables if name in grd]
    return xr.merge([ds, grd[available]], compat="override", join="exact")


def add_horizontal_metrics(ds: xr.Dataset) -> xr.Dataset:
    """Add horizontal distances, areas, masks, and missing lon/lat positions."""
    ds = normalize_croco_coords(ds)

    for metric in ("pm", "pn"):
        if metric not in ds:
            raise KeyError(
                f"{metric!r} is required. Merge the CROCO grid file first."
            )

    # Build a temporary grid without metrics to generate staggered fields.
    bare_grid = Grid(
        ds,
        coords=_XGCM_COORDS,
        padding={"X": "extend", "Y": "extend"},
        autoparse_metadata=False,
    )

    ds = ds.assign(
        dx_rho=1.0 / ds.pm,
        dy_rho=1.0 / ds.pn,
    )

    ds = ds.assign(
        dx_u=bare_grid.interp(ds.dx_rho, "X"),
        dy_u=bare_grid.interp(ds.dy_rho, "X"),
        dx_v=bare_grid.interp(ds.dx_rho, "Y"),
        dy_v=bare_grid.interp(ds.dy_rho, "Y"),
    )
    ds = ds.assign(
        dx_psi=bare_grid.interp(ds.dx_v, "X"),
        dy_psi=bare_grid.interp(ds.dy_u, "Y"),
    )

    ds = ds.assign(
        area_rho=ds.dx_rho * ds.dy_rho,
        area_u=ds.dx_u * ds.dy_u,
        area_v=ds.dx_v * ds.dy_v,
        area_psi=ds.dx_psi * ds.dy_psi,
    )

    if "mask_rho" not in ds:
        ds["mask_rho"] = xr.ones_like(ds.pm)

    # A conservative wet mask: all four surrounding rho cells must be wet.
    mask_psi = bare_grid.interp(
        bare_grid.interp(ds.mask_rho.astype(float), "X"), "Y"
    )
    ds["mask_psi"] = mask_psi.where(mask_psi >= 0.999, 0.0)
    ds["mask_psi"] = xr.where(ds.mask_psi > 0, 1, 0)

    if "lon_rho" in ds and "lat_rho" in ds:
        missing_coords: dict[str, xr.DataArray] = {}
        if "lon_u" not in ds:
            missing_coords["lon_u"] = bare_grid.interp(ds.lon_rho, "X")
        if "lat_u" not in ds:
            missing_coords["lat_u"] = bare_grid.interp(ds.lat_rho, "X")
        if "lon_v" not in ds:
            missing_coords["lon_v"] = bare_grid.interp(ds.lon_rho, "Y")
        if "lat_v" not in ds:
            missing_coords["lat_v"] = bare_grid.interp(ds.lat_rho, "Y")
        if "lon_psi" not in ds:
            missing_coords["lon_psi"] = bare_grid.interp(
                bare_grid.interp(ds.lon_rho, "X"), "Y"
            )
        if "lat_psi" not in ds:
            missing_coords["lat_psi"] = bare_grid.interp(
                bare_grid.interp(ds.lat_rho, "X"), "Y"
            )
        if missing_coords:
            ds = ds.assign_coords(missing_coords)

    for name in (
        "dx_rho", "dy_rho", "dx_u", "dy_u", "dx_v", "dy_v",
        "dx_psi", "dy_psi"
    ):
        ds[name].attrs.update(units="m")

    for name in ("area_rho", "area_u", "area_v", "area_psi"):
        ds[name].attrs.update(units="m2")

    return ds


def build_xgcm_grid(ds: xr.Dataset) -> Grid:
    """Construct the horizontal xgcm grid for a prepared CROCO dataset."""
    required = (
        "dx_rho", "dy_rho", "dx_u", "dy_u",
        "dx_v", "dy_v", "dx_psi", "dy_psi",
        "area_rho", "area_u", "area_v", "area_psi",
    )
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(
            "Horizontal metrics have not been added. "
            f"Missing: {missing}. Call add_horizontal_metrics(ds)."
        )

    metrics = {
        ("X",): ["dx_rho", "dx_u", "dx_v", "dx_psi"],
        ("Y",): ["dy_rho", "dy_u", "dy_v", "dy_psi"],
        ("X", "Y"): ["area_rho", "area_u", "area_v", "area_psi"],
    }

    return Grid(
        ds,
        coords=_XGCM_COORDS,
        metrics=metrics,
        padding={"X": "extend", "Y": "extend"},
        autoparse_metadata=False,
    )


def prepare_xgcm(
    ds: xr.Dataset,
    grd: xr.Dataset | None = None,
) -> tuple[xr.Dataset, Grid]:
    """Normalize CROCO output, optionally merge its grid, and build xgcm."""
    ds = normalize_croco_coords(ds)
    if grd is not None:
        ds = merge_grid(ds, grd)
    ds = add_horizontal_metrics(ds)
    return ds, build_xgcm_grid(ds)



GridLocation = Literal["native", "rho", "psi", "u", "v"]


def _target_location(grid: GridLocation, native: str) -> str:
    """Resolve and validate an output-grid request."""
    allowed = {"native", "rho", "psi", "u", "v"}
    if grid not in allowed:
        raise ValueError(
            f"grid must be one of {sorted(allowed)}, got {grid!r}."
        )
    return native if grid == "native" else grid


def _move_to_grid(
    field: xr.DataArray,
    xgrid: Grid,
    *,
    source: str,
    target: str,
) -> xr.DataArray:
    """Interpolate a horizontal field between CROCO C-grid locations."""
    if source == target:
        return field

    operations = {
        ("rho", "u"): ("X",),
        ("rho", "v"): ("Y",),
        ("rho", "psi"): ("X", "Y"),
        ("psi", "u"): ("Y",),
        ("psi", "v"): ("X",),
        ("psi", "rho"): ("X", "Y"),
        ("u", "rho"): ("X",),
        ("u", "psi"): ("Y",),
        ("u", "v"): ("X", "Y"),
        ("v", "rho"): ("Y",),
        ("v", "psi"): ("X",),
        ("v", "u"): ("Y", "X"),
    }

    try:
        axes = operations[(source, target)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported grid interpolation: {source!r} -> {target!r}."
        ) from exc

    result = field
    for axis in axes:
        result = xgrid.interp(result, axis)
    return result


def _mask_at_grid(ds: xr.Dataset, xgrid: Grid, location: str) -> xr.DataArray:
    """Return a conservative wet mask at a requested C-grid location."""
    if "mask_rho" not in ds:
        return xr.ones_like(ds.pm, dtype=bool)

    mask = ds.mask_rho.astype(float)
    if location == "rho":
        return mask == 1
    if location == "u":
        return xgrid.interp(mask, "X") >= 0.999
    if location == "v":
        return xgrid.interp(mask, "Y") >= 0.999
    if location == "psi":
        if "mask_psi" in ds:
            return ds.mask_psi == 1
        return xgrid.interp(xgrid.interp(mask, "X"), "Y") >= 0.999
    raise ValueError(f"Unknown CROCO grid location {location!r}.")


def _coriolis_at_grid(
    ds: xr.Dataset,
    xgrid: Grid,
    location: str,
) -> xr.DataArray:
    """Interpolate the rho-point Coriolis parameter to a C-grid location."""
    if "f" not in ds:
        raise KeyError(
            "The Coriolis parameter 'f' is required for normalization. "
            "Merge the CROCO grid file first."
        )
    return _move_to_grid(ds.f, xgrid, source="rho", target=location)


def _normalize_by_coriolis(
    field: xr.DataArray,
    f: xr.DataArray,
    *,
    signed: bool,
    min_abs_f: float,
) -> xr.DataArray:
    """Normalize a diagnostic by f or |f| while masking small values."""
    valid_f = abs(f) > min_abs_f
    denominator = f if signed else abs(f)
    return field / denominator.where(valid_f)


