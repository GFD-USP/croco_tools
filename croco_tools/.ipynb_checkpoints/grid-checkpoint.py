"""C-grid interpolation and vector rotation."""

from __future__ import annotations

import numpy as np
import xarray as xr


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


def _rename_dimension(
    array: xr.DataArray,
    old_name: str,
    new_name: str,
) -> xr.DataArray:
    return array.rename({old_name: new_name})

def rho_to_u(
    array: xr.DataArray,
    *,
    x_dimension: str = "xi_rho",
) -> xr.DataArray:
    """Average rho-point values to u points."""
    _require_dimension(array, x_dimension)

    west = array.isel({x_dimension: slice(None, -1)})
    east = array.isel({x_dimension: slice(1, None)})

    result = 0.5 * (west + east)
    result = _rename_dimension(
        result,
        x_dimension,
        "xi_u",
    )

    return result


def rho_to_v(
    array: xr.DataArray,
    *,
    y_dimension: str = "eta_rho",
) -> xr.DataArray:
    """Average rho-point values to v points."""
    _require_dimension(array, y_dimension)

    south = array.isel({y_dimension: slice(None, -1)})
    north = array.isel({y_dimension: slice(1, None)})

    result = 0.5 * (south + north)
    result = _rename_dimension(
        result,
        y_dimension,
        "eta_v",
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

    southwest = array.isel(
        {
            y_dimension: slice(None, -1),
            x_dimension: slice(None, -1),
        }
    )
    southeast = array.isel(
        {
            y_dimension: slice(None, -1),
            x_dimension: slice(1, None),
        }
    )
    northwest = array.isel(
        {
            y_dimension: slice(1, None),
            x_dimension: slice(None, -1),
        }
    )
    northeast = array.isel(
        {
            y_dimension: slice(1, None),
            x_dimension: slice(1, None),
        }
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
    """Average psi-point values to rho points.

    The interior uses four-point averages. Boundary values are
    extended from the nearest interior rho point.
    """
    _require_dimension(array, x_dimension)
    _require_dimension(array, y_dimension)

    southwest = array.isel(
        {
            y_dimension: slice(None, -1),
            x_dimension: slice(None, -1),
        }
    )
    southeast = array.isel(
        {
            y_dimension: slice(None, -1),
            x_dimension: slice(1, None),
        }
    )
    northwest = array.isel(
        {
            y_dimension: slice(1, None),
            x_dimension: slice(None, -1),
        }
    )
    northeast = array.isel(
        {
            y_dimension: slice(1, None),
            x_dimension: slice(1, None),
        }
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

    return result


def rotate_velocity(
    u_velocity: xr.DataArray,
    v_velocity: xr.DataArray,
    angle: xr.DataArray,
    *,
    angle_units: str = "radians",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate grid-aligned velocity to eastward and northward components.

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
