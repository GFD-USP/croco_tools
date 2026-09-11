"""Vertical coordinates and interpolation."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

_VERTICAL_ALIASES = {
    "s_rho": ("s_rho", "sc_r"),
    "s_w": ("s_w", "sc_w"),
    "Cs_r": ("Cs_r", "Cs_rho"),
    "Cs_w": ("Cs_w",),
    "theta_s": ("theta_s", "THETA_S"),
    "theta_b": ("theta_b", "THETA_B"),
    "hc": ("hc", "Hc", "HC"),
}


def _get_scalar(
    dataset: xr.Dataset,
    names: tuple[str, ...],
    supplied_value: float | None = None,
) -> float | None:
    """Return a scalar from an argument, variable, coordinate, or attribute."""

    if supplied_value is not None:
        return float(supplied_value)

    for name in names:
        if name in dataset.variables:
            value = dataset[name]

            if value.size != 1:
                raise ValueError(
                    f"Expected {name!r} to be scalar, but its shape is "
                    f"{value.shape}."
                )

            return float(value.values.squeeze())

    for name in names:
        if name in dataset.attrs:
            return float(dataset.attrs[name])

    return None


def _find_variable(
    dataset: xr.Dataset,
    names: tuple[str, ...],
) -> xr.DataArray | None:
    """Return the first matching variable or coordinate."""

    for name in names:
        if name in dataset.variables:
            return dataset[name]

    return None


def _sigma_coordinates(
    n_rho: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct CROCO sigma coordinates at rho and w points."""

    if n_rho < 1:
        raise ValueError("The vertical grid must contain at least one rho level.")

    s_rho = (
        np.arange(1, n_rho + 1, dtype=np.float64)
        - n_rho
        - 0.5
    ) / n_rho

    s_w = (
        np.arange(0, n_rho + 1, dtype=np.float64)
        - n_rho
    ) / n_rho

    return s_rho, s_w


def _stretching_curve(
    sigma: np.ndarray,
    theta_s: float,
    theta_b: float,
) -> np.ndarray:
    """Return the CROCO vertical stretching curve."""

    if theta_s > 0.0:
        csrf = (
            1.0 - np.cosh(theta_s * sigma)
        ) / (
            np.cosh(theta_s) - 1.0
        )
    else:
        csrf = -(sigma**2)

    if theta_b > 0.0:
        return (
            np.exp(theta_b * csrf) - 1.0
        ) / (
            1.0 - np.exp(-theta_b)
        )

    return csrf


def ensure_vertical_coordinates(
    dataset: xr.Dataset,
    *,
    theta_s: float | None = None,
    theta_b: float | None = None,
    hc: float | None = None,
) -> xr.Dataset:
    """
    Ensure that a CROCO dataset contains its vertical-grid coordinates.

    Existing coordinates are preserved. Missing ``s_rho``, ``s_w``,
    ``Cs_r``, and ``Cs_w`` are reconstructed from the vertical dimensions
    and the stretching parameters.

    Parameters
    ----------
    dataset
        CROCO dataset.
    theta_s, theta_b, hc
        Optional vertical-grid parameters. Explicit values take precedence
        over dataset variables and global attributes.

    Returns
    -------
    xarray.Dataset
        A shallow copy containing the required vertical coordinates.

    Raises
    ------
    KeyError
        If the stretching parameters required for reconstruction cannot be
        found.
    ValueError
        If the vertical dimensions are inconsistent.
    """

    ds = dataset.copy(deep=False)

    theta_s_value = _get_scalar(
        ds,
        _VERTICAL_ALIASES["theta_s"],
        supplied_value=theta_s,
    )
    theta_b_value = _get_scalar(
        ds,
        _VERTICAL_ALIASES["theta_b"],
        supplied_value=theta_b,
    )
    hc_value = _get_scalar(
        ds,
        _VERTICAL_ALIASES["hc"],
        supplied_value=hc,
    )

    s_rho = _find_variable(ds, _VERTICAL_ALIASES["s_rho"])
    s_w = _find_variable(ds, _VERTICAL_ALIASES["s_w"])
    cs_r = _find_variable(ds, _VERTICAL_ALIASES["Cs_r"])
    cs_w = _find_variable(ds, _VERTICAL_ALIASES["Cs_w"])

    if "s_rho" not in ds.sizes:
        raise KeyError(
            "The dataset has no 's_rho' dimension, so its CROCO vertical "
            "grid cannot be reconstructed."
        )

    n_rho = ds.sizes["s_rho"]

    if "s_w" in ds.sizes:
        n_w = ds.sizes["s_w"]

        if n_w != n_rho + 1:
            raise ValueError(
                "Inconsistent CROCO vertical dimensions: expected "
                f"len(s_w)={n_rho + 1}, but found {n_w}."
            )
    else:
        n_w = n_rho + 1

    generated_s_rho, generated_s_w = _sigma_coordinates(n_rho)

    if s_rho is None:
        ds = ds.assign_coords(
            s_rho=("s_rho", generated_s_rho)
        )
        s_rho = ds["s_rho"]
    elif s_rho.name != "s_rho":
        ds = ds.assign_coords(
            s_rho=("s_rho", np.asarray(s_rho.values))
        )
        s_rho = ds["s_rho"]
    elif "s_rho" not in ds.coords:
        ds = ds.set_coords("s_rho")
        s_rho = ds["s_rho"]

    if s_w is None:
        ds = ds.assign_coords(
            s_w=("s_w", generated_s_w)
        )
        s_w = ds["s_w"]
    elif s_w.name != "s_w":
        ds = ds.assign_coords(
            s_w=("s_w", np.asarray(s_w.values))
        )
        s_w = ds["s_w"]
    elif "s_w" not in ds.coords:
        ds = ds.set_coords("s_w")
        s_w = ds["s_w"]

    needs_stretching = cs_r is None or cs_w is None

    if needs_stretching:
        missing_parameters = []

        if theta_s_value is None:
            missing_parameters.append("theta_s")

        if theta_b_value is None:
            missing_parameters.append("theta_b")

        if missing_parameters:
            available = sorted(
                name
                for name in ds.variables
                if name.lower().startswith(
                    ("cs", "sc", "s_", "theta", "hc")
                )
            )

            raise KeyError(
                "Cannot reconstruct the CROCO stretching curves because "
                f"{', '.join(missing_parameters)} is missing. Pass these "
                "values explicitly or store them as variables or global "
                f"attributes. Available vertical metadata: {available}"
            )

    if cs_r is None:
        cs_r_values = _stretching_curve(
            np.asarray(s_rho.values, dtype=np.float64),
            theta_s_value,
            theta_b_value,
        )

        ds["Cs_r"] = xr.DataArray(
            cs_r_values,
            dims=("s_rho",),
            coords={"s_rho": ds["s_rho"]},
            attrs={
                "long_name": "vertical stretching curve at rho points",
            },
        )
    elif cs_r.name != "Cs_r":
        ds["Cs_r"] = xr.DataArray(
            np.asarray(cs_r.values),
            dims=("s_rho",),
            coords={"s_rho": ds["s_rho"]},
            attrs=cs_r.attrs,
        )

    if cs_w is None:
        cs_w_values = _stretching_curve(
            np.asarray(s_w.values, dtype=np.float64),
            theta_s_value,
            theta_b_value,
        )

        ds["Cs_w"] = xr.DataArray(
            cs_w_values,
            dims=("s_w",),
            coords={"s_w": ds["s_w"]},
            attrs={
                "long_name": "vertical stretching curve at w points",
            },
        )

    if theta_s_value is not None and "theta_s" not in ds:
        ds["theta_s"] = xr.DataArray(theta_s_value)

    if theta_b_value is not None and "theta_b" not in ds:
        ds["theta_b"] = xr.DataArray(theta_b_value)

    if hc_value is not None and "hc" not in ds:
        ds["hc"] = xr.DataArray(hc_value)

    return ds


def _infer_vtransform(
    dataset: xr.Dataset,
) -> int:
    if "Vtransform" in dataset.variables:
        return int(dataset["Vtransform"].values)

    if "Vtransform" in dataset.attrs:
        return int(dataset.attrs["Vtransform"])

    cpp_options = str(
        dataset.attrs.get(
            "CPP-options",
            dataset.attrs.get("CPPS", ""),
        )
    )

    if "NEW_S_COORD" in cpp_options:
        return 2

    return 1


def compute_depths(
    dataset: xr.Dataset,
    zeta: xr.DataArray | None = None,
    vtransform: int = 2,
    *,
    theta_s: float | None = None,
    theta_b: float | None = None,
    hc: float | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute physical depths at CROCO rho and w points.
    """

    dataset = ensure_vertical_coordinates(
        dataset,
        theta_s=theta_s,
        theta_b=theta_b,
        hc=hc,
    )
    
    bathymetry = dataset["h"]

    if zeta is None:
        surface_height = dataset["zeta"]
    else:
        surface_height = zeta

    critical_depth = _get_scalar(
        dataset,
        ("hc",),
    )

    if vtransform is None:
        vtransform = _infer_vtransform(dataset)

    if "sc_r" in dataset:
        sigma_rho = dataset["sc_r"]
    else:
        sigma_rho = dataset["s_rho"]

    if "sc_w" in dataset:
        sigma_w = dataset["sc_w"]
    else:
        sigma_w = dataset["s_w"]


    if "Cs_r" in dataset:
        stretching_rho = dataset["Cs_r"]
    elif "Cs_rho" in dataset:
        stretching_rho = dataset["Cs_rho"]
    else:
        message = (
            "Could not find the rho-point stretching curve. "
            "Expected either 'Cs_r' or 'Cs_rho'."
        )
        raise KeyError(message)
    
    if "Cs_w" in dataset:
        stretching_w = dataset["Cs_w"]
    else:
        message = (
            "Could not find the w-point stretching curve. "
            "Expected 'Cs_w'."
        )
        raise KeyError(message)
    

    if vtransform == 1:
        z0_rho = (
            critical_depth
            * (sigma_rho - stretching_rho)
            + stretching_rho
            * bathymetry
        )
        z0_w = (
            critical_depth
            * (sigma_w - stretching_w)
            + stretching_w
            * bathymetry
        )

        z_rho = (
            z0_rho
            + surface_height
            * (1.0 + z0_rho / bathymetry)
        )
        z_w = (
            z0_w
            + surface_height
            * (1.0 + z0_w / bathymetry)
        )

    elif vtransform == 2:
        z0_rho = (
            critical_depth * sigma_rho
            + stretching_rho * bathymetry
        ) / (
            critical_depth + bathymetry
        )

        z0_w = (
            critical_depth * sigma_w
            + stretching_w * bathymetry
        ) / (
            critical_depth + bathymetry
        )

        z_rho = (
            surface_height
            + (surface_height + bathymetry)
            * z0_rho
        )
        z_w = (
            surface_height
            + (surface_height + bathymetry)
            * z0_w
        )

    else:
        message = "vtransform must be either 1 or 2."
        raise ValueError(message)

    z_rho = z_rho.rename("z_rho")
    z_w = z_w.rename("z_w")

    z_rho.attrs.update(
        {
            "long_name": "depth at rho points",
            "units": "m",
            "positive": "up",
        }
    )
    z_w.attrs.update(
        {
            "long_name": "depth at w points",
            "units": "m",
            "positive": "up",
        }
    )

    return z_rho, z_w


def _interpolate_column(
    values: np.ndarray,
    depths: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    valid = (
        np.isfinite(values)
        & np.isfinite(depths)
    )

    if valid.sum() < 2:
        return np.full(
            targets.shape,
            np.nan,
            dtype=float,
        )

    valid_depths = depths[valid]
    valid_values = values[valid]

    order = np.argsort(valid_depths)

    valid_depths = valid_depths[order]
    valid_values = valid_values[order]

    result = np.interp(
        targets,
        valid_depths,
        valid_values,
    )

    outside = (
        (targets < valid_depths[0])
        | (targets > valid_depths[-1])
    )
    result[outside] = np.nan

    return result


def interpolate_to_depth(
    values: xr.DataArray,
    z_rho: xr.DataArray,
    depths: float | list[float] | np.ndarray,
    *,
    vertical_dimension: str = "s_rho",
    target_dimension: str = "depth",
) -> xr.DataArray:
    """Interpolate a rho-level field to fixed physical depths.

    Target depths follow CROCO's sign convention and should therefore
    be negative below sea level.
    """
    target_values = np.atleast_1d(
        np.asarray(
            depths,
            dtype=float,
        )
    )

    target_array = xr.DataArray(
        target_values,
        dims=(target_dimension,),
        coords={
            target_dimension: target_values,
        },
    )

    result = xr.apply_ufunc(
        _interpolate_column,
        values,
        z_rho,
        target_array,
        input_core_dims=[
            [vertical_dimension],
            [vertical_dimension],
            [target_dimension],
        ],
        output_core_dims=[
            [target_dimension],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {
                target_dimension: target_values.size,
            }
        },
    )

    result = result.assign_coords(
        {
            target_dimension: target_values,
        }
    )

    result.attrs.update(values.attrs)
    result.attrs["interpolation"] = (
        "linear in physical depth"
    )

    return result
