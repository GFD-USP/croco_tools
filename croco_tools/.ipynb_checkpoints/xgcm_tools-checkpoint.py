"""xgcm helpers for CROCO C-grid output.

This module focuses on horizontal grid-aware operations.  It deliberately
keeps the xgcm.Grid object separate from the xarray.Dataset and avoids
computing vertical coordinates unless they are needed elsewhere.
"""

from __future__ import annotations

from typing import Iterable, Literal

import xarray as xr
from xgcm import Grid


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
        boundary={"X": "extend", "Y": "extend"},
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
        boundary={"X": "extend", "Y": "extend"},
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


def relative_vorticity(
    ds: xr.Dataset,
    xgrid: Grid,
    *,
    u: str = "u",
    v: str = "v",
    normalized: bool = False,
    grid: GridLocation = "native",
    mask: bool = True,
    min_abs_f: float = 1.0e-10,
) -> xr.DataArray:
    r"""Compute vertical relative vorticity on a CROCO C-grid.

    The native psi-point discretization is circulation divided by cell area,

    .. math::

       \zeta = [\Delta_X(v\,dy) - \Delta_Y(u\,dx)] / A_\psi.

    Parameters
    ----------
    normalized
        If True, return ``zeta / f``.
    grid
        Output location: ``"native"`` (psi), ``"rho"``, ``"psi"``,
        ``"u"``, or ``"v"``.
    min_abs_f
        Values where ``abs(f)`` is not greater than this threshold are masked
        when normalization is requested.
    """
    for name in (u, v, "dx_u", "dy_v", "area_psi"):
        if name not in ds:
            raise KeyError(f"Required variable {name!r} is missing.")

    circulation_x = xgrid.diff(ds[v] * ds.dy_v, "X")
    circulation_y = xgrid.diff(ds[u] * ds.dx_u, "Y")
    field = (circulation_x - circulation_y) / ds.area_psi

    target = _target_location(grid, native="psi")
    field = _move_to_grid(field, xgrid, source="psi", target=target)

    if normalized:
        f_target = _coriolis_at_grid(ds, xgrid, target)
        field = _normalize_by_coriolis(
            field, f_target, signed=True, min_abs_f=min_abs_f
        )

    if mask:
        field = field.where(_mask_at_grid(ds, xgrid, target))

    field.name = "normalized_vorticity" if normalized else "vorticity"
    field.attrs.update(
        long_name=(
            "relative vorticity normalized by Coriolis parameter"
            if normalized
            else "vertical component of relative vorticity"
        ),
        standard_name="ocean_relative_vorticity",
        units="1" if normalized else "s-1",
        normalization="f" if normalized else "none",
        grid_location=target,
    )
    return field


def horizontal_divergence(
    ds: xr.Dataset,
    xgrid: Grid,
    *,
    u: str = "u",
    v: str = "v",
    normalized: bool = False,
    grid: GridLocation = "native",
    mask: bool = True,
    min_abs_f: float = 1.0e-10,
) -> xr.DataArray:
    r"""Compute horizontal divergence on a CROCO C-grid.

    The native rho-point discretization is the net volume flux per unit area,

    .. math::

       \delta = [\Delta_X(u\,dy) + \Delta_Y(v\,dx)] / A_\rho.

    If ``normalized=True``, the result is ``delta / abs(f)``.
    """
    for name in (u, v, "dy_u", "dx_v", "area_rho"):
        if name not in ds:
            raise KeyError(f"Required variable {name!r} is missing.")

    flux_x = xgrid.diff(ds[u] * ds.dy_u, "X")
    flux_y = xgrid.diff(ds[v] * ds.dx_v, "Y")
    field = (flux_x + flux_y) / ds.area_rho

    target = _target_location(grid, native="rho")
    field = _move_to_grid(field, xgrid, source="rho", target=target)

    if normalized:
        f_target = _coriolis_at_grid(ds, xgrid, target)
        field = _normalize_by_coriolis(
            field, f_target, signed=False, min_abs_f=min_abs_f
        )

    if mask:
        field = field.where(_mask_at_grid(ds, xgrid, target))

    field.name = "normalized_divergence" if normalized else "divergence"
    field.attrs.update(
        long_name=(
            "horizontal divergence normalized by absolute Coriolis parameter"
            if normalized
            else "horizontal divergence"
        ),
        standard_name="divergence_of_sea_water_velocity",
        units="1" if normalized else "s-1",
        normalization="abs(f)" if normalized else "none",
        grid_location=target,
    )
    return field


def strain_rate(
    ds: xr.Dataset,
    xgrid: Grid,
    *,
    u: str = "u",
    v: str = "v",
    normalized: bool = False,
    grid: GridLocation = "native",
    mask: bool = True,
    min_abs_f: float = 1.0e-10,
) -> xr.DataArray:
    r"""Compute total horizontal strain rate on a CROCO C-grid.

    The scalar strain rate is

    .. math::

       \alpha = \sqrt{S_n^2 + S_s^2},

    where ``S_n`` is computed natively at rho points and ``S_s`` at psi
    points.  The shear component is interpolated to rho points before the
    magnitude is formed, so the native output location is rho.

    If ``normalized=True``, the result is ``alpha / abs(f)``.
    """
    for name in (
        u, v, "dy_u", "dx_v", "dx_u", "dy_v", "area_rho", "area_psi"
    ):
        if name not in ds:
            raise KeyError(f"Required variable {name!r} is missing.")

    du_dx = xgrid.diff(ds[u] * ds.dy_u, "X") / ds.area_rho
    dv_dy = xgrid.diff(ds[v] * ds.dx_v, "Y") / ds.area_rho
    normal_rho = du_dx - dv_dy

    dv_dx = xgrid.diff(ds[v] * ds.dy_v, "X") / ds.area_psi
    du_dy = xgrid.diff(ds[u] * ds.dx_u, "Y") / ds.area_psi
    shear_psi = dv_dx + du_dy
    shear_rho = _move_to_grid(
        shear_psi, xgrid, source="psi", target="rho"
    )

    field = xr.apply_ufunc(
        lambda sn, ss: (sn * sn + ss * ss) ** 0.5,
        normal_rho,
        shear_rho,
        dask="allowed",
    )

    target = _target_location(grid, native="rho")
    field = _move_to_grid(field, xgrid, source="rho", target=target)

    if normalized:
        f_target = _coriolis_at_grid(ds, xgrid, target)
        field = _normalize_by_coriolis(
            field, f_target, signed=False, min_abs_f=min_abs_f
        )

    if mask:
        field = field.where(_mask_at_grid(ds, xgrid, target))

    field.name = "normalized_strain_rate" if normalized else "strain_rate"
    field.attrs.update(
        long_name=(
            "total horizontal strain rate normalized by absolute Coriolis parameter"
            if normalized
            else "total horizontal strain rate"
        ),
        units="1" if normalized else "s-1",
        normalization="abs(f)" if normalized else "none",
        grid_location=target,
    )
    return field


def vorticity_at_rho(
    ds: xr.Dataset,
    xgrid: Grid,
    *,
    u: str = "u",
    v: str = "v",
    mask: bool = True,
) -> xr.DataArray:
    """Backward-compatible wrapper returning unnormalized rho-point vorticity."""
    return relative_vorticity(
        ds,
        xgrid,
        u=u,
        v=v,
        normalized=False,
        grid="rho",
        mask=mask,
    )
