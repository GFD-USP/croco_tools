"""Horizontal kinematic diagnostics for CROCO C-grid output."""

from __future__ import annotations

import xarray as xr
from xgcm import Grid

from .grid import (
    GridLocation,
    _coriolis_at_grid,
    _mask_at_grid,
    _move_to_grid,
    _normalize_by_coriolis,
    _target_location,
)

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


