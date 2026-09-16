"""CROCO equation-of-state-of-seawater diagnostics.

This module is a direct Python/xarray translation of CROCO's ``rho_eos.F``
nonlinear equation of state of seawater.  The formulas, constants, and
parenthesization follow the Fortran source intentionally; do not replace
them with TEOS-10 or rearrange the polynomials when CROCO reproducibility
is required.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from xgcm import Grid


def _as_dtype(value: float, dtype: np.dtype) -> np.generic:
    """Cast a scalar constant to the model field's floating-point dtype."""
    return np.asarray(value, dtype=dtype)[()]


def _require(ds: xr.Dataset, *names: str) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"Required CROCO variables are missing: {missing}.")


def _eos_surface_terms(
    temperature: xr.DataArray,
    salinity: xr.DataArray,
    *,
    rho0: float,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return CROCO rho1, K0, K1, and K2 using Fortran operation order."""
    dtype = np.result_type(temperature.dtype, salinity.dtype)
    Tt = temperature.astype(dtype, copy=False)
    Ts = salinity.astype(dtype, copy=False).clip(min=_as_dtype(0.0, dtype))
    sqrtTs = np.sqrt(Ts)

    r00 = _as_dtype(999.842594, dtype)
    r01 = _as_dtype(6.793952e-2, dtype)
    r02 = _as_dtype(-9.095290e-3, dtype)
    r03 = _as_dtype(1.001685e-4, dtype)
    r04 = _as_dtype(-1.120083e-6, dtype)
    r05 = _as_dtype(6.536332e-9, dtype)
    r10 = _as_dtype(0.824493, dtype)
    r11 = _as_dtype(-4.08990e-3, dtype)
    r12 = _as_dtype(7.64380e-5, dtype)
    r13 = _as_dtype(-8.24670e-7, dtype)
    r14 = _as_dtype(5.38750e-9, dtype)
    rS0 = _as_dtype(-5.72466e-3, dtype)
    rS1 = _as_dtype(1.02270e-4, dtype)
    rS2 = _as_dtype(-1.65460e-6, dtype)
    r20 = _as_dtype(4.8314e-4, dtype)

    K01 = _as_dtype(209.8925, dtype)
    K02 = _as_dtype(-3.041638, dtype)
    K03 = _as_dtype(-1.852732e-3, dtype)
    K04 = _as_dtype(-1.361629e-5, dtype)
    K10 = _as_dtype(104.4077, dtype)
    K11 = _as_dtype(-6.500517, dtype)
    K12 = _as_dtype(0.1553190, dtype)
    K13 = _as_dtype(2.326469e-4, dtype)
    KS0 = _as_dtype(-5.587545, dtype)
    KS1 = _as_dtype(+0.7390729, dtype)
    KS2 = _as_dtype(-1.909078e-2, dtype)

    B00 = _as_dtype(0.4721788, dtype)
    B01 = _as_dtype(0.01028859, dtype)
    B02 = _as_dtype(-2.512549e-4, dtype)
    B03 = _as_dtype(-5.939910e-7, dtype)
    B10 = _as_dtype(-0.01571896, dtype)
    B11 = _as_dtype(-2.598241e-4, dtype)
    B12 = _as_dtype(7.267926e-6, dtype)
    BS1 = _as_dtype(2.042967e-3, dtype)

    E00 = _as_dtype(+1.045941e-5, dtype)
    E01 = _as_dtype(-5.782165e-10, dtype)
    E02 = _as_dtype(+1.296821e-7, dtype)
    E10 = _as_dtype(-2.595994e-7, dtype)
    E11 = _as_dtype(-1.248266e-9, dtype)
    E12 = _as_dtype(-3.508914e-9, dtype)

    dr00 = r00 - _as_dtype(rho0, dtype)

    rho1 = (
        dr00
        + Tt * (r01 + Tt * (r02 + Tt * (r03 + Tt * (r04 + Tt * r05))))
        + Ts
        * (
            r10
            + Tt * (r11 + Tt * (r12 + Tt * (r13 + Tt * r14)))
            + sqrtTs * (rS0 + Tt * (rS1 + Tt * rS2))
            + Ts * r20
        )
    )

    K0 = (
        Tt * (K01 + Tt * (K02 + Tt * (K03 + Tt * K04)))
        + Ts
        * (
            K10
            + Tt * (K11 + Tt * (K12 + Tt * K13))
            + sqrtTs * (KS0 + Tt * (KS1 + Tt * KS2))
        )
    )

    K1 = (
        B00
        + Tt * (B01 + Tt * (B02 + Tt * B03))
        + Ts * (B10 + Tt * (B11 + Tt * B12) + sqrtTs * BS1)
    )
    K2 = E00 + Tt * (E01 + Tt * E02) + Ts * (E10 + Tt * (E11 + Tt * E12))

    return rho1, K0, K1, K2


def density_anomaly(
    ds: xr.Dataset,
    *,
    temperature: str = "temp",
    salinity: str = "salt",
    z_r: str = "z_r",
    z_w: str = "z_w",
    rho0: float = 1025.0,
    split_eos: bool = False,
    qp2: float = 0.0,
    mask: bool = True,
    return_terms: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Compute CROCO's nonlinear density anomaly exactly as in ``rho_eos.F``.

    The returned ``rho`` is CROCO's pressure-gradient density anomaly, not
    generic in-situ density.  ``split_eos`` must match the CROCO CPP switch.
    When ``split_eos=True``, ``qp2`` must match CROCO's run-time scalar.
    """
    _require(ds, temperature, salinity, z_r, z_w)
    T = ds[temperature]
    S = ds[salinity]
    zr = ds[z_r]
    zw = ds[z_w]
    dtype = np.result_type(T.dtype, S.dtype, zr.dtype, zw.dtype)

    rho1, K0, K1, K2 = _eos_surface_terms(T, S, rho0=rho0)
    K00 = _as_dtype(19092.56, dtype)
    tenth = _as_dtype(0.1, dtype)
    one = _as_dtype(1.0, dtype)
    rho0_t = _as_dtype(rho0, dtype)

    vertical_dims = [d for d in zw.dims if d.startswith("s_")]
    if len(vertical_dims) != 1:
        raise ValueError("Could not identify the vertical dimension of z_w.")
    surface = zw.isel({vertical_dims[0]: -1})
    dpth = surface - zr

    terms: dict[str, xr.DataArray] = {"rho1": rho1, "K0": K0}

    if split_eos:
        Tduk = _as_dtype(3.8, dtype)
        Sduk = _as_dtype(34.5, dtype)
        sqrtSduk = np.sqrt(Sduk)
        K01 = _as_dtype(209.8925, dtype)
        K02 = _as_dtype(-3.041638, dtype)
        K03 = _as_dtype(-1.852732e-3, dtype)
        K04 = _as_dtype(-1.361629e-5, dtype)
        K10 = _as_dtype(104.4077, dtype)
        K11 = _as_dtype(-6.500517, dtype)
        K12 = _as_dtype(0.1553190, dtype)
        K13 = _as_dtype(2.326469e-4, dtype)
        KS0 = _as_dtype(-5.587545, dtype)
        KS1 = _as_dtype(+0.7390729, dtype)
        KS2 = _as_dtype(-1.909078e-2, dtype)
        K0_duk = (
            Tduk * (K01 + Tduk * (K02 + Tduk * (K03 + Tduk * K04)))
            + Sduk
            * (
                K10
                + Tduk * (K11 + Tduk * (K12 + Tduk * K13))
                + sqrtSduk * (KS0 + Tduk * (KS1 + Tduk * KS2))
            )
        )
        qp1 = (
            tenth
            * (rho0_t + rho1)
            * (K0_duk - K0)
            / ((K00 + K0) * (K00 + K0_duk))
        )
        rho = rho1 + qp1 * dpth * (one - _as_dtype(qp2, dtype) * dpth)
        terms["qp1"] = qp1
    else:
        cff = K00 - tenth * dpth
        cff1 = K0 + dpth * (K1 + K2 * dpth)
        rho = (
            rho1 * cff * (K00 + cff1) - tenth * dpth * rho0_t * cff1
        ) / (cff * (cff + cff1))
        terms.update(K1=K1, K2=K2)

    if mask and "mask_rho" in ds:
        rho1 = rho1 * ds.mask_rho
        rho = rho * ds.mask_rho
        terms["rho1"] = rho1
        for key in tuple(terms):
            if key not in {"rho1", "K0", "K1", "K2"}:
                terms[key] = terms[key] * ds.mask_rho

    rho.name = "rho"
    rho.attrs.update(
        long_name="CROCO nonlinear-EOS density anomaly",
        units="kg m-3",
        reference_density=rho0,
        split_eos=split_eos,
        qp2=qp2 if split_eos else "not used",
        equation_of_state="CROCO rho_eos.F nonlinear EOS",
        note="Not generic in-situ density; this is CROCO's pressure-gradient density anomaly.",
        grid_location="rho",
    )

    if not return_terms:
        return rho

    out = xr.Dataset({"rho": rho, **terms})
    out.attrs.update(equation_of_state="CROCO rho_eos.F nonlinear EOS")
    return out


def buoyancy_frequency(
    ds: xr.Dataset,
    *,
    temperature: str = "temp",
    salinity: str = "salt",
    z_r: str = "z_r",
    z_w: str = "z_w",
    rho0: float = 1025.0,
    gravity: float = 9.81,
    split_eos: bool = False,
    qp2: float = 0.0,
    mask: bool = True,
) -> xr.DataArray:
    """Compute CROCO's adiabatic Brunt-Väisälä frequency at interior W levels."""
    terms = density_anomaly(
        ds,
        temperature=temperature,
        salinity=salinity,
        z_r=z_r,
        z_w=z_w,
        rho0=rho0,
        split_eos=split_eos,
        qp2=qp2,
        mask=False,
        return_terms=True,
    )
    assert isinstance(terms, xr.Dataset)
    zr = ds[z_r]
    zw = ds[z_w]
    s_rho_dims = [d for d in zr.dims if d.startswith("s_")]
    s_w_dims = [d for d in zw.dims if d.startswith("s_")]
    if len(s_rho_dims) != 1 or len(s_w_dims) != 1:
        raise ValueError("Could not uniquely identify s_rho and s_w dimensions.")
    sr, sw = s_rho_dims[0], s_w_dims[0]
    surface = zw.isel({sw: -1})

    upper = {sr: slice(1, None)}
    lower = {sr: slice(None, -1)}
    zup = zr.isel(upper)
    zdw = zr.isel(lower)
    dz = zup - zdw
    dtype = np.result_type(zr.dtype, terms.rho1.dtype)
    cff = _as_dtype(gravity, dtype) / _as_dtype(rho0, dtype)

    interior_coord = zw[sw].isel({sw: slice(1, -1)})

    def _at_interior_w(array: xr.DataArray, selector: dict[str, slice]) -> xr.DataArray:
        out = array.isel(selector)
        if sw in out.coords and sw not in out.dims:
            out = out.drop_vars(sw)
        out = out.rename({sr: sw})
        return out.assign_coords({sw: interior_coord})

    rho1_up = _at_interior_w(terms.rho1, upper)
    rho1_dw = _at_interior_w(terms.rho1, lower)
    zup = _at_interior_w(zr, upper)
    zdw = _at_interior_w(zr, lower)
    dz = zup - zdw

    if split_eos:
        qp1_up = _at_interior_w(terms.qp1, upper)
        qp1_dw = _at_interior_w(terms.qp1, lower)
        dpth = surface - _as_dtype(0.5, dtype) * (zup + zdw)
        cff2 = (rho1_up - rho1_dw) + (qp1_up - qp1_dw) * dpth * (
            _as_dtype(1.0, dtype) - _as_dtype(qp2, dtype) * dpth
        )
        bvf = -cff * cff2 / dz
    else:
        K00 = _as_dtype(19092.56, dtype)
        tenth = _as_dtype(0.1, dtype)
        K0_up = _at_interior_w(terms.K0, upper)
        K0_dw = _at_interior_w(terms.K0, lower)
        K1_up = _at_interior_w(terms.K1, upper)
        K1_dw = _at_interior_w(terms.K1, lower)
        K2_up = _at_interior_w(terms.K2, upper)
        K2_dw = _at_interior_w(terms.K2, lower)

        zw_interior = zw.isel({sw: slice(1, -1)})
        dpth_interface = surface - zw_interior
        Kup = K0_dw + dpth_interface * (K1_dw + K2_dw * dpth_interface)
        Kdw = K0_up + dpth_interface * (K1_up + K2_up * dpth_interface)
        cff1 = tenth * (surface - zw_interior)
        drho1 = rho1_up - rho1_dw
        bvf = -cff * (
            drho1 * (K00 + Kdw) * (K00 + Kup)
            - cff1
            * (
                _as_dtype(rho0, dtype) * (Kdw - Kup)
                + K00 * drho1
                + rho1_up * Kdw
                - rho1_dw * Kup
            )
        ) / ((K00 + Kdw - cff1) * (K00 + Kup - cff1) * dz)

    if mask and "mask_rho" in ds:
        bvf = bvf * ds.mask_rho

    bvf.name = "bvf"
    bvf.attrs.update(
        long_name="CROCO adiabatic Brunt-Vaisala frequency squared",
        standard_name="square_of_brunt_vaisala_frequency_in_sea_water",
        units="s-2",
        reference_density=rho0,
        split_eos=split_eos,
        grid_location="w",
    )
    return bvf


# -----------------------------------------------------------------------------
# Horizontal density-anomaly gradients
#
# Moved here from stratification.py so that all density-related
# diagnostics (EOS — equation of state of seawater, and its horizontal
# gradients) live in one module.
# stratification.py re-exports these two names for backward compatibility.
# -----------------------------------------------------------------------------


def horizontal_density_gradient(
    density: xr.DataArray,
    xgrid: Grid,
    *,
    grid: Literal["native", "rho", "psi", "u", "v"] = "rho",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return horizontal density-anomaly gradients."""

    if "xi_rho" not in density.dims or "eta_rho" not in density.dims:
        raise ValueError(
            "density must be defined at CROCO rho points."
        )

    # Native derivatives:
    #   dρ/dx: rho -> u
    #   dρ/dy: rho -> v
    #
    # Use edge extension to avoid differencing against an artificial
    # zero value outside the model domain.
    drho_dx = xgrid.derivative(
        density,
        "X",
        boundary="extend",
    )

    drho_dy = xgrid.derivative(
        density,
        "Y",
        boundary="extend",
    )

    if grid == "native":
        return drho_dx, drho_dy

    if grid == "u":
        drho_dy = xgrid.interp(
            drho_dy,
            "X",
            boundary="extend",
        )

    elif grid == "v":
        drho_dx = xgrid.interp(
            drho_dx,
            "Y",
            boundary="extend",
        )

    elif grid == "psi":
        drho_dx = xgrid.interp(
            drho_dx,
            "Y",
            boundary="extend",
        )
        drho_dy = xgrid.interp(
            drho_dy,
            "X",
            boundary="extend",
        )

    elif grid == "rho":
        drho_dx = xgrid.interp(
            drho_dx,
            "X",
            boundary="extend",
        )
        drho_dy = xgrid.interp(
            drho_dy,
            "Y",
            boundary="extend",
        )

        # Centered rho-point gradients are undefined at the external
        # boundary. Mask only the edge relevant to each component.
        drho_dx = drho_dx.where(
            (drho_dx["xi_rho"] != drho_dx["xi_rho"].isel(xi_rho=0))
            & (drho_dx["xi_rho"] != drho_dx["xi_rho"].isel(xi_rho=-1))
        )

        drho_dy = drho_dy.where(
            (drho_dy["eta_rho"] != drho_dy["eta_rho"].isel(eta_rho=0))
            & (drho_dy["eta_rho"] != drho_dy["eta_rho"].isel(eta_rho=-1))
        )

    else:
        raise ValueError(
            "grid must be one of 'native', 'rho', 'psi', 'u', or 'v'."
        )

    drho_dx.name = "density_anomaly_gradient_x"
    drho_dy.name = "density_anomaly_gradient_y"

    return drho_dx, drho_dy


def density_gradient_magnitude(
    density: xr.DataArray,
    xgrid: Grid,
    *,
    grid: Literal["rho", "psi", "u", "v"] = "rho",
) -> xr.DataArray:
    """Return the magnitude of the horizontal density-anomaly gradient."""

    drho_dx, drho_dy = horizontal_density_gradient(
        density,
        xgrid,
        grid=grid,
    )

    magnitude = np.hypot(drho_dx, drho_dy)
    magnitude.name = "density_anomaly_gradient_magnitude"
    magnitude.attrs.update(
        {
            "long_name": "horizontal density anomaly gradient magnitude",
            "units": "kg m-4",
        }
    )

    return magnitude
