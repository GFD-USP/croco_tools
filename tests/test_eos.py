"""Tests for croco_tools.eos.

density_anomaly implements CROCO's full nonlinear EOS polynomial, which is
not something to hand-verify in general. One special case *is* exactly
hand-checkable though: at T=0, S=0, and at the sea surface (z_r == top of
z_w, so the pressure-correction depth term is exactly zero), every
temperature/salinity-dependent term in rho_eos.F vanishes and the density
anomaly reduces exactly to ``r00 - rho0``. That gives a real regression
check on the full pipeline (not just a shape check) without transcribing
the polynomial by hand.

horizontal_density_gradient / density_gradient_magnitude also live in this
module (moved here from the former stratification.py, which now just
re-exports them for backward compatibility -- see test_stratification.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from croco_tools.eos import (
    buoyancy_frequency,
    density_anomaly,
    density_gradient_magnitude,
    horizontal_density_gradient,
)
from croco_tools.grid import prepare_xgcm

# CROCO rho_eos.F reference density coefficient (r00).
_R00 = 999.842594


def _surface_dataset(rho0: float = 1025.0) -> xr.Dataset:
    """A one-column dataset with T=S=0 and z_r exactly at the surface."""
    return xr.Dataset(
        {
            "temp": xr.DataArray([0.0], dims=("s_rho",)),
            "salt": xr.DataArray([0.0], dims=("s_rho",)),
            # z_r sits exactly at the surface, so surface - z_r == 0.
            "z_r": xr.DataArray([-5.0], dims=("s_rho",)),
            "z_w": xr.DataArray([-10.0, -5.0], dims=("s_w",)),
        }
    )


def test_density_anomaly_reduces_to_r00_minus_rho0_at_surface() -> None:
    rho0 = 1025.0
    ds = _surface_dataset(rho0=rho0)

    rho = density_anomaly(ds, rho0=rho0, split_eos=False)

    expected = _R00 - rho0
    np.testing.assert_allclose(rho.values, [expected], rtol=1.0e-10)


def test_density_anomaly_reduces_to_r00_minus_rho0_at_surface_split_eos() -> None:
    """The split_eos branch should agree with the standard branch when
    dpth == 0, since qp1 * dpth == 0 regardless of qp1."""
    rho0 = 1025.0
    ds = _surface_dataset(rho0=rho0)

    rho = density_anomaly(ds, rho0=rho0, split_eos=True)

    expected = _R00 - rho0
    np.testing.assert_allclose(rho.values, [expected], rtol=1.0e-10)


def test_density_anomaly_changes_with_rho0() -> None:
    ds = _surface_dataset()
    rho_a = density_anomaly(ds, rho0=1025.0)
    rho_b = density_anomaly(ds, rho0=1000.0)

    # rho = r00 - rho0 at the surface, so the difference is exactly
    # rho0_b - rho0_a.
    np.testing.assert_allclose(
        rho_b.item() - rho_a.item(),
        1025.0 - 1000.0,
        rtol=1.0e-10,
    )


def test_density_anomaly_missing_variable_raises() -> None:
    ds = _surface_dataset().drop_vars("salt")
    with pytest.raises(KeyError):
        density_anomaly(ds)


def test_density_anomaly_mask_zeroes_dry_cells() -> None:
    """Masking in eos.py multiplies by mask_rho (0/1), unlike the
    .where()-based NaN masking used in kinematics.py."""
    # Shape (s_rho=1, eta_rho=1, xi_rho=2): one wet cell, one dry cell,
    # both at the sea surface (dpth == 0) so the expected physical value
    # is the same closed form used above.
    ds = xr.Dataset(
        {
            "temp": xr.DataArray([[[0.0, 0.0]]], dims=("s_rho", "eta_rho", "xi_rho")),
            "salt": xr.DataArray([[[0.0, 0.0]]], dims=("s_rho", "eta_rho", "xi_rho")),
            "z_r": xr.DataArray([[[-5.0, -5.0]]], dims=("s_rho", "eta_rho", "xi_rho")),
            "z_w": xr.DataArray(
                [[[-10.0, -10.0]], [[-5.0, -5.0]]],
                dims=("s_w", "eta_rho", "xi_rho"),
            ),
            "mask_rho": xr.DataArray([[1.0, 0.0]], dims=("eta_rho", "xi_rho")),
        }
    )

    rho = density_anomaly(ds, rho0=1025.0, mask=True)

    # Masked cell is exactly zero, not NaN.
    assert rho.values[0, 0, 1] == 0.0
    assert not np.isnan(rho.values[0, 0, 1])
    # Wet cell keeps the physical value.
    np.testing.assert_allclose(rho.values[0, 0, 0], _R00 - 1025.0, rtol=1.0e-10)


def test_density_anomaly_return_terms_includes_rho1_and_k0() -> None:
    ds = _surface_dataset()
    out = density_anomaly(ds, return_terms=True)

    assert isinstance(out, xr.Dataset)
    assert "rho" in out
    assert "rho1" in out
    assert "K0" in out


# ---------------------------------------------------------------------------
# buoyancy_frequency: previously called an undefined `croco_density` name
# (a NameError on every call). Now fixed to call `density_anomaly`, the
# actual function name. This test verifies the fix and guards against a
# future regression back to the broken state.
# ---------------------------------------------------------------------------


def test_buoyancy_frequency_runs_without_error() -> None:
    ds = xr.Dataset(
        {
            "temp": xr.DataArray([1.0, 2.0, 3.0], dims=("s_rho",)),
            "salt": xr.DataArray([35.0, 35.0, 35.0], dims=("s_rho",)),
            "z_r": xr.DataArray([-30.0, -20.0, -10.0], dims=("s_rho",)),
            "z_w": xr.DataArray(
                [-40.0, -25.0, -15.0, -5.0], dims=("s_w",)
            ),
        }
    )

    bvf = buoyancy_frequency(ds)

    # Two interior w-levels between three rho-levels.
    assert bvf.sizes["s_w"] == 2
    assert np.all(np.isfinite(bvf.values))


def test_buoyancy_frequency_split_eos_runs_without_error() -> None:
    ds = xr.Dataset(
        {
            "temp": xr.DataArray([1.0, 2.0, 3.0], dims=("s_rho",)),
            "salt": xr.DataArray([35.0, 35.0, 35.0], dims=("s_rho",)),
            "z_r": xr.DataArray([-30.0, -20.0, -10.0], dims=("s_rho",)),
            "z_w": xr.DataArray(
                [-40.0, -25.0, -15.0, -5.0], dims=("s_w",)
            ),
        }
    )

    bvf = buoyancy_frequency(ds, split_eos=True, qp2=0.0)

    assert bvf.sizes["s_w"] == 2
    assert np.all(np.isfinite(bvf.values))


# ---------------------------------------------------------------------------
# horizontal_density_gradient / density_gradient_magnitude, now defined in
# this module (moved from stratification.py).
# ---------------------------------------------------------------------------


def test_density_gradient_functions_importable_from_eos(uniform_grid) -> None:
    ds = uniform_grid.copy()
    ny = ds.sizes["eta_rho"]
    nx = ds.sizes["xi_rho"]
    ds["density"] = xr.DataArray(
        np.full((ny, nx), 2.0), dims=("eta_rho", "xi_rho")
    )
    prepared, xgrid = prepare_xgcm(ds)

    dx, dy = horizontal_density_gradient(prepared["density"], xgrid, grid="native")
    magnitude = density_gradient_magnitude(prepared["density"], xgrid, grid="rho")

    np.testing.assert_allclose(dx.values, 0.0, atol=1e-12)
    np.testing.assert_allclose(dy.values, 0.0, atol=1e-12)
    assert magnitude.shape == prepared["density"].shape
