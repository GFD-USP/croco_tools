"""Tests for horizontal kinematic diagnostics (croco_tools.kinematics).

Strategy
--------
xgcm's exact index bookkeeping is easy to get subtly wrong by hand, so
these tests lean on two properties that hold *regardless* of that
bookkeeping:

1. Any finite difference of a spatially uniform field is exactly zero,
   so a uniform flow must give exactly zero vorticity/divergence/strain
   everywhere.
2. For a fixed field, the ratio of the ``normalized=True`` output to the
   ``normalized=False`` output must equal ``1/f`` (signed diagnostics)
   or ``1/|f|`` (unsigned diagnostics) at every point -- independent of
   what the raw, unnormalized value actually is.
"""

from __future__ import annotations

import numpy as np
import pytest

from croco_tools.kinematics import (
    horizontal_divergence,
    relative_vorticity,
    strain_rate,
    vorticity_at_rho,
)


# ---------------------------------------------------------------------------
# Uniform flow -> exactly zero
# ---------------------------------------------------------------------------


def test_relative_vorticity_zero_for_uniform_flow(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    zeta = relative_vorticity(ds, xgrid, grid="psi")
    np.testing.assert_allclose(zeta.values, 0.0, atol=1.0e-12)


def test_horizontal_divergence_zero_for_uniform_flow(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    delta = horizontal_divergence(ds, xgrid, grid="rho")
    np.testing.assert_allclose(delta.values, 0.0, atol=1.0e-12)


def test_strain_rate_zero_for_uniform_flow(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    alpha = strain_rate(ds, xgrid, grid="rho")
    np.testing.assert_allclose(alpha.values, 0.0, atol=1.0e-12)


# ---------------------------------------------------------------------------
# Sheared flow -> nonzero, and normalization is exactly 1/f or 1/|f|
# ---------------------------------------------------------------------------


def test_relative_vorticity_is_nonzero_for_sheared_flow(
    prepared_sheared_flow,
) -> None:
    ds, xgrid, _alpha = prepared_sheared_flow
    zeta = relative_vorticity(ds, xgrid, grid="rho")
    assert np.all(np.isfinite(zeta.values))
    assert not np.allclose(zeta.values, 0.0)


def test_relative_vorticity_normalization_matches_signed_f(
    prepared_sheared_flow,
) -> None:
    ds, xgrid, _alpha = prepared_sheared_flow
    raw = relative_vorticity(ds, xgrid, grid="rho", normalized=False)
    normalized = relative_vorticity(ds, xgrid, grid="rho", normalized=True)
    f = ds["f"]

    np.testing.assert_allclose(
        normalized.values,
        (raw / f).values,
        rtol=1.0e-10,
    )


def test_horizontal_divergence_normalization_matches_abs_f(
    prepared_sheared_flow,
) -> None:
    ds, xgrid, _alpha = prepared_sheared_flow
    raw = horizontal_divergence(ds, xgrid, grid="rho", normalized=False)
    normalized = horizontal_divergence(ds, xgrid, grid="rho", normalized=True)
    f = ds["f"]

    np.testing.assert_allclose(
        normalized.values,
        (raw / abs(f)).values,
        rtol=1.0e-10,
    )


def test_strain_rate_normalization_matches_abs_f(prepared_sheared_flow) -> None:
    ds, xgrid, _alpha = prepared_sheared_flow
    raw = strain_rate(ds, xgrid, grid="rho", normalized=False)
    normalized = strain_rate(ds, xgrid, grid="rho", normalized=True)
    f = ds["f"]

    np.testing.assert_allclose(
        normalized.values,
        (raw / abs(f)).values,
        rtol=1.0e-10,
    )


# ---------------------------------------------------------------------------
# Grid-location handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("location", "expected_dims"),
    [
        ("rho", {"eta_rho", "xi_rho"}),
        ("psi", {"eta_v", "xi_u"}),
        ("u", {"eta_rho", "xi_u"}),
        ("v", {"eta_v", "xi_rho"}),
    ],
)
def test_relative_vorticity_grid_locations(
    prepared_uniform_flow, location, expected_dims
) -> None:
    ds, xgrid = prepared_uniform_flow
    zeta = relative_vorticity(ds, xgrid, grid=location)
    assert set(zeta.dims) == expected_dims


def test_relative_vorticity_native_is_psi(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    native = relative_vorticity(ds, xgrid, grid="native")
    psi = relative_vorticity(ds, xgrid, grid="psi")
    assert set(native.dims) == set(psi.dims)


def test_horizontal_divergence_native_is_rho(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    native = horizontal_divergence(ds, xgrid, grid="native")
    assert set(native.dims) == {"eta_rho", "xi_rho"}


def test_invalid_grid_location_raises(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    with pytest.raises(ValueError):
        relative_vorticity(ds, xgrid, grid="not-a-location")


# ---------------------------------------------------------------------------
# Required-variable checks
# ---------------------------------------------------------------------------


def test_relative_vorticity_missing_velocity_raises(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    ds = ds.drop_vars("u")
    with pytest.raises(KeyError):
        relative_vorticity(ds, xgrid)


def test_horizontal_divergence_missing_velocity_raises(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    ds = ds.drop_vars("v")
    with pytest.raises(KeyError):
        horizontal_divergence(ds, xgrid)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_relative_vorticity_masks_dry_cells(prepared_uniform_flow) -> None:
    ds, xgrid = prepared_uniform_flow
    ds = ds.copy()
    ds["mask_rho"] = ds["mask_rho"].copy(deep=True)
    ds["mask_rho"][0, 0] = 0.0

    zeta = relative_vorticity(ds, xgrid, grid="rho", mask=True)
    assert np.isnan(zeta.values).any()

    zeta_unmasked = relative_vorticity(ds, xgrid, grid="rho", mask=False)
    assert not np.isnan(zeta_unmasked.values).any()


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------


def test_vorticity_at_rho_matches_unnormalized_relative_vorticity(
    prepared_uniform_flow,
) -> None:
    ds, xgrid = prepared_uniform_flow
    wrapper = vorticity_at_rho(ds, xgrid)
    direct = relative_vorticity(ds, xgrid, grid="rho", normalized=False)
    np.testing.assert_allclose(wrapper.values, direct.values)
    assert set(wrapper.dims) == {"eta_rho", "xi_rho"}
