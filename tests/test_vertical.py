"""Tests for CROCO vertical coordinates."""

import numpy as np
import pytest
import xarray as xr

from croco_tools.vertical import (
    compute_depths,
    ensure_vertical_coordinates,
    interpolate_to_depth,
)


def make_dataset(
    vtransform: int,
) -> xr.Dataset:
    return xr.Dataset(
        {
            "h": xr.DataArray(
                [[100.0]],
                dims=("eta_rho", "xi_rho"),
            ),
            "zeta": xr.DataArray(
                [[[0.0]]],
                dims=("time", "eta_rho", "xi_rho"),
            ),
            "hc": xr.DataArray(20.0),
            "Vtransform": xr.DataArray(vtransform),
            "sc_r": xr.DataArray(
                [-0.75, -0.25],
                dims=("s_rho",),
            ),
            "sc_w": xr.DataArray(
                [-1.0, -0.5, 0.0],
                dims=("s_w",),
            ),
            "Cs_r": xr.DataArray(
                [-0.75, -0.25],
                dims=("s_rho",),
            ),
            "Cs_w": xr.DataArray(
                [-1.0, -0.5, 0.0],
                dims=("s_w",),
            ),
        }
    )


def test_compute_depth_shapes() -> None:
    dataset = make_dataset(vtransform=2)

    z_rho, z_w = compute_depths(dataset)

    assert z_rho.sizes["s_rho"] == 2
    assert z_w.sizes["s_w"] == 3


def test_interpolate_to_depth_is_linear() -> None:
    depths = xr.DataArray(
        [-100.0, -50.0, 0.0],
        dims=("s_rho",),
    )
    values = xr.DataArray(
        [0.0, 5.0, 10.0],
        dims=("s_rho",),
    )

    result = interpolate_to_depth(
        values,
        depths,
        [-75.0, -25.0],
    )

    np.testing.assert_allclose(
        result.values,
        [2.5, 7.5],
    )


# ---------------------------------------------------------------------------
# Exact numeric checks of compute_depths, hand-derived from the same
# formulas implemented in vertical.py (see docs for the worked algebra).
# ---------------------------------------------------------------------------


def _single_point_dataset(
    *,
    vtransform: int,
    h: float,
    hc: float,
    zeta: float,
    sc: list[float],
    cs: list[float],
) -> xr.Dataset:
    return xr.Dataset(
        {
            "h": xr.DataArray([[h]], dims=("eta_rho", "xi_rho")),
            "zeta": xr.DataArray([[zeta]], dims=("eta_rho", "xi_rho")),
            "hc": xr.DataArray(hc),
            "Vtransform": xr.DataArray(vtransform),
            "sc_r": xr.DataArray(sc, dims=("s_rho",)),
            "sc_w": xr.DataArray([sc[0] - 0.1, *sc, 0.0][: len(sc) + 1], dims=("s_w",)),
            "Cs_r": xr.DataArray(cs, dims=("s_rho",)),
            "Cs_w": xr.DataArray([cs[0] - 0.1, *cs, 0.0][: len(cs) + 1], dims=("s_w",)),
        }
    )


def test_compute_depths_vtransform_1_matches_hand_derivation() -> None:
    # h=100, hc=20, zeta=0, sigma=[-0.6, -0.1], Cs=[-0.5, -0.05].
    # z0 = hc*(sigma - Cs) + Cs*h; z_rho = z0 (since zeta == 0).
    dataset = _single_point_dataset(
        vtransform=1,
        h=100.0,
        hc=20.0,
        zeta=0.0,
        sc=[-0.6, -0.1],
        cs=[-0.5, -0.05],
    )

    z_rho, _ = compute_depths(dataset, vtransform=1)

    expected = [-52.0, -6.0]
    np.testing.assert_allclose(
        z_rho.isel(eta_rho=0, xi_rho=0).values,
        expected,
        rtol=1.0e-10,
    )


def test_compute_depths_vtransform_2_matches_hand_derivation() -> None:
    # Same inputs as above, but the vtransform=2 (NEW_S_COORD) formula:
    # z0 = (hc*sigma + Cs*h) / (hc + h); z_rho = z0 (since zeta == 0).
    dataset = _single_point_dataset(
        vtransform=2,
        h=100.0,
        hc=20.0,
        zeta=0.0,
        sc=[-0.6, -0.1],
        cs=[-0.5, -0.05],
    )

    z_rho, _ = compute_depths(dataset, vtransform=2)

    expected = [-62.0 / 120.0 * 100.0, -7.0 / 120.0 * 100.0]
    np.testing.assert_allclose(
        z_rho.isel(eta_rho=0, xi_rho=0).values,
        expected,
        rtol=1.0e-10,
    )


def test_compute_depths_vtransform_2_with_nonzero_zeta() -> None:
    # h=100, hc=20, zeta=1.0, sigma=Cs=-0.5 (single level).
    # z0 = (20*-0.5 + -0.5*100) / 120 = -0.5
    # z_rho = zeta + (zeta + h) * z0 = 1 + 101 * -0.5 = -49.5
    dataset = _single_point_dataset(
        vtransform=2,
        h=100.0,
        hc=20.0,
        zeta=1.0,
        sc=[-0.5],
        cs=[-0.5],
    )

    z_rho, _ = compute_depths(dataset, vtransform=2)

    np.testing.assert_allclose(
        float(z_rho.isel(eta_rho=0, xi_rho=0, s_rho=0)),
        -49.5,
        rtol=1.0e-10,
    )


def test_compute_depths_invalid_vtransform_raises() -> None:
    dataset = make_dataset(vtransform=2)
    with pytest.raises(ValueError):
        compute_depths(dataset, vtransform=3)


def test_compute_depths_accepts_explicit_zeta_argument() -> None:
    dataset = _single_point_dataset(
        vtransform=2,
        h=100.0,
        hc=20.0,
        zeta=0.0,
        sc=[-0.5],
        cs=[-0.5],
    )
    # Passing zeta explicitly should override the dataset's own zeta=0.
    explicit_zeta = xr.DataArray([[1.0]], dims=("eta_rho", "xi_rho"))

    z_rho, _ = compute_depths(dataset, zeta=explicit_zeta, vtransform=2)

    np.testing.assert_allclose(
        float(z_rho.isel(eta_rho=0, xi_rho=0, s_rho=0)),
        -49.5,
        rtol=1.0e-10,
    )


def test_ensure_vertical_coordinates_generates_missing_stretching() -> None:
    dataset = xr.Dataset(
        {
            "temp": xr.DataArray(np.zeros(4), dims=("s_rho",)),
        }
    )

    out = ensure_vertical_coordinates(dataset, theta_s=5.0, theta_b=2.0, hc=10.0)

    assert "Cs_r" in out
    assert "Cs_w" in out
    assert out.sizes["s_rho"] == 4
    assert out.sizes["s_w"] == 5


def test_ensure_vertical_coordinates_requires_theta_when_missing() -> None:
    dataset = xr.Dataset({"temp": xr.DataArray(np.zeros(4), dims=("s_rho",))})
    with pytest.raises(KeyError):
        ensure_vertical_coordinates(dataset)


# ---------------------------------------------------------------------------
# interpolate_to_depth edge cases
# ---------------------------------------------------------------------------


def test_interpolate_to_depth_outside_range_is_nan() -> None:
    depths = xr.DataArray([-100.0, -50.0, 0.0], dims=("s_rho",))
    values = xr.DataArray([0.0, 5.0, 10.0], dims=("s_rho",))

    result = interpolate_to_depth(values, depths, [-150.0, 50.0])

    assert np.isnan(result.values).all()


def test_interpolate_to_depth_ignores_nan_gaps() -> None:
    depths = xr.DataArray([-100.0, -50.0, 0.0], dims=("s_rho",))
    values = xr.DataArray([0.0, np.nan, 10.0], dims=("s_rho",))

    # With the middle point dropped, interpolation is linear between
    # (-100, 0) and (0, 10): value(-50) == 5.0 still, since the column
    # happens to remain linear -- but crucially this must not raise or
    # silently propagate the NaN to a target far from it.
    result = interpolate_to_depth(values, depths, [-50.0])

    np.testing.assert_allclose(result.values, [5.0])


def test_interpolate_to_depth_too_few_valid_points_returns_nan() -> None:
    depths = xr.DataArray([-100.0, -50.0, 0.0], dims=("s_rho",))
    values = xr.DataArray([np.nan, 5.0, np.nan], dims=("s_rho",))

    result = interpolate_to_depth(values, depths, [-75.0])

    assert np.isnan(result.values).all()


def test_interpolate_to_depth_preserves_source_attrs() -> None:
    depths = xr.DataArray([-100.0, 0.0], dims=("s_rho",))
    values = xr.DataArray(
        [0.0, 10.0],
        dims=("s_rho",),
        attrs={"units": "degC", "long_name": "temperature"},
    )

    result = interpolate_to_depth(values, depths, [-50.0])

    assert result.attrs["units"] == "degC"
    assert result.attrs["interpolation"] == "linear in physical depth"
