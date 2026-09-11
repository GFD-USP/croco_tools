"""xarray Dataset accessor for CROCO operations."""

from __future__ import annotations

import xarray as xr

from .grid import rotate_velocity
from .vertical import compute_depths, interpolate_to_depth


@xr.register_dataset_accessor("croco")
class CrocoAccessor:
    """Convenient CROCO operations attached to xarray datasets."""

    def __init__(
        self,
        dataset: xr.Dataset,
    ) -> None:
        self._dataset = dataset


    def depths(
        self,
        *,
        zeta: xr.DataArray | None = None,
        vtransform: int | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return depths at rho and w points."""
        return compute_depths(
            self._dataset,
            zeta=zeta,
            vtransform=vtransform,
        )


    def velocity(
        self,
        *,
        u_name: str = "u",
        v_name: str = "v",
        angle_name: str = "angle",
        angle_units: str = "radians",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return eastward and northward velocity on rho points."""
        return rotate_velocity(
            self._dataset[u_name],
            self._dataset[v_name],
            self._dataset[angle_name],
            angle_units=angle_units,
        )


    def to_depth(
        self,
        variable: str,
        depths: float | list[float],
        *,
        vertical_dimension: str = "s_rho",
    ) -> xr.DataArray:
        """Interpolate a named variable to fixed depths."""
        z_rho, _ = self.depths()

        return interpolate_to_depth(
            self._dataset[variable],
            z_rho,
            depths,
            vertical_dimension=vertical_dimension,
        )
