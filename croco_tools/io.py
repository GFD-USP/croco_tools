"""Input/output helpers for CROCO datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import xarray as xr


GRID_VARIABLES = (
    "lon_rho",
    "lat_rho",
    "lon_u",
    "lat_u",
    "lon_v",
    "lat_v",
    "lon_psi",
    "lat_psi",
    "mask_rho",
    "mask_u",
    "mask_v",
    "mask_psi",
    "h",
    "pm",
    "pn",
    "angle",
    "f",
)

VERTICAL_VARIABLES = (
    "s_rho",
    "s_w",
    "sc_r",
    "sc_w",
    "Cs_r",
    "Cs_w",
)


def open_croco(
    history: str | Path | Iterable[str | Path],
    *,
    grid: str | Path | None = None,
    chunks: Mapping[str, int] | None = None,
    decode_times: bool = True,
) -> xr.Dataset:
    """Open one or more CROCO history files.

    Parameters
    ----------
    history
        Path to one history file, or an iterable of paths.
    grid
        Optional CROCO grid file. Static variables are attached to the
        returned dataset.
    chunks
        Optional Dask chunk mapping passed to xarray.
    decode_times
        Decode CF-compliant time coordinates when possible.

    Returns
    -------
    xarray.Dataset
        CROCO history dataset with standard coordinates promoted.

    Examples
    --------
    >>> ds = open_croco(
    ...     "HIS/rio1km_his.nc",
    ...     grid="CROCO_FILES/rio1km_grd.nc",
    ... )
    """
    if isinstance(history, (str, Path)):
        dataset = xr.open_dataset(
            history,
            chunks=chunks,
            decode_times=decode_times,
        )
    else:
        paths = [str(Path(path)) for path in history]

        dataset = xr.open_mfdataset(
            paths,
            combine="by_coords",
            chunks=chunks,
            decode_times=decode_times,
        )

    dataset = promote_coordinates(dataset)

    if grid is not None:
        with xr.open_dataset(grid, decode_times=False) as grid_dataset:
            loaded_grid = grid_dataset.load()

        dataset = attach_grid(dataset, loaded_grid)

    return dataset


def attach_grid(
    dataset: xr.Dataset,
    grid: xr.Dataset,
    *,
    overwrite: bool = False,
) -> xr.Dataset:
    """Attach static CROCO grid variables to a history dataset.

    Existing variables are preserved unless ``overwrite=True``.
    """
    output = dataset.copy()

    for name in GRID_VARIABLES:
        if name not in grid:
            continue

        if name in output and not overwrite:
            continue

        output[name] = grid[name]

    return promote_coordinates(output)


def promote_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    """Promote standard grid and vertical variables to coordinates."""
    coordinate_names = [
        name
        for name in (*GRID_VARIABLES, *VERTICAL_VARIABLES)
        if name in dataset.variables
    ]

    if not coordinate_names:
        return dataset

    return dataset.set_coords(coordinate_names)
