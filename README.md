# croco_tools

A modern, xarray-first Python toolbox for analyzing output from the
[CROCO](https://www.croco-ocean.org/) ocean model, built on top of
[xgcm](https://xgcm.readthedocs.io/en/latest/) for C-grid-aware
operations. It's a from-scratch package, informed by (but independent
of) the upstream `croco_pytools`/`xcroco` toolboxes: the goal is the
same class of analysis, with a flatter, more explicit, more xarray-idiomatic
API that's easier to read, extend, and trust.

For the full function-by-function inventory, known issues, and a gap map
against real notebook usage, see [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

## Current scope

Version 0.1 provides:

- opening one or multiple CROCO history files, and attaching static grid
  variables (`io.py`);
- CROCO C-grid interpolation (rho ⇄ u/v/psi) and velocity rotation to
  east/north (`grid.py`);
- one-call `xgcm.Grid` setup from raw CROCO output (`prepare_xgcm`);
- vorticity, divergence, and strain-rate diagnostics (`kinematics.py`);
- the CROCO nonlinear equation of state of seawater, buoyancy frequency,
  and horizontal density-anomaly gradients (`eos.py`);
- physical depths for both vertical transformations, and interpolation
  from sigma levels to fixed depths (`vertical.py`);
- geographic transects with Delaunay-based horizontal interpolation
  (`section.py`);
- a `.croco` xarray Dataset accessor wrapping the most common operations
  (`accessor.py`);
- an expanded, from-scratch test suite (~100 tests) covering all of the
  above.

## Installation

```bash
python -m pip install -e ".[test]"
```

This installs `croco_tools` in editable mode along with its runtime
dependencies (`xarray`, `dask`, `netCDF4`, `numpy`, `xgcm`, `scipy`,
`pyproj`) and `pytest` for the test suite.

**Requires `xgcm>=0.10`.** xgcm 0.10 renamed its `boundary=` keyword to
`padding=` across `Grid(...)`, `.interp()`, and `.derivative()`, and
raises a hard `ValueError` for the old name rather than a deprecation
warning — `croco_tools` uses the new `padding=` API throughout, so it
will not import correctly against xgcm 0.9.x.

## Running the tests

```bash
pytest -q
```

The suite is fully synthetic — small, hand-built C-grid fixtures in
`tests/conftest.py`, no real CROCO output required — so it runs in well
under a second. See `docs/DOCUMENTATION.md` for notes on what each test
file covers and the reasoning behind the trickier numeric checks (e.g.
`compute_depths` under both vertical transformations, the EOS
surface-density reduction).

## Quick start

### Opening a dataset and using the `.croco` accessor

```python
import croco_tools

dataset = croco_tools.open_croco(
    "HIS/rio1km_his.nc",
    grid="CROCO_FILES/rio1km_grd.nc",
)

z_rho, z_w = dataset.croco.depths()

u_east, v_north = dataset.croco.velocity()

temperature_100m = dataset.croco.to_depth(
    "temp",
    depths=-100.0,
)
```

### The function-based API and xgcm

For anything involving horizontal derivatives (vorticity, divergence,
strain, density gradients), work directly with `prepare_xgcm` and an
`xgcm.Grid`:

```python
import croco_tools as ct

dataset = ct.open_croco("HIS/rio1km_his.nc", grid="CROCO_FILES/rio1km_grd.nc")
dataset, xgrid = ct.prepare_xgcm(dataset)

zeta = ct.relative_vorticity(dataset, xgrid, normalized=True, grid="rho")
divergence = ct.horizontal_divergence(dataset, xgrid, grid="rho")
strain = ct.strain_rate(dataset, xgrid, grid="rho")

z_rho, z_w = ct.compute_depths(dataset)
temp_100m = ct.interpolate_to_depth(dataset["temp"], z_rho, -100.0)
```

### Density and buoyancy

Continuing from the `dataset`/`xgrid` above:

```python
from croco_tools import eos

z_rho, z_w = ct.compute_depths(dataset)
dataset = dataset.assign(z_r=z_rho, z_w=z_w)

rho = eos.density_anomaly(dataset, rho0=1025.0)
bvf = eos.buoyancy_frequency(dataset)

drho_dx, drho_dy = eos.horizontal_density_gradient(rho, xgrid, grid="rho")
gradient_magnitude = eos.density_gradient_magnitude(rho, xgrid, grid="rho")
```

### Transects

Also continuing from `dataset` above (opened with `grid=...`, so
`lon_rho`/`lat_rho`/`pm`/`pn` are already attached):

```python
from croco_tools import Transect
from croco_tools.section import interpolate_section

track = Transect.from_endpoints(
    dataset,
    start=(-42.0, -23.0),
    end=(-40.5, -24.5),
)

section = interpolate_section(
    dataset["temp"].isel(s_rho=-1),
    track,
    longitude=dataset["lon_rho"],
    latitude=dataset["lat_rho"],
)
```

## Package layout

```
croco_tools/
├── grid.py             # C-grid interpolation, coord normalization, xgcm setup
├── kinematics.py         # vorticity / divergence / strain
├── eos.py                 # equation of state of seawater, buoyancy frequency, density gradients
├── vertical.py             # sigma coordinates, compute_depths, interpolate_to_depth
├── io.py                   # open_croco, attach_grid, promote_coordinates
├── section.py               # Transect, interpolate_section
├── geo.py                    # geodesic distance / bearing (used by section.py)
├── accessor.py                 # `.croco` xarray Dataset accessor
├── xgcm_tools.py                 # backward-compat re-export shim
└── stratification.py               # backward-compat re-export shim (see eos.py)
```

## Style

The code uses:

- short lines;
- two blank lines between top-level definitions;
- descriptive names;
- NumPy-style docstrings;
- xarray-aware arithmetic;
- no GUI dependencies;
- no broad exception suppression;
- no silent coordinate deletion.
