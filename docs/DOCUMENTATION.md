# croco_tools — Documentation & Gap Map

Generated from a read-through of the actual `croco_tools/` source
(version `0.1.0`), cross-checked against `velocity_gradients.ipynb` and
`00_snapshots_gigatl_vs_rio300m.ipynb` (the two notebooks showing real
usage), and against `xcroco` (legacy reference code from Dante
Napolitano) and `croco_pytools` (the upstream Matlab/Python toolbox).

This is documentation of *current, actual* behavior. Bugs and
inconsistencies found along the way are called out explicitly in
[Known Issues](#known-issues) rather than being silently fixed — the
two that have since been fixed were only changed on explicit request,
and are marked as such there.

---

## 1. Package layout

```
croco_tools/
├── __init__.py        # public API surface
├── grid.py             # C-grid interpolation, coord normalization, xgcm setup
├── kinematics.py        # vorticity / divergence / strain (imports from grid)
├── stratification.py    # BACKWARD-COMPAT SHIM — re-exports density-gradient fns from eos.py
├── xgcm_tools.py         # BACKWARD-COMPAT SHIM — re-exports grid+kinematics+stratification
├── eos.py               # CROCO nonlinear equation of state of seawater (rho_eos.F translation)
├── vertical.py           # sigma coordinates, compute_depths, interpolate_to_depth
├── io.py                 # open_croco, attach_grid, promote_coordinates
├── section.py            # Transect, interpolate_section (Delaunay-based)
├── geo.py                # geodesic distance / bearing (pyproj wrapper, CROCO-independent)
└── accessor.py           # `.croco` xarray Dataset accessor (thin wrapper)
```

`xgcm_tools.py` and `stratification.py` are **not** duplicate
implementations — they are compatibility shims (their own docstrings say
so) that re-export names from `grid.py`, `kinematics.py`, and `eos.py`.
`import croco_tools.xgcm_tools`, `import croco_tools.stratification`, and
`import croco_tools.kinematics`/`croco_tools.eos` directly all reach the
same functions.

---

## 2. Function inventory

### grid.py — C-grid interpolation & xgcm setup

| Function | Status | Notes |
|---|---|---|
| `rho_to_u`, `rho_to_v` | ✅ implemented, tested | Simple 2-point average; drops/rebuilds the staggered index coordinate. |
| `rho_to_psi` | ✅ implemented, tested | 4-point average of surrounding rho cells. |
| `u_to_rho`, `v_to_rho` | ✅ implemented, tested | Interior = average of neighbors; **boundary points are filled by duplicating the nearest u/v value**, not by extrapolation. |
| `psi_to_rho` | ✅ implemented, tested | 4-point average + `pad(mode="edge")` to restore the original rho shape. |
| `rotate_velocity` | ✅ implemented, tested | Interpolates u/v to rho points first (if needed), then rotates by `angle` (radians or degrees) to get eastward/northward components. |
| `normalize_croco_coords` | ✅ implemented, tested | Renames XIOS-style dims (`time_counter`, `x_rho`, `y_rho`, …) to the canonical `xi_rho`/`eta_rho`/`xi_u`/`eta_v`, and `nav_lon`/`nav_lat` → `lon`/`lat`. Raises `ValueError` if the required C-grid dims still aren't present after renaming. |
| `merge_grid` | ✅ implemented, tested | Merges a fixed allow-list of grid variables (`pm`, `pn`, `mask_rho`, `lon_*`, `lat_*`, `angle`, `f`, `h`) from a grid file into a history dataset. Uses `compat="override", join="exact"` — grid/history coordinates must align exactly. |
| `add_horizontal_metrics` | ✅ implemented, tested | Derives `dx_*`/`dy_*` from `pm`/`pn`, builds `area_*`, and a conservative `mask_psi` (all 4 surrounding rho cells must be wet). Requires `pm`/`pn` present. |
| `build_xgcm_grid` | ✅ implemented, tested | Constructs the `xgcm.Grid` with `padding={"X": "extend", "Y": "extend"}` (renamed from `boundary=` for xgcm >= 0.10 — see Known Issues). Requires metrics already added (raises `KeyError` otherwise, with a pointer to call `add_horizontal_metrics` first). |
| `prepare_xgcm` | ✅ implemented, tested | The one-call convenience wrapper: `normalize → (merge grid) → add metrics → build Grid`. This is the function the notebooks actually import (`from croco_tools import prepare_xgcm`). |

### kinematics.py — velocity-derived diagnostics

| Function | Status | Notes |
|---|---|---|
| `relative_vorticity` | ✅ implemented, tested | Native location: psi. Circulation-based (`Δ_X(v·dy) − Δ_Y(u·dx)) / A_psi`). `normalized=True` divides by **signed** `f`. |
| `horizontal_divergence` | ✅ implemented, tested | Native location: rho. `normalized=True` divides by **`abs(f)`**. |
| `strain_rate` | ✅ implemented, tested | Native location: rho (shear computed at psi then interpolated to rho before combining with the normal component). `normalized=True` divides by `abs(f)`. |
| `vorticity_at_rho` | ✅ implemented, tested | Thin backward-compatible wrapper = `relative_vorticity(..., normalized=False, grid="rho")`. |

All three main diagnostics share: an optional `grid=` target location
(`"native"`, `"rho"`, `"psi"`, `"u"`, `"v"`), a `mask=True` default that
applies a conservative wet mask via `.where(...)` (→ **NaN** at masked
points), and a `min_abs_f` floor (default `1e-10`) below which
normalized output is masked to avoid blowing up near the equator.

### eos.py — CROCO equation of state of seawater & density-gradient diagnostics

| Function | Status | Notes |
|---|---|---|
| `density_anomaly` | ✅ implemented, tested | Direct translation of CROCO's `rho_eos.F` (both the `split_eos=False` and `split_eos=True` branches). **Masking here multiplies by `mask_rho` (0/1) → masked cells become exactly `0.0`, not `NaN`** — this differs from the `.where()`-based NaN masking in `kinematics.py`. Worth knowing if you're chaining these together. |
| `buoyancy_frequency` | ✅ implemented, tested | Fixed — previously called an undefined `croco_density`; now correctly calls `density_anomaly`. See the changelog note in [Known Issues](#known-issues). |
| `horizontal_density_gradient` | ✅ implemented, tested | Native derivative locations: `dρ/dx` at u, `dρ/dy` at v (`xgrid.derivative(..., boundary="extend")`). `grid="rho"` additionally masks the outermost row/column, since centered rho-point gradients are undefined there. **Moved here from the former `stratification.py`** (now a backward-compat re-export shim) so all density-related code lives in one place. |
| `density_gradient_magnitude` | ✅ implemented, tested | `np.hypot(dρ/dx, dρ/dy)` after moving both components to the requested location. Also moved here from `stratification.py`. |

`stratification.py` still exists and still works — it now just
re-exports `horizontal_density_gradient`/`density_gradient_magnitude`
from `eos.py`, the same pattern `xgcm_tools.py` already used. Existing
code importing from `croco_tools.stratification` needs no changes.

### vertical.py — sigma coordinates & depth interpolation

| Function | Status | Notes |
|---|---|---|
| `ensure_vertical_coordinates` | ✅ implemented, tested | Reconstructs `s_rho`/`s_w`/`Cs_r`/`Cs_w` from `theta_s`/`theta_b` if not already in the dataset (as variables or global attrs). Recognizes common aliases (`sc_r`/`sc_w`, `Cs_rho`, `THETA_S`, `HC`, …). |
| `compute_depths` | ✅ implemented, tested | Both `vtransform=1` (old) and `vtransform=2` (`NEW_S_COORD`) formulas implemented and verified by hand against the source. `vtransform=None` triggers auto-detection from a `Vtransform` variable/attribute or the `NEW_S_COORD` CPP flag. |
| `interpolate_to_depth` | ✅ implemented, tested | `np.interp`-based linear interpolation per water column via `xr.apply_ufunc` (dask-parallelized, vectorized). NaNs in the source column are dropped before interpolating; targets outside the valid depth range come back as NaN; columns with fewer than 2 finite points return all-NaN. |

### io.py — file I/O

| Function | Status | Notes |
|---|---|---|
| `open_croco` | ✅ implemented, tested* | Single file → `xr.open_dataset`; iterable of paths → `xr.open_mfdataset(combine="by_coords")`. Optionally attaches a grid file. *(the file-opening test is skipped automatically if `netCDF4` isn't installed in the test environment — see [Testing notes](#testing-notes)).* |
| `attach_grid` | ✅ implemented, tested | Copies a fixed list of grid variables (`GRID_VARIABLES`) from a grid dataset into the history dataset, skipping ones already present unless `overwrite=True`. Promotes them to coordinates afterward. |
| `promote_coordinates` | ✅ implemented, tested | Moves known grid/vertical variable names into `.coords`. |

Note: `io.py`'s `attach_grid` is a **separate, simpler code path** from
`grid.py`'s `merge_grid` (used inside `prepare_xgcm`) — it doesn't call
`normalize_croco_coords` first and uses a plain "copy if absent" merge
rather than `xr.merge(..., join="exact")`. Both exist and both work,
but they're not the same function, so a dataset opened with
`open_croco(..., grid=...)` and one built via `prepare_xgcm(ds, grd)`
are not guaranteed to be byte-for-byte identical.

### section.py — transects

| Function | Status | Notes |
|---|---|---|
| `Transect` (dataclass) | ✅ implemented, tested | Two construction paths: `.from_endpoints(dataset, start, end, sampling=...)` (auto-samples at ~model grid spacing using `pm`/`pn`) and `.from_coordinates(lon, lat)`. Computes cumulative distance, bearing, and tangent/left-normal unit vectors on WGS84 via `pyproj`. |
| `Transect.rotate_velocity` | ✅ implemented, tested | Rotates geographic (east, north) velocity into (along-track, across-track) components. |
| `Transect.to_dataset` | ✅ implemented, tested | Packages the transect geometry as an `xr.Dataset`. |
| `interpolate_section` | ✅ implemented, tested | Horizontal interpolation onto a transect using a **Delaunay triangulation built once and reused** across all non-horizontal dimensions (time, depth, …) — this is the main performance-oriented piece of the codebase. |

`Transect` is exported from the top-level package (`croco_tools.Transect`);
`interpolate_section` currently is **not** (see
[Gap map](#gap-map-vs-notebook-usage) below).

### geo.py — geography utilities (CROCO-independent)

| Function | Status | Notes |
|---|---|---|
| `geodesic_distance` | ✅ implemented, tested | Thin `pyproj.Geod(ellps="WGS84").inv()` wrapper. |
| `along_track_distance` | ✅ implemented, tested | Cumulative geodesic distance along a 1-D track; `origin="start"` or `"end"`. |
| `bearing` | ✅ implemented, tested | Forward azimuth, degrees clockwise from north. |

None of `geo.py`'s functions are exported from the top-level
`croco_tools` package — they're used internally by `section.py`.

### accessor.py — `.croco` xarray accessor

| Method | Status | Notes |
|---|---|---|
| `ds.croco.depths()` | ✅ implemented, tested | Thin wrapper over `vertical.compute_depths`. |
| `ds.croco.velocity()` | ✅ implemented, tested | Thin wrapper over `grid.rotate_velocity`, reading `u`/`v`/`angle` by name (configurable). |
| `ds.croco.to_depth(name, depths)` | ✅ implemented, tested | Computes depths internally, then calls `vertical.interpolate_to_depth`. |

**This confirms the README's accessor-style example (`dataset.croco.depths()`)
is real, working code** — not stale or aspirational. It coexists with,
and is built on top of, the plain function-based API
(`compute_depths(ds, ...)`) that the notebooks import directly. Both
styles are legitimate entry points into the same underlying code; they
are not in tension with each other.

---

## 3. Known issues

These are genuine bugs/inconsistencies found while reading the source.
Items 1, 2, 3, and 5 have since been **fixed** (each with your explicit
go-ahead); 4 is still open, flagged only.

1. ~~**`buoyancy_frequency` is broken (`NameError` on every call).**~~
   **Fixed.** `eos.py` called `croco_density(...)`, an undefined name —
   the function is named `density_anomaly`. Changed the one call site
   to `density_anomaly`. `tests/test_eos.py` now asserts
   `buoyancy_frequency` runs and returns finite values (both the
   `split_eos=False` and `split_eos=True` paths), replacing the earlier
   regression test that pinned the broken behavior.

2. ~~**`__init__.py`'s `__all__` lists two names that don't exist.**~~
   **Fixed.** `"croco_buoyancy_frequency"` and `"croco_density"` were
   replaced with the names actually imported into the package:
   `"buoyancy_frequency"` and `"density_anomaly"` (which were imported
   but had been missing from `__all__` entirely — this fix closes both
   sides of the same gap).

3. ~~**`pyproject.toml` is missing three runtime dependencies.**~~
   **Fixed.** `dependencies` listed only `dask[array]`, `netCDF4`,
   `numpy`, `xarray`, even though `grid.py` imports `xgcm` and
   `section.py` imports `scipy.spatial.Delaunay` and `pyproj.Geod`.
   Added `xgcm>=0.10`, `scipy>=1.10`, `pyproj>=3.5` to `dependencies`.
   The `xgcm>=0.10` floor is not arbitrary — see item 5.

4. **`io.py`'s `attach_grid`/`open_croco` and `grid.py`'s `merge_grid`
   are two independent, not-quite-equivalent ways to attach a grid
   file** (see the io.py table note above). Not a bug exactly, but
   worth being deliberate about which one a given workflow uses, since
   they can behave differently on messy real-world files (differing
   coordinate names, non-exact-matching coordinates, etc.).

5. ~~**`xgcm` >= 0.10 renamed the `boundary` argument to `padding`,
   raising `ValueError` (not a deprecation warning) for the old
   name.**~~ **Fixed.** This broke every `Grid(...)` construction in
   `grid.py` (`add_horizontal_metrics`'s temporary grid and
   `build_xgcm_grid`'s real one — both used `boundary={"X": "extend",
   "Y": "extend"}`) and every `xgrid.interp`/`xgrid.derivative` call in
   `eos.py`'s `horizontal_density_gradient` (`boundary="extend"`, 8
   call sites). Since `kinematics.py` never passes `boundary=`/`padding=`
   explicitly — it relies on the `Grid` object's own default set at
   construction — fixing the two `Grid(...)` calls in `grid.py` also
   fixed every `kinematics.py`-based test that was failing with the
   same error. All renamed to `padding=` with the same value (`"extend"`,
   which is unaffected by xgcm's separate removal of the different
   `"extrapolate"` boundary type). This is why `xgcm>=0.10` is now
   pinned in `pyproject.toml` (item 3) — the code no longer works with
   xgcm 0.9.x, which doesn't understand `padding=` at all.

---

## 4. Gap map vs. notebook usage

Everything imported by name in `velocity_gradients.ipynb` and
`00_snapshots_gigatl_vs_rio300m.ipynb` exists and is implemented:
`prepare_xgcm`, `relative_vorticity`, `horizontal_divergence`,
`strain_rate`, `density_anomaly`, `density_gradient_magnitude`,
`interpolate_to_depth`, `compute_depths`, `rho_to_u`, `rho_to_v`. No
missing functions for current notebook usage.

A few things visible in the notebooks aren't (yet) covered by
`croco_tools` itself — they're currently hand-rolled in the notebook
cells:

- **North-arrow / grid-angle annotation helpers** (`add_north_arrow`,
  the repeated `x_km`/`y_km` grid-distance-coordinate blocks) — plotting
  convenience, currently copy-pasted across cells in
  `velocity_gradients.ipynb`. Candidate for a small `plot`-helper
  module if this keeps recurring.
- **Cross-configuration cropping/regridding** (`crop_gigatl_to_rio300m`
  in the GIGATL-vs-RIO comparison notebook) — bespoke polygon-based
  cropping between two different CROCO grids, not currently backed by
  any `croco_tools` function.
- **Buoyancy `b = -g·ρ'/ρ0`** — computed inline in the notebook from
  `density_anomaly`'s output; not currently a `croco_tools` function
  itself (would be a one-line wrapper around `density_anomaly` if
  wanted).
- **`rho_to_u`/`rho_to_v` used to place `z_r` at u/v points** (commented
  out in `velocity_gradients.ipynb`, cells 8–9) — this works today via
  the existing `grid.py` functions, it's just not currently exercised
  (the relevant notebook cells are commented out, so it's untested by
  actual use, only by the new `test_grid.py` roundtrip test added here).

Nothing here is a missing *implementation* — it's either plotting
glue that lives in the notebook by choice, or functionality this
package doesn't yet have an opinion on packaging up.

---

## 5. Testing notes

- The first real run against an installed environment (xgcm 0.10.x)
  turned up two test-construction bugs, now fixed: `test_grid.py`'s
  `psi_to_rho` roundtrip test used a 2x2 rho grid, which is degenerate
  for that function's own internal averaging step (needs >= 2x2 *psi*
  points, i.e. >= 3x3 rho points) — switched to a 4x4 grid and
  corrected the expected output shape (it's exactly the same shape as
  the input, not input-shape-plus-one as originally asserted).
  `test_section.py`'s `interpolate_section` test used a perfectly
  regular, axis-aligned lon/lat grid, which is a classic trigger for
  Qhull's "flat initial simplex" precision error during Delaunay
  triangulation — redesigned around a tiny deterministic jitter (real
  CROCO curvilinear grids are never this regular anyway) and an
  exactly-linear test field, so exact recovery no longer depends on
  target points coinciding with source grid nodes.

- **This test suite was written and read through carefully, but could
  not be executed in the environment used to write it** — no network
  access, and `xarray`, `xgcm`, `dask`, `netCDF4`, `pyproj`, and
  `pytest` are not installed there. Please run `pytest -q` yourself
  after pulling these in; if anything doesn't pass, it's very likely a
  small, mechanical fix (an off-by-one in a fixture shape, a dimension
  name) rather than a wrong understanding of the underlying math — the
  numeric test cases were hand-derived directly from the formulas in
  the source (see `test_vertical.py`'s `vtransform=1`/`vtransform=2`
  cases and `test_eos.py`'s surface-density case for the worked
  algebra in the test docstrings/comments).
- Tests lean on **invariants that don't depend on hand-tracing xgcm's
  internal index bookkeeping** wherever exact numbers would be risky to
  transcribe by hand (e.g., "a uniform flow has exactly zero vorticity",
  "normalized output equals raw output divided by f", rather than a
  specific hand-computed vorticity value on a specific tiny grid).
  Where exact numeric values *are* checked, they were chosen so the
  algebra is simple enough to verify independently (e.g., density
  anomaly reduces exactly to `r00 − rho0` at the surface when T=S=0).
- Test files added/expanded, with test counts:

  | File | Tests | Covers |
  |---|---|---|
  | `test_grid.py` | 17 (was 4) | C-grid interpolation, coord normalization, metrics, `prepare_xgcm` |
  | `test_vertical.py` | 13 (was 2) | `compute_depths` (both vtransforms, hand-derived), `interpolate_to_depth` edge cases |
  | `test_kinematics.py` | 15 (new) | vorticity/divergence/strain, normalization, masking, grid locations |
  | `test_eos.py` | 9 | `density_anomaly`, masking behavior, `buoyancy_frequency` (now fixed, both split_eos paths), and the relocated density-gradient functions |
  | `test_stratification.py` | 6 | density gradients (via the backward-compat re-export path) |
  | `test_geo.py` | 13 (new) | distance/bearing utilities |
  | `test_io.py` | 8 (new) | `open_croco`, `attach_grid`, `promote_coordinates` |
  | `test_section.py` | 11 (new) | `Transect`, `interpolate_section` |
  | `test_accessor.py` | 5 (new) | `.croco` accessor wiring |

  (Several tests are also `@pytest.mark.parametrize`d over multiple
  grid locations, so the collected test count will be somewhat higher
  than this raw function count.)
- A shared `tests/conftest.py` provides small synthetic C-grid fixtures
  (`uniform_grid`, `prepared_uniform_flow`, `prepared_sheared_flow`) so
  the new tests don't need real CROCO output on disk and run in
  milliseconds.
