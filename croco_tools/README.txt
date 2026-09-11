CROCO tools grid compatibility fix

This grid.py preserves the original public functions used by accessor.py and
__init__.py:
  rho_to_u, rho_to_v, rho_to_psi, u_to_rho, v_to_rho, psi_to_rho,
  rotate_velocity

It also includes the new xgcm infrastructure:
  normalize_croco_coords, merge_grid, add_horizontal_metrics,
  build_xgcm_grid, prepare_xgcm

Copy grid.py into the croco_tools package. The other files are included so the
horizontal diagnostic split remains internally consistent.
