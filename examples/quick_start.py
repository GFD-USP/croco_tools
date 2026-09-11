"""Minimal RIO1km analysis example."""

from pathlib import Path

import croco_tools


root = Path(
    "/home/crocha/RESEARCH/CROCO-RIO/RIO_1km"
)

dataset = croco_tools.open_croco(
    root / "HIS/rio1km_his.nc",
    grid=root / "CROCO_FILES/rio1km_grd.nc",
)

z_rho, z_w = dataset.croco.depths()

u_east, v_north = dataset.croco.velocity()

temperature_100m = dataset.croco.to_depth(
    "temp",
    depths=-100.0,
)

print(dataset)
print(z_rho)
print(z_w)
print(u_east)
print(v_north)
print(temperature_100m)
