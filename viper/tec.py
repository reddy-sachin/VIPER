"""Bin geographic TEC maps onto the daily magnetic grid VIPER was trained on.

Conventions below (integer truncation of MLAT/MLT, nearest-centre sectors) are kept
exactly as in the training pipeline, because the model weights depend on them.
"""

from __future__ import annotations

import apexpy
import numpy as np
import pandas as pd
import xarray as xr

# Longitude sector centres (Fejer et al., 2008); 195 - 22.5 = 172.5 is the W. Pacific sector.
# Assignment is nearest-centre with no 360 deg wrap, as in training.
SECTORS = np.sort(np.array([195, -120, -75, -30, 15, 60, 105, 150]) - 22.5)
MLAT_GRID = np.arange(-29, 30)
MLT_GRID = np.arange(24)


def sector_of(lon: np.ndarray) -> np.ndarray:
    """Map geographic longitude [deg] to the nearest sector centre."""
    return SECTORS[np.abs(np.asarray(lon)[:, None] - SECTORS).argmin(axis=1)]


def to_magnetic_grid(tec: xr.DataArray) -> pd.DataFrame:
    """Daily-mean TEC binned by integer apex MLAT, integer MLT and longitude sector.

    Returns a full 8 x 59 x 24 grid with columns (mlat, mlt, glon, tec); empty bins are NaN.
    """
    tec = tec.where(np.abs(tec.lat) <= 35, drop=True)
    times = pd.to_datetime(tec.time.values).to_pydatetime()
    apex = apexpy.Apex(times[0].year, refh=0)

    # apex MLAT does not depend on time for a fixed epoch, so convert the grid once ...
    lat2d, lon2d = np.meshgrid(tec.lat.values, tec.lon.values, indexing="ij")
    mlat, mlon = apex.geo2apex(lat2d, lon2d, height=0)
    # ... while MLT changes with every 15-min map
    mlt = np.stack([apex.mlon2mlt(mlon, t) for t in times])

    df = pd.DataFrame({
        "mlat": np.broadcast_to(mlat, tec.shape).ravel().astype(int),
        "mlt": mlt.ravel().astype(int),
        "glon": np.broadcast_to(sector_of(tec.lon.values), tec.shape).ravel(),
        "tec": tec.values.ravel(),
    })
    daily = df.groupby(["glon", "mlat", "mlt"])["tec"].mean()
    full = pd.MultiIndex.from_product([SECTORS, MLAT_GRID, MLT_GRID], names=daily.index.names)
    return daily.reindex(full).reset_index()
