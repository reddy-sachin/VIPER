"""Download and read one day of JPL Global Ionospheric Maps (JPLD, 96 x 15-min TEC maps)."""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import pandas as pd
import requests
import xarray as xr

JPLD_URL = (
    "https://sideshow.jpl.nasa.gov/pub/iono_daily/gim_for_research/jpld/"
    "{year}/jpld{doy:03d}0.{yy:02d}i.nc.gz"
)
CACHE_DIR = Path(os.environ.get("VIPER_CACHE", Path.home() / ".cache" / "viper"))
J2000 = pd.Timestamp("2000-01-01T12:00:00")  # JPLD epochs are seconds since J2000


def fetch_jpld(day, cache_dir: Path = CACHE_DIR) -> Path:
    """Download (once) the JPLD file for `day` and return the path to the unzipped netCDF.

    A manually downloaded .nc.gz placed in <cache_dir>/jpld/ is used as-is.
    """
    day = pd.Timestamp(day)
    url = JPLD_URL.format(year=day.year, doy=day.dayofyear, yy=day.year % 100)
    gz_path = Path(cache_dir) / "jpld" / Path(url).name
    nc_path = gz_path.with_suffix("")  # strip .gz
    if nc_path.exists():
        return nc_path

    if not gz_path.exists():
        try:
            resp = requests.get(url, timeout=(20, 300), headers={"User-Agent": "Mozilla/5.0 (VIPER)"})
        except requests.ConnectionError as err:
            raise ConnectionError(
                f"Could not reach {url.split('/pub')[0]} ({type(err).__name__}). "
                "If this keeps failing, check your network or proxy settings.\n"
                f"Or download the file by hand into {gz_path.parent}/ :\n  {url}"
            ) from None
        if resp.status_code != 200:
            raise FileNotFoundError(f"No JPLD map for {day.date()} (HTTP {resp.status_code}): {url}")
        gz_path.parent.mkdir(parents=True, exist_ok=True)
        gz_path.write_bytes(resp.content)

    # decompress via a temporary name so an interrupted run never looks cached
    tmp = nc_path.with_suffix(".part")
    tmp.write_bytes(gzip.decompress(gz_path.read_bytes()))
    tmp.rename(nc_path)
    return nc_path


def read_jpld(nc_path: Path) -> xr.DataArray:
    """Return TEC [TECU] as DataArray(time, lat, lon) on the native 1 x 1 deg grid.

    Each epoch keeps its own map and the true cell-centre coordinates (+-89.5, +-179.5).
    """
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        times = J2000 + pd.to_timedelta(ds["time"].values, unit="s")
        return xr.DataArray(
            ds["tecmap"].values,
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": ds["lat"].values, "lon": ds["lon"].values},
            name="tec",
        )
