"""Daily solar and geomagnetic drivers from NASA OMNI via the CDAWeb HAPI server (no login)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

HAPI = "https://cdaweb.gsfc.nasa.gov/hapi/data"


def _hapi(dataset: str, params: list[str], day: pd.Timestamp) -> pd.DataFrame:
    """Fetch one UT day of `params` from a CDAWeb dataset as a DataFrame."""
    end = day + pd.Timedelta(days=1)
    url = (f"{HAPI}?id={dataset}&parameters={','.join(params)}&format=csv"
           f"&time.min={day:%Y-%m-%dT%H:%M:%SZ}&time.max={end:%Y-%m-%dT%H:%M:%SZ}")
    resp = requests.get(url, timeout=(20, 300))
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), header=None, names=["time", *params])


def daily_indices(day) -> dict:
    """F10.7 (daily mean), Kp (daily max) and SYM-H (daily min / max / range) for `day`."""
    day = pd.Timestamp(day).normalize()
    hourly = _hapi("OMNI2_H0_MRG1HR", ["F10_INDEX1800", "KP1800"], day)
    minute = _hapi("OMNI_HRO_1MIN", ["SYM_H"], day)

    # mask OMNI fill values
    f107 = hourly.F10_INDEX1800.where(hourly.F10_INDEX1800 < 999)
    kp = hourly.KP1800.where(hourly.KP1800 < 99) / 10  # OMNI stores Kp x 10; training used Kp
    symh = minute.SYM_H.where(minute.SYM_H.abs() < 99999)

    out = {"f107": f107.mean(), "kp": kp.max(), "sym_h_min": symh.min(),
           "sym_h_max": symh.max(), "sym_h_range": symh.max() - symh.min()}
    if any(np.isnan(v) for v in out.values()):
        raise ValueError(f"OMNI indices incomplete for {day.date()}: {out}")
    return out
