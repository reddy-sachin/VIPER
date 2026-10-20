"""Gaussian fits of the EIA crests and derived features (paper Sec. 3.1-3.2, Fig. A1).

Thresholds follow the code used to build the training set, which differs slightly
from the paper text (e.g. reduced chi^2 > 3, crest location 7-15 deg MLAT).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lmfit import Parameters, minimize

FIT_MLT = range(9, 22)  # crests are only fitted 09-21 MLT; other hours are zero-filled
NORTH, SOUTH = ["peak_n", "mlat_n", "width_n"], ["peak_s", "mlat_s", "width_s"]


def _residual(p, x, y):
    model = p["peak"] * np.exp(-((x - p["location"]) ** 2) / (2 * p["width"] ** 2))
    return np.log10(model) - np.log10(y)


def _fit_crest(profile: pd.Series, loc: float, loc_min: float, loc_max: float) -> list:
    """Fit one crest; returns [peak, location, width, reduced chi^2 x 1e4]."""
    p = Parameters()
    p.add("peak", value=80, min=10, max=150)
    p.add("location", value=loc, min=loc_min, max=loc_max)
    p.add("width", value=12, min=7, max=20)
    r = minimize(_residual, p, args=(profile.index.values, profile.values),
                 nan_policy="raise", method="leastsq")
    return [r.params["peak"].value, r.params["location"].value, r.params["width"].value, r.redchi * 1e4]


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Reject unphysical crests and compute PTC (peak-to-trough), DOP and AOP."""
    df = df.copy()
    # crest must sit 7-15 deg from the magnetic equator and be < 19 deg wide
    df.loc[(df.mlat_n > 15) | (df.mlat_n < 7) | (df.width_n > 19), NORTH] = 0
    df.loc[(df.mlat_s < -15) | (df.mlat_s > -7) | (df.width_s > 19), SOUTH] = 0
    # quirk kept from training: a poor fit in either hemisphere zeroes the northern crest only
    df.loc[(df.redchi_n > 3) | (df.redchi_s > 3), NORTH] = 0

    two = (df.peak_n > 0) & (df.peak_s > 0)  # double-crested EIA
    df["PTC"] = np.where(two, (df.peak_n + df.peak_s) / 2 / df.trough, 0).round(2)
    df["DOP"] = np.where(two, (df.mlat_n - df.mlat_s).abs(), 0).round(2)
    df["AOP"] = np.where(two, (df.peak_n - df.peak_s) / (2 * (df.peak_n + df.peak_s)) * 100, 0).round(2)
    return df.drop(columns=["redchi_n", "redchi_s"])


def fit_eia(grid: pd.DataFrame) -> pd.DataFrame:
    """EIA features per (mlt, glon) for a daily grid from `tec.to_magnetic_grid`.

    Bins whose MLAT profile contains NaNs fail the fit and are dropped, as in training.
    """
    cols = NORTH + ["redchi_n"] + SOUTH + ["redchi_s", "trough"]
    rows = []
    for (mlt, glon), g in grid.groupby(["mlt", "glon"]):
        values = [0.0] * len(cols)
        if mlt in FIT_MLT:
            prof = g.set_index("mlat")["tec"]
            try:
                values = (_fit_crest(prof.loc[4:25], 15, 6, 25)
                          + _fit_crest(prof.loc[-25:-4], -15, -25, -6) + [prof.loc[0]])
            except Exception:
                continue
        rows.append([mlt, glon, *values])
    return classify(pd.DataFrame(rows, columns=["mlt", "glon", *cols]))
