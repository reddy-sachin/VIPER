"""Day -> ML-ready features -> Vz prediction (the full Fig. A1 chain)."""

from __future__ import annotations

import pandas as pd

from .download import fetch_jpld, read_jpld
from .eia import fit_eia
from .indices import daily_indices
from .model import INPUT_COLUMNS, load_model, predict
from .tec import to_magnetic_grid


KP_MAX = 3  # VIPER was trained on geomagnetically quiet days only (daily max Kp <= 3)


class DisturbedDayError(ValueError):
    """Raised when a day is outside the quiet-time conditions VIPER was trained on."""


def build_features(day, alt: float = 500.0) -> pd.DataFrame:
    """Build the 21-column VIPER input table for one UT day from public data."""
    day = pd.Timestamp(day)

    # check geomagnetic activity first, so disturbed days are rejected before any TEC download
    indices = daily_indices(day)
    if indices["kp"] > KP_MAX:
        raise DisturbedDayError(f"{day.date()}: daily max Kp = {indices['kp']:.1f} > {KP_MAX}; "
                                "VIPER is only valid for quiet days.")

    grid = to_magnetic_grid(read_jpld(fetch_jpld(day)))

    # EIA features are per (mlt, glon) and are broadcast over every mlat;
    # bins with gaps or failed fits are dropped, as in training
    df = grid.merge(fit_eia(grid), on=["mlt", "glon"]).dropna()
    df = df.assign(doy=day.dayofyear, alt=alt, **indices)
    return df[INPUT_COLUMNS].sort_values(["glon", "mlat", "mlt"]).reset_index(drop=True)


def run_day(day, alt: float = 500.0, **predict_kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, predictions) for one day."""
    features = build_features(day, alt=alt)
    return features, predict(features, **predict_kw)


def run_days(days, alt: float = 500.0, net=None, **predict_kw) -> pd.DataFrame:
    """Predictions for many days, with a `date` column; disturbed or missing days are skipped.

    Loads the network once and reuses it, e.g. run_days(pd.date_range("2014-03-01", periods=31)).
    """
    net = load_model() if net is None else net
    out = []
    for day in pd.to_datetime(list(days)):
        try:
            features = build_features(day, alt=alt)
        except (DisturbedDayError, FileNotFoundError, ConnectionError, ValueError) as err:
            print(f"skipped {day.date()}: {err}")
            continue
        out.append(predict(features, net=net, **predict_kw).assign(date=day.normalize()))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
