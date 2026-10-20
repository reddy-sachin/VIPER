"""VIPER inference: fetch the model from Hugging Face, scale features, Monte-Carlo-dropout prediction."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.exceptions import InconsistentVersionWarning

HF_REPO = "reddysachin/VIPER"
MODEL, WEIGHTS, SCALER = "VIPER_model.pt", "VIPER_weights.pt", "VIPER_scaler.pkl"

# Column order the scaler was fitted on (it stores no names, so order is everything)
INPUT_COLUMNS = ["mlt", "mlat", "glon", "tec", "peak_n", "mlat_n", "width_n", "peak_s", "mlat_s",
                 "width_s", "trough", "PTC", "DOP", "AOP", "doy", "alt", "f107", "kp",
                 "sym_h_min", "sym_h_max", "sym_h_range"]
GLOBAL_ERROR = 8.3  # test-set MAE [m/s], paper Sec. 4


def _file(path, name: str) -> str:
    """A local `path` if given, otherwise `name` downloaded (once, then cached) from Hugging Face."""
    return str(path) if path else hf_hub_download(HF_REPO, name)


def load_model(model=None, weights=None, device="cpu") -> torch.nn.Module:
    """Load the TorchScript VIPER architecture and its trained weights."""
    net = torch.jit.load(_file(model, MODEL), map_location=device)
    net.load_state_dict(torch.load(_file(weights, WEIGHTS), map_location=device))
    return net.eval()


def load_scaler(scaler=None):
    """Training-time StandardScaler (pickled with sklearn 1.3; only mean_ and scale_ are used)."""
    with warnings.catch_warnings(), open(_file(scaler, SCALER), "rb") as fh:
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return pickle.load(fh)


def add_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    """Append sin/cos encodings of MLT, day-of-year and longitude (paper Sec. 3.2)."""
    df = df.copy()
    for col, name, period, offset in [("mlt", "mlt", 24, 0), ("doy", "doy", 365, 1), ("glon", "lon", 360, 0)]:
        angle = (df[col] - offset) * 2 * np.pi / period
        df[f"{name}_sin"], df[f"{name}_cos"] = np.sin(angle), np.cos(angle)
    return df


def predict(features: pd.DataFrame, net=None, scaler=None, samples=500,
            global_error=GLOBAL_ERROR, mlat_range=(-5, 5)) -> pd.DataFrame:
    """Predict Vz [m/s] for each (mlt, mlat, glon) row of a 21-column feature table.

    vz_pred is the median of `samples` MC-dropout passes; error = MAD + global_error.
    """
    net = load_model() if net is None else net
    df = features[features.mlat.between(*mlat_range)][INPUT_COLUMNS].reset_index(drop=True)
    x = load_scaler(scaler).transform(add_cyclic(df).values)
    x = torch.tensor(x[:, 3:], dtype=torch.float32)  # mlt, mlat, glon are scaled, then dropped

    with torch.no_grad():
        draws = np.stack([net(x).numpy().ravel() for _ in range(samples)])
    median = np.median(draws, axis=0)

    out = df[["mlt", "mlat", "glon"]].copy()
    out["vz_pred"] = median
    out["error"] = np.median(np.abs(draws - median), axis=0) + global_error
    return out
