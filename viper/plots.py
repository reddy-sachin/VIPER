"""Figures: paper-style curves, all eight sectors, an MLT x longitude heatmap, and a world map."""

from __future__ import annotations

import apexpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from .tec import SECTORS, sector_of

try:  # coastlines are optional: pip install cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    ccrs = cfeature = None

PASTEL = ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b"]  # seaborn "pastel", as in the paper
CMAP = "RdBu_r"  # red = upward, blue = downward drift
OCEAN, LAND, COAST = "#dce8f2", "#efe9df", "#9aa7b1"
NIGHT = (0.04, 0.10, 0.17)  # deep blue, alpha-blended over the basemap
TWILIGHT_DEPTH = 18.0  # solar elevation [deg] at which night shading saturates


def _band_mean(pred: pd.DataFrame) -> pd.DataFrame:
    """Vz and error averaged over the -5..5 deg MLAT band, per (glon, mlt)."""
    return pred.groupby(["glon", "mlt"], as_index=False)[["vz_pred", "error"]].mean()


def _curve(ax, s: pd.DataFrame, color, label=None) -> None:
    """Vz vs MLT with its error band."""
    ax.plot(s.mlt, s.vz_pred, color=color, label=label)
    ax.fill_between(s.mlt, s.vz_pred - s.error, s.vz_pred + s.error, alpha=0.3, color=color)
    ax.axhline(0, color="k", alpha=0.5, linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def plot_vz(pred: pd.DataFrame, title: str):
    """The four western-hemisphere sectors on one axis (paper Fig. 8)."""
    mean = _band_mean(pred)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for color, glon in zip(PASTEL, SECTORS[:4]):
        _curve(ax, mean[mean.glon == glon], color, f"{glon}°")
    ax.set(title=title, xlabel="MLT [hr]", ylabel="Vz [m/s]", xticks=np.arange(0, 25, 3))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def plot_sectors(pred: pd.DataFrame, title: str):
    """One panel per longitude sector, all eight, on shared axes."""
    mean = _band_mean(pred)
    fig, axs = plt.subplots(2, 4, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, glon in zip(axs.flat, SECTORS):
        _curve(ax, mean[mean.glon == glon], "C0")
        ax.set_title(f"{glon}°", fontsize=10)
        ax.set_xticks(np.arange(0, 25, 6))
    for ax in axs[:, 0]:
        ax.set_ylabel("Vz [m/s]")
    for ax in axs[1]:
        ax.set_xlabel("MLT [hr]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_heatmap(pred: pd.DataFrame, title: str):
    """Vz on the MLT x longitude-sector grid; empty cells are bins with no TEC/EIA fit."""
    z = (_band_mean(pred).pivot(index="glon", columns="mlt", values="vz_pred")
         .reindex(index=SECTORS, columns=range(24)).values)
    vmax = np.nanmax(np.abs(z))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.pcolormesh(np.arange(25), np.arange(9), z, cmap=CMAP, vmin=-vmax, vmax=vmax)
    ax.set(title=title, xlabel="MLT [hr]", ylabel="Longitude sector", xticks=np.arange(0, 25, 3))
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_yticklabels([f"{g}°" for g in SECTORS])
    fig.colorbar(im, ax=ax, label="Vz [m/s]")
    fig.tight_layout()
    return fig


def _solar(t: pd.Timestamp) -> tuple[float, float]:
    """Subsolar latitude and longitude [deg]; mean sun, no equation of time."""
    decl = np.degrees(np.radians(23.44) * np.sin(2 * np.pi * (284 + t.dayofyear) / 365))
    return decl, -15 * (t.hour + t.minute / 60 - 12)


def _solar_elevation(t: pd.Timestamp, n: int = 721) -> tuple[np.ndarray, float, float]:
    """Solar elevation [deg] on a global grid, plus the subsolar latitude and longitude."""
    lon, lat = np.meshgrid(np.linspace(-180, 180, n), np.linspace(90, -90, n // 2))
    sublat, sublon = _solar(t)
    cos_sza = (np.sin(np.radians(lat)) * np.sin(np.radians(sublat))
               + np.cos(np.radians(lat)) * np.cos(np.radians(sublat))
               * np.cos(np.radians(lon - sublon)))
    return np.degrees(np.arcsin(np.clip(cos_sza, -1, 1))), sublat, sublon


def _sun_overlay(ax, t: pd.Timestamp, **kw) -> list:
    """Smooth dusk-to-night gradient, a warm terminator line and the subsolar point."""
    elev, sublat, sublon = _solar_elevation(t)

    # alpha ramps from 0 at sunset to full at astronomical night: a soft twilight band
    rgba = np.zeros((*elev.shape, 4))
    rgba[..., :3] = NIGHT
    rgba[..., 3] = 0.46 * np.clip(-elev / TWILIGHT_DEPTH, 0, 1) ** 0.65

    art = [ax.imshow(rgba, extent=(-180, 180, -90, 90), interpolation="bilinear",
                     zorder=2, **kw)]
    # draw the terminator only away from the poles, where it would otherwise close
    # along the map edges and read as a box
    lat1d, keep = np.linspace(90, -90, elev.shape[0]), 80
    m = np.abs(lat1d) <= keep
    art.append(ax.contour(np.linspace(-180, 180, elev.shape[1]), lat1d[m], elev[m],
                          levels=[0], colors="#f0b957", linewidths=1.1, zorder=3, **kw))
    art += ax.plot(sublon, sublat, marker="o", markersize=8, color="#ffd572",
                   markeredgecolor="#c9922f", markeredgewidth=0.8, zorder=4, **kw)
    return art


def _world(pred: pd.DataFrame, day, figsize=(11, 6.4)):
    """Figure plus a draw(ut_hours) function painting Vz on the +-5 deg apex-MLAT band."""
    day = pd.Timestamp(day).normalize()
    apex = apexpy.Apex(day.year, refh=0)

    # where the magnetic band actually lies on the globe: apexpy inverts IGRF for us
    mlat, mlon = np.meshgrid(np.arange(-5, 6), np.arange(-180, 180), indexing="ij")
    glat, glon, _ = apex.apex2geo(mlat, mlon, height=0)
    isec = np.searchsorted(SECTORS, sector_of(glon.ravel()))
    eq = np.argsort(glon[5])  # mlat = 0 row, sorted so the equator draws as one line

    # prediction cube [sector, mlat, mlt]; missing bins stay NaN and are drawn transparent
    cube = (pred.set_index(["glon", "mlat", "mlt"]).vz_pred
            .reindex(pd.MultiIndex.from_product([SECTORS, range(-5, 6), range(24)]))
            .values.reshape(8, 11, 24))

    kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={"projection": ccrs.PlateCarree()} if ccrs else {})
    norm = plt.Normalize(-np.nanmax(np.abs(cube)), np.nanmax(np.abs(cube)))
    # cartopy locks the axes to the map's 2:1 aspect, so the figure is sized to match
    # (plus room for the colourbar) to avoid wide empty margins
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax,
                       orientation="horizontal", fraction=0.05, pad=0.04,
                       shrink=0.45, aspect=38)
    bar.set_label("Upward plasma drift, Vz [m/s]", fontsize=9)
    bar.outline.set_visible(False)

    # static basemap, drawn once; only the sun overlay and drift band change per frame
    if ccrs:
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=LAND, zorder=0)
        ax.coastlines(linewidth=0.4, color=COAST, zorder=1)
    else:
        ax.set(xlim=(-180, 180), ylim=(-90, 90), facecolor=OCEAN,
               xlabel="Geographic longitude [°]", ylabel="Geographic latitude [°]")
    ax.plot(glon[5][eq], glat[5][eq], color="#44515c", linewidth=0.7,
            linestyle=(0, (4, 3)), zorder=5, **kw)
    title = ax.set_title("", fontsize=12, pad=8)
    frame: list = []

    def draw(ut_hours: float):
        t = day + pd.Timedelta(hours=float(ut_hours))
        mlt = apex.mlon2mlt(mlon, t.to_pydatetime()).astype(int) % 24  # integer bins, as in training
        vz = cube[isec, mlat.ravel() + 5, mlt.ravel()]

        while frame:  # clear only the time-varying artists
            a = frame.pop()
            (a.remove() if hasattr(a, "remove") else None)
        frame.extend(_sun_overlay(ax, t, **kw))
        # soft glow under the band, then the band itself
        frame.append(ax.scatter(glon.ravel(), glat.ravel(), c=vz, cmap=CMAP, norm=norm,
                                s=95, marker="o", alpha=0.11, linewidths=0, zorder=6, **kw))
        frame.append(ax.scatter(glon.ravel(), glat.ravel(), c=vz, cmap=CMAP, norm=norm,
                                s=12, marker="s", linewidths=0, zorder=7, **kw))
        title.set_text(f"VIPER predicted equatorial drift   {t:%Y-%m-%d  %H:%M} UT")

    return fig, draw


def plot_world(pred: pd.DataFrame, day, ut: float = 0.0):
    """World map of predicted Vz along the magnetic equator at one UT (hours)."""
    fig, draw = _world(pred, day)
    draw(ut)
    return fig


def animate_world(pred: pd.DataFrame, day, path, step: float = 0.5, fps: float = 6,
                  dpi: int = 100) -> None:
    """Animate the world map through the UT day, every `step` hours.

    An .mp4 path uses ffmpeg (crisp, small); .gif falls back to Pillow, which
    quantises to 256 colours. For a high-quality GIF, write .mp4 first and
    convert it with ffmpeg's palettegen filter.
    """
    fig, draw = _world(pred, day)
    anim = FuncAnimation(fig, draw, frames=np.arange(0, 24, step))
    writer = FFMpegWriter(fps=fps, bitrate=-1) if str(path).endswith(".mp4") else PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
