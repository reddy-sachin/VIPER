"""Command line: `viper 2014-03-16` or `viper --input features.csv`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from . import model, plots
from .pipeline import DisturbedDayError, build_features


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Predict equatorial vertical plasma drift (Vz) with VIPER.")
    p.add_argument("date", nargs="?", help="UT day to predict, YYYY-MM-DD (downloads all inputs).")
    p.add_argument("--input", help="Skip downloads and use a prepared 21-column feature CSV.")
    p.add_argument("--out", default="output", help="Output directory (default: ./output).")
    p.add_argument("--alt", type=float, default=500.0, help="Altitude feature [km] (default 500).")
    p.add_argument("--samples", type=int, default=500, help="MC-dropout samples (default 500).")
    p.add_argument("--gif", nargs="?", const="gif", choices=["gif", "mp4"],
                   help="Also animate the world map: 'gif', or 'mp4' for HD (needs ffmpeg).")
    p.add_argument("--model", help="Local model file (default: download from Hugging Face).")
    p.add_argument("--weights", help="Local weights file (default: Hugging Face).")
    p.add_argument("--scaler", help="Local scaler file (default: Hugging Face).")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if not (args.date or args.input):
        p.error("give a DATE or --input CSV")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(args.date) if args.date else None
    if args.input:  # with --input, DATE is optional and only used for the world map
        features, tag = pd.read_csv(args.input), Path(args.input).stem
    else:
        tag = str(day.date())
        print(f"Building features for {tag} (OMNI Kp check -> JPLD GIM -> apex -> EIA fit) ...")
        try:
            features = build_features(day, alt=args.alt)
        except DisturbedDayError as err:
            raise SystemExit(f"Rejected: {err}")
        features.to_csv(out / f"{tag}_features.csv", index=False)

    net = model.load_model(args.model, args.weights, device=args.device)
    pred = model.predict(features, net=net, scaler=args.scaler, samples=args.samples)
    pred.to_csv(out / f"{tag}_vz.csv", index=False)

    figures = {"vz": plots.plot_vz(pred, tag), "sectors": plots.plot_sectors(pred, tag),
               "heatmap": plots.plot_heatmap(pred, tag)}
    if day is not None:  # the map needs the date for apex coordinates and day/night
        figures["world"] = plots.plot_world(pred, day, ut=0)
    for name, fig in figures.items():
        fig.savefig(out / f"{tag}_{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    if args.gif and day is not None:
        plots.animate_world(pred, day, out / f"{tag}_world.{args.gif}",
                            **({"step": 0.25, "fps": 12, "dpi": 170} if args.gif == "mp4" else {}))
    print(f"Wrote {tag}_vz.csv and {', '.join(figures)} figures to {out}/")


if __name__ == "__main__":
    main()
