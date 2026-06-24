from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "ADS-C anchors": {"color": "black", "marker": "*", "s": 120, "linestyle": "None"},
    "BiMamba": {"color": "red", "linestyle": "-", "linewidth": 2.2},
    "UniLSTM-proto": {"color": "tab:blue", "linestyle": "-", "linewidth": 1.6},
    "BiLSTM-proto": {"color": "tab:green", "linestyle": "-", "linewidth": 1.6},
    "CNN-LSTM-proto": {"color": "tab:purple", "linestyle": "-", "linewidth": 1.6},
    "Transformer-proto": {"color": "tab:orange", "linestyle": "-", "linewidth": 1.8},
}

MODELS = [
    "BiMamba",
    "UniLSTM-proto",
    "BiLSTM-proto",
    "CNN-LSTM-proto",
    "Transformer-proto",
]


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "savefig.dpi": 600,
            "axes.linewidth": 1.2,
        }
    )


def _plot_one(csv_path: Path, out_base: Path) -> None:
    df = pd.read_csv(csv_path)
    anchors = df["is_adsc_anchor"].astype(int) == 1

    fig = plt.figure(figsize=(11.5, 8.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    ax.scatter(
        pd.to_numeric(df.loc[anchors, "obs_lon"], errors="coerce"),
        pd.to_numeric(df.loc[anchors, "obs_lat"], errors="coerce"),
        pd.to_numeric(df.loc[anchors, "obs_alt"], errors="coerce"),
        label="ADS-C anchors",
        zorder=10,
        color=STYLE["ADS-C anchors"]["color"],
        marker=STYLE["ADS-C anchors"]["marker"],
        s=STYLE["ADS-C anchors"]["s"],
    )

    for name in MODELS:
        ax.plot(
            pd.to_numeric(df[f"{name}_pred_lon"], errors="coerce"),
            pd.to_numeric(df[f"{name}_pred_lat"], errors="coerce"),
            pd.to_numeric(df[f"{name}_pred_alt"], errors="coerce"),
            label=name,
            zorder=8 if name == "BiMamba" else 5,
            **STYLE[name],
        )

    fid = str(df["flight_id"].iloc[0])
    ax.set_title(f"3D Trajectory Recovery Comparison: {fid}", pad=20, fontsize=15, fontweight="semibold")
    ax.set_xlabel("Longitude", fontsize=13, labelpad=12, fontweight="semibold")
    ax.set_ylabel("Latitude", fontsize=13, labelpad=12, fontweight="semibold")
    ax.set_zlabel("Altitude (m)", fontsize=13, labelpad=14, fontweight="semibold")
    ax.view_init(elev=21, azim=-124)
    ax.set_box_aspect((1.45, 1.10, 0.95))
    ax.grid(True, alpha=0.34)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
        axis.pane.set_edgecolor((0.45, 0.45, 0.45, 1.0))
        axis._axinfo["grid"]["color"] = (0.70, 0.70, 0.70, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.95
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["axisline"]["color"] = (0.25, 0.25, 0.25, 1.0)
        axis._axinfo["axisline"]["linewidth"] = 1.2
    for axis_name in ["xaxis", "yaxis", "zaxis"]:
        axis_obj = getattr(ax, axis_name)
        if hasattr(axis_obj, "line"):
            axis_obj.line.set_color((0.2, 0.2, 0.2, 1.0))
            axis_obj.line.set_linewidth(1.3)
    ax.tick_params(axis="x", labelsize=14, width=1.2, length=5.5, pad=4)
    ax.tick_params(axis="y", labelsize=14, width=1.2, length=5.5, pad=4)
    ax.tick_params(axis="z", labelsize=14, width=1.2, length=5.5, pad=6)
    ax.legend(
        fontsize=10.5,
        loc="upper left",
        bbox_to_anchor=(0.72, 0.92),
        borderaxespad=0.2,
        framealpha=0.90,
        facecolor=(1.0, 1.0, 1.0, 0.90),
        edgecolor=(0.35, 0.35, 0.35, 1.0),
        borderpad=0.5,
        labelspacing=0.35,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.90)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="outputs/runs/real_adsc_anchor_only_current_models_20260531",
    )
    ap.add_argument("--pair-ids", default="", help="Comma-separated pair_ids. Empty means all.")
    args = ap.parse_args()

    _set_plot_style()
    base_dir = Path(args.base_dir)
    if args.pair_ids.strip():
        wanted = {x.strip() for x in args.pair_ids.split(",") if x.strip()}
        csvs = [base_dir / pid / "recovered_minute_compare_current_models.csv" for pid in sorted(wanted)]
    else:
        csvs = sorted(base_dir.glob("*/recovered_minute_compare_current_models.csv"))

    for csv_path in csvs:
        if not csv_path.exists():
            continue
        pair_id = csv_path.parent.name
        _plot_one(csv_path, csv_path.parent / f"real_adsc_3d_compare_{pair_id}")

    print(base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
