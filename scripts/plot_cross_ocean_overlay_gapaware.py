from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd


BASE = Path(
    "outputs/runs/adsc_cross_ocean_adsc_adsb_selected10_20260428"
)
OUT_DIR = BASE / "cross_ocean_overlay_gapaware_20260429"


def build_adsb_plot_parts(adsb_g: pd.DataFrame, adsc_start: pd.Timestamp, adsc_end: pd.Timestamp):
    before = adsb_g[adsb_g["minute_ts"] < adsc_start].copy()
    inside = adsb_g[adsb_g["minute_ts"].between(adsc_start, adsc_end, inclusive="both")].copy()
    after = adsb_g[adsb_g["minute_ts"] > adsc_end].copy()
    return before, inside, after


def main():
    pairs = pd.read_csv(BASE / "selected_10_cross_ocean_adsc_adsb_pairs.csv")
    pairs["adsc_start"] = pd.to_datetime(pairs["adsc_start"], utc=True)
    pairs["adsc_end"] = pd.to_datetime(pairs["adsc_end"], utc=True)
    pairs["flight_start"] = pd.to_datetime(pairs["flight_start"], utc=True)
    pairs["flight_end"] = pd.to_datetime(pairs["flight_end"], utc=True)

    adsc = pd.read_csv(BASE / "selected_10_cross_ocean_adsc_points.csv")
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True)

    adsb = pd.read_csv(BASE / "selected_10_cross_ocean_adsb_minute_full_flights.csv")
    adsb["minute_ts"] = pd.to_datetime(adsb["minute_ts"], utc=True)

    csv_dir = OUT_DIR / "per_pair_overlay_csv"
    plot2d_dir = OUT_DIR / "plots_2d_alt"
    plot3d_dir = OUT_DIR / "plots_3d"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot2d_dir.mkdir(parents=True, exist_ok=True)
    plot3d_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_overlay = []

    for _, row in pairs.iterrows():
        pair_id = str(row["pair_id"])
        adsb_flight_id = str(row["flight_id"])
        adsc_g = (
            adsc[adsc["pair_id"].astype(str) == pair_id]
            .copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        adsb_g = (
            adsb[adsb["pair_id"].astype(str) == pair_id]
            .copy()
            .sort_values("minute_ts")
            .reset_index(drop=True)
        )
        if adsc_g.empty or adsb_g.empty:
            continue

        adsc_part = pd.DataFrame(
            {
                "pair_id": pair_id,
                "icao24": str(row["icao24"]).lower(),
                "adsb_flight_id": adsb_flight_id,
                "plot_ts": adsc_g["timestamp"],
                "source": "adsc_anchor",
                "lat": pd.to_numeric(adsc_g["latitude"], errors="coerce"),
                "lon": pd.to_numeric(adsc_g["longitude"], errors="coerce"),
                "alt": pd.to_numeric(adsc_g["altitude_m"], errors="coerce"),
                "num_points_in_minute": pd.NA,
                "in_adsc_window": 1,
                "is_adsc_anchor": 1,
            }
        )

        adsb_part = adsb_g[
            ["pair_id", "adsb_icao", "flight_id", "minute_ts", "lat", "lon", "alt", "num_points_in_minute"]
        ].copy()
        adsb_part = adsb_part.rename(
            columns={"adsb_icao": "icao24", "flight_id": "adsb_flight_id", "minute_ts": "plot_ts"}
        )
        adsb_part["source"] = "adsb_minute"
        adsb_part["in_adsc_window"] = (
            adsb_part["plot_ts"]
            .between(row["adsc_start"].floor("min"), row["adsc_end"].ceil("min"), inclusive="both")
            .astype(int)
        )
        adsb_part["is_adsc_anchor"] = 0
        adsb_part = adsb_part[
            [
                "pair_id",
                "icao24",
                "adsb_flight_id",
                "plot_ts",
                "source",
                "lat",
                "lon",
                "alt",
                "num_points_in_minute",
                "in_adsc_window",
                "is_adsc_anchor",
            ]
        ]

        overlay = pd.concat([adsb_part, adsc_part], ignore_index=True).sort_values("plot_ts").reset_index(drop=True)
        overlay["adsc_start_ts"] = row["adsc_start"]
        overlay["adsc_end_ts"] = row["adsc_end"]
        overlay["adsb_flight_start_ts"] = row["flight_start"]
        overlay["adsb_flight_end_ts"] = row["flight_end"]
        overlay["distance_km"] = row["distance_km"]
        overlay["dep_airport_icao"] = row["dep_airport_icao"]
        overlay["arr_airport_icao"] = row["arr_airport_icao"]
        overlay["adsc_points"] = row["adsc_points"]
        overlay_csv = csv_dir / f"{pair_id}_overlay.csv"
        overlay.to_csv(overlay_csv, index=False)
        all_overlay.append(overlay)

        before, inside, after = build_adsb_plot_parts(adsb_g, row["adsc_start"], row["adsc_end"])
        inside_count = int(len(inside))
        has_gap = inside_count == 0

        x0 = adsb_g["minute_ts"].min()
        x_adsc = (adsc_g["timestamp"] - x0).dt.total_seconds() / 60.0
        left = (row["adsc_start"] - x0).total_seconds() / 60.0
        right = (row["adsc_end"] - x0).total_seconds() / 60.0

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
        ax.set_facecolor("white")

        if has_gap:
            if not before.empty:
                x_before = (before["minute_ts"] - x0).dt.total_seconds() / 60.0
                ax.plot(
                    x_before,
                    pd.to_numeric(before["alt"], errors="coerce"),
                    color="#1f77b4",
                    lw=1.8,
                    alpha=0.95,
                    label="ADS-B minute",
                )
                last_before = before.tail(1)
                x_last = (last_before["minute_ts"] - x0).dt.total_seconds() / 60.0
                ax.scatter(
                    x_last,
                    pd.to_numeric(last_before["alt"], errors="coerce"),
                    color="#1f77b4",
                    s=26,
                    alpha=0.98,
                    zorder=7,
                    label="ADS-B edge points",
                )
            if not after.empty:
                x_after = (after["minute_ts"] - x0).dt.total_seconds() / 60.0
                ax.plot(
                    x_after,
                    pd.to_numeric(after["alt"], errors="coerce"),
                    color="#1f77b4",
                    lw=1.8,
                    alpha=0.95,
                )
                first_after = after.head(1)
                x_first = (first_after["minute_ts"] - x0).dt.total_seconds() / 60.0
                ax.scatter(
                    x_first,
                    pd.to_numeric(first_after["alt"], errors="coerce"),
                    color="#1f77b4",
                    s=26,
                    alpha=0.98,
                    zorder=7,
                )
        else:
            x_adsb = (adsb_g["minute_ts"] - x0).dt.total_seconds() / 60.0
            ax.plot(
                x_adsb,
                pd.to_numeric(adsb_g["alt"], errors="coerce"),
                color="#1f77b4",
                lw=1.8,
                alpha=0.95,
                label="ADS-B minute",
            )

        ax.scatter(
            x_adsc,
            pd.to_numeric(adsc_g["altitude_m"], errors="coerce"),
            color="#d95f02",
            s=30,
            alpha=0.98,
            label="ADS-C anchors",
            zorder=6,
        )
        ax.axvspan(left, right, color="#f6d55c", alpha=0.14, label="ADS-C window")
        ax.set_title(f"{pair_id} | Cross-ocean ADS-B full flight with ADS-C anchor overlay")
        ax.set_xlabel("Minutes from ADS-B flight start")
        ax.set_ylabel("Altitude (m)")
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=3, loc="best")
        fig.tight_layout()
        fig.savefig(plot2d_dir / f"{pair_id}_alt_2d_overlay.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig = plt.figure(figsize=(11.2, 7.4), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")

        if has_gap:
            if not before.empty:
                ax.plot(before["lon"], before["lat"], before["alt"], color="#1f77b4", lw=1.6, alpha=0.9, label="ADS-B minute")
                last_before = before.tail(1)
                ax.scatter(last_before["lon"], last_before["lat"], last_before["alt"], color="#1f77b4", s=24, alpha=0.98, zorder=7, label="ADS-B edge points")
            if not after.empty:
                ax.plot(after["lon"], after["lat"], after["alt"], color="#1f77b4", lw=1.6, alpha=0.9)
                first_after = after.head(1)
                ax.scatter(first_after["lon"], first_after["lat"], first_after["alt"], color="#1f77b4", s=24, alpha=0.98, zorder=7)
        else:
            ax.plot(adsb_g["lon"], adsb_g["lat"], adsb_g["alt"], color="#1f77b4", lw=1.6, alpha=0.9, label="ADS-B minute")

        ax.scatter(
            adsc_g["longitude"],
            adsc_g["latitude"],
            adsc_g["altitude_m"],
            color="#d95f02",
            s=28,
            alpha=0.98,
            label="ADS-C anchors",
            zorder=6,
        )

        lon = pd.to_numeric(adsb_g["lon"], errors="coerce")
        lat = pd.to_numeric(adsb_g["lat"], errors="coerce")
        alt = pd.to_numeric(adsb_g["alt"], errors="coerce")
        lon_span = float(max(lon.max() - lon.min(), 1e-6))
        lat_span = float(max(lat.max() - lat.min(), 1e-6))
        alt_span = float(max(alt.max() - alt.min(), 1e-6))
        mean_lat = float(lat.mean()) if len(lat) else 0.0
        lon_m = lon_span * 111000.0 * max(0.2, abs(math.cos(math.radians(mean_lat))))
        lat_m = lat_span * 111000.0
        z_display = max(alt_span * 0.22, min(lon_m, lat_m) * 0.45, 1.0)

        ax.set_title(f"{pair_id} | 3D cross-ocean trajectory overlay")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude (m)")
        ax.view_init(elev=20, azim=-122)
        ax.set_proj_type("persp")
        ax.set_box_aspect((max(lon_m, 1.0), max(lat_m, 1.0), z_display))
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(plot3d_dir / f"{pair_id}_3d_overlay.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        summary_rows.append(
            {
                "pair_id": pair_id,
                "icao24": row["icao24"],
                "adsb_flight_id": adsb_flight_id,
                "distance_km": row["distance_km"],
                "dep_airport_icao": row["dep_airport_icao"],
                "arr_airport_icao": row["arr_airport_icao"],
                "adsc_points": int(row["adsc_points"]),
                "adsb_minute_rows": int(len(adsb_g)),
                "adsb_rows_inside_adsc_window": inside_count,
                "gap_aware_break_applied": int(has_gap),
                "overlay_csv": str(overlay_csv),
                "plot_2d": str(plot2d_dir / f"{pair_id}_alt_2d_overlay.png"),
                "plot_3d": str(plot3d_dir / f"{pair_id}_3d_overlay.png"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "overlay_summary.csv", index=False)
    if all_overlay:
        pd.concat(all_overlay, ignore_index=True).to_csv(OUT_DIR / "overlay_all_10.csv", index=False)


if __name__ == "__main__":
    main()
