#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASET_PATH = Path(
    "/Users/natomanzolli/Downloads/DonneesOuvertes2025_010203040506070809101112.csv"
)
REPO_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "case_study"
PAPER_OUTPUT_DIR = Path(
    "/Users/natomanzolli/Downloads/A Multi-Objective Framework for Bicycle Infrastructure and Bike-Sharing Stations Design/Figures"
)
SUMMARY_PATH = REPO_OUTPUT_DIR / "bixi_2025_summary.json"
STREET_NETWORK_PATH = Path(__file__).resolve().parents[1] / "RuesEtSentiers.gpkg"
PAPER_FIGURES = {
    "fig6a": "BIXI_case_top30_station_demand.pdf",
    "fig6b": "BIXI_case_top30_station_map.pdf",
    "fig6c": "BIXI_case_borough_activity.pdf",
    "fig7a": "BIXI_case_daily_ridership.pdf",
    "fig7b": "BIXI_case_monthly_ridership.pdf",
    "fig7c": "BIXI_case_weekday_hour_heatmap.pdf",
}


def ensure_dirs() -> None:
    REPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#4a4a4a",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def build_summary() -> dict:
    total_rows = 0
    daily_counts: Counter = Counter()
    monthly_counts: Counter = Counter()
    weekday_hour = np.zeros((7, 24), dtype=np.int64)
    start_borough_counts: Counter = Counter()
    end_borough_counts: Counter = Counter()
    start_station_counts: Counter = Counter()
    end_station_counts: Counter = Counter()
    station_coords: dict[str, tuple[float, float]] = {}

    utc = timezone.utc
    dst_start_utc = int(datetime(2025, 3, 9, 7, 0, tzinfo=utc).timestamp())
    dst_end_utc = int(datetime(2025, 11, 2, 6, 0, tzinfo=utc).timestamp())
    day_cache: dict[int, tuple[str, str, int]] = {}

    with DATASET_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1

            station_name = row["STARTSTATIONNAME"]
            if station_name:
                start_station_counts[station_name] += 1
                if station_name not in station_coords and row["STARTSTATIONLATITUDE"] and row["STARTSTATIONLONGITUDE"]:
                    station_coords[station_name] = (
                        float(row["STARTSTATIONLONGITUDE"]),
                        float(row["STARTSTATIONLATITUDE"]),
                    )

            end_station_name = row["ENDSTATIONNAME"]
            if end_station_name:
                end_station_counts[end_station_name] += 1
                if end_station_name not in station_coords and row["ENDSTATIONLATITUDE"] and row["ENDSTATIONLONGITUDE"]:
                    station_coords[end_station_name] = (
                        float(row["ENDSTATIONLONGITUDE"]),
                        float(row["ENDSTATIONLATITUDE"]),
                    )

            if row["STARTSTATIONARRONDISSEMENT"]:
                start_borough_counts[row["STARTSTATIONARRONDISSEMENT"]] += 1
            if row["ENDSTATIONARRONDISSEMENT"]:
                end_borough_counts[row["ENDSTATIONARRONDISSEMENT"]] += 1

            utc_seconds = int(row["STARTTIMEMS"]) // 1000
            offset_seconds = -4 * 3600 if dst_start_utc <= utc_seconds < dst_end_utc else -5 * 3600
            local_seconds = utc_seconds + offset_seconds
            local_day_key = local_seconds // 86400
            local_hour = (local_seconds % 86400) // 3600

            day_info = day_cache.get(local_day_key)
            if day_info is None:
                local_dt = datetime.utcfromtimestamp(local_seconds)
                day_info = (local_dt.strftime("%Y-%m-%d"), local_dt.strftime("%Y-%m"), local_dt.weekday())
                day_cache[local_day_key] = day_info

            day_label, month_label, weekday = day_info
            daily_counts[day_label] += 1
            monthly_counts[month_label] += 1
            weekday_hour[weekday, local_hour] += 1

    start_station = pd.Series(start_station_counts).sort_values(ascending=False)
    end_station = pd.Series(end_station_counts).sort_values(ascending=False)
    station_demand = (
        pd.concat([start_station.rename("starts"), end_station.rename("ends")], axis=1)
        .fillna(0)
        .sum(axis=1)
        .sort_values(ascending=False)
    )
    start_borough = pd.Series(start_borough_counts).sort_values(ascending=False)
    end_borough = pd.Series(end_borough_counts).sort_values(ascending=False)

    summary = {
        "total_rows": int(total_rows),
        "unique_start_stations": int(start_station.shape[0]),
        "daily_counts": dict(sorted(daily_counts.items())),
        "monthly_counts": dict(sorted(monthly_counts.items())),
        "weekday_hour_counts": weekday_hour.tolist(),
        "start_borough_counts": {str(k): int(v) for k, v in start_borough.items()},
        "end_borough_counts": {str(k): int(v) for k, v in end_borough.items()},
        "station_demand_counts": {str(k): int(v) for k, v in station_demand.items()},
        "station_coordinates": {
            str(k): {"lon": float(v[0]), "lat": float(v[1])} for k, v in station_coords.items()
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


def load_summary() -> dict:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text())
    return build_summary()


def to_series(summary: dict) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    daily = pd.Series(summary["daily_counts"], dtype="int64")
    daily.index = pd.to_datetime(daily.index)
    monthly = pd.Series(summary["monthly_counts"], dtype="int64")
    monthly.index = pd.to_datetime(monthly.index + "-01")
    weekday_hour = pd.DataFrame(
        summary["weekday_hour_counts"],
        index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        columns=list(range(24)),
    )
    station_demand = pd.Series(summary["station_demand_counts"], dtype="int64").sort_values(ascending=False)
    return daily, monthly, weekday_hour, station_demand


def station_short_label(name: str) -> str:
    replacements = {
        "Métro ": "Metro ",
        "de Maisonneuve": "Maisonneuve",
        "Utilités publiques": "Utilities",
        "Place Jacques-Cartier": "Jacques-Cartier",
        "de Châteaubriand": "Chateaubriand",
        "de Brébeuf": "Brebeuf",
        "de Rigaud": "Rigaud",
        "de la Commune": "Commune",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    if " (" in name:
        base, suffix = name.split(" (", 1)
        suffix = suffix.rstrip(")")
        suffix = suffix.replace(" / ", " / ")
        return f"{base}\n({suffix})"
    if " / " in name and len(name) > 20:
        return name.replace(" / ", "\n", 1)
    return name


def station_multiline_label(name: str, width: int = 20) -> str:
    short_name = station_short_label(name).replace("\n/\n", " / ").replace("\n", " ")
    parts = [part.strip() for part in short_name.split(" / ")]
    wrapped_parts = []
    for part in parts:
        wrapped = textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False)
        wrapped_parts.append("\n".join(wrapped) if wrapped else part)
    return "\n/\n".join(wrapped_parts)


def save_both(fig: plt.Figure, repo_name: str, paper_name: str) -> None:
    fig.savefig(REPO_OUTPUT_DIR / repo_name)
    fig.savefig(PAPER_OUTPUT_DIR / paper_name)
    plt.close(fig)


def plot_top30_station_demand(station_demand: pd.Series) -> None:
    top30 = station_demand.head(30)
    groups = [top30.iloc[:15].iloc[::-1], top30.iloc[15:30].iloc[::-1]]
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 10.6), sharex=True)
    palette = sns.color_palette("crest", n_colors=15)
    xmax = top30.max() * 1.08

    for panel_idx, (ax, group) in enumerate(zip(axes, groups)):
        labels = [station_multiline_label(i, width=26) for i in group.index]
        y = np.arange(len(group))
        ax.barh(
            y,
            group.values,
            color=palette,
            edgecolor="#36454f",
            linewidth=0.35,
            height=0.64,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", labelsize=8.2, pad=8)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, xmax)
        start_rank = 1 if panel_idx == 0 else 16
        for yi, rank in zip(y, range(start_rank, start_rank + len(group))):
            ax.text(
                0.008,
                yi,
                f"{rank}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.8,
                color="#5a5a5a",
            )
        ax.set_title(
            f"Stations ranked {start_rank}-{start_rank + len(group) - 1}",
            loc="left",
            fontsize=9,
            pad=4,
        )
        ax.set_ylabel("")

    axes[-1].set_xlabel("Annual station activity (departures + arrivals)")
    fig.text(
        0.98,
        0.015,
        f"Top 30 total: {int(top30.sum()):,}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#3d3d3d",
    )
    fig.subplots_adjust(left=0.36, right=0.98, top=0.95, bottom=0.08, hspace=0.16)
    save_both(fig, "bixi_case_top30_station_demand.pdf", PAPER_FIGURES["fig6a"])


def plot_top30_station_map(summary: dict, station_demand: pd.Series) -> None:
    coords = summary["station_coordinates"]
    top30 = station_demand.head(30)
    rows = []
    for station, demand in top30.items():
        if station in coords:
            rows.append(
                {
                    "station": station,
                    "demand": int(demand),
                    "lon": coords[station]["lon"],
                    "lat": coords[station]["lat"],
                }
            )
    station_gdf = gpd.GeoDataFrame(
        pd.DataFrame(rows),
        geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2950")

    roads = gpd.read_file(STREET_NETWORK_PATH)
    xmin, ymin, xmax, ymax = station_gdf.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = dx * 0.16
    pad_y = dy * 0.16
    focus = roads.cx[xmin - pad_x : xmax + pad_x, ymin - pad_y : ymax + pad_y].copy()
    local = focus[~focus["CLASSE"].isin([0, 1, 4])]
    arterial = focus[focus["CLASSE"].isin([0, 1, 4])]

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.set_facecolor("#eef2f4")
    local.plot(ax=ax, color="#d9dee2", linewidth=0.24, alpha=0.9)
    arterial.plot(ax=ax, color="#c7b08a", linewidth=0.95, alpha=0.95)
    arterial.plot(ax=ax, color="#f6f1e8", linewidth=0.35, alpha=1.0)

    sizes = 28 + 210 * (station_gdf["demand"] - station_gdf["demand"].min()) / (
        station_gdf["demand"].max() - station_gdf["demand"].min()
    )
    sc = ax.scatter(
        station_gdf.geometry.x,
        station_gdf.geometry.y,
        s=sizes,
        c=station_gdf["demand"],
        cmap="viridis",
        edgecolors="white",
        linewidths=0.4,
        alpha=0.95,
        zorder=3,
    )
    for _, row in station_gdf.head(6).iterrows():
        ax.annotate(
            station_short_label(row["station"]).replace("\n", " "),
            (row.geometry.x, row.geometry.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#2f2f2f",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.3},
        )

    fxmin, fymin, fxmax, fymax = focus.total_bounds
    ax.set_xlim(fxmin, fxmax)
    ax.set_ylim(fymin, fymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Annual station activity")
    fig.subplots_adjust(left=0.03, right=0.92, top=0.95, bottom=0.05)
    save_both(fig, "bixi_case_top30_station_map.pdf", PAPER_FIGURES["fig6b"])


def plot_borough_activity(summary: dict) -> None:
    start = pd.Series(summary["start_borough_counts"], dtype="int64")
    end = pd.Series(summary["end_borough_counts"], dtype="int64")
    borough = pd.DataFrame({"Origins": start, "Destinations": end}).fillna(0)
    borough["Total"] = borough["Origins"] + borough["Destinations"]
    plot_df = borough.sort_values("Total", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    y = np.arange(len(plot_df))
    ax.barh(y - 0.18, plot_df["Origins"], height=0.34, color="#2a7f9e", label="Origins")
    ax.barh(y + 0.18, plot_df["Destinations"], height=0.34, color="#d08770", label="Destinations")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.index)
    ax.set_xlabel("Trips")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.95, bottom=0.14)
    save_both(fig, "bixi_case_borough_activity.pdf", PAPER_FIGURES["fig6c"])


def plot_daily_ridership(daily: pd.Series) -> None:
    rolling = daily.rolling(7, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.fill_between(daily.index, daily.values, color="#b7d3e7", alpha=0.45, linewidth=0)
    ax.plot(daily.index, daily.values, color="#90a4b4", linewidth=0.7, alpha=0.45)
    ax.plot(daily.index, rolling.values, color="#1f5a7a", linewidth=1.7)
    ax.set_ylabel("Trips/day")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.text(
        0.98,
        0.92,
        f"Total trips: {int(daily.sum()):,}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        color="#3d3d3d",
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.18)
    save_both(fig, "bixi_case_daily_ridership.pdf", PAPER_FIGURES["fig7a"])


def plot_monthly_ridership(monthly: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    colors = sns.color_palette("mako", n_colors=len(monthly))
    ax.bar(monthly.index, monthly.values, width=24, color=colors, edgecolor="#36454f", linewidth=0.3)
    ax.set_ylabel("Trips/month")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.18)
    save_both(fig, "bixi_case_monthly_ridership.pdf", PAPER_FIGURES["fig7b"])


def plot_weekday_hour_heatmap(weekday_hour: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    sns.heatmap(
        weekday_hour,
        cmap=sns.color_palette("blend:#f7fbff,#6baed6,#2171b5,#08306b", as_cmap=True),
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Trips"},
        ax=ax,
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    fig.subplots_adjust(left=0.12, right=0.92, top=0.98, bottom=0.12)
    save_both(fig, "bixi_case_weekday_hour_heatmap.pdf", PAPER_FIGURES["fig7c"])


def write_stats_table(summary: dict) -> dict:
    total = summary["total_rows"]
    monthly = summary["monthly_counts"]
    station_demand = list(summary["station_demand_counts"].values())
    start_borough = summary["start_borough_counts"]
    stats = {
        "Annual BIXI trips (2025)": f"{total:,}",
        "Distinct origin stations": f"{summary['unique_start_stations']:,}",
        "Share of origins in Le Plateau-Mont-Royal and Ville-Marie": f"{100 * (start_borough['Le Plateau-Mont-Royal'] + start_borough['Ville-Marie']) / total:.1f}\\%",
        "Activity captured by top 30 stations": f"{sum(station_demand[:30]):,}",
        "Activity captured by top 10 stations": f"{sum(station_demand[:10]):,}",
        "Trips occurring from April to November": f"{100 * sum(v for k, v in monthly.items() if '2025-04' <= k <= '2025-11') / total:.1f}\\%",
        "Trips occurring from June to September": f"{100 * sum(monthly[k] for k in ['2025-06', '2025-07', '2025-08', '2025-09']) / total:.1f}\\%",
    }
    table_lines = [
        "\\begin{table}[h!]",
        "  \\centering",
        "  \\caption{Descriptive statistics of the 2025 BIXI dataset used in the case study.}",
        "  \\label{tab:bixi_data_summary}",
        "  \\small",
        "  \\begin{tabularx}{0.82\\textwidth}{@{} l X @{} }",
        "    \\hline",
        "    \\textbf{Indicator} & \\textbf{Value} \\\\",
        "    \\hline",
    ]
    for key, value in stats.items():
        table_lines.append(f"    {key} & {value} \\\\")
    table_lines.extend(["    \\hline", "  \\end{tabularx}", "\\end{table}"])
    (REPO_OUTPUT_DIR / "bixi_data_summary_table.tex").write_text("\n".join(table_lines) + "\n")
    return stats


def main() -> None:
    ensure_dirs()
    apply_style()
    summary = load_summary()
    daily, monthly, weekday_hour, station_demand = to_series(summary)
    plot_top30_station_demand(station_demand)
    plot_top30_station_map(summary, station_demand)
    plot_borough_activity(summary)
    plot_daily_ridership(daily)
    plot_monthly_ridership(monthly)
    plot_weekday_hour_heatmap(weekday_hour)
    stats = write_stats_table(summary)
    print(f"Saved figure PDFs to {PAPER_OUTPUT_DIR}")
    print(f"Summary saved to {SUMMARY_PATH}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
