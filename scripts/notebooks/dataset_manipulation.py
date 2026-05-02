# Converted from dataset_manipulation.ipynb
# This script preserves notebook cells as Python sections for maintainability.

# %% [markdown] Cell 1
# ## BIXI dataset analysis (13413136 samples)

# %% [markdown] Cell 2
# ### Overview
# This notebook loads the BIXI trip dataset, cleans it (drops missing fields and removes outliers), and then performs exploratory analysis, mapping, and station sizing estimates.

# %% [markdown] Cell 3
# ### 1. Imports
# Basic libraries used throughout the notebook.

# %% Cell 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans

# %% [markdown] Cell 5
# ### 2. Load data
# Read the CSV into a DataFrame.

# %% Cell 6
# Read only the first rows to play with
file= 'DonneesOuvertes2025_01020304050607080910.csv'
sample_df = pd.read_csv(file)
sample_df.head()

# %% [markdown] Cell 7
# ### 3. Clean data
# Drop rows with missing critical fields, compute duration and distance, then remove outliers.

# %% Cell 8
# Clean dataset: remove empty fields and outliers
required_cols = [
    "STARTSTATIONNAME",
    "ENDSTATIONNAME",
    "STARTSTATIONLATITUDE",
    "STARTSTATIONLONGITUDE",
    "ENDSTATIONLATITUDE",
    "ENDSTATIONLONGITUDE",
    "STARTTIMEMS",
    "ENDTIMEMS",
]

clean_df = sample_df.dropna(subset=required_cols).copy()

# Duration (minutes)
clean_df["duration_min"] = (clean_df["ENDTIMEMS"] - clean_df["STARTTIMEMS"]) / 1000 / 60
clean_df = clean_df[clean_df["duration_min"] > 0]

# Distance (km) using haversine

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

clean_df["trip_distance_km"] = haversine_km(
    clean_df["STARTSTATIONLATITUDE"],
    clean_df["STARTSTATIONLONGITUDE"],
    clean_df["ENDSTATIONLATITUDE"],
    clean_df["ENDSTATIONLONGITUDE"],
)

clean_df = clean_df[clean_df["trip_distance_km"] > 0]

# Remove outliers (1st–99th percentile) for duration and distance
if not clean_df.empty:
    dur_low, dur_high = clean_df["duration_min"].quantile([0.01, 0.99])
    dist_low, dist_high = clean_df["trip_distance_km"].quantile([0.01, 0.99])

    cleaned_df = clean_df[
        clean_df["duration_min"].between(dur_low, dur_high)
        & clean_df["trip_distance_km"].between(dist_low, dist_high)
    ].copy()
else:
    cleaned_df = clean_df.copy()

print("Raw rows:", len(sample_df))
print("Cleaned rows:", len(cleaned_df))

cleaned_df.head()

# %% [markdown] Cell 9
# ### 4. Quick summary
# Basic shape, dtypes, and missingness after cleaning.

# %% Cell 10
# Quick exploratory analysis on the cleaned data
print("Rows, columns:", cleaned_df.shape)

# Column types and missingness
missing_pct = (cleaned_df.isna().mean() * 100).sort_values(ascending=False)

summary = pd.DataFrame({
    "dtype": cleaned_df.dtypes,
    "missing_%": missing_pct
})

summary

# %% [markdown] Cell 11
# ### 5. Station demand (busiest stations)
# Outbound + inbound trip counts per station.

# %% Cell 12
# Busiest stations (outbound + inbound)
start_counts = cleaned_df["STARTSTATIONNAME"].value_counts(dropna=True)
end_counts = cleaned_df["ENDSTATIONNAME"].value_counts(dropna=True)

busiest_total = (
    start_counts.rename("outbound")
    .to_frame()
    .join(end_counts.rename("inbound"), how="outer")
    .fillna(0)
)

busiest_total["total_trips"] = busiest_total["outbound"] + busiest_total["inbound"]

busiest_total.sort_values("total_trips", ascending=False).head(10)

# %% [markdown] Cell 13
# ### 6. Map of top-demand stations
# Geospatial view of the top stations by total demand.

# %% Cell 14
top_n = 40

# Demand per station (outbound + inbound)
station_demand = (
    cleaned_df["STARTSTATIONNAME"].value_counts().rename("outbound")
    .to_frame()
    .join(cleaned_df["ENDSTATIONNAME"].value_counts().rename("inbound"), how="outer")
    .fillna(0)
 )
station_demand["total_trips"] = station_demand["outbound"] + station_demand["inbound"]

# Station coordinates (prefer start coords, fallback to end coords)
start_coords = cleaned_df.groupby("STARTSTATIONNAME")[["STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE"]].median()
start_coords.columns = ["lat", "lon"]

end_coords = cleaned_df.groupby("ENDSTATIONNAME")[["ENDSTATIONLATITUDE", "ENDSTATIONLONGITUDE"]].median()
end_coords.columns = ["lat", "lon"]

coords = start_coords.combine_first(end_coords)

station_map = station_demand.join(coords, how="left").dropna(subset=["lat", "lon"])

# Top N stations by total demand
station_map = station_map.sort_values("total_trips", ascending=False).head(top_n).reset_index()
station_map = station_map.rename(columns={"index": "station"})

center = [station_map["lat"].median(), station_map["lon"].median()]
m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

min_trips = station_map["total_trips"].min()
max_trips = station_map["total_trips"].max()

def scale_size(value, min_val, max_val, min_size=5, max_size=20):
    if max_val == min_val:
        return (min_size + max_size) / 2
    return min_size + (value - min_val) * (max_size - min_size) / (max_val - min_val)

for _, row in station_map.iterrows():
    radius = scale_size(row["total_trips"], min_trips, max_trips)
    tooltip = f"{row['station']}<br>Total: {int(row['total_trips'])}<br>Out: {int(row['outbound'])} | In: {int(row['inbound'])}"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color="#2C7FB8",
        fill=True,
        fill_color="#2C7FB8",
        fill_opacity=0.7,
        tooltip=tooltip,
    ).add_to(m)

m

# %% [markdown] Cell 15
# ### 7. Trip density heatmap
# Origins + destinations density across Montreal.

# %% Cell 16
top_x = 20  # choose top-X stations by total demand

# Identify top-X stations by total demand
station_demand = (
    cleaned_df["STARTSTATIONNAME"].value_counts().rename("outbound")
    .to_frame()
    .join(cleaned_df["ENDSTATIONNAME"].value_counts().rename("inbound"), how="outer")
    .fillna(0)
 )
station_demand["total_trips"] = station_demand["outbound"] + station_demand["inbound"]

top_stations = station_demand.sort_values("total_trips", ascending=False).head(top_x).index

# Filter trips where either start or end is in top-X stations
filtered = cleaned_df[
    cleaned_df["STARTSTATIONNAME"].isin(top_stations)
    | cleaned_df["ENDSTATIONNAME"].isin(top_stations)
 ]

# Build point set from origins and destinations
points_df = pd.concat([
    filtered[["STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE"]].rename(
        columns={"STARTSTATIONLATITUDE": "lat", "STARTSTATIONLONGITUDE": "lon"}
    ),
    filtered[["ENDSTATIONLATITUDE", "ENDSTATIONLONGITUDE"]].rename(
        columns={"ENDSTATIONLATITUDE": "lat", "ENDSTATIONLONGITUDE": "lon"}
    ),
], ignore_index=True).dropna()

center = [points_df["lat"].median(), points_df["lon"].median()]
m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

HeatMap(
    data=points_df[["lat", "lon"]].values.tolist(),
    radius=18,
    blur=15,
    min_opacity=0.2,
).add_to(m)

m

# %% [markdown] Cell 17
# ### 8. Temporal patterns
# Heatmaps by hour and weekday, plus daily totals and anomaly highlights.

# %% Cell 18
# Hour x Weekday heatmap
starts = pd.to_datetime(cleaned_df["STARTTIMEMS"], unit="ms")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

heat_df = (
    pd.DataFrame({"weekday": starts.dt.day_name(), "hour": starts.dt.hour})
    .groupby(["weekday", "hour"]).size()
    .unstack(fill_value=0)
    .reindex(weekday_order)
)

plt.figure(figsize=(12, 4))
plt.imshow(heat_df, aspect="auto", cmap="viridis")
plt.yticks(range(len(weekday_order)), weekday_order)
plt.xticks(range(0, 24, 2))
plt.xlabel("Hour of day")
plt.ylabel("Weekday")
plt.title("Trips by weekday and hour")
plt.colorbar(label="Trips")
plt.tight_layout()
plt.show()

# Daily totals + anomaly highlights (z-score > 2)
daily = starts.dt.floor("D").value_counts().sort_index()

plt.figure(figsize=(12, 4))
plt.plot(daily.index, daily.values, color="#4C78A8", linewidth=1)

if daily.std() > 0:
    z = (daily - daily.mean()) / daily.std()
    anomalies = daily[z.abs() > 2]
    plt.scatter(anomalies.index, anomalies.values, color="#E45756", label="Anomaly", zorder=3)

plt.title("Daily trip totals")
plt.xlabel("Date")
plt.ylabel("Trips")
plt.tight_layout()
plt.show()

# %% [markdown] Cell 19
# ### 9. Duration vs distance (density)
# Hexbin plot to show the relationship between trip duration and distance.

# %% Cell 20
plt.figure(figsize=(6, 5))
plt.hexbin(
    cleaned_df["duration_min"],
    cleaned_df["trip_distance_km"],
    gridsize=50,
    bins="log",
    cmap="magma",
    mincnt=1,
)
plt.xlabel("Duration (minutes)")
plt.ylabel("Distance (km)")
plt.title("Trip duration vs distance (log density)")
plt.colorbar(label="log(count)")
plt.tight_layout()
plt.show()

# %% [markdown] Cell 21
# ### 10. Station imbalance
# Net flow (outbound − inbound) highlights rebalancing needs.

# %% Cell 22
outbound_counts = cleaned_df["STARTSTATIONNAME"].value_counts()
inbound_counts = cleaned_df["ENDSTATIONNAME"].value_counts()

imbalance = (
    outbound_counts.rename("outbound")
    .to_frame()
    .join(inbound_counts.rename("inbound"), how="outer")
    .fillna(0)
)
imbalance["net"] = imbalance["outbound"] - imbalance["inbound"]

# Top absolute imbalances
imbalance_top = imbalance.reindex(imbalance["net"].abs().sort_values(ascending=False).head(20).index)

plt.figure(figsize=(8, 6))
colors = ["#E45756" if v > 0 else "#4C78A8" for v in imbalance_top["net"]]
plt.barh(imbalance_top.index, imbalance_top["net"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top station net flows (outbound − inbound)")
plt.xlabel("Net trips")
plt.tight_layout()
plt.show()

# %% [markdown] Cell 23
# ### 11. OD flow matrix (top stations)
# Heatmap of trips between the most active stations.

# %% Cell 24
top_k = 15

top_stations = (
    cleaned_df["STARTSTATIONNAME"].value_counts().head(top_k).index
    .union(cleaned_df["ENDSTATIONNAME"].value_counts().head(top_k).index)
)

od_matrix = (
    cleaned_df[cleaned_df["STARTSTATIONNAME"].isin(top_stations) & cleaned_df["ENDSTATIONNAME"].isin(top_stations)]
    .groupby(["STARTSTATIONNAME", "ENDSTATIONNAME"]).size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(8, 6))
plt.imshow(od_matrix, aspect="auto", cmap="viridis")
plt.colorbar(label="Trips")
plt.title("OD flow matrix (top stations)")
plt.xticks(range(len(od_matrix.columns)), od_matrix.columns, rotation=90, fontsize=7)
plt.yticks(range(len(od_matrix.index)), od_matrix.index, fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown] Cell 25
# ### 12. Popular routes by hour
# Top routes and how their demand changes over the day.

# %% Cell 26
# Top 3 OD routes and hourly profiles
route_counts = (
    cleaned_df.groupby(["STARTSTATIONNAME", "ENDSTATIONNAME"]).size().sort_values(ascending=False)
)

top_routes = route_counts.head(3).index

route_df = cleaned_df[["STARTSTATIONNAME", "ENDSTATIONNAME", "STARTTIMEMS"]].copy()
route_df["hour"] = pd.to_datetime(route_df["STARTTIMEMS"], unit="ms").dt.hour

plt.figure(figsize=(8, 4))
for route in top_routes:
    r = route_df[(route_df["STARTSTATIONNAME"] == route[0]) & (route_df["ENDSTATIONNAME"] == route[1])]
    hourly = r.groupby("hour").size().reindex(range(24), fill_value=0)
    plt.plot(hourly.index, hourly.values, label=f"{route[0]} → {route[1]}")

plt.title("Top routes by hour")
plt.xlabel("Hour of day")
plt.ylabel("Trips")
plt.legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown] Cell 27
# ### 13. Weekly trend
# Rolling weekly totals to show seasonality across the year.

# %% Cell 28
daily = starts.dt.floor("D").value_counts().sort_index()
weekly = daily.rolling(7, center=True).mean()

plt.figure(figsize=(12, 4))
plt.plot(daily.index, daily.values, color="#9ecae1", linewidth=0.8, label="Daily")
plt.plot(weekly.index, weekly.values, color="#08519c", linewidth=2, label="7-day mean")
plt.title("Daily trips and 7-day rolling average")
plt.xlabel("Date")
plt.ylabel("Trips")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown] Cell 29
# ### 14. Station clusters
# Group stations by demand patterns (outbound/inbound) for a quick segmentation view.

# %% Cell 30
station_stats = (
    cleaned_df["STARTSTATIONNAME"].value_counts().rename("outbound")
    .to_frame()
    .join(cleaned_df["ENDSTATIONNAME"].value_counts().rename("inbound"), how="outer")
    .fillna(0)
)

# Simple 2D clustering on outbound/inbound
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
station_stats["cluster"] = kmeans.fit_predict(station_stats[["outbound", "inbound"]])

plt.figure(figsize=(6, 5))
for c in sorted(station_stats["cluster"].unique()):
    subset = station_stats[station_stats["cluster"] == c]
    plt.scatter(subset["outbound"], subset["inbound"], s=20, alpha=0.6, label=f"Cluster {c}")

plt.xlabel("Outbound trips")
plt.ylabel("Inbound trips")
plt.title("Station clusters by demand")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown] Cell 31
# ### 15. Trip duration distribution
# Summary stats and histogram after cleaning.

# %% Cell 32
# Trip duration distribution (minutes) — cleaned
valid_duration_clean = cleaned_df["duration_min"].dropna()

valid_duration_clean.describe(percentiles=[0.5, 0.9, 0.95, 0.99])

# %% Cell 33
# Trip duration histogram (outliers removed)
plt.figure(figsize=(6, 4))
plt.hist(valid_duration_clean, bins=40, color="#4C78A8")
plt.title("Trip duration (minutes)")
plt.xlabel("Minutes")
plt.ylabel("Trips")
plt.tight_layout()
plt.show()

# %% [markdown] Cell 34
# ### 16. Trip distance distribution
# Summary stats and histogram after cleaning.

# %% Cell 35
# Trip distance distribution (km) — cleaned
trip_distance_km_clean = cleaned_df["trip_distance_km"].dropna()
trip_distance_km_clean.describe(percentiles=[0.5, 0.9, 0.95, 0.99])

# %% Cell 36
# Trip distance histogram (outliers removed)
plt.figure(figsize=(6, 4))
plt.hist(trip_distance_km_clean, bins=40, color="#72B7B2")
plt.title("Trip distance (km)")
plt.xlabel("Kilometers")
plt.ylabel("Trips")
plt.tight_layout()
plt.show()

# %% [markdown] Cell 37
# ### 17. Origin–destination pairs
# Most common start–end station pairs.

# %% Cell 38
# Top origin–destination pairs
od_pairs = (
    cleaned_df
    .groupby(["STARTSTATIONNAME", "ENDSTATIONNAME"], dropna=True)
    .size()
    .sort_values(ascending=False)
    .head(10)
    .rename("trip_count")
)

od_pairs

# %% [markdown] Cell 39
# ### 18. Station sizing (worst-case day)
# Estimate docks based on the busiest day and peak time buckets. We can approximate required docks by the **peak number of trips per time bucket** (e.g., 15 minutes) for a specific day and station. A conservative estimate uses the 95th percentile or maximum of **inbound + outbound** trips per bucket, with a small safety factor.

# %% Cell 40
# Parameters
bucket = "15min"  # 5min, 15min, 30min, 1H
safety_factor = 1.2

# Convert timestamps (ms) to datetime
starts = pd.to_datetime(cleaned_df["STARTTIMEMS"], unit="ms")
ends = pd.to_datetime(cleaned_df["ENDTIMEMS"], unit="ms")

# Find busiest day by total trips (based on start time)
trip_day = starts.dt.floor("D")
trip_day_counts = trip_day.value_counts(dropna=True)

if trip_day_counts.empty:
    print("No timestamps found in the cleaned data. Check the CSV or cleaning rules.")
else:
    busiest_day = trip_day_counts.idxmax().strftime("%Y-%m-%d")
    print(f"Busiest day in cleaned data: {busiest_day} (trips: {trip_day_counts.max()})")

    # Filter to the busiest day
    mask_start = starts.dt.strftime("%Y-%m-%d") == busiest_day
    mask_end = ends.dt.strftime("%Y-%m-%d") == busiest_day

    starts_df = cleaned_df.loc[mask_start, ["STARTSTATIONNAME"]].copy()
    starts_df["time_bucket"] = starts[mask_start].dt.floor(bucket)

    ends_df = cleaned_df.loc[mask_end, ["ENDSTATIONNAME"]].copy()
    ends_df["time_bucket"] = ends[mask_end].dt.floor(bucket)

    # Count trips per station per time bucket
    outbound = (
        starts_df.groupby(["STARTSTATIONNAME", "time_bucket"]).size().rename("outbound")
    )

    inbound = (
        ends_df.groupby(["ENDSTATIONNAME", "time_bucket"]).size().rename("inbound")
    )

    # Align by station and time bucket
    flow = outbound.to_frame().join(
        inbound.to_frame(),
        how="outer"
    ).fillna(0)

    flow["total"] = flow["outbound"] + flow["inbound"]

    # Estimate dock size per station (peak bucket * safety factor)
    station_docks_est = (
        flow.groupby(level=0)["total"]
        .quantile(0.95)
        .rename("p95_bucket_trips")
        .to_frame()
    )

station_docks_est["estimated_docks"] = (station_docks_est["p95_bucket_trips"] * safety_factor).round().astype(int)
station_docks_est.sort_values("estimated_docks", ascending=False).head(40)

# %% Cell 41
# Top 40 busiest stations with full info
outbound = cleaned_df["STARTSTATIONNAME"].value_counts().rename("outbound")
inbound = cleaned_df["ENDSTATIONNAME"].value_counts().rename("inbound")

station_summary = (
    outbound.to_frame()
    .join(inbound.to_frame(), how="outer")
    .fillna(0)
)
station_summary["total_trips"] = station_summary["outbound"] + station_summary["inbound"]

# Attach estimated docks if available
if "station_docks_est" in globals():
    station_summary = station_summary.join(
        station_docks_est[["estimated_docks"]],
        how="left"
    )
else:
    station_summary["estimated_docks"] = pd.NA

# Build final list
busiest_40 = (
    station_summary
    .sort_values("total_trips", ascending=False)
    .head(40)
    .reset_index()
    .rename(columns={"index": "station"})
)

busiest_40

# %% Cell 42
# Add coordinates to busiest_40
start_coords = cleaned_df.groupby("STARTSTATIONNAME")[["STARTSTATIONLATITUDE", "STARTSTATIONLONGITUDE"]].median()
start_coords.columns = ["lat", "lon"]

end_coords = cleaned_df.groupby("ENDSTATIONNAME")[["ENDSTATIONLATITUDE", "ENDSTATIONLONGITUDE"]].median()
end_coords.columns = ["lat", "lon"]

coords = start_coords.combine_first(end_coords)

busiest_40_map = busiest_40.set_index("station").join(coords, how="left").reset_index()
busiest_40_map = busiest_40_map.dropna(subset=["lat", "lon"])

center = [busiest_40_map["lat"].median(), busiest_40_map["lon"].median()]
m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

min_trips = busiest_40_map["total_trips"].min()
max_trips = busiest_40_map["total_trips"].max()

def scale_size(value, min_val, max_val, min_size=6, max_size=24):
    if max_val == min_val:
        return (min_size + max_size) / 2
    return min_size + (value - min_val) * (max_size - min_size) / (max_val - min_val)

for _, row in busiest_40_map.iterrows():
    radius = scale_size(row["total_trips"], min_trips, max_trips)
    docks = row["estimated_docks"] if pd.notna(row["estimated_docks"]) else "NA"
    tooltip = f"{row['station']}<br>Total: {int(row['total_trips'])}<br>Docks: {docks}"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color="#1D91C0",
        fill=True,
        fill_color="#1D91C0",
        fill_opacity=0.75,
        tooltip=tooltip,
    ).add_to(m)

m

# %% Cell 43
# Export for opt.ipynb (include coordinates)
export_df = busiest_40.set_index("station").join(coords, how="left").reset_index()
export_cols = ["station", "outbound", "inbound", "total_trips", "estimated_docks", "lat", "lon"]

export_df[export_cols].to_csv("busiest_40.csv", index=False)
print("Saved busiest_40.csv")

