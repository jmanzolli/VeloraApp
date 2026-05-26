from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString


SL_LEVELS = (3.5, 5.0, 6.5, 8.0, 9.5)
EFFORT_COLUMNS = [
    "elev_start_m",
    "elev_end_m",
    "rise_m",
    "grade_pct",
    "uphill_grade_pct",
    "uphill_gain_m",
    "steepness_level",
    "effort_exposure",
]


@dataclass(frozen=True)
class EffortConfig:
    short_segment_exemption_m: float = 50.0


class EffortCalculator:
    """Calculate directional cyclist effort from edge elevations."""

    def __init__(self, config: EffortConfig | None = None) -> None:
        self.config = config or EffortConfig()

    def make_directed(self, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if "direction" in edges and edges["direction"].notna().any():
            return edges.copy().reset_index(drop=True)
        rows: list[pd.Series] = []
        for _, row in edges.iterrows():
            geom = row.geometry
            parts = [geom] if isinstance(geom, LineString) else list(geom.geoms) if isinstance(geom, MultiLineString) else []
            for part in parts:
                forward = row.copy()
                forward.geometry = part
                forward["direction"] = "forward"
                rows.append(forward)
                if not bool(row.get("one_way", False)):
                    reverse = row.copy()
                    reverse.geometry = LineString(list(part.coords)[::-1])
                    reverse["direction"] = "reverse"
                    reverse["elev_start_m"] = row.get("elev_end_m", np.nan)
                    reverse["elev_end_m"] = row.get("elev_start_m", np.nan)
                    rows.append(reverse)
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=edges.crs).reset_index(drop=True)

    def score_dataframe(self, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        directed = self.make_directed(edges)
        if directed.empty:
            return directed

        metric = directed.to_crs(self.metric_crs(directed))
        lengths = metric.geometry.length.astype(float)
        start = pd.to_numeric(directed.get("elev_start_m", pd.Series(np.nan, index=directed.index)), errors="coerce")
        end = pd.to_numeric(directed.get("elev_end_m", pd.Series(np.nan, index=directed.index)), errors="coerce")
        rise = end - start
        grade_pct = np.where(lengths > 0, rise / lengths * 100.0, np.nan)
        uphill_grade = np.maximum(grade_pct, 0.0)
        uphill_gain = np.maximum(rise, 0.0)

        directed["length"] = lengths
        directed["elev_start_m"] = start
        directed["elev_end_m"] = end
        directed["rise_m"] = rise
        directed["grade_pct"] = grade_pct
        directed["uphill_grade_pct"] = uphill_grade
        directed["uphill_gain_m"] = uphill_gain
        directed["steepness_level"] = [
            self.classify(length_m=float(length), uphill_grade_pct=float(grade) if pd.notna(grade) else np.nan)
            for length, grade in zip(lengths, uphill_grade)
        ]
        directed["effort_exposure"] = np.where(
            pd.notna(uphill_grade),
            lengths * (1.0 + (uphill_grade / 3.5) ** 2),
            np.nan,
        )
        directed["effort_available"] = pd.notna(directed["effort_exposure"])
        return directed

    def classify(self, length_m: float, uphill_grade_pct: float) -> float | np.nan:
        if not np.isfinite(uphill_grade_pct):
            return np.nan
        if length_m <= self.config.short_segment_exemption_m or uphill_grade_pct <= 0:
            return SL_LEVELS[0]
        for level in SL_LEVELS:
            if uphill_grade_pct <= level:
                return level
        return SL_LEVELS[-1]

    @staticmethod
    def metric_crs(gdf: gpd.GeoDataFrame) -> str:
        if gdf.crs is not None and getattr(gdf.crs, "is_projected", False):
            return str(gdf.crs)
        try:
            centroid = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
            zone = int((centroid.x + 180) // 6) + 1
            if centroid.y >= 0:
                return f"EPSG:{32600 + zone}"
            return f"EPSG:{32700 + zone}"
        except Exception:
            return "EPSG:3857"


def sample_dem_endpoints(edges: gpd.GeoDataFrame, raster_path: str) -> gpd.GeoDataFrame:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("DEM uploads require rasterio. Install project requirements and try again.") from exc

    sampled = edges.copy()
    with rasterio.open(raster_path) as raster:
        metric = sampled.to_crs(raster.crs) if sampled.crs != raster.crs else sampled
        starts = []
        ends = []
        for geom in metric.geometry:
            coords = list(geom.coords)
            starts.append(coords[0])
            ends.append(coords[-1])
        sampled["elev_start_m"] = [value[0] for value in raster.sample(starts)]
        sampled["elev_end_m"] = [value[0] for value in raster.sample(ends)]
    return sampled


def fetch_online_endpoint_elevations(edges: gpd.GeoDataFrame, batch_size: int = 100) -> gpd.GeoDataFrame:
    """Fetch endpoint elevations with a batched primary source and fallback."""
    sampled = edges.to_crs("EPSG:4326").copy()
    endpoint_rows: list[tuple[float, float]] = []
    for geom in sampled.geometry:
        coords = list(geom.coords)
        endpoint_rows.extend([(coords[0][1], coords[0][0]), (coords[-1][1], coords[-1][0])])

    unique_endpoints = list(dict.fromkeys(endpoint_rows))
    elevation_by_point: dict[tuple[float, float], float] = {}
    providers: set[str] = set()
    for chunk in _chunks(unique_endpoints, min(max(1, int(batch_size)), 100)):
        try:
            values = _fetch_open_meteo_chunk(chunk)
            providers.add("Open-Meteo Copernicus DEM GLO-90")
        except requests.RequestException:
            values = _fetch_open_elevation_chunk(chunk)
            providers.add("Open-Elevation fallback")
        elevation_by_point.update(dict(zip(chunk, values)))

    sampled["elev_start_m"] = [elevation_by_point[point] for point in endpoint_rows[0::2]]
    sampled["elev_end_m"] = [elevation_by_point[point] for point in endpoint_rows[1::2]]
    sampled["elevation_source"] = "; ".join(sorted(providers))
    return sampled.to_crs(edges.crs)


def ensure_effort_columns(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    enriched = network.copy()
    for column in EFFORT_COLUMNS:
        if column not in enriched:
            enriched[column] = np.nan
    if "effort_available" not in enriched:
        enriched["effort_available"] = enriched["effort_exposure"].notna()
    return enriched


def _chunks(values: list[tuple[float, float]], size: int) -> Iterable[list[tuple[float, float]]]:
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def _fetch_open_meteo_chunk(chunk: list[tuple[float, float]]) -> list[float]:
    response = requests.get(
        "https://api.open-meteo.com/v1/elevation",
        params={
            "latitude": ",".join(str(point[0]) for point in chunk),
            "longitude": ",".join(str(point[1]) for point in chunk),
        },
        headers={"User-Agent": "BikeNetworkPlanner/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    values = response.json().get("elevation", [])
    if len(values) != len(chunk):
        raise requests.RequestException("Elevation response did not match requested coordinate count.")
    return [float(value) for value in values]


def _fetch_open_elevation_chunk(chunk: list[tuple[float, float]]) -> list[float]:
    values: list[float] = []
    for subchunk in _chunks(chunk, 25):
        payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in subchunk]}
        response = requests.post("https://api.open-elevation.com/api/v1/lookup", json=payload, timeout=20)
        response.raise_for_status()
        values.extend(float(item["elevation"]) for item in response.json()["results"])
    return values
