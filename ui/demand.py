from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


SUPPORTED_VECTOR_SUFFIXES = {".gpkg", ".geojson", ".json", ".shp", ".parquet", ".zip"}


@dataclass(frozen=True)
class DemandConfig:
    adoption_rate: float = 0.08
    daily_trip_rate: float = 0.35
    annualization_days: int = 365
    minimum_population: float = 50.0
    minimum_trips: float = 1.0
    minimum_docks: float = 8.0
    trips_per_dock: float = 4000.0
    target_spacing_meters: float = 250.0
    max_candidates: int = 250


class CanadianPopulationDemandBuilder:
    """Generate Velora-compatible demand points from Canadian census polygons."""

    POPULATION_ALIASES = {
        "population",
        "pop",
        "pop2021",
        "population_2021",
        "population2021",
        "totpop",
        "total_pop",
        "total_population",
        "dbpop2021",
        "dbpop_2021",
        "c1_count_total",
        "c1_count",
        "tpop",
    }
    NAME_ALIASES = {
        "geoname",
        "name",
        "da_name",
        "ctname",
        "dauid",
        "ctuid",
        "dguid",
        "geo_uid",
        "geocode",
    }

    def load_population_layer(self, path: str | Path) -> gpd.GeoDataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_VECTOR_SUFFIXES:
            raise ValueError("Unsupported population layer type: " + suffix)
        if suffix == ".parquet":
            gdf = gpd.read_parquet(path)
        elif suffix == ".zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(tmpdir)
                shapefiles = list(Path(tmpdir).rglob("*.shp"))
                if not shapefiles:
                    raise ValueError("Zip upload does not contain a shapefile.")
                gdf = gpd.read_file(shapefiles[0])
        else:
            gdf = gpd.read_file(path)
        return self.prepare_population_layer(gdf)

    def prepare_population_layer(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.empty:
            raise ValueError("Population layer is empty.")
        if "geometry" not in gdf:
            raise ValueError("Population layer must include geometry.")
        prepared = gdf.copy().dropna(subset=["geometry"])
        if prepared.crs is None:
            prepared = prepared.set_crs("EPSG:4326", allow_override=True)
        prepared = prepared.to_crs("EPSG:4326")

        population_column = self.find_population_column(prepared)
        if population_column is None:
            raise ValueError(
                "Could not find a population column. Rename the field to one of: "
                + ", ".join(sorted(self.POPULATION_ALIASES))
            )
        prepared["population"] = pd.to_numeric(prepared[population_column], errors="coerce").fillna(0.0)

        name_column = self.find_name_column(prepared)
        if name_column is not None:
            prepared["area_name"] = prepared[name_column].astype(str)
        else:
            prepared["area_name"] = [f"census_area_{idx + 1}" for idx in range(len(prepared))]
        return prepared

    def clip_to_bbox(self, population_gdf: gpd.GeoDataFrame, bbox: dict[str, float]) -> gpd.GeoDataFrame:
        area = gpd.GeoDataFrame(
            [{"geometry": box(float(bbox["west"]), float(bbox["south"]), float(bbox["east"]), float(bbox["north"]))}],
            crs="EPSG:4326",
        )
        clipped = gpd.overlay(population_gdf.to_crs("EPSG:4326"), area, how="intersection")
        if clipped.empty:
            return clipped

        metric_crs = self.metric_crs(clipped)
        original_metric = population_gdf[["area_name", "population", "geometry"]].to_crs(metric_crs).copy()
        original_metric["_original_area"] = original_metric.geometry.area
        clipped_metric = clipped.to_crs(metric_crs).copy()
        clipped_metric["_clipped_area"] = clipped_metric.geometry.area
        area_lookup = original_metric.set_index("area_name")["_original_area"]
        clipped_metric["_original_area"] = clipped_metric["area_name"].map(area_lookup)
        clipped_metric["_area_share"] = (
            clipped_metric["_clipped_area"] / clipped_metric["_original_area"].replace({0: np.nan})
        ).fillna(1.0).clip(0.0, 1.0)
        clipped_metric["population"] = clipped_metric["population"] * clipped_metric["_area_share"]
        return clipped_metric.drop(columns=["_clipped_area", "_original_area", "_area_share"]).to_crs("EPSG:4326")

    def generate_assumed_population_layer(
        self,
        bbox: dict[str, float],
        total_population: float,
        cell_size_meters: float = 500.0,
        center_concentration: float = 1.8,
    ) -> gpd.GeoDataFrame:
        area = gpd.GeoDataFrame(
            [{"geometry": box(float(bbox["west"]), float(bbox["south"]), float(bbox["east"]), float(bbox["north"]))}],
            crs="EPSG:4326",
        )
        metric_crs = self.metric_crs(area)
        area_metric = area.to_crs(metric_crs).geometry.iloc[0]
        minx, miny, maxx, maxy = area_metric.bounds
        size = max(100.0, float(cell_size_meters))
        center = area_metric.centroid
        max_distance = max(center.distance(box(minx, miny, minx, miny).centroid), 1.0)

        rows: list[dict[str, Any]] = []
        idx = 1
        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                cell = box(x, y, min(x + size, maxx), min(y + size, maxy)).intersection(area_metric)
                if not cell.is_empty and cell.area > 0:
                    centroid = cell.centroid
                    normalized_distance = min(1.0, centroid.distance(center) / max_distance)
                    weight = max(0.15, (1.0 - normalized_distance) ** max(0.1, float(center_concentration)))
                    rows.append({"area_name": f"assumed_cell_{idx}", "weight": weight, "geometry": cell})
                    idx += 1
                x += size
            y += size

        assumed = gpd.GeoDataFrame(rows, geometry="geometry", crs=metric_crs)
        if assumed.empty:
            raise ValueError("Selected area is too small to generate assumed population cells.")
        assumed["population"] = assumed["weight"] / assumed["weight"].sum() * max(0.0, float(total_population))
        assumed["source"] = "assumed_population_grid"
        return assumed.drop(columns=["weight"]).to_crs("EPSG:4326")

    def generate_station_table(
        self,
        population_gdf: gpd.GeoDataFrame,
        config: DemandConfig,
    ) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        filtered = population_gdf[population_gdf["population"] >= float(config.minimum_population)].copy()
        if filtered.empty:
            raise ValueError("No census areas meet the minimum population threshold inside the selected area.")

        metric = filtered.to_crs(self.metric_crs(filtered))
        centroids = metric.geometry.representative_point()
        points = gpd.GeoDataFrame(
            filtered.drop(columns=["geometry"]).copy(),
            geometry=centroids,
            crs=metric.crs,
        ).to_crs("EPSG:4326")

        trips = (
            points["population"].astype(float)
            * float(config.adoption_rate)
            * float(config.daily_trip_rate)
            * int(config.annualization_days)
        )
        points["Trips"] = trips.clip(lower=float(config.minimum_trips))
        points["estimated_docks"] = np.maximum(
            float(config.minimum_docks),
            np.ceil(points["Trips"] / max(1.0, float(config.trips_per_dock))),
        )
        points["Station_Name"] = [
            f"Population Candidate {idx + 1} - {name}"
            for idx, name in enumerate(points["area_name"].astype(str).tolist())
        ]
        points["Latitude"] = points.geometry.y
        points["Longitude"] = points.geometry.x
        if "source" not in points:
            points["source"] = "canadian_population_polygon"

        points = self.apply_candidate_density(points, config)

        station_table = points[
            ["Station_Name", "Latitude", "Longitude", "Trips", "estimated_docks", "population", "area_name", "source"]
        ].copy()
        return station_table.reset_index(drop=True), points.reset_index(drop=True)

    def apply_candidate_density(self, points: gpd.GeoDataFrame, config: DemandConfig) -> gpd.GeoDataFrame:
        spacing = max(0.0, float(config.target_spacing_meters))
        max_candidates = max(0, int(config.max_candidates))
        ranked = points.sort_values("Trips", ascending=False).reset_index(drop=True)
        if spacing <= 0:
            selected = ranked
        else:
            metric = ranked.to_crs(self.metric_crs(ranked))
            selected_indexes: list[int] = []
            selected_geometries = []
            for idx, row in metric.iterrows():
                if selected_geometries and min(row.geometry.distance(geometry) for geometry in selected_geometries) < spacing:
                    continue
                selected_indexes.append(idx)
                selected_geometries.append(row.geometry)
                if max_candidates and len(selected_indexes) >= max_candidates:
                    break
            selected = ranked.loc[selected_indexes]
        if max_candidates:
            selected = selected.head(max_candidates)
        return selected.sort_index().reset_index(drop=True)

    def find_population_column(self, gdf: gpd.GeoDataFrame) -> str | None:
        return self._find_column(gdf.columns, self.POPULATION_ALIASES)

    def find_name_column(self, gdf: gpd.GeoDataFrame) -> str | None:
        return self._find_column(gdf.columns, self.NAME_ALIASES)

    @staticmethod
    def write_uploaded_file(uploaded_file) -> Path:
        suffix = Path(uploaded_file.name or "").suffix
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp.write(uploaded_file.getvalue())
        temp.close()
        return Path(temp.name)

    @staticmethod
    def metric_crs(gdf: gpd.GeoDataFrame) -> str:
        try:
            centroid = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
            if -142 <= centroid.x <= -52 and 41 <= centroid.y <= 84:
                zone = int((centroid.x + 180) // 6) + 1
                epsg = 32600 + zone
                return f"EPSG:{epsg}"
        except Exception:
            pass
        return "EPSG:3857"

    @staticmethod
    def _find_column(columns: Any, aliases: set[str]) -> str | None:
        normalized = {str(column).strip().lower(): str(column) for column in columns}
        for alias in aliases:
            if alias in normalized:
                return normalized[alias]
        for column in columns:
            lower = str(column).strip().lower()
            if any(alias in lower for alias in aliases):
                return str(column)
        return None
