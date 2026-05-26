from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from ui.lts.features import LTSFeatureExtractor


SUPPORTED_VECTOR_SUFFIXES = {".gpkg", ".geojson", ".json", ".shp", ".parquet", ".zip"}


@dataclass(frozen=True)
class BoundingBox:
    north: float
    south: float
    east: float
    west: float

    def as_osmnx_bbox(self) -> tuple[float, float, float, float]:
        return self.west, self.south, self.east, self.north


class CityNetworkBuilder:
    """Build a segment-level street network from OSM and optional municipal layers."""

    def fetch_osm_edges(self, bbox: BoundingBox, network_type: str = "bike") -> gpd.GeoDataFrame:
        try:
            import osmnx as ox
        except ImportError as exc:
            raise RuntimeError("OSM fetching requires osmnx. Install project requirements and try again.") from exc

        useful_way_tags = set(getattr(ox.settings, "useful_tags_way", []))
        useful_way_tags.update(
            {
                "adt",
                "bicycle",
                "cycleway",
                "cycleway:both",
                "cycleway:left",
                "cycleway:right",
                "cycleway:width",
                "dual_carriageway",
                "highway",
                "lanes",
                "maxspeed",
                "oneway",
                "oneway:bicycle",
                "parking",
                "parking:both",
                "parking:left",
                "parking:right",
                "parking:lane:both",
                "parking:lane:left",
                "parking:lane:right",
                "segregated",
                "surface",
            }
        )
        ox.settings.useful_tags_way = sorted(useful_way_tags)

        try:
            graph = ox.graph_from_bbox(
                bbox=bbox.as_osmnx_bbox(),
                network_type=network_type,
                simplify=True,
            )
        except TypeError:
            graph = ox.graph_from_bbox(
                bbox.north,
                bbox.south,
                bbox.east,
                bbox.west,
                network_type=network_type,
                simplify=True,
            )
        edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True).reset_index()
        return self.prepare_edges(edges)

    def load_vector_layer(self, path: str | Path) -> gpd.GeoDataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_VECTOR_SUFFIXES:
            raise ValueError("Unsupported vector file type: " + suffix)
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
        return self.prepare_edges(gdf)

    def prepare_edges(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf
        prepared = gdf.copy()
        if "geometry" not in prepared:
            raise ValueError("Network layer must include geometry.")
        prepared = prepared.dropna(subset=["geometry"])
        prepared = prepared[prepared.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        prepared = prepared.explode(index_parts=False).reset_index(drop=True)
        if prepared.crs is None:
            prepared = prepared.set_crs("EPSG:4326", allow_override=True)
        prepared["segment_id"] = [
            str(row.get("segment_id") or row.get("osmid") or row.get("id") or f"seg_{idx:06d}")
            for idx, row in prepared.iterrows()
        ]
        return prepared

    def build_network(
        self,
        bbox: BoundingBox | None = None,
        base_network: gpd.GeoDataFrame | None = None,
        municipal_layers: Iterable[gpd.GeoDataFrame] | None = None,
    ) -> gpd.GeoDataFrame:
        if base_network is None:
            if bbox is None:
                raise ValueError("Provide either a bounding box or an uploaded base network.")
            base = self.fetch_osm_edges(bbox)
        else:
            base = self.prepare_edges(base_network)
        return self.conflate_municipal_layers(base, list(municipal_layers or []))

    def conflate_municipal_layers(
        self,
        base_edges: gpd.GeoDataFrame,
        municipal_layers: list[gpd.GeoDataFrame],
        max_distance_meters: float = 25.0,
    ) -> gpd.GeoDataFrame:
        if base_edges.empty or not municipal_layers:
            return base_edges

        enriched = base_edges.copy()
        metric_crs = self._metric_crs(enriched)
        base_metric = enriched.to_crs(metric_crs)
        aliases = LTSFeatureExtractor.MUNICIPAL_FIELD_ALIASES

        for layer in municipal_layers:
            if layer.empty:
                continue
            municipal = self.prepare_edges(layer)
            if municipal.empty:
                continue
            municipal_metric = municipal.to_crs(metric_crs)
            matches = gpd.sjoin_nearest(
                base_metric[["segment_id", "geometry"]],
                municipal_metric,
                how="left",
                max_distance=max_distance_meters,
                distance_col="_municipal_distance_m",
            )
            matches = matches.sort_values("_municipal_distance_m").drop_duplicates("segment_id")
            matches = matches.set_index("segment_id")
            for normalized_field, candidates in aliases.items():
                source_column = self._find_column(matches.columns, candidates)
                if source_column is None:
                    continue
                target = f"municipal_{normalized_field}"
                values = enriched["segment_id"].map(matches[source_column])
                if target not in enriched:
                    enriched[target] = pd.NA
                enriched[target] = enriched[target].where(enriched[target].notna(), values)
        return enriched

    @staticmethod
    def write_uploaded_file(uploaded_file) -> Path:
        suffix = Path(uploaded_file.name or "").suffix
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp.write(uploaded_file.getvalue())
        temp.close()
        return Path(temp.name)

    @staticmethod
    def _find_column(columns: Iterable[Any], candidates: set[str]) -> str | None:
        normalized = {str(column).strip().lower(): str(column) for column in columns}
        for candidate in candidates:
            if candidate.lower() in normalized:
                return normalized[candidate.lower()]
        return None

    @staticmethod
    def _metric_crs(gdf: gpd.GeoDataFrame) -> str:
        try:
            centroid = gdf.to_crs("EPSG:4326").unary_union.centroid
            if -75.0 <= centroid.x <= -73.0 and 44.5 <= centroid.y <= 46.5:
                return "EPSG:2950"
        except Exception:
            pass
        return "EPSG:3857"


def geometry_to_linestring(geometry: Any) -> LineString | MultiLineString | None:
    if isinstance(geometry, (LineString, MultiLineString)):
        return geometry
    return None
