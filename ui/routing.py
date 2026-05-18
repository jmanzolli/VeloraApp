from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from plotly.utils import PlotlyJSONEncoder
from shapely.geometry import LineString, MultiLineString, Point

from ui.optimizer import load_network


@dataclass(frozen=True)
class RouteEndpoint:
    label: str
    query: str
    latitude: float
    longitude: float
    node: tuple[float, float]


@dataclass(frozen=True)
class RouteResult:
    route_name: str
    labels: list[str]
    path: list[tuple[float, float]]
    geometry: LineString
    distance_m: float
    mean_lts: float
    max_lts: float
    stress_exposure: float
    elevation_gain_m: float
    max_uphill_grade_pct: float
    mean_effort: float | None
    effort_exposure: float | None
    max_steepness_level: float | None
    sl_exceedance_m: float | None
    estimated_minutes: float
    lts_share: dict[int, float]

    def summary_row(self) -> dict[str, Any]:
        return {
            "Route": " / ".join(self.labels),
            "Distance (km)": self.distance_m / 1000.0,
            "Mean LTS": self.mean_lts,
            "Max LTS": self.max_lts,
            "Stress Exposure": self.stress_exposure,
            "Elevation Gain (m)": self.elevation_gain_m,
            "Max Uphill Grade (%)": self.max_uphill_grade_pct,
            "Mean Effort": self.mean_effort,
            "Effort Exposure": self.effort_exposure,
            "Max Steepness Level": self.max_steepness_level,
            "SL Exceedance (m)": self.sl_exceedance_m,
            "Estimated Time (min)": self.estimated_minutes,
            "LTS 1 Share": self.lts_share.get(1, 0.0),
            "LTS 2 Share": self.lts_share.get(2, 0.0),
            "LTS 3 Share": self.lts_share.get(3, 0.0),
            "LTS 4 Share": self.lts_share.get(4, 0.0),
        }


class RoutePlanner:
    """Point-to-point route calculations over an LTS-scored network."""

    def load_lts_network(self, uploaded_path: str | Path) -> gpd.GeoDataFrame:
        network = load_network(uploaded_path)
        if network.crs is None:
            network = network.set_crs("EPSG:4326", allow_override=True)
        return network

    def build_graph(self, network_gdf: gpd.GeoDataFrame) -> nx.DiGraph:
        if network_gdf.empty:
            raise ValueError("The uploaded LTS network is empty.")

        metric_crs = self.metric_crs(network_gdf)
        network_metric = network_gdf.to_crs(metric_crs).copy()
        graph = nx.DiGraph()

        for _, row in network_metric.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            parts = [geom] if isinstance(geom, LineString) else list(geom.geoms) if isinstance(geom, MultiLineString) else []
            for part in parts:
                coords = list(part.coords)
                for u, v in zip(coords[:-1], coords[1:]):
                    line = LineString([u, v])
                    length_m = float(line.length)
                    if length_m <= 0:
                        continue
                    lts = float(np.clip(float(row.get("lts", 4)), 1, 4))
                    stress_exposure = length_m * lts
                    graph.add_edge(
                        tuple(u),
                        tuple(v),
                        geometry=line,
                        length_m=length_m,
                        lts=lts,
                        stress_exposure=stress_exposure,
                        uphill_gain_m=float(row.get("uphill_gain_m", 0.0)) if pd.notna(row.get("uphill_gain_m")) else 0.0,
                        uphill_grade_pct=float(row.get("uphill_grade_pct", 0.0)) if pd.notna(row.get("uphill_grade_pct")) else 0.0,
                        effort_exposure=float(row.get("effort_exposure")) if pd.notna(row.get("effort_exposure")) else np.nan,
                        steepness_level=float(row.get("steepness_level")) if pd.notna(row.get("steepness_level")) else np.nan,
                    )
                    if not bool(row.get("one_way", False)) and str(row.get("direction", "")).lower() not in {"forward", "reverse"}:
                        graph.add_edge(
                            tuple(v),
                            tuple(u),
                            geometry=LineString([v, u]),
                            length_m=length_m,
                            lts=lts,
                            stress_exposure=stress_exposure,
                            uphill_gain_m=0.0,
                            uphill_grade_pct=0.0,
                            effort_exposure=float(row.get("effort_exposure")) if pd.notna(row.get("effort_exposure")) else np.nan,
                            steepness_level=float(row.get("steepness_level")) if pd.notna(row.get("steepness_level")) else np.nan,
                        )

        if graph.number_of_edges() == 0:
            raise ValueError("The uploaded LTS network did not produce any routable graph edges.")
        graph.graph["crs"] = metric_crs
        return graph

    def geocode_endpoint(self, label: str, place_text: str, city_context: str, graph: nx.Graph) -> RouteEndpoint:
        try:
            import osmnx as ox
        except ImportError as exc:
            raise RuntimeError("Route geocoding requires osmnx. Install project requirements and try again.") from exc

        query = f"{place_text}, {city_context}" if city_context.strip() else place_text
        try:
            latitude, longitude = ox.geocode(query)
        except Exception as exc:
            raise ValueError(f"Could not geocode {label}: {query}. Try a more specific street or place name.") from exc
        node = self.snap_lonlat_to_graph(longitude=longitude, latitude=latitude, graph=graph)
        return RouteEndpoint(
            label=label,
            query=query,
            latitude=float(latitude),
            longitude=float(longitude),
            node=node,
        )

    def snap_lonlat_to_graph(self, longitude: float, latitude: float, graph: nx.Graph) -> tuple[float, float]:
        point_4326 = gpd.GeoSeries([Point(float(longitude), float(latitude))], crs="EPSG:4326")
        point_metric = point_4326.to_crs(graph.graph["crs"]).iloc[0]
        return self.nearest_graph_node(point_metric, graph)

    @staticmethod
    def nearest_graph_node(point: Point, graph: nx.Graph) -> tuple[float, float]:
        nodes = list(graph.nodes)
        if not nodes:
            raise ValueError("Graph has no nodes.")
        distances = np.array([point.distance(Point(node)) for node in nodes], dtype=float)
        return nodes[int(np.argmin(distances))]

    def calculate_routes(
        self,
        graph: nx.Graph,
        origin_node: tuple[float, float],
        destination_node: tuple[float, float],
        balanced_stress_weight: float = 0.34,
        balanced_effort_weight: float = 0.33,
        steepness_threshold: float = 5.0,
    ) -> list[RouteResult]:
        if origin_node == destination_node:
            raise ValueError("Origin and destination snapped to the same network node.")

        self.assign_balanced_cost(graph, balanced_stress_weight, balanced_effort_weight)
        objectives = [
            ("Shortest Distance", "length_m"),
            ("Lowest LTS", "stress_exposure"),
            ("Balanced Comfort", "balanced_cost"),
        ]
        if self.effort_available(graph):
            objectives.insert(2, ("Lowest Effort", "effort_exposure"))
        by_path: dict[tuple[tuple[float, float], ...], RouteResult] = {}

        for label, weight in objectives:
            try:
                path = nx.shortest_path(graph, origin_node, destination_node, weight=weight)
            except nx.NetworkXNoPath as exc:
                raise ValueError("No route connects point A and point B in the uploaded LTS network.") from exc
            key = tuple(path)
            if key in by_path:
                by_path[key].labels.append(label)
            else:
                by_path[key] = self.summarize_path(label, [label], path, graph, steepness_threshold=steepness_threshold)

        return list(by_path.values())

    def assign_balanced_cost(self, graph: nx.Graph, stress_weight: float, effort_weight: float) -> None:
        stress = float(np.clip(stress_weight, 0.0, 1.0))
        effort = float(np.clip(effort_weight, 0.0, 1.0))
        distance = max(0.0, 1.0 - stress - effort)
        max_length = max((float(data["length_m"]) for _, _, data in graph.edges(data=True)), default=1.0)
        max_stress = max((float(data["stress_exposure"]) for _, _, data in graph.edges(data=True)), default=1.0)
        max_effort = max((float(data["effort_exposure"]) for _, _, data in graph.edges(data=True) if pd.notna(data.get("effort_exposure"))), default=1.0)
        for _, _, data in graph.edges(data=True):
            normalized_length = float(data["length_m"]) / max(max_length, 1e-9)
            normalized_stress = float(data["stress_exposure"]) / max(max_stress, 1e-9)
            normalized_effort = (
                float(data["effort_exposure"]) / max(max_effort, 1e-9)
                if pd.notna(data.get("effort_exposure"))
                else 0.0
            )
            data["balanced_cost"] = distance * normalized_length + stress * normalized_stress + effort * normalized_effort

    def summarize_path(
        self,
        route_name: str,
        labels: list[str],
        path: list[tuple[float, float]],
        graph: nx.Graph,
        cycling_speed_kmh: float = 15.0,
        steepness_threshold: float = 5.0,
    ) -> RouteResult:
        edge_rows: list[dict[str, float]] = []
        for u, v in zip(path[:-1], path[1:]):
            if not graph.has_edge(u, v):
                continue
            data = graph.get_edge_data(u, v)
            edge_rows.append(
                {
                    "length_m": float(data["length_m"]),
                    "lts": float(data["lts"]),
                    "stress_exposure": float(data["stress_exposure"]),
                    "uphill_gain_m": float(data.get("uphill_gain_m", 0.0)),
                    "uphill_grade_pct": float(data.get("uphill_grade_pct", 0.0)),
                    "effort_exposure": float(data["effort_exposure"]) if pd.notna(data.get("effort_exposure")) else np.nan,
                    "steepness_level": float(data["steepness_level"]) if pd.notna(data.get("steepness_level")) else np.nan,
                }
            )
        if not edge_rows:
            raise ValueError("Route path has no graph edges.")

        distance_m = float(sum(row["length_m"] for row in edge_rows))
        stress_exposure = float(sum(row["stress_exposure"] for row in edge_rows))
        mean_lts = stress_exposure / max(distance_m, 1e-9)
        max_lts = float(max(row["lts"] for row in edge_rows))
        effort_values = [row["effort_exposure"] for row in edge_rows if pd.notna(row["effort_exposure"])]
        effort_exposure = float(sum(effort_values)) if effort_values else None
        mean_effort = effort_exposure / max(distance_m, 1e-9) if effort_exposure is not None else None
        elevation_gain_m = float(sum(row["uphill_gain_m"] for row in edge_rows))
        max_uphill_grade_pct = float(max(row["uphill_grade_pct"] for row in edge_rows))
        sl_values = [row["steepness_level"] for row in edge_rows if pd.notna(row["steepness_level"])]
        max_steepness_level = float(max(sl_values)) if sl_values else None
        sl_exceedance_m = (
            float(sum(row["length_m"] for row in edge_rows if pd.notna(row["steepness_level"]) and row["steepness_level"] > steepness_threshold))
            if sl_values
            else None
        )
        lts_share = self.lts_share(edge_rows, distance_m)
        geometry = LineString(path)
        estimated_minutes = (distance_m / 1000.0) / max(cycling_speed_kmh, 1e-9) * 60.0
        return RouteResult(
            route_name=route_name,
            labels=labels,
            path=path,
            geometry=geometry,
            distance_m=distance_m,
            mean_lts=mean_lts,
            max_lts=max_lts,
            stress_exposure=stress_exposure,
            elevation_gain_m=elevation_gain_m,
            max_uphill_grade_pct=max_uphill_grade_pct,
            mean_effort=mean_effort,
            effort_exposure=effort_exposure,
            max_steepness_level=max_steepness_level,
            sl_exceedance_m=sl_exceedance_m,
            estimated_minutes=estimated_minutes,
            lts_share=lts_share,
        )

    @staticmethod
    def lts_share(edge_rows: list[dict[str, float]], distance_m: float) -> dict[int, float]:
        shares = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for row in edge_rows:
            level = int(np.clip(round(float(row["lts"])), 1, 4))
            shares[level] += float(row["length_m"]) / max(distance_m, 1e-9)
        return shares

    @staticmethod
    def routes_to_geodataframe(routes: list[RouteResult], crs: Any) -> gpd.GeoDataFrame:
        rows = []
        for route in routes:
            rows.append(
                {
                    "route_name": " / ".join(route.labels),
                    "distance_m": route.distance_m,
                    "mean_lts": route.mean_lts,
                    "max_lts": route.max_lts,
                    "stress_exposure": route.stress_exposure,
                    "elevation_gain_m": route.elevation_gain_m,
                    "max_uphill_grade_pct": route.max_uphill_grade_pct,
                    "mean_effort": route.mean_effort,
                    "effort_exposure": route.effort_exposure,
                    "max_steepness_level": route.max_steepness_level,
                    "sl_exceedance_m": route.sl_exceedance_m,
                    "estimated_minutes": route.estimated_minutes,
                    "lts_1_share": route.lts_share.get(1, 0.0),
                    "lts_2_share": route.lts_share.get(2, 0.0),
                    "lts_3_share": route.lts_share.get(3, 0.0),
                    "lts_4_share": route.lts_share.get(4, 0.0),
                    "geometry": route.geometry,
                }
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_crs("EPSG:4326")

    def export_routes_geojson(self, routes: list[RouteResult], graph: nx.Graph) -> bytes:
        routes_gdf = self.routes_to_geodataframe(routes, graph.graph["crs"])
        payload = json.loads(routes_gdf.to_json())
        return json.dumps(payload, cls=PlotlyJSONEncoder).encode("utf-8")

    @staticmethod
    def effort_available(graph: nx.Graph) -> bool:
        return any(pd.notna(data.get("effort_exposure")) for _, _, data in graph.edges(data=True))

    @staticmethod
    def metric_crs(gdf: gpd.GeoDataFrame) -> str:
        try:
            centroid = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
            if -142 <= centroid.x <= -52 and 41 <= centroid.y <= 84:
                zone = int((centroid.x + 180) // 6) + 1
                return f"EPSG:{32600 + zone}"
        except Exception:
            pass
        return "EPSG:3857"

    @staticmethod
    def write_uploaded_file(uploaded_file) -> Path:
        suffix = Path(uploaded_file.name or "").suffix
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp.write(uploaded_file.getvalue())
        temp.close()
        return Path(temp.name)
