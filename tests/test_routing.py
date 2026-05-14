from __future__ import annotations

import json
import unittest

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

from ui.routing import RoutePlanner


class RoutePlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = RoutePlanner()

    def build_test_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.graph["crs"] = "EPSG:3857"

        def add_edge(u, v, length_m: float, lts: float) -> None:
            graph.add_edge(
                u,
                v,
                length_m=length_m,
                lts=lts,
                stress_exposure=length_m * lts,
                geometry=LineString([u, v]),
            )

        add_edge((0.0, 0.0), (1.0, 0.0), 1.0, 4.0)
        add_edge((1.0, 0.0), (2.0, 0.0), 1.0, 4.0)
        add_edge((0.0, 0.0), (0.0, 2.0), 2.0, 1.0)
        add_edge((0.0, 2.0), (2.0, 2.0), 2.0, 1.0)
        add_edge((2.0, 2.0), (2.0, 0.0), 2.0, 1.0)
        return graph

    def test_shortest_and_lowest_lts_choose_different_paths(self) -> None:
        graph = self.build_test_graph()
        routes = self.planner.calculate_routes(graph, (0.0, 0.0), (2.0, 0.0), balanced_stress_weight=0.5)
        by_label = {label: route for route in routes for label in route.labels}

        self.assertEqual(by_label["Shortest Distance"].distance_m, 2.0)
        self.assertEqual(by_label["Shortest Distance"].mean_lts, 4.0)
        self.assertEqual(by_label["Lowest LTS"].distance_m, 6.0)
        self.assertEqual(by_label["Lowest LTS"].mean_lts, 1.0)

    def test_balanced_route_changes_with_weight(self) -> None:
        graph = self.build_test_graph()
        distance_weighted = self.planner.calculate_routes(graph, (0.0, 0.0), (2.0, 0.0), balanced_stress_weight=0.0)
        stress_weighted = self.planner.calculate_routes(graph, (0.0, 0.0), (2.0, 0.0), balanced_stress_weight=1.0)

        distance_balanced = next(route for route in distance_weighted for label in route.labels if label == "Balanced")
        stress_balanced = next(route for route in stress_weighted for label in route.labels if label == "Balanced")
        self.assertEqual(distance_balanced.distance_m, 2.0)
        self.assertEqual(stress_balanced.distance_m, 6.0)

    def test_nearest_graph_node_snaps_to_closest_node(self) -> None:
        graph = self.build_test_graph()
        node = self.planner.nearest_graph_node(Point(0.1, 0.1), graph)
        self.assertEqual(node, (0.0, 0.0))

    def test_disconnected_graph_raises_clear_error(self) -> None:
        graph = self.build_test_graph()
        graph.add_node((10.0, 10.0))
        with self.assertRaisesRegex(ValueError, "No route connects"):
            self.planner.calculate_routes(graph, (0.0, 0.0), (10.0, 10.0), balanced_stress_weight=0.5)

    def test_route_geojson_export_includes_metadata(self) -> None:
        graph = self.build_test_graph()
        routes = self.planner.calculate_routes(graph, (0.0, 0.0), (2.0, 0.0), balanced_stress_weight=0.5)
        payload = json.loads(self.planner.export_routes_geojson(routes, graph).decode("utf-8"))
        self.assertEqual(payload["type"], "FeatureCollection")
        self.assertGreaterEqual(len(payload["features"]), 2)
        self.assertIn("route_name", payload["features"][0]["properties"])

    def test_build_graph_uses_lts_and_lengths(self) -> None:
        gdf = gpd.GeoDataFrame(
            [
                {"lts": 3, "geometry": LineString([(-73.6, 45.5), (-73.59, 45.5)])},
            ],
            crs="EPSG:4326",
        )
        graph = self.planner.build_graph(gdf)
        edge_data = next(iter(graph.edges(data=True)))[2]
        self.assertEqual(edge_data["lts"], 3.0)
        self.assertGreater(edge_data["length_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
