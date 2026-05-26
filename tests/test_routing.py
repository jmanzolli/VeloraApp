from __future__ import annotations

import json
import unittest
from unittest.mock import patch

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

        distance_balanced = next(route for route in distance_weighted for label in route.labels if label == "Balanced Comfort")
        stress_balanced = next(route for route in stress_weighted for label in route.labels if label == "Balanced Comfort")
        self.assertEqual(distance_balanced.distance_m, 2.0)
        self.assertEqual(stress_balanced.distance_m, 6.0)

    def test_nearest_graph_node_snaps_to_closest_node(self) -> None:
        graph = self.build_test_graph()
        node = self.planner.nearest_graph_node(Point(0.1, 0.1), graph)
        self.assertEqual(node, (0.0, 0.0))

    def test_selected_points_snap_to_connected_nearby_nodes(self) -> None:
        graph = nx.DiGraph()
        graph.graph["crs"] = "EPSG:3857"
        graph.add_edge((0.0, 0.0), (0.0, 1.0))
        graph.add_edge((1.0, 0.0), (2.0, 0.0))

        origin, destination = self.planner.snap_connected_endpoint_nodes(
            0.0,
            0.0,
            2.0,
            0.0,
            graph,
            candidate_count=3,
        )

        self.assertEqual(origin, (1.0, 0.0))
        self.assertEqual(destination, (2.0, 0.0))

    def test_effort_network_offers_lowest_effort_route(self) -> None:
        graph = nx.DiGraph()
        graph.graph["crs"] = "EPSG:3857"
        for u, v, length, effort in [
            ((0.0, 0.0), (2.0, 0.0), 2.0, 20.0),
            ((0.0, 0.0), (0.0, 2.0), 2.0, 2.0),
            ((0.0, 2.0), (2.0, 0.0), 2.0, 2.0),
        ]:
            graph.add_edge(
                u,
                v,
                geometry=LineString([u, v]),
                length_m=length,
                lts=2.0,
                stress_exposure=length * 2.0,
                uphill_gain_m=0.0,
                uphill_grade_pct=0.0,
                effort_exposure=effort,
                steepness_level=3.5,
            )

        routes = self.planner.calculate_routes(graph, (0.0, 0.0), (2.0, 0.0))
        by_label = {label: route for route in routes for label in route.labels}

        self.assertIn("Lowest Effort", by_label)
        self.assertEqual(by_label["Lowest Effort"].distance_m, 4.0)

    @patch("osmnx.geocode")
    def test_geocode_falls_back_to_quebec_for_montreal_area_places(self, geocode) -> None:
        geocode.side_effect = [ValueError("not in municipality"), (45.4856917, -73.596562)]

        endpoint = self.planner.geocode_endpoint("Destination", "Westmount", "Montreal, Quebec, Canada", self.build_test_graph())

        self.assertEqual(endpoint.query, "Westmount, Quebec, Canada")
        self.assertEqual(geocode.call_args_list[0].args[0], "Westmount, Montreal, Quebec, Canada")
        self.assertEqual(geocode.call_args_list[1].args[0], "Westmount, Quebec, Canada")

    @patch("osmnx.geocode")
    def test_geocode_does_not_duplicate_context_for_selected_suggestion(self, geocode) -> None:
        geocode.return_value = (45.5088, -73.5540)

        self.planner.geocode_endpoint(
            "Destination",
            "Ville-Marie, Montreal, Quebec, Canada",
            "Montreal, Quebec, Canada",
            self.build_test_graph(),
        )

        self.assertEqual(geocode.call_args.args[0], "Ville-Marie, Montreal, Quebec, Canada")

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
