from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import geopandas as gpd
import requests
from shapely.geometry import LineString

from ui.effort import EffortCalculator, fetch_online_endpoint_elevations


class EffortCalculatorTests(unittest.TestCase):
    def test_effort_is_directional(self) -> None:
        edges = gpd.GeoDataFrame(
            [
                {
                    "one_way": False,
                    "elev_start_m": 100.0,
                    "elev_end_m": 108.0,
                    "geometry": LineString([(0.0, 0.0), (100.0, 0.0)]),
                }
            ],
            crs="EPSG:3857",
        )
        scored = EffortCalculator().score_dataframe(edges)
        forward = scored[scored["direction"] == "forward"].iloc[0]
        reverse = scored[scored["direction"] == "reverse"].iloc[0]
        self.assertAlmostEqual(float(forward["uphill_grade_pct"]), 8.0)
        self.assertAlmostEqual(float(reverse["uphill_grade_pct"]), 0.0)
        self.assertGreater(float(forward["effort_exposure"]), float(reverse["effort_exposure"]))

    def test_short_segment_is_exempt(self) -> None:
        self.assertEqual(EffortCalculator().classify(length_m=40.0, uphill_grade_pct=9.0), 3.5)

    def test_flat_edge_has_lowest_steepness_level(self) -> None:
        self.assertEqual(EffortCalculator().classify(length_m=100.0, uphill_grade_pct=0.0), 3.5)

    def test_long_uphill_segment_is_not_always_lowest_level(self) -> None:
        self.assertEqual(EffortCalculator().classify(length_m=100.0, uphill_grade_pct=8.0), 8.0)

    def test_geographic_geometry_is_measured_in_meters(self) -> None:
        edges = gpd.GeoDataFrame(
            [
                {
                    "one_way": True,
                    "elev_start_m": 10.0,
                    "elev_end_m": 18.0,
                    "geometry": LineString([(-73.6, 45.5), (-73.599, 45.5)]),
                }
            ],
            crs="EPSG:4326",
        )
        scored = EffortCalculator().score_dataframe(edges)
        self.assertGreater(float(scored.iloc[0]["length"]), 50.0)
        self.assertEqual(float(scored.iloc[0]["steepness_level"]), 9.5)

    @patch("ui.effort.requests.get")
    def test_online_elevation_uses_open_meteo_by_default(self, get: Mock) -> None:
        get.return_value.json.return_value = {"elevation": [20.0, 28.0]}
        edges = gpd.GeoDataFrame(
            [{"geometry": LineString([(-73.60, 45.50), (-73.59, 45.51)])}],
            crs="EPSG:4326",
        )

        sampled = fetch_online_endpoint_elevations(edges)

        get.return_value.raise_for_status.assert_called_once()
        self.assertEqual(sampled.iloc[0]["elev_start_m"], 20.0)
        self.assertEqual(sampled.iloc[0]["elev_end_m"], 28.0)
        self.assertEqual(sampled.iloc[0]["elevation_source"], "Open-Meteo Copernicus DEM GLO-90")

    @patch("ui.effort.requests.post")
    @patch("ui.effort.requests.get")
    def test_online_elevation_falls_back_when_primary_times_out(self, get: Mock, post: Mock) -> None:
        get.side_effect = requests.Timeout("timeout")
        post.return_value.json.return_value = {"results": [{"elevation": 20.0}, {"elevation": 28.0}]}
        edges = gpd.GeoDataFrame(
            [{"geometry": LineString([(-73.60, 45.50), (-73.59, 45.51)])}],
            crs="EPSG:4326",
        )

        sampled = fetch_online_endpoint_elevations(edges)

        post.return_value.raise_for_status.assert_called_once()
        self.assertEqual(sampled.iloc[0]["elev_end_m"], 28.0)
        self.assertEqual(sampled.iloc[0]["elevation_source"], "Open-Elevation fallback")


if __name__ == "__main__":
    unittest.main()
