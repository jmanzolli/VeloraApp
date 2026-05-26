from __future__ import annotations

import unittest

import geopandas as gpd
from shapely.geometry import LineString

from ui.lts import LTSCalculator, LTSFeatureExtractor


class LTSCalculatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = LTSCalculator()

    def test_mixed_low_speed_local_street_returns_low_lts(self) -> None:
        result = self.calculator.calculate(
            {
                "road_class": "residential",
                "bike_facility_type": "none",
                "one_way": False,
                "centerline": False,
                "lanes_per_direction": 1,
                "speed_q85_kmh": 35,
                "adt": 500,
                "confidence": 1.0,
            }
        )
        self.assertEqual(result.lts, 1)

    def test_high_speed_multilane_arterial_returns_high_lts(self) -> None:
        result = self.calculator.calculate(
            {
                "road_class": "primary",
                "bike_facility_type": "none",
                "one_way": False,
                "centerline": True,
                "lanes_per_direction": 3,
                "speed_q85_kmh": 65,
                "adt": 15000,
                "confidence": 1.0,
            }
        )
        self.assertEqual(result.lts, 4)

    def test_protected_path_uses_driveway_visibility(self) -> None:
        no_driveways = self.calculator.calculate(
            {
                "road_class": "cycleway",
                "bike_facility_type": "path",
                "path_driveway_visible": False,
                "winter_maintained": True,
                "confidence": 1.0,
            }
        )
        driveways = self.calculator.calculate(
            {
                "road_class": "cycleway",
                "bike_facility_type": "path",
                "path_driveway_visible": True,
                "winter_maintained": True,
                "confidence": 1.0,
            }
        )
        self.assertEqual(no_driveways.lts, 1)
        self.assertEqual(driveways.lts, 2)

    def test_painted_bike_lane_thresholds_with_and_without_parking(self) -> None:
        without_parking = self.calculator.calculate(
            {
                "road_class": "secondary",
                "bike_facility_type": "bike_lane",
                "one_way": False,
                "centerline": True,
                "lanes_per_direction": 1,
                "speed_q85_kmh": 50,
                "bike_lane_width_m": 1.8,
                "effective_reach_m": 1.8,
                "parking_width_m": 0,
                "confidence": 1.0,
            }
        )
        with_parking = self.calculator.calculate(
            {
                "road_class": "secondary",
                "bike_facility_type": "bike_lane",
                "one_way": False,
                "centerline": True,
                "lanes_per_direction": 1,
                "speed_q85_kmh": 50,
                "bike_lane_width_m": 1.5,
                "effective_reach_m": 3.6,
                "parking_width_m": 2.1,
                "confidence": 1.0,
            }
        )
        self.assertEqual(without_parking.lts, 1)
        self.assertEqual(with_parking.lts, 2)


class LTSFeatureExtractorTests(unittest.TestCase):
    def test_osm_tags_normalize_to_schema(self) -> None:
        extractor = LTSFeatureExtractor()
        gdf = gpd.GeoDataFrame(
            [
                {
                    "osmid": 10,
                    "highway": "residential",
                    "oneway": "no",
                    "lanes": "2",
                    "maxspeed": "30",
                    "cycleway:right": "lane",
                    "parking:right": "lane",
                    "geometry": LineString([(-73.6, 45.5), (-73.59, 45.5)]),
                }
            ],
            crs="EPSG:4326",
        )
        normalized = extractor.normalize(gdf)
        row = normalized.iloc[0]
        self.assertEqual(row["road_class"], "residential")
        self.assertEqual(row["bike_facility_type"], "bike_lane")
        self.assertEqual(row["lanes_per_direction"], 1.0)
        self.assertGreater(row["parking_width_m"], 0.0)

    def test_municipal_overrides_osm_values(self) -> None:
        extractor = LTSFeatureExtractor()
        gdf = gpd.GeoDataFrame(
            [
                {
                    "highway": "residential",
                    "maxspeed": "30",
                    "municipal_speed_q85_kmh": 55,
                    "municipal_lanes_per_direction": 2,
                    "geometry": LineString([(-73.6, 45.5), (-73.59, 45.5)]),
                }
            ],
            crs="EPSG:4326",
        )
        normalized = extractor.normalize(gdf)
        row = normalized.iloc[0]
        self.assertEqual(row["speed_q85_kmh"], 55.0)
        self.assertEqual(row["lanes_per_direction"], 2.0)
        self.assertEqual(row["source_priority"], "municipal+osm")

    def test_missing_values_record_conservative_fallbacks(self) -> None:
        extractor = LTSFeatureExtractor()
        calculator = LTSCalculator()
        gdf = gpd.GeoDataFrame(
            [{"highway": "primary", "geometry": LineString([(-73.6, 45.5), (-73.59, 45.5)])}],
            crs="EPSG:4326",
        )
        normalized = extractor.normalize(gdf)
        scored = calculator.score_dataframe(normalized)
        self.assertIn("speed_q85_kmh", scored.iloc[0]["missing_inputs"])
        self.assertGreaterEqual(scored.iloc[0]["lts"], 3)


if __name__ == "__main__":
    unittest.main()
