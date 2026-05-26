from __future__ import annotations

import unittest

import geopandas as gpd
from shapely.geometry import box

from ui.demand import CanadianPopulationDemandBuilder, DemandConfig
from ui.optimizer import validate_station_table


class CanadianPopulationDemandBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = CanadianPopulationDemandBuilder()

    def test_population_alias_is_detected_and_station_schema_is_valid(self) -> None:
        population = gpd.GeoDataFrame(
            [
                {
                    "DGUID": "area-1",
                    "DBPOP2021": 1000,
                    "geometry": box(-73.6, 45.5, -73.59, 45.51),
                }
            ],
            crs="EPSG:4326",
        )
        prepared = self.builder.prepare_population_layer(population)
        station_table, points = self.builder.generate_station_table(
            prepared,
            DemandConfig(adoption_rate=0.1, daily_trip_rate=0.5, annualization_days=100),
        )

        validate_station_table(station_table)
        self.assertEqual(len(points), 1)
        self.assertEqual(station_table.iloc[0]["Trips"], 5000.0)
        self.assertGreaterEqual(station_table.iloc[0]["estimated_docks"], 8.0)

    def test_clip_to_bbox_area_weights_population(self) -> None:
        population = gpd.GeoDataFrame(
            [
                {
                    "name": "area-1",
                    "population": 1000,
                    "geometry": box(0, 0, 2, 2),
                }
            ],
            crs="EPSG:4326",
        )
        prepared = self.builder.prepare_population_layer(population)
        clipped = self.builder.clip_to_bbox(prepared, {"west": 0, "south": 0, "east": 1, "north": 2})
        self.assertEqual(len(clipped), 1)
        self.assertAlmostEqual(float(clipped.iloc[0]["population"]), 500.0, delta=25.0)

    def test_empty_threshold_raises_clear_error(self) -> None:
        population = gpd.GeoDataFrame(
            [{"population": 10, "geometry": box(-73.6, 45.5, -73.59, 45.51)}],
            crs="EPSG:4326",
        )
        prepared = self.builder.prepare_population_layer(population)
        with self.assertRaisesRegex(ValueError, "minimum population"):
            self.builder.generate_station_table(prepared, DemandConfig(minimum_population=50))

    def test_generate_assumed_population_layer_matches_requested_total(self) -> None:
        assumed = self.builder.generate_assumed_population_layer(
            {"north": 45.53, "south": 45.50, "east": -73.56, "west": -73.61},
            total_population=10000,
            cell_size_meters=500,
            center_concentration=1.5,
        )
        self.assertGreater(len(assumed), 1)
        self.assertAlmostEqual(float(assumed["population"].sum()), 10000.0, places=6)
        self.assertTrue((assumed["source"] == "assumed_population_grid").all())


if __name__ == "__main__":
    unittest.main()
