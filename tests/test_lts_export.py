from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString

from ui.lts.export import export_network_bytes, prepare_optimizer_export
from ui.optimizer import load_network


class LTSExportTests(unittest.TestCase):
    def test_exported_geojson_can_be_loaded_by_optimizer(self) -> None:
        scored = gpd.GeoDataFrame(
            [
                {
                    "segment_id": "a",
                    "lts": 2,
                    "lts_raw": 2,
                    "confidence": 0.9,
                    "geometry": LineString([(-73.6, 45.5), (-73.59, 45.5)]),
                },
                {
                    "segment_id": "freeway",
                    "lts": 4,
                    "lts_raw": 6,
                    "confidence": 0.9,
                    "geometry": LineString([(-73.6, 45.51), (-73.59, 45.51)]),
                },
            ],
            crs="EPSG:4326",
        )
        exportable = prepare_optimizer_export(scored, include_inaccessible=False)
        self.assertEqual(len(exportable), 1)

        payload, _, _ = export_network_bytes(exportable, "GeoJSON")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "network.geojson"
            path.write_bytes(payload)
            loaded = load_network(path)
        self.assertEqual(len(loaded), 1)
        self.assertIn("lts", loaded.columns)
        self.assertEqual(float(loaded.iloc[0]["lts"]), 2.0)


if __name__ == "__main__":
    unittest.main()
