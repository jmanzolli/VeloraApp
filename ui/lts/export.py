from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd


def prepare_optimizer_export(scored: gpd.GeoDataFrame, include_inaccessible: bool = False) -> gpd.GeoDataFrame:
    export = scored.copy()
    if not include_inaccessible and "lts_raw" in export:
        export = export[export["lts_raw"].astype(float) <= 4].copy()
    export["lts"] = export["lts"].astype(float).clip(1, 4)
    return export.dropna(subset=["geometry"]).reset_index(drop=True)


def export_network_bytes(gdf: gpd.GeoDataFrame, export_format: str) -> tuple[bytes, str, str]:
    normalized = export_format.lower()
    if normalized == "geojson":
        return gdf.to_json().encode("utf-8"), "velora_lts_network.geojson", "application/geo+json"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "velora_lts_network.gpkg"
        gdf.to_file(path, layer="lts_network", driver="GPKG")
        return path.read_bytes(), "velora_lts_network.gpkg", "application/geopackage+sqlite3"
