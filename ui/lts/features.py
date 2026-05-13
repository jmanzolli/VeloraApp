from __future__ import annotations

import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


class LTSFeatureExtractor:
    """Normalize OSM and municipal road attributes into Velora's LTS schema."""

    NORMALIZED_COLUMNS = [
        "segment_id",
        "source_ids",
        "source_priority",
        "road_class",
        "bicycle_access",
        "one_way",
        "divided",
        "lanes_per_direction",
        "speed_q85_kmh",
        "adt",
        "centerline",
        "bike_facility_type",
        "bike_facility_subtype",
        "bike_lane_width_m",
        "effective_reach_m",
        "parking_width_m",
        "contraflow",
        "contraflow_lane_width_m",
        "protected",
        "path_driveway_visible",
        "winter_maintained",
        "asymmetric",
        "imagery_lanes_per_direction",
        "imagery_parking_present",
        "imagery_protection_present",
        "imagery_confidence",
        "confidence",
        "missing_inputs",
        "inferred_fields",
    ]

    MUNICIPAL_FIELD_ALIASES = {
        "road_class": {"classe", "class", "road_class", "highway"},
        "one_way": {"sens_cir", "oneway", "one_way"},
        "divided": {"divided", "separe", "median"},
        "lanes_per_direction": {"nblane", "lanes_per_direction", "num_lanes", "lanes"},
        "speed_q85_kmh": {"q85", "spd_q85", "speed_q85", "v85", "speed_kmh", "maxspeed"},
        "adt": {"djma_2src", "djma", "adt", "aadt", "traffic_volume"},
        "centerline": {"cl_d_2", "centerline"},
        "bike_facility_type": {"type_voie", "bike_facility_type", "facility_type"},
        "bike_facility_subtype": {"type_voie2", "bike_facility_subtype", "facility_subtype"},
        "bike_lane_width_m": {"blanw", "bike_lane_width_m", "lane_width"},
        "effective_reach_m": {"reach", "effective_reach_m"},
        "parking_width_m": {"parkw", "parking_width_m"},
        "contraflow": {"contraflow", "contra"},
        "contraflow_lane_width_m": {"contralanw", "contraflow_lane_width_m"},
        "protected": {"protege", "protected"},
        "path_driveway_visible": {"drvewayvis", "driveway_visible"},
        "winter_maintained": {"protege_4s", "winter_maintained", "four_season"},
        "asymmetric": {"asymmetric", "asymetrique"},
    }

    SPEED_DEFAULTS = {
        "motorway": 90.0,
        "trunk": 80.0,
        "primary": 60.0,
        "secondary": 50.0,
        "tertiary": 45.0,
        "residential": 35.0,
        "living_street": 25.0,
        "service": 25.0,
        "cycleway": 20.0,
        "path": 20.0,
    }

    ADT_DEFAULTS = {
        "motorway": 25000.0,
        "trunk": 18000.0,
        "primary": 10000.0,
        "secondary": 6000.0,
        "tertiary": 3000.0,
        "residential": 1000.0,
        "living_street": 500.0,
        "service": 500.0,
        "cycleway": 0.0,
        "path": 0.0,
    }

    def normalize(self, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if edges.empty:
            return gpd.GeoDataFrame(columns=[*self.NORMALIZED_COLUMNS, "geometry"], geometry="geometry", crs=edges.crs)

        normalized_rows = [self._normalize_row(row, idx) for idx, row in edges.reset_index(drop=True).iterrows()]
        normalized = gpd.GeoDataFrame(normalized_rows, geometry=edges.reset_index(drop=True).geometry, crs=edges.crs)
        return normalized[[*self.NORMALIZED_COLUMNS, "geometry"]]

    def _normalize_row(self, row: pd.Series, index: int) -> dict[str, Any]:
        row_dict = row.to_dict()
        missing: list[str] = []
        inferred: list[str] = []

        road_class = self._municipal_or_osm(row_dict, "road_class", "highway")
        road_class = self._first(road_class) or "residential"
        if "road_class" not in row_dict and "highway" not in row_dict:
            inferred.append("road_class")

        bicycle_access = str(self._first(row_dict.get("bicycle")) or "").lower()
        source_ids = self._source_ids(row_dict)

        one_way = self._bool(self._municipal_or_osm(row_dict, "one_way", "oneway"), default=False)
        divided = self._bool(self._municipal_or_osm(row_dict, "divided", "dual_carriageway"), default=False)

        lanes_raw = self._municipal_or_osm(row_dict, "lanes_per_direction", "lanes")
        lanes_are_per_direction = any(
            self._present(self._row_get(row_dict, key))
            for key in ["municipal_lanes_per_direction", "lanes_per_direction", "nblane", "num_lanes"]
        )
        lanes_total = self._number(lanes_raw)
        if lanes_total is None:
            lanes_per_direction = self._default_lanes(road_class)
            missing.append("lanes_per_direction")
            inferred.append("lanes_per_direction")
        elif one_way or lanes_are_per_direction:
            lanes_per_direction = max(1.0, lanes_total)
        else:
            lanes_per_direction = max(1.0, round(lanes_total / 2.0))

        speed_raw = self._municipal_or_osm(row_dict, "speed_q85_kmh", "maxspeed")
        speed_q85 = self._speed(speed_raw)
        if speed_q85 is None:
            speed_q85 = self.SPEED_DEFAULTS.get(str(road_class), 40.0)
            missing.append("speed_q85_kmh")
            inferred.append("speed_q85_kmh")

        adt_raw = self._municipal_or_osm(row_dict, "adt", "adt")
        adt = self._number(adt_raw)
        if adt is None:
            adt = self.ADT_DEFAULTS.get(str(road_class), 2000.0)
            missing.append("adt")
            inferred.append("adt")

        centerline_raw = self._municipal_or_osm(row_dict, "centerline", "centerline")
        centerline = self._bool(centerline_raw, default=bool(lanes_per_direction > 1 or str(road_class) in {"primary", "secondary", "tertiary"}))
        if centerline_raw is None:
            inferred.append("centerline")

        facility_type, facility_subtype = self._facility(row_dict, road_class)
        bike_lane_width = self._number(self._municipal_or_osm(row_dict, "bike_lane_width_m", "cycleway:width"))
        if facility_type in {"bike_lane", "shared_bus_bike_lane"} and bike_lane_width is None:
            bike_lane_width = 1.5
            missing.append("bike_lane_width_m")
            inferred.append("bike_lane_width_m")
        bike_lane_width = bike_lane_width or 0.0

        parking_width = self._number(self._municipal_or_osm(row_dict, "parking_width_m", "parking_width_m"))
        parking_present = self._parking_present(row_dict)
        if parking_width is None:
            parking_width = 2.2 if parking_present else 0.0
            if parking_present:
                missing.append("parking_width_m")
                inferred.append("parking_width_m")

        effective_reach = self._number(self._municipal_or_osm(row_dict, "effective_reach_m", "effective_reach_m"))
        if effective_reach is None:
            effective_reach = bike_lane_width + parking_width if parking_width > 0 else bike_lane_width
            if bike_lane_width > 0:
                inferred.append("effective_reach_m")

        contraflow = self._contraflow(row_dict)
        contraflow_width = self._number(self._municipal_or_osm(row_dict, "contraflow_lane_width_m", "contraflow_lane_width_m"))
        if contraflow and contraflow_width is None:
            contraflow_width = bike_lane_width or 1.5
            missing.append("contraflow_lane_width_m")
            inferred.append("contraflow_lane_width_m")

        protected = self._bool(self._municipal_or_osm(row_dict, "protected", "protected"), default=facility_type in {"protected_track", "path"})
        winter_maintained = self._bool(self._municipal_or_osm(row_dict, "winter_maintained", "winter_maintained"), default=True)

        confidence = max(0.35, 1.0 - 0.08 * len(set(missing)))
        if self._has_municipal_override(row_dict):
            confidence = min(1.0, confidence + 0.08)

        return {
            "segment_id": str(row_dict.get("segment_id") or f"lts_{index:06d}"),
            "source_ids": source_ids,
            "source_priority": "municipal+osm" if self._has_municipal_override(row_dict) else "osm",
            "road_class": str(road_class),
            "bicycle_access": bicycle_access,
            "one_way": bool(one_way),
            "divided": bool(divided),
            "lanes_per_direction": float(lanes_per_direction),
            "speed_q85_kmh": float(speed_q85),
            "adt": float(adt),
            "centerline": bool(centerline),
            "bike_facility_type": facility_type,
            "bike_facility_subtype": facility_subtype,
            "bike_lane_width_m": float(bike_lane_width),
            "effective_reach_m": float(effective_reach),
            "parking_width_m": float(parking_width),
            "contraflow": bool(contraflow),
            "contraflow_lane_width_m": float(contraflow_width or 0.0),
            "protected": bool(protected),
            "path_driveway_visible": self._bool(self._municipal_or_osm(row_dict, "path_driveway_visible", "path_driveway_visible"), default=False),
            "winter_maintained": bool(winter_maintained),
            "asymmetric": self._bool(self._municipal_or_osm(row_dict, "asymmetric", "asymmetric"), default=False),
            "imagery_lanes_per_direction": np.nan,
            "imagery_parking_present": np.nan,
            "imagery_protection_present": np.nan,
            "imagery_confidence": np.nan,
            "confidence": confidence,
            "missing_inputs": ";".join(sorted(set(missing))),
            "inferred_fields": ";".join(sorted(set(inferred))),
        }

    def _municipal_or_osm(self, row: dict[str, Any], normalized_field: str, osm_field: str) -> Any:
        municipal_key = f"municipal_{normalized_field}"
        municipal_value = self._row_get(row, municipal_key)
        if self._present(municipal_value):
            return municipal_value
        for key, aliases in self.MUNICIPAL_FIELD_ALIASES.items():
            if key == normalized_field:
                for alias in aliases:
                    value = self._row_get(row, alias)
                    if self._present(value):
                        return value
        return self._row_get(row, osm_field)

    def _facility(self, row: dict[str, Any], road_class: str) -> tuple[str, str]:
        municipal = self._municipal_or_osm(row, "bike_facility_type", "bike_facility_type")
        if self._present(municipal):
            parsed = self._parse_municipal_facility(municipal)
            if parsed:
                return parsed

        cycleway_tags = [
            self._first(row.get("cycleway")),
            self._first(row.get("cycleway:left")),
            self._first(row.get("cycleway:right")),
            self._first(row.get("cycleway:both")),
        ]
        cycleway_text = " ".join(str(value).lower() for value in cycleway_tags if value)
        if str(road_class) in {"cycleway", "path"}:
            return "path", str(road_class)
        if any(token in cycleway_text for token in ["track", "separate"]):
            return "protected_track", cycleway_text
        if any(token in cycleway_text for token in ["lane", "opposite_lane"]):
            return "bike_lane", cycleway_text
        if any(token in cycleway_text for token in ["shared_lane", "share_busway"]):
            return "shared_bus_bike_lane", cycleway_text
        return "none", ""

    @staticmethod
    def _parse_municipal_facility(value: Any) -> tuple[str, str] | None:
        raw = str(value).strip().lower()
        code = LTSFeatureExtractor._number(value)
        if code in {3, 9} or "lane" in raw or "bande" in raw:
            return ("shared_bus_bike_lane" if code == 9 else "bike_lane", raw)
        if code == 4 or "protected" in raw or "piste" in raw:
            return "protected_track", raw
        if code in {5, 6, 7} or "path" in raw or "sentier" in raw:
            return "path", raw
        if code in {0, 1, 8}:
            return "none", raw
        return None

    def _parking_present(self, row: dict[str, Any]) -> bool:
        parking_keys = [key for key in row if str(key).startswith("parking")]
        for key in parking_keys:
            value = str(self._first(row.get(key)) or "").lower()
            if value in {"lane", "street_side", "yes", "parallel", "diagonal", "perpendicular"}:
                return True
        return self._number(self._municipal_or_osm(row, "parking_width_m", "parking_width_m")) not in {None, 0.0}

    def _contraflow(self, row: dict[str, Any]) -> bool:
        municipal = self._municipal_or_osm(row, "contraflow", "contraflow")
        if self._present(municipal):
            return self._bool(municipal, default=False)
        values = [
            self._first(row.get("cycleway")),
            self._first(row.get("cycleway:left")),
            self._first(row.get("cycleway:right")),
            self._first(row.get("oneway:bicycle")),
        ]
        text = " ".join(str(value).lower() for value in values if value)
        return "opposite" in text or "no" == str(self._first(row.get("oneway:bicycle"))).lower()

    @staticmethod
    def _source_ids(row: dict[str, Any]) -> str:
        for key in ["osmid", "osm_id", "id", "ID_TRC", "ID_CYCL"]:
            value = LTSFeatureExtractor._row_get(row, key)
            if LTSFeatureExtractor._present(value):
                return str(value)
        return ""

    @staticmethod
    def _row_get(row: dict[str, Any], key: str) -> Any:
        if key in row:
            return row[key]
        key_lower = key.lower()
        for existing_key, value in row.items():
            if str(existing_key).lower() == key_lower:
                return value
        return None

    @staticmethod
    def _default_lanes(road_class: str) -> float:
        return 2.0 if str(road_class) in {"motorway", "trunk", "primary"} else 1.0

    @staticmethod
    def _speed(value: Any) -> float | None:
        first = LTSFeatureExtractor._first(value)
        if first is None:
            return None
        if isinstance(first, (int, float)) and not pd.isna(first):
            return float(first)
        match = re.search(r"\d+(?:\.\d+)?", str(first))
        if not match:
            return None
        speed = float(match.group(0))
        if "mph" in str(first).lower():
            speed *= 1.60934
        return speed

    @staticmethod
    def _number(value: Any) -> float | None:
        first = LTSFeatureExtractor._first(value)
        if first is None:
            return None
        if isinstance(first, (int, float)) and not pd.isna(first):
            return float(first)
        match = re.search(r"-?\d+(?:\.\d+)?", str(first))
        return float(match.group(0)) if match else None

    @staticmethod
    def _bool(value: Any, default: bool = False) -> bool:
        first = LTSFeatureExtractor._first(value)
        if first is None:
            return default
        if isinstance(first, bool):
            return first
        if isinstance(first, (int, float)) and not pd.isna(first):
            return bool(first)
        text = str(first).strip().lower()
        if text in {"yes", "true", "1", "y", "oui", "divided", "separate", "protected", "o"}:
            return True
        if text in {"no", "false", "0", "n", "non", "not", "none"}:
            return False
        return default

    @staticmethod
    def _first(value: Any) -> Any:
        if isinstance(value, (list, tuple, set)):
            return next(iter(value), None)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    @staticmethod
    def _present(value: Any) -> bool:
        first = LTSFeatureExtractor._first(value)
        if first is None:
            return False
        return str(first).strip() != ""

    @staticmethod
    def _has_municipal_override(row: dict[str, Any]) -> bool:
        return any(str(key).startswith("municipal_") and LTSFeatureExtractor._present(value) for key, value in row.items())
