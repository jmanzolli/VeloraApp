from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LTSResult:
    lts: int
    lts_raw: float
    lts_winter: int | None
    lts_contraflow: int | None
    confidence: float
    missing_inputs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "lts": self.lts,
            "lts_raw": self.lts_raw,
            "lts_winter": self.lts_winter,
            "lts_contraflow": self.lts_contraflow,
            "confidence": self.confidence,
            "missing_inputs": ";".join(self.missing_inputs),
        }


class LTSCalculator:
    """Pure rule-based bicycle Level of Traffic Stress calculator."""

    FREEWAY_CLASSES = {"motorway", "motorway_link", "freeway", "freeway_ramp"}

    def score_dataframe(self, segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        scored = segments.copy()
        result_rows = [self.calculate(row).as_dict() for _, row in scored.iterrows()]
        result_df = pd.DataFrame(result_rows, index=scored.index)
        for column in result_df.columns:
            scored[column] = result_df[column]
        return scored

    def calculate(self, row: pd.Series | dict[str, Any]) -> LTSResult:
        values = dict(row)
        missing_inputs = self._list_field(values.get("missing_inputs"))
        confidence = self._confidence(values.get("confidence"), missing_inputs)

        road_class = str(values.get("road_class") or "").lower()
        bicycle_access = str(values.get("bicycle_access") or "").lower()
        if road_class in self.FREEWAY_CLASSES or bicycle_access in {"no", "private"}:
            return LTSResult(
                lts=4,
                lts_raw=6,
                lts_winter=4,
                lts_contraflow=None,
                confidence=max(0.2, confidence),
                missing_inputs=missing_inputs,
            )

        facility = str(values.get("bike_facility_type") or "none").lower()
        if facility in {"path", "protected_track", "cycle_track"}:
            lts_raw = self._path_lts(values)
            lts_winter = int(lts_raw) if self._bool(values.get("winter_maintained"), default=True) else 4
            return LTSResult(
                lts=int(np.clip(lts_raw, 1, 4)),
                lts_raw=float(lts_raw),
                lts_winter=int(np.clip(lts_winter, 1, 4)),
                lts_contraflow=int(np.clip(lts_raw, 1, 4)) if self._bool(values.get("contraflow"), default=False) else None,
                confidence=confidence,
                missing_inputs=missing_inputs,
            )

        if facility in {"bike_lane", "shared_bus_bike_lane"}:
            lts_raw = self._bike_lane_lts(values)
        else:
            lts_raw = self._mixed_traffic_lts(values)

        lts_winter = lts_raw
        if facility in {"bike_lane", "protected_track", "cycle_track", "path"} and not self._bool(
            values.get("winter_maintained"),
            default=True,
        ):
            lts_winter = self._mixed_traffic_lts(values) if facility == "bike_lane" else 4

        lts_contraflow = None
        if self._bool(values.get("contraflow"), default=False):
            contra_width = self._number(values.get("contraflow_lane_width_m"), default=0.0)
            if contra_width > 0:
                contra_values = {**values, "bike_lane_width_m": contra_width, "effective_reach_m": max(contra_width, 1.5)}
                lts_contraflow = self._bike_lane_lts(contra_values)
            else:
                contra_values = {**values, "one_way": False, "centerline": False}
                lts_contraflow = self._mixed_traffic_lts(contra_values)

        return LTSResult(
            lts=int(np.clip(round(lts_raw), 1, 4)),
            lts_raw=float(lts_raw),
            lts_winter=int(np.clip(round(lts_winter), 1, 4)),
            lts_contraflow=int(np.clip(round(lts_contraflow), 1, 4)) if lts_contraflow is not None else None,
            confidence=confidence,
            missing_inputs=missing_inputs,
        )

    def _mixed_traffic_lts(self, values: dict[str, Any]) -> int:
        speed = self._number(values.get("speed_q85_kmh"), default=50.0)
        adt = self._number(values.get("adt"), default=3000.0)
        if self._bool(values.get("divided"), default=False):
            adt *= 2
        lanes = max(1, int(round(self._number(values.get("lanes_per_direction"), default=1.0))))
        one_way = self._bool(values.get("one_way"), default=False)
        centerline = self._bool(values.get("centerline"), default=lanes > 1)

        if not one_way and not centerline:
            if adt <= 750:
                return 1 if speed <= 46 else 2 if speed <= 62 else 3
            if adt <= 1500:
                return 1 if speed <= 46 else 2 if speed <= 54 else 3 if speed <= 70 else 4
            if adt <= 3000:
                return 2 if speed <= 54 else 3 if speed <= 62 else 4
            return 2 if speed <= 46 else 3 if speed <= 62 else 4

        if lanes == 1:
            if adt <= 1000:
                return 1 if speed <= 46 else 2 if speed <= 62 else 3
            if adt <= 1500:
                return 2 if speed <= 54 else 3 if speed <= 70 else 4
            return 2 if speed <= 38 else 3 if speed <= 62 else 4

        if lanes == 2:
            if adt <= 8000:
                return 3 if speed <= 62 else 4
            return 3 if speed <= 46 else 4

        return 3 if speed <= 46 else 4

    def _bike_lane_lts(self, values: dict[str, Any]) -> int:
        speed = self._number(values.get("speed_q85_kmh"), default=50.0)
        lanes = max(1, int(round(self._number(values.get("lanes_per_direction"), default=1.0))))
        one_way = self._bool(values.get("one_way"), default=False)
        centerline = self._bool(values.get("centerline"), default=lanes > 1)
        lane_width = self._number(values.get("bike_lane_width_m"), default=1.5)
        reach = self._number(values.get("effective_reach_m"), default=lane_width)
        parking_width = self._number(values.get("parking_width_m"), default=0.0)

        if lane_width <= 0:
            return self._mixed_traffic_lts(values)

        if parking_width > 0:
            if lanes == 1:
                if reach >= 4.45:
                    return 1 if speed <= 46 else 2 if speed <= 62 else 3
                if reach >= 3.5:
                    return 2 if speed <= 54 else 3
                return 3
            if lanes == 2 and not one_way:
                return 2 if reach >= 4.45 and speed <= 46 else 3
            return 3 if speed <= 70 else 4

        if lanes == 1 or not centerline:
            if reach >= 1.8:
                return 1 if speed <= 54 else 2 if speed <= 62 else 3
            return 2 if speed <= 62 else 3 if speed <= 78 else 4

        if lanes == 2:
            if reach >= 1.8:
                return 2 if speed <= 62 else 3
            return 2 if speed <= 62 else 3 if speed <= 70 else 4

        return 3 if speed <= 62 else 4

    def _path_lts(self, values: dict[str, Any]) -> int:
        driveway_visible = self._bool(values.get("path_driveway_visible"), default=False)
        return 2 if driveway_visible else 1

    @staticmethod
    def _number(value: Any, default: float) -> float:
        if isinstance(value, (list, tuple, set)):
            value = next(iter(value), None)
        try:
            if pd.isna(value):
                return default
        except ValueError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, (list, tuple, set)):
            value = next(iter(value), None)
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except ValueError:
            pass
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"yes", "true", "1", "y", "oui", "divided", "separate"}:
            return True
        if normalized in {"no", "false", "0", "n", "non", "not", "none"}:
            return False
        return default

    @staticmethod
    def _list_field(value: Any) -> list[str]:
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        if isinstance(value, tuple | set):
            return [str(item) for item in value if str(item)]
        if isinstance(value, str):
            if not value.strip():
                return []
            return [item.strip() for item in value.split(";") if item.strip()]
        return [str(value)]

    @staticmethod
    def _confidence(value: Any, missing_inputs: list[str]) -> float:
        try:
            base = float(value)
            if not np.isnan(base):
                return float(np.clip(base, 0.0, 1.0))
        except (TypeError, ValueError):
            pass
        return float(np.clip(1.0 - 0.08 * len(missing_inputs), 0.35, 1.0))
