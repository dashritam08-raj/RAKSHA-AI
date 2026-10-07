from __future__ import annotations

import math
from typing import Any, Dict, Mapping

SEVERITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

CORE_FEATURES = [
    "disaster_type",
    "severity_encoded",
    "people_affected",
    "log_people_affected",
]

ENVIRONMENTAL_FEATURES = [
    "rainfall_1h_mm",
    "rainfall_24h_mm",
    "rainfall_probability_pct",
    "wind_speed_kmh",
    "wind_gust_kmh",
    "temperature_c",
    "humidity_pct",
    "water_level_m",
    "water_level_change_m",
    "population_density_km2",
    "elevation_m",
    "distance_to_hospital_km",
    "road_accessibility_score",
    "nearby_incidents_24h",
    "earthquake_magnitude",
    "resource_pressure_pct",
    "official_alert_level",
    "fused_risk_signal",
    "data_coverage_pct",
]

ALL_FEATURES = CORE_FEATURES + ENVIRONMENTAL_FEATURES

DEFAULTS = {
    "rainfall_1h_mm": 0.0,
    "rainfall_24h_mm": 0.0,
    "rainfall_probability_pct": 0.0,
    "wind_speed_kmh": 0.0,
    "wind_gust_kmh": 0.0,
    "temperature_c": 25.0,
    "humidity_pct": 50.0,
    "water_level_m": 0.0,
    "water_level_change_m": 0.0,
    "population_density_km2": 1000.0,
    "elevation_m": 20.0,
    "distance_to_hospital_km": 5.0,
    "road_accessibility_score": 70.0,
    "nearby_incidents_24h": 0.0,
    "earthquake_magnitude": 0.0,
    "resource_pressure_pct": 0.0,
    "official_alert_level": 0.0,
    "fused_risk_signal": 0.0,
    "data_coverage_pct": 50.0,
}

BOUNDS = {
    "people_affected": (0, 1_000_000_000),
    "rainfall_1h_mm": (0, 1000),
    "rainfall_24h_mm": (0, 5000),
    "rainfall_probability_pct": (0, 100),
    "wind_speed_kmh": (0, 500),
    "wind_gust_kmh": (0, 700),
    "temperature_c": (-80, 70),
    "humidity_pct": (0, 100),
    "water_level_m": (-100, 1000),
    "water_level_change_m": (-100, 100),
    "population_density_km2": (0, 1_000_000),
    "elevation_m": (-500, 9000),
    "distance_to_hospital_km": (0, 500),
    "road_accessibility_score": (0, 100),
    "nearby_incidents_24h": (0, 10000),
    "earthquake_magnitude": (0, 10),
    "resource_pressure_pct": (0, 100),
    "official_alert_level": (0, 4),
    "fused_risk_signal": (0, 100),
    "data_coverage_pct": (0, 100),
}


def normalize_severity(value: str) -> str:
    v = str(value or "Medium").strip().capitalize()
    return v if v in SEVERITY_MAP else "Medium"


def _coerce_numeric(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def build_features(
    disaster_type: str,
    severity: str,
    people_affected: int,
    extra_features: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a stable feature dictionary for both legacy and enriched models."""
    people = max(0, min(int(people_affected or 0), 1_000_000_000))
    features: Dict[str, Any] = {
        "disaster_type": str(disaster_type or "Other").strip().lower() or "other",
        "severity_encoded": SEVERITY_MAP[normalize_severity(severity)],
        "people_affected": people,
        "log_people_affected": math.log1p(people),
    }
    extras = dict(extra_features or {})
    for name in ENVIRONMENTAL_FEATURES:
        value = extras.get(name, DEFAULTS[name])
        features[name] = _coerce_numeric(value, DEFAULTS[name])
    return features
