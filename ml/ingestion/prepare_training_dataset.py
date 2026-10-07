"""Join real environmental observations with labeled incident outcomes.

This step requires a domain-reviewed target_risk_score in the incident labels.
It never creates a target from environmental inputs.

Usage:
    python -m ingestion.prepare_training_dataset \
        --incidents data/incident_labels_template.csv \
        --weather data/raw/weather_observations.csv \
        --earthquakes data/raw/earthquake_events.csv \
        --output data/processed/real_training_ready.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, a)))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a real-data RAKSHA training table")
    parser.add_argument("--incidents", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--earthquakes", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-weather-gap-hours", type=float, default=6.0)
    parser.add_argument("--earthquake-window-hours", type=float, default=24.0)
    parser.add_argument("--earthquake-radius-km", type=float, default=300.0)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: List[str], name: str):
    missing = set(columns) - set(df.columns)
    if missing:
        raise SystemExit(f"{name} is missing: " + ", ".join(sorted(missing)))


def main():
    args = parse_args()
    incidents = pd.read_csv(args.incidents)
    weather = pd.read_csv(args.weather)

    require_columns(
        incidents,
        ["incident_id", "location_id", "observed_at", "latitude", "longitude", "disaster_type", "severity", "people_affected", "target_risk_score"],
        "Incident labels CSV",
    )
    require_columns(
        weather,
        ["location_id", "observed_at", "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"],
        "Weather observations CSV",
    )

    if incidents.empty:
        raise SystemExit("No incident labels supplied.")
    if not set(pd.to_numeric(incidents["target_risk_score"], errors="coerce").dropna().unique()).issubset(set(range(0, 101))):
        raise SystemExit("target_risk_score must be numeric in [0, 100].")

    incidents = incidents.copy()
    weather = weather.copy()
    incidents["observed_at"] = pd.to_datetime(incidents["observed_at"], errors="coerce", utc=True)
    weather["observed_at"] = pd.to_datetime(weather["observed_at"], errors="coerce", utc=True)
    incidents = incidents.dropna(subset=["observed_at", "location_id", "target_risk_score"])
    weather = weather.dropna(subset=["observed_at", "location_id"])

    incidents = incidents.sort_values(["location_id", "observed_at"])
    weather = weather.sort_values(["location_id", "observed_at"])

    # Nearest prior observation only: do not use future environmental information.
    merged = pd.merge_asof(
        incidents,
        weather,
        on="observed_at",
        by="location_id",
        direction="backward",
        tolerance=pd.Timedelta(hours=args.max_weather_gap_hours),
        suffixes=("", "_weather"),
    )

    merged["rainfall_1h_mm"] = pd.to_numeric(merged.get("precipitation"), errors="coerce")
    merged["temperature_c"] = pd.to_numeric(merged.get("temperature_2m"), errors="coerce")
    merged["humidity_pct"] = pd.to_numeric(merged.get("relative_humidity_2m"), errors="coerce")
    merged["wind_speed_kmh"] = pd.to_numeric(merged.get("wind_speed_10m"), errors="coerce")
    merged["wind_gust_kmh"] = pd.to_numeric(merged.get("wind_gusts_10m"), errors="coerce")
    merged["rainfall_24h_mm"] = np.nan
    merged["rainfall_probability_pct"] = np.nan

    if args.earthquakes:
        quakes = pd.read_csv(args.earthquakes)
        require_columns(quakes, ["location_id", "event_time", "magnitude", "distance_km"], "Earthquake CSV")
        quakes["event_time"] = pd.to_datetime(quakes["event_time"], errors="coerce", utc=True)
        quakes["magnitude"] = pd.to_numeric(quakes["magnitude"], errors="coerce")
        quakes["distance_km"] = pd.to_numeric(quakes["distance_km"], errors="coerce")
        quakes = quakes.dropna(subset=["event_time", "location_id"])

        max_magnitudes = []
        counts = []
        for row in merged.itertuples(index=False):
            start = row.observed_at - pd.Timedelta(hours=args.earthquake_window_hours)
            mask = (
                (quakes["location_id"] == row.location_id)
                & (quakes["event_time"] <= row.observed_at)
                & (quakes["event_time"] >= start)
                & (quakes["distance_km"] <= args.earthquake_radius_km)
            )
            subset = quakes.loc[mask]
            magnitudes = pd.to_numeric(subset["magnitude"], errors="coerce").dropna()
            max_magnitudes.append(float(magnitudes.max()) if not magnitudes.empty else 0.0)
            counts.append(float(len(subset)))
        merged["earthquake_magnitude"] = max_magnitudes
        merged["nearby_earthquakes_24h"] = counts
    else:
        merged["earthquake_magnitude"] = 0.0
        merged["nearby_earthquakes_24h"] = 0.0

    # Features not observable from the two feeds remain explicitly missing and are
    # expected to be supplied later from authoritative operational sources.
    for column in [
        "water_level_m", "water_level_change_m", "population_density_km2", "elevation_m",
        "distance_to_hospital_km", "road_accessibility_score", "nearby_incidents_24h",
        "resource_pressure_pct", "official_alert_level", "fused_risk_signal", "data_coverage_pct",
    ]:
        if column not in merged.columns:
            merged[column] = np.nan

    severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    merged["severity_encoded"] = merged["severity"].astype(str).str.strip().str.lower().map(severity_map).fillna(2).astype(int)
    merged["log_people_affected"] = np.log1p(pd.to_numeric(merged["people_affected"], errors="coerce").clip(lower=0))
    merged["dataset_type"] = "real_pending_review"

    keep = [
        "incident_id", "observed_at", "dataset_type", "location_id", "latitude", "longitude", "disaster_type",
        "severity_encoded", "people_affected", "rainfall_1h_mm", "rainfall_24h_mm", "rainfall_probability_pct",
        "wind_speed_kmh", "wind_gust_kmh", "temperature_c", "humidity_pct", "water_level_m", "water_level_change_m",
        "population_density_km2", "elevation_m", "distance_to_hospital_km", "road_accessibility_score",
        "nearby_incidents_24h", "earthquake_magnitude", "resource_pressure_pct", "official_alert_level",
        "fused_risk_signal", "data_coverage_pct", "target_risk_score",
    ]
    for column in keep:
        if column not in merged.columns:
            merged[column] = np.nan
    output = merged[keep].copy()
    output["observed_at"] = output["observed_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    coverage = 1.0 - output.isna().mean()
    report = {
        "rows": int(len(output)),
        "target_rows": int(output["target_risk_score"].notna().sum()),
        "weather_joined_rows": int(output["rainfall_1h_mm"].notna().sum()),
        "feature_coverage_pct": {k: round(float(v * 100), 2) for k, v in coverage.to_dict().items()},
        "dataset_type": "real_pending_review",
        "warning": "This table still requires review of labels, provenance, missing features, geography and target definition before model training.",
    }
    report_path = output_path.with_suffix(".quality.json")
    report_path.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    print(__import__("json").dumps({"output": str(output_path), "quality_report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
