"""Fetch real historical weather and earthquake data for RAKSHA training/analysis.

Usage:
    python -m ingestion.fetch_environment \
        --locations data/locations_template.csv \
        --start-date 2024-01-01 \
        --end-date 2024-12-31 \
        --output-dir data/raw

This collector never fabricates observations. It records provenance and fails
clearly when a source is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode

import pandas as pd

from .http_utils import fetch_json

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch real historical weather and earthquakes for RAKSHA")
    parser.add_argument("--locations", required=True, help="CSV with location_id,latitude,longitude")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--earthquake-radius-km", type=int, default=300, help="USGS search radius around each location")
    parser.add_argument("--output-dir", default="data/raw")
    return parser.parse_args()


def read_locations(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"location_id", "latitude", "longitude"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit("Locations CSV is missing: " + ", ".join(sorted(missing)))
    output = []
    for row in rows:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except ValueError as exc:
            raise SystemExit(f"Invalid coordinates for {row.get('location_id')}: {exc}")
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise SystemExit(f"Coordinates out of range for {row.get('location_id')}")
        output.append({
            "location_id": row["location_id"].strip(),
            "latitude": lat,
            "longitude": lon,
            "label": row.get("label", "").strip(),
            "region": row.get("region", "").strip(),
        })
    if not output:
        raise SystemExit("No locations found in the input CSV.")
    return output


def fetch_weather(location: Dict, start_date: str, end_date: str) -> List[Dict]:
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
    }
    url = OPEN_METEO_URL + "?" + urlencode(params)
    payload, headers = fetch_json(url)
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    rows: List[Dict] = []
    for i, timestamp in enumerate(times):
        row = {
            "location_id": location["location_id"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "region": location["region"],
            "label": location["label"],
            "observed_at": timestamp,
            "source": "Open-Meteo Historical Weather / ERA5-family reanalysis",
            "source_url": OPEN_METEO_URL,
        }
        for variable in WEATHER_VARIABLES:
            values = hourly.get(variable) or []
            row[variable] = values[i] if i < len(values) else None
        rows.append(row)
    return rows


def haversine_km(lat1, lon1, lat2, lon2):
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, a)))


def fetch_earthquakes(location: Dict, start_date: str, end_date: str, radius_km: int) -> List[Dict]:
    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "maxradiuskm": radius_km,
        "orderby": "time",
        "limit": 20000,
    }
    url = USGS_URL + "?" + urlencode(params)
    payload, headers = fetch_json(url)
    rows: List[Dict] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or [None, None, None]
        event_lon, event_lat, depth = (coords + [None, None, None])[:3]
        event_time_ms = properties.get("time")
        if event_time_ms is None:
            continue
        event_dt = datetime.fromtimestamp(float(event_time_ms) / 1000.0, tz=timezone.utc).isoformat()
        distance = None
        if event_lat is not None and event_lon is not None:
            distance = haversine_km(location["latitude"], location["longitude"], float(event_lat), float(event_lon))
        rows.append({
            "location_id": location["location_id"],
            "monitor_latitude": location["latitude"],
            "monitor_longitude": location["longitude"],
            "event_id": feature.get("id"),
            "event_time": event_dt,
            "magnitude": properties.get("mag"),
            "place": properties.get("place"),
            "alert": properties.get("alert"),
            "status": properties.get("status"),
            "latitude": event_lat,
            "longitude": event_lon,
            "depth_km": depth,
            "distance_km": round(distance, 3) if distance is not None else None,
            "source": "USGS Earthquake Catalog",
            "source_url": USGS_URL,
        })
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    locations = read_locations(Path(args.locations))

    weather_rows: List[Dict] = []
    earthquake_rows: List[Dict] = []
    failures = []

    for location in locations:
        try:
            weather_rows.extend(fetch_weather(location, args.start_date, args.end_date))
        except Exception as exc:
            failures.append({"source": "Open-Meteo", "location_id": location["location_id"], "error": str(exc)})
        try:
            earthquake_rows.extend(fetch_earthquakes(location, args.start_date, args.end_date, args.earthquake_radius_km))
        except Exception as exc:
            failures.append({"source": "USGS", "location_id": location["location_id"], "error": str(exc)})

    weather_path = output_dir / "weather_observations.csv"
    quakes_path = output_dir / "earthquake_events.csv"
    manifest_path = output_dir / "ingestion_manifest.json"

    pd.DataFrame(weather_rows).to_csv(weather_path, index=False)
    pd.DataFrame(earthquake_rows).drop_duplicates(subset=["location_id", "event_id"]).to_csv(quakes_path, index=False)

    manifest = {
        "ingestion_started_at": datetime.now(timezone.utc).isoformat(),
        "locations_file": str(Path(args.locations)),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "sources": [
            {"name": "Open-Meteo Historical Weather", "endpoint": OPEN_METEO_URL, "rows": len(weather_rows)},
            {"name": "USGS Earthquake Catalog", "endpoint": USGS_URL, "rows": len(earthquake_rows)},
        ],
        "failures": failures,
        "weather_output": str(weather_path),
        "earthquake_output": str(quakes_path),
        "source_policy": "Real observations only; no synthetic values generated by this collector.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "weather_rows": len(weather_rows),
        "earthquake_rows": len(earthquake_rows),
        "failures": failures,
        "weather_output": str(weather_path),
        "earthquake_output": str(quakes_path),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
