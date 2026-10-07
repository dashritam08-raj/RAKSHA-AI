import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]


def request_json(url, params, retries=4, timeout=90, user_agent="RAKSHA-AI/5.2 historical-environment-backfill"):
    query = urlencode(params, doseq=True)
    full = url + "?" + query
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(full, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8")
            return json.loads(payload), full
        except HTTPError as exc:
            last_error = "HTTP {}: {}".format(exc.code, exc.reason)
            if exc.code == 429 or 500 <= exc.code <= 599:
                time.sleep(min(8.0, 1.5 * attempt))
            else:
                break
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = repr(exc)
            time.sleep(min(8.0, 1.5 * attempt))
    raise RuntimeError("Request failed after {} attempts: {}".format(retries, last_error))


def read_gazetteer(path):
    df = pd.read_csv(path, low_memory=False)
    required = ["location_id", "district_en", "province_en", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Gazetteer is missing columns: {}".format(missing))
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    # Keep one coordinate per district ID.
    df = df.drop_duplicates(subset=["location_id"], keep="first")
    return df


def fetch_weather(gaz, start_date, end_date, output, delay):
    rows = []
    failures = []
    total = len(gaz)
    cache_dir = os.path.splitext(output)[0] + "_cache"
    os.makedirs(cache_dir, exist_ok=True)

    for idx, r in enumerate(gaz.itertuples(index=False), 1):
        location_id = str(r.location_id)
        district = str(r.district_en)
        cache_path = os.path.join(cache_dir, location_id + ".json")
        print("[weather {}/{}] {} ({})".format(idx, total, district, location_id))
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                source_url = payload.get("_source_url", "")
            else:
                params = {
                    "latitude": "{:.6f}".format(float(r.latitude)),
                    "longitude": "{:.6f}".format(float(r.longitude)),
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": ",".join(DAILY_VARS),
                    "timezone": "Asia/Kathmandu",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "precipitation_unit": "mm",
                    "cell_selection": "land",
                }
                payload, source_url = request_json(OPEN_METEO_URL, params)
                payload["_source_url"] = source_url
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                time.sleep(delay)

            daily = payload.get("daily") or {}
            dates = daily.get("time") or []
            for i, date_str in enumerate(dates):
                row = {
                    "location_id": location_id,
                    "latitude": float(r.latitude),
                    "longitude": float(r.longitude),
                    "region": str(r.province_en),
                    "label": district,
                    "observed_at": date_str + "T00:00:00",
                    "source": "Open-Meteo Historical Weather API",
                    "source_url": source_url,
                }
                for var in DAILY_VARS:
                    vals = daily.get(var) or []
                    row[var] = vals[i] if i < len(vals) else None
                rows.append(row)
        except Exception as exc:
            failures.append({"location_id": location_id, "district": district, "error": str(exc)})
            print("      FAILED:", exc)

    fieldnames = [
        "location_id", "latitude", "longitude", "region", "label", "observed_at",
        "source", "source_url"
    ] + DAILY_VARS
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return {"rows": len(rows), "locations": total, "failures": failures, "cache_dir": cache_dir}


def fetch_earthquakes(start_year, end_year, output, min_magnitude, min_lat, max_lat, min_lon, max_lon, delay):
    rows = {}
    failures = []
    years = list(range(start_year, end_year + 1))
    for idx, year in enumerate(years, 1):
        start = "{}-01-01T00:00:00".format(year)
        end = "{}-01-01T00:00:00".format(year + 1) if year < end_year else "{}-12-31T23:59:59".format(end_year)
        print("[earthquake {}/{}] {}".format(idx, len(years), year))
        params = {
            "format": "geojson",
            "starttime": start,
            "endtime": end,
            "minmagnitude": min_magnitude,
            "minlatitude": min_lat,
            "maxlatitude": max_lat,
            "minlongitude": min_lon,
            "maxlongitude": max_lon,
            "orderby": "time-asc",
            "limit": 20000,
        }
        try:
            payload, source_url = request_json(USGS_URL, params, timeout=120)
            features = payload.get("features") or []
            for feature in features:
                props = feature.get("properties") or {}
                geom = feature.get("geometry") or {}
                coords = geom.get("coordinates") or []
                if len(coords) < 2:
                    continue
                event_id = str(feature.get("id") or "")
                if not event_id:
                    continue
                event_ms = props.get("time")
                if event_ms is None:
                    continue
                event_dt = datetime.fromtimestamp(float(event_ms) / 1000.0, tz=timezone.utc)
                rows[event_id] = {
                    "event_id": event_id,
                    "event_time": event_dt.isoformat().replace("+00:00", "Z"),
                    "magnitude": props.get("mag"),
                    "place": props.get("place"),
                    "alert": props.get("alert"),
                    "status": props.get("status"),
                    "latitude": coords[1],
                    "longitude": coords[0],
                    "depth_km": coords[2] if len(coords) > 2 else None,
                    "source": "USGS Earthquake Catalog",
                    "source_url": source_url,
                }
            time.sleep(delay)
        except Exception as exc:
            failures.append({"year": year, "error": str(exc)})
            print("      FAILED:", exc)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fieldnames = [
        "event_id", "event_time", "magnitude", "place", "alert", "status",
        "latitude", "longitude", "depth_km", "source", "source_url"
    ]
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda x: x["event_time"]))
    return {"rows": len(rows), "years": len(years), "failures": failures}


def main():
    ap = argparse.ArgumentParser(description="Backfill real historical weather and earthquake data for RAKSHA AI ML development.")
    ap.add_argument("--gazetteer", default="data/mappings/nepal_district_locations.csv")
    ap.add_argument("--start-date", default="1971-01-01")
    ap.add_argument("--end-date", default="2013-12-31")
    ap.add_argument("--weather-output", default="data/raw/historical_weather_observations.csv")
    ap.add_argument("--earthquake-output", default="data/raw/historical_earthquake_events.csv")
    ap.add_argument("--earthquake-min-magnitude", type=float, default=3.0)
    ap.add_argument("--earthquake-min-latitude", type=float, default=24.5)
    ap.add_argument("--earthquake-max-latitude", type=float, default=33.0)
    ap.add_argument("--earthquake-min-longitude", type=float, default=78.0)
    ap.add_argument("--earthquake-max-longitude", type=float, default=91.5)
    ap.add_argument("--delay-seconds", type=float, default=0.5)
    ap.add_argument("--manifest", default="data/raw/historical_environment_manifest.json")
    args = ap.parse_args()

    print("RAKSHA AI — historical environmental backfill")
    print("Period:", args.start_date, "to", args.end_date)
    print("No synthetic values are generated by this collector.")

    gaz = read_gazetteer(args.gazetteer)
    print("Gazetteer locations:", len(gaz))

    weather = fetch_weather(gaz, args.start_date, args.end_date, args.weather_output, args.delay_seconds)
    earthquake = fetch_earthquakes(
        int(args.start_date[:4]),
        int(args.end_date[:4]),
        args.earthquake_output,
        args.earthquake_min_magnitude,
        args.earthquake_min_latitude,
        args.earthquake_max_latitude,
        args.earthquake_min_longitude,
        args.earthquake_max_longitude,
        args.delay_seconds,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"start_date": args.start_date, "end_date": args.end_date},
        "weather": weather,
        "earthquake": earthquake,
        "weather_source": "https://archive-api.open-meteo.com/v1/archive",
        "earthquake_source": "https://earthquake.usgs.gov/fdsnws/event/1/query",
        "earthquake_filter": {
            "min_magnitude": args.earthquake_min_magnitude,
            "min_latitude": args.earthquake_min_latitude,
            "max_latitude": args.earthquake_max_latitude,
            "min_longitude": args.earthquake_min_longitude,
            "max_longitude": args.earthquake_max_longitude,
        },
        "notes": [
            "Weather is daily historical data at the 77-district gazetteer coordinates and is intended for same-day feature matching.",
            "Earthquake data is fetched year-by-year to stay below the USGS per-query result limit.",
            "The earthquake feature stage later applies the exact 300 km / 7 day leakage-safe spatial-temporal filter relative to each incident.",
            "These are source observations/catalog records; no synthetic environmental values are produced by this collector."
        ]
    }
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDONE")
    print("Historical weather rows:", weather["rows"])
    print("Historical earthquake rows:", earthquake["rows"])
    print("Weather failures:", len(weather["failures"]))
    print("Earthquake failures:", len(earthquake["failures"]))
    print("Weather output:", args.weather_output)
    print("Earthquake output:", args.earthquake_output)
    print("Manifest:", args.manifest)


if __name__ == "__main__":
    main()
