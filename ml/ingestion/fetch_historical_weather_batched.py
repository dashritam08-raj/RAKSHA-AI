from __future__ import print_function

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

API_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]


def load_locations(path):
    df = pd.read_csv(path, low_memory=False)
    required = ["location_id", "district_en", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Missing gazetteer columns: {}".format(missing))

    rows = []
    for _, r in df.iterrows():
        try:
            lat = float(r["latitude"])
            lon = float(r["longitude"])
        except Exception:
            continue
        rows.append({
            "location_id": str(r["location_id"]),
            "label": str(r["district_en"]),
            "latitude": lat,
            "longitude": lon,
        })
    return rows


def fetch(url, attempts=7):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "RAKSHA-AI-historical-weather/1.0",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = max(5.0, float(retry_after))
                except Exception:
                    delay = min(180.0, 15.0 * attempt)
            else:
                delay = min(90.0, 5.0 * attempt)
            print(
                "      attempt {} failed with HTTP {}; waiting {:.1f}s".format(
                    attempt, exc.code, delay
                )
            )
            time.sleep(delay)

        except Exception as exc:
            last_error = exc
            delay = min(90.0, 5.0 * attempt)
            print(
                "      attempt {} failed: {}; waiting {:.1f}s".format(
                    attempt, exc, delay
                )
            )
            time.sleep(delay)

    raise RuntimeError(
        "Request failed after {} attempts: {}".format(attempts, last_error)
    )


def build_url(batch, start_date, end_date):
    params = [
        ("latitude", ",".join("{:.6f}".format(x["latitude"]) for x in batch)),
        ("longitude", ",".join("{:.6f}".format(x["longitude"]) for x in batch)),
        ("start_date", start_date),
        ("end_date", end_date),
        ("daily", ",".join(DAILY_VARS)),
        ("timezone", "Asia/Kathmandu"),
        ("temperature_unit", "celsius"),
        ("wind_speed_unit", "kmh"),
        ("precipitation_unit", "mm"),
    ]
    return API_URL + "?" + urllib.parse.urlencode(params)


def normalize(payload, batch):
    items = [payload] if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise RuntimeError("Unexpected Open-Meteo response type.")

    if len(items) != len(batch):
        raise RuntimeError(
            "Expected {} location responses, received {}.".format(
                len(batch), len(items)
            )
        )

    rows = []
    for i, item in enumerate(items):
        daily = item.get("daily") or {}
        dates = daily.get("time") or []

        for j, date_value in enumerate(dates):
            values = {}
            for var in DAILY_VARS:
                arr = daily.get(var) or []
                values[var] = arr[j] if j < len(arr) else None

            rows.append({
                "location_id": batch[i]["location_id"],
                "label": batch[i]["label"],
                "latitude": batch[i]["latitude"],
                "longitude": batch[i]["longitude"],
                "observed_at": "{}T00:00:00".format(date_value),
                "source": "Open-Meteo Historical Weather Archive",
                "source_url": API_URL,
                # Names intentionally match the existing RAKSHA AI training pipeline.
                "temperature_mean_c": values["temperature_2m_mean"],
                "temperature_max_c": values["temperature_2m_max"],
                "temperature_min_c": values["temperature_2m_min"],
                "precipitation_mm": values["precipitation_sum"],
                "wind_max_kmh": values["wind_speed_10m_max"],
                "wind_gust_max_kmh": values["wind_gusts_10m_max"],
            })

    return pd.DataFrame(rows)


def cache_is_complete(path, expected_ids):
    if not os.path.exists(path):
        return None
    try:
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty or "location_id" not in frame.columns:
            return None
        ids = set(frame["location_id"].astype(str))
        if set(expected_ids).issubset(ids):
            return frame
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--locations",
        default="data/mappings/nepal_district_locations.csv"
    )
    ap.add_argument("--start-date", default="1971-01-01")
    ap.add_argument("--end-date", default="2013-12-31")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pause-seconds", type=float, default=5.0)
    ap.add_argument(
        "--output",
        default="data/raw/historical_weather_observations_complete.csv"
    )
    args = ap.parse_args()

    locations = load_locations(args.locations)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    batch_dir = os.path.join("data", "raw", "weather_batches")
    os.makedirs(batch_dir, exist_ok=True)

    print("RAKSHA AI — batched historical weather recovery")
    print("Period: {} to {}".format(args.start_date, args.end_date))
    print("Gazetteer locations:", len(locations))
    print("Batch size:", args.batch_size)
    print("No synthetic values are generated.")

    batches = [
        locations[i:i + args.batch_size]
        for i in range(0, len(locations), args.batch_size)
    ]

    manifest = {
        "api": API_URL,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "daily_variables": DAILY_VARS,
        "batch_size": args.batch_size,
        "location_count": len(locations),
        "batches": [],
        "failures": [],
    }

    frames = []

    for number, batch in enumerate(batches, 1):
        path = os.path.join(
            batch_dir, "weather_batch_{:02d}.csv".format(number)
        )
        expected_ids = [x["location_id"] for x in batch]

        cached = cache_is_complete(path, expected_ids)
        if cached is not None:
            print(
                "[batch {}/{}] cached: {} rows".format(
                    number, len(batches), len(cached)
                )
            )
            frames.append(cached)
            manifest["batches"].append({
                "batch": number,
                "status": "cached",
                "location_ids": expected_ids,
                "rows": int(len(cached)),
                "output": path,
            })
            continue

        print(
            "[batch {}/{}] fetching {} locations...".format(
                number, len(batches), len(batch)
            )
        )

        try:
            url = build_url(batch, args.start_date, args.end_date)
            payload = json.loads(fetch(url))
            frame = normalize(payload, batch)

            if frame.empty:
                raise RuntimeError("Returned zero weather rows.")

            frame.to_csv(path, index=False)
            frames.append(frame)

            manifest["batches"].append({
                "batch": number,
                "status": "downloaded",
                "location_ids": expected_ids,
                "rows": int(len(frame)),
                "output": path,
            })
            print("      downloaded rows:", len(frame))

        except Exception as exc:
            print("      FAILED:", exc)
            manifest["failures"].append({
                "batch": number,
                "location_ids": expected_ids,
                "error": str(exc),
            })

        if number < len(batches):
            time.sleep(max(0.0, args.pause_seconds))

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged["observed_at"] = pd.to_datetime(
            merged["observed_at"], errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S")
        merged = merged.drop_duplicates(
            subset=["location_id", "observed_at"]
        ).sort_values(["location_id", "observed_at"])
        merged.to_csv(args.output, index=False)

    successful = set()
    for item in manifest["batches"]:
        if item["status"] in ("cached", "downloaded"):
            successful.update(item["location_ids"])

    manifest["successful_locations"] = len(successful)
    manifest["failed_batches"] = len(manifest["failures"])
    manifest["final_rows"] = int(len(pd.concat(frames, ignore_index=True))) if frames else 0

    manifest_path = os.path.join(
        "data", "raw", "historical_weather_batch_manifest.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDONE")
    print(
        "Successful locations: {} / {}".format(
            manifest["successful_locations"], len(locations)
        )
    )
    print("Failed batches:", manifest["failed_batches"])
    print("Rows:", manifest["final_rows"])
    print("Weather output:", args.output)
    print("Manifest:", manifest_path)

    if manifest["failed_batches"]:
        print(
            "\nSome batches failed. Run the SAME command again; "
            "successful batches will be reused from cache."
        )


if __name__ == "__main__":
    main()
