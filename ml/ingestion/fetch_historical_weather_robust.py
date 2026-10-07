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


def valid_frame(df):
    needed = ["location_id", "observed_at"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df.copy()
    df["location_id"] = df["location_id"].astype(str)
    df["observed_at"] = pd.to_datetime(df["observed_at"], errors="coerce")
    df = df[df["observed_at"].notna()].copy()
    return df


def import_existing(paths):
    frames = []
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            if os.path.isdir(path):
                files = [os.path.join(path, x) for x in os.listdir(path) if x.lower().endswith(".csv")]
            else:
                files = [path]
            for fp in files:
                try:
                    df = valid_frame(pd.read_csv(fp, low_memory=False))
                    if not df.empty:
                        frames.append(df)
                        print("Imported existing:", fp, "rows:", len(df))
                except Exception as exc:
                    print("Skipped existing file:", fp, "because:", exc)
        except Exception as exc:
            print("Skipped path:", path, "because:", exc)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["location_id", "observed_at"], keep="last")
    return merged


def build_url(location, start_date, end_date):
    params = [
        ("latitude", "{:.6f}".format(location["latitude"])),
        ("longitude", "{:.6f}".format(location["longitude"])),
        ("start_date", start_date),
        ("end_date", end_date),
        ("daily", ",".join(DAILY_VARS)),
        ("timezone", "Asia/Kathmandu"),
        ("temperature_unit", "celsius"),
        ("wind_speed_unit", "kmh"),
        ("precipitation_unit", "mm"),
        ("cell_selection", "land"),
    ]
    return API_URL + "?" + urllib.parse.urlencode(params)


def fetch_json(url, attempts, rate_wait):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "RAKSHA-AI-historical-weather/1.1",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    wait = float(retry_after)
                except Exception:
                    wait = rate_wait
                wait = max(wait, rate_wait)
                print("      HTTP 429 on attempt {}; waiting {:.0f}s...".format(attempt, wait))
                time.sleep(wait)
            else:
                wait = min(90.0, 10.0 * attempt)
                print("      HTTP {} on attempt {}; waiting {:.0f}s...".format(exc.code, attempt, wait))
                time.sleep(wait)
        except Exception as exc:
            last = exc
            wait = min(90.0, 10.0 * attempt)
            print("      request error on attempt {}: {}; waiting {:.0f}s...".format(attempt, exc, wait))
            time.sleep(wait)
    raise RuntimeError("Request failed after {} attempts: {}".format(attempts, last))


def parse_single(payload, location):
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected API response type: {}".format(type(payload)))
    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Open-Meteo returned an error"))

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        raise RuntimeError("Open-Meteo returned no daily dates.")

    rows = []
    for i, day in enumerate(dates):
        row = {
            "location_id": location["location_id"],
            "label": location["label"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "observed_at": "{}T00:00:00".format(day),
            "source": "Open-Meteo Historical Weather Archive",
            "source_url": API_URL,
        }
        for var in DAILY_VARS:
            values = daily.get(var) or []
            row[var] = values[i] if i < len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def cache_is_complete(path, location, start_date, end_date):
    if not os.path.exists(path):
        return False
    try:
        df = valid_frame(pd.read_csv(path, low_memory=False))
        if df.empty:
            return False
        expected_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
        return (
            df["location_id"].nunique() == 1 and
            str(df["location_id"].iloc[0]) == location["location_id"] and
            len(df) >= int(expected_days * 0.995)
        )
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", default="data/mappings/nepal_district_locations.csv")
    ap.add_argument("--start-date", default="1971-01-01")
    ap.add_argument("--end-date", default="2013-12-31")
    ap.add_argument("--pause-seconds", type=float, default=45.0)
    ap.add_argument("--rate-limit-wait-seconds", type=float, default=600.0)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--output", default="data/raw/historical_weather_observations_robust.csv")
    ap.add_argument("--manifest", default="data/raw/historical_weather_robust_manifest.json")
    args = ap.parse_args()

    locations = load_locations(args.locations)
    cache_dir = os.path.join("data", "raw", "weather_location_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("RAKSHA AI — robust historical weather recovery")
    print("Period: {} to {}".format(args.start_date, args.end_date))
    print("Gazetteer locations:", len(locations))
    print("Strategy: one district per request + checkpoint/resume")
    print("No synthetic values are generated.")

    existing_paths = [
        "data/raw/historical_weather_observations.csv",
        "data/raw/historical_weather_observations_complete.csv",
        "data/raw/weather_batches",
    ]
    existing = import_existing(existing_paths)
    if not existing.empty:
        existing.to_csv(args.output, index=False)
        print("Existing combined rows available:", len(existing))

    manifest = {
        "api": API_URL,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "locations": len(locations),
        "strategy": "one location per request, checkpoint/resume",
        "successful": [],
        "failed": [],
        "reused_existing": [],
    }

    all_frames = [existing] if not existing.empty else []

    for index, location in enumerate(locations, 1):
        lid = location["location_id"]
        cache_path = os.path.join(cache_dir, "weather_{}.csv".format(lid))

        if cache_is_complete(cache_path, location, args.start_date, args.end_date):
            frame = valid_frame(pd.read_csv(cache_path, low_memory=False))
            all_frames.append(frame)
            manifest["successful"].append(lid)
            print("[{} / {}] {} ({}) — cached".format(index, len(locations), location["label"], lid))
            continue

        if not existing.empty and lid in set(existing["location_id"].astype(str)):
            subset = existing[existing["location_id"].astype(str) == lid].copy()
            expected_days = (pd.Timestamp(args.end_date) - pd.Timestamp(args.start_date)).days + 1
            if len(subset) >= int(expected_days * 0.995):
                subset.to_csv(cache_path, index=False)
                all_frames.append(subset)
                manifest["reused_existing"].append(lid)
                manifest["successful"].append(lid)
                print("[{} / {}] {} ({}) — reused existing rows: {}".format(index, len(locations), location["label"], lid, len(subset)))
                continue

        print("[{} / {}] Fetching {} ({})".format(index, len(locations), location["label"], lid))
        try:
            payload = fetch_json(
                build_url(location, args.start_date, args.end_date),
                attempts=args.attempts,
                rate_wait=args.rate_limit_wait_seconds,
            )
            frame = parse_single(payload, location)
            frame.to_csv(cache_path, index=False)
            all_frames.append(frame)
            manifest["successful"].append(lid)
            print("      rows:", len(frame))
        except Exception as exc:
            print("      FAILED:", exc)
            manifest["failed"].append({"location_id": lid, "label": location["label"], "error": str(exc)})

        if index < len(locations):
            time.sleep(max(0.0, args.pause_seconds))

        # Persist progress after every district.
        try:
            if all_frames:
                combined = pd.concat(all_frames, ignore_index=True)
                combined["observed_at"] = pd.to_datetime(combined["observed_at"], errors="coerce")
                combined = combined[combined["observed_at"].notna()]
                combined = combined.drop_duplicates(subset=["location_id", "observed_at"], keep="last")
                combined = combined.sort_values(["location_id", "observed_at"])
                combined["observed_at"] = combined["observed_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
                combined.to_csv(args.output, index=False)
        except Exception as exc:
            print("      WARNING: could not update combined output:", exc)

        try:
            with open(args.manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined["observed_at"] = pd.to_datetime(combined["observed_at"], errors="coerce")
        combined = combined[combined["observed_at"].notna()]
        combined = combined.drop_duplicates(subset=["location_id", "observed_at"], keep="last")
        combined = combined.sort_values(["location_id", "observed_at"])
        combined["observed_at"] = combined["observed_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        combined.to_csv(args.output, index=False)
        manifest["final_rows"] = int(len(combined))
    else:
        manifest["final_rows"] = 0

    manifest["successful_locations"] = len(set(manifest["successful"]))
    manifest["failed_locations"] = len(manifest["failed"])
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDONE")
    print("Successful locations: {} / {}".format(manifest["successful_locations"], len(locations)))
    print("Failed locations:", manifest["failed_locations"])
    print("Rows:", manifest["final_rows"])
    print("Weather output:", args.output)
    print("Manifest:", args.manifest)
    if manifest["failed_locations"]:
        print("\nRe-run the SAME command later. Completed district caches will be reused.")


if __name__ == "__main__":
    main()
