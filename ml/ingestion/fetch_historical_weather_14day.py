from __future__ import print_function

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

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


def build_windows(start_date, end_date, days=14):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    result = []
    cursor = start
    while cursor <= end:
        w_end = min(cursor + timedelta(days=days - 1), end)
        result.append((cursor.isoformat(), w_end.isoformat()))
        cursor = w_end + timedelta(days=1)
    return result


def build_url(locations, start_date, end_date):
    # Open-Meteo accepts comma-separated latitude/longitude lists and returns
    # one JSON structure per requested location.
    params = [
        ("latitude", ",".join("{:.6f}".format(x["latitude"]) for x in locations)),
        ("longitude", ",".join("{:.6f}".format(x["longitude"]) for x in locations)),
        ("start_date", start_date),
        ("end_date", end_date),
        ("daily", ",".join(DAILY_VARS)),
        ("timezone", "Asia/Kathmandu"),
        ("temperature_unit", "celsius"),
        ("wind_speed_unit", "kmh"),
        ("precipitation_unit", "mm"),
    ]
    return API_URL + "?" + urllib.parse.urlencode(params)


def request_json(url, attempts, base_delay, max_429_wait):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "RAKSHA-AI-historical-weather/1.1",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=240) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after)
                except Exception:
                    delay = min(max_429_wait, base_delay * (2 ** (attempt - 1)))
                delay = min(max_429_wait, max(base_delay, delay))
                print("      HTTP 429; waiting {:.0f}s (attempt {}/{})".format(
                    delay, attempt, attempts
                ))
                time.sleep(delay)
            elif 500 <= exc.code < 600:
                delay = min(120.0, base_delay * attempt)
                print("      HTTP {}; waiting {:.0f}s (attempt {}/{})".format(
                    exc.code, delay, attempt, attempts
                ))
                time.sleep(delay)
            else:
                # 400/401/403 should not be retried unchanged. The caller may
                # split a location batch on 400.
                raise
        except Exception as exc:
            last_error = exc
            delay = min(60.0, base_delay * attempt)
            print("      request error {}; waiting {:.0f}s (attempt {}/{})".format(
                exc, delay, attempt, attempts
            ))
            time.sleep(delay)

    raise RuntimeError("Request failed after {} attempts: {}".format(attempts, last_error))


def normalize(payload, locations, start_date, end_date):
    items = [payload] if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise RuntimeError("Unexpected Open-Meteo response type.")
    if len(items) != len(locations):
        raise RuntimeError(
            "Expected {} location responses, received {}.".format(
                len(locations), len(items)
            )
        )

    rows = []
    for i, item in enumerate(items):
        daily = item.get("daily") or {}
        dates = daily.get("time") or []
        for j, date_value in enumerate(dates):
            row = {
                "location_id": locations[i]["location_id"],
                "label": locations[i]["label"],
                "latitude": locations[i]["latitude"],
                "longitude": locations[i]["longitude"],
                "observed_at": "{}T00:00:00".format(date_value),
                "source": "Open-Meteo Historical Weather Archive",
                "source_url": API_URL,
            }
            for var in DAILY_VARS:
                values = daily.get(var) or []
                row[var] = values[j] if j < len(values) else None
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Zero weather rows returned for {} to {}".format(start_date, end_date))
    return frame


def fetch_with_fallback(locations, start_date, end_date, args):
    """Try one multi-location request; on HTTP 400 split locations recursively."""
    try:
        payload = request_json(
            build_url(locations, start_date, end_date),
            attempts=args.attempts,
            base_delay=args.retry_delay,
            max_429_wait=args.max_429_wait,
        )
        return [normalize(payload, locations, start_date, end_date)]
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        if len(locations) == 1:
            raise RuntimeError(
                "HTTP 400 for single location {} ({}) during {} to {}".format(
                    locations[0]["location_id"], locations[0]["label"], start_date, end_date
                )
            )
        mid = len(locations) // 2
        print("      HTTP 400; splitting {} locations into {} + {}".format(
            len(locations), mid, len(locations) - mid
        ))
        left = fetch_with_fallback(locations[:mid], start_date, end_date, args)
        time.sleep(args.split_pause)
        right = fetch_with_fallback(locations[mid:], start_date, end_date, args)
        return left + right


def read_cache(path, expected_ids):
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
    ap.add_argument("--locations", default="data/mappings/nepal_district_locations.csv")
    ap.add_argument("--start-date", default="1971-01-01")
    ap.add_argument("--end-date", default="2013-12-31")
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--pause-seconds", type=float, default=3.0)
    ap.add_argument("--split-pause", type=float, default=2.0)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--retry-delay", type=float, default=10.0)
    ap.add_argument("--max-429-wait", type=float, default=600.0)
    ap.add_argument("--output", default="data/raw/historical_weather_observations_14day.csv")
    ap.add_argument("--batch-dir", default="data/raw/weather_14day_batches")
    args = ap.parse_args()

    locations = load_locations(args.locations)
    windows = build_windows(args.start_date, args.end_date, args.window_days)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.batch_dir, exist_ok=True)

    print("RAKSHA AI — corrected 14-day historical weather recovery")
    print("Period: {} to {}".format(args.start_date, args.end_date))
    print("Gazetteer locations:", len(locations))
    print("Windows:", len(windows))
    print("Primary strategy: all locations per 14-day request, automatic 400 fallback splitting")
    print("No synthetic values are generated.")

    frames = []
    manifest = {
        "api": API_URL,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "window_days": args.window_days,
        "location_count": len(locations),
        "windows_total": len(windows),
        "successful_windows": [],
        "failures": [],
    }

    expected_ids = [x["location_id"] for x in locations]

    for number, (start, end) in enumerate(windows, 1):
        merged_cache = os.path.join(args.batch_dir, "weather_{:04d}_merged_{}_{}.csv".format(
            number, start, end
        ))
        cached = read_cache(merged_cache, expected_ids)
        if cached is not None:
            print("[{}/{}] cached {} rows: {} to {}".format(
                number, len(windows), len(cached), start, end
            ))
            frames.append(cached)
            manifest["successful_windows"].append({
                "window": number,
                "start": start,
                "end": end,
                "status": "cached",
                "rows": int(len(cached)),
                "output": merged_cache,
            })
            continue

        print("[{}/{}] fetching {} to {} for {} locations".format(
            number, len(windows), start, end, len(locations)
        ))
        try:
            pieces = fetch_with_fallback(locations, start, end, args)
            frame = pd.concat(pieces, ignore_index=True)
            frame["observed_at"] = pd.to_datetime(frame["observed_at"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
            frame = frame.drop_duplicates(subset=["location_id", "observed_at"]).sort_values(["location_id", "observed_at"])
            frame.to_csv(merged_cache, index=False)
            frames.append(frame)
            manifest["successful_windows"].append({
                "window": number,
                "start": start,
                "end": end,
                "status": "downloaded",
                "rows": int(len(frame)),
                "output": merged_cache,
            })
            print("      rows:", len(frame))
        except Exception as exc:
            print("      FAILED:", exc)
            manifest["failures"].append({
                "window": number,
                "start": start,
                "end": end,
                "error": str(exc),
            })
        if number < len(windows):
            time.sleep(max(0.0, args.pause_seconds))

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged["observed_at"] = pd.to_datetime(merged["observed_at"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
        merged = merged.drop_duplicates(subset=["location_id", "observed_at"]).sort_values(["location_id", "observed_at"])
        merged.to_csv(args.output, index=False)
    else:
        merged = pd.DataFrame()

    manifest["successful_window_count"] = len(manifest["successful_windows"])
    manifest["failed_window_count"] = len(manifest["failures"])
    manifest["final_rows"] = int(len(merged))
    manifest_path = os.path.join(os.path.dirname(args.output), "historical_weather_14day_manifest.json")
    manifest["output"] = args.output
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDONE")
    print("Successful windows: {} / {}".format(manifest["successful_window_count"], len(windows)))
    print("Failed windows:", manifest["failed_window_count"])
    print("Rows:", manifest["final_rows"])
    print("Weather output:", args.output)
    print("Manifest:", manifest_path)
    if manifest["failures"]:
        print("\nRun the same command again. Successful 14-day windows will be reused.")


if __name__ == "__main__":
    main()