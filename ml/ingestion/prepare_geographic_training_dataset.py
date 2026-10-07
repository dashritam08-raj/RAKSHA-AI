import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_col(df, names, required=False):
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in norm:
            return norm[key]
    if required:
        raise RuntimeError("Required column not found. Tried: {}. Columns: {}".format(names, list(df.columns)))
    return None


def num_series(s):
    return pd.to_numeric(s, errors="coerce")


def norm_text(s):
    return s.astype(str).str.strip().str.casefold()


def parse_dt(s):
    return pd.to_datetime(s, errors="coerce", utc=True)


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180.0
    a1 = lat1 * p
    a2 = lat2 * p
    da = (lat2 - lat1) * p
    dl = (lon2 - lon1) * p
    a = math.sin(da / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def prepare_events(path):
    df = load_csv(path).copy()
    id_col = pick_col(df, ["incident_id", "serial", "id"], required=True)
    dt_col = pick_col(df, ["observed_at", "date", "fechano", "event_date", "datetime"], required=True)
    # The Phase 5D geocoder writes resolved coordinates as geo_latitude/geo_longitude.
    # Prefer those over generic latitude/longitude fields because the source XML
    # coordinates are preserved separately as xml_latitude/xml_longitude.
    lat_col = pick_col(df, ["geo_latitude", "latitude", "lat"], required=True)
    lon_col = pick_col(df, ["geo_longitude", "longitude", "lon", "lng"], required=True)
    district_col = pick_col(df, ["district", "district_name", "name2"])
    province_col = pick_col(df, ["region", "province", "province_name", "name1"])
    type_col = pick_col(df, ["disaster_type", "event", "evento", "hazard_type"])
    band_col = pick_col(df, ["impact_band", "outcome_band", "consequence_band", "label", "risk_band"])
    score_col = pick_col(df, ["impact_score", "consequence_score", "outcome_score"])

    out = pd.DataFrame()
    # Use a stable internal row key for feature joins. Historical source files can
    # contain blank incident IDs, and converting NaN to the string "nan" creates
    # duplicate merge keys. The original incident_id is still preserved.
    out["event_row_id"] = range(len(df))
    out["incident_id"] = df[id_col].astype("string")
    out["observed_at"] = parse_dt(df[dt_col])
    out["latitude"] = num_series(df[lat_col])
    out["longitude"] = num_series(df[lon_col])
    out["district"] = df[district_col].astype(str).str.strip() if district_col else ""
    out["province"] = df[province_col].astype(str).str.strip() if province_col else ""
    out["disaster_type"] = df[type_col].astype(str).str.strip().str.upper() if type_col else "UNKNOWN"

    # Preserve every numeric impact-like field that is already in the geocoded file.
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["death", "dead", "injur", "missing", "affected", "evacuat", "relocat", "destroy", "damage", "loss"]):
            if c not in [lat_col, lon_col, dt_col, id_col]:
                n = num_series(df[c])
                if n.notna().any():
                    safe = re.sub(r"[^a-z0-9]+", "_", lc).strip("_")
                    if safe and safe not in out.columns:
                        out[safe] = n

    if band_col:
        out["impact_band"] = df[band_col].astype("string").str.upper().str.strip()
    else:
        out["impact_band"] = "NO_DATA"

    if score_col:
        out["impact_score"] = num_series(df[score_col])
    else:
        out["impact_score"] = pd.NA

    out["event_date"] = out["observed_at"].dt.floor("D")
    return out


def prepare_weather(path):
    df = load_csv(path).copy()
    dt_col = pick_col(df, ["observed_at", "datetime", "date", "time", "timestamp"], required=True)
    lat_col = pick_col(df, ["latitude", "lat"])
    lon_col = pick_col(df, ["longitude", "lon", "lng"])
    if lat_col is None or lon_col is None:
        raise RuntimeError("Weather file must contain latitude and longitude columns. Columns: {}".format(list(df.columns)))

    w = pd.DataFrame()
    w["weather_time"] = parse_dt(df[dt_col])
    w["weather_latitude"] = num_series(df[lat_col])
    w["weather_longitude"] = num_series(df[lon_col])
    w["weather_date"] = w["weather_time"].dt.floor("D")

    variable_aliases = {
        "temperature_mean_c": ["temperature_2m_mean", "temperature_mean", "temp_mean_c", "temperature"],
        "temperature_max_c": ["temperature_2m_max", "temperature_max", "temp_max_c"],
        "temperature_min_c": ["temperature_2m_min", "temperature_min", "temp_min_c"],
        "precipitation_mm": ["precipitation_sum", "precipitation", "rainfall", "rain_mm"],
        "wind_max_kmh": ["wind_speed_10m_max", "wind_max", "wind_speed_max", "wind_kmh"],
        "wind_gust_max_kmh": ["wind_gusts_10m_max", "wind_gust_max", "gust_kmh"],
    }
    for target, aliases in variable_aliases.items():
        c = pick_col(df, aliases)
        if c:
            w[target] = num_series(df[c])

    value_cols = [c for c in w.columns if c in variable_aliases]
    if value_cols:
        agg = {c: "mean" for c in value_cols}
        w = w.groupby(["weather_date", "weather_latitude", "weather_longitude"], as_index=False).agg(agg)
    else:
        w = w.drop_duplicates(subset=["weather_date", "weather_latitude", "weather_longitude"])
    return w


def weather_features(events, weather, max_distance_km=150.0):
    # Match same-day weather station/grid cell by nearest haversine distance.
    weather_by_day = {}
    for day, g in weather.groupby("weather_date"):
        weather_by_day[day] = g.reset_index(drop=True)

    result = []
    matched = 0
    for row in events.itertuples(index=False):
        base = {"event_row_id": row.event_row_id}
        candidates = weather_by_day.get(row.event_date)
        if candidates is None or pd.isna(row.latitude) or pd.isna(row.longitude):
            base["weather_match_distance_km"] = pd.NA
            for c in ["temperature_mean_c", "temperature_max_c", "temperature_min_c", "precipitation_mm", "wind_max_kmh", "wind_gust_max_kmh"]:
                base[c] = pd.NA
            result.append(base)
            continue
        best_i = None
        best_d = None
        for i, w in candidates.iterrows():
            if pd.isna(w.weather_latitude) or pd.isna(w.weather_longitude):
                continue
            d = haversine_km(row.latitude, row.longitude, w.weather_latitude, w.weather_longitude)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        if best_i is not None and best_d <= max_distance_km:
            matched += 1
            best = candidates.iloc[best_i]
            base["weather_match_distance_km"] = round(best_d, 3)
            for c in ["temperature_mean_c", "temperature_max_c", "temperature_min_c", "precipitation_mm", "wind_max_kmh", "wind_gust_max_kmh"]:
                base[c] = best[c] if c in best.index else pd.NA
        else:
            base["weather_match_distance_km"] = best_d if best_d is not None else pd.NA
            for c in ["temperature_mean_c", "temperature_max_c", "temperature_min_c", "precipitation_mm", "wind_max_kmh", "wind_gust_max_kmh"]:
                base[c] = pd.NA
        result.append(base)
    return pd.DataFrame(result), matched


def prepare_earthquakes(path):
    df = load_csv(path).copy()
    dt_col = pick_col(df, ["observed_at", "time", "datetime", "timestamp", "event_time"], required=True)
    lat_col = pick_col(df, ["latitude", "lat"], required=True)
    lon_col = pick_col(df, ["longitude", "lon", "lng"], required=True)
    mag_col = pick_col(df, ["magnitude", "mag", "magnitude_mw", "magtype_value"])
    q = pd.DataFrame()
    q["eq_time"] = parse_dt(df[dt_col])
    q["eq_latitude"] = num_series(df[lat_col])
    q["eq_longitude"] = num_series(df[lon_col])
    q["eq_magnitude"] = num_series(df[mag_col]) if mag_col else pd.NA
    return q.dropna(subset=["eq_time", "eq_latitude", "eq_longitude"])


def earthquake_features(events, eq, radius_km=300.0, lookback_days=7):
    eq = eq.sort_values("eq_time").reset_index(drop=True)
    out = []
    for row in events.itertuples(index=False):
        f = {"event_row_id": row.event_row_id, "eq_count_7d_300km": 0, "eq_max_magnitude_7d_300km": pd.NA, "eq_nearest_km_7d_300km": pd.NA}
        if pd.isna(row.latitude) or pd.isna(row.longitude) or pd.isna(row.observed_at):
            out.append(f)
            continue
        start = row.observed_at - pd.Timedelta(days=lookback_days)
        end = row.observed_at
        cand = eq[(eq["eq_time"] >= start) & (eq["eq_time"] <= end)]
        if cand.empty:
            out.append(f)
            continue
        mags = []
        dists = []
        for r in cand.itertuples(index=False):
            d = haversine_km(row.latitude, row.longitude, r.eq_latitude, r.eq_longitude)
            if d <= radius_km:
                dists.append(d)
                if not pd.isna(r.eq_magnitude):
                    mags.append(float(r.eq_magnitude))
        if dists:
            f["eq_count_7d_300km"] = len(dists)
            f["eq_nearest_km_7d_300km"] = round(min(dists), 3)
            if mags:
                f["eq_max_magnitude_7d_300km"] = max(mags)
        out.append(f)
    return pd.DataFrame(out)


def add_temporal_features(df):
    dt = df["observed_at"]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["day_of_week"] = dt.dt.dayofweek
    df["month_sin"] = (2 * math.pi * df["month"] / 12.0).apply(math.sin)
    df["month_cos"] = (2 * math.pi * df["month"] / 12.0).apply(math.cos)
    return df


def main():
    ap = argparse.ArgumentParser(description="Build RAKSHA AI geographic ML training dataset from historical incidents, weather and earthquakes.")
    ap.add_argument("--events", default="data/processed/desinventar_nepal_geocoded.csv")
    ap.add_argument("--weather", default="data/raw/weather_observations.csv")
    ap.add_argument("--earthquake", default="data/raw/earthquake_events.csv")
    ap.add_argument("--output", default="data/processed/nepal_geographic_training_dataset.csv")
    ap.add_argument("--quality-output", default="data/processed/nepal_geographic_training_quality.json")
    ap.add_argument("--weather-max-distance-km", type=float, default=150.0)
    ap.add_argument("--earthquake-radius-km", type=float, default=300.0)
    ap.add_argument("--earthquake-lookback-days", type=int, default=7)
    args = ap.parse_args()

    print("[1/7] Loading geocoded historical incidents...")
    events = prepare_events(args.events)
    print("      Historical rows:", len(events))

    valid_geo = events["latitude"].notna() & events["longitude"].notna()
    labeled = events["impact_band"].isin(["LOW", "MODERATE", "HIGH", "CRITICAL"])
    print("      Rows with coordinates:", int(valid_geo.sum()))
    print("      Rows with impact labels:", int(labeled.sum()))

    print("[2/7] Loading historical weather...")
    weather = prepare_weather(args.weather)
    print("      Weather rows:", len(weather))

    print("[3/7] Matching same-day nearest weather observation...")
    wf, weather_matches = weather_features(events, weather, args.weather_max_distance_km)
    out = events.merge(wf, on="event_row_id", how="left", validate="one_to_one")
    print("      Weather-matched incidents:", weather_matches)

    print("[4/7] Loading earthquake catalog...")
    eq = prepare_earthquakes(args.earthquake)
    print("      Earthquake rows:", len(eq))

    print("[5/7] Creating leakage-safe earthquake features...")
    ef = earthquake_features(events, eq, args.earthquake_radius_km, args.earthquake_lookback_days)
    out = out.merge(ef, on="event_row_id", how="left", validate="one_to_one")

    print("[6/7] Adding temporal and data-quality features...")
    out = add_temporal_features(out)
    # Phase 5D geocoded files use geo_source; retain it in the training artifact.
    if "geo_source" not in out.columns:
        if "coordinate_source" in out.columns:
            out["geo_source"] = out["coordinate_source"]
        else:
            out["geo_source"] = "unknown"
    out["has_coordinates"] = out["latitude"].notna() & out["longitude"].notna()
    out["has_weather"] = out["temperature_mean_c"].notna() | out["precipitation_mm"].notna() | out["wind_max_kmh"].notna()
    out["has_earthquake_context"] = out["eq_count_7d_300km"].fillna(0).astype(float) > 0
    out = out.sort_values(["observed_at", "event_row_id"], na_position="last").reset_index(drop=True)
    # The internal join key is an implementation detail and should not leak into
    # downstream training artifacts.
    out = out.drop(columns=["event_row_id"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    quality = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "events_rows": int(len(out)),
        "coordinate_rows": int(out["has_coordinates"].sum()),
        "coordinate_coverage_pct": round(float(out["has_coordinates"].mean() * 100.0), 2),
        "impact_labeled_rows": int(labeled.sum()),
        "impact_label_counts": {str(k): int(v) for k, v in Counter(out["impact_band"].fillna("NO_DATA")).items()},
        "weather_rows": int(len(weather)),
        "weather_matched_rows": int(out["has_weather"].sum()),
        "weather_match_coverage_pct": round(float(out["has_weather"].mean() * 100.0), 2),
        "earthquake_rows": int(len(eq)),
        "earthquake_context_rows": int(out["has_earthquake_context"].sum()),
        "earthquake_context_coverage_pct": round(float(out["has_earthquake_context"].mean() * 100.0), 2),
        "weather_max_distance_km": args.weather_max_distance_km,
        "earthquake_radius_km": args.earthquake_radius_km,
        "earthquake_lookback_days": args.earthquake_lookback_days,
        "notes": [
            "Weather features use observations on or matching the event date and nearest station/grid point within the configured distance.",
            "Earthquake features only use earthquakes from the prior lookback window through the event timestamp; no future observations are used.",
            "NO_DATA is retained as a label state from historical consequence labeling and is not treated as zero impact.",
            "Resolved training coordinates come from geo_latitude/geo_longitude produced by Phase 5D; original XML coordinates remain preserved separately.",
            "Historical Nawalparasi and Rukum records remain unresolved geographically when the legacy district name is ambiguous.",
            "This dataset is an ML development/training artifact and requires time-aware validation before operational use."
        ]
    }
    with open(args.quality_output, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    print("[7/7] DONE")
    print("Output:", args.output)
    print("Quality:", args.quality_output)
    print("Rows:", len(out))
    print("Coordinate coverage: {:.2f}%".format(quality["coordinate_coverage_pct"]))
    print("Weather coverage: {:.2f}%".format(quality["weather_match_coverage_pct"]))
    print("Earthquake context coverage: {:.2f}%".format(quality["earthquake_context_coverage_pct"]))


if __name__ == "__main__":
    main()
