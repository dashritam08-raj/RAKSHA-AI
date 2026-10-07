"""Build a reviewable geographic mapping table from cleaned DesInventar events.

The output contains one row per (region, district, location) combination with event counts.
Coordinates are intentionally blank unless a matching reviewed mapping already exists.
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
import pandas as pd


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return s or "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    required = {"region", "district", "location"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit("Missing columns: " + ", ".join(sorted(missing)))
    for c in required:
        df[c] = df[c].fillna("").astype(str).str.strip()
    grouped = (df.groupby(["region","district","location"], dropna=False)
                 .size().reset_index(name="event_count")
                 .sort_values("event_count", ascending=False))
    rows = []
    for r in grouped.itertuples(index=False):
        # District-level location ids are stable enough for first-pass environmental joins.
        base = r.district or r.region or "unknown"
        loc_id = "npl_" + slug(base)
        rows.append({
            "source_region": r.region,
            "source_district": r.district,
            "source_location": r.location,
            "location_id": loc_id,
            "latitude": "",
            "longitude": "",
            "coordinate_level": "DISTRICT_CENTROID",
            "review_status": "REVIEW_REQUIRED",
            "event_count": int(r.event_count),
        })
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print({"output": str(out), "mapping_rows": len(rows), "events": int(len(df)), "warning": "Coordinates are blank and require review/geocoding before training."})

if __name__ == "__main__":
    main()
