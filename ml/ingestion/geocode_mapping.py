"""One-time, cached geocoding helper for the event-location mapping.

Uses the public OpenStreetMap Nominatim service only for a small, user-initiated
mapping job. It enforces a single request stream, a delay between requests, a
custom User-Agent, and local caching. Results remain REVIEW_REQUIRED until a
human approves them.
"""
from __future__ import annotations
import argparse, csv, json, time, urllib.parse, urllib.request
from pathlib import Path

NOMINATIM = "https://nominatim.openstreetmap.org/search"


def search(q: str, email: str | None):
    params = {"q": q, "format": "jsonv2", "limit": 1, "countrycodes": "np", "addressdetails": 1}
    if email:
        params["email"] = email
    url = NOMINATIM + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "RAKSHA-AI/phase5d-location-mapper (research; contact required in deployment)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default="data/processed/nominatim_cache.json")
    p.add_argument("--delay-seconds", type=float, default=1.1)
    p.add_argument("--email", default=None)
    p.add_argument("--acknowledge-policy", action="store_true")
    args = p.parse_args()
    if not args.acknowledge_policy:
        raise SystemExit("Add --acknowledge-policy after reviewing https://operations.osmfoundation.org/policies/nominatim/")
    inp, out, cache_path = Path(args.input), Path(args.output), Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True); out.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    with inp.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f)); fields = f.readline() if False else None
    # Geocode unique districts only. Location rows inherit district coordinates.
    district_rows = {}
    for row in rows:
        key = (row.get("source_region","").strip(), row.get("source_district","").strip())
        district_rows.setdefault(key, row)
    results = {}
    failures=[]
    for (region, district), row in district_rows.items():
        if not district:
            continue
        key = f"{region}|{district}|Nepal"
        if key in cache:
            results[key] = cache[key]; continue
        q = f"{district}, {region}, Nepal" if region else f"{district}, Nepal"
        try:
            data = search(q, args.email)
            if data:
                item = data[0]
                cache[key] = {"latitude": float(item["lat"]), "longitude": float(item["lon"]), "display_name": item.get("display_name",""), "source": "Nominatim/OSM"}
                results[key] = cache[key]
            else:
                failures.append({"query":q,"error":"no_result"})
        except Exception as exc:
            failures.append({"query":q,"error":str(exc)})
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        time.sleep(max(args.delay_seconds, 1.0))

    for row in rows:
        key=f"{row.get('source_region','').strip()}|{row.get('source_district','').strip()}|Nepal"
        hit=results.get(key)
        if hit:
            row["latitude"]=hit["latitude"]; row["longitude"]=hit["longitude"]
            row["coordinate_source"]=hit["source"]
            row["geocode_name"]=hit.get("display_name","")
            row["review_status"]="AUTO_GEOCODED_REVIEW_REQUIRED"
        else:
            row.setdefault("coordinate_source",""); row.setdefault("geocode_name","")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["source_region"])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output":str(out),"unique_district_queries":len(district_rows),"geocoded":len(results),"failures":failures,"cache":str(cache_path),"warning":"Auto-geocoded coordinates must be human reviewed before training."}, indent=2))

if __name__=="__main__": main()
