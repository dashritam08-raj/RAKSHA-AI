"""Import DesInventar Standard XML (raw XML or ZIP) into a clean normalized CSV.

The DesInventar export can contain non-event ``TR`` rows.  This importer keeps
only event-like records with a serial/id, a valid date, and a disaster type,
and de-duplicates repeated incident IDs by retaining the richest observed
impact information. Coordinates are never invented.
"""
import argparse
import csv
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

ALIASES = {
    "serial": ["serial", "id"],
    "year": ["fechano", "year"], "month": ["fechames", "month"], "day": ["fechadia", "day"],
    "event": ["evento", "event", "event_type"], "location": ["lugar", "location", "locality"],
    "level0": ["level0", "country", "pais"], "level1": ["level1", "region", "state", "province"],
    "level2": ["level2", "district", "municipio", "county"],
    "deaths": ["muertos", "muertes", "dead", "deaths"],
    "missing": ["desaparece", "missing", "desaparecidos"],
    "injured": ["heridos", "injured", "lesionados"],
    "directly_affected": ["afectados", "directly_affected", "affected"],
    "indirectly_affected": ["indirectamente", "indirectly_affected", "indirect_affected"],
    "evacuated": ["evacuados", "evacuated"], "relocated": ["reubicados", "relocated"],
    "houses_destroyed": ["viviendas_destruidas", "casas_destruidas", "houses_destroyed"],
    "houses_damaged": ["viviendas_afectadas", "casas_afectadas", "houses_damaged"],
    "road_damage_m": ["damages_in_roads_mts", "roads_damaged", "road_damage_m"],
    "economic_loss_usd": ["losses_usd", "loss_usd", "economic_loss_usd"],
    "economic_loss_local": ["losses_local", "loss_local", "economic_loss_local"],
}

OUT = [
    "incident_id", "observed_at", "disaster_type", "country_code", "region", "district", "location",
    "deaths", "missing", "injured", "directly_affected", "indirectly_affected", "evacuated", "relocated",
    "houses_destroyed", "houses_damaged", "road_damage_m", "economic_loss_usd", "economic_loss_local",
    "source_name", "source_url", "raw_record_json"
]

NUMERIC_FIELDS = [
    "deaths", "missing", "injured", "directly_affected", "indirectly_affected", "evacuated", "relocated",
    "houses_destroyed", "houses_damaged", "road_damage_m", "economic_loss_usd", "economic_loss_local"
]


def local(tag):
    return tag.split("}")[-1].strip().lower()


def clean(v):
    return re.sub(r"\s+", " ", str(v or "").strip())


def find(d, keys):
    for k in keys:
        value = clean(d.get(k))
        if value:
            return value
    return ""


def num(v):
    s = clean(v).replace(",", "")
    if not s:
        return 0.0
    try:
        return float(s)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else 0.0


def strict_date(y, m, d):
    try:
        ys, ms, ds = clean(y), clean(m), clean(d)
        if not (ys and ms and ds):
            return ""
        return datetime(int(float(ys)), int(float(ms)), int(float(ds))).date().isoformat()
    except Exception:
        return ""


def children(tr):
    return {local(c.tag): clean(c.text) for c in list(tr)}


def extract_tree(path):
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".xml") and "di_export" in Path(n).name.lower()]
            if not names:
                names = [n for n in z.namelist() if n.lower().endswith(".xml")]
            if not names:
                raise ValueError("No XML file found in ZIP")
            with z.open(names[0]) as f:
                return ET.parse(f)
    return ET.parse(p)


def normalize(d, source_url):
    v = lambda k: find(d, ALIASES[k])
    cc = v("level0").upper()
    sid = v("serial")
    return {
        "incident_id": f"{(cc or 'unk').lower()}-{sid}",
        "observed_at": strict_date(v("year"), v("month"), v("day")),
        "disaster_type": v("event").upper(),
        "country_code": cc,
        "region": v("level1"),
        "district": v("level2"),
        "location": v("location"),
        "deaths": int(num(v("deaths"))),
        "missing": int(num(v("missing"))),
        "injured": int(num(v("injured"))),
        "directly_affected": int(num(v("directly_affected"))),
        "indirectly_affected": int(num(v("indirectly_affected"))),
        "evacuated": int(num(v("evacuated"))),
        "relocated": int(num(v("relocated"))),
        "houses_destroyed": int(num(v("houses_destroyed"))),
        "houses_damaged": int(num(v("houses_damaged"))),
        "road_damage_m": num(v("road_damage_m")),
        "economic_loss_usd": num(v("economic_loss_usd")),
        "economic_loss_local": num(v("economic_loss_local")),
        "source_name": "DesInventar",
        "source_url": source_url,
        "raw_record_json": json.dumps(d, ensure_ascii=False, sort_keys=True),
    }


def richness(row):
    positive = sum(float(row.get(c, 0) or 0) > 0 for c in NUMERIC_FIELDS)
    geo = sum(bool(clean(row.get(c))) for c in ["country_code", "region", "district", "location"])
    return positive, geo, len(clean(row.get("raw_record_json")))


def main():
    p = argparse.ArgumentParser(description="Normalize DesInventar XML/ZIP event records")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-url", default="https://www.desinventar.net/DesInventar/")
    args = p.parse_args()

    root = extract_tree(args.input).getroot()
    raw_count = 0
    skipped_missing_id = 0
    skipped_invalid_date = 0
    skipped_missing_event = 0
    duplicates = 0
    by_id = {}

    for node in root.iter():
        if local(node.tag) != "tr":
            continue
        raw_count += 1
        row_data = children(node)
        normalized = normalize(row_data, args.source_url)

        serial = find(row_data, ALIASES["serial"])
        if not serial:
            skipped_missing_id += 1
            continue
        if not normalized["observed_at"]:
            skipped_invalid_date += 1
            continue
        if not normalized["disaster_type"]:
            skipped_missing_event += 1
            continue

        incident_id = normalized["incident_id"]
        if incident_id in by_id:
            duplicates += 1
            if richness(normalized) > richness(by_id[incident_id]):
                by_id[incident_id] = normalized
        else:
            by_id[incident_id] = normalized

    rows = list(by_id.values())
    rows.sort(key=lambda x: (x["observed_at"], x["incident_id"]))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUT)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "input": args.input,
        "output": str(out),
        "raw_tr_rows_seen": raw_count,
        "accepted_unique_events": len(rows),
        "skipped_missing_id": skipped_missing_id,
        "skipped_invalid_date": skipped_invalid_date,
        "skipped_missing_event": skipped_missing_event,
        "deduplicated_event_rows": duplicates,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
