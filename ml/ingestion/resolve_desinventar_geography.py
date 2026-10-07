from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None


NEPAL_BOUNDS = {
    "lat_min": 25.0,
    "lat_max": 31.5,
    "lon_min": 80.0,
    "lon_max": 89.5,
}


def local_name(tag):
    if not tag:
        return ""
    return tag.rsplit("}", 1)[-1]


def text_of(parent, name):
    for child in list(parent):
        if local_name(child.tag) == name:
            value = child.text
            return (value or "").strip()
    return ""


def normalize_text(value):
    value = "" if value is None else str(value)
    value = value.strip().upper()
    value = re.sub(r"[\u00a0\t\r\n]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_code(value):
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # Recover codes that pandas/CSV may render as x.0.
    if re.fullmatch(r"[+-]?\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s


def normalize_number(value):
    try:
        x = float(str(value).strip())
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def valid_coord(lat, lon):
    if lat is None or lon is None:
        return False
    return (
        NEPAL_BOUNDS["lat_min"] <= lat <= NEPAL_BOUNDS["lat_max"]
        and NEPAL_BOUNDS["lon_min"] <= lon <= NEPAL_BOUNDS["lon_max"]
    )


def read_zip_xml(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        xml_names = [n for n in names if n.lower().endswith(".xml")]
        if not xml_names:
            raise RuntimeError("No XML file found inside: %s" % zip_path)

        preferred = [
            n for n in xml_names
            if os.path.basename(n).lower() == "di_export_npl.xml"
        ]
        xml_name = preferred[0] if preferred else xml_names[0]
        raw = z.read(xml_name)

    root = ET.fromstring(raw)
    return xml_name, root


def extract_records(root):
    records = []
    accepted = 0
    skipped_missing_id = 0
    skipped_invalid_date = 0
    skipped_missing_event = 0

    for tr in root.iter():
        if local_name(tr.tag) != "TR":
            continue

        serial = text_of(tr, "serial")
        event = text_of(tr, "evento")
        observed = text_of(tr, "fechano")
        month = text_of(tr, "fechames")
        day = text_of(tr, "fechadia")

        if not serial:
            skipped_missing_id += 1
            continue
        if not event:
            skipped_missing_event += 1
            continue

        # Keep the same validity spirit as the existing importer: the XML
        # must contain a usable year. The exact date is reconstructed where possible.
        if not re.fullmatch(r"\d{4}", observed):
            skipped_invalid_date += 1
            continue

        level1 = normalize_code(text_of(tr, "level1"))
        name1 = text_of(tr, "name1")
        level2 = normalize_code(text_of(tr, "level2"))
        name2 = text_of(tr, "name2")
        lat = normalize_number(text_of(tr, "latitude"))
        lon = normalize_number(text_of(tr, "longitude"))

        records.append(
            {
                "xml_serial": normalize_code(serial),
                "observed_year": int(observed),
                "observed_month": int(month) if month.isdigit() else None,
                "observed_day": int(day) if day.isdigit() else None,
                "disaster_type_raw": event,
                "region_code_xml": level1,
                "district_code_xml": level2,
                "district_name_xml": name1,
                "locality_name_xml": name2,
                "xml_latitude": lat,
                "xml_longitude": lon,
                "event_coordinates_valid": bool(valid_coord(lat, lon)),
            }
        )
        accepted += 1

    stats = {
        "raw_tr_rows_seen": sum(1 for x in root.iter() if local_name(x.tag) == "TR"),
        "accepted_event_rows": accepted,
        "skipped_missing_id": skipped_missing_id,
        "skipped_invalid_date": skipped_invalid_date,
        "skipped_missing_event": skipped_missing_event,
    }
    return records, stats


def build_admin_catalog(records):
    region_names = defaultdict(set)
    district_names = defaultdict(set)
    district_code_to_names = defaultdict(set)

    for r in records:
        if r["region_code_xml"] and r["district_name_xml"]:
            region_names[r["region_code_xml"]].add(r["district_name_xml"].strip())
        if r["district_code_xml"] and r["district_name_xml"]:
            district_code_to_names[r["district_code_xml"]].add(r["district_name_xml"].strip())
        if r["district_name_xml"]:
            district_names[normalize_text(r["district_name_xml"])].add(r["district_name_xml"].strip())

    rows = []
    for code in sorted(region_names):
        for name in sorted(region_names[code]):
            rows.append({
                "admin_level": "level1",
                "code": code,
                "name": name,
            })

    for code in sorted(district_code_to_names):
        for name in sorted(district_code_to_names[code]):
            rows.append({
                "admin_level": "level2_code_but_district_name1",
                "code": code,
                "name": name,
            })

    return rows, {
        "distinct_level1_codes": len(region_names),
        "distinct_district_codes": len(district_code_to_names),
        "distinct_district_names": len(district_names),
    }


def find_col(columns, aliases):
    lookup = {normalize_text(c).replace(" ", "_"): c for c in columns}
    for alias in aliases:
        key = normalize_text(alias).replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None


def load_gazetteer(path):
    if not path or not os.path.exists(path):
        return {}, {
            "gazetteer_loaded": False,
            "reason": "file_not_found",
        }

    if pd is not None:
        df = pd.read_csv(path)
        columns = list(df.columns)
        district_col = find_col(columns, [
            "district", "district_name", "district_en", "name", "name2", "admin2", "admin_name"
        ])
        lat_col = find_col(columns, ["latitude", "lat", "y"])
        lon_col = find_col(columns, ["longitude", "lon", "lng", "x"])
        if not district_col or not lat_col or not lon_col:
            raise RuntimeError(
                "Could not identify district/latitude/longitude columns in %s. Columns: %s"
                % (path, columns)
            )
        out = {}
        for _, row in df.iterrows():
            name = normalize_text(row[district_col])
            lat = normalize_number(row[lat_col])
            lon = normalize_number(row[lon_col])
            if name and valid_coord(lat, lon):
                out[name] = {
                    "district_name": str(row[district_col]).strip(),
                    "latitude": lat,
                    "longitude": lon,
                }
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            district_col = find_col(columns, ["district", "district_name", "district_en", "name", "name2", "admin2", "admin_name"])
            lat_col = find_col(columns, ["latitude", "lat", "y"])
            lon_col = find_col(columns, ["longitude", "lon", "lng", "x"])
            if not district_col or not lat_col or not lon_col:
                raise RuntimeError("Could not identify gazetteer columns: %s" % columns)
            out = {}
            for row in reader:
                name = normalize_text(row.get(district_col, ""))
                lat = normalize_number(row.get(lat_col, ""))
                lon = normalize_number(row.get(lon_col, ""))
                if name and valid_coord(lat, lon):
                    out[name] = {
                        "district_name": str(row[district_col]).strip(),
                        "latitude": lat,
                        "longitude": lon,
                    }

    return out, {
        "gazetteer_loaded": True,
        "gazetteer_districts": len(out),
    }


def load_labeled(path):
    if not os.path.exists(path):
        raise RuntimeError("Labeled dataset not found: %s" % path)
    if pd is None:
        raise RuntimeError("pandas is required to merge the labeled dataset. Install it with: python -m pip install pandas")

    df = pd.read_csv(path)
    columns = list(df.columns)
    id_col = find_col(columns, [
        "incident_id", "id", "serial", "uu_id", "unique_id"
    ])
    if not id_col:
        raise RuntimeError("Could not identify an incident ID column in %s. Columns: %s" % (path, columns))

    rows = df.to_dict(orient="records")
    return rows, id_col, columns


def id_candidates(value):
    s = normalize_code(value)
    if not s:
        return []
    out = [s]
    # Handle common importer forms such as NPL-12345 or npl_12345.
    m = re.search(r"(\d+)$", s)
    if m and m.group(1) not in out:
        out.append(m.group(1))
    if s.upper().startswith("NPL-"):
        tail = s[4:]
        if tail and tail not in out:
            out.append(tail)
    if s.upper().startswith("NPL_"):
        tail = s[4:]
        if tail and tail not in out:
            out.append(tail)
    return out


def map_labels_to_xml(xml_records, labeled_rows, id_col):
    by_id = {}
    for row in labeled_rows:
        key = row.get(id_col, "")
        for candidate in id_candidates(key):
            by_id.setdefault(candidate, row)

    joined = []
    matched = 0
    unmatched = 0
    used_label_ids = set()

    for x in xml_records:
        row = None
        for candidate in id_candidates(x["xml_serial"]):
            if candidate in by_id:
                row = by_id[candidate]
                break
        merged = dict(row or {})
        merged.update(x)
        if row is not None:
            matched += 1
            used_label_ids.add(str(row.get(id_col, "")))
        else:
            unmatched += 1
        joined.append(merged)

    return joined, {
        "xml_events": len(xml_records),
        "labeled_rows": len(labeled_rows),
        "matched_to_labeled": matched,
        "unmatched_xml_events": unmatched,
        "unmatched_labeled_rows": max(0, len(labeled_rows) - len(used_label_ids)),
    }


DISTRICT_NAME_ALIASES = {
    # Historical/short spellings observed in the Nepal DesInventar export.
    "MAHOTARI": "MAHOTTARI",
    "KAVRE": "KAVREPALANCHOK",
    "SINDHUPALCHOKE": "SINDHUPALCHOK",
    "ARGHAKANCHI": "ARGHAKHANCHI",
}


def canonical_district_name(value):
    norm = normalize_text(value)
    return DISTRICT_NAME_ALIASES.get(norm, norm)


def enrich_coordinates(rows, gazetteer):
    source_counts = Counter()
    unmatched = Counter()
    results = []

    for row in rows:
        district_xml = (row.get("district_name_xml") or "").strip()
        district_norm = normalize_text(district_xml)
        district_canonical = canonical_district_name(district_xml)

        lat = row.get("xml_latitude")
        lon = row.get("xml_longitude")
        if valid_coord(lat, lon):
            row["geo_latitude"] = lat
            row["geo_longitude"] = lon
            row["geo_source"] = "desinventar_event_coordinates"
            row["geo_match_level"] = "event"
            source_counts["desinventar_event_coordinates"] += 1
        elif district_canonical and district_canonical in gazetteer:
            g = gazetteer[district_canonical]
            row["geo_latitude"] = g["latitude"]
            row["geo_longitude"] = g["longitude"]
            row["geo_source"] = "nepal_district_gazetteer"
            row["geo_match_level"] = "district_centroid"
            if district_canonical != district_norm:
                source_counts["nepal_district_gazetteer_alias"] += 1
                row["geo_resolution_note"] = "normalized historical/alternate district spelling: %s -> %s" % (district_xml, g["district_name"])
            else:
                source_counts["nepal_district_gazetteer"] += 1
        else:
            row["geo_latitude"] = ""
            row["geo_longitude"] = ""
            row["geo_source"] = "unresolved"
            row["geo_match_level"] = "none"
            if district_norm in ("NAWALPARASI", "RUKUM"):
                row["geo_resolution_note"] = "legacy district split after 2015; source export does not identify the modern east/west district, so no coordinate was forced"
            unmatched[district_xml or "<blank>"] += 1

        row["district_name_normalized"] = district_norm
        row["district_name_canonical"] = district_canonical
        row["geo_country"] = "Nepal"
        results.append(row)

    return results, source_counts, unmatched


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    # Preserve the existing labeled columns first, then append new enrichment fields.
    ordered = []
    for row in rows:
        for key in row.keys():
            if key not in ordered:
                ordered.append(key)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Resolve DesInventar Nepal administrative codes/names and geographic coordinates without public geocoding.")
    parser.add_argument("--input-zip", default="data/raw/desinventar/DI_export_npl.zip")
    parser.add_argument("--labeled", default="data/processed/desinventar_nepal_labeled.csv")
    parser.add_argument("--gazetteer", default="data/mappings/nepal_district_locations.csv")
    parser.add_argument("--output", default="data/processed/desinventar_nepal_geocoded.csv")
    parser.add_argument("--quality", default="data/processed/desinventar_nepal_geo_quality.json")
    parser.add_argument("--catalog", default="data/mappings/desinventar_nepal_admin_catalog.csv")
    args = parser.parse_args()

    print("[1/6] Reading DesInventar XML from ZIP...")
    xml_name, root = read_zip_xml(args.input_zip)
    print("      XML:", xml_name)

    print("[2/6] Extracting event + administrative fields...")
    xml_records, xml_stats = extract_records(root)
    print("      Accepted event rows:", len(xml_records))

    print("[3/6] Building administrative catalog from name1/name2...")
    catalog_rows, catalog_stats = build_admin_catalog(xml_records)
    write_csv(args.catalog, catalog_rows)
    print("      Distinct level1 codes:", catalog_stats["distinct_level1_codes"])
    print("      Distinct district codes:", catalog_stats["distinct_district_codes"])
    print("      Distinct district names:", catalog_stats["distinct_district_names"])

    print("[4/6] Loading labeled historical outcomes...")
    labeled_rows, id_col, _ = load_labeled(args.labeled)
    joined, join_stats = map_labels_to_xml(xml_records, labeled_rows, id_col)
    print("      ID column:", id_col)
    print("      Matched XML events to labeled rows:", join_stats["matched_to_labeled"])
    print("      Unmatched XML events:", join_stats["unmatched_xml_events"])

    print("[5/6] Loading local Nepal district gazetteer...")
    gazetteer, gaz_stats = load_gazetteer(args.gazetteer)
    print("      Gazetteer districts:", gaz_stats.get("gazetteer_districts", 0))

    print("[6/6] Enriching coordinates...")
    enriched, coord_sources, unmatched = enrich_coordinates(joined, gazetteer)
    write_csv(args.output, enriched)

    total = len(enriched)
    valid = sum(1 for r in enriched if valid_coord(normalize_number(r.get("geo_latitude")), normalize_number(r.get("geo_longitude"))))
    district_names = sorted({normalize_text(r.get("district_name_xml")) for r in enriched if normalize_text(r.get("district_name_xml"))})

    quality = {
        "pipeline": "Phase 5D.1 DesInventar code resolution",
        "input_zip": os.path.abspath(args.input_zip),
        "xml_file": xml_name,
        "labeled_dataset": os.path.abspath(args.labeled),
        "gazetteer": os.path.abspath(args.gazetteer),
        "output_dataset": os.path.abspath(args.output),
        "admin_catalog": os.path.abspath(args.catalog),
        "xml_stats": xml_stats,
        "join_stats": join_stats,
        "gazetteer_stats": gaz_stats,
        "coordinate_sources": dict(coord_sources),
        "rows_total": total,
        "rows_with_valid_coordinates": valid,
        "coordinate_coverage_pct": round((valid / total) * 100.0, 4) if total else 0.0,
        "distinct_desinventar_district_names": len(district_names),
        "unmatched_district_names_top50": unmatched.most_common(50),
        "notes": [
            "DesInventar name1 is treated as the district-level administrative name for this Nepal export because the observed XML hierarchy uses level1/name1 for names such as Taplejung and level2/name2 for localities.",
            "Event-level latitude/longitude are preferred when they fall inside Nepal bounds.",
            "District-centroid coordinates from the local gazetteer are used only when event coordinates are missing/invalid.",
            "No public geocoding service is used.",
            "This geographic enrichment is for historical ML data preparation and does not by itself validate future-risk predictions."
        ],
    }
    write_json(args.quality, quality)

    print("")
    print("DONE")
    print("Output:", args.output)
    print("Catalog:", args.catalog)
    print("Quality:", args.quality)
    print("Coordinate coverage: %.2f%% (%d/%d)" % (quality["coordinate_coverage_pct"], valid, total))
    print("Coordinate source counts:", dict(coord_sources))
    if unmatched:
        print("Top unresolved district names:")
        for name, count in unmatched.most_common(15):
            print("  %s -> %d" % (name, count))


if __name__ == "__main__":
    main()
