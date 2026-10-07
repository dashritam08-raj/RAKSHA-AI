# Phase 5D — Local Geographic Matching

The previous optional Nominatim workflow could require thousands of sequential requests. This phase adds a local Nepal district gazetteer instead.

The gazetteer contains 77 Nepal district reference points with WGS84 coordinates and province names. Source: **Nepal Administrative Divisions Dataset (CC-BY-4.0)**, which documents 7 provinces, 77 districts, and coordinates for administrative levels.

Source: https://github.com/open-admin-data/nepal-administrative-divisions

These are **district reference points**, not exact incident coordinates. They are appropriate for first-pass historical environmental joins when event-level coordinates are unavailable, but should remain explicitly marked as district-level approximations.

## Workflow

```powershell
python -m ingestion.build_event_location_mapping --input data\processed\desinventar_nepal_labeled.csv --output data\mappings\nepal_event_location_mapping.csv
python -m ingestion.build_local_geo_mapping --input data\mappings\nepal_event_location_mapping.csv --gazetteer data\gazetteers\nepal_districts.csv --output data\mappings\nepal_event_location_mapping_local.csv --locations-output data\mappings\nepal_district_locations.csv
```

Then inspect the output counts. Only the approved local-gazetteer rows will be placed in `nepal_district_locations.csv` for weather collection. Ambiguous and unmatched historical rows stay review-required.

The existing `geocode_mapping.py` remains available as an optional one-time helper, but it is no longer necessary for the standard Nepal pipeline.
