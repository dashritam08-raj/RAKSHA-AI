# RAKSHA AI — Phase 5B: Historical Disaster Outcomes

This phase converts reviewed disaster-loss records into a normalized event table for later joining with RAKSHA environmental observations.

## Source
DesInventar is a disaster information-management system. Its current site lists downloadable country/region databases and notes that posted databases are contributed by governments/institutions. Its documentation supports XML export. An IFRC GO data extension documents the `DI_export_{country_code}.zip` export pattern and the XML `<fichas><TR>` event structure.

## Workflow
1. Obtain an authorized/public DesInventar export.
2. Store it under `ml/data/raw/desinventar/` without committing it unless its source terms permit redistribution.
3. Optionally download with `ingestion.download_desinventar`.
4. Normalize events with `ingestion.import_desinventar`.
5. Review geography in `data/event_location_mapping_template.csv`. Coordinates are never invented by RAKSHA; the Assam row is a placeholder and must be verified.
6. Create transparent observed-impact labels with `ingestion.build_incident_outcomes`.
7. Inspect class balance, missingness, duplicates, time coverage and pre-event environmental coverage before any ML training.

## Example — Nepal
```powershell
cd C:\Users\Acer\Downloads\RAKSHA_AI_Ultimate\RAKSHA_AI_Ultimate\ml
python -m ingestion.download_desinventar --country-code npl --output-dir data\raw\desinventar
python -m ingestion.import_desinventar --input data\raw\desinventar\DI_export_npl.zip --output data\processed\desinventar_nepal_events.csv
python -m ingestion.build_incident_outcomes --input data\processed\desinventar_nepal_events.csv --output data\processed\desinventar_nepal_labeled.csv --policy ingestion\label_policy.json
```

The label is an **observed impact band**, not a future-risk probability. Thresholds are an initial policy and require domain review.

## Phase 5C validation
Before ML training, run the dataset validator and build the observed-impact target. Relative bands are computed from the empirical distribution of labeled events and are not universal operational risk thresholds.

```powershell
python -m ingestion.validate_incident_dataset --input data\processed\desinventar_nepal_events.csv --output data\processed\desinventar_nepal_quality.json
python -m ingestion.build_incident_outcomes --input data\processed\desinventar_nepal_events.csv --output data\processed\desinventar_nepal_labeled.csv --policy ingestion\label_policy.json
```
