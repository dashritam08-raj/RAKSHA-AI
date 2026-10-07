# RAKSHA AI — Production ML data plan

The bundled model is synthetic and exists only to verify the software pipeline.
For production training, use representative, permissioned historical observations with timestamps and documented provenance.

## Candidate data sources

- **ECMWF / Copernicus ERA5** — global hourly reanalysis with multi-decadal coverage. Useful for historical weather features and retrospective model training.
- **Copernicus GloFAS** — historical river discharge and related hydrological data. Useful for flood and river-risk features.
- **USGS Earthquake Feeds** — programmatic GeoJSON earthquake feeds and catalog services. Useful for seismic event features.
- **NDMA SACHET** — official Indian disaster-alert source. Use only the configured/authorized feed and keep official alerts authoritative.

## Production data rules

1. Record source, retrieval time, location/grid, and license/provenance for every observation.
2. Keep an untouched future test period for temporal evaluation.
3. Define the prediction target with operational/domain experts before training.
4. Do not train on future information that would not be available at prediction time.
5. Validate geographic and seasonal generalization before activation.
6. Monitor missingness, drift, calibration/uncertainty, and performance after deployment.

The application should continue to distinguish source data from operator-reported and simulated data.

## Phase 5A real-data ingestion

The `ml/ingestion/` tools provide a reproducible collector and join step. They create raw weather observations, earthquake events, an ingestion manifest and a training-ready table after a separately supplied incident-label file.

The ingestion tools deliberately keep the target label separate from environmental predictors. The collector does not invent risk labels from the fetched weather or earthquake data.
