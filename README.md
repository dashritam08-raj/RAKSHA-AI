# RAKSHA AI — Disaster Intelligence & Emergency Response

RAKSHA is a disaster command platform combining authentication, role-based operations, live GPS tracking, environmental intelligence, multi-source fusion, alerts, Emergency SOS, audit logging and an explicit ML decision-support layer.

## Current architecture

```text
GPS / SOS / Incidents / Weather / Earthquakes / Official Alerts
                         ↓
                 Multi-Source Fusion
                         ↓
              Operational Risk Index
                         +
               Historical ML Screening
                         ↓
              Command Center / Live Map
```

## ML status

The current bundled ML candidate is the **RAKSHA-ML-v1 70/30 HGB ensemble**:

- 70% Log-Target HistGradientBoosting
- 30% Shallow HistGradientBoosting
- 24 pre-event/contextual features
- Chronological training period: ≤ 2011
- Final historical test period: 2012–2013
- Test MAE: 2.8280
- Test RMSE: 4.6932
- Test R²: 0.1027
- Test Spearman: 0.4957

These are historical validation results, **not calibrated disaster probabilities and not field validation**. The final test set contained zero HIGH-impact events, so HIGH-impact generalization was not established.

### Geography limitation

The training data is Nepal-only (historical coordinates approximately latitude 26.584–30.026 and longitude 80.283–87.920). The frontend currently uses a Kolkata monitoring point for demonstration, which is outside that historical geography. RAKSHA now exposes an explicit out-of-domain warning during ML inference. For Indian operational deployment, retrain and validate on representative India-specific data before relying on the ML score operationally.

## Run

Backend:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Runtime integrations

- Open-Meteo: live weather
- USGS Earthquake Catalog: 7-day / 300-km ML earthquake context
- NDMA SACHET: optional official alert feed
- SQLite for local development; PostgreSQL configuration is supported

## Development notes

The legacy `ml/predictor.py` and `backend/models/risk_model.joblib` are retained for compatibility/history, but the FastAPI backend uses `backend/ml/predictor.py` and the two final component models in `backend/models/`.
