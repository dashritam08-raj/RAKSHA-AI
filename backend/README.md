# RAKSHA AI Backend — Final Hardened Edition

FastAPI + SQLite backend for RAKSHA AI, including authentication, role-based permissions, incident intelligence, resources, alerts, Emergency SOS, and WebSocket updates.

## Run

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

## Demo accounts

```text
Admin
Username: admin
Password: Admin@123

Dispatcher
Username: dispatcher
Password: Dispatcher@123

Responder
Username: responder
Password: Responder@123
```

## Security hardening included

- PBKDF2-HMAC-SHA256 password hashing using Python standard library.
- Signed, expiring access tokens.
- Disabled/deleted accounts are rejected even when an old token is still unexpired.
- Protected read and write API routes.
- Admin-only user management.
- Stronger password validation for newly created/reset passwords.
- Generic 500 error messages for incident/SOS failures so internal exception details are not exposed.
- Basic security response headers (`nosniff`, `DENY`, strict referrer policy).
- WebSocket authentication remains enabled.

Set `RAKSHA_SECRET_KEY` to a strong random value before any public deployment.

## API

- `POST /api/auth/login`
- `GET /api/auth/me`
- Existing incident/resource/alert/SOS/statistics APIs require `Authorization: Bearer <token>`.
- WebSocket: `ws://127.0.0.1:8000/ws?token=<token>`

The SQLite database file `raksha.db` is created automatically when the backend starts. A clean release package does not include a database snapshot.

## Swagger authentication

The API uses FastAPI HTTP Bearer security. Open `/docs`, click **Authorize**, and paste the bearer token (or use the lock control on protected endpoints). Protected requests will then include `Authorization: Bearer <token>` automatically.

## ML endpoints

- `GET /api/ml/status` — authenticated model status
- `POST /api/ml/predict` — Admin/Dispatcher prediction
- `GET /api/ml/fused-risk?latitude=<lat>&longitude=<lon>` — live environment + fusion + ML screening at a monitoring point
- `GET /api/ml/model-card` — model scope, controls and limitations

Example prediction body:

```json
{
  "disaster_type": "Flood",
  "severity": "Critical",
  "people_affected": 250
}
```


### Environmental sources
Set `SACHET_CAP_URL` only when you have an authorized/public CAP feed URL. RAKSHA uses ETag-aware caching for this source. Open-Meteo weather and USGS earthquake data are fetched server-side.


### Phase 4 fusion APIs
- `GET /api/fusion/overview?latitude=<lat>&longitude=<lon>` — combines incident, environmental, official-alert and resource signals.
- `GET /api/fusion/history?latitude=<lat>&longitude=<lon>&hours=24` — returns stored risk-index snapshots for the selected monitoring grid.

The fusion score is an operational index for decision support, not a disaster probability.


## Current ML applicability

The runtime uses the historical RAKSHA-ML-v1 70/30 HGB ensemble. Its training data is Nepal historical data through 2013. The backend therefore surfaces temporal and geographic out-of-domain warnings. This is not a substitute for India-specific retraining and field validation.
