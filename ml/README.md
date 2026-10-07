# RAKSHA AI ML

The ML stack contains the historical Phase 5D data-ingestion, evaluation, explainability and production-inference tooling.

## Current candidate

The FastAPI backend uses the final **RAKSHA-ML-v1 70/30 HGB ensemble** stored in:

```text
backend/models/raksha_final_shallow_hgb.joblib
backend/models/raksha_final_log_target_hgb.joblib
```

The root `ml/models/` copies are training artifacts and match the backend component models byte-for-byte.

The final model was selected using chronological holdout and rolling time-series validation:

- training: ≤ 2011
- final test: 2012–2013
- test MAE: 2.8280
- test RMSE: 4.6932
- test R²: 0.1027
- test Spearman: 0.4957

The model is a **decision-support screening model**, not a calibrated probability or field-validated emergency predictor. The final test set contained zero HIGH-impact events and moderate/high-impact events were strongly underpredicted.

## Geography

The training data is Nepal historical data. Live coordinates outside the training geography are flagged by the backend. India-specific operational use requires representative Indian disaster data and independent validation.

## Legacy artifacts

The old demo model `backend/models/risk_model.joblib` and the older root `ml/predictor.py` are retained for historical/compatibility reasons. The FastAPI runtime imports `backend/ml/predictor.py`.

Do not approve the legacy demo model merely by changing its status; production activation should follow independent review and deployment controls.
