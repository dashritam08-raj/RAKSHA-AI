"""Train and evaluate the RAKSHA production-oriented risk model.

Usage:
    python train_risk_model.py input.csv output.joblib

The script creates a candidate bundle. Synthetic data is explicitly marked as
'demo'; only validated bundles should be activated in operational deployments.
For production training, provide a timestamped dataset with representative,
permissioned data and keep the final test period untouched.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import ALL_FEATURES, CORE_FEATURES, ENVIRONMENTAL_FEATURES

REQUIRED = {"disaster_type", "severity_encoded", "people_affected", "target_risk_score"}


def risk_band(score):
    if score >= 85: return "Critical"
    if score >= 70: return "High"
    if score >= 40: return "Moderate"
    return "Low"


def dataset_quality(df: pd.DataFrame) -> dict:
    numeric_cols = [c for c in ALL_FEATURES if c in df.columns]
    missing_pct = float(df[numeric_cols].isna().mean().mean() * 100) if numeric_cols else 0.0
    duplicate_pct = float(df.duplicated().mean() * 100)
    target = pd.to_numeric(df["target_risk_score"], errors="coerce")
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cell_pct_across_features": round(missing_pct, 3),
        "duplicate_row_pct": round(duplicate_pct, 3),
        "target_min": round(float(target.min()), 3),
        "target_max": round(float(target.max()), 3),
        "target_mean": round(float(target.mean()), 3),
    }


def build_pipeline(model):
    categorical = ["disaster_type"]
    numeric = [c for c in ALL_FEATURES if c != "disaster_type"]
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python train_risk_model.py input.csv output.joblib")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    df = pd.read_csv(input_path)

    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(sorted(missing)))
    if len(df) < 100:
        raise SystemExit("Need at least 100 labeled rows for a serious candidate training run.")

    df = df.copy()
    df["disaster_type"] = df["disaster_type"].astype(str).str.strip().str.lower()
    for col in ["severity_encoded", "people_affected", "target_risk_score"] + [c for c in ENVIRONMENTAL_FEATURES if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log_people_affected"] = np.log1p(df["people_affected"].clip(lower=0))
    df["target_risk_score"] = df["target_risk_score"].clip(0, 100)
    for col in ALL_FEATURES:
        if col not in df.columns and col != "disaster_type":
            df[col] = np.nan
    df = df.dropna(subset=list(REQUIRED))
    if len(df) < 100:
        raise SystemExit("Fewer than 100 valid labeled rows remain after cleaning.")

    feature_profile = "enriched" if all(c in df.columns for c in ENVIRONMENTAL_FEATURES) else "core"
    feature_columns = ALL_FEATURES if feature_profile == "enriched" else CORE_FEATURES

    X = df[feature_columns]
    y = df["target_risk_score"]

    if "observed_at" in df.columns:
        timestamps = pd.to_datetime(df["observed_at"], errors="coerce", utc=True)
    else:
        timestamps = pd.Series(pd.NaT, index=df.index)
    temporal = timestamps.notna().sum() == len(df)
    if temporal:
        df = df.assign(_observed_at=timestamps).sort_values("_observed_at")
        X = df[feature_columns]
        y = df["target_risk_score"]
        split = max(1, int(len(df) * 0.8))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        evaluation_method = "time_ordered_holdout_plus_timeseries_cv"
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        evaluation_method = "random_holdout_fallback; add observed_at for production temporal validation"

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=-1
        ),
    }

    cv = TimeSeriesSplit(n_splits=4) if temporal and len(X_train) >= 50 else None
    cv_results = {}
    fitted_candidates = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        if cv is not None:
            mae_scores = -cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
            r2_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            cv_results[name] = {
                "mae_mean": round(float(mae_scores.mean()), 4),
                "mae_std": round(float(mae_scores.std()), 4),
                "r2_mean": round(float(r2_scores.mean()), 4),
                "r2_std": round(float(r2_scores.std()), 4),
            }
        else:
            pipeline.fit(X_train, y_train)
            pred_cv = pipeline.predict(X_test)
            cv_results[name] = {"mae_holdout": round(float(mean_absolute_error(y_test, pred_cv)), 4), "r2_holdout": round(float(r2_score(y_test, pred_cv)), 4)}
        fitted_candidates[name] = pipeline

    if cv_results and cv is not None:
        best_name = min(cv_results, key=lambda n: cv_results[n]["mae_mean"])
    else:
        best_name = min(cv_results, key=lambda n: cv_results[n]["mae_holdout"])

    best_model = fitted_candidates[best_name]
    best_model.fit(X_train, y_train)
    pred = np.clip(best_model.predict(X_test), 0, 100)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    band_acc = float(accuracy_score([risk_band(v) for v in y_test], [risk_band(v) for v in pred]))

    ranges = {}
    for col in feature_columns:
        if col == "disaster_type":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            ranges[col] = [float(series.min()), float(series.max())]

    dataset_type = "real"
    if "dataset_type" in df.columns:
        value = str(df["dataset_type"].iloc[0]).strip().lower()
        dataset_type = value if value else "real"

    file_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    model_version = "raksha-risk-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "model_family": best_name,
        "dataset_path": str(input_path),
        "dataset_type": dataset_type,
        "dataset_sha256": file_hash,
        "feature_profile": feature_profile,
        "rows_used": int(len(df)),
        "feature_columns": feature_columns,
        "feature_ranges": ranges,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "risk_band_accuracy": round(band_acc, 4),
        },
        "cv_metrics": cv_results,
        "data_quality": dataset_quality(df),
        "evaluation_method": evaluation_method,
        "validation_note": "Candidate model; production activation requires human/data validation and representative real data.",
    }
    bundle = {
        "model": best_model,
        "metadata": metadata,
        "status": "demo" if dataset_type == "synthetic" else "candidate",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print(json.dumps({"saved_to": str(output_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
