from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from .features import ALL_FEATURES, DEFAULTS, BOUNDS, build_features

DEFAULT_MODEL_PATH = os.getenv("RAKSHA_ML_MODEL_PATH", "./models/risk_model.joblib")
ALLOW_DEMO = os.getenv("RAKSHA_ML_ALLOW_DEMO", "false").strip().lower() == "true"


def _load_bundle(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    return {"model": bundle, "status": "candidate", "metadata": {}}


def _numeric_warnings(features: Mapping[str, Any], feature_ranges: Mapping[str, Any] | None):
    warnings = []
    if not feature_ranges:
        return warnings
    for name, bounds in feature_ranges.items():
        if name not in features or features[name] is None:
            continue
        try:
            value = float(features[name])
            low, high = float(bounds[0]), float(bounds[1])
            if value < low or value > high:
                warnings.append(f"{name} is outside the training range ({low:g}–{high:g}).")
        except Exception:
            continue
    return warnings


def model_status(model_path: Optional[str] = None) -> Dict:
    path = model_path or DEFAULT_MODEL_PATH
    bundle = _load_bundle(path)
    if bundle is None:
        return {
            "mode": "rules_baseline",
            "model_available": False,
            "model_path": path,
            "status": "missing",
            "message": "No ML model is deployed; the transparent rules baseline is active.",
        }

    status = str(bundle.get("status", "candidate"))
    active = status == "validated" or (status == "demo" and ALLOW_DEMO)
    metadata = bundle.get("metadata") or {}
    return {
        "mode": "ml" if active else "rules_baseline",
        "model_available": active,
        "model_path": path,
        "status": status,
        "dataset_type": metadata.get("dataset_type"),
        "trained_at": metadata.get("trained_at"),
        "model_version": metadata.get("model_version"),
        "model_family": metadata.get("model_family"),
        "feature_profile": metadata.get("feature_profile", "core"),
        "metrics": metadata.get("metrics", {}),
        "cv_metrics": metadata.get("cv_metrics", {}),
        "feature_columns": metadata.get("feature_columns", []),
        "feature_ranges": metadata.get("feature_ranges", {}),
        "data_quality": metadata.get("data_quality", {}),
        "evaluation_method": metadata.get("evaluation_method"),
        "message": (
            "Validated ML model active."
            if status == "validated"
            else "Demo ML model active for local testing only."
            if status == "demo" and ALLOW_DEMO
            else "Model exists but is not activated; baseline remains active."
        ),
    }


def predict_risk(
    disaster_type: str,
    severity: str,
    people_affected: int,
    model_path: Optional[str] = None,
    extra_features: Optional[Mapping[str, Any]] = None,
):
    path = model_path or DEFAULT_MODEL_PATH
    bundle = _load_bundle(path)
    if bundle is None:
        return None

    status = str(bundle.get("status", "candidate"))
    if status != "validated" and not (status == "demo" and ALLOW_DEMO):
        return None

    model = bundle["model"]
    features = build_features(disaster_type, severity, people_affected, extra_features)
    metadata = bundle.get("metadata") or {}
    columns = metadata.get("feature_columns") or ALL_FEATURES
    row_df = pd.DataFrame([{k: features.get(k, DEFAULTS.get(k)) for k in columns}])

    prediction = float(model.predict(row_df)[0])
    prediction = max(0.0, min(100.0, prediction))

    interval = None
    estimator = model
    if hasattr(model, "named_steps"):
        for name in reversed(list(model.named_steps)):
            candidate = model.named_steps[name]
            if hasattr(candidate, "estimators_"):
                estimator = candidate
                break
    if hasattr(estimator, "estimators_"):
        try:
            tree_input = row_df
            if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
                tree_input = model.named_steps["preprocess"].transform(row_df)
            vals = [float(tree.predict(tree_input)[0]) for tree in estimator.estimators_]
            if vals:
                arr = np.array(vals, dtype=float)
                interval = [
                    round(float(np.percentile(arr, 10)), 1),
                    round(float(np.percentile(arr, 90)), 1),
                ]
        except Exception:
            interval = None

    ranges = metadata.get("feature_ranges") or {}
    warnings = _numeric_warnings(features, ranges)
    supplied = set(extra_features or {})
    env_columns = [c for c in columns if c not in {"disaster_type", "severity_encoded", "people_affected", "log_people_affected"}]
    supplied_env = sum(1 for c in env_columns if c in supplied and extra_features.get(c) is not None)
    completeness = 100 if not env_columns else round((supplied_env / len(env_columns)) * 100, 1)

    return {
        "risk_score": int(round(prediction)),
        "risk_level": "CRITICAL" if prediction >= 85 else "HIGH" if prediction >= 70 else "MODERATE" if prediction >= 40 else "LOW",
        "source": "ml",
        "prediction_interval": interval,
        "model_status": status,
        "model_version": metadata.get("model_version"),
        "model_family": metadata.get("model_family"),
        "feature_profile": metadata.get("feature_profile", "core"),
        "model_metrics": metadata.get("metrics", {}),
        "data_quality": metadata.get("data_quality", {}),
        "feature_completeness_pct": completeness,
        "warnings": warnings,
    }
