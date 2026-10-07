from __future__ import print_function

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


def pick_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def norm_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    return s if s in ("LOW", "MODERATE", "HIGH", "CRITICAL") else None


def build_features(df):
    out = pd.DataFrame(index=df.index)

    # Date/time features
    if "observed_at" in df.columns:
        dt = pd.to_datetime(df["observed_at"], errors="coerce")
        out["observed_year"] = dt.dt.year
        out["observed_month"] = dt.dt.month
        out["observed_day"] = dt.dt.day
        out["observed_dayofyear"] = dt.dt.dayofyear
        out["observed_dayofweek"] = dt.dt.dayofweek
        out["observed_weekofyear"] = dt.dt.isocalendar().week.astype(float).values
        doy = out["observed_dayofyear"].fillna(1.0)
        out["season_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        out["season_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    # Geographic fields
    for c in ("geo_latitude", "geo_longitude", "latitude", "longitude"):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    # Stable categorical context
    for c in ("disaster_type", "district", "region",
              "district_name_canonical", "geo_match_level"):
        if c in df.columns:
            out[c] = df[c].fillna("UNKNOWN").astype(str)

    # Known environmental columns by exact/common names
    explicit_numeric = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "wind_speed_10m", "wind_gusts_10m",
        "weather_temperature_2m", "weather_relative_humidity_2m",
        "weather_precipitation", "weather_wind_speed_10m",
        "weather_wind_gusts_10m",
        "eq_count_7d_300km", "eq_max_magnitude_7d_300km",
        "eq_nearest_km_7d_300km",
        "has_earthquake_context",
    ]
    for c in explicit_numeric:
        if c in df.columns and c not in out.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
            out[c + "_missing"] = out[c].isna().astype(int)

    # Add additional numeric environmental columns without pulling outcome/label fields.
    blocked_tokens = (
        "impact", "death", "dead", "injur", "affected", "evacuat", "relocat",
        "destroy", "damage", "loss", "label", "outcome", "band", "score",
        "missing", "serial", "id", "uuid", "source", "raw"
    )
    environmental_tokens = (
        "temp", "humid", "precip", "rain", "wind", "gust",
        "earthquake", "seismic", "eq_", "pressure", "snow",
        "latitude", "longitude", "elevation"
    )

    for c in df.columns:
        if c in out.columns:
            continue
        lc = str(c).lower()
        if any(t in lc for t in blocked_tokens):
            continue
        if not any(t in lc for t in environmental_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = pd.to_numeric(df[c], errors="coerce")
            out[c + "_missing"] = out[c].isna().astype(int)

    # Remove obvious high-cardinality IDs accidentally included.
    keep = {}
    for c in out.columns:
        if out[c].dtype == object:
            nunique = out[c].nunique(dropna=False)
            if nunique <= max(1000, int(len(out) * 0.10)):
                keep[c] = out[c]
        else:
            keep[c] = out[c]

    return pd.DataFrame(keep, index=df.index)


def make_pipeline(X, model):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    prep = ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="drop",
    )
    return Pipeline([("prep", prep), ("model", model)])


def eval_binary(name, pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    try:
        prob = pipe.predict_proba(Xte)[:, 1]
        roc = float(roc_auc_score(yte, prob)) if len(np.unique(yte)) == 2 else None
        pr = float(average_precision_score(yte, prob)) if len(np.unique(yte)) == 2 else None
    except Exception:
        prob = None
        roc = None
        pr = None

    result = {
        "model": name,
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "positive_precision": float(precision_score(yte, pred, zero_division=0)),
        "positive_recall": float(recall_score(yte, pred, zero_division=0)),
        "positive_f1": float(f1_score(yte, pred, zero_division=0)),
        "predicted_positive_rate": float(np.mean(pred)),
        "actual_positive_rate": float(np.mean(yte)),
        "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
        "roc_auc": roc,
        "average_precision_pr_auc": pr,
    }
    return pipe, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/nepal_geographic_training_dataset_historical.csv")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("RAKSHA AI — Phase 5D.4 imbalanced real-data ML")
    print("Input:", args.input)

    df = pd.read_csv(args.input, low_memory=False)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    label_col = pick_col(df, ["observed_impact_band", "impact_band", "label"])
    if not label_col:
        raise RuntimeError("No impact-band label column found.")

    df["_label"] = df[label_col].map(norm_label)
    df["_date"] = pd.to_datetime(df["observed_at"], errors="coerce")

    print("\nLABEL COUNTS")
    print(df["_label"].value_counts(dropna=False).to_string())

    print("\nPOTENTIAL FEATURE COLUMNS")
    for c in df.columns:
        lc = str(c).lower()
        if any(t in lc for t in ("temp", "humid", "precip", "rain", "wind",
                                 "earthquake", "seismic", "eq_", "latitude", "longitude")):
            print(" ", c)

    work = df[df["_label"].notna() & df["_date"].notna()].copy()
    work = work.sort_values("_date").reset_index(drop=True)

    split = int(len(work) * 0.80)
    train = work.iloc[:split].copy()
    test = work.iloc[split:].copy()

    print("\nTIME SPLIT")
    print("Train:", train["_date"].min(), "to", train["_date"].max(), "rows", len(train))
    print("Test :", test["_date"].min(), "to", test["_date"].max(), "rows", len(test))

    # Binary target: MODERATE/HIGH/CRITICAL = 1, LOW = 0.
    train_y = (train["_label"] != "LOW").astype(int)
    test_y = (test["_label"] != "LOW").astype(int)

    Xall = build_features(work)
    Xtr = Xall.iloc[:split].copy()
    Xte = Xall.iloc[split:].copy()

    print("\nPREDICTORS:", len(Xall.columns))
    print(", ".join(Xall.columns))

    candidates = [
        ("ExtraTreesBinary", ExtraTreesClassifier(
            n_estimators=700, random_state=42, class_weight="balanced",
            min_samples_leaf=2, max_features="sqrt", n_jobs=-1
        )),
        ("RandomForestBinary", RandomForestClassifier(
            n_estimators=700, random_state=42, class_weight="balanced_subsample",
            min_samples_leaf=2, max_features="sqrt", n_jobs=-1
        )),
    ]

    results = []
    fitted = {}
    for name, model in candidates:
        print("\nTraining:", name)
        pipe = make_pipeline(Xtr, model)
        pipe, metrics = eval_binary(name, pipe, Xtr, train_y, Xte, test_y)
        fitted[name] = pipe
        results.append(metrics)
        print("  accuracy={:.4f}".format(metrics["accuracy"]))
        print("  balanced_accuracy={:.4f}".format(metrics["balanced_accuracy"]))
        print("  positive_precision={:.4f}".format(metrics["positive_precision"]))
        print("  positive_recall={:.4f}".format(metrics["positive_recall"]))
        print("  positive_f1={:.4f}".format(metrics["positive_f1"]))
        print("  average_precision_pr_auc={}".format(
            None if metrics["average_precision_pr_auc"] is None else round(metrics["average_precision_pr_auc"], 4)
        ))

    # Select by positive F1, then PR-AUC, then balanced accuracy.
    selected = sorted(
        results,
        key=lambda x: (
            x["positive_f1"],
            -1 if x["average_precision_pr_auc"] is None else x["average_precision_pr_auc"],
            x["balanced_accuracy"],
        ),
        reverse=True,
    )[0]
    selected_pipe = fitted[selected["model"]]

    model_path = os.path.join(args.model_dir, "geographic_elevated_risk_model.joblib")
    metrics_path = os.path.join(args.model_dir, "geographic_elevated_risk_metrics.json")
    joblib.dump(
        {
            "pipeline": selected_pipe,
            "model_name": selected["model"],
            "target_definition": "1=MODERATE/HIGH/CRITICAL, 0=LOW",
            "feature_columns": list(Xall.columns),
            "train_period": [str(train["_date"].min()), str(train["_date"].max())],
            "test_period": [str(test["_date"].min()), str(test["_date"].max())],
            "warning": "Research/validation only; not a sole emergency decision authority.",
        },
        model_path,
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": selected,
            "all_candidates": results,
            "label_counts": {str(k): int(v) for k, v in work["_label"].value_counts().items()},
            "time_split": 0.80,
            "note": "Binary target intentionally combines MODERATE/HIGH/CRITICAL because HIGH has only 8 historical examples.",
            "warning": "Metrics are not a forecast guarantee and require external validation/calibration.",
        }, f, indent=2)

    print("\nDONE")
    print("Selected model:", selected["model"])
    print("Positive F1:", round(selected["positive_f1"], 4))
    print("Positive recall:", round(selected["positive_recall"], 4))
    print("Positive precision:", round(selected["positive_precision"], 4))
    print("PR-AUC:", None if selected["average_precision_pr_auc"] is None else round(selected["average_precision_pr_auc"], 4))
    print("Balanced accuracy:", round(selected["balanced_accuracy"], 4))
    print("Model:", model_path)
    print("Metrics:", metrics_path)
    print("\nThe 98.58% Phase 5D.3 accuracy must NOT be used as the performance headline.")

if __name__ == "__main__":
    main()
