from __future__ import print_function

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore")


def pick_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_label(value):
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    return s if s in ("LOW", "MODERATE", "HIGH", "CRITICAL") else None


def build_features(df):
    out = pd.DataFrame(index=df.index)
    for c in ("geo_latitude", "geo_longitude"):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    dt = pd.to_datetime(df["observed_at"], errors="coerce")
    out["observed_year"] = dt.dt.year
    out["observed_month"] = dt.dt.month
    out["observed_dayofyear"] = dt.dt.dayofyear
    out["observed_dayofweek"] = dt.dt.dayofweek
    out["observed_weekofyear"] = dt.dt.isocalendar().week.astype(float).values
    doy = out["observed_dayofyear"].fillna(1.0)
    out["observed_year_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["observed_year_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    for c in ("disaster_type", "region", "district", "district_name_canonical", "geo_match_level"):
        if c in df.columns:
            out[c] = df[c].fillna("UNKNOWN").astype(str)

    known_numeric = (
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "wind_speed_10m", "wind_gusts_10m", "weather_temperature_2m",
        "weather_relative_humidity_2m", "weather_precipitation",
        "weather_wind_speed_10m", "weather_wind_gusts_10m",
    )
    for c in known_numeric:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
            out[c + "_missing"] = out[c].isna().astype(int)

    for c in df.columns:
        lc = c.lower()
        if any(t in lc for t in ("earthquake", "eq_", "seismic")) and c not in out.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                out[c] = pd.to_numeric(df[c], errors="coerce")
                out[c + "_missing"] = out[c].isna().astype(int)

    safe = {}
    for c in out.columns:
        if out[c].dtype == object and out[c].astype(str).str.len().mean() > 120:
            continue
        safe[c] = out[c]
    return pd.DataFrame(safe, index=df.index)


def build_pipeline(X, model):
    # Pandas StringDtype / nullable dtypes are not always `object`.
    # Classify columns by actual numeric-ness so values such as FIRE never
    # enter the median numeric imputer.
    numeric_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def evaluate(name, pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    labels = sorted(pd.unique(pd.concat([y_test.reset_index(drop=True), pd.Series(pred)]).dropna()).tolist())
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "classification_report": classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
    }
    yb = (pd.Series(y_test).astype(str) != "LOW").astype(int)
    pb = (pd.Series(pred).astype(str) != "LOW").astype(int)
    metrics["binary_elevated"] = {
        "definition": "MODERATE/HIGH/CRITICAL vs LOW",
        "precision": float(precision_score(yb, pb, zero_division=0)),
        "recall": float(recall_score(yb, pb, zero_division=0)),
        "f1": float(f1_score(yb, pb, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yb, pb)),
    }
    return pipe, metrics


def feature_importance_rows(pipe):
    try:
        names = pipe.named_steps["prep"].get_feature_names_out()
        vals = pipe.named_steps["model"].feature_importances_
        return pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/nepal_geographic_training_dataset_historical.csv")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    print("RAKSHA AI — Phase 5D.3 real-data geographic ML training")
    print("Input:", args.input)
    df = pd.read_csv(args.input, low_memory=False)
    label_col = pick_col(df, ["observed_impact_band", "impact_band", "label"])
    if not label_col or "observed_at" not in df.columns:
        raise RuntimeError("Required label or observed_at column is missing.")

    df["_label"] = df[label_col].map(normalize_label)
    df["_date"] = pd.to_datetime(df["observed_at"], errors="coerce")
    before = len(df)
    df = df[df["_label"].notna() & df["_date"].notna()].sort_values("_date").reset_index(drop=True)
    print("All rows:", before)
    print("Labeled/dated rows:", len(df))
    print("Label counts:")
    print(df["_label"].value_counts().to_string())

    split = int(len(df) * 0.80)
    if split <= 0 or split >= len(df):
        raise RuntimeError("Dataset is too small for an 80/20 chronological split.")
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    print("Train period:", train["_date"].min(), "to", train["_date"].max())
    print("Test period :", test["_date"].min(), "to", test["_date"].max())

    Xall = build_features(df)
    Xtr, Xte = Xall.iloc[:split], Xall.iloc[split:]
    ytr, yte = train["_label"], test["_label"]
    print("Predictor columns:", len(Xall.columns))
    print("Predictors:", ", ".join(Xall.columns))

    candidates = [
        ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced", min_samples_leaf=2, max_features="sqrt", n_jobs=-1)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", n_jobs=-1)),
    ]
    results, fitted = [], {}
    for name, model in candidates:
        print("\nTraining:", name)
        try:
            p, m = evaluate(name, build_pipeline(Xtr, model), Xtr, ytr, Xte, yte)
            results.append(m); fitted[name] = p
            print("  accuracy={:.4f} balanced_accuracy={:.4f} macro_f1={:.4f}".format(m["accuracy"], m["balanced_accuracy"], m["macro_f1"]))
        except ValueError as exc:
            print("  FAILED:", exc)

    if not results:
        raise RuntimeError("All candidate models failed.")
    selected = sorted(results, key=lambda x: (x["macro_f1"], x["balanced_accuracy"]), reverse=True)[0]
    selected_name = selected["model"]
    model_path = os.path.join(args.model_dir, "geographic_risk_model.joblib")
    metrics_path = os.path.join(args.model_dir, "geographic_risk_metrics.json")
    fi_path = os.path.join(args.model_dir, "geographic_feature_importance.csv")
    summary_path = os.path.join(args.model_dir, "geographic_training_summary.json")

    joblib.dump({
        "pipeline": fitted[selected_name],
        "model_name": selected_name,
        "label_column": label_col,
        "classes": selected["labels"],
        "feature_columns": list(Xall.columns),
        "train_start": str(train["_date"].min()), "train_end": str(train["_date"].max()),
        "test_start": str(test["_date"].min()), "test_end": str(test["_date"].max()),
        "purpose": "Historical disaster impact-band decision-support research model",
    }, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"selected_model": selected, "all_candidates": results, "dataset_rows_all": int(before), "dataset_rows_labeled": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)), "time_split_fraction": 0.80, "note": "Metrics come from a chronological holdout, not a random split."}, f, indent=2)

    feature_importance_rows(fitted[selected_name]).to_csv(fi_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "historical_rows": int(before), "labeled_rows": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
            "class_counts": {k: int(v) for k, v in df["_label"].value_counts().items()},
            "selected_model": selected_name,
            "selection_rule": "highest chronological-holdout macro-F1, then balanced accuracy",
            "leakage_protection": "observed impact/outcome fields are not added to build_features",
        }, f, indent=2)

    print("\nDONE")
    print("Selected model:", selected_name)
    print("Holdout accuracy:", round(selected["accuracy"], 4))
    print("Holdout balanced accuracy:", round(selected["balanced_accuracy"], 4))
    print("Holdout macro-F1:", round(selected["macro_f1"], 4))
    print("Binary elevated F1:", round(selected["binary_elevated"]["f1"], 4))
    print("Model:", model_path)
    print("Metrics:", metrics_path)
    print("Feature importance:", fi_path)
    print("Summary:", summary_path)
    print("WARNING: research/validation model; not a sole emergency decision authority.")


if __name__ == "__main__":
    main()
