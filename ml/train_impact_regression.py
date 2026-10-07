from __future__ import print_function

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


def pick_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def build_features(df):
    out = pd.DataFrame(index=df.index)

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

    for c in ("geo_latitude", "geo_longitude", "latitude", "longitude"):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("disaster_type", "district", "region",
              "district_name_canonical", "geo_match_level"):
        if c in df.columns:
            out[c] = df[c].fillna("UNKNOWN").astype(str)

    numeric = [
        "temperature_mean_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_max_kmh", "wind_gust_max_kmh",
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "wind_speed_10m", "wind_gusts_10m",
        "eq_count_7d_300km", "eq_max_magnitude_7d_300km",
        "eq_nearest_km_7d_300km", "has_earthquake_context",
    ]
    for c in numeric:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
            out[c + "_missing"] = out[c].isna().astype(int)

    # Add other numeric environmental/geographic predictors while explicitly
    # excluding outcome/label leakage.
    blocked = (
        "impact", "death", "dead", "injur", "affected", "evacuat", "relocat",
        "destroy", "damage", "loss", "label", "outcome", "band", "score",
        "serial", "incident_id", "uuid", "raw_", "source_"
    )
    allowed_tokens = (
        "temp", "humid", "precip", "rain", "wind", "gust",
        "earthquake", "seismic", "eq_", "latitude", "longitude", "elevation"
    )

    for c in df.columns:
        if c in out.columns:
            continue
        lc = str(c).lower()
        if any(b in lc for b in blocked):
            continue
        if not any(a in lc for a in allowed_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = pd.to_numeric(df[c], errors="coerce")
            out[c + "_missing"] = out[c].isna().astype(int)

    # Avoid very high-cardinality categorical fields.
    keep = {}
    for c in out.columns:
        if out[c].dtype == object:
            if out[c].nunique(dropna=False) <= max(1000, int(len(out) * 0.10)):
                keep[c] = out[c]
        else:
            keep[c] = out[c]

    return pd.DataFrame(keep, index=df.index)


def make_pipeline(X, model):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    prep = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ])

    return Pipeline([
        ("prep", prep),
        ("model", model)
    ])


def score_regression(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "spearman_rank_correlation": float(
            pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        ),
    }


def bands(values, q70, q90):
    values = np.asarray(values, dtype=float)
    out = np.where(values < q70, "LOW",
                   np.where(values < q90, "MODERATE", "HIGH"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/processed/nepal_geographic_training_dataset_historical.csv"
    )
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("RAKSHA AI — Phase 5D.5 real-data impact regression")
    print("Input:", args.input)

    df = pd.read_csv(args.input, low_memory=False)
    if "observed_at" not in df.columns:
        raise RuntimeError("observed_at is required.")

    target_col = pick_col(df, ["impact_score", "observed_impact_score"])
    if not target_col:
        raise RuntimeError("impact_score / observed_impact_score was not found.")

    df["_target"] = pd.to_numeric(df[target_col], errors="coerce")
    df["_date"] = pd.to_datetime(df["observed_at"], errors="coerce")

    valid = df[df["_target"].notna() & (df["_target"] >= 0) & df["_date"].notna()].copy()
    valid = valid.sort_values("_date").reset_index(drop=True)

    print("All rows:", len(df))
    print("Rows with valid impact_score:", len(valid))
    print("Target min:", float(valid["_target"].min()))
    print("Target max:", float(valid["_target"].max()))
    print("Target mean:", round(float(valid["_target"].mean()), 4))
    print("Target median:", round(float(valid["_target"].median()), 4))
    print("\nTarget quantiles:")
    print(valid["_target"].quantile([0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).to_string())

    split = int(len(valid) * 0.80)
    train = valid.iloc[:split].copy()
    test = valid.iloc[split:].copy()

    y_train_raw = train["_target"].values
    y_test_raw = test["_target"].values

    Xall = build_features(valid)
    Xtr = Xall.iloc[:split]
    Xte = Xall.iloc[split:]

    # Learn relative-impact thresholds from training data only.
    q70 = float(np.quantile(y_train_raw, 0.70))
    q90 = float(np.quantile(y_train_raw, 0.90))

    print("\nTIME SPLIT")
    print("Train:", train["_date"].min(), "to", train["_date"].max(), "rows", len(train))
    print("Test :", test["_date"].min(), "to", test["_date"].max(), "rows", len(test))
    print("Relative band thresholds learned from TRAIN ONLY:")
    print("  MODERATE >=", q70)
    print("  HIGH     >=", q90)

    print("\nPREDICTORS:", len(Xall.columns))
    print(", ".join(Xall.columns))

    y_train_log = np.log1p(y_train_raw)

    candidates = [
        ("ExtraTreesRegressor", ExtraTreesRegressor(
            n_estimators=700,
            random_state=42,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1
        )),
        ("RandomForestRegressor", RandomForestRegressor(
            n_estimators=700,
            random_state=42,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1
        )),
    ]

    all_results = []
    fitted = {}

    for name, model in candidates:
        print("\nTraining:", name)
        pipe = make_pipeline(Xtr, model)
        pipe.fit(Xtr, y_train_log)

        pred_log = pipe.predict(Xte)
        pred_raw = np.maximum(0.0, np.expm1(pred_log))

        metrics = score_regression(y_test_raw, pred_raw)
        true_band = bands(y_test_raw, q70, q90)
        pred_band = bands(pred_raw, q70, q90)

        metrics.update({
            "model": name,
            "relative_band_accuracy": float(np.mean(true_band == pred_band)),
            "relative_band_balanced_accuracy": float(
                balanced_accuracy_score(
                    pd.Categorical(true_band, categories=["LOW", "MODERATE", "HIGH"]).codes,
                    pd.Categorical(pred_band, categories=["LOW", "MODERATE", "HIGH"]).codes
                )
            ),
            "relative_band_confusion_matrix": confusion_matrix(
                true_band,
                pred_band,
                labels=["LOW", "MODERATE", "HIGH"]
            ).tolist()
        })

        all_results.append(metrics)
        fitted[name] = pipe

        print("  MAE={:.4f}".format(metrics["mae"]))
        print("  RMSE={:.4f}".format(metrics["rmse"]))
        print("  R2={:.4f}".format(metrics["r2"]))
        print("  Spearman={:.4f}".format(metrics["spearman_rank_correlation"]))
        print("  Relative band accuracy={:.4f}".format(metrics["relative_band_accuracy"]))

    selected = sorted(
        all_results,
        key=lambda x: (
            x["spearman_rank_correlation"],
            x["r2"],
            -x["mae"]
        ),
        reverse=True
    )[0]

    selected_pipe = fitted[selected["model"]]

    model_path = os.path.join(args.model_dir, "geographic_impact_regressor.joblib")
    metrics_path = os.path.join(args.model_dir, "geographic_impact_regression_metrics.json")
    fi_path = os.path.join(args.model_dir, "geographic_impact_regression_feature_importance.csv")

    joblib.dump({
        "pipeline": selected_pipe,
        "model_name": selected["model"],
        "target": target_col,
        "target_transform": "log1p; prediction transformed with expm1",
        "relative_band_thresholds": {
            "moderate": q70,
            "high": q90,
            "critical": None
        },
        "feature_columns": list(Xall.columns),
        "train_period": [str(train["_date"].min()), str(train["_date"].max())],
        "test_period": [str(test["_date"].min()), str(test["_date"].max())],
        "warning": "Research/validation only; not a sole emergency decision authority."
    }, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": selected,
            "all_candidates": all_results,
            "rows_all": int(len(df)),
            "rows_valid_target": int(len(valid)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "relative_band_thresholds_train_only": {
                "moderate": q70,
                "high": q90
            },
            "selection_rule": "highest Spearman rank correlation, then R2, then lower MAE",
            "note": "Relative bands are historical impact magnitude bands, not future-risk probabilities.",
            "warning": "External validation, calibration, monitoring, governance and institutional approval are required for operational use."
        }, f, indent=2)

    try:
        prep = selected_pipe.named_steps["prep"]
        model = selected_pipe.named_steps["model"]
        names = prep.get_feature_names_out()
        fi = pd.DataFrame({
            "feature": names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(fi_path, index=False)
    except Exception:
        pd.DataFrame(columns=["feature", "importance"]).to_csv(fi_path, index=False)

    print("\nDONE")
    print("Selected model:", selected["model"])
    print("MAE:", round(selected["mae"], 4))
    print("RMSE:", round(selected["rmse"], 4))
    print("R2:", round(selected["r2"], 4))
    print("Spearman:", round(selected["spearman_rank_correlation"], 4))
    print("Relative band accuracy:", round(selected["relative_band_accuracy"], 4))
    print("Model:", model_path)
    print("Metrics:", metrics_path)
    print("Feature importance:", fi_path)

if __name__ == "__main__":
    main()
