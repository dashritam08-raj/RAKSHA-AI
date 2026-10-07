import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# PHASE 5D.16
# THRESHOLD OPTIMIZATION + ROBUST HGB TUNING
# ============================================================

print("=" * 70)
print("PHASE 5D.16 - THRESHOLD OPTIMIZATION + ROBUST HGB TUNING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/nepal_ml_features_v1.csv"

OUTPUT_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"\nLoaded rows: {len(df):,}")
print(f"Loaded columns: {len(df.columns)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "impact_score"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "latitude",
    "longitude",
    "district",
    "province",
    "disaster_type",
    "weather_match_distance_km",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "precipitation_mm",
    "wind_max_kmh",
    "wind_gust_max_kmh",
    "eq_count_7d_300km",
    "eq_max_magnitude_7d_300km",
    "eq_nearest_km_7d_300km",
    "year",
    "month",
    "day_of_year",
    "day_of_week",
    "month_sin",
    "month_cos",
    "has_coordinates",
    "has_weather",
    "has_earthquake_context",
]


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")

    for col in missing_columns:
        print(f" - {col}")

    raise ValueError(
        "Dataset does not contain all required columns."
    )


# ============================================================
# IMPACT BAND
# ============================================================

def classify_impact_band(score):

    if score < 30:
        return "LOW"

    elif score < 55:
        return "MODERATE"

    else:
        return "HIGH"


df["impact_band_analysis"] = (
    df[TARGET]
    .apply(classify_impact_band)
)

df["tail_flag"] = (
    df[TARGET] >= 30
).astype(int)


# ============================================================
# X / y
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_mask = df["year"] <= 2009

val_mask = df["year"].between(
    2010,
    2011,
)

test_mask = df["year"] >= 2012


X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[val_mask]
y_val = y.loc[val_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]


df_train = df.loc[train_mask].copy()
df_val = df.loc[val_mask].copy()
df_test = df.loc[test_mask].copy()


print("\nChronological split:")

print(
    f"Train      : {len(X_train):,} rows (<= 2009)"
)

print(
    f"Validation : {len(X_val):,} rows (2010-2011)"
)

print(
    f"Test       : {len(X_test):,} rows (>= 2012)"
)


# ============================================================
# FEATURE GROUPS
# ============================================================

CATEGORICAL_FEATURES = [
    "district",
    "province",
    "disaster_type",
]

NUMERICAL_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor():

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# HGB MODEL FACTORY
# ============================================================

def build_hgb(config):

    model = HistGradientBoostingRegressor(
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        max_leaf_nodes=config["max_leaf_nodes"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=config["l2_regularization"],
        random_state=42,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# REGRESSION METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    spearman = spearmanr(
        y_true,
        y_pred,
    ).statistic

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": spearman,
    }


# ============================================================
# TAIL METRICS
# ============================================================

def tail_regression_metrics(
    y_true,
    y_pred,
):

    tail_mask = (
        y_true >= 30
    )

    moderate_mask = (
        (y_true >= 30)
        & (y_true < 55)
    )

    high_mask = (
        y_true >= 55
    )

    result = {
        "tail_events": int(tail_mask.sum()),
        "moderate_events": int(
            moderate_mask.sum()
        ),
        "high_events": int(
            high_mask.sum()
        ),
    }

    if tail_mask.sum() > 0:

        result["tail_MAE"] = (
            mean_absolute_error(
                y_true[tail_mask],
                y_pred[tail_mask],
            )
        )

        result["tail_actual_mean"] = (
            y_true[tail_mask].mean()
        )

        result["tail_predicted_mean"] = (
            np.mean(y_pred[tail_mask])
        )

        result["tail_bias"] = (
            np.mean(
                y_pred[tail_mask]
                - y_true[tail_mask]
            )
        )

    else:

        result["tail_MAE"] = np.nan
        result["tail_actual_mean"] = np.nan
        result["tail_predicted_mean"] = np.nan
        result["tail_bias"] = np.nan

    if moderate_mask.sum() > 0:

        result["moderate_MAE"] = (
            mean_absolute_error(
                y_true[moderate_mask],
                y_pred[moderate_mask],
            )
        )

    else:

        result["moderate_MAE"] = np.nan

    if high_mask.sum() > 0:

        result["high_MAE"] = (
            mean_absolute_error(
                y_true[high_mask],
                y_pred[high_mask],
            )
        )

    else:

        result["high_MAE"] = np.nan

    return result


# ============================================================
# ROBUST HGB CONFIGURATIONS
# ============================================================

configs = {

    "Baseline HGB": {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    "Shallow HGB": {
        "max_iter": 400,
        "learning_rate": 0.03,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    "Deep HGB": {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },

    "Strong Regularization HGB": {
        "max_iter": 350,
        "learning_rate": 0.04,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 2.0,
    },

    "Low Leaf HGB": {
        "max_iter": 400,
        "learning_rate": 0.04,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 0.5,
    },

    "Slow Learning HGB": {
        "max_iter": 500,
        "learning_rate": 0.03,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.5,
    },
}


# ============================================================
# HGB TUNING
# ============================================================

print("\n" + "=" * 70)
print("ROBUST HGB CONFIGURATION SEARCH")
print("=" * 70)


tuning_rows = []

trained_models = {}


for model_name, config in configs.items():

    print(
        f"\nTraining: {model_name}"
    )

    print(
        f"  max_iter={config['max_iter']}"
    )

    print(
        f"  learning_rate={config['learning_rate']}"
    )

    print(
        f"  max_leaf_nodes={config['max_leaf_nodes']}"
    )

    print(
        f"  min_samples_leaf={config['min_samples_leaf']}"
    )

    print(
        f"  l2_regularization={config['l2_regularization']}"
    )

    model = build_hgb(config)

    model.fit(
        X_train,
        y_train,
    )

    val_pred = model.predict(
        X_val
    )

    metrics = regression_metrics(
        y_val,
        val_pred,
    )

    tail_metrics = tail_regression_metrics(
        y_val,
        val_pred,
    )

    row = {
        "model": model_name,

        "max_iter":
            config["max_iter"],

        "learning_rate":
            config["learning_rate"],

        "max_leaf_nodes":
            config["max_leaf_nodes"],

        "min_samples_leaf":
            config["min_samples_leaf"],

        "l2_regularization":
            config["l2_regularization"],

        "validation_MAE":
            metrics["MAE"],

        "validation_RMSE":
            metrics["RMSE"],

        "validation_R2":
            metrics["R2"],

        "validation_Spearman":
            metrics["Spearman"],

        "validation_tail_MAE":
            tail_metrics["tail_MAE"],

        "validation_moderate_MAE":
            tail_metrics["moderate_MAE"],

        "validation_high_MAE":
            tail_metrics["high_MAE"],
    }

    tuning_rows.append(row)

    trained_models[
        model_name
    ] = model

    print(
        f"  Validation MAE      : "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"  Validation RMSE     : "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"  Validation R2       : "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"  Validation Spearman : "
        f"{metrics['Spearman']:.4f}"
    )


tuning_results = pd.DataFrame(
    tuning_rows
)


tuning_results = (
    tuning_results
    .sort_values(
        [
            "validation_R2",
            "validation_MAE",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("HGB TUNING RESULTS")
print("=" * 70)


print(
    tuning_results
    .round(4)
    .to_string(index=False)
)


# ============================================================
# SELECT BEST HGB
# ============================================================

best_model_name = (
    tuning_results
    .iloc[0]["model"]
)

best_hgb = trained_models[
    best_model_name
]

best_config = configs[
    best_model_name
]


print("\n" + "=" * 70)
print("BEST HGB CANDIDATE")
print("=" * 70)

print(
    f"\nSelected by validation R2: "
    f"{best_model_name}"
)

print(
    f"Validation R2: "
    f"{tuning_results.iloc[0]['validation_R2']:.4f}"
)

print(
    f"Validation MAE: "
    f"{tuning_results.iloc[0]['validation_MAE']:.4f}"
)


# ============================================================
# BASELINE VS TUNED VALIDATION
# ============================================================

baseline_row = tuning_results[
    tuning_results["model"]
    == "Baseline HGB"
].iloc[0]

best_row = tuning_results.iloc[0]


print("\n" + "=" * 70)
print("BASELINE VS TUNED")
print("=" * 70)

print(
    f"\nBaseline validation R2: "
    f"{baseline_row['validation_R2']:.4f}"
)

print(
    f"Tuned validation R2   : "
    f"{best_row['validation_R2']:.4f}"
)

print(
    f"\nBaseline validation MAE: "
    f"{baseline_row['validation_MAE']:.4f}"
)

print(
    f"Tuned validation MAE   : "
    f"{best_row['validation_MAE']:.4f}"
)


# ============================================================
# TAIL CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("TAIL CLASSIFIER")
print("=" * 70)


y_train_tail = (
    y_train >= 30
).astype(int)

y_val_tail = (
    y_val >= 30
).astype(int)

y_test_tail = (
    y_test >= 30
).astype(int)


tail_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            build_preprocessor(),
        ),
        (
            "model",
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


tail_classifier.fit(
    X_train,
    y_train_tail,
)


val_tail_probability = (
    tail_classifier
    .predict_proba(X_val)[:, 1]
)

test_tail_probability = (
    tail_classifier
    .predict_proba(X_test)[:, 1]
)


print(
    f"\nTraining tail events: "
    f"{int(y_train_tail.sum()):,}"
)

print(
    f"Validation tail events: "
    f"{int(y_val_tail.sum()):,}"
)

print(
    f"Test tail events: "
    f"{int(y_test_tail.sum()):,}"
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 70)
print("TAIL THRESHOLD OPTIMIZATION")
print("=" * 70)


thresholds = np.arange(
    0.05,
    0.51,
    0.05,
)


threshold_rows = []


for threshold in thresholds:

    val_tail_class = (
        val_tail_probability
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_val_tail,
        val_tail_class,
        zero_division=0,
    )

    recall = recall_score(
        y_val_tail,
        val_tail_class,
        zero_division=0,
    )

    f1 = f1_score(
        y_val_tail,
        val_tail_class,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_val_tail,
        val_tail_class,
        beta=2,
        zero_division=0,
    )

    predicted_tail_count = int(
        val_tail_class.sum()
    )

    true_positive = int(
        (
            (val_tail_class == 1)
            & (y_val_tail == 1)
        ).sum()
    )

    false_positive = int(
        (
            (val_tail_class == 1)
            & (y_val_tail == 0)
        ).sum()
    )

    false_negative = int(
        (
            (val_tail_class == 0)
            & (y_val_tail == 1)
        ).sum()
    )

    threshold_rows.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "predicted_tail_count":
                predicted_tail_count,
            "true_positive":
                true_positive,
            "false_positive":
                false_positive,
            "false_negative":
                false_negative,
        }
    )


threshold_results = pd.DataFrame(
    threshold_rows
)


threshold_results = (
    threshold_results
    .sort_values(
        [
            "f2",
            "recall",
            "precision",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)


print("\nThreshold results:")

print(
    threshold_results
    .round(4)
    .to_string(index=False)
)


# ============================================================
# SELECT THRESHOLD
# ============================================================

best_threshold = float(
    threshold_results
    .iloc[0]["threshold"]
)


best_threshold_row = (
    threshold_results.iloc[0]
)


print("\n" + "=" * 70)
print("SELECTED TAIL THRESHOLD")
print("=" * 70)

print(
    f"\nSelected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation precision: "
    f"{best_threshold_row['precision']:.4f}"
)

print(
    f"Validation recall: "
    f"{best_threshold_row['recall']:.4f}"
)

print(
    f"Validation F1: "
    f"{best_threshold_row['f1']:.4f}"
)

print(
    f"Validation F2: "
    f"{best_threshold_row['f2']:.4f}"
)


# ============================================================
# TRAIN LOW REGRESSOR
# ============================================================

print("\n" + "=" * 70)
print("TRAINING TWO-STAGE REGRESSORS")
print("=" * 70)


low_train_mask = (
    y_train < 30
)

tail_train_mask = (
    y_train >= 30
)


X_train_low = X_train.loc[
    low_train_mask
]

y_train_low = y_train.loc[
    low_train_mask
]


X_train_tail = X_train.loc[
    tail_train_mask
]

y_train_tail_reg = y_train.loc[
    tail_train_mask
]


print(
    f"\nLOW training rows: "
    f"{len(X_train_low):,}"
)

print(
    f"TAIL training rows: "
    f"{len(X_train_tail):,}"
)


# ============================================================
# LOW REGRESSOR
# ============================================================

low_regressor = build_hgb(
    best_config
)

low_regressor.fit(
    X_train_low,
    y_train_low,
)


# ============================================================
# TAIL REGRESSOR
# ============================================================

tail_regressor = build_hgb(
    best_config
)

tail_regressor.fit(
    X_train_tail,
    y_train_tail_reg,
)


# ============================================================
# TWO-STAGE VALIDATION
# ============================================================

val_low_pred = (
    low_regressor
    .predict(X_val)
)

val_tail_value_pred = (
    tail_regressor
    .predict(X_val)
)


val_two_stage_pred = np.where(
    val_tail_probability
    >= best_threshold,

    val_tail_value_pred,

    val_low_pred,
)


val_two_stage_pred = np.maximum(
    val_two_stage_pred,
    0,
)


val_two_stage_metrics = (
    regression_metrics(
        y_val,
        val_two_stage_pred,
    )
)

val_two_stage_tail = (
    tail_regression_metrics(
        y_val,
        val_two_stage_pred,
    )
)


print("\nTwo-stage validation:")

print(
    f"MAE      : "
    f"{val_two_stage_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{val_two_stage_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{val_two_stage_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{val_two_stage_metrics['Spearman']:.4f}"
)

print(
    f"Tail MAE : "
    f"{val_two_stage_tail['tail_MAE']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Best HGB test prediction
# ------------------------------------------------------------

best_hgb_test_pred = (
    best_hgb
    .predict(X_test)
)


best_hgb_test_metrics = (
    regression_metrics(
        y_test,
        best_hgb_test_pred,
    )
)

best_hgb_test_tail = (
    tail_regression_metrics(
        y_test,
        best_hgb_test_pred,
    )
)


print("\nBest tuned HGB - Test:")

print(
    f"MAE      : "
    f"{best_hgb_test_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{best_hgb_test_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{best_hgb_test_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{best_hgb_test_metrics['Spearman']:.4f}"
)


# ------------------------------------------------------------
# Two-stage test prediction
# ------------------------------------------------------------

test_low_pred = (
    low_regressor
    .predict(X_test)
)

test_tail_value_pred = (
    tail_regressor
    .predict(X_test)
)


test_two_stage_pred = np.where(
    test_tail_probability
    >= best_threshold,

    test_tail_value_pred,

    test_low_pred,
)


test_two_stage_pred = np.maximum(
    test_two_stage_pred,
    0,
)


test_two_stage_metrics = (
    regression_metrics(
        y_test,
        test_two_stage_pred,
    )
)

test_two_stage_tail = (
    tail_regression_metrics(
        y_test,
        test_two_stage_pred,
    )
)


print("\nTwo-stage - Test:")

print(
    f"MAE      : "
    f"{test_two_stage_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{test_two_stage_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{test_two_stage_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{test_two_stage_metrics['Spearman']:.4f}"
)


# ============================================================
# FINAL TAIL CLASSIFICATION TEST
# ============================================================

test_tail_class = (
    test_tail_probability
    >= best_threshold
).astype(int)


test_precision = precision_score(
    y_test_tail,
    test_tail_class,
    zero_division=0,
)

test_recall = recall_score(
    y_test_tail,
    test_tail_class,
    zero_division=0,
)

test_f1 = f1_score(
    y_test_tail,
    test_tail_class,
    zero_division=0,
)

test_f2 = fbeta_score(
    y_test_tail,
    test_tail_class,
    beta=2,
    zero_division=0,
)


print("\n" + "=" * 70)
print("FINAL TAIL CLASSIFICATION")
print("=" * 70)


print(
    f"\nThreshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Precision : "
    f"{test_precision:.4f}"
)

print(
    f"Recall    : "
    f"{test_recall:.4f}"
)

print(
    f"F1        : "
    f"{test_f1:.4f}"
)

print(
    f"F2        : "
    f"{test_f2:.4f}"
)


print("\nTest confusion matrix:")

print(
    confusion_matrix(
        y_test_tail,
        test_tail_class,
    )
)


# ============================================================
# TAIL EVENT DETAILS
# ============================================================

print("\n" + "=" * 70)
print("TEST TAIL EVENT DETAILS")
print("=" * 70)


tail_details = df_test.copy()


tail_details[
    "tail_probability"
] = test_tail_probability


tail_details[
    "predicted_tail"
] = test_tail_class


tail_details[
    "tuned_hgb_prediction"
] = best_hgb_test_pred


tail_details[
    "two_stage_prediction"
] = test_two_stage_pred


tail_details[
    "tuned_hgb_abs_error"
] = np.abs(
    tail_details[TARGET]
    - tail_details["tuned_hgb_prediction"]
)


tail_details[
    "two_stage_abs_error"
] = np.abs(
    tail_details[TARGET]
    - tail_details["two_stage_prediction"]
)


tail_details = tail_details[
    tail_details["tail_flag"] == 1
].copy()


print(
    f"\nActual tail events: "
    f"{len(tail_details)}"
)


if len(tail_details) > 0:

    print(
        tail_details[
            [
                "year",
                "disaster_type",
                "district",
                "province",
                "impact_score",
                "impact_band_analysis",
                "tail_probability",
                "predicted_tail",
                "tuned_hgb_prediction",
                "two_stage_prediction",
                "tuned_hgb_abs_error",
                "two_stage_abs_error",
            ]
        ]
        .sort_values(
            "impact_score",
            ascending=False,
        )
        .to_string(index=False)
    )

else:

    print(
        "No MODERATE/HIGH events "
        "exist in the test set."
    )


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)


final_comparison = pd.DataFrame(
    [
        {
            "model": "Baseline HGB",
            "test_MAE": 2.9911,
            "test_RMSE": 4.5895,
            "test_R2": 0.1419,
            "test_Spearman": 0.4413,
        },

        {
            "model": "Best Tuned HGB",
            "test_MAE":
                best_hgb_test_metrics["MAE"],
            "test_RMSE":
                best_hgb_test_metrics["RMSE"],
            "test_R2":
                best_hgb_test_metrics["R2"],
            "test_Spearman":
                best_hgb_test_metrics["Spearman"],
        },

        {
            "model": "Threshold Two-Stage HGB",
            "test_MAE":
                test_two_stage_metrics["MAE"],
            "test_RMSE":
                test_two_stage_metrics["RMSE"],
            "test_R2":
                test_two_stage_metrics["R2"],
            "test_Spearman":
                test_two_stage_metrics["Spearman"],
        },
    ]
)


print(
    final_comparison
    .round(4)
    .to_string(index=False)
)


# ============================================================
# OUTPUT FILES
# ============================================================

tuning_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d16_hgb_tuning_results.csv",
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d16_threshold_results.csv",
)

tail_details_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d16_tail_test_events.csv",
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d16_test_predictions.csv",
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d16_final_model_comparison.csv",
)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d16_selected_configuration.json",
)


# ============================================================
# SAVE DATA
# ============================================================

tuning_results.to_csv(
    tuning_path,
    index=False,
)

threshold_results.to_csv(
    threshold_path,
    index=False,
)

tail_details.to_csv(
    tail_details_path,
    index=False,
)


# ------------------------------------------------------------
# Save predictions for every test event
# ------------------------------------------------------------

test_predictions = df_test.copy()

test_predictions[
    "tail_probability"
] = test_tail_probability

test_predictions[
    "predicted_tail"
] = test_tail_class

test_predictions[
    "best_tuned_hgb_prediction"
] = best_hgb_test_pred

test_predictions[
    "two_stage_prediction"
] = test_two_stage_pred

test_predictions[
    "best_tuned_hgb_abs_error"
] = np.abs(
    test_predictions[TARGET]
    - test_predictions[
        "best_tuned_hgb_prediction"
    ]
)

test_predictions[
    "two_stage_abs_error"
] = np.abs(
    test_predictions[TARGET]
    - test_predictions[
        "two_stage_prediction"
    ]
)

test_predictions.to_csv(
    prediction_path,
    index=False,
)


final_comparison.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

selected_configuration = {

    "phase": "5D.16",

    "selected_hgb_model":
        best_model_name,

    "selected_hgb_config":
        best_config,

    "selection_metric":
        "validation_R2",

    "selected_tail_threshold":
        best_threshold,

    "threshold_selection_metric":
        "validation_F2",

    "tail_definition":
        "impact_score >= 30",

    "chronological_split": {
        "train":
            "<= 2009",
        "validation":
            "2010-2011",
        "test":
            ">= 2012",
    },

    "test_results": {
        "best_tuned_hgb":
            {
                "MAE":
                    float(
                        best_hgb_test_metrics[
                            "MAE"
                        ]
                    ),
                "RMSE":
                    float(
                        best_hgb_test_metrics[
                            "RMSE"
                        ]
                    ),
                "R2":
                    float(
                        best_hgb_test_metrics[
                            "R2"
                        ]
                    ),
                "Spearman":
                    float(
                        best_hgb_test_metrics[
                            "Spearman"
                        ]
                    ),
            },

        "two_stage":
            {
                "MAE":
                    float(
                        test_two_stage_metrics[
                            "MAE"
                        ]
                    ),
                "RMSE":
                    float(
                        test_two_stage_metrics[
                            "RMSE"
                        ]
                    ),
                "R2":
                    float(
                        test_two_stage_metrics[
                            "R2"
                        ]
                    ),
                "Spearman":
                    float(
                        test_two_stage_metrics[
                            "Spearman"
                        ]
                    ),
                "tail_precision":
                    float(
                        test_precision
                    ),
                "tail_recall":
                    float(
                        test_recall
                    ),
                "tail_f1":
                    float(
                        test_f1
                    ),
                "tail_f2":
                    float(
                        test_f2
                    ),
            },
    },

    "note":
        "HIGH-impact events are absent "
        "from validation and test periods."
}


with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selected_configuration,
        f,
        indent=4,
    )


# ============================================================
# SAVE MODELS
# ============================================================

best_hgb_model_path = os.path.join(
    MODEL_DIR,
    "phase_5d16_best_tuned_hgb.joblib",
)

tail_classifier_path = os.path.join(
    MODEL_DIR,
    "phase_5d16_tail_classifier.joblib",
)

low_regressor_path = os.path.join(
    MODEL_DIR,
    "phase_5d16_low_regressor.joblib",
)

tail_regressor_path = os.path.join(
    MODEL_DIR,
    "phase_5d16_tail_regressor.joblib",
)


joblib.dump(
    best_hgb,
    best_hgb_model_path,
)

joblib.dump(
    tail_classifier,
    tail_classifier_path,
)

joblib.dump(
    low_regressor,
    low_regressor_path,
)

joblib.dump(
    tail_regressor,
    tail_regressor_path,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5D.16 COMPLETE")
print("=" * 70)


print(
    "\nSelected HGB:"
)

print(
    best_model_name
)


print(
    f"\nSelected tail threshold: "
    f"{best_threshold:.2f}"
)


print(
    "\nBest tuned HGB test:"
)

print(
    f"MAE      : "
    f"{best_hgb_test_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{best_hgb_test_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{best_hgb_test_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{best_hgb_test_metrics['Spearman']:.4f}"
)


print(
    "\nTwo-stage test:"
)

print(
    f"MAE      : "
    f"{test_two_stage_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{test_two_stage_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{test_two_stage_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{test_two_stage_metrics['Spearman']:.4f}"
)


print(
    "\nTail classifier test:"
)

print(
    f"Precision : "
    f"{test_precision:.4f}"
)

print(
    f"Recall    : "
    f"{test_recall:.4f}"
)

print(
    f"F1        : "
    f"{test_f1:.4f}"
)

print(
    f"F2        : "
    f"{test_f2:.4f}"
)


print("\nSaved data files:")

print(
    f"1. {tuning_path}"
)

print(
    f"2. {threshold_path}"
)

print(
    f"3. {tail_details_path}"
)

print(
    f"4. {prediction_path}"
)

print(
    f"5. {comparison_path}"
)

print(
    f"6. {metadata_path}"
)


print("\nSaved model files:")

print(
    f"1. {best_hgb_model_path}"
)

print(
    f"2. {tail_classifier_path}"
)

print(
    f"3. {low_regressor_path}"
)

print(
    f"4. {tail_regressor_path}"
)


print("\nImportant:")

print(
    "The test set was NOT used to select the HGB configuration "
    "or the tail threshold."
)

print(
    "HIGH-impact generalization cannot be established because "
    "there are no HIGH events in validation or test."
)


print("\nPhase 5D.16 finished successfully.")