import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# PHASE 5D.17
# ROLLING TIME-SERIES VALIDATION + MODEL STABILITY
#
# IMPORTANT:
# 2012-2013 remains completely untouched as final test data.
#
# Rolling validation uses only data <= 2011.
# ============================================================

print("=" * 70)
print("PHASE 5D.17 - ROLLING TIME-SERIES VALIDATION + MODEL STABILITY")
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


# ============================================================
# X / y
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


# ============================================================
# FINAL TEST
# ============================================================

# DO NOT use these rows during rolling model selection.

final_test_mask = df["year"] >= 2012

X_final_test = X.loc[
    final_test_mask
]

y_final_test = y.loc[
    final_test_mask
]

df_final_test = df.loc[
    final_test_mask
].copy()


print("\nFinal untouched test period:")
print(
    f"2012-2013 rows: "
    f"{len(X_final_test):,}"
)


# ============================================================
# PRE-TEST DATA
# ============================================================

pre_test_mask = (
    df["year"] <= 2011
)

df_pre_test = df.loc[
    pre_test_mask
].copy()

X_pre_test = X.loc[
    pre_test_mask
]

y_pre_test = y.loc[
    pre_test_mask
]


print(
    f"Pre-test rows <= 2011: "
    f"{len(X_pre_test):,}"
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
# PREPROCESSOR FACTORY
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
# MODEL CONFIGURATIONS
# ============================================================

MODEL_CONFIGS = {

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

    "Log Target HGB": {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
}


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(
    model_name,
):

    config = MODEL_CONFIGS[
        model_name
    ]

    model = HistGradientBoostingRegressor(
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        max_leaf_nodes=config["max_leaf_nodes"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=config["l2_regularization"],
        random_state=42,
    )

    pipeline = Pipeline(
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

    return pipeline


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
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

    if pd.isna(spearman):
        spearman = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": spearman,
    }


# ============================================================
# TAIL METRICS
# ============================================================

def calculate_tail_metrics(
    y_true,
    y_pred,
):

    y_true_array = np.asarray(
        y_true
    )

    y_pred_array = np.asarray(
        y_pred
    )

    tail_mask = (
        y_true_array >= 30
    )

    moderate_mask = (
        (y_true_array >= 30)
        & (y_true_array < 55)
    )

    high_mask = (
        y_true_array >= 55
    )

    result = {
        "tail_events": int(
            tail_mask.sum()
        ),

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
                y_true_array[tail_mask],
                y_pred_array[tail_mask],
            )
        )

        result["tail_bias"] = (
            np.mean(
                y_pred_array[tail_mask]
                - y_true_array[tail_mask]
            )
        )

    else:

        result["tail_MAE"] = np.nan
        result["tail_bias"] = np.nan

    if moderate_mask.sum() > 0:

        result["moderate_MAE"] = (
            mean_absolute_error(
                y_true_array[moderate_mask],
                y_pred_array[moderate_mask],
            )
        )

    else:

        result["moderate_MAE"] = np.nan

    if high_mask.sum() > 0:

        result["high_MAE"] = (
            mean_absolute_error(
                y_true_array[high_mask],
                y_pred_array[high_mask],
            )
        )

    else:

        result["high_MAE"] = np.nan

    return result


# ============================================================
# ROLLING FOLDS
# ============================================================

# Each validation period occurs strictly AFTER
# its corresponding training period.
#
# The final 2012-2013 test period is excluded.

FOLDS = [
    {
        "fold": "Fold 1",
        "train_end": 1994,
        "val_start": 1995,
        "val_end": 1998,
    },

    {
        "fold": "Fold 2",
        "train_end": 1998,
        "val_start": 1999,
        "val_end": 2002,
    },

    {
        "fold": "Fold 3",
        "train_end": 2002,
        "val_start": 2003,
        "val_end": 2006,
    },

    {
        "fold": "Fold 4",
        "train_end": 2006,
        "val_start": 2007,
        "val_end": 2009,
    },

    {
        "fold": "Fold 5",
        "train_end": 2009,
        "val_start": 2010,
        "val_end": 2011,
    },
]


# ============================================================
# FOLD OVERVIEW
# ============================================================

print("\n" + "=" * 70)
print("ROLLING FOLD DESIGN")
print("=" * 70)


for fold in FOLDS:

    train_count = int(
        (
            df["year"]
            <= fold["train_end"]
        ).sum()
    )

    val_count = int(
        (
            df["year"].between(
                fold["val_start"],
                fold["val_end"],
            )
        ).sum()
    )

    print(
        f"\n{fold['fold']}:"
    )

    print(
        f"  Train: <= {fold['train_end']} "
        f"({train_count:,} rows)"
    )

    print(
        f"  Validation: "
        f"{fold['val_start']}-{fold['val_end']} "
        f"({val_count:,} rows)"
    )


# ============================================================
# ROLLING VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ROLLING VALIDATION")
print("=" * 70)


fold_results = []


for fold in FOLDS:

    fold_name = fold["fold"]

    train_mask = (
        df["year"]
        <= fold["train_end"]
    )

    val_mask = df["year"].between(
        fold["val_start"],
        fold["val_end"],
    )

    X_train = X.loc[
        train_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    X_val = X.loc[
        val_mask
    ]

    y_val = y.loc[
        val_mask
    ]

    print(
        f"\n{'-' * 70}"
    )

    print(
        f"{fold_name}: "
        f"Train <= {fold['train_end']} "
        f"| Validation "
        f"{fold['val_start']}-{fold['val_end']}"
    )

    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val):,}"
    )


    # --------------------------------------------------------
    # Validation distribution
    # --------------------------------------------------------

    val_bands = (
        y_val
        .apply(classify_impact_band)
        .value_counts()
    )

    print("\nValidation bands:")

    for band in [
        "LOW",
        "MODERATE",
        "HIGH",
    ]:

        print(
            f"  {band}: "
            f"{int(val_bands.get(band, 0))}"
        )


    # --------------------------------------------------------
    # Each model
    # --------------------------------------------------------

    for model_name in MODEL_CONFIGS:

        print(
            f"\nTraining {model_name}..."
        )

        model = build_model(
            model_name
        )

        if model_name == "Log Target HGB":

            y_train_fit = np.log1p(
                y_train
            )

            model.fit(
                X_train,
                y_train_fit,
            )

            val_pred = np.expm1(
                model.predict(X_val)
            )

            val_pred = np.maximum(
                val_pred,
                0,
            )

        else:

            model.fit(
                X_train,
                y_train,
            )

            val_pred = model.predict(
                X_val
            )

        metrics = calculate_metrics(
            y_val,
            val_pred,
        )

        tail_metrics = (
            calculate_tail_metrics(
                y_val,
                val_pred,
            )
        )

        row = {

            "fold":
                fold_name,

            "train_end":
                fold["train_end"],

            "validation_start":
                fold["val_start"],

            "validation_end":
                fold["val_end"],

            "train_rows":
                len(X_train),

            "validation_rows":
                len(X_val),

            "model":
                model_name,

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"],

            "Spearman":
                metrics["Spearman"],

            **tail_metrics,
        }

        fold_results.append(
            row
        )

        print(
            f"  MAE      : "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"  RMSE     : "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"  R2       : "
            f"{metrics['R2']:.4f}"
        )

        print(
            f"  Spearman : "
            f"{metrics['Spearman']:.4f}"
        )

        if tail_metrics[
            "tail_events"
        ] > 0:

            print(
                f"  Tail MAE : "
                f"{tail_metrics['tail_MAE']:.4f}"
            )


# ============================================================
# FOLD RESULTS DATAFRAME
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)


print("\n" + "=" * 70)
print("ALL ROLLING FOLD RESULTS")
print("=" * 70)


print(
    fold_results_df[
        [
            "fold",
            "model",
            "validation_rows",
            "MAE",
            "RMSE",
            "R2",
            "Spearman",
            "tail_events",
            "tail_MAE",
            "tail_bias",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# MODEL STABILITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL STABILITY SUMMARY")
print("=" * 70)


stability_summary = (
    fold_results_df
    .groupby("model")
    .agg(

        folds=(
            "fold",
            "count",
        ),

        mean_MAE=(
            "MAE",
            "mean",
        ),

        std_MAE=(
            "MAE",
            "std",
        ),

        mean_RMSE=(
            "RMSE",
            "mean",
        ),

        std_RMSE=(
            "RMSE",
            "std",
        ),

        mean_R2=(
            "R2",
            "mean",
        ),

        std_R2=(
            "R2",
            "std",
        ),

        mean_Spearman=(
            "Spearman",
            "mean",
        ),

        std_Spearman=(
            "Spearman",
            "std",
        ),

        mean_tail_MAE=(
            "tail_MAE",
            "mean",
        ),

        std_tail_MAE=(
            "tail_MAE",
            "std",
        ),
    )
    .reset_index()
)


# ------------------------------------------------------------
# Stability score
# ------------------------------------------------------------

# Higher mean R2 is better.
# Lower std R2 indicates more stable performance.

stability_summary[
    "R2_stability_score"
] = (
    stability_summary["mean_R2"]
    - stability_summary["std_R2"]
)


stability_summary = (
    stability_summary
    .sort_values(
        [
            "mean_R2",
            "mean_MAE",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)


print(
    stability_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# FOLD-BY-FOLD R2 STABILITY
# ============================================================

print("\n" + "=" * 70)
print("R² BY FOLD")
print("=" * 70)


r2_pivot = (
    fold_results_df
    .pivot(
        index="fold",
        columns="model",
        values="R2",
    )
)


print(
    r2_pivot
    .round(4)
    .to_string()
)


# ============================================================
# MAE BY FOLD
# ============================================================

print("\n" + "=" * 70)
print("MAE BY FOLD")
print("=" * 70)


mae_pivot = (
    fold_results_df
    .pivot(
        index="fold",
        columns="model",
        values="MAE",
    )
)


print(
    mae_pivot
    .round(4)
    .to_string()
)


# ============================================================
# SELECT MODEL
# ============================================================

selected_model_name = (
    stability_summary
    .iloc[0]["model"]
)


print("\n" + "=" * 70)
print("SELECTED MODEL")
print("=" * 70)


print(
    f"\nSelected based primarily on "
    f"mean rolling-validation R2:"
)

print(
    selected_model_name
)


selected_summary = (
    stability_summary[
        stability_summary["model"]
        == selected_model_name
    ]
    .iloc[0]
)


print(
    f"\nMean validation R2: "
    f"{selected_summary['mean_R2']:.4f}"
)

print(
    f"Validation R2 std: "
    f"{selected_summary['std_R2']:.4f}"
)

print(
    f"Mean validation MAE: "
    f"{selected_summary['mean_MAE']:.4f}"
)

print(
    f"Mean validation Spearman: "
    f"{selected_summary['mean_Spearman']:.4f}"
)


# ============================================================
# TRAIN FINAL SELECTED MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL SELECTED MODEL")
print("=" * 70)


final_model = build_model(
    selected_model_name
)


if selected_model_name == "Log Target HGB":

    y_pre_test_fit = np.log1p(
        y_pre_test
    )

    final_model.fit(
        X_pre_test,
        y_pre_test_fit,
    )

    final_test_pred = np.expm1(
        final_model.predict(
            X_final_test
        )
    )

    final_test_pred = np.maximum(
        final_test_pred,
        0,
    )

else:

    final_model.fit(
        X_pre_test,
        y_pre_test,
    )

    final_test_pred = (
        final_model.predict(
            X_final_test
        )
    )


# ============================================================
# FINAL TEST METRICS
# ============================================================

final_test_metrics = calculate_metrics(
    y_final_test,
    final_test_pred,
)


final_test_tail_metrics = (
    calculate_tail_metrics(
        y_final_test,
        final_test_pred,
    )
)


print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)


print(
    f"\nSelected model: "
    f"{selected_model_name}"
)


print(
    f"\nTest MAE      : "
    f"{final_test_metrics['MAE']:.4f}"
)

print(
    f"Test RMSE     : "
    f"{final_test_metrics['RMSE']:.4f}"
)

print(
    f"Test R2       : "
    f"{final_test_metrics['R2']:.4f}"
)

print(
    f"Test Spearman : "
    f"{final_test_metrics['Spearman']:.4f}"
)


print(
    f"\nTest tail events: "
    f"{final_test_tail_metrics['tail_events']}"
)

print(
    f"Test tail MAE: "
    f"{final_test_tail_metrics['tail_MAE']:.4f}"
)


# ============================================================
# FINAL TEST BAND DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST IMPACT-BAND DISTRIBUTION")
print("=" * 70)


test_bands = (
    y_final_test
    .apply(classify_impact_band)
    .value_counts()
    .reindex(
        [
            "LOW",
            "MODERATE",
            "HIGH",
        ],
        fill_value=0,
    )
)


print(
    test_bands.to_string()
)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

test_predictions = df_final_test.copy()


test_predictions[
    "predicted_impact"
] = final_test_pred


test_predictions[
    "absolute_error"
] = np.abs(
    test_predictions[
        TARGET
    ]
    - test_predictions[
        "predicted_impact"
    ]
)


test_predictions[
    "signed_error"
] = (
    test_predictions[
        "predicted_impact"
    ]
    - test_predictions[
        TARGET
    ]
)


test_predictions[
    "predicted_band"
] = (
    test_predictions[
        "predicted_impact"
    ]
    .apply(
        classify_impact_band
    )
)


# ============================================================
# TEST ERROR BY ACTUAL BAND
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST ERROR BY ACTUAL IMPACT BAND")
print("=" * 70)


band_test_error = (
    test_predictions
    .groupby("impact_band_analysis")
    .agg(

        events=(
            TARGET,
            "size",
        ),

        actual_mean=(
            TARGET,
            "mean",
        ),

        predicted_mean=(
            "predicted_impact",
            "mean",
        ),

        MAE=(
            "absolute_error",
            "mean",
        ),

        bias=(
            "signed_error",
            "mean",
        ),
    )
    .reset_index()
)


band_test_error[
    "impact_band_analysis"
] = pd.Categorical(
    band_test_error[
        "impact_band_analysis"
    ],
    categories=[
        "LOW",
        "MODERATE",
        "HIGH",
    ],
    ordered=True,
)


band_test_error = (
    band_test_error
    .sort_values(
        "impact_band_analysis"
    )
)


print(
    band_test_error
    .round(4)
    .to_string(index=False)
)


# ============================================================
# TEST ERROR BY DISASTER TYPE
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST ERROR BY DISASTER TYPE")
print("=" * 70)


disaster_test_error = (
    test_predictions
    .groupby("disaster_type")
    .agg(

        events=(
            TARGET,
            "size",
        ),

        actual_mean=(
            TARGET,
            "mean",
        ),

        predicted_mean=(
            "predicted_impact",
            "mean",
        ),

        MAE=(
            "absolute_error",
            "mean",
        ),

        bias=(
            "signed_error",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "MAE",
        ascending=False,
    )
)


print(
    disaster_test_error
    .head(15)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# WORST TEST ERRORS
# ============================================================

print("\n" + "=" * 70)
print("WORST FINAL TEST PREDICTIONS")
print("=" * 70)


worst_test = (
    test_predictions
    .sort_values(
        "absolute_error",
        ascending=False,
    )
    [
        [
            "year",
            "disaster_type",
            "district",
            "province",
            TARGET,
            "predicted_impact",
            "absolute_error",
            "signed_error",
            "impact_band_analysis",
            "predicted_band",
        ]
    ]
    .head(20)
)


print(
    worst_test
    .round(4)
    .to_string(index=False)
)


# ============================================================
# COMPARE WITH PREVIOUS BENCHMARKS
# ============================================================

previous_benchmarks = pd.DataFrame(
    [
        {
            "model": "Phase 5D.12 Random Forest",
            "test_MAE": 2.9029,
            "test_RMSE": 4.6165,
            "test_R2": 0.1320,
            "test_Spearman": 0.4413,
        },

        {
            "model": "Phase 5D.13 HistGradientBoosting",
            "test_MAE": 2.9911,
            "test_RMSE": 4.5895,
            "test_R2": 0.1419,
            "test_Spearman": 0.4413,
        },

        {
            "model": "Phase 5D.16 Shallow HGB",
            "test_MAE": 2.9877,
            "test_RMSE": 4.5726,
            "test_R2": 0.1482,
            "test_Spearman": 0.4410,
        },

        {
            "model": "Phase 5D.17 Rolling-selected "
                     + selected_model_name,
            "test_MAE":
                final_test_metrics["MAE"],
            "test_RMSE":
                final_test_metrics["RMSE"],
            "test_R2":
                final_test_metrics["R2"],
            "test_Spearman":
                final_test_metrics["Spearman"],
        },
    ]
)


print("\n" + "=" * 70)
print("BENCHMARK COMPARISON")
print("=" * 70)


print(
    previous_benchmarks
    .round(4)
    .to_string(index=False)
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

fold_results_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_rolling_fold_results.csv",
)


stability_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_model_stability_summary.csv",
)


r2_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_r2_by_fold.csv",
)


mae_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_mae_by_fold.csv",
)


band_error_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_test_error_by_band.csv",
)


disaster_error_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_test_error_by_disaster_type.csv",
)


worst_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_worst_test_predictions.csv",
)


test_predictions_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_final_test_predictions.csv",
)


benchmark_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_benchmark_comparison.csv",
)


configuration_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d17_selected_configuration.json",
)


fold_results_df.to_csv(
    fold_results_path,
    index=False,
)


stability_summary.to_csv(
    stability_path,
    index=False,
)


r2_pivot.to_csv(
    r2_path
)


mae_pivot.to_csv(
    mae_path
)


band_test_error.to_csv(
    band_error_path,
    index=False,
)


disaster_test_error.to_csv(
    disaster_error_path,
    index=False,
)


worst_test.to_csv(
    worst_path,
    index=False,
)


test_predictions.to_csv(
    test_predictions_path,
    index=False,
)


previous_benchmarks.to_csv(
    benchmark_path,
    index=False,
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

selected_config = {
    "phase": "5D.17",

    "selection_method":
        "Expanding rolling time-series validation",

    "selected_model":
        selected_model_name,

    "model_configuration":
        MODEL_CONFIGS[
            selected_model_name
        ],

    "rolling_folds":
        FOLDS,

    "final_test_period":
        "2012-2013",

    "final_test_rows":
        int(
            len(
                X_final_test
            )
        ),

    "selection_metric":
        "mean rolling validation R2",

    "selected_model_mean_validation_R2":
        float(
            selected_summary[
                "mean_R2"
            ]
        ),

    "selected_model_validation_R2_std":
        float(
            selected_summary[
                "std_R2"
            ]
        ),

    "final_test_metrics": {
        "MAE":
            float(
                final_test_metrics[
                    "MAE"
                ]
            ),

        "RMSE":
            float(
                final_test_metrics[
                    "RMSE"
                ]
            ),

        "R2":
            float(
                final_test_metrics[
                    "R2"
                ]
            ),

        "Spearman":
            float(
                final_test_metrics[
                    "Spearman"
                ]
            ),
    },

    "high_event_evaluation_note":
        "No HIGH-impact events exist in "
        "the 2012-2013 final test period.",
}


with open(
    configuration_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selected_config,
        f,
        indent=4,
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "phase_5d17_rolling_selected_model.joblib",
)


joblib.dump(
    final_model,
    final_model_path,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5D.17 COMPLETE")
print("=" * 70)


print(
    f"\nSelected model: "
    f"{selected_model_name}"
)


print(
    f"Mean rolling validation R2: "
    f"{selected_summary['mean_R2']:.4f}"
)


print(
    f"Rolling validation R2 std: "
    f"{selected_summary['std_R2']:.4f}"
)


print(
    "\nFinal untouched 2012-2013 test:"
)


print(
    f"MAE      : "
    f"{final_test_metrics['MAE']:.4f}"
)


print(
    f"RMSE     : "
    f"{final_test_metrics['RMSE']:.4f}"
)


print(
    f"R2       : "
    f"{final_test_metrics['R2']:.4f}"
)


print(
    f"Spearman : "
    f"{final_test_metrics['Spearman']:.4f}"
)


print("\nSaved data files:")

print(
    f"1. {fold_results_path}"
)

print(
    f"2. {stability_path}"
)

print(
    f"3. {r2_path}"
)

print(
    f"4. {mae_path}"
)

print(
    f"5. {band_error_path}"
)

print(
    f"6. {disaster_error_path}"
)

print(
    f"7. {worst_path}"
)

print(
    f"8. {test_predictions_path}"
)

print(
    f"9. {benchmark_path}"
)

print(
    f"10. {configuration_path}"
)


print("\nSaved model:")

print(
    final_model_path
)


print("\nImportant:")

print(
    "2012-2013 was never used to select the "
    "model or rolling-validation configuration."
)

print(
    "HIGH-impact performance cannot be established "
    "because the final test period contains zero HIGH events."
)


print("\nPhase 5D.17 finished successfully.")