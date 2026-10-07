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
# PHASE 5D.18
# FINAL ENSEMBLE / BLEND SELECTION
#
# Models:
#   1. Shallow HGB
#   2. Log-Target HGB
#
# Blend:
#   prediction = alpha * LogHGB
#              + (1-alpha) * ShallowHGB
#
# IMPORTANT:
# - Rolling validation is used for selecting alpha.
# - 2012-2013 remains completely untouched until final test.
# - No post-event outcome features are used.
# ============================================================

print("=" * 70)
print("PHASE 5D.18 - FINAL ENSEMBLE / BLEND SELECTION")
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
        "Required dataset columns are missing."
    )


# ============================================================
# X / y
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


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
# PREPROCESSING GROUPS
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
# MODEL FACTORY
# ============================================================

def build_shallow_hgb():

    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
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


def build_log_hgb():

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
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

        "tail_events":
            int(tail_mask.sum()),

        "moderate_events":
            int(moderate_mask.sum()),

        "high_events":
            int(high_mask.sum()),
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
# BLEND WEIGHTS
# ============================================================

# alpha = weight of Log Target HGB
#
# alpha = 0.00 -> 100% Shallow HGB
# alpha = 0.25 -> 25% Log + 75% Shallow
# alpha = 0.50 -> 50% Log + 50% Shallow
# alpha = 1.00 -> 100% Log Target HGB

ALPHAS = np.arange(
    0.00,
    1.01,
    0.05,
)


# ============================================================
# STORAGE
# ============================================================

fold_results = []

blend_results = []

all_oof_predictions = []


# ============================================================
# ROLLING VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ROLLING BLEND VALIDATION")
print("=" * 70)


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
        f"{fold['val_start']}-"
        f"{fold['val_end']}"
    )

    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val):,}"
    )


    # ========================================================
    # SHALLOW HGB
    # ========================================================

    print(
        "\nTraining Shallow HGB..."
    )

    shallow_model = (
        build_shallow_hgb()
    )

    shallow_model.fit(
        X_train,
        y_train,
    )

    shallow_pred = (
        shallow_model.predict(
            X_val
        )
    )


    # ========================================================
    # LOG TARGET HGB
    # ========================================================

    print(
        "Training Log Target HGB..."
    )

    log_model = (
        build_log_hgb()
    )

    y_train_log = np.log1p(
        y_train
    )

    log_model.fit(
        X_train,
        y_train_log,
    )

    log_pred = np.expm1(
        log_model.predict(
            X_val
        )
    )

    log_pred = np.maximum(
        log_pred,
        0,
    )


    # ========================================================
    # INDIVIDUAL MODEL RESULTS
    # ========================================================

    shallow_metrics = (
        calculate_metrics(
            y_val,
            shallow_pred,
        )
    )

    log_metrics = (
        calculate_metrics(
            y_val,
            log_pred,
        )
    )


    fold_results.append(
        {
            "fold":
                fold_name,

            "model":
                "Shallow HGB",

            "MAE":
                shallow_metrics["MAE"],

            "RMSE":
                shallow_metrics["RMSE"],

            "R2":
                shallow_metrics["R2"],

            "Spearman":
                shallow_metrics["Spearman"],
        }
    )


    fold_results.append(
        {
            "fold":
                fold_name,

            "model":
                "Log Target HGB",

            "MAE":
                log_metrics["MAE"],

            "RMSE":
                log_metrics["RMSE"],

            "R2":
                log_metrics["R2"],

            "Spearman":
                log_metrics["Spearman"],
        }
    )


    print(
        "\nShallow HGB:"
    )

    print(
        f"  MAE      : "
        f"{shallow_metrics['MAE']:.4f}"
    )

    print(
        f"  RMSE     : "
        f"{shallow_metrics['RMSE']:.4f}"
    )

    print(
        f"  R2       : "
        f"{shallow_metrics['R2']:.4f}"
    )


    print(
        "\nLog Target HGB:"
    )

    print(
        f"  MAE      : "
        f"{log_metrics['MAE']:.4f}"
    )

    print(
        f"  RMSE     : "
        f"{log_metrics['RMSE']:.4f}"
    )

    print(
        f"  R2       : "
        f"{log_metrics['R2']:.4f}"
    )


    # ========================================================
    # BLEND SEARCH FOR THIS FOLD
    # ========================================================

    for alpha in ALPHAS:

        blend_pred = (
            alpha * log_pred
            + (1.0 - alpha)
            * shallow_pred
        )

        metrics = calculate_metrics(
            y_val,
            blend_pred,
        )

        blend_tail = (
            calculate_tail_metrics(
                y_val,
                blend_pred,
            )
        )


        blend_results.append(
            {
                "fold":
                    fold_name,

                "alpha_log":
                    alpha,

                "alpha_shallow":
                    1.0 - alpha,

                "MAE":
                    metrics["MAE"],

                "RMSE":
                    metrics["RMSE"],

                "R2":
                    metrics["R2"],

                "Spearman":
                    metrics["Spearman"],

                **blend_tail,
            }
        )


    # ========================================================
    # STORE OUT-OF-FOLD PREDICTIONS
    # ========================================================

    fold_prediction_df = pd.DataFrame(
        {
            "fold":
                fold_name,

            "year":
                df.loc[
                    val_mask,
                    "year"
                ].values,

            "disaster_type":
                df.loc[
                    val_mask,
                    "disaster_type"
                ].values,

            "actual_impact":
                y_val.values,

            "shallow_prediction":
                shallow_pred,

            "log_prediction":
                log_pred,
        }
    )


    all_oof_predictions.append(
        fold_prediction_df
    )


# ============================================================
# CONVERT RESULTS
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

blend_results_df = pd.DataFrame(
    blend_results
)

oof_predictions_df = pd.concat(
    all_oof_predictions,
    ignore_index=True,
)


# ============================================================
# ROLLING BLEND SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BLEND SUMMARY ACROSS ROLLING FOLDS")
print("=" * 70)


blend_summary = (
    blend_results_df
    .groupby("alpha_log")
    .agg(

        mean_MAE=(
            "MAE",
            "mean"
        ),

        std_MAE=(
            "MAE",
            "std"
        ),

        mean_RMSE=(
            "RMSE",
            "mean"
        ),

        std_RMSE=(
            "RMSE",
            "std"
        ),

        mean_R2=(
            "R2",
            "mean"
        ),

        std_R2=(
            "R2",
            "std"
        ),

        mean_Spearman=(
            "Spearman",
            "mean"
        ),

        std_Spearman=(
            "Spearman",
            "std"
        ),

        mean_tail_MAE=(
            "tail_MAE",
            "mean"
        ),

        std_tail_MAE=(
            "tail_MAE",
            "std"
        ),
    )
    .reset_index()
)


blend_summary[
    "alpha_shallow"
] = (
    1.0
    - blend_summary["alpha_log"]
)


# Sort primarily by mean R2.

blend_summary = (
    blend_summary
    .sort_values(
        [
            "mean_R2",
            "mean_MAE",
        ],
        ascending=[
            False,
            True,
        ]
    )
    .reset_index(drop=True)
)


print(
    blend_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# BEST BLEND
# ============================================================

best_blend = (
    blend_summary
    .iloc[0]
)


best_alpha_log = float(
    best_blend[
        "alpha_log"
    ]
)

best_alpha_shallow = float(
    best_blend[
        "alpha_shallow"
    ]
)


print("\n" + "=" * 70)
print("SELECTED BLEND")
print("=" * 70)


print(
    f"\nLog Target HGB weight : "
    f"{best_alpha_log:.2f}"
)

print(
    f"Shallow HGB weight    : "
    f"{best_alpha_shallow:.2f}"
)

print(
    f"\nMean rolling MAE      : "
    f"{best_blend['mean_MAE']:.4f}"
)

print(
    f"Mean rolling RMSE     : "
    f"{best_blend['mean_RMSE']:.4f}"
)

print(
    f"Mean rolling R2       : "
    f"{best_blend['mean_R2']:.4f}"
)

print(
    f"Mean rolling Spearman : "
    f"{best_blend['mean_Spearman']:.4f}"
)


# ============================================================
# INDIVIDUAL VS BEST BLEND
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL MODELS VS BEST BLEND")
print("=" * 70)


individual_summary = (
    fold_results_df
    .groupby("model")
    .agg(

        mean_MAE=(
            "MAE",
            "mean"
        ),

        mean_RMSE=(
            "RMSE",
            "mean"
        ),

        mean_R2=(
            "R2",
            "mean"
        ),

        mean_Spearman=(
            "Spearman",
            "mean"
        ),
    )
    .reset_index()
)


individual_summary = pd.concat(
    [
        individual_summary,

        pd.DataFrame(
            [
                {
                    "model":
                        "Best Blend",

                    "mean_MAE":
                        best_blend[
                            "mean_MAE"
                        ],

                    "mean_RMSE":
                        best_blend[
                            "mean_RMSE"
                        ],

                    "mean_R2":
                        best_blend[
                            "mean_R2"
                        ],

                    "mean_Spearman":
                        best_blend[
                            "mean_Spearman"
                        ],
                }
            ]
        ),
    ],
    ignore_index=True,
)


print(
    individual_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# FINAL TRAINING DATA
# ============================================================

# Everything <= 2011 is used to train the final
# ensemble models.
#
# 2012-2013 remains untouched.

final_train_mask = (
    df["year"] <= 2011
)

final_test_mask = (
    df["year"] >= 2012
)


X_final_train = X.loc[
    final_train_mask
]

y_final_train = y.loc[
    final_train_mask
]


X_final_test = X.loc[
    final_test_mask
]

y_final_test = y.loc[
    final_test_mask
]


df_final_test = df.loc[
    final_test_mask
].copy()


print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)


print(
    f"\nFinal training rows: "
    f"{len(X_final_train):,}"
)

print(
    f"Final test rows: "
    f"{len(X_final_test):,}"
)


# ============================================================
# FINAL SHALLOW MODEL
# ============================================================

print(
    "\nTraining final Shallow HGB..."
)


final_shallow_model = (
    build_shallow_hgb()
)

final_shallow_model.fit(
    X_final_train,
    y_final_train,
)


final_shallow_test_pred = (
    final_shallow_model.predict(
        X_final_test
    )
)


# ============================================================
# FINAL LOG MODEL
# ============================================================

print(
    "Training final Log Target HGB..."
)


final_log_model = (
    build_log_hgb()
)


final_y_train_log = np.log1p(
    y_final_train
)


final_log_model.fit(
    X_final_train,
    final_y_train_log,
)


final_log_test_pred = np.expm1(
    final_log_model.predict(
        X_final_test
    )
)


final_log_test_pred = np.maximum(
    final_log_test_pred,
    0,
)


# ============================================================
# FINAL BLEND
# ============================================================

final_blend_test_pred = (
    best_alpha_log
    * final_log_test_pred

    +

    best_alpha_shallow
    * final_shallow_test_pred
)


final_blend_test_pred = np.maximum(
    final_blend_test_pred,
    0,
)


# ============================================================
# TEST METRICS
# ============================================================

shallow_test_metrics = (
    calculate_metrics(
        y_final_test,
        final_shallow_test_pred,
    )
)


log_test_metrics = (
    calculate_metrics(
        y_final_test,
        final_log_test_pred,
    )
)


blend_test_metrics = (
    calculate_metrics(
        y_final_test,
        final_blend_test_pred,
    )
)


print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)


print(
    "\nShallow HGB:"
)

print(
    f"MAE      : "
    f"{shallow_test_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{shallow_test_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{shallow_test_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{shallow_test_metrics['Spearman']:.4f}"
)


print(
    "\nLog Target HGB:"
)

print(
    f"MAE      : "
    f"{log_test_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{log_test_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{log_test_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{log_test_metrics['Spearman']:.4f}"
)


print(
    "\nFINAL BLEND:"
)

print(
    f"MAE      : "
    f"{blend_test_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{blend_test_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{blend_test_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{blend_test_metrics['Spearman']:.4f}"
)


# ============================================================
# TEST TAIL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST TAIL PERFORMANCE")
print("=" * 70)


shallow_tail = (
    calculate_tail_metrics(
        y_final_test,
        final_shallow_test_pred,
    )
)


log_tail = (
    calculate_tail_metrics(
        y_final_test,
        final_log_test_pred,
    )
)


blend_tail = (
    calculate_tail_metrics(
        y_final_test,
        final_blend_test_pred,
    )
)


tail_test_comparison = pd.DataFrame(
    [

        {
            "model":
                "Shallow HGB",

            **shallow_tail,
        },

        {
            "model":
                "Log Target HGB",

            **log_tail,
        },

        {
            "model":
                "Best Blend",

            **blend_tail,
        },
    ]
)


print(
    tail_test_comparison
    .round(4)
    .to_string(index=False)
)


# ============================================================
# TEST PREDICTION FILE
# ============================================================

test_predictions = df_final_test.copy()


test_predictions[
    "shallow_prediction"
] = final_shallow_test_pred


test_predictions[
    "log_prediction"
] = final_log_test_pred


test_predictions[
    "blend_prediction"
] = final_blend_test_pred


test_predictions[
    "absolute_error_blend"
] = np.abs(
    test_predictions[
        TARGET
    ]
    - test_predictions[
        "blend_prediction"
    ]
)


test_predictions[
    "signed_error_blend"
] = (
    test_predictions[
        "blend_prediction"
    ]
    - test_predictions[
        TARGET
    ]
)


test_predictions[
    "predicted_band"
] = (
    test_predictions[
        "blend_prediction"
    ]
    .apply(
        classify_impact_band
    )
)


# ============================================================
# TEST ERROR BY IMPACT BAND
# ============================================================

band_test_error = (
    test_predictions
    .groupby(
        "impact_band_analysis"
    )
    .agg(

        events=(
            TARGET,
            "size"
        ),

        actual_mean=(
            TARGET,
            "mean"
        ),

        predicted_mean=(
            "blend_prediction",
            "mean"
        ),

        MAE=(
            "absolute_error_blend",
            "mean"
        ),

        bias=(
            "signed_error_blend",
            "mean"
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


print("\n" + "=" * 70)
print("BLEND TEST ERROR BY IMPACT BAND")
print("=" * 70)


print(
    band_test_error
    .round(4)
    .to_string(index=False)
)


# ============================================================
# WORST BLEND PREDICTIONS
# ============================================================

worst_blend = (
    test_predictions
    .sort_values(
        "absolute_error_blend",
        ascending=False,
    )
    [
        [
            "year",
            "disaster_type",
            "district",
            "province",
            TARGET,
            "blend_prediction",
            "absolute_error_blend",
            "signed_error_blend",
            "impact_band_analysis",
            "predicted_band",
        ]
    ]
    .head(20)
)


print("\n" + "=" * 70)
print("WORST BLEND TEST PREDICTIONS")
print("=" * 70)


print(
    worst_blend
    .round(4)
    .to_string(index=False)
)


# ============================================================
# HISTORICAL BENCHMARK COMPARISON
# ============================================================

benchmark_comparison = pd.DataFrame(
    [

        {
            "model":
                "Phase 5D.12 Random Forest",

            "test_MAE":
                2.9029,

            "test_RMSE":
                4.6165,

            "test_R2":
                0.1320,

            "test_Spearman":
                0.4413,
        },

        {
            "model":
                "Phase 5D.16 Shallow HGB",

            "test_MAE":
                2.9877,

            "test_RMSE":
                4.5726,

            "test_R2":
                0.1482,

            "test_Spearman":
                0.4410,
        },

        {
            "model":
                "Phase 5D.17 Log Target HGB",

            "test_MAE":
                2.8545,

            "test_RMSE":
                4.7916,

            "test_R2":
                0.0646,

            "test_Spearman":
                0.4962,
        },

        {
            "model":
                "Phase 5D.18 Final Blend",

            "test_MAE":
                blend_test_metrics[
                    "MAE"
                ],

            "test_RMSE":
                blend_test_metrics[
                    "RMSE"
                ],

            "test_R2":
                blend_test_metrics[
                    "R2"
                ],

            "test_Spearman":
                blend_test_metrics[
                    "Spearman"
                ],
        },
    ]
)


print("\n" + "=" * 70)
print("HISTORICAL BENCHMARK COMPARISON")
print("=" * 70)


print(
    benchmark_comparison
    .round(4)
    .to_string(index=False)
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

blend_results_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_blend_fold_results.csv",
)


blend_summary_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_blend_summary.csv",
)


individual_summary_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_individual_vs_blend.csv",
)


oof_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_oof_predictions.csv",
)


test_predictions_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_final_test_predictions.csv",
)


band_error_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_test_error_by_band.csv",
)


worst_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_worst_blend_predictions.csv",
)


benchmark_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_benchmark_comparison.csv",
)


configuration_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d18_selected_configuration.json",
)


fold_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase_5d18_individual_fold_results.csv",
    ),
    index=False,
)


blend_results_df.to_csv(
    blend_results_path,
    index=False,
)


blend_summary.to_csv(
    blend_summary_path,
    index=False,
)


individual_summary.to_csv(
    individual_summary_path,
    index=False,
)


oof_predictions_df.to_csv(
    oof_path,
    index=False,
)


test_predictions.to_csv(
    test_predictions_path,
    index=False,
)


band_test_error.to_csv(
    band_error_path,
    index=False,
)


worst_blend.to_csv(
    worst_path,
    index=False,
)


benchmark_comparison.to_csv(
    benchmark_path,
    index=False,
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

configuration = {

    "phase":
        "5D.18",

    "ensemble_type":
        "weighted prediction blend",

    "log_target_weight":
        best_alpha_log,

    "shallow_hgb_weight":
        best_alpha_shallow,

    "selection_metric":
        "mean rolling validation R2",

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

    "test_metrics":
        {
            "MAE":
                float(
                    blend_test_metrics[
                        "MAE"
                    ]
                ),

            "RMSE":
                float(
                    blend_test_metrics[
                        "RMSE"
                    ]
                ),

            "R2":
                float(
                    blend_test_metrics[
                        "R2"
                    ]
                ),

            "Spearman":
                float(
                    blend_test_metrics[
                        "Spearman"
                    ]
                ),
        },

    "tail_metrics":
        {
            "tail_events":
                int(
                    blend_tail[
                        "tail_events"
                    ]
                ),

            "tail_MAE":
                (
                    None
                    if pd.isna(
                        blend_tail[
                            "tail_MAE"
                        ]
                    )
                    else float(
                        blend_tail[
                            "tail_MAE"
                        ]
                    )
                ),
        },

    "high_event_warning":
        "No HIGH-impact events "
        "exist in 2012-2013 test.",

    "test_not_used_for_selection":
        True,
}


with open(
    configuration_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        configuration,
        f,
        indent=4,
    )


# ============================================================
# SAVE FINAL MODELS
# ============================================================

final_shallow_path = os.path.join(
    MODEL_DIR,
    "phase_5d18_final_shallow_hgb.joblib",
)


final_log_path = os.path.join(
    MODEL_DIR,
    "phase_5d18_final_log_hgb.joblib",
)


joblib.dump(
    final_shallow_model,
    final_shallow_path,
)


joblib.dump(
    final_log_model,
    final_log_path,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5D.18 COMPLETE")
print("=" * 70)


print(
    "\nSelected blend:"
)


print(
    f"Log Target HGB : "
    f"{best_alpha_log:.2f}"
)


print(
    f"Shallow HGB    : "
    f"{best_alpha_shallow:.2f}"
)


print(
    "\nFinal 2012-2013 test:"
)


print(
    f"MAE      : "
    f"{blend_test_metrics['MAE']:.4f}"
)


print(
    f"RMSE     : "
    f"{blend_test_metrics['RMSE']:.4f}"
)


print(
    f"R2       : "
    f"{blend_test_metrics['R2']:.4f}"
)


print(
    f"Spearman : "
    f"{blend_test_metrics['Spearman']:.4f}"
)


print(
    "\nFinal tail MAE:"
)


print(
    f"{blend_tail['tail_MAE']:.4f}"
)


print(
    "\nSaved data files:"
)


print(
    f"1. {blend_results_path}"
)


print(
    f"2. {blend_summary_path}"
)


print(
    f"3. {individual_summary_path}"
)


print(
    f"4. {oof_path}"
)


print(
    f"5. {test_predictions_path}"
)


print(
    f"6. {band_error_path}"
)


print(
    f"7. {worst_path}"
)


print(
    f"8. {benchmark_path}"
)


print(
    f"9. {configuration_path}"
)


print(
    "\nSaved model files:"
)


print(
    f"1. {final_shallow_path}"
)


print(
    f"2. {final_log_path}"
)


print(
    "\nImportant:"
)


print(
    "The blend weight was selected using rolling "
    "validation only."
)


print(
    "The 2012-2013 test set was not used for "
    "blend selection."
)


print(
    "HIGH-impact generalization cannot be "
    "established because the final test period "
    "contains zero HIGH events."
)


print(
    "\nPhase 5D.18 finished successfully."
)