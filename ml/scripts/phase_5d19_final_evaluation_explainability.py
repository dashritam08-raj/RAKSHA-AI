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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# PHASE 5D.19
# FINAL EVALUATION + EXPLAINABILITY
#
# Primary production candidate:
#   70% Log-Target HGB
#   30% Shallow HGB
#
# Final test:
#   2012-2013
#
# IMPORTANT:
# - Test data is NOT used for feature-importance selection.
# - Test data is used only for final evaluation.
# - No post-event outcome features are used.
# ============================================================

print("=" * 70)
print("PHASE 5D.19 - FINAL EVALUATION + EXPLAINABILITY")
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
# REQUIRED COLUMNS
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


df["impact_band"] = (
    df[TARGET]
    .apply(classify_impact_band)
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
# MODEL FACTORIES
# ============================================================

def build_shallow_hgb():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),

            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=400,
                    learning_rate=0.03,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def build_log_hgb():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),

            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=42,
                ),
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

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    low_mask = y_true_array < 30

    moderate_mask = (
        (y_true_array >= 30)
        & (y_true_array < 55)
    )

    high_mask = y_true_array >= 55

    tail_mask = y_true_array >= 30

    result = {
        "low_events": int(low_mask.sum()),
        "moderate_events": int(moderate_mask.sum()),
        "high_events": int(high_mask.sum()),
        "tail_events": int(tail_mask.sum()),
    }

    if low_mask.sum() > 0:

        result["low_MAE"] = (
            mean_absolute_error(
                y_true_array[low_mask],
                y_pred_array[low_mask],
            )
        )

    else:

        result["low_MAE"] = np.nan

    if moderate_mask.sum() > 0:

        result["moderate_MAE"] = (
            mean_absolute_error(
                y_true_array[moderate_mask],
                y_pred_array[moderate_mask],
            )
        )

        result["moderate_bias"] = np.mean(
            y_pred_array[moderate_mask]
            - y_true_array[moderate_mask]
        )

    else:

        result["moderate_MAE"] = np.nan
        result["moderate_bias"] = np.nan

    if high_mask.sum() > 0:

        result["high_MAE"] = (
            mean_absolute_error(
                y_true_array[high_mask],
                y_pred_array[high_mask],
            )
        )

    else:

        result["high_MAE"] = np.nan

    if tail_mask.sum() > 0:

        result["tail_MAE"] = (
            mean_absolute_error(
                y_true_array[tail_mask],
                y_pred_array[tail_mask],
            )
        )

        result["tail_bias"] = np.mean(
            y_pred_array[tail_mask]
            - y_true_array[tail_mask]
        )

    else:

        result["tail_MAE"] = np.nan
        result["tail_bias"] = np.nan

    return result


# ============================================================
# CHRONOLOGICAL DATA SPLITS
# ============================================================

train_mask = df["year"] <= 2011

test_mask = df["year"] >= 2012


X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

df_train = df.loc[train_mask].copy()
df_test = df.loc[test_mask].copy()


print("\nFinal training period:")
print("<= 2011")

print(
    f"Training rows: "
    f"{len(X_train):,}"
)

print("\nFinal test period:")
print("2012-2013")

print(
    f"Test rows: "
    f"{len(X_test):,}"
)


# ============================================================
# FINAL MODEL WEIGHTS
# ============================================================

LOG_WEIGHT = 0.70
SHALLOW_WEIGHT = 0.30


print("\n" + "=" * 70)
print("FINAL PRODUCTION CANDIDATE")
print("=" * 70)

print(
    f"\nLog Target HGB weight : "
    f"{LOG_WEIGHT:.2f}"
)

print(
    f"Shallow HGB weight    : "
    f"{SHALLOW_WEIGHT:.2f}"
)


# ============================================================
# TRAIN FINAL SHALLOW HGB
# ============================================================

print(
    "\nTraining final Shallow HGB..."
)


shallow_model = build_shallow_hgb()

shallow_model.fit(
    X_train,
    y_train,
)


shallow_test_pred = (
    shallow_model.predict(
        X_test
    )
)


# ============================================================
# TRAIN FINAL LOG HGB
# ============================================================

print(
    "Training final Log Target HGB..."
)


log_model = build_log_hgb()

y_train_log = np.log1p(
    y_train
)

log_model.fit(
    X_train,
    y_train_log,
)


log_test_pred = np.expm1(
    log_model.predict(
        X_test
    )
)

log_test_pred = np.maximum(
    log_test_pred,
    0,
)


# ============================================================
# FINAL BLEND
# ============================================================

blend_test_pred = (
    LOG_WEIGHT
    * log_test_pred

    +

    SHALLOW_WEIGHT
    * shallow_test_pred
)


blend_test_pred = np.maximum(
    blend_test_pred,
    0,
)


# ============================================================
# FINAL METRICS
# ============================================================

shallow_metrics = calculate_metrics(
    y_test,
    shallow_test_pred,
)

log_metrics = calculate_metrics(
    y_test,
    log_test_pred,
)

blend_metrics = calculate_metrics(
    y_test,
    blend_test_pred,
)


shallow_tail = calculate_tail_metrics(
    y_test,
    shallow_test_pred,
)

log_tail = calculate_tail_metrics(
    y_test,
    log_test_pred,
)

blend_tail = calculate_tail_metrics(
    y_test,
    blend_test_pred,
)


# ============================================================
# PRINT FINAL METRICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)


print("\nShallow HGB:")

print(
    f"MAE      : "
    f"{shallow_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{shallow_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{shallow_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{shallow_metrics['Spearman']:.4f}"
)


print("\nLog Target HGB:")

print(
    f"MAE      : "
    f"{log_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{log_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{log_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{log_metrics['Spearman']:.4f}"
)


print("\nFINAL 70/30 BLEND:")

print(
    f"MAE      : "
    f"{blend_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{blend_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{blend_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{blend_metrics['Spearman']:.4f}"
)


# ============================================================
# TAIL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("FINAL TAIL PERFORMANCE")
print("=" * 70)


tail_summary = pd.DataFrame(
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
                "70/30 Blend",

            **blend_tail,
        },
    ]
)


print(
    tail_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

test_predictions = df_test.copy()


test_predictions[
    "shallow_prediction"
] = shallow_test_pred


test_predictions[
    "log_prediction"
] = log_test_pred


test_predictions[
    "blend_prediction"
] = blend_test_pred


test_predictions[
    "absolute_error"
] = np.abs(
    test_predictions[TARGET]
    - test_predictions[
        "blend_prediction"
    ]
)


test_predictions[
    "signed_error"
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
# BAND PERFORMANCE
# ============================================================

band_performance = (
    test_predictions
    .groupby("impact_band")
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
            "blend_prediction",
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


band_performance[
    "impact_band"
] = pd.Categorical(
    band_performance[
        "impact_band"
    ],
    categories=[
        "LOW",
        "MODERATE",
        "HIGH",
    ],
    ordered=True,
)


band_performance = (
    band_performance
    .sort_values(
        "impact_band"
    )
)


print("\n" + "=" * 70)
print("ERROR BY IMPACT BAND")
print("=" * 70)


print(
    band_performance
    .round(4)
    .to_string(index=False)
)


# ============================================================
# DISASTER TYPE PERFORMANCE
# ============================================================

disaster_performance = (
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
            "blend_prediction",
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


print("\n" + "=" * 70)
print("ERROR BY DISASTER TYPE")
print("=" * 70)


print(
    disaster_performance
    .head(15)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

worst_predictions = (
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
            "shallow_prediction",
            "log_prediction",
            "blend_prediction",
            "absolute_error",
            "signed_error",
            "impact_band",
            "predicted_band",
        ]
    ]
    .head(25)
)


print("\n" + "=" * 70)
print("WORST FINAL BLEND PREDICTIONS")
print("=" * 70)


print(
    worst_predictions
    .round(4)
    .to_string(index=False)
)


# ============================================================
# EXPLAINABILITY DATA
#
# IMPORTANT:
# The final model was trained through 2011.
# For honest permutation importance, we build separate
# explainability models trained only through 2009 and
# evaluate importance on the 2010-2011 validation period.
#
# This avoids calculating importance on the same rows used
# to train the final production models.
# ============================================================

print("\n" + "=" * 70)
print("EXPLAINABILITY ANALYSIS")
print("=" * 70)


explain_train_mask = (
    df["year"] <= 2009
)

explain_val_mask = (
    df["year"].between(
        2010,
        2011,
    )
)


X_explain_train = X.loc[
    explain_train_mask
]

y_explain_train = y.loc[
    explain_train_mask
]

X_explain_val = X.loc[
    explain_val_mask
]

y_explain_val = y.loc[
    explain_val_mask
]


print(
    f"\nExplainability training rows: "
    f"{len(X_explain_train):,}"
)

print(
    f"Explainability validation rows: "
    f"{len(X_explain_val):,}"
)


# ============================================================
# EXPLAINABILITY SHALLOW MODEL
# ============================================================

print(
    "\nTraining explainability Shallow HGB..."
)


explain_shallow = build_shallow_hgb()

explain_shallow.fit(
    X_explain_train,
    y_explain_train,
)


# ============================================================
# EXPLAINABILITY LOG MODEL
# ============================================================

print(
    "Training explainability Log Target HGB..."
)


explain_log = build_log_hgb()

explain_log.fit(
    X_explain_train,
    np.log1p(
        y_explain_train
    ),
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print(
    "\nCalculating permutation importance..."
)

print(
    "Validation period: 2010-2011"
)


shallow_importance = (
    permutation_importance(
        explain_shallow,
        X_explain_val,
        y_explain_val,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
)


# For log model, define a scorer operating on original
# impact-score scale.

def log_model_scorer(
    estimator,
    X_data,
    y_data,
):

    predictions = np.expm1(
        estimator.predict(
            X_data
        )
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    return -mean_absolute_error(
        y_data,
        predictions,
    )


log_importance = (
    permutation_importance(
        explain_log,
        X_explain_val,
        y_explain_val,
        scoring=log_model_scorer,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
)


# ============================================================
# BUILD FEATURE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature":
            FEATURES,

        "shallow_importance":
            shallow_importance
            .importances_mean,

        "shallow_std":
            shallow_importance
            .importances_std,

        "log_importance":
            log_importance
            .importances_mean,

        "log_std":
            log_importance
            .importances_std,
    }
)


importance_df[
    "blend_importance"
] = (
    SHALLOW_WEIGHT
    * importance_df[
        "shallow_importance"
    ]

    +

    LOG_WEIGHT
    * importance_df[
        "log_importance"
    ]
)


importance_df = (
    importance_df
    .sort_values(
        "blend_importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


importance_df[
    "rank"
] = (
    np.arange(
        len(importance_df)
    )
    + 1
)


print("\n" + "=" * 70)
print("TOP FEATURES FOR FINAL BLEND")
print("=" * 70)


print(
    importance_df[
        [
            "rank",
            "feature",
            "blend_importance",
            "shallow_importance",
            "log_importance",
        ]
    ]
    .head(20)
    .round(6)
    .to_string(index=False)
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)


prediction_distribution = pd.DataFrame(
    {
        "model": [
            "Actual",
            "Shallow HGB",
            "Log Target HGB",
            "70/30 Blend",
        ],

        "mean": [
            y_test.mean(),
            shallow_test_pred.mean(),
            log_test_pred.mean(),
            blend_test_pred.mean(),
        ],

        "median": [
            y_test.median(),
            np.median(
                shallow_test_pred
            ),
            np.median(
                log_test_pred
            ),
            np.median(
                blend_test_pred
            ),
        ],

        "std": [
            y_test.std(),
            np.std(
                shallow_test_pred
            ),
            np.std(
                log_test_pred
            ),
            np.std(
                blend_test_pred
            ),
        ],

        "min": [
            y_test.min(),
            shallow_test_pred.min(),
            log_test_pred.min(),
            blend_test_pred.min(),
        ],

        "max": [
            y_test.max(),
            shallow_test_pred.max(),
            log_test_pred.max(),
            blend_test_pred.max(),
        ],
    }
)


print(
    prediction_distribution
    .round(4)
    .to_string(index=False)
)


# ============================================================
# FINAL MODEL SCORECARD
# ============================================================

scorecard = pd.DataFrame(
    [
        {
            "model":
                "Random Forest",
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
                "Shallow HGB",
            "test_MAE":
                shallow_metrics[
                    "MAE"
                ],
            "test_RMSE":
                shallow_metrics[
                    "RMSE"
                ],
            "test_R2":
                shallow_metrics[
                    "R2"
                ],
            "test_Spearman":
                shallow_metrics[
                    "Spearman"
                ],
        },

        {
            "model":
                "Log Target HGB",
            "test_MAE":
                log_metrics[
                    "MAE"
                ],
            "test_RMSE":
                log_metrics[
                    "RMSE"
                ],
            "test_R2":
                log_metrics[
                    "R2"
                ],
            "test_Spearman":
                log_metrics[
                    "Spearman"
                ],
        },

        {
            "model":
                "Final 70/30 Blend",
            "test_MAE":
                blend_metrics[
                    "MAE"
                ],
            "test_RMSE":
                blend_metrics[
                    "RMSE"
                ],
            "test_R2":
                blend_metrics[
                    "R2"
                ],
            "test_Spearman":
                blend_metrics[
                    "Spearman"
                ],
        },
    ]
)


print("\n" + "=" * 70)
print("FINAL MODEL SCORECARD")
print("=" * 70)


print(
    scorecard
    .round(4)
    .to_string(index=False)
)


# ============================================================
# FINAL MODEL DECISION
# ============================================================

# We choose the blend as the production candidate because
# rolling validation selected it and it has the strongest
# overall MAE/ranking combination.
#
# R2/RMSE remain secondary monitoring metrics.

production_model = (
    "70/30 Log Target HGB + Shallow HGB Blend"
)


print("\n" + "=" * 70)
print("FINAL PRODUCTION DECISION")
print("=" * 70)


print(
    f"\nProduction candidate:"
)

print(
    production_model
)


print(
    "\nWeights:"
)

print(
    f"Log Target HGB = "
    f"{LOG_WEIGHT:.2f}"
)

print(
    f"Shallow HGB    = "
    f"{SHALLOW_WEIGHT:.2f}"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

scorecard_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_final_scorecard.csv",
)


tail_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_final_tail_performance.csv",
)


band_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_final_band_performance.csv",
)


disaster_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_final_disaster_performance.csv",
)


worst_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_final_worst_predictions.csv",
)


prediction_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_final_test_predictions.csv",
)


importance_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_final_feature_importance.csv",
)


distribution_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_prediction_distribution.csv",
)


report_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_model_report.json",
)


scorecard.to_csv(
    scorecard_path,
    index=False,
)


tail_summary.to_csv(
    tail_path,
    index=False,
)


band_performance.to_csv(
    band_path,
    index=False,
)


disaster_performance.to_csv(
    disaster_path,
    index=False,
)


worst_predictions.to_csv(
    worst_path,
    index=False,
)


test_predictions.to_csv(
    prediction_path,
    index=False,
)


importance_df.to_csv(
    importance_path,
    index=False,
)


prediction_distribution.to_csv(
    distribution_path,
    index=False,
)


# ============================================================
# SAVE MODEL REPORT
# ============================================================

report = {

    "phase":
        "5D.19",

    "production_model":
        production_model,

    "weights": {

        "log_target_hgb":
            LOG_WEIGHT,

        "shallow_hgb":
            SHALLOW_WEIGHT,
    },

    "training_period":
        "<= 2011",

    "final_test_period":
        "2012-2013",

    "test_rows":
        int(
            len(
                X_test
            )
        ),

    "test_metrics": {

        "MAE":
            float(
                blend_metrics[
                    "MAE"
                ]
            ),

        "RMSE":
            float(
                blend_metrics[
                    "RMSE"
                ]
            ),

        "R2":
            float(
                blend_metrics[
                    "R2"
                ]
            ),

        "Spearman":
            float(
                blend_metrics[
                    "Spearman"
                ]
            ),
    },

    "tail_metrics": {

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

        "moderate_events":
            int(
                blend_tail[
                    "moderate_events"
                ]
            ),

        "high_events":
            int(
                blend_tail[
                    "high_events"
                ]
            ),
    },

    "most_important_features":
        importance_df[
            [
                "feature",
                "blend_importance",
            ]
        ]
        .head(10)
        .to_dict(
            orient="records"
        ),

    "limitations": [

        "The final test period contains zero HIGH-impact events.",

        "Therefore HIGH-impact generalization cannot be established.",

        "Moderate/high-impact events are strongly underpredicted.",

        "Model performance varies across historical time periods.",

        "Disaster type is substantially more informative than most individual environmental features.",
    ],

    "test_not_used_for_feature_selection":
        True,

    "test_not_used_for_blend_selection":
        True,

    "explainability_period":
        "2010-2011 validation",

    "explainability_training_period":
        "<= 2009",
}


with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=4,
    )


# ============================================================
# SAVE FINAL MODEL COMPONENTS
# ============================================================

final_shallow_path = os.path.join(
    MODEL_DIR,
    "raksha_final_shallow_hgb.joblib",
)


final_log_path = os.path.join(
    MODEL_DIR,
    "raksha_final_log_target_hgb.joblib",
)


joblib.dump(
    shallow_model,
    final_shallow_path,
)


joblib.dump(
    log_model,
    final_log_path,
)


# ============================================================
# SAVE MODEL CONFIGURATION
# ============================================================

model_config = {

    "production_model":
        "70/30 prediction blend",

    "log_target_hgb_weight":
        LOG_WEIGHT,

    "shallow_hgb_weight":
        SHALLOW_WEIGHT,

    "shallow_configuration": {

        "max_iter":
            400,

        "learning_rate":
            0.03,

        "max_leaf_nodes":
            15,

        "min_samples_leaf":
            20,

        "l2_regularization":
            1.0,
    },

    "log_target_configuration": {

        "max_iter":
            300,

        "learning_rate":
            0.05,

        "max_leaf_nodes":
            31,

        "min_samples_leaf":
            20,

        "l2_regularization":
            1.0,
    },

    "features":
        FEATURES,

    "target":
        TARGET,

    "training_period":
        "<= 2011",

    "final_test_period":
        "2012-2013",

    "post_event_features_used":
        False,

    "high_event_test_count":
        int(
            blend_tail[
                "high_events"
            ]
        ),
}


model_config_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d19_production_model_config.json",
)


with open(
    model_config_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        model_config,
        f,
        indent=4,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5D.19 COMPLETE")
print("=" * 70)


print(
    "\nPRODUCTION CANDIDATE:"
)

print(
    "70% Log Target HGB "
    "+ 30% Shallow HGB"
)


print(
    "\nFINAL TEST PERFORMANCE:"
)


print(
    f"MAE      : "
    f"{blend_metrics['MAE']:.4f}"
)


print(
    f"RMSE     : "
    f"{blend_metrics['RMSE']:.4f}"
)


print(
    f"R2       : "
    f"{blend_metrics['R2']:.4f}"
)


print(
    f"Spearman : "
    f"{blend_metrics['Spearman']:.4f}"
)


print(
    "\nFINAL TAIL MAE:"
)


print(
    f"{blend_tail['tail_MAE']:.4f}"
)


print(
    "\nHIGH events in final test:"
)


print(
    f"{blend_tail['high_events']}"
)


print(
    "\nSaved data files:"
)


print(
    f"1. {scorecard_path}"
)

print(
    f"2. {tail_path}"
)

print(
    f"3. {band_path}"
)

print(
    f"4. {disaster_path}"
)

print(
    f"5. {worst_path}"
)

print(
    f"6. {prediction_path}"
)

print(
    f"7. {importance_path}"
)

print(
    f"8. {distribution_path}"
)

print(
    f"9. {report_path}"
)

print(
    f"10. {model_config_path}"
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
    "The 2012-2013 test set was used only for final evaluation."
)


print(
    "Feature importance was calculated on a separate "
    "2010-2011 validation setup."
)


print(
    "No post-event outcome feature was used."
)


print(
    "HIGH-impact generalization cannot be established "
    "because the final test set contains zero HIGH events."
)


print(
    "\nPhase 5D.19 finished successfully."
)