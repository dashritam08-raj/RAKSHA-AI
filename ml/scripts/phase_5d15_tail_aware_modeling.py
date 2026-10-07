import os
import warnings

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
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# PHASE 5D.15
# TAIL-AWARE MODELING
# ============================================================

print("=" * 70)
print("PHASE 5D.15 - TAIL-AWARE MODELING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/nepal_ml_features_v1.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# VALIDATE COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
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
# CREATE X / y
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# IMPACT BAND FUNCTION
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
# TAIL FLAG
# ============================================================

# Tail = MODERATE + HIGH
# Threshold = impact_score >= 30

df["tail_flag"] = (
    df[TARGET] >= 30
).astype(int)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_mask = df["year"] <= 2009
val_mask = df["year"].between(2010, 2011)
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

    preprocessor = ColumnTransformer(
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

    return preprocessor


# ============================================================
# MODEL FACTORY
# ============================================================

def build_regressor():

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
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

def calculate_regression_metrics(
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

def calculate_tail_metrics(
    y_true,
    y_pred,
):

    bands = y_true.apply(
        classify_impact_band
    )

    low_mask = bands == "LOW"
    moderate_mask = bands == "MODERATE"
    high_mask = bands == "HIGH"
    tail_mask = bands != "LOW"

    results = {}

    # Overall
    results["overall_mae"] = (
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    # LOW
    results["low_events"] = (
        int(low_mask.sum())
    )

    if low_mask.sum() > 0:
        results["low_mae"] = (
            mean_absolute_error(
                y_true[low_mask],
                y_pred[low_mask],
            )
        )
    else:
        results["low_mae"] = np.nan

    # MODERATE
    results["moderate_events"] = (
        int(moderate_mask.sum())
    )

    if moderate_mask.sum() > 0:
        results["moderate_mae"] = (
            mean_absolute_error(
                y_true[moderate_mask],
                y_pred[moderate_mask],
            )
        )
    else:
        results["moderate_mae"] = np.nan

    # HIGH
    results["high_events"] = (
        int(high_mask.sum())
    )

    if high_mask.sum() > 0:
        results["high_mae"] = (
            mean_absolute_error(
                y_true[high_mask],
                y_pred[high_mask],
            )
        )
    else:
        results["high_mae"] = np.nan

    # Tail overall
    results["tail_events"] = (
        int(tail_mask.sum())
    )

    if tail_mask.sum() > 0:
        results["tail_mae"] = (
            mean_absolute_error(
                y_true[tail_mask],
                y_pred[tail_mask],
            )
        )

        results["tail_actual_mean"] = (
            y_true[tail_mask].mean()
        )

        results["tail_predicted_mean"] = (
            np.mean(y_pred[tail_mask])
        )

        results["tail_bias"] = (
            np.mean(
                y_pred[tail_mask]
                - y_true[tail_mask]
            )
        )

    else:
        results["tail_mae"] = np.nan
        results["tail_actual_mean"] = np.nan
        results["tail_predicted_mean"] = np.nan
        results["tail_bias"] = np.nan

    return results


# ============================================================
# BASELINE HGB
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 - BASELINE HISTGRADIENTBOOSTING")
print("=" * 70)

baseline_model = build_regressor()

baseline_model.fit(
    X_train,
    y_train,
)

baseline_val_pred = (
    baseline_model.predict(X_val)
)

baseline_test_pred = (
    baseline_model.predict(X_test)
)


baseline_val_metrics = (
    calculate_regression_metrics(
        y_val,
        baseline_val_pred,
    )
)

baseline_test_metrics = (
    calculate_regression_metrics(
        y_test,
        baseline_test_pred,
    )
)


print("\nValidation:")
print(
    f"MAE      : "
    f"{baseline_val_metrics['MAE']:.4f}"
)
print(
    f"RMSE     : "
    f"{baseline_val_metrics['RMSE']:.4f}"
)
print(
    f"R2       : "
    f"{baseline_val_metrics['R2']:.4f}"
)
print(
    f"Spearman : "
    f"{baseline_val_metrics['Spearman']:.4f}"
)


print("\nTest:")
print(
    f"MAE      : "
    f"{baseline_test_metrics['MAE']:.4f}"
)
print(
    f"RMSE     : "
    f"{baseline_test_metrics['RMSE']:.4f}"
)
print(
    f"R2       : "
    f"{baseline_test_metrics['R2']:.4f}"
)
print(
    f"Spearman : "
    f"{baseline_test_metrics['Spearman']:.4f}"
)


# ============================================================
# MODEL 2 - WEIGHTED HGB
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 - WEIGHTED HISTGRADIENTBOOSTING")
print("=" * 70)


# Training-only weights.
#
# LOW       = 1
# MODERATE  = 5
# HIGH      = 15
#
# These weights do NOT modify validation or test metrics.

train_bands = (
    y_train.apply(
        classify_impact_band
    )
)

sample_weights = np.ones(
    len(y_train),
    dtype=float,
)

sample_weights[
    train_bands == "MODERATE"
] = 5.0

sample_weights[
    train_bands == "HIGH"
] = 15.0


print("\nTraining sample weights:")

print(
    f"LOW weight      : 1.0"
)

print(
    f"MODERATE weight : 5.0"
)

print(
    f"HIGH weight     : 15.0"
)


weighted_model = build_regressor()

weighted_model.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weights,
)


weighted_val_pred = (
    weighted_model.predict(X_val)
)

weighted_test_pred = (
    weighted_model.predict(X_test)
)


weighted_val_metrics = (
    calculate_regression_metrics(
        y_val,
        weighted_val_pred,
    )
)

weighted_test_metrics = (
    calculate_regression_metrics(
        y_test,
        weighted_test_pred,
    )
)


print("\nValidation:")
print(
    f"MAE      : "
    f"{weighted_val_metrics['MAE']:.4f}"
)
print(
    f"RMSE     : "
    f"{weighted_val_metrics['RMSE']:.4f}"
)
print(
    f"R2       : "
    f"{weighted_val_metrics['R2']:.4f}"
)
print(
    f"Spearman : "
    f"{weighted_val_metrics['Spearman']:.4f}"
)


print("\nTest:")
print(
    f"MAE      : "
    f"{weighted_test_metrics['MAE']:.4f}"
)
print(
    f"RMSE     : "
    f"{weighted_test_metrics['RMSE']:.4f}"
)
print(
    f"R2       : "
    f"{weighted_test_metrics['R2']:.4f}"
)
print(
    f"Spearman : "
    f"{weighted_test_metrics['Spearman']:.4f}"
)


# ============================================================
# MODEL 3 - LOG TARGET HGB
# ============================================================

print("\n" + "=" * 70)
print("MODEL 3 - LOG TARGET HISTGRADIENTBOOSTING")
print("=" * 70)


log_model = build_regressor()

y_train_log = np.log1p(
    y_train
)

log_model.fit(
    X_train,
    y_train_log,
)


log_val_pred = np.expm1(
    log_model.predict(X_val)
)

log_test_pred = np.expm1(
    log_model.predict(X_test)
)


# Prevent impossible negative predictions

log_val_pred = np.maximum(
    log_val_pred,
    0,
)

log_test_pred = np.maximum(
    log_test_pred,
    0,
)


log_val_metrics = (
    calculate_regression_metrics(
        y_val,
        log_val_pred,
    )
)

log_test_metrics = (
    calculate_regression_metrics(
        y_test,
        log_test_pred,
    )
)


print("\nValidation:")
print(
    f"MAE      : "
    f"{log_val_metrics['MAE']:.4f}"
)
print(
    f"RMSE     : "
    f"{log_val_metrics['RMSE']:.4f}"
)
print(
    f"R2       : "
    f"{log_val_metrics['R2']:.4f}"
)
print(
    f"Spearman : "
    f"{log_val_metrics['Spearman']:.4f}"
)


print("\nTest:")
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


# ============================================================
# MODEL 4 - TWO-STAGE TAIL MODEL
# ============================================================

print("\n" + "=" * 70)
print("MODEL 4 - TWO-STAGE TAIL MODEL")
print("=" * 70)


# ------------------------------------------------------------
# STAGE 1
# LOW vs TAIL
# ------------------------------------------------------------

print("\nStage 1: Tail classification")

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


# ------------------------------------------------------------
# CLASSIFIER PERFORMANCE
# ------------------------------------------------------------

# We use 0.50 as the initial classification threshold.

classification_threshold = 0.50

val_tail_class = (
    val_tail_probability
    >= classification_threshold
).astype(int)

test_tail_class = (
    test_tail_probability
    >= classification_threshold
).astype(int)


val_precision = precision_score(
    y_val_tail,
    val_tail_class,
    zero_division=0,
)

val_recall = recall_score(
    y_val_tail,
    val_tail_class,
    zero_division=0,
)

val_f1 = f1_score(
    y_val_tail,
    val_tail_class,
    zero_division=0,
)


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


print("\nTail classification - Validation:")

print(
    f"Precision : {val_precision:.4f}"
)

print(
    f"Recall    : {val_recall:.4f}"
)

print(
    f"F1        : {val_f1:.4f}"
)


print("\nTail classification - Test:")

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1        : {test_f1:.4f}"
)


print("\nValidation confusion matrix:")

print(
    confusion_matrix(
        y_val_tail,
        val_tail_class,
    )
)


print("\nTest confusion matrix:")

print(
    confusion_matrix(
        y_test_tail,
        test_tail_class,
    )
)


# ------------------------------------------------------------
# STAGE 2A - LOW REGRESSOR
# ------------------------------------------------------------

print("\nStage 2A: LOW-impact regression")

low_train_mask = (
    y_train < 30
)

X_train_low = X_train.loc[
    low_train_mask
]

y_train_low = y_train.loc[
    low_train_mask
]


low_regressor = build_regressor()

low_regressor.fit(
    X_train_low,
    y_train_low,
)


# ------------------------------------------------------------
# STAGE 2B - TAIL REGRESSOR
# ------------------------------------------------------------

print("Stage 2B: MODERATE/HIGH regression")

tail_train_mask = (
    y_train >= 30
)

X_train_tail = X_train.loc[
    tail_train_mask
]

y_train_tail_reg = y_train.loc[
    tail_train_mask
]


print(
    f"Tail training rows: "
    f"{len(X_train_tail):,}"
)


tail_regressor = build_regressor()

tail_regressor.fit(
    X_train_tail,
    y_train_tail_reg,
)


# ------------------------------------------------------------
# PREDICT LOW / TAIL VALUES
# ------------------------------------------------------------

val_low_pred = (
    low_regressor.predict(X_val)
)

test_low_pred = (
    low_regressor.predict(X_test)
)


val_tail_pred_value = (
    tail_regressor.predict(X_val)
)

test_tail_pred_value = (
    tail_regressor.predict(X_test)
)


# ------------------------------------------------------------
# HARD TWO-STAGE PREDICTION
# ------------------------------------------------------------

two_stage_val_pred = np.where(
    val_tail_probability
    >= classification_threshold,

    val_tail_pred_value,

    val_low_pred,
)


two_stage_test_pred = np.where(
    test_tail_probability
    >= classification_threshold,

    test_tail_pred_value,

    test_low_pred,
)


# Ensure non-negative impact scores

two_stage_val_pred = np.maximum(
    two_stage_val_pred,
    0,
)

two_stage_test_pred = np.maximum(
    two_stage_test_pred,
    0,
)


two_stage_val_metrics = (
    calculate_regression_metrics(
        y_val,
        two_stage_val_pred,
    )
)

two_stage_test_metrics = (
    calculate_regression_metrics(
        y_test,
        two_stage_test_pred,
    )
)


print("\nTwo-stage regression - Validation:")

print(
    f"MAE      : "
    f"{two_stage_val_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{two_stage_val_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{two_stage_val_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{two_stage_val_metrics['Spearman']:.4f}"
)


print("\nTwo-stage regression - Test:")

print(
    f"MAE      : "
    f"{two_stage_test_metrics['MAE']:.4f}"
)

print(
    f"RMSE     : "
    f"{two_stage_test_metrics['RMSE']:.4f}"
)

print(
    f"R2       : "
    f"{two_stage_test_metrics['R2']:.4f}"
)

print(
    f"Spearman : "
    f"{two_stage_test_metrics['Spearman']:.4f}"
)


# ============================================================
# TAIL PERFORMANCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TAIL PERFORMANCE COMPARISON")
print("=" * 70)


baseline_tail = calculate_tail_metrics(
    y_test,
    baseline_test_pred,
)

weighted_tail = calculate_tail_metrics(
    y_test,
    weighted_test_pred,
)

log_tail = calculate_tail_metrics(
    y_test,
    log_test_pred,
)

two_stage_tail = calculate_tail_metrics(
    y_test,
    two_stage_test_pred,
)


tail_comparison = pd.DataFrame(
    [
        {
            "model": "Baseline HGB",
            **baseline_tail,
        },

        {
            "model": "Weighted HGB",
            **weighted_tail,
        },

        {
            "model": "Log Target HGB",
            **log_tail,
        },

        {
            "model": "Two-Stage HGB",
            **two_stage_tail,
        },
    ]
)


print(
    tail_comparison.to_string(
        index=False
    )
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


comparison_rows = []


comparison_rows.append(
    {
        "model": "Baseline HGB",
        "validation_MAE":
            baseline_val_metrics["MAE"],
        "validation_RMSE":
            baseline_val_metrics["RMSE"],
        "validation_R2":
            baseline_val_metrics["R2"],
        "validation_Spearman":
            baseline_val_metrics["Spearman"],
        "test_MAE":
            baseline_test_metrics["MAE"],
        "test_RMSE":
            baseline_test_metrics["RMSE"],
        "test_R2":
            baseline_test_metrics["R2"],
        "test_Spearman":
            baseline_test_metrics["Spearman"],
    }
)


comparison_rows.append(
    {
        "model": "Weighted HGB",
        "validation_MAE":
            weighted_val_metrics["MAE"],
        "validation_RMSE":
            weighted_val_metrics["RMSE"],
        "validation_R2":
            weighted_val_metrics["R2"],
        "validation_Spearman":
            weighted_val_metrics["Spearman"],
        "test_MAE":
            weighted_test_metrics["MAE"],
        "test_RMSE":
            weighted_test_metrics["RMSE"],
        "test_R2":
            weighted_test_metrics["R2"],
        "test_Spearman":
            weighted_test_metrics["Spearman"],
    }
)


comparison_rows.append(
    {
        "model": "Log Target HGB",
        "validation_MAE":
            log_val_metrics["MAE"],
        "validation_RMSE":
            log_val_metrics["RMSE"],
        "validation_R2":
            log_val_metrics["R2"],
        "validation_Spearman":
            log_val_metrics["Spearman"],
        "test_MAE":
            log_test_metrics["MAE"],
        "test_RMSE":
            log_test_metrics["RMSE"],
        "test_R2":
            log_test_metrics["R2"],
        "test_Spearman":
            log_test_metrics["Spearman"],
    }
)


comparison_rows.append(
    {
        "model": "Two-Stage HGB",
        "validation_MAE":
            two_stage_val_metrics["MAE"],
        "validation_RMSE":
            two_stage_val_metrics["RMSE"],
        "validation_R2":
            two_stage_val_metrics["R2"],
        "validation_Spearman":
            two_stage_val_metrics["Spearman"],
        "test_MAE":
            two_stage_test_metrics["MAE"],
        "test_RMSE":
            two_stage_test_metrics["RMSE"],
        "test_R2":
            two_stage_test_metrics["R2"],
        "test_Spearman":
            two_stage_test_metrics["Spearman"],
    }
)


model_comparison = pd.DataFrame(
    comparison_rows
)


print(
    model_comparison
    .round(4)
    .to_string(index=False)
)


# ============================================================
# WORST / TAIL TEST EVENTS
# ============================================================

print("\n" + "=" * 70)
print("TEST TAIL EVENTS")
print("=" * 70)


tail_test_results = df_test.copy()


tail_test_results[
    "baseline_prediction"
] = baseline_test_pred


tail_test_results[
    "weighted_prediction"
] = weighted_test_pred


tail_test_results[
    "log_prediction"
] = log_test_pred


tail_test_results[
    "two_stage_prediction"
] = two_stage_test_pred


tail_test_results[
    "tail_probability"
] = test_tail_probability


tail_test_results[
    "baseline_abs_error"
] = np.abs(
    tail_test_results["impact_score"]
    - tail_test_results["baseline_prediction"]
)


tail_test_results[
    "weighted_abs_error"
] = np.abs(
    tail_test_results["impact_score"]
    - tail_test_results["weighted_prediction"]
)


tail_test_results[
    "log_abs_error"
] = np.abs(
    tail_test_results["impact_score"]
    - tail_test_results["log_prediction"]
)


tail_test_results[
    "two_stage_abs_error"
] = np.abs(
    tail_test_results["impact_score"]
    - tail_test_results["two_stage_prediction"]
)


tail_test_results = tail_test_results[
    tail_test_results["tail_flag"] == 1
].copy()


print(
    f"\nTail events in test set: "
    f"{len(tail_test_results)}"
)


if len(tail_test_results) > 0:

    print(
        tail_test_results[
            [
                "year",
                "disaster_type",
                "district",
                "province",
                "impact_score",
                "impact_band_analysis",
                "tail_probability",
                "baseline_prediction",
                "weighted_prediction",
                "log_prediction",
                "two_stage_prediction",
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
        "No MODERATE/HIGH events exist "
        "in the test set."
    )


# ============================================================
# BAND DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("IMPACT BAND DISTRIBUTION")
print("=" * 70)


for split_name, split_df in [
    ("TRAIN", df_train),
    ("VALIDATION", df_val),
    ("TEST", df_test),
]:

    print(f"\n{split_name}:")

    distribution = (
        split_df[
            "impact_band_analysis"
        ]
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
        distribution.to_string()
    )


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d15_model_comparison.csv",
)

tail_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d15_tail_performance.csv",
)

tail_events_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d15_tail_test_events.csv",
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d15_all_test_predictions.csv",
)

classifier_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d15_tail_classifier_results.csv",
)


model_comparison.to_csv(
    comparison_path,
    index=False,
)

tail_comparison.to_csv(
    tail_path,
    index=False,
)

tail_test_results.to_csv(
    tail_events_path,
    index=False,
)


# ============================================================
# SAVE ALL TEST PREDICTIONS
# ============================================================

all_predictions = df_test.copy()

all_predictions[
    "baseline_prediction"
] = baseline_test_pred

all_predictions[
    "weighted_prediction"
] = weighted_test_pred

all_predictions[
    "log_prediction"
] = log_test_pred

all_predictions[
    "two_stage_prediction"
] = two_stage_test_pred

all_predictions[
    "tail_probability"
] = test_tail_probability

all_predictions[
    "actual_band"
] = all_predictions[
    "impact_score"
].apply(classify_impact_band)

all_predictions.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# SAVE CLASSIFIER RESULTS
# ============================================================

classifier_results = pd.DataFrame(
    {
        "actual_tail": y_test_tail.values,

        "tail_probability":
            test_tail_probability,

        "predicted_tail":
            test_tail_class,
    }
)


classifier_results.to_csv(
    classifier_path,
    index=False,
)


# ============================================================
# DETERMINE CANDIDATE WINNER
# ============================================================

# Primary selection metric:
# validation R2
#
# Secondary:
# test R2
#
# We do NOT automatically select a production model
# solely from the test score.

winner_row = (
    model_comparison
    .sort_values(
        [
            "validation_R2",
            "test_R2",
        ],
        ascending=False,
    )
    .iloc[0]
)


candidate_winner = winner_row[
    "model"
]


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5D.15 COMPLETE")
print("=" * 70)


print("\nCandidate based primarily on validation R²:")

print(
    candidate_winner
)


print("\nOutput files:")

print(
    f"1. {comparison_path}"
)

print(
    f"2. {tail_path}"
)

print(
    f"3. {tail_events_path}"
)

print(
    f"4. {predictions_path}"
)

print(
    f"5. {classifier_path}"
)


print("\nImportant:")
print(
    "HIGH-impact events are absent from the "
    "2010-2013 validation/test periods."
)

print(
    "Therefore HIGH-risk generalization cannot "
    "be reliably established from this chronological split."
)

print("\nPhase 5D.15 finished successfully.")