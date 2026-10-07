import os
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# PHASE 5D.14
# FEATURE IMPORTANCE + ERROR ANALYSIS
# ============================================================

print("=" * 70)
print("PHASE 5D.14 - FEATURE IMPORTANCE + ERROR ANALYSIS")
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
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Required columns are missing:")
    for col in missing_columns:
        print(f" - {col}")

    raise ValueError(
        "Dataset does not contain all required columns."
    )


# ============================================================
# CREATE X AND y
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()


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
print(f"Train      : {len(X_train):,} rows (<= 2009)")
print(f"Validation : {len(X_val):,} rows (2010-2011)")
print(f"Test       : {len(X_test):,} rows (>= 2012)")


# ============================================================
# IMPACT BAND FUNCTION
# ============================================================

def classify_impact_band(score):

    if score < 30:
        return "LOW"

    elif score <= 55:
        return "MODERATE"

    else:
        return "HIGH"


# Create analysis-only impact bands.
#
# IMPORTANT:
# impact_band is NOT used as a model feature.
# It is created only for error analysis.

df_train["impact_band_analysis"] = (
    df_train["impact_score"]
    .apply(classify_impact_band)
)

df_val["impact_band_analysis"] = (
    df_val["impact_score"]
    .apply(classify_impact_band)
)

df_test["impact_band_analysis"] = (
    df_test["impact_score"]
    .apply(classify_impact_band)
)


# ============================================================
# PREPROCESSING
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


# ============================================================
# HISTGRADIENTBOOSTING MODEL
# ============================================================

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
            preprocessor,
        ),

        (
            "model",
            model,
        ),
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining HistGradientBoostingRegressor...")

pipeline.fit(
    X_train,
    y_train,
)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

val_pred = pipeline.predict(X_val)

test_pred = pipeline.predict(X_test)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

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

    return (
        mae,
        rmse,
        r2,
        spearman,
    )


# ============================================================
# VALIDATION METRICS
# ============================================================

(
    val_mae,
    val_rmse,
    val_r2,
    val_spearman,
) = calculate_metrics(
    y_val,
    val_pred,
)


# ============================================================
# TEST METRICS
# ============================================================

(
    test_mae,
    test_rmse,
    test_r2,
    test_spearman,
) = calculate_metrics(
    y_test,
    test_pred,
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)


print("\nValidation:")

print(f"MAE       : {val_mae:.4f}")
print(f"RMSE      : {val_rmse:.4f}")
print(f"R²        : {val_r2:.4f}")
print(f"Spearman  : {val_spearman:.4f}")


print("\nTest:")

print(f"MAE       : {test_mae:.4f}")
print(f"RMSE      : {test_rmse:.4f}")
print(f"R²        : {test_r2:.4f}")
print(f"Spearman  : {test_spearman:.4f}")


# ============================================================
# ADD PREDICTIONS
# ============================================================

df_val["predicted_impact"] = val_pred

df_val["absolute_error"] = np.abs(
    df_val["impact_score"]
    - df_val["predicted_impact"]
)

df_val["signed_error"] = (
    df_val["predicted_impact"]
    - df_val["impact_score"]
)


df_test["predicted_impact"] = test_pred

df_test["absolute_error"] = np.abs(
    df_test["impact_score"]
    - df_test["predicted_impact"]
)

df_test["signed_error"] = (
    df_test["predicted_impact"]
    - df_test["impact_score"]
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING FEATURE IMPORTANCE")
print("=" * 70)


print(
    "\nUsing permutation importance on VALIDATION data."
)


permutation = permutation_importance(
    pipeline,
    X_val,
    y_val,

    scoring="neg_mean_absolute_error",

    n_repeats=5,

    random_state=42,

    n_jobs=-1,
)


importance_df = pd.DataFrame(
    {
        "feature": FEATURES,

        "importance_mean":
            permutation.importances_mean,

        "importance_std":
            permutation.importances_std,
    }
)


importance_df = (
    importance_df
    .sort_values(
        "importance_mean",
        ascending=False,
    )
    .reset_index(drop=True)
)


importance_df["rank"] = (
    np.arange(
        len(importance_df)
    )
    + 1
)


print("\nTop 15 features:")


print(
    importance_df[
        [
            "rank",
            "feature",
            "importance_mean",
            "importance_std",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# ERROR ANALYSIS BY DISASTER TYPE
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS BY DISASTER TYPE")
print("=" * 70)


disaster_error = (
    df_test
    .groupby("disaster_type")
    .agg(
        events=(
            "impact_score",
            "size",
        ),

        actual_mean=(
            "impact_score",
            "mean",
        ),

        predicted_mean=(
            "predicted_impact",
            "mean",
        ),

        mae=(
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
        "mae",
        ascending=False,
    )
)


print("\nWorst disaster types by MAE:")


print(
    disaster_error
    .head(15)
    .to_string(index=False)
)


# ============================================================
# ERROR ANALYSIS BY IMPACT BAND
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS BY IMPACT BAND")
print("=" * 70)


band_error = (
    df_test
    .groupby("impact_band_analysis")
    .agg(
        events=(
            "impact_score",
            "size",
        ),

        actual_mean=(
            "impact_score",
            "mean",
        ),

        predicted_mean=(
            "predicted_impact",
            "mean",
        ),

        mae=(
            "absolute_error",
            "mean",
        ),

        bias=(
            "signed_error",
            "mean",
        ),
    )
    .reset_index()
    .rename(
        columns={
            "impact_band_analysis":
                "impact_band"
        }
    )
)


# Keep logical order

band_order = [
    "LOW",
    "MODERATE",
    "HIGH",
]


band_error["impact_band"] = pd.Categorical(
    band_error["impact_band"],
    categories=band_order,
    ordered=True,
)


band_error = (
    band_error
    .sort_values("impact_band")
)


print("\nImpact-band performance:")


print(
    band_error
    .to_string(index=False)
)


# ============================================================
# WORST TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("WORST TEST PREDICTIONS")
print("=" * 70)


# incident_id is NOT available in
# nepal_ml_features_v1.csv.
#
# Therefore we use the available fields.

worst_predictions = (
    df_test
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
            "impact_score",
            "predicted_impact",
            "absolute_error",
            "signed_error",
            "impact_band_analysis",
        ]
    ]
    .head(20)
)


print(
    worst_predictions
    .to_string(index=False)
)


# ============================================================
# HIGH-IMPACT EVENTS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-IMPACT EVENTS IN TEST SET")
print("=" * 70)


high_test = df_test[
    df_test["impact_band_analysis"]
    == "HIGH"
]


print(
    f"\nHIGH events in test set: "
    f"{len(high_test)}"
)


if len(high_test) > 0:

    print(
        high_test[
            [
                "year",
                "disaster_type",
                "district",
                "province",
                "impact_score",
                "predicted_impact",
                "absolute_error",
            ]
        ]
        .to_string(index=False)
    )

else:

    print(
        "No HIGH-impact events exist "
        "in the 2012-2013 test period."
    )


# ============================================================
# TARGET / BAND DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("IMPACT BAND DISTRIBUTION BY SPLIT")
print("=" * 70)


def band_distribution(
    data,
    name,
):

    bands = (
        data["impact_score"]
        .apply(classify_impact_band)
    )

    counts = (
        bands
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

    print(f"\n{name}:")

    print(
        counts.to_string()
    )


band_distribution(
    df_train,
    "TRAIN",
)


band_distribution(
    df_val,
    "VALIDATION",
)


band_distribution(
    df_test,
    "TEST",
)


# ============================================================
# ERROR SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)


print(
    f"\nValidation MAE : "
    f"{val_mae:.4f}"
)

print(
    f"Validation RMSE: "
    f"{val_rmse:.4f}"
)

print(
    f"Validation R²  : "
    f"{val_r2:.4f}"
)

print(
    f"Validation Spearman: "
    f"{val_spearman:.4f}"
)


print(
    f"\nTest MAE       : "
    f"{test_mae:.4f}"
)

print(
    f"Test RMSE      : "
    f"{test_rmse:.4f}"
)

print(
    f"Test R²        : "
    f"{test_r2:.4f}"
)

print(
    f"Test Spearman  : "
    f"{test_spearman:.4f}"
)


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

importance_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d14_feature_importance.csv",
)


disaster_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d14_error_by_disaster_type.csv",
)


band_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d14_error_by_impact_band.csv",
)


worst_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d14_worst_predictions.csv",
)


validation_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d14_validation_predictions.csv",
)


test_path = os.path.join(
    OUTPUT_DIR,
    "phase_5d14_test_predictions.csv",
)


# Save feature importance

importance_df.to_csv(
    importance_path,
    index=False,
)


# Save disaster error

disaster_error.to_csv(
    disaster_path,
    index=False,
)


# Save band error

band_error.to_csv(
    band_path,
    index=False,
)


# Save worst predictions

worst_predictions.to_csv(
    worst_path,
    index=False,
)


# Save validation predictions

df_val.to_csv(
    validation_path,
    index=False,
)


# Save test predictions

df_test.to_csv(
    test_path,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5D.14 COMPLETE")
print("=" * 70)


print("\nSaved files:")

print(
    f"1. {importance_path}"
)

print(
    f"2. {disaster_path}"
)

print(
    f"3. {band_path}"
)

print(
    f"4. {worst_path}"
)

print(
    f"5. {validation_path}"
)

print(
    f"6. {test_path}"
)


print("\nCurrent candidate model:")
print(
    "HistGradientBoostingRegressor"
)


print("\nPhase 5D.14 finished successfully.")