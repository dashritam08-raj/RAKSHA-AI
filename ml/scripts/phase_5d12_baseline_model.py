import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import spearmanr


INPUT = Path(
    "data/processed/nepal_ml_features_v1.csv"
)

MODEL_OUTPUT = Path(
    "models/phase_5d12_baseline_rf.joblib"
)


def main():

    print("=" * 70)
    print("PHASE 5D.12 - BASELINE REGRESSION MODEL")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    print(
        f"\nDataset rows: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    target = "impact_score"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # ---------------------------------------------------------
    # IMPORTANT:
    # Convert year to numeric and sort chronologically
    # ---------------------------------------------------------

    df_sorted = df.sort_values(
        "year"
    ).reset_index(
        drop=True
    )

    X = df_sorted.drop(
        columns=[target]
    )

    y = df_sorted[target]

    # ---------------------------------------------------------
    # Chronological split
    #
    # Train: <= 2009
    # Validation: 2010-2011
    # Test: >= 2012
    # ---------------------------------------------------------

    train_mask = (
        df_sorted["year"] <= 2009
    )

    validation_mask = (
        (df_sorted["year"] >= 2010)
        &
        (df_sorted["year"] <= 2011)
    )

    test_mask = (
        df_sorted["year"] >= 2012
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_val = X.loc[validation_mask]
    y_val = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print("\nChronological split:")

    print(
        f"Train      : {len(X_train):,}"
    )

    print(
        f"Validation : {len(X_val):,}"
    )

    print(
        f"Test       : {len(X_test):,}"
    )

    print(
        f"\nTrain years      : "
        f"{df_sorted.loc[train_mask, 'year'].min():.0f}"
        f" - "
        f"{df_sorted.loc[train_mask, 'year'].max():.0f}"
    )

    print(
        f"Validation years : "
        f"{df_sorted.loc[validation_mask, 'year'].min():.0f}"
        f" - "
        f"{df_sorted.loc[validation_mask, 'year'].max():.0f}"
    )

    print(
        f"Test years       : "
        f"{df_sorted.loc[test_mask, 'year'].min():.0f}"
        f" - "
        f"{df_sorted.loc[test_mask, 'year'].max():.0f}"
    )

    # ---------------------------------------------------------
    # Feature groups
    # ---------------------------------------------------------

    categorical_features = [
        "district",
        "province",
        "disaster_type"
    ]

    numerical_features = [
        c for c in X.columns
        if c not in categorical_features
    ]

    print(
        f"\nCategorical features: "
        f"{len(categorical_features)}"
    )

    print(
        f"Numerical features: "
        f"{len(numerical_features)}"
    )

    # ---------------------------------------------------------
    # Numerical preprocessing
    # ---------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # ---------------------------------------------------------
    # Categorical preprocessing
    # ---------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # ---------------------------------------------------------
    # Preprocessor
    # ---------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # ---------------------------------------------------------
    # Random Forest baseline
    # ---------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # Evaluation function
    # ---------------------------------------------------------

    def evaluate(
        name,
        X_eval,
        y_eval
    ):

        predictions = pipeline.predict(
            X_eval
        )

        mae = mean_absolute_error(
            y_eval,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_eval,
                predictions
            )
        )

        r2 = r2_score(
            y_eval,
            predictions
        )

        spearman = spearmanr(
            y_eval,
            predictions
        ).statistic

        print(
            f"\n{name}"
        )

        print(
            f"MAE       : {mae:.4f}"
        )

        print(
            f"RMSE      : {rmse:.4f}"
        )

        print(
            f"R²        : {r2:.4f}"
        )

        print(
            f"Spearman  : {spearman:.4f}"
        )

        return predictions

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    val_predictions = evaluate(
        "VALIDATION RESULTS",
        X_val,
        y_val
    )

    test_predictions = evaluate(
        "TEST RESULTS",
        X_test,
        y_test
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    import joblib

    MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_OUTPUT
    )

    print(
        f"\nModel saved:"
    )

    print(
        MODEL_OUTPUT
    )

    print("\n" + "=" * 70)
    print("PHASE 5D.12 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()