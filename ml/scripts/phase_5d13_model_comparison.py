import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import spearmanr


INPUT = Path(
    "data/processed/nepal_ml_features_v1.csv"
)


def evaluate(name, model, X, y):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    spearman = spearmanr(
        y,
        predictions
    ).statistic

    print(
        f"{name:25s}"
        f" MAE={mae:7.4f}"
        f" RMSE={rmse:7.4f}"
        f" R2={r2:8.4f}"
        f" Spearman={spearman:7.4f}"
    )

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman
    }


def main():

    print("=" * 70)
    print("PHASE 5D.13 - MODEL COMPARISON")
    print("=" * 70)

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    df = df.sort_values(
        "year"
    ).reset_index(
        drop=True
    )

    target = "impact_score"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # ---------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------

    train_mask = df["year"] <= 2009

    validation_mask = (
        (df["year"] >= 2010)
        &
        (df["year"] <= 2011)
    )

    test_mask = df["year"] >= 2012

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_val = X.loc[validation_mask]
    y_val = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print(
        f"\nTrain:      {len(X_train):,}"
    )

    print(
        f"Validation: {len(X_val):,}"
    )

    print(
        f"Test:       {len(X_test):,}"
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

    # ---------------------------------------------------------
    # Preprocessor
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
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

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
    # Models
    # ---------------------------------------------------------

    models = {

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
    }

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    results = []

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 70
    )

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        print(
            f"\nTraining {name}..."
        )

        pipeline.fit(
            X_train,
            y_train
        )

        result = evaluate(
            name,
            pipeline,
            X_val,
            y_val
        )

        result["split"] = "validation"

        results.append(result)

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 70
    )

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        print(
            f"\nTraining {name}..."
        )

        pipeline.fit(
            X_train,
            y_train
        )

        result = evaluate(
            name,
            pipeline,
            X_test,
            y_test
        )

        result["split"] = "test"

        results.append(result)

    # ---------------------------------------------------------
    # Mean baseline
    # ---------------------------------------------------------

    mean_prediction = np.full(
        len(y_test),
        y_train.mean()
    )

    baseline_r2 = r2_score(
        y_test,
        mean_prediction
    )

    baseline_mae = mean_absolute_error(
        y_test,
        mean_prediction
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            mean_prediction
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MEAN BASELINE - TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"MAE       : {baseline_mae:.4f}"
    )

    print(
        f"RMSE      : {baseline_rmse:.4f}"
    )

    print(
        f"R²        : {baseline_r2:.4f}"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    output = Path(
        "data/processed/phase_5d13_model_comparison.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print(
        f"\nResults saved:"
    )

    print(
        output
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 5D.13 COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()