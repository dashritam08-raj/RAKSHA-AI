import pandas as pd
from pathlib import Path


INPUT = Path(
    "data/processed/nepal_ml_supervised_dataset.csv"
)

OUTPUT = Path(
    "data/processed/nepal_ml_features_v1.csv"
)


def main():

    print("=" * 70)
    print("PHASE 5D.11 - LEAKAGE-SAFE FEATURE DATASET")
    print("=" * 70)

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    print(f"\nOriginal rows: {len(df):,}")

    # ---------------------------------------------------------
    # TARGET
    # ---------------------------------------------------------

    target = "impact_score"

    # ---------------------------------------------------------
    # POST-EVENT OUTCOME / LEAKAGE FEATURES
    # ---------------------------------------------------------

    leakage_columns = [
        "deaths",
        "missing",
        "injured",
        "directly_affected",
        "indirectly_affected",
        "evacuated",
        "relocated",
        "houses_destroyed",
        "houses_damaged",
        "road_damage_m",
        "economic_loss_usd",
        "economic_loss_local",
    ]

    # ---------------------------------------------------------
    # IDENTIFIERS / NON-PREDICTIVE METADATA
    # ---------------------------------------------------------

    metadata_columns = [
        "incident_id",
        "observed_at",
        "event_date",
        "geo_source",
        "impact_band",
        "impact_band_clean",
    ]

    # ---------------------------------------------------------
    # Explicit feature list
    # ---------------------------------------------------------

    feature_columns = [
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

    # ---------------------------------------------------------
    # Verify columns
    # ---------------------------------------------------------

    missing_features = [
        c for c in feature_columns
        if c not in df.columns
    ]

    if missing_features:

        print("\nERROR: Missing feature columns:")

        for column in missing_features:
            print("  -", column)

        raise ValueError(
            "Required feature columns are missing."
        )

    # ---------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------

    selected_columns = (
        feature_columns
        + [target]
    )

    features = df[
        selected_columns
    ].copy()

    # ---------------------------------------------------------
    # Verify target
    # ---------------------------------------------------------

    features[target] = pd.to_numeric(
        features[target],
        errors="coerce"
    )

    before = len(features)

    features = features[
        features[target].notna()
    ].copy()

    removed = before - len(features)

    print(
        f"\nRows removed due to missing target: "
        f"{removed:,}"
    )

    # ---------------------------------------------------------
    # Show final columns
    # ---------------------------------------------------------

    print(
        f"\nFinal rows: {len(features):,}"
    )

    print(
        f"Feature count: {len(feature_columns):,}"
    )

    print("\nFeatures:")

    for i, column in enumerate(
        feature_columns,
        1
    ):
        print(
            f"{i:02d}. {column}"
        )

    # ---------------------------------------------------------
    # Missingness
    # ---------------------------------------------------------

    print("\nFeature missingness:")

    missingness = (
        features[feature_columns]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(
            ascending=False
        )
    )

    print(
        missingness.to_string()
    )

    # ---------------------------------------------------------
    # Target summary
    # ---------------------------------------------------------

    print("\nTarget summary:")

    print(
        features[target]
        .describe()
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)

    print("\n" + "=" * 70)
    print("PHASE 5D.11 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()