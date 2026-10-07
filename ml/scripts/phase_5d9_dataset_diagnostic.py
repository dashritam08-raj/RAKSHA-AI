import pandas as pd
from pathlib import Path


INPUT = Path(
    "data/processed/nepal_ml_supervised_dataset.csv"
)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():

    section("PHASE 5D.9 - DATASET DIAGNOSTIC")

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # ---------------------------------------------------------
    # 1. TARGET DISTRIBUTION
    # ---------------------------------------------------------

    section("1. TARGET DISTRIBUTION")

    print(
        df["impact_band"]
        .value_counts()
    )

    # ---------------------------------------------------------
    # 2. DISASTER TYPE
    # ---------------------------------------------------------

    section("2. DISASTER TYPE DISTRIBUTION")

    if "disaster_type" in df.columns:

        print(
            df["disaster_type"]
            .value_counts()
            .head(30)
            .to_string()
        )

    else:

        print("disaster_type column not found.")

    # ---------------------------------------------------------
    # 3. HIGH IMPACT EVENTS
    # ---------------------------------------------------------

    section("3. HIGH IMPACT EVENTS")

    high = df[
        df["impact_band"] == "HIGH"
    ].copy()

    print(f"HIGH events: {len(high)}")

    important_columns = [
        "date",
        "year",
        "disaster_type",
        "district",
        "province",
        "impact_score",
        "impact_band"
    ]

    available = [
        c for c in important_columns
        if c in high.columns
    ]

    if len(high) > 0:

        print(
            high[available]
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # 4. MODERATE EVENTS
    # ---------------------------------------------------------

    section("4. MODERATE EVENTS")

    moderate = df[
        df["impact_band"] == "MODERATE"
    ]

    print(
        f"MODERATE events: {len(moderate):,}"
    )

    if "disaster_type" in df.columns:

        print("\nModerate by disaster type:")

        print(
            moderate["disaster_type"]
            .value_counts()
            .to_string()
        )

    # ---------------------------------------------------------
    # 5. FEATURE COVERAGE
    # ---------------------------------------------------------

    section("5. FEATURE COVERAGE")

    candidate_features = [
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "precipitation_mm",
        "wind_max_kmh",
        "eq_count_7d_300km",
        "eq_max_magnitude_7d_300km"
    ]

    existing = [
        c for c in candidate_features
        if c in df.columns
    ]

    if existing:

        coverage = (
            df[existing]
            .notna()
            .mean()
            .mul(100)
            .round(2)
            .sort_values()
        )

        print(
            coverage.to_string()
        )

    else:

        print(
            "None of the expected weather/"
            "earthquake columns were found."
        )

    # ---------------------------------------------------------
    # 6. FEATURE COVERAGE BY TARGET
    # ---------------------------------------------------------

    section(
        "6. FEATURE COVERAGE BY IMPACT BAND"
    )

    if existing:

        coverage_by_band = (
            df.groupby("impact_band")[existing]
            .apply(
                lambda x: x.notna().mean() * 100
            )
            .round(2)
        )

        print(
            coverage_by_band.to_string()
        )

    # ---------------------------------------------------------
    # 7. YEAR DISTRIBUTION
    # ---------------------------------------------------------

    section("7. YEAR DISTRIBUTION")

    year_column = None

    if "year" in df.columns:

        year_column = "year"

    elif "date" in df.columns:

        df["_year"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dt.year

        year_column = "_year"

    if year_column:

        print(
            df[year_column]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nEvents by year and impact band:")

        print(
            pd.crosstab(
                df[year_column],
                df["impact_band"]
            ).to_string()
        )

    else:

        print("No year/date column found.")

    # ---------------------------------------------------------
    # 8. MISSINGNESS SUMMARY
    # ---------------------------------------------------------

    section("8. MISSINGNESS SUMMARY")

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print(
        missing.head(30)
        .to_string()
    )

    print("\n" + "=" * 70)
    print("PHASE 5D.9 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()