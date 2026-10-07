import pandas as pd
from pathlib import Path


INPUT = Path(
    "data/processed/nepal_geographic_training_dataset_historical.csv"
)

OUTPUT = Path(
    "data/processed/nepal_ml_supervised_dataset.csv"
)


def main():

    print("=" * 70)
    print("PHASE 5D.8 - SUPERVISED ML DATASET")
    print("=" * 70)

    df = pd.read_csv(INPUT, low_memory=False)

    print(f"\nOriginal rows: {len(df):,}")

    # Convert impact score to numeric
    df["impact_score"] = pd.to_numeric(
        df["impact_score"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Only use records whose ORIGINAL impact_band is known.
    #
    # NO_DATA = no reliable target → remove
    # NaN     = no reliable target → remove
    # ---------------------------------------------------------

    valid_bands = [
        "LOW",
        "MODERATE",
        "HIGH"
    ]

    ml_df = df[
        df["impact_band"]
        .isin(valid_bands)
    ].copy()

    print(f"Supervised rows: {len(ml_df):,}")
    print(
        f"Removed rows: "
        f"{len(df) - len(ml_df):,}"
    )

    # ---------------------------------------------------------
    # Verify that every supervised record has a score
    # ---------------------------------------------------------

    missing_scores = ml_df["impact_score"].isna().sum()

    print(
        f"\nSupervised rows with missing "
        f"impact_score: {missing_scores}"
    )

    if missing_scores > 0:

        print(
            "\nWARNING: Removing rows with "
            "missing impact_score."
        )

        ml_df = ml_df[
            ml_df["impact_score"].notna()
        ].copy()

    # ---------------------------------------------------------
    # Distribution
    # ---------------------------------------------------------

    print("\nFinal target distribution:")

    distribution = (
        ml_df["impact_band"]
        .value_counts()
        .reindex(valid_bands)
        .fillna(0)
        .astype(int)
    )

    print(distribution)

    # Percentages
    print("\nTarget percentages:")

    percentages = (
        ml_df["impact_band"]
        .value_counts(normalize=True)
        .reindex(valid_bands)
        .fillna(0)
        * 100
    )

    for band, percentage in percentages.items():

        print(
            f"{band:10s}: "
            f"{percentage:.2f}%"
        )

    # ---------------------------------------------------------
    # Impact score statistics by band
    # ---------------------------------------------------------

    print("\nImpact score by band:")

    stats = (
        ml_df
        .groupby("impact_band")["impact_score"]
        .agg(
            count="count",
            min="min",
            median="median",
            mean="mean",
            max="max"
        )
        .reindex(valid_bands)
    )

    print(stats)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ml_df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)

    print("\nPHASE 5D.8 COMPLETE")


if __name__ == "__main__":
    main()