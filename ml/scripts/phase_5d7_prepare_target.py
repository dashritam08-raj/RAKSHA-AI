import pandas as pd
from pathlib import Path


INPUT = Path(
    "data/processed/nepal_geographic_training_dataset_historical.csv"
)

OUTPUT = Path(
    "data/processed/nepal_ml_labeled_dataset.csv"
)


def main():

    print("=" * 70)
    print("PHASE 5D.7 - TARGET CLEANING")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(INPUT, low_memory=False)

    print(f"\nOriginal rows: {len(df):,}")

    # Convert target to numeric
    df["impact_score"] = pd.to_numeric(
        df["impact_score"],
        errors="coerce"
    )

    # Keep only rows with a real target
    labeled = df[
        df["impact_score"].notna()
    ].copy()

    print(f"Labeled rows: {len(labeled):,}")
    print(f"Removed rows: {len(df) - len(labeled):,}")

    # Create clean impact bands
    def create_band(score):

        if score < 30:
            return "LOW"

        elif score < 55:
            return "MODERATE"

        else:
            return "HIGH"

    labeled["impact_band_clean"] = (
        labeled["impact_score"]
        .apply(create_band)
    )

    # Show distribution
    print("\nClean target distribution:")
    print(
        labeled["impact_band_clean"]
        .value_counts()
        .sort_index()
    )

    # Show statistics
    print("\nImpact score statistics:")
    print(
        labeled["impact_score"]
        .describe()
    )

    # Save cleaned dataset
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    labeled.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)

    print("\nDONE")


if __name__ == "__main__":
    main()