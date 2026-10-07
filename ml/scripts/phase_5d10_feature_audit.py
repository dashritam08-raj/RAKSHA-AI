import pandas as pd
from pathlib import Path


INPUT = Path(
    "data/processed/nepal_ml_supervised_dataset.csv"
)


def main():

    print("=" * 70)
    print("PHASE 5D.10 - LEAKAGE & FEATURE AUDIT")
    print("=" * 70)

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nALL COLUMNS:")
    for i, column in enumerate(df.columns, 1):
        print(f"{i:02d}. {column}")

    # ---------------------------------------------------------
    # TARGET COLUMNS
    # ---------------------------------------------------------

    target_columns = {
        "impact_score",
        "impact_band",
        "impact_band_clean"
    }

    # ---------------------------------------------------------
    # POST-EVENT / OUTCOME COLUMNS
    #
    # These must NOT be used to predict disaster impact.
    # ---------------------------------------------------------

    leakage_columns = {
        "deaths",
        "injured",
        "economic_loss_usd",
        "economic_loss_local",
        "houses_damaged",
        "road_damage_m",
        "directly_affected",
        "indirectly_affected",
        "evacuated",
        "relocated"
    }

    # ---------------------------------------------------------
    # IDENTIFIER / ADMINISTRATIVE COLUMNS
    # ---------------------------------------------------------

    identifier_columns = {
        "incident_id",
        "missing",
        "observed_at",
        "event_date"
    }

    # ---------------------------------------------------------
    # Potential geographic identifiers
    # ---------------------------------------------------------

    geographic_columns = {
        "district",
        "province",
        "latitude",
        "longitude"
    }

    # ---------------------------------------------------------
    # Contextual disaster information
    # ---------------------------------------------------------

    context_columns = {
        "disaster_type"
    }

    excluded = (
        target_columns
        | leakage_columns
        | identifier_columns
    )

    candidate_features = [
        c for c in df.columns
        if c not in excluded
    ]

    print("\n" + "=" * 70)
    print("POST-EVENT / LEAKAGE COLUMNS")
    print("=" * 70)

    for column in sorted(leakage_columns):
        if column in df.columns:
            print("EXCLUDE:", column)

    print("\n" + "=" * 70)
    print("TARGET COLUMNS")
    print("=" * 70)

    for column in sorted(target_columns):
        if column in df.columns:
            print("TARGET:", column)

    print("\n" + "=" * 70)
    print("CANDIDATE PREDICTIVE FEATURES")
    print("=" * 70)

    for i, column in enumerate(candidate_features, 1):

        coverage = (
            df[column]
            .notna()
            .mean()
            * 100
        )

        print(
            f"{i:02d}. "
            f"{column:40s} "
            f"coverage={coverage:6.2f}%"
        )

    print("\n" + "=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print("""
The following outcome variables must NOT be used as
model inputs:

    deaths
    injured
    economic_loss_usd
    economic_loss_local
    houses_damaged
    road_damage_m
    directly_affected
    indirectly_affected
    evacuated
    relocated

They describe what happened AFTER the disaster and
would cause target leakage.

Next phase will use only pre-event/context features.
""")

    print("=" * 70)
    print("PHASE 5D.10 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    