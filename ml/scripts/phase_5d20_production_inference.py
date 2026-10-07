import argparse
import json
import math
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PHASE 5D.20
# RAKSHA AI - PRODUCTION INFERENCE PIPELINE
#
# Production model:
#   70% Log-Target HGB
#   30% Shallow HGB
#
# Training data:
#   <= 2011
#
# Final historical test:
#   2012-2013
#
# IMPORTANT:
#   No post-event outcome features are accepted.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"

SHALLOW_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "raksha_final_shallow_hgb.joblib",
)

LOG_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "raksha_final_log_target_hgb.joblib",
)


MODEL_VERSION = "RAKSHA-ML-v1"


LOG_WEIGHT = 0.70
SHALLOW_WEIGHT = 0.30


# ============================================================
# REQUIRED ML FEATURES
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
# POST-EVENT COLUMNS
# ============================================================

FORBIDDEN_POST_EVENT_FIELDS = [
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
    "impact_band",
    "impact_score",
]


# ============================================================
# IMPACT BAND
# ============================================================

def classify_impact_band(score):
    """
    Convert continuous impact score into
    RAKSHA screening bands.
    """

    if score < 30:
        return "LOW"

    elif score < 55:
        return "MODERATE"

    return "HIGH"


# ============================================================
# NUMBER HELPERS
# ============================================================

def clean_numeric(value):
    """
    Convert a value to float when possible.
    Return NaN for missing/invalid values.
    """

    if value is None:
        return np.nan

    if isinstance(value, str):

        value = value.strip()

        if value == "":
            return np.nan

    try:
        return float(value)

    except (TypeError, ValueError):
        return np.nan


def clean_code(value):
    """
    District/province were represented as numeric
    codes in the training dataset.

    Convert incoming numeric-like codes to floats.
    """

    if value is None:
        return np.nan

    if isinstance(value, str):

        value = value.strip()

        if value == "":
            return np.nan

    try:
        return float(value)

    except (TypeError, ValueError):

        # Preserve non-numeric values as strings.
        # This will be treated as an unknown category
        # by the trained OneHotEncoder.

        return str(value)


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

def add_date_features(
    payload,
):
    """
    Derive year/month/day-of-year/day-of-week
    and cyclic month features from event_date.

    event_date format:
        YYYY-MM-DD

    Example:
        2015-08-21
    """

    event_date = payload.get(
        "event_date"
    )

    if event_date:

        try:

            parsed_date = datetime.strptime(
                str(event_date),
                "%Y-%m-%d",
            )

        except ValueError as exc:

            raise ValueError(
                "event_date must use YYYY-MM-DD format."
            ) from exc

        payload["year"] = (
            parsed_date.year
        )

        payload["month"] = (
            parsed_date.month
        )

        payload["day_of_year"] = (
            parsed_date.timetuple().tm_yday
        )

        payload["day_of_week"] = (
            parsed_date.weekday()
        )

        payload["month_sin"] = math.sin(
            2
            * math.pi
            * parsed_date.month
            / 12.0
        )

        payload["month_cos"] = math.cos(
            2
            * math.pi
            * parsed_date.month
            / 12.0
        )

    else:

        # Allow explicit temporal fields when event_date
        # is not supplied.

        temporal_fields = [
            "year",
            "month",
            "day_of_year",
            "day_of_week",
            "month_sin",
            "month_cos",
        ]

        missing_temporal = [
            field
            for field in temporal_fields
            if field not in payload
        ]

        if missing_temporal:

            raise ValueError(
                "Provide event_date (YYYY-MM-DD) "
                "or all temporal ML fields: "
                + ", ".join(
                    temporal_fields
                )
            )

    return payload


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_features(payload):
    """
    Convert user-facing JSON into the exact feature
    dataframe expected by the trained models.
    """

    if not isinstance(
        payload,
        dict,
    ):

        raise TypeError(
            "Prediction input must be a JSON object/dictionary."
        )


    payload = payload.copy()


    # --------------------------------------------------------
    # Protect against accidental target leakage
    # --------------------------------------------------------

    forbidden_present = [
        field
        for field in FORBIDDEN_POST_EVENT_FIELDS
        if field in payload
    ]

    if forbidden_present:

        raise ValueError(
            "Post-event/target fields are not allowed "
            "during prediction: "
            + ", ".join(
                forbidden_present
            )
        )


    # --------------------------------------------------------
    # Derive date features
    # --------------------------------------------------------

    payload = add_date_features(
        payload
    )


    # --------------------------------------------------------
    # Geographic fields
    # --------------------------------------------------------

    payload["latitude"] = clean_numeric(
        payload.get(
            "latitude"
        )
    )

    payload["longitude"] = clean_numeric(
        payload.get(
            "longitude"
        )
    )

    payload["district"] = clean_code(
        payload.get(
            "district"
        )
    )

    payload["province"] = clean_code(
        payload.get(
            "province"
        )
    )


    # --------------------------------------------------------
    # Disaster type
    # --------------------------------------------------------

    disaster_type = payload.get(
        "disaster_type"
    )

    if disaster_type is None:

        raise ValueError(
            "disaster_type is required."
        )

    payload["disaster_type"] = (
        str(disaster_type)
        .strip()
        .upper()
    )


    # --------------------------------------------------------
    # Environmental / weather features
    # --------------------------------------------------------

    numeric_fields = [
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
    ]


    for field in numeric_fields:

        payload[field] = clean_numeric(
            payload.get(field)
        )


    # --------------------------------------------------------
    # Boolean/context fields
    # --------------------------------------------------------

    boolean_fields = [
        "has_coordinates",
        "has_weather",
        "has_earthquake_context",
    ]


    for field in boolean_fields:

        value = payload.get(
            field
        )

        if isinstance(
            value,
            bool,
        ):

            payload[field] = int(
                value
            )

        elif value is None:

            payload[field] = 0

        else:

            text_value = (
                str(value)
                .strip()
                .lower()
            )

            if text_value in [
                "true",
                "1",
                "yes",
                "y",
            ]:

                payload[field] = 1

            else:

                payload[field] = 0


    # --------------------------------------------------------
    # Temporal numeric fields
    # --------------------------------------------------------

    temporal_numeric_fields = [
        "year",
        "month",
        "day_of_year",
        "day_of_week",
        "month_sin",
        "month_cos",
    ]


    for field in temporal_numeric_fields:

        payload[field] = clean_numeric(
            payload.get(field)
        )


    # --------------------------------------------------------
    # Coordinate consistency
    # --------------------------------------------------------

    if (
        not pd.isna(
            payload["latitude"]
        )
        and not pd.isna(
            payload["longitude"]
        )
    ):

        payload["has_coordinates"] = 1


    # --------------------------------------------------------
    # Build exact feature row
    # --------------------------------------------------------

    feature_row = {}

    for feature in FEATURES:

        feature_row[feature] = (
            payload.get(feature)
        )


    X_new = pd.DataFrame(
        [feature_row],
        columns=FEATURES,
    )


    return X_new


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """
    Load the two final production model components.
    """

    if not os.path.exists(
        SHALLOW_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Missing model file: "
            f"{SHALLOW_MODEL_PATH}"
        )


    if not os.path.exists(
        LOG_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Missing model file: "
            f"{LOG_MODEL_PATH}"
        )


    shallow_model = joblib.load(
        SHALLOW_MODEL_PATH
    )

    log_model = joblib.load(
        LOG_MODEL_PATH
    )


    return (
        shallow_model,
        log_model,
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_impact(
    payload,
):
    """
    Main production prediction function.

    Returns:
        impact score
        risk band
        component predictions
        model metadata
    """

    X_new = prepare_features(
        payload
    )


    shallow_model, log_model = (
        load_models()
    )


    # --------------------------------------------------------
    # Shallow HGB prediction
    # --------------------------------------------------------

    shallow_prediction = float(
        shallow_model.predict(
            X_new
        )[0]
    )


    # --------------------------------------------------------
    # Log-target HGB prediction
    #
    # Model was trained on:
    #     log1p(impact_score)
    #
    # Convert back with:
    #     expm1()
    # --------------------------------------------------------

    log_prediction = float(
        np.expm1(
            log_model.predict(
                X_new
            )[0]
        )
    )


    log_prediction = max(
        0.0,
        log_prediction,
    )


    # --------------------------------------------------------
    # Final 70/30 blend
    # --------------------------------------------------------

    blend_prediction = (
        LOG_WEIGHT
        * log_prediction

        +

        SHALLOW_WEIGHT
        * shallow_prediction
    )


    blend_prediction = max(
        0.0,
        float(
            blend_prediction
        ),
    )


    # --------------------------------------------------------
    # Risk band
    # --------------------------------------------------------

    risk_band = classify_impact_band(
        blend_prediction
    )


    # --------------------------------------------------------
    # Input coverage
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if pd.isna(
            X_new.iloc[0][feature]
        )
    ]


    weather_features = [
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "precipitation_mm",
        "wind_max_kmh",
        "wind_gust_max_kmh",
    ]


    earthquake_features = [
        "eq_count_7d_300km",
        "eq_max_magnitude_7d_300km",
        "eq_nearest_km_7d_300km",
    ]


    weather_available = all(
        not pd.isna(
            X_new.iloc[0][field]
        )
        for field in weather_features
    )


    earthquake_available = any(
        not pd.isna(
            X_new.iloc[0][field]
        )
        for field in earthquake_features
    )


    # --------------------------------------------------------
    # Important production note
    # --------------------------------------------------------

    warning = (
        "Model-derived screening score. "
        "HIGH-impact generalization was not "
        "validated because the final historical "
        "test period contained zero HIGH events."
    )


    return {

        "model_version":
            MODEL_VERSION,

        "impact_score":
            round(
                blend_prediction,
                4,
            ),

        "risk_band":
            risk_band,

        "component_predictions": {

            "log_target_hgb":
                round(
                    log_prediction,
                    4,
                ),

            "shallow_hgb":
                round(
                    shallow_prediction,
                    4,
                ),
        },

        "blend_weights": {

            "log_target_hgb":
                LOG_WEIGHT,

            "shallow_hgb":
                SHALLOW_WEIGHT,
        },

        "input_quality": {

            "missing_feature_count":
                len(
                    missing_features
                ),

            "missing_features":
                missing_features,

            "weather_available":
                weather_available,

            "earthquake_context_available":
                earthquake_available,

            "coordinates_available":
                bool(
                    X_new.iloc[0][
                        "has_coordinates"
                    ]
                    == 1
                ),
        },

        "warning":
            warning,
    }


# ============================================================
# SAMPLE INPUT
# ============================================================

SAMPLE_INPUT = {
    "event_date": "2026-10-06",

    "latitude": 27.7172,
    "longitude": 85.3240,

    "district": 3008036,
    "province": 3008,

    "disaster_type": "FLOOD",

    "weather_match_distance_km": 5.0,

    "temperature_mean_c": 24.5,
    "temperature_max_c": 28.0,
    "temperature_min_c": 21.5,

    "precipitation_mm": 35.0,

    "wind_max_kmh": 22.0,
    "wind_gust_max_kmh": 35.0,

    "eq_count_7d_300km": 0,
    "eq_max_magnitude_7d_300km": np.nan,
    "eq_nearest_km_7d_300km": np.nan,

    "has_coordinates": 1,
    "has_weather": 1,
    "has_earthquake_context": 1,
}


# ============================================================
# JSON LOADER
# ============================================================

def load_input_json(
    path,
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Input JSON file not found: {path}"
        )


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )


    if not isinstance(
        payload,
        dict,
    ):

        raise ValueError(
            "Input JSON must contain an object."
        )


    return payload


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    result,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            allow_nan=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RAKSHA AI production impact-score inference"
        )
    )


    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to input JSON file. "
            "When omitted, a built-in sample is used."
        ),
    )


    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional path to save prediction JSON."
        ),
    )


    args = parser.parse_args()


    print("=" * 70)
    print(
        "RAKSHA AI - PRODUCTION INFERENCE"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    if args.input:

        print(
            f"\nInput file: "
            f"{args.input}"
        )

        payload = load_input_json(
            args.input
        )

    else:

        print(
            "\nNo input file supplied."
        )

        print(
            "Using built-in sample input."
        )

        payload = SAMPLE_INPUT.copy()


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        result = predict_impact(
            payload
        )

    except Exception as exc:

        print(
            "\nERROR:"
        )

        print(
            str(exc)
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICTION RESULT"
    )

    print(
        "=" * 70
    )


    print(
        f"\nModel: "
        f"{result['model_version']}"
    )


    print(
        f"Impact Score: "
        f"{result['impact_score']:.4f}"
    )


    print(
        f"Risk Band: "
        f"{result['risk_band']}"
    )


    print(
        "\nComponent predictions:"
    )


    print(
        f"  Log Target HGB: "
        f"{result['component_predictions']['log_target_hgb']:.4f}"
    )


    print(
        f"  Shallow HGB: "
        f"{result['component_predictions']['shallow_hgb']:.4f}"
    )


    print(
        "\nBlend:"
    )


    print(
        f"  Log weight: "
        f"{LOG_WEIGHT:.2f}"
    )


    print(
        f"  Shallow weight: "
        f"{SHALLOW_WEIGHT:.2f}"
    )


    print(
        "\nInput quality:"
    )


    print(
        f"  Missing features: "
        f"{result['input_quality']['missing_feature_count']}"
    )


    print(
        f"  Weather available: "
        f"{result['input_quality']['weather_available']}"
    )


    print(
        f"  Earthquake context: "
        f"{result['input_quality']['earthquake_context_available']}"
    )


    print(
        f"  Coordinates available: "
        f"{result['input_quality']['coordinates_available']}"
    )


    print(
        "\nWarning:"
    )


    print(
        result["warning"]
    )


    # --------------------------------------------------------
    # Save optional output
    # --------------------------------------------------------

    if args.output:

        save_result(
            result,
            args.output,
        )

        print(
            f"\nPrediction saved to: "
            f"{args.output}"
        )


    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 5D.20 COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()