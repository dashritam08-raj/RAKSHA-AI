from __future__ import annotations

import math
import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np
import pandas as pd


# ============================================================
# RAKSHA AI - PRODUCTION ML PREDICTOR
#
# Production model:
#   70% Log-Target HGB
#   30% Shallow HGB
#
# Training data:
#   Historical Nepal disaster dataset
#
# Final historical test period:
#   2012-2013
#
# IMPORTANT:
#   This predictor uses only pre-event / contextual features.
#   Post-event outcomes are never accepted as model inputs.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(
    __file__
).resolve().parents[1]

MODEL_DIR = BACKEND_DIR / "models"


SHALLOW_MODEL_PATH = (
    MODEL_DIR
    / "raksha_final_shallow_hgb.joblib"
)


LOG_MODEL_PATH = (
    MODEL_DIR
    / "raksha_final_log_target_hgb.joblib"
)


# Keep this name because backend/main.py already imports it.
#
# It points to one component of the production ensemble.
# The predictor itself loads both production models.

DEFAULT_MODEL_PATH = os.getenv(
    "RAKSHA_ML_MODEL_PATH",
    str(SHALLOW_MODEL_PATH),
)


MODEL_VERSION = "RAKSHA-ML-v1"


LOG_WEIGHT = 0.70

SHALLOW_WEIGHT = 0.30


# ============================================================
# MODEL FEATURES
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
# TRAINING RANGE
# ============================================================

TRAINING_YEAR_MIN = 1971

TRAINING_YEAR_MAX = 2013

# Coordinate envelope observed in the Phase 5D.19 Nepal training dataset.
# These bounds are used only to flag out-of-domain inference; predictions
# are still returned so the UI can expose the limitation explicitly.
TRAINING_LAT_MIN = 26.584
TRAINING_LAT_MAX = 30.026
TRAINING_LON_MIN = 80.283
TRAINING_LON_MAX = 87.920


# ============================================================
# POST-EVENT / LEAKAGE FIELDS
# ============================================================

FORBIDDEN_FIELDS = {
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
    "impact_score",
    "impact_band",
}


# ============================================================
# NUMERIC FEATURE GROUPS
# ============================================================

NUMERIC_FIELDS = [
    "latitude",
    "longitude",
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
]


BOOLEAN_FIELDS = [
    "has_coordinates",
    "has_weather",
    "has_earthquake_context",
]


# ============================================================
# IMPACT BAND
# ============================================================

def classify_impact_band(
    score: float,
) -> str:

    if score < 30:
        return "LOW"

    if score < 55:
        return "MODERATE"

    return "HIGH"


# ============================================================
# MISSING VALUE
# ============================================================

def is_missing(
    value: Any,
) -> bool:

    if value is None:
        return True

    try:
        result = np.isnan(value)

        if isinstance(
            result,
            (bool, np.bool_),
        ):
            return bool(result)

    except (
        TypeError,
        ValueError,
    ):
        pass

    return False


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def to_float(
    value: Any,
) -> Optional[float]:

    if is_missing(value):
        return None

    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        if not value:
            return None

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# CATEGORY CONVERSION
# ============================================================

def clean_category(
    value: Any,
) -> Any:

    if is_missing(value):
        return np.nan

    if value is None:
        return np.nan

    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        if not value:
            return np.nan

        # Try numeric category first.
        try:
            return float(value)
        except ValueError:
            return value

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


# ============================================================
# EVENT DATE
# ============================================================

def parse_event_date(
    value: Any,
) -> Optional[date]:

    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    text = str(value).strip()

    if not text:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]

    for fmt in formats:

        try:

            return datetime.strptime(
                text,
                fmt,
            ).date()

        except ValueError:
            continue

    raise ValueError(
        "event_date must use YYYY-MM-DD format."
    )


# ============================================================
# BUILD TEMPORAL FEATURES
# ============================================================

def build_temporal_features(
    event_date_value: Any,
    warnings: list[str],
) -> Dict[str, Optional[float]]:

    parsed = parse_event_date(
        event_date_value
    )

    # --------------------------------------------------------
    # If no event date is supplied, use the current calendar
    # month/day for seasonal features but DO NOT force the
    # current year into a model trained only through 2013.
    # --------------------------------------------------------

    if parsed is None:

        today = datetime.utcnow().date()

        parsed = today

        warnings.append(
            "event_date was not supplied; "
            "current month/day were used for temporal "
            "seasonality and year was left missing."
        )

        use_year = False

    else:

        use_year = True


    month = parsed.month

    day_of_year = (
        parsed.timetuple().tm_yday
    )

    day_of_week = (
        parsed.weekday()
    )


    month_sin = math.sin(
        2
        * math.pi
        * month
        / 12.0
    )


    month_cos = math.cos(
        2
        * math.pi
        * month
        / 12.0
    )


    year_value = (
        float(parsed.year)
        if use_year
        else None
    )


    # --------------------------------------------------------
    # Historical training-range warning
    # --------------------------------------------------------

    if use_year:

        if (
            parsed.year
            < TRAINING_YEAR_MIN
            or
            parsed.year
            > TRAINING_YEAR_MAX
        ):

            warnings.append(
                f"event_date year {parsed.year} "
                f"is outside the historical training range "
                f"{TRAINING_YEAR_MIN}-{TRAINING_YEAR_MAX}."
            )


    return {
        "year": year_value,
        "month": float(month),
        "day_of_year": float(day_of_year),
        "day_of_week": float(day_of_week),
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


# ============================================================
# LOAD MODELS
# ============================================================

@lru_cache(maxsize=1)
def _load_production_models():
    """
    Load the two final production models once.

    The models remain in memory so every API request
    does not reload the .joblib files.
    """

    if not SHALLOW_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Missing production model: "
            f"{SHALLOW_MODEL_PATH}"
        )


    if not LOG_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Missing production model: "
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
# PRODUCTION MODEL STATUS
# ============================================================

def model_status(
    model_path: Optional[str] = None,
) -> Dict:

    production_available = (
        SHALLOW_MODEL_PATH.exists()
        and
        LOG_MODEL_PATH.exists()
    )


    if production_available:

        return {

            "mode": "ml",

            "model_available": True,

            "model_path": str(
                MODEL_DIR
            ),

            "status": "validated",

            "model_version":
                MODEL_VERSION,

            "model_family":
                "70/30 HGB ensemble",

            "feature_profile":
                "phase_5d19_final_ensemble",

            "feature_columns":
                FEATURES,

            "metrics": {

                "test_MAE":
                    2.8280,

                "test_RMSE":
                    4.6932,

                "test_R2":
                    0.1027,

                "test_Spearman":
                    0.4957,
            },

            "training_period":
                "<= 2011",

            "final_test_period":
                "2012-2013",

            "ensemble_weights": {

                "log_target_hgb":
                    LOG_WEIGHT,

                "shallow_hgb":
                    SHALLOW_WEIGHT,
            },

            "data_quality": {

                "high_events_in_test":
                    0,

                "moderate_events_in_test":
                    13,

            },

            "evaluation_method":
                "Chronological holdout + rolling time-series validation",

            "message":
                (
                    "RAKSHA ML historical-validation ensemble is active for "
                    "decision-support screening. Real-world field validation "
                    "and India-specific calibration are still required."
                ),

        }


    # ========================================================
    # LEGACY FALLBACK STATUS
    # ========================================================

    path = Path(
        model_path or DEFAULT_MODEL_PATH
    )


    if path.exists():

        return {

            "mode":
                "legacy",

            "model_available":
                True,

            "model_path":
                str(path),

            "status":
                "legacy",

            "model_version":
                None,

            "message":
                (
                    "Production ensemble files are missing; "
                    "legacy model is available."
                ),

        }


    return {

        "mode":
            "rules_baseline",

        "model_available":
            False,

        "model_path":
            str(path),

        "status":
            "missing",

        "message":
            (
                "No production ML model is available."
            ),

    }


# ============================================================
# PREPARE PRODUCTION FEATURES
# ============================================================

def prepare_features(
    disaster_type: str,
    severity: str,
    people_affected: int,
    extra_features: Optional[
        Mapping[str, Any]
    ] = None,
):

    del severity
    del people_affected

    extra = dict(
        extra_features or {}
    )


    warnings: list[str] = []


    # --------------------------------------------------------
    # Leakage protection
    # --------------------------------------------------------

    forbidden_present = [
        field
        for field in FORBIDDEN_FIELDS
        if field in extra
    ]


    if forbidden_present:

        raise ValueError(
            "Post-event/target fields are not allowed "
            "for ML prediction: "
            + ", ".join(
                sorted(
                    forbidden_present
                )
            )
        )


    # --------------------------------------------------------
    # Base row
    # --------------------------------------------------------

    row: Dict[str, Any] = {
        feature: np.nan
        for feature in FEATURES
    }


    # --------------------------------------------------------
    # Disaster type
    # --------------------------------------------------------

    if not disaster_type:
        raise ValueError(
            "disaster_type is required."
        )


    row[
        "disaster_type"
    ] = (
        str(disaster_type)
        .strip()
        .upper()
    )


    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    row[
        "latitude"
    ] = to_float(
        extra.get("latitude")
    )


    row[
        "longitude"
    ] = to_float(
        extra.get("longitude")
    )


    coordinates_available = (
        row["latitude"] is not None
        and
        row["longitude"] is not None
    )

    geography_within_training_domain = True

    if coordinates_available:
        geography_within_training_domain = (
            TRAINING_LAT_MIN <= float(row["latitude"]) <= TRAINING_LAT_MAX
            and
            TRAINING_LON_MIN <= float(row["longitude"]) <= TRAINING_LON_MAX
        )
        if not geography_within_training_domain:
            warnings.append(
                "Coordinates are outside the historical training geography "
                f"(lat {TRAINING_LAT_MIN:.3f}-{TRAINING_LAT_MAX:.3f}, "
                f"lon {TRAINING_LON_MIN:.3f}-{TRAINING_LON_MAX:.3f}; "
                "training data is Nepal)."
            )

    row[
        "has_coordinates"
    ] = int(
        coordinates_available
    )


    # --------------------------------------------------------
    # District / Province
    # --------------------------------------------------------

    row[
        "district"
    ] = clean_category(
        extra.get("district")
    )


    row[
        "province"
    ] = clean_category(
        extra.get("province")
    )


    # --------------------------------------------------------
    # Weather
    #
    # Backend currently has fields such as:
    # temperature_c
    # rainfall_24h_mm
    # wind_speed_kmh
    # wind_gust_kmh
    #
    # They are mapped into the Phase 5D feature schema.
    # --------------------------------------------------------

    temperature = to_float(
        extra.get(
            "temperature_c"
        )
    )


    temperature_mean = to_float(
        extra.get(
            "temperature_mean_c"
        )
    )


    temperature_max = to_float(
        extra.get(
            "temperature_max_c"
        )
    )


    temperature_min = to_float(
        extra.get(
            "temperature_min_c"
        )
    )


    if temperature_mean is None:
        temperature_mean = temperature


    if temperature_max is None:
        temperature_max = temperature


    if temperature_min is None:
        temperature_min = temperature


    row[
        "temperature_mean_c"
    ] = temperature_mean


    row[
        "temperature_max_c"
    ] = temperature_max


    row[
        "temperature_min_c"
    ] = temperature_min


    precipitation = to_float(
        extra.get(
            "precipitation_mm"
        )
    )


    if precipitation is None:

        precipitation = to_float(
            extra.get(
                "rainfall_24h_mm"
            )
        )


    row[
        "precipitation_mm"
    ] = precipitation


    wind_max = to_float(
        extra.get(
            "wind_max_kmh"
        )
    )


    if wind_max is None:

        wind_max = to_float(
            extra.get(
                "wind_speed_kmh"
            )
        )


    row[
        "wind_max_kmh"
    ] = wind_max


    wind_gust = to_float(
        extra.get(
            "wind_gust_max_kmh"
        )
    )


    if wind_gust is None:

        wind_gust = to_float(
            extra.get(
                "wind_gust_kmh"
            )
        )


    row[
        "wind_gust_max_kmh"
    ] = wind_gust


    weather_available = any(
        value is not None
        for value in [
            row[
                "temperature_mean_c"
            ],
            row[
                "temperature_max_c"
            ],
            row[
                "temperature_min_c"
            ],
            row[
                "precipitation_mm"
            ],
            row[
                "wind_max_kmh"
            ],
            row[
                "wind_gust_max_kmh"
            ],
        ]
    )


    row[
        "has_weather"
    ] = int(
        weather_available
    )


    # --------------------------------------------------------
    # Weather matching distance
    # --------------------------------------------------------

    row[
        "weather_match_distance_km"
    ] = to_float(
        extra.get(
            "weather_match_distance_km"
        )
    )


    # --------------------------------------------------------
    # Earthquake context
    # --------------------------------------------------------

    eq_count = to_float(
        extra.get(
            "eq_count_7d_300km"
        )
    )


    earthquake_magnitude = to_float(
        extra.get(
            "eq_max_magnitude_7d_300km"
        )
    )


    if earthquake_magnitude is None:

        earthquake_magnitude = to_float(
            extra.get(
                "earthquake_magnitude"
            )
        )


    eq_nearest = to_float(
        extra.get(
            "eq_nearest_km_7d_300km"
        )
    )


    row[
        "eq_count_7d_300km"
    ] = eq_count


    row[
        "eq_max_magnitude_7d_300km"
    ] = earthquake_magnitude


    row[
        "eq_nearest_km_7d_300km"
    ] = eq_nearest


    earthquake_available = (
        eq_count is not None
        or
        earthquake_magnitude is not None
        or
        eq_nearest is not None
    )


    row[
        "has_earthquake_context"
    ] = int(
        earthquake_available
    )


    # --------------------------------------------------------
    # Event date / temporal features
    # --------------------------------------------------------

    temporal = build_temporal_features(
        extra.get(
            "event_date"
        ),
        warnings,
    )


    for key, value in temporal.items():

        row[key] = value


    # --------------------------------------------------------
    # Build exact dataframe
    # --------------------------------------------------------

    row_df = pd.DataFrame(
        [row],
        columns=FEATURES,
    )


    return (
        row_df,
        warnings,
        {
            "coordinates_available":
                coordinates_available,

            "weather_available":
                weather_available,

            "earthquake_context_available":
                earthquake_available,

            "geography_within_training_domain":
                geography_within_training_domain,

            "training_geography":
                "Nepal historical dataset",
        },
    )


# ============================================================
# PREDICT RISK
# ============================================================

def predict_risk(
    disaster_type: str,
    severity: str,
    people_affected: int,
    model_path: Optional[str] = None,
    extra_features: Optional[
        Mapping[str, Any]
    ] = None,
):
    """
    Main function already used by backend/main.py.

    The function signature intentionally remains compatible
    with the existing FastAPI backend.
    """


    status = model_status(
        model_path
    )


    # ========================================================
    # PRODUCTION ENSEMBLE
    # ========================================================

    if (
        status.get(
            "mode"
        )
        == "ml"
        and
        status.get(
            "model_available"
        )
    ):

        (
            row_df,
            warnings,
            input_quality,
        ) = prepare_features(
            disaster_type=disaster_type,
            severity=severity,
            people_affected=people_affected,
            extra_features=extra_features,
        )


        try:

            (
                shallow_model,
                log_model,
            ) = _load_production_models()


            # ------------------------------------------------
            # Shallow HGB
            # ------------------------------------------------

            shallow_prediction = float(
                shallow_model.predict(
                    row_df
                )[0]
            )


            # ------------------------------------------------
            # Log-target HGB
            #
            # Convert prediction back from log1p scale.
            # ------------------------------------------------

            log_prediction = float(
                np.expm1(
                    log_model.predict(
                        row_df
                    )[0]
                )
            )


            log_prediction = max(
                0.0,
                log_prediction,
            )


            shallow_prediction = max(
                0.0,
                shallow_prediction,
            )


            # ------------------------------------------------
            # Final 70/30 blend
            # ------------------------------------------------

            prediction = (
                LOG_WEIGHT
                * log_prediction
                +
                SHALLOW_WEIGHT
                * shallow_prediction
            )


            prediction = max(
                0.0,
                min(
                    100.0,
                    prediction,
                ),
            )


            risk_band = (
                classify_impact_band(
                    prediction
                )
            )


            # ------------------------------------------------
            # Feature completeness
            # ------------------------------------------------

            optional_context_fields = [
                "latitude",
                "longitude",
                "district",
                "province",
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


            supplied_count = sum(
                not is_missing(
                    row_df.iloc[0][field]
                )
                for field in optional_context_fields
            )


            completeness = round(
                (
                    supplied_count
                    /
                    len(
                        optional_context_fields
                    )
                )
                * 100.0,
                1,
            )


            # ------------------------------------------------
            # OOD / model limitation warnings
            # ------------------------------------------------

            if row_df.iloc[0][
                "year"
            ] is None or is_missing(
                row_df.iloc[0][
                    "year"
                ]
            ):

                warnings.append(
                    "Year was not supplied because "
                    "the current calendar year is outside "
                    "the historical training range."
                )


            warnings.append(
                "This is a model-derived screening score, "
                "not a calibrated disaster probability."
            )


            warnings.append(
                "The historical test period contained "
                "zero HIGH-impact events, so HIGH-risk "
                "generalization was not validated."
            )


            return {

                "risk_score":
                    int(
                        round(
                            prediction
                        )
                    ),

                "impact_score":
                    round(
                        prediction,
                        4,
                    ),

                "risk_level":
                    risk_band,

                "risk_band":
                    risk_band,

                "source":
                    "ml",

                "model_status":
                    "validated",

                "model_version":
                    MODEL_VERSION,

                "model_family":
                    "70/30 HGB ensemble",

                "feature_profile":
                    "phase_5d19_final_ensemble",

                "prediction_interval":
                    None,

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

                "model_metrics": status.get(
                    "metrics",
                    {},
                ),

                "feature_completeness_pct":
                    completeness,

                "input_quality":
                    input_quality,

                "warnings":
                    warnings,

            }


        except Exception:

            # Never crash the entire RAKSHA backend
            # because ML inference failed.

            raise


    # ========================================================
    # NO PRODUCTION MODEL
    # ========================================================

    return None