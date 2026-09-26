from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import pandas as pd

from src.exception import DataValidationError

FieldKind = Literal["number", "integer", "category"]
NormalizedValue = str | int | float


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: FieldKind
    section: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()


PREDICTION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("G1", "First period grade", "number", "academic", 0, 20),
    FieldSpec("G2", "Second period grade", "number", "academic", 0, 20),
    FieldSpec("studytime", "Weekly study time", "integer", "academic", 1, 4),
    FieldSpec("failures", "Past class failures", "integer", "academic", 0, 4),
    FieldSpec("Medu", "Mother's education", "integer", "academic", 0, 4),
    FieldSpec("Fedu", "Father's education", "integer", "academic", 0, 4),
    FieldSpec("schoolsup", "School support", "category", "academic", choices=("no", "yes")),
    FieldSpec("famsup", "Family support", "category", "academic", choices=("no", "yes")),
    FieldSpec("paid", "Paid classes", "category", "academic", choices=("no", "yes")),
    FieldSpec("absences", "School absences", "integer", "habits", 0, 93),
    FieldSpec("traveltime", "Travel time", "integer", "habits", 1, 4),
    FieldSpec("famrel", "Family relationships", "integer", "habits", 1, 5),
    FieldSpec("freetime", "Free time", "integer", "habits", 1, 5),
    FieldSpec("goout", "Time out with friends", "integer", "habits", 1, 5),
    FieldSpec("Dalc", "Weekday alcohol use", "integer", "habits", 1, 5),
    FieldSpec("Walc", "Weekend alcohol use", "integer", "habits", 1, 5),
    FieldSpec("health", "Current health", "integer", "habits", 1, 5),
    FieldSpec("activities", "Extracurricular activities", "category", "habits", choices=("no", "yes")),
    FieldSpec("nursery", "Attended nursery", "category", "habits", choices=("no", "yes")),
    FieldSpec("romantic", "Romantic relationship", "category", "habits", choices=("no", "yes")),
    FieldSpec("school", "School", "category", "social", choices=("GP", "MS")),
    FieldSpec("sex", "Sex", "category", "social", choices=("F", "M")),
    FieldSpec("age", "Age", "integer", "social", 15, 22),
    FieldSpec("address", "Home address", "category", "social", choices=("R", "U")),
    FieldSpec("famsize", "Family size", "category", "social", choices=("GT3", "LE3")),
    FieldSpec("Pstatus", "Parents live together", "category", "social", choices=("A", "T")),
    FieldSpec("Mjob", "Mother's job", "category", "social", choices=("at_home", "health", "other", "services", "teacher")),
    FieldSpec("Fjob", "Father's job", "category", "social", choices=("at_home", "health", "other", "services", "teacher")),
    FieldSpec("reason", "School choice reason", "category", "social", choices=("course", "home", "other", "reputation")),
    FieldSpec("guardian", "Guardian", "category", "social", choices=("father", "mother", "other")),
    FieldSpec("higher", "Plans for higher education", "category", "social", choices=("no", "yes")),
    FieldSpec("internet", "Internet at home", "category", "social", choices=("no", "yes")),
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(field.name for field in PREDICTION_FIELDS)
NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
)
TARGET_COLUMN = "G3"
PASS_THRESHOLD = 10.0

if set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) != set(FEATURE_COLUMNS):
    raise RuntimeError("Feature schema groups do not match the prediction schema")


def coerce_prediction_input(payload: Mapping[str, Any]) -> dict[str, NormalizedValue]:
    if not isinstance(payload, Mapping):
        raise DataValidationError("The request body must be a JSON object.")

    expected = set(FEATURE_COLUMNS)
    received = set(payload)
    errors: dict[str, list[str]] = {}
    normalized: dict[str, NormalizedValue] = {}

    unknown = sorted(received - expected)
    if unknown:
        errors["_schema"] = [f"Unknown fields: {', '.join(unknown)}"]

    for field in PREDICTION_FIELDS:
        if field.name not in payload:
            errors[field.name] = ["This field is required."]
            continue
        raw_value = payload[field.name]
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            errors[field.name] = ["This field is required."]
            continue
        if isinstance(raw_value, (dict, list, tuple, set, bool)):
            errors[field.name] = ["Enter a single valid value."]
            continue
        if field.kind == "category":
            value = str(raw_value).strip()
            if value not in field.choices:
                errors[field.name] = [f"Choose one of: {', '.join(field.choices)}."]
            else:
                normalized[field.name] = value
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            errors[field.name] = ["Enter a valid number."]
            continue
        if not math.isfinite(numeric_value):
            errors[field.name] = ["Enter a finite number."]
            continue
        if field.kind == "integer" and not numeric_value.is_integer():
            errors[field.name] = ["Enter a whole number."]
            continue
        if field.minimum is not None and numeric_value < field.minimum:
            errors[field.name] = [f"Must be at least {field.minimum:g}."]
            continue
        if field.maximum is not None and numeric_value > field.maximum:
            errors[field.name] = [f"Must be at most {field.maximum:g}."]
            continue
        normalized[field.name] = int(numeric_value) if field.kind == "integer" else numeric_value

    if errors:
        raise DataValidationError("Input validation failed.", details={"fields": errors})

    return {field.name: normalized[field.name] for field in PREDICTION_FIELDS}


def validate_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = sorted(required - set(frame.columns))
    extra = sorted(set(frame.columns) - required)
    if missing:
        raise DataValidationError(
            "Training data is missing required columns.",
            details={"missing_columns": missing, "unexpected_columns": extra},
        )
    if frame.empty:
        raise DataValidationError("Training data cannot be empty.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DataValidationError(
            "Training data contains duplicate columns.",
            details={"duplicate_columns": duplicates},
        )

    cleaned = frame.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    if cleaned.isna().any().any():
        missing_by_column = {
            column: int(cleaned[column].isna().sum())
            for column in cleaned.columns
            if cleaned[column].isna().any()
        }
        raise DataValidationError(
            "Training data contains missing values.",
            details={"missing_by_column": missing_by_column},
        )

    invalid_columns: dict[str, list[int]] = {}
    for column in (*NUMERIC_FEATURES, TARGET_COLUMN):
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        invalid_mask = converted.isna()
        if invalid_mask.any():
            invalid_columns[column] = cleaned.index[invalid_mask].tolist()
        cleaned[column] = converted
    if invalid_columns:
        raise DataValidationError(
            "Training data contains non-numeric values.",
            details={"invalid_columns": invalid_columns},
        )

    bounded_columns: dict[str, list[int]] = {}
    for column in (*NUMERIC_FEATURES, TARGET_COLUMN):
        minimum = 0 if column in {"absences", "G1", "G2", TARGET_COLUMN} else None
        maximum = 20 if column in {"G1", "G2", TARGET_COLUMN} else None
        invalid_mask = pd.Series(False, index=cleaned.index)
        if minimum is not None:
            invalid_mask |= cleaned[column] < minimum
        if maximum is not None:
            invalid_mask |= cleaned[column] > maximum
        if invalid_mask.any():
            bounded_columns[column] = cleaned.index[invalid_mask].tolist()
    if bounded_columns:
        raise DataValidationError(
            "Training data contains out-of-range grades.",
            details={"invalid_rows": bounded_columns},
        )

    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype(str)

    features = cleaned.loc[:, list(FEATURE_COLUMNS)]
    target = cleaned[TARGET_COLUMN].astype(float)
    return features, target
