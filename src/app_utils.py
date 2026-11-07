"""Shared validation and metric helpers for the Streamlit application."""

import pandas as pd

from src.data_validation import FEATURE_COLUMNS
from src.evaluation import evaluate_model


def validate_upload(data: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + ["result"])
    missing = required.difference(data.columns.str.lower())
    if missing:
        raise ValueError(f"Uploaded CSV is missing columns: {sorted(missing)}")


def metrics_for_upload(model, data: pd.DataFrame):
    normalized = data.copy()
    normalized.columns = normalized.columns.str.lower()
    validate_upload(normalized)
    return evaluate_model(model, normalized[FEATURE_COLUMNS], normalized["result"])

