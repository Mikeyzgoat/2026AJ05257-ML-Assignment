import pandas as pd

from src.data_validation import FEATURE_COLUMNS, validate_dataset


def test_feature_list_meets_assignment_requirement():
    assert len(FEATURE_COLUMNS) == 30


def test_dataset_validation_accepts_expected_schema():
    frame = pd.DataFrame({column: [1, -1] for column in FEATURE_COLUMNS})
    frame["Result"] = [-1, 1]
    validate_dataset(frame)

