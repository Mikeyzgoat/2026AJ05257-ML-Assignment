"""Target mapping and reproducible stratified split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_validation import FEATURE_COLUMNS


def map_target(target: pd.Series) -> pd.Series:
    mapped = target.map({-1: 1, 1: 0})
    if mapped.isna().any():
        raise ValueError("Target contains values other than -1 and +1")
    return mapped.astype(int)


def prepare_data(raw_path: str | Path = "datasets/phishing_dataset.csv", test_path: str | Path = "test_data.csv"):
    frame = pd.read_csv(raw_path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    target_column = "result"
    X = frame[FEATURE_COLUMNS].apply(pd.to_numeric)
    y = map_target(frame[target_column])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    test_frame = X_test.copy()
    test_frame["result"] = y_test
    test_frame.to_csv(test_path, index=False)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data()
    print("Created stratified test_data.csv")
