"""Download and validate the UCI Phishing Websites dataset."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

FEATURE_COLUMNS = [
    "having_ip_address", "url_length", "shortining_service", "having_at_symbol",
    "double_slash_redirecting", "prefix_suffix", "having_sub_domain", "sslfinal_state",
    "domain_registration_length", "favicon", "port", "https_token", "request_url",
    "url_of_anchor", "links_in_tags", "sfh", "submitting_to_email", "abnormal_url",
    "redirect", "on_mouseover", "rightclick", "popupwindow", "iframe", "age_of_domain",
    "dnsrecord", "web_traffic", "page_rank", "google_index", "links_pointing_to_page",
    "statistical_report",
]


def download_dataset(output_path: str | Path = "datasets/phishing_dataset.csv") -> pd.DataFrame:
    dataset = fetch_ucirepo(id=327)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    features.columns = [str(column).strip().lower() for column in features.columns]
    target_name = targets.columns[0]
    frame = features.copy()
    frame["Result"] = targets[target_name].to_numpy()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if len(frame) < 500 or len(FEATURE_COLUMNS) < 12:
        raise ValueError("Dataset does not meet the assignment size requirements")
    labels = set(frame["Result"].dropna().unique())
    if not labels.issubset({-1, 1}):
        raise ValueError(f"Unexpected target labels: {labels}")


if __name__ == "__main__":
    data = download_dataset()
    print(f"Saved {len(data)} rows and {len(FEATURE_COLUMNS)} features")
