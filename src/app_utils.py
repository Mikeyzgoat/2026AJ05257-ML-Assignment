"""Shared validation and metric helpers for the Streamlit application."""

import pandas as pd
import streamlit as st
from ucimlrepo import fetch_ucirepo

from src.data_validation import FEATURE_COLUMNS


@st.cache_data(show_spinner="Loading the UCI dataset...")
def load_uci_data() -> pd.DataFrame:
    """Fetch UCI dataset 327 and map its labels for evaluation."""
    dataset = fetch_ucirepo(id=327)
    features = dataset.data.features.copy()
    target = dataset.data.targets.iloc[:, 0].copy()
    features.columns = [str(column).strip().lower() for column in features.columns]
    data = features[FEATURE_COLUMNS].apply(pd.to_numeric)
    data["result"] = target.map({-1: 1, 1: 0}).astype(int).to_numpy()
    return data
