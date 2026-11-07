"""Streamlit interface for evaluating saved phishing classifiers."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.app_utils import validate_upload
from src.data_validation import FEATURE_COLUMNS
from src.evaluation import evaluate_model
from src.models import MODEL_FILES

st.set_page_config(page_title="Phishing Website Classifier", layout="wide")
st.title("Phishing Website Classification")
st.write("Upload a labeled CSV with the 30 input features and result column to evaluate a saved model.")

uploaded = st.file_uploader("Upload test CSV", type="csv")
selected = st.selectbox("Select model", list(MODEL_FILES))

if uploaded is not None:
    data = pd.read_csv(uploaded)
    data.columns = data.columns.str.lower()
    try:
        validate_upload(data)
        model = joblib.load(Path("model") / MODEL_FILES[selected])
        metrics = evaluate_model(model, data[FEATURE_COLUMNS], data["result"])
        st.subheader("Metrics")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
        predictions = model.predict(data[FEATURE_COLUMNS])
        matrix = pd.crosstab(data["result"], predictions, rownames=["Actual"], colnames=["Predicted"], dropna=False)
        st.subheader("Confusion matrix")
        st.dataframe(matrix, use_container_width=True)
        st.subheader("Classification report")
        from sklearn.metrics import classification_report
        st.code(classification_report(data["result"], predictions, target_names=["legitimate", "phishing"], zero_division=0))
    except (ValueError, KeyError) as error:
        st.error(str(error))
else:
    st.info("Upload a labeled CSV to begin.")

st.caption("2026AJ05257")

