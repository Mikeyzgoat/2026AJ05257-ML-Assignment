"""Shared layout for each model page."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

from src.app_utils import validate_upload
from src.data_validation import FEATURE_COLUMNS
from src.evaluation import evaluate_model
from src.models import MODEL_FILES

ANALYSIS = {
    "Logistic Regression": "This is a simple linear baseline. It gives balanced results, but it may miss complex patterns.",
    "Decision Tree": "This model makes clear rule-based decisions. It gives the strongest overall result on this test set.",
    "KNN": "This model compares each row with nearby rows. It works well, but prediction can be slower on larger data.",
    "Naive Bayes": "This model catches almost all phishing rows, but it also marks many legitimate rows as phishing.",
    "SVM": "This model finds a strong separating boundary. It gives the best AUC, so its ranking of cases is very good.",
}


def render_model_page(model_name: str) -> None:
    st.title(model_name)
    st.write("Upload a labeled CSV to test this saved model.")

    uploaded = st.file_uploader("Upload CSV", type="csv", key=f"upload_{model_name}")
    if uploaded is None:
        st.info("The CSV must contain all 30 features and the result column.")
        st.caption("2026AJ05257")
        return

    data = pd.read_csv(uploaded)
    data.columns = data.columns.str.lower()
    try:
        validate_upload(data)
        model = joblib.load(Path("model") / MODEL_FILES[model_name])
        X = data[FEATURE_COLUMNS]
        y = data["result"]
        metrics = evaluate_model(model, X, y)
        predictions = model.predict(X)

        st.subheader("Metrics")
        st.dataframe(pd.DataFrame([metrics]).round(4), hide_index=True, use_container_width=True)

        st.subheader("Confusion matrix")
        matrix = pd.crosstab(y, predictions, rownames=["Actual"], colnames=["Predicted"], dropna=False)
        st.dataframe(matrix, use_container_width=True)

        st.subheader("Analysis")
        st.write(ANALYSIS[model_name])

        st.subheader("Classification report")
        st.code(classification_report(y, predictions, target_names=["legitimate", "phishing"], zero_division=0))
    except (ValueError, KeyError) as error:
        st.error(str(error))

    st.caption("2026AJ05257")
