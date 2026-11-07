"""Shared layout for each model page."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

from src.app_utils import load_uci_data
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
    st.write("This page uses the UCI Phishing Websites dataset directly.")
    try:
        data = load_uci_data()
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
