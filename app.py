"""Home page for the Streamlit multipage application."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phishing Website Classifier", layout="wide")
st.title("Phishing Website Classification")
st.write("Use the page menu to open one model at a time.")

st.subheader("How to use the app")
st.write("1. Open a model page from the menu.")
st.write("2. Upload a labeled CSV with the 30 features and a result column.")
st.write("3. Review the metrics, confusion matrix, and short analysis.")

st.subheader("Available models")
st.dataframe(
    pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"],
            "Purpose": ["Linear baseline", "Rule-based decisions", "Nearest examples", "Probability-based baseline", "Strong boundary model"],
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.info("The app loads saved models. It does not train models on uploaded files.")
st.caption("2026AJ05257")
