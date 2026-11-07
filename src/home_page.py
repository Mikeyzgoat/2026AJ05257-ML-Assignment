"""Home page with dataset details and exploratory analysis."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_utils import load_uci_data
from src.data_validation import FEATURE_COLUMNS

st.title("Phishing Website Classification")
st.write("This page shows the dataset and a short exploratory analysis.")
st.link_button("Open the GitHub repository", "https://github.com/Mikeyzgoat/2026AJ05257-ML-Assignment")

st.subheader("Dataset downloads")
download_github, download_uci = st.columns(2)
with download_github:
    st.link_button(
        "Download from GitHub",
        "https://raw.githubusercontent.com/Mikeyzgoat/2026AJ05257-ML-Assignment/main/datasets/phishing_dataset.csv",
    )
with download_uci:
    st.link_button(
        "Download from UCI",
        "https://archive.ics.uci.edu/dataset/327/phishing",
    )

data = load_uci_data()

st.subheader("Dataset overview")
st.write(f"Rows: {len(data):,} | Input features: {len(FEATURE_COLUMNS)}")
st.write("The target is 1 for phishing and 0 for legitimate websites.")

st.subheader("Dataset features")
feature_table = pd.DataFrame(
    {
        "Feature": FEATURE_COLUMNS,
        "Type": [str(data[feature].dtype) for feature in FEATURE_COLUMNS],
        "Unique values": [data[feature].nunique() for feature in FEATURE_COLUMNS],
    }
)
st.dataframe(feature_table, hide_index=True, use_container_width=True)

st.subheader("Exploratory analysis")
st.write("The class chart shows the number of legitimate and phishing rows.")
class_counts = data["result"].map({0: "Legitimate", 1: "Phishing"}).value_counts()
st.bar_chart(class_counts)

heatmap = Path("assets/correlation_heatmap.png")
if heatmap.exists():
    st.write("The heatmap shows how strongly the numeric features move together.")
    st.image(str(heatmap), use_container_width=True)

st.subheader("Model selection")
st.write("Choose a model below, then open the matching page from the menu.")
selected = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"])
page_links = {
    "Logistic Regression": "pages/1_Logistic_Regression.py",
    "Decision Tree": "pages/2_Decision_Tree.py",
    "KNN": "pages/3_KNN.py",
    "Naive Bayes": "pages/4_Naive_Bayes.py",
    "SVM": "pages/5_SVM.py",
}
st.page_link(page_links[selected], label=f"Open {selected} page")

comparison = Path("results/model_comparison.csv")
if comparison.exists():
    st.subheader("Test-set comparison")
    st.write("These results come from the saved 20% test split.")
    st.dataframe(pd.read_csv(comparison).round(4), hide_index=True, use_container_width=True)

st.caption("2026AJ05257")
