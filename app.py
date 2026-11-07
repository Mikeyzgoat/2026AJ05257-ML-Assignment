"""Streamlit entry point and page navigation."""

import streamlit as st

st.set_page_config(page_title="Phishing Website Classifier", layout="wide")

pages = [
    st.Page("src/home_page.py", title="Home Page"),
    st.Page("pages/1_Logistic_Regression.py", title="Logistic Regression"),
    st.Page("pages/2_Decision_Tree.py", title="Decision Tree"),
    st.Page("pages/3_KNN.py", title="KNN"),
    st.Page("pages/4_Naive_Bayes.py", title="Naive Bayes"),
    st.Page("pages/5_SVM.py", title="SVM"),
]
st.navigation(pages).run()
