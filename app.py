"""Streamlit entry point for the two-page application."""

import streamlit as st

st.set_page_config(page_title="Phishing Website Classifier", layout="wide")

pages = [
    st.Page("src/home_page.py", title="Home Page"),
    st.Page("src/model_selection.py", title="Model Selection"),
]
st.navigation(pages).run()
