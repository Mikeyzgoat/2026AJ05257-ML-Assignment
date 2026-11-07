import streamlit as st

from src.streamlit_page import render_model_page

st.set_page_config(page_title="Logistic Regression", layout="wide")
render_model_page("Logistic Regression")
