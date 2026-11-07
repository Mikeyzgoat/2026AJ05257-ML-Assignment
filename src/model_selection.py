"""Model selection page."""

import streamlit as st

from src.models import MODEL_FILES
from src.streamlit_page import render_model_page

st.title("Model Selection")
st.write("Choose a model to view its code, metrics, and analysis.")
selected = st.selectbox("Choose a model", list(MODEL_FILES))
render_model_page(selected)
