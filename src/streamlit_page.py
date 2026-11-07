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

ALGORITHM_CODE = {
    "Logistic Regression": '''from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)
    return model''',
    "Decision Tree": '''from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model''',
    "KNN": '''from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_knn(X_train, y_train):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
    model.fit(X_train, y_train)
    return model''',
    "Naive Bayes": '''from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model''',
    "SVM": '''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def train_svm(X_train, y_train):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model''',
}


def render_model_page(model_name: str) -> None:
    st.title(model_name)
    st.subheader("Code used")
    st.code(ALGORITHM_CODE[model_name], language="python")
    st.write("This model uses the UCI Phishing Websites dataset directly.")
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
