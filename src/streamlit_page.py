"""Shared layout for each model page."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

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

    st.subheader("Test data upload")
    st.write("Upload a CSV containing all 30 input features and the `result` target column.")
    uploaded_file = st.file_uploader("Upload test_data.csv", type="csv")
    try:
        if uploaded_file is not None:
            test_data = pd.read_csv(uploaded_file)
        else:
            test_data = pd.read_csv("test_data.csv")

        test_data.columns = [str(column).strip().lower() for column in test_data.columns]
        if "result" not in test_data.columns:
            raise ValueError("The CSV must contain a 'result' target column.")
        missing_features = [feature for feature in FEATURE_COLUMNS if feature not in test_data.columns]
        if missing_features:
            raise ValueError(f"The CSV is missing feature columns: {missing_features}")

        test_data[FEATURE_COLUMNS] = test_data[FEATURE_COLUMNS].apply(pd.to_numeric)
        test_data["result"] = pd.to_numeric(test_data["result"])
        if set(test_data["result"].dropna().unique()).issubset({-1, 1}):
            test_data["result"] = test_data["result"].map({-1: 1, 1: 0})
        if not set(test_data["result"].dropna().unique()).issubset({0, 1}):
            raise ValueError("The result column must contain only 0/1 or -1/+1 labels.")

        st.caption(f"Using {'uploaded' if uploaded_file else 'repository'} CSV: {len(test_data):,} rows")
        st.write("First 10 rows loaded with `pd.read_csv(...).head(10)`:")
        st.dataframe(test_data.head(10), hide_index=True, use_container_width=True)

        X = test_data[FEATURE_COLUMNS]
        y = test_data["result"].astype(int)

        st.subheader("Results for all models")
        all_results = []
        for name, filename in MODEL_FILES.items():
            saved_model = joblib.load(Path("model") / filename)
            all_results.append({"Model": name, **evaluate_model(saved_model, X, y)})
        st.dataframe(pd.DataFrame(all_results).round(4), hide_index=True, use_container_width=True)

        model = joblib.load(Path("model") / MODEL_FILES[model_name])
        predictions = model.predict(X)
        metrics = evaluate_model(model, X, y)

        st.subheader(f"{model_name} details")
        st.write("Metrics")
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
