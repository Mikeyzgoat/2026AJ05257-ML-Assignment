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
    "Logistic Regression": '''def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)
    return model''',
    "Decision Tree": '''def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model''',
    "KNN": '''def train_knn(X_train, y_train):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
    model.fit(X_train, y_train)
    return model''',
    "Naive Bayes": '''def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model''',
    "SVM": '''def train_svm(X_train, y_train):
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
    st.write("The UCI dataset is used by default. You can also upload the labeled test CSV.")
    source = st.radio("Data source", ["UCI dataset", "Upload test CSV"], horizontal=True)
    try:
        if source == "UCI dataset":
            data = load_uci_data()
        else:
            uploaded = st.file_uploader("Upload labeled test CSV", type="csv", key=f"upload_{model_name}")
            if uploaded is None:
                st.info("Upload a CSV containing all 30 features and the result column.")
                st.caption("2026AJ05257")
                return
            data = pd.read_csv(uploaded)
            data.columns = data.columns.str.lower()
            required_columns = set(FEATURE_COLUMNS + ["result"])
            missing_columns = required_columns.difference(data.columns)
            if missing_columns:
                raise ValueError(f"Uploaded CSV is missing columns: {sorted(missing_columns)}")
            data = data[FEATURE_COLUMNS + ["result"]]

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
