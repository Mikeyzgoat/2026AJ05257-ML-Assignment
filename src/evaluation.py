"""Evaluate saved classifiers with the six assignment metrics."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

from src.data_validation import FEATURE_COLUMNS
from src.models import MODEL_FILES


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    return {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, scores),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1-score": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }


def evaluate_saved_models(test_path="test_data.csv", model_dir="model") -> pd.DataFrame:
    test = pd.read_csv(test_path)
    X, y = test[FEATURE_COLUMNS], test["result"]
    rows = []
    details = Path("results/evaluation_details")
    details.mkdir(parents=True, exist_ok=True)
    for name, filename in MODEL_FILES.items():
        model = joblib.load(Path(model_dir) / filename)
        rows.append({"Model": name, **evaluate_model(model, X, y)})
        predictions = model.predict(X)
        pd.DataFrame(confusion_matrix(y, predictions, labels=[0, 1]), index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(details / f"{filename.removesuffix('.joblib')}_confusion_matrix.csv")
        report = classification_report(y, predictions, target_names=["legitimate", "phishing"], zero_division=0)
        (details / f"{filename.removesuffix('.joblib')}_classification_report.txt").write_text(report)
    result = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    result.to_csv("results/model_comparison.csv", index=False)
    return result


if __name__ == "__main__":
    print(evaluate_saved_models().to_string(index=False))

