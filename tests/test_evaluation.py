import pandas as pd
from sklearn.naive_bayes import GaussianNB

from src.evaluation import evaluate_model


def test_evaluation_returns_six_metrics():
    model = GaussianNB().fit([[0], [1], [0], [1]], [0, 1, 0, 1])
    result = evaluate_model(model, pd.DataFrame({"x": [0, 1, 0, 1]}), pd.Series([0, 1, 0, 1]))
    assert set(result) == {"Accuracy", "AUC", "Precision", "Recall", "F1-score", "MCC"}

