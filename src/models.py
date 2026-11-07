"""Model definitions used by training and evaluation."""

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_models():
    return {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
        "Naive Bayes": GaussianNB(),
        "SVM": Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, random_state=42))]),
    }


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

