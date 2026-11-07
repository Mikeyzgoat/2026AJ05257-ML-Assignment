"""Train and save the five required classifiers."""

from pathlib import Path

import joblib

from src.models import MODEL_FILES, build_models
from src.preprocessing import prepare_data


def train_and_save(model_dir: str | Path = "model"):
    X_train, _, y_train, _ = prepare_data()
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, output / MODEL_FILES[name])
        print(f"Saved {name}")


if __name__ == "__main__":
    train_and_save()

