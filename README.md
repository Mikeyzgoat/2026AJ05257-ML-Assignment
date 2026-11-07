# Phishing Website Classification

This project compares five supervised classification models on the UCI Machine Learning Repository Phishing Websites dataset.

## Problem statement

The task is to classify a website as phishing or legitimate from URL and webpage characteristics. The same held-out test set is used to compare Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine classifiers.

## Dataset description

The dataset is UCI dataset 327, Phishing Websites: <https://archive.ics.uci.edu/dataset/327/phishing>. It contains 11,055 instances, 30 integer input features, and the original labels `-1` and `+1`. In this project, `-1` is mapped to phishing (`1`) and `+1` is mapped to legitimate (`0`). The data is downloaded programmatically with `ucimlrepo` and saved under `data/raw/`.

## GitHub repository

Repository: [2026AJ05257-ML-Assignment](https://github.com/Mikeyzgoat/2026AJ05257-ML-Assignment)

## How to run

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m src.data_validation
.venv/bin/python -m src.eda
.venv/bin/python -m src.train_models
.venv/bin/python -m src.evaluation
.venv/bin/pytest
.venv/bin/streamlit run app.py
```

## Model comparison

The table below will be filled after the models are trained and evaluated on the fixed stratified test set.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Pending | Pending | Pending | Pending | Pending | Pending |
| Decision Tree | Pending | Pending | Pending | Pending | Pending | Pending |
| KNN | Pending | Pending | Pending | Pending | Pending | Pending |
| Naive Bayes | Pending | Pending | Pending | Pending | Pending | Pending |
| SVM | Pending | Pending | Pending | Pending | Pending | Pending |

## Observations

Observations for each model and the overall winner will be added from the measured test-set results after training.

## Streamlit application

The app loads saved models from `model/`, accepts a labeled CSV with all 30 features and `result`, and displays the selected model's six metrics, confusion matrix, and classification report. It does not retrain models or fit preprocessing on uploaded data.

Live app link: To be added after deployment.

## Assignment submission checklist

- GitHub repository link: included above.
- Live Streamlit app link: to be added after deployment.
- BITS Virtual Lab execution screenshot: must be captured by the student.
- Final PDF: must contain the links, genuine screenshot, and this README content.

