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

The models were evaluated on the same 20% stratified test split, created with `random_state=42`.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.928087 | 0.978503 | 0.934392 | 0.901020 | 0.917403 | 0.854206 |
| Decision Tree | 0.969697 | 0.976431 | 0.976017 | 0.955102 | 0.965446 | 0.938643 |
| KNN | 0.947987 | 0.985727 | 0.950052 | 0.931633 | 0.940752 | 0.894544 |
| Naive Bayes | 0.616011 | 0.967709 | 0.535851 | 0.998980 | 0.697542 | 0.406650 |
| SVM | 0.956128 | 0.989042 | 0.963274 | 0.936735 | 0.949819 | 0.911141 |

## Observations

- Logistic Regression gives a balanced baseline, with 0.928087 accuracy and 0.917403 F1-score.
- Decision Tree has the best accuracy, precision, recall, F1-score, and MCC in this comparison.
- KNN performs well overall and has the second-highest AUC at 0.985727.
- Naive Bayes has very high recall but low precision, which leads to many false positives and the lowest overall scores.
- SVM has the best AUC at 0.989042 and strong results across the remaining metrics.

Overall winner: Decision Tree, because it leads on five of the six reported metrics. SVM is the strongest model when ranking by AUC.

## Streamlit application

The app loads saved models from `model/`, accepts a labeled CSV with all 30 features and `result`, and displays the selected model's six metrics, confusion matrix, and classification report. It does not retrain models or fit preprocessing on uploaded data.

Live app link: To be added after deployment.

## Assignment submission checklist

- GitHub repository link: included above.
- Live Streamlit app link: to be added after deployment.
- BITS Virtual Lab execution screenshot: must be captured by the student.
- Final PDF: must contain the links, genuine screenshot, and this README content.
