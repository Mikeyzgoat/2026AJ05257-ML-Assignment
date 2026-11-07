# Phishing Website Classification

This project compares five supervised classification models on the UCI Machine Learning Repository Phishing Websites dataset.

## Problem statement

The task is to classify a website as phishing or legitimate from URL and webpage characteristics. The same held-out test set is used to compare Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine classifiers.

## Dataset description

The dataset is UCI dataset 327, Phishing Websites: <https://archive.ics.uci.edu/dataset/327/phishing>. It contains 11,055 instances, 30 integer input features, and the original labels `-1` and `+1`. In this project, `-1` is mapped to phishing (`1`) and `+1` is mapped to legitimate (`0`). The data is downloaded programmatically with `ucimlrepo` and saved under `datasets/`.

## GitHub Repository Link

Repository: [2026AJ05257-ML-Assignment](https://github.com/Mikeyzgoat/2026AJ05257-ML-Assignment)

Dataset CSV: [datasets/phishing_dataset.csv](https://github.com/Mikeyzgoat/2026AJ05257-ML-Assignment/blob/main/datasets/phishing_dataset.csv)

## Live Streamlit App Link

[Open the deployed Streamlit application](https://2026aj05257-ml-assignment-submission.streamlit.app/)

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

## Models used

The five models are Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine.

### Model comparison

The models were evaluated on the same 20% stratified test split, created with `random_state=42`.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.928087 | 0.978503 | 0.934392 | 0.901020 | 0.917403 | 0.854206 |
| Decision Tree | 0.969697 | 0.976431 | 0.976017 | 0.955102 | 0.965446 | 0.938643 |
| KNN | 0.947987 | 0.985727 | 0.950052 | 0.931633 | 0.940752 | 0.894544 |
| Naive Bayes | 0.616011 | 0.967709 | 0.535851 | 0.998980 | 0.697542 | 0.406650 |
| SVM | 0.956128 | 0.989042 | 0.963274 | 0.936735 | 0.949819 | 0.911141 |

## Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Gives a balanced baseline, with 0.928087 accuracy and 0.917403 F1-score. |
| Decision Tree | Has the best accuracy, precision, recall, F1-score, and MCC in this comparison. |
| KNN | Performs well overall and has the second-highest AUC at 0.985727. |
| Naive Bayes | Has very high recall but low precision, causing many false positives and the lowest overall scores. |
| Support Vector Machine | Has the best AUC at 0.989042 and strong results across the other metrics. |

## Overall winner for the dataset

Decision Tree is the overall winner because it leads on five of the six reported metrics. SVM is the strongest model when ranking by AUC.

## Streamlit application

The app has a Home Page and one separate page for each model. The Home Page shows the feature list, class balance, correlation heatmap, and test-set comparison. Each model page shows the one-function training code, fetches UCI dataset 327 with `ucimlrepo`, and displays six metrics, a confusion matrix, a classification report, and a short analysis. It does not retrain models.

Live app link: [2026aj05257-ml-assignment-submission.streamlit.app](https://2026aj05257-ml-assignment-submission.streamlit.app/)

## Assignment submission checklist

- GitHub repository link: included above.
- Live Streamlit app link: included above.
- BITS Virtual Lab execution screenshot: must be captured by the student.
- Final PDF: must contain the links, genuine screenshot, and this README content.
