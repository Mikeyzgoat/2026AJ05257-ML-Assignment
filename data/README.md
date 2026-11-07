# Data

The raw data is downloaded from the UCI Machine Learning Repository using `ucimlrepo` dataset ID 327, Phishing Websites. The dataset has 11,055 instances, 30 integer input features, and the original target labels `-1` and `+1`.

The download command is:

```bash
python -m src.data_validation
```

The saved dataset copy is `datasets/phishing_dataset.csv`. The target is mapped for modeling as `-1` phishing to `1` and `+1` legitimate to `0`.
