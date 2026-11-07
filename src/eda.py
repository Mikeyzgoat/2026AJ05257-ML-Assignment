"""Create basic exploratory plots from the downloaded dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_plots(path="datasets/phishing_dataset.csv"):
    data = pd.read_csv(path)
    Path("assets").mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    data["Result"].value_counts().sort_index().plot(kind="bar", title="Original target distribution")
    plt.xlabel("Target label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("assets/class_distribution.png", dpi=160)
    plt.close()
    plt.figure(figsize=(14, 10))
    sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig("assets/correlation_heatmap.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    create_plots()
