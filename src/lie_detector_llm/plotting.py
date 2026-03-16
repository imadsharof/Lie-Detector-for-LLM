from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_transfer_results(results: pd.DataFrame, title: str = "Probe transfer performance"):
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.barplot(data=results, x="eval_dataset", y="grouped_accuracy", hue="probe_method", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.set_ylabel("Grouped accuracy")
    axis.set_xlabel("Evaluation dataset")
    figure.tight_layout()
    return figure, axis