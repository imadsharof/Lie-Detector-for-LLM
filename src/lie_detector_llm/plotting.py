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


def plot_layer_sweep(results: pd.DataFrame, title: str = "Probe accuracy across layers") -> tuple:
    """Line plot showing grouped accuracy at each transformer layer.

    Pass the full ExperimentOutputs.results DataFrame from run_layer_sweep().
    Separate lines are drawn for each split (train / validation / test).
    """
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=results,
        x="layer",
        y="grouped_accuracy",
        hue="split",
        style="split",
        markers=True,
        dashes=False,
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance (0.5)")
    axis.set_title(title)
    axis.set_ylabel("Grouped accuracy")
    axis.set_xlabel("Layer index")
    axis.legend(title="Split")
    figure.tight_layout()
    return figure, axis


def plot_transfer_heatmap(results: pd.DataFrame, title: str = "Transfer matrix (grouped accuracy)") -> tuple:
    """Heatmap of the NxN cross-dataset transfer matrix.

    Pass the full ExperimentOutputs.results DataFrame from run_full_transfer_matrix().
    Rows = train dataset, columns = eval dataset.
    The diagonal shows in-distribution performance.
    Off-diagonal shows transfer.
    """
    pivot = results.pivot(index="train_dataset", columns="eval_dataset", values="grouped_accuracy")
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_ylabel("Train dataset")
    axis.set_xlabel("Eval dataset")
    figure.tight_layout()
    return figure, axis
