"""Matplotlib/seaborn figures for the Phi-2 truth-probe study."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .probes import PROBE_DISPLAY_NAMES


def _with_probe_labels(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "probe_method" in output.columns:
        output["probe"] = output["probe_method"].map(PROBE_DISPLAY_NAMES).fillna(output["probe_method"])
    return output


def plot_layer_method_sweep(
    results: pd.DataFrame,
    eval_type: str = "out_of_distribution",
    title: str | None = None,
):
    """Accuracy vs layer, one line per probe method.

    Pass the ``.results`` DataFrame from ``run_layer_method_sweep``. For
    out-of-distribution rows, accuracy is averaged over all evaluation datasets
    different from the training dataset.
    """
    subset = results[results["eval_type"] == eval_type].copy()
    subset = _with_probe_labels(subset)
    aggregated = (
        subset.groupby(["layer", "probe"], as_index=False)["grouped_accuracy"]
        .mean()
        .sort_values(["probe", "layer"])
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=aggregated,
        x="layer",
        y="grouped_accuracy",
        hue="probe",
        style="probe",
        markers=True,
        dashes=False,
        ax=axis,
    )
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Phi-2 layer")
    axis.set_ylabel("Grouped accuracy")
    label = "out-of-distribution" if eval_type == "out_of_distribution" else "in-distribution"
    axis.set_title(title or f"Truth-probe accuracy by layer ({label})")
    axis.legend(title="Probe", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    return figure, axis


def plot_transfer_heatmap(
    results: pd.DataFrame,
    title: str = "Transfer matrix (grouped accuracy)",
    probe_method: str | None = None,
):
    """Train-dataset by eval-dataset generalisation heatmap."""
    subset = results.copy()
    if probe_method is not None and "probe_method" in subset.columns:
        subset = subset[subset["probe_method"] == probe_method]
    if "probe_method" in subset.columns and subset["probe_method"].nunique() > 1:
        subset = (
            subset.groupby(["train_dataset", "eval_dataset"], as_index=False)
            ["grouped_accuracy"]
            .mean()
        )

    pivot = subset.pivot(
        index="train_dataset",
        columns="eval_dataset",
        values="grouped_accuracy",
    )
    size = max(6, 0.72 * max(len(pivot.index), len(pivot.columns)))
    figure, axis = plt.subplots(figsize=(size + 1.5, size))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Grouped accuracy"},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_ylabel("Train dataset")
    axis.set_xlabel("Eval dataset")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    return figure, axis


def plot_layer_sweep(
    results: pd.DataFrame,
    title: str = "Probe accuracy across Phi-2 layers",
):
    """Single-method layer sweep: one line per split."""
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
    axis.set_xlabel("Phi-2 layer")
    axis.set_ylabel("Grouped accuracy")
    axis.set_title(title)
    axis.legend(title="Split")
    figure.tight_layout()
    return figure, axis


def plot_transfer_results(
    results: pd.DataFrame,
    title: str = "Probe transfer performance",
):
    """Bar plot for one-to-many transfer results."""
    subset = _with_probe_labels(results)
    figure, axis = plt.subplots(figsize=(12, 5))
    hue = "probe" if subset["probe"].nunique() > 1 else None
    sns.barplot(data=subset, x="eval_dataset", y="grouped_accuracy", hue=hue, ax=axis)
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.set_ylabel("Grouped accuracy")
    axis.set_xlabel("Evaluation dataset")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    return figure, axis


def paper_style_transfer_barplot(
    results: pd.DataFrame,
    train_dataset: str = "dbpedia_14",
    title: str | None = None,
    dataset_order: list[str] | None = None,
    probe_order: list[str] | None = None,
):
    """Paper-style transfer bar plot with train-source and train-eval panels.

    The top panel fixes the training dataset, analogous to the paper's
    ``type=train_boolq`` panel. The bottom panel uses each evaluation dataset's
    own training split, analogous to ``type=train_eval``.
    """
    required = {"train_dataset", "eval_dataset", "probe_method", "grouped_accuracy"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    source_rows = results[results["train_dataset"] == train_dataset].copy()
    source_rows["type"] = f"train_{train_dataset}"

    train_eval_rows = results[results["train_dataset"] == results["eval_dataset"]].copy()
    train_eval_rows["type"] = "train_eval"

    plot_data = pd.concat([source_rows, train_eval_rows], ignore_index=True)
    plot_data = _with_probe_labels(plot_data)

    dataset_order = dataset_order or sorted(plot_data["eval_dataset"].unique().tolist())
    probe_order = probe_order or [
        PROBE_DISPLAY_NAMES.get(method, method)
        for method in ["dim", "lat", "lr", "pca-g"]
        if method in set(plot_data["probe_method"])
    ]
    panel_order = [f"train_{train_dataset}", "train_eval"]

    palette = {
        "DIM": "#5B6CF9",
        "LAT": "#FF8C42",
        "LR": "#2ECC71",
        "PCA-G": "#A66CFF",
    }

    sns.set_theme(style="darkgrid", context="talk")
    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(12, 1.25 * len(dataset_order)), 7),
        sharex=True,
        sharey=True,
    )

    for axis, panel in zip(axes, panel_order):
        subset = plot_data[plot_data["type"] == panel]
        sns.barplot(
            data=subset,
            x="eval_dataset",
            y="grouped_accuracy",
            hue="probe",
            order=dataset_order,
            hue_order=probe_order,
            palette=palette,
            ax=axis,
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("accuracy")
        axis.set_xlabel("")
        axis.set_facecolor("#E8EEF7")
        axis.grid(True, axis="y", color="white", linewidth=1.2)
        axis.text(
            1.01,
            0.5,
            f"type={panel}",
            transform=axis.transAxes,
            rotation=-90,
            va="center",
            ha="left",
            fontsize=12,
        )
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    axes[-1].set_xlabel("eval")
    axes[-1].tick_params(axis="x", rotation=55)
    for label in axes[-1].get_xticklabels():
        label.set_horizontalalignment("right")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="algorithm",
        bbox_to_anchor=(1.02, 0.98),
        loc="upper left",
        frameon=False,
    )
    figure.suptitle(title or f"Phi-2 transfer by evaluation dataset, layer 18", y=1.02)
    figure.tight_layout()
    return figure, axes


def plot_probe_comparison(
    summary: pd.DataFrame,
    value_column: str = "transfer_mean",
    title: str = "Mean transfer accuracy by probe method",
):
    """Bar plot comparing transfer statistics for DIM, LAT, LR and PCA-G."""
    ordered = _with_probe_labels(summary).sort_values(value_column, ascending=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ordered, x="probe", y=value_column, ax=axis)
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.set_ylabel(value_column.replace("_", " "))
    axis.set_xlabel("Probe")
    figure.tight_layout()
    return figure, axis
