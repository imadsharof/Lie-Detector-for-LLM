"""Run local multi-model comparisons for the Lie Detector project.

This script mirrors the empirical work from notebooks 2-5, but runs the same
settings across several local Hugging Face models and writes comparison plots.

Run from the project root:

    .venv/bin/python run_local_model_comparison.py

Outputs are written to results/local_model_comparison/.
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lie_detector_llm.datasets import build_dataset_collection
from lie_detector_llm.experiment import (
    run_full_transfer_matrix,
    run_layer_sweep,
    run_probe_experiment,
    run_transfer_experiment,
)
from lie_detector_llm.models import clear_model_cache


OUTPUT_DIR = PROJECT_ROOT / "results" / "local_model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    {"label": "DistilGPT-2", "model_name": "distilgpt2", "batch_size": 4},
    {"label": "GPT-2", "model_name": "gpt2", "batch_size": 4},
    {"label": "GPT-2 Medium", "model_name": "gpt2-medium", "batch_size": 2},
    {"label": "Phi-2", "model_name": "microsoft/phi-2", "batch_size": 2},
]

PROBE_METHODS = ["dim", "lat", "lr", "pca-g"]
TRAIN_DATASET = "repeng_truthful"
LAYER_INDEX = -1


@dataclass
class ModelRun:
    model_label: str
    model_name: str
    runtime_seconds: float


def save_csv(frame: pd.DataFrame, name: str) -> None:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False)
    print(f"[saved] {path}")


def save_figure(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def add_model_columns(frame: pd.DataFrame, label: str, model_name: str) -> pd.DataFrame:
    output = frame.copy()
    output["model_label"] = label
    output["model_name"] = model_name
    return output


def run_for_model(config: dict, collection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelRun]:
    label = config["label"]
    model_name = config["model_name"]
    batch_size = config["batch_size"]
    start = time.perf_counter()

    print("\n" + "=" * 80)
    print(f"Running {label} ({model_name})")
    print("=" * 80)

    single_rows = []
    frame = collection.subset(TRAIN_DATASET)
    for method in PROBE_METHODS:
        print(f"[{label}] single probe: {method}")
        result = run_probe_experiment(
            frame=frame,
            model_name=model_name,
            probe_method=method,
            layer_index=LAYER_INDEX,
            activation_batch_size=batch_size,
            show_progress=True,
        )
        single_rows.append(add_model_columns(result.summary_table(), label, model_name))

    transfer_rows = []
    dataset_names = collection.dataset_names()
    for method in PROBE_METHODS:
        print(f"[{label}] transfer from {TRAIN_DATASET}: {method}")
        result = run_transfer_experiment(
            collection=collection,
            train_dataset_name=TRAIN_DATASET,
            eval_dataset_names=dataset_names,
            model_name=model_name,
            probe_method=method,
            layer_index=LAYER_INDEX,
            split_evaluation=True,
            activation_batch_size=batch_size,
            show_progress=True,
        )
        transfer_rows.append(add_model_columns(result.summary_table(), label, model_name))

    print(f"[{label}] LR layer sweep")
    layer_result = run_layer_sweep(
        frame=frame,
        model_name=model_name,
        probe_method="lr",
        activation_batch_size=batch_size,
        show_progress=True,
    )
    layer_frame = add_model_columns(layer_result.summary_table(), label, model_name)

    print(f"[{label}] LR full transfer matrix")
    matrix_result = run_full_transfer_matrix(
        collection=collection,
        model_name=model_name,
        probe_method="lr",
        layer_index=LAYER_INDEX,
        split_evaluation=True,
        activation_batch_size=batch_size,
        show_progress=True,
    )
    matrix_frame = add_model_columns(matrix_result.summary_table(), label, model_name)

    runtime_seconds = time.perf_counter() - start
    clear_model_cache()
    gc.collect()

    return (
        pd.concat(single_rows, ignore_index=True),
        pd.concat(transfer_rows, ignore_index=True),
        layer_frame,
        matrix_frame,
        ModelRun(
            model_label=label,
            model_name=model_name,
            runtime_seconds=runtime_seconds,
        ),
    )


def plot_single_probe(single: pd.DataFrame) -> None:
    test_rows = single[single["split"] == "test"].copy()
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=test_rows,
        x="model_label",
        y="grouped_accuracy",
        hue="probe_method",
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.set_title("Single-dataset test accuracy on repeng_truthful")
    axis.set_xlabel("Model")
    axis.set_ylabel("Grouped accuracy")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(title="Probe")
    fig.tight_layout()
    save_figure(fig, "single_probe_test_accuracy_by_model.png")


def plot_transfer(transfer: pd.DataFrame) -> None:
    lr_rows = transfer[transfer["probe_method"] == "lr"].copy()
    fig, axis = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=lr_rows,
        x="eval_dataset",
        y="grouped_accuracy",
        hue="model_label",
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.set_title("Transfer from repeng_truthful with LR probe")
    axis.set_xlabel("Evaluation dataset")
    axis.set_ylabel("Grouped accuracy")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Model", ncols=2)
    fig.tight_layout()
    save_figure(fig, "transfer_from_repeng_lr_by_model.png")

    off_domain = transfer[transfer["eval_dataset"] != TRAIN_DATASET].copy()
    summary = (
        off_domain.groupby(["model_label", "probe_method"], as_index=False)
        .agg(mean_transfer_accuracy=("grouped_accuracy", "mean"))
        .sort_values("mean_transfer_accuracy", ascending=False)
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=summary,
        x="model_label",
        y="mean_transfer_accuracy",
        hue="probe_method",
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.set_title("Mean off-domain transfer accuracy")
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean grouped accuracy")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(title="Probe")
    fig.tight_layout()
    save_figure(fig, "mean_off_domain_transfer_by_model.png")


def plot_layers(layer_sweep: pd.DataFrame) -> None:
    test_rows = layer_sweep[layer_sweep["split"] == "test"].copy()
    fig, axis = plt.subplots(figsize=(11, 5))
    sns.lineplot(
        data=test_rows,
        x="layer",
        y="grouped_accuracy",
        hue="model_label",
        marker="o",
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.set_title("LR test accuracy by layer on repeng_truthful")
    axis.set_xlabel("Layer index")
    axis.set_ylabel("Grouped accuracy")
    axis.legend(title="Model")
    fig.tight_layout()
    save_figure(fig, "layer_sweep_lr_test_by_model.png")


def plot_matrices(matrix: pd.DataFrame) -> None:
    models = matrix["model_label"].drop_duplicates().tolist()
    ncols = 2
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, model_label in zip(axes_list, models):
        subset = matrix[matrix["model_label"] == model_label]
        pivot = subset.pivot(
            index="train_dataset",
            columns="eval_dataset",
            values="grouped_accuracy",
        )
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
        axis.set_title(model_label)
        axis.set_xlabel("Eval dataset")
        axis.set_ylabel("Train dataset")

    for axis in axes_list[len(models):]:
        axis.axis("off")

    fig.suptitle("LR full transfer matrices by model", y=1.02, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "full_transfer_matrices_lr_by_model.png")


def build_summary(single: pd.DataFrame, transfer: pd.DataFrame, layer_sweep: pd.DataFrame, matrix: pd.DataFrame, runs: list[ModelRun]) -> pd.DataFrame:
    lr_single = single[
        (single["split"] == "test")
        & (single["probe_method"] == "lr")
    ][["model_label", "grouped_accuracy"]].rename(
        columns={"grouped_accuracy": "repeng_truthful_lr_test_accuracy"}
    )

    lr_transfer = transfer[
        (transfer["probe_method"] == "lr")
        & (transfer["eval_dataset"] != TRAIN_DATASET)
    ]
    lr_transfer_summary = (
        lr_transfer.groupby("model_label", as_index=False)
        .agg(mean_lr_off_domain_transfer=("grouped_accuracy", "mean"))
    )

    best_layers = (
        layer_sweep[layer_sweep["split"] == "test"]
        .sort_values(["model_label", "grouped_accuracy", "layer"], ascending=[True, False, False])
        .groupby("model_label", as_index=False)
        .first()[["model_label", "layer", "grouped_accuracy"]]
        .rename(
            columns={
                "layer": "best_lr_test_layer",
                "grouped_accuracy": "best_layer_test_accuracy",
            }
        )
    )

    matrix_off_diag = matrix[matrix["train_dataset"] != matrix["eval_dataset"]]
    matrix_summary = (
        matrix_off_diag.groupby("model_label", as_index=False)
        .agg(mean_lr_matrix_off_diagonal=("grouped_accuracy", "mean"))
    )

    runtime = pd.DataFrame([run.__dict__ for run in runs])
    summary = runtime.merge(lr_single, on="model_label", how="left")
    summary = summary.merge(lr_transfer_summary, on="model_label", how="left")
    summary = summary.merge(best_layers, on="model_label", how="left")
    summary = summary.merge(matrix_summary, on="model_label", how="left")
    return summary.sort_values("mean_lr_off_domain_transfer", ascending=False)


def main() -> None:
    sns.set_theme(style="whitegrid")
    collection = build_dataset_collection(include_hf_datasets=False)
    print("Datasets:", collection.dataset_names())
    print("Rows:", len(collection.frame), "Groups:", collection.frame["group_id"].nunique())

    all_single = []
    all_transfer = []
    all_layers = []
    all_matrices = []
    runs = []

    for config in MODEL_CONFIGS:
        single, transfer, layer_sweep, matrix, run = run_for_model(config, collection)
        all_single.append(single)
        all_transfer.append(transfer)
        all_layers.append(layer_sweep)
        all_matrices.append(matrix)
        runs.append(run)

    single = pd.concat(all_single, ignore_index=True)
    transfer = pd.concat(all_transfer, ignore_index=True)
    layer_sweep = pd.concat(all_layers, ignore_index=True)
    matrix = pd.concat(all_matrices, ignore_index=True)
    summary = build_summary(single, transfer, layer_sweep, matrix, runs)

    save_csv(single, "single_probe_results.csv")
    save_csv(transfer, "transfer_from_repeng_results.csv")
    save_csv(layer_sweep, "layer_sweep_lr_results.csv")
    save_csv(matrix, "full_transfer_matrix_lr_results.csv")
    save_csv(summary, "local_model_summary.csv")

    plot_single_probe(single)
    plot_transfer(transfer)
    plot_layers(layer_sweep)
    plot_matrices(matrix)

    print("\nSummary:")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
