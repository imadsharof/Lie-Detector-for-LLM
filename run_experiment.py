"""Run the Phi-2 truth-probe generalisation pipeline.

This script reproduces the core experiment from "How well do truth probes
generalise?" on one model only: microsoft/phi-2.

Run from the project root:

    python3 run_experiment.py

Outputs are written to ``results/`` and activations are cached under
``data/activations/``.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lie_detector_llm.datasets import DEFAULT_DATASET_NAMES, build_dataset_collection
from lie_detector_llm.experiment import (
    DEFAULT_MODEL,
    PROBE_METHODS,
    extract_activations_for_collection,
    run_full_transfer_matrix,
    run_layer_method_sweep,
    run_probe_comparison,
    transfer_matrix_summary,
)
from lie_detector_llm.plotting import (
    plot_layer_method_sweep,
    plot_probe_comparison,
    plot_transfer_heatmap,
)


RESULTS_DIR = PROJECT_ROOT / "results"
ACTIVATION_CACHE_DIR = PROJECT_ROOT / "data" / "activations"
RESULTS_DIR.mkdir(exist_ok=True)
ACTIVATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = DEFAULT_MODEL
DATASET_NAMES = DEFAULT_DATASET_NAMES
TRAIN_DATASET = "dbpedia_14"
MAX_GROUPS = 50
LAYER_INDEX = 18
ACTIVATION_BATCH_SIZE = 2
MAX_LENGTH = 512
LOAD_IN_4BIT = False


def banner(text: str) -> None:
    print("\n" + "-" * 78)
    print(text)
    print("-" * 78)


def save_figure(fig: plt.Figure, name: str) -> None:
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def save_csv(frame: pd.DataFrame, name: str) -> None:
    path = RESULTS_DIR / name
    frame.to_csv(path, index=False)
    print(f"[saved] {path}")


def main() -> None:
    sns.set_theme(style="whitegrid")

    banner("Step 1 - Build Phi-2 prompt datasets")
    collection = build_dataset_collection(
        dataset_names=DATASET_NAMES,
        max_groups=MAX_GROUPS,
        seed=0,
    )
    print(collection.summary().to_string(index=False))

    banner("Step 2 - Extract and cache Phi-2 last-token activations")
    activation_cache = extract_activations_for_collection(
        collection=collection,
        model_name=MODEL_NAME,
        activation_batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        show_progress=True,
        activation_cache_dir=ACTIVATION_CACHE_DIR,
    )
    print(f"Activation tensor: {activation_cache.activations.shape}")

    banner(f"Step 3 - Compare the four probes on {TRAIN_DATASET}")
    probe_comparison = run_probe_comparison(
        frame=collection.subset(TRAIN_DATASET),
        model_name=MODEL_NAME,
        layer_index=LAYER_INDEX,
        activation_batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        activation_cache_dir=ACTIVATION_CACHE_DIR,
    )
    save_csv(probe_comparison.results, "phi2_single_dataset_probe_comparison.csv")
    print(
        probe_comparison.results[probe_comparison.results["split"] == "test"][
            ["probe_method", "grouped_accuracy"]
        ].to_string(index=False)
    )

    banner("Step 4 - Accuracy by Phi-2 layer")
    sweep = run_layer_method_sweep(
        collection=collection,
        train_dataset_name=TRAIN_DATASET,
        probe_methods=PROBE_METHODS,
        model_name=MODEL_NAME,
        activation_batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        show_progress=False,
        activation_cache_dir=ACTIVATION_CACHE_DIR,
    )
    save_csv(sweep.results, "phi2_layer_method_sweep.csv")
    fig, _ = plot_layer_method_sweep(
        sweep.results,
        eval_type="out_of_distribution",
        title=f"Phi-2 OOD truth-probe accuracy by layer, train={TRAIN_DATASET}",
    )
    save_figure(fig, "phi2_layer_sweep_ood.png")

    banner("Step 5 - Full transfer matrix for each probe")
    summary_rows = []
    for method in PROBE_METHODS:
        matrix = run_full_transfer_matrix(
            collection=collection,
            model_name=MODEL_NAME,
            probe_method=method,
            layer_index=LAYER_INDEX,
            activation_batch_size=ACTIVATION_BATCH_SIZE,
            max_length=MAX_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
            activation_cache_dir=ACTIVATION_CACHE_DIR,
        )
        save_csv(matrix.results, f"phi2_transfer_matrix_{method}.csv")
        stats = transfer_matrix_summary(matrix.results)
        stats["probe_method"] = method
        summary_rows.append(stats)
        if method == "dim":
            fig, _ = plot_transfer_heatmap(
                matrix.results,
                title=f"Phi-2 transfer matrix, DIM, layer {LAYER_INDEX}",
            )
            save_figure(fig, "phi2_transfer_matrix_dim.png")

    probe_summary = pd.DataFrame(summary_rows).sort_values("transfer_mean", ascending=False)
    save_csv(probe_summary, "phi2_probe_transfer_summary.csv")
    print(probe_summary.round(3).to_string(index=False))
    fig, _ = plot_probe_comparison(
        probe_summary,
        value_column="transfer_mean",
        title=f"Phi-2 mean off-diagonal transfer accuracy, layer {LAYER_INDEX}",
    )
    save_figure(fig, "phi2_probe_transfer_comparison.png")

    banner("Done")


if __name__ == "__main__":
    main()
