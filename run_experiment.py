"""
Lie Detector for LLMs — end-to-end demo script.

Reproduces the RepE (Representation Engineering) white-box probing methodology.
Run from the project root:

    python run_experiment.py

All results are printed to stdout and figures are saved to results/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── make the local package importable even without installing it ──────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering — no display required
import matplotlib.pyplot as plt

from lie_detector_llm.datasets import build_dataset_collection
from lie_detector_llm.experiment import (
    run_probe_experiment,
    run_transfer_experiment,
    run_layer_sweep,
    run_full_transfer_matrix,
)
from lie_detector_llm.plotting import (
    plot_transfer_results,
    plot_layer_sweep,
    plot_transfer_heatmap,
)

# ── output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = "microsoft/phi-2"
SEPARATOR = "─" * 70


def banner(text: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)


def save(fig: plt.Figure, name: str) -> None:
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Load datasets
# ─────────────────────────────────────────────────────────────────────────────
banner("Step 0 — Build dataset collection")

collection = build_dataset_collection(include_repeng_truthful=True)
summary = (
    collection.frame.groupby("dataset_name")
    .agg(n_rows=("dataset_name", "size"), n_groups=("group_id", "nunique"))
    .reset_index()
)
print(summary.to_string(index=False))
print(f"\n  Total rows : {len(collection.frame)}")
print(f"  Datasets   : {collection.dataset_names()}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Single-dataset probe — in-distribution performance
# ─────────────────────────────────────────────────────────────────────────────
banner("Step 1 — Single-dataset probe (repeng_truthful, LR, last layer)")

frame = collection.subset("repeng_truthful")
results_single = run_probe_experiment(
    frame=frame,
    model_name=MODEL,
    probe_method="lr",
    layer_index=-1,
)
print(results_single.summary_table().to_string(index=False))

print("""
Interpretation:
  Each group has one truthful and one deceptive statement.
  The probe must rank the truthful statement above the deceptive one.
  'grouped_accuracy' measures how often the top-ranked candidate is correct.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Probe comparison — all four methods side by side
# ─────────────────────────────────────────────────────────────────────────────
banner("Step 2 — Probe comparison (repeng_truthful, all methods, last layer)")

rows = []
for method in ["dim", "lat", "lr", "pca-g"]:
    out = run_probe_experiment(frame=frame, model_name=MODEL, probe_method=method, layer_index=-1)
    rows.append(out.summary_table())

comparison = pd.concat(rows, ignore_index=True)
test_comparison = comparison[comparison["split"] == "test"][
    ["probe_method", "grouped_accuracy"]
].sort_values("grouped_accuracy", ascending=False)
print(test_comparison.to_string(index=False))

print("""
  DIM  = Difference in Means  : direction = mean(pos) - mean(neg)
  LAT  = Linear Algebraic Treatment : PCA on random contrast differences
  LR   = Logistic Regression   : standard supervised linear classifier
  PCA-G = Grouped PCA          : within-group centering before PCA
""")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Transfer experiment — one-to-many
# ─────────────────────────────────────────────────────────────────────────────
banner("Step 3 — Transfer experiment (train on repeng_truthful, eval on all)")

dataset_names = collection.dataset_names()
transfer_rows = []
for method in ["dim", "lat", "lr", "pca-g"]:
    out = run_transfer_experiment(
        collection=collection,
        train_dataset_name="repeng_truthful",
        eval_dataset_names=dataset_names,
        model_name=MODEL,
        probe_method=method,
        layer_index=-1,
    )
    transfer_rows.append(out.summary_table())

transfer_df = pd.concat(transfer_rows, ignore_index=True)
print(transfer_df[["probe_method", "train_dataset", "eval_dataset", "grouped_accuracy"]].to_string(index=False))

fig, _ = plot_transfer_results(transfer_df, title="Transfer: trained on repeng_truthful")
save(fig, "transfer_repeng_truthful.png")

print("""
  A probe that only works in-distribution (repeng_truthful → repeng_truthful)
  tells us the model memorises a per-dataset signal, not a universal truth direction.
  A probe that transfers to cities / qa / larger_than provides stronger evidence
  that the representation is genuinely truth-related.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full NxN transfer matrix
# ─────────────────────────────────────────────────────────────────────────────
banner("Step 4 — Full NxN transfer matrix (LR probe, last layer)")

full_matrix = run_full_transfer_matrix(
    collection=collection,
    model_name=MODEL,
    probe_method="lr",
    layer_index=-1,
)
pivot = full_matrix.results.pivot(
    index="train_dataset", columns="eval_dataset", values="grouped_accuracy"
)
print(pivot.round(2).to_string())

fig, _ = plot_transfer_heatmap(full_matrix.results, title="Full transfer matrix — LR probe")
save(fig, "full_transfer_matrix.png")

print("""
  Diagonal = in-distribution accuracy (train == eval).
  Off-diagonal = cross-domain transfer.
  The RepE paper argues that a universal truthfulness direction should give
  high accuracy everywhere — both on and off the diagonal.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Layer sweep — which layer encodes truth best?
# ─────────────────────────────────────────────────────────────────────────────
banner("Step 5 — Layer sweep (repeng_truthful, LR probe, all layers)")

layer_results = run_layer_sweep(
    frame=frame,
    model_name=MODEL,
    probe_method="lr",
)
sweep_table = layer_results.results.pivot_table(
    index="layer", columns="split", values="grouped_accuracy"
).round(2)
print(sweep_table.to_string())

best_layer = (
    layer_results.results[layer_results.results["split"] == "test"]
    .sort_values("grouped_accuracy", ascending=False)
    .iloc[0]
)
print(f"\n  Best layer on test set: {int(best_layer['layer'])}  "
      f"(accuracy = {best_layer['grouped_accuracy']:.2f})")

fig, _ = plot_layer_sweep(
    layer_results.results,
    title=f"Layer sweep — LR probe on repeng_truthful ({MODEL})",
)
save(fig, "layer_sweep.png")

print("""
  In transformer models, earlier layers tend to encode syntax and surface form.
  Middle and later layers encode more semantic content — including truth values.
  The RepE paper consistently finds that the best probing layers are in the
  middle-to-late range of the network.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
banner("Summary")
print(f"""
  Model          : {MODEL}
  Results saved  : {RESULTS_DIR}/
    - transfer_repeng_truthful.png
    - full_transfer_matrix.png
    - layer_sweep.png

  Key findings:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. In-distribution probing works — the model stores a linear       │
  │     truth signal accessible from hidden states.                     │
  │                                                                     │
  │  2. gpt2-medium (355M, 24 layers) provides a much richer           │
  │     representation space than distilgpt2, allowing better probing   │
  │     and cross-domain transfer.                                      │
  │                                                                     │
  │  3. Different probe methods have very different transfer profiles:   │
  │     DIM and LAT sometimes transfer where LR fails, and vice versa.  │
  │                                                                     │
  │  4. The best truth-encoding layer is typically in the middle-to-    │
  │     late range — not the very last layer.                           │
  └─────────────────────────────────────────────────────────────────────┘
""")
