"""End-to-end Phi-2 truth-probe experiments.

The expensive step is activation extraction. Every public runner extracts or
loads activations once, then reuses them across probes, layers and evaluation
datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DatasetCollection, split_groups
from .models import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    ActivationCache,
    extract_or_load_last_token_activations,
)
from .probes import ALL_PROBE_METHODS, grouped_accuracy, train_probe


PROBE_METHODS: list[str] = list(ALL_PROBE_METHODS)


@dataclass
class ExperimentOutputs:
    results: pd.DataFrame

    def summary_table(self) -> pd.DataFrame:
        return self.results.copy()


def _resolve_layer_index(layer_index: int, num_layers: int) -> int:
    resolved = num_layers + layer_index if layer_index < 0 else layer_index
    if not 0 <= resolved < num_layers:
        raise ValueError(f"layer_index {layer_index} is outside 0..{num_layers - 1}.")
    return resolved


def _layer_slice(activations: np.ndarray, layer_index: int) -> tuple[np.ndarray, int]:
    resolved = _resolve_layer_index(layer_index, activations.shape[1])
    return activations[:, resolved, :], resolved


def _extract_for_frame(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    activation_batch_size: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
    load_in_4bit: bool = False,
    show_progress: bool = False,
    activation_cache_dir: str | Path | None = None,
    force_recompute_activations: bool = False,
) -> ActivationCache:
    return extract_or_load_last_token_activations(
        frame["prompt"].tolist(),
        model_name=model_name,
        batch_size=activation_batch_size,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
        cache_dir=activation_cache_dir,
        force_recompute=force_recompute_activations,
    )


def _split_frames(frame: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    indexed = frame.reset_index(drop=True).copy()
    indexed["row_index"] = np.arange(len(indexed))
    splits = split_groups(indexed, seed=seed)
    return {
        name: indexed[indexed["group_id"].isin(part["group_id"])].copy()
        for name, part in splits.items()
    }


def extract_activations_for_collection(
    collection: DatasetCollection,
    model_name: str = DEFAULT_MODEL,
    activation_batch_size: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
    load_in_4bit: bool = False,
    show_progress: bool = True,
    activation_cache_dir: str | Path = "data/activations",
    force_recompute_activations: bool = False,
) -> ActivationCache:
    """Explicit activation extraction/saving step for notebooks."""
    return _extract_for_frame(
        collection.frame.reset_index(drop=True),
        model_name=model_name,
        activation_batch_size=activation_batch_size,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
        activation_cache_dir=activation_cache_dir,
        force_recompute_activations=force_recompute_activations,
    )


def run_probe_experiment(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    probe_method: str = "lr",
    layer_index: int = -1,
    split_seed: int = 0,
    activation_batch_size: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
    load_in_4bit: bool = False,
    show_progress: bool = False,
    activation_cache_dir: str | Path | None = None,
    force_recompute_activations: bool = False,
) -> ExperimentOutputs:
    """Train one probe on one dataset and evaluate train/validation/test."""
    frame = frame.reset_index(drop=True).copy()
    activation_cache = _extract_for_frame(
        frame,
        model_name=model_name,
        activation_batch_size=activation_batch_size,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
        activation_cache_dir=activation_cache_dir,
        force_recompute_activations=force_recompute_activations,
    )
    layer_activations, resolved_layer = _layer_slice(activation_cache.activations, layer_index)
    split_frames = _split_frames(frame, seed=split_seed)

    train = split_frames["train"]
    probe = train_probe(
        probe_method,
        activations=layer_activations[train["row_index"].to_numpy()],
        labels=train["label"].to_numpy(dtype=bool),
        groups=train["group_id"].to_numpy(),
    )

    records = []
    dataset = frame["dataset_name"].iloc[0]
    for split_name, split_frame in split_frames.items():
        row_index = split_frame["row_index"].to_numpy()
        split_groups = split_frame["group_id"].to_numpy()
        scores = probe.score(layer_activations[row_index], groups=split_groups)
        records.append(
            {
                "split": split_name,
                "dataset": dataset,
                "probe_method": probe_method,
                "model_name": model_name,
                "layer": resolved_layer,
                "requested_layer": layer_index,
                "grouped_accuracy": grouped_accuracy(
                    scores,
                    split_frame["label"].to_numpy(dtype=bool),
                    split_groups,
                ),
            }
        )
    return ExperimentOutputs(results=pd.DataFrame(records))


def run_probe_comparison(
    frame: pd.DataFrame,
    probe_methods: list[str] | None = None,
    **kwargs,
) -> ExperimentOutputs:
    """Run all requested probes on the same dataset/layer."""
    rows = []
    for method in probe_methods or PROBE_METHODS:
        rows.append(run_probe_experiment(frame, probe_method=method, **kwargs).summary_table())
    return ExperimentOutputs(results=pd.concat(rows, ignore_index=True))


def _splits_by_dataset(
    frame: pd.DataFrame,
    dataset_names: list[str],
    split_seed: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        name: split_groups(
            frame[frame["dataset_name"] == name].reset_index(drop=True),
            seed=split_seed,
        )
        for name in dataset_names
    }


def run_transfer_experiment(
    collection: DatasetCollection,
    train_dataset_name: str,
    eval_dataset_names: list[str],
    model_name: str = DEFAULT_MODEL,
    probe_method: str = "lr",
    layer_index: int = -1,
    split_seed: int = 0,
    activation_batch_size: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
    load_in_4bit: bool = False,
    show_progress: bool = False,
    activation_cache_dir: str | Path | None = None,
    force_recompute_activations: bool = False,
) -> ExperimentOutputs:
    """Train on one dataset and evaluate on the test split of many datasets."""
    frame = collection.frame.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame))
    activation_cache = _extract_for_frame(
        frame,
        model_name=model_name,
        activation_batch_size=activation_batch_size,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
        activation_cache_dir=activation_cache_dir,
        force_recompute_activations=force_recompute_activations,
    )
    layer_activations, resolved_layer = _layer_slice(activation_cache.activations, layer_index)

    dataset_names = sorted(set([train_dataset_name, *eval_dataset_names]))
    splits = _splits_by_dataset(frame, dataset_names, split_seed)

    train_groups = set(splits[train_dataset_name]["train"]["group_id"])
    train_rows = frame[frame["group_id"].isin(train_groups)]
    probe = train_probe(
        probe_method,
        activations=layer_activations[train_rows["row_index"].to_numpy()],
        labels=train_rows["label"].to_numpy(dtype=bool),
        groups=train_rows["group_id"].to_numpy(),
    )

    records = []
    for eval_name in eval_dataset_names:
        test_groups = set(splits[eval_name]["test"]["group_id"])
        eval_rows = frame[frame["group_id"].isin(test_groups)]
        eval_group_ids = eval_rows["group_id"].to_numpy()
        scores = probe.score(
            layer_activations[eval_rows["row_index"].to_numpy()],
            groups=eval_group_ids,
        )
        records.append(
            {
                "train_dataset": train_dataset_name,
                "eval_dataset": eval_name,
                "probe_method": probe_method,
                "model_name": model_name,
                "layer": resolved_layer,
                "requested_layer": layer_index,
                "is_in_distribution": eval_name == train_dataset_name,
                "grouped_accuracy": grouped_accuracy(
                    scores,
                    eval_rows["label"].to_numpy(dtype=bool),
                    eval_group_ids,
                ),
            }
        )
    return ExperimentOutputs(results=pd.DataFrame(records))


def run_layer_method_sweep(
    collection: DatasetCollection,
    train_dataset_name: str,
    probe_methods: list[str] | None = None,
    eval_dataset_names: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    split_seed: int = 0,
    activation_batch_size: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
    load_in_4bit: bool = False,
    show_progress: bool = False,
    activation_cache_dir: str | Path | None = None,
    force_recompute_activations: bool = False,
) -> ExperimentOutputs:
    """Accuracy-by-layer figure: train on one dataset, test on all datasets."""
    probe_methods = probe_methods or PROBE_METHODS
    eval_dataset_names = eval_dataset_names or collection.dataset_names()

    frame = collection.frame.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame))
    activation_cache = _extract_for_frame(
        frame,
        model_name=model_name,
        activation_batch_size=activation_batch_size,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
        activation_cache_dir=activation_cache_dir,
        force_recompute_activations=force_recompute_activations,
    )
    activations = activation_cache.activations
    num_layers = activations.shape[1]

    dataset_names = sorted(set([train_dataset_name, *eval_dataset_names]))
    splits = _splits_by_dataset(frame, dataset_names, split_seed)
    train_groups = set(splits[train_dataset_name]["train"]["group_id"])
    train_rows = frame[frame["group_id"].isin(train_groups)]
    train_idx = train_rows["row_index"].to_numpy()
    train_labels = train_rows["label"].to_numpy(dtype=bool)
    train_group_ids = train_rows["group_id"].to_numpy()

    eval_rows_by_dataset = {
        name: frame[frame["group_id"].isin(set(splits[name]["test"]["group_id"]))]
        for name in eval_dataset_names
    }

    records = []
    for layer in range(num_layers):
        layer_activations = activations[:, layer, :]
        for method in probe_methods:
            probe = train_probe(
                method,
                activations=layer_activations[train_idx],
                labels=train_labels,
                groups=train_group_ids,
            )
            for eval_name, eval_rows in eval_rows_by_dataset.items():
                row_index = eval_rows["row_index"].to_numpy()
                eval_group_ids = eval_rows["group_id"].to_numpy()
                scores = probe.score(layer_activations[row_index], groups=eval_group_ids)
                records.append(
                    {
                        "layer": layer,
                        "probe_method": method,
                        "train_dataset": train_dataset_name,
                        "eval_dataset": eval_name,
                        "eval_type": (
                            "in_distribution"
                            if eval_name == train_dataset_name
                            else "out_of_distribution"
                        ),
                        "model_name": model_name,
                        "grouped_accuracy": grouped_accuracy(
                            scores,
                            eval_rows["label"].to_numpy(dtype=bool),
                            eval_group_ids,
                        ),
                    }
                )
    return ExperimentOutputs(results=pd.DataFrame(records))


def run_layer_sweep(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    probe_method: str = "lr",
    split_seed: int = 0,
    activation_batch_size: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
    load_in_4bit: bool = False,
    show_progress: bool = False,
    activation_cache_dir: str | Path | None = None,
    force_recompute_activations: bool = False,
) -> ExperimentOutputs:
    """Single-dataset layer sweep for one probe method."""
    frame = frame.reset_index(drop=True).copy()
    activation_cache = _extract_for_frame(
        frame,
        model_name=model_name,
        activation_batch_size=activation_batch_size,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
        activation_cache_dir=activation_cache_dir,
        force_recompute_activations=force_recompute_activations,
    )
    activations = activation_cache.activations
    split_frames = _split_frames(frame, seed=split_seed)
    train = split_frames["train"]
    train_idx = train["row_index"].to_numpy()
    train_labels = train["label"].to_numpy(dtype=bool)
    train_group_ids = train["group_id"].to_numpy()

    records = []
    for layer in range(activations.shape[1]):
        layer_activations = activations[:, layer, :]
        probe = train_probe(
            probe_method,
            activations=layer_activations[train_idx],
            labels=train_labels,
            groups=train_group_ids,
        )
        for split_name, split_frame in split_frames.items():
            row_index = split_frame["row_index"].to_numpy()
            split_groups = split_frame["group_id"].to_numpy()
            scores = probe.score(layer_activations[row_index], groups=split_groups)
            records.append(
                {
                    "layer": layer,
                    "split": split_name,
                    "dataset": split_frame["dataset_name"].iloc[0],
                    "probe_method": probe_method,
                    "model_name": model_name,
                    "grouped_accuracy": grouped_accuracy(
                        scores,
                        split_frame["label"].to_numpy(dtype=bool),
                        split_groups,
                    ),
                }
            )
    return ExperimentOutputs(results=pd.DataFrame(records))


def run_full_transfer_matrix(
    collection: DatasetCollection,
    model_name: str = DEFAULT_MODEL,
    probe_method: str = "lr",
    layer_index: int = -1,
    split_seed: int = 0,
    activation_batch_size: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
    load_in_4bit: bool = False,
    show_progress: bool = False,
    activation_cache_dir: str | Path | None = None,
    force_recompute_activations: bool = False,
) -> ExperimentOutputs:
    """Train on every dataset and evaluate on every dataset."""
    dataset_names = collection.dataset_names()
    frame = collection.frame.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame))
    activation_cache = _extract_for_frame(
        frame,
        model_name=model_name,
        activation_batch_size=activation_batch_size,
        max_length=max_length,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
        activation_cache_dir=activation_cache_dir,
        force_recompute_activations=force_recompute_activations,
    )
    layer_activations, resolved_layer = _layer_slice(activation_cache.activations, layer_index)
    splits = _splits_by_dataset(frame, dataset_names, split_seed)

    probes = {}
    for train_name in dataset_names:
        train_groups = set(splits[train_name]["train"]["group_id"])
        train_rows = frame[frame["group_id"].isin(train_groups)]
        probes[train_name] = train_probe(
            probe_method,
            activations=layer_activations[train_rows["row_index"].to_numpy()],
            labels=train_rows["label"].to_numpy(dtype=bool),
            groups=train_rows["group_id"].to_numpy(),
        )

    records = []
    for train_name in dataset_names:
        for eval_name in dataset_names:
            test_groups = set(splits[eval_name]["test"]["group_id"])
            eval_rows = frame[frame["group_id"].isin(test_groups)]
            eval_group_ids = eval_rows["group_id"].to_numpy()
            scores = probes[train_name].score(
                layer_activations[eval_rows["row_index"].to_numpy()],
                groups=eval_group_ids,
            )
            records.append(
                {
                    "train_dataset": train_name,
                    "eval_dataset": eval_name,
                    "probe_method": probe_method,
                    "model_name": model_name,
                    "layer": resolved_layer,
                    "requested_layer": layer_index,
                    "grouped_accuracy": grouped_accuracy(
                        scores,
                        eval_rows["label"].to_numpy(dtype=bool),
                        eval_group_ids,
                    ),
                }
            )
    return ExperimentOutputs(results=pd.DataFrame(records))


def run_all_probe_transfer_matrices(
    collection: DatasetCollection,
    probe_methods: list[str] | None = None,
    **kwargs,
) -> ExperimentOutputs:
    """Run the full transfer matrix for all four probes."""
    rows = []
    for method in probe_methods or PROBE_METHODS:
        rows.append(
            run_full_transfer_matrix(
                collection=collection,
                probe_method=method,
                **kwargs,
            ).summary_table()
        )
    return ExperimentOutputs(results=pd.concat(rows, ignore_index=True))


def transfer_matrix_summary(results: pd.DataFrame) -> dict[str, float]:
    """Mean diagonal vs off-diagonal grouped accuracy."""
    diagonal = results[results["train_dataset"] == results["eval_dataset"]]
    off_diagonal = results[results["train_dataset"] != results["eval_dataset"]]
    return {
        "in_distribution_mean": float(diagonal["grouped_accuracy"].mean()),
        "transfer_mean": float(off_diagonal["grouped_accuracy"].mean()),
        "transfer_min": float(off_diagonal["grouped_accuracy"].min()),
        "transfer_max": float(off_diagonal["grouped_accuracy"].max()),
    }
