from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datasets import DatasetCollection, split_groups
from .models import extract_last_token_activations
from .probes import grouped_accuracy, train_probe

PROBE_METHODS = ["dim", "lat", "lr", "pca-g"]


@dataclass
class ExperimentOutputs:
    results: pd.DataFrame

    def summary_table(self) -> pd.DataFrame:
        return self.results.copy()


def _prepare_layer_slice(activations: np.ndarray, layer_index: int) -> np.ndarray:
    return activations[:, layer_index, :]


def run_probe_experiment(
    frame: pd.DataFrame,
    model_name: str = "microsoft/phi-2",
    probe_method: str = "lr",
    layer_index: int = -1,
    activation_batch_size: int = 2,
    load_in_4bit: bool = False,
    show_progress: bool = False,
) -> ExperimentOutputs:
    splits = split_groups(frame)
    prompts = frame["prompt"].tolist()
    activation_cache = extract_last_token_activations(
        prompts,
        model_name=model_name,
        batch_size=activation_batch_size,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
    )
    layer_activations = _prepare_layer_slice(activation_cache.activations, layer_index)

    indexed = frame.copy()
    indexed["row_index"] = np.arange(len(indexed))

    split_frames = {}
    for split_name, split_frame in splits.items():
        split_frames[split_name] = indexed[
            indexed["group_id"].isin(split_frame["group_id"])
        ].copy()

    train_index = split_frames["train"]["row_index"].to_numpy()
    probe = train_probe(
        probe_method,
        activations=layer_activations[train_index],
        labels=split_frames["train"]["label"].to_numpy(dtype=bool),
        groups=split_frames["train"]["group_id"].to_numpy(),
    )

    records = []
    for split_name, split_frame in split_frames.items():
        row_index = split_frame["row_index"].to_numpy()
        scores = probe.score(layer_activations[row_index])
        accuracy = grouped_accuracy(
            scores=scores,
            labels=split_frame["label"].to_numpy(dtype=bool),
            groups=split_frame["group_id"].to_numpy(),
        )
        records.append(
            {
                "split": split_name,
                "dataset": split_frame["dataset_name"].iloc[0],
                "probe_method": probe_method,
                "model_name": model_name,
                "layer_index": layer_index,
                "grouped_accuracy": accuracy,
            }
        )
    return ExperimentOutputs(results=pd.DataFrame(records))


def run_transfer_experiment(
    collection: DatasetCollection,
    train_dataset_name: str,
    eval_dataset_names: list[str],
    model_name: str = "microsoft/phi-2",
    probe_method: str = "lr",
    layer_index: int = -1,
    split_evaluation: bool = False,
    eval_split: str = "test",
    split_seed: int = 0,
    activation_batch_size: int = 2,
    load_in_4bit: bool = False,
    show_progress: bool = False,
) -> ExperimentOutputs:
    if split_evaluation:
        train_splits = split_groups(collection.subset(train_dataset_name), seed=split_seed)
        if eval_split not in train_splits:
            raise ValueError(f"Unknown eval split: {eval_split}")
        train_frame = train_splits["train"].copy()
        train_frame["transfer_role"] = "train"
        eval_frames = []
        for name in eval_dataset_names:
            splits = split_groups(collection.subset(name), seed=split_seed)
            if eval_split not in splits:
                raise ValueError(f"Unknown eval split: {eval_split}")
            eval_frame = splits[eval_split].copy()
            eval_frame["transfer_role"] = "eval"
            eval_frames.append(eval_frame)
        combined_frame = pd.concat([train_frame, *eval_frames], ignore_index=True)
    else:
        combined_frame = pd.concat(
            [
                collection.subset(train_dataset_name),
                *[
                    collection.subset(name)
                    for name in eval_dataset_names
                    if name != train_dataset_name
                ],
            ],
            ignore_index=True,
        )
        combined_frame["transfer_role"] = "all"

    activation_cache = extract_last_token_activations(
        combined_frame["prompt"].tolist(),
        model_name=model_name,
        batch_size=activation_batch_size,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
    )
    layer_activations = _prepare_layer_slice(activation_cache.activations, layer_index)

    combined_frame = combined_frame.copy()
    combined_frame["row_index"] = np.arange(len(combined_frame))

    if split_evaluation:
        train_group_ids = set(train_frame["group_id"])
    else:
        train_group_ids = set(collection.subset(train_dataset_name)["group_id"])
    train_rows = combined_frame[
        (combined_frame["group_id"].isin(train_group_ids))
        & (combined_frame["transfer_role"].isin(["train", "all"]))
    ]
    train_idx = train_rows["row_index"].to_numpy()

    probe = train_probe(
        probe_method,
        activations=layer_activations[train_idx],
        labels=train_rows["label"].to_numpy(dtype=bool),
        groups=train_rows["group_id"].to_numpy(),
    )

    records = []
    for eval_name in eval_dataset_names:
        eval_rows = combined_frame[
            (combined_frame["dataset_name"] == eval_name)
            & (combined_frame["transfer_role"].isin(["eval", "all"]))
        ]
        row_index = eval_rows["row_index"].to_numpy()
        scores = probe.score(layer_activations[row_index])
        accuracy = grouped_accuracy(
            scores=scores,
            labels=eval_rows["label"].to_numpy(dtype=bool),
            groups=eval_rows["group_id"].to_numpy(),
        )
        records.append(
            {
                "train_dataset": train_dataset_name,
                "eval_dataset": eval_name,
                "probe_method": probe_method,
                "model_name": model_name,
                "layer_index": layer_index,
                "split_evaluation": split_evaluation,
                "eval_split": eval_split if split_evaluation else "all",
                "grouped_accuracy": accuracy,
            }
        )
    return ExperimentOutputs(results=pd.DataFrame(records))


def run_layer_sweep(
    frame: pd.DataFrame,
    model_name: str = "microsoft/phi-2",
    probe_method: str = "lr",
    activation_batch_size: int = 2,
    load_in_4bit: bool = False,
    show_progress: bool = False,
) -> ExperimentOutputs:
    """Train and evaluate a probe at every transformer layer.

    This reveals which layers carry the most linearly accessible truth signal —
    a key analysis from the RepE methodology.
    """
    splits = split_groups(frame)
    activation_cache = extract_last_token_activations(
        frame["prompt"].tolist(),
        model_name=model_name,
        batch_size=activation_batch_size,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
    )
    num_layers = activation_cache.activations.shape[1]

    indexed = frame.copy()
    indexed["row_index"] = np.arange(len(indexed))

    split_frames = {}
    for split_name, split_frame in splits.items():
        split_frames[split_name] = indexed[
            indexed["group_id"].isin(split_frame["group_id"])
        ].copy()

    train_index = split_frames["train"]["row_index"].to_numpy()
    train_labels = split_frames["train"]["label"].to_numpy(dtype=bool)
    train_groups = split_frames["train"]["group_id"].to_numpy()

    records = []
    for layer_idx in range(num_layers):
        layer_activations = activation_cache.activations[:, layer_idx, :]
        probe = train_probe(
            probe_method,
            activations=layer_activations[train_index],
            labels=train_labels,
            groups=train_groups,
        )
        for split_name, split_frame in split_frames.items():
            row_index = split_frame["row_index"].to_numpy()
            scores = probe.score(layer_activations[row_index])
            accuracy = grouped_accuracy(
                scores=scores,
                labels=split_frame["label"].to_numpy(dtype=bool),
                groups=split_frame["group_id"].to_numpy(),
            )
            records.append(
                {
                    "layer": layer_idx,
                    "split": split_name,
                    "dataset": frame["dataset_name"].iloc[0],
                    "probe_method": probe_method,
                    "model_name": model_name,
                    "grouped_accuracy": accuracy,
                }
            )

    return ExperimentOutputs(results=pd.DataFrame(records))


def run_full_transfer_matrix(
    collection: DatasetCollection,
    model_name: str = "microsoft/phi-2",
    probe_method: str = "lr",
    layer_index: int = -1,
    split_evaluation: bool = False,
    eval_split: str = "test",
    split_seed: int = 0,
    activation_batch_size: int = 2,
    load_in_4bit: bool = False,
    show_progress: bool = False,
) -> ExperimentOutputs:
    """Train a probe on each dataset and evaluate on every other dataset.

    This produces the NxN transfer matrix that tests whether a truthfulness
    direction learned on one domain generalises to others — the central claim
    of the RepE paper.
    """
    dataset_names = collection.dataset_names()

    if split_evaluation:
        frames = []
        for dataset_name in dataset_names:
            splits = split_groups(collection.subset(dataset_name), seed=split_seed)
            if eval_split not in splits:
                raise ValueError(f"Unknown eval split: {eval_split}")
            train_frame = splits["train"].copy()
            train_frame["matrix_role"] = "train"
            eval_frame = splits[eval_split].copy()
            eval_frame["matrix_role"] = "eval"
            frames.extend([train_frame, eval_frame])
        all_frame = pd.concat(frames, ignore_index=True)
    else:
        all_frame = collection.frame.copy()
        all_frame["matrix_role"] = "all"

    activation_cache = extract_last_token_activations(
        all_frame["prompt"].tolist(),
        model_name=model_name,
        batch_size=activation_batch_size,
        load_in_4bit=load_in_4bit,
        show_progress=show_progress,
    )
    layer_activations = activation_cache.activations[:, layer_index, :]

    all_frame["row_index"] = np.arange(len(all_frame))

    records = []
    for train_name in dataset_names:
        train_rows = all_frame[
            (all_frame["dataset_name"] == train_name)
            & (all_frame["matrix_role"].isin(["train", "all"]))
        ]
        train_idx = train_rows["row_index"].to_numpy()
        probe = train_probe(
            probe_method,
            activations=layer_activations[train_idx],
            labels=train_rows["label"].to_numpy(dtype=bool),
            groups=train_rows["group_id"].to_numpy(),
        )
        for eval_name in dataset_names:
            eval_rows = all_frame[
                (all_frame["dataset_name"] == eval_name)
                & (all_frame["matrix_role"].isin(["eval", "all"]))
            ]
            row_index = eval_rows["row_index"].to_numpy()
            scores = probe.score(layer_activations[row_index])
            accuracy = grouped_accuracy(
                scores=scores,
                labels=eval_rows["label"].to_numpy(dtype=bool),
                groups=eval_rows["group_id"].to_numpy(),
            )
            records.append(
                {
                    "train_dataset": train_name,
                    "eval_dataset": eval_name,
                    "probe_method": probe_method,
                    "model_name": model_name,
                    "layer_index": layer_index,
                    "split_evaluation": split_evaluation,
                    "eval_split": eval_split if split_evaluation else "all",
                    "grouped_accuracy": accuracy,
                }
            )

    return ExperimentOutputs(results=pd.DataFrame(records))
