from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datasets import DatasetCollection, split_groups
from .models import extract_last_token_activations
from .probes import grouped_accuracy, train_probe


@dataclass
class ExperimentOutputs:
    results: pd.DataFrame

    def summary_table(self) -> pd.DataFrame:
        return self.results.copy()


def _prepare_layer_slice(activations: np.ndarray, layer_index: int) -> np.ndarray:
    return activations[:, layer_index, :]


def run_probe_experiment(
    frame: pd.DataFrame,
    model_name: str = "distilgpt2",
    probe_method: str = "lr",
    layer_index: int = -1,
) -> ExperimentOutputs:
    splits = split_groups(frame)
    prompts = frame["prompt"].tolist()
    activation_cache = extract_last_token_activations(prompts, model_name=model_name)
    layer_activations = _prepare_layer_slice(activation_cache.activations, layer_index)

    indexed = frame.copy()
    indexed["row_index"] = np.arange(len(indexed))

    split_frames = {}
    for split_name, split_frame in splits.items():
        split_frames[split_name] = indexed[indexed["group_id"].isin(split_frame["group_id"])].copy()

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
    model_name: str = "distilgpt2",
    probe_method: str = "lr",
    layer_index: int = -1,
) -> ExperimentOutputs:
    combined_frame = pd.concat(
        [collection.subset(train_dataset_name), *[collection.subset(name) for name in eval_dataset_names if name != train_dataset_name]],
        ignore_index=True,
    )
    activation_cache = extract_last_token_activations(
        combined_frame["prompt"].tolist(),
        model_name=model_name,
    )
    layer_activations = _prepare_layer_slice(activation_cache.activations, layer_index)

    combined_frame = combined_frame.copy()
    combined_frame["row_index"] = np.arange(len(combined_frame))

    train_frame = collection.subset(train_dataset_name).copy()
    train_group_ids = set(train_frame["group_id"])
    train_rows = combined_frame[combined_frame["group_id"].isin(train_group_ids)]
    train_idx = train_rows["row_index"].to_numpy()

    probe = train_probe(
        probe_method,
        activations=layer_activations[train_idx],
        labels=train_rows["label"].to_numpy(dtype=bool),
        groups=train_rows["group_id"].to_numpy(),
    )

    records = []
    for eval_name in eval_dataset_names:
        eval_rows = combined_frame[combined_frame["dataset_name"] == eval_name]
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
                "grouped_accuracy": accuracy,
            }
        )
    return ExperimentOutputs(results=pd.DataFrame(records))