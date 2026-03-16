from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbeResult:
    scores: np.ndarray


class BaseProbe:
    def score(self, activations: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DotProductProbe(BaseProbe):
    direction: np.ndarray
    center: np.ndarray

    def score(self, activations: np.ndarray) -> np.ndarray:
        centered = activations - self.center
        return centered @ self.direction


@dataclass
class LogisticRegressionProbe(BaseProbe):
    model: LogisticRegression

    def score(self, activations: np.ndarray) -> np.ndarray:
        return self.model.decision_function(activations)


def _orient_direction(direction: np.ndarray, activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = activations @ direction
    positive_mean = scores[labels].mean()
    negative_mean = scores[~labels].mean()
    if positive_mean < negative_mean:
        return -direction
    return direction


def train_dim_probe(activations: np.ndarray, labels: np.ndarray) -> DotProductProbe:
    pos_mean = activations[labels].mean(axis=0)
    neg_mean = activations[~labels].mean(axis=0)
    direction = pos_mean - neg_mean
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("DIM direction has zero norm.")
    direction = direction / norm
    center = (pos_mean + neg_mean) / 2
    return DotProductProbe(direction=direction, center=center)


def train_lat_probe(activations: np.ndarray, labels: np.ndarray, seed: int = 0) -> DotProductProbe:
    random = np.random.default_rng(seed)
    indices = random.permutation(len(activations))
    usable = indices[: len(indices) // 2 * 2]
    first = usable[0::2]
    second = usable[1::2]
    diffs = activations[first] - activations[second]
    center = diffs.mean(axis=0)
    centered = diffs - center
    pca = PCA(n_components=1)
    pca.fit(centered)
    direction = pca.components_[0]
    direction = _orient_direction(direction, activations, labels)
    return DotProductProbe(direction=direction / np.linalg.norm(direction), center=activations.mean(axis=0))


def train_grouped_pca_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> DotProductProbe:
    centered = activations.copy()
    for group in np.unique(groups):
        mask = groups == group
        centered[mask] = centered[mask] - centered[mask].mean(axis=0, keepdims=True)
    pca = PCA(n_components=1)
    pca.fit(centered)
    direction = pca.components_[0]
    direction = _orient_direction(direction, centered, labels)
    return DotProductProbe(direction=direction / np.linalg.norm(direction), center=np.zeros(centered.shape[1]))


def train_logistic_regression_probe(activations: np.ndarray, labels: np.ndarray) -> LogisticRegressionProbe:
    model = LogisticRegression(max_iter=10_000, solver="lbfgs")
    model.fit(activations, labels)
    return LogisticRegressionProbe(model=model)


def train_probe(
    method: str,
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None = None,
) -> BaseProbe:
    if method == "dim":
        return train_dim_probe(activations, labels)
    if method == "lat":
        return train_lat_probe(activations, labels)
    if method == "lr":
        return train_logistic_regression_probe(activations, labels)
    if method == "pca-g":
        if groups is None:
            raise ValueError("Grouped PCA requires group identifiers.")
        return train_grouped_pca_probe(activations, labels, groups)
    raise ValueError(f"Unknown probe method: {method}")


def grouped_accuracy(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    correct = 0
    total = 0
    for group in np.unique(groups):
        mask = groups == group
        group_scores = scores[mask]
        group_labels = labels[mask]
        predicted_index = int(np.argmax(group_scores))
        correct += int(group_labels[predicted_index])
        total += 1
    return correct / total if total else 0.0